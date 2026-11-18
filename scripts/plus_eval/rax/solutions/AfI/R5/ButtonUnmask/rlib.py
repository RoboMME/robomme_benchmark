"""Helper library: decode RAX packets, geometry, closed-loop motion.

Used both in development scripts and the frozen controller. Only uses ctx passed in.
"""
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

WS = Path("/workspace")
POS_SCALE = 0.05
ROT_SCALE = 0.5


def log(*a):
    print("[%.1f]" % time.time(), *a, flush=True)


class Obs:
    def __init__(self, reply):
        content = reply["content"]
        self.meta = json.loads(content[0]["text"])
        m = self.meta
        self.step_id = m.get("step_id")
        self.state = np.array(m.get("state", [0] * 8), dtype=float)
        self.ended = bool(m.get("episode_ended", False))
        self.task_complete = False
        tr = m.get("transition") or {}
        if tr.get("task_complete") or m.get("task_complete"):
            self.task_complete = True
        if tr.get("episode_ended"):
            self.ended = True
        for s in tr.get("steps", []) or []:
            if s.get("task_complete"):
                self.task_complete = True
            if s.get("episode_ended"):
                self.ended = True
        lim = m.get("episode_limits") or {}
        self.remaining = lim.get("remaining_episode_steps")
        self.limit = lim.get("episode_step_limit")
        self.rgb, self.depth = {}, {}
        for att in m.get("attachments", []) or []:
            blk = content[att["content_index"]]
            if blk.get("type") != "image":
                continue
            img = Image.open(io.BytesIO(base64.b64decode(blk["data"])))
            arr = np.array(img)
            if att["kind"] == "rgb":
                self.rgb[att["camera"]] = arr[..., :3] if arr.ndim == 3 else arr
            elif att["kind"] == "depth":
                mpu = (m.get("depth", {}).get(att["camera"], {}) or {}).get("meters_per_unit", 0.001)
                self.depth[att["camera"]] = arr.astype(np.float64) * mpu
        self.K = {c: np.array(v["matrix"], float) for c, v in (m.get("camera_intrinsics") or {}).items()}
        self.pose = {}
        for c, v in (m.get("camera_poses") or {}).items():
            self.pose[c] = (np.array(v["position_m"], float), quat_to_R(v["quaternion_wxyz"]))

    @property
    def pos(self):
        return self.state[0:3]

    @property
    def grip(self):
        return float(self.state[6] + self.state[7])

    def deproject(self, cam, u, v, r=1):
        d = self.depth[cam]
        ui, vi = int(round(u)), int(round(v))
        win = d[max(0, vi - r):vi + r + 1, max(0, ui - r):ui + r + 1]
        win = win[win > 0]
        if win.size == 0:
            return None
        z = float(np.median(win))
        K = self.K[cam]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pc = np.array([(u - cx) * z / fx, -(v - cy) * z / fy, -z])
        p, R = self.pose[cam]
        return R @ pc + p

    def cloud(self, cam):
        """World points for every pixel (H,W,3); invalid -> nan."""
        d = self.depth[cam]
        H, W = d.shape
        K = self.K[cam]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        pc = np.stack([(uu - cx) * d / fx, -(vv - cy) * d / fy, -d], -1)
        p, R = self.pose[cam]
        w = pc @ R.T + p
        w[d <= 0] = np.nan
        return w

    def project(self, cam, xyz):
        p, R = self.pose[cam]
        pc = R.T @ (np.asarray(xyz, float) - p)
        z = -pc[2]
        K = self.K[cam]
        u = K[0, 0] * pc[0] / z + K[0, 2]
        v = -K[1, 1] * pc[1] / z + K[1, 2]
        return u, v

    def save(self, prefix):
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        for c, a in self.rgb.items():
            Image.fromarray(a).save(str(prefix) + f"_{c}.png")


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def observe(ctx):
    return Obs(ctx.call("robomme_observe"))


def step(ctx, rows):
    rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
    return Obs(ctx.call("robomme_step", actions=rows))


def plan_rows(delta, grip, max_frac=1.0):
    """Rows moving by delta (m) with per-step translation <= max_frac*0.05 m."""
    delta = np.asarray(delta, float)
    n = int(np.ceil(np.max(np.abs(delta)) / (POS_SCALE * max_frac) + 1e-9))
    n = max(n, 1)
    per = delta / n / POS_SCALE
    return [[per[0], per[1], per[2], 0, 0, 0, grip] for _ in range(n)]


def move_to(ctx, obs, target, grip, tol=0.004, max_iters=6, max_frac=1.0, settle=0, max_rows=100):
    """Closed-loop: plan rows to target from measured pos, execute, re-measure."""
    target = np.asarray(target, float)
    for it in range(max_iters):
        err = target - obs.pos
        if np.linalg.norm(err) < tol:
            break
        rows = plan_rows(err, grip, max_frac)[:max_rows]
        rows += [[0, 0, 0, 0, 0, 0, grip]] * settle
        obs = step(ctx, rows)
        if obs.ended:
            break
    return obs


def hold(ctx, grip, n):
    return step(ctx, [[0, 0, 0, 0, 0, 0, grip]] * n)


class FrameObs(Obs):
    """Obs built from a saved frame record (manifest 'frames' entry)."""

    def __init__(self, frame, root=WS):
        self.meta = frame
        self.step_id = frame.get("step_id")
        self.state = np.array(frame.get("state", [0] * 8), dtype=float)
        self.ended, self.task_complete = False, False
        self.remaining = self.limit = None
        self.rgb, self.depth = {}, {}
        for f in frame.get("files", []):
            arr = np.array(Image.open(Path(root) / f["path"]))
            if f["kind"] == "rgb":
                self.rgb[f["camera"]] = arr[..., :3]
            elif f["kind"] == "depth":
                mpu = (frame.get("depth", {}).get(f["camera"], {}) or {}).get("meters_per_unit", 0.001)
                self.depth[f["camera"]] = arr.astype(np.float64) * mpu
        self.K = {c: np.array(v["matrix"], float) for c, v in (frame.get("camera_intrinsics") or {}).items()}
        self.pose = {c: (np.array(v["position_m"], float), quat_to_R(v["quaternion_wxyz"]))
                     for c, v in (frame.get("camera_poses") or {}).items()}


def load_run_frames(run_dir):
    import glob
    out = {}
    for m in sorted(glob.glob(str(Path(run_dir) / "frames" / "*" / "manifest.json"))):
        for fr in json.load(open(m))["frames"]:
            out[fr["step_id"]] = fr
    return [out[k] for k in sorted(out)]
