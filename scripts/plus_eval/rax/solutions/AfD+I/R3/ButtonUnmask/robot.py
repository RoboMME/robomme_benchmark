"""Robot I/O helpers for ButtonUnmask: reply decoding, geometry, motion primitives.

All functions take the supplied ctx (Context or rehearsal context). No new Context().
"""
import base64
import io
import json
import math
import time

import numpy as np
from PIL import Image

POS_SCALE = 0.05   # 1.0 action == 0.05 m
ROT_SCALE = 0.5    # 1.0 action == 0.5 rad


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- decoding
class Obs:
    """Decoded observation: meta, rgb/depth arrays per camera, calibration."""

    def __init__(self, meta, images):
        self.meta = meta
        self.step = meta.get("step_id")
        self.state = np.array(meta.get("state", [0] * 8), dtype=float)
        self.rgb = {}
        self.depth = {}
        for att in meta.get("attachments", []):
            blk = images.get(att["content_index"])
            if blk is None:
                continue
            raw = base64.b64decode(blk)
            im = Image.open(io.BytesIO(raw))
            if att["kind"] == "rgb":
                self.rgb[att["camera"]] = np.asarray(im.convert("RGB"))
            else:
                mpu = meta.get("depth", {}).get(att["camera"], {}).get("meters_per_unit", 0.001)
                self.depth[att["camera"]] = np.asarray(im).astype(np.float32) * mpu
        self.K = {c: np.array(v["matrix"], dtype=float) for c, v in meta.get("camera_intrinsics", {}).items()}
        self.pose = {}
        for c, v in meta.get("camera_poses", {}).items():
            self.pose[c] = (np.array(v["position_m"], dtype=float), quat_to_mat(v["quaternion_wxyz"]))
        self.ended = bool(meta.get("episode_ended", False))
        self.complete = False
        tr = meta.get("transition") or {}
        if tr.get("task_complete"):
            self.complete = True
        if tr.get("episode_ended"):
            self.ended = True
        for s in tr.get("steps", []) or []:
            if s.get("task_complete"):
                self.complete = True
            if s.get("episode_ended"):
                self.ended = True
        lim = meta.get("episode_limits") or {}
        self.remaining = lim.get("remaining_episode_steps")

    @property
    def pos(self):
        return self.state[0:3].copy()

    @property
    def rotvec(self):
        return self.state[3:6].copy()

    @property
    def grip(self):
        return float(self.state[6] + self.state[7])

    def points(self, cam):
        """World xyz for every pixel of camera cam (H,W,3); invalid depth -> nan."""
        d = self.depth[cam]
        K = self.K[cam]
        pos, R = self.pose[cam]
        H, W = d.shape
        u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        x = (u - cx) * d / fx
        y = -(v - cy) * d / fy
        z = -d
        cam_pts = np.stack([x, y, z], -1)
        w = cam_pts @ R.T + pos
        w[d <= 0] = np.nan
        return w

    def project(self, cam, xyz):
        """World point -> (u, v) pixel in camera cam (may be outside image)."""
        pos, R = self.pose[cam]
        K = self.K[cam]
        p = R.T @ (np.asarray(xyz, float) - pos)
        if p[2] >= -1e-6:
            return None
        d = -p[2]
        u = K[0, 0] * p[0] / d + K[0, 2]
        v = -K[1, 1] * p[1] / d + K[1, 2]
        return float(u), float(v)


def decode(result):
    content = result.get("content", [])
    meta = {}
    images = {}
    for i, b in enumerate(content):
        if b.get("type") == "text" and not meta:
            try:
                meta = json.loads(b["text"])
            except Exception:
                meta = {"raw": b.get("text")}
        elif b.get("type") == "image":
            images[i] = b.get("data")
    return Obs(meta, images)


# ---------------------------------------------------------------- rotations
def quat_to_mat(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotvec_to_mat(r):
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def mat_to_rotvec(R):
    c = (np.trace(R) - 1) / 2
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: use diagonal
        M = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(M), 0, None))
        i = int(np.argmax(k))
        if k[i] < 1e-9:
            return np.zeros(3)
        k[(i + 1) % 3] = M[i, (i + 1) % 3] / k[i]
        k[(i + 2) % 3] = M[i, (i + 2) % 3] / k[i]
        k /= np.linalg.norm(k)
        return k * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return v * th


def down_rot(yaw):
    """Gripper pointing down (as at reset: rotvec [-pi,0,0]) then rotated by yaw about world z."""
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return Rz @ rotvec_to_mat([-math.pi, 0, 0])


def rot_err(R_target, rotvec_cur):
    """World-frame rotation vector taking current to target."""
    Rc = rotvec_to_mat(rotvec_cur)
    return mat_to_rotvec(R_target @ Rc.T)


def current_yaw(rotvec_cur):
    """Yaw of the gripper relative to the reset orientation."""
    Rc = rotvec_to_mat(rotvec_cur)
    Rrel = Rc @ rotvec_to_mat([-math.pi, 0, 0]).T
    return math.atan2(Rrel[1, 0], Rrel[0, 0])


# ---------------------------------------------------------------- robot calls
class Robot:
    def __init__(self, ctx, on_obs=None):
        self.ctx = ctx
        self.obs = None
        self.on_obs = on_obs      # callback(obs) after every reply (perception accumulation)
        self.ended = False
        self.complete = False
        self.calls = 0

    def observe(self):
        o = decode(self.ctx.call("robomme_observe"))
        self._update(o)
        return o

    def _update(self, o):
        self.obs = o
        self.calls += 1
        if o.ended:
            self.ended = True
        if o.complete:
            self.complete = True
        if self.on_obs is not None:
            try:
                self.on_obs(o)
            except Exception as e:  # perception must never kill control
                log("on_obs error", repr(e))

    def step(self, rows):
        if self.ended:
            return self.obs
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows]
        if self.obs is not None and self.obs.remaining is not None:
            rows = rows[:max(1, int(self.obs.remaining))]
        o = decode(self.ctx.call("robomme_step", actions=rows))
        self._update(o)
        return o

    # -- motion
    def rows_toward(self, target, R_target, grip, n, vmax):
        """n identical rows that together move toward target (clipped to vmax m/step)."""
        o = self.obs
        d = np.asarray(target, float) - o.pos
        per = d / max(1, n)
        nrm = np.linalg.norm(per)
        if nrm > vmax:
            per = per * (vmax / nrm)
        re = rot_err(R_target, o.rotvec) / max(1, n)
        rn = np.linalg.norm(re)
        if rn > 0.25:
            re = re * (0.25 / rn)
        row = list(per / POS_SCALE) + list(re / ROT_SCALE) + [grip]
        return [row] * n

    def move_to(self, target, yaw=0.0, grip=-1.0, vmax=0.05, tol=0.004, max_steps=80, chunk=4,
                rot_tol=0.03, gain=2.0):
        """Closed-loop Cartesian move on measured state; returns (ok, steps_used).

        Measured response (ep1): one row commanding c metres moves ~0.4c (max ~0.02 m/step),
        so far away full-speed chunks are sent, and near the target single rows with gain*remaining.
        """
        R_t = down_rot(yaw)
        used = 0
        target = np.asarray(target, float)
        while used < max_steps and not self.ended:
            o = self.obs
            d = target - o.pos
            dist = float(np.linalg.norm(d))
            re = rot_err(R_t, o.rotvec)
            rerr = float(np.linalg.norm(re))
            if dist < tol and rerr < rot_tol:
                return True, used
            if dist > 0.06:
                v = d / dist * vmax
                n = int(max(1, min(chunk, (dist - 0.04) / (0.4 * vmax))))
            else:
                v = d * gain
                nv = float(np.linalg.norm(v))
                if nv > vmax:
                    v = v * (vmax / nv)
                n = 1
            r = re * 1.5
            rn = float(np.linalg.norm(r))
            if rn > 0.4:
                r = r * (0.4 / rn)
            row = list(v / POS_SCALE) + list(r / ROT_SCALE) + [grip]
            self.step([row] * min(n, max_steps - used))
            used += min(n, max_steps - used)
        o = self.obs
        return float(np.linalg.norm(target - o.pos)) < 3 * tol, used

    def hold(self, n, grip=-1.0, yaw=None):
        o = self.obs
        R_t = down_rot(current_yaw(o.rotvec) if yaw is None else yaw)
        rows = []
        for _ in range(n):
            re = rot_err(R_t, o.rotvec)
            rows.append([0, 0, 0] + list(np.clip(re / ROT_SCALE, -0.3, 0.3)) + [grip])
        return self.step(rows)


def obs_from_frame(frame, root="/workspace"):
    """Build an Obs from a frame-sequence manifest entry (saved PNG files)."""
    meta = dict(frame)
    o = Obs({k: v for k, v in meta.items() if k != "files"}, {})
    for f in frame.get("files", []):
        im = Image.open(f"{root}/{f['path']}")
        if f["kind"] == "rgb":
            o.rgb[f["camera"]] = np.asarray(im.convert("RGB"))
        else:
            mpu = frame.get("depth", {}).get(f["camera"], {}).get("meters_per_unit", 0.001)
            o.depth[f["camera"]] = np.asarray(im).astype(np.float32) * mpu
    return o
