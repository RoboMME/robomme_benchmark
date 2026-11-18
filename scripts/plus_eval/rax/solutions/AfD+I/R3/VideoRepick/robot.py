"""MCP packet decoding, geometry and motion primitives (RAX RoboMME-Plus)."""
import base64, io, json, time
from pathlib import Path
import numpy as np
from PIL import Image

WS = Path("/workspace")
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def decode(result):
    """Return (meta dict, {(camera,kind): ndarray}) from an MCP tool result."""
    content = result["content"] if isinstance(result, dict) else result
    meta = json.loads(content[0]["text"])
    imgs = {}
    for att in meta.get("attachments", []) or []:
        blk = content[att["content_index"]]
        data = base64.b64decode(blk["data"])
        im = Image.open(io.BytesIO(data))
        arr = np.array(im)
        if att["kind"] == "rgb":
            arr = np.array(im.convert("RGB"))
        imgs[(att["camera"], att["kind"])] = arr
    return meta, imgs


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def deproject(meta, depth, cam, u, v, win=1):
    """pixel (u=col, v=row) -> world xyz using OpenGL camera convention; median depth in window."""
    K = np.array(meta["camera_intrinsics"][cam]["matrix"], float)
    pose = meta["camera_poses"][cam]
    mpu = meta.get("depth", {}).get(cam, {}).get("meters_per_unit", 0.001)
    ui, vi = int(round(u)), int(round(v))
    patch = depth[max(0, vi - win):vi + win + 1, max(0, ui - win):ui + win + 1].astype(float)
    vals = patch[patch > 0]
    if len(vals) == 0:
        return None
    d = float(np.min(vals)) * mpu  # nearest surface (object top) in window
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.array([(u - cx) * d / fx, -(v - cy) * d / fy, -d])
    R = quat_to_R(pose["quaternion_wxyz"])
    return R @ pc + np.array(pose["position_m"], float)


def project(meta, cam, p):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"], float)
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"])
    pc = R.T @ (np.asarray(p, float) - np.array(pose["position_m"], float))
    d = -pc[2]
    u = K[0, 0] * pc[0] / d + K[0, 2]
    v = -K[1, 1] * pc[1] / d + K[1, 2]
    return u, v


def world_map(meta, depth, cam):
    """Full HxWx3 world-coordinate map for a camera depth image (nan where invalid)."""
    K = np.array(meta["camera_intrinsics"][cam]["matrix"], float)
    pose = meta["camera_poses"][cam]
    mpu = meta.get("depth", {}).get(cam, {}).get("meters_per_unit", 0.001)
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W].astype(float)
    d = depth.astype(float) * mpu
    pc = np.stack([(uu - K[0, 2]) * d / K[0, 0], -(vv - K[1, 2]) * d / K[1, 1], -d], -1)
    R = quat_to_R(pose["quaternion_wxyz"])
    P = pc @ R.T + np.array(pose["position_m"], float)
    P[depth == 0] = np.nan
    return P


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = None
        self.imgs = None
        self.ended = False
        self.complete = False
        self.grip = -1.0

    def observe(self):
        self.meta, self.imgs = decode(self.ctx.call("robomme_observe"))
        return self.meta

    @property
    def state(self):
        return np.array(self.meta["state"], float)

    @property
    def remaining(self):
        lim = self.meta.get("episode_limits") or {}
        return lim.get("remaining_episode_steps", 10 ** 6)

    def step(self, rows):
        if self.ended:
            return self.meta
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        rem = self.remaining
        if rem is not None and rem <= 0:
            self.ended = True
            return self.meta
        rows = rows[:max(1, min(100, rem))]
        res = self.ctx.call("robomme_step", actions=rows)
        self.meta, self.imgs = decode(res)
        if self.meta.get("episode_ended"):
            self.ended = True
            tr = self.meta.get("transition") or {}
            for st in (tr.get("steps") or []) + (self.meta.get("steps") or []):
                if st.get("task_complete"):
                    self.complete = True
            if self.meta.get("task_complete") or tr.get("task_complete"):
                self.complete = True
            log("[robot] episode ended; task_complete=", self.complete, "step", self.meta.get("step_id"))
        return self.meta

    def move_to(self, target, grip=None, tol=0.006, max_steps=120, speed=1.0, chunk=8):
        """Closed-loop Cartesian move of TCP to target (world m), orientation held."""
        if grip is not None:
            self.grip = grip
        target = np.asarray(target, float)
        used = 0
        while used < max_steps and not self.ended:
            pos = self.state[:3]
            err = target - pos
            dist = np.linalg.norm(err)
            if dist < tol:
                break
            n = int(min(chunk, max(1, np.ceil(dist / (0.05 * speed) * 1.0))))
            per = err / n
            per_cmd = np.clip(per / 0.05, -speed, speed)
            rows = [list(per_cmd) + [0, 0, 0, self.grip] for _ in range(n)]
            self.step(rows)
            used += n
        return np.linalg.norm(target - self.state[:3])

    def gripper(self, g, n=8):
        self.grip = g
        self.step([[0, 0, 0, 0, 0, 0, g]] * n)

    def opening(self):
        s = self.state
        return float(s[6] + s[7])
