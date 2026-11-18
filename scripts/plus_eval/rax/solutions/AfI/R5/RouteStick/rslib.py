"""Packet decoding, geometry and motion helpers for RouteStick (RAX MCP)."""
import base64, io, json, math, time
from pathlib import Path
import numpy as np
from PIL import Image

WS = Path("/workspace")

def decode(result):
    blocks = result.get("content", [])
    meta = json.loads(blocks[0]["text"])
    imgs = {}
    for a in meta.get("attachments", []) or []:
        b = blocks[a["content_index"]]
        raw = base64.b64decode(b["data"])
        im = Image.open(io.BytesIO(raw))
        if a["kind"] == "rgb":
            imgs[(a["camera"], "rgb")] = np.array(im.convert("RGB"))
        else:
            mpu = meta.get("depth", {}).get(a["camera"], {}).get("meters_per_unit", 0.001)
            imgs[(a["camera"], "depth")] = np.array(im).astype(np.float64) * mpu
    return meta, imgs

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def deproject(meta, cam, depth):
    """Return HxWx3 world points (nan where invalid)."""
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    d = depth.copy(); d[d <= 0] = np.nan
    pc = np.stack([(u - K[0, 2]) * d / K[0, 0], -(v - K[1, 2]) * d / K[1, 1], -d], -1)
    return pc @ R.T + t

def project(meta, cam, pts):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    pc = (np.atleast_2d(pts) - t) @ R
    d = -pc[:, 2]
    u = K[0, 0] * pc[:, 0] / d + K[0, 2]
    v = -K[1, 1] * pc[:, 1] / d + K[1, 2]
    return np.stack([u, v], -1)

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx, self.log = ctx, log
        self.meta, self.imgs = None, None
        self.ended = False; self.complete = False
        self.limit = None; self.remaining = None

    def _absorb(self, meta):
        tr = meta.get("transition") or {}
        for s in tr.get("steps", []) or []:
            if s.get("task_complete"): self.complete = True
            if s.get("episode_ended"): self.ended = True
        if meta.get("episode_ended"): self.ended = True
        if meta.get("task_complete"): self.complete = True
        lim = meta.get("episode_limits") or {}
        if lim:
            self.limit = lim.get("episode_step_limit", self.limit)
            self.remaining = lim.get("remaining_episode_steps", self.remaining)

    def observe(self):
        meta, imgs = decode(self.ctx.call("robomme_observe"))
        self.meta, self.imgs = meta, imgs; self._absorb(meta)
        return meta, imgs

    def step(self, rows):
        if self.ended: return None
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        meta, imgs = decode(self.ctx.call("robomme_step", actions=rows))
        self.meta = meta
        if imgs: self.imgs = imgs
        self._absorb(meta)
        return meta

    @property
    def pos(self):
        return np.array(self.meta["state"][:3])

    def move_to(self, target, max_steps=60, tol=0.004, vmax=0.6, alpha=0.4, **_):
        """Single-row closed-loop translation to target (world m); per-tick gain compensation."""
        target = np.asarray(target, float)
        used = 0
        while used < max_steps and not self.ended:
            err = target - self.pos
            if np.linalg.norm(err) < tol: break
            a = err / alpha / 0.05
            m = np.max(np.abs(a))
            if m > vmax: a = a * vmax / m
            self.step([list(a) + [0, 0, 0, 1.0]]); used += 1
        return float(np.linalg.norm(target - self.pos)), used
