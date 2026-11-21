"""Robot helper library for SwingXtimes (RAX MCP)."""
import base64, io, json, math, time
import numpy as np
from PIL import Image

LOG = []

def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def parse(reply):
    blocks = reply.get("content", [])
    meta = json.loads(blocks[0]["text"])
    imgs = {}
    for att in meta.get("attachments", []) or []:
        b = blocks[att["content_index"]]
        data = base64.b64decode(b["data"])
        im = Image.open(io.BytesIO(data))
        if att["kind"] == "rgb":
            imgs[(att["camera"], "rgb")] = np.asarray(im.convert("RGB"))
        else:
            imgs[(att["camera"], "depth")] = np.asarray(im).astype(np.float64) * 0.001
    return meta, imgs


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(meta, depth, cam):
    """World xyz for every pixel (H,W,3); NaN where invalid."""
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    d = depth.copy(); d[d <= 0] = np.nan
    X = (u - K[0, 2]) * d / K[0, 0]
    Y = -(v - K[1, 2]) * d / K[1, 1]
    Z = -d
    P = np.stack([X, Y, Z], -1) @ R.T + t
    return P


def project(meta, cam, pw):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    pc = R.T @ (np.asarray(pw) - t)
    d = -pc[2]
    u = pc[0] / d * K[0, 0] + K[0, 2]
    v = -pc[1] / d * K[1, 1] + K[1, 2]
    return u, v


class Robot:
    def __init__(self, ctx, outdir=None):
        self.ctx = ctx
        self.meta = None; self.imgs = None
        self.ended = False; self.complete = False
        self.outdir = outdir
        self.nsave = 0

    def observe(self):
        r = self.ctx.call("robomme_observe")
        self.meta, self.imgs = parse(r)
        self._check(self.meta)
        return self.meta

    def _check(self, meta):
        if meta.get("episode_ended"):
            self.ended = True
        tr = meta.get("transition") or {}
        for s in tr.get("steps", []) or []:
            if s.get("episode_ended"):
                self.ended = True
            if s.get("task_complete"):
                self.complete = True

    @property
    def state(self):
        return np.array(self.meta["state"])

    @property
    def step_id(self):
        return self.meta.get("step_id")

    def remaining(self):
        el = self.meta.get("episode_limits") or {}
        return el.get("remaining_episode_steps")

    def step(self, rows):
        if self.ended:
            return self.meta
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        out = None
        for i in range(0, len(rows), 100):
            r = self.ctx.call("robomme_step", actions=rows[i:i + 100])
            self.meta, self.imgs = parse(r)
            self._check(self.meta)
            out = self.meta
            if self.ended:
                break
        return out

    def save(self, tag):
        if not self.outdir:
            return
        import os
        os.makedirs(self.outdir, exist_ok=True)
        self.nsave += 1
        f = self.imgs[("front", "rgb")]; w = self.imgs[("wrist", "rgb")]
        Image.fromarray(np.concatenate([f, w], 1)).save(
            f"{self.outdir}/s{self.step_id:04d}_{tag}.png")

    # ---- motion ----
    def move_to(self, target, grip, tol=0.004, max_steps=80, gain=1.0, vmax=1.0, settle=0):
        """Closed-loop translate TCP to world target (orientation held)."""
        target = np.asarray(target, float)
        n = 0
        while n < max_steps and not self.ended:
            p = self.state[:3]
            err = target - p
            if np.linalg.norm(err) < tol:
                break
            k = max(1, min(10, int(np.linalg.norm(err) / 0.03)))
            d = err / 0.05 * gain
            m = np.max(np.abs(d))
            if m > vmax:
                d = d / m * vmax
            rows = [[d[0], d[1], d[2], 0, 0, 0, grip]]
            # take one step at a time near target, a few when far
            if k > 1:
                step_d = d * min(1.0, 1.0)
                rows = [[*(step_d), 0, 0, 0, grip]] * min(k, 3)
                # recompute each block of 3
            self.step(rows)
            n += len(rows)
        for _ in range(settle):
            self.step([[0, 0, 0, 0, 0, 0, grip]])
        return np.linalg.norm(target - self.state[:3])

    def grip(self, g, n=10):
        self.step([[0, 0, 0, 0, 0, 0, g]] * n)
