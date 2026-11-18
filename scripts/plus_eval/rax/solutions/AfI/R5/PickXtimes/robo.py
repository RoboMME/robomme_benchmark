"""Generic RoboMME-Plus helpers: packet decoding, RGB-D back-projection, motion."""
import base64
import io
import json
import time

import numpy as np
from PIL import Image

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


class Obs:
    def __init__(self, result):
        blocks = result.get("content", [])
        self.meta = json.loads(blocks[0]["text"])
        m = self.meta
        self.step_id = m.get("step_id")
        self.state = np.array(m.get("state", [0] * 8), dtype=float)
        self.instruction = m.get("instruction", "")
        self.ended = bool(m.get("episode_ended", False))
        self.limits = m.get("episode_limits", {}) or {}
        self.rgb, self.depth = {}, {}
        for att in m.get("attachments", []):
            idx = att.get("content_index")
            if idx is None or idx >= len(blocks):
                continue
            blk = blocks[idx]
            data = blk.get("data")
            if data is None:
                continue
            img = Image.open(io.BytesIO(base64.b64decode(data)))
            cam, kind = att.get("camera"), att.get("kind")
            if kind == "rgb":
                self.rgb[cam] = np.array(img.convert("RGB"))
            elif kind == "depth":
                scale = (m.get("depth", {}).get(cam, {}) or {}).get("meters_per_unit", 0.001)
                self.depth[cam] = np.array(img).astype(np.float64) * scale
        self.K = {c: np.array(v["matrix"], dtype=float) for c, v in (m.get("camera_intrinsics") or {}).items()}
        self.poses = m.get("camera_poses") or {}
        tr = m.get("transition") or {}
        self.task_complete = bool(tr.get("task_complete")) or bool(m.get("task_complete")) or \
            any(s.get("task_complete") for s in tr.get("steps", []) or [])
        if tr.get("episode_ended") or any(s.get("episode_ended") for s in tr.get("steps", []) or []):
            self.ended = True

    @property
    def remaining(self):
        return self.limits.get("remaining_episode_steps")

    def world_points(self, cam):
        """HxWx3 world points for every pixel (NaN where depth invalid)."""
        d = self.depth[cam]
        K = self.K[cam]
        H, W = d.shape
        u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        x = (u - cx) * d / fx
        y = -(v - cy) * d / fy
        z = -d
        pc = np.stack([x, y, z], -1)
        q = self.poses[cam]["quaternion_wxyz"]
        R = quat_to_R(q)
        p = np.array(self.poses[cam]["position_m"], dtype=float)
        pw = pc @ R.T + p
        pw[d <= 0] = np.nan
        return pw

    def project(self, cam, pw):
        """world point -> (u, v) pixel in cam."""
        R = quat_to_R(self.poses[cam]["quaternion_wxyz"])
        p = np.array(self.poses[cam]["position_m"], dtype=float)
        pc = R.T @ (np.asarray(pw, dtype=float) - p)
        K = self.K[cam]
        d = -pc[2]
        u = pc[0] * K[0, 0] / d + K[0, 2]
        v = -pc[1] * K[1, 1] / d + K[1, 2]
        return u, v, d


def quat_to_R(q):
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def axis_angle_to_R(r):
    r = np.asarray(r, dtype=float)
    th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def R_to_axis_angle(R):
    c = (np.trace(R) - 1) / 2
    c = np.clip(c, -1, 1)
    th = np.arccos(c)
    if th < 1e-6:
        return np.zeros(3)
    if np.pi - th < 1e-3:
        # near pi
        M = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / np.sqrt(max(M[i, i], 1e-12))
        return k * th
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return k * th


BASE_XY = np.array([-0.615, 0.0])
R_MAX = 0.70       # hard clamp on horizontal reach from the base (IK failures cost 100 steps)


def clamp_ws(t):
    t = np.array(t, dtype=float)
    d = t[:2] - BASE_XY
    n = np.linalg.norm(d)
    if n > R_MAX:
        t[:2] = BASE_XY + d / n * R_MAX
    t[1] = np.clip(t[1], -0.40, 0.40)
    t[0] = max(t[0], -0.42)
    t[2] = np.clip(t[2], 0.012, 0.35)
    return t


def reach(xy):
    return float(np.linalg.norm(np.asarray(xy[:2], dtype=float) - BASE_XY))


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.obs = None
        self.ended = False
        self.success = False
        self.t0 = time.time()
        self.R_des = None
        self.penalties = 0

    def observe(self):
        self.obs = Obs(self.ctx.call("robomme_observe"))
        self._check(self.obs)
        return self.obs

    def _check(self, o):
        if o.ended:
            self.ended = True
        if o.task_complete:
            self.success = True

    def step(self, rows):
        if self.ended:
            return self.obs
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        rem = self.obs.remaining if self.obs is not None else None
        if rem is not None:
            rows = rows[:max(0, int(rem))]
        if not rows:
            self.ended = True
            return self.obs
        prev = self.obs.step_id if self.obs is not None else None
        o = Obs(self.ctx.call("robomme_step", actions=rows))
        if prev is not None and o.step_id is not None and o.step_id - prev > len(rows) + 2:
            self.penalties += 1
            log("PENALTY: step jumped %s -> %s for %d rows at %s" % (prev, o.step_id, len(rows), np.round(o.state[:3], 3)))
        self.obs = o
        self._check(o)
        return o

    def pos(self):
        return self.obs.state[:3].copy()

    def grip_open(self):
        return float(self.obs.state[6] + self.obs.state[7])

    ALPHA = 0.4   # measured: achieved displacement per step ~= 0.4 * commanded delta
    GAIN = 0.8

    def plan_rows(self, targets, grip, p=None, max_delta=0.05):
        """Rows driving the EE through a list of per-step world targets using the response model."""
        p = self.pos() if p is None else np.asarray(p, dtype=float).copy()
        rows = []
        rot = np.zeros(3)
        if self.R_des is not None:
            Rc = axis_angle_to_R(self.obs.state[3:6])
            rv = R_to_axis_angle(self.R_des @ Rc.T)
            if np.linalg.norm(rv) > 0.01:
                rot = np.clip(rv * 0.6 / 0.5, -0.3, 0.3)
        for k, tgt in enumerate(targets):
            e = clamp_ws(tgt) - p
            cmd = e / self.ALPHA * self.GAIN
            cmd = np.clip(cmd, -max_delta, max_delta)
            rr = rot if k == 0 else np.zeros(3)
            rows.append([cmd[0] / 0.05, cmd[1] / 0.05, cmd[2] / 0.05, rr[0], rr[1], rr[2], grip])
            p = p + self.ALPHA * cmd
        return rows

    def move_to(self, target, grip, tol=0.004, max_steps=80, speed=1.0, chunk=6, rot_target=None):
        """Closed-loop Cartesian move; grip -1 open, +1 closed. Returns final error."""
        target = np.asarray(target, dtype=float)
        used = 0
        pen0 = self.penalties
        err = float(np.linalg.norm(target - self.pos()))
        while used < max_steps and not self.ended and self.penalties == pen0:
            err = float(np.linalg.norm(target - self.pos()))
            if err < tol:
                break
            n = max(1, min(chunk, max_steps - used))
            if reach(self.pos()) > R_MAX - 0.04 or reach(target) > R_MAX - 0.04:
                n = min(n, 2)  # near the reach limit: short chunks so an IK failure costs less
            rows = self.plan_rows([target] * n, grip, max_delta=0.05 * speed)
            self.step(rows)
            used += n
        return float(np.linalg.norm(target - self.pos()))

    def hold(self, grip, n):
        return self.step([[0, 0, 0, 0, 0, 0, grip]] * n)
