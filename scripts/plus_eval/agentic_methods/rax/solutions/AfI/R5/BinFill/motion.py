"""Motion primitives over the official 7-D delta action rows.

Measured response (ep1, 264 five-step windows): each tick the TCP moves ALPHA ~0.37 of the
commanded delta on every axis (x 0.39, y 0.34, z 0.36), negligible dependence on older rows.
Rows inside a chunk are precomputed with that first-order model; every reply re-measures.
"""
import math
import numpy as np
from obs import decode

POS_SCALE = 0.05   # m per unit action
ROT_SCALE = 0.5    # rad per unit action
ALPHA = 0.37       # per-tick tracking fraction (fit ep1)
ALPHA_ROT = 0.37   # assumed same for rotation


def rotvec_to_R(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2; c = max(-1.0, min(1.0, c)); th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(A), 0, None))
        i = int(np.argmax(k))
        if k[i] > 1e-6:
            for j in range(3):
                if j != i:
                    k[j] = A[i, j] / k[i]
        k /= np.linalg.norm(k)
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


DOWN = rotvec_to_R([math.pi, 0, 0])


def down_R(yaw):
    """Gripper pointing down then yawed by `yaw` about world z (fingers close along gripper y)."""
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return Rz @ DOWN


def yaw_of(state):
    R = rotvec_to_R(state[3:6]) @ DOWN.T
    return math.atan2(R[1, 0], R[0, 0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log
        self.meta = None; self.imgs = None
        self.ended = False; self.complete = False
        self.remaining = None; self.step_id = 0
        self.frame_no = 0
        self.on_frame = None   # callback after each reply (perception hooks)

    def _ingest(self, result):
        meta, imgs = decode(result)
        self.meta, self.imgs = meta, imgs
        self.state = np.array(meta["state"], float)
        self.step_id = meta.get("step_id", self.step_id)
        self.frame_no += 1
        lim = meta.get("episode_limits") or {}
        self.remaining = lim.get("remaining_episode_steps", self.remaining)
        tr = meta.get("transition") or {}
        if meta.get("episode_ended") or (isinstance(tr, dict) and tr.get("episode_ended")):
            self.ended = True
        if isinstance(tr, dict):
            for s in tr.get("steps", []) or []:
                if isinstance(s, dict):
                    if s.get("task_complete"):
                        self.complete = True
                    if s.get("episode_ended"):
                        self.ended = True
            if tr.get("task_complete"):
                self.complete = True
        if meta.get("task_complete"):
            self.complete = True
        if self.on_frame is not None and not self.ended:
            try:
                self.on_frame(self)
            except Exception as e:  # perception hook must never break motion
                self.log(f"[on_frame] error {e!r}")
        return meta

    def observe(self):
        return self._ingest(self.ctx.call("robomme_observe"))

    def step(self, rows):
        if self.ended:
            return None
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows]
        if self.remaining is not None:
            rows = rows[:max(0, self.remaining)]
        if not rows:
            self.ended = True
            return None
        out = None
        for i in range(0, len(rows), 100):
            out = self._ingest(self.ctx.call("robomme_step", actions=rows[i:i + 100]))
            if self.ended:
                break
        return out

    def pos(self):
        return self.state[:3].copy()

    def grip_open(self):
        return float(self.state[6] + self.state[7])

    def plan_rows(self, goal_fn, yaw, grip, k, gain=0.6, vmax=1.0, ff=None):
        """Precompute k rows with the first-order model.  goal_fn(i) -> xyz goal for tick i+1."""
        p = self.pos().copy()
        Rc = rotvec_to_R(self.state[3:6])
        rows = []
        for i in range(k):
            g = np.asarray(goal_fn(i + 1), float)
            e = g - p
            d = gain * e / ALPHA
            if ff is not None:
                d = d + np.asarray(ff, float) / ALPHA
            d = d / POS_SCALE
            m = np.max(np.abs(d))
            if m > vmax:
                d = d * (vmax / m)
            if yaw is not None:
                rv = R_to_rotvec(down_R(yaw) @ Rc.T)
                rr = 0.6 * rv / ALPHA_ROT / ROT_SCALE
                mr = np.max(np.abs(rr))
                if mr > 0.6:
                    rr = rr * (0.6 / mr)
                Rc = rotvec_to_R(rr * ROT_SCALE * ALPHA_ROT) @ Rc
            else:
                rr = np.zeros(3)
            p = p + d * POS_SCALE * ALPHA
            rows.append(list(d) + list(rr) + [grip])
        return rows

    def yaw_err(self, yaw):
        if yaw is None:
            return 0.0
        return float(np.linalg.norm(R_to_rotvec(down_R(yaw) @ rotvec_to_R(self.state[3:6]).T)))

    def move_to(self, target, yaw=None, grip=-1.0, tol=0.004, max_steps=80, vmax=1.0, chunk=None):
        """Closed-loop move to a fixed target; chunk size shrinks near the goal."""
        target = np.asarray(target, float)
        used = 0
        while used < max_steps and not self.ended:
            dist = float(np.linalg.norm(target - self.pos()))
            if dist < tol and self.yaw_err(yaw) < 0.04:
                return True
            k = chunk or (8 if dist > 0.10 else 5 if dist > 0.03 else 3)
            rows = self.plan_rows(lambda i: target, yaw, grip, k, vmax=vmax)
            self.step(rows); used += k
        return float(np.linalg.norm(target - self.pos())) < 3 * tol

    def hold(self, n, grip):
        return self.step([[0, 0, 0, 0, 0, 0, grip]] * n)
