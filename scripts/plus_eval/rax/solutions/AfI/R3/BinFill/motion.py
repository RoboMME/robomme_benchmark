"""Motion helpers over robomme_step (relative world-frame delta rows)."""
import json
import numpy as np
from perception import decode_obs

POS_SCALE = 0.05   # m per unit action
ROT_SCALE = 0.5    # rad per unit action
DOWN = np.array([-np.pi, 0.0, 0.0])


def rotvec_to_R(rv):
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2
    c = np.clip(c, -1, 1)
    th = np.arccos(c)
    if th < 1e-6:
        return np.zeros(3)
    if np.pi - th < 1e-3:
        # near pi
        M = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(M), 0, None))
        i = np.argmax(k)
        k[i] = k[i]
        for j in range(3):
            if j != i:
                k[j] = M[i, j] / k[i]
        k /= np.linalg.norm(k)
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx
        self.log = log
        self.meta = None
        self.obs = None
        self.ended = False
        self.complete = False
        self.grip = -1.0

    def observe(self):
        r = self.ctx.call("robomme_observe")
        self.obs = decode_obs(r)
        first = self.meta is None
        self.meta = self.obs["meta"]
        self._upd(self.meta)
        if first:
            self.grip = 1.0 if sum(self.meta["state"][6:8]) < 0.07 else -1.0
        return self.obs

    def _upd(self, meta):
        if meta is None:
            return
        if meta.get("episode_ended"):
            self.ended = True
        tr = meta.get("transition") or {}
        if meta.get("task_complete") or tr.get("task_complete"):
            self.complete = True
        if tr.get("episode_ended"):
            self.ended = True

    @property
    def state(self):
        return np.array(self.meta["state"])

    @property
    def pos(self):
        return self.state[:3]

    def remaining(self):
        lim = self.meta.get("episode_limits", {})
        return lim.get("remaining_episode_steps", 10**6)

    def step(self, rows):
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        rem = self.remaining()
        if rem <= 0 or self.ended:
            self.ended = True
            return None
        rows = rows[:max(1, rem)]
        out = None
        for i in range(0, len(rows), 100):
            r = self.ctx.call("robomme_step", actions=rows[i:i + 100])
            self.obs = decode_obs(r)
            m = self.obs["meta"]
            if m is not None and "state" in m:
                self.meta = m
            self._upd(m)
            out = m
            if self.ended:
                break
        return out

    def rot_delta(self, target_rv):
        """World-frame rotvec taking current orientation to target."""
        Rc = rotvec_to_R(self.state[3:6])
        Rt = rotvec_to_R(np.asarray(target_rv, float))
        return R_to_rotvec(Rt @ Rc.T)

    def move_to(self, target, grip=None, rot=None, tol=0.004, rot_tol=0.03, max_steps=120,
                gain=2.2, vmax=1.0, rgain=2.0, rmax=0.6, **_):
        """Closed-loop P control, one row per call. Returns True if within tol."""
        if grip is not None:
            self.grip = grip
        target = np.asarray(target, float)
        for k in range(max_steps):
            if self.ended:
                return False
            d = target - self.pos
            rd = self.rot_delta(rot) if rot is not None else np.zeros(3)
            if np.linalg.norm(d) < tol and np.linalg.norm(rd) < rot_tol:
                return True
            a = gain * d / POS_SCALE
            n = np.linalg.norm(a)
            if n > vmax:
                a = a * vmax / n
            r_ = rgain * rd / ROT_SCALE
            nr = np.linalg.norm(r_)
            if nr > rmax:
                r_ = r_ * rmax / nr
            self.step([list(a) + list(r_) + [self.grip]])
        return bool(np.linalg.norm(target - self.pos) < 2 * tol)

    def gripper(self, g, n=15, min_n=4):
        """Command gripper until fingers stop moving (or n steps)."""
        self.grip = g
        prev = self.state[6:8].sum()
        for k in range(n):
            self.step([[0, 0, 0, 0, 0, 0, g]])
            cur = self.state[6:8].sum()
            if k + 1 >= min_n and abs(cur - prev) < 0.0005:
                break
            prev = cur
            if self.ended:
                break
