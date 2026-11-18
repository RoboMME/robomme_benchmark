"""Motion helpers over robomme_step (relative world-frame deltas)."""
import json, math, time
import numpy as np

POS_SCALE = 0.05
ROT_SCALE = 0.5


def meta(reply):
    for b in reply.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                return {"raw": b["text"]}
    return {}


def rotvec_to_R(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2; c = max(-1.0, min(1.0, c)); th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: axis from diagonal
        ax = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        i = int(np.argmax(ax)); a = np.zeros(3); a[i] = ax[i]
        for j in range(3):
            if j != i:
                a[j] = R[i, j] / (2 * a[i])
        return a / np.linalg.norm(a) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def down_R(yaw=0.0):
    """Gripper pointing down (tool z = -world z), rotated by yaw about world z.
    Reference: reset orientation rotvec (pi,0,0)."""
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return Rz @ rotvec_to_R([math.pi, 0, 0])


class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log
        self.state = None; self.ended = False; self.complete = False; self.last = None
        self.remaining = None; self.step_id = 0

    def _update(self, m):
        self.last = m
        st = m.get("state") or (m.get("observation") or {}).get("state")
        if st is None and isinstance(m.get("transition"), dict):
            st = m["transition"].get("state")
        if st is not None:
            self.state = np.array(st, float)
        if m.get("episode_ended"):
            self.ended = True
        if m.get("task_complete") is True or (isinstance(m.get("transition"), dict) and m["transition"].get("task_complete") is True):
            self.complete = True
        lim = m.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        if "step_id" in m:
            self.step_id = m["step_id"]

    def observe(self):
        r = self.ctx.call("robomme_observe"); m = meta(r); self._update(m); return m

    def step(self, rows):
        if self.ended:
            return None
        rows = [[float(max(-1, min(1, v))) for v in row] for row in rows]
        if self.remaining is not None:
            rows = rows[:max(1, self.remaining)]
        r = self.ctx.call("robomme_step", actions=rows); m = meta(r); self._update(m); return m

    def rot_err(self, R_goal):
        R_cur = rotvec_to_R(self.state[3:6])
        return R_to_rotvec(R_goal @ R_cur.T)  # world-frame rotation from current to goal

    def move_to(self, target, grip=-1.0, R_goal=None, tol=0.004, max_steps=80, chunk=1, gain=1.0, vmax=1.0):
        """Closed-loop: each call sends `chunk` rows computed from the latest measured state."""
        target = np.asarray(target, float); used = 0
        while used < max_steps and not self.ended:
            err = target - self.state[:3]
            rerr = self.rot_err(R_goal) if R_goal is not None else np.zeros(3)
            if np.linalg.norm(err) < tol and np.linalg.norm(rerr) < 0.03:
                return True
            d = np.clip(gain * err / POS_SCALE, -vmax, vmax)
            r = np.clip(gain * rerr / ROT_SCALE, -1, 1)
            rows = [list(d / chunk) + list(r / chunk) + [grip]] * chunk
            self.step(rows); used += chunk
        return np.linalg.norm(target - self.state[:3]) < tol * 2

    def hold(self, n, grip):
        return self.step([[0, 0, 0, 0, 0, 0, grip]] * n)
