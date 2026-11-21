import json, numpy as np
from perc import parse

def rotvec_to_R(r):
    r = np.asarray(r, float); th = np.linalg.norm(r)
    if th < 1e-9: return np.eye(3)
    k = r / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2; c = np.clip(c, -1, 1); th = np.arccos(c)
    if th < 1e-6: return np.zeros(3)
    if th > np.pi - 1e-4:
        # near pi
        A = (R + np.eye(3)) / 2; i = np.argmax(np.diag(A)); k = A[:, i] / np.sqrt(max(A[i, i], 1e-12))
        return k * th
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

R_DOWN = rotvec_to_R([-np.pi, 0, 0])

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log; self.meta = None; self.imgs = None
        self.ended = False; self.complete = False; self.grip = -1.0
        self.observe()

    def observe(self):
        self.meta, self.imgs = parse(self.ctx.call("robomme_observe"))
        return self.meta

    @property
    def state(self): return np.array(self.meta["state"], float)
    @property
    def pos(self): return self.state[:3]
    @property
    def remaining(self): return self.meta.get("episode_limits", {}).get("remaining_episode_steps", 0)

    def step(self, rows):
        if self.ended: return
        reply = self.ctx.call("robomme_step", actions=[[float(np.clip(x, -1, 1)) for x in r] for r in rows])
        meta, imgs = parse(reply)
        self.meta, self.imgs = meta, imgs
        self.ended = bool(meta.get("episode_ended"))
        tr = meta.get("transition") or {}
        if isinstance(tr, dict):
            if tr.get("task_complete"): self.complete = True
            if tr.get("episode_ended"): self.ended = True
            for s in tr.get("steps", []) or []:
                if isinstance(s, dict) and s.get("task_complete"): self.complete = True
                if isinstance(s, dict) and s.get("episode_ended"): self.ended = True
        if meta.get("task_complete"): self.complete = True
        return meta

    def rot_cmd(self, R_t=R_DOWN, gain=0.8):
        Rc = rotvec_to_R(self.state[3:6])
        e = R_to_rotvec(R_t @ Rc.T)
        return np.clip(gain * e / 0.5, -0.3, 0.3)

    def move_to(self, target, tol=0.004, vmax=0.04, max_steps=200, batch=3, grip=None, lag=0.45):
        """Closed-loop straight-line move of the TCP to target (world m).
        The arm realises only ~45% of a commanded delta per step, so rows are
        planned against a predicted position p += lag*cmd."""
        if grip is not None: self.grip = grip
        target = np.asarray(target, float); used = 0
        while used < max_steps and not self.ended:
            err = target - self.pos
            if np.linalg.norm(err) < tol: return True
            rows = []; p = self.pos.copy()
            for _ in range(min(batch, max_steps - used)):
                e = target - p; d = np.linalg.norm(e)
                cmd = e if d <= vmax else e / d * vmax
                p = p + lag * cmd
                rows.append(np.concatenate([cmd / 0.05, self.rot_cmd(), [self.grip]]))
                if np.linalg.norm(target - p) < tol * 0.5: break
            self.step(rows); used += len(rows)
        return np.linalg.norm(target - self.pos) < tol * 2

    def hold(self, n, grip=None):
        if grip is not None: self.grip = grip
        rows = [np.concatenate([[0, 0, 0], self.rot_cmd(), [self.grip]]) for _ in range(n)]
        self.step(rows)
