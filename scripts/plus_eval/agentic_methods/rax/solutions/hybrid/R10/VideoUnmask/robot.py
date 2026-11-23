"""Robot motion helpers over ctx (MCP)."""
import json
import time
import numpy as np


def rotvec_to_R(r):
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2
    c = np.clip(c, -1, 1)
    th = np.arccos(c)
    if th < 1e-6:
        return np.zeros(3)
    if np.pi - th < 1e-3:
        # near pi
        M = (R + np.eye(3)) / 2
        k = np.sqrt(np.maximum(np.diag(M), 0))
        i = np.argmax(k)
        k = M[i] / np.sqrt(M[i, i])
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


R_DOWN = rotvec_to_R([np.pi, 0, 0])


class EpisodeEnded(Exception):
    pass


class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx
        self.log = log
        self.state = None
        self.steps_used = 0
        self.remaining = None
        self.ended = False
        self.task_complete = None
        self.last_meta = None

    def _meta(self, reply):
        blocks = reply.get("content", reply) if isinstance(reply, dict) else reply
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                try:
                    return json.loads(b["text"])
                except Exception:
                    continue
        raise RuntimeError("no json in reply")

    def observe(self):
        meta = self._meta(self.ctx.call("robomme_observe"))
        self._update(meta)
        return meta

    def _update(self, meta):
        self.last_meta = meta
        if "state" in meta and meta["state"] is not None:
            self.state = np.array(meta["state"], float)
        lim = meta.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        if meta.get("episode_ended"):
            self.ended = True
        tr = meta.get("transition") or {}
        if tr.get("episode_ended"):
            self.ended = True
        for s in list(meta.get("steps", []) or []) + list(tr.get("steps", []) or []):
            if isinstance(s, dict):
                if s.get("task_complete"):
                    self.task_complete = True
                if s.get("episode_ended"):
                    self.ended = True
        if meta.get("task_complete"):
            self.task_complete = True

    def step(self, rows):
        if self.ended:
            raise EpisodeEnded()
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        if self.remaining is not None:
            rows = rows[:max(0, self.remaining)]
            if not rows:
                raise EpisodeEnded("no steps left")
        meta = self._meta(self.ctx.call("robomme_step", actions=rows))
        self._update(meta)
        if self.ended:
            raise EpisodeEnded()
        return meta

    @property
    def pos(self):
        return self.state[:3]

    @property
    def R(self):
        return rotvec_to_R(self.state[3:6])

    @property
    def grip_open(self):
        return float(self.state[6] + self.state[7])

    def row_towards(self, target, R_target, grip, max_lin=0.6, max_rot=0.5):
        d = (np.asarray(target) - self.pos) / 0.05
        n = np.max(np.abs(d))
        if n > max_lin:
            d = d * max_lin / n
        rv = R_to_rotvec(R_target @ self.R.T) / 0.5
        nr = np.max(np.abs(rv))
        if nr > max_rot:
            rv = rv * max_rot / nr
        return list(d) + list(rv) + [grip]

    def move_to(self, target, yaw, grip, tol=0.004, rot_tol=0.03, max_iters=40, max_lin=0.6, chunk=1):
        R_t = Rz(yaw) @ R_DOWN
        for it in range(max_iters):
            err = np.linalg.norm(np.asarray(target) - self.pos)
            rerr = np.linalg.norm(R_to_rotvec(R_t @ self.R.T))
            if err < tol and rerr < rot_tol:
                return True
            self.step([self.row_towards(target, R_t, grip, max_lin=max_lin)])
        self.log(f"move_to not converged: err={err:.4f} rerr={rerr:.3f} pos={np.round(self.pos,3).tolist()}")
        return False

    def hold(self, n, yaw, grip):
        """Hold position while actuating gripper for n steps."""
        R_t = Rz(yaw) @ R_DOWN
        tgt = self.pos.copy()
        rows = []
        for _ in range(n):
            rows.append(self.row_towards(tgt, R_t, grip))
        # rows identical relative, so send one by one for closed loop on pose
        for r in rows:
            self.step([self.row_towards(tgt, R_t, grip)])

    def current_yaw(self):
        Rr = self.R @ R_DOWN.T
        return float(np.arctan2(Rr[1, 0], Rr[0, 0]))
