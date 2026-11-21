import math, json
import numpy as np
from perception import parse_reply


def rotvec_to_R(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2
    c = max(-1.0, min(1.0, c)); th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if abs(math.pi - th) < 1e-3:
        # near pi: axis from diagonal
        ax = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        i = int(np.argmax(ax))
        a = R[:, i] + np.eye(3)[:, i]
        a = a / np.linalg.norm(a)
        return a * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


RX_PI = np.diag([1.0, -1.0, -1.0])


def yaw_of(state):
    R = rotvec_to_R(state[3:6]) @ RX_PI.T  # R = Rz(yaw) Rx(pi)
    return math.atan2(R[1, 0], R[0, 0])


class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx
        self.log = log
        self.meta = None
        self.ended = False
        self.success = False
        self.steps_used = 0

    def observe(self):
        r = self.ctx.call("robomme_observe")
        meta, ims = parse_reply(r)
        self.meta = meta
        self._check_end(meta)
        return meta, ims

    def _check_end(self, meta):
        if meta is None:
            return
        if meta.get("episode_ended"):
            self.ended = True
        for s in meta.get("steps", []) or []:
            if isinstance(s, dict):
                if s.get("task_complete"):
                    self.success = True
                if s.get("episode_ended"):
                    self.ended = True
        if meta.get("task_complete"):
            self.success = True

    def remaining(self):
        try:
            return int(self.meta["episode_limits"]["remaining_episode_steps"])
        except Exception:
            return 999

    def step(self, rows):
        if self.ended:
            return None
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows]
        r = self.ctx.call("robomme_step", actions=rows)
        meta, ims = parse_reply(r)
        self.steps_used += len(rows)
        if meta is not None:
            self.meta = meta
            self._check_end(meta)
        return meta, ims

    def state(self):
        return np.array(self.meta["state"], float)

    def move_to(self, xyz, grip, yaw=None, tol=0.004, max_steps=80, vmax=0.6, chunk=4, final_settle=2):
        """Closed-loop Cartesian move. grip: -1 open, +1 close."""
        xyz = np.asarray(xyz, float)
        used = 0
        while used < max_steps and not self.ended:
            s = self.state()
            err = xyz - s[:3]
            rot_err = np.zeros(3)
            if yaw is not None:
                Rt = Rz(yaw) @ RX_PI
                Rc = rotvec_to_R(s[3:6])
                rot_err = R_to_rotvec(Rt @ Rc.T)
            if np.linalg.norm(err) < tol and np.linalg.norm(rot_err) < 0.03:
                break
            rows = []
            n = min(chunk, max_steps - used)
            # plan a chunk assuming each step achieves ~the commanded delta
            pos = s[:3].copy(); rerr = rot_err.copy()
            for _ in range(n):
                e = xyz - pos
                d = e / 0.05
                m = np.max(np.abs(d))
                if m > vmax:
                    d = d * vmax / m
                rr = rerr / 0.5
                mr = np.max(np.abs(rr)) if rr.size else 0
                if mr > 0.5:
                    rr = rr * 0.5 / mr
                rows.append([d[0], d[1], d[2], rr[0], rr[1], rr[2], grip])
                pos = pos + d * 0.05
                rerr = rerr - rr * 0.5
                if np.linalg.norm(xyz - pos) < 1e-4 and np.linalg.norm(rerr) < 1e-3:
                    break
            self.step(rows)
            used += len(rows)
        if final_settle and not self.ended:
            self.step([[0, 0, 0, 0, 0, 0, grip]] * final_settle)
            used += final_settle
        return used

    def gripper(self, grip, n=8):
        return self.step([[0, 0, 0, 0, 0, 0, grip]] * n)
