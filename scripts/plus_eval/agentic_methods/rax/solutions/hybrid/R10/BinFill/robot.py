import json, math, time
import numpy as np
from vision import parse


def rotvec_to_R(r):
    r = np.asarray(r, float); th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2; c = max(-1.0, min(1.0, c)); th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi
        M = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(M), 0, None))
        i = int(np.argmax(k)); k[i] = k[i]
        for j in range(3):
            if j != i:
                k[j] = M[i, j] / k[i]
        return k / np.linalg.norm(k) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Ended(Exception):
    pass


class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log
        self.meta = None; self.imgs = None
        self.ended = False; self.complete = False
        self.grip = -1.0
        self.ref_R = None

    def observe(self):
        res = self.ctx.call("robomme_observe")
        self.meta, self.imgs = parse(res)
        return self.meta, self.imgs

    @property
    def state(self):
        return np.array(self.meta["state"], float)

    @property
    def pos(self):
        return self.state[:3]

    @property
    def R(self):
        return rotvec_to_R(self.state[3:6])

    def remaining(self):
        return self.meta.get("episode_limits", {}).get("remaining_episode_steps", 0)

    def step(self, rows):
        if self.ended:
            raise Ended()
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        res = self.ctx.call("robomme_step", actions=rows)
        self.meta, self.imgs = parse(res)
        tr = self.meta.get("transition", {})
        for s in tr.get("steps", []):
            if s.get("task_complete"):
                self.complete = True
        if self.meta.get("episode_ended") or tr.get("episode_ended"):
            self.ended = True
            for s in tr.get("steps", []):
                if s.get("episode_ended"):
                    self.log("EPISODE ENDED at step %s task_complete=%s" % (s.get("step_id"), s.get("task_complete")))
            raise Ended()
        return self.meta

    def yaw(self):
        # yaw of gripper x axis in world
        R = self.R
        return math.atan2(R[1, 0], R[0, 0])

    def rot_delta(self, target_R):
        # world-frame rotation vector from current to target
        return R_to_rotvec(target_R @ self.R.T)

    def move_to(self, target, yaw=None, grip=None, max_d=0.035, tol=0.006, max_iter=12, chunk=6, rot_tol=0.03):
        """Move EE to target xyz (and optionally yaw for top-down grasp)."""
        target = np.asarray(target, float)
        if grip is not None:
            self.grip = grip
        if self.ref_R is None:
            self.ref_R = self.R.copy()
        tgtR = None
        if yaw is not None:
            # top-down: base orientation = R of [pi,0,0] rotated by yaw about world z
            base = rotvec_to_R([math.pi, 0, 0])
            tgtR = Rz(yaw) @ base
        for it in range(max_iter):
            err = target - self.pos
            rv = self.rot_delta(tgtR) if tgtR is not None else np.zeros(3)
            if np.linalg.norm(err) < tol and np.linalg.norm(rv) < rot_tol:
                return True
            dist = np.linalg.norm(err)
            n = int(min(chunk, max(1, math.ceil(dist / max_d))))
            rows = []
            # plan n rows dividing remaining error
            for k in range(n):
                d = err / n
                r = rv / n if k == 0 or True else 0
                rr = np.clip(r / 0.5, -0.6, 0.6)
                rows.append(list(np.clip(d / 0.05, -1, 1)) + list(rr) + [self.grip])
            self.step(rows)
        return np.linalg.norm(target - self.pos) < 3 * tol

    def hold(self, n, grip=None):
        if grip is not None:
            self.grip = grip
        self.step([[0, 0, 0, 0, 0, 0, self.grip]] * n)

    def fingers(self):
        return float(self.state[6] + self.state[7])
