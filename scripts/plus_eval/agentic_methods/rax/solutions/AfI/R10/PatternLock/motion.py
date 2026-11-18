"""Closed-loop end-effector motion with the official 7-D delta action."""
import json
import numpy as np


def rotvec_to_R(r):
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th)*Kx + (1-np.cos(th))*Kx@Kx


def R_to_rotvec(R):
    c = (np.trace(R)-1)/2
    c = min(1.0, max(-1.0, c))
    th = np.arccos(c)
    if th < 1e-6:
        return np.zeros(3)
    if abs(np.pi-th) < 1e-3:
        # near pi
        A = (R+np.eye(3))/2
        k = np.sqrt(np.maximum(np.diag(A), 0))
        i = int(np.argmax(k))
        k = A[:, i]/np.sqrt(max(A[i, i], 1e-12))
        return k*th
    w = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])/(2*np.sin(th))
    return w*th


def parse(reply):
    if isinstance(reply, dict):
        reply = reply.get("content", [])
    for b in reply:
        if isinstance(b, dict) and b.get("type") == "text":
            return json.loads(b["text"])
    raise RuntimeError("no json block")


class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx
        self.log = log
        self.last = None
        self.ended = False
        self.task_complete = False
        self.R_ref = None

    def observe(self):
        self.last = parse(self.ctx.call("robomme_observe"))
        self._check(self.last)
        return self.last

    def _check(self, pkt):
        if pkt.get("episode_ended"):
            self.ended = True
        tr = pkt.get("transition") or {}
        if tr.get("episode_ended"):
            self.ended = True
        if tr.get("task_complete"):
            self.task_complete = True
        for s in (pkt.get("steps") or []) + (tr.get("steps") or []):
            if isinstance(s, dict):
                if s.get("task_complete"):
                    self.task_complete = True
                if s.get("episode_ended"):
                    self.ended = True
        if pkt.get("task_complete"):
            self.task_complete = True

    def remaining(self):
        lim = (self.last or {}).get("episode_limits", {})
        return lim.get("remaining_episode_steps", 10**6)

    def state(self):
        return np.array(self.last["state"], float)

    def step(self, rows):
        if self.ended:
            return self.last
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        pkt = parse(self.ctx.call("robomme_step", actions=rows))
        self.last = pkt
        self._check(pkt)
        return pkt

    def row_towards(self, target, vmax=0.6, gain=1.0, rot_gain=0.5):
        s = self.state()
        err = np.asarray(target) - s[:3]
        d = err / 0.05 * gain
        n = np.linalg.norm(d)
        if n > vmax:
            d = d * vmax / n
        rot = np.zeros(3)
        if self.R_ref is not None:
            Rc = rotvec_to_R(s[3:6])
            rv = R_to_rotvec(self.R_ref @ Rc.T)
            rot = np.clip(rv / 0.5 * rot_gain, -0.3, 0.3)
        return list(d) + list(rot) + [0.0]

    def goto(self, target, tol=0.004, vmax=0.6, max_steps=60, chunk=1):
        """Move TCP to target (world m) with per-step feedback. Returns final error."""
        target = np.asarray(target, float)
        n = 0
        while n < max_steps and not self.ended and self.remaining() > 0:
            err = np.linalg.norm(target - self.state()[:3])
            if err < tol:
                break
            rows = [self.row_towards(target, vmax=vmax) for _ in range(chunk)]
            self.step(rows)
            n += chunk
        return float(np.linalg.norm(target - self.state()[:3])), n
