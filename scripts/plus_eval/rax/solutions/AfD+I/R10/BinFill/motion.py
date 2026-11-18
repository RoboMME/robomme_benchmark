"""Closed-loop Cartesian motion primitives on official 7-D rows."""
import math, time
import numpy as np
from rio import decode

DOWN = np.array([math.pi, 0.0, 0.0])

def aa_to_R(v):
    th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th)*K + (1-math.cos(th))*K@K

def R_to_aa(R):
    c = max(-1.0, min(1.0, (np.trace(R)-1)/2)); th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if abs(th - math.pi) < 1e-3:
        # axis from diagonal
        ax = np.sqrt(np.maximum((np.diag(R)+1)/2, 0))
        i = int(np.argmax(ax)); k = R[:, i] + np.eye(3)[:, i]; k = k/np.linalg.norm(k)
        return k*th
    w = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])/(2*math.sin(th))
    return w*th

def target_R(yaw):
    """Gripper pointing down, rotated by yaw about world z."""
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return Rz @ aa_to_R(DOWN)

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log
        self.meta = None; self.imgs = None; self.ended = False; self.complete = False
        self.grip = -1.0; self.yaw = 0.0

    def observe(self):
        m, im = decode(self.ctx.call("robomme_observe"))
        self.meta, self.imgs = m, im
        return m, im

    @property
    def state(self):
        return np.array(self.meta["state"])

    def remaining(self):
        try:
            return int(self.meta["episode_limits"]["remaining_episode_steps"])
        except Exception:
            return 10**6

    def step(self, rows):
        if self.ended:
            return
        rows = [[float(max(-1, min(1, x))) for x in r] for r in rows]
        rem = self.remaining()
        rows = rows[:max(0, min(100, rem))]
        if not rows:
            self.ended = True; return
        m, im = decode(self.ctx.call("robomme_step", actions=rows))
        self.meta, self.imgs = m, im
        if m.get("episode_ended"):
            self.ended = True
        for s in (m.get("steps") or []):
            if isinstance(s, dict) and s.get("task_complete"):
                self.complete = True
        if m.get("task_complete"):
            self.complete = True
        tr = m.get("transition") or {}
        if tr.get("task_complete"):
            self.complete = True
        if tr.get("episode_ended"):
            self.ended = True
        for s in (tr.get("steps") or []):
            if isinstance(s, dict) and s.get("task_complete"):
                self.complete = True
            if isinstance(s, dict) and s.get("episode_ended"):
                self.ended = True

    def rot_cmd(self, yaw=None):
        yaw = self.yaw if yaw is None else yaw
        Rc = aa_to_R(self.state[3:6]); Rt = target_R(yaw)
        e = R_to_aa(Rt @ Rc.T)
        return np.clip(e/0.5*0.8, -0.6, 0.6)

    def sid(self):
        return int(self.meta.get("step_id", 0))

    def move_to(self, xyz, grip=None, vmax=0.6, tol=0.006, max_iter=12, yaw=None, chunk=8, vel=None, t0=None, lead=0):
        """vmax in action units (x0.05 m per step). vel: target velocity (m/step) from time t0."""
        if grip is not None:
            self.grip = grip
        if yaw is not None:
            self.yaw = yaw
        base = np.array(xyz, float)
        vel = np.zeros(3) if vel is None else np.array(vel, float)
        t0 = self.sid() if t0 is None else t0
        xyz = base
        if np.linalg.norm(vel) > 0:
            tol = tol + 2.5*float(np.linalg.norm(vel))
        for it in range(max_iter):
            if self.ended:
                return False
            xyz = base + vel*(self.sid()-t0+lead)
            p = self.state[:3]; d = xyz - p; dist = np.linalg.norm(d)
            rc = self.rot_cmd()
            if dist < tol and np.linalg.norm(rc) < 0.08:
                return True
            n = int(math.ceil(dist/(0.05*vmax)))
            n = max(1, min(chunk if dist > 0.03 else 3, n))
            step = d/max(1, math.ceil(dist/(0.05*vmax)))/0.05
            if dist <= 0.05*vmax*n:
                step = d/n/0.05
            step = step + vel/0.05
            rows = []
            for k in range(n):
                rr = rc if k == 0 else rc*0.0
                rows.append([step[0], step[1], step[2], rr[0], rr[1], rr[2], self.grip])
            self.step(rows)
        p = self.state[:3]
        xyz = base + vel*(self.sid()-t0)
        return np.linalg.norm(xyz-p) < 3*tol

    def gripper(self, g, n=8):
        self.grip = g
        self.step([[0, 0, 0, 0, 0, 0, g]]*n)

    def width(self):
        return float(self.state[6] + self.state[7])
