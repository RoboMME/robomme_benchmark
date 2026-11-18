import numpy as np, math, time
from rlib import parse

def rotvec_to_R(rv):
    rv = np.asarray(rv, float); th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K

def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2; c = max(-1, min(1, c)); th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        # near pi
        w, V = np.linalg.eigh((R + np.eye(3)) / 2)
        k = V[:, np.argmax(w)]
        return k * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return v * th

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log
        self.meta = None; self.imgs = None; self.ended = False; self.success = False
        self.grip = -1.0
        self.n_calls = 0
        self.chunk = 100
        self.on_obs = None
        self.observe()

    def _take(self, reply):
        meta, imgs = parse(reply)
        self.meta, self.imgs = meta, imgs
        if meta.get("episode_ended"):
            self.ended = True
        tr = meta.get("transition") or {}
        for s in tr.get("steps", []):
            if s.get("task_complete"):
                self.success = True
            if s.get("episode_ended"):
                self.ended = True
        return meta

    def observe(self):
        return self._take(self.ctx.call("robomme_observe"))

    @property
    def state(self):
        return np.array(self.meta["state"], float)

    @property
    def pos(self):
        return self.state[:3]

    @property
    def R(self):
        return rotvec_to_R(self.state[3:6])

    def yaw(self):
        # closing axis direction = hand y axis, projected
        y = self.R[:, 1]
        return math.atan2(y[1], y[0])

    def remaining(self):
        lim = self.meta.get("episode_limits") or {}
        return lim.get("remaining_episode_steps", 9999)

    def step(self, rows):
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        i = 0
        while i < len(rows) and not self.ended:
            rem = self.remaining()
            n = min(self.chunk, 100, max(0, rem), len(rows) - i)
            if n <= 0:
                self.ended = True; return
            self.n_calls += 1
            self._take(self.ctx.call("robomme_step", actions=rows[i:i + n]))
            i += n
            if self.on_obs is not None:
                try:
                    self.on_obs(self)
                except Exception as e:
                    self.log("on_obs error", repr(e))

    def hold(self, n, grip=None):
        if grip is not None:
            self.grip = grip
        self.step([[0, 0, 0, 0, 0, 0, self.grip]] * n)

    def move_to(self, target, yaw=None, grip=None, vmax=0.6, tol=0.004, max_iter=8, yaw_tol=0.03):
        """closed loop move of TCP to world target xyz, optional yaw of closing axis (mod pi)"""
        if grip is not None:
            self.grip = grip
        target = np.asarray(target, float)
        for it in range(max_iter):
            if self.ended:
                return False
            d = target - self.pos
            dyaw = 0.0
            if yaw is not None:
                dyaw = wrap(yaw - self.yaw())
                # closing axis symmetric under pi
                if dyaw > math.pi / 2: dyaw -= math.pi
                if dyaw < -math.pi / 2: dyaw += math.pi
            dist = np.linalg.norm(d)
            if dist < tol and abs(dyaw) < yaw_tol:
                return True
            n_t = int(math.ceil(dist / (0.05 * vmax)))
            n_r = int(math.ceil(abs(dyaw) / (0.5 * 0.5)))
            n = max(1, n_t, n_r)
            # full-orientation correction: keep gripper pointing down
            rows = []
            per = d / n / 0.05
            # rotation: compute desired R (down-pointing, given yaw) error as rotvec
            Rc = self.R
            cy = self.yaw() + dyaw
            Rd = desired_R(cy)
            err = R_to_rotvec(Rd @ Rc.T)
            rper = err / n / 0.5
            for i in range(n):
                rows.append([per[0], per[1], per[2], rper[0], rper[1], rper[2], self.grip])
            rows.append([0, 0, 0, 0, 0, 0, self.grip])
            self.step(rows)
        return np.linalg.norm(target - self.pos) < 3 * tol

def desired_R(yaw):
    # gripper pointing down (z axis = -world z), hand y axis along yaw
    y = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    z = np.array([0, 0, -1.0])
    x = np.cross(y, z)
    return np.stack([x, y, z], 1)
