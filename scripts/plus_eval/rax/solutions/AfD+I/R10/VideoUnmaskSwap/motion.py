import numpy as np
from scipy.spatial.transform import Rotation as Rot
from io_util import parse

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx, self.log = ctx, log
        self.meta, self.imgs = None, None
        self.ended = False
        self.complete = False
        self.R_down = None

    def step(self, rows):
        r = self.ctx.call("robomme_step", actions=[list(map(float, row)) for row in rows])
        meta, imgs = parse(r)
        self.meta, self.imgs = meta, imgs
        tr = meta.get("transition") or {}
        for s in tr.get("steps", []) or []:
            if s.get("task_complete"):
                self.complete = True
            if s.get("episode_ended"):
                self.ended = True
        if meta.get("episode_ended"):
            self.ended = True
        if meta.get("task_complete"):
            self.complete = True
        return meta, imgs

    def observe(self):
        meta, imgs = parse(self.ctx.call("robomme_observe"))
        self.meta, self.imgs = meta, imgs
        return meta, imgs

    @property
    def state(self):
        return np.array(self.meta["state"], float)

    @property
    def remaining(self):
        return self.meta.get("episode_limits", {}).get("remaining_episode_steps", 9999)

    def rot(self):
        return Rot.from_rotvec(self.state[3:6])

    def yaw(self):
        Rm = self.rot().as_matrix()
        return float(np.arctan2(Rm[1, 0], Rm[0, 0]))

    def rot_err(self, yaw_target):
        if self.R_down is None:
            self.R_down = Rot.from_rotvec([np.pi, 0, 0])
        des = Rot.from_euler("z", yaw_target) * self.R_down
        return (des * self.rot().inv()).as_rotvec()

    def move_to(self, xyz, yaw=None, grip=-1.0, tol=0.004, max_steps=80, gain=1.0, vmax=0.6, chunk=1):
        """Closed-loop move to xyz (and absolute world yaw of tool x-axis if given)."""
        xyz = np.asarray(xyz, float)
        for i in range(max_steps):
            if self.ended:
                return False
            s = self.state
            e = xyz - s[:3]
            re = self.rot_err(yaw) if yaw is not None else self.rot_err(self.yaw())
            if np.linalg.norm(e) < tol and np.linalg.norm(re) < 0.03:
                return True
            dp = np.clip(e / 0.05 * gain, -vmax, vmax)
            dr = np.clip(re / 0.5, -0.6, 0.6)
            self.step([[*dp, *dr, grip]] * chunk)
        return False

    def hold(self, n, grip):
        rows = [[0, 0, 0, 0, 0, 0, grip]] * n
        return self.step(rows)
