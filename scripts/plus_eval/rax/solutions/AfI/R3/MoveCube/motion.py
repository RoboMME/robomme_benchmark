"""Closed-loop end-effector motion on top of robomme_step."""
import json, math, time
import numpy as np
from scipy.spatial.transform import Rotation as Rot

POS_SCALE = 0.05
ROT_SCALE = 0.5


class Ended(Exception):
    pass


class Stalled(Ended):
    """Env burned far more steps than rows sent (seen in ep2 at the workspace edge: state froze, 100 steps/row)."""
    pass


def R_topdown(yaw):
    """Gripper pointing down, rotated by yaw about world z (yaw=0 is the reset pose)."""
    return Rot.from_euler('z', yaw) * Rot.from_rotvec([math.pi, 0, 0])


def yaw_of(rotvec):
    """Yaw of a (roughly) top-down orientation relative to the reset pose."""
    R = Rot.from_rotvec(rotvec) * Rot.from_rotvec([math.pi, 0, 0]).inv()
    # R ~ Rz(yaw) (tilt small): take x-axis image
    x = R.apply([1, 0, 0])
    return math.atan2(x[1], x[0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx
        self.log = log
        self.meta = None
        self.state = None
        self.ended = False
        self.remaining = None
        self.steps = 0
        self.grip = -1.0
        self.last_reply = None

    def _update(self, reply):
        meta = json.loads(reply["content"][0]["text"])
        self.meta = meta
        self.last_reply = reply
        if "state" in meta:
            self.state = np.array(meta["state"], dtype=float)
        self.ended = bool(meta.get("episode_ended", False))
        lim = meta.get("episode_limits") or {}
        self.remaining = lim.get("remaining_episode_steps", self.remaining)
        return meta

    def observe(self):
        r = self.ctx.call("robomme_observe")
        self._update(r)
        return r

    def step(self, rows):
        if self.ended:
            raise Ended()
        rows = [[float(np.clip(x, -1, 1)) for x in row] for row in rows]
        before = self.remaining
        r = self.ctx.call("robomme_step", actions=rows)
        self._update(r)
        self.steps += len(rows)
        if self.ended:
            raise Ended()
        if before is not None and self.remaining is not None and before - self.remaining > len(rows) + 3:
            self.log(f"STALL: {before - self.remaining} steps consumed by {len(rows)} rows at {np.round(self.pos,3)}")
            raise Stalled()
        return r

    @property
    def pos(self):
        return self.state[0:3].copy()

    @property
    def rotvec(self):
        return self.state[3:6].copy()

    @property
    def width(self):
        return float(self.state[6] + self.state[7])

    def row_towards(self, target, yaw=None, grip=None, vmax=0.6, wmax=0.6, R_des=None):
        d = np.asarray(target, dtype=float) - self.pos
        a = d / POS_SCALE
        n = np.linalg.norm(a)
        if n > vmax:
            a = a * vmax / n
        rv = np.zeros(3)
        if R_des is None and yaw is not None:
            R_des = R_topdown(yaw)
        if R_des is not None:
            Rc = Rot.from_rotvec(self.rotvec)
            rv = (R_des * Rc.inv()).as_rotvec() / ROT_SCALE
            m = np.linalg.norm(rv)
            if m > wmax:
                rv = rv * wmax / m
        g = self.grip if grip is None else grip
        return [a[0], a[1], a[2], rv[0], rv[1], rv[2], g]

    def move_to(self, target, yaw=None, grip=None, tol=0.004, ytol=0.03, max_steps=150,
                vmax=0.6, wmax=0.6, chunk=1, R_des=None):
        """Closed-loop move; returns True when within tolerance."""
        if grip is not None:
            self.grip = grip
        target = np.asarray(target, dtype=float)
        if yaw is not None and R_des is None:
            R_des = R_topdown(yaw)
        n = 0
        while n < max_steps:
            perr = np.linalg.norm(target - self.pos)
            rerr = 0.0
            if R_des is not None:
                rerr = np.linalg.norm((R_des * Rot.from_rotvec(self.rotvec).inv()).as_rotvec())
            if perr < tol and rerr < ytol:
                return True
            # chunk: repeat the same row a few times when far away
            k = chunk if perr > 0.03 else 1
            row = self.row_towards(target, grip=self.grip, vmax=vmax, wmax=wmax, R_des=R_des)
            self.step([row] * k)
            n += k
        self.log(f"move_to timeout target={np.round(target,3)} pos={np.round(self.pos,3)} rerr={rerr:.3f}")
        return False

    def hold(self, n, grip=None):
        if grip is not None:
            self.grip = grip
        p = self.pos
        R = Rot.from_rotvec(self.rotvec)
        rows = []
        for _ in range(n):
            rows.append([0, 0, 0, 0, 0, 0, self.grip])
        self.step(rows)

    def set_gripper(self, g, n=12):
        self.grip = g
        self.step([[0, 0, 0, 0, 0, 0, g]] * n)
