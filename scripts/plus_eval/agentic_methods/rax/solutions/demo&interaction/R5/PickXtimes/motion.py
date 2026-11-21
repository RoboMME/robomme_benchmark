import json, math, time
import numpy as np
from perception import decode

LOG = []
GAIN = 0.02  # measured: action 1.0 moves ~0.02 m per step


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = None
        self.imgs = None
        self.ended = False
        self.complete = False
        self.grip = -1.0
        self.observe()

    def observe(self):
        r = self.ctx.call("robomme_observe")
        self.meta, self.imgs = decode(r)
        self._upd(self.meta)
        return self.meta, self.imgs

    def _upd(self, meta):
        if meta.get("episode_ended"):
            self.ended = True
        tr = meta.get("transition") or {}
        for s in tr.get("steps", []) or []:
            if s.get("task_complete"):
                self.complete = True
            if s.get("episode_ended"):
                self.ended = True

    @property
    def state(self):
        return np.array(self.meta["state"], float)

    @property
    def pos(self):
        return self.state[:3]

    @property
    def remaining(self):
        lim = self.meta.get("episode_limits") or {}
        return lim.get("remaining_episode_steps", 10 ** 6)

    def step(self, rows):
        if self.ended:
            return
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        for i in range(0, len(rows), 100):
            if self.ended:
                return
            r = self.ctx.call("robomme_step", actions=rows[i:i + 100])
            self.meta, self.imgs = decode(r)
            self._upd(self.meta)
            st = (self.meta.get("transition") or {}).get("steps") or []
            if st and st[-1]["step_id"] - st[0]["step_id"] + 1 > len(st):
                log(f"WARNING: step penalty (IK?) steps {st[0]['step_id']}..{st[-1]['step_id']} for {len(st)} rows, pos={self.pos.round(3)}")

    def hold(self, n, grip=None):
        g = self.grip if grip is None else grip
        self.grip = g
        self.step([[0, 0, 0, 0, 0, 0, g]] * n)

    def gripper(self, g, n=12):
        self.hold(n, g)

    def move_to(self, target, speed=0.018, tol=0.004, max_iter=8, drz_total=0.0, settle=1):
        """Move EE to world target (x,y,z) with max speed m/step; optional yaw rotation (rad)."""
        target = np.array(target, float)
        for it in range(max_iter):
            if self.ended:
                return False
            d = target - self.pos
            dist = np.linalg.norm(d)
            if dist < tol and abs(drz_total) < 1e-3:
                return True
            speed = min(speed, 0.019)
            n = max(1, int(math.ceil(dist / speed)))
            nrot = int(math.ceil(abs(drz_total) / 0.15)) if abs(drz_total) > 1e-3 else 0
            n = max(n, nrot)
            per = d / n
            rz = drz_total / n if nrot else 0.0
            rows = [[per[0] / GAIN, per[1] / GAIN, per[2] / GAIN, 0, 0, rz / 0.5, self.grip] for _ in range(n)]
            rows += [[0, 0, 0, 0, 0, 0, self.grip]] * settle
            self.step(rows)
            drz_total = 0.0
        return np.linalg.norm(target - self.pos) < tol * 3


def _fyaw(state):
    from perception import finger_yaw
    return finger_yaw(state)


def rotate_yaw(R, dyaw, tol=0.03, max_iter=4):
    """Rotate the gripper about world z by dyaw (rad) in closed loop."""
    import math
    from perception import wrap90
    goal = _fyaw(R.state) + dyaw
    for _ in range(max_iter):
        err = math.atan2(math.sin(goal - _fyaw(R.state)), math.cos(goal - _fyaw(R.state)))
        if abs(err) < tol or R.ended:
            return True
        n = max(1, int(math.ceil(abs(err) / 0.08)))
        rz = err / n / ROT_GAIN
        R.step([[0, 0, 0, 0, 0, rz, R.grip]] * n + [[0, 0, 0, 0, 0, 0, R.grip]])
    return False


ROT_GAIN = 0.2  # rad per step at action 1.0 (estimate, corrected in closed loop)
