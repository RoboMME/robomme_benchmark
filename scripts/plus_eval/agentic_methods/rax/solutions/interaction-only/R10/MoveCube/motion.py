"""Closed-loop motion primitives over robomme_step."""
import json, math
import numpy as np
from scipy.spatial.transform import Rotation as Rot

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.state = None
        self.ended = False
        self.complete = False
        self.step_id = 0
        self.remaining = 10 ** 6
        self.last_meta = None

    def _meta(self, reply):
        return json.loads(reply["content"][0]["text"])

    def observe(self):
        from perc import decode
        reply = self.ctx.call("robomme_observe")
        meta, imgs = decode(reply)
        self.state = np.array(meta["state"], float)
        self.step_id = meta.get("step_id", self.step_id)
        self.last_meta = meta
        lim = meta.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        if meta.get("episode_ended"):
            self.ended = True
        return meta, imgs

    def step(self, rows):
        if self.ended:
            return None
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        rows = rows[: max(0, min(100, self.remaining))]
        if not rows:
            self.ended = True
            return None
        reply = self.ctx.call("robomme_step", actions=rows)
        meta = self._meta(reply)
        self.state = np.array(meta["state"], float)
        self.step_id = meta.get("step_id", self.step_id)
        lim = meta.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        for s in meta.get("steps", []) or []:
            if s.get("task_complete"):
                self.complete = True
            if s.get("episode_ended"):
                self.ended = True
        if meta.get("task_complete"):
            self.complete = True
        if meta.get("episode_ended"):
            self.ended = True
        self.last_meta = meta
        return meta

    # --- pose helpers
    def pos(self):
        return self.state[:3].copy()

    def R(self):
        return Rot.from_rotvec(self.state[3:6]).as_matrix()

    def yaw(self):
        R = self.R()
        return math.atan2(R[1, 0], R[0, 0])

    def grip_open(self):
        return float(self.state[6] + self.state[7])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def target_rot(yaw):
    # gripper pointing down (x-axis rotated by yaw), same as reset orientation (-pi,0,0) at yaw 0
    return Rot.from_euler("z", yaw) * Rot.from_rotvec([math.pi, 0, 0])


def move_to(rb, xyz, yaw=None, grip=-1.0, tol=0.004, max_steps=150, vmax=1.0, chunk=4, rtol=0.03):
    """Closed-loop move of TCP to xyz with given yaw (keep pointing down)."""
    xyz = np.asarray(xyz, float)
    used = 0
    while used < max_steps and not rb.ended:
        p = rb.pos()
        dp = xyz - p
        Rc = Rot.from_rotvec(rb.state[3:6])
        if yaw is None:
            yaw_t = rb.yaw()
        else:
            yaw_t = yaw
        Rt = target_rot(yaw_t)
        drot = (Rt * Rc.inv()).as_rotvec()
        if np.linalg.norm(dp) < tol and np.linalg.norm(drot) < rtol:
            return True
        rows = []
        n = chunk
        for k in range(n):
            frac = 1.0 / max(1, n - k) if False else 1.0
            a = np.clip(dp / 0.05, -vmax, vmax)
            # do not overshoot: scale for remaining distance assuming it is executed
            r = np.clip(drot / 0.5, -1, 1)
            rows.append(list(a) + list(r) + [grip])
            dp = dp - a * 0.05
            drot = drot - r * 0.5
            if np.linalg.norm(dp) < tol * 0.5 and np.linalg.norm(drot) < rtol * 0.5:
                break
        rb.step(rows)
        used += len(rows)
    return False


def hold(rb, n, grip):
    rows = [[0, 0, 0, 0, 0, 0, grip]] * n
    rb.step(rows)
