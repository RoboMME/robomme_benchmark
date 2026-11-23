"""Closed-loop end-effector motion helpers on the official 7-D action."""
import json
import numpy as np

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
        self.remaining = None
        self.step_id = 0
        self.grip = -1.0
        self.last_meta = None

    def observe(self):
        from robo import decode
        meta, rgb, dep = decode(self.ctx.call("robomme_observe"))
        self._upd(meta)
        return meta, rgb, dep

    def _upd(self, meta):
        self.last_meta = meta
        if "state" in meta and meta["state"] is not None:
            self.state = np.array(meta["state"], float)
        lim = meta.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        if meta.get("episode_ended"):
            self.ended = True
        for s in (meta.get("transition") or {}).get("steps", []) or meta.get("steps", []) or []:
            if isinstance(s, dict):
                if s.get("task_complete"):
                    self.complete = True
                if s.get("episode_ended"):
                    self.ended = True
        if (meta.get("transition") or {}).get("task_complete"):
            self.complete = True
        if meta.get("task_complete"):
            self.complete = True
        self.step_id = meta.get("step_id", self.step_id)

    def step(self, rows):
        if self.ended:
            return None
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        reply = self.ctx.call("robomme_step", actions=rows)
        meta = json.loads(reply["content"][0]["text"])
        self._upd(meta)
        return meta

    def move_to(self, target, grip=None, max_v=0.6, tol=0.004, max_steps=80, chunk=4):
        """Proportional position control to `target` (world m), keeping orientation down."""
        if grip is not None:
            self.grip = grip
        target = np.asarray(target, float)
        n = 0
        while n < max_steps and not self.ended:
            p = self.state[:3]
            err = target - p
            if np.linalg.norm(err) < tol:
                break
            rows = []
            # predict: send a few rows using decaying error estimates
            e = err.copy()
            for _ in range(chunk):
                d = e / 0.05
                m = np.max(np.abs(d))
                if m > max_v:
                    d = d * max_v / m
                rot = self.rot_correction()
                rows.append(list(d) + list(rot) + [self.grip])
                e = e - d * 0.05 * 0.8
                if np.linalg.norm(e) < tol:
                    break
            self.step(rows)
            n += len(rows)
        return np.linalg.norm(target - self.state[:3])

    def rot_correction(self, yaw=None):
        """Keep gripper pointing down: target axis-angle (pi,0,0)-like orientation."""
        from scipy.spatial.transform import Rotation as R
        cur = R.from_rotvec(self.state[3:6])
        tgt = R.from_rotvec([np.pi, 0, 0])
        if yaw is not None:
            tgt = R.from_euler("z", yaw) * tgt
        err = (tgt * cur.inv()).as_rotvec()
        d = err / 0.5
        m = np.max(np.abs(d))
        if m > 0.5:
            d = d * 0.5 / m
        return d

    def gripper(self, value, steps=8):
        self.grip = value
        p = self.state[:3]
        rows = [[0, 0, 0] + list(self.rot_correction()) + [value] for _ in range(steps)]
        self.step(rows)

    def hold(self, steps):
        rows = [[0, 0, 0] + list(self.rot_correction()) + [self.grip] for _ in range(steps)]
        self.step(rows)
