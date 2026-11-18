import json
import numpy as np
from obs import parse

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx
        self.log = log
        self.meta = None
        self.imgs = None
        self.ended = False
        self.complete = False
        self.grip = -1.0
        self.frames = []   # (step_id, imgs) of returned final frames
        self.Rt = None     # target orientation (world rotation matrix)
        self.on_frame = None   # callback after every step call (e.g. tracker update)
        self.track_chunk = None  # if set, limit rows per call (so each frame is seen)

    def set_down(self, yaw):
        """Target: gripper pointing down, hand y-axis at world angle yaw."""
        c, s = np.cos(yaw), np.sin(yaw)
        y = np.array([c, s, 0.0]); z = np.array([0, 0, -1.0]); x = np.cross(y, z)
        self.Rt = np.stack([x, y, z], 1)

    def rot_err(self):
        Rc = rotvec_to_R(self.state[3:6])
        return R_to_rotvec(self.Rt @ Rc.T)

    def observe(self):
        r = self.ctx.call("robomme_observe")
        self.meta, self.imgs = parse(r)
        return self.meta, self.imgs

    @property
    def state(self):
        return np.array(self.meta["state"])

    @property
    def remaining(self):
        return self.meta["episode_limits"]["remaining_episode_steps"]

    def step(self, rows):
        if self.ended:
            return self.meta
        if self.track_chunk and len(rows) > self.track_chunk:
            for k in range(0, len(rows), self.track_chunk):
                self.step(rows[k:k + self.track_chunk])
            return self.meta
        r = self.ctx.call("robomme_step", actions=[[float(x) for x in row] for row in rows])
        self.meta, self.imgs = parse(r)
        if self.meta.get("episode_ended"):
            self.ended = True
        for s in (self.meta.get("transition") or {}).get("steps", []) or []:
            if isinstance(s, dict) and s.get("task_complete"):
                self.complete = True
            if isinstance(s, dict) and s.get("episode_ended"):
                self.ended = True
        tr = self.meta.get("transition") or {}
        if tr.get("episode_ended"):
            self.ended = True
        if self.meta.get("task_complete") or tr.get("task_complete"):
            self.complete = True
        if self.on_frame is not None and not self.ended:
            try:
                self.on_frame(self)
            except Exception as e:
                self.log("on_frame error", repr(e))
        return self.meta

    def move_to(self, target, grip=None, tol=0.004, max_steps=60, gain=1.0, vmax=1.0, chunk=4):
        """Closed-loop translate TCP to target (world m), keeping orientation."""
        if grip is not None:
            self.grip = grip
        target = np.asarray(target, float)
        n = 0
        while n < max_steps and not self.ended:
            pos = self.state[:3]
            err = target - pos
            if np.linalg.norm(err) < tol and (self.Rt is None or np.linalg.norm(self.rot_err()) < 0.03):
                return True
            rows = []
            # plan a short chunk assuming nominal tracking
            p = pos.copy()
            rerr = self.rot_err() if self.Rt is not None else np.zeros(3)
            for _ in range(min(chunk, max_steps - n)):
                e = target - p
                d = np.clip(gain * e / 0.05, -vmax, vmax)
                r = np.clip(rerr / 0.5, -0.3, 0.3)
                rerr = rerr - r * 0.5
                rows.append(list(d) + list(r) + [self.grip])
                p = p + d * 0.05
                if np.linalg.norm(target - p) < tol:
                    break
            self.step(rows)
            n += len(rows)
        return np.linalg.norm(target - self.state[:3]) < tol * 2

    def hold(self, n, grip=None):
        if grip is not None:
            self.grip = grip
        self.step([[0, 0, 0, 0, 0, 0, self.grip]] * n)

def rotvec_to_R(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2
    th = np.arccos(np.clip(c, -1, 1))
    if th < 1e-6:
        return np.zeros(3)
    if th > np.pi - 1e-3:
        w, V = np.linalg.eigh((R + np.eye(3)) / 2)
        k = V[:, -1]
        return k * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return v * th

def wrap(a, period):
    return (a + period / 2) % period - period / 2

def hand_yaw(state):
    R = rotvec_to_R(state[3:6])
    y = R[:, 1]
    return float(np.arctan2(y[1], y[0]))

def rotate_yaw(robot, dyaw, max_rate=0.3):
    """Rotate about world z by dyaw radians (closed loop on measured yaw)."""
    target = hand_yaw(robot.state) + dyaw
    for _ in range(6):
        err = wrap(target - hand_yaw(robot.state), 2 * np.pi)
        if abs(err) < 0.02:
            return True
        n = int(np.ceil(abs(err) / (0.5 * max_rate)))
        per = err / n / 0.5
        robot.step([[0, 0, 0, 0, 0, per, robot.grip]] * n)
    return False
