"""Motion helpers over the MCP robot tools."""
import math, time
import numpy as np
from perception import parse_reply

POS_SCALE = 0.05
ROT_SCALE = 0.5


def axang_to_R(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def R_to_axang(R):
    c = (np.trace(R) - 1) / 2; c = max(-1.0, min(1.0, c)); th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if abs(th - math.pi) < 1e-3:
        # near pi: axis from diagonal
        w, V = np.linalg.eigh((R + R.T) / 2)
        k = V[:, np.argmax(w)]
        return k * th
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return k * th


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class EpisodeEnded(Exception):
    pass


class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log
        self.meta = None; self.imgs = None
        self.ended = False; self.task_complete = None
        self.step_id = 0; self.remaining = None; self.limit = None
        self.grip_cmd = -1.0

    # ---------- observation ----------
    def _absorb(self, meta):
        if "state" in meta:
            self.state = np.array(meta["state"], float)
        if "step_id" in meta:
            self.step_id = meta["step_id"]
        lim = meta.get("episode_limits") or {}
        if lim:
            self.remaining = lim.get("remaining_episode_steps", self.remaining)
            self.limit = lim.get("episode_step_limit", self.limit)
        if meta.get("episode_ended"):
            self.ended = True
        for k in ("task_complete",):
            if k in meta:
                self.task_complete = meta[k]
        tr = meta.get("transition") or {}
        if isinstance(tr, dict):
            if tr.get("episode_ended"):
                self.ended = True
            if "task_complete" in tr:
                self.task_complete = tr["task_complete"]

    def observe(self):
        reply = self.ctx.call("robomme_observe")
        meta, imgs = parse_reply(reply)
        self.meta, self.imgs = meta, imgs
        self._absorb(meta)
        return meta, imgs

    @property
    def pos(self):
        return self.state[0:3].copy()

    @property
    def Rm(self):
        return axang_to_R(self.state[3:6])

    @property
    def yaw(self):
        R = self.Rm
        return math.atan2(R[1, 0], R[0, 0])

    @property
    def opening(self):
        return float(self.state[6] + self.state[7])

    # ---------- action ----------
    def step(self, rows):
        if self.ended:
            raise EpisodeEnded()
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows]
        if self.remaining is not None:
            rows = rows[:max(1, self.remaining)]
        reply = self.ctx.call("robomme_step", actions=rows)
        meta, imgs = parse_reply(reply)
        self.last_step_meta = meta
        self._absorb(meta)
        if imgs:
            self.imgs = imgs; self.meta = meta
        if self.ended:
            raise EpisodeEnded()
        return meta

    def rot_err(self, yaw_target):
        """world-frame rotation vector taking current R to R_target (down-pointing, yaw)."""
        Rt = rotz(yaw_target) @ np.diag([1.0, -1.0, -1.0])
        E = Rt @ self.Rm.T
        return R_to_axang(E)

    def move_to(self, target, yaw=None, grip=None, speed=1.0, tol=0.004, max_steps=60,
                rot_tol=0.03, gain=1.0, settle=0):
        """Closed-loop per-step move of the TCP to world target (and optional yaw).
        Returns final position error (m)."""
        target = np.asarray(target, float)
        if grip is not None:
            self.grip_cmd = grip
        for it in range(max_steps):
            d = target - self.pos
            dist = np.linalg.norm(d)
            rv = self.rot_err(yaw) if yaw is not None else np.zeros(3)
            ang = np.linalg.norm(rv)
            if dist < tol and ang < rot_tol:
                break
            dp = d / POS_SCALE * gain
            nrm = np.linalg.norm(dp)
            if nrm > speed:
                dp = dp * (speed / nrm)
            dr = rv / ROT_SCALE * gain
            rn = np.linalg.norm(dr)
            if rn > 0.6:
                dr = dr * (0.6 / rn)
            self.step([list(dp) + list(dr) + [self.grip_cmd]])
        if settle:
            self.hold(settle)
        return float(np.linalg.norm(target - self.pos))

    def hold(self, n, grip=None):
        if grip is not None:
            self.grip_cmd = grip
        self.step([[0, 0, 0, 0, 0, 0, self.grip_cmd]] * n)

    def gripper(self, g, n=10):
        self.grip_cmd = g
        self.hold(n)


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
