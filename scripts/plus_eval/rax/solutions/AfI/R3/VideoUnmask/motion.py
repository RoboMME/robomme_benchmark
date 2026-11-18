"""Motion helpers: closed-loop Cartesian moves with official delta rows."""
import json
import time
import numpy as np

POS_SCALE = 0.05
ROT_SCALE = 0.5


def meta(reply):
    for b in reply.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                pass
    return {}


class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx
        self.log = log
        self.last = None
        self.ended = False
        self.steps_used = 0
        self.remaining = None
        self.gripper = -1.0

    def observe(self):
        m = meta(self.ctx.call("robomme_observe"))
        self._update(m)
        return m

    def _update(self, m):
        if not m:
            return
        if "state" in m:
            self.last = m
        lim = m.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        if m.get("episode_ended"):
            self.ended = True

    @property
    def state(self):
        return np.array(self.last["state"], float)

    @property
    def pos(self):
        return self.state[:3]

    def step(self, rows):
        if self.ended:
            return self.last
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        for i in range(0, len(rows), 100):
            m = meta(self.ctx.call("robomme_step", actions=rows[i:i + 100]))
            self._update(m)
            self.steps_used += len(rows[i:i + 100])
            if self.ended:
                break
        return self.last

    def hold(self, n, gripper=None):
        g = self.gripper if gripper is None else gripper
        self.gripper = g
        return self.step([[0, 0, 0, 0, 0, 0, g]] * n)

    def set_gripper(self, g, n=15):
        return self.hold(n, g)

    def move_to(self, target, speed=0.6, tol=0.004, max_iter=12, chunk=40, yaw_rate=None):
        """Move TCP to world target xyz keeping orientation. speed: fraction of 0.05 m/step."""
        target = np.asarray(target, float)
        for _ in range(max_iter):
            if self.ended:
                return False
            err = target - self.pos
            dist = np.linalg.norm(err)
            if dist < tol:
                return True
            n = int(np.ceil(dist / (POS_SCALE * speed)))
            n = max(1, min(chunk, n))
            rows = []
            rem = err.copy()
            rerr = tilt_error(self.state)
            rrow = np.clip(rerr / n / ROT_SCALE * 2.0, -0.3, 0.3) if np.linalg.norm(rerr) > 0.02 else np.zeros(3)
            for k in range(n):
                d = rem / max(1, n - k)
                rows.append([d[0] / POS_SCALE, d[1] / POS_SCALE, d[2] / POS_SCALE,
                             rrow[0], rrow[1], rrow[2], self.gripper])
                rem = rem - d
            self.step(rows)
        return np.linalg.norm(target - self.pos) < tol * 3


def rotvec_to_R(rv):
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def gripper_yaw(state):
    """Yaw (rad) of the EE x-axis projected on the world xy plane."""
    R = rotvec_to_R(state[3:6])
    x = R[:, 0]
    return float(np.arctan2(x[1], x[0]))


def wrap(a, period=2 * np.pi):
    return (a + period / 2) % period - period / 2


def _rotate_yaw(self, target_yaw, tol=np.radians(2.0), max_iter=10):
    """Closed-loop yaw rotation about world z (gripper pointing down)."""
    for _ in range(max_iter):
        if self.ended:
            return False
        err = wrap(target_yaw - gripper_yaw(self.state))
        if abs(err) < tol:
            return True
        n = int(np.clip(np.ceil(abs(err) / (ROT_SCALE * 0.4)), 1, 12))
        rows = [[0, 0, 0, 0, 0, float(np.clip(err / n / ROT_SCALE * 2.0, -0.6, 0.6)), self.gripper]] * n
        self.step(rows)
        self.hold(2)
    return abs(wrap(target_yaw - gripper_yaw(self.state))) < tol * 2


Robot.rotate_yaw = _rotate_yaw


def R_to_rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-6:
        return np.zeros(3)
    if np.pi - th < 1e-3:
        # near pi: axis from diagonal
        ax = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        i = int(np.argmax(ax))
        ax[(i + 1) % 3] *= np.sign(R[i, (i + 1) % 3]) if R[i, (i + 1) % 3] != 0 else 1
        ax[(i + 2) % 3] *= np.sign(R[i, (i + 2) % 3]) if R[i, (i + 2) % 3] != 0 else 1
        return ax / np.linalg.norm(ax) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def tilt_error(state):
    """World-frame rotation vector taking the current EE frame to pointing straight down, same yaw."""
    R = rotvec_to_R(state[3:6])
    yaw = gripper_yaw(state)
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    Rdes = Rz @ np.diag([1.0, -1.0, -1.0])
    return R_to_rotvec(Rdes @ R.T)
