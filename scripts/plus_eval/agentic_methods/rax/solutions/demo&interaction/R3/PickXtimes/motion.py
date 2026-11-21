"""Closed-loop end-effector motion through robomme_step."""
import math, time
import numpy as np
from perception import decode

GAIN_MODEL = 0.4      # fraction of commanded delta realised per tick (measured)
MAX_CMD = 0.6         # max |action| per axis (0.6*0.05 = 3 cm per tick commanded)


def rotvec_to_R(rv):
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def R_to_rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        # near pi: axis from diagonal
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.maximum(np.diag(A), 0))
        i = int(np.argmax(k))
        k[i] = math.sqrt(A[i, i])
        for j in range(3):
            if j != i:
                k[j] = A[i, j] / k[i]
        return k / np.linalg.norm(k) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def down_R(yaw):
    """Gripper pointing down (rot pi about x) then yawed about world z."""
    Rx = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return Rz @ Rx


def gripper_yaw(state):
    R = rotvec_to_R(state[3:6])
    x = R[:, 0]
    return math.atan2(x[1], x[0])


class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx
        self.log = log
        self.meta = None
        self.imgs = None
        self.ended = False
        self.task_complete = False
        self.grip = -1.0
        self.yaw = 0.0
        self.listeners = []
        self.observe()

    def observe(self):
        r = self.ctx.call("robomme_observe")
        self.meta, self.imgs = decode(r)
        self.ended = bool(self.meta.get("episode_ended"))
        self._notify()
        return self.meta

    def _notify(self):
        for f in self.listeners:
            try:
                f(self.meta, self.imgs)
            except Exception as e:
                self.log(f"listener error {e!r}")

    @property
    def state(self):
        return np.array(self.meta["state"], float)

    @property
    def remaining(self):
        return self.meta["episode_limits"]["remaining_episode_steps"]

    def step(self, rows):
        if self.ended:
            return self.meta
        rows = [[float(max(-1, min(1, v))) for v in r] for r in rows]
        r = self.ctx.call("robomme_step", actions=rows)
        self.meta, self.imgs = decode(r)
        self._notify()
        tr = self.meta.get("transition", {})
        for s in tr.get("steps", []):
            if s.get("task_complete"):
                self.task_complete = True
            if s.get("episode_ended"):
                self.ended = True
        if self.meta.get("episode_ended"):
            self.ended = True
        if self.ended:
            self.log(f"EPISODE ENDED step={self.meta.get('step_id')} task_complete={self.task_complete} steps_tail={tr.get('steps', [])[-1:]}")
        return self.meta

    def hold(self, n, grip=None):
        if grip is not None:
            self.grip = grip
        return self.step([[0, 0, 0, 0, 0, 0, self.grip]] * n)

    def move_to(self, target, yaw=None, grip=None, tol=0.004, rtol=0.03, max_steps=120, chunk=5, max_cmd=MAX_CMD):
        """Servo the TCP to world `target` with gripper down at `yaw`."""
        if grip is not None:
            self.grip = grip
        if yaw is not None:
            self.yaw = yaw
        target = np.asarray(target, float)
        Rt = down_R(self.yaw)
        used = 0
        while used < max_steps and not self.ended:
            st = self.state
            pos = st[:3].copy()
            Rc = rotvec_to_R(st[3:6])
            err = target - pos
            rerr = R_to_rotvec(Rt @ Rc.T)
            if np.linalg.norm(err) < tol and np.linalg.norm(rerr) < rtol:
                return True
            rows = []
            n = min(chunk, max_steps - used)
            pe, re = err.copy(), rerr.copy()
            for _ in range(n):
                a = np.clip(pe / 0.05, -max_cmd, max_cmd)
                b = np.clip(re / 0.5, -0.6, 0.6)
                rows.append(list(a) + list(b) + [self.grip])
                pe = pe - GAIN_MODEL * a * 0.05
                re = re - GAIN_MODEL * b * 0.5
            self.step(rows)
            used += n
        st = self.state
        self.log(f"move_to not converged target={np.round(target,4)} pos={np.round(st[:3],4)}")
        return False
