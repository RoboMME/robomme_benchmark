"""Robot wrapper: every control step re-observes and updates the container tracker."""
import time, json, os, math
import numpy as np
from obs import decode, world_points
from core import band_grid, WorldTracker


def aa_to_R(a):
    a = np.asarray(a, float); th = np.linalg.norm(a)
    if th < 1e-9:
        return np.eye(3)
    k = a / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def R_to_aa(R):
    th = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if th < 1e-9:
        return np.zeros(3)
    if abs(th - math.pi) < 1e-4:
        # axis from diagonal
        k = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        i = int(np.argmax(k)); k[i] = k[i]
        for j in range(3):
            if j != i:
                k[j] = math.copysign(k[j], R[i, j] + R[j, i]) if k[i] > 0 else k[j]
        return k / np.linalg.norm(k) * th
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th)) * th


class Ended(Exception):
    pass


class Robot:
    def __init__(self, ctx, log_path="output/ctrl_log.jsonl"):
        self.ctx = ctx
        self.tracker = None
        self.ended = False
        self.task_complete = None
        self.log_f = open(log_path, "a")
        r = ctx.call("robomme_observe")
        self._ingest(r)
        self.track_ee_r = 0.075

    def log(self, **kw):
        kw["t"] = round(time.time(), 2)
        self.log_f.write(json.dumps(kw, default=lambda o: np.asarray(o).tolist()) + "\n")
        self.log_f.flush()

    def _ingest(self, r):
        meta, imgs = decode(r)
        self.meta, self.imgs = meta, imgs
        self.state = np.array(meta["state"], float)
        self.step_id = meta.get("step_id", 0)
        lim = meta.get("episode_limits", {})
        self.remaining = lim.get("remaining_episode_steps", 10 ** 6)
        tr = meta.get("transition") or {}
        for s in tr.get("steps", []) or []:
            if s.get("episode_ended"):
                self.ended = True
                self.task_complete = s.get("task_complete")
        if meta.get("episode_ended"):
            self.ended = True
        return meta, imgs

    @property
    def pos(self):
        return self.state[:3]

    def step(self, a):
        if self.ended:
            raise Ended()
        if self.remaining <= 1:
            raise Ended()
        a = [float(np.clip(x, -1, 1)) for x in a]
        r = self.ctx.call("robomme_step", actions=[a])
        self._ingest(r)
        if self.tracker is not None:
            G = band_grid(self.meta, self.imgs)
            ee = self.pos
            r_ex = self.track_ee_r if ee[2] < 0.3 else 0.0
            self.tracker.update(G, ee_xy=ee[:2], ee_r=r_ex)
        if self.ended:
            self.log(ev="ended", step=self.step_id, task_complete=self.task_complete)
            raise Ended()

    def yaw(self):
        R = aa_to_R(self.state[3:6])
        return math.atan2(R[1, 0], R[0, 0])

    def rot_err(self, target_yaw=None):
        """World-frame rotation vector taking current orientation to gripper-down with target yaw."""
        R = aa_to_R(self.state[3:6])
        yaw = self.yaw() if target_yaw is None else target_yaw
        c, s = math.cos(yaw), math.sin(yaw)
        # gripper-down reference: R0 = rot_x(pi); with yaw about world z
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        Rt = Rz @ np.diag([1.0, -1.0, -1.0])
        return R_to_aa(Rt @ R.T)

    def move_to(self, xyz, grip=-1.0, tol=0.006, max_steps=60, vmax=0.7, yaw=None, yaw_tol=0.03):
        xyz = np.asarray(xyz, float)
        for _ in range(max_steps):
            d = xyz - self.pos
            rv = self.rot_err(yaw)
            if np.linalg.norm(d) < tol and (yaw is None or np.linalg.norm(rv) < yaw_tol):
                return True
            a = np.clip(d / 0.05 * 0.9, -vmax, vmax)
            r = np.clip(rv / 0.5 * 0.9, -1.0, 1.0)
            self.step([a[0], a[1], a[2], r[0], r[1], r[2], grip])
        return False

    def hold(self, n, grip=-1.0):
        for _ in range(n):
            self.step([0, 0, 0, 0, 0, 0, grip])
