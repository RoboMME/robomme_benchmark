"""Motion helpers for RoboMME-Plus Panda: closed-loop pose tracking with delta actions."""
import json, time
import numpy as np

POS_SCALE = 0.05   # m per unit action
ROT_SCALE = 0.5    # rad per unit action


def rotvec_to_mat(r):
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def mat_to_rotvec(R):
    R = np.asarray(R, float)
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c))
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > np.pi - 1e-4:
        # near pi: axis from R + I
        B = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(B)))
        k = B[:, i] / np.sqrt(max(B[i, i], 1e-12))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def rot_err(r_tgt, r_cur):
    """World-frame rotation vector taking current orientation to target."""
    return mat_to_rotvec(rotvec_to_mat(r_tgt) @ rotvec_to_mat(r_cur).T)


def parse(reply):
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
        self.state = None
        self.ended = False
        self.complete = False
        self.step_id = 0
        self.remaining = None
        self.last = None
        self.calls = 0
        self.ik_errors = 0
        self.ik_flag = False      # set when the last step call hit a native IK penalty

    def observe(self):
        d = parse(self.ctx.call("robomme_observe"))
        self._update(d)
        return d

    def _update(self, d):
        self.last = d
        if "state" in d and d["state"] is not None:
            self.state = np.array(d["state"], float)
        if "step_id" in d:
            self.step_id = d["step_id"]
        lim = d.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        if d.get("episode_ended"):
            self.ended = True
        tr = d.get("transition") or {}
        if tr.get("task_complete"):
            self.complete = True
        if tr.get("episode_ended"):
            self.ended = True
        for s in (d.get("steps", []) or []) + (tr.get("steps", []) or []):
            if isinstance(s, dict):
                if s.get("task_complete"):
                    self.complete = True
                if s.get("episode_ended"):
                    self.ended = True
        if d.get("task_complete"):
            self.complete = True

    def step(self, rows):
        if self.ended:
            return None
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        s0 = self.step_id
        d = parse(self.ctx.call("robomme_step", actions=rows))
        self.calls += 1
        self._update(d)
        self.ik_flag = False
        try:
            if self.step_id - s0 > len(rows):
                # recoverable native IK error: step clock jumped, pose did not move
                self.ik_flag = True
                self.ik_errors += 1
                self.log("IK PENALTY: step %s -> %s for %d rows (errors %d)" % (s0, self.step_id, len(rows), self.ik_errors))
        except Exception:
            pass
        return d

    @property
    def pos(self):
        return self.state[0:3].copy()

    @property
    def rot(self):
        return self.state[3:6].copy()

    @property
    def width(self):
        return float(self.state[6] + self.state[7])

    def row_toward(self, p_tgt, r_tgt, grip, max_lin=0.02, max_ang=0.15, gain=1.0):
        dp = (np.asarray(p_tgt) - self.pos) * gain
        n = np.linalg.norm(dp)
        if n > max_lin:
            dp *= max_lin / n
        dr = rot_err(r_tgt, self.rot) * gain
        a = np.linalg.norm(dr)
        if a > max_ang:
            dr *= max_ang / a
        return list(dp / POS_SCALE) + list(dr / ROT_SCALE) + [grip]

    def goto(self, p_tgt, r_tgt, grip, tol_p=0.002, tol_r=0.02, max_steps=60,
             max_lin=0.02, max_ang=0.15, gain=1.0, settle=0):
        """Closed loop, one row per call. Returns (converged, steps)."""
        n = 0
        while n < max_steps and not self.ended:
            ep = np.linalg.norm(np.asarray(p_tgt) - self.pos)
            er = np.linalg.norm(rot_err(r_tgt, self.rot))
            if ep < tol_p and er < tol_r:
                break
            self.step([self.row_toward(p_tgt, r_tgt, grip, max_lin, max_ang, gain)])
            n += 1
            if self.ik_flag:
                break
        ep = np.linalg.norm(np.asarray(p_tgt) - self.pos)
        er = np.linalg.norm(rot_err(r_tgt, self.rot))
        if self.ik_flag:
            return False, n, ep, er
        for _ in range(settle):
            if self.ended:
                break
            self.step([self.row_toward(p_tgt, r_tgt, grip, max_lin, max_ang, gain)])
            n += 1
        return (ep < tol_p and er < tol_r), n, ep, er

    def hold(self, grip, n):
        """Hold the current pose for n steps with gripper command (one batched call)."""
        if self.ended or n <= 0:
            return
        p, r = self.pos, self.rot
        rows = []
        for _ in range(n):
            rows.append([0, 0, 0, 0, 0, 0, grip])
        self.step(rows)
