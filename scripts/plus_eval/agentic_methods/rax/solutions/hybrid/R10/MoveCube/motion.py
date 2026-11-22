"""Reusable motion helpers for the Panda via robomme_step (delta rows)."""
import json, math
import numpy as np

def packet(reply):
    if isinstance(reply, dict):
        reply = reply.get("content", [])
    for b in reply:
        if isinstance(b, dict) and b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                pass
    raise RuntimeError("no json block")

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx
        self.log = log
        self.ended = False
        self.complete = False
        self.state = None
        self.step_id = 0
        self.remaining = None
        self.gripper = -1.0
        self.rot_hold = None

    def observe(self):
        p = packet(self.ctx.call("robomme_observe"))
        self._upd(p)
        return p

    def _upd(self, p):
        if "state" in p and p["state"] is not None:
            self.state = np.array(p["state"], dtype=float)
        self.step_id = p.get("step_id", self.step_id)
        lim = p.get("episode_limits") or {}
        self.remaining = lim.get("remaining_episode_steps", self.remaining)
        if p.get("episode_ended"):
            self.ended = True
        for s in p.get("steps", []) or []:
            if s.get("task_complete"):
                self.complete = True
            if s.get("episode_ended"):
                self.ended = True
        if p.get("task_complete"):
            self.complete = True

    def step(self, rows):
        if self.ended:
            return None
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        p = packet(self.ctx.call("robomme_step", actions=rows))
        self._upd(p)
        return p

    def pos(self):
        return self.state[:3].copy()

    def hold(self, n, grip=None):
        g = self.gripper if grip is None else grip
        self.gripper = g
        for i in range(0, n, 50):
            self.step([[0, 0, 0, 0, 0, 0, g]] * min(50, n - i))

    def set_gripper(self, g, n=12):
        self.hold(n, g)

    def move_to(self, target, tol=0.004, max_steps=200, speed=0.6, chunk=4, grip=None,
                rot_target=None, stall_steps=24):
        """Closed-loop move of the TCP to target (world m). speed: max fraction of 0.05 m per step.
        rot_target: axis-angle to servo (defaults to self.rot_hold if set)."""
        target = np.asarray(target, dtype=float)
        g = self.gripper if grip is None else grip
        self.gripper = g
        if rot_target is None:
            rot_target = getattr(self, "rot_hold", None)
        used = 0
        best = 1e9; best_at = 0
        while used < max_steps and not self.ended:
            err = target - self.pos()
            d = np.linalg.norm(err)
            rerr = np.zeros(3)
            if rot_target is not None:
                rerr = rot_error(self.state[3:6], rot_target)
            if d < tol and np.linalg.norm(rerr) < 0.05:
                return True
            if d < best - 0.002:
                best = d; best_at = used
            elif used - best_at > stall_steps:
                self.log("move_to stall at", np.round(self.pos(), 3).tolist(), "target", np.round(target, 3).tolist())
                return False
            n = chunk if d > 0.02 else 2
            step = err / max(1, n)
            mag = np.linalg.norm(step)
            lim = speed * 0.05
            if mag > lim:
                step = step * lim / mag
            rstep = np.clip(rerr / 0.5 * 0.5, -0.3, 0.3)
            rows = [list(step / 0.05) + list(rstep) + [g] for _ in range(n)]
            self.step(rows)
            used += n
        return np.linalg.norm(target - self.pos()) < 3 * tol

def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def aa_to_mat(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K

def mat_to_aa(R):
    c = (np.trace(R) - 1) / 2
    c = min(1, max(-1, c)); th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if abs(th - math.pi) < 1e-3:
        # axis from diagonal
        xx = math.sqrt(max(0, (R[0, 0] + 1) / 2)); yy = math.sqrt(max(0, (R[1, 1] + 1) / 2)); zz = math.sqrt(max(0, (R[2, 2] + 1) / 2))
        ax = np.array([xx, yy if R[0, 1] >= 0 else -yy, zz if R[0, 2] >= 0 else -zz])
        return ax / np.linalg.norm(ax) * th
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return ax * th

def rot_error(cur_aa, tgt_aa):
    """world-frame rotation vector taking current to target."""
    Rc = aa_to_mat(cur_aa); Rt = aa_to_mat(tgt_aa)
    return mat_to_aa(Rt @ Rc.T)

def down_yaw(yaw):
    """gripper pointing down with yaw rotation about world z (yaw=0 equals reset pose)."""
    R0 = aa_to_mat([-math.pi, 0, 0])
    Rz = aa_to_mat([0, 0, yaw])
    return mat_to_aa(Rz @ R0)
