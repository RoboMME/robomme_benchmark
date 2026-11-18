import json
import numpy as np

def rotvec_to_R(rv):
    rv = np.asarray(rv, float); th = np.linalg.norm(rv)
    if th < 1e-9: return np.eye(3)
    k = rv / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx

def R_to_rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(c)
    if th < 1e-9: return np.zeros(3)
    if th > np.pi - 1e-4:
        # near pi
        M = (R + np.eye(3)) / 2; i = np.argmax(np.diag(M)); k = M[:, i] / np.sqrt(M[i, i])
        return k * th
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

def Rz(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def parse(reply):
    if isinstance(reply, dict): reply = reply.get("content", [])
    for b in reply:
        if b.get("type") == "text":
            return json.loads(b["text"])
    return {}

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log
        self.R0 = None; self.last = None; self.ended = False; self.success = False

    def observe(self):
        raw = self.ctx.call("robomme_observe"); self.raw = raw
        d = parse(raw); self.last = d
        if self.R0 is None: self.R0 = np.diag([1.0, -1.0, -1.0])
        return d

    def state(self):
        return np.asarray(self.last["state"], float)

    def remaining(self):
        return self.last.get("episode_limits", {}).get("remaining_episode_steps", 0)

    def step(self, rows):
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        d = parse(self.ctx.call("robomme_step", actions=rows)); self.last = d
        if d.get("episode_ended"):
            self.ended = True
        for s in d.get("steps", []) or []:
            if isinstance(s, dict) and s.get("task_complete"): self.success = True
            if isinstance(s, dict) and s.get("episode_ended"): self.ended = True
        if d.get("task_complete"): self.success = True
        tr = d.get("transition") or {}
        if tr.get("task_complete"): self.success = True
        if tr.get("episode_ended"): self.ended = True
        for s in tr.get("steps", []) or []:
            if isinstance(s, dict) and s.get("task_complete"): self.success = True
            if isinstance(s, dict) and s.get("episode_ended"): self.ended = True
        return d

    def target_R(self, yaw):
        # yaw rotation about world z applied to the initial (downward) orientation
        return Rz(yaw) @ self.R0

    def move_to(self, xyz, yaw=0.0, grip=-1.0, tol=0.004, max_iter=12, chunk=None, speed=1.0):
        xyz = np.asarray(xyz, float); Rt = self.target_R(yaw)
        for it in range(max_iter):
            if self.ended: return False
            st = self.state(); p = st[:3]; Rc = rotvec_to_R(st[3:6])
            dp = xyz - p; drv = R_to_rotvec(Rt @ Rc.T)
            if np.linalg.norm(dp) < tol and np.linalg.norm(drv) < 0.03: return True
            n = int(np.ceil(max(np.max(np.abs(dp)) / (0.05 * speed), np.max(np.abs(drv)) / 0.4, 1)))
            n = min(n, 40, max(1, self.remaining() - 1))
            if self.remaining() <= 1: self.ended = True; return False
            # send n rows of equal deltas based on the current error; then re-measure
            a = dp / n / 0.05; r = drv / n / 0.5
            rows = [[a[0], a[1], a[2], r[0], r[1], r[2], grip]] * n
            self.step(rows)
        st = self.state()
        return np.linalg.norm(xyz - st[:3]) < 3 * tol

    def gripper(self, g, n=10):
        st = self.state()
        return self.step([[0, 0, 0, 0, 0, 0, g]] * n)

    def lift_container(self, x, y, yaw, grasp_z=0.047, lift_z=0.16, hover_z=0.15):
        """Approach from above, grasp, lift. Returns gripper opening after lift."""
        self.move_to([x, y, hover_z], yaw=yaw, grip=-1)
        self.move_to([x, y, grasp_z], yaw=yaw, grip=-1, speed=0.5)
        if self.ended: return 0
        self.gripper(1, 12)
        self.move_to([x, y, lift_z], yaw=yaw, grip=1, speed=0.6, tol=0.01)
        st = self.state(); return float(st[6] + st[7])

    def put_back(self, x, y, yaw, place_z=0.05, hover_z=0.15):
        self.move_to([x, y, place_z + 0.01], yaw=yaw, grip=1, speed=0.6)
        self.move_to([x, y, place_z], yaw=yaw, grip=1, speed=0.3, max_iter=4)
        self.gripper(-1, 8)
        self.move_to([x, y, hover_z], yaw=yaw, grip=-1, speed=0.8, tol=0.01)
