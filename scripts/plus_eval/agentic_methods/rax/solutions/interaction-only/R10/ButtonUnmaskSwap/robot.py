import json, time
import numpy as np

def rotvec_to_R(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-9: return np.eye(3)
    k = v / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2; c = np.clip(c, -1, 1); th = np.arccos(c)
    if th < 1e-9: return np.zeros(3)
    if th > np.pi - 1e-4:
        # near pi
        w, V = np.linalg.eigh((R + R.T) / 2)
        k = V[:, np.argmax(w)]
        return k * th
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return k * th

R_DOWN = rotvec_to_R([-np.pi, 0, 0])

def rot_err(cur_rv, R_target=R_DOWN):
    """world-frame rotvec taking current to target"""
    return R_to_rotvec(R_target @ rotvec_to_R(cur_rv).T)

LOG = []

def packet(reply):
    if isinstance(reply, dict) and "content" in reply:
        blocks = reply["content"]
    else:
        blocks = reply
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                continue
    raise RuntimeError("no json packet")

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx
        self.log = log
        self.pkt = None
        self.frames = []   # list of (step_id, frame dict) saved frames
        self.ended = False
        self.task_complete = False
        self.grip = -1.0
        self.R_target = R_DOWN

    def observe(self):
        self.pkt = packet(self.ctx.call("robomme_observe"))
        return self.pkt

    @property
    def state(self):
        return np.array(self.pkt["state"])

    @property
    def step_id(self):
        return self.pkt["step_id"]

    def remaining(self):
        return self.pkt["episode_limits"]["remaining_episode_steps"]

    def step(self, rows):
        if self.ended:
            return self.pkt
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        p = packet(self.ctx.call("robomme_step", actions=rows))
        self.pkt = p
        tr = p.get("transition", {})
        for s in tr.get("steps", []):
            if s.get("task_complete"):
                self.task_complete = True
            if s.get("episode_ended"):
                self.ended = True
        if p.get("episode_ended"):
            self.ended = True
        fs = p.get("frame_sequence")
        if fs and fs.get("manifest_path"):
            self.frames.append(fs["manifest_path"])
        return p

    def move_to(self, target, grip=None, max_steps=60, tol=0.004, gain=1.0, vmax=1.0, chunk=4, on_chunk=None):
        """Closed-loop translation to world target (orientation held)."""
        if grip is not None:
            self.grip = grip
        target = np.array(target, float)
        used = 0
        while used < max_steps and not self.ended:
            cur = self.state[:3]
            err = target - cur
            if np.linalg.norm(err) < tol:
                break
            n = min(chunk, max_steps - used)
            rows = []
            # plan n rows assuming the controller roughly tracks
            pos = cur.copy()
            re_ = rot_err(self.state[3:6], self.R_target)
            for i in range(n):
                e = target - pos
                d = np.clip(e * gain / 0.05, -vmax, vmax)
                rr = np.clip(re_ / 0.5, -0.4, 0.4) if i == 0 else np.zeros(3)
                rows.append([d[0], d[1], d[2], rr[0], rr[1], rr[2], self.grip])
                pos = pos + d * 0.05
                if np.linalg.norm(target - pos) < tol * 0.5:
                    break
            self.step(rows)
            used += len(rows)
            if on_chunk:
                on_chunk(self)
        return np.linalg.norm(target - self.state[:3])

    def hold(self, n, grip=None):
        if grip is not None:
            self.grip = grip
        re_ = np.clip(rot_err(self.state[3:6], self.R_target) / 0.5, -0.4, 0.4)
        rows = [[0, 0, 0, 0, 0, 0, self.grip] for _ in range(n)]
        rows[0][3:6] = list(re_)
        return self.step(rows)

def pick_container(r, xy, z_grasp=0.058, z_safe=0.16, z_lift=0.20, log=print):
    x, y = float(xy[0]), float(xy[1])
    r.move_to([x, y, z_safe], grip=-1.0, max_steps=50, chunk=4)
    r.move_to([x, y, z_grasp + 0.03], grip=-1.0, max_steps=20, chunk=4, vmax=0.6)
    r.move_to([x, y, z_grasp], grip=-1.0, max_steps=15, chunk=3, vmax=0.3)
    r.hold(8, grip=1.0)
    r.move_to([x, y, z_lift], grip=1.0, max_steps=30, chunk=4, vmax=0.6)
    r.hold(3)
    fing = float(r.state[6] + r.state[7])
    log(f"pick at {x:.3f},{y:.3f}: ee={r.state[:3].round(3)} opening={fing:.4f} step={r.step_id}")
    return fing

def place_container(r, xy, z_place=0.062, log=print):
    x, y = float(xy[0]), float(xy[1])
    r.move_to([x, y, z_place + 0.04], grip=1.0, max_steps=40, chunk=4)
    r.move_to([x, y, z_place], grip=1.0, max_steps=15, chunk=3, vmax=0.3)
    r.hold(6, grip=-1.0)
    r.move_to([x, y, 0.16], grip=-1.0, max_steps=20, chunk=4)
    log(f"placed at {x:.3f},{y:.3f} step={r.step_id}")


def Rz(a):
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])

def yaw_err(r):
    e = rot_err(r.state[3:6], r.R_target)
    return float(np.linalg.norm(e))

def rotate_to(r, yaw, max_steps=12):
    """rotate gripper about world z so R = Rz(yaw) @ R_DOWN; hold position"""
    r.R_target = Rz(yaw) @ R_DOWN
    p = r.state[:3].copy()
    for _ in range(max_steps):
        e = rot_err(r.state[3:6], r.R_target)
        if np.linalg.norm(e) < 0.03 or r.ended:
            break
        rr = np.clip(e / 0.5, -0.6, 0.6)
        d = np.clip((p - r.state[:3]) / 0.05, -1, 1)
        r.step([[d[0], d[1], d[2], rr[0], rr[1], rr[2], r.grip]])
