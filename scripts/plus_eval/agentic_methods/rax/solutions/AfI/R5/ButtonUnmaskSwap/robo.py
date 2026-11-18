"""Robot helpers: observation decoding, back-projection, closed-loop motion."""
import base64, io, json, math, time
import numpy as np
from PIL import Image

LOG = []

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

def _decode(result):
    blocks = result.get("content", [])
    meta = json.loads(next(b["text"] for b in blocks if b.get("type") == "text"))
    imgs = {}
    for att in meta.get("attachments", []):
        b = blocks[att["content_index"]]
        data = base64.b64decode(b["data"])
        im = Image.open(io.BytesIO(data))
        if att["kind"] == "rgb":
            imgs[(att["camera"], "rgb")] = np.array(im.convert("RGB"))
        else:
            imgs[(att["camera"], "depth")] = np.array(im).astype(np.float32) * 0.001
    meta["_imgs"] = imgs
    return meta

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

class Obs:
    def __init__(self, meta):
        self.meta = meta
        self.step_id = meta.get("step_id")
        self.state = np.array(meta["state"], dtype=float)
        self.ended = bool(meta.get("episode_ended"))
        self.rgb = {c: meta["_imgs"].get((c, "rgb")) for c in ("front", "wrist")}
        self.depth = {c: meta["_imgs"].get((c, "depth")) for c in ("front", "wrist")}
        self.K = {c: np.array(meta["camera_intrinsics"][c]["matrix"]) for c in meta.get("camera_intrinsics", {})}
        self.pose = {c: (np.array(p["position_m"]), quat_to_R(p["quaternion_wxyz"])) for c, p in meta.get("camera_poses", {}).items()}
        self.limits = meta.get("episode_limits", {})
        self.complete = False
        tr = meta.get("transition") or {}
        for s in tr.get("steps", []) or []:
            if s.get("task_complete"):
                self.complete = True

    def world_map(self, cam):
        """Per-pixel world xyz (H,W,3) and valid mask."""
        d = self.depth[cam]
        K = self.K[cam]; pos, R = self.pose[cam]
        H, W = d.shape
        u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pc = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1)
        pw = pc @ R.T + pos
        return pw, d > 0

    def project(self, cam, pw):
        pos, R = self.pose[cam]; K = self.K[cam]
        pc = (np.asarray(pw) - pos) @ R
        d = -pc[..., 2]
        u = K[0, 0] * pc[..., 0] / d + K[0, 2]
        v = -K[1, 1] * pc[..., 1] / d + K[1, 2]
        return u, v

def observe(ctx):
    return Obs(_decode(ctx.call("robomme_observe")))

def step(ctx, rows):
    rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
    meta = _decode(ctx.call("robomme_step", actions=rows))
    return Obs(meta)

def rotvec_to_R(rv):
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx

def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        # near pi
        M = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-9))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th

class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.obs = observe(ctx)
        self.grip = -1.0
        self.R0 = rotvec_to_R(self.obs.state[3:6])  # keep initial orientation
        self.yaw_target = None
        self.steps_used = 0

    @property
    def pos(self):
        return self.obs.state[:3]

    def rot_cmd(self, Rt):
        Rc = rotvec_to_R(self.obs.state[3:6])
        err = R_to_rotvec(Rt @ Rc.T)  # world frame
        return np.clip(err / 0.5, -1, 1)

    def rows_for(self, target, n, Rt=None, gain=1.0, vmax=1.0):
        """Open-loop-ish: plan n rows from current pos to target (one call)."""
        p = self.pos.copy()
        rows = []
        d = np.asarray(target) - p
        for i in range(n):
            rem = n - i
            dd = d / rem if rem > 0 else d
            r = list(np.clip(dd / 0.05 * gain, -vmax, vmax)) + [0, 0, 0] + [self.grip]
            rows.append(r)
            d = d - np.array(r[:3]) * 0.05
        return rows

    def do(self, rows):
        if not rows:
            return self.obs
        self.obs = step(self.ctx, rows)
        self.steps_used += len(rows)
        return self.obs

    def move_to(self, target, tol=0.005, max_iter=12, vmax=0.6, Rt=None, chunk=None):
        """Closed loop: chunks of rows toward target, re-measure after each chunk."""
        target = np.asarray(target, float)
        if Rt is None:
            Rt = self.R0
        for it in range(max_iter):
            if self.obs.ended:
                return False
            e = target - self.pos
            dist = np.linalg.norm(e)
            rot = self.rot_cmd(Rt)
            if dist < tol and np.linalg.norm(rot) < 0.04:
                return True
            nsteps = int(max(1, math.ceil(dist / (0.05 * vmax))))
            if chunk:
                nsteps = min(nsteps, chunk)
            rows = []
            p = self.pos.copy()
            for i in range(nsteps):
                d = (target - p)
                v = np.clip(d / 0.05, -vmax, vmax)
                r = list(v) + list(rot if i == 0 else np.zeros(3)) + [self.grip]
                rows.append(r)
                p = p + v * 0.05
            # extra settle row
            rows.append([0, 0, 0, 0, 0, 0, self.grip])
            self.do(rows)
        return np.linalg.norm(target - self.pos) < tol * 3

    def set_grip(self, g, n=8):
        self.grip = g
        rows = [[0, 0, 0, 0, 0, 0, g]] * n
        return self.do(rows)

    def hold(self, n):
        return self.do([[0, 0, 0, 0, 0, 0, self.grip]] * n)
