"""BinFill helpers: packet decoding, RGB-D geometry, color segmentation, motion."""
import base64, io, json, math, time
import numpy as np
from PIL import Image

LOG = []

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

def decode(result):
    blocks = result["content"]
    meta = json.loads(blocks[0]["text"])
    imgs = {}
    for att in meta.get("attachments", []):
        b = blocks[att["content_index"]]
        data = base64.b64decode(b["data"])
        im = Image.open(io.BytesIO(data))
        if att["kind"] == "rgb":
            imgs[(att["camera"], "rgb")] = np.array(im.convert("RGB"))
        else:
            imgs[(att["camera"], "depth")] = np.array(im).astype(np.float32) * 0.001
    return meta, imgs

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

class Obs:
    def __init__(self, result):
        self.meta, self.imgs = decode(result)
        m = self.meta
        self.step = m.get("step_id")
        self.state = np.array(m["state"]) if "state" in m else None
        self.ended = m.get("episode_ended", False)
        self.cal = {}
        for cam in ("front", "wrist"):
            if "camera_intrinsics" in m and cam in m["camera_intrinsics"]:
                K = np.array(m["camera_intrinsics"][cam]["matrix"])
                P = m["camera_poses"][cam]
                self.cal[cam] = (K, quat_to_R(P["quaternion_wxyz"]), np.array(P["position_m"]))

    def rgb(self, cam):
        return self.imgs[(cam, "rgb")]

    def depth(self, cam):
        return self.imgs[(cam, "depth")]

    def world_map(self, cam):
        """HxWx3 world points for every pixel (nan where depth invalid)."""
        K, R, t = self.cal[cam]
        d = self.depth(cam)
        H, W = d.shape
        v, u = np.mgrid[0:H, 0:W].astype(np.float32)
        x = (u - K[0, 2]) * d / K[0, 0]
        y = -(v - K[1, 2]) * d / K[1, 1]
        z = -d
        pc = np.stack([x, y, z], -1).reshape(-1, 3)
        pw = pc @ R.T + t
        pw = pw.reshape(H, W, 3)
        pw[d <= 0] = np.nan
        return pw

def masks(rgb):
    r, g, b = [rgb[..., i].astype(np.int32) for i in range(3)]
    red = (r > 90) & (r > 2.2 * g) & (r > 2.2 * b)
    blue = (b > 70) & (b > 1.8 * r) & (b > 1.4 * g)
    green = (g > 70) & (g > 1.6 * r) & (g > 1.4 * b)
    yellow = (r > 120) & (g > 120) & (b < 0.45 * g) & (abs(r - g) < 0.25 * r)
    orange = (r > 140) & (g > 0.35 * r) & (g < 0.75 * r) & (b < 0.3 * r)
    cyan = (g > 100) & (b > 100) & (r < 0.45 * g) & (abs(g - b) < 0.3 * g)
    magenta = (r > 100) & (b > 100) & (g < 0.45 * r) & (abs(r - b) < 0.35 * r)
    return {"red": red, "blue": blue, "green": green, "yellow": yellow, "orange": orange, "cyan": cyan,
            "magenta": magenta, "purple": magenta}

def components(mask, min_px=4):
    """Simple connected components (4-neighbour) -> list of pixel index arrays."""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    out = []
    cur = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]:
            continue
        cur += 1
        stack = [(y0, x0)]
        lab[y0, x0] = cur
        pix = []
        while stack:
            y, x = stack.pop()
            pix.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = cur
                    stack.append((yy, xx))
        if len(pix) >= min_px:
            out.append(np.array(pix))
    return out

# ---------------- motion ----------------
def rotvec_to_R(rv):
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx

def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        # near pi
        w, V = np.linalg.eigh((R + np.eye(3)) / 2)
        k = V[:, np.argmax(w)]
        return k * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return v * th

def R_down(yaw=0.0):
    """Gripper pointing down (base orientation rotvec (-pi,0,0)) then yawed about world z."""
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return Rz @ rotvec_to_R(np.array([-math.pi, 0, 0]))

class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.obs = None
        self.ended = False
        self.complete = False
        self.grip = -1.0
        self.yaw = 0.0
        self.steps_used = 0
        self.remaining = None

    def observe(self):
        self.obs = Obs(self.ctx.call("robomme_observe"))
        lim = self.obs.meta.get("episode_limits") or {}
        self.remaining = lim.get("remaining_episode_steps", self.remaining)
        return self.obs

    def step(self, rows):
        if self.ended:
            return None
        r = self.ctx.call("robomme_step", actions=[list(map(float, x)) for x in rows])
        o = Obs(r)
        self.obs = o
        m = o.meta
        lim = m.get("episode_limits") or {}
        self.remaining = lim.get("remaining_episode_steps", self.remaining)
        tr = m.get("transition") or {}
        if tr.get("task_complete"):
            self.complete = True
        if tr.get("episode_ended"):
            self.ended = True
        steps = (m.get("steps") or []) + (tr.get("steps") or [])
        for s in steps:
            if s.get("task_complete"):
                self.complete = True
            if s.get("episode_ended"):
                self.ended = True
        if m.get("episode_ended"):
            self.ended = True
        if m.get("task_complete"):
            self.complete = True
        return o

    @property
    def pos(self):
        return self.obs.state[:3]

    def rot_cmd(self, cur_rv):
        Rt = R_down(self.yaw)
        Rc = rotvec_to_R(np.array(cur_rv))
        err = R_to_rotvec(Rt @ Rc.T)  # world-frame rotation from current to target
        return np.clip(err / 0.5, -0.6, 0.6)

    K_RESP = 0.37   # measured fraction of a commanded delta realised per step (ep3 free motion)

    def servo(self, target_fn, vff=(0, 0, 0), n_max=40, grip=None, tol=0.005, chunk=2, gain=0.5, min_steps=0):
        """Drive TCP to target_fn(step) (xyz) with velocity feed-forward vff (m/step), lag-aware.
        Stops when within tol (after min_steps) or n_max steps. Returns final error norm."""
        if grip is not None:
            self.grip = grip
        if self.obs is None:
            self.observe()
        vff = np.array(vff, float)
        k = 0
        err = 1.0
        while k < n_max and not self.ended:
            s0 = self.obs.step
            p = self.obs.state[:3].copy()
            err = np.linalg.norm(np.array(target_fn(s0)) - p)
            if k >= min_steps and err < tol:
                break
            rc = self.rot_cmd(self.obs.state[3:6])
            n = min(chunk, n_max - k)
            rows = []
            for j in range(n):
                s = s0 + j
                tgt = np.array(target_fn(s + 1), float)
                dm = vff + gain * (tgt - p) + (1 - gain) * 0.0
                du = np.clip(dm / (0.05 * self.K_RESP), -1, 1)
                p = p + du * 0.05 * self.K_RESP
                rows.append(list(du) + (list(rc) if j == 0 else [0, 0, 0]) + [self.grip])
            self.step(rows)
            k += n
        return err

    def move_to(self, target, tol=0.004, max_iter=12, speed=1.0, chunk=1, grip=None):
        target = np.array(target, float)
        n_max = int(max_iter * 6)
        e = self.servo(lambda s: target, n_max=n_max, grip=grip, tol=tol, chunk=3, gain=0.5 * speed)
        return e < tol * 3

    def hold(self, n, grip=None):
        if grip is not None:
            self.grip = grip
        return self.step([[0, 0, 0, 0, 0, 0, self.grip]] * n)
