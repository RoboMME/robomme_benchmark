"""Robot helpers for PickXtimes: observation decoding, geometry, motion."""
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
    meta = json.loads(blocks[0]["text"])
    return meta, blocks

def parse_obs(result):
    meta, blocks = _decode(result)
    rgb, depth = {}, {}
    for att in meta.get("attachments", []):
        b = blocks[att["content_index"]]
        data = base64.b64decode(b["data"])
        im = Image.open(io.BytesIO(data))
        if att["kind"] == "rgb":
            rgb[att["camera"]] = np.asarray(im.convert("RGB"))
        elif att["kind"] == "depth":
            arr = np.asarray(im).astype(np.float32)
            mpu = meta.get("depth", {}).get(att["camera"], {}).get("meters_per_unit", 0.001)
            depth[att["camera"]] = arr * mpu
    meta["rgb"], meta["depth_m"] = rgb, depth
    return meta

def observe(ctx):
    return parse_obs(ctx.call("robomme_observe"))

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def deproject(meta, cam, u, v, d=None):
    """Pixel (u=col, v=row) -> world xyz using depth (metres)."""
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    if d is None:
        D = meta["depth_m"][cam]
        r = 2
        vi, ui = int(round(v)), int(round(u))
        win = D[max(0, vi-r):vi+r+1, max(0, ui-r):ui+r+1]
        win = win[win > 0]
        if win.size == 0:
            return None
        d = float(np.median(win))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.array([(u-cx)*d/fx, -(v-cy)*d/fy, -d])
    R = quat_to_R(pose["quaternion_wxyz"])
    return R @ pc + np.array(pose["position_m"])

def deproject_mask(meta, cam, mask):
    """World points for every masked pixel with valid depth."""
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    D = meta["depth_m"][cam]
    vs, us = np.nonzero(mask & (D > 0))
    if len(vs) == 0:
        return np.zeros((0, 3))
    d = D[vs, us]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.stack([(us-cx)*d/fx, -(vs-cy)*d/fy, -d], 1)
    R = quat_to_R(pose["quaternion_wxyz"])
    return pc @ R.T + np.array(pose["position_m"])

def project(meta, cam, p):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"])
    pc = R.T @ (np.asarray(p) - np.array(pose["position_m"]))
    d = -pc[2]
    u = K[0, 0]*pc[0]/d + K[0, 2]
    v = -K[1, 1]*pc[1]/d + K[1, 2]
    return u, v

# ---------------- colour masks ----------------
def hsv(rgb):
    a = rgb.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1); mn = a.min(-1); df = mx - mn
    s = np.where(mx > 0, df / np.maximum(mx, 1e-6), 0)
    h = np.zeros_like(mx)
    m = df > 1e-6
    rr = m & (mx == r); gg = m & (mx == g) & ~rr; bb = m & ~rr & ~gg
    h[rr] = ((g - b)[rr] / df[rr]) % 6
    h[gg] = (b - r)[gg] / df[gg] + 2
    h[bb] = (r - g)[bb] / df[bb] + 4
    return h * 60.0, s, mx

def mask_green(rgb):
    h, s, v = hsv(rgb)
    return (h > 90) & (h < 150) & (s > 0.5) & (v > 0.25)

def mask_target(rgb):
    # purple/magenta rings of the target
    h, s, v = hsv(rgb)
    return (h > 260) & (h < 320) & (s > 0.25) & (v > 0.35)

def components(mask, min_px=3):
    """Simple 4-connected components; returns list of (count, (vs, us))."""
    try:
        import cv2
        n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        out = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_px:
                vs, us = np.nonzero(lab == i)
                out.append((len(vs), (vs, us)))
        out.sort(key=lambda t: -t[0])
        return out
    except ImportError:
        from scipy import ndimage
        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        out = []
        for i in range(1, n+1):
            vs, us = np.nonzero(lab == i)
            if len(vs) >= min_px:
                out.append((len(vs), (vs, us)))
        out.sort(key=lambda t: -t[0])
        return out

# ---------------- rotations ----------------
def rotvec_to_R(rv):
    rv = np.asarray(rv, float)
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
        # near pi: axis from diagonal
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.maximum(np.diag(A), 0))
        i = int(np.argmax(k))
        k = A[:, i] / max(k[i], 1e-9)
        return k / np.linalg.norm(k) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th

def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def yaw_of(rv):
    """Yaw of the gripper relative to the reset orientation Rx(pi)."""
    R = rotvec_to_R(rv) @ Rx(math.pi).T
    return math.atan2(R[1, 0], R[0, 0])

# ---------------- motion ----------------
class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = None
        self.ended = False
        self.complete = False
        self.step_id = 0
        self.remaining = None
        self.grip = -1.0
        self.on_obs = None

    def observe(self):
        self.meta = observe(self.ctx)
        self._limits(self.meta)
        self._hook()
        return self.meta

    def _limits(self, meta):
        lim = meta.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        if meta.get("episode_ended"):
            self.ended = True
        if "step_id" in meta:
            self.step_id = meta["step_id"]

    @property
    def state(self):
        return np.array(self.meta["state"])

    def step(self, rows):
        if self.ended:
            return None
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        res = self.ctx.call("robomme_step", actions=rows)
        meta = parse_obs(res)
        for s in meta.get("steps", []) or []:
            if isinstance(s, dict):
                if s.get("task_complete"):
                    self.complete = True
                if s.get("episode_ended"):
                    self.ended = True
        if meta.get("task_complete"):
            self.complete = True
        tr = meta.get("transition") or {}
        if tr.get("task_complete"):
            self.complete = True
        if tr.get("episode_ended"):
            self.ended = True
        for s in tr.get("steps", []) or []:
            if isinstance(s, dict) and s.get("task_complete"):
                self.complete = True
            if isinstance(s, dict) and s.get("episode_ended"):
                self.ended = True
        self.meta = meta
        self._limits(meta)
        self._hook()
        return meta

    def _hook(self):
        if self.on_obs is not None:
            try:
                self.on_obs(self.meta)
            except Exception as e:
                log("on_obs error: %r" % (e,))

    def move_to(self, target, grip=None, tol=0.004, max_steps=80, speed=1.0, chunk=4):
        """Closed-loop move of the TCP to target xyz, holding orientation down."""
        if grip is None:
            grip = self.grip
        self.grip = grip
        target = np.asarray(target, float)
        used = 0
        while used < max_steps and not self.ended:
            st = self.state
            err = target - st[:3]
            dist = np.linalg.norm(err)
            if dist < tol:
                break
            rows = []
            # plan a chunk assuming each row achieves its delta
            pos = st[:3].copy()
            n = min(chunk, max_steps - used)
            for _ in range(n):
                e = target - pos
                dd = e / 0.05
                m = np.max(np.abs(dd))
                if m > speed:
                    dd = dd * speed / m
                rows.append(list(dd) + list(self.rot_fix(st)) + [grip])
                pos = pos + dd * 0.05
                if np.linalg.norm(target - pos) < 1e-4:
                    break
            self.step(rows)
            used += len(rows)
        return np.linalg.norm(target - self.state[:3])

    yaw_d = 0.0

    def rot_fix(self, st, yaw_target=None):
        yd = self.yaw_d if yaw_target is None else yaw_target
        Rc = rotvec_to_R(st[3:6])
        Rd = Rz(yd) @ Rx(math.pi)
        rv = R_to_rotvec(Rd @ Rc.T)
        d = rv / 0.5
        m = np.max(np.abs(d))
        if m > 0.6:
            d = d * 0.6 / m
        return list(d)

    def yaw(self):
        return yaw_of(self.state[3:6])

    def rotate_to(self, yaw_d, tol=0.02, max_steps=30):
        self.yaw_d = yaw_d
        used = 0
        while used < max_steps and not self.ended:
            Rc = rotvec_to_R(self.state[3:6])
            Rd = Rz(yaw_d) @ Rx(math.pi)
            if np.linalg.norm(R_to_rotvec(Rd @ Rc.T)) < tol:
                break
            self.step([[0, 0, 0] + self.rot_fix(self.state) + [self.grip]] * 2)
            used += 2

    def gripper(self, g, n=8):
        self.grip = g
        st = self.state
        self.step([[0, 0, 0, 0, 0, 0, g]] * n)

    def hold(self, n):
        self.step([[0, 0, 0, 0, 0, 0, self.grip]] * n)
