import base64, io, json, time
import numpy as np
from PIL import Image

LOG = []

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

def parse(result):
    blocks = result["content"]
    meta = json.loads(blocks[0]["text"])
    imgs = {}
    for att in meta.get("attachments", []):
        b = blocks[att["content_index"]]
        data = base64.b64decode(b["data"])
        im = Image.open(io.BytesIO(data))
        arr = np.array(im)
        imgs[(att["camera"], att["kind"])] = arr
    return meta, imgs

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def backproject(meta, imgs, cam, us, vs):
    """us, vs arrays of pixel coords -> Nx3 world points (nan if invalid)."""
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    d = imgs[(cam, "depth")].astype(np.float64) * meta["depth"][cam]["meters_per_unit"]
    us = np.asarray(us).astype(int); vs = np.asarray(vs).astype(int)
    dd = d[vs, us]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.stack([(us - cx) * dd / fx, -(vs - cy) * dd / fy, -dd], 1)
    pw = pc @ R.T + t
    pw[dd <= 0] = np.nan
    return pw

def world_cloud(meta, imgs, cam):
    H, W = imgs[(cam, "depth")].shape
    vs, us = np.mgrid[0:H, 0:W]
    return backproject(meta, imgs, cam, us.ravel(), vs.ravel()).reshape(H, W, 3)

def hsv(rgb):
    a = rgb[..., :3].astype(np.float64) / 255.0
    mx = a.max(-1); mn = a.min(-1); d = mx - mn + 1e-9
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    sat = (mx - mn) / (mx + 1e-9)
    return h, sat, mx

def color_masks(rgb):
    h, s, v = hsv(rgb)
    ok = (s > 0.55) & (v > 0.22)
    m = {}
    m["red"] = ok & ((h < 12) | (h > 340))
    m["green"] = ok & (h > 95) & (h < 150)
    m["blue"] = ok & (h > 215) & (h < 255)
    return m

def components(mask, min_px=4):
    from collections import deque
    H, W = mask.shape
    seen = np.zeros_like(mask, bool)
    comps = []
    for y in range(H):
        for x in range(W):
            if mask[y, x] and not seen[y, x]:
                q = deque([(y, x)]); seen[y, x] = True; pts = []
                while q:
                    cy, cx = q.popleft(); pts.append((cy, cx))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; q.append((ny, nx))
                if len(pts) >= min_px:
                    comps.append(np.array(pts))
    return comps

def detect_cubes(meta, imgs, cam="front"):
    """Return list of dict(color, px, world, n) for colored blobs."""
    rgb = imgs[(cam, "rgb")]
    cloud = world_cloud(meta, imgs, cam)
    out = []
    for color, mk in color_masks(rgb).items():
        for c in components(mk):
            pts = cloud[c[:, 0], c[:, 1]]
            pts = pts[np.isfinite(pts).all(1)]
            if len(pts) == 0: continue
            top = np.percentile(pts[:, 2], 90)
            tp = pts[pts[:, 2] > top - 0.008]
            w = np.median(tp, 0) if len(tp) >= 3 else np.median(pts, 0)
            out.append(dict(color=color, px=(float(c[:, 1].mean()), float(c[:, 0].mean())), n=len(c),
                            world=[float(w[0]), float(w[1]), float(w[2])], ztop=float(top)))
    return out

# ---------------- motion ----------------
def rotvec_to_R(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-9: return np.eye(3)
    k = v / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx

def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2; c = np.clip(c, -1, 1); th = np.arccos(c)
    if th < 1e-6: return np.zeros(3)
    if th > np.pi - 1e-4:
        # near pi
        w, V = np.linalg.eigh((R + np.eye(3)) / 2)
        k = V[:, np.argmax(w)]
        return k * th
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

def down_R(yaw):
    # gripper pointing down (rotvec [-pi,0,0] ~ diag(1,-1,-1)), then yaw about world z
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ np.diag([1.0, -1.0, -1.0])

class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = None; self.imgs = None
        self.ended = False; self.complete = False
        self.grip = -1.0
        self.yaw = 0.0
        self.steps_used = 0

    def observe(self):
        self.meta, self.imgs = parse(self.ctx.call("robomme_observe"))
        return self.meta

    @property
    def state(self):
        return np.array(self.meta["state"])

    def step(self, rows):
        res = self.ctx.call("robomme_step", actions=[list(map(float, r)) for r in rows])
        meta, imgs = parse(res)
        self.meta = meta; self.imgs = imgs
        self.steps_used += len(rows)
        for s in meta.get("steps", []) or []:
            if isinstance(s, dict):
                if s.get("task_complete"): self.complete = True
                if s.get("episode_ended"): self.ended = True
        if meta.get("episode_ended"): self.ended = True
        tr = meta.get("transition") or {}
        if tr.get("task_complete"): self.complete = True
        if tr.get("episode_ended"): self.ended = True
        if meta.get("task_complete"): self.complete = True
        return meta

    def rot_action(self, yaw):
        Rc = rotvec_to_R(self.state[3:6])
        err = R_to_rotvec(down_R(yaw) @ Rc.T)
        return np.clip(err / 0.5, -1, 1)

    def move_to(self, target, grip=None, tol=0.004, max_iter=40, chunk=1, gain=1.0, vmax=1.0, yaw=None):
        if grip is not None: self.grip = grip
        if yaw is not None: self.yaw = yaw
        target = np.asarray(target, float)
        for it in range(max_iter):
            if self.ended: return False
            p = self.state[:3]
            e = target - p
            rerr = np.linalg.norm(R_to_rotvec(down_R(self.yaw) @ rotvec_to_R(self.state[3:6]).T))
            if np.linalg.norm(e) < tol and rerr < 0.06:
                return True
            a = np.clip(gain * e / 0.05, -vmax, vmax)
            r = self.rot_action(self.yaw)
            row = list(a) + list(r) + [self.grip]
            self.step([row] * chunk)
        return False

    def gripper(self, g, n=8):
        self.grip = g
        p = self.state[:3]
        rows = []
        for i in range(n):
            rows.append([0, 0, 0] + list(self.rot_action(self.yaw)) + [g])
        self.step(rows)

# ---------------- scene perception ----------------
NUM = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"a":1,"an":1,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6}
COLORS = ["red", "blue", "green"]

def parse_instruction(text):
    import re
    t = text.lower()
    need = {}
    for m in re.finditer(r"\b(one|two|three|four|five|six|a|an|\d)\s+(red|blue|green)\s+cubes?\b", t):
        need[m.group(2)] = need.get(m.group(2), 0) + NUM[m.group(1)]
    return need

def find_bin(meta, imgs, cam="front", dark=20):
    rgb = imgs[(cam, "rgb")][..., :3].astype(int)
    cloud = world_cloud(meta, imgs, cam)
    z = cloud[..., 2]
    m = (rgb.max(2) < dark) & (z > 0.003) & (z < 0.045) & np.isfinite(z) & (cloud[..., 0] > -0.5)
    comps = components(m, 20)
    if not comps: return None
    c = max(comps, key=len)
    pts = cloud[c[:, 0], c[:, 1]]
    return dict(center=np.median(pts, 0), n=len(c), lo=pts.min(0), hi=pts.max(0))

def find_button(meta, imgs, cam="front", bin_xy=None, satmax=6, xmin=-0.5):
    rgb = imgs[(cam, "rgb")][..., :3].astype(int)
    cloud = world_cloud(meta, imgs, cam)
    z = cloud[..., 2]
    sat = rgb.max(2) - rgb.min(2)
    m = (sat < satmax) & (rgb.max(2) > 35) & (z > 0.004) & (z < 0.06) & np.isfinite(z) & (cloud[..., 0] > xmin)
    comps = components(m, 6)
    best = None
    for c in comps:
        pts = cloud[c[:, 0], c[:, 1]]
        ctr = np.median(pts, 0)
        if bin_xy is not None and np.linalg.norm(ctr[:2] - bin_xy[:2]) < 0.07: continue
        zt = float(np.percentile(pts[:, 2], 95))
        if len(c) > 1500 or zt < 0.022 or zt > 0.06: continue
        if np.ptp(pts[:, 0]) > 0.08 or np.ptp(pts[:, 1]) > 0.08: continue
        tp = pts[pts[:, 2] > zt - 0.006]
        if len(tp) >= 3: ctr = np.array([np.median(tp[:, 0]), np.median(tp[:, 1]), ctr[2]])
        if best is None or len(c) > best["n"]:
            best = dict(center=ctr, n=len(c), ztop=zt)
    return best

def cube_yaw(meta, imgs, cam, comp_px_center, color):
    """Estimate yaw (mod pi/2) of a cube from its top-face points."""
    rgb = imgs[(cam, "rgb")]
    cloud = world_cloud(meta, imgs, cam)
    mk = color_masks(rgb)[color]
    u, v = int(round(comp_px_center[0])), int(round(comp_px_center[1]))
    for c in components(mk, 4):
        if np.any((np.abs(c[:, 1] - u) <= 1) & (np.abs(c[:, 0] - v) <= 1)) or True:
            cy, cx = c[:, 0].mean(), c[:, 1].mean()
            if abs(cx - comp_px_center[0]) < 2 and abs(cy - comp_px_center[1]) < 2:
                pts = cloud[c[:, 0], c[:, 1]]
                pts = pts[np.isfinite(pts).all(1)]
                top = pts[pts[:, 2] > pts[:, 2].max() - 0.006][:, :2]
                if len(top) < 8: return 0.0
                top = top - top.mean(0)
                best, ba = None, 0.0
                for a in np.radians(np.arange(0, 90, 3)):
                    ca, sa = np.cos(a), np.sin(a)
                    p1 = top @ np.array([ca, sa]); p2 = top @ np.array([-sa, ca])
                    area = (np.ptp(p1)) * (np.ptp(p2))
                    if best is None or area < best: best, ba = area, a
                if ba > np.pi / 4: ba -= np.pi / 2
                return float(ba)
    return 0.0

def rect_yaw(pts2):
    top = pts2 - pts2.mean(0)
    best, ba = None, 0.0
    for a in np.radians(np.arange(0, 90, 2)):
        ca, sa = np.cos(a), np.sin(a)
        p1 = top @ np.array([ca, sa]); p2 = top @ np.array([-sa, ca])
        area = np.ptp(p1) * np.ptp(p2)
        if best is None or area < best: best, ba = area, a
    if ba > np.pi / 4: ba -= np.pi / 2
    return float(ba)

def find_bin_full(meta, imgs, cam="wrist"):
    rgb = imgs[(cam, "rgb")][..., :3].astype(int)
    cloud = world_cloud(meta, imgs, cam)
    z = cloud[..., 2]
    m = (rgb.max(2) < 20) & (z > 0.003) & (z < 0.045) & np.isfinite(z) & (cloud[..., 0] > -0.5)
    comps = components(m, 20)
    if not comps: return None
    c = max(comps, key=len)
    pts = cloud[c[:, 0], c[:, 1]]
    yaw = rect_yaw(pts[:, :2])
    ca, sa = np.cos(yaw), np.sin(yaw)
    rel = pts[:, :2] - pts[:, :2].mean(0)
    p1 = rel @ np.array([ca, sa]); p2 = rel @ np.array([-sa, ca])
    ctr = pts[:, :2].mean(0) + np.array([ca, sa]) * (p1.max() + p1.min()) / 2 + np.array([-sa, ca]) * (p2.max() + p2.min()) / 2
    return dict(center=np.array([ctr[0], ctr[1], float(np.median(pts[:, 2]))]), n=len(c), yaw=yaw, size=(float(np.ptp(p1)), float(np.ptp(p2))))

def track_step(R, p0, v, s0, z, grip, vmax=1.0, vz=0.6, gain=1.0):
    """One control step toward moving target p0 + v*(t - s0) at height z, with feedforward."""
    t = R.meta["step_id"] + 1
    tgt = np.array([p0[0] + v[0] * (t - s0), p0[1] + v[1] * (t - s0), z])
    p = R.state[:3]
    e = tgt - p
    a = gain * e / 0.05
    a[:2] += np.array(v[:2]) / 0.05
    a[:2] = np.clip(a[:2], -vmax, vmax); a[2] = np.clip(a[2], -vz, vz)
    r = R.rot_action(R.yaw)
    R.step([list(a) + list(r) + [grip]])
    t = R.meta["step_id"]
    tgt = np.array([p0[0] + v[0] * (t - s0), p0[1] + v[1] * (t - s0), z])
    return np.linalg.norm(tgt - R.state[:3]), np.linalg.norm(R_to_rotvec(down_R(R.yaw) @ rotvec_to_R(R.state[3:6]).T))
