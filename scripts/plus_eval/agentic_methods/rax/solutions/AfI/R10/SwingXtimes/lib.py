"""Helpers for SwingXtimes controller: packet decoding, perception, motion."""
import base64, io, json, math, time
import numpy as np
from PIL import Image

LOG = []

def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)

# ---------------- packet decoding ----------------
def parse(reply):
    blocks = reply.get("content", [])
    meta = None
    for b in blocks:
        if b.get("type") == "text":
            try:
                meta = json.loads(b["text"]); break
            except Exception:
                pass
    return meta, blocks

def get_images(meta, blocks):
    out = {}
    for a in meta.get("attachments", []) or []:
        i = a.get("content_index")
        if i is None or i >= len(blocks):
            continue
        b = blocks[i]
        data = b.get("data")
        if data is None:
            continue
        im = Image.open(io.BytesIO(base64.b64decode(data)))
        if a["kind"] == "rgb":
            out[(a["camera"], "rgb")] = np.array(im.convert("RGB")).astype(np.int32)
        else:
            arr = np.array(im).astype(np.float64)
            mpu = (meta.get("depth", {}).get(a["camera"], {}) or {}).get("meters_per_unit", 0.001)
            out[(a["camera"], "depth")] = arr * mpu
    return out

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def backproject(meta, cam, depth):
    """Return HxWx3 world points (nan where invalid)."""
    K = np.array(meta["camera_intrinsics"][cam]["matrix"], dtype=float)
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"], dtype=float)
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W].astype(float)
    d = depth
    pc = np.stack([(u-K[0, 2])*d/K[0, 0], -(v-K[1, 2])*d/K[1, 1], -d], -1)
    pw = pc @ R.T + t
    pw[d <= 0] = np.nan
    return pw

def label(mask):
    """4-connected component labelling (pure numpy/python)."""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32); n = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]:
            continue
        n += 1; stack = [(y0, x0)]; lab[y0, x0] = n
        while stack:
            y, x = stack.pop()
            for yy, xx in ((y+1, x), (y-1, x), (y, x+1), (y, x-1)):
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n; stack.append((yy, xx))
    return lab, n

# ---------------- perception ----------------
def perceive(meta, imgs, cam="front"):
    rgb = imgs[(cam, "rgb")]; dep = imgs[(cam, "depth")]
    P = backproject(meta, cam, dep)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    valid = np.isfinite(P[..., 2])
    z = np.where(valid, P[..., 2], 9.0)
    x = np.where(valid, P[..., 0], 9.0)
    table = valid & (z < 0.08) & (z > -0.02) & (x > -0.45) & (x < 0.25)
    res = {}
    red = table & color_mask(rgb) & (z > 0.022)
    lab, n = label(red)
    best = None
    for k in range(1, n+1):
        m = lab == k
        if m.sum() < 8:
            continue
        if best is None or m.sum() > best.sum():
            best = m
    if best is not None:
        pts = P[best]
        top = np.nanpercentile(pts[:, 2], 80)
        res["red"] = {"xy": np.nanmedian(pts[:, :2], 0), "top": float(top), "n": int(best.sum()),
                      "pix": np.argwhere(best).mean(0)[::-1]}
    grey = table & (np.abs(r-g) < 12) & (np.abs(g-b) < 12) & (r > 80)
    lab, n = label(grey)
    clusters = []
    for k in range(1, n+1):
        m = lab == k
        if m.sum() < 10:
            continue
        pts = P[m]
        clusters.append({"xy": np.nanmedian(pts[:, :2], 0), "ztop": float(np.nanpercentile(pts[:, 2], 90)),
                         "n": int(m.sum()), "pix": np.argwhere(m).mean(0)[::-1]})
    # merge clusters whose xy are within 3 cm (rings of the same target)
    merged = []
    for c in sorted(clusters, key=lambda c: -c["n"]):
        for m in merged:
            if np.linalg.norm(m["xy"] - c["xy"]) < 0.035:
                m["n"] += c["n"]; m["ztop"] = max(m["ztop"], c["ztop"]); break
        else:
            merged.append(dict(c))
    res["grey"] = merged
    targets = [c for c in merged if c["ztop"] < 0.018 and c["n"] >= 20]
    buttons = [c for c in merged if c["ztop"] >= 0.018 and c["n"] >= 10]
    targets = sorted(targets, key=lambda c: -c["n"])[:2]
    res["targets"] = sorted(targets, key=lambda c: c["xy"][1])  # ascending y
    res["button"] = max(buttons, key=lambda c: c["n"]) if buttons else None
    for k in ("green", "blue"):
        pass
    return res

# ---------------- rotation utils ----------------
def rotvec_to_R(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v/th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th)*Kx + (1-math.cos(th))*Kx@Kx

def R_to_rotvec(R):
    c = (np.trace(R)-1)/2; c = max(-1.0, min(1.0, c)); th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi
        M = (R + np.eye(3))/2
        k = np.sqrt(np.maximum(np.diag(M), 0))
        i = int(np.argmax(k)); k = M[:, i]/math.sqrt(max(M[i, i], 1e-12))
        return k*th
    w = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])/(2*math.sin(th))
    return w*th

def down_R(yaw):
    """Gripper pointing down with given yaw (about world z)."""
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return Rz @ rotvec_to_R([math.pi, 0, 0])

# ---------------- robot I/O ----------------
class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = None; self.blocks = None
        self.ended = False; self.complete = False
        self.remaining = None
        self.grip = -1.0

    def observe(self):
        meta, blocks = parse(self.ctx.call("robomme_observe"))
        self._update(meta, blocks)
        return meta, blocks

    def _update(self, meta, blocks):
        self.meta, self.blocks = meta, blocks
        if meta.get("episode_ended"):
            self.ended = True
        lim = meta.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        for s in meta.get("steps", []) or []:
            if isinstance(s, dict):
                if s.get("episode_ended"):
                    self.ended = True
                if s.get("task_complete"):
                    self.complete = True
        if meta.get("task_complete"):
            self.complete = True
        tr = meta.get("transition") or {}
        if isinstance(tr, dict):
            if tr.get("episode_ended"):
                self.ended = True
            if tr.get("task_complete"):
                self.complete = True
            for s_ in tr.get("steps", []) or []:
                if isinstance(s_, dict) and s_.get("task_complete"):
                    self.complete = True

    @property
    def state(self):
        return np.array(self.meta["state"], float)

    def step(self, rows):
        if self.ended:
            return False
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows]
        if self.remaining is not None:
            rows = rows[:max(0, self.remaining)]
        if not rows:
            return False
        meta, blocks = parse(self.ctx.call("robomme_step", actions=rows))
        self._update(meta, blocks)
        return not self.ended

    def images(self):
        return get_images(self.meta, self.blocks)

    def rot_err(self, Rt):
        Rc = rotvec_to_R(self.state[3:6])
        return R_to_rotvec(Rt @ Rc.T)

    def move_to(self, target, Rt=None, grip=None, tol=0.004, max_steps=60, vmax=0.8, chunk=6, rot_tol=0.03):
        """Closed-loop move of the TCP to target xyz (and orientation Rt)."""
        if grip is not None:
            self.grip = grip
        target = np.asarray(target, float)
        used = 0
        while used < max_steps and not self.ended:
            s = self.state
            d = target - s[:3]
            re = self.rot_err(Rt) if Rt is not None else np.zeros(3)
            if np.linalg.norm(d) < tol and np.linalg.norm(re) < rot_tol:
                return True
            dist = np.linalg.norm(d)
            n = int(min(chunk, max(1, math.ceil(dist / (0.05*vmax)))))
            # per-row command: remaining/n, capped
            dp = d / max(1, math.ceil(dist/(0.05*vmax))) if dist > 0 else d
            dp_cmd = dp / 0.05
            dr_cmd = (re / max(1, n)) / 0.5 if Rt is not None else np.zeros(3)
            dr_cmd = np.clip(dr_cmd, -0.6, 0.6)
            k = n
            if dist < 0.02:
                k = 1; dp_cmd = d / 0.05
                dr_cmd = np.clip(re/0.5, -0.6, 0.6) if Rt is not None else dr_cmd
            row = list(np.clip(dp_cmd, -vmax, vmax)) + list(dr_cmd) + [self.grip]
            self.step([row]*k)
            used += k
        return False

    def hold(self, n, grip=None):
        if grip is not None:
            self.grip = grip
        return self.step([[0, 0, 0, 0, 0, 0, self.grip]]*n)

COLOR = {"name": "red"}

def color_mask(rgb, name=None):
    name = name or COLOR["name"]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if name == "green":
        return (g > 90) & (r < 80) & (b < 80) & (g - r > 60)
    if name == "blue":
        return (b > 90) & (r < 80) & (g < 80) & (b - r > 60)
    if name == "yellow":
        return (r > 140) & (g > 140) & (b < 80)
    if name in ("purple", "magenta", "pink"):
        return (r > 90) & (b > 90) & (g < 80)
    if name == "cyan":
        return (g > 90) & (b > 90) & (r < 80)
    if name == "orange":
        return (r > 180) & (g > 70) & (g < 150) & (b < 40) & (r - g > 80)
    return (r > 100) & (g < 70) & (b < 70) & (r - g > 80)

def red_top_points(meta, imgs, cam):
    rgb = imgs[(cam, "rgb")]; P = backproject(meta, cam, imgs[(cam, "depth")])
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    valid = np.isfinite(P[..., 2])
    zz = np.where(valid, P[..., 2], 9.0)
    m = valid & color_mask(rgb) & (zz > 0.022) & (zz < 0.12)
    if cam == "front":
        z = np.where(valid, P[..., 2], 9.0); x = np.where(valid, P[..., 0], 9.0)
        m &= (z < 0.08) & (x > -0.45) & (x < 0.25)
    lab, n = label(m)
    if n == 0:
        return None
    sizes = [(lab == k).sum() for k in range(1, n+1)]
    k = int(np.argmax(sizes)) + 1
    if sizes[k-1] < 8:
        return None
    pts = P[lab == k]
    zt = np.nanmax(pts[:, 2])
    top = pts[pts[:, 2] > zt - 0.006]
    return pts, top

def square_yaw(xy):
    """Yaw (rad, in [-pi/4, pi/4)) of the min-area bounding square of 2D points."""
    best = None
    c = xy.mean(0)
    for deg in np.arange(-45, 45, 1.0):
        a = math.radians(deg)
        R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        q = (xy - c) @ R.T
        ext = q.max(0) - q.min(0)
        area = ext[0]*ext[1]
        if best is None or area < best[0]:
            best = (area, a, ext, (q.max(0)+q.min(0))/2 @ R + c)
    return best[1], best[2], best[3]

def visit(r, xy, Rt, z_travel=0.09, z_touch=0.032, dwell=2):
    """Carry held cube over xy, lower to touch, lift."""
    s = r.state
    if s[2] < z_travel - 0.01:
        r.move_to([s[0], s[1], z_travel], Rt, tol=0.01, max_steps=20)
    r.move_to([xy[0], xy[1], z_travel], Rt, tol=0.008, max_steps=40)
    r.move_to([xy[0], xy[1], z_touch], Rt, tol=0.004, max_steps=30)
    if dwell:
        r.hold(dwell)
    r.move_to([xy[0], xy[1], z_travel], Rt, tol=0.01, max_steps=20)


def flat_discs(meta, imgs, cam):
    """Flat target-disc candidates: greyish pixels in a thin height band above the table."""
    rgb = imgs[(cam, "rgb")]; P = backproject(meta, cam, imgs[(cam, "depth")])
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    valid = np.isfinite(P[..., 2])
    z = np.where(valid, P[..., 2], 9.0); x = np.where(valid, P[..., 0], 9.0)
    grey = (np.abs(r-g) < 14) & (np.abs(g-b) < 14) & (r > 80)
    sat = (np.max(rgb, -1) > 120) & (np.min(rgb, -1) < 45) & ((np.abs(r-g) > 100) | (np.abs(g-b) > 100) | (np.abs(r-b) > 100))
    m = valid & (z > 0.003) & (z < 0.017) & (x > -0.5) & (x < 0.3) & (grey | sat)
    lab, n = label(m)
    out = []
    for k in range(1, n+1):
        mm = lab == k
        c = int(mm.sum())
        if c < 12:
            continue
        pts = P[mm]
        out.append((np.nanmedian(pts[:, :2], 0), c))
    merged = []
    for xy, c in sorted(out, key=lambda t: -t[1]):
        for mtd in merged:
            if np.linalg.norm(mtd[0] - xy) < 0.035:
                mtd[1] += c; break
        else:
            merged.append([xy, c])
    return [(m_[0], m_[1]) for m_ in merged]


def disc_stats(meta, imgs, xy, cam="front", rad=0.03):
    """Pixels near a target disc (thin height band): (#pixels, #highlight-coloured, #grey)."""
    rgb = imgs[(cam, "rgb")]; P = backproject(meta, cam, imgs[(cam, "depth")])
    valid = np.isfinite(P[..., 2])
    z = np.where(valid, P[..., 2], 9.0)
    dx = np.where(valid, P[..., 0] - xy[0], 9.0); dy = np.where(valid, P[..., 1] - xy[1], 9.0)
    m = valid & (z > 0.003) & (z < 0.02) & (dx*dx + dy*dy < rad*rad)
    srt = np.sort(rgb, -1)
    colored = (srt[..., 2] > 100) & (srt[..., 1] < 50)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    grey = (np.abs(r-g) < 14) & (np.abs(g-b) < 14) & (r > 80)
    return int(m.sum()), int((m & colored).sum()), int((m & grey).sum())
