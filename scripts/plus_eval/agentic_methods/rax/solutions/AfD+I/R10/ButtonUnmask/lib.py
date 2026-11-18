import json, base64, io, time, math
from pathlib import Path
import numpy as np
from PIL import Image

WS = Path("/workspace")
LOG = []

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

def decode(blocks):
    """Return (meta dict, {(camera,kind): ndarray})"""
    if isinstance(blocks, dict) and "content" in blocks:
        blocks = blocks["content"]
    meta = None
    for b in blocks:
        if b.get("type") == "text":
            try:
                meta = json.loads(b["text"]); break
            except Exception:
                continue
    imgs = {}
    if meta:
        for att in meta.get("attachments", []):
            i = att["content_index"]
            if i < len(blocks) and blocks[i].get("type") == "image":
                data = base64.b64decode(blocks[i]["data"])
                im = Image.open(io.BytesIO(data))
                imgs[(att["camera"], att["kind"])] = np.array(im)
    return meta, imgs

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def deproject(meta, cam, u, v, depth_img, rad=1):
    """u,v pixel; depth_img uint16 mm array. returns world xyz or None"""
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    h, w = depth_img.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    win = depth_img[max(0, vi-rad):vi+rad+1, max(0, ui-rad):ui+rad+1].astype(float)
    win = win[win > 0]
    if win.size == 0:
        return None
    d = float(np.median(win)) * 0.001
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    pc = np.array([(u-cx)*d/fx, -(v-cy)*d/fy, -d])
    R = quat_to_R(pose["quaternion_wxyz"])
    return R @ pc + np.array(pose["position_m"])

def deproject_all(meta, cam, depth_img):
    """world xyz for every pixel (H,W,3); invalid -> nan"""
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    h, w = depth_img.shape[:2]
    d = depth_img.astype(float) * 0.001
    d[depth_img == 0] = np.nan
    vv, uu = np.mgrid[0:h, 0:w]
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    pc = np.stack([(uu-cx)*d/fx, -(vv-cy)*d/fy, -d], -1)
    R = quat_to_R(pose["quaternion_wxyz"])
    return pc @ R.T + np.array(pose["position_m"])

def color_masks(a):
    a = a[..., :3].astype(int); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1); mn = a.min(-1)
    sat = (mx - mn) > 80
    return {
        'red': sat & (r > 150) & (g < 70) & (b < 70),
        'green': sat & (g > 110) & (r < 90) & (b < 90),
        'blue': sat & (b > 110) & (r < 70) & (g < 70),
        'yellow': sat & (r > 170) & (g > 170) & (b < 70),
        'purple': sat & (r > 110) & (b > 140) & (g < 100),
        'orange': sat & (r > 200) & (g > 100) & (g < 170) & (b < 50),
    }

def components(mask, min_px=4):
    """simple connected components (4-neigh) -> list of pixel arrays"""
    from collections import deque
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    out = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        q = deque([(y0, x0)]); seen[y0, x0] = True; pts = []
        while q:
            y, x = q.popleft(); pts.append((y, x))
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True; q.append((ny, nx))
        if len(pts) >= min_px:
            out.append(np.array(pts))
    return out

def load_png(rel):
    return np.array(Image.open(WS / rel))

def seq_frames(meta):
    """list of (frame_meta, {(cam,kind): path}) from a reply's frame_sequence"""
    fs = meta.get("frame_sequence") if meta else None
    if not fs:
        return []
    try:
        man = json.loads((WS / fs["manifest_path"]).read_text())
    except Exception as e:
        log("manifest read fail", e)
        return []
    out = []
    for f in man.get("frames", []):
        files = {(x["camera"], x["kind"]): x["path"] for x in f.get("files", [])}
        out.append((f, files))
    return out

class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = None
        self.frames = []   # (step_id, frame_meta, files) all saved frames seen
        self.ended = False
        self.complete = False
        self.grip = -1.0
        self.step_id = 0
        self.jumps = 0

    def observe(self):
        meta, imgs = decode(self.ctx.call("robomme_observe"))
        self.meta = meta
        self.step_id = meta.get("step_id", self.step_id)
        self._ingest(meta)
        return meta, imgs

    def _ingest(self, meta):
        for f, files in seq_frames(meta):
            self.frames.append((f.get("step_id"), f, files))
        if meta.get("episode_ended"):
            self.ended = True
        tr = meta.get("transition") or {}
        if tr.get("task_complete"):
            self.complete = True
        if tr.get("episode_ended"):
            self.ended = True
        for s in tr.get("steps", []) or []:
            if s.get("task_complete"):
                self.complete = True
            if s.get("episode_ended"):
                self.ended = True

    @property
    def pos(self):
        return np.array(self.meta["state"][:3])

    @property
    def remaining(self):
        return self.meta["episode_limits"]["remaining_episode_steps"]

    def step(self, rows):
        if self.ended:
            return None
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        before = self.step_id
        meta, imgs = decode(self.ctx.call("robomme_step", actions=rows))
        self.meta = meta
        if meta.get("step_id", before) - before > len(rows) + 2:
            self.jumps += 1
            log("WARN step jump", before, "->", meta.get("step_id"), "rows", len(rows), "pos", np.round(meta["state"][:3], 3))
        self.step_id = meta.get("step_id", self.step_id)
        self._ingest(meta)
        return meta, imgs

    def hold(self, n, grip=None):
        g = self.grip if grip is None else grip
        self.grip = g
        out = None
        while n > 0 and not self.ended:
            k = min(n, 50)
            out = self.step([[0, 0, 0, 0, 0, 0, g]] * k)
            n -= k
        return out

    def move_to(self, target, speed=0.8, tol=0.004, max_iter=14, chunk=6, grip=None, max_steps=200):
        g = self.grip if grip is None else grip
        self.grip = g
        target = np.array(target, float)
        used = 0
        for it in range(max_iter):
            if self.ended:
                return False
            err = target - self.pos
            dist = np.abs(err).max()
            if np.linalg.norm(err) < tol:
                return True
            eff = 0.4 * 0.05 * speed          # measured effective metres/step at command `speed`
            n = int(math.ceil(dist / eff))
            n = max(1, min(n, chunk))
            d = err / (n * 0.4 * 0.05)
            d = np.clip(d, -speed, speed)
            j0 = self.jumps
            self.step([[d[0], d[1], d[2], 0, 0, 0, g]] * n)
            used += n
            if self.jumps > j0:
                log("move_to aborted after IK penalty; target", np.round(target, 3))
                return False
            if used >= max_steps:
                break
        return np.linalg.norm(target - self.pos) < tol * 3

def table_objects(fmeta, depth, zmin=0.012, zmax=0.15, xlim=(-0.16, 0.35), ylim=(-0.4, 0.4), cell=0.004, views=None):
    """cluster raised points on the table from depth views. views: list of (meta, cam, depth).
    returns list of dict(center, n, zmax, pts, tall)"""
    if views is None:
        views = [(fmeta, "front", depth)]
    allp = []
    for v in views:
        vm, cam, dep = v[:3]
        P = deproject_all(vm, cam, dep).reshape(-1, 3)
        okv = np.isfinite(P).all(-1)
        if len(v) > 3 and v[3] is not None:
            okv &= v[3].reshape(-1)
        allp.append(P[okv])
    P = np.concatenate(allp, 0)
    inxy = (P[:, 0] > xlim[0]) & (P[:, 0] < xlim[1]) & (P[:, 1] > ylim[0]) & (P[:, 1] < ylim[1])
    tall = P[inxy & (P[:, 2] > 0.11) & (P[:, 2] < 0.30)]
    ok = inxy & (P[:, 2] > zmin) & (P[:, 2] < zmax)
    pts = P[ok]
    if len(pts) == 0:
        return []
    # grid-based clustering in xy
    g = np.floor(pts[:, :2] / cell).astype(int)
    from collections import defaultdict, deque
    cells = defaultdict(list)
    for i, k in enumerate(map(tuple, g)):
        cells[k].append(i)
    seen = set(); clusters = []
    for k in cells:
        if k in seen: continue
        q = deque([k]); seen.add(k); idx = []
        while q:
            c = q.popleft(); idx += cells[c]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (c[0]+dx, c[1]+dy)
                    if n in cells and n not in seen:
                        seen.add(n); q.append(n)
        cp = pts[idx]
        if len(cp) < 15: continue
        clusters.append(dict(center=np.array([np.median(cp[:, 0]), np.median(cp[:, 1])]),
                             n=len(cp), zmax=float(np.percentile(cp[:, 2], 95)), pts=cp,
                             ext=(cp[:, :2].max(0) - cp[:, :2].min(0))))
    for c in clusters:
        c["tall"] = int(((np.abs(tall[:, :2] - c["center"]) < 0.03).all(-1)).sum()) if len(tall) else 0
    return clusters

def top_center_yaw(cl, band=0.010):
    p = cl["pts"]
    top = p[p[:, 2] > cl["zmax"] - band][:, :2]
    if len(top) < 8:
        top = p[:, :2]
    c = np.median(top, 0)
    best = None
    for deg in range(0, 90, 2):
        a = math.radians(deg)
        Rm = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        q = (top - c) @ Rm.T
        lo = np.percentile(q, 3, 0); hi = np.percentile(q, 97, 0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            best = (area, deg, (lo + hi) / 2 @ Rm + c, hi - lo)
    return best[2], best[1], best[3], len(top)

def axang_to_R(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-9: return np.eye(3)
    k = v / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K

def R_to_axang(Rm):
    th = math.acos(max(-1, min(1, (np.trace(Rm) - 1) / 2)))
    if th < 1e-9: return np.zeros(3)
    if abs(th - math.pi) < 1e-4:
        w, V = np.linalg.eig(Rm); k = np.real(V[:, np.argmin(abs(w - 1))]); return k * th
    k = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]]) / (2 * math.sin(th))
    return k * th

def gripper_yaw(state):
    """angle of the gripper's finger-closing axis (tool y) in world xy, radians"""
    Rm = axang_to_R(state[3:6])
    y = Rm[:, 1]
    return math.atan2(y[1], y[0])

def wrap90(a):
    """wrap degrees to [-45,45)"""
    return (a + 45) % 90 - 45

def rotate_yaw(R, target_finger_deg, grip=None, max_rate=0.3):
    """rotate about world z so finger axis angle == target (mod 90)"""
    g = R.grip if grip is None else grip
    for _ in range(4):
        cur = math.degrees(gripper_yaw(R.meta["state"]))
        d = wrap90(target_finger_deg - cur)
        if abs(d) < 2:
            return True
        rad = math.radians(d)
        n = max(1, int(math.ceil(abs(rad) / (0.5 * max_rate))))
        r = rad / n / 0.5
        R.step([[0, 0, 0, 0, 0, r, g]] * n)
    return False

def place_back(R, c, z_place=0.055, z_up=0.17):
    R.move_to([c[0], c[1], z_place + 0.01], speed=0.4)
    R.hold(6, grip=-1.0)
    R.move_to([c[0], c[1], z_up], speed=0.5)

def travel(R, c, z_end, z_safe=0.36, speed=0.9):
    """rise to safe height, move laterally, then descend (avoid tall occluders)"""
    if np.linalg.norm(R.pos[:2] - np.asarray(c[:2])) > 0.03:
        if R.pos[2] < z_safe - 0.02:
            R.move_to([R.pos[0], R.pos[1], z_safe], speed=speed, tol=0.015)
        R.move_to([c[0], c[1], max(z_safe, z_end)], speed=speed, tol=0.01)
    R.move_to([c[0], c[1], z_end], speed=0.6)


def pick_lift(R, c, yaw, z_grasp=0.05, z_up=0.17, z_app=0.15):
    R.grip = -1.0
    R.move_to([c[0], c[1], z_app], speed=0.6)
    rotate_yaw(R, yaw)
    R.move_to([c[0], c[1], z_grasp], speed=0.4)
    R.hold(8, grip=1.0)
    R.move_to([c[0], c[1], z_up], speed=0.4)
    w = R.meta["state"][6] + R.meta["state"][7]
    return w
