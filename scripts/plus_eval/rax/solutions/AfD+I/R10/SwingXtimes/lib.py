import base64, io, json, math, re, time
import numpy as np
from PIL import Image

LOG = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

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

def decode_obs(reply):
    meta, blocks = parse(reply)
    out = {"meta": meta}
    for att in meta.get("attachments", []):
        b = blocks[att["content_index"]]
        data = base64.b64decode(b["data"])
        im = Image.open(io.BytesIO(data))
        if att["kind"] == "rgb":
            out[(att["camera"], "rgb")] = np.asarray(im.convert("RGB"))
        else:
            out[(att["camera"], "depth")] = np.asarray(im).astype(np.float64) * 0.001
    return out

def observe(ctx):
    return decode_obs(ctx.call("robomme_observe"))

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def backproject(o, cam, us, vs):
    """us, vs arrays of pixel coords; returns Nx3 world points (nan for invalid)."""
    meta = o["meta"]
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    D = o[(cam, "depth")]
    us = np.asarray(us); vs = np.asarray(vs)
    d = D[vs.astype(int), us.astype(int)]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.stack([(us - cx) * d / fx, -(vs - cy) * d / fy, -d], axis=-1)
    pw = pc @ R.T + t
    pw[d <= 0] = np.nan
    return pw

def world_map(o, cam):
    H, W = o[(cam, "depth")].shape
    vs, us = np.mgrid[0:H, 0:W]
    return backproject(o, cam, us.ravel(), vs.ravel()).reshape(H, W, 3)

def components(mask, min_px=5):
    """simple connected components (4-neighbour) returning list of (ys, xs)."""
    H, W = mask.shape
    lab = np.zeros((H, W), int); comps = []
    cur = 0
    for y0, x0 in zip(*np.nonzero(mask)):
        if lab[y0, x0]:
            continue
        cur += 1; stack = [(y0, x0)]; lab[y0, x0] = cur; pts = []
        while stack:
            y, x = stack.pop(); pts.append((y, x))
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                yy, xx = y+dy, x+dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = cur; stack.append((yy, xx))
        if len(pts) >= min_px:
            a = np.array(pts); comps.append((a[:, 0], a[:, 1]))
    return comps

def color_mask(rgb, color):
    a = rgb.astype(int); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    if color == "red":
        return (r > 80) & (g < 0.35 * r) & (b < 0.35 * r)
    if color == "green":
        return (g > 70) & (r < 0.5 * g) & (b < 0.5 * g)
    if color == "blue":
        return (b > 70) & (r < 0.5 * b) & (g < 0.6 * b)
    raise ValueError(color)

def white_mask(rgb):
    a = rgb.astype(int); mx = a.max(-1); mn = a.min(-1)
    return (mn > 75) & (mx - mn < 28)

def step(ctx, rows):
    """execute rows (list of 7-lists), returns meta of reply"""
    rows = [[float(max(-1, min(1, v))) for v in r] for r in rows]
    out = None
    for i in range(0, len(rows), 100):
        rep = ctx.call("robomme_step", actions=rows[i:i+100])
        out, _ = parse(rep)
        if out.get("episode_ended"):
            break
    return out

class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = None
        self.grip = -1.0
        self.ended = False
        self.complete = False
        self.refresh()
        self.grip = 1.0 if self.fingers < 0.075 else -1.0

    def refresh(self):
        rep = self.ctx.call("robomme_observe")
        self.last_reply = rep
        self.meta, _ = parse(rep)
        return self.meta

    @property
    def pos(self):
        return np.array(self.meta["state"][:3])

    @property
    def fingers(self):
        return sum(self.meta["state"][6:8])

    @property
    def remaining(self):
        return self.meta["episode_limits"]["remaining_episode_steps"]

    def send(self, rows):
        if self.ended:
            return self.meta
        rows = [[float(max(-1, min(1, v))) for v in r] for r in rows]
        rep = self.ctx.call("robomme_step", actions=rows)
        self.last_reply = rep
        m, _ = parse(rep)
        self.meta = m
        tr = m.get("transition", {})
        for s in tr.get("steps", []):
            if s.get("task_complete"):
                self.complete = True
        if m.get("episode_ended") or tr.get("episode_ended"):
            self.ended = True
            log("EPISODE ENDED step", m.get("step_id"), "complete", self.complete, json.dumps(tr.get("steps", [])[-1:]))
        return m

    def rot_fix(self):
        """rotation delta (in action units) to hold gripper pointing down with yaw self.yaw"""
        return [0.0, 0.0, 0.0]

    def move_to(self, target, tol=0.006, max_steps=80, speed=1.0, chunk=4, settle=True):
        target = np.asarray(target, float)
        used = 0
        while used < max_steps and not self.ended:
            err = target - self.pos
            dist = np.linalg.norm(err)
            if dist < tol:
                break
            n = int(min(chunk, max_steps - used))
            rows = []
            for k in range(n):
                # proportional, saturating at speed*0.05 m per step
                d = err / 0.05
                mag = np.max(np.abs(d))
                lim = speed
                if mag > lim:
                    d = d * lim / mag
                if dist < 0.03:
                    d = d * 1.0
                rows.append(list(d) + self.rot_fix() + [self.grip])
                err = err - d * 0.05 * 0.9
            self.send(rows)
            used += n
        return used

    def hold(self, n, grip=None):
        if grip is not None:
            self.grip = grip
        return self.send([[0, 0, 0, 0, 0, 0, self.grip]] * n)


def find_color(o, cam, color, zmax=None):
    rgb = o[(cam, "rgb")]; P = world_map(o, cam)
    best = None
    for ys, xs in components(color_mask(rgb, color), 4):
        pts = P[ys, xs]
        ok = np.isfinite(pts[:, 0])
        if ok.sum() < 3: continue
        p = np.nanmedian(pts[ok], axis=0)
        if zmax is not None and p[2] > zmax: continue
        if p[2] < 0.015: continue
        if best is None or len(xs) > best[1]:
            best = (p, len(xs), (xs.mean(), ys.mean()))
    return best

def grasp(r, color, approach_z=0.10):
    o = decode_obs(r.last_reply)
    f = find_color(o, "wrist", color, zmax=0.08) or find_color(o, "front", color, zmax=0.08)
    if f is None:
        log("grasp: cube not found"); return False
    p = f[0]; log("grasp: cube at", np.round(p, 3))
    r.grip = -1.0
    r.move_to([p[0], p[1], max(approach_z, r.pos[2]) if r.pos[2] < approach_z else approach_z])
    # refine from close wrist view
    r.hold(1)
    o = decode_obs(r.last_reply)
    f2 = find_color(o, "wrist", color, zmax=0.08)
    if f2 is not None and np.linalg.norm(f2[0][:2] - p[:2]) < 0.05:
        p = f2[0]; log("grasp: refined", np.round(p, 3))
        r.move_to([p[0], p[1], approach_z], tol=0.004)
    zc = max(p[2] - 0.02, 0.015)
    r.move_to([p[0], p[1], zc], speed=0.5, max_steps=40)
    r.hold(8, grip=1.0)
    r.move_to([p[0], p[1], 0.14], max_steps=40)
    r.hold(2)
    ok = 0.02 < r.fingers < 0.07
    log("grasp: fingers", round(r.fingers, 4), "ok", ok)
    return ok
