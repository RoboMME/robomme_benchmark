"""Helpers: decode MCP observations, back-project depth, color segmentation."""
import base64, io, json, math, re
from pathlib import Path
import numpy as np
from PIL import Image

WS = Path("/workspace")


def parse(result):
    blocks = result["content"]
    meta = json.loads(blocks[0]["text"])
    imgs = {}
    for a in meta.get("attachments", []):
        b = blocks[a["content_index"]]
        data = base64.b64decode(b["data"])
        im = Image.open(io.BytesIO(data))
        if a["kind"] == "rgb":
            imgs[(a["camera"], "rgb")] = np.asarray(im.convert("RGB"))
        else:
            imgs[(a["camera"], "depth")] = np.asarray(im).astype(np.float32) * meta["depth"][a["camera"]].get("meters_per_unit", 0.001)
    return meta, imgs


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def backproject(meta, cam, depth, us, vs):
    """us, vs arrays of pixel coords -> world points Nx3 (uses depth image)."""
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    us = np.asarray(us); vs = np.asarray(vs)
    d = depth[vs.astype(int), us.astype(int)]
    pc = np.stack([(us - cx) * d / fx, -(vs - cy) * d / fy, -d], -1)
    return pc @ R.T + t, d


def world_points(meta, imgs, cam):
    depth = imgs[(cam, "depth")]
    H, W = depth.shape
    vs, us = np.mgrid[0:H, 0:W]
    P, d = backproject(meta, cam, depth, us.ravel(), vs.ravel())
    return P.reshape(H, W, 3), d.reshape(H, W)


def project(meta, cam, P):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    pc = (np.atleast_2d(P) - t) @ R
    d = -pc[:, 2]
    u = K[0, 0] * pc[:, 0] / d + K[0, 2]
    v = -K[1, 1] * pc[:, 1] / d + K[1, 2]
    return u, v


def hsv(rgb):
    im = Image.fromarray(rgb).convert("HSV")
    return np.asarray(im).astype(np.int32)


COLOR_HUE = {"red": (0, 10), "green": (60, 110), "blue": (150, 185), "yellow": (30, 50)}


def color_mask(rgb, color):
    h = hsv(rgb)
    H, S, V = h[..., 0], h[..., 1], h[..., 2]
    if color == "red":
        m = ((H <= 8) | (H >= 245)) & (S > 170) & (V > 60)
    elif color == "green":
        m = (H >= 60) & (H <= 110) & (S > 120) & (V > 50)
    elif color == "blue":
        m = (H >= 145) & (H <= 185) & (S > 120) & (V > 50)
    elif color == "yellow":
        m = (H >= 28) & (H <= 50) & (S > 120) & (V > 80)
    else:
        m = np.zeros(H.shape, bool)
    return m


def components(mask, min_px=4):
    """Simple 4-connected labeling; returns list of (ys, xs) arrays."""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    comps = []
    n = 0
    for y0, x0 in zip(*np.nonzero(mask)):
        if lab[y0, x0]:
            continue
        n += 1
        stack = [(y0, x0)]; lab[y0, x0] = n; ys = []; xs = []
        while stack:
            y, x = stack.pop(); ys.append(y); xs.append(x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n; stack.append((yy, xx))
        if len(ys) >= min_px:
            comps.append((np.array(ys), np.array(xs)))
    return comps


# ---------------- rotation helpers ----------------
def aa_to_R(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def yaw_of(aa):
    """Yaw of the gripper's x axis projected onto the world xy plane."""
    R = aa_to_R(aa)
    return math.atan2(R[1, 0], R[0, 0])


def wrap(a, period=2 * math.pi):
    return (a + period / 2) % period - period / 2


# ---------------- robot motion ----------------
class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log
        self.meta = None; self.imgs = None; self.ended = False; self.complete = False
        self.grip = -1.0

    def observe(self):
        r = self.ctx.call("robomme_observe")
        self.meta, self.imgs = parse(r)
        return self.meta

    @property
    def state(self):
        return np.array(self.meta["state"], float)

    def step(self, rows):
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        r = self.ctx.call("robomme_step", actions=rows)
        self.meta, self.imgs = parse(r)
        if self.meta.get("episode_ended"):
            self.ended = True
            tr = self.meta.get("transition") or {}
            if tr.get("task_complete"):
                self.complete = True
            for s in tr.get("steps", []) or []:
                if isinstance(s, dict) and s.get("task_complete"):
                    self.complete = True
            if self.meta.get("task_complete"):
                self.complete = True
        return self.meta

    def remaining(self):
        return self.meta.get("episode_limits", {}).get("remaining_episode_steps", 999)

    def move_to(self, target, tol=0.004, max_iter=12, vmax=0.8, yaw=None, settle=True):
        """Move TCP to world target (x,y,z); optional absolute yaw (mod pi/2 handled by caller)."""
        target = np.asarray(target, float)
        for it in range(max_iter):
            if self.ended:
                return False
            s = self.state
            err = target - s[:3]
            yerr = 0.0
            if yaw is not None:
                yerr = wrap(yaw - yaw_of(s[3:6]))
            dist = np.linalg.norm(err)
            if dist < tol and abs(yerr) < 0.03:
                return True
            n = int(max(1, math.ceil(dist / (0.05 * vmax)), math.ceil(abs(yerr) / (0.5 * 0.5))))
            n = min(n, 40)
            d = err / n / 0.05
            dr = yerr / n / 0.5
            rows = [[d[0], d[1], d[2], 0, 0, dr, self.grip]] * n
            if settle and dist < 0.03:
                rows = rows + [[0, 0, 0, 0, 0, 0, self.grip]] * 2
            self.step(rows)
        s = self.state
        self.log(f"move_to not converged: target {target.round(3)} at {s[:3].round(3)}")
        return np.linalg.norm(target - s[:3]) < 3 * tol

    def gripper(self, value, n=8):
        self.grip = value
        s = self.state
        self.step([[0, 0, 0, 0, 0, 0, value]] * n)
