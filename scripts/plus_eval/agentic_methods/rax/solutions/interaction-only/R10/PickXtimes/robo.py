"""Observation decoding, geometry and motion helpers for PickXtimes."""
import base64, io, json, time
import numpy as np
from PIL import Image

LOG = []

def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def decode(result):
    blocks = result.get("content", [])
    meta = None
    for b in blocks:
        if b.get("type") == "text":
            try:
                meta = json.loads(b["text"]); break
            except Exception:
                pass
    imgs = {}
    if meta is not None:
        for att in meta.get("attachments", []):
            b = blocks[att["content_index"]]
            if b.get("type") != "image":
                continue
            im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
            arr = np.array(im)
            if att["kind"] == "rgb":
                arr = np.array(im.convert("RGB"))
            imgs[(att["camera"], att["kind"])] = arr
    return meta, imgs


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def backproject(meta, cam, uv, depth_img):
    """uv: (N,2) array of (u,v) pixel coords. Returns (N,3) world points; NaN if invalid."""
    K = np.array(meta["camera_intrinsics"][cam]["matrix"], float)
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"], float)
    uv = np.asarray(uv, float).reshape(-1, 2)
    ui = np.clip(np.round(uv[:, 0]).astype(int), 0, depth_img.shape[1]-1)
    vi = np.clip(np.round(uv[:, 1]).astype(int), 0, depth_img.shape[0]-1)
    d = depth_img[vi, ui].astype(float) * 0.001
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.stack([(uv[:, 0]-cx)*d/fx, -(uv[:, 1]-cy)*d/fy, -d], 1)
    pw = pc @ R.T + t
    pw[d <= 0] = np.nan
    return pw


def all_points(meta, cam, depth_img):
    H, W = depth_img.shape
    vv, uu = np.mgrid[0:H, 0:W]
    pts = backproject(meta, cam, np.stack([uu.ravel(), vv.ravel()], 1), depth_img)
    return pts.reshape(H, W, 3)


def hsv(rgb):
    import cv2
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  # H 0..179, S,V 0..255


COLOR_HUES = {"red": [(0, 8), (172, 180)], "green": [(45, 85)], "blue": [(100, 130)],
              "yellow": [(22, 35)], "purple": [(130, 160)], "orange": [(9, 20)],
              "cyan": [(86, 99)], "magenta": [(150, 171)], "pink": [(150, 171)]}


def color_mask(rgb, color, smin=120, vmin=60):
    h = hsv(rgb)
    m = np.zeros(rgb.shape[:2], bool)
    for lo, hi in COLOR_HUES[color]:
        m |= (h[..., 0] >= lo) & (h[..., 0] < hi)
    m &= (h[..., 1] >= smin) & (h[..., 2] >= vmin)
    return m


def components(mask, min_area=8):
    import cv2
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out.append({"area": int(stats[i, cv2.CC_STAT_AREA]), "centroid": cents[i],
                        "mask": lab == i})
    out.sort(key=lambda c: -c["area"])
    return out


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = None; self.imgs = None
        self.ended = False; self.complete = False
        self.steps_used = 0
        self.goal_yaw = 0.0
        self.hooks = []

    def observe(self):
        r = self.ctx.call("robomme_observe")
        self.meta, self.imgs = decode(r)
        self._check_end(self.meta)
        self._run_hooks()
        return self.meta, self.imgs

    @property
    def state(self):
        return np.array(self.meta["state"], float)

    def _check_end(self, meta):
        if meta is None:
            return
        if meta.get("episode_ended"):
            self.ended = True
        for s in meta.get("steps", []) or []:
            if isinstance(s, dict):
                if s.get("task_complete"):
                    self.complete = True
                if s.get("episode_ended"):
                    self.ended = True
        if meta.get("task_complete"):
            self.complete = True
        tr = meta.get("transition") or {}
        if isinstance(tr, dict):
            if tr.get("episode_ended"):
                self.ended = True
            if tr.get("task_complete"):
                self.complete = True
            for s in tr.get("steps", []) or []:
                if isinstance(s, dict) and s.get("episode_ended"):
                    self.ended = True

    def step(self, rows):
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        for i in range(0, len(rows), 100):
            if self.ended:
                return
            r = self.ctx.call("robomme_step", actions=rows[i:i+100])
            self.meta, self.imgs = decode(r)
            self._check_end(self.meta)
            self.steps_used += len(rows[i:i+100])
            self._run_hooks()

    def _run_hooks(self):
        for h in self.hooks:
            try:
                h(self)
            except Exception as e:
                log("hook error", e)

    def remaining(self):
        try:
            return int(self.meta["episode_limits"]["remaining_episode_steps"])
        except Exception:
            return 10**6

    def move_to(self, target, grip, tol=0.004, max_steps=80, gain=1.0, vmax=1.0):
        """Closed-loop move of TCP to target xyz holding gripper command grip."""
        target = np.asarray(target, float)
        n = 0
        while n < max_steps and not self.ended:
            p = self.state[:3]
            err = target - p
            if np.linalg.norm(err) < tol and np.linalg.norm(self.rot_err()) < 0.05:
                return True
            d = np.clip(err / 0.05 * gain, -vmax, vmax)
            r = np.clip(self.rot_err() / 0.5 * 0.8, -0.4, 0.4)
            self.step([[d[0], d[1], d[2], r[0], r[1], r[2], grip]])
            n += 1
        return np.linalg.norm(target - self.state[:3]) < tol * 3

    def rot_err(self):
        """World-frame rotation vector taking current orientation to straight-down yaw-0."""
        from scipy.spatial.transform import Rotation as Rot
        cur = Rot.from_rotvec(self.state[3:6])
        goal = Rot.from_rotvec([0, 0, self.goal_yaw]) * Rot.from_rotvec([np.pi, 0, 0])
        e = (goal * cur.inv()).as_rotvec()
        return e

    def hold(self, grip, n):
        for _ in range(n):
            if self.ended:
                return
            r = np.clip(self.rot_err() / 0.5 * 0.8, -0.4, 0.4)
            self.step([[0, 0, 0, r[0], r[1], r[2], grip]])
