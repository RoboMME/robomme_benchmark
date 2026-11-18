"""VideoUnmask helpers: parsing, perception from conditioning video, closed-loop motion."""
import base64, io, json, math, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial.transform import Rotation as Rot

WS = Path("/workspace")
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def parse(result):
    blocks = result.get("content", [])
    meta = json.loads(next(b["text"] for b in blocks if b.get("type") == "text"))
    imgs = {}
    for att in meta.get("attachments", []):
        b = blocks[att["content_index"]]
        if b.get("type") != "image":
            continue
        raw = base64.b64decode(b["data"])
        im = Image.open(io.BytesIO(raw))
        imgs[(att["camera"], att["kind"])] = np.array(im)
    return meta, imgs


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta, self.imgs = parse(ctx.call("robomme_observe"))
        self.ended = False
        self.complete = False

    @property
    def state(self):
        return np.array(self.meta["state"], dtype=float)

    @property
    def remaining(self):
        lim = self.meta.get("episode_limits") or {}
        return lim.get("remaining_episode_steps", 0)

    def observe(self):
        self.meta, self.imgs = parse(self.ctx.call("robomme_observe"))
        return self.meta

    def step(self, rows):
        if self.ended:
            return False
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        rem = self.remaining
        if rem <= 0:
            self.ended = True
            return False
        rows = rows[:rem]
        self.meta, self.imgs = parse(self.ctx.call("robomme_step", actions=rows))
        tr = self.meta.get("transition") or {}
        for s in tr.get("steps", []):
            if s.get("task_complete"):
                self.complete = True
            if s.get("episode_ended"):
                self.ended = True
        if self.meta.get("episode_ended"):
            self.ended = True
        if self.ended:
            log("EPISODE ENDED at step", self.meta.get("step_id"), "complete", self.complete)
        return not self.ended

    # ---- pose helpers
    def yaw(self):
        R = Rot.from_rotvec(self.state[3:6]).as_matrix()
        # gripper x-axis projected in world xy
        return math.atan2(R[1, 0], R[0, 0])

    def move_to(self, target, yaw=None, grip=-1.0, tol=0.004, max_steps=80, chunk=2, gain=1.6, vmax=1.0,
                yaw_tol=0.03):
        target = np.asarray(target, float)
        used = 0
        while used < max_steps and not self.ended:
            p = self.state[:3]
            err = target - p
            yerr = 0.0
            if yaw is not None:
                yerr = wrap(yaw - self.yaw())
            if np.linalg.norm(err) < tol and abs(yerr) < yaw_tol:
                return True
            d = np.clip(err / 0.05 * gain, -vmax, vmax)
            n = np.linalg.norm(d)
            if n > vmax:
                d = d / n * vmax
            rz = float(np.clip(yerr / 0.5 * 1.5, -0.6, 0.6))
            row = [d[0], d[1], d[2], 0.0, 0.0, rz, grip]
            # keep the gripper pointing down: correct roll/pitch
            R = Rot.from_rotvec(self.state[3:6]).as_matrix()
            z = R[:, 2]  # tool z axis, should be (0,0,-1)
            corr = np.cross(z, np.array([0, 0, -1.0]))
            row[3] = float(np.clip(corr[0] / 0.5 * 1.5, -0.3, 0.3))
            row[4] = float(np.clip(corr[1] / 0.5 * 1.5, -0.3, 0.3))
            self.step([row] * chunk)
            used += chunk
        return False

    def hold(self, n, grip):
        rows = [[0, 0, 0, 0, 0, 0, grip]] * n
        self.step(rows)

    def gripper_open(self):
        return float(self.state[6] + self.state[7])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------- geometry
def backproject(meta, depth_img, cam, u, v):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"], float)
    pose = meta["camera_poses"][cam]
    d = float(depth_img[int(v), int(u)]) * 0.001
    if d <= 0:
        return None
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.array([(u - cx) * d / fx, -(v - cy) * d / fy, -d])
    q = pose["quaternion_wxyz"]
    R = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return R @ pc + np.array(pose["position_m"])


def project(meta, cam, pw):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"], float)
    pose = meta["camera_poses"][cam]
    q = pose["quaternion_wxyz"]
    R = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    pc = R.T @ (np.asarray(pw) - np.array(pose["position_m"]))
    d = -pc[2]
    u = K[0, 0] * pc[0] / d + K[0, 2]
    v = -K[1, 1] * pc[1] / d + K[1, 2]
    return u, v


def world_points(meta, depth_img, cam):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"], float)
    pose = meta["camera_poses"][cam]
    h, w = depth_img.shape
    vv, uu = np.mgrid[0:h, 0:w]
    d = depth_img.astype(float) * 0.001
    pc = np.stack([(uu - K[0, 2]) * d / K[0, 0], -(vv - K[1, 2]) * d / K[1, 1], -d], -1)
    q = pose["quaternion_wxyz"]
    R = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    pw = pc @ R.T + np.array(pose["position_m"])
    pw[d <= 0] = np.nan
    return pw


# ---------- colour naming
COLORS = {
    "red": (0, 0), "orange": (20, 0), "yellow": (42, 0), "green": (85, 0), "cyan": (128, 0),
    "blue": (170, 0), "purple": (200, 0), "magenta": (213, 0), "pink": (230, 0),
}


def name_color(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    import colorsys
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hd = h * 360
    if hd < 12 or hd >= 340:
        return "red"
    if hd < 50:
        return "orange"
    if hd < 75:
        return "yellow"
    if hd < 165:
        return "green"
    if hd < 195:
        return "cyan"
    if hd < 255:
        return "blue"
    if hd < 290:
        return "purple"
    return "pink"


def color_blobs(rgb_img, min_px=6, with_black=False):
    im = Image.fromarray(rgb_img[..., :3].astype(np.uint8))
    hsv = np.array(im.convert("HSV")).astype(int)
    rgb = rgb_img[..., :3].astype(int)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # the wooden table has PIL hue 13-23 and saturation 125-176
    mask = ((sat > 215) & (val > 60)) | ((sat > 80) & (val > 40) & ((hue < 8) | (hue > 32)))
    lab, n = ndimage.label(mask)
    out = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        if len(ys) < min_px:
            continue
        c = np.median(rgb[ys, xs], axis=0)
        out.append(dict(u=float(xs.mean()), v=float(ys.mean()), n=int(len(ys)), rgb=tuple(int(x) for x in c),
                        color=name_color(c)))
    if with_black:
        bm = (val < 35)
        lab, n = ndimage.label(bm)
        for k in range(1, n + 1):
            ys, xs = np.nonzero(lab == k)
            if len(ys) < min_px or len(ys) > 2500:
                continue
            c = np.median(rgb[ys, xs], axis=0)
            out.append(dict(u=float(xs.mean()), v=float(ys.mean()), n=int(len(ys)), rgb=tuple(int(x) for x in c),
                            color="black"))
    return out


def analyze_video(manifest_path):
    """Return per colour the last-seen front pixel position (u,v) and frame index."""
    m = json.loads((WS / manifest_path).read_text())
    frames = sorted(m["frames"], key=lambda f: f["frame_index"])
    last = {}
    first = {}
    for f in frames:
        if f.get("phase") != "video_demo":
            continue
        files = {(x["camera"], x["kind"]): x["path"] for x in f["files"]}
        p = files.get(("front", "rgb"))
        if not p:
            continue
        a = np.array(Image.open(WS / p).convert("RGB"))
        for b in color_blobs(a):
            if b["n"] < 30:
                continue
            c = b["color"]
            last[c] = dict(u=b["u"], v=b["v"], n=b["n"], frame=f["frame_index"], rgb=b["rgb"])
            first.setdefault(c, f["frame_index"])
    return last, first, frames


def parse_targets(instruction):
    low = instruction.lower()
    found = []
    for c in ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta", "pink", "violet", "black", "white", "gray", "grey", "brown"]:
        i = 0
        while True:
            j = low.find(c, i)
            if j < 0:
                break
            found.append((j, c))
            i = j + 1
    found.sort()
    out = []
    for _, c in found:
        c = {"violet": "purple", "magenta": "pink"}.get(c, c)
        out.append(c)
    return out


# ---------- container detection from a top view
def detect_containers(meta, depth_img, cam="wrist", zlo=0.055, zhi=0.095, min_pts=150):
    import cv2
    P = world_points(meta, depth_img, cam)
    z = P[..., 2]
    sel = (z > zlo) & (z < zhi)
    sel &= ~np.isnan(z)
    lab, n = ndimage.label(sel)
    out = []
    for k in range(1, n + 1):
        q = P[lab == k]
        if len(q) < min_pts:
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect((q[:, :2] * 1000).astype(np.float32))
        out.append(dict(xy=np.array([cx, cy]) / 1000.0, size=(w / 1000.0, h / 1000.0), ang=math.radians(ang),
                        ztop=float(np.median(q[:, 2])), npts=int(len(q))))
    return out


def nearest_yaw(box_ang, cur):
    """box yaw is defined mod 90deg; choose equivalent closest to current gripper yaw."""
    best = None
    for k in range(-4, 5):
        a = box_ang + k * math.pi / 2
        if best is None or abs(wrap(a - cur)) < abs(wrap(best - cur)):
            best = a
    return cur + wrap(best - cur)


def cube_world_positions(manifest_path, last):
    """Back-project each colour's last-seen video pixel using that frame's front depth."""
    m = json.loads((WS / manifest_path).read_text())
    byidx = {f["frame_index"]: f for f in m["frames"]}
    out = {}
    for c, info in last.items():
        f = byidx[info["frame"]]
        files = {(x["camera"], x["kind"]): x["path"] for x in f["files"]}
        dp = files.get(("front", "depth"))
        if not dp:
            continue
        d = np.array(Image.open(WS / dp))
        u, v = int(round(info["u"])), int(round(info["v"]))
        pts = []
        for dv in range(-2, 3):
            for du in range(-2, 3):
                p = backproject(f, d, "front", min(255, max(0, u + du)), min(255, max(0, v + dv)))
                if p is not None and -0.02 < p[2] < 0.06:
                    pts.append(p)
        if pts:
            out[c] = np.median(np.array(pts), axis=0)
    return out


def analyze_video_v2(manifest_path, cams=("front", "wrist"), min_px=12, zmax=0.07, zmin=0.012):
    """Detect saturated colour cubes in every conditioning frame from both cameras and
    cluster them in world xy. Returns list of dict(color, xy, n_obs, first, last, cams)."""
    m = json.loads((WS / manifest_path).read_text())
    frames = sorted(m["frames"], key=lambda f: f["frame_index"])
    obs = []  # (color, x, y, z, frame, cam, npx)
    for f in frames:
        if f.get("phase") != "video_demo":
            continue
        files = {(x["camera"], x["kind"]): x["path"] for x in f["files"]}
        for cam in cams:
            pr, pd = files.get((cam, "rgb")), files.get((cam, "depth"))
            if not pr or not pd:
                continue
            a = np.array(Image.open(WS / pr).convert("RGB"))
            blobs = color_blobs(a, min_px=min_px, with_black=True)
            if not blobs:
                continue
            d = np.array(Image.open(WS / pd))
            P = None
            for b in blobs:
                if P is None:
                    P = world_points(f, d, cam)
                # pixels of blob: approximate with window around centroid, take median of valid pts
                u, v = int(round(b["u"])), int(round(b["v"]))
                win = P[max(0, v - 2):v + 3, max(0, u - 2):u + 3].reshape(-1, 3)
                win = win[~np.isnan(win[:, 0])]
                if len(win) == 0:
                    continue
                p = np.median(win, axis=0)
                if not (zmin < p[2] < zmax):
                    continue
                # push ~1cm along the viewing ray (surface -> cube centre)
                cpos = np.array(f["camera_poses"][cam]["position_m"])
                ray = p - cpos
                ray /= np.linalg.norm(ray)
                z_surf = p[2]
                p = p + 0.01 * ray
                obs.append((b["color"], p[0], p[1], z_surf, f["frame_index"], cam, b["n"]))
    # cluster per colour
    clusters = []
    for o in obs:
        best = None
        for c in clusters:
            if c["color"] != o[0]:
                continue
            dist = np.linalg.norm(np.array(o[1:3]) - c["xy"])
            if dist < 0.03 and (best is None or dist < best[0]):
                best = (dist, c)
        if best is None:
            clusters.append(dict(color=o[0], pts=[o], xy=np.array(o[1:3])))
        else:
            c = best[1]
            c["pts"].append(o)
            # wrist observations are better (top-down); weight them
            w = np.array([3.0 if q[5] == "wrist" else 1.0 for q in c["pts"]])
            xyz = np.array([q[1:3] for q in c["pts"]])
            c["xy"] = (xyz * w[:, None]).sum(0) / w.sum()
    out = []
    for c in clusters:
        fr = [q[4] for q in c["pts"]]
        out.append(dict(color=c["color"], xy=c["xy"], n_obs=len(c["pts"]), first=min(fr), last=max(fr),
                        cams=sorted(set(q[5] for q in c["pts"])), zmed=float(np.median([q[3] for q in c["pts"]])),
                        npx=float(np.median([q[6] for q in c["pts"]]))))
    return out


def parse_goal(instruction):
    """Return ordered list of (colour, want_all)."""
    import re
    low = instruction.lower()
    names = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta", "pink", "violet",
             "black", "white", "gray", "grey", "brown"]
    out = []
    for mt in re.finditer(r"\b(" + "|".join(names) + r")\b", low):
        c = mt.group(1)
        c = {"violet": "purple", "magenta": "pink", "grey": "gray"}.get(c, c)
        pre = low[max(0, mt.start() - 40):mt.start()]
        # look only at the clause containing this colour
        for sep in [",", " then ", " followed by ", " and finally ", " finally "]:
            k = pre.rfind(sep)
            if k >= 0:
                pre = pre[k + len(sep):]
        post = low[mt.end():mt.end() + 8]
        want_all = ("all" in pre.split()) or ("every" in pre) or post.strip().startswith("cubes")
        if out and out[-1][0] == c:
            continue
        out.append((c, want_all))
    return out


def finger_clearance(P, xy, yaw, inner=0.027, outer=0.07, half_w=0.016, zmin=0.015):
    """Count obstacle points (z>zmin) in the two finger footprints for gripper yaw."""
    ax = np.array([math.sin(yaw), -math.cos(yaw)])
    perp = np.array([math.cos(yaw), math.sin(yaw)])
    Q = P.reshape(-1, 3)
    Q = Q[~np.isnan(Q[:, 0])]
    rel = Q[:, :2] - xy
    a = rel @ ax
    b = rel @ perp
    sel = (np.abs(a) > inner) & (np.abs(a) < outer) & (np.abs(b) < half_w) & (Q[:, 2] > zmin) & (Q[:, 2] < 0.12)
    return int(sel.sum()), (float(Q[sel, 2].max()) if sel.any() else 0.0)


def color_blobs_depth(rgb_img, P, zlo=0.012, zhi=0.065, min_px=10, max_px=3000):
    """Cube blobs = pixels whose world height is cube-like and that are coloured (or black).
    Independent of the table appearance."""
    im = Image.fromarray(rgb_img[..., :3].astype(np.uint8))
    hsv = np.array(im.convert("HSV")).astype(int)
    rgb = rgb_img[..., :3].astype(int)
    z = P[..., 2]
    zok = (z > zlo) & (z < zhi)
    zok &= ~np.isnan(z)
    sat, val = hsv[..., 1], hsv[..., 2]
    out = []
    for kind, mask in (("color", zok & (sat > 90) & (val > 40)), ("black", zok & (val < 35))):
        lab, n = ndimage.label(mask)
        for k in range(1, n + 1):
            ys, xs = np.nonzero(lab == k)
            if len(ys) < min_px or len(ys) > max_px:
                continue
            c = np.median(rgb[ys, xs], axis=0)
            pts = P[ys, xs]
            out.append(dict(u=float(xs.mean()), v=float(ys.mean()), n=int(len(ys)), rgb=tuple(int(x) for x in c),
                            color=name_color(c) if kind == "color" else "black",
                            p=np.nanmedian(pts, axis=0)))
    return out


def analyze_video_v3(manifest_path, cams=("front", "wrist")):
    m = json.loads((WS / manifest_path).read_text())
    frames = sorted(m["frames"], key=lambda f: f["frame_index"])
    obs = []
    for f in frames:
        if f.get("phase") != "video_demo":
            continue
        files = {(x["camera"], x["kind"]): x["path"] for x in f["files"]}
        for cam in cams:
            pr, pd = files.get((cam, "rgb")), files.get((cam, "depth"))
            if not pr or not pd:
                continue
            a = np.array(Image.open(WS / pr).convert("RGB"))
            d = np.array(Image.open(WS / pd))
            P = world_points(f, d, cam)
            for b in color_blobs_depth(a, P):
                p = b["p"]
                cpos = np.array(f["camera_poses"][cam]["position_m"])
                ray = p - cpos
                ray /= np.linalg.norm(ray)
                z_surf = p[2]
                p = p + 0.01 * ray
                obs.append((b["color"], p[0], p[1], z_surf, f["frame_index"], cam, b["n"]))
    return _cluster(obs)


def _cluster(obs):
    clusters = []
    for o in obs:
        best = None
        for c in clusters:
            if c["color"] != o[0]:
                continue
            dist = np.linalg.norm(np.array(o[1:3]) - c["xy"])
            if dist < 0.03 and (best is None or dist < best[0]):
                best = (dist, c)
        if best is None:
            clusters.append(dict(color=o[0], pts=[o], xy=np.array(o[1:3])))
        else:
            c = best[1]
            c["pts"].append(o)
            w = np.array([3.0 if q[5] == "wrist" else 1.0 for q in c["pts"]])
            xyz = np.array([q[1:3] for q in c["pts"]])
            c["xy"] = (xyz * w[:, None]).sum(0) / w.sum()
    out = []
    for c in clusters:
        fr = [q[4] for q in c["pts"]]
        out.append(dict(color=c["color"], xy=c["xy"], n_obs=len(c["pts"]), first=min(fr), last=max(fr),
                        cams=sorted(set(q[5] for q in c["pts"])), zmed=float(np.median([q[3] for q in c["pts"]])),
                        npx=float(np.median([q[6] for q in c["pts"]]))))
    return out
