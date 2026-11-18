"""Helpers: decode MCP observation packets, back-project depth, simple motion."""
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
    meta = json.loads(next(b["text"] for b in blocks if b.get("type") == "text"))
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
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

class Obs:
    def __init__(self, result):
        self.meta, self.imgs = decode(result)
        m = self.meta
        self.step_id = m.get("step_id")
        self.state = np.array(m.get("state", [0]*8), float)
        self.ended = bool(m.get("episode_ended", False))
        self.limits = m.get("episode_limits", {})
        self.calib = (m.get("camera_intrinsics"), m.get("camera_poses"))

    def rgb(self, cam):
        a = self.imgs.get((cam, "rgb"))
        return None if a is None else a[..., :3]

    def depth(self, cam):
        d = self.imgs.get((cam, "depth"))
        return None if d is None else d.astype(np.float64) * 0.001

    def world_points(self, cam):
        """HxWx3 world points (nan where invalid)."""
        K = np.array(self.calib[0][cam]["matrix"])
        pose = self.calib[1][cam]
        R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
        d = self.depth(cam)
        H, W = d.shape
        v, u = np.mgrid[0:H, 0:W].astype(np.float64)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pc = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1)
        pw = pc @ R.T + t
        pw[d <= 0] = np.nan
        return pw

def observe(ctx):
    return Obs(ctx.call("robomme_observe"))

def step(ctx, rows):
    rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
    res = ctx.call("robomme_step", actions=rows)
    return Obs(res), res

def ended_info(res):
    try:
        meta = json.loads(next(b["text"] for b in res["content"] if b.get("type") == "text"))
    except Exception:
        return None
    return meta

DOWN_AA = np.array([np.pi, 0.0, 0.0])

def rot_err(aa_cur, aa_tgt=DOWN_AA):
    """World-frame rotation vector taking current to target."""
    from scipy.spatial.transform import Rotation as Rr
    rc = Rr.from_rotvec(aa_cur); rt = Rr.from_rotvec(aa_tgt)
    return (rt * rc.inv()).as_rotvec()

def move_to(ctx, obs, target, grip=-1.0, tol=0.004, max_steps=80, vmax=0.5, chunk=4, yaw=None, frames=None):
    """Closed-loop move of the TCP to target xyz. Returns latest Obs."""
    target = np.array(target, float)
    o = obs
    used = 0
    while used < max_steps:
        pos = o.state[:3]
        err = target - pos
        if np.linalg.norm(err) < tol:
            break
        rows = []
        # plan chunk rows assuming moves are achieved
        p = pos.copy()
        for _ in range(chunk):
            e = target - p
            d = np.clip(e / 0.05, -vmax, vmax)
            # scale vector uniformly to keep direction
            n = np.max(np.abs(e / 0.05))
            if n > vmax:
                d = (e / 0.05) * (vmax / n)
            r = rot_err(o.state[3:6], DOWN_AA if yaw is None else yaw_aa(yaw))
            rr = np.clip(r / 0.5, -0.3, 0.3)
            rows.append(list(d) + list(rr if _ == 0 else [0, 0, 0]) + [grip])
            p = p + d * 0.05
            if np.linalg.norm(target - p) < tol:
                break
        o, res = step(ctx, rows)
        used += len(rows)
        if frames is not None:
            frames.append(o)
        if o.ended:
            break
    return o

def yaw_aa(yaw):
    from scipy.spatial.transform import Rotation as Rr
    return (Rr.from_euler("z", yaw) * Rr.from_rotvec(DOWN_AA)).as_rotvec()

def hold(ctx, n, grip):
    o = None
    while n > 0:
        k = min(n, 50)
        o, res = step(ctx, [[0, 0, 0, 0, 0, 0, grip]] * k)
        n -= k
        if o.ended:
            break
    return o

def find_cubes(o, cam):
    """Return list of dict(xy, z, n) for green-ish objects above the table (cubes)."""
    from scipy import ndimage
    rgb = o.rgb(cam).astype(int); P = o.world_points(cam)
    z = P[..., 2]
    green = (rgb[..., 1] > rgb[..., 0] + 25) & (rgb[..., 1] > rgb[..., 2] + 5) & (z > 0.008) & (z < 0.08)
    lab, n = ndimage.label(green)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < 12:
            continue
        pts = P[m]
        out.append(dict(xy=np.nanmedian(pts[:, :2], 0), z=float(np.nanmax(pts[:, 2])), n=int(m.sum()),
                        uv=(float(np.nonzero(m)[1].mean()), float(np.nonzero(m)[0].mean()))))
    return out

def find_discs(o, cam):
    """White highlight discs on the table: bright, low pixels. Returns list of dict(xy, n)."""
    from scipy import ndimage
    rgb = o.rgb(cam).astype(int); P = o.world_points(cam)
    z = P[..., 2]
    white = (rgb.min(-1) > 200) & (z > -0.01) & (z < 0.025) & (np.abs(P[..., 1]) < 0.6) & (P[..., 0] > -0.5)
    lab, n = ndimage.label(white)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < 8:
            continue
        pts = P[m]
        out.append(dict(xy=np.nanmean(pts[:, :2], 0), n=int(m.sum()),
                        uv=(float(np.nonzero(m)[1].mean()), float(np.nonzero(m)[0].mean()))))
    return out

def find_button(o, cam):
    """Button: grey/white square base with top ~0.02-0.04 m; returns xy of white base cluster."""
    from scipy import ndimage
    rgb = o.rgb(cam).astype(int); P = o.world_points(cam)
    z = P[..., 2]
    sat = rgb.max(-1) - rgb.min(-1)
    m0 = (sat < 20) & (rgb.min(-1) > 60) & (z > 0.006) & (z < 0.06) & (P[..., 0] > -0.45)
    lab, n = ndimage.label(m0)
    best = None
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < 15:
            continue
        if best is None or m.sum() > best[1]:
            best = (i, m.sum())
    if best is None:
        return None
    m = lab == best[0]
    pts = P[m]
    return dict(xy=np.nanmedian(pts[:, :2], 0), z=float(np.nanmax(pts[:, 2])), n=int(best[1]))

def cube_tops(o, cam="wrist"):
    """Cube top faces from a top-down view: list of dict(xy, yaw, ztop, n)."""
    import cv2
    from scipy import ndimage
    rgb = o.rgb(cam).astype(int); P = o.world_points(cam)
    z = P[..., 2]
    green = (rgb[..., 1] > rgb[..., 0] + 25) & (rgb[..., 1] > rgb[..., 2] + 5) & (z > 0.008) & (z < 0.08)
    lab, n = ndimage.label(green)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < 12:
            continue
        pts = P[m]
        zt = np.nanmax(pts[:, 2])
        top = pts[(pts[:, 2] > zt - 0.008) & np.isfinite(pts[:, 0])]
        if len(top) < 5:
            top = pts[np.isfinite(pts[:, 0])]
        xy = top[:, :2].astype(np.float32)
        (cx, cy), (w, h), ang = cv2.minAreaRect(xy * 1000.0)
        yaw = np.deg2rad(ang) % (np.pi / 2)
        if yaw > np.pi / 4:
            yaw -= np.pi / 2
        out.append(dict(xy=np.array([cx, cy]) / 1000.0, yaw=float(yaw), ztop=float(zt), n=int(m.sum()),
                        size=(w / 1000.0, h / 1000.0)))
    return out

def set_yaw_target(o, yaw_tool):
    return yaw_aa(yaw_tool)

def rotate_to(ctx, o, aa_tgt, grip, max_steps=30, tol=0.03):
    used = 0
    while used < max_steps:
        r = rot_err(o.state[3:6], aa_tgt)
        if np.linalg.norm(r) < tol:
            break
        rr = np.clip(r / 0.5, -0.4, 0.4)
        o, _ = step(ctx, [[0, 0, 0] + list(rr) + [grip]])
        used += 1
        if o.ended:
            break
    return o

def current_yaw(o):
    from scipy.spatial.transform import Rotation as Rr
    R = Rr.from_rotvec(o.state[3:6]) * Rr.from_rotvec(DOWN_AA).inv()
    return float(R.as_euler("xyz")[2])

def pick_lift(ctx, o, xy, yaw, z_grasp=0.018, z_hover=0.12, z_lift=0.15, place=True):
    """Grasp cube at xy with tool yaw, lift, optionally put back down. Returns (obs, grasped_flag)."""
    # choose the equivalent yaw (mod 90 deg) closest to current tool yaw
    cy = current_yaw(o)
    cands = [yaw + k * np.pi / 2 for k in range(-3, 4)]
    yaw_t = min(cands, key=lambda a: abs(((a - cy + np.pi) % (2 * np.pi)) - np.pi))
    aa = yaw_aa(yaw_t)
    o = move_to(ctx, o, [xy[0], xy[1], z_hover], grip=-1.0, yaw=yaw_t, vmax=0.5)
    o = rotate_to(ctx, o, aa, -1.0)
    o = move_to(ctx, o, [xy[0], xy[1], z_hover], grip=-1.0, yaw=yaw_t, vmax=0.3, tol=0.003)
    o = move_to(ctx, o, [xy[0], xy[1], z_grasp], grip=-1.0, yaw=yaw_t, vmax=0.25, tol=0.004)
    o = hold(ctx, 12, 1.0)
    o = move_to(ctx, o, [xy[0], xy[1], z_lift], grip=1.0, yaw=yaw_t, vmax=0.3, tol=0.01)
    o = hold(ctx, 5, 1.0)
    width = float(o.state[6] + o.state[7])
    grasped = width > 0.02
    log("pick at", np.round(xy, 3), "yaw", round(np.rad2deg(yaw_t), 1), "lifted z", round(o.state[2], 3), "width", round(width, 4))
    if place:
        o = move_to(ctx, o, [xy[0], xy[1], z_grasp + 0.006], grip=1.0, yaw=yaw_t, vmax=0.3, tol=0.006)
        o = hold(ctx, 8, -1.0)
        o = move_to(ctx, o, [xy[0], xy[1], z_hover], grip=-1.0, yaw=yaw_t, vmax=0.4, tol=0.01)
    return o, grasped, width

class FrameObs(Obs):
    """Obs built from a saved frame-manifest entry (offline analysis)."""
    def __init__(self, frame, root="/workspace"):
        self.meta = frame
        self.imgs = {}
        for f in frame["files"]:
            self.imgs[(f["camera"], f["kind"])] = np.array(Image.open(root + "/" + f["path"]))
        self.step_id = frame.get("step_id")
        self.state = np.array(frame.get("state", [0] * 8), float)
        self.ended = False
        self.limits = {}
        self.calib = (frame.get("camera_intrinsics"), frame.get("camera_poses"))

def ring_white(o, cam, centers, r_in=0.03, r_out=0.042, exclude=None, ex_r=0.05):
    """For each cube centre xy, fraction of white pixels among table-height points in an annulus."""
    rgb = o.rgb(cam).astype(int); P = o.world_points(cam)
    ok = np.isfinite(P[..., 0]) & (P[..., 2] < 0.03)
    white = rgb.min(-1) > 195
    if exclude is not None:
        ok = ok & (np.hypot(P[..., 0] - exclude[0], P[..., 1] - exclude[1]) > ex_r)
    res = []
    for c in centers:
        d = np.hypot(P[..., 0] - c[0], P[..., 1] - c[1])
        m = ok & (d > r_in) & (d < r_out)
        n = int(m.sum())
        res.append((float(white[m].mean()) if n >= 10 else float("nan"), n))
    return res

def find_cap(o, cam="wrist"):
    """Button cap: low-saturation grey pixels with z in [0.026, 0.06]; largest cluster centroid."""
    from scipy import ndimage
    rgb = o.rgb(cam).astype(int); P = o.world_points(cam)
    z = P[..., 2]
    sat = rgb.max(-1) - rgb.min(-1)
    m0 = (sat < 18) & (rgb.min(-1) > 60) & (z > 0.026) & (z < 0.06) & (P[..., 0] > -0.5)
    lab, n = ndimage.label(m0)
    best = None
    for i in range(1, n + 1):
        k = int((lab == i).sum())
        if k >= 6 and (best is None or k > best[1]):
            best = (i, k)
    if best is None:
        return None
    pts = P[lab == best[0]]
    return dict(xy=np.nanmedian(pts[:, :2], 0), z=float(np.nanmax(pts[:, 2])), n=best[1])

# ---- colour-agnostic perception (v2) ----
def cube_tops2(o, cam="wrist", zmin_top=0.036, zmax=0.07):
    """Cube top faces by height only (cube top ~0.041, button cap ~0.034)."""
    import cv2
    from scipy import ndimage
    P = o.world_points(cam)
    z = P[..., 2]
    ok = np.isfinite(z) & (z > zmin_top) & (z < zmax) & (P[..., 0] > -0.55) & (P[..., 0] < 0.3) & (np.abs(P[..., 1]) < 0.5)
    lab, n = ndimage.label(ok)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < 8:
            continue
        pts = P[m]
        xy = pts[:, :2].astype(np.float32)
        (cx, cy), (w, h), ang = cv2.minAreaRect(xy * 1000.0)
        w, h = w / 1000.0, h / 1000.0
        yaw = np.deg2rad(ang) % (np.pi / 2)
        if yaw > np.pi / 4:
            yaw -= np.pi / 2
        out.append(dict(xy=np.array([cx, cy]) / 1000.0, yaw=float(yaw), ztop=float(np.nanmax(pts[:, 2])),
                        n=int(m.sum()), size=(w, h)))
    return out

def find_cap2(o, cubes, cam="wrist"):
    """Button cap: pixels with z in [0.027,0.0365] away from cube centres; largest cluster."""
    from scipy import ndimage
    P = o.world_points(cam)
    z = P[..., 2]
    ok = np.isfinite(z) & (z > 0.027) & (z < 0.0365) & (P[..., 0] > -0.55) & (P[..., 0] < 0.3) & (np.abs(P[..., 1]) < 0.5)
    for c in cubes:
        ok &= np.hypot(P[..., 0] - c["xy"][0], P[..., 1] - c["xy"][1]) > 0.034
    lab, n = ndimage.label(ok)
    best = None
    for i in range(1, n + 1):
        k = int((lab == i).sum())
        if k >= 6 and (best is None or k > best[1]):
            best = (i, k)
    if best is None:
        return None
    pts = P[lab == best[0]]
    return dict(xy=np.nanmedian(pts[:, :2], 0), z=float(np.nanmax(pts[:, 2])), n=best[1])
