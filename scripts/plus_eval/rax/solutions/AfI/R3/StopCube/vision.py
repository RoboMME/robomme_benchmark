"""Image decoding, camera geometry and cube/target detection (no robot I/O)."""
import base64, io, json
import numpy as np
from PIL import Image


def parse_reply(reply):
    """Return (meta, imgs) where imgs[(camera, kind)] is a numpy array (rgb uint8 / depth metres)."""
    blocks = reply["content"]
    meta = json.loads(blocks[0]["text"])
    imgs = {}
    for att in meta.get("attachments", []):
        b = blocks[att["content_index"]]
        if b.get("type") != "image":
            continue
        im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
        if att["kind"] == "rgb":
            imgs[(att["camera"], "rgb")] = np.array(im.convert("RGB"))
        else:
            mpu = meta["depth"][att["camera"]].get("meters_per_unit", 0.001)
            imgs[(att["camera"], "depth")] = np.array(im).astype(np.float64) * mpu
    return meta, imgs


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Camera:
    def __init__(self, meta, cam):
        K = np.array(meta["camera_intrinsics"][cam]["matrix"])
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        p = meta["camera_poses"][cam]
        self.t = np.array(p["position_m"])
        self.R = quat_to_R(p["quaternion_wxyz"])  # camera(OpenGL) -> world

    def backproject(self, u, v, d):
        """Pixel (u,v) with linear depth d (m) -> world point. OpenGL: looks along -Z, +Y up."""
        xc = (u - self.cx) / self.fx * d
        yc = -(v - self.cy) / self.fy * d
        pc = np.array([xc, yc, -d])
        return self.R @ pc + self.t

    def project(self, w):
        pc = self.R.T @ (np.asarray(w) - self.t)
        d = -pc[2]
        u = self.fx * pc[0] / d + self.cx
        v = -self.fy * pc[1] / d + self.cy
        return u, v, d


def cube_mask(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    # beige cube: top (218,205,170), side (140,131,108); wood has b << g, purple has b > r.
    return (b > 95) & (r >= b + 25) & (g >= b + 15) & (abs(r - g) < 30)


def detect_cube(rgb, depth, cam):
    m = cube_mask(rgb)
    n = int(m.sum())
    if n < 6:
        return None
    vs, us = np.nonzero(m)
    u, v = float(np.median(us)), float(np.median(vs))
    pts = np.array([cam.backproject(uu, vv, depth[vv, uu]) for uu, vv in zip(us, vs)])
    w = np.median(pts, axis=0)
    return {"u": u, "v": v, "n": n, "world": w.tolist()}


def target_mask(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (b > 150) & (r > g + 50) & (b > g + 70)


def detect_target(rgb, depth, cam):
    """Purple/white target disk; restricted to disk-top height (z 0.008-0.024) so a purple cube (top 0.031) is excluded."""
    m = target_mask(rgb)
    W = world_map(cam, depth)
    with np.errstate(invalid="ignore"):
        m = m & (W[..., 2] > 0.008) & (W[..., 2] < 0.024)
    n = int(m.sum())
    if n < 6:
        return None
    vs, us = np.nonzero(m)
    pts = W[m]
    return {"u": float(us.mean()), "v": float(vs.mean()), "n": n, "world": np.median(pts, axis=0).tolist()}


def world_map(cam, depth):
    """Vectorized back-projection: HxWx3 world points (invalid depth -> nan)."""
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
    d = depth.astype(np.float64)
    xc = (uu - cam.cx) / cam.fx * d
    yc = -(vv - cam.cy) / cam.fy * d
    pc = np.stack([xc, yc, -d], axis=-1)
    w = pc @ cam.R.T + cam.t
    w[d <= 0] = np.nan
    return w


def low_sat(rgb, tol=18):
    x = rgb.astype(int)
    return (x.max(axis=-1) - x.min(axis=-1)) <= tol


def detect_button_front(rgb, depth, cam, target_xy):
    """Button = low-saturation (grey cap / white base) blob on the table, away from the target."""
    W = world_map(cam, depth)
    z = W[..., 2]
    with np.errstate(invalid="ignore"):
        m = low_sat(rgb) & (z > 0.008) & (z < 0.05) & (W[..., 0] > -0.45) & (W[..., 0] < 0.40)
        if target_xy is not None:
            dt = np.hypot(W[..., 0] - target_xy[0], W[..., 1] - target_xy[1])
            m &= dt > 0.05
    if m.sum() < 8:
        return None
    pts = W[m]
    # largest cluster by iterative median trimming (button is a compact ~6 cm blob)
    c = np.median(pts[:, :2], axis=0)
    for _ in range(4):
        dd = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
        sel = pts[dd < 0.05]
        if len(sel) < 8:
            break
        c = np.median(sel[:, :2], axis=0)
    dd = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    sel = pts[dd < 0.05]
    cap = sel[sel[:, 2] > 0.024]
    if len(cap) >= 5:
        c = np.median(cap[:, :2], axis=0)
        top = float(np.percentile(cap[:, 2], 80))
    else:
        top = float(np.max(sel[:, 2])) if len(sel) else 0.033
    return {"xy": c.tolist(), "cap_top": top, "n": int(len(sel)), "n_cap": int(len(cap))}


def detect_cap_wrist(rgb, depth, cam, approx_xy, approx_top):
    """Refine cap centre/top from a wrist view: low-saturation pixels near approx_top height."""
    W = world_map(cam, depth)
    z = W[..., 2]
    with np.errstate(invalid="ignore"):
        dd = np.hypot(W[..., 0] - approx_xy[0], W[..., 1] - approx_xy[1])
        m = low_sat(rgb, 25) & (np.abs(z - approx_top) < 0.008) & (dd < 0.05)
    if m.sum() < 30:
        return None
    pts = W[m]
    return {"xy": np.median(pts[:, :2], axis=0).tolist(), "cap_top": float(np.percentile(pts[:, 2], 80)),
            "n": int(m.sum())}


def detect_cube_geom(rgb, depth, cam, button_xy, z_lo=0.020, z_hi=0.07):
    """Fallback: anything standing above the target-disk height, away from the button/arm."""
    W = world_map(cam, depth)
    z = W[..., 2]
    with np.errstate(invalid="ignore"):
        m = (z > z_lo) & (z < z_hi)
        if button_xy is not None:
            m &= np.hypot(W[..., 0] - button_xy[0], W[..., 1] - button_xy[1]) > 0.07
        m &= (W[..., 0] > -0.45) & (W[..., 0] < 0.40) & (np.abs(W[..., 1]) < 0.5)
    n = int(m.sum())
    if n < 6:
        return None
    pts = W[m]
    c = np.median(pts, axis=0)
    dd = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    pts = pts[dd < 0.04]
    if len(pts) < 6:
        return None
    return {"world": np.median(pts, axis=0).tolist(), "n": int(len(pts))}


def detect_target_geom(rgb, depth, cam, button_xy, cube_xy):
    """Fallback target: thin raised disk (top ~0.016 m), away from the button and the cube."""
    W = world_map(cam, depth)
    z = W[..., 2]
    with np.errstate(invalid="ignore"):
        m = (z > 0.010) & (z < 0.021) & (W[..., 0] > -0.45) & (W[..., 0] < 0.40) & (np.abs(W[..., 1]) < 0.5)
        if button_xy is not None:
            m &= np.hypot(W[..., 0] - button_xy[0], W[..., 1] - button_xy[1]) > 0.07
        if cube_xy is not None:
            m &= np.hypot(W[..., 0] - cube_xy[0], W[..., 1] - cube_xy[1]) > 0.035
    if m.sum() < 8:
        return None
    pts = W[m]
    c = np.median(pts[:, :2], axis=0)
    for _ in range(3):
        sel = pts[np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) < 0.03]
        if len(sel) < 8:
            break
        c = np.median(sel[:, :2], axis=0)
    return {"xy": c.tolist(), "n": int(m.sum())}
