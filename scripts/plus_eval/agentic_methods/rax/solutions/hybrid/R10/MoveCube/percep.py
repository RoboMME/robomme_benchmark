"""Perception: depth back-projection, segmentation of objects on the table."""
import json, math
from pathlib import Path
import numpy as np
from PIL import Image

WS = Path("/workspace")

def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def load_frame(frame, cam="front"):
    files = {(f["camera"], f["kind"]): f["path"] for f in frame["files"]}
    rgb = np.array(Image.open(WS / files[(cam, "rgb")]).convert("RGB")).astype(np.int16)
    dep = np.array(Image.open(WS / files[(cam, "depth")])).astype(np.float32) * 0.001
    return rgb, dep

def backproject(dep, intr, pose):
    K = np.array(intr["matrix"]); fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = dep.shape
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    d = dep
    pc = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1)
    R = quat_to_mat(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    pw = pc @ R.T + t
    pw[d <= 0] = np.nan
    return pw

def project(pw, intr, pose):
    """world points (N,3) -> pixel (u,v)"""
    K = np.array(intr["matrix"]); fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    R = quat_to_mat(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    pc = (np.asarray(pw) - t) @ R
    d = -pc[..., 2]
    u = pc[..., 0] * fx / d + cx; v = -pc[..., 1] * fy / d + cy
    return np.stack([u, v], -1)

def rgb2hsv(rgb):
    r, g, b = [rgb[..., i] / 255.0 for i in range(3)]
    mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
    df = mx - mn + 1e-9
    h = np.where(mx == r, ((g - b) / df) % 6, np.where(mx == g, (b - r) / df + 2, (r - g) / df + 4)) * 60
    s = np.where(mx > 0, df / (mx + 1e-9), 0)
    return h, s, mx

def wood_mask(rgb):
    h, s, v = rgb2hsv(rgb)
    return (h >= 5) & (h <= 40) & (s > 0.35) & (s < 0.95) & (v > 0.25)

def color_class(rgb):
    """coarse per-pixel color label"""
    h, s, v = rgb2hsv(rgb)
    lab = np.full(rgb.shape[:2], "other", dtype=object)
    lab[(s < 0.18) & (v > 0.55)] = "white"
    lab[(s < 0.25) & (v <= 0.55)] = "gray"
    lab[(h >= 160) & (h < 200) & (s > 0.3)] = "teal"
    lab[(h >= 195) & (h < 230) & (s > 0.25)] = "blue"
    lab[(h >= 45) & (h < 70) & (s > 0.5)] = "yellow"
    lab[((h < 8) | (h > 345)) & (s > 0.6) & (v > 0.4)] = "red"
    lab[(h >= 260) & (h < 320) & (s > 0.2)] = "purple"
    return lab

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None

def label(mask):
    if _ndi is not None:
        lab, n = _ndi.label(mask, structure=np.ones((3, 3)))
        return lab, n
    H, W = mask.shape; lab = np.zeros((H, W), int); n = 0
    for y in range(H):
        for x in range(W):
            if mask[y, x] and lab[y, x] == 0:
                n += 1; st = [(y, x)]; lab[y, x] = n
                while st:
                    cy, cx = st.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            yy, xx = cy + dy, cx + dx
                            if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and lab[yy, xx] == 0:
                                lab[yy, xx] = n; st.append((yy, xx))
    return lab, n

CLASSES = ["white", "gray", "teal", "blue", "yellow", "red", "purple", "other"]

def scene_blobs(rgb, dep, intr, pose, robot_mask=None, min_px=12):
    pw = backproject(dep, intr, pose)
    z = pw[..., 2]
    wm = wood_mask(rgb)
    valid = np.isfinite(z) & (z > -0.02) & (z < 0.25) & (np.abs(pw[..., 0]) < 1.0) & (np.abs(pw[..., 1]) < 1.0)
    obj = valid & ((~wm) | (z > 0.006))
    if robot_mask is not None:
        obj &= ~robot_mask
    lab, n = label(obj)
    cc = color_class(rgb)
    blobs = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < min_px:
            continue
        pts = pw[m]
        hist = np.array([(cc[m] == c).mean() for c in CLASSES])
        xy = pts[:, :2]
        c = xy.mean(0)
        if len(xy) > 2:
            ev, evec = np.linalg.eigh(np.cov((xy - c).T))
            ax = evec[:, -1]
            proj = (xy - c) @ ax
            length = proj.max() - proj.min()
        else:
            ax = np.array([1, 0.]); length = 0
        blobs.append(dict(id=i, mask=m, n=int(m.sum()), hist=hist, center=c, zmax=float(np.nanmax(pts[:, 2])),
                          axis=ax, length=float(length), pts=pts))
    return blobs, lab, pw

def describe(b):
    return dict(n=b["n"], hist={c: round(float(h), 2) for c, h in zip(CLASSES, b["hist"]) if h > 0.02},
                center=[round(float(x), 3) for x in b["center"]], zmax=round(b["zmax"], 3), length=round(b["length"], 3))
