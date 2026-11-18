"""Perception helpers: decode observation, back-project depth, colour masks."""
import base64, io, json
import numpy as np
from PIL import Image

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def parse_reply(res):
    """MCP result -> (meta dict, {(cam,kind): ndarray})"""
    blocks = res.get("content", [])
    meta = json.loads(blocks[0]["text"])
    imgs = {}
    for a in meta.get("attachments", []):
        b = blocks[a["content_index"]]
        data = base64.b64decode(b["data"])
        im = Image.open(io.BytesIO(data))
        arr = np.array(im)
        imgs[(a["camera"], a["kind"])] = arr
    return meta, imgs

def load_frame(root, frame):
    files = {(f["camera"], f["kind"]): f["path"] for f in frame["files"]}
    out = {}
    for k, p in files.items():
        im = Image.open(f"{root}/{p}")
        out[k] = np.array(im.convert("RGB")) if k[1] == "rgb" else np.array(im)
    return out

def world_points(depth_u16, intr, pose):
    """Full (H,W,3) world points from depth image; nan where invalid."""
    K = np.array(intr["matrix"]); fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    H, W = depth_u16.shape
    d = depth_u16.astype(np.float64) * 0.001
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    pc = np.stack([(u-cx)*d/fx, -(v-cy)*d/fy, -d], -1)
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    pw = pc @ R.T + t
    pw[d == 0] = np.nan
    return pw

def hsv(rgb):
    return np.array(Image.fromarray(rgb.astype(np.uint8)).convert("HSV")).astype(np.int32)

def mask_red(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (r > 150) & (g < 60) & (b < 60)

def mask_purple(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (b > 120) & (r > 110) & (g < r - 30) & (g < b - 30)

def blob_point(mask, P, min_px=3):
    """median world point of mask pixels with valid depth"""
    if mask.sum() < min_px:
        return None
    pts = P[mask]
    pts = pts[~np.isnan(pts).any(1)]
    if len(pts) < min_px:
        return None
    return np.median(pts, 0), int(mask.sum())
