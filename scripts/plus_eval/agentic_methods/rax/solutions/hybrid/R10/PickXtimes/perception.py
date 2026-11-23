"""Observation decoding, depth back-projection and colour segmentation."""
import base64, io, json
import numpy as np
from PIL import Image


def parse_reply(reply):
    content = reply.get("content", [])
    meta = {}
    for b in content:
        if b.get("type") == "text":
            try:
                meta = json.loads(b["text"])
            except Exception:
                meta = {}
            break
    imgs = {}
    for att in meta.get("attachments", []) or []:
        i = att.get("content_index")
        if i is None or i >= len(content):
            continue
        b = content[i]
        if b.get("type") != "image":
            continue
        im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
        arr = np.array(im)
        if att.get("kind") == "depth":
            mpu = (meta.get("depth", {}).get(att["camera"], {}) or {}).get("meters_per_unit", 0.001)
            arr = arr.astype(np.float64) * mpu
        else:
            arr = np.array(im.convert("RGB"))
        imgs[(att["camera"], att["kind"])] = arr
    return meta, imgs


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(meta, depth, cam):
    """World xyz (H,W,3) for every pixel; invalid depth -> nan."""
    K = np.array(meta["camera_intrinsics"][cam]["matrix"], dtype=float)
    pose = meta["camera_poses"][cam]
    R = quat_to_mat(pose["quaternion_wxyz"])
    t = np.array(pose["position_m"], dtype=float)
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W].astype(float)
    d = depth.astype(float)
    xc = (u - K[0, 2]) * d / K[0, 0]
    yc = -(v - K[1, 2]) * d / K[1, 1]
    zc = -d
    P = np.stack([xc, yc, zc], -1) @ R.T + t
    P[d <= 0] = np.nan
    return P


def hsv(rgb):
    a = rgb.astype(float) / 255.0
    mx = a.max(-1); mn = a.min(-1)
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    return mx, s


def masks(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    val, sat = hsv(rgb)
    green = (g > 120) & (g - r > 60) & (g - b > 60)
    # purple / magenta ring of the target
    purple = (r > 110) & (b > 130) & (g < 0.75 * np.minimum(r, b)) & (sat > 0.3)
    red = (r > 120) & (r - g > 70) & (r - b > 70)
    blue = (b > 100) & (b - r > 60) & (b - g > 40)
    yellow = (r > 130) & (g > 130) & (b < 0.5 * np.minimum(r, g))
    return {"green": green, "red": red, "blue": blue, "yellow": yellow, "purple": purple, "sat": sat, "val": val}


def largest_blob(mask):
    """Return mask of largest 4-connected component (simple BFS, small images)."""
    H, W = mask.shape
    lab = np.zeros((H, W), int)
    best, bestn, cur = 0, 0, 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]:
            continue
        cur += 1
        stack = [(y0, x0)]; lab[y0, x0] = cur; n = 0
        while stack:
            y, x = stack.pop(); n += 1
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = cur; stack.append((yy, xx))
        if n > bestn:
            best, bestn = cur, n
    return lab == best if bestn else np.zeros_like(mask)


def blob_stats(mask, P):
    pts = P[mask]
    pts = pts[np.all(np.isfinite(pts), 1)]
    if len(pts) < 5:
        return None
    return {"n": int(len(pts)), "center": np.median(pts, 0), "top": float(np.percentile(pts[:, 2], 90)),
            "pts": pts}


def yaw_of_square(pts_xy):
    """Estimate square yaw modulo 90deg (rad, in [-pi/4, pi/4)) from xy points."""
    c = pts_xy - pts_xy.mean(0)
    best, by = None, 0.0
    for a in np.deg2rad(np.arange(-45, 45, 1.0)):
        ca, sa = np.cos(a), np.sin(a)
        x = c[:, 0] * ca + c[:, 1] * sa; y = -c[:, 0] * sa + c[:, 1] * ca
        area = (np.percentile(x, 98) - np.percentile(x, 2)) * (np.percentile(y, 98) - np.percentile(y, 2))
        if best is None or area < best:
            best, by = area, a
    return float(by)


# Bullseye radial template (metres): purple bands and white bands, outer radius 0.04
_BANDS = [(0.0, 0.0085, 1.0), (0.0085, 0.0155, -1.0), (0.0155, 0.0245, 1.0), (0.0245, 0.031, -1.0),
          (0.031, 0.0405, 1.0), (0.0405, 0.044, 0.0), (0.044, 0.09, -1.0)]


def _score(pts, centers):
    d = np.linalg.norm(pts[None, :, :] - centers[:, None, :], axis=-1)
    w = np.zeros_like(d)
    for a, b, v in _BANDS:
        w[(d >= a) & (d < b)] = v
    return w.sum(1)


def fit_target(pts_xy, rough, span=0.05):
    """Find bullseye centre maximising radial-template agreement of purple points."""
    pts = pts_xy[np.linalg.norm(pts_xy - rough, axis=1) < span + 0.045]
    if len(pts) < 10:
        return rough, 0.0
    if len(pts) > 600:
        pts = pts[np.random.RandomState(0).choice(len(pts), 600, replace=False)]
    g = np.arange(-span, span + 1e-9, 0.002)
    C = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2) + rough
    s = _score(pts, C)
    best = C[np.argmax(s)]
    g2 = np.arange(-0.003, 0.0031, 0.0005)
    C2 = np.stack(np.meshgrid(g2, g2), -1).reshape(-1, 2) + best
    s2 = _score(pts, C2)
    return C2[np.argmax(s2)], float(s2.max() / max(len(pts), 1))
