"""Perception for VideoUnmask: cube colours from the conditioning video, containers from live front view."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

WS = Path("/workspace")

# PIL HSV hue (0-255) ranges
COLOR_HUES = {
    "red": [(0, 10), (236, 256)],
    "orange": [(14, 34)],
    "yellow": [(35, 56)],
    "green": [(60, 125)],
    "blue": [(140, 190)],
    "purple": [(191, 235)],
}


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(us, vs, ds, K, pose):
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    us, vs, ds = np.asarray(us, float), np.asarray(vs, float), np.asarray(ds, float)
    cam = np.stack([(us - cx) * ds / fx, -(vs - cy) * ds / fy, -ds], -1)
    R = quat_to_R(pose["quaternion_wxyz"])
    return cam @ R.T + np.asarray(pose["position_m"])


def load_rgb_hsv_depth(paths):
    rgb = Image.open(WS / paths[("front", "rgb")]).convert("RGB")
    hsv = np.asarray(rgb.convert("HSV")).astype(int)
    depth = np.asarray(Image.open(WS / paths[("front", "depth")])).astype(float) * 0.001
    return np.asarray(rgb).astype(int), hsv, depth


def components(mask, min_size=15):
    """Simple 4-connected components; returns list of (ys, xs)."""
    H, W = mask.shape
    lab = np.zeros(mask.shape, int)
    out = []
    cur = 0
    ys0, xs0 = np.nonzero(mask)
    for y0, x0 in zip(ys0, xs0):
        if lab[y0, x0]:
            continue
        cur += 1
        stack = [(y0, x0)]
        lab[y0, x0] = cur
        pts = []
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = cur
                    stack.append((yy, xx))
        if len(pts) >= min_size:
            a = np.array(pts)
            out.append((a[:, 0], a[:, 1]))
    return out


def color_mask(hsv, color):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = np.zeros(h.shape, bool)
    for lo, hi in COLOR_HUES[color]:
        m |= (h >= lo) & (h < hi)
    smin = 195 if color in ("red", "orange") else 100  # table hue ~10-20, sat ~140-170
    return m & (s > smin) & (v > 60)


def frame_files(frame):
    return {(f["camera"], f["kind"]): f["path"] for f in frame["files"]}


def cube_tracks(manifest_path, colors, log=print):
    """For each colour, the world xy of its cube in the last video frame where it is visible."""
    man = json.loads((WS / manifest_path).read_text())
    frames = sorted([f for f in man["frames"] if f["phase"] == "video_demo"], key=lambda f: f["frame_index"])
    result = {}
    for c in colors:
        result[c] = None
    for fr in frames:
        files = frame_files(fr)
        if ("front", "rgb") not in files or ("front", "depth") not in files:
            continue
        _, hsv, depth = load_rgb_hsv_depth(files)
        K = fr["camera_intrinsics"]["front"]["matrix"]
        pose = fr["camera_poses"]["front"]
        for c in colors:
            comps = components(color_mask(hsv, c), min_size=12)
            if not comps:
                continue
            ys, xs = max(comps, key=lambda p: len(p[0]))
            d = depth[ys, xs]
            ok = d > 0
            if ok.sum() < 5:
                continue
            P = backproject(xs[ok], ys[ok], d[ok], K, pose)
            xyz = np.median(P, 0)
            result[c] = {"frame": fr["frame_index"], "uv": [float(xs.mean()), float(ys.mean())],
                         "xyz": xyz.tolist(), "n": int(len(xs))}
    for c in colors:
        log(f"cube {c}: {result[c]}")
    return result


def find_containers(rgb, hsv, depth, K, pose, log=print):
    """White/grey low-saturation blobs above the table. Returns list of dicts with world xyz (top centre)."""
    s, v = hsv[..., 1], hsv[..., 2]
    m = (s < 60) & (v > 150)
    H, W = m.shape
    # world points for all pixels to reject robot/background
    vv, uu = np.mgrid[0:H, 0:W]
    P = backproject(uu.ravel(), vv.ravel(), depth.ravel(), K, pose).reshape(H, W, 3)
    valid = depth > 0
    # table height estimate: median z of saturated-ish table pixels
    table = valid & (s > 80) & (P[..., 2] < 0.2)
    tz = float(np.median(P[..., 2][table])) if table.sum() > 100 else 0.0
    m &= valid & (P[..., 2] > tz + 0.01) & (P[..., 2] < tz + 0.25)
    comps = components(m, min_size=25)
    out = []
    for ys, xs in comps:
        pts = P[ys, xs]
        top = pts[pts[:, 2] >= np.percentile(pts[:, 2], 60)]
        c = np.median(top, 0)
        out.append({"uv": [float(xs.mean()), float(ys.mean())], "xyz": c.tolist(), "n": int(len(xs)),
                    "zmax": float(pts[:, 2].max())})
    log(f"table z {tz:.4f}; containers: " + "; ".join(
        f"uv=({o['uv'][0]:.0f},{o['uv'][1]:.0f}) xyz=({o['xyz'][0]:.3f},{o['xyz'][1]:.3f},{o['xyz'][2]:.3f}) n={o['n']}" for o in out))
    return out, tz
