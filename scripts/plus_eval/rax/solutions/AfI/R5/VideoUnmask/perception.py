"""Perception helpers for VideoUnmask: colour cubes in conditioning video -> world XY;
white containers in live front view -> world XY clusters."""
import json, math
from pathlib import Path
import numpy as np
from PIL import Image

WS = Path("/workspace")

COLORS = {  # hue ranges (0..1)
    "red": [(0.0, 0.04), (0.93, 1.01)],
    "orange": [(0.06, 0.135)],
    "yellow": [(0.135, 0.22)],
    "green": [(0.22, 0.45)],
    "cyan": [(0.45, 0.54)],
    "blue": [(0.54, 0.75)],
    "purple": [(0.75, 0.93)],
}


def rgb_to_hsv_np(img):
    a = img.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1); mn = a.min(-1); d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0)
    h = np.zeros_like(mx)
    dd = np.maximum(d, 1e-6)
    hr = ((g - b) / dd) % 6
    hg = (b - r) / dd + 2
    hb = (r - g) / dd + 4
    h = np.where(mx == r, hr, np.where(mx == g, hg, hb)) / 6.0
    h = np.where(d < 1e-6, 0, h)
    return h, s, mx


def color_masks(img, s_min=0.85, v_min=0.25):
    h, s, v = rgb_to_hsv_np(img)
    sat = (s >= s_min) & (v >= v_min)
    out = {}
    for name, ranges in COLORS.items():
        m = np.zeros_like(sat)
        for lo, hi in ranges:
            m |= (h >= lo) & (h < hi)
        out[name] = sat & m
    return out


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(u, v, d, K, pose):
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    cam = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1)
    R = quat_to_mat(pose["quaternion_wxyz"])
    return cam @ R.T + np.array(pose["position_m"])


def world_points(depth_m, K, pose):
    H, W = depth_m.shape
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
    return backproject(uu + 0.5 * 0, vv, depth_m, K, pose)


def load_manifest(manifest_path):
    return json.loads((WS / manifest_path).read_text())


def frame_files(frame):
    return {(x["camera"], x["kind"]): x["path"] for x in frame["files"]}


def load_rgb(path):
    with Image.open(WS / path) as im:
        return np.array(im.convert("RGB"))


def load_depth(path, scale=0.001):
    with Image.open(WS / path) as im:
        return np.array(im).astype(np.float64) * scale


def label(mask):
    try:
        from scipy import ndimage
        return ndimage.label(mask)
    except Exception:
        pass
    H, W = mask.shape
    lab = np.zeros((H, W), int); n = 0
    for y0 in range(H):
        for x0 in range(W):
            if mask[y0, x0] and lab[y0, x0] == 0:
                n += 1; st = [(y0, x0)]; lab[y0, x0] = n
                while st:
                    y, x = st.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and lab[yy, xx] == 0:
                            lab[yy, xx] = n; st.append((yy, xx))
    return lab, n


def dilate(g):
    o = g.copy()
    o[1:] |= g[:-1]; o[:-1] |= g[1:]; o[:, 1:] |= g[:, :-1]; o[:, :-1] |= g[:, 1:]
    return o


def blobs(mask, min_px=6):
    lab, n = label(mask)
    res = []
    for k in range(1, n + 1):
        mm = lab == k
        c = int(mm.sum())
        if c >= min_px:
            yy, xx = np.nonzero(mm)
            res.append((c, mm))
    res.sort(key=lambda t: -t[0])
    return res


def cube_positions_from_video(manifest, log=print):
    """Return {color: dict(xy, frame, px, world)} using the last video frame where the
    colour is visible in the front camera."""
    frames = sorted([f for f in manifest["frames"] if f["phase"] == "video_demo"],
                    key=lambda f: f["frame_index"])
    last = {}
    first = {}
    for fr in frames:
        files = frame_files(fr)
        if ("front", "rgb") not in files:
            continue
        img = load_rgb(files[("front", "rgb")])
        masks = color_masks(img)
        for name, m in masks.items():
            bl = blobs(m, 6)
            if not bl:
                continue
            c, mm = bl[0]
            if len(bl) > 1:
                log(f"warn: {name} has {len(bl)} blobs at frame {fr['frame_index']}")
            last[name] = (fr, mm, c)
            first.setdefault(name, fr["frame_index"])
    out = {}
    for name, (fr, mm, c) in last.items():
        files = frame_files(fr)
        dep = load_depth(files[("front", "depth")], fr["depth"]["front"].get("meters_per_unit", 0.001))
        K = fr["camera_intrinsics"]["front"]["matrix"]
        pose = fr["camera_poses"]["front"]
        P = world_points(dep, K, pose)
        sel = mm & (dep > 0)
        pts = P[sel]
        if len(pts) == 0:
            continue
        w = np.median(pts, 0)
        yy, xx = np.nonzero(mm)
        out[name] = {"xy": [float(w[0]), float(w[1])], "z": float(w[2]),
                     "frame": fr["frame_index"], "first_frame": first[name],
                     "px": [float(xx.mean()), float(yy.mean())], "npx": c}
    return out


def white_clusters(img, dep, K, pose, z_lo=0.005, z_hi=0.15, xy_box=(-0.35, 0.45, -0.45, 0.45), link=0.025):
    """Cluster low-saturation bright pixels near the table into objects. Returns list of
    dicts with centroid xy, top z, npts, sorted by npts desc."""
    h, s, v = rgb_to_hsv_np(img)
    m = (s < 0.18) & (v > 0.35) & (dep > 0)
    P = world_points(dep, K, pose)
    pts = P[m]
    keep = (pts[:, 2] > z_lo) & (pts[:, 2] < z_hi) & (pts[:, 0] > xy_box[0]) & (pts[:, 0] < xy_box[1]) \
        & (pts[:, 1] > xy_box[2]) & (pts[:, 1] < xy_box[3])
    pts = pts[keep]
    if len(pts) == 0:
        return []
    # grid-based connected components in XY (1 cm cells)
    cell = 0.01
    ij = np.floor(pts[:, :2] / cell).astype(int)
    mn = ij.min(0)
    g = np.zeros(tuple(ij.max(0) - mn + 1), bool)
    g[tuple((ij - mn).T)] = True
    g2 = dilate(g)
    lab, n = label(g2)
    labels = lab[tuple((ij - mn).T)]
    res = []
    for k in range(1, n + 1):
        q = pts[labels == k]
        if len(q) < 15:
            continue
        res.append({"xy": [float(np.median(q[:, 0])), float(np.median(q[:, 1]))],
                    "ztop": float(np.percentile(q[:, 2], 95)), "n": int(len(q)),
                    "xmin": float(q[:, 0].min()), "xmax": float(q[:, 0].max()),
                    "ymin": float(q[:, 1].min()), "ymax": float(q[:, 1].max())})
    res.sort(key=lambda r: -r["n"])
    return res


# ---------------- v2: multi-instance, two-camera cube memory ----------------
CUBE_Z = (0.003, 0.06)


def color_masks_v2(img):
    h, s, v = rgb_to_hsv_np(img)
    out = {}
    sat = (s >= 0.85) & (v >= 0.25)
    for name, ranges in COLORS.items():
        m = np.zeros_like(sat)
        for lo, hi in ranges:
            m |= (h >= lo) & (h < hi)
        out[name] = sat & m
    # lower-saturation colours far from the table hue
    mid = (s >= 0.4) & (v >= 0.3)
    out["purple"] |= mid & (h >= 0.72) & (h < 0.95)
    out["blue"] |= mid & (h >= 0.55) & (h < 0.72)
    out["green"] |= mid & (h >= 0.22) & (h < 0.45)
    out["cyan"] |= mid & (h >= 0.45) & (h < 0.55)
    out["black"] = (v < 0.27) & (s < 0.35)
    return out


def detect_cubes_frame(fr, cam, min_px=8):
    files = frame_files(fr)
    if (cam, "rgb") not in files or (cam, "depth") not in files:
        return []
    img = load_rgb(files[(cam, "rgb")])
    dep = load_depth(files[(cam, "depth")], fr["depth"][cam].get("meters_per_unit", 0.001))
    P = world_points(dep, fr["camera_intrinsics"][cam]["matrix"], fr["camera_poses"][cam])
    valid = dep > 0
    dets = []
    for name, m in color_masks_v2(img).items():
        if not m.any():
            continue
        lab, n = label(m)
        for k in range(1, n + 1):
            mm = lab == k
            c = int(mm.sum())
            if c < min_px:
                continue
            q = P[mm & valid]
            q = q[(q[:, 2] > CUBE_Z[0]) & (q[:, 2] < CUBE_Z[1])]
            if len(q) < max(5, 0.4 * c):
                continue
            ext = q[:, :2].max(0) - q[:, :2].min(0)
            if ext.max() > 0.08:
                continue
            dets.append({"color": name, "xy": np.median(q[:, :2], 0), "z": float(np.median(q[:, 2])),
                         "n": len(q), "frame": fr["frame_index"], "cam": cam})
    return dets


def _cluster(ds, link):
    clusters = []
    for d in ds:
        best = None
        for c in clusters:
            dist = np.linalg.norm(np.median(np.array([x["xy"] for x in c]), 0) - d["xy"])
            if dist < link and (best is None or dist < best[0]):
                best = (dist, c)
        if best:
            best[1].append(d)
        else:
            clusters.append([d])
    return clusters


def cube_instances(manifest, log=print, stride=1, link=0.03):
    """Static cube instances: spatial clusters of detections over all video frames and both
    cameras. Chromatic detections are clustered regardless of colour and the colour is decided
    by total pixel vote (a cube's lit faces can straddle a hue boundary). Black is separate."""
    frames = sorted([f for f in manifest["frames"] if f["phase"] == "video_demo"],
                    key=lambda f: f["frame_index"])
    dets = []
    for fr in frames[::stride]:
        for cam in ("front", "wrist"):
            try:
                dets += detect_cubes_frame(fr, cam)
            except Exception as e:
                log(f"detect error frame {fr.get('frame_index')} {cam}: {e!r}")
    inst = []
    groups = [[d for d in dets if d["color"] != "black"], [d for d in dets if d["color"] == "black"]]
    for ds in groups:
        for c in _cluster(ds, link):
            votes = {}
            for d in c:
                votes[d["color"]] = votes.get(d["color"], 0) + d["n"]
            color = max(votes, key=votes.get)
            if len(votes) > 1:
                log(f"colour vote at {np.round(np.median(np.array([d['xy'] for d in c]),0),3)}: {votes} -> {color}")
            w = [d["xy"] for d in c if d["cam"] == "wrist"]
            src = w if len(w) >= 2 else [d["xy"] for d in c]
            inst.append({"color": color, "xy": [float(x) for x in np.median(np.array(src), 0)],
                         "ndet": len(c), "nwrist": len(w), "cams": sorted(set(d["cam"] for d in c)),
                         "f0": int(min(d["frame"] for d in c)), "f1": int(max(d["frame"] for d in c)),
                         "votes": votes})
    return inst, dets


def container_tops(P, lo=0.058, hi=0.09, min_px=30):
    """Container top-part components from a world point image P (HxWx3, NaN invalid)."""
    Z = P[..., 2]
    with np.errstate(invalid="ignore"):
        m = (Z > lo) & (Z < hi)
    lab, n = label(m)
    res = []
    for k in range(1, n + 1):
        sel = lab == k
        if sel.sum() < min_px:
            continue
        xy = P[sel][:, :2]
        ctr, ang, ext = min_area_rect(xy)
        res.append({"xy": [float(ctr[0]), float(ctr[1])], "ang": float(ang), "ext": [float(ext[0]), float(ext[1])],
                    "n": int(sel.sum())})
    return res


def min_area_rect(xy):
    m = xy.mean(0)
    best = None
    for a in np.radians(np.arange(0, 90, 1.0)):
        R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        r = (xy - m) @ R.T
        lo, hi = np.percentile(r, 2, 0), np.percentile(r, 98, 0)
        ext = hi - lo
        ar = ext[0] * ext[1]
        if best is None or ar < best[0]:
            best = (ar, a, ext, ((hi + lo) / 2) @ R + m)
    return best[3], best[1], best[2]


def is_container_top(c):
    e = sorted(c["ext"])
    return 0.022 < e[0] < 0.055 and 0.025 < e[1] < 0.06
