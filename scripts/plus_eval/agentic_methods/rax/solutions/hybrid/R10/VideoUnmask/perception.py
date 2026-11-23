"""Perception helpers for VideoUnmask: geometry, colour masks, cube & container detection."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

WS = Path("/workspace")


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(depth_m, K, pose):
    """depth (H,W) metres -> world points (H,W,3); invalid -> nan."""
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    d = depth_m
    cam = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1)
    R = quat_to_R(pose["quaternion_wxyz"])
    pts = cam @ R.T + np.array(pose["position_m"])
    pts[d <= 0] = np.nan
    return pts


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).astype(np.float32)


def load_depth(path, mpu=0.001):
    with Image.open(path) as im:
        return np.asarray(im).astype(np.float32) * mpu


def hsv(rgb):
    r, g, b = rgb[..., 0] / 255, rgb[..., 1] / 255, rgb[..., 2] / 255
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    c = mx - mn
    h = np.zeros_like(mx)
    m = c > 1e-6
    rr = m & (mx == r)
    gg = m & (mx == g) & ~rr
    bb = m & ~rr & ~gg
    h[rr] = ((g - b)[rr] / c[rr]) % 6
    h[gg] = ((b - r)[gg] / c[gg]) + 2
    h[bb] = ((r - g)[bb] / c[bb]) + 4
    h = h * 60
    s = np.where(mx > 0, c / np.maximum(mx, 1e-6), 0)
    return h, s, mx


def color_masks(rgb):
    h, s, v = hsv(rgb)
    sat = (s > 0.6) & (v > 0.25)
    red = sat & ((h < 12) | (h > 340))
    green = sat & (h > 90) & (h < 160)
    blue = sat & (h > 210) & (h < 260)
    return {"red": red, "green": green, "blue": blue}


def components(mask, min_px=3, max_px=900):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if min_px <= len(xs) <= max_px:
            out.append((xs, ys))
    return out


def cluster_points(pts, radius):
    """Greedy clustering of Nx3 (or Nx2) points; returns list of (centroid, count)."""
    clusters = []
    for p in pts:
        for c in clusters:
            if np.linalg.norm(c["sum"][:2] / c["n"] - p[:2]) < radius:
                c["sum"] += p
                c["n"] += 1
                break
        else:
            clusters.append({"sum": np.array(p, dtype=float), "n": 1})
    # merge pass
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                a, b = clusters[i], clusters[j]
                if np.linalg.norm(a["sum"][:2] / a["n"] - b["sum"][:2] / b["n"]) < radius:
                    a["sum"] += b["sum"]; a["n"] += b["n"]
                    clusters.pop(j); merged = True
                    break
            if merged:
                break
    return [(c["sum"] / c["n"], c["n"]) for c in clusters]


def table_z_estimate(pts):
    z = pts[..., 2]
    z = z[np.isfinite(z)]
    # table is the most common low height
    hist, edges = np.histogram(z, bins=200, range=(-0.2, 0.3))
    i = np.argmax(hist)
    return 0.5 * (edges[i] + edges[i + 1])


def cubes_from_video(manifest_path, log=print, cams=("front", "wrist")):
    """Return dict colour -> list of world centroids where cubes rested on the table during the video."""
    man = json.loads(Path(manifest_path).read_text())
    frames = [f for f in man["frames"] if f["phase"] == "video_demo"]
    dets = {"red": [], "green": [], "blue": []}
    tz = None
    for f in frames:
        files = {(x["camera"], x["kind"]): x["path"] for x in f["files"]}
        for cam in cams:
            if (cam, "rgb") not in files or (cam, "depth") not in files:
                continue
            try:
                rgb = load_rgb(WS / files[(cam, "rgb")])
                dep = load_depth(WS / files[(cam, "depth")], f["depth"][cam].get("meters_per_unit", 0.001))
                pts = backproject(dep, f["camera_intrinsics"][cam]["matrix"], f["camera_poses"][cam])
            except Exception as e:
                log(f"frame {f.get('frame_index')} {cam} load error {e}")
                continue
            if tz is None and cam == "front":
                tz = table_z_estimate(pts)
            if tz is None:
                continue
            for col, m in color_masks(rgb).items():
                for xs, ys in components(m, 3):
                    p = pts[ys, xs]
                    p = p[np.isfinite(p).all(1)]
                    if len(p) < 3:
                        continue
                    c = np.median(p, 0)
                    if c[2] - tz > 0.06 or c[2] - tz < -0.02 or c[0] < -0.4:
                        continue  # not resting on table (held / falling) or robot
                    dets[col].append((f["frame_index"], cam, c))
    out = {}
    counts = {}
    for col, lst in dets.items():
        if not lst:
            out[col] = []
            counts[col] = []
            log(f"video {col}: none")
            continue
        cl = cluster_points(np.array([c for _, _, c in lst]), 0.03)
        cl = sorted(cl, key=lambda t: -t[1])
        out[col] = [c for c, n in cl if n >= 6]
        counts[col] = [n for c, n in cl if n >= 6]
        log(f"video {col}: {len(lst)} dets -> {[(np.round(c, 3).tolist(), n) for c, n in cl]}")
    out["_counts"] = counts
    return out, (tz if tz is not None else 0.0)


def frame_points(frame, cam):
    files = {(x["camera"], x["kind"]): x["path"] for x in frame["files"]}
    rgb = load_rgb(WS / files[(cam, "rgb")])
    dep = load_depth(WS / files[(cam, "depth")], frame["depth"][cam].get("meters_per_unit", 0.001))
    pts = backproject(dep, frame["camera_intrinsics"][cam]["matrix"], frame["camera_poses"][cam])
    return rgb, pts


def detect_containers(rgb, pts, tz, log=print):
    """Colour-agnostic: flat top faces 4-12 cm above the table with a 3.5-9 cm footprint,
    not attached to taller structures (pillars). Returns list of dicts {xy, ztop, yaw, n, extent}."""
    from scipy import ndimage
    z = pts[..., 2] - tz
    fin = np.isfinite(z)
    band = fin & (z > 0.03) & (z < 0.13)
    high = ndimage.binary_dilation(fin & (z >= 0.13), iterations=2)
    lab, n = ndimage.label(band)
    out = []
    for i in range(1, n + 1):
        comp = lab == i
        cnt = int(comp.sum())
        if cnt < 30:
            continue
        Q = pts[comp]
        zz = Q[:, 2] - tz
        ztop = float(np.percentile(zz, 90))
        topm = zz > ztop - 0.012
        top = Q[topm]
        touch = float((comp & high).sum()) / cnt
        ext = top[:, :2].max(0) - top[:, :2].min(0)
        c = np.median(top[:, :2], 0)
        info = {"xy": c, "ztop": ztop + tz, "n": cnt, "extent": ext, "touch_high": touch,
                "flat_frac": float(topm.mean())}
        ok = (0.035 < ext.max() < 0.10) and (ext.min() > 0.025) and touch < 0.15 and c[0] > -0.4 and 0.04 < ztop < 0.12
        if not ok and ext.max() >= 0.10 and touch < 0.15 and 0.04 < ztop < 0.12:
            # possibly two touching containers: split by k-means on top points
            for sub in split_top(top):
                e2 = sub[:, :2].max(0) - sub[:, :2].min(0)
                if 0.035 < e2.max() < 0.10 and e2.min() > 0.025 and np.median(sub[:, 0]) > -0.4:
                    out.append({"xy": np.median(sub[:, :2], 0), "ztop": ztop + tz, "n": len(sub), "extent": e2,
                                "touch_high": touch, "yaw": estimate_yaw(sub[:, :2]), "split": True})
            continue
        if ok:
            info["yaw"] = estimate_yaw(top[:, :2])
            out.append(info)
    for o in out:
        log(f"container xy={np.round(o['xy'],3).tolist()} ztop={o['ztop']:.3f} yaw={np.degrees(o['yaw']):.1f} n={o['n']} ext={np.round(o['extent'],3).tolist()} touch={o['touch_high']:.2f}")
    return out


def split_top(top, k_max=4):
    """Split a merged top-face cloud into ~6cm pieces using simple k-means on xy."""
    xy = top[:, :2]
    ext = xy.max(0) - xy.min(0)
    k = int(min(k_max, max(2, round(ext.max() / 0.065))))
    rng = np.random.default_rng(0)
    cen = xy[rng.choice(len(xy), k, replace=False)]
    for _ in range(20):
        d = ((xy[:, None, :] - cen[None]) ** 2).sum(-1)
        a = d.argmin(1)
        cen = np.array([xy[a == j].mean(0) if (a == j).any() else cen[j] for j in range(k)])
    return [top[a == j] for j in range(k) if (a == j).sum() > 15]


def estimate_yaw(xy):
    """Yaw (rad, in [-pi/4, pi/4)) of a square footprint via minAreaRect, fallback PCA-free search."""
    try:
        import cv2
        rect = cv2.minAreaRect((xy * 1000).astype(np.float32))
        a = np.radians(rect[2])
    except Exception:
        best, a = 1e9, 0.0
        c = xy - xy.mean(0)
        for t in np.radians(np.arange(-45, 45, 2)):
            R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
            q = c @ R.T
            area = np.ptp(q[:, 0]) * np.ptp(q[:, 1])
            if area < best:
                best, a = area, t
    return float((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)
