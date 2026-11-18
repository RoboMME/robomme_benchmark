"""Perception helpers: depth back-projection, object segmentation, video tracking."""
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
        return np.array(im.convert("RGB"))


def load_depth(path, mpu=0.001):
    with Image.open(path) as im:
        return np.array(im).astype(np.float64) * mpu


def saturation(rgb):
    a = rgb.astype(np.float64)
    mx, mn = a.max(-1), a.min(-1)
    return (mx - mn) / np.maximum(mx, 1)


def table_mask_colors(rgb):
    """Table is orange/brown wood: r > g > b strongly."""
    a = rgb.astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r > g + 15) & (g > b) & (r > 90)


def estimate_table_z(pts, rgb):
    tm = table_mask_colors(rgb)
    z = pts[..., 2][tm & np.isfinite(pts[..., 2])]
    if z.size < 100:
        return 0.0
    return float(np.median(z))


BUTTON_HALF = 0.045   # half-size of the excluded button footprint (m)
_EXCLUDE = []         # list of xy centres to exclude (button), set by set_exclusions


def set_exclusions(xys):
    _EXCLUDE.clear()
    _EXCLUDE.extend([np.asarray(x, float) for x in xys])


def object_mask(rgb, pts, table_z, zmin=0.006, zmax=0.09, xy_box=(-0.45, 0.35, -0.45, 0.45)):
    """Foreground object pixels: above table, below arm, saturated, outside the button footprint."""
    z = pts[..., 2]
    x, y = pts[..., 0], pts[..., 1]
    ok = np.isfinite(z)
    h = np.where(ok, z - table_z, -1)
    m = ok & (h > zmin) & (h < zmax)
    m &= (x > xy_box[0]) & (x < xy_box[1]) & (y > xy_box[2]) & (y < xy_box[3])
    sat = saturation(rgb)
    m &= sat > 0.25
    for c in _EXCLUDE:
        m &= ~((np.abs(x - c[0]) < BUTTON_HALF) & (np.abs(y - c[1]) < BUTTON_HALF))
    return m


def blobs(mask, min_px=12):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) >= min_px:
            out.append((xs, ys))
    return out


def frame_points(frame, cam="front", ws=WS):
    files = {(f["camera"], f["kind"]): f["path"] for f in frame["files"]}
    rgb = load_rgb(ws / files[(cam, "rgb")])
    mpu = frame.get("depth", {}).get(cam, {}).get("meters_per_unit", 0.001)
    d = load_depth(ws / files[(cam, "depth")], mpu)
    pts = backproject(d, frame["camera_intrinsics"][cam]["matrix"], frame["camera_poses"][cam])
    return rgb, pts


def frame_object_xy(frame, ws=WS):
    rgb, pts = frame_points(frame, "front", ws)
    tz = estimate_table_z(pts, rgb)
    om = object_mask(rgb, pts, tz)
    return pts[om][:, :2], tz


MIN_BLOB = 25


def frame_blobs(frame, ws=WS, min_px=MIN_BLOB):
    """List of (P (n,2) world xy points) for each object blob in a frame."""
    rgb, pts = frame_points(frame, "front", ws)
    tz = estimate_table_z(pts, rgb)
    om = object_mask(rgb, pts, tz)
    return [pts[ys, xs][:, :2] for xs, ys in blobs(om, min_px)]


def count_objects(frame, ws=WS):
    rgb, pts = frame_points(frame, "front", ws)
    tz = estimate_table_z(pts, rgb)
    om = object_mask(rgb, pts, tz)
    bl = blobs(om, MIN_BLOB)
    cents = [np.nanmean(pts[ys, xs][:, :2], 0) for xs, ys in bl]
    sizes = [len(xs) for xs, ys in bl]
    return cents, sizes


def assign_step(P, C, gate=0.07, iters=4):
    """Refine centroids C (N,2) on points P (M,2) by gated nearest-centroid EM."""
    C = C.copy()
    seen = np.zeros(len(C), bool)
    if len(P) == 0:
        return C, seen
    for _ in range(iters):
        d = np.linalg.norm(P[:, None, :] - C[None, :, :], axis=2)
        a = d.argmin(1)
        dm = d[np.arange(len(P)), a]
        newC = C.copy()
        seen = np.zeros(len(C), bool)
        for j in range(len(C)):
            sel = (a == j) & (dm < gate)
            if sel.sum() >= 5:
                newC[j] = P[sel].mean(0)
                seen[j] = True
        C = newC
    return C, seen


def track(frames, start_k, init_C, ws=WS, log=None):
    """Track N object centroids from frame start_k to the end. Returns list of (k, C)."""
    C = np.array(init_C, float)
    V = np.zeros_like(C)
    hist = [(start_k, C.copy())]
    for k in range(start_k + 1, len(frames)):
        P, _ = frame_object_xy(frames[k], ws)
        pred = C + V
        newC, seen = assign_step(P, pred)
        newC[~seen] = C[~seen]
        V = 0.5 * V + 0.5 * (newC - C)
        V[~seen] = 0
        C = newC
        hist.append((k, C.copy()))
    return hist


def live_points(robot, cam):
    imgs = robot.images()
    p = robot.last
    rgb = imgs[(cam, "rgb")]
    pts = backproject(imgs[(cam, "depth")], p["camera_intrinsics"][cam]["matrix"], p["camera_poses"][cam])
    return rgb, pts


def live_objects(robot, cam="front", table_z=None):
    rgb, pts = live_points(robot, cam)
    tz = estimate_table_z(pts, rgb) if table_z is None else table_z
    om = object_mask(rgb, pts, tz)
    out = []
    for xs, ys in blobs(om, min_px=8):
        P = pts[ys, xs]
        out.append({"xy": np.nanmean(P[:, :2], 0), "ztop": float(np.nanpercentile(P[:, 2], 95)),
                    "n": len(xs), "uv": (float(xs.mean()), float(ys.mean()))})
    return out, tz


def button_location(rgb, pts, table_z):
    """White/grey low-saturation raised part on the table (the button)."""
    z = pts[..., 2]
    ok = np.isfinite(z)
    h = np.where(ok, z - table_z, -1)
    sat = saturation(rgb)
    bright = rgb.astype(int).max(-1)
    m = ok & (h > 0.003) & (h < 0.06) & (sat < 0.15) & (bright > 90)
    x, y = pts[..., 0], pts[..., 1]
    m &= (x > -0.45) & (x < 0.35) & (np.abs(y) < 0.45)
    bl = blobs(m, min_px=10)
    if not bl:
        return None
    xs, ys = max(bl, key=lambda b: len(b[0]))
    P = pts[ys, xs]
    return {"xy": np.nanmean(P[:, :2], 0), "ztop": float(np.nanpercentile(P[:, 2], 90)), "n": len(xs)}


def _kmeans(P, C0, iters=8):
    C = np.array(C0, float)
    for _ in range(iters):
        a = np.linalg.norm(P[:, None] - C[None], axis=2).argmin(1)
        for j in range(len(C)):
            if (a == j).sum() > 0:
                C[j] = P[a == j].mean(0)
    return C


def associate(B, pred, unit, gate=0.045):
    """Capacity-aware Hungarian association of tracks (pred) to blobs B.
    Returns new positions (N,2) and seen mask."""
    from scipy.optimize import linear_sum_assignment
    N = len(pred)
    newC = pred.copy()
    seen = np.zeros(N, bool)
    if not B:
        return newC, seen
    cents = [b.mean(0) for b in B]
    cols = []
    for bi, b in enumerate(B):
        cap = max(1, int(round(len(b) / unit)))
        cap = min(cap, 3)
        cols += [bi] * cap
    BIG = 1e3
    D = np.full((N, len(cols)), BIG)
    for ci, bi in enumerate(cols):
        for t in range(N):
            # distance from prediction to the nearest point of the blob, plus to centroid
            dc = np.linalg.norm(cents[bi] - pred[t])
            dp = np.min(np.linalg.norm(B[bi] - pred[t], axis=1))
            d = 0.5 * dc + 0.5 * dp
            if dp < gate:
                D[t, ci] = d
    r, c = linear_sum_assignment(D)
    groups = {}
    for t, ci in zip(r, c):
        if D[t, ci] < BIG:
            groups.setdefault(cols[ci], []).append(t)
    for bi, ts in groups.items():
        if len(ts) == 1:
            newC[ts[0]] = cents[bi]
        else:
            Cs = _kmeans(B[bi], pred[ts])
            for j, t in enumerate(ts):
                newC[t] = Cs[j]
        for t in ts:
            seen[t] = True
    return newC, seen


def track2(frames, start_k, init_C, unit, ws=WS):
    C = np.array(init_C, float)
    V = np.zeros_like(C)
    hist = [(start_k, C.copy())]
    for k in range(start_k + 1, len(frames)):
        B = frame_blobs(frames[k], ws)
        pred = C + V
        newC, seen = associate(B, pred, unit)
        newC[~seen] = C[~seen]
        dv = newC - C
        V = np.where(seen[:, None], 0.6 * dv + 0.4 * V, 0.0)
        C = newC
        hist.append((k, C.copy()))
    return hist


# ---------------- fused front+wrist occupancy grid (v3) ----------------
GRID = 0.005          # cell size (m)
GRID_MIN_CELLS = 6    # min blob area in cells
GRID_X = (-0.45, 0.35)
GRID_Y = (-0.45, 0.45)


def _cells(rgb, pts, tz, zmax=0.09):
    om = object_mask(rgb, pts, tz, zmax=zmax)
    P = pts[om][:, :2]
    P = P[np.isfinite(P).all(1)]
    ij = np.floor((P - np.array([GRID_X[0], GRID_Y[0]])) / GRID).astype(int)
    nx = int((GRID_X[1] - GRID_X[0]) / GRID)
    ny = int((GRID_Y[1] - GRID_Y[0]) / GRID)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
    g = np.zeros((nx, ny), bool)
    g[ij[ok, 0], ij[ok, 1]] = True
    return g


def grid_blobs_from_views(views, tz, zmax=0.09):
    """views: list of (rgb, pts). Returns list of (cell-centre xy points) per blob."""
    from scipy import ndimage
    g = None
    for rgb, pts in views:
        c = _cells(rgb, pts, tz, zmax)
        g = c if g is None else (g | c)
    g = ndimage.binary_closing(g, structure=np.ones((3, 3)))
    lab, n = ndimage.label(g, structure=np.ones((3, 3)))
    out = []
    for i in range(1, n + 1):
        ii, jj = np.nonzero(lab == i)
        if len(ii) >= GRID_MIN_CELLS:
            xy = np.stack([GRID_X[0] + (ii + 0.5) * GRID, GRID_Y[0] + (jj + 0.5) * GRID], 1)
            out.append(xy)
    return out


def frame_blobs_fused(frame, ws=WS, tz=None):
    views = []
    for cam in ("front", "wrist"):
        try:
            rgb, pts = frame_points(frame, cam, ws)
            views.append((rgb, pts))
        except Exception:
            pass
    if tz is None:
        tz = estimate_table_z(views[0][1], views[0][0])
    return grid_blobs_from_views(views, tz)


def live_blobs_fused(robot, tz):
    views = [live_points(robot, cam) for cam in ("front", "wrist")]
    return grid_blobs_from_views(views, tz)


def track3(frames, start_k, init_C, unit, ws=WS, tz=None):
    C = np.array(init_C, float)
    V = np.zeros_like(C)
    hist = [(start_k, C.copy())]
    for k in range(start_k + 1, len(frames)):
        B = frame_blobs_fused(frames[k], ws, tz)
        pred = C + V
        newC, seen = associate(B, pred, unit)
        newC[~seen] = C[~seen]
        dv = newC - C
        V = np.where(seen[:, None], 0.6 * dv + 0.4 * V, 0.0)
        C = newC
        hist.append((k, C.copy()))
    return hist
