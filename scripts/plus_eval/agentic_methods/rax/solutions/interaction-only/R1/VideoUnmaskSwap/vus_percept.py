"""Depth-based scene perception: table plane, raised-object components, colours, top faces."""
import math

import numpy as np

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None
try:
    import cv2 as _cv2
except Exception:  # pragma: no cover
    _cv2 = None

# Tunables (blind defaults; no live data seen). Distances in metres.
MIN_RAISE = 0.006        # points this far above the table count as objects
MAX_RAISE = 0.40         # ignore points higher than this above the table
VOXEL = 0.01             # voxel size for 3-D connected components
MIN_COMPONENT_PX = 12    # smaller components are noise
ROBOT_HEIGHT = 0.30      # component reaching this high above the table is robot
ROBOT_SPAN = 0.25        # component with this vertical span is robot
EE_RADIUS = 0.05         # component with a point this close to the TCP is robot (gripper)
CUBE_MAX_EXTENT = 0.065  # cube footprint / height upper bounds
CUBE_MIN_COLOR_FRAC = 0.30

COLOR_HUES = {"red": 0.0, "orange": 30.0, "yellow": 58.0, "green": 120.0, "cyan": 180.0,
              "blue": 225.0, "purple": 280.0, "pink": 325.0}
COLOR_SYNONYMS = {"violet": "purple", "magenta": "pink", "lime": "green", "teal": "cyan",
                  "navy": "blue", "gold": "yellow", "grey": "gray", "crimson": "red"}
ACHROMATIC = ("white", "black", "gray")
ALL_COLORS = list(COLOR_HUES) + list(ACHROMATIC) + ["brown"]


def rgb_to_hsv(rgb):
    a = rgb.astype(np.float64) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = np.max(a, axis=-1)
    mn = np.min(a, axis=-1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-9
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = (60 * ((g[rm] - b[rm]) / d[rm])) % 360
    h[gm] = 60 * ((b[gm] - r[gm]) / d[gm]) + 120
    h[bm] = 60 * ((r[bm] - g[bm]) / d[bm]) + 240
    s = np.where(mx > 1e-9, d / np.maximum(mx, 1e-9), 0)
    return h, s, mx


def color_labels(rgb):
    """Per-pixel colour name index into ALL_COLORS (int array), -1 for unclassified."""
    h, s, v = rgb_to_hsv(rgb)
    lab = np.full(h.shape, -1, dtype=np.int32)
    chrom = (s > 0.35) & (v > 0.18)
    names = list(COLOR_HUES)
    hues = np.array([COLOR_HUES[n] for n in names])
    dist = np.abs(((h[..., None] - hues[None, None, :]) + 180) % 360 - 180)
    best = np.argmin(dist, axis=-1)
    lab[chrom] = best[chrom]
    # brown: orange-ish hue, low value
    brown = chrom & (h > 10) & (h < 45) & (v < 0.55)
    lab[brown] = ALL_COLORS.index("brown")
    white = (~chrom) & (s < 0.2) & (v > 0.75)
    black = (~chrom) & (v < 0.18)
    gray = (~chrom) & ~white & ~black
    lab[white] = ALL_COLORS.index("white")
    lab[black] = ALL_COLORS.index("black")
    lab[gray] = ALL_COLORS.index("gray")
    return lab


def hue_distance(rgb_mean, name):
    """Circular hue distance (deg) of a mean colour to a named chromatic colour; 999 if n/a."""
    if name not in COLOR_HUES:
        return 999.0
    h, s, v = rgb_to_hsv(np.asarray(rgb_mean, dtype=np.float64).reshape(1, 1, 3))
    return float(abs(((h[0, 0] - COLOR_HUES[name]) + 180) % 360 - 180))


# ----------------------------------------------------------------------------- table

def table_height(P, valid):
    z = P[..., 2][valid]
    z = z[np.isfinite(z)]
    if z.size < 100:
        return None
    lo, hi = np.percentile(z, 0.5), np.percentile(z, 99.5)
    if hi - lo < 1e-3:
        return float(np.median(z))
    bins = np.arange(lo, hi + 0.002, 0.002)
    if bins.size < 3:
        return float(np.median(z))
    hist, edges = np.histogram(z, bins=bins)
    n = z.size
    peaks = [i for i in range(len(hist)) if hist[i] >= 0.08 * n]
    if not peaks:
        i = int(np.argmax(hist))
    else:
        i = max(peaks)  # highest large plane is the table (floor lies below)
    zc = 0.5 * (edges[i] + edges[i + 1])
    near = z[np.abs(z - zc) < 0.004]
    return float(np.median(near)) if near.size else float(zc)


def table_bounds(P, valid, tz):
    m = valid & (np.abs(P[..., 2] - tz) < 0.004)
    if m.sum() < 50:
        return None
    xy = P[..., :2][m]
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))


# ----------------------------------------------------------------------------- components

def _label3d(occ):
    if _ndi is not None:
        lab, n = _ndi.label(occ, structure=np.ones((3, 3, 3), dtype=bool))
        return lab, n
    # fallback BFS
    lab = np.zeros(occ.shape, dtype=np.int32)
    n = 0
    idx = np.argwhere(occ)
    for start in map(tuple, idx):
        if lab[start]:
            continue
        n += 1
        stack = [start]
        lab[start] = n
        while stack:
            c = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        q = (c[0] + dx, c[1] + dy, c[2] + dz)
                        if all(0 <= q[i] < occ.shape[i] for i in range(3)) and occ[q] and not lab[q]:
                            lab[q] = n
                            stack.append(q)
    return lab, n


def footprint(xy):
    """xy Nx2 -> (cx, cy, w, l, yaw) of the min-area rectangle; w <= l, yaw = direction of w axis."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[0] < 3:
        c = xy.mean(axis=0) if xy.shape[0] else np.zeros(2)
        return float(c[0]), float(c[1]), 0.0, 0.0, 0.0
    if _cv2 is not None:
        try:
            pts = (xy * 1000.0).astype(np.float32)
            (cx, cy), (a, b), ang = _cv2.minAreaRect(pts)
            th = math.radians(ang)
            # axis of side 'a' is (cos th, sin th); side 'b' perpendicular
            if a <= b:
                w, l, yaw = a, b, th
            else:
                w, l, yaw = b, a, th + math.pi / 2
            return cx / 1000.0, cy / 1000.0, w / 1000.0, l / 1000.0, yaw
        except Exception:
            pass
    best = None
    c = xy.mean(axis=0)
    for deg in range(0, 90, 2):
        th = math.radians(deg)
        u = np.array([math.cos(th), math.sin(th)])
        v = np.array([-math.sin(th), math.cos(th)])
        pu, pv = (xy - c) @ u, (xy - c) @ v
        a, b = pu.max() - pu.min(), pv.max() - pv.min()
        if best is None or a * b < best[0]:
            ctr = c + u * (pu.max() + pu.min()) / 2 + v * (pv.max() + pv.min()) / 2
            best = (a * b, a, b, th, ctr)
    _, a, b, th, ctr = best
    if a <= b:
        return float(ctr[0]), float(ctr[1]), float(a), float(b), th
    return float(ctr[0]), float(ctr[1]), float(b), float(a), th + math.pi / 2


def segment(P, valid, rgb, tz, ee_pos=None, bounds=None, colors=None, min_raise=MIN_RAISE):
    """Raised-object components. Returns list of dicts (pixel indices, geometry, colour stats)."""
    z = P[..., 2]
    m = valid & np.isfinite(z) & (z > tz + min_raise) & (z < tz + MAX_RAISE)
    if bounds is not None:
        x0, y0, x1, y1 = bounds
        mg = 0.05
        m &= (P[..., 0] > x0 - mg) & (P[..., 0] < x1 + mg) & (P[..., 1] > y0 - mg) & (P[..., 1] < y1 + mg)
    pix = np.flatnonzero(m.ravel())
    if pix.size == 0:
        return []
    pts = P.reshape(-1, 3)[pix]
    vmin = pts.min(axis=0)
    vi = np.floor((pts - vmin) / VOXEL).astype(np.int64)
    shape = tuple(int(s) + 1 for s in vi.max(axis=0))
    if np.prod(shape) > 40_000_000:
        return []
    occ = np.zeros(shape, dtype=bool)
    occ[vi[:, 0], vi[:, 1], vi[:, 2]] = True
    lab, n = _label3d(occ)
    plab = lab[vi[:, 0], vi[:, 1], vi[:, 2]]
    if colors is None and rgb is not None:
        colors = color_labels(rgb)
    flat_rgb = rgb.reshape(-1, 3) if rgb is not None else None
    flat_col = colors.ravel() if colors is not None else None
    comps = []
    order = np.argsort(plab, kind="stable")
    plab_s = plab[order]
    splits = np.flatnonzero(np.diff(plab_s)) + 1
    groups = np.split(order, splits)
    for g in groups:
        if g.size < MIN_COMPONENT_PX:
            continue
        cp = pts[g]
        zmax = float(cp[:, 2].max())
        zmin = float(cp[:, 2].min())
        top = cp[cp[:, 2] > zmax - 0.012]
        c = {
            "pix": pix[g],
            "n": int(g.size),
            "zmax": zmax, "zmin": zmin,
            "height": zmax - tz,
            "xy_all": cp[:, :2].mean(axis=0),
            "top_xy": top[:, :2].mean(axis=0),
            "top_n": int(top.shape[0]),
            "top_pts": top[:, :2][:: max(1, top.shape[0] // 400)],
            "ext": float(max(np.ptp(cp[:, 0]), np.ptp(cp[:, 1]))),
            "robot": False,
        }
        fp = footprint(cp[:, :2])
        c["fp"] = fp  # (cx, cy, w, l, yaw) all visible points
        c["fp_top"] = footprint(top[:, :2]) if top.shape[0] >= 5 else fp
        if flat_rgb is not None:
            c["rgb_mean"] = flat_rgb[pix[g]].mean(axis=0)
        if flat_col is not None:
            cl = flat_col[pix[g]]
            cnt = np.bincount(cl[cl >= 0], minlength=len(ALL_COLORS))
            c["color_counts"] = cnt
            c["color_frac"] = cnt / max(1, g.size)
            k = int(np.argmax(cnt))
            c["color"] = ALL_COLORS[k]
            c["color_dom_frac"] = float(cnt[k] / max(1, g.size))
        if (zmax - tz) > ROBOT_HEIGHT or (zmax - zmin) > ROBOT_SPAN:
            c["robot"] = True
        if ee_pos is not None:
            d = np.linalg.norm(cp - np.asarray(ee_pos)[None, :], axis=1).min()
            c["ee_dist"] = float(d)
            if d < EE_RADIUS:
                c["robot"] = True
        comps.append(c)
    return comps


def is_cube_like(c):
    return (not c["robot"]) and c["ext"] < CUBE_MAX_EXTENT and c["height"] < CUBE_MAX_EXTENT


def cube_candidates(comps, color=None):
    """Small, chromatic components. If colour is given, rank by its pixel fraction."""
    out = []
    for c in comps:
        if not is_cube_like(c) or "color_frac" not in c:
            continue
        if color is not None and color in ALL_COLORS:
            f = float(c["color_frac"][ALL_COLORS.index(color)])
            if f >= CUBE_MIN_COLOR_FRAC:
                out.append((f, c))
        else:
            chrom = sum(float(c["color_frac"][ALL_COLORS.index(k)]) for k in COLOR_HUES)
            if chrom >= CUBE_MIN_COLOR_FRAC:
                out.append((chrom, c))
    out.sort(key=lambda t: -t[0])
    return [c for _, c in out]


def target_color_pixels(P, valid, colors, tz, color, max_raise=0.07):
    """Raw fallback: raised pixels of the target colour -> (count, median xy)."""
    if color not in ALL_COLORS:
        return 0, None
    k = ALL_COLORS.index(color)
    z = P[..., 2]
    m = valid & (colors == k) & (z > tz + 0.003) & (z < tz + max_raise)
    n = int(m.sum())
    if n == 0:
        return 0, None
    return n, np.median(P[..., :2][m], axis=0)
