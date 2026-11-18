"""Colour segmentation and blob extraction (numpy only; scipy optional)."""
import numpy as np

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None

# hue ranges in degrees (inclusive lo, exclusive hi); red wraps
COLOR_HUES = {
    'red': [(345, 360), (0, 11)],
    'orange': [(14, 51)],
    'yellow': [(51, 80)],
    'green': [(85, 160)],
    'cyan': [(170, 200)],
    'blue': [(212, 258)],
    'purple': [(262, 290)],
    'magenta': [(292, 342)],
}
COLOR_ALIASES = {
    'pink': 'magenta', 'violet': 'purple', 'teal': 'cyan', 'gold': 'yellow',
}


def to_hsv(rgb):
    a = rgb.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1)
    mn = a.min(-1)
    d = mx - mn + 1e-6
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60.0
    h = np.mod(h, 360.0)
    s = np.where(mx > 1e-6, (mx - mn) / (mx + 1e-6), 0.0)
    return h, s, mx


def color_mask(hsv, color, s_min=0.75, v_min=0.16):
    h, s, v = hsv
    ranges = COLOR_HUES[color]
    m = np.zeros(h.shape, bool)
    for lo, hi in ranges:
        m |= (h >= lo) & (h < hi)
    smin = s_min
    if color in ('orange', 'red', 'yellow'):
        smin = max(s_min, 0.8)  # wood table is orange-ish, S<=0.72
    return m & (s >= smin) & (v >= v_min)


def label(mask):
    if _ndi is not None:
        lab, n = _ndi.label(mask)
        return lab, n
    # simple two-pass union find fallback (4-connectivity)
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    parent = [0]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    nxt = 1
    for y in range(H):
        for x in range(W):
            if not mask[y, x]:
                continue
            up = lab[y - 1, x] if y > 0 else 0
            lf = lab[y, x - 1] if x > 0 else 0
            if up and lf:
                a, b = find(up), find(lf)
                lab[y, x] = min(a, b)
                if a != b:
                    parent[max(a, b)] = min(a, b)
            elif up or lf:
                lab[y, x] = up or lf
            else:
                parent.append(nxt)
                lab[y, x] = nxt
                nxt += 1
    for y in range(H):
        for x in range(W):
            if lab[y, x]:
                lab[y, x] = find(lab[y, x])
    uniq = np.unique(lab[lab > 0])
    remap = np.zeros(nxt, np.int32)
    for i, u in enumerate(uniq):
        remap[u] = i + 1
    return remap[lab], len(uniq)


def erode(mask, it=1):
    m = mask.copy()
    for _ in range(it):
        p = np.pad(m, 1, constant_values=False)
        m = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m


def dilate(mask, it=1):
    m = mask.copy()
    for _ in range(it):
        p = np.pad(m, 1, constant_values=False)
        m = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return m


def blobs(mask, min_area=6):
    lab, n = label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < min_area:
            continue
        out.append({
            'area': int(len(xs)),
            'u': float(xs.mean()), 'v': float(ys.mean()),
            'umin': int(xs.min()), 'umax': int(xs.max()),
            'vmin': int(ys.min()), 'vmax': int(ys.max()),
            'xs': xs, 'ys': ys,
        })
    out.sort(key=lambda b: -b['area'])
    return out


def interior_pixel(b, k=5):
    """Pick pixels of a blob near its centroid but deep inside the mask (avoid edges)."""
    xs, ys = b['xs'], b['ys']
    d = (xs - b['u']) ** 2 + (ys - b['v']) ** 2
    order = np.argsort(d)
    pts = [(float(xs[i]), float(ys[i])) for i in order[:k]]
    return pts


def min_area_angle(xs, ys):
    """Angle (deg, image coords u right / v down) in [0,90) of the min-area bounding rectangle."""
    pts = np.stack([xs - xs.mean(), ys - ys.mean()], 1).astype(np.float64)
    best = None
    for a in np.arange(0.0, 90.0, 1.0):
        t = np.deg2rad(a)
        c, s = np.cos(t), np.sin(t)
        p = pts @ np.array([[c, -s], [s, c]])
        area = (np.ptp(p[:, 0]) + 1) * (np.ptp(p[:, 1]) + 1)
        if best is None or area < best[0]:
            best = (area, a)
    # refine
    a0 = best[1]
    for a in np.arange(a0 - 1.0, a0 + 1.0, 0.1):
        t = np.deg2rad(a)
        c, s = np.cos(t), np.sin(t)
        p = pts @ np.array([[c, -s], [s, c]])
        area = (np.ptp(p[:, 0]) + 1) * (np.ptp(p[:, 1]) + 1)
        if area < best[0]:
            best = (area, a)
    return float(best[1] % 90.0), float(best[0])


def min_area_angle_world(xy):
    """Yaw (deg, world x->y, in [0,90)) of the min-area rectangle of float 2-D points."""
    pts = np.asarray(xy, float) - np.mean(xy, 0)
    best = None
    for a in np.arange(0.0, 90.0, 1.0):
        t = np.deg2rad(a)
        c, s = np.cos(t), np.sin(t)
        p = pts @ np.array([[c, -s], [s, c]])
        area = np.ptp(p[:, 0]) * np.ptp(p[:, 1])
        if best is None or area < best[0]:
            best = (area, a)
    a0 = best[1]
    for a in np.arange(a0 - 1.0, a0 + 1.0, 0.1):
        t = np.deg2rad(a)
        c, s = np.cos(t), np.sin(t)
        p = pts @ np.array([[c, -s], [s, c]])
        area = np.ptp(p[:, 0]) * np.ptp(p[:, 1])
        if area < best[0]:
            best = (area, a)
    return float(best[1] % 90.0)


def split_blob(b, shape, max_it=10, min_part=12):
    """Split a blob of touching same-coloured objects by iterative erosion.
    Returns a list of blob dicts (the original blob alone if it does not split)."""
    m = np.zeros(shape, bool)
    m[b['ys'], b['xs']] = True
    for it in range(1, max_it + 1):
        e = erode(m, it)
        if not e.any():
            break
        parts = blobs(e, min_part // 2 + 1)
        if len(parts) >= 2:
            out = []
            for pt in parts:
                pm = np.zeros(shape, bool)
                pm[pt['ys'], pt['xs']] = True
                pm = dilate(pm, it) & m
                bb = blobs(pm, min_part)
                if bb:
                    out.append(bb[0])
            if len(out) >= 2:
                return out
    return [b]


def extent_world(xy, yaw_deg):
    """Robust side lengths (m) of 2-D points along a rectangle at yaw_deg."""
    a = np.deg2rad(yaw_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pr = (np.asarray(xy, float) - np.mean(xy, 0)) @ R
    return (float(np.percentile(pr[:, 0], 97) - np.percentile(pr[:, 0], 3)),
            float(np.percentile(pr[:, 1], 97) - np.percentile(pr[:, 1], 3)))
