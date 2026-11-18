"""Pure-numpy image segmentation for the SwingXtimes scene."""
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


def _i(img):
    a = np.asarray(img).astype(np.int32)
    return a[..., 0], a[..., 1], a[..., 2]


def color_mask(img, color):
    r, g, b = _i(img)
    if color == 'red':
        return (r > 70) & (g < 45) & (b < 45) & (r > 2.5 * np.maximum(g, b) + 10)
    if color == 'green':
        return (g > 80) & (r < 90) & (b < 90) & (g > 1.8 * np.maximum(r, b) + 10)
    if color == 'blue':
        return (b > 80) & (r < 70) & (g < 70) & (b > 1.8 * np.maximum(r, g) + 10)
    if color == 'yellow':
        return (r > 150) & (g > 150) & (b < 70)
    raise ValueError(color)


def neutral_mask(img, lo=75):
    """White/gray objects (targets, button, robot); tolerant to dim light and coloured light tints."""
    r, g, b = _i(img)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    return ((mx - mn) < np.maximum(30, 0.28 * mx)) & (mx > lo)


def dark_neutral_mask(img):
    r, g, b = _i(img)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    return ((mx - mn) < 25) & (mx > 65) & (mx < 138)


def wood_mask(img):
    r, g, b = _i(img)
    return (r > 110) & (r - g > 35) & (r - g < 120) & (g - b > 10) & (b < 140)


def pink_mask(img):
    r, g, b = _i(img)
    return (r > 215) & (g > 150) & (g < 232) & (b > 150) & (b < 235) & (r - g > 22) & (r - b > 22) & (np.abs(g - b) < 30)


def ringred_mask(img):
    r, g, b = _i(img)
    return (r > 130) & (g > 18) & (g < 75) & (b < 75) & (r - g > 90)


def label(mask):
    mask = np.asarray(mask, dtype=bool)
    if _ndi is not None:
        lab, n = _ndi.label(mask, structure=np.ones((3, 3), int))
        return lab, n
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    n = 0
    for y in range(H):
        for x in range(W):
            if mask[y, x] and lab[y, x] == 0:
                n += 1
                stack = [(y, x)]
                lab[y, x] = n
                while stack:
                    cy, cx = stack.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            yy, xx = cy + dy, cx + dx
                            if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and lab[yy, xx] == 0:
                                lab[yy, xx] = n
                                stack.append((yy, xx))
    return lab, n


def dilate(mask, it=1):
    m = np.asarray(mask, dtype=bool)
    for _ in range(it):
        p = np.pad(m, 1)
        m = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:] | \
            p[:-2, :-2] | p[2:, 2:] | p[:-2, 2:] | p[2:, :-2]
    return m


def components(mask, min_area=1):
    lab, n = label(mask)
    out = []
    if n == 0:
        return out
    ys, xs = np.nonzero(lab)
    ls = lab[ys, xs]
    order = np.argsort(ls, kind='stable')
    ys, xs, ls = ys[order], xs[order], ls[order]
    bounds = np.searchsorted(ls, np.arange(1, n + 2))
    for i in range(n):
        a, b = bounds[i], bounds[i + 1]
        if b - a < min_area:
            continue
        py, px = ys[a:b], xs[a:b]
        out.append({'area': int(b - a), 'cx': float(px.mean()), 'cy': float(py.mean()),
                    'x0': int(px.min()), 'x1': int(px.max()), 'y0': int(py.min()), 'y1': int(py.max()),
                    'ys': py, 'xs': px})
    out.sort(key=lambda c: -c['area'])
    return out


def sample_pixels(comp, n, rng=None, shrink=0.6):
    """Pick up to n pixels of a component, preferring its interior (near the centroid)."""
    xs, ys = comp['xs'], comp['ys']
    d = np.hypot(xs - comp['cx'], ys - comp['cy'])
    if len(d) == 0:
        return []
    lim = np.quantile(d, shrink) if len(d) > 4 else d.max()
    idx = np.nonzero(d <= lim + 1e-6)[0]
    if len(idx) > n:
        sel = idx[np.linspace(0, len(idx) - 1, n).astype(int)]
    else:
        sel = idx
    return [(float(xs[i]), float(ys[i])) for i in sel]


def min_area_angle(xs, ys):
    """Angle (rad, image coords u right / v down) of a min-area rectangle edge of a point set."""
    pts = np.stack([xs, ys], 1).astype(np.float32)
    if len(pts) < 5:
        return None
    if _cv2 is not None:
        try:
            (cx, cy), (w, h), ang = _cv2.minAreaRect(pts)
            return math.radians(ang)
        except Exception:
            pass
    best, ba = None, None
    c = pts.mean(0)
    q = pts - c
    for deg in np.arange(0, 90, 1.0):
        t = math.radians(deg)
        R = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])
        p = q @ R.T
        area = (p[:, 0].max() - p[:, 0].min()) * (p[:, 1].max() - p[:, 1].min())
        if best is None or area < best:
            best, ba = area, t
    return ba


def count_in_box(mask, box, pad=2):
    x0, x1, y0, y1 = box
    H, W = mask.shape
    xa, xb = max(0, int(x0) - pad), min(W, int(x1) + pad + 1)
    ya, yb = max(0, int(y0) - pad), min(H, int(y1) + pad + 1)
    return int(np.asarray(mask[ya:yb, xa:xb]).sum())
