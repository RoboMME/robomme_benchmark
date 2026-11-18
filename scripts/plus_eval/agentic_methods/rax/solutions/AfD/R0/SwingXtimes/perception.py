"""Pure image perception for SwingXtimes (numpy arrays HxWx3 uint8). No I/O."""
import numpy as np

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None


def label(mask):
    """Connected components (8-connectivity). Returns (labels, n)."""
    mask = np.asarray(mask, bool)
    if _ndi is not None:
        lab, n = _ndi.label(mask, structure=np.ones((3, 3), int))
        return lab, int(n)
    h, w = mask.shape
    lab = np.zeros((h, w), int)
    n = 0
    for y0 in range(h):
        for x0 in range(w):
            if mask[y0, x0] and lab[y0, x0] == 0:
                n += 1
                stack = [(y0, x0)]
                lab[y0, x0] = n
                while stack:
                    y, x = stack.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            yy, xx = y + dy, x + dx
                            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and lab[yy, xx] == 0:
                                lab[yy, xx] = n
                                stack.append((yy, xx))
    return lab, n


def dilate(mask, it=1):
    mask = np.asarray(mask, bool)
    if _ndi is not None:
        return _ndi.binary_dilation(mask, iterations=it)
    m = mask.copy()
    for _ in range(it):
        p = np.pad(m, 1)
        m = np.zeros_like(m)
        for dy in range(3):
            for dx in range(3):
                m |= p[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return m


def closing(mask, it=1):
    mask = np.asarray(mask, bool)
    if _ndi is not None:
        return _ndi.binary_closing(mask, iterations=it)
    d = dilate(mask, it)
    return ~dilate(~d, it)


def blobs(mask, min_px=1):
    lab, n = label(mask)
    out = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        if len(ys) < min_px:
            continue
        out.append(dict(n=int(len(ys)), cx=float(xs.mean()), cy=float(ys.mean()),
                        x0=int(xs.min()), x1=int(xs.max()), y0=int(ys.min()), y1=int(ys.max()),
                        xs=xs, ys=ys))
    out.sort(key=lambda b: -b['n'])
    return out


def color_mask(img, color):
    a = np.asarray(img).astype(np.int32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    if color == 'red':
        return (r > 55) & (g * 4 < r) & (b * 4 < r)
    if color == 'green':
        return (g > 55) & (r * 3 < g) & (b * 3 < g)
    if color == 'blue':
        return (b > 55) & (r * 3 < b) & (g * 3 < b)
    if color == 'yellow':
        return (r > 90) & (g > 90) & (b * 3 < np.minimum(r, g)) & (np.abs(r - g) < 0.25 * r)
    if color == 'purple':
        return (r > 60) & (b > 60) & (g * 2 < np.minimum(r, b))
    if color == 'orange':
        return (r > 120) & (g > 40) & (g < 0.65 * r) & (b * 4 < r)
    return np.zeros(a.shape[:2], bool)


def find_cube(img, color, min_px=6, max_px=20000, ymin=0, prefer=None):
    """Largest blob of the cube colour. prefer=(x,y) picks the blob nearest to it."""
    m = color_mask(img, color)
    m[:ymin] = False
    bl = [b for b in blobs(m, min_px) if b['n'] <= max_px]
    if not bl:
        return None
    if prefer is not None:
        px, py = prefer
        bl.sort(key=lambda b: (b['cx'] - px) ** 2 + (b['cy'] - py) ** 2 - 4 * b['n'])
    return bl[0]


def table_brightness(img):
    a = np.asarray(img).astype(np.float32)
    lower = a[int(a.shape[0] * 0.55):]
    return float(np.median(lower.mean(axis=2)))


def neutral_mask(img, rel=0.80, sat=0.22):
    a = np.asarray(img).astype(np.float32)
    mx, mn, mean = a.max(2), a.min(2), a.mean(2)
    tb = table_brightness(img)
    return ((mx - mn) < sat * mx + 8) & (mean > max(45.0, rel * tb))


def find_fixtures(img, top_exclude=24):
    """Candidate targets/button: neutral (white/grey) blobs on the table.
    Returns list of blob dicts (largest first) with shape stats."""
    res = []
    for rel, sat in ((0.80, 0.22), (0.70, 0.28), (0.95, 0.18), (0.60, 0.32)):
        m = closing(neutral_mask(img, rel, sat), 1)
        m[:top_exclude] = False
        bl = []
        for b in blobs(m, 20):
            w = b['x1'] - b['x0'] + 1
            h = b['y1'] - b['y0'] + 1
            if b['n'] > 2500 or w > 70 or h > 60 or w < 6 or h < 4:
                continue
            if b['y0'] <= top_exclude + 1 and b['n'] > 400:
                continue
            b['w'], b['h'] = w, h
            b['fill'] = b['n'] / float(w * h)
            bl.append(b)
        res = bl
        if len(bl) >= 3:
            break
    return res


def classify_fixtures(cands, heights=None):
    """Pick the button and two targets among candidate blobs.
    heights: optional list of measured world z (top) per candidate (None if unknown).
    Returns (button, [t1, t2]) or raises ValueError."""
    if len(cands) < 3:
        raise ValueError('need >=3 fixture candidates, got %d' % len(cands))
    cands = cands[:8]
    best = None
    idx = range(len(cands))
    import itertools
    for bi in idx:
        for ti, tj in itertools.combinations([i for i in idx if i != bi], 2):
            b, t1, t2 = cands[bi], cands[ti], cands[tj]
            # targets: similar size, similar shape
            size_ratio = max(t1['n'], t2['n']) / max(1.0, min(t1['n'], t2['n']))
            score = -abs(np.log(size_ratio)) * 2.0
            score += 0.002 * (t1['n'] + t2['n'] + b['n'])
            # targets wider than tall (flat discs seen obliquely)
            for t in (t1, t2):
                score += 0.8 * min(1.5, t['w'] / float(t['h'])) - 0.8
            # button: squarer, with a dark cap (lower fill of bright pixels inside bbox)
            score += 0.5 * (1.0 - abs(np.log(b['w'] / float(b['h']))))
            if heights is not None and None not in (heights[bi], heights[ti], heights[tj]):
                dz = heights[bi] - max(heights[ti], heights[tj])
                score += 40.0 * np.clip(dz, -0.03, 0.03)  # +1.2 for 3 cm taller
                if abs(heights[ti] - heights[tj]) > 0.02:
                    score -= 1.0
            if best is None or score > best[0]:
                best = (score, b, [t1, t2])
    return best[1], best[2]


def red_lit_fraction(img, box):
    """Fraction of strongly red pixels inside box=(x0,y0,x1,y1)."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    h, w = np.asarray(img).shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w - 1, x1), min(h - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    p = np.asarray(img)[y0:y1 + 1, x0:x1 + 1].astype(np.int32)
    r, g, b = p[..., 0], p[..., 1], p[..., 2]
    return float(((r > 90) & (r > 2.2 * g) & (r > 2.2 * b)).mean())


def sample_points(blob, k=32, margin=0.0, rng_seed=0):
    """Up to k pixel coordinates (u,v) spread over the blob."""
    xs, ys = blob['xs'], blob['ys']
    if len(xs) == 0:
        return []
    if margin > 0:
        cx, cy = xs.mean(), ys.mean()
        keep = ((xs - cx) ** 2 + (ys - cy) ** 2) <= (np.sqrt(len(xs) / np.pi) * (1 - margin)) ** 2
        if keep.sum() >= 4:
            xs, ys = xs[keep], ys[keep]
    n = len(xs)
    if n <= k:
        sel = np.arange(n)
    else:
        sel = np.linspace(0, n - 1, k).astype(int)
    return [(float(xs[i]), float(ys[i])) for i in sel]


def min_area_yaw(xy):
    """Yaw (rad, in [0, pi/2)) of the minimum-area bounding square/rectangle of 2-D points."""
    xy = np.asarray(xy, float)
    if len(xy) < 5:
        return None, None
    xy = xy - xy.mean(0)
    best = None
    for deg in np.arange(0.0, 90.0, 1.0):
        t = np.deg2rad(deg)
        c, s = np.cos(t), np.sin(t)
        u = xy[:, 0] * c + xy[:, 1] * s
        v = -xy[:, 0] * s + xy[:, 1] * c
        area = (u.max() - u.min()) * (v.max() - v.min())
        if best is None or area < best[0]:
            best = (area, t, (u.max() - u.min(), v.max() - v.min()))
    return best[1], best[2]
