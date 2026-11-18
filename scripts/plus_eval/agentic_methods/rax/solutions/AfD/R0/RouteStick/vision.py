"""Image analysis for RouteStick (numpy only; scipy optional)."""
import numpy as np

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None


def label(mask):
    if _ndi is not None:
        lab, n = _ndi.label(mask)
        return lab, n
    # simple 4-connected BFS fallback
    h, w = mask.shape
    lab = np.zeros((h, w), dtype=np.int32)
    n = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]:
            continue
        n += 1
        stack = [(y0, x0)]
        lab[y0, x0] = n
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n
                    stack.append((yy, xx))
    return lab, n


def red_mask(rgb):
    a = rgb.astype(np.int32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r > 100) & (g < 0.35 * r) & (b < 0.4 * r)


def white_blobs(rgb, top_exclude=0):
    a = rgb.astype(np.int32)
    mn = a.min(-1)
    mx = a.max(-1)
    mask = (mn > 140) & (mx - mn < 45)
    if top_exclude:
        mask[:top_exclude] = False
    lab, n = label(mask)
    out = []
    for k in range(1, n + 1):
        sel = lab == k
        area = int(sel.sum())
        if area < 12 or area > 400:
            continue
        ys, xs = np.nonzero(sel)
        p90 = float(np.percentile(mn[sel], 90))
        sat = float((mx - mn)[sel].mean())
        w = xs.max() - xs.min() + 1
        h = ys.max() - ys.min() + 1
        out.append(dict(x=float(xs.mean()), y=float(ys.mean()), area=area, p90=p90, sat=sat, w=int(w), h=int(h)))
    return out


def detect_targets(rgb, top_exclude=20, expected=None):
    """Return target disc centres (x, y) in image pixels ordered along the row
    (left to right in image).  Candidates: bright, unsaturated, compact blobs,
    then the largest roughly collinear subset."""
    cands = [b for b in white_blobs(rgb, top_exclude) if b['p90'] > 205 and b['sat'] < 28
             and b['w'] <= 3 * b['h'] + 2 and b['h'] <= 1.6 * b['w'] + 2 and b['area'] <= 200]
    pts = np.array([[b['x'], b['y']] for b in cands]) if cands else np.zeros((0, 2))
    if len(pts) <= 2:
        best = list(range(len(pts)))
    else:
        best, best_score = None, -1
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = pts[j] - pts[i]
                L = np.linalg.norm(d)
                if L < 5:
                    continue
                nrm = np.array([-d[1], d[0]]) / L
                res = np.abs((pts - pts[i]) @ nrm)
                inl = np.nonzero(res < 6.0)[0]
                score = len(inl) - 0.01 * res[inl].sum()
                if score > best_score:
                    best_score, best = score, list(inl)
    sel = pts[best]
    if len(sel) >= 2:
        # order along principal direction, oriented left->right in image
        c = sel.mean(0)
        u, s, vt = np.linalg.svd(sel - c)
        d = vt[0]
        if d[0] < 0:
            d = -d
        order = np.argsort((sel - c) @ d)
        sel = sel[order]
    return [tuple(map(float, p)) for p in sel]


def red_counts(rgb, targets, radius):
    m = red_mask(rgb)
    h, w = m.shape
    yy, xx = np.mgrid[0:h, 0:w]
    out = []
    for (x, y) in targets:
        sel = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        out.append(int(m[sel].sum()))
    return out
