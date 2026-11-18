"""Pure image-processing helpers for RouteStick (numpy only; cv2 optional)."""
import numpy as np

try:
    import cv2  # optional
except Exception:  # pragma: no cover
    cv2 = None


def hsv(a):
    a = a.astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1); mn = a.min(-1); d = mx - mn + 1e-6
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    s = d / (mx + 1e-6)
    return h, s, mx


def red_mask(a):
    h, s, v = hsv(a)
    return ((h < 12) | (h > 345)) & (s > 0.55) & (v > 90)


def white_mask(a, vmin=150, smax=0.25):
    h, s, v = hsv(a)
    return (s < smax) & (v > vmin)


def components(mask):
    """Return list of dicts (area, bbox, centroid(u,v), pixels) of 8-connected components."""
    mask = mask.astype(np.uint8)
    out = []
    if cv2 is not None:
        n, lab, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for k in range(1, n):
            x, y, w, h, area = [int(t) for t in st[k]]
            out.append(dict(area=area, bbox=(x, y, w, h), c=(float(cen[k][0]), float(cen[k][1])), label=k, lab=lab))
        return out
    # fallback flood fill
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32); k = 0
    for y0 in range(H):
        for x0 in range(W):
            if mask[y0, x0] and not lab[y0, x0]:
                k += 1; stack = [(y0, x0)]; lab[y0, x0] = k; pts = []
                while stack:
                    y, x = stack.pop(); pts.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            yy, xx = y + dy, x + dx
                            if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                                lab[yy, xx] = k; stack.append((yy, xx))
                pts = np.array(pts)
                ys, xs = pts[:, 0], pts[:, 1]
                out.append(dict(area=len(pts), bbox=(int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)),
                                c=(float(xs.mean()), float(ys.mean())), label=k, lab=lab))
    return out


def disk_candidates(front, vmin_row=28):
    """Small whitish elliptical blobs = candidate disks (front image)."""
    comps = components(white_mask(front))
    cands = []
    for c in comps:
        x, y, w, h = c['bbox']
        if c['area'] < 15 or c['area'] > 140:
            continue
        if y < vmin_row:
            continue
        if w > 20 or h > 16:
            continue
        cands.append(dict(u=c['c'][0], v=c['c'][1], area=c['area'], w=w, h=h))
    return cands


def fit_line(pts):
    pts = np.asarray(pts, float)
    m = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - m)
    d = vt[0]
    if d[0] < 0:
        d = -d
    return m, d


def choose_row(cands, n=5, max_resid=None):
    """Choose n candidates best forming a row (collinear, roughly even spacing)."""
    import itertools
    if len(cands) < n:
        return None
    best = None
    P = np.array([[c['u'], c['v']] for c in cands])
    idx_all = range(len(cands))
    combos = itertools.combinations(idx_all, n) if len(cands) <= 12 else [tuple(np.argsort([-c['area'] for c in cands])[:n])]
    for comb in combos:
        pts = P[list(comb)]
        m, d = fit_line(pts)
        t = (pts - m) @ d
        nrm = np.array([-d[1], d[0]])
        resid = np.abs((pts - m) @ nrm)
        order = np.argsort(t)
        gaps = np.diff(t[order])
        if gaps.min() <= 0:
            continue
        cost = resid.max() + 0.5 * (gaps.max() - gaps.min())
        if best is None or cost < best[0]:
            best = (cost, [comb[i] for i in order])
    if best is None:
        return None
    return best[1], best[0]
