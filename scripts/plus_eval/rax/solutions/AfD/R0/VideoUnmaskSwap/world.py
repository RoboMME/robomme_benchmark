"""World-frame perception: container / cube points from RGB-D, clustering, fused XY tracking."""
import math
import numpy as np

import tracker as T2

SAT_MAX = T2.SAT_MAX
BRIGHT_MIN = T2.BRIGHT_MIN
FG_DIFF = T2.FG_DIFF
CELL = 0.005          # XY grid cell (m) for clustering / static map


def lowsat_bright(rgb):
    im = rgb.astype(np.int16)
    return ((im.max(2) - im.min(2)) < SAT_MAX) & (im.mean(2) > BRIGHT_MIN)


def changed(rgb, bg):
    return np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).max(2) > FG_DIFF


def pix_to_world(cam, depth, mask, stride=1, rgb=None):
    """World points of mask pixels with valid depth. Returns (P (n,3), colors (n,3) or None, uv)."""
    m = mask.copy()
    if stride > 1:
        sub = np.zeros_like(m)
        sub[::stride, ::stride] = True
        m &= sub
    if depth is not None:
        m &= (depth > 0.01) & np.isfinite(depth)
    vs, us = np.nonzero(m)
    if len(vs) == 0:
        return np.zeros((0, 3)), (np.zeros((0, 3)) if rgb is not None else None), np.zeros((0, 2))
    d = depth[vs, us]
    P = cam.backproject(us.astype(float), vs.astype(float), d)
    C = rgb[vs, us].astype(float) if rgb is not None else None
    return P, C, np.stack([us, vs], 1)


def table_points(cam, rgb, depth):
    im = rgb.astype(np.int16)
    wood = (im.max(2) - im.min(2)) > 60
    P, _, _ = pix_to_world(cam, depth, wood, stride=2)
    return P


def plane_stats(P):
    if len(P) < 50:
        return None
    z = P[:, 2]
    med = float(np.median(z))
    mad = float(np.median(np.abs(z - med)))
    # tilt: fit z = ax + by + c
    A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    try:
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        tilt = float(math.hypot(coef[0], coef[1]))
    except Exception:
        tilt = 1.0
    return dict(z=med, mad=mad, tilt=tilt, n=len(P))


# ------------------------------------------------------------ clustering
def grid_clusters(XY, cell=CELL, min_pts=5, link=1):
    """Connected components of occupied XY cells. Returns list of index arrays."""
    if len(XY) == 0:
        return []
    ij = np.floor(XY / cell).astype(np.int64)
    keys = {}
    for n, (i, j) in enumerate(map(tuple, ij)):
        keys.setdefault((i, j), []).append(n)
    seen = set()
    clusters = []
    for k in keys:
        if k in seen:
            continue
        stack = [k]
        seen.add(k)
        idx = []
        while stack:
            c = stack.pop()
            idx.extend(keys[c])
            for di in range(-link, link + 1):
                for dj in range(-link, link + 1):
                    nb = (c[0] + di, c[1] + dj)
                    if nb in keys and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        if len(idx) >= min_pts:
            clusters.append(np.array(idx))
    return clusters


def square_fit(XY):
    """Orientation (rad, in [0, pi/2)) and side lengths of the min-area rectangle of points."""
    if len(XY) < 5:
        return 0.0, (0.0, 0.0)
    c = XY - XY.mean(0)
    best = None
    for deg in range(0, 90, 2):
        a = math.radians(deg)
        R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        q = c @ R.T
        lo = np.percentile(q, 2, axis=0)
        hi = np.percentile(q, 98, axis=0)
        ext = hi - lo
        area = ext[0] * ext[1]
        if best is None or area < best[0]:
            best = (area, a, tuple(ext.tolist()))
    return best[1], best[2]


def classify_pixels(C):
    """Per-pixel color class names for an (n,3) array."""
    return [T2.classify_color(c) for c in C]


def color_vote(C, weights=None):
    if len(C) == 0:
        return 'unknown', {}
    names = classify_pixels(C)
    w = np.ones(len(names)) if weights is None else weights
    tally = {}
    for n, ww in zip(names, w):
        tally[n] = tally.get(n, 0.0) + ww
    # 'gray' pixels are shading of white cubes / ambiguous
    best = max(tally, key=lambda k: tally[k])
    return best, tally


# ------------------------------------------------------------ tracker in XY
class XYTracker:
    def __init__(self, centers, side):
        self.c = np.array(centers, float)
        self.v = np.zeros_like(self.c)      # m/s
        self.side = side
        self.R = 0.8 * side
        self.lost = np.zeros(len(self.c), int)
        self.t = None
        self.npts = np.zeros(len(self.c), int)

    def update(self, XY, t, iters=5, min_pts=6):
        dt = 0.05 if self.t is None else max(1e-3, min(0.5, t - self.t))
        self.t = t
        pred = self.c + self.v * dt
        if len(XY) == 0:
            self.c = pred
            self.v *= 0.5
            self.lost += 1
            return
        c = pred.copy()
        got = np.zeros(len(c), bool)
        cnt = np.zeros(len(c), int)
        for _ in range(iters):
            d = np.linalg.norm(XY[:, None, :] - c[None], axis=2)
            a = d.argmin(1)
            dm = d[np.arange(len(XY)), a]
            newc = c.copy()
            for k in range(len(c)):
                sel = (a == k) & (dm < self.R)
                cnt[k] = int(sel.sum())
                if cnt[k] >= min_pts:
                    newc[k] = XY[sel].mean(0)
                    got[k] = True
                else:
                    got[k] = False
            c = newc
        # keep tracks from collapsing onto each other
        for k in range(len(c)):
            for j in range(k + 1, len(c)):
                dd = np.linalg.norm(c[k] - c[j])
                if dd < 0.6 * self.side:
                    # revert the one that moved most from its prediction
                    if np.linalg.norm(c[k] - pred[k]) > np.linalg.norm(c[j] - pred[j]):
                        c[k] = pred[k]; got[k] = False
                    else:
                        c[j] = pred[j]; got[j] = False
        for k in range(len(c)):
            if got[k]:
                vel = (c[k] - self.c[k]) / dt
                self.v[k] = 0.6 * self.v[k] + 0.4 * vel
                self.c[k] = c[k]
                self.lost[k] = 0
            else:
                self.c[k] = pred[k]
                self.v[k] *= 0.6
                self.lost[k] += 1
        self.npts = cnt


def hungarian(cost):
    return T2.hungarian(np.asarray(cost, float))
