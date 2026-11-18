"""World-frame perception: table plane, static mask, foreground points, cubes, containers."""
import re

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

COLOR_WORDS = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'black', 'white',
               'cyan', 'brown', 'gray', 'grey', 'magenta', 'violet']


def table_z_from_points(z):
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    if len(z) < 50:
        return None
    lo, hi = np.percentile(z, 2), np.percentile(z, 98)
    if hi - lo < 1e-4:
        return float(np.median(z))
    bins = np.arange(lo, hi + 0.002, 0.002)
    h, e = np.histogram(z, bins)
    k = int(h.argmax())
    sel = (z >= e[k] - 0.002) & (z <= e[k + 1] + 0.002)
    return float(np.median(z[sel]))


class Grid:
    def __init__(self, xmin, ymin, xmax, ymax, g):
        self.x0, self.y0, self.g = float(xmin), float(ymin), float(g)
        self.nx = max(1, int(np.ceil((xmax - xmin) / g)))
        self.ny = max(1, int(np.ceil((ymax - ymin) / g)))

    def idx(self, xy):
        ix = np.floor((xy[:, 0] - self.x0) / self.g).astype(int)
        iy = np.floor((xy[:, 1] - self.y0) / self.g).astype(int)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return ix, iy, ok

    def occupancy(self, xy):
        occ = np.zeros((self.nx, self.ny), bool)
        ix, iy, ok = self.idx(xy)
        occ[ix[ok], iy[ok]] = True
        return occ

    def lookup(self, occ, xy):
        ix, iy, ok = self.idx(xy)
        out = np.zeros(len(xy), bool)
        out[ok] = occ[ix[ok], iy[ok]]
        return out, ok


def dilate(occ, r=1):
    if r <= 0:
        return occ
    if cv2 is not None:
        k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
        return cv2.dilate(occ.astype(np.uint8), k) > 0
    out = occ.copy()
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            out |= np.roll(np.roll(occ, dx, 0), dy, 1)
    return out


def voxel_xy(xy, g=0.004):
    if len(xy) == 0:
        return xy.reshape(0, 2)
    q = np.floor(xy / g).astype(np.int64)
    key = q[:, 0] * 1000003 + q[:, 1]
    _, first = np.unique(key, return_index=True)
    return (q[first] + 0.5) * g


def rgb_to_hsv(rgb):
    rgb = np.asarray(rgb, np.uint8).reshape(-1, 1, 3)
    if cv2 is not None:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(float)
    # fallback, OpenCV scale (H 0..180, S,V 0..255)
    a = rgb.reshape(-1, 3).astype(float) / 255
    mx, mn = a.max(1), a.min(1)
    d = mx - mn + 1e-9
    h = np.zeros(len(a))
    r, g, b = a.T
    m = mx == r; h[m] = ((g - b)[m] / d[m]) % 6
    m = mx == g; h[m] = ((b - r)[m] / d[m]) + 2
    m = mx == b; h[m] = ((r - g)[m] / d[m]) + 4
    return np.stack([h * 30, np.where(mx > 0, d / (mx + 1e-9), 0) * 255, mx * 255], 1)


def color_name(rgb_pixels):
    """Name the dominant color of a set of RGB pixels (N,3)."""
    px = np.asarray(rgb_pixels, float).reshape(-1, 3)
    if len(px) == 0:
        return 'unknown', 0.0
    hsv = rgb_to_hsv(px)
    S, V = hsv[:, 1], hsv[:, 2]
    sat = (S > 90) & (V > 50)
    if sat.mean() > 0.3:
        H = hsv[sat, 0]
        votes = {}
        for h in H:
            if h < 7 or h >= 165:
                n = 'red'
            elif h < 19:
                n = 'orange'
            elif h < 35:
                n = 'yellow'
            elif h < 88:
                n = 'green'
            elif h < 100:
                n = 'cyan'
            elif h < 132:
                n = 'blue'
            elif h < 150:
                n = 'purple'
            else:
                n = 'pink'
            votes[n] = votes.get(n, 0) + 1
        n = max(votes, key=votes.get)
        conf = votes[n] / max(1, len(H))
        # purple cubes are often moderately saturated with hue ~130-160
        return n, float(conf * sat.mean())
    Vm = np.median(V)
    Sm = np.median(S)
    if Vm < 70:
        return 'black', 0.5
    if Sm < 60 and Vm > 110:
        return 'white', 0.5
    if Sm < 60:
        return 'gray', 0.3
    # moderately saturated: use median hue
    h = np.median(hsv[:, 0])
    if 128 <= h < 165:
        return 'purple', 0.3
    return 'unknown', 0.0


def parse_instruction(text):
    """-> list of (color, all_flag) in the order requested."""
    t = (text or '').lower()
    out = []
    # split into clauses at commas / 'followed by' / 'then' / 'and finally'
    clauses = re.split(r',|;|\bfollowed by\b|\bthen\b|\band finally\b|\bfinally\b|\band\b', t)
    for c in clauses:
        for w in re.findall(r'[a-z]+', c):
            w2 = 'gray' if w == 'grey' else w
            if w2 in COLOR_WORDS:
                allf = bool(re.search(r'\ball\b|\bevery\b|\beach\b|containers|cubes', c))
                if not out or out[-1][0] != w2 or allf:
                    out.append((w2, allf))
                break
    # remove video-related color mentions? keep as is
    return out


def components(occ, min_cells=2):
    occ8 = occ.astype(np.uint8)
    if cv2 is not None:
        n, lab, st, _ = cv2.connectedComponentsWithStats(occ8, connectivity=8)
        return n, lab, st[:, cv2.CC_STAT_AREA]
    # simple BFS fallback
    lab = np.zeros(occ.shape, int)
    n = 1
    areas = [0]
    for i, j in zip(*np.nonzero(occ)):
        if lab[i, j]:
            continue
        stack = [(i, j)]; lab[i, j] = n; a = 0
        while stack:
            x, y = stack.pop(); a += 1
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < occ.shape[0] and 0 <= yy < occ.shape[1] and occ[xx, yy] and not lab[xx, yy]:
                        lab[xx, yy] = n; stack.append((xx, yy))
        areas.append(a); n += 1
    return n, lab, np.array(areas)


def square_yaw(xy):
    """Orientation (rad, modulo pi/2) of a square footprint from its points."""
    xy = np.asarray(xy, float)
    if len(xy) < 5:
        return None
    best, ba = None, 1e9
    c = xy.mean(0)
    d = xy - c
    for a in np.deg2rad(np.arange(0, 90, 2.0)):
        ca, sa = np.cos(a), np.sin(a)
        p = d[:, 0] * ca + d[:, 1] * sa
        q = -d[:, 0] * sa + d[:, 1] * ca
        ext = (np.percentile(p, 97) - np.percentile(p, 3)) * (np.percentile(q, 97) - np.percentile(q, 3))
        if ext < ba:
            ba, best = ext, a
    return float(best)


def square_size(xy):
    xy = np.asarray(xy, float)
    if len(xy) < 5:
        return None
    v = xy.var(0).sum() / 2
    return float(np.sqrt(12 * v))
