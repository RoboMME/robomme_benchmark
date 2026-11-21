import numpy as np
from scipy import ndimage
from geom import backproject
from frames import frame_files, load_depth, load_rgb

RES = 0.002
X0, X1, Y0, Y1 = -0.45, 0.35, -0.4, 0.4
ZLO, ZHI = 0.004, 0.07

def object_points(frame, cams=('front', 'wrist'), zlo=ZLO, zhi=ZHI, want_rgb=False):
    """Unique 2mm XY cells occupied by points at object height. Returns (N,2) xy and (N,) z max."""
    arr = frame.get('_arrays')
    ff = None if arr is not None else frame_files(frame)
    pts = []
    for cam in cams:
        key = (cam, 'depth')
        if arr is not None:
            if key not in arr:
                continue
            D = arr[key]
        else:
            if key not in ff:
                continue
            D = load_depth(ff[key])
        P, uv = backproject(D, frame['camera_intrinsics'][cam]['matrix'], frame['camera_poses'][cam])
        m = (P[:, 2] > zlo) & (P[:, 2] < zhi) & (P[:, 0] > X0) & (P[:, 0] < X1) & (P[:, 1] > Y0) & (P[:, 1] < Y1)
        if want_rgb:
            rk = (cam, 'rgb')
            img = arr[rk] if arr is not None else load_rgb(ff[rk])
            uvm = uv[m].astype(int)
            cols = img[uvm[:, 1], uvm[:, 0]].astype(float)
            P = np.concatenate([P[m], cols], 1)
            pts.append(P)
        else:
            pts.append(P[m])
    if not pts:
        return np.zeros((0, 2)), np.zeros(0)
    P = np.concatenate(pts)
    if want_rgb:
        return P
    ij = np.floor((P[:, :2] - [X0, Y0]) / RES).astype(int)
    key = ij[:, 0] * 100000 + ij[:, 1]
    uk, inv = np.unique(key, return_inverse=True)
    zmax = np.full(len(uk), -1.0)
    np.maximum.at(zmax, inv, P[:, 2])
    xy = np.stack([uk // 100000, uk % 100000], 1) * RES + [X0, Y0] + RES / 2
    return xy, zmax

def clusters(xy, link=0.006, minpts=8):
    """Connected components of occupied cells (dilated by link). Returns list of (centroid, npts, indices)."""
    if len(xy) == 0:
        return []
    ij = np.floor((xy - [X0, Y0]) / RES).astype(int)
    H = int((X1 - X0) / RES) + 2; W = int((Y1 - Y0) / RES) + 2
    g = np.zeros((H, W), bool)
    g[ij[:, 0], ij[:, 1]] = True
    r = max(1, int(round(link / RES)))
    gd = ndimage.binary_dilation(g, iterations=r)
    lab, n = ndimage.label(gd)
    l = lab[ij[:, 0], ij[:, 1]]
    out = []
    for k in range(1, n + 1):
        idx = np.nonzero(l == k)[0]
        if len(idx) >= minpts:
            out.append((xy[idx].mean(0), len(idx), idx))
    return out

class Tracker:
    def __init__(self, centers, radius=0.025):
        self.c = np.array(centers, float)
        self.v = np.zeros_like(self.c)
        self.radius = radius
        self.lost = np.zeros(len(self.c), int)

    def update(self, xy, iters=6, fixed=None):
        if len(xy) == 0:
            self.lost += 1
            return
        pred = self.c + self.v
        c = pred.copy()
        for _ in range(iters):
            d = np.linalg.norm(xy[:, None, :] - c[None], axis=2)
            a = d.argmin(1)
            dm = d[np.arange(len(xy)), a]
            ok = dm < self.radius
            newc = c.copy()
            cnt = np.zeros(len(c), int)
            for k in range(len(c)):
                sel = ok & (a == k)
                cnt[k] = sel.sum()
                if cnt[k] >= 3:
                    newc[k] = xy[sel].mean(0)
            c = newc
        moved = cnt >= 3
        nv = c - self.c
        self.v = np.where(moved[:, None], 0.5 * self.v + 0.5 * nv, self.v * 0.5)
        self.c = np.where(moved[:, None], c, pred)
        self.lost = np.where(moved, 0, self.lost + 1)
        self.cnt = cnt

def grasp_from_cells(cells, z):
    """Pick grasp point on material and closing direction for a cluster of xy cells."""
    c = cells.mean(0)
    d = np.linalg.norm(cells - c, axis=1)
    # material near centroid: mean of cells within 4mm of the nearest cell
    j = d.argmin()
    near = np.linalg.norm(cells - cells[j], axis=1) < 0.005
    g = cells[near].mean(0)
    n = g - c
    if np.linalg.norm(n) < 0.003:
        # convex-ish: use principal minor axis as closing direction
        C = np.cov((cells - c).T)
        w, V = np.linalg.eigh(C)
        n = V[:, 0]
    n = n / (np.linalg.norm(n) + 1e-9)
    zt = float(np.percentile(z, 90))
    return g, n, zt, c

def major_axis(cells):
    """Angle (rad, world xy) of the principal major axis of a cluster's cells, and elongation ratio."""
    c = cells - cells.mean(0)
    C = np.cov(c.T)
    w, V = np.linalg.eigh(C)
    v = V[:, 1]
    return float(np.arctan2(v[1], v[0])), float(np.sqrt(max(w[1], 1e-12) / max(w[0], 1e-12)))
