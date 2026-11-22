import numpy as np
from scipy import ndimage

XMIN, XMAX, YMIN, YMAX = -0.35, 0.30, -0.40, 0.40
CELL = 0.005
STATIC = None  # boolean xy grid of static tall objects (occluders) to ignore


def grid_idx(P):
    ix = np.clip(((P[:, 0] - XMIN) / CELL).astype(int), 0, int((XMAX - XMIN) / CELL))
    iy = np.clip(((P[:, 1] - YMIN) / CELL).astype(int), 0, int((YMAX - YMIN) / CELL))
    return ix, iy


def set_static(P, dilate_cells=3):
    global STATIC
    H = int((XMAX - XMIN) / CELL) + 1; W = int((YMAX - YMIN) / CELL) + 1
    g = np.zeros((H, W), bool)
    if len(P):
        ix, iy = grid_idx(P)
        g[ix, iy] = True
        g = ndimage.binary_dilation(g, iterations=dilate_cells)
    STATIC = g
    return int(g.sum())


def gather(obs, zlo, zhi, cams=("front", "wrist"), ee_mask=None):
    """Return Nx3 points and Nx3 colors with z in [zlo,zhi] inside table region."""
    pts, cols = [], []
    for cam in cams:
        if cam not in obs.cams:
            continue
        c = obs.cams[cam]
        P = c["P"].reshape(-1, 3)
        C = c["rgb"].reshape(-1, 3)
        ok = np.isfinite(P).all(1)
        ok &= (P[:, 2] >= zlo) & (P[:, 2] <= zhi)
        ok &= (P[:, 0] > XMIN) & (P[:, 0] < XMAX) & (P[:, 1] > YMIN) & (P[:, 1] < YMAX)
        pts.append(P[ok]); cols.append(C[ok])
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3))
    P = np.concatenate(pts); C = np.concatenate(cols)
    if STATIC is not None and len(P):
        ix, iy = grid_idx(P)
        keep = ~STATIC[ix, iy]
        P, C = P[keep], C[keep]
    if ee_mask is not None and len(P):
        ee = ee_mask  # xyz of TCP; mask fingers (r<0.035) and hand body (r<0.11, above TCP+0.03)
        d = np.hypot(P[:, 0] - ee[0], P[:, 1] - ee[1])
        drop = (d < 0.035) | ((d < 0.11) & (P[:, 2] > ee[2] + 0.03))
        P, C = P[~drop], C[~drop]
    return P, C


def clusters(P, C, min_cells=4):
    """Connected components on an xy occupancy grid. Returns list of dict(xy, n, color, pts)."""
    if len(P) == 0:
        return []
    ix = ((P[:, 0] - XMIN) / CELL).astype(int)
    iy = ((P[:, 1] - YMIN) / CELL).astype(int)
    H = int((XMAX - XMIN) / CELL) + 1; W = int((YMAX - YMIN) / CELL) + 1
    occ = np.zeros((H, W), bool)
    occ[ix, iy] = True
    lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
    pl = lab[ix, iy]
    out = []
    for k in range(1, n + 1):
        m = pl == k
        cells = (lab == k).sum()
        if cells < min_cells:
            continue
        out.append(dict(xy=P[m, :2].mean(0), n=int(cells), color=np.median(C[m], 0),
                        ztop=float(np.percentile(P[m, 2], 95)), pts=P[m]))
    return out


def classify_color(rgb):
    r, g, b = [float(x) for x in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    if mx - mn < 40:
        return "white" if mx > 150 else ("black" if mx < 80 else "gray")
    if r > 150 and g > 195 and b < 100:
        return "yellow"
    if r > 150 and 60 < g <= 195 and b < 80:
        return "orange"
    if r > 120 and b > 120 and g < 100:
        return "purple"
    if r >= g and r >= b:
        return "red"
    if g >= r and g >= b:
        return "green"
    return "blue"


def kmeans_track(P, init, iters=6, radius=0.05):
    """Lloyd iterations in xy from init centers; points farther than radius from all centers ignored."""
    X = P[:, :2]
    c = init.copy()
    counts = np.zeros(len(c), int)
    if len(X) == 0:
        return c, counts
    for _ in range(iters):
        d = np.linalg.norm(X[:, None, :] - c[None, :, :], axis=2)
        a = d.argmin(1)
        dm = d.min(1)
        for k in range(len(c)):
            m = (a == k) & (dm < radius)
            counts[k] = m.sum()
            if counts[k] > 3:
                c[k] = X[m].mean(0)
    return c, counts
