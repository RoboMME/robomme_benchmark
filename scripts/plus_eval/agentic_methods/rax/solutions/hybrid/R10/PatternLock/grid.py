import numpy as np
from scipy import ndimage


def detect_discs(P, valid, zmin=0.006, zmax=0.035, min_px=4):
    """Return list of (centroid_world_xyz, pixel_count, (u,v)) for raised discs."""
    z = P[..., 2]
    mask = valid & (z > zmin) & (z < zmax)
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        sel = lab == i
        cnt = int(sel.sum())
        if cnt < min_px:
            continue
        pts = P[sel]
        vs, us = np.nonzero(sel)
        tz = float(np.percentile(pts[:, 2], 80))
        top = pts[np.abs(pts[:, 2] - tz) < 0.002]
        if len(top) < 3:
            top = pts
        c = np.array([top[:, 0].mean(), top[:, 1].mean(), tz])
        out.append((c, cnt, (float(us.mean()), float(vs.mean()))))
    return out


def fit_grid(centers, n=5):
    """centers: Kx2 world xy. Return dict (i,j)->xy and axes. i along world x-ish axis, j along y-ish axis."""
    C = np.asarray(centers, float)
    # nearest-neighbour vectors to estimate spacing and orientation
    D = np.linalg.norm(C[:, None] - C[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    s = float(np.median(nn))
    vecs = []
    for a in range(len(C)):
        for b in range(len(C)):
            if a != b and D[a, b] < 1.3 * s:
                v = C[b] - C[a]
                ang = np.arctan2(v[1], v[0]) % (np.pi / 2)  # fold into [0, 90)
                vecs.append(ang)
    vecs = np.array(vecs)
    # circular mean on 4-fold symmetry
    th = np.angle(np.mean(np.exp(4j * vecs))) / 4.0
    ex = np.array([np.cos(th), np.sin(th)])
    ey = np.array([-np.sin(th), np.cos(th)])
    # choose ex as the axis closest to world +x
    if abs(ex[0]) < abs(ey[0]):
        ex, ey = ey, -ex
    if ex[0] < 0:
        ex = -ex
    if ey[1] < 0:
        ey = -ey
    ctr = C.mean(0)
    a = (C - ctr) @ ex / s
    b = (C - ctr) @ ey / s
    return dict(center=ctr, ex=ex, ey=ey, spacing=s, a=a, b=b, theta=th)


def grid_points(g, n=5):
    """Ideal grid lattice positions (i,j)->xy, i=row along ex index 0..n-1, j along ey."""
    pts = {}
    h = (n - 1) / 2.0
    for i in range(n):
        for j in range(n):
            pts[(i, j)] = g["center"] + (i - h) * g["spacing"] * g["ex"] + (j - h) * g["spacing"] * g["ey"]
    return pts


def refine_grid(centers, n=5):
    """Fit grid, then snap lattice to detected centres (least squares on translation)."""
    C = np.asarray(centers, float)
    g = fit_grid(C, n)
    for _ in range(3):
        L = grid_points(g, n)
        keys = list(L)
        Lp = np.array([L[k] for k in keys])
        # match each detection to nearest lattice point
        d = np.linalg.norm(C[:, None] - Lp[None], axis=-1)
        idx = d.argmin(1)
        ok = d[np.arange(len(C)), idx] < 0.4 * g["spacing"]
        if ok.sum() < 4:
            break
        shift = (C[ok] - Lp[idx[ok]]).mean(0)
        g["center"] = g["center"] + shift
    L = grid_points(g, n)
    # replace lattice points by matched detections when close
    keys = list(L)
    Lp = np.array([L[k] for k in keys])
    d = np.linalg.norm(C[:, None] - Lp[None], axis=-1)
    matched = {}
    for ci in range(len(C)):
        k = int(d[ci].argmin())
        if d[ci, k] < 0.35 * g["spacing"]:
            matched[keys[k]] = C[ci]
    g["lattice"] = L
    g["matched"] = matched
    return g


def nearest_index(g, xy):
    best, bd = None, 1e9
    for k, p in g["lattice"].items():
        dd = float(np.linalg.norm(np.asarray(xy) - p))
        if dd < bd:
            best, bd = k, dd
    return best, bd
