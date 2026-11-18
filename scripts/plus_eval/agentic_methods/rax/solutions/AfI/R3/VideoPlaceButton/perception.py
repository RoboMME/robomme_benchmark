"""Colour segmentation + depth back-projection. Pure functions over numpy arrays."""
import numpy as np
from geom import Cam

COLOR_RULES = {
    # rgb int arrays -> bool masks
    "red": lambda r, g, b: (r > 140) & (g < 70) & (b < 70),
    "green": lambda r, g, b: (g > 120) & (r < 90) & (b < 90),
    "blue": lambda r, g, b: (b > 120) & (r < 70) & (g < 90),
    "yellow": lambda r, g, b: (r > 160) & (g > 160) & (b < 80),
    "pink": lambda r, g, b: (r > 150) & (b > 100) & (b - g > 20),
    # fallback for target rings of another colour: their white rings
    "white": lambda r, g, b: (r > 200) & (g > 200) & (b > 200) & (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b) < 25),
}


def mask(rgb, color):
    a = rgb.astype(int)
    return COLOR_RULES[color](a[..., 0], a[..., 1], a[..., 2])


def world_points(rgb, depth_m, cam, color):
    m = mask(rgb, color) & (depth_m > 0)
    vv, uu = np.nonzero(m)
    if len(vv) == 0:
        return np.zeros((0, 3)), uu, vv
    P = cam.deproject(uu + 0.0, vv + 0.0, depth_m[vv, uu])
    return P, uu, vv


def cluster_xy(P, radius=0.05, min_pts=8):
    """Greedy single-link-ish clustering on xy. Returns list of index arrays (largest first)."""
    n = len(P)
    if n == 0:
        return []
    labels = -np.ones(n, int)
    xy = P[:, :2]
    cur = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        stack = [i]; labels[i] = cur
        while stack:
            j = stack.pop()
            d = np.linalg.norm(xy - xy[j], axis=1)
            nb = np.nonzero((d < radius * 0.25) & (labels < 0))[0]
            labels[nb] = cur
            stack.extend(nb.tolist())
        cur += 1
    out = [np.nonzero(labels == k)[0] for k in range(cur)]
    out = [o for o in out if len(o) >= min_pts]
    out.sort(key=len, reverse=True)
    return out


def subsample(P, k=3000, seed=0):
    if len(P) <= k:
        return P
    idx = np.random.default_rng(seed).choice(len(P), k, replace=False)
    return P[idx]


def find_targets(rgb, depth_m, cam, color="pink"):
    """Ring targets lying on the table: list of dict(center xy, n)."""
    P, _, _ = world_points(rgb, depth_m, cam, color)
    P = P[(P[:, 2] < (0.03 if color == "pink" else 0.015)) & (P[:, 2] > -0.02)]
    P = subsample(P, 4000)
    res = []
    for idx in cluster_xy(P, radius=0.05, min_pts=10):
        Q = P[idx]
        c = (Q[:, :2].min(0) + Q[:, :2].max(0)) / 2
        res.append({"xy": c, "mean": Q[:, :2].mean(0), "n": int(len(idx)),
                    "extent": (Q[:, :2].max(0) - Q[:, :2].min(0))})
    return res


def find_cubes(rgb, depth_m, cam, color, zmin=0.005, zmax=0.12):
    """Cubes of a colour: list of dict(xy top-face centre, top z, n), largest first."""
    P, _, _ = world_points(rgb, depth_m, cam, color)
    P = P[(P[:, 2] > zmin) & (P[:, 2] < zmax)]
    P = subsample(P, 4000)
    res = []
    for idx in cluster_xy(P, radius=0.04, min_pts=6):
        Q = P[idx]
        ztop = np.percentile(Q[:, 2], 90)
        T = Q[Q[:, 2] > ztop - 0.006]
        if len(T) < 4:
            T = Q
        res.append({"xy": T[:, :2].mean(0), "ztop": float(ztop), "n": int(len(idx)), "top": T[:, :2]})
    return res


def min_area_yaw(xy):
    """Yaw (rad, in [-pi/4, pi/4)) of a square from its top-face points via min-area bounding box."""
    if len(xy) < 6:
        return None
    best = None
    c = xy - xy.mean(0)
    for deg in np.arange(-45, 45, 1.0):
        t = np.radians(deg)
        Rm = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        q = c @ Rm.T
        ext = q.max(0) - q.min(0)
        area = ext[0] * ext[1]
        if best is None or area < best[0]:
            best = (area, t, ext)
    return float(best[1]), best[2]
