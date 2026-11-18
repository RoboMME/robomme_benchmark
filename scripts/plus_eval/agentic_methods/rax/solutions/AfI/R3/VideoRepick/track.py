"""Track identical objects through the conditioning video (front RGB-D) in world XY."""
import numpy as np
from PIL import Image
from geom import backproject

ZMIN = 0.006   # metres above table for object pixels
ZMAX = 0.07


SAT_MIN = 40   # max-min RGB; ep1 objects (green) ~100, table excluded by height, robot/button gray


def green_mask(rgb):
    """Colour-agnostic 'saturated' mask (name kept for history; v0 was green-only)."""
    rgb = rgb.astype(int)
    return (rgb.max(-1) - rgb.min(-1)) > SAT_MIN


def object_points(rgb, depth_m, K, pose, xy_box=None):
    W = backproject(depth_m, K, pose)
    z = W[..., 2]
    m = green_mask(rgb) & (z > ZMIN) & (z < ZMAX)
    pts = W[m]
    return pts


def cluster_count(pts, radius=0.025, min_pts=8):
    """Greedy connected components in XY on a 5mm grid. Returns list of centroids (xy)."""
    if len(pts) == 0:
        return []
    cell = 0.005
    keys = np.floor(pts[:, :2] / cell).astype(int)
    occ = {}
    for i, k in enumerate(map(tuple, keys)):
        occ.setdefault(k, []).append(i)
    seen, comps = set(), []
    for k in occ:
        if k in seen:
            continue
        stack, comp = [k], []
        seen.add(k)
        while stack:
            c = stack.pop()
            comp.extend(occ[c])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (c[0] + dx, c[1] + dy)
                    if n in occ and n not in seen:
                        seen.add(n)
                        stack.append(n)
        if len(comp) >= min_pts:
            comps.append(pts[comp, :2].mean(0))
    return comps


def kmeans_track(pts_xy, centers, iters=5, gate=0.04):
    """Update centres by constrained k-means from previous centres."""
    C = centers.copy()
    if len(pts_xy) == 0:
        return C, np.zeros(len(C), int)
    for _ in range(iters):
        d = np.linalg.norm(pts_xy[:, None, :] - C[None], axis=2)
        a = d.argmin(1)
        dmin = d.min(1)
        counts = np.zeros(len(C), int)
        for j in range(len(C)):
            sel = (a == j) & (dmin < gate)
            counts[j] = sel.sum()
            if counts[j] >= 3:
                C[j] = pts_xy[sel].mean(0)
    return C, counts


def mean_shift(pts_xy, xy, r=0.022, iters=6):
    """Local centroid of points within r of xy, iterated. Returns (xy, n) or (None, 0)."""
    c = np.asarray(xy, float).copy()
    n = 0
    for _ in range(iters):
        if len(pts_xy) == 0:
            return None, 0
        sel = np.linalg.norm(pts_xy - c, axis=1) < r
        n = int(sel.sum())
        if n < 5:
            return None, n
        c_new = pts_xy[sel].mean(0)
        if np.linalg.norm(c_new - c) < 1e-4:
            c = c_new
            break
        c = c_new
    return c, n
