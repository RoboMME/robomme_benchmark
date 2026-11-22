import numpy as np
from geom import world_points


def stick_geometry(dep, K, pose, tcp):
    """Measure the stick in the wrist depth: returns dict(tip_dz, axis_xy (rel TCP), n)."""
    P = world_points(dep, K, pose)
    z = P[..., 2]
    sel = (dep > 0) & (dep < 0.16) & (z > 0.03)
    pts = P[sel]
    if len(pts) < 50:
        return None
    rel = pts - np.asarray(tcp)[None]
    order = np.argsort(rel[:, 2])
    low = rel[order[: max(10, len(order) // 50)]]
    # axis: mean xy of points in the lower half of the stick
    zr = rel[:, 2]
    lowhalf = rel[zr < np.percentile(zr, 40)]
    return dict(tip_dz=float(np.percentile(zr, 1)), tip_xy=low[:, :2].mean(0), axis_xy=lowhalf[:, :2].mean(0),
                n=int(len(pts)), dmax=float(dep[sel].max()))


def tip_from_front(dep, bg_dep, K, pose, tcp, zmin=0.03):
    """Lowest robot point in the front view: pixels closer than the background by >1.5 cm."""
    P = world_points(dep, K, pose)
    robot = (dep > 0) & (bg_dep > 0) & (dep < bg_dep - 0.015) & (P[..., 2] > zmin)
    if robot.sum() < 20:
        return None
    pts = P[robot]
    pts = pts[np.linalg.norm(pts[:, :2] - np.asarray(tcp)[:2], axis=1) < 0.2]
    if len(pts) < 20:
        return None
    o = np.argsort(pts[:, 2])
    low = pts[o[:6]]
    tip = low.mean(0)
    return dict(tip_rel=tip - np.asarray(tcp), n=int(len(pts)))


def wrist_stick(dep, K, pose, tcp, zmin=0.03, rmax=0.09):
    """Stick seen in the wrist depth: lowest point and principal axis (pointing down).
    Returns None when the stick is not fully in view (lowest point on the image border)."""
    P = world_points(dep, K, pose)
    tcp = np.asarray(tcp, float)
    m = (dep > 0) & (P[..., 2] > zmin) & (np.linalg.norm(P[..., :2] - tcp[:2], axis=-1) < rmax)
    n = int(m.sum())
    if n < 150:
        return None
    vs, us = np.nonzero(m)
    Q = P[m]
    o = np.argsort(Q[:, 2])[:6]
    lo = Q[o].mean(0)
    uu, vv = us[o].mean(), vs[o].mean()
    H, W = dep.shape
    if min(uu, vv) < 6 or uu > W - 7 or vv > H - 7:
        return None
    c0 = Q.mean(0)
    _, S, Vt = np.linalg.svd(Q - c0, full_matrices=False)
    a = Vt[0]
    if a[2] > 0:
        a = -a
    if a[2] > -0.5:  # implausibly flat axis
        return None
    return dict(tip_rel=lo - tcp, axis=a, n=n)


def hit_rel(tip_rel, axis, z_tcp, z_plane):
    """xy (relative to TCP) where the stick axis meets the plane z_plane for a TCP at height z_tcp."""
    tip_rel = np.asarray(tip_rel, float)
    a = np.asarray(axis, float)
    zt = z_tcp + tip_rel[2]
    t = (z_plane - zt) / a[2]
    return tip_rel[:2] + t * a[:2]
