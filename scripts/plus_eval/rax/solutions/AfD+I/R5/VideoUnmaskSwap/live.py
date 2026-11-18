import numpy as np
from geom import depth_to_world
from track import container_mask, XMIN, YMIN

def obs_points(m, cams=('front', 'wrist')):
    out = []
    for cam in cams:
        rgb = m['_imgs'].get((cam, 'rgb')); dep = m['_imgs'].get((cam, 'depth'))
        if rgb is None or dep is None: continue
        rgb = rgb[..., :3]
        d = dep.astype(np.float32) * m['depth'][cam].get('meters_per_unit', 0.001)
        P, ok = depth_to_world(d, m['camera_intrinsics'][cam]['matrix'], m['camera_poses'][cam])
        ok = ok & (P[..., 0] > -0.45) & (P[..., 0] < 0.35) & (np.abs(P[..., 1]) < 0.45)
        msk = container_mask(rgb, P, ok)
        out.append(P[msk])
    return np.concatenate(out, 0) if out else np.zeros((0, 3))

def refine_container(pts, guess, radius=0.03):
    """returns (xy centre, yaw, top_z, n) of container near guess."""
    g = np.asarray(guess, float)
    c = g.copy()
    for _ in range(5):
        d = np.linalg.norm(pts[:, :2] - c, axis=1)
        sel = pts[d < radius]
        if len(sel) < 10: return None
        top = np.percentile(sel[:, 2], 95)
        topsel = sel[sel[:, 2] > top - 0.012]
        if len(topsel) < 8: topsel = sel
        c = topsel[:, :2].mean(0) if len(topsel) >= 8 else sel[:, :2].mean(0)
    xy = topsel[:, :2] - topsel[:, :2].mean(0)
    best = None
    for th in np.deg2rad(np.arange(0, 90, 2)):
        R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        q = xy @ R.T
        a = np.ptp(np.percentile(q[:, 0], [3, 97])) * np.ptp(np.percentile(q[:, 1], [3, 97]))
        if best is None or a < best[0]: best = (a, th)
    yaw = best[1]
    if yaw > np.pi / 4: yaw -= np.pi / 2
    return c, float(yaw), float(top), len(sel)

def top_components(pts, zmin=0.062, cell=0.004, min_pts=15):
    from track import cluster_cc
    top = pts[pts[:, 2] > zmin]
    comps = cluster_cc(top[:, :2], cell=cell, min_pts=min_pts)
    out = []
    for c in comps:
        p = top[c]
        out.append({'xy': p[:, :2].mean(0), 'n': len(c), 'yaw': square_yaw(p[:, :2]), 'ext': np.ptp(p[:, :2], 0)})
    return out

def square_yaw(xy):
    xy = xy - xy.mean(0); best = None
    for th in np.deg2rad(np.arange(0, 90, 2)):
        R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        q = xy @ R.T
        a = np.ptp(np.percentile(q[:, 0], [2, 98])) * np.ptp(np.percentile(q[:, 1], [2, 98]))
        if best is None or a < best[0]: best = (a, th)
    yaw = best[1]
    return float(yaw - np.pi / 2 if yaw > np.pi / 4 else yaw)
