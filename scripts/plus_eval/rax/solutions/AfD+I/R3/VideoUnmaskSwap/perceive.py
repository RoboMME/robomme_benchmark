import numpy as np
from wtrack import frame_points, CELL, LIM

def container_points(meta, imgs, zr=(0.05, 0.11)):
    pts = frame_points(meta, imgs)
    allp = np.concatenate([P for P, _ in pts], 0)
    s = (allp[:, 2] > zr[0]) & (allp[:, 2] < zr[1]) & (np.abs(allp[:, 0]) < LIM) & (np.abs(allp[:, 1]) < LIM)
    return allp[s]

def refine(meta, imgs, xy, rad=0.035):
    """Refine container centre and yaw near xy from current observation."""
    P = container_points(meta, imgs)
    xy = np.asarray(xy, float)
    for it in range(3):
        s = np.linalg.norm(P[:, :2] - xy, axis=1) < rad
        if s.sum() < 20: return None
        Q = P[s]
        top = Q[Q[:, 2] > Q[:, 2].max() - 0.012]
        if len(top) >= 20: Q2 = top
        else: Q2 = Q
        # dedupe in grid
        g = np.unique(np.round(Q2[:, :2] / 0.002).astype(int), axis=0) * 0.002
        xy = 0.5 * (g.max(0) + g.min(0)) if False else g.mean(0)
    # yaw via min-area bounding box over 0..90 deg
    best = None
    for a in np.deg2rad(np.arange(0, 90, 2)):
        c, s_ = np.cos(a), np.sin(a)
        R = np.array([[c, s_], [-s_, c]])
        q = (g - xy) @ R.T
        area = np.ptp(q[:, 0]) * np.ptp(q[:, 1])
        if best is None or area < best[0]: best = (area, a, np.ptp(q[:, 0]), np.ptp(q[:, 1]), 0.5 * (q.max(0) + q.min(0)) @ R + xy)
    area, a, w, h, ctr = best
    return dict(xy=ctr, xy_mean=xy, yaw=float(a), size=(float(w), float(h)), n=int(len(g)), ztop=float(Q[:, 2].max()))
