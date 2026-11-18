"""Live perception of containers from the front RGB-D observation."""
import numpy as np
from track import white_mask, track_step2, cube_blobs

def container_points(obs, cam='front', zmin=0.01):
    rgb = obs.rgb(cam); P = obs.points(cam)
    m = white_mask(rgb, P)
    with np.errstate(invalid='ignore'):
        m &= P[..., 2] > zmin
    return P[m]

def rect_fit(xy):
    """Min-area bounding rectangle over angles in [0, 90). Returns (centre, yaw, (w,h))."""
    best = None
    for a in np.deg2rad(np.arange(0, 90, 2.0)):
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, s], [-s, c]])      # world->box
        q = xy @ R.T
        lo = np.percentile(q, 3, 0); hi = np.percentile(q, 97, 0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            mid = (lo + hi) / 2
            centre = mid @ R                 # box->world
            best = (area, centre, a, hi - lo)
    return best[1], best[2], best[3]

def locate_containers(obs, prior, radius=0.04):
    """Refine prior centres (kx2) with the live front view. Returns list of dicts."""
    pts = container_points(obs)
    xy = pts[:, :2]
    C, counts = track_step2(xy, np.asarray(prior, float), np.zeros_like(prior))
    out = []
    d = np.sqrt(((xy[:, None] - C[None]) ** 2).sum(-1))
    a = d.argmin(1)
    for j in range(len(C)):
        sel = (a == j) & (d[np.arange(len(xy)), a] < radius)
        top = sel & (pts[:, 2] > 0.06)
        info = {'centre': C[j].copy(), 'n': int(sel.sum()), 'n_top': int(top.sum()), 'yaw': None, 'size': None,
                'ztop': float(np.percentile(pts[top, 2], 90)) if top.sum() > 5 else None}
        if top.sum() >= 15:
            c, yaw, wh = rect_fit(xy[top])
            info.update(centre=c, yaw=float(yaw), size=wh.tolist())
        out.append(info)
    return out
