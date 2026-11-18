import numpy as np
from geom import depth_to_world

def _clusters(pts, radius=0.02):
    """greedy clustering of Nx2 points; returns list of (centroid, count)."""
    pts = np.asarray(pts)
    out = []
    used = np.zeros(len(pts), bool)
    for i in range(len(pts)):
        if used[i]:
            continue
        c = pts[i]
        for _ in range(5):
            m = (~used) & (np.linalg.norm(pts - c, axis=1) < radius)
            if m.sum() == 0:
                break
            c = pts[m].mean(0)
        m = (~used) & (np.linalg.norm(pts - c, axis=1) < radius)
        used |= m
        out.append((c, int(m.sum())))
    return out

def detect_scene(rgb, depth_m, K, pose, xy_box=(-0.45, 0.25, -0.40, 0.40)):
    """Return targets (Nx2 world xy, sorted by y) and obstacles (Mx3: x,y,height)."""
    P = depth_to_world(depth_m, K, pose)
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    inbox = (x > xy_box[0]) & (x < xy_box[1]) & (y > xy_box[2]) & (y < xy_box[3])
    mx = rgb.max(-1); mn = rgb.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1)
    white = inbox & (z > 0.007) & (z < 0.03) & (mx > 140) & (sat < 0.12)
    tg = [c for c, n in _clusters(np.c_[x[white], y[white]], 0.025) if n >= 4]
    tg = np.array(sorted(tg, key=lambda c: c[1])) if tg else np.zeros((0, 2))
    tall = inbox & (z > 0.045) & (z < 0.30)
    obs = []
    if tall.any():
        for c, n in _clusters(np.c_[x[tall], y[tall]], 0.04):
            if n >= 15:
                m = tall & (np.hypot(x - c[0], y - c[1]) < 0.04)
                obs.append([c[0], c[1], float(np.nanmax(z[m]))])
    obs = np.array(sorted(obs, key=lambda o: o[1])) if obs else np.zeros((0, 3))
    return tg, obs
