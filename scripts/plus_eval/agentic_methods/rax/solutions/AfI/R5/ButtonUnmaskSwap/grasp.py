import math
import numpy as np
import robo, tracker

def yaw_of(pts):
    """Square yaw (rad, in [-pi/4, pi/4)) via min-area bounding box."""
    best = (1e9, 0.0)
    c = pts.mean(0)
    q = pts - c
    for a in np.radians(np.arange(-45, 45, 2.0)):
        ca, sa = math.cos(a), math.sin(a)
        u = q[:, 0] * ca + q[:, 1] * sa
        v = -q[:, 0] * sa + q[:, 1] * ca
        area = (np.percentile(u, 97) - np.percentile(u, 3)) * (np.percentile(v, 97) - np.percentile(v, 3))
        if area < best[0]:
            best = (area, a)
    return best[1]

def refine_container(obs, xy, cam="front", r=0.035, ee=None):
    pw, ok = obs.world_map(cam)
    pts, _ = tracker.cap_points(pw, ok, ee=ee)
    if len(pts) == 0:
        return None
    c = np.asarray(xy, float)
    for _ in range(4):
        sel = np.linalg.norm(pts - c, axis=1) < r
        if sel.sum() < 5:
            return None
        c = pts[sel].mean(0)
    sel = np.linalg.norm(pts - c, axis=1) < r
    return c, yaw_of(pts[sel]), int(sel.sum())

def Rz(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
