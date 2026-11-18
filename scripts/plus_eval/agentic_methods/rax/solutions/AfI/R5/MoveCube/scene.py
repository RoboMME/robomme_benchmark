"""Scene perception from one observation (both cameras fused)."""
import math
import numpy as np
from perc import world_points

def masks(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return {
        "red": (r > 150) & (g < 60) & (b < 60),
        "purple": (b > 120) & (r > 110) & (g < r - 30) & (g < b - 30),
        "teal": (r < 100) & (g > 95) & (b > 95) & (abs(g - b) < 18) & (b < 165) & (g > r + 30),
        "shaft": (r > 165) & (r < 215) & (b > r + 6) & (b < r + 30) & (g > r) & (g < b),
        "yellow": (r > 150) & (g > 130) & (b < 90) & (abs(r - g) < 80),
        "blue": (b > 150) & (b > r + 40) & (g > r + 10) & (b > g),
        "pink": (r > 200) & (g > 135) & (g < 185) & (b > 110) & (b < 160) & (r - g > 50),
    }

def cloud(meta, imgs, cams=("front", "wrist")):
    out = {}
    for cam in cams:
        if (cam, "depth") not in imgs: continue
        P = world_points(imgs[(cam, "depth")], meta["camera_intrinsics"][cam], meta["camera_poses"][cam])
        out[cam] = (imgs[(cam, "rgb")], P)
    return out

def pts_of(cl, name, cams=("front", "wrist"), zmin=-0.01, zmax=0.6):
    acc = []
    for cam in cams:
        if cam not in cl: continue
        rgb, P = cl[cam]; m = masks(rgb)[name]
        q = P[m]; q = q[~np.isnan(q).any(1)]
        q = q[(q[:, 2] > zmin) & (q[:, 2] < zmax)]
        acc.append(q)
    return np.concatenate(acc) if acc else np.zeros((0, 3))

def cluster_main(q, rad=0.06):
    """keep points near the densest point (simple)"""
    if len(q) < 3: return q
    med = np.median(q, 0)
    for _ in range(3):
        d = np.linalg.norm(q[:, :2] - med[:2], axis=1)
        sel = q[d < rad]
        if len(sel) < 3: break
        med = np.median(sel, 0)
    d = np.linalg.norm(q[:, :2] - med[:2], axis=1)
    return q[d < rad]
