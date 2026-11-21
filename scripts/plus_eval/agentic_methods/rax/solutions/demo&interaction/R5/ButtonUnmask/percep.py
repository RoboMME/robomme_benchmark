import numpy as np
from robo import backproject, load_frame

COLORS = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']

def color_masks(rgb):
    a = rgb.astype(np.int32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
    sat = mx - mn
    m = {}
    low = mn < 45
    m['red'] = low & (r > 100) & (g < 50) & (b < 50)
    m['green'] = low & (g > 90) & (r < 70) & (b < 70)
    m['blue'] = low & (b > 80) & (r < 60) & (g < 60)
    m['yellow'] = low & (r > 130) & (g > 130) & (b < 45) & (r - g < 40)
    m['orange'] = low & (r > 150) & (g > 70) & (r - g >= 40) & (b < 40) & (g - b > 50)
    m['purple'] = (b > 120) & (r > 90) & (g < 110) & (b - g > 60) & (r - g > 30)
    return m

def blobs(mask, min_px=4):
    """simple connected components (4-neigh) -> list of (n, u, v)"""
    H, W = mask.shape
    lab = np.zeros(mask.shape, np.int32)
    out = []
    cur = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]:
            continue
        cur += 1
        stack = [(y0, x0)]; lab[y0, x0] = cur; pts = []
        while stack:
            y, x = stack.pop(); pts.append((y, x))
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                yy, xx = y+dy, x+dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = cur; stack.append((yy, xx))
        if len(pts) >= min_px:
            p = np.array(pts)
            out.append((len(pts), p[:, 1], p[:, 0]))
    return out

def detect_cubes(fr, cam='front', min_px=4):
    """returns list of dict(color, n, uv, xyz) from a frame record"""
    d = load_frame(fr)
    rgb = d[(cam, 'rgb')]; dep = d[(cam, 'depth')]
    K = fr['camera_intrinsics'][cam]['matrix']; pose = fr['camera_poses'][cam]
    res = []
    for c, m in color_masks(rgb).items():
        for n, us, vs in blobs(m, min_px):
            P, dd = backproject(dep, K, pose, us, vs)
            ok = dd > 0
            if ok.sum() == 0:
                continue
            xyz = np.median(P[ok], 0)
            res.append(dict(color=c, n=n, uv=(float(us.mean()), float(vs.mean())), xyz=xyz.tolist()))
    return res

def white_mask(rgb):
    a = rgb.astype(np.int32)
    mx = a.max(2); mn = a.min(2)
    return (mn > 150) & (mx - mn < 30)

def containers_from_frame(fr, cam, zmin=0.015):
    """segment white objects; return list of dict(xyz top center, n, yaw, size)"""
    d = load_frame(fr)
    rgb = d[(cam, 'rgb')]; dep = d[(cam, 'depth')]
    K = fr['camera_intrinsics'][cam]['matrix']; pose = fr['camera_poses'][cam]
    m = white_mask(rgb)
    H, W = m.shape
    vs, us = np.nonzero(m)
    P, dd = backproject(dep, K, pose, us, vs)
    return P, dd, us, vs

def cluster_xy(P, radius=0.03):
    """greedy clustering of points in xy"""
    pts = P.copy()
    clusters = []
    remaining = np.ones(len(pts), bool)
    while remaining.any():
        i = np.nonzero(remaining)[0][0]
        c = pts[i, :2]
        for _ in range(5):
            sel = remaining & (np.linalg.norm(pts[:, :2] - c, axis=1) < radius)
            c = pts[sel, :2].mean(0)
        sel = remaining & (np.linalg.norm(pts[:, :2] - c, axis=1) < radius)
        if sel.sum() == 0:
            remaining[i] = False; continue
        clusters.append((c, sel.sum(), pts[sel]))
        remaining &= ~sel
    return clusters

def box_yaw(pts_xy):
    """yaw in [-pi/4, pi/4) of square footprint via min-area bounding box"""
    c = pts_xy.mean(0)
    q = pts_xy - c
    best = None
    for deg in np.arange(-45, 45, 1.0):
        t = np.deg2rad(deg)
        R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        r = q @ R.T
        ext = np.percentile(r, 97, 0) - np.percentile(r, 3, 0)
        a = ext[0] * ext[1]
        if best is None or a < best[0]:
            best = (a, t, ext)
    return best[1], best[2]

def find_container(fr, cam, target_xy, top_z=None, radius=0.04):
    P, dd, us, vs = containers_from_frame(fr, cam)
    ok = (dd > 0) & (P[:, 2] > 0.02) & (P[:, 2] < 0.15)
    P = P[ok]
    near = P[np.linalg.norm(P[:, :2] - np.asarray(target_xy)[None], axis=1) < radius]
    if len(near) < 20:
        return None
    zt = np.percentile(near[:, 2], 95) if top_z is None else top_z
    top = near[np.abs(near[:, 2] - zt) < 0.006]
    # recentre on top surface and recollect
    c = top[:, :2].mean(0)
    sel = P[(np.linalg.norm(P[:, :2] - c[None], axis=1) < 0.04) & (np.abs(P[:, 2] - zt) < 0.006)]
    c = sel[:, :2].mean(0)
    yaw, ext = box_yaw(sel[:, :2])
    return dict(xy=c, top_z=float(zt), yaw=float(yaw), ext=ext, n=len(sel))
