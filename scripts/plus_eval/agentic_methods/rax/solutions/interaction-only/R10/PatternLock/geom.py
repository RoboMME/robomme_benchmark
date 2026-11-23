import numpy as np

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def backproject(depth_m, K, pose):
    """depth_m HxW metres -> HxWx3 world points (nan where invalid)."""
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    v, u = np.mgrid[0:H, 0:W].astype(float)
    d = depth_m
    pc = np.stack([(u-cx)*d/fx, -(v-cy)*d/fy, -d], -1)
    R = quat_to_R(pose["quaternion_wxyz"])
    pw = pc @ R.T + np.array(pose["position_m"])
    pw[d <= 0] = np.nan
    return pw

def find_buttons(rgb, pts, min_px=3):
    """Find white button centres: bright low-saturation pixels raised above the table.
    Returns list of world (x,y,z) cluster centres."""
    rgb = rgb.astype(float)
    bright = rgb.min(-1)
    sat = rgb.max(-1) - rgb.min(-1)
    mask = (bright > 150) & (sat < 60) & np.isfinite(pts[..., 0])
    ys, xs = np.nonzero(mask)
    P = pts[ys, xs]
    # simple connected components on mask
    from collections import deque
    H, W = mask.shape
    lab = -np.ones((H, W), int)
    comps = []
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0] >= 0:
            continue
        q = deque([(y0, x0)]); lab[y0, x0] = len(comps); cur = []
        while q:
            y, x = q.popleft(); cur.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and lab[yy, xx] < 0:
                        lab[yy, xx] = len(comps); q.append((yy, xx))
        comps.append(cur)
    out = []
    for c in comps:
        if len(c) < min_px:
            continue
        c = np.array(c)
        p = pts[c[:, 0], c[:, 1]]
        out.append((np.nanmedian(p, 0), len(c), c.mean(0)))
    return out
