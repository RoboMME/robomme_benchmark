import numpy as np
from geom import backproject

def hsv(rgb):
    a = rgb.astype(float) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1); mn = a.min(-1); c = mx - mn
    h = np.zeros_like(mx)
    m = c > 1e-6
    rr = m & (mx == r); gg = m & (mx == g) & ~rr; bb = m & ~rr & ~gg
    h[rr] = ((g - b)[rr] / c[rr]) % 6
    h[gg] = (b - r)[gg] / c[gg] + 2
    h[bb] = (r - g)[bb] / c[bb] + 4
    h = h * 60
    s = np.where(mx > 0, c / np.maximum(mx, 1e-6), 0)
    return h, s, mx

def color_masks(rgb):
    h, s, v = hsv(rgb)
    sat = (s > 0.5) & (v > 0.3)
    return {
        'black': (s < 0.2) & (v < 0.3),
        'red': sat & ((h < 8) | (h > 340)),
        'orange': sat & (h > 20) & (h < 45),
        'yellow': sat & (h >= 50) & (h < 70),
        'green': sat & (h > 100) & (h < 140),
        'blue': sat & (h > 220) & (h < 250),
        'purple': sat & (h > 260) & (h < 300),
    }

def components(mask, min_px=4):
    from collections import deque
    H, W = mask.shape
    seen = np.zeros_like(mask, bool)
    out = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        q = deque([(y0, x0)]); seen[y0, x0] = True; pts = []
        while q:
            y, x = q.popleft(); pts.append((y, x))
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                yy, xx = y+dy, x+dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True; q.append((yy, xx))
        if len(pts) >= min_px:
            out.append(np.array(pts))
    return out

def locate(pts, depth, K, pose):
    """world point from pixel set (rows, cols): median of back-projected points."""
    P = np.array([backproject(x, y, depth[y, x], K, pose) for y, x in pts if depth[y, x] > 0])
    if len(P):
        P = P[(P[:, 2] > 0.008) & (P[:, 2] < 0.12)]
    return np.median(P, axis=0) if len(P) else None

def find_colored(rgb, depth, K, pose, min_px=4):
    res = {}
    for name, m in color_masks(rgb).items():
        comps = components(m, min_px)
        res[name] = [(locate(c, depth, K, pose), len(c), c.mean(0)[::-1]) for c in comps]
    return res

def white_mask(rgb):
    h, s, v = hsv(rgb)
    return (s < 0.12) & (v > 0.55)

def container_points(rgb, depth, K, pose, zmin=0.012, zmax=0.12):
    """World XY points of white, above-table pixels in the region of the table."""
    wm = white_mask(rgb)
    ys, xs = np.nonzero(wm & (depth > 0))
    from geom import quat_to_R
    R = quat_to_R(pose['quaternion_wxyz']); t = np.array(pose['position_m'])
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    d = depth[ys, xs]
    pc = np.stack([(xs - cx) / fx * d, -(ys - cy) / fy * d, -d], 1)
    P = pc @ R.T + t
    keep = (P[:, 2] > zmin) & (P[:, 2] < zmax)
    return P[keep], np.stack([xs, ys], 1)[keep]

def cluster_xy(P, cell=0.006, min_pts=20):
    """Grid-based connected components on XY occupancy."""
    if len(P) == 0:
        return []
    ij = np.floor(P[:, :2] / cell).astype(int)
    keys = {}
    for n, k in enumerate(map(tuple, ij)):
        keys.setdefault(k, []).append(n)
    seen = set(); out = []
    for k in keys:
        if k in seen:
            continue
        stack = [k]; seen.add(k); idx = []
        while stack:
            c = stack.pop(); idx += keys[c]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (c[0]+di, c[1]+dj)
                    if nb in keys and nb not in seen:
                        seen.add(nb); stack.append(nb)
        if len(idx) >= min_pts:
            Q = P[idx]
            out.append({'xy': Q[:, :2].mean(0), 'zmax': float(np.percentile(Q[:, 2], 95)), 'n': len(idx),
                        'ext': Q[:, :2].max(0) - Q[:, :2].min(0)})
    return out
