import numpy as np, cv2, math

HUES = {  # OpenCV hue ranges (0-180)
    "red": [(0, 6), (170, 180)],
    "orange": [(17, 25)],
    "yellow": [(26, 38)],
    "green": [(42, 85)],
    "cyan": [(86, 99)],
    "blue": [(100, 128)],
    "purple": [(129, 160)],
    "pink": [(161, 169)],
}

def color_blobs(rgb, min_area=6, sat=215, val=50):
    """rgb: HxWx3 uint8 RGB. Returns {color: [(area, cx, cy), ...]} sorted by area desc."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    out = {}
    for name, ranges in HUES.items():
        base = (s >= (120 if name in ("purple", "pink", "cyan") else sat)) & (v >= val)
        m = np.zeros_like(base)
        for lo, hi in ranges:
            m |= (h >= lo) & (h <= hi)
        m &= base
        m = m.astype(np.uint8)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
        bl = []
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            if a >= min_area:
                ys, xs = np.nonzero(lab == i)
                bl.append((int(a), float(xs.mean()), float(ys.mean()), (xs, ys)))
        bl.sort(key=lambda b: -b[0])
        if bl:
            out[name] = bl
    return out

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def backproject(u, v, d, K, pose):
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    pc = np.array([(u - cx) * d / fx, -(v - cy) * d / fy, -d])
    R = quat_to_R(pose["quaternion_wxyz"])
    return R @ pc + np.array(pose["position_m"])

def blob_world(xs, ys, depth_m, K, pose):
    """median world point of blob pixels with valid depth"""
    pts = []
    for u, v in zip(xs, ys):
        d = depth_m[v, u]
        if d > 0:
            pts.append(backproject(u, v, d, K, pose))
    if not pts:
        return None
    return np.median(np.array(pts), axis=0)
