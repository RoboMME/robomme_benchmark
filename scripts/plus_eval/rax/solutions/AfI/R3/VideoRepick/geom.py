"""Camera geometry: back-project depth pixels to world (OpenGL camera convention)."""
import numpy as np


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(depth_m, K, pose):
    """depth_m HxW metres (0 invalid). Returns HxWx3 world points (nan where invalid)."""
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    d = depth_m.astype(np.float64)
    xc = (u - cx) / fx * d
    yc = -(v - cy) / fy * d
    zc = -d
    P = np.stack([xc, yc, zc], -1)
    R = quat_to_R(pose["quaternion_wxyz"])
    t = np.array(pose["position_m"])
    Wp = P @ R.T + t
    Wp[d <= 0] = np.nan
    return Wp
