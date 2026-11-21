import numpy as np

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])

def backproject(depth_m, K, pose):
    """depth_m HxW metres -> HxWx3 world points (nan where invalid)."""
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    d = depth_m
    cam = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1)
    R = quat_to_R(pose["quaternion_wxyz"])
    P = cam @ R.T + np.array(pose["position_m"])
    P[d <= 0] = np.nan
    return P

def pixel_to_plane(u, v, K, pose, z0):
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    d = np.stack([(np.asarray(u, float) - cx) / fx, -(np.asarray(v, float) - cy) / fy, -np.ones_like(np.asarray(u, float))], -1)
    R = quat_to_R(pose["quaternion_wxyz"])
    dw = d @ R.T
    p = np.array(pose["position_m"])
    t = (z0 - p[2]) / dw[..., 2]
    return p + t[..., None] * dw
