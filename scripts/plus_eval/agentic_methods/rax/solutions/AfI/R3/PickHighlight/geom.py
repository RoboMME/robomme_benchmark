"""Geometry helpers: back-projection of RGB-D pixels into the world frame (OpenGL camera convention)."""
import numpy as np


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(depth_m, K, pose, us, vs):
    """depth_m: HxW metres. us, vs: arrays of pixel coords. Returns Nx3 world points."""
    us = np.asarray(us, float); vs = np.asarray(vs, float)
    d = depth_m[np.clip(vs.round().astype(int), 0, depth_m.shape[0] - 1),
                np.clip(us.round().astype(int), 0, depth_m.shape[1] - 1)]
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    xc = (us - cx) / fx * d
    yc = -(vs - cy) / fy * d
    zc = -d
    P = np.stack([xc, yc, zc], 1)
    R = quat_to_R(pose["quaternion_wxyz"])
    return P @ R.T + np.asarray(pose["position_m"])


def world_map(depth_m, K, pose):
    """Full HxWx3 world-point map."""
    H, W = depth_m.shape
    vs, us = np.mgrid[0:H, 0:W]
    pts = backproject(depth_m, K, pose, us.ravel(), vs.ravel())
    return pts.reshape(H, W, 3)
