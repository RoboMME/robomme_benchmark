import numpy as np

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])

def backproject(depth_m, K, pose, stride=1):
    """depth_m HxW metres (0 invalid). Returns (N,3) world points and (N,2) pixel uv."""
    H, W = depth_m.shape
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    d = depth_m[::stride, ::stride]
    ok = d > 0
    u = us[ok].astype(float); v = vs[ok].astype(float); d = d[ok]
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    cam = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], axis=1)
    R = quat_to_R(pose['quaternion_wxyz'])
    world = cam @ R.T + np.asarray(pose['position_m'])
    return world, np.stack([u, v], axis=1)

def project(points, K, pose):
    """world (N,3) -> pixel uv (N,2), depth (N,)"""
    R = quat_to_R(pose['quaternion_wxyz'])
    cam = (np.asarray(points) - np.asarray(pose['position_m'])) @ R
    d = -cam[:, 2]
    u = K[0][0] * cam[:, 0] / d + K[0][2]
    v = -K[1][1] * cam[:, 1] / d + K[1][2]
    return np.stack([u, v], axis=1), d
