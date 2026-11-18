"""Pixel/depth back-projection (OpenGL camera convention, world frame)."""
import numpy as np

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def backproject(depth_m, K, pose):
    """depth_m: HxW metres (0 invalid). Returns HxWx3 world points (nan where invalid)."""
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    d = depth_m
    xc = (u - cx) / fx * d
    yc = -(v - cy) / fy * d
    zc = -d
    P = np.stack([xc, yc, zc], -1)
    R = quat_to_R(pose['quaternion_wxyz'])
    t = np.array(pose['position_m'])
    Pw = P @ R.T + t
    Pw[d <= 0] = np.nan
    return Pw

def project(Pw, K, pose):
    """World Nx3 -> pixel (u,v) Nx2."""
    R = quat_to_R(pose['quaternion_wxyz']); t = np.array(pose['position_m'])
    Pc = (np.asarray(Pw) - t) @ R
    d = -Pc[:, 2]
    u = K[0][0] * Pc[:, 0] / d + K[0][2]
    v = -K[1][1] * Pc[:, 1] / d + K[1][2]
    return np.stack([u, v], -1)
