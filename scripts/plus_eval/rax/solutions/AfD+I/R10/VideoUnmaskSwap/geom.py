import numpy as np
from PIL import Image

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def backproject(depth_m, K, pose):
    """depth_m HxW meters -> HxWx3 world points (nan where invalid)."""
    H, W = depth_m.shape
    K = K["matrix"] if isinstance(K, dict) else K
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    d = depth_m
    cam = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1)
    R = quat_to_R(pose["quaternion_wxyz"])
    pts = cam @ R.T + np.array(pose["position_m"])
    pts[d <= 0] = np.nan
    return pts

def project(pts, K, pose):
    """world Nx3 -> pixel u,v (N,2) and depth."""
    K = K["matrix"] if isinstance(K, dict) else K
    R = quat_to_R(pose["quaternion_wxyz"])
    cam = (np.asarray(pts) - np.array(pose["position_m"])) @ R
    d = -cam[:, 2]
    u = cam[:, 0] / d * K[0][0] + K[0][2]
    v = -cam[:, 1] / d * K[1][1] + K[1][2]
    return np.stack([u, v], -1), d

def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))

def load_depth(path, mpu=0.001):
    return np.asarray(Image.open(path)).astype(np.float64) * mpu
