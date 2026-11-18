import numpy as np
from PIL import Image

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def backproject(depth_m, K, pose, us, vs):
    """depth_m HxW metres; us,vs pixel arrays -> Nx3 world points (nan when invalid)."""
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    d = depth_m[vs, us]
    pc = np.stack([(us - cx) * d / fx, -(vs - cy) * d / fy, -d], axis=1)
    R = quat_to_R(pose["quaternion_wxyz"])
    pw = pc @ R.T + np.asarray(pose["position_m"])
    pw[d <= 0] = np.nan
    return pw

def world_map(depth_m, K, pose):
    H, W = depth_m.shape
    vs, us = np.mgrid[0:H, 0:W]
    return backproject(depth_m, K, pose, us.ravel(), vs.ravel()).reshape(H, W, 3)

def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32)

def load_depth(path, scale=0.001):
    return np.asarray(Image.open(path)).astype(np.float32) * scale

def hsv(rgb):
    r, g, b = rgb[..., 0] / 255., rgb[..., 1] / 255., rgb[..., 2] / 255.
    mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn + 1e-9
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    s = np.where(mx > 0, (mx - mn) / (mx + 1e-9), 0)
    return h, s, mx
