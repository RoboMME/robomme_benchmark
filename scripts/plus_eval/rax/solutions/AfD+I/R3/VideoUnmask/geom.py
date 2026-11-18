import numpy as np, json
from PIL import Image

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def backproject(depth_m, K, pose):
    """depth_m HxW metres -> HxWx3 world points (nan where invalid)."""
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    d = depth_m
    pc = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1)
    R = quat_to_R(pose['quaternion_wxyz']); t = np.array(pose['position_m'])
    pw = pc @ R.T + t
    pw[d <= 0] = np.nan
    return pw

def load_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))

def load_depth(path, mpu=0.001):
    return np.asarray(Image.open(path)).astype(np.float64) * mpu

def rgb_to_hsv(rgb):
    a = rgb.astype(np.float64) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1); mn = a.min(-1); df = mx - mn
    h = np.zeros_like(mx)
    m = df > 1e-6
    rm = m & (mx == r); gm = m & (mx == g) & ~rm; bm = m & ~rm & ~gm
    h[rm] = ((g - b)[rm] / df[rm]) % 6
    h[gm] = ((b - r)[gm] / df[gm]) + 2
    h[bm] = ((r - g)[bm] / df[bm]) + 4
    h = h * 60.0
    s = np.where(mx > 0, df / np.maximum(mx, 1e-6), 0)
    return h, s, mx
