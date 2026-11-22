import numpy as np
from PIL import Image

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def deproject(depth_m, intr, pose):
    K = np.array(intr["matrix"]); fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    H, W = depth_m.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = depth_m
    pc = np.stack([(u-cx)*d/fx, -(v-cy)*d/fy, -d], -1)
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    return pc @ R.T + t

def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).astype(np.int16)

def load_depth(path, mpu=0.001):
    with Image.open(path) as im:
        return np.asarray(im).astype(np.float64) * mpu

def green_mask(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (g > r + 25) & (g > b + 25)

def green_points(rgb, depth, intr, pose, zmax=0.12):
    """world xyz of green pixels (valid depth, near table)."""
    P = deproject(depth, intr, pose)
    m = green_mask(rgb) & (depth > 0)
    pts = P[m]
    uv = np.argwhere(m)[:, ::-1]
    keep = pts[:, 2] < zmax
    return pts[keep], uv[keep]

def components(mask, min_px=5):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n+1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) >= min_px:
            out.append((xs, ys))
    return out

def project(xyz, intr, pose):
    K = np.array(intr["matrix"]); R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    pc = (np.asarray(xyz) - t) @ R
    d = -pc[..., 2]
    u = K[0, 0] * pc[..., 0] / d + K[0, 2]; v = -K[1, 1] * pc[..., 1] / d + K[1, 2]
    return np.stack([u, v], -1)
