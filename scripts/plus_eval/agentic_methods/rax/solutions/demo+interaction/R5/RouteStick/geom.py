import numpy as np
from PIL import Image

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def depth_to_world(depth_m, K, pose):
    """depth_m HxW metres -> HxWx3 world points (nan where invalid)."""
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = depth_m
    cam = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1)
    R = quat_to_R(pose['quaternion_wxyz'])
    P = cam @ R.T + np.array(pose['position_m'])
    P[d <= 0] = np.nan
    return P

def load_depth(path, mpu=0.001):
    return np.asarray(Image.open(path)).astype(np.float64) * mpu

def load_rgb(path):
    return np.asarray(Image.open(path).convert('RGB')).astype(np.float64)

def world_to_pixel(Pw, K, pose):
    R = quat_to_R(pose['quaternion_wxyz'])
    c = (np.asarray(Pw) - np.array(pose['position_m'])) @ R  # world->cam (R^T p)
    d = -c[2]
    if d <= 1e-6:
        return None
    u = K[0][0] * c[0] / d + K[0][2]
    v = -K[1][1] * c[1] / d + K[1][2]
    return u, v

def red_count(rgb, uv, r=6):
    if uv is None:
        return 0
    u, v = int(round(uv[0])), int(round(uv[1]))
    H, W = rgb.shape[:2]
    if not (0 <= u < W and 0 <= v < H):
        return 0
    patch = rgb[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
    m = (patch[..., 0] > 150) & (patch[..., 1] < 90) & (patch[..., 2] < 90)
    return int(m.sum())

def rotvec_to_R(r):
    r = np.asarray(r, float); th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

def R_to_rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if np.pi - th < 1e-4:
        # near pi: axis from the largest diagonal
        i = int(np.argmax(np.diag(R))); v = R[:, i] + np.eye(3)[i]
        return v / np.linalg.norm(v) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th

def rot_error(ref_rv, cur_rv):
    """world-frame rotation vector taking current orientation to reference."""
    return R_to_rotvec(rotvec_to_R(ref_rv) @ rotvec_to_R(cur_rv).T)
