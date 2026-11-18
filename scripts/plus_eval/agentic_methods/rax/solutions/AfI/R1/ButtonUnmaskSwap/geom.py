"""Rotations and camera back-projection (parsed calibration + empirical linear fit)."""
import math

import numpy as np


def rotvec_to_mat(r):
    r = np.asarray(r, dtype=float)
    th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def mat_to_rotvec(R):
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1) / 2
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: use the symmetric part
        M = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        k = k / np.linalg.norm(k)
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def quat_wxyz_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------- calibration parsing
def parse_intrinsics(v):
    """-> (fx, fy, cx, cy) or None"""
    try:
        if isinstance(v, dict):
            for k in ("matrix", "K", "intrinsic", "intrinsics", "data"):
                if k in v:
                    r = parse_intrinsics(v[k])
                    if r:
                        return r
            if "fx" in v:
                return float(v["fx"]), float(v.get("fy", v["fx"])), float(v["cx"]), float(v["cy"])
            return None
        a = np.asarray(v, dtype=float)
        if a.shape == (3, 3):
            return a[0, 0], a[1, 1], a[0, 2], a[1, 2]
        if a.size == 9:
            a = a.reshape(3, 3)
            return a[0, 0], a[1, 1], a[0, 2], a[1, 2]
        if a.size == 4:
            return tuple(float(x) for x in a)
    except Exception:
        return None
    return None


def parse_pose(v):
    """-> (R (3x3), t (3,)) camera-to-world, or None"""
    try:
        if isinstance(v, dict):
            pos = None
            for k in ('position_m', "position", "p", "translation", "xyz"):
                if k in v:
                    pos = np.asarray(v[k], dtype=float).reshape(3)
            q = None
            for k in ("quaternion_wxyz", "wxyz", "q", "quaternion", "orientation"):
                if k in v:
                    q = np.asarray(v[k], dtype=float).reshape(4)
            if pos is not None and q is not None:
                return quat_wxyz_to_mat(q), pos
            for k in ("matrix", "pose", "transform", "cam2world", "camera_to_world", "data"):
                if k in v:
                    r = parse_pose(v[k])
                    if r:
                        return r
            return None
        a = np.asarray(v, dtype=float)
        if a.size == 16:
            a = a.reshape(4, 4)
            return a[:3, :3], a[:3, 3]
        if a.size == 12:
            a = a.reshape(3, 4)
            return a[:, :3], a[:, 3]
        if a.size == 7:
            return quat_wxyz_to_mat(a[3:]), a[:3]
    except Exception:
        return None
    return None


def get_cam_entry(container, cam):
    if not isinstance(container, dict):
        return None
    for k, v in container.items():
        if str(k).lower() == cam or str(k).lower().startswith(cam):
            return v
    return None


class LinearCam:
    """world = M @ [u*d, v*d, d, 1]  (exact for a pinhole camera with linear depth)."""

    def __init__(self, M, swap_uv=False, source=""):
        self.M = np.asarray(M, dtype=float)
        self.swap_uv = swap_uv
        self.source = source

    def backproject_image(self, depth):
        H, W = depth.shape
        v, u = np.mgrid[0:H, 0:W].astype(np.float32)
        d = depth.astype(np.float32)
        F = np.stack([u * d, v * d, d, np.ones_like(d)], axis=-1)
        P = F @ self.M.T.astype(np.float32)
        return P  # H x W x 3

    def backproject(self, u, v, d):
        return self.M @ np.array([u * d, v * d, d, 1.0])


def pinhole_linear(intr, pose, opengl=True, offset=0.0):
    fx, fy, cx, cy = intr
    R, t = pose
    # p_cam = d * [(u+off-cx)/fx, -(v+off-cy)/fy, -1]  (OpenGL)  ; world = R p + t
    if opengl:
        A = np.array([[1 / fx, 0, (offset - cx) / fx],
                      [0, -1 / fy, -(offset - cy) / fy],
                      [0, 0, -1.0]])
    else:
        A = np.array([[1 / fx, 0, (offset - cx) / fx],
                      [0, 1 / fy, (offset - cy) / fy],
                      [0, 0, 1.0]])
    M = np.zeros((3, 4))
    M[:, :3] = R @ A
    M[:, 3] = t
    return M


def fit_linear(samples):
    """samples: list of (u, v, d, xyz). Returns (M, rms)."""
    F = np.array([[u * d, v * d, d, 1.0] for u, v, d, _ in samples])
    Y = np.array([p for _, _, _, p in samples])
    M, *_ = np.linalg.lstsq(F, Y, rcond=None)
    M = M.T
    res = F @ M.T - Y
    rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
    return M, rms
