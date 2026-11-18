"""Rotation helpers and a pinhole camera model (OpenGL convention by default)."""
import math
import numpy as np


def rotvec_to_mat(rv):
    rv = np.asarray(rv, float)
    th = float(np.linalg.norm(rv))
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def mat_to_rotvec(R):
    R = np.asarray(R, float)
    c = (np.trace(R) - 1) / 2
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if math.pi - th < 1e-4:
        # near pi: axis from the diagonal
        M = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return k / np.linalg.norm(k) * th
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


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def wrap_half_quarter(a_deg):
    """Wrap an angle in degrees into [-45, 45) modulo 90."""
    return (a_deg + 45.0) % 90.0 - 45.0


class Camera:
    """Pinhole camera. conv='opengl': looks along -Z, +Y up; 'opencv': looks along +Z, +Y down."""

    def __init__(self, K, R, t, conv='opengl', depth_kind='z', pix_offset=0.0):
        self.K = np.asarray(K, float)
        self.R = np.asarray(R, float)
        self.t = np.asarray(t, float).reshape(3)
        self.conv = conv
        self.depth_kind = depth_kind
        self.pix_offset = pix_offset

    def ray_cam(self, u, v):
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        x = (np.asarray(u, float) + self.pix_offset - cx) / fx
        y = (np.asarray(v, float) + self.pix_offset - cy) / fy
        if self.conv == 'opengl':
            return np.stack([x, -y, -np.ones_like(x)], -1)
        return np.stack([x, y, np.ones_like(x)], -1)

    def backproject(self, u, v, d):
        r = self.ray_cam(u, v)
        d = np.asarray(d, float)
        if self.depth_kind == 'ray':
            r = r / np.linalg.norm(r, axis=-1, keepdims=True)
        pc = r * d[..., None]
        return pc @ self.R.T + self.t

    def ray_world(self, u, v):
        r = self.ray_cam(u, v) @ self.R.T
        return r / np.linalg.norm(r, axis=-1, keepdims=True)

    def project(self, P):
        pc = (np.asarray(P, float) - self.t) @ self.R
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        if self.conv == 'opengl':
            z = -pc[..., 2]
            u = fx * pc[..., 0] / z + cx - self.pix_offset
            v = -fy * pc[..., 1] / z + cy - self.pix_offset
        else:
            z = pc[..., 2]
            u = fx * pc[..., 0] / z + cx - self.pix_offset
            v = fy * pc[..., 1] / z + cy - self.pix_offset
        return u, v, z

    def dense(self, depth):
        H, W = depth.shape
        vv, uu = np.mgrid[0:H, 0:W]
        P = self.backproject(uu, vv, depth)
        return P
