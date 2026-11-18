"""Geometry helpers: rotations, camera models, back-projection.  numpy only."""
import math
import numpy as np


def quat_wxyz_to_R(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotvec_to_R(r):
    r = np.asarray(r, float)
    th = float(np.linalg.norm(r))
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def R_to_rotvec(R):
    R = np.asarray(R, float)
    c = (np.trace(R) - 1) / 2
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: use the diagonal
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.maximum(np.diag(A), 0))
        i = int(np.argmax(k))
        if k[i] < 1e-9:
            return np.zeros(3)
        k = A[:, i] / k[i]
        k = k / (np.linalg.norm(k) or 1)
        return k * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2 * math.sin(th)) * th


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Camera:
    """Pinhole camera with world pose T_world_cam.

    convention 'opengl': camera looks along -Z, +Y up (image row 0 at top).
    convention 'opencv': camera looks along +Z, +Y down.
    """

    def __init__(self, K, R, t, convention='opengl', W=256, H=256):
        self.K = np.asarray(K, float)
        self.R = np.asarray(R, float)
        self.t = np.asarray(t, float).reshape(3)
        self.conv = convention
        self.W, self.H = W, H

    def copy_with_pose(self, R, t):
        return Camera(self.K, R, t, self.conv, self.W, self.H)

    def cam_points(self, u, v, d):
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        d = np.asarray(d, float)
        x = (u - cx) / fx * d
        y = (v - cy) / fy * d
        if self.conv == 'opengl':
            return np.stack([x, -y, -d], -1)
        return np.stack([x, y, d], -1)

    def backproject(self, u, v, d):
        pc = self.cam_points(u, v, d)
        return pc @ self.R.T + self.t

    def backproject_image(self, depth_m):
        H, W = depth_m.shape
        vv, uu = np.mgrid[0:H, 0:W]
        return self.backproject(uu + 0.0, vv + 0.0, depth_m)

    def project(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        pc = (P - self.t) @ self.R
        if self.conv == 'opengl':
            d = -pc[:, 2]
            x, y = pc[:, 0], -pc[:, 1]
        else:
            d = pc[:, 2]
            x, y = pc[:, 0], pc[:, 1]
        d_safe = np.where(np.abs(d) < 1e-9, 1e-9, d)
        u = self.K[0, 0] * x / d_safe + self.K[0, 2]
        v = self.K[1, 1] * y / d_safe + self.K[1, 2]
        return u, v, d

    def x_axis(self):
        """Image-right direction in world (finger axis on the wrist camera)."""
        return self.R[:, 0].copy()

    def optical_axis(self):
        return -self.R[:, 2].copy() if self.conv == 'opengl' else self.R[:, 2].copy()


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti
