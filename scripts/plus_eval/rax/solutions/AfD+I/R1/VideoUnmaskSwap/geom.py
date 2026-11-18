"""Geometry helpers: rotations, per-camera linear back-projection, wrist pose transfer.

A pinhole camera maps pixel (u, v) with linear depth d to the world point
    P = M @ [u*d, v*d, d] + C
for a fixed 3x3 matrix M and camera centre C.  We fit (M, C) by least squares against
the robomme_pixel_to_position tool on the first execution observation, which removes
any dependence on intrinsics/pose conventions.  A convention-based construction from
camera_intrinsics / camera_poses (OpenGL) is the fallback.
"""
import numpy as np


def rodrigues(aa):
    aa = np.asarray(aa, float)
    th = np.linalg.norm(aa)
    if th < 1e-12:
        return np.eye(3)
    k = aa / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def log_rot(R):
    """Rotation matrix -> axis-angle vector (robust near pi)."""
    R = np.asarray(R, float)
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if np.pi - th < 1e-4:
        # axis from the diagonal
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(A), 0, None))
        i = int(np.argmax(k))
        if k[i] < 1e-9:
            return np.zeros(3)
        k = A[:, i] / k[i]
        k /= np.linalg.norm(k)
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def quat_wxyz_to_R(q):
    w, x, y, z = [float(t) for t in q]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class CamModel:
    """World point = M @ [u*d, v*d, d] + C (u = column, v = row, d metres)."""

    def __init__(self, M, C, source='fit', rms=None):
        self.M = np.asarray(M, float)
        self.C = np.asarray(C, float)
        self.source = source
        self.rms = rms

    def backproject(self, u, v, d):
        u = np.asarray(u, float); v = np.asarray(v, float); d = np.asarray(d, float)
        X = np.stack([u * d, v * d, d], -1)
        return X @ self.M.T + self.C

    def backproject_image(self, depth_m, mask=None, stride=1):
        H, W = depth_m.shape
        vv, uu = np.mgrid[0:H:stride, 0:W:stride]
        d = depth_m[::stride, ::stride]
        ok = d > 0
        if mask is not None:
            ok &= mask[::stride, ::stride]
        P = self.backproject(uu[ok], vv[ok], d[ok])
        return P, vv[ok], uu[ok]

    def project(self, P):
        """World points -> (u, v, d)."""
        P = np.atleast_2d(np.asarray(P, float))
        X = np.linalg.solve(self.M, (P - self.C).T).T
        d = X[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            return X[:, 0] / d, X[:, 1] / d, d

    def ray_plane(self, u, v, z0):
        """Intersect pixel rays with the horizontal plane z = z0."""
        u = np.asarray(u, float); v = np.asarray(v, float)
        dirs = np.stack([u, v, np.ones_like(u)], -1) @ self.M.T
        with np.errstate(divide='ignore', invalid='ignore'):
            d = (z0 - self.C[2]) / dirs[:, 2]
        return dirs * d[:, None] + self.C, d

    def transformed(self, R_rel, p_from, p_to):
        """Camera rigidly moved with an end effector from pose (R0,p_from) to (R_rel R0, p_to)."""
        M = R_rel @ self.M
        C = R_rel @ (self.C - p_from) + p_to
        return CamModel(M, C, source=self.source + '+ee', rms=self.rms)

    def u_axis_world(self):
        a = self.M[:, 0]
        return a / (np.linalg.norm(a) + 1e-12)


def fit_cam(uv, d, P, swap=False):
    """Least squares fit of M, C from pixels (N,2 as u,v), depth (N,), world points (N,3)."""
    uv = np.asarray(uv, float); d = np.asarray(d, float); P = np.asarray(P, float)
    u, v = (uv[:, 1], uv[:, 0]) if swap else (uv[:, 0], uv[:, 1])
    A = np.stack([u * d, v * d, d, np.ones_like(d)], 1)
    X, *_ = np.linalg.lstsq(A, P, rcond=None)
    res = A @ X - P
    rms = float(np.sqrt((res ** 2).sum(1).mean()))
    return CamModel(X[:3].T, X[3], 'fit', rms), rms


def cam_from_calib(K, pos, quat_wxyz, convention='opengl'):
    K = np.asarray(K, float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    R = quat_wxyz_to_R(quat_wxyz)
    if str(convention).lower().startswith('opengl'):
        B = np.array([[1 / fx, 0, -cx / fx], [0, -1 / fy, cy / fy], [0, 0, -1.0]])
    else:  # opencv: x right, y down, z forward
        B = np.array([[1 / fx, 0, -cx / fx], [0, 1 / fy, -cy / fy], [0, 0, 1.0]])
    return CamModel(R @ B, np.asarray(pos, float), 'calib-' + str(convention))
