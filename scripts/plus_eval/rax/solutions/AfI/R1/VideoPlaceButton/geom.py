"""Rotation and camera-mapping utilities (numpy only)."""
import math
import numpy as np


def rotvec_to_mat(r):
    r = np.asarray(r, dtype=float)
    th = float(np.linalg.norm(r))
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def mat_to_rotvec(R):
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if math.pi - th < 1e-3:
        # near pi: axis from symmetric part
        A = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(A)))
        ax = A[:, i] / math.sqrt(max(A[i, i], 1e-12))
        ax = ax / np.linalg.norm(ax)
        # fix sign using skew part for consistency
        w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if np.dot(w, ax) < 0:
            ax = -ax
        return ax * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2 * math.sin(th)) * th


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def wrap_quarter(a):
    """Wrap an angle to [-pi/4, pi/4) (square symmetry)."""
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


def quat_wxyz_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class DepthMap:
    """world = M @ [u*d, v*d, d] + t  (linear in depth; fitted or from calibration)."""

    def __init__(self, M, t, source="", residual=None):
        self.M = np.asarray(M, dtype=float)
        self.t = np.asarray(t, dtype=float)
        self.source = source
        self.residual = residual
        try:
            self.Minv = np.linalg.inv(self.M)
        except Exception:
            self.Minv = None

    def backproject(self, depth):
        """depth: HxW metres (nan invalid) -> HxWx3 world points (nan invalid)."""
        H, W = depth.shape
        v, u = np.mgrid[0:H, 0:W].astype(float)
        d = depth
        X = np.stack([u * d, v * d, d], axis=-1)
        P = X @ self.M.T + self.t
        return P

    def pixel_point(self, u, v, d):
        return self.M @ np.array([u * d, v * d, d]) + self.t

    def project(self, pts):
        """world Nx3 -> (u, v, d) arrays."""
        if self.Minv is None:
            return None
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        X = (pts - self.t) @ self.Minv.T
        d = X[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            return X[:, 0] / d, X[:, 1] / d, d


def fit_depth_map(uvd, world):
    """Least squares fit with outlier rejection. uvd: Nx3 (u,v,d), world Nx3."""
    uvd = np.asarray(uvd, dtype=float)
    world = np.asarray(world, dtype=float)
    keep = np.ones(len(uvd), dtype=bool)
    M = t = None
    res = None
    for _ in range(3):
        if keep.sum() < 6:
            return None
        A = np.stack([uvd[keep, 0] * uvd[keep, 2], uvd[keep, 1] * uvd[keep, 2], uvd[keep, 2], np.ones(keep.sum())], axis=1)
        sol, *_ = np.linalg.lstsq(A, world[keep], rcond=None)
        M = sol[:3].T
        t = sol[3]
        Aall = np.stack([uvd[:, 0] * uvd[:, 2], uvd[:, 1] * uvd[:, 2], uvd[:, 2], np.ones(len(uvd))], axis=1)
        pred = Aall @ sol
        res = np.linalg.norm(pred - world, axis=1)
        newkeep = res < max(0.006, 3 * np.median(res[keep]))
        if (newkeep == keep).all():
            break
        keep = newkeep
    return DepthMap(M, t, "fit", residual=float(np.median(res[keep])) if res is not None else None), keep


def calib_depth_map(K, R_wc, t_wc, convention="opengl"):
    """Depth map from intrinsics and camera->world pose."""
    K = np.asarray(K, dtype=float)
    Kinv = np.linalg.inv(K)
    if convention == "opengl":
        F = np.diag([1.0, -1.0, -1.0])
    else:
        F = np.eye(3)
    M = np.asarray(R_wc) @ F @ Kinv
    return DepthMap(M, t_wc, "calib-" + convention)


def min_area_yaw(xy):
    """Yaw (mod 90 deg, in [-45,45)) of the min-area bounding rectangle of 2-D points."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 5:
        return None, None
    c = xy - xy.mean(0)
    best = None
    for deg in np.arange(-45.0, 45.0, 1.0):
        a = math.radians(deg)
        ca, sa = math.cos(a), math.sin(a)
        x = c[:, 0] * ca + c[:, 1] * sa
        y = -c[:, 0] * sa + c[:, 1] * ca
        lx = np.percentile(x, 97) - np.percentile(x, 3)
        ly = np.percentile(y, 97) - np.percentile(y, 3)
        area = lx * ly
        if best is None or area < best[0]:
            best = (area, a, lx, ly)
    return best[1], (best[2], best[3])
