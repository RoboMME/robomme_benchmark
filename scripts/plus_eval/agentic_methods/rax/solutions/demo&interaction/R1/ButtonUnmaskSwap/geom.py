"""Geometry helpers: rotations, camera back-projection with convention candidates."""
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


def rotvec_to_R(v):
    v = np.asarray(v, float)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def R_to_rotvec(R):
    R = np.asarray(R, float)
    c = (np.trace(R) - 1) / 2
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if math.pi - th < 1e-4:
        # near pi: use the diagonal
        M = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(M), 0, None))
        i = int(np.argmax(k))
        if k[i] > 1e-9:
            k = M[:, i] / k[i]
        k = k / (np.linalg.norm(k) or 1)
        return k * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return v * th


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def wrap_quarter(a):
    """wrap into (-pi/4, pi/4]"""
    q = math.pi / 2
    return (a + q / 2) % q - q / 2


# ---------------------------------------------------------------------------
# Camera models.  A camera is (K 3x3, Rcw 3x3, t 3) with a convention id.
# conv: 'gl'  camera looks -Z, +Y up (OpenGL, documented);
#       'cv'  camera looks +Z, +Y down (OpenCV)
# inv:  False -> (R,t) is camera-to-world; True -> world-to-camera
# off:  pixel-centre offset 0 or 0.5
CONVENTIONS = [(c, inv, off) for c in ('gl', 'cv') for inv in (False, True) for off in (0.5, 0.0)]


class Camera:
    def __init__(self, K, R, t, conv=('gl', False, 0.5)):
        self.K = np.asarray(K, float)
        R = np.asarray(R, float)
        t = np.asarray(t, float).reshape(3)
        c, inv, off = conv
        if inv:
            R, t = R.T, -R.T @ t
        self.R, self.t, self.c, self.off, self.conv = R, t, c, off, conv
        self.fx, self.fy = self.K[0, 0], self.K[1, 1]
        self.cx, self.cy = self.K[0, 2], self.K[1, 2]

    def backproject(self, u, v, d):
        """u, v pixel coords (arrays), d depth (m) -> Nx3 world"""
        u = np.asarray(u, float) + self.off
        v = np.asarray(v, float) + self.off
        d = np.asarray(d, float)
        x = (u - self.cx) / self.fx * d
        y = (v - self.cy) / self.fy * d
        if self.c == 'gl':
            P = np.stack([x, -y, -d], -1)
        else:
            P = np.stack([x, y, d], -1)
        return P @ self.R.T + self.t

    def project(self, X):
        X = np.asarray(X, float).reshape(-1, 3)
        P = (X - self.t) @ self.R
        if self.c == 'gl':
            d = -P[:, 2]
            x, y = P[:, 0], -P[:, 1]
        else:
            d = P[:, 2]
            x, y = P[:, 0], P[:, 1]
        d = np.where(np.abs(d) < 1e-9, 1e-9, d)
        u = x / d * self.fx + self.cx - self.off
        v = y / d * self.fy + self.cy - self.off
        return u, v, d

    def position(self):
        return self.t.copy()

    def axis(self):
        """optical axis direction in world"""
        return -self.R[:, 2] if self.c == 'gl' else self.R[:, 2]


def cloud(cam, depth, step=1, umin=0, umax=None, vmin=0, vmax=None):
    """Back-project an entire depth image (subsampled). Returns (pts Nx3, u, v)."""
    H, W = depth.shape
    vs, us = np.mgrid[0:H:step, 0:W:step]
    d = depth[::step, ::step]
    ok = d > 1e-4
    us, vs, d = us[ok], vs[ok], d[ok]
    return cam.backproject(us, vs, d), us, vs
