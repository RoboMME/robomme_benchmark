"""Camera geometry: OpenGL-convention camera poses (looks along -Z, +Y up), depth along optical axis."""
import numpy as np


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Cam:
    def __init__(self, K, pose):
        K = np.asarray(K, float)
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        self.R = quat_to_R(pose["quaternion_wxyz"])
        self.t = np.asarray(pose["position_m"], float)

    def deproject(self, u, v, d):
        """u,v pixel (arrays ok), d depth in metres -> world points (...,3)."""
        u = np.asarray(u, float); v = np.asarray(v, float); d = np.asarray(d, float)
        xc = (u - self.cx) / self.fx * d
        yc = -(v - self.cy) / self.fy * d
        zc = -d
        pc = np.stack([xc, yc, zc], -1)
        return pc @ self.R.T + self.t

    def project(self, p):
        p = np.asarray(p, float)
        pc = (p - self.t) @ self.R
        d = -pc[..., 2]
        u = self.cx + self.fx * pc[..., 0] / d
        v = self.cy - self.fy * pc[..., 1] / d
        return u, v, d

    def cloud(self, depth_m):
        h, w = depth_m.shape
        vv, uu = np.mgrid[0:h, 0:w]
        return self.deproject(uu + 0.0, vv + 0.0, depth_m)


def rotvec_to_R(r):
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def R_to_rotvec(R):
    R = np.asarray(R, float)
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > np.pi - 1e-4:
        # near pi: axis from diagonal
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(A), 0, None))
        i = int(np.argmax(k))
        k[i] = np.sqrt(A[i, i])
        for j in range(3):
            if j != i:
                k[j] = A[i, j] / k[i]
        k /= np.linalg.norm(k)
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def yaw_of_down_gripper(rotvec):
    """Yaw (rad) of the gripper's x-axis projected to the world xy plane."""
    R = rotvec_to_R(rotvec)
    ax = R[:, 0]
    return float(np.arctan2(ax[1], ax[0]))
