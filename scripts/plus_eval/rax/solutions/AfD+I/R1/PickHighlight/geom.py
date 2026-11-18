"""Rotations and camera back-projection."""
import numpy as np


def rotvec_to_mat(r):
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def mat_to_rotvec(R):
    R = np.asarray(R, float)
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c))
    th = np.arccos(c)
    if th < 1e-7:
        return np.zeros(3)
    if np.pi - th < 1e-4:
        # near pi: axis from diagonal
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.maximum(np.diag(A), 0))
        i = int(np.argmax(k))
        if k[i] < 1e-9:
            return np.zeros(3)
        axis = A[:, i] / k[i]
        axis /= np.linalg.norm(axis)
        return axis * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def quat_wxyz_to_mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def wrap_quarter(a):
    """Wrap to [-pi/4, pi/4) (square symmetry)."""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


class CameraModel:
    """Maps (u, v, depth) -> world xyz.  Either analytic (intrinsics+pose) or a fitted affine model.

    `variant` selects the analytic convention tried during calibration.
    """

    def __init__(self, kind, **kw):
        self.kind = kind
        self.__dict__.update(kw)

    def backproject(self, u, v, d):
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        d = np.asarray(d, float)
        if self.kind == "affine":
            X = np.stack([u * d, v * d, d, np.ones_like(d)], axis=-1)
            return X @ self.A  # A is 4x3
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        uu = u + self.uoff
        vv = v + self.voff
        if self.swap_uv:
            uu, vv = vv, uu
        xc = (uu - cx) / fx * d
        yc = (vv - cy) / fy * d
        if self.convention == "opengl":
            pc = np.stack([xc, -yc, -d], axis=-1)
        else:  # opencv
            pc = np.stack([xc, yc, d], axis=-1)
        return pc @ self.R.T + self.t


def analytic_variants(intr, pose):
    if intr is None or pose is None:
        return []
    if "mat" in pose:
        M = np.asarray(pose["mat"], float)
        R, t = M[:3, :3], M[:3, 3]
    else:
        R, t = quat_wxyz_to_mat(pose["quat"]), np.asarray(pose["pos"], float)
    out = []
    for conv in ("opengl", "opencv"):
        for off in (0.0, 0.5):
            for swap in (False, True):
                out.append(CameraModel("analytic", fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"],
                                       R=R, t=t, convention=conv, uoff=off, voff=off, swap_uv=swap,
                                       name=f"{conv}-off{off}-swap{swap}"))
    return out


def fit_affine(uvd, xyz):
    uvd = np.asarray(uvd, float)
    xyz = np.asarray(xyz, float)
    u, v, d = uvd[:, 0], uvd[:, 1], uvd[:, 2]
    X = np.stack([u * d, v * d, d, np.ones_like(d)], axis=-1)
    A, *_ = np.linalg.lstsq(X, xyz, rcond=None)
    return CameraModel("affine", A=A, name="affine")


def model_error(model, uvd, xyz):
    uvd = np.asarray(uvd, float)
    p = model.backproject(uvd[:, 0], uvd[:, 1], uvd[:, 2])
    return np.linalg.norm(p - np.asarray(xyz, float), axis=1)
