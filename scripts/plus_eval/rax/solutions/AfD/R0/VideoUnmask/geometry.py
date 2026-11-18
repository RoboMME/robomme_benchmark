"""Rotations and a convention-free camera back-projection model.

Camera model:  W = C + d * A @ [u, v, 1]   (d = depth value from the depth image in metres)
fitted by least squares against robomme_pixel_to_position samples, or built from
the published calibration.  For a wrist camera rigidly attached to the end
effector, the model at another end-effector pose follows by the rigid transform.
"""
import math
import numpy as np


def rotvec_to_mat(rv):
    rv = np.asarray(rv, float)
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def mat_to_rotvec(R):
    R = np.asarray(R, float)
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: use the symmetric part
        M = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        # fix sign using the antisymmetric part when available
        w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if np.dot(w, k) < 0:
            k = -k
        return k / np.linalg.norm(k) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2 * math.sin(th)) * th


def quat_wxyz_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def wrap_half_pi(a):
    """wrap to [-pi/4, pi/4) (symmetry of a square)."""
    q = math.pi / 2
    return (a + q / 2) % q - q / 2


class CamModel:
    def __init__(self, C, A, kind='fit', resid=None):
        self.C = np.asarray(C, float).reshape(3)
        self.A = np.asarray(A, float).reshape(3, 3)
        self.kind = kind
        self.resid = resid
        try:
            self.Ainv = np.linalg.inv(self.A)
        except Exception:
            self.Ainv = np.linalg.pinv(self.A)

    def backproject(self, u, v, d):
        """u,v,d arrays (same shape) -> (...,3) world points."""
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        d = np.asarray(d, float)
        p = np.stack([u, v, np.ones_like(u)], -1)
        ray = p @ self.A.T
        return self.C + d[..., None] * ray

    def project(self, W):
        """(...,3) world -> (u, v, d)."""
        W = np.asarray(W, float)
        q = (W - self.C) @ self.Ainv.T
        d = q[..., 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = q[..., 0] / d
            v = q[..., 1] / d
        return u, v, d

    def moved(self, R_old, p_old, R_new, p_new):
        """Model for the same rigidly attached camera after the carrier moved."""
        dR = np.asarray(R_new) @ np.asarray(R_old).T
        C = np.asarray(p_new) + dR @ (self.C - np.asarray(p_old))
        return CamModel(C, dR @ self.A, kind=self.kind + '+moved', resid=self.resid)

    def to_dict(self):
        return {'C': self.C.tolist(), 'A': self.A.tolist(), 'kind': self.kind, 'resid': self.resid}


def fit_cam_model(uv, d, W):
    """Least squares fit of W = C + d*A*[u,v,1].  uv (N,2), d (N,), W (N,3)."""
    uv = np.asarray(uv, float)
    d = np.asarray(d, float)
    W = np.asarray(W, float)
    N = len(d)
    if N < 6:
        return None
    p = np.concatenate([uv, np.ones((N, 1))], 1)
    # unknowns per world axis: C_i (1) + A_i (3) -> design [1, d*u, d*v, d]
    X = np.concatenate([np.ones((N, 1)), d[:, None] * p], 1)
    C = np.zeros(3)
    A = np.zeros((3, 3))
    for i in range(3):
        sol, *_ = np.linalg.lstsq(X, W[:, i], rcond=None)
        C[i] = sol[0]
        A[i] = sol[1:]
    m = CamModel(C, A, 'fit')
    res = np.linalg.norm(m.backproject(uv[:, 0], uv[:, 1], d) - W, axis=1)
    m.resid = float(np.median(res))
    m.resid_max = float(res.max())
    return m


def fit_cam_model_fixed_C(uv, d, W, C):
    """Fit A only, given camera centre C."""
    uv = np.asarray(uv, float)
    d = np.asarray(d, float)
    W = np.asarray(W, float)
    N = len(d)
    p = np.concatenate([uv, np.ones((N, 1))], 1)
    X = d[:, None] * p
    A = np.zeros((3, 3))
    for i in range(3):
        sol, *_ = np.linalg.lstsq(X, W[:, i] - C[i], rcond=None)
        A[i] = sol
    m = CamModel(C, A, 'fitC')
    res = np.linalg.norm(m.backproject(uv[:, 0], uv[:, 1], d) - W, axis=1)
    m.resid = float(np.median(res))
    return m


def models_from_calibration(K, pos, quat_wxyz, W=256, H=256):
    """Candidate analytic models for several conventions (OpenGL first)."""
    K = np.asarray(K, float)
    if K.shape != (3, 3):
        K = K.reshape(3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Kinv_gl = np.array([[1 / fx, 0, -cx / fx], [0, -1 / fy, cy / fy], [0, 0, -1.0]])  # OpenGL cam: x right, y up, -z fwd
    Kinv_cv = np.array([[1 / fx, 0, -cx / fx], [0, 1 / fy, -cy / fy], [0, 0, 1.0]])
    R = quat_wxyz_to_mat(quat_wxyz)
    pos = np.asarray(pos, float)
    out = []
    out.append(CamModel(pos, R @ Kinv_gl, 'calib_gl'))
    out.append(CamModel(pos, R @ Kinv_cv, 'calib_cv'))
    # inverse pose interpretation (world->camera)
    Rt = R.T
    Cinv = -Rt @ pos
    out.append(CamModel(Cinv, Rt @ Kinv_gl, 'calib_gl_inv'))
    out.append(CamModel(Cinv, Rt @ Kinv_cv, 'calib_cv_inv'))
    # ROS/sapien camera link convention: x fwd, y left, z up
    M_ros = np.array([[0, 0, 1.0], [-1, 0, 0], [0, -1, 0]]) @ Kinv_cv
    out.append(CamModel(pos, R @ M_ros, 'calib_ros'))
    return out
