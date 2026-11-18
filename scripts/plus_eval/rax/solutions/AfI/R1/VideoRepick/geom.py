"""Geometry: back-projection models (self-calibrated against the pixel tool), rotations."""
import math

import numpy as np


def quat_to_R(q, order="wxyz"):
    q = np.asarray(q, float)
    if order == "xyzw":
        x, y, z, w = q
    else:
        w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def axang_to_R(v):
    v = np.asarray(v, float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def wrap_quarter(a):
    """Wrap angle to [-pi/4, pi/4) (cube 90-degree symmetry)."""
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


def hand_yaw(state):
    """Yaw (rad) of a horizontal hand axis; meaningful modulo 90 degrees."""
    R = axang_to_R(state[3:6])
    # the axis most aligned with vertical is the approach axis; use the most horizontal other
    zs = np.abs(R[2, :])
    order = np.argsort(zs)  # most horizontal first
    ax = R[:, order[0]]
    return math.atan2(ax[1], ax[0])


def approach_axis_down_err(state):
    """Angle (deg) between the hand's most vertical axis and the world vertical."""
    R = axang_to_R(state[3:6])
    zs = np.abs(R[2, :])
    return math.degrees(math.acos(min(1.0, float(zs.max()))))


class LinearBackproj:
    """world = A @ [u*d, v*d, d, 1]; exact for any pinhole convention with z-depth."""

    def __init__(self, A, kind="linear", resid=None):
        self.A = A
        self.kind = kind
        self.resid = resid

    def points(self, depth):
        H, W = depth.shape
        v, u = np.mgrid[0:H, 0:W].astype(np.float64)
        d = depth
        X = np.stack([u * d, v * d, d, np.ones_like(d)], -1)
        P = X @ self.A.T
        return P

    def point(self, u, v, d):
        return self.A @ np.array([u * d, v * d, d, 1.0])

    def project(self, xyz):
        """Approximate inverse: world -> (u, v). Solve A[:, :3] @ [ud, vd, d] = xyz - A[:,3]."""
        M = self.A[:, :3]
        try:
            s = np.linalg.solve(M, np.asarray(xyz, float) - self.A[:, 3])
        except Exception:
            return None
        if abs(s[2]) < 1e-9:
            return None
        return s[0] / s[2], s[1] / s[2]


def fit_linear(samples):
    """samples: list of (u, v, d, xyz). Returns LinearBackproj and median residual (m)."""
    if len(samples) < 8:
        return None, None
    X = np.array([[u * d, v * d, d, 1.0] for u, v, d, _ in samples])
    Y = np.array([p for *_, p in samples])
    A, *_ = np.linalg.lstsq(X, Y, rcond=None)
    A = A.T
    res = np.linalg.norm(X @ A.T - Y, axis=1)
    # one round of outlier rejection
    keep = res < max(0.01, 3 * np.median(res))
    if keep.sum() >= 8 and keep.sum() < len(samples):
        A2, *_ = np.linalg.lstsq(X[keep], Y[keep], rcond=None)
        A = A2.T
        res = np.linalg.norm(X @ A.T - Y, axis=1)
    return LinearBackproj(A, "linear", float(np.median(res))), float(np.median(res))


def param_models(K, pose):
    """Candidate parametric models from metadata calibration, as LinearBackproj-like
    callables. Returns list of (name, model)."""
    out = []
    if K is None or pose is None:
        return out
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if "matrix" in pose:
        Rs = [("mat", pose["matrix"][:3, :3], pose["matrix"][:3, 3])]
    else:
        Rs = []
        for order in ("wxyz", "xyzw"):
            Rs.append((order, quat_to_R(pose["quat"], order), pose["position"]))
    for name, R, t in Rs:
        for conv in ("gl", "cv"):
            for off in (0.0, 0.5):
                # camera coords = C @ [u*d, v*d, d]
                if conv == "gl":
                    C = np.array([[1 / fx, 0, -(cx - off) / fx], [0, -1 / fy, (cy - off) / fy], [0, 0, -1.0]])
                else:
                    C = np.array([[1 / fx, 0, -(cx - off) / fx], [0, 1 / fy, -(cy - off) / fy], [0, 0, 1.0]])
                M = R @ C
                A = np.concatenate([M, np.asarray(t, float).reshape(3, 1)], 1)
                out.append((f"{name}-{conv}-{off}", LinearBackproj(A, f"param:{name}-{conv}-{off}")))
    return out


def choose_model(samples, K, pose):
    """Pick the best back-projection model given tool samples. Returns (model, resid)."""
    best = (None, 1e9)
    lin, r = fit_linear(samples) if samples else (None, None)
    if lin is not None:
        best = (lin, r)
    if samples:
        X = [(u, v, d) for u, v, d, _ in samples]
        Y = np.array([p for *_, p in samples])
        for name, m in param_models(K, pose):
            P = np.array([m.point(u, v, d) for u, v, d in X])
            r2 = float(np.median(np.linalg.norm(P - Y, axis=1)))
            if r2 < best[1] - 0.002:
                best = (m, r2)
    elif K is not None and pose is not None:
        ms = param_models(K, pose)
        if ms:
            best = (ms[0][1], None)  # documented convention: wxyz, OpenGL, no offset
    return best
