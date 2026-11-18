"""Geometry helpers: rotations, camera models, back-projection with convention self-calibration.

No simulator imports; numpy only.
"""
import math
import numpy as np


# ----------------------------------------------------------------------------- rotations
def rotvec_to_R(r):
    r = np.asarray(r, dtype=float).reshape(3)
    th = float(np.linalg.norm(r))
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def R_to_rotvec(R):
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: axis from the symmetric part
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        v = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return v / np.linalg.norm(v) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def quat_wxyz_to_R(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ----------------------------------------------------------------------------- parsing calibration
def _num_list(v):
    try:
        a = np.asarray(v, dtype=float)
        return a
    except Exception:
        return None


def parse_K(v):
    """Return 3x3 intrinsics from many plausible encodings, or None."""
    if v is None:
        return None
    if isinstance(v, dict):
        lower = {str(k).lower(): val for k, val in v.items()}
        if all(k in lower for k in ('fx', 'fy', 'cx', 'cy')):
            return np.array([[float(lower['fx']), 0, float(lower['cx'])],
                             [0, float(lower['fy']), float(lower['cy'])], [0, 0, 1.0]])
        for key in ('k', 'matrix', 'intrinsic', 'intrinsics', 'intrinsic_cv', 'intrinsic_matrix',
                    'camera_matrix', 'value', 'data'):
            if key in lower:
                K = parse_K(lower[key])
                if K is not None:
                    return K
        for val in v.values():
            if isinstance(val, (list, dict)):
                K = parse_K(val)
                if K is not None:
                    return K
        return None
    a = _num_list(v)
    if a is None:
        return None
    if a.shape == (3, 3):
        return a
    if a.size == 9:
        return a.reshape(3, 3)
    if a.size == 4:  # fx fy cx cy
        return np.array([[a[0], 0, a[2]], [0, a[1], a[3]], [0, 0, 1.0]])
    return None


def parse_pose(v):
    """Return (list of candidate 4x4 cam->world matrices, declared convention or None)."""
    conv = None
    cands = []
    if v is None:
        return cands, conv
    if isinstance(v, dict):
        lower = {str(k).lower(): val for k, val in v.items()}
        for ck in ('convention', 'frame_convention', 'camera_convention'):
            if ck in lower and isinstance(lower[ck], str):
                conv = lower[ck].lower()
        pos = None
        for pk in ('position', 'p', 'translation', 't', 'pos', 'xyz', 'position_m'):
            if pk in lower:
                a = _num_list(lower[pk])
                if a is not None and a.size == 3:
                    pos = a.reshape(3)
                    break
        R = None
        for qk in ('quaternion_wxyz', 'wxyz', 'quat_wxyz', 'q', 'quaternion', 'orientation', 'quat',
                   'rotation_wxyz'):
            if qk in lower:
                a = _num_list(lower[qk])
                if a is not None and a.size == 4:
                    if 'xyzw' in qk:
                        a = np.array([a[3], a[0], a[1], a[2]])
                    R = quat_wxyz_to_R(a)
                    break
        if R is None:
            for qk in ('quaternion_xyzw', 'xyzw', 'quat_xyzw'):
                if qk in lower:
                    a = _num_list(lower[qk])
                    if a is not None and a.size == 4:
                        R = quat_wxyz_to_R([a[3], a[0], a[1], a[2]])
                        break
        if R is None:
            for rk in ('rotation_matrix', 'rotation', 'r'):
                if rk in lower:
                    a = _num_list(lower[rk])
                    if a is not None and a.size == 9:
                        R = a.reshape(3, 3)
                        break
        if pos is not None and R is not None:
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = pos
            cands.append(T)
        for mk in ('cam2world_gl', 'cam2world', 'matrix', 'pose', 'transform', 'world_from_camera',
                   'extrinsic', 'extrinsic_cv', 'world2cam', 'value', 'data'):
            if mk in lower:
                a = _num_list(lower[mk]) if not isinstance(lower[mk], dict) else None
                if a is not None and a.size == 16:
                    T = a.reshape(4, 4)
                    cands.append(T)
                elif a is not None and a.size == 12:
                    T = np.eye(4)
                    T[:3, :] = a.reshape(3, 4)
                    cands.append(T)
                elif isinstance(lower[mk], dict):
                    c2, cv2 = parse_pose(lower[mk])
                    cands.extend(c2)
                    conv = conv or cv2
    else:
        a = _num_list(v)
        if a is not None and a.size == 16:
            cands.append(a.reshape(4, 4))
        elif a is not None and a.size == 7:  # xyz + wxyz
            T = np.eye(4)
            T[:3, :3] = quat_wxyz_to_R(a[3:])
            T[:3, 3] = a[:3]
            cands.append(T)
    out = []
    for T in cands:
        T = np.asarray(T, dtype=float)
        if not np.all(np.isfinite(T)):
            continue
        out.append(T)
        try:
            out.append(np.linalg.inv(T))  # in case it is world->camera
        except Exception:
            pass
    return out, conv


# ----------------------------------------------------------------------------- camera model
class CamModel:
    """Back-projection with explicit convention parameters."""

    def __init__(self, K, T, gl=True, half=0.0, ray=False, swap=False):
        self.K = np.asarray(K, dtype=float)
        self.T = np.asarray(T, dtype=float)
        self.gl, self.half, self.ray, self.swap = gl, half, ray, swap

    def desc(self):
        return dict(gl=self.gl, half=self.half, ray=self.ray, swap=self.swap)

    def backproject(self, u, v, d):
        """u: column, v: row (arrays), d: depth in metres -> (N,3) world."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        d = np.asarray(d, dtype=float)
        if self.swap:
            u, v = v, u
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        xn = (u + self.half - cx) / fx
        yn = (v + self.half - cy) / fy
        if self.ray:
            d = d / np.sqrt(1 + xn * xn + yn * yn)
        if self.gl:
            pc = np.stack([xn * d, -yn * d, -d], axis=-1)
        else:
            pc = np.stack([xn * d, yn * d, d], axis=-1)
        return pc @ self.T[:3, :3].T + self.T[:3, 3]

    def project(self, P):
        """world (N,3) -> (u, v, depth_along_axis)."""
        P = np.atleast_2d(np.asarray(P, dtype=float))
        Ti = np.linalg.inv(self.T)
        pc = P @ Ti[:3, :3].T + Ti[:3, 3]
        if self.gl:
            d = -pc[:, 2]
            xn, yn = pc[:, 0] / np.where(np.abs(d) < 1e-9, 1e-9, d), -pc[:, 1] / np.where(np.abs(d) < 1e-9, 1e-9, d)
        else:
            d = pc[:, 2]
            xn, yn = pc[:, 0] / np.where(np.abs(d) < 1e-9, 1e-9, d), pc[:, 1] / np.where(np.abs(d) < 1e-9, 1e-9, d)
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        u = xn * fx + cx - self.half
        v = yn * fy + cy - self.half
        if self.swap:
            u, v = v, u
        return u, v, d

    def with_pose(self, T):
        return CamModel(self.K, T, self.gl, self.half, self.ray, self.swap)


def candidate_models(K, Ts, declared=None, W=256):
    out = []
    Ks = [K]
    try:
        if abs(K[0, 2] - W / 2.0) > 20 and K[0, 2] > 1:
            sc = (W / 2.0) / K[0, 2]
            K2 = K.copy()
            K2[:2, :] *= sc
            Ks.append(K2)
    except Exception:
        pass
    for K in Ks:
        out.extend(_cands(K, Ts, declared))
    return out


def _cands(K, Ts, declared=None):
    out = []
    gls = [True, False]
    if declared and 'cv' in declared:
        gls = [False, True]
    for T in Ts:
        for gl in gls:
            for half in (0.0, 0.5):
                for ray in (False, True):
                    for swap in (False, True):
                        out.append(CamModel(K, T, gl, half, ray, swap))
    return out


def flatness_score(model, depth, stride=6):
    """Fraction of sampled valid points within 1 cm of the dominant horizontal plane (z mode)."""
    H, W = depth.shape
    vv, uu = np.mgrid[0:H:stride, 0:W:stride]
    d = depth[vv, uu]
    ok = np.isfinite(d) & (d > 0)
    if ok.sum() < 20:
        return 0.0
    P = model.backproject(uu[ok], vv[ok], d[ok])
    z = P[:, 2]
    hist, edges = np.histogram(z, bins=200)
    i = int(np.argmax(hist))
    z0 = 0.5 * (edges[i] + edges[i + 1])
    return float(np.mean(np.abs(z - z0) < 0.01))
