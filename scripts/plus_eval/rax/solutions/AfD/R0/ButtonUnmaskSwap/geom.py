"""Rotations and camera back-projection models (numpy only)."""
import numpy as np


def rotvec_to_mat(v):
    v = np.asarray(v, dtype=np.float64)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def mat_to_rotvec(R):
    R = np.asarray(R, dtype=np.float64)
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c))
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > np.pi - 1e-4:
        # near pi: axis from the symmetric part
        M = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / np.sqrt(max(M[i, i], 1e-12))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def quat_wxyz_to_mat(q):
    w, x, y, z = [float(a) for a in q]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def wrap90(a):
    """Wrap an angle to [-pi/4, pi/4) (square symmetry)."""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


class CamModel:
    """world = A @ [u*d, v*d, d] + t  (u column, v row, d linear depth in metres)."""

    def __init__(self, A, t, kind="fit", rms=None):
        self.A = np.asarray(A, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64)
        self.kind = kind
        self.rms = rms

    def backproject(self, u, v, d):
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        F = np.stack([u * d, v * d, d], axis=-1)
        return F @ self.A.T + self.t

    def image_axis_world(self, which=0):
        """World direction of increasing u (which=0) or v (which=1)."""
        a = self.A[:, which]
        return a / (np.linalg.norm(a) + 1e-12)

    def transformed(self, R_rel, p_new, p_old):
        """Model for a camera rigidly attached to a frame that moved: X' = R_rel (X - p_old) + p_new."""
        return CamModel(R_rel @ self.A, R_rel @ (self.t - p_old) + p_new, kind=self.kind + "+ee", rms=self.rms)

    def to_dict(self):
        return {"A": self.A.tolist(), "t": self.t.tolist(), "kind": self.kind, "rms": self.rms}


def fit_linear(us, vs, ds, X):
    us, vs, ds = [np.asarray(a, dtype=np.float64) for a in (us, vs, ds)]
    X = np.asarray(X, dtype=np.float64)
    F = np.stack([us * ds, vs * ds, ds, np.ones_like(ds)], axis=1)
    if len(ds) < 6:
        return None
    sc = np.abs(F).max(axis=0) + 1e-9
    Fn = F / sc
    sv = np.linalg.svd(Fn, compute_uv=False)
    M, *_ = np.linalg.lstsq(Fn, X, rcond=None)
    M = M / sc[:, None]
    pred = F @ M
    res = np.linalg.norm(pred - X, axis=1)
    A = M[:3].T
    t = M[3]
    cond = float(sv[0] / max(sv[-1], 1e-12))
    return CamModel(A, t, kind="fit", rms=float(np.sqrt(np.mean(res ** 2)))), res, cond


def pinhole_candidates(K, pos, quat_wxyz):
    """Back-projection models for a few common conventions."""
    out = []
    try:
        K = np.asarray(K, dtype=np.float64)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        R = quat_wxyz_to_mat(quat_wxyz)
        p = np.asarray(pos, dtype=np.float64)
        for name, sy, sz in [("gl", -1.0, -1.0), ("cv", 1.0, 1.0)]:
            for off in (0.0, 0.5):
                # camera coords: xc = (u+off-cx)/fx*d, yc = sy*(v+off-cy)/fy*d, zc = sz*d
                C = np.array([[1 / fx, 0, (off - cx) / fx],
                              [0, sy / fy, sy * (off - cy) / fy],
                              [0, 0, sz]])
                out.append(CamModel(R @ C, p, kind=f"pinhole-{name}-{off}"))
    except Exception:
        pass
    return out


def parse_K(obj):
    """Intrinsics from a 3x3 list, a flat 9-list, or a dict with fx/fy/cx/cy."""
    try:
        if isinstance(obj, dict):
            for k in ("K", "matrix", "intrinsic_matrix", "camera_matrix", "k"):
                if k in obj:
                    return parse_K(obj[k])
            if all(k in obj for k in ("fx", "fy", "cx", "cy")):
                return np.array([[obj["fx"], 0, obj["cx"]], [0, obj["fy"], obj["cy"]], [0, 0, 1.0]], dtype=np.float64)
            return None
        a = np.asarray(obj, dtype=np.float64)
        if a.size == 9:
            return a.reshape(3, 3)
        if a.size == 4:
            return np.array([[a[0], 0, a[2]], [0, a[1], a[3]], [0, 0, 1.0]])
    except Exception:
        return None
    return None


def parse_pose(obj):
    """(position, quaternion_wxyz) from assorted pose layouts."""
    try:
        if isinstance(obj, dict):
            pos = None
            for k in ('position_m', "position", "p", "translation", "pos", "xyz"):
                if k in obj:
                    pos = np.asarray(obj[k], dtype=np.float64).reshape(3)
                    break
            q = None
            for k in ("quaternion_wxyz", "quat_wxyz", "wxyz", "quaternion", "q", "orientation"):
                if k in obj:
                    q = np.asarray(obj[k], dtype=np.float64).reshape(4)
                    if k in ("quaternion", "q", "orientation") and "xyzw" in str(obj.get("convention", "")).lower():
                        q = np.array([q[3], q[0], q[1], q[2]])
                    break
            if pos is None:
                for k in ("matrix", "T", "pose", "transform", "extrinsics", "cam2world", "cam2world_gl"):
                    if k in obj:
                        T = np.asarray(obj[k], dtype=np.float64).reshape(4, 4)
                        return T[:3, 3], T[:3, :3]
            if pos is not None and q is not None:
                return pos, quat_wxyz_to_mat(q)
            return None
        a = np.asarray(obj, dtype=np.float64)
        if a.size == 16:
            T = a.reshape(4, 4)
            return T[:3, 3], T[:3, :3]
        if a.size == 7:
            return a[:3], quat_wxyz_to_mat(a[3:])
    except Exception:
        return None
    return None


def pinhole_from(K, pose):
    """Models from K and (pos, R)."""
    out = []
    if K is None or pose is None:
        return out
    pos, R = pose
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    for name, sy, sz in [("gl", -1.0, -1.0), ("cv", 1.0, 1.0)]:
        for off in (0.0, 0.5):
            C = np.array([[1 / fx, 0, (off - cx) / fx],
                          [0, sy / fy, sy * (off - cy) / fy],
                          [0, 0, sz]])
            out.append(CamModel(R @ C, np.asarray(pos, dtype=np.float64), kind=f"pinhole-{name}-{off}"))
    return out
