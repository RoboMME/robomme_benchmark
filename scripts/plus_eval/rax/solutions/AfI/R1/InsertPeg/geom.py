"""Rotation / pose helpers (numpy only)."""
import numpy as np


def rotvec_to_mat(r):
    r = np.asarray(r, dtype=float)
    th = float(np.linalg.norm(r))
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def mat_to_rotvec(R):
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = float(np.arccos(c))
    if th < 1e-9:
        return np.zeros(3)
    if th > np.pi - 1e-4:
        # near pi: use the symmetric part
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        v = M[:, i] / np.sqrt(max(M[i, i], 1e-12))
        v = v / np.linalg.norm(v)
        # pick sign consistent with antisymmetric part (if any)
        w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if np.dot(w, v) < 0:
            v = -v
        return v * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * np.sin(th)) * th


def rot_angle(R):
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(min(1.0, max(-1.0, c))))


def rotz(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def slerp_mat(R0, R1, a):
    d = mat_to_rotvec(R0.T @ R1)
    return R0 @ rotvec_to_mat(a * d)


def wrap(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class SE2:
    """Planar transform about world z: p' = Rz(th) (p - c) + c + t  (c pivot)."""

    def __init__(self, th=0.0, t=(0.0, 0.0), c=(0.0, 0.0)):
        self.th = float(th)
        self.t = np.array(t, dtype=float)[:2]
        self.c = np.array(c, dtype=float)[:2]

    def is_identity(self):
        return abs(self.th) < 1e-9 and np.linalg.norm(self.t) < 1e-9

    def apply_pos(self, p):
        p = np.array(p, dtype=float)
        q = p.copy()
        R = rotz(self.th)[:2, :2]
        q[:2] = R @ (p[:2] - self.c) + self.c + self.t
        return q

    def apply_rot(self, R):
        return rotz(self.th) @ R

    @staticmethod
    def blend(a, b, w):
        # linear blend of parameters (same pivot assumed close enough)
        w = float(min(1.0, max(0.0, w)))
        c = (1 - w) * a.c + w * b.c
        # re-express both with pivot c: t' = R(c - c0) + c0 + t - c
        def re(T):
            R = rotz(T.th)[:2, :2]
            return R @ (c - T.c) + T.c + T.t - c
        ta, tb = re(a), re(b)
        th = a.th + w * wrap(b.th - a.th)
        return SE2(th, (1 - w) * ta + w * tb, c)

    def __repr__(self):
        return "SE2(th=%.4f, t=[%.4f, %.4f], c=[%.3f, %.3f])" % (
            self.th, self.t[0], self.t[1], self.c[0], self.c[1])
