"""Small rotation helpers (numpy only)."""
import numpy as np


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=float)


def rotvec_to_mat(r):
    r = np.asarray(r, dtype=float)
    th = float(np.linalg.norm(r))
    if th < 1e-9:
        return np.eye(3) + skew(r)
    k = r / th
    K = skew(k)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def mat_to_rotvec(R):
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = float(np.arccos(c))
    if th < 1e-6:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    if np.pi - th < 1e-2:
        # near pi: axis from the diagonal
        A = ((R + R.T) / 2.0 - c * np.eye(3)) / (1.0 - c)
        i = int(np.argmax(np.diag(A)))
        k = A[:, i] / np.sqrt(max(A[i, i], 1e-12))
        k = k / np.linalg.norm(k)
        w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if float(np.dot(k, w)) < 0:
            k = -k
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(th))
    return w * th


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_angle(Ra, Rb):
    """Angle of relative rotation between two rotation matrices."""
    return float(np.linalg.norm(mat_to_rotvec(Ra @ Rb.T)))


def quat_wxyz_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
