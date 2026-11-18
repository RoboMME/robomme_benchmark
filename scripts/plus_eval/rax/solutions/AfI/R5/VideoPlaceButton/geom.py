import numpy as np


def aa_to_R(v):
    v = np.asarray(v, float)
    th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def R_to_aa(R):
    c = (np.trace(R) - 1) / 2
    c = np.clip(c, -1, 1)
    th = np.arccos(c)
    if th < 1e-7:
        return np.zeros(3)
    if np.pi - th < 1e-4:
        # near pi: axis from diagonal
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(A), 0, None))
        i = int(np.argmax(k))
        k[:] = A[i] / k[i]
        k /= np.linalg.norm(k)
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


DOWN = np.diag([1.0, -1.0, -1.0])


def grip_yaw(aa):
    R = aa_to_R(aa)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def wrap_quarter(a):
    """wrap angle to [-pi/4, pi/4)"""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def rot_error(aa_cur, yaw_target):
    Rt = Rz(yaw_target) @ DOWN
    Rc = aa_to_R(aa_cur)
    return R_to_aa(Rt @ Rc.T)


def min_area_yaw(xy):
    """Brute-force minimum-area bounding box orientation of 2-D points; yaw in [0, pi/2)."""
    xy = np.asarray(xy, float)
    best = None
    for a in np.deg2rad(np.arange(0, 90, 1.0)):
        c, s = np.cos(a), np.sin(a)
        p = xy @ np.array([[c, -s], [s, c]])
        ext = p.max(0) - p.min(0)
        area = ext[0] * ext[1]
        if best is None or area < best[0]:
            ctr_rot = (p.max(0) + p.min(0)) / 2
            ctr = np.array([[c, -s], [s, c]]) @ ctr_rot
            best = (area, a, ctr, ext)
    return best[1], best[2], best[3]
