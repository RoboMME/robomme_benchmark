"""Rotation / camera geometry helpers (numpy only)."""
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


def rotvec_to_R(r):
    r = np.asarray(r, dtype=float)
    th = float(np.linalg.norm(r))
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def R_to_rotvec(R):
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: use the diagonal
        A = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(A)))
        v = A[:, i] / math.sqrt(max(A[i, i], 1e-12))
        return v * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return v * th


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_err_world(R_goal, R_cur):
    """World-frame rotation vector e with R_goal = exp(e) R_cur."""
    return R_to_rotvec(R_goal @ R_cur.T)


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def wrap_quarter(a):
    """Wrap angle into [-pi/4, pi/4) (square symmetry)."""
    q = math.pi / 2
    return (a + q / 2) % q - q / 2


def straighten_down(R):
    """Return R rotated minimally so its approach axis points to world -z.
    Returns (R_down, approach_col_index, tilt_rad)."""
    down = np.array([0, 0, -1.0])
    dots = [float(R[:, i] @ down) for i in range(3)]
    idx = int(np.argmax([abs(d) for d in dots]))
    sgn = 1.0 if dots[idx] > 0 else -1.0
    a = sgn * R[:, idx]
    tilt = math.acos(max(-1.0, min(1.0, float(a @ down))))
    axis = np.cross(a, down)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return R.copy(), idx, sgn, tilt
    corr = rotvec_to_R(axis / n * tilt)
    return corr @ R, idx, sgn, tilt


def hand_yaw(R, approach_idx):
    """Yaw (world) of the first hand axis that is not the approach axis."""
    j = 0 if approach_idx != 0 else 1
    v = R[:, j]
    return math.atan2(v[1], v[0])


# ---------------- camera back-projection -----------------
CONVENTIONS = []
for conv in ("gl", "cv", "ros"):
    for half in (0.5, 0.0):
        for inv in (False, True):
            CONVENTIONS.append((conv, half, inv))


def cam_points(depth, K, R, t, convention=("gl", 0.5, False), us=None, vs=None):
    """Back-project depth (HxW metres, linear along the optical axis) to world.
    If us/vs given, only those pixels (depth must then be an array of same shape)."""
    conv, half, inv = convention
    fx, fy, cx, cy = K
    if us is None:
        H, W = depth.shape
        vs, us = np.mgrid[0:H, 0:W].astype(float)
        d = depth
    else:
        d = depth
    x = (us + half - cx) / fx
    y = (vs + half - cy) / fy
    if conv == "gl":
        pc = np.stack([x * d, -y * d, -d], axis=-1)
    elif conv == "cv":
        pc = np.stack([x * d, y * d, d], axis=-1)
    else:  # ros: x forward, y left, z up
        pc = np.stack([d, -x * d, -y * d], axis=-1)
    if inv:
        Rw = R.T
        tw = -R.T @ t
    else:
        Rw, tw = R, t
    return pc @ Rw.T + tw


def project(points, K, R, t, convention=("gl", 0.5, False)):
    """World points (N,3) -> pixel (u,v) and depth, inverse of cam_points."""
    conv, half, inv = convention
    fx, fy, cx, cy = K
    if inv:
        Rw, tw = R.T, -R.T @ t
    else:
        Rw, tw = R, t
    pc = (np.asarray(points) - tw) @ Rw  # = Rw^T (p - t)
    if conv == "gl":
        d = -pc[:, 2]
        x, y = pc[:, 0] / d, -pc[:, 1] / d
    elif conv == "cv":
        d = pc[:, 2]
        x, y = pc[:, 0] / d, pc[:, 1] / d
    else:
        d = pc[:, 0]
        x, y = -pc[:, 1] / d, -pc[:, 2] / d
    u = x * fx + cx - half
    v = y * fy + cy - half
    return u, v, d


def min_area_rect_angle(xy):
    """Angle (mod pi/2) of the min-area bounding rectangle of 2-D points, plus (side_a, side_b)."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 3:
        return 0.0, (0.0, 0.0)
    c = xy - xy.mean(0)
    best = None
    for deg in np.arange(0.0, 90.0, 1.0):
        a = math.radians(deg)
        ca, sa = math.cos(a), math.sin(a)
        px = c[:, 0] * ca + c[:, 1] * sa
        py = -c[:, 0] * sa + c[:, 1] * ca
        lo_x, hi_x = np.percentile(px, [2, 98])
        lo_y, hi_y = np.percentile(py, [2, 98])
        area = (hi_x - lo_x) * (hi_y - lo_y)
        if best is None or area < best[0]:
            best = (area, a, hi_x - lo_x, hi_y - lo_y)
    # refine
    a0 = best[1]
    for a in np.linspace(a0 - math.radians(1), a0 + math.radians(1), 9):
        ca, sa = math.cos(a), math.sin(a)
        px = c[:, 0] * ca + c[:, 1] * sa
        py = -c[:, 0] * sa + c[:, 1] * ca
        lo_x, hi_x = np.percentile(px, [2, 98])
        lo_y, hi_y = np.percentile(py, [2, 98])
        area = (hi_x - lo_x) * (hi_y - lo_y)
        if area < best[0]:
            best = (area, a, hi_x - lo_x, hi_y - lo_y)
    return wrap_quarter(best[1]), (best[2], best[3])
