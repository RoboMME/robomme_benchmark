"""Rotations, camera parsing and back-projection."""
import math
import numpy as np


def quat_wxyz_to_R(q):
    w, x, y, z = [float(a) for a in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotvec_to_R(r):
    r = np.asarray(r, dtype=float)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def R_to_rotvec(R):
    R = np.asarray(R, dtype=float)
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: use diagonal
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(A), 0, None))
        i = int(np.argmax(k))
        k[i] = math.sqrt(max(A[i, i], 0))
        for j in range(3):
            if j != i:
                k[j] = A[i, j] / k[i] if k[i] > 1e-9 else 0
        k = k / (np.linalg.norm(k) or 1)
        return k * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return v * th


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _num_list(v):
    return isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v)


def parse_K(v):
    """Intrinsics in many possible encodings -> (fx, fy, cx, cy)."""
    if isinstance(v, dict):
        lk = {k.lower(): val for k, val in v.items()}
        if all(k in lk for k in ("fx", "fy", "cx", "cy")):
            return tuple(float(lk[k]) for k in ("fx", "fy", "cx", "cy"))
        for key in ("k", "matrix", "intrinsic", "intrinsics", "intrinsic_cv", "camera_matrix"):
            if key in lk:
                r = parse_K(lk[key])
                if r:
                    return r
        for val in v.values():
            r = parse_K(val)
            if r:
                return r
        return None
    if isinstance(v, (list, tuple)):
        a = None
        try:
            a = np.array(v, dtype=float)
        except Exception:
            return None
        if a.shape == (3, 3):
            return (a[0, 0], a[1, 1], a[0, 2], a[1, 2])
        if a.shape == (9,):
            a = a.reshape(3, 3)
            return (a[0, 0], a[1, 1], a[0, 2], a[1, 2])
        if a.shape == (4,):
            return tuple(a)
    return None


def parse_pose(v):
    """Pose -> (R 3x3, t 3, convention string)."""
    conv = ""
    if isinstance(v, dict):
        lk = {k.lower(): val for k, val in v.items()}
        for ck in ("convention", "frame_convention", "camera_convention"):
            if isinstance(lk.get(ck), str):
                conv = lk[ck].lower()
        pos = None
        for pk in ('position_m', "position", "p", "translation", "xyz", "pos"):
            if _num_list(lk.get(pk)) and len(lk[pk]) == 3:
                pos = np.array(lk[pk], dtype=float)
        q = None
        for qk in ("quaternion_wxyz", "wxyz", "quaternion", "q", "quat", "orientation"):
            if _num_list(lk.get(qk)) and len(lk[qk]) == 4:
                q = lk[qk]
                break
        if pos is not None and q is not None:
            return quat_wxyz_to_R(q), pos, conv
        for mk in ("matrix", "cam2world", "cam2world_gl", "pose", "extrinsic", "transform", "camera_to_world"):
            if mk in lk:
                r = parse_pose(lk[mk])
                if r:
                    return r[0], r[1], conv or r[2] or mk
        return None
    if isinstance(v, (list, tuple)):
        try:
            a = np.array(v, dtype=float)
        except Exception:
            return None
        if a.shape == (4, 4):
            return a[:3, :3], a[:3, 3], conv
        if a.shape == (16,):
            a = a.reshape(4, 4)
            return a[:3, :3], a[:3, 3], conv
        if a.shape == (7,):
            return quat_wxyz_to_R(a[3:]), a[:3], conv
    return None


def camera_params(m, cam):
    """Find intrinsics and pose for camera `cam` in metadata m."""
    from mcio import find_all
    K = pose = None
    for _, v in find_all(m, "camera_intrinsics"):
        if isinstance(v, dict) and cam in v:
            K = parse_K(v[cam])
            if K:
                break
    for _, v in find_all(m, "camera_poses"):
        if isinstance(v, dict) and cam in v:
            pose = parse_pose(v[cam])
            if pose:
                break
    return K, pose


VARIANTS = [(c, o, inv) for c in ("gl", "cv") for o in (0.0, 0.5) for inv in (False, True)]


def backproject(u, v, d, K, pose, variant):
    """u,v,d arrays -> world points Nx3 for a given convention variant."""
    conv, off, inv = variant
    fx, fy, cx, cy = K
    R, t, _ = pose
    x = (np.asarray(u, float) + off - cx) / fx * d
    y = (np.asarray(v, float) + off - cy) / fy * d
    if conv == "gl":
        pc = np.stack([x, -y, -np.asarray(d, float)], -1)
    else:
        pc = np.stack([x, y, np.asarray(d, float)], -1)
    if not inv:
        return pc @ np.asarray(R).T + np.asarray(t)
    return (pc - np.asarray(t)) @ np.asarray(R)


def project(points, K, pose, variant):
    """World Nx3 -> (u, v, depth) under variant (inverse of backproject)."""
    conv, off, inv = variant
    fx, fy, cx, cy = K
    R, t, _ = pose
    P = np.asarray(points, float)
    if not inv:
        pc = (P - t) @ np.asarray(R)
    else:
        pc = P @ np.asarray(R).T + t
    if conv == "gl":
        d = -pc[:, 2]
        x, y = pc[:, 0], -pc[:, 1]
    else:
        d = pc[:, 2]
        x, y = pc[:, 0], pc[:, 1]
    ds = np.where(np.abs(d) < 1e-9, 1e-9, d)
    u = x / ds * fx + cx - off
    v = y / ds * fy + cy - off
    return u, v, d
