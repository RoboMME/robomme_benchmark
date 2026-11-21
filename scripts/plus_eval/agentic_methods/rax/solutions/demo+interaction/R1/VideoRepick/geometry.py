"""Camera geometry: parse calibration, back-project/project, verify convention against
robomme_pixel_to_position, and EE<->wrist-camera transfer for video hold frames."""
import numpy as np


def quat_wxyz_to_R(q):
    w, x, y, z = [float(a) for a in q]
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def axang_to_R(v):
    v = np.asarray(v, float)
    th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def R_to_axang(R):
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c))
    th = np.arccos(c)
    if th < 1e-6:
        return np.zeros(3)
    if np.pi - th < 1e-3:
        # near pi: axis from largest diagonal
        A = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(A)))
        k = A[:, i] / np.sqrt(max(A[i, i], 1e-12))
        return k * th
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return k * th


def _as_K(v):
    if v is None:
        return None
    if isinstance(v, dict):
        for k in ('matrix', 'K', 'intrinsic', 'intrinsics', 'camera_matrix', 'k'):
            if k in v:
                K = _as_K(v[k])
                if K is not None:
                    return K
        keys = {k.lower(): v[k] for k in v}
        if 'fx' in keys and 'fy' in keys:
            cx = keys.get('cx', keys.get('ppx', 127.5))
            cy = keys.get('cy', keys.get('ppy', 127.5))
            return np.array([[float(keys['fx']), 0, float(cx)], [0, float(keys['fy']), float(cy)], [0, 0, 1.0]])
        return None
    try:
        a = np.array(v, dtype=float)
    except Exception:
        return None
    if a.size == 9:
        return a.reshape(3, 3)
    if a.size == 4:  # fx fy cx cy
        return np.array([[a[0], 0, a[2]], [0, a[1], a[3]], [0, 0, 1.0]])
    if a.shape == (3, 4):
        return a[:, :3]
    return None


def _as_pose(v):
    """-> (R_world_cam, t) or None."""
    if v is None:
        return None
    if isinstance(v, dict):
        pos = None
        for k in ('position_m', 'position', 'p', 'translation', 'pos', 'xyz'):
            if k in v:
                pos = v[k]
                break
        q = None
        for k in ('quaternion_wxyz', 'quaternion', 'wxyz', 'q', 'quat'):
            if k in v:
                q = v[k]
                break
        if pos is not None and q is not None:
            try:
                return quat_wxyz_to_R(q), np.array(pos, float)[:3]
            except Exception:
                return None
        for k in ('matrix', 'pose', 'T', 'cam2world', 'extrinsic'):
            if k in v:
                r = _as_pose(v[k])
                if r is not None:
                    return r
        return None
    try:
        a = np.array(v, dtype=float)
    except Exception:
        return None
    if a.size == 16:
        a = a.reshape(4, 4)
        return a[:3, :3], a[:3, 3]
    if a.size == 7:
        return quat_wxyz_to_R(a[3:7]), a[:3]
    return None


def _per_camera(obj, cam):
    if isinstance(obj, dict):
        for k in obj:
            if str(k).lower().startswith(cam):
                return obj[k]
    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict) and str(it.get('camera', it.get('name', ''))).lower().startswith(cam):
                return it
    return None


def parse_calibration(meta, find_key):
    """Return {cam: {'K':K, 'R':R, 't':t}} for cams with both available."""
    out = {}
    intr = find_key(meta, 'camera_intrinsics')
    poses = find_key(meta, 'camera_poses')
    for cam in ('front', 'wrist'):
        K = _as_K(_per_camera(intr, cam)) if intr is not None else None
        P = _as_pose(_per_camera(poses, cam)) if poses is not None else None
        if K is not None and P is not None:
            out[cam] = {'K': K, 'R': P[0], 't': P[1]}
    return out


# conventions: (name, flip) ; flip=True => OpenGL (cam looks -Z, +Y up)
CONVENTIONS = [('opengl', True, 0.0), ('opengl', True, 0.5), ('opencv', False, 0.0), ('opencv', False, 0.5)]


def backproject(cal, u, v, d, conv=('opengl', True, 0.0)):
    K, R, t = cal['K'], cal['R'], cal['t']
    _, flip, off = conv
    x = (u + off - K[0, 2]) / K[0, 0] * d
    y = (v + off - K[1, 2]) / K[1, 1] * d
    if flip:
        pc = np.array([x, -y, -d])
    else:
        pc = np.array([x, y, d])
    return R @ pc + t


def project(cal, pw, conv=('opengl', True, 0.0)):
    K, R, t = cal['K'], cal['R'], cal['t']
    _, flip, off = conv
    pc = R.T @ (np.asarray(pw, float) - t)
    if flip:
        x, y, z = pc[0], -pc[1], -pc[2]
    else:
        x, y, z = pc
    if z <= 1e-6:
        return None
    u = K[0, 0] * x / z + K[0, 2] - off
    v = K[1, 1] * y / z + K[1, 2] - off
    return np.array([u, v]), z


def choose_convention(cal, samples):
    """samples: list of (u, v, depth_m, world_xyz). Returns (conv, median_err)."""
    best = (None, 1e9)
    for conv in CONVENTIONS:
        errs = []
        for u, v, d, pw in samples:
            if d is None or d <= 0:
                continue
            errs.append(np.linalg.norm(backproject(cal, u, v, d, conv) - np.asarray(pw)))
        if len(errs) >= 2:
            e = float(np.median(errs))
            if e < best[1]:
                best = (conv, e)
    return best


def pose_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def ee_T(state):
    return pose_T(axang_to_R(state[3:6]), np.asarray(state[:3], float))


def transfer_camera(cal_exec, state_exec, state_other):
    """Wrist camera calibration at another EE pose, assuming a rigid hand-camera mount."""
    T_wc = pose_T(cal_exec['R'], cal_exec['t'])
    T_ee_c = np.linalg.inv(ee_T(state_exec)) @ T_wc
    T2 = ee_T(state_other) @ T_ee_c
    return {'K': cal_exec['K'], 'R': T2[:3, :3], 't': T2[:3, 3]}
