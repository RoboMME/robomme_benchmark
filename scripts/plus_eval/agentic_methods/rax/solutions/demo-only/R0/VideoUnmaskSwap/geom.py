"""Camera geometry: parse intrinsics/poses in several plausible layouts, back-project depth.

Conventions are candidates; the controller selects the one that matches the
simulator's own robomme_pixel_to_position answers (self-check), so a wrong guess
here is detected live instead of silently corrupting positions.
"""
import math
import numpy as np


# ---------------------------------------------------------------- rotations
def quat_to_R(q, order='wxyz'):
    q = [float(v) for v in q]
    if order == 'wxyz':
        w, x, y, z = q
    else:
        x, y, z, w = q
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotvec_to_R(r):
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def R_to_rotvec(R):
    R = np.asarray(R, float)
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: use the diagonal
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(A), 0, None))
        i = int(np.argmax(k))
        if k[i] < 1e-9:
            return np.zeros(3)
        k = A[:, i] / k[i]
        k = k / np.linalg.norm(k)
        return k * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return v * th


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def T_from(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------- parsing
def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _as_matrix(x, n):
    try:
        a = np.array(x, float)
    except Exception:
        return None
    if a.shape == (n, n):
        return a
    if a.size == n * n and a.ndim == 1:
        return a.reshape(n, n)
    return None


def parse_intrinsics(obj):
    """Return 3x3 K from many layouts, else None."""
    if obj is None:
        return None
    K = _as_matrix(obj, 3)
    if K is not None:
        return K
    if isinstance(obj, dict):
        low = {str(k).lower(): v for k, v in obj.items()}
        if all(k in low for k in ('fx', 'fy', 'cx', 'cy')):
            return np.array([[float(low['fx']), 0, float(low['cx'])], [0, float(low['fy']), float(low['cy'])], [0, 0, 1.0]])
        for key in ('k', 'matrix', 'intrinsic', 'intrinsics', 'intrinsic_matrix', 'camera_matrix', 'intrinsic_cv', 'k_matrix'):
            if key in low:
                K = parse_intrinsics(low[key])
                if K is not None:
                    return K
        for v in obj.values():
            if isinstance(v, (dict, list)):
                K = parse_intrinsics(v)
                if K is not None:
                    return K
    return None


def parse_pose_candidates(obj):
    """Return list of (name, T 4x4 camera-to-world candidate)."""
    out = []
    if obj is None:
        return out
    M = _as_matrix(obj, 4)
    if M is not None:
        out.append(('m', M))
        try:
            out.append(('m_inv', np.linalg.inv(M)))
        except Exception:
            pass
        return out
    if isinstance(obj, dict):
        low = {str(k).lower(): v for k, v in obj.items()}
        pos = None
        for key in ('position_m', 'position', 'pos', 'p', 't', 'translation', 'xyz'):
            if key in low and isinstance(low[key], (list, tuple)) and len(low[key]) == 3:
                pos = np.array(low[key], float)
                break
        quat = None
        qorder = None
        for key, order in (('quaternion_wxyz', 'wxyz'), ('quat_wxyz', 'wxyz'), ('wxyz', 'wxyz'),
                           ('quaternion_xyzw', 'xyzw'), ('quat_xyzw', 'xyzw'), ('xyzw', 'xyzw'),
                           ('quaternion', 'wxyz'), ('quat', 'wxyz'), ('q', 'wxyz'), ('orientation', 'wxyz'),
                           ('rotation', 'wxyz')):
            if key in low and isinstance(low[key], (list, tuple)) and len(low[key]) == 4:
                quat = low[key]
                qorder = order
                break
        if pos is not None and quat is not None:
            out.append(('q_' + qorder, T_from(quat_to_R(quat, qorder), pos)))
            other = 'xyzw' if qorder == 'wxyz' else 'wxyz'
            out.append(('q_' + other, T_from(quat_to_R(quat, other), pos)))
            for n, T in list(out[-2:]):
                out.append((n + '_inv', np.linalg.inv(T)))
        for key in ('matrix', 'pose', 'cam2world', 'cam2world_gl', 'extrinsic', 'extrinsics', 'transform', 'model_matrix'):
            if key in low:
                for n, T in parse_pose_candidates(low[key]):
                    out.append((key + '_' + n, T))
        if not out:
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    out.extend(parse_pose_candidates(v))
    return out


def find_key(obj, names, depth=0):
    """Depth-first search for the first value under any of `names`."""
    if depth > 8:
        return None
    if isinstance(obj, dict):
        for n in names:
            if n in obj:
                return obj[n]
        for v in obj.values():
            r = find_key(v, names, depth + 1)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, names, depth + 1)
            if r is not None:
                return r
    return None


def per_camera(obj, cam):
    """Pick the entry for `cam` from a per-camera container (dict or list)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower().startswith(cam):
                return v
        for k, v in obj.items():
            if cam in str(k).lower():
                return v
        # maybe a list-of-dicts style inside
        if obj.get('camera') == cam:
            return obj
    if isinstance(obj, list):
        for v in obj:
            if isinstance(v, dict) and (str(v.get('camera', v.get('name', ''))).lower().startswith(cam)):
                return v
    return None


# ---------------------------------------------------------------- projection
class Camera:
    """A camera with intrinsics K and a cam-to-world transform T, under a convention.

    convention 'gl': camera looks along -Z, +Y up (OpenGL); 'cv': looks +Z, +Y down.
    """

    def __init__(self, K, T, convention='gl'):
        self.K = np.asarray(K, float)
        self.T = np.asarray(T, float)
        self.conv = convention

    def backproject(self, u, v, d):
        u = np.asarray(u, float); v = np.asarray(v, float); d = np.asarray(d, float)
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        x = (u - cx) / fx * d
        y = (v - cy) / fy * d
        if self.conv == 'gl':
            pc = np.stack([x, -y, -d], -1)
        else:
            pc = np.stack([x, y, d], -1)
        return pc @ self.T[:3, :3].T + self.T[:3, 3]

    def project(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        Ti = np.linalg.inv(self.T)
        pc = P @ Ti[:3, :3].T + Ti[:3, 3]
        if self.conv == 'gl':
            d = -pc[:, 2]; x = pc[:, 0]; y = -pc[:, 1]
        else:
            d = pc[:, 2]; x = pc[:, 0]; y = pc[:, 1]
        d = np.where(np.abs(d) < 1e-9, 1e-9, d)
        u = self.K[0, 0] * x / d + self.K[0, 2]
        v = self.K[1, 1] * y / d + self.K[1, 2]
        return np.stack([u, v], -1), d

    def ray_plane(self, u, v, z):
        """Intersect pixel rays with the world plane Z=z."""
        o = self.T[:3, 3]
        p1 = self.backproject(u, v, np.ones_like(np.asarray(u, float)))
        dirs = p1 - o
        dz = dirs[..., 2]
        dz = np.where(np.abs(dz) < 1e-9, -1e-9, dz)
        s = (z - o[2]) / dz
        return o + dirs * s[..., None]

    def axis_world(self, i):
        """Camera image axis i (0=right, 1=down) as a world direction."""
        R = self.T[:3, :3]
        if i == 0:
            return R[:, 0]
        return -R[:, 1] if self.conv == 'gl' else R[:, 1]


def camera_candidates(K, pose_obj):
    out = []
    for name, T in parse_pose_candidates(pose_obj):
        for conv in ('gl', 'cv'):
            out.append((name + '/' + conv, Camera(K, T, conv)))
    return out
