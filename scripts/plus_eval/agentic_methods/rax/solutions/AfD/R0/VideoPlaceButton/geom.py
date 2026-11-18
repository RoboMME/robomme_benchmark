"""Geometry helpers: rotations, back-projection from calibration, yaw estimation."""
import math
import numpy as np


def rotvec_to_mat(r):
    r = np.asarray(r, float)
    th = float(np.linalg.norm(r))
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def quat_wxyz_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def wrap90(a):
    """Wrap an angle (rad) to [-pi/4, pi/4) (cube 4-fold symmetry)."""
    q = math.pi / 2
    return (a + q / 2) % q - q / 2


def gripper_yaw(rotvec):
    """Yaw (rad) of the gripper's local x axis projected on world xy; meaningful mod 90deg."""
    R = rotvec_to_mat(rotvec)
    v = R[:, 0] if math.hypot(R[0, 0], R[1, 0]) > 0.3 else R[:, 1]
    return math.atan2(v[1], v[0])


def pointing_down(rotvec):
    R = rotvec_to_mat(rotvec)
    return R[2, 2] < -0.8


def square_yaw(xy):
    """Yaw (rad, in [-pi/4, pi/4)) of a square from points on its face; None if unreliable."""
    xy = np.asarray(xy, float)
    if len(xy) < 12:
        return None, 0.0
    c = xy - np.median(xy, axis=0)
    areas = []
    angs = np.radians(np.arange(-45, 45, 1.0))
    for a in angs:
        ca, sa = math.cos(a), math.sin(a)
        u = c[:, 0] * ca + c[:, 1] * sa
        v = -c[:, 0] * sa + c[:, 1] * ca
        w = np.percentile(u, 97) - np.percentile(u, 3)
        h = np.percentile(v, 97) - np.percentile(v, 3)
        areas.append(w * h)
    areas = np.asarray(areas)
    i = int(np.argmin(areas))
    contrast = float((areas.max() - areas.min()) / max(areas.max(), 1e-9))
    return float(angs[i]), contrast


class Calib:
    """Own back-projection from packet calibration (validated against the tool before use)."""

    def __init__(self, K, pos, R, convention="opengl"):
        self.K = np.asarray(K, float)
        self.pos = np.asarray(pos, float)
        self.R = np.asarray(R, float)
        self.convention = convention

    def backproject(self, u, v, d):
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        x = (u - cx) / fx * d
        y = (v - cy) / fy * d
        if self.convention == "opengl":
            pc = np.stack([x, -y, -d * np.ones_like(x)], -1)
        else:
            pc = np.stack([x, y, d * np.ones_like(x)], -1)
        return pc @ self.R.T + self.pos


def parse_K(v):
    try:
        if isinstance(v, dict):
            for k in ("matrix", "K", "intrinsic", "intrinsics"):
                if k in v:
                    return parse_K(v[k])
            if all(k in v for k in ("fx", "fy", "cx", "cy")):
                return np.array([[v["fx"], 0, v["cx"]], [0, v["fy"], v["cy"]], [0, 0, 1]], float)
            return None
        a = np.asarray(v, float)
        if a.shape == (3, 3):
            return a
        if a.size == 9:
            return a.reshape(3, 3)
        if a.size == 4:
            return np.array([[a[0], 0, a[2]], [0, a[1], a[3]], [0, 0, 1]], float)
    except Exception:
        return None
    return None


def parse_pose(v):
    """-> (pos, R, convention) or None."""
    try:
        conv = "opengl"
        if isinstance(v, dict):
            c = str(v.get("convention", v.get("frame_convention", "opengl"))).lower()
            conv = "opencv" if "cv" in c else "opengl"
            for k in ("matrix", "pose", "cam2world", "camera_to_world", "T"):
                if k in v:
                    r = parse_pose(v[k])
                    if r:
                        return r[0], r[1], conv
            p = v.get('position_m', v.get("position", v.get("p", v.get("translation"))))
            q = v.get("quaternion_wxyz", v.get("q", v.get("quaternion")))
            if p is not None and q is not None:
                return np.asarray(p, float), quat_wxyz_to_mat(q), conv
            return None
        a = np.asarray(v, float)
        if a.size == 16:
            a = a.reshape(4, 4)
            return a[:3, 3], a[:3, :3], conv
        if a.size == 12:
            a = a.reshape(3, 4)
            return a[:, 3], a[:, :3], conv
    except Exception:
        return None
    return None
