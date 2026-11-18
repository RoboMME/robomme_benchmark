"""Rotations and depth back-projection with conventions validated against the API."""
import math

import numpy as np


def rotvec_to_R(rv):
    rv = np.asarray(rv, dtype=np.float64)
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def R_to_quat(R):
    """Shepperd's method; returns (w, x, y, z) with w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S
    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def R_to_rotvec(R):
    q = R_to_quat(R)
    w = min(1.0, max(-1.0, q[0]))
    th = 2 * math.acos(w)
    s = math.sqrt(max(0.0, 1 - w * w))
    if s < 1e-9:
        return np.zeros(3)
    return q[1:] / s * th


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def quat_to_R(q, order="wxyz"):
    q = np.asarray(q, dtype=np.float64)
    if order == "xyzw":
        x, y, z, w = q
    else:
        w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def wrap(a, period=2 * math.pi):
    return (a + period / 2) % period - period / 2


def tool_yaw(R):
    """Yaw of the tool x axis projected on the world xy plane."""
    return math.atan2(R[1, 0], R[0, 0])


# ----------------------------------------------------------------------------- deprojection
def pose_candidates(pose):
    """List of (name, R_cam2world, t) candidates for a parsed camera pose."""
    out = []
    if pose is None:
        return out
    if pose.get("matrix") is not None:
        M = np.asarray(pose["matrix"], dtype=np.float64)
        out.append(("mat", M[:3, :3], M[:3, 3]))
        try:
            Mi = np.linalg.inv(M)
            out.append(("mat_inv", Mi[:3, :3], Mi[:3, 3]))
        except Exception:
            pass
        out.append(("matT", M[:3, :3].T, M[3, :3] if abs(M[3, 3] - 1) < 1e-6 and np.any(M[3, :3]) else M[:3, 3]))
        return out
    q, p = pose.get("quat"), pose.get("position")
    if q is None or p is None:
        return out
    orders = [pose.get("order")] if pose.get("order") else ["wxyz", "xyzw"]
    if pose.get("order"):
        orders = [pose["order"], "xyzw" if pose["order"] == "wxyz" else "wxyz"]
    for o in orders:
        R = quat_to_R(q, o)
        out.append(("q_" + o, R, np.asarray(p, dtype=np.float64)))
        out.append(("q_" + o + "_inv", R.T, -R.T @ np.asarray(p, dtype=np.float64)))
    return out


def deproject(depth, K, R, t, conv="gl", off=0.0, rows=None, cols=None):
    """World points for pixels; depth is linear along the optical axis (m)."""
    H, W = depth.shape
    if rows is None:
        vv, uu = np.mgrid[0:H, 0:W]
    else:
        vv, uu = np.asarray(rows), np.asarray(cols)
    d = depth[vv, uu]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xc = (uu + off - cx) / fx * d
    yc = (vv + off - cy) / fy * d
    if conv == "gl":
        pc = np.stack([xc, -yc, -d], axis=-1)
    else:
        pc = np.stack([xc, yc, d], axis=-1)
    return pc @ R.T + t


class Projector:
    """A chosen back-projection convention for one camera."""

    def __init__(self, K, R, t, conv, off, name=""):
        self.K, self.R, self.t, self.conv, self.off, self.name = K, R, t, conv, off, name

    def cloud(self, depth):
        return deproject(depth, self.K, self.R, self.t, self.conv, self.off)

    def project(self, P):
        """World points -> (u=col, v=row) pixel coords."""
        pc = (np.asarray(P) - self.t) @ self.R
        if self.conv == "gl":
            x, y, z = pc[..., 0], -pc[..., 1], -pc[..., 2]
        else:
            x, y, z = pc[..., 0], pc[..., 1], pc[..., 2]
        u = x / z * self.K[0, 0] + self.K[0, 2] - self.off
        v = y / z * self.K[1, 1] + self.K[1, 2] - self.off
        return u, v

    def cam_right_world(self):
        return self.R[:, 0]


def fit_projector(depth, K, pose, samples):
    """samples: list of (row, col, world_xyz) from the API. Returns (Projector, err_m)."""
    best, best_err = None, float("inf")
    if depth is None or K is None or pose is None:
        return None, best_err
    rows = np.array([s[0] for s in samples], dtype=int)
    cols = np.array([s[1] for s in samples], dtype=int)
    ref = np.array([s[2] for s in samples], dtype=np.float64)
    for name, R, t in pose_candidates(pose):
        for conv in ("gl", "cv"):
            for off in (0.0, 0.5):
                try:
                    P = deproject(depth, K, R, t, conv, off, rows, cols)
                    e = np.linalg.norm(P - ref, axis=1)
                    e = e[np.isfinite(e)]
                    if e.size < 4:
                        continue
                    err = float(np.median(e))
                    if err < best_err:
                        best_err = err
                        best = Projector(K, R, t, conv, off, "%s/%s/%.1f" % (name, conv, off))
                except Exception:
                    continue
    return best, best_err


def default_projector(K, pose):
    c = pose_candidates(pose)
    if not c or K is None:
        return None
    name, R, t = c[0]
    conv = "cv" if "cv" in str(pose.get("convention", "")).lower() else "gl"
    return Projector(K, R, t, conv, 0.0, name + "/default")
