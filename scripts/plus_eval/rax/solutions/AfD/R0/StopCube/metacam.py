"""Fallback back-projection from observation metadata (camera_intrinsics, camera_poses).

Documented convention: camera pose in the world frame, quaternion [w, x, y, z],
OpenGL camera (looks along -Z, +Y up), image row 0 at the top; depth is linear along
the optical axis. Key names are searched defensively; returns None when absent.
"""
import numpy as np
import mcp_io


def _quat_wxyz_to_R(q):
    w, x, y, z = [float(a) for a in q]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        return None
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _as_K(v):
    if isinstance(v, dict):
        fx = mcp_io.find_key(v, ["fx"], lambda a: isinstance(a, (int, float)))
        fy = mcp_io.find_key(v, ["fy"], lambda a: isinstance(a, (int, float)))
        cx = mcp_io.find_key(v, ["cx", "ppx"], lambda a: isinstance(a, (int, float)))
        cy = mcp_io.find_key(v, ["cy", "ppy"], lambda a: isinstance(a, (int, float)))
        if None not in (fx, fy, cx, cy):
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        m = mcp_io.find_key(v, ["matrix", "K", "intrinsic_matrix", "intrinsics", "camera_matrix"],
                            lambda a: isinstance(a, list))
        return _as_K(m) if m is not None else None
    if isinstance(v, list):
        a = np.array(v, float) if all(isinstance(r, (list, int, float)) for r in v) else None
        if a is None:
            return None
        if a.shape == (3, 3):
            return a
        if a.size == 9:
            return a.reshape(3, 3)
        if a.size == 4:   # fx fy cx cy
            return np.array([[a[0], 0, a[2]], [0, a[1], a[3]], [0, 0, 1.0]])
    return None


def parse(meta):
    """{camera: dict(K, R, t)} for cameras with both intrinsics and pose."""
    out = {}
    try:
        intr = mcp_io.find_key(meta, ["camera_intrinsics", "intrinsics"], lambda a: isinstance(a, dict))
        poses = mcp_io.find_key(meta, ["camera_poses", "camera_pose", "extrinsics"], lambda a: isinstance(a, dict))
        if not isinstance(intr, dict) or not isinstance(poses, dict):
            return out
        for cam in ("front", "wrist"):
            K = _as_K(intr.get(cam))
            p = poses.get(cam)
            if K is None or not isinstance(p, dict):
                continue
            t = mcp_io.find_key(p, ["position_m", "position", "translation", "p", "xyz"], lambda a: mcp_io.is_num_list(a, 3))
            q = mcp_io.find_key(p, ["quaternion_wxyz", "quaternion", "quat", "q", "orientation"],
                                lambda a: mcp_io.is_num_list(a, 4))
            if t is None or q is None:
                continue
            R = _quat_wxyz_to_R(q)
            if R is None:
                continue
            out[cam] = {"K": K, "R": R, "t": np.array(t, float)}
    except Exception:
        return {}
    return out


def backproject(cam, u, v, d):
    """World point for pixel (u, v) at optical-axis depth d (metres), OpenGL camera."""
    K = cam["K"]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    pc = np.array([x, -y, -d])
    return cam["R"] @ pc + cam["t"]
