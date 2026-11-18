"""Rotations, camera calibration parsing and depth back-projection."""
import math

import numpy as np

from vus_io import find_key, log


# ----------------------------------------------------------------------------- rotations

def quat_wxyz_to_R(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotvec_to_R(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
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
        # near pi: axis from the diagonal
        A = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(A)))
        axis = A[:, i] / math.sqrt(max(A[i, i], 1e-12))
        axis /= np.linalg.norm(axis) or 1.0
        return axis * th
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return axis * th


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ----------------------------------------------------------------------------- calibration

def _as_K(v):
    try:
        if isinstance(v, dict):
            for k in ("K", "matrix", "intrinsic_matrix", "intrinsics", "camera_matrix", "k"):
                if k in v:
                    K = _as_K(v[k])
                    if K is not None:
                        return K
            if all(k in v for k in ("fx", "fy", "cx", "cy")):
                return np.array([[float(v["fx"]), 0, float(v["cx"])], [0, float(v["fy"]), float(v["cy"])], [0, 0, 1.0]])
            return None
        a = np.array(v, dtype=float)
        if a.shape == (3, 3):
            return a
        if a.size == 9:
            return a.reshape(3, 3)
        if a.size == 4:
            fx, fy, cx, cy = a.tolist()
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    except Exception:
        return None
    return None


def _as_pose(v):
    """-> (R 3x3, t 3, convention str or None, is_matrix_source)"""
    try:
        if isinstance(v, dict):
            conv = None
            for ck in ("convention", "camera_convention", "frame_convention", "axes"):
                if isinstance(v.get(ck), str):
                    conv = v[ck]
            for mk in ("matrix", "cam2world", "camera_to_world", "pose_matrix", "transform", "T", "extrinsic_cam2world"):
                if mk in v:
                    a = np.array(v[mk], dtype=float)
                    if a.size == 16:
                        a = a.reshape(4, 4)
                        return a[:3, :3], a[:3, 3], conv
            pos = None
            for pk in ("position_m", "position", "p", "translation", "pos", "xyz"):
                if pk in v:
                    pos = np.array(v[pk], dtype=float).reshape(3)
                    break
            q = None
            for qk in ("quaternion_wxyz", "quat_wxyz", "wxyz", "quaternion", "q", "orientation"):
                if qk in v:
                    q = np.array(v[qk], dtype=float).reshape(4)
                    break
            if q is None and "quaternion_xyzw" in v:
                x, y, z, w = np.array(v["quaternion_xyzw"], dtype=float).reshape(4)
                q = np.array([w, x, y, z])
            if pos is not None and q is not None:
                return quat_wxyz_to_R(q), pos, conv
            return None
        a = np.array(v, dtype=float)
        if a.size == 16:
            a = a.reshape(4, 4)
            return a[:3, :3], a[:3, 3], None
        if a.size == 7:
            return quat_wxyz_to_R(a[3:7]), a[:3], None
    except Exception:
        return None
    return None


CAM_ALIASES = {"front": ("front", "third", "base", "static", "agent", "scene"),
               "wrist": ("wrist", "hand", "gripper", "eye")}


def _per_camera(container, cam):
    if container is None:
        return None
    if isinstance(container, dict):
        if cam in container:
            return container[cam]
        for alias in CAM_ALIASES.get(cam, (cam,)):
            for k, v in container.items():
                if isinstance(k, str) and alias in k.lower():
                    return v
    if isinstance(container, list):
        for v in container:
            if isinstance(v, dict) and (v.get("camera") == cam or v.get("name") == cam):
                return v
    return None


def get_calibration(meta, cam):
    """-> dict(K, R, t, conv) or None"""
    intr = _per_camera(find_key(meta, "camera_intrinsics"), cam)
    pose = _per_camera(find_key(meta, "camera_poses"), cam)
    K = _as_K(intr) if intr is not None else None
    P = _as_pose(pose) if pose is not None else None
    if K is None or P is None:
        return None
    R, t, conv = P
    return {"K": K, "R": R, "t": t, "conv": conv}


# ----------------------------------------------------------------------------- projection

# Variants: (axis convention, pixel offset, depth kind, pose inverted)
VARIANTS = []
for _ax in ("gl", "cv"):
    for _off in (0.5, 0.0):
        for _dk in ("z", "ray"):
            for _inv in (False, True):
                VARIANTS.append((_ax, _off, _dk, _inv))
DEFAULT_VARIANT = ("gl", 0.5, "z", False)


def backproject(depth, calib, variant=DEFAULT_VARIANT, us=None, vs=None):
    """depth HxW metres (0 invalid) -> world points HxWx3 (or Nx3 for us/vs) and valid mask."""
    ax, off, dk, inv = variant
    K = calib["K"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if us is None:
        H, W = depth.shape
        vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
        d = depth.astype(np.float64)
    else:
        uu = np.asarray(us, dtype=float)
        vv = np.asarray(vs, dtype=float)
        d = np.asarray(depth, dtype=float)
    xn = (uu + off - cx) / fx
    yn = (vv + off - cy) / fy
    if dk == "ray":
        z = d / np.sqrt(xn * xn + yn * yn + 1.0)
    else:
        z = d
    if ax == "gl":
        pc = np.stack([xn * z, -yn * z, -z], axis=-1)
    else:
        pc = np.stack([xn * z, yn * z, z], axis=-1)
    R, t = calib["R"], calib["t"]
    if inv:
        Rw = R.T
        tw = -R.T @ t
    else:
        Rw, tw = R, t
    pw = pc @ Rw.T + tw
    valid = np.isfinite(d) & (d > 1e-4)
    return pw, valid


def project(points, calib, variant=DEFAULT_VARIANT):
    """world Nx3 -> pixel (u, v) arrays and camera depth (for checks)."""
    ax, off, dk, inv = variant
    R, t = calib["R"], calib["t"]
    if inv:
        Rw, tw = R.T, -R.T @ t
    else:
        Rw, tw = R, t
    pc = (np.asarray(points) - tw) @ Rw
    if ax == "gl":
        z = -pc[:, 2]
        x, y = pc[:, 0], -pc[:, 1]
    else:
        z = pc[:, 2]
        x, y = pc[:, 0], pc[:, 1]
    K = calib["K"]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * x / z + K[0, 2] - off
    v = K[1, 1] * y / z + K[1, 2] - off
    return u, v, z


def plane_score(depth, calib, variant):
    """Fraction of valid points on the dominant horizontal plane (tool-free sanity check)."""
    try:
        sub = depth[::4, ::4]
        H, W = depth.shape
        vv, uu = np.mgrid[0:H:4, 0:W:4]
        p, ok = backproject(sub.ravel(), calib, variant, us=uu.ravel(), vs=vv.ravel())
        z = p[:, 2][ok]
        if z.size < 50:
            return 0.0, None
        hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.004, 0.004))
        if hist.size == 0:
            return 0.0, None
        i = int(np.argmax(hist))
        frac = float(hist[max(0, i - 1):i + 2].sum() / z.size)
        zt = float(edges[i])
        cam_z = float((calib["t"] if not variant[3] else -calib["R"].T @ calib["t"])[2])
        if zt > cam_z:  # a table above the camera is a mirrored convention
            frac *= 0.5
        return frac, zt
    except Exception:
        return 0.0, None


def fit_variant(depth, calib, samples):
    """samples: list of (u, v, world xyz from the tool). Returns (variant, median error m)."""
    if not samples:
        return (DEFAULT_VARIANT, False), None
    us = np.array([s[0] for s in samples], dtype=float)
    vs = np.array([s[1] for s in samples], dtype=float)
    ref = np.array([s[2] for s in samples], dtype=float)
    best = (DEFAULT_VARIANT, None)
    H, W = depth.shape
    for swap in (False, True):
        for var in VARIANTS:
            uu, vv = (vs, us) if swap else (us, vs)
            ui = np.clip(np.round(uu).astype(int), 0, W - 1)
            vi = np.clip(np.round(vv).astype(int), 0, H - 1)
            d = depth[vi, ui]
            p, ok = backproject(d, calib, var, us=uu, vs=vv)
            if ok.sum() < max(3, len(samples) // 2):
                continue
            err = np.linalg.norm(p[ok] - ref[ok], axis=1)
            med = float(np.median(err))
            if best[1] is None or med < best[1]:
                best = ((var, swap), med)
    if best[1] is None:
        return (DEFAULT_VARIANT, False), None
    return best
