"""World-frame localisation of front-camera pixels and path planning."""
import math

import numpy as np

from mcp_io import log, reply_meta, parse_points, find_key


# ---------------------------------------------------------------- back-projection
def tool_backproject(ctx, step_id, pixels, camera="front", radius=1):
    """pixels: list of (u, v) = (column, row).  Returns list of xyz arrays or None."""
    out = []
    for k in range(0, len(pixels), 32):
        chunk = pixels[k:k + 32]
        sels = [dict(camera=camera, u=float(u), v=float(v)) for (u, v) in chunk]
        try:
            res = ctx.call("robomme_pixel_to_position", observation_step_id=int(step_id),
                           selections=sels, depth_window_radius=int(radius))
            pts = parse_points(reply_meta(res), len(chunk))
            out.extend([None if p is None else np.array(p, float) for p, _ in pts])
        except Exception as e:  # noqa
            log("pixel_to_position failed:", repr(e)[:300])
            out.extend([None] * len(chunk))
    return out


def _quat_wxyz_to_R(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def parse_calibration(meta, camera="front"):
    """Return (K 3x3, R 3x3, t 3, convention) or None."""
    intr = find_key(meta, "camera_intrinsics")
    poses = find_key(meta, "camera_poses")
    if intr is None or poses is None:
        return None
    try:
        ci = intr.get(camera) if isinstance(intr, dict) else None
        if ci is None and isinstance(intr, list):
            ci = next((c for c in intr if isinstance(c, dict) and c.get("camera") == camera), None)
        if isinstance(ci, dict):
            for k in ("matrix", "K", "intrinsic_cv", "intrinsics", "k"):
                if k in ci:
                    ci = ci[k] if not isinstance(ci[k], dict) else ci[k]
                    break
        if isinstance(ci, dict) and "fx" in ci:
            K = np.array([[ci["fx"], 0, ci["cx"]], [0, ci["fy"], ci["cy"]], [0, 0, 1]], float)
        else:
            K = np.array(ci, float).reshape(3, 3)
        cp = poses.get(camera) if isinstance(poses, dict) else None
        if cp is None and isinstance(poses, list):
            cp = next((c for c in poses if isinstance(c, dict) and c.get("camera") == camera), None)
        pos = cp.get('position_m', cp.get("position")) or cp.get("p") or cp.get("translation")
        q = cp.get("quaternion_wxyz") or cp.get("q") or cp.get("quaternion")
        conv = str(cp.get("convention", poses.get("convention", "opengl") if isinstance(poses, dict) else "opengl")).lower()
        return K, _quat_wxyz_to_R(q), np.array(pos, float), conv
    except Exception as e:
        log("calibration parse failed:", repr(e)[:200])
        return None


def own_backproject(depth_m, calib, pixels, win=1):
    K, R, t, conv = calib
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = depth_m.shape
    out = []
    for (u, v) in pixels:
        ui, vi = int(round(u)), int(round(v))
        y0, y1, x0, x1 = max(0, vi - win), min(h, vi + win + 1), max(0, ui - win), min(w, ui + win + 1)
        patch = depth_m[y0:y1, x0:x1]
        patch = patch[patch > 0]
        if patch.size == 0:
            out.append(None)
            continue
        d = float(np.median(patch))
        x = (u - cx) / fx * d
        y = (v - cy) / fy * d
        if "gl" in conv:
            pc = np.array([x, -y, -d])
        else:
            pc = np.array([x, y, d])
        out.append(R @ pc + t)
    return out


# ---------------------------------------------------------------- rotations
def rotvec_to_R(rv):
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if abs(math.pi - th) < 1e-3:
        # near pi: axis from the largest diagonal
        M = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(M)))
        ax = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return ax * th
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return ax * th


# ---------------------------------------------------------------- planning
def detour_offset(Pi, Pj, side_vec, footprints, clear_r, c_min, c_max, extra_far=0.0):
    """Max lateral offset c of an elliptical detour from Pi to Pj (xy) on the side
    given by unit vector side_vec, so that every obstacle footprint point stays
    clear_r away from the ellipse (approximate, lateral)."""
    d = Pj - Pi
    L = float(np.linalg.norm(d))
    u = d / L
    M = (Pi + Pj) / 2
    c = c_min
    for q in footprints:
        a = float((q - M) @ u)
        b = float((q - M) @ side_vec) + extra_far
        if abs(a) >= 0.5 * L + clear_r * 0.5:
            continue
        if b + clear_r <= 0:
            continue
        f = math.sqrt(max(1.0 - (2 * a / L) ** 2, 0.25))
        c = max(c, (b + clear_r) / f)
    return min(c, c_max), c


def ellipse_path(Pi, Pj, side_vec, c, step):
    """Points (xy) along a half-ellipse from Pi to Pj bulging by c along side_vec,
    spaced about `step` metres."""
    d = Pj - Pi
    L = float(np.linalg.norm(d))
    u = d / L
    M = (Pi + Pj) / 2
    # dense sampling then resample by arc length
    th = np.linspace(0, math.pi, 400)
    pts = M[None, :] + (-0.5 * L * np.cos(th))[:, None] * u[None, :] + (c * np.sin(th))[:, None] * side_vec[None, :]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(math.ceil(s[-1] / step)) + 1)
    targets = np.linspace(0, s[-1], n)
    out = np.stack([np.interp(targets, s, pts[:, 0]), np.interp(targets, s, pts[:, 1])], 1)
    return out, float(s[-1])
