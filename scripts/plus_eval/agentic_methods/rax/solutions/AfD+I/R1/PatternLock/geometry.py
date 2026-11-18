"""World localisation via robomme_pixel_to_position (preferred) or own depth back-projection."""
import numpy as np
import io_utils as iu

PREF_KEYS = ('position', 'world_position', 'position_m', 'world', 'point', 'xyz', 'world_point', 'p')

def _vec3(d):
    if not isinstance(d, dict):
        return None
    for k in PREF_KEYS:
        v = d.get(k)
        if isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
            return np.array(v, float)
        if isinstance(v, dict) and all(c in v for c in 'xyz'):
            return np.array([v['x'], v['y'], v['z']], float)
    for k, v in d.items():
        if k in ('camera_position', 'ray', 'direction', 'pixel', 'uv'):
            continue
        if isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
            return np.array(v, float)
    return None

def parse_points(meta, n):
    """Find the per-selection result list; return list of (xyz or None, spread or None)."""
    best = None
    for _, k, v in iu.walk(meta):
        if isinstance(v, list) and len(v) == n and all(isinstance(x, dict) for x in v):
            if sum(_vec3(x) is not None for x in v) >= 1:
                best = v; break
    if best is None:
        for _, k, v in iu.walk(meta):
            if isinstance(v, list) and len(v) == n and all(
                    isinstance(x, list) and len(x) == 3 and all(isinstance(c, (int, float)) for c in x) for x in v):
                return [(np.array(x, float), None) for x in v]
    if best is None and isinstance(meta, list) and len(meta) == n:
        best = meta
    if best is None:
        return None
    out = []
    for d in best:
        p = _vec3(d)
        valid = d.get('valid', True)
        spread = d.get('depth_spread_m')
        if p is None or valid is False or not np.all(np.isfinite(p)):
            out.append((None, spread))
        else:
            out.append((p, spread))
    return out

def pixels_to_world(ctx, step_id, camera, uvs, log, radius=1):
    """uvs: list of (u,v). Returns list of xyz-or-None. Batches of 32."""
    res = []
    for s in range(0, len(uvs), 32):
        chunk = uvs[s:s + 32]
        sel = [{'camera': camera, 'u': float(u), 'v': float(v)} for u, v in chunk]
        try:
            r = ctx.call('robomme_pixel_to_position', observation_step_id=int(step_id),
                         selections=sel, depth_window_radius=int(radius))
            meta = iu.meta_of(r)
            if s == 0:
                log.dump('pixel_to_position_reply.json', meta)
            pts = parse_points(meta, len(chunk))
            if pts is None:
                log('pixel_to_position: could not parse reply keys', list(meta.keys()) if isinstance(meta, dict) else type(meta))
                res.extend([None] * len(chunk))
            else:
                res.extend([p for p, _ in pts])
        except Exception as e:
            log('pixel_to_position failed', repr(e)[:300])
            res.extend([None] * len(chunk))
    return res

# ---------- fallback: own back-projection ----------
def quat_wxyz_to_R(q):
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z); w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def _intr(meta, camera):
    ci = iu.find_key(meta, ('camera_intrinsics',))
    if isinstance(ci, dict) and camera in ci:
        ci = ci[camera]
    if isinstance(ci, dict):
        if all(k in ci for k in ('fx', 'fy', 'cx', 'cy')):
            return ci['fx'], ci['fy'], ci['cx'], ci['cy']
        for k in ('K', 'matrix', 'intrinsic_matrix', 'k'):
            if k in ci:
                K = np.array(ci[k], float).reshape(3, 3)
                return K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if isinstance(ci, list):
        K = np.array(ci, float).reshape(3, 3)
        return K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return None

def _pose(meta, camera):
    cp = iu.find_key(meta, ('camera_poses',))
    if isinstance(cp, dict) and camera in cp:
        cp = cp[camera]
    if isinstance(cp, dict):
        p = cp.get('position_m', cp.get('position')) or cp.get('p') or cp.get('translation')
        q = cp.get('quaternion_wxyz') or cp.get('q') or cp.get('quaternion')
        if p is not None and q is not None:
            return np.array(p, float), quat_wxyz_to_R(q)
    return None

def backproject(meta, depth, camera, uvs, mpu=0.001):
    K = _intr(meta, camera); P = _pose(meta, camera)
    if K is None or P is None or depth is None:
        return [None] * len(uvs)
    fx, fy, cx, cy = K; t, R = P
    out = []
    for u, v in uvs:
        ui, vi = int(round(u)), int(round(v))
        win = depth[max(0, vi - 1):vi + 2, max(0, ui - 1):ui + 2].astype(float).ravel()
        win = win[win > 0]
        if len(win) == 0:
            out.append(None); continue
        d = np.median(win) * (mpu if depth.dtype != np.float32 and depth.dtype != np.float64 else 1.0)
        pc = np.array([(u - cx) / fx * d, -(v - cy) / fy * d, -d])
        out.append(R @ pc + t)
    return out
