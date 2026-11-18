"""Pure planning: pillar models and arc waypoints between adjacent disks (world frame, metres)."""
import numpy as np

CLEARANCE_M = 0.030        # pillar surface -> stick axis clearance (stick ~1 cm dia + tracking error)
MIN_LAT_M = 0.045          # minimum arc apex lateral offset
MAX_LAT_M = 0.13           # maximum arc apex lateral offset
DEFAULT_PILLAR_HALF_M = 0.025
WAYPOINT_SPACING_M = 0.008


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def pillar_model(points, A, B, front_dir, table_z, log=print):
    """points: (N,3) world points sampled around pillar between disks A,B (xyz).

    Returns dict(lat_front, lat_back, along_lo, along_hi, top_z, n_pts) relative to the
    midpoint M of A,B with u=(B-A)/|B-A| and n = front_dir (horizontal, perpendicular to u).
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    M = (A + B) / 2
    u = unit((B - A)[:2]); n = np.asarray(front_dir, float)[:2]
    half = np.linalg.norm((B - A)[:2]) / 2
    res = dict(lat_front=DEFAULT_PILLAR_HALF_M, lat_back=-DEFAULT_PILLAR_HALF_M,
               along_lo=-DEFAULT_PILLAR_HALF_M, along_hi=DEFAULT_PILLAR_HALF_M,
               top_z=None, n_pts=0, default=True)
    if points is None or len(points) == 0:
        return res
    P = np.asarray(points, float)
    rel = P[:, :2] - M[:2]
    lat = rel @ n
    along = rel @ u
    keep = (P[:, 2] > table_z + 0.015) & (np.abs(along) < 0.85 * half) & (np.abs(lat) < 0.08)
    if keep.sum() < 4:
        log(f"[plan] pillar: only {int(keep.sum())} points; default model")
        return res
    lat, along, z = lat[keep], along[keep], P[keep, 2]
    top = z.max()
    lat_front = float(np.percentile(lat, 98))
    along_lo = float(np.percentile(along, 2)); along_hi = float(np.percentile(along, 98))
    # The back face is hidden from the front camera. Conservative: assume a footprint at least
    # as deep as it is wide (square box, any yaw), and that the pillar straddles the disk line.
    width = float(np.clip(along_hi - along_lo, 0.02, 0.07))
    lat_back = min(float(np.percentile(lat, 2)), lat_front - width, -0.015)
    lat_front = max(lat_front, 0.015)
    res.update(lat_front=lat_front, lat_back=lat_back, along_lo=along_lo, along_hi=along_hi,
               top_z=float(top), n_pts=int(keep.sum()), default=False)
    # sanity: footprint must be plausible
    if lat_front - lat_back > 0.14 or lat_front - lat_back < 0.005:
        log(f"[plan] pillar footprint implausible {res}; default model")
        res.update(lat_front=DEFAULT_PILLAR_HALF_M, lat_back=-DEFAULT_PILLAR_HALF_M, default=True)
    return res


def arc_lateral(pm, side, half):
    """Semi-minor axis b of the half-ellipse so the arc clears the pillar on `side`."""
    s = 1.0 if side == 'front' else -1.0
    need = (pm['lat_front'] if s > 0 else -pm['lat_back']) + CLEARANCE_M
    w = max(abs(pm['along_lo']), abs(pm['along_hi']))
    ratio = min(w / max(half, 1e-6), 0.9)
    b = need / np.sqrt(1 - ratio ** 2)
    return float(np.clip(max(b, MIN_LAT_M), MIN_LAT_M, MAX_LAT_M))


def arc_points(A, B, front_dir, side, b, z_of_s=None, spacing=WAYPOINT_SPACING_M):
    """Half-ellipse from A to B bulging toward `side` with semi-minor b. Returns (K,3)."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    M = (A + B) / 2
    u = np.zeros(3); u[:2] = unit((B - A)[:2])
    n = np.zeros(3); n[:2] = np.asarray(front_dir, float)[:2]
    s = 1.0 if side == 'front' else -1.0
    a = np.linalg.norm((B - A)[:2]) / 2
    approx_len = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))) / 2
    K = max(12, int(np.ceil(approx_len / spacing)))
    th = np.linspace(0, np.pi, K + 1)
    pts = M[None, :] - np.cos(th)[:, None] * a * u[None, :] + s * np.sin(th)[:, None] * b * n[None, :]
    if z_of_s is not None:
        pts[:, 2] = [z_of_s(t / np.pi) for t in th]
    return pts


def front_direction(A, B, table_front_point):
    """Horizontal unit normal to AB pointing toward a table point known to be on the camera side."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    u = unit((B - A)[:2])
    n = np.array([-u[1], u[0]])
    M = (A + B) / 2
    if np.dot(np.asarray(table_front_point, float)[:2] - M[:2], n) < 0:
        n = -n
    return n
