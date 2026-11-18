"""Scene perception: table plane, cubes, bin, button (front and wrist cameras)."""
import math
import numpy as np

import vision as V
from mcpio import log

CUBE_COLORS = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'magenta']


def _valid(P):
    return ~np.isnan(P).any(-1)


def table_z(rb, obs, cam='front'):
    P = rb.dense(cam, obs)
    if P is not None:
        H = P.shape[0]
        z = P[H // 2:, :, 2]
        z = z[~np.isnan(z)]
        if len(z) > 100:
            lo, hi = np.percentile(z, 2), np.percentile(z, 98)
            hist, edges = np.histogram(z, bins=max(10, int((hi - lo) / 0.002) + 1), range=(lo, hi + 1e-6))
            i = int(np.argmax(hist))
            zz = z[(z >= edges[i] - 0.003) & (z <= edges[i + 1] + 0.003)]
            return float(np.median(zz))
    # sparse: sample lower-image pixels via the tool
    H, W = obs.rgb[cam].shape[:2]
    uvs = [(int(W * fu), int(H * fv)) for fv in (0.7, 0.8, 0.9) for fu in (0.15, 0.3, 0.5, 0.7, 0.85)]
    pts = [p for p in rb.points(cam, uvs, obs) if p is not None]
    if not pts:
        return None
    z = np.array([p[2] for p in pts])
    return float(np.median(z))


def blob_points(rb, obs, cam, b, max_pts=24):
    """3-D points of a blob's pixels (dense: all; sparse: a sample)."""
    P = rb.dense(cam, obs)
    if P is not None:
        Q = P[b['ys'], b['xs']]
        return Q[_valid(Q)]
    n = len(b['xs'])
    idx = np.linspace(0, n - 1, min(n, max_pts)).astype(int)
    # prefer interior pixels: erode-like by distance to centroid
    uvs = [(float(b['xs'][i]), float(b['ys'][i])) for i in idx]
    pts = [p for p in rb.points(cam, uvs, obs) if p is not None]
    return np.array(pts) if pts else np.zeros((0, 3))


def detect_cubes(rb, obs, tz, colors=None, cam='front', min_area=18, exclude=None):
    """Return list of cubes {color, u, v, area, pos(top-face centre xyz), top, n}."""
    rgb = obs.rgb[cam]
    hsv = V.to_hsv(rgb)
    P = rb.dense(cam, obs)
    out = []
    for c in (colors or CUBE_COLORS):
        m = V.color_mask(hsv, c)
        if P is not None and tz is not None:
            hz = P[..., 2] - tz
            m &= ~np.isnan(hz) & (hz > 0.003) & (hz < 0.10)
        for b in V.blobs(m, min_area):
            if b['area'] > 3000:
                continue
            Q = blob_points(rb, obs, cam, b)
            if len(Q) < 3:
                continue
            top = float(np.percentile(Q[:, 2], 90))
            if tz is not None and not (0.008 < top - tz < 0.10):
                continue
            T = Q[Q[:, 2] > top - 0.008]
            if len(T) < 2:
                T = Q
            xy = np.median(T[:, :2], 0)
            ext = float(np.max(np.ptp(Q[:, :2], 0)))
            pos = np.array([xy[0], xy[1], top])
            if exclude is not None and any(np.linalg.norm(pos[:2] - np.asarray(e)[:2]) < r for e, r in exclude):
                continue
            yaw = None
            if P is not None and len(T) >= 8:
                try:
                    yaw = V.min_area_angle_world(T[:, :2])
                except Exception:
                    yaw = None
            out.append(dict(color=c, u=b['u'], v=b['v'], area=b['area'], pos=pos, top=top, ext=ext,
                            blob=b, yaw=yaw))
    return out


def ray_plane(rb, obs, cam, uvs, z0):
    c = rb.camera(cam, obs)
    if c is None:
        return None
    uv = np.asarray(uvs, float)
    r = c.ray_world(uv[:, 0], uv[:, 1])
    t = (z0 - c.t[2]) / r[:, 2]
    return c.t[None, :] + t[:, None] * r


def detect_bin(rb, obs, tz, cam='front'):
    """Bin = black interior blob with a light rim. Returns dict(center xyz(rim top), half, rim_z) or None."""
    rgb = obs.rgb[cam]
    h, s, v = V.to_hsv(rgb)
    black = v < 0.09
    H, W = black.shape
    cands = []
    for b in V.blobs(black, 12):
        if b['area'] > 0.25 * H * W:
            continue
        m = np.zeros_like(black)
        m[b['ys'], b['xs']] = True
        ring = V.dilate(m, 3) & ~V.dilate(m, 1)
        ring_rim = ring & (v > 0.25) & (s < 0.55)
        frac = ring_rim.sum() / max(1, ring.sum())
        if frac < 0.45:
            continue
        ys, xs = np.nonzero(ring_rim)
        rb_ = {'xs': xs, 'ys': ys, 'u': xs.mean(), 'v': ys.mean(), 'area': len(xs)}
        R = blob_points(rb, obs, cam, rb_, max_pts=24)
        if len(R) < 3:
            continue
        rim_z = float(np.percentile(R[:, 2], 80))
        if tz is not None and not (0.01 < rim_z - tz < 0.25):
            continue
        # interior points (floor) should be below rim
        I = blob_points(rb, obs, cam, b, max_pts=12)
        if len(I) and np.median(I[:, 2]) > rim_z + 0.01:
            continue
        pts = ray_plane(rb, obs, cam, np.stack([b['xs'], b['ys']], 1), rim_z)
        if pts is not None and np.isfinite(pts).all():
            ctr = pts[:, :2].mean(0)
            half = float(np.percentile(np.abs(pts[:, :2] - ctr), 95, axis=0).max())
            method = 'rayplane'
        else:
            ctr = np.median(R[:, :2], 0)
            half = 0.04
            method = 'rim'
        score = b['area'] * frac
        cands.append(dict(center=np.array([ctr[0], ctr[1], rim_z]), half=half, rim_z=rim_z, area=b['area'],
                          frac=frac, u=b['u'], v=b['v'], score=score, method=method))
    if not cands:
        return None
    cands.sort(key=lambda c: -c['score'])
    return cands[0]


def detect_button(rb, obs, tz, cam='front', bin_info=None, avoid=()):
    """Button = compact, low-saturation raised object that is not the bin or the robot."""
    rgb = obs.rgb[cam]
    h, s, v = V.to_hsv(rgb)
    H, W = s.shape
    P = rb.dense(cam, obs)
    low = (s < 0.22) & (v > 0.12)
    if P is not None and tz is not None:
        hz = P[..., 2] - tz
        low &= ~np.isnan(hz) & (hz > 0.004) & (hz < 0.09)
    low = V.erode(low, 1)
    cands = []
    for b in V.blobs(low, 6):
        if b['area'] > 1500:
            continue
        if b['vmin'] <= 1 or b['umin'] <= 0 or b['umax'] >= W - 1:
            continue
        Q = blob_points(rb, obs, cam, b, max_pts=16)
        if len(Q) < 3:
            continue
        top = float(np.percentile(Q[:, 2], 90))
        if tz is not None and not (0.006 < top - tz < 0.09):
            continue
        ext = float(np.max(np.ptp(Q[:, :2], 0)))
        if ext > 0.14:
            continue
        xy = np.median(Q[:, :2], 0)
        if bin_info is not None and np.linalg.norm(xy - bin_info['center'][:2]) < bin_info['half'] + 0.05:
            continue
        if any(np.linalg.norm(xy - np.asarray(a)[:2]) < r for a, r in avoid):
            continue
        fill = b['area'] / float((b['umax'] - b['umin'] + 1) * (b['vmax'] - b['vmin'] + 1))
        score = fill * min(b['area'], 300) * (1.0 if 0.02 < ext < 0.10 else 0.3)
        cands.append(dict(pos=np.array([xy[0], xy[1], top]), top=top, ext=ext, area=b['area'], fill=fill,
                          u=b['u'], v=b['v'], score=score))
    if not cands:
        return None
    cands.sort(key=lambda c: -c['score'])
    log('button candidates', [(round(c['u']), round(c['v']), c['area'], round(c['ext'], 3), round(c['top'] - (tz or 0), 3), round(c['score'], 1)) for c in cands[:5]])
    return cands[0]
