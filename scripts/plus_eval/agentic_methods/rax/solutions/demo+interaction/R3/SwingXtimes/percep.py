import numpy as np
from rlib import backproject, log

COLORS = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta", "pink", "white", "black"]


def hsv(rgb):
    a = rgb.astype(float) / 255.0
    mx = a.max(-1); mn = a.min(-1); d = mx - mn + 1e-9
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    s = np.where(mx > 0, (mx - mn) / (mx + 1e-9), 0)
    return h, s, mx


def color_mask(rgb, name):
    h, s, v = hsv(rgb)
    sat = (s > 0.6) & (v > 0.25)
    if name == "red":
        return sat & ((h < 15) | (h > 340))
    if name == "green":
        return sat & (h > 90) & (h < 160)
    if name == "blue":
        return sat & (h > 200) & (h < 260)
    if name == "yellow":
        return sat & (h > 45) & (h < 70)
    if name in ("purple", "magenta", "pink"):
        return sat & (h > 270) & (h < 335)
    if name == "orange":
        return sat & (h > 15) & (h < 40)
    if name == "cyan":
        return sat & (h > 165) & (h < 200)
    return sat


def clusters(points, radius=0.025, min_n=5):
    """Greedy xy clustering of Nx3 points. Returns list of arrays."""
    pts = points[np.all(np.isfinite(points), 1)]
    out = []
    used = np.zeros(len(pts), bool)
    order = np.argsort(-pts[:, 2])
    for i in order:
        if used[i]:
            continue
        c = pts[i, :2]
        for _ in range(3):
            m = (~used) & (np.linalg.norm(pts[:, :2] - c, axis=1) < radius)
            if m.sum() == 0:
                break
            c = pts[m, :2].mean(0)
        m = (~used) & (np.linalg.norm(pts[:, :2] - c, axis=1) < radius)
        used |= m
        if m.sum() >= min_n:
            out.append(pts[m])
    return out


def cloud(meta, imgs, cam):
    P = backproject(meta, imgs[(cam, "depth")], cam)
    return P, imgs[(cam, "rgb")]


def workspace_mask(P):
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    return np.isfinite(z) & (x > -0.45) & (x < 0.25) & (np.abs(y) < 0.4) & (z > -0.01) & (z < 0.2)


def find_cube(meta, imgs, color, cams=("wrist", "front"), exclude_gripper=None):
    best = None
    for cam in cams:
        P, rgb = cloud(meta, imgs, cam)
        m = color_mask(rgb, color) & workspace_mask(P) & (P[..., 2] > 0.005)
        if exclude_gripper is not None:
            m &= np.linalg.norm(P[..., :2] - exclude_gripper[:2], axis=-1) > 0.0
        cl = clusters(P[m], radius=0.03, min_n=4)
        if not cl:
            continue
        c = max(cl, key=len)
        top = np.percentile(c[:, 2], 90)
        tf = c[c[:, 2] > top - 0.008]
        cen = np.array([np.median(tf[:, 0]), np.median(tf[:, 1]), top])
        if best is None or len(c) > best[1]:
            best = (cen, len(c), cam)
    return best


def find_whites(meta, imgs, cams=("wrist", "front")):
    """Return list of (xy center, max z, n) for low-saturation bright blobs (targets and button)."""
    allp = []
    for cam in cams:
        P, rgb = cloud(meta, imgs, cam)
        h, s, v = hsv(rgb)
        m = (s < 0.25) & (v > 0.2) & workspace_mask(P) & (P[..., 2] > 0.003) & (P[..., 2] < 0.08)
        allp.append(P[m])
    pts = np.concatenate(allp, 0)
    cl = clusters(pts, radius=0.06, min_n=8)
    res = []
    for c in cl:
        res.append((np.median(c[:, :2], 0), float(np.percentile(c[:, 2], 95)), len(c)))
    return res


def find_scene(meta, imgs, robot_xy=None):
    whites = find_whites(meta, imgs)
    # exclude anything near the robot gripper (gripper is white/gray)
    if robot_xy is not None:
        whites = [w for w in whites if np.linalg.norm(w[0] - robot_xy) > 0.06]
    flat = [w for w in whites if w[1] < 0.02]
    tall = [w for w in whites if 0.02 <= w[1] < 0.07]
    flat.sort(key=lambda w: -w[2])
    targets = flat[:2]
    button = max(tall, key=lambda w: w[2]) if tall else None
    return targets, button, whites


def find_flat_discs(meta, imgs, robot_xy=None):
    """Geometry-only fallback: flat objects 4-14 mm above the table in the wrist view."""
    P, rgb = cloud(meta, imgs, "wrist")
    z = P[..., 2]
    m = workspace_mask(P) & (z > 0.004) & (z < 0.014)
    tall = workspace_mask(P) & (z > 0.02) & (z < 0.08)
    tc = [c[:, :2].mean(0) for c in clusters(P[tall], radius=0.04, min_n=8)]
    cl = clusters(P[m], radius=0.05, min_n=15)
    res = []
    for c in cl:
        xy = np.median(c[:, :2], 0)
        if any(np.linalg.norm(xy - t) < 0.05 for t in tc):
            continue
        if robot_xy is not None and np.linalg.norm(xy - robot_xy) < 0.06:
            continue
        res.append((xy, float(np.percentile(c[:, 2], 95)), len(c)))
    res.sort(key=lambda w: -w[2])
    return res


def flat_points(meta, imgs, cams=("front", "wrist")):
    """World points of flat target-height surfaces (4-15 mm above table) and their rgb."""
    pts = []; cols = []
    for cam in cams:
        if cam == "wrist" and meta["camera_poses"]["wrist"]["position_m"][2] < 0.07:
            continue
        P, rgb = cloud(meta, imgs, cam)
        z = P[..., 2]
        m = workspace_mask(P) & (z > 0.004) & (z < 0.015)
        pts.append(P[m]); cols.append(rgb[m])
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(pts, 0), np.concatenate(cols, 0)


def track_target(meta, imgs, prior, radius_est=None, max_jump=0.12, exclude=None):
    """Find the flat disc nearest `prior` (xy). Returns (xy, n, red_count) or None."""
    pts, cols = flat_points(meta, imgs)
    if len(pts) < 10:
        return None
    sel = np.linalg.norm(pts[:, :2] - prior, axis=1) < max_jump + 0.05
    pts, cols = pts[sel], cols[sel]
    if len(pts) < 10:
        return None
    idx = np.arange(len(pts))
    cl = []
    used = np.zeros(len(pts), bool)
    # clustering with index retention
    order = np.argsort(np.linalg.norm(pts[:, :2] - prior, axis=1))
    for i in order:
        if used[i]:
            continue
        c = pts[i, :2]
        for _ in range(4):
            m = (~used) & (np.linalg.norm(pts[:, :2] - c, axis=1) < 0.05)
            if m.sum() == 0:
                break
            c = np.median(pts[m, :2], 0)
        m = (~used) & (np.linalg.norm(pts[:, :2] - c, axis=1) < 0.05)
        used |= m
        if m.sum() >= 25:
            cl.append(m)
    best = None
    for m in cl:
        p = pts[m]
        xy = np.median(p[:, :2], 0)
        if exclude is not None and np.linalg.norm(xy - exclude) < 0.06:
            continue
        d = np.linalg.norm(xy - prior)
        if d > max_jump:
            continue
        # occlusion-robust centre: y from extent mid, x from the +x (camera-side) edge
        y0, y1 = np.percentile(p[:, 1], [3, 97])
        x1 = np.percentile(p[:, 0], 97)
        cen = xy.copy()
        cen[1] = 0.5 * (y0 + y1)
        if radius_est is not None:
            cen[0] = min(xy[0], x1 - radius_est) if (x1 - radius_est) < xy[0] else xy[0]
        h, s, v = hsv(cols[m][None])
        red = int((((h < 12) | (h > 345)) & (s > 0.55) & (v > 0.3)).sum())
        if best is None or d < best[3]:
            best = (cen, int(m.sum()), red, d)
    if best is None:
        return None
    return best[0], best[1], best[2]


def disc_radius(meta, imgs, xy):
    pts, _ = flat_points(meta, imgs)
    m = np.linalg.norm(pts[:, :2] - xy, axis=1) < 0.06
    if m.sum() < 20:
        return None
    y0, y1 = np.percentile(pts[m, 1], [3, 97])
    return 0.5 * (y1 - y0)
