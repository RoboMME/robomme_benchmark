"""Pure perception functions for PatternLock (no I/O to the robot).

Grid model: buttons on a (possibly rotated, shifted, rescaled) rectangular lattice.
Index (i, j): i along the grid axis closest to world +x, j along the axis closest to world +y.
"""
import numpy as np

from geom import backproject, project, label

BTN_DZ_MIN, BTN_DZ_MAX = 0.007, 0.03   # button top above table: ~0.015 (ep1, ep2)
SLOW_SPEED = 0.0015      # m/frame; expert corners had >=8 slow frames, pass-throughs 0 (ep1, ep2 videos)
MIN_SLOW_FRAMES = 3


def table_z(P, valid):
    z = P[..., 2][valid]
    z = z[(z > -0.2) & (z < 0.2)]
    if len(z) == 0:
        return 0.0
    h, e = np.histogram(z, bins=200, range=(-0.05, 0.05))
    k = int(np.argmax(h))
    return float((e[k] + e[k + 1]) / 2)


def detect_blobs(depth_m, K, pose, min_px=8):
    """Raised blobs: list of dicts {xy, z, n}."""
    P = backproject(depth_m, K, pose)
    valid = depth_m > 0
    tz = table_z(P, valid)
    z = P[..., 2]
    mask = valid & (z > tz + BTN_DZ_MIN) & (z < tz + BTN_DZ_MAX)
    lab, n = label(mask)
    out = []
    for i in range(1, n + 1):
        s = lab == i
        k = int(s.sum())
        if k < min_px:
            continue
        pts = P[s]
        out.append({"xy": np.array([np.median(pts[:, 0]), np.median(pts[:, 1])]),
                    "z": float(np.median(pts[:, 2])), "n": k})
    return out, tz


def nn_spacing(pts):
    if len(pts) < 2:
        return None
    D = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    D[np.arange(len(pts)), np.arange(len(pts))] = 1e9
    return float(np.median(D.min(1)))


def keep_lattice_members(pts, s):
    """Drop isolated blobs (fewer than 2 neighbours near spacing s)."""
    D = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    cnt = ((D > 0.6 * s) & (D < 1.5 * s)).sum(1)
    return cnt >= 2


def merge_views(wrist, front):
    """Prefer full wrist blobs; fill gaps with bias-corrected front blobs."""
    W = [b for b in wrist]
    if W:
        med_n = np.median([b["n"] for b in W])
        W = [b for b in W if b["n"] >= 0.6 * med_n]
    F = list(front)
    pts_w = np.array([b["xy"] for b in W]) if W else np.zeros((0, 2))
    pts_f = np.array([b["xy"] for b in F]) if F else np.zeros((0, 2))
    if len(pts_w) == 0:
        return pts_f
    if len(pts_f) == 0:
        return pts_w
    s = nn_spacing(np.vstack([pts_w])) or nn_spacing(pts_f) or 0.08
    # bias front -> wrist from matched pairs
    diffs = []
    for p in pts_f:
        d = np.linalg.norm(pts_w - p, axis=1)
        k = int(np.argmin(d))
        if d[k] < 0.3 * s:
            diffs.append(pts_w[k] - p)
    bias = np.median(diffs, axis=0) if diffs else np.zeros(2)
    out = list(pts_w)
    for p in pts_f:
        q = p + bias
        if np.min(np.linalg.norm(pts_w - q, axis=1)) > 0.5 * s:
            out.append(q)
    return np.array(out)


def fit_grid(pts):
    """Fit lattice. Returns dict with theta, spacing, A (2x3 affine idx->xy), idx list, shape."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 4:
        return None
    s = nn_spacing(pts)
    keep = keep_lattice_members(pts, s)
    pts = pts[keep]
    if len(pts) < 4:
        return None
    s = nn_spacing(pts)
    # orientation (period 90 deg)
    D = pts[None] - pts[:, None]
    L = np.linalg.norm(D, axis=2)
    sel = (L > 0.7 * s) & (L < 1.3 * s)
    ang = np.arctan2(D[..., 1][sel], D[..., 0][sel])
    c = np.mean(np.exp(4j * ang))
    theta = float(np.angle(c) / 4)
    Rm = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
    q = pts @ Rm.T

    def clusters(v):
        order = np.argsort(v)
        lab = np.zeros(len(v), int)
        cur = 0
        for a, b in zip(order[:-1], order[1:]):
            if v[b] - v[a] > 0.45 * s:
                cur += 1
            lab[b] = cur
        lab[order[0]] = 0
        # renumber by gaps of spacing (robust to a missing middle row)
        cent = np.array([v[lab == k].mean() for k in range(cur + 1)])
        steps = np.round((cent - cent[0]) / s).astype(int)
        return steps[lab]

    I = clusters(q[:, 0])
    J = clusters(q[:, 1])
    X = np.stack([I, J, np.ones(len(I))], 1)
    A, *_ = np.linalg.lstsq(X, pts, rcond=None)  # 3x2
    resid = np.linalg.norm(X @ A - pts, axis=1)
    return {"theta": theta, "spacing": s, "A": A, "ni": int(I.max()) + 1, "nj": int(J.max()) + 1,
            "resid_max": float(resid.max()), "resid_med": float(np.median(resid)), "n": len(pts)}


def node_xy(grid, i, j):
    return np.array([i, j, 1.0]) @ grid["A"]


def all_nodes(grid):
    return {(i, j): node_xy(grid, i, j) for i in range(grid["ni"]) for j in range(grid["nj"])}


def nearest(nodes, xy):
    best, bd = None, 1e9
    for k, p in nodes.items():
        d = float(np.linalg.norm(p - xy))
        if d < bd:
            best, bd = k, d
    return best, bd


def rdp(points, eps):
    points = np.asarray(points)
    if len(points) < 3:
        return list(range(len(points)))
    a, b = points[0], points[-1]
    ab = b - a
    n = np.linalg.norm(ab)
    if n < 1e-9:
        d = np.linalg.norm(points - a, axis=1)
    else:
        d = np.abs(ab[0] * (points[:, 1] - a[1]) - ab[1] * (points[:, 0] - a[0])) / n
    k = int(np.argmax(d))
    if d[k] > eps:
        left = rdp(points[: k + 1], eps)
        right = rdp(points[k:], eps)
        return left[:-1] + [r + k for r in right]
    return [0, len(points) - 1]


def sequence_from_states(states, grid, default_offset, low_margin=0.015):
    """Visited nodes (incl. pass-through) along the low-z TCP path, and corner flags.

    Returns dict(seq, corners(set of positions in seq), offset, touch_z, low_frames) or None.
    """
    S = np.asarray(states, dtype=float)
    if len(S) < 5 or float(S[:, 2].max() - S[:, 2].min()) < 0.02:
        return None
    nodes = all_nodes(grid)
    s = grid["spacing"]
    z = S[:, 2]
    zmin = float(z.min())
    low_idx = np.nonzero(z < zmin + low_margin)[0]
    if len(low_idx) == 0:
        return None
    offset = np.asarray(default_offset, dtype=float)
    radius = 0.3 * s
    speed = np.r_[0.0, np.linalg.norm(np.diff(S[:, :2], axis=0), axis=1)]
    visits = []
    for _ in range(3):
        visits = []  # [node, best_d, best_idx, n_slow]
        for idx in low_idx:
            k, d = nearest(nodes, S[idx, :2] - offset)
            if d > radius:
                continue
            slow = int(speed[idx] < SLOW_SPEED)
            if visits and visits[-1][0] == k:
                visits[-1][3] += slow
                if d < visits[-1][1]:
                    visits[-1][1], visits[-1][2] = d, idx
            else:
                visits.append([k, d, idx, slow])
        if not visits:
            break
        diffs = np.array([S[v[2], :2] - nodes[v[0]] for v in visits])
        new = np.median(diffs, axis=0)
        if np.linalg.norm(new - default_offset) > 0.6 * s:
            break
        offset = new
    seq = [v[0] for v in visits]
    dwell_nodes = []
    for v in visits:
        if v[3] >= MIN_SLOW_FRAMES and (not dwell_nodes or dwell_nodes[-1] != v[0]):
            dwell_nodes.append(v[0])
    # corners: RDP on the low-z path
    path = S[low_idx, :2]
    kp = rdp(path, 0.25 * s)
    corner_nodes = []
    for k in kp:
        n_, d = nearest(nodes, path[k] - offset)
        if d < 0.45 * s:
            corner_nodes.append(n_)
    return {"seq": seq, "corner_nodes": corner_nodes, "dwell_nodes": dwell_nodes,
            "n_slow": [v[3] for v in visits], "offset": offset,
            "touch_z": float(np.median(z[low_idx])), "n_low": int(len(low_idx)),
            "visit_d": [round(v[1], 4) for v in visits]}


def red_mask(rgb):
    rgb = rgb.astype(int)
    return (rgb[..., 0] > 140) & (rgb[..., 1] < 90) & (rgb[..., 2] < 90) & (rgb[..., 0] - rgb[..., 1] > 80)


def node_pixels(nodes, top_z, K, pose):
    return {k: project(np.array([p[0], p[1], top_z]), K, pose) for k, p in nodes.items()}


def red_nodes(rgb, npix, radius=5, min_px=5):
    m = red_mask(rgb[..., :3])
    H, W = m.shape
    res = set()
    for k, (u, v) in npix.items():
        u0, v0 = int(round(u)), int(round(v))
        if not (0 <= u0 < W and 0 <= v0 < H):
            continue
        win = m[max(0, v0 - radius):v0 + radius + 1, max(0, u0 - radius):u0 + radius + 1]
        if int(win.sum()) >= min_px:
            res.add(k)
    return res


def sequence_from_red(rgb_frames, npix):
    seq, prev = [], set()
    for rgb in rgb_frames:
        cur = red_nodes(rgb, npix)
        for k in sorted(cur - prev):
            if k not in seq:
                seq.append(k)
        prev = cur
    return seq


def plan_path(corners, grid):
    """Waypoints [(xy, kind, node)] visiting corner nodes in order. Segments whose straight line
    would cross an intermediate lattice node get a parallel detour halfway between lattice lines."""
    s = grid["spacing"]
    nodes = all_nodes(grid)
    wps = []
    for n, c in enumerate(corners):
        p = node_xy(grid, *c)
        if n == 0:
            wps.append((p, "corner", c))
            continue
        a = corners[n - 1]
        di, dj = c[0] - a[0], c[1] - a[1]
        g = int(np.gcd(abs(di), abs(dj)))
        if g > 1:
            A = node_xy(grid, *a)
            B = p
            ab = B - A
            L = np.linalg.norm(ab)
            perp = np.array([-ab[1], ab[0]]) / L
            # lattice lines parallel to (di,dj)/g are spaced s / |(di,dj)/g|
            h = 0.5 * s / np.hypot(di / g, dj / g)
            best = None
            for sign in (1, -1):
                v1, v2 = A + sign * h * perp, B + sign * h * perp
                poly = [A, v1, v2, B]
                clear = min(seg_clearance(poly, nodes, exclude={a, c}))
                if best is None or clear > best[0]:
                    best = (clear, v1, v2)
            wps.append((best[1], "via", None))
            wps.append((best[2], "via", None))
        wps.append((p, "corner", c))
    return wps


def seg_clearance(poly, nodes, exclude):
    out = []
    for k, q in nodes.items():
        if k in exclude:
            continue
        dmin = 1e9
        for P0, P1 in zip(poly[:-1], poly[1:]):
            d = P1 - P0
            t = np.clip(np.dot(q - P0, d) / max(np.dot(d, d), 1e-12), 0, 1)
            dmin = min(dmin, float(np.linalg.norm(P0 + t * d - q)))
        out.append(dmin)
    return out if out else [1e9]


def path_min_clearance(wps, nodes, corners):
    """Min distance of the whole planned polyline to nodes other than the two segment-end corners."""
    worst = 1e9
    pts = [w[0] for w in wps]
    kinds = [w[2] for w in wps]
    # walk segments; track last and next corner
    ci = 0
    for k in range(len(pts) - 1):
        if kinds[k] is not None:
            last = kinds[k]
        nxt = next(kk for kk in kinds[k + 1:] if kk is not None)
        cl = seg_clearance([pts[k], pts[k + 1]], nodes, exclude={last, nxt})
        worst = min(worst, min(cl))
    return worst
