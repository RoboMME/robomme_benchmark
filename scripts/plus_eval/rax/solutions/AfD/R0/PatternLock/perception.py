"""Perception helpers for PatternLock (pure numpy/scipy; no I/O).

Coordinates: image pixels are (u, v) = (column, row) in a 256x256 image.
World points are metres in the world frame.
"""
import math
import numpy as np

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover - fallback when scipy is missing
    _ndi = None


# ----------------------------------------------------------------- basics
def label(mask):
    """Connected components (8-connectivity). Returns (labels, n)."""
    mask = np.asarray(mask, bool)
    if _ndi is not None:
        return _ndi.label(mask, structure=np.ones((3, 3), int))
    lab = np.zeros(mask.shape, int)
    n = 0
    H, W = mask.shape
    for y0, x0 in zip(*np.nonzero(mask)):
        if lab[y0, x0]:
            continue
        n += 1
        stack = [(y0, x0)]
        lab[y0, x0] = n
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                        lab[yy, xx] = n
                        stack.append((yy, xx))
    return lab, n


def components(mask, min_size=1, max_size=10 ** 9):
    lab, n = label(mask)
    out = []
    if n == 0:
        return out
    ys, xs = np.nonzero(lab)
    ids = lab[ys, xs]
    cnt = np.bincount(ids, minlength=n + 1)
    sx = np.bincount(ids, weights=xs, minlength=n + 1)
    sy = np.bincount(ids, weights=ys, minlength=n + 1)
    for i in range(1, n + 1):
        c = int(cnt[i])
        if min_size <= c <= max_size:
            out.append((sx[i] / c, sy[i] / c, c))
    return out


def median_filter(img, size):
    if _ndi is not None:
        return _ndi.median_filter(img, size=size)
    # crude fallback: box mean
    k = size
    pad = np.pad(img, k // 2, mode='edge')
    c = np.cumsum(np.cumsum(pad, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    H, W = img.shape
    s = c[k:k + H, k:k + W] - c[0:H, k:k + W] - c[k:k + H, 0:W] + c[0:H, 0:W]
    return s / (k * k)


# ----------------------------------------------------------------- red
def red_mask(rgb):
    a = np.asarray(rgb).astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    return (R > 110) & (G < 75) & (B < 75) & (R - G > 80) & (np.abs(G - B) < 40)


def red_blobs(rgb, min_size=6):
    return components(red_mask(rgb), min_size=min_size)


# ----------------------------------------------------------------- pucks
def detect_pucks_rgb(rgb, min_size=6, max_size=400):
    """Bright, low-saturation discs relative to the local background."""
    a = np.asarray(rgb).astype(np.float32)
    g = a.mean(-1)
    mx = a.max(-1)
    mn = a.min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    bg = median_filter(g, 21)
    mask = (g - bg > 18) & (sat < 0.3)
    if _ndi is not None:
        mask = _ndi.binary_opening(mask, iterations=1)
    return components(mask, min_size=min_size, max_size=max_size)


def plane_residual_depth(depth, valid=None):
    """Fit 1/d = a*u + b*v + c robustly (a world plane seen by a pinhole camera)
    and return residual in metres (positive = closer to camera than the plane)."""
    d = np.asarray(depth, np.float64)
    H, W = d.shape
    vv, uu = np.mgrid[0:H, 0:W]
    ok = (d > 0.05) & np.isfinite(d)
    if valid is not None:
        ok &= valid
    inv = np.zeros_like(d)
    inv[ok] = 1.0 / d[ok]
    sel = ok.copy()
    coef = None
    for _ in range(6):
        A = np.stack([uu[sel], vv[sel], np.ones(sel.sum())], 1)
        if len(A) < 50:
            break
        coef, *_ = np.linalg.lstsq(A, inv[sel], rcond=None)
        pred = coef[0] * uu + coef[1] * vv + coef[2]
        with np.errstate(divide='ignore', invalid='ignore'):
            dp = np.where(pred > 1e-6, 1.0 / pred, 0)
        res = dp - d
        r = np.abs(res[sel])
        thr = max(0.002, 2.5 * np.median(r))
        sel = ok & (np.abs(res) < thr)
    if coef is None:
        return None
    pred = coef[0] * uu + coef[1] * vv + coef[2]
    with np.errstate(divide='ignore', invalid='ignore'):
        dp = np.where(pred > 1e-6, 1.0 / pred, 0)
    res = np.where(ok, dp - d, 0.0)
    return res


def detect_pucks_depth(depth, roi_mask=None, min_h=0.003, max_h=0.06, min_size=6, max_size=400):
    """Pucks are small blobs slightly above the dominant (table) plane.
    The plane is fitted on the lower 75 % of the image (table) to avoid the robot."""
    d = np.asarray(depth, np.float64)
    H, W = d.shape
    valid = np.zeros(d.shape, bool)
    valid[int(0.12 * H):, :] = True
    res = plane_residual_depth(d, valid)
    if res is None:
        return [], None
    mask = (res > min_h) & (res < max_h)
    if roi_mask is not None:
        mask &= roi_mask
    if _ndi is not None:
        mask = _ndi.binary_opening(mask, iterations=1)
    return components(mask, min_size=min_size, max_size=max_size), res


# ----------------------------------------------------------------- lattice
def _wrap(a, period):
    return (a + period / 2) % period - period / 2


def fit_lattice(pts, n=5, ref_theta=None, spacing_hint=None):
    """Fit an n x n square lattice to 2-D points (world xy, metres).

    Returns dict with keys: theta (rad, direction of axis 'i'), spacing, origin
    (xy of cell (0,0)), cells {(i,j): xy predicted}, assign {point_index: (i,j)},
    inliers, rms.  Axis i ~ (cos t, sin t), axis j ~ (-sin t, cos t).
    theta is chosen in (-45, 45] deg, or nearest to ref_theta modulo 90 deg.
    """
    P = np.asarray(pts, np.float64)
    if len(P) < 6:
        return None
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    s0 = float(np.median(nn)) if spacing_hint is None else float(spacing_hint)
    # direction histogram from near-neighbour vectors
    vec = []
    for a in range(len(P)):
        for b in range(len(P)):
            if a != b and D[a, b] < 1.3 * s0:
                vec.append(P[b] - P[a])
    vec = np.asarray(vec)
    if len(vec) < 4:
        return None
    ang = np.arctan2(vec[:, 1], vec[:, 0])
    z = np.exp(4j * ang).mean()
    theta = float(np.angle(z) / 4.0)
    if ref_theta is not None:
        k = round((ref_theta - theta) / (math.pi / 2))
        theta += k * math.pi / 2
    else:
        theta = _wrap(theta, math.pi / 2)
    best = None
    for it in range(3):
        u = np.array([math.cos(theta), math.sin(theta)])
        v = np.array([-math.sin(theta), math.cos(theta)])
        a = P @ u
        b = P @ v
        s = s0
        # refine spacing by projecting near-neighbour vectors
        pa = vec @ u
        pb = vec @ v
        near = np.concatenate([np.abs(pa[np.abs(pb) < 0.3 * s]), np.abs(pb[np.abs(pa) < 0.3 * s])])
        near = near[(near > 0.6 * s) & (near < 1.4 * s)]
        if len(near) >= 4:
            s = float(np.median(near))
        oa = float(np.angle(np.exp(2j * math.pi * a / s).mean()) / (2 * math.pi))
        ob = float(np.angle(np.exp(2j * math.pi * b / s).mean()) / (2 * math.pi))
        ia = np.round(a / s - oa).astype(int)
        ib = np.round(b / s - ob).astype(int)
        ra = a / s - oa - ia
        rb = b / s - ob - ib
        inl = (np.abs(ra) < 0.3) & (np.abs(rb) < 0.3)
        if inl.sum() < 4:
            return None
        # choose the n-window with most inliers on each axis
        def window(idx):
            lo, hi = idx.min(), idx.max()
            bestc, bestw = -1, lo
            for w in range(lo - n + 1, hi + 1):
                c = int(((idx >= w) & (idx < w + n)).sum())
                if c > bestc:
                    bestc, bestw = c, w
            return bestw
        wa = window(ia[inl])
        wb = window(ib[inl])
        inl &= (ia >= wa) & (ia < wa + n) & (ib >= wb) & (ib < wb + n)
        I = ia - wa
        J = ib - wb
        # least-squares refine: P = O + I*e1 + J*e2
        idx = np.nonzero(inl)[0]
        if len(idx) < 4:
            return None
        A = np.stack([np.ones(len(idx)), I[idx], J[idx]], 1)
        sol, *_ = np.linalg.lstsq(A, P[idx], rcond=None)
        O, e1, e2 = sol[0], sol[1], sol[2]
        pred = O + I[:, None] * e1 + J[:, None] * e2
        err = np.linalg.norm(pred - P, axis=1)
        rms = float(np.sqrt(np.mean(err[idx] ** 2)))
        best = dict(theta=theta, spacing=float((np.linalg.norm(e1) + np.linalg.norm(e2)) / 2),
                    origin=O, e1=e1, e2=e2, rms=rms,
                    assign={int(k): (int(I[k]), int(J[k])) for k in idx},
                    inliers=int(len(idx)))
        # update theta from the fitted e1 (keeps the branch near the previous one)
        t1 = math.atan2(e1[1], e1[0])
        theta = theta + _wrap(t1 - theta, 2 * math.pi)
        s0 = best['spacing']
    cells = {}
    for i in range(n):
        for j in range(n):
            cells[(i, j)] = best['origin'] + i * best['e1'] + j * best['e2']
    best['cells'] = cells
    return best


def relabel_to_reference(lat, ref_theta):
    """Return a mapping function idx->idx so that lattice `lat` axes are the 90-degree
    variant closest to ref_theta.  (fit_lattice with ref_theta already does this.)"""
    return lat


# ----------------------------------------------------------------- homography
def fit_homography(src, dst):
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    A = []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, x * X, y * X, X])
        A.append([0, 0, 0, -x, -y, -1, x * Y, y * Y, Y])
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    Hm = Vt[-1].reshape(3, 3)
    return Hm / Hm[2, 2]


def apply_homography(Hm, pts):
    pts = np.asarray(pts, np.float64)
    ph = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ Hm.T
    return ph[:, :2] / ph[:, 2:3]


# ----------------------------------------------------------------- events
def extract_events(frames_red, puck_px, max_dist_px, times, min_gap_s=3.0, absent_s=0.4):
    """frames_red: list over frames of red blob lists [(u,v,size)];
    puck_px: (N,2) pixel centres of the video grid pucks (index aligned with ids);
    times: per-frame timestamps (s).  Returns ordered list of (puck_id, onset_time, frame_idx)."""
    P = np.asarray(puck_px, np.float64)
    N = len(P)
    last_seen = {}
    last_onset = {}
    events = []
    for f, blobs in enumerate(frames_red):
        t = times[f]
        seen = {}
        for (u, v, sz) in blobs:
            d = np.hypot(P[:, 0] - u, P[:, 1] - v)
            k = int(np.argmin(d))
            if d[k] <= max_dist_px:
                seen[k] = seen.get(k, 0) + sz
        for k, sz in seen.items():
            new = False
            if k not in last_onset:
                new = True
            elif (t - last_onset[k] > min_gap_s) and (t - last_seen.get(k, -1e9) > absent_s):
                new = True
            if new:
                last_onset[k] = t
                events.append((k, t, f, sz))
            last_seen[k] = t
    events.sort(key=lambda e: (e[1], -e[3]))
    return events
