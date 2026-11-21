"""Perception: point clouds, table plane, cube segmentation, highlight scoring."""
import math
import numpy as np

from geom import cam_points, min_area_rect_angle, project

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None

# Tunables (none tested live; chosen from typical ManiSkill tabletop scales)
OBJ_MIN_H = 0.007        # m above table to count as object
OBJ_MAX_H = 0.25         # ignore higher points (arm)
EDGE_JUMP = 0.012        # m 3-D jump between neighbouring pixels = discontinuity
CUBE_MIN_SIDE = 0.012
CUBE_MAX_SIDE = 0.09
CUBE_MAX_H = 0.10
MIN_PIX = 8


def label(mask):
    if ndi is not None:
        lab, n = ndi.label(mask)
        return lab, n
    # fallback BFS
    H, W = mask.shape
    lab = np.zeros((H, W), dtype=np.int32)
    n = 0
    for y in range(H):
        for x in range(W):
            if mask[y, x] and lab[y, x] == 0:
                n += 1
                stack = [(y, x)]
                lab[y, x] = n
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        yy, xx = cy + dy, cx + dx
                        if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and lab[yy, xx] == 0:
                            lab[yy, xx] = n
                            stack.append((yy, xx))
    return lab, n


def dilate(mask, r):
    if r <= 0:
        return mask.copy()
    if ndi is not None:
        return ndi.binary_dilation(mask, iterations=r)
    m = mask.copy()
    for _ in range(r):
        m2 = m.copy()
        m2[1:] |= m[:-1]
        m2[:-1] |= m[1:]
        m2[:, 1:] |= m[:, :-1]
        m2[:, :-1] |= m[:, 1:]
        m = m2
    return m


def world_cloud(depth_m, cal, conv):
    K, R, t = cal["K"], cal["R"], cal["t"]
    P = cam_points(depth_m, K, R, t, conv)
    valid = np.isfinite(depth_m) & (depth_m > 0.01) & (depth_m < 5.0)
    return P, valid


def estimate_table(P, valid):
    """Height of the dominant horizontal surface (mode of z)."""
    z = P[..., 2][valid]
    if z.size < 50:
        return None
    lo, hi = np.percentile(z, [1, 99])
    bins = np.arange(lo - 0.005, hi + 0.005, 0.004)
    if len(bins) < 3:
        return float(np.median(z))
    h, e = np.histogram(z, bins=bins)
    i = int(np.argmax(h))
    # prefer the highest horizontal surface holding a substantial share of points (table above floor)
    hs = np.convolve(h, [1, 1, 1], mode="same")
    big = [j for j in range(len(h)) if hs[j] >= 0.15 * z.size and hs[j] >= max(hs[max(0, j - 1):j + 2])]
    if big:
        i = max(big)
        # use the raw-bin max within the 3-bin window
        lo_j, hi_j = max(0, i - 1), min(len(h) - 1, i + 1)
        i = lo_j + int(np.argmax(h[lo_j:hi_j + 1]))
    sel = z[(z >= e[i] - 0.004) & (z <= e[i + 1] + 0.004)]
    return float(np.median(sel)), float(h[i]) / z.size


def plane_horizontality(P, valid, z_t):
    """Fraction of valid points within 6 mm of z_t (high = table plane horizontal)."""
    z = P[..., 2][valid]
    if z.size == 0 or z_t is None:
        return 0.0
    return float(np.mean(np.abs(z - z_t) < 0.006))


def discontinuity(P, m):
    """Boolean map of pixels with a 3-D jump to a 4-neighbour in m."""
    H, W, _ = P.shape
    bad = np.zeros((H, W), dtype=bool)
    for dy, dx in ((0, 1), (1, 0)):
        a = P[: H - dy, : W - dx]
        b = P[dy:, dx:]
        ma = m[: H - dy, : W - dx]
        mb = m[dy:, dx:]
        d = np.linalg.norm(a - b, axis=-1)
        j = (d > EDGE_JUMP) & ma & mb
        bad[: H - dy, : W - dx] |= j
        bad[dy:, dx:] |= j
    return bad


def graph_label(P, m, cam_pos=None):
    """Connected components over 4-neighbour edges whose 3-D jump is small (range-scaled)."""
    H, W, _ = P.shape
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except Exception:
        core = m & ~discontinuity(P, m)
        return label(core)
    idx = -np.ones((H, W), dtype=np.int64)
    ids = np.nonzero(m.ravel())[0]
    idx.ravel()[ids] = np.arange(len(ids))
    if len(ids) == 0:
        return np.zeros((H, W), dtype=np.int32), 0
    rows, cols = [], []
    for dy, dx in ((0, 1), (1, 0)):
        a = P[: H - dy, : W - dx]
        b = P[dy:, dx:]
        ma = m[: H - dy, : W - dx] & m[dy:, dx:]
        d = np.linalg.norm(a - b, axis=-1)
        if cam_pos is not None:
            rng = np.linalg.norm(a - np.asarray(cam_pos), axis=-1)
        else:
            rng = 0.8
        thr = EDGE_JUMP_BASE + EDGE_JUMP_SCALE * rng
        ok = ma & (d < thr)
        ia = idx[: H - dy, : W - dx][ok]
        ib = idx[dy:, dx:][ok]
        rows.append(ia)
        cols.append(ib)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    g = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(ids), len(ids)))
    n, labels = connected_components(g, directed=False)
    lab = np.zeros((H, W), dtype=np.int32)
    lab.ravel()[ids] = labels + 1
    return lab, n


EDGE_JUMP_BASE = 0.010
EDGE_JUMP_SCALE = 0.010


def split_1d(x, side):
    """Split 1-D projections into k in 2..4 groups whose lengths best match `side`.
    Returns list of (lo, hi) intervals."""
    xs = np.sort(np.asarray(x, dtype=float))
    best = None
    for k in (2, 3, 4):
        if len(xs) < 4 * k:
            break
        # init: equal quantiles, then Lloyd iterations
        cen = np.quantile(xs, (np.arange(k) + 0.5) / k)
        for _ in range(20):
            lab = np.argmin(np.abs(xs[:, None] - cen[None, :]), axis=1)
            newc = np.array([xs[lab == i].mean() if np.any(lab == i) else cen[i] for i in range(k)])
            if np.allclose(newc, cen):
                break
            cen = newc
        lab = np.argmin(np.abs(xs[:, None] - cen[None, :]), axis=1)
        ivs = []
        cost = 0.0
        for i in range(k):
            g = xs[lab == i]
            if len(g) < 3:
                cost += side
                continue
            lo, hi = np.percentile(g, [2, 98])
            ivs.append((float(lo), float(hi)))
            cost += abs((hi - lo) - side)
        cost /= k
        if best is None or cost < best[0]:
            best = (cost, ivs)
    if best is None:
        return [None]
    # widen intervals to the midpoints between groups
    ivs = sorted(best[1])
    out = []
    for i, (lo, hi) in enumerate(ivs):
        l2 = lo if i == 0 else 0.5 * (ivs[i - 1][1] + lo)
        h2 = hi if i == len(ivs) - 1 else 0.5 * (hi + ivs[i + 1][0])
        out.append((l2, h2))
    return out


def rect_center(xy, ang):
    ca, sa = math.cos(ang), math.sin(ang)
    c0 = xy.mean(0)
    c = xy - c0
    px = c[:, 0] * ca + c[:, 1] * sa
    py = -c[:, 0] * sa + c[:, 1] * ca
    lx, hx = np.percentile(px, [2, 98])
    ly, hy = np.percentile(py, [2, 98])
    mx, my = (lx + hx) / 2, (ly + hy) / 2
    return c0 + np.array([mx * ca - my * sa, mx * sa + my * ca]), (hx - lx, hy - ly), (px, py)


def segment_cubes(P, valid, z_t, rgb=None, xy_bounds=None, cam_pos=None, min_pix=MIN_PIX):
    """Return list of cube dicts: center(3), xy, yaw, side, h, mask (pixel), npix, color."""
    z = P[..., 2]
    m = valid & (z > z_t + OBJ_MIN_H) & (z < z_t + OBJ_MAX_H)
    if xy_bounds is not None:
        (x0, x1), (y0, y1) = xy_bounds
        m &= (P[..., 0] > x0) & (P[..., 0] < x1) & (P[..., 1] > y0) & (P[..., 1] < y1)
    lab, n = graph_label(P, m, cam_pos)
    cubes = []
    rejected = []
    for i in range(1, n + 1):
        pm = lab == i
        npx = int(pm.sum())
        if npx < min_pix:
            continue
        pts = P[pm]
        ztop = float(np.percentile(pts[:, 2], 95))
        hgt = ztop - z_t
        zmax = float(pts[:, 2].max()) - z_t
        top = pts[pts[:, 2] > ztop - 0.007]
        # footprint from all object points above the bottom rim (sides project onto the footprint edges)
        foot = pts[pts[:, 2] > z_t + 0.004]
        if len(foot) >= 8:
            top_xy_src = top
            top = np.concatenate([foot[:, :2], np.zeros((len(foot), 1)) + ztop], axis=1)
        if len(top) < min(5, min_pix):
            rejected.append(("few_top", npx, hgt))
            continue
        ang, (a, b) = min_area_rect_angle(top[:, :2])
        info = dict(npix=npx, h=hgt, a=a, b=b)
        if hgt > CUBE_MAX_H or zmax > CUBE_MAX_H + 0.03:
            rejected.append(("tall", info))
            continue
        if hgt < CUBE_MIN_SIDE * 0.8:
            rejected.append(("flat", info))
            continue
        long_s, short_s = max(a, b), min(a, b)
        if short_s < CUBE_MIN_SIDE * 0.6 or long_s > 2.2 * CUBE_MAX_SIDE:
            rejected.append(("size", info))
            continue
        ctr, sides, (px, py) = rect_center(top[:, :2], ang)
        # split merged cubes along the long side
        k = 1
        # cubes are as tall as they are wide: the visible height is the side estimate
        side_est = hgt if CUBE_MIN_SIDE < hgt < CUBE_MAX_H else max(short_s, CUBE_MIN_SIDE)
        if long_s > 1.55 * side_est and long_s > 1.4 * short_s:
            k = int(round(long_s / side_est))
            k = max(1, min(k, 4))
        if long_s > CUBE_MAX_SIDE and k == 1:
            rejected.append(("long", info))
            continue
        pix_y, pix_x = np.nonzero(pm)
        pix_pts = P[pix_y, pix_x]
        ca, sa = math.cos(ang), math.sin(ang)
        axis = np.array([ca, sa]) if sides[0] >= sides[1] else np.array([-sa, ca])
        parts = [None]
        if k > 1:
            proj_top = (top[:, :2] - ctr) @ axis
            parts = split_1d(proj_top, side_est)
        for j, part in enumerate(parts):
            if part is None:
                c_xy = ctr
                sub = np.ones(len(pix_x), dtype=bool)
            else:
                lo_e, hi_e = part
                proj_top = (top[:, :2] - ctr) @ axis
                selt = (proj_top >= lo_e) & (proj_top <= hi_e)
                if selt.sum() < 3:
                    continue
                tt = top[selt][:, :2]
                ang2, _ = min_area_rect_angle(tt)
                c_xy, _, _ = rect_center(tt, ang2)
                proj_pix = (pix_pts[:, :2] - ctr) @ axis
                sub = (proj_pix >= lo_e - 0.004) & (proj_pix <= hi_e + 0.004)
            mask = np.zeros_like(pm)
            mask[pix_y[sub], pix_x[sub]] = True
            col = None
            if rgb is not None:
                col = [float(x) for x in np.median(rgb[mask], axis=0)]
            cubes.append(dict(xy=np.array(c_xy, dtype=float), yaw=float(ang), side=float(short_s if k > 1 else min(a, b)),
                              side_long=float(long_s / k), h=float(hgt), ztop=ztop, npix=int(sub.sum()),
                              mask=mask, color=col, split=k))
    return cubes, rejected


def color_name(rgb):
    if rgb is None:
        return "?"
    r, g, b = rgb
    mx, mn = max(rgb), min(rgb)
    if mx < 50:
        return "black"
    if mx - mn < 30:
        return "white" if mx > 170 else "gray"
    if r >= g and r >= b:
        if g > 0.75 * r and b < 0.6 * r:
            return "yellow"
        if b > 0.7 * r:
            return "purple"
        if g > 0.45 * r:
            return "orange"
        return "red"
    if g >= r and g >= b:
        if b > 0.8 * g:
            return "cyan"
        return "green"
    if r > 0.7 * b:
        return "purple"
    return "blue"


def highlight_scores(exec_rgb, exec_depth, cubes, frames, table_mask=None, rgb_thr=30, dep_thr=0.012, ring=4,
                     mask_key="mask"):
    """frames: list of (t, rgb, depth_or_None). Returns per-cube dict of score arrays."""
    nC = len(cubes)
    empty = np.zeros(exec_rgb.shape[:2], dtype=bool)
    masks = [dilate(c[mask_key], ring) if (c.get(mask_key) is not None and c[mask_key].shape == empty.shape)
             else empty for c in cubes]
    cores = [c[mask_key] if (c.get(mask_key) is not None and c[mask_key].shape == empty.shape) else empty
             for c in cubes]
    # remove other cubes' own pixels from each dilated region (cross-talk between neighbours)
    masks = [m & ~np.logical_or.reduce([cores[j] for j in range(nC) if j != i] + [empty]) for i, m in
             enumerate(masks)]
    allc = np.zeros(exec_rgb.shape[:2], dtype=bool)
    for m in masks:
        allc |= m
    if table_mask is not None:
        bg = table_mask & ~dilate(allc, 10)
    else:
        bg = ~dilate(allc, 10)
    E = exec_rgb.astype(np.int32)
    Ef = exec_rgb.astype(np.float32) + 1.0
    T = len(frames)
    S = np.zeros((nC, T))       # shadow-excluded, depth-gated
    Sraw = np.zeros((nC, T))    # depth-gated, shadows included
    Sany = np.zeros((nC, T))    # ungated
    BG = np.zeros(T)
    GL = np.zeros(T)
    for fi, (t, rgb, dep) in enumerate(frames):
        if rgb is None or rgb.shape != exec_rgb.shape:
            continue
        d = np.abs(rgb.astype(np.int32) - E).sum(-1)
        ch = d > rgb_thr
        if dep is not None and exec_depth is not None and dep.shape == exec_depth.shape:
            same = np.abs(dep - exec_depth) < dep_thr
            ch_d = ch & same
        else:
            ch_d = ch
        ratio = (rgb.astype(np.float32) + 1.0) / Ef
        shadow = (ratio.max(-1) < 0.92) & ((ratio.max(-1) - ratio.min(-1)) < 0.15)
        ch_s = ch_d & ~shadow
        bgv = float(ch_s[bg].mean()) if bg.any() else 0.0
        BG[fi] = bgv
        GL[fi] = float(ch.mean())
        for ci, m in enumerate(masks):
            if not m.any():
                continue
            S[ci, fi] = float(ch_s[m].mean()) - bgv
            Sraw[ci, fi] = float(ch_d[m].mean()) - float(ch_d[bg].mean() if bg.any() else 0.0)
            Sany[ci, fi] = float(ch[m].mean()) - float(ch[bg].mean() if bg.any() else 0.0)
    return dict(S=S, Sraw=Sraw, Sany=Sany, BG=BG, GL=GL)


def select_targets(scores, n_hint=None, singular=False, abs_thr=0.10):
    """scores: 1-D array (per cube peak score). Return indices of highlighted cubes."""
    s = np.asarray(scores, dtype=float)
    n = len(s)
    if n == 0:
        return [], "no cubes"
    order = list(np.argsort(-s))
    if n_hint:
        k = min(n_hint, n)
        return [int(i) for i in order[:k]], "count hint %d" % k
    if singular:
        return [int(order[0])], "singular"
    smax = float(s.max())
    if smax < 0.05:
        return [int(order[0])], "all weak (max %.3f); top-1 fallback" % smax
    thr = max(0.05, 0.3 * smax)
    sel = [int(i) for i in order if s[i] > thr]
    return sel, "thr %.3f (max %.3f)" % (thr, smax)
