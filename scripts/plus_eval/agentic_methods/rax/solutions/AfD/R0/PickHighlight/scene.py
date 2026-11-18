"""Scene analysis from the front camera + pixel_to_position back-projection (colour-agnostic).

highlighted cube = elevated cluster whose nearest visible ground, all around it, is
newly white (the highlight disc).  The button = persistent white plate with a grey cap.
"""
import math
import time

import numpy as np

from mcp_io import log

# ---------------------------------------------------------------- tunables (see notes/README)
ELEV_MIN = 0.010         # m above table: elevated (cube) point
ELEV_MAX = 0.10          # m above table: ignore robot / tall stuff
LINK = 0.012             # m single-linkage distance for elevated clusters (stride-2 sampling ~7 mm)
DEFAULT_SIZE = 0.04      # m cube size if it cannot be measured
COVER_BINS = 12
COVER_MIN_BINS = 5       # bins with ground data needed to judge (mock: hl cubes judged 7-8)
COVER_THRESH = 0.70      # fraction of judged bins whose nearest ground is white (mock: hl >=0.75, neighbour-disc FP 0.62)
FRAG_MERGE = 0.75        # x cube size: merge elevated clusters whose centroids are closer
GROUND_MAX = 0.15        # x cube size: max height of a ground/disc point (cube side strips excluded)
JUDGE_R = 0.82           # x cube size: ground nearer than this judges a bin (corner at 0.707)
REGION_PAD_PX = 14       # px padding around highlight components
MAX_SAMPLES = 2600       # px back-projected in the main pass


def white_mask(img, table_ref=None):
    a = img.astype(np.int16)
    mn, mx = a.min(axis=2), a.max(axis=2)
    thr = 195
    if table_ref is not None:
        thr = max(195, min(225, table_ref + 25))
    return (mn > thr) & (mx - mn < 45)


def gray_mask(img):
    a = img.astype(np.int16)
    mn, mx = a.min(axis=2), a.max(axis=2)
    return (mx - mn < 22) & (mn > 80) & (mn < 200)


def label(mask):
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        out = []
        for k, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            ys, xs = np.nonzero(lab[sl] == k)
            out.append((ys + sl[0].start, xs + sl[1].start))
        return out
    except Exception:
        pass
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]; seen[y0, x0] = True; ys = []; xs = []
        while stack:
            y, x = stack.pop(); ys.append(y); xs.append(x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True; stack.append((yy, xx))
        out.append((np.array(ys), np.array(xs)))
    return out


def dilate(mask, r):
    out = mask.copy()
    H, W = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx > r * r or (dy == 0 and dx == 0):
                continue
            sh = np.zeros_like(mask)
            sh[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = \
                mask[max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
            out |= sh
    return out


def table_brightness(img):
    a = img.astype(np.int16)
    return float(np.percentile(a[int(img.shape[0] * 0.15):].min(axis=2), 99.0))


def white_count(img, table_ref=None):
    m = white_mask(img, table_ref)
    return int(m[int(img.shape[0] * 0.08):].sum())


# ---------------------------------------------------------------- geometry helpers

def min_area_rect(xy):
    """Return (center_xy, yaw_mod_90_rad, side_a, side_b) of the min-area bounding rectangle."""
    xy = np.asarray(xy, dtype=float)
    best = None
    for deg in np.arange(0.0, 90.0, 1.0):
        t = math.radians(deg)
        c, s = math.cos(t), math.sin(t)
        x = xy[:, 0] * c + xy[:, 1] * s
        y = -xy[:, 0] * s + xy[:, 1] * c
        a = (x.max() - x.min()); b = (y.max() - y.min())
        area = a * b
        if best is None or area < best[0]:
            cx, cy = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
            center = np.array([cx * c - cy * s, cx * s + cy * c])
            best = (area, center, t, a, b)
    return best[1], best[2], best[3], best[4]


def single_linkage(xy, dist):
    n = len(xy)
    lab = -np.ones(n, dtype=int)
    k = 0
    d2 = dist * dist
    for i in range(n):
        if lab[i] >= 0:
            continue
        lab[i] = k
        stack = [i]
        while stack:
            j = stack.pop()
            dd = ((xy - xy[j]) ** 2).sum(axis=1)
            nb = np.nonzero((dd <= d2) & (lab < 0))[0]
            lab[nb] = k
            stack.extend(nb.tolist())
        k += 1
    return lab, k


def kmeans(xy, k, iters=20):
    idx = np.linspace(0, len(xy) - 1, k).astype(int)
    # farthest-point init
    cents = [xy[0]]
    for _ in range(1, k):
        d = np.min([((xy - c) ** 2).sum(1) for c in cents], axis=0)
        cents.append(xy[int(np.argmax(d))])
    cents = np.array(cents)
    for _ in range(iters):
        d = np.stack([((xy - c) ** 2).sum(1) for c in cents], axis=1)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                cents[j] = xy[lab == j].mean(0)
    return lab


# ---------------------------------------------------------------- sampling

def sample_points(robot, cam, pixels, budget_s=60.0):
    pts = robot.backproject(cam, pixels, budget_s=budget_s)
    U, V, P = [], [], []
    for (u, v), p in zip(pixels, pts):
        if p is not None and np.all(np.isfinite(p)):
            U.append(u); V.append(v); P.append(p)
    if not P:
        return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3))
    return np.array(U), np.array(V), np.array(P)


def estimate_table(robot, img_shape=(256, 256)):
    H, W = img_shape[:2]
    pix = [(u, v) for v in range(int(H * 0.2), int(H * 0.95), max(8, H // 13))
           for u in range(int(W * 0.05), int(W * 0.95), max(8, W // 13))]
    U, V, P = sample_points(robot, 'front', pix, budget_s=40)
    if len(P) < 10:
        return None, (U, V, P)
    z = P[:, 2]
    hist, edges = np.histogram(z, bins=np.arange(z.min() - 0.0025, z.max() + 0.005, 0.005))
    i = int(np.argmax(hist))
    sel = (z >= edges[i] - 0.004) & (z <= edges[i + 1] + 0.004)
    return float(np.median(z[sel])), (U, V, P)


def find_button(img_ref, table_ref, exclude_mask=None):
    """White plate component with the most grey (cap) pixels inside. Returns dict or None."""
    wm = white_mask(img_ref, table_ref)
    if exclude_mask is not None:
        wm &= ~exclude_mask
    H = img_ref.shape[0]
    wm[:int(H * 0.06)] = False
    gm = gray_mask(img_ref)
    best = None
    for ys, xs in label(wm):
        if len(ys) < 8:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        fill = len(ys) / float(bw * bh)
        g = gm[y0:y1 + 1, x0:x1 + 1]
        gy, gx = np.nonzero(g)
        score = len(gy) + 0.2 * len(ys) + 10 * fill
        if best is None or score > best['score']:
            best = dict(score=score, bbox=(int(x0), int(y0), int(x1), int(y1)), area=len(ys),
                        plate_px=(ys, xs), gray_px=(gy + y0, gx + x0), fill=fill)
    return best


def locate_button(robot, btn, table_z):
    """Back-project button cap/plate pixels. Returns dict(xy, top_z, plate_z) or None."""
    if btn is None:
        return None
    gy, gx = btn['gray_px']
    ys, xs = btn['plate_px']
    pix = []
    if len(gy) >= 4:
        sel = np.linspace(0, len(gy) - 1, min(24, len(gy))).astype(int)
        pix += [(int(gx[i]), int(gy[i])) for i in sel]
    sel = np.linspace(0, len(ys) - 1, min(24, len(ys))).astype(int)
    pix += [(int(xs[i]), int(ys[i])) for i in sel]
    U, V, P = sample_points(robot, 'front', pix, budget_s=20)
    if len(P) < 3:
        return None
    ng = min(24, len(gy)) if len(gy) >= 4 else 0
    # valid gray points are those whose pixel came from the gray list
    pxset = set(pix[:ng])
    isg = np.array([(int(u), int(v)) in pxset for u, v in zip(U, V)])
    P_ok = P[np.abs(P[:, 2] - table_z) < 0.08]
    if len(P_ok) < 3:
        return None
    if isg.sum() >= 3:
        Pg = P[isg]
        Pg = Pg[np.abs(Pg[:, 2] - table_z) < 0.08]
    else:
        Pg = np.zeros((0, 3))
    if len(Pg) >= 3:
        xy = np.median(Pg[:, :2], axis=0)
        top = float(np.percentile(Pg[:, 2], 80))
    else:
        xy = np.median(P_ok[:, :2], axis=0)
        top = float(np.percentile(P_ok[:, 2], 90))
    plate = float(np.percentile(P_ok[:, 2], 30))
    return dict(xy=xy, top_z=max(top, plate), plate_z=plate, n=len(P_ok))


# ---------------------------------------------------------------- highlight analysis

def highlight_components(hl_img, base_img, table_ref, btn_mask=None):
    w1 = white_mask(hl_img, table_ref)
    if base_img is not None:
        w0 = white_mask(base_img, table_ref)
        new = w1 & ~dilate(w0, 2)
    else:
        new = w1.copy()
        if btn_mask is not None:
            new &= ~btn_mask
    new[:int(hl_img.shape[0] * 0.06)] = False
    comps = [c for c in label(new) if len(c[0]) >= 15]
    return new, comps


def region_pixels(comps, shape, pad, max_samples):
    H, W = shape[:2]
    reg = np.zeros((H, W), dtype=bool)
    for ys, xs in comps:
        reg[max(0, ys.min() - pad):min(H, ys.max() + pad + 1), max(0, xs.min() - pad):min(W, xs.max() + pad + 1)] = True
    vv, uu = np.nonzero(reg)
    n = len(vv)
    for stride in (1, 2, 3, 4, 5):
        if stride == 1:
            sel = ((uu + vv) % 2 == 0)        # checkerboard ~ half density
        else:
            sel = (uu % stride == 0) & (vv % stride == 0)
        if sel.sum() <= max_samples:
            break
    return [(int(u), int(v)) for u, v in zip(uu[sel], vv[sel])], reg


def cluster_cubes(U, V, P, table_z, size_guess=None):
    """Elevated clusters -> list of cube dicts (center xy, top_z, size, yaw, pixels)."""
    h = P[:, 2] - table_z
    el = (h > ELEV_MIN) & (h < ELEV_MAX)
    idx = np.nonzero(el)[0]
    if len(idx) < 3:
        return [], size_guess or DEFAULT_SIZE
    lab, k = single_linkage(P[idx, :2], LINK)
    raw = []
    for j in range(k):
        m = idx[lab == j]
        if len(m) < 3:
            continue
        top = float(np.percentile(P[m, 2], 90))
        raw.append((m, top))
    sizes = [t - table_z for m, t in raw if len(m) >= 6 and 0.012 < t - table_z < 0.09]
    s = float(np.median(sizes)) if sizes else (size_guess or DEFAULT_SIZE)
    if size_guess:
        s = size_guess if not sizes else s
    # merge fragments of one cube (sparse sampling of far cubes splits them); distinct cubes
    # cannot have upper-part centroids closer than ~0.75 s
    merged = True
    while merged and len(raw) > 1:
        merged = False
        cents = [P[m, :2].mean(0) for m, t in raw]
        for a in range(len(raw)):
            for b in range(a + 1, len(raw)):
                if np.linalg.norm(cents[a] - cents[b]) < FRAG_MERGE * s:
                    m = np.concatenate([raw[a][0], raw[b][0]])
                    xy = P[m, :2]
                    ext = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)).max()
                    if ext < 1.6 * s:
                        raw[a] = (m, max(raw[a][1], raw[b][1])); del raw[b]
                        merged = True
                        break
            if merged:
                break
    cubes = []
    for m, top in raw:
        topm = m[P[m, 2] > table_z + 0.7 * s]
        if len(topm) < 2:
            topm = m[P[m, 2] > table_z + 0.5 * (top - table_z)]
        if len(topm) < 2:
            continue
        xy = P[topm, :2]
        ext = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)).max()
        groups = [topm]
        if ext > 1.75 * s and len(topm) >= 6:
            kk = 2 if ext < 2.9 * s else 3
            l2 = kmeans(xy, kk)
            groups = [topm[l2 == q] for q in range(kk) if (l2 == q).sum() >= 2]
        for g in groups:
            xy = P[g, :2]
            if len(g) >= 4:
                c, yaw, a, b = min_area_rect(xy)
            else:
                c, yaw, a, b = xy.mean(0), None, 0.0, 0.0
            cubes.append(dict(xy=np.array(c), top_z=float(np.percentile(P[g, 2], 80)), yaw=yaw,
                              n_top=int(len(g)), rect=(float(a), float(b)),
                              u=float(np.mean(U[g])), v=float(np.mean(V[g])),
                              pix=[(int(U[i]), int(V[i])) for i in g]))
    return cubes, s


def coverage(cube, U, V, P, white, table_z, s):
    """Fraction of judged angular bins whose nearest visible ground point is white.

    A bin is judged only if visible ground exists just outside the cube footprint
    (r < JUDGE_R * s); bins occluded by the cube itself (seen from the camera) are skipped.
    """
    h = P[:, 2] - table_z
    ground = h < GROUND_MAX * s
    d = P[:, :2] - cube['xy']
    r = np.sqrt((d ** 2).sum(1))
    sel = ground & (r > 0.45 * s) & (r < JUDGE_R * s)
    if cube.get('yaw') is not None:
        c, sn = math.cos(cube['yaw']), math.sin(cube['yaw'])
        lx = np.abs(d[:, 0] * c + d[:, 1] * sn)
        ly = np.abs(-d[:, 0] * sn + d[:, 1] * c)
        sel &= ~((lx < 0.5 * s + 0.003) & (ly < 0.5 * s + 0.003))
    if sel.sum() < 4:
        return 0.0, 0
    ang = np.arctan2(d[sel, 1], d[sel, 0])
    rb = r[sel]
    wb = white[sel]
    bins = ((ang + math.pi) / (2 * math.pi) * COVER_BINS).astype(int) % COVER_BINS
    judged = 0; good = 0
    for b in range(COVER_BINS):
        m = np.nonzero(bins == b)[0]
        if len(m) == 0:
            continue
        o = m[np.argsort(rb[m])][:2]
        judged += 1
        if wb[o].mean() >= 0.5:
            good += 1
    return (good / judged if judged else 0.0), judged


def top_face_center(P, xy0, top_z, s):
    """Centre/yaw from points on the top face only (|z - top| small) near xy0."""
    r = np.sqrt(((P[:, :2] - xy0) ** 2).sum(1))
    m = (np.abs(P[:, 2] - top_z) < max(0.005, 0.12 * s)) & (r < 0.95 * s)
    if m.sum() < 6:
        return None
    c, yaw, a, b = min_area_rect(P[m, :2])
    if not (0.45 * s < max(a, b) < 1.35 * s):
        return None
    if max(a, b) > 0.85 * s and min(a, b) > 0.6 * s:
        return np.array(c), yaw, int(m.sum())
    # partly seen top face: only trust the yaw, shift centre away from the visible part is
    # unknown -> keep mean but report
    return np.array(c), yaw, int(m.sum())


def rejudge(robot, c, U, V, P, new, shape, table_z, s, pad=10, cap=900):
    us = [p[0] for p in c['pix']]; vs = [p[1] for p in c['pix']]
    have = set(zip(U.tolist(), V.tolist()))
    dp = [(u, v) for v in range(max(0, min(vs) - pad), min(shape[0], max(vs) + pad + 1))
          for u in range(max(0, min(us) - pad), min(shape[1], max(us) + pad + 1))
          if (u, v) not in have]
    if len(dp) > cap:
        dp = dp[::2]
    U2, V2, P2 = sample_points(robot, 'front', dp, budget_s=25)
    if len(P2) < 6:
        return None
    Ua, Va, Pa = np.concatenate([U, U2]), np.concatenate([V, V2]), np.concatenate([P, P2])
    wa = new[Va, Ua]
    xy, yaw = c['xy'], c.get('yaw')
    # top height: robust high percentile of elevated points near the cluster
    r = np.sqrt(((Pa[:, :2] - xy) ** 2).sum(1))
    el = (Pa[:, 2] > table_z + 0.5 * s) & (Pa[:, 2] < table_z + 1.6 * s) & (r < 0.9 * s)
    if el.sum() >= 4:
        top = float(np.percentile(Pa[el, 2], 85))
        for _ in range(2):
            tf = top_face_center(Pa, xy, top, s)
            if tf is None:
                break
            if np.linalg.norm(tf[0] - xy) < 0.6 * s:
                xy, yaw = tf[0], tf[1]
    cube = dict(c); cube['xy'] = xy; cube['yaw'] = yaw
    cov, judged = coverage(cube, Ua, Va, Pa, wa, table_z, s)
    return cov, judged, xy, yaw


def white_profile(c, P, white, table_z, s, others):
    """White fraction of visible ground vs radius (bins of 0.1 s from 0.5 s)."""
    h = P[:, 2] - table_z
    g = h < GROUND_MAX * s
    for o in others:
        g &= np.sqrt(((P[:, :2] - o['xy']) ** 2).sum(1)) > 0.75 * s
    r = np.sqrt(((P[:, :2] - c['xy']) ** 2).sum(1)) / s
    prof = []
    for k in range(17):
        lo = 0.5 + 0.1 * k
        m = g & (r >= lo) & (r < lo + 0.1)
        prof.append((lo + 0.05, int(m.sum()), float(white[m].mean()) if m.sum() else None))
    return prof


def disc_consistency(robot, hl, cubes, U, V, P, white, new, shape, table_z, s):
    """Veto 'highlighted' cubes whose white ring is explained by neighbouring discs while the
    wider annulus (up to ~0.9 R_disc) is not white all around (a cube sandwiched between
    highlighted neighbours with large discs)."""
    if len(hl) < 2:
        return hl
    Rs = []
    for c in hl:
        if c['cov'] < 0.85:
            continue
        prof = white_profile(c, P, white, table_z, s, [o for o in cubes if o is not c])
        R = None
        for rad, n, f in prof:
            if n >= 3 and f is not None:
                if f < 0.5:
                    R = rad; break
                R = rad + 0.05
        if R is not None:
            Rs.append(R)
    if len(Rs) < 2:
        log('  disc radius: too few estimates', Rs)
        return hl
    R = float(np.median(Rs))
    log('  disc radius estimate %.2f s (from %s)' % (R, np.round(Rs, 2).tolist()))
    if 0.9 * R < 0.95:
        return hl
    keep = []
    h = P[:, 2] - table_z
    for c in hl:
        others = [o for o in hl if o is not c]
        d = np.sqrt(((P[:, :2] - c['xy']) ** 2).sum(1))
        ring = (h < GROUND_MAX * s) & (d > 0.5 * s) & (d < JUDGE_R * s) & white
        if ring.sum() < 3 or not others:
            keep.append(c); continue
        expl = np.zeros(len(P), bool)
        for o in others:
            expl |= np.sqrt(((P[:, :2] - o['xy']) ** 2).sum(1)) < 0.95 * R * s
        ef = float(expl[ring].mean())
        if ef < 0.6:
            keep.append(c); continue
        Ua, Va, Pa, wa = U, V, P, white
        ann_fn = lambda Pa: ((Pa[:, 2] - table_z < GROUND_MAX * s) &
                             (np.sqrt(((Pa[:, :2] - c['xy']) ** 2).sum(1)) > 0.55 * s) &
                             (np.sqrt(((Pa[:, :2] - c['xy']) ** 2).sum(1)) < 0.9 * R * s))
        ann = ann_fn(Pa)
        for o in cubes:
            if o is not c:
                ann &= np.sqrt(((Pa[:, :2] - o['xy']) ** 2).sum(1)) > 0.75 * s
        frac = float(wa[ann].mean()) if ann.sum() else 1.0
        # angular completeness of the annulus whiteness
        dd = Pa[ann, :2] - c['xy']
        bins = ((np.arctan2(dd[:, 1], dd[:, 0]) + math.pi) / (2 * math.pi) * 12).astype(int) % 12
        bad_bins = sum(1 for b in range(12) if (bins == b).sum() >= 2 and wa[ann][bins == b].mean() < 0.4)
        log('  disc check cube (%.3f,%.3f): explained %.2f annulus white %.2f n=%d bad_bins %d' % (
            c['xy'][0], c['xy'][1], ef, frac, int(ann.sum()), bad_bins))
        if frac < 0.7 and bad_bins >= 2:
            log('  -> vetoed (white ring belongs to neighbouring discs)')
            c['highlighted'] = False
            continue
        keep.append(c)
    return keep


def analyze(robot, hl_img, base_img, table_z, btn_mask=None, time_budget=110.0, max_samples=None):
    """Main front-camera analysis on the current (static) observation."""
    t0 = time.time()
    tref = table_brightness(base_img if base_img is not None else hl_img)
    new, comps = highlight_components(hl_img, base_img, tref, btn_mask)
    log('highlight components:', len(comps), [len(c[0]) for c in comps])
    if not comps:
        return dict(cubes=[], highlighted=[], size=DEFAULT_SIZE, comps=0)
    if max_samples is None:
        lat = robot.p2p_latency()
        # keep the main pass around <= 60 s even with slow back-projection calls
        max_samples = int(max(700, min(MAX_SAMPLES, 32 * 60.0 / max(lat, 1e-3))))
        log('p2p latency %.3f s -> max_samples %d' % (lat, max_samples))
    pix, reg = region_pixels(comps, hl_img.shape, REGION_PAD_PX, max_samples)
    log('sampling', len(pix), 'front pixels')
    U, V, P = sample_points(robot, 'front', pix, budget_s=time_budget * 0.6)
    log('valid points', len(P), 'p2p calls', robot.p2p_calls, 'p2p time %.1f' % robot.p2p_time)
    if len(P) < 20:
        return dict(cubes=[], highlighted=[], size=DEFAULT_SIZE, comps=len(comps))
    white = new[V, U]
    cubes, s = cluster_cubes(U, V, P, table_z)
    log('cube size est %.4f, clusters %d' % (s, len(cubes)))
    for c in cubes:
        cov, judged = coverage(c, U, V, P, white, table_z, s)
        c['cov'], c['judged'] = cov, judged
        hgt = c['top_z'] - table_z
        c['size_ok'] = 0.55 * s < hgt < 1.6 * s
        c['highlighted'] = bool(cov >= COVER_THRESH and judged >= COVER_MIN_BINS and c['size_ok'])
        # colour of the cube top in the front image
        cols = np.array([hl_img[v, u] for (u, v) in c['pix']], dtype=float)
        c['color'] = cols.mean(0) if len(cols) else None
        log('  cube xy=(%.3f,%.3f) top=%.3f n=%d cov=%.2f judged=%d hl=%s uv=(%.0f,%.0f)' % (
            c['xy'][0], c['xy'][1], c['top_z'], c['n_top'], cov, judged, c['highlighted'], c['u'], c['v']))
    # ambiguous cubes (sparse ring or mid coverage): dense resample, recentre on the true top
    # face (points near the top height; the upper side strip biases centres toward the camera)
    # and re-judge
    amb = [c for c in cubes if not c['highlighted'] and c['size_ok'] and c['n_top'] >= 3 and
           ((c['judged'] < COVER_MIN_BINS and c['cov'] >= 0.3) or (c['judged'] < 2) or
            (0.3 <= c['cov'] < COVER_THRESH))]
    amb.sort(key=lambda c: (-c['cov'], -c['n_top']))
    for c in amb[:7]:
        if time.time() - t0 > time_budget * 0.85:
            log('  re-judge time budget reached')
            break
        rj = rejudge(robot, c, U, V, P, new, hl_img.shape, table_z, s)
        if rj is None:
            continue
        cov, judged, xy, yaw = rj
        c['cov'], c['judged'], c['xy'], c['yaw'] = cov, judged, xy, yaw
        c['highlighted'] = bool(cov >= COVER_THRESH and judged >= COVER_MIN_BINS - 1 and c['size_ok'])
        log('  re-judged cube xy=(%.3f,%.3f) cov=%.2f judged=%d hl=%s' % (
            c['xy'][0], c['xy'][1], cov, judged, c['highlighted']))
    hl = []
    for c in sorted([c for c in cubes if c['highlighted']], key=lambda c: -c['n_top']):
        if any(np.linalg.norm(c['xy'] - o['xy']) < 0.65 * s for o in hl):
            log('  duplicate hl cluster at (%.3f,%.3f) dropped' % (c['xy'][0], c['xy'][1]))
            c['highlighted'] = False
            continue
        hl.append(c)
    try:
        hl = disc_consistency(robot, hl, cubes, U, V, P, white, new, hl_img.shape, table_z, s)
    except Exception as e:
        log('  disc consistency check failed', repr(e)[:200])
    # dense refinement of each highlighted cube's top face (stride 1)
    for c in hl:
        if time.time() - t0 > time_budget:
            break
        us = [p[0] for p in c['pix']]; vs = [p[1] for p in c['pix']]
        pad = 4
        dp = [(u, v) for v in range(max(0, min(vs) - pad), min(hl_img.shape[0], max(vs) + pad + 1))
              for u in range(max(0, min(us) - pad), min(hl_img.shape[1], max(us) + pad + 1))]
        if len(dp) > 640:
            dp = dp[::2]
        U2, V2, P2 = sample_points(robot, 'front', dp, budget_s=20)
        if len(P2) < 6:
            continue
        r = np.sqrt(((P2[:, :2] - c['xy']) ** 2).sum(1))
        top = (P2[:, 2] > table_z + 0.7 * s) & (r < 0.75 * s)
        if top.sum() >= 6:
            cc, yaw, a, b = min_area_rect(P2[top, :2])
            if np.linalg.norm(cc - c['xy']) < 0.4 * s:
                c['xy'], c['yaw'], c['rect'] = np.array(cc), yaw, (float(a), float(b))
                c['top_z'] = float(np.percentile(P2[top, 2], 80))
                c['n_top'] = int(top.sum())
                cols = np.array([hl_img[int(v), int(u)] for u, v in zip(U2[top], V2[top])], dtype=float)
                c['color'] = cols.mean(0)
                log('  refined hl cube xy=(%.3f,%.3f) yaw=%.1f rect=(%.3f,%.3f) n=%d' % (
                    cc[0], cc[1], math.degrees(yaw), a, b, top.sum()))
    return dict(cubes=cubes, highlighted=hl, size=s, comps=len(comps), points=(U, V, P, white))
