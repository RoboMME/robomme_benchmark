"""Perception: colour masks, blobs, video cube tracking, container segmentation from point clouds."""
import math
import re
from collections import deque

import numpy as np

# hue ranges in degrees (lo, hi) ; red wraps
COLOR_HUES = {
    'red': [(345, 360), (0, 14)],
    'orange': [(15, 38)],
    'yellow': [(40, 70)],
    'green': [(75, 165)],
    'cyan': [(166, 199)],
    'blue': [(200, 255)],
    'purple': [(256, 295)],
    'violet': [(256, 295)],
    'magenta': [(296, 340)],
    'pink': [(296, 344)],
}
COLOR_WORDS = list(COLOR_HUES.keys()) + ['white', 'black', 'gray', 'grey', 'brown']
S_MIN, V_MIN = 0.45, 0.22
SEED_PALETTE = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'magenta']


def parse_colors(instruction):
    """Ordered colour words (dedup) as they appear in the instruction."""
    text = (instruction or '').lower()
    found = []
    for m in re.finditer(r'[a-z]+', text):
        w = m.group(0)
        if w in COLOR_WORDS and w not in found:
            found.append(w)
    return found


def rgb_to_hsv(rgb):
    a = rgb.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = np.max(a, axis=-1)
    mn = np.min(a, axis=-1)
    df = mx - mn
    h = np.zeros_like(mx)
    m = df > 1e-6
    rm = m & (mx == r)
    gm = m & (mx == g) & ~rm
    bm = m & ~rm & ~gm
    h[rm] = (60 * ((g[rm] - b[rm]) / df[rm]) + 360) % 360
    h[gm] = 60 * ((b[gm] - r[gm]) / df[gm]) + 120
    h[bm] = 60 * ((r[bm] - g[bm]) / df[bm]) + 240
    s = np.where(mx > 1e-6, df / np.maximum(mx, 1e-6), 0)
    return h, s, mx


def color_mask(rgb, color, s_min=S_MIN, v_min=V_MIN, hsv=None):
    h, s, v = hsv if hsv is not None else rgb_to_hsv(rgb)
    if color == 'any':
        return (s >= 0.55) & (v >= 0.25)
    if color in COLOR_HUES:
        m = np.zeros(h.shape, bool)
        for lo, hi in COLOR_HUES[color]:
            m |= (h >= lo) & (h <= hi)
        return m & (s >= s_min) & (v >= v_min)
    if color == 'white':
        return (s < 0.15) & (v > 0.85)
    if color == 'black':
        return v < 0.12
    if color in ('gray', 'grey'):
        return (s < 0.12) & (v > 0.3) & (v < 0.75)
    if color == 'brown':
        return (h >= 10) & (h <= 40) & (s > 0.4) & (v > 0.15) & (v < 0.6)
    # unknown -> any saturated
    return (s >= s_min + 0.1) & (v >= v_min)


def saturated_mask(rgb):
    h, s, v = rgb_to_hsv(rgb)
    return (s >= 0.55) & (v >= 0.25)


try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None


def label(mask, conn8=True):
    """Connected components on a boolean mask. Returns list of (rows, cols) arrays."""
    if _ndi is not None:
        try:
            st = np.ones((3, 3), int) if conn8 else None
            lab, n = _ndi.label(mask, structure=st)
            if n == 0:
                return []
            rr, cc = np.nonzero(lab)
            ll = lab[rr, cc]
            order = np.argsort(ll, kind='stable')
            rr, cc, ll = rr[order], cc[order], ll[order]
            cuts = np.flatnonzero(np.diff(ll)) + 1
            return [(a, b) for a, b in zip(np.split(rr, cuts), np.split(cc, cuts))]
        except Exception:
            pass
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    idx = np.argwhere(mask)
    nb = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if conn8:
        nb += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r0, c0 in idx:
        if seen[r0, c0]:
            continue
        q = deque([(r0, c0)])
        seen[r0, c0] = True
        rs, cs = [], []
        while q:
            r, c = q.popleft()
            rs.append(r)
            cs.append(c)
            for dr, dc in nb:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append((np.array(rs), np.array(cs)))
    return comps


def blobs(mask, min_area=4, max_area=4000):
    out = []
    if mask.sum() > 20000:
        return out
    for rs, cs in label(mask):
        a = len(rs)
        if a < min_area or a > max_area:
            continue
        h = rs.max() - rs.min() + 1
        w = cs.max() - cs.min() + 1
        fill = a / float(h * w)
        out.append(dict(area=a, v=float(rs.mean()), u=float(cs.mean()), rs=rs, cs=cs,
                        bbox=(int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())), fill=fill,
                        aspect=max(h, w) / max(1, min(h, w))))
    return out


def track_blobs(per_frame, gate_px=7.0):
    """per_frame: list of (frame_pos, [blobs]). Greedy nearest-centroid tracking."""
    tracks = []
    for fi, bl in per_frame:
        used = set()
        for t in tracks:
            if fi - t['last_fi'] > 40:
                continue
            best, bd = None, gate_px
            for j, b in enumerate(bl):
                if j in used:
                    continue
                d = math.hypot(b['u'] - t['u'], b['v'] - t['v'])
                if d < bd:
                    best, bd = j, d
            if best is not None:
                b = bl[best]
                used.add(best)
                t['obs'].append((fi, b))
                t['u'], t['v'], t['last_fi'] = b['u'], b['v'], fi
        for j, b in enumerate(bl):
            if j not in used:
                tracks.append(dict(obs=[(fi, b)], u=b['u'], v=b['v'], last_fi=fi, first_fi=fi))
    return tracks


def present_in(mask, b, frac=0.3):
    """Is a blob's region (dilated bbox) still occupied in `mask`?"""
    r0, r1, c0, c1 = b['bbox']
    r0, c0 = max(0, r0 - 1), max(0, c0 - 1)
    sub = mask[r0:r1 + 2, c0:c1 + 2]
    return sub.sum() >= frac * b['area']


# ----------------------------------------------------------------------------- point-cloud objects
def table_height(P, center_xy=None, radius=0.35):
    """Dominant horizontal plane height (mode of z) near center."""
    Q = P
    if center_xy is not None:
        d = np.hypot(P[:, 0] - center_xy[0], P[:, 1] - center_xy[1])
        Q = P[d < radius]
        if len(Q) < 200:
            Q = P
    z = Q[:, 2]
    z = z[np.isfinite(z)]
    z = z[(z > -1.5) & (z < 1.5)]
    if len(z) < 10:
        return None
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.004, 0.003))
    if len(hist) == 0:
        return float(np.median(z))
    i = int(np.argmax(hist))
    sel = z[(z >= edges[i] - 0.004) & (z <= edges[i + 1] + 0.004)]
    return float(np.median(sel))


def rect_fit(xy):
    """Min-area rectangle via angle sweep on 5-95 percentile extents. -> (angle, w_a, w_b, center)"""
    best = None
    c = xy.mean(axis=0)
    X = xy - c
    for deg in range(0, 90, 2):
        a = math.radians(deg)
        e1 = np.array([math.cos(a), math.sin(a)])
        e2 = np.array([-math.sin(a), math.cos(a)])
        p1, p2 = X @ e1, X @ e2
        l1 = np.percentile(p1, 97) - np.percentile(p1, 3)
        l2 = np.percentile(p2, 97) - np.percentile(p2, 3)
        area = l1 * l2
        if best is None or area < best[0]:
            m1 = 0.5 * (np.percentile(p1, 97) + np.percentile(p1, 3))
            m2 = 0.5 * (np.percentile(p2, 97) + np.percentile(p2, 3))
            best = (area, a, l1, l2, c + m1 * e1 + m2 * e2)
    _, a, l1, l2, cc = best
    return a, float(l1), float(l2), cc


def find_objects(P, zt, center_xy, half=0.18, cell=0.005, zmin=0.012, zmax=0.32,
                 exclude=None, excl_r=0.09, dilate=True):
    """Elevated objects near center_xy from world points P (N,3). Returns list of dicts."""
    if P is None or len(P) == 0:
        return []
    m = np.isfinite(P).all(axis=1)
    P = P[m]
    dx = P[:, 0] - center_xy[0]
    dy = P[:, 1] - center_xy[1]
    h = P[:, 2] - zt
    sel = (np.abs(dx) < half) & (np.abs(dy) < half) & (h > zmin) & (h < zmax)
    if exclude is not None:
        for ex in exclude:
            sel &= np.linalg.norm(P - np.asarray(ex)[None, :], axis=1) > excl_r
    Q = P[sel]
    if len(Q) < 5:
        return []
    n = int(math.ceil(2 * half / cell))
    ix = np.clip(((Q[:, 0] - center_xy[0] + half) / cell).astype(int), 0, n - 1)
    iy = np.clip(((Q[:, 1] - center_xy[1] + half) / cell).astype(int), 0, n - 1)
    occ = np.zeros((n, n), bool)
    zmaxg = np.full((n, n), -np.inf)
    occ[ix, iy] = True
    np.maximum.at(zmaxg, (ix, iy), Q[:, 2])
    # close single-cell holes so one object stays one component
    occ2 = occ.copy()
    occ2[1:-1, 1:-1] |= (occ[:-2, 1:-1] & occ[2:, 1:-1]) | (occ[1:-1, :-2] & occ[1:-1, 2:])
    comps = label(occ2)
    objs = []
    cell_id = -np.ones((n, n), int)
    for k, (rs, cs) in enumerate(comps):
        cell_id[rs, cs] = k
    pid = cell_id[ix, iy]
    for k, (rs, cs) in enumerate(comps):
        pts = Q[pid == k]
        if len(pts) < 5 or len(rs) < 3:
            continue
        top = float(np.percentile(pts[:, 2], 92))
        # split by top faces: adjacent containers share an occupancy component but not a top face
        tmask = np.zeros((n, n), bool)
        sel_c = zmaxg[rs, cs] > top - 0.014
        tmask[rs[sel_c], cs[sel_c]] = True
        tdil = tmask.copy()
        if not dilate:
            tmask = tmask & False
        tdil[1:, :] |= tmask[:-1, :]
        tdil[:-1, :] |= tmask[1:, :]
        tdil[:, 1:] |= tmask[:, :-1]
        tdil[:, :-1] |= tmask[:, 1:]
        tdil &= occ2
        subs = [sc for sc in label(tdil) if len(sc[0]) >= 4]
        subs.sort(key=lambda sc: -len(sc[0]))
        if len(subs) > 1 and len(subs[1][0]) >= 0.3 * len(subs[0][0]):
            groups = [sc for sc in subs if len(sc[0]) >= 0.3 * len(subs[0][0])]
        else:
            groups = [(rs, cs)]
        if len(groups) > 1:
            gc = np.array([[center_xy[0] - half + (g[0].mean() + 0.5) * cell,
                            center_xy[1] - half + (g[1].mean() + 0.5) * cell] for g in groups])
            dd = np.linalg.norm(pts[:, None, :2] - gc[None, :, :], axis=2)
            asg = np.argmin(dd, axis=1)
        else:
            asg = np.zeros(len(pts), int)
        for gi, (grs, gcs) in enumerate(groups):
            gpts = pts[asg == gi]
            if len(gpts) < 5:
                continue
            gtop = float(np.percentile(gpts[:, 2], 92))
            topm = gpts[:, 2] > gtop - 0.012
            tp = gpts[topm]
            if len(tp) < 4:
                tp = gpts
            a, l1, l2, cc = rect_fit(tp[:, :2])
            cxy = np.array([center_xy[0] - half + (grs + 0.5) * cell, center_xy[1] - half + (gcs + 0.5) * cell]).T
            objs.append(dict(center=np.array([cc[0], cc[1]]), mean_top=tp[:, :2].mean(axis=0), top=gtop,
                             height=gtop - zt, angle=a, l1=l1, l2=l2, n_pts=int(len(gpts)), n_cells=int(len(grs)),
                             cells_xy=cxy, foot_ext=float(max(np.ptp(cxy[:, 0]), np.ptp(cxy[:, 1])) + cell),
                             split=len(groups), pts=gpts, zt=zt))
    return objs


def dist_to_obj(o, xy):
    d = np.hypot(o['cells_xy'][:, 0] - xy[0], o['cells_xy'][:, 1] - xy[1])
    return float(d.min()), float(np.hypot(*(o['center'] - np.asarray(xy[:2]))))


def obj_from_points(pts, zt):
    if len(pts) < 5:
        return None
    gtop = float(np.percentile(pts[:, 2], 92))
    tp = pts[pts[:, 2] > gtop - 0.012]
    if len(tp) < 4:
        tp = pts
    a, l1, l2, cc = rect_fit(tp[:, :2])
    cxy = pts[:, :2]
    return dict(center=np.array([cc[0], cc[1]]), mean_top=tp[:, :2].mean(axis=0), top=gtop, height=gtop - zt,
                angle=a, l1=l1, l2=l2, n_pts=int(len(pts)), n_cells=int(len(pts)), cells_xy=cxy,
                foot_ext=float(max(np.ptp(cxy[:, 0]), np.ptp(cxy[:, 1]))), split=-1, pts=pts, zt=zt)


def split_by_seeds(o, seeds, target_xy, near=0.025):
    """If several cube seeds fall inside one object, keep the target's Voronoi cell."""
    if o is None or 'pts' not in o or seeds is None or len(seeds) < 2:
        return o
    inside = [np.asarray(s_[:2], float) for s_ in seeds if dist_to_obj(o, s_)[0] < near]
    if len(inside) < 2:
        return o
    S = np.array(inside)
    t = np.asarray(target_xy[:2], float)
    ti = int(np.argmin(np.linalg.norm(S - t, axis=1)))
    d = np.linalg.norm(o['pts'][:, None, :2] - S[None, :, :], axis=2)
    keep = np.argmin(d, axis=1) == ti
    o2 = obj_from_points(o['pts'][keep], o['zt'])
    if o2 is None:
        return o
    o2['split'] = 100 + len(inside)
    return o2
