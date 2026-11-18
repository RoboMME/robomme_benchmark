"""Geometric perception from one camera's RGB-D: table plane, raised objects,
flat coloured patches, and the moving cube.  Pure numpy functions (offline-testable)."""
import numpy as np

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None
try:
    import cv2 as _cv2
except Exception:  # pragma: no cover
    _cv2 = None

# ---- tunables (blind defaults; tested only on the synthetic offline scene) ----
RAISED_MIN_H = 0.008      # m above table to count as an object pixel
OBJECT_MAX_H = 0.12       # components reaching above this are robot / tall structure
TRACK_MAX_H = 0.10        # cube pixels must be below this height
FLAT_MAX_H = 0.006        # flat-patch pixels
EDGE_JUMP = 0.02          # 3-D neighbour distance that cuts components
MIN_OBJ_AREA = 12         # px


def label(mask):
    mask = np.asarray(mask, bool)
    if _ndi is not None:
        lab, n = _ndi.label(mask)
        return lab, n
    if _cv2 is not None:
        n, lab = _cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
        return lab, n - 1
    # pure python fallback
    H, W = mask.shape
    lab = np.zeros((H, W), int)
    n = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and lab[i, j] == 0:
                n += 1
                stack = [(i, j)]
                lab[i, j] = n
                while stack:
                    a, b = stack.pop()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + da, b + db
                        if 0 <= x < H and 0 <= y < W and mask[x, y] and lab[x, y] == 0:
                            lab[x, y] = n
                            stack.append((x, y))
    return lab, n


def dilate(mask, r=1):
    mask = np.asarray(mask, bool)
    if r <= 0:
        return mask
    if _ndi is not None:
        return _ndi.binary_dilation(mask, iterations=r)
    out = mask.copy()
    for _ in range(r):
        m = out.copy()
        m[1:] |= out[:-1]
        m[:-1] |= out[1:]
        m[:, 1:] |= out[:, :-1]
        m[:, :-1] |= out[:, 1:]
        out = m
    return out


def erode(mask, r=1):
    return ~dilate(~np.asarray(mask, bool), r)


def table_height(P, valid, ee_z=None):
    z = P[..., 2][valid]
    if z.size < 100:
        return None
    if ee_z is not None:
        sel = z[(z < ee_z - 0.02) & (z > ee_z - 0.9)]
        if sel.size > 500:
            z = sel
    lo, hi = np.percentile(z, 1), np.percentile(z, 99)
    if hi - lo < 1e-3:
        return float(np.median(z))
    bins = np.arange(lo, hi + 0.002, 0.002)
    h, e = np.histogram(z, bins=bins)
    i = int(np.argmax(h))
    z0 = 0.5 * (e[i] + e[i + 1])
    near = z[np.abs(z - z0) < 0.004]
    return float(np.median(near)) if near.size else float(z0)


def edge_mask(P, valid):
    """Pixels with a 3-D jump > EDGE_JUMP to a 4-neighbour (or invalid neighbour)."""
    E = ~valid.copy()
    d = np.linalg.norm(P[1:] - P[:-1], axis=-1)
    j = (d > EDGE_JUMP) | ~valid[1:] | ~valid[:-1]
    E[1:] |= j
    E[:-1] |= j
    d = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=-1)
    j = (d > EDGE_JUMP) | ~valid[:, 1:] | ~valid[:, :-1]
    E[:, 1:] |= j
    E[:, :-1] |= j
    return E


def comp_stats(lab, n, P, Hm, rgb, min_area=MIN_OBJ_AREA):
    out = []
    if n == 0:
        return out
    idx = lab.reshape(-1)
    counts = np.bincount(idx, minlength=n + 1)
    Pf = P.reshape(-1, 3)
    Hf = Hm.reshape(-1)
    Cf = rgb.reshape(-1, 3).astype(float) if rgb is not None else None
    H, W = lab.shape
    for k in range(1, n + 1):
        if counts[k] < min_area:
            continue
        sel = np.nonzero(idx == k)[0]
        pts = Pf[sel]
        hs = Hf[sel]
        top = float(np.percentile(hs, 90))
        topsel = hs >= top - 0.01
        xy_top = np.median(pts[topsel, :2], axis=0)
        vs, us = np.divmod(sel, W)
        c = {
            'id': int(k), 'area': int(counts[k]),
            'uv': (float(np.mean(us)), float(np.mean(vs))),
            'xyz': np.median(pts, axis=0), 'xy_top': xy_top,
            'top_h': top, 'max_h': float(np.max(hs)),
            'spread': float(np.sqrt(np.mean(np.var(pts[:, :2], axis=0)))),
            'rgb': (np.median(Cf[sel], axis=0) if Cf is not None else None),
        }
        out.append(c)
    return out


def raised_components(P, valid, Hm, rgb, edges, max_h=OBJECT_MAX_H):
    """Return (objects, robot_mask): components of raised pixels; those touching
    pixels higher than max_h are robot/tall structure."""
    raised = valid & (Hm > RAISED_MIN_H) & (Hm < 0.6) & ~edges
    lab, n = label(raised)
    comps = comp_stats(lab, n, P, Hm, rgb, min_area=1)
    objs = []
    robot = np.zeros(lab.shape, bool)
    for c in comps:
        if c['max_h'] > max_h:
            robot |= (lab == c['id'])
        elif c['area'] >= MIN_OBJ_AREA:
            c['mask'] = (lab == c['id'])
            objs.append(c)
    return objs, robot


def chroma(rgb):
    f = rgb.astype(float)
    s = f.sum(-1, keepdims=True) + 1.0
    return f / s, f.mean(-1)


def flat_patches(P, valid, Hm, rgb, exclude, min_area=15):
    """Flat (table-height) regions whose colour differs from the table.
    Shadows (darker, same chroma) are ignored."""
    flat = valid & (np.abs(Hm) < FLAT_MAX_H) & ~dilate(exclude, 3)
    if flat.sum() < 200:
        return [], None
    ch, br = chroma(rgb)
    ct = np.median(ch[flat], axis=0)
    bt = np.median(br[flat])
    dc = np.linalg.norm(ch - ct, axis=-1)
    mad_c = np.median(dc[flat]) + 1e-4
    mad_b = np.median(np.abs(br[flat] - bt)) + 1e-3
    thr_c = max(0.05, 6.0 * mad_c)
    thr_b = max(35.0, 6.0 * mad_b)
    distinct = flat & ((dc > thr_c) | (br > bt + thr_b) | ((br < bt - 2.5 * thr_b) & (dc > 0.5 * thr_c)))
    distinct = dilate(erode(distinct, 1), 1)
    lab, n = label(distinct)
    comps = comp_stats(lab, n, P, Hm, rgb, min_area=min_area)
    for c in comps:
        m = lab == c['id']
        pts = P[m][:, :2]
        cen = pts.mean(0)
        cov = np.cov((pts - cen).T) if pts.shape[0] > 3 else np.eye(2) * 1e-4
        ev = np.linalg.eigvalsh(cov)
        c['center'] = cen
        c['radius'] = float(2.0 * np.sqrt(max(ev[-1], 1e-8)))
        c['ecc'] = float(np.sqrt(max(ev[-1], 1e-10) / max(ev[0], 1e-10)))
        c['mask'] = m
    # merge patches whose centres are close (concentric rings etc.)
    merged = []
    used = [False] * len(comps)
    for i, c in enumerate(comps):
        if used[i]:
            continue
        group = [c]
        used[i] = True
        for j in range(i + 1, len(comps)):
            if not used[j] and np.linalg.norm(comps[j]['center'] - c['center']) < max(0.03, c['radius']):
                group.append(comps[j])
                used[j] = True
        if len(group) == 1:
            merged.append(c)
        else:
            m = np.zeros_like(group[0]['mask'])
            for g in group:
                m |= g['mask']
            pts = P[m][:, :2]
            cen = pts.mean(0)
            ev = np.linalg.eigvalsh(np.cov((pts - cen).T))
            g0 = dict(group[0])
            g0.update(center=cen, radius=float(2.0 * np.sqrt(max(ev[-1], 1e-8))), mask=m,
                      area=int(m.sum()))
            merged.append(g0)
    info = {'table_chroma': ct.tolist(), 'table_bright': float(bt), 'thr_c': thr_c, 'thr_b': thr_b}
    return merged, info


def track_mask(P, valid, Hm, edges, static_mask, ee_xyz, ee_excl=0.075, H0=None):
    m = valid & (Hm > RAISED_MIN_H) & (Hm < TRACK_MAX_H) & ~edges
    if static_mask is not None:
        if H0 is not None:
            m &= ~(static_mask & (np.abs(Hm - H0) < 0.008))
        else:
            m &= ~static_mask
    if ee_xyz is not None:
        dxy = np.linalg.norm(P[..., :2] - ee_xyz[None, None, :2], axis=-1)
        m &= ~((dxy < ee_excl) & (P[..., 2] > ee_xyz[2] - 0.06))
        # anything vertically under the hand within a small radius is gripper
        m &= ~(dxy < 0.035)
    return m


def cube_candidates(P, valid, Hm, rgb, edges, static_mask, ee_xyz, min_area=6, H0=None):
    m = track_mask(P, valid, Hm, edges, static_mask, ee_xyz, H0=H0)
    lab, n = label(m)
    return comp_stats(lab, n, P, Hm, rgb, min_area=min_area)


def point_line_dist(p, a, d):
    """Distance of point p (2,) to line through a with unit direction d."""
    w = np.asarray(p) - np.asarray(a)
    return float(abs(w[0] * d[1] - w[1] * d[0]))
