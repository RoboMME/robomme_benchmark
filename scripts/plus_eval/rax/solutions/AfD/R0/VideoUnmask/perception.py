"""Colour-cube detection and helpers (pure numpy / cv2)."""
import numpy as np
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

COLOR_WORDS = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'black', 'white', 'pink', 'cyan', 'brown', 'gray', 'grey']


def color_mask(img, color):
    """img: HxWx3 uint8 RGB. Returns bool mask of pixels of the given cube colour."""
    a = img.astype(np.int32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    mx = np.maximum(np.maximum(R, G), B)
    if color == 'red':
        return (R > 70) & (G * 4 < R) & (B * 4 < R)
    if color == 'green':
        return (G > 60) & (R * 2 < G) & (B * 2 < G)
    if color == 'blue':
        return (B > 50) & (R * 2 < B) & (G * 10 < B * 6)
    if color == 'yellow':
        return (R > 120) & (G > 0.75 * R) & (B < 0.35 * R)
    if color == 'purple':
        return (B > 80) & (R > 0.6 * B) & (G < 0.6 * B) & (G < 0.6 * R)
    if color == 'orange':
        return (R > 150) & (G > 0.45 * R) & (G < 0.72 * R) & (B < 0.15 * R)
    if color == 'cyan':
        return (B > 80) & (G > 0.7 * B) & (R < 0.4 * B)
    if color == 'pink':
        return (R > 150) & (B > 0.55 * R) & (G < 0.6 * R)
    return np.zeros(R.shape, bool)


def blobs(mask, min_area=6):
    """Connected components -> list of dicts (area, cx, cy, bbox, pixels)."""
    out = []
    if mask is None or not mask.any():
        return out
    m = mask.astype(np.uint8)
    if cv2 is not None:
        n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
        for k in range(1, n):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            ys, xs = np.nonzero(lab == k)
            out.append({'area': area, 'cx': float(xs.mean()), 'cy': float(ys.mean()),
                        'xs': xs, 'ys': ys})
    else:
        ys, xs = np.nonzero(m)
        out.append({'area': int(len(xs)), 'cx': float(xs.mean()), 'cy': float(ys.mean()), 'xs': xs, 'ys': ys})
    return out


def parse_instruction(text):
    """Return ordered list of (color, mode) with mode 'one' or 'all'."""
    t = (text or '').lower().replace('grey', 'gray')
    import re
    # split into clauses around ordering connectives
    parts = re.split(r'\bthen\b|followed by|\bfinally\b|\bnext\b|\bafter that\b|,', t)
    seq = []
    for p in parts:
        found = [(p.find(c), c) for c in COLOR_WORDS if re.search(r'\b%s\b' % c, p)]
        found.sort()
        for _, c in found:
            mode = 'all' if re.search(r'\ball\b|\bevery\b|\beach\b|cubes\b|containers\b', p) else 'one'
            if 'watch' in p and 'video' in p and not found:
                continue
            seq.append((c, mode))
    # dedupe consecutive duplicates
    out = []
    for s in seq:
        if not out or out[-1] != s:
            out.append(s)
    return out


# ---------------------------------------------------------------- 3-D scene analysis

def table_height(points):
    """Mode of z among points (N,3) -> table height estimate."""
    z = points[:, 2]
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return None
    lo, hi = np.percentile(z, 1), np.percentile(z, 99)
    if hi - lo < 1e-4:
        return float(np.median(z))
    bins = np.arange(lo, hi + 0.004, 0.004)
    h, e = np.histogram(z, bins=bins)
    k = int(np.argmax(h))
    sel = z[(z >= e[k] - 0.004) & (z <= e[k + 1] + 0.004)]
    return float(np.median(sel))


def cluster_points(pts, weights=None, radius=0.03):
    """Greedy xy clustering.  pts (N,3).  Returns list of dict(center, n, members)."""
    pts = np.asarray(pts, float)
    if len(pts) == 0:
        return []
    if weights is None:
        weights = np.ones(len(pts))
    order = np.argsort(-np.asarray(weights))
    clusters = []
    for i in order:
        p = pts[i]
        best, bd = None, 1e9
        for c in clusters:
            d = np.linalg.norm(c['center'][:2] - p[:2])
            if d < bd:
                best, bd = c, d
        if best is not None and bd < radius:
            best['members'].append(i)
            m = pts[best['members']]
            best['center'] = np.median(m, axis=0)
            best['n'] += 1
        else:
            clusters.append({'center': p.copy(), 'n': 1, 'members': [i]})
    # merge clusters whose centres are close
    merged = True
    while merged:
        merged = False
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if np.linalg.norm(clusters[a]['center'][:2] - clusters[b]['center'][:2]) < radius * 0.8:
                    clusters[a]['members'] += clusters[b]['members']
                    clusters[a]['n'] += clusters[b]['n']
                    clusters[a]['center'] = np.median(pts[clusters[a]['members']], axis=0)
                    clusters.pop(b)
                    merged = True
                    break
            if merged:
                break
    return clusters


def object_region(points, target_xy, table_z, min_h=0.012, radius=0.06, link=0.012):
    """From scattered surface points, extract the object region connected to target_xy.

    Returns dict(center_xy, top_z, yaw, size, pts) or None."""
    P = np.asarray(points, float)
    if len(P) == 0:
        return None
    P = P[np.all(np.isfinite(P), axis=1)]
    d = np.linalg.norm(P[:, :2] - np.asarray(target_xy)[None, :2], axis=1)
    sel = (d < radius) & (P[:, 2] > table_z + min_h) & (P[:, 2] < table_z + 0.30)
    Q = P[sel]
    dq = d[sel]
    if len(Q) < 3:
        return None
    # seed: the point closest to target among high points
    seed = int(np.argmin(dq))
    inreg = np.zeros(len(Q), bool)
    inreg[seed] = True
    frontier = [seed]
    while frontier:
        i = frontier.pop()
        dd = np.linalg.norm(Q[:, :2] - Q[i, :2], axis=1)
        nb = np.nonzero((dd < link) & (~inreg) & (np.abs(Q[:, 2] - Q[i, 2]) < 0.03))[0]
        inreg[nb] = True
        frontier.extend(nb.tolist())
    R = Q[inreg]
    top_z = float(np.percentile(R[:, 2], 90))
    top = R[R[:, 2] > top_z - 0.012]
    if len(top) < 3:
        top = R
    cxy = top[:, :2].mean(0)
    yaw, size = square_yaw(top[:, :2])
    return {'center_xy': cxy, 'top_z': top_z, 'yaw': yaw, 'size': size, 'n': int(len(R)), 'n_top': int(len(top)),
            'pts': R}


def square_yaw(xy):
    """Orientation (mod pi/2) of a roughly rectangular xy point set: minimise the
    bounding-box area over angles.  Returns (yaw in [-pi/4, pi/4), (w, h))."""
    xy = np.asarray(xy, float)
    if len(xy) < 4:
        return 0.0, (0.0, 0.0)
    c = xy - xy.mean(0)
    best = (1e9, 0.0, (0, 0))
    for a in np.deg2rad(np.arange(-45, 45, 0.75)):
        ca, sa = np.cos(a), np.sin(a)
        x = c[:, 0] * ca + c[:, 1] * sa
        y = -c[:, 0] * sa + c[:, 1] * ca
        w = np.percentile(x, 99.5) - np.percentile(x, 0.5)
        h = np.percentile(y, 99.5) - np.percentile(y, 0.5)
        if w * h < best[0]:
            best = (w * h, float(a), (float(w), float(h)))
    return best[1], best[2]


def label2d(mask):
    """4-connected components of a bool grid -> (labels, n)."""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        return lab, int(n)
    except Exception:
        pass
    H, W = mask.shape
    lab = np.zeros((H, W), int)
    n = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and lab[i, j] == 0:
                n += 1
                st = [(i, j)]
                lab[i, j] = n
                while st:
                    a, b = st.pop()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + da, b + db
                        if 0 <= x < H and 0 <= y < W and mask[x, y] and lab[x, y] == 0:
                            lab[x, y] = n
                            st.append((x, y))
    return lab, n


def segment_objects(points, table_z, cell=0.005, min_h=0.02, max_h=0.20, xy_bounds=None):
    """Height-map segmentation of objects standing on the table.
    Returns list of dict(center_xy, top_z, extent, n_cells)."""
    P = np.asarray(points, float)
    if len(P) == 0 or table_z is None:
        return []
    P = P[np.all(np.isfinite(P), 1)]
    sel = (P[:, 2] > table_z + min_h) & (P[:, 2] < table_z + max_h)
    Q = P[sel]
    if len(Q) < 5:
        return []
    lo = Q[:, :2].min(0) - 2 * cell
    hi = Q[:, :2].max(0) + 2 * cell
    if np.any(hi - lo > 3.0):
        return []
    shape = tuple(np.ceil((hi - lo) / cell).astype(int) + 1)
    ij = np.floor((Q[:, :2] - lo) / cell).astype(int)
    zmax = np.full(shape, -np.inf)
    np.maximum.at(zmax, (ij[:, 0], ij[:, 1]), Q[:, 2])
    occ = np.isfinite(zmax)
    # close small holes
    occ2 = occ.copy()
    occ2[1:, :] |= occ[:-1, :]
    occ2[:-1, :] |= occ[1:, :]
    occ2[:, 1:] |= occ[:, :-1]
    occ2[:, :-1] |= occ[:, 1:]
    occ = occ2 & (np.isfinite(zmax) | (np.roll(occ, 1, 0) & np.roll(occ, -1, 0)) | (np.roll(occ, 1, 1) & np.roll(occ, -1, 1)))
    lab, n = label2d(occ)
    out = []
    for k in range(1, n + 1):
        cells = np.argwhere(lab == k)
        if len(cells) < 6:
            continue
        xy = lo + (cells + 0.5) * cell
        zs = zmax[cells[:, 0], cells[:, 1]]
        zs = zs[np.isfinite(zs)]
        ext = xy.max(0) - xy.min(0) + cell
        out.append({'center_xy': xy.mean(0), 'top_z': float(np.percentile(zs, 90)) if len(zs) else table_z,
                    'extent': ext, 'n_cells': int(len(cells))})
    return out
