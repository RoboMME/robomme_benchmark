"""Pure numpy/PIL perception for PatternLock: target blobs, grid fitting, red activations."""
import numpy as np

def _box_blur(a, r):
    # separable box blur with edge padding
    k = 2 * r + 1
    p = np.pad(a, r, mode='edge').astype(np.float64)
    c = np.cumsum(p, axis=0); c = np.vstack([np.zeros((1, c.shape[1])), c])
    a1 = (c[k:] - c[:-k]) / k
    c = np.cumsum(a1, axis=1); c = np.hstack([np.zeros((c.shape[0], 1)), c])
    return (c[:, k:] - c[:, :-k]) / k

def label(mask):
    """4-connected components; returns list of index arrays (ys, xs)."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]; seen[y0, x0] = True; py = []; px = []
        while stack:
            y, x = stack.pop(); py.append(y); px.append(x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True; stack.append((yy, xx))
        comps.append((np.array(py), np.array(px)))
    return comps

def _dilate(m, r=1):
    p = np.pad(m, r)
    out = np.zeros_like(m)
    H, W = m.shape
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out |= p[dy:dy + H, dx:dx + W]
    return out

def _erode(m, r=1):
    return ~_dilate(~m, r)

def target_mask(rgb, diff_min=40, sat_max=50, v_min=100, red_ok=True):
    im = rgb.astype(np.float64)
    bgc = np.stack([_box_blur(im[..., k], 12) for k in range(3)], -1)
    diff = np.linalg.norm(im - bgc, axis=2)
    v = im.max(2); s = v - im.min(2)
    m = (diff > diff_min) & (s < sat_max) & (v > v_min)
    if red_ok:
        m |= redness(rgb) & (diff > 30)
    # drop speckle, then light closing to merge ring fragments
    keep = np.zeros_like(m)
    for ys, xs in label(m):
        if len(ys) >= 5:
            keep[ys, xs] = True
    m = _erode(_dilate(keep, 1), 1)
    return m

def detect_blobs(rgb, min_area=8, max_area=220, roi=None):
    m = target_mask(rgb)
    if roi is not None:
        u0, v0, u1, v1 = roi
        mm = np.zeros_like(m); mm[v0:v1, u0:u1] = m[v0:v1, u0:u1]; m = mm
    out = []
    for ys, xs in label(m):
        a = len(ys)
        if min_area <= a <= max_area:
            h = ys.max() - ys.min() + 1; w = xs.max() - xs.min() + 1
            if h > 18 or w > 22:
                continue
            out.append((xs.mean(), ys.mean(), a))
    return out

def redness(rgb):
    im = rgb.astype(np.int32)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return (r > 110) & (r - g > 80) & (np.abs(g - b) < 25) & (g < 0.45 * r)

# ---------- lattice fitting (works in image or world-plane coordinates) ----------

def fit_lattice(pts, n=5):
    """pts: (N,2). Returns dict with indices (N,2 ints or -1), basis, origin, residual.
    Finds two lattice directions from nearest-neighbour vectors, assigns integer coords.
    Index convention: returned 'ij' with i along axis A, j along axis B (sign fixed later)."""
    pts = np.asarray(pts, float)
    N = len(pts)
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2); np.fill_diagonal(d, np.inf)
    nn = np.sort(d, axis=1)[:, 0]
    sp = np.median(nn)
    vecs = []
    for i in range(N):
        for j in range(N):
            if i != j and d[i, j] < 1.35 * sp:
                vecs.append(pts[j] - pts[i])
    vecs = np.array(vecs)
    ang = np.arctan2(vecs[:, 1], vecs[:, 0])
    # dominant angle mod 90 deg via circular mean of 4*angle
    z = np.exp(4j * ang).mean()
    th = np.angle(z) / 4
    e1 = np.array([np.cos(th), np.sin(th)]); e2 = np.array([-np.sin(th), np.cos(th)])
    # spacing along each axis
    p1 = vecs @ e1; p2 = vecs @ e2
    s1 = np.median(np.abs(p1[np.abs(p1) > np.abs(p2)])); s2 = np.median(np.abs(p2[np.abs(p2) > np.abs(p1)]))
    a = pts @ e1 / s1; b = pts @ e2 / s2
    # robust offsets
    oa = np.angle(np.exp(2j * np.pi * a).mean()) / (2 * np.pi)
    ob = np.angle(np.exp(2j * np.pi * b).mean()) / (2 * np.pi)
    ia = np.round(a - oa).astype(int); ib = np.round(b - ob).astype(int)
    res = np.hypot(a - oa - ia, b - ob - ib)
    return dict(ia=ia, ib=ib, theta=th, s1=s1, s2=s2, res=res, e1=e1, e2=e2, oa=oa, ob=ob)

def index_lattice_image(pts, n=5):
    """Assign integer lattice coords to image points (handles perspective) by BFS over
    neighbour links classified into two axis directions. Returns (idx (N,2) or None per pt, axes)."""
    pts = np.asarray(pts, float); N = len(pts)
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2); np.fill_diagonal(d, np.inf)
    nn = d.min(1)
    # candidate link vectors
    links = []
    for i in range(N):
        for j in range(N):
            if i != j and d[i, j] < 1.55 * nn[i] and d[i, j] < 1.55 * nn[j]:
                links.append((i, j))
    vec = np.array([pts[j] - pts[i] for i, j in links])
    ang = np.arctan2(vec[:, 1], vec[:, 0]) % np.pi
    # two axis directions: histogram-based clustering on angle mod pi
    hist, edges = np.histogram(ang, bins=36, range=(0, np.pi))
    hs = hist + np.roll(hist, 1) + np.roll(hist, -1)
    a1 = (np.argmax(hs) + 0.5) * np.pi / 36
    diff1 = np.abs(((ang - a1 + np.pi / 2) % np.pi) - np.pi / 2)
    a1 = a1 + np.median(((ang - a1 + np.pi / 2) % np.pi - np.pi / 2)[diff1 < 0.3])
    diff1 = np.abs(((ang - a1 + np.pi / 2) % np.pi) - np.pi / 2)
    rest = ang[diff1 > 0.5]
    h2, _ = np.histogram(rest, bins=36, range=(0, np.pi)); h2 = h2 + np.roll(h2, 1) + np.roll(h2, -1)
    a2 = (np.argmax(h2) + 0.5) * np.pi / 36
    diff2 = np.abs(((ang - a2 + np.pi / 2) % np.pi) - np.pi / 2)
    a2 = a2 + np.median(((ang - a2 + np.pi / 2) % np.pi - np.pi / 2)[diff2 < 0.3])
    u1 = np.array([np.cos(a1), np.sin(a1)]); u2 = np.array([np.cos(a2), np.sin(a2)])
    # BFS
    idx = [None] * N
    seed = int(np.argmin(np.linalg.norm(pts - np.median(pts, 0), axis=1)))
    idx[seed] = (0, 0); queue = [seed]
    adj = {}
    for (i, j), v in zip(links, vec):
        an = np.arctan2(v[1], v[0])
        c1 = np.dot(v, u1) / np.linalg.norm(v); c2 = np.dot(v, u2) / np.linalg.norm(v)
        if abs(c1) > 0.9:
            step = (int(np.sign(c1)), 0)
        elif abs(c2) > 0.9:
            step = (0, int(np.sign(c2)))
        else:
            continue
        adj.setdefault(i, []).append((j, step))
    while queue:
        i = queue.pop(0)
        for j, (s1, s2) in adj.get(i, []):
            if idx[j] is None:
                idx[j] = (idx[i][0] + s1, idx[i][1] + s2); queue.append(j)
    return idx, (u1, u2)

def grid_from_blobs_image(blobs, n=5):
    """Return (grid (n,n,2) image coords with NaN for missing, info). Axis order: rows = the
    lattice axis closer to image-horizontal lines (i.e. row index increases downward in image),
    cols increase to the right in image. Only meaningful for near-axis-aligned grids."""
    pts = np.array([(u, v) for u, v, a in blobs])
    idx, (u1, u2) = index_lattice_image(pts, n)
    return pts, idx, (u1, u2)


def merge_close(blobs, dist=5.0):
    """Merge blob centroids closer than dist (area-weighted)."""
    bl = [list(b) for b in blobs]
    changed = True
    while changed:
        changed = False
        for i in range(len(bl)):
            for j in range(i + 1, len(bl)):
                if np.hypot(bl[i][0] - bl[j][0], bl[i][1] - bl[j][1]) < dist:
                    a = bl[i][2] + bl[j][2]
                    bl[i] = [(bl[i][0] * bl[i][2] + bl[j][0] * bl[j][2]) / a,
                             (bl[i][1] * bl[i][2] + bl[j][1] * bl[j][2]) / a, a]
                    del bl[j]; changed = True; break
            if changed:
                break
    return [tuple(b) for b in bl]

def homography(src, dst):
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.array(A))
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]

def apply_h(H, pts):
    pts = np.asarray(pts, float)
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return p[:, :2] / p[:, 2:3]

def axis_aligned_grid(blobs, n=5):
    """For a roughly axis-aligned grid in the image: choose n*n blobs forming n rows of n.
    Returns grid (n,n,2) [row, col] with row 0 = top of image (far from camera), col 0 = left.
    Raises ValueError if not found."""
    pts = np.array([(u, v) for u, v, a in blobs], float)
    best = None
    order = np.argsort(pts[:, 1])
    P = pts[order]
    # cluster rows by v-gaps
    rows = [[P[0]]]
    for p in P[1:]:
        if p[1] - rows[-1][-1][1] < 4.5:
            rows[-1].append(p)
        else:
            rows.append([p])
    rows = [np.array(r) for r in rows]
    # take sequences of n consecutive rows each with >= n points; select n points by u spacing
    cands = []
    for s in range(len(rows) - n + 1):
        sel = rows[s:s + n]
        if all(len(r) >= n for r in sel):
            g = []
            ok = True
            for r in sel:
                r = r[np.argsort(r[:, 0])]
                if len(r) > n:
                    # pick the n consecutive points with most uniform spacing
                    bestk, bestc = None, 1e9
                    for k in range(len(r) - n + 1):
                        dd = np.diff(r[k:k + n, 0]); c = dd.std() / dd.mean()
                        if c < bestc:
                            bestk, bestc = k, c
                    r = r[bestk:bestk + n]
                dd = np.diff(r[:, 0])
                if dd.min() < 0.5 * dd.max():
                    ok = False
                g.append(r)
            if ok:
                g = np.array(g)
                cv = np.std(np.diff(g[:, :, 0], axis=1)) / np.mean(np.diff(g[:, :, 0], axis=1))
                cands.append((cv, g))
    if not cands:
        raise ValueError('axis-aligned grid not found')
    cands.sort(key=lambda c: c[0])
    return cands[0][1]

def fit_grid_plane(W, n=5, res_max=0.3):
    """W: (N,2) points in a metric (or rectified) plane. Returns dict:
    grid (n,n,2) fitted positions indexed [a,b] along lattice axes e1,e2 (a along e1),
    count of supporting points, affine (origin, A1, A2), rms residual, member mapping."""
    W = np.asarray(W, float)
    keep = np.ones(len(W), bool)
    for _ in range(3):
        L = fit_lattice(W[keep])
        e1, e2, s1, s2 = L['e1'], L['e2'], L['s1'], L['s2']
        a = W @ e1 / s1 - L['oa']; b = W @ e2 / s2 - L['ob']
        ia = np.round(a).astype(int); ib = np.round(b).astype(int)
        res = np.hypot(a - ia, b - ib)
        newkeep = res < res_max
        if newkeep.sum() < 6:
            break
        keep = newkeep
    # best n x n window
    best = None
    for a0 in range(ia[keep].min(), ia[keep].max() - n + 2):
        for b0 in range(ib[keep].min(), ib[keep].max() - n + 2):
            inwin = keep & (ia >= a0) & (ia < a0 + n) & (ib >= b0) & (ib < b0 + n)
            cells = set(zip(ia[inwin], ib[inwin]))
            c = len(cells)
            if best is None or c > best[0]:
                best = (c, a0, b0, inwin)
    c, a0, b0, inwin = best
    # affine least squares: P = O + i*A1 + j*A2
    I = ia[inwin] - a0; J = ib[inwin] - b0
    X = np.stack([np.ones_like(I), I, J], 1).astype(float)
    coef, *_ = np.linalg.lstsq(X, W[inwin], rcond=None)
    O, A1, A2 = coef
    pred = X @ coef
    rms = float(np.sqrt(((pred - W[inwin]) ** 2).sum(1).mean()))
    grid = np.array([[O + i * A1 + j * A2 for j in range(n)] for i in range(n)])
    members = {}
    for k in np.nonzero(inwin)[0]:
        members.setdefault((int(ia[k] - a0), int(ib[k] - b0)), []).append(int(k))
    return dict(grid=grid, count=c, O=O, A1=A1, A2=A2, rms=rms, members=members,
                theta=float(L['theta']))
