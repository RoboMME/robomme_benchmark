"""Front-camera perception: table, button, cubes, highlight rings.

Detections are clustered in world XY (from depth back-projection) so cubes that
touch in the image stay separate.  Highlight classification is done in image
space: the fraction of new white table-level pixels in a ring around each
cube's pixel mask (validated on the released demos: highlighted 0.44-0.76,
neighbours <= 0.20, others 0).
"""
import numpy as np

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None
try:
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:  # pragma: no cover
    _lsa = None

# ---- tunables (see memory/PROGRESS.md for evidence) ----
GRID_RES = 0.005           # m per XY cell for clustering
MIN_CUBE_PIX = 10          # pixels for a detection
SAT_MIN = 28               # max-min RGB for "coloured" (cubes)
WHITE_MIN = 175            # min channel for highlight white
WHITE_SAT_MAX = 45
HL_RATIO_THR = 0.32        # demo: hl 0.44-0.76, non-hl <= 0.20
HL_ON_PIX = 120            # new white px for "highlight visible" (demo on 700-1300, off 15-70)


# ----------------------------------------------------------------- utils
def label(mask, conn8=True):
    mask = np.asarray(mask, bool)
    if _ndi is not None:
        st = np.ones((3, 3), int) if conn8 else None
        lab, n = _ndi.label(mask, structure=st)
        return lab, n
    lab = np.zeros(mask.shape, np.int32)
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
                    if not conn8 and dy and dx:
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                        lab[yy, xx] = n
                        stack.append((yy, xx))
    return lab, n


def dilate(mask, r):
    mask = np.asarray(mask, bool)
    if r <= 0:
        return mask.copy()
    if _ndi is not None:
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        st = (yy * yy + xx * xx) <= r * r + 0.5
        return _ndi.binary_dilation(mask, structure=st)
    out = mask.copy()
    H, W = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx > r * r + 0.5:
                continue
            sh = np.zeros_like(mask)
            ys = slice(max(dy, 0), H + min(dy, 0))
            yd = slice(max(-dy, 0), H + min(-dy, 0))
            xs = slice(max(dx, 0), W + min(dx, 0))
            xd = slice(max(-dx, 0), W + min(-dx, 0))
            sh[ys, xs] = mask[yd, xd]
            out |= sh
    return out


def assign(cost):
    """Min-cost assignment; returns list of (row, col)."""
    cost = np.asarray(cost, float)
    if cost.size == 0:
        return []
    if _lsa is not None:
        r, c = _lsa(cost)
        return list(zip(r.tolist(), c.tolist()))
    pairs = []
    used_r, used_c = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    for r, c in order:
        if r in used_r or c in used_c:
            continue
        pairs.append((int(r), int(c)))
        used_r.add(r)
        used_c.add(c)
    return pairs


def kmeans(X, init, iters=12):
    C = np.asarray(init, float).copy()
    k = len(C)
    lab = np.zeros(len(X), int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return lab, C


def farthest_init(X, k):
    c = [X.mean(0)]
    idx = int(((X - c[0]) ** 2).sum(1).argmax())
    c = [X[idx]]
    for _ in range(k - 1):
        d = np.min([((X - ci) ** 2).sum(1) for ci in c], axis=0)
        c.append(X[int(d.argmax())])
    return np.array(c)


def _fits(X, lab, k, rmax):
    for j in range(k):
        P = X[lab == j]
        if len(P) == 0:
            continue
        r = np.sqrt(((P - P.mean(0)) ** 2).sum(1))
        if np.percentile(r, 98) > rmax:
            return False
    return True


def split_component(X, s, preds=None, kmax=6):
    """Split world-XY points of one connected blob into cube-sized clusters.

    Smallest k such that every cluster fits in one cube footprint (radius 0.707 s plus margin).
    Returns list of index arrays into X.
    """
    rmax = 0.707 * s + 0.006
    n = len(X)
    lab = np.zeros(n, int)
    if _fits(X, lab, 1, rmax):
        return [np.arange(n)]
    best = None
    for k in range(2, kmax + 1):
        if n < k * MIN_CUBE_PIX:
            break
        inits = []
        if preds is not None and len(preds) == k:
            inits.append(np.asarray(preds, float))
        inits.append(farthest_init(X, k))
        for init in inits:
            lab, C = kmeans(X, init)
            if _fits(X, lab, k, rmax):
                return [np.flatnonzero(lab == j) for j in range(k)]
            best = lab, k
    if best is None:
        return [np.arange(n)]
    lab, k = best
    return [np.flatnonzero(lab == j) for j in range(k)]


def square_yaw(xy):
    """Yaw (mod 90 deg) of a square point set via min-area bounding box."""
    if len(xy) < 6:
        return None
    c = xy - xy.mean(0)
    best = (1e9, 0.0)
    for a in np.deg2rad(np.arange(0, 90, 3.0)):
        ca, sa = np.cos(a), np.sin(a)
        x = c[:, 0] * ca + c[:, 1] * sa
        y = -c[:, 0] * sa + c[:, 1] * ca
        area = (np.percentile(x, 97) - np.percentile(x, 3)) * (np.percentile(y, 97) - np.percentile(y, 3))
        if area < best[0]:
            best = (area, a)
    return float(best[1])


def rgb_stats(rgb):
    a = rgb.astype(np.int16)
    mx = a.max(2)
    mn = a.min(2)
    return mx, mn, mx - mn


# ----------------------------------------------------------------- scene
class Scene:
    """Static front-camera knowledge established at step 0."""

    def __init__(self):
        self.model = None
        self.table_z = None
        self.cube_size = 0.04
        self.button = None          # dict(xy, cap_xy, cap_top_z, pix_mask)
        self.excl = None            # pixel mask excluded from cube detection (button)
        self.base_white = None      # white pixels at step 0 (dilated)
        self.mode = "depth"


def backproject_all(model, depth):
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W]
    P = model.backproject(uu.ravel(), vv.ravel(), depth.ravel()).reshape(H, W, 3)
    valid = (depth > 1e-4) & np.isfinite(depth)
    return P, valid


def estimate_table_z(P, valid):
    H = P.shape[0]
    sel = valid.copy()
    sel[: int(H * 0.45)] = False
    z = P[..., 2][sel]
    if z.size < 100:
        z = P[..., 2][valid]
    if z.size == 0:
        return None, None
    lo, hi = np.percentile(z, 2), np.percentile(z, 98)
    bins = np.arange(lo - 0.002, hi + 0.003, 0.002)
    if len(bins) < 3:
        return float(np.median(z)), 0.0
    hist, edges = np.histogram(z, bins=bins)
    i = int(hist.argmax())
    z0 = 0.5 * (edges[i] + edges[i + 1])
    near = z[np.abs(z - z0) < 0.006]
    return float(np.median(near)), float(np.std(near))


def find_button(rgb, P, valid, table_z):
    mx, mn, sat = rgb_stats(rgb)
    h = P[..., 2] - table_z
    H, W = h.shape
    cand = valid & (sat < 40) & (mx > 80) & (h > 0.003) & (h < 0.08)
    cand[: int(H * 0.06)] = False
    lab, n = label(cand)
    best = None
    for k in range(1, n + 1):
        m = lab == k
        a = int(m.sum())
        if a < 15 or a > 2500:
            continue
        pts = P[m]
        ext = pts[:, :2].max(0) - pts[:, :2].min(0)
        if ext.max() > 0.12 or ext.max() < 0.015:
            continue
        bright = (mn[m] > 190).mean()
        gray = ((mx[m] > 90) & (mx[m] < 195)).mean()
        score = a * (0.3 + min(bright, 0.6)) * (0.3 + min(gray, 0.6))
        ys = np.nonzero(m)[0]
        if ys.mean() < H * 0.12:
            score *= 0.2
        if best is None or score > best[0]:
            best = (score, k, m)
    if best is None:
        return None
    m = best[2]
    pts = P[m]
    hh = pts[:, 2] - table_z
    top = np.percentile(hh, 92)
    cap = pts[hh >= top - 0.006]
    return dict(xy=pts[:, :2].mean(0), cap_xy=cap[:, :2].mean(0), cap_top_z=float(table_z + top),
                plate_ext=(pts[:, :2].max(0) - pts[:, :2].min(0)).tolist(), mask=m, npix=int(m.sum()))


def cube_pixel_mask(rgb, P, valid, scene):
    mx, mn, sat = rgb_stats(rgb)
    h = P[..., 2] - scene.table_z
    s = scene.cube_size
    m = valid & (h > max(0.007, 0.22 * s)) & (h < s + 0.025) & (sat >= SAT_MIN) & (mx >= 40)
    if scene.excl is not None:
        m &= ~scene.excl
    return m


def detect_cubes(rgb, P, valid, scene, pred=None):
    """Return list of detections: dict(xy, top_xy, npix, color, yaw, pix(flat idx), top_z)."""
    H, W = valid.shape
    m = cube_pixel_mask(rgb, P, valid, scene)
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return [], m
    pts = P.reshape(-1, 3)[idx]
    xy = pts[:, :2]
    x0, y0 = xy.min(0) - 2 * GRID_RES
    gx = ((xy[:, 0] - x0) / GRID_RES).astype(int)
    gy = ((xy[:, 1] - y0) / GRID_RES).astype(int)
    grid = np.zeros((gy.max() + 3, gx.max() + 3), bool)
    grid[gy, gx] = True
    # oblique view: neighbouring image rows can be > 1 cell apart on top faces -> bridge 1-cell gaps
    gridd = dilate(grid, 1)
    glab, n = label(gridd)
    glab = np.where(grid, glab, 0)
    plab = glab[gy, gx]
    s = scene.cube_size
    cell_area = GRID_RES * GRID_RES
    dets = []
    rgbf = rgb.reshape(-1, 3).astype(float)
    pred = np.asarray(pred, float).reshape(-1, 2) if pred is not None and len(pred) else np.zeros((0, 2))
    # which component does each predicted centre fall into?
    pred_comp = []
    for p in pred:
        d2 = ((xy - p) ** 2).sum(1)
        j = int(d2.argmin())
        pred_comp.append(plab[j] if d2[j] < (0.6 * s) ** 2 else -1)
    pred_comp = np.asarray(pred_comp, int)
    for k in range(1, n + 1):
        sel = np.flatnonzero(plab == k)
        if sel.size < MIN_CUBE_PIX:
            continue
        cxy = xy[sel]
        pin = np.flatnonzero(pred_comp == k)
        groups = split_component(cxy, s, pred[pin] if len(pin) else None)
        groups = [sel[g] for g in groups]
        merged = False
        first_new = len(dets)
        for g in groups:
            if g.size < MIN_CUBE_PIX:
                continue
            gp = pts[g]
            hz = gp[:, 2]
            ztop = np.percentile(hz, 90)
            topm = hz >= ztop - 0.007
            top_xy = gp[topm, :2].mean(0) if topm.sum() >= 5 else gp[:, :2].mean(0)
            col = rgbf[idx[g][topm]] if topm.sum() >= 5 else rgbf[idx[g]]
            c = col.mean(0)
            chroma = c / max(c.sum(), 1.0)
            dets.append(dict(xy=top_xy, all_xy=gp[:, :2].mean(0), npix=int(g.size), color=chroma,
                             rgb=c, yaw=square_yaw(gp[topm, :2]) if topm.sum() >= 8 else None,
                             pix=idx[g], top_z=float(ztop), merged=merged))
        # ambiguous split: clusters of one blob closer than a cube width (interpenetrating / occluding cubes)
        new = dets[first_new:]
        for a in range(len(new)):
            for b in range(a + 1, len(new)):
                if np.linalg.norm(np.asarray(new[a]["all_xy"]) - np.asarray(new[b]["all_xy"])) < 0.85 * s:
                    new[a]["merged"] = True
                    new[b]["merged"] = True
    dets = merge_fragments(dets, s)
    return dets, m


def merge_fragments(dets, s):
    """Fold small detections lying inside the footprint of a much larger neighbour into it."""
    if len(dets) < 2:
        return dets
    big = np.median([d["npix"] for d in dets])
    keep = []
    order = sorted(range(len(dets)), key=lambda i: -dets[i]["npix"])
    for i in order:
        d = dets[i]
        host = None
        if d["npix"] < 0.35 * big:
            for k in keep:
                if np.linalg.norm(np.asarray(k["all_xy"]) - np.asarray(d["all_xy"])) < 0.75 * s and k["npix"] > 2 * d["npix"]:
                    host = k
                    break
        if host is None:
            keep.append(d)
        else:
            host["pix"] = np.concatenate([host["pix"], d["pix"]])
            host["npix"] = int(host["pix"].size)
    return keep


def white_new_mask(rgb, P, valid, scene):
    mx, mn, sat = rgb_stats(rgb)
    h = P[..., 2] - scene.table_z
    w = valid & (mn >= WHITE_MIN) & (sat <= WHITE_SAT_MAX) & (h < 0.015)
    if scene.base_white is not None:
        w &= ~scene.base_white
    return w


WR_IN, WR_OUT, WR_Q = 0.65, 0.85, 20


def world_ring_ratios(dets, cube_mask, white_new, P, valid, scene, r_in=None, r_out=None):
    """Per detection: fraction of visible table points at world distance [r_in*s, r_out*s] from the cube
    centre that are newly white.  Highlight discs are ~2x the cube width (demo frames), so this annulus lies
    inside a highlighted cube's own disc and is only partly reached by a neighbour's disc."""
    s = scene.cube_size
    r_in = WR_IN if r_in is None else r_in
    r_out = WR_OUT if r_out is None else r_out
    h = P[..., 2] - scene.table_z
    tab = valid & (h < 0.012) & ~cube_mask
    if scene.excl is not None:
        tab &= ~scene.excl
    idx = np.flatnonzero(tab)
    xy = P.reshape(-1, 3)[idx, :2]
    wn = white_new.reshape(-1)[idx]
    out = []
    for d in dets:
        c = np.asarray(d["xy"], float)
        dxy = xy - c
        r = np.sqrt((dxy ** 2).sum(1))
        ring = (r >= r_in * s) & (r <= r_out * s)
        t = int(ring.sum())
        w = int((ring & wn).sum())
        # own disc covers the annulus all around; a neighbour's disc only the side facing it
        ang = np.arctan2(dxy[ring, 1], dxy[ring, 0])
        sec = ((ang + np.pi) / (2 * np.pi) * 8).astype(int) % 8
        wr = wn[ring]
        fr = []
        for k in range(8):
            mk = sec == k
            if mk.sum() >= 3:
                fr.append(float(wr[mk].mean()))
        score = float(np.percentile(fr, WR_Q)) if len(fr) >= 4 else w / max(t, 1)
        out.append((score, w, t))
    return out


def ring_ratios(dets, cube_mask, white_new, P, valid, scene, shape):
    """Per detection: fraction of table-level ring pixels that are newly white."""
    H, W = shape
    h = P[..., 2] - scene.table_z
    tablelev = valid & (h < 0.012) & ~cube_mask
    if scene.excl is not None:
        tablelev &= ~scene.excl
    out = []
    for d in dets:
        m = np.zeros(H * W, bool)
        m[d["pix"]] = True
        m = m.reshape(H, W)
        ys, xs = np.nonzero(m)
        y0, y1 = max(ys.min() - 9, 0), min(ys.max() + 10, H)
        x0, x1 = max(xs.min() - 9, 0), min(xs.max() + 10, W)
        sub = m[y0:y1, x0:x1]
        r2 = int(round(np.clip(0.4 * np.sqrt(d["npix"]), 3, 8)))
        ring = dilate(sub, r2) & ~dilate(sub, 1)
        ring &= tablelev[y0:y1, x0:x1]
        t = int(ring.sum())
        wcount = int((ring & white_new[y0:y1, x0:x1]).sum())
        out.append((wcount / max(t, 1), wcount, t))
    return out
