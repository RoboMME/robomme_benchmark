"""RGB-D perception: back-projection, fused top-down height map, object segmentation.

All tunables below are priors (no scene of this task was ever observed before freezing).
"""
import math
from collections import deque

import numpy as np

CELL = 0.005            # height-map cell size (m)
H_MIN = 0.010           # cell counts as elevated above table (m)
H_OBJ_MAX = 0.20        # points higher than this are treated as robot, not objects
SAT_MIN = 0.30          # HSV saturation for a "coloured" cell
VAL_MIN = 0.15
CUBE_EXT = (0.018, 0.080)   # plausible cube footprint extent (m)
BIN_MIN_EXT = 0.085     # a component this wide in some direction is a bin candidate

COLOR_HUES = {          # hue centre in degrees
    "red": 0, "orange": 28, "yellow": 55, "green": 120, "cyan": 180, "blue": 225,
    "purple": 275, "violet": 275, "magenta": 305, "pink": 330,
}
NEUTRAL_COLORS = ("white", "gray", "grey", "black", "brown")


# --------------------------------------------------------------------------- projection
def backproject(depth, calib, variant):
    """depth (H,W) metres -> world points (H,W,3). variant = (convention, pixel_offset)."""
    fx, fy, cx, cy = calib["K"]
    H, W = depth.shape
    conv, off = variant
    u = np.arange(W)[None, :] + off
    v = np.arange(H)[:, None] + off
    d = depth
    x = (u - cx) / fx * d
    y = (v - cy) / fy * d
    if conv == "opengl":
        pc = np.stack([x, -y, -d], -1)
    elif conv == "opencv":
        pc = np.stack([x, y, d], -1)
    elif conv == "sapien":       # x forward, y left, z up
        pc = np.stack([d, -x, -y], -1)
    else:
        raise ValueError(conv)
    R, t = calib["R"], calib["t"]
    return pc @ R.T + t


def point_at(depth, calib, variant, u, v):
    fx, fy, cx, cy = calib["K"]
    conv, off = variant
    d = float(depth[int(v), int(u)])
    x = (u + off - cx) / fx * d
    y = (v + off - cy) / fy * d
    if conv == "opengl":
        pc = np.array([x, -y, -d])
    elif conv == "opencv":
        pc = np.array([x, y, d])
    else:
        pc = np.array([d, -x, -y])
    return calib["R"] @ pc + calib["t"]


VARIANTS = [("opengl", 0.0), ("opengl", 0.5), ("opencv", 0.0), ("opencv", 0.5), ("sapien", 0.0)]


def choose_variant(depth, calib, samples):
    """samples: list of (u, v, xyz_tool). Return (variant, mean_err)."""
    best = (None, 1e9)
    for var in VARIANTS:
        errs = []
        for u, v, xyz in samples:
            if xyz is None or depth[int(v), int(u)] <= 0:
                continue
            errs.append(np.linalg.norm(point_at(depth, calib, var, u, v) - xyz))
        if len(errs) >= 2:
            e = float(np.median(errs))
            if e < best[1]:
                best = (var, e)
    return best


def sample_pixels(depth, n=8, margin=40):
    H, W = depth.shape
    out = []
    rng = np.random.RandomState(0)
    tries = 0
    while len(out) < n and tries < 500:
        tries += 1
        u = int(rng.randint(margin, W - margin))
        v = int(rng.randint(margin, H - margin))
        win = depth[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
        if depth[v, u] > 0 and (win.max() - win.min()) < 0.01:
            out.append((u, v))
    return out


def edge_mask(depth, thr_rel=0.02):
    d = depth
    m = np.zeros_like(d, dtype=bool)
    dd = np.abs(np.diff(d, axis=0))
    m[1:] |= dd > thr_rel * np.maximum(d[1:], 0.2)
    m[:-1] |= dd > thr_rel * np.maximum(d[:-1], 0.2)
    dd = np.abs(np.diff(d, axis=1))
    m[:, 1:] |= dd > thr_rel * np.maximum(d[:, 1:], 0.2)
    m[:, :-1] |= dd > thr_rel * np.maximum(d[:, :-1], 0.2)
    return m


def cloud_from(depth, rgb, calib, variant, max_range=2.5):
    pts = backproject(depth, calib, variant)
    valid = (depth > 0.02) & (depth < max_range) & ~edge_mask(depth)
    P = pts[valid]
    C = rgb[valid].astype(float) / 255.0 if rgb is not None and rgb.shape[:2] == depth.shape else \
        np.full((P.shape[0], 3), 0.5)
    return P, C


# --------------------------------------------------------------------------- colour
def rgb_to_hsv(c):
    c = np.asarray(c, dtype=float)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx = np.max(c, -1)
    mn = np.min(c, -1)
    diff = mx - mn
    h = np.zeros_like(mx)
    nz = diff > 1e-6
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = (60 * ((g[rm] - b[rm]) / diff[rm]) + 360) % 360
    h[gm] = 60 * ((b[gm] - r[gm]) / diff[gm]) + 120
    h[bm] = 60 * ((r[bm] - g[bm]) / diff[bm]) + 240
    s = np.where(mx > 1e-6, diff / np.maximum(mx, 1e-6), 0)
    return h, s, mx


def hue_class(h):
    names = list(COLOR_HUES)
    centers = np.array([COLOR_HUES[n] for n in names])
    d = np.abs(((h[..., None] - centers) + 180) % 360 - 180)
    idx = np.argmin(d, -1)
    return idx, names


def hue_dist(h, name):
    c = COLOR_HUES.get(name)
    if c is None:
        return 180.0
    return abs(((h - c) + 180) % 360 - 180)


# --------------------------------------------------------------------------- table
def table_height(P, center_xy=None, radius=0.6):
    z = P[:, 2]
    sel = np.isfinite(z) & (z > -0.5) & (z < 0.6)
    if center_xy is not None:
        sel &= np.hypot(P[:, 0] - center_xy[0], P[:, 1] - center_xy[1]) < radius
    z = z[sel]
    if z.size < 50:
        return 0.0
    hist, edges = np.histogram(z, bins=np.arange(-0.5, 0.6, 0.004))
    i = int(np.argmax(hist))
    zc = z[(z >= edges[i] - 0.004) & (z <= edges[i + 1] + 0.004)]
    return float(np.median(zc))


# --------------------------------------------------------------------------- height map
class HeightMap:
    def __init__(self, x0, y0, nx, ny, table_z):
        self.x0, self.y0, self.nx, self.ny, self.table_z = x0, y0, nx, ny, table_z
        self.h = np.full((nx, ny), -1.0)            # max object height (<= H_OBJ_MAX)
        self.cnt = np.zeros((nx, ny))
        self.seen = np.zeros((nx, ny), dtype=bool)  # any observation (table or object)
        self.tall = np.zeros((nx, ny))              # points above H_OBJ_MAX (robot)
        self.rgb_sum = np.zeros((nx, ny, 3))
        self.rgb_n = np.zeros((nx, ny))

    def idx(self, x, y):
        return ((np.asarray(x) - self.x0) / CELL).astype(int), ((np.asarray(y) - self.y0) / CELL).astype(int)

    def xy(self, i, j):
        return self.x0 + (np.asarray(i) + 0.5) * CELL, self.y0 + (np.asarray(j) + 0.5) * CELL

    def add(self, P, C, exclude=None):
        h = P[:, 2] - self.table_z
        i, j = self.idx(P[:, 0], P[:, 1])
        ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny) & (h > -0.03) & (h < 0.6)
        if exclude is not None:
            ok &= ~exclude(P)
        i, j, h, C = i[ok], j[ok], h[ok], C[ok]
        self.seen[i, j] = True
        tall = h > H_OBJ_MAX
        np.add.at(self.tall, (i[tall], j[tall]), 1)
        o = (h > 0.004) & ~tall
        i, j, h, C = i[o], j[o], h[o], C[o]
        np.maximum.at(self.h, (i, j), h)
        np.add.at(self.cnt, (i, j), 1)
        top = h >= self.h[i, j] - 0.012
        np.add.at(self.rgb_sum, (i[top], j[top]), C[top])
        np.add.at(self.rgb_n, (i[top], j[top]), 1)

    def colors(self):
        return self.rgb_sum / np.maximum(self.rgb_n, 1)[..., None]


def make_heightmap(clouds, table_z, center_xy, half=0.55, exclude=None):
    x0 = center_xy[0] - half
    y0 = center_xy[1] - half
    n = int(2 * half / CELL)
    hm = HeightMap(x0, y0, n, n, table_z)
    for cl in clouds:
        P, C = cl[0], cl[1]
        splat = cl[2] if len(cl) > 2 else 0
        if splat:
            offs = [(a, b) for a in range(-splat, splat + 1) for b in range(-splat, splat + 1)]
            P = np.concatenate([P + np.array([a * CELL, b * CELL, 0]) for a, b in offs])
            C = np.concatenate([C] * len(offs))
        hm.add(P, C, exclude)
    return hm


# --------------------------------------------------------------------------- components
def label(mask, conn8=True):
    nx, ny = mask.shape
    lab = np.zeros((nx, ny), dtype=int)
    cur = 0
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if conn8:
        nbrs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    idx = np.argwhere(mask)
    for a, b in idx:
        if lab[a, b]:
            continue
        cur += 1
        lab[a, b] = cur
        q = deque([(a, b)])
        while q:
            x, y = q.popleft()
            for dx, dy in nbrs:
                u, v = x + dx, y + dy
                if 0 <= u < nx and 0 <= v < ny and mask[u, v] and not lab[u, v]:
                    lab[u, v] = cur
                    q.append((u, v))
    return lab, cur


def dilate(mask, r=1):
    out = mask.copy()
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            sh = np.zeros_like(mask)
            xs = slice(max(dx, 0), mask.shape[0] + min(dx, 0))
            xd = slice(max(-dx, 0), mask.shape[0] + min(-dx, 0))
            ys = slice(max(dy, 0), mask.shape[1] + min(dy, 0))
            yd = slice(max(-dy, 0), mask.shape[1] + min(-dy, 0))
            sh[xs, ys] = mask[xd, yd]
            out |= sh
    return out


def min_area_yaw(pts_xy):
    """Edge orientation (rad, in [-pi/4, pi/4)) and side lengths of min-area rectangle."""
    if len(pts_xy) < 3:
        return 0.0, (0.0, 0.0)
    c = pts_xy - pts_xy.mean(0)
    best = (1e9, 0.0, (0, 0))
    for deg in np.arange(-45, 45, 2.0):
        a = math.radians(deg)
        R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        q = c @ R.T
        ext = q.max(0) - q.min(0)
        area = ext[0] * ext[1]
        if area < best[0]:
            best = (area, a, (float(ext[0]), float(ext[1])))
    return best[1], best[2]


class Obj:
    def __repr__(self):
        return "%s(%s @%.3f,%.3f top=%.3f ext=%.3fx%.3f n=%d hue=%.0f sat=%.2f)" % (
            self.kind, self.color, self.x, self.y, self.top, self.ext[0], self.ext[1], self.n,
            self.hue, self.sat)


def _obj_from_cells(hm, cells, kind, color):
    o = Obj()
    xs, ys = hm.xy(cells[:, 0], cells[:, 1])
    hs = hm.h[cells[:, 0], cells[:, 1]]
    cols = hm.colors()[cells[:, 0], cells[:, 1]]
    top_thr = np.percentile(hs, 60)
    topm = hs >= top_thr - 0.004
    o.cells = cells
    o.kind, o.color = kind, color
    o.x, o.y = float(np.mean(xs)), float(np.mean(ys))
    o.bbox = (float(xs.min() - CELL / 2), float(xs.max() + CELL / 2), float(ys.min() - CELL / 2),
              float(ys.max() + CELL / 2))
    o.cx, o.cy = (o.bbox[0] + o.bbox[1]) / 2, (o.bbox[2] + o.bbox[3]) / 2
    o.top = float(np.percentile(hs, 90))
    o.hmax = float(hs.max())
    yaw, ext = min_area_yaw(np.stack([xs, ys], 1))
    o.yaw = yaw
    o.ext = (max(ext) + CELL, min(ext) + CELL)
    o.n = len(cells)
    mc = cols[topm].mean(0) if topm.any() else cols.mean(0)
    h, s, v = rgb_to_hsv(mc)
    o.hue, o.sat, o.val = float(h), float(s), float(v)
    o.rgb = mc
    return o


def segment(hm, tcp_xy=None, tcp_h=None, log=print):
    """Return dict(bin=Obj|None, cubes=[Obj], others=[Obj], all=[Obj])."""
    elev = (hm.h > H_MIN) & (hm.cnt >= 1)
    # suppress robot gripper column if the TCP is low
    if tcp_xy is not None and tcp_h is not None and tcp_h < 0.25:
        i, j = hm.idx(tcp_xy[0], tcp_xy[1])
        r = int(0.07 / CELL)
        ii, jj = np.ogrid[:hm.nx, :hm.ny]
        elev &= ((ii - i) ** 2 + (jj - j) ** 2) > r * r
    cols = hm.colors()
    H, S, V = rgb_to_hsv(cols)
    colored = elev & (S > SAT_MIN) & (V > VAL_MIN) & (hm.rgb_n > 0)
    cls_idx, names = hue_class(H)

    lab, n = label(dilate(elev, 1) & elev | elev, True)
    comps = []
    for k in range(1, n + 1):
        cells = np.argwhere(lab == k)
        if len(cells) < 3:
            continue
        o = _obj_from_cells(hm, cells, "obj", None)
        tall_frac = float((dilate(lab == k, 2) & (hm.tall > 0)).sum()) / max(1, len(cells))
        o.tall_frac = tall_frac
        o.colored_frac = float(colored[cells[:, 0], cells[:, 1]].mean())
        if o.hmax > 0.17 and tall_frac > 0.05:
            o.kind = "robot"
        comps.append(o)

    # bin: largest non-robot component that is wide
    cand = [c for c in comps if c.kind != "robot" and c.ext[0] >= BIN_MIN_EXT and c.ext[1] >= 0.05]
    for c in cand:
        # hollowness: share of cells inside the inner 60% of the bbox lower than half the rim
        x0, x1, y0, y1 = c.bbox
        ii, jj = hm.idx(np.array([x0 + 0.2 * (x1 - x0), x1 - 0.2 * (x1 - x0)]),
                        np.array([y0 + 0.2 * (y1 - y0), y1 - 0.2 * (y1 - y0)]))
        sub_h = hm.h[max(0, ii[0]):ii[1] + 1, max(0, jj[0]):jj[1] + 1]
        sub_seen = hm.seen[max(0, ii[0]):ii[1] + 1, max(0, jj[0]):jj[1] + 1]
        rim = float(np.percentile(hm.h[c.cells[:, 0], c.cells[:, 1]], 95))
        low = (sub_h < 0.5 * rim) & sub_seen
        c.hollow = float(low.sum()) / max(1, sub_seen.sum())
    binobj = None
    if cand:
        binobj = max(cand, key=lambda c: c.n * min(1.0, c.ext[1] / 0.08) * (0.4 + c.hollow))
        binobj.kind = "bin"
        # bin footprint from its non-coloured / dominant cells only (cubes beside it may merge)
        cells = binobj.cells
        sub = cells[~colored[cells[:, 0], cells[:, 1]]]
        if len(sub) > 0.4 * len(cells) and len(sub) > 20:
            b2 = _obj_from_cells(hm, sub, "bin", None)
            binobj.bbox, binobj.cx, binobj.cy, binobj.ext = b2.bbox, b2.cx, b2.cy, b2.ext
            binobj.yaw = b2.yaw
        log("bin candidates: %s" % ["%.3fx%.3f n=%d hollow=%.2f" % (c.ext[0], c.ext[1], c.n, c.hollow)
                                     for c in cand])
        # rim height: high percentile of the wall cells
        binobj.rim = float(np.percentile(hm.h[cells[:, 0], cells[:, 1]], 95))

    # cubes: per-hue components of coloured cells
    cubes = []
    for ci, name in enumerate(names):
        m = colored & (cls_idx == ci)
        if m.sum() < 3:
            continue
        lab2, n2 = label(m, True)
        for k in range(1, n2 + 1):
            cells = np.argwhere(lab2 == k)
            if len(cells) < 12:
                continue
            o = _obj_from_cells(hm, cells, "cube", name)
            if o.ext[1] < 0.015:
                continue
            parent = lab[cells[0, 0], cells[0, 1]]
            par = next((c for c in comps if lab[c.cells[0, 0], c.cells[0, 1]] == parent), None)
            o.parent_kind = par.kind if par is not None else None
            o.parent_n = par.n if par is not None else 0
            o.in_bin = False
            if binobj is not None:
                x0, x1, y0, y1 = binobj.bbox
                o.in_bin = (x0 - 0.005 <= o.x <= x1 + 0.005) and (y0 - 0.005 <= o.y <= y1 + 0.005)
            if par is not None and par.kind == "robot":
                continue
            ok_ext = CUBE_EXT[0] <= o.ext[0] <= CUBE_EXT[1] * 1.6 and o.ext[1] >= 0.012
            ok_h = 0.012 <= o.top <= 0.10
            o.score = (1.0 if ok_ext else 0.0) + (1.0 if ok_h else 0.0)
            if not ok_h:
                continue
            if o.ext[0] > 0.11:
                continue
            cubes.append(o)
    # size consistency: cubes are assumed to share one size; outliers (e.g. a button cap) are flagged
    real = [o for o in cubes if not o.in_bin]
    for o in cubes:
        o.odd = False
    if len(real) >= 3:
        mt = float(np.median([o.top for o in real]))
        me = float(np.median([o.ext[0] for o in real]))
        for o in real:
            if abs(o.top - mt) > 0.18 * mt + 0.003 or abs(o.ext[0] - me) > 0.3 * me + 0.004:
                o.odd = True
    # a "cube" whose parent object is much bigger and mostly uncoloured is a part of something else
    # (e.g. a button cap on a base) unless that parent is the bin
    for o in cubes:
        o.part_of_other = False
    for c in comps:
        if c.kind in ("robot", "bin"):
            continue
        kids = [o for o in cubes if lab[o.cells[0, 0], o.cells[0, 1]] == lab[c.cells[0, 0], c.cells[0, 1]]]
        kid_cells = sum(o.n for o in kids)
        if kids and c.n > 1.8 * kid_cells and c.colored_frac < 0.55:
            for o in kids:
                o.part_of_other = True
    others = []
    for c in comps:
        if c.kind in ("robot", "bin"):
            continue
        kids = [o for o in cubes if lab[o.cells[0, 0], o.cells[0, 1]] == lab[c.cells[0, 0], c.cells[0, 1]]
                and not o.part_of_other]
        if sum(o.n for o in kids) < 0.5 * c.n:
            others.append(c)
    return {"bin": binobj, "cubes": cubes, "others": others, "all": comps}


def pick_button(seg, cubes_free, log=print):
    """Choose the most button-like non-bin object."""
    cands = []
    for c in seg["others"]:
        if seg["bin"] is not None:
            x0, x1, y0, y1 = seg["bin"].bbox
            if x0 <= c.x <= x1 and y0 <= c.y <= y1:
                continue
        if c.ext[0] > 0.20 or c.n < 4:
            continue
        cands.append(c)
    # cube-like parts of other objects (button caps) as candidates too
    for o in seg["cubes"]:
        if o.part_of_other or getattr(o, "odd", False):
            cands.append(o)
    if not cands:
        # fallback: the free cube candidate whose height deviates most from the others
        free = [o for o in seg["cubes"] if not o.in_bin]
        if len(free) >= 2:
            mt = float(np.median([o.top for o in free]))
            dev = max(free, key=lambda o: abs(o.top - mt))
            if abs(dev.top - mt) > 0.006:
                return dev
        return None
    def score(c):
        s = 0.0
        s += min(c.n, 400) / 400.0
        s += 0.5 * (1.0 - getattr(c, "colored_frac", 1.0))
        return s
    return max(cands, key=score)


# --------------------------------------------------------------------------- linear camera model
def fit_linear(depth, samples):
    """Fit world = M @ [u*d, v*d, d, 1] from tool correspondences (pinhole camera, any convention).
    Returns (M, median_residual) or (None, inf)."""
    A, B = [], []
    for u, v, xyz in samples:
        if xyz is None:
            continue
        d = float(depth[int(v), int(u)])
        if d <= 0:
            continue
        A.append([u * d, v * d, d, 1.0])
        B.append(xyz)
    if len(A) < 6:
        return None, float("inf")
    A, B = np.array(A), np.array(B)
    M, *_ = np.linalg.lstsq(A, B, rcond=None)
    res = np.linalg.norm(A @ M - B, axis=1)
    return M.T, float(np.median(res))


def backproject_linear(depth, M):
    H, W = depth.shape
    u = np.arange(W)[None, :].astype(float)
    v = np.arange(H)[:, None].astype(float)
    d = depth
    F = np.stack([u * d, v * d, d, np.ones_like(d)], -1)
    return F @ M.T


def cloud_from_points(pts, depth, rgb, max_range=2.5):
    valid = (depth > 0.02) & (depth < max_range) & ~edge_mask(depth)
    P = pts[valid]
    C = rgb[valid].astype(float) / 255.0 if rgb is not None and rgb.shape[:2] == depth.shape else \
        np.full((P.shape[0], 3), 0.5)
    return P, C


def spread_pixels(depth, n=12, margin=24):
    """Well-spread pixels with locally smooth valid depth (for model fitting)."""
    H, W = depth.shape
    out = []
    g = int(math.ceil(math.sqrt(n * 2)))
    for a in range(g):
        for b in range(g):
            u = int(margin + (W - 2 * margin) * (a + 0.5) / g)
            v = int(margin + (H - 2 * margin) * (b + 0.5) / g)
            best = None
            for du in range(-6, 7, 3):
                for dv in range(-6, 7, 3):
                    uu, vv = u + du, v + dv
                    win = depth[max(0, vv - 2):vv + 3, max(0, uu - 2):uu + 3]
                    if depth[vv, uu] > 0 and win.min() > 0 and (win.max() - win.min()) < 0.006:
                        best = (uu, vv)
                        break
                if best:
                    break
            if best:
                out.append(best)
    if len(out) > n:
        idx = np.linspace(0, len(out) - 1, n).astype(int)
        out = [out[i] for i in idx]
    return out
