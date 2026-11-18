"""Front/wrist image perception for StopCube (pure numpy; scipy optional).

All image coordinates are (u=column, v=row) in 256x256 images.
"""
import numpy as np

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover - fallback when scipy is unavailable
    _ndi = None


# ----------------------------------------------------------------- labelling
def label(mask):
    """Connected components (4-connectivity). Returns (labels, n)."""
    mask = np.asarray(mask, bool)
    if _ndi is not None:
        lab, n = _ndi.label(mask)
        return lab, int(n)
    lab = np.zeros(mask.shape, np.int32)
    n = 0
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]:
            continue
        n += 1
        stack = [(y0, x0)]
        lab[y0, x0] = n
        while stack:
            y, x = stack.pop()
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n
                    stack.append((yy, xx))
    return lab, n


def components(mask, min_area=1):
    lab, n = label(mask)
    out = []
    if n == 0:
        return out, lab
    idx = np.arange(1, n + 1)
    ys, xs = np.nonzero(lab)
    ls = lab[ys, xs]
    counts = np.bincount(ls, minlength=n + 1)
    sx = np.bincount(ls, weights=xs, minlength=n + 1)
    sy = np.bincount(ls, weights=ys, minlength=n + 1)
    # bounding boxes
    order = np.argsort(ls, kind="stable")
    ls_s, xs_s, ys_s = ls[order], xs[order], ys[order]
    starts = np.searchsorted(ls_s, idx, side="left")
    ends = np.searchsorted(ls_s, idx, side="right")
    for k, a, b in zip(idx, starts, ends):
        c = int(counts[k])
        if c < min_area:
            continue
        xk, yk = xs_s[a:b], ys_s[a:b]
        out.append({
            "id": int(k), "area": c,
            "cx": float(sx[k] / c), "cy": float(sy[k] / c),
            "x0": int(xk.min()), "x1": int(xk.max()), "y0": int(yk.min()), "y1": int(yk.max()),
        })
    return out, lab


# ----------------------------------------------------------------- colours
def table_mask(img):
    """Pixels that look like the orange wood table."""
    a = img.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r - g >= 35) & (r - b >= 60) & (g >= b) & (r >= 90)


def table_top_row(img):
    """First row (from the top) where the table dominates."""
    tm = table_mask(img)
    frac = tm.mean(axis=1)
    rows = np.nonzero(frac > 0.6)[0]
    return int(rows[0]) if len(rows) else 0


def nontable_components(img, top_row=None, min_area=12):
    if top_row is None:
        top_row = table_top_row(img)
    m = ~table_mask(img)
    m[: top_row + 1] = False
    comps, lab = components(m, min_area)
    res = []
    for c in comps:
        if c["y0"] <= top_row + 2:      # attached to the far edge: robot base / wall / arm
            c["touches_top"] = True
        else:
            c["touches_top"] = False
        pix = img[lab == c["id"]].astype(np.int16)
        mx, mn = pix.max(1), pix.min(1)
        c["frac_white"] = float(np.mean((mn > 195) & (mx - mn < 30)))
        c["frac_gray"] = float(np.mean((mx - mn < 22) & (mx > 90) & (mx <= 205)))
        c["frac_sat"] = float(np.mean(mx - mn > 50))
        c["mean_rgb"] = [float(v) for v in pix.mean(0)]
        res.append(c)
    return res, lab


def gray_centroid(img, comp, lab):
    """Centroid of grey (button cap) pixels inside a component, or None."""
    sel = lab == comp["id"]
    a = img.astype(np.int16)
    mx, mn = a.max(-1), a.min(-1)
    g = sel & (mx - mn < 22) & (mx > 90) & (mx <= 205)
    ys, xs = np.nonzero(g)
    if len(xs) < 4:
        return None
    return float(xs.mean()), float(ys.mean())


def dark_bottom_score(img):
    """Fraction of very dark pixels in the bottom quarter (gripper fingers in the wrist view)."""
    a = np.asarray(img)
    h = a.shape[0]
    sub = a[int(h * 0.72):].astype(np.int16)
    return float(np.mean(sub.max(-1) < 70))


# ----------------------------------------------------------------- wrist view
def wrist_button(img, max_row=200):
    """Locate the button (white square + grey cap) in the wrist image.

    Returns dict with square centroid/width and cap centroid, or None."""
    a = img.astype(np.int16)
    mx, mn = a.max(-1), a.min(-1)
    white = (mn > 212) & (mx - mn < 18)
    white[max_row:] = False
    comps, lab = components(white, 30)
    if not comps:
        return None
    sq = max(comps, key=lambda c: c["area"])
    ys, xs = np.nonzero(lab == sq["id"])
    width = float(np.percentile(xs, 98) - np.percentile(xs, 2))
    # cap: grey pixels inside the square's bbox
    x0, x1, y0, y1 = sq["x0"], sq["x1"], sq["y0"], max(sq["y1"], sq["y0"] + 1)
    sub_g = np.zeros_like(white)
    sub_g[y0:y1 + 1, x0:x1 + 1] = ((mx - mn < 14) & (mx > 120) & (mx < 205))[y0:y1 + 1, x0:x1 + 1]
    sub_g[max_row:] = False
    gcomps, glab = components(sub_g, 15)
    cap = None
    if gcomps:
        gc = max(gcomps, key=lambda c: c["area"])
        cap = (gc["cx"], gc["cy"], gc["area"])
    return {"sq_cx": sq["cx"], "sq_cy": sq["cy"], "sq_area": sq["area"], "width": width,
            "sq_bbox": (sq["x0"], sq["y0"], sq["x1"], sq["y1"]), "cap": cap}


# ----------------------------------------------------------------- tracking
class CubeTracker:
    """Background-subtraction tracker for the moving cube in the front view."""

    DIFF_THR = 60          # sum of |dRGB| over channels; tested on demos (thr 60)

    def __init__(self, top_row, cube_area0=None, cube_xy0=None):
        self.top_row = top_row
        self.frames = []           # list of uint8 images (front)
        self.bg = None
        self.bg_n = 0
        self.cube_area0 = cube_area0
        self.cube_xy0 = cube_xy0
        self.pos = []              # per frame (x, y) or None
        self.areas = []

    def add(self, img):
        self.frames.append(None if img is None else np.asarray(img, np.uint8))

    def rebuild_background(self, max_frames=41):
        valid = [i for i, f in enumerate(self.frames) if f is not None]
        n = len(valid)
        if n == 0:
            return
        sel = np.unique(np.linspace(0, n - 1, min(n, max_frames)).round().astype(int))
        stack = np.stack([self.frames[valid[i]] for i in sel]).astype(np.int16)
        self.bg = np.median(stack, axis=0).astype(np.int16)
        self.bg_n = len(self.frames)

    def _detect(self, img, prev, vel, area_ref):
        d = np.abs(img.astype(np.int16) - self.bg).sum(-1)
        m = d > self.DIFF_THR
        m[: self.top_row + 1] = False
        comps, _ = components(m, 8)
        best = None
        for c in comps:
            if c["y0"] <= self.top_row + 2:
                continue
            if c["area"] > 2500:
                continue
            if area_ref:
                if c["area"] < 0.25 * area_ref or c["area"] > 3.5 * area_ref:
                    continue
            elif c["area"] < 30 or c["area"] > 900:
                continue
            if prev is not None:
                px, py = prev[0] + vel[0], prev[1] + vel[1]
                dist = np.hypot(c["cx"] - px, c["cy"] - py)
                gate = max(14.0, 3.0 * np.hypot(*vel) + 8.0)
                if dist > gate:
                    continue
                score = dist
            else:
                score = -c["area"]
                if self.cube_xy0 is not None:
                    score = np.hypot(c["cx"] - self.cube_xy0[0], c["cy"] - self.cube_xy0[1])
            if best is None or score < best[0]:
                best = (score, c)
        return None if best is None else best[1]

    def retrack(self):
        """Re-run tracking over the full history with the current background."""
        self.pos, self.areas = [], []
        prev, vel, miss = None, (0.0, 0.0), 0
        areas_ok = []
        for img in self.frames:
            area_ref = float(np.median(areas_ok[-15:])) if len(areas_ok) >= 3 else self.cube_area0
            c = None if (img is None or self.bg is None) else self._detect(img, prev, vel, area_ref)
            if c is None:
                self.pos.append(None)
                self.areas.append(None)
                miss += 1
                if prev is not None:
                    prev = (prev[0] + vel[0], prev[1] + vel[1])
                if miss > 6:
                    prev, vel = None, (0.0, 0.0)
                continue
            miss = 0
            p = (c["cx"], c["cy"])
            if prev is not None:
                vel = (0.5 * vel[0] + 0.5 * (p[0] - prev[0]), 0.5 * vel[1] + 0.5 * (p[1] - prev[1]))
            prev = p
            self.pos.append(p)
            self.areas.append(c["area"])
            areas_ok.append(c["area"])

    def step(self):
        """Track only the newest frame (after add)."""
        img = self.frames[-1]
        prev, vel = None, (0.0, 0.0)
        # last known position and velocity
        known = [(i, p) for i, p in enumerate(self.pos) if p is not None]
        if known:
            i1, p1 = known[-1]
            gap = len(self.pos) - i1
            if len(known) >= 2 and gap <= 6:
                i0, p0 = known[-2]
                vel = ((p1[0] - p0[0]) / (i1 - i0), (p1[1] - p0[1]) / (i1 - i0))
                prev = (p1[0] + vel[0] * (gap - 1), p1[1] + vel[1] * (gap - 1))
            elif gap <= 6:
                prev = p1
        areas_ok = [a for a in self.areas if a]
        area_ref = float(np.median(areas_ok[-15:])) if len(areas_ok) >= 3 else self.cube_area0
        c = None if (img is None or self.bg is None) else self._detect(img, prev, vel, area_ref)
        if c is None:
            self.pos.append(None)
            self.areas.append(None)
        else:
            self.pos.append((c["cx"], c["cy"]))
            self.areas.append(c["area"])
        return self.pos[-1]

    def cube_size(self):
        a = [x for x in self.areas if x]
        if not a:
            return 13.0
        return float(np.sqrt(np.median(a)))

    def positions_array(self):
        """(n,2) array with NaN for missing; short gaps (<=4) linearly interpolated."""
        n = len(self.pos)
        P = np.full((n, 2), np.nan)
        for i, p in enumerate(self.pos):
            if p is not None:
                P[i] = p
        ok = ~np.isnan(P[:, 0])
        if ok.sum() >= 2:
            t = np.arange(n)
            idx = np.nonzero(ok)[0]
            for j in range(2):
                interp = np.interp(t, idx, P[idx, j])
                # only fill short gaps that are bracketed
                for i in np.nonzero(~ok)[0]:
                    lo = idx[idx < i]
                    hi = idx[idx > i]
                    if len(lo) and len(hi) and hi[0] - lo[-1] <= 5:
                        P[i, j] = interp[i]
        return P


# ----------------------------------------------------------------- arrivals
def distance_series(P, target):
    return np.hypot(P[:, 0] - target[0], P[:, 1] - target[1])


def find_arrivals(P, target, size, upto=None):
    """Arrival events from position history.

    An arrival is an excursion with D < R_ENTER*size whose minimum D < R_MIN*size,
    ended when D > R_EXIT*size. Returns list of dicts (t_min fractional, d_min, closed)."""
    R_ENTER, R_MIN, R_EXIT = 0.85, 0.55, 1.05
    n = len(P) if upto is None else min(len(P), upto)
    D = distance_series(P[:n], target)
    events = []
    inside = False
    cur = None
    for i in range(n):
        d = D[i]
        if np.isnan(d):
            continue
        if not inside:
            if d < R_ENTER * size:
                inside = True
                cur = {"t_enter": i, "i_min": i, "d_min": d}
        else:
            if d < cur["d_min"]:
                cur["i_min"], cur["d_min"] = i, d
            if d > R_EXIT * size:
                inside = False
                cur["t_exit"] = i
                events.append(cur)
                cur = None
    if inside and cur is not None:
        cur["t_exit"] = None
        events.append(cur)
    out = []
    for e in events:
        if e["d_min"] >= R_MIN * size:
            continue
        # fractional closest-approach time by fine interpolation around i_min
        i = e["i_min"]
        lo, hi = max(0, i - 2), min(n - 1, i + 2)
        tt = np.arange(lo, hi + 1)
        ok = ~np.isnan(P[lo:hi + 1, 0])
        t_min = float(i)
        if ok.sum() >= 3:
            tf = np.arange(lo, hi, 0.02)
            x = np.interp(tf, tt[ok], P[lo:hi + 1, 0][ok])
            y = np.interp(tf, tt[ok], P[lo:hi + 1, 1][ok])
            dd = np.hypot(x - target[0], y - target[1])
            t_min = float(tf[np.argmin(dd)])
        e["t_min"] = t_min
        e["closed"] = e["t_exit"] is not None
        out.append(e)
    return out


# ----------------------------------------------------------------- prediction
def predict_quadratic(P, target, t0, w=4, horizon=14):
    ts = np.arange(t0 - w + 1, t0 + 1)
    if ts[0] < 0:
        return None
    X = P[ts]
    if np.isnan(X).any():
        return None
    deg = 2 if w >= 4 else 1
    cx = np.polyfit(ts - t0, X[:, 0], deg)
    cy = np.polyfit(ts - t0, X[:, 1], deg)
    tf = np.arange(0, horizon, 0.02)
    D = np.hypot(np.polyval(cx, tf) - target[0], np.polyval(cy, tf) - target[1])
    i = int(np.argmin(D))
    return {"t": float(t0 + tf[i]), "d": float(D[i]), "method": "quad"}


def predict_replay(P, target, t0, prev_arrivals, w=6, max_ahead=16.0):
    """Align the last w positions with the approach to an earlier arrival.

    Returns predicted closest-approach time, the match cost (px) and the time scale."""
    ks = np.arange(w)
    cur = P[t0 - ks]
    if t0 - w + 1 < 0 or np.isnan(cur).any():
        return None
    t = np.arange(len(P))
    ok = ~np.isnan(P[:, 0])
    if ok.sum() < 10:
        return None
    tv = t[ok]
    Px, Py = P[ok, 0], P[ok, 1]
    S = np.arange(0.85, 1.1501, 0.025)
    Dg = np.arange(-1.0, max_ahead, 0.05)
    best = None
    for tau in prev_arrivals:
        # src[s, d, k] = tau - (D + k) * s
        src = tau - (Dg[None, :, None] + ks[None, None, :]) * S[:, None, None]
        valid = (src.min(axis=2) >= 0) & (src.max(axis=2) <= t0 - 1)
        if not valid.any():
            continue
        qx = np.interp(src, tv, Px)
        qy = np.interp(src, tv, Py)
        c = np.mean(np.hypot(qx - cur[None, None, :, 0], qy - cur[None, None, :, 1]), axis=2)
        c = np.where(valid, c, np.inf)
        j = np.unravel_index(int(np.argmin(c)), c.shape)
        if best is None or c[j] < best[0]:
            best = (float(c[j]), float(Dg[j[1]]), float(S[j[0]]), float(tau))
    if best is None or not np.isfinite(best[0]):
        return None
    return {"t": t0 + best[1], "cost": best[0], "scale": best[2], "tau": best[3], "method": "replay"}
