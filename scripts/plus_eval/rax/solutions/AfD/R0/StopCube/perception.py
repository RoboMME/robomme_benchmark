"""Front-camera perception for StopCube: table model, static objects, cube tracking.

Pure numpy / PIL; no simulator access.  Coordinates are image pixels (u = column,
v = row) unless stated otherwise.
"""
from collections import deque

import numpy as np
from PIL import Image


# ----------------------------------------------------------------- colour
def to_hsv(rgb):
    hsv = np.asarray(Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8))).convert("HSV")).astype(np.float32)
    return hsv[..., 0] * (360.0 / 255.0), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0


class TableModel:
    """Dominant table colour learned from the lower part of the first front image."""

    def __init__(self, rgb):
        H, S, V = to_hsv(rgb)
        h, w = H.shape
        sl = (slice(int(h * 0.55), int(h * 0.97)), slice(int(w * 0.03), int(w * 0.97)))
        hh, ss = H[sl].ravel(), S[sl].ravel()
        # circular median of hue
        ang = np.deg2rad(hh)
        mean_ang = np.arctan2(np.median(np.sin(ang)), np.median(np.cos(ang)))
        self.hue = float(np.rad2deg(mean_ang) % 360)
        dh = np.abs(((hh - self.hue) + 180) % 360 - 180)
        self.hue_tol = float(np.clip(np.percentile(dh, 99) + 4.0, 7.0, 25.0))
        self.s_lo = float(np.clip(np.percentile(ss, 1) - 0.12, 0.15, 0.6))
        self.s_med = float(np.median(ss))
        self.ok = self.s_med > 0.25  # a saturated table colour; otherwise rely on diffs only

    def nonwood(self, rgb, hsv=None):
        H, S, V = hsv if hsv is not None else to_hsv(rgb)
        dh = np.abs(((H - self.hue) + 180) % 360 - 180)
        if not self.ok:
            return np.ones(H.shape, bool)
        return (dh > self.hue_tol) | (S < self.s_lo)


# ----------------------------------------------------------------- components
def components(mask, min_area=1):
    """8-connected components of a boolean mask -> list of dict(pixels (N,2) [v,u], area, bbox)."""
    vs, us = np.nonzero(mask)
    if len(vs) == 0:
        return []
    h, w = mask.shape
    idx = -np.ones(mask.shape, np.int32)
    idx[vs, us] = np.arange(len(vs))
    parent = np.arange(len(vs))

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    # union with left, up-left, up, up-right neighbours (vectorised neighbour lookup)
    for dv, du in ((0, -1), (-1, -1), (-1, 0), (-1, 1)):
        nv, nu = vs + dv, us + du
        ok = (nv >= 0) & (nu >= 0) & (nu < w)
        j = np.full(len(vs), -1)
        j[ok] = idx[nv[ok], nu[ok]]
        for a, b in zip(np.nonzero(j >= 0)[0], j[j >= 0]):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    roots = np.array([find(i) for i in range(len(vs))])
    out = []
    for r in np.unique(roots):
        sel = roots == r
        if sel.sum() < min_area:
            continue
        pv, pu = vs[sel], us[sel]
        out.append({"v": pv, "u": pu, "area": int(sel.sum()),
                    "bbox": (int(pu.min()), int(pv.min()), int(pu.max()), int(pv.max())),
                    "c": (float(pu.mean()), float(pv.mean()))})
    return out


def dilate(mask, r=1):
    m = mask.copy()
    for _ in range(r):
        n = m.copy()
        n[1:] |= m[:-1]
        n[:-1] |= m[1:]
        n[:, 1:] |= m[:, :-1]
        n[:, :-1] |= m[:, 1:]
        m = n
    return m


def comp_mask(shape, comp):
    m = np.zeros(shape, bool)
    m[comp["v"], comp["u"]] = True
    return m


# ----------------------------------------------------------------- static objects
def robot_mask_from(nonwood, top_rows=6):
    """Non-table components touching the top rows of the image (robot base, arm, far wall)."""
    comps = components(nonwood, 1)
    m = np.zeros(nonwood.shape, bool)
    for c in comps:
        if c["bbox"][1] <= top_rows:
            m[c["v"], c["u"]] = True
    return m


def describe(comp, rgb, hsv):
    H, S, V = hsv
    v, u = comp["v"], comp["u"]
    s, val = S[v, u], V[v, u]
    lowS = float(np.mean(s < 0.12))
    sat = float(np.mean(s > 0.25))
    w = comp["bbox"][2] - comp["bbox"][0] + 1
    hgt = comp["bbox"][3] - comp["bbox"][1] + 1
    return {"lowS": lowS, "sat": sat, "w": w, "h": hgt, "area": comp["area"], "c": comp["c"],
            "rgb": np.median(rgb[v, u], 0).tolist()}


def static_candidates(rgb, table, robot_extra=None, min_area=12):
    hsv = to_hsv(rgb)
    nw = table.nonwood(rgb, hsv)
    rob = robot_mask_from(nw)
    if robot_extra is not None:
        rob |= robot_extra
    objs = components(nw & ~dilate(rob, 2), min_area)
    out = []
    for c in objs:
        d = describe(c, rgb, hsv)
        d["comp"] = c
        out.append(d)
    return out, nw, rob


def button_core(rgb, comp):
    """Grey cap pixels of the button component: low saturation, darker than the white base."""
    hsv = to_hsv(rgb)
    H, S, V = hsv
    v, u = comp["v"], comp["u"]
    s, val = S[v, u], V[v, u]
    low = s < 0.15
    if low.sum() < 4:
        return comp["c"]
    vals = val[low]
    hi = np.percentile(vals, 85)
    grey = low & (val < hi - 0.08)
    if grey.sum() < 3:
        return comp["c"]
    return float(u[grey].mean()), float(v[grey].mean())


# ----------------------------------------------------------------- background & cube
class CubeTracker:
    """Background-subtraction tracker of the moving cube in the front image."""

    def __init__(self, table, shape=(256, 256)):
        self.table = table
        self.samples = deque(maxlen=13)      # (step, rgb)
        self.sample_every = 2
        self.bg = None
        self.frames = {}                     # step -> rgb (recent)
        self.static_mask = np.zeros(shape, bool)   # button etc. (excluded)
        self.target_zone = np.zeros(shape, bool)   # where diff alone decides
        self.cube_area = None
        self.track = {}                      # step -> (u, v, area) or None
        self.last_good = None
        self.diff_thr = 55.0
        self.cache = {}

    def add_frame(self, step, rgb):
        self.frames[step] = rgb
        hsv = to_hsv(rgb)
        nw = self.table.nonwood(rgb, hsv)
        self.cache[step] = (nw, dilate(robot_mask_from(nw), 3))
        for s in [k for k in self.frames if k < step - 40]:
            del self.frames[s]
            self.cache.pop(s, None)
        if not self.samples or step - self.samples[-1][0] >= self.sample_every:
            self.samples.append((step, rgb.astype(np.int16)))
            stack = np.stack([s[1] for s in self.samples])
            self.bg = np.median(stack, axis=0).astype(np.int16)
            return True
        return False

    def detect(self, step, predicted=None, gate=None, roi=None):
        """Cube centroid at `step` or None.  predicted: (u,v) prior, gate: radius px."""
        rgb = self.frames.get(step)
        if rgb is None or self.bg is None:
            return None
        nw, rob = self.cache[step]
        diff = np.abs(rgb.astype(np.int16) - self.bg).sum(-1)
        cand = (diff > self.diff_thr) & (nw | self.target_zone) & ~rob & ~self.static_mask
        if roi is not None:
            cand &= roi
        comps = components(cand, 4)
        if not comps:
            return None
        exp_area = self.cube_area or 70.0
        best, best_score = None, 1e18
        for c in comps:
            if c["area"] > 6 * exp_area:
                continue
            if predicted is not None:
                dist = np.hypot(c["c"][0] - predicted[0], c["c"][1] - predicted[1])
                if gate is not None and dist > gate:
                    continue
            else:
                dist = 0.0
            # prefer close to prediction and area close to expectation
            score = dist + 8.0 * abs(np.log(max(c["area"], 1) / exp_area))
            if c["area"] < 0.25 * exp_area:
                score += 30
            if score < best_score:
                best, best_score = c, score
        if best is None:
            return None
        # merge nearby fragments (cube split by similar-coloured target ring)
        u0, v0 = best["c"]
        us, vs, area = [best["u"]], [best["v"]], best["area"]
        rad = 1.2 * np.sqrt(exp_area) + 2
        for c in comps:
            if c is best or c["area"] > 6 * exp_area:
                continue
            if np.hypot(c["c"][0] - u0, c["c"][1] - v0) < rad:
                us.append(c["u"]); vs.append(c["v"]); area += c["area"]
        u = np.concatenate(us); v = np.concatenate(vs)
        return (float(u.mean()), float(v.mean()), int(area))


# ----------------------------------------------------------------- wrist view
def wrist_button_cap(rgb):
    """Centre (u, v) of the grey button cap in a downward wrist view, or None.

    The button is a white square base with a grey round cap; the gripper's own
    fingers are dark and the hand is light grey at the image bottom/corners.
    """
    H, S, V = to_hsv(rgb)
    h, w = S.shape
    white = (S < 0.10) & (V > 0.86)
    comps = components(white, 30)
    if not comps:
        return None
    best = None
    for c in comps:
        x0, y0, x1, y1 = c["bbox"]
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        fill = c["area"] / float(bw * bh)
        if bw < 10 or bh < 10:
            continue
        # grey cap pixels inside the white bbox
        sub = (slice(y0, y1 + 1), slice(x0, x1 + 1))
        grey = (S[sub] < 0.12) & (V[sub] > 0.45) & (V[sub] < 0.82)
        if grey.sum() < 20:
            continue
        gc = components(grey, 15)
        if not gc:
            continue
        g = max(gc, key=lambda q: q["area"])
        score = g["area"] + 0.3 * c["area"]
        if best is None or score > best[0]:
            gu, gv = g["c"]
            best = (score, (gu + x0, gv + y0), g["area"], c["area"], (x0, y0, x1, y1))
    if best is None:
        return None
    return {"c": best[1], "cap_area": best[2], "white_area": best[3], "bbox": best[4]}
