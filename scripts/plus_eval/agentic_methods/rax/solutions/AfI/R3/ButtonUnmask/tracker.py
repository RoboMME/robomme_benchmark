"""Cube tracking (pure functions, testable offline).

Per-pixel colour classification (HSV) of saturated, non-table pixels on both cameras, connected
components per colour, world xyz from depth.  Instances of each colour are clustered in world xy;
each instance keeps its last-seen xy.  When bins drop over the cubes the cubes disappear and the
last-seen xy marks the bin hiding each.
"""
import colorsys, re
import numpy as np
from scipy import ndimage

PALETTE_HUE = {"red": 0, "orange": 40, "yellow": 60, "green": 120, "cyan": 180,
               "blue": 240, "purple": 283, "pink": 320}
HUE_TOL = {"red": 10, "orange": 9, "yellow": 9}   # rendered: red 0, orange 41.5, yellow 60 (ep1, reh1)
SYN = {"grey": "gray", "violet": "purple", "magenta": "pink"}
NEUTRAL = ("white", "gray", "black")
COLOR_WORDS = list(PALETTE_HUE) + list(NEUTRAL) + list(SYN)
MIN_PX = {"front": 12, "wrist": 6}
MAX_PX = 1500
INSTANCE_RADIUS = 0.035   # cubes of the same colour closer than this are one instance

def parse_targets(instruction):
    """Ordered list of (colour, all_flag)."""
    text = instruction.lower()
    i = text.find("button")
    if i >= 0: text = text[i:]
    parts = re.split(r"(?:,|\bthen\b|\bfollowed by\b|\bfinally\b|\band\b)", text)
    out = []
    for p in parts:
        ws = re.findall(r"[a-z]+", p)
        cols = [SYN.get(w, w) for w in ws if w in COLOR_WORDS]
        if not cols: continue
        allf = ("all" in ws) or any(w.endswith("cubes") for w in ws) or ("containers" in ws)
        for c in cols:
            out.append((c, allf))
    return out

def hsv_image(rgb):
    a = rgb.astype(float) / 255.0
    mx = a.max(2); mn = a.min(2); d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    h = np.zeros_like(mx)
    dd = np.maximum(d, 1e-6)
    h = np.where(mx == r, ((g - b) / dd) % 6, h)
    h = np.where(mx == g, (b - r) / dd + 2, h)
    h = np.where(mx == b, (r - g) / dd + 4, h)
    h = (h * 60) % 360
    return h, s, mx

def color_masks(rgb):
    h, s, v = hsv_image(rgb)
    sat = (s > 0.8) & (v > 0.25)        # table: s 0.59-0.67, hue 20-23
    out = {}
    for name, hh in PALETTE_HUE.items():
        dh = np.minimum(np.abs(h - hh), 360 - np.abs(h - hh))
        tol = HUE_TOL.get(name, 20)
        m = sat & (dh < tol)
        if name == "purple":   # rendered purple has s~0.62
            m = (s > 0.45) & (v > 0.3) & (dh < 20)
        out[name] = m
    return out

def detect_cubes(rgb, P, cam="front", colors=None):
    """List of dict(color, px, xy, z) for cube-like blobs of saturated colours, with world xyz."""
    masks = color_masks(rgb)
    out = []
    for col, m in masks.items():
        if colors is not None and col not in colors: continue
        if not m.any(): continue
        lab, n = ndimage.label(m)
        for i in range(1, n + 1):
            msk = lab == i
            npx = int(msk.sum())
            if npx < MIN_PX.get(cam, 10) or npx > MAX_PX: continue
            pts = P[msk]; pts = pts[~np.isnan(pts).any(1)]
            if len(pts) < 3: continue
            xyz = np.median(pts, 0)
            if not (-0.01 < xyz[2] < 0.07): continue     # resting on the table (not held / falling high)
            if not (-0.5 < xyz[0] < 0.4 and abs(xyz[1]) < 0.5): continue
            out.append(dict(color=col, px=npx, xy=[float(xyz[0]), float(xyz[1])], z=float(xyz[2]), cam=cam))
    return out

class CubeMemory:
    """Instances per colour with last-seen xy and sighting counts."""
    def __init__(self):
        self.inst = {}            # color -> list of dict(xy, first, last, n)
        self.visible_now = set()  # (color, idx)
        self.last_any_visible = -1

    def update(self, step, cubes):
        self.visible_now = set()
        for c in cubes:
            lst = self.inst.setdefault(c["color"], [])
            xy = np.array(c["xy"])
            j = None
            for k, it in enumerate(lst):
                if np.linalg.norm(np.array(it["xy"]) - xy) < INSTANCE_RADIUS:
                    j = k; break
            if j is None:
                lst.append(dict(xy=c["xy"], first=step, last=step, n=1, px=c["px"], cams={c["cam"]}))
                j = len(lst) - 1
            else:
                it = lst[j]
                # prefer front+wrist average: running mean of position
                w = min(it["n"], 10)
                it["xy"] = [(it["xy"][0]*w + c["xy"][0]) / (w+1), (it["xy"][1]*w + c["xy"][1]) / (w+1)]
                it["last"] = step; it["n"] += 1; it["cams"].add(c["cam"])
            self.visible_now.add((c["color"], j))
        if self.visible_now:
            self.last_any_visible = step

    def instances(self, color, min_n=2):
        return [it for it in self.inst.get(color, []) if it["n"] >= min_n]

    def visible_colors(self):
        return sorted({c for c, _ in self.visible_now})

    def summary(self):
        return {k: [(round(it["xy"][0], 3), round(it["xy"][1], 3), it["first"], it["last"], it["n"]) for it in v]
                for k, v in self.inst.items()}

def detect_neutral(rgb, P, cam, exclude_xy=(), colors=("white", "gray")):
    """Small low white/gray objects (cubes), as separate components whose whole extent is low."""
    h, s, v = hsv_image(rgb)
    out = []
    for col in colors:
        if col == "white": m = (s < 0.15) & (v > 0.6)
        elif col == "gray": m = (s < 0.15) & (v > 0.3) & (v <= 0.6)
        else: m = (v < 0.15)
        lab, n = ndimage.label(m)
        for i in range(1, n + 1):
            msk = lab == i
            npx = int(msk.sum())
            if npx < MIN_PX.get(cam, 10) or npx > 600: continue
            pts = P[msk]; pts = pts[~np.isnan(pts).any(1)]
            if len(pts) < 3: continue
            if pts[:, 2].max() > 0.05 or not (0.0 < np.median(pts[:, 2]) < 0.045): continue
            ext = pts[:, :2].max(0) - pts[:, :2].min(0)
            if ext.max() > 0.05: continue
            xyz = np.median(pts, 0)
            if any(np.linalg.norm(xyz[:2] - np.array(e)) < 0.045 for e in exclude_xy): continue
            if not (-0.5 < xyz[0] < 0.4 and abs(xyz[1]) < 0.5): continue
            out.append(dict(color=col, px=npx, xy=[float(xyz[0]), float(xyz[1])], z=float(xyz[2]), cam=cam))
    return out
