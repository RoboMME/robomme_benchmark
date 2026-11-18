"""Perception for ButtonUnmask: button, coloured cubes, white containers.

Uses RGB + depth back-projection only (no hidden state).
"""
import numpy as np
from PIL import Image

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None

COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "black"]
# PIL HSV hue (0..255) ranges; cubes are strongly saturated (s>=200), wood s~150-170, hue 10-18
HUE = {
    "red": [(0, 8), (235, 256)],
    "orange": [(19, 32)],
    "yellow": [(33, 52)],
    "green": [(55, 120)],
    "blue": [(140, 181)],
    "purple": [(181, 228)],
}
SAT_MIN = 185
SAT_MIN_BY_COLOR = {"purple": 110}   # ep1/demos: purple cube s~155, wood s<=170 but hue 10-18
VAL_MIN = 50
BLACK_ZMIN = 0.012   # black needs to stand above the table (shadows are on the table plane)


def hsv(rgb):
    return np.asarray(Image.fromarray(rgb).convert("HSV")).astype(np.int32)


def label(mask):
    if _ndi is not None:
        lab, n = _ndi.label(mask)
        return lab, n
    # simple fallback flood fill
    lab = np.zeros(mask.shape, np.int32)
    n = 0
    H, W = mask.shape
    for y0 in range(H):
        for x0 in range(W):
            if mask[y0, x0] and lab[y0, x0] == 0:
                n += 1
                stack = [(y0, x0)]
                lab[y0, x0] = n
                while stack:
                    y, x = stack.pop()
                    for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                        if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and lab[yy, xx] == 0:
                            lab[yy, xx] = n
                            stack.append((yy, xx))
    return lab, n


def color_masks(rgb):
    a = hsv(rgb)
    h, s, v = a[..., 0], a[..., 1], a[..., 2]
    out = {}
    for c, rngs in HUE.items():
        base = (s >= SAT_MIN_BY_COLOR.get(c, SAT_MIN)) & (v >= VAL_MIN)
        m = np.zeros(h.shape, bool)
        for lo, hi in rngs:
            m |= (h >= lo) & (h < hi)
        out[c] = m & base
    out["black"] = v < 45   # depth filter (z>BLACK_ZMIN) applied in detect_cubes
    return out


def detect_cubes(obs, cams=("front", "wrist"), min_px=5, zmax=0.07):
    """Coloured cube detections: list of dict(color, xyz, cam, px, uv)."""
    dets = []
    for cam in cams:
        if cam not in obs.rgb or cam not in obs.depth:
            continue
        P = obs.points(cam)
        masks = color_masks(obs.rgb[cam])
        for c, m in masks.items():
            if m.sum() < min_px:
                continue
            lab, n = label(m)
            for i in range(1, n + 1):
                sel = lab == i
                k = int(sel.sum())
                if k < min_px:
                    continue
                pts = P[sel]
                pts = pts[np.isfinite(pts).all(1)]
                if len(pts) < max(3, k // 3):
                    continue
                xyz = np.median(pts, 0)
                if not (-0.01 < xyz[2] < zmax):
                    continue
                if c == "black" and xyz[2] < BLACK_ZMIN:
                    continue
                if c == "black" and np.linalg.norm(xyz - obs.pos) < 0.10:
                    continue   # black gripper parts
                vs, us = np.nonzero(sel)
                dets.append(dict(color=c, xyz=xyz, cam=cam, px=k, uv=(float(us.mean()), float(vs.mean()))))
    return dets


class CubeMemory:
    """Accumulates coloured-cube sightings over time into tracks (colour + world xy)."""

    def __init__(self, merge_dist=0.03):
        self.tracks = []
        self.merge = merge_dist
        self.frames = 0
        self.history = []   # (step, n_dets)

    def add(self, obs, dets):
        self.frames += 1
        self.history.append((obs.step, len(dets)))
        for d in dets:
            best, bd = None, 1e9
            for t in self.tracks:
                if t["color"] != d["color"]:
                    continue
                dd = float(np.linalg.norm(t["xy"] - d["xyz"][:2]))
                if dd < bd:
                    best, bd = t, dd
            if best is not None and bd < self.merge:
                w = best["w"]
                # wrist detections (close, top-down) weigh more than front
                wd = 2.0 if d["cam"] == "wrist" else 1.0
                best["xy"] = (best["xy"] * w + d["xyz"][:2] * wd) / (w + wd)
                best["z"] = (best["z"] * w + d["xyz"][2] * wd) / (w + wd)
                best["w"] = w + wd
                best["n"] += 1
                best["last"] = obs.step
                best["cams"].add(d["cam"])
                best["px"] = max(best["px"], d["px"])
            else:
                self.tracks.append(dict(color=d["color"], xy=d["xyz"][:2].copy(), z=float(d["xyz"][2]),
                                        w=2.0 if d["cam"] == "wrist" else 1.0, n=1, first=obs.step,
                                        last=obs.step, cams={d["cam"]}, px=d["px"]))

    def good_tracks(self, min_n=2):
        return [t for t in self.tracks if t["n"] >= min_n]

    def summary(self):
        return [dict(color=t["color"], xy=[round(float(x), 4) for x in t["xy"]], z=round(t["z"], 4), n=t["n"],
                     first=t["first"], last=t["last"], cams=sorted(t["cams"]), px=t["px"]) for t in self.tracks]


def find_button(obs, cam="wrist"):
    """Button (white base + grey cap) on an otherwise empty table at reset: xy and top z."""
    P = obs.points(cam)
    a = hsv(obs.rgb[cam])
    s, v = a[..., 1], a[..., 2]
    z = P[..., 2]
    m = (s < 60) & (v > 90) & np.isfinite(z) & (z > 0.006) & (z < 0.07)
    lab, n = label(m)
    best = None
    for i in range(1, n + 1):
        sel = lab == i
        k = int(sel.sum())
        if k < 8:
            continue
        pts = P[sel]
        pts = pts[np.isfinite(pts).all(1)]
        if len(pts) < 5 or pts[:, 2].max() > 0.07:
            continue
        if best is None or k > best[0]:
            best = (k, pts)
    if best is None:
        return None
    pts = best[1]
    top = float(np.percentile(pts[:, 2], 90))
    # cap centre: the highest pixels
    hi = pts[pts[:, 2] > top - 0.006]
    xy = np.median(hi[:, :2], 0)
    return dict(xy=xy, top=top, n=best[0], base_xy=np.median(pts[:, :2], 0))


def white_mask(obs, cam):
    a = hsv(obs.rgb[cam])
    s, v = a[..., 1], a[..., 2]
    # container tops: s=0, v=231 (ep1); grey pillar top in demos s~25, v~191
    return (s < 30) & (v > 205)


def find_containers(obs, cam="wrist", ztop_min=0.05, zmax=0.11, min_px=25):
    """White containers (inverted cups ~7 cm tall): top-face components -> xy, yaw, top z.

    yaw is the box edge angle in world xy, reduced to [-pi/4, pi/4).
    """
    import math
    W = obs.points(cam)
    z = W[..., 2]
    m = white_mask(obs, cam) & np.isfinite(z) & (z > ztop_min) & (z < zmax)
    lab, n = label(m)
    out = []
    for i in range(1, n + 1):
        sel = lab == i
        k = int(sel.sum())
        if k < min_px:
            continue
        pts = W[sel]
        pts = pts[np.isfinite(pts).all(1)]
        top = float(np.percentile(pts[:, 2], 90))
        face = pts[pts[:, 2] > top - 0.008]
        if len(face) < 10:
            face = pts
        xy = face[:, :2]
        yaw, size = 0.0, (0.0, 0.0)
        try:
            import cv2
            rect = cv2.minAreaRect((xy * 1000.0).astype(np.float32))
            (cxm, cym), (w, h), ang = rect
            yaw = math.radians(ang)
            size = (w / 1000.0, h / 1000.0)
            center = np.array([cxm, cym]) / 1000.0
        except Exception:
            center = np.median(xy, 0)
        yaw = (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
        out.append(dict(xy=center, med_xy=np.median(xy, 0), yaw=yaw, top=top, n=k, size=size))
    # dedupe: oblique views split a container into top-face and side components at the same xy
    # (rehearsal 1: duplicates with top 0.052 at the same xy as 0.072 tops); keep the highest top.
    out.sort(key=lambda c: -c["top"])
    kept = []
    for c in out:
        if all(np.linalg.norm(c["xy"] - k["xy"]) > 0.02 for k in kept):
            kept.append(c)
    return kept
