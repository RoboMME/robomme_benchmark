"""Front-camera RGB-D perception: elevated-object components and colored (cube) blobs."""
import math

import cv2
import numpy as np
from scipy import ndimage

# ---- tunables (untested on the real scene: chosen from expected object sizes) ----
ELEV_MIN_H = 0.006      # m above table to count as an object point
ELEV_MAX_H = 0.40
VOXEL = 0.01
ROBOT_VERT_EXTENT = 0.19  # component spanning more than this vertically is the arm
ROBOT_MAX_H = 0.33
HAND_XY_R = 0.11          # exclusion cylinder above the TCP (hand + fingers)
HAND_DZ_LO = -0.025
HAND_DZ_HI = 0.30
MIN_COMP_PTS = 12
FLOAT_MIN_H = 0.12          # component not touching the table region below this = arm part
SAT_MIN = 90              # 0..255
VAL_MIN = 50
CUBE_MAX_EXT = 0.065
CUBE_MAX_H = 0.07
CUBE_MIN_PIX = 5
BUTTON_R = 0.045            # set by controller from the button footprint

COLOR_HUES = {  # OpenCV hue 0..179 ranges
    "red": [(0, 8), (165, 180)],
    "orange": [(8, 19)],
    "yellow": [(19, 35)],
    "green": [(35, 85)],
    "cyan": [(85, 100)],
    "blue": [(100, 130)],
    "purple": [(130, 150)],
    "pink": [(150, 165)],
}
COLOR_NAMES = list(COLOR_HUES.keys())
SYNONYMS = {"magenta": "pink", "violet": "purple", "teal": "cyan"}


def hue_name_map(hsv):
    """int label per pixel: index into COLOR_NAMES, -1 if not saturated."""
    H, S, V = hsv[..., 0].astype(np.int32), hsv[..., 1].astype(np.int32), hsv[..., 2].astype(np.int32)
    lab = np.full(H.shape, -1, dtype=np.int32)
    sat = (S >= SAT_MIN) & (V >= VAL_MIN)
    for i, name in enumerate(COLOR_NAMES):
        for lo, hi in COLOR_HUES[name]:
            lab[sat & (H >= lo) & (H < hi)] = i
    return lab


class Scene:
    """Static geometry learned at start: camera mapping, table height, ROI."""

    def __init__(self, cam, z_table, roi):
        self.cam = cam
        self.z_table = z_table
        self.roi = roi  # (xmin, xmax, ymin, ymax)


def estimate_table(P, valid):
    z = P[..., 2][valid]
    if z.size < 100:
        return None
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.004, 0.002))
    # dominant horizontal planes: pick the highest one with large support (table above floor)
    k = int(np.argmax(hist))
    big = [i for i in range(len(hist)) if hist[i] >= max(0.15 * z.size, 0.35 * hist[k])]
    if big:
        k = max(big)
    z0 = 0.5 * (edges[k] + edges[k + 1])
    sel = np.abs(z - z0) < 0.005
    z_table = float(np.median(z[sel]))
    tab = valid & (np.abs(P[..., 2] - z_table) < 0.004)
    xy = P[..., :2][tab]
    xmin, xmax = np.percentile(xy[:, 0], [1, 99])
    ymin, ymax = np.percentile(xy[:, 1], [1, 99])
    frac = float(tab.sum()) / float(valid.sum())
    return z_table, (float(xmin) - 0.02, float(xmax) + 0.02, float(ymin) - 0.02, float(ymax) + 0.02), frac


class Comp:
    __slots__ = ("idx", "n", "centroid", "top", "hmin", "hmax", "rect", "ext_long", "ext_short",
                 "angle", "rgb", "colorful", "color", "robot", "pix", "kind", "xy_pts", "top_pts")

    def as_dict(self):
        return {"n": self.n, "top": [round(float(x), 4) for x in self.top], "hmin": round(self.hmin, 4),
                "hmax": round(self.hmax, 4), "ext": [round(self.ext_long, 3), round(self.ext_short, 3)],
                "ang": round(self.angle, 3), "rgb": [int(x) for x in self.rgb], "colorful": round(self.colorful, 2),
                "color": self.color, "robot": self.robot, "kind": self.kind}


def min_rect(xy):
    """xy: (N,2) metres -> (long, short, angle_of_long_side_rad)."""
    if len(xy) < 3:
        return 0.0, 0.0, 0.0
    pts = np.round(xy * 1000).astype(np.int32).reshape(-1, 1, 2)
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    a = math.radians(ang)
    if w >= h:
        return w / 1000.0, h / 1000.0, a
    return h / 1000.0, w / 1000.0, a + math.pi / 2


def analyze_frame(scene, rgb, depth, tcp=None):
    """Returns dict with components and colored blobs for one front frame."""
    P = scene.cam.backproject_image(depth)
    valid = depth > 0
    h = P[..., 2] - scene.z_table
    x, y = P[..., 0], P[..., 1]
    xmin, xmax, ymin, ymax = scene.roi
    roi = valid & (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)
    elev = roi & (h > ELEV_MIN_H) & (h < ELEV_MAX_H)
    hand = np.zeros_like(elev)
    if tcp is not None:
        dxy = np.hypot(x - tcp[0], y - tcp[1])
        dz = P[..., 2] - tcp[2]
        hand = (dxy < HAND_XY_R) & (dz > HAND_DZ_LO) & (dz < HAND_DZ_HI)
        # fingers/hand when roughly at TCP: sphere
        hand |= (np.sqrt(dxy ** 2 + dz ** 2) < 0.07)
    elev &= ~hand
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    clab = hue_name_map(hsv)
    vs, us = np.nonzero(elev)
    pts = P[vs, us]
    hh = h[vs, us]
    comps = []
    if len(vs) > 0:
        ix = np.floor((pts[:, 0] - xmin) / VOXEL).astype(np.int32)
        iy = np.floor((pts[:, 1] - ymin) / VOXEL).astype(np.int32)
        iz = np.floor(hh / VOXEL).astype(np.int32)
        nx, ny, nz = ix.max() + 1, iy.max() + 1, iz.max() + 1
        grid = np.zeros((nx, ny, nz), dtype=bool)
        grid[ix, iy, iz] = True
        labels, nlab = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=bool))
        plab = labels[ix, iy, iz]
        order = np.argsort(plab, kind="stable")
        bounds = np.searchsorted(plab[order], np.arange(1, nlab + 2))
        for li in range(nlab):
            sel = order[bounds[li]:bounds[li + 1]]
            if len(sel) < MIN_COMP_PTS:
                continue
            c = Comp()
            c.idx = li + 1
            c.n = int(len(sel))
            cp = pts[sel]
            ch = hh[sel]
            c.hmin, c.hmax = float(ch.min()), float(np.percentile(ch, 98))
            c.centroid = cp.mean(axis=0)
            topsel = ch >= c.hmax - 0.012
            c.top = np.array([cp[topsel, 0].mean(), cp[topsel, 1].mean(), scene.z_table + c.hmax])
            c.ext_long, c.ext_short, c.angle = min_rect(cp[:, :2])
            c.top_pts = cp[topsel, :2]
            c.xy_pts = cp[:, :2]
            pv, pu = vs[sel], us[sel]
            c.pix = (pv, pu)
            c.rgb = rgb[pv, pu].mean(axis=0)
            cl = clab[pv, pu]
            c.colorful = float((cl >= 0).mean())
            if c.colorful > 0.05:
                bc = np.bincount(cl[cl >= 0], minlength=len(COLOR_NAMES))
                c.color = COLOR_NAMES[int(np.argmax(bc))]
            else:
                c.color = None
            c.robot = bool((c.hmax - c.hmin) > ROBOT_VERT_EXTENT or c.hmax > ROBOT_MAX_H or c.hmin > FLOAT_MIN_H)
            c.kind = None
            comps.append(c)
    # colored blobs (cube candidates): saturated, low, not hand
    blobs = []
    lowmask = roi & (h > 0.003) & (h < CUBE_MAX_H + 0.01) & (~hand)
    for i, name in enumerate(COLOR_NAMES):
        m = (clab == i) & lowmask
        if m.sum() < CUBE_MIN_PIX:
            continue
        nl, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        for j in range(1, nl):
            bv, bu = np.nonzero(lab == j)
            if len(bv) < CUBE_MIN_PIX:
                continue
            bp = P[bv, bu]
            bh = h[bv, bu]
            lg, sh, _ = min_rect(bp[:, :2])
            med = np.median(bp, axis=0)
            blobs.append({"color": name, "n": int(len(bv)), "pos": med, "hmax": float(bh.max()),
                          "ext": float(lg), "pix": (int(np.median(bv)), int(np.median(bu)))})
    return {"comps": comps, "blobs": blobs}


def cube_blobs(res, bin_color=None, button_xy=None, button_r=None):
    button_r = BUTTON_R if button_r is None else button_r
    out = []
    for b in res["blobs"]:
        if b["ext"] > CUBE_MAX_EXT or b["hmax"] > CUBE_MAX_H + 0.005:
            continue
        if bin_color is not None and b["color"] == bin_color:
            continue
        if button_xy is not None and math.hypot(b["pos"][0] - button_xy[0], b["pos"][1] - button_xy[1]) < button_r:
            continue
        out.append(b)
    # keep the biggest blob per color nearby duplicates merged
    return out


def is_cubelike(c):
    return (c.colorful > 0.45 and c.ext_long <= CUBE_MAX_EXT and c.hmax <= CUBE_MAX_H)


def bin_candidates(res, button_xy=None, button_r=0.04, bin_color=None):
    out = []
    for c in res["comps"]:
        if c.robot:
            continue
        if button_xy is not None and math.hypot(c.top[0] - button_xy[0], c.top[1] - button_xy[1]) < button_r:
            continue
        if is_cubelike(c) and (bin_color is None or c.color != bin_color):
            continue
        if c.ext_long < 0.025 or c.hmax < 0.015:
            continue
        out.append(c)
    return out


def final_bins(res, button_xy=None, button_r=None):
    button_r = BUTTON_R if button_r is None else button_r
    """Bin-like components in a (static) frame. Cube-like comps are kept only if another comp has the
    same colour and similar size (identical containers form a group; cubes have distinct colours)."""
    cands = []
    for c in res["comps"]:
        if c.robot:
            continue
        if button_xy is not None and math.hypot(c.top[0] - button_xy[0], c.top[1] - button_xy[1]) < button_r:
            continue
        if c.ext_long < 0.025 or c.hmax < 0.015:
            continue
        cands.append(c)
    out = []
    for c in cands:
        if is_cubelike(c):
            same = [o for o in cands if o is not c and o.color == c.color
                    and abs(o.ext_long - c.ext_long) < 0.25 * c.ext_long and abs(o.hmax - c.hmax) < 0.25 * c.hmax]
            if not same:
                continue
        out.append(c)
    return out
