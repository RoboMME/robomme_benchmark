"""Colour/depth perception for BinFill: cubes, bin, button, table plane."""
import math
import re

import cv2
import numpy as np

# Hue windows in degrees (OpenCV HSV_FULL rescaled to 0-360). Measured on demo frames:
# red 0-9, orange 43, yellow 60, green 112-120, cyan 180, blue 239-241, magenta 299-300;
# cube saturation 230-255, wood 150-215 at hue 16-23.
HUES = {
    "red": [(0, 12), (345, 360)],
    "orange": [(33, 52)],
    "yellow": [(53, 78)],
    "green": [(90, 150)],
    "cyan": [(165, 200)],
    "blue": [(215, 262)],
    "magenta": [(280, 325)],
    "purple": [(262, 290)],
}
SAT_MIN = 200
VAL_MIN = 28

NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def parse_instruction(text):
    """'put three red cubes and two blue cubes into the bin' -> {'red': 3, 'blue': 2}."""
    t = text.lower()
    t = t.split("into")[0] if "into" in t else t
    goal = {}
    for m in re.finditer(r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+([a-z]+)\s+(?:cube|cubes|block|blocks)\b", t):
        n = int(m.group(1)) if m.group(1).isdigit() else NUM_WORDS[m.group(1)]
        col = m.group(2)
        goal[col] = goal.get(col, 0) + n
    return goal


def hsv_deg(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL)
    h = hsv[..., 0].astype(np.float32) * (360.0 / 256.0)
    return h, hsv[..., 1].astype(np.int32), hsv[..., 2].astype(np.int32)


def color_mask(rgb, color, hsv=None):
    h, s, v = hsv if hsv is not None else hsv_deg(rgb)
    m = np.zeros(h.shape, bool)
    for lo, hi in HUES[color]:
        m |= (h >= lo) & (h < hi)
    return m & (s >= SAT_MIN) & (v >= VAL_MIN)


def table_z(points):
    """Mode of the world z of the front view (the table/most common plane)."""
    z = points[..., 2]
    z = z[np.isfinite(z)]
    if z.size == 0:
        return 0.0
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.004, 0.002))
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2)


def cube_blobs(rgb, pts, colors, min_area=12, z_floor=None, hsv=None):
    """Connected saturated-colour blobs with 3D summaries.

    Returns a list of dicts: color, area, uv (centroid), center (xy of top face, z=top),
    top_z, extent (metric xy extent), npts.
    """
    hsv = hsv if hsv is not None else hsv_deg(rgb)
    out = []
    for col in colors:
        m = color_mask(rgb, col, hsv).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        for k in range(1, n):
            area = int(st[k, 4])
            if area < min_area:
                continue
            sel = lab == k
            P = pts[sel]
            P = P[np.isfinite(P[:, 0])]
            if len(P) < 5:
                continue
            if z_floor is not None:
                P = P[P[:, 2] > z_floor]
                if len(P) < 5:
                    continue
            zt = np.percentile(P[:, 2], 95)
            top = P[P[:, 2] >= zt - 0.008]
            cxy = np.median(top[:, :2], axis=0)
            ext = np.percentile(P[:, :2], 95, axis=0) - np.percentile(P[:, :2], 5, axis=0)
            out.append({
                "color": col, "area": area, "uv": (float(cen[k][0]), float(cen[k][1])),
                "center": np.array([cxy[0], cxy[1], zt]), "top_z": float(zt),
                "extent": ext, "npts": int(len(P)), "bbox": tuple(int(x) for x in st[k, :4]),
                "label": k, "top_pts": top,
            })
    return out


def top_face_yaw(top_pts):
    """Edge orientation (rad, in [-pi/4, pi/4)) of a cube's top face from its XY points."""
    if top_pts is None or len(top_pts) < 8:
        return None
    xy = top_pts[:, :2].astype(np.float32)
    xy = xy - xy.mean(axis=0)
    rect = cv2.minAreaRect((xy * 1000.0).astype(np.float32))
    ang = math.radians(rect[2])
    return (ang + math.pi / 4) % (math.pi / 2) - math.pi / 4


def bin_candidates(rgb, pts, tz, hsv=None, vmax=22):
    """The bin's interior is near-black. Returns candidate dicts, best first."""
    h, s, v = hsv if hsv is not None else hsv_deg(rgb)
    dark = (v <= vmax).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark)
    cands = []
    for k in range(1, n):
        area = int(st[k, 4])
        if area < 6:
            continue
        x, y, w, hh = st[k, :4]
        sel = lab == k
        P = pts[sel]
        P = P[np.isfinite(P[:, 0])]
        if len(P) < 4:
            continue
        # walls around the interior: dilate and take points above table
        ring = cv2.dilate(sel.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~sel
        W = pts[ring]
        W = W[np.isfinite(W[:, 0])]
        W = W[W[:, 2] > tz + 0.015] if len(W) else W
        floor = np.median(P, axis=0)
        if len(W) >= 5:
            lo = np.percentile(W[:, :2], 5, axis=0)
            hi = np.percentile(W[:, :2], 95, axis=0)
            cxy = (lo + hi) / 2
            top = float(np.percentile(W[:, 2], 95))
            size = hi - lo
        else:
            cxy = floor[:2]
            top = tz + 0.05
            size = np.array([0.0, 0.0])
        cands.append({"area": area, "uv": (float(cen[k][0]), float(cen[k][1])), "center": np.array([cxy[0], cxy[1], top]),
                      "floor": floor, "top_z": top, "size": size, "nwall": int(len(W)), "bbox": tuple(int(a) for a in st[k, :4])})
    # Prefer components whose floor is low (near table) and whose walls rise above it
    cands.sort(key=lambda c: (-(c["nwall"] >= 5), -c["area"]))
    return cands


def button_candidates(rgb, pts, tz, exclude_xy=(), hsv=None, robot_xy=None):
    """White/grey low-saturation compact objects standing on the table."""
    h, s, v = hsv if hsv is not None else hsv_deg(rgb)
    z = pts[..., 2]
    hgt = z - tz
    low = (s < 70) & (v > 50) & np.isfinite(z) & (hgt > 0.006) & (hgt < 0.08)
    m = low.astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    cands = []
    for k in range(1, n):
        area = int(st[k, 4])
        if area < 8:
            continue
        sel = lab == k
        P = pts[sel]
        P = P[np.isfinite(P[:, 0])]
        if len(P) < 6:
            continue
        lo = np.percentile(P[:, :2], 5, axis=0)
        hi = np.percentile(P[:, :2], 95, axis=0)
        ext = hi - lo
        if ext.max() > 0.12 or ext.min() < 0.015:
            continue  # too large (mat, robot) or thin (mat line fragment)
        if st[k, 1] <= 2:
            continue  # touches the image top: robot base / background
        if float(np.percentile(P[:, 2], 97)) - tz > 0.065:
            continue  # cut by the height band: a tall object, not the button
        cxy = (lo + hi) / 2
        if any(np.linalg.norm(cxy - np.asarray(e[:2])) < 0.07 for e in exclude_xy):
            continue
        if robot_xy is not None and np.linalg.norm(cxy - np.asarray(robot_xy[:2])) < 0.12:
            continue
        top = float(np.percentile(P[:, 2], 97))
        brightness = float(np.median(v[sel]))
        cands.append({"area": area, "uv": (float(cen[k][0]), float(cen[k][1])), "center": np.array([cxy[0], cxy[1], top]),
                      "top_z": top, "ext": ext, "v": brightness, "bbox": tuple(int(a) for a in st[k, :4])})
    cands.sort(key=lambda c: -c["v"] * min(c["area"], 300))
    return cands


def match_velocity(prev, cur, dt_steps, max_move_per_step=0.012):
    """Median per-step XY velocity from blob lists of two observations (same colours matched)."""
    if dt_steps <= 0:
        return None, 0
    disp = []
    for b in cur:
        best, bd = None, 1e9
        for a in prev:
            if a["color"] != b["color"]:
                continue
            d = np.linalg.norm(b["center"][:2] - a["center"][:2])
            if d < bd:
                best, bd = a, d
        if best is not None and bd < max_move_per_step * dt_steps:
            disp.append((b["center"][:2] - best["center"][:2]) / dt_steps)
    if len(disp) < 1:
        return None, 0
    D = np.array(disp)
    return np.median(D, axis=0), len(D)
