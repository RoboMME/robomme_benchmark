"""Scene perception: cube, target, button in world frame from wrist (and front) views."""
import math
import numpy as np
from perception import world_points, color_mask, components, hsv, yaw_of_square

COLORS = ["green", "red", "blue", "yellow", "purple", "orange", "cyan", "magenta"]


def parse_instruction(text):
    t = text.lower()
    color = "green"
    for c in ["green", "red", "blue", "yellow", "orange", "cyan", "purple", "magenta"]:
        if c + " cube" in t or c + " block" in t:
            color = c
            break
    words = {"once": 1, "one": 1, "twice": 2, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
    n = None
    import re
    m = re.search(r"repeating this pick-and-place action (\w+) time", t) or re.search(r"(\w+) times", t)
    if m:
        w = m.group(1)
        n = int(w) if w.isdigit() else words.get(w)
    if n is None:
        for w, v in words.items():
            if re.search(r"\b" + w + r"\b", t):
                n = v
                break
    return color, (n or 1)


def table_z(P):
    z = P[..., 2]
    z = z[np.isfinite(z)]
    z = z[(z > -0.05) & (z < 0.05)]
    return float(np.median(z)) if z.size else 0.0


def find_cube(meta, imgs, color, cam="wrist", near=None):
    rgb = imgs[(cam, "rgb")]
    P = world_points(meta, imgs, cam)
    tz = table_z(P)
    m = color_mask(rgb, color) & np.isfinite(P[..., 2]) & (P[..., 2] > tz + 0.01) & (P[..., 2] < tz + 0.12)
    cs = components(m, min_px=6)
    best = None
    for n, ys, xs in cs:
        pts = P[ys, xs]
        c = np.nanmedian(pts, 0)
        if near is not None and np.linalg.norm(c[:2] - np.asarray(near)[:2]) > 0.08:
            continue
        best = (n, ys, xs, pts)
        break
    if best is None:
        return None
    n, ys, xs, pts = best
    ztop = float(np.nanpercentile(pts[:, 2], 90))
    top = pts[pts[:, 2] > ztop - 0.006]
    c = np.nanmedian(top, 0)
    yaw = yaw_of_square(top[:, :2]) if len(top) >= 12 else 0.0
    return {"xyz": c, "top_z": ztop, "yaw": yaw, "npx": n, "uv": (float(xs.mean()), float(ys.mean())), "table_z": tz}


def find_target(meta, imgs, cam="wrist"):
    rgb = imgs[(cam, "rgb")]
    P = world_points(meta, imgs, cam)
    tz = table_z(P)
    m = color_mask(rgb, "purple", smin=0.25) & np.isfinite(P[..., 2]) & (P[..., 2] < tz + 0.03)
    cs = components(m, min_px=4)
    if not cs:
        return None
    n, ys, xs = cs[0]
    c0 = np.nanmedian(P[ys, xs], 0)
    ally, allx = np.nonzero(m)
    pts = P[ally, allx]
    sel = np.linalg.norm(pts[:, :2] - c0[:2], axis=1) < 0.07
    pts = pts[sel]
    c = np.array([(pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 1].min() + pts[:, 1].max()) / 2, float(np.nanmax(pts[:, 2]))])
    return {"xyz": c, "npx": int(sel.sum()), "table_z": tz, "radius": float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) / 2)}


def find_button(meta, imgs, cam="wrist"):
    rgb = imgs[(cam, "rgb")]
    P = world_points(meta, imgs, cam)
    tz = table_z(P)
    h, s, v = hsv(rgb)
    z = P[..., 2] - tz
    m = np.isfinite(z) & (z > 0.011) & (z < 0.05) & (s < 0.2)
    cs = components(m, min_px=10)
    for n, ys, xs in cs:
        pts = P[ys, xs]
        if np.ptp(pts[:, 0]) > 0.09 or np.ptp(pts[:, 1]) > 0.09:
            continue  # too large: robot base / arm
        if np.nanmedian(pts[:, 0]) < -0.5:
            continue  # robot base region
        zz = pts[:, 2]
        zmax = float(np.nanpercentile(zz, 95))
        if zmax - tz < 0.022:
            continue  # no raised cap
        cap = pts[zz > zmax - 0.006]
        c = np.nanmedian(cap, 0)
        base = pts[zz < zmax - 0.008]
        if len(base) > 10:
            cb = np.array([(base[:, 0].min() + base[:, 0].max()) / 2, (base[:, 1].min() + base[:, 1].max()) / 2])
        else:
            cb = c[:2]
        return {"xyz": np.array([c[0], c[1], zmax]), "base_xy": cb, "npx": n, "table_z": tz}
    return None
