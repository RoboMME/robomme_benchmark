import math, re
import numpy as np
import cv2
from perception import world_map, hsv, color_mask, components, COLOR_HUES
from robot import log

NUMS = {"one": 1, "once": 1, "two": 2, "twice": 2, "three": 3, "thrice": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def parse_instruction(text):
    t = text.lower()
    color = None
    m = re.search(r"pick up the (\w+) cube", t)
    if m:
        color = m.group(1)
    else:
        for c in COLOR_HUES:
            if c in t:
                color = c
                break
    count = None
    m = re.search(r"(\w+) times", t)
    if m:
        w = m.group(1)
        count = int(w) if w.isdigit() else NUMS.get(w)
    if count is None:
        for w, v in NUMS.items():
            if re.search(r"\b%s\b" % w, t):
                count = v
                break
    return color, count


def detect(meta, imgs, cam, color):
    """Return dict with cube (xy, top z, yaw), target (xy), button (xy, top z)."""
    pw, valid = world_map(meta, imgs, cam)
    rgb = imgs[(cam, "rgb")]
    h = hsv(rgb)
    z = pw[..., 2]
    sat = h[..., 1]
    out = {}
    # table region sanity: points within workspace
    inws = valid & (np.abs(pw[..., 0]) < 0.6) & (np.abs(pw[..., 1]) < 0.6) & (z > -0.02) & (z < 0.2)
    # cube: colour mask, elevated
    cm = color_mask(rgb, color) & inws & (z > 0.015)
    cs = components(cm, 6)
    if cs:
        c = cs[0]
        P = pw[c["ys"], c["xs"]]
        ztop = np.percentile(P[:, 2], 90)
        top = P[P[:, 2] > ztop - 0.006]
        xy = np.median(top[:, :2], 0)
        yaw = None
        if len(top) >= 8:
            rect = cv2.minAreaRect(top[:, :2].astype(np.float32) * 1000)
            yaw = math.radians(rect[2])
        out["cube"] = {"xy": xy, "z": float(ztop), "yaw": yaw, "n": int(c["n"])}
    # target: saturated, flat on table (low z), not cube colour
    flat = inws & (z < 0.013)
    hue = h[..., 0]
    tm = flat & (hue >= 180) & (hue < 235) & (sat > 70) & (h[..., 2] > 50)
    if color in ("purple", "magenta", "pink"):
        tm &= ~color_mask(rgb, color)
    ts = components(tm, 6)
    if not ts:
        # fallback: saturated flat pixels whose hue differs from the table's dominant hue
        th = np.median(hue[flat]) if flat.any() else 0
        dh = np.abs(((hue - th) + 128) % 256 - 128)
        tm = flat & (sat > 90) & (dh > 30) & (h[..., 2] > 50) & ~color_mask(rgb, color)
        ts = components(tm, 6)
    if ts:
        # merge components close to the largest one (rings)
        c0 = ts[0]
        P0 = pw[c0["ys"], c0["xs"]][:, :2]
        cen = P0.mean(0)
        allp = [P0]
        for c in ts[1:]:
            P = pw[c["ys"], c["xs"]][:, :2]
            if np.linalg.norm(P.mean(0) - cen) < 0.06:
                allp.append(P)
        A = np.concatenate(allp)
        # bullseye centre: centre of the bounding extent of ring pixels
        ctr = (A.min(0) + A.max(0)) / 2
        out["target"] = {"xy": ctr, "n": len(A), "radius": float(np.max(A.max(0) - A.min(0)) / 2)}
    # button: grey/low saturation elevated parts (cap) with z 0.022..0.07
    bm = inws & (sat < 40) & (z > 0.022) & (z < 0.07) & (h[..., 2] > 40)
    bs = components(bm, 6)
    good = []
    for c in bs:
        P = pw[c["ys"], c["xs"]]
        ext = P[:, :2].max(0) - P[:, :2].min(0)
        if ext.max() < 0.07 and np.median(P[:, 0]) > -0.45:
            good.append((c, P))
    if good:
        c, P = good[0]
        out["button"] = {"xy": np.median(P[:, :2], 0), "z": float(np.percentile(P[:, 2], 90)), "n": int(c["n"])}
    return out
