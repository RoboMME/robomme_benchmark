"""Scene parsing: targets, button, red cube, swing count."""
import re
import numpy as np
from percep import color_mask, components, cube_estimate

WORDS = {"once": 1, "one": 1, "twice": 2, "two": 2, "thrice": 3, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def parse_count(instr):
    s = instr.lower()
    m = re.search(r"(\d+)\s*(?:times|x\b)", s)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(once|twice|thrice)\b", s)
    if m:
        return WORDS[m.group(1)]
    m = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+times?\b", s)
    if m:
        return WORDS[m.group(1)]
    for w, n in WORDS.items():
        if re.search(r"\b%s\b" % w, s):
            return n
    return None


def parse_order(instr):
    """Return 'right' or 'left': the side visited first in each swing."""
    s = instr.lower()
    i_l = s.find("left"); i_r = s.find("right")
    if i_l < 0 and i_r < 0:
        return "right"
    if i_l < 0:
        return "right"
    if i_r < 0:
        return "left"
    return "left" if i_l < i_r else "right"


def white_blobs(o, cam):
    out = []
    for a, m, c in components(color_mask(o.rgb(cam), "white"), min_px=6):
        P = o.deproject_mask(cam, m)
        if len(P) < 4:
            continue
        med = np.median(P, axis=0)
        out.append(dict(area=a, pix=c, xyz=med, zmax=float(np.percentile(P[:, 2], 95))))
    return out


def find_targets_button(o, cam="front"):
    blobs = white_blobs(o, cam)
    # table-level white blobs on the workspace (exclude far-away robot base etc.)
    tb = [b for b in blobs if b["xyz"][2] < 0.013 and -0.45 < b["xyz"][0] < 0.25 and abs(b["xyz"][1]) < 0.45]
    clusters = []
    for b in sorted(tb, key=lambda b: -b["area"]):
        for cl in clusters:
            if np.linalg.norm(cl["xy"] - b["xyz"][:2]) < 0.04:
                cl["members"].append(b)
                w = np.array([m["area"] for m in cl["members"]], float)
                cl["xy"] = np.average([m["xyz"][:2] for m in cl["members"]], axis=0, weights=w)
                cl["area"] += b["area"]
                break
        else:
            clusters.append(dict(xy=b["xyz"][:2].copy(), area=b["area"], members=[b]))
    clusters.sort(key=lambda c: -c["area"])
    targets = [c["xy"] for c in clusters[:2]]
    # button: white blob raised above table (base top ~0.015-0.035)
    bb = [b for b in blobs if 0.011 <= b["xyz"][2] < 0.05 and -0.45 < b["xyz"][0] < 0.25 and abs(b["xyz"][1]) < 0.45]
    button = None
    if bb:
        b = max(bb, key=lambda b: b["area"])
        button = dict(xy=b["xyz"][:2].copy(), ztop=b["zmax"])
    return targets, button, clusters, blobs


COLORS = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "pink"]


def parse_color(instr):
    s = instr.lower()
    m = re.search(r"pick up the (\w+) cube", s)
    if m and m.group(1) in COLORS:
        return m.group(1)
    for c in COLORS:
        if re.search(r"\b%s\b" % c, s):
            return c
    return "red"
