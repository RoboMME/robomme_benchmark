import numpy as np
import re

COLOR_WORDS = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "magenta", "cyan", "violet"]

def parse_order(instruction):
    """Return list of (color, all_flag) in instructed order."""
    text = instruction.lower()
    idx = text.find("then")
    if idx >= 0:
        text = text[idx:]
    found = []
    # split into clauses at commas / 'then' / 'followed' / 'finally'
    clauses = re.split(r",|\bthen\b|\bfollowed by\b|\bfinally\b|\band\b", text)
    for cl in clauses:
        words = re.findall(r"[a-z]+", cl)
        for w in words:
            w2 = {"violet": "purple", "magenta": "pink"}.get(w, w)
            if w2 in COLOR_WORDS:
                allf = ("all" in words) or ("containers" in words) or ("cubes" in words)
                if not any(c == w2 for c, _ in found):
                    found.append((w2, allf))
                break
    return found

def rgb_to_hsv(img):
    f = img.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx = f.max(-1); mn = f.min(-1); d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-6
    rm = m & (mx == r); gm = m & (mx == g) & ~rm; bm = m & ~rm & ~gm
    h[rm] = ((g - b)[rm] / d[rm]) % 6
    h[gm] = (b - r)[gm] / d[gm] + 2
    h[bm] = (r - g)[bm] / d[bm] + 4
    h = h * 60.0
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    return h, s, mx

# (hue ranges, min saturation, min value)
BANDS = {
    "red": ([(0, 12), (345, 360)], 0.85, 0.35),
    "orange": ([(30, 50)], 0.85, 0.5),
    "yellow": ([(50, 70)], 0.7, 0.5),
    "green": ([(85, 160)], 0.6, 0.3),
    "cyan": ([(170, 198)], 0.6, 0.3),
    "blue": ([(200, 255)], 0.6, 0.25),
    "purple": ([(258, 300)], 0.4, 0.3),
    "pink": ([(300, 345)], 0.45, 0.4),
}

def color_mask(hsv, name):
    h, s, v = hsv
    ranges, smin, vmin = BANDS[name]
    m = np.zeros(h.shape, bool)
    for lo, hi in ranges:
        m |= (h >= lo) & (h < hi)
    return m & (s >= smin) & (v >= vmin)

def components(mask, min_area=8):
    """Simple 4-connected components; returns list of (us, vs)."""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    out = []
    ys, xs = np.nonzero(mask)
    cur = 0
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]:
            continue
        cur += 1
        stack = [(y0, x0)]; lab[y0, x0] = cur; pts = []
        while stack:
            y, x = stack.pop(); pts.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = cur; stack.append((yy, xx))
        if len(pts) >= min_area:
            p = np.array(pts)
            out.append((p[:, 1], p[:, 0]))
    return out

def find_cubes(robot, colors, cam="front", min_area=10, max_area=4000):
    """Return {color: [(world_xyz, (u,v), area), ...]} for elevated colour blobs on the table."""
    img = robot.rgb(cam)
    D = robot.depth(cam)
    hsv = rgb_to_hsv(img)
    vs, us = np.mgrid[0:D.shape[0], 0:D.shape[1]]
    P = robot.deproject(cam, us.ravel(), vs.ravel(), D.ravel()).reshape(D.shape[0], D.shape[1], 3)
    z = P[..., 2]
    elev = (D > 0) & (z > 0.006) & (z < 0.075)
    res = {}
    for c in colors:
        blobs = []
        for bu, bv in components(color_mask(hsv, c) & elev, min_area):
            if len(bu) > max_area:
                continue
            pts = P[bv, bu]
            if len(pts) < 5:
                continue
            xyz = np.median(pts, axis=0)
            blobs.append((xyz, (float(bu.mean()), float(bv.mean())), int(len(bu))))
        res[c] = blobs
    return res
