"""Colour naming and blob detection on 256x256 RGB frames."""
import numpy as np

from bu_geom import label

COLORS = ["red", "green", "blue", "yellow", "orange", "purple", "black", "white", "gray", "pink", "cyan", "brown"]


def rgb_to_hsv(rgb):
    a = np.asarray(rgb, float) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1)
    mn = a.min(-1)
    df = mx - mn
    h = np.zeros_like(mx)
    m = df > 1e-6
    rm = m & (mx == r)
    gm = m & (mx == g) & ~rm
    bm = m & ~rm & ~gm
    h[rm] = ((g[rm] - b[rm]) / df[rm]) % 6
    h[gm] = (b[gm] - r[gm]) / df[gm] + 2
    h[bm] = (r[bm] - g[bm]) / df[bm] + 4
    h = h * 60.0
    s = np.where(mx > 1e-6, df / np.maximum(mx, 1e-6), 0)
    return h, s, mx


def color_class_map(rgb):
    """Per-pixel saturated colour class (strings array, '' = none/wood/neutral)."""
    rgb = np.asarray(rgb)
    h, s, v = rgb_to_hsv(rgb)
    R = rgb[..., 0].astype(float)
    G = rgb[..., 1].astype(float)
    out = np.full(h.shape, "", dtype=object)
    sat = (s > 0.72) & (v > 0.25)
    red = sat & ((h < 12) | (h >= 340)) & (s > 0.85)
    orange_yellow = sat & (h >= 28) & (h < 72) & (s > 0.8)
    gr = G / np.maximum(R, 1)
    yellow = orange_yellow & (gr >= 0.78)
    orange = orange_yellow & (gr < 0.78)
    green = (s > 0.45) & (v > 0.2) & (h >= 80) & (h < 165)
    cyan = (s > 0.45) & (v > 0.2) & (h >= 165) & (h < 200)
    blue = (s > 0.5) & (v > 0.2) & (h >= 200) & (h < 265)
    purple = (s > 0.3) & (v > 0.25) & (h >= 265) & (h < 335)
    out[red] = "red"
    out[orange] = "orange"
    out[yellow] = "yellow"
    out[green] = "green"
    out[cyan] = "cyan"
    out[blue] = "blue"
    out[purple] = "purple"
    return out


def name_color(pixels):
    """Name the colour of a set of RGB pixels (N,3) by majority class, else neutral rules."""
    px = np.asarray(pixels, float).reshape(-1, 3)
    if len(px) == 0:
        return None, 0.0
    cm = color_class_map(px.reshape(-1, 1, 3).astype(np.uint8)).reshape(-1)
    vals, counts = np.unique(cm, return_counts=True)
    best = None
    for vv, cc in zip(vals, counts):
        if vv == "":
            continue
        if best is None or cc > best[1]:
            best = (vv, cc)
    if best is not None and best[1] >= 0.35 * len(px):
        return best[0], best[1] / len(px)
    h, s, v = rgb_to_hsv(np.median(px, 0).reshape(1, 3))
    h, s, v = float(h[0]), float(s[0]), float(v[0])
    if v < 0.25:
        return "black", 0.5
    if s < 0.18:
        return ("white" if v > 0.78 else "gray"), 0.5
    return None, 0.0


def is_wood_like(rgb):
    h, s, v = rgb_to_hsv(rgb)
    return (h >= 8) & (h < 32) & (s > 0.3) & (s < 0.8)


def neutral_bright(rgb):
    """White / light-grey pixels (bins are white with grey shaded sides)."""
    h, s, v = rgb_to_hsv(rgb)
    return (s < 0.16) & (v > 0.5)


def blobs(mask, min_px=6):
    lab, n = label(mask)
    out = []
    for k in range(1, n + 1):
        m = lab == k
        c = int(m.sum())
        if c < min_px:
            continue
        ys, xs = np.nonzero(m)
        out.append({"mask": m, "n": c, "u": float(xs.mean()), "v": float(ys.mean()),
                    "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))})
    return out


def interior_pixels(mask, k=6):
    """Up to k pixels well inside a blob (for depth sampling)."""
    from bu_geom import erode
    m = np.asarray(mask, bool)
    for r in (2, 1, 0):
        mm = erode(m, r) if r > 0 else m
        if mm.sum() >= 1:
            ys, xs = np.nonzero(mm)
            break
    cy, cx = ys.mean(), xs.mean()
    order = np.argsort((ys - cy) ** 2 + (xs - cx) ** 2)
    sel = order[:: max(1, len(order) // k)][:k]
    return [(int(xs[i]), int(ys[i])) for i in sel]
