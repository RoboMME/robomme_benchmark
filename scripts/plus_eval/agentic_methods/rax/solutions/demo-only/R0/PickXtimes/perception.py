"""Colour segmentation for PickXtimes (cubes, purple target, white button).

Pure numpy/cv2 functions on 256x256 RGB uint8 arrays (R,G,B order).
Thresholds were checked on demo frames (marble and wood tables, JPEG); live PNGs
are cleaner. No layout-specific coordinates.
"""
import numpy as np
import cv2

COLORS = ("green", "red", "blue", "yellow")


def _f(img):
    a = img.astype(np.int32)
    return a[..., 0], a[..., 1], a[..., 2]


def color_mask(img, color):
    R, G, B = _f(img)
    if color == "green":
        m = (G > 70) & (R * 100 < G * 50) & (B * 100 < G * 50)
    elif color == "red":
        m = (R > 80) & (G * 100 < R * 35) & (B * 100 < R * 35)
    elif color == "blue":
        m = (B > 70) & (R * 100 < B * 45) & (G * 100 < B * 45)
    elif color == "yellow":
        mn = np.minimum(R, G)
        m = (R > 80) & (G > 80) & (G * 100 > R * 75) & (R * 100 > G * 75) & (B * 100 < mn * 40)
    else:
        raise ValueError(color)
    return m.astype(np.uint8)


def target_mask(img):
    """Purple/magenta rings of the target: blue and red clearly above green."""
    R, G, B = _f(img)
    m = (B > G + 22) & (R > G + 15) & (B > 110) & (R > 90) & (R * 100 > B * 60)
    return m.astype(np.uint8)


def white_mask(img):
    R, G, B = _f(img)
    mx = np.maximum(np.maximum(R, G), B)
    mn = np.minimum(np.minimum(R, G), B)
    return ((mn > 215) & (mx - mn < 22)).astype(np.uint8)


def blobs(mask, min_area=4):
    """Connected components -> list of dicts sorted by area (desc)."""
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area:
            continue
        out.append(dict(label=i, area=int(a), bbox=(int(x), int(y), int(w), int(h)),
                        cx=float(cent[i][0]), cy=float(cent[i][1]), mask=(lab == i)))
    out.sort(key=lambda b: -b["area"])
    return out


def merge_close_blobs(bl, dist=6.0):
    """Merge blobs whose bboxes are within `dist` px (target rings, shaded cube faces)."""
    if not bl:
        return bl
    bl = list(bl)
    merged = True
    while merged:
        merged = False
        for i in range(len(bl)):
            for j in range(i + 1, len(bl)):
                a, b = bl[i], bl[j]
                ax, ay, aw, ah = a["bbox"]; bx, by, bw, bh = b["bbox"]
                gx = max(0, max(ax, bx) - min(ax + aw, bx + bw))
                gy = max(0, max(ay, by) - min(ay + ah, by + bh))
                if gx <= dist and gy <= dist:
                    m = a["mask"] | b["mask"]
                    ys, xs = np.nonzero(m)
                    x0, y0 = xs.min(), ys.min()
                    bl[i] = dict(label=a["label"], area=int(m.sum()),
                                 bbox=(int(x0), int(y0), int(xs.max() - x0 + 1), int(ys.max() - y0 + 1)),
                                 cx=float(xs.mean()), cy=float(ys.mean()), mask=m)
                    del bl[j]
                    merged = True
                    break
            if merged:
                break
    bl.sort(key=lambda b: -b["area"])
    return bl


def find_color_blobs(img, color, min_area=6, exclude=None):
    m = color_mask(img, color)
    if exclude is not None:
        m = m & (~exclude.astype(bool)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return merge_close_blobs(blobs(m, min_area), dist=2)


def find_target_blobs(img, min_area=5):
    """Purple ring blobs that contain saturated core purple pixels (rejects blue/wood fringes)."""
    m = target_mask(img)
    R, G, B = _f(img)
    core = (B > 195) & (R > 150) & (B - G > 30)
    out = []
    for b in merge_close_blobs(blobs(m, 2), dist=5):
        b["core"] = int((core & b["mask"]).sum())
        if b["core"] >= 3:
            out.append(b)
    return out


def find_button_blobs(img, exclude=None, min_area=6):
    """White square plate (with grey disc inside). Returns candidate blobs with score."""
    w = white_mask(img)
    if exclude is not None:
        w = w & (~exclude.astype(bool)).astype(np.uint8)
    w = cv2.morphologyEx(w, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    out = []
    for b in blobs(w, min_area):
        x, y, bw, bh = b["bbox"]
        # fill holes: grey disc inside the plate
        sub = b["mask"][y:y + bh, x:x + bw].astype(np.uint8)
        filled = sub.copy()
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(filled, cnts, -1, 1, thickness=-1)
        hole = int(filled.sum() - sub.sum())
        aspect = max(bw, bh) / max(1, min(bw, bh))
        fill_ratio = filled.sum() / float(bw * bh)
        b["hole"] = hole
        b["aspect"] = aspect
        b["fill_ratio"] = fill_ratio
        full = np.zeros_like(b["mask"])
        full[y:y + bh, x:x + bw] = filled.astype(bool)
        b["filled"] = full
        ys, xs = np.nonzero(full)
        b["fcx"], b["fcy"] = float(xs.mean()), float(ys.mean())
        out.append(b)
    return out


def interior_samples(mask, n=24, erode=1, rng_seed=0):
    """Pick up to n pixel coordinates (u, v) well inside a mask, spread evenly."""
    m = mask.astype(np.uint8)
    if erode > 0:
        e = cv2.erode(m, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
        if e.sum() >= 3:
            m = e
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return []
    if len(xs) <= n:
        idx = np.arange(len(xs))
    else:
        idx = np.linspace(0, len(xs) - 1, n).astype(int)
    return [(float(xs[i]), float(ys[i])) for i in idx]


def min_area_angle(mask):
    """Angle (deg, image coords, x right / y down) of the blob's min-area rectangle edge, in [-45,45)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    a = ((ang + 45.0) % 90.0) - 45.0
    return float(a)
