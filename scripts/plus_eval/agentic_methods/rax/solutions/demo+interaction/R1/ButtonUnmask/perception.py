"""Image-space perception for ButtonUnmask (pure functions, offline-testable).

All images are HxWx3 uint8 RGB arrays (256x256 in the benchmark).
OpenCV HSV: H in [0,180), S,V in [0,255].
Thresholds were calibrated on the three released demonstrations (JPEG); live
frames are lossless so colours are cleaner.
"""
import math
import numpy as np
import cv2

# colour classes: (name, hue ranges, min S, min V)
COLOR_CLASSES = [
    ("red", [(0, 5), (171, 180)], 170, 60),
    ("orange", [(16, 24)], 205, 90),
    ("yellow", [(25, 36)], 170, 80),
    ("green", [(42, 80)], 140, 60),
    ("cyan", [(84, 102)], 150, 60),
    ("blue", [(110, 130)], 185, 70),
    ("purple", [(132, 152)], 110, 70),
    ("pink", [(153, 170)], 110, 70),
]
COLOR_NAMES = [c[0] for c in COLOR_CLASSES]
MIN_CUBE_AREA = {"front": 18, "wrist": 30}
MAX_CUBE_AREA = {"front": 1500, "wrist": 6000}


def hsv_of(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def wrist_gripper_mask(h=256, w=256):
    """Pixels of the wrist image that may show the gripper itself (True = ignore)."""
    m = np.zeros((h, w), bool)
    yy, xx = np.mgrid[0:h, 0:w]
    m |= (yy > int(0.70 * h)) & (xx < int(0.40 * w))
    m |= (yy > int(0.68 * h)) & (xx > int(0.82 * w))
    m |= yy > int(0.92 * h)
    return m


def color_mask(hsv, name):
    for cname, ranges, smin, vmin in COLOR_CLASSES:
        if cname != name:
            continue
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        m = np.zeros(H.shape, bool)
        for lo, hi in ranges:
            m |= (H >= lo) & (H < hi)
        return m & (S >= smin) & (V >= vmin)
    raise KeyError(name)


def detect_cubes(rgb, camera, colors=None, ignore=None):
    """Return list of dicts {color, u, v, area, pts[(u,v)...]} of saturated blobs.

    pts are interior pixels (eroded) usable for depth queries.
    """
    hsv = hsv_of(rgb)
    out = []
    colors = colors or COLOR_NAMES
    amin, amax = MIN_CUBE_AREA.get(camera, 20), MAX_CUBE_AREA.get(camera, 6000)
    for name in colors:
        m = color_mask(hsv, name)
        if ignore is not None:
            m &= ~ignore
        m = m.astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        for k in range(1, n):
            area = int(st[k, 4])
            if area < amin or area > amax:
                continue
            x, y, w, h = st[k, :4]
            # cubes are compact
            if w > 4 * h + 4 or h > 4 * w + 4:
                continue
            sel = (lab == k).astype(np.uint8)
            er = cv2.erode(sel, np.ones((3, 3), np.uint8))
            ys, xs = np.nonzero(er if er.sum() >= 3 else sel)
            cu, cv_ = float(cen[k][0]), float(cen[k][1])
            # interior pixel closest to centroid
            j = int(np.argmin((xs - cu) ** 2 + (ys - cv_) ** 2))
            out.append({"color": name, "u": int(xs[j]), "v": int(ys[j]),
                        "cu": cu, "cv": cv_, "area": area,
                        "bbox": [int(x), int(y), int(w), int(h)]})
    return out


def wood_mask(hsv):
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    return (H >= 5) & (H <= 17) & (S >= 60) & (V >= 50)


def detect_button(rgb, window, expect, use_gripper_mask, max_area=4000):
    """Compact non-wood blob inside window=(y0,y1,x0,x1) fractions, nearest expect=(u,v) fractions."""
    h, w = rgb.shape[:2]
    hsv = hsv_of(rgb)
    non = ~wood_mask(hsv)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    highlight = (S < 60) & (S > 22) & (H >= 4) & (H <= 20)
    non &= ~highlight
    if use_gripper_mask:
        non &= ~wrist_gripper_mask(h, w)
    y0, y1, x0, x1 = window
    win = np.zeros((h, w), bool)
    win[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = True
    non &= win
    m = cv2.morphologyEx(non.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    best, bscore = None, -1e9
    for k in range(1, n):
        area = int(st[k, 4])
        x, y, bw, bh = st[k, :4]
        if area < 25 or area > max_area:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 2.2:
            continue
        fill = area / float(bw * bh)
        if fill < 0.45:
            continue
        cu, cv_ = cen[k]
        dist = math.hypot(cu - expect[0] * w, cv_ - expect[1] * h)
        score = math.sqrt(area) - 0.15 * dist
        if score > bscore:
            bscore, best = score, {"u": float(cu), "v": float(cv_), "area": area,
                                   "bbox": [int(x), int(y), int(bw), int(bh)]}
    return best


def detect_button_wrist(rgb):
    """Button in the home-pose wrist image (compact non-wood blob near lower centre)."""
    return detect_button(rgb, (0.36, 0.86, 0.22, 0.80), (0.5, 0.62), True)


def detect_button_front(rgb):
    """Button in the front image at reset (small blob in the upper middle)."""
    return detect_button(rgb, (0.17, 0.40, 0.30, 0.70), (0.49, 0.26), False, max_area=600)


def light_mask(hsv):
    """White/light-grey container pixels (tops and sides)."""
    S, V = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    return (S <= 26) & (V >= 120)


def wrap90(a):
    """Map an angle in degrees to (-45, 45]."""
    a = (a + 45.0) % 90.0 - 45.0
    if a <= -45.0:
        a += 90.0
    return a


def rect_angle(mask_u8):
    """Edge angle (deg, image frame, in (-45,45]) of the min-area rectangle."""
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    # angle of first edge
    e = box[1] - box[0]
    ang = math.degrees(math.atan2(float(e[1]), float(e[0])))
    return wrap90(ang), rect


def container_tops(rgb, depth=None, ignore_gripper=True):
    """Segment container top faces in a (roughly top-down) wrist image.

    Returns list of dicts {u, v, area, angle, rect, mask(bool), bbox}.
    With depth (metres, same size) the top face is the near-minimum-depth part of
    each light blob; otherwise the brightest part.
    """
    h, w = rgb.shape[:2]
    hsv = hsv_of(rgb)
    lm = light_mask(hsv)
    if ignore_gripper:
        lm &= ~wrist_gripper_mask(h, w)
    lm8 = cv2.morphologyEx(lm.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(lm8)
    V = hsv[..., 2].astype(int)
    out = []
    for k in range(1, n):
        area = int(st[k, 4])
        if area < 60:
            continue
        blob = lab == k
        top = None
        if depth is not None:
            d = depth.copy()
            valid = blob & (d > 0.01)
            if valid.sum() > 30:
                dmin = np.percentile(d[valid], 3)
                top = valid & (d <= dmin + 0.008)
        if top is None:
            vmax = np.percentile(V[blob], 95)
            top = blob & (V >= vmax - 14)
        top8 = top.astype(np.uint8)
        top8 = cv2.morphologyEx(top8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        # keep the largest connected part
        n2, lab2, st2, cen2 = cv2.connectedComponentsWithStats(top8)
        if n2 < 2:
            continue
        j = 1 + int(np.argmax(st2[1:, 4]))
        tarea = int(st2[j, 4])
        if tarea < 40:
            continue
        tmask = (lab2 == j)
        ang, rect = rect_angle(tmask.astype(np.uint8))
        x, y, bw, bh = st2[j, :4]
        # touching the border => partially visible, angle less reliable
        border = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
        er = cv2.erode(tmask.astype(np.uint8), np.ones((5, 5), np.uint8))
        ys, xs = np.nonzero(er if er.sum() > 5 else tmask)
        cu, cv_ = float(cen2[j][0]), float(cen2[j][1])
        i = int(np.argmin((xs - cu) ** 2 + (ys - cv_) ** 2))
        out.append({"u": int(xs[i]), "v": int(ys[i]), "cu": cu, "cv": cv_,
                    "area": tarea, "blob_area": area, "angle": ang,
                    "rect": rect, "bbox": [int(x), int(y), int(bw), int(bh)],
                    "border": bool(border), "mask": tmask})
    return out
