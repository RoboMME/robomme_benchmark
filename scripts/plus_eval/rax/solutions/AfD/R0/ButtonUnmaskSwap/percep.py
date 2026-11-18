"""Image and point-cloud perception for ButtonUnmaskSwap."""
import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

COLOR_WORDS = ["red", "green", "blue", "yellow", "purple", "orange", "black", "white",
               "pink", "cyan", "brown", "gray", "grey", "magenta", "violet"]


def parse_targets(instruction):
    """Ordered target colours from the instruction (the part after the buttons clause)."""
    import re
    if not instruction:
        return []
    s = instruction.lower()
    idx = s.find("then")
    if idx >= 0:
        s = s[idx:]
    words = re.findall(r"[a-z]+", s)
    out = []
    for w in words:
        if w in COLOR_WORDS:
            w = {"grey": "gray", "violet": "purple", "magenta": "purple"}.get(w, w)
            if w not in out:
                out.append(w)
    return out


def rgb_to_hsv(rgb):
    """OpenCV-style HSV: H 0..180, S, V 0..255."""
    if cv2 is not None:
        return cv2.cvtColor(np.ascontiguousarray(rgb.astype(np.uint8)), cv2.COLOR_RGB2HSV).astype(np.int32)
    a = rgb.astype(np.float64) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1)
    mn = a.min(-1)
    df = mx - mn + 1e-12
    h = np.zeros_like(mx)
    m = mx == r
    h[m] = ((g - b)[m] / df[m]) % 6
    m2 = (mx == g) & ~m
    h[m2] = ((b - r)[m2] / df[m2]) + 2
    m3 = (mx == b) & ~m & ~m2
    h[m3] = ((r - g)[m3] / df[m3]) + 4
    h = h * 30.0
    s = np.where(mx > 0, (mx - mn) / (mx + 1e-12), 0)
    return np.stack([h, s * 255, mx * 255], -1).astype(np.int32)


def color_mask(hsv, color):
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    strong = (S > 190) & (V > 70)
    if color == "red":
        return strong & ((H <= 5) | (H >= 170))
    if color == "orange":
        return strong & (H >= 15) & (H <= 25)
    if color == "yellow":
        return strong & (H >= 26) & (H <= 38)
    if color == "green":
        return strong & (H >= 45) & (H <= 85)
    if color == "cyan":
        return strong & (H >= 86) & (H <= 100)
    if color == "blue":
        return strong & (H >= 105) & (H <= 131)
    if color == "purple":
        return (S > 110) & (V > 70) & (H >= 134) & (H <= 160)
    if color == "pink":
        return (S > 80) & (V > 120) & (H >= 160) & (H <= 169)
    if color == "black":
        return (V < 65) & (S < 120)
    if color == "white":
        return (S < 30) & (V > 200)
    if color == "gray":
        return (S < 30) & (V > 90) & (V < 180)
    if color == "brown":
        return (S > 150) & (V > 40) & (V < 120) & (H >= 5) & (H <= 20)
    return np.zeros(H.shape, bool)


def components(mask, min_area=3):
    """List of (area, cy, cx, ys, xs) connected components (8-connected)."""
    mask = mask.astype(np.uint8)
    out = []
    if cv2 is not None:
        n, lab, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            if st[i, 4] >= min_area:
                ys, xs = np.nonzero(lab == i)
                out.append((int(st[i, 4]), float(cen[i, 1]), float(cen[i, 0]), ys, xs))
        return out
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if ys.size >= min_area:
            out.append((int(ys.size), float(ys.mean()), float(xs.mean()), ys, xs))
    return out


def gripper_region_mask(shape):
    """Wrist-image pixels typically covered by the gripper fingers (bottom band)."""
    h, w = shape[:2]
    m = np.zeros((h, w), bool)
    m[int(h * 0.68):, :] = True
    return m


def whiteish(hsv):
    """Low-saturation light pixels: containers, buttons, robot, pillar."""
    return (hsv[..., 1] < 60) & (hsv[..., 2] > 105)


def world_points(model, depth, mask, stride=1):
    """Back-project masked pixels with valid depth. Returns (N,3) and pixel (v,u)."""
    if model is None or depth is None:
        return np.zeros((0, 3)), np.zeros((0, 2), int)
    m = mask & (depth > 0.01) & (depth < 5.0)
    if stride > 1:
        sm = np.zeros_like(m)
        sm[::stride, ::stride] = True
        m &= sm
    vs, us = np.nonzero(m)
    d = depth[vs, us]
    X = model.backproject(us.astype(np.float64), vs.astype(np.float64), d)
    return X, np.stack([vs, us], 1)


def grid_components(xy, cell=0.004, min_pts=6, dilate=1):
    """Cluster XY points with an occupancy grid. Returns list of index arrays."""
    if xy.shape[0] == 0:
        return []
    mn = xy.min(0) - cell * 3
    ij = np.floor((xy - mn) / cell).astype(int)
    W = ij.max(0) + 3
    if W[0] * W[1] > 4_000_000:
        return []
    occ = np.zeros((W[0], W[1]), np.uint8)
    occ[ij[:, 0], ij[:, 1]] = 1
    if dilate and cv2 is not None:
        occ = cv2.dilate(occ, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
    if cv2 is not None:
        n, lab = cv2.connectedComponents(occ, connectivity=8)
    else:
        from scipy import ndimage
        lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
        n += 1
    labels = lab[ij[:, 0], ij[:, 1]]
    out = []
    for k in range(1, n):
        idx = np.nonzero(labels == k)[0]
        if idx.size >= min_pts:
            out.append(idx)
    return out


def square_yaw(xy):
    """Edge direction (rad, mod pi/2) of a roughly square top face from XY points."""
    if xy.shape[0] < 12:
        return None
    try:
        if cv2 is not None:
            pts = (xy * 1000.0).astype(np.float32)
            (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
            if min(w, h) < 1e-6:
                return None
            from geom import wrap90
            return float(wrap90(np.deg2rad(ang)))
    except Exception:
        pass
    # fallback: search the angle minimising the bounding-box area
    best = None
    c = xy - xy.mean(0)
    for a in np.deg2rad(np.arange(-45, 45, 3)):
        R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        q = c @ R.T
        area = np.ptp(q[:, 0]) * np.ptp(q[:, 1])
        if best is None or area < best[0]:
            best = (area, a)
    return float(best[1])


def detect_buttons_img(rgb):
    """Candidate buttons: white square blobs containing a gray disk. Returns [(cy,cx,area)]."""
    hsv = rgb_to_hsv(rgb)
    white = (hsv[..., 1] < 40) & (hsv[..., 2] > 195)
    gray = (hsv[..., 1] < 40) & (hsv[..., 2] > 90) & (hsv[..., 2] <= 195)
    out = []
    for area, cy, cx, ys, xs in components(white, 12):
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        box = (y1 - y0) * (x1 - x0)
        g = gray[y0:y1, x0:x1]
        gfrac = g.mean() if box > 0 else 0
        if gfrac > 0.12 and box >= 30:
            gy, gx = np.nonzero(g)
            out.append((float(gy.mean() + y0), float(gx.mean() + x0), int(box), float(gfrac)))
    return out
