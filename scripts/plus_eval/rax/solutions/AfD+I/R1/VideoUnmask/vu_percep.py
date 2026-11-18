"""Perception: colour-cube detection and depth-based container segmentation."""
import math
import numpy as np

try:
    import cv2  # noqa
    HAVE_CV2 = True
except Exception:  # pragma: no cover
    HAVE_CV2 = False
try:
    from scipy import ndimage as ndi  # noqa
    HAVE_NDI = True
except Exception:  # pragma: no cover
    HAVE_NDI = False

# Hue thresholds in OpenCV units (H 0..180, S,V 0..255), measured on all three released
# demonstrations (cube hue medians: red 0-1/178, orange 19-20, yellow 29-30, green 60,
# blue 120, purple 141-143; table 9-13 S<=185; ring projections H14-25 S<=140).
COLOR_RULES = {
    'red':    dict(h=[(0, 5), (172, 180)], s=150, v=50),
    'orange': dict(h=[(16, 23)], s=185, v=60),
    'yellow': dict(h=[(25, 37)], s=150, v=60),
    'green':  dict(h=[(42, 88)], s=100, v=35),
    'blue':   dict(h=[(104, 133)], s=120, v=30),
    'purple': dict(h=[(134, 166)], s=70, v=40),
    'pink':   dict(h=[(160, 172)], s=60, v=120),
    'cyan':   dict(h=[(88, 103)], s=100, v=40),
}
COLOR_WORDS = list(COLOR_RULES.keys()) + ['black', 'white', 'gray', 'grey', 'brown', 'magenta', 'violet']
COLOR_ALIASES = {'magenta': 'pink', 'violet': 'purple', 'grey': 'gray'}


def rgb_to_hsv_cv(img):
    """OpenCV-compatible HSV for uint8 RGB."""
    if HAVE_CV2:
        return cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_RGB2HSV).astype(np.int32)
    a = img[..., :3].astype(np.float64) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1)
    mn = a.min(-1)
    df = mx - mn
    h = np.zeros_like(mx)
    m = df > 1e-9
    rm = m & (mx == r)
    gm = m & (mx == g) & ~rm
    bm = m & ~rm & ~gm
    h[rm] = (60 * ((g[rm] - b[rm]) / df[rm]) + 360) % 360
    h[gm] = 60 * ((b[gm] - r[gm]) / df[gm]) + 120
    h[bm] = 60 * ((r[bm] - g[bm]) / df[bm]) + 240
    s = np.where(mx > 0, df / np.maximum(mx, 1e-9), 0)
    return np.stack([np.round(h / 2) % 180, np.round(s * 255), np.round(mx * 255)], -1).astype(np.int32)


def color_mask(hsv, color):
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if color == 'black':
        return (V < 45) & (S < 90)
    if color in ('white', 'gray'):
        return (S < 30) & (V > 150)
    r = COLOR_RULES.get(color)
    if r is None:
        return np.zeros(H.shape, bool)
    hm = np.zeros(H.shape, bool)
    for lo, hi in r['h']:
        hm |= (H >= lo) & (H <= hi)
    return hm & (S >= r['s']) & (V >= r['v'])


def label(mask):
    """Connected components (8-connectivity).  Returns (n, labels)."""
    mask = mask.astype(np.uint8)
    if HAVE_CV2:
        n, lab = cv2.connectedComponents(mask, connectivity=8)
        return n - 1, lab
    if HAVE_NDI:
        lab, n = ndi.label(mask, structure=np.ones((3, 3)))
        return n, lab
    # tiny union-find fallback
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    cur = 0
    for y in range(H):
        for x in range(W):
            if mask[y, x] and not lab[y, x]:
                cur += 1
                st = [(y, x)]
                lab[y, x] = cur
                while st:
                    a, b = st.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            yy, xx = a + dy, b + dx
                            if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                                lab[yy, xx] = cur
                                st.append((yy, xx))
    return cur, lab


def erode(mask, it=1):
    if HAVE_CV2:
        return cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=it) > 0
    if HAVE_NDI:
        return ndi.binary_erosion(mask, np.ones((3, 3)), iterations=it)
    return mask


def dilate(mask, it=1):
    if HAVE_CV2:
        return cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=it) > 0
    if HAVE_NDI:
        return ndi.binary_dilation(mask, np.ones((3, 3)), iterations=it)
    return mask


def open_mask(mask):
    if HAVE_CV2:
        m = mask.astype(np.uint8)
        return (cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0) & (m > 0)
    return mask


def detect_blobs(rgb, colors, min_area=12):
    """List of dict(color, u, v, area, pix=(ys, xs)) for each colour blob."""
    hsv = rgb_to_hsv_cv(rgb)
    out = []
    for c in colors:
        m = open_mask(color_mask(hsv, c))
        if not m.any():
            continue
        n, lab = label(m)
        for k in range(1, n + 1):
            ys, xs = np.nonzero(lab == k)
            if len(ys) < min_area:
                continue
            out.append(dict(color=c, u=float(xs.mean()), v=float(ys.mean()), area=int(len(ys)),
                            pix=(ys, xs), bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))))
    return out


def blob_world_point(blob, depth, cam):
    """World point of a blob: median depth over its (eroded) pixels at its centroid."""
    ys, xs = blob['pix']
    d = depth[ys, xs]
    ok = np.isfinite(d) & (d > 0.02) & (d < 5)
    if ok.sum() < 3:
        return None
    dz = float(np.median(d[ok]))
    P = cam.backproject(xs[ok].astype(float), ys[ok].astype(float), d[ok])
    # robust centre: median of per-pixel world points (handles oblique views)
    return np.median(P, axis=0), dz


def table_height(z, lo=-0.5, hi=0.5):
    """Most common height (5 mm bins) — the table dominates every view."""
    z = z[np.isfinite(z)]
    z = z[(z > lo) & (z < hi)]
    if len(z) < 50:
        return None
    bins = np.arange(lo, hi + 0.005, 0.005)
    h, e = np.histogram(z, bins)
    i = int(np.argmax(h))
    sel = z[(z >= e[i] - 0.005) & (z <= e[i + 1] + 0.005)]
    return float(np.median(sel))


def segment_objects(P, table_z, min_h=0.012, max_h=0.14, valid=None, min_px=25):
    """Connected components of pixels standing above the table (but below max_h).

    P: HxWx3 world points.  Returns list of dicts with pixel mask and world stats."""
    z = P[..., 2]
    ok = np.isfinite(z)
    if valid is not None:
        ok &= valid
    hgt = np.where(ok, z - table_z, np.nan)
    m = ok & (hgt > min_h) & (hgt < max_h)
    m = open_mask(m)
    n, lab = label(m)
    comps = []
    for k in range(1, n + 1):
        sel = lab == k
        cnt = int(sel.sum())
        if cnt < min_px:
            continue
        pts = P[sel]
        comps.append(dict(mask=sel, pts=pts, n=cnt,
                          cx=float(np.median(pts[:, 0])), cy=float(np.median(pts[:, 1])),
                          top=float(np.percentile(pts[:, 2], 95))))
    return comps, lab


def split_by_top(P, comp, table_z, band=0.012):
    """Split a merged component using only its top band (tops are separated by gaps)."""
    z = P[..., 2]
    sel = comp['mask'] & np.isfinite(z) & (z > comp['top'] - band)
    n, lab = label(open_mask(sel))
    parts = []
    for k in range(1, n + 1):
        s = lab == k
        if s.sum() < 10:
            continue
        pts = P[s]
        parts.append(dict(mask=s, pts=pts, n=int(s.sum()), cx=float(np.median(pts[:, 0])),
                          cy=float(np.median(pts[:, 1])), top=float(np.percentile(pts[:, 2], 95))))
    return parts


def min_area_rect(xy):
    """Minimum-area rectangle of 2-D points: (cx, cy, w, h, angle_rad) with w along angle."""
    xy = np.asarray(xy, float)
    if len(xy) < 5:
        c = xy.mean(0) if len(xy) else np.zeros(2)
        return float(c[0]), float(c[1]), 0.0, 0.0, 0.0
    if HAVE_CV2:
        (cx, cy), (w, h), ang = cv2.minAreaRect((xy * 1000.0).astype(np.float32))
        return cx / 1000.0, cy / 1000.0, w / 1000.0, h / 1000.0, math.radians(ang)
    best = None
    c0 = xy.mean(0)
    for deg in range(0, 90, 2):
        a = math.radians(deg)
        R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        q = (xy - c0) @ R.T
        mn, mx = q.min(0), q.max(0)
        area = float(np.prod(mx - mn))
        if best is None or area < best[0]:
            cc = (mn + mx) / 2
            cw = c0 + cc @ R
            best = (area, float(cw[0]), float(cw[1]), float(mx[0] - mn[0]), float(mx[1] - mn[1]), a)
    return best[1:]


def robust_rect(pts_xy):
    """Rectangle from the points after trimming 3% outliers around the median."""
    xy = np.asarray(pts_xy, float)
    if len(xy) > 40:
        c = np.median(xy, 0)
        r = np.linalg.norm(xy - c, axis=1)
        xy = xy[r <= np.percentile(r, 97)]
    return min_area_rect(xy)
