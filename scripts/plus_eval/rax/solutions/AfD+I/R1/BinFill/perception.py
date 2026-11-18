"""Image-only perception for BinFill (no simulator state). All functions are pure:
numpy RGB uint8 HxWx3 in, plain python/numpy out. Tested offline on demo frames."""
import numpy as np
try:
    import cv2
except Exception:  # pragma: no cover - fallback if OpenCV is missing
    cv2 = None

# hue windows in degrees (cube colours are fully saturated renders)
HUE_WIN = {
    'red': [(0, 9), (340, 360)],
    'orange': [(30, 50)],
    'yellow': [(52, 75)],
    'green': [(90, 150)],
    'cyan': [(165, 200)],
    'blue': [(215, 262)],
    'purple': [(262, 285)],
    'magenta': [(285, 330)],
    'pink': [(320, 340)],
}
CUBE_COLOURS = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'magenta']
SAT_MIN = 0.78
VAL_MIN = 0.12


def hsv(img):
    """Return H in degrees [0,360), S, V in [0,1] (pure numpy)."""
    f = img[..., :3].astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx = f.max(-1)
    mn = f.min(-1)
    d = mx - mn
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    h = np.zeros_like(mx)
    dd = np.maximum(d, 1e-6)
    m = (mx == r) & (d > 0)
    h[m] = (60.0 * ((g[m] - b[m]) / dd[m])) % 360.0
    m2 = (mx == g) & (d > 0) & ~m
    h[m2] = 60.0 * ((b[m2] - r[m2]) / dd[m2]) + 120.0
    m3 = (mx == b) & (d > 0) & ~m & ~m2
    h[m3] = 60.0 * ((r[m3] - g[m3]) / dd[m3]) + 240.0
    return h, s, mx


def colour_mask(img, colour, hsv_cache=None, smin=SAT_MIN, vmin=VAL_MIN):
    h, s, v = hsv_cache if hsv_cache is not None else hsv(img)
    m = np.zeros(h.shape, bool)
    for lo, hi in HUE_WIN[colour]:
        m |= (h >= lo) & (h < hi)
    return m & (s > smin) & (v > vmin)


def components(mask, min_area=1):
    """Connected components -> list of dict(area, cx, cy, bbox(x,y,w,h), mask-index)."""
    mask = mask.astype(np.uint8)
    out = []
    if cv2 is not None:
        n, lab, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            a = int(st[i, 4])
            if a >= min_area:
                out.append(dict(area=a, cx=float(cen[i][0]), cy=float(cen[i][1]),
                                bbox=(int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3])),
                                label=i, labels=lab))
        return out
    # fallback: scipy or simple flood fill
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            if len(xs) >= min_area:
                out.append(dict(area=int(len(xs)), cx=float(xs.mean()), cy=float(ys.mean()),
                                bbox=(int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                                      int(ys.max() - ys.min() + 1)), label=i, labels=lab))
    except Exception:
        pass
    return out


def comp_mask(c):
    return c['labels'] == c['label']


def detect_cubes(img, colours=None, min_area=12, top_exclude=0, hsv_cache=None):
    """Saturated colour blobs. Returns list of dict(colour, area, cx, cy, bbox, px, py) where
    (px,py) is the pixel recommended for depth lookup (inside the blob)."""
    colours = colours or CUBE_COLOURS
    hc = hsv_cache if hsv_cache is not None else hsv(img)
    out = []
    for col in colours:
        m = colour_mask(img, col, hc)
        if top_exclude:
            m[:top_exclude] = False
        for c in components(m, min_area):
            cm = comp_mask(c)
            ys, xs = np.nonzero(cm)
            # pick an interior pixel nearest the centroid (robust to concave blobs)
            d = (xs - c['cx']) ** 2 + (ys - c['cy']) ** 2
            j = int(np.argmin(d))
            out.append(dict(colour=col, area=c['area'], cx=c['cx'], cy=c['cy'], bbox=c['bbox'],
                            px=int(xs[j]), py=int(ys[j])))
    return out


def detect_bin(img, top_exclude=20, hsv_cache=None):
    """Bin = compact very dark region (black interior). Returns dict or None."""
    h, s, v = hsv_cache if hsv_cache is not None else hsv(img)
    m = v < 0.09
    m[:top_exclude] = False
    best = None
    for c in components(m, 12):
        x, y, w, hh = c['bbox']
        fill = c['area'] / float(max(1, w * hh))
        aspect = max(w, hh) / float(max(1, min(w, hh)))
        if fill < 0.35 or aspect > 3.5 or c['area'] > 4000:
            continue
        # rim check: ring around bbox should be brighter / not dark
        score = c['area'] * fill
        if best is None or score > best['score']:
            best = dict(c, score=score, fill=fill)
    if best is None:
        return None
    best = {k: v_ for k, v_ in best.items() if k not in ('labels',)}
    return best


def ring_pixels(img_shape, mask, cx, cy, n_dirs=12, extra=2, max_r=80):
    """Walk from (cx,cy) outward until leaving mask, then `extra` px further. Returns list of (u,v)."""
    H, W = img_shape[:2]
    pts = []
    for k in range(n_dirs):
        a = 2 * np.pi * k / n_dirs
        dx, dy = np.cos(a), np.sin(a)
        inside = False
        for r in range(0, max_r):
            u, v = int(round(cx + dx * r)), int(round(cy + dy * r))
            if not (0 <= u < W and 0 <= v < H):
                break
            if mask[v, u]:
                inside = True
                continue
            if inside or r > 2:
                u2, v2 = int(round(cx + dx * (r + extra))), int(round(cy + dy * (r + extra)))
                if 0 <= u2 < W and 0 <= v2 < H:
                    pts.append((u2, v2))
                break
    return pts


def detect_button(img, bin_det=None, top_exclude=28, hsv_cache=None, min_area=60, max_area=1500):
    """Button = compact low-saturation (gray/white) blob not in the top wall band and not the mat."""
    h, s, v = hsv_cache if hsv_cache is not None else hsv(img)
    m = (s < 0.2) & (v > 0.15)
    m[:top_exclude] = False
    cands = []
    for c in components(m, min_area):
        if c['area'] > max_area:
            continue
        x, y, w, hh = c['bbox']
        fill = c['area'] / float(max(1, w * hh))
        aspect = max(w, hh) / float(max(1, min(w, hh)))
        if fill < 0.45 or aspect > 2.5:
            continue
        if y <= top_exclude:  # touches the wall band
            continue
        if bin_det is not None:
            bx, by, bw, bh = bin_det['bbox']
            if not (x > bx + bw + 3 or x + w < bx - 3 or y > by + bh + 3 or y + hh < by - 3):
                continue
        cm = comp_mask(c)
        ys, xs = np.nonzero(cm)
        d = (xs - c['cx']) ** 2 + (ys - c['cy']) ** 2
        j = int(np.argmin(d))
        cands.append(dict(area=c['area'], cx=c['cx'], cy=c['cy'], bbox=c['bbox'], fill=fill,
                          px=int(xs[j]), py=int(ys[j]), vmean=float(v[cm].mean())))
    if not cands:
        return None
    # prefer squarest, best filled, mid-size blob
    cands.sort(key=lambda c: -(c['fill'] - abs(np.log(c['area'] / 220.0)) * 0.2))
    return cands[0]


def detect_mat(img, top_exclude=28, hsv_cache=None):
    """Conveyor mat = large low-saturation region. Returns dict(mask, bbox) or None."""
    h, s, v = hsv_cache if hsv_cache is not None else hsv(img)
    m = (s < 0.3) & (v > 0.15) & (v < 0.75)
    m[:top_exclude] = False
    best = None
    for c in components(m, 1500):
        if best is None or c['area'] > best['area']:
            best = c
    if best is None:
        return None
    return dict(mask=comp_mask(best), bbox=best['bbox'], area=best['area'])


def classify_camera(img):
    """Higher = more like the front (third-person) view. In every demo frame the front view's
    bottom corners are saturated table wood (S~0.62) while the wrist view has a gripper finger
    (low saturation) in at least one bottom corner (min S <= 0.27). Use pairwise (argmax)."""
    h, s, v = hsv(img)
    bl = float(s[-30:, :25].mean())
    br = float(s[-30:, -20:].mean())
    return min(bl, br)


def blob_top_face(img, cube, hsv_cache=None):
    """Return (mask_top, mask_all) for the cube blob containing pixel (px,py)."""
    hc = hsv_cache if hsv_cache is not None else hsv(img)
    m = colour_mask(img, cube['colour'], hc)
    comps = components(m, 5)
    for c in comps:
        cm = comp_mask(c)
        if cm[cube['py'], cube['px']]:
            vv = hc[2]
            vals = vv[cm]
            thr = np.percentile(vals, 95) * 0.92
            top = cm & (vv >= thr)
            # keep the largest connected piece of the bright face
            cs = components(top, 3)
            if cs:
                big = max(cs, key=lambda c: c['area'])
                top = comp_mask(big)
            return top, cm
    return None, None


def min_area_angle(mask):
    """Dominant edge angle (radians, image coords x right, y down) of a blob, mod pi/2, via
    minAreaRect. Returns (angle, (w,h), (cx,cy)) or None."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    if cv2 is not None:
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
        return np.deg2rad(ang) % (np.pi / 2), (w, h), (cx, cy)
    # PCA fallback
    c = pts.mean(0)
    cov = np.cov((pts - c).T)
    ev, evec = np.linalg.eigh(cov)
    a = np.arctan2(evec[1, 1], evec[0, 1])
    return a % (np.pi / 2), (0, 0), tuple(c)
