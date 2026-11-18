"""Appearance-agnostic object-colour learning and segmentation (RGB only).

Objects in VideoRepick are several identical, saturated, compact blobs on a wood
table that may carry coloured lighting patches.  The colour prototype is learned per
episode: candidate hue peaks away from the table hue are scored by how blob-like
(compact, small, several similar components) their masks are.
"""
import numpy as np

def rgb_to_hsv(rgb):
    a = rgb.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1); mn = a.min(-1); d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-6
    rm = m & (mx == r); gm = m & (mx == g) & ~rm; bm = m & ~rm & ~gm
    h[rm] = ((g - b)[rm] / d[rm]) % 6
    h[gm] = (b - r)[gm] / d[gm] + 2
    h[bm] = (r - g)[bm] / d[bm] + 4
    h = h * 60.0
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    return h, s, mx

def rgb_to_lab(rgb):
    a = rgb.astype(np.float32) / 255.0
    a = np.where(a > 0.04045, ((a + 0.055) / 1.055) ** 2.4, a / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = a @ M.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    A = 500 * (f[..., 0] - f[..., 1])
    B = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, A, B], -1)

def hue_dist(h, h0):
    d = np.abs(h - h0) % 360
    return np.minimum(d, 360 - d)

def label(mask):
    """4-connected components; returns (labels, n). Pure numpy union-find via scipy if present."""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        return lab, n
    except Exception:
        pass
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32); n = 0
    for y in range(H):
        for x in range(W):
            if mask[y, x] and lab[y, x] == 0:
                n += 1; stack = [(y, x)]; lab[y, x] = n
                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in ((cy+1, cx), (cy-1, cx), (cy, cx+1), (cy, cx-1)):
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and lab[ny, nx] == 0:
                            lab[ny, nx] = n; stack.append((ny, nx))
    return lab, n

def components(mask, min_area=1):
    lab, n = label(mask)
    out = []
    if n == 0:
        return lab, out
    idx = np.arange(1, n + 1)
    ys, xs = np.nonzero(lab)
    ls = lab[ys, xs]
    area = np.bincount(ls, minlength=n + 1)
    sy = np.bincount(ls, ys, minlength=n + 1); sx = np.bincount(ls, xs, minlength=n + 1)
    for i in idx:
        if area[i] >= min_area:
            out.append({'id': int(i), 'area': int(area[i]), 'cy': sy[i] / area[i], 'cx': sx[i] / area[i]})
    return lab, out

def blob_score(mask, img_area):
    """How much the mask looks like several small compact similar blobs."""
    lab, comps = components(mask, 4)
    if not comps:
        return 0.0, 0
    areas = np.array([c['area'] for c in comps], float)
    small = areas[(areas >= 6) & (areas <= 0.02 * img_area)]
    big = areas[areas > 0.02 * img_area].sum()
    if len(small) == 0:
        return 0.0, 0
    k = min(len(small), 15)
    score = k * np.sqrt(np.median(small)) / (1.0 + big / max(small.sum(), 1))
    return float(score), int(len(small))

class ColorModel:
    def __init__(self, hue, lab_mean, lab_std, s_min, v_min, hue_tol, L_lo=-1e9, L_hi=1e9):
        self.L_lo = L_lo; self.L_hi = L_hi
        self.hue = hue; self.lab_mean = lab_mean; self.lab_std = lab_std
        self.s_min = s_min; self.v_min = v_min; self.hue_tol = hue_tol

    def mask(self, rgb):
        h, s, v = rgb_to_hsv(rgb)
        m = (s >= self.s_min) & (v >= self.v_min) & (hue_dist(h, self.hue) <= self.hue_tol)
        if m.any():
            lab = rgb_to_lab(rgb)
            # Mahalanobis-like distance on the chroma channels (a,b), loose on L (lighting)
            da = (lab[..., 1] - self.lab_mean[1]) / max(self.lab_std[1], 3.0)
            db = (lab[..., 2] - self.lab_mean[2]) / max(self.lab_std[2], 3.0)
            m &= (da * da + db * db) <= 3.0 ** 2 * 2
            m &= (lab[..., 0] >= self.L_lo) & (lab[..., 0] <= self.L_hi)
        return m

    def to_dict(self):
        return {'hue': float(self.hue), 'lab_mean': [float(x) for x in self.lab_mean],
                'lab_std': [float(x) for x in self.lab_std], 's_min': float(self.s_min),
                'v_min': float(self.v_min), 'hue_tol': float(self.hue_tol),
                'L_lo': float(self.L_lo), 'L_hi': float(self.L_hi)}

def learn_color(images, extra_masks=None, log=print):
    """images: list of HxWx3 uint8 (frames where objects are visible).
    extra_masks: optional list of bool masks (e.g. height band) restricting candidates."""
    H = []; S = []; V = []; RGB = []
    for i, im in enumerate(images):
        h, s, v = rgb_to_hsv(im)
        sel = (s > 0.25) & (v > 0.15)
        if extra_masks is not None and extra_masks[i] is not None:
            sel &= extra_masks[i]
        H.append(h[sel]); S.append(s[sel]); V.append(v[sel]); RGB.append(im[sel])
    h = np.concatenate(H); s = np.concatenate(S); v = np.concatenate(V); rgb = np.concatenate(RGB)
    if len(h) < 20:
        return None
    hist = np.histogram(h, bins=36, range=(0, 360))[0].astype(float)
    table_bin = int(np.argmax(hist)) if extra_masks is None else None
    cands = []
    sm = np.convolve(np.concatenate([hist[-2:], hist, hist[:2]]), np.ones(3), 'valid')[1:-1]
    for b in range(36):
        if table_bin is not None and min(abs(b - table_bin), 36 - abs(b - table_bin)) <= 2:
            continue
        if sm[b] < 15:
            continue
        if sm[b] >= sm[(b - 1) % 36] and sm[b] >= sm[(b + 1) % 36]:
            cands.append(b * 10 + 5)
    best = None
    for hc in cands:
        for tol in (20, 30):
            sc_total = 0; ncomp = 0
            for i, im in enumerate(images):
                hh, ss, vv = rgb_to_hsv(im)
                m = (ss > 0.25) & (vv > 0.15) & (hue_dist(hh, hc) <= tol)
                if extra_masks is not None and extra_masks[i] is not None:
                    m &= extra_masks[i]
                sc, nc = blob_score(m, m.size)
                sc_total += sc; ncomp += nc
            log(f'  colour cand hue={hc} tol={tol} score={sc_total:.1f} comps={ncomp}')
            if best is None or sc_total > best[0]:
                best = (sc_total, hc, tol)
    if best is None:
        return None
    _, hc, tol = best
    sel = hue_dist(h, hc) <= tol
    # refine hue centre (circular mean)
    ang = np.deg2rad(h[sel])
    hc2 = float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360)
    sel = hue_dist(h, hc2) <= tol
    lab = rgb_to_lab(rgb[sel][None])[0]
    lm = np.median(lab, 0); ls = np.percentile(lab, 84, 0) - np.percentile(lab, 16, 0)
    s_min = max(0.18, float(np.percentile(s[sel], 3)) * 0.85)
    v_min = max(0.08, float(np.percentile(v[sel], 2)) * 0.8)
    L_lo = float(np.percentile(lab[:, 0], 2)) - 10.0
    L_hi = float(np.percentile(lab[:, 0], 97)) + 8.0
    cm = ColorModel(hc2, lm, ls / 2 + 2.0, s_min, v_min, tol + 5, L_lo, L_hi)
    log(f'  colour model {cm.to_dict()}')
    return cm
