"""Pure-numpy perception helpers (no simulator access). Tested offline on demo frames."""
import numpy as np

def hsv(rgb):
    a = rgb.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1); mn = a.min(-1); d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-6
    rm = m & (mx == r); gm = m & (mx == g) & ~rm; bm = m & ~rm & ~gm
    h[rm] = ((g - b)[rm] / d[rm]) % 6
    h[gm] = ((b - r)[gm] / d[gm]) + 2
    h[bm] = ((r - g)[bm] / d[bm]) + 4
    h = h * 60.0
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    return h, s, mx

COLOR_RULES = {
    'red':    lambda h, s, v: ((h < 10) | (h > 345)) & (s > 0.6) & (v > 0.3),
    'green':  lambda h, s, v: (h > 85) & (h < 160) & (s > 0.45) & (v > 0.25),
    'blue':   lambda h, s, v: (h > 200) & (h < 255) & (s > 0.45) & (v > 0.2),
    'yellow': lambda h, s, v: (h > 45) & (h < 72) & (s > 0.5) & (v > 0.35),
    'purple': lambda h, s, v: (h > 265) & (h < 325) & (s > 0.25) & (v > 0.35),
    'orange': lambda h, s, v: (h > 14) & (h < 40) & (s > 0.7) & (v > 0.6),
    'cyan':   lambda h, s, v: (h > 165) & (h < 200) & (s > 0.45) & (v > 0.3),
    'white':  lambda h, s, v: (s < 0.09) & (v > 0.84),
    'grey':   lambda h, s, v: (s < 0.09) & (v > 0.55) & (v < 0.76),
}

def color_mask(rgb, name):
    h, s, v = hsv(rgb)
    return COLOR_RULES[name](h, s, v)

def label(mask):
    """Connected components (4-neighbour). Returns (labels, n)."""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        return lab, n
    except Exception:
        pass
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32); n = 0
    for y0 in range(H):
        for x0 in range(W):
            if mask[y0, x0] and lab[y0, x0] == 0:
                n += 1; stack = [(y0, x0)]; lab[y0, x0] = n
                while stack:
                    y, x = stack.pop()
                    for yy, xx in ((y+1, x), (y-1, x), (y, x+1), (y, x-1)):
                        if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and lab[yy, xx] == 0:
                            lab[yy, xx] = n; stack.append((yy, xx))
    return lab, n

def blobs(mask, min_area=1):
    lab, n = label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < min_area:
            continue
        out.append({'area': int(len(ys)), 'u': float(xs.mean()), 'v': float(ys.mean()),
                    'umin': int(xs.min()), 'umax': int(xs.max()), 'vmin': int(ys.min()), 'vmax': int(ys.max()),
                    'xs': xs, 'ys': ys})
    out.sort(key=lambda b: -b['area'])
    return out

def dilate(mask, r):
    m = mask.copy()
    for _ in range(r):
        n = m.copy()
        n[1:] |= m[:-1]; n[:-1] |= m[1:]; n[:, 1:] |= m[:, :-1]; n[:, :-1] |= m[:, 1:]
        m = n
    return m

def nearest_inside(b):
    """Pixel of blob nearest its centroid (centroid may fall outside non-convex blobs)."""
    d = (b['xs'] - b['u']) ** 2 + (b['ys'] - b['v']) ** 2
    k = int(np.argmin(d))
    return float(b['xs'][k]), float(b['ys'][k])

def interior_pixels(b, k=6):
    """Pixels of the blob farthest from its boundary approx: those closest to centroid."""
    d = (b['xs'] - b['u']) ** 2 + (b['ys'] - b['v']) ** 2
    idx = np.argsort(d)[:k]
    return [(float(b['xs'][i]), float(b['ys'][i])) for i in idx]

def blob_axis_angle(b):
    """Edge orientation of a roughly square blob in image (radians, mod pi/2) via min-area-rect search."""
    xs = b['xs'].astype(float) - b['u']; ys = b['ys'].astype(float) - b['v']
    best = None
    for deg in np.arange(0, 90, 1.0):
        t = np.deg2rad(deg)
        c, s = np.cos(t), np.sin(t)
        p = xs * c + ys * s; q = -xs * s + ys * c
        area = (p.max() - p.min() + 1) * (q.max() - q.min() + 1)
        if best is None or area < best[0]:
            best = (area, t)
    return best[1]

def front_detect(rgb, cube_color, elev=None):
    """Detect target cube, target bullseye and button in the front image.
    elev: optional boolean mask of pixels elevated above the table plane."""
    H, W = rgb.shape[:2]
    res = {'cube': None, 'target': None, 'button': None, 'others': []}
    cm = color_mask(rgb, cube_color)
    cb = [b for b in blobs(cm, 6) if b['area'] < 2500]
    if elev is not None:
        cb2 = [b for b in cb if elev[b['ys'], b['xs']].mean() > 0.3]
        if cb2:
            cb = cb2
    if cb:
        res['cube'] = cb[0]
    pm = color_mask(rgb, 'purple')
    pb = [b for b in blobs(pm, 4) if b['area'] < 3000]
    if pb:
        # merge purple blobs near largest (rings can split)
        b0 = pb[0]
        near = [b for b in pb if abs(b['u'] - b0['u']) < 14 and abs(b['v'] - b0['v']) < 10]
        xs = np.concatenate([b['xs'] for b in near]); ys = np.concatenate([b['ys'] for b in near])
        res['target'] = {'area': int(len(xs)), 'u': float(xs.mean()), 'v': float(ys.mean()), 'xs': xs, 'ys': ys,
                         'umin': int(xs.min()), 'umax': int(xs.max()), 'vmin': int(ys.min()), 'vmax': int(ys.max())}
    wm = color_mask(rgb, 'white')
    if res['target'] is not None:
        tm = np.zeros((H, W), bool); tm[res['target']['ys'], res['target']['xs']] = True
        wm &= ~dilate(tm, 4)
    wm[:10] = False
    gm = color_mask(rgb, 'grey') | (hsv(rgb)[1] < 0.09) & (hsv(rgb)[2] > 0.35) & (hsv(rgb)[2] < 0.76)
    cands = []
    for b in blobs(wm, 6):
        if b['area'] > 1500:
            continue
        u0, u1, v0, v1 = b['umin'], b['umax'], b['vmin'], b['vmax']
        box = gm[max(v0-2, 0):v1+3, max(u0-2, 0):u1+3]
        grey_frac = box.mean() if box.size else 0
        score = b['area'] * (0.3 + grey_frac)
        if elev is not None:
            e = elev[max(v0-2, 0):v1+3, max(u0-2, 0):u1+3].mean()
            score *= (0.2 + e)
        cands.append((score, b, grey_frac))
    cands.sort(key=lambda c: -c[0])
    if cands:
        res['button'] = cands[0][1]
        res['button']['grey_frac'] = float(cands[0][2])
    return res

def wrist_candidates(rgb, what, cube_color=None):
    """Candidate blobs for 'cube' (cube_color), 'target' (purple) or 'button' (white base) in a wrist
    image, best first. Excludes the bottom gripper-finger regions."""
    H, W = rgb.shape[:2]
    if what == 'cube':
        m = color_mask(rgb, cube_color)
    elif what == 'target':
        m = color_mask(rgb, 'purple')
    else:
        m = color_mask(rgb, 'white')
    m[int(H * 0.80):, :int(W * 0.40)] = False
    m[int(H * 0.80):, int(W * 0.70):] = False
    bl = blobs(m, 15)
    if not bl:
        return []
    if what == 'target':
        b0 = bl[0]
        near = [b for b in bl if abs(b['u'] - b0['u']) < 60 and abs(b['v'] - b0['v']) < 60]
        xs = np.concatenate([b['xs'] for b in near]); ys = np.concatenate([b['ys'] for b in near])
        cu, cv = float((xs.min() + xs.max()) / 2), float((ys.min() + ys.max()) / 2)
        out = [{'area': int(len(xs)), 'u': cu, 'v': cv, 'xs': xs, 'ys': ys, 'umin': int(xs.min()),
                'umax': int(xs.max()), 'vmin': int(ys.min()), 'vmax': int(ys.max())}]
        for b in bl:
            if b not in near:
                out.append(b)
        return out
    if what == 'button':
        sc = []
        for b in bl:
            w = b['umax'] - b['umin'] + 1; h = b['vmax'] - b['vmin'] + 1
            fill = b['area'] / float(w * h)
            sq = min(w, h) / float(max(w, h))
            b = dict(b); b['u'] = (b['umin'] + b['umax']) / 2.0; b['v'] = (b['vmin'] + b['vmax']) / 2.0
            sc.append((b['area'] * sq * (0.5 + fill), b))
        sc.sort(key=lambda t: -t[0])
        return [b for _, b in sc]
    return bl


def wrist_detect(rgb, what, cube_color=None):
    c = wrist_candidates(rgb, what, cube_color)
    return c[0] if c else None
