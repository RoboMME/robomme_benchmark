"""Colour segmentation for MoveCube front-camera images (256x256 RGB uint8)."""
import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def rgb2hsv(im):
    """OpenCV-style HSV (H 0..180, S,V 0..255) without requiring cv2."""
    if cv2 is not None:
        return cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_RGB2HSV).astype(np.int16)
    f = im.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx = f.max(-1); mn = f.min(-1); d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-6
    rr = m & (mx == r); gg = m & (mx == g) & ~rr; bb = m & ~rr & ~gg
    h[rr] = ((g - b)[rr] / d[rr]) % 6
    h[gg] = ((b - r)[gg] / d[gg]) + 2
    h[bb] = ((r - g)[bb] / d[bb]) + 4
    h = h * 30.0
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0) * 255
    return np.stack([h, s, mx * 255], -1).astype(np.int16)


def masks(im):
    hsv = rgb2hsv(im)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    R, G, B = [im[..., i].astype(np.int16) for i in range(3)]
    m = {}
    m['red'] = (R > 110) & (G < 70) & (B < 70) & (R > 2 * G + 40)
    m['target'] = (H >= 125) & (H <= 165) & (S > 35) & (V > 90)
    m['yellow'] = (H >= 17) & (H <= 34) & (S > 110) & (V > 120)
    m['ramp'] = (H >= 96) & (H <= 112) & (S > 60) & (S < 175) & (V > 120)
    m['teal'] = (H >= 83) & (H <= 96) & (S > 90) & (V > 70) & (V < 200)
    m['slate'] = (H >= 94) & (H <= 118) & (S >= 20) & (S <= 95) & (V >= 45) & (V <= 115)
    rb = R - B
    ratio = (G - B).astype(np.float32) / np.maximum(rb, 1).astype(np.float32)
    m['pink'] = (ratio < 0.295) & (ratio > 0.05) & (rb >= 45) & (rb <= 112) & (R >= 140) & (G <= R - 25)
    m['gray'] = (S < 45) & (V > 140)
    m['tan'] = (H >= 13) & (H <= 20) & (S >= 70) & (S <= 128) & (V >= 165)
    return m


def components(mask, min_area=1):
    """Connected components -> list of dicts (area, centroid (u,v), pixels)."""
    mask = mask.astype(np.uint8)
    out = []
    if cv2 is not None:
        n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a < min_area:
                continue
            vs, us = np.nonzero(lab == i)
            out.append({'area': a, 'u': float(cents[i][0]), 'v': float(cents[i][1]), 'us': us, 'vs': vs})
    else:
        from scipy import ndimage
        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        for i in range(1, n + 1):
            vs, us = np.nonzero(lab == i)
            if len(us) < min_area:
                continue
            out.append({'area': len(us), 'u': float(us.mean()), 'v': float(vs.mean()), 'us': us, 'vs': vs})
    out.sort(key=lambda c: -c['area'])
    return out


def table_like(im, hsv=None):
    if hsv is None:
        hsv = rgb2hsv(im)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (H >= 4) & (H <= 18) & (S >= 128) & (V >= 60)


def ray_profile(ok, u0, v0, ang, r0=3, rmax=110, gap=3):
    """March from (u0,v0) along angle; return (length reached, hits) allowing small gaps."""
    h, w = ok.shape
    du, dv = np.cos(ang), np.sin(ang)
    last, miss, hits = 0, 0, 0
    for r in range(r0, rmax):
        u = int(round(u0 + du * r)); v = int(round(v0 + dv * r))
        if u < 0 or v < 0 or u >= w or v >= h:
            break
        if ok[v, u]:
            last = r; miss = 0; hits += 1
        else:
            miss += 1
            if miss > gap:
                break
    return last, hits


def find_tools(im, m=None):
    """Detect stick tools. Returns list of dicts with kind, handle (u,v), end (u,v), axis angle."""
    hsv = rgb2hsv(im)
    if m is None:
        m = masks(im)
    tab = table_like(im, hsv)
    tools = []
    # thin, non-table, non-red, non-target pixels may belong to shafts/ends
    shaftish = (~tab) & (~m['red']) & (~m['target']) & (~m['yellow'])
    h, w = tab.shape
    shaftish[:14, :] = False  # top background band
    for kind, mk, minarea in (('teal', m['teal'], 10), ('slate', m['slate'], 10)):
        for c in components(mk, minarea):
            if c['area'] > 400:
                continue
            u0, v0 = c['u'], c['v']
            best = None
            for k in range(120):
                ang = k * np.pi / 60.0
                L, hits = ray_profile(shaftish & ~mk, u0, v0, ang, r0=int(2 + np.sqrt(c['area']) / 2))
                score = hits
                if best is None or score > best[0]:
                    best = (score, L, ang)
            score, L, ang = best
            if L < 15:
                continue
            eu, ev = u0 + np.cos(ang) * L, v0 + np.sin(ang) * L
            tools.append({'part': kind, 'handle': (u0, v0), 'end': (eu, ev), 'angle_img': ang,
                          'len_px': L, 'handle_area': c['area'], 'comp': c})
    for t in tools:
        t['end_kind'], t['end_feat'] = classify_end(im, m, t)
    return tools


def row_scale(v):
    """Apparent size factor of a table object at image row v relative to row 60 (fixed front camera)."""
    return max(0.5, (8.5 + 0.06 * float(v)) / 12.1)


def _hull_area(us, vs):
    pts = np.stack([us, vs], 1).astype(np.float32)
    if len(pts) < 3:
        return float(len(pts))
    if cv2 is not None:
        hull = cv2.convexHull(pts)
        return float(cv2.contourArea(hull)) + 0.5 * len(hull) + 1.0
    from scipy.spatial import ConvexHull
    try:
        h = ConvexHull(pts)
        return float(h.volume) + 0.5 * len(h.vertices) + 1.0
    except Exception:
        return float(len(pts))


def end_features(im, m, t):
    u0, v0 = t['handle']; ang = t['angle_img']; L = t['len_px']
    a = np.array([np.cos(ang), np.sin(ang)]); p = np.array([-a[1], a[0]])
    s = row_scale(t['end'][1])
    vs, us = np.nonzero(m['pink'])
    empty = {'n': 0.0, 'spread': 0.0, 'along': 0.0, 'fill': 0.0, 'solidity': 0.0, 'gray': 0.0}
    if len(us) == 0:
        return empty
    d = np.stack([us - u0, vs - v0], 1).astype(float)
    along = d @ a; perp = d @ p
    sel = (along > L - 12 * s) & (along < L + 7 * s) & (np.abs(perp) <= 10 * s)
    n = int(sel.sum())
    if n < 3:
        e = dict(empty); e['n'] = n / s ** 2
        return e
    pa = perp[sel]; al = along[sel]
    spread = float(np.percentile(pa, 95) - np.percentile(pa, 5)) + 1
    dep = float(np.percentile(al, 95) - np.percentile(al, 5)) + 1
    # largest pink component inside the window -> solidity
    win = np.zeros(m['pink'].shape, bool)
    win[vs[sel], us[sel]] = True
    comps = components(win, 1)
    c = comps[0]
    sol = c['area'] / _hull_area(c['us'], c['vs'])
    # gray rim pixels in the same window (paddle has a light rim)
    gv, gu = np.nonzero(m['gray'])
    g = 0
    if len(gu):
        dg = np.stack([gu - u0, gv - v0], 1).astype(float)
        ga = dg @ a; gp = dg @ p
        g = int(((ga > L - 12 * s) & (ga < L + 7 * s) & (np.abs(gp) <= 10 * s) & (np.abs(gp) > 2.5 * s)).sum())
    asym = abs(float(np.mean(pa))) / s
    return {'n': n / s ** 2, 'spread': spread / s, 'along': dep / s, 'fill': n / (spread * dep),
            'solidity': float(sol), 'gray': g / s ** 2, 'asym': asym}


def class_scores(f):
    """Heuristic scores per teal-tool class from normalised pink-end features.
    hook: pink bar offset to one side of the shaft axis (L/U shape) -> large asym.
    paddle: wide, large filled plate. ball: compact blob centred on the axis."""
    n, sp, asym = f['n'], f['spread'], f.get('asym', 0.0)
    hook = (asym - 3.0) / 0.6
    paddle = min((sp - 10.0) / 1.0, (n - 55.0) / 8.0) - max(0.0, asym - 3.0) / 0.6
    ball = min(-(asym - 3.0) / 0.6, -min((sp - 10.0) / 1.0, (n - 55.0) / 8.0))
    return {'hook': hook, 'paddle': paddle, 'ball': ball}


def classify_end(im, m, t):
    """Classify a tool's working end: hammer / paddle / hook / ball."""
    if t['part'] == 'slate':
        return 'hammer', {}
    f = end_features(im, m, t)
    if f['n'] < 2:
        return 'unknown', f
    sc = class_scores(f)
    f['scores'] = sc
    return max(sc, key=sc.get), f


def find_hammers(im, m=None):
    """Hammer = dark slate head bar with a tan wooden handle nearby."""
    if m is None:
        m = masks(im)
    out = []
    tan_c = [c for c in components(m['tan'], 6)]
    for c in components(m['slate'], 25):
        if c['area'] > 600:
            continue
        us, vs = c['us'].astype(float), c['vs'].astype(float)
        best = None
        for t in tan_c:
            d = np.min((us[:, None] - t['us'][None, :]) ** 2 + (vs[:, None] - t['vs'][None, :]) ** 2)
            if d <= 15 ** 2 and (best is None or t['area'] > best['area']):
                best = t
        if best is None:
            continue
        # head principal axis
        pts = np.stack([us - us.mean(), vs - vs.mean()], 1)
        w, V = np.linalg.eigh(pts.T @ pts / len(pts))
        ax = V[:, 1]
        # handle: tan pixels; handle tip = tan pixel farthest from head centroid
        tu, tv = best['us'].astype(float), best['vs'].astype(float)
        dd = (tu - c['u']) ** 2 + (tv - c['v']) ** 2
        k = int(np.argmax(dd))
        out.append({'part': 'slate', 'end_kind': 'hammer', 'head': (c['u'], c['v']), 'head_axis_img': float(np.arctan2(ax[1], ax[0])),
                    'head_elong': float(np.sqrt(w[1] / max(w[0], 1e-3))), 'handle_tip': (float(tu[k]), float(tv[k])),
                    'tan_centroid': (best['u'], best['v']), 'comp': c, 'tan_comp': best, 'area': c['area']})
    out.sort(key=lambda h: -h['area'])
    return out


def scene_objects(im):
    """All MoveCube-relevant objects in a front image."""
    m = masks(im)
    objs = {}
    tools = [t for t in find_tools(im, m) if t['part'] == 'teal']
    objs['teal_tools'] = tools
    objs['hammers'] = find_hammers(im, m)
    ys = [c for c in components(m['yellow'], 8) if c['v'] > 12]
    objs['ball'] = ys[0] if ys else None
    rs = [c for c in components(m['ramp'], 150)]
    objs['ramp'] = rs[0] if rs else None
    reds = [c for c in components(m['red'], 12) if c['v'] > 12]
    objs['cube'] = reds[0] if reds else None
    tg = [c for c in components(m['target'], 10) if c['v'] > 12]
    objs['target'] = tg[0] if tg else None
    objs['masks'] = m
    return objs


def _presence(im_k, us, vs, key):
    m = masks(im_k)
    tab = table_like(im_k)
    if key == 'hammer':
        mk = m['slate'] | m['tan']
    else:
        mk = m[key]
    same = float(mk[vs, us].mean()) if len(us) else 0.0
    table = float(tab[vs, us].mean()) if len(us) else 0.0
    return same, table


def classify_video(front_frames, log=None):
    """Decide the manner shown in the conditioning video.

    front_frames: list of RGB arrays (video_demo phase, in order).
    Returns (manner, info). manner in {'ramp','hook','ball','paddle','hammer','push'} or None.
    """
    if not front_frames:
        return None, {}
    f0 = front_frames[0]
    objs = scene_objects(f0)
    cands = []
    if objs['ball'] is not None:
        c = objs['ball']; cands.append(('ramp', c['us'], c['vs'], 'yellow', {}))
    for t in objs['teal_tools']:
        c = t['comp']; cands.append((t['end_kind'], c['us'], c['vs'], 'teal', t))
    for h in objs['hammers']:
        us = np.concatenate([h['comp']['us'], h['tan_comp']['us']])
        vs = np.concatenate([h['comp']['vs'], h['tan_comp']['vs']])
        cands.append(('hammer', us, vs, 'hammer', h))
    n = len(front_frames)
    idx = sorted(set(int(round(x)) for x in np.linspace(0, n - 1, min(n, 40))))
    scores = []
    for kind, us, vs, key, t in cands:
        moved = 0.0
        for k in idx[1:]:
            same, table = _presence(front_frames[k], us, vs, key)
            # moved away: original pixels now show bare table, not the object
            moved = max(moved, table * (1.0 - same))
        scores.append(moved)
        if log:
            log(f"video cand {kind} at ({us.mean():.0f},{vs.mean():.0f}) moved={moved:.2f}")
    # did the cube move at all?
    cube_moved = 0.0
    if objs['cube'] is not None:
        c = objs['cube']
        for k in idx[1:]:
            same, table = _presence(front_frames[k], c['us'], c['vs'], 'red')
            cube_moved = max(cube_moved, table * (1.0 - same))
    info = {'scores': [(c[0], s) for c, s in zip(cands, scores)], 'objs': objs, 'chosen': None, 'cube_moved': cube_moved}
    if log:
        log(f'video cube moved={cube_moved:.2f}')
    if not cands or max(scores) < 0.2:
        if cube_moved >= 0.4:
            return 'push', info
        return None, info
    i = int(np.argmax(scores))
    if cands[i][0] == 'ramp':
        tool_i = [j for j in range(len(cands)) if cands[j][0] != 'ramp']
        if tool_i:
            j = max(tool_i, key=lambda j: scores[j])
            if scores[j] >= 0.5:
                i = j
    info['chosen'] = cands[i][4]
    return cands[i][0], info
