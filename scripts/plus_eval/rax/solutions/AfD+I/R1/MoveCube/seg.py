"""Color segmentation of RoboMME-Plus MoveCube scenes (front camera, RGB uint8 HxWx3)."""
import numpy as np
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def rgb2hsv(im):
    """OpenCV-convention HSV (H 0..180, S,V 0..255) as int arrays."""
    if cv2 is not None:
        return cv2.cvtColor(np.ascontiguousarray(im.astype(np.uint8)), cv2.COLOR_RGB2HSV).astype(np.int32)
    f = im.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx = f.max(-1); mn = f.min(-1); d = mx - mn + 1e-9
    h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 30
    h = np.where(mx - mn < 1e-6, 0, h)
    s = np.where(mx > 0, (mx - mn) / (mx + 1e-9), 0)
    return np.stack([h, s * 255, mx * 255], -1).astype(np.int32)


def class_masks(im):
    hsv = rgb2hsv(im)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    r, g, b = [im[..., i].astype(np.int32) for i in range(3)]
    m = {}
    m['red'] = ((h <= 5) | (h >= 172)) & (s > 190) & (v > 60)
    m['purple'] = (h >= 132) & (h <= 162) & (s > 45) & (v > 90)
    m['blue'] = (h >= 93) & (h <= 107) & (s > 70) & (v > 100)
    m['yellow'] = (h >= 19) & (h <= 34) & (s > 110) & (v > 110)
    m['teal'] = (h >= 80) & (h <= 95) & (s > 90) & (v > 50) & (v < 200)
    m['pink'] = ((h <= 10) | (h >= 165)) & (s > 45) & (s < 125) & (v > 140)
    m['tan'] = (h >= 12) & (h <= 22) & (s > 40) & (s < 140) & (v > 90)
    m['dark'] = (s >= 12) & (s <= 85) & (v >= 50) & (v <= 128) & (h >= 84) & (h <= 112)
    m['white'] = (s < 40) & (v >= 135)
    return m


def components(mask, min_area=4):
    """Connected components -> list of dicts (area, centroid (u,v), pixel arrays)."""
    mask = mask.astype(np.uint8)
    out = []
    if cv2 is not None:
        n, lab, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            if st[i, 4] < min_area:
                continue
            ys, xs = np.nonzero(lab == i)
            out.append(dict(area=int(st[i, 4]), u=float(xs.mean()), v=float(ys.mean()), xs=xs, ys=ys,
                            bbox=(int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3]))))
    else:
        for ys, xs in _label_np(mask):
            if len(xs) < min_area:
                continue
            out.append(dict(area=len(xs), u=float(xs.mean()), v=float(ys.mean()), xs=xs, ys=ys,
                            bbox=(int(xs.min()), int(ys.min()), int(np.ptp(xs) + 1), int(np.ptp(ys) + 1))))
    out.sort(key=lambda c: -c['area'])
    return out


def _label_np(mask):
    """8-connected components without cv2/scipy (BFS). Yields (ys, xs) arrays."""
    from collections import deque
    m = mask.astype(bool)
    H, W = m.shape
    seen = np.zeros_like(m)
    ys0, xs0 = np.nonzero(m)
    for y0, x0 in zip(ys0, xs0):
        if seen[y0, x0]:
            continue
        q = deque([(y0, x0)]); seen[y0, x0] = True
        cy = []; cx = []
        while q:
            y, x = q.popleft(); cy.append(y); cx.append(x)
            for dy in (-1, 0, 1):
                yy = y + dy
                if yy < 0 or yy >= H:
                    continue
                for dx in (-1, 0, 1):
                    xx = x + dx
                    if 0 <= xx < W and m[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True; q.append((yy, xx))
        yield np.array(cy), np.array(cx)


def table_mask(im):
    hsv = rgb2hsv(im)
    h, s = hsv[..., 0], hsv[..., 1]
    return (h >= 5) & (h <= 17) & (s >= 118)


def dilate(mask, r=1):
    m = mask.astype(bool)
    out = m.copy()
    H, W = m.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            sh = np.zeros_like(m)
            ys = slice(max(0, dy), H + min(0, dy)); yd = slice(max(0, -dy), H + min(0, -dy))
            xs = slice(max(0, dx), W + min(0, dx)); xd = slice(max(0, -dx), W + min(0, -dx))
            sh[ys, xs] = m[yd, xd]
            out |= sh
    return out


def tool_groups(im, top_cut=14, robot_mask=None):
    """Find tool instances: connected non-table structures containing a teal or dark handle.
    Returns list of dicts with pixel arrays and per-class pixels."""
    ms = class_masks(im)
    nt = ~table_mask(im)
    other = ms['red'] | ms['purple'] | ms['blue'] | ms['yellow']
    tool = nt & ~other
    tool[:top_cut] = False
    if robot_mask is not None:
        tool &= ~robot_mask
    # remove robot base: big white blob touching the top area
    comps = components(dilate(tool, 1) & ~dilate(other, 1), min_area=8)
    groups = []
    for c in comps:
        xs, ys = c['xs'], c['ys']
        keep = tool[ys, xs]
        xs, ys = xs[keep], ys[keep]
        if len(xs) < 8:
            continue
        teal = ms['teal'][ys, xs]; dark = ms['dark'][ys, xs]; pink = ms['pink'][ys, xs]; tan = ms['tan'][ys, xs]
        white = ms['white'][ys, xs]
        if teal.sum() < 4 and dark.sum() < 6:
            continue
        if c['bbox'][1] <= top_cut + 2 and white.mean() > 0.5:
            continue
        groups.append(dict(xs=xs, ys=ys, teal=teal, dark=dark, pink=pink, tan=tan, white=white,
                           area=len(xs), u=float(xs.mean()), v=float(ys.mean())))
    return groups


def group_geometry(g):
    """Handle centroid, axis direction (handle->tip) and tip pixel in image space."""
    xs, ys = g['xs'].astype(float), g['ys'].astype(float)
    hsel = g['teal'] if g['teal'].sum() >= 4 else g['dark']
    hu, hv = xs[hsel].mean(), ys[hsel].mean()
    d = np.hypot(xs - hu, ys - hv)
    # tip = farthest pixel from handle centroid (robust: mean of farthest 3%)
    k = max(1, int(0.03 * len(xs)))
    idx = np.argsort(-d)[:k]
    tu, tv = xs[idx].mean(), ys[idx].mean()
    return dict(hu=hu, hv=hv, tu=tu, tv=tv, length_px=float(np.hypot(tu - hu, tv - hv)))


def tip_features(g, geom, frac=0.3):
    """Shape features of the tip region (the part of the tool beyond the handle, near the far end)."""
    xs, ys = g['xs'].astype(float), g['ys'].astype(float)
    ax = np.array([geom['tu'] - geom['hu'], geom['tv'] - geom['hv']]); L = np.linalg.norm(ax) + 1e-9
    ax = ax / L; pe = np.array([-ax[1], ax[0]])
    rel = np.stack([xs - geom['hu'], ys - geom['hv']], 1)
    along = rel @ ax; perp = rel @ pe
    nonhandle = ~(g['teal'] | g['dark'])
    sel = (along > (1 - frac) * L) & nonhandle
    if sel.sum() < 2:
        return dict(n=int(sel.sum()), w=0.0, fill=0.0, pinkfrac=0.0, along_ext=0.0)
    w = perp[sel].max() - perp[sel].min() + 1
    a_ext = along[sel].max() - along[sel].min() + 1
    fill = sel.sum() / (w * a_ext)
    return dict(n=int(sel.sum()), w=float(w), fill=float(fill), along_ext=float(a_ext),
                pinkfrac=float(g['pink'][sel].mean()), tanfrac=float(g['tan'][sel].mean()),
                L=float(L))


def tool_mask(im, ms=None, top_cut=14):
    ms = ms or class_masks(im)
    T = ~table_mask(im) & ~(ms['red'] | ms['purple'] | ms['blue'] | ms['yellow'])
    T &= ~dilate(ms['blue'], 2)
    T[:top_cut] = False
    return T


def _pca(xs, ys):
    P = np.stack([xs, ys], 1).astype(float)
    c = P.mean(0)
    if len(P) < 3:
        return c, np.array([1.0, 0.0]), 1.0, 1.0
    C = np.cov((P - c).T)
    w, V = np.linalg.eigh(C)
    e = V[:, 1]
    rel = (P - c) @ e
    relp = (P - c) @ V[:, 0]
    return c, e, float(rel.max() - rel.min() + 1), float(relp.max() - relp.min() + 1)


def find_handles(im, ms=None, exclude=None):
    ms = ms or class_masks(im)
    hs = []
    for kind, minar in (('teal', 5), ('dark', 15)):
        mk = ms[kind].copy()
        mk[:14] = False
        if kind == 'dark':
            mk &= ~dilate(ms['blue'], 2) & ~dilate(ms['teal'], 2)
        if exclude is not None:
            mk &= ~exclude
        for c in components(mk, min_area=minar):
            cen, e, L, W = _pca(c['xs'], c['ys'])
            hs.append(dict(kind=kind, u=float(cen[0]), v=float(cen[1]), axis=e, hlen=L, hwid=W,
                           area=c['area'], xs=c['xs'], ys=c['ys']))
    return hs


def march(T, c, e, start, maxlen=220, half=2, gap=4):
    """March along direction e from c (pixels); return last hit distance along e and hit list."""
    H, W = T.shape
    pe = np.array([-e[1], e[0]])
    last = start; miss = 0; hits = []
    t = start
    while t < maxlen:
        found = False
        for k in range(-half, half + 1):
            p = c + e * t + pe * k
            x, y = int(round(p[0])), int(round(p[1]))
            if 0 <= x < W and 0 <= y < H and T[y, x]:
                found = True
                break
        if found:
            last = t; miss = 0; hits.append(t)
        else:
            miss += 1
            if miss > gap:
                break
        t += 1
    return last, hits


def analyze_tool(im, h, T, others=()):
    """Shaft direction, tip point and tip shape for a handle h."""
    c = np.array([h['u'], h['v']])
    best = None
    for sgn in (1, -1):
        e = h['axis'] * sgn
        last, hits = march(T, c, e, start=h['hlen'] / 2)
        if best is None or last > best[0]:
            best = (last, e)
    last, e = best
    pe = np.array([-e[1], e[0]])
    end = c + e * last
    # tip region: T pixels within radius r of the end, beyond the handle
    ys, xs = np.nonzero(T)
    rel = np.stack([xs - c[0], ys - c[1]], 1).astype(float)
    along = rel @ e; perp = rel @ pe
    R = float(np.clip(2.6 * h['hwid'], 8.0, 24.0))
    near = (np.hypot(xs - end[0], ys - end[1]) < R) & (along > h['hlen'] / 2 + 1)
    # exclude pixels belonging to other handles' corridors
    for o in others:
        if o is h:
            continue
        oc = np.array([o['u'], o['v']]); oe = o.get('shaft', o['axis'])
        op = np.array([-oe[1], oe[0]])
        rel_o = np.stack([xs - oc[0], ys - oc[1]], 1).astype(float)
        d_o = np.abs(rel_o @ op)
        near &= ~((d_o < np.abs(perp) - 0.5) & (d_o < 3))
    sel_x, sel_y = xs[near], ys[near]
    if len(sel_x) >= 2:
        pa = along[near]; pp = perp[near]
        w = float(pp.max() - pp.min() + 1); a_ext = float(pa.max() - pa.min() + 1)
        fill = len(sel_x) / (w * a_ext)
        # far tip = farthest point from handle centroid
        dd = np.hypot(sel_x - c[0], sel_y - c[1])
        k = max(1, int(0.05 * len(dd)))
        idx = np.argsort(-dd)[:k]
        tip = np.array([sel_x[idx].mean(), sel_y[idx].mean()])
        perp_max = float(np.abs(pp).max())
    else:
        w = a_ext = fill = perp_max = 0.0
        tip = end
    feat = dict(length=float(last), tip_w=w, tip_along=a_ext, tip_fill=float(fill), tip_n=int(len(sel_x)),
                perp_max=perp_max, hwid=h['hwid'], hlen=h['hlen'])
    return dict(shaft=e, end=end, tip=tip, feat=feat, tip_xs=sel_x, tip_ys=sel_y)


def classify_tool(kind, f):
    """Return (type, score dict). Types: hook, ball, spatula, wood."""
    if kind == 'dark':
        return 'wood'
    hw = max(f['hwid'], 2.0)
    wr = f['tip_w'] / hw
    if f['tip_w'] >= 3.0 * hw and f['tip_fill'] >= 0.45:
        return 'spatula'
    if f['perp_max'] >= 2.2 * hw and f['tip_fill'] < 0.45:
        return 'hook'
    return 'ball'


def erode(mask, r=1):
    return ~dilate(~mask.astype(bool), r)


def opening(mask, r=1):
    return dilate(erode(mask, r), r)


def tip_blobs(im, ms=None, r=2):
    """Thick non-handle tool blobs (ball tips, spatula pads) surviving a morphological opening."""
    ms = ms or class_masks(im)
    T = tool_mask(im, ms)
    base = T & ~ms['teal'] & ~ms['dark'] & ~ms['white']
    op = opening(base, r)
    return components(op, min_area=4), op


def handle_shaft_dir(T, h):
    c = np.array([h['u'], h['v']])
    best = None
    for sgn in (1, -1):
        e = h['axis'] * sgn
        last, hits = march(T, c, e, start=h['hlen'] / 2)
        score = len(hits)
        if best is None or score > best[0]:
            best = (score, e, last)
    return best[1], best[2]


def label_tools(im, ms=None):
    """Return list of tools: dict(kind, type, handle(u,v), shaft dir, length_px, tip(u,v))."""
    ms = ms or class_masks(im)
    T = tool_mask(im, ms)
    hs = find_handles(im, ms)
    blobs, _ = tip_blobs(im, ms)
    tools = []
    for h in hs:
        e, last = handle_shaft_dir(T, h)
        h['shaft'] = e; h['shaft_len'] = last
    for h in hs:
        c = np.array([h['u'], h['v']]); e = h['shaft']; pe = np.array([-e[1], e[0]])
        best = None
        for b in blobs:
            rel = np.array([b['u'], b['v']]) - c
            al = rel @ e; pp = abs(rel @ pe)
            if al < h['hlen'] * 0.3 or pp > max(5.0, 0.6 * np.sqrt(b['area'])):
                continue
            if al > h['shaft_len'] + 14:
                continue
            if best is None or al < best[0]:
                best = (al, b)
        t = dict(kind=h['kind'], u=h['u'], v=h['v'], shaft=e, length_px=h['shaft_len'], hwid=h['hwid'],
                 hlen=h['hlen'], area=h['area'], handle_xs=h['xs'], handle_ys=h['ys'])
        if h['kind'] == 'dark':
            t['type'] = 'wood'
        elif best is None:
            t['type'] = 'hook'
        else:
            b = best[1]
            t['blob_area'] = b['area']; t['blob_bbox'] = b['bbox']
            ratio = b['area'] / max(h['hwid'], 2.0) ** 2
            t['blob_ratio'] = ratio
            t['type'] = 'spatula' if ratio > 4.0 else 'ball'
        tools.append(t)
    return tools


def end_analysis(T, h, e, last, others_lines=()):
    """Classify a teal tool by the structure at the far end of its shaft."""
    hw = max(h['hwid'], 3.0)
    c = np.array([h['u'], h['v']]); pe = np.array([-e[1], e[0]])
    E = c + e * last
    H, W = T.shape
    ys, xs = np.nonzero(T)
    rel = np.stack([xs - c[0], ys - c[1]], 1).astype(float)
    al = rel @ e; pp = rel @ pe
    dE = np.hypot(xs - E[0], ys - E[1])
    # remove pixels lying on other tools' shaft lines
    keep = np.ones(len(xs), bool)
    for (oc, oe, olen) in others_lines:
        op = np.array([-oe[1], oe[0]])
        r2 = np.stack([xs - oc[0], ys - oc[1]], 1).astype(float)
        oal = r2 @ oe; opp = np.abs(r2 @ op)
        keep &= ~((opp < 2.5) & (oal > -3) & (oal < olen + 3) & (np.abs(pp) > 2.5))
    R = 3.2 * hw
    region = keep & (dE < R) & (al > h['hlen'] / 2 + 2)
    side = region & (np.abs(pp) > 0.9 * hw)          # off-axis material near the end
    core = region & (dE < 1.2 * hw)
    n_side = int(side.sum()); n_core = int(core.sum())
    side_ext = float(np.abs(pp[side]).max()) if n_side else 0.0
    # 2-D area of thick blob near the end
    return dict(E=E, n_side=n_side, n_core=n_core, side_ext=side_ext, hw=hw,
                side_norm=n_side / hw ** 2, core_norm=n_core / hw ** 2, ext_norm=side_ext / hw)


def label_tools2(im, ms=None):
    ms = ms or class_masks(im)
    T = tool_mask(im, ms)
    hs = [h for h in find_handles(im, ms) if h['area'] >= (15 if h['kind'] == 'teal' else 25)]
    blobs, _ = tip_blobs(im, ms)
    lines = []
    for h in hs:
        c = np.array([h['u'], h['v']])
        best = None
        for sgn in (1, -1):
            e = h['axis'] * sgn
            last, hits = march(T, c, e, start=h['hlen'] / 2, gap=8)
            if best is None or len(hits) > best[0]:
                best = (len(hits), e, last)
        h['shaft'], h['shaft_len'] = best[1], best[2]
        lines.append((c, h['shaft'], h['shaft_len']))
    tools = []
    for i, h in enumerate(hs):
        e = h['shaft']; last = h['shaft_len']
        t = dict(kind=h['kind'], u=h['u'], v=h['v'], shaft=e, length_px=last, hwid=h['hwid'],
                 hlen=h['hlen'], area=h['area'], handle_xs=h['xs'], handle_ys=h['ys'])
        if h['kind'] == 'dark':
            t['type'] = 'wood'
        else:
            ea = end_analysis(T, h, e, last, [l for j, l in enumerate(lines) if j != i])
            t.update({k: v for k, v in ea.items() if k != 'E'}); t['end'] = ea['E']
            # thick blob at the end?
            c = np.array([h['u'], h['v']]); pe = np.array([-e[1], e[0]])
            bl = 0.0
            for b in blobs:
                rel = np.array([b['u'], b['v']]) - c
                if abs(rel @ pe) < max(4.0, 0.7 * np.sqrt(b['area'])) and h['hlen'] * 0.3 < rel @ e < last + 12:
                    bl = max(bl, b['area'] / h['hwid'] ** 2 if h['hwid'] > 0 else 0)
            t['blob_norm'] = bl
            if bl > 4.0 or (ea['core_norm'] > 1.6 and ea['side_norm'] > 3.0 and ea['ext_norm'] < 2.6):
                t['type'] = 'spatula'
            elif ea['ext_norm'] >= 1.9 and ea['side_norm'] > 0.6:
                t['type'] = 'hook'
            else:
                t['type'] = 'ball'
        tools.append(t)
    return tools


def label_tools3(im, ms=None):
    """Handle-based tools; teal tools typed by the pink component(s) attached at the shaft end."""
    ms = ms or class_masks(im)
    T = tool_mask(im, ms)
    hs = [h for h in find_handles(im, ms) if h['area'] >= (15 if h['kind'] == 'teal' else 25)]
    pink = ms['pink'] & T
    pcs = components(pink, min_area=2)
    tools = []
    for h in hs:
        c = np.array([h['u'], h['v']])
        best = None
        for sgn in (1, -1):
            e = h['axis'] * sgn
            last, hits = march(T, c, e, start=h['hlen'] / 2, gap=8)
            if best is None or len(hits) > best[0]:
                best = (len(hits), e, last)
        e, last = best[1], best[2]
        pe = np.array([-e[1], e[0]])
        hw = max(h['hwid'], 3.0)
        t = dict(kind=h['kind'], u=h['u'], v=h['v'], shaft=e, length_px=float(last), hwid=float(h['hwid']),
                 hlen=float(h['hlen']), area=h['area'], handle_xs=h['xs'], handle_ys=h['ys'])
        if h['kind'] == 'dark':
            t['type'] = 'wood'
            tools.append(t)
            continue
        sel_x = []; sel_y = []
        for pc in pcs:
            rel = np.stack([pc['xs'] - c[0], pc['ys'] - c[1]], 1).astype(float)
            al = rel @ e; pp = rel @ pe
            touch = (np.abs(pp) < max(2.5, 0.8 * hw)) & (al > last - 3.5 * hw) & (al < last + 2.5 * hw) & (al > h['hlen'] / 2)
            if touch.any():
                sel_x.append(pc['xs']); sel_y.append(pc['ys'])
        if sel_x:
            xs = np.concatenate(sel_x); ys = np.concatenate(sel_y)
            rel = np.stack([xs - c[0], ys - c[1]], 1).astype(float)
            al = rel @ e; pp = rel @ pe
            perp_ext = float(np.abs(pp).max())
            n = len(xs)
            t.update(pink_n=n, pink_perp=perp_ext / hw, pink_norm=n / hw ** 2,
                     tip=(float(xs.mean()), float(ys.mean())), far=float(al.max()), pink_xs=xs, pink_ys=ys)
            if n / hw ** 2 > 3.5 and perp_ext / hw < 2.6:
                t['type'] = 'spatula'
            elif perp_ext / hw >= 1.45:
                t['type'] = 'hook'
            else:
                t['type'] = 'ball'
        else:
            t.update(pink_n=0, pink_perp=0.0, pink_norm=0.0)
            t['type'] = 'unknown'
        tools.append(t)
    return tools


def yellow_balls(im, ms=None):
    ms = ms or class_masks(im)
    y = ms['yellow'].copy(); y[:14] = False
    return [c for c in components(y, min_area=6)]


def persistence(mask_xs, mask_ys, cls_mask):
    if len(mask_xs) == 0:
        return 1.0
    return float(cls_mask[mask_ys, mask_xs].mean())


def _onset(p, thr=0.5, run=3):
    """First index from which persistence stays below thr for `run` samples (None if never)."""
    p = np.asarray(p)
    for i in range(1, len(p)):
        seg_ = p[i:i + run]
        if len(seg_) >= min(run, len(p) - 1) and np.all(seg_ < thr):
            return i
    return None


def classify_video(frames, log=print):
    """frames: list of front RGB arrays in time order spanning the whole conditioning video.
    The demonstrated object is the one whose frame-0 pixels vanish FIRST (persistently).
    Returns dict(manner, tool (from frame0) or None, tools, pers)."""
    f0 = frames[0]
    ms0 = class_masks(f0)
    tools = label_tools3(f0, ms0)
    balls = yellow_balls(f0, ms0)
    lms = [class_masks(f) for f in frames[1:] if f.shape[:2] == f0.shape[:2]]
    n = len(lms)
    objs = [('tool', i, t['handle_xs'], t['handle_ys'], t['kind']) for i, t in enumerate(tools)] + \
           [('ball', i, b['xs'], b['ys'], 'yellow') for i, b in enumerate(balls)]
    rows = []
    for kind, i, xs, ys, cls in objs:
        p = [persistence(xs, ys, m[cls]) for m in lms]
        on = _onset(p)
        late = float(np.median(p[int(n * 0.7):])) if n >= 4 else float(p[-1]) if p else 1.0
        rows.append(((on if on is not None else 10 ** 6), late, kind, i))
    rows.sort()
    log('video tools', [(t['type'], round(t['u']), round(t['v'])) for t in tools],
        'balls', [(round(b['u']), round(b['v'])) for b in balls],
        'onsets(onset,late,kind,i)', [(r[0] if r[0] < 10 ** 6 else None, round(r[1], 2), r[2], r[3]) for r in rows], 'n', n)
    if not rows:
        return dict(manner='unknown', tool=None, tools=tools)
    on, p, kind, i = rows[0]
    if on >= 10 ** 6:
        # nothing vanished persistently: fall back to the lowest late persistence
        rows.sort(key=lambda r: r[1])
        on, p, kind, i = rows[0]
    if kind == 'ball':
        return dict(manner='ramp', tool=None, tools=tools, pers=p, ball=balls[i])
    t = tools[i]
    return dict(manner=t['type'], tool=t, tools=tools, pers=p)
