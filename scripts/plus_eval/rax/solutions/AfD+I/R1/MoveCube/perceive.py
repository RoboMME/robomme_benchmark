"""3-D scene perception from the front camera (RGB-D)."""
import numpy as np
from mcpio import log
from geom import tool_points
import seg


class Proj:
    """Pixel -> world, own calibration if validated, else the MCP tool (subsampled)."""
    def __init__(self, ctx, obs, cam_model, cam='front'):
        self.ctx, self.obs, self.cm, self.cam = ctx, obs, cam_model, cam

    def pts(self, xs, ys, max_tool=48):
        xs = np.asarray(xs).astype(int); ys = np.asarray(ys).astype(int)
        if len(xs) == 0:
            return np.zeros((0, 3))
        if self.cm is not None and self.cam in self.obs.depth:
            self.cm.depth = self.obs.depth[self.cam]
            P = self.cm.pixels_to_world(xs, ys)
            return P[np.all(np.isfinite(P), 1)]
        if len(xs) > max_tool:
            idx = np.linspace(0, len(xs) - 1, max_tool).round().astype(int)
            xs, ys = xs[idx], ys[idx]
        res = tool_points(self.ctx, self.obs.step_id, [(self.cam, x, y) for x, y in zip(xs, ys)], radius=0)
        P = np.array([p for p in res if p is not None]) if any(p is not None for p in res) else np.zeros((0, 3))
        return P


def robust_mean(P):
    if len(P) == 0:
        return None
    med = np.median(P, 0)
    d = np.linalg.norm(P - med, axis=1)
    keep = d <= max(np.percentile(d, 80), 1e-4)
    return P[keep].mean(0)


def table_height(proj, im, ms):
    T = seg.table_mask(im)
    T[:20] = False; T[200:] = False
    ys, xs = np.nonzero(T)
    if len(xs) == 0:
        return None
    idx = np.random.RandomState(1).choice(len(xs), min(400, len(xs)), replace=False)
    P = proj.pts(xs[idx], ys[idx])
    if len(P) == 0:
        return None
    return float(np.median(P[:, 2]))


def perceive_scene(proj, im, table_z=None):
    ms = seg.class_masks(im)
    sc = dict(ms=ms)
    if table_z is None:
        table_z = table_height(proj, im, ms)
    sc['table_z'] = table_z
    # cube
    red = ms['red'].copy(); red[:14] = False
    cs = seg.components(red, min_area=6)
    sc['cube'] = None
    if cs:
        c = cs[0]
        P = proj.pts(c['xs'], c['ys'])
        if len(P) >= 3:
            top = np.percentile(P[:, 2], 90)
            topP = P[P[:, 2] > top - 0.006]
            ctr = robust_mean(topP)
            side = float(np.clip(top - (table_z if table_z is not None else top - 0.04), 0.015, 0.08))
            sc['cube'] = dict(xy=ctr[:2], top=float(top), side=side, px=(c['u'], c['v']), area=c['area'])
    # target
    pu = ms['purple'].copy(); pu[:14] = False
    ts = seg.components(pu, min_area=6)
    sc['target'] = None
    if ts:
        t = ts[0]
        P = proj.pts(t['xs'], t['ys'])
        if len(P) >= 3:
            ctr = robust_mean(P)
            sc['target'] = dict(xy=ctr[:2], z=float(ctr[2]), px=(t['u'], t['v']), area=t['area'])
    # balls
    sc['balls'] = []
    for b in seg.yellow_balls(im, ms)[:3]:
        P = proj.pts(b['xs'], b['ys'])
        if len(P) < 3:
            continue
        top = float(np.percentile(P[:, 2], 95))
        r = float(np.clip((top - table_z) / 2 if table_z is not None else 0.02, 0.008, 0.04))
        cap = P[P[:, 2] > top - r * 0.8]
        ctr = robust_mean(cap if len(cap) >= 2 else P)
        sc['balls'].append(dict(xy=ctr[:2], top=top, r=r, px=(b['u'], b['v']), area=b['area']))
    # ramp
    sc['ramp'] = None
    bl = ms['blue'].copy(); bl[:14] = False
    rs = seg.components(bl, min_area=80)
    if rs:
        r = rs[0]
        idx = np.linspace(0, len(r['xs']) - 1, min(600, len(r['xs']))).round().astype(int)
        P = proj.pts(r['xs'][idx], r['ys'][idx], max_tool=96)
        if len(P) >= 10:
            sc['ramp'] = dict(P=P, px=(r['u'], r['v']), area=r['area'], top=float(np.percentile(P[:, 2], 98)))
    # tools
    sc['tools'] = []
    try:
        tools = seg.label_tools3(im, ms)
    except Exception as e:
        log('label_tools3 failed', e); tools = []
    T = seg.tool_mask(im, ms)
    for t in tools:
        try:
            tool3d(proj, im, ms, T, t, table_z, unit=sc['cube']['side'] if sc['cube'] is not None else None)
            if t.get('grasp') is not None:
                sc['tools'].append(t)
        except Exception as e:
            log('tool3d failed', e)
    return sc


def tool3d(proj, im, ms, T, t, table_z, unit=None):
    hp = proj.pts(t['handle_xs'], t['handle_ys'])
    if len(hp) < 3:
        t['grasp'] = None; return
    g = robust_mean(hp)
    top = float(np.percentile(hp[:, 2], 90))
    # handle axis in world xy
    Q = hp[:, :2] - hp[:, :2].mean(0)
    w, V = np.linalg.eigh(np.cov(Q.T)) if len(Q) >= 3 else (None, np.eye(2))
    ax = V[:, 1] if w is not None else np.array([1.0, 0.0])
    t['grasp'] = np.array([g[0], g[1], top])
    t['handle_axis'] = ax / (np.linalg.norm(ax) + 1e-9)
    t['handle_top'] = top
    # tip
    c = np.array([t['u'], t['v']]); e = t['shaft']
    tipP = None
    if t['kind'] == 'teal':
        if t.get('pink_xs') is not None and len(t['pink_xs']) >= 2:
            tipP = proj.pts(t['pink_xs'], t['pink_ys'], max_tool=40)
    else:
        # wood: tan pixels of the connected structure
        comp = seg.components(seg.dilate(T, 1), min_area=10)
        for cc in comp:
            hit = np.isin(cc['xs'] * 1000 + cc['ys'], t['handle_xs'] * 1000 + t['handle_ys']).any()
            if hit:
                tan = ms['tan'][cc['ys'], cc['xs']] & T[cc['ys'], cc['xs']]
                if tan.sum() >= 3:
                    tipP = proj.pts(cc['xs'][tan], cc['ys'][tan])
                break
    if tipP is not None and len(tipP) >= 2:
        d = np.linalg.norm(tipP[:, :2] - g[:2], axis=1)
        k = max(1, int(0.1 * len(d)))
        far = tipP[np.argsort(-d)[:k]]
        tip = far.mean(0)
        t['tip'] = tip
        t['tipP'] = tipP
        # metric typing for teal tools
        if t['kind'] == 'teal':
            # metric typing relative to the cube size (cube ~ unit; ball tip ~1.2 unit; hook arm ~3-4 unit;
            # spatula pad ~2x2 unit)
            u_ = unit if unit is not None else 0.04
            tp = tipP[np.linalg.norm(tipP - np.median(tipP, 0), axis=1) < 0.25]
            Qp = tp[:, :2] - tp[:, :2].mean(0)
            if len(Qp) >= 4:
                ww, VV = np.linalg.eigh(np.cov(Qp.T))
                a1 = Qp @ VV[:, 1]; a2 = Qp @ VV[:, 0]
                ext1 = float(np.percentile(a1, 95) - np.percentile(a1, 5))
                ext2 = float(np.percentile(a2, 95) - np.percentile(a2, 5))
                t['tip_ext'] = (ext1, ext2)
                t['type_px'] = t['type']
                if ext1 < 1.6 * u_:
                    t['type_m'] = 'ball'
                elif ext2 > 1.2 * u_ and ext2 > 0.55 * ext1:
                    t['type_m'] = 'spatula'
                else:
                    t['type_m'] = 'hook'
                # pixel spatula detection is reliable; otherwise trust the metric type when enough points
                if t['type'] != 'spatula' and len(tp) >= 6:
                    t['type'] = t['type_m'] if t['type_m'] != 'spatula' else t['type']
    else:
        # fall back: along shaft by pixel length (approximate scale from handle)
        t['tip'] = None
    if t.get('tip') is not None:
        o = t['tip'][:2] - t['grasp'][:2]
        t['L'] = float(np.linalg.norm(o))


def ramp_plane(P):
    """RANSAC plane for the sloped top surface. Returns (n, c, inliers) with n_z>0, or None."""
    rng = np.random.RandomState(0)
    best = None
    n_pts = len(P)
    for _ in range(300):
        i = rng.choice(n_pts, 3, replace=False)
        a, b, c = P[i]
        n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        if n[2] < 0:
            n = -n
        if not (0.35 < n[2] < 0.985):
            continue
        d = np.abs((P - a) @ n)
        inl = d < 0.006
        s = inl.sum()
        if best is None or s > best[0]:
            best = (s, n, a, inl)
    if best is None or best[0] < 10:
        return None
    s, n, a, inl = best
    Q = P[inl]; cen = Q.mean(0)
    U, S, Vt = np.linalg.svd(Q - cen)
    n2 = Vt[2]
    if n2[2] < 0:
        n2 = -n2
    return n2, cen, inl
