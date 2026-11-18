import json, numpy as np
from geom import deproject, load_rgb, load_depth

ZLO, ZHI = 0.045, 0.079   # container upper part heights

def frame_files(f):
    return {(x['camera'], x['kind']): x['path'] for x in f['files']}

def world_front(f, root='/workspace/'):
    files = frame_files(f)
    d = load_depth(root + files[('front', 'depth')], f['depth']['front'].get('meters_per_unit', 0.001))
    rgb = load_rgb(root + files[('front', 'rgb')])
    W = deproject(d, f['camera_intrinsics']['front']['matrix'], f['camera_poses']['front'])
    return W, rgb

def container_points(W, rgb):
    z = W[..., 2]
    rgbi = rgb.astype(int)
    mn = rgbi.min(-1); mx = rgbi.max(-1)
    whiteish = (mn > 90) & (mx - mn < 45)
    sel = (z > ZLO) & (z < ZHI) & whiteish & np.isfinite(z)
    return W[sel][:, :2]

def hue_class(rgb):
    """rgb Nx3 int -> labels list for saturated colors."""
    import colorsys
    out = []
    for r, g, b in rgb:
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        if s < 0.45 or v < 0.2:
            out.append(None); continue
        hd = h * 360
        if hd < 15 or hd >= 340: out.append('red')
        elif hd < 51: out.append('orange')
        elif hd < 75: out.append('yellow')
        elif hd < 170: out.append('green')
        elif hd < 260: out.append('blue')
        else: out.append('purple')
    return out

def cube_detect(W, rgb, cell=0.004):
    """Detect colored cubes on table: returns list of (color, xy, npts)."""
    from scipy import ndimage
    z = W[..., 2]
    sel = (z > 0.003) & (z < 0.06) & np.isfinite(z)
    P = W[sel]; C = rgb[sel].astype(int)
    labs = hue_class(C)
    res = []
    for col in ['red', 'orange', 'yellow', 'green', 'blue', 'purple']:
        idx = [i for i, l in enumerate(labs) if l == col]
        if len(idx) < 5: continue
        Q = P[idx][:, :2]
        x0, y0 = -0.8, -0.6
        gi = ((Q[:, 0] - x0) / cell).astype(int); gj = ((Q[:, 1] - y0) / cell).astype(int)
        grid = np.zeros((int(1.4 / cell) + 2, int(1.2 / cell) + 2), bool)
        ok = (gi >= 0) & (gi < grid.shape[0]) & (gj >= 0) & (gj < grid.shape[1])
        grid[gi[ok], gj[ok]] = True
        grid = ndimage.binary_dilation(grid, iterations=2)
        lab, n = ndimage.label(grid)
        pl = lab[gi[ok], gj[ok]]
        Qo = Q[ok]
        for k in range(1, n + 1):
            m = pl == k
            if m.sum() >= 5:
                res.append((col, Qo[m].mean(0), int(m.sum())))
    return res

def kmeans_track(P, seeds, iters=6, radius=0.045):
    C = seeds.copy()
    for _ in range(iters):
        d = np.linalg.norm(P[:, None, :] - C[None], axis=-1)
        a = d.argmin(1); dm = d.min(1)
        newC = C.copy()
        for k in range(len(C)):
            m = (a == k) & (dm < radius)
            if m.sum() >= 8:
                newC[k] = P[m].mean(0)
        C = newC
    counts = np.array([((a == k) & (dm < radius)).sum() for k in range(len(C))])
    return C, counts

def components_xy(P, cell=0.004, min_pts=40):
    from scipy import ndimage
    if len(P) == 0: return []
    x0, y0 = P.min(0) - 0.02
    gi = ((P[:, 0] - x0) / cell).astype(int); gj = ((P[:, 1] - y0) / cell).astype(int)
    g = np.zeros((gi.max() + 3, gj.max() + 3), bool); g[gi, gj] = True
    g = ndimage.binary_closing(g, iterations=1)
    lab, n = ndimage.label(g)
    pl = lab[gi, gj]
    out = []
    for k in range(1, n + 1):
        m = pl == k
        if m.sum() >= min_pts: out.append((P[m].mean(0), int(m.sum())))
    return out

def merge_cubes(dets, r=0.03):
    """dets: list of (color, xy, n) from several frames -> merged list (color, xy, votes)."""
    groups = []
    for c, p, n in dets:
        for g in groups:
            if np.linalg.norm(g['xy'] - p) < r:
                g['pts'].append(p); g['cols'][c] = g['cols'].get(c, 0) + n
                g['xy'] = np.mean(g['pts'], 0); break
        else:
            groups.append({'xy': p.copy(), 'pts': [p], 'cols': {c: n}})
    return [(max(g['cols'], key=g['cols'].get), g['xy'], len(g['pts'])) for g in groups]

def world_cam(f, cam, root='/workspace/'):
    files = frame_files(f)
    d = load_depth(root + files[(cam, 'depth')], f['depth'][cam].get('meters_per_unit', 0.001))
    rgb = load_rgb(root + files[(cam, 'rgb')])
    W = deproject(d, f['camera_intrinsics'][cam]['matrix'], f['camera_poses'][cam])
    return W, rgb

def wrist_top_points(W, rgb):
    z = W[..., 2]
    rgbi = rgb.astype(int); mn = rgbi.min(-1); mx = rgbi.max(-1)
    sel = (z > 0.064) & (z < 0.079) & (mn > 90) & (mx - mn < 45) & np.isfinite(z)
    sel[:3, :] = False; sel[-3:, :] = False; sel[:, :3] = False; sel[:, -3:] = False
    return W[sel][:, :2]

def fused_points(views):
    """views: list of (W, rgb) (front first, then wrist) -> container xy points."""
    Ps = [container_points(W, rgb) if k == 0 else wrist_top_points(W, rgb) for k, (W, rgb) in enumerate(views)]
    Ps = [P for P in Ps if len(P)]
    if not Ps: return np.zeros((0, 2))
    P = np.concatenate(Ps)
    ok = (P[:, 0] > -0.6) & (P[:, 0] < 0.35) & (np.abs(P[:, 1]) < 0.5)
    return P[ok]

class Tracker:
    def __init__(self, seeds, radius=0.04, min_cnt=15, log=print):
        self.C = np.array(seeds, float); self.v = np.zeros_like(self.C)
        self.lost = np.zeros(len(self.C), int); self.radius = radius; self.min_cnt = min_cnt
        self.log = log; self.recoveries = 0; self.counts = np.zeros(len(self.C), int)

    def update(self, P, dt=1.0):
        C0 = self.C.copy()
        pred = self.C + self.v * dt * (self.lost == 0)[:, None]
        if len(P) == 0:
            self.lost += 1; self.v *= 0.5; return
        Cn = pred.copy()
        for _ in range(6):
            d = np.linalg.norm(P[:, None, :] - Cn[None], axis=-1)
            a = d.argmin(1); dm = d.min(1)
            for k in range(len(Cn)):
                m = (a == k) & (dm < self.radius)
                if m.sum() >= 8: Cn[k] = P[m].mean(0)
        counts = np.array([((a == k) & (dm < self.radius)).sum() for k in range(len(Cn))])
        good = counts >= self.min_cnt
        # tracks that are poorly observed: hold last position, kill velocity
        Cn[~good] = np.where(self.lost[~good, None] < 2, pred[~good], self.C[~good])
        self.v[good] = 0.5 * self.v[good] + 0.5 * (Cn[good] - C0[good]) / dt
        self.v[~good] *= 0.3
        self.lost[good] = 0; self.lost[~good] += 1
        self.C = Cn; self.counts = counts
        # recovery: unexplained components -> lost tracks
        lost_idx = [k for k in range(len(Cn)) if self.lost[k] >= 2]
        if lost_idx:
            dmin = np.linalg.norm(P[:, None, :] - Cn[None], axis=-1).min(1)
            U = P[dmin > 0.045]
            comps = [c for c in components_xy(U, min_pts=60)]
            if comps:
                from scipy.optimize import linear_sum_assignment
                D = np.array([[np.linalg.norm(c - Cn[k]) for k in lost_idx] for c, _ in comps])
                r, cidx = linear_sum_assignment(D)
                for ri, ci in zip(r, cidx):
                    if D[ri, ci] < 0.3:
                        k = lost_idx[ci]
                        self.log(f'    recover track {k}: ({Cn[k][0]:.3f},{Cn[k][1]:.3f}) -> ({comps[ri][0][0]:.3f},{comps[ri][0][1]:.3f}) after {self.lost[k]} lost frames')
                        self.C[k] = comps[ri][0]; self.v[k] = 0; self.lost[k] = 0; self.recoveries += 1

def frame_views(f, root='/workspace/'):
    views = []
    for cam in ('front', 'wrist'):
        try: views.append(world_cam(f, cam, root))
        except Exception: pass
    return views

def track_video(manifest, root='/workspace/', log=print):
    frames = manifest['frames']
    vid = [f for f in frames if f['phase'] == 'video_demo']
    vid.sort(key=lambda f: f['frame_index'])
    exe = [f for f in frames if f['phase'] == 'execution']
    first_cont = None; pre = []
    for f in vid:
        views = frame_views(f, root)
        P = container_points(*views[0])
        if len(P) > 150:
            first_cont = (f, P); break
        c = cube_detect(*views[0])
        if len(views) > 1:
            for d in cube_detect(*views[1]):
                if all(np.linalg.norm(d[1] - e[1]) > 0.04 for e in c): c.append(d)
        if c: pre.append((f['frame_index'], c))
    if first_cont is None or not pre:
        raise RuntimeError('no cube/container transition found')
    dets = [d for _, cs in pre[-12:] for d in cs]
    nlast = len(pre[-12:])
    cubes = [c for c in merge_cubes(dets) if c[2] >= max(1, nlast // 3)]
    log(f'cubes: ' + ', '.join(f'{c}@({p[0]:.3f},{p[1]:.3f}) v={n}' for c, p, n in cubes))
    f0, P0 = first_cont
    comps = components_xy(P0, min_pts=60)
    seeds = [p for _, p, _ in cubes]; colors = [c for c, _, _ in cubes]
    for xy, n in comps:
        if all(np.linalg.norm(xy - p) > 0.045 for p in seeds):
            seeds.append(xy); colors.append('none')
    seeds = np.array(seeds)
    log(f'container seeds at frame {f0["frame_index"]}: ' + ', '.join(f'{c}@({p[0]:.3f},{p[1]:.3f})' for c, p in zip(colors, seeds)))
    seq = [f for f in vid if f['frame_index'] >= f0['frame_index']] + exe
    tr = Tracker(seeds, log=log); hist = []
    prev_src = None
    for f in seq:
        src = f.get('source_frame_index', f['frame_index'])
        P = fused_points(frame_views(f, root))
        dt = 1.0 if prev_src is None or f['phase'] == 'execution' else max(1.0, src - prev_src)
        tr.update(P, dt)
        prev_src = src
        hist.append((f['frame_index'], tr.C.copy(), tr.counts.copy()))
    return colors, seeds, hist, tr

def box_yaw_width(P):
    """P Nx2 top-face points -> (yaw in [0,90) deg rad, width along best axes)."""
    best = None
    for a in np.deg2rad(np.arange(0, 90, 1.0)):
        c, s = np.cos(a), np.sin(a)
        Q = P @ np.array([[c, -s], [s, c]])
        lo = np.percentile(Q, 2, axis=0); hi = np.percentile(Q, 98, axis=0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            best = (area, a, hi - lo)
    return best[1], best[2]

def container_tops(W, rgb, centers, r=0.04):
    z = W[..., 2]
    sel = (z > 0.066) & (z < 0.079) & np.isfinite(z)
    P = W[sel][:, :2]
    out = []
    for c in centers:
        m = np.linalg.norm(P - c, axis=1) < r
        Q = P[m]
        if len(Q) < 15:
            out.append((c, None, None, len(Q))); continue
        yaw, wh = box_yaw_width(Q)
        out.append((Q.mean(0), yaw, wh, len(Q)))
    return out
