"""Container tracking through the conditioning video using front depth -> world top-down points."""
import json, numpy as np
from geom import backproject, load_depth, load_rgb

ZLO, ZHI = 0.008, 0.13
XLIM = (-0.45, 0.35); YLIM = (-0.40, 0.40)

def frame_points(root, f, cam='front', with_rgb=False):
    files = {(x['camera'], x['kind']): x['path'] for x in f['files']}
    d = load_depth(f"{root}/{files[(cam,'depth')]}")
    pw, ok = backproject(d, f['camera_intrinsics'][cam], f['camera_poses'][cam])
    sel = ok & (pw[...,2] > ZLO) & (pw[...,2] < ZHI) & (pw[...,0] > XLIM[0]) & (pw[...,0] < XLIM[1]) & (pw[...,1] > YLIM[0]) & (pw[...,1] < YLIM[1])
    if with_rgb:
        rgb = load_rgb(f"{root}/{files[(cam,'rgb')]}")
        return pw[sel], rgb[sel], sel
    return pw[sel]

SAT_MAX = 40

def white_filter(P, C):
    c = C.astype(int)
    sat = c.max(1) - c.min(1)
    return P[(sat < SAT_MAX) & (c.max(1) > 60)]

def scene_points(root, f, cams=('front', 'wrist')):
    """world points of low-saturation (container-coloured) surfaces from all cameras"""
    out = []
    for cam in cams:
        try:
            P, C, _ = frame_points(root, f, cam=cam, with_rgb=True)
            out.append(white_filter(P, C))
        except Exception:
            pass
    return np.concatenate(out) if out else np.zeros((0, 3))

def components(P, res=0.004, min_pts=15):
    """grid connected components of XY points -> list of (centroid_xy, n, zmax)"""
    if len(P) == 0: return []
    from collections import deque
    g = np.floor(P[:, :2] / res).astype(int)
    cells = {}
    for i, k in enumerate(map(tuple, g)):
        cells.setdefault(k, []).append(i)
    seen = set(); out = []
    for k in cells:
        if k in seen: continue
        q = deque([k]); seen.add(k); idx = []
        while q:
            c = q.popleft(); idx += cells[c]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (c[0]+dx, c[1]+dy)
                    if n in cells and n not in seen:
                        seen.add(n); q.append(n)
        if len(idx) >= min_pts:
            Q = P[idx]
            out.append((Q[:, :2].mean(0), len(idx), Q[:, 2].max(), np.array(idx)))
    return out

COLORS = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'white', 'black']

def classify_rgb(c):
    r, g, b = c[:, 0].astype(int), c[:, 1].astype(int), c[:, 2].astype(int)
    lab = np.full(len(c), -1)
    mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
    sat = (mx - mn) > 60
    lab[sat & (r > g + 60) & (r > b + 60) & (g < 0.45 * r)] = 0   # red
    lab[sat & (g > r + 30) & (g > b + 30)] = 1                   # green
    lab[sat & (b > r + 30) & (b > g + 10)] = 2                   # blue
    lab[sat & (r > 120) & (g > 0.75 * r) & (b < 0.5 * g)] = 3    # yellow
    lab[sat & (r > 120) & (g >= 0.45 * r) & (g <= 0.75 * r) & (b < 0.4 * r)] = 4  # orange
    lab[sat & (r > g + 40) & (b > g + 40)] = 5                   # purple
    lab[((mx - mn) < 30) & (mn > 150)] = 6                       # white
    lab[(mx < 50)] = 7                                           # black
    return lab

def detect_cubes(root, f):
    P, C, _ = frame_points(root, f, with_rgb=True)
    try:
        P2, C2, _ = frame_points(root, f, cam='wrist', with_rgb=True)
        P = np.concatenate([P, P2]); C = np.concatenate([C, C2])
    except Exception:
        pass
    Pall = P
    keep = P[:, 2] < 0.06
    P, C = P[keep], C[keep]
    lab = classify_rgb(C)
    cubes = []
    for k in range(len(COLORS)):
        Q = P[lab == k]
        if len(Q) < 5: continue
        for cen, n, z, idx in components(Q, res=0.006, min_pts=5):
            cubes.append({'color': COLORS[k], 'xy': cen.tolist(), 'n': int(n)})
    # reject tall structures (occluders) masquerading as cubes
    good = []
    for c in cubes:
        near = Pall[np.linalg.norm(Pall[:, :2] - np.array(c['xy']), axis=1) < 0.025]
        if len(near) and near[:, 2].max() > 0.055:
            continue
        if c['n'] < 8:
            continue
        good.append(c)
    cubes = good
    # merge same-colour fragments (occlusion splits a cube)
    merged = []
    for c in sorted(cubes, key=lambda c: -c['n']):
        for m in merged:
            if m['color'] == c['color'] and np.linalg.norm(np.array(m['xy']) - c['xy']) < 0.04:
                w = m['n'] + c['n']
                m['xy'] = ((np.array(m['xy']) * m['n'] + np.array(c['xy']) * c['n']) / w).tolist(); m['n'] = w
                break
        else:
            merged.append(dict(c))
    return merged

def kmeans_track(P, centers, iters=6, radius=0.05, return_mask=False):
    C = centers.copy()
    upd = np.zeros(len(C), dtype=bool)
    if len(P) == 0:
        return (C, upd) if return_mask else C
    for _ in range(iters):
        d = np.linalg.norm(P[:, None, :2] - C[None], axis=2)
        a = d.argmin(1); dm = d.min(1)
        for k in range(len(C)):
            m = (a == k) & (dm < radius)
            if m.sum() >= 8:
                C[k] = P[m, :2].mean(0); upd[k] = True
    return (C, upd) if return_mask else C

def track_video(root, manifest, log=None):
    fr = manifest['frames']
    vid = [f for f in fr if f['phase'] == 'video_demo']
    # find covering: first frame where container-top points (z>0.055) are plentiful
    zs = []
    first_cov = None; last_uncov = None
    for i, f in enumerate(vid):
        P = frame_points(root, f)
        ntop = int((P[:, 2] > 0.055).sum()) if len(P) else 0
        if ntop > 300:
            first_cov = i; break
        last_uncov = i
    if first_cov is None or last_uncov is None:
        raise RuntimeError('no covering event found')
    cubes = detect_cubes(root, vid[last_uncov])
    P = frame_points(root, vid[first_cov]); T = P[P[:, 2] > 0.055]
    comps = components(T, res=0.004, min_pts=20)
    cen = np.array([c[0] for c in comps]) if comps else np.zeros((0, 2))
    # initialize container centers from cube positions refined by kmeans on tops
    C0 = np.array([c['xy'] for c in cubes])
    C = kmeans_track(T, C0)
    hist = [C.copy()]
    prev = C.copy(); vel = np.zeros_like(C)
    for f in vid[first_cov + 1:] + [fr[-1]]:
        P = frame_points(root, f); T = P[P[:, 2] > 0.055]
        pred = prev + vel
        Cn = kmeans_track(T, pred)
        vel = 0.5 * vel + 0.5 * (Cn - prev)
        prev = Cn; hist.append(Cn.copy())
    return {'cubes': cubes, 'first_cov': first_cov, 'last_uncov': last_uncov,
            'final': prev.tolist(), 'hist': np.array(hist), 'n_comp_cov': len(comps)}

def container_geometry(P, centers, radius=0.05):
    """For each center: refined xy, yaw (mod 90deg) and half-size from top points."""
    T = top_points(P)
    out = []
    if len(T) == 0:
        return [{'xy': c.tolist(), 'yaw': 0.0, 'half': None, 'n': 0, 'ztop': None} for c in centers]
    d = np.linalg.norm(T[:, None, :2] - centers[None], axis=2); a = d.argmin(1); dm = d.min(1)
    for k in range(len(centers)):
        Q = T[(a == k) & (dm < radius), :2]
        if len(Q) < 10:
            out.append({'xy': centers[k].tolist(), 'yaw': 0.0, 'half': None, 'n': int(len(Q)), 'ztop': None}); continue
        best = None
        c = Q.mean(0)
        for yaw in np.radians(np.arange(0, 90, 2)):
            R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
            q = (Q - c) @ R.T
            ext = np.percentile(q, 98, axis=0) - np.percentile(q, 2, axis=0)
            area = ext[0] * ext[1]
            if best is None or area < best[0]:
                lo = np.percentile(q, 2, axis=0); hi = np.percentile(q, 98, axis=0)
                cc = c + ((lo + hi) / 2) @ R
                best = (area, yaw, ext, cc)
        zt = float(np.percentile(T[(a == k) & (dm < radius), 2], 90))
        out.append({'xy': best[3].tolist(), 'yaw': float(best[1]), 'ext': best[2].tolist(), 'n': int(len(Q)), 'ztop': zt})
    return out

STATIC = set()      # 1cm xy cells occupied by static tall structures (occluders)
CELL = 0.01

def build_static(P):
    """mark xy cells of tall static structures seen before any container exists"""
    STATIC.clear()
    Q = P[(P[:, 2] > 0.045) & (P[:, 2] < 0.15)]
    if len(Q) == 0: return 0
    g = np.floor(Q[:, :2] / CELL).astype(int)
    cells, cnt = np.unique(g, axis=0, return_counts=True)
    for (a, b), n in zip(cells, cnt):
        if n >= 2:
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    STATIC.add((a + da, b + db))
    return len(STATIC)

def not_static(P):
    if not STATIC or len(P) == 0: return P
    g = np.floor(P[:, :2] / CELL).astype(int)
    keep = np.array([(a, b) not in STATIC for a, b in g], dtype=bool)
    return P[keep]

def top_points(P):
    return not_static(P[(P[:, 2] > 0.055) & (P[:, 2] < 0.09)])

class Tracker:
    def __init__(self, centers, labels):
        self.C = np.array(centers, float); self.labels = list(labels)
        self.V = np.zeros_like(self.C); self.n = 0
    def update(self, P, dt=1.0):
        T = top_points(P)
        pred = self.C + self.V * dt
        Cn, upd = kmeans_track(T, pred, return_mask=True)
        # velocity per unit dt; decay when a container is unobserved (occluded)
        Vn = 0.5 * self.V + 0.5 * (Cn - self.C) / max(dt, 1e-6)
        Vn[~upd] = 0.6 * self.V[~upd]
        Cn[~upd] = self.C[~upd] + Vn[~upd] * dt
        self.V = Vn
        mv = float(np.linalg.norm(Cn - self.C, axis=1).max())
        self.C = Cn; self.n += 1
        return mv

def track_video2(root, manifest):
    """returns Tracker after processing all video frames + first exec frame"""
    fr = manifest['frames']
    vid = [f for f in fr if f['phase'] == 'video_demo']
    first_cov = None; last_uncov = None
    P0 = np.concatenate([frame_points(root, vid[0], cam=c) for c in ('front', 'wrist')])
    ns = build_static(P0)
    for i, f in enumerate(vid):
        P = scene_points(root, f)
        if len(top_points(P)) > 300:
            first_cov = i; break
        last_uncov = i
    if first_cov == 0:
        # video may start already covered: no static estimate possible
        STATIC.clear(); last_uncov = None
    if first_cov is None or last_uncov is None:
        raise RuntimeError('no covering event found')
    cubes = detect_cubes(root, vid[last_uncov])
    T0 = top_points(scene_points(root, vid[first_cov]))
    comps = [c for c in components(T0, res=0.004, min_pts=40)]
    centers = [c['xy'] for c in cubes]; labels = [c['color'] for c in cubes]
    for cen, n, z, idx in comps:   # containers with no detected cube underneath
        if not centers or min(np.linalg.norm(np.array(centers) - cen, axis=1)) > 0.05:
            centers.append(cen.tolist()); labels.append('unknown')
    tr = Tracker(centers, labels)
    tr.C = kmeans_track(T0, tr.C)
    prev_t = vid[first_cov]['source_frame_index']
    for f in vid[first_cov + 1:] + [fr[-1]]:
        t = f['source_frame_index'] if f['phase'] == 'video_demo' else vid[-1]['source_frame_index'] + 1
        tr.update(scene_points(root, f), dt=max(1, t - prev_t)); prev_t = t
    return tr, cubes
