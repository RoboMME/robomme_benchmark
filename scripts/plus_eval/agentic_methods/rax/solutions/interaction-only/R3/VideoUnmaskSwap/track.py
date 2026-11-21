"""Track the containers through the conditioning video in world XY (front camera RGB-D)."""
import json
import numpy as np
from pathlib import Path
from PIL import Image
from geom import backproject

X_MIN, X_MAX, Y_ABS = -0.45, 0.30, 0.40   # workspace box (m)
CONT_Z = (0.01, 0.085)                    # container pixels (top 0.074); excludes robot hand above
TOP_Z = 0.055                             # container top face ~0.074
CUBE_Z = (0.003, 0.05)

def load_frame(root, frame):
    files = {(x['camera'], x['kind']): x for x in frame['files']}
    rgb = np.array(Image.open(root / files[('front', 'rgb')]['path']).convert('RGB')).astype(np.float32)
    dfile = files[('front', 'depth')]
    mpu = dfile.get('meters_per_unit', 0.001) or 0.001
    d = np.array(Image.open(root / dfile['path'])).astype(np.float32) * mpu
    P = backproject(d, frame['camera_intrinsics']['front']['matrix'], frame['camera_poses']['front'])
    return rgb, P

def in_box(P):
    x, y = P[..., 0], P[..., 1]
    with np.errstate(invalid='ignore'):
        return (x > X_MIN) & (x < X_MAX) & (np.abs(y) < Y_ABS)

def white_mask(rgb, P):
    mx = rgb.max(-1); mn = rgb.min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    z = P[..., 2]
    with np.errstate(invalid='ignore'):
        return in_box(P) & (z > CONT_Z[0]) & (z < CONT_Z[1]) & (sat < 0.25) & (mx > 100)

def classify_color(rgb):
    """Per-pixel colour class for saturated pixels: returns HxW int (-1 none)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1); mn = rgb.min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    out = -np.ones(r.shape, int)
    s = (sat > 0.5) & (mx > 60)
    # hue
    import colorsys
    h = np.zeros_like(r)
    rr, gg, bb = r / 255, g / 255, b / 255
    mxn, mnn = mx / 255, mn / 255
    dl = mxn - mnn + 1e-6
    h = np.where(mxn == rr, ((gg - bb) / dl) % 6, np.where(mxn == gg, (bb - rr) / dl + 2, (rr - gg) / dl + 4)) * 60
    out[s & ((h < 15) | (h > 330))] = 0          # red
    out[s & (h >= 15) & (h < 45)] = 1            # orange
    out[s & (h >= 45) & (h < 75)] = 2            # yellow
    out[s & (h >= 75) & (h < 170)] = 3           # green
    out[s & (h >= 190) & (h < 270)] = 4          # blue
    out[s & (h >= 270) & (h <= 330)] = 5         # purple/magenta
    return out

COLOR_NAMES = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']

def cluster_xy(pts, link=0.02):
    """Greedy single-link clustering of Nx2 points; returns list of index arrays."""
    n = len(pts)
    lab = -np.ones(n, int)
    cur = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack = [i]; lab[i] = cur
        while stack:
            j = stack.pop()
            d = np.sqrt(((pts - pts[j]) ** 2).sum(-1))
            nb = np.where((d < link) & (lab < 0))[0]
            lab[nb] = cur
            stack.extend(nb.tolist())
        cur += 1
    return [np.where(lab == c)[0] for c in range(cur)]

def cube_blobs(rgb, P, min_px=4):
    cls = classify_color(rgb)
    z = P[..., 2]
    with np.errstate(invalid='ignore'):
        ok = in_box(P) & (z > CUBE_Z[0]) & (z < CUBE_Z[1])
    cubes = []
    for c in range(len(COLOR_NAMES)):
        pts = P[(cls == c) & ok][:, :2]
        if len(pts) == 0:
            continue
        for idx in cluster_xy(pts):
            if len(idx) < min_px:
                continue
            cubes.append({'color': COLOR_NAMES[c], 'xy': pts[idx].mean(0), 'px': int(len(idx))})
    return cubes

def assign_min_cost(D):
    """Exact min-cost assignment rows->cols (rows<=cols, small sizes) by DP over subsets."""
    import itertools
    nr, nc = D.shape
    best = (np.inf, None)
    if nr <= 8 and nc <= 8:
        for perm in itertools.permutations(range(nc), nr):
            c = sum(D[i, perm[i]] for i in range(nr))
            if c < best[0]:
                best = (c, perm)
        return list(range(nr)), list(best[1])
    # greedy fallback
    rows, cols = [], []
    Dm = D.copy()
    for _ in range(min(nr, nc)):
        i, j = np.unravel_index(np.argmin(Dm), Dm.shape)
        rows.append(i); cols.append(j); Dm[i, :] = np.inf; Dm[:, j] = np.inf
    return rows, cols

def white_points(rgb, P, top_only=False):
    m = white_mask(rgb, P)
    if top_only:
        with np.errstate(invalid='ignore'):
            m = m & (P[..., 2] > TOP_Z)
    return P[m]

def kmeans_init(pts, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for trial in range(8):
        C = pts[rng.choice(len(pts), k, replace=False)]
        for _ in range(iters):
            d = ((pts[:, None] - C[None]) ** 2).sum(-1)
            a = d.argmin(1)
            C = np.array([pts[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
        sse = ((pts - C[a]) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, C)
    return best[1]

def track_step(pts, C, V, radius=0.035, iters=4):
    """pts Nx2, C kx2 previous centres, V kx2 velocity. Returns new C, counts."""
    pred = C + V
    newC = pred.copy()
    counts = np.zeros(len(C), int)
    for _ in range(iters):
        d = np.sqrt(((pts[:, None] - newC[None]) ** 2).sum(-1))
        a = d.argmin(1)
        dm = d[np.arange(len(pts)), a]
        for j in range(len(C)):
            sel = (a == j) & (dm < radius)
            counts[j] = sel.sum()
            if counts[j] >= 3:
                newC[j] = pts[sel].mean(0)
    return newC, counts

def run_tracking(root, manifest, log=print):
    frames = sorted(manifest['frames'], key=lambda f: f['frame_index'])
    vids = [f for f in frames if f.get('phase') == 'video_demo']
    execf = [f for f in frames if f.get('phase') != 'video_demo']
    seq = vids + execf[:1]
    # 1) find first frame where containers exist (white low-z mass jumps) and last frame with cubes
    n_white = []
    cache = {}
    for i, f in enumerate(seq):
        rgb, P = load_frame(root, f)
        cache[i] = (rgb, P)
        n_white.append(int(white_mask(rgb, P).sum()))
    n_white = np.array(n_white)
    big = n_white > 0.5 * np.median(n_white[-20:])
    first_c = int(np.argmax(big))
    log(f'track: {len(seq)} frames, containers first at {first_c}, white px median {np.median(n_white[-20:]):.0f}')
    # 2) cubes from the last frames before containers
    cubes = []
    for j in range(first_c - 1, max(-1, first_c - 6), -1):
        cubes = cube_blobs(*cache[j])
        if cubes:
            log(f'track: cubes from frame {j}: ' + ', '.join(f"{c['color']}@({c['xy'][0]:.3f},{c['xy'][1]:.3f})/{c['px']}" for c in cubes))
            break
    k = len(cubes)
    # 3) initial container centres
    pts = white_points(*cache[first_c], top_only=True)[:, :2]
    if len(pts) < 50:
        pts = white_points(*cache[first_c])[:, :2]
    C = kmeans_init(pts, k)
    cube_xy = np.array([c['xy'] for c in cubes])
    D = np.sqrt(((C[:, None] - cube_xy[None]) ** 2).sum(-1))
    r, c = assign_min_cost(D)
    labels = [None] * k
    for a, b in zip(r, c):
        labels[a] = cubes[b]['color']
        log(f'track: container {a} at ({C[a][0]:.3f},{C[a][1]:.3f}) covers {cubes[b]["color"]} dist {D[a,b]:.3f}')
    V = np.zeros_like(C)
    hist = [C.copy()]
    minc = []
    for i in range(first_c + 1, len(seq)):
        rgb, P = cache[i]
        pts = white_points(rgb, P, top_only=True)[:, :2]
        C2, counts = track_step(pts, C, V)
        V = 0.5 * V + 0.5 * (C2 - C)
        C = C2
        hist.append(C.copy())
        minc.append(counts.min())
    hist = np.array(hist)
    return {'labels': labels, 'final': C, 'hist': hist, 'first_c': first_c, 'cubes': cubes,
            'min_counts': np.array(minc), 'seq_len': len(seq)}


# ---------------- v2 tracker -----------------
MIN_SEP = 0.042     # container top is ~0.045-0.05 m square; centres cannot be closer

def track_step2(pts, C, V, radius=0.04, iters=5, min_sep=MIN_SEP):
    pred = C + V
    newC = pred.copy()
    counts = np.zeros(len(C), int)
    for _ in range(iters):
        d = np.sqrt(((pts[:, None] - newC[None]) ** 2).sum(-1))
        a = d.argmin(1)
        dm = d[np.arange(len(pts)), a]
        for j in range(len(C)):
            sel = (a == j) & (dm < radius)
            counts[j] = sel.sum()
            if counts[j] >= 5:
                newC[j] = 0.7 * pts[sel].mean(0) + 0.3 * np.median(pts[sel], 0)
            else:
                newC[j] = C[j] + 0.5 * V[j]
        # separation constraint
        for _r in range(3):
            for p in range(len(C)):
                for q in range(p + 1, len(C)):
                    dv = newC[q] - newC[p]
                    dist = np.sqrt((dv ** 2).sum())
                    if dist < min_sep:
                        push = (min_sep - dist) / 2 * (dv / (dist + 1e-9))
                        wp = counts[q] + 1.0; wq = counts[p] + 1.0  # fewer points -> moves more
                        newC[p] -= push * 2 * wp / (wp + wq)
                        newC[q] += push * 2 * wq / (wp + wq)
    return newC, counts

def track_sequence(point_sets, C0, step=track_step2, **kw):
    C = C0.copy(); V = np.zeros_like(C)
    hist = [C.copy()]; counts_h = []
    for pts in point_sets:
        C2, counts = step(pts, C, V, **kw)
        V = 0.5 * V + 0.5 * (C2 - C)
        C = C2
        hist.append(C.copy()); counts_h.append(counts)
    return np.array(hist), np.array(counts_h)

def prepare(root, manifest, log=print, top_only=False):
    frames = sorted(manifest['frames'], key=lambda f: f['frame_index'])
    vids = [f for f in frames if f.get('phase') == 'video_demo']
    execf = [f for f in frames if f.get('phase') != 'video_demo']
    seq = vids + execf[:1]
    data = [load_frame(root, f) for f in seq]
    n_white = np.array([int(white_mask(rgb, P).sum()) for rgb, P in data])
    big = n_white > 0.5 * np.median(n_white[-20:])
    first_c = int(np.argmax(big))
    cubes = []
    for j in range(first_c - 1, max(-1, first_c - 8), -1):
        cubes = cube_blobs(*data[j])
        if cubes:
            break
    pts = [white_points(rgb, P, top_only=top_only)[:, :2] for rgb, P in data[first_c:]]
    pts = [p[::2] if len(p) > 3000 else p for p in pts]
    return {'seq': seq, 'first_c': first_c, 'cubes': cubes, 'pts': pts, 'cube_frame': j}

def init_centres(pts, cubes):
    k = len(cubes)
    C = kmeans_init(pts, k)
    cube_xy = np.array([c['xy'] for c in cubes])
    D = np.sqrt(((C[:, None] - cube_xy[None]) ** 2).sum(-1))
    r, c = assign_min_cost(D)
    order = [None] * k
    for a, b in zip(r, c):
        order[b] = a
    # centre j corresponds to cube j
    return C[order], D[r, c].max()


# ---------------- v3: time-aware tracker usable live ----------------
class Tracker:
    def __init__(self, C0, labels, t0):
        self.C = np.asarray(C0, float).copy()
        self.V = np.zeros_like(self.C)     # m per time unit (source frame / control step)
        self.labels = list(labels)
        self.t = t0
        self.frozen = np.zeros(len(self.C), bool)   # e.g. container in gripper / removed
        self.last_counts = None

    def update(self, pts, t, exclude_xy=None, exclude_r=0.0):
        dt = max(1e-3, t - self.t)
        pred = self.C + self.V * dt
        sub = ~self.frozen
        if exclude_xy is not None and len(pts):
            keep = np.sqrt(((pts - np.asarray(exclude_xy)) ** 2).sum(-1)) > exclude_r
            pts = pts[keep]
        Vd = pred - self.C; Vd[self.frozen] = 0
        if len(pts):
            Call, counts_all = track_step2(pts, self.C, Vd)
        else:
            Call, counts_all = pred, np.zeros(len(self.C), int)
        Cs, counts = Call[sub], counts_all[sub]
        newC = self.C.copy(); newC[sub] = Cs
        alpha = 0.5
        self.V[sub] = (1 - alpha) * self.V[sub] + alpha * (newC[sub] - self.C[sub]) / dt
        # clamp unrealistic velocity (m per frame)
        sp = np.sqrt((self.V ** 2).sum(-1, keepdims=True))
        self.V = np.where(sp > 0.012, self.V / (sp + 1e-9) * 0.012, self.V)
        self.C = newC
        self.t = t
        c = np.zeros(len(self.C), int); c[sub] = counts
        self.last_counts = c
        return self.C

def video_track(root, manifest, log=print):
    """Track through the conditioning video. Returns Tracker positioned at the first execution frame."""
    frames = sorted(manifest['frames'], key=lambda f: f['frame_index'])
    vids = [f for f in frames if f.get('phase') == 'video_demo']
    execf = [f for f in frames if f.get('phase') != 'video_demo']
    seq = vids + execf[:1]
    data = [load_frame(root, f) for f in seq]
    n_white = np.array([int(white_mask(rgb, P).sum()) for rgb, P in data])
    ref = np.median(n_white[-20:])
    big = n_white > 0.5 * ref
    first_c = int(np.argmax(big))
    # cubes: scan frames before the containers appear; keep the frame with most cubes (latest on ties)
    cubes, j = [], -1
    for jj in range(first_c - 1, max(-1, first_c - 25), -1):
        cb = cube_blobs(*data[jj])
        if len(cb) > len(cubes):
            cubes, j = cb, jj
    log(f'video: {len(seq)} frames, containers from {first_c}, cubes from {j}: ' +
        ', '.join(f"{c['color']}({c['xy'][0]:.3f},{c['xy'][1]:.3f})" for c in cubes))
    def tof(f, i):
        s = f.get('source_frame_index')
        return float(s) if s is not None else float(i)
    pts0 = white_points(*data[first_c])[:, :2]
    C0, err = init_centres(pts0, cubes)
    log(f'video: init assignment max dist {err:.3f}')
    tr = Tracker(C0, [c['color'] for c in cubes], tof(seq[first_c], first_c))
    for i in range(first_c + 1, len(seq)):
        t = tof(seq[i], i)
        if seq[i].get('phase') != 'video_demo':
            t = tr.t + 1.0
        tr.update(white_points(*data[i])[:, :2], t)
    return tr, {'first_c': first_c, 'cubes': cubes, 'init_err': err, 'n_frames': len(seq)}


# ---------------- v4: multi-camera fusion + colour-agnostic cube detection ----------------
CAMS = ('front', 'wrist')
CUBE_ZMAX = 0.045

def load_frame_cams(root, frame, cams=CAMS):
    files = {(x['camera'], x['kind']): x for x in frame['files']}
    out = {}
    for cam in cams:
        if (cam, 'rgb') not in files or (cam, 'depth') not in files:
            continue
        if cam not in (frame.get('camera_poses') or {}) or cam not in (frame.get('camera_intrinsics') or {}):
            continue
        rgb = np.array(Image.open(root / files[(cam, 'rgb')]['path']).convert('RGB')).astype(np.float32)
        dfile = files[(cam, 'depth')]
        mpu = dfile.get('meters_per_unit', 0.001) or 0.001
        d = np.array(Image.open(root / dfile['path'])).astype(np.float32) * mpu
        P = backproject(d, frame['camera_intrinsics'][cam]['matrix'], frame['camera_poses'][cam])
        out[cam] = (rgb, P, np.array(frame['camera_poses'][cam]['position_m'], float))
    return out

FUSE_ZMIN = 0.042      # above cube tops (~0.035): exposed grey/white cubes are not container points
WRIST_MIN_DIST = 0.09  # wrist points closer than this to the camera are the robot's own fingers

def fused_white_points(views, zmax=CONT_Z[1], zmin=FUSE_ZMIN, exclude_xy=None, exclude_r=0.0):
    pts = []
    for cam, v in views.items():
        rgb, P = v[0], v[1]
        m = white_mask(rgb, P)
        with np.errstate(invalid='ignore'):
            m &= (P[..., 2] < zmax) & (P[..., 2] > zmin)
            if cam == 'wrist' and len(v) > 2:
                m &= np.sqrt(((P - v[2]) ** 2).sum(-1)) > WRIST_MIN_DIST
        p = P[m]
        if exclude_xy is not None and len(p):
            p = p[np.sqrt(((p[:, :2] - np.asarray(exclude_xy)[:2]) ** 2).sum(-1)) > exclude_r]
        if cam == 'wrist' and len(p) > 1500:
            p = p[:: int(np.ceil(len(p) / 1500))]   # keep camera weights comparable
        if cam == 'front' and len(p) > 3000:
            p = p[::2]
        pts.append(p)
    return np.concatenate(pts, 0) if pts else np.zeros((0, 3))

def name_colour(rgb_med):
    r, g, b = rgb_med
    mx, mn = max(r, g, b), min(r, g, b)
    sat = (mx - mn) / (mx + 1e-6)
    if sat < 0.25:
        if mx < 80:
            return 'black'
        return 'white' if mx > 190 else 'gray'
    if mx == r:
        h = ((g - b) / (mx - mn)) % 6
    elif mx == g:
        h = (b - r) / (mx - mn) + 2
    else:
        h = (r - g) / (mx - mn) + 4
    h *= 60
    if h < 15 or h >= 330: return 'red'
    if h < 45: return 'orange'
    if h < 75: return 'yellow'
    if h < 170: return 'green'
    if h < 190: return 'cyan'
    if h < 255: return 'blue'
    return 'purple'

def detect_cubes(views, min_px=6):
    """Small objects on the table (no containers present yet). Returns list of dicts."""
    dets = []
    for cam, v in views.items():
        rgb, P = v[0], v[1]
        z = P[..., 2]
        with np.errstate(invalid='ignore'):
            m = in_box(P) & (z > 0.004) & (z < CUBE_ZMAX)
        if m.sum() == 0:
            continue
        pts = P[m]; cols = rgb[m]
        if len(pts) > 6000:
            sel = np.arange(0, len(pts), 2); pts, cols = pts[sel], cols[sel]
        for idx in cluster_xy(pts[:, :2], link=0.008):
            if len(idx) < min_px:
                continue
            p = pts[idx]
            ext = p[:, :2].max(0) - p[:, :2].min(0)
            if ext.max() > 0.065:
                continue
            col = name_colour(np.median(cols[idx], 0))
            dets.append({'cam': cam, 'xy': p[:, :2].mean(0), 'px': len(idx), 'colour': col, 'ext': ext})
    # reject parts of tall objects: same-colour points well above cube height near the detection
    tall = []
    for cam, v in views.items():
        rgb, P = v[0], v[1]
        z = P[..., 2]
        with np.errstate(invalid='ignore'):
            m = in_box(P) & (z > 0.06) & (z < 0.5)
        mx = rgb.max(-1); mn = rgb.min(-1)
        m &= (mx - mn) / (mx + 1e-6) > 0.3
        if m.sum():
            tall.append((P[m][:, :2], rgb[m]))
    keep = []
    for d in dets:
        n_t = 0
        for xy, cols_t in tall:
            near = np.sqrt(((xy - d['xy']) ** 2).sum(-1)) < 0.03
            if near.sum():
                n_t += sum(1 for c in cols_t[near][::max(1, near.sum() // 50)] if name_colour(c) == d['colour'])
        if n_t >= 5:
            continue
        keep.append(d)
    dets = keep
    # fuse across cameras (same colour, within 3 cm)
    fused = []
    for d in sorted(dets, key=lambda d: -d['px']):
        for f in fused:
            if f['colour'] == d['colour'] and np.linalg.norm(f['xy'] - d['xy']) < 0.038:
                f['cams'].append(d['cam']); break
            if np.linalg.norm(f['xy'] - d['xy']) < 0.02:
                f['cams'].append(d['cam'] + ':' + d['colour']); break
        else:
            fused.append({'xy': d['xy'], 'colour': d['colour'], 'px': d['px'], 'cams': [d['cam']]})
    return fused


def video_track2(root, manifest, log=print):
    """Fused front+wrist tracking through the conditioning video."""
    frames = sorted(manifest['frames'], key=lambda f: f['frame_index'])
    vids = [f for f in frames if f.get('phase') == 'video_demo']
    execf = [f for f in frames if f.get('phase') != 'video_demo']
    seq = vids + execf[:1]
    views = [load_frame_cams(root, f) for f in seq]
    n_white = np.array([len(fused_white_points(v)) for v in views])
    ref = np.median(n_white[-20:])
    first_c = int(np.argmax(n_white > 0.5 * ref))
    best = (None, -1)
    for jj in range(first_c - 1, max(-1, first_c - 25), -1):
        cb = detect_cubes(views[jj])
        if best[0] is None or len(cb) > len(best[0]):
            best = (cb, jj)
    cubes, j = best
    cubes = cubes or []
    log(f'video2: {len(seq)} frames, containers from {first_c}, {len(cubes)} cubes from frame {j}: ' +
        ', '.join(f"{c['colour']}({c['xy'][0]:.3f},{c['xy'][1]:.3f})" for c in cubes))
    k = len(cubes)
    pts0 = fused_white_points(views[first_c])[:, :2]
    cube_xy = np.array([c['xy'] for c in cubes])
    # containers are dropped exactly over the cubes: start at the cubes and refine on container points
    C0 = cube_xy.copy()
    for _ in range(3):
        C0, _cnt = track_step2(pts0, C0, np.zeros_like(C0))
    err = float(np.sqrt(((C0 - cube_xy) ** 2).sum(-1)).max())
    log(f'video2: init assignment max dist {err:.3f}')
    def tof(f, i):
        s = f.get('source_frame_index')
        return float(s) if s is not None else float(i)
    tr = Tracker(C0, [c['colour'] for c in cubes], tof(seq[first_c], first_c))
    hist = [tr.C.copy()]
    ref_counts = None
    for i in range(first_c + 1, len(seq)):
        t = tof(seq[i], i)
        if seq[i].get('phase') != 'video_demo':
            t = tr.t + 1.0
        prevC = tr.C.copy(); t_prev = tr.t
        tr.update(fused_white_points(views[i])[:, :2], t)
        cnt = tr.last_counts
        if ref_counts is None:
            ref_counts = np.maximum(cnt.astype(float), 1.0)
        occl = cnt < 0.5 * ref_counts
        tr.C[occl] = prevC[occl] + tr.V[occl] * (tr.t - t_prev)
        ref_counts[~occl] = 0.95 * ref_counts[~occl] + 0.05 * cnt[~occl]
        hist.append(tr.C.copy())
    return tr, {'first_c': first_c, 'cubes': cubes, 'init_err': err, 'n_frames': len(seq),
                'hist': np.array(hist), 'cube_frame': j}


def tall_obstacles(views, zmin=0.10):
    """XY points of saturated (non-white) objects standing above zmin (e.g. a pillar)."""
    out = []
    for cam, v in views.items():
        rgb, P = v[0], v[1]
        mx = rgb.max(-1); mn = rgb.min(-1)
        with np.errstate(invalid='ignore'):
            m = in_box(P) & (P[..., 2] > zmin) & (P[..., 2] < 0.6) & ((mx - mn) / (mx + 1e-6) > 0.3)
        if m.sum():
            out.append(P[m][::5, :2])
    return np.concatenate(out, 0) if out else np.zeros((0, 2))
