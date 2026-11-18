import json, numpy as np, colorsys
from geom import depth_to_world, load_frame

XMIN, XMAX, YMIN, YMAX = -0.45, 0.35, -0.45, 0.45

def frame_points(root, f, cam='front'):
    rgb, d = load_frame(root, f, cam)
    P, ok = depth_to_world(d, f['camera_intrinsics'][cam]['matrix'], f['camera_poses'][cam])
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    ok = ok & (X > XMIN) & (X < XMAX) & (Y > YMIN) & (Y < YMAX)
    return rgb, P, ok

def hsv_img(rgb):
    a = rgb.astype(np.float32) / 255.0
    mx = a.max(-1); mn = a.min(-1); s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    h = np.zeros_like(mx); dlt = np.maximum(mx - mn, 1e-6)
    h = np.where(mx == r, ((g - b) / dlt) % 6, h)
    h = np.where(mx == g, (b - r) / dlt + 2, h)
    h = np.where(mx == b, (r - g) / dlt + 4, h)
    return h * 60, s, mx

def tall_columns(P, ok, cell=0.01):
    Z = P[..., 2]
    t = ok & (Z > 0.09) & (Z < 0.14)
    if t.sum() < 15: return None
    g = np.floor(P[t][:, :2] / cell).astype(int)
    S = set()
    for a, b in g:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1): S.add((a + dx, b + dy))
    return S

def container_mask(rgb, P, ok):
    h, s, v = hsv_img(rgb)
    Z = P[..., 2]
    m = ok & (Z > 0.035) & (Z < 0.11) & (s < 0.18) & (v > 0.45)
    S = tall_columns(P, ok)
    if S and m.any():
        ii = np.nonzero(m)
        g = np.floor(P[ii][:, :2] / 0.01).astype(int)
        bad = np.array([(a, b) in S for a, b in g])
        m[ii[0][bad], ii[1][bad]] = False
    return m

def cube_mask(rgb, P, ok):
    h, s, v = hsv_img(rgb)
    Z = P[..., 2]
    return ok & (Z > 0.002) & (Z < 0.05) & (s > 0.45) & (v > 0.2)

def color_name(hue, sat, val):
    if hue < 15 or hue >= 340: return 'red'
    if hue < 45: return 'orange'
    if hue < 70: return 'yellow'
    if hue < 165: return 'green'
    if hue < 200: return 'cyan'
    if hue < 260: return 'blue'
    if hue < 340: return 'purple'
    return 'unknown'

def cluster_cc(xy, cell=0.006, min_pts=8):
    """grid connected components of xy points. returns list of index arrays"""
    if len(xy) == 0: return []
    g = np.floor((xy - np.array([XMIN, YMIN])) / cell).astype(int)
    keys = {}
    for i, k in enumerate(map(tuple, g)): keys.setdefault(k, []).append(i)
    seen = set(); comps = []
    for k in keys:
        if k in seen: continue
        stack = [k]; seen.add(k); comp = []
        while stack:
            c = stack.pop(); comp.extend(keys[c])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (c[0] + dx, c[1] + dy)
                    if n in keys and n not in seen: seen.add(n); stack.append(n)
        if len(comp) >= min_pts: comps.append(np.array(comp))
    return comps

def detect_cubes(rgb, P, ok):
    m = cube_mask(rgb, P, ok)
    xy = np.stack([P[..., 0][m], P[..., 1][m]], 1)
    h, s, v = hsv_img(rgb)
    hh, ss, vv = h[m], s[m], v[m]
    out = []
    for comp in cluster_cc(xy, cell=0.005, min_pts=4):
        # hue circular mean
        ang = np.deg2rad(hh[comp]); hue = np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360
        if np.ptp(xy[comp], 0).max() > 0.07: continue
        out.append({'xy': xy[comp].mean(0), 'color': color_name(hue, ss[comp].mean(), vv[comp].mean()), 'hue': float(hue), 'n': len(comp)})
    return [c for c in merge_cubes(out) if c['n'] >= 10]

def merge_cubes(out):
    merged = []
    for c in sorted(out, key=lambda c: -c['n']):
        for m2 in merged:
            if m2['color'] == c['color'] and np.linalg.norm(m2['xy'] - c['xy']) < 0.03:
                w = m2['n'] + c['n']; m2['xy'] = (m2['xy'] * m2['n'] + c['xy'] * c['n']) / w; m2['n'] = w; break
        else:
            merged.append(dict(c))
    return merged

def kmeans_track(xy, centers, iters=8, max_r=0.035):
    c = centers.copy()
    if len(xy) == 0: return c, np.zeros(len(c), int)
    for _ in range(iters):
        d = ((xy[:, None, :] - c[None]) ** 2).sum(-1)
        lab = d.argmin(1); dm = np.sqrt(d.min(1))
        cnt = np.zeros(len(c), int)
        for k in range(len(c)):
            sel = (lab == k) & (dm < max_r)
            cnt[k] = sel.sum()
            if cnt[k] >= 5: c[k] = xy[sel].mean(0)
    return c, cnt

TOP_Z = -1.0
FUSE_WRIST = True

def tracking_points(root, f):
    rgb, P, ok = frame_points(root, f)
    m = container_mask(rgb, P, ok) & (P[..., 2] > TOP_Z)
    pts = [P[m][:, :2]]
    if FUSE_WRIST:
        try:
            rgb2, P2, ok2 = frame_points(root, f, 'wrist')
            m2 = container_mask(rgb2, P2, ok2) & (P2[..., 2] > TOP_Z)
            q = P2[m2][:, :2]
            if len(q) > 0:
                q = q[::max(1, len(q) // max(1, 3 * len(pts[0])))]  # keep wrist from dominating
                pts.append(q)
        except Exception:
            pass
    return np.concatenate(pts, 0)

def load_manifest(root, manifest_path):
    m = json.load(open(f"{root}/{manifest_path}"))
    return m['frames']

def run_tracking(root, frames, log=None):
    """returns dict: cube colors->final container xy, tracks."""
    vid = [f for f in frames if f['phase'] == 'video_demo']
    ex = [f for f in frames if f['phase'] == 'execution']
    allf = sorted(vid, key=lambda f: f['frame_index']) + ex[-1:]
    # find cube phase: frames with cubes visible & no containers
    cube_obs = None; start = None
    for i, f in enumerate(allf):
        rgb, P, ok = frame_points(root, f)
        cm = container_mask(rgb, P, ok).sum()
        cubes = detect_cubes(rgb, P, ok)
        try:
            rgb2, P2, ok2 = frame_points(root, f, 'wrist')
            cubes = merge_cubes(cubes + detect_cubes(rgb2, P2, ok2 & (P2[..., 2] < 0.05)))
        except Exception:
            pass
        if cm < 60 and len(cubes) > 0:
            cube_obs = (i, cubes)
        if cm >= 60 and cube_obs is not None:
            start = i; break
    if start is None: raise RuntimeError('no container start found')
    cubes0 = cube_obs[1]
    # containers at the first container frame: top-face components from both cameras
    f = allf[start]
    pts = []
    for cam in ('front', 'wrist'):
        try:
            rgb, P, ok = frame_points(root, f, cam)
            mm = container_mask(rgb, P, ok) & (P[..., 2] > 0.06)
            pts.append(P[mm][:, :2])
        except Exception:
            pass
    pts = np.concatenate(pts, 0)
    comps = [pts[c].mean(0) for c in cluster_cc(pts, cell=0.004, min_pts=15)]
    # merge components closer than 2.5 cm (split top faces)
    cents = []
    for c in comps:
        for k, e in enumerate(cents):
            if np.linalg.norm(e - c) < 0.025: cents[k] = (e + c) / 2; break
        else: cents.append(c)
    cubes = []
    for c in cents:
        d = [np.linalg.norm(q['xy'] - c) for q in cubes0]
        if d and min(d) < 0.03:
            q = dict(cubes0[int(np.argmin(d))]); q['xy'] = c; cubes.append(q)
        else:
            cubes.append({'xy': c, 'color': 'unknown', 'hue': -1, 'n': 0})
    # cubes detected but no container component found: add as tracks too
    tall = []
    for cam in ('front', 'wrist'):
        try:
            rgb, P, ok = frame_points(root, allf[cube_obs[0]], cam)
            h_, s_, v_ = hsv_img(rgb)
            tm = ok & (s_ > 0.3) & (P[..., 2] > 0.07) & (P[..., 2] < 0.5)
            tall.append(P[tm][:, :2])
        except Exception:
            pass
    tall = np.concatenate(tall, 0) if tall else np.zeros((0, 2))
    for q in cubes0:
        if all(np.linalg.norm(q['xy'] - c['xy']) > 0.03 for c in cubes):
            if len(tall) and (np.linalg.norm(tall - q['xy'], axis=1) < 0.04).sum() > 10: continue
            cubes.append(dict(q))
    centers = np.array([c['xy'] for c in cubes], float)
    tracks = [centers.copy()]
    counts = []
    for f in allf[start:]:
        xy = tracking_points(root, f)
        pred = centers + (tracks[-1] - tracks[-2]) * 0.5 if len(tracks) > 1 else centers
        centers, cnt = kmeans_track(xy, pred)
        tracks.append(centers.copy()); counts.append(cnt)
    return {'cubes': cubes, 'cube_frame': cube_obs[0], 'start': start,
            'final': centers, 'tracks': np.array(tracks), 'counts': np.array(counts)}
