"""Track containers through the conditioning video in world XY using RGB-D of both cameras."""
import json, colorsys
import numpy as np
from geom import load_frame

XMIN, XMAX, YMIN, YMAX = -0.5, 0.35, -0.45, 0.45
CELL = 0.004
CAMS = ('front', 'wrist')
OBST = None  # set of (i,j) XY cells (CELL grid) occupied by static tall obstacles

def set_obstacles(fr, ee_xy, radius=0.09, zmin=0.13, dil=3):
    global OBST
    OBST = None
    pts = []
    for cam in ('front', 'wrist'):
        if cam not in fr: continue
        P = fr[cam][1]; z = P[..., 2]
        ok = np.isfinite(z) & (z > zmin) & (z < 0.6) & (P[..., 0] > XMIN) & (P[..., 0] < XMAX) & (P[..., 1] > YMIN) & (P[..., 1] < YMAX)
        pts.append(P[ok][:, :2])
    p = np.concatenate(pts) if pts else np.zeros((0, 2))
    if len(p): p = p[np.linalg.norm(p - np.asarray(ee_xy[:2]), axis=1) > radius]
    if len(p) < 20: return 0
    k = np.unique(np.floor(p / CELL).astype(int), axis=0)
    S = set()
    for a, b in k:
        for dx in range(-dil, dil + 1):
            for dy in range(-dil, dil + 1): S.add((a + dx, b + dy))
    OBST = S
    return len(k)

def frame_points(fr, zlo, zhi):
    pts, cols = [], []
    for cam in CAMS:
        if cam not in fr: continue
        rgb, P = fr[cam]
        z = P[..., 2]
        ok = np.isfinite(z) & (z > zlo) & (z < zhi) & (P[..., 0] > XMIN) & (P[..., 0] < XMAX) \
             & (P[..., 1] > YMIN) & (P[..., 1] < YMAX)
        pts.append(P[ok]); cols.append(rgb[ok])
    if not pts: return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    p, c = np.concatenate(pts), np.concatenate(cols)
    if OBST and len(p):
        k = np.floor(p[:, :2] / CELL).astype(int)
        keep = np.array([(a, b) not in OBST for a, b in k], bool)
        p, c = p[keep], c[keep]
    return p, c

def cells(p):
    """Unique XY occupancy cells -> centers (M,2)."""
    if len(p) == 0: return np.zeros((0, 2))
    k = np.unique(np.floor(p[:, :2] / CELL).astype(int), axis=0)
    return (k + 0.5) * CELL

def components(c, link=0.009, min_cells=8):
    """Simple connected components over cell centers (single linkage at `link`)."""
    n = len(c)
    if n == 0: return []
    keys = {tuple(np.floor(x / CELL).astype(int)): i for i, x in enumerate(c)}
    lab = -np.ones(n, int); cur = 0
    r = int(np.ceil(link / CELL))
    for i in range(n):
        if lab[i] >= 0: continue
        stack = [i]; lab[i] = cur
        while stack:
            j = stack.pop()
            kj = np.floor(c[j] / CELL).astype(int)
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nb = keys.get((kj[0] + dx, kj[1] + dy))
                    if nb is not None and lab[nb] < 0:
                        lab[nb] = cur; stack.append(nb)
        cur += 1
    comps = [c[lab == k] for k in range(cur)]
    return [x for x in comps if len(x) >= min_cells]

def classify_color(rgb_med):
    r, g, b = [v / 255.0 for v in rgb_med]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h *= 360
    if v < 0.25: return 'black'
    if s < 0.25: return 'white' if v > 0.6 else 'gray'
    if h < 15 or h >= 335: return 'red'
    if h < 45: return 'orange'
    if h < 70: return 'yellow'
    if h < 170: return 'green'
    if h < 200: return 'cyan'
    if h < 260: return 'blue'
    return 'purple'

def load_manifest(root, manifest_path):
    return json.loads(open(f"{root}/{manifest_path}").read())

def detect_cubes(fr):
    p, col = frame_points(fr, 0.006, 0.045)
    out = []
    if len(p) == 0: return out
    c = cells(p)
    for comp in components(c, link=0.006, min_cells=4):
        ctr = comp.mean(0)
        # colors of raw points near this component
        d = np.linalg.norm(p[:, :2] - ctr, axis=1)
        sel = d < 0.02
        med = np.median(col[sel].astype(float), 0)
        # prefer saturated pixels: median over top-half saturation
        cs = col[sel].astype(float) / 255.0
        sat = cs.max(1) - cs.min(1)
        if sel.sum() > 10:
            hi = sat >= np.percentile(sat, 50)
            med = np.median(col[sel][hi].astype(float), 0)
        if len(comp) > 200: continue
        out.append({'xy': ctr.tolist(), 'n': int(len(comp)), 'rgb': med.tolist(), 'color': classify_color(med)})
    # merge fragments of the same cube
    merged = True
    while merged:
        merged = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                if out[i]['color'] == out[j]['color'] and np.linalg.norm(np.subtract(out[i]['xy'], out[j]['xy'])) < 0.04:
                    a, b = out[i], out[j]; w = a['n'] + b['n']
                    a['xy'] = ((np.array(a['xy']) * a['n'] + np.array(b['xy']) * b['n']) / w).tolist(); a['n'] = w
                    out.pop(j); merged = True; break
            if merged: break
    return out

def detect_containers(fr, zlo=0.045, zhi=0.13):
    p, _ = frame_points(fr, zlo, zhi)
    c = cells(p)
    return [{'xy': comp.mean(0), 'n': len(comp), 'cells': comp} for comp in components(c, link=0.009, min_cells=10)]

def fix_count(groups, N):
    groups = [g for g in groups]
    while len(groups) > N and len(groups) > 1:
        ctr = np.array([g.mean(0) for g in groups])
        d = np.linalg.norm(ctr[:, None] - ctr[None], axis=2) + np.eye(len(groups)) * 9
        i, j = np.unravel_index(np.argmin(d), d.shape)
        merged = np.concatenate([groups[i], groups[j]])
        groups = [g for k, g in enumerate(groups) if k not in (i, j)] + [merged]
    while len(groups) < N and groups:
        k = int(np.argmax([len(g) for g in groups]))
        g = groups.pop(k)
        # split by 2-means along principal axis
        m = g.mean(0); u, s_, vt = np.linalg.svd(g - m, full_matrices=False)
        proj = (g - m) @ vt[0]
        groups += [g[proj < 0], g[proj >= 0]]
    return groups

def hungarian_min(cost):
    """Small brute/greedy assignment fallback (scipy if available)."""
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost)
    except Exception:
        cost = cost.copy(); rows, cols = [], []
        for _ in range(min(cost.shape)):
            i, j = np.unravel_index(np.argmin(cost), cost.shape)
            rows.append(i); cols.append(j); cost[i, :] = np.inf; cost[:, j] = np.inf
        return np.array(rows), np.array(cols)

def track(root, manifest, log=print):
    frames = [f for f in manifest['frames'] if f['phase'] == 'video_demo']
    frames.sort(key=lambda f: f['frame_index'])
    # 1. cube detection before containers appear; find first container frame
    cube_obs = []; first_cont = None
    for k, f in enumerate(frames):
        fr = load_frame(root, f)
        conts = detect_containers(fr)
        if len(conts) > 0 and sum(c['n'] for c in conts) > 40:
            first_cont = k; break
        cube_obs.append(detect_cubes(fr))
        if len(cube_obs) > 12:  # enough samples; skip ahead quickly
            pass
    if first_cont is None:
        raise RuntimeError('no containers found in video')
    # choose cube observation: last few pre-container frames, pick most common count
    cubes = cube_obs[-1] if cube_obs else []
    for obs in reversed(cube_obs[-6:]):
        if len(obs) >= len(cubes): cubes = obs
    log(f"first container frame {first_cont}; cubes: {[ (c['color'], [round(v,3) for v in c['xy']]) for c in cubes]}")
    # 2. initialise tracks a few frames after containers appear
    init_k = min(first_cont + 2, len(frames) - 1)
    fr = load_frame(root, frames[init_k])
    conts = detect_containers(fr)
    groups = [c['cells'] for c in conts]
    N = len(cubes) if cubes else len(groups)
    groups = fix_count(groups, N)
    tracks = np.array([g.mean(0) for g in groups])
    log(f"init frame {init_k}: {len(tracks)} containers")
    # assign cubes -> tracks by Hungarian on distance
    content = [None] * len(tracks)
    if len(cubes):
        cost = np.linalg.norm(np.array([c['xy'] for c in cubes])[:, None] - tracks[None], axis=2)
        ri, ci = hungarian_min(cost)
        for a, b in zip(ri, ci):
            if cost[a, b] < 0.05:
                content[b] = cubes[a]['color']
            else:
                log(f"warn cube {cubes[a]['color']} far from container ({cost[a,b]:.3f})")
    vel = np.zeros_like(tracks)
    hist = [tracks.copy()]
    # 3. iterate frames
    for k in range(init_k + 1, len(frames)):
        fr = load_frame(root, frames[k])
        p, _ = frame_points(fr, 0.045, 0.13)
        c = cells(p)
        pred = tracks + 0.5 * vel
        new = pred.copy()
        if len(c):
            for it in range(3):
                d = np.linalg.norm(c[:, None] - new[None], axis=2)
                a = d.argmin(1); dm = d.min(1)
                upd = new.copy()
                for t in range(len(new)):
                    sel = (a == t) & (dm < 0.04)
                    if sel.sum() >= 6:
                        upd[t] = c[sel].mean(0)
                new = upd
        vel = 0.6 * vel + 0.4 * (new - tracks)
        tracks = new
        hist.append(tracks.copy())
    return {'tracks': tracks, 'content': content, 'cubes': cubes, 'hist': np.array(hist),
            'first_cont': first_cont, 'init_k': init_k}

def rect_fit(cells_xy):
    """Return (yaw in [-pi/4, pi/4), side_a, side_b) of min-area enclosing rect over angles."""
    best = None
    c = cells_xy - cells_xy.mean(0)
    for deg in range(0, 90, 2):
        t = np.deg2rad(deg); R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        q = c @ R.T
        lo = np.percentile(q, 3, axis=0); hi = np.percentile(q, 97, axis=0)
        ext = hi - lo; area = ext[0] * ext[1]
        if best is None or area < best[0]: best = (area, t, ext)
    t = best[1]
    if t >= np.pi / 4: t -= np.pi / 2
    return t, float(best[2][0]), float(best[2][1])

def top_detections(fr, zlo=0.062, zhi=0.13, min_cells=12, unit=None):
    p, _ = frame_points(fr, zlo, zhi)
    c = cells(p)
    comps = components(c, link=0.006, min_cells=min_cells)
    dets = []
    for comp in comps:
        k = 1
        if unit and len(comp) > 1.6 * unit:
            k = int(round(len(comp) / unit))
        if k <= 1:
            dets.append((comp.mean(0), len(comp)))
        else:
            # k-means split
            m = comp.mean(0); u, s_, vt = np.linalg.svd(comp - m, full_matrices=False)
            proj = (comp - m) @ vt[0]
            qs = np.quantile(proj, (np.arange(k) + 0.5) / k)
            ctr = m + qs[:, None] * vt[0][None]
            for _ in range(8):
                a = np.linalg.norm(comp[:, None] - ctr[None], axis=2).argmin(1)
                ctr = np.array([comp[a == j].mean(0) if (a == j).any() else ctr[j] for j in range(k)])
            for j in range(k):
                dets.append((ctr[j], int((a == j).sum())))
    return dets

def track2(root, manifest, log=print, gate=0.03, reverse=False, cams=None):
    frames = [f for f in manifest['frames'] if f['phase'] == 'video_demo']
    frames.sort(key=lambda f: f['frame_index'])
    global CAMS
    CAMS = cams or ('front', 'wrist')
    cube_obs = []; first_cont = None
    for k, f in enumerate(frames):
        fr = load_frame(root, f)
        conts = detect_containers(fr)
        if len(conts) > 0 and sum(c['n'] for c in conts) > 40:
            first_cont = k; break
        if k % 3 == 0 or k < 3:
            cube_obs.append(detect_cubes(fr))
    if first_cont is None:
        raise RuntimeError('no containers found in video')
    cubes = cube_obs[-1] if cube_obs else []
    for obs in reversed(cube_obs[-6:]):
        if len(obs) > len(cubes): cubes = obs
    log(f"first container frame {first_cont}; cubes: {[(c['color'], [round(v,3) for v in c['xy']]) for c in cubes]}")
    init_k = min(first_cont + 2, len(frames) - 1)
    fr = load_frame(root, frames[init_k])
    groups = [c['cells'] for c in detect_containers(fr)]
    N = len(cubes) if cubes else len(groups)
    groups = fix_count(groups, N)
    tracks = np.array([g.mean(0) for g in groups])
    content = [None] * len(tracks)
    if len(cubes):
        cost = np.linalg.norm(np.array([c['xy'] for c in cubes])[:, None] - tracks[None], axis=2)
        ri, ci = hungarian_min(cost)
        for a, b in zip(ri, ci):
            content[b] = cubes[a]['color']
            if cost[a, b] > 0.05: log(f"warn cube {cubes[a]['color']} far from container ({cost[a,b]:.3f})")
    # unit size of a fully visible top face
    dets = top_detections(fr)
    unit = float(np.median([n for _, n in dets])) if dets else 80.0
    unit = max(unit, 40.0)
    log(f"init {init_k}: N={N} unit={unit:.0f}")
    # refine init positions with top detections
    if dets:
        D = np.array([d for d, _ in dets])
        cost = np.linalg.norm(tracks[:, None] - D[None], axis=2)
        ri, ci = hungarian_min(cost)
        for a, b in zip(ri, ci):
            if cost[a, b] < 0.03 and dets[b][1] > 0.6 * unit: tracks[a] = D[b]
    vel = np.zeros_like(tracks); lastsrc = frames[init_k].get('source_frame_index', init_k)
    hist = [tracks.copy()]; misses = np.zeros(len(tracks), int); nmiss_total = 0
    order = list(range(init_k + 1, len(frames)))
    for k in order:
        src = frames[k].get('source_frame_index', k); dt = max(1, src - lastsrc); lastsrc = src
        fr = load_frame(root, frames[k])
        dets = top_detections(fr, unit=unit)
        pred = tracks + vel * dt
        new = pred.copy(); matched = np.zeros(len(tracks), bool)
        if dets:
            D = np.array([d for d, _ in dets]); n = np.array([m for _, m in dets])
            cost = np.linalg.norm(pred[:, None] - D[None], axis=2)
            ri, ci = hungarian_min(np.where(cost < gate, cost, 1e3))
            for a, b in zip(ri, ci):
                if cost[a, b] < gate:
                    w = 1.0 if n[b] > 0.6 * unit else 0.5
                    new[a] = w * D[b] + (1 - w) * pred[a]; matched[a] = True
        # unmatched: try all-height points near prediction
        if not matched.all():
            p, _ = frame_points(fr, 0.015, 0.13); c = cells(p)
            for t in np.where(~matched)[0]:
                if len(c) == 0: break
                d = np.linalg.norm(c - pred[t], axis=1)
                # exclude cells closer to another track
                others = np.delete(new, t, axis=0)
                do = np.linalg.norm(c[:, None] - others[None], axis=2).min(1) if len(others) else np.full(len(c), 9.)
                sel = (d < 0.035) & (d < do)
                if sel.sum() >= 15:
                    new[t] = 0.5 * c[sel].mean(0) + 0.5 * pred[t]
                misses[t] += 1; nmiss_total += 1
        v_new = (new - tracks) / dt
        vel = np.where(matched[:, None], 0.5 * vel + 0.5 * v_new, 0.8 * vel)
        tracks = new
        hist.append(tracks.copy())
    log(f"misses per track {misses.tolist()} total {nmiss_total}")
    return {'tracks': tracks, 'content': content, 'cubes': cubes, 'hist': np.array(hist),
            'first_cont': first_cont, 'init_k': init_k, 'misses': misses.tolist(), 'unit': unit}
