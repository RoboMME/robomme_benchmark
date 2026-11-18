"""Conditioning-video analysis for VideoPlaceButton.
Input: list of (t, front_rgb uint8 HxWx3, wrist_rgb) in time order.
Output: pixel of target (in last frame) where the instructed cube was first placed after the button press."""
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

COLORS = ('red', 'green', 'blue')

def target_mask(a):
    R, G, B = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    return (B > G + 12) & (R > 130) & (R - B < 90)

def cube_mask(a, color):
    R, G, B = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    if color == 'red':
        return (R > 120) & (G < 50) & (B < 50)
    if color == 'green':
        return (G > 120) & (R < 80) & (B < 80)
    return (B > 120) & (R < 60) & (G < 60)

def blobs(mk, minsz=15, close=True):
    if close:
        mk = ndimage.binary_closing(mk, iterations=1)
    lab, n = ndimage.label(mk)
    out = []
    if n == 0:
        return out
    idx = np.arange(1, n + 1)
    sizes = ndimage.sum(mk, lab, idx)
    coms = ndimage.center_of_mass(mk, lab, idx)
    for s, (cy, cx) in zip(sizes, coms):
        if s >= minsz:
            out.append((float(cx), float(cy), float(s)))
    return out

def cube_pos(a, color, ymax=200):
    bs = [b for b in blobs(cube_mask(a, color), 6) if b[1] < ymax]
    if not bs:
        return None
    b = max(bs, key=lambda b: b[2])
    return (b[0], b[1])

def press_feature(w):
    """Large solid white button pad near bottom-centre of the wrist view."""
    a = w.astype(int)
    mn = a.min(2); mx = a.max(2)
    wh = (mn > 215) & (mx - mn < 15)
    lab, n = ndimage.label(wh)
    if n == 0:
        return 0.0, None
    sizes = ndimage.sum(wh, lab, np.arange(1, n + 1))
    k = int(np.argmax(sizes))
    cy, cx = ndimage.center_of_mass(wh, lab, k + 1)
    return float(sizes[k]), (float(cx), float(cy))

def is_press(w):
    s, c = press_feature(w)
    return s > 3500 and c is not None and c[1] > 140 and abs(c[0] - 128) < 60, s, c

def home_slots(alld, rad=6.0, frac=0.3):
    """Cluster detections; slots are clusters present in >= frac of frames."""
    pts = [(d[0], d[1]) for ds in alld for d in ds]
    n = len(alld)
    cl = []  # [sum_x, sum_y, count, set(frames)]
    for fi, ds in enumerate(alld):
        for d in ds:
            for c in cl:
                if (c[0] / c[2] - d[0]) ** 2 + (c[1] / c[2] - d[1]) ** 2 < rad ** 2:
                    c[0] += d[0]; c[1] += d[1]; c[2] += 1; c[3].add(fi); break
            else:
                cl.append([d[0], d[1], 1, {fi}])
    good = [np.array([c[0] / c[2], c[1] / c[2]]) for c in cl if len(c[3]) >= frac * n]
    return good

class Tracker:
    def __init__(self, gate=22.0):
        self.tracks = []  # dict(pos, vel, hist, miss, hits)
        self.gate = gate

    def step(self, t, dets):
        P = [np.array(d[:2]) for d in dets]
        nT = len(self.tracks)
        assigned = set()
        if nT and P:
            C = np.zeros((nT, len(P)))
            for i, tr in enumerate(self.tracks):
                pred = tr['pos'] + tr['vel']
                for j, p in enumerate(P):
                    C[i, j] = np.linalg.norm(pred - p)
            ri, ci = linear_sum_assignment(C)
            for i, j in zip(ri, ci):
                if C[i, j] < self.gate:
                    tr = self.tracks[i]
                    nv = P[j] - tr['pos']
                    tr['vel'] = 0.5 * nv if tr['miss'] == 0 else np.zeros(2)
                    tr['pos'] = P[j]; tr['miss'] = 0; tr['hits'] += 1
                    assigned.add(j); tr['seen'] = True
                    continue
        for tr in self.tracks:
            if not tr.get('seen'):
                tr['miss'] += 1; tr['vel'] = tr['vel'] * 0.5
            tr.pop('seen', None)
        for j, p in enumerate(P):
            if not getattr(self, 'spawn', True):
                break
            if j in assigned:
                continue
            # new track only if far from all existing tracks
            if all(np.linalg.norm(tr['pos'] - p) > 14 for tr in self.tracks):
                self.tracks.append(dict(pos=p, vel=np.zeros(2), hist=[], miss=0, hits=1, born=t))
        for tr in self.tracks:
            tr['hist'].append((t, tr['pos'].copy(), tr['miss']))

def analyze(frames, color, log=print, gate=20.0):
    """frames: list of (t, front, wrist). Returns dict with target pixel in the last frame."""
    n = len(frames)
    ts = [f[0] for f in frames]
    # press
    press = [is_press(f[2])[0] for f in frames]
    press_idx = [i for i, p in enumerate(press) if p]
    log('press frames', [round(ts[i], 2) for i in press_idx][:40])
    # tracks
    alld = []
    cubes = []
    for i, (t, fr, wr) in enumerate(frames):
        alld.append([b for b in blobs(target_mask(fr), 15) if b[1] < 200])
        cubes.append(cube_pos(fr, color))
    slots = home_slots(alld)
    log('home slots', [tuple(np.round(s_, 1)) for s_ in slots])
    tk = Tracker(gate=gate)
    for s_ in slots:
        tk.tracks.append(dict(pos=np.array(s_), vel=np.zeros(2), hist=[], miss=0, hits=1, born=0))
    tk.spawn = False
    for i, (t, fr, wr) in enumerate(frames):
        tk.step(t, alld[i])
    res = dict(press_t=None, place_t=None, track=None, pixel=None, ok=False)
    if not press_idx:
        log('NO PRESS DETECTED')
        return res
    # first run of press
    runs = []
    for i in press_idx:
        if runs and i - runs[-1][-1] <= 2:
            runs[-1].append(i)
        else:
            runs.append([i])
    pr = runs[0]
    p_end = pr[-1]
    res['press_t'] = ts[pr[0]]
    # cube position at press (last known before end of press)
    ref = None
    for i in range(p_end, -1, -1):
        if cubes[i] is not None:
            ref = np.array(cubes[i]); break
    log('press run t', ts[pr[0]], ts[pr[-1]], 'cube at press', ref)
    # stationary segments after press
    seg_start = None
    place = None
    for i in range(p_end + 1, n):
        c = cubes[i]
        if c is None:
            seg_start = None; continue
        c = np.array(c)
        if seg_start is None:
            seg_start = i; continue
        c0 = np.array(cubes[seg_start])
        if np.linalg.norm(c - c0) > 2.0:
            seg_start = i; continue
        if ts[i] - ts[seg_start] >= 1.0:
            if ref is None or np.linalg.norm(c0 - ref) > 8:
                place = seg_start; break
    if place is None:
        log('NO PLACEMENT FOUND')
        return res
    res['place_t'] = ts[place]
    cpos = np.array(cubes[place]) + np.array([0, 4.0])
    # nearest track at placement time (use hist at that index)
    best = None
    for k, tr in enumerate(tk.tracks):
        h = [x for x in tr['hist'] if abs(x[0] - ts[place]) < 1e-6]
        if not h:
            continue
        d = np.linalg.norm(h[0][1] - cpos)
        if best is None or d < best[0]:
            best = (d, k)
    log('placement t', ts[place], 'cube px', cubes[place], 'track', best)
    if best is None or best[0] > 20:
        log('placement not near a target')
        if best is None:
            return res
    tr = tk.tracks[best[1]]
    res['track'] = best[1]
    res['pixel'] = tuple(float(x) for x in tr['pos'])
    res['ok'] = True
    res['all_final'] = [tuple(np.round(t_['pos'], 1)) for t_ in tk.tracks]
    res['track_path'] = [(round(h[0], 2), tuple(np.round(h[1], 1)), h[2]) for h in tr['hist'][::5]]
    return res


def swap_events(ts, alld, slots, occ_r=5.0, off_r=8.0, sym_tol=10.0, log=print, vac_min=0.6):
    """Return list of (t_start, t_end, a, b) slot-index swaps in time order."""
    S = np.array(slots)
    K = len(S)
    off = []
    for ds in alld:
        o = []
        for d in ds:
            p = np.array(d[:2])
            if np.min(np.linalg.norm(S - p, axis=1)) > off_r:
                o.append(p)
        off.append(o)
    # per-frame pair scores
    pair_frames = {}
    for fi, o in enumerate(off):
        if len(o) < 2:
            continue
        for a in range(K):
            for b in range(a + 1, K):
                mid = S[a] + S[b]
                sep = np.linalg.norm(S[a] - S[b])
                hit = False
                for i in range(len(o)):
                    for j in range(i + 1, len(o)):
                        if np.linalg.norm(o[i] + o[j] - mid) < sym_tol:
                            dd = np.linalg.norm(o[i] - o[j])
                            if 0.45 * sep < dd < 1.6 * sep:
                                hit = True
                if hit:
                    pair_frames.setdefault((a, b), []).append(fi)
    # group each pair's frames into runs (events)
    events = []
    for (a, b), fl in pair_frames.items():
        run = [fl[0]]
        for f in fl[1:] + [None]:
            if f is not None and ts[f] - ts[run[-1]] <= 1.0:
                run.append(f)
            else:
                if len(run) >= 2:
                    # vacancy check: both slots empty in most frames of the run
                    i0, i1 = run[0], run[-1]
                    vac = 0; tot = 0
                    for fi in range(i0, i1 + 1):
                        ds = alld[fi]
                        tot += 1
                        oa = any((d[0] - S[a][0]) ** 2 + (d[1] - S[a][1]) ** 2 < occ_r ** 2 for d in ds)
                        ob = any((d[0] - S[b][0]) ** 2 + (d[1] - S[b][1]) ** 2 < occ_r ** 2 for d in ds)
                        vac += (not oa) and (not ob)
                    vfrac = vac / max(1, tot)
                    if vfrac >= vac_min:
                        events.append([ts[run[0]], ts[run[-1]], a, b, len(run), round(vfrac, 2)])
                    else:
                        log('reject event', round(ts[i0], 2), a, b, len(run), round(vfrac, 2))
                if f is not None:
                    run = [f]
    events.sort()
    # resolve conflicts: overlapping events sharing a slot -> keep stronger
    keep = []
    for e in events:
        clash = [k for k in keep if (k[0] <= e[1] + 0.3 and e[0] <= k[1] + 0.3) and ({k[2], k[3]} & {e[2], e[3]})]
        if clash:
            k = clash[0]
            if e[4] > k[4]:
                keep.remove(k); keep.append(e)
            continue
        keep.append(e)
    keep.sort()
    return keep


def analyze2(frames, color, log=print):
    n = len(frames)
    ts = [f[0] for f in frames]
    res = dict(press_t=None, place_t=None, slot=None, pixel=None, ok=False)
    press_idx = [i for i, f in enumerate(frames) if is_press(f[2])[0]]
    alld, cubes = [], []
    for (t, fr, wr) in frames:
        alld.append([b for b in blobs(target_mask(fr), 15) if b[1] < 200])
        cubes.append(cube_pos(fr, color))
    slots = home_slots(alld, frac=0.15)
    res['slots'] = [tuple(np.round(s_, 1)) for s_ in slots]
    log('home slots', res['slots'])
    ev = swap_events(ts, alld, slots, log=log)
    res['events'] = [tuple(e) for e in ev]
    log('swap events', res['events'])
    if not press_idx:
        log('NO PRESS DETECTED')
        return res
    runs = []
    for i in press_idx:
        if runs and i - runs[-1][-1] <= 2:
            runs[-1].append(i)
        else:
            runs.append([i])
    pr = runs[0]; p_end = pr[-1]
    res['press_t'] = ts[pr[0]]
    ref = None
    for i in range(p_end, -1, -1):
        if cubes[i] is not None:
            ref = np.array(cubes[i]); break
    log('press t', ts[pr[0]], ts[p_end], 'cube at press', ref, 'runs', len(runs))
    seg_start = None; place = None
    for i in range(p_end + 1, n):
        c = cubes[i]
        if c is None:
            seg_start = None; continue
        c = np.array(c)
        if seg_start is None:
            seg_start = i; continue
        if np.linalg.norm(c - np.array(cubes[seg_start])) > 2.0:
            seg_start = i; continue
        if ts[i] - ts[seg_start] >= 1.0:
            if ref is None or np.linalg.norm(np.array(cubes[seg_start]) - ref) > 8:
                place = seg_start; break
    if place is None:
        log('NO PLACEMENT FOUND')
        return res
    res['place_t'] = ts[place]
    cpos = np.array(cubes[place]) + np.array([0, 4.0])
    S = np.array(slots)
    dists = np.linalg.norm(S - cpos, axis=1)
    s = int(np.argmin(dists))
    log('placement t', ts[place], 'cube px', np.round(cubes[place], 1), 'slot', s, 'dist', round(float(dists[s]), 1))
    res['place_slot'] = s; res['place_dist'] = float(dists[s])
    tp = ts[place]
    for e in ev:
        if e[0] > tp - 0.2:
            if s == e[2]:
                s = e[3]
            elif s == e[3]:
                s = e[2]
    res['slot'] = s
    res['pixel'] = tuple(float(x) for x in S[s])
    res['ok'] = True
    log('final slot', s, res['pixel'])
    return res
