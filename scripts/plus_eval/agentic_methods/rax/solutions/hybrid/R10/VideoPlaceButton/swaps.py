"""Slot + pairwise-swap model of the target shell game."""
import numpy as np
from vision import detect_targets, cube_mask, blobs

def arm_mask(a):
    a = a.astype(np.int32); R, G, B = a[..., 0], a[..., 1], a[..., 2]
    return (abs(R - G) < 14) & (abs(G - B) < 14)

def cube_blobs(a):
    out = []
    for c in ('red', 'green', 'blue'):
        out += blobs(cube_mask(a, c), 25)
    return out

def find_slots(dets_list, min_support=0.2, merge=7.0, min_sep=13.0, min_area=35):
    pts = [(d[0], d[1]) for dets in dets_list for d in dets if d[2] >= min_area]
    centers = []
    for p in pts:
        best = None
        for c in centers:
            dd = np.hypot(c[0] - p[0], c[1] - p[1])
            if dd < merge and (best is None or dd < best[0]):
                best = (dd, c)
        if best:
            c = best[1]; c[2] += 1; c[3] += p[0]; c[4] += p[1]; c[0] = c[3] / c[2]; c[1] = c[4] / c[2]
        else:
            centers.append([p[0], p[1], 1, p[0], p[1]])
    T = len(dets_list)
    cand = sorted([c for c in centers if c[2] >= min_support * T], key=lambda c: -c[2])
    slots = []
    for c in cand:
        if all(np.hypot(c[0] - s[0], c[1] - s[1]) >= min_sep for s in slots):
            slots.append((c[0], c[1]))
    return np.array(slots)

def slot_status(img, dets, slots, cubes, occ_r=7, near=7.0, off_min=9.0, off_area=50):
    """status per slot: 1 occupied, 0 vacant, -1 occluded by arm, -2 covered by a cube."""
    am = arm_mask(img)
    st = np.zeros(len(slots), int)
    for s, (x, y) in enumerate(slots):
        dd = [np.hypot(d[0] - x, d[1] - y) for d in dets]
        if dd and min(dd) < near:
            st[s] = 1
        elif any(np.hypot(c[0] - x, c[1] + 3 - y) < 9 for c in cubes):
            st[s] = -2
        else:
            x0, x1 = int(max(0, x - occ_r)), int(min(img.shape[1], x + occ_r + 1))
            y0, y1 = int(max(0, y - occ_r)), int(min(img.shape[0], y + occ_r + 1))
            if am[y0:y1, x0:x1].mean() > 0.3:
                st[s] = -1
    off = [d for d in dets if d[2] >= off_area and min(np.hypot(slots[:, 0] - d[0], slots[:, 1] - d[1])) >= off_min]
    return st, off

def _pick_pair(seg_stat):
    vac = (seg_stat == 0).sum(0) + 0.3 * (seg_stat == -1).sum(0)
    frac = vac / max(1, len(seg_stat))
    o = np.argsort(-frac)
    return (int(o[0]), int(o[1])), frac

def detect_events(frames, times, cam, log=None, max_gap=0.35):
    """General shuffle events: per motion interval, track movers in table-plane coordinates
    and return the slot permutation."""
    from perm import track_event
    dets_list = [detect_targets(f) for f in frames]
    slots = find_slots(dets_list)
    stat = []; offs = []
    for f, d in zip(frames, dets_list):
        st, off = slot_status(f, d, slots, cube_blobs(f))
        stat.append(st); offs.append(off)
    stat = np.array(stat); times = np.asarray(times, float)
    Wslots = cam.to_world(slots)[:, :2] if len(slots) else np.zeros((0, 2))
    Wdets = [cam.to_world([[d[0], d[1]] for d in ds if d[2] >= 40])[:, :2] if any(d[2] >= 40 for d in ds) else np.zeros((0, 2)) for ds in dets_list]
    moving = np.array([len(o) > 0 for o in offs])
    idx = np.nonzero(moving)[0]
    intervals = []
    if len(idx):
        a = b = idx[0]
        for i in idx[1:]:
            if times[i] - times[b] <= max_gap:
                b = i
            else:
                intervals.append((a, b)); a = b = i
        intervals.append((a, b))
    events = []
    for (a, b) in intervals:
        if times[b] - times[a] < 0.3:
            if log: log('skip short interval t=%.2f-%.2f' % (times[a], times[b]))
            continue
        perm, q = track_event(Wslots, Wdets, stat, a, b)
        if perm is None:
            if log: log('interval t=%.2f-%.2f no event %s' % (times[a], times[b], q))
            continue
        nontriv = {k: v for k, v in perm.items() if k != v}
        events.append(dict(a=int(a), b=int(b), t=float(times[(a + b) // 2]), perm=perm, q=q))
        if log: log('event t=%.2f-%.2f perm %s q %s' % (times[a], times[b], nontriv, q))
    return slots, events, stat, dets_list


def follow_perm(slot, t0, events):
    s = slot
    for e in events:
        if e['t'] > t0:
            s = e['perm'].get(s, s)
    return s


_DEF_K = {'matrix': [[128.0, 0, 128.0], [0, 128.0, 128.0], [0, 0, 1]]}
_DEF_P = {'position_m': [0.3, 0.0, 0.4],
          'quaternion_wxyz': [0.6881909515745244, 0.1624598848052982, 0.16245988480529822, 0.6881909515745244]}


def _multi_pairs(slots, dets_list, stat, a, b, cam, log=None):
    """For a single motion interval with >=3 vacated slots: simultaneous pairwise swaps,
    identified by point symmetry of the movers about each pair's midpoint."""
    from perm import pair_evidence, best_matching
    Ws = cam.to_world(slots)[:, :2]
    Wd = []
    for k in range(len(dets_list)):
        if k < a or k > b:
            Wd.append(np.zeros((0, 2))); continue
        pts = [[x[0], x[1]] for x in dets_list[k] if x[2] >= 40]
        Wd.append(cam.to_world(pts)[:, :2] if pts else np.zeros((0, 2)))
    V, sc, frac = pair_evidence(Ws, Wd, stat, a, b)
    if len(V) < 3:
        return None
    M, tot = best_matching(V, sc)
    unp = [v for v in V if all(v not in pr for pr in M)]
    if len(unp) == 2:
        M = M + [tuple(unp)]; unp = []
    if log: log('multi-swap V=%s matching=%s unpaired=%s scores=%s' % (V, M, unp, {k: round(v, 2) for k, v in sc.items() if v > 0.05}))
    if len(M) < 2:
        return None
    return M


def detect_swaps(frames, times, slots=None, log=None, max_gap=0.35, swap_dur=1.45, cam=None):
    if cam is None:
        from geom import PlaneCam
        cam = PlaneCam(_DEF_K, _DEF_P)
    dets_list = [detect_targets(f) for f in frames]
    if slots is None:
        slots = find_slots(dets_list)
    stat = []; offs = []
    for f, d in zip(frames, dets_list):
        st, off = slot_status(f, d, slots, cube_blobs(f))
        stat.append(st); offs.append(off)
    stat = np.array(stat); times = np.asarray(times, float)
    moving = np.array([len(o) > 0 for o in offs])
    idx = np.nonzero(moving)[0]
    intervals = []
    if len(idx):
        a = b = idx[0]
        for i in idx[1:]:
            if times[i] - times[b] <= max_gap:
                b = i
            else:
                intervals.append((a, b)); a = b = i
        intervals.append((a, b))
    events = []
    for (a, b) in intervals:
        dur = times[b] - times[a]
        if dur < 0.3:
            if log: log('skip short interval t=%.2f-%.2f' % (times[a], times[b]))
            continue
        nsw = max(1, int(round(dur / swap_dur)))
        if nsw == 1 and len(slots) >= 4:
            try:
                M = _multi_pairs(slots, dets_list, stat, a, b, cam, log=log)
            except Exception as ex:
                M = None
                if log: log('multi-swap check failed: %r' % ex)
            if M:
                tm = float(times[(a + b) // 2])
                fr = (stat[a:b + 1] == 0).mean(0)
                for pr in M:
                    events.append(dict(a=int(a), b=int(b), t=tm, pair=(int(pr[0]), int(pr[1])), frac=np.round(fr, 2).tolist(), ok=True, multi=True))
                if log: log('swaps(multi) %s t=%.2f-%.2f' % (M, times[a], times[b]))
                continue
        edges = np.linspace(times[a], times[b] + 1e-6, nsw + 1)
        for k in range(nsw):
            sel = [i for i in range(a, b + 1) if edges[k] <= times[i] <= edges[k + 1]]
            if not sel: continue
            p, fr = _pick_pair(stat[sel])
            ok = fr[p[1]] >= 0.4
            ev = dict(a=sel[0], b=sel[-1], t=float(times[sel[len(sel) // 2]]), pair=p, frac=np.round(fr, 2).tolist(), ok=bool(ok))
            if ok:
                events.append(ev)
            if log: log('swap%s %s t=%.2f-%.2f frac=%s %s' % ('' if nsw == 1 else '(%d/%d)' % (k + 1, nsw), p, times[sel[0]], times[sel[-1]], np.round(fr, 2).tolist(), '' if ok else 'REJECT'))
    return slots, events, stat, dets_list

def follow(slot, t0, events):
    s = slot
    for e in events:
        if e['t'] > t0:
            a, b = e['pair']
            if s == a: s = b
            elif s == b: s = a
    return s
