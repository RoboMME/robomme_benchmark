"""Perception for VideoPlaceOrder: target (bullseye) tracking through swaps and
cube placement-event detection in the conditioning video (front camera RGB)."""
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

def target_mask(a):
    a = a.astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    purple = (b > g + 8) & (r > g + 8) & (b > 80) & (r > 0.5 * b)
    cubes = cube_mask(a, 'blue') | cube_mask(a, 'red') | cube_mask(a, 'green')
    purple &= ~ndimage.binary_dilation(cubes, iterations=3)
    m = ndimage.binary_closing(purple, iterations=2)
    return m

def detect_targets(a, min_px=50, max_px=900):
    m = target_mask(a)
    m[:14, :] = False  # top strip: robot base/background
    grp, n = ndimage.label(ndimage.binary_dilation(m, iterations=3))
    lab = grp * m
    out = []
    if n == 0:
        return out
    idx = np.arange(1, n + 1)
    sizes = ndimage.sum(m, lab, idx)
    coms = ndimage.center_of_mass(m, lab, idx)
    sl = ndimage.find_objects(lab)
    for s, c, k in zip(sizes, coms, sl):
        if s < min_px or s > max_px:
            continue
        h = k[0].stop - k[0].start; w = k[1].stop - k[1].start
        if h > 40 or w > 50:
            continue
        out.append((c[1], c[0], s))
    return out

def cube_mask(a, color):
    a = a.astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    if color == 'red':
        return (r > 110) & (r > 2.2 * g) & (r > 2.2 * b)
    if color == 'green':
        return (g > 100) & (g > 1.6 * r) & (g > 1.6 * b)
    if color == 'blue':
        return (b > 90) & (b > 2.0 * r) & (b > 1.6 * g)
    raise ValueError(color)

def detect_cube(a, color, min_px=12):
    m = cube_mask(a, color)
    m[:10, :] = False
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    idx = np.arange(1, n + 1)
    sizes = ndimage.sum(m, lab, idx)
    k = int(np.argmax(sizes))
    if sizes[k] < min_px:
        return None
    c = ndimage.center_of_mass(m, lab, k + 1)
    return (c[1], c[0], float(sizes[k]))

def _sep_ok(pts, d=10):
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) < d:
                return False
    return True

def track_targets(frames_rgb, gate=28.0, slot_tol=7.0, log=None):
    """Track N identical targets through swaps. Slots (rest positions) come from
    the final (execution) frame. Returns tracks [T,N,2], N, dets."""
    dets = [detect_targets(a) for a in frames_rgb]
    T = len(frames_rgb)
    slots = np.array([[d[0], d[1]] for d in dets[-1]], float).reshape(-1, 2)
    counts = [len(d) for d in dets]
    vals, cnt = np.unique(counts, return_counts=True)
    if log: log(f"target count histogram {dict(zip(vals.tolist(), cnt.tolist()))}; final-frame N={len(slots)}")
    N = len(slots)
    if N == 0:
        raise RuntimeError("no targets detected in execution frame")
    tracks = np.full((T, N, 2), np.nan)
    # init: first frame whose detections all sit on slots and cover every slot
    start = None
    for t in range(T):
        D = np.array([[d[0], d[1]] for d in dets[t]], float).reshape(-1, 2)
        if len(D) == N:
            C = np.linalg.norm(slots[:, None] - D[None], axis=2)
            ri, ci = linear_sum_assignment(C)
            if C[ri, ci].max() < slot_tol:
                start = t; break
    if start is None:
        start = 0
    cur = slots.copy(); vel = np.zeros_like(cur)
    home = np.arange(N)
    tracks[:start + 1] = cur
    nres = 0
    for t in range(start + 1, T):
        D = np.array([[d[0], d[1]] for d in dets[t]], float).reshape(-1, 2)
        pred = cur + 0.5 * vel
        new = cur.copy(); matched = np.zeros(N, bool); used = np.zeros(len(D), bool)
        if len(D):
            C = np.linalg.norm(pred[:, None, :] - D[None, :, :], axis=2)
            ri, ci = linear_sum_assignment(C)
            for i, j in zip(ri, ci):
                if C[i, j] < gate:
                    new[i] = D[j]; matched[i] = True; used[j] = True
        vel = 0.5 * vel + 0.5 * (new - cur)
        vel[~matched] = 0
        cur = new
        # slot bookkeeping
        dslot = np.linalg.norm(cur[:, None] - slots[None], axis=2)
        near = dslot.min(axis=1) < slot_tol
        # lost tracks: unmatched and not resting at a slot
        lost = [i for i in range(N) if not matched[i] and not near[i]]
        if lost and len(D):
            free = []
            for j in range(len(D)):
                if used[j]:
                    continue
                ds = np.linalg.norm(slots - D[j], axis=1)
                if ds.min() < slot_tol:
                    free.append(j)
            if free:
                C = np.zeros((len(lost), len(free)))
                for a, i in enumerate(lost):
                    for b, j in enumerate(free):
                        s_idx = int(np.argmin(np.linalg.norm(slots - D[j], axis=1)))
                        C[a, b] = np.linalg.norm(cur[i] - D[j]) + (200.0 if s_idx == home[i] else 0.0)
                ri, ci = linear_sum_assignment(C)
                for a, b in zip(ri, ci):
                    i = lost[a]; j = free[b]
                    if log: log(f"  resolve t={t}: track {i} (home slot {home[i]}) lost at ({cur[i][0]:.0f},{cur[i][1]:.0f}) -> ({D[j][0]:.0f},{D[j][1]:.0f})")
                    cur[i] = D[j]; vel[i] = 0; used[j] = True; nres += 1
        dslot = np.linalg.norm(cur[:, None] - slots[None], axis=2)
        for i in range(N):
            k = int(np.argmin(dslot[i]))
            if dslot[i, k] < 3.5 and np.linalg.norm(vel[i]) < 1.0:
                home[i] = k
        tracks[t] = cur
    return tracks, N, dets

def placement_events(frames_rgb, times, color, tracks, dwell_s=1.0, still_px=2.5, near_px=16.0, log=None):
    """Return list of placement events [(t_index, track_id, (x,y))] in order."""
    cubes = [detect_cube(a, color) for a in frames_rgb]
    T = len(frames_rgb)
    segs = []
    i = 0
    while i < T:
        if cubes[i] is None:
            i += 1; continue
        j = i
        while j + 1 < T and cubes[j + 1] is not None and np.hypot(cubes[j + 1][0] - cubes[i][0], cubes[j + 1][1] - cubes[i][1]) < still_px:
            j += 1
        if times[j] - times[i] >= dwell_s:
            segs.append((i, j))
            i = j + 1
        else:
            i += 1
    events = []
    for (i, j) in segs:
        m = (i + j) // 2
        cx, cy = np.mean([cubes[k][0] for k in range(i, j + 1)]), np.mean([cubes[k][1] for k in range(i, j + 1)])
        # cube sits on target: target centre is slightly below cube centroid in image
        dists = np.hypot(tracks[m, :, 0] - cx, tracks[m, :, 1] - (cy + 2))
        k = int(np.argmin(dists))
        if log: log(f"dwell {i}-{j} t={times[i]:.1f}-{times[j]:.1f}s cube=({cx:.0f},{cy:.0f}) n={cubes[i][2]:.0f} nearest track {k} d={dists[k]:.1f}")
        if dists[k] > near_px:
            continue
        if events and events[-1][1] == k:
            # same target, check whether cube left in between
            pi = events[-1][3]
            left = False
            for q in range(pi, i):
                c = cubes[q]
                if c is None or np.hypot(c[0] - cx, c[1] - cy) > 15:
                    left = True; break
            if not left:
                events[-1] = (events[-1][0], k, events[-1][2], j)
                continue
        events.append((i, k, (cx, cy), j))
    return [(e[0], e[1], e[2]) for e in events], cubes

ORD = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6,
       'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10, '1st': 1, '2nd': 2, '3rd': 3,
       '4th': 4, '5th': 5, '6th': 6}

def parse_instruction(text):
    import re
    t = text.lower()
    color = None
    for c in ('red', 'green', 'blue'):
        if re.search(r'\b%s cube' % c, t):
            color = c; break
    if color is None:
        for c in ('red', 'green', 'blue'):
            if c in t:
                color = c; break
    k = None
    for w, v in ORD.items():
        if re.search(r'\b%s\b' % re.escape(w), t):
            k = v; break
    if k is None:
        m = re.search(r'(\d+)', t)
        k = int(m.group(1)) if m else 1
    return color, k

def solve_video(frames_rgb, times, instruction, log=None):
    color, k = parse_instruction(instruction)
    tracks, N, dets = track_targets(frames_rgb, log=log)
    events, cubes = placement_events(frames_rgb, times, color, tracks, log=log)
    if log: log(f"color={color} k={k} N={N} events={[(e[0], e[1]) for e in events]}")
    if not events:
        return None
    idx = min(k, len(events)) - 1
    tid = events[idx][1]
    final = tracks[-1, tid]
    return dict(color=color, k=k, track=tid, final_px=(float(final[0]), float(final[1])),
                events=events, N=N, final_tracks=tracks[-1].tolist())

# ---------------- slot/swap model (v2) ----------------
def slot_swap_solve(frames_rgb, times, instruction, slot_tol=6.0, mid_tol=9.0, min_ev=3, min_dur=0.5, log=None):
    color, k = parse_instruction(instruction)
    dets = [detect_targets(a) for a in frames_rgb]
    slots = np.array([[d[0], d[1]] for d in dets[-1]], float).reshape(-1, 2)
    N = len(slots)
    T = len(frames_rgb)
    pairs = [(a, b) for a in range(N) for b in range(a + 1, N)]
    mids = {p: (slots[p[0]] + slots[p[1]]) / 2 for p in pairs}
    lens = {p: np.linalg.norm(slots[p[0]] - slots[p[1]]) for p in pairs}
    evidence = []  # (t, pair, err)
    for t in range(T):
        D = np.array([[d[0], d[1]] for d in dets[t]], float).reshape(-1, 2)
        if len(D) < 2: continue
        dsm = np.linalg.norm(D[:, None] - slots[None], axis=2)
        ds = dsm.min(axis=1)
        occupied = (dsm < slot_tol).any(axis=0)
        tr = D[ds > slot_tol]
        if len(tr) < 2: continue
        cands = []
        for i in range(len(tr)):
            for j in range(i + 1, len(tr)):
                m = (tr[i] + tr[j]) / 2; L = np.linalg.norm(tr[i] - tr[j])
                for p in pairs:
                    if occupied[p[0]] or occupied[p[1]]: continue
                    if min(np.linalg.norm(tr[i] - slots[p[0]]), np.linalg.norm(tr[i] - slots[p[1]])) < 0.2 * lens[p]: continue
                    e = np.linalg.norm(m - mids[p])
                    if e < mid_tol and 0.7 < L / max(lens[p], 1e-6) < 1.3:
                        cands.append((e + 10 * abs(1 - L / lens[p]), i, j, p))
        cands.sort()
        usedd = set(); useds = set()
        for e, i, j, p in cands:
            if i in usedd or j in usedd or p[0] in useds or p[1] in useds: continue
            usedd |= {i, j}; useds |= {p[0], p[1]}
            evidence.append((t, p, e))
    # group evidence into swap events
    events = []
    by_pair = {}
    for t, p, e in evidence:
        by_pair.setdefault(p, []).append(t)
    for p, ts in by_pair.items():
        ts = sorted(ts); grp = [ts[0]]
        for t in ts[1:]:
            if times[t] - times[grp[-1]] <= 1.0:
                grp.append(t)
            else:
                events.append((grp, p)); grp = [t]
        events.append((grp, p))
    swaps = []
    for grp, p in events:
        if len(grp) >= min_ev and times[grp[-1]] - times[grp[0]] >= min_dur:
            swaps.append(((grp[0] + grp[-1]) / 2.0, p, grp[0], grp[-1]))
    swaps.sort()
    if log: log('swaps', [(round(times[int(s[2])], 1), round(times[int(s[3])], 1), s[1]) for s in swaps])
    # placement dwell detection
    cubes = [detect_cube(a, color) for a in frames_rgb]
    segs = []; i = 0
    while i < T - 1:  # exclude final execution frame
        if cubes[i] is None: i += 1; continue
        j = i
        while j + 1 < T - 1 and cubes[j + 1] is not None and np.hypot(cubes[j+1][0]-cubes[i][0], cubes[j+1][1]-cubes[i][1]) < 2.5:
            j += 1
        if times[j] - times[i] >= 0.8:
            segs.append((i, j)); i = j + 1
        else:
            i += 1
    # permutation: ident_at_slot[s] = identity (initial slot index)
    ident = list(range(N))
    si = 0
    placements = []
    def apply_until(tidx):
        nonlocal si
        while si < len(swaps) and swaps[si][0] < tidx:
            a, b = swaps[si][1]; ident[a], ident[b] = ident[b], ident[a]; si += 1
    for (i, j) in segs:
        mid = (i + j) / 2.0
        apply_until(mid)
        cx = np.mean([cubes[q][0] for q in range(i, j + 1)]); cy = np.mean([cubes[q][1] for q in range(i, j + 1)])
        dd = np.hypot(slots[:, 0] - cx, slots[:, 1] - (cy + 2)); s = int(np.argmin(dd))
        if log: log(f"dwell t={times[i]:.1f}-{times[j]:.1f} cube=({cx:.0f},{cy:.0f}) slot {s} d={dd[s]:.1f} ident {ident[s]}")
        if dd[s] > 14: continue
        if placements and placements[-1][1] == ident[s]:
            # merge if the cube did not leave
            left = any(cubes[q] is None or np.hypot(cubes[q][0]-cx, cubes[q][1]-cy) > 15 for q in range(placements[-1][3], i))
            if not left:
                placements[-1] = (placements[-1][0], ident[s], s, j); continue
        placements.append((i, ident[s], s, j))
    apply_until(T + 1)
    final_slot_of = {ident[s]: s for s in range(N)}
    if log: log(f"color={color} k={k} N={N} placements(ident)={[p[1] for p in placements]} slots={[p[2] for p in placements]}")
    if not placements: return None
    idx = min(k, len(placements)) - 1
    tid = placements[idx][1]
    fs = final_slot_of[tid]
    order = [fs]
    # fallbacks: final slots of neighbouring placements, then the rest
    for d in (1, -1, 2, -2, 3, -3):
        q = idx + d
        if 0 <= q < len(placements):
            s2 = final_slot_of[placements[q][1]]
            if s2 not in order: order.append(s2)
    for s in range(N):
        if s not in order: order.append(s)
    return dict(color=color, k=k, final_px=tuple(slots[fs]), order_px=[tuple(slots[s]) for s in order],
                placements=[(p[0], p[1], p[2]) for p in placements], N=N, nswaps=len(swaps))

# ---------------- continuous tracker with coasting + rest snapping (v3) ----------------
NMISS = 1
VK = 1.0
GBASE = 22.0
GSLOT = 5.0
def track_v3(frames_rgb, slot_tol=None, log=None):
    slot_tol = GSLOT if slot_tol is None else slot_tol
    dets = [detect_targets(a) for a in frames_rgb]
    slots = np.array([[d[0], d[1]] for d in dets[-1]], float).reshape(-1, 2)
    N = len(slots); T = len(frames_rgb)
    cur = slots.copy(); vel = np.zeros_like(cur); missed = np.zeros(N, int)
    tracks = np.zeros((T, N, 2))
    # find first full rest frame
    def rest_match(D):
        if len(D) != N: return None
        C = np.linalg.norm(slots[:, None] - D[None], axis=2)
        ri, ci = linear_sum_assignment(C)
        if C[ri, ci].max() < slot_tol: return True
        return None
    start = 0
    for t in range(T):
        D = np.array([[d[0], d[1]] for d in dets[t]], float).reshape(-1, 2)
        if rest_match(D): start = t; break
    tracks[:start + 1] = cur
    for t in range(start + 1, T):
        D = np.array([[d[0], d[1]] for d in dets[t]], float).reshape(-1, 2)
        pred = cur + VK * vel
        new = pred.copy(); matched = np.zeros(N, bool)
        if len(D):
            C = np.linalg.norm(pred[:, None, :] - D[None, :, :], axis=2)
            gate = GBASE + 8.0 * np.minimum(missed, 4)
            ri, ci = linear_sum_assignment(C)
            for i, j in zip(ri, ci):
                if C[i, j] < gate[i]:
                    new[i] = D[j]; matched[i] = True
        v_new = new - cur
        vel = np.where(matched[:, None], 0.6 * v_new + 0.4 * vel, 0.85 * vel)
        missed = np.where(matched, 0, missed + 1)
        cur = new
        partial_rest = False
        if len(D) >= max(1, N - NMISS):
            dsm = np.linalg.norm(D[:, None] - slots[None], axis=2)
            partial_rest = bool((dsm.min(axis=1) < slot_tol).all())
        if partial_rest:
            C = np.linalg.norm(cur[:, None] - slots[None], axis=2)
            ri, ci = linear_sum_assignment(C)
            if C[ri, ci].max() > slot_tol and log:
                log(f"  snap t={t}: " + ', '.join(f"{i}->{j} ({C[i, j]:.0f}px)" for i, j in zip(ri, ci) if C[i, j] > slot_tol))
            cur = slots[ci[np.argsort(ri)]].copy(); vel[:] = 0; missed[:] = 0
        tracks[t] = cur
    return tracks, slots

def solve_v3(frames_rgb, times, instruction, log=None):
    color, k = parse_instruction(instruction)
    tracks, slots = track_v3(frames_rgb, log=log)
    N = len(slots); T = len(frames_rgb)
    cubes = [detect_cube(a, color) for a in frames_rgb]
    segs = []; i = 0
    while i < T - 1:
        if cubes[i] is None: i += 1; continue
        j = i
        while j + 1 < T - 1 and cubes[j + 1] is not None and np.hypot(cubes[j+1][0]-cubes[i][0], cubes[j+1][1]-cubes[i][1]) < 2.5:
            j += 1
        if times[j] - times[i] >= 0.8:
            segs.append((i, j)); i = j + 1
        else:
            i += 1
    placements = []
    for (i, j) in segs:
        m = (i + j) // 2
        cx = np.mean([cubes[q][0] for q in range(i, j + 1)]); cy = np.mean([cubes[q][1] for q in range(i, j + 1)])
        dd = np.hypot(tracks[m, :, 0] - cx, tracks[m, :, 1] - (cy + 2)); tid = int(np.argmin(dd))
        if log: log(f"dwell t={times[i]:.1f}-{times[j]:.1f} cube=({cx:.0f},{cy:.0f}) track {tid} d={dd[tid]:.1f}")
        if dd[tid] > 14: continue
        if placements and placements[-1][1] == tid:
            left = any(cubes[q] is None or np.hypot(cubes[q][0]-cx, cubes[q][1]-cy) > 15 for q in range(placements[-1][2], i))
            if not left:
                placements[-1] = (placements[-1][0], tid, j); continue
        placements.append((i, tid, j))
    if log: log(f"color={color} k={k} N={N} placements={[p[1] for p in placements]}")
    if not placements: return None
    final = tracks[-1]
    idx = min(k, len(placements)) - 1
    order = [placements[idx][1]]
    for d in (1, -1, 2, -2):
        q = idx + d
        if 0 <= q < len(placements) and placements[q][1] not in order: order.append(placements[q][1])
    for s in range(N):
        if s not in order: order.append(s)
    return dict(color=color, k=k, final_px=tuple(final[order[0]]), order_px=[tuple(final[s]) for s in order],
                placements=placements, N=N)
