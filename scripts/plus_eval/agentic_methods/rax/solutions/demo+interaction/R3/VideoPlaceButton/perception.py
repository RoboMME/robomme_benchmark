"""Offline-capable perception for VideoPlaceButton: target tracking through swaps,
button-press detection, cube placement detection. Inputs: lists of HxWx3 uint8 front images + timestamps."""
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

def target_mask(a):
    a = a.astype(int); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    purple = (b > g + 12) & (r > g + 35) & (r > 130)
    pinkw = (r > 230) & (b > g + 6) & (g > 170)
    return purple, pinkw

def detect_targets(a, min_px=25):
    purple, pinkw = target_mask(a)
    mk = ndimage.binary_closing(purple | pinkw, iterations=1)
    lab, n = ndimage.label(mk)
    out = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        npur = int(purple[ys, xs].sum())
        if len(xs) >= min_px and npur >= 8:
            out.append((float(xs.mean()), float(ys.mean()), len(xs)))
    return out

CUBE_COLORS = ("red", "green", "blue")
def cube_mask(a, color):
    a = a.astype(int); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    if color == "red":
        return (r > 120) & (g < 60) & (b < 60)
    if color == "blue":
        return (b > 100) & (r < 60) & (g < 80)
    return (g > 110) & (r < 90) & (b < 90)

def cube_pos(a, color, min_px=15):
    mk = cube_mask(a, color)
    lab, n = ndimage.label(mk)
    if n == 0: return None
    sizes = ndimage.sum(mk, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    if sizes[k - 1] < min_px: return None
    ys, xs = np.nonzero(lab == k)
    return (float(xs.mean()), float(ys.mean()), int(len(xs)))

def instruction_color(instr):
    s = instr.lower()
    for c in CUBE_COLORS:
        if c + " cube" in s: return c
    for c in CUBE_COLORS:
        if c in s: return c
    return "red"

def detect_button(a):
    """White square with grey disk. Return (x,y) centre in front image."""
    a = a.astype(int); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    white = (r > 200) & (g > 200) & (b > 200) & (np.abs(r - b) < 25) & (np.abs(r - g) < 25)
    white[:40, :] = False  # exclude robot base area at the top
    lab, n = ndimage.label(white)
    best = None
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        if 40 <= len(xs) <= 1500:
            w = xs.max() - xs.min() + 1; h = ys.max() - ys.min() + 1
            fill = len(xs) / (w * h)
            score = len(xs) * (1 if 0.6 < w / h < 1.7 else 0.1)
            if best is None or score > best[0]:
                best = (score, (float(xs.mean()), float(ys.mean())), (w, h))
    return None if best is None else best[1]

def track_targets(frames, times, init=None, gate=30.0):
    """Track targets through the sequence. Returns tracks: array [T, K, 2] of positions,
    and K initial positions."""
    dets0 = detect_targets(frames[0])
    if init is None:
        init = [(x, y) for x, y, n in dets0]
    K = len(init)
    pos = np.array(init, float); vel = np.zeros_like(pos); miss = np.zeros(K, int)
    out = [pos.copy()]
    for t in range(1, len(frames)):
        dt = max(1e-3, times[t] - times[t - 1])
        dets = detect_targets(frames[t])
        pred = pos + vel * dt * (miss < 3)[:, None]
        if dets:
            D = np.array([(x, y) for x, y, n in dets])
            C = np.linalg.norm(pred[:, None, :] - D[None, :, :], axis=2)
            ri, ci = linear_sum_assignment(C)
            matched = set()
            for i, j in zip(ri, ci):
                if C[i, j] < gate:
                    newp = D[j]
                    v = (newp - pos[i]) / dt
                    vel[i] = 0.5 * vel[i] + 0.5 * v if miss[i] == 0 else v * 0.5
                    pos[i] = newp; miss[i] = 0; matched.add(i)
            for i in range(K):
                if i not in matched:
                    pos[i] = pred[i] if miss[i] < 3 else pos[i]; miss[i] += 1
        else:
            pos = pred; miss += 1
        out.append(pos.copy())
    return np.array(out), np.array(init)

def button_activity(frames, btn, r=14):
    """Fraction of pixels near the button that changed vs frame 0 AND look like the grey/white arm."""
    bx, by = int(round(btn[0])), int(round(btn[1]))
    y0, y1, x0, x1 = max(0, by - r), by + r, max(0, bx - r), bx + r
    f0 = frames[0][y0:y1, x0:x1].astype(int)
    out = []
    for f in frames:
        a = f[y0:y1, x0:x1].astype(int)
        chg = np.abs(a - f0).sum(axis=2) > 60
        sat = a.max(axis=2) - a.min(axis=2)
        grey = (sat < 30) | ((a.max(axis=2) < 70))
        out.append(100.0 * float((chg & grey).mean()))
    return np.array(out)

def find_press(frames, times, btn, thr=None):
    act = button_activity(frames, btn)
    if thr is None:
        thr = max(8.0, 0.5 * float(act.max()))
    idx = np.nonzero(act > thr)[0]
    if len(idx) == 0:
        return None, act
    # pick the longest contiguous run (the press), return its end time index
    runs = []; s = idx[0]; p = idx[0]
    for i in idx[1:]:
        if times[i] - times[p] > 0.6:
            runs.append((s, p)); s = i
        p = i
    runs.append((s, p))
    best = max(runs, key=lambda sp: (act[sp[0]:sp[1] + 1].max(), times[sp[1]] - times[sp[0]]))
    return best, act

def find_placements(frames, times, color, tracks, still_s=0.8, still_px=1.5, near_px=12.0):
    """Return list of (t_index_start, target_id, dist) for cube rest events on targets."""
    cp = [cube_pos(f, color) for f in frames]
    events = []
    T = len(frames); i = 0
    while i < T:
        if cp[i] is None: i += 1; continue
        j = i
        while j + 1 < T and cp[j + 1] is not None and np.hypot(cp[j + 1][0] - cp[i][0], cp[j + 1][1] - cp[i][1]) < still_px * 2:
            j += 1
        if times[j] - times[i] >= still_s:
            # cube rests; check nearest target
            k = min(j, i + max(1, (j - i) // 2))
            p = np.array(cp[k][:2]); p[1] += 3.0  # cube centre sits slightly above target centre
            d = np.linalg.norm(tracks[k] - p, axis=1)
            tid = int(np.argmin(d))
            events.append((i, j, tid, float(d[tid]), tuple(np.round(cp[k][:2], 1))))
        i = j + 1
    return events, cp

def analyse(frames, times, instruction, log=print):
    color = instruction_color(instruction)
    btn = detect_button(frames[0])
    tracks, init = track_targets(frames, times)
    run, act = find_press(frames, times, btn)
    events, cp = find_placements(frames, times, color, tracks)
    log(f"color={color} button={btn} n_targets={len(init)} press_run={run}")
    for e in events: log(f"rest i={e[0]}-{e[1]} t={times[e[0]]:.2f}-{times[e[1]]:.2f} tid={e[2]} d={e[3]:.1f} cube={e[4]}")
    home = cp[0][:2] if cp[0] else None
    chosen = None
    if run is not None:
        tpress = run[1]
        prev = None
        for e in events:
            moved = prev is None or np.hypot(e[4][0] - prev[0], e[4][1] - prev[1]) > 8.0
            if e[0] > tpress and e[3] < 12.0 and moved:
                chosen = e; break
            prev = e[4]
    if chosen is None:
        cands = [e for e in events if e[3] < 12.0]
        chosen = cands[0] if cands else None
    if chosen is None:
        return dict(color=color, ok=False)
    tid = chosen[2]
    final = tracks[-1][tid]
    log("final tracks " + str(np.round(tracks[-1]).tolist()))
    log(f"chosen tid={tid} at rest {chosen[4]} final_pos={final}")
    return dict(color=color, ok=True, tid=tid, final_px=(float(final[0]), float(final[1])),
                button=btn, events=events, press=run, init=init.tolist(), final_tracks=tracks[-1].tolist())


def track_targets_v2(frames, times, claim_px=4.5, gate_px=18.0, spawn_px=22.0, snap_px=5.0, coast_s=1.2, log=None):
    """Slot-aware tracker. Targets rest on slots (positions in frame 0) and occasionally move.
    Returns per-frame list of positions [T,K,2], per-frame slot index of each target (-1 moving), slots."""
    slots = np.array([(x, y) for x, y, _ in detect_targets(frames[0])], float)
    K = len(slots)
    pos = slots.copy(); vel = np.zeros_like(pos)
    slot_of = np.arange(K)            # current slot for static targets, -1 for moving
    last_seen = np.zeros(K)
    hist_pos = [pos.copy()]; hist_slot = [slot_of.copy()]
    for t in range(1, len(frames)):
        now = times[t]; dt = max(1e-3, times[t] - times[t - 1])
        D = np.array([(x, y) for x, y, _ in detect_targets(frames[t])], float).reshape(-1, 2)
        used = np.zeros(len(D), bool)
        # 1. static claims
        claimed = np.zeros(K, bool)
        for k in range(K):
            if slot_of[k] >= 0 and len(D):
                dd = np.hypot(D[:, 0] - slots[slot_of[k], 0], D[:, 1] - slots[slot_of[k], 1])
                j = int(np.argmin(dd))
                if dd[j] < claim_px and not used[j]:
                    used[j] = True; claimed[k] = True; last_seen[k] = now
        # 2. movers
        movers = [k for k in range(K) if slot_of[k] < 0]
        free = np.nonzero(~used)[0]
        if movers and len(free):
            pred = np.array([pos[k] + vel[k] * dt for k in movers])
            C = np.linalg.norm(pred[:, None, :] - D[free][None, :, :], axis=2)
            ri, ci = linear_sum_assignment(C)
            matched = set()
            for a, b in zip(ri, ci):
                g = gate_px * max(1.0, dt / 0.1)
                if C[a, b] < g:
                    k = movers[a]; j = free[b]
                    v = (D[j] - pos[k]) / dt
                    vel[k] = 0.6 * v + 0.4 * vel[k]
                    pos[k] = D[j]; used[j] = True; matched.add(k); last_seen[k] = now
            for k in movers:
                if k not in matched:
                    if now - last_seen[k] < coast_s:
                        pos[k] = pos[k] + vel[k] * dt
        elif movers:
            for k in movers:
                if now - last_seen[k] < coast_s:
                    pos[k] = pos[k] + vel[k] * dt
        # 3. spawn movers from unclaimed static targets
        free = [j for j in np.nonzero(~used)[0]]
        cand = [k for k in range(K) if slot_of[k] >= 0 and not claimed[k]]
        if free and cand:
            C = np.array([[np.hypot(D[j, 0] - slots[slot_of[k], 0], D[j, 1] - slots[slot_of[k], 1]) for j in free] for k in cand])
            ri, ci = linear_sum_assignment(C)
            for a, b in zip(ri, ci):
                if C[a, b] < spawn_px:
                    k = cand[a]; j = free[b]
                    vel[k] = (D[j] - slots[slot_of[k]]) / dt * 0.5
                    pos[k] = D[j]; slot_of[k] = -1; used[j] = True; last_seen[k] = now
                    if log: log(f"t={now:.2f} target {k} departs")
        # 4. snap movers to free slots
        occupied = set(int(s) for s in slot_of if s >= 0)
        for k in range(K):
            if slot_of[k] < 0:
                dd = np.hypot(slots[:, 0] - pos[k, 0], slots[:, 1] - pos[k, 1])
                order = np.argsort(dd)
                for s in order:
                    if int(s) in occupied: continue
                    if dd[s] < snap_px and np.linalg.norm(vel[k]) < 60.0:
                        slot_of[k] = int(s); pos[k] = slots[s]; vel[k] = 0; occupied.add(int(s))
                        if log: log(f"t={now:.2f} target {k} arrives slot {s}")
                    break
        hist_pos.append(pos.copy()); hist_slot.append(slot_of.copy())
    # finalize: movers snap to nearest free slots
    occupied = set(int(s) for s in slot_of if s >= 0)
    for k in range(K):
        if slot_of[k] < 0:
            dd = np.hypot(slots[:, 0] - pos[k, 0], slots[:, 1] - pos[k, 1])
            for s in np.argsort(dd):
                if int(s) not in occupied:
                    slot_of[k] = int(s); occupied.add(int(s)); break
    hist_slot[-1] = slot_of.copy()
    return np.array(hist_pos), np.array(hist_slot), slots


def analyse_v2(frames, times, instruction, log=print):
    color = instruction_color(instruction)
    btn = detect_button(frames[0])
    hp, hs, slots = track_targets_v2(frames, times)
    run, act = find_press(frames, times, btn) if btn is not None else (None, None)
    cps = [cube_pos(f, color) for f in frames]
    # rest events of the cube
    events = []
    T = len(frames); i = 0
    while i < T:
        if cps[i] is None: i += 1; continue
        j = i
        while j + 1 < T and cps[j + 1] is not None and np.hypot(cps[j + 1][0] - cps[i][0], cps[j + 1][1] - cps[i][1]) < 3.0:
            j += 1
        if times[j] - times[i] >= 0.6:
            p = np.array(cps[i][:2]) + np.array([0.0, 3.0])
            ds = np.hypot(slots[:, 0] - p[0], slots[:, 1] - p[1]); s = int(np.argmin(ds))
            k = (i + j) // 2
            occupant = [q for q in range(len(slots)) if hs[k][q] == s]
            events.append(dict(i=i, j=j, t0=times[i], t1=times[j], slot=s, d=float(ds[s]), cube=cps[i][:2],
                               occupant=occupant[0] if occupant else None))
        i = j + 1
    log(f"color={color} button={btn} slots={np.round(slots).astype(int).tolist()} press={run} "
        f"press_t={(times[run[0]], times[run[1]]) if run else None}")
    for e in events:
        log(f"rest t={e['t0']:.2f}-{e['t1']:.2f} slot={e['slot']} d={e['d']:.1f} cube=({e['cube'][0]:.0f},{e['cube'][1]:.0f}) occ={e['occupant']}")
    chosen = None; prev = None
    tpress = times[run[0]] if run else -1
    for e in events:
        moved = prev is None or np.hypot(e['cube'][0] - prev[0], e['cube'][1] - prev[1]) > 8.0
        if e['t0'] > tpress and e['d'] < 12.0 and moved and e['occupant'] is not None:
            chosen = e; break
        prev = e['cube']
    if chosen is None:
        return dict(ok=False, color=color, slots=slots.tolist())
    tid = chosen['occupant']; fslot = int(hs[-1][tid])
    log(f"chosen target id {tid} (placed on slot {chosen['slot']} at t={chosen['t0']:.2f}); final slot {fslot} px={slots[fslot]}")
    return dict(ok=True, color=color, tid=tid, final_slot=fslot, final_px=slots[fslot].tolist(),
                slots=slots.tolist(), button=btn, press=None if run is None else [times[run[0]], times[run[1]]])


def moving_detections(frames, slots, far_px=7.0):
    out = []
    for f in frames:
        D = detect_targets(f)
        M = []
        for x, y, n in D:
            dd = np.hypot(slots[:, 0] - x, slots[:, 1] - y)
            if dd.min() > far_px:
                M.append((x, y))
        out.append(np.array(M, float).reshape(-1, 2))
    return out

SECOND_LOG = []
SECOND_SYM_FRAC = 0.2

def find_swaps(frames, times, slots, mid_tol=7.0, gap_s=0.45, log=None):
    """Detect pair swaps: returns list of (t_start, t_end, i, j)."""
    M = moving_detections(frames, slots)
    K = len(slots)
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    mids = {p: (slots[p[0]] + slots[p[1]]) / 2 for p in pairs}
    half = {p: np.linalg.norm(slots[p[0]] - slots[p[1]]) / 2 for p in pairs}
    # events
    dtm = float(np.median(np.diff(times))) if len(times) > 1 else 0.1
    gap_s = max(gap_s, 1.6 * dtm)
    min_score = float(np.clip(0.25 / max(dtm, 1e-3), 1.0, 2.0))
    idx = [t for t in range(len(frames)) if len(M[t])]
    events = []
    if idx:
        s = idx[0]; p = idx[0]
        for t in idx[1:]:
            if times[t] - times[p] > gap_s:
                events.append((s, p)); s = t
            p = t
        events.append((s, p))
    swaps = []
    def score_pairs(ptsets, a, b):
        score = {p: 0.0 for p in pairs}; sym = {p: 0 for p in pairs}; frames_of = {p: [] for p in pairs}
        used = {p: {} for p in pairs}
        for t in range(a, b + 1):
            pts = ptsets[t]
            for p in pairs:
                m = mids[p]; r = half[p]
                best = 0.0; bu = []
                for u in range(len(pts)):
                    du = np.linalg.norm(pts[u] - m)
                    if not (0.45 * r < du < 1.35 * r + 4):
                        continue
                    if best < 0.3: best = 0.3; bu = [u]
                    for v in range(u + 1, len(pts)):
                        mm = (pts[u] + pts[v]) / 2
                        dv = np.linalg.norm(pts[v] - m)
                        if np.linalg.norm(mm - m) < mid_tol and 0.45 * r < dv < 1.35 * r + 4:
                            best = 1.0; bu = [u, v]
                if best > 0:
                    score[p] += best; frames_of[p].append(t); used[p][t] = bu
                    if best == 1.0: sym[p] += 1
        return score, sym, frames_of, used
    for (a, b) in events:
        ptsets = {t: M[t] for t in range(a, b + 1)}
        score, sym, frames_of, used = score_pairs(ptsets, a, b)
        ranked = sorted(pairs, key=lambda p: -score[p])
        chosen = []
        p1 = ranked[0]
        if score[p1] >= min_score:
            chosen.append(p1)
            # remove explained points and look for a simultaneous second swap
            rem = {t: np.delete(ptsets[t], used[p1].get(t, []), axis=0) for t in ptsets}
            score2, sym2, frames2, _ = score_pairs(rem, a, b)
            cand = [p for p in pairs if not (set(p) & set(p1))]
            if cand:
                p2 = max(cand, key=lambda p: (sym2[p], score2[p]))
                nfr = b - a + 1
                SECOND_LOG.append((nfr, sym2[p2], round(score2[p2], 1)))
                if sym2[p2] >= max(2 if dtm < 0.3 else 1, SECOND_SYM_FRAC * nfr) and score2[p2] >= min_score:
                    chosen.append(p2); frames_of[p2] = frames2[p2]
        if log:
            log(f"event t={times[a]:.2f}-{times[b]:.2f} n={b-a+1} top={[(q, round(score[q],1), sym[q]) for q in ranked[:4]]} chosen={chosen}")
        for p in chosen:
            fo = frames_of[p]
            swaps.append((times[fo[0]], times[fo[-1]], p[0], p[1]))
    return swaps

def identity_after(slot, t_from, swaps):
    """Follow the target that occupies `slot` at time t_from through subsequent swaps."""
    s = slot
    for (t0, t1, i, j) in sorted(swaps, key=lambda w: w[0]):
        if t1 <= t_from:  # completed before
            continue
        if t0 < t_from <= t1:
            # swap in progress at placement time: treat target as already moving; ambiguous
            pass
        if s == i: s = j
        elif s == j: s = i
    return s

def cube_rests(frames, times, color, slots):
    cps = [cube_pos(f, color) for f in frames]
    events = []
    T = len(frames); i = 0
    while i < T:
        if cps[i] is None: i += 1; continue
        j = i
        while j + 1 < T and cps[j + 1] is not None and np.hypot(cps[j + 1][0] - cps[i][0], cps[j + 1][1] - cps[i][1]) < 3.0:
            j += 1
        if times[j] - times[i] >= 1.0:
            p = np.array(cps[i][:2]) + np.array([0.0, 3.0])
            ds = np.hypot(slots[:, 0] - p[0], slots[:, 1] - p[1]); s = int(np.argmin(ds))
            events.append(dict(i=i, j=j, t0=times[i], t1=times[j], slot=s, d=float(ds[s]), cube=cps[i][:2]))
        i = j + 1
    return events, cps

PLACE_D = 7.5

def analyse_v3(frames, times, instruction, log=print, live=None):
    color = instruction_color(instruction)
    slots = np.array([(x, y) for x, y, _ in detect_targets(frames[0])], float)
    if live is not None:
        ls = np.array([(x, y) for x, y, _ in detect_targets(live)], float).reshape(-1, 2)
        log(f"slots frame0={len(slots)} live={len(ls)}")
        if len(ls) >= len(slots):
            slots = ls
    btn = detect_button(frames[0])
    run, act = find_press(frames, times, btn) if btn is not None else (None, None)
    swaps = find_swaps(frames, times, slots, log=log)
    rests, cps = cube_rests(frames, times, color, slots)
    log(f"color={color} button={btn} slots={np.round(slots).astype(int).tolist()} "
        f"press_t={(times[run[0]], times[run[1]]) if run else None}")
    log("swaps " + str([(round(a, 2), round(b, 2), i, j) for a, b, i, j in swaps]))
    for e in rests:
        log(f"rest t={e['t0']:.2f}-{e['t1']:.2f} slot={e['slot']} d={e['d']:.1f} cube=({e['cube'][0]:.0f},{e['cube'][1]:.0f})")
    tpress = times[run[0]] if run else -1
    chosen = None; prev = None
    for e in rests:
        if prev is None:
            moved = True
        else:
            seg = [c for c in cps[prev['j'] + 1:e['i']] if c is not None]
            moved = np.hypot(e['cube'][0] - prev['cube'][0], e['cube'][1] - prev['cube'][1]) > 8.0 or \
                any(np.hypot(c[0] - e['cube'][0], c[1] - e['cube'][1]) > 8.0 for c in seg)
        if e['t0'] > tpress and e['d'] < PLACE_D and moved:
            chosen = e; break
        prev = e
    if chosen is None:
        return dict(ok=False, color=color, slots=slots.tolist())
    tplace = (chosen['t0'] + chosen['t1']) / 2 if False else chosen['t0']
    fslot = identity_after(chosen['slot'], tplace, swaps)
    log(f"chosen placement slot {chosen['slot']} at t={chosen['t0']:.2f}; final slot {fslot} px={slots[fslot]}")
    return dict(ok=True, color=color, place_slot=chosen['slot'], t_place=chosen['t0'], final_slot=int(fslot),
                final_px=slots[fslot].tolist(), slots=slots.tolist(), button=btn, swaps=swaps,
                press=None if run is None else [times[run[0]], times[run[1]]])


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def backproject(depth_m, K, pose, us, vs):
    """OpenGL convention as documented. depth_m: HxW metres. Returns Nx3 world points (nan if invalid)."""
    us = np.asarray(us, float); vs = np.asarray(vs, float)
    d = depth_m[np.clip(vs.round().astype(int), 0, depth_m.shape[0] - 1), np.clip(us.round().astype(int), 0, depth_m.shape[1] - 1)]
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    cam = np.stack([(us - cx) * d / fx, -(vs - cy) * d / fy, -d], axis=1)
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"], float)
    pw = cam @ R.T + t
    pw[d <= 0] = np.nan
    return pw

def cube_top_center(rgb, depth_m, K, pose, color):
    mk = cube_mask(rgb, color)
    lab, n = ndimage.label(mk)
    if n == 0: return None, None
    sizes = ndimage.sum(mk, lab, range(1, n + 1)); k = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(lab == k)
    pw = backproject(depth_m, K, pose, xs, ys)
    ok = np.isfinite(pw).all(axis=1); pw = pw[ok]
    if len(pw) < 5: return None, None
    ztop = np.percentile(pw[:, 2], 90)
    top = pw[pw[:, 2] > ztop - 0.006]
    c = np.median(top, axis=0)
    return c, dict(ztop=float(ztop), n_top=int(len(top)), n=int(len(pw)), zmin=float(np.percentile(pw[:, 2], 5)))
