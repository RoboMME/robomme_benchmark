"""Conditioning-video analysis for VideoPlaceButton (pure, offline-testable).

Rule (confirmed on all 3 released demos): the goal target is the target OBJECT on
which the instructed-colour cube was first released after the button press, followed
through every later pairwise swap of the (identical-looking) targets until the end of
the video.  Swaps: two targets rotate about their midpoint and exchange slots
(~1.35 s); sometimes two swaps run simultaneously.

analyse(frames, color) -> dict with 'slot_xy' (front pixel of the goal slot at the
end of the video), 'slots', 'log', 'confidence'.
frames: list of (time_s, front_rgb uint8 HxWx3, wrist_rgb or None).
"""
import itertools
import numpy as np
from scipy import ndimage
import perception as P

SLOT_AT_PX = 6.0           # blob centroid within this of a slot centre = stationary at that slot
FILLED_FRAC = 0.5          # footprint target|white fraction -> filled
VACANT_WOOD = 0.7          # footprint wood fraction -> vacant
SWAP_START_TOL_S = 0.45    # simultaneous swaps start within this (sampling stride up to ~0.25 s)
RELEASE_SLOT_MAX_PX = 14.0
CUBE_Y_OFFSET = 4.0        # cube centroid appears ~4 px above the target centre in front view


def find_slots(fronts):
    acc = np.zeros(fronts[0].shape[:2])
    for f in fronts:
        acc += P.target_mask(f)
    acc /= max(len(fronts), 1)
    comps = P.components(acc > P.SLOT_FREQ, P.SLOT_MIN_PIXELS)
    # merge concentric pieces (inner/outer rings) whose centres coincide
    merged = []
    for x, y, c, reg in sorted(comps, key=lambda t: -t[2]):
        for m in merged:
            if np.hypot(m[0] - x, m[1] - y) < 4:
                m[3] |= reg
                break
        else:
            merged.append([x, y, c, reg.copy()])
    slots = []
    for x, y, c, reg in merged:
        reg = ndimage.binary_fill_holes(reg)
        ys, xs = np.nonzero(reg)
        slots.append({"xy": np.array([xs.mean(), ys.mean()]), "foot": reg, "n": int(reg.sum())})
    slots.sort(key=lambda s: (s["xy"][1], s["xy"][0]))
    return slots


def slot_states(front, slots):
    tw, wood = P.tw_wood(front)
    out = []
    for s in slots:
        t = tw[s["foot"]].mean()
        w = wood[s["foot"]].mean()
        out.append("X" if t > FILLED_FRAC else ("." if w > VACANT_WOOD else "?"))
    return "".join(out)


def moving_blobs(front, slots):
    S = np.array([s["xy"] for s in slots])
    out = []
    for x, y, c in P.target_blobs(front):
        d = np.hypot(S[:, 0] - x, S[:, 1] - y)
        if d.min() > SLOT_AT_PX:
            out.append((x, y, c))
    return out


def vacancy_runs(states, times, k):
    """Runs of frames where slot k is not filled and at least one frame is vacant('.').
    Returns list of (i_start, i_end) inclusive indices into frames."""
    runs = []
    n = len(states)
    i = 0
    while i < n:
        if states[i][k] != "X":
            j = i
            while j + 1 < n and states[j + 1][k] != "X":
                j += 1
            seg = [states[t][k] for t in range(i, j + 1)]
            if "." in seg:
                # start/end of the strong ('.') part
                a = i + seg.index(".")
                b = i + len(seg) - 1 - seg[::-1].index(".")
                runs.append((i, j, a, b))
            i = j + 1
        else:
            i += 1
    return runs


def pair_cost(blobs_per_frame, M, sep):
    """Mean over frames of the best blob-pair midpoint error to M (clipped)."""
    costs = []
    for blobs in blobs_per_frame:
        best = 25.0
        for (x1, y1, _), (x2, y2, _) in itertools.combinations(blobs, 2):
            if np.hypot(x1 - x2, y1 - y2) < 0.3 * sep:
                continue
            e = np.hypot((x1 + x2) / 2 - M[0], (y1 + y2) / 2 - M[1])
            best = min(best, e)
        if len(blobs) >= 2:
            costs.append(best)
    return (float(np.mean(costs)), len(costs)) if costs else (25.0, 0)


def matching_cost(V, slots, blobs_per_frame):
    """Best perfect matching of the vacated set V by summed pair costs; returns {slot: partner}."""
    V = list(V)
    best = None
    def matchings(items):
        if not items:
            yield []
            return
        a = items[0]
        for i in range(1, len(items)):
            rest = items[1:i] + items[i + 1:]
            for m in matchings(rest):
                yield [(a, items[i])] + m
    if len(V) % 2 == 1:
        return None, None
    for m in matchings(V):
        tot = 0.0
        for a, b in m:
            M = (slots[a]["xy"] + slots[b]["xy"]) / 2
            sep = float(np.linalg.norm(slots[a]["xy"] - slots[b]["xy"]))
            c, _ = pair_cost(blobs_per_frame, M, sep)
            tot += c
        if best is None or tot < best[0]:
            best = (tot, m)
    part = {}
    for a, b in best[1]:
        part[a] = b
        part[b] = a
    return part, best[0]


def partner_of(k, i0, i1, states, times, slots, fronts, log):
    """Partner slot of k in the swap whose vacancy of k spans frames i0..i1."""
    n_slots = len(slots)
    t0 = times[i0]
    # other slots that are vacant (strong '.') overlapping k's run, starting close in time
    cands = []
    for j in range(n_slots):
        if j == k:
            continue
        for (a, b, sa, sb) in vacancy_runs(states, times, j):
            if b < i0 - 1 or a > i1 + 1:
                continue
            if abs(times[a] - t0) <= SWAP_START_TOL_S or abs(times[b] - times[i1]) <= SWAP_START_TOL_S:
                cands.append(j)
                break
    lo, hi = max(0, i0 - 1), min(len(fronts) - 1, i1 + 1)
    blobs = [moving_blobs(fronts[t], slots) for t in range(lo, hi + 1)]
    if len(cands) == 1:
        log.append("swap slot %d<->%d (unique vacancy partner) t=%.2f" % (k, cands[0], t0))
        return cands[0], 1.0
    if len(cands) >= 3 and len(cands) % 2 == 1:
        part, cost = matching_cost([k] + cands, slots, blobs)
        log.append("swap slot %d<->%d (matching of %s cost %.1f) t=%.2f" % (k, part[k], [k] + cands, cost, t0))
        return part[k], 0.8
    # 0 or an even number of other vacancies: score every slot not seen filled throughout
    pool = cands if cands else [j for j in range(n_slots) if j != k and
                                any(states[t][j] != "X" for t in range(lo, hi + 1))]
    if not pool:
        pool = [j for j in range(n_slots) if j != k]
    scored = []
    for j in pool:
        M = (slots[k]["xy"] + slots[j]["xy"]) / 2
        sep = float(np.linalg.norm(slots[k]["xy"] - slots[j]["xy"]))
        c, nf = pair_cost(blobs, M, sep)
        scored.append((c, j, nf))
    scored.sort()
    log.append("swap slot %d<->%d (scored pool %s) t=%.2f" % (k, scored[0][1], [(j, round(c, 1)) for c, j, _ in scored], t0))
    return scored[0][1], 0.5


def held_timeline(wrists):
    return [P.held_color(w) if w is not None else None for w in wrists]


def analyse(frames, color):
    log = []
    times = [float(t) for t, _, _ in frames]
    fronts = [f for _, f, _ in frames]
    wrists = [w for _, _, w in frames]
    slots = find_slots(fronts)
    log.append("slots %s" % [tuple(np.round(s["xy"]).astype(int)) for s in slots])
    states = [slot_states(f, slots) for f in fronts]
    # button press
    press_i = None
    for i, w in enumerate(wrists):
        if w is not None and P.is_press(w):
            press_i = i
            break
    log.append("press frame %s t=%s" % (press_i, None if press_i is None else round(times[press_i], 2)))
    held = held_timeline(wrists)
    releases = []   # (index, color)
    for i in range(1, len(held)):
        if held[i - 1] is not None and held[i] != held[i - 1]:
            releases.append((i, held[i - 1]))
    log.append("releases %s" % [(i, round(times[i], 2), c) for i, c in releases])
    start = press_i if press_i is not None else 0
    chosen = None
    for i, c in releases:
        if i > start and c == color:
            chosen = i
            break
    conf = 1.0
    if chosen is None:
        conf = 0.3
        for i, c in releases:
            if i > start:
                chosen = i
                break
    if chosen is None and releases:
        chosen = releases[-1][0]
        conf = 0.2
    if chosen is None:
        return {"slot_xy": None, "slots": slots, "log": log, "confidence": 0.0, "states": states}
    # cube position after release: median over frames until the gripper grasps again
    pts = []
    for t in range(chosen, min(len(fronts), chosen + 40)):
        if held[t] is not None and t > chosen + 1:
            break
        b = P.cube_blob(fronts[t], color)
        if b is not None:
            pts.append((b[0], b[1] + CUBE_Y_OFFSET))
    S = np.array([s["xy"] for s in slots])
    if not pts:
        log.append("no cube seen after release")
        return {"slot_xy": None, "slots": slots, "log": log, "confidence": 0.0, "states": states}
    cx, cy = np.median(np.array(pts), axis=0)
    d = np.hypot(S[:, 0] - cx, S[:, 1] - cy)
    k = int(np.argmin(d))
    log.append("release i=%d t=%.2f cube (%.0f,%.0f) -> slot %d d=%.1f" % (chosen, times[chosen], cx, cy, k, d[k]))
    if d[k] > RELEASE_SLOT_MAX_PX:
        conf *= 0.5
    # follow slot k through later swaps
    i = chosen
    guard = 0
    while guard < 200:
        guard += 1
        nxt = None
        for (a, b, sa, sb) in vacancy_runs(states, times, k):
            if sa > i:
                nxt = (a, b, sa, sb)
                break
        if nxt is None:
            break
        a, b, sa, sb = nxt
        j, c = partner_of(k, a, b, states, times, slots, fronts, log)
        conf *= c
        k = j
        i = b
    log.append("final slot %d at %s" % (k, tuple(np.round(slots[k]["xy"]).astype(int))))
    return {"slot_xy": slots[k]["xy"].copy(), "slot": k, "slots": slots, "log": log,
            "confidence": conf, "states": states}
