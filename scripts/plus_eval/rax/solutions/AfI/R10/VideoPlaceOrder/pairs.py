"""Pair-swap inference: in each motion interval targets swap in pairs, moving point-symmetrically
about the pair midpoint. Identity: permutation composed of the inferred pair swaps."""
import numpy as np
from wtrack import frame_points, init_centers

def involutions(idx):
    res = []
    def rec(rem, pairs):
        if not rem:
            res.append(list(pairs)); return
        a = rem[0]
        rec(rem[1:], pairs)
        for b in rem[1:]:
            rec([x for x in rem[1:] if x != b], pairs + [(a, b)])
    rec(list(idx), [])
    return [r for r in res if r]

def analyse(frames, log=print, rest_thr=0.025, vac_r=0.02):
    frames = sorted(frames, key=lambda f: f["frame_index"])
    cl = [init_centers(frame_points(f)) for f in frames]
    counts = [len(c[0]) for c in cl]
    vals, cnts = np.unique(counts, return_counts=True)
    N = int(vals[np.argmax(cnts)])
    start = next(i for i, c in enumerate(cl) if len(c[0]) == N)
    slots = cl[start][0].copy()
    log("target count", N, "slots", np.round(slots, 3).tolist())
    moving = []
    for i, (D, ns) in enumerate(cl):
        if len(D) == 0 or i <= start:
            moving.append(False); continue
        dmin = np.linalg.norm(D[:, None] - slots[None], axis=2).min(1)
        moving.append(bool((dmin > rest_thr).any()))
    # intervals with gap tolerance 2
    ivs = []; cur = None; gap = 0
    for i, mv in enumerate(moving):
        if mv:
            if cur is None: cur = [i, i]
            cur[1] = i; gap = 0
        elif cur is not None:
            gap += 1
            if gap > 2:
                ivs.append(cur); cur = None; gap = 0
    if cur is not None: ivs.append(cur)
    occupant = np.arange(N)  # slot -> target id
    occ_hist = {}
    events = []
    iv_ptr = 0
    swaps_at = {}
    for (a, b) in ivs:
        F = list(range(a, b + 1))
        dets = [cl[i][0] for i in F]
        # vacancy per slot
        vac = np.zeros(N)
        for D in dets:
            for s in range(N):
                if len(D) == 0 or np.linalg.norm(D - slots[s], axis=1).min() > vac_r:
                    vac[s] += 1
        vac /= len(F)
        movers = [s for s in range(N) if vac[s] > 0.6]
        best = None
        cands = involutions(movers if len(movers) >= 2 else range(N))
        for H in cands:
            cost = 0.0; nm = 0
            for D in dets:
                if len(D) == 0: continue
                dmin = np.linalg.norm(D[:, None] - slots[None], axis=2).min(1)
                for d, dm in zip(D, dmin):
                    if dm < rest_thr: continue
                    nm += 1
                    c = min(min(np.linalg.norm(D - (slots[p] + slots[q] - d), axis=1).min(), 0.06) for p, q in H)
                    cost += c
            covered = set(x for p in H for x in p)
            pen = sum(1 for s in movers if s not in covered) * 0.05 + sum(1 for s in covered if vac[s] < 0.6) * 0.5
            tot = cost / max(nm, 1) + pen
            if best is None or tot < best[0]:
                best = (tot, H)
        far = 0
        for D in dets:
            if len(D) and np.linalg.norm(D[:, None] - slots[None], axis=2).min(1).max() > 0.05:
                far += 1
        if len(F) < 3 or far < 3 or best is None or best[0] > 0.3:
            log("interval %d-%d (frames %d) vac %s movers %s -> IGNORED (best %s)" % (frames[a]["frame_index"], frames[b]["frame_index"], len(F), np.round(vac, 2).tolist(), movers, best))
            continue
        log("far %d interval %d-%d vac %s movers %s -> pairs %s cost %.4f" % (far, frames[a]["frame_index"], frames[b]["frame_index"], np.round(vac, 2).tolist(), movers, best[1], best[0]))
        swaps_at[b] = best[1]
        events.append((frames[a]["frame_index"], frames[b]["frame_index"], best[1], best[0]))
    for i, f in enumerate(frames):
        if i in swaps_at:
            for p, q in swaps_at[i]:
                occupant[p], occupant[q] = occupant[q], occupant[p]
        occ_hist[f["frame_index"]] = occupant.copy()
    return dict(N=N, slots=slots, events=events, occ=occ_hist, final=occupant.copy(), ivs=ivs)
