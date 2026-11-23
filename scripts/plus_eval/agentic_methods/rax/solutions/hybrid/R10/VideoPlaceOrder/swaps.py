"""Swap inference: targets exchange slots moving point-symmetrically about the pair midpoint."""
import numpy as np
from itertools import combinations

C = 0.05

def matchings(n, kmax):
    pairs = list(combinations(range(n), 2))
    out = [()]
    def rec(start, used, cur):
        for idx in range(start, len(pairs)):
            i, j = pairs[idx]
            if i in used or j in used:
                continue
            nc = cur + ((i, j),)
            out.append(nc)
            if len(nc) < kmax:
                rec(idx + 1, used | {i, j}, nc)
    rec(0, frozenset(), ())
    return out

def frame_cost(D, slots, H, static):
    """D: array (k,2). returns total cost, per-pair evidence (max displacement of symmetric explained pts)."""
    cost = 0.0
    ev = {pr: 0.0 for pr in H}
    for a, p in enumerate(D):
        best = C
        if static:
            best = min(best, min(np.linalg.norm(slots[s] - p) for s in static))
        bp = None
        for pr in H:
            m2 = slots[pr[0]] + slots[pr[1]]
            # merged at midpoint
            dm = np.linalg.norm(2 * p - m2) / 2
            dq = np.min(np.linalg.norm(D + p - m2, axis=1)) if len(D) else C
            dpp = min(dm, dq)
            if dpp < best:
                best = dpp; bp = pr
        cost += best
        if bp is not None and best < 0.015:
            disp = min(np.linalg.norm(p - slots[bp[0]]), np.linalg.norm(p - slots[bp[1]]))
            ev[bp] = max(ev[bp], disp)
    return cost, ev

def infer_segment(dets, slots, a, b, kmax=3, log=None):
    """dets: list of lists of xy arrays for frames a..b. Returns chosen pairs."""
    n = len(slots)
    frames = [np.array([p for p, _ in dets[f]]).reshape(-1, 2) for f in range(a, b + 1)]
    best = None
    results = []
    for H in matchings(n, kmax):
        inH = {i for pr in H for i in pr}
        static = [i for i in range(n) if i not in inH]
        tot = 0.0
        evc = {pr: 0 for pr in H}
        evmax = {pr: 0.0 for pr in H}
        for D in frames:
            c, ev = frame_cost(D, slots, H, static)
            tot += c
            for pr, v in ev.items():
                if v > 0.04:
                    evc[pr] += 1
                evmax[pr] = max(evmax[pr], v)
        valid = all(evc[pr] >= 1 and evmax[pr] > 0.06 for pr in H)
        if not valid:
            continue
        score = tot + 0.004 * len(H) * len(frames)
        results.append((score, H, tot))
    results.sort(key=lambda r: r[0])
    if log:
        log(f'  seg {a}-{b}: ' + '; '.join(f'{list(H)} {sc:.3f}' for sc, H, t in results[:3]))
    return list(results[0][1]) if results else [], results
