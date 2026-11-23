"""Per-event permutation tracking of identical targets in table-plane coordinates."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def track_event(Wslots, Wdets, stat, a, b, log=None, gate=0.085, excl=0.014, vac_frac=0.45):
    """Wslots: (S,2) world slot xy. Wdets: list per frame of (n,2) world xy of target detections.
    stat: (T,S) slot status (0 vacant, 1 occupied, -1 occluded, -2 cube-covered).
    Returns (perm dict src->dst, quality dict)."""
    T = len(Wdets)
    seg = stat[a:b + 1]
    frac = (seg == 0).mean(0)
    V = [int(s) for s in np.nonzero(frac >= vac_frac)[0]]
    if len(V) < 2:
        return None, dict(V=V, reason='lt2')
    nonV = [s for s in range(len(Wslots)) if s not in V]
    pos = Wslots[V].copy()
    vel = np.zeros_like(pos)
    miss = np.zeros(len(V), int)
    i0, i1 = max(0, a - 1), min(T - 1, b + 2)
    for i in range(i0, i1 + 1):
        D = np.asarray(Wdets[i]).reshape(-1, 2)
        if len(D) and nonV:
            dn = np.min(np.linalg.norm(D[:, None, :] - Wslots[nonV][None, :, :], axis=2), axis=1)
            D = D[dn > excl]
        pred = pos + vel
        if len(D):
            C = np.linalg.norm(pred[:, None, :] - D[None, :, :], axis=2)
            C0 = np.linalg.norm(pos[:, None, :] - D[None, :, :], axis=2)
            C = np.minimum(C, C0 + 0.01)
            r, c = linear_sum_assignment(C)
            matched = set()
            for ti, dj in zip(r, c):
                if C[ti, dj] < gate:
                    v = D[dj] - pos[ti]
                    vel[ti] = 0.6 * v + 0.4 * vel[ti]
                    pos[ti] = D[dj]
                    miss[ti] = 0
                    matched.add(ti)
            for ti in range(len(V)):
                if ti not in matched:
                    miss[ti] += 1
                    if miss[ti] <= 4:
                        pos[ti] = pred[ti]
                    vel[ti] *= 0.7
        else:
            miss += 1
            pos = pos + vel * (miss[:, None] <= 4)
            vel *= 0.7
    # final assignment to V slots
    C = np.linalg.norm(pos[:, None, :] - Wslots[V][None, :, :], axis=2)
    r, c = linear_sum_assignment(C)
    perm = {V[ti]: V[sj] for ti, sj in zip(r, c)}
    err = float(max(C[ti, sj] for ti, sj in zip(r, c)))
    q = dict(V=V, err=round(err, 3), frac=np.round(frac, 2).tolist())
    return perm, q


def pair_evidence(Wslots, Wdets, stat, a, b, vac_frac=0.4, excl=0.014, tol=0.012):
    """Pairwise swaps rotate both targets about their midpoint: at every moving frame the two
    movers are point-symmetric about m=(Sa+Sb)/2.  Score every pair of vacated slots by the
    fraction of frames that contain such a symmetric detection pair."""
    seg = stat[a:b + 1]
    frac = (seg == 0).mean(0)
    V = [int(s) for s in np.nonzero(frac >= vac_frac)[0]]
    nonV = [s for s in range(len(Wslots)) if s not in V]
    sc = {}
    n = b - a + 1
    for ii in range(len(V)):
        for jj in range(ii + 1, len(V)):
            i, j = V[ii], V[jj]
            m = 0.5 * (Wslots[i] + Wslots[j]); h = 0.5 * np.linalg.norm(Wslots[i] - Wslots[j])
            good = 0
            for k in range(a, b + 1):
                D = np.asarray(Wdets[k]).reshape(-1, 2)
                if len(D) and nonV:
                    dn = np.min(np.linalg.norm(D[:, None, :] - Wslots[nonV][None, :, :], axis=2), axis=1)
                    D = D[dn > excl]
                if len(D) < 2:
                    continue
                r = np.linalg.norm(D - m, axis=1)
                ok = False
                for p in range(len(D)):
                    if r[p] < 0.3 * h or r[p] > 3.0 * h + 0.03:
                        continue
                    ref = 2 * m - D[p]
                    dd = np.linalg.norm(D - ref, axis=1); dd[p] = 9
                    if dd.min() < tol:
                        ok = True; break
                good += ok
            sc[(i, j)] = good / n
    return V, sc, frac


def best_matching(V, sc, min_score=0.25):
    """Maximum-weight set of disjoint pairs (V small -> brute force)."""
    best = (0.0, [])
    def rec(rem, cur, tot):
        nonlocal best
        if tot > best[0] + 1e-9:
            best = (tot, list(cur))
        if len(rem) < 2:
            return
        x = rem[0]
        rec(rem[1:], cur, tot)  # x unpaired
        for y in rem[1:]:
            s = sc.get((x, y), sc.get((y, x), 0.0))
            if s >= min_score:
                rec([z for z in rem[1:] if z != y], cur + [(x, y)], tot + s)
    rec(list(V), [], 0.0)
    return best[1], best[0]
