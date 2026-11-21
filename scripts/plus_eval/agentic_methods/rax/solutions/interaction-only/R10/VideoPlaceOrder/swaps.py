"""Infer target identity permutations (pairwise semicircle swaps) from video detections in world XY."""
import itertools
import numpy as np

def involutions(n):
    """All partitions of range(n) into fixed points and disjoint pairs (excluding identity)."""
    res = []
    def rec(rem, pairs):
        if not rem:
            if pairs:
                res.append(list(pairs))
            return
        a = rem[0]
        rec(rem[1:], pairs)  # a fixed
        for b in rem[1:]:
            r2 = [x for x in rem[1:] if x != b]
            rec(r2, pairs + [(a, b)])
    rec(list(range(n)), [])
    return res

def predict(slots, pairs, dirs, theta):
    P = slots.copy()
    for (a, b), d in zip(pairs, dirs):
        m = (slots[a] + slots[b]) / 2
        c, s = np.cos(d * theta), np.sin(d * theta)
        Rm = np.array([[c, -s], [s, c]])
        P[a] = m + Rm @ (slots[a] - m)
        P[b] = m + Rm @ (slots[b] - m)
    return P

def frame_cost(D, P, cap=0.05):
    if len(D) == 0:
        return 0.0
    C = np.linalg.norm(D[:, None, :] - P[None, :, :], axis=2)
    c1 = np.minimum(C.min(axis=1), cap).mean()
    c2 = np.minimum(C.min(axis=0), cap).mean()
    return c1 + 0.5 * c2

def fit_event(slots, dets, log=print):
    """dets: list of (k,2) arrays for frames of one motion interval. Returns best pairs."""
    n = len(slots)
    thetas = np.linspace(0, np.pi, 37)
    best = None
    for pairs in involutions(n):
        for dirs in itertools.product([1, -1], repeat=len(pairs)):
            total = 0.0
            for D in dets:
                cs = [frame_cost(D, predict(slots, pairs, dirs, th)) for th in thetas]
                total += min(cs)
            if best is None or total < best[0]:
                best = (total, pairs, dirs)
    # null hypothesis cost
    null = sum(frame_cost(D, slots) for D in dets)
    log("swap fit: cost %.4f pairs %s dirs %s (null %.4f, frames %d)" % (best[0], best[1], best[2], null, len(dets)))
    return best[1], best[0], null

def segment(slot_xy, det_list, thr=0.025):
    """det_list: list of (frame_index, (k,2) array). Return list of motion intervals (lists of indices)."""
    moving = []
    for fi, D in det_list:
        if len(D) == 0:
            moving.append(False)
            continue
        dmin = np.linalg.norm(D[:, None, :] - slot_xy[None], axis=2).min(axis=1)
        moving.append(bool((dmin > thr).any()))
    intervals = []
    cur = []
    gap = 0
    for i, mv in enumerate(moving):
        if mv:
            cur.append(i)
            gap = 0
        elif cur:
            gap += 1
            if gap > 1:
                intervals.append(cur)
                cur = []
                gap = 0
    if cur:
        intervals.append(cur)
    return intervals
