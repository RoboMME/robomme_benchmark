"""Slot/swap model of the moving targets in the conditioning video (v3: antipodal pairing).

Observed (ep1, rehearsal 1): K pink/purple targets occupy K fixed slots. Two targets swap places
by moving on an ellipse centred on the midpoint of their two slots; at every instant the two
moving targets are point-symmetric about that midpoint. Several pairs can swap at once, and
swaps can follow each other without pause. A placement's target identity must be followed
through all later swaps.

Detection of a swap of pair (i, j) in frame t: two off-slot detections p, q with
|(p+q)/2 - mid_ij| small and |p - q| large. Runs of such frames form one swap occurrence; the
number of half turns in the run (unwrapped psi = 2*phi about the midpoint) decides whether the
occupants exchanged (odd) or not (even).
"""
import itertools
import numpy as np

ON_SLOT = 0.025     # detection within this of a slot = sitting on the slot
MID_TOL = 0.025     # |(p+q)/2 - pair midpoint| for antipodal support
MIN_SEP = 0.05      # |p - q| minimum for support (not one merged blob)
GAP = 4             # frames: bridge support gaps up to this length within one run
MIN_SUPPORT = 4     # supported frames needed for a swap occurrence
H_PERP = 0.160      # perpendicular semi-axis of the swap ellipse (m); measured 0.158-0.167 on 28 swaps, 2 videos
ELL_TOL = 0.015     # off-slot point within this of the pair ellipse counts as on the path
_TH = np.linspace(-np.pi, np.pi, 181)


def find_slots(dets_all, K):
    """Slots from the last frame with K detections (robot at home there)."""
    for d in reversed(dets_all):
        if len(d) == K:
            return np.array([x["xy"] for x in d], float)
    return np.array([x["xy"] for x in max(dets_all, key=len)], float)


def _off_points(dets_all, slots):
    off = []
    for d in dets_all:
        pts = [x["xy"] for x in d if np.min(np.linalg.norm(slots - x["xy"], axis=1)) > ON_SLOT]
        off.append(np.array(pts, float).reshape(-1, 2))
    return off


def _support(off, mid):
    """List of (p, q) antipodal off-slot pairs about mid in one frame (best one), or None."""
    if len(off) < 2:
        return None
    best = None
    for a, b in itertools.combinations(range(len(off)), 2):
        p, q = off[a], off[b]
        e = np.linalg.norm((p + q) / 2 - mid)
        if e < MID_TOL and np.linalg.norm(p - q) > MIN_SEP and (best is None or e < best[0]):
            best = (e, p, q)
    return None if best is None else (best[1], best[2])


def _runs(frames_supported):
    runs = []
    for t in frames_supported:
        if runs and t - runs[-1][-1] <= GAP + 1:
            runs[-1].append(t)
        else:
            runs.append([t])
    return runs


def _half_turns(slots, i, j, samples):
    """samples: list of (t, p) points of the target that started on slot i... we use psi=2*phi so
    either target works. Returns unwrapped half turns."""
    mid = (slots[i] + slots[j]) / 2
    d = slots[j] - slots[i]
    r = np.linalg.norm(d) / 2
    u = d / (2 * r); w = np.array([-u[1], u[0]])
    al = np.array([(p - mid) @ u for _, p in samples])
    pp = np.array([(p - mid) @ w for _, p in samples])
    h = max(float(np.max(np.abs(pp))) if len(pp) else r, 0.02)
    psi = [0.0] + list(2 * np.arctan2(pp / h, al / r)) + [0.0]
    un = np.unwrap(np.array(psi))
    return float(un[-1] - un[0]) / (2 * np.pi)


def _frame(slots, i, j):
    mid = (slots[i] + slots[j]) / 2
    d = slots[j] - slots[i]
    r = np.linalg.norm(d) / 2
    u = d / (2 * r)
    w = np.array([-u[1], u[0]])
    return mid, r, u, w


def _ellipse_dist(P, mid, r, u, w):
    """Distance of points P (n,2) to the ellipse centred at mid, semi-axes r (along u), H_PERP (along w)."""
    if len(P) == 0:
        return np.zeros(0)
    E = mid + np.outer(r * np.cos(_TH), u) + np.outer(H_PERP * np.sin(_TH), w)
    return np.min(np.linalg.norm(P[:, None, :] - E[None], axis=2), axis=1)


def _occupied(dets, slots):
    dd = np.array([x["xy"] for x in dets]).reshape(-1, 2)
    if len(dd) == 0:
        return np.zeros(len(slots), bool)
    return np.min(np.linalg.norm(slots[:, None, :] - dd[None], axis=2), axis=1) <= ON_SLOT


def _half_turns2(samples, mid, r, u, w):
    if not samples:
        return 0.0
    P = np.array([p for _, p in samples])
    al = (P - mid) @ u / r
    pp = (P - mid) @ w / H_PERP
    ts = np.array([t for t, _ in samples])
    psi_pts = 2 * np.arctan2(pp, al)
    # one circular-mean psi per frame (both antipodal targets share psi)
    psis = [0.0]
    for t in sorted(set(ts.tolist())):
        v = psi_pts[ts == t]
        psis.append(float(np.arctan2(np.sin(v).mean(), np.cos(v).mean())))
    psis.append(0.0)
    un = np.unwrap(np.array(psis))
    return float(un[-1] - un[0]) / (2 * np.pi)


def swap_events(dets_all, K, frame_ids, log=print):
    slots = find_slots(dets_all, K)
    off = _off_points(dets_all, slots)
    occ = [_occupied(d, slots) for d in dets_all]
    cands = []
    for i, j in itertools.combinations(range(len(slots)), 2):
        mid, r, u, w = _frame(slots, i, j)
        sup, anti, samp = {}, set(), []
        for t in range(len(off)):
            s = _support(off[t], mid)
            on_path = off[t][_ellipse_dist(off[t], mid, r, u, w) < ELL_TOL] if len(off[t]) else off[t]
            if s is not None:
                sup[t] = 2; anti.add(t)
                samp += [(t, s[0]), (t, s[1])]
            elif len(on_path) and not occ[t][i] and not occ[t][j]:
                sup[t] = 1
                samp += [(t, p) for p in on_path]
        for run in _runs(sorted(sup)):
            n_anti = sum(1 for t in run if t in anti)
            if len(run) < MIN_SUPPORT or n_anti < 1:
                continue
            rs = set(run)
            ht = _half_turns2([x for x in samp if x[0] in rs], mid, r, u, w)
            cands.append({"pair": (i, j), "t": run[0], "t_end": run[-1], "support": len(run), "anti": n_anti,
                          "score": len(run) + n_anti, "half_turns": ht,
                          "frame_start": frame_ids[run[0]], "frame_end": frame_ids[run[-1]]})
    cands.sort(key=lambda c: -c["score"])
    accepted = []
    for c in cands:
        clash = None
        for a in accepted:
            if set(a["pair"]) & set(c["pair"]) and not (c["t_end"] < a["t"] - 1 or c["t"] > a["t_end"] + 1):
                clash = a; break
        if clash is not None:
            log("  swap cand %s f%d-%d support %d/%d REJECTED (clash with %s f%d-%d)" % (
                c["pair"], c["frame_start"], c["frame_end"], c["support"], c["anti"], clash["pair"],
                clash["frame_start"], clash["frame_end"]))
            continue
        accepted.append(c)
    accepted.sort(key=lambda c: c["t"])
    events = []
    for c in accepted:
        nh = int(round(abs(c["half_turns"])))
        c["swapped"] = (nh % 2 == 1)
        log("  swap occurrence %s f%d-%d support %d anti %d half-turns %.2f swapped=%s" % (
            c["pair"], c["frame_start"], c["frame_end"], c["support"], c["anti"], c["half_turns"], c["swapped"]))
        if c["swapped"]:
            events.append(c)
    return slots, events


def follow(slot0, t_from, events):
    """Slot index of the target that was on slot0 at time index t_from, after all later swaps."""
    s = slot0
    for e in events:
        if e["t"] <= t_from:
            continue
        i, j = e["pair"]
        if s == i:
            s = j
        elif s == j:
            s = i
    return s
