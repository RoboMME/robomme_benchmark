"""Slot-based swap detection for the target rings in the conditioning video.

Model (observed in exploration episode 1): the rings rest on fixed slots; occasionally a pair
of rings rotates about the pair's midpoint (usually 180 deg = exchange of places), possibly two
pairs at once. Each ring keeps its identity; the task refers to ring identities.
We find motion segments (frames with ring detections away from every slot), explain the
off-slot detections by the circle of a slot pair, unwrap the pair's line orientation to get
the number of half turns, and apply the resulting permutations in time order.
"""
import itertools, math
import numpy as np
import tracking as T

OFF_SLOT_M = 0.035
CIRCLE_TOL_M = 0.017
MIN_OFF_PX = 25
SYM_TOL_M = 0.022
SWAP_RESID_DEG = 45.0   # true swaps: |total| 163-185 deg (ep1 + rehearsal1 videos)
MIN_RUN_N = 3           # true swaps: 9-11 evidence frames at 512-frame sampling


def slots_from(dets_list, ks):
    """Slots = ring positions in a rest frame with the most well-separated big detections
    (latest such frame among ks order of preference)."""
    best, bestn = None, 0
    for k in ks:
        d = dets_list[k]
        if not d:
            continue
        big = [c for c, n in d if n >= 40]
        if len(big) < 2:
            continue
        sep = min(np.linalg.norm(a - b) for i, a in enumerate(big) for b in big[i + 1:])
        if sep < 0.08:
            continue
        if len(big) > bestn:
            best, bestn = np.array(big), len(big)
    return best


def analyse(frames, log=print):
    dets = [T.frame_detections(f) for f in frames]
    N = len(frames)
    # slots from the final frames (execution boundary), fallback early frames
    S = slots_from(dets, list(range(N - 1, -1, -1)))
    if S is None:
        return None
    K = len(S)
    # off-slot detections per frame
    off = []
    for k in range(N):
        d = dets[k] or []
        o = [c for c, n in d if n >= MIN_OFF_PX and np.min(np.linalg.norm(S - c, axis=1)) > OFF_SLOT_M]
        off.append(o)
    moving = [len(o) > 0 for o in off]
    # segments: runs of moving frames, merging gaps <= 2
    segs = []
    k = 0
    while k < N:
        if moving[k]:
            s = k; e = k; g = 0; j = k + 1
            while j < N and g <= 2:
                if moving[j]:
                    e = j; g = 0
                else:
                    g += 1
                j += 1
            segs.append((s, e)); k = e + 1
        else:
            k += 1
    pairs = list(itertools.combinations(range(K), 2))
    mids = {p: (S[p[0]] + S[p[1]]) / 2 for p in pairs}

    def offslot(c):
        return np.min(np.linalg.norm(S - c, axis=1)) > OFF_SLOT_M

    def sym_obs(k, p, need_off):
        """Detection pairs symmetric about the pair midpoint in frame k -> list of line angles."""
        d = [c for c, n in (dets[k] or []) if n >= T.FRAG_PX]
        out = []
        for a in range(len(d)):
            for b in range(a + 1, len(d)):
                c1, c2 = d[a], d[b]
                if np.linalg.norm(c1 - c2) < 0.06:
                    continue
                if np.linalg.norm((c1 + c2) / 2 - mids[p]) > SYM_TOL_M:
                    continue
                if need_off and not (offslot(c1) or offslot(c2)):
                    continue
                v = c1 - c2
                out.append(math.atan2(v[1], v[0]) % math.pi)
        return out

    # per-frame evidence of each pair moving, then runs per pair
    ev_frames = {p: [k for k in range(N) if moving[k] and sym_obs(k, p, True)] for p in pairs}
    runs = []
    for p, ks in ev_frames.items():
        if not ks:
            continue
        s = ks[0]; prev = ks[0]
        for k in ks[1:] + [None]:
            if k is not None and k - prev <= 2:
                prev = k; continue
            runs.append({"pair": p, "start": s, "end": prev, "n": sum(1 for x in ks if s <= x <= prev)})
            if k is not None:
                s = prev = k
    runs.sort(key=lambda r: (-r["n"], r["start"]))
    accepted = []
    for r in runs:
        if r["n"] < 2:
            continue
        clash = any(set(r["pair"]) & set(a["pair"]) and
                    (min(r["end"], a["end"]) - max(r["start"], a["start"]) + 1) > 1
                    for a in accepted)
        if not clash:
            accepted.append(r)
    accepted.sort(key=lambda r: r["start"])
    events = []
    for r in accepted:
        p = r["pair"]; m = mids[p]
        # unwrap window bounded by neighbouring accepted runs that share a slot
        lo = max(0, r["start"] - 2); hi = min(N - 1, r["end"] + 2)
        for a in accepted:
            if a is r or not (set(a["pair"]) & set(p)):
                continue
            if a["start"] < r["start"]:
                lo = max(lo, min(a["end"] + 1, r["start"]))
            if a["start"] > r["start"]:
                hi = min(hi, max(a["start"] - 1, r["end"]))
        phi0 = math.atan2(*(S[p[0]] - S[p[1]])[::-1]) % math.pi
        cur = phi0; total = 0.0; nobs = 0; nsingle = 0
        others = [S[q] for q in range(K) if q not in p]
        for k in range(lo, hi + 1):
            cands = sym_obs(k, p, False)
            if not cands:
                for c, n in (dets[k] or []):
                    if n >= MIN_OFF_PX and offslot(c) and np.linalg.norm(c - m) > 0.04 \
                            and all(np.linalg.norm(c - o) > 0.03 for o in others):
                        cands.append(math.atan2(*(c - m)[::-1]) % math.pi); nsingle += 1
            if not cands:
                continue
            dl = [((ph - cur + math.pi / 2) % math.pi - math.pi / 2) for ph in cands]
            dphi = min(dl, key=abs)
            if abs(dphi) > math.radians(75):
                continue
            total += dphi; cur = (cur + dphi) % math.pi; nobs += 1
        halfturns = int(round(total / math.pi))
        resid = abs(total - halfturns * math.pi)
        # accept a swap only for a clear odd number of half turns (spurious short runs give ~0 or ~90 deg)
        swap = (halfturns % 2 == 1) and resid < math.radians(SWAP_RESID_DEG) and r["n"] >= MIN_RUN_N
        events.append({"start": r["start"], "end": r["end"], "pair": p, "n": r["n"], "total_deg": round(math.degrees(total), 1),
                       "halfturns": halfturns, "nobs": nobs, "nsingle": nsingle, "swap": swap})
    rejected = [r for r in runs if r not in accepted and r["n"] >= 2]
    return {"slots": S, "segments": segs, "events": events, "dets": dets, "rejected": rejected}


def occupant_timeline(res, N):
    """occ[k][slot] = ring identity sitting on slot at frame k (identities = slot index at frame 0)."""
    K = len(res["slots"])
    occ = list(range(K))
    ends = {}
    for ev in res["events"]:
        if ev.get("pair") is not None and ev.get("swap"):
            ends.setdefault(ev["end"], []).append(ev["pair"])
    timeline = []
    for k in range(N):
        timeline.append(list(occ))
        for p in ends.get(k, []):
            a, b = p
            occ[a], occ[b] = occ[b], occ[a]
    return timeline, occ


def nearest_slot(S, xy, max_d=0.06):
    d = np.linalg.norm(S - np.asarray(xy[:2]), axis=1)
    i = int(np.argmin(d))
    return i if d[i] < max_d else None
