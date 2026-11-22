"""Track (identical-looking) targets through the conditioning video; they swap positions over time."""
import itertools, json
import numpy as np
from PIL import Image
from lib import WS, world_points, hsv, components


def detect_targets_img(meta, rgb, depth):
    imgs = {("front", "rgb"): rgb, ("front", "depth"): depth}
    P, _ = world_points(meta, imgs, "front")
    h = hsv(rgb); H, S, V = h[..., 0], h[..., 1], h[..., 2]
    low = (P[..., 2] < 0.013) & (P[..., 2] > -0.01)
    colored = ((H >= 180) | (H <= 4)) & (S > 50) & (S <= 185) & (V > 140)
    white = (S < 50) & (V > 200)
    ring = (colored | white) & low
    pts = []
    for ys, xs in components(ring, 3):
        p = np.median(P[ys, xs], 0)
        pts.append((p[:2], len(ys)))
    clusters = []
    for p, n in sorted(pts, key=lambda t: -t[1]):
        for c in clusters:
            if np.linalg.norm(c["sum"] / c["n"] - p) < 0.035:
                c["sum"] += p * n; c["n"] += n; break
        else:
            clusters.append(dict(sum=p * n, n=n))
    return [c["sum"] / c["n"] for c in clusters if c["n"] >= 15]


def best_assignment(T, D, gate):
    """Min-cost matching of track predictions T (list) to detections D (list) with gating."""
    nT, nD = len(T), len(D)
    if nT == 0 or nD == 0:
        return {}
    C = np.array([[np.linalg.norm(t - d) for d in D] for t in T])
    best, bestc = {}, None
    # brute force over assignments (small sizes)
    idx = list(range(nD))
    for k in range(min(nT, nD), 0, -1):
        found = False
        for ts in itertools.combinations(range(nT), k):
            for ds in itertools.permutations(idx, k):
                c = sum(C[t, d] for t, d in zip(ts, ds))
                if any(C[t, d] > gate for t, d in zip(ts, ds)):
                    continue
                if bestc is None or c < bestc:
                    bestc, best = c, dict(zip(ts, ds)); found = True
        if found:
            return best
    return {}


def track_targets(meta, frames, init=None, gate=0.07, log=None):
    """frames: list of manifest frames (ordered). Returns (positions[frame][track], final tracks)."""
    hist = []
    tracks = None; vel = None; missing = None
    for fi, f in enumerate(frames):
        files = {(x["camera"], x["kind"]): x["path"] for x in f["files"]}
        rgb = np.asarray(Image.open(WS / files[("front", "rgb")]).convert("RGB"))
        dep = np.asarray(Image.open(WS / files[("front", "depth")])).astype(np.float32) * 0.001
        D = detect_targets_img(meta, rgb, dep)
        if tracks is None:
            tracks = [d.copy() for d in (init if init is not None else D)]
            vel = [np.zeros(2) for _ in tracks]; missing = [0 for _ in tracks]
            hist.append([t.copy() for t in tracks])
            continue
        pred = [t + v for t, v in zip(tracks, vel)]
        A = best_assignment(pred, D, gate)
        usedD = set(A.values())
        newt = [t.copy() for t in tracks]
        for ti, di in A.items():
            nv = D[di] - tracks[ti]
            vel[ti] = 0.6 * nv + 0.4 * vel[ti] if missing[ti] == 0 else np.zeros(2)
            newt[ti] = D[di].copy(); missing[ti] = 0
        lostT = [ti for ti in range(len(tracks)) if ti not in A]
        freeD = [di for di in range(len(D)) if di not in usedD
                 and min(np.linalg.norm(D[di] - t) for t in newt) > 0.05]
        if lostT and freeD:
            A2 = best_assignment([tracks[t] for t in lostT], [D[d] for d in freeD], 1.0)
            for a, b in A2.items():
                ti, di = lostT[a], freeD[b]
                if log:
                    log(f"frame {fi}: re-acquire track {ti} {np.round(tracks[ti],3)} -> {np.round(D[di],3)}")
                newt[ti] = D[di].copy(); vel[ti] = np.zeros(2); missing[ti] = 0
        for ti in range(len(tracks)):
            if ti not in A and missing[ti] >= 0 and not np.array_equal(newt[ti], tracks[ti]) is False:
                pass
        for ti in lostT:
            if np.array_equal(newt[ti], tracks[ti]):
                missing[ti] += 1
                vel[ti] = vel[ti] * 0.5
                newt[ti] = tracks[ti] + vel[ti]
        tracks = newt
        hist.append([t.copy() for t in tracks])
    return hist, tracks


def frame_detections(meta, frames):
    out = []
    for f in frames:
        files = {(x["camera"], x["kind"]): x["path"] for x in f["files"]}
        rgb = np.asarray(Image.open(WS / files[("front", "rgb")]).convert("RGB"))
        dep = np.asarray(Image.open(WS / files[("front", "depth")])).astype(np.float32) * 0.001
        out.append(detect_targets_img(meta, rgb, dep))
    return out


def swap_events(slots, dets, slot_tol=0.03, vac_tol=0.015, gap=4, min_frames=3, log=None):
    """Detect swap events: runs of frames with in-transit ring detections; the swapped pair
    = the two slots most often vacant during the run. Returns list of dict(start,end,pair,vac)."""
    n = len(slots)
    transit = []
    vac = []
    for D in dets:
        t = 0
        for d in D:
            if min(np.linalg.norm(d - s) for s in slots) > slot_tol:
                t += 1
        transit.append(t)
        vac.append([all(np.linalg.norm(d - s) > vac_tol for d in D) for s in slots])
    runs = []
    cur = None
    for fi, t in enumerate(transit):
        if t:
            if cur is not None and fi - cur[1] <= gap:
                cur[1] = fi
            else:
                if cur is not None:
                    runs.append(cur)
                cur = [fi, fi]
    if cur is not None:
        runs.append(cur)
    out = []
    for a, b in runs:
        if b - a + 1 < min_frames:
            continue
        frac = np.mean([vac[f] for f in range(a, b + 1)], axis=0)
        order = np.argsort(-frac)
        p = tuple(sorted(int(x) for x in order[:2]))
        # symmetry votes: swapping pair moves point-symmetrically about its midpoint
        votes = {}
        for f in range(a, b + 1):
            T = [d for d in dets[f] if min(np.linalg.norm(d - s) for s in slots) > slot_tol]
            for u in range(len(T)):
                for w in range(u + 1, len(T)):
                    m = (T[u] + T[w]) / 2
                    best = None
                    for i in range(n):
                        for j in range(i + 1, n):
                            e = np.linalg.norm(m - (slots[i] + slots[j]) / 2)
                            L = np.linalg.norm(slots[i] - slots[j]); l = np.linalg.norm(T[u] - T[w])
                            if l > L + 0.03 or l < 0.35 * L:
                                continue
                            if best is None or e < best[0]:
                                best = (e, (i, j))
                    if best is not None and best[0] < 0.02:
                        votes[best[1]] = votes.get(best[1], 0) + 1
        if not votes:
            if log:
                log(f"ignore transit run {a}-{b} (no symmetric votes) vac {np.round(frac,2).tolist()}")
            continue
        vp, vk = max(votes.items(), key=lambda t: t[1])
        if vk < 3:
            if log:
                log(f"ignore transit run {a}-{b} weak votes {votes}")
            continue
        p = vp
        e = dict(start=a, end=b, pair=p, vac=np.round(frac, 2).tolist(), votes=votes,
                 margin=float(frac[order[1]] - (frac[order[2]] if n > 2 else 0)))
        out.append(e)
        if log:
            log(f"swap {a}-{b} pair {p} vac {e['vac']}")
    return out


def identity_timeline(n, events, n_frames):
    """ident[f][slot] = identity at slot for frame f (swap applied at event end)."""
    ident = list(range(n))
    tl = []
    ev = sorted(events, key=lambda e: e["start"])
    k = 0
    for f in range(n_frames):
        while k < len(ev) and ev[k]["end"] < f:
            i, j = ev[k]["pair"]; ident[i], ident[j] = ident[j], ident[i]; k += 1
        tl.append(list(ident))
    return tl, ident


def phase_swaps(slots, dets, slot_tol=0.03, mid_tol=0.02, hi=np.radians(55), lo=np.radians(35), min_support=4, log=None):
    """Count swaps per slot pair from per-frame symmetric detection pairs.
    For a swapping pair (i,j) the two moving rings are point-symmetric about the pair midpoint;
    the line through them rotates 0 -> 90 -> 0 deg relative to the s_i-s_j line once per swap.
    Returns list of (frame, (i,j)) in order of the 90-deg crossing."""
    n = len(slots)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    state = {p: "low" for p in pairs}
    last_seen = {p: -99 for p in pairs}
    seen_frames = {}
    out = []
    for f, D in enumerate(dets):
        T = [d for d in D if min(np.linalg.norm(d - s) for s in slots) > slot_tol]
        cands = []
        for u in range(len(T)):
            for w in range(u + 1, len(T)):
                m = (T[u] + T[w]) / 2
                l = np.linalg.norm(T[u] - T[w])
                for (i, j) in pairs:
                    L = np.linalg.norm(slots[i] - slots[j])
                    if l < 0.3 * L:
                        continue
                    e = np.linalg.norm(m - (slots[i] + slots[j]) / 2)
                    if e < mid_tol:
                        a = slots[i] - slots[j]; b = T[u] - T[w]
                        c = abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
                        ang = np.arccos(np.clip(c, 0, 1))
                        cands.append((e, u, w, (i, j), ang))
        # choose the disjoint candidate set with max cardinality, then min total error
        cands.sort(key=lambda t: t[0])
        cands = cands[:14]
        best = (0, 0.0, [])
        def rec(k, used, usedp, chosen, err):
            nonlocal best
            if (len(chosen), -err) > (best[0], -best[1]):
                best = (len(chosen), err, list(chosen))
            for q in range(k, len(cands)):
                e, u, w, p, ang = cands[q]
                if u in used or w in used or p[0] in usedp or p[1] in usedp:
                    continue
                chosen.append(cands[q])
                rec(q + 1, used | {u, w}, usedp | {p[0], p[1]}, chosen, err + e)
                chosen.pop()
        rec(0, frozenset(), frozenset(), [], 0.0)
        for e, u, w, p, ang in best[2]:
            last_seen[p] = f
            seen_frames.setdefault(p, []).append(f)
            if state[p] == "low" and ang > hi:
                state[p] = "high"
                out.append((f, p))
                if log:
                    log(f"swap {p} at frame {f} ang {np.degrees(ang):.0f}")
            elif state[p] == "high" and ang < lo:
                state[p] = "low"
        # pairs not seen for a while (swap finished, rings back at slots) reset to low
        occ = [any(np.linalg.norm(d - s) < 0.015 for d in D) for s in slots]
        for p in pairs:
            if state[p] == "high" and last_seen[p] != f and occ[p[0]] and occ[p[1]]:
                state[p] = "low"
    good = []
    for f, p in out:
        k = sum(1 for g in seen_frames.get(p, []) if abs(g - f) <= 8)
        if k >= min_support:
            good.append((f, p))
        elif log:
            log(f"drop weak swap {p} at {f} support {k}")
    return good


def resolve_identity_slot(slots, swaps, place_frame, place_xy, tol=0.04):
    """Return (final slot index of the target identity placed at place_xy at place_frame, dist)."""
    d = [np.linalg.norm(s - place_xy) for s in slots]
    s0 = int(np.argmin(d))
    if d[s0] > tol:
        return None, d[s0]
    # identity at slot s0 at place_frame: apply swaps up to place_frame to identity map
    ident = list(range(len(slots)))   # ident[slot] = identity
    for f, (i, j) in swaps:
        if f <= place_frame:
            ident[i], ident[j] = ident[j], ident[i]
    idt = ident[s0]
    for f, (i, j) in swaps:
        if f > place_frame:
            ident[i], ident[j] = ident[j], ident[i]
    return ident.index(idt), d[s0]
