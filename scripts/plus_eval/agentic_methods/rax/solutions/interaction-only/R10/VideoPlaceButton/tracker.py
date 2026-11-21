"""Track identical-looking targets through the conditioning video (they swap slots)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

WS = Path("/workspace")


def quat_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def ring_mask(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (b - g > 20) & (r > 100) & (b > 70)


def detect(rgb, depth_mm, intr, pose, min_px=6, merge=0.035):
    """Return list of (xy world, pixel count) ring detections on the table."""
    m = ring_mask(rgb)
    lab, n = ndimage.label(m)
    if n == 0:
        return []
    K = np.array(intr["matrix"], float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    R = quat_R(pose["quaternion_wxyz"])
    t = np.array(pose["position_m"], float)
    comps = []
    for k in range(1, n + 1):
        vs, us = np.nonzero(lab == k)
        if len(vs) < min_px:
            continue
        d = depth_mm[vs, us].astype(float) * 0.001
        ok = d > 0
        if ok.sum() < min_px:
            continue
        vs, us, d = vs[ok], us[ok], d[ok]
        pc = np.stack([(us + 0.5 - cx) * d / fx, -(vs + 0.5 - cy) * d / fy, -d], 1)
        pw = pc @ R.T + t
        keep = (pw[:, 2] > -0.01) & (pw[:, 2] < 0.03)
        if keep.sum() < min_px:
            continue
        pw = pw[keep]
        comps.append([pw[:, :2].mean(0) * len(pw), len(pw)])
    # merge components (outer ring + bullseye centre) within `merge` metres
    merged = True
    while merged and len(comps) > 1:
        merged = False
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                ci = comps[i][0] / comps[i][1]
                cj = comps[j][0] / comps[j][1]
                if np.linalg.norm(ci - cj) < merge:
                    comps[i] = [comps[i][0] + comps[j][0], comps[i][1] + comps[j][1]]
                    comps.pop(j)
                    merged = True
                    break
            if merged:
                break
    return [(c[0] / c[1], c[1]) for c in comps]


def load_frame(frame):
    files = {(f["camera"], f["kind"]): f["path"] for f in frame["files"]}
    rgb = np.array(Image.open(WS / files[("front", "rgb")]).convert("RGB"))
    dep = np.array(Image.open(WS / files[("front", "depth")]))
    return rgb, dep


def track(frames, log=print, gate=0.07):
    """Track targets through ordered frames. Returns dict:
    times (frame_index list), tracks: array [T, N, 2], slots: initial positions,
    observed: bool [T, N]."""
    frames = sorted(frames, key=lambda f: f["frame_index"])
    dets_all = []
    for f in frames:
        try:
            rgb, dep = load_frame(f)
            d = detect(rgb, dep, f["camera_intrinsics"]["front"], f["camera_poses"]["front"])
        except Exception as e:  # missing file etc.
            d = []
        dets_all.append(d)
    # number of targets: the mode of detection count over frames with large blobs
    counts = [len([x for x in d if x[1] >= 40]) for d in dets_all]
    vals, cnts = np.unique(counts, return_counts=True)
    # take the most common count among the first and last 10% frames (static scene)
    edge = counts[: max(1, len(counts) // 20)] + counts[-max(1, len(counts) // 20):]
    ev, ec = np.unique(edge, return_counts=True)
    N = int(ev[np.argmax(ec)])
    # init from first frame having N big detections
    init_i = next(i for i, c in enumerate(counts) if c == N)
    pos = np.array([x[0] for x in dets_all[init_i] if x[1] >= 40])
    vel = np.zeros_like(pos)
    T = len(frames)
    tr = np.zeros((T, N, 2))
    obs = np.zeros((T, N), bool)
    miss = np.zeros(N, int)
    for i in range(T):
        if i < init_i:
            tr[i] = pos
            continue
        d = [x for x in dets_all[i] if x[1] >= 8]
        pred = pos + vel
        if d:
            D = np.array([x[0] for x in d])
            C = np.linalg.norm(pred[:, None, :] - D[None, :, :], axis=2)
            ri, ci = linear_sum_assignment(C)
            matched = set()
            newpos = pred.copy()
            for a, b in zip(ri, ci):
                if C[a, b] < gate:
                    newpos[a] = D[b]
                    matched.add(a)
            for a in range(N):
                if a in matched:
                    vel[a] = 0.6 * vel[a] + 0.4 * (newpos[a] - pos[a])
                    miss[a] = 0
                    obs[i, a] = True
                else:
                    miss[a] += 1
                    vel[a] *= 0.8
            pos = newpos
        else:
            pos = pred
            vel *= 0.8
            miss += 1
        tr[i] = pos
    return {"times": [f["frame_index"] for f in frames], "tracks": tr, "observed": obs,
            "N": N, "init": init_i, "counts": counts}


def track_manifest(manifest, log=print):
    frames = [f for f in manifest["frames"] if f.get("phase") in ("video_demo", "execution")]
    return track(frames, log=log)


if __name__ == "__main__":
    import sys
    m = json.load(open(sys.argv[1]))
    res = track_manifest(m)
    tr, obs = res["tracks"], res["observed"]
    print("N", res["N"], "init", res["init"])
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    for i in range(0, len(tr), step):
        print(res["times"][i], res["counts"][i], " ".join("%s(%.3f,%.3f)" % ("" if obs[i, k] else "*", tr[i, k, 0], tr[i, k, 1]) for k in range(res["N"])))
    print("final", " ".join("(%.3f,%.3f)" % tuple(tr[-1, k]) for k in range(res["N"])))


# ---------------------------------------------------------------- slot/pair model
def _angle(v):
    return float(np.arctan2(v[1], v[0]))


def swap_analysis(frames, log=print, slot_tol=0.03):
    """Model: N fixed slots; targets exchange slots pairwise, the two movers staying
    point-symmetric about the pair midpoint while both slots are vacant.
    Returns slots and events (t0, t1, i, j, total rotation, k half turns)."""
    frames = sorted(frames, key=lambda f: f["frame_index"])
    times = [f["frame_index"] for f in frames]
    dets = []
    for f in frames:
        try:
            rgb, dep = load_frame(f)
            dets.append(detect(rgb, dep, f["camera_intrinsics"]["front"], f["camera_poses"]["front"]))
        except Exception:
            dets.append([])

    def big(d):
        return [x[0] for x in d if x[1] >= 40]
    last = big(dets[-1])
    first = big(dets[0])
    slots = np.array(last if len(last) >= len(first) else first)
    N = len(slots)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    mids = {p: (slots[p[0]] + slots[p[1]]) / 2 for p in pairs}
    dist = {p: float(np.linalg.norm(slots[p[0]] - slots[p[1]])) for p in pairs}
    T = len(dets)
    empty = []
    movers = []
    for d in dets:
        occ = set()
        mv = []
        for (xy, cnt) in d:
            ds = np.linalg.norm(slots - xy, axis=1)
            if ds.min() < slot_tol:
                occ.add(int(ds.argmin()))
            elif cnt >= 8:
                mv.append(np.asarray(xy))
        empty.append(set(range(N)) - occ)
        movers.append(mv)
    # per frame: pairs supported by two symmetric movers among empty slots
    support = [dict() for _ in range(T)]  # pair -> list of mover xy
    for t in range(T):
        mv = movers[t]
        cands = []
        for a in range(len(mv)):
            for b in range(a + 1, len(mv)):
                m = (mv[a] + mv[b]) / 2
                sep = np.linalg.norm(mv[a] - mv[b])
                for p in pairs:
                    if p[0] not in empty[t] or p[1] not in empty[t]:
                        continue
                    e = np.linalg.norm(m - mids[p])
                    if e < 0.035 and 0.4 * dist[p] < sep < 1.15 * dist[p]:
                        cands.append((e, a, b, p))
        cands.sort(key=lambda x: x[0])
        cands = cands[:12]
        best_sel = [()]
        best_key = [(0, 0.0)]

        def search(idx, sel, um, us, cost):
            key = (len(sel), -cost)
            if key > best_key[0]:
                best_key[0], best_sel[0] = key, tuple(sel)
            for q in range(idx, len(cands)):
                e, a, b, p = cands[q]
                if a in um or b in um or p[0] in us or p[1] in us:
                    continue
                search(q + 1, sel + [q], um | {a, b}, us | {p[0], p[1]}, cost + e)
        search(0, [], frozenset(), frozenset(), 0.0)
        used = set()
        for q in best_sel[0]:
            e, a, b, p = cands[q]
            used.update([a, b])
            support[t][p] = [mv[a], mv[b]]
        # single movers -> a pair whose both slots are empty and radius fits, preferring
        # pairs supported in neighbouring frames
        for a in range(len(mv)):
            if a in used:
                continue
            best, bs = None, -1
            for p in pairs:
                if p[0] not in empty[t] or p[1] not in empty[t]:
                    continue
                rr = np.linalg.norm(mv[a] - mids[p])
                if not (0.15 * dist[p] < rr < max(0.65 * dist[p], 0.5 * dist[p] + 0.06)):
                    continue
                score = sum(1 for u in range(max(0, t - 3), min(T, t + 4)) if p in support[u])
                if score > bs:
                    best, bs = p, score
            if best is not None and bs > 0:
                support[t].setdefault(best, []).append(mv[a])
    events = []
    for p in pairs:
        i, j = p
        # runs of frames with both slots empty
        t = 0
        while t < T:
            if not (i in empty[t] and j in empty[t]):
                t += 1
                continue
            t0 = t
            t1 = t
            while t < T:
                if i in empty[t] and j in empty[t]:
                    t1 = t
                    t += 1
                    continue
                # tolerate a short interruption (a mover passing over a slot)
                if t + 1 < T and i in empty[t + 1] and j in empty[t + 1] and t - t1 <= 1:
                    t += 1
                    continue
                break
            t = t1 + 1
            fr_sup = [u for u in range(t0, t1 + 1) if p in support[u]]
            if len(fr_sup) < 2:
                continue
            c = mids[p]
            a0 = _angle(slots[i] - c)
            prev, total = 0.0, 0.0
            for u in fr_sup:
                best = None
                for xy in support[u][p]:
                    val = np.mod(_angle(xy - c) - a0, np.pi)
                    dlt = (val - prev + np.pi / 2) % np.pi - np.pi / 2
                    if best is None or abs(dlt) < abs(best):
                        best = dlt
                total += best
                prev = (prev + best) % np.pi
            if abs(total) >= 0.35 * np.pi:
                k = int(np.sign(total)) * max(1, int(np.round(abs(total) / np.pi)))
            else:
                k = 0
            events.append({"t0": times[t0], "t1": times[t1], "i": i, "j": j,
                           "total": round(float(total), 3), "k": k, "n": len(fr_sup)})
    # a slot can take part in only one swap at a time: resolve overlaps
    events = [e for e in events if e["k"] != 0]
    events.sort(key=lambda e: -(e["n"] + abs(e["total"])))
    kept = []
    for e in events:
        clash = False
        for q in kept:
            if {e["i"], e["j"]} & {q["i"], q["j"]} and not (e["t1"] < q["t0"] or q["t1"] < e["t0"]):
                clash = True
                break
        if not clash:
            kept.append(e)
        else:
            log("drop overlapping event", e)
    events = kept
    events.sort(key=lambda e: e["t0"])
    for e in events:
        log("swap event", e)
    return slots, events


def target_slot_after(slots, events, slot_at_time, t_query, t_end=None):
    """Identity tracking: target in `slot_at_time` at time t_query -> its slot at the end."""
    s = slot_at_time
    for e in events:
        if e["t0"] < t_query:
            continue  # completed before (or ongoing at) the query; slot assignment already observed
        if t_end is not None and e["t0"] > t_end:
            continue
        if e["k"] % 2 != 0:
            if s == e["i"]:
                s = e["j"]
            elif s == e["j"]:
                s = e["i"]
    return s
