"""Offline (batch) reverse-time bin tracking and cube->bin covering assignment."""
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

import perception as pc

GATE_BASE = 0.04
GATE_SPEED = 0.6       # m/s allowance
GATE_MAX = 0.20
COAST_DAMP = 0.7
VEL_ALPHA = 0.85
MERGE_EXT = 1.35
COVER_R1 = 0.05        # bin within this of the cube's last-seen XY right after it vanished
COVER_W1 = 1.5         # s
COVER_R2 = 0.04
COVER_W2 = 4.0


def bin_profile(dets):
    if not dets:
        return None
    ext = float(np.median([c.ext_long for c in dets]))
    hmax = float(np.median([c.hmax for c in dets]))
    cols = [c.color for c in dets if c.colorful > 0.3]
    color = None
    if cols and len(cols) >= max(1, len(dets) // 2 + (len(dets) % 2)):
        vals, cnt = np.unique(cols, return_counts=True)
        color = str(vals[int(np.argmax(cnt))])
    return {"ext": ext, "hmax": hmax, "color": color}


def frame_detections(res, button_xy, prof, button_r=None):
    button_r = pc.BUTTON_R if button_r is None else button_r
    """bin-like detections (xy of top centroid) in one analyzed frame."""
    out = []
    bc = prof["color"] if prof else None
    for c in res["comps"]:
        if button_xy is not None and math.hypot(c.top[0] - button_xy[0], c.top[1] - button_xy[1]) < button_r:
            continue
        if pc.is_cubelike(c) and (bc is None or c.color != bc):
            continue
        vert = c.hmax - c.hmin
        if c.robot:
            # floating bin (lifted during a swap / cover) - accept only if bin-sized
            if prof is None or c.hmax > pc.ROBOT_MAX_H:
                continue
            if not (0.65 * prof["ext"] <= c.ext_long <= 1.4 * prof["ext"]):
                continue
            if not (0.4 * prof["hmax"] <= vert <= 1.5 * prof["hmax"]):
                continue
        else:
            if c.ext_long < 0.025 or c.hmax < 0.015:
                continue
            if prof is not None and c.ext_long < 0.45 * prof["ext"]:
                continue
        out.append((np.array([c.top[0], c.top[1]]), float(c.ext_long)))
    return out


def reverse_track(frames, final_xy, button_xy, prof, log=None):
    """frames: list of dicts with 't', 'res'. final_xy: list of bin xy at the last frame.
    Returns hist: array [ntracks, nframes, 2] and matched mask [ntracks, nframes]."""
    K = len(frames)
    N = len(final_xy)
    hist = np.zeros((N, K, 2))
    matched = np.zeros((N, K), dtype=bool)
    pos = np.array(final_xy, dtype=float).reshape(N, 2)
    vel = np.zeros((N, 2))
    unmatched_time = np.zeros(N)
    hist[:, K - 1] = pos
    matched[:, K - 1] = True
    pext = prof["ext"] if prof else 0.07
    for k in range(K - 2, -1, -1):
        dt = max(1e-3, frames[k + 1]["t"] - frames[k]["t"])
        pred = pos - vel * dt
        alld = frame_detections(frames[k]["res"], button_xy, prof)
        dets = [d for d, e in alld if e <= MERGE_EXT * pext]
        merged = [(d, e) for d, e in alld if e > MERGE_EXT * pext]
        gate = min(GATE_MAX, GATE_BASE + GATE_SPEED * dt)
        assigned = -np.ones(N, dtype=int)
        if dets and N:
            D = np.array(dets)
            cost = np.linalg.norm(pred[:, None, :] - D[None, :, :], axis=2)
            big = 1e3
            cm = np.where(cost <= gate, cost, big)
            r, c = linear_sum_assignment(cm)
            for i, j in zip(r, c):
                if cm[i, j] < big:
                    assigned[i] = j
        for i in range(N):
            if assigned[i] >= 0:
                newp = np.array(dets[assigned[i]])
                v_new = (pos[i] - newp) / dt
                sp = np.linalg.norm(v_new)
                if sp > 1.5:
                    v_new *= 1.5 / sp
                vel[i] = (1 - VEL_ALPHA) * vel[i] + VEL_ALPHA * v_new
                pos[i] = newp
                unmatched_time[i] = 0.0
                matched[i, k] = True
            else:
                in_merge = any(np.linalg.norm(pred[i] - d) < 0.5 * e + 0.02 for d, e in merged)
                pos[i] = pred[i]
                if not in_merge:
                    vel[i] *= COAST_DAMP
                unmatched_time[i] += dt
                if unmatched_time[i] > 1.5:
                    vel[i] = 0.0
            hist[i, k] = pos[i]
    return hist, matched


def cube_tracks(frames, prof, button_xy):
    """color -> list of (k, pos, n) visible detections (largest blob per frame)."""
    out = {}
    bc = prof["color"] if prof else None
    for k, f in enumerate(frames):
        best = {}
        for b in pc.cube_blobs(f["res"], bin_color=bc, button_xy=button_xy):
            if b["color"] not in best or b["n"] > best[b["color"]]["n"]:
                best[b["color"]] = b
        for col, b in best.items():
            out.setdefault(col, []).append((k, np.array(b["pos"][:2]), b["n"]))
    return out


def assign_cubes(frames, hist, cubes, log=None):
    """color -> dict(track, how, last_k, last_pos)."""
    K = len(frames)
    times = np.array([f["t"] for f in frames])
    out = {}
    for col, obs in cubes.items():
        ks = [o[0] for o in obs]
        if not ks:
            continue
        # last robust visibility: last k that has another visible frame within 0.6 s before it, else last
        ks_sorted = sorted(ks)
        last = ks_sorted[-1]
        if len(ks_sorted) >= 3:
            for kk in reversed(ks_sorted):
                prev = [q for q in ks_sorted if q < kk and times[kk] - times[q] <= 0.6]
                if prev:
                    last = kk
                    break
        p = [o[1] for o in obs if o[0] == last][0]
        info = {"last_k": int(last), "last_t": float(times[last]), "last_pos": [float(p[0]), float(p[1])],
                "n_visible": len(ks), "track": None, "how": None, "dist": None}
        if last >= K - 1:
            info["how"] = "visible_at_end"
            out[col] = info
            continue
        for (W, R, how) in ((COVER_W1, COVER_R1, "cover1"), (COVER_W2, COVER_R2, "cover2")):
            sel = [k for k in range(last + 1, K) if times[k] - times[last] <= W]
            if not sel:
                sel = [last + 1]
            best = None
            dists = []
            for j in range(hist.shape[0]):
                d = float(np.min(np.linalg.norm(hist[j, sel] - p[None, :], axis=1)))
                dists.append(d)
            order = np.argsort(dists)
            j0 = int(order[0])
            if dists[j0] <= R:
                info["track"], info["how"], info["dist"] = j0, how, round(dists[j0], 4)
                if len(order) > 1:
                    info["second_dist"] = round(dists[int(order[1])], 4)
                break
        out[col] = info
    return out


def decide(targets, assignments, final_xy, cubes_last=None):
    """targets: list of color names. Returns list of (color, track_index or None, why)."""
    used = {a["track"] for a in assignments.values() if a.get("track") is not None}
    res = []
    for col in targets:
        a = assignments.get(col)
        if a and a.get("track") is not None:
            res.append((col, a["track"], a["how"]))
            continue
        # fallbacks
        free = [j for j in range(len(final_xy)) if j not in used]
        if a and a.get("last_pos") is not None:
            p = np.array(a["last_pos"])
            pool = free if free else list(range(len(final_xy)))
            if pool:
                j = min(pool, key=lambda j: float(np.linalg.norm(np.array(final_xy[j]) - p)))
                res.append((col, j, "nearest_last_seen" if a.get("how") != "visible_at_end" else "visible_at_end_nearest"))
                continue
        if free:
            res.append((col, free[0], "elimination" if len(free) == 1 else "guess"))
        elif len(final_xy):
            res.append((col, 0, "guess_any"))
        else:
            res.append((col, None, "no_bins"))
    return res
