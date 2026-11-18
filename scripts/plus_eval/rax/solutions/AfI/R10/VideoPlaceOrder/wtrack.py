"""World-space tracking of identical targets through the conditioning video."""
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R
from pathlib import Path

WS = Path("/workspace")

def target_mask(im):
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return (r > 120) & (b > 150) & (g < 130) & (b - g > 60)

def frame_points(f, max_pts=4000):
    files = {(x["camera"], x["kind"]): x["path"] for x in f["files"]}
    im = np.array(Image.open(WS / files[("front", "rgb")]).convert("RGB")).astype(int)
    dp = np.array(Image.open(WS / files[("front", "depth")])).astype(float) * 0.001
    K = f["camera_intrinsics"]["front"]["matrix"]
    pose = f["camera_poses"]["front"]
    m = target_mask(im) & (dp > 0)
    vs, us = np.nonzero(m)
    d = dp[vs, us]
    pc = np.stack([(us - K[0][2]) * d / K[0][0], -(vs - K[1][2]) * d / K[1][1], -d], axis=1)
    w, x, y, z = pose["quaternion_wxyz"]
    P = R.from_quat([x, y, z, w]).apply(pc) + np.array(pose["position_m"])
    P = P[P[:, 2] < 0.03]
    if len(P) > max_pts:
        P = P[np.linspace(0, len(P) - 1, max_pts).astype(int)]
    return P[:, :2]

def init_centers(P, link=0.02, min_pts=15):
    """Connected clustering of points (single-link via grid label)."""
    if len(P) == 0:
        return np.zeros((0, 2)), []
    res = 0.004
    mn = P.min(0)
    ij = ((P - mn) / res).astype(int)
    G = np.zeros(ij.max(0) + 1, bool)
    G[ij[:, 0], ij[:, 1]] = True
    G2 = ndimage.binary_dilation(G, iterations=int(link / res / 2) + 1)
    lab, n = ndimage.label(G2)
    L = lab[ij[:, 0], ij[:, 1]]
    cs, ns = [], []
    for j in range(1, n + 1):
        s = P[L == j]
        if len(s) >= min_pts:
            cs.append(s.mean(0)); ns.append(len(s))
    return np.array(cs).reshape(-1, 2), ns

def track(frames, log=print, radius=0.045, min_obs=15, merge_d=0.03):
    frames = sorted(frames, key=lambda f: f["frame_index"])
    pts = [frame_points(f) for f in frames]
    cl = [init_centers(P) for P in pts]
    counts = [len(c[0]) for c in cl]
    vals, cnts = np.unique(counts, return_counts=True)
    N = int(vals[np.argmax(cnts)])
    log("target count", N, "hist", dict(zip(vals.tolist(), cnts.tolist())))
    start = next(i for i, c in enumerate(cl) if len(c[0]) == N)
    pos = cl[start][0].copy()
    vel = np.zeros_like(pos)
    hist = {frames[i]["frame_index"]: pos.copy() for i in range(start + 1)}
    flags = {}
    for i in range(start + 1, len(frames)):
        P = pts[i]
        pred = pos + vel
        c = pred.copy()
        nobs = np.zeros(N, int)
        if len(P):
            for it in range(4):
                D = np.linalg.norm(P[:, None, :] - c[None], axis=2)
                a = D.argmin(1)
                ok = D[np.arange(len(P)), a] < radius
                for k in range(N):
                    s = P[ok & (a == k)]
                    nobs[k] = len(s)
                    if len(s) >= min_obs:
                        c[k] = s.mean(0)
        observed = nobs >= min_obs
        merged = np.zeros(N, bool)
        for a_ in range(N):
            for b_ in range(a_ + 1, N):
                if observed[a_] and observed[b_] and np.linalg.norm(c[a_] - c[b_]) < merge_d:
                    merged[a_] = merged[b_] = True
        # also: a cluster blob that is large and one neighbour unobserved => merge-like; handled by coast
        for k in range(N):
            if observed[k] and not merged[k]:
                nv = c[k] - pos[k]
                vel[k] = 0.6 * nv + 0.4 * vel[k]
                pos[k] = c[k]
            else:
                pos[k] = pred[k]
                vel[k] = vel[k] * (0.9 if merged[k] else 0.7)
        # re-identify against clean clustered detections when all N present & separated
        hist[frames[i]["frame_index"]] = pos.copy()
        flags[frames[i]["frame_index"]] = (observed.sum(), merged.sum())
    return hist, N, cl, frames

def rest_slots(frames, cl, N):
    """Slots = cluster centers in the final execution frame (or last frame with N clusters)."""
    for i in range(len(frames) - 1, -1, -1):
        if len(cl[i][0]) == N:
            return cl[i][0]
    return None

def track2(frames, log=print, radius=0.045, min_obs=15, merge_d=0.03, rest_thr=0.025):
    """Slot-snapping tracker. Returns (events, N, slots, occupancy history).
    occ[frame_index] = array slot->track id (identity), valid at rest frames."""
    frames = sorted(frames, key=lambda f: f["frame_index"])
    pts = [frame_points(f) for f in frames]
    cl = [init_centers(P) for P in pts]
    counts = [len(c[0]) for c in cl]
    vals, cnts = np.unique(counts, return_counts=True)
    N = int(vals[np.argmax(cnts)])
    log("target count", N, "hist", dict(zip(vals.tolist(), cnts.tolist())))
    # slots: from first frame with N clusters
    start = next(i for i, c in enumerate(cl) if len(c[0]) == N)
    slots = cl[start][0].copy()
    # track identity = slot index at start
    pos = slots.copy()
    vel = np.zeros_like(pos)
    slot_of = np.arange(N)  # track k -> slot
    moving_prev = False
    intervals = []
    cur = None
    occ = {}
    for i in range(len(frames)):
        fi = frames[i]["frame_index"]
        c0 = cl[i][0]
        if len(c0):
            dmin = np.linalg.norm(c0[:, None] - slots[None], axis=2).min(1)
            rest = bool((dmin < rest_thr).all())
        else:
            rest = True
        if i <= start:
            rest = True
        if rest:
            if moving_prev:
                # snap tracks to slots by Hungarian
                C = np.linalg.norm(pos[:, None] - slots[None], axis=2)
                ri, ci = linear_sum_assignment(C)
                new = np.empty(N, int)
                new[ri] = ci
                log("motion %d-%d: track->slot %s (snap err max %.3f)" % (cur[0], cur[1], new.tolist(), C[ri, ci].max()))
                intervals.append((cur[0], cur[1], slot_of.copy(), new.copy(), float(C[ri, ci].max())))
                slot_of = new
                cur = None
            pos = slots[slot_of].copy()
            vel[:] = 0
            moving_prev = False
        else:
            if cur is None:
                cur = [fi, fi]
            cur[1] = fi
            moving_prev = True
            P = pts[i]
            pred = pos + vel
            c = pred.copy()
            nobs = np.zeros(N, int)
            if len(P):
                for it in range(4):
                    D = np.linalg.norm(P[:, None, :] - c[None], axis=2)
                    a = D.argmin(1)
                    ok = D[np.arange(len(P)), a] < radius
                    for k in range(N):
                        s = P[ok & (a == k)]
                        nobs[k] = len(s)
                        if len(s) >= min_obs:
                            c[k] = s.mean(0)
            observed = nobs >= min_obs
            merged = np.zeros(N, bool)
            for a_ in range(N):
                for b_ in range(a_ + 1, N):
                    if observed[a_] and observed[b_] and np.linalg.norm(c[a_] - c[b_]) < merge_d:
                        merged[a_] = merged[b_] = True
            for k in range(N):
                if observed[k] and not merged[k]:
                    vel[k] = 0.6 * (c[k] - pos[k]) + 0.4 * vel[k]
                    pos[k] = c[k]
                else:
                    pos[k] = pred[k]
                    vel[k] = vel[k] * (0.9 if merged[k] else 0.7)
        occ[fi] = slot_of.copy()
    if moving_prev:
        C = np.linalg.norm(pos[:, None] - slots[None], axis=2)
        ri, ci = linear_sum_assignment(C)
        new = np.empty(N, int); new[ri] = ci
        log("motion %s (unfinished): track->slot %s" % (cur, new.tolist()))
        intervals.append((cur[0], cur[1], slot_of.copy(), new.copy(), float(C[ri, ci].max())))
        slot_of = new
        occ[frames[-1]["frame_index"]] = slot_of.copy()
    return dict(N=N, slots=slots, intervals=intervals, occ=occ, final_slot_of=slot_of, frames=frames)

def track3(frames, log=print, gate=0.07, rest_thr=0.025):
    """Detection-level tracker with merge coasting and slot snapping."""
    frames = sorted(frames, key=lambda f: f["frame_index"])
    pts = [frame_points(f) for f in frames]
    cl = [init_centers(P) for P in pts]
    counts = [len(c[0]) for c in cl]
    vals, cnts = np.unique(counts, return_counts=True)
    N = int(vals[np.argmax(cnts)])
    log("target count", N, "hist", dict(zip(vals.tolist(), cnts.tolist())))
    start = next(i for i, c in enumerate(cl) if len(c[0]) == N)
    slots = cl[start][0].copy()
    size = np.array(cl[start][1], float)
    pos = slots.copy(); vel = np.zeros_like(pos)
    slot_of = np.arange(N)
    moving_prev = False; cur = None; intervals = []; occ = {}; coast = np.zeros(N, int)
    for i in range(len(frames)):
        fi = frames[i]["frame_index"]
        D, ns = cl[i][0], np.array(cl[i][1], float)
        if len(D):
            dmin = np.linalg.norm(D[:, None] - slots[None], axis=2).min(1)
            rest = bool((dmin < rest_thr).all())
        else:
            rest = True
        if i <= start:
            rest = True
        if rest:
            if moving_prev:
                C = np.linalg.norm(pos[:, None] - slots[None], axis=2)
                ri, ci = linear_sum_assignment(C)
                new = np.empty(N, int); new[ri] = ci
                log("motion %d-%d: track->slot %s (snap err max %.3f)" % (cur[0], cur[1], new.tolist(), C[ri, ci].max()))
                intervals.append((cur[0], cur[1], slot_of.copy(), new.copy(), float(C[ri, ci].max())))
                slot_of = new; cur = None
            pos = slots[slot_of].copy(); vel[:] = 0; coast[:] = 0
            moving_prev = False
        else:
            if cur is None:
                cur = [fi, fi]
            cur[1] = fi
            moving_prev = True
            pred = pos + vel
            assigned = -np.ones(N, int)
            if len(D):
                Cm = np.linalg.norm(pred[:, None] - D[None], axis=2)
                ri, ci = linear_sum_assignment(Cm)
                for a, b in zip(ri, ci):
                    if Cm[a, b] < gate:
                        assigned[a] = b
                # merge detection: unassigned track whose prediction is near an assigned big blob
                for a in range(N):
                    if assigned[a] >= 0:
                        continue
                    if len(D) == 0:
                        continue
                    b = int(np.argmin(Cm[a]))
                    if Cm[a, b] < gate:
                        owners = [t for t in range(N) if assigned[t] == b]
                        for t in owners:
                            if ns[b] > 1.4 * size[t]:
                                assigned[t] = -2  # merged: coast
                        assigned[a] = -2
            for k in range(N):
                if assigned[k] >= 0:
                    nv = D[assigned[k]] - pos[k]
                    vel[k] = 0.6 * nv + 0.4 * vel[k]
                    pos[k] = D[assigned[k]]
                    size[k] = 0.8 * size[k] + 0.2 * ns[assigned[k]]
                    coast[k] = 0
                else:
                    coast[k] += 1
                    pos[k] = pred[k]
                    if coast[k] > 6:
                        vel[k] *= 0.7
        occ[fi] = slot_of.copy()
    if moving_prev:
        C = np.linalg.norm(pos[:, None] - slots[None], axis=2)
        ri, ci = linear_sum_assignment(C)
        new = np.empty(N, int); new[ri] = ci
        log("motion %s (unfinished): track->slot %s" % (cur, new.tolist()))
        intervals.append((cur[0], cur[1], slot_of.copy(), new.copy(), float(C[ri, ci].max())))
        slot_of = new
        occ[frames[-1]["frame_index"]] = slot_of.copy()
    return dict(N=N, slots=slots, intervals=intervals, occ=occ, final_slot_of=slot_of, frames=frames)

def _rot(v, w):
    c, s = np.cos(w), np.sin(w)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

def track4(frames, log=print, gate=0.08, rest_thr=0.025, merge_r=0.055):
    """Constant-turn-rate tracker, all tracks touching a merged blob coast."""
    frames = sorted(frames, key=lambda f: f["frame_index"])
    pts = [frame_points(f) for f in frames]
    cl = [init_centers(P) for P in pts]
    counts = [len(c[0]) for c in cl]
    vals, cnts = np.unique(counts, return_counts=True)
    N = int(vals[np.argmax(cnts)])
    log("target count", N, "hist", dict(zip(vals.tolist(), cnts.tolist())))
    start = next(i for i, c in enumerate(cl) if len(c[0]) == N)
    slots = cl[start][0].copy()
    size = np.array(cl[start][1], float)
    pos = slots.copy(); vel = np.zeros_like(pos); om = np.zeros(N)
    slot_of = np.arange(N)
    moving_prev = False; cur = None; intervals = []; occ = {}; coast = np.zeros(N, int)
    for i in range(len(frames)):
        fi = frames[i]["frame_index"]
        D, ns = cl[i][0], np.array(cl[i][1], float)
        if len(D):
            dmin = np.linalg.norm(D[:, None] - slots[None], axis=2).min(1)
            rest = bool((dmin < rest_thr).all())
        else:
            rest = True
        if i <= start:
            rest = True
        if rest and moving_prev and coast.max() > 0:
            # movers still hidden: only accept rest if all N blobs visible
            rest = len(D) == N
        if rest:
            if moving_prev:
                C = np.linalg.norm(pos[:, None] - slots[None], axis=2)
                ri, ci = linear_sum_assignment(C)
                new = np.empty(N, int); new[ri] = ci
                log("motion %d-%d: track->slot %s (snap err max %.3f)" % (cur[0], cur[1], new.tolist(), C[ri, ci].max()))
                intervals.append((cur[0], cur[1], slot_of.copy(), new.copy(), float(C[ri, ci].max())))
                slot_of = new; cur = None
            pos = slots[slot_of].copy(); vel[:] = 0; om[:] = 0; coast[:] = 0
            moving_prev = False
            occ[fi] = slot_of.copy()
            continue
        if cur is None:
            cur = [fi, fi]
        cur[1] = fi
        moving_prev = True
        pred = np.array([pos[k] + _rot(vel[k], om[k]) for k in range(N)])
        assigned = -np.ones(N, int)
        if len(D):
            Cm = np.linalg.norm(pred[:, None] - D[None], axis=2)
            ri, ci = linear_sum_assignment(Cm)
            for a, b in zip(ri, ci):
                if Cm[a, b] < gate:
                    assigned[a] = b
            # merged blobs: >=2 tracks predicted within merge_r of a blob that is bigger than one target
            for b in range(len(D)):
                near = [t for t in range(N) if Cm[t, b] < merge_r]
                if len(near) >= 2 and ns[b] > 1.3 * min(size[t] for t in near):
                    for t in near:
                        assigned[t] = -2
        for k in range(N):
            if assigned[k] >= 0:
                nv = D[assigned[k]] - pos[k]
                if np.linalg.norm(vel[k]) > 0.005 and np.linalg.norm(nv) > 0.005:
                    w = np.arctan2(vel[k][0] * nv[1] - vel[k][1] * nv[0], vel[k] @ nv)
                    om[k] = 0.5 * om[k] + 0.5 * np.clip(w, -0.8, 0.8)
                vel[k] = 0.7 * nv + 0.3 * vel[k]
                pos[k] = D[assigned[k]]
                size[k] = 0.8 * size[k] + 0.2 * ns[assigned[k]]
                coast[k] = 0
            else:
                coast[k] += 1
                vel[k] = _rot(vel[k], om[k])
                pos[k] = pos[k] + vel[k]
                if coast[k] > 8:
                    vel[k] *= 0.6
        occ[fi] = slot_of.copy()
    if moving_prev:
        C = np.linalg.norm(pos[:, None] - slots[None], axis=2)
        ri, ci = linear_sum_assignment(C)
        new = np.empty(N, int); new[ri] = ci
        log("motion %s (unfinished): track->slot %s" % (cur, new.tolist()))
        intervals.append((cur[0], cur[1], slot_of.copy(), new.copy(), float(C[ri, ci].max())))
        slot_of = new
        occ[frames[-1]["frame_index"]] = slot_of.copy()
    return dict(N=N, slots=slots, intervals=intervals, occ=occ, final_slot_of=slot_of, frames=frames)
