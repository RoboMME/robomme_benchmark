"""Track identical target discs through a front-camera video (they swap places)."""
import math
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


def ring_mask(a):
    r, g, b = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    return (r > 130) & (b > 95) & (g < r - 45) & (b > g + 12)


def detect(a, min_px=20, merge_px=6.0):
    m = ring_mask(a)
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    comps = []
    for l in range(1, n + 1):
        ys, xs = np.nonzero(lab == l)
        if len(xs) >= 4:
            comps.append([xs.mean(), ys.mean(), len(xs)])
    # merge concentric pieces (inner/outer ring) of one disc
    merged = True
    while merged:
        merged = False
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                a_, b_ = comps[i], comps[j]
                if np.hypot(a_[0] - b_[0], a_[1] - b_[1]) < merge_px:
                    w = a_[2] + b_[2]
                    comps[i] = [(a_[0] * a_[2] + b_[0] * b_[2]) / w, (a_[1] * a_[2] + b_[1] * b_[2]) / w, w]
                    del comps[j]; merged = True; break
            if merged:
                break
    return [tuple(c) for c in comps if c[2] >= min_px]


class Tracker:
    def __init__(self, gate=22.0):
        self.gate = gate
        self.pos = None; self.vel = None; self.hist = []

    def init(self, dets):
        self.pos = np.array([[d[0], d[1]] for d in dets], float)
        self.vel = np.zeros_like(self.pos)
        self.miss = np.zeros(len(self.pos), int)
        self.hist.append(self.pos.copy())

    def update(self, dets, dt=1.0):
        K = len(self.pos)
        pred = self.pos + self.vel * dt
        if dets:
            D = np.array([[d[0], d[1]] for d in dets], float)
            C = np.linalg.norm(pred[:, None, :] - D[None, :, :], axis=2)
            ri, ci = linear_sum_assignment(C)
            matched = set()
            for i, j in zip(ri, ci):
                if C[i, j] < self.gate:
                    newp = D[j]
                    self.vel[i] = 0.5 * self.vel[i] + 0.5 * (newp - self.pos[i]) / dt
                    self.pos[i] = newp; self.miss[i] = 0; matched.add(i)
            for i in range(K):
                if i not in matched:
                    self.miss[i] += 1
                    if self.miss[i] <= 3:
                        self.pos[i] = pred[i]
                    self.vel[i] *= 0.5
        else:
            self.miss += 1
        self.hist.append(self.pos.copy())
        return self.pos


def run(frames_rgb, k_expected=None):
    """frames_rgb: list of HxWx3 arrays. Returns (tracker, list of positions per frame)."""
    dets0 = [detect(a) for a in frames_rgb[:5]]
    best = max(range(len(dets0)), key=lambda i: len(dets0[i]))
    tr = Tracker()
    tr.init(dets0[0] if len(dets0[0]) >= len(dets0[best]) else dets0[best])
    tr.hist = [tr.pos.copy()]
    for a in frames_rgb[1:]:
        tr.update(detect(a))
    return tr


def find_homes(dets_list, min_frac=0.35, rad=4.0):
    pts = [(d[0], d[1]) for D in dets_list for d in D if d[2] >= 60]
    if not pts:
        return np.zeros((0, 2))
    P = np.array(pts)
    homes = []
    used = np.zeros(len(P), bool)
    # greedy density clustering
    while True:
        best, bi = 0, -1
        for i in np.nonzero(~used)[0][::max(1, len(P) // 3000)]:
            c = np.sum((~used) & (np.linalg.norm(P - P[i], axis=1) < rad))
            if c > best:
                best, bi = c, i
        if bi < 0 or best < min_frac * len(dets_list):
            break
        sel = (~used) & (np.linalg.norm(P - P[bi], axis=1) < rad)
        homes.append(P[sel].mean(0))
        used |= np.linalg.norm(P - P[bi], axis=1) < 2.5 * rad
    return np.array(homes)


def swap_events(dets_list, homes, off_px=6.0, min_size=60, min_len=3, max_gap=2):
    """Return list of (start, end, (i, j)) home-index pairs that exchanged targets."""
    off = []
    for D in dets_list:
        o = []
        for d in D:
            if d[2] < min_size:
                continue
            dist = np.linalg.norm(homes - np.array(d[:2]), axis=1)
            if dist.min() > off_px:
                o.append(np.array(d[:2]))
        off.append(o)
    events = []
    t = 0; n = len(off)
    while t < n:
        if off[t]:
            s = t; e = t; gap = 0
            while e + 1 < n and (off[e + 1] or gap < max_gap):
                if off[e + 1]:
                    gap = 0
                else:
                    gap += 1
                e += 1
            while e > s and not off[e]:
                e -= 1
            if e - s + 1 >= min_len:
                votes = {}
                for k in list(range(s, min(s + 2, e + 1))) + list(range(max(s, e - 1), e + 1)):
                    for p in off[k]:
                        h = int(np.argmin(np.linalg.norm(homes - p, axis=1)))
                        votes[h] = votes.get(h, 0) + 1
                pair = sorted(votes, key=lambda h: -votes[h])[:2]
                events.append((s, e, tuple(sorted(pair)), votes))
            t = e + 1
        else:
            t += 1
    return events


# ------------------------------------------------------------ geometry
from scipy.spatial.transform import Rotation as _R

FRONT_POS = np.array([0.30000004079654374, 0.0, 0.4000000783857871])
FRONT_Q = [0.6881909515745244, 0.1624598848052982, 0.16245988480529822, 0.6881909515745244]
FRONT_K = (128.0, 128.0, 128.0, 128.0)


def pix_to_plane(u, v, z=0.0, pos=FRONT_POS, q=FRONT_Q, K=FRONT_K):
    fx, fy, cx, cy = K
    Rm = _R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    dcam = np.array([(u - cx) / fx, -(v - cy) / fy, -1.0])
    dw = Rm @ dcam
    t = (z - pos[2]) / dw[2]
    return pos + t * dw


def plane_to_pix(p, pos=FRONT_POS, q=FRONT_Q, K=FRONT_K):
    fx, fy, cx, cy = K
    Rm = _R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    c = Rm.T @ (np.asarray(p, float) - pos)
    d = -c[2]
    return np.array([cx + fx * c[0] / d, cy - fy * c[1] / d])


def world_dets(dets):
    return [[(pix_to_plane(d[0], d[1])[:2], d[2]) for d in D] for D in dets]


def find_homes_w(wd, min_frac=0.2, rad=0.008, min_size=25):
    pts = np.array([p for D in wd for (p, s) in D if s >= min_size])
    if len(pts) == 0:
        return np.zeros((0, 2))
    homes = []; alive = np.ones(len(pts), bool)
    while alive.any():
        P = pts[alive]
        sub = P[::max(1, len(P) // 1500)]
        cnt = [(np.linalg.norm(P - q, axis=1) < rad).sum() for q in sub]
        k = int(np.argmax(cnt))
        if cnt[k] < min_frac * len(wd):
            break
        c = P[np.linalg.norm(P - sub[k], axis=1) < rad].mean(0)
        homes.append(c)
        alive &= np.linalg.norm(pts - c, axis=1) > 3 * rad
    return np.array(homes)


def swap_events_w(wd, homes, off_m=0.02, res_m=0.012, min_support=3, max_gap=3, min_size=30):
    n = len(wd); K = len(homes)
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    mids = np.array([(homes[i] + homes[j]) / 2 for i, j in pairs])
    rads = np.array([np.linalg.norm(homes[i] - homes[j]) / 2 for i, j in pairs])
    per_frame = []
    for D in wd:
        hits = []
        for p, s in D:
            if s < min_size:
                continue
            dh = np.linalg.norm(homes - p, axis=1)
            if dh.min() < off_m:
                continue
            res = np.abs(np.linalg.norm(mids - p, axis=1) - rads)
            k = int(np.argmin(res))
            if res[k] < res_m:
                hits.append(pairs[k])
        per_frame.append(hits)
    # group into events
    events = []
    t = 0
    while t < n:
        if per_frame[t]:
            s = t; e = t; last = t
            while e + 1 < n and (e + 1 - last) <= max_gap:
                e += 1
                if per_frame[e]:
                    last = e
            e = last
            sup = {}
            for k in range(s, e + 1):
                for pr in set(per_frame[k]):
                    sup[pr] = sup.get(pr, 0) + 1
            chosen = []
            for pr, c in sorted(sup.items(), key=lambda x: -x[1]):
                if c >= min_support and not any(set(pr) & set(q) for q in chosen):
                    chosen.append(pr)
            if chosen:
                events.append(dict(s=s, e=e, pairs=chosen, support=sup))
            t = e + 1
        else:
            t += 1
    return events


def follow(home_idx, events, from_frame):
    cur = home_idx
    for ev in events:
        if ev["s"] < from_frame:
            # swap in progress at from_frame counts only if it started after placement
            continue
        for a, b in ev["pairs"]:
            if cur == a:
                cur = b; break
            if cur == b:
                cur = a; break
    return cur


def swap_hits(wd, homes, src, off_m=0.02, res_m=0.012, margin=0.006, min_size=30):
    K = len(homes)
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    mids = np.array([(homes[i] + homes[j]) / 2 for i, j in pairs])
    rads = np.array([np.linalg.norm(homes[i] - homes[j]) / 2 for i, j in pairs])
    th_i = np.array([math.atan2(*(homes[i] - mids[k])[::-1]) for k, (i, j) in enumerate(pairs)])
    hits = []
    for t, D in enumerate(wd):
        offs = []
        for p, s in D:
            if s < min_size:
                continue
            if np.linalg.norm(homes - p, axis=1).min() < off_m:
                continue
            offs.append(p)
        used = set()
        # strong: two blobs symmetric about a pair midpoint
        for a in range(len(offs)):
            for b in range(a + 1, len(offs)):
                m_ = (offs[a] + offs[b]) / 2; r_ = np.linalg.norm(offs[a] - offs[b]) / 2
                err = np.linalg.norm(mids - m_, axis=1) + np.abs(rads - r_)
                k = int(np.argmin(err))
                if err[k] < 0.02:
                    th = math.atan2(*(offs[a] - mids[k])[::-1])
                    psi = (th - th_i[k]) % math.pi
                    hits.append((src[t], pairs[k], psi, 2)); used |= {a, b}
        for a in range(len(offs)):
            if a in used:
                continue
            res = np.abs(np.linalg.norm(mids - offs[a], axis=1) - rads)
            o = np.argsort(res)
            if res[o[0]] < res_m and (len(o) < 2 or res[o[1]] > res[o[0]] + margin):
                k = int(o[0])
                th = math.atan2(*(offs[a] - mids[k])[::-1])
                psi = (th - th_i[k]) % math.pi
                hits.append((src[t], pairs[k], psi, 1))
    return hits


def swap_instances(hits, max_gap=24, reset=1.2, min_score=3, min_span=0.8):
    by = {}
    for h in hits:
        by.setdefault(h[1], []).append(h)
    inst = []
    for pr, hs in by.items():
        hs.sort(key=lambda h: h[0])
        cur = [hs[0]]
        groups = []
        for h in hs[1:]:
            prev = cur[-1]
            dpsi = abs(h[2] - prev[2])
            dpsi = min(dpsi, math.pi - dpsi) if False else dpsi
            if h[0] - prev[0] > max_gap or (dpsi > reset and h[0] - prev[0] > 2):
                groups.append(cur); cur = [h]
            else:
                cur.append(h)
        groups.append(cur)
        for g in groups:
            score = sum(h[3] for h in g)
            ps = [h[2] for h in g]
            span = max(ps) - min(ps)
            if score >= min_score and span >= min_span:
                inst.append(dict(s=g[0][0], e=g[-1][0], t=float(np.mean([h[0] for h in g])), pair=pr, score=score, span=round(span, 2)))
    inst.sort(key=lambda d: d["t"])
    return inst


def follow2(home_idx, inst, from_src):
    cur = home_idx
    for d in inst:
        if d["e"] <= from_src:
            continue
        a, b = d["pair"]
        if cur == a:
            cur = b
        elif cur == b:
            cur = a
    return cur


def track_world(wd, homes, src, gate=0.06, min_size=30, coast=6):
    """Hungarian tracking of K targets initialised at homes. Returns hist (T,K,2)."""
    K = len(homes)
    pos = homes.copy(); vel = np.zeros_like(pos); miss = np.zeros(K, int)
    hist = [pos.copy()]
    for t in range(1, len(wd)):
        dt = max(1, src[t] - src[t - 1])
        D = [p for p, s in wd[t] if s >= min_size]
        pred = pos + vel * dt
        # occluded/stationary tracks drift toward nearest home
        if D:
            D = np.array(D)
            C = np.linalg.norm(pred[:, None] - D[None], axis=2)
            ri, ci = linear_sum_assignment(C)
            got = set()
            for i, j in zip(ri, ci):
                if C[i, j] < gate:
                    v = (D[j] - pos[i]) / dt
                    vel[i] = 0.6 * v + 0.4 * vel[i]
                    pos[i] = D[j]; miss[i] = 0; got.add(i)
            for i in range(K):
                if i not in got:
                    miss[i] += 1
                    if miss[i] <= coast:
                        pos[i] = pred[i]
                    else:
                        vel[i] = 0
        else:
            miss += 1
            pos = np.where((miss <= coast)[:, None], pred, pos)
        hist.append(pos.copy())
    return np.array(hist)


BMINOR = 0.17


def fit_swaps(wd, homes, src, T=33.0, tol=0.025, off_m=0.025, min_size=15,
              min_info=None, min_ratio=0.35, min_match=None):
    K = len(homes)
    src = np.asarray(src, float)
    if min_info is None or min_match is None:
        stride = float(np.median(np.diff(src))) if len(src) > 1 else 1.0
        expected = 2.0 * 0.7 * T / max(stride, 1.0)
        if min_info is None:
            min_info = max(3.0, 0.25 * expected)
        if min_match is None:
            min_match = max(3.0, 0.25 * expected)
    t0_lo, t0_hi = int(src[0] - T), int(src[-1])
    t0s = np.arange(t0_lo, t0_hi + 1)
    hyps = []
    det = [np.array([p for p, s in D if s >= min_size]).reshape(-1, 2) for D in wd]
    for i in range(K):
        for j in range(i + 1, K):
            mid = (homes[i] + homes[j]) / 2; r = np.linalg.norm(homes[i] - homes[j]) / 2
            u = (homes[i] - mid) / max(r, 1e-6); vv = np.array([-u[1], u[0]])
            bminor = min(r, BMINOR)
            thi = 0.0
            for dr in (-1, 1):
                match = np.zeros(len(t0s)); info = np.zeros(len(t0s))
                for t, st in enumerate(src):
                    # candidate starts for which this frame is inside the window
                    k0 = np.searchsorted(t0s, st - T); k1 = np.searchsorted(t0s, st, side="right")
                    if k1 <= k0:
                        continue
                    prog = (st - t0s[k0:k1]) / T  # 0..1
                    ang = thi + dr * math.pi * prog
                    for off in (0.0, math.pi):
                        P = mid + (r * np.cos(ang + off))[:, None] * u[None] + (bminor * np.sin(ang + off))[:, None] * vv[None]
                        dh = np.linalg.norm(P[:, None] - homes[None], axis=2).min(1)
                        inf = dh > off_m
                        info[k0:k1] += inf
                        if len(det[t]):
                            dd = np.linalg.norm(P[:, None] - det[t][None], axis=2).min(1)
                            match[k0:k1] += inf & (dd < tol)
                ok = (info >= min_info) & (match >= min_match) & (match >= min_ratio * np.maximum(info, 1))
                for k in np.nonzero(ok)[0]:
                    hyps.append((float(match[k]), float(match[k] / info[k]), int(t0s[k]), (i, j), dr))
    # non-maximum suppression
    hyps.sort(key=lambda h: (-h[0], -h[1]))
    acc = []
    for h in hyps:
        clash = False
        for a in acc:
            if abs(a[2] - h[2]) < T and (set(a[3]) & set(h[3])):
                clash = True; break
        if not clash:
            acc.append(h)
    acc.sort(key=lambda h: h[2])
    return [dict(t0=h[2], pair=h[3], dir=h[4], match=h[0], ratio=round(h[1], 2)) for h in acc]


def follow3(home_idx, swaps, from_src, T=33.0):
    cur = home_idx
    for d in swaps:
        if d["t0"] + T / 2 <= from_src:
            continue
        a, b = d["pair"]
        if cur == a:
            cur = b
        elif cur == b:
            cur = a
    return cur
