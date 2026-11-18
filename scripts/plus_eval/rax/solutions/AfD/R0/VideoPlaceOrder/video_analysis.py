"""Offline-developed analysis of the conditioning video for VideoPlaceOrder.

Pipeline: per-frame target-colour masks -> persistent HOME positions -> swap events
(pairs of targets exchange homes; mid-swap the two movers are point-symmetric about
the midpoint of their homes, and both homes are vacant) -> cube placement events ->
identity bookkeeping -> answer home.
"""
import numpy as np
from vision import label, blobs, cube_mask, closing, dilate

K_ASPECT = 1.3          # (legacy) image-space y stretch
# Ground-plane map for the fixed front camera (no roll): fitted offline from the target
# ellipse sizes (width grows linearly with row: horizon row ~ -190; circle aspect c ~ 355).
V_H = -190.0
C_ASP = 355.0
U_C = 128.0
G_SCALE = 300.0


def ground_inv(g):
    g = np.asarray(g, float)
    d = G_SCALE * C_ASP / g[..., 1]
    return np.stack([g[..., 0] * d / G_SCALE + U_C, d + V_H], -1)


def ground(p):
    p = np.asarray(p, float)
    d = p[..., 1] - V_H
    return np.stack([G_SCALE * (p[..., 0] - U_C) / d, G_SCALE * C_ASP / d], -1)
TABLE_TOP_V = 16        # rows above this are background wall


def target_mask(im):
    im = im.astype(np.int32)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    mx = np.maximum(np.maximum(R, G), B); mn = np.minimum(np.minimum(R, G), B)
    sat = mx - mn
    purple = (B - G > -8) & (R - G > 18) & (B > 60) & (R > 80)
    bluish = (B - G > 30) & (B - R > 25) & (R > 0.5 * B) & (B > 110)   # tinted target, not blue cube
    m = (purple | bluish) & (sat > 22)
    cubes = np.zeros_like(m)
    for c in ("red", "green", "blue"):
        cubes |= cube_mask(im, c)
    if cubes.any():
        m &= ~dilate(cubes, 2)
    m[:TABLE_TOP_V] = False
    return m


def table_mask(im):
    im = im.astype(np.int32)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    return (R > G + 15) & (G > B + 8) & (R > 60)


def find_homes(masks, min_support=0.45):
    P = np.mean(np.stack(masks).astype(np.float32), axis=0)
    pm = closing(P > min_support, 2)
    bl = blobs(pm, min_area=15, max_area=1500)
    homes = []
    for b in bl:
        if b["y"] < TABLE_TOP_V + 4 or b["w"] > 45 or b["h"] > 35:
            continue
        homes.append(dict(x=b["x"], y=b["y"], w=max(b["w"], 8), h=max(b["h"], 5), area=b["area"]))
    if homes:
        med = float(np.median([h["area"] for h in homes]))
        homes = [h for h in homes if h["area"] >= 0.3 * med]
    homes.sort(key=lambda h: (h["y"], h["x"]))
    return homes, P


def _integral(m):
    ii = np.zeros((m.shape[0] + 1, m.shape[1] + 1), np.float32)
    ii[1:, 1:] = np.cumsum(np.cumsum(m.astype(np.float32), 0), 1)
    return ii


def box_frac(ii, x, y, rx, ry):
    H, W = ii.shape[0] - 1, ii.shape[1] - 1
    x = np.atleast_1d(np.asarray(x, float)); y = np.atleast_1d(np.asarray(y, float))
    rx = np.broadcast_to(np.asarray(rx, float), x.shape); ry = np.broadcast_to(np.asarray(ry, float), x.shape)
    x0 = np.clip(np.round(x - rx).astype(int), 0, W); x1 = np.clip(np.round(x + rx).astype(int) + 1, 0, W)
    y0 = np.clip(np.round(y - ry).astype(int), 0, H); y1 = np.clip(np.round(y + ry).astype(int) + 1, 0, H)
    area = (x1 - x0) * (y1 - y0)
    s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    return np.where(area > 0, s / np.maximum(area, 1), 0.0)


class Analysis:
    def __init__(self, frames, times, log=print):
        self.log = log
        self.frames = frames
        self.times = np.asarray(times, float)
        self.masks = [target_mask(im) for im in frames]
        self.homes, self.P = find_homes(self.masks)
        n = len(self.homes)
        self.Hc = np.array([[h["x"], h["y"]] for h in self.homes], float).reshape(-1, 2)
        self.Hw = np.array([h["w"] for h in self.homes], float)
        self.Hh = np.array([h["h"] for h in self.homes], float)
        self.ref_area = float(np.median([h["area"] for h in self.homes])) if n else 100.0
        self.iis = [_integral(m) for m in self.masks]
        # home occupancy evidence per frame
        self.occ = np.array([box_frac(ii, self.Hc[:, 0], self.Hc[:, 1], 0.3 * self.Hw, 0.3 * self.Hh)
                             for ii in self.iis]) if n else np.zeros((len(frames), 0))
        self.occ_ref = np.median(self.occ, axis=0) if n else np.zeros(0)
        # fraction of bare-table pixels in each home core (true vacancy vs occlusion by arm/cube)
        self.tab = np.array([box_frac(_integral(table_mask(im)), self.Hc[:, 0], self.Hc[:, 1], 0.3 * self.Hw, 0.3 * self.Hh)
                             for im in frames]) if n else np.zeros((len(frames), 0))
        self.blobs = [self._target_blobs(m) for m in self.masks]

    def _target_blobs(self, m):
        bl = blobs(closing(m, 1), min_area=max(6, int(0.12 * self.ref_area)), max_area=int(4 * self.ref_area) + 50)
        mw = float(np.median(self.Hw)) if len(self.Hw) else 20.0
        out = []
        for b in bl:
            if b["w"] > 2.6 * mw or b["h"] > 2.6 * mw:
                continue
            out.append(b)
        return out

    def g(self, p):
        return ground(p)

    def off_home(self, t):
        res = []
        for b in self.blobs[t]:
            dx = np.abs(self.Hc[:, 0] - b["x"]) / (0.5 * self.Hw)
            dy = np.abs(self.Hc[:, 1] - b["y"]) / (0.5 * self.Hh)
            dd = np.maximum(dx, dy) if len(dx) else np.array([9.0])
            k = int(np.argmin(dd))
            if b["area"] > 1.45 * self.ref_area and dd[k] < 1.6 and self.occ[t, k] > 0.3 * self.occ_ref[k]:
                # a mover merged with the stationary target at home k: unmix the centroid
                ah = self.homes[k]["area"]
                am = max(b["area"] - ah, 0.35 * ah)
                vx = (b["x"] * b["area"] - self.Hc[k, 0] * ah) / am
                vy = (b["y"] * b["area"] - self.Hc[k, 1] * ah) / am
                v = dict(b); v.update(x=float(vx), y=float(vy), area=int(am), virtual=True)
                ddv = np.maximum(np.abs(self.Hc[:, 0] - vx) / (0.5 * self.Hw), np.abs(self.Hc[:, 1] - vy) / (0.5 * self.Hh))
                if ddv.min() >= 0.4:
                    res.append(v)
                continue
            if dd[k] < 0.55:
                continue
            res.append(b)
        return res

    # ------------------------------------------------------------------ swaps
    def pair_errors(self, vac_ratio=0.45):
        """For every pair and frame: best point-symmetric (about the pair midpoint, in ground
        coordinates) pair of off-home blobs, normalised error, and the blob indices."""
        n = len(self.homes); T = len(self.frames)
        vac = (self.occ < vac_ratio * np.maximum(self.occ_ref, 1e-3)[None, :]) & (self.tab > 0.45)
        self.vac = vac
        self.offs = [self.off_home(t) for t in range(T)]
        G = [self.g(np.array([[b["x"], b["y"]] for b in o]).reshape(-1, 2)) for o in self.offs]
        gH = self.g(self.Hc)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        E = np.full((len(pairs), T), np.inf)
        BI = np.full((len(pairs), T, 2), -1, int)
        for q, (i, j) in enumerate(pairs):
            gM = (gH[i] + gH[j]) / 2.0
            ra = np.linalg.norm(gH[i] - gM)
            tol = 1.5 + 0.08 * ra
            for t in range(T):
                if not (vac[t, i] and vac[t, j]) or len(G[t]) < 2:
                    continue
                P = G[t]
                rad = np.linalg.norm(P - gM, axis=1)
                okp = (rad > 0.35 * ra) & (rad < 4.0 * ra + 20)
                S = np.linalg.norm(P[:, None, :] + P[None, :, :] - 2 * gM, axis=2)
                np.fill_diagonal(S, np.inf)
                S[~okp, :] = np.inf; S[:, ~okp] = np.inf
                k = int(np.argmin(S))
                a_, b_ = divmod(k, S.shape[1])
                if np.isfinite(S[a_, b_]):
                    E[q, t] = S[a_, b_] / tol
                    BI[q, t] = (a_, b_)
                if not np.isfinite(E[q, t]) or E[q, t] > 1.0:
                    # partner hidden by a stationary target at another home (merged blob)
                    for a_ in np.nonzero(okp)[0]:
                        r = ground_inv(2 * gM - P[a_])
                        dx = np.abs(self.Hc[:, 0] - r[0]) / (0.5 * self.Hw)
                        dy = np.abs(self.Hc[:, 1] - r[1]) / (0.5 * self.Hh)
                        dd = np.maximum(dx, dy); dd[[i, j]] = 9
                        if dd.min() < 0.8:
                            if not np.isfinite(E[q, t]) or E[q, t] > 1.0:
                                E[q, t] = 1.0
                                BI[q, t] = (a_, a_)
        self.pairs, self.E, self.BI = pairs, E, BI
        return pairs, E, BI

    def detect_swaps(self, loose=2.5, min_active=2, max_span=1.9, gap_s=1.0):
        n = len(self.homes); T = len(self.frames)
        if n < 2:
            return []
        pairs, E, BI = self.pair_errors()
        times = self.times
        # candidate events: runs of loosely-active frames per pair
        cands = []
        for q, (i, j) in enumerate(pairs):
            idx = np.nonzero(E[q] < loose)[0]
            runs = []
            for t in idx:
                if runs and times[t] - times[runs[-1][-1]] <= gap_s and times[t] - times[runs[-1][0]] <= max_span:
                    runs[-1].append(t)
                else:
                    runs.append([t])
            for r in runs:
                if len(r) < min_active and not (len(r) == 1 and E[q, r[0]] < 0.35):
                    continue
                cands.append(dict(q=q, i=i, j=j, frames=r, t0=float(times[r[0]]), t1=float(times[r[-1]]),
                                  t=float(0.5 * (times[r[0]] + times[r[-1]])),
                                  n_active=len(r), err=float(np.mean(E[q, r]))))
        # components of temporally-overlapping candidates
        cands.sort(key=lambda c: c["t0"])
        comps = []
        for c in cands:
            if comps and c["t0"] <= max(x["t1"] for x in comps[-1]) + 0.3:
                comps[-1].append(c)
            else:
                comps.append([c])
        chosen = []
        for comp in comps:
            chosen.extend(self._best_subset(comp))
        chosen.sort(key=lambda e: e["t"])
        self.swap_candidates = cands
        return chosen

    def _subset_cost(self, sub, frames):
        cost = 0.0
        for t in frames:
            nb = len(self.offs[t])
            used = np.zeros(nb, int)
            for c in sub:
                if c["t0"] - 0.05 <= self.times[t] <= c["t1"] + 0.05:
                    e = self.E[c["q"], t]
                    if np.isfinite(e):
                        cost += min(e, 1.5)
                        a, b = self.BI[c["q"], t]
                        used[a] += 1
                        if b != a:
                            used[b] += 1
                    else:
                        cost += 0.6
            for k in range(nb):
                if used[k] == 0:
                    cost += 1.0
                elif used[k] > 1:
                    big = self.offs[t][k]["area"] > 1.4 * self.ref_area
                    cost += (0.2 if big else 1.0) * (used[k] - 1)
        return cost

    def _compatible(self, a, b):
        if not ({a["i"], a["j"]} & {b["i"], b["j"]}):
            return True
        return a["t1"] + 0.3 < b["t0"] or b["t1"] + 0.3 < a["t0"]

    def _best_subset(self, comp):
        frames = sorted(set(t for c in comp for t in range(
            int(np.searchsorted(self.times, c["t0"])), int(np.searchsorted(self.times, c["t1"], side="right")))))
        best = ([], self._subset_cost([], frames))
        if len(comp) <= 16:
            def rec(k, cur):
                nonlocal best
                if k == len(comp):
                    c = self._subset_cost(cur, frames)
                    if c < best[1] - 1e-9:
                        best = (list(cur), c)
                    return
                rec(k + 1, cur)
                if all(self._compatible(comp[k], x) for x in cur):
                    cur.append(comp[k]); rec(k + 1, cur); cur.pop()
            rec(0, [])
        else:
            cur = []
            curc = best[1]
            while True:
                bestgain, bestc = 0.0, None
                for c in comp:
                    if c in cur or not all(self._compatible(c, x) for x in cur):
                        continue
                    v = self._subset_cost(cur + [c], frames)
                    if curc - v > bestgain:
                        bestgain, bestc = curc - v, c
                if bestc is None:
                    break
                cur.append(bestc); curc -= bestgain
            best = (cur, curc)
        return best[0]

    # ------------------------------------------------------------- placements
    def cube_track(self, color):
        tr = []
        for im in self.frames:
            bl = blobs(cube_mask(im, color), min_area=6, max_area=4000)
            bl = [b for b in bl if b["y"] >= TABLE_TOP_V]
            if not bl:
                tr.append(None); continue
            b = max(bl, key=lambda b: b["area"])
            tr.append(b)
        return tr

    def detect_placements(self, color, min_dur=0.6, still_px=2.0):
        tr = self.cube_track(color)
        T = len(tr); n = len(self.homes)
        at = np.full(T, -1)
        for t in range(T):
            b = tr[t]
            if b is None or n == 0:
                continue
            # cube rests on a target: its centroid appears slightly above the target centre
            dx = (b["x"] - self.Hc[:, 0]) / (0.5 * self.Hw)
            dy = (b["y"] - (self.Hc[:, 1] - 0.2 * self.Hh)) / (0.5 * self.Hh)
            d = np.hypot(dx, dy)
            k = int(np.argmin(d))
            if d[k] < 0.75:
                # stationary w.r.t. neighbours
                still = True
                for u in (t - 1, t + 1):
                    if 0 <= u < T and tr[u] is not None and \
                            np.hypot(tr[u]["x"] - b["x"], tr[u]["y"] - b["y"]) > still_px * max(1.0, (self.times[u] - self.times[t]) / 0.05 * 0.0 + 1.0):
                        still = False
                if still:
                    at[t] = k
        # runs (missing detections inside a run are tolerated if the cube is not seen elsewhere)
        places = []
        cur = None
        for t in range(T):
            if at[t] >= 0:
                if cur and cur["h"] == at[t]:
                    cur["t1"] = self.times[t]; cur["n"] += 1
                else:
                    cur = dict(h=int(at[t]), t0=self.times[t], t1=self.times[t], n=1)
                    places.append(cur)
            elif tr[t] is not None:
                # cube visible but not resting on this home: does it break the run?
                if cur is not None:
                    b = tr[t]
                    hx, hy = self.Hc[cur["h"]]
                    if np.hypot((b["x"] - hx) / (0.5 * self.Hw[cur["h"]]), (b["y"] - hy + 0.2 * self.Hh[cur["h"]]) / (0.5 * self.Hh[cur["h"]])) > 1.2:
                        cur = None
        out = [p for p in places if p["t1"] - p["t0"] >= min_dur]
        # merge consecutive placements on the same home separated only by occlusion
        merged = []
        for p in out:
            if merged and merged[-1]["h"] == p["h"] and p.get("_same_run"):
                merged[-1]["t1"] = p["t1"]
            else:
                merged.append(dict(p))
        for p in merged:
            p["t0"] = float(p["t0"]); p["t1"] = float(p["t1"])
        self.cube = tr
        return merged


def solve(frames, times, color, ordinal, log=print):
    """Returns dict with the answer home (pixel in front image) and diagnostics."""
    A = Analysis(frames, times, log=log)
    swaps = A.detect_swaps()
    places = A.detect_placements(color)
    n = len(A.homes)
    ident = list(range(n))
    seq = []
    evs = [("swap", e["t"], e) for e in swaps] + [("place", p["t0"], p) for p in places]
    evs.sort(key=lambda x: x[1])
    for kind, t, e in evs:
        if kind == "swap":
            ident[e["i"]], ident[e["j"]] = ident[e["j"]], ident[e["i"]]
        else:
            seq.append(dict(identity=ident[e["h"]], home=e["h"], t=t))
    res = dict(homes=A.homes, swaps=swaps, placements=places, sequence=seq, final_ident=ident, analysis=A)
    if not seq:
        res["answer_home"] = None
        return res
    if ordinal == "last" or ordinal is None:
        k = len(seq) - 1
    else:
        k = min(max(ordinal - 1, 0), len(seq) - 1)
    target_id = seq[k]["identity"]
    res["answer_home"] = ident.index(target_id)
    res["k"] = k
    return res
