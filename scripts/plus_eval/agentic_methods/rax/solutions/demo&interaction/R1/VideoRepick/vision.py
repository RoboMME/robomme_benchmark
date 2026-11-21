"""Color-based object detection and multi-object tracking (numpy/cv2)."""
import numpy as np
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
from scipy import ndimage


def rgb_to_hsv(rgb):
    """Return hue in degrees [0,360), sat [0,255], val [0,255] as int arrays."""
    if cv2 is not None:
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb.astype(np.uint8)), cv2.COLOR_RGB2HSV_FULL).astype(np.int32)
        return hsv[..., 0] * 360 // 256, hsv[..., 1], hsv[..., 2]
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = x.max(-1); mn = x.min(-1); d = mx - mn + 1e-9
    h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    s = np.where(mx > 0, d / (mx + 1e-9), 0)
    return h.astype(np.int32) % 360, (s * 255).astype(np.int32), (mx * 255).astype(np.int32)


def hue_dist(h, c):
    d = np.abs(h - c) % 360
    return np.minimum(d, 360 - d)


def table_hue(rgb):
    h, s, v = rgb_to_hsv(rgb)
    sel = s > 60
    if sel.sum() < 100:
        return 20
    hist = np.bincount(h[sel] // 5, minlength=72)
    return int(np.argmax(hist) * 5 + 2)


def candidate_hue_bands(rgb, thue, min_dist=40):
    """Histogram of saturated pixels far from table hue -> list of (center, count)."""
    h, s, v = rgb_to_hsv(rgb)
    sel = (s > 50) & (v > 40) & (hue_dist(h, thue) >= min_dist)
    hist = np.bincount(h[sel] // 10, minlength=36).astype(float)
    return hist


def mask_for_band(rgb, center, width=30, smin=50, vmin=35):
    h, s, v = rgb_to_hsv(rgb)
    return (hue_dist(h, center) <= width) & (s > smin) & (v > vmin)


def components(mask, minarea=3):
    lab, n = ndimage.label(mask)
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    areas = ndimage.sum(np.ones_like(mask, dtype=float), lab, idx)
    com = ndimage.center_of_mass(np.ones_like(mask, dtype=float), lab, idx)
    sl = ndimage.find_objects(lab)
    out = []
    for i in range(n):
        if areas[i] < minarea:
            continue
        y, x = com[i]
        ys, xs = sl[i]
        out.append({'x': float(x), 'y': float(y), 'area': float(areas[i]),
                    'bbox': (xs.start, ys.start, xs.stop, ys.stop), 'label': i + 1})
    return out


def choose_object_hue(images, thue=None):
    """Pick the hue band whose blobs look like several similar compact objects.

    images: list of RGB arrays (same camera). Returns (center_hue, scores)."""
    if thue is None:
        thue = table_hue(images[0])
    hist = sum(candidate_hue_bands(im, thue) for im in images)
    sm = hist + 0.5 * (np.roll(hist, 1) + np.roll(hist, -1))
    peaks = [b for b in range(36) if sm[b] >= sm[(b - 1) % 36] and sm[b] >= sm[(b + 1) % 36] and hist[b] >= 15 * len(images)]
    scores = []
    for b in peaks:
        # weighted centre over neighbouring bins
        bins = [(b - 1) % 36, b, (b + 1) % 36]
        w = np.array([hist[q] for q in bins]) + 1e-9
        ang = np.array([q * 10 + 5 for q in bins], float)
        ang = np.where(ang - ang[1] > 180, ang - 360, np.where(ang - ang[1] < -180, ang + 360, ang))
        c = int(round(float((w * ang).sum() / w.sum()))) % 360
        tot = 0.0
        for im in images:
            sm_ = adaptive_smin([im], c)
            m = mask_for_band(im, c, width=25, smin=sm_, vmin=40)
            comps = [q for q in components(m, minarea=6)]
            if not comps:
                continue
            areas = np.array([q['area'] for q in comps])
            med = float(np.median(areas))
            good = 0
            good_area = 0.0
            for q in comps:
                x0, y0, x1, y1 = q['bbox']
                fill = q['area'] / max(1, (x1 - x0) * (y1 - y0))
                if 0.4 * med <= q['area'] <= 2.5 * med and fill > 0.3 and q['area'] >= 10:
                    good += 1
                    good_area += q['area']
            ratio = good_area / max(1.0, float(areas.sum()))
            tot += min(good, 16) * ratio
        scores.append((tot / len(images), c))
    scores.sort(reverse=True)
    if not scores:
        return None, scores
    return scores[0][1], scores


# ---------------------------------------------------------------- tracking
from scipy.optimize import linear_sum_assignment


def detect(rgb, hue, width=25, smin=70, vmin=40, minarea=3, maxarea=None, exclude=None):
    m = mask_for_band(rgb, hue, width=width, smin=smin, vmin=vmin)
    if exclude is not None:
        m &= ~exclude
    comps = components(m, minarea=minarea)
    if maxarea is not None:
        comps = [c for c in comps if c['area'] <= maxarea]
    return comps


class Tracker:
    """Multi-object tracker for N identical blobs from a static camera.

    Handles merges (one blob covering several objects) and occlusion (coasting)."""

    def __init__(self, init_pts, single_area, gate=10.0, grow=30.0, max_missing=2.0):
        self.grow = grow
        self.max_missing = max_missing
        self.n = len(init_pts)
        self.pos = np.array(init_pts, float).reshape(-1, 2)
        self.vel = np.zeros_like(self.pos)
        self.a1 = max(single_area, 4.0)
        self.gate = gate
        self.missing = np.zeros(self.n)        # seconds since last single detection
        self.merged = np.zeros(self.n, bool)
        self.amb = np.zeros(self.n)           # ambiguity score accumulated
        self.history = [self.pos.copy()]
        self.t = None

    def step(self, dets, t):
        dt = 0.05 if self.t is None else max(1e-3, t - self.t)
        self.t = t
        pred = self.pos + self.vel * dt
        slots, owner = [], []
        for j, d in enumerate(dets):
            k = int(np.clip(round(d['area'] / self.a1), 1, 3))
            for _ in range(k):
                slots.append((d['x'], d['y'])); owner.append(j)
        n, m = self.n, len(slots)
        BIG = 1e6
        g = self.gate + self.grow * np.minimum(self.missing, self.max_missing)
        C = np.full((n, m + n), BIG)
        if m:
            S = np.array(slots)
            D = np.linalg.norm(pred[:, None, :] - S[None, :, :], axis=2)
            C[:, :m] = np.where(D <= g[:, None], D, BIG)
            # slight penalty for using the 2nd/3rd slot of a blob (prefer one-to-one)
            seen = {}
            for s, j in enumerate(owner):
                r = seen.get(j, 0); seen[j] = r + 1
                C[:, s] = np.where(C[:, s] < BIG, C[:, s] + 2.0 * r, BIG)
        for i in range(n):
            C[i, m + i] = g[i]
        rows, cols = linear_sum_assignment(C)
        assign = {}
        for r, c in zip(rows, cols):
            if c < m and C[r, c] < BIG:
                assign[r] = owner[c]
        # ambiguity: cost of the best alternative assignment for each track
        users = {}
        for r, j in assign.items():
            users.setdefault(j, []).append(r)
        newpos = pred.copy()
        for i in range(n):
            if i in assign:
                j = assign[i]
                d = dets[j]
                p = np.array([d['x'], d['y']])
                shared = len(users[j]) > 1
                if not shared:
                    if self.missing[i] > 0 or self.merged[i]:
                        # re-acquired after merge/occlusion: keep velocity estimate fresh
                        self.vel[i] = 0.5 * self.vel[i]
                    else:
                        self.vel[i] = 0.6 * self.vel[i] + 0.4 * (p - self.pos[i]) / dt
                    newpos[i] = p
                    self.missing[i] = 0
                    self.merged[i] = False
                else:
                    # merged blob: move towards blob centroid but keep own prediction offset
                    newpos[i] = 0.5 * pred[i] + 0.5 * p
                    self.merged[i] = True
                    self.missing[i] += dt * 0.5
            else:
                self.missing[i] += dt
                self.vel[i] *= 0.9
        vmax = 400.0
        sp = np.linalg.norm(self.vel, axis=1, keepdims=True)
        self.vel = np.where(sp > vmax, self.vel * vmax / np.maximum(sp, 1e-9), self.vel)
        self.pos = newpos
        self.history.append(self.pos.copy())
        return assign


class PixelTracker:
    """k-means style tracker: object-mask pixels are split between track predictions.

    Robust to touching objects; identity through full overlaps relies on velocity."""

    def __init__(self, init_pts, single_area, rad_mult=1.6, vel_alpha=0.5):
        self.n = len(init_pts)
        self.pos = np.array(init_pts, float).reshape(-1, 2)
        self.vel = np.zeros_like(self.pos)
        self.a1 = max(float(single_area), 4.0)
        self.r = np.sqrt(self.a1 / np.pi)
        self.R = rad_mult * self.r + 2.0
        self.alpha = vel_alpha
        self.support = np.ones(self.n)
        self.lost = np.zeros(self.n)      # seconds with poor support
        self.contested = np.zeros(self.n)  # seconds sharing a blob with another track
        self.history = [self.pos.copy()]
        self.t = None
        self.events = []

    def step(self, mask, t, comps=None):
        dt = 0.05 if self.t is None else max(1e-3, t - self.t)
        self.t = t
        pred = self.pos + self.vel * dt
        ys, xs = np.nonzero(mask)
        P = np.stack([xs, ys], 1).astype(float) if len(xs) else np.zeros((0, 2))
        newpos = pred.copy()
        supp = np.zeros(self.n)
        if len(P):
            c = pred.copy()
            Rr = self.R + 25.0 * np.minimum(self.lost, 1.0)
            for it in range(3):
                D = np.linalg.norm(P[:, None, :] - c[None, :, :], axis=2)
                # normalise by per-track radius so lost tracks reach further
                Dn = D / Rr[None, :]
                a = np.argmin(Dn, 1)
                ok = Dn[np.arange(len(P)), a] <= 1.0
                for i in range(self.n):
                    sel = ok & (a == i)
                    supp[i] = sel.sum()
                    if supp[i] >= 0.25 * self.a1:
                        c[i] = P[sel].mean(0)
                Rr = np.minimum(Rr, self.R + 6.0)  # tighten after first pass
            newpos = np.where((supp >= 0.25 * self.a1)[:, None], c, pred)
        good = supp >= 0.35 * self.a1
        for i in range(self.n):
            if good[i]:
                v = (newpos[i] - self.pos[i]) / dt
                if self.lost[i] > 0:
                    self.vel[i] = 0.3 * v
                else:
                    self.vel[i] = (1 - self.alpha) * self.vel[i] + self.alpha * v
                self.lost[i] = 0
            else:
                self.lost[i] += dt
                self.vel[i] *= 0.85
        # contested: two tracks closer than one object radius
        D = np.linalg.norm(newpos[:, None] - newpos[None], axis=2) + np.eye(self.n) * 1e9
        close = D.min(1) < 1.2 * self.r
        self.contested += np.where(close, dt, 0)
        if close.any():
            self.events.append((t, np.nonzero(close)[0].tolist()))
        self.support = supp
        sp = np.linalg.norm(self.vel, axis=1, keepdims=True)
        vmax = 300.0
        self.vel = np.where(sp > vmax, self.vel * vmax / np.maximum(sp, 1e-9), self.vel)
        self.pos = newpos
        self.history.append(self.pos.copy())


def adaptive_smin(images, hue, width=25, base=60):
    """Saturation threshold that keeps the (most saturated) objects and drops tinted scenery."""
    meds = []
    for im in images:
        h, s, v = rgb_to_hsv(im)
        mk = (hue_dist(h, hue) <= width) & (s > base) & (v > 40)
        lab, n = ndimage.label(mk)
        if n == 0:
            continue
        idx = np.arange(1, n + 1)
        area = ndimage.sum(np.ones_like(s, float), lab, idx)
        med = ndimage.median(s.astype(float), lab, idx)
        for a, md in zip(area, med):
            if a >= 8:
                meds.append(md)
    if not meds:
        return base
    ref = float(np.percentile(meds, 75))
    return int(max(base, min(0.75 * ref, 150)))


class BlobTracker:
    """Component-level one-to-one assignment (Hungarian) with pixel k-means splitting of
    shared blobs, own-area memory, and orphan-blob reacquisition."""

    def __init__(self, init_pts, single_area, init_areas=None, gate=8.0):
        self.n = len(init_pts)
        self.pos = np.array(init_pts, float).reshape(-1, 2)
        self.vel = np.zeros_like(self.pos)
        self.a1 = max(float(single_area), 4.0)
        self.A = np.array(init_areas if init_areas is not None else [self.a1] * self.n, float)
        self.A = np.clip(self.A, 0.3 * self.a1, 3 * self.a1)
        self.r = np.sqrt(self.a1 / np.pi)
        self.gate = gate
        self.lost = np.zeros(self.n)
        self.contested = np.zeros(self.n)
        self.support = np.ones(self.n)
        self.history = [self.pos.copy()]
        self.t = None
        self.events = []

    def step(self, mask, t):
        dt = 0.05 if self.t is None else max(1e-3, t - self.t)
        self.t = t
        pred = self.pos + self.vel * dt
        lab, nc = ndimage.label(mask)
        newpos = pred.copy()
        supp = np.zeros(self.n)
        shared = np.zeros(self.n, bool)
        if nc:
            idx = np.arange(1, nc + 1)
            areas = ndimage.sum(np.ones(mask.shape), lab, idx)
            keep = [i for i in range(nc) if areas[i] >= max(6, 0.15 * self.a1)]
            pix = {}
            ys, xs = np.nonzero(lab)
            L = lab[ys, xs]
            order = np.argsort(L, kind='stable')
            ys, xs, L = ys[order], xs[order], L[order]
            bounds = np.searchsorted(L, idx)
            bounds = list(bounds) + [len(L)]
            for i in keep:
                s0, s1 = bounds[i], bounds[i + 1]
                pix[i] = np.stack([xs[s0:s1], ys[s0:s1]], 1).astype(float)
            comp = keep
            # distance track -> nearest pixel of comp
            D = np.full((self.n, len(comp)), 1e9)
            Cc = np.zeros((len(comp), 2))
            for jj, j in enumerate(comp):
                P = pix[j]
                Cc[jj] = P.mean(0)
                if len(P) > 400:
                    P = P[::max(1, len(P) // 400)]
                dd = np.linalg.norm(P[None] - pred[:, None], axis=2).min(1)
                D[:, jj] = dd
            g = self.gate + 30.0 * np.minimum(self.lost, 1.5)
            assign = -np.ones(self.n, int)
            if len(comp):
                cost = np.where(D <= g[:, None], D + 0.2 * np.linalg.norm(pred[:, None] - Cc[None], axis=2), 1e6)
                ri, ci = linear_sum_assignment(cost)
                for a, b in zip(ri, ci):
                    if cost[a, b] < 1e6:
                        assign[a] = b
                # leftover tracks share their nearest gated blob
                for i in range(self.n):
                    if assign[i] < 0:
                        jj = int(np.argmin(D[i]))
                        if D[i, jj] <= g[i]:
                            assign[i] = jj
                # orphan blobs -> nearest lost track, or a duplicate on an undersized shared blob
                used = set(assign[assign >= 0].tolist())
                for jj in range(len(comp)):
                    if jj in used:
                        continue
                    a_j = len(pix[comp[jj]])
                    if a_j < 0.4 * min(self.A.min(), self.a1):
                        continue
                    cand = []
                    for i in range(self.n):
                        if assign[i] < 0:
                            cand.append((np.linalg.norm(pred[i] - Cc[jj]), i))
                        else:
                            k = assign[i]
                            sharers = np.nonzero(assign == k)[0]
                            if len(sharers) > 1 and len(pix[comp[k]]) < 0.8 * self.A[sharers].sum():
                                cand.append((np.linalg.norm(pred[i] - Cc[jj]) + 5.0, i))
                    cand = [c for c in cand if c[0] < 70]
                    if cand:
                        cand.sort()
                        i = cand[0][1]
                        assign[i] = jj
                        self.events.append((t, 'reacq', i))
                        self.lost[i] = max(self.lost[i], 0.5)
            # positions
            for jj in set(assign[assign >= 0].tolist()):
                tr = np.nonzero(assign == jj)[0]
                P = pix[comp[jj]]
                if len(tr) == 1:
                    newpos[tr[0]] = P.mean(0)
                    supp[tr[0]] = len(P)
                    continue
                shared[tr] = True
                c = pred[tr].copy()
                for _ in range(5):
                    a = np.argmin(np.linalg.norm(P[:, None] - c[None], axis=2), 1)
                    for q in range(len(tr)):
                        if (a == q).sum() > 0:
                            c[q] = P[a == q].mean(0)
                for q, i in enumerate(tr):
                    supp[i] = (a == q).sum()
                if len(P) < 0.8 * self.A[tr].sum():
                    # strong overlap: blend predictions with split centroids to keep identities
                    c = 0.5 * c + 0.5 * (pred[tr] + (P.mean(0) - pred[tr].mean(0)))
                newpos[tr] = c
        for i in range(self.n):
            if supp[i] > 0:
                v = (newpos[i] - self.pos[i]) / dt
                if self.lost[i] > 0:
                    self.vel[i] = 0.3 * v
                else:
                    self.vel[i] = 0.5 * self.vel[i] + 0.5 * v
                self.lost[i] = 0
                if not shared[i]:
                    self.A[i] = 0.8 * self.A[i] + 0.2 * supp[i]
            else:
                self.lost[i] += dt
                self.vel[i] *= 0.8
        self.contested += np.where(shared, dt, 0)
        sp = np.linalg.norm(self.vel, axis=1, keepdims=True)
        self.vel = np.where(sp > 300, self.vel * 300 / np.maximum(sp, 1e-9), self.vel)
        self.support = supp
        self.pos = newpos
        self.history.append(self.pos.copy())
