"""Multi-object tracking of identical objects in a 2-D plane (world XY live; image
pixels in the offline proxy test).  Units are normalised by the object diameter D.

Frames are a list of point sets (Px2 arrays).  Tracks are initialised from the first
frame by clustering.  Each subsequent frame: constant-velocity prediction, then a few
EM iterations assigning points to the nearest predicted centre within a gate.
Tracks without support coast.  Identity of the target is the main output.
"""
import numpy as np

def grid_clusters(pts, cell, min_pts):
    """Connected components of an occupancy grid; returns list of index arrays."""
    if len(pts) == 0:
        return []
    from scipy import ndimage
    mn = pts.min(0) - cell
    ij = np.floor((pts - mn) / cell).astype(int)
    shape = ij.max(0) + 2
    occ = np.zeros(shape, bool); occ[ij[:, 0], ij[:, 1]] = True
    lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
    pl = lab[ij[:, 0], ij[:, 1]]
    out = []
    for k in range(1, n + 1):
        idx = np.nonzero(pl == k)[0]
        if len(idx) >= min_pts:
            out.append(idx)
    return out

def kmeans(pts, k, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    # farthest-point init
    c = [pts[rng.integers(len(pts))]]
    for _ in range(1, k):
        d = np.min([((pts - ci) ** 2).sum(1) for ci in c], 0)
        c.append(pts[int(np.argmax(d))])
    c = np.array(c, float)
    for _ in range(iters):
        lab = np.argmin(((pts[:, None] - c[None]) ** 2).sum(-1), 1)
        for j in range(k):
            if (lab == j).any():
                c[j] = pts[lab == j].mean(0)
    return c, lab

def detect_objects(pts, D, min_pts=5, cell=None):
    """Cluster a point set into object centres. D: object diameter estimate (or None)."""
    if len(pts) < min_pts:
        return np.zeros((0, 2)), D
    if cell is None:
        cell = D / 4.0
    cl = grid_clusters(pts, cell, min_pts)
    if not cl:
        return np.zeros((0, 2)), D
    ext = np.array([np.ptp(pts[i], 0).max() for i in cl])
    if D is None:
        D = float(np.median(ext))
    cents = []
    for i, e in zip(cl, ext):
        k = max(1, int(round(e / D))) if e > 1.5 * D else 1
        if k == 1:
            cents.append(pts[i].mean(0))
        else:
            c, _ = kmeans(pts[i], min(k, 4))
            cents.extend(list(c))
    return np.array(cents), D

class Tracker:
    def __init__(self, centres, D, gate=1.2, min_support=4, log=None):
        self.D = D
        self.pos = np.array(centres, float)
        self.vel = np.zeros_like(self.pos)
        self.n = len(self.pos)
        self.missing = np.zeros(self.n, int)
        self.support = np.zeros(self.n, int)
        self.gate = gate * D
        self.min_support = min_support
        self.hist = [self.pos.copy()]
        self.miss_hist = [self.missing.copy()]
        self.log = log

    def step(self, pts, dt=1.0, frozen=None):
        """pts: Px2.  frozen: optional bool array; frozen tracks do not move (e.g. lifted target)."""
        pred = self.pos + self.vel * dt
        for i in range(self.n):
            if self.missing[i] > 0:
                pred[i] = self.pos[i] + self.vel[i] * dt * 0.5
        c = pred.copy()
        assigned = np.full(len(pts), -1)
        fz = np.zeros(self.n, bool) if frozen is None else np.asarray(frozen, bool)
        if len(pts):
            for it in range(4):
                d = np.sqrt(((pts[:, None] - c[None]) ** 2).sum(-1))  # P x N
                d[:, fz] = np.inf   # frozen (lifted) tracks do not claim points
                lab = np.argmin(d, 1)
                ok = d[np.arange(len(pts)), lab] <= self.gate
                assigned = np.where(ok, lab, -1)
                newc = c.copy()
                for j in range(self.n):
                    sel = assigned == j
                    if sel.sum() >= self.min_support:
                        newc[j] = pts[sel].mean(0)
                    else:
                        newc[j] = pred[j]
                c = newc
        cnt = np.array([(assigned == j).sum() for j in range(self.n)])
        newpos = self.pos.copy()
        for j in range(self.n):
            if frozen is not None and frozen[j]:
                self.missing[j] += 0 if cnt[j] >= self.min_support else 1
                continue
            jump_ok = True
            if cnt[j] >= self.min_support and self.missing[j] == 0:
                lim = 0.45 * self.D + 1.5 * float(np.linalg.norm(self.vel[j])) * dt
                if np.linalg.norm(c[j] - pred[j]) > lim:
                    jump_ok = False     # sudden jump: points of a neighbour, not motion
                    self.jumps = getattr(self, 'jumps', 0) + 1
            if cnt[j] >= self.min_support and jump_ok:
                v = (c[j] - self.pos[j]) / max(dt, 1e-6)
                if self.missing[j] > 0:
                    self.vel[j] = 0.3 * v
                else:
                    self.vel[j] = 0.6 * self.vel[j] + 0.4 * v
                newpos[j] = c[j]
                self.missing[j] = 0
            else:
                newpos[j] = pred[j]
                self.vel[j] *= 0.7
                self.missing[j] += 1
        # re-acquire: unassigned point clusters near a missing track
        if len(pts):
            free = pts[assigned < 0]
            miss = [j for j in range(self.n) if cnt[j] < self.min_support and not (frozen is not None and frozen[j])]
            if len(free) >= self.min_support and miss:
                cents, _ = detect_objects(free, self.D, self.min_support)
                sup = [j for j in range(self.n) if cnt[j] >= self.min_support]
                for cc in cents:
                    dd = [np.linalg.norm(cc - newpos[j]) for j in miss]
                    if not dd:
                        break
                    # never re-acquire onto points that belong next to a supported track
                    if sup and min(np.linalg.norm(cc - newpos[j]) for j in sup) < 1.2 * self.D:
                        continue
                    k = int(np.argmin(dd))
                    gate_re = min(2.5 * self.D, 1.0 * self.D + 0.25 * self.D * self.missing[miss[k]])
                    if dd[k] <= gate_re:
                        j = miss.pop(k)
                        newpos[j] = cc; self.vel[j] = 0; self.missing[j] = 0
        self.pos = newpos
        self.support = cnt
        self.hist.append(self.pos.copy())
        self.miss_hist.append(self.missing.copy())
        return cnt

    def reacquire_near(self, j, pts, start, radius, max_shift=None):
        """Place track j on the mean-shift centre of points near `start` (after a put-back).
        Rejects a result that drifted more than max_shift (likely a neighbour)."""
        c = np.asarray(start, float).copy()
        for _ in range(6):
            sel = np.linalg.norm(pts - c, axis=1) < radius
            if sel.sum() < self.min_support:
                self.pos[j] = np.asarray(start, float); self.vel[j] = 0
                return False
            c = pts[sel].mean(0)
        if max_shift is not None and np.linalg.norm(c - start) > max_shift:
            self.pos[j] = np.asarray(start, float); self.vel[j] = 0
            return False
        self.pos[j] = c; self.vel[j] = 0; self.missing[j] = 0
        return True
