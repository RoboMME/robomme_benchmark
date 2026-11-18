"""Tracker v2: grid detections of container caps + exclusive Hungarian assignment."""
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

CELL = 0.005
X0, Y0, NX, NY = -0.50, -0.50, 200, 200

def detections(pts, a1=None, prior=None):
    """pts: Nx2 cap points (fused). Returns list of (xy, area_cells)."""
    if len(pts) == 0:
        return []
    ix = ((pts[:, 0] - X0) / CELL).astype(int); iy = ((pts[:, 1] - Y0) / CELL).astype(int)
    k = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
    g = np.zeros((NX, NY), bool); g[ix[k], iy[k]] = True
    g = ndimage.binary_closing(g, iterations=1)
    lab, n = ndimage.label(g)
    out = []
    a1 = a1 or 110.0
    for i in range(1, n + 1):
        cells = np.argwhere(lab == i)
        area = len(cells)
        if area < 12:
            continue
        xy = np.stack([X0 + (cells[:, 0] + 0.5) * CELL, Y0 + (cells[:, 1] + 0.5) * CELL], 1)
        kk = int(max(1, round(area / a1)))
        ext = xy.max(0) - xy.min(0)
        if kk == 1 and max(ext) > 0.085:
            kk = 2
        if kk == 1:
            out.append((xy.mean(0), area))
            continue
        # k-means split, init from priors inside the component when available
        init = None
        if prior is not None and len(prior):
            d = np.linalg.norm(prior[:, None, :] - xy[None, :, :], axis=-1).min(1)
            inside = prior[d < 0.03]
            if len(inside) >= kk:
                init = inside[np.argsort(np.linalg.norm(inside - xy.mean(0), axis=1))][:kk]
        if init is None:
            # farthest-point init
            init = [xy[np.argmax(np.linalg.norm(xy - xy.mean(0), axis=1))]]
            while len(init) < kk:
                dd = np.min(np.linalg.norm(xy[:, None, :] - np.array(init)[None], axis=-1), 1)
                init.append(xy[np.argmax(dd)])
            init = np.array(init)
        c = np.array(init, float)
        for _ in range(8):
            a = np.argmin(np.linalg.norm(xy[:, None, :] - c[None], axis=-1), 1)
            for j in range(kk):
                if (a == j).any():
                    c[j] = xy[a == j].mean(0)
        for j in range(kk):
            out.append((c[j], int((a == j).sum())))
    return out

class Tracker:
    def __init__(self, centers, names):
        self.c = np.array(centers, float)[:, :2]
        self.v = np.zeros_like(self.c)
        self.names = list(names)
        self.last_step = None
        self.lost = np.zeros(len(self.c), int)
        self.hist = []
        self.events = []
        self.a1 = None

    def update(self, pts, step_id, gate=0.045):
        dt = 1 if self.last_step is None else max(1, step_id - self.last_step)
        pred = self.c + self.v * dt
        dets = detections(pts, self.a1, prior=pred)
        if self.a1 is None and len(dets) >= len(self.c) - 1:
            areas = sorted(a for _, a in dets)
            self.a1 = float(np.median(areas))
            dets = detections(pts, self.a1, prior=pred)
        cnt = np.zeros(len(self.c))
        newc = pred.copy()
        matched = np.zeros(len(self.c), bool)
        if dets:
            D = np.array([d[0] for d in dets])
            cost = np.linalg.norm(pred[:, None, :] - D[None], axis=-1)
            # gate grows with time lost
            g = gate + 0.004 * np.minimum(self.lost, 15)[:, None] * dt
            big = 10.0
            C = np.where(cost < g, cost, big)
            ri, ci = linear_sum_assignment(C)
            used = set()
            for r, cidx in zip(ri, ci):
                if C[r, cidx] < big:
                    newc[r] = D[cidx]; matched[r] = True; cnt[r] = dets[cidx][1]; used.add(cidx)
            # unmatched detections -> lost tracks (nearest, wider gate)
            free = [j for j in range(len(D)) if j not in used]
            lostidx = [i for i in range(len(self.c)) if not matched[i] and self.lost[i] >= 2]
            maxd = 0.03 + 0.008 * self.lost * dt
            if free and lostidx:
                C2 = np.linalg.norm(pred[lostidx][:, None, :] - D[free][None], axis=-1)
                r2, c2 = linear_sum_assignment(C2)
                for a, b in zip(r2, c2):
                    if C2[a, b] < min(0.2, maxd[lostidx[a]]):
                        i = lostidx[a]; newc[i] = D[free[b]]; matched[i] = True; cnt[i] = -1
                        self.events.append((step_id, "reacq", self.names[i], float(C2[a, b])))
        vel = (newc - self.c) / dt
        self.v = np.where(matched[:, None] & (cnt[:, None] > 0), 0.6 * vel + 0.4 * self.v, self.v * 0.7)
        sp = np.linalg.norm(self.v, axis=1, keepdims=True)
        self.v = np.where(sp > 0.02, self.v / np.maximum(sp, 1e-9) * 0.02, self.v)
        self.c = newc
        self.lost = np.where(matched, 0, self.lost + 1)
        self.last_step = step_id
        self.hist.append((step_id, self.c.copy(), cnt.copy()))

    def pos(self, name):
        return self.c[self.names.index(name)]
