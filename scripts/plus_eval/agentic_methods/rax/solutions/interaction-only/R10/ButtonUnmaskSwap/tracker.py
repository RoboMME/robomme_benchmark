import numpy as np
from scipy.optimize import linear_sum_assignment

class Tracker:
    """Track N identical containers in the world x-y plane by gated assignment."""
    def __init__(self, init_xy, gate=0.045):
        self.p = np.array(init_xy, float)          # N x 2
        self.v = np.zeros_like(self.p)             # per-step velocity
        self.t = None
        self.gate = gate
        self.missing = np.zeros(len(self.p), int)
        self.history = []
        self.reacq = []

    def hist_at(self, step):
        best = None
        for s, p, _ in self.history:
            if s <= step: best = p
        return best

    def update(self, step, blobs):
        if self.t is None:
            self.t = step
        dt = max(1, step - self.t)
        pred = self.p + self.v * dt
        B = np.array([b["xy"] for b in blobs], float).reshape(-1, 2)
        N = len(self.p)
        if len(B):
            C = np.linalg.norm(pred[:, None, :] - B[None, :, :], axis=2)
            big = 1e3
            Cg = np.where(C < self.gate + 0.004 * dt, C, big)
            ri, ci = linear_sum_assignment(Cg)
        else:
            ri, ci = [], []
        matched = set()
        newp = pred.copy()
        for i, j in zip(ri, ci):
            if Cg[i, j] >= 1e3:
                continue
            newp[i] = B[j]
            matched.add(i)
        # re-acquire: unmatched tracks vs unmatched full-size blobs
        um_t = [i for i in range(N) if i not in matched]
        used_b = set(int(j) for i, j in zip(ri, ci) if Cg[i, j] < 1e3)
        um_b = [j for j in range(len(B)) if j not in used_b and blobs[j].get("n", 999) >= 80]
        if um_t and um_b:
            C2 = np.linalg.norm(pred[um_t][:, None, :] - B[um_b][None, :, :], axis=2)
            r2, c2 = linear_sum_assignment(C2)
            for a, b in zip(r2, c2):
                if C2[a, b] < 0.20 or (len(um_t) == 1 and len(um_b) == 1):
                    i = um_t[a]; newp[i] = B[um_b[b]]; matched.add(i)
                    self.v[i] = 0
                    self.reacq.append((step, i, float(C2[a, b])))
        for i in range(N):
            if i in matched:
                nv = (newp[i] - self.p[i]) / dt
                self.v[i] = 0.5 * self.v[i] + 0.5 * nv
                self.missing[i] = 0
            else:
                self.missing[i] += 1
                self.v[i] *= 0.5
        self.p = newp
        self.t = step
        self.history.append((step, self.p.copy(), sorted(matched)))
        return len(matched)
