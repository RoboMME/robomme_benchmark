import numpy as np
from scipy.optimize import linear_sum_assignment

class Tracker:
    def __init__(self, xys):
        self.pos = np.array(xys, float)
        self.vel = np.zeros_like(self.pos)
        self.miss = np.zeros(len(self.pos), int)

    def update(self, dets, gate=0.035, reacq=0.07):
        """dets: (M,2). Returns per-track displacement this frame."""
        pred = self.pos + self.vel
        old = self.pos.copy()
        if len(dets) == 0:
            self.pos = pred; self.miss += 1
            return np.linalg.norm(self.pos - old, axis=1)
        dets = np.asarray(dets, float)
        C = np.linalg.norm(pred[:, None, :] - dets[None, :, :], axis=-1)
        Cg = np.where(C < gate, C, 1e3)
        r, c = linear_sum_assignment(Cg)
        matched = np.zeros(len(self.pos), bool)
        for i, j in zip(r, c):
            if Cg[i, j] < 1e3:
                newp = dets[j]
                v = newp - self.pos[i]
                self.vel[i] = 0.5 * self.vel[i] + 0.5 * v
                self.pos[i] = newp
                self.miss[i] = 0
                matched[i] = True
        # second pass: re-acquire lost tracks with unclaimed detections (wider gate)
        used = set(j for i, j in zip(r, c) if Cg[i, j] < 1e3)
        free = [j for j in range(len(dets)) if j not in used]
        lost = [i for i in np.nonzero(~matched)[0] if self.miss[i] > 0]
        if free and lost:
            C2 = np.linalg.norm(self.pos[lost][:, None, :] - dets[free][None, :, :], axis=-1)
            C2g = np.where(C2 < reacq, C2, 1e3)
            r2, c2 = linear_sum_assignment(C2g)
            for a, b in zip(r2, c2):
                if C2g[a, b] < 1e3:
                    i, j = lost[a], free[b]
                    self.pos[i] = dets[j]; self.vel[i] = 0; self.miss[i] = 0; matched[i] = True
        for i in np.nonzero(~matched)[0]:
            self.pos[i] = pred[i]
            self.vel[i] *= 0.7
            self.miss[i] += 1
        return np.linalg.norm(self.pos - old, axis=1)
