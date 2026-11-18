"""Multi-target tracker for identical containers in world XY."""
import numpy as np
from scipy.optimize import linear_sum_assignment

GATE = 0.06        # m: max association distance (prediction to detection)
VEL_DECAY = 0.85   # per-update velocity decay when coasting


class Track:
    def __init__(self, tid, xy, label, step):
        self.id, self.label = tid, label
        self.p = np.asarray(xy, float)
        self.v = np.zeros(2)
        self.last_step = step
        self.miss = 0
        self.hist = [(step, *self.p)]


class Tracker:
    def __init__(self):
        self.tracks = []

    def init(self, xys, labels, step):
        self.tracks = [Track(i, xy, lab, step) for i, (xy, lab) in enumerate(zip(xys, labels))]

    def predict(self, step):
        out = []
        for t in self.tracks:
            dt = step - t.last_step
            out.append(t.p + t.v * dt)
        return np.array(out)

    def update(self, dets, step, frozen=()):
        """dets: (M,2) detections. frozen: track ids that must not move (e.g. held)."""
        dets = np.asarray(dets, float).reshape(-1, 2)
        pred = self.predict(step)
        n, m = len(self.tracks), len(dets)
        assigned = {}
        if n and m:
            C = np.linalg.norm(pred[:, None, :] - dets[None, :, :], axis=2)
            Cg = np.where(C < GATE, C, 1e3)
            r, c = linear_sum_assignment(Cg)
            for i, j in zip(r, c):
                if Cg[i, j] < 1e3:
                    assigned[i] = j
        for i, t in enumerate(self.tracks):
            dt = max(1, step - t.last_step)
            if i in assigned and t.id not in frozen:
                newp = dets[assigned[i]]
                vnew = (newp - t.p) / dt
                t.v = 0.5 * t.v + 0.5 * vnew if t.miss == 0 else vnew
                t.p = newp
                t.last_step = step
                t.miss = 0
            else:
                t.miss += 1
            t.hist.append((step, *t.p))
        return assigned

    def get(self, label):
        return [t for t in self.tracks if t.label == label]
