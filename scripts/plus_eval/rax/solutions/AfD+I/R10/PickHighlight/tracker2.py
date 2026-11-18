import numpy as np
from percep import components

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


def greedy_assign(C):
    pairs = []
    C = C.copy()
    while True:
        i, j = np.unravel_index(np.argmin(C), C.shape)
        if not np.isfinite(C[i, j]) or C[i, j] >= 1e8:
            break
        pairs.append((i, j))
        C[i, :] = 1e9; C[:, j] = 1e9
    return pairs


def front_components(P, tcp=None):
    m = (P[..., 2] > 0.034) & (P[..., 2] < 0.07) & (P[..., 0] > -0.5)
    comps = []
    for c in components(m, 8):
        pts = P[c[:, 0], c[:, 1]][:, :2]
        if tcp is not None and tcp[2] < 0.07:
            if np.linalg.norm(pts.mean(0) - tcp[:2]) < 0.06:
                continue
        comps.append(pts)
    return comps


class Tracker2:
    def __init__(self, init_xy):
        self.pos = np.array(init_xy, float).reshape(-1, 2)
        self.vel = np.zeros_like(self.pos)
        self.lost = np.zeros(len(self.pos), int)
        self.moved = np.zeros(len(self.pos))
        self.frozen = False
        self.regate = 0.0
        self.hold_r = 0.06
        self.hold_z = 0.15
        self.reacq = 0

    def update_P(self, P, tcp=None, gate=0.03, hold=False):
        if self.frozen or len(self.pos) == 0:
            return 0.0
        comps = front_components(P, tcp)
        pred = self.pos + self.vel
        # split merged components using predicted tracks inside them
        dets = []
        for pts in comps:
            mu = pts.mean(0)
            ext = (pts.max(0) - pts.min(0)).max()
            inside = [j for j in range(len(pred)) if np.min(np.linalg.norm(pts - pred[j][None], axis=1)) < 0.02]
            if ext > 0.06 and len(inside) >= 2:
                cent = pred[inside].copy()
                for _ in range(8):
                    lab = np.argmin(((pts[:, None] - cent[None]) ** 2).sum(-1), 1)
                    cent = np.array([pts[lab == k].mean(0) if (lab == k).any() else cent[k] for k in range(len(cent))])
                dets.extend(list(cent))
            else:
                dets.append(mu)
        dets = np.array(dets).reshape(-1, 2)
        maxd = 0.0
        assigned = np.zeros(len(pred), bool)
        if len(dets):
            C = np.linalg.norm(pred[:, None] - dets[None], axis=-1)
            C = np.where(C < gate, C, 1e9)
            if linear_sum_assignment is not None:
                ri, ci = linear_sum_assignment(C)
                pairs = [(i, j) for i, j in zip(ri, ci) if C[i, j] < 1e8]
            else:
                pairs = greedy_assign(C)
            held = np.zeros(len(pred), bool)
            if hold and tcp is not None and tcp[2] < self.hold_z:
                held = np.linalg.norm(self.pos - np.asarray(tcp[:2])[None], axis=1) < self.hold_r
            for i in np.where(held)[0]:
                self.vel[i] = 0; self.lost[i] = 0; assigned[i] = True
            for i, j in pairs:
                if held[i]:
                    continue
                v = dets[j] - self.pos[i]
                self.vel[i] = np.clip(0.7 * v + 0.3 * self.vel[i], -0.015, 0.015)
                self.moved[i] += np.linalg.norm(v)
                maxd = max(maxd, float(np.linalg.norm(v)))
                self.pos[i] = dets[j]
                self.lost[i] = 0
                assigned[i] = True
        # re-acquisition: lost/unmatched tracks vs leftover detections, wider gate
        if self.regate > 0 and len(dets) and (~assigned).any():
            used = set(j for _, j in pairs)
            free = [j for j in range(len(dets)) if j not in used]
            ua = list(np.where(~assigned)[0])
            if free:
                C2 = np.linalg.norm(pred[ua][:, None] - dets[free][None], axis=-1)
                C2 = np.where(C2 < self.regate, C2, 1e9)
                if linear_sum_assignment is not None:
                    r2, c2 = linear_sum_assignment(C2)
                    p2 = [(a, b) for a, b in zip(r2, c2) if C2[a, b] < 1e8]
                else:
                    p2 = greedy_assign(C2)
                for a, b in p2:
                    i = ua[a]; d = dets[free[b]]
                    self.moved[i] += np.linalg.norm(d - self.pos[i])
                    self.pos[i] = d; self.vel[i] = 0; self.lost[i] = 0; assigned[i] = True
                    self.reacq += 1
        for i in np.where(~assigned)[0]:
            self.lost[i] += 1
            self.pos[i] = self.pos[i] + self.vel[i]
            self.vel[i] *= 0.7
        return maxd
