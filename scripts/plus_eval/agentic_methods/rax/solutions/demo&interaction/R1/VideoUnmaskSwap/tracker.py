"""Generic multi-object tracker on 2-D point sets (no robot / IO dependencies).

Each frame supplies an (N,2) array of foreground points in a common plane (top-down
world metres in the controller; image pixels in offline tests). Objects are of
roughly known size `size` (side length). Tracks are updated by constrained k-means
seeded with constant-velocity predictions, with a minimum-separation guard so two
tracks cannot collapse onto one object.
"""
import numpy as np


class Tracker:
    def __init__(self, centers, size, radius_frac=0.75, min_sep_frac=0.55,
                 min_pts=6, vel_alpha=0.5, max_step_frac=0.6):
        self.c = np.asarray(centers, float).copy()
        self.v = np.zeros_like(self.c)
        self.size = float(size)
        self.R = radius_frac * size
        self.min_sep = min_sep_frac * size
        self.min_pts = min_pts
        self.alpha = vel_alpha
        self.max_step = max_step_frac * size
        self.support = np.zeros(len(self.c))
        self.lost = np.zeros(len(self.c), int)
        self.history = [self.c.copy()]
        self.events = []

    def step(self, pts, dt=1.0, frame=None, iters=4):
        K = len(self.c)
        pred = self.c + self.v * dt
        cen = pred.copy()
        pts = np.asarray(pts, float)
        cnt = np.zeros(K)
        if len(pts):
            for _ in range(iters):
                d = np.linalg.norm(pts[:, None, :] - cen[None, :, :], axis=2)
                a = d.argmin(1)
                ok = d[np.arange(len(pts)), a] < self.R
                new = cen.copy()
                cnt = np.zeros(K)
                for k in range(K):
                    sel = ok & (a == k)
                    cnt[k] = sel.sum()
                    if cnt[k] >= self.min_pts:
                        new[k] = pts[sel].mean(0)
                    else:
                        new[k] = pred[k]
                # limit jump relative to prediction
                dv = new - pred
                n = np.linalg.norm(dv, axis=1)
                big = n > self.max_step
                new[big] = pred[big] + dv[big] * (self.max_step / n[big])[:, None]
                cen = new
        # separation guard: if two tracks collapse, the weaker one reverts to prediction
        for _ in range(2):
            d = np.linalg.norm(cen[:, None] - cen[None], axis=2) + np.eye(K) * 1e9
            i, j = np.unravel_index(d.argmin(), d.shape)
            if d[i, j] >= self.min_sep:
                break
            weak = i if cnt[i] < cnt[j] else j
            cen[weak] = pred[weak] if np.linalg.norm(pred[weak] - cen[i if weak == j else j]) >= self.min_sep else self.c[weak]
            self.events.append((frame, 'collapse', int(i), int(j)))
        vel = (cen - self.c) / max(dt, 1e-6)
        seen = cnt >= self.min_pts
        self.v[seen] = self.alpha * vel[seen] + (1 - self.alpha) * self.v[seen]
        self.v[~seen] *= 0.5
        self.lost[seen] = 0
        self.lost[~seen] += 1
        self.support = cnt
        self.c = cen
        self.history.append(cen.copy())
        return cen


def refine(centers, pts, size, iters=8):
    """Plain k-means refinement of initial centers on one frame's points."""
    c = np.asarray(centers, float).copy()
    pts = np.asarray(pts, float)
    for _ in range(iters):
        d = np.linalg.norm(pts[:, None] - c[None], axis=2)
        a = d.argmin(1)
        ok = d[np.arange(len(pts)), a] < size
        for k in range(len(c)):
            sel = ok & (a == k)
            if sel.sum() >= 3:
                c[k] = pts[sel].mean(0)
    return c
