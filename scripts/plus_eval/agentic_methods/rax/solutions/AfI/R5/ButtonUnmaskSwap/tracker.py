"""Track containers (top caps) in the front camera's world point cloud."""
import numpy as np
from scipy import ndimage

CAP_ZMIN, CAP_ZMAX = 0.05, 0.095

def cap_points(pw, ok, ee=None, xlim=(-0.45, 0.4), ylim=(-0.5, 0.5)):
    z = pw[..., 2]
    m = ok & (z > CAP_ZMIN) & (z < CAP_ZMAX) & (pw[..., 0] > xlim[0]) & (pw[..., 0] < xlim[1]) & (pw[..., 1] > ylim[0]) & (pw[..., 1] < ylim[1])
    if ee is not None and ee[2] < 0.25:
        dxy = np.linalg.norm(pw[..., :2] - np.asarray(ee[:2]), axis=-1)
        m &= ~(dxy < 0.05)
    return pw[m][:, :2], m

def cube_points(pw, ok, rgb, ee=None):
    """Colored cube pixels (z 0.005-0.045) excluding buttons area handled by caller."""
    z = pw[..., 2]
    m = ok & (z > 0.008) & (z < 0.045)
    return m

class Tracker:
    def __init__(self, centers, names):
        self.c = np.array(centers, float)[:, :2]
        self.v = np.zeros_like(self.c)
        self.names = list(names)
        self.last_step = None
        self.hist = []
        self.lost = np.zeros(len(self.c), int)

    def update(self, pts, step_id, radius=0.035, iters=4):
        dt = 1 if self.last_step is None else max(1, step_id - self.last_step)
        pred = self.c + self.v * dt
        if len(pts) == 0:
            self.c = pred; self.last_step = step_id; self.lost += 1
            return
        cen = pred.copy()
        cnt = np.zeros(len(cen))
        for _ in range(iters):
            d = np.linalg.norm(pts[:, None, :] - cen[None, :, :], axis=-1)  # P x N
            a = d.argmin(1)
            dm = d[np.arange(len(pts)), a]
            new = cen.copy()
            for i in range(len(cen)):
                sel = (a == i) & (dm < radius)
                cnt[i] = sel.sum()
                if cnt[i] >= 3:
                    new[i] = pts[sel].mean(0)
            cen = new
        moved = cnt >= 3
        newv = np.where(moved[:, None], (cen - self.c) / dt, self.v * 0.5)
        self.v = 0.5 * self.v + 0.5 * newv
        # limit velocity
        sp = np.linalg.norm(self.v, axis=1, keepdims=True)
        self.v = np.where(sp > 0.012, self.v / np.maximum(sp, 1e-9) * 0.012, self.v)
        self.c = np.where(moved[:, None], cen, pred)
        self.lost = np.where(moved, 0, self.lost + 1)
        self.last_step = step_id
        self._reacquire(pts, radius)
        self.hist.append((step_id, self.c.copy(), cnt.copy()))

    def _reacquire(self, pts, radius):
        lost = [i for i in range(len(self.c)) if self.lost[i] >= 3]
        if not lost:
            return
        d = np.linalg.norm(pts[:, None, :] - self.c[None, :, :], axis=-1)
        free = pts[d.min(1) > radius * 1.1]
        for i in sorted(lost, key=lambda i: -self.lost[i]):
            if len(free) < 25:
                return
            j = np.argmin(np.linalg.norm(free - self.c[i], axis=1))
            c = free[j]
            for _ in range(4):
                sel = np.linalg.norm(free - c, axis=1) < radius
                c = free[sel].mean(0)
            sel = np.linalg.norm(free - c, axis=1) < radius
            if sel.sum() < 25:
                return
            self.c[i] = c; self.v[i] = 0; self.lost[i] = 0
            self.reacq = getattr(self, 'reacq', []) + [(self.last_step, self.names[i], c.copy())]
            free = free[~sel]

    def pos(self, name):
        return self.c[self.names.index(name)]
