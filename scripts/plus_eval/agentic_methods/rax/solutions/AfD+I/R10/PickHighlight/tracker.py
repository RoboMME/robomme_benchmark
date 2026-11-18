import numpy as np
from percep import backproject

def front_points(R):
    P = R.world("front")
    m = (P[..., 2] > 0.034) & (P[..., 2] < 0.07) & (P[..., 0] > -0.5)
    pts = P[m]
    st = R.state()
    if st[2] < 0.2:
        far = np.linalg.norm(pts[:, :2] - st[None, :2], axis=1) > 0.05
        pts = pts[far]
    return pts[:, :2]

class Tracker:
    def __init__(self, init_xy):
        self.pos = np.array(init_xy, float).reshape(-1, 2)
        self.vel = np.zeros_like(self.pos)
        self.lost = np.zeros(len(self.pos), int)
        self.moved = np.zeros(len(self.pos))

    def update(self, pts, r=0.028, iters=3):
        if len(self.pos) == 0:
            return 0.0
        pred = self.pos + self.vel
        newp = pred.copy()
        for _ in range(iters):
            d = np.linalg.norm(pts[:, None] - newp[None], axis=-1)  # N x T
            nearest = d.argmin(1)
            dmin = d.min(1)
            for j in range(len(newp)):
                sel = (nearest == j) & (dmin < r)
                if sel.sum() >= 6:
                    newp[j] = pts[sel].mean(0)
        maxd = 0.0
        for j in range(len(newp)):
            d = np.linalg.norm(pts - newp[j][None], axis=1)
            cnt = int((d < r).sum())
            if cnt >= 6:
                v = newp[j] - self.pos[j]
                self.vel[j] = np.clip(0.6 * v, -0.012, 0.012)
                self.moved[j] += np.linalg.norm(v)
                maxd = max(maxd, float(np.linalg.norm(v)))
                self.pos[j] = newp[j]
                self.lost[j] = 0
            else:
                self.lost[j] += 1
                self.vel[j] *= 0.5
                self.pos[j] = self.pos[j] + self.vel[j]
        return maxd

def detect_front_cubes(R):
    """Initial cube list from front camera: cluster points."""
    from percep import components
    P = R.world("front")
    m = (P[..., 2] > 0.034) & (P[..., 2] < 0.07) & (P[..., 0] > -0.5)
    out = []
    for c in components(m, 15):
        pts = P[c[:, 0], c[:, 1]][:, :2]
        ext = (pts.max(0) - pts.min(0)).max()
        if ext > 0.075:
            k = int(min(4, np.ceil(ext / 0.05)))
            cent = pts[np.linspace(0, len(pts) - 1, k).astype(int)]
            for _ in range(15):
                lab = np.argmin(((pts[:, None] - cent[None]) ** 2).sum(-1), 1)
                cent = np.array([pts[lab == j].mean(0) if (lab == j).any() else cent[j] for j in range(k)])
            out.extend(list(cent))
        else:
            out.append(pts.mean(0))
    return np.array(out).reshape(-1, 2)
