import numpy as np
from rlib import backproject

ZLO, ZHI = 0.064, 0.088
COLORS = {"red": (200, 3, 3), "green": (4, 193, 4), "blue": (3, 3, 191)}

def table_points(meta, depth, cam="front"):
    return backproject(meta, depth, cam)

def detect_cubes(meta, rgb, depth, cam="front"):
    """Return dict name->xy for colored cubes (red/green/blue)."""
    P = backproject(meta, depth, cam)
    z = P[..., 2]
    m = (z > 0.01) & (z < 0.06) & (P[..., 0] > -0.35) & (depth > 0.12)
    rgbf = rgb.astype(float)
    out = {}
    for name, c in COLORS.items():
        dist = np.linalg.norm(rgbf - np.array(c, float), axis=-1)
        mm = m & (dist < 60)
        if mm.sum() >= 10:
            out[name] = P[mm][:, :2].mean(0)
    return out

def wrist_top_points(meta, wdepth):
    P = backproject(meta, wdepth, "wrist")
    z = P[..., 2]
    m = (wdepth > 0.12) & (z > ZLO) & (z < ZHI) & (P[..., 0] > -0.40) & (P[..., 0] < 0.30) & (np.abs(P[..., 1]) < 0.40)
    return P[m][:, :2]

def top_mask(meta, depth, ee=None):
    P = backproject(meta, depth)
    z = P[..., 2]
    m = (z > ZLO) & (z < ZHI) & (P[..., 0] > -0.40) & (P[..., 0] < 0.30) & (np.abs(P[..., 1]) < 0.40)
    if ee is not None and ee[2] < 0.25:
        d = np.linalg.norm(P[..., :2] - np.array(ee[:2]), axis=-1)
        m &= d > 0.045
    return P, m

class Tracker:
    def __init__(self, centers, labels):
        self.p = np.array(centers, float)
        self.v = np.zeros_like(self.p)
        self.labels = list(labels)
        self.miss = np.zeros(len(self.p), int)
        self.counts = np.zeros(len(self.p), int)
        self.hist = [self.p.copy()]
        self.events = []
        self.gate = 0.05
        self.typ = np.full(len(self.p), 60.0)

    def update(self, meta, depth, ee=None, radius=0.035, wdepth=None):
        P, m = top_mask(meta, depth, ee)
        pts = P[m][:, :2]
        if wdepth is not None:
            try:
                wp = wrist_top_points(meta, wdepth)
                if len(wp):
                    pts = np.concatenate([pts, wp[::2]], 0)
            except Exception:
                pass
        pred = self.p + self.v
        c = pred.copy()
        cnt = np.zeros(len(c), int)
        if len(pts):
            for _ in range(3):
                d = np.linalg.norm(pts[:, None, :] - c[None], axis=-1)
                a = d.argmin(1); dm = d.min(1)
                newc = c.copy()
                for k in range(len(c)):
                    sel = (a == k) & (dm < radius)
                    cnt[k] = sel.sum()
                    if cnt[k] >= 15:
                        newc[k] = pts[sel].mean(0)
                    else:
                        newc[k] = pred[k]
                c = newc
        for k in range(len(c)):
            if cnt[k] >= 15:
                if cnt[k] >= 40:
                    nv = c[k] - self.p[k]
                    if np.linalg.norm(nv) > 0.012: nv = nv / np.linalg.norm(nv) * 0.012
                    self.v[k] = 0.6 * self.v[k] + 0.4 * nv
                    self.p[k] = c[k]
                else:  # weakly seen (partly occluded): blend with prediction, keep velocity
                    self.p[k] = 0.5 * pred[k] + 0.5 * c[k]
                self.miss[k] = 0
            else:
                self.p[k] = pred[k]; self.miss[k] += 1
                if self.miss[k] > 20: self.v[k] *= 0.85
        self.counts = cnt
        # lost-track recovery: unclaimed blob -> nearest missing track
        missing = [k for k in range(len(self.p)) if self.miss[k] >= 3]
        if missing and len(pts):
            from scipy import ndimage
            lab, n = ndimage.label(m)
            for i in range(1, n + 1):
                sel = lab == i
                if sel.sum() < 40: continue
                cc = P[sel][:, :2].mean(0)
                live = [k for k in range(len(self.p)) if self.miss[k] == 0]
                if live and min(np.linalg.norm(self.p[k] - cc) for k in live) < 0.035: continue
                if not missing: break
                k = min(missing, key=lambda j: np.linalg.norm(self.p[j] - cc))
                if np.linalg.norm(self.p[k] - cc) > self.gate + 0.004 * self.miss[k]: continue
                self.p[k] = cc; self.v[k] = 0; self.miss[k] = 0; missing.remove(k)
                self.events.append((len(self.hist), k, cc.tolist()))
        self.hist.append(self.p.copy())
        return self.p

def init_from_cubes(meta, depth, cube_xy):
    """cube_xy: dict label->xy of all 8 cubes. Containers spawn on cubes."""
    labels = list(cube_xy.keys())
    return Tracker([cube_xy[l] for l in labels], labels)

def components(meta, depth, ee=None, min_px=30):
    from scipy import ndimage
    P, m = top_mask(meta, depth, ee)
    lab, n = ndimage.label(m)
    out = []
    for i in range(1, n + 1):
        sel = lab == i
        if sel.sum() >= min_px:
            out.append(P[sel][:, :2].mean(0))
    return out

def init_tracker(meta, depth, cubes):
    cs = components(meta, depth)
    labels = []
    for c in cs:
        lab = None
        for name, xy in cubes.items():
            if np.linalg.norm(c - xy) < 0.03:
                lab = name
        labels.append(lab)
    return Tracker(cs, labels)


def union_points(meta, fdepth, wdepth=None, ee=None):
    P, m = top_mask(meta, fdepth, ee)
    pts = P[m][:, :2]
    if wdepth is not None:
        wp = wrist_top_points(meta, wdepth)
        if len(wp): pts = np.concatenate([pts, wp], 0)
    return pts

def grid_components(pts, res=0.005, min_cells=8):
    from scipy import ndimage
    if len(pts) == 0: return []
    lo = pts.min(0) - 0.01
    ij = ((pts - lo) / res).astype(int)
    G = np.zeros(ij.max(0) + 2, bool); G[ij[:, 0], ij[:, 1]] = True
    lab, n = ndimage.label(G)
    out = []
    for i in range(1, n + 1):
        cells = np.argwhere(lab == i)
        if len(cells) >= min_cells:
            out.append(lo + (cells.mean(0) + 0.5) * res)
    return out

def init_tracker_union(meta, fdepth, wdepth, cubes):
    cs = grid_components(union_points(meta, fdepth, wdepth))
    labels = []
    for c in cs:
        lab = None
        for name, xy in cubes.items():
            if np.linalg.norm(c - xy) < 0.03: lab = name
        labels.append(lab)
    return Tracker(cs, labels)
