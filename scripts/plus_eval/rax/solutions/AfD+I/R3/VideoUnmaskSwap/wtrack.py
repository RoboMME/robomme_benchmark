"""World-frame (top-down XY) container tracker using RGB-D from front + wrist cameras."""
import numpy as np
from scipy import ndimage
from mcpio import depth_m, backproject
from colors import classify_rgb as classify

CELL = 0.0025
LIM = 0.45
NG = int(2 * LIM / CELL)
ZC = (0.05, 0.11)      # container upper body
ZK = (0.004, 0.042)    # cube points

def classify_old(c):
    r, g, b = c
    if r > 120 and g < 70 and b < 70 and r > 2 * g: return 'red'
    if g > r + 30 and g > b + 40: return 'green'
    if b > r + 40 and b > g + 30: return 'blue'
    return 'other'

def frame_points(fr, imgs):
    """Return list of (P Nx3, rgb Nx3) per camera, finite points only."""
    out = []
    for cam in ('front', 'wrist'):
        if (cam, 'depth') not in imgs: continue
        mpu = (fr.get('depth') or {}).get(cam, {}).get('meters_per_unit', 0.001)
        d = depth_m(imgs[(cam, 'depth')], mpu)
        P = backproject(d, fr['camera_intrinsics'][cam]['matrix'], fr['camera_poses'][cam])
        ok = np.isfinite(P).all(-1)
        out.append((P[ok], imgs[(cam, 'rgb')][ok][:, :3]))
    return out

def occupancy(pts_list, zr):
    occ = np.zeros((NG, NG), bool)
    for P, _ in pts_list:
        s = (P[:, 2] > zr[0]) & (P[:, 2] < zr[1]) & (np.abs(P[:, 0]) < LIM) & (np.abs(P[:, 1]) < LIM)
        ix = ((P[s, 0] + LIM) / CELL).astype(int); iy = ((P[s, 1] + LIM) / CELL).astype(int)
        occ[ix, iy] = True
    return occ

def occ_points(occ):
    ix, iy = np.nonzero(occ)
    return np.stack([ix * CELL - LIM + CELL / 2, iy * CELL - LIM + CELL / 2], 1)

def detect_cubes(pts_list, static=None):
    """Colored cube blobs in XY (cube-height points)."""
    acc = np.zeros((NG, NG, 3)); cnt = np.zeros((NG, NG))
    for P, rgb in pts_list:
        s = (P[:, 2] > ZK[0]) & (P[:, 2] < ZK[1]) & (np.abs(P[:, 0]) < LIM) & (np.abs(P[:, 1]) < LIM)
        ix = ((P[s, 0] + LIM) / CELL).astype(int); iy = ((P[s, 1] + LIM) / CELL).astype(int)
        np.add.at(acc, (ix, iy), rgb[s].astype(float)); np.add.at(cnt, (ix, iy), 1)
    occ0 = cnt > 0
    if static is not None: occ0 &= ~static
    occ = ndimage.binary_dilation(occ0, iterations=3)
    lab, n = ndimage.label(occ)
    cubes = []
    for i in range(1, n + 1):
        m = (lab == i) & occ0
        if m.sum() < 12 or m.sum() > 700: continue
        c = acc[m].sum(0) / max(cnt[m].sum(), 1)
        ix, iy = np.nonzero(m)
        cubes.append(dict(xy=[float(ix.mean() * CELL - LIM + CELL / 2), float(iy.mean() * CELL - LIM + CELL / 2)],
                          color=classify(c), rgb=c.round().tolist(), cells=int(m.sum())))
    # merge same-colour fragments
    merged = []
    for c in sorted(cubes, key=lambda c: -c['cells']):
        dup = next((m for m in merged if m['color'] == c['color'] and np.hypot(m['xy'][0] - c['xy'][0], m['xy'][1] - c['xy'][1]) < 0.035), None)
        if dup is None: merged.append(c)
        else: dup['cells'] += c['cells']
    return merged

class WorldTracker:
    def __init__(self, radius=0.03, alpha=0.6, iters=6, min_pts=20, dil=2):
        self.dil = dil
        self.radius, self.alpha, self.iters, self.min_pts = radius, alpha, iters, min_pts
        self.T = None; self.V = None; self.hist = []; self.cubes = None
        self.pre = None; self.started = False; self.static = None; self.nframes = 0; self.log = []

    def feed(self, fr, imgs):
        pts = frame_points(fr, imgs)
        occ = occupancy(pts, ZC)
        if self.static is None:
            self.static = ndimage.binary_dilation(occ, iterations=4)
            self.log.append(f'static cells {int(occ.sum())}')
        occ &= ~self.static
        n_c = int(occ.sum())
        self.nframes += 1
        if not self.started:
            if n_c < 300:
                self.pre = pts  # latest frame with no containers
                return
            # containers appeared
            if self.pre is None:
                self.pre = pts
            self.cubes = detect_cubes(self.pre, self.static)
            self.T = np.array([c['xy'] for c in self.cubes], float).reshape(-1, 2)
            self.V = np.zeros_like(self.T)
            self.started = True
            self.log.append(f'containers start at fed frame {self.nframes - 1}; cubes={[(c["color"], c["cells"]) for c in self.cubes]}')
        occd = ndimage.binary_dilation(occ, iterations=self.dil)
        self._update(occ_points(occd), occd)

    def _update(self, P, occd=None):
        T, V = self.T, self.V
        pred = T + V
        C = pred.copy(); cnt = np.zeros(len(C), int)
        for it in range(self.iters):
            if len(P) == 0 or len(C) == 0: break
            d = ((P[:, None, :] - C[None]) ** 2).sum(2)
            k = d.argmin(1); dm = d.min(1)
            newC = C.copy()
            for j in range(len(C)):
                sel = (k == j) & (dm < self.radius ** 2)
                cnt[j] = sel.sum()
                newC[j] = P[sel].mean(0) if cnt[j] >= self.min_pts else pred[j]
            C = newC
        ok = cnt >= self.min_pts
        if occd is not None and (~ok).any():
            lab, n = ndimage.label(occd)
            orphans = []
            for i in range(1, n + 1):
                ix, iy = np.nonzero(lab == i)
                if len(ix) < 60: continue
                c = np.array([ix.mean() * CELL - LIM + CELL / 2, iy.mean() * CELL - LIM + CELL / 2])
                if np.min(np.linalg.norm(C[ok] - c, axis=1), initial=1e9) > 0.035:
                    orphans.append(c)
            if orphans:
                from scipy.optimize import linear_sum_assignment
                lost = np.nonzero(~ok)[0]; O = np.array(orphans)
                D = np.linalg.norm(C[lost][:, None] - O[None], axis=2)
                r, c = linear_sum_assignment(D)
                for ri, ci in zip(r, c):
                    if D[ri, ci] < 0.15:
                        C[lost[ri]] = O[ci]; ok[lost[ri]] = True; self.log.append(f'reacquire track {lost[ri]} jump {D[ri,ci]:.3f} at {self.nframes}')
        V = np.where(ok[:, None], self.alpha * (C - T) + (1 - self.alpha) * V, 0.0)
        self.T, self.V = C, V
        self.hist.append((C.copy(), cnt.copy()))

    def result(self):
        return [dict(color=c['color'], rgb=c['rgb'], start=c['xy'], final=self.T[j].tolist(),
                     min_cnt=int(min(h[1][j] for h in self.hist)) if self.hist else 0)
                for j, c in enumerate(self.cubes or [])]
