"""Image-space container tracker for VideoUnmaskSwap (front camera RGB).
frames: list of HxWx3 uint8 arrays (front view), in time order.
"""
import numpy as np
from scipy import ndimage

def sat_mask(a):
    a = a.astype(int); mx = a.max(2); mn = a.min(2)
    return (mx - mn), mx, mn

def whiteish(a):
    s, mx, mn = sat_mask(a)
    return (mn > 135) & (s < 45)

def cube_blobs(a, static=None):
    """Return list of (x, y, color, size) colored blobs."""
    a = a.astype(int); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    s = a.max(2) - a.min(2)
    cols = {}
    cols['red'] = (r > 120) & (g < 70) & (b < 70)
    cols['green'] = (g > r + 50) & (g > b + 50)
    cols['blue'] = (b > r + 50) & (b > g + 30)
    out = []
    for name, m in cols.items():
        if static is not None: m = m & ~static
        lab, n = ndimage.label(m)
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            if 2 <= len(xs) <= 250:
                out.append((xs.mean(), ys.mean(), name, len(xs)))
    return out

def colored_count(a):
    a = a.astype(int); s = a.max(2) - a.min(2)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    # saturated, not table-orange/brown hue
    m = (s > 90) & ~((r > g) & (g > b) & (r - b > 60) & (r < 255) & (b < 150) & (g < 200) & (r > 120) & (g > 50))
    return m


def classify(c):
    r, g, b = c
    if r > 130 and g < 70 and b < 70: return 'red'
    if g > r + 30 and g > b + 50: return "green"
    if b > r + 50 and b > g + 50: return 'blue'
    return 'other'

def detect_cubes(bg, a, min_px=6):
    bg = bg.astype(int); a = a.astype(int)
    diff = np.abs(a - bg).sum(2) > 60
    lab, n = ndimage.label(diff)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < min_px or len(xs) > 600: continue
        c = a[ys, xs].mean(0)
        out.append((float(xs.mean()), float(ys.mean()), classify(c), len(xs), c.astype(int).tolist()))
    return out

class Tracker:
    def __init__(self, frames, radius=14, alpha=0.6, iters=6, is_exec=None):
        self.frames = [np.asarray(f) for f in frames]
        self.is_exec = is_exec if is_exec is not None else [False] * len(self.frames)
        self.radius, self.alpha, self.iters = radius, alpha, iters

    def step_more(self, frame):
        pass

    def find_phases(self):
        F = self.frames
        bg = F[0].astype(int)
        D = [int((np.abs(f.astype(int) - bg).sum(2) > 60).sum()) for f in F]
        W = [int(whiteish(f).sum()) for f in F]
        self.D, self.W = D, W
        base = W[0]
        cube_frame = next(i for i in range(len(F)) if D[i] > 150)
        cont_start = next(i for i in range(cube_frame, len(F)) if W[i] > base + 300)
        return cube_frame, cont_start

    def run(self):
        F = self.frames
        cf, cs = self.find_phases()
        cf = cs - 1
        self.cube_frame, self.cont_start = cf, cs
        static = ndimage.binary_dilation(whiteish(F[0]), iterations=2)
        self.static = static
        top = np.zeros_like(static); top[:28] = True
        exec_static = top
        self.cubes = detect_cubes(F[0], F[cf])
        self.N = len(self.cubes)
        T = np.array([[c[0], c[1] - 4] for c in self.cubes], float)
        V = np.zeros_like(T)
        hist = []; npx = []
        for i in range(cs, len(F)):
            m = whiteish(F[i]) & ~(exec_static if self.is_exec[i] else static)
            ys, xs = np.nonzero(m)
            P = np.stack([xs, ys], 1).astype(float)
            pred = T + V
            C = pred.copy(); cnt = np.zeros(len(C), int)
            for it in range(self.iters):
                if len(P) == 0: break
                d = ((P[:, None, :] - C[None]) ** 2).sum(2)
                k = d.argmin(1); dm = d.min(1)
                newC = C.copy()
                for j in range(len(C)):
                    sel = (k == j) & (dm < self.radius ** 2)
                    cnt[j] = sel.sum()
                    newC[j] = P[sel].mean(0) if cnt[j] >= 5 else pred[j]
                C = newC
            ok = cnt >= 5
            newV = C - T
            V = np.where(ok[:, None], self.alpha * newV + (1 - self.alpha) * V, 0.0)
            T = C
            hist.append(T.copy()); npx.append(cnt.copy())
        self.hist = hist; self.npx = npx
        self.assignment = [dict(color=c[2], rgb=c[4], start=[c[0], c[1]], final=self.hist[-1][j].tolist())
                           for j, c in enumerate(self.cubes)]
        return self.assignment
