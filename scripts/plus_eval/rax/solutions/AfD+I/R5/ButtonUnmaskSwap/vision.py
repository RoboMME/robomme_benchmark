"""2-D front-camera perception for ButtonUnmaskSwap: cubes, buttons, container tracking."""
import numpy as np

COLORS = ("red", "green", "blue")


def white_mask(im):
    im = im.astype(int)
    mn = im.min(2); rng = im.max(2) - mn
    return (mn > 120) & (rng < 40)


def color_masks(im):
    im = im.astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    out = {}
    out["red"] = (r > 140) & (g < 60) & (b < 60)
    out["green"] = (g > 130) & (r < 90) & (b < 90)
    out["blue"] = (b > 120) & (r < 60) & (g < 60)
    return out


def blob_centroids(mask, min_px=6):
    """Return list of (u, v, n) for connected components."""
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    res = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) >= min_px:
            res.append((xs.mean(), ys.mean(), len(xs)))
    return res


def detect_cubes(im):
    out = {}
    for c, m in color_masks(im).items():
        bl = blob_centroids(m, 4)
        if bl:
            bl.sort(key=lambda t: -t[2])
            out[c] = bl[0]
    return out


class Tracker:
    """Tracks K container centroids in image space with constant-velocity prediction
    and nearest-centre pixel assignment."""

    def __init__(self, centers, radius=16.0, exclude=None):
        self.c = np.array(centers, float)
        self.v = np.zeros_like(self.c)
        self.radius = radius
        self.exclude = exclude  # boolean mask of static pixels to ignore (buttons)
        self.hist = [self.c.copy()]
        self.lost = np.zeros(len(self.c), int)

    def update(self, im, extra_exclude=None, dt=1.0):
        m = white_mask(im)
        if self.exclude is not None:
            m &= ~self.exclude
        if extra_exclude is not None:
            m &= ~extra_exclude
        ys, xs = np.nonzero(m)
        P = np.stack([xs, ys], 1).astype(float)
        pred = self.c + self.v * dt
        new = pred.copy()
        if len(P):
            cen = pred.copy()
            for _ in range(3):
                d = ((P[:, None, :] - cen[None]) ** 2).sum(2)
                a = d.argmin(1)
                dm = d[np.arange(len(P)), a]
                for k in range(len(cen)):
                    sel = (a == k) & (dm < self.radius ** 2)
                    if sel.sum() >= 15:
                        cen[k] = P[sel].mean(0)
                        self.lost[k] = 0
                    else:
                        cen[k] = pred[k]
            new = cen
        vel = (new - self.c) / dt
        self.v = 0.5 * self.v + 0.5 * vel
        self.c = new
        self.hist.append(self.c.copy())
        return self.c
