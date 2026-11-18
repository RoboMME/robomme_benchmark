"""Small geometry helpers."""
import math
import numpy as np

DOWN = np.diag([1.0, -1.0, -1.0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def R2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def R_down(yaw):
    return Rz(yaw) @ DOWN


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def R_goal(yaw, pitch=0.0):
    """Down-pointing gripper with yaw, then rotated about the hand y (finger) axis by pitch."""
    return R_down(yaw) @ Ry(pitch)


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


def ang(v):
    return math.atan2(v[1], v[0])


def unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0


def perp(v):
    return np.array([-v[1], v[0]])


class ObstacleGrid:
    """2-D max-height map of obstacle points (1 cm cells)."""

    def __init__(self, pts, res=0.01, lim=0.9):
        self.res, self.lim = res, lim
        n = int(2 * lim / res) + 1
        self.h = np.full((n, n), -1.0)
        if len(pts):
            ij = np.floor((pts[:, :2] + lim) / res).astype(int)
            ok = (ij >= 0).all(1) & (ij < n).all(1)
            for (i, j), z in zip(ij[ok], pts[ok, 2]):
                if z > self.h[i, j]: self.h[i, j] = z

    def dilated(self, margin):
        if not hasattr(self, '_dil'): self._dil = {}
        r = int(math.ceil(margin / self.res))
        if r not in self._dil:
            h = self.h; n = h.shape[0]; out = h.copy()
            pad = np.full((n + 2 * r, n + 2 * r), -1.0); pad[r:r + n, r:r + n] = h
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if di * di + dj * dj > r * r + 1: continue
                    out = np.maximum(out, pad[r + di:r + di + n, r + dj:r + dj + n])
            self._dil[r] = out
        return self._dil[r]

    def heights(self, xy, margin):
        """Per-point max obstacle height within margin."""
        xy = np.atleast_2d(xy); D = self.dilated(margin); n = D.shape[0]
        ij = np.floor((xy + self.lim) / self.res).astype(int)
        ok = (ij >= 0).all(1) & (ij < n).all(1)
        out = np.full(len(xy), -1.0)
        out[ok] = D[ij[ok, 0], ij[ok, 1]]
        return out

    def max_height_near(self, xy, margin):
        """Max obstacle height within `margin` of any of the xy points."""
        xy = np.atleast_2d(xy); n = self.h.shape[0]; r = int(math.ceil(margin / self.res))
        best = -1.0
        ij = np.floor((xy + self.lim) / self.res).astype(int)
        for i, j in set(map(tuple, ij.tolist())):
            i0, i1, j0, j1 = max(0, i - r), min(n, i + r + 1), max(0, j - r), min(n, j + r + 1)
            if i0 >= i1 or j0 >= j1: continue
            m = self.h[i0:i1, j0:j1].max()
            if m > best: best = m
        return best
