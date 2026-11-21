"""World-frame container tracking + motion for ButtonUnmaskSwap."""
import time, json, os
import numpy as np
from scipy import ndimage
from obs import decode, world_points

BAND = (0.05, 0.115)     # container body heights (cubes/buttons are below 0.05)
RES = 0.004
X0, Y0, NX, NY = -0.35, -0.45, 160, 225   # grid over workspace


def grid_points(P):
    """Unique XY cells (centres) of points."""
    ix = ((P[:, 0] - X0) / RES).astype(int); iy = ((P[:, 1] - Y0) / RES).astype(int)
    ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
    G = np.zeros((NX, NY), bool); G[ix[ok], iy[ok]] = True
    return G


def cells_to_xy(G):
    cx, cy = np.nonzero(G)
    return np.stack([cx * RES + X0 + RES / 2, cy * RES + Y0 + RES / 2], 1)


WRIST_ONLY_Z = 0.40


def band_grid(meta, imgs, lo=BAND[0], hi=BAND[1], fuse=True):
    Gs = []
    cams = ("front", "wrist")
    try:
        if fuse and meta["camera_poses"]["wrist"]["position_m"][2] > WRIST_ONLY_Z:
            cams = ("wrist",)
    except Exception:
        pass
    for cam in cams:
        try:
            pw, _ = world_points(meta, imgs, cam)
        except Exception:
            continue
        s = (pw[:, 2] > lo) & (pw[:, 2] < hi)
        Gs.append(grid_points(pw[s]))
    G = Gs[0]
    for g in Gs[1:]:
        G = G | g
    return G


def low_objects(meta, imgs):
    """Clusters of objects 1.2..6 cm tall at reset: cubes (small) and buttons (large)."""
    G = band_grid(meta, imgs, 0.012, 0.06)
    lab, n = ndimage.label(ndimage.binary_closing(G, iterations=1))
    cubes, buttons = [], []
    for i in range(1, n + 1):
        cx, cy = np.nonzero(lab == i)
        if len(cx) < 20:
            continue
        xy = (cx.mean() * RES + X0 + RES / 2, cy.mean() * RES + Y0 + RES / 2)
        (buttons if len(cx) > 220 else cubes).append((xy, len(cx)))
    return cubes, buttons


class WorldTracker:
    def __init__(self, seeds, radius=0.035, static=None):
        self.c = np.array(seeds, float)
        self.v = np.zeros_like(self.c)
        self.radius = radius
        self.static = static
        self.hist = [self.c.copy()]
        self.support = np.zeros(len(self.c), int)

    def update(self, G, ee_xy=None, ee_r=0.0):
        G = G.copy()
        if self.static is not None:
            G &= ~self.static
        P = cells_to_xy(G)
        if ee_xy is not None and ee_r > 0 and len(P):
            P = P[np.linalg.norm(P - np.asarray(ee_xy)[None], axis=1) > ee_r]
        pred = self.c + self.v
        cen = pred.copy()
        sup = np.zeros(len(cen), int)
        if len(P):
            for _ in range(4):
                d = ((P[:, None, :] - cen[None]) ** 2).sum(2)
                a = d.argmin(1); dm = d[np.arange(len(P)), a]
                for k in range(len(cen)):
                    sel = (a == k) & (dm < self.radius ** 2)
                    sup[k] = sel.sum()
                    if sup[k] >= 8:
                        # blend with prediction when support is weak (occlusion/merge)
                        w = min(1.0, sup[k] / 60.0)
                        cen[k] = w * P[sel].mean(0) + (1 - w) * pred[k]
                    else:
                        cen[k] = pred[k]
        vel = cen - self.c
        self.v = 0.6 * self.v + 0.4 * vel
        weak = sup < 8
        self.v[weak] *= 0.5          # coasting decays; no runaway tracks
        cen[:, 0] = np.clip(cen[:, 0], -0.33, 0.30)
        cen[:, 1] = np.clip(cen[:, 1], -0.43, 0.43)
        sp = np.linalg.norm(self.v, axis=1)
        lim = 0.012
        self.v[sp > lim] *= (lim / sp[sp > lim])[:, None]
        self.c = cen
        self.support = sup
        self.hist.append(self.c.copy())
        return self.c


def static_mask(meta, imgs):
    """Tall static things present at reset (pillars) in the container band."""
    G = band_grid(meta, imgs)
    return ndimage.binary_dilation(G, iterations=4)


class BlobTracker:
    """Blob-association tracker: tracks assigned to connected cell blobs; merged blobs are split by
    k-means seeded at predictions, and fully-merged tracks coast on their prior velocity."""

    def __init__(self, seeds, static=None, sep=0.03, **_):
        self.c = np.array(seeds, float)
        self.v = np.zeros_like(self.c)
        self.static = static
        self.sep = sep
        self.hist = [self.c.copy()]
        self.support = np.zeros(len(self.c), int)
        self.coast = np.zeros(len(self.c), int)

    def update(self, G, ee_xy=None, ee_r=0.0):
        G = G.copy()
        if self.static is not None:
            G &= ~self.static
        if ee_xy is not None and ee_r > 0:
            P0 = cells_to_xy(np.ones_like(G))
        lab, n = ndimage.label(G, structure=np.ones((3, 3)))
        P = cells_to_xy(G)
        L = lab[G]
        if ee_xy is not None and ee_r > 0 and len(P):
            keep = np.linalg.norm(P - np.asarray(ee_xy)[None], axis=1) > ee_r
            P, L = P[keep], L[keep]
        K = len(self.c)
        pred = self.c + self.v
        new = pred.copy()
        newv = self.v.copy()
        sup = np.zeros(K, int)
        if len(P):
            d = np.linalg.norm(P[:, None, :] - pred[None], axis=2)  # (N,K)
            near = d.argmin(0)
            dn = d[near, np.arange(K)]
            blob_of = np.where(dn < 0.03, L[near], -1)
            for b in set(blob_of.tolist()):
                if b < 0:
                    continue
                ks = [k for k in range(K) if blob_of[k] == b]
                Q = P[L == b]
                if len(ks) == 1:
                    k = ks[0]
                    # restrict to points within 4.5 cm of prediction (blob may hold untracked neighbours)
                    sel = np.linalg.norm(Q - pred[k], axis=1) < 0.045
                    if sel.sum() >= 8:
                        new[k] = Q[sel].mean(0); sup[k] = sel.sum()
                else:
                    cen = pred[ks].copy()
                    for _ in range(5):
                        dd = np.linalg.norm(Q[:, None, :] - cen[None], axis=2)
                        a = dd.argmin(1)
                        for j in range(len(ks)):
                            s = (a == j) & (dd[:, j] < 0.045)
                            if s.sum() >= 8:
                                cen[j] = Q[s].mean(0)
                    # pairs that collapsed together coast on prediction
                    for j, k in enumerate(ks):
                        others = [cen[i] for i in range(len(ks)) if i != j]
                        mind = min(np.linalg.norm(cen[j] - o) for o in others)
                        if mind < self.sep:
                            new[k] = pred[k]; sup[k] = 0; self.coast[k] += 1
                        else:
                            new[k] = cen[j]; sup[k] = 20
        # re-acquire: unclaimed blobs go to the nearest unsupported track
        if len(P):
            claimed = set(int(b) for b in blob_of.tolist() if b >= 0)
            for b in set(L.tolist()) - claimed:
                Q = P[L == b]
                if len(Q) < 40:
                    continue
                q = Q.mean(0)
                cand = [k for k in range(K) if sup[k] == 0]
                if not cand:
                    break
                dist = [np.linalg.norm(new[k] - q) for k in cand]
                j = int(np.argmin(dist))
                if dist[j] < 0.12:
                    k = cand[j]; new[k] = q; sup[k] = 1; self.v[k] *= 0.0; self.c[k] = q
        for k in range(K):
            if sup[k] > 0:
                vel = new[k] - self.c[k]
                newv[k] = 0.6 * self.v[k] + 0.4 * vel
                self.coast[k] = 0
            else:
                newv[k] = self.v[k] * 0.9
                self.coast[k] += 1
        sp = np.linalg.norm(newv, axis=1); lim = 0.015
        newv[sp > lim] *= (lim / sp[sp > lim])[:, None]
        new[:, 0] = np.clip(new[:, 0], -0.33, 0.30)
        new[:, 1] = np.clip(new[:, 1], -0.43, 0.43)
        self.c, self.v, self.support = new, newv, sup
        self.hist.append(self.c.copy())
        return self.c
