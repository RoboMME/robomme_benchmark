"""Convention-free camera models fitted to robomme_pixel_to_position samples.

Model A (depth available): world = M @ [d*u, d*v, d, 1]   (3x4, linear least squares)
Model B (no depth):        DLT projection matrix from world <-> pixel pairs
Model C (fallback):        table-plane homography pixel -> (x, y)
All models expose ray(u, v) -> (C, dir) where possible and plane_point(u, v, z).
"""
import numpy as np


class CamModel:
    def __init__(self):
        self.kind = None
        self.M = None
        self.P = None
        self.H = None
        self.z_table = None
        self.C = None
        self.resid = None
        self.Cplane = None   # camera centre used with the table homography

    def ray_dist(self, C, uv, xyz):
        """Mean distance of 3-D points from the rays C -> H(u, v) on the table (needs H)."""
        ds = []
        for (u, v), P in zip(uv, xyz):
            q = np.array([*self._h(u, v), self.z_table])
            d = q - C
            n = np.linalg.norm(d)
            if n < 1e-6:
                continue
            d = d / n
            w = np.asarray(P) - C
            ds.append(np.linalg.norm(w - (w @ d) * d))
        return float(np.median(ds)) if ds else 1e9

    # ------------------------------------------------------------ fitting
    def fit_depth(self, uv, d, xyz):
        uv, d, xyz = np.asarray(uv, float), np.asarray(d, float), np.asarray(xyz, float)
        X = np.stack([d * uv[:, 0], d * uv[:, 1], d, np.ones_like(d)], 1)
        M, *_ = np.linalg.lstsq(X, xyz, rcond=None)
        M = M.T
        res = np.linalg.norm(X @ M.T - xyz, axis=1)
        # one robust refit
        keep = res < max(3 * np.median(res), 0.004)
        if keep.sum() >= 8 and keep.sum() < len(res):
            M, *_ = np.linalg.lstsq(X[keep], xyz[keep], rcond=None)
            M = M.T
            res = np.linalg.norm(X[keep] @ M.T - xyz[keep], axis=1)
        self.M, self.kind = M, "depth"
        self.C = M[:, 3].copy()
        self.resid = float(np.median(res))
        return self.resid

    def fit_dlt(self, uv, xyz):
        uv, xyz = np.asarray(uv, float), np.asarray(xyz, float)
        n = len(uv)
        if n < 8:
            return None
        # normalise
        mu, su = uv.mean(0), uv.std() + 1e-9
        mx, sx = xyz.mean(0), xyz.std() + 1e-9
        un = (uv - mu) / su
        xn = (xyz - mx) / sx
        A = []
        for (u, v), (x, y, z) in zip(un, xn):
            A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
            A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
        A = np.array(A)
        _, s, Vt = np.linalg.svd(A)
        Pn = Vt[-1].reshape(3, 4)
        Tu = np.array([[su, 0, mu[0]], [0, su, mu[1]], [0, 0, 1]])
        Tx = np.array([[1 / sx, 0, 0, -mx[0] / sx], [0, 1 / sx, 0, -mx[1] / sx], [0, 0, 1 / sx, -mx[2] / sx], [0, 0, 0, 1]])
        P = Tu @ Pn @ Tx
        # camera centre = null space of P
        _, _, Vp = np.linalg.svd(P)
        Ch = Vp[-1]
        C = Ch[:3] / Ch[3]
        self.P, self.C, self.kind = P, C, "dlt"
        proj = self.project(xyz)
        self.resid = float(np.median(np.linalg.norm(proj - uv, axis=1)))  # pixels
        return self.resid

    def fit_homography(self, uv, xy):
        uv, xy = np.asarray(uv, float), np.asarray(xy, float)
        A = []
        for (u, v), (x, y) in zip(uv, xy):
            A.append([u, v, 1, 0, 0, 0, -x * u, -x * v, -x])
            A.append([0, 0, 0, u, v, 1, -y * u, -y * v, -y])
        _, _, Vt = np.linalg.svd(np.array(A))
        self.H = Vt[-1].reshape(3, 3)
        if self.kind is None:
            self.kind = "homography"
        pr = np.array([self._h(u, v) for u, v in uv])
        return float(np.median(np.linalg.norm(pr - xy, axis=1)))

    def _h(self, u, v):
        p = self.H @ np.array([u, v, 1.0])
        return p[:2] / p[2]

    # ------------------------------------------------------------ queries
    def project(self, xyz):
        xyz = np.atleast_2d(np.asarray(xyz, float))
        if self.kind == "dlt":
            X = np.concatenate([xyz, np.ones((len(xyz), 1))], 1)
            p = X @ self.P.T
            return p[:, :2] / p[:, 2:3]
        if self.kind == "depth":
            A = self.M[:, :3]
            q = np.linalg.solve(A, (xyz - self.M[:, 3]).T).T  # [d u, d v, d]
            return q[:, :2] / q[:, 2:3]
        return None

    def ray(self, u, v):
        if self.kind == "depth":
            d = self.M[:, :3] @ np.array([u, v, 1.0])
            return self.C, d / np.linalg.norm(d)
        if self.kind == "dlt":
            Pm = self.P[:, :3]
            d = np.linalg.solve(Pm, np.array([u, v, 1.0]))
            d = d / np.linalg.norm(d)
            # orient the ray from the camera towards the scene (downwards to the table)
            if self.z_table is not None and (self.C[2] - self.z_table) * (-d[2]) < 0:
                d = -d
            return self.C, d
        return None, None

    def plane_point(self, u, v, z):
        """World point where pixel (u, v)'s ray meets the horizontal plane at height z."""
        if self.H is not None and self.Cplane is not None and self.z_table is not None:
            q = np.array([*self._h(u, v), self.z_table])
            C = self.Cplane
            den = C[2] - self.z_table
            if abs(den) > 1e-3:
                return C + (q - C) * (C[2] - z) / den
        C, d = self.ray(u, v)
        if C is not None and abs(d[2]) > 1e-6:
            t = (z - C[2]) / d[2]
            if t > 0:
                return C + t * d
        if self.H is not None:
            x, y = self._h(u, v)
            return np.array([x, y, self.z_table if self.z_table is not None else z])
        return None

    def backproject(self, u, v, depth):
        if self.kind == "depth":
            return self.M @ np.array([depth * u, depth * v, depth, 1.0])
        return None
