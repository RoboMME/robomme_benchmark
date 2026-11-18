"""Geometry helpers: rotations, a convention-free pixel+depth -> world model, rectangle fits."""
import math

import numpy as np

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None


# ----------------------------------------------------------------- rotations
def rotvec_to_mat(r):
    r = np.asarray(r, dtype=float)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def mat_to_rotvec(R):
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: axis from the symmetric part
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(A), 0, None))
        i = int(np.argmax(k))
        if k[i] < 1e-9:
            return np.zeros(3)
        axis = A[:, i] / k[i]
        axis = axis / np.linalg.norm(axis)
        # sign fix using off-diagonal skew part (tiny near pi but keeps consistency)
        w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if np.dot(w, axis) < 0:
            axis = -axis
        return axis * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def wrap_half_pi(a):
    """Wrap an angle to [-pi/4, pi/4) (square symmetry)."""
    q = math.pi / 2
    return (a + q / 2) % q - q / 2


# ----------------------------------------------------------------- camera model
class LinearCam:
    """world = A @ [u*d, v*d, d] + t  (exact for a pinhole camera with planar depth).

    Fitted from (u, v, raw_depth, world) samples returned by the official
    pixel_to_position tool, so no intrinsics/pose convention is assumed.
    """

    def __init__(self, A, t, resid):
        self.A = A
        self.t = t
        self.resid = resid
        try:
            self.Ainv = np.linalg.inv(A)
        except Exception:
            self.Ainv = None

    @staticmethod
    def fit(uvd, xyz):
        uvd = np.asarray(uvd, float)
        xyz = np.asarray(xyz, float)
        if len(uvd) < 8:
            return None
        u, v, d = uvd[:, 0], uvd[:, 1], uvd[:, 2]
        X = np.stack([u * d, v * d, d, np.ones_like(d)], 1)
        sol, *_ = np.linalg.lstsq(X, xyz, rcond=None)
        pred = X @ sol
        err = np.linalg.norm(pred - xyz, axis=1)
        # one robust re-fit dropping outliers (edge pixels, depth windows)
        keep = err < max(0.004, 3 * np.median(err))
        if keep.sum() >= 8 and keep.sum() < len(err):
            sol, *_ = np.linalg.lstsq(X[keep], xyz[keep], rcond=None)
            pred = X @ sol
            err = np.linalg.norm(pred - xyz, axis=1)
        A = sol[:3].T
        t = sol[3]
        return LinearCam(A, t, (float(np.median(err)), float(np.percentile(err, 90))))

    def backproject(self, us, vs, ds):
        us = np.asarray(us, float)
        vs = np.asarray(vs, float)
        ds = np.asarray(ds, float)
        M = np.stack([us * ds, vs * ds, ds], -1)
        return M @ self.A.T + self.t

    def project(self, xyz):
        if self.Ainv is None:
            return None
        q = (np.asarray(xyz, float) - self.t) @ self.Ainv.T
        d = q[..., 2]
        return q[..., 0] / d, q[..., 1] / d, d

    def u_axis_world(self):
        """World direction of increasing image column u (at fixed depth)."""
        a = self.A[:, 0]
        return a / (np.linalg.norm(a) + 1e-12)

    def v_axis_world(self):
        a = self.A[:, 1]
        return a / (np.linalg.norm(a) + 1e-12)


def pointmap(cam_model, depth, stride=1):
    """Dense world point map (H', W', 3) and valid mask for a depth image."""
    H, W = depth.shape[:2]
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    d = depth[vs, us].astype(float)
    pts = pinhole_backproject(cam_model, us, vs, d)
    valid = d > 0
    return pts, valid, us, vs


# ----------------------------------------------------------------- labeling
def label(mask):
    mask = np.asarray(mask, bool)
    if _ndi is not None:
        lab, n = _ndi.label(mask)
        return lab, n
    H, W = mask.shape
    lab = np.zeros((H, W), int)
    n = 0
    for y in range(H):
        for x in range(W):
            if mask[y, x] and lab[y, x] == 0:
                n += 1
                stack = [(y, x)]
                lab[y, x] = n
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        yy, xx = cy + dy, cx + dx
                        if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and lab[yy, xx] == 0:
                            lab[yy, xx] = n
                            stack.append((yy, xx))
    return lab, n


def dilate(mask, r=1):
    mask = np.asarray(mask, bool)
    if _ndi is not None:
        return _ndi.binary_dilation(mask, iterations=r)
    out = mask.copy()
    for _ in range(r):
        m = out.copy()
        m[1:] |= out[:-1]
        m[:-1] |= out[1:]
        m[:, 1:] |= out[:, :-1]
        m[:, :-1] |= out[:, 1:]
        out = m
    return out


def erode(mask, r=1):
    return ~dilate(~np.asarray(mask, bool), r)


# ----------------------------------------------------------------- rectangle fit
def min_area_rect(xy):
    """Min-area rectangle of 2-D points (robust percentiles).

    Returns (center(2), angle in [-pi/4, pi/4), (w, h)) or None.
    """
    xy = np.asarray(xy, float)
    if len(xy) < 6:
        return None
    best = None
    for deg in np.arange(-45.0, 45.0, 1.0):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, s], [-s, c]])  # rotate points by -a
        p = xy @ R.T
        lo = np.percentile(p, 2, axis=0)
        hi = np.percentile(p, 98, axis=0)
        ext = hi - lo
        area = ext[0] * ext[1]
        if best is None or area < best[0]:
            best = (area, a, lo, hi)
    area, a, lo, hi = best
    # refine with 0.25 deg steps around best
    for deg in np.arange(math.degrees(a) - 1.0, math.degrees(a) + 1.0, 0.25):
        aa = math.radians(deg)
        c, s = math.cos(aa), math.sin(aa)
        R = np.array([[c, s], [-s, c]])
        p = xy @ R.T
        l = np.percentile(p, 2, axis=0)
        h = np.percentile(p, 98, axis=0)
        e = h - l
        if e[0] * e[1] < area:
            area, a, lo, hi = e[0] * e[1], aa, l, h
    cen_r = (lo + hi) / 2
    c, s = math.cos(a), math.sin(a)
    cen = np.array([c * cen_r[0] - s * cen_r[1], s * cen_r[0] + c * cen_r[1]])
    return cen, wrap_half_pi(a), tuple(hi - lo)


# ----------------------------------------------------------------- pinhole + Kabsch fit
def kabsch(P, Q):
    """Rigid R, t minimising |R P + t - Q| (rows are points); works for coplanar points."""
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    pm, qm = P.mean(0), Q.mean(0)
    Hm = (P - pm).T @ (Q - qm)
    U, S, Vt = np.linalg.svd(Hm)
    dd = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    D = np.diag([1.0, 1.0, dd])
    R = Vt.T @ D @ U.T
    t = qm - R @ pm
    return R, t


def _cam_points(uvd, f, cx, cy, mode):
    u, v, d = uvd[:, 0], uvd[:, 1], uvd[:, 2]
    x = (u - cx) / f
    y = (v - cy) / f
    if mode == "ray":
        n = np.sqrt(x * x + y * y + 1.0)
        z = d / n
    else:
        z = d
    return np.stack([x * z, y * z, z], 1)


def fit_pinhole(uvd, xyz, W=256, H=256, f_cands=None):
    """Fit world = R [ (u-cx)/f z, (v-cy)/f z, z ] + t from tool samples.

    Searches focal length (and z-depth vs ray-distance) by Kabsch residual; returns a
    LinearCam (A = R K^-1 with z-depth) or None.  `uvd` depth must already be metric.
    """
    uvd = np.asarray(uvd, float)
    xyz = np.asarray(xyz, float)
    if len(uvd) < 6:
        return None
    cx, cy = W / 2.0, H / 2.0
    best = None

    def evaluate(f, mode, ccx=cx, ccy=cy):
        P = _cam_points(uvd, f, ccx, ccy, mode)
        R, t = kabsch(P, xyz)
        err = np.linalg.norm(P @ R.T + t - xyz, axis=1)
        return float(np.median(err) + 0.3 * np.mean(err)), R, t, err

    cands = list(f_cands or [])
    grid = list(np.exp(np.linspace(math.log(60), math.log(1500), 40)))
    for mode in ("z", "ray"):
        fs = [(f, cx, cy) for f in grid] + [c for c in cands if c is not None]
        for item in fs:
            f, ccx, ccy = item if isinstance(item, tuple) else (item, cx, cy)
            sc, R, t, err = evaluate(f, mode, ccx, ccy)
            if best is None or sc < best[0]:
                best = (sc, f, mode, ccx, ccy, R, t, err)
        # golden-section refine around the best grid f
        sc0, f0, m0, ccx0, ccy0 = best[:5]
        if m0 != mode:
            continue
        lo, hi = f0 / 1.1, f0 * 1.1
        for _ in range(30):
            a = lo + (hi - lo) * 0.382
            b = lo + (hi - lo) * 0.618
            if evaluate(a, mode, ccx0, ccy0)[0] < evaluate(b, mode, ccx0, ccy0)[0]:
                hi = b
            else:
                lo = a
        f1 = (lo + hi) / 2
        sc, R, t, err = evaluate(f1, mode, ccx0, ccy0)
        if sc < best[0]:
            best = (sc, f1, mode, ccx0, ccy0, R, t, err)
    sc, f, mode, ccx, ccy, R, t, err = best
    # robust refit without outliers
    keep = err < max(0.004, 3 * np.median(err))
    if 6 <= keep.sum() < len(err):
        P = _cam_points(uvd[keep], f, ccx, ccy, mode)
        R, t = kabsch(P, xyz[keep])
        P = _cam_points(uvd, f, ccx, ccy, mode)
        err = np.linalg.norm(P @ R.T + t - xyz, axis=1)
    Kinv = np.array([[1 / f, 0, -ccx / f], [0, 1 / f, -ccy / f], [0, 0, 1.0]])
    cam = LinearCam(R @ Kinv, t, (float(np.median(err)), float(np.percentile(err, 90))))
    cam.f = f
    cam.depth_mode = mode
    cam.cx, cam.cy = ccx, ccy
    cam.R = R
    return cam


def pinhole_backproject(cam, us, vs, ds):
    if getattr(cam, "depth_mode", "z") == "ray":
        x = (np.asarray(us, float) - cam.cx) / cam.f
        y = (np.asarray(vs, float) - cam.cy) / cam.f
        ds = np.asarray(ds, float) / np.sqrt(x * x + y * y + 1.0)
    return cam.backproject(us, vs, ds)


def square_fit(xy, anchor, side, search=0.015, step=0.0015):
    """Fit a square of known side to 2-D points near `anchor` (maximise points inside).

    Returns (center, yaw in [-pi/4, pi/4), inside_fraction_of_expected) or None.
    """
    xy = np.asarray(xy, float)
    anchor = np.asarray(anchor, float)
    if len(xy) < 10:
        return None
    sel = np.linalg.norm(xy - anchor, axis=1) < side * 0.75 + search + 0.01
    P = xy[sel] - anchor
    if len(P) < 10:
        return None
    if len(P) > 4000:
        P = P[np.random.RandomState(0).choice(len(P), 4000, replace=False)]
    offs = np.arange(-search, search + 1e-9, step)
    best = None
    h = side / 2
    for deg in np.arange(-45.0, 45.0, 2.0):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        q = P @ np.array([[c, -s], [s, c]])  # coordinates in square frame
        for ox in offs:
            inx = np.abs(q[:, 0] - ox) <= h
            if not inx.any():
                continue
            qy = q[inx, 1]
            for oy in offs:
                n = int((np.abs(qy - oy) <= h).sum())
                # tie-break towards the anchor
                sc = n - 0.5 * len(P) * 0.0 - (ox * ox + oy * oy) * 50.0 * len(P) / max(len(P), 1)
                if best is None or sc > best[0]:
                    best = (sc, a, ox, oy, n)
    if best is None:
        return None
    sc, a, ox, oy, n = best
    # refine angle +-2 deg in 0.5 deg steps at fixed centre search +-step
    for deg in np.arange(math.degrees(a) - 2, math.degrees(a) + 2.01, 0.5):
        aa = math.radians(deg)
        c, s = math.cos(aa), math.sin(aa)
        q = P @ np.array([[c, -s], [s, c]])
        for dx in (-step / 2, 0, step / 2):
            for dy in (-step / 2, 0, step / 2):
                nn = int(((np.abs(q[:, 0] - ox - dx) <= h) & (np.abs(q[:, 1] - oy - dy) <= h)).sum())
                scc = nn - ((ox + dx) ** 2 + (oy + dy) ** 2) * 50.0
                if scc > sc:
                    sc, a, n = scc, aa, nn
                    ox2, oy2 = ox + dx, oy + dy
                    ox, oy = ox2, oy2
    c, s = math.cos(a), math.sin(a)
    cen = anchor + np.array([c * ox - s * oy, s * ox + c * oy])
    return cen, wrap_half_pi(a), n / max(1, len(P))
