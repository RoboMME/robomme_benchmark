"""Wrist-depth peg measurement: back-project near pixels above the fingers and fit a 3-D line."""
import numpy as np
from scipy.spatial.transform import Rotation as R


def backproject(dep, K, pose, us, vs):
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    d = dep[vs, us]
    pc = np.stack([(us - cx) * d / fx, -(vs - cy) * d / fy, -d], axis=1)
    q = pose["quaternion_wxyz"]
    rot = R.from_quat([q[1], q[2], q[3], q[0]])
    return rot.apply(pc) + np.asarray(pose["position_m"])


def peg_points(dep, K, pose, max_depth=0.095, min_depth=0.03, row_max=200):
    """Peg pixels in a wrist view while holding the peg: close depth, above the finger region."""
    vs, us = np.nonzero((dep > min_depth) & (dep < max_depth))
    keep = vs < row_max
    vs, us = vs[keep], us[keep]
    if len(vs) < 50:
        return None, None
    # keep the largest column band around the median column (removes finger edges)
    med = np.median(us)
    keep = np.abs(us - med) < 40
    vs, us = vs[keep], us[keep]
    return backproject(dep, K, pose, us, vs), (us, vs)


def peg_line(dep, K, pose, **kw):
    pts, pix = peg_points(dep, K, pose, **kw)
    if pts is None or len(pts) < 50:
        return None
    c = pts.mean(0)
    # direction: horizontal principal axis
    xy = pts[:, :2] - c[:2]
    w, V = np.linalg.eigh(xy.T @ xy)
    d = V[:, -1]
    us, vs = pix
    # orient direction toward the image top (away from the gripper = the free end)
    top = pts[vs < np.percentile(vs, 10)].mean(0)
    if np.dot(top[:2] - c[:2], d) < 0:
        d = -d
    proj = xy @ d
    tip = c[:2] + d * np.percentile(proj, 98)
    return {"center": c, "dir": np.array([d[0], d[1], 0.0]), "yaw": float(np.degrees(np.arctan2(d[1], d[0]))),
            "top_z": float(np.percentile(pts[:, 2], 90)), "tip_xy": tip, "n": int(len(pts)),
            "len": float(np.percentile(proj, 98) - np.percentile(proj, 2))}


def tcp_matrix(state):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(np.asarray(state[3:6], dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(state[:3], dtype=float)
    return T


def peg_frame(dep, K, pose, state, axis_hint):
    """World pose (4x4) of the in-hand peg's top face: x = peg axis toward the free end, z = face normal (up).
    Points are restricted to a box around the expected in-hand peg region relative to the TCP."""
    tcp = np.asarray(state[:3], dtype=float)
    vs, us = np.nonzero((dep > 0.02) & (dep < 0.2))
    pts = backproject(dep, K, pose, us, vs)
    rel = pts - tcp[None, :]
    ah = np.asarray(axis_hint[:2], dtype=float)
    ah = ah / np.linalg.norm(ah)
    latv = np.array([-ah[1], ah[0]])
    al = rel[:, :2] @ ah
    la = rel[:, :2] @ latv
    sel = (al > 0.015) & (al < 0.07) & (np.abs(la) < 0.015) & (np.abs(rel[:, 2]) < 0.016)
    pts = pts[sel]
    if len(pts) < 500:
        return None
    z90 = np.percentile(pts[:, 2], 90)
    top = pts[pts[:, 2] > z90 - 0.003]
    if len(top) < 300:
        return None
    c = top.mean(0)
    X = top - c
    w, V = np.linalg.eigh(X.T @ X)
    n = V[:, 0]
    if n[2] < 0:
        n = -n
    a = V[:, 2]
    a = a - n * np.dot(a, n)
    a /= np.linalg.norm(a)
    if np.dot(a[:2], ah) < 0:
        a = -a
    y = np.cross(n, a)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = a, y, n, c
    return T, int(len(top))


def box_mask(dep, K, pose, state, axis_hint, zmin=0.012, amin=0.07):
    """Pixels of scene structure above the table (box) excluding the in-hand peg and gripper."""
    H, W = dep.shape
    vs, us = np.mgrid[0:H, 0:W]
    us, vs = us.ravel(), vs.ravel()
    d = dep[vs, us]
    ok = d > 0.025
    pts = backproject(dep, K, pose, us, vs)
    tcp = np.asarray(state[:3], dtype=float)
    rel = pts - tcp[None, :]
    ah = np.asarray(axis_hint[:2], dtype=float)
    ah = ah / np.linalg.norm(ah)
    latv = np.array([-ah[1], ah[0]])
    al = rel[:, :2] @ ah
    la = rel[:, :2] @ latv
    peg = (al > -0.03) & (al < 0.085) & (np.abs(la) < 0.02) & (np.abs(rel[:, 2]) < 0.025)
    m = ok & (~peg) & (pts[:, 2] > zmin) & (al > amin)
    return m.reshape(H, W), pts.reshape(H, W, 3)


def _bilinear(img, u, v):
    H, W = img.shape
    u0 = np.clip(np.floor(u).astype(int), 0, W - 2)
    v0 = np.clip(np.floor(v).astype(int), 0, H - 2)
    a = np.clip(u - u0, 0, 1)
    b = np.clip(v - v0, 0, 1)
    return (img[v0, u0] * (1 - a) * (1 - b) + img[v0, u0 + 1] * a * (1 - b) +
            img[v0 + 1, u0] * (1 - a) * b + img[v0 + 1, u0 + 1] * a * b)


def register_shift(dv, mv, dl, ml, rng=14, clipm=0.01):
    """Find image shift (du, dv) so that live depth at (u+du, v+dv) matches video depth at (u, v) on the video box mask.
    Returns (du, dv, cost, n)."""
    vv, uu = np.nonzero(mv)
    if len(vv) < 200:
        return None
    if len(vv) > 6000:
        idx = np.linspace(0, len(vv) - 1, 6000).astype(int)
        vv, uu = vv[idx], uu[idx]
    ref = dv[vv, uu]
    mlf = ml.astype(float)
    dl_masked = np.where(ml, dl, 0.0)

    def cost(du, dvv):
        u = uu + du
        v = vv + dvv
        inside = (u >= 0) & (u < dv.shape[1] - 1) & (v >= 0) & (v < dv.shape[0] - 1)
        w = _bilinear(mlf, u, v)
        val = _bilinear(dl_masked, u, v) / np.maximum(w, 1e-6)
        good = inside & (w > 0.99)
        if good.sum() < 100:
            return 1e9
        r = np.minimum(np.abs(val[good] - ref[good]), clipm)
        # penalise reference pixels whose match falls outside the live mask
        return (r.sum() + clipm * (len(ref) - good.sum())) / len(ref)

    best = (1e9, 0, 0)
    for du in range(-rng, rng + 1):
        for dvv in range(-rng, rng + 1):
            c = cost(du, dvv)
            if c < best[0]:
                best = (c, du, dvv)
    c0, bu, bv = best
    for step in [0.5, 0.25, 0.1]:
        improved = True
        while improved:
            improved = False
            for du, dvv in [(bu + step, bv), (bu - step, bv), (bu, bv + step), (bu, bv - step)]:
                c = cost(du, dvv)
                if c < c0:
                    c0, bu, bv, improved = c, du, dvv, True
    return bu, bv, c0, len(ref)
