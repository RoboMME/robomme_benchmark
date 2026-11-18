"""Video wrist-camera pose from scene geometry (fallback / check for the frame state).

The wrist camera is static during the conditioning video but at a different pose than
at execution start.  When the frames' robot state is missing or inconsistent, the pose
is recovered from the last video frame's wrist depth:
  1. fit the table plane in camera coordinates (RANSAC) -> camera tilt and height;
  2. segment containers standing on the plane -> centroids in a yaw-free frame;
  3. 2-D rigid registration (pair RANSAC) to the containers seen at execution start.
"""
import math
import numpy as np

import vu_geom as G
import vu_percep as PC


def _rot_a_to_b(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # 180 deg about any axis perpendicular to a
        p = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        k = np.cross(a, p)
        k /= np.linalg.norm(k)
        return G.rotvec_to_R(k * math.pi)
    k = v / s
    return G.rotvec_to_R(k * math.atan2(s, c))


def fit_plane(pts, rng, iters=200, tol=0.006):
    """RANSAC plane through Nx3 points: (normal, d) with n.p + d = 0 and inlier mask."""
    n_pts = len(pts)
    if n_pts < 50:
        return None
    best = None
    for _ in range(iters):
        idx = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = pts[idx]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -float(n @ p0)
        inl = np.abs(pts @ n + d) < tol
        c = int(inl.sum())
        if best is None or c > best[0]:
            best = (c, n, d, inl)
    if best is None:
        return None
    c, n, d, inl = best
    # refine by least squares on inliers
    q = pts[inl]
    cen = q.mean(0)
    _, _, vt = np.linalg.svd(q - cen, full_matrices=False)
    n = vt[-1]
    d = -float(n @ cen)
    inl = np.abs(pts @ n + d) < tol
    return n, d, inl


def register_2d(V, W, pair_tol=0.012, inl_tol=0.018):
    """Rigid 2-D transform (theta, t) mapping points V onto W (no correspondences known)."""
    V = np.asarray(V, float)
    W = np.asarray(W, float)
    if len(V) < 2 or len(W) < 2:
        return None
    best = None
    nv, nw = len(V), len(W)
    for i in range(nv):
        for j in range(nv):
            if i == j:
                continue
            dv = V[j] - V[i]
            lv = np.linalg.norm(dv)
            if lv < 0.03:
                continue
            for k in range(nw):
                dw_all = W - W[k]
                lw = np.linalg.norm(dw_all, axis=1)
                for l in np.nonzero(np.abs(lw - lv) < pair_tol)[0]:
                    if l == k:
                        continue
                    dw = dw_all[l]
                    th = math.atan2(dw[1], dw[0]) - math.atan2(dv[1], dv[0])
                    c, s = math.cos(th), math.sin(th)
                    Rm = np.array([[c, -s], [s, c]])
                    t = W[k] - Rm @ V[i]
                    Vt = V @ Rm.T + t
                    dd = np.linalg.norm(Vt[:, None, :] - W[None, :, :], axis=2).min(1)
                    inl = dd < inl_tol
                    score = int(inl.sum()) - float(np.mean(np.minimum(dd, inl_tol))) * 10
                    if best is None or score > best[0]:
                        best = (score, th, t, int(inl.sum()), float(np.median(dd[inl])) if inl.any() else 1.0)
    if best is None:
        return None
    score, th, t, n_in, med = best
    # refine with the inlier correspondences (Procrustes)
    c, s = math.cos(th), math.sin(th)
    Rm = np.array([[c, -s], [s, c]])
    Vt = V @ Rm.T + t
    D = np.linalg.norm(Vt[:, None, :] - W[None, :, :], axis=2)
    j = D.argmin(1)
    ok = D[np.arange(len(V)), j] < inl_tol
    if ok.sum() >= 2:
        A, B = V[ok], W[j[ok]]
        ca, cb = A.mean(0), B.mean(0)
        H = (A - ca).T @ (B - cb)
        th = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
        c, s = math.cos(th), math.sin(th)
        Rm = np.array([[c, -s], [s, c]])
        t = cb - Rm @ ca
        Vt = V @ Rm.T + t
        dd = np.linalg.norm(Vt[:, None, :] - W[None, :, :], axis=2).min(1)
        n_in = int((dd < inl_tol).sum())
        med = float(np.median(dd[dd < inl_tol])) if n_in else 1.0
    return dict(theta=th, t=t, n_inliers=n_in, n_v=len(V), median=med)


def centroids_from_points(P, table_z):
    """Container-like component centroids (xy) from an HxWx3 world-point image."""
    comps, _ = PC.segment_objects(P, table_z, min_h=0.015, max_h=0.15, min_px=12)
    out = []
    for c in comps:
        parts = [c]
        cx, cy, w, h, a = PC.robust_rect(c['pts'][:, :2])
        if max(w, h) > 0.085:
            sp = PC.split_by_top(P, c, table_z)
            if len(sp) >= 2:
                parts = sp
        for p in parts:
            cx, cy, w, h, a = PC.robust_rect(p['pts'][:, :2])
            if 0.025 < min(w, h) and max(w, h) < 0.09:
                tb = p['pts'][p['pts'][:, 2] > p['top'] - 0.015]
                if len(tb) < 5:
                    tb = p['pts']
                out.append([float(np.median(tb[:, 0])), float(np.median(tb[:, 1]))])
    return np.array(out).reshape(-1, 2)


def pose_quality(cam, depth, table_z, overview_xy):
    """How well a candidate wrist pose explains the video's containers: dict or None."""
    P = cam.backproject_image(depth)
    z = P[..., 2]
    ok = np.isfinite(z)
    if ok.sum() < 200:
        return None
    tz = PC.table_height(z[ok].ravel(), lo=table_z - 0.3, hi=table_z + 0.3)
    V = centroids_from_points(P, table_z)
    if len(V) == 0 or len(overview_xy) == 0:
        return dict(n=len(V), n_in=0, med=1.0, table_err=abs((tz if tz is not None else 9) - table_z))
    dd = np.linalg.norm(V[:, None, :] - np.asarray(overview_xy)[None, :, :], axis=2).min(1)
    return dict(n=len(V), n_in=int((dd < 0.02).sum()), med=float(np.median(dd)),
                table_err=abs((tz if tz is not None else 9) - table_z))


def estimate_wrist_pose(depth, cam_template, table_z, overview_xy, seed=0, log=print):
    """Return a Camera (world pose) for a static video wrist view, or None."""
    rng = np.random.default_rng(seed)
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W]
    ok = np.isfinite(depth) & (depth > 0.05) & (depth < 3.0)
    if ok.sum() < 300:
        return None
    pc = cam_template.cam_points(uu[ok] + 0.0, vv[ok] + 0.0, depth[ok])     # camera coords
    sub = pc[rng.choice(len(pc), min(4000, len(pc)), replace=False)]
    fp = fit_plane(sub, rng)
    if fp is None:
        return None
    n, d, inl = fp
    if inl.mean() < 0.3:
        log('reg: table plane covers only', round(float(inl.mean()), 2))
        return None
    # normal pointing towards the camera (camera origin has n.0 + d = d)
    if d < 0:
        n, d = -n, -d
    h = d                               # camera height above the plane
    R0 = _rot_a_to_b(n, np.array([0, 0, 1.0]))     # camera -> yaw-free world
    t0 = np.array([0, 0, table_z + h])
    C0 = cam_template.copy_with_pose(R0, t0)
    P0 = C0.backproject_image(depth)
    V = centroids_from_points(P0, table_z)
    oa = R0 @ cam_template.optical_axis() if cam_template.conv == 'opengl' else R0 @ np.array([0, 0, 1.0])
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(oa[2])))))
    log(f'reg: camera height {h:.3f} tilt {tilt:.1f} deg, {len(V)} video containers vs {len(overview_xy)} overview')
    if len(V) < 3 or len(overview_xy) < 3:
        return None
    reg = register_2d(V, overview_xy)
    if reg is None:
        return None
    log(f"reg: inliers {reg['n_inliers']}/{reg['n_v']} median {reg['median']:.4f} "
        f"theta {math.degrees(reg['theta']):.1f}")
    if reg['n_inliers'] < max(3, int(0.6 * len(V))) or reg['median'] > 0.012:
        return None
    Rz = G.Rz(reg['theta'])
    R = Rz @ R0
    t = Rz @ t0
    t[:2] += reg['t']
    t[2] = table_z + h
    return cam_template.copy_with_pose(R, t)
