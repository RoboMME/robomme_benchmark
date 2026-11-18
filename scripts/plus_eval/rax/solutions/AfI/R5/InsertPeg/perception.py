import numpy as np
from geom import Cam

def cloud(depth_m, intr, pose):
    c = Cam(intr, pose)
    vv, uu = np.mgrid[0:depth_m.shape[0], 0:depth_m.shape[1]]
    P = c.backproject(uu.ravel(), vv.ravel(), depth_m.ravel()).reshape(depth_m.shape + (3,))
    return P, depth_m > 0

def min_area_yaw(xy):
    """yaw (mod 90 deg) of the min-area bounding rectangle of 2D points; returns yaw, center, size"""
    best = None
    c0 = xy.mean(0)
    for a in np.radians(np.arange(0, 90, 0.5)):
        Rm = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        q = (xy - c0) @ Rm.T
        lo, hi = np.percentile(q, 1, 0), np.percentile(q, 99, 0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            mid = (lo + hi) / 2
            center = c0 + mid @ Rm
            best = (area, a, center, hi - lo)
    return best[1], best[2], best[3]

def find_box_top(P, valid, zmin=0.06, zmax=0.12, region=None):
    """Box = the tallest object; its top face points are near max z."""
    z = P[..., 2]
    m = valid & (z > zmin) & (z < zmax)
    if region is not None:
        x0, x1, y0, y1 = region
        m &= (P[..., 0] > x0) & (P[..., 0] < x1) & (P[..., 1] > y0) & (P[..., 1] < y1)
    pts = P[m]
    if len(pts) < 30:
        return None
    ztop = np.percentile(pts[:, 2], 90)
    top = pts[np.abs(pts[:, 2] - ztop) < 0.006]
    yaw, center, size = min_area_yaw(top[:, :2])
    return dict(ztop=float(ztop), yaw=float(yaw), center=center, size=size, n=len(top))


def peg_rows(depth_m, dmax=0.095, vmin=0, vmax=190, umin=60, umax=200):
    """For each image row, centre column and mean depth of near (in-hand) pixels."""
    out = {}
    for v in range(vmin, vmax):
        row = depth_m[v, umin:umax]
        m = (row > 0.01) & (row < dmax)
        if m.sum() >= 6:
            us = np.where(m)[0] + umin
            # keep the largest contiguous run
            splits = np.where(np.diff(us) > 1)[0]
            runs = np.split(us, splits + 1)
            run = max(runs, key=len)
            if len(run) >= 6:
                out[v] = (float(run.mean()), float(np.mean(depth_m[v, run])), len(run))
    return out


def peg_offset(depth_ref, depth_cur, fx=128.0, fy=128.0):
    """Camera-frame offset (OpenGL: x right, y up, z back) of the in-hand peg in cur vs ref,
    from row-wise comparisons of near pixels. Returns (dp_cam[3], n_rows, info)."""
    a = peg_rows(depth_ref); b = peg_rows(depth_cur)
    rows = sorted(set(a) & set(b))
    if len(rows) < 15:
        return None, len(rows), {}
    du = np.array([b[v][0] - a[v][0] for v in rows])
    dd = np.array([b[v][1] - a[v][1] for v in rows])
    dw = np.array([b[v][2] - a[v][2] for v in rows])
    d = np.array([a[v][1] for v in rows])
    dx = float(np.median(du * d / fx))
    dz = float(np.median(dd))
    info = dict(du_px=float(np.median(du)), dd_mm=dz * 1000, dwidth_px=float(np.median(dw)),
                rows=len(rows), d_ref=float(np.median(d)),
                dd_top=float(np.median(dd[: len(dd) // 3]) * 1000), dd_bot=float(np.median(dd[-len(dd) // 3:]) * 1000))
    return np.array([dx, 0.0, -dz]), len(rows), info


def wrist_box_points(depth_m, intr, pose, peg_band=None, vmax=200, dmin=0.02, dmax=0.16, step=2):
    """World points seen by the wrist camera excluding the in-hand peg band and the fingers."""
    c = Cam(intr, pose)
    H, W = depth_m.shape
    vv, uu = np.mgrid[0:vmax:step, 0:W:step]
    d = depth_m[vv, uu]
    m = (d > dmin) & (d < dmax)
    if peg_band is not None:
        u0, u1, vtop = peg_band
        m &= ~((uu >= u0) & (uu <= u1) & (vv >= vtop))
    P = c.backproject(uu[m], vv[m], d[m])
    return P


def icp_translation(A, B, iters=30, trim=0.8, max_shift=0.01):
    """translation t minimizing |(B + t) - nn_A(B + t)|; A reference, B current. returns t (B->A)"""
    t = np.zeros(3)
    if len(A) < 50 or len(B) < 50:
        return None, None
    rng = np.random.default_rng(0)
    if len(A) > 2500:
        A = A[rng.choice(len(A), 2500, replace=False)]
    if len(B) > 1200:
        B = B[rng.choice(len(B), 1200, replace=False)]
    for _ in range(iters):
        Q = B + t
        d2 = ((Q[:, None, :] - A[None, :, :]) ** 2).sum(-1)
        j = np.argmin(d2, 1)
        dist = np.sqrt(d2[np.arange(len(Q)), j])
        keep = dist <= np.quantile(dist, trim)
        step = (A[j[keep]] - Q[keep]).mean(0)
        t = t + step
        if np.linalg.norm(t) > max_shift:
            t = t / np.linalg.norm(t) * max_shift
        if np.linalg.norm(step) < 1e-5:
            break
    Q = B + t
    d2 = ((Q[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    res = float(np.median(np.sqrt(d2.min(1))))
    return t, res


def _rv_to_R(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def peg_line_tool(depth_m, intr, cam_pose, tcp_state):
    """Fit the in-hand peg's top-surface centre line; return (point, unit dir) in the TOOL frame."""
    from geom import quat_to_R
    rows = peg_rows(depth_m)
    if len(rows) < 20:
        return None
    vs = np.array(sorted(rows))
    us = np.array([rows[v][0] for v in vs]); ds = np.array([rows[v][1] for v in vs])
    c = Cam(intr, cam_pose)
    Pw = c.backproject(us, vs.astype(float), ds)
    Rt = _rv_to_R(tcp_state[3:6]); pt = np.asarray(tcp_state[:3], float)
    Pt = (Pw - pt) @ Rt  # tool frame
    m = Pt.mean(0)
    U, S, Vt = np.linalg.svd(Pt - m)
    e = Vt[0]
    if e[0] < 0:
        e = -e  # peg points along +tool x (toward the tip)
    # robust refit: drop outliers
    r = np.linalg.norm(np.cross(Pt - m, e), axis=1)
    keep = r < max(0.0015, np.quantile(r, 0.9))
    m = Pt[keep].mean(0)
    U, S, Vt = np.linalg.svd(Pt[keep] - m)
    e = Vt[0] if Vt[0][0] >= 0 else -Vt[0]
    return m, e, int(keep.sum())


def peg_correction(ref, cur, max_rot_deg=2.0, max_trans=0.003):
    """ref/cur = (point, dir) tool-frame lines (video, ours). Return tool-frame correction (R_d, t_d)
    mapping our peg onto the video's peg: X_v = R_d X_o + t_d."""
    mv, ev = ref[0], ref[1]
    mo, eo = cur[0], cur[1]
    ax = np.cross(eo, ev); s = np.linalg.norm(ax); cth = float(np.dot(eo, ev))
    ang = np.arctan2(s, cth)
    if s > 1e-12:
        ax = ax / s
    ang = min(ang, np.radians(max_rot_deg))
    Rd = _rv_to_R(ax * ang) if s > 1e-12 else np.eye(3)
    q = Rd @ mo
    # closest point on the video line to q
    foot = mv + ev * np.dot(q - mv, ev)
    td = foot - q
    n = np.linalg.norm(td)
    if n > max_trans:
        td = td / n * max_trans
    return Rd, td, float(np.degrees(ang)), ax
