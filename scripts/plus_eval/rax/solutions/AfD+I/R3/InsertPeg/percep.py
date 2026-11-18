"""Depth-based perception helpers."""
import base64, io, json
import numpy as np
from PIL import Image


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(depth_m, K, pose):
    """depth (H,W) metres -> world points (H,W,3); invalid -> nan."""
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = depth_m
    pc = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1)
    R = quat_to_mat(pose["quaternion_wxyz"])
    pw = pc @ R.T + np.asarray(pose["position_m"])
    pw[d <= 0] = np.nan
    return pw


def load_depth_png(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        im = Image.open(io.BytesIO(path_or_bytes))
    else:
        im = Image.open(path_or_bytes)
    return np.asarray(im).astype(np.float64) * 0.001


def images_from_reply(reply):
    """Return dict {(camera, kind): np.array} from an MCP reply's image blocks."""
    content = reply.get("content", [])
    meta = None
    for b in content:
        if b.get("type") == "text":
            try:
                meta = json.loads(b["text"]); break
            except Exception:
                pass
    out = {}
    if not meta:
        return meta, out
    for a in meta.get("attachments", []):
        idx = a.get("content_index")
        if idx is None or idx >= len(content):
            continue
        blk = content[idx]
        if blk.get("type") != "image":
            continue
        raw = base64.b64decode(blk["data"])
        im = Image.open(io.BytesIO(raw))
        if a["kind"] == "depth":
            out[(a["camera"], "depth")] = np.asarray(im).astype(np.float64) * 0.001
        else:
            out[(a["camera"], "rgb")] = np.asarray(im.convert("RGB"))
    return meta, out


def peg_in_hand(depth_m, K, pose, ee_pos, max_reach=0.16, zband=(0.002, 0.016)):
    """Estimate the held peg's top-surface points from the wrist depth.
    Returns dict(dir=unit xy axis pointing away from EE (3,), lateral point, z_top, tip, n) or None."""
    P = backproject(depth_m, K, pose)
    ok = np.isfinite(P[..., 0])
    rel = P - np.asarray(ee_pos)
    dz = rel[..., 2]
    hr = np.hypot(rel[..., 0], rel[..., 1])
    m = ok & (dz > zband[0]) & (dz < zband[1]) & (hr < max_reach)
    pts = P[m]
    if len(pts) < 30:
        return None
    xy = pts[:, :2] - np.asarray(ee_pos)[:2]
    # principal axis
    c = xy.mean(0)
    U, S, Vt = np.linalg.svd(xy - c, full_matrices=False)
    d = Vt[0]
    proj = (xy - c) @ d
    # point direction away from EE: the far end of peg is away from EE
    if np.mean(proj) < 0 and abs(np.median(xy @ d)) > 0:  # noop guard
        pass
    if (c @ d) < 0:
        d = -d
    proj = xy @ d   # along axis from EE
    perp = xy @ np.array([-d[1], d[0]])
    return dict(dir=np.array([d[0], d[1], 0.0]), off_perp=float(np.median(perp)),
                z_top=float(np.median(pts[:, 2])), tip=float(np.percentile(proj, 99)),
                tail=float(np.percentile(proj, 1)), n=int(len(pts)), mask=m)


def peg_line(depth_m, K, pose, ee_pos, u, a_rng=(0.0, 0.07), zband=(0.004, 0.016), reach=0.075):
    """Fit the held peg's top-surface line relative to the EE in a fixed world frame:
    u = unit horizontal insertion direction (2,), w = left normal. Returns
    dict(ps, po, zs, zo, n): perp = ps*a + po, ztop_rel = zs*a + zo, where a is distance along u from EE."""
    u = np.asarray(u, float)[:2]; u = u / np.linalg.norm(u)
    w = np.array([-u[1], u[0]])
    P = backproject(depth_m, K, pose)
    rel = P - np.asarray(ee_pos)
    ok = np.isfinite(rel[..., 0])
    m = ok & (rel[..., 2] > zband[0]) & (rel[..., 2] < zband[1]) & (np.hypot(rel[..., 0], rel[..., 1]) < reach)
    q = rel[m]
    if len(q) < 200:
        return None
    a = q[:, :2] @ u; p = q[:, :2] @ w; z = q[:, 2]
    sel = (a > a_rng[0]) & (a < a_rng[1])
    if sel.sum() < 200:
        return None
    A = np.stack([a[sel], np.ones(sel.sum())], 1)
    cp, res_p = np.linalg.lstsq(A, p[sel], rcond=None)[:2]
    cz = np.linalg.lstsq(A, z[sel], rcond=None)[0]
    # robust refit: drop outliers
    rp = p[sel] - A @ cp
    keep = np.abs(rp) < max(0.004, 3 * np.std(rp))
    cp = np.linalg.lstsq(A[keep], p[sel][keep], rcond=None)[0]
    cz = np.linalg.lstsq(A[keep], z[sel][keep], rcond=None)[0]
    return dict(ps=float(cp[0]), po=float(cp[1]), zs=float(cz[0]), zo=float(cz[1]), n=int(keep.sum()),
                amax=float(np.percentile(a, 99)))


def detect_hole(depth_m, K, pose, u, a_face, w_range, z_top, deep=0.006, margin=0.003, log=None):
    """Find the hole opening on the vertical face a = a_face (a = P.u, u horizontal unit into the box).
    w_range=(wmin,wmax) face extent along w (left normal). Returns dict(w, z, n, w_span, z_span, a_face) or None."""
    u = np.asarray(u, float)[:2]; u = u / np.linalg.norm(u)
    w = np.array([-u[1], u[0]])
    P = backproject(depth_m, K, pose)
    ok = np.isfinite(P[..., 0])
    A = P[..., :2] @ u; W = P[..., :2] @ w; Z = P[..., 2]
    # refine the face plane from points near it
    fm = ok & (np.abs(A - a_face) < 0.012) & (Z > 0.008) & (Z < z_top - 0.004) & (W > w_range[0] + margin) & (W < w_range[1] - margin)
    if fm.sum() > 50:
        a_face = float(np.median(A[fm]))
    c = np.asarray(pose["position_m"], float)
    cu = c[:2] @ u; cw = c[:2] @ w; cz = c[2]
    dA = A - cu
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (a_face - cu) / dA
    Wi = cw + t * (W - cw); Zi = cz + t * (Z - cz)
    hole = ok & (dA > 0.01) & (A > a_face + deep) & (t > 0) & (t < 1) & \
        (Wi > w_range[0] + margin) & (Wi < w_range[1] - margin) & (Zi > 0.004) & (Zi < z_top - 0.004)
    n = int(hole.sum())
    nface = int((ok & (np.abs(A - a_face) < 0.004)).sum())
    if log:
        log("detect_hole: a_face %.4f face_pts %d hole_px %d" % (a_face, nface, n))
    if n < 15:
        return None
    wi, zi, zh = Wi[hole], Zi[hole], Z[hole]
    # keep the largest cluster: points within 2.5 cm of the median
    mw, mz = np.median(wi), np.median(zi)
    k = (np.abs(wi - mw) < 0.025) & (np.abs(zi - mz) < 0.025)
    wi, zi, zh = wi[k], zi[k], zh[k]
    w_lo, w_hi = np.percentile(wi, [2, 98]); z_lo, z_hi = np.percentile(zi, [2, 98])
    # seen from above, the lower part of the opening is not "deep" -> use the hole floor hit points
    z_floor = float(np.percentile(zh, 2))
    z_bot = min(z_lo, z_floor)
    return dict(w=float((w_lo + w_hi) / 2), z=float((z_bot + z_hi) / 2), z_open=float((z_lo + z_hi) / 2),
                wm=float(np.median(wi)), zm=float(np.median(zi)),
                n=int(len(wi)), w_span=float(w_hi - w_lo), z_span=float(z_hi - z_bot), a_face=a_face, z_floor=z_floor,
                w_lo=float(w_lo), w_hi=float(w_hi), z_lo=float(z_bot), z_hi=float(z_hi), mask=hole)


def box_from_front(depth_m, K, pose, u, near_xy, radius=0.25, log=None):
    """Box top surface from the front depth. u = insertion direction (into the box).
    Returns dict(z_top, a_face, w_lo, w_hi, a_back, n) using top-surface points."""
    u = np.asarray(u, float)[:2]; u = u / np.linalg.norm(u)
    w = np.array([-u[1], u[0]])
    P = backproject(depth_m, K, pose)
    ok = np.isfinite(P[..., 0])
    d = np.hypot(P[..., 0] - near_xy[0], P[..., 1] - near_xy[1])
    m = ok & (P[..., 2] > 0.03) & (P[..., 2] < 0.25) & (d < radius)
    if m.sum() < 30:
        return None
    zt = float(np.percentile(P[..., 2][m], 97))
    top = m & (P[..., 2] > zt - 0.006) & (P[..., 2] < zt + 0.006)
    q = P[top]
    A = q[:, :2] @ u; W = q[:, :2] @ w
    a_face = float(np.percentile(A, 1))
    near = A < a_face + 0.02
    w_lo, w_hi = np.percentile(W[near], [1, 99])
    res = dict(z_top=zt, a_face=a_face, a_back=float(np.percentile(A, 99)), w_lo=float(w_lo), w_hi=float(w_hi),
               w_c=float(np.median(W)), n=int(len(q)))
    if log:
        log("box_from_front: %s" % {k: round(v, 4) for k, v in res.items()})
    return res
