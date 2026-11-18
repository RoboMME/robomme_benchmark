"""Live hole localisation on the box face from one wrist RGB-D frame.

Coordinates: u = horizontal insertion direction (into the box), v = (-u_y, u_x) lateral, z up.
Returns the opening centre in (lat, z) world terms plus extents, or None with a reason.
"""
import numpy as np


def _plane_fit(L, Z, S):
    A = np.c_[L, Z, np.ones(len(L))]
    return np.linalg.lstsq(A, S, rcond=None)[0]


def hole_measure(P, valid, cam_pos, u, origin_xy, peg_mask=None, max_ahead=0.12):
    """P: HxWx3 world points, valid: HxW bool, cam_pos: camera world position,
    u: unit 2-D insertion direction, origin_xy: reference point (peg tip) for s/lat coordinates."""
    u = np.asarray(u, float)[:2]; u = u / np.linalg.norm(u)
    v = np.array([-u[1], u[0]])
    rel = P[..., :2] - np.asarray(origin_xy)[:2]
    s = rel @ u; lat = rel @ v; z = P[..., 2]
    m = valid & (s > 0.002) & (s < max_ahead) & (np.abs(lat) < 0.07) & (z > 0.005)
    if peg_mask is not None:
        m &= ~peg_mask
    if m.sum() < 500:
        return None, "few points ahead (%d)" % m.sum()
    # face = dominant vertical plane: mode of s among points ahead
    sv = s[m]
    hist, edges = np.histogram(sv, bins=np.arange(0.002, max_ahead + 0.002, 0.002))
    k = int(np.argmax(hist)); s0 = edges[k] + 0.001
    fm = m & (np.abs(s - s0) < 0.004)
    c = None
    for _ in range(3):
        if fm.sum() < 200:
            return None, "few face points (%d)" % fm.sum()
        c = _plane_fit(lat[fm], z[fm], s[fm])
        res = s - (c[0] * lat + c[1] * z + c[2])
        fm = m & (np.abs(res) < 0.0025)
    res = s - (c[0] * lat + c[1] * z + c[2])
    face_z_top = float(np.percentile(z[fm], 99.5))
    face_z_bot = float(np.percentile(z[fm], 0.5))
    # behind-face points inside the box outline
    bm = valid & (res > 0.004) & (res < 0.12) & (z < face_z_top - 0.002) & (z > face_z_bot - 0.002) & (np.abs(lat) < 0.07)
    if peg_mask is not None:
        bm &= ~peg_mask
    face_s_mid = float(np.median(s[fm]))
    if bm.sum() < 60:
        return dict(open=False, face_s=face_s_mid, n_face=int(fm.sum())), "no opening visible (%d)" % bm.sum()
    # project rays from the camera onto the face plane
    cam = np.asarray(cam_pos, float)
    rc = cam[:2] - np.asarray(origin_xy)[:2]
    cs, cl, cz = rc @ u, rc @ v, cam[2]
    ds, dl, dz = s - cs, lat - cl, z - cz
    # plane: s = c0*lat + c1*z + c2 ; along ray: cs + t ds = c0 (cl + t dl) + c1 (cz + t dz) + c2
    den = ds - c[0] * dl - c[1] * dz
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (c[0] * cl + c[1] * cz + c[2] - cs) / den
    Lp = cl + t * dl; Zp = cz + t * dz
    Lb, Zb = Lp[bm], Zp[bm]
    # keep the main blob: around the median, within 3 cm
    lm, zm = np.median(Lb), np.median(Zb)
    keep = (np.abs(Lb - lm) < 0.03) & (np.abs(Zb - zm) < 0.03)
    Lb, Zb = Lb[keep], Zb[keep]
    zb_actual = z[bm][keep]
    if len(Lb) < 60:
        return dict(open=False, face_s=face_s_mid, n_face=int(fm.sum())), "opening blob too small"
    lat_lo, lat_hi = float(np.percentile(Lb, 1)), float(np.percentile(Lb, 99))
    z_top = float(np.percentile(Zb, 99))
    # bottom wall: lowest well-populated z level of the actual behind points
    z_floor = float(np.percentile(zb_actual, 3))
    out = dict(open=True, lat_lo=lat_lo, lat_hi=lat_hi, lat_c=(lat_lo + lat_hi) / 2, z_top=z_top, z_floor=z_floor,
               z_c=(z_top + z_floor) / 2, width=lat_hi - lat_lo, height=z_top - z_floor,
               face_s=float(c[0] * (lat_lo + lat_hi) / 2 + c[1] * (z_top + z_floor) / 2 + c[2]),
               face_coef=[float(x) for x in c], n_face=int(fm.sum()), n_open=int(len(Lb)),
               face_z_top=face_z_top, v=v.tolist(), origin=list(np.asarray(origin_xy)[:2]))
    return out, "ok"
