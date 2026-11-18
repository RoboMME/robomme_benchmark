"""Hole localisation on the box face from one or more wrist depth clouds (v7)."""
import numpy as np


def cloud_saw(Q, P0, a, u, ref):
    s = (Q - P0) @ a
    uu = (Q - ref) @ u
    return s, uu, Q[:, 2]


def face_s(s, uu, z, lat=0.045):
    sel = (z > 0.015) & (z < 0.075) & (np.abs(uu) < lat)
    if sel.sum() < 200:
        return None
    hb = np.round(s[sel] * 1000).astype(int)
    vals, cnt = np.unique(hb, return_counts=True)
    # the face is the farthest dense plane (peg/gripper points lie before it)
    dense = vals[cnt >= max(150, 0.05 * cnt.max())]
    if len(dense) == 0:
        return None
    return float(dense.max()) / 1000.0


def detect_hole(clouds, P0, a, u, ref, peg_h=0.0185, log=print):
    """clouds: list of (Nx3) world points.  Returns dict(u_c, z_c, ...) or None."""
    Q = np.concatenate(clouds, 0)
    s, uu, z = cloud_saw(Q, P0, a, u, ref)
    sf = face_s(s, uu, z)
    if sf is None:
        log("hole: no face")
        return None
    near = (s > sf + 0.004) & (s < sf + 0.06) & (np.abs(uu) < 0.045)
    if near.sum() < 50:
        log("hole: no points beyond face")
        return None
    z_top = float(np.percentile(z[near], 95))
    rec = near & (z < z_top - 0.008) & (z > 0.008)
    n = int(rec.sum())
    if n < 30:
        log(f"hole: no recess (face s={sf:.4f} top={z_top:.4f})")
        return None
    ur, zr = uu[rec], z[rec]
    u_lo, u_hi = float(np.percentile(ur, 2)), float(np.percentile(ur, 98))
    z_floor = float(np.percentile(zr, 5))
    # top edge of the opening: lowest face point above the floor inside the u-range
    fsel = (np.abs(s - sf) < 0.002) & (uu > u_lo + 0.003) & (uu < u_hi - 0.003) & (z > z_floor + 0.004)
    z_edge = float(np.percentile(z[fsel], 3)) if fsel.sum() > 10 else None
    out = dict(face_s=sf, z_top=z_top, n=n, u_lo=u_lo, u_hi=u_hi, z_floor=z_floor, z_edge=z_edge)
    log("hole raw", {k: (round(v * 1000, 1) if isinstance(v, float) else v) for k, v in out.items()})
    return out


def peg_section(Q_list_with_ee, P0, a, u, ref, sf):
    """Peg top points (before the face) -> (u_center, z_top) relative to the EE."""
    res = []
    for Q, ee in Q_list_with_ee:
        s, uu, z = cloud_saw(Q, P0, a, u, ref)
        se = float((ee - P0) @ a)
        sel = (s < sf - 0.003) & (s > se + 0.02) & (z > ee[2]) & (z < ee[2] + 0.03) & (np.abs(uu - (ee - ref) @ u) < 0.03)
        if sel.sum() < 100:
            continue
        ztop = float(np.percentile(z[sel], 90))
        top = sel & (z > ztop - 0.002)
        uc = float(np.median(uu[top]))
        res.append((uc - float((ee - ref) @ u), ztop - ee[2]))
    if not res:
        return None
    r = np.array(res)
    return float(np.median(r[:, 0])), float(np.median(r[:, 1]))
