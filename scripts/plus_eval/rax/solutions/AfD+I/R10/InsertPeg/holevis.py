"""Hole / peg features from the wrist depth image (uint16 mm)."""
import numpy as np
from cc import label

def feats(d_mm):
    d = d_mm.astype(float) * 0.001
    H, W = d.shape
    # peg: near band at bottom-centre of the image
    band = d[200:250, 96:160]
    vals = band[(band > 0) & (band < 0.09)]
    if len(vals) < 50:
        return None
    pz = float(np.median(vals))
    pm = (np.abs(d - pz) < 0.004) & (d > 0)
    # connected component touching the bottom centre
    lab, nl = label(pm)
    ids = lab[230:250, 110:150].ravel(); ids = ids[ids > 0]
    if len(ids) == 0:
        return None
    pid = np.bincount(ids).argmax()
    pegm = lab == pid
    vv, uu = np.nonzero(pegm)
    tip_v = int(vv.min())
    rows = (vv < tip_v + 25)
    peg_u = float(uu[rows].mean())
    # hole: deeper than the box-face plane (inverse depth is affine in u,v), above the peg tip
    vg, ug = np.mgrid[0:H, 0:W]
    reg = (d > pz + 0.004) & (d < 0.112) & (~pegm) & (vg < tip_v - 2)
    if reg.sum() < 200:
        return {"peg_u": peg_u, "tip_v": tip_v, "peg_z": pz, "hole": None}
    U = ug[reg].astype(float); Vv = vg[reg].astype(float); iz = 1.0 / d[reg]
    A = np.stack([U, Vv, np.ones_like(U)], 1)
    keep = np.ones(len(U), bool)
    for _ in range(4):
        coef, *_ = np.linalg.lstsq(A[keep], iz[keep], rcond=None)
        res = 1.0 / np.clip(A @ coef, 1e-3, None) - d[reg]   # plane depth - actual (negative => deeper)
        keep = np.abs(res) < 0.003
    plane = 1.0 / np.clip(coef[0] * ug + coef[1] * vg + coef[2], 1e-3, None)
    hm = np.zeros_like(reg); hm[reg] = (d[reg] - plane[reg]) > 0.006
    hm &= (d < 0.112)
    mf = plane
    lab2, n2 = label(hm)
    if n2 == 0:
        return {"peg_u": peg_u, "tip_v": tip_v, "peg_z": pz, "hole": None}
    sizes = np.bincount(lab2.ravel()); sizes[0] = 0
    # choose the component nearest the peg column with reasonable size
    best = None
    for k in range(1, n2 + 1):
        if sizes[k] < 30: continue
        v2, u2 = np.nonzero(lab2 == k)
        score = abs(u2.mean() - peg_u) + 0.5 * abs(v2.mean() - (tip_v - 30)) - 0.05 * sizes[k]
        if best is None or score < best[0]:
            best = (score, k, u2.mean(), v2.mean(), sizes[k], float(np.median(mf[lab2 == k])),
                    int(u2.min()), int(u2.max()), int(v2.min()), int(v2.max()))
    if best is None:
        return {"peg_u": peg_u, "tip_v": tip_v, "peg_z": pz, "hole": None}
    _, k, hu, hv, n, fz, u0, u1, v0, v1 = best
    hmask = lab2 == k
    return {"peg_u": peg_u, "tip_v": tip_v, "peg_z": pz,
            "hole": {"u": float(hu), "v": float(hv), "n": int(n), "face_z": fz, "bbox": [u0, u1, v0, v1]},
            "_mask": hmask, "_plane": plane}

def hole3d(F, frame, quat_to_R):
    """centroid (world) of the hole opening projected onto the fitted face plane"""
    m = F["_mask"]; pl = F["_plane"]
    v, u = np.nonzero(m); z = pl[v, u]
    K = np.array(frame["camera_intrinsics"]["wrist"]["matrix"])
    P = np.stack([(u - K[0, 2]) * z / K[0, 0], -(v - K[1, 2]) * z / K[1, 1], -z], 1)
    cp = frame["camera_poses"]["wrist"]
    Pw = P @ quat_to_R(cp["quaternion_wxyz"]).T + np.array(cp["position_m"])
    return Pw.mean(0), Pw
