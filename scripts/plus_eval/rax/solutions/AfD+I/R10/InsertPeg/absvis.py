"""Absolute peg-hole alignment from one wrist depth frame.
Peg: connected surface at the near depth touching the image bottom (held peg top face).
Hole: pixels deeper than the box-face plane; both measured in world coordinates,
basis e1 (lateral, horizontal), e2 (up) perpendicular to the insertion direction n."""
import numpy as np
from cc import label
from rlib import quat_to_R

def basis(n):
    a = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    u1 = np.cross(n, a); u1 /= np.linalg.norm(u1); u2 = np.cross(n, u1)
    if u2[2] < 0: u2 = -u2
    return u1, u2

def backproject(u, v, z, cam):
    Rc = quat_to_R(cam["quaternion_wxyz"]); t = np.array(cam["position_m"])
    P = np.stack([(u - 128.0) * z / 128.0, -(v - 128.0) * z / 128.0, -z], -1)
    return P @ Rc.T + t

def measure(d_mm, cam, n, dbg=False):
    d = d_mm.astype(float) * 0.001
    H, W = d.shape
    e1, e2 = basis(n)
    band = d[200:250, 96:160]; band = band[band > 0]
    if band.size < 50: return None
    pz = float(np.median(band))
    if pz > 0.09: return None
    pm = (np.abs(d - pz) < 0.004) & (d > 0)
    lab, k = label(pm)
    ids = lab[230:250, 100:156].ravel(); ids = ids[ids > 0]
    if ids.size == 0: return None
    pid = np.bincount(ids).argmax(); peg = lab == pid
    vv, uu = np.nonzero(peg)
    tip_v = int(vv.min())
    sel = (vv < tip_v + 40) & (vv > tip_v + 2)
    if sel.sum() < 50: return None
    Pp = backproject(uu[sel].astype(float), vv[sel].astype(float), d[vv[sel], uu[sel]], cam)
    A = Pp @ e1; B = Pp @ e2
    a_lo, a_hi = np.percentile(A, 1), np.percentile(A, 99)
    peg_a = float((a_lo + a_hi) / 2); peg_w = float(a_hi - a_lo)
    peg_top = float(np.median(B))
    # face region: farther than the peg, above the peg tip, not the far table
    vg, ug = np.mgrid[0:H, 0:W]
    reg = (d > pz + 0.004) & (d < 0.112) & (~peg) & (vg < tip_v - 1)
    if reg.sum() < 300: return dict(peg_a=peg_a, peg_w=peg_w, peg_top=peg_top, tip_v=tip_v, hole=None)
    U = ug[reg].astype(float); V = vg[reg].astype(float); iz = 1.0 / d[reg]
    M = np.stack([U, V, np.ones_like(U)], 1); keep = np.ones(len(U), bool)
    for _ in range(5):
        coef, *_ = np.linalg.lstsq(M[keep], iz[keep], rcond=None)
        res = 1.0 / np.clip(M @ coef, 1e-3, None) - d[reg]
        keep = np.abs(res) < 0.003
    plane = 1.0 / np.clip(coef[0] * ug + coef[1] * vg + coef[2], 1e-3, None)
    hm = np.zeros_like(reg); hm[reg] = (d[reg] - plane[reg]) > 0.006
    hm &= d < 0.112
    lab2, k2 = label(hm)
    best = None
    for j in range(1, k2 + 1):
        m = lab2 == j; s = int(m.sum())
        if s < 40: continue
        v2, u2 = np.nonzero(m)
        sc = abs(u2.mean() - uu[sel].mean()) - 0.02 * s
        if best is None or sc < best[0]: best = (sc, j)
    if best is None: return dict(peg_a=peg_a, peg_w=peg_w, peg_top=peg_top, tip_v=tip_v, hole=None)
    m = lab2 == best[1]; v2, u2 = np.nonzero(m)
    Ph = backproject(u2.astype(float), v2.astype(float), plane[v2, u2], cam)
    Ah = Ph @ e1; Bh = Ph @ e2
    bb = [int(u2.min()), int(u2.max()), int(v2.min()), int(v2.max())]
    # truncation flags: touching image border or adjacent to the peg/tip
    dil = np.zeros_like(m); dil[1:-1, 1:-1] = m[1:-1, 1:-1]
    from cc import binary_dilation
    touch_peg = bool((binary_dilation(m, iterations=3) & peg).any())
    full = bb[0] > 2 and bb[1] < W - 3 and bb[2] > 2 and bb[3] < tip_v - 3 and not touch_peg
    a_lo, a_hi = np.percentile(Ah, 1), np.percentile(Ah, 99)
    b_lo, b_hi = np.percentile(Bh, 1), np.percentile(Bh, 99)
    hole = dict(a=float((a_lo + a_hi) / 2), b=float((b_lo + b_hi) / 2), w=float(a_hi - a_lo), h=float(b_hi - b_lo),
                n=int(m.sum()), bbox=bb, full=bool(full), touch_peg=touch_peg)
    return dict(peg_a=peg_a, peg_w=peg_w, peg_top=peg_top, tip_v=tip_v, hole=hole)

def required_shift(Mz):
    """EE shift (in e1, e2 metres) that puts the peg axis on the hole centre."""
    h = Mz["hole"]; r = Mz["peg_w"] / 2
    return h["a"] - Mz["peg_a"], h["b"] - (Mz["peg_top"] - r)
