"""Locate the box hole relative to the demo by registering 'deep' (hole) regions of the
wrist depth image, expressed in the plane perpendicular to the insertion axis."""
import numpy as np
from boxshift import backproject


def perp_basis(axis):
    a = axis / np.linalg.norm(axis)
    up = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(up, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    return a, e1, e2


def hole_map(d_mm, cam, K, p_ref, axis, res=0.0005, half=0.03, near_mm=70):
    """Raster in the perpendicular plane centred on p_ref: 1 deep(hole), 0 face, -1 unknown."""
    a, e1, e2 = perp_basis(axis)
    d = d_mm.astype(float)
    P = backproject(d, cam, K)
    valid = (d > 0) & (d < 118)
    # exclude the hand/peg: very near the camera
    valid &= d > near_mm * 0 + 1
    rel = P - p_ref
    ax_c = rel @ a; u = rel @ e1; v = rel @ e2
    n = int(2 * half / res)
    grid = -np.ones((n, n), int)
    return ax_c, u, v, valid, grid, n


def classify(d_mm, cam, K, p_ref, axis, peg_mask, res=0.0005, half=0.03):
    a, e1, e2 = perp_basis(axis)
    d = d_mm.astype(float)
    P = backproject(d, cam, K)
    rel = P - p_ref
    ax_c = rel @ a; u = rel @ e1; v = rel @ e2
    ok = (d > 0) & (d < 118) & ~peg_mask
    if ok.sum() < 100:
        return None
    # face plane = dominant axial coordinate among valid points (histogram mode)
    h, edges = np.histogram(ax_c[ok], bins=np.arange(ax_c[ok].min(), ax_c[ok].max() + 0.001, 0.001))
    face = edges[np.argmax(h)] + 0.0005
    facem = ok & (np.abs(ax_c - face) < 0.003)
    deep = ok & (ax_c > face + 0.006) & (ax_c < face + 0.04)
    n = int(2 * half / res)
    g = -np.ones((n, n), int)
    for m, val in ((facem, 0), (deep, 1)):
        iu = np.clip(((u[m] + half) / res).astype(int), 0, n - 1)
        iv = np.clip(((v[m] + half) / res).astype(int), 0, n - 1)
        g[iv, iu] = np.maximum(g[iv, iu], val)
    return g, face, int(facem.sum()), int(deep.sum())


def register(g_live, g_demo, res=0.0005, max_shift=0.016):
    """Shift (du,dv) metres such that live(x) ~ demo(x - shift): hole moved by +shift."""
    n = g_live.shape[0]
    ms = int(max_shift / res)
    best = None
    L = g_live; D = g_demo
    for sv in range(-ms, ms + 1):
        for su in range(-ms, ms + 1):
            # demo shifted by (su,sv)
            Ds = -np.ones_like(D)
            ys = slice(max(sv, 0), n + min(sv, 0)); yd = slice(max(-sv, 0), n + min(-sv, 0))
            xs = slice(max(su, 0), n + min(su, 0)); xd = slice(max(-su, 0), n + min(-su, 0))
            Ds[ys, xs] = D[yd, xd]
            both = (L >= 0) & (Ds >= 0)
            nb = both.sum()
            if nb < 200:
                continue
            # only count cells where either says deep
            dl = (L == 1) & both; dd = (Ds == 1) & both
            inter = (dl & dd).sum(); uni = (dl | dd).sum()
            if uni == 0:
                continue
            iou = inter / uni
            if best is None or iou > best[0] or (iou == best[0] and abs(su) + abs(sv) < abs(best[1]) + abs(best[2])):
                best = (iou, su, sv)
    if best is None:
        return None
    return best[1] * res, best[2] * res, best[0]


def hole_mask_img(d_mm, cam, K, p_ref, axis, peg):
    """Image-space masks: ok (known, non-peg) and hole (deeper than face by >6 mm)."""
    d = d_mm.astype(float)
    P = backproject(d, cam, K)
    a = (P - p_ref) @ (axis / np.linalg.norm(axis))
    ok = (d > 0) & (d < 118) & ~peg
    if ok.sum() < 200:
        return None
    h, e = np.histogram(a[ok], bins=np.arange(a[ok].min(), a[ok].max() + 0.001, 0.001))
    face = e[np.argmax(h)] + 0.0005
    hole = ok & (a > face + 0.006)
    facem = ok & (np.abs(a - face) < 0.003)
    return ok, hole, facem, face, d


def register_img(okL, holeL, okD, holeD, max_px=40):
    """Pixel shift (du,dv) with holeL(x) ~ holeD(x - (du,dv)); returns du,dv,score."""
    H, W = holeL.shape
    best = None
    for dv in range(-max_px, max_px + 1, 1):
        for du in range(-max_px, max_px + 1, 1):
            ys = slice(max(dv, 0), H + min(dv, 0)); yd = slice(max(-dv, 0), H + min(-dv, 0))
            xs = slice(max(du, 0), W + min(du, 0)); xd = slice(max(-du, 0), W + min(-du, 0))
            both = okL[ys, xs] & okD[yd, xd]
            hl = holeL[ys, xs] & both; hd = holeD[yd, xd] & both
            uni = (hl | hd).sum()
            if uni < 30:
                continue
            iou = (hl & hd).sum() / uni
            sc = iou - 0.0005 * (abs(du) + abs(dv))
            if best is None or sc > best[2]:
                best = (du, dv, sc, iou, int(uni))
    return best


def px_shift_to_world(du, dv, z, K, cam, axis):
    from boxshift import quat_to_mat
    R = quat_to_mat(cam["quaternion_wxyz"])
    dc = np.array([du * z / K[0][0], -dv * z / K[1][1], 0.0])
    w = R @ dc
    a = axis / np.linalg.norm(axis)
    return w - (w @ a) * a


def register_3d(dL, camL, dD, camD, K, p_ref, axis, peg, max_shift=0.014, step=0.0005):
    """Find perpendicular world shift of the hole between demo (D) and live (L).
    Returns (shift_world[3], score, A, B, nL, nD) or None."""
    from boxshift import project
    mL = hole_mask_img(dL, camL, K, p_ref, axis, peg)
    mD = hole_mask_img(dD, camD, K, p_ref, axis, peg)
    if mL is None or mD is None:
        return None
    okL, holeL = mL[0], mL[1]
    okD, holeD = mD[0], mD[1]
    if holeL.sum() < 40 or holeD.sum() < 40:
        return None
    PL = backproject(dL.astype(float), camL, K)[holeL]
    PD = backproject(dD.astype(float), camD, K)[holeD]
    _, e1, e2 = perp_basis(axis)
    H, W = holeL.shape

    def frac(P, sh, cam, ok, hole):
        u, v, z = project(P + sh, cam, K)
        ui = np.round(u).astype(int); vi = np.round(v).astype(int)
        inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        ui, vi = ui[inb], vi[inb]
        known = ok[vi, ui] | hole[vi, ui]
        if known.sum() < 20:
            return 0.0
        return float(hole[vi, ui][known].mean())

    best = None
    rng = np.arange(-max_shift, max_shift + 1e-9, step)
    for a in rng:
        for b in rng:
            sh = a * e1 + b * e2
            A = frac(PD, sh, camL, okL, holeL)
            if A < 0.3:
                continue
            B = frac(PL, -sh, camD, okD, holeD)
            sc = A * B - 0.5 * np.hypot(a, b)  # tiny preference for small shifts
            if best is None or sc > best[1]:
                best = (sh, sc, A, B)
    if best is None:
        return None
    return best[0], best[1], best[2], best[3], int(holeL.sum()), int(holeD.sum())


def face_maps(d_mm, cam, K, p_ref, axis, peg, res=0.0005, half=0.035):
    """Face-plane occupancy around p_ref (world-fixed, perpendicular to axis):
    returns (face_grid, deep_grid) boolean arrays indexed [v, u] and the face axial coord."""
    a_, e1, e2 = perp_basis(axis)
    d = d_mm.astype(float)
    P = backproject(d, cam, K)
    rel = P - p_ref
    a = rel @ a_; u = rel @ e1; v = rel @ e2
    ok = (d > 0) & (d < 118) & ~peg
    if ok.sum() < 200:
        return None
    h, e = np.histogram(a[ok], bins=np.arange(a[ok].min(), a[ok].max() + 0.001, 0.001))
    face = e[np.argmax(h)] + 0.0005
    facem = ok & (np.abs(a - face) < 0.002)
    deep = ok & (a > face + 0.004) & (a < face + 0.05)
    n = int(round(2 * half / res))
    F = np.zeros((n, n), bool); D = np.zeros((n, n), bool)
    for m, G in ((facem, F), (deep, D)):
        iu = np.floor((u[m] + half) / res).astype(int); iv = np.floor((v[m] + half) / res).astype(int)
        k = (iu >= 0) & (iu < n) & (iv >= 0) & (iv < n)
        G[iv[k], iu[k]] = True
    D &= ~F
    return F, D, face


def _integral(G):
    I = np.zeros((G.shape[0] + 1, G.shape[1] + 1))
    I[1:, 1:] = np.cumsum(np.cumsum(G, 0), 1)
    return I


def square_fit(F, D, S_cells, res=0.0005, half=0.035, max_face=2):
    """Centres (u,v metres rel. p_ref) of SxS squares containing <=max_face face cells and
    the maximal number of deep cells; returns centroid of the best set, count, deep_in."""
    IF = _integral(F); ID = _integral(D)
    n = F.shape[0]
    S = S_cells
    if S >= n:
        return None
    # top-left corners (r, c)
    r = np.arange(0, n - S + 1); c = np.arange(0, n - S + 1)
    R, C = np.meshgrid(r, c, indexing="ij")
    def box(I):
        return I[R + S, C + S] - I[R, C + S] - I[R + S, C] + I[R, C]
    fin = box(IF); din = box(ID)
    feas = fin <= max_face
    if not feas.any():
        return None
    best = din[feas].max()
    if best < 5:
        return None
    sel = feas & (din >= best - 1)
    rr = R[sel]; cc = C[sel]
    cu = (cc.mean() + S / 2) * res - half
    cv = (rr.mean() + S / 2) * res - half
    spread = float(max(np.ptp(cc), np.ptp(rr)) * res)
    return cu, cv, int(sel.sum()), float(best), spread


def best_square(F, D, res=0.0005, half=0.035, S_range=(20.0, 26.0), wf=3.0):
    """Maximise deep_in - wf*face_in over square size and centre; returns (cu, cv, S_mm, score, n_ties)."""
    IF = _integral(F); ID = _integral(D)
    n = F.shape[0]
    best = None
    for S_mm in np.arange(S_range[0], S_range[1] + 1e-9, 0.5):
        S = int(round(S_mm / 1000 / res))
        r = np.arange(0, n - S + 1)
        R, C = np.meshgrid(r, r, indexing="ij")
        def box(I):
            return I[R + S, C + S] - I[R, C + S] - I[R + S, C] + I[R, C]
        sc = box(ID) - wf * box(IF)
        m = sc.max()
        if best is None or m > best[0] + 1e-9:
            sel = sc >= m - 0.5
            best = (m, S_mm, (C[sel].mean() + S / 2) * res - half, (R[sel].mean() + S / 2) * res - half, int(sel.sum()))
    m, S_mm, cu, cv, nt = best
    return cu, cv, S_mm, float(m), nt


def hole_shift_square(dL, camL, dD, camD, K, p_ref, axis, peg, res=0.0005):
    """Shift of hole centre (world, perpendicular) live minus demo using square fits."""
    mL = face_maps(dL, camL, K, p_ref, axis, peg, res)
    mD = face_maps(dD, camD, K, p_ref, axis, peg, res)
    if mL is None or mD is None:
        return None
    _, e1, e2 = perp_basis(axis)
    fl = best_square(mL[0], mL[1], res)
    fd = best_square(mD[0], mD[1], res)
    du = fl[0] - fd[0]; dv = fl[1] - fd[1]
    return du * e1 + dv * e2, {"live": [round(float(x), 5) for x in fl], "demo": [round(float(x), 5) for x in fd],
                               "face_live": float(mL[2]), "face_demo": float(mD[2])}


def clean_deep(F, D):
    """Drop deep cells at/above the top of the face in each column (box top / background)."""
    n = F.shape[0]
    D = D.copy()
    rows = np.arange(n)[:, None]
    has = F.any(0)
    topf = np.where(has, (F * rows).max(0), -1)  # highest face row per column
    D &= rows < (topf[None, :] - 2)
    # also drop deep cells far below the lowest face row band (floor seen past box)
    return D


def fit_square_constrained(F, D, S_mm, res=0.0005, half=0.035, face_tol=0.01):
    """Squares of side S with (almost) no face cells inside, covering as many deep cells as
    possible. Returns (cu, cv, deep_in, n_feasible_best, spread_u, spread_v) metres or None."""
    D = clean_deep(F, D)
    IF = _integral(F); ID = _integral(D)
    n = F.shape[0]
    S = int(round(S_mm / 1000 / res))
    r = np.arange(0, n - S + 1)
    R, C = np.meshgrid(r, r, indexing="ij")
    def box(I):
        return I[R + S, C + S] - I[R, C + S] - I[R + S, C] + I[R, C]
    fin = box(IF); din = box(ID)
    feas = fin <= face_tol * S * S
    if not feas.any():
        return None
    best = din[feas].max()
    sel = feas & (din >= best - max(1, 0.03 * best))
    cu = (C[sel].mean() + S / 2) * res - half
    cv = (R[sel].mean() + S / 2) * res - half
    return cu, cv, float(best), int(sel.sum()), float(np.ptp(C[sel]) * res), float(np.ptp(R[sel]) * res)


def hole_shift_edges(dL, camL, dD, camD, K, p_ref, axis, peg, S_mm=21.0, res=0.0005):
    mL = face_maps(dL, camL, K, p_ref, axis, peg, res)
    mD = face_maps(dD, camD, K, p_ref, axis, peg, res)
    if mL is None or mD is None:
        return None
    fl = fit_square_constrained(mL[0], mL[1], S_mm, res)
    fd = fit_square_constrained(mD[0], mD[1], S_mm, res)
    if fl is None or fd is None:
        return None
    _, e1, e2 = perp_basis(axis)
    du = fl[0] - fd[0]; dv = fl[1] - fd[1]
    return du * e1 + dv * e2, (du, dv), fl, fd
