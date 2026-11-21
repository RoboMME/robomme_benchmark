"""World-frame multi-view tracker: front camera (static, calibrated) + wrist camera
(calibrated at frame 0 / execution; parked pose estimated from table plane + point registration)."""
import json, os
import numpy as np
from geom import depth_to_world, quat_to_R, label
import vtrack

def cam_points(depth, K):
    H, W = depth.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    return np.stack([(u - cx) * depth / fx, -(v - cy) * depth / fy, -depth], -1)

def fit_plane_ransac(Q, iters=200, thr=0.004, rng=None):
    rng = rng or np.random.RandomState(0)
    best = None
    n = len(Q)
    for _ in range(iters):
        s = Q[rng.choice(n, 3, replace=False)]
        nv = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(nv)
        if nn < 1e-9:
            continue
        nv /= nn
        d = -nv @ s[0]
        inl = np.abs(Q @ nv + d) < thr
        c = inl.sum()
        if best is None or c > best[0]:
            best = (c, nv, d, inl)
    _, nv, d, inl = best
    # refine with least squares
    P = Q[inl]; c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c)
    nv = Vt[-1]; d = -nv @ c
    if d < 0:  # camera (origin) on positive side
        nv, d = -nv, -d
    return nv, d, inl.mean()

def rot_to_z(nv):
    z = np.array([0, 0, 1.0])
    v = np.cross(nv, z); s = np.linalg.norm(v); c = nv @ z
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)

def obj_and_button(rgb, Pw):
    z = Pw[..., 2]
    sat = vtrack.sat_of(rgb); val = rgb.max(-1)
    obj = np.isfinite(z) & (z > 0.006) & (z < 0.08) & (sat > 35) & (Pw[..., 0] > -0.45)
    btn = np.isfinite(z) & (z > 0.004) & (z < 0.06) & (sat < 30) & (val > 120) & (Pw[..., 0] > -0.45)
    return obj, btn

def nn_dist(A, B):
    # mean truncated nearest distance from A to B (A,B Nx2)
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(1)
    return np.minimum(d, 0.03).mean(), d

def sub(X, n, seed=0):
    if len(X) <= n:
        return X
    return X[np.random.RandomState(seed).choice(len(X), n, replace=False)]

class ParkedWrist:
    """Maps parked-wrist depth images to world points."""
    def __init__(self, R0, d, yaw, t):
        self.R0 = R0; self.d = d; self.yaw = yaw; self.t = np.asarray(t)
    def to_world(self, Qc):
        q = Qc.reshape(-1, 3) @ self.R0.T
        h = q[:, 2] + self.d  # height above plane (R0 maps normal to +z; plane at -d)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        xy = q[:, :2] @ np.array([[c, s], [-s, c]]) + self.t
        return np.concatenate([xy, h[:, None]], 1).reshape(Qc.shape)

def estimate_parked(rgb, depth, K, ref_obj_xy, ref_btn_xy, log=print):
    Q = cam_points(depth, K)
    valid = depth > 0
    Qv = Q[valid]
    nv, d, frac = fit_plane_ransac(sub(Qv, 4000))
    R0 = rot_to_z(nv)
    # R0 @ nv = z ; height = nv.q + d = (R0 q)_z + d
    q = Q.reshape(-1, 3) @ R0.T
    h = (q[:, 2] + d).reshape(depth.shape)
    plane_xy = q[:, :2].reshape(depth.shape + (2,))
    sat = vtrack.sat_of(rgb); val = rgb.max(-1)
    om = valid & (h > 0.006) & (h < 0.08) & (sat > 35)
    bm = valid & (h > 0.004) & (h < 0.06) & (sat < 30) & (val > 120)
    A_o = plane_xy[om]; A_b = plane_xy[bm]
    log(f"parked plane inlier frac {frac:.2f} cam height {d:.3f} obj px {om.sum()} btn px {bm.sum()}")
    if len(A_o) < 30 or len(ref_obj_xy) < 30:
        return None, None
    Ao = sub(A_o, 400, 1); Bo = sub(ref_obj_xy, 1500, 2)
    best = None
    seeds = []
    if len(A_b) >= 20 and len(ref_btn_xy) >= 20:
        seeds.append(("button", A_b.mean(0), ref_btn_xy.mean(0)))
    seeds.append(("centroid", A_o.mean(0), ref_obj_xy.mean(0)))
    for name, ca, cb in seeds:
        for yaw in np.deg2rad(np.arange(0, 360, 4)):
            c, s = np.cos(yaw), np.sin(yaw)
            Rm = np.array([[c, s], [-s, c]])
            t = cb - ca @ Rm
            sc, _ = nn_dist(Ao @ Rm + t, Bo)
            if best is None or sc < best[0]:
                best = (sc, yaw, t, name)
    sc, yaw, t, name = best
    # ICP refine (2D rigid), using objects + button
    A_all = np.concatenate([Ao, sub(A_b, 150, 3)]) if len(A_b) else Ao
    B_all = np.concatenate([Bo, sub(ref_btn_xy, 400, 4)]) if len(ref_btn_xy) else Bo
    for _ in range(15):
        c, s = np.cos(yaw), np.sin(yaw); Rm = np.array([[c, s], [-s, c]])
        X = A_all @ Rm + t
        dd = ((X[:, None] - B_all[None]) ** 2).sum(-1)
        j = dd.argmin(1); ok = np.sqrt(dd[np.arange(len(X)), j]) < 0.02
        if ok.sum() < 20:
            break
        Pa = A_all[ok]; Pb = B_all[j[ok]]
        ma, mb = Pa.mean(0), Pb.mean(0)
        H = (Pa - ma).T @ (Pb - mb)
        U, S, Vt = np.linalg.svd(H)
        Rr = (U @ Vt)  # maps a-> b as a @ Rr
        if np.linalg.det(Rr) < 0:
            break
        yaw = np.arctan2(Rr[0, 1], Rr[0, 0]); t = mb - ma @ Rr
    c, s = np.cos(yaw), np.sin(yaw); Rm = np.array([[c, s], [-s, c]])
    fin, dd = nn_dist(Ao @ Rm + t, Bo)
    log(f"parked registration seed={name} yaw {np.rad2deg(yaw):.1f} t {np.round(t,3)} coarse {sc:.4f} final mean nn {fin:.4f} frac<1cm {np.mean(dd<0.01):.2f}")
    if fin > 0.008:
        log("parked registration poor; wrist parked frames unused")
        return None, fin
    return ParkedWrist(R0, d, yaw, t), fin

def run(manifest_path, workspace="/workspace", log=print, max_pts=2500):
    m = json.load(open(manifest_path))
    frames = sorted(m["frames"], key=lambda f: f["frame_index"])
    ex = [f for f in frames if f["phase"] == "execution"]
    vid = [f for f in frames if f["phase"] == "video_demo"]
    last = ex[-1] if ex else frames[-1]
    Kf = last["camera_intrinsics"]["front"]["matrix"]; pf = last["camera_poses"]["front"]
    Kw = last["camera_intrinsics"]["wrist"]["matrix"]; pw = last["camera_poses"]["wrist"]
    seq = vid + [last]
    T = len(seq)
    # front-only analysis (pick cues) from the legacy tracker
    fpos, fcnts, finfo, _, _ = vtrack.run(manifest_path, workspace=workspace, log=log)
    # load wrist depths to find parked interval
    wd = [None] * T
    def wdepth(t):
        if wd[t] is None:
            wd[t] = vtrack.load_frame(workspace, seq[t], "wrist")
        return wd[t]
    rgbL, dL = wdepth(T - 2) if T >= 2 else wdepth(T - 1)
    t_park = T - 2
    for t in range(T - 3, 0, -1):
        if np.median(np.abs(wdepth(t)[1] - dL)) > 0.002:
            break
        t_park = t
    # also identical-to-home frames at start
    rgbE, dE = wdepth(T - 1)
    home = [t for t in range(0, min(T - 1, 40)) if np.median(np.abs(wdepth(t)[1] - dE)) < 0.001]
    log(f"wrist parked from t={t_park} to {T-2}; home-pose frames {home[:3]}..{len(home)}")
    # reference world points at execution (front + wrist home)
    def world_pts(t, use_wrist):
        rgb, dep = vtrack.load_frame(workspace, seq[t], "front")
        Pf = depth_to_world(dep, Kf, pf)
        o, b = obj_and_button(rgb, Pf)
        obj = [Pf[o][:, :2]]; btn = [Pf[b][:, :2]]; src = [np.zeros(o.sum(), int)]
        if use_wrist == "home":
            rgbw, depw = wdepth(t)
            Pw = depth_to_world(depw, Kw, pw)
            o2, b2 = obj_and_button(rgbw, Pw)
            obj.append(Pw[o2][:, :2]); btn.append(Pw[b2][:, :2]); src.append(np.ones(o2.sum(), int))
        elif use_wrist == "parked" and parked is not None:
            rgbw, depw = wdepth(t)
            Pw = parked.to_world(cam_points(depw, Kw))
            o2, b2 = obj_and_button(rgbw, Pw)
            o2 &= depw > 0
            obj.append(Pw[o2][:, :2]); src.append(np.ones(o2.sum(), int))
        return np.concatenate(obj), np.concatenate(btn), np.concatenate(src)
    parked = None
    refO, refB, _ = world_pts(T - 1, "home")
    try:
        parked, fit = estimate_parked(rgbL, dL, Kw, refO, refB, log=log)
    except Exception as e:
        log("parked estimate failed", repr(e))
    # ---- init tracks at t=0 from front + wrist(home) ----
    O0, _, src0 = world_pts(0, "home" if 0 in home else None)
    if 0 in home and (src0 == 1).sum() > 100:
        # wrist home view is a full top-down view: use it alone (front adds side-surface fragments)
        O0 = O0[src0 == 1]
    cell = 0.003
    g = np.floor(O0 / cell).astype(int)
    gmin = g.min(0); gs = g - gmin
    occ = np.zeros(gs.max(0) + 1, bool); occ[gs[:, 0], gs[:, 1]] = True
    dil = occ.copy()
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            if abs(dx) + abs(dy) > 3:
                continue
            sh = np.zeros_like(occ)
            xs0, xs1 = max(dx, 0), occ.shape[0] + min(dx, 0)
            ys0, ys1 = max(dy, 0), occ.shape[1] + min(dy, 0)
            sh[xs0:xs1, ys0:ys1] = occ[xs0 - dx:xs1 - dx, ys0 - dy:ys1 - dy]
            dil |= sh
    lab, n = label(dil)
    lab = np.where(occ, lab, 0)
    cl = []
    for i in range(1, n + 1):
        cells = np.argwhere(lab == i)
        if len(cells) < 4:
            continue
        sel = lab[gs[:, 0], gs[:, 1]] == i
        cl.append(dict(area=len(cells), pts=O0[sel]))
    areas = np.array([c["area"] for c in cl], float)
    unit = float(np.median(areas)) if len(areas) else 1.0
    centers = []
    for c in cl:
        k = int(round(c["area"] / unit))
        if k > 3:
            log(f"drop init cluster area {c['area']}"); continue
        k = max(1, k)
        P = c["pts"]
        if k == 1:
            centers.append(P.mean(0)); continue
        C = [P[np.argmax(((P - P.mean(0)) ** 2).sum(1))]]
        while len(C) < k:
            dd = np.min([((P - cc) ** 2).sum(1) for cc in C], axis=0); C.append(P[np.argmax(dd)])
        C = np.array(C)
        for _ in range(15):
            a = ((P[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            C = np.array([P[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
        centers.extend(list(C))
    X0 = np.array(centers)
    Kn = len(X0)
    log(f"world init: {Kn} objects (areas {areas.astype(int).tolist()}, unit {unit:.0f}) at {np.round(X0,3).tolist()}")
    # ---- pick identification ----
    target = None; t_lift = None; t_lend = None
    # lift cue from legacy (world xy of lifted pixels)
    lc = finfo.get("lift")
    if lc is not None:
        dd = np.linalg.norm(X0 - np.array(lc["xy"]), axis=1)
        log(f"lift cue xy {np.round(lc['xy'],3)} nearest world obj {int(dd.argmin())} dist {dd.min():.3f}")
        if dd.min() < 0.045:
            target = int(dd.argmin()); t_lift = lc["t"]; t_lend = lc["t_end"]
    re_ = finfo.get("raw_empty")
    if re_ is not None:
        dd = np.linalg.norm(X0 - np.array(re_[0]), axis=1)
        et = int(dd.argmin())
        if dd.min() < 0.045 and et != target:
            log(f"unambiguous empty cue -> world obj {et} (lift cue said {target}); using empty cue")
            target = et; t_lift = re_[1]; t_lend = re_[2]
        elif et == target:
            log("empty cue agrees with lift cue")
    if target is None and finfo.get("t_pick") is not None:
        exy = finfo.get("empty_xy")
        if exy is not None:
            dd = np.linalg.norm(X0 - np.array(exy), axis=1)
            target = int(dd.argmin()); t_lift = finfo["t_pick"]; t_lend = finfo["t_back"]
            log(f"empty cue -> world obj {target} dist {dd.min():.3f}")
    if target is None:
        log("no pick cue; world tracker cannot identify target")
        return None, dict(front=finfo, fpos=fpos)
    # ---- world k-means tracking ----
    pos = [X0.copy()]; cnt_hist = [np.ones(Kn, int) * 99]
    vel = np.zeros_like(X0)
    radius = 0.022
    for t in range(1, T):
        mode = "home" if (t == T - 1 or t in home) else ("parked" if (parked is not None and t_park <= t <= T - 2) else None)
        O, _, _ = world_pts(t, mode)
        O = sub(O, max_pts, t)
        prev = pos[-1]
        frozen = np.zeros(Kn, bool)
        if t_lift is not None and t_lift <= t < (t_lend or t_lift) + 3:
            frozen[target] = True
        C = prev + vel * 0.6
        C[frozen] = prev[frozen]
        cnt = np.zeros(Kn, int)
        if len(O):
            for _ in range(6):
                dd = ((O[:, None] - C[None]) ** 2).sum(-1)
                dd[:, frozen] = 1e9
                a = dd.argmin(1); ok = dd[np.arange(len(O)), a] < radius ** 2
                newC = C.copy()
                for k in range(Kn):
                    s_ = ok & (a == k); cnt[k] = s_.sum()
                    if cnt[k] >= 4:
                        newC[k] = O[s_].mean(0)
                C = newC
        vel = np.where(cnt[:, None] >= 4, C - prev, vel * 0.8); vel[frozen] = 0
        pos.append(C); cnt_hist.append(cnt)
    pos = np.array(pos); cnt_hist = np.array(cnt_hist)
    d_t = np.linalg.norm(pos - pos[:, target:target + 1], axis=2); d_t[:, target] = 9
    log(f"world track: target {target} final xy {np.round(pos[-1, target], 3)} final count {cnt_hist[-1, target]} "
        f"min sep to others {d_t[t_park:].min():.3f} zero-count frames {(cnt_hist[:, target] < 4).sum()}")
    info = dict(target=target, final_xy=pos[-1, target].tolist(), t_lift=t_lift, t_lend=t_lend, t_park=t_park,
                parked_ok=parked is not None, min_sep=float(d_t[t_park:].min()), final_count=int(cnt_hist[-1, target]))
    return pos, info
