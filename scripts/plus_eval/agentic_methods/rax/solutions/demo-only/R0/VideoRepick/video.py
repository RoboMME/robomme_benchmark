"""Scene perception: camera calibration, table plane, object colour, video target
identification (vacancy test) and multi-object tracking in world XY."""
import time
import numpy as np
from scipy import ndimage
from io_utils import log, reply_json, find, is_num_list, load_rgb, load_depth
from geom import fit_M, backproject, project, moved_M, table_depth_along, M_from_intrinsics_pose
from percep import learn_color, rgb_to_hsv, components
from tracker import detect_objects, Tracker

CAMS = ('front', 'wrist')

# ---------------------------------------------------------------- calibration
def _extract_points(res_json, n):
    """Return list of (valid, xyz) from a pixel_to_position reply."""
    cands = None
    for k in ('points', 'results', 'positions', 'selections', 'world_points'):
        v = find(res_json, k)
        if isinstance(v, list) and len(v) == n:
            cands = v; break
    if cands is None:
        # any list of length n of dicts
        stack = [res_json]
        while stack and cands is None:
            x = stack.pop()
            if isinstance(x, dict):
                for v in x.values():
                    if isinstance(v, list) and len(v) == n and all(isinstance(e, (dict, list)) for e in v):
                        cands = v; break
                    if isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(x, list):
                stack.extend(e for e in x if isinstance(e, (dict, list)))
    out = []
    for e in cands or []:
        if is_num_list(e, 3):
            out.append((True, np.array(e, float))); continue
        if not isinstance(e, dict):
            out.append((False, None)); continue
        valid = e.get('valid', True)
        xyz = None
        for k in ('position_m', 'position', 'world_position', 'world', 'point', 'xyz', 'world_point', 'point_m'):
            if is_num_list(e.get(k), 3):
                xyz = np.array(e[k], float); break
        if xyz is None:
            for k, v in e.items():
                if is_num_list(v, 3) and k not in ('rgb', 'color'):
                    xyz = np.array(v, float); break
        out.append((bool(valid) and xyz is not None, xyz))
    return out

def _offplane_pixels(depth, n=32):
    """Pixels whose depth departs from the dominant plane (disparity linear in u,v)."""
    H, W = depth.shape
    vv, uu = np.mgrid[6:H - 6:2, 6:W - 6:2]
    d = depth[vv, uu]; ok = d > 0
    if ok.sum() < 50:
        return []
    u = uu[ok].astype(float); v = vv[ok].astype(float); dd = d[ok]
    A = np.stack([u, v, np.ones_like(u)], 1); y = 1.0 / dd
    inl = np.ones(len(y), bool)
    for _ in range(4):
        c, *_ = np.linalg.lstsq(A[inl], y[inl], rcond=None)
        dev = np.abs(1.0 / np.maximum(A @ c, 1e-6) - dd)
        inl = dev < max(0.004, np.percentile(dev, 50))
    cand = np.where(dev > 0.015)[0]
    if len(cand) == 0:
        return []
    # farthest-point sampling in pixel space, first point = largest deviation
    sel = [int(cand[np.argmax(dev[cand])])]
    dmin = np.hypot(u[cand] - u[sel[0]], v[cand] - v[sel[0]])
    while len(sel) < min(n, len(cand)):
        j = int(np.argmax(dmin))
        if dmin[j] < 3:
            break
        sel.append(int(cand[j]))
        dmin = np.minimum(dmin, np.hypot(u[cand] - u[cand[j]], v[cand] - v[cand[j]]))
    return [(float(u[i]), float(v[i])) for i in sel]

def _design_rank_ok(UV, D):
    X = np.stack([UV[:, 0] * D, UV[:, 1] * D, D, np.ones_like(D)], 1)
    X = X / np.maximum(np.abs(X).max(0), 1e-9)
    s = np.linalg.svd(X, compute_uv=False)
    return float(s[-1] / s[0])

def _query(ctx, step_id, cam, sel, depth):
    UV = []; D = []; P = []
    for i in range(0, len(sel), 32):
        chunk = sel[i:i + 32]
        try:
            r = ctx.call('robomme_pixel_to_position', observation_step_id=int(step_id), depth_window_radius=0,
                         selections=[{'camera': cam, 'u': u, 'v': v} for u, v in chunk])
        except Exception as e:
            log('pixel_to_position failed', cam, repr(e)[:300]); continue
        pts = _extract_points(reply_json(r), len(chunk))
        for (u, v), (ok, xyz) in zip(chunk, pts):
            if ok and xyz is not None and np.all(np.isfinite(xyz)) and depth[int(v), int(u)] > 0:
                UV.append((u, v)); D.append(depth[int(v), int(u)]); P.append(xyz)
    return UV, D, P

def calibrate(ctx, step_id, depth, cam, packet=None):
    """Fit the back-projection matrix of `cam` for the current observation.
    Samples a pixel grid plus off-plane pixels (objects, robot, background) so the fit
    is constrained in 3-D; cross-checks against intrinsics/pose candidates from the packet."""
    H, W = depth.shape
    us, vs = np.meshgrid(np.linspace(12, W - 13, 8), np.linspace(12, H - 13, 8))
    sel = []
    for u, v in zip(us.ravel(), vs.ravel()):
        iu, iv = int(round(u)), int(round(v))
        if depth[iv, iu] > 0:
            sel.append((float(iu), float(iv)))
    sel += _offplane_pixels(depth, 32)
    UV, D, P = _query(ctx, step_id, cam, sel, depth)
    UV = np.array(UV, float).reshape(-1, 2); D = np.array(D, float); P = np.array(P, float).reshape(-1, 3)
    M = None; med = None; cond = 0.0
    if len(P) >= 8:
        cond = _design_rank_ok(UV, D)
        best = None
        for swap in (False, True):
            uv = UV[:, ::-1] if swap else UV
            Mi, res = fit_M(uv, D, P)
            keep = res < max(0.01, np.median(res) * 3)
            if keep.sum() >= 8:
                Mi, _ = fit_M(uv[keep], D[keep], P[keep])
                res = np.linalg.norm((np.stack([uv[:, 0] * D, uv[:, 1] * D, D, np.ones_like(D)], 1) @ Mi.T) - P, axis=1)
            mres = float(np.median(res))
            if best is None or mres < best[0]:
                best = (mres, Mi, swap)
        med, M, swap = best
        if swap:
            M = M.copy(); M[:, [0, 1]] = M[:, [1, 0]]
        log(f'calib {cam}: n={len(P)} median residual {med*1000:.2f} mm swap={swap} cond={cond:.2e}')
        if med > 0.01:
            log(f'calib {cam}: residual too large, rejecting fit'); M = None
    fit_ok = M is not None and cond > 1e-4
    # packet candidates
    pk_best = None
    if packet is not None:
        for Mc, tag in packet_candidates(packet, cam):
            if len(P) >= 4:
                r = np.linalg.norm(backproject(Mc, UV[:, 0], UV[:, 1], D) - P, axis=1)
                mr = float(np.median(r))
            else:
                mr = np.inf
            if pk_best is None or mr < pk_best[0]:
                pk_best = (mr, Mc, tag)
        if pk_best is not None:
            log(f'calib {cam}: best packet model {pk_best[2]} residual {pk_best[0]*1000:.1f} mm')
    if fit_ok:
        return M
    if pk_best is not None and pk_best[0] < 0.008:
        log(f'calib {cam}: fit ill-conditioned or missing; using packet model {pk_best[2]}')
        return pk_best[1]
    if M is not None:
        log(f'calib {cam}: WARNING fit ill-conditioned (cond {cond:.1e}); 3-D off-plane geometry unreliable')
        return M
    if pk_best is not None and len(P) < 4:
        # choose the candidate that makes the dominant surface a flat horizontal table
        best = None
        for Mc, tag in packet_candidates(packet, cam):
            try:
                th = table_height(zmap(Mc, depth)[2])
                sc = th[1] if th else 0.0
                # the table must lie below the camera centre
                if th is None or th[0] > Mc[2, 3] - 0.05:
                    sc = 0.0
            except Exception:
                sc = 0.0
            if tag == 'opengl':
                sc += 0.15   # documented convention preferred unless clearly worse
            log(f'calib {cam}: candidate {tag} table score {sc:.2f}')
            if best is None or sc > best[0]:
                best = (sc, Mc, tag)
        log(f'calib {cam}: no tool samples; using packet model {best[2]} (table-plane check)')
        return best[1]
    return None

def packet_candidates(packet, cam):
    out = []
    for conv in ('opengl', 'opencv'):
        for inv in (False, True):
            Mc = M_from_packet(packet, cam, conv, inv)
            if Mc is not None:
                out.append((Mc, f'{conv}{"-inv" if inv else ""}'))
    return out

def M_from_packet(packet, cam, conv_override=None, invert=False):
    try:
        intr = find(packet, 'camera_intrinsics'); poses = find(packet, 'camera_poses')
        def pick(x):
            if isinstance(x, dict):
                for k, v in x.items():
                    if str(k).lower().startswith(cam):
                        return v
            if isinstance(x, list):
                for v in x:
                    if isinstance(v, dict) and str(v.get('camera', v.get('name', ''))).lower().startswith(cam):
                        return v
            return None
        K = pick(intr); P = pick(poses)
        if isinstance(K, dict):
            K = K.get('K') or K.get('matrix') or K.get('intrinsics') or (
                [[K['fx'], 0, K['cx']], [0, K['fy'], K['cy']], [0, 0, 1]] if 'fx' in K else None)
        if K is None or P is None:
            return None
        pos = P.get('position_m', P.get('position')) or P.get('p') or P.get('translation')
        q = P.get('quaternion_wxyz') or P.get('quaternion') or P.get('q')
        conv = conv_override or P.get('convention', find(packet, 'convention') or 'opengl')
        if invert:
            from geom import quat_wxyz_to_R
            R = quat_wxyz_to_R(q); t = np.asarray(pos, float)
            Ri = R.T; ti = -Ri @ t
            # back to quaternion via a rotation matrix path
            Mx = M_from_intrinsics_pose(np.array(K, float), ti, [1, 0, 0, 0], conv)
            Mx[:, :3] = Ri @ Mx[:, :3]
            return Mx
        return M_from_intrinsics_pose(np.array(K, float), pos, q, conv)
    except Exception as e:
        log('M_from_packet failed', repr(e)[:200])
        return None

# ---------------------------------------------------------------- per-frame geometry
def cloud(M, depth, mask=None, stride=1):
    H, W = depth.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    d = depth[::stride, ::stride]
    ok = d > 0
    if mask is not None:
        ok &= mask[::stride, ::stride]
    P = backproject(M, u[ok].astype(float), v[ok].astype(float), d[ok])
    return P, u[ok], v[ok]

def zmap(M, depth):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    P = backproject(M, u.ravel().astype(float), v.ravel().astype(float), depth.ravel())
    Z = P[:, 2].reshape(H, W); X = P[:, 0].reshape(H, W); Y = P[:, 1].reshape(H, W)
    bad = depth <= 0
    Z[bad] = np.nan; X[bad] = np.nan; Y[bad] = np.nan
    return X, Y, Z

def table_height(Z):
    z = Z[np.isfinite(Z)]
    if len(z) < 100:
        return None
    lo, hi = np.percentile(z, 1), np.percentile(z, 99)
    hi = min(hi, lo + 3.0)
    hist, edges = np.histogram(z, bins=np.arange(lo, hi + 0.002, 0.001))
    k = int(np.argmax(hist))
    zz = z[(z > edges[k] - 0.003) & (z < edges[k + 1] + 0.003)]
    return float(np.median(zz)), float(hist[k] / len(z))

# ---------------------------------------------------------------- scene model
class Scene:
    """Holds calibration, table height, colour model and helpers for one episode."""
    def __init__(self, M_front, M_wrist0, state0, z0):
        self.Mf = M_front; self.Mw0 = M_wrist0; self.state0 = state0; self.z0 = z0
        self.cm = None; self.h_obj = 0.04; self.D = 0.035
        self.front_static = None; self.static_xy = np.zeros((0, 2))

    def M_for(self, cam, state):
        if cam == 'front':
            return self.Mf
        if self.Mw0 is None or state is None or self.state0 is None:
            return None
        return moved_M(self.Mw0, self.state0, state)

    def obj_points(self, cam, rgb, depth, M, zmax=None, with_high=False):
        """World points of object-coloured pixels in the object height band."""
        if rgb is None or depth is None or M is None or self.cm is None:
            return np.zeros((0, 3)), np.zeros((0, 3))
        mk = self.cm.mask(rgb) & (depth > 0)
        if cam == 'front' and self.front_static is not None:
            mk &= ~self.front_static
        P, u, v = cloud(M, depth, mk)
        if len(P) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        zmax = self.z0 + (zmax if zmax is not None else self.h_obj + 0.015)
        band = (P[:, 2] > self.z0 + 0.004) & (P[:, 2] < zmax)
        high = (P[:, 2] >= zmax + 0.01) & (P[:, 2] < self.z0 + 0.35)
        Pb = P[band]
        if len(self.static_xy) and len(Pb):
            d = np.min(np.linalg.norm(Pb[:, None, :2] - self.static_xy[None], axis=2), 1)
            Pb = Pb[d > 0.008]
        return Pb, (P[high] if with_high else np.zeros((0, 3)))

def learn_scene_colour(scene, views):
    """views: list of (rgb, depth, M) at the execution frame (arm at rest)."""
    imgs = []; masks = []
    for rgb, depth, M in views:
        if rgb is None or depth is None or M is None:
            continue
        X, Y, Z = zmap(M, depth)
        band = np.isfinite(Z) & (Z > scene.z0 + 0.005) & (Z < scene.z0 + 0.10)
        imgs.append(rgb); masks.append(band)
    if not imgs:
        return None
    cm = learn_color(imgs, masks, log=log)
    if cm is None:
        log('colour learning with height band failed; trying RGB-only')
        cm = learn_color(imgs, None, log=log)
    return cm

def fused_points(scene, frame, zmax=None, use_wrist=True, with_high=False):
    pts = []; high = []
    for cam in CAMS:
        if cam == 'wrist' and not use_wrist:
            continue
        M = scene.M_for(cam, frame.get('state'))
        P, Hh = scene.obj_points(cam, frame.get(cam + '_rgb'), frame.get(cam + '_depth'), M, zmax, with_high)
        if len(P) > 1500:
            P = P[np.random.default_rng(0).choice(len(P), 1500, replace=False)]
        pts.append(P); high.append(Hh)
    P = np.concatenate(pts) if pts else np.zeros((0, 3))
    Hh = np.concatenate(high) if high else np.zeros((0, 3))
    return P, Hh

def detect_world_objects(scene, P):
    if len(P) < 5:
        return np.zeros((0, 2)), [], scene.D
    from tracker import grid_clusters
    cl = grid_clusters(P[:, :2], 0.004, 6)
    ext = [np.ptp(P[i, :2], 0).max() for i in cl]
    D = float(np.clip(np.median(ext), 0.015, 0.08)) if ext else scene.D
    cents, _ = detect_objects(P[:, :2], D, 6, cell=0.004)
    return cents, cl, D

# ---------------------------------------------------------------- vacancy test
def vacancy_status(scene, frame, centres, r_s, h):
    """Per object: 'vacant' / 'present' / 'unknown' at this frame."""
    n = len(centres)
    ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    offs = np.concatenate([[[0, 0]], np.stack([np.cos(ang), np.sin(ang)], 1) * r_s])
    status = []
    for i in range(n):
        votes = []
        for cam in CAMS:
            M = scene.M_for(cam, frame.get('state'))
            depth = frame.get(cam + '_depth')
            if M is None or depth is None:
                continue
            Hh, Ww = depth.shape
            P = np.concatenate([centres[i][None] + offs, np.full((len(offs), 1), scene.z0 + 0.6 * h)], 1)
            u, v, d_obj = project(M, P)
            ok = np.isfinite(u) & (u >= 0) & (u < Ww - 0.5) & (v >= 0) & (v < Hh - 0.5) & (d_obj > 0)
            if ok.sum() < 4:
                continue
            iu = np.round(u[ok]).astype(int); iv = np.round(v[ok]).astype(int)
            d_act = depth[iv, iu]
            d_tab = table_depth_along(M, iu.astype(float), iv.astype(float), scene.z0)
            good = d_act > 0
            if good.sum() < 4:
                continue
            d_act = d_act[good]; d_tab = d_tab[good]; d_o = d_obj[ok][good]
            vac = d_act >= d_tab - 0.004
            pres = (~vac) & (d_act >= d_o - 0.035)
            occ = (~vac) & (~pres)
            votes.append((vac.mean(), pres.mean(), occ.mean()))
        if not votes:
            status.append('unknown'); continue
        if any(p >= 0.4 for _, p, _ in votes):
            status.append('present')
        elif any(v >= 0.7 for v, _, _ in votes):
            status.append('vacant')
        else:
            status.append('unknown')
    return status

# ---------------------------------------------------------------- full video analysis
def ee_grasp_target(scene, frames, cents, D):
    """From per-frame robot state: first descent of the end effector (first run of frames within
    2 cm of the lowest EE height in the first 45% of the video, while low over the objects); the
    object whose frame-0 centre is under the EE at the lowest frame of that run. Returns
    (index, frame) or (None, None) when ambiguous or unavailable."""
    try:
        n = len(frames)
        ks, Z, XY = [], [], []
        for k in range(min(n, int(0.45 * n) + 1)):
            s_k = frames[k].get('state')
            if s_k is None or len(s_k) < 3:
                continue
            s_k = np.asarray(s_k, float)
            if not np.all(np.isfinite(s_k[:3])):
                continue
            ks.append(k); Z.append(s_k[2]); XY.append(s_k[:2])
        if len(ks) < 10:
            return None, None
        Z = np.array(Z); XY = np.array(XY)
        dmin = np.array([np.linalg.norm(cents - p, axis=1).min() for p in XY])
        over = dmin < 1.0 * D
        if not over.any():
            return None, None
        zmin = Z[over].min()
        if zmin > Z.max() - 0.04:          # no real descent
            return None, None
        low = over & (Z < zmin + 0.02)
        idx = np.nonzero(low)[0]
        run_end = idx[0]
        while run_end + 1 < len(low) and low[run_end + 1]:
            run_end += 1
        seg = np.arange(idx[0], run_end + 1)
        j = seg[int(np.argmin(Z[seg]))]
        d = np.linalg.norm(cents - XY[j], axis=1)
        o = np.argsort(d)
        if d[o[0]] > 0.45 * D:
            return None, None
        if len(o) > 1 and d[o[1]] < d[o[0]] + 0.5 * D:
            return None, None
        return int(o[0]), int(ks[idx[0]])
    except Exception as e:
        log('ee_grasp_target error', repr(e)[:150])
        return None, None

def ee_low_runs(scene, frames, cents, D, frac=0.45):
    """Runs of frames (first `frac` of the video) where the EE is within 2 cm of its lowest height
    over the objects. Returns list of (k_first, k_lowest, xy_lowest) in video-frame indices."""
    n = len(frames)
    ks, Z, XY = [], [], []
    for k in range(min(n, int(frac * n) + 1)):
        s_k = frames[k].get('state')
        if s_k is None or len(s_k) < 3:
            continue
        s_k = np.asarray(s_k, float)
        if not np.all(np.isfinite(s_k[:3])):
            continue
        ks.append(k); Z.append(s_k[2]); XY.append(s_k[:2])
    if len(ks) < 10:
        return []
    Z = np.array(Z); XY = np.array(XY)
    dmin = np.array([np.linalg.norm(cents - p, axis=1).min() for p in XY])
    over = dmin < 2.0 * D
    if not over.any():
        return []
    zmin = Z[over].min()
    if zmin > Z.max() - 0.04:
        return []
    low = over & (Z < zmin + 0.02)
    runs = []; i = 0
    while i < len(low):
        if low[i]:
            j = i
            while j + 1 < len(low) and low[j + 1]:
                j += 1
            seg = np.arange(i, j + 1); m = seg[int(np.argmin(Z[seg]))]
            runs.append((int(ks[i]), int(ks[m]), XY[m].copy()))
            i = j + 1
        else:
            i += 1
    return runs

def ee_place_xy(scene, frames, cents, target, D):
    """Where the expert put the target back: EE XY at the second low run (release) plus the
    object-to-EE offset at the first (grasp). None when unavailable or implausible."""
    try:
        runs = ee_low_runs(scene, frames, cents, D)
        if len(runs) < 2:
            return None, None
        g, r = runs[0], runs[1]
        off = cents[target] - g[2]
        if np.linalg.norm(off) > 0.6 * D:
            return None, None
        return r[2] + off, r[1]
    except Exception as e:
        log('ee_place_xy error', repr(e)[:150])
        return None, None

def analyse_video(scene, frames, exec_frame):
    """frames: list of dicts with front_rgb/front_depth/wrist_rgb/wrist_depth/state/t (video order).
    Returns dict with target index, final XY, diagnostics."""
    out = {'ok': False}
    n = len(frames)
    # wrist geometry validity per frame: table plane must come out at z0
    wrist_ok = []
    for k, fr in enumerate(frames):
        okw = False
        M = scene.M_for('wrist', fr.get('state'))
        if M is not None and fr.get('wrist_depth') is not None and (k % 8 == 0 or k == n - 1 or k < 5):
            X, Y, Z = zmap(M, fr['wrist_depth'])
            th = table_height(Z)
            okw = th is not None and abs(th[0] - scene.z0) < 0.012 and th[1] > 0.05
            fr['_wchk'] = (None if th is None else round(th[0] - scene.z0, 4))
        elif M is not None and k > 0:
            okw = None  # decide from neighbours
        wrist_ok.append(okw)
    # fill undecided by nearest checked
    checked = [k for k, w in enumerate(wrist_ok) if w is not None]
    for k in range(n):
        if wrist_ok[k] is None:
            j = min(checked, key=lambda c: abs(c - k)) if checked else None
            wrist_ok[k] = bool(wrist_ok[j]) if j is not None else False
    frac_w = np.mean(wrist_ok) if n else 0
    log(f'wrist geometry valid in {frac_w*100:.0f}% of video frames; samples', [frames[k].get('_wchk') for k in checked[:12]])
    def fr_view(fr, k):
        return {'front_rgb': fr.get('front_rgb'), 'front_depth': fr.get('front_depth'),
                'wrist_rgb': fr.get('wrist_rgb') if wrist_ok[k] else None,
                'wrist_depth': fr.get('wrist_depth') if wrist_ok[k] else None,
                'state': fr.get('state')}

    # front static mask: object-coloured & in-band in most frames
    acc = None; cnt = 0
    for k in range(0, n, 3):
        fr = frames[k]
        if fr.get('front_rgb') is None or fr.get('front_depth') is None:
            continue
        X, Y, Z = zmap(scene.Mf, fr['front_depth'])
        m = scene.cm.mask(fr['front_rgb']) & np.isfinite(Z) & (Z > scene.z0 + 0.004)
        acc = m.astype(float) if acc is None else acc + m
        cnt += 1
    if acc is not None and cnt:
        pers = acc / cnt > 0.7
        scene.front_static = ndimage.binary_dilation(pers, iterations=1)
        log(f'front static pixels: {int(pers.sum())}')

    # initial objects at frame 0
    v0 = fr_view(frames[0], 0)
    P0, _ = fused_points(scene, v0)
    cents, cl, D = detect_world_objects(scene, P0)
    scene.D = D
    Pe, _ = fused_points(scene, exec_frame)
    ce, _, De = detect_world_objects(scene, Pe)
    log(f'frame0 objects: {len(cents)} D={D:.3f}; exec objects: {len(ce)} D={De:.3f}')
    out['n0'] = len(cents); out['nexec'] = len(ce)
    if len(cents) < 2:
        out['why'] = 'too few objects at frame 0'
        return out
    # object height estimate from exec frame clusters
    try:
        hs = []
        from tracker import grid_clusters
        for idx in grid_clusters(Pe[:, :2], 0.004, 10):
            hs.append(np.percentile(Pe[idx, 2], 95) - scene.z0)
        if hs:
            scene.h_obj = float(np.clip(np.median(hs), 0.012, 0.10))
    except Exception:
        pass
    log(f'object height ~{scene.h_obj:.3f}')

    # vacancy over the first part of the video
    N = len(cents)
    first_vac = [None] * N
    run = [0] * N
    stat_hist = []
    r_s = 0.3 * D
    for k in range(1, n):
        if time.time() > getattr(scene, 'deadline', 1e18):
            log(f'analysis deadline reached at frame {k}; stopping this pass'); break
        st = vacancy_status(scene, fr_view(frames[k], k), cents, r_s, scene.h_obj)
        stat_hist.append(st)
        for i in range(N):
            if st[i] == 'vacant':
                run[i] += 1
                if run[i] >= 3 and first_vac[i] is None:
                    first_vac[i] = k - 2
            elif st[i] == 'present':
                run[i] = 0
        nv = sum(1 for f in first_vac if f is not None)
        if nv >= max(3, N // 2):
            break
    log('first vacancy per object:', first_vac)
    order = sorted([(f, i) for i, f in enumerate(first_vac) if f is not None])
    target = None; conf = 0.0
    if order:
        target = order[0][1]
        gap = (order[1][0] - order[0][0]) if len(order) > 1 else 99
        conf = float(gap)
    # lifted-point evidence: high object-coloured points (front + valid wrist) before shuffle
    lift_votes = np.zeros(N); first_lift = [None] * N
    kmax = min(n, (order[0][0] + 40) if order else n // 3)
    for k in range(1, kmax):
        if time.time() > getattr(scene, 'deadline', 1e18):
            log(f'analysis deadline reached at frame {k}; stopping this pass'); break
        _, Hh = fused_points(scene, fr_view(frames[k], k), with_high=True)
        if len(Hh) >= 8:
            c = np.median(Hh[:, :2], 0)
            d = np.linalg.norm(cents - c, axis=1)
            j = int(np.argmin(d))
            if d[j] < 1.5 * D:
                lift_votes[j] += 1
                if first_lift[j] is None:
                    first_lift[j] = k
    log('lift votes:', lift_votes.tolist())
    # robot-state evidence: the object under the gripper at the first low point of its descent
    ee_t, ee_k = ee_grasp_target(scene, frames, cents, D)
    if ee_t is not None:
        log(f'EE grasp evidence: object {ee_t} at frame {ee_k}')
    if target is None and lift_votes.max() > 0:
        target = int(np.argmax(lift_votes)); conf = 0.5
    elif target is not None and lift_votes.max() >= 3 and np.argmax(lift_votes) != target and conf < 6:
        log('vacancy and lift disagree; using lift evidence')
        target = int(np.argmax(lift_votes)); conf = 0.5
    if ee_t is not None and target != ee_t:
        strong_vis = target is not None and conf >= 6 and lift_votes[target] >= 3
        if not strong_vis:
            log(f'using EE grasp evidence (visual target {target}, conf {conf})')
            target = ee_t; conf = 0.5
        else:
            log('EE grasp evidence disagrees with strong visual evidence; keeping visual target')
    if target is None:
        out['why'] = 'no target identified'
        return out
    cand = [x for x in (first_vac[target], first_lift[target]) if x is not None]
    if ee_t == target and ee_k is not None:
        cand.append(ee_k)
    # EE approach (per-frame state): first frame the gripper is low and close above the target
    t_app = None
    lim = min(cand) + 5 if cand else n
    for k in range(1, min(n, lim)):
        s_k = frames[k].get('state')
        if s_k is None or len(s_k) < 3:
            continue
        s_k = np.asarray(s_k, float)
        if s_k[2] < scene.z0 + scene.h_obj + 0.08 and np.linalg.norm(s_k[:2] - cents[target]) < 1.5 * D:
            t_app = k; break
    if t_app is not None:
        cand.append(t_app + 4)
    t_lift = max(1, min(cand) - 4) if cand else 1
    log(f'target object {target} at {cents[target].round(3).tolist()} (conf gap {conf}) freeze from frame {t_lift}')

    # tracking
    tr = Tracker(cents, D, gate=1.1, min_support=6)
    frozen_until = None; seen_absent = 0; pres_run = 0; frozen_pos = None
    place_xy, place_k = ee_place_xy(scene, frames, cents, target, D)
    if place_xy is not None:
        log(f'EE put-back estimate {np.round(place_xy, 3).tolist()} at frame {place_k}')
    close = 0; min_sep = []
    for k in range(1, n):
        if time.time() > getattr(scene, 'deadline', 1e18):
            log(f'analysis deadline reached at frame {k}; stopping this pass'); break
        fr = fr_view(frames[k], k)
        P, _ = fused_points(scene, fr)
        dt = 1.0
        if frames[k].get('t') is not None and frames[k - 1].get('t') is not None:
            dt = max(0.2, (frames[k]['t'] - frames[k - 1]['t']) / 0.1)
        frozen = np.zeros(N, bool)
        P_full = P
        # while the target is frozen, points near a low gripper belong to the held object/gripper
        if k >= t_lift and frozen_until is None and len(P):
            s_k = frames[k].get('state')
            if s_k is not None and len(s_k) >= 3:
                s_k = np.asarray(s_k, float)
                if s_k[2] < scene.z0 + scene.h_obj + 0.15:
                    P = P[np.linalg.norm(P[:, :2] - s_k[:2], axis=1) > 1.0 * D]
        # target frozen from just before its lift until it is back (seen absent, then present twice)
        if k >= t_lift and frozen_until is None:
            if frozen_pos is None:
                # objects are static before the pick: the frame-0 centre is the reliable one
                # (the approaching arm can pull the track); the put-back point from EE state
                # is the best prior for re-acquisition
                if np.linalg.norm(tr.pos[target] - cents[target]) > 0.3 * D:
                    log(f'track drifted {np.linalg.norm(tr.pos[target]-cents[target])*100:.1f} cm before the pick; using frame-0 centre')
                frozen_pos = (place_xy if place_xy is not None else cents[target]).copy()
                tr.pos[target] = cents[target].copy(); tr.vel[target] = 0.0
            st = stat_hist[k - 1] if k - 1 < len(stat_hist) else None
            if st is None:
                frozen_until = k
                if place_xy is not None:
                    # robot-state put-back point (vision is still disturbed by the nearby arm)
                    tr.pos[target] = place_xy.copy(); tr.vel[target] = 0.0; tr.missing[target] = 0; ok = 'ee'
                else:
                    ok = tr.reacquire_near(target, P_full[:, :2], frozen_pos, 0.5 * D, max_shift=0.6 * D) if len(P_full) else False
                log(f'freeze ended at frame {k} (status history end); reacquired={ok} at {tr.pos[target].round(3).tolist()}')
            else:
                if st[target] == 'vacant':
                    seen_absent += 2
                elif st[target] == 'unknown':
                    seen_absent += 1
                pres_run = pres_run + 1 if st[target] == 'present' else 0
                # the arm must have left the object (per-frame EE state, when available)
                arm_away = True
                s_k = frames[k].get('state')
                if s_k is not None and len(s_k) >= 3:
                    s_k = np.asarray(s_k, float)
                    arm_away = (s_k[2] > scene.z0 + scene.h_obj + 0.04) or (np.linalg.norm(s_k[:2] - frozen_pos) > 1.5 * D)
                ee_back = place_xy is not None and place_k is not None and k > place_k and arm_away
                if (seen_absent >= 3 and pres_run >= 2 and arm_away) or ee_back:
                    frozen_until = k
                    if place_xy is not None:
                        # robot-state put-back point (vision is still disturbed by the nearby arm)
                        tr.pos[target] = place_xy.copy(); tr.vel[target] = 0.0; tr.missing[target] = 0; ok = 'ee'
                    else:
                        ok = tr.reacquire_near(target, P_full[:, :2], frozen_pos, 0.5 * D, max_shift=0.6 * D) if len(P_full) else False
                    log(f'target back at frame {k}; reacquired={ok} at {tr.pos[target].round(3).tolist()} (frozen at {frozen_pos.round(3).tolist()})')
                else:
                    frozen[target] = True
        tr.step(P[:, :2], dt=dt, frozen=frozen)
        others = np.delete(tr.pos, target, 0)
        if len(others):
            dmin = float(np.min(np.linalg.norm(others - tr.pos[target], axis=1)))
            min_sep.append(dmin)
            if dmin < 0.8 * D:
                close += 1
    final = tr.pos[target].copy()
    # final association with exec detections
    out.update({'ok': True, 'target': target, 'final_xy': final, 'conf': conf, 'close': close,
                'min_sep': float(np.min(min_sep)) if min_sep else None, 'cents0': cents,
                'tracks_final': tr.pos.copy(), 'missing_final': int(tr.missing[target]), 'D': D,
                'exec_cents': ce, 'hist': np.array(tr.hist), 'miss': np.array(tr.miss_hist), 'cents0_target': target})
    log(f'track final target xy {final.round(3).tolist()} missing={tr.missing[target]} close_frames={close}')
    return out

# ---------------------------------------------------------------- button
def find_button(scene, views, hover_xy=None, ee_xy=None):
    """views: list of (cam, rgb, depth, M). Returns (xy, top_z) or None."""
    pts = []; tall = []
    for cam, rgb, depth, M in views:
        if rgb is None or depth is None or M is None:
            continue
        h, s, v = rgb_to_hsv(rgb)
        X, Y, Z = zmap(M, depth)
        m = (s < 0.22) & (v > 0.35) & np.isfinite(Z) & (Z > scene.z0 + 0.003) & (Z < scene.z0 + 0.06)
        P = np.stack([X[m], Y[m], Z[m], v[m]], 1)
        pts.append(P)
        # tall structures (occluder blocks, robot base): any surface 7-25 cm above the table
        t = np.isfinite(Z) & (Z > scene.z0 + 0.07) & (Z < scene.z0 + 0.20)
        if ee_xy is not None:
            t &= np.hypot(X - ee_xy[0], Y - ee_xy[1]) > 0.10
        tall.append(np.stack([X[t], Y[t]], 1))
    if not pts:
        return None
    P = np.concatenate(pts)
    T = np.concatenate(tall) if tall else np.zeros((0, 2))
    tk = set()
    if len(T) > 20:
        tk = set(map(tuple, np.floor(T / 0.006).astype(int).tolist()))
        tk = {(a + i, b + j) for a, b in tk for i in (-1, 0, 1) for j in (-1, 0, 1)}
    if len(P) < 10:
        return None
    from tracker import grid_clusters
    cl = grid_clusters(P[:, :2], 0.004, 15)
    best = None
    for idx in cl:
        Q = P[idx]
        ext = np.ptp(Q[:, :2], 0)
        if ext.max() < 0.02 or ext.max() > 0.09 or ext.min() < 0.012:
            continue
        top = float(np.percentile(Q[:, 2], 97))
        if top - scene.z0 > 0.055:
            continue
        if tk:
            pk = np.floor(Q[:, :2] / 0.006).astype(int).tolist()
            under = float(np.mean([(a, b) in tk for a, b in pk]))
            if under > 0.4:
                log(f'button: candidate at {np.round(np.median(Q[:, :2], 0), 3).tolist()} under tall structure ({under:.2f}); skipped')
                continue
        bright = float(np.mean(Q[:, 3] > 0.75))
        c = np.median(Q[:, :2], 0)
        # cap = highest points
        cap = Q[Q[:, 2] > top - 0.006]
        cxy = np.median(cap[:, :2], 0) if len(cap) >= 5 else c
        score = len(idx) * (0.5 + bright)
        if hover_xy is not None:
            score /= (1.0 + (np.linalg.norm(cxy - hover_xy) / 0.08) ** 2)
        if best is None or score > best[0]:
            best = (score, cxy, top, ext.tolist(), len(idx), bright)
    if best is None:
        return None
    log(f'button: xy={np.round(best[1],3).tolist()} top={best[2]-scene.z0:.3f} above table ext={np.round(best[3],3).tolist()} n={best[4]} bright={best[5]:.2f}')
    return best[1], best[2]
