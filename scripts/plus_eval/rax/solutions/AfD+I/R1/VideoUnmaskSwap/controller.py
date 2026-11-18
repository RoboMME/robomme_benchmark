"""VideoUnmaskSwap controller.

Pipeline
  1. Read the reset sequence (conditioning video + first execution frame) and instruction.
  2. Fit a linear back-projection per camera against robomme_pixel_to_position.
  3. Video: detect coloured cubes before the containers drop, then track every container
     in a top-down world plane, fusing front and wrist depth points.
  4. Execution: the shuffle continues for a while; step with zero actions and keep tracking
     until nothing moves.
  5. Pick (lift) the containers in the instructed colour order; put each back down.
"""
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import obsio  # noqa: E402
from obsio import best_tf, log, find_key, find_dicts, files_of, load_manifest, frame_order_key, is_vec  # noqa: E402
from geom import (CamModel, fit_cam, cam_from_calib, rodrigues, log_rot, rotz)  # noqa: E402
from percep import (table_z_from_points, Grid, dilate, voxel_xy, color_name, parse_instruction,  # noqa: E402
                    components, square_yaw, square_size)
from tracker import Tracker, refine  # noqa: E402
from robot import Robot, EpisodeOver  # noqa: E402

OUT = os.environ.get('VUS_OUT', '/workspace/output')
WALL_LIMIT = 1150.0          # controller wall budget (s), runtime kills at 1200
CFG = dict(
    vmax=0.02,               # m per control step for free moves
    vslow=0.008,             # near objects
    wmax=0.12,               # rad per step
    above=0.09,              # hover height above container top (m)
    lift=0.12,               # lift height (m)
    grip_steps=14,
    open_steps=10,
    hold_steps=6,
    refine=False,            # close-up refinement: biased ~2 cm in mock (arm occlusion), never helped
    refine_cov=0.85,
    survey_dz=0.22,          # survey height above container tops after each put-back
    survey=False,            # re-localize after each put-back (biased by arm occlusion in tests)
)


def wall():
    return time.time() - obsio.T0


# ====================================================================== state
class World:
    def __init__(self):
        self.cam_alts = {}
        self.cams = {}        # 'front' / 'wrist' -> CamModel at exec step 0
        self.state0 = None
        self.table_z = None
        self.grid = None
        self.static_video = None
        self.static_exec = None
        self.H = 0.05
        self.s = 0.05
        self.tracker = None
        self.colors = []
        self.color_conf = []
        self.instruction = ''
        self.video_tcp = None
        self.diag = {}


W = World()


def save_json(name, obj):
    try:
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, name), 'w') as fh:
            json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    except Exception as e:
        log('save_json failed', name, repr(e))


# =============================================================== calibration
def pick_calib_pixels(depth, n=16):
    H, Wd = depth.shape
    out = []
    gs = int(np.ceil(np.sqrt(n * 2)))
    cand = []
    for v in np.linspace(12, H - 13, gs):
        for u in np.linspace(12, Wd - 13, gs):
            u, v = int(u), int(v)
            win = depth[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
            if (win <= 0).any():
                continue
            spread = float(win.max() - win.min())
            cand.append((spread, u, v))
    cand.sort()
    # prefer low spread, keep spatial spread by taking the first n
    for s, u, v in cand[:n]:
        out.append((u, v))
    return out


def parse_points(reply, n):
    """Extract n world points (or None) from a pixel_to_position reply."""
    for m in reply.all():
        lists = []
        def rec(o):
            if isinstance(o, list):
                if len(o) == n and all(isinstance(x, dict) for x in o):
                    lists.append(o)
                for x in o:
                    rec(x)
            elif isinstance(o, dict):
                for x in o.values():
                    rec(x)
        rec(m)
        for L in lists:
            pts = []
            for d in L:
                p = None
                for k in ('position', 'world_position', 'world', 'point', 'xyz', 'position_m', 'world_point', 'point_world'):
                    if k in d and is_vec(d[k], 3):
                        p = d[k]; break
                if p is None:
                    for k, v in d.items():
                        if is_vec(v, 3) and not any(t in k.lower() for t in ('pixel', 'uv', 'cam', 'normal')):
                            p = v; break
                if p is None:
                    for k, v in d.items():
                        if isinstance(v, dict):
                            for kk in ('x', 'X'):
                                if kk in v and 'y' in v and 'z' in v:
                                    p = [v['x'], v['y'], v['z']]
                valid = d.get('valid', True)
                pts.append(np.asarray(p, float) if (p is not None and valid is not False) else None)
            if any(p is not None for p in pts):
                return pts
    return None


def calibrate(R, step_id, imgs, meta):
    sel = []
    pix = {}
    for cam in ('front', 'wrist'):
        dep = imgs.get((cam, 'depth'))
        if dep is None:
            continue
        pp = pick_calib_pixels(dep, 16)
        pix[cam] = pp
        for u, v in pp:
            sel.append({'camera': cam, 'u': float(u), 'v': float(v)})
    pts = None
    if sel:
        try:
            r = R.pix2pos(step_id, sel[:32], 0)
            pts = parse_points(r, len(sel[:32]))
            W.diag['pix2pos_sample'] = r.meta if isinstance(r.meta, dict) else str(r.meta)[:2000]
        except Exception as e:
            log('pix2pos failed', repr(e)[:400])
    # convention-based candidates
    calib = calib_candidates(meta)
    i = 0
    for cam in ('front', 'wrist'):
        if cam not in pix:
            continue
        dep = imgs[(cam, 'depth')]
        uv = np.array(pix[cam], float)
        d = np.array([dep[int(v), int(u)] for u, v in pix[cam]])
        P = pts[i:i + len(uv)] if pts else None
        i += len(uv)
        best = None
        if P is not None:
            ok = np.array([p is not None for p in P])
            if ok.sum() >= 6:
                PP = np.array([p for p in P if p is not None])
                cands = []   # (priority, rms, model)
                for swap in (False, True):
                    if swap:
                        dd = np.array([dep[int(u), int(v)] if (int(u) < dep.shape[0] and int(v) < dep.shape[1]) else 0
                                       for u, v in pix[cam]])[ok]
                        if (dd <= 0).any():
                            continue
                        m, rms = fit_cam(uv[ok][:, ::-1], dd, PP)
                        m.source = 'fit-transposed'
                    else:
                        m, rms = fit_cam(uv[ok], d[ok], PP)
                    log('fit', cam, 'transposed' if swap else 'direct', 'rms', round(rms, 5))
                    cands.append((2, rms, m))
                for name, Km in calib_K(meta, cam):
                    cm = fit_rigid_with_K(Km, name, uv[ok], d[ok], PP)
                    if cm is not None:
                        log('rigid-K fit', cam, name, 'rms', round(cm.rms, 5))
                        cands.append((0, cm.rms, cm))
                for name, cm in calib.get(cam, []):
                    pr = cm.backproject(uv[ok][:, 0], uv[ok][:, 1], d[ok])
                    rms = float(np.sqrt(((pr - PP) ** 2).sum(1).mean()))
                    cm.rms = rms
                    log('calib candidate', cam, name, 'rms', round(rms, 5))
                    cands.append((1, rms, cm))
                good = [c for c in cands if c[1] < 0.008]
                if good:
                    pool = sorted(good, key=lambda c: (c[0], c[1]))
                else:
                    pool = sorted(cands, key=lambda c: (c[1], c[0]))
                if pool:
                    best = (pool[0][2], pool[0][1])
                W.cam_alts[cam] = [c[2] for c in sorted(cands, key=lambda c: (c[1] >= 0.008, c[0], c[1]))]
        if best is None and calib.get(cam):
            best = (calib[cam][0][1], None)
            log('using unverified calib for', cam, calib[cam][0][0])
        if best is not None and (best[1] is None or best[1] < 0.02):
            W.cams[cam] = best[0]
            log('camera', cam, 'model', best[0].source, 'rms', best[1])
        else:
            log('camera', cam, 'NO MODEL', best[1] if best else None)


def calib_K(meta, cam):
    """[(convention, K)] for a camera from the metadata intrinsics."""
    out = []
    try:
        v = find_key(meta, 'camera_intrinsics')
        intr = v[0] if v else None
        if not intr:
            return out
        K = _get_cam_entry(intr, cam)
        Km = _to_K(K) if K is not None else None
        if Km is not None:
            out = [('opengl', Km), ('opencv', Km)]
    except Exception as e:
        log('calib_K failed', repr(e)[:200])
    return out


def fit_rigid_with_K(Km, conv, uv, d, P):
    """World = R @ Xc + C with Xc = B(K) [u d, v d, d]; Kabsch fit of R, C."""
    try:
        fx, fy, cx, cy = Km[0, 0], Km[1, 1], Km[0, 2], Km[1, 2]
        if conv == 'opengl':
            B = np.array([[1 / fx, 0, -cx / fx], [0, -1 / fy, cy / fy], [0, 0, -1.0]])
        else:
            B = np.array([[1 / fx, 0, -cx / fx], [0, 1 / fy, -cy / fy], [0, 0, 1.0]])
        U = np.stack([uv[:, 0] * d, uv[:, 1] * d, d], 1)
        Xc = U @ B.T
        mx, mp = Xc.mean(0), P.mean(0)
        Hm = (Xc - mx).T @ (P - mp)
        Us, S, Vt = np.linalg.svd(Hm)
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ Us.T))])
        Rm = Vt.T @ D @ Us.T
        C = mp - Rm @ mx
        res = Xc @ Rm.T + C - P
        rms = float(np.sqrt((res ** 2).sum(1).mean()))
        return CamModel(Rm @ B, C, 'rigidK-' + conv, rms)
    except Exception as e:
        log('rigid fit failed', repr(e)[:200])
        return None


def calib_candidates(meta):
    out = {}
    try:
        intr = None
        poses = None
        for m in [meta]:
            v = find_key(m, 'camera_intrinsics')
            intr = v[0] if v else None
            v = find_key(m, 'camera_poses')
            poses = v[0] if v else None
        if not intr or not poses:
            return out
        for cam in ('front', 'wrist'):
            K = _get_cam_entry(intr, cam)
            P = _get_cam_entry(poses, cam)
            if K is None or P is None:
                continue
            Km = _to_K(K)
            pos, q, conv = _to_pose(P)
            if Km is None or pos is None:
                continue
            out[cam] = [('opengl', cam_from_calib(Km, pos, q, 'opengl')),
                        ('opencv', cam_from_calib(Km, pos, q, 'opencv'))]
            if str(conv).lower().startswith('opencv'):
                out[cam].reverse()
    except Exception as e:
        log('calib parse failed', repr(e)[:300])
    return out


def _get_cam_entry(obj, cam):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if cam in str(k).lower() or (cam == 'front' and 'base' in str(k).lower()):
                return v
        if 'camera' in obj and cam in str(obj['camera']).lower():
            return obj
    if isinstance(obj, list):
        for v in obj:
            if isinstance(v, dict) and cam in str(v.get('camera', v.get('name', ''))).lower():
                return v
    return None


def _to_K(K):
    if isinstance(K, dict):
        for k in ('K', 'matrix', 'intrinsic_matrix', 'intrinsics'):
            if k in K:
                return _to_K(K[k])
        if all(k in K for k in ('fx', 'fy', 'cx', 'cy')):
            return np.array([[K['fx'], 0, K['cx']], [0, K['fy'], K['cy']], [0, 0, 1.0]])
        return None
    a = np.asarray(K, float)
    if a.shape == (3, 3):
        return a
    if a.size == 9:
        return a.reshape(3, 3)
    if a.size == 4:
        return np.array([[a[0], 0, a[2]], [0, a[1], a[3]], [0, 0, 1.0]])
    return None


def _to_pose(P):
    if isinstance(P, dict):
        pos = None
        q = None
        for k in ('position_m', 'position', 'p', 'translation', 'pos'):
            if k in P and is_vec(P[k], 3):
                pos = P[k]
        for k in ('quaternion_wxyz', 'quat_wxyz', 'wxyz', 'q', 'quaternion'):
            if k in P and is_vec(P[k], 4):
                q = P[k]
        conv = P.get('convention', 'opengl')
        return pos, q, conv
    a = np.asarray(P, float)
    if a.size == 7:
        return a[:3], a[3:], 'opengl'
    return None, None, None


# ================================================================ point sets
def cam_points(cam, rgb, depth, stride=1):
    P, vv, uu = cam.backproject_image(depth, None, stride)
    cols = rgb[vv, uu] if rgb is not None else None
    return P, cols


def robot_filter(P, tcp):
    if tcp is None or len(P) == 0:
        return np.ones(len(P), bool)
    dxy = np.linalg.norm(P[:, :2] - tcp[None, :2], axis=1)
    return ~((dxy < 0.07) & (P[:, 2] > tcp[2] - 0.02))


def frame_points(frame_imgs, cams, tcp, static, zlo, zhi, with_colors=False):
    xs, cs = [], []
    for cam, cm in cams.items():
        dep = frame_imgs.get((cam, 'depth'))
        if dep is None or cm is None:
            continue
        rgb = frame_imgs.get((cam, 'rgb'))
        P, C = cam_points(cm, rgb, dep)
        if len(P) == 0:
            continue
        z = P[:, 2] - W.table_z
        ok = (z > zlo) & (z < zhi) & robot_filter(P, tcp)
        if static is not None:
            st, inside = W.grid.lookup(static, P[:, :2])
            ok &= inside & ~st
        xs.append(P[ok])
        if with_colors and C is not None:
            cs.append(C[ok])
        elif with_colors:
            cs.append(np.zeros((ok.sum(), 3)))
    if not xs:
        return np.zeros((0, 3)), np.zeros((0, 3))
    X = np.concatenate(xs)
    Cc = np.concatenate(cs) if with_colors else None
    return X, Cc


# ===================================================================== video
def state_of(frame):
    s = frame.get('state')
    if is_vec(s, 8):
        return np.asarray(s, float)
    return None


def wrist_model_for(state):
    cm0 = W.cams.get('wrist')
    if cm0 is None or state is None or W.state0 is None:
        return None
    R0 = rodrigues(W.state0[3:6]); R1 = rodrigues(state[3:6])
    return cm0.transformed(R1 @ R0.T, W.state0[:3], state[:3])


def process_video(R, vframes):
    """Detect cubes, initialise containers, track through the video."""
    t0 = wall()
    nf = len(vframes)
    log('video frames', nf)
    if nf < 5:
        return False
    # --- per-frame camera models (wrist moves with the EE state)
    def cams_for(fr):
        d = {}
        if 'front' in W.cams:
            d['front'] = W.cams['front']
        if W.diag.get('use_wrist_video', True):
            wm = wrist_model_for(state_of(fr))
            if wm is not None:
                d['wrist'] = wm
        return d

    f0 = R.frame_images(vframes[0])
    st0 = state_of(vframes[0])
    W.video_tcp = st0[:3] if st0 is not None else None
    # table z and grid from front frame 0
    Pf, _ = cam_points(W.cams['front'], f0.get(('front', 'rgb')), f0[('front', 'depth')], 2)
    if W.table_z is None:
        W.table_z = table_z_from_points(Pf[:, 2])
    tz = W.table_z
    near = np.abs(Pf[:, 2] - tz) < 0.01
    xy = Pf[near, :2]
    lo = np.percentile(xy, 1, axis=0) - 0.1
    hi = np.percentile(xy, 99, axis=0) + 0.1
    W.grid = Grid(lo[0], lo[1], hi[0], hi[1], 0.005)
    log('table z', round(tz, 4), 'grid', W.grid.nx, W.grid.ny, 'xy range', lo.round(3), hi.round(3))
    # wrist sanity on frame 0 (try alternative wrist models if the chosen one fails)
    def wrist_ok(st):
        wm0 = wrist_model_for(st)
        if wm0 is None or ('wrist', 'depth') not in f0:
            return False, None, None
        Pw, _ = cam_points(wm0, None, f0[('wrist', 'depth')], 2)
        tzw = table_z_from_points(Pw[:, 2]) if len(Pw) else None
        frac = float(np.mean(np.abs(Pw[:, 2] - tz) < 0.01)) if len(Pw) else 0
        return (tzw is not None and abs(tzw - tz) <= 0.012 and frac >= 0.25), tzw, frac
    okw, tzw, frac = wrist_ok(st0)
    log('wrist video sanity: table z', tzw, 'frac on table', None if frac is None else round(frac, 3),
        'model', getattr(W.cams.get('wrist'), 'source', None))
    if not okw and W.cams.get('wrist') is not None:
        cur = W.cams['wrist']
        for alt in W.cam_alts.get('wrist', []):
            if alt is cur or (alt.rms is not None and alt.rms > 0.02):
                continue
            W.cams['wrist'] = alt
            okw, tzw, frac = wrist_ok(st0)
            log('  alt wrist model', alt.source, 'rms', alt.rms, 'sanity', okw, tzw, frac)
            if okw:
                break
        if not okw:
            W.cams['wrist'] = cur
    if not okw:
        W.diag['use_wrist_video'] = False
        log('WRIST VIDEO DISABLED')
    # static occupancy from frame 0
    Xs, _ = frame_points(f0, cams_for(vframes[0]), None, None, 0.006, 0.25)
    occ = W.grid.occupancy(Xs[:, :2]) if len(Xs) else np.zeros((W.grid.nx, W.grid.ny), bool)
    W.static_video = dilate(occ, 1)
    # exec static: remove cells near the video robot TCP (the robot moves away)
    se = W.static_video.copy()
    if W.video_tcp is not None and W.state0 is not None and np.linalg.norm(W.video_tcp - W.state0[:3]) > 0.05:
        gx = W.grid.x0 + (np.arange(W.grid.nx) + 0.5) * W.grid.g
        gy = W.grid.y0 + (np.arange(W.grid.ny) + 0.5) * W.grid.g
        GX, GY = np.meshgrid(gx, gy, indexing='ij')
        near_r = np.hypot(GX - W.video_tcp[0], GY - W.video_tcp[1]) < 0.13
        se &= ~near_r
    W.static_exec = se
    log('static cells video', int(W.static_video.sum()), 'exec', int(W.static_exec.sum()))

    # --- pass 1: foreground counts to find cube frames and the container drop
    cache = {}
    counts = []
    for i, fr in enumerate(vframes[:min(nf, 120)]):
        im = R.frame_images(fr)
        cache[i] = im
        X, _ = frame_points(im, cams_for(fr), (state_of(fr)[:3] if state_of(fr) is not None else None),
                            W.static_video, 0.005, 0.25)
        counts.append(len(voxel_xy(X[:, :2], 0.004)))
    counts = np.array(counts)
    drop = None
    for i in range(1, len(counts)):
        if counts[i - 1] > 3 and counts[i] > 2.0 * counts[i - 1] + 20:
            drop = i
            break
    if drop is None:
        for i in range(1, len(counts)):
            if counts[i] > 30:
                drop = i; break
    log('fg counts head', counts[:60].tolist(), 'drop index', drop)
    if drop is None or drop < 1:
        return False
    # --- cubes from the frames just before the drop
    cube_pts, cube_cols = [], []
    for j in (drop - 1, drop - 2):
        if j < 0 or counts[j] <= 3:
            continue
        fr = vframes[j]
        X, C = frame_points(cache[j], cams_for(fr), (state_of(fr)[:3] if state_of(fr) is not None else None),
                            W.static_video, 0.004, 0.12, with_colors=True)
        cube_pts.append(X); cube_cols.append(C)
        break
    if not cube_pts or len(cube_pts[0]) == 0:
        log('no cube points')
        return False
    X = cube_pts[0]; C = cube_cols[0]
    g3 = Grid(W.grid.x0, W.grid.y0, W.grid.x0 + W.grid.nx * W.grid.g, W.grid.y0 + W.grid.ny * W.grid.g, 0.003)
    occ = g3.occupancy(X[:, :2])
    n, lab, areas = components(dilate(occ, 1))
    ix, iy, ok = g3.idx(X[:, :2])
    plab = np.zeros(len(X), int)
    plab[ok] = lab[ix[ok], iy[ok]]
    cubes = []
    for k in range(1, n):
        sel = plab == k
        if sel.sum() < 6:
            continue
        cubes.append(dict(xy=np.median(X[sel, :2], 0), n=int(sel.sum()), area=int(areas[k]),
                          sel=sel, zmax=float(np.percentile(X[sel, 2] - W.table_z, 95))))
    if not cubes:
        log('no cubes found')
        return False
    med_area = np.median([c['area'] for c in cubes])
    final_cubes = []
    for c in cubes:
        m = int(round(c['area'] / med_area))
        if m >= 2 and len(cubes) >= 3:
            pts = X[c['sel'], :2]
            init = pts[np.random.RandomState(0).choice(len(pts), m, replace=False)]
            cc = refine(init, pts, 0.1, 10)
            dd = np.linalg.norm(pts[:, None] - cc[None], axis=2).argmin(1)
            idx = np.nonzero(c['sel'])[0]
            for q in range(m):
                s2 = np.zeros(len(X), bool); s2[idx[dd == q]] = True
                if s2.sum() >= 4:
                    final_cubes.append(dict(xy=np.median(X[s2, :2], 0), n=int(s2.sum()), sel=s2))
            log('split merged cube blob into', m)
        else:
            final_cubes.append(c)
    for c in final_cubes:
        nm, conf = color_name(C[c['sel']])
        c['color'] = nm; c['conf'] = conf
    log('cubes:', [(c['color'], round(c['conf'], 2), c['xy'].round(3).tolist(), c['n']) for c in final_cubes])
    W.diag['cubes'] = [dict(color=c['color'], conf=c['conf'], xy=c['xy'].tolist(), n=c['n']) for c in final_cubes]

    # --- containers at the drop frame
    fr = vframes[drop]
    Xd, _ = frame_points(cache[drop], cams_for(fr), (state_of(fr)[:3] if state_of(fr) is not None else None),
                         W.static_video, 0.01, 0.25)
    if len(Xd) < 20:
        log('too few drop points')
        return False
    H = float(np.clip(np.percentile(Xd[:, 2] - W.table_z, 98), 0.02, 0.2))
    W.H = H
    Vd = voxel_xy(Xd[:, :2], 0.004)
    centers = np.array([c['xy'] for c in final_cubes])
    centers = refine(centers, Vd, 0.07, 6)
    dd = np.linalg.norm(Vd[:, None] - centers[None], axis=2)
    a = dd.argmin(1)
    sizes = []
    for k in range(len(centers)):
        pts = Vd[(a == k) & (dd[np.arange(len(Vd)), a] < 0.07)]
        s = square_size(pts)
        if s is not None and len(pts) > 20:
            sizes.append(s)
    s = float(np.median(sizes)) if sizes else 0.05
    s = float(np.clip(s, 0.025, 0.09))
    W.s = s
    log('container height', round(H, 4), 'size', round(s, 4), 'sizes', np.round(sizes, 3).tolist())
    # residual blobs = containers whose cube we missed
    colors = [c['color'] for c in final_cubes]
    confs = [c['conf'] for c in final_cubes]
    for _ in range(3):
        dd = np.linalg.norm(Vd[:, None] - centers[None], axis=2).min(1)
        res = Vd[dd > 0.8 * s]
        if len(res) < 10:
            break
        occ = W.grid.occupancy(res)
        n, lab, areas = components(occ)
        k = int(np.argmax(areas[1:]) + 1) if n > 1 else None
        if k is None:
            break
        cell_area = W.grid.g ** 2
        if areas[k] * cell_area < 0.35 * s * s:
            break
        ix, iy, ok = W.grid.idx(res)
        sel = np.zeros(len(res), bool); sel[ok] = lab[ix[ok], iy[ok]] == k
        newc = res[sel].mean(0)
        centers = np.vstack([centers, newc]); colors.append('unknown'); confs.append(0.0)
        centers = refine(centers, Vd, 0.8 * s, 4)
        log('added container without cube at', newc.round(3))
    W.colors = colors; W.color_conf = confs
    tr = Tracker(centers, s, radius_frac=0.75, min_sep_frac=0.7, min_pts=4, vel_alpha=0.5, max_step_frac=0.5)
    W.tracker = tr
    # --- track through remaining video frames
    def clock(fr, i):
        for k in ('source_frame_index',):
            if isinstance(fr.get(k), (int, float)):
                return float(fr[k])
        ts = fr.get('timestamp_s')
        if isinstance(ts, (int, float)):
            return float(ts) * 20.0
        return float(i)
    prev = clock(vframes[drop], drop)
    W.diag['video_clock_end'] = None
    supp = []
    W.diag['hist_src'] = [prev]
    for i in range(drop + 1, nf):
        fr = vframes[i]
        im = cache.pop(i, None) or R.frame_images(fr)
        st = state_of(fr)
        X, _ = frame_points(im, cams_for(fr), (st[:3] if st is not None else None), W.static_video, 0.01, H + 0.02)
        V = voxel_xy(X[:, :2], 0.004)
        c = clock(fr, i)
        dt = max(c - prev, 1e-3)
        tr.step(V, dt=dt, frame=i)
        supp.append(int((tr.support >= tr.min_pts).sum()))
        W.diag['hist_src'].append(c)
        prev = c
        if wall() > 520:
            log('video processing too slow, stopping at', i)
            break
    W.diag['video_clock_end'] = prev
    if W.diag.get('debug'):
        try:
            np.savez(f'{OUT}/video_hist.npz', hist=np.array(tr.history), src=np.array(W.diag['hist_src']))
        except Exception as e:
            log('hist save failed', repr(e))
    log('video tracked in', round(wall() - t0, 1), 's; collapse events', len(tr.events),
        'mean visible tracks', round(float(np.mean(supp)) if supp else 0, 2), 'of', len(tr.c))
    log('positions end video', np.round(tr.c, 3).tolist())
    return True


# ============================================================ motion helpers
class Motion:
    def __init__(self, R):
        self.R = R
        self.grip = -1.0
        self.R0 = rodrigues(R.state[3:6]) if R.state is not None else np.eye(3)
        self.yaw_base = None

    def pose(self):
        s = self.R.state
        return s[:3].copy(), rodrigues(s[3:6])

    def finger_axis(self, Rcur=None):
        """Horizontal unit vector along which the fingers close (world)."""
        if Rcur is None:
            _, Rcur = self.pose()
        wm = W.cams.get('wrist')
        if wm is not None:
            a = (Rcur @ self.R0.T) @ wm.u_axis_world()
        else:
            a = Rcur[:, 1]
        a = np.array([a[0], a[1], 0.0])
        n = np.linalg.norm(a)
        return a / n if n > 1e-6 else np.array([1.0, 0, 0])

    def approach_axis(self, Rcur):
        cols = [Rcur[:, i] for i in range(3)] + [-Rcur[:, i] for i in range(3)]
        dots = [c[2] for c in cols]
        k = int(np.argmin(dots))  # most downward
        return cols[k], -dots[k]

    def target_rot(self, yaw_delta):
        _, Rc = self.pose()
        a, cosd = self.approach_axis(Rc)
        Ralign = np.eye(3)
        if cosd > 0.75:
            down = np.array([0, 0, -1.0])
            ax = np.cross(a, down)
            sn = np.linalg.norm(ax)
            ang = np.arctan2(sn, float(np.dot(a, down)))
            if sn > 1e-8 and ang > 0.01:
                Ralign = rodrigues(ax / sn * ang)
        return rotz(yaw_delta) @ Ralign @ Rc

    def move(self, target, Rtarget=None, vmax=None, tol=0.004, rtol=0.03, max_calls=25, grip=None):
        vmax = vmax or CFG['vmax']
        if grip is not None:
            self.grip = grip
        last_err = None
        stall = 0
        for it in range(max_calls):
            p, Rc = self.pose()
            dp = np.asarray(target, float) - p
            dr = log_rot(Rtarget @ Rc.T) if Rtarget is not None else np.zeros(3)
            ep, er = np.linalg.norm(dp), np.linalg.norm(dr)
            if ep < tol and er < rtol:
                return True
            if last_err is not None and ep > last_err - 0.0005 and er < rtol:
                stall += 1
                if stall >= 3:
                    log('move stalled at err', round(ep, 4))
                    return False
            else:
                stall = 0
            last_err = ep
            n = int(np.ceil(max(ep / vmax, er / CFG['wmax'], 1)))
            n = min(n, 8)
            row = list(dp / n / 0.05) + list(dr / n / 0.5) + [self.grip]
            self.R.step([row] * n)
            self.check_budget()
        return False

    def gripper(self, value, steps):
        self.grip = value
        self.R.step([[0, 0, 0, 0, 0, 0, value]] * steps)

    def check_budget(self):
        if wall() > WALL_LIMIT:
            raise TimeoutError('wall budget')


# ========================================================== exec perception
def exec_points(imgs, state, zlo, zhi, with_colors=False, cams=('front', 'wrist')):
    d = {}
    if 'front' in cams and 'front' in W.cams:
        d['front'] = W.cams['front']
    if 'wrist' in cams:
        wm = wrist_model_for(state) if state is not None else W.cams.get('wrist')
        if wm is not None:
            d['wrist'] = wm
    return frame_points(imgs, d, state[:3] if state is not None else None, W.static_exec, zlo, zhi, with_colors)


def get_imgs(R, reply):
    """Current front/wrist rgb+depth: from this reply, else the last step reply of the same
    step, else a fresh observation."""
    need = (('front', 'depth'), ('wrist', 'depth'), ('front', 'rgb'), ('wrist', 'rgb'))
    imgs, src = R.current_images(reply, R.step_id) if reply is not None else ({}, {})
    cands = []
    lsr = getattr(R, 'last_step_reply', None)
    if lsr is not None and lsr is not reply and getattr(R, 'last_step_reply_id', None) == R.step_id:
        cands.append(('laststep', lambda: lsr))
    cands.append(('observe', R.observe))
    for name, fn in cands:
        if all(k in imgs for k in need):
            break
        try:
            r2 = fn()
            im2, s2 = R.current_images(r2, R.step_id)
            for k, v in im2.items():
                if k not in imgs:
                    imgs[k] = v; src[k] = name + ':' + s2[k]
        except EpisodeOver:
            raise
        except Exception as e:
            log('image fetch failed', name, repr(e)[:200])
    return imgs, src


def track_exec(R, imgs0):
    """Keep tracking while the containers still move; return when static."""
    tr = W.tracker
    H = W.H
    # step 0 frame (identical moment to the video end)
    X, _ = exec_points(imgs0, R.state, 0.01, H + 0.02)
    tr.step(voxel_xy(X[:, :2]), dt=1.0, frame='exec0')
    prev_rgb = imgs0.get(('front', 'rgb'))
    still = 0
    moved_any = False
    n = 0
    max_wait = 220
    if R.max_steps:
        max_wait = int(min(max_wait, max(30, R.max_steps * 0.25)))
    src_log = None
    probe_w = []
    while n < max_wait:
        g = 0.0 if 2 <= n < 10 else -1.0      # gripper probe: half command for 8 steps
        r = R.step([[0, 0, 0, 0, 0, 0, g]])
        n += 1
        if R.state is not None:
            probe_w.append((n, g, float(R.state[6] + R.state[7])))
        if n == 10:
            gripper_probe(probe_w)
        imgs, src = get_imgs(R, r) if r is not None else get_imgs(R, R.observe())
        if src_log is None:
            src_log = {str(k): v for k, v in src.items()}
            log('exec image sources', src_log)
        X, _ = exec_points(imgs, R.state, 0.01, H + 0.02)
        old = tr.c.copy()
        tr.step(voxel_xy(X[:, :2]), dt=1.0, frame=f'exec{n}')
        mv = float(np.linalg.norm(tr.c - old, axis=1).max())
        rgb = imgs.get(('front', 'rgb'))
        diff = None
        if rgb is not None and prev_rgb is not None and rgb.shape == prev_rgb.shape:
            diff = int((np.abs(rgb.astype(int) - prev_rgb.astype(int)).max(2) > 40).sum())
        prev_rgb = rgb
        moving = (diff is not None and diff > 25) or mv > 0.0015
        if moving:
            moved_any = True
            still = 0
        else:
            still += 1
        if n % 10 == 0 or not moving:
            log('exec step', n, 'R.step', R.step_id, 'diff', diff, 'mv', round(mv, 4), 'still', still)
        if still >= 8 and n >= 30 and n >= 22:
            break
        if wall() > WALL_LIMIT - 300:
            log('stop exec tracking (wall)')
            break
    log('exec tracking done after', n, 'steps; moved_any', moved_any)
    return imgs


def gripper_probe(probe_w):
    """Learn whether intermediate gripper commands give intermediate openings."""
    try:
        w_open = max(w for n, g, w in probe_w if n <= 2) if any(n <= 2 for n, g, w in probe_w) else 0.08
        half = [w for n, g, w in probe_w if g == 0.0]
        w_half = half[-1] if half else None
        W.diag['gripper_probe'] = dict(samples=probe_w, w_open=w_open, w_half=w_half)
        if w_half is not None and w_open > 0.05 and 0.15 * w_open < w_half < 0.85 * w_open:
            # assume linear: w(g) = w_open + (g + 1) * (w_half - w_open)
            W.grip_map = (w_open, w_half)
            log('gripper is continuous: open', round(w_open, 4), 'at g=0', round(w_half, 4))
        else:
            W.grip_map = None
            log('gripper probe: binary or unclear', round(w_open, 4), w_half)
    except Exception as e:
        log('gripper probe failed', repr(e)[:200])
        W.grip_map = None


def open_cmd(width):
    """Gripper command giving about `width` total opening (or full open)."""
    gm = getattr(W, 'grip_map', None)
    if not gm:
        return -1.0
    w_open, w_half = gm
    if width >= w_open - 0.002:
        return -1.0
    g = (width - w_open) / (w_half - w_open) - 1.0
    return float(np.clip(g, -1.0, 0.6))


def measure_finger_offset(imgs, state):
    """EE z minus the lowest finger point, from near-range wrist depth (camera fitted at step 0)."""
    try:
        dep = imgs.get(('wrist', 'depth'))
        wm = W.cams.get('wrist')
        if dep is None or wm is None or state is None:
            return None
        m = (dep > 0.01) & (dep < 0.22)
        if m.sum() < 30:
            log('finger offset: few near pixels', int(m.sum()))
            return None
        P, _, _ = wm.backproject_image(dep, m, 1)
        P = P[np.linalg.norm(P[:, :2] - state[:2], axis=1) < 0.12]
        if len(P) < 30:
            return None
        tip = float(np.percentile(P[:, 2], 2))
        off = float(state[2] - tip)
        log('finger offset (ee z - fingertip z)', round(off, 4), 'n', len(P))
        W.diag['finger_offset'] = off
        if -0.03 < off < 0.16:
            return off
    except Exception as e:
        log('finger offset failed', repr(e)[:200])
    return None


def localize_all(imgs, state):
    """Refine every track center (and yaw) from the current frame."""
    tr = W.tracker
    s, H = W.s, W.H
    X, _ = exec_points(imgs, state, 0.6 * H, H + 0.02)
    c = tr.c.copy()
    out = []
    if len(X):
        V = X[:, :2]
        for _ in range(3):
            d = np.linalg.norm(V[:, None] - c[None], axis=2)
            a = d.argmin(1)
            for k in range(len(c)):
                sel = (a == k) & (d[np.arange(len(V)), a] < 0.75 * s)
                if sel.sum() > 15:
                    c[k] = V[sel].mean(0)
        d = np.linalg.norm(V[:, None] - c[None], axis=2)
        a = d.argmin(1)
        for k in range(len(c)):
            sel = (a == k) & (d[np.arange(len(V)), a] < 0.75 * s)
            cov, ss = face_quality(V[sel], s)
            out.append(dict(c=c[k], yaw=square_yaw(V[sel]) if sel.sum() > 15 else None, n=int(sel.sum()),
                            cov=cov, size=ss))
    else:
        out = [dict(c=c[k], yaw=None, n=0) for k in range(len(c))]
    return out


def survey(R, motion, loc, centers):
    """Rise above the last pick and re-localize all containers; accept only small updates."""
    try:
        p, Rc = motion.pose()
        z = W.table_z + W.H + CFG['survey_dz']
        motion.move([p[0], p[1], z], None, vmax=0.025, grip=motion.grip, max_calls=10)
        imgs, _ = get_imgs(R, None)
        new = localize_all(imgs, R.state)
        upd = []
        for k, l in enumerate(new):
            dd = float(np.linalg.norm(l['c'] - centers[k]))
            if l['n'] > 25 and dd < 0.025:
                if dd > 0.002:
                    upd.append((k, round(dd, 4)))
                centers[k] = l['c']
                loc[k]['c'] = l['c'].copy()
                if l['yaw'] is not None:
                    loc[k]['yaw'] = l['yaw']
        log('survey updates', upd)
        tw = W.diag.get('truth_world')
        if tw is not None:
            log('  [mock] survey errs', [round(float(np.linalg.norm(tw.pos - centers[k], axis=1).min()), 4) for k in range(len(centers))])
    except EpisodeOver:
        raise
    except TimeoutError:
        raise
    except Exception as e:
        log('survey failed', repr(e)[:200])


def plan_targets():
    spec = parse_instruction(W.instruction)
    log('instruction spec', spec)
    used = set()
    plan = []
    unknown = [k for k, c in enumerate(W.colors) if c == 'unknown']
    for color, allf in spec:
        ks = [k for k, c in enumerate(W.colors) if c == color and k not in used]
        if not ks:
            # try a close colour family or an unknown-colour container
            alt = {'purple': ['pink'], 'pink': ['purple', 'red'], 'gray': ['white'], 'white': ['gray'], 'cyan': ['blue', 'green']}.get(color, [])
            ks = [k for k, c in enumerate(W.colors) if c in alt and k not in used]
            if not ks:
                ks = [k for k in unknown if k not in used][:1]
            log('no container for', color, '-> fallback', ks)
        if not allf and len(ks) > 1:
            ks = sorted(ks, key=lambda k: -W.color_conf[k])[:1]
        for k in ks:
            used.add(k)
            plan.append((color, k))
    return plan


def choose_yaw(motion, k, yaw_box, centers, half_w=None, avoid=None, prefer=None):
    """Yaw rotation (rad) for the gripper so fingers close on box faces with most clearance.
    avoid: a previously used finger-axis angle to steer away from (retry with the other faces)."""
    fa = motion.finger_axis()
    phi = np.arctan2(fa[1], fa[0])
    if yaw_box is None:
        yaw_box = phi  # no information: keep current
    best = None
    s = W.s
    hw = half_w if half_w is not None else 0.5 * s + 0.012
    for kk in range(4):
        tgt = yaw_box + kk * np.pi / 2
        dlt = (tgt - phi + np.pi / 2) % np.pi - np.pi / 2   # finger axis symmetric (mod pi)
        ang = phi + dlt
        ax = np.array([np.cos(ang), np.sin(ang)])
        c = centers[k]
        clear = 1e9
        for sgn in (-1, 1):
            f = c + sgn * ax * (hw + 0.008)
            for j in range(len(centers)):
                if j != k:
                    clear = min(clear, np.linalg.norm(centers[j] - f) - 0.5 * s)
        score = clear - 0.01 * abs(dlt)
        if avoid is not None:
            da = abs(((ang - avoid) + np.pi / 2) % np.pi - np.pi / 2)
            if da < np.deg2rad(30):
                score -= 1.0
        if prefer is not None:
            dp = abs(((ang - prefer) + np.pi / 2) % np.pi - np.pi / 2)
            if dp < np.deg2rad(30):
                score += 1.0
        if best is None or score > best[0]:
            best = (score, dlt, clear, ang)
    return best[1], best[2], best[3]


def _cluster_at(V, guess, others, s, iters=3):
    """Points of the container near `guess` (nearest-center assignment vs others)."""
    c = np.asarray(guess, float).copy()
    sel = np.zeros(len(V), bool)
    for _ in range(iters):
        cs = np.vstack([c[None], others]) if len(others) else c[None]
        d = np.linalg.norm(V[:, None] - cs[None], axis=2)
        a = d.argmin(1)
        sel = (a == 0) & (d[:, 0] < 0.75 * s)
        if sel.sum() < 10:
            return None, sel
        c = V[sel].mean(0)
    return c, sel


def face_quality(pts, s):
    """Coverage of a container top face by observed points (1 = full square), and its size."""
    if len(pts) < 5:
        return 0.0, None
    vox = voxel_xy(pts, 0.004)
    cov = len(vox) * 0.004 ** 2 / (s * s)
    return float(cov), square_size(vox)


def refine_target(R, motion, k, guess, others):
    """Close-up re-localisation of container k; accepted only if its top face is fully seen."""
    try:
        imgs, _ = get_imgs(R, None)
        s = W.s
        tw = W.diag.get('truth_world')
        for cams in (('wrist',), ('front', 'wrist')):
            X, _ = exec_points(imgs, R.state, 0.6 * W.H, W.H + 0.015, cams=cams)
            if len(X) < 20:
                continue
            V = X[:, :2]
            c, sel = _cluster_at(V, guess, others, s)
            if c is None:
                log('refine', cams, 'no target points')
                continue
            cov, ss = face_quality(V[sel], s)
            dm = float(np.linalg.norm(c - guess))
            ok = cov > CFG['refine_cov'] and ss is not None and 0.85 * s < ss < 1.2 * s and dm < 0.6 * s
            msg = ''
            if tw is not None:
                msg = f' [mock] err {np.linalg.norm(tw.pos - c, axis=1).min():.4f} vs guess {np.linalg.norm(tw.pos - guess, axis=1).min():.4f}'
            log('refine', cams, 'cov', round(cov, 2), 'size', None if ss is None else round(ss, 4), 'moved', round(dm, 4), 'accept', ok, msg)
            if ok:
                return c, square_yaw(V[sel])
        return guess, None
    except EpisodeOver:
        raise
    except Exception as e:
        log('refine failed', repr(e)[:200])
        return guess, None


def pick(R, motion, k, loc, centers, last, max_attempts=4):
    s, H, tz = W.s, W.H, W.table_z
    top = tz + H
    off = W.diag.get('finger_offset_used', 0.008)
    def ee(tip_z):
        return tip_z + off
    c = loc[k]['c'].copy()
    yaw_box = loc[k]['yaw']
    others = np.array([centers[j] for j in range(len(centers)) if j != k])
    tip_above = top + CFG['above']
    tip_pre = top + 0.018
    tip_grasp = max(tz + 0.008, top - 0.6 * H)
    width = float(min(0.08, s * 1.06 + 0.022))
    gopen = gopen_nom = open_cmd(width)
    w_open = W.grip_map[0] if getattr(W, 'grip_map', None) else 0.08
    hw = 0.5 * min(width, w_open)
    avoid = None
    nblocked = 0
    c0 = c.copy()
    shifts = []          # pending (offset vector, finger-axis angle) after an empty close
    for attempt in range(max_attempts):
        if R.remaining() < 45:
            log('not enough steps for another attempt')
            return False
        prefer = None
        c = c0.copy()
        if shifts:
            dv, prefer = shifts.pop(0)
            c = c0 + dv
            log('empty close before: shifting target along the finger axis by', np.round(dv, 4).tolist())
        dlt, clear, ang = choose_yaw(motion, k, yaw_box, np.vstack([centers[:k], c[None], centers[k + 1:]]), hw, avoid, prefer)
        Rt = motion.target_rot(dlt)
        # narrower opening when neighbours are close or the fingers were blocked before
        hw_min = 0.5 * s + 0.006
        hw_att = hw - max(0.0, 0.008 - clear) - 0.006 * nblocked
        hw_att = float(min(hw, max(hw_min, hw_att)))
        gopen = open_cmd(2 * hw_att) if hw_att < hw - 1e-4 else gopen_nom
        log(f'pick k={k} color={W.colors[k]} attempt {attempt} c={np.round(c, 3).tolist()} yawbox={None if yaw_box is None else round(yaw_box, 2)} '
            f'dyaw={round(float(dlt), 3)} clear={round(float(clear), 3)} gopen={round(gopen, 2)}')
        motion.move([c[0], c[1], ee(tip_above)], Rt, grip=gopen, max_calls=30)
        # close-up refinement only after a failed attempt (overview estimate is better on average)
        for _ in range(2 if (CFG['refine'] and attempt >= 1 and prefer is None) else 0):
            c2, y2 = refine_target(R, motion, k, c, others)
            if y2 is not None:
                yaw_box = y2
            moved = np.linalg.norm(c2 - c)
            c = c2
            dlt, clear, ang = choose_yaw(motion, k, yaw_box, np.vstack([centers[:k], c[None], centers[k + 1:]]), hw, avoid)
            Rt = motion.target_rot(dlt)
            motion.move([c[0], c[1], ee(tip_above)], Rt, grip=gopen, max_calls=12)
            if moved < 0.003 and abs(dlt) < 0.05:
                break
        if prefer is None:
            c0 = c.copy()
        tw = W.diag.get('truth_world')
        if tw is not None:   # offline mock evaluation only
            dtr = np.linalg.norm(tw.pos - c, axis=1)
            log('  [mock] refined err', round(float(dtr.min()), 4), 'unrefined err',
                round(float(np.linalg.norm(tw.pos - loc[k]['c'], axis=1).min()), 4))
        log('refined c', np.round(c, 3).tolist(), 'yaw', None if yaw_box is None else round(yaw_box, 2), 'clear', round(float(clear), 3))
        centers = np.vstack([centers[:k], c[None], centers[k + 1:]])
        # descend
        motion.move([c[0], c[1], ee(tip_pre)], Rt, vmax=0.015, grip=gopen, max_calls=14)
        motion.move([c[0], c[1], ee(tip_grasp)], Rt, vmax=CFG['vslow'], tol=0.003, grip=gopen, max_calls=14)
        p, _ = motion.pose()
        log('at grasp pose', np.round(p, 3).tolist(), 'target z', round(ee(tip_grasp), 3))
        blocked = p[2] > ee(tip_grasp) + 0.012
        if blocked and attempt >= 2 and p[2] < ee(top + 0.005):
            log('descent blocked again but below the container top; closing anyway')
            blocked = False
        if blocked:
            log('descent blocked; rise and retry with other faces')
            nblocked += 1
            avoid = ang
            motion.move([p[0], p[1], ee(tip_above)], Rt, vmax=0.015, grip=gopen, max_calls=10)
            continue
        motion.gripper(1.0, CFG['grip_steps'])
        w = float(R.state[6] + R.state[7])
        log('finger opening after close', round(w, 4))
        if w < 0.006 or w > s + 0.03:
            log('grasp missed; retry')
            motion.gripper(gopen, CFG['open_steps'])
            motion.move([c[0], c[1], ee(tip_above)], Rt, vmax=0.015, grip=gopen, max_calls=10)
            if w < 0.006 and prefer is None and not shifts:
                # nothing between the fingers: the box lies beyond the swept span along this axis
                ax = np.array([np.cos(ang), np.sin(ang)])
                dsh = max(0.008, hw_att - 0.5 * s)
                c0 = c.copy()
                shifts = [(dsh * ax, ang), (-dsh * ax, ang)]
                avoid = None
            else:
                avoid = ang if attempt % 2 == 0 else None
            continue
        # lift
        motion.move([c[0], c[1], ee(tip_grasp) + CFG['lift']], Rt, vmax=0.012, grip=1.0, max_calls=15)
        motion.gripper(1.0, CFG['hold_steps'])
        w2 = float(R.state[6] + R.state[7])
        log('lifted; opening', round(w2, 4), 'ended', R.ended, 'success', R.success_seen)
        if w2 < 0.006:
            log('lost the container during lift; retry')
            motion.gripper(gopen, CFG['open_steps'])
            continue
        if last:
            motion.gripper(1.0, 10)
            return True
        # put back
        motion.move([c[0], c[1], ee(tip_grasp) + 0.006], Rt, vmax=0.012, grip=1.0, max_calls=15)
        motion.gripper(gopen, CFG['open_steps'])
        motion.move([c[0], c[1], ee(tip_above)], Rt, vmax=0.015, grip=gopen, max_calls=10)
        loc[k]['c'] = c.copy()
        return True
    return False


# ====================================================================== main
def run(ctx):
    R = Robot(ctx)
    st = R.status()
    if st is not None:
        W.diag['status_meta'] = st.meta
    r0 = R.observe()
    meta = r0.meta
    W.diag['observe0_meta_keys'] = list(meta.keys()) if isinstance(meta, dict) else str(type(meta))
    save_json('observe0_meta.json', meta)
    instr = find_key(meta, 'instruction')
    W.instruction = instr[0] if instr and isinstance(instr[0], str) else ''
    if not W.instruction and st is not None:
        v = find_key(st.meta, 'instruction')
        W.instruction = v[0] if v and isinstance(v[0], str) else ''
    log('instruction:', W.instruction)
    log('max steps', R.max_steps, 'step', R.step_id, 'state', None if R.state is None else np.round(R.state, 4).tolist())
    # reset manifest
    man = None
    for mp in obsio.manifest_paths(r0):
        m = load_manifest(mp)
        if m and m.get('frames'):
            if man is None or len(m['frames']) > len(man['frames']):
                man = m
    if man is None and st is not None:
        for mp in obsio.manifest_paths(st):
            m = load_manifest(mp)
            if m and m.get('frames'):
                man = m
    frames = sorted(man['frames'], key=frame_order_key) if man else []
    vframes = [f for f in frames if str(f.get('phase', '')).lower() in ('video_demo', 'video', 'conditioning')]
    eframes = [f for f in frames if str(f.get('phase', '')).lower().startswith('exec')]
    log('manifest frames', len(frames), 'video', len(vframes), 'exec', len(eframes))
    # step-0 images: the observation's own images are the reference for pixel_to_position;
    # the manifest's execution frame tells us how saved frame files are oriented.
    man_imgs = R.frame_images(eframes[-1]) if eframes else {}
    if eframes and R.state is None:
        s0 = state_of(eframes[-1])
        if s0 is not None:
            R.state = s0
    for k, v in man_imgs.items():
        R.ref_imgs[k] = v
    obs_imgs, obs_src = R.current_images(r0, None)
    obs_src = {k: v for k, v in obs_src.items() if v != 'manifest'}
    obs_imgs = {k: v for k, v in obs_imgs.items() if k in obs_src}
    log('observe0 image sources', {str(k): v for k, v in obs_src.items()})
    for cam in ('front', 'wrist'):
        a, b = man_imgs.get((cam, 'rgb')), obs_imgs.get((cam, 'rgb'))
        if a is not None and b is not None and obs_src.get((cam, 'rgb')) != 'json-path':
            tf, diff = best_tf(a, b)
            ident = obsio._mad(a, b)
            log('frame orientation', cam, 'best', tf, 'diff', round(diff, 2), 'identity diff', round(ident, 2))
            W.diag['frame_tf_' + cam] = [tf, diff, ident]
            if tf != (False, 0) and diff < 10 and diff < 0.5 * ident:
                R.frame_tf[cam] = tf
                log('APPLYING frame transform for', cam, tf)
    if R.frame_tf:
        man_imgs = R.frame_images(eframes[-1]) if eframes else {}
    imgs0 = dict(man_imgs)
    for k, v in obs_imgs.items():
        if v is not None and (k not in imgs0 or v.shape == imgs0[k].shape):
            imgs0[k] = v
    log('step0 images', {str(k): v.shape for k, v in imgs0.items()})
    for k, v in imgs0.items():
        R.ref_imgs[k] = v
    W.state0 = R.state.copy() if R.state is not None else None
    calibrate(R, R.step_id, imgs0, meta)
    if 'front' not in W.cams:
        log('FATAL: no front camera model')
        return
    # table from the execution front view
    Pf, _ = cam_points(W.cams['front'], None, imgs0[('front', 'depth')], 2)
    W.table_z = table_z_from_points(Pf[:, 2])
    log('table z (exec front)', W.table_z)
    ok = process_video(R, vframes)
    if not ok:
        log('video processing failed; no reliable memory')
        return
    save_json('video_tracks.json', dict(colors=W.colors, conf=W.color_conf, c=W.tracker.c, s=W.s, H=W.H,
                                        events=W.tracker.events[:50]))
    fo = measure_finger_offset(imgs0, R.state)
    W.diag['finger_offset_used'] = float(np.clip(fo, -0.01, 0.12)) if fo is not None else 0.008
    log('finger offset used', W.diag['finger_offset_used'])
    plan0 = plan_targets()
    log('plan from video', plan0)
    imgs = track_exec(R, imgs0)
    loc = localize_all(imgs, R.state)
    centers = np.array([l['c'] for l in loc])
    W.tracker.c = centers.copy()
    log('final centers', np.round(centers, 3).tolist(), 'yaws', [None if l['yaw'] is None else round(l['yaw'], 2) for l in loc], 'cov', [round(l.get('cov', 0), 2) for l in loc])
    plan = plan_targets()
    save_json('plan.json', dict(plan=plan, colors=W.colors, centers=centers, instruction=W.instruction))
    motion = Motion(R)
    for i, (color, k) in enumerate(plan):
        if R.ended:
            break
        ok = pick(R, motion, k, loc, centers, last=(i == len(plan) - 1))
        log('pick', i, color, k, 'ok', ok, 'step', R.step_id, 'wall', round(wall(), 1))
        if not ok and not R.ended and R.remaining() >= 60:
            log('pick failed; one more full try on the same target')
            ok = pick(R, motion, k, loc, centers, last=(i == len(plan) - 1), max_attempts=3)
            log('pick retry', i, color, k, 'ok', ok, 'step', R.step_id)
        if not ok:
            log('target could not be picked; continuing anyway (nothing to lose)')
        if CFG['survey'] and i < len(plan) - 1 and not R.ended and R.remaining() > 80:
            survey(R, motion, loc, centers)
        if R.remaining() < 30:
            log('step budget nearly exhausted')
            break
    if not R.ended:
        try:
            R.status()
        except Exception:
            pass


def main(ctx):
    obsio.T0 = time.time()
    try:
        run(ctx)
    except EpisodeOver as e:
        log('episode over:', e)
    except TimeoutError as e:
        log('timeout guard:', e)
    except Exception:
        log('CONTROLLER ERROR', traceback.format_exc())
    finally:
        try:
            save_json('diag.json', {k: v for k, v in W.diag.items() if k != 'truth_world'})
            os.makedirs(OUT, exist_ok=True)
            with open(os.path.join(OUT, 'controller_log.txt'), 'w') as fh:
                fh.write('\n'.join(obsio.LOG))
        except Exception:
            pass
