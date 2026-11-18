"""VideoUnmask controller (RAX, AfD, developed offline).

Pipeline
 1. observe step 0, parse instruction -> ordered (colour, one|all) goals
 2. read the reset frame manifest (conditioning video + first execution frame)
 3. self-calibrate front / wrist back-projection models against
    robomme_pixel_to_position on the execution frame depth
 4. detect coloured cubes in every video frame (front + wrist), back-project with
    that frame's depth (wrist camera moved rigidly with the recorded EE state),
    cluster per colour -> hidden cube positions
 5. for each goal in order: hover over the cube position, refine the covering
    container (top surface, centre, yaw) from wrist/front depth samples, align the
    gripper, descend, grasp, lift; put it back unless it is the last one.
"""
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np

import io_utils as iu
from geometry import (CamModel, fit_cam_model, models_from_calibration, rotvec_to_mat,
                      mat_to_rotvec, rotz, wrap_half_pi, wrap_pi)
from perception import (color_mask, blobs, parse_instruction, table_height, cluster_points,
                        object_region, segment_objects, COLOR_WORDS)
from robot import Robot, EpisodeEnded

OUT_DIR = '/workspace/output'
T0 = time.time()
CFG = {
    'transit_h': 0.20,      # above table
    'hover_h': 0.09,        # above container top for the close refinement view
    'grasp_below_top': 0.022,
    'lift_h': 0.13,
    'min_tip_h': 0.010,
    'wall_limit_s': 1080.0,
}


class Logger:
    def __init__(self):
        self.lines = []
        try:
            os.makedirs(OUT_DIR, exist_ok=True)
            self.f = open(os.path.join(OUT_DIR, 'vu_log.txt'), 'a')
        except Exception:
            self.f = None

    def __call__(self, msg):
        s = '[%7.2f] %s' % (time.time() - T0, msg)
        try:
            print(s, flush=True)
        except Exception:
            pass
        if self.f:
            try:
                self.f.write(s + '\n')
                self.f.flush()
            except Exception:
                pass


def save_json(name, obj):
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, name), 'w') as f:
            json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    except Exception:
        pass


# ====================================================================== calibration

def calibrate_camera(robot, cam, depth, meta, log):
    """Return the best CamModel for `cam` at the current observation."""
    H, W = depth.shape
    sels, ds = [], []
    rng = np.random.RandomState(1)
    grid = [(u, v) for v in np.linspace(12, H - 13, 9) for u in np.linspace(12, W - 13, 9)]
    for (u, v) in grid:
        u, v = int(round(u)), int(round(v))
        dv = depth[v, u]
        if dv > 0.02 and dv < 20:
            # avoid depth edges
            win = depth[max(0, v - 1):v + 2, max(0, u - 1):u + 2]
            if np.ptp(win) < 0.01:
                sels.append((cam, u, v))
                ds.append(dv)
    sels, ds = sels[:64], ds[:64]
    res = robot.p2p(sels, radius=0) if sels else []
    uv, dd, WW = [], [], []
    for (c, u, v), d, (xyz, sp) in zip(sels, ds, res):
        if xyz is not None:
            uv.append((u, v))
            dd.append(d)
            WW.append(xyz)
    best = None
    if len(uv) >= 8:
        m = fit_cam_model(np.array(uv), np.array(dd), np.array(WW))
        if m is not None:
            log('calib %s: fit on %d pts, median resid %.4f max %.4f' % (cam, len(uv), m.resid, m.resid_max))
            best = m
            if m.resid > 0.006:
                # maybe (u,v) are swapped in the tool: refit using depth at [u,v]
                dd2 = [depth[int(v), int(u)] if False else depth[int(u), int(v)] for (u, v) in uv]
                m2 = fit_cam_model(np.array(uv)[:, ::-1], np.array(dd2), np.array(WW))
                if m2 is not None and m2.resid < m.resid * 0.5:
                    log('calib %s: swapped uv fits better (%.4f) -- unexpected' % (cam, m2.resid))
    # analytic candidates from the published calibration
    cands = analytic_models(meta, cam)
    if cands:
        if len(uv) >= 3:
            U = np.array(uv, float)
            D = np.array(dd)
            WWa = np.array(WW)
            scored = []
            for c in cands:
                e = np.median(np.linalg.norm(c.backproject(U[:, 0], U[:, 1], D) - WWa, axis=1))
                c.resid = float(e)
                scored.append((e, c))
            scored.sort(key=lambda t: t[0])
            log('calib %s analytic: %s' % (cam, ', '.join('%s=%.4f' % (c.kind, e) for e, c in scored)))
            if best is None or scored[0][0] < best.resid * 0.7:
                best = scored[0][1]
        elif best is None:
            best = cands[0]
            log('calib %s: no p2p points, using %s blindly' % (cam, best.kind))
    return best


def analytic_models(meta, cam):
    try:
        intr = iu.find_key(meta, ('camera_intrinsics', 'intrinsics'))
        poses = iu.find_key(meta, ('camera_poses', 'extrinsics', 'camera_pose'))
        K = pick_cam(intr, cam)
        P = pick_cam(poses, cam)
        if K is None or P is None:
            return []
        if isinstance(K, dict):
            if 'matrix' in K or 'K' in K:
                K = K.get('matrix', K.get('K'))
            elif all(k in K for k in ('fx', 'fy', 'cx', 'cy')):
                K = [[K['fx'], 0, K['cx']], [0, K['fy'], K['cy']], [0, 0, 1]]
        K = np.array(K, float).reshape(3, 3)
        pos, q = None, None
        if isinstance(P, dict):
            pos = P.get('position_m', P.get('position')) or P.get('p') or P.get('translation')
            q = P.get('quaternion_wxyz') or P.get('quaternion') or P.get('q')
            if pos is None and ('matrix' in P or 'pose' in P):
                M = np.array(P.get('matrix', P.get('pose')), float).reshape(4, 4)
                return _models_from_matrix(K, M)
        elif iu.is_num_list(P, 16) or (isinstance(P, list) and len(P) == 4):
            M = np.array(P, float).reshape(4, 4)
            return _models_from_matrix(K, M)
        if pos is None or q is None:
            return []
        return models_from_calibration(K, pos, q)
    except Exception as e:
        return []


def _models_from_matrix(K, M):
    from geometry import models_from_calibration as mfc
    R = M[:3, :3]
    t = M[:3, 3]
    # convert R to quaternion
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        q = [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]
    else:
        q = [1, 0, 0, 0]
    return mfc(K, t, q)


def pick_cam(obj, cam):
    if obj is None:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if cam in kl or (cam == 'wrist' and 'hand' in kl) or (cam == 'front' and ('base' in kl or 'third' in kl)):
                return v
        if 'camera' in obj:
            return obj if str(obj.get('camera')).lower().startswith(cam) else None
    if isinstance(obj, list):
        for v in obj:
            if isinstance(v, dict) and str(v.get('camera', v.get('name', ''))).lower().startswith(cam):
                return v
    return None


# ====================================================================== video analysis

DETECT_COLORS = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']


def analyze_video(frames, mdir, models, R0, p0, colors, table_z, log):
    """Return per-colour list of cube clusters {center, n, cams}."""
    dets = []   # dict(color, cam, fi, xyz, area)
    n_vid = 0
    wrist_state_ok = True
    last_wrist = None   # (depth, model) of the last video frame
    det_colors = list(colors) + [c for c in DETECT_COLORS if c not in colors]
    for fi, fr in enumerate(frames):
        if str(fr.get('phase', '')).lower() not in ('video_demo', 'video', 'conditioning'):
            continue
        n_vid += 1
        files = iu.frame_files(fr, mdir)
        st = fr.get('state')
        wmodel = None
        if models.get('wrist') is not None:
            if iu.is_num_list(st) and len(st) >= 6:
                Rf = rotvec_to_mat(st[3:6])
                pf = np.array(st[:3], float)
                wmodel = models['wrist'].moved(R0, p0, Rf, pf)
            else:
                wrist_state_ok = False
                wmodel = models['wrist']
        for cam in ('front', 'wrist'):
            m = models.get(cam) if cam == 'front' else wmodel
            if m is None or (cam, 'rgb') not in files:
                continue
            try:
                rgb = iu.read_rgb(files[(cam, 'rgb')][0])
            except Exception:
                continue
            depth = None
            if cam == 'wrist':
                last_wrist = (files.get(('wrist', 'depth')), m)
            for col in det_colors:
                mask = color_mask(rgb, col)
                if mask.sum() < 5:
                    continue
                for b in blobs(mask, min_area=6 if cam == 'front' else 12):
                    if depth is None and (cam, 'depth') in files:
                        try:
                            depth = iu.read_depth(*files[(cam, 'depth')])
                        except Exception:
                            depth = None
                    u = float(np.median(b['xs']))
                    v = float(np.median(b['ys']))
                    if depth is not None:
                        dv = depth[b['ys'], b['xs']]
                        ok = dv > 0
                        if ok.sum() < max(3, 0.3 * len(dv)):
                            continue
                        d = float(np.median(dv[ok]))
                    elif table_z is not None:
                        # no depth: intersect the pixel ray with the cube-centre plane
                        ray = m.A @ np.array([u, v, 1.0])
                        if abs(ray[2]) < 1e-9:
                            continue
                        d = (table_z + 0.012 - m.C[2]) / ray[2]
                        if d <= 0:
                            continue
                    else:
                        continue
                    xyz = m.backproject(np.array(u), np.array(v), np.array(d))
                    dets.append({'color': col, 'cam': cam, 'fi': fi, 'xyz': np.asarray(xyz, float).reshape(3),
                                 'area': b['area'], 'uv': (u, v)})
    log('video: %d video frames, %d raw detections, wrist_state_ok=%s' % (n_vid, len(dets), wrist_state_ok))
    if table_z is not None and wrist_state_ok:
        dets = [d for d in dets if table_z - 0.03 < d['xyz'][2] < table_z + 0.08]
    elif table_z is not None:
        dets = [d for d in dets if d['cam'] == 'wrist' or table_z - 0.03 < d['xyz'][2] < table_z + 0.08]
    log('video: %d detections kept' % len(dets))
    # front/wrist consistency -> rigid correction for wrist points
    T = wrist_correction(dets, last_wrist, table_z, wrist_state_ok, log)
    if T is not None:
        Rt, tt = T
        for d in dets:
            if d['cam'] == 'wrist':
                d['xyz'] = Rt @ d['xyz'] + tt
        if table_z is not None:
            dets = [d for d in dets if table_z - 0.03 < d['xyz'][2] < table_z + 0.08]
    out = {}
    for col in colors:
        D = [d for d in dets if d['color'] == col]
        if not D:
            out[col] = []
            continue
        pts = np.array([d['xyz'] for d in D])
        w = np.array([d['area'] for d in D], float)
        cl = cluster_points(pts, w, radius=0.03)
        res = []
        for c in cl:
            mem = [D[i] for i in c['members']]
            fis = sorted(set(m['fi'] for m in mem))
            cams = sorted(set(m['cam'] for m in mem))
            # prefer front-camera estimate when available (static calibration) but
            # use all members' median
            res.append({'center': np.median(np.array([m['xyz'] for m in mem]), axis=0), 'n': len(mem),
                        'n_frames': len(fis), 'cams': cams, 'first': fis[0], 'last': fis[-1]})
        res.sort(key=lambda r: -r['n'])
        out[col] = res
        log('video: colour %s clusters: %s' % (col, '; '.join('(%.3f,%.3f,%.3f) n=%d frames=%d cams=%s' % (
            r['center'][0], r['center'][1], r['center'][2], r['n'], r['n_frames'], '+'.join(r['cams'])) for r in res)))
    return out, dets


def fit_plane(P, iters=200, thr=0.004, rng=None):
    rng = rng or np.random.RandomState(0)
    best = (0, None)
    n = len(P)
    if n < 50:
        return None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        a, b, c = P[idx]
        nv = np.cross(b - a, c - a)
        if np.linalg.norm(nv) < 1e-9:
            continue
        nv = nv / np.linalg.norm(nv)
        dist = np.abs((P - a) @ nv)
        k = int((dist < thr).sum())
        if k > best[0]:
            best = (k, (nv, a))
    if best[1] is None:
        return None
    nv, a = best[1]
    inl = P[np.abs((P - a) @ nv) < thr]
    cen = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - cen, full_matrices=False)
    nv = vt[2]
    if nv[2] < 0:
        nv = -nv
    return nv, cen, len(inl) / n


def rot_between(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def correspondences(dets):
    """Pairs (front_xyz, wrist_xyz) of the same cube: same colour, overlapping frames."""
    pairs = []
    for col in set(d['color'] for d in dets):
        F = [d for d in dets if d['color'] == col and d['cam'] == 'front']
        Wr = [d for d in dets if d['color'] == col and d['cam'] == 'wrist']
        if not F or not Wr:
            continue
        fcl = [c for c in cluster_points(np.array([d['xyz'] for d in F]), radius=0.03) if c['n'] >= 3]
        wcl = [c for c in cluster_points(np.array([d['xyz'] for d in Wr]), radius=0.03) if c['n'] >= 3]
        for fc in fcl:
            ff = set(F[i]['fi'] for i in fc['members'])
            matches = []
            for wc in wcl:
                wf = set(Wr[i]['fi'] for i in wc['members'])
                ov = len(ff & wf) / max(1, min(len(ff), len(wf)))
                if ov > 0.5:
                    matches.append(wc)
            # ambiguous if several same-colour cubes appear at the same time
            others = [c for c in fcl if c is not fc and len(set(F[i]['fi'] for i in c['members']) & ff) > 0.5 * len(ff)]
            if len(matches) == 1 and not others:
                pairs.append((fc['center'], matches[0]['center'], col))
    return pairs


def wrist_correction(dets, last_wrist, table_z, state_ok, log):
    """Rigid transform (R, t) mapping wrist-derived points onto the front frame, or None."""
    Rt, tt = np.eye(3), np.zeros(3)
    changed = False
    if not state_ok and last_wrist is not None and last_wrist[0] is not None and table_z is not None:
        try:
            depth = iu.read_depth(*last_wrist[0])
            P = depth_cloud(last_wrist[1], depth, stride=3)
            pl = fit_plane(P)
            if pl is not None:
                nv, cen, frac = pl
                Rp = rot_between(nv, np.array([0, 0, 1.0]))
                tp = np.array([0, 0, table_z]) - Rp @ cen
                tp[:2] = 0.0
                tp = tp + (np.array([cen[0], cen[1], 0]) - Rp @ np.array([cen[0], cen[1], 0])) * np.array([1, 1, 0])
                Rt, tt = Rp, tp
                changed = True
                log('wrist plane alignment: normal %s inliers %.2f -> tilt %.1f deg, dz %.3f' % (
                    np.round(nv, 3).tolist(), frac, math.degrees(math.acos(min(1, nv[2]))), table_z - cen[2]))
        except Exception as e:
            log('plane alignment failed %r' % (e,))
    # apply provisional transform for correspondence search
    D2 = []
    for d in dets:
        dd = dict(d)
        if d['cam'] == 'wrist':
            dd['xyz'] = Rt @ d['xyz'] + tt
        D2.append(dd)
    pairs = correspondences(D2)
    if not pairs:
        log('wrist correction: no front/wrist correspondences (state_ok=%s)' % state_ok)
        return (Rt, tt) if changed else None
    F = np.array([p[0] for p in pairs])
    Wp = np.array([p[1] for p in pairs])
    offs = F - Wp
    log('wrist correction: %d correspondences %s offsets %s' % (
        len(pairs), [p[2] for p in pairs], np.round(offs, 3).tolist()))
    med = np.median(offs, axis=0)
    if state_ok and np.linalg.norm(med[:2]) < 0.012:
        return (Rt, tt) if changed else None
    if state_ok:
        # the recorded EE state should already be right: only accept a small,
        # consistent correction (guards against false colour pairs)
        consistent = np.linalg.norm(offs[:, :2] - med[None, :2], axis=1) < 0.015
        if len(offs) == 1 and np.linalg.norm(med[:2]) > 0.03:
            log('wrist correction: single pair with large offset rejected')
            return None
        if len(offs) >= 2 and consistent.sum() < 2:
            log('wrist correction: inconsistent pairs rejected')
            return None
        if np.linalg.norm(med[:2]) > 0.06:
            log('wrist correction: offset too large for state_ok, rejected')
            return None
        keep = consistent if len(offs) >= 2 else np.ones(len(offs), bool)
        F, Wp, offs = F[keep], Wp[keep], offs[keep]
        med = np.median(offs, axis=0)
    # 2-D rigid (yaw + xy) if >= 2 well separated pairs, else translation
    if len(pairs) >= 2 and np.max(np.linalg.norm(Wp[:, None, :2] - Wp[None, :, :2], axis=2)) > 0.05:
        fc, wc = F[:, :2].mean(0), Wp[:, :2].mean(0)
        H = (Wp[:, :2] - wc).T @ (F[:, :2] - fc)
        U, S, Vt = np.linalg.svd(H)
        R2 = Vt.T @ U.T
        if np.linalg.det(R2) < 0:
            Vt[1] *= -1
            R2 = Vt.T @ U.T
        yaw = math.atan2(R2[1, 0], R2[0, 0])
        if state_ok and abs(yaw) > math.radians(20):
            R2 = np.eye(2)
            yaw = 0.0
        t2 = fc - R2 @ wc
        R3 = np.eye(3)
        R3[:2, :2] = R2
        t3 = np.array([t2[0], t2[1], 0.0])
        res = np.linalg.norm((Wp[:, :2] @ R2.T + t2) - F[:, :2], axis=1)
        log('wrist correction: 2-D rigid yaw %.1f deg t %s residuals %s' % (math.degrees(yaw), np.round(t2, 3).tolist(), np.round(res, 3).tolist()))
    else:
        R3 = np.eye(3)
        t3 = np.array([med[0], med[1], 0.0])
        log('wrist correction: translation %s' % np.round(t3, 3).tolist())
    return (R3 @ Rt, R3 @ tt + t3)


# ====================================================================== scene clouds

def depth_cloud(model, depth, stride=2):
    H, W = depth.shape
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    d = depth[vs, us]
    ok = d > 0.02
    P = model.backproject(us[ok].astype(float), vs[ok].astype(float), d[ok])
    return P


def sample_region(robot, models_now, center_xy, z_ref, half=0.07, n=13, cams=('wrist', 'front')):
    """Query pixel_to_position on pixels that see a world grid around center_xy.
    Returns (N,3) points."""
    gx = np.linspace(-half, half, n)
    pts = []
    for cam in cams:
        m = models_now.get(cam)
        if m is None:
            continue
        sels = []
        for dx in gx:
            for dy in gx:
                Wp = np.array([center_xy[0] + dx, center_xy[1] + dy, z_ref])
                u, v, d = m.project(Wp)
                u, v = float(u), float(v)
                if not (np.isfinite(u) and np.isfinite(v)) or d <= 0:
                    continue
                if 0 <= u <= 255 and 0 <= v <= 255:
                    sels.append((cam, round(u, 1), round(v, 1)))
        sels = list(dict.fromkeys(sels))
        if not sels:
            continue
        res = robot.p2p(sels, radius=0)
        got = [x for x, _ in res if x is not None]
        pts.extend(got)
    return np.array(pts) if pts else np.zeros((0, 3))


# ====================================================================== main task logic

class Task:
    def __init__(self, ctx):
        self.log = Logger()
        self.robot = Robot(ctx, self.log)
        self.models = {}
        self.table_z = None
        self.R0 = None
        self.p0 = None
        self.grip_map = []   # (cmd, opening)
        self.picked = []

    # -------------------------------------------------- setup
    def setup(self):
        log, robot = self.log, self.robot
        st = robot.status()
        log('status keys: %s; max_steps=%s' % (list(st.keys())[:30], robot.max_steps))
        r0, meta0 = robot.observe()
        self.meta0 = meta0
        log('observe keys: %s' % (list(meta0.keys())[:40],))
        save_json('meta0_keys.json', {k: (str(v)[:400]) for k, v in meta0.items()})
        instr = iu.find_key(meta0, ('instruction', 'language_instruction', 'goal', 'task_instruction'))
        if not isinstance(instr, str):
            instr = iu.find_key(st, ('instruction', 'language_instruction', 'goal'))
        self.instruction = instr if isinstance(instr, str) else ''
        self.plan = parse_instruction(self.instruction)
        log('instruction: %r -> plan %s' % (self.instruction, self.plan))
        if robot.state is None:
            raise RuntimeError('no robot state in observation')
        self.R0 = robot.R.copy()
        self.p0 = robot.pos.copy()
        log('state0 %s' % np.round(robot.state, 4).tolist())
        # manifest
        mp = iu.manifest_path_from(meta0)
        if mp is None:
            mp = iu.manifest_path_from(st)
        if mp is None:
            mp = find_reset_manifest(log)
        log('manifest: %s' % mp)
        self.frames, self.mdir = [], None
        if mp:
            man = iu.load_manifest(mp)
            self.frames = iu.sorted_frames(man)
            self.mdir = os.path.dirname(mp)
        phases = {}
        for f in self.frames:
            phases[f.get('phase')] = phases.get(f.get('phase'), 0) + 1
        log('frames: %d phases %s' % (len(self.frames), phases))
        exec_frames = [f for f in self.frames if str(f.get('phase', '')).lower() not in ('video_demo', 'video', 'conditioning')]
        self.exec_frame = exec_frames[-1] if exec_frames else (self.frames[-1] if self.frames else None)
        self.exec_depth = {}
        if self.exec_frame is not None:
            files = iu.frame_files(self.exec_frame, self.mdir)
            for cam in ('front', 'wrist'):
                if (cam, 'depth') in files:
                    try:
                        self.exec_depth[cam] = iu.read_depth(*files[(cam, 'depth')])
                    except Exception as e:
                        log('depth read fail %s %r' % (cam, e))
            est = self.exec_frame.get('state')
            if iu.is_num_list(est) and len(est) >= 6:
                dp = np.linalg.norm(np.array(est[:3]) - self.p0)
                if dp > 0.005:
                    log('WARNING exec frame state differs from observe state by %.3f' % dp)
        # depth fallback: image attachments in observe reply
        if not self.exec_depth:
            self.exec_depth = depth_from_blocks(r0, meta0, log)
        for cam in ('front', 'wrist'):
            if cam in self.exec_depth:
                self.models[cam] = calibrate_camera(robot, cam, self.exec_depth[cam], meta0, log)
            else:
                c = analytic_models(meta0, cam)
                self.models[cam] = c[0] if c else None
                log('calib %s: no depth; analytic=%s' % (cam, bool(c)))
        save_json('models.json', {k: (v.to_dict() if v else None) for k, v in self.models.items()})
        # table height and exec scene cloud
        clouds = []
        for cam in ('front', 'wrist'):
            if cam in self.exec_depth and self.models.get(cam) is not None:
                clouds.append(depth_cloud(self.models[cam], self.exec_depth[cam]))
        self.scene = np.concatenate(clouds, 0) if clouds else np.zeros((0, 3))
        if 'front' in self.exec_depth and self.models.get('front') is not None:
            self.table_z = table_height(depth_cloud(self.models['front'], self.exec_depth['front']))
        elif len(self.scene):
            self.table_z = table_height(self.scene)
        log('table z = %s' % self.table_z)
        try:
            self.objects = segment_objects(self.scene, self.table_z)
        except Exception as e:
            log('segment_objects failed %r' % (e,))
            self.objects = []
        log('step-0 objects: %d: %s' % (len(self.objects), '; '.join('(%.3f,%.3f) top %.3f ext %s' % (
            o['center_xy'][0], o['center_xy'][1], o['top_z'], np.round(o['extent'], 3).tolist()) for o in self.objects[:20])))

    # -------------------------------------------------- targets
    def find_targets(self):
        log = self.log
        colors = []
        for c, mode in self.plan:
            if c not in colors:
                colors.append(c)
        self.cubes, dets = analyze_video(self.frames, self.mdir, self.models, self.R0, self.p0, colors,
                                         self.table_z, log)
        save_json('cubes.json', self.cubes)
        targets = []
        for c, mode in self.plan:
            cl = [k for k in self.cubes.get(c, []) if k['n'] >= 3]
            cl = self.snap_and_merge(cl, c)
            if not cl:
                log('WARNING: no cube found for colour %s' % c)
                continue
            # container presence check in the exec scene
            for k in cl:
                k['support'] = self.container_support(k['center'])
            if mode == 'one':
                # most supported cluster
                best = max(cl, key=lambda k: (k['support'] > 0, k['n']))
                group = [best]
            else:
                nmax = max(k['n'] for k in cl)
                group = [k for k in cl if k['n'] >= max(3, 0.12 * nmax) or k['n_frames'] >= 5]
            targets.append((c, mode, group))
            log('goal %s/%s: %d target(s): %s' % (c, mode, len(group), [
                (np.round(k['center'], 3).tolist(), k['n'], k['support']) for k in group]))
        return targets

    def snap_and_merge(self, cl, col):
        """Snap cube clusters to step-0 container-like objects; merge duplicates."""
        objs = [o for o in getattr(self, 'objects', []) if max(o['extent']) < 0.085 and o['top_z'] - (self.table_z or 0) > 0.025]
        for k in cl:
            k['obj'] = None
            if objs:
                d = [np.linalg.norm(o['center_xy'] - k['center'][:2]) for o in objs]
                j = int(np.argmin(d))
                if d[j] < 0.035:
                    k['obj'] = j
                    k['snap_d'] = float(d[j])
        out = []
        for k in sorted(cl, key=lambda kk: -kk['n']):
            dup = None
            for o in out:
                same_obj = k['obj'] is not None and k['obj'] == o['obj']
                disjoint = not (set(k['cams']) & set(o['cams']))
                overlap = not (k['last'] < o['first'] or o['last'] < k['first'])
                close = np.linalg.norm(k['center'][:2] - o['center'][:2]) < 0.045
                if same_obj or (disjoint and overlap and close):
                    dup = o
                    break
            if dup is not None:
                self.log('merge duplicate %s cluster (%.3f,%.3f) n=%d into (%.3f,%.3f)' % (
                    col, k['center'][0], k['center'][1], k['n'], dup['center'][0], dup['center'][1]))
                dup['n'] += k['n']
                dup['cams'] = sorted(set(dup['cams']) | set(k['cams']))
                continue
            out.append(k)
        for k in out:
            if k['obj'] is not None:
                o = objs[k['obj']]
                self.log('snap %s (%.3f,%.3f) -> object (%.3f,%.3f) d=%.3f' % (
                    col, k['center'][0], k['center'][1], o['center_xy'][0], o['center_xy'][1], k['snap_d']))
                k['center'] = np.array([o['center_xy'][0], o['center_xy'][1], k['center'][2]])
        return out

    def container_support(self, xyz):
        if self.table_z is None or not len(self.scene):
            return -1
        d = np.linalg.norm(self.scene[:, :2] - xyz[None, :2], axis=1)
        return int(((d < 0.025) & (self.scene[:, 2] > self.table_z + 0.03)).sum())

    # -------------------------------------------------- live models
    def models_now(self):
        out = {'front': self.models.get('front')}
        if self.models.get('wrist') is not None:
            out['wrist'] = self.models['wrist'].moved(self.R0, self.p0, self.robot.R, self.robot.pos)
        return out

    def refine(self, xy, z_ref, cams=('wrist', 'front'), half=0.06, n=15):
        pts = sample_region(self.robot, self.models_now(), xy, z_ref, half=half, n=n, cams=cams)
        if len(pts):
            # drop points on the gripper / arm itself
            pts = pts[pts[:, 2] < self.robot.pos[2] - 0.03]
        if len(pts) < 5:
            self.log('refine: only %d points' % len(pts))
            return None, pts
        reg = object_region(pts, xy, self.table_z, radius=0.06, link=0.013)
        if reg is not None and max(reg['size']) > 0.08:
            self.log('refine: region too large %s, restricting' % (np.round(reg['size'], 3).tolist(),))
            sub = pts[np.linalg.norm(pts[:, :2] - np.asarray(xy)[None, :2], axis=1) < 0.035]
            reg2 = object_region(sub, xy, self.table_z, radius=0.035, link=0.013)
            if reg2 is not None:
                reg = reg2
        return reg, pts

    def view_pose(self, target_xy, top_z, cam_h):
        """EE position (with current orientation R0) that puts the wrist camera's
        optical centre ray through (target_xy, top_z), camera cam_h above top_z."""
        m = self.models.get('wrist')
        if m is None:
            return np.array([target_xy[0], target_xy[1], top_z + cam_h])
        mw = m.moved(self.R0, self.p0, self.R0, self.p0)
        C_rel = mw.C - self.p0
        dirv = mw.A @ np.array([128.0, 128.0, 1.0])
        if dirv[2] > -0.2 * np.linalg.norm(dirv):
            # camera not looking down: just hover above
            return np.array([target_xy[0], target_xy[1], top_z + cam_h])
        Cz = top_z + cam_h
        pz = Cz - C_rel[2]
        t = (top_z - Cz) / dirv[2]
        pxy = np.asarray(target_xy[:2]) - C_rel[:2] - t * dirv[:2]
        pz = max(pz, top_z + 0.06)
        if np.linalg.norm(pxy - np.asarray(target_xy[:2])) > 0.15:
            return np.array([target_xy[0], target_xy[1], max(pz, top_z + 0.06)])
        return np.array([pxy[0], pxy[1], pz])

    # -------------------------------------------------- gripper
    def grip_cmd_for(self, width):
        """Gripper command expected to give opening `width` (learned linear map)."""
        if len(self.grip_map) >= 2:
            g = np.array([a for a, b in self.grip_map])
            o = np.array([b for a, b in self.grip_map])
            if np.ptp(g) > 0.1 and np.ptp(o) > 0.005:
                A = np.vstack([g, np.ones_like(g)]).T
                k, c = np.linalg.lstsq(A, o, rcond=None)[0]
                if k < -1e-3:
                    return float(np.clip((width - c) / k, -1, 1))
        return -1.0

    # -------------------------------------------------- one pick
    def yaw_ee(self, R=None):
        R = self.robot.R if R is None else R
        return math.atan2(R[1, 0], R[0, 0])

    def plan_yaw(self, reg, pts, R_cur):
        """Choose a world-z rotation delta aligning the closing axis with the box."""
        b = reg['yaw'] if reg is not None else None
        if b is None:
            return 0.0
        cur = self.yaw_ee(R_cur)
        d0 = wrap_half_pi(b - cur)
        cands = [d0, d0 + math.pi / 2, d0 - math.pi / 2]
        cands = [c for c in cands if abs(c) <= math.radians(100)]
        # finger clearance: obstacle points near finger positions
        best = None
        hw = 0.5 * max(reg['size']) if reg.get('size') else 0.025
        others = pts
        if len(pts):
            regset = reg['pts']
            # obstacles: high points not belonging to the target region
            dmin = np.min(np.linalg.norm(pts[:, None, :2] - regset[None, :, :2], axis=2), axis=1) if len(regset) else np.ones(len(pts))
            others = pts[(pts[:, 2] > self.table_z + 0.012) & (dmin > 0.008)]
        for c in cands:
            Rn = rotz(c) @ R_cur
            ax = Rn[:2, 1] / (np.linalg.norm(Rn[:2, 1]) + 1e-9)
            perp = np.array([-ax[1], ax[0]])
            cost = 0
            if len(others):
                rel = others[:, :2] - reg['center_xy'][None]
                along = np.abs(rel @ ax)
                across = np.abs(rel @ perp)
                cost = int(((along > hw) & (along < 0.058) & (across < 0.015)).sum())
            score = cost * 10 + abs(c)
            if best is None or score < best[0]:
                best = (score, c, cost)
        self.log('yaw plan: box %.1f deg, ee %.1f deg -> rotate %.1f deg (clearance cost %d)' % (
            math.degrees(b), math.degrees(cur), math.degrees(best[1]), best[2]))
        return best[1]

    def pick(self, xyz, last, label):
        log, robot = self.log, self.robot
        tz = self.table_z if self.table_z is not None else float(xyz[2]) - 0.01
        transit_z = tz + CFG['transit_h']
        xy = np.array(xyz[:2], float)
        log('--- pick %s at (%.3f, %.3f) last=%s, steps used %s/%s' % (label, xy[0], xy[1], last, robot.step_id, robot.max_steps))
        R_down = self.R0
        # 1. transit
        cur = robot.pos
        if cur[2] < transit_z - 0.02:
            robot.move_to([cur[0], cur[1], transit_z], R_down, vmax=0.03, tol=0.01, max_rows=40)
        robot.grip_cmd = -1.0
        top_guess = tz + 0.06
        v1 = self.view_pose(xy, top_guess, 0.24)
        v1[2] = max(v1[2], transit_z)
        robot.move_to([v1[0], v1[1], max(robot.pos[2], transit_z)], R_down, vmax=0.035, tol=0.008, max_rows=80)
        robot.move_to(v1, R_down, vmax=0.03, tol=0.005, max_rows=40)
        # 2. refine from the high view
        reg, pts = self.refine(xy, tz + 0.04)
        if reg is None or np.linalg.norm(reg['center_xy'] - xy) > 0.045:
            if reg is not None:
                self.log('refine1 centre (%.3f,%.3f) too far from video estimate; ignoring' % tuple(reg['center_xy']))
            self.log('refine1 failed; using video estimate')
            top_z = top_guess
            c_xy = xy
            reg = None
        else:
            log('refine1: center (%.3f,%.3f) top %.3f (h=%.3f) size %s n=%d' % (
                reg['center_xy'][0], reg['center_xy'][1], reg['top_z'], reg['top_z'] - tz,
                np.round(reg['size'], 3).tolist(), reg['n']))
            c_xy, top_z = reg['center_xy'], reg['top_z']
        # 3. closer view, camera centred over the container
        v2 = self.view_pose(c_xy, top_z, 0.15)
        robot.move_to(v2, R_down, vmax=0.03, tol=0.004, max_rows=50)
        reg2, pts2 = self.refine(c_xy, top_z - 0.01)
        if reg2 is not None and np.linalg.norm(reg2['center_xy'] - c_xy) < 0.03:
            log('refine2: center (%.3f,%.3f) top %.3f size %s yaw %.1f n=%d' % (
                reg2['center_xy'][0], reg2['center_xy'][1], reg2['top_z'], np.round(reg2['size'], 3).tolist(),
                math.degrees(reg2['yaw']), reg2['n']))
            reg, pts = reg2, pts2
            c_xy, top_z = reg2['center_xy'], reg2['top_z']
        elif reg2 is not None:
            log('refine2 centre (%.3f,%.3f) disagrees; keeping refine1' % tuple(reg2['center_xy']))
        # 4. align yaw
        dyaw = self.plan_yaw(reg, pts, robot.R)
        R_g = rotz(dyaw) @ robot.R
        # keep gripper pointing down: rebuild from R0 with yaw only
        R_g = rotz(self.yaw_ee(R_g) - self.yaw_ee(self.R0)) @ self.R0
        width = max(reg['size']) if reg is not None and reg.get('size') else 0.05
        g_open = self.grip_cmd_for(min(0.08, width + 0.03))
        robot.grip_cmd = g_open
        robot.move_to([c_xy[0], c_xy[1], top_z + 0.05], R_g, vmax=0.02, tol=0.003, rot_tol=0.02, max_rows=50)
        robot.hold(2)
        self.grip_map.append((g_open, robot.opening))
        log('open cmd %.2f -> opening %.4f' % (g_open, robot.opening))
        if robot.opening < min(0.075, width + 0.015) and g_open > -1.0:
            robot.gripper(-1.0, n=5)
            log('opening too small; fully open -> %.4f' % robot.opening)
        # 5. descend & grasp
        height = top_z - tz
        grasp_z = max(top_z - min(CFG['grasp_below_top'], 0.5 * height), tz + CFG['min_tip_h'])
        e = robot.move_to([c_xy[0], c_xy[1], grasp_z], R_g, vmax=0.012, tol=0.003, max_rows=40)
        log('at grasp z %.3f (err %.4f) pos %s' % (grasp_z, e, np.round(robot.pos, 3).tolist()))
        robot.gripper(1.0, n=10)
        op_a = robot.opening
        robot.hold(2)
        op = robot.opening
        log('closed: opening %.4f (%.4f two steps earlier)' % (op, op_a))
        if op < 0.004:
            log('grasp missed (opening %.4f)' % op)
            return False, c_xy, grasp_z
        # 6. lift
        lift_z = grasp_z + CFG['lift_h']
        robot.move_to([c_xy[0], c_xy[1], lift_z], R_g, vmax=0.015, tol=0.006, max_rows=40)
        robot.hold(3)
        op2 = robot.opening
        log('lifted to %.3f, opening %.4f, complete=%s ended=%s' % (robot.pos[2], op2, robot.complete, robot.ended))
        if op2 < 0.004:
            log('lost the container during lift')
            return False, c_xy, grasp_z
        return True, c_xy, grasp_z

    def put_back(self, c_xy, grasp_z):
        robot = self.robot
        robot.move_to([c_xy[0], c_xy[1], grasp_z + 0.006], None, vmax=0.015, tol=0.004, max_rows=40)
        robot.gripper(-1.0, n=7)
        tz = self.table_z if self.table_z is not None else grasp_z - 0.03
        robot.move_to([robot.pos[0], robot.pos[1], max(grasp_z + 0.08, tz + 0.15)], None, vmax=0.03, tol=0.01, max_rows=20)

    def retreat_after_miss(self):
        robot = self.robot
        robot.gripper(-1.0, n=6)
        robot.move_to(robot.pos + np.array([0, 0, 0.07]), None, vmax=0.03, tol=0.01, max_rows=15)

    def run(self):
        self.setup()
        targets = self.find_targets()
        seq = []
        for c, mode, group in targets:
            seq.append((c, mode, group))
        total = sum(len(g) for _, _, g in seq)
        log = self.log
        log('total picks planned: %d' % total)
        if total == 0:
            log('nothing to do')
            return
        done = 0
        pos = self.robot.pos
        for gi, (c, mode, group) in enumerate(seq):
            rem = list(group)
            while rem:
                # nearest first within a colour group
                ki = int(np.argmin([np.linalg.norm(kk['center'][:2] - self.robot.pos[:2]) for kk in rem]))
                k = rem.pop(ki)
                done += 1
                last = (done == total)
                ok = False
                for attempt in range(3):
                    if time.time() - T0 > CFG['wall_limit_s']:
                        log('wall limit reached')
                        return
                    ok, c_xy, gz = self.pick(k['center'], last, '%s#%d try%d' % (c, done, attempt))
                    if ok:
                        break
                    self.retreat_after_miss()
                    k = dict(k)
                    k['center'] = np.array([c_xy[0], c_xy[1], k['center'][2]])
                if not ok:
                    log('giving up on %s after retries' % c)
                    continue
                self.picked.append((c, np.round(c_xy, 3).tolist()))
                if self.robot.complete or self.robot.ended:
                    log('episode reports complete=%s ended=%s' % (self.robot.complete, self.robot.ended))
                    return
                if not last:
                    self.put_back(c_xy, gz)
        # final: keep holding the last container
        try:
            self.robot.hold(10)
        except EpisodeEnded:
            pass
        log('finished: picked %s complete=%s ended=%s' % (self.picked, self.robot.complete, self.robot.ended))


def find_reset_manifest(log):
    import glob
    cands = []
    for pat in ('/workspace/runtime/autonomous_controller/runs/*/frames/**/*.json',
                '/workspace/runtime/**/manifest*.json'):
        try:
            cands += glob.glob(pat, recursive=True)
        except Exception:
            pass
    best = None
    for p in sorted(set(cands), key=lambda q: -os.path.getmtime(q))[:200]:
        try:
            if os.path.getsize(p) > 50e6:
                continue
            m = iu.load_manifest(p)
            fr = m.get('frames') if isinstance(m, dict) else None
            if fr and any(str(f.get('phase')) == 'video_demo' for f in fr if isinstance(f, dict)) and \
                    any(isinstance(f, dict) and f.get('files') for f in fr):
                best = p
                break
        except Exception:
            continue
    log('manifest fallback search -> %s' % best)
    return best


def depth_from_blocks(reply, meta, log):
    """Fallback: decode 16-bit depth image attachments from an observe reply."""
    out = {}
    try:
        imgs = iu.reply_images(reply)
        kinds = []
        for b in imgs:
            k, arr = iu.decode_image_block(b)
            kinds.append(k)
            if k == 'depth':
                cam = 'front' if 'front' not in out else 'wrist'
                out[cam] = arr.astype(np.float64) * 0.001
        log('image blocks: %s' % kinds)
    except Exception as e:
        log('depth_from_blocks failed %r' % (e,))
    return out


def main(ctx):
    task = None
    try:
        task = Task(ctx)
        task.run()
    except EpisodeEnded:
        if task:
            task.log('episode ended (complete=%s)' % task.robot.complete)
    except Exception as e:
        msg = traceback.format_exc()
        if task:
            task.log('ERROR %s' % msg)
        else:
            print(msg)
    finally:
        if task:
            task.log('main done; steps=%s calls=%d wall=%.1fs' % (task.robot.step_id, task.robot.n_calls, time.time() - T0))
