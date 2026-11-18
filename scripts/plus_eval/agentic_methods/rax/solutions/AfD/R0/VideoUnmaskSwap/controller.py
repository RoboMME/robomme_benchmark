"""VideoUnmaskSwap controller (RAX, RoboMME-Plus).

Pipeline (step -> exits):
  S0 observe      -> instruction, conditioning-video manifest, calibration          (fail: abort)
  S1 geometry     -> select camera conventions by matching robomme_pixel_to_position (fail: 2D mode)
  S2 video track  -> cube colours at cover time, container tracks to the end of video (fail: 2D mode)
  S3 exec track   -> hold pose, step 1 tick at a time, keep tracking until shuffle stops (timeout: proceed)
  S4 picks        -> for each instructed colour: relocalise, align yaw, descend, close,
                     verify fingers, lift, (lower + release unless last)                 (miss: retry x2)
  S5 hold         -> keep the last container lifted until the episode ends or budget used
"""
import json
import math
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import geom  # noqa: E402
import obsio  # noqa: E402
import tracker as T2  # noqa: E402
import world  # noqa: E402

VERSION = 'vus-v1'
OUT = '/workspace/output'

# ---- tunables (untested live: AfD, zero simulator starts) ----
HOLD_STABLE_STEPS = 10        # consecutive still steps to call the shuffle finished
EXEC_TRACK_MAX = 220          # max steps waiting for the shuffle to stop
STILL_MOVE_M = 0.0008         # max per-step track motion counted as still
STILL_PIX = 60                # max changed front pixels per step counted as still
SIDE_DEFAULT = 0.045
H_DEFAULT = 0.05
VMAX = 0.5                    # action units (x0.05 m) for transit
VSLOW = 0.2
APPROACH_ABOVE = 0.09         # m above container top for approach
LIFT = 0.08                   # m lift
CLOSE_STEPS = 12
OPEN_STEPS = 8
FINGER_MIN = 0.006            # opening sum (m) below which the grasp missed
COLORS = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'black', 'white', 'pink', 'brown', 'cyan', 'gray']


class EpisodeEnded(Exception):
    pass


class Log:
    def __init__(self):
        os.makedirs(OUT, exist_ok=True)
        self.f = open(os.path.join(OUT, 'vus_log.txt'), 'a')
        self.t0 = time.time()

    def __call__(self, *a):
        s = '[%7.1fs] ' % (time.time() - self.t0) + ' '.join(str(x) for x in a)
        try:
            print(s, flush=True)
        except Exception:
            pass
        try:
            self.f.write(s + '\n'); self.f.flush()
        except Exception:
            pass


LOG = None


def log(*a):
    if LOG:
        LOG(*a)


# ======================================================================= instruction
def parse_instruction(text):
    """-> list of (color, all_flag) in order."""
    t = (text or '').lower().replace(';', ',')
    for sep in ['followed by', ' then ', 'and finally', 'finally', 'after that', ' and ']:
        t = t.replace(sep, ',')
    out = []
    for frag in t.split(','):
        import re
        cols = [(m.start(), c) for c in COLORS for m in [re.search(r'\b' + c + r'\b', frag)] if m]
        if not cols:
            continue
        cols.sort()
        c = cols[0][1]
        allf = (' all ' in ' ' + frag + ' ') or ('containers' in frag) or ((c + ' cubes') in frag)
        out.append((c, allf))
    return out


# ======================================================================= robot I/O
class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = {}
        self.state = None
        self.steps = 0
        self.limit = None
        self.ended = False
        self.success = False
        self.grip = -1.0

    def _absorb(self, meta):
        self.meta = meta
        s = obsio.get_state(meta)
        if s is not None:
            self.state = s
        e, ok = obsio.ended(meta)
        lim = meta.get('episode_limits')
        if isinstance(lim, dict):
            self.limit = lim
        if ok:
            self.success = True
        if e:
            self.ended = True

    def observe(self):
        rep = self.ctx.call('robomme_observe')
        meta = obsio.reply_meta(rep)
        self._absorb(meta)
        return rep, meta

    def remaining(self):
        lim = self.limit or {}
        for k in ('remaining_steps', 'steps_remaining', 'remaining'):
            v = lim.get(k)
            if isinstance(v, (int, float)):
                return int(v)
        mx = lim.get('max_steps') or lim.get('step_limit') or lim.get('max_episode_steps')
        cur = lim.get('step') or lim.get('elapsed_steps') or lim.get('steps')
        if isinstance(mx, (int, float)) and isinstance(cur, (int, float)):
            return int(mx - cur)
        return None

    def step(self, rows):
        if self.ended:
            raise EpisodeEnded()
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows]
        rem = self.remaining()
        if rem is not None:
            if rem <= 0:
                self.ended = True
                raise EpisodeEnded()
            rows = rows[:max(1, rem)]
        rep = self.ctx.call('robomme_step', actions=rows)
        meta = obsio.reply_meta(rep)
        self.steps += len(rows)
        self._absorb(meta)
        if obsio.get_state(meta) is None:
            try:
                self.observe()
            except Exception as e:
                log('observe after step failed', e)
        if self.ended:
            log('episode ended; success flag', self.success)
            raise EpisodeEnded()
        return rep, meta

    # ---- pose helpers
    def pos(self):
        return self.state[:3].copy()

    def R(self):
        return geom.rotvec_to_R(self.state[3:6])

    def fingers(self):
        return float(self.state[6] + self.state[7])

    def move_to(self, target, R_des, grip, vmax=VMAX, tol=0.004, ang_tol=0.03, max_steps=120, chunk=4):
        """Closed-loop move of the TCP to `target` with orientation `R_des`."""
        target = np.asarray(target, float)
        used = 0
        best = 1e9
        stall = 0
        while used < max_steps:
            p = self.pos()
            err = target - p
            dist = float(np.linalg.norm(err))
            erot = geom.R_to_rotvec(R_des @ self.R().T)
            ang = float(np.linalg.norm(erot))
            if dist < tol and ang < ang_tol:
                return True
            if dist < best - 0.001:
                best = dist; stall = 0
            else:
                stall += 1
            if stall > 12:
                log('move_to stall', np.round(p, 3), 'target', np.round(target, 3), 'dist', round(dist, 4))
                return False
            n = int(max(1, min(chunk, math.ceil(max(dist / (0.05 * vmax), ang / (0.5 * 0.4))))))
            dp = err / n / 0.05
            dr = erot / n / 0.5
            # cap per-row magnitude
            sp = np.linalg.norm(dp)
            if sp > vmax:
                dp = dp * vmax / sp
            rn = np.linalg.norm(dr)
            if rn > 0.4:
                dr = dr * 0.4 / rn
            row = list(dp) + list(dr) + [grip]
            self.step([row] * n)
            used += n
        return False

    def hold(self, n, grip, p_hold=None, R_hold=None):
        p_hold = self.pos() if p_hold is None else p_hold
        R_hold = self.R() if R_hold is None else R_hold
        for _ in range(n):
            err = (p_hold - self.pos()) / 0.05
            er = geom.R_to_rotvec(R_hold @ self.R().T) / 0.5
            self.step([list(np.clip(err, -0.3, 0.3)) + list(np.clip(er, -0.3, 0.3)) + [grip]])


# ======================================================================= geometry selection
def extract_points(rep_meta, n):
    """Parse robomme_pixel_to_position reply -> list of (xyz or None)."""
    keys = ('position', 'world_position', 'world', 'point', 'xyz', 'position_m', 'world_point', 'world_xyz', 'point_world')
    hits = obsio.find_all(rep_meta, lambda d: any(isinstance(d.get(k), list) and len(d.get(k)) == 3 for k in keys))
    out = []
    for h in hits:
        v = None
        for k in keys:
            if isinstance(h.get(k), list) and len(h[k]) == 3:
                v = h[k]
                break
        valid = h.get('valid', True)
        out.append(np.array(v, float) if (v is not None and valid is not False) else None)
    if len(out) != n:
        # maybe a list of lists
        lst = obsio.find_all(rep_meta, lambda d: False)
        cand = None
        for k in ('points', 'positions', 'results', 'world_points'):
            v = geom.find_key(rep_meta, [k])
            if isinstance(v, list) and len(v) == n:
                cand = v
                break
        if cand is not None:
            out = []
            for v in cand:
                if isinstance(v, list) and len(v) == 3:
                    out.append(np.array(v, float))
                elif isinstance(v, dict):
                    vv = None
                    for k in keys:
                        if isinstance(v.get(k), list) and len(v[k]) == 3:
                            vv = np.array(v[k], float)
                    out.append(vv)
                else:
                    out.append(None)
    return out


def select_cameras(ctx, meta, frame):
    """Return dict cam -> geom.Camera chosen by agreement with the simulator's back-projection."""
    intr = geom.find_key(meta, ['camera_intrinsics', 'intrinsics'])
    poses = geom.find_key(meta, ['camera_poses', 'extrinsics', 'camera_pose'])
    sid = obsio.get_step_id(meta)
    chosen = {}
    report = {}
    for cam in ('front', 'wrist'):
        K = geom.parse_intrinsics(geom.per_camera(intr, cam))
        pobj = geom.per_camera(poses, cam)
        depth = frame.get(cam + '_depth')
        if K is None or pobj is None or depth is None:
            report[cam] = 'missing K/pose/depth: %s %s %s' % (K is not None, pobj is not None, depth is not None)
            continue
        cands = geom.camera_candidates(K, pobj)
        if not cands:
            report[cam] = 'no pose candidates'
            continue
        H, W = depth.shape
        # pick sample pixels with valid depth on a grid
        sel = []
        for v in range(int(H * 0.15), int(H * 0.95), int(H / 6)):
            for u in range(int(W * 0.1), int(W * 0.95), int(W / 5)):
                if depth[v, u] > 0.01:
                    sel.append((u, v))
        sel = sel[:16]
        truth = []
        if sid is not None and sel:
            try:
                rep = ctx.call('robomme_pixel_to_position', observation_step_id=int(sid), depth_window_radius=0,
                               selections=[{'camera': cam, 'u': float(u), 'v': float(v)} for u, v in sel])
                rm = obsio.reply_meta(rep)
                truth = extract_points(rm, len(sel))
                if len(truth) != len(sel):
                    log('pixel_to_position parse mismatch', cam, len(truth), len(sel), json.dumps(rm)[:600])
                    truth = []
            except Exception as e:
                log('pixel_to_position failed', cam, e)
        best = None
        for name, c in cands:
            if truth:
                errs = []
                for (u, v), tp in zip(sel, truth):
                    if tp is None:
                        continue
                    for off in (0.0, 0.5):
                        pass
                    P = c.backproject(np.array([u]), np.array([v]), np.array([depth[v, u]]))[0]
                    P2 = c.backproject(np.array([u + 0.5]), np.array([v + 0.5]), np.array([depth[v, u]]))[0]
                    errs.append(min(np.linalg.norm(P - tp), np.linalg.norm(P2 - tp)))
                score = float(np.median(errs)) if errs else 9.0
            else:
                # plausibility: table should be a horizontal plane below the camera
                st = world.plane_stats(world.table_points(c, frame[cam + '_rgb'], depth))
                score = 9.0 if st is None else st['mad'] + st['tilt'] + (0 if c.T[2, 3] > st['z'] else 1.0)
            if best is None or score < best[0]:
                best = (score, name, c)
        thr = 0.012 if truth else 0.02
        report[cam] = (round(best[0], 4), best[1], 'truth' if truth else 'plausibility')
        if best[0] < thr:
            best[2]._name = best[1]
            chosen[cam] = best[2]
    log('camera selection', report)
    return chosen


def derive_wrist_video_cam(cam_exec, state_exec, state_video):
    """Wrist camera pose during the conditioning video from the EE poses (rigid mount)."""
    Te_x = geom.T_from(geom.rotvec_to_R(state_exec[3:6]), state_exec[:3])
    Te_v = geom.T_from(geom.rotvec_to_R(state_video[3:6]), state_video[:3])
    Tc = Te_v @ np.linalg.inv(Te_x) @ cam_exec.T
    return geom.Camera(cam_exec.K, Tc, cam_exec.conv)


# ======================================================================= perception helpers
class Geo:
    def __init__(self):
        self.cams = {}
        self.table_z = None
        self.side = SIDE_DEFAULT
        self.H = H_DEFAULT
        self.static_cells = set()
        self.state0 = None

    def zband(self, P, lo=0.008, hi=None):
        hi = (self.H + 0.025) if hi is None else hi
        z = P[:, 2] - self.table_z
        return (z > lo) & (z < hi)

    def container_xy(self, frame, bgs=None, cams=None, exclude_static=False):
        pts = []
        for cam in ('front', 'wrist'):
            c = (cams or self.cams).get(cam)
            rgb = frame.get(cam + '_rgb'); depth = frame.get(cam + '_depth')
            if c is None or rgb is None or depth is None:
                continue
            m = world.lowsat_bright(rgb)
            if bgs is not None and bgs.get(cam) is not None:
                m &= world.changed(rgb, bgs[cam])
            P, _, _ = world.pix_to_world(c, depth, m, stride=1 if cam == 'front' else 2)
            if len(P) == 0:
                continue
            P = P[self.zband(P)]
            pts.append(P)
        if not pts:
            return np.zeros((0, 2)), np.zeros((0, 3))
        P = np.concatenate(pts)
        if exclude_static and self.static_cells and len(P):
            ij = np.floor(P[:, :2] / world.CELL).astype(np.int64)
            keep = np.array([(int(a), int(b)) not in self.static_cells for a, b in ij], bool)
            P = P[keep]
        return P[:, :2], P


def detect_cubes_world(geo, frame, bgs, cams):
    pts, cols = [], []
    for cam in ('front', 'wrist'):
        c = cams.get(cam)
        rgb = frame.get(cam + '_rgb'); depth = frame.get(cam + '_depth')
        if c is None or rgb is None or depth is None or bgs.get(cam) is None:
            continue
        m = world.changed(rgb, bgs[cam])
        P, C, _ = world.pix_to_world(c, depth, m, stride=1, rgb=rgb)
        if len(P) == 0:
            continue
        k = geo.zband(P, lo=0.004, hi=0.06)
        P, C = P[k], C[k]
        w = np.full(len(P), 1.0 / max(1, len(P)))   # each view gets equal total weight
        pts.append(np.c_[P, w]); cols.append(C)
    if not pts:
        return []
    A = np.concatenate(pts); C = np.concatenate(cols)
    clusters = world.grid_clusters(A[:, :2], cell=0.005, min_pts=6, link=1)
    cubes = []
    for idx in clusters:
        XY = A[idx, :2]
        ext = XY.max(0) - XY.min(0)
        if max(ext) > 0.08:
            # probably two cubes merged; still keep as one (logged)
            log('cube cluster large', np.round(ext, 3))
        col, tally = world.color_vote(C[idx], A[idx, 3])
        if col == 'gray':
            col = 'white'
        cubes.append(dict(xy=XY.mean(0), n=len(idx), color=col, tally={k: round(v, 3) for k, v in tally.items()}))
    # drop tiny clusters relative to the others
    if cubes:
        mx = max(q['n'] for q in cubes)
        cubes = [q for q in cubes if q['n'] >= max(6, 0.08 * mx)]
    return cubes


# ======================================================================= main pipeline
def load_video(meta):
    seq = meta.get('frame_sequence') or geom.find_key(meta, ['frame_sequence'])
    man, mdir = obsio.load_manifest(seq) if seq else (None, None)
    if not man:
        return [], None, None
    frames = sorted(man.get('frames', []), key=lambda f: f.get('frame_index', 0))
    vid = [f for f in frames if f.get('phase') == 'video_demo']
    ex = [f for f in frames if f.get('phase') != 'video_demo']
    return vid, (ex[-1] if ex else None), mdir


def frame_arrays(rec, mdir, want=('front_rgb', 'front_depth', 'wrist_rgb', 'wrist_depth')):
    fr = obsio.files_to_frame([f for f in rec.get('files', []) if (str(f.get('camera')).lower() + '_' + str(f.get('kind')).lower()) in want or True], mdir)
    return fr


def run(ctx):
    global LOG
    LOG = Log()
    log(VERSION, 'start')
    rob = Robot(ctx)
    rep0, meta0 = rob.observe()
    instr = geom.find_key(meta0, ['instruction', 'goal', 'language_instruction']) or ''
    plan = parse_instruction(instr)
    log('instruction:', instr, '-> plan', plan)
    log('meta keys', list(meta0.keys()))
    try:
        with open(os.path.join(OUT, 'meta0.json'), 'w') as fh:
            json.dump(meta0, fh, default=str)
    except Exception:
        pass
    vid, exrec, mdir = load_video(meta0)
    log('video frames', len(vid), 'exec rec', exrec is not None)
    frame0 = obsio.frame_from_reply(rep0, meta0)
    if exrec is not None:
        fx = obsio.files_to_frame(exrec.get('files', []), mdir)
        for k, v in fx.items():
            frame0.setdefault(k, v)
    log('exec frame0 keys', {k: v.shape for k, v in frame0.items()})
    state0 = rob.state.copy() if rob.state is not None else None
    log('state0', None if state0 is None else np.round(state0, 4).tolist())

    # ---------------- S1 geometry
    cams = {}
    try:
        cams = select_cameras(ctx, meta0, frame0)
    except Exception:
        log('select_cameras crashed', traceback.format_exc())
    geo = Geo()
    geo.cams = cams
    geo.state0 = state0
    if 'front' in cams and frame0.get('front_depth') is not None:
        st = world.plane_stats(world.table_points(cams['front'], frame0['front_rgb'], frame0['front_depth']))
        log('table from front', st)
        if st:
            geo.table_z = st['z']
    if geo.table_z is None and 'wrist' in cams:
        st = world.plane_stats(world.table_points(cams['wrist'], frame0['wrist_rgb'], frame0['wrist_depth']))
        log('table from wrist', st)
        if st:
            geo.table_z = st['z']

    # ---------------- S2 video tracking
    result = None
    if vid and geo.table_z is not None and cams:
        try:
            result = video_track_world(geo, vid, mdir, frame0, state0)
        except Exception:
            log('video_track_world crashed', traceback.format_exc())
            result = None
    mode = 'world' if result is not None else '2d'
    if result is None:
        try:
            result = video_track_2d(vid, mdir)
        except Exception:
            log('video_track_2d crashed', traceback.format_exc())
            result = None
    if result is None:
        log('no tracking result; will pick by colour-less fallback')
    log('mode', mode)

    # ---------------- S3 exec tracking
    p_hold = rob.pos() if rob.state is not None else None
    R_hold = rob.R() if rob.state is not None else None
    if result is not None:
        try:
            exec_track(rob, geo, result, mode, frame0, p_hold, R_hold)
        except EpisodeEnded:
            raise
        except Exception:
            log('exec_track crashed', traceback.format_exc())
    # ---------------- S4 picks
    targets = make_targets(plan, result)
    log('targets', [(c, np.round(xy, 4).tolist() if xy is not None else None) for c, xy in targets])
    picks(rob, geo, targets, mode, R_hold)


def video_track_world(geo, vid, mdir, frame0, state0):
    cams_v = {}
    if 'front' in geo.cams:
        cams_v['front'] = geo.cams['front']
    # wrist camera during the video
    sv = None
    for f in vid:
        s = f.get('state')
        if isinstance(s, list) and len(s) >= 8:
            sv = np.array(s[:8], float)
            break
    first = obsio.files_to_frame(vid[0].get('files', []), mdir)
    if 'wrist' in geo.cams and sv is not None and state0 is not None and first.get('wrist_depth') is not None:
        try:
            cw = derive_wrist_video_cam(geo.cams['wrist'], state0, sv)
            st = world.plane_stats(world.table_points(cw, first['wrist_rgb'], first['wrist_depth']))
            log('wrist video table', st, 'front table', geo.table_z)
            if st and abs(st['z'] - geo.table_z) < 0.015 and st['tilt'] < 0.15:
                cams_v['wrist'] = cw
        except Exception:
            log('wrist video derivation failed', traceback.format_exc())
    log('video cams', list(cams_v))
    if 'front' not in cams_v and 'wrist' not in cams_v:
        return None
    bgs = {'front': first.get('front_rgb'), 'wrist': first.get('wrist_rgb')}
    # cover index from front 2D container pixel counts (fast, RGB only)
    t_load = time.time()
    recs = []
    for f in vid:
        recs.append(f)
    counts = []
    fronts = []
    for f in recs:
        fr = obsio.files_to_frame([x for x in f.get('files', []) if str(x.get('camera')).lower().startswith('front') and x.get('kind') == 'rgb'], mdir)
        fronts.append(fr.get('front_rgb'))
    bgf = fronts[0]
    for im in fronts:
        counts.append(int(T2.container_mask(im, bgf).sum()) if im is not None and bgf is not None else 0)
    base = max(1.0, float(np.median(counts[1:6])) if len(counts) > 6 else 1.0)
    ci = None
    for i in range(1, len(counts)):
        if counts[i] > T2.COVER_JUMP * base + 150:
            ci = i
            break
    log('cover index', ci, 'of', len(recs), 'counts', counts[:3], counts[ci] if ci else None, 'load %.1fs' % (time.time() - t_load))
    if ci is None or ci < 2:
        return None
    # cube frame: two samples before cover (fall back to one)
    cube_i = max(1, ci - 2)
    frc = obsio.files_to_frame(recs[cube_i].get('files', []), mdir)
    cubes = detect_cubes_world(geo, frc, bgs, cams_v)
    log('cubes', [(q['color'], np.round(q['xy'], 3).tolist(), q['n']) for q in cubes])
    # containers at the cover frame
    frv = obsio.files_to_frame(recs[ci].get('files', []), mdir)
    XY, P = geo.container_xy(frv, bgs, cams_v)
    if len(P):
        top = np.percentile(P[:, 2], 97) - geo.table_z
        if 0.02 < top < 0.15:
            geo.H = float(top)
    XY, P = geo.container_xy(frv, bgs, cams_v)
    cl = world.grid_clusters(XY, cell=0.005, min_pts=20, link=1)
    sides = []
    ccent = []
    for idx in cl:
        a, ext = world.square_fit(XY[idx])
        ccent.append(XY[idx].mean(0))
        if 0.02 < max(ext) < 0.09 and min(ext) > 0.6 * max(ext):
            sides.append(max(ext))
    if sides:
        geo.side = float(np.clip(np.median(sides), 0.025, 0.08))
    log('container H %.3f side %.3f clusters %d cubes %d' % (geo.H, geo.side, len(cl), len(cubes)))
    if not cubes:
        return None
    centers = [q['xy'] for q in cubes]
    labels = [q['color'] for q in cubes]
    # extra container clusters not explained by any cube -> unknown tracks
    if ccent:
        C = np.array(ccent)
        Q = np.array(centers)
        d = np.linalg.norm(C[:, None] - Q[None], axis=2)
        for k in range(len(C)):
            if d[k].min() > 0.8 * geo.side:
                centers.append(C[k]); labels.append('unknown')
    tr = world.XYTracker(centers, geo.side)
    t0 = float(recs[ci].get('timestamp_s', 0.0))
    for _ in range(3):
        tr.t = t0
        tr.update(XY, t0)
    tr.v[:] = 0
    tl = time.time()
    for f in recs[ci + 1:]:
        fr = obsio.files_to_frame(f.get('files', []), mdir)
        XY, _ = geo.container_xy(fr, bgs, cams_v)
        tr.update(XY, float(f.get('timestamp_s', tr.t + 0.075)))
    log('video tracked in %.1fs; lost' % (time.time() - tl), tr.lost.tolist(), 'final', np.round(tr.c, 3).tolist(), labels)
    # static (tall) objects for the execution phase
    try:
        build_static(geo, first, frame0)
    except Exception:
        log('build_static failed', traceback.format_exc())
    return dict(tracker=tr, labels=labels, t_end=tr.t, cubes=cubes)


def build_static(geo, bg_frame, frame0):
    c = geo.cams.get('front')
    if c is None or frame0.get('front_depth') is None:
        return
    bg, ex = bg_frame.get('front_rgb'), frame0.get('front_rgb')
    m = world.lowsat_bright(bg) & world.lowsat_bright(ex) & ~world.changed(ex, bg)
    P, _, _ = world.pix_to_world(c, frame0['front_depth'], m)
    if len(P) == 0:
        return
    z = P[:, 2] - geo.table_z
    P = P[z > 0.008]
    cells = {}
    for (x, y, zz) in P:
        k = (int(math.floor(x / world.CELL)), int(math.floor(y / world.CELL)))
        cells[k] = max(cells.get(k, -1), zz - geo.table_z)
    tall = [k for k, v in cells.items() if v > geo.H + 0.03]
    s = set()
    for (i, j) in tall:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                s.add((i + di, j + dj))
    geo.static_cells = s
    log('static tall cells', len(tall))


def video_track_2d(vid, mdir):
    if not vid:
        return None
    frames = []
    for f in vid:
        fr = obsio.files_to_frame([x for x in f.get('files', []) if str(x.get('camera')).lower().startswith('front') and x.get('kind') == 'rgb'], mdir)
        if fr.get('front_rgb') is not None:
            frames.append((float(f.get('timestamp_s', 0)), fr['front_rgb']))
    r = T2.track(frames)
    log('2d track: cover', r['cover_index'], 'colors', r['colors'], 'final', np.round(r['final'], 1).tolist())
    r['labels'] = r['colors']
    r['bg'] = frames[0][1]
    return r


def exec_track(rob, geo, result, mode, frame0, p_hold, R_hold):
    tr = result['tracker']
    labels = result['labels']
    t = result.get('t_end', 0.0) or 0.0
    static2d = None
    if mode == 'world':
        XY, _ = geo.container_xy(frame0, None, geo.cams, exclude_static=True)
        t += 0.05
        tr.update(XY, t)
    else:
        static2d = T2.static_mask(result['bg'], frame0['front_rgb'], tr.c, result['size'])
        tr.update(T2.exec_mask(frame0['front_rgb'], static2d))
    prev_front = frame0.get('front_rgb')
    still = 0
    prev_c = tr.c.copy()
    for k in range(EXEC_TRACK_MAX):
        # hold the pose (closed loop on the measured TCP)
        if p_hold is not None and rob.state is not None:
            err = np.clip((p_hold - rob.pos()) / 0.05, -0.2, 0.2)
            er = np.clip(geom.R_to_rotvec(R_hold @ rob.R().T) / 0.5, -0.2, 0.2)
            row = list(err) + list(er) + [-1.0]
        else:
            row = [0, 0, 0, 0, 0, 0, -1.0]
        rep, meta = rob.step([row])
        fr = obsio.frame_from_reply(rep, meta, allow_manifest=False)
        if fr.get('front_rgb') is None or fr.get('wrist_depth') is None or fr.get('front_depth') is None:
            rep2, meta2 = rob.observe()
            fr2 = obsio.frame_from_reply(rep2, meta2)
            for kk, v in fr2.items():
                fr.setdefault(kk, v)
        t += 0.05
        if mode == 'world':
            XY, _ = geo.container_xy(fr, None, geo.cams, exclude_static=True)
            if len(XY) == 0 and fr.get('wrist_rgb') is not None and 'wrist' in geo.cams:
                # no depth: intersect top-surface pixels with the container-top plane
                m = world.lowsat_bright(fr['wrist_rgb'])
                vs, us = np.nonzero(m)
                if len(vs):
                    P = geo.cams['wrist'].ray_plane(us[::2].astype(float), vs[::2].astype(float), geo.table_z + geo.H)
                    XY = P[:, :2]
            tr.update(XY, t)
        else:
            tr.update(T2.exec_mask(fr['front_rgb'], static2d))
        mv = float(np.abs(tr.c - prev_c).max())
        prev_c = tr.c.copy()
        pix = 9999
        if fr.get('front_rgb') is not None and prev_front is not None:
            pix = int(world.changed(fr['front_rgb'], prev_front).sum())
            prev_front = fr['front_rgb']
        thr_mv = STILL_MOVE_M if mode == 'world' else 0.3
        if (mv < thr_mv and pix < STILL_PIX) or pix < 4:
            still += 1
        else:
            still = 0
        if k % 10 == 0:
            log('exec track step', k, 'move %.4f' % mv, 'pix', pix, 'still', still)
        if still >= HOLD_STABLE_STEPS:
            log('shuffle finished after', k + 1, 'steps')
            break
    result['final'] = tr.c.copy()
    log('final tracks', [(labels[i], np.round(tr.c[i], 4).tolist()) for i in range(len(labels))])
    if mode == '2d':
        # convert front pixels to world with the simulator's back-projection
        result['final_px'] = tr.c.copy()
        result['final'] = pixels_to_world(rob, tr.c)


def pixels_to_world(rob, px):
    sid = obsio.get_step_id(rob.meta)
    out = []
    try:
        sel = [{'camera': 'front', 'u': float(x), 'v': float(y)} for x, y in px]
        rep = rob.ctx.call('robomme_pixel_to_position', observation_step_id=int(sid), depth_window_radius=1, selections=sel[:32])
        pts = extract_points(obsio.reply_meta(rep), len(sel))
        for p in pts:
            out.append(None if p is None else np.array(p, float))
    except Exception as e:
        log('pixels_to_world failed', e)
        out = [None] * len(px)
    return out


def make_targets(plan, result):
    if result is None:
        return []
    labels = result['labels']
    final = result['final']
    used = set()
    targets = []
    for color, allf in plan:
        idx = [i for i, l in enumerate(labels) if l == color and i not in used]
        if not idx:
            # fall back to unknown-labelled containers
            idx = [i for i, l in enumerate(labels) if l == 'unknown' and i not in used][:1]
        if not allf and len(idx) > 1:
            cubes = result.get('cubes') or []

            def conf(i):
                if i < len(cubes) and isinstance(cubes[i].get('tally'), dict):
                    t = cubes[i]['tally']
                    return t.get(color, 0.0) / max(1e-9, sum(t.values()))
                return 0.0
            idx = sorted(idx, key=conf, reverse=True)[:1]
        elif not allf:
            idx = idx[:1]
        for i in idx:
            used.add(i)
            xy = final[i]
            targets.append((color, None if xy is None else np.asarray(xy, float)))
    return targets


# ======================================================================= picking
def relocalize(rob, geo, xy_guess, others):
    """From the current pose, refine target XY, top z and yaw using the wrist (and front) depth."""
    rep, meta = rob.observe()
    fr = obsio.frame_from_reply(rep, meta)
    cams = dict(geo.cams)
    if 'wrist' in geo.cams and geo.state0 is not None and rob.state is not None:
        cams['wrist'] = derive_wrist_video_cam(geo.cams['wrist'], geo.state0, rob.state)
    XY, P = geo.container_xy(fr, None, cams, exclude_static=True)
    if len(XY) == 0:
        return xy_guess, None, None
    d = np.linalg.norm(XY - xy_guess[None], axis=1)
    sel = d < 0.8 * geo.side
    if sel.sum() < 15:
        return xy_guess, None, None
    # iterate mean-shift once more around the guess, excluding points nearer other containers
    c = XY[sel].mean(0)
    for _ in range(3):
        d = np.linalg.norm(XY - c[None], axis=1)
        sel = d < 0.75 * geo.side
        for o in others:
            sel &= np.linalg.norm(XY - o[None], axis=1) > d
        if sel.sum() < 15:
            break
        c = XY[sel].mean(0)
    Ps = P[sel]
    top = float(np.percentile(Ps[:, 2], 95))
    topm = Ps[:, 2] > top - 0.01
    yaw, ext = world.square_fit(Ps[topm, :2]) if topm.sum() > 10 else (None, None)
    if np.linalg.norm(c - xy_guess) > 0.6 * geo.side:
        log('relocalize jump too large', np.round(c, 3), np.round(xy_guess, 3))
        return xy_guess, top, yaw
    return c, top, yaw


def finger_axis_yaw(geo, R0, Rcur):
    """World yaw of the finger closing axis (wrist camera image x-axis)."""
    c = geo.cams.get('wrist')
    if c is None:
        return None
    ax = c.axis_world(0)        # at exec start orientation R0
    ax = Rcur @ R0.T @ ax
    return math.atan2(ax[1], ax[0])


def wrap(a, period):
    return (a + period / 2) % period - period / 2


def choose_yaw_delta(geo, cont_yaw, axis_yaw, xy, others):
    """Rotation about world z aligning the finger axis with a container side; pick the freer side."""
    base = wrap(cont_yaw - axis_yaw, math.pi / 2)
    best = None
    for delta in (base, base + math.pi / 2 if base < 0 else base - math.pi / 2):
        a = axis_yaw + delta
        u = np.array([math.cos(a), math.sin(a)])
        clear = 1.0
        for o in others:
            r = o - xy
            along = abs(float(r @ u)); perp = abs(float(r @ np.array([-u[1], u[0]])))
            if perp < geo.side:
                clear = min(clear, along)
        cost = -clear + 0.02 * abs(delta)
        if best is None or cost < best[0]:
            best = (cost, delta, clear)
    return best[1], best[2]


def picks(rob, geo, targets, mode, R0):
    if rob.state is None:
        log('no state; cannot pick')
        return
    R0 = rob.R() if R0 is None else R0
    if geo.table_z is None:
        # crude: use the targets' z if available
        zs = [xy[2] for _, xy in targets if xy is not None and len(xy) == 3]
        geo.table_z = (min(zs) - geo.H) if zs else rob.pos()[2] - 0.3
    z_high = geo.table_z + geo.H + APPROACH_ABOVE + 0.03
    all_xy = [np.asarray(xy[:2], float) for _, xy in targets if xy is not None]
    for n, (color, xy) in enumerate(targets):
        last = (n == len(targets) - 1)
        if xy is None:
            log('target', color, 'has no position; skip')
            continue
        top_hint = float(xy[2]) if len(xy) == 3 else None
        xy = np.asarray(xy[:2], float)
        others = [o for o in all_xy if np.linalg.norm(o - xy) > 1e-6]
        ok = False
        for attempt in range(3):
            try:
                ok = pick_one(rob, geo, color, xy, others, last, R0, z_high, top_hint, attempt)
            except EpisodeEnded:
                raise
            except Exception:
                log('pick_one crashed', traceback.format_exc())
                ok = False
            if ok:
                break
        log('pick', color, 'ok' if ok else 'FAILED')
    # S5 hold the final container up
    log('all picks done; holding')
    p, R = rob.pos(), rob.R()
    for _ in range(8):
        err = list(np.clip((p - rob.pos()) / 0.05, -0.2, 0.2)) + [0.0, 0.0, 0.0] + [rob.grip]
        rob.step([err] * 5)


def pick_one(rob, geo, color, xy, others, last, R0, z_high, top_hint, attempt):
    log('pick', color, 'attempt', attempt, 'xy', np.round(xy, 4).tolist())
    # 1) go above, open
    rob.grip = -1.0
    p = rob.pos()
    if p[2] < z_high - 0.01:
        rob.move_to([p[0], p[1], z_high], rob.R(), -1.0, vmax=VMAX, max_steps=40)
    rob.move_to([xy[0], xy[1], z_high], R0, -1.0, vmax=VMAX, max_steps=80)
    # 2) relocalise from above
    top, yaw = None, None
    if geo.cams:
        try:
            xy2, top, yaw = relocalize(rob, geo, xy, others)
            log('relocalized', np.round(xy2, 4).tolist(), 'top', None if top is None else round(top - geo.table_z, 4),
                'yaw', None if yaw is None else round(math.degrees(yaw), 1))
            if np.linalg.norm(xy2 - xy) > 0.003:
                xy = xy2
                rob.move_to([xy[0], xy[1], z_high], R0, -1.0, vmax=VSLOW, max_steps=30)
        except EpisodeEnded:
            raise
        except Exception:
            log('relocalize failed', traceback.format_exc())
    if top is None:
        top = top_hint if top_hint is not None else geo.table_z + geo.H
    Hc = max(0.015, top - geo.table_z)
    # 3) yaw alignment
    R_des = R0
    ax = finger_axis_yaw(geo, R0, R0)
    if yaw is not None and ax is not None:
        delta, clear = choose_yaw_delta(geo, yaw, ax, xy, others)
        if attempt == 2:
            delta = delta + math.pi / 2 if delta < 0 else delta - math.pi / 2
        log('yaw: container %.1f finger %.1f delta %.1f clear %.3f' % (math.degrees(yaw), math.degrees(ax), math.degrees(delta), clear))
        R_des = geom.Rz(delta) @ R0
    rob.move_to([xy[0], xy[1], z_high], R_des, -1.0, vmax=VSLOW, max_steps=40, ang_tol=0.03)
    # 4) descend
    grasp_depth = float(np.clip(0.45 * Hc, 0.012, 0.025)) + (0.004 if attempt == 1 else 0.0)
    z_g = top - grasp_depth
    rob.move_to([xy[0], xy[1], top + 0.03], R_des, -1.0, vmax=VMAX, max_steps=40)
    rob.move_to([xy[0], xy[1], z_g], R_des, -1.0, vmax=VSLOW, tol=0.003, max_steps=40)
    log('at grasp pose', np.round(rob.pos(), 4).tolist(), 'z_g', round(z_g, 4), 'top', round(top, 4))
    # 5) close
    rob.grip = 1.0
    rob.hold(CLOSE_STEPS, 1.0, p_hold=np.array([xy[0], xy[1], z_g]), R_hold=R_des)
    fsum = rob.fingers()
    log('fingers after close %.4f' % fsum)
    if fsum < FINGER_MIN:
        log('grasp missed; reopen and retry')
        rob.grip = -1.0
        rob.hold(OPEN_STEPS, -1.0)
        rob.move_to([xy[0], xy[1], z_high], R_des, -1.0, vmax=VMAX, max_steps=40)
        return False
    # 6) lift
    rob.move_to([xy[0], xy[1], z_g + LIFT], R_des, 1.0, vmax=VSLOW, max_steps=40)
    rob.hold(4, 1.0)
    fsum2 = rob.fingers()
    log('fingers after lift %.4f' % fsum2, 'tcp', np.round(rob.pos(), 4).tolist())
    if fsum2 < FINGER_MIN:
        log('dropped during lift')
        rob.grip = -1.0
        rob.move_to([xy[0], xy[1], z_high], R_des, -1.0, vmax=VMAX, max_steps=40)
        return False
    if last:
        return True
    # 7) put back and release
    rob.move_to([xy[0], xy[1], z_g + 0.003], R_des, 1.0, vmax=VSLOW, max_steps=40)
    rob.grip = -1.0
    rob.hold(OPEN_STEPS, -1.0)
    rob.move_to([xy[0], xy[1], z_high], R_des, -1.0, vmax=VMAX, max_steps=40)
    return True


def main(ctx):
    try:
        run(ctx)
    except EpisodeEnded:
        log('episode ended')
    except Exception:
        log('FATAL', traceback.format_exc())
    log('controller exit')
