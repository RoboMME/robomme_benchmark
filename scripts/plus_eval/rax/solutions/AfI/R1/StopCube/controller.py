"""StopCube controller (RAX, AfI, R1) -- written offline, blind to the real scene.

Assumed task (public task name + instruction, unverified): a cube slides by itself,
crossing a target; press the button so that the cube stops on the target at the
N-th visit, N parsed from the instruction.

Pipeline / transition table
  INIT      observe, parse instruction -> N, fit front-camera geometry  -> HOLD
  HOLD      zero rows (gripper closing), segment raised objects, find the one that
            moves (= cube); timeout 45 steps -> SELECT with best guess
  SELECT    target = flat coloured patch (or static object) closest to the cube path;
            button = static raised object farthest from the path       -> APPROACH
  APPROACH  servo above the button (+0.12 m) while tracking; wrist refine; press
            dynamics calibration (2 rows dz=-1); servo to hover (+GAP)  -> WAIT
  WAIT      hold hover, track cube, count visits; trigger when the N-th visit's
            closest approach is L steps away                            -> PRESS
  PRESS     dz=-1 rows until the cap is pushed (z below top-0.012 or stalled), lift
            -> VERIFY: cube stopped? yes -> POST (hold until episode ends);
               no -> WAIT again for the next visit (max 2 retries)
Every robot reply is checked for episode_ended; tool errors are logged and the loop
re-observes instead of crashing.
"""
import json
import math
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sc_io as io_  # noqa: E402
import sc_percept as pc  # noqa: E402
import sc_logic as lg  # noqa: E402

VERSION = 'stopcube-v1'
OUT = '/workspace/output'

# ---- tunables (blind defaults; tested on synthetic offline scenes only) ----
WALL_BUDGET_S = 1100.0     # controller wall limit is 1200 s
HOLD_MIN_STEPS = 4
HOLD_MAX_STEPS = 45
MOVE_DETECT_M = 0.012      # displacement that identifies the moving cube
STATIC_TOL_M = 0.006
SAFE_DZ = 0.10             # approach height above button top
PRESS_DZ = -0.5            # dz row used for calibration and pressing (2.5 cm request)
GAP = 0.025                # hover gap above the button top
PRESS_DEPTH = 0.012        # push the TCP this far below the measured top
REG_TRAVEL = 0.006         # assumed travel before the press registers
VMAX = 0.6                 # max |row| for servo moves (0.6 -> 3 cm/step request)
R_IN_MIN, R_IN_MAX = 0.02, 0.06
GRIP = 1.0                 # closed gripper for pressing


def _now():
    return time.time()


class Abort(Exception):
    pass


class Runner:
    def __init__(self, ctx):
        self.ctx = ctx
        self.t0 = _now()
        os.makedirs(OUT, exist_ok=True)
        self.logf = open(os.path.join(OUT, 'run_log.jsonl'), 'a')
        self.track = []           # (t, xy) detections of the cube
        self.track_meta = []
        self.ended = False
        self.complete = False
        self.lat = []
        self.model = {}           # camera -> geometry model
        self.first_meta_logged = False
        self.frame = None
        self.remaining = None
        self.max_steps = None
        self.phase = 'INIT'
        self.cube = None          # dict with area_ref, top_h
        self.static_mask = None
        self.table_z = None
        self.target = None        # dict center, r_in
        self.button = None        # dict xy, top_z
        self.N = 1
        self.L = 2
        self.press_t = None
        self.press_count = 0
        self.saved = 0
        self.frames_hold = []
        self.n_calls = 0
        self.video = None
        self.video_used = False
        self.H0 = None
        self.swap = False

    # ------------------------------------------------------------ logging
    def log(self, ev, **kw):
        rec = {'v': VERSION, 'ev': ev, 'phase': self.phase, 'wall': round(_now() - self.t0, 2),
               't': (self.frame or {}).get('t')}
        rec.update(kw)
        try:
            line = json.dumps(rec, default=_jsonable)
        except Exception:
            line = json.dumps({'ev': ev, 'repr': repr(kw)[:2000]})
        print(line, flush=True)
        try:
            self.logf.write(line + '\n')
            self.logf.flush()
        except Exception:
            pass

    def save_img(self, name, arr):
        if self.saved > 40 or arr is None:
            return
        try:
            from PIL import Image
            Image.fromarray(np.asarray(arr, np.uint8)).save(os.path.join(OUT, name))
            self.saved += 1
        except Exception:
            pass

    def wall_left(self):
        return WALL_BUDGET_S - (_now() - self.t0)

    # ------------------------------------------------------------ MCP
    def call(self, tool, **args):
        t = _now()
        self.n_calls += 1
        res = self.ctx.call(tool, **args)
        self.lat.append(_now() - t)
        return res

    def ingest(self, res, src):
        meta = io_.meta_of(res)
        imgs, notes = io_.extract_images(res, meta)
        self._dumped = getattr(self, '_dumped', set())
        if src not in self._dumped:
            self.log('meta_dump', src=src, meta=io_.strip_meta(meta), notes=notes,
                     n_blocks=len(io_.blocks(res)),
                     block_types=[(b.get('type'), sorted(k for k in b.keys() if k != 'data'))
                                  for b in io_.blocks(res) if isinstance(b, dict)][:12])
            self._dumped.add(src)
        state = io_.get_state(meta)
        t = io_.get_step_id(meta)
        ended, complete = io_.get_flags(meta)
        rem, mx = io_.get_remaining_steps(meta)
        if rem is not None:
            self.remaining = rem
        if mx is not None:
            self.max_steps = mx
        Ks, Ps = io_.get_calib(meta)
        if self.swap:
            imgs = _swap(imgs)
        need = [k for k in (('front', 'rgb'), ('front', 'depth')) if k not in imgs]
        if (need or state is None) and not ended and src != 'observe':
            try:
                r2 = self.call('robomme_observe')
                m2 = io_.meta_of(r2)
                im2, n2 = io_.extract_images(r2, m2)
                if self.swap:
                    im2 = _swap(im2)
                for k, v in im2.items():
                    imgs.setdefault(k, v)
                if state is None:
                    state = io_.get_state(m2)
                if t is None:
                    t = io_.get_step_id(m2)
                K2, P2 = io_.get_calib(m2)
                Ks = Ks or K2
                Ps = Ps or P2
                if not getattr(self, '_obs_note', False):
                    self.log('observe_merge', missing=[list(k) for k in need], notes=n2)
                    self._obs_note = True
            except Exception as e:
                self.log('observe_merge_err', err=repr(e))
        if ended:
            self.ended = True
        if complete:
            self.complete = True
        if t is None:
            t = (self.frame['t'] + 1) if self.frame else 0
        # calibration cache: intrinsics for both cameras, pose only for the static front camera
        self._Kc = getattr(self, '_Kc', {})
        self._Pc = getattr(self, '_Pc', {})
        Ks = dict(Ks or {})
        Ps = dict(Ps or {})
        for cam, K in list(Ks.items()):
            self._Kc[cam] = K
        if 'front' in Ps:
            self._Pc['front'] = Ps['front']
        for cam, K in self._Kc.items():
            Ks.setdefault(cam, K)
        if 'front' not in Ps and 'front' in self._Pc:
            Ps['front'] = self._Pc['front']
        fr = {'t': t, 'state': state, 'imgs': imgs, 'Ks': Ks, 'Ps': Ps, 'ended': ended,
              'complete': complete, 'meta': meta}
        if state is not None:
            fr['ee'] = np.asarray(state[:3], float)
        elif self.frame is not None:
            fr['ee'] = self.frame.get('ee')
        else:
            fr['ee'] = None
        self.frame = fr
        return fr

    def observe(self, first=False):
        res = self.call('robomme_observe')
        return self.ingest(res, 'observe0' if first else 'observe')

    def step(self, rows):
        rows = [[float(np.clip(x, -1.0, 1.0)) for x in r] for r in rows]
        if self.remaining is not None:
            rows = rows[:max(1, self.remaining)]
        try:
            res = self.call('robomme_step', actions=rows)
        except RuntimeError as e:
            self.log('step_error', err=str(e)[:500])
            # do not retry blindly: re-observe
            try:
                st = self.call('robomme_status')
                m = io_.meta_of(st)
                ended, complete = io_.get_flags(m)
                self.log('status_after_error', meta=io_.strip_meta(m))
                if ended:
                    self.ended = True
                    raise Abort('episode ended (after error)')
            except Abort:
                raise
            except Exception as e2:
                self.log('status_err', err=repr(e2))
            fr = self.observe()
            if self.ended:
                raise Abort('ended')
            return fr
        fr = self.ingest(res, 'step')
        if self.ended:
            self.log('episode_ended', complete=self.complete)
            raise Abort('episode ended')
        if self.wall_left() < 15:
            raise Abort('wall budget')
        return fr

    # ------------------------------------------------------------ geometry
    def fit_geometry(self, fr, cam):
        """Fit X = C + d*(u*a1 + v*a2 + a3) from tool back-projections; compare
        with calibration-based variants; keep the best model."""
        dep = fr['imgs'].get((cam, 'depth'))
        if dep is None:
            return None
        H, W = dep.shape
        sel = []
        grid = np.linspace(12, min(H, W) - 13, 8)
        for vv in grid:
            for uu in grid:
                u, v = int(round(uu)), int(round(vv))
                d = dep[v, u]
                win = dep[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
                if d > 1e-4 and np.all(win > 1e-4) and np.ptp(win) < 0.03 * d:
                    sel.append((u, v, float(d)))
        if len(sel) > 32:
            idx = np.linspace(0, len(sel) - 1, 32).round().astype(int)
            sel = [sel[i] for i in idx]
        pts = []
        if sel:
            try:
                res = self.call('robomme_pixel_to_position', observation_step_id=int(fr['t']),
                                selections=[{'camera': cam, 'u': float(u), 'v': float(v)} for u, v, _ in sel],
                                depth_window_radius=0)
                parsed = io_.parse_p2p(res)
                if len(parsed) != len(sel):
                    self.log('p2p_len_mismatch', n=len(parsed), want=len(sel),
                             meta=io_.strip_meta(io_.meta_of(res)))
                for (u, v, d), (p, sp) in zip(sel, parsed):
                    if p is not None:
                        pts.append((u, v, d, p))
                if not getattr(self, '_p2p_logged', False):
                    self.log('p2p_sample', meta=io_.strip_meta(io_.meta_of(res)))
                    self._p2p_logged = True
            except Exception as e:
                self.log('p2p_err', err=str(e)[:400])
        K = (fr.get('Ks') or {}).get(cam)
        P = (fr.get('Ps') or {}).get(cam)
        cal = []
        if K is not None and P is not None:
            for var in io_.VARIANTS:
                e = None
                if pts:
                    U = np.array([p[0] for p in pts], float)
                    V = np.array([p[1] for p in pts], float)
                    D = np.array([p[2] for p in pts], float)
                    X = np.array([p[3] for p in pts], float)
                    e = float(np.median(np.linalg.norm(io_.backproject(U, V, D, K, P, var) - X, axis=1)))
                cal.append((var, e))
        lin = None
        if len(pts) >= 8:
            U = np.array([p[0] for p in pts], float)
            V = np.array([p[1] for p in pts], float)
            D = np.array([p[2] for p in pts], float)
            X = np.array([p[3] for p in pts], float)
            # camera centre: pose translation if available, else free (needs non-planar samples)
            Xc = X - X.mean(0)
            planar = float(np.sqrt(np.linalg.eigvalsh(np.cov(Xc.T))[0]))
            Mf = np.stack([np.ones_like(D), D * U, D * V, D], 1)
            Cs = []
            if P is not None:
                Cs = [P[1], -P[0].T @ P[1]]
            fits = []
            for Cfix in Cs:
                M = np.stack([D * U, D * V, D], 1)
                coef3, *_ = np.linalg.lstsq(M, X - Cfix, rcond=None)
                coef = np.vstack([Cfix[None], coef3])
                err = np.linalg.norm(Mf @ coef - X, axis=1)
                fits.append((float(np.median(err)), float(err.max()), coef, True))
            if planar > 0.02 or not fits:
                # free camera centre (degenerate when all samples lie on one plane)
                coef, *_ = np.linalg.lstsq(Mf, X, rcond=None)
                err = np.linalg.norm(Mf @ coef - X, axis=1)
                fits.append((float(np.median(err)), float(err.max()), coef, False))
            e_med, e_max, coef, cfix = min(fits, key=lambda f: f[0])
            lin = {'type': 'lin', 'coef': coef, 'err': e_med, 'err_max': e_max,
                   'cfix': cfix, 'planar_sd': planar}
        self.log('geom_fit', cam=cam, n_pts=len(pts), lin=None if lin is None else
                 {k: (round(v, 5) if isinstance(v, float) else v) for k, v in lin.items() if k != 'coef'},
                 cal=[(list(v), None if e is None else round(e, 4)) for v, e in cal],
                 conv=None if P is None else P[2])
        cal_ok = [c for c in cal if c[1] is not None]
        if cal_ok:
            var, e = min(cal_ok, key=lambda c: c[1])
            if e < 0.008:
                return {'type': 'cal', 'variant': var, 'err': e}
        if lin is not None and lin['err'] < 0.01 and (lin['cfix'] or lin['planar_sd'] > 0.02):
            return lin
        if cal_ok:
            var, e = min(cal_ok, key=lambda c: c[1])
            if lin is None or e <= lin['err']:
                return {'type': 'cal', 'variant': var, 'err': e}
            return lin
        if K is not None and P is not None:
            conv = (P[2] or '')
            var = ('cv' if ('cv' in conv and 'gl' not in conv) else 'gl', 0.0, False)
            return {'type': 'cal', 'variant': var, 'err': None}
        return lin

    def points(self, fr, cam, model=None):
        model = model or self.model.get(cam)
        dep = fr['imgs'].get((cam, 'depth'))
        if dep is None or model is None:
            return None, None
        H, W = dep.shape
        valid = np.isfinite(dep) & (dep > 1e-4) & (dep < 20)
        if model['type'] == 'lin':
            c = model['coef']
            vv, uu = np.mgrid[0:H, 0:W]
            d = dep[..., None]
            P = c[0][None, None] + d * (uu[..., None] * c[1][None, None] + vv[..., None] * c[2][None, None] + c[3][None, None])
        else:
            K = (fr.get('Ks') or {}).get(cam)
            Pz = (fr.get('Ps') or {}).get(cam)
            if K is None or Pz is None:
                return None, None
            P, _ = io_.dense_points(dep, K, Pz, model['variant'])
        return P, valid

    def analyze(self, fr):
        """Per-frame front analysis: points, heights, edges."""
        P, valid = self.points(fr, 'front')
        if P is None:
            return None
        Hm = P[..., 2] - self.table_z if self.table_z is not None else None
        edges = pc.edge_mask(P, valid)
        return {'P': P, 'valid': valid, 'H': Hm, 'edges': edges, 'rgb': fr['imgs'].get(('front', 'rgb'))}

    # ------------------------------------------------------------ motion
    def ee(self):
        return self.frame.get('ee') if self.frame else None

    def servo_row(self, target, vmax=VMAX, gain=0.9):
        e = np.asarray(target, float) - self.ee()
        d = np.clip(gain * e / 0.05, -vmax, vmax)
        return [d[0], d[1], d[2], 0.0, 0.0, 0.0, GRIP]

    def servo_to(self, target, tol=0.004, max_steps=60, vmax=VMAX, why=''):
        target = np.asarray(target, float)
        n0 = self.frame['t']
        stall = 0
        last_err = None
        for i in range(max_steps):
            err = float(np.linalg.norm(target - self.ee()))
            if err < tol:
                self.log('servo_done', why=why, err=round(err, 4), steps=i)
                return True
            if last_err is not None and last_err - err < 0.0005:
                stall += 1
            else:
                stall = 0
            if stall >= 6:
                self.log('servo_stall', why=why, err=round(err, 4), ee=self.ee())
                return False
            last_err = err
            self.step([self.servo_row(target, vmax)])
            self.on_frame()
        self.log('servo_timeout', why=why, err=round(float(np.linalg.norm(target - self.ee())), 4),
                 steps=self.frame['t'] - n0)
        return False

    # ------------------------------------------------------------ cube tracking
    def on_frame(self):
        """Called after every step: track the cube once identified."""
        fr = self.frame
        if self.cube is None or self.static_mask is None:
            if self.phase in ('HOLD',) or self.cube is None:
                self.frames_hold.append(self._light(fr))
                self.frames_hold = self.frames_hold[-80:]
            return
        try:
            A = self.analyze(fr)
            if A is None:
                return
            det = self.detect_cube(A, fr)
            if det is not None:
                self.track.append((fr['t'], det['xy']))
                self.track_meta.append({'t': fr['t'], 'area': det['area'], 'top_h': det['top_h']})
                self.track = self.track[-3000:]
                self.track_meta = self.track_meta[-60:]
        except Exception as e:
            self.log('track_err', err=repr(e), tb=traceback.format_exc()[-800:])

    def _light(self, fr):
        return {'t': fr['t'], 'imgs': {k: v for k, v in fr['imgs'].items() if k[0] == 'front'},
                'Ks': fr.get('Ks'), 'Ps': fr.get('Ps'), 'ee': fr.get('ee')}

    def detect_cube(self, A, fr, gate_override=None):
        cands = pc.cube_candidates(A['P'], A['valid'], A['H'], A['rgb'], A['edges'], self.static_mask,
                                   fr.get('ee'), min_area=5, H0=self.H0)
        if not cands:
            return None
        ref = self.cube.get('area_ref', 30)
        recent = [m['area'] for m in self.track_meta[-10:]]
        if recent:
            ref = float(np.median(recent))
        last_t, last_xy = (self.track[-1] if self.track else (None, self.cube.get('xy0')))
        pred = np.asarray(last_xy, float)
        gap = 1
        if last_t is not None:
            gap = max(1, fr['t'] - last_t)
            fm = lg.fit_motion(self.track, last_t, window=6)
            if fm is not None:
                pred = fm[0] + fm[1] * gap
                sp = float(np.linalg.norm(fm[1]))
            else:
                sp = 0.01
        else:
            sp = 0.01
        base = 0.03 if gap <= 3 else 0.05
        gate = gate_override or min(0.6, base + 2.0 * sp * gap + 0.004 * gap)
        th = self.cube.get('top_h', 0.04)
        best, bs = None, 1e9
        for c in cands:
            if c['area'] < max(5, 0.25 * ref) or c['area'] > 3.5 * ref + 20:
                continue
            if c['max_h'] > max(0.03, 2.5 * th):
                continue
            if abs(c['top_h'] - th) > max(0.012, 0.4 * th):
                continue
            xy = c['xy_top']
            dist = float(np.linalg.norm(xy - pred))
            if dist > gate:
                continue
            s = dist + 0.02 * abs(math.log((c['area'] + 1) / (ref + 1)))
            if s < bs:
                best, bs = c, s
        if best is None:
            return None
        return {'xy': np.asarray(best['xy_top'], float), 'area': best['area'], 'top_h': best['top_h']}

    # ------------------------------------------------------------ phases
    def init(self):
        fr = self.observe(first=True)
        instr = io_.get_instruction(fr['meta'])
        self.N, how = lg.parse_count(instr)
        seq = io_.bfs_find(fr['meta'], ['frame_sequence'])
        vid = None
        if isinstance(seq, dict) and isinstance(seq.get('manifest_path'), str):
            try:
                man = json.load(open(os.path.join(io_.WS, seq['manifest_path'])))
                frames = man.get('frames', [])
                vid = {'n_frames': len(frames), 'n_video': sum(1 for f in frames if f.get('phase') == 'video_demo'),
                       'sampling': man.get('video_sampling') or seq.get('video_sampling'),
                       'freq': man.get('frequency_hz')}
                vfr = []
                ex_sfi = None
                for i, f in enumerate(frames):
                    sfi = f.get('source_frame_index', f.get('frame_index', i))
                    if f.get('phase') == 'execution':
                        if ex_sfi is None:
                            ex_sfi = sfi
                        continue
                    if f.get('phase') != 'video_demo':
                        continue
                    files = {}
                    for x in f.get('files', []):
                        if isinstance(x, dict) and 'path' in x:
                            files[(x.get('camera'), x.get('kind'))] = (x['path'], x.get('meters_per_unit'))
                    vfr.append({'sfi': sfi, 'ts': f.get('timestamp_s'), 'files': files})
                if vfr:
                    if ex_sfi is None:
                        ex_sfi = vfr[-1]['sfi'] + 1
                    self.video = {'frames': vfr, 'ex_sfi': ex_sfi}
            except Exception as e:
                vid = {'err': repr(e)}
        self.log('init', instruction=instr, N=self.N, how=how, state=fr.get('state'), video=vid,
                 remaining=self.remaining, max_steps=self.max_steps)
        try:
            st = self.call('robomme_status')
            m = io_.meta_of(st)
            self.log('robomme_status', meta=io_.strip_meta(m))
            rem, mx = io_.get_remaining_steps(m)
            if rem is not None:
                self.remaining = rem
            if mx is not None:
                self.max_steps = mx
            if not instr:
                instr = io_.get_instruction(m)
                if instr:
                    self.N, how = lg.parse_count(instr)
                    self.log('instr_from_status', instruction=instr, N=self.N, how=how)
        except Exception as e:
            self.log('status_err', err=repr(e))
        self.save_img('init_front.png', fr['imgs'].get(('front', 'rgb')))
        self.save_img('init_wrist.png', fr['imgs'].get(('wrist', 'rgb')))
        m = self.fit_geometry(fr, 'front')
        if (m is None or (m.get('err') is not None and m['err'] > 0.02)) and ('wrist', 'depth') in fr['imgs']:
            # maybe the attachment order was the other way round: try swapping cameras
            self.swap = True
            sw = _swap(fr['imgs'])
            fr2 = dict(fr)
            fr2['imgs'] = sw
            m2 = self.fit_geometry(fr2, 'front')
            if m2 is not None and m2.get('err') is not None and (m is None or m.get('err') is None or m2['err'] < m['err']):
                self.log('camera_swap', err_before=None if m is None else m.get('err'), err_after=m2['err'])
                fr['imgs'] = sw
                m = m2
            else:
                self.swap = False
        if m is None:
            raise RuntimeError('no front geometry')
        self.model['front'] = m
        P, valid = self.points(fr, 'front')
        self.table_z = pc.table_height(P, valid, None if fr['ee'] is None else fr['ee'][2])
        self.log('table', table_z=self.table_z, ee=fr['ee'], valid_frac=float(valid.mean()))
        self.frames_hold = [self._light(fr)]

    def objects_of(self, fr):
        A = self.analyze(fr)
        if A is None:
            return None, None
        objs, robot = pc.raised_components(A['P'], A['valid'], A['H'], A['rgb'], A['edges'])
        ee = fr.get('ee')
        if ee is not None:
            objs = [o for o in objs if np.linalg.norm(o['xyz'][:2] - ee[:2]) > 0.06 or o['xyz'][2] < ee[2] - 0.08]
        return objs, (A, robot)

    def hold_and_identify(self):
        self.phase = 'HOLD'
        fr0 = self.frames_hold[0]
        objs0, (A0, robot0) = self.objects_of(fr0)
        self.log('objects0', objs=[_cdesc(o) for o in objs0])
        self.A0, self.robot0, self.objs0 = A0, robot0, objs0
        cube_idx, cube_now = None, None
        for i in range(HOLD_MAX_STEPS):
            self.step([[0, 0, 0, 0, 0, 0, GRIP]])
            self.frames_hold.append(self._light(self.frame))
            if i + 1 < HOLD_MIN_STEPS:
                continue
            objs, _ = self.objects_of(self.frame)
            if objs is None:
                continue
            res = match_objects(objs0, objs)
            moved = [(j, d, c) for j, (d, c) in res.items() if c is not None and d > MOVE_DETECT_M]
            vanished = [j for j, (d, c) in res.items() if c is None]
            if moved:
                j, d, c = max(moved, key=lambda m: m[1])
                cube_idx, cube_now = j, c
            elif vanished:
                news = [c for c in objs if all(np.linalg.norm(c['xy_top'] - o['xy_top']) > 0.02 for o in objs0)]
                if news:
                    best = None
                    for j in vanished:
                        for c in news:
                            dd = np.linalg.norm(c['xy_top'] - objs0[j]['xy_top'])
                            if dd < 0.3 and (best is None or dd < best[0]):
                                best = (dd, j, c)
                    if best:
                        cube_idx, cube_now = best[1], best[2]
            if cube_idx is not None:
                self.log('cube_identified', obj=_cdesc(objs0[cube_idx]), now=_cdesc(cube_now), hold_steps=i + 1,
                         match={int(j): (round(d, 4) if d is not None else None) for j, (d, c) in res.items()})
                break
        if cube_idx is None:
            self.log('cube_not_identified')
            return False
        o = objs0[cube_idx]
        self.cube = {'idx': cube_idx, 'xy0': o['xy_top'], 'area_ref': o['area'], 'top_h': o['top_h']}
        self.static_ids = [j for j in range(len(objs0)) if j != cube_idx]
        sm = np.zeros(A0['valid'].shape, bool)
        for j in self.static_ids:
            sm |= objs0[j]['mask']
        # robot base: low pixels of robot components far from the hand
        ee0 = fr0.get('ee')
        if ee0 is not None:
            far = np.linalg.norm(A0['P'][..., :2] - ee0[None, None, :2], axis=-1) > 0.25
            sm |= robot0 & far & (A0['H'] < pc.TRACK_MAX_H)
        self.static_mask = pc.dilate(sm, 2)
        self.H0 = A0['H']
        # rebuild the track over the hold frames
        self.track = [(fr0['t'], np.asarray(o['xy_top'], float))]
        self.track_meta = [{'t': fr0['t'], 'area': o['area'], 'top_h': o['top_h']}]
        for f in self.frames_hold[1:]:
            A = self.analyze(f)
            det = self.detect_cube(A, f)
            if det is not None:
                self.track.append((f['t'], det['xy']))
                self.track_meta.append({'t': f['t'], 'area': det['area'], 'top_h': det['top_h']})
        self.log('track_init', n=len(self.track), first=self.track[0], last=self.track[-1])
        self.frames_hold = []
        try:
            self.backtrack_video()
        except Exception as e:
            self.log('video_err', err=repr(e), tb=traceback.format_exc()[-1200:])
        return True

    def backtrack_video(self):
        """If a conditioning video of the same scene precedes execution, track the cube
        backwards through it and prepend those samples (negative times) to the track,
        so visits before control count.  Continuity-gated."""
        if not self.video:
            return
        vf = self.video['frames']
        ex = self.video['ex_sfi']
        t_start = _now()
        fr0 = self.frame
        pos_t, pos = self.track[0]
        vtrack = []
        checked_scene = False
        n_seen = 0
        for f in reversed(vf):
            if _now() - t_start > 90:
                self.log('video_time_cap', n=n_seen)
                break
            fp = f['files'].get(('front', 'depth'))
            fc = f['files'].get(('front', 'rgb'))
            if fp is None or fc is None:
                continue
            kd, dep = io_.load_file_image(fp[0], fp[1])
            kc, rgb = io_.load_file_image(fc[0])
            if kd != 'depth' or kc != 'rgb':
                continue
            n_seen += 1
            t = int(round(f['sfi'] - ex))
            fr = {'t': t, 'imgs': {('front', 'depth'): dep, ('front', 'rgb'): rgb}, 'Ks': fr0.get('Ks'),
                  'Ps': fr0.get('Ps'), 'ee': None}
            A = self.analyze(fr)
            if A is None:
                return
            if not checked_scene:
                objs, _r = pc.raised_components(A['P'], A['valid'], A['H'], A['rgb'], A['edges'])
                stat0 = [self.objs0[j] for j in self.static_ids]
                res = match_objects(stat0, objs)
                same = all(d is not None and d < 0.012 for d, c in res.values()) if stat0 else False
                self.log('video_scene_check', same=same, n_static=len(stat0),
                         d=[None if d is None else round(d, 4) for d, c in res.values()])
                if not same:
                    return
                checked_scene = True
            cands = pc.cube_candidates(A['P'], A['valid'], A['H'], A['rgb'], A['edges'], self.static_mask,
                                       None, min_area=5, H0=self.H0)
            gap = abs(pos_t - t)
            gate = min(0.5, 0.04 + 0.02 * gap)
            ref = self.cube['area_ref']
            best, bd = None, 1e9
            for c in cands:
                if c['area'] < max(5, 0.2 * ref) or c['area'] > 5 * ref + 30:
                    continue
                d = float(np.linalg.norm(c['xy_top'] - pos))
                if d < gate and d < bd:
                    best, bd = c, d
            if best is None:
                if not vtrack:
                    self.log('video_no_continuity', t=t)
                    return
                continue
            vtrack.append((t, np.asarray(best['xy_top'], float)))
            pos_t, pos = t, vtrack[-1][1]
        if len(vtrack) >= 3:
            vtrack.reverse()
            self.track = vtrack + self.track
            self.video_used = True
        self.log('video_track', n_frames=len(vf), n_used=n_seen, n_track=len(vtrack),
                 first=vtrack[0] if vtrack else None, last=vtrack[-1] if vtrack else None)

    def path_line(self):
        X = np.array([xy for _, xy in self.track])
        if len(X) < 2:
            return None, None
        c = X.mean(0)
        if np.ptp(X, axis=0).max() < 0.01:
            d = X[-1] - X[0]
        else:
            w, V = np.linalg.eigh(np.cov((X - c).T))
            d = V[:, -1]
        n = np.linalg.norm(d)
        if n < 1e-9:
            return c, None
        return c, d / n

    def select_target_button(self):
        self.phase = 'SELECT'
        A0, objs0 = self.A0, self.objs0
        excl = (A0['valid'] & (A0['H'] > pc.RAISED_MIN_H)) | self.robot0
        patches, info = pc.flat_patches(A0['P'], A0['valid'], A0['H'], A0['rgb'], excl)
        c, d = self.path_line()
        self.log('patches', info=info, patches=[{'center': p['center'], 'radius': round(p['radius'], 4),
                                                 'area': p['area'], 'rgb': p['rgb'], 'ecc': round(p['ecc'], 2)}
                                                for p in patches[:12]], path_c=c, path_d=d)
        X = np.array([xy for _, xy in self.track])
        cands = []
        for p in patches:
            if not (0.008 <= p['radius'] <= 0.15):
                continue
            perp = pc.point_line_dist(p['center'], c, d) if d is not None else float(np.min(np.linalg.norm(X - p['center'], axis=1)))
            cands.append((perp, -p['area'], 'patch', p))
        for j in self.static_ids:
            o = objs0[j]
            if o['top_h'] < 0.02:
                perp = pc.point_line_dist(o['xy_top'], c, d) if d is not None else 1.0
                cands.append((perp + 0.01, -o['area'], 'lowobj', o))
        cands.sort(key=lambda x: (x[0], x[1]))
        near = [c_ for c_ in cands if c_[0] < 0.04 and (c_[2] != 'patch' or c_[3]['area'] >= 25)]
        if near:
            cands = sorted(near, key=lambda x: x[1]) + cands
        tgt = None
        if cands and cands[0][0] < 0.07:
            perp, _, kind, p = cands[0]
            if kind == 'patch':
                tgt = {'center': np.asarray(p['center'], float), 'radius': p['radius'], 'kind': kind, 'perp': perp}
            else:
                tgt = {'center': np.asarray(p['xy_top'], float), 'radius': max(0.02, 2 * p['spread']), 'kind': kind,
                       'perp': perp, 'obj': p}
        if tgt is None:
            mid = 0.5 * (X.min(0) + X.max(0))
            tgt = {'center': mid, 'radius': 0.03, 'kind': 'fallback_mid', 'perp': None}
        r_in = float(np.clip(tgt['radius'], R_IN_MIN, R_IN_MAX))
        tgt['r_in'] = r_in
        self.target = tgt
        # button
        bcs = []
        for j in self.static_ids:
            o = objs0[j]
            if tgt.get('obj') is o:
                continue
            if not (0.008 <= o['top_h'] <= pc.OBJECT_MAX_H) or o['area'] < 15:
                continue
            if np.linalg.norm(o['xy_top'] - tgt['center']) < 0.03:
                continue
            perp = pc.point_line_dist(o['xy_top'], c, d) if d is not None else 0.0
            bcs.append((perp, o))
        self.log('button_cands', cands=[(round(p, 4), _cdesc(o)) for p, o in bcs])
        if not bcs:
            return False
        far = [b for b in bcs if b[0] > 0.04]
        pick = max(far, key=lambda b: b[1]['area']) if far else max(bcs, key=lambda b: b[0])
        o = pick[1]
        self.button = self.button_geometry(o['mask'], A0)
        self.button['obj'] = _cdesc(o)
        self.log('selected', target={k: v for k, v in tgt.items() if k != 'obj'}, button=self.button, N=self.N)
        return True

    def button_geometry(self, mask, A):
        hs = A['H'][mask]
        pts = A['P'][mask]
        top = float(np.percentile(hs, 97))
        cap = hs >= top - 0.004
        if cap.sum() < 5:
            cap = hs >= top - 0.008
        xy = np.median(pts[cap, :2], axis=0)
        return {'xy': xy, 'top_z': float(self.table_z + top), 'top_h': top, 'n_cap': int(cap.sum())}

    def wrist_refine(self):
        fr = self.frame
        if ('wrist', 'depth') not in fr['imgs'] or 'wrist' not in (fr.get('Ps') or {}):
            try:
                fr = self.observe()
            except Exception as e:
                self.log('wrist_observe_err', err=repr(e))
                fr = self.frame
        if ('wrist', 'depth') not in fr['imgs']:
            self.log('wrist_refine_skip', why='no wrist depth')
            return
        m = self.fit_geometry(fr, 'wrist')
        if m is None or (m.get('err') is not None and m['err'] > 0.01):
            self.log('wrist_refine_skip', why='geometry', model=None if m is None else m.get('err'))
            return
        P, valid = self.points(fr, 'wrist', m)
        if P is None:
            return
        self.save_img('wrist_above_button.png', fr['imgs'].get(('wrist', 'rgb')))
        b = self.button
        dxy = np.linalg.norm(P[..., :2] - b['xy'][None, None], axis=-1)
        near = valid & (dxy < 0.05)
        if near.sum() < 20:
            self.log('wrist_refine_skip', why='few points', n=int(near.sum()))
            return
        z = P[..., 2]
        H = z - self.table_z
        cand = near & (H > 0.004) & (H < pc.OBJECT_MAX_H) & (z < self.ee()[2] - 0.03)
        if cand.sum() < 15:
            self.log('wrist_refine_skip', why='few raised', n=int(cand.sum()))
            return
        top = float(np.percentile(H[cand], 97))
        cap = cand & (H >= top - 0.008)
        xy = np.median(P[cap][:, :2], axis=0)
        shift = float(np.linalg.norm(xy - b['xy']))
        dtop = top - b['top_h']
        self.log('wrist_refine', xy=xy, top_h=top, shift=round(shift, 4), dtop=round(dtop, 4), n_cap=int(cap.sum()))
        if shift < 0.03 and abs(dtop) < 0.02:
            b['xy'] = 0.5 * (b['xy'] + xy) if shift > 0.015 else xy
            b['top_h'] = max(b['top_h'], top)
            b['top_z'] = float(self.table_z + b['top_h'])

    def approach(self):
        self.phase = 'APPROACH'
        b = self.button
        above = np.array([b['xy'][0], b['xy'][1], b['top_z'] + SAFE_DZ])
        if self.ee()[2] < above[2] - 0.02:
            up = self.ee().copy()
            up[2] = above[2]
            self.servo_to(up, tol=0.01, max_steps=25, why='lift')
        self.servo_to(above, tol=0.006, max_steps=110, why='above_button')
        urgent = self.urgent()
        if not urgent:
            self.wrist_refine()
            b = self.button
            above = np.array([b['xy'][0], b['xy'][1], b['top_z'] + SAFE_DZ])
            self.servo_to(above, tol=0.004, max_steps=20, why='above_button2')
        if not urgent and not self.urgent():
            # press-dynamics calibration in free air
            z0 = self.ee()[2]
            zs = [z0]
            for _ in range(2):
                self.step([[0, 0, PRESS_DZ, 0, 0, 0, GRIP]])
                self.on_frame()
                zs.append(self.ee()[2])
            d = [max(2e-3, zs[i] - zs[i + 1]) for i in range(2)]
            self.dz_step = d
        else:
            self.dz_step = [0.008, 0.012]
            self.log('urgent_skip_calib')
        self.L = self.latency(b['top_z'] + GAP)
        self.log('press_calib', dz=[round(x, 4) for x in self.dz_step], L=self.L)
        self.hover = np.array([b['xy'][0], b['xy'][1], b['top_z'] + GAP])
        # descend to hover inside the WAIT loop (it servos to hover and can trigger early)

    def latency(self, z_from):
        """Steps from issuing dz=-1 rows at height z_from until the cap has travelled REG_TRAVEL."""
        d = self.dz_step
        need = max(0.0, z_from - (self.button['top_z'] - REG_TRAVEL))
        cum, L = 0.0, 0
        while cum < need and L < 12:
            cum += d[0] if L == 0 else d[1]
            L += 1
        return int(np.clip(L, 1, 12))

    def urgent(self):
        """The N-th visit is close: skip optional steps."""
        try:
            cv = lg.count_visits(self.track, self.target['center'], self.target['r_in'])
            if cv['completed'] < self.N - 1:
                return False
            ts = lg.steps_to_event(self.track, self.target['center'], self.frame['t'])
            u = ts < 25 or cv['inside']
            if u:
                self.log('urgent', ts=ts, completed=cv['completed'])
            return u
        except Exception:
            return False

    # ------------------------------------------------------------ waiting & pressing
    def effective_track(self):
        """Track with virtual samples from a periodic model while the cube is not seen."""
        if not self.track:
            return self.track, None
        last_t = self.track[-1][0]
        now = self.frame['t']
        if now - last_t <= 2:
            return self.track, None
        P = estimate_period(self.track)
        if P is None:
            return self.track, None
        ts = np.array([t for t, _ in self.track], float)
        X = np.array([xy for _, xy in self.track])
        virt = []
        for t in range(last_t + 1, now + 1):
            tp = t - P
            if tp < ts[0]:
                continue
            x = np.array([np.interp(tp, ts, X[:, 0]), np.interp(tp, ts, X[:, 1])])
            virt.append((t, x))
        return self.track + virt, P

    def wait_and_press(self):
        self.phase = 'WAIT'
        N_eff = self.N
        retries = 0
        last_log = -999
        n_exc = 0
        while True:
          try:
            trk, per = self.effective_track()
            last_seen = trk[-1][0] if trk else None
            now = self.frame['t']
            T = self.target['center']
            exy = float(np.linalg.norm(self.ee()[:2] - self.button['xy']))
            z = self.ee()[2]
            L_now = self.latency(z)
            ok, info = lg.press_decision(trk, T, self.target['r_in'], N_eff, now, L_now, last_seen)
            info['L'] = L_now
            if ok and (exy > 0.008 or z > self.button['top_z'] + 0.08):
                info['blocked'] = 'not_aligned exy=%.4f z=%.3f' % (exy, z)
                ok = False
            if now - last_log >= 10 or ok:
                self.log('wait', N=N_eff, info=info, period=per, cube=trk[-1][1] if trk else None,
                         ee=self.ee(), remaining=self.remaining)
                last_log = now
            if ok:
                self.log('PRESS_TRIGGER', info=info, L=L_now)
                stopped = self.press()
                self.press_count += 1
                if stopped:
                    return True
                if retries >= 2:
                    return False
                retries += 1
                cv = lg.count_visits(self.track, T, self.target['r_in'])
                N_eff = cv['completed'] + 1 + (0 if not cv['inside'] else 1)
                self.log('retry_arm', N_eff=N_eff)
                self.phase = 'WAIT'
                continue
            k = self.rows_per_call(trk, info, N_eff)
            if float(np.linalg.norm(self.ee() - self.hover)) > 0.006:
                k = 1
                # descend only when aligned in xy; never below the hover height
                tgt = self.hover.copy()
                if exy > 0.01:
                    tgt[2] = max(z, self.hover[2]) if z < self.hover[2] + 0.06 else z
                row = self.servo_row(tgt, vmax=0.45, gain=0.8)
                row[2] = max(row[2], (self.hover[2] - z) / 0.05)
            else:
                row = self.servo_row(self.hover, vmax=0.25, gain=0.6)
                if not getattr(self, '_hover_logged', False):
                    self._hover_logged = True
                    self.log('at_hover', ee=self.ee(), hover=self.hover)
                    self.save_img('hover_front.png', self.frame['imgs'].get(('front', 'rgb')))
            rows = [row] + [[0, 0, 0, 0, 0, 0, GRIP]] * (k - 1)
            self.step(rows)
            self.on_frame()
          except Abort:
            raise
          except Exception as e:
            n_exc += 1
            self.log('wait_exc', err=repr(e), tb=traceback.format_exc()[-1500:], n=n_exc)
            if n_exc > 25:
                return False
            self.step([[0, 0, 0, 0, 0, 0, GRIP]])
            self.on_frame()

    def rows_per_call(self, trk, info, N_eff):
        lat = float(np.median(self.lat[-20:])) if self.lat else 0.5
        rem = self.remaining if self.remaining is not None else 600
        if lat * rem < 0.8 * self.wall_left():
            return 1
        if info.get('completed', 0) >= N_eff - 1:
            ts = lg.steps_to_event(trk, self.target['center'], self.frame['t'])
            if ts < self.L + 8:
                return 1
            return int(np.clip(math.floor((ts - self.L - 4) / 2), 1, 4))
        # earlier visits: coarse but bounded sampling
        if trk:
            d = float(np.linalg.norm(trk[-1][1] - self.target['center']))
            if d < self.target['r_in'] + 0.04:
                return 1
        return 3

    def press(self):
        self.phase = 'PRESS'
        b = self.button
        z_goal = b['top_z'] - PRESS_DEPTH
        zs = [self.ee()[2]]
        t_start = self.frame['t']
        self.press_t = t_start
        dz = PRESS_DZ
        n_stall = 0
        for i in range(16):
            e = b['xy'] - self.ee()[:2]
            cx, cy = np.clip(0.6 * e / 0.05, -0.2, 0.2)
            self.step([[cx, cy, dz, 0, 0, 0, GRIP]])
            self.on_frame()
            zs.append(self.ee()[2])
            if zs[-1] <= z_goal:
                break
            if zs[-2] - zs[-1] < 0.0015:
                n_stall += 1
                if zs[-1] < b['top_z'] - 0.004 and n_stall >= 2:
                    break  # bottomed out after travelling
                if n_stall >= 2 and dz > -1.0:
                    dz = -1.0  # resisting: push harder
                if n_stall >= 5:
                    break
            else:
                n_stall = 0
        self.log('pressed', zs=[round(z, 4) for z in zs], z_goal=round(z_goal, 4), top_z=round(b['top_z'], 4),
                 steps=self.frame['t'] - t_start)
        self.save_img('press_front_%d.png' % self.press_count, self.frame['imgs'].get(('front', 'rgb')))
        self.save_img('press_wrist_%d.png' % self.press_count, self.frame['imgs'].get(('wrist', 'rgb')))
        # brief hold, then lift
        self.step([[0, 0, -0.3, 0, 0, 0, GRIP]])
        self.on_frame()
        up = np.array([b['xy'][0], b['xy'][1], b['top_z'] + 0.05])
        self.servo_to(up, tol=0.008, max_steps=12, vmax=0.4, why='lift_after_press')
        # verify: is the cube still moving?
        self.phase = 'VERIFY'
        for _ in range(6):
            self.step([[0, 0, 0, 0, 0, 0, GRIP]])
            self.on_frame()
        recent = [(t, xy) for t, xy in self.track if t > self.frame['t'] - 6]
        moving = None
        if len(recent) >= 3:
            X = np.array([xy for _, xy in recent])
            moving = float(np.linalg.norm(X[-1] - X[0]))
        d_t = None
        if recent:
            d_t = float(np.linalg.norm(recent[-1][1] - self.target['center']))
        self.log('verify', moved_6=moving, dist_target=d_t, n=len(recent))
        self.save_img('verify_front_%d.png' % self.press_count, self.frame['imgs'].get(('front', 'rgb')))
        if moving is None:
            return True   # cannot tell; do not press again blindly
        return moving < 0.006

    def post(self):
        self.phase = 'POST'
        self.log('post_hold', remaining=self.remaining)
        while not self.ended and self.wall_left() > 30:
            n = 20
            if self.remaining is not None:
                if self.remaining <= 0:
                    break
                n = min(n, self.remaining)
            self.step([[0, 0, 0, 0, 0, 0, GRIP]] * n)
            self.on_frame()

    def run(self):
        self.init()
        ok = self.hold_and_identify()
        if not ok:
            # keep watching longer while doing nothing
            ok = self.hold_and_identify() if self.wall_left() > 200 else False
        if not ok:
            self.log('ABORT_NO_CUBE')
            self.post()
            return
        if not self.select_target_button():
            self.log('ABORT_NO_BUTTON')
            self.post()
            return
        self.approach()
        self.wait_and_press()
        self.post()


def match_objects(objs0, objs, max_d=0.2):
    """For each initial object, nearest current object with similar area.
    Returns {j: (displacement or None, comp or None)}."""
    out = {}
    used = set()
    pairs = []
    for j, o in enumerate(objs0):
        for i, c in enumerate(objs):
            r = (c['area'] + 1.0) / (o['area'] + 1.0)
            if 0.35 < r < 2.8:
                d = float(np.linalg.norm(c['xy_top'] - o['xy_top']))
                if d < max_d:
                    pairs.append((d, j, i))
    pairs.sort()
    for d, j, i in pairs:
        if j in out or i in used:
            continue
        out[j] = (d, objs[i])
        used.add(i)
    for j in range(len(objs0)):
        out.setdefault(j, (None, None))
    return out


def estimate_period(track, pmin=8, pmax=600):
    if len(track) < 20:
        return None
    ts = np.array([t for t, _ in track], float)
    X = np.array([xy for _, xy in track])
    span = ts[-1] - ts[0]
    best = None
    for P in range(pmin, int(min(pmax, span - 5)) + 1):
        m = ts >= ts[0] + P
        if m.sum() < 8:
            break
        xs = np.interp(ts[m] - P, ts, X[:, 0])
        ys = np.interp(ts[m] - P, ts, X[:, 1])
        e = float(np.sqrt(np.mean((xs - X[m, 0]) ** 2 + (ys - X[m, 1]) ** 2)))
        if best is None or e < best[0]:
            best = (e, P)
    if best is None:
        return None
    amp = float(np.ptp(X, axis=0).max())
    if best[0] < max(0.006, 0.1 * amp) and amp > 0.03:
        return best[1]
    return None


def _swap(imgs):
    out = {}
    for (cam, kind), v in imgs.items():
        out[({'front': 'wrist', 'wrist': 'front'}.get(cam, cam), kind)] = v
    return out


def _cdesc(o):
    return {'area': o.get('area'), 'xy_top': o.get('xy_top'), 'top_h': round(o.get('top_h', 0), 4),
            'max_h': round(o.get('max_h', 0), 4), 'rgb': o.get('rgb'), 'uv': o.get('uv')}


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return [round(float(x), 5) for x in o.reshape(-1)] if o.size <= 64 else 'array%s' % (o.shape,)
    if isinstance(o, (np.floating,)):
        return round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, tuple):
        return list(o)
    return repr(o)[:200]


def main(ctx):
    r = Runner(ctx)
    try:
        r.run()
    except Abort as e:
        r.log('abort', why=str(e), complete=r.complete)
    except Exception as e:
        r.log('EXCEPTION', err=repr(e), tb=traceback.format_exc()[-3000:])
        # never leave with steps unused if something crashed before pressing:
        try:
            if not r.ended and r.wall_left() > 60:
                r.post()
        except Exception as e2:
            r.log('post_exc', err=repr(e2))
    finally:
        r.log('done', complete=r.complete, ended=r.ended, calls=r.n_calls,
              lat_med=float(np.median(r.lat)) if r.lat else None, press_count=r.press_count)
        try:
            r.logf.close()
        except Exception:
            pass
