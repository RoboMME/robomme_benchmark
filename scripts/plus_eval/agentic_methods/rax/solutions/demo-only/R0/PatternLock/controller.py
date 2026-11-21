"""PatternLock controller (RAX, demo-only).

Pipeline (each step logs; any exception falls through to the next safe step):
  S0 read status/observation and the reset frame-sequence manifest
  S1 locate execution grid: detect pucks in the front image (RGB + depth), get
     world points with robomme_pixel_to_position, fit a 5x5 lattice (world xy)
  S2 locate the conditioning-video grid (front image of early video frames),
     map its pixels to world xy with a homography fitted on the execution pucks,
     fit a lattice with the axis branch nearest the execution lattice
  S3 extract the pattern: order of red-highlight onsets per video puck
  S4 execute: for each pattern cell -> move at travel height above the
     execution cell, dip until the cell turns red in the front image (or contact),
     lift; escalate with small xy offsets; stop when the episode ends.
Transitions: S0->S1->S2->S3->S4; failure in S1/S2/S3 -> best-effort fallbacks
(documented in each function); S4 ends on episode end, success or step budget.
"""
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perception as PC  # noqa: E402
import rio  # noqa: E402

VERSION = 'patternlock-v1'
OUT = Path('/workspace/output')

# ---- tunables (no live test was possible in demo-only; values chosen from demo geometry) ----
TRAVEL_ABOVE_TOP = 0.030      # first travel height above puck tops (m)
TRAVEL_ABOVE_ACT = 0.022      # travel height above the found activation height (m)
PROBE_STEP = 0.004            # descent increment while searching activation height (m)
FLOOR_BELOW_TOP = 0.012       # never command the tip lower than puck top minus this (m)
VMAX_TRAVEL = 0.022           # max per-step translation during travel (m)
VMAX_FINE = 0.010             # per-step translation for vertical moves (m)
XY_TOL = 0.003                # xy arrival tolerance (m)
Z_TOL = 0.003
MAX_ROWS_PER_CALL = 8
RED_MIN_PIX = 8               # red pixels near the target pixel to call it lit
WALL_LIMIT_S = 1100.0         # stop issuing motions after this (controller limit 1200 s)


class Log:
    def __init__(self):
        self.t0 = time.time()
        try:
            OUT.mkdir(parents=True, exist_ok=True)
            self.f = open(OUT / 'patternlock_log.txt', 'a')
        except Exception:
            self.f = None

    def __call__(self, *a):
        s = '[%7.2f] ' % (time.time() - self.t0) + ' '.join(str(x) for x in a)
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


LOG = Log()


def save_json(name, obj):
    try:
        (OUT / name).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    except Exception:
        pass


# =====================================================================  pattern
def video_frames_from_manifest(man):
    frames = man.get('frames') or []
    frames = sorted(frames, key=lambda f: (f.get('frame_index', 0)))
    vid = [f for f in frames if f.get('phase') == 'video_demo']
    exe = [f for f in frames if f.get('phase') != 'video_demo']
    return vid, exe


def grid_candidates_px(rgb, depth=None):
    """Candidate puck pixel lists from RGB and from depth."""
    cands = []
    try:
        cands.append(('rgb', PC.detect_pucks_rgb(rgb)))
    except Exception as e:
        LOG('rgb puck detection failed', e)
    if depth is not None:
        try:
            p, _ = PC.detect_pucks_depth(depth)
            cands.append(('depth', p))
        except Exception as e:
            LOG('depth puck detection failed', e)
    return cands


def lattice_score(lat):
    if lat is None:
        return -1e9
    return lat['inliers'] - 40.0 * lat['rms'] / max(lat['spacing'], 1e-6)


def extract_pattern(vid, man_file, H_px2w, ref_theta, spacing_hint, log=LOG):
    """Return (pattern [(i,j)...], info).  H_px2w maps front pixels -> world xy
    (puck-top plane).  If H_px2w is None, pixel coordinates are used directly."""
    info = {}
    if not vid:
        return [], {'error': 'no video frames'}
    # --- video grid from a few early frames
    best = None
    tried = sorted(set([0, 1, 2, 5, 10, len(vid) // 4]))
    for fi in tried:
        if fi >= len(vid):
            continue
        files = rio.frame_files(vid[fi])
        if ('front', 'rgb') not in files:
            continue
        try:
            rgb = rio.load_rgb(files[('front', 'rgb')], man_file)
        except Exception:
            continue
        depth = None
        if ('front', 'depth') in files:
            try:
                depth = rio.load_depth(files[('front', 'depth')], man_file)
            except Exception:
                depth = None
        for name, pk in grid_candidates_px(rgb, depth):
            if len(pk) < 6:
                continue
            px = np.array([p[:2] for p in pk])
            xy = rio_apply(H_px2w, px)
            lat = PC.fit_lattice(xy, ref_theta=ref_theta,
                                 spacing_hint=None)
            sc = lattice_score(lat)
            if lat is not None and spacing_hint is not None:
                # video grid spacing should be similar to the execution spacing
                ratio = lat['spacing'] / spacing_hint
                if ratio < 0.6 or ratio > 1.6:
                    sc -= 30
            if best is None or sc > best[0]:
                best = (sc, fi, name, px, xy, lat)
        if best is not None and best[5] is not None and best[5]['inliers'] >= 25 and best[5]['rms'] < 0.15 * best[5]['spacing']:
            break
    if best is None or best[5] is None:
        return [], {'error': 'video grid not found'}
    sc, fi, name, px, xy, lat = best
    info.update(video_grid_frame=fi, video_grid_src=name, video_inliers=lat['inliers'],
                video_rms=lat['rms'], video_spacing=lat['spacing'], video_theta_deg=math.degrees(lat['theta']))
    log('video grid', info)
    # pixel centre per cell: detected where assigned, else predicted via inverse homography
    cellpx = np.zeros((25, 2))
    Hinv = None if H_px2w is None else np.linalg.inv(H_px2w)
    for i in range(5):
        for j in range(5):
            w = lat['cells'][(i, j)]
            cellpx[i * 5 + j] = rio_apply(Hinv, np.array([w]))[0]
    for k, (i, j) in lat['assign'].items():
        cellpx[i * 5 + j] = px[k]
    dd = np.linalg.norm(cellpx[:, None] - cellpx[None], axis=-1)
    np.fill_diagonal(dd, np.inf)
    sp_px = float(np.median(dd.min(1)))
    # --- red events over all video frames
    reds, times = [], []
    for k, f in enumerate(vid):
        files = rio.frame_files(f)
        e = files.get(('front', 'rgb'))
        if e is None:
            reds.append([])
            times.append(float(f.get('timestamp_s', k / 20.0)))
            continue
        try:
            rgb = rio.load_rgb(e, man_file)
            reds.append(PC.red_blobs(rgb, min_size=5))
        except Exception:
            reds.append([])
        times.append(float(f.get('timestamp_s', k / 20.0)))
    events = PC.extract_events(reds, cellpx, max_dist_px=0.45 * sp_px, times=times)
    # optional refinement of near-simultaneous onsets with recorded video states
    events = refine_with_states(events, vid, lat, times, log)
    pattern = []
    for (cid, t, f, sz) in events:
        cell = (cid // 5, cid % 5)
        if pattern and pattern[-1] == cell:
            continue
        pattern.append(cell)
    info['events'] = [(int(c), round(float(t), 2), int(f), int(s)) for c, t, f, s in events]
    info['pattern'] = pattern
    info['cellpx'] = cellpx.tolist()
    info['video_lattice'] = {'origin': lat['origin'].tolist(), 'e1': lat['e1'].tolist(), 'e2': lat['e2'].tolist()}
    return pattern, info


def refine_with_states(events, vid, lat, times, log):
    """If video frames carry varying robot states, reorder events whose onsets are
    within 0.3 s of each other by the time the TCP came closest to each cell."""
    try:
        st = [f.get('state') for f in vid]
        if not all(rio.is_vec(s) and len(s) >= 3 for s in st):
            return events
        S = np.asarray([s[:3] for s in st], float)
        if np.ptp(S[:, 0]) < 0.02 and np.ptp(S[:, 1]) < 0.02:
            return events
        T = np.asarray(times)
        out = []
        good = 0
        for ev in events:
            cid, t0 = ev[0], ev[1]
            c = lat['cells'][(cid // 5, cid % 5)]
            win = (T >= t0 - 2.6) & (T <= t0 + 0.2)
            if not win.any():
                out.append((ev, t0))
                continue
            d = np.linalg.norm(S[win, :2] - c[None], axis=1)
            k = int(np.argmin(d))
            if d[k] < 0.5 * lat['spacing']:
                good += 1
                out.append((ev, float(T[win][k])))
            else:
                out.append((ev, t0))
        if good < max(1, len(events) // 2):
            log('video states do not match red events; ignoring states', good, len(events))
            return events
        res = []
        for i, (ev, tc) in enumerate(out):
            res.append((ev, ev[1], tc))
        # only swap order inside clusters of near-simultaneous onsets
        res.sort(key=lambda r: (round(r[1] / 0.3), r[2]))
        new = [r[0] for r in res]
        if [e[0] for e in new] != [e[0] for e in events]:
            log('state refinement reordered events', [e[0] for e in events], '->', [e[0] for e in new])
        return new
    except Exception as e:
        log('refine_with_states failed', e)
        return events


def rio_apply(Hm, pts):
    pts = np.asarray(pts, float)
    if Hm is None:
        return pts.copy()
    return PC.apply_homography(Hm, pts)


# =====================================================================  robot
class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = {}
        self.state = None
        self.step_id = None
        self.front = None
        self.wrist = None
        self.ended = False
        self.success = False
        self.steps_used = 0
        self.max_steps = None
        self.remaining = None
        self.ref_front = None
        self.R0 = None
        self.calls = 0
        self.errors = 0

    def _absorb(self, res):
        meta = rio.meta_of(res)
        self.meta = meta
        st = rio.state_of(meta)
        if st is not None:
            self.state = st
        sid = rio.step_id_of(meta)
        if sid is not None:
            self.step_id = sid
        if rio.flag(meta, 'episode_ended') or rio.flag(meta, 'terminated') or rio.flag(meta, 'truncated'):
            self.ended = True
        if rio.flag(meta, 'task_complete'):
            self.success = True
        self._limits(meta)
        try:
            f, w = rio.rgb_pair(res, self.ref_front)
            if f is not None:
                self.front = f
            if w is not None:
                self.wrist = w
        except Exception as e:
            LOG('image decode failed', e)
        return meta

    def _limits(self, meta):
        lim = rio.find_first(meta, 'episode_limits', pred=lambda v: isinstance(v, dict))
        if isinstance(lim, dict):
            for k, v in lim.items():
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                kl = k.lower()
                if 'remain' in kl and 'step' in kl:
                    self.remaining = int(v)
                elif ('max' in kl or 'limit' in kl) and 'step' in kl:
                    self.max_steps = int(v)
        for key in ('remaining_steps', 'steps_remaining'):
            v = rio.find_first(meta, key, pred=lambda v: isinstance(v, int))
            if v is not None:
                self.remaining = v
        for key in ('max_steps', 'step_limit', 'max_episode_steps'):
            v = rio.find_first(meta, key, pred=lambda v: isinstance(v, int))
            if v is not None and self.max_steps is None:
                self.max_steps = v

    def status(self):
        try:
            res = self.ctx.call('robomme_status')
            meta = rio.meta_of(res)
            self._limits(meta)
            if rio.flag(meta, 'episode_ended'):
                self.ended = True
            if rio.flag(meta, 'task_complete'):
                self.success = True
            return meta
        except Exception as e:
            LOG('status failed', e)
            return {}

    def observe(self):
        res = self.ctx.call('robomme_observe')
        self.calls += 1
        return res, self._absorb(res)

    def rows_left(self):
        if self.remaining is not None:
            return self.remaining
        if self.max_steps is not None and self.step_id is not None:
            return self.max_steps - self.step_id
        return 10 ** 6

    def send(self, rows):
        if self.ended:
            return False
        n = min(len(rows), 100, max(0, self.rows_left()))
        if n <= 0:
            LOG('no steps left')
            self.ended = True
            return False
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows[:n]]
        prev_sid = self.step_id
        try:
            res = self.ctx.call('robomme_step', actions=rows)
            self.calls += 1
        except Exception as e:
            self.errors += 1
            LOG('robomme_step error', str(e)[:300])
            # do not retry blindly: re-read the robot state
            try:
                self.observe()
            except Exception as e2:
                LOG('observe after error failed', str(e2)[:200])
            self.status()
            return False
        self._absorb(res)
        if self.step_id is not None and prev_sid is not None:
            self.steps_used += max(0, self.step_id - prev_sid)
            if self.remaining is not None and rio.find_first(self.meta, 'episode_limits') is None:
                self.remaining -= max(0, self.step_id - prev_sid)
        else:
            self.steps_used += n
            if self.remaining is not None:
                self.remaining -= n
        return True

    # ---------------------------------------------------------- motion
    def pos(self):
        return np.asarray(self.state[:3], float)

    def rot_correction(self):
        if self.R0 is None or self.state is None:
            return np.zeros(3)
        try:
            Rc = rio.rotvec_to_mat(self.state[3:6])
            e = rio.mat_to_rotvec(self.R0 @ Rc.T)
            if np.linalg.norm(e) > 0.6:  # implausible: do not fight it
                return np.zeros(3)
            return e
        except Exception:
            return np.zeros(3)

    def move_to(self, target, vmax=VMAX_TRAVEL, tol_xy=XY_TOL, tol_z=Z_TOL, max_iter=12):
        target = np.asarray(target, float)
        stall = 0
        for it in range(max_iter):
            if self.ended or time.time() - LOG.t0 > WALL_LIMIT_S:
                return False
            p = self.pos()
            d = target - p
            if np.linalg.norm(d[:2]) <= tol_xy and abs(d[2]) <= tol_z:
                return True
            dist = float(np.linalg.norm(d))
            need = max(1, int(math.ceil(dist / vmax)))
            n = min(need, MAX_ROWS_PER_CALL)
            per = d / need
            e = self.rot_correction()
            rot_per = np.clip(e / need / 0.5, -0.15, 0.15)
            row = [per[0] / 0.05, per[1] / 0.05, per[2] / 0.05, rot_per[0], rot_per[1], rot_per[2], -1.0]
            if not self.send([row] * n):
                if self.ended:
                    return False
                continue
            commanded = float(np.linalg.norm(per) * n)
            moved = float(np.linalg.norm(self.pos() - p))
            if commanded > 0.002 and moved < 0.3 * commanded:
                stall += 1
                if stall >= 2:
                    LOG('move_to stalled at', np.round(self.pos(), 4), 'target', np.round(target, 4))
                    return False
            else:
                stall = 0
        p = self.pos()
        d = target - p
        return np.linalg.norm(d[:2]) <= 2 * tol_xy and abs(d[2]) <= 2 * tol_z

    def hold(self, n=1):
        return self.send([[0, 0, 0, 0, 0, 0, -1.0]] * n)


# =====================================================================  grid
def pixel_to_world(robot, pixels, cal_front=None, depth=None, variant=None):
    """World points for front pixels: native tool first, own back-projection fallback."""
    pts = [None] * len(pixels)
    sid = robot.step_id if robot.step_id is not None else 0
    try:
        for s in range(0, len(pixels), 32):
            chunk = pixels[s:s + 32]
            sel = [{'camera': 'front', 'u': int(round(u)), 'v': int(round(v))} for (u, v) in chunk]
            res = robot.ctx.call('robomme_pixel_to_position', observation_step_id=int(sid),
                                 selections=sel, depth_window_radius=1)
            got = rio.p2p_points(res, len(chunk))
            if s == 0:
                LOG('pixel_to_position sample reply', (rio.texts_of(res) or [''])[0][:600])
            for k, g in enumerate(got[:len(chunk)]):
                pts[s + k] = g
    except Exception as e:
        LOG('pixel_to_position failed', str(e)[:300])
    if cal_front is not None and depth is not None and variant is not None:
        for k, (u, v) in enumerate(pixels):
            if pts[k] is None:
                ui, vi = int(round(u)), int(round(v))
                if 0 <= vi < depth.shape[0] and 0 <= ui < depth.shape[1] and depth[vi, ui] > 0:
                    pts[k] = rio.backproject(cal_front, u, v, depth[vi, ui], variant)
    return pts


def choose_variant(robot, cal_front, depth, log=LOG):
    """Pick the back-projection convention that matches the native tool."""
    if cal_front is None or depth is None:
        return None
    H, W = depth.shape
    pix = [(W * a, H * b) for a in (0.3, 0.5, 0.7) for b in (0.45, 0.65, 0.85)]
    ref = pixel_to_world(robot, pix)
    best = None
    for var in range(4):
        errs = []
        for (u, v), r in zip(pix, ref):
            if r is None:
                continue
            d = depth[int(round(v)), int(round(u))]
            if d <= 0:
                continue
            errs.append(np.linalg.norm(rio.backproject(cal_front, u, v, d, var) - r))
        if errs:
            m = float(np.median(errs))
            if best is None or m < best[0]:
                best = (m, var)
    if best is None:
        # no tool reference: default OpenGL convention as documented
        log('backprojection: no tool reference, default variant 0')
        return 0
    log('backprojection variant', best[1], 'median err', round(best[0], 4))
    return best[1] if best[0] < 0.02 else (0 if all(r is None for r in ref) else None)


def locate_exec_grid(robot, rgb, depth, cal_front, variant, log=LOG):
    cands = grid_candidates_px(rgb, depth)
    best = None
    for name, pk in cands:
        if len(pk) < 6:
            continue
        # keep the 40 largest blobs at most
        pk = sorted(pk, key=lambda p: -p[2])[:40]
        px = [(p[0], p[1]) for p in pk]
        w = pixel_to_world(robot, px, cal_front, depth, variant)
        keep = [k for k, x in enumerate(w) if x is not None]
        if len(keep) < 6:
            continue
        W3 = np.array([w[k] for k in keep])
        PX = np.array([px[k] for k in keep])
        # discard points far from the dominant height (robot parts)
        zmed = np.median(W3[:, 2])
        m = np.abs(W3[:, 2] - zmed) < 0.03
        W3, PX = W3[m], PX[m]
        lat = PC.fit_lattice(W3[:, :2])
        sc = lattice_score(lat)
        log('exec grid candidate', name, 'n', len(W3), 'score', round(sc, 2),
            None if lat is None else (lat['inliers'], round(lat['rms'], 4), round(lat['spacing'], 4), round(math.degrees(lat['theta']), 1)))
        if lat is not None and (best is None or sc > best[0]):
            best = (sc, name, PX, W3, lat)
    return best


def refine_centers(robot, PX, W3, lat, depth, cal_front, variant, log=LOG):
    """Improve per-puck world centres: average the top-surface points around each
    detected blob (own back-projection when validated, else extra tool samples)."""
    sp = lat['spacing']
    z_est = float(np.median(W3[list(lat['assign'].keys()), 2]))
    out = W3.copy()
    used = 'none'
    try:
        if cal_front is not None and depth is not None and variant is not None:
            X = rio.backproject_all(cal_front, depth, variant)
            Hh, Ww = depth.shape
            zmax_all = []
            for k in range(len(PX)):
                u, v = int(round(PX[k][0])), int(round(PX[k][1]))
                win = X[max(0, v - 8):v + 9, max(0, u - 8):u + 9].reshape(-1, 3)
                win = win[np.isfinite(win).all(1)]
                if len(win) == 0:
                    continue
                win = win[np.linalg.norm(win[:, :2] - W3[k, :2], axis=1) < 0.5 * sp]
                if len(win) == 0:
                    continue
                zt = np.percentile(win[:, 2], 90)
                top = win[win[:, 2] > zt - 0.002]
                if len(top) >= 3:
                    out[k, :2] = top[:, :2].mean(0)
                    out[k, 2] = float(np.median(top[:, 2]))
            used = 'backprojection'
        else:
            pix, owner = [], []
            for k in lat['assign'].keys():
                u, v = PX[k]
                for du in (-1, 0, 1):
                    for dv in (-2, -1, 0):
                        pix.append((u + du, v + dv))
                        owner.append(k)
            pts = pixel_to_world(robot, pix)
            per = {}
            for k, p in zip(owner, pts):
                if p is not None:
                    per.setdefault(k, []).append(p)
            for k, lst in per.items():
                A = np.asarray(lst)
                A = A[np.linalg.norm(A[:, :2] - W3[k, :2], axis=1) < 0.5 * sp]
                if len(A) == 0:
                    continue
                top = A[A[:, 2] > A[:, 2].max() - 0.002]
                out[k, :2] = top[:, :2].mean(0)
                out[k, 2] = float(np.median(top[:, 2]))
            used = 'tool-samples'
    except Exception as e:
        log('refine_centers failed', e)
        return W3
    shift = np.linalg.norm(out[:, :2] - W3[:, :2], axis=1)
    log('refine_centers', used, 'median shift', round(float(np.median(shift)), 4), 'max', round(float(shift.max()), 4))
    return out


# =====================================================================  main
def main(ctx):
    LOG(VERSION, 'start')
    robot = Robot(ctx)
    summary = {'version': VERSION}
    try:
        run(robot, summary)
    except Exception as e:
        LOG('FATAL', repr(e))
        LOG(traceback.format_exc())
        summary['fatal'] = repr(e)
    summary.update(steps_used=robot.steps_used, success=robot.success, ended=robot.ended,
                   calls=robot.calls, errors=robot.errors, wall=time.time() - LOG.t0)
    save_json('patternlock_summary.json', summary)
    LOG('done', {k: summary[k] for k in ('steps_used', 'success', 'ended', 'calls', 'errors', 'wall')})


def run(robot, summary):
    st = robot.status()
    LOG('status', json.dumps(st)[:800])
    res, meta = robot.observe()
    LOG('observe meta keys', list(meta.keys())[:30])
    LOG('instruction', rio.find_first(meta, 'instruction'))
    LOG('state', robot.state, 'step', robot.step_id, 'limits', robot.max_steps, robot.remaining)
    save_json('observe0_meta.json', meta)
    if robot.state is None:
        raise RuntimeError('no state in observation')
    robot.R0 = rio.rotvec_to_mat(robot.state[3:6])

    # ---------------- reset manifest
    mpath = rio.manifest_path_of(meta)
    man, man_file = ({}, None)
    if mpath:
        try:
            man, man_file = rio.load_manifest(mpath)
        except Exception as e:
            LOG('manifest load failed', mpath, e)
    vid, exe = video_frames_from_manifest(man)
    LOG('manifest', mpath, 'video frames', len(vid), 'exec frames', len(exe))

    # ---------------- current front rgb/depth
    front_rgb, front_depth = robot.front, None
    if exe:
        files = rio.frame_files(exe[-1])
        try:
            if ('front', 'rgb') in files:
                front_rgb = rio.load_rgb(files[('front', 'rgb')], man_file)
        except Exception as e:
            LOG('exec rgb load failed', e)
        try:
            if ('front', 'depth') in files:
                front_depth = rio.load_depth(files[('front', 'depth')], man_file)
        except Exception as e:
            LOG('exec depth load failed', e)
    if front_depth is None or front_rgb is None:
        # generic search: any {camera, kind, path} entries in the observation metadata
        try:
            ents = {}

            def walk(o, d=0):
                if d > 10:
                    return
                if isinstance(o, dict):
                    if isinstance(o.get('camera'), str) and isinstance(o.get('kind'), str) and isinstance(o.get('path'), str):
                        ents.setdefault((o['camera'], o['kind']), o)
                    for v in o.values():
                        walk(v, d + 1)
                elif isinstance(o, list):
                    for v in o:
                        walk(v, d + 1)
            walk(meta)
            if front_depth is None and ('front', 'depth') in ents:
                front_depth = rio.load_depth(ents[('front', 'depth')], man_file)
                LOG('front depth from observation metadata')
            if front_rgb is None and ('front', 'rgb') in ents:
                front_rgb = rio.load_rgb(ents[('front', 'rgb')], man_file)
        except Exception as e:
            LOG('metadata file search failed', e)
    if front_rgb is None:
        raise RuntimeError('no front image')
    LOG('front rgb', front_rgb.shape, 'depth', None if front_depth is None else (front_depth.shape, float(np.median(front_depth[front_depth > 0])) if (front_depth > 0).any() else 0))
    robot.ref_front = front_rgb
    if robot.front is None:
        robot.front = front_rgb
    cal = rio.calibration(meta)
    cal_front = cal.get('front')
    LOG('calibration cams', list(cal.keys()))
    variant = choose_variant(robot, cal_front, front_depth)

    # ---------------- S1 execution grid
    g = locate_exec_grid(robot, front_rgb, front_depth, cal_front, variant)
    if g is None:
        raise RuntimeError('execution grid not found')
    _, gname, PX, W3, lat_e = g
    W3r = refine_centers(robot, PX, W3, lat_e, front_depth, cal_front, variant)
    lat_r = PC.fit_lattice(W3r[:, :2], ref_theta=lat_e['theta'])
    if lat_r is not None and lat_r['inliers'] >= lat_e['inliers'] - 1 and lat_r['rms'] <= lat_e['rms'] * 1.5 + 0.002:
        W3, lat_e = W3r, lat_r
    LOG('exec lattice rms', round(lat_e['rms'], 4), 'inliers', lat_e['inliers'])
    z_top = float(np.median([W3[k, 2] for k in lat_e['assign']]))
    summary['exec_grid'] = dict(src=gname, inliers=lat_e['inliers'], rms=lat_e['rms'], spacing=lat_e['spacing'],
                                theta_deg=math.degrees(lat_e['theta']), z_top=z_top,
                                origin=lat_e['origin'].tolist())
    LOG('exec grid', summary['exec_grid'])
    # homography front pixel -> world xy on the puck-top plane
    idx = list(lat_e['assign'].keys())
    Hm = None
    if len(idx) >= 6:
        try:
            Hm = PC.fit_homography(PX[idx], W3[idx, :2])
            err = np.linalg.norm(PC.apply_homography(Hm, PX[idx]) - W3[idx, :2], axis=1)
            LOG('homography err median/max', round(float(np.median(err)), 4), round(float(err.max()), 4))
        except Exception as e:
            LOG('homography failed', e)
            Hm = None
    # exec cell pixel centres (for red checks)
    cell_px_e = {}
    Hinv = None if Hm is None else np.linalg.inv(Hm)
    for i in range(5):
        for j in range(5):
            if Hinv is not None:
                cell_px_e[(i, j)] = PC.apply_homography(Hinv, np.array([lat_e['cells'][(i, j)]]))[0]
    for k, (i, j) in lat_e['assign'].items():
        cell_px_e[(i, j)] = PX[k]

    # ---------------- S2+S3 pattern
    pattern, info = [], {}
    if vid:
        # video grid lattice: canonical branch nearest the execution lattice axes
        pattern, info = extract_pattern(vid, man_file, Hm, ref_theta=lat_e['theta'],
                                        spacing_hint=lat_e['spacing'])
    summary['pattern'] = pattern
    summary['pattern_info'] = {k: v for k, v in info.items() if k != 'cellpx'}
    LOG('PATTERN', pattern, 'events', info.get('events'))
    save_json('patternlock_pattern.json', info)
    if not pattern:
        raise RuntimeError('no pattern extracted')

    # ---------------- S4 execute
    execute(robot, pattern, lat_e, cell_px_e, z_top, summary)


def red_near(rgb, px, radius):
    if rgb is None or px is None:
        return 0
    m = PC.red_mask(rgb)
    H, W = m.shape
    u, v = float(px[0]), float(px[1])
    r = int(math.ceil(radius))
    u0, u1 = max(0, int(u) - r), min(W, int(u) + r + 2)
    v0, v1 = max(0, int(v) - r - 2), min(H, int(v) + r + 1)
    if u0 >= u1 or v0 >= v1:
        return 0
    yy, xx = np.mgrid[v0:v1, u0:u1]
    # red sits on the puck top, slightly above the detected centroid in the image
    sel = ((xx - u) ** 2 + ((yy - (v - 1.0)) * 1.0) ** 2) <= radius ** 2
    return int(m[v0:v1, u0:u1][sel].sum())


def execute(robot, pattern, lat, cell_px, z_top, summary):
    sp = lat['spacing']
    pts = {c: np.asarray(lat['cells'][c], float) for c in lat['cells']}
    dd = []
    for c, p in cell_px.items():
        for c2, p2 in cell_px.items():
            if c != c2:
                dd.append(np.linalg.norm(np.asarray(p) - np.asarray(p2)))
    sp_px = float(np.min(dd)) if dd else 15.0
    radius = max(3.5, 0.33 * sp_px)
    z_act = None
    z_floor = z_top - FLOOR_BELOW_TOP
    mode = 'probe'      # probe -> dip (z_act known) ; 'blind' if red never seen
    log_rows = []
    p0 = robot.pos()
    LOG('start pos', np.round(p0, 4), 'z_top', round(z_top, 4), 'spacing', round(sp, 4), 'sp_px', round(sp_px, 1))
    # estimated step need; reduce travel clearance if the budget is tight
    travel = TRAVEL_ABOVE_TOP
    left = robot.rows_left()
    path = 0.0
    prev = p0[:2]
    for c in pattern:
        path += np.linalg.norm(pts[c] - prev)
        prev = pts[c]
    est = path / VMAX_TRAVEL + len(pattern) * 10 + 30
    LOG('budget: rows left', left, 'estimated need', int(est))
    fast = left < est * 1.2
    if fast:
        LOG('tight budget -> fast mode')
    # first, rise/descend to travel height at the current xy if far above
    for k, c in enumerate(pattern):
        if robot.ended:
            break
        if time.time() - LOG.t0 > WALL_LIMIT_S:
            LOG('wall limit reached')
            break
        tgt = pts[c]
        px = cell_px.get(c)
        z_travel = (z_act + TRAVEL_ABOVE_ACT) if z_act is not None else z_top + travel
        if fast and z_act is not None:
            z_travel = z_act + 0.012
        cur = robot.pos()
        # go to travel height first if we are below it and must cross other pucks
        if cur[2] < z_travel - 0.004:
            robot.move_to(np.array([cur[0], cur[1], z_travel]), vmax=VMAX_FINE * 1.5, tol_xy=0.02)
        ok = robot.move_to(np.array([tgt[0], tgt[1], z_travel]), vmax=VMAX_TRAVEL * (1.4 if fast else 1.0))
        if not ok:
            ok = robot.move_to(np.array([tgt[0], tgt[1], z_travel]), vmax=VMAX_FINE)
        base = red_near(robot.front, px, radius)
        lit, how = False, ''
        rec = {'k': k, 'cell': c, 'target': tgt.tolist(), 'arrive': robot.pos().tolist(), 'base_red': base}
        if base >= RED_MIN_PIX and (k == 0 or pattern[k - 1] != c):
            # already lit above the puck (xy-only activation)
            lit, how = True, 'hover'
        offsets = [(0, 0), (0.004, 0), (0, 0.004), (-0.004, 0), (0, -0.004), (0.007, 0.007), (-0.007, -0.007)]
        if mode == 'blind':
            offsets = offsets[:1]
        for oi, (ox, oy) in enumerate(offsets):
            if lit or robot.ended:
                break
            if oi > 0:
                robot.move_to(np.array([tgt[0] + ox, tgt[1] + oy, robot.pos()[2] + 0.008]), vmax=VMAX_FINE)
            if z_act is None:
                # probe downwards
                z = robot.pos()[2]
                while z > z_floor and not robot.ended:
                    z = max(z_floor, z - PROBE_STEP * (2 if z > z_top + 0.015 else 1))
                    before = robot.pos()[2]
                    robot.move_to(np.array([tgt[0] + ox, tgt[1] + oy, z]), vmax=VMAX_FINE, tol_z=0.0015, max_iter=3)
                    n = red_near(robot.front, px, radius)
                    if n >= max(RED_MIN_PIX, base + 6):
                        lit, how = True, 'probe'
                        z_act = robot.pos()[2]
                        break
                    if before - robot.pos()[2] < 0.001 and z < before - 0.002:
                        LOG('contact/stall during probe at z', round(robot.pos()[2], 4))
                        robot.hold(1)
                        n = red_near(robot.front, px, radius)
                        if n >= max(RED_MIN_PIX, base + 6):
                            lit, how = True, 'probe-contact'
                            z_act = robot.pos()[2]
                        break
            else:
                zt = z_act - 0.002 - 0.002 * oi
                robot.move_to(np.array([tgt[0] + ox, tgt[1] + oy, max(zt, z_floor)]), vmax=VMAX_FINE, tol_z=0.002, max_iter=4)
                n = red_near(robot.front, px, radius)
                if n < max(RED_MIN_PIX, base + 6):
                    robot.hold(1)
                    n = red_near(robot.front, px, radius)
                if n >= max(RED_MIN_PIX, base + 6):
                    lit, how = True, 'dip'
                elif mode != 'blind':
                    # go a bit deeper before shifting
                    robot.move_to(np.array([tgt[0] + ox, tgt[1] + oy, max(z_act - 0.008, z_floor)]), vmax=VMAX_FINE, tol_z=0.002, max_iter=3)
                    n = red_near(robot.front, px, radius)
                    if n >= max(RED_MIN_PIX, base + 6):
                        lit, how = True, 'dip-deep'
            if mode == 'blind':
                break
        if not lit and z_act is None and k == 0:
            # never saw red at the first target: assume red detection is unusable,
            # use the lowest safe height as the touch height from now on
            z_act = max(robot.pos()[2], z_floor + 0.004)
            mode = 'blind'
            LOG('no red seen at first target -> blind mode, z_act', round(z_act, 4))
        rec.update(lit=lit, how=how, final=robot.pos().tolist(), z_act=z_act, step=robot.step_id,
                   red=red_near(robot.front, px, radius))
        LOG('target', k, c, 'lit', lit, how, 'pos', np.round(robot.pos(), 4), 'z_act', None if z_act is None else round(z_act, 4),
            'step', robot.step_id, 'ended', robot.ended, 'success', robot.success)
        log_rows.append(rec)
        if robot.success or robot.ended:
            break
    # after the pattern: lift slightly and wait a couple of steps for the verdict
    if not robot.ended:
        cur = robot.pos()
        robot.move_to(np.array([cur[0], cur[1], cur[2] + 0.02]), vmax=VMAX_FINE, max_iter=2)
        robot.hold(2)
    if not robot.ended and not robot.success:
        # re-touch the final target once more (in case the last touch was not registered)
        c = pattern[-1]
        tgt = pts[c]
        zt = (z_act if z_act is not None else z_top) - 0.004
        robot.move_to(np.array([tgt[0], tgt[1], max(zt, z_floor)]), vmax=VMAX_FINE, max_iter=4)
        robot.hold(2)
    summary['targets'] = log_rows
    summary['z_act'] = z_act
    summary['mode'] = mode
    if not robot.ended:
        robot.status()
    LOG('execution finished; success flag', robot.success, 'ended', robot.ended)
