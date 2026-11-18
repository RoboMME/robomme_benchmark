"""PatternLock controller (RAX native main(ctx)).

Pipeline / transition table
  P0 OBSERVE      robomme_observe at step 0 -> meta, conditioning manifest, exec images.
                  fail -> retry observe (3x) -> abort (return).
  P1 DECODE       video front frames -> video grid (axis aligned) -> red-onset sequence of (row,col).
                  fail -> abort (nothing sensible to draw).
  P2 LOCALIZE     exec front image blobs -> rectify with video homography -> 5x5 lattice indices
                  -> world points (pixel_to_position on blob centroids) -> world lattice fit.
                  fail (few points) -> retry with fallback back-projection -> abort.
  P3 TOUCH(i)     for each pattern cell: lift to transit height, move over cell, fast descend to
                  hover, slow descend checking the cell turns red in the front image (or episode
                  ends). Success -> next cell. No activation at max depth -> xy search (4 offsets)
                  -> accept and continue.
  P4 DONE         episode ended / pattern exhausted -> return.
Never raises out of main: every phase is wrapped, errors logged.
"""
import math
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_utils as iu      # noqa: E402
import vision              # noqa: E402
import pattern             # noqa: E402
import geometry            # noqa: E402

VERSION = 'patternlock-v1'

# ---- tunables (no live testing possible: R=1; values chosen with margin, see memory/PROGRESS.md) ----
POS_SCALE = 0.05          # m per unit action (documented)
ROT_SCALE = 0.5           # rad per unit action (documented)
FAST_STEP = 0.018         # m per row in transit
MID_STEP = 0.010          # m per row for approach descent
SLOW_STEP = 0.003         # m per row in final (touch) descent
ROWS_PER_CALL = 12        # max rows per robomme_step call in transit
SLOW_ROWS = 2             # rows per call during touch descent (then check for red)
TRANSIT_CLEAR = 0.045     # m above estimated target top while travelling
HOVER_CLEAR = 0.015       # m above estimated top where slow descent starts (first target)
PRESS_DEPTH_FIRST = 0.022 # m below estimated top allowed on first (calibration) touch
PRESS_AFTER_ACT = 0.008   # m below learnt activation height allowed on later touches
POS_TOL = 0.004           # m position tolerance for transit waypoints
GRIPPER = -1.0            # ignored by stick tools
ROT_HOLD_THRESH = 0.08    # rad orientation drift before correcting
WALL_LIMIT = 1150.0       # s controller wall budget (runtime limit 1200)


def rotvec_to_R(r):
    r = np.asarray(r, float); th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2; c = min(1.0, max(-1.0, c)); th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        w, V = np.linalg.eigh((R + np.eye(3)) / 2)
        return V[:, -1] * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return v * th


class Episode:
    def __init__(self, ctx, log):
        self.ctx = ctx; self.log = log
        self.meta = {}; self.imgs = {}; self.state = None; self.step_id = 0
        self.ended = False; self.complete = False
        self.R0 = None; self.rot_hold = True; self.last_rot_err = None
        self.n_calls = 0; self.n_rows = 0; self.t0 = time.time()
        self.remaining = None
        self.front_ref = None; self._sel_logged = False
        self.frozen = 0

    # ---------------- I/O ----------------
    def _absorb(self, result, tag):
        meta = iu.meta_of(result)
        self.meta = meta
        st = iu.get_state(meta)
        if st is not None:
            self.state = st
        sid = iu.get_step_id(meta)
        if sid is not None:
            self.step_id = sid
        e, c = iu.episode_flags(meta)
        self.ended = self.ended or e; self.complete = self.complete or c
        rem = iu.remaining_steps(meta)
        if rem is not None:
            self.remaining = rem
        imgs = iu.images_from_result(result, meta, self.log)
        if imgs and any(isinstance(k, tuple) for k in imgs):
            inl = imgs.get('_inline_rgb') or []
            if self.front_ref is not None and not imgs.get('_inline_labeled') and len(inl) >= 2:
                d = [float(np.mean(np.abs(a.astype(np.int16) - self.front_ref.astype(np.int16)))) if a.shape == self.front_ref.shape else 1e9 for a in inl]
                j = int(np.argmin(d))
                imgs[('front', 'rgb')] = inl[j]
                others = [a for i, a in enumerate(inl) if i != j]
                if others:
                    imgs[('wrist', 'rgb')] = others[0]
                if not self._sel_logged:
                    self.log('unlabeled inline images: front chosen by similarity, diffs', [round(x, 1) for x in d])
                    self._sel_logged = True
            self.imgs = imgs
        return meta

    def observe(self):
        r = self.ctx.call('robomme_observe')
        self.n_calls += 1
        return self._absorb(r, 'observe')

    def step(self, rows):
        if self.ended:
            return None
        if self.remaining is not None and self.remaining >= 1:
            rows = rows[:self.remaining]
        rows = [[float(min(1.0, max(-1.0, x))) for x in row] for row in rows]
        p_before = self.pos()
        r = self.ctx.call('robomme_step', actions=rows)
        self.n_calls += 1; self.n_rows += len(rows)
        meta = self._absorb(r, 'step')
        p_after = self.pos()
        cmd = float(np.linalg.norm(np.sum([row[:3] for row in rows], axis=0))) * POS_SCALE
        if p_before is not None and p_after is not None and cmd > 0.002:
            if np.linalg.norm(p_after - p_before) < 0.0003:
                self.frozen += 1
            else:
                self.frozen = 0
        if self.frozen >= 40:
            self.log('WATCHDOG: TCP has not moved for 40 commanded calls; treating episode as over')
            self.ended = True
        if self.n_calls <= 3:
            self.log.dump('step_reply_%d.json' % self.n_calls, meta)
        return meta

    def pos(self):
        return None if self.state is None else self.state[:3].copy()

    def rot_correction(self):
        """World-frame rotation delta (action units) bringing orientation back to R0."""
        if not self.rot_hold or self.R0 is None or self.state is None:
            return [0.0, 0.0, 0.0]
        R = rotvec_to_R(self.state[3:6])
        err = R_to_rotvec(self.R0 @ R.T)
        n = np.linalg.norm(err)
        if self.last_rot_err is not None and self.last_rot_err > ROT_HOLD_THRESH and n > self.last_rot_err + 0.02:
            self.log('orientation correction seems to diverge (%.3f -> %.3f rad); disabling' % (self.last_rot_err, n))
            self.rot_hold = False
            return [0.0, 0.0, 0.0]
        was = self.last_rot_err
        self.last_rot_err = n
        if n < ROT_HOLD_THRESH:
            return [0.0, 0.0, 0.0]
        corr = err * min(1.0, 0.04 / n) if n > 0.04 else err
        return list(corr / ROT_SCALE)

    def wall(self):
        return time.time() - self.t0

    # ---------------- motion ----------------
    def move_to(self, target, step_max, tol=POS_TOL, max_calls=12, rows_per_call=ROWS_PER_CALL, why=''):
        """Closed-loop straight-line move of the TCP to target (world xyz)."""
        target = np.asarray(target, float)
        for it in range(max_calls):
            if self.ended:
                return False
            p = self.pos()
            if p is None:
                self.observe(); p = self.pos()
                if p is None:
                    self.log('no state available'); return False
            err = target - p; d = float(np.linalg.norm(err))
            if d < tol:
                return True
            nsteps = max(1, int(math.ceil(d / step_max)))
            per = err / nsteps
            n = min(nsteps, rows_per_call)
            rc = self.rot_correction()
            rows = []
            for k in range(n):
                rot = rc if k == 0 else [0.0, 0.0, 0.0]
                rows.append(list(per / POS_SCALE) + rot + [GRIPPER])
            self.step(rows)
        p = self.pos()
        ok = p is not None and np.linalg.norm(target - p) < 2.5 * tol
        if not ok:
            self.log('move_to %s did not converge: target %s pos %s' % (why, np.round(target, 4), None if p is None else np.round(p, 4)))
        return ok


def load_conditioning(ep, log):
    """Return (video_front_frames list, exec_frame_files or None)."""
    man, mpath = iu.manifest_from_meta(ep.meta)
    if man is None:
        log('no frame_sequence manifest in observe reply; keys:', list(ep.meta.keys()))
        return [], None
    log('manifest', mpath, 'frames', len(man.get('frames', [])))
    iu.MANIFEST_DIR[0] = os.path.dirname(mpath)
    frames = man.get('frames', [])
    frames = sorted(frames, key=lambda f: f.get('frame_index', 0))
    vid = [f for f in frames if f.get('phase') == 'video_demo']
    exe = [f for f in frames if f.get('phase') == 'execution']
    ims = []
    for f in vid:
        p = None
        for fl in f.get('files', []):
            if str(fl.get('camera', '')).lower() == 'front' and str(fl.get('kind', 'rgb')).lower() == 'rgb':
                p = fl.get('path')
        if p is None:
            continue
        try:
            ims.append(np.array(iu.load_path(p).convert('RGB')))
        except Exception as e:
            log('frame load failed', p, e)
    exec_files = exe[-1].get('files') if exe else None
    return ims, exec_files


def localize_grid(ep, H, log):
    """Find exec grid cells. Returns dict with world[a][b] (xyz), img[a][b] (u,v), masks, z_top."""
    front = ep.imgs.get(('front', 'rgb'))
    if front is None:
        raise RuntimeError('no front rgb at execution start')
    best = None
    for params in ({}, {'diff_min': 30}, {'diff_min': 50}, {'sat_max': 70}):
        bl = vision.merge_close(pattern.blobs_with(front, **params))
        if len(bl) < 12:
            continue
        pts = np.array([(u, v) for u, v, a in bl])
        try:
            G = vision.fit_grid_plane(vision.apply_h(H, pts))
        except Exception as e:
            log('fit_grid_plane failed', params, e); continue
        score = G['count'] - 10 * G['rms']
        if best is None or score > best[0]:
            best = (score, G, pts, params)
        if G['count'] == 25 and G['rms'] < 0.06:
            break
    if best is None:
        raise RuntimeError('exec grid not found')
    _, G, pts, params = best
    log('exec grid: count %d rms %.3f theta %.1f deg params %s' % (G['count'], G['rms'], math.degrees(G['theta']), params))
    Hi = np.linalg.inv(H)
    plane_pred = G['grid']  # [a,b] plane coords
    # choose one blob per cell (closest to prediction)
    planeW = vision.apply_h(H, pts)
    cell_uv = {}
    for (a, b), ks in G['members'].items():
        k = min(ks, key=lambda k: np.linalg.norm(planeW[k] - plane_pred[a, b]))
        cell_uv[(a, b)] = tuple(pts[k])
    # image positions for all cells (measured or predicted)
    img = {}
    for a in range(5):
        for b in range(5):
            img[(a, b)] = cell_uv.get((a, b), tuple(vision.apply_h(Hi, [plane_pred[a, b]])[0]))
    # world points
    keys = sorted(cell_uv.keys())
    uvs = [cell_uv[k] for k in keys]
    W = geometry.pixels_to_world(ep.ctx, ep.step_id, 'front', uvs, log, radius=1)
    ok = [(k, w) for k, w in zip(keys, W) if w is not None]
    if len(ok) < 10:
        log('pixel_to_position gave %d points; trying own back-projection' % len(ok))
        depth = ep.imgs.get(('front', 'depth'))
        W = geometry.backproject(ep.meta, depth, 'front', uvs)
        ok = [(k, w) for k, w in zip(keys, W) if w is not None]
    if len(ok) < 10:
        raise RuntimeError('too few world points (%d)' % len(ok))
    def fit_world(ok):
        K = np.array([k for k, _ in ok], float); Wp = np.array([w for _, w in ok])
        z_top = float(np.median(Wp[:, 2]))
        keep = np.abs(Wp[:, 2] - z_top) < 0.03
        coef = None; res = None
        for it in range(4):
            X = np.hstack([np.ones((keep.sum(), 1)), K[keep]])
            coef, *_ = np.linalg.lstsq(X, Wp[keep, :2], rcond=None)
            pred = np.hstack([np.ones((len(K), 1)), K]) @ coef
            res = np.linalg.norm(pred - Wp[:, :2], axis=1)
            med = np.median(res[keep])
            newkeep = keep & (res < max(0.008, 4 * med))
            if newkeep.sum() == keep.sum() or newkeep.sum() < 8:
                break
            keep = newkeep
        O, A1, A2 = coef
        rms = float(np.sqrt((res[keep] ** 2).mean()))
        z_top = float(np.median(Wp[keep, 2]))
        sp1, sp2 = float(np.linalg.norm(A1)), float(np.linalg.norm(A2))
        cosang = abs(float(np.dot(A1, A2) / (sp1 * sp2 + 1e-9)))
        problems = []
        if not (0.015 < sp1 < 0.12 and 0.015 < sp2 < 0.12):
            problems.append('spacing')
        if abs(sp1 - sp2) > 0.25 * max(sp1, sp2):
            problems.append('anisotropic')
        if cosang > 0.26:
            problems.append('non-orthogonal')
        if rms > 0.008:
            problems.append('rms')
        if ep.state is not None and not (z_top < ep.state[2] - 0.01):
            problems.append('targets-not-below-tcp')
        log('world lattice: O %s A1 %s A2 %s |A1| %.4f |A2| %.4f cos %.3f rms %.4f used %d/%d z_top %.4f problems %s' %
            (np.round(O, 4), np.round(A1, 4), np.round(A2, 4), sp1, sp2, cosang, rms, keep.sum(), len(K), z_top, problems))
        return O, A1, A2, z_top, (sp1, sp2), rms, problems

    O, A1, A2, z_top, (sp1, sp2), rms, problems = fit_world(ok)
    if problems:
        depth = ep.imgs.get(('front', 'depth'))
        W2 = geometry.backproject(ep.meta, depth, 'front', uvs)
        ok2 = [(k, w) for k, w in zip(keys, W2) if w is not None]
        if len(ok2) >= 10:
            log('trying own back-projection because of', problems)
            alt = fit_world(ok2)
            if len(alt[6]) < len(problems):
                O, A1, A2, z_top, (sp1, sp2), rms, problems = alt
                log('using own back-projection')
    world = {}
    for a in range(5):
        for b in range(5):
            xy = O + a * A1 + b * A2
            world[(a, b)] = np.array([xy[0], xy[1], z_top])
    # per-cell measured points where trustworthy (blend: prefer lattice prediction, robust)
    sps = []
    for a in range(5):
        for b in range(5):
            u, v = img[(a, b)]
            if a < 4:
                u2, v2 = img[(a + 1, b)]
            else:
                u2, v2 = img[(a - 1, b)]
            sps.append(max(6.0, float(np.hypot(u2 - u, v2 - v))))
    centers = [img[(a, b)] for a in range(5) for b in range(5)]
    masks = pattern.cell_masks(centers, sps)
    mask_of = {(a, b): masks[a * 5 + b] for a in range(5) for b in range(5)}
    return dict(world=world, img=img, mask=mask_of, z_top=z_top, spacing=(sp1, sp2), rms=rms)


def cell_red(ep, grid, key):
    front = ep.imgs.get(('front', 'rgb'))
    if front is None:
        return 0
    R = vision.redness(front)
    return int(R[grid['mask'][key]].sum())


def all_red(ep, grid):
    front = ep.imgs.get(('front', 'rgb'))
    if front is None:
        return {}
    R = vision.redness(front)
    return {k: int(R[m].sum()) for k, m in grid['mask'].items()}


def descend_and_check(ep, grid, key, xy, z_min, base, log):
    """Slow descent at xy until the cell lights (front image), contact stall, or z_min.
    Returns ('red'|'ended'|'stall'|'bottom', z)."""
    stall = 0; last_z = None
    for k in range(45):
        if ep.ended:
            return 'ended', None
        p = ep.pos()
        red = cell_red(ep, grid, key)
        if red >= pattern.ON_MIN and red >= base + pattern.ON_OVER_BASE:
            log('  cell %s ACTIVATED red=%d z=%.4f step=%s' % (key, red, p[2], ep.step_id))
            return 'red', float(p[2])
        if p[2] <= z_min + 0.0005:
            return 'bottom', float(p[2])
        if last_z is not None and last_z - p[2] < 0.0007:
            stall += 1
        else:
            stall = 0
        if stall >= 3:
            return 'stall', float(p[2])
        last_z = p[2]
        dz = -min(SLOW_STEP, max(0.0008, p[2] - z_min))
        exy = (np.asarray(xy) - p[:2]) * 0.5
        row = [exy[0] / POS_SCALE / SLOW_ROWS, exy[1] / POS_SCALE / SLOW_ROWS, dz / POS_SCALE, 0, 0, 0, GRIPPER]
        ep.step([row] * SLOW_ROWS)
    if ep.ended:
        return 'ended', None
    p = ep.pos()
    red = cell_red(ep, grid, key)
    if red >= pattern.ON_MIN and red >= base + pattern.ON_OVER_BASE:
        return 'red', float(p[2])
    return 'timeout', float(p[2])


def touch_cell(ep, grid, key, cal, log, idx):
    """Touch one cell. cal: dict with learnt 'z_act'. Returns (activated, why).
    Contact stall near the target top counts as a touch (no re-press, to avoid double touches);
    the xy search runs only when the descent found no contact at target height."""
    w = grid['world'][key]
    z_top = grid['z_top']
    z_tr = z_top + TRANSIT_CLEAR + max(0.0, (cal.get('z_act') or z_top) - z_top - 0.005)
    p = ep.pos()
    if p is not None and p[2] < z_tr - 0.005:
        ep.move_to([p[0], p[1], z_tr], FAST_STEP, max_calls=4, why='lift')
    if ep.ended:
        return False, 'ended'
    ep.move_to([w[0], w[1], z_tr], FAST_STEP, max_calls=8, why='transit')
    if ep.ended:
        return False, 'ended'
    z_ref = cal.get('z_act')
    if z_ref is not None:
        z_start = z_ref + 0.006
        z_min = z_ref - PRESS_AFTER_ACT
    else:
        z_start = z_top + HOVER_CLEAR
        z_min = z_top - PRESS_DEPTH_FIRST
    ep.move_to([w[0], w[1], z_start], MID_STEP, tol=0.003, max_calls=6, why='approach')
    if ep.ended:
        return False, 'ended'
    base = cell_red(ep, grid, key)
    offsets = [(0.0, 0.0), (0.004, 0.0), (-0.004, 0.0), (0.0, 0.004), (0.0, -0.004)]
    for oi, (ox, oy) in enumerate(offsets):
        xy = (w[0] + ox, w[1] + oy)
        if oi > 0:
            log('  cell %s: no contact/activation, xy search offset %s' % (key, (ox, oy)))
            p = ep.pos()
            ep.move_to([p[0], p[1], z_start], MID_STEP, tol=0.003, max_calls=3, why='search-lift')
            ep.move_to([xy[0], xy[1], z_start], MID_STEP, tol=0.002, max_calls=3, why='search-xy')
            if ep.ended:
                return False, 'ended'
        res, z = descend_and_check(ep, grid, key, xy, z_min, base, log)
        if res == 'ended':
            return True, 'ended'
        if res == 'red':
            if cal.get('z_act') is None:
                cal['z_act'] = z
                log('  learnt activation height z_act=%.4f (z_top est %.4f)' % (z, z_top))
            return True, 'red'
        if res == 'stall':
            if z_ref is not None:
                near_top = abs(z - z_ref) < 0.010
            else:
                near_top = z_top - 0.008 < z < z_top + 0.035   # first touch: contact between table level and +3.5 cm
            log('  descent stalled at z=%.4f (near target top: %s)' % (z, near_top))
            if near_top:
                if cal.get('z_act') is None:
                    cal['z_act'] = z
                    log('  learnt contact height z_act=%.4f' % z)
                # give the renderer a moment: one hold row, then check red once more
                ep.step([[0, 0, -0.02, 0, 0, 0, GRIPPER]])
                red = cell_red(ep, grid, key)
                log('  contact touch accepted (red=%d)' % red)
                return True, 'contact'
        log('  descent result %s at z=%s' % (res, z))
        if oi >= 2 and cal.get('z_act') is None:
            z_min -= 0.006
    return False, 'no-activation'


def run(ctx, log):
    ep = Episode(ctx, log)
    log(VERSION, 'start')
    # P0 observe
    for attempt in range(3):
        try:
            ep.observe(); break
        except Exception as e:
            log('observe failed', repr(e)[:300]); time.sleep(1.0)
    log.dump('observe_reply.json', ep.meta)
    log('state', None if ep.state is None else np.round(ep.state, 4).tolist(), 'step', ep.step_id,
        'images', [k for k in ep.imgs.keys() if isinstance(k, tuple)], 'remaining', ep.remaining)
    log('instruction:', iu.find_key(ep.meta, ('instruction',)))
    if ep.state is not None:
        ep.R0 = rotvec_to_R(ep.state[3:6])
    # P1 decode
    vid, exec_files = load_conditioning(ep, log)
    log('video frames loaded:', len(vid))
    if exec_files:
        ex = iu.images_from_files(exec_files)
        for k, v in ex.items():
            ep.imgs.setdefault(k, v)
        if ('front', 'rgb') in ex:
            ep.imgs[('front', 'rgb')] = ex[('front', 'rgb')]
    if not vid:
        log('ABORT: no conditioning video frames'); return
    dec = pattern.decode(vid, log=log)
    seq = dec['sequence']
    log('PATTERN (row,col):', seq)
    if not seq:
        log('ABORT: empty pattern'); return
    g = dec['grid']
    H = vision.homography(g.reshape(-1, 2), np.array([(c, r) for r in range(5) for c in range(5)], float))
    # P2 localize
    ep.front_ref = ep.imgs.get(('front', 'rgb'))
    grid = localize_grid(ep, H, log)
    if ep.state is not None:
        log('initial TCP %s ; z above est. target top: %.4f' % (np.round(ep.state[:3], 4), ep.state[2] - grid['z_top']))
    # P3 touch sequence (second pass only if the episode is still running after the first)
    cal = {}
    for pass_idx in range(2):
        if ep.ended or ep.complete:
            break
        if pass_idx == 1:
            used = ep.n_rows
            if ep.wall() > 600 or (ep.remaining is not None and ep.remaining < used + 20):
                log('no budget for a second pass (wall %.0f, remaining %s, pass used %d rows)' % (ep.wall(), ep.remaining, used))
                break
            log('PASS 2: episode still running after full pattern; repeating the pattern once')
        for i, (r, c) in enumerate(seq):
            if ep.ended:
                break
            if ep.wall() > WALL_LIMIT:
                log('wall budget exhausted'); break
            key = (c, r)   # lattice a = column axis, b = row axis
            log('target %d/%d cell(row=%d,col=%d) world %s step=%s wall=%.0f' %
                (i + 1, len(seq), r, c, np.round(grid['world'][key], 4), ep.step_id, ep.wall()))
            ok, why = False, 'not-run'
            for attempt in range(2):
                try:
                    ok, why = touch_cell(ep, grid, key, cal, log, i)
                    break
                except Exception as e:
                    log('touch_cell exception (attempt %d)' % attempt, repr(e)[:300]); log(traceback.format_exc()[-1500:])
                    ok, why = False, 'exception'
                    try:
                        ep.observe()
                    except Exception as e2:
                        log('observe after exception failed', repr(e2)[:200])
                    if ep.ended:
                        break
            lit = {k: v for k, v in all_red(ep, grid).items() if v >= pattern.ON_MIN}
            log('  result ok=%s why=%s lit=%s ended=%s complete=%s' % (ok, why, lit, ep.ended, ep.complete))
    # P4 done: lift slightly if episode still running (harmless)
    log('DONE ended=%s complete=%s calls=%d rows=%d step=%s wall=%.0f' %
        (ep.ended, ep.complete, ep.n_calls, ep.n_rows, ep.step_id, ep.wall()))
    if not ep.ended:
        try:
            p = ep.pos()
            ep.move_to([p[0], p[1], grid['z_top'] + TRANSIT_CLEAR], FAST_STEP, max_calls=2, why='final-lift')
            log('after final lift ended=%s complete=%s' % (ep.ended, ep.complete))
        except Exception as e:
            log('final lift failed', e)


def main(ctx):
    log = iu.Logger()
    try:
        run(ctx, log)
    except Exception as e:
        log('FATAL', repr(e)[:500]); log(traceback.format_exc()[-3000:])
