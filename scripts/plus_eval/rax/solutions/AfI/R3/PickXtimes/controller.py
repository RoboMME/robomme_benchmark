"""PickXtimes controller v2 (moving-target aware).

Transition table
  INIT     : observe step 0 -> parse instruction (colour, N); localize cube, button, others (wrist top-down,
             front fallback).
  CALIB    : hold at reset pose, watch the target -> mode static | loop (period T, loop L(m)) | linear;
             no target seen -> mode static at step-0 estimate (or abort if none).
  CYCLE k  : GRASP (hover over cube, wrist re-locate, align yaw, descend, close, lift, finger check)
               ok -> PLACE ; fail -> reopen, re-locate (look pose if not seen), retry (<=4 per cycle)
             PLACE static: lower on target xy, release, rise, verify cube near target.
             PLACE loop  : resync phase, pick intercept point P=L(m_now+D) reachable in time, go above P,
                           descend to drop height, wait for arrival, release 3 steps early, rise,
                           verify target stationary with cube on it (=> mark stopped, k+=1).
               fail -> GRASP again for the same k (k not counted)
  PRESS    : after N verified placements: closed gripper over button, staged descent until the episode
             ends (task_complete).  not ended -> re-localize button, retry with offsets (<=3).
  END      : episode_ended -> return.
"""
import json, math, os, re, time, traceback
import numpy as np
import rlib
import track
from rlib import log

VERSION = 'pickx-v2'

NUM_WORDS = {'once': 1, 'twice': 2, 'thrice': 3, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
             'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12}
COLORS = ['green', 'red', 'blue', 'yellow']
LOOK_POSE = (-0.06, 0.0, 0.45)     # similar to reset pose (ep1 reset: -0.0605, 0, 0.521)
ROBOT_BASE_X = -0.45               # button/target candidates with x below this are the robot base
DROP_Z = 0.034                     # TCP z for release over a moving target (cube bottom ~0.014, target top ~0.008)
RELEASE_LEAD = 3                   # steps: open gripper this many steps before predicted arrival
SPEED = 0.008                      # m/step conservative travel estimate for move_to (measured ~0.009-0.011)
CLEAR_OTHERS = 0.065               # m: intercept point clearance to other cubes / button
REACH_X = (-0.36, 0.20)            # intercept point x range (tested -0.30..0.08)
REACH_Y = (-0.30, 0.30)


def parse_instruction(text):
    t = (text or '').lower()
    color = None
    m = re.search(r'pick up the (\w+) cube', t)
    if m and m.group(1) in COLORS:
        color = m.group(1)
    if color is None:
        for c in COLORS:
            if c + ' cube' in t:
                color = c
                break
    n = None
    m = re.search(r'(\d+)\s*times', t)
    if m:
        n = int(m.group(1))
    if n is None:
        m = re.search(r'(\w+)\s+times', t)
        if m and m.group(1) in NUM_WORDS:
            n = NUM_WORDS[m.group(1)]
    if n is None:
        for w in ('once', 'twice', 'thrice'):
            if re.search(r'\b' + w + r'\b', t):
                n = NUM_WORDS[w]
    return color, n


def height_map(obs, cam):
    K = obs.K[cam]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d = obs.depth[cam]
    vv, uu = np.mgrid[0:d.shape[0], 0:d.shape[1]].astype(float)
    pc = np.stack([(uu - cx) / fx * d, -(vv - cy) / fy * d, -d], -1)
    Rm = rlib.quat_to_R(obs.pose[cam]['quaternion_wxyz'])
    t = np.array(obs.pose[cam]['position_m'])
    pw = pc @ Rm.T + t
    pw[d <= 0] = np.nan
    return pw


def find_button(obs, cam, exclude_xy):
    P = height_map(obs, cam)
    rgb = obs.rgb[cam].astype(int)
    sat = rgb.max(-1) - rgb.min(-1)
    z = P[..., 2]
    with np.errstate(invalid='ignore'):
        mask = (z > 0.008) & (z < 0.07) & (sat < 45) & (P[..., 0] > ROBOT_BASE_X)
    for (ex, ey, r) in exclude_xy:
        with np.errstate(invalid='ignore'):
            mask &= ~(np.hypot(P[..., 0] - ex, P[..., 1] - ey) < r)
    bl = rlib.blobs(mask, 15)
    best = None
    for n, cu, cv, pix in bl[:8]:
        pts = P[pix[:, 0], pix[:, 1]]
        pts = pts[~np.isnan(pts).any(1)]
        if len(pts) < 15:
            continue
        zmax = np.percentile(pts[:, 2], 95)
        ext = pts[:, :2].max(0) - pts[:, :2].min(0)
        if zmax < 0.018 or ext.max() > 0.15:
            continue
        cap = pts[pts[:, 2] > zmax - 0.006]
        c = np.median(cap, 0)
        c[2] = zmax
        if best is None or n > best[0]:
            best = (n, c, n)
    if best is None:
        return None
    return best[1], best[2]


def other_cubes(obs, cam, color):
    out = []
    for c in COLORS:
        if c == color:
            continue
        r = rlib.locate_top(obs, cam, c, 20)
        if r is not None and r[0][2] > 0.02:     # table-coloured false positives sit at z~0
            out.append(r[0])
    return out


def perceive_scene(obs, color):
    res = {}
    for cam in ('wrist', 'front'):
        if 'target' not in res:
            t = track.detect_target(obs, cam)
            if t is not None:
                res['target'] = np.array([t[0][0], t[0][1], t[3]]); res['target_cam'] = cam
        if 'cube' not in res:
            c = rlib.locate_top(obs, cam, color, 15)
            if c is not None and c[0][2] > 0.02:
                res['cube'] = c[0]; res['cube_cam'] = cam
    for cam in ('wrist', 'front'):
        if 'button' in res:
            break
        cubes = [rlib.locate_top(obs, cam, c, 15) for c in COLORS]
        ex = [(c[0][0], c[0][1], 0.04) for c in cubes if c is not None]
        if 'target' in res:
            ex.append((res['target'][0], res['target'][1], 0.06))
        b = find_button(obs, cam, ex)
        if b is not None:
            res['button'] = b[0]; res['button_cam'] = cam
    res['others'] = other_cubes(obs, 'wrist', color)
    return res


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.R = rlib.Robot(ctx)
        self.trk = track.Tracker()
        self.R.on_obs = self.trk.add
        self.t0 = time.time()
        self.events = []

    def ev(self, **kw):
        kw['t'] = round(time.time() - self.t0, 2)
        kw['step'] = self.R.last.step_id if self.R.last is not None else None
        self.events.append(kw)
        log('[ev]', json.dumps(kw, default=lambda x: np.asarray(x).round(4).tolist()))

    def save(self):
        try:
            os.makedirs('/workspace/output', exist_ok=True)
            with open('/workspace/output/pickx_events.json', 'w') as f:
                json.dump({'events': self.events, 'track': self.trk.hist[-2000:]}, f,
                          default=lambda x: np.asarray(x).round(4).tolist())
        except Exception:
            pass

    def look(self):
        R = self.R
        R.move_to(LOOK_POSE, grip=R.grip)
        rlib.rotate_yaw(R, 0.0)
        return R.hold(3)

    # ------------------------------------------------------------------
    def run(self):
        R = self.R
        o = R.observe()
        instr = o.meta.get('instruction', '')
        color, n_rep = parse_instruction(instr)
        self.ev(phase='init', version=VERSION, instruction=instr, color=color, n=n_rep, limits=o.limits)
        if color is None:
            color = 'green'
        if n_rep is None:
            n_rep = 1
        scene = perceive_scene(o, color)
        self.ev(phase='perceive', **{k: v for k, v in scene.items()})
        self.color = color
        self.scene = scene
        # ---- target motion calibration (arm still at reset pose) ----
        ok = self.trk.calibrate(R)
        if not ok:
            if 'target' in scene:
                self.trk.mode = 'static'
                self.trk.static_xy = np.asarray(scene['target'][:2])
            else:
                o = self.look()
                self.trk.calib_scale(o)
                if self.trk.add(o) is None:
                    self.ev(phase='abort', why='no target found')
                    return
                self.trk.mode = 'static'
                self.trk.static_xy = np.array(self.trk.hist[-1][1:3])
        self.ev(phase='calibrated', mode=self.trk.mode, T=self.trk.T, static_xy=self.trk.static_xy)
        if 'cube' not in scene or 'button' not in scene:
            o = self.look()
            s2 = perceive_scene(o, color)
            for k, v in s2.items():
                scene.setdefault(k, v)
            self.ev(phase='perceive2', **{k: v for k, v in scene.items()})
        cube_xy = scene['cube'][:2] if 'cube' in scene else self.trk.predict(R.last.step_id)

        done = 0
        attempts = 0
        while done < n_rep and not R.ended and attempts < n_rep + 8:
            attempts += 1
            rem = R.last.limits.get('remaining_episode_steps', 9999)
            if rem < 120:
                self.ev(phase='budget', why='few steps left', rem=rem)
                break
            g = None
            for gtry in range(4):
                g = rlib.grasp_cube(R, color, cube_xy, f'#{done+1}.{gtry}', others=scene.get('others'))
                if R.ended or g['ok']:
                    break
                R.hold(6, grip=-1)
                o = R.move_to([R.last.state[0], R.last.state[1], rlib.HOVER_Z], grip=-1)
                c = rlib.locate_top(o, 'wrist', color, 15)
                if c is None:
                    o = self.look()
                    c = rlib.locate_top(o, 'wrist', color, 15) or rlib.locate_top(o, 'front', color, 15)
                if c is not None:
                    cube_xy = c[0][:2]
                self.ev(phase='grasp_retry', k=done + 1, why=g.get('why', 'fingers'), cube_xy=cube_xy)
            if R.ended:
                break
            if not g or not g['ok']:
                self.ev(phase='grasp_failed', k=done + 1)
                continue
            if not self.trk.moving:
                self.trk.mark_resumed(g['lift_step'] + 2)
            if self.trk.mode in ('loop', 'linear'):
                p = self.place_moving(g['lift_step'])
            else:
                p = self.place_static()
            if R.ended:
                break
            if p['ok']:
                done += 1
                self.ev(phase='cycle_done', k=done, **{k: v for k, v in p.items() if k != 'ok'})
                cube_xy = p['cube'][:2]
            else:
                self.ev(phase='place_failed', k=done + 1, **{k: v for k, v in p.items() if k != 'ok'})
                if self.trk.mode in ('loop', 'linear') and p.get('cube') is not None:
                    g2 = self.guard_after_miss(p['cube'])
                    if g2 is not None and g2.get('ok'):
                        done += 1
                        self.ev(phase='cycle_done_late', k=done, **{k: v for k, v in g2.items() if k != 'ok'})
                        cube_xy = g2['cube'][:2]
                        self.save()
                        continue
                c = p.get('cube')
                if c is None:
                    o = self.look()
                    cc = rlib.locate_top(o, 'wrist', color, 15) or rlib.locate_top(o, 'front', color, 15)
                    c = cc[0] if cc is not None else None
                cube_xy = c[:2] if c is not None else cube_xy
            self.save()
        if R.ended:
            self.ev(phase='end_early', complete=R.complete)
            return
        self.ev(phase='cycles_finished', done=done, n=n_rep)
        self.press_button(scene)
        self.ev(phase='end', complete=R.complete, ended=R.ended)

    # ------------------------------------------------------------------
    def place_static(self):
        R = self.R
        txy = self.trk.static_xy
        p = rlib.place_cube(R, self.color, txy, '')
        o = R.last
        t = track.detect_target(o, 'wrist')
        out = {'cube': p.get('cube'), 'dist': p.get('dist'), 'target_seen': None if t is None else t[0]}
        out['ok'] = bool(p['ok'])
        if p['ok'] and t is not None and p.get('cube') is not None:
            d2 = float(np.linalg.norm(np.asarray(p['cube'][:2]) - t[0]))
            out['dist_seen'] = d2
            if d2 > 0.03:
                out['ok'] = False
        return out

    def _intercept(self):
        """Choose (P, m_P) on the loop reachable before the target gets there."""
        R, trk = self.R, self.trk
        o = R.last
        s = o.step_id
        pos = o.state[:3]
        avoid = [np.asarray(c[:2]) for c in self.scene.get('others', [])]
        if 'button' in self.scene:
            avoid.append(np.asarray(self.scene['button'][:2]))
        m_now = trk.m_at(s)
        horizon = (trk.T + 40) if trk.mode == 'loop' else 60
        for D in range(10, horizon):
            if trk.mode == 'loop':
                P = trk.L(m_now + D)
            else:
                P = trk.predict(s + D)
            if not (REACH_X[0] <= P[0] <= REACH_X[1] and REACH_Y[0] <= P[1] <= REACH_Y[1]):
                continue
            if any(np.linalg.norm(P - a) < CLEAR_OTHERS for a in avoid):
                continue
            dh = float(np.linalg.norm(P - pos[:2]))
            dv = abs(rlib.HOVER_Z - pos[2]) + (rlib.HOVER_Z - DROP_Z)
            est = (dh + dv) / SPEED + 10
            if est + 4 <= D:
                return P, (m_now + D if trk.mode == 'loop' else s + D), D, est
        return None

    def place_moving(self, lift_step):
        R, trk = self.R, self.trk
        if trk.mode == 'loop':
            # collect detections of the target moving away, then lock the phase globally
            for i in range(6):
                if trk.n_good_since(lift_step + 4) >= 4:
                    break
                R.hold(2, grip=1)
            if trk.relock(lift_step + 4) is None:
                trk.resync(lift_step + 4, max_corr=30)
        for ptry in range(3):
            plan = self._intercept()
            if plan is None:
                self.ev(phase='intercept_none')
                return {'ok': False, 'why': 'no intercept', 'cube': None}
            P, mP, D, est = plan
            self.ev(phase='intercept', P=P, D=D, est=est, m_now=trk.m_at(R.last.step_id), tryno=ptry)
            s_plan = R.last.step_id
            R.move_to([P[0], P[1], rlib.HOVER_Z], grip=1)
            if trk.mode == 'loop':
                trk.resync(s_plan)
            R.move_to([P[0], P[1], DROP_Z], grip=1, tol=0.003)

            def remaining():
                s = R.last.step_id
                if trk.mode == 'loop':
                    return (mP - RELEASE_LEAD) - trk.m_at(s)
                return (mP - RELEASE_LEAD) - s
            rem = remaining()
            if rem < -1.5 and ptry < 2:
                self.ev(phase='late', rem=rem)
                R.move_to([P[0], P[1], rlib.HOVER_Z], grip=1)
                continue
            while rem > 0.5 and not R.ended:
                R.hold(int(min(max(1, round(rem)), 4)), grip=1)
                rem = remaining()
            break
        if R.ended:
            return {'ok': False, 'why': 'ended', 'cube': None}
        self.ev(phase='release', rem=rem, tcp=R.last.state[:3], pred=trk.predict(R.last.step_id))
        R.hold(6, grip=-1)
        R.move_to([P[0], P[1], rlib.HOVER_Z], grip=-1)
        return self.verify_moving(P)

    def verify_moving(self, P):
        """Target stationary with the cube on it?  Decide from per-camera target motion vs loop speed.
        If undecided from hover, rise to 0.30 m and look again once."""
        R = self.R
        res = self._verify_once()
        if res.get('verdict') is None and not R.ended:
            o = R.last
            R.move_to([o.state[0], o.state[1], 0.30], grip=-1)
            res = self._verify_once()
            R.move_to([P[0], P[1], rlib.HOVER_Z], grip=-1)
        return res

    def _verify_once(self):
        R, trk = self.R, self.trk
        v = trk.v_loop if trk.v_loop else 0.004
        dets = {'wrist': [], 'front': []}
        cube = None
        verdict, rate = None, None
        for i in range(12):
            o = R.hold(2, grip=-1)
            if R.ended:
                break
            for cam in ('wrist', 'front'):
                t = track.detect_target(o, cam)
                if t is not None:
                    dets[cam].append((o.step_id, t[0]))
            c = rlib.locate_top(o, 'wrist', self.color, 15)
            if c is None:
                c = rlib.locate_top(o, 'front', self.color, 15)
            if c is not None:
                cube = c[0]
            rates = []
            for cam, L in dets.items():
                if len(L) >= 3 and L[-1][0] - L[0][0] >= 6:
                    k = max(1, len(L) // 3)
                    a = np.mean([x[1] for x in L[:k]], 0); b = np.mean([x[1] for x in L[-k:]], 0)
                    sa = np.mean([x[0] for x in L[:k]]); sb = np.mean([x[0] for x in L[-k:]])
                    rates.append(float(np.linalg.norm(b - a) / max(1.0, sb - sa)))
            if rates:
                rate = min(rates)
                if rate < 0.2 * v and i >= 3:
                    verdict = 'stationary'; break
                if rate > 0.5 * v and i >= 3:
                    verdict = 'moving'; break
        allt = dets['wrist'] or dets['front']
        if verdict is None and rate is not None:
            verdict = 'stationary' if rate < 0.35 * v else 'moving'
        if verdict is None or not allt or cube is None:
            return {'ok': False, 'why': 'not seen', 'n_t': len(allt), 'cube': cube, 'verdict': verdict}
        tmean = np.mean([x[1] for x in allt[-3:]], 0)
        dist = float(np.linalg.norm(cube[:2] - tmean))
        ok = verdict == 'stationary' and dist < 0.035 and cube[2] < 0.06
        if verdict == 'stationary' and not ok:
            log(f'[verify] stationary target but cube dist {dist:.3f} z {cube[2]:.3f}')
        if verdict == 'stationary':
            self.trk.mark_stopped(tmean, R.last.step_id)
        return {'ok': bool(ok), 'verdict': verdict, 'rate': rate, 'dist': dist, 'cube': cube, 'target': tmean}

    def guard_after_miss(self, cube):
        """After a missed drop the target may still slide under the resting cube and stop there
        (which the task would count).  Wait until the target is clearly away from the cube, or report
        a late success when it is stationary under it."""
        R = self.R
        for i in range(4):
            if R.ended:
                return None
            v = self.verify_moving(cube[:2])
            if v.get('ok'):
                return v
            t = v.get('target')
            c = v.get('cube')
            if t is None or c is None:
                return None
            d = float(np.linalg.norm(np.asarray(t) - np.asarray(c[:2])))
            self.ev(phase='guard', verdict=v.get('verdict'), dist=d)
            if v.get('verdict') == 'moving' and d > 0.06:
                return v
        return None

    # ------------------------------------------------------------------
    def press_button(self, scene):
        R = self.R
        b = scene.get('button')
        for btry in range(3):
            if R.ended:
                return
            if b is None:
                o = self.look()
                cubes = [rlib.locate_top(o, 'wrist', c, 15) for c in COLORS]
                ex = [(c[0][0], c[0][1], 0.04) for c in cubes if c is not None]
                t = track.detect_target(o, 'wrist')
                if t is not None:
                    ex.append((t[0][0], t[0][1], 0.06))
                bb = None
                for cam in ('wrist', 'front'):
                    r = find_button(o, cam, ex)
                    if r is not None:
                        bb = r[0]
                        break
                if bb is None:
                    self.ev(phase='press', why='button not found')
                    return
                b = bb
            bx, by, bz = float(b[0]), float(b[1]), float(b[2])
            off = [(0, 0), (0.006, 0.0), (-0.006, 0.0)][btry]
            bx += off[0]; by += off[1]
            R.move_to([R.last.state[0], R.last.state[1], max(R.last.state[2], 0.12)], grip=1)
            R.move_to([bx, by, 0.10], grip=1)
            R.hold(4, grip=1)
            R.move_to([bx, by, bz + 0.015], grip=1)
            self.ev(phase='press_start', button=[bx, by, bz], tryno=btry)
            for z in np.arange(bz + 0.008, bz - 0.022, -0.004):
                if R.ended:
                    break
                R.move_to([bx, by, z], grip=1, tol=0.002, max_iter=3)
                R.hold(2, grip=1)
            if R.ended:
                self.ev(phase='pressed', complete=R.complete)
                return
            R.move_to([bx, by, 0.12], grip=1)
            b = None


def main(ctx):
    c = Controller(ctx)
    try:
        c.run()
    except Exception:
        log('[fatal]', traceback.format_exc())
        try:
            if not c.R.ended:
                c.R.hold(2)
        except Exception:
            log('[fatal2]', traceback.format_exc())
    c.save()
    log('[done]', 'complete=', c.R.complete, 'ended=', c.R.ended, 'steps=', c.R.steps,
        'wall=', round(time.time() - c.t0, 1))
