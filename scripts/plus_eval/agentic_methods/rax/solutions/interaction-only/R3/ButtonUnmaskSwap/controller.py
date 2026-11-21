"""ButtonUnmaskSwap controller v2 (fixed-N fused tracker, early presses, high park, fast picks).

Transition table (one control row per robomme_step call so every step gives an RGB-D frame):
  INIT   : step-0 observe -> targets from instruction, cube colours/positions, buttons.  -> PRESS
  PRESS  : for each button: transit(z=0.12) -> descend until stall -> hold -> rise.       -> PARK
           (timeout per button; deadline: all presses before step 170)
  PARK   : move to (-0.06,0,0.45) (out of the front camera's view of the table).          -> WAIT
  WAIT   : zero rows until all container tracks static for STATIC_STEPS (cap step 650).   -> PICK
  PICK   : for each target colour in order: hover, refine top face, align yaw, descend,
           close, lift, (verify width) ; place back unless it is the last target.          -> DONE
  Container tracking runs on every returned frame during PRESS/PARK/WAIT.
  Any exception: logged; the controller continues with the next phase where possible.
"""
import json, time, traceback, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot import meta, reply_images, finger_yaw, rot_err, wrap
from tracker3 import FixedTracker, detect_tops
from scene import parse_targets, initial_scene, initial_cubes_fused
from grasp import top_face

VERSION = 'bus-v2'
VMAX = 0.8            # max per-row translation command (x0.05 m)
KP = 1.0              # rows command full error (arm achieves ~35-40%/step, ep1)
WS_LO = np.array([-0.33, -0.30, 0.0])
WS_HI = np.array([0.15, 0.30, 0.50])  # (0.147,0.161,0.205) was the reach edge in ep1
TRANSIT_Z = 0.12
BASE_XY = np.array([-0.615, 0.0])   # inferred: ep1 reach stop (0.147,0.161,z0.2) is r=0.779 from here
R_MAX = 0.765
PRESS_DEADLINE = 170
INIT_HOLD = 12        # hold still while containers drop
PARK = np.array([-0.12, 0.0, 0.50])   # wrist view: finger blind zone beyond x<-0.31
STATIC_STEPS = 14     # container tracks static this long => shuffle over (reh1: stopped ~173)
STATIC_TOL = 0.004    # metres moved over the window
WAIT_CAP = 260        # 480-step motion limit (PUBLIC_RULES)
HOVER_Z = 0.13
LIFT_Z = 0.16
OUT = '/workspace/output'


class Ctl:
    def __init__(self, ctx):
        self.ctx = ctx
        self.m = None
        self.ims = None
        self.step_id = 0
        self.ended = False
        self.tracker = None
        self.track_on = False
        self.penalties = 0
        self.t0 = time.time()
        os.makedirs(OUT, exist_ok=True)
        self.logf = open(os.path.join(OUT, 'controller_log.txt'), 'a')

    def log(self, *a):
        s = f'[{VERSION} s={self.step_id} t={time.time()-self.t0:.1f}] ' + ' '.join(str(x) for x in a)
        print(s)
        try:
            self.logf.write(s + '\n'); self.logf.flush()
        except Exception:
            pass

    @property
    def state(self):
        return np.array(self.m['state'])

    def observe(self):
        rep = self.ctx.call('robomme_observe')
        self._ingest(rep)

    def _ingest(self, rep):
        m, ims = reply_images(rep)
        if not m:
            return
        prev = self.step_id
        self.m, self.ims = m, ims
        self.step_id = m.get('step_id', self.step_id)
        self.ended = bool(m.get('episode_ended', False))
        return self.step_id - prev

    def step_row(self, row):
        """Send one row; update tracking; return number of steps consumed."""
        if self.ended:
            return 0
        row = [float(np.clip(x, -1, 1)) for x in row]
        rep = self.ctx.call('robomme_step', actions=[row])
        n = self._ingest(rep)
        if n is not None and n > 3:
            self.penalties += 1
            self.log(f'IK PENALTY: step jumped by {n}')
        if self.track_on:
            self.track_update()
        return n or 0

    # ---------------- tracking
    def track_update(self):
        try:
            if self.ims.get(('front', 'rgb')) is None or self.tracker is None:
                return
            dets = detect_tops(self.m, self.ims, self.state)
            tr = self.tracker
            if not tr.initialised:
                if (self.step_id >= 8 and len(dets) >= len(tr.tracks) - 1) or self.step_id >= 13:
                    k = tr.init_from(dets, self.step_id)
                    tr.initialised = True
                    # containers without a step-0 cube (cube hidden at step 0) get an 'unk' track
                    for d in dets:
                        if all(np.linalg.norm(t['xy'] - d['xy']) > 0.035 for t in tr.tracks) and d['n'] >= 40:
                            tr.tracks.append({'label': 'unk', 'xy': d['xy'].copy(), 'v': np.zeros(2), 'miss': 0,
                                              'hist': [], 'reacq': 0, 'seen': 0})
                            self.log('unk track', d['xy'].round(3))
                    self.log(f'tracker init: matched {k}/{len(tr.tracks)} ndet={len(dets)}')
                    self.last_track_step = self.step_id
                return
            full = self.m['camera_poses']['wrist']['position_m'][2] > 0.40
            tr.update(dets, self.step_id, dt=max(1, self.step_id - self.last_track_step), full_view=full)
            self.last_track_step = self.step_id
            if self.step_id % 10 == 0:
                self.log('tracks', ' '.join(f"{t['label'][:3]}:{t['xy'][0]:.3f},{t['xy'][1]:.3f}{'*'+str(t['miss']) if t['miss'] else ''}"
                                             for t in tr.tracks), f'ndet={len(dets)} full={full}')
        except Exception as e:
            self.log('track error', repr(e))

    def containers(self):
        return self.tracker.tracks if self.tracker else []

    # ---------------- motion
    def move(self, target, grip, yaw, tol=0.005, max_steps=60, vmax=VMAX, stall_exit=False):
        target = np.clip(np.asarray(target, float), WS_LO, WS_HI)
        r = np.linalg.norm(target[:2] - BASE_XY)
        if r > R_MAX:
            target[:2] = BASE_XY + (target[:2] - BASE_XY) * (R_MAX / r)
        hist = []
        for k in range(max_steps):
            if self.ended:
                return False
            s = self.state
            e = target - s[:3]
            re = rot_err(s, yaw)
            if np.linalg.norm(e) < tol and np.linalg.norm(re) < 0.05:
                return True
            a = e / 0.05 * KP
            n = np.max(np.abs(a))
            if n > vmax:
                a *= vmax / n
            ra = np.clip(re / 0.5 * 0.5, -0.5, 0.5)
            jumped = self.step_row([a[0], a[1], a[2], ra[0], ra[1], ra[2], grip])
            if jumped > 3:
                return False
            hist.append(self.state[:3].copy())
            if stall_exit and len(hist) > 4 and np.linalg.norm(hist[-1] - hist[-4]) < 0.0015:
                return 'stall'
        return False

    def hold(self, n, grip, yaw=None):
        for _ in range(n):
            if self.ended:
                return
            re = rot_err(self.state, yaw) if yaw is not None else np.zeros(3)
            ra = np.clip(re / 0.5 * 0.5, -0.5, 0.5)
            self.step_row([0, 0, 0, ra[0], ra[1], ra[2], grip])

    # ---------------- phases
    def init_scene(self):
        self.observe()
        instr = self.m.get('instruction', '')
        self.targets = parse_targets(instr)
        K = self.m['camera_intrinsics']['front']['matrix']; P = self.m['camera_poses']['front']
        cubes, btns, whites = initial_scene(self.ims[('front', 'rgb')], self.ims[('front', 'depth')], K, P)
        fused = []
        try:
            fused = [(c[0], c[1]) for c in initial_cubes_fused(self.m, self.ims)]
        except Exception:
            self.log('initial_cubes_fused error', traceback.format_exc())
        if not fused:
            fused = list(cubes.items())
        allc = fused + [('white', w) for w in whites]
        self.buttons = btns
        self.log('instruction:', instr)
        self.log('targets', self.targets, 'cubes', [(l, np.round(x, 3).tolist()) for l, x in allc],
                 'buttons', [b.round(3).tolist() for b in btns])
        self.tracker = FixedTracker(allc)
        self.yaw0 = finger_yaw(self.state)
        self.last_track_step = self.step_id
        self.track_on = True

    def press_buttons(self):
        # hold still until the containers have landed and every track is anchored (reh1: step 10)
        while not self.ended and not self.tracker.initialised and self.step_id < 14:
            self.hold(1, 1, self.yaw0)
        # containers drop at steps ~5-10 (ep1); stay clear and let tracking initialise
        btns = sorted(self.buttons, key=lambda b: np.linalg.norm(b - self.state[:2]))
        if len(btns) > 2:
            btns = btns[:2]
        for i, b in enumerate(btns):
            if self.step_id > PRESS_DEADLINE - 25:
                self.log('press deadline near; skipping button', i)
                break
            self.log(f'press {i} at', b.round(3))
            self.move([b[0], b[1], TRANSIT_Z], 1, self.yaw0, tol=0.012, max_steps=50, vmax=1.0)
            self.move([b[0], b[1], 0.05], 1, self.yaw0, tol=0.006, max_steps=15, vmax=1.0)
            r = self.move([b[0], b[1], 0.0], 1, self.yaw0, tol=0.003, max_steps=15, vmax=0.5, stall_exit=True)
            z = self.state[2]
            for _ in range(3):
                re = rot_err(self.state, self.yaw0); ra = np.clip(re / 0.5 * 0.5, -0.5, 0.5)
                self.step_row([0, 0, -0.3, ra[0], ra[1], ra[2], 1])
            self.log(f'press {i} result={r} z_stall={z:.4f} z_now={self.state[2]:.4f}')
            self.move([b[0], b[1], TRANSIT_Z], 1, self.yaw0, tol=0.02, max_steps=20, vmax=1.0)

    def park_and_wait(self):
        self.move(PARK, -1, self.yaw0, tol=0.02, max_steps=40, vmax=1.0)
        window = []
        while not self.ended and self.step_id < WAIT_CAP:
            self.hold(1, -1, self.yaw0)
            C = self.containers()
            if not C:
                continue
            window.append(np.array([t['xy'] for t in C]))
            if len(window) > STATIC_STEPS:
                window.pop(0)
                moved = np.max(np.linalg.norm(window[-1] - window[0], axis=1))
                if moved < STATIC_TOL and self.step_id > 120:
                    self.log(f'shuffle over (moved {moved:.4f} in {STATIC_STEPS})')
                    break
        # one more full-view detection to re-anchor every track
        try:
            dets = detect_tops(self.m, self.ims, self.state)
            fa = self.tracker.final_assign(dets)
            self.log('final assign', [(l, np.round(a, 3).tolist(), np.round(b, 3).tolist(), round(d, 3), mi) for l, a, b, d, mi in fa])
        except Exception:
            self.log('final assign error', traceback.format_exc())
        self.track_on = False
        self.log('final tracks', [(t['label'], t['xy'].round(3).tolist(), t['miss'], t['reacq']) for t in self.containers()])
        try:
            json.dump([{'label': t['label'], 'reacq': t.get('reacq_log', []),
                        'hist': [(s, xy.tolist(), ok) for s, xy, ok in t['hist']]}
                       for t in self.tracker.tracks], open(os.path.join(OUT, 'tracks.json'), 'w'))
        except Exception as e:
            self.log('dump tracks failed', repr(e))

    def choose_grasp_yaw(self, center, face_ang, others):
        cur = finger_yaw(self.state)
        best = None
        for k in range(2):
            ax = face_ang + k * np.pi / 2
            u = np.array([np.cos(ax), np.sin(ax)]); w = np.array([-u[1], u[0]])
            pen = 0.0
            for o in others:
                d = o - center
                if abs(d @ w) < 0.05:
                    pen += max(0.0, 0.08 - abs(d @ u))
            # equivalent finger yaw closest to current (finger axis is symmetric under +pi)
            cands = [ax + j * np.pi for j in range(-3, 4)]
            yaw = min(cands, key=lambda a: abs(a - cur))
            score = pen * 10 + abs(yaw - cur) * 0.01
            if best is None or score < best[0]:
                best = (score, yaw, pen)
        return best[1], best[2]

    def pick(self, label, last):
        t = self.tracker.get(label, self.taken)
        if t is None:
            t = self.tracker.get('unk', self.taken)
            self.log(f'no track for {label}; fallback unk track: {t is not None}')
            if t is None:
                return False
        self.taken.add(id(t))
        xy = t['xy'].copy()
        others = [o['xy'] for o in self.containers() if o is not t]
        self.log(f'pick {label} track xy={xy.round(3)} miss={t["miss"]} reacq={t["reacq"]}')
        for attempt in range(2):
            if self.ended:
                return False
            self.move([xy[0], xy[1], HOVER_Z], -1, finger_yaw(self.state), tol=0.012, max_steps=50, vmax=1.0)
            self.hold(1, -1)
            tf = top_face(self.m, self.ims, xy)
            self.log('top face', None if tf is None else {k: (np.round(v, 3).tolist() if hasattr(v, 'tolist') else v) for k, v in tf.items()})
            if tf is not None and np.linalg.norm(tf['xy'] - xy) < 0.035 and tf['z'] > 0.055:
                c = tf['xy']; ztop = tf['z']; ang = tf['ang']
            else:
                c = xy; ztop = 0.073; ang = 0.0
            yaw, pen = self.choose_grasp_yaw(c, ang, others)
            self.log(f'grasp center={np.round(c,3)} ztop={ztop:.3f} yaw={np.degrees(yaw):.1f} pen={pen:.3f}')
            self.move([c[0], c[1], HOVER_Z], -1, yaw, tol=0.005, max_steps=25, vmax=0.8)
            gz = min(0.045, ztop - 0.028)
            self.move([c[0], c[1], gz], -1, yaw, tol=0.004, max_steps=30, vmax=0.8)
            self.hold(9, 1, yaw)
            self.move([c[0], c[1], LIFT_Z], 1, yaw, tol=0.012, max_steps=30, vmax=1.0)
            width = float(self.state[6] + self.state[7])
            self.log(f'lifted {label}: width={width:.4f} z={self.state[2]:.3f}')
            if width > 0.01:
                if last:
                    self.hold(10, 1, yaw)
                else:
                    self.hold(3, 1, yaw)
                    self.move([c[0], c[1], gz + 0.012], 1, yaw, tol=0.006, max_steps=30, vmax=1.0)
                    self.hold(5, -1, yaw)
                    self.move([c[0], c[1], HOVER_Z], -1, yaw, tol=0.02, max_steps=20, vmax=1.0)
                return True
            # failed grasp: open, retry once from a fresh view
            self.hold(4, -1, yaw)
            xy = c
        return False

    def run(self):
        self.taken = set()
        self.init_scene()
        for name, fn in (('press', self.press_buttons), ('wait', self.park_and_wait)):
            if self.ended:
                break
            try:
                fn()
            except Exception:
                self.log(f'phase {name} error', traceback.format_exc())
        for i, lab in enumerate(self.targets):
            if self.ended:
                break
            try:
                ok = self.pick(lab, last=(i == len(self.targets) - 1))
                self.log(f'pick {lab} ok={ok}')
            except Exception:
                self.log(f'pick {lab} error', traceback.format_exc())
        # hold until episode ends or a few seconds pass
        n = 0
        while not self.ended and n < 40:
            self.hold(1, 1)
            n += 1
        self.log('done; ended=', self.ended, 'penalties=', self.penalties)


def main(ctx):
    c = Ctl(ctx)
    try:
        c.run()
    except Exception:
        c.log('FATAL', traceback.format_exc())
