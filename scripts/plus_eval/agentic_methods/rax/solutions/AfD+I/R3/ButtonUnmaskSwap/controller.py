"""ButtonUnmaskSwap controller v2 (v1 + ep2 fixes: tilt/yaw control, refine-locked grasp XY,
stall abort, same-colour retries, never touch a later colour after a failed pick).

Pipeline: perceive cubes at step 0 -> track the containers that drop on them (fused front+wrist
RGB-D top-face cells, world XY) -> press +y button then -y button (hover, wait for clearance,
quick press) -> return to overview -> wait for the shuffle to settle -> pick the containers hiding
the instructed colours in order (lift, put back, release).
"""
import os, re, sys, time, json, math, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import obsio, percep, track2, geomx  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

VERSION = 'v2'
OUT_DIR = '/workspace/output'

# ---- tunables (evidence: exploration episode 1 unless noted) ----
CHUNK = 2                 # control steps per call while tracking (containers move ~3 mm/step)
HOVER_Z = 0.16            # TCP height above buttons/containers that clears containers (top 0.073)
PRESS_Z_STOP = 0.0275     # ep1: press registered between TCP z 0.032 and 0.026
PRESS_Z_CMD = 0.020
OVERVIEW = np.array([-0.06, 0.0, 0.52])   # start pose: wrist sees whole table
GRASP_Z = 0.038           # TCP z for grasping container body (body ~6 cm wide, top 0.073)
LIFT_Z = 0.16
PLACE_Z = 0.042
VMAX = 0.035              # m/step commanded (achieved ~70%)
HAND_ACROSS, HAND_ALONG = 0.035, 0.107   # half sizes of hand body (across / along finger axis)
CONT_R = 0.035            # container half-diagonal-ish radius at hand height
PRESS_DEADLINE = 179
SETTLE_MAX_STEP = 520     # stop waiting for the shuffle to settle after this step
APPROACH_Z = 0.20         # measurement hover above target (hand above top band)
MAX_ATTEMPTS = 4          # per colour; never move to the next colour after a failure
YAW_LIMIT = math.pi / 2   # keep gripper yaw within +-90 deg of home (ep2: yaw 152 deg + contact -> 27 deg tilt)


class Ended(Exception):
    pass


class Ctl:
    def __init__(self, ctx):
        self.ctx = ctx
        self.t0 = time.time()
        self.logf = None
        for d in (OUT_DIR, '/tmp/bus_out'):
            try:
                os.makedirs(d, exist_ok=True)
                self.logf = open(os.path.join(d, 'controller_log.txt'), 'a')
                self.out_dir = d
                break
            except Exception:
                continue
        self.o = None
        self.tr = None
        self.frozen = set()
        self.complete = None
        self.ended = False

    def log(self, *a):
        s = f'[{VERSION} t={time.time() - self.t0:6.1f} step={getattr(self.o, "step_id", None)}] ' + ' '.join(str(x) for x in a)
        print(s, flush=True)
        try:
            self.logf.write(s + '\n'); self.logf.flush()
        except Exception:
            pass

    # ---------- robot I/O ----------
    def observe(self):
        r = self.ctx.call('robomme_observe')
        self.o = obsio.decode_reply(r)
        self._check_end()
        return self.o

    def _check_end(self):
        if self.o.task_complete is not None:
            self.complete = self.o.task_complete
        if self.o.ended:
            self.ended = True
            self.log('EPISODE ENDED task_complete=', self.o.task_complete)
            raise Ended()

    def step(self, rows):
        rem = self.o.limits.get('remaining_episode_steps') if self.o is not None else None
        if rem is not None and rem <= 0:
            self.ended = True
            raise Ended()
        if rem is not None:
            rows = rows[:max(1, rem)]
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        r = self.ctx.call('robomme_step', actions=rows)
        self.o = obsio.decode_reply(r)
        self.track_update()
        self._check_end()
        return self.o

    def hold(self, n, grip):
        return self.step([[0, 0, 0, 0, 0, 0, grip]] * n)

    # ---------- perception ----------
    def track_update(self):
        if self.tr is None or not self.o.views:
            return
        try:
            P = track2.top_points(self.o.views, self.o.state)
            if self.o.step_id is not None and self.o.step_id >= 9:
                self.tr.update(P, self.o.step_id, frozen=self.frozen)
        except Exception as e:
            self.log('track_update error', repr(e))

    def preds(self, k):
        return np.array([t.p + t.v * k for t in self.tr.tracks])

    # ---------- motion ----------
    def yaw(self):
        return track2.ee_yaw(self.o.state)[0]

    def wrap_yaw(self, y):
        """Gripper is symmetric under 180 deg: map yaw into [home-90, home+90]."""
        d = (y - self.home_yaw + math.pi / 2) % math.pi - math.pi / 2
        return self.home_yaw + d

    def rot_err(self, yaw_des):
        """World-frame rotvec taking current orientation to gripper-down at yaw_des."""
        Rc = Rot.from_rotvec(self.o.state[3:6])
        Rd = Rot.from_euler('z', yaw_des - self.home_yaw) * self.R_home
        return (Rd * Rc.inv()).as_rotvec()

    def tilt(self):
        R = Rot.from_rotvec(self.o.state[3:6]).as_matrix()
        return math.degrees(math.acos(float(np.clip(-R[2, 2], -1, 1))))

    def move(self, target, grip, yaw=None, vmax=VMAX, tol=0.004, max_calls=40, chunk=CHUNK, retarget=None, yaw_tol=0.03):
        """Closed-loop straight-line move of the TCP; keeps gripper pointing down (tilt correction);
        optional world yaw target (else hold current yaw); retarget() -> new target."""
        target = np.asarray(target, float)
        yaw_hold = self.yaw() if yaw is None else None
        for _ in range(max_calls):
            if retarget is not None:
                nt = retarget()
                if nt is not None:
                    target = np.asarray(nt, float)
            d = target - self.o.state[:3]
            dist = float(np.linalg.norm(d))
            yd = yaw if yaw is not None else yaw_hold
            e = self.rot_err(yd)
            ang = float(np.linalg.norm(e))
            if dist < tol and (ang < yaw_tol or (yaw is None and ang < 0.06)):
                return True
            n = max(1, min(chunk, int(math.ceil(max(dist / vmax, ang / 0.35)))))
            per = d / n if dist / n <= vmax else d / dist * vmax
            r = e / n
            rn = float(np.linalg.norm(r))
            if rn > 0.35:
                r = r / rn * 0.35
            row = list(per / 0.05) + list(r / 0.5) + [grip]
            self.step([row] * n)
        return False

    # ---------- phases ----------
    def perceive_initial(self):
        o = self.observe()
        Rc = Rot.from_rotvec(o.state[3:6])
        if Rc.as_matrix()[2, 2] < -0.995:
            self.R_home = Rc
        else:
            self.R_home = Rot.from_rotvec([math.pi, 0, 0])
        self.home_yaw = float(np.arctan2(self.R_home.as_matrix()[1, 0], self.R_home.as_matrix()[0, 0]))
        self.log('home yaw', round(math.degrees(self.home_yaw), 1))
        self.instruction = o.meta.get('instruction', '')
        self.order = re.findall(r'hiding the (\w+) cube', self.instruction)
        self.log('instruction:', self.instruction, '-> order', self.order)
        self.buttons = percep.detect_buttons(o.views)
        self.buttons = sorted(self.buttons, key=lambda b: -b[1])  # +y first
        self.log('buttons', [b.round(3).tolist() for b in self.buttons])
        cubes = percep.detect_cubes(o.views, buttons=self.buttons)
        self.log('cubes', [(c['color'], c['c'].round(3).tolist()) for c in cubes])
        self.tr = track2.Tracker([c['c'] for c in cubes], [c['color'] for c in cubes], 0)
        self.cubes = cubes

    def wait_drop(self):
        while self.o.step_id < 12:
            self.hold(CHUNK, -1)
        self.log('tracks after drop', self.tracks_str())

    def tracks_str(self):
        return ' '.join(f'{t.label}:({t.p[0]:.3f},{t.p[1]:.3f})m{t.miss}n{t.n}' for t in self.tr.tracks)

    def press_clearance(self, bxy, yaw, horizon=12):
        """Min margin (m) of predicted containers outside the inflated hand footprint at bxy/yaw."""
        ex = np.array([math.cos(yaw), math.sin(yaw)]); ey = np.array([-math.sin(yaw), math.cos(yaw)])
        worst = 9.0
        for k in range(0, horizon + 1, 2):
            for p in self.preds(k):
                d = p - bxy
                a, b = abs(d @ ex), abs(d @ ey)
                ma = a - (HAND_ACROSS + CONT_R)
                mb = b - (HAND_ALONG + CONT_R)
                margin = max(ma, mb)
                worst = min(worst, margin)
        return worst

    def press(self, bxy, deadline, label):
        bxy = np.asarray(bxy, float)
        yaw0 = self.yaw()
        cands = [self.wrap_yaw(yaw0), self.wrap_yaw(yaw0 + math.pi / 2)]
        # approach hover
        best = max(cands, key=lambda y: self.press_clearance(bxy, y))
        self.move([bxy[0], bxy[1], HOVER_Z], grip=1, yaw=best, vmax=0.045)
        self.log(f'press {label}: hover reached', self.o.state[:3].round(3))
        # wait for clearance
        while True:
            cl = {y: self.press_clearance(bxy, y) for y in cands}
            best = max(cl, key=cl.get)
            if cl[best] > 0.01:
                break
            if self.o.step_id >= deadline - 22:
                self.log(f'press {label}: deadline forcing press; clearance {cl[best]:.3f}')
                break
            self.move([bxy[0], bxy[1], HOVER_Z], grip=1, yaw=best, max_calls=1)
            self.hold(CHUNK, 1)
        self.log(f'press {label}: clearance', {round(k, 2): round(v, 3) for k, v in cl.items()}, 'yaw', round(best, 2))
        self.move([bxy[0], bxy[1], HOVER_Z], grip=1, yaw=best, max_calls=6)
        # fast descent
        self.move([bxy[0], bxy[1], 0.055], grip=1, vmax=0.05, tol=0.006, max_calls=10)
        # press
        zs = []
        for i in range(12):
            z = self.o.state[2]
            zs.append(z)
            if z <= PRESS_Z_STOP:
                break
            if len(zs) >= 3 and z < 0.036 and zs[-3] - z < 0.0015:
                self.log('press stall at z', round(z, 4)); break
            dz = max(PRESS_Z_CMD - z, -0.012)
            exy = (bxy - self.o.state[:2])
            r = np.clip(self.rot_err(best), -0.2, 0.2) / 0.5
            self.step([[exy[0] / 0.05, exy[1] / 0.05, dz / 0.05, r[0], r[1], r[2], 1]])
        self.log(f'press {label}: bottom z', round(self.o.state[2], 4))
        self.move([bxy[0], bxy[1], HOVER_Z], grip=1, vmax=0.05, tol=0.01, max_calls=10)
        self.log(f'press {label}: done, up', self.tracks_str())

    def overview(self):
        self.move(OVERVIEW, grip=-1, vmax=0.05, tol=0.01, max_calls=20)

    def wait_settle(self):
        hist = []
        while self.o.step_id < SETTLE_MAX_STEP:
            self.hold(CHUNK, -1)
            sp = max(float(np.linalg.norm(t.v)) for t in self.tr.tracks)
            hist.append((self.o.step_id, sp))
            if self.o.step_id % 40 < 4:
                self.log('settle: max speed', round(sp, 5), self.tracks_str())
            recent = [s for st, s in hist if st > self.o.step_id - 24]
            if len(hist) > 8 and max(recent) < 0.0004:
                self.log('settled at step', self.o.step_id)
                return True
        self.log('settle timeout; picking while moving')
        return False

    def target_track(self, color):
        ts = [t for t in self.tr.tracks if t.label == color and t.id not in self.done_ids]
        if not ts:
            return None
        # prefer best-observed
        return min(ts, key=lambda t: t.miss)

    def measure(self, center, mask=False, r=0.035, min_n=25):
        """Container top cells near center -> (centroid xy, rect yaw, n) or None."""
        P = track2.top_points(self.o.views, self.o.state, mask_hand=mask)
        if len(P) == 0:
            return None
        sel = P[np.linalg.norm(P - np.asarray(center), axis=1) < r]
        if len(sel) < min_n:
            return None
        a, e1, e2, mid = geomx.rect_fit(sel)
        return sel.mean(0), a, len(sel)

    def choose_yaw(self, t, cyaw):
        cands = [self.wrap_yaw(cyaw + k * math.pi / 2) for k in range(4)]
        def nb_margin(y):
            ex = np.array([math.cos(y), math.sin(y)]); ey = np.array([-math.sin(y), math.cos(y)])
            m = 9
            for o_ in self.tr.tracks:
                if o_ is t:
                    continue
                d = o_.p - t.p
                m = min(m, max(abs(d @ ex) - (HAND_ACROSS + CONT_R), abs(d @ ey) - (HAND_ALONG + CONT_R)))
            return m
        y0 = self.yaw()
        # prefer neighbour margin; tie-break by smaller rotation
        gy = max(cands, key=lambda y: (round(nb_margin(y), 2), -abs((y - y0 + math.pi) % (2 * math.pi) - math.pi)))
        return gy, nb_margin(gy)

    def pick(self, color, attempt):
        """One attempt: returns 'held', 'miss' (nothing grasped) or 'notrack'."""
        t = self.target_track(color)
        if t is None:
            self.log('no track for', color); return 'notrack'
        self.cur_id = t.id
        moving = not self.settled
        tag = f'pick {color}#{attempt}'
        # 1) high, hand-free view
        if self.o.state[2] < 0.40:
            self.move(OVERVIEW, grip=-1, yaw=self.home_yaw, vmax=0.05, tol=0.015, max_calls=25)
        self.hold(2, -1)
        m0 = self.measure(t.p, r=0.04)
        self.log(f'{tag}: track id {t.id} p {t.p.round(3)} v {t.v.round(4)} overview meas',
                 None if m0 is None else (m0[0].round(3).tolist(), round(math.degrees(m0[1]), 1), m0[2]))
        if m0 is not None and not moving:
            t.p = m0[0].copy()
        cyaw = m0[1] if m0 is not None else 0.0
        gy, nbm = self.choose_yaw(t, cyaw)
        self.log(f'{tag}: gripper yaw {math.degrees(gy):.1f} nb_margin {nbm:.3f} tilt {self.tilt():.1f}')
        # 2) approach hover
        if moving:
            rt = lambda: [*(t.p + t.v * 4), APPROACH_Z]
        else:
            rt = None
        self.move([t.p[0], t.p[1], APPROACH_Z], grip=-1, yaw=gy, vmax=0.05, tol=0.006, max_calls=30, retarget=rt)
        m1 = self.measure(t.p, r=0.035)
        tgt = t.p.copy()
        if m1 is not None and np.linalg.norm(m1[0] - t.p) < 0.025:
            tgt = m1[0].copy()
            if abs(((m1[1] - cyaw) + math.pi / 4) % (math.pi / 2) - math.pi / 4) > math.radians(12) and m1[2] > 60:
                gy, nbm = self.choose_yaw(t, m1[1])
        self.log(f'{tag}: hover meas', None if m1 is None else (m1[0].round(3).tolist(), round(math.degrees(m1[1]), 1), m1[2]),
                 'tgt', tgt.round(3).tolist(), 'yaw', round(math.degrees(gy), 1))
        if not moving:
            t.p = tgt.copy()
        # 3) align above at 0.12 (still above the container top 0.073)
        if moving:
            rt2 = lambda: [*(t.p + t.v * 3), 0.12]
        else:
            rt2 = None
        self.move([tgt[0], tgt[1], 0.12], grip=-1, yaw=gy, vmax=0.04, tol=0.003, max_calls=25, retarget=rt2)
        if moving:
            tgt = t.p + t.v * 4
        err = float(np.linalg.norm(self.o.state[:2] - tgt))
        self.log(f'{tag}: at 0.12', self.o.state[:3].round(4), 'xy err', round(err, 4), 'tilt', round(self.tilt(), 1))
        # 4) straight descent with stall detection (fixed XY)
        self.frozen.add(t.id)
        zs = []
        stalled = False
        for _ in range(16):
            z = float(self.o.state[2])
            zs.append(z)
            if z <= GRASP_Z + 0.003:
                break
            if len(zs) >= 3 and zs[-3] - z < 0.003 and z > GRASP_Z + 0.01:
                stalled = True; break
            exy = tgt - self.o.state[:2]
            dz = max(GRASP_Z - z, -0.03)
            r = np.clip(self.rot_err(gy), -0.2, 0.2) / 0.5
            self.step([[exy[0] / 0.05, exy[1] / 0.05, dz / 0.05, r[0], r[1], r[2], -1]] * 2)
        self.log(f'{tag}: descent end', self.o.state[:3].round(4), 'stalled', stalled, 'tilt', round(self.tilt(), 1))
        if stalled or self.o.state[2] > 0.055:
            # blocked on top of something: back off without closing
            self.move([self.o.state[0], self.o.state[1], APPROACH_Z], grip=-1, yaw=gy, vmax=0.04, tol=0.01, max_calls=12)
            self.frozen.discard(t.id)
            return 'miss'
        # 5) grasp, lift, check
        for _ in range(4):
            self.hold(2, 1)
        fing = float(self.o.state[6] + self.o.state[7])
        self.log(f'{tag}: closed fingers sum {fing:.4f}')
        self.move([self.o.state[0], self.o.state[1], LIFT_Z], grip=1, yaw=gy, vmax=0.03, tol=0.008, max_calls=15)
        self.hold(4, 1)
        fing2 = float(self.o.state[6] + self.o.state[7])
        held = fing2 > 0.02
        self.log(f'{tag}: lifted z {self.o.state[2]:.3f} fingers {fing2:.4f} held={held}')
        # 6) put back down and release
        self.move([self.o.state[0], self.o.state[1], PLACE_Z + 0.01], grip=1, yaw=gy, vmax=0.03, tol=0.006, max_calls=15)
        for _ in range(3):
            self.hold(2, -1)
        t.p = self.o.state[:2].copy(); t.v[:] = 0
        self.move([self.o.state[0], self.o.state[1], APPROACH_Z], grip=-1, yaw=gy, vmax=0.04, tol=0.01, max_calls=12)
        self.frozen.discard(t.id)
        self.log(f'{tag}: done held={held}', self.tracks_str())
        return 'held' if held else 'miss'

    def run(self):
        self.done_ids = set()
        self.settled = False
        self.perceive_initial()
        self.wait_drop()
        if len(self.buttons) >= 2:
            self.press(self.buttons[0], deadline=150, label='B1(+y)')
            self.press(self.buttons[1], deadline=PRESS_DEADLINE, label='B2(-y)')
        else:
            self.log('WARNING: buttons not found', len(self.buttons))
        self.overview()
        self.settled = self.wait_settle()
        for ci, color in enumerate(self.order):
            ok = False
            for att in range(1, MAX_ATTEMPTS + 1):
                rem = self.o.limits.get('remaining_episode_steps')
                if rem is not None and rem < 90:
                    self.log('too few steps left for another attempt', rem); break
                res = self.pick(color, att)
                if res == 'held':
                    ok = True; self.done_ids.add(self.cur_id); break
                if res == 'notrack':
                    break
            if not ok:
                self.log(f'pick {color}: FAILED after retries; not touching later colours')
                break
        self.log('all picks attempted; holding')
        for _ in range(10):
            self.hold(5, -1)


def main(ctx):
    try:
        c = Ctl(ctx)
    except Exception:
        print('FATAL init', traceback.format_exc(), flush=True)
        return
    try:
        c.run()
    except Ended:
        pass
    except Exception:
        c.log('FATAL', traceback.format_exc())
        # keep the episode safe: do nothing further
    try:
        if c.tr is not None:
            with open(os.path.join(getattr(c, 'out_dir', '/tmp'), 'tracks.json'), 'w') as f:
                json.dump({t.label + '_' + str(t.id): [list(map(float, h)) for h in t.hist] for t in c.tr.tracks}, f)
    except Exception as e:
        c.log('track dump failed', repr(e))
    c.log('exit; complete=', c.complete, 'ended=', c.ended)
