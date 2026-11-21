"""ButtonUnmaskSwap controller (bundle 'bums').

Pipeline / transition table
  INIT      observe step 0, parse instruction, calibrate back-projection, scene (table, buttons, cubes)
            -> WAIT_LAND (cubes visible)   | VIDEO (cubes only in conditioning video) | FALLBACK
  WAIT_LAND hold until bins appear over cubes (<=40 steps) -> TRACK_INIT
  PRESS     press both buttons (nearest first), closed gripper, vertical -> HOME  (each press: stall/timeout bounded)
  HOME      return to start pose (wrist camera overview) -> WAIT_STATIC
  WAIT      hold & track until bins static 15 steps, or step/time/step-budget reserve -> PICK
  PICK(c)   for each colour: hover, refine with wrist top face (centre+yaw), descend, close, verify
            opening, lift, (episode may end = success), reveal check, put back -> next colour
            wrong colour -> relabel track, try swap partner (bounded)
Tracking runs on every reply (front+wrist depth -> world XY, k-means association).
"""
import json
import math
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '1')  # BLAS spin threads throttled the 2-CPU dev container badly

import numpy as np  # noqa: E402

from geom import Rz, rotvec_to_R, R_to_rotvec, wrap_quarter, Camera  # noqa: E402
from obsio import (parse, log, strip_big, instruction_of, target_colors, manifest_frames,  # noqa: E402
                   load_frame_images, Reply)
from percept import Calib  # noqa: E402
from world import World  # noqa: E402
from robot import Robot, EpisodeOver  # noqa: E402

VERSION = 'bums-v5'
OUT = '/workspace/output'

# tunables (untested live; derived from demo images and mock tests)
HOVER_DZ = 0.10          # hover height above bin top
SAFE_DZ = 0.12           # travel height above bin top
PRESS_DEPTH = 0.015      # push below detected cap top
PRESS_HOVER = 0.06       # above cap top before pressing
STATIC_STEPS = 20        # all tracks still for this many steps => start picking (demo motion stops abruptly)
WAIT_MAX_STEP = 300      # start picking at the latest at this step
PICK_RESERVE = 70        # steps reserved per remaining pick (mock: ~72 with put-back, ~50 for the last)
CTRL_WALL = 1100.0       # controller wall limit (s) is 1200; hard guard in Robot.send


def save_json(name, obj):
    try:
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, name), 'w') as f:
            json.dump(obj, f, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o), indent=1)
    except Exception as e:
        log('save_json failed', name, e)


def save_img(name, arr):
    try:
        from PIL import Image
        os.makedirs(OUT, exist_ok=True)
        Image.fromarray(arr).save(os.path.join(OUT, name))
    except Exception as e:
        log('save_img failed', name, e)


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.t0 = time.time()
        self.calib = Calib()
        self.world = World(self.calib)
        self.robot = Robot(ctx, self.world, self.t0)
        self.robot.wall_limit = CTRL_WALL
        self.robot.on_reply = self.on_reply
        self.tracking = False
        self.targets = []
        self.done = []
        self.home_p = None
        self.home_R = None
        self.R_down = None
        self.approach_axis = 2
        self.events = []
        self.track_log = []
        self.grip_choice = {}

    # ---------------------------------------------------------------- hooks
    def on_reply(self, r):
        if self.tracking and self.world.tracker is not None and r.step is not None:
            counts = self.world.track(r, r.step)
            tr = self.world.tracker
            if r.step % 5 == 0:
                rec = dict(step=r.step, p=np.round(tr.p, 4).tolist(), counts=None if counts is None else counts.tolist(),
                           per=getattr(self.world, 'last_per', None))
                self.track_log.append(rec)
                if r.step % 25 == 0:
                    tg = {c: np.round(tr.p[j], 3).tolist() for j, c in self.world.track_color.items() if c in self.targets}
                    log('[%s] step %d t=%.0fs targets %s seen=%s per=%s' % (VERSION, r.step, self.robot.elapsed(), tg,
                        None if counts is None else int((counts >= tr.min_pts).sum()), getattr(self.world, 'last_per', None)))

    # ---------------------------------------------------------------- orientation
    def setup_orientation(self, r):
        R0 = rotvec_to_R(r.state[3:6])
        self.home_R = R0
        self.home_p = np.array(r.state[:3], float)
        axis_i, sign = 2, 1.0
        cam = self.calib.camera(r, 'wrist')
        if cam is not None:
            ca = cam.axis()
            dots = [float(R0[:, i] @ ca) for i in range(3)]
            i = int(np.argmax(np.abs(dots)))
            if abs(dots[i]) > 0.6:
                axis_i, sign = i, (1.0 if dots[i] > 0 else -1.0)
            log('wrist cam axis', np.round(ca, 3).tolist(), 'EE axes dots', np.round(dots, 3).tolist(), '-> approach axis', axis_i, sign)
        self.approach_axis = (axis_i, sign)
        a = R0[:, axis_i] * sign
        target = np.array([0, 0, -1.0])
        v = np.cross(a, target)
        s = np.linalg.norm(v)
        c = float(a @ target)
        if s < 1e-8:
            Rot = np.eye(3) if c > 0 else rotvec_to_R(np.array([1.0, 0, 0]) * math.pi)
        else:
            Rot = rotvec_to_R(v / s * math.atan2(s, c))
        self.R_down = Rot @ R0
        log('home p', np.round(self.home_p, 4).tolist(), 'tilt from vertical (deg) %.1f' % math.degrees(math.atan2(s, c)))

    def finger_yaw(self, R):
        """angle in XY of the EE axes perpendicular to approach (mod 90 deg)"""
        i = (self.approach_axis[0] + 1) % 3
        ax = R[:, i]
        return math.atan2(ax[1], ax[0])

    def R_for_yaw(self, bin_yaw, j=None):
        """vertical gripper rotated so that fingers align with bin faces (mod 90).  If track j is given,
        choose among the two face-aligned options the one whose finger-closing axis has the most clearance
        from neighbouring bins (fingers must not land on a touching neighbour)."""
        cur = self.finger_yaw(self.R_down)
        d = wrap_quarter(bin_yaw - cur)
        if j is None or self.world.tracker is None:
            return Rz(d) @ self.R_down
        best = None
        locked = self.grip_choice.get(j)
        for extra in ((locked,) if locked is not None else (0.0, math.pi / 2)):
            dd = d + extra
            dd = (dd + math.pi / 2) % math.pi - math.pi / 2  # gripper symmetric under 180 deg
            th = cur + dd  # angle of the finger_yaw axis after rotation
            cd = th + math.pi / 2  # closing direction (perpendicular in the plane)
            # closing-axis index differs from finger_yaw axis; check both perpendicular options by clearance
            clr = self._clearance(j, cd)
            key = (round(clr, 3), -abs(dd))
            if best is None or key > best[0]:
                best = (key, dd, clr, extra)
        self.last_clearance = best[2]
        if locked is None:
            self.grip_choice[j] = best[3]
            log('grip choice track %d: extra %.0f deg clearance %.3f' % (j, math.degrees(best[3]), best[2]))
        return Rz(best[1]) @ self.R_down

    def _clearance(self, j, ang):
        tr = self.world.tracker
        w = self.world.w_bin or 0.045
        c = tr.p[j]
        dv = np.array([math.cos(ang), math.sin(ang)])
        pv = np.array([-dv[1], dv[0]])
        clr = 1.0
        for k in range(len(tr.p)):
            if k == j:
                continue
            rel = tr.p[k] - c
            if abs(rel @ pv) < w:
                clr = min(clr, abs(rel @ dv))
        return clr

    # ---------------------------------------------------------------- phases
    def init(self):
        r = self.robot.observe()
        meta = strip_big(r.meta)
        save_json('first_meta.json', meta)
        log('[%s] first reply summary' % VERSION, json.dumps(r.summary(), default=str)[:1500])
        log('meta keys', list(r.meta.keys())[:40])
        for (cam, kind), arr in r.images.items():
            if kind == 'rgb':
                save_img('step0_%s.png' % cam, arr)
        ins = instruction_of(r.meta)
        self.targets = target_colors(ins)
        log('instruction:', ins, '-> targets', self.targets)
        if len(self.targets) < 1:
            self.targets = ['blue', 'green', 'red']
            log('WARNING: no colours parsed; default', self.targets)
        ok = self.calib.calibrate(self.ctx, r)
        log('calibration ok=%s conv=%s err=%s swap=%s' % (ok, self.calib.conv, self.calib.err, self.calib.depth_swap))
        if r.state is None:
            log('WARNING: no state in reply')
        self.setup_orientation(r)
        return r

    def init_scene_and_landing(self, r):
        w = self.world
        ok = w.init_scene(r)
        landed, cnt = (False, 0)
        if ok:
            landed, cnt = w.landing_check(r)
        log('scene ok=%s landed=%s (%d)' % (ok, landed, cnt))
        cols = [c['color'] for c in w.cubes]
        has_targets = any(t in cols for t in self.targets)
        has_video = bool(r.manifest) and any(f.get('phase') == 'video_demo' for f in (r.manifest.get('frames') or []))
        if has_video and (not ok or not has_targets):
            # the colours were only shown in the conditioning video: track through it
            log('no target colours visible now (%s); using conditioning video' % cols)
            if self.video_init(r):
                self.tracking = True
                return True
            log('video init failed; continuing with live scene')
            ok = w.init_scene(r)
            landed, cnt = w.landing_check(r) if ok else (False, 0)
        if ok and not landed:
            # wait for bins to appear
            for k in range(45):
                r = self.robot.hold(1, grip=-1.0)
                landed, cnt = w.landing_check(r)
                if landed:
                    log('bins landed at step', r.step, cnt)
                    break
            w.init_tracker(r, r.step)
            self.tracking = True
            return True
        # cubes not visible now: try conditioning video
        if r.manifest:
            if self.video_init(r):
                self.tracking = True
                return True
        if ok:
            # cubes found but already covered?  initialise anyway
            w.init_tracker(r, r.step)
            self.tracking = True
            return True
        return False

    def video_init(self, r0):
        def _ts(fr, i):
            for k in ('timestamp_s', 'timestamp'):
                v = fr.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
            sfi = fr.get('source_frame_index')
            if isinstance(sfi, (int, float)):
                return float(sfi) / 20.0
            return i * 0.25  # unknown clock: assume 5-step sampling

        """run scene init + tracking over the conditioning video frames (robot static)."""
        frames = manifest_frames(r0.manifest, load=False)
        vid = [f for f in frames if f.get('phase') != 'execution']
        log('video frames', len(vid))
        if not vid:
            return False
        w = self.world

        st0 = None if r0.state is None else np.asarray(r0.state, float)

        def mk(rec):
            load_frame_images(rec)
            rr = Reply()
            rr.images = dict(rec['images'])
            rr.Ks, rr.poses = r0.Ks, r0.poses
            rr.step = 0
            fs = rec.get('state')
            if fs is not None and st0 is not None and np.linalg.norm(fs[:3]) > 1e-6:
                if np.linalg.norm(fs[:3] - st0[:3]) > 0.01 or np.linalg.norm(fs[3:6] - st0[3:6]) > 0.05:
                    # robot was elsewhere in this frame: the wrist pose of r0 does not apply
                    rr.images.pop(('wrist', 'rgb'), None)
                    rr.images.pop(('wrist', 'depth'), None)
            return rr

        def n_targets(rr):
            w2 = World(self.calib)
            try:
                if not w2.init_scene(rr):
                    return 0
            except Exception as e:
                log('video scan init_scene error', e)
                return 0
            cols = [c['color'] for c in w2.cubes]
            return sum(1 for t in self.targets if t in cols)
        # latest frame showing the target cubes: forward scan until the colours disappear (bins land)
        idx_ok = None
        best = 0
        stride = max(1, len(vid) // 60)
        i = 0
        while i < len(vid):
            k = n_targets(mk(vid[i]))
            vid[i]['images'] = {}
            if k >= max(1, best) and k >= min(2, len(self.targets)):
                idx_ok, best = i, k
            elif idx_ok is not None and k < best:
                break
            i += stride
        if idx_ok is not None and stride > 1:
            # refine inside the last stride
            for j in range(idx_ok + 1, min(len(vid), idx_ok + stride)):
                k = n_targets(mk(vid[j]))
                vid[j]['images'] = {}
                if k >= best:
                    idx_ok = j
                else:
                    break
        if idx_ok is None:
            log('video: no cube-visible frame')
            return False
        log('video: cubes visible up to frame', idx_ok, 'of', len(vid), 'targets seen', best)
        rr = mk(vid[idx_ok])
        w.init_scene(rr)
        t0 = _ts(vid[idx_ok], idx_ok)
        landed = False
        clock0 = None
        for i in range(idx_ok + 1, len(vid)):
            rr = mk(vid[i])
            ts = _ts(vid[i], i)
            clock = int(round((ts - t0) * 20)) - 100000
            if not landed:
                if w.landing_check(rr)[0]:
                    w.init_tracker(rr, clock)
                    landed = True
            else:
                w.track(rr, clock)
            vid[i]['images'] = {}
        if not landed:
            log('video: landing not seen')
            return False
        # shift tracker clock to execution steps
        w.tracker.step = r0.step or 0
        w.track(r0, r0.step or 0)
        log('video init done; tracks', np.round(w.tracker.p, 3).tolist())
        return True

    # ---------------------------------------------------------------- buttons
    def press_buttons(self):
        w = self.world
        btns = list(w.buttons)
        if len(btns) < 2:
            log('WARNING: buttons found', len(btns))
        p = self.robot.pos()
        btns.sort(key=lambda b: float(np.linalg.norm(b['xy'] - p[:2])))
        for bi, b in enumerate(btns[:2]):
            if self.robot.step_id > 170:
                log('WARNING: past press deadline window; pressing anyway')
            self.press(b, bi)

    def press(self, b, bi):
        rb = self.robot
        R = self.R_down
        top = b['cap_top']
        x, y = float(b['xy'][0]), float(b['xy'][1])
        safe_z = max(top + PRESS_HOVER, rb.pos()[2] if rb.pos()[2] < top + 0.2 else top + PRESS_HOVER)
        # go above the button (closing the gripper on the way)
        rb.move([x, y, top + PRESS_HOVER], R, grip=1.0, tol=0.008, rtol=0.08, max_steps=45, vmax=0.45, label='to_button%d' % bi)
        rb.move([x, y, top + PRESS_HOVER * 0.5], R, grip=1.0, tol=0.005, rtol=0.06, max_steps=20, vmax=0.3, label='button_hover%d' % bi)
        # push down slowly until depth reached or stalled
        z_goal = top - PRESS_DEPTH
        zs = []
        for k in range(30):
            zc = rb.pos()[2]
            zs.append(zc)
            if zc <= z_goal + 0.002:
                break
            if len(zs) > 4 and zs[-4] - zc < 0.0015 and zc < top + 0.01:
                log('press %d: stalled at z=%.4f (cap_top %.4f)' % (bi, zc, top))
                break
            tgt = [x, y, max(z_goal, zc - 0.006)]
            rb.send([rb.row_towards(tgt, R, 1.0, vmax=0.15)])
        log('press %d: lowest z=%.4f cap_top=%.4f step=%d' % (bi, min(zs), top, rb.step_id))
        rb.hold(2, grip=1.0)
        rb.move([x, y, top + PRESS_HOVER], R, grip=1.0, tol=0.01, rtol=0.1, max_steps=20, vmax=0.4, label='button_up%d' % bi)
        self.events.append(dict(ev='press', button=bi, step=rb.step_id, lowest=min(zs), cap_top=top))

    def go_home(self):
        rb = self.robot
        p = rb.pos()
        # rise first if low
        if p[2] < self.home_p[2] - 0.05:
            rb.move([p[0], p[1], self.home_p[2]], None, grip=-1.0, tol=0.015, max_steps=15, vmax=0.4, label='rise')
        rb.move(self.home_p, self.home_R, grip=-1.0, tol=0.01, rtol=0.08, max_steps=50, vmax=0.4, label='home')

    # ---------------------------------------------------------------- waiting
    def wait_static(self):
        rb = self.robot
        tr = self.world.tracker
        n_left = len(self.targets) - len(self.done)
        while True:
            st = tr.static_for()
            left = rb.steps_left()
            lat = float(np.median(rb.lat[-20:])) if rb.lat else 0.5
            if st >= STATIC_STEPS:
                log('bins static for %d steps at step %d' % (st, rb.step_id))
                return
            if rb.step_id >= WAIT_MAX_STEP:
                log('wait: step cap reached', rb.step_id)
                return
            if left is not None and left < PICK_RESERVE * n_left + 20:
                log('wait: step reserve reached; left', left)
                return
            if rb.elapsed() + lat * PICK_RESERVE * n_left * 1.3 > CTRL_WALL - 60:
                log('wait: wall time reserve reached', rb.elapsed())
                return
            n = 1 if lat < 0.8 else 2
            rb.hold(n, grip=-1.0)

    # ---------------------------------------------------------------- picking
    def pick_color(self, color, attempt_budget=3):
        w = self.world
        cands = [j for j, c in w.track_color.items() if c == color and j not in self.done]
        if not cands:
            log('no track labelled', color, 'labels', w.track_color)
            return False
        j = cands[0]
        tried = set()
        for att in range(attempt_budget):
            res = self.pick_track(j, color)
            tried.add(j)
            if res == 'ok':
                self.done.append(j)
                return True
            if isinstance(res, tuple) and res[0] == 'wrong':
                seen = res[1]
                w.track_color[j] = seen
                self.done.append(j)
                # whichever track was labelled `seen` is likely the swapped one
                k = [i for i, c in w.track_color.items() if c == seen and i != j and i not in self.done]
                if k:
                    w.track_color[k[0]] = color
                    j = k[0]
                else:
                    pj = w.tracker.partner(j, exclude=set(self.done) | tried)
                    if pj is None:
                        return False
                    j = pj
                log('retry %s with track %d' % (color, j))
                continue
            if res == 'miss':
                continue
            return False
        return False

    def pick_track(self, j, color):
        rb, w = self.robot, self.world
        tr = w.tracker
        zt = w.z_table
        ztop = zt + w.h_bin
        z_hover = ztop + HOVER_DZ
        grasp_depth = min(0.025, 0.5 * w.h_bin)
        z_grasp = max(zt + 0.012, ztop - grasp_depth)
        yaw = w.yaw.get(j, (None,))[0]
        R = self.R_for_yaw(yaw, j) if yaw is not None else self.R_down
        self.grip_choice.pop(j, None)
        log('pick %s track %d at %s step %d' % (color, j, np.round(tr.p[j], 3).tolist(), rb.step_id))
        # 1. travel at safe height
        p = rb.pos()
        if p[2] < ztop + 0.05:
            rb.move([p[0], p[1], z_hover], R, grip=-1.0, tol=0.01, rtol=0.2, max_steps=15, vmax=0.4, label='lift_first')

        def tgt_hover():
            q = tr.predict(rb.step_id)[j]
            return [q[0], q[1], z_hover], None
        rb.move([tr.p[j][0], tr.p[j][1], z_hover], R, grip=-1.0, tol=0.008, rtol=0.06, max_steps=50, vmax=0.45,
                target_fn=tgt_hover, label='hover')
        # 2. from here on the target track is owned by the pick (wrist refinement only):
        #    the arm occludes it in the front camera and would corrupt the fused estimate.
        tr.frozen.add(j)
        est = dict(p=np.array(tr.p[j], float), v=np.array(tr.v[j], float), step=rb.step_id)

        def est_update(c):
            dt = max(1, rb.step_id - est['step'])
            vel = (np.asarray(c) - est['p']) / dt
            if np.linalg.norm(vel) > 0.012:
                vel = vel / np.linalg.norm(vel) * 0.012
            est['v'] = 0.5 * est['v'] + 0.5 * vel
            est['p'] = np.asarray(c, float)
            est['step'] = rb.step_id
            tr.p[j] = est['p']

        def est_pred():
            dt = max(0, rb.step_id - est['step'])
            d = est['v'] * min(dt, 15)
            return est['p'] + d

        centre, n_ok = None, 0
        for it in range(6):
            ref = w.refine_target(rb.r, j, centre=est_pred())
            if ref is None:
                log('refine: no top face (it %d)' % it)
                rb.hold(1, grip=-1.0)
                continue
            centre, yw, ntop = ref
            if yw is not None:
                yaw = yw
                w.yaw[j] = (yw, rb.step_id)
                R = self.R_for_yaw(yaw, j)
            d = float(np.linalg.norm(centre - est_pred()))
            est_update(centre)

            def tgt_al():
                q = est_pred()
                return [q[0], q[1], z_hover], None
            ok = rb.move([centre[0], centre[1], z_hover], R, grip=-1.0, tol=0.005, rtol=0.04, max_steps=12, vmax=0.3,
                         target_fn=tgt_al, label='align')
            log('refine it %d: centre %s yaw %s ntop %d d=%.4f v=%s' % (it, np.round(centre, 4).tolist(),
                None if yaw is None else round(math.degrees(yaw), 1), ntop, d, np.round(est['v'], 4).tolist()))
            if d < 0.006 and ok:
                n_ok += 1
                if n_ok >= 1 and yaw is not None:
                    break
        # 3. descend with wrist servo until close, then extrapolate
        def tgt_desc():
            if rb.pos()[2] > ztop + 0.045:
                ref = w.refine_target(rb.r, j, centre=est_pred())
                if ref is not None and np.linalg.norm(ref[0] - est_pred()) < 0.02:
                    est_update(ref[0])
            q = est_pred()
            return [q[0], q[1], ztop + 0.03], None
        q0 = est_pred()
        rb.move([q0[0], q0[1], ztop + 0.03], R, grip=-1.0, tol=0.006, rtol=0.05, max_steps=30, vmax=0.3,
                target_fn=tgt_desc, label='descend1')
        q0 = est_pred()
        gx, gy = float(q0[0]), float(q0[1])
        v_ff = np.array(est['v'], float)
        if np.linalg.norm(v_ff) < 0.0005:
            v_ff = np.zeros(2)
        s_freeze = rb.step_id

        def ff_xy():
            # feed-forward of a still-moving bin (linear extrapolation, capped at 3 cm)
            dxy = v_ff * max(0, rb.step_id - s_freeze)
            n = np.linalg.norm(dxy)
            if n > 0.03:
                dxy = dxy / n * 0.03
            return gx + dxy[0], gy + dxy[1]

        def tgt_d2():
            x, y = ff_xy()
            return [x, y, z_grasp], None
        rb.move([gx, gy, z_grasp], R, grip=-1.0, tol=0.004, rtol=0.05, max_steps=25, vmax=0.2, target_fn=tgt_d2,
                label='descend2')
        log('at grasp: pos %s target z %.4f v_ff %s' % (np.round(rb.pos(), 4).tolist(), z_grasp, np.round(v_ff, 4).tolist()))
        # 4. close
        gx, gy = ff_xy()
        op = rb.gripper(1.0, n=12, R_t=R, p_t=[gx, gy, z_grasp])
        log('closed: opening %.4f (w_bin %.4f)' % (op, w.w_bin))
        if op < 0.25 * w.w_bin or op < 0.008:
            log('grasp missed')
            rb.gripper(-1.0, n=6, R_t=R)
            rb.move([gx, gy, z_hover], R, grip=-1.0, tol=0.01, max_steps=20, vmax=0.3, label='retreat')
            tr.frozen.discard(j)
            return 'miss'
        # 5. lift
        self.events.append(dict(ev='grasp', color=color, track=j, step=rb.step_id, opening=op))
        rb.move([gx, gy, z_grasp + 0.12], R, grip=1.0, tol=0.01, rtol=0.1, max_steps=30, vmax=0.3, label='lift')
        rb.hold(2, grip=1.0)
        op2 = rb.opening()
        seen = w.visible_colors_near(rb.r, np.array([gx, gy]), radius=0.045)
        log('lifted: opening %.4f colours under %s step %d' % (op2, seen, rb.step_id))
        self.events.append(dict(ev='lift', color=color, track=j, step=rb.step_id, seen=seen, opening=op2))
        for (cam, kind), arr in rb.r.images.items():
            if kind == 'rgb':
                save_img('lift_%s_%s.png' % (color, cam), arr)
        if op2 < 0.25 * w.w_bin:
            log('bin dropped during lift?')
        verdict = 'ok'
        if seen:
            best = max(seen.items(), key=lambda kv: kv[1])
            if best[0] != color and best[1] >= 12 and seen.get(color, 0) < 4:
                verdict = ('wrong', best[0])
                log('WRONG colour revealed', best)
        # 6. put back (unless it was the last one: episode should have ended)
        last = (len(self.done) + 1 >= len(self.targets)) and verdict == 'ok'
        if last:
            rb.hold(3, grip=1.0)
        rb.move([gx, gy, z_grasp + 0.004], R, grip=1.0, tol=0.005, rtol=0.1, max_steps=30, vmax=0.25, label='put_back')
        rb.gripper(-1.0, n=8, R_t=R)
        rb.move([gx, gy, z_hover], R, grip=-1.0, tol=0.01, rtol=0.1, max_steps=25, vmax=0.35, label='retreat')
        tr.frozen.discard(j)
        tr.p[j] = np.array([gx, gy])
        tr.v[j] = 0
        return verdict

    # ---------------------------------------------------------------- main
    def run(self):
        r = self.init()
        if not self.init_scene_and_landing(r):
            log('FATAL: scene init failed; pressing nothing meaningful')
            return
        save_json('scene.json', dict(z_table=self.world.z_table, buttons=self.world.buttons, cubes=self.world.cubes,
                                     h_bin=self.world.h_bin, w_bin=self.world.w_bin, targets=self.targets,
                                     track_color=self.world.track_color))
        self.press_buttons()
        self.go_home()
        self.wait_static()
        for c in self.targets:
            ok = self.pick_color(c)
            log('pick %s -> %s (step %d, t=%.0fs)' % (c, ok, self.robot.step_id, self.robot.elapsed()))
        # hold a little so the final state registers
        for k in range(5):
            self.robot.hold(1)


def main(ctx):
    c = None
    try:
        c = Controller(ctx)
        c.run()
    except EpisodeOver:
        log('episode over (complete=%s)' % (c.robot.complete if c else None))
    except Exception as e:
        log('UNCAUGHT', e, traceback.format_exc())
        # keep the robot alive: try to continue with picks if scene exists
    finally:
        if c is not None:
            save_json('events.json', dict(version=VERSION, events=c.events, step=c.robot.step_id,
                                          complete=c.robot.complete, elapsed=c.robot.elapsed(),
                                          latency=c.robot.lat[-50:], tracks=c.track_log[-200:]))
            log('done: step %d complete=%s elapsed %.0fs calls %d' % (c.robot.step_id, c.robot.complete, c.robot.elapsed(), c.robot.ncalls))
