"""SwingXtimes controller (RAX / RoboMME-Plus).  main(ctx) is the entry point.

Pipeline (state machine, see TRANSITIONS):
  perceive -> grasp -> swing (2N contact visits, alternating, first side from instruction)
           -> place (put cube back where it was picked) -> press (button, gripper closed) -> done
Exits: every phase returns ok / fail.  fail -> retry the phase (bounded), grasp-loss during
swing -> grasp (keeps the visit counter), unrecoverable -> skip forward / stop.

Conventions learned offline from the 3 released demos (no simulator access, demo-only):
  * front camera: robot base at the top of the image; robot-RIGHT target = IMAGE-LEFT target.
  * a visited target turns red and stays red until the other one is visited.
  * N swings = 2N target visits (ep0: 6 swings -> 12 visits).
  * episode ends when the button is pressed.
"""
import json
import math
import os
import re
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perception as P  # noqa: E402
import io_utils as U  # noqa: E402

VERSION = 'swing-v1'

# ------------------------------------------------------------------ tunables (untested live: demo-only)
POS_SCALE = 0.05          # m per unit action (documented)
ROT_SCALE = 0.5           # rad per unit action (documented)
SPEED_FREE = 0.025        # m/step empty-hand travel
SPEED_CARRY = 0.020       # m/step while holding the cube
SPEED_SLOW = 0.006        # m/step near contact
APPROACH_ABOVE = 0.10     # TCP above cube top for the first wrist look
REFINE_ABOVE = 0.065      # TCP above cube top for the second wrist look
TRAVEL_CLEAR = 0.065      # cube bottom above table while travelling
HOVER_GAP = 0.008         # cube bottom above target top on a normal visit (demo ep0 hovers higher)
TOUCH_PRESS = 0.004       # retry: command this far below estimated contact on a target
TOUCH_EXTRA = 0.008       # max extra descent searching for contact
DEFAULT_SIDE = 0.04       # cube side if it cannot be measured
GRIP_MIN_OPEN = 0.004     # finger sum below this -> nothing in the hand
GRIP_MAX_HELD = 0.075     # finger sum above this -> gripper not closed on cube
BUTTON_PRESS_DEPTH = (0.012, 0.022, 0.032)  # successive press depths (TCP) below detected cap top
MAX_GRASP_ATTEMPTS = 4

OUT_DIR = '/workspace/output'


class EpisodeEnded(Exception):
    pass


# ------------------------------------------------------------------ logging
class Log:
    def __init__(self):
        self.t0 = time.time()
        self.steps_fn = None
        self.f = None
        try:
            os.makedirs(OUT_DIR, exist_ok=True)
            self.f = open(os.path.join(OUT_DIR, 'swing_log.txt'), 'a')
        except Exception:
            self.f = None

    def __call__(self, *a):
        st = self.steps_fn() if self.steps_fn else -1
        s = '[%s %6.1fs st%4d] ' % (VERSION, time.time() - self.t0, st) + ' '.join(str(x) for x in a)
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


def fmt(v):
    try:
        return '[' + ','.join('%.3f' % float(x) for x in v) + ']'
    except Exception:
        return str(v)


# ------------------------------------------------------------------ instruction
WORDS = {'once': 1, 'twice': 2, 'thrice': 3, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
         'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12}
COLORS = ('red', 'green', 'blue', 'yellow', 'purple', 'orange')


def parse_instruction(text):
    t = (text or '').lower()
    color = None
    m = re.search(r'pick up the (\w+) cube', t)
    if m and m.group(1) in COLORS:
        color = m.group(1)
    if color is None:
        for c in COLORS:
            if c in t:
                color = c
                break
    first = None
    m = re.search(r'(left|right)[- ]side target and then to the (left|right)', t)
    if m:
        first = m.group(1)
    else:
        m = re.search(r'(left|right)', t)
        if m:
            first = m.group(1)
    n = None
    m = re.search(r'(\w+)[- ]to[- ](\w+) swing motion (\w+) times?', t)
    cand = None
    if m:
        cand = m.group(3)
    else:
        m = re.search(r'(\w+) times', t)
        if m:
            cand = m.group(1)
        elif 'once' in t:
            cand = 'once'
        elif 'twice' in t:
            cand = 'twice'
    if cand is not None:
        if cand.isdigit():
            n = int(cand)
        else:
            n = WORDS.get(cand)
    return dict(color=color or 'red', first=first or 'right', n=n if n else 1,
                parsed=dict(color=color, first=first, n=n))


# ------------------------------------------------------------------ robot wrapper
class Robot:
    def __init__(self, ctx, log):
        self.ctx = ctx
        self.log = log
        self.meta = {}
        self.state = None
        self.step_id = None
        self.images = {}
        self.ended = False
        self.complete = None
        self.rgb_order = None      # index of front among unlabeled RGB images
        self.ncalls = 0
        self.nsteps = 0
        self.rot_sign = 1.0
        self.rot_hold_ok = False
        self.R_ref = None
        self.limits = None
        self.ref_front = None
        self.track = 0.8           # measured fraction of a commanded step that is executed

    # -- parsing
    def _ingest(self, result, from_observe):
        meta = U.reply_meta(result)
        self.meta = meta
        st = U.find_key(meta, 'state', lambda v: U.is_vec(v, 8))
        if st is None:
            for k in ('proprio', 'robot_state', 'proprioception'):
                st = U.find_key(meta, k, lambda v: U.is_vec(v, 8))
                if st is not None:
                    break
        if st is not None:
            self.state = np.array(st, float)
        sid = U.find_key(meta, 'step_id', lambda v: isinstance(v, int) and not isinstance(v, bool))
        got_sid = sid is not None
        if got_sid:
            self.step_id = sid
        ee = U.find_key(meta, 'episode_ended', lambda v: isinstance(v, bool))
        if ee:
            self.ended = True
        tc = U.find_key(meta, 'task_complete', lambda v: isinstance(v, bool))
        if tc is not None:
            self.complete = tc
            if tc:
                self.ended = True
        lim = U.find_key(meta, 'episode_limits')
        if lim is not None:
            self.limits = lim
        imgs = U.reply_images(result)
        got_imgs = self._assign_images(imgs, meta)
        return st is not None, got_sid, got_imgs

    def _assign_images(self, imgs, meta):
        new = {}
        rgbs = [i for i in imgs if i['kind'] == 'rgb' and i['arr'].ndim == 3]
        deps = [i for i in imgs if i['kind'] == 'depth' and i['arr'].ndim == 2]
        for i in rgbs:
            if i['camera'] in ('front', 'wrist'):
                new[(i['camera'], 'rgb')] = i['arr']
        unl = [i for i in rgbs if i['camera'] not in ('front', 'wrist')]
        if len(unl) == 1 and (('front', 'rgb') in new) != (('wrist', 'rgb') in new):
            # one labelled, one not: the unlabelled one is the other camera
            other = 'wrist' if ('front', 'rgb') in new else 'front'
            new[(other, 'rgb')] = unl[0]['arr']
            unl = []
        if unl and ('front', 'rgb') not in new and ('wrist', 'rgb') not in new:
            fi = self.rgb_order if self.rgb_order is not None else 0
            if len(unl) >= 2:
                new[('front', 'rgb')] = unl[fi % 2]['arr']
                new[('wrist', 'rgb')] = unl[(fi + 1) % 2]['arr']
            else:
                new[('front', 'rgb')] = unl[0]['arr']
        for i in deps:
            if i['camera'] in ('front', 'wrist'):
                new[(i['camera'], 'depth')] = i['arr']
        unld = [i for i in deps if i['camera'] not in ('front', 'wrist')]
        if len(unld) >= 2 and ('front', 'depth') not in new:
            fi = self.rgb_order if self.rgb_order is not None else 0
            new[('front', 'depth')] = unld[fi % 2]['arr']
            new[('wrist', 'depth')] = unld[(fi + 1) % 2]['arr']
        # file references (if images are not inline)
        if ('front', 'rgb') not in new:
            files = U.meta_files(meta)
            for (cam, kind), (path, mpu) in files.items():
                if cam in ('front', 'wrist') and kind in ('rgb', 'depth') and (cam, kind) not in new:
                    try:
                        new[(cam, kind)] = U.load_file(path)
                    except Exception:
                        pass
        if new:
            self.images = new
            return True
        return False

    def _check_end(self):
        if self.ended:
            raise EpisodeEnded()

    # -- tool calls
    def observe(self):
        self._check_end()
        r = self.ctx.call('robomme_observe')
        self.ncalls += 1
        self._ingest(r, True)
        return r

    def step(self, rows):
        self._check_end()
        rows = [[float(max(-1.0, min(1.0, x))) for x in row] for row in rows][:100]
        if not rows:
            return
        r = self.ctx.call('robomme_step', actions=rows)
        self.ncalls += 1
        self.nsteps += len(rows)
        got_state, got_sid, _ = self._ingest(r, False)
        if not (got_state and got_sid) and not self.ended:
            self.observe()
        if self.ended:
            self.log('episode ended after step call; complete=%s steps=%d' % (self.complete, self.nsteps))
            raise EpisodeEnded()

    def pos(self):
        return self.state[:3].copy()

    def R(self):
        return U.rotvec_to_mat(self.state[3:6])

    def fingers(self):
        return float(self.state[6] + self.state[7])

    def img(self, cam):
        return self.images.get((cam, 'rgb'))

    # -- 3-D points
    def points(self, cam, uvs, radius=1):
        """World points for pixels (u,v) in the CURRENT observation. Returns list of xyz|None."""
        out = []
        uvs = list(uvs)
        for k in range(0, len(uvs), 32):
            chunk = uvs[k:k + 32]
            res = None
            try:
                r = self.ctx.call('robomme_pixel_to_position', observation_step_id=int(self.step_id),
                                  depth_window_radius=int(radius),
                                  selections=[dict(camera=cam, u=float(u), v=float(v)) for (u, v) in chunk])
                self.ncalls += 1
                res = [p for (p, s) in U.parse_points(r, len(chunk))]
            except Exception as e:
                self.log('pixel_to_position failed:', repr(e)[:300])
                res = None
            if res is None or all(p is None for p in res):
                res = self._own_backproject(cam, chunk)
            out.extend(res)
        return out

    def _own_backproject(self, cam, uvs):
        try:
            dep = self.images.get((cam, 'depth'))
            intr = U.find_key(self.meta, 'camera_intrinsics')
            poses = U.find_key(self.meta, 'camera_poses')
            if dep is None or intr is None or poses is None:
                return [None] * len(uvs)
            K = intr.get(cam) if isinstance(intr, dict) else None
            if isinstance(K, dict):
                if 'matrix' in K:
                    K = K['matrix']
                elif 'fx' in K:
                    K = [[K['fx'], 0, K['cx']], [0, K['fy'], K['cy']], [0, 0, 1]]
            pose = poses.get(cam) if isinstance(poses, dict) else None
            pos = U.find_key(pose, 'position', lambda v: U.is_vec(v, 3))
            quat = U.find_key(pose, 'quaternion_wxyz', lambda v: U.is_vec(v, 4))
            conv = 'opengl'
            if isinstance(pose, dict) and 'convention' in pose and 'cv' in str(pose['convention']).lower():
                conv = 'opencv'
            mpu = 0.001
            out = []
            for (u, v) in uvs:
                ui, vi = int(round(u)), int(round(v))
                win = dep[max(0, vi - 1):vi + 2, max(0, ui - 1):ui + 2].astype(float)
                win = win[win > 0]
                if win.size == 0:
                    out.append(None)
                    continue
                d = float(np.median(win)) * mpu
                out.append(U.backproject(u, v, d, K, pos, quat, conv))
            return out
        except Exception as e:
            self.log('own backprojection failed', repr(e)[:200])
            return [None] * len(uvs)

    # -- motion
    def _rot_correction(self):
        if not (self.rot_hold_ok and self.R_ref is not None):
            return np.zeros(3)
        try:
            E = self.R_ref @ self.R().T
            rv = U.mat_to_rotvec(E)
            ang = np.linalg.norm(rv)
            if ang < 0.035 or ang > 0.6:
                return np.zeros(3)
            return np.clip(rv * 0.6 / ROT_SCALE, -0.12, 0.12) * self.rot_sign
        except Exception:
            return np.zeros(3)

    def move_to(self, target, speed, grip, tol=0.004, max_calls=40, chunk_max=10, label=''):
        target = np.asarray(target, float)
        hist = []
        for it in range(max_calls):
            p = self.pos()
            err = target - p
            d = float(np.linalg.norm(err))
            hist.append(d)
            if d < tol:
                return True
            if len(hist) >= 4 and hist[-4] - hist[-1] < 0.002:
                self.log('move_to stall', label, 'err', '%.4f' % d, 'pos', fmt(p))
                return False
            n = int(min(chunk_max, max(1, math.ceil(d / speed))))
            # compensate a measured sluggish response far from the goal only (no overshoot near it)
            gain = 1.0 / max(self.track, 0.4) if d > 0.02 else 1.0
            per = err / n * gain
            if np.linalg.norm(per) * (n - 1) > d:  # never command beyond the goal over the chunk
                per = err / n
            rows = []
            rc = self._rot_correction()
            for k in range(n):
                r = list(per / POS_SCALE) + (list(rc) if k == 0 else [0.0, 0.0, 0.0]) + [grip]
                rows.append(r)
            self.step(rows)
            moved = self.pos() - p
            cmdv = per * n
            cn = float(np.linalg.norm(cmdv))
            if cn > 0.006:
                r_ = float(np.dot(moved, cmdv) / (cn * cn))
                r_ = min(1.0, max(0.2, r_))
                self.track = 0.7 * self.track + 0.3 * r_
        self.log('move_to max_calls', label, 'err', '%.4f' % float(np.linalg.norm(target - self.pos())))
        return float(np.linalg.norm(target - self.pos())) < 3 * tol

    def hold(self, n, grip):
        self.step([[0, 0, 0, 0, 0, 0, grip]] * n)

    def descend_until_contact(self, z_goal, grip, speed=SPEED_SLOW, max_steps=40, z_expect=None,
                              gate=0.008, label=''):
        """Move down in single steps to z_goal; stop on a stall (contact).
        A stall = a step that moved < 30% of the free-motion response (measured above contact).
        It counts as contact after 2 stalls below z_expect+gate, or 6 stalls anywhere.
        Returns (contact, z)."""
        lag = 0
        ratios = []
        for k in range(max_steps):
            p = self.pos()
            dz = z_goal - p[2]
            if dz > -0.0015:
                return False, p[2]
            cmd = max(dz, -speed)
            self.step([[0, 0, cmd / POS_SCALE, 0, 0, 0, grip]])
            z = self.pos()[2]
            ratio = (z - p[2]) / cmd  # >0 when moving down as commanded
            ref = float(np.median(ratios)) if len(ratios) >= 2 else self.track
            if ratio < 0.3 * ref:
                lag += 1
                near = z_expect is None or z < z_expect + gate
                if (near and lag >= 2) or lag >= 6:
                    self.log('contact detected', label, 'z=%.4f' % z, 'expected', z_expect,
                             'free-ratio %.2f' % ref)
                    return True, z
            else:
                lag = 0
                if z_expect is None or p[2] > z_expect + 0.010:
                    ratios.append(ratio)
        return False, self.pos()[2]

    def yaw_col(self):
        R = self.R()
        i = int(np.argmin(np.abs(R[2, :2])))  # most horizontal of x,y columns
        return i

    def yaw_of(self, col):
        R = self.R()
        return math.atan2(R[1, col], R[0, col])

    def rotate_yaw(self, delta, grip):
        """Rotate about world z by delta (rad), closed loop on the measured orientation."""
        col = self.yaw_col()
        y0 = self.yaw_of(col)
        sign = self.rot_sign
        flipped = False
        for it in range(10):
            ach = U.wrap(self.yaw_of(col) - y0)
            rem = U.wrap(delta - ach)
            if abs(rem) < 0.03:
                break
            stepang = max(-0.25, min(0.25, rem))
            before = self.yaw_of(col)
            self.step([[0, 0, 0, 0, 0, sign * stepang / ROT_SCALE, grip], [0, 0, 0, 0, 0, 0, grip]])
            got = U.wrap(self.yaw_of(col) - before)
            if it == 0 and abs(got) > 0.02 and got * stepang < 0 and not flipped:
                sign = -sign
                flipped = True
                self.log('rotation sign flipped (local-frame behaviour?)')
        self.rot_sign = sign
        self.rot_hold_ok = not flipped
        ach = U.wrap(self.yaw_of(col) - y0)
        self.log('rotate_yaw want %.3f got %.3f sign %+.0f' % (delta, ach, sign))
        return abs(U.wrap(delta - ach)) < 0.08


# ------------------------------------------------------------------ the task
class Swing:
    def __init__(self, ctx):
        self.log = Log()
        self.rb = Robot(ctx, self.log)
        self.log.steps_fn = lambda: self.rb.nsteps
        self.task = None
        self.table_z = None
        self.cube_side = DEFAULT_SIDE
        self.cube_xy0 = None        # where the cube was picked (put-down spot)
        self.cube_px = None         # front pixel of cube
        self.grasp_offset = None    # TCP z - cube bottom while held
        self.targets = {}           # side -> dict(px=(u,v), box, xyz)
        self.button = None
        self.visit_idx = 0
        self.use_contact = False    # switched on if hovering visits do not register
        self.visits_needed = 0
        self.phase_fail = {}
        self.home = None

    # ---------------- perception
    def perceive_scene(self):
        rb = self.rb
        rb.observe()
        front = rb.img('front')
        if front is None:
            raise RuntimeError('no front image')
        cands = P.find_fixtures(front)
        if len(cands) < 3:
            # maybe the camera mapping is swapped
            wr = rb.img('wrist')
            if wr is not None and len(P.find_fixtures(wr)) >= 3 and rb.rgb_order is None:
                self.log('front/wrist swap: fixtures found in the other image')
                rb.rgb_order = 1
                rb.images[('front', 'rgb')], rb.images[('wrist', 'rgb')] = wr, front
                front = wr
                cands = P.find_fixtures(front)
        heights = []
        for b in cands[:8]:
            pts = rb.points('front', P.sample_points(b, 9, margin=0.3))
            zs = [p[2] for p in pts if p is not None]
            heights.append(float(np.percentile(zs, 80)) if zs else None)
        self.log('fixture candidates', [(round(b['cx']), round(b['cy']), b['n'], b['w'], b['h'],
                                         None if h is None else round(h, 3)) for b, h in zip(cands, heights)])
        hh = heights if all(h is not None for h in heights) else None
        button, ts = P.classify_fixtures(cands, hh)
        ts = sorted(ts, key=lambda b: b['cx'])
        # image-left target is the robot's RIGHT-side target (verified on demos ep0 & ep2)
        sides = {'right': ts[0], 'left': ts[1]}
        for side, b in sides.items():
            c = self._fixture_point(front, b)
            self.targets[side] = dict(px=(b['cx'], b['cy']), box=(b['x0'] - 2, b['y0'] - 2, b['x1'] + 2, b['y1'] + 2),
                                      xyz=c, n=b['n'], base_lit=P.red_lit_fraction(front, (b['x0'] - 2, b['y0'] - 2, b['x1'] + 2, b['y1'] + 2)))
        bpt = self._fixture_point(front, button, top_pct=90, top_xy=True)
        self.button = dict(px=(button['cx'], button['cy']), xyz=bpt, box=(button['x0'], button['y0'], button['x1'], button['y1']))
        self.log('targets', {k: (fmt(v['xyz']), tuple(round(x) for x in v['px'])) for k, v in self.targets.items()},
                 'button', fmt(bpt), tuple(round(x) for x in self.button['px']))
        if self.targets['right']['xyz'] is not None and self.targets['left']['xyz'] is not None:
            yr, yl = self.targets['right']['xyz'][1], self.targets['left']['xyz'][1]
            self.log('world-y check: right y=%.3f left y=%.3f (%s)' % (yr, yl, 'consistent' if yr < yl else 'INCONSISTENT with +x-facing robot'))
        # table height: neutral-free table pixels below the fixtures
        self.table_z = self._table_z_front(front, cands)
        self.log('table_z', self.table_z)

    def _fixture_point(self, img, b, top_pct=60, top_xy=False):
        pts = self.rb.points('front', P.sample_points(b, 24, margin=0.3))
        good = [p for p in pts if p is not None]
        if not good:
            pts = self.rb.points('front', [(b['cx'], b['cy'])], radius=2)
            good = [p for p in pts if p is not None]
            if not good:
                return None
        g = np.array(good)
        z = float(np.percentile(g[:, 2], top_pct))
        if top_xy:
            sel = g[g[:, 2] > z - 0.004]
            if len(sel) >= 3:
                g = sel
        xy = np.median(g[:, :2], axis=0)
        return np.array([xy[0], xy[1], z])

    def _table_z_front(self, img, cands):
        h, w = img.shape[:2]
        uvs = []
        for (u, v) in [(40, 200), (128, 200), (216, 200), (60, 150), (196, 150), (128, 160)]:
            uvs.append((u, v))
        pts = [p for p in self.rb.points('front', uvs, radius=2) if p is not None]
        if not pts:
            return None
        return float(np.median([p[2] for p in pts]))

    def find_cube_front(self, prefer=None):
        front = self.rb.img('front')
        if self.targets:
            front = front.copy()
            for t in self.targets.values():
                x0, y0, x1, y1 = [int(v) for v in t['box']]
                front[max(0, y0 - 2):y1 + 3, max(0, x0 - 2):x1 + 3] = 0
        b = P.find_cube(front, self.task['color'], ymin=20, prefer=prefer)
        if b is None:
            return None, None
        pts = self.rb.points('front', P.sample_points(b, 12, margin=0.3))
        good = [p for p in pts if p is not None]
        if not good:
            return b, None
        g = np.array(good)
        return b, np.array([np.median(g[:, 0]), np.median(g[:, 1]), float(np.max(g[:, 2]))])

    def wrist_cube(self):
        """Refine the cube from the wrist camera: (center_xy, top_z, yaw, table_z, side) or None."""
        rb = self.rb
        rb.observe()
        wr = rb.img('wrist')
        if wr is None:
            return None
        b = P.find_cube(wr, self.task['color'], min_px=20, prefer=(128, 128))
        if b is None:
            self.log('wrist: cube not found')
            return None
        uvs = P.sample_points(b, 64, margin=0.0)
        pts = rb.points('wrist', uvs, radius=0)
        good = np.array([p for p in pts if p is not None]) if any(p is not None for p in pts) else None
        if good is None or len(good) < 6:
            return None
        ztop = float(np.percentile(good[:, 2], 90))
        top = good[good[:, 2] > ztop - 0.006]
        if len(top) < 5:
            top = good
        yaw, ext = P.min_area_yaw(top[:, :2])
        # rect center in rotated frame
        if yaw is not None:
            c, s = math.cos(yaw), math.sin(yaw)
            u = top[:, 0] * c + top[:, 1] * s
            v = -top[:, 0] * s + top[:, 1] * c
            uc, vc = (u.max() + u.min()) / 2, (v.max() + v.min()) / 2
            cxy = np.array([uc * c - vc * s, uc * s + vc * c])
        else:
            cxy = np.median(top[:, :2], axis=0)
        # table ring around the cube in the wrist image
        ring = []
        bw = max(b['x1'] - b['x0'], b['y1'] - b['y0'])
        for ang in np.linspace(0, 2 * math.pi, 12, endpoint=False):
            u = b['cx'] + math.cos(ang) * (0.5 * bw + 10)
            v = b['cy'] + math.sin(ang) * (0.5 * bw + 10)
            if 2 <= u <= 253 and 2 <= v <= 253:
                ring.append((u, v))
        tz = None
        if ring:
            rp = [p for p in rb.points('wrist', ring, radius=1) if p is not None]
            if rp:
                zs = np.array([p[2] for p in rp])
                zs = zs[zs < ztop - 0.01]
                if len(zs):
                    tz = float(np.median(zs))
        side = None
        if tz is not None:
            side = ztop - tz
        self.log('wrist cube: px (%.0f,%.0f) n=%d center %s top %.4f yaw %s ext %s table %s side %s' % (
            b['cx'], b['cy'], b['n'], fmt(cxy), ztop, None if yaw is None else '%.1fdeg' % math.degrees(yaw),
            None if ext is None else fmt(ext), tz, side))
        return dict(xy=cxy, top=ztop, yaw=yaw, table=tz, side=side, npx=b['n'])

    def finger_axis_yaw(self):
        """World yaw of the wrist image horizontal axis (= finger closing axis), via depth."""
        rb = self.rb
        try:
            pts = rb.points('wrist', [(50, 100), (206, 100), (50, 130), (206, 130)], radius=2)
            dirs = []
            for a, b in ((0, 1), (2, 3)):
                if pts[a] is not None and pts[b] is not None:
                    d = pts[b][:2] - pts[a][:2]
                    if np.linalg.norm(d) > 0.02:
                        dirs.append(math.atan2(d[1], d[0]))
            img_yaw = dirs[0] if dirs else None
            if len(dirs) == 2 and abs((dirs[0] - dirs[1] + math.pi / 2) % math.pi - math.pi / 2) > math.radians(10):
                self.log('finger axis: image rows disagree (%.1f vs %.1f deg); using R[:,1]' % (
                    math.degrees(dirs[0]), math.degrees(dirs[1])))
                img_yaw = None
        except Exception:
            img_yaw = None
        R = rb.R()
        ry = math.atan2(R[1, 1], R[0, 1])
        self.log('finger axis: image-based %s, R[:,1] %.1fdeg' % (
            None if img_yaw is None else '%.1fdeg' % math.degrees(img_yaw), math.degrees(ry)))
        if img_yaw is not None:
            # both are axes (mod pi); if they disagree strongly, still prefer the image (demo-verified)
            return img_yaw
        return ry

    # ---------------- phases
    def phase_grasp(self):
        rb = self.rb
        for attempt in range(MAX_GRASP_ATTEMPTS):
            if attempt > 0 or self.visit_idx > 0:
                # clear the front view before looking for the cube again
                q = rb.pos()
                if self.table_z is not None and q[2] < self.table_z + 0.15:
                    q[2] = self.table_z + 0.15
                    rb.move_to(q, SPEED_FREE, -1, label='clear-up')
                if self.home is not None:
                    rb.move_to(self.home, SPEED_FREE, -1, tol=0.01, label='home')
            rb.observe()
            b, c = self.find_cube_front(prefer=self.cube_px)
            if c is None:
                self.log('grasp: cube not found in front image (attempt %d)' % attempt)
                # look from above the previous spot with the wrist camera
                if self.cube_xy0 is None:
                    return False
                c = np.array([self.cube_xy0[0], self.cube_xy0[1], (self.table_z or 0) + self.cube_side])
            else:
                self.cube_px = (b['cx'], b['cy'])
            self.log('grasp attempt %d: front cube %s' % (attempt, fmt(c)))
            tz = self.table_z if self.table_z is not None else c[2] - DEFAULT_SIDE
            # 1) above the cube, gripper open
            above = np.array([c[0], c[1], max(c[2], tz + 0.02) + APPROACH_ABOVE])
            if rb.pos()[2] < tz + 0.06:
                up = rb.pos()
                up[2] = above[2]
                rb.move_to(up, SPEED_FREE, -1, label='lift-before-approach')
            rb.move_to(above, SPEED_FREE, -1, label='above-cube')
            info = self.wrist_cube()
            if info is not None and np.linalg.norm(info['xy'] - c[:2]) < 0.08:
                c = np.array([info['xy'][0], info['xy'][1], info['top']])
                rb.move_to(np.array([c[0], c[1], c[2] + REFINE_ABOVE]), SPEED_FREE, -1, label='refine')
                info2 = self.wrist_cube()
                if info2 is not None and np.linalg.norm(info2['xy'] - c[:2]) < 0.04:
                    info = info2
                    c = np.array([info['xy'][0], info['xy'][1], info['top']])
                if info['table'] is not None and abs(info['table'] - tz) < 0.03:
                    tz = 0.5 * (tz + info['table']) if self.table_z is not None else info['table']
            side = c[2] - tz
            if not (0.015 <= side <= 0.08):
                self.log('implausible cube side %.3f; using default' % side)
                side = DEFAULT_SIDE
                tz = c[2] - side
            self.cube_side = side
            # 2) yaw alignment
            if info is not None and info.get('yaw') is not None:
                try:
                    fy = self.finger_axis_yaw()
                    delta = U.wrap90(info['yaw'] - fy)
                    self.log('yaw align: cube %.1f finger %.1f delta %.1f deg' % (
                        math.degrees(info['yaw']), math.degrees(fy), math.degrees(delta)))
                    if abs(delta) > math.radians(6):
                        rb.rotate_yaw(delta, -1)
                except EpisodeEnded:
                    raise
                except Exception as e:
                    self.log('yaw align failed', repr(e)[:200])
            rb.R_ref = rb.R()
            # 3) descend and close
            gz = tz + max(0.5 * side, 0.012)
            gz = min(gz, c[2] - 0.008)
            pre = np.array([c[0], c[1], gz + 0.04])
            rb.move_to(pre, SPEED_FREE, -1, label='pre-grasp')
            rb.move_to(np.array([c[0], c[1], gz]), SPEED_SLOW * 1.5, -1, tol=0.003, label='grasp-down')
            rb.hold(10, 1.0)
            f = rb.fingers()
            self.log('closed: fingers %.4f, tcp %s' % (f, fmt(rb.pos())))
            # 4) lift
            self.grasp_offset = rb.pos()[2] - tz
            self.table_z = tz
            lift = rb.pos()
            lift[2] = tz + self.grasp_offset + TRAVEL_CLEAR
            rb.move_to(lift, SPEED_CARRY * 0.7, 1.0, label='lift')
            rb.hold(2, 1.0)
            f = rb.fingers()
            held = GRIP_MIN_OPEN < f < GRIP_MAX_HELD
            # second signal: cube no longer on the table at its old pixel
            rb.observe()
            b2 = P.find_cube(rb.img('front'), self.task['color'], ymin=20, prefer=self.cube_px)
            moved = None
            if b2 is not None and self.cube_px is not None:
                moved = math.hypot(b2['cx'] - self.cube_px[0], b2['cy'] - self.cube_px[1])
            self.log('lift check: fingers %.4f held=%s cube-blob moved %s px' % (f, held, moved))
            if held:
                if self.cube_xy0 is None:
                    self.cube_xy0 = c[:2].copy()
                return True
            # failed: open and retry
            rb.hold(6, -1.0)
            self.cube_px = None
        return False

    def holding(self):
        f = self.rb.fingers()
        return GRIP_MIN_OPEN < f < GRIP_MAX_HELD

    def travel_z(self):
        return self.table_z + self.grasp_offset + TRAVEL_CLEAR

    def visit(self, side, retry=False):
        rb = self.rb
        t = self.targets[side]
        txyz = t['xyz']
        tzt = txyz[2]
        # contact height: cube bottom on target top
        z_touch = tzt + self.grasp_offset
        p = rb.pos()
        if p[2] < self.travel_z() - 0.02:
            q = p.copy()
            q[2] = self.travel_z()
            rb.move_to(q, SPEED_CARRY, 1.0, label='rise')
        rb.move_to(np.array([txyz[0], txyz[1], self.travel_z()]), SPEED_CARRY, 1.0, tol=0.004, label='over-' + side)
        rb.move_to(np.array([txyz[0], txyz[1], z_touch + 0.02]), SPEED_CARRY, 1.0, tol=0.004, label='down-' + side)
        if retry or self.use_contact:
            goal = z_touch - TOUCH_PRESS - TOUCH_EXTRA
        else:
            goal = z_touch + HOVER_GAP
        contact, z = rb.descend_until_contact(goal, 1.0, z_expect=z_touch, label=side)
        rb.hold(6 if retry else 4, 1.0)
        bottom = None
        if self.task['color'] != 'red':
            try:
                bottom = self.lit_state(side)
            except EpisodeEnded:
                raise
            except Exception:
                bottom = None
        self.log('visit %d %s: target %s z_touch %.4f reached %.4f contact=%s fingers %.4f' % (
            self.visit_idx, side, fmt(txyz), z_touch, z, contact, rb.fingers()))
        q = rb.pos()
        q[2] = self.travel_z()
        rb.move_to(q, SPEED_CARRY, 1.0, label='up-' + side)
        return contact, z, bottom

    def lit_state(self, side):
        rb = self.rb
        rb.observe()
        img = rb.img('front')
        t = self.targets[side]
        red = P.red_lit_fraction(img, t['box'])
        x0, y0, x1, y1 = [int(v) for v in t['box']]
        patch = img[max(0, y0):y1 + 1, max(0, x0):x1 + 1]
        neu = float(P.neutral_mask(img)[max(0, y0):y1 + 1, max(0, x0):x1 + 1].mean()) if patch.size else 0.0
        other = 'left' if side == 'right' else 'right'
        red_o = P.red_lit_fraction(img, self.targets[other]['box'])
        return red, neu, red_o

    def phase_swing(self):
        rb = self.rb
        order = [self.task['first'], 'left' if self.task['first'] == 'right' else 'right']
        while self.visit_idx < self.visits_needed:
            if not self.holding():
                self.log('cube lost before visit %d (fingers %.4f)' % (self.visit_idx, rb.fingers()))
                return 'lost'
            side = order[self.visit_idx % 2]
            contact, z, bottom = self.visit(side)
            if not self.holding():
                self.log('cube lost during visit %d' % self.visit_idx)
                return 'lost'
            red, neu, red_o = self.lit_state(side)
            self.log('lit check visit %d %s: red %.2f neutral %.2f other-red %.2f bottom %s' % (
                self.visit_idx, side, red, neu, red_o, None if bottom is None else '%.2f/%.2f' % bottom[:2]))
            for rtry in range(2):
                lit_bottom = bottom is not None and bottom[0] >= 0.05
                if lit_bottom or not (red < 0.03 and neu > 0.25):
                    break
                # clearly visible, clearly not lit -> deeper, re-centred retry
                self.log('visit %d looks unregistered; retry %d' % (self.visit_idx, rtry))
                self.recentre_target(side)
                contact, z, bottom = self.visit(side, retry=True)
                if not self.holding():
                    return 'lost'
                red, neu, red_o = self.lit_state(side)
                self.log('lit re-check: red %.2f neutral %.2f other-red %.2f' % (red, neu, red_o))
                if red >= 0.05 or (bottom is not None and bottom[0] >= 0.05):
                    if not self.use_contact:
                        self.log('contact visit registered where hover did not: using contact from now on')
                    self.use_contact = True
            self.visit_idx += 1
        return 'ok'

    def recentre_target(self, side):
        """Re-detect the target near its stored pixel in the current front image."""
        try:
            img = self.rb.img('front')
            t = self.targets[side]
            cands = P.find_fixtures(img)
            best = None
            for b in cands:
                d = math.hypot(b['cx'] - t['px'][0], b['cy'] - t['px'][1])
                if d < 15 and 0.5 * t['n'] < b['n'] < 2.0 * t['n'] and (best is None or d < best[0]):
                    best = (d, b)
            if best is not None and best[0] > 1.0:
                b = best[1]
                c = self._fixture_point(img, b)
                if c is not None and np.linalg.norm(c[:2] - t['xyz'][:2]) < 0.05:
                    self.log('target %s re-centred %s -> %s' % (side, fmt(t['xyz']), fmt(c)))
                    t['xyz'] = c
                    t['px'] = (b['cx'], b['cy'])
        except EpisodeEnded:
            raise
        except Exception as e:
            self.log('recentre failed', repr(e)[:200])

    def phase_place(self):
        rb = self.rb
        xy = self.cube_xy0 if self.cube_xy0 is not None else rb.pos()[:2]
        # keep clear of targets / button
        for o in [self.targets['left']['xyz'], self.targets['right']['xyz'], self.button['xyz']]:
            if o is not None and np.linalg.norm(o[:2] - xy) < 0.05:
                self.log('put-down spot near a fixture; using current xy')
                xy = rb.pos()[:2]
        rb.move_to(np.array([xy[0], xy[1], self.travel_z()]), SPEED_CARRY, 1.0, label='over-place')
        zp = self.table_z + self.grasp_offset + 0.004
        rb.move_to(np.array([xy[0], xy[1], zp + 0.02]), SPEED_CARRY, 1.0, label='down-place')
        rb.descend_until_contact(zp - 0.008, 1.0, z_expect=zp - 0.004, label='place')
        rb.hold(8, -1.0)
        q = rb.pos()
        q[2] += 0.08
        rb.move_to(q, SPEED_FREE, -1.0, label='retreat')
        self.log('placed: fingers %.4f tcp %s' % (rb.fingers(), fmt(rb.pos())))
        return True

    def phase_press(self):
        rb = self.rb
        bxyz = self.button['xyz']
        if bxyz is None:
            raise RuntimeError('button position unknown')
        top = bxyz[2]
        # close the gripper while moving above the button
        above = np.array([bxyz[0], bxyz[1], top + 0.08])
        q = rb.pos()
        if q[2] < top + 0.06:
            q[2] = top + 0.08
            rb.move_to(q, SPEED_FREE, 1.0, label='rise-press')
        rb.move_to(above, SPEED_FREE, 1.0, label='above-button')
        rb.hold(6, 1.0)
        for k, depth in enumerate(BUTTON_PRESS_DEPTH):
            rb.move_to(np.array([bxyz[0], bxyz[1], top + 0.02]), SPEED_FREE, 1.0, tol=0.003, label='pre-press')
            # push straight down (no stop on resistance: the button must be pressed through)
            rb.move_to(np.array([bxyz[0], bxyz[1], top - depth]), 0.006, 1.0, tol=0.002, max_calls=12,
                       chunk_max=3, label='press')
            rb.hold(5, 1.0)
            self.log('press %d: top %.4f target %.4f reached %.4f' % (k, top, top - depth, rb.pos()[2]))
            q = rb.pos()
            q[2] = top + 0.05
            rb.move_to(q, SPEED_FREE, 1.0, label='release-press')
            rb.observe()
            if rb.ended:
                raise EpisodeEnded()
        return True

    # ---------------- main
    def run(self):
        rb = self.rb
        self.log('start', VERSION)
        rb.observe()
        instr = U.find_key(rb.meta, 'instruction', lambda v: isinstance(v, str))
        if not instr:
            for t in U.text_blocks({'content': []}):
                pass
        self.task = parse_instruction(instr or '')
        self.visits_needed = 2 * self.task['n']
        self.log('instruction:', repr(instr), '->', self.task, 'visits', self.visits_needed)
        self.log('state', fmt(rb.state) if rb.state is not None else None, 'step_id', rb.step_id,
                 'images', sorted(rb.images.keys()), 'limits', json.dumps(rb.limits)[:200] if rb.limits else None)
        self._calibrate_camera_order()
        self.home = rb.pos()

        phase = 'perceive'
        tries = {}
        while phase != 'done':
            tries[phase] = tries.get(phase, 0) + 1
            if tries[phase] > 6:
                self.log('phase %s exceeded retries; stopping' % phase)
                break
            try:
                if phase == 'perceive':
                    self.perceive_scene()
                    phase = 'grasp'
                elif phase == 'grasp':
                    ok = self.phase_grasp()
                    phase = 'swing' if ok else 'grasp'
                elif phase == 'swing':
                    r = self.phase_swing()
                    phase = 'place' if r == 'ok' else 'grasp'
                elif phase == 'place':
                    self.phase_place()
                    phase = 'press'
                elif phase == 'press':
                    self.phase_press()
                    phase = 'done'
            except EpisodeEnded:
                self.log('episode ended in phase', phase, 'complete=', rb.complete)
                return
            except Exception as e:
                self.log('EXCEPTION in phase', phase, repr(e)[:300])
                self.log(traceback.format_exc()[-1500:])
                try:
                    rb.observe()
                except EpisodeEnded:
                    return
                except Exception:
                    time.sleep(0.5)
        self.log('finished: phase=%s visits=%d/%d steps=%d calls=%d complete=%s' % (
            phase, self.visit_idx, self.visits_needed, rb.nsteps, rb.ncalls, rb.complete))

    def _calibrate_camera_order(self):
        """If RGB images were unlabeled, check order with the reset frame manifest (if any)."""
        rb = self.rb
        try:
            seq = U.find_key(rb.meta, 'frame_sequence')
            mp = seq.get('manifest_path') if isinstance(seq, dict) else None
            if not mp:
                return
            man = json.load(open(os.path.join(U.WORKSPACE, mp)))
            frames = man.get('frames', [])
            ex = [f for f in frames if f.get('phase') == 'execution'] or frames[-1:]
            if not ex:
                return
            files = {(f.get('camera'), f.get('kind')): f.get('path') for f in ex[-1].get('files', [])}
            fp = files.get(('front', 'rgb'))
            if not fp:
                return
            fimg = U.load_file(fp).astype(np.int32)
            cur_f, cur_w = rb.img('front'), rb.img('wrist')
            if cur_f is None or cur_w is None or cur_f.shape != fimg.shape:
                return
            df = np.abs(cur_f.astype(np.int32) - fimg).mean()
            dw = np.abs(cur_w.astype(np.int32) - fimg).mean()
            self.log('camera order check vs manifest: d(front)=%.1f d(wrist)=%.1f' % (df, dw))
            if dw + 3 < df:
                self.log('swapping front/wrist mapping')
                rb.rgb_order = 1
                rb.images[('front', 'rgb')], rb.images[('wrist', 'rgb')] = cur_w, cur_f
                if ('front', 'depth') in rb.images and ('wrist', 'depth') in rb.images:
                    rb.images[('front', 'depth')], rb.images[('wrist', 'depth')] = rb.images[('wrist', 'depth')], rb.images[('front', 'depth')]
        except Exception as e:
            self.log('camera order check skipped:', repr(e)[:200])


def main(ctx):
    s = Swing(ctx)
    try:
        s.run()
    except EpisodeEnded:
        pass
    except Exception as e:
        s.log('FATAL', repr(e)[:300])
        s.log(traceback.format_exc()[-2000:])
