"""SwingXtimes controller (RAX, RoboMME-Plus).  Version: see VERSION.

Transition table
  INIT      -> parse instruction (colour, first/second side, N swings); exits: always -> SCENE
  SCENE     -> detect table z, cube, 2 targets, button (front RGB + depth via pixel_to_position);
               retries with short holds (moving light spots).  fail after retries -> best effort / return
  PICK      -> hover, wrist refine (xy, top z, yaw), rotate yaw, descend, close, lift, verify fingers.
               fail -> re-detect cube and retry (<=4) ; still failing -> return
  VISIT i   -> (i = 0..2N-1, alternating first/second target) ascend, relocalize target (front),
               transit at carry height (waypoint around the other target on the first transit),
               descend to contact height, hold until highlight (or timeout, then press 5 mm lower).
               dropped cube -> PICK then repeat VISIT i (<=2 recoveries)
  RELEASE   -> open gripper on the last target, ascend
  BUTTON    -> close gripper, hover over button, descend with stall detection, ascend; second deeper
               press only if the first stalled high / missed and episode not complete
  DONE      -> return
Global guards: wall-clock deadline (WALL_LIMIT_S), episode_ended, every phase in try/except.
"""
import math
import os
import re
import time
import traceback

import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import geom  # noqa: E402
import percep as P
from rio import Robot, log, LOG_T0

VERSION = 'swing-v1'

# ------------------------------------------------------------------ tunables (untested live: R1, no exploration)
STEP_M = 0.05            # action unit -> metres (documented)
ROT_RAD = 0.5            # action unit -> radians (documented)
V_FREE = 0.6             # max action norm in free space (3 cm/step)
V_CARRY = 0.35           # with cube (1.75 cm/step)
V_FINE = 0.22            # final descents
MAX_ROWS_PER_CALL = 4
BOOST_MAX = 2.0          # far-field command gain bound (1/eff), only beyond BOOST_MIN_DIST
BOOST_MIN_DIST = 0.03
HOVER_ABOVE_TOP = 0.12   # first hover above cube top
HOVER2_ABOVE_TOP = 0.07
CARRY_LIFT = 0.09        # cube bottom height above table during transit
VISIT_GAP = 0.001        # cube bottom above target surface at first contact attempt
VISIT_PRESS = 0.005      # extra lowering if no highlight seen
HOLD_MIN = 8
HOLD_MAX = 12
AVOID_R = 0.09           # first transit keeps this xy distance from the other target
WALL_LIMIT_S = 1020.0    # controller hard limit is 1200 s
COLORS = ['red', 'green', 'blue', 'yellow']
NUMS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
        'nine': 9, 'ten': 10, 'once': 1, 'twice': 2, 'thrice': 3}


class Abort(Exception):
    pass


def parse_instruction(text):
    t = (text or '').lower()
    color = 'red'
    m = re.search(r'pick up the (\w+) cube', t)
    if m and m.group(1) in COLORS:
        color = m.group(1)
    else:
        for c in COLORS:
            if c + ' cube' in t:
                color = c
                break
    first = 'right'
    m = re.search(r'to the (left|right)[- ]side target and then to the (left|right)', t)
    if m:
        first = m.group(1)
    else:
        m = re.search(r'(left|right)[- ]to[- ](left|right)', t)
        if m:
            first = m.group(1)
    n = None
    m = re.search(r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+times', t)
    if m:
        g = m.group(1)
        n = int(g) if g.isdigit() else NUMS[g]
    else:
        for w in ('once', 'twice', 'thrice'):
            if w in t:
                n = NUMS[w]
    if n is None:
        n = 3
    return color, first, n


class Swing:
    def __init__(self, ctx):
        self.r = Robot(ctx)
        self.t0 = time.time()
        self.R_ref = None
        self.table_z = None
        self.cube = None       # dict(xy, top, h, uv)
        self.targets = {}      # side -> dict(p, uv, box, base_pink, base_red)
        self.button = None
        self.z_g = None
        self.color = 'red'
        self.n_img = 0
        self.toward_cam = None
        self.eff = 0.8            # online estimate of achieved / commanded displacement
        self.rot_mode = 'world'   # documented convention; switched on observed divergence
        self.rot_strikes = 0

    # ------------------------------------------------------------ utilities
    def check(self):
        if self.r.ended:
            raise Abort('episode ended')
        if time.time() - self.t0 > WALL_LIMIT_S:
            raise Abort('wall limit')

    def pos(self):
        return np.array(self.r.state[:3], dtype=float)

    def Rcur(self):
        return geom.aa_to_mat(self.r.state[3:6])

    def opening(self):
        return float(self.r.state[6] + self.r.state[7])

    def save_img(self, name, cam='front'):
        try:
            if self.n_img > 40 or cam not in self.r.rgb:
                return
            from PIL import Image
            os.makedirs('/workspace/output/img', exist_ok=True)
            Image.fromarray(self.r.rgb[cam]).save('/workspace/output/img/%02d_%s_%s.png' % (self.n_img, name, cam))
            self.n_img += 1
        except Exception:
            pass

    def rot_cmd(self, k):
        if self.R_ref is None:
            return [0.0, 0.0, 0.0]
        E = self.R_ref @ self.Rcur().T
        w = geom.mat_to_aa(E)
        ang = float(np.linalg.norm(w))
        if ang < 0.008 or self.rot_mode == 'off':
            return [0.0, 0.0, 0.0]
        if self.rot_mode == 'tool':
            w = self.Rcur().T @ w
        w = w / k / ROT_RAD
        nrm = float(np.linalg.norm(w))
        if nrm > 0.3:
            w = w * 0.3 / nrm
        return [float(x) for x in w]

    def rot_err(self):
        if self.R_ref is None:
            return 0.0
        return float(np.linalg.norm(geom.mat_to_aa(self.R_ref @ self.Rcur().T)))

    def watch_rotation(self, re0, rc):
        if not any(abs(x) > 1e-6 for x in rc):
            return
        re1 = self.rot_err()
        if re1 > re0 + 0.02:
            self.rot_strikes += 1
        elif re1 < re0 - 0.005:
            self.rot_strikes = 0
        if self.rot_strikes >= 2:
            nxt = {'world': 'tool', 'tool': 'off'}.get(self.rot_mode, 'off')
            log('ROTATION DIVERGES in mode %s (err %.3f -> %.3f); switching to %s' % (self.rot_mode, re0, re1, nxt))
            self.rot_mode = nxt
            self.rot_strikes = 0
            if nxt == 'off':
                self.R_ref = self.Rcur()

    def hold(self, n, grip):
        self.check()
        rows = []
        k = max(1, n)
        rc = self.rot_cmd(k)
        for _ in range(n):
            rows.append([0.0, 0.0, 0.0] + rc + [grip])
        re0 = self.rot_err()
        self.r.step(rows)
        self.watch_rotation(re0, rc)

    def move_to(self, target, grip, vmax=V_FREE, tol=0.006, max_steps=80, rot_tol=0.04, name=''):
        target = np.asarray(target, dtype=float)
        used = 0
        best = 1e9
        stall = 0
        while True:
            self.check()
            err = target - self.pos()
            d = float(np.linalg.norm(err))
            if d < tol and self.rot_err() < rot_tol:
                return True
            if used >= max_steps:
                log('move_to %s: step budget out, err %.4f' % (name, d))
                return False
            if d < best - 0.0015:
                best = d
                stall = 0
            else:
                stall += 1
                if stall >= 4 and self.rot_err() < rot_tol:
                    log('move_to %s: stalled at err %.4f' % (name, d))
                    return False
            boost = min(BOOST_MAX, max(1.0, 1.0 / self.eff)) if d > BOOST_MIN_DIST else 1.0
            vstep = min(1.0, vmax * boost) * STEP_M
            re0 = self.rot_err()
            k = int(min(MAX_ROWS_PER_CALL, max(1, math.ceil(d / vstep), math.ceil(re0 / 0.15))))
            k = min(k, max_steps - used)
            per = err * boost / k
            pn = float(np.linalg.norm(per))
            if pn > vstep:
                per = per * vstep / pn
            a = [float(x) for x in per / STEP_M]
            rc = self.rot_cmd(k)
            p0 = self.pos()
            self.r.step([a + rc + [grip]] * k)
            used += k
            self.watch_rotation(re0, rc)
            cmd = per * k
            cn = float(np.linalg.norm(cmd))
            if cn > 0.01 and self.r.state is not None:
                e = float((self.pos() - p0) @ cmd) / (cn * cn)
                self.eff = 0.7 * self.eff + 0.3 * min(1.2, max(0.2, e))

    # ------------------------------------------------------------ 3-D helpers
    def world(self, cam, uvs, radius=1):
        return self.r.pixels_to_world(cam, uvs, radius)

    def measure_table(self):
        img = self.r.rgb['front']
        wm = P.wood_mask(img)
        H, W = wm.shape
        cand = []
        for v in range(30, 200, 17):
            for u in range(20, 236, 23):
                if wm[max(0, v - 2):v + 3, max(0, u - 2):u + 3].all():
                    cand.append((u, v))
        cand = cand[:32]
        raw = self.world('front', cand)
        pts = [p for p in raw if p is not None]
        if len(pts) < 3:
            return None
        zs = np.array([p[2] for p in pts])
        zt = float(np.median(zs))
        # horizontal direction toward the front camera = table motion for increasing image row
        vs = []
        for (u1, v1), p1 in zip(cand, raw):
            for (u2, v2), p2 in zip(cand, raw):
                if p1 is not None and p2 is not None and u1 == u2 and v2 > v1 and \
                        abs(p1[2] - zt) < 0.01 and abs(p2[2] - zt) < 0.01:
                    d = (p2 - p1)[:2]
                    if np.linalg.norm(d) > 1e-3:
                        vs.append(d / np.linalg.norm(d))
        if vs:
            m = np.mean(vs, axis=0)
            if np.linalg.norm(m) > 0.5:
                self.toward_cam = m / np.linalg.norm(m)
        return zt

    # ------------------------------------------------------------ perception
    def detect_cube_front(self):
        img = self.r.rgb['front']
        comps = [c for c in P.components(P.color_mask(img, self.color), 4)][:5]
        best = None
        for c in comps:
            uvs = [(c['cx'], c['cy'])] + P.sample_pixels(c, 6)
            pts = [p for p in self.world('front', uvs) if p is not None]
            if not pts:
                continue
            zs = np.array([p[2] for p in pts])
            top = float(np.max(zs))
            dz = top - self.table_z
            log('cube cand area=%d uv=(%.0f,%.0f) top_dz=%.3f' % (c['area'], c['cx'], c['cy'], dz))
            if 0.008 < dz < 0.09:
                xy = np.mean(np.array([p[:2] for p in pts]), axis=0)
                if self.toward_cam is not None:
                    xy = xy - 0.25 * dz * self.toward_cam  # visible faces are biased toward the camera
                if best is None or c['area'] > best['area']:
                    best = {'area': c['area'], 'xy': xy, 'top': top, 'uv': (c['cx'], c['cy'])}
        return best

    def detect_flat_objects(self):
        """Return (targets list, button or None) from the current front image."""
        img = self.r.rgb['front']
        nm = P.dilate(P.neutral_mask(img), 1)
        comps = []
        for c in P.components(nm, 15):
            w, h = c['x1'] - c['x0'] + 1, c['y1'] - c['y0'] + 1
            if c['y0'] <= 2 or w > 48 or h > 40 or c['area'] > 900:
                continue
            comps.append(c)
        comps = comps[:12]
        uvs, owners = [], []
        for i, c in enumerate(comps):
            s = [(c['cx'], c['cy'])] + P.sample_pixels(c, 5, shrink=0.8)
            uvs += s
            owners += [i] * len(s)
        pts = self.world('front', uvs) if uvs else []
        per = {}
        for i, p in zip(owners, pts):
            if p is not None:
                per.setdefault(i, []).append(p)
        targets, buttons = [], []
        pk, rr = P.pink_mask(img), P.ringred_mask(img)
        for i, c in enumerate(comps):
            ps = per.get(i, [])
            if len(ps) < 2:
                continue
            A = np.array(ps)
            dz = A[:, 2] - self.table_z
            zmax, zmed = float(dz.max()), float(np.median(dz))
            box = (c['x0'], c['x1'], c['y0'], c['y1'])
            rec = {'area': c['area'], 'uv': (c['cx'], c['cy']), 'box': box, 'zmax': zmax, 'zmed': zmed}
            log('flat cand area=%d uv=(%.0f,%.0f) wh=(%d,%d) dz med=%.3f max=%.3f' % (
                c['area'], c['cx'], c['cy'], box[1] - box[0] + 1, box[3] - box[2] + 1, zmed, zmax))
            if zmax < 0.013 and zmed > -0.01:
                c0 = pts[owners.index(i)]  # centroid pixel projects close to the disc centre
                cxy = c0[:2] if c0 is not None else np.median(A[:, :2], axis=0)
                rec['p'] = np.array([float(cxy[0]), float(cxy[1]), float(np.median(A[:, 2]))])
                rec['base_pink'] = P.count_in_box(pk, box)
                rec['base_red'] = P.count_in_box(rr, box)
                targets.append(rec)
            elif 0.013 <= zmax < 0.08 and zmed < 0.045:
                hi = A[A[:, 2] >= A[:, 2].max() - 0.008]
                rec['p'] = np.array([float(hi[:, 0].mean()), float(hi[:, 1].mean()), float(A[:, 2].max())])
                buttons.append(rec)
        return targets, buttons

    def scene(self):
        for attempt in range(6):
            self.check()
            if self.table_z is None or attempt % 2 == 1:
                tz = self.measure_table()
                if tz is not None:
                    self.table_z = tz if self.table_z is None else 0.5 * (self.table_z + tz)
            log('table_z', self.table_z)
            if self.table_z is None:
                self.hold(3, -1.0)
                continue
            cube = self.detect_cube_front()
            targets, buttons = self.detect_flat_objects()
            if cube is not None:
                self.cube = cube
            if not buttons and len(targets) >= 3:
                # low button misclassified as flat: take the highest flat candidate
                bt = max(targets, key=lambda t: t['zmax'])
                targets = [t for t in targets if t is not bt]
                buttons = [bt]
                log('button taken from flat candidates (zmax %.3f)' % bt['zmax'])
            # keep the two largest flat targets that are well separated
            targets.sort(key=lambda t: -t['area'])
            sel = []
            for t in targets:
                if all(np.linalg.norm(t['p'][:2] - s['p'][:2]) > 0.05 for s in sel):
                    sel.append(t)
                if len(sel) == 2:
                    break
            if len(sel) == 2 and max(sel[0]['area'], sel[1]['area']) > 1.8 * min(sel[0]['area'], sel[1]['area']) \
                    and attempt < 5:
                log('target areas inconsistent %d vs %d; retry' % (sel[0]['area'], sel[1]['area']))
                sel = []
            if len(sel) == 2:
                sel.sort(key=lambda t: t['uv'][0])
                # image-left = robot's right side (demos); image-right = left side
                self.targets = {'right': sel[0], 'left': sel[1]}
            if buttons:
                buttons.sort(key=lambda b: -b['area'])
                self.button = buttons[0]
            log('scene attempt %d: cube=%s targets=%d buttons=%d' % (
                attempt, None if cube is None else np.round(cube['xy'], 3).tolist(), len(targets), len(buttons)))
            if self.cube is not None and len(self.targets) == 2 and self.button is not None:
                break
            self.hold(4, -1.0)
            self.r.observe()
        for side, t in self.targets.items():
            log('target %s: p=%s uv=(%.0f,%.0f)' % (side, np.round(t['p'], 3).tolist(), t['uv'][0], t['uv'][1]))
        if self.button is not None:
            log('button p=%s' % np.round(self.button['p'], 3).tolist())
        if len(self.targets) == 2:
            dy = self.targets['left']['p'][1] - self.targets['right']['p'][1]
            log('cross-check: left.y - right.y = %.3f (expect >0 if robot faces +x)' % dy)
        self.save_img('scene')

    def wrist_view_cube(self, xy, z):
        """Shift the hover so the cube is centred in the wrist view, then refine."""
        w = None
        try:
            cands = [(128.0, 128.0), (128.0, 96.0), (128.0, 160.0), (96.0, 128.0), (160.0, 128.0)]
            pts = self.world('wrist', cands, radius=2)
            c0 = next((p for p in pts if p is not None), None)
            if c0 is not None:
                o = c0[:2] - self.pos()[:2]
                log('wrist view centre offset %s (h=%.3f)' % (np.round(o, 3).tolist(), self.pos()[2] - c0[2]))
                if 0.01 < np.linalg.norm(o) < 0.25:
                    v = np.asarray(xy) - o
                    self.move_to([v[0], v[1], z], -1.0, V_FREE, 0.006, 30, 0.04, 'view')
            self.save_img('view', 'wrist')
            w = self.wrist_cube()
        except Exception:
            log('wrist_view_cube error %s' % traceback.format_exc())
        if w is None:
            # fall back: view from straight above
            self.move_to([xy[0], xy[1], z], -1.0, V_FREE, 0.006, 30, 0.04, 'view-fallback')
            w = self.wrist_cube()
        return w

    def wrist_cube(self):
        """Refine cube xy/top/yaw from the wrist camera (current observation)."""
        img = self.r.rgb.get('wrist')
        if img is None:
            return None
        comps = P.components(P.color_mask(img, self.color), 25)
        if not comps:
            log('wrist: no cube pixels')
            return None
        c = comps[0]
        Hh, Ww = img.shape[:2]
        if c['x0'] <= 0 or c['y0'] <= 0 or c['x1'] >= Ww - 1 or c['y1'] >= Hh - 1:
            log('wrist: cube touches image border (partial view) area=%d' % c['area'])
            return None
        br = img[c['ys'], c['xs']].max(axis=1).astype(float)
        thr = 0.85 * np.percentile(br, 95)
        keep = br >= thr
        top = {'xs': c['xs'][keep], 'ys': c['ys'][keep]}
        if len(top['xs']) < 12:
            top = {'xs': c['xs'], 'ys': c['ys']}
        top['cx'], top['cy'] = float(top['xs'].mean()), float(top['ys'].mean())
        uvs = [(top['cx'], top['cy'])] + P.sample_pixels(top, 10, shrink=0.7)
        ang = P.min_area_angle(top['xs'], top['ys'])
        L = 0.3 * math.sqrt(len(top['xs']))
        if ang is not None:
            ca, sa = math.cos(ang), math.sin(ang)
            uvs += [(top['cx'] + L * ca, top['cy'] + L * sa), (top['cx'] - L * ca, top['cy'] - L * sa)]
        pts = self.world('wrist', uvs, radius=1)
        main = [p for p in pts[:len(pts) - (2 if ang is not None else 0)] if p is not None]
        if len(main) < 3:
            log('wrist: too few 3-D points')
            return None
        A = np.array(main)
        ztop = float(np.median(np.sort(A[:, 2])[len(A) // 2:]))
        sel = A[A[:, 2] > ztop - 0.007]
        xy = sel[:, :2].mean(axis=0)
        yaw = None
        if ang is not None and pts[-1] is not None and pts[-2] is not None:
            dv = pts[-2] - pts[-1]
            if np.linalg.norm(dv[:2]) > 0.004 and abs(dv[2]) < 0.01:
                yaw = math.atan2(dv[1], dv[0])
        log('wrist cube: area=%d xy=%s top=%.3f yaw=%s' % (c['area'], np.round(xy, 4).tolist(), ztop,
                                                           None if yaw is None else round(math.degrees(yaw), 1)))
        return {'xy': xy, 'top': ztop, 'yaw': yaw}

    def target_highlight(self, side):
        t = self.targets[side]
        img = self.r.rgb.get('front')
        if img is None:
            return False, 0
        pk = P.count_in_box(P.pink_mask(img), t['box'], pad=3)
        rr = P.count_in_box(P.ringred_mask(img), t['box'], pad=3)
        on = pk >= max(4, 2 * t.get('base_pink', 0) + 3)
        if self.color != 'red':
            on = on or rr >= max(6, 2 * t.get('base_red', 0) + 4)
        return on, pk

    def relocalize_target(self, side):
        t = self.targets[side]
        try:
            img = self.r.rgb['front']
            nm = P.dilate(P.neutral_mask(img), 1)
            x0, x1, y0, y1 = t['box']
            pad = 8
            win = np.zeros_like(nm)
            win[max(0, y0 - pad):y1 + pad + 1, max(0, x0 - pad):x1 + pad + 1] = True
            comps = P.components(nm & win, 10)
            best = None
            for c in comps:
                dd = math.hypot(c['cx'] - t['uv'][0], c['cy'] - t['uv'][1])
                w, h = c['x1'] - c['x0'] + 1, c['y1'] - c['y0'] + 1
                if dd < 9 and 0.6 * t['area'] < c['area'] < 1.4 * t['area'] and w < (x1 - x0 + 8) and h < (y1 - y0 + 8):
                    if best is None or dd < best[0]:
                        best = (dd, c)
            if best is None:
                return False
            c = best[1]
            pts = [p for p in self.world('front', [(c['cx'], c['cy'])] + P.sample_pixels(c, 4, 0.8)) if p is not None]
            if len(pts) < 2:
                return False
            A = np.array(pts)
            raw0 = self.world('front', [(c['cx'], c['cy'])], radius=1)[0]
            cxy = raw0[:2] if raw0 is not None else np.median(A[:, :2], axis=0)
            p = np.array([cxy[0], cxy[1], np.median(A[:, 2])])
            if abs(p[2] - self.table_z) > 0.013 or np.linalg.norm(p[:2] - t['p'][:2]) > 0.04:
                return False
            shift = float(np.linalg.norm(p[:2] - t['p'][:2]))
            if shift > 0.003:
                log('target %s relocalized, shift %.3f' % (side, shift))
            t['p'] = p
            t['uv'] = (c['cx'], c['cy'])
            t['box'] = (c['x0'], c['x1'], c['y0'], c['y1'])
            return True
        except Exception as e:
            log('relocalize error', e)
            return False

    # ------------------------------------------------------------ phases
    def pick(self, redetect=False):
        for attempt in range(4):
            self.check()
            if attempt > 0 or redetect:
                self.r.observe()
                cube = self.detect_cube_front()
                if cube is not None:
                    self.cube = cube
            c = self.cube
            log('PICK attempt %d cube xy=%s top=%.3f' % (attempt, np.round(c['xy'], 3).tolist(), c['top']))
            hz = max(c['top'] + HOVER_ABOVE_TOP, self.table_z + 0.15)
            self.move_to([c['xy'][0], c['xy'][1], max(hz, self.pos()[2])], -1.0, V_FREE, 0.02, 30, 0.2, 'up')
            self.move_to([c['xy'][0], c['xy'][1], hz], -1.0, V_FREE, 0.006, 60, name='hover')
            xy, top = np.array(c['xy']), c['top']
            aligned, refined = False, False
            for rnd in range(3):
                z = hz if rnd == 0 else top + HOVER2_ABOVE_TOP
                self.move_to([xy[0], xy[1], z], -1.0, V_FREE, 0.006, 60, 0.04, 'hover%d' % rnd)
                w = self.wrist_view_cube(xy, z)
                lim_xy, lim_z = (0.06, 0.03) if not refined else (0.04, 0.02)
                if w is None or np.linalg.norm(w['xy'] - xy) > lim_xy or abs(w['top'] - top) > lim_z:
                    continue
                xy, top, refined = w['xy'], w['top'], True
                if w['yaw'] is not None:
                    d = geom.wrap_quarter(w['yaw'] - geom.hand_yaw(self.Rcur()))
                    log('yaw: cube %.1f hand %.1f -> rotate %.1f deg' % (
                        math.degrees(w['yaw']), math.degrees(geom.hand_yaw(self.Rcur())), math.degrees(d)))
                    if abs(d) > math.radians(5) and self.rot_mode != 'off':
                        self.R_ref = geom.rotz(d) @ self.R_ref
                        aligned = False
                        continue
                    aligned = True
                if rnd >= 1 and (aligned or w['yaw'] is None):
                    break
            h2 = top + HOVER2_ABOVE_TOP
            h = max(0.015, top - self.table_z)
            gz = max(self.table_z + 0.010, top - 0.5 * h)
            self.move_to([xy[0], xy[1], h2], -1.0, V_FREE, 0.004, 20, 0.03, 'align')
            ok = self.move_to([xy[0], xy[1], gz], -1.0, V_FINE, 0.004, 40, 0.05, 'descend')
            self.hold(2, -1.0)  # settle before closing
            if np.linalg.norm(self.pos() - np.array([xy[0], xy[1], gz])) > 0.006:
                ok = self.move_to([xy[0], xy[1], gz], -1.0, V_FINE, 0.004, 12, 0.05, 'descend-fix')
            log('descend ok=%s pos=%s gz=%.3f' % (ok, np.round(self.pos(), 4).tolist(), gz))
            prev_op = self.opening()
            for _ in range(4):
                self.hold(5, 1.0)
                op = self.opening()
                if abs(op - prev_op) < 0.0015:
                    break
                prev_op = op
            op = self.opening()
            log('closed: opening=%.4f' % op)
            self.z_g = float(self.pos()[2])
            self.cube_h = h
            if op < 0.006:
                log('grasp missed (fingers closed)')
                self.hold(5, -1.0)
                self.move_to(self.pos() + [0, 0, 0.08], -1.0, V_FREE, 0.01, 20, 0.1, 'retreat')
                continue
            self.move_to([xy[0], xy[1], self.z_g + CARRY_LIFT], 1.0, V_CARRY, 0.01, 30, 0.08, 'lift')
            op = self.opening()
            log('lifted: opening=%.4f z=%.3f' % (op, self.pos()[2]))
            self.save_img('lifted')
            if op < 0.006:
                log('lost cube during lift')
                self.hold(4, -1.0)
                continue
            self.pick_xy = np.array(xy)
            return True
        return False

    def transit(self, dest_xy, zc, other_xy=None):
        p = self.pos()
        self.move_to([p[0], p[1], zc], 1.0, V_CARRY, 0.012, 20, 0.08, 'ascend')
        waypoints = []
        if other_xy is not None:
            a, b = self.pos()[:2], np.asarray(dest_xy)
            ab = b - a
            L2 = float(ab @ ab)
            s = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, float((other_xy - a) @ ab) / L2))
            cpt = a + s * ab
            dd = float(np.linalg.norm(cpt - other_xy))
            if dd < AVOID_R and 0.02 < s < 0.98:
                v = cpt - other_xy
                if np.linalg.norm(v) < 1e-4:
                    v = np.array([-ab[1], ab[0]])
                v = v / np.linalg.norm(v)
                waypoints.append(other_xy + v * (AVOID_R + 0.02))
                log('transit waypoint around other target: %s' % np.round(waypoints[-1], 3).tolist())
        for wp in waypoints:
            self.move_to([wp[0], wp[1], zc], 1.0, V_CARRY, 0.02, 40, 0.1, 'waypoint')
        return self.move_to([dest_xy[0], dest_xy[1], zc], 1.0, V_CARRY, 0.006, 70, 0.05, 'transit')

    def visit(self, i, side, other_side, first):
        t = self.targets[side]
        zc = self.z_g + CARRY_LIFT
        self.r.observe()
        self.relocalize_target(side)
        p = t['p']
        other = self.targets[other_side]['p'][:2] if first else None
        self.transit(p[:2], zc, other)
        if self.opening() < 0.006:
            log('visit %d: cube missing after transit' % i)
            return False
        z_contact = p[2] + VISIT_GAP + (self.z_g - self.table_z)
        self.move_to([p[0], p[1], z_contact + 0.02], 1.0, V_CARRY, 0.006, 20, 0.05, 'pre-descend')
        self.move_to([p[0], p[1], z_contact], 1.0, V_FINE, 0.004, 25, 0.05, 'contact')
        held, seen = 0, False
        while held < HOLD_MAX:
            self.hold(2, 1.0)
            held += 2
            on, pk = self.target_highlight(side)
            if on and not seen:
                log('visit %d %s: highlight at hold %d (pink=%d)' % (i, side, held, pk))
                self.save_img('hl%d' % i)
            seen = seen or on
            if seen and held >= HOLD_MIN:
                break
        if not seen:
            log('visit %d %s: no highlight seen; pressing %.3f lower' % (i, side, VISIT_PRESS))
            self.save_img('nohl%d' % i)
            self.move_to([p[0], p[1], z_contact - VISIT_PRESS], 1.0, V_FINE, 0.003, 10, 0.05, 'press')
            for _ in range(4):
                self.hold(2, 1.0)
                on, pk = self.target_highlight(side)
                if on:
                    log('visit %d %s: highlight after press (pink=%d)' % (i, side, pk))
                    seen = True
                    break
        op = self.opening()
        log('visit %d %s done: seen=%s opening=%.4f pos=%s target=%s' % (
            i, side, seen, op, np.round(self.pos(), 3).tolist(), np.round(p, 3).tolist()))
        return op >= 0.006

    def release(self):
        self.hold(6, -1.0)
        p = self.pos()
        self.move_to([p[0], p[1], self.z_g + CARRY_LIFT], -1.0, V_FREE, 0.012, 25, 0.1, 'release-up')
        log('released, opening=%.4f' % self.opening())

    def press_button(self):
        b = self.button
        if b is None:
            self.r.observe()
            _, buttons = self.detect_flat_objects()
            if not buttons:
                log('no button found')
                return
            b = self.button = buttons[0]
        bp = b['p']
        log('BUTTON at %s' % np.round(bp, 3).tolist())
        p = self.pos()
        self.move_to([p[0], p[1], max(p[2], bp[2] + 0.10)], 1.0, V_FREE, 0.02, 15, 0.2, 'b-up')
        self.move_to([bp[0], bp[1], bp[2] + 0.10], 1.0, V_FREE, 0.006, 60, 0.05, 'b-hover')
        for attempt in range(2):
            self.check()
            depth = 0.03 if attempt == 0 else 0.05
            z0 = self.pos()[2]
            self.move_to([bp[0], bp[1], bp[2] - depth], 1.0, 0.25, 0.003, 30, 0.08, 'b-press')
            zl = self.pos()[2]
            self.hold(3, 1.0)
            log('press %d: lowest z %.3f (button top %.3f) complete=%s ended=%s' % (
                attempt, zl, bp[2], self.r.complete, self.r.ended))
            self.save_img('press%d' % attempt)
            if self.r.ended or self.r.complete:
                return
            self.move_to([bp[0], bp[1], bp[2] + 0.08], 1.0, V_FREE, 0.01, 15, 0.1, 'b-up2')
            if self.r.ended or self.r.complete:
                return
            # pressed to (or below) the cap: accept; a stall well above the cap means we hit something else
            if zl < bp[2] + 0.004:
                return
            self.r.observe()
            _, buttons = self.detect_flat_objects()
            if buttons:
                near = min(buttons, key=lambda q: np.linalg.norm(q['p'][:2] - bp[:2]))
                if np.linalg.norm(near['p'][:2] - bp[:2]) < 0.05:
                    bp = near['p']
            self.move_to([bp[0], bp[1], bp[2] + 0.08], 1.0, V_FREE, 0.006, 30, 0.05, 'b-hover2')

    # ------------------------------------------------------------ main
    def safe(self, fn, tries=2):
        for k in range(tries):
            try:
                return fn()
            except Abort:
                raise
            except Exception:
                log('%s EXCEPTION %s' % (fn.__name__, traceback.format_exc()))
                try:
                    self.r.observe()
                except Exception:
                    pass
        return False

    def run(self):
        r = self.r
        r.observe()
        if not r.instruction:
            r.status()
        else:
            r.status()
        self.color, first, n = parse_instruction(r.instruction)
        second = 'left' if first == 'right' else 'right'
        log('%s instruction=%r -> color=%s first=%s N=%d; steps max=%s remaining=%s' % (
            VERSION, r.instruction, self.color, first, n, r.max_steps, r.remaining))
        log('state0', None if r.state is None else np.round(r.state, 4).tolist(), 'cams', list(r.rgb.keys()),
            'step_id', r.step_id)
        self.R_ref = self.Rcur()
        self.safe(self.scene)
        if self.cube is None or len(self.targets) < 2:
            log('scene incomplete: cube=%s targets=%d' % (self.cube is not None, len(self.targets)))
            if self.cube is None or len(self.targets) < 2:
                return
        if not self.safe(self.pick):
            log('pick failed')
            return
        seq = [first if j % 2 == 0 else second for j in range(2 * n)]
        i, recoveries, errors = 0, 0, 0
        while i < len(seq):
            self.check()
            side = seq[i]
            other = second if side == first else first
            try:
                ok = self.visit(i, side, other, first=(i == 0 or recoveries_pending(self)))
            except Abort:
                raise
            except Exception:
                log('visit %d EXCEPTION %s' % (i, traceback.format_exc()))
                errors += 1
                if errors > 3:
                    break
                self.r.observe()
                ok = self.r.state is not None and self.opening() >= 0.006
                if ok:
                    continue  # retry the same visit
            self.after_pick = False
            if not ok:
                log('cube lost at visit %d' % i)
                recoveries += 1
                if recoveries > 2:
                    return
                self.hold(4, -1.0)
                p = self.pos()
                self.move_to([p[0], p[1], p[2] + 0.08], -1.0, V_FREE, 0.02, 15, 0.2, 'recover-up')
                if not self.safe(lambda: self.pick(redetect=True)):
                    return
                self.after_pick = True
                continue  # repeat this visit
            i += 1
        self.safe(self.release)
        self.safe(self.press_button)
        log('DONE complete=%s ended=%s steps_used=%d calls=%d wall=%.0fs eff=%.2f rot_mode=%s' % (
            r.complete, r.ended, r.steps_used, r.n_calls, time.time() - self.t0, self.eff, self.rot_mode))


def recoveries_pending(sw):
    return bool(getattr(sw, 'after_pick', False))


def main(ctx):
    LOG_T0[0] = time.time()
    sw = None
    try:
        sw = Swing(ctx)
        sw.run()
    except Abort as e:
        log('abort:', e)
    except Exception:
        log('EXCEPTION', traceback.format_exc())
        # last resort: if the scene is known and the cube might be held, nothing safe to do but stop.
    finally:
        try:
            if sw is not None:
                log('final: complete=%s ended=%s steps=%d wall=%.0f' % (
                    sw.r.complete, sw.r.ended, sw.r.steps_used, time.time() - sw.t0))
        except Exception:
            pass
