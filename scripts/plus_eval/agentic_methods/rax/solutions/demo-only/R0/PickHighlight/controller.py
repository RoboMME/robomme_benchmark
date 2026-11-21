"""PickHighlight controller (RAX, demo-only offline-developed, version v1).

Task: "first press the button, then pick up all highlighted cubes one by one, finally press
the button to stop".  Highlights are white discs drawn under some cubes for a short time
(demos: execution steps ~10 to ~33-54); cubes are colour-varied, so geometry comes from
depth via robomme_pixel_to_position.

Pipeline / transition table
  S0 observe            -> S1
  S1 wait_highlight     (zero rows, watch white-pixel count; jump up/down)   -> S2 | timeout -> S2 (no hl)
  S2 analyze            (front back-projection at the frozen step)           -> S3
  S3 press_button #1                                                         -> S4
  S4 for each highlighted cube: hover -> wrist refine -> align yaw -> descend -> close ->
        verify fingers -> lift -> verify -> lower -> open -> rise
        (grasp miss -> retry up to 3 attempts, re-refining; then skip)        -> S5
  S5 press_button #2 (repeat once deeper if the episode has not ended)       -> end
Every step is wrapped: exceptions are logged and the pipeline continues to the final press.
"""
import math
import sys
import time
import traceback
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import mcp_io as io_
from mcp_io import log
from robot import Robot, EpisodeOver, rotz, rotvec_to_mat, mat_to_rotvec
import scene
import wrist

VERSION = 'pickhighlight-v2'

# ---------------------------------------------------------------- tunables
WAIT_MAX_STEPS = 90          # zero rows while waiting for highlights (demos: appear at step 10)
HL_JUMP_PX = 60              # white-pixel increase that signals highlights
HOVER_DZ = 0.10              # m above cube top for the wrist look
LIFT_DZ = 0.13               # m cube bottom lifted above table
PRESS_DEPTH = 0.015          # m below the cap top for the press
PRESS_APPROACH_DZ = 0.07
FAST = 0.022                 # m/step free-space speed
SLOW = 0.010                 # m/step near objects
WALL_LIMIT_S = 1050          # stop picking and do the final press after this (controller limit 1200)
FINAL_PRESS_STEPS = 70       # steps reserved for the final press


class Ctl:
    def __init__(self, ctx):
        self.r = Robot(ctx)
        self.table_z = None
        self.size = scene.DEFAULT_SIZE
        self.button = None
        self.cubes = []
        self.targets = []
        self.R0 = None
        self.close_axis_world = None
        self.home = None

    # ------------------------------------------------------------ helpers
    def time_left(self):
        return WALL_LIMIT_S - io_.elapsed()

    def steps_left(self):
        return self.r.remaining if self.r.remaining is not None else 10 ** 6

    def safe_z(self):
        return self.table_z + self.size + HOVER_DZ

    def move(self, xyz, grip, speed=FAST, R=None, tol=0.004, label=''):
        R = self.R0 if R is None else R
        ok, why = self.r.goto(xyz, R_target=R, grip=grip, vmax=speed, tol=tol,
                              min_z=self.table_z + 0.006 if self.table_z is not None else None)
        log('move %s -> (%.3f,%.3f,%.3f) ok=%s %s pos=%s' % (label, xyz[0], xyz[1], xyz[2], ok, why,
            np.round(self.r.pos, 3).tolist() if self.r.state else None))
        return ok

    # ------------------------------------------------------------ S0/S1
    def wait_for_highlight(self):
        r = self.r
        r.observe()
        r.status()
        log('camera assignment:', r.cam_how, r.order_hint)
        try:
            r.identify_cameras()
        except EpisodeOver:
            raise
        except Exception as e:
            log('identify_cameras failed', repr(e)[:200])
        log('start step', r.step_id, 'state', r.state, 'remaining', r.remaining, 'max', r.max_steps)
        if r.state is not None:
            self.home = r.pos.copy()
            self.R0 = r.R
            self.level_orientation()
        f0 = r.cams['front']
        if f0 is None:
            log('no front image at start')
            return None, None
        tref = scene.table_brightness(f0)
        frames = [f0.copy()]
        counts = [scene.white_count(f0, tref)]
        log('white count step0', counts[0])
        hl, base = None, None
        steps = 0
        while steps < WAIT_MAX_STEPS and not r.ended:
            r.send([[0, 0, 0, 0, 0, 0, -1.0]])
            steps += 1
            f = r.cams['front']
            if f is None:
                continue
            c = scene.white_count(f, tref)
            counts.append(c)
            frames.append(f.copy())
            frames = frames[-3:]
            lo = min(counts)
            if c > lo + HL_JUMP_PX and c > counts[-2] + HL_JUMP_PX * 0.5:
                # appeared now; confirm with one more frame
                r.send([[0, 0, 0, 0, 0, 0, -1.0]])
                f2 = r.cams['front']
                c2 = scene.white_count(f2, tref) if f2 is not None else c
                log('highlight appeared at step', r.step_id, 'count', c, '->', c2, 'base', lo)
                hl = f2 if (f2 is not None and c2 >= c * 0.9) else f
                base = frames[-2] if len(frames) >= 2 else None
                break
            if counts[0] > c + HL_JUMP_PX and counts[0] > lo + HL_JUMP_PX and c == lo:
                # highlights were visible at the start and vanished
                log('highlight vanished at step', r.step_id, 'count0', counts[0], 'now', c)
                hl, base = f0, f
                break
        if hl is None:
            log('no highlight event; counts', counts[:5], '...', counts[-5:])
            hl, base = frames[-1], None
        return hl, base

    def level_orientation(self):
        """If the gripper approach axis is tilted a little, target a vertical one (keeps yaw)."""
        R = self.R0
        z = R[:, 2]
        if z[2] > -0.7:
            log('approach axis not near -z (', np.round(z, 3), '); keeping initial orientation')
            return
        tgt = np.array([0, 0, -1.0])
        v = np.cross(z, tgt)
        sn = np.linalg.norm(v)
        if sn < math.sin(math.radians(3)):
            return
        ang = math.atan2(sn, float(np.dot(z, tgt)))
        self.R0 = rotvec_to_mat(v / sn * ang) @ R
        log('levelling orientation by %.1f deg' % math.degrees(ang))

    # ------------------------------------------------------------ S2
    def analyze(self, hl, base):
        r = self.r
        r.observe()   # authoritative step id for back-projection
        tz, _ = scene.estimate_table(r, hl.shape)
        if tz is None:
            log('table estimate failed')
            return
        self.table_z = tz
        log('table z %.4f' % tz)
        tref = scene.table_brightness(base if base is not None else hl)
        btn = scene.find_button(base if base is not None else hl, tref)
        btn_mask = None
        if btn is not None:
            btn_mask = np.zeros(hl.shape[:2], dtype=bool)
            x0, y0, x1, y1 = btn['bbox']
            btn_mask[max(0, y0 - 2):y1 + 3, max(0, x0 - 2):x1 + 3] = True
        if base is None and btn is not None:
            # without a baseline the button plate must not count as highlight
            pass
        res = scene.analyze(r, hl, base, tz, btn_mask=btn_mask)
        self.size = res.get('size', scene.DEFAULT_SIZE)
        self.cubes = res.get('cubes', [])
        b = scene.locate_button(r, btn, tz)
        if b is not None:
            self.button = b
            log('button xy=(%.3f,%.3f) top=%.3f plate=%.3f' % (b['xy'][0], b['xy'][1], b['top_z'], b['plate_z']))
        hlc = res.get('highlighted', [])
        if self.button is not None:
            hlc = [c for c in hlc if np.linalg.norm(c['xy'] - self.button['xy']) > 0.6 * self.size + 0.01]
        self.targets = hlc
        log('highlighted cubes:', len(hlc), 'of', len(self.cubes), 'components', res.get('comps'))
        # gripper closing axis from the wrist camera (image horizontal)
        try:
            self.close_axis_world = wrist.closing_axis(r, tz)
        except Exception as e:
            log('closing axis failed', repr(e)[:200])

    # ------------------------------------------------------------ S3/S5
    def press_button(self, depth=PRESS_DEPTH, label='press'):
        if self.button is None:
            log('no button located; cannot press')
            return False
        b = self.button
        x, y = float(b['xy'][0]), float(b['xy'][1])
        top = b['top_z']
        z_app = top + PRESS_APPROACH_DZ
        cur = self.r.pos
        if cur[2] < z_app - 0.01:
            self.move([cur[0], cur[1], z_app], grip=1.0, label=label + ':up')
        self.move([x, y, z_app], grip=1.0, label=label + ':above')
        self.r.hold(4, grip=1.0)
        self.r.goto([x, y, top - depth], R_target=self.R0, grip=1.0, vmax=0.008, tol=0.003, max_calls=12)
        log('%s bottom pos=%s' % (label, np.round(self.r.pos, 4).tolist()))
        self.r.hold(3, grip=1.0)
        self.move([x, y, z_app], grip=1.0, speed=0.015, label=label + ':retract')
        return True

    # ------------------------------------------------------------ S4
    def grasp_yaw_rotation(self, yaw_cube, xy):
        """Rotation about world z aligning the closing axis with a cube edge; picks the option
        (two per 90 deg) with more clearance to neighbouring cubes, preferring small turns."""
        if yaw_cube is None:
            return 0.0
        R = self.r.R
        if self.close_axis_world is not None:
            ax = self.close_axis_world
            # express closing axis in the gripper frame once; rotate with current R
            if not hasattr(self, '_close_local'):
                self._close_local = self.R0.T @ np.array([ax[0], ax[1], 0.0])
            a = R @ self._close_local
        else:
            a = R[:, 1]
        psi = math.atan2(a[1], a[0])
        d = ((yaw_cube - psi + math.pi / 4) % (math.pi / 2)) - math.pi / 4
        options = [d, d + math.pi / 2 if d < 0 else d - math.pi / 2]
        best = None
        for o in options:
            ang = psi + o
            fdir = np.array([math.cos(ang), math.sin(ang)])
            clr = 1.0
            for c in self.cubes:
                if np.linalg.norm(c['xy'] - xy) < 1e-3:
                    continue
                for sgn in (-1, 1):
                    fp = xy + sgn * fdir * (self.size / 2 + 0.02)
                    clr = min(clr, float(np.linalg.norm(c['xy'] - fp)) - 0.7 * self.size)
            score = clr - 0.02 * abs(o)
            if best is None or score > best[0]:
                best = (score, o, clr)
        log('yaw: cube %.1f grip %.1f -> rotate %.1f deg (clearance %.3f)' % (
            math.degrees(yaw_cube), math.degrees(psi), math.degrees(best[1]), best[2]))
        return best[1]

    def pick(self, cube):
        r = self.r
        s = self.size
        tz = self.table_z
        xy = np.array(cube['xy'], dtype=float)
        yaw = cube.get('yaw')
        top = cube.get('top_z', tz + s)
        for attempt in range(3):
            if r.ended or self.time_left() < 60 or self.steps_left() < FINAL_PRESS_STEPS + 60:
                return False
            hover = [xy[0], xy[1], tz + s + HOVER_DZ]
            cur = r.pos
            if cur[2] < tz + s + 0.05:
                self.move([cur[0], cur[1], tz + s + HOVER_DZ], grip=-1.0, label='rise')
            self.move(hover, grip=-1.0, label='hover')
            # wrist refinement
            try:
                ref = wrist.refine(r, xy, top, s, cube.get('color'), tz)
            except Exception as e:
                log('wrist refine error', repr(e)[:200]); ref = None
            if ref is not None:
                xy = ref['xy']; top = ref.get('top_z', top)
                if ref.get('yaw') is not None:
                    yaw = ref['yaw']
                hover = [xy[0], xy[1], tz + s + HOVER_DZ]
                self.move(hover, grip=-1.0, tol=0.003, label='hover-refined')
            # yaw alignment
            dyaw = self.grasp_yaw_rotation(yaw, xy)
            Rg = rotz(dyaw) @ self.R0 if abs(dyaw) > math.radians(4) else self.R0
            self.move(hover, grip=-1.0, R=Rg, tol=0.004, label='align')
            # descend: pre-grasp then grasp height
            gz = tz + max(0.5 * s, 0.012)
            self.move([xy[0], xy[1], top + 0.03], grip=-1.0, R=Rg, speed=SLOW, label='pregrasp')
            self.r.goto([xy[0], xy[1], gz], R_target=Rg, grip=-1.0, vmax=0.008, tol=0.003, max_calls=15,
                        min_z=tz + 0.008)
            log('grasp height pos=%s' % np.round(r.pos, 4).tolist())
            r.hold(12, grip=1.0)
            fs = r.finger_sum
            log('fingers after close', fs)
            held = (fs is None) or (fs > 0.3 * s)
            if not held:
                log('grasp missed (attempt %d)' % attempt)
                r.hold(6, grip=-1.0)
                self.move([xy[0], xy[1], tz + s + HOVER_DZ], grip=-1.0, R=Rg, label='retreat')
                xy = xy  # refine again on next attempt
                continue
            # lift
            self.move([xy[0], xy[1], tz + 0.5 * s + LIFT_DZ], grip=1.0, R=Rg, speed=0.015, label='lift')
            r.hold(4, grip=1.0)
            fs2 = r.finger_sum
            log('fingers after lift', fs2)
            if fs2 is not None and fs2 < 0.3 * s:
                log('cube dropped during lift (attempt %d)' % attempt)
                r.hold(4, grip=-1.0)
                continue
            # place back where it was
            self.move([xy[0], xy[1], tz + 0.5 * s + 0.012], grip=1.0, R=Rg, speed=0.012, label='lower')
            r.hold(8, grip=-1.0)
            self.move([xy[0], xy[1], tz + s + 0.06], grip=-1.0, R=Rg, speed=0.015, label='release-rise')
            cube['picked'] = True
            return True
        return False

    # ------------------------------------------------------------ main
    def run(self):
        r = self.r
        hl, base = None, None
        try:
            hl, base = self.wait_for_highlight()
        except EpisodeOver:
            return
        except Exception:
            log('wait error', traceback.format_exc()[-800:])
        try:
            if hl is not None:
                self.analyze(hl, base)
        except EpisodeOver:
            return
        except Exception:
            log('analyze error', traceback.format_exc()[-800:])
        if self.table_z is None and r.state is not None:
            self.table_z = float(r.pos[2]) - 0.3
            log('table unknown; guessing', self.table_z)
        try:
            self.press_button(label='press1')
        except EpisodeOver:
            return
        except Exception:
            log('press1 error', traceback.format_exc()[-800:])
        if r.ended:
            return
        # order targets greedily from the button
        todo = list(self.targets)
        cur = self.button['xy'] if self.button is not None else (r.pos[:2] if r.state else np.zeros(2))
        order = []
        while todo:
            j = int(np.argmin([np.linalg.norm(c['xy'] - cur) for c in todo]))
            order.append(todo.pop(j)); cur = order[-1]['xy']
        for i, c in enumerate(order):
            if r.ended or self.time_left() < 60 or self.steps_left() < FINAL_PRESS_STEPS + 60:
                log('stopping picks: time/steps', self.time_left(), self.steps_left())
                break
            log('=== pick %d/%d at (%.3f,%.3f)' % (i + 1, len(order), c['xy'][0], c['xy'][1]))
            try:
                ok = self.pick(c)
                log('pick result', ok)
            except EpisodeOver:
                return
            except Exception:
                log('pick error', traceback.format_exc()[-800:])
                try:
                    r.hold(4, grip=-1.0)
                    p = r.pos
                    self.move([p[0], p[1], self.safe_z()], grip=-1.0, label='recover-rise')
                except EpisodeOver:
                    return
                except Exception:
                    pass
        # final press
        for k, depth in enumerate((PRESS_DEPTH, PRESS_DEPTH + 0.01)):
            if r.ended:
                break
            try:
                self.press_button(depth=depth, label='press_final%d' % k)
                r.hold(4, grip=1.0)
            except EpisodeOver:
                break
            except Exception:
                log('final press error', traceback.format_exc()[-800:])
        log('done; ended=%s complete=%s steps_sent=%d p2p_calls=%d' % (r.ended, r.complete, r.steps_sent, r.p2p_calls))


def main(ctx):
    log('controller', VERSION, 'start')
    try:
        Ctl(ctx).run()
    except EpisodeOver:
        log('episode over')
    except Exception:
        log('fatal', traceback.format_exc()[-1500:])
