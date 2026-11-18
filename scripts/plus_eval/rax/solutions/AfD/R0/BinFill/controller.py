"""BinFill controller: put the instructed numbers of coloured cubes into the bin, then press the button.

Observation-driven: front camera finds cubes/bin/button (depth + calibration or the
pixel_to_position tool), the wrist camera refines each target from above, motion is
closed-loop on the measured TCP. Deposits are counted in memory (bin contents are invisible).
"""
import math
import os
import re
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import vision as V  # noqa: E402
import scene as S  # noqa: E402
from geom import wrap_half_quarter, mat_to_rotvec, wrap_pi  # noqa: E402
from mcpio import log, elapsed  # noqa: E402
from motion import Motion  # noqa: E402
from robot import Robot, EpisodeOver  # noqa: E402

NUM = {'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
       'eight': 8, 'nine': 9, 'ten': 10, 'single': 1, 'couple': 2}
COLOR_WORDS = {'red': 'red', 'orange': 'orange', 'yellow': 'yellow', 'green': 'green', 'cyan': 'cyan',
               'teal': 'cyan', 'blue': 'blue', 'purple': 'purple', 'violet': 'purple', 'magenta': 'magenta',
               'pink': 'magenta', 'gold': 'yellow'}

WALL_LIMIT = 1100.0      # controller wall clock budget (s); hard limit 1200
BUTTON_RESERVE_STEPS = 90
CYCLE_STEPS = 140
V_EMPTY = 0.04          # transit speed limit (m per step) with an empty gripper
V_LOADED = 0.035        # transit speed limit while carrying a cube


def parse_goal(text):
    t = text.lower().replace(',', ' , ')
    goal = {}
    for m in re.finditer(r'\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|single)\s+([a-z]+)\s+(cube|cubes|block|blocks)\b', t):
        n = int(m.group(1)) if m.group(1).isdigit() else NUM.get(m.group(1), 0)
        c = COLOR_WORDS.get(m.group(2))
        if c and n > 0:
            goal[c] = goal.get(c, 0) + n
    if not goal:
        # "three cubes of red" style
        for m in re.finditer(r'\b(\d+|one|two|three|four|five|six)\s+(?:cubes?|blocks?)\s+(?:of\s+)?([a-z]+)', t):
            n = int(m.group(1)) if m.group(1).isdigit() else NUM.get(m.group(1), 0)
            c = COLOR_WORDS.get(m.group(2))
            if c and n > 0:
                goal[c] = goal.get(c, 0) + n
    return goal


class BinFill:
    def __init__(self, ctx):
        self.rb = Robot(ctx)
        self.mo = Motion(self.rb)
        self.goal = {}
        self.placed = {}
        self.tz = None
        self.bin = None
        self.button = None
        self.home = None
        self.failed_spots = []   # (xy, color) of cubes that failed repeatedly
        self.yaw_sign = 1.0      # image-angle -> world-yaw sign (auto-checked)
        self.yaw_sign_checked = False
        self.cube_h = 0.04
        self.t_start = time.time()
        self.cube_side = 0.04      # running estimate of the cube edge length
        self.held_color = None     # colour of the cube currently held (None if empty)
        self.open_width = 0.08     # finger opening used for grasp approach
        self.open_cmd = -1.0       # gripper command giving that opening
        self.grip_map = None       # (g0, w0) calibration point if continuous
        self.grasp_conflicts = 0

    # ------------------------------------------------------------------ helpers
    def remaining_goal(self):
        return {c: n - self.placed.get(c, 0) for c, n in self.goal.items() if n - self.placed.get(c, 0) > 0}

    def time_left(self):
        return WALL_LIMIT - (time.time() - self.t_start)

    def front_obs(self):
        o = self.rb.observe()
        if 'front' not in o.rgb:
            raise RuntimeError('no front image')
        return o

    def transit_z(self):
        z = (self.tz if self.tz is not None else 0.0) + 0.17
        if self.bin is not None:
            z = max(z, self.bin['rim_z'] + 0.13)
        return z

    # ------------------------------------------------------------------ setup
    def setup(self):
        self.rb.status()
        o = self.front_obs()
        log('first obs', o.summary(), 'keys', o.raw_keys[:40])
        log('instruction:', repr(o.instruction))
        self.goal = parse_goal(o.instruction or '')
        if not self.goal:
            st = self.rb.status()
            for k, v in st.items():
                if isinstance(v, str) and 'cube' in v.lower():
                    self.goal = parse_goal(v)
                    if self.goal:
                        break
        log('goal', self.goal, 'limit', self.rb.limit)
        self.rb.validate_dense(o)
        log('dense ok', self.rb.dense_ok)
        self.mo.init_ref()
        self.home = self.rb.tcp().copy()
        R = self.rb.R()
        log('home', np.round(self.home, 4).tolist(), 'tool z axis', np.round(R[:, 2], 3).tolist(), 'fingers', self.rb.fingers())
        self.perceive_static(o)
        try:
            self.calibrate_gripper()
        except EpisodeOver:
            raise
        except Exception as e:
            log('gripper calibration failed', repr(e))

    def calibrate_gripper(self):
        w_open = self.rb.fingers()
        if w_open < 0.06:
            self.mo.set_gripper(-1.0, 8)
            w_open = self.rb.fingers()
        self.mo.set_gripper(0.0, 9)
        w0 = self.rb.fingers()
        self.mo.set_gripper(-1.0, 9)
        w1 = self.rb.fingers()
        log('gripper calibration: open %.4f  g=0 -> %.4f  reopen %.4f' % (w_open, w0, w1))
        if 0.015 < w0 < w_open - 0.015 and w1 > w_open - 0.01:
            self.grip_map = (w_open, w0)
            self.set_open_width(0.04)
        else:
            log('gripper looks binary; using full opening')

    def set_open_width(self, side, tight=False):
        if self.grip_map is None:
            return
        w_open, w0 = self.grip_map
        W = float(np.clip(side + (0.012 if tight else 0.022), 0.05 if tight else 0.056, w_open))
        g = -1.0 + (W - w_open) / (w0 - w_open)
        self.open_cmd = float(np.clip(g, -1.0, -0.05))
        self.open_width = W
        log('approach opening %.4f m via gripper cmd %.3f' % (W, self.open_cmd))

    def perceive_static(self, o):
        try:
            tz = S.table_z(self.rb, o)
            if tz is not None:
                self.tz = tz if self.tz is None else self.tz
        except Exception as e:
            log('table_z failed', repr(e))
        log('table z', self.tz)
        if self.bin is None:
            try:
                self.bin = S.detect_bin(self.rb, o, self.tz)
                log('bin (front)', None if self.bin is None else {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v) for k, v in self.bin.items()})
            except Exception as e:
                log('bin detect failed', repr(e), traceback.format_exc()[-600:])
        if self.button is None:
            try:
                self.button = S.detect_button(self.rb, o, self.tz, bin_info=self.bin)
                log('button (front)', None if self.button is None else np.round(self.button['pos'], 4).tolist())
            except Exception as e:
                log('button detect failed', repr(e), traceback.format_exc()[-600:])

    # ------------------------------------------------------------------ wrist perception
    def wrist_blobs(self, o, mask, min_area=30, near=0.075):
        # drop pixels very close to the wrist camera (fingers / held object)
        d = o.depth.get('wrist')
        if d is not None and d.shape == mask.shape:
            mask = mask & ((d <= 0) | (d > near))
        return V.blobs(mask, min_area)

    def wrist_cube(self, color, expect):
        """Observe and locate the target cube in the wrist image near the expected world position."""
        o = self.rb.observe()
        if 'wrist' not in o.rgb:
            return None
        hsv = V.to_hsv(o.rgb['wrist'])
        m = V.color_mask(hsv, color)
        bl = self.wrist_blobs(o, m, 40)[:8]
        if not bl:
            log('wrist: no %s blob' % color)
            return None
        uvs = []
        for b in bl:
            uvs.append(V.interior_pixel(b, 1)[0])
        pts = self.rb.points('wrist', uvs, o)
        best = None
        for b, p in zip(bl, pts):
            if p is None:
                continue
            dxy = float(np.linalg.norm(p[:2] - np.asarray(expect)[:2]))
            if best is None or dxy < best[0]:
                best = (dxy, b, p)
        if best is None or best[0] > 0.07:
            log('wrist: no %s blob near expected' % color, None if best is None else round(best[0], 3))
            return None
        dxy, b, p = best
        b, p = self.split_if_merged(o, m, b, p, expect)
        touches = b['umin'] <= 0 or b['vmin'] <= 0 or b['umax'] >= o.rgb['wrist'].shape[1] - 1 or b['vmax'] >= o.rgb['wrist'].shape[0] - 1
        # top-face pixels: use dense wrist geometry if available
        xs, ys = b['xs'], b['ys']
        P = self.rb.dense('wrist', o)
        top = None
        wyaw = None
        if P is not None:
            Q = P[ys, xs]
            ok = ~np.isnan(Q).any(1)
            if ok.sum() > 10:
                top = float(np.percentile(Q[ok, 2], 90))
                sel = ok & (Q[:, 2] > top - 0.007)
                if sel.sum() > 15:
                    xs, ys = xs[sel], ys[sel]
                    xy = np.median(Q[sel, :2], 0)
                    p = np.array([xy[0], xy[1], top])
                    try:
                        wyaw = V.min_area_angle_world(Q[sel, :2])
                    except Exception:
                        wyaw = None
        if top is None:
            # sparse: sample a few interior pixels
            ip = V.interior_pixel(b, 5)
            qs = [q for q in self.rb.points('wrist', ip, o) if q is not None]
            if qs:
                q = np.median(np.array(qs), 0)
                p = q
            top = float(p[2])
        ang, _ = V.min_area_angle(xs, ys)
        ang = wrap_half_quarter(ang)
        # ground near the cube: ring around the blob
        ground = None
        try:
            mm = np.zeros(m.shape, bool)
            mm[b['ys'], b['xs']] = True
            ring = V.dilate(mm, 6) & ~V.dilate(mm, 3)
            hs = hsv[1]
            ring &= hs < 0.99
            ry, rx = np.nonzero(ring)
            if len(rx):
                idx = np.linspace(0, len(rx) - 1, min(len(rx), 16)).astype(int)
                gp = [q for q in self.rb.points('wrist', [(rx[i], ry[i]) for i in idx], o) if q is not None]
                if gp:
                    gz = np.array([q[2] for q in gp])
                    gz = gz[gz < top - 0.012]
                    if len(gz):
                        ground = float(np.median(gz))
        except Exception as e:
            log('ground est failed', repr(e))
        # neighbouring cubes (any colour) seen by the wrist camera
        nbrs = []
        side = None
        try:
            Pd = self.rb.dense('wrist', o)
            if Pd is not None:
                Q = Pd[ys, xs]
                Q = Q[~np.isnan(Q).any(1)]
                if len(Q) > 15:
                    a = math.radians(V.min_area_angle_world(Q[:, :2]))
                    R2 = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
                    pr = (Q[:, :2] - Q[:, :2].mean(0)) @ R2
                    side = float(np.mean([np.percentile(pr[:, 0], 97) - np.percentile(pr[:, 0], 3),
                                          np.percentile(pr[:, 1], 97) - np.percentile(pr[:, 1], 3)]))
            for cc in S.CUBE_COLORS:
                mk = V.color_mask(hsv, cc)
                for nb in self.wrist_blobs(o, mk, 30)[:10]:
                    if cc == color and abs(nb['u'] - b['u']) < 1 and abs(nb['v'] - b['v']) < 1:
                        continue
                    if Pd is not None:
                        Qn = Pd[nb['ys'], nb['xs']]
                        Qn = Qn[~np.isnan(Qn).any(1)]
                        if len(Qn) < 5:
                            continue
                        tn = float(np.percentile(Qn[:, 2], 90))
                        qn = np.median(Qn[Qn[:, 2] > tn - 0.008][:, :2], 0)
                        if top is not None and tn > top + 0.05:
                            continue   # something high (held/in-air), ignore
                        nbrs.append(np.array([qn[0], qn[1]]))
                    else:
                        nbrs.append(None)
                        nbrs[-1] = (nb['u'], nb['v'])
            if Pd is None and nbrs:
                pts_n = self.rb.points('wrist', nbrs, o)
                nbrs = [q[:2] for q in pts_n if q is not None]
            nbrs = [q for q in nbrs if np.linalg.norm(q - np.asarray(p)[:2]) > 0.012]
            nbrs += list(getattr(self, 'merged_nbrs', []) or [])
        except Exception as e:
            log('wrist neighbours failed', repr(e))
        return dict(pos=np.asarray(p, float), top=top, ang=ang, area=b['area'], touches=touches, ground=ground,
                    nbrs=nbrs, side=side, wyaw=wyaw,
                    u=b['u'], v=b['v'], dxy=dxy)

    def split_if_merged(self, o, m, b, p, expect):
        """If the wrist blob spans several touching same-coloured cubes, keep only the cube-sized
        cell nearest `expect` (grid split of the top face along its rectangle axes).  The other
        cells' centres are stored in self.merged_nbrs for the finger-clearance check."""
        self.merged_nbrs = []
        try:
            P = self.rb.dense('wrist', o)
            if P is None:
                return b, p
            Qa = P[b['ys'], b['xs']]
            ok = ~np.isnan(Qa).any(1)
            if ok.sum() < 20:
                return b, p
            top = float(np.percentile(Qa[ok, 2], 90))
            tsel = ok & (Qa[:, 2] > top - 0.007)
            Qt = Qa[tsel][:, :2]
            if len(Qt) < 15:
                return b, p
            yaw = V.min_area_angle_world(Qt)
            ex = V.extent_world(Qt, yaw)
            ref = max(0.03, min(0.06, self.cube_side))
            if max(ex) < 1.45 * ref:
                return b, p
            na = int(min(4, max(1, round(ex[0] / ref))))
            nb = int(min(4, max(1, round(ex[1] / ref))))
            if na * nb < 2:
                return b, p
            a = math.radians(yaw)
            Rm = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            c0 = Qt.mean(0)
            prt = (Qt - c0) @ Rm
            lo = np.percentile(prt, 3, axis=0)
            hi = np.percentile(prt, 97, axis=0)
            span = np.maximum(hi - lo, 1e-6)
            XY = np.where(ok[:, None], Qa[:, :2], 0.0)
            pra = (XY - c0) @ Rm
            ia = np.clip(((pra[:, 0] - lo[0]) / span[0] * na).astype(int), 0, na - 1)
            ib = np.clip(((pra[:, 1] - lo[1]) / span[1] * nb).astype(int), 0, nb - 1)
            cells = []
            for i in range(na):
                for j in range(nb):
                    sel_t = tsel & (ia == i) & (ib == j)
                    if sel_t.sum() < max(10, 0.4 * len(Qt) / (na * nb)):
                        continue
                    xy = np.median(Qa[sel_t, :2], 0)
                    cells.append((float(np.linalg.norm(xy - np.asarray(expect)[:2])), i, j, xy))
            if len(cells) < 2:
                return b, p
            cells.sort(key=lambda t: t[0])
            d, i, j, xy = cells[0]
            pick = ok & (ia == i) & (ib == j)
            xs, ys = b['xs'][pick], b['ys'][pick]
            if len(xs) < 12:
                return b, p
            nbl = dict(b)
            nbl.update(xs=xs, ys=ys, area=int(len(xs)), u=float(xs.mean()), v=float(ys.mean()),
                       umin=int(xs.min()), umax=int(xs.max()), vmin=int(ys.min()), vmax=int(ys.max()))
            self.merged_nbrs = [np.array(cc[3]) for cc in cells[1:]]
            log('split merged wrist blob (extent %s, grid %dx%d, %d cells); using cell at %s' % (
                np.round(ex, 3).tolist(), na, nb, len(cells), np.round(xy, 3).tolist()))
            return nbl, np.array([xy[0], xy[1], top])
        except Exception as e:
            log('split check failed', repr(e)[:200])
            return b, p

    # ------------------------------------------------------------------ finger clearance
    def closing_dir(self, o=None):
        o = o or self.rb.obs
        try:
            cam = self.rb.camera('wrist', o)
            if cam is not None:
                d = cam.R[:2, 0]
                if np.linalg.norm(d) > 0.3:
                    d = d / np.linalg.norm(d)
                    # snap to the nearer TCP horizontal axis (fingers move along a TCP axis)
                    Rt = self.rb.R()
                    best = None
                    for ax in (Rt[:2, 0], Rt[:2, 1]):
                        if np.linalg.norm(ax) < 0.3:
                            continue
                        ax = ax / np.linalg.norm(ax)
                        sc = abs(float(ax @ d))
                        if best is None or sc > best[0]:
                            best = (sc, ax)
                    if best is not None:
                        return best[1]
                    return d
        except Exception:
            pass
        d = self.rb.R()[:2, 1]
        return d / max(np.linalg.norm(d), 1e-6)

    def conflicts(self, xy, cdir, neighbors):
        cdir = np.asarray(cdir, float)
        n = np.array([-cdir[1], cdir[0]])
        w = self.open_width
        rn = 0.72 * self.cube_side + 0.004   # neighbour half-diagonal plus margin
        amax = w / 2 + 0.008 + rn
        bmax = 0.012 + rn
        k = 0
        margin = 1.0
        for q in neighbors:
            d = np.asarray(q)[:2] - np.asarray(xy)[:2]
            a, b = abs(d @ cdir), abs(d @ n)
            if a < 0.012 and b < 0.012:
                continue
            if a < amax and b < bmax:
                k += 1
            margin = min(margin, max(a - amax, b - bmax))
        return k, margin

    def neighbors_of(self, xy, wrist_nbrs=()):
        out = [np.asarray(q, float) for q in wrist_nbrs]
        for c in getattr(self, 'last_all', []) or []:
            q = c['pos'][:2]
            if np.linalg.norm(q - np.asarray(xy)[:2]) > 0.012 and all(np.linalg.norm(q - r) > 0.02 for r in out):
                out.append(q)
        return out

    # ------------------------------------------------------------------ choose cube
    def choose(self, cubes, all_cubes):
        rem = self.remaining_goal()
        cands = []
        for c in cubes:
            if c['color'] not in rem:
                continue
            xy = c['pos'][:2]
            if self.bin is not None and np.linalg.norm(xy - self.bin['center'][:2]) < self.bin['half'] + 0.045:
                continue
            if self.button is not None and np.linalg.norm(xy - self.button['pos'][:2]) < 0.05:
                continue
            fails = sum(1 for (fxy, fc) in self.failed_spots if fc == c['color'] and np.linalg.norm(fxy - xy) < 0.03)
            if fails >= 2:
                continue
            others = [o for o in all_cubes if o is not c and np.linalg.norm(o['pos'][:2] - xy) > 0.005]
            nn = min([np.linalg.norm(o['pos'][:2] - xy) for o in others], default=1.0)
            dist_bin = np.linalg.norm(xy - self.bin['center'][:2]) if self.bin is not None else 0.0
            dist_tcp = np.linalg.norm(xy - self.rb.tcp()[:2])
            score = -min(nn, 0.12) * 4.0 + 0.6 * dist_bin + 0.3 * dist_tcp + 0.15 * fails
            if c.get('yaw') is not None:
                a = math.radians(c['yaw'])
                nb = [o['pos'][:2] for o in others]
                k0, _ = self.conflicts(xy, (math.cos(a), math.sin(a)), nb)
                k1, _ = self.conflicts(xy, (-math.sin(a), math.cos(a)), nb)
                score += 0.25 * min(k0, k1)
            if c['ext'] > 0.075:   # probably two merged cubes
                score += 0.3
            cands.append((score, c, nn))
        if not cands:
            return None
        cands.sort(key=lambda x: x[0])
        s, c, nn = cands[0]
        log('chosen', c['color'], np.round(c['pos'], 3).tolist(), 'nn %.3f' % nn, 'score %.3f' % s, 'n cands', len(cands))
        return c

    # ------------------------------------------------------------------ pick
    def yaw_limits(self, yaw):
        # keep commanded yaw within +-100 deg of the reference
        while yaw > math.radians(100):
            yaw -= math.pi / 2
        while yaw < -math.radians(100):
            yaw += math.pi / 2
        return yaw

    def pick(self, cube):
        color = cube['color']
        est = cube['pos'].copy()
        hover_dz = 0.13
        tz = self.tz if self.tz is not None else est[2] - 0.04
        zt = self.transit_z()
        # go above the cube at transit height, then to hover
        p = self.rb.tcp()
        if p[2] < zt - 0.03:
            self.mo.move(np.array([p[0], p[1], zt]), tol=0.01, vmax=0.025, label='lift-pre')
        self.mo.grip = self.open_cmd
        self.mo.move(np.array([est[0], est[1], max(est[2] + hover_dz, tz + 0.12)]), tol=0.008, vmax=V_EMPTY, max_steps=80, label='to-hover')
        info = None
        prev_ang = None
        for it in range(4):
            w = self.wrist_cube(color, est)
            if w is None:
                if it == 0:
                    # widen: look from a bit higher
                    q = self.rb.tcp()
                    self.mo.move(np.array([est[0], est[1], q[2] + 0.05]), tol=0.01, vmax=0.025, label='hover-up')
                    continue
                break
            info = w
            new = w['pos']
            shift = float(np.linalg.norm(new[:2] - est[:2]))
            est = np.array([new[0], new[1], w['top']])
            ang = w['ang']
            ang_w = None
            if w.get('wyaw') is not None:
                Rt = self.rb.R()
                tyaw = math.degrees(math.atan2(Rt[1, 0], Rt[0, 0]))
                ang_w = wrap_half_quarter(w['wyaw'] - tyaw)
            log('wrist it%d: pos %s ang %.1f world-err %s shift %.3f area %d ground %s' % (it, np.round(est, 4).tolist(), ang, None if ang_w is None else round(ang_w, 1), shift, w['area'], w['ground']))
            if ang_w is not None:
                # world-frame yaw error (cube top-face yaw minus TCP yaw, mod 90): independent of camera mounting
                if abs(abs(ang_w) - abs(ang)) > 6:
                    log('note: image angle %.1f disagrees with world error %.1f (camera/finger offset?)' % (ang, ang_w))
                Rt = self.rb.R()
                Rg = self.mo.R_goal(self.mo.yaw_off)
                lag = math.degrees(wrap_pi(math.atan2(Rt[1, 0], Rt[0, 0]) - math.atan2(Rg[1, 0], Rg[0, 0])))
                ang = -(ang_w + lag)   # in the servo convention below (new = off - sign*ang) with sign fixed +1
                self.yaw_sign_checked = True
                self.yaw_sign = 1.0
            # yaw sign self-check after first correction
            if prev_ang is not None and not self.yaw_sign_checked and abs(prev_ang) > 8:
                # wrong sign doubles the apparent angle; right sign nulls it
                if abs(wrap_half_quarter(ang - 2 * prev_ang)) + 2.0 < abs(ang):
                    self.yaw_sign = -self.yaw_sign
                    log('yaw sign flipped ->', self.yaw_sign)
                else:
                    log('yaw sign confirmed', self.yaw_sign)
                self.yaw_sign_checked = True
            done_ang = abs(ang) < 5.0
            done_pos = shift < 0.004
            new_yaw = self.mo.yaw_off
            if not done_ang:
                new_yaw = self.yaw_limits(self.mo.yaw_off - self.yaw_sign * math.radians(ang))
                prev_ang = ang
            else:
                prev_ang = None
            # pick the grasp axis (new_yaw or new_yaw+-90) with fewer finger conflicts
            try:
                cd = self.closing_dir()
                nb = self.neighbors_of(est, w.get('nbrs') or ())
                opts = []
                for extra in (0.0, math.pi / 2, -math.pi / 2):
                    y = new_yaw + extra
                    if abs(y) > math.radians(100):
                        continue
                    dy = y - self.mo.yaw_off
                    cdy = np.array([math.cos(dy) * cd[0] - math.sin(dy) * cd[1], math.sin(dy) * cd[0] + math.cos(dy) * cd[1]])
                    k, mg = self.conflicts(est, cdy, nb)
                    opts.append((k, abs(extra), -mg, abs(y), y))
                opts.sort()
                if opts and abs(opts[0][4] - new_yaw) > 0.1:
                    log('grasp axis switched by 90 deg to avoid %s' % (opts,))
                    prev_ang = None  # sign check would be confused by the 90-degree switch
                    done_pos = False
                if opts:
                    new_yaw = opts[0][4]
                    self.grasp_conflicts = opts[0][0]
            except Exception as e:
                log('axis choice failed', repr(e))
            if done_ang and done_pos and it > 0:
                break
            self.mo.yaw_off = new_yaw
            self.mo.move(np.array([est[0], est[1], max(est[2] + hover_dz - 0.02 * it, tz + 0.10)]), tol=0.004,
                         vmax=0.02, max_steps=40, label='refine%d' % it)
        if info is None:
            log('pick: target lost in wrist view')
            return False, None
        top = est[2]
        ground = info['ground'] if info.get('ground') is not None else tz
        h = top - ground
        if not (0.015 < h < 0.07):
            h = 0.04
        self.cube_h = 0.7 * self.cube_h + 0.3 * h
        grasp_z = max(top - min(0.6 * h, 0.026), ground + 0.008)
        log('grasp plan: top %.4f ground %.4f h %.4f grasp_z %.4f' % (top, ground, h, grasp_z))
        # descend
        if info.get('side') and 0.02 < info['side'] < 0.06:
            self.cube_side = 0.7 * self.cube_side + 0.3 * info['side']
        if info.get('side') and 0.02 < info['side'] < 0.06 and self.grip_map is not None:
            self.set_open_width(info['side'])
            self.mo.grip = self.open_cmd
        self.mo.move(np.array([est[0], est[1], top + 0.045]), tol=0.004, vmax=0.02, max_steps=40, label='pre-grasp')
        wf = self.rb.fingers()
        if wf < (info.get('side') or 0.04) + 0.008:
            log('fingers too closed before descent (%.4f); opening fully' % wf)
            self.mo.set_gripper(-1.0, 6)
        self.mo.move(np.array([est[0], est[1], grasp_z]), tol=0.003, vmax=0.01, max_steps=40, chunk=3, label='descend', learn=False)
        zr = self.rb.tcp()[2]
        if zr > grasp_z + 0.012:
            # blocked (finger on a neighbour): retry once with the grasp axis turned 90 deg and a tighter opening
            log('descend blocked at %.4f (goal %.4f); retrying with the other grasp axis' % (zr, grasp_z))
            self.mo.move(np.array([est[0], est[1], top + 0.045]), tol=0.006, vmax=0.015, max_steps=20, label='unblock-up')
            y2 = self.mo.yaw_off + math.pi / 2
            if abs(y2) > math.radians(100):
                y2 = self.mo.yaw_off - math.pi / 2
            self.mo.yaw_off = y2
            side = info.get('side') if info.get('side') and 0.02 < info['side'] < 0.06 else self.cube_side
            if self.grip_map is not None:
                self.set_open_width(side, tight=True)
                self.mo.grip = self.open_cmd
            self.mo.move(np.array([est[0], est[1], top + 0.045]), tol=0.004, vmax=0.015, max_steps=30, label='pre-grasp2')
            self.mo.move(np.array([est[0], est[1], grasp_z]), tol=0.003, vmax=0.01, max_steps=40, chunk=3, label='descend2', learn=False)
            zr = self.rb.tcp()[2]
            if self.grip_map is not None:
                self.set_open_width(side)   # back to the normal opening for later grasps
        if zr > grasp_z + 0.012:
            log('descend blocked at %.4f (goal %.4f); aborting this grasp' % (zr, grasp_z))
            self.mo.move(np.array([est[0], est[1], top + 0.08]), tol=0.01, vmax=0.02, max_steps=20, label='abort-up')
            self.failed_spots.append((est[:2].copy(), color))
            self.failed_spots.append((est[:2].copy(), color))
            return False, info
        self.mo.set_gripper(1.0, 10)
        f = self.rb.fingers()
        log('fingers after close %.4f' % f)
        # lift
        # short vertical lift clear of neighbouring cubes; the transit to the bin rises diagonally
        z_lift = min(self.transit_z(), max(grasp_z + 0.075, top + 0.05))
        self.mo.move(np.array([est[0], est[1], z_lift]), tol=0.01, vmax=0.022, max_steps=40, label='lift')
        f2 = self.rb.fingers()
        holding = f2 > 0.008
        log('fingers after lift %.4f holding=%s' % (f2, holding))
        if not holding:
            self.failed_spots.append((est[:2].copy(), color))
            self.mo.grip = -1.0
            self.mo.hold(4, -1.0)
        return holding, info

    # ------------------------------------------------------------------ place
    def refine_bin_wrist(self):
        """Hover above the bin with an empty gripper and refine its centre from the wrist view."""
        if self.bin is None:
            return
        c = self.bin['center']
        self.mo.move(np.array([c[0], c[1], self.bin['rim_z'] + 0.22]), tol=0.01, vmax=V_EMPTY, max_steps=80, label='bin-view')
        for it in range(2):
            o = self.rb.observe()
            if 'wrist' not in o.rgb:
                return
            h, s, v = V.to_hsv(o.rgb['wrist'])
            black = v < 0.09
            bl = self.wrist_blobs(o, black, 60, near=0.12)
            if not bl:
                log('bin wrist: no black blob')
                return
            best = None
            for b in bl[:5]:
                pts = None
                cam = self.rb.camera('wrist', o)
                if cam is not None:
                    uv = np.stack([b['xs'], b['ys']], 1).astype(float)
                    r = cam.ray_world(uv[:, 0], uv[:, 1])
                    t = (self.bin['rim_z'] - cam.t[2]) / r[:, 2]
                    pts = cam.t[None, :] + t[:, None] * r
                    ctr = pts[:, :2].mean(0)
                    half = float(np.percentile(np.abs(pts[:, :2] - ctr), 95, axis=0).max())
                else:
                    q = self.rb.points('wrist', [(b['u'], b['v'])], o)[0]
                    if q is None:
                        continue
                    ctr = q[:2]
                    half = self.bin['half']
                d = float(np.linalg.norm(ctr - c[:2]))
                if d < 0.08 and (best is None or d < best[0]):
                    best = (d, ctr, half, b)
            if best is None:
                log('bin wrist: no blob near front estimate')
                return
            d, ctr, half, b = best
            log('bin wrist it%d: centre %s (front %s) half %.3f touches=%s' % (it, np.round(ctr, 4).tolist(), np.round(c[:2], 4).tolist(), half, b['umin'] <= 0 or b['vmin'] <= 0))
            self.bin['center'] = np.array([ctr[0], ctr[1], self.bin['rim_z']])
            if 0.02 < half < 0.12:
                self.bin['half'] = half
            c = self.bin['center']
            if d < 0.01:
                break
            self.mo.move(np.array([c[0], c[1], self.bin['rim_z'] + 0.22]), tol=0.006, vmax=0.02, max_steps=40, label='bin-view2')

    def place(self, color):
        c = self.bin['center']
        zt = max(self.transit_z(), self.bin['rim_z'] + 0.12)
        self.mo.move(np.array([c[0], c[1], zt]), tol=0.008, vmax=V_LOADED, max_steps=90, label='to-bin')
        if self.rb.fingers() < 0.006:
            log('cube lost during transit')
            self.held_color = None
            return False
        drop_z = self.bin['rim_z'] + 0.5 * self.cube_h + 0.045
        self.mo.move(np.array([c[0], c[1], drop_z]), tol=0.006, vmax=0.018, max_steps=40, label='bin-lower')
        if self.rb.fingers() < 0.006:
            log('cube lost before release')
            self.held_color = None
            return False
        # release over the bin; retry the open command until the fingers are seen open
        last_err = None
        for k in range(3):
            try:
                self.mo.set_gripper(-1.0, 8)
                last_err = None
                break
            except EpisodeOver:
                self.placed[color] = self.placed.get(color, 0) + 1
                self.held_color = None
                raise
            except Exception as e:
                last_err = e
                log('release error (try %d)' % k, repr(e)[:200])
                try:
                    self.rb.observe()
                    if self.rb.fingers() > 0.05:
                        last_err = None
                        break
                except EpisodeOver:
                    raise
                except Exception:
                    pass
        if last_err is not None:
            raise last_err
        self.held_color = None
        self.placed[color] = self.placed.get(color, 0) + 1
        log('RELEASED %s into bin -> placed %s' % (color, self.placed))
        self.mo.move(np.array([c[0], c[1], zt]), tol=0.015, vmax=V_EMPTY, max_steps=30, label='bin-up')
        return True

    # ------------------------------------------------------------------ button
    def refine_button_wrist(self):
        b = self.button
        hover = b['top'] + 0.17
        self.mo.move(np.array([b['pos'][0], b['pos'][1], hover]), tol=0.008, vmax=V_EMPTY, max_steps=90, label='btn-view')
        for it in range(3):
            o = self.rb.observe()
            if 'wrist' not in o.rgb:
                return
            h, s, v = V.to_hsv(o.rgb['wrist'])
            low = V.erode((s < 0.22) & (v > 0.12), 1)
            bl = self.wrist_blobs(o, low, 40)[:10]
            P = self.rb.dense('wrist', o)
            best = None
            for bb in bl:
                if P is not None:
                    Q = P[bb['ys'], bb['xs']]
                    Q = Q[~np.isnan(Q).any(1)]
                    if len(Q) < 10:
                        continue
                    topz = float(np.percentile(Q[:, 2], 95))
                    ext = float(np.max(np.ptp(Q[:, :2], 0)))
                    capsel = Q[Q[:, 2] > topz - 0.004]
                    xy = np.median(capsel[:, :2], 0)
                else:
                    ip = V.interior_pixel(bb, 5)
                    qs = [q for q in self.rb.points('wrist', ip, o) if q is not None]
                    if not qs:
                        continue
                    q = np.median(np.array(qs), 0)
                    topz, xy, ext = float(q[2]), q[:2], 0.05
                d = float(np.linalg.norm(xy - b['pos'][:2]))
                hz = topz - (self.tz if self.tz is not None else topz - 0.02)
                if d > 0.08 or ext > 0.15 or not (0.004 < hz < 0.09):
                    continue
                sc = d - 0.002 * min(bb['area'], 2000) / 100.0
                if best is None or sc < best[0]:
                    best = (sc, xy, topz, ext, d)
            if best is None:
                log('button wrist: none found')
                return
            sc, xy, topz, ext, d = best
            log('button wrist it%d: xy %s top %.4f ext %.3f shift %.3f' % (it, np.round(xy, 4).tolist(), topz, ext, d))
            self.button['pos'] = np.array([xy[0], xy[1], topz])
            self.button['top'] = topz
            if d < 0.006:
                break
            self.mo.move(np.array([xy[0], xy[1], topz + 0.15 - 0.03 * it]), tol=0.005, vmax=0.02, max_steps=40, label='btn-view%d' % it)

    def press_button(self):
        if self.button is None:
            log('no button known; looking from home')
            self.mo.move(self.home, tol=0.01, vmax=V_EMPTY, max_steps=80, label='home')
            o = self.front_obs()
            self.button = S.detect_button(self.rb, o, self.tz, bin_info=self.bin)
            if self.button is None:
                log('button not found')
                return False
        try:
            self.refine_button_wrist()
        except EpisodeOver:
            raise
        except Exception as e:
            log('button refine failed', repr(e), traceback.format_exc()[-500:])
        b = self.button
        xy = b['pos'][:2]
        top = b['top']
        self.mo.grip = 1.0
        self.mo.move(np.array([xy[0], xy[1], top + 0.04]), tol=0.004, vmax=0.02, max_steps=50, label='btn-pre')
        for depth in (0.012, 0.022, 0.032):
            self.mo.move(np.array([xy[0], xy[1], top - depth]), tol=0.003, vmax=0.008, max_steps=25, chunk=3, label='btn-press', learn=False)
            self.mo.hold(3)
            if self.rb.ended:
                return True
            log('button not ended yet after depth', depth, 'tcp', np.round(self.rb.tcp(), 4).tolist())
        self.mo.move(np.array([xy[0], xy[1], top + 0.05]), tol=0.006, vmax=0.02, max_steps=20, label='btn-up')
        return self.rb.ended

    # ------------------------------------------------------------------ main loop
    def run(self):
        if not getattr(self, 'setup_done', False):
            self.setup()
            self.setup_done = True
        else:
            log('resuming run after error; placed', self.placed)
        if not self.goal:
            log('WARNING: could not parse goal; defaulting to nothing (press button)')
        # wait for a bin detection; refine it with the wrist camera
        tries = 0
        while self.bin is None and tries < 4:
            tries += 1
            self.mo.hold(10)
            self.perceive_static(self.front_obs())
        if self.bin is None:
            log('bin not found; cannot place')
        elif not getattr(self, 'bin_refined', False):
            self.bin_refined = True
            try:
                self.refine_bin_wrist()
            except EpisodeOver:
                raise
            except Exception as e:
                log('bin refine failed', repr(e), traceback.format_exc()[-500:])
        attempts = 0
        idle = 0
        while self.remaining_goal() and self.bin is not None:
            # stopping early with the goal unmet fails anyway, so only the wall clock limits the loop
            if self.time_left() < 90 or self.rb.remaining() <= 0:
                log('budget exhausted: steps left', self.rb.remaining(), 'time left', round(self.time_left()))
                break
            attempts += 1
            if attempts > 40:
                break
            try:
                if self.held_color is not None:
                    # a previous cycle ended while holding a cube: place it before anything else
                    self.rb.observe()
                    if self.rb.fingers() > 0.008:
                        log('still holding %s at cycle start; placing it' % self.held_color)
                        self.mo.grip = 1.0
                        self.place(self.held_color)
                        continue
                    self.held_color = None
                o = self.front_obs()
                if self.button is None:
                    self.perceive_static(o)
                rem = self.remaining_goal()
                tcp = self.rb.tcp()
                excl = [(tcp, 0.05)]
                all_c = S.detect_cubes(self.rb, o, self.tz, exclude=excl)
                self.last_all = all_c
                cubes = [c for c in all_c if c['color'] in rem]
                log('remaining', rem, 'visible targets', [(c['color'], np.round(c['pos'][:2], 3).tolist()) for c in cubes])
                cube = self.choose(cubes, all_c)
                if cube is None and cubes and self.failed_spots and idle >= 2:
                    log('all visible targets previously failed; clearing failed spots and retrying')
                    self.failed_spots = []
                    cube = self.choose(cubes, all_c)
                if cube is None:
                    idle += 1
                    log('no target cube visible; idle', idle)
                    # go to a viewing pose (home) and wait for new cubes
                    if np.linalg.norm(self.rb.tcp() - self.home) > 0.03:
                        self.mo.move(self.home, tol=0.02, vmax=V_EMPTY, max_steps=60, label='home')
                    else:
                        self.mo.hold(15)
                    if idle > 60:
                        break
                    continue
                idle = 0
                ok, info = self.pick(cube)
                if ok:
                    self.held_color = cube['color']
                    self.place(cube['color'])
            except EpisodeOver:
                raise
            except Exception as e:
                log('cycle error', repr(e), traceback.format_exc()[-1200:])
                try:
                    self.rb.observe()
                    if self.rb.fingers() > 0.008 and self.bin is not None and self.held_color is not None:
                        log('still holding after error; trying to place')
                        self.mo.grip = 1.0
                        self.place(self.held_color)
                except EpisodeOver:
                    raise
                except Exception as e2:
                    log('recovery failed', repr(e2))
        log('placing finished: placed', self.placed, 'goal', self.goal, 'steps', self.rb.steps, 'time', round(elapsed()))
        if self.rb.fingers() > 0.008:
            self.mo.set_gripper(-1.0, 6)
        # lift before travelling to the button
        p = self.rb.tcp()
        zt = self.transit_z()
        if p[2] < zt - 0.03:
            self.mo.move(np.array([p[0], p[1], zt]), tol=0.01, vmax=0.025, max_steps=30, label='lift-final')
        for k in range(4):
            if self.rb.ended or self.time_left() < 25:
                break
            try:
                ok = self.press_button()
            except EpisodeOver:
                raise
            except Exception as e:
                log('press attempt %d error' % k, repr(e)[:300])
                ok = False
            if ok or self.rb.ended:
                break
            # re-detect the button from the front camera and try again
            try:
                self.mo.move(self.home, tol=0.02, vmax=V_EMPTY, max_steps=60, label='home')
                o = self.front_obs()
                btn = S.detect_button(self.rb, o, self.tz, bin_info=self.bin)
                if btn is not None:
                    self.button = btn
            except EpisodeOver:
                raise
            except Exception as e:
                log('button re-detect failed', repr(e))
        log('END ended=%s success=%s steps=%s time=%.0f placed=%s' % (self.rb.ended, self.rb.success, self.rb.steps, elapsed(), self.placed))


def main(ctx):
    log('BinFill controller start')
    bf = None
    for run_k in range(3):
        try:
            if bf is None:
                bf = BinFill(ctx)
            bf.run()
            break
        except EpisodeOver:
            log('episode over', 'success', None if bf is None else bf.rb.success)
            break
        except Exception as e:
            log('FATAL (run %d)' % run_k, repr(e), traceback.format_exc()[-2000:])
            if bf is None or bf.rb.ended or bf.time_left() < 120:
                break
            time.sleep(0.5)
    # last resort: if the goal is satisfied but the episode has not ended, try pressing the button
    for k in range(2):
        try:
            if bf is not None and not bf.rb.ended and not bf.remaining_goal() and bf.button is not None \
                    and bf.time_left() > 20:
                log('last-resort button press', k)
                if bf.press_button():
                    break
            else:
                break
        except EpisodeOver:
            break
        except Exception as e2:
            log('last resort failed', repr(e2))
    log('controller exit; placed', None if bf is None else bf.placed)
