"""MoveCube controller (RoboMME-Plus, RAX).

Pipeline: observe step 0 -> read conditioning video -> decide the demonstrated manner (which tool/ball moved)
-> perceive the execution scene in 3-D -> execute the matching strategy:
  * stick tools (hook / ball-stick / spatula / wood stick): grasp handle, push cube onto target with the tool tip,
    closed-loop re-pushes;
  * ramp: grasp yellow ball, release it at the top of the ramp in line with the cube.
Never raises: every failure is logged and falls through to the next option.
"""
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcpio import log, dump_json, strip_big, split_result, find_key, video_frames, load_png, parse_obs  # noqa: E402
from geom import calibrate, Camera, Rz, wrap, R_to_rotvec, rotvec_to_R  # noqa: E402
from robot import Robot, EpisodeOver  # noqa: E402
import seg  # noqa: E402
import perceive  # noqa: E402

WALL_LIMIT = 1080.0      # controller wall budget (hard limit 1200 s)
TEAL_TYPES = ('ball', 'hook', 'spatula')


# ----------------------------------------------------------------------------------------------
def get_step_limit(ctx):
    try:
        r = ctx.call('robomme_status')
        meta, _ = split_result(r)
        dump_json('robomme_status.json', strip_big(meta))
        lim = find_key(meta, 'episode_limits')
        cands = []

        def scan(o, path=''):
            if isinstance(o, dict):
                for k, v in o.items():
                    scan(v, path + '/' + str(k))
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                p = path.lower()
                if 'step' in p and ('max' in p or 'limit' in p) and 'id' not in p.split('/')[-1]:
                    cands.append((p, int(o)))
        scan(lim if lim is not None else meta)
        log('step limit candidates', cands)
        vals = [v for p, v in cands if v > 50]
        if vals:
            return min(vals)
    except Exception as e:
        log('robomme_status failed', e)
    return None


# ----------------------------------------------------------------------------------------------
def decide_manner(ctx, obs0, cam):
    vf = video_frames(obs0.manifest)
    log('video frames', len(vf))
    if len(vf) < 2:
        return 'unknown', None
    # images: ~60 frames evenly over the whole video (frame 0 first)
    n = len(vf)
    sel = sorted(set(np.linspace(0, n - 1, min(n, 60)).round().astype(int).tolist()))
    try:
        frames = [np.array(load_png(vf[i]['rgb_path']).convert('RGB')) for i in sel]
    except Exception as e:
        log('video load failed', e)
        return 'unknown', None
    info = {}
    try:
        res = seg.classify_video(frames, log=log)
        info['persist'] = (res['manner'], res.get('pers'))
    except Exception as e:
        log('classify_video failed', traceback.format_exc())
        res = dict(manner='unknown', tools=[])
    f0 = frames[0]
    ms0 = seg.class_masks(f0)
    tools0 = res.get('tools') or []
    balls0 = seg.yellow_balls(f0, ms0)
    # metric typing on video frame 0 if depth exists
    proj0 = None
    try:
        dinfo = vf[0]['depth']
        if dinfo is not None and cam is not None:
            D = np.array(load_png(dinfo['path'])).astype(np.float64)
            if D.ndim == 3:
                D = D[..., 0]
            cm0 = Camera(cam.K, cam.pose, D, float(dinfo.get('meters_per_unit') or cam.scale), cam.v)

            class _O:
                pass
            o = _O(); o.depth = {'front': D}; o.step_id = None
            proj0 = perceive.Proj(None, o, cm0)
            tz0 = perceive.table_height(proj0, f0, ms0)
            T0 = seg.tool_mask(f0, ms0)
            unit0 = None
            try:
                sc0 = perceive.perceive_scene(proj0, f0, tz0)
                if sc0['cube'] is not None:
                    unit0 = sc0['cube']['side']
                log('video frame0 scene: cube', None if sc0['cube'] is None else (sc0['cube']['xy'].round(3), round(unit0, 3)),
                    'target', None if sc0['target'] is None else sc0['target']['xy'].round(3))
            except Exception as e:
                log('video scene fail', e)
            for t in tools0:
                try:
                    perceive.tool3d(proj0, f0, ms0, T0, t, tz0, unit=unit0)
                except Exception as e:
                    log('video tool3d fail', e)
            log('video metric types', [(t['type'], t.get('type_m'), t.get('tip_ext')) for t in tools0])
    except Exception as e:
        log('video depth typing failed', e)
    # --- state-based: where did the gripper close? ---
    manner_state = None
    obj_state = None
    try:
        states = [v['state'] for v in vf]
        if all(s is not None for s in states):
            S = np.array(states)
            op = S[:, 6] + S[:, 7]
            mx = float(np.max(op[:max(3, len(op) // 2)]))
            g = None
            for i in range(1, len(op)):
                if op[i] < 0.8 * mx and op[i] < mx - 0.012:
                    nxt = op[i:i + 6]
                    if len(nxt) >= 3 and np.all(nxt < 0.85 * mx):
                        g = i; break
            info['grasp_idx'] = g
            if g is not None:
                gi = min(len(S) - 1, g + 3)
                pg = S[gi, :3]
                log('video grasp at frame', gi, 'ee', pg.round(3), 'opening', op[gi].round(4))
                best = None
                cands = []
                for t in tools0:
                    cands.append((t['type'], t))
                for b in balls0:
                    cands.append(('ramp', b))
                for typ, c in cands:
                    if typ != 'ramp' and c.get('grasp') is not None:
                        dist = float(np.linalg.norm(c['grasp'][:2] - pg[:2]))
                        if best is None or dist < best[0]:
                            best = (dist, typ, c)
                        continue
                    dist = None
                    if proj0 is not None:
                        if typ == 'ramp':
                            P = proj0.pts(c['xs'], c['ys'])
                        else:
                            P = proj0.pts(c['handle_xs'], c['handle_ys'])
                        if len(P):
                            dist = float(np.linalg.norm(P.mean(0)[:2] - pg[:2]))
                    if dist is None and cam is not None:
                        uvd = cam.project(pg[None])[0]
                        dist = float(np.hypot(uvd[0] - c['u'], uvd[1] - c['v'])) * 0.004  # ~4 mm / px
                    if dist is not None and (best is None or dist < best[0]):
                        best = (dist, typ, c)
                log('video grasp nearest', None if best is None else best[:2])
                if best is not None and best[0] < 0.10:
                    manner_state = best[1]
                    obj_state = best[2]
    except Exception as e:
        log('state-based manner failed', traceback.format_exc())
    info['state'] = manner_state
    manner = res.get('manner', 'unknown')
    vobj = res.get('tool') if manner != 'ramp' else None
    if manner_state is not None:
        if manner_state != manner:
            log('manner disagreement: persistence', manner, 'state', manner_state, '-> state')
        manner = manner_state
        vobj = obj_state if manner != 'ramp' else None
    log('MANNER', manner, info, 'video tool', None if vobj is None else tool_desc(vobj))
    return manner, vobj


def tool_desc(t):
    return dict(kind=t.get('kind'), type=t.get('type'), type_m=t.get('type_m'), pn=round(float(t.get('pink_norm', 0)), 2),
                pp=round(float(t.get('pink_perp', 0)), 2), ext=None if t.get('tip_ext') is None else tuple(round(x, 3) for x in t['tip_ext']),
                L=None if t.get('L') is None else round(t['L'], 3), Lpx=round(float(t.get('length_px', 0)), 1))


def match_score(a, b):
    """Dissimilarity between a video tool a and an execution tool b (lower = more similar)."""
    if a.get('kind') != b.get('kind'):
        return 99.0
    s = 0.0
    s += 2.0 * (a.get('type') != b.get('type'))
    if a.get('kind') == 'teal':
        s += 0.5 * abs(np.log((a.get('pink_norm', 0) + 0.5) / (b.get('pink_norm', 0) + 0.5)))
        s += 0.35 * abs(a.get('pink_perp', 0) - b.get('pink_perp', 0))
        if a.get('tip_ext') is not None and b.get('tip_ext') is not None:
            s += 8.0 * abs(a['tip_ext'][0] - b['tip_ext'][0]) + 8.0 * abs(a['tip_ext'][1] - b['tip_ext'][1])
            s += 0.3 * (a.get('type_m') != b.get('type_m'))
    if a.get('L') is not None and b.get('L') is not None:
        s += 3.0 * abs(a['L'] - b['L'])
    return float(s)


# ----------------------------------------------------------------------------------------------
class Ctl:
    def __init__(self, ctx):
        self.ctx = ctx
        self.t0 = time.time()
        self.robot = None
        self.cam = None
        self.table_z = None
        self.base_xy = None
        self.R0 = None
        self.home = None
        self.nimg = 0
        self.finger_off = 0.0

    def save_img(self, obs, tag):
        try:
            from PIL import Image
            if 'front' in obs.rgb and self.nimg < 40:
                self.nimg += 1
                Image.fromarray(obs.rgb['front']).save(os.path.join('/workspace/output', 'f%02d_%s.png' % (self.nimg, tag)))
                if 'wrist' in obs.rgb:
                    Image.fromarray(obs.rgb['wrist']).save(os.path.join('/workspace/output', 'w%02d_%s.png' % (self.nimg, tag)))
        except Exception as e:
            log('save img fail', e)

    def look(self, tag):
        obs = self.robot.observe()
        if self.cam is not None and 'front' in obs.depth:
            self.cam.depth = obs.depth['front']
        self.save_img(obs, tag)
        return obs

    def proj(self, obs):
        return perceive.Proj(self.ctx, obs, self.cam)

    def scene(self, obs):
        return perceive.perceive_scene(self.proj(obs), obs.rgb['front'], self.table_z)

    # ---------------- geometry helpers ----------------
    def reach_pen(self, p):
        if self.base_xy is None:
            return 0.0
        r = float(np.linalg.norm(np.asarray(p[:2]) - self.base_xy))
        pen = 0.0
        if r > 0.74:
            pen += (r - 0.74) * 40
        if r < 0.22:
            pen += (0.22 - r) * 40
        return pen

    def clamp_reach(self, p):
        p = np.array(p, float)
        if self.base_xy is None:
            return p
        v = p[:2] - self.base_xy; r = np.linalg.norm(v)
        if r > 0.80:
            p[:2] = self.base_xy + v / r * 0.80
        return p

    def yaw_R(self, dth):
        return Rz(dth) @ self.robot.R

    # ---------------- grasping ----------------
    @staticmethod
    def wrap_half(a):
        """wrap to [-pi/2, pi/2) (parallel gripper is symmetric under 180 deg)."""
        a = wrap(a)
        if a >= np.pi / 2:
            a -= np.pi
        if a < -np.pi / 2:
            a += np.pi
        return a

    def fyaw(self):
        return self.robot.finger_yaw() + self.finger_off

    def wrist_cam(self, obs):
        if not ('wrist' in obs.K and 'wrist' in obs.pose and 'wrist' in obs.depth):
            return None
        D = obs.depth['wrist']
        if D.ndim == 3:
            D = D[..., 0]; obs.depth['wrist'] = D
        if self.cam is not None:
            return Camera(obs.K['wrist'], obs.pose['wrist'], D, obs.depth_scale.get('wrist', 0.001), self.cam.v)
        try:
            return calibrate(self.ctx, obs, cam='wrist', npts=8)
        except Exception as e:
            log('wrist calibrate failed', e)
            return None

    def wrist_refine(self, cls, xy_guess, max_dist=0.06):
        """Close-range localisation of a teal/dark handle or yellow ball with the wrist camera.
        Returns dict(xy, axis, top, fyaw_cam, n) or None."""
        try:
            obs = self.look('wrist_%s' % cls)
            im = obs.rgb.get('wrist')
            if im is None:
                log('wrist refine: no wrist image'); return None
            cmw = self.wrist_cam(obs)
            if cmw is None:
                log('wrist refine: no wrist geometry'); return None
            H, W = im.shape[:2]
            ms = seg.class_masks(im)
            m = ms[cls].copy()
            m[int(0.6 * H):, :int(0.32 * W)] = False   # finger regions (bottom corners)
            m[int(0.6 * H):, int(0.68 * W):] = False
            comps = seg.components(m, min_area=25)
            Rw, _ = cmw._Rt()
            f = Rw[:, 0]
            fyaw_cam = float(np.arctan2(f[1], f[0]))
            best = None
            for c in comps[:8]:
                xs, ys = c['xs'], c['ys']
                if len(xs) > 600:
                    idx = np.linspace(0, len(xs) - 1, 600).round().astype(int); xs, ys = xs[idx], ys[idx]
                P = cmw.pixels_to_world(xs, ys)
                P = P[np.all(np.isfinite(P), 1)]
                if len(P) < 10:
                    continue
                if np.median(P[:, 2]) > self.table_z + 0.09:
                    continue
                ctr = np.median(P[:, :2], 0)
                dist = float(np.linalg.norm(ctr - np.asarray(xy_guess)[:2]))
                if dist > max_dist:
                    continue
                if best is None or dist < best[0]:
                    best = (dist, P, ctr, c)
            if best is None:
                log('wrist refine: no %s component near guess' % cls, np.round(xy_guess, 3), 'comps', [cc['area'] for cc in comps[:8]],
                    'fyaw_cam %.1f deg' % np.rad2deg(fyaw_cam))
                return dict(xy=None, axis=None, top=None, fyaw_cam=fyaw_cam, n=0)
            dist, P, ctr, c = best
            Q = P[:, :2] - P[:, :2].mean(0)
            w, V = np.linalg.eigh(np.cov(Q.T))
            ax = V[:, 1]
            elong = float(np.sqrt(max(w[1], 1e-12) / max(w[0], 1e-12)))
            top = float(np.percentile(P[:, 2], 90))
            xy = P[:, :2].mean(0)
            log('wrist refine %s: xy' % cls, xy.round(3), 'shift %.3f' % dist, 'axis', ax.round(2), 'elong %.1f' % elong,
                'top %.3f' % top, 'n', len(P), 'fyaw_cam %.1f deg, fyaw_state %.1f deg' % (np.rad2deg(fyaw_cam), np.rad2deg(self.robot.finger_yaw())))
            return dict(xy=xy, axis=ax if elong > 1.6 else None, top=top, fyaw_cam=fyaw_cam, n=len(P))
        except EpisodeOver:
            raise
        except Exception:
            log('wrist refine crashed', traceback.format_exc())
            return None

    def grasp_at(self, xy, z_grasp, finger_yaw_des, z_app, refine=None, axis_perp=True):
        """Top-down grasp. refine: colour class for wrist-camera refinement ('teal','dark','yellow') or None.
        axis_perp: fingers close across the object's major axis (handles)."""
        rb = self.robot
        xy = np.asarray(xy, float)[:2].copy()
        dth = self.wrap_half(finger_yaw_des - self.fyaw())
        Rg = Rz(dth) @ rb.R
        rb.yaw_total += dth
        p = rb.pos.copy()
        rb.gripper(False, 4)
        if p[2] < z_app - 0.01:
            rb.move([p[0], p[1], z_app], Rg, max_lin=0.02)
        rb.move([xy[0], xy[1], z_app], Rg, max_lin=0.022, max_steps=120)
        if refine is not None:
            xy0 = xy.copy()
            for it in range(2):
                r = self.wrist_refine(refine, xy0, max_dist=0.06)
                if r is None:
                    break
                off = self.wrap_half(r['fyaw_cam'] - rb.finger_yaw())
                if abs(off) > np.deg2rad(20) and abs(self.wrap_half(off - self.finger_off)) > np.deg2rad(20):
                    log('finger axis from wrist camera differs from state assumption by %.1f deg -> adopt' % np.rad2deg(off))
                    self.finger_off = off
                if r['xy'] is None:
                    break
                shift = float(np.linalg.norm(r['xy'] - xy))
                xy = r['xy']
                dyaw = 0.0
                if r['axis'] is not None and axis_perp:
                    phi = float(np.arctan2(r['axis'][1], r['axis'][0]))
                    dyaw = self.wrap_half(phi + np.pi / 2 - self.fyaw())
                if r['top'] is not None and refine != 'yellow':
                    zt = r['top'] - 0.012
                    if abs(zt - z_grasp) < 0.03:
                        z_grasp = 0.5 * (z_grasp + max(zt, self.table_z + 0.008))
                Rg = Rz(dyaw) @ rb.R
                rb.yaw_total += dyaw
                log('refine pass %d: shift %.3f dyaw %.1f deg z_grasp %.3f' % (it, shift, np.rad2deg(dyaw), z_grasp))
                rb.move([xy[0], xy[1], z_app], Rg, max_lin=0.015, max_rot=0.1, max_steps=60)
                if shift < 0.008 and abs(dyaw) < np.deg2rad(6):
                    break
        rb.move([xy[0], xy[1], z_grasp + 0.035], Rg, max_lin=0.015, max_steps=60)
        rb.move([xy[0], xy[1], z_grasp], Rg, max_lin=0.006, max_steps=40, tol=0.003)
        rb.gripper(True, 14)
        op = rb.opening
        log('grasp closed: opening %.4f at' % op, rb.pos.round(3), 'target', np.round([xy[0], xy[1], z_grasp], 3))
        return op, Rg

    # ---------------- tool push ----------------
    def measure_tip_in_hand(self, tool, obs):
        """Tip position while the tool is lifted. Returns world xyz or None."""
        try:
            im = obs.rgb['front']; ms = seg.class_masks(im)
            m = ms['pink'] if tool['kind'] == 'teal' else ms['tan']
            m = m & seg.tool_mask(im, ms)
            ys, xs = np.nonzero(m)
            if len(xs) < 3:
                return None
            P = self.proj(obs).pts(xs, ys)
            if len(P) < 3:
                return None
            tcp = self.robot.pos
            el = P[:, 2] > self.table_z + 0.03
            P = P[el]
            if len(P) < 3:
                return None
            L = tool.get('L') or 0.3
            d = np.linalg.norm(P[:, :2] - tcp[:2], axis=1)
            ok = (d > 0.4 * L) & (d < 1.4 * L + 0.05)
            P = P[ok]; d = d[ok]
            if len(P) < 2:
                return None
            k = max(1, int(0.15 * len(P)))
            tip = P[np.argsort(-d)[:k]].mean(0)
            return tip
        except Exception as e:
            log('measure tip fail', e)
            return None

    def plan_push(self, cube_xy, side, target_xy, o_xy, tcp_xy):
        e = np.asarray(target_xy) - np.asarray(cube_xy)
        D = float(np.linalg.norm(e))
        d = e / max(D, 1e-6)
        tip0 = cube_xy - d * (side / 2 + 0.035)
        tip1 = target_xy - d * (side / 2 - 0.002)
        best = None
        for deg in range(-120, 121, 5):
            th = np.deg2rad(deg)
            if abs(self.robot.yaw_total + th) > np.deg2rad(175):
                continue
            o2 = Rz(th)[:2, :2] @ o_xy
            a = o2 / (np.linalg.norm(o2) + 1e-9)
            beta = float(np.arccos(np.clip(a @ d, -1, 1)))
            p0 = tip0 - o2; p1 = tip1 - o2
            cost = beta + 0.15 * abs(th) + self.reach_pen(p0) + self.reach_pen(p1)
            if beta > np.deg2rad(75):
                cost += 3.0
            if best is None or cost < best[0]:
                best = (cost, th, p0, p1, beta)
        cost, th, p0, p1, beta = best
        return dict(d=d, D=D, th=th, p0=p0, p1=p1, beta=beta, cost=cost)

    def tool_push(self, sc, tool):
        rb = self.robot
        tz = self.table_z
        ax = tool['handle_axis']
        phi_h = float(np.arctan2(ax[1], ax[0]))
        gxy = tool['grasp'][:2]
        zg = max(tool['handle_top'] - 0.012, tz + 0.010)
        z_app = tz + 0.13
        ramp_top = sc['ramp']['top'] if sc.get('ramp') else tz
        z_carry = max(tz + 0.16, ramp_top + 0.07)
        held = False
        for attempt in range(3):
            op, Rg = self.grasp_at(gxy, zg, phi_h + np.pi / 2, z_app, refine=tool['kind'])
            if op > 0.004:
                held = True
                break
            log('grasp missed (opening %.4f), retry' % op)
            rb.gripper(False, 8)
            rb.move(rb.pos + np.array([0, 0, 0.08]), None, max_lin=0.02, max_steps=20)
            obs = self.look('regrasp')
            sc2 = self.scene(obs)
            same = [t for t in sc2['tools'] if t['kind'] == tool['kind']]
            if same:
                t2 = min(same, key=lambda t: np.linalg.norm(t['grasp'][:2] - gxy))
                if np.linalg.norm(t2['grasp'][:2] - gxy) < 0.08:
                    tool = t2
                    ax = tool['handle_axis']; phi_h = float(np.arctan2(ax[1], ax[0]))
                    gxy = tool['grasp'][:2]; zg = max(tool['handle_top'] - 0.012, tz + 0.010)
            zg = max(zg - 0.004, tz + 0.007)
        if not held:
            log('could not grasp tool')
            return False
        grasp_pos = rb.pos.copy()
        # lift
        rb.move([grasp_pos[0], grasp_pos[1], z_carry], None, max_lin=0.012, max_steps=60, grip=1.0)
        if rb.opening < 0.002:
            log('tool dropped during lift (opening %.4f)' % rb.opening)
            rb.gripper(False, 6)
            return 'dropped'
        obs = self.look('lifted')
        tip_now = self.measure_tip_in_hand(tool, obs)
        o_pred = None
        if tool.get('tip') is not None:
            o_pred = tool['tip'] - np.array([grasp_pos[0], grasp_pos[1], zg])
        if tip_now is not None:
            o_meas = tip_now - rb.pos
            log('tip in hand', tip_now.round(3), 'offset', o_meas.round(3), 'pred', None if o_pred is None else o_pred.round(3))
            if o_pred is None or np.linalg.norm(o_meas[:2] - o_pred[:2]) < 0.10:
                o = o_meas
            else:
                o = o_pred
        else:
            o = o_pred
        if o is None:
            log('no tip estimate; using handle axis guess')
            e = tool['shaft']
            o = np.array([0.0, 0.0, 0.0])
            # world direction of shaft unknown in pixels -> use handle axis with sign towards image shaft
            o[:2] = ax * 0.25
        # droop: tip height relative to tcp vs on table
        droop = 0.0
        if tip_now is not None and o_pred is not None:
            droop = float((tip_now[2] - rb.pos[2]) - o_pred[2])
            droop = float(np.clip(droop, -0.05, 0.02))
        z_push = zg + 0.008 + max(0.0, -droop) * 0.8
        log('z_push %.3f droop %.3f' % (z_push, droop))
        o_xy = o[:2].copy()
        # ---- push cycles ----
        cube = sc['cube']; target = sc['target']
        for cyc in range(5):
            if cube is None or target is None:
                log('cube/target missing; stop'); return False
            err = float(np.linalg.norm(target['xy'] - cube['xy']))
            log('cycle', cyc, 'cube', cube['xy'].round(3), 'target', target['xy'].round(3), 'err %.3f' % err,
                'steps left', rb.steps_left())
            if err < 0.010 or rb.success:
                return True
            if rb.steps_left() < 90:
                log('few steps left'); return False
            pl = self.plan_push(cube['xy'], cube['side'], target['xy'], o_xy, rb.pos[:2])
            log('push plan th %.1f deg beta %.1f deg D %.3f' % (np.rad2deg(pl['th']), np.rad2deg(pl['beta']), pl['D']),
                'p0', pl['p0'].round(3), 'p1', pl['p1'].round(3))
            Rp = Rz(pl['th']) @ rb.R
            p = rb.pos.copy()
            if p[2] < z_carry - 0.01:
                rb.move([p[0], p[1], z_carry], None, max_lin=0.015, max_steps=40)
            y0 = rb.finger_yaw()
            rb.move([rb.pos[0], rb.pos[1], z_carry], Rp, max_lin=0.015, max_rot=0.08, max_steps=60)
            act = wrap(rb.finger_yaw() - y0)
            rb.yaw_total += act
            if abs(act - pl['th']) > np.deg2rad(5):
                log('yaw not reached: wanted %.1f got %.1f deg; re-plan with actual' % (np.rad2deg(pl['th']), np.rad2deg(act)))
            o_xy = Rz(act)[:2, :2] @ o_xy
            Rp = rb.R.copy()
            e_ = target['xy'] - cube['xy']; d_ = e_ / max(np.linalg.norm(e_), 1e-6)
            pl['p0'] = cube['xy'] - d_ * (cube['side'] / 2 + 0.035) - o_xy
            pl['p1'] = target['xy'] - d_ * (cube['side'] / 2 - 0.002) - o_xy
            if rb.opening < 0.002:
                log('tool lost (opening %.4f)' % rb.opening); return 'dropped'
            p0 = self.clamp_reach([pl['p0'][0], pl['p0'][1], z_carry])
            rb.move(p0, Rp, max_lin=0.02, max_steps=100)
            rb.move([p0[0], p0[1], z_push + 0.03], Rp, max_lin=0.012, max_steps=40)
            rb.move([p0[0], p0[1], z_push], Rp, max_lin=0.006, max_steps=30)
            p1 = self.clamp_reach([pl['p1'][0], pl['p1'][1], z_push])
            rb.move_linear(p1, Rp, speed=0.006, max_steps=int(pl['D'] / 0.005) + 40)
            # retreat
            back = p1 - np.r_[pl['d'] * 0.03, 0.0]
            rb.move_linear(back, Rp, speed=0.008, max_steps=15)
            rb.move([back[0], back[1], z_carry], Rp, max_lin=0.015, max_steps=40)
            # park out of view for measurement (towards the robot base)
            if self.home is not None:
                hp = self.home.copy(); hp[2] = z_carry
                rb.move(hp, Rp, max_lin=0.022, max_steps=60)
            obs = self.look('push%d' % cyc)
            sc2 = self.scene(obs)
            if sc2['cube'] is not None:
                cube = sc2['cube']
            else:
                log('cube not visible after push')
            if sc2['target'] is not None:
                target = sc2['target']
            elif sc2['cube'] is not None:
                log('target hidden (cube on it?)')
                if np.linalg.norm(cube['xy'] - target['xy']) < 0.02:
                    return True
        return True

    # ---------------- ramp ----------------
    def ramp_roll(self, sc):
        rb = self.robot; tz = self.table_z
        if not sc['balls'] or sc.get('ramp') is None or sc['cube'] is None:
            log('ramp: missing ball/ramp/cube'); return False
        ball = sc['balls'][0]
        pl = perceive.ramp_plane(sc['ramp']['P'])
        if pl is None:
            log('ramp plane fit failed; least-squares fallback')
            P = sc['ramp']['P']
            A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
            coef = np.linalg.lstsq(A, P[:, 2], rcond=None)[0]
            n = np.array([-coef[0], -coef[1], 1.0]); n /= np.linalg.norm(n)
            if np.linalg.norm(n[:2]) < 0.05:
                # no measurable slope: roll towards the cube from the ramp's far side
                dvec = sc['cube']['xy'] - P[:, :2].mean(0)
                dvec /= (np.linalg.norm(dvec) + 1e-9)
                n = np.array([dvec[0] * 0.3, dvec[1] * 0.3, 0.95]); n /= np.linalg.norm(n)
            cen = P.mean(0); inl = np.ones(len(P), bool)
            pl = (n, cen, inl)
        n, cen, inl = pl
        g = n[:2] / (np.linalg.norm(n[:2]) + 1e-9)   # downhill
        Q = sc['ramp']['P'][inl]
        al = (Q[:, :2] - cen[:2]) @ g
        pe = np.array([-g[1], g[0]])
        lat = (Q[:, :2] - cen[:2]) @ pe
        a_top = float(np.percentile(al, 3))
        half_w = float(np.percentile(np.abs(lat), 90))
        cube = sc['cube']['xy']
        c_lat = float((cube - cen[:2]) @ pe)
        c_lat = float(np.clip(c_lat, -0.6 * half_w, 0.6 * half_w))
        r = ball['r']
        rel_xy = cen[:2] + g * (a_top + r + 0.02) + pe * c_lat
        z_surf = float(cen[2] - (n[0] * (rel_xy[0] - cen[0]) + n[1] * (rel_xy[1] - cen[1])) / max(n[2], 0.2))
        z_rel = z_surf + r * 1.05 + 0.006
        tgt = sc['target']['xy'] if sc['target'] is not None else None
        log('ramp: n', n.round(3), 'downhill', g.round(3), 'release', rel_xy.round(3), 'z_rel %.3f' % z_rel,
            'cube', cube.round(3), 'target', None if tgt is None else tgt.round(3), 'ball', ball['xy'].round(3), 'r %.3f' % r)
        if tgt is not None:
            dct = (tgt - cube) / (np.linalg.norm(tgt - cube) + 1e-9)
            log('ramp alignment cos(downhill, cube->target) %.2f' % float(dct @ g))
        z_top = sc['ramp']['top']
        z_carry = max(z_top + r + 0.08, tz + 0.15)
        for attempt in range(2):
            zg = max(tz + r, tz + 0.012)
            op, Rg = self.grasp_at(ball['xy'], zg, float(np.arctan2(g[1], g[0])) + np.pi / 2, max(tz + 0.12, ball['top'] + 0.08), refine='yellow', axis_perp=False)
            if op > 0.004:
                break
            log('ball grasp missed %.4f' % op)
            rb.gripper(False, 6)
            rb.move(rb.pos + np.array([0, 0, 0.08]), None, max_lin=0.02, max_steps=20)
            obs = self.look('reball')
            sc2 = self.scene(obs)
            if not sc2['balls']:
                return False
            ball = sc2['balls'][0]
        else:
            return False
        rb.move([rb.pos[0], rb.pos[1], z_carry], None, max_lin=0.012, max_steps=60, grip=1.0)
        rb.move([rel_xy[0], rel_xy[1], z_carry], None, max_lin=0.018, max_steps=100)
        rb.move([rel_xy[0], rel_xy[1], z_rel + 0.03], None, max_lin=0.01, max_steps=40)
        rb.move([rel_xy[0], rel_xy[1], z_rel], None, max_lin=0.004, max_steps=30, tol=0.003)
        rb.hold(4, 1.0)
        rb.gripper(False, 10)
        rb.move(rb.pos + np.array([0, 0, 0.10]), None, max_lin=0.02, max_steps=20, grip=-1.0)
        if self.home is not None:
            hp = self.home.copy(); hp[2] = max(hp[2], rb.pos[2])
            rb.move(hp, None, max_lin=0.022, max_steps=60)
        rb.hold(50, -1.0)
        obs = self.look('after_roll')
        sc2 = self.scene(obs)
        if sc2['cube'] is not None and sc2['target'] is not None:
            log('after roll: cube', sc2['cube']['xy'].round(3), 'target', sc2['target']['xy'].round(3),
                'err %.3f' % np.linalg.norm(sc2['cube']['xy'] - sc2['target']['xy']))
        return True

    # ---------------- main ----------------
    def run(self):
        ctx = self.ctx
        max_steps = get_step_limit(ctx)
        log('step limit', max_steps)
        self.robot = rb = Robot(ctx, max_steps=max_steps, deadline=self.t0 + WALL_LIMIT)
        obs0 = rb.observe()
        dump_json('obs0_meta.json', strip_big(obs0.meta))
        log('obs0 step', obs0.step_id, 'state', None if obs0.state is None else np.round(obs0.state, 4),
            'rgb', list(obs0.rgb), 'depth', list(obs0.depth), 'K', list(obs0.K), 'pose', list(obs0.pose),
            'instruction', obs0.instruction)
        if obs0.manifest:
            dump_json('obs0_manifest_head.json', strip_big({k: v for k, v in obs0.manifest.items() if k != 'frames'}))
        self.save_img(obs0, 'start')
        if obs0.state is None:
            rb.hold(1, -1.0)
        self.R0 = rb.R.copy()
        self.home = rb.pos.copy()
        try:
            self.cam = calibrate(ctx, obs0)
        except Exception:
            log('calibrate crashed', traceback.format_exc()); self.cam = None
        try:
            manner, vobj = decide_manner(ctx, obs0, self.cam)
        except Exception:
            log('decide_manner crashed', traceback.format_exc()); manner = 'unknown'; vobj = None
        im = obs0.rgb['front']
        proj = self.proj(obs0)
        self.table_z = perceive.table_height(proj, im, seg.class_masks(im))
        if self.table_z is None:
            self.table_z = 0.0
        log('table z %.4f' % self.table_z)
        sc = perceive.perceive_scene(proj, im, self.table_z)
        log('scene: cube', None if sc['cube'] is None else (sc['cube']['xy'].round(3), round(sc['cube']['side'], 3)),
            'target', None if sc['target'] is None else sc['target']['xy'].round(3),
            'balls', [(b['xy'].round(3), round(b['r'], 3)) for b in sc['balls']],
            'ramp', None if sc['ramp'] is None else round(sc['ramp']['top'], 3),
            'tools', [(t['kind'], t['type'], t.get('type_px'), np.round(t['grasp'], 3), None if t.get('tip') is None else np.round(t['tip'], 3), t.get('tip_ext')) for t in sc['tools']])
        # robot base estimate: white blob at image top
        try:
            ms = sc['ms']
            wb = ms['white'].copy(); wb[:8] = False; wb[45:] = False
            cs = seg.components(wb, min_area=30)
            if cs:
                P = proj.pts(cs[0]['xs'], cs[0]['ys'])
                if len(P):
                    b = np.median(P, 0)
                    hd = np.linalg.norm(b[:2] - self.home[:2])
                    if hd < 0.6:
                        self.base_xy = b[:2]
            log('base estimate', None if self.base_xy is None else self.base_xy.round(3), 'home', self.home.round(3))
        except Exception as e:
            log('base estimate failed', e)
        # ---- dispatch ----
        order = []
        if manner == 'ramp':
            order = ['ramp', 'tool:ball', 'tool:wood', 'tool:hook', 'tool:spatula']
        elif manner in TEAL_TYPES or manner == 'wood':
            order = ['tool:' + manner] + ['tool:' + t for t in ('ball', 'wood', 'hook', 'spatula') if t != manner]
        else:
            order = ['tool:ball', 'tool:wood', 'tool:hook', 'tool:spatula', 'ramp']
        log('strategy order', order)
        for strat in order[:2]:
            if rb.ended or rb.steps_left() < 120:
                break
            try:
                if strat == 'ramp':
                    if self.ramp_roll(sc):
                        break
                else:
                    typ = strat.split(':')[1]
                    pool = [t for t in sc['tools'] if t.get('grasp') is not None]
                    if strat == order[0] and vobj is not None and pool:
                        scored = sorted(((match_score(vobj, t), i) for i, t in enumerate(pool)))
                        log('tool match scores', [(round(x, 2), tool_desc(pool[i])) for x, i in scored])
                        cands = [pool[i] for x, i in scored if x < 50]
                    else:
                        cands = [t for t in pool if t['type'] == typ]
                        if not cands and typ in TEAL_TYPES:
                            cands = [t for t in pool if t.get('type_m') == typ]
                        if sc['cube'] is not None:
                            cands.sort(key=lambda t: np.linalg.norm(t['grasp'][:2] - sc['cube']['xy']))
                    if not cands:
                        log('no tool of type', typ); continue
                    log('chosen tool', tool_desc(cands[0]), 'grasp', np.round(cands[0]['grasp'], 3))
                    res = self.tool_push(sc, cands[0])
                    if res == 'dropped' and not rb.ended and rb.steps_left() > 200:
                        log('retry after drop')
                        rb.move(rb.pos + np.array([0, 0, 0.06]), None, max_lin=0.02, max_steps=20, grip=-1.0)
                        if self.home is not None:
                            rb.move(self.home, None, max_lin=0.022, max_steps=60)
                        obs = self.look('after_drop')
                        sc2 = self.scene(obs)
                        same = [t for t in sc2['tools'] if t['kind'] == cands[0]['kind']]
                        if same and sc2['cube'] is not None and sc2['target'] is not None:
                            t2 = min(same, key=lambda t: match_score(cands[0], t))
                            self.tool_push(sc2, t2)
                    break  # do not try a second tool once one was grasped/used
            except EpisodeOver:
                raise
            except Exception:
                log('strategy', strat, 'crashed', traceback.format_exc())
                try:
                    obs = self.look('recover')
                    sc = self.scene(obs)
                except EpisodeOver:
                    raise
                except Exception:
                    pass
        log('run finished; steps', rb.step_id, 'success flag', rb.success)


def main(ctx):
    c = Ctl(ctx)
    try:
        c.run()
    except EpisodeOver:
        log('episode over (terminal)')
    except Exception:
        log('FATAL', traceback.format_exc())
    finally:
        try:
            if c.robot is not None:
                log('final step', c.robot.step_id, 'success', c.robot.success, 'calls', c.robot.n_calls)
        except Exception:
            pass
