"""MoveCube controller: imitate the manner shown in the conditioning video.

Manners (from the released demos): 'ramp' (drop the yellow ball on the high end of the ramp so it
rolls into the cube), or use a tool to sweep the cube onto the target: 'hook', 'ball' (ball-tipped
stick), 'paddle' (square paddle), 'hammer'.
"""
import json
import math
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import io_utils as iu  # noqa: E402
import vision as vz  # noqa: E402
from robot import Robot, EpisodeOver, wrap, rotz  # noqa: E402

OUT = '/workspace/output'
TOOL_MANNERS = ('hook', 'ball', 'paddle', 'hammer')


class Logger:
    def __init__(self):
        self.t0 = time.time()
        self.f = None
        try:
            os.makedirs(OUT, exist_ok=True)
            self.f = open(os.path.join(OUT, 'controller_log.txt'), 'a')
        except Exception:
            self.f = None

    def __call__(self, msg):
        line = f'[{time.time() - self.t0:7.1f}s] {msg}'
        try:
            print(line, flush=True)
        except Exception:
            pass
        if self.f:
            try:
                self.f.write(line + '\n'); self.f.flush()
            except Exception:
                pass


def save_image(arr, name):
    try:
        from PIL import Image
        Image.fromarray(arr).save(os.path.join(OUT, name))
    except Exception:
        pass


# ---------------------------------------------------------------------------- perception (3D)

def comp_pixels(comp, k=7):
    """Pick up to k pixels of a component near its centroid (robust to edges)."""
    us, vs = comp['us'], comp['vs']
    d = (us - comp['u']) ** 2 + (vs - comp['v']) ** 2
    order = np.argsort(d)
    n = len(order)
    take = order[:max(1, min(n, max(k, n // 3)))]
    if len(take) > k:
        take = take[np.linspace(0, len(take) - 1, k).astype(int)]
    return [(int(us[i]), int(vs[i])) for i in take]


def median_point(pts):
    good = [p for p, _ in pts if p is not None]
    if not good:
        return None
    return np.median(np.array(good), axis=0)


class Scene:
    def __init__(self, robot, log):
        self.r = robot
        self.log = log
        self.table_z = None
        self.objs = None
        self.im = None

    def refresh(self, tag=''):
        self.r.observe()
        im = self.r.images.get('front')
        if im is None:
            raise RuntimeError('no front image')
        self.im = im
        self.objs = vz.scene_objects(im)
        save_image(im, f'front_{tag}_{self.r.steps_used}.png')
        if self.table_z is None:
            self.estimate_table()
        return self.objs

    def estimate_table(self):
        tab = vz.table_like(self.im)
        sel = []
        for v in range(110, 240, 14):
            for u in range(20, 240, 36):
                if tab[max(0, v - 2):v + 3, max(0, u - 2):u + 3].all():
                    sel.append((u, v))
        sel = sel[:32]
        pts = self.r.pixels_to_points(sel)
        zs = [p[2] for p, _ in pts if p is not None]
        if zs:
            self.table_z = float(np.median(zs))
        else:
            self.table_z = 0.0
        self.log(f'table z = {self.table_z:.4f} from {len(zs)} pts')

    def point_of(self, comp, k=7):
        if comp is None:
            return None, []
        pts = self.r.pixels_to_points(comp_pixels(comp, k))
        good = [p for p, _ in pts if p is not None]
        return median_point(pts), good

    def point_uv(self, u, v):
        pts = self.r.pixels_to_points([(u, v)])
        return pts[0][0]

    def cube(self):
        c = self.objs.get('cube')
        p, good = self.point_of(c, 9)
        if p is None:
            return None, None
        zt = max(g[2] for g in good) if good else p[2]
        size = float(np.clip(zt - self.table_z, 0.015, 0.06))
        center = np.array([p[0], p[1], self.table_z + size / 2])
        return center, size

    def target(self):
        c = self.objs.get('target')
        p, _ = self.point_of(c, 9)
        if p is None:
            return None
        return np.array([p[0], p[1], self.table_z])


# ---------------------------------------------------------------------------- manner

def decide_manner(robot, log):
    frames, idx = iu.load_video_frames(robot.meta, max_frames=80, log=log)
    frames = [f if f.shape[:2] == (256, 256) else np.array(__import__('PIL.Image', fromlist=['Image']).fromarray(f).resize((256, 256))) for f in frames]
    if not frames:
        log('no video frames; default manner')
        return None, None
    manner, info = vz.classify_video(frames, log=log)
    try:
        save_image(frames[0], 'video_first.png')
        save_image(frames[-1], 'video_last.png')
    except Exception:
        pass
    log(f'video manner = {manner}; scores {[(k, round(s, 2)) for k, s in info.get("scores", [])]}')
    chosen = info.get('chosen')
    feat = None
    if isinstance(chosen, dict) and 'end_feat' in chosen:
        feat = chosen['end_feat']
    return manner, feat


# ---------------------------------------------------------------------------- skills

class Skills:
    def __init__(self, robot, scene, log):
        self.r = robot
        self.s = scene
        self.log = log

    @property
    def zt(self):
        return self.s.table_z

    def go_home(self, z_min=None):
        h = self.r.home.copy()
        if z_min is not None:
            h[2] = max(h[2], z_min)
        p = self.r.pos
        if p[2] < h[2] - 0.01:
            self.r.move_to([p[0], p[1], h[2]], vmax=0.02, label='up')
        self.r.move_to(h, psi=None, vmax=0.025, tol=0.01, atol=0.2, label='home')

    def calibrate_close_axis(self):
        """Fingers close along the wrist image's horizontal axis: measure it in the world."""
        try:
            self.r.observe()
            if 'wrist' not in self.r.images:
                self.log('no wrist image; keep default close axis')
                return
            pts = self.r.pixels_to_points([(40, 128), (216, 128), (40, 170), (216, 170)], camera='wrist')
            dirs = []
            for a, b in ((0, 1), (2, 3)):
                if pts[a][0] is not None and pts[b][0] is not None:
                    d = pts[b][0] - pts[a][0]
                    d[2] = 0
                    if np.linalg.norm(d) > 0.01:
                        dirs.append(d / np.linalg.norm(d))
            if not dirs:
                self.log('wrist calibration: no valid points')
                return
            d = dirs[0]
            R0 = self.r.R0
            best, bi = -1, self.r.close_axis
            for i in range(3):
                if i == self.r.approach_axis:
                    continue
                ax = R0[:, i].copy(); ax[2] = 0
                n = np.linalg.norm(ax)
                if n < 0.3:
                    continue
                c = abs(ax @ d) / n
                if c > best:
                    best, bi = c, i
            self.log(f'wrist image-x dir {np.round(d, 2)}; close axis -> {bi} (|cos|={best:.2f})')
            if best > 0.7:
                self.r.close_axis = bi
        except EpisodeOver:
            raise
        except Exception as e:
            self.log(f'calibrate_close_axis failed: {e}')

    def grasp_at(self, p, close_dir_xy, z_grasp, approach_h=0.10, label=''):
        """Top-down grasp at xy of p with fingers closing along close_dir_xy. Returns True if holding."""
        psi = self.r.yaw_for_close_dir(close_dir_xy)
        above = np.array([p[0], p[1], max(z_grasp + approach_h, self.zt + 0.12)])
        self.r.grip = -1.0
        cur = self.r.pos
        if cur[2] < above[2] - 0.02:
            self.r.move_to([cur[0], cur[1], above[2]], label='lift-before-approach')
        self.r.move_to(above, psi=psi, vmax=0.025, tol=0.006, atol=0.04, label=f'above {label}')
        self.r.set_gripper(-1.0, n=6)
        self.r.move_to([p[0], p[1], z_grasp + 0.03], psi=psi, vmax=0.012, tol=0.004, label=f'pre {label}')
        self.r.move_to([p[0], p[1], z_grasp], psi=psi, vmax=0.006, tol=0.003, max_iter=40, label=f'down {label}')
        self.r.set_gripper(1.0, n=14)
        f = self.r.fingers
        self.log(f'grasp {label}: fingers sum {f:.4f} at {np.round(self.r.pos, 3)}')
        return f > 0.004, psi

    # ------------------------------------------------------------- ramp manner
    def do_ramp(self):
        for attempt in range(3):
            objs = self.s.refresh(f'ramp{attempt}')
            if objs['ball'] is None or objs['ramp'] is None:
                self.log('ramp manner: ball or ramp not visible')
                return False
            ball_p, ball_pts = self.s.point_of(objs['ball'], 9)
            if ball_p is None:
                self.log('ball 3D failed'); return False
            ztop = max(p[2] for p in ball_pts)
            r = float(np.clip((ztop - self.zt) / 2, 0.012, 0.035))
            ball_c = np.array([ball_p[0], ball_p[1], self.zt + r])
            cube_c, cube_size = self.s.cube()
            # ramp samples
            rc = objs['ramp']
            n = len(rc['us'])
            ii = np.linspace(0, n - 1, min(n, 64)).astype(int)
            pts = self.r.pixels_to_points([(int(rc['us'][i]), int(rc['vs'][i])) for i in ii], radius=0)
            P = np.array([p for p, sp in pts if p is not None and (sp is None or sp < 0.03)])
            if len(P) < 8:
                self.log('ramp 3D failed'); return False
            zmax = float(np.percentile(P[:, 2], 97))
            top = P[P[:, 2] >= zmax - 0.012]
            top_c = top.mean(0)
            low = P[P[:, 2] <= np.percentile(P[:, 2], 30)]
            down = (low.mean(0) - top_c); down[2] = 0
            if cube_c is not None:
                dc = cube_c - top_c; dc[2] = 0
                if np.linalg.norm(dc) > 0.02:
                    # the ramp is aimed at the cube; prefer that heading when consistent
                    dcn = dc / np.linalg.norm(dc)
                    if np.linalg.norm(down) < 1e-3 or (down / np.linalg.norm(down)) @ dcn > -0.2:
                        down = dcn
            down = down / max(np.linalg.norm(down), 1e-6)
            self.log(f'ramp zmax {zmax:.3f} (table {self.zt:.3f}) top {np.round(top_c, 3)} down {np.round(down, 2)} ball r {r:.3f} at {np.round(ball_c, 3)}')
            # grasp ball (fingers close perpendicular to nothing in particular: keep current yaw)
            ok, psi = self.grasp_at(ball_c, self.r.close_dir(), z_grasp=ball_c[2], label='ball')
            if not ok:
                self.log('ball grasp failed; retry')
                self.r.set_gripper(-1.0, n=6)
                self.go_home()
                continue
            place = top_c + down * 0.03
            surf = zmax - 0.01
            zc = max(surf + r + 0.015, self.zt + 2 * r + 0.01)
            self.r.move_to([self.r.pos[0], self.r.pos[1], max(zmax + 0.10, zc + 0.06)], vmax=0.02, label='lift ball')
            self.r.move_to([place[0], place[1], max(zmax + 0.10, zc + 0.06)], vmax=0.025, tol=0.006, label='over ramp')
            self.r.move_to([place[0], place[1], zc], vmax=0.01, tol=0.004, label='lower ball')
            self.r.set_gripper(-1.0, n=8)
            self.r.move_to([place[0], place[1], zc + 0.08], vmax=0.02, tol=0.01, label='retreat up')
            self.r.hold(20)
            self.go_home()
            self.r.hold(30)
            objs = self.s.refresh('after_ramp')
            c2, _ = self.s.cube()
            t = self.s.target()
            if c2 is not None and t is not None:
                d = np.linalg.norm((c2 - t)[:2])
                moved = None if cube_c is None else np.linalg.norm((c2 - cube_c)[:2])
                self.log(f'after ball: cube-target {d:.3f} m, cube moved {moved}')
                if d < 0.03 or self.r.success:
                    return True
                if moved is not None and moved > 0.03:
                    return False  # it did roll into the cube; nothing more within this manner
            if self.r.success:
                return True
        return False

    # ------------------------------------------------------------- gripper push
    def do_push(self):
        self.s.refresh('push')
        cube_c, size = self.s.cube()
        tgt = self.s.target()
        self.r.grip = 1.0
        self.r.set_gripper(1.0, n=10)
        z_push = self.zt + float(np.clip(size * 0.45, 0.008, 0.025)) if size else self.zt + 0.012
        z_travel = self.zt + 0.10
        for k in range(3):
            if cube_c is None or tgt is None:
                return False
            u = tgt - cube_c; u[2] = 0
            D = float(np.linalg.norm(u))
            self.log(f'push {k}: cube {np.round(cube_c, 3)} target {np.round(tgt, 3)} dist {D:.3f}')
            if D < 0.012:
                return True
            u /= D
            half = size / 2
            gs = cube_c - u * (half + 0.035)
            ge = tgt - u * (half + 0.008)
            psi = self.r.yaw_for_close_dir(u)  # fingers side by side across the push direction -> flat face
            self.r.move_to([self.r.pos[0], self.r.pos[1], z_travel], vmax=0.02, label='push up')
            self.r.move_to([gs[0], gs[1], z_travel], psi=psi, vmax=0.025, tol=0.006, label='push over start')
            self.r.move_to([gs[0], gs[1], z_push], psi=psi, vmax=0.01, tol=0.004, label='push lower')
            self.r.move_to([ge[0], ge[1], z_push], psi=psi, vmax=0.006, tol=0.004, max_iter=120, chunk=6, label='push')
            back = self.r.pos - u * 0.03
            self.r.move_to([back[0], back[1], z_push], psi=psi, vmax=0.006, label='push back')
            self.r.move_to([back[0], back[1], z_travel], psi=psi, vmax=0.02, label='push lift')
            if self.r.success:
                return True
            self.go_home(z_min=z_travel)
            self.s.refresh(f'after_push{k}')
            c2, s2 = self.s.cube()
            t2 = self.s.target()
            if c2 is not None:
                cube_c, size = c2, s2
            if t2 is not None:
                tgt = t2
        return False

    # ------------------------------------------------------------- tool manners
    def pick_tool(self, manner, video_feat):
        objs = self.s.objs
        if manner == 'hammer':
            if objs['hammers']:
                return objs['hammers'][0]
            return None
        tools = objs['teal_tools']
        if not tools:
            return None
        def score(t):
            sc = t['end_feat'].get('scores') or {}
            return sc.get(manner, -99)
        best = max(tools, key=score)
        self.log('tools: ' + '; '.join(f"{t['end_kind']}@({t['handle'][0]:.0f},{t['handle'][1]:.0f}) s={score(t):.2f}" for t in tools))
        return best

    def tool_geometry(self, tool):
        """3D grasp point, finger closing direction, shaft direction and contact offset."""
        if tool['part'] == 'teal':
            g, gpts = self.s.point_of(tool['comp'], 7)
            eu, ev = tool['end']
            hu, hv = tool['handle']
            # points along the shaft from the handle to the end
            alongs = [0.35, 0.55, 0.75, 0.95]
            sels = [(hu + (eu - hu) * a, hv + (ev - hv) * a) for a in alongs]
            sp = self.s.r.pixels_to_points(sels, radius=0)
            if g is None:
                return None
            ends = [p for p, _ in sp if p is not None]
            if not ends:
                return None
            e = ends[-1]
            a = e - g; a[2] = 0
            L = float(np.linalg.norm(a))
            if L < 0.04:
                return None
            a = a / L
            close = np.array([-a[1], a[0], 0.0])
            ztop = max(p[2] for p in gpts) if gpts else g[2]
            hgt = max(0.008, ztop - self.zt)
            zg = self.zt + float(np.clip(hgt * 0.55, 0.008, 0.022))
            # contact point on the shaft, ~60% of the way to the end (beyond the fingers)
            s_c = float(np.clip(0.62 * L, 0.06, L - 0.02))
            v = a * s_c
            return {'grasp': g, 'close': close, 'axis': a, 'L': L, 'z_grasp': zg, 'v': v, 'shaft_h': hgt}
        else:  # hammer
            head = tool['comp']
            g, gpts = self.s.point_of(head, 7)
            if g is None:
                return None
            ang = tool['head_axis_img']
            hu, hv = tool['head']
            span = max(4.0, 0.35 * math.sqrt(head['area']) * 1.5)
            p1 = self.s.point_uv(hu - math.cos(ang) * span, hv - math.sin(ang) * span)
            p2 = self.s.point_uv(hu + math.cos(ang) * span, hv + math.sin(ang) * span)
            if p1 is None or p2 is None:
                return None
            hax = p2 - p1; hax[2] = 0
            if np.linalg.norm(hax) < 1e-3:
                return None
            hax /= np.linalg.norm(hax)
            close = np.array([-hax[1], hax[0], 0.0])
            tip = self.s.point_uv(*tool['handle_tip'])
            mid = self.s.point_of(tool['tan_comp'], 5)[0]
            if tip is None or mid is None:
                return None
            a = tip - g; a[2] = 0
            L = float(np.linalg.norm(a))
            if L < 0.04:
                return None
            a /= L
            v = (mid - g) * 0.4 + (tip - g) * 0.6
            v[2] = 0
            ztop = max(p[2] for p in gpts) if gpts else g[2]
            hgt = max(0.008, ztop - self.zt)
            zg = self.zt + float(np.clip(hgt * 0.55, 0.008, 0.022))
            return {'grasp': g, 'close': close, 'axis': a, 'L': L, 'z_grasp': zg, 'v': v, 'shaft_h': hgt}

    def plan_sweep(self, geo, psi_g, cube_c, cube_size, tgt, tool_psi0):
        """Return (psi, gripper start, gripper end) for a side sweep of the shaft across the cube."""
        u = tgt - cube_c; u[2] = 0
        D = float(np.linalg.norm(u))
        if D < 1e-4:
            return None
        u /= D
        half = cube_size / 2
        best = None
        for sgn in (1, -1):
            want = np.array([-u[1], u[0], 0.0]) * sgn  # desired shaft direction
            a0 = geo['axis']
            d_ang = wrap(math.atan2(want[1], want[0]) - math.atan2(a0[1], a0[0]))
            psi = wrap(psi_g + d_ang)
            Rz = rotz(d_ang)
            v = Rz @ geo['v']
            q_start = cube_c - u * (half + 0.04)
            q_end = tgt - u * (half + 0.007)
            gs = q_start - v; ge = q_end - v
            cost = abs(psi) * 0.08 + 0.5 * np.linalg.norm((ge - self.r.home)[:2]) + 0.3 * np.linalg.norm((gs - self.r.home)[:2])
            if abs(psi) > 2.4:
                cost += 1.0
            if best is None or cost < best[0]:
                best = (cost, psi, gs, ge, d_ang)
        _, psi, gs, ge, d_ang = best
        return {'psi': psi, 'start': gs, 'end': ge, 'u': u, 'D': D, 'd_ang': d_ang}

    def do_tool(self, manner, video_feat):
        objs = self.s.refresh('tool')
        tool = self.pick_tool(manner, video_feat)
        if tool is None:
            self.log(f'no tool for manner {manner}')
            return False
        geo = self.tool_geometry(tool)
        if geo is None:
            self.log('tool geometry failed')
            return False
        self.log(f"tool {tool['end_kind']}: grasp {np.round(geo['grasp'], 3)} axis {np.round(geo['axis'], 2)} L {geo['L']:.3f} zg {geo['z_grasp']:.3f}")
        held = False
        for attempt in range(2):
            ok, psi_g = self.grasp_at(geo['grasp'], geo['close'], geo['z_grasp'], label=f'tool{attempt}')
            if ok:
                held = True
                break
            self.r.set_gripper(-1.0, n=6)
            self.r.move_to(self.r.pos + np.array([0, 0, 0.08]), label='up after miss')
            self.go_home()
            objs = self.s.refresh(f'tool_re{attempt}')
            tool = self.pick_tool(manner, video_feat)
            if tool is None:
                return False
            geo = self.tool_geometry(tool) or geo
        if not held:
            return False
        z_sweep = geo['z_grasp'] + 0.006
        z_travel = self.zt + 0.11
        # initial cube/target from the pre-grasp scene
        cube_c, cube_size = self.s.cube()
        tgt = self.s.target()
        self.r.move_to([self.r.pos[0], self.r.pos[1], z_travel], vmax=0.015, label='lift tool')
        for sweep in range(3):
            if cube_c is None or tgt is None:
                self.log('cube/target unknown'); return False
            dist = float(np.linalg.norm((tgt - cube_c)[:2]))
            self.log(f'sweep {sweep}: cube {np.round(cube_c, 3)} size {cube_size:.3f} target {np.round(tgt, 3)} dist {dist:.3f}')
            if dist < 0.012:
                return True
            plan = self.plan_sweep(geo, psi_g, cube_c, cube_size, tgt, psi_g)
            if plan is None:
                return True
            self.log(f"plan psi {plan['psi']:.2f} start {np.round(plan['start'], 3)} end {np.round(plan['end'], 3)}")
            gs, ge = plan['start'], plan['end']
            self.r.move_to([self.r.pos[0], self.r.pos[1], z_travel], vmax=0.015, label='travel up')
            self.r.move_to([self.r.pos[0], self.r.pos[1], z_travel], psi=plan['psi'], wmax=0.08, vmax=0.01, atol=0.05, max_iter=80, label='rotate tool')
            self.r.move_to([gs[0], gs[1], z_travel], psi=plan['psi'], vmax=0.02, tol=0.006, atol=0.05, label='over start')
            self.r.move_to([gs[0], gs[1], z_sweep], psi=plan['psi'], vmax=0.008, tol=0.004, atol=0.05, label='lower tool')
            self.r.move_to([ge[0], ge[1], z_sweep], psi=plan['psi'], vmax=0.006, tol=0.004, atol=0.05, max_iter=120, chunk=6, label='sweep')
            self.r.hold(4)
            # back off a little against the sweep direction, then lift
            back = self.r.pos - plan['u'] * 0.02
            self.r.move_to([back[0], back[1], z_sweep + 0.002], psi=plan['psi'], vmax=0.006, label='back off')
            self.r.move_to([back[0], back[1], z_travel], psi=plan['psi'], vmax=0.015, label='lift after sweep')
            if self.r.success:
                return True
            self.go_home(z_min=z_travel)
            self.r.hold(3)
            self.s.refresh(f'after_sweep{sweep}')
            c2, sz2 = self.s.cube()
            t2 = self.s.target()
            if c2 is not None:
                cube_c, cube_size = c2, sz2
            if t2 is not None:
                tgt = t2
            if self.r.fingers < 0.003:
                self.log('tool lost from the gripper')
                return False
        return False


# ---------------------------------------------------------------------------- main

def main(ctx):
    log = Logger()
    log('MoveCube controller start')
    robot = Robot(ctx, log, wall_limit=1120.0)
    try:
        robot.observe()
        log('first packet: ' + json.dumps({k: (v if not isinstance(v, (dict, list)) else type(v).__name__) for k, v in robot.meta.items()})[:800])
        robot.read_limits()
        if robot.state is None:
            log('no state in observation; abort')
            return
        robot.init_frames()
        instr = iu.find_key(robot.meta, ('instruction',), lambda x: isinstance(x, str))
        log(f'instruction: {instr}')
        try:
            manner, feat = decide_manner(robot, log)
        except Exception as e:
            log('decide_manner failed: ' + traceback.format_exc()[-800:])
            manner, feat = None, None
        scene = Scene(robot, log)
        sk = Skills(robot, scene, log)
        sk.calibrate_close_axis()
        scene.refresh('start')
        if manner is None:
            manner = 'push'
            log(f'fallback manner {manner}')
        log(f'executing manner {manner}')
        ok = False
        if manner == 'ramp':
            ok = sk.do_ramp()
        elif manner == 'push':
            ok = sk.do_push()
        else:
            ok = sk.do_tool(manner, feat)
        log(f'manner {manner} finished ok={ok} success_flag={robot.success}')
    except EpisodeOver as e:
        log(f'episode over: {e}')
    except Exception:
        log('controller exception: ' + traceback.format_exc()[-1500:])
    finally:
        log(f'done; steps used {robot.steps_used} errors {robot.n_errors} success_flag={robot.success}')
