"""Manipulation skills for MoveCube: tool grasping, tool models, planning and pushing."""
import math
import numpy as np
import perception as P
from scene import Scene, CUBE_H, BASE_XY
from geom import wrap, Rz, R2, R_down, R_goal, yaw_of, ang, unit, perp, ObstacleGrid
from motion import rotate_yaw, EpisodeEnded

YAW_MIN, YAW_MAX = math.radians(-165), math.radians(170)
REACH_MIN, REACH_MAX = 0.22, 0.74


class Fail(Exception):
    pass


# ----------------------------------------------------------------------------- tool model
class ToolModel:
    """Tool geometry in the gripper's yaw frame: xy rotated by -yaw about the TCP, z relative to TCP.
    pitch: hand rotation about its finger (y) axis; 0 = tool lying flat as grasped."""

    def __init__(self, kind, pts_q, contact_q, dir_q, r_contact, standoff, lat_tol, max_dev,
                 contact_z=-0.004, lift=0.004):
        self.kind = kind; self.pts_q = pts_q; self.contact_q = np.asarray(contact_q, float)
        self.dir_q = unit(dir_q); self.r = r_contact; self.standoff = standoff
        self.lat_tol = lat_tol; self.max_dev = max_dev
        self.zbot = float(np.percentile(pts_q[:, 2], 3)) if len(pts_q) else -0.01
        self.bot_q = np.full(len(pts_q), self.zbot)
        self.extent = float(np.max(np.linalg.norm(pts_q[:, :2], axis=1))) if len(pts_q) else 0.3
        self.contact_z = contact_z; self.lift = lift
        self.pitch = 0.0; self.phi = 0.0

    def set_table(self, tcp):
        self.zbot = -float(tcp[2]); self.bot_q = np.full(len(self.pts_q), self.zbot)

    def work_z(self, grasp_z):
        if self.phi == 0.0:
            return grasp_z + self.lift
        return 0.018 - self.contact_z

    def pitched(self, phi):
        """Copy with the tool tipped down by phi (the far end along dir_q goes down)."""
        import copy
        m = copy.copy(self)
        a = unit(self.contact_q)  # tool axis from TCP toward the working end
        s_i = self.pts_q[:, :2] @ a
        q = self.pts_q.copy()
        q[:, :2] = q[:, :2] + np.outer(s_i * (math.cos(phi) - 1), a)
        q[:, 2] = q[:, 2] - s_i * math.sin(phi)
        m.pts_q = q
        m.bot_q = self.zbot - s_i * math.sin(phi)
        sc = self.contact_q @ a
        m.contact_q = self.contact_q + a * sc * (math.cos(phi) - 1)
        m.contact_z = self.contact_z - sc * math.sin(phi)
        m.phi = phi
        m.pitch = -phi if a[0] >= 0 else phi
        m.pitched_base = self
        m.extent = float(np.max(np.linalg.norm(q[:, :2], axis=1)))
        return m

    def contact_world(self, tcp, yaw):
        return np.asarray(tcp[:2]) + R2(yaw) @ self.contact_q

    def dir_world(self, yaw):
        return R2(yaw) @ self.dir_q

    def pts_world(self, tcp, yaw, sub=3):
        q = self.pts_q[::sub]
        xy = (R2(yaw) @ q[:, :2].T).T + np.asarray(tcp[:2])
        return xy, q[:, 2] + tcp[2]

    def bottoms_world(self, tcp, yaw, sub=3):
        q = self.pts_q[::sub]
        xy = (R2(yaw) @ q[:, :2].T).T + np.asarray(tcp[:2])
        return xy, self.bot_q[::sub] + tcp[2]

    def tcp_for(self, C, yaw):
        return np.asarray(C) - R2(yaw) @ self.contact_q


def to_q(pts, tcp, yaw):
    q = np.zeros_like(pts)
    q[:, :2] = (R2(-yaw) @ (pts[:, :2] - np.asarray(tcp[:2])).T).T
    q[:, 2] = pts[:, 2] - tcp[2]
    return q


def build_teal_model(tool, tcp, yaw):
    """tool: Scene tool dict (pre-grasp points). tcp/yaw: grasp pose."""
    q = to_q(tool['pts'], tcp, yaw)
    a_q = unit(R2(-yaw) @ tool['axis'])
    s = q[:, :2] @ a_q; l = q[:, :2] @ perp(a_q)
    Lmax = float(np.percentile(s, 99))
    tip = s > Lmax - 0.08
    kind = tool['kind']
    if kind == 'hook':
        st, lt = s[tip], l[tip]
        s_c = float(np.percentile(st, 97))
        llo, lhi = float(np.percentile(lt, 4)), float(np.percentile(lt, 96))
        lmid = 0.5 * (llo + lhi)
        # arm depth: points on the side away from the shaft
        shaft_l = float(np.median(l[(s > 0.08) & (s < Lmax - 0.1)])) if ((s > 0.08) & (s < Lmax - 0.1)).sum() > 3 else llo
        other = lhi if abs(lhi - shaft_l) > abs(llo - shaft_l) else llo
        arm = tip & (np.abs(l - other) < 0.012)
        depth = float(s_c - np.percentile(s[arm], 5)) if arm.sum() >= 3 else 0.045
        depth = float(np.clip(depth, 0.025, 0.07))
        contact = a_q * (s_c - 0.008) + perp(a_q) * lmid
        half_in = 0.5 * (lhi - llo) - 0.006
        m = ToolModel('hook', q, contact, -a_q, 0.0, CUBE_H + depth + 0.012,
                      lat_tol=max(0.003, half_in - CUBE_H - 0.002), max_dev=math.radians(12))
        m.set_table(tcp)
        m.depth = depth; m.half_in = half_in
        return m
    if kind == 'plate':
        lt = l[tip]
        lmid = float(np.median(lt))
        contact = a_q * Lmax + perp(a_q) * lmid
        half = 0.5 * (np.percentile(lt, 95) - np.percentile(lt, 5))
        m = ToolModel('plate', q, contact, a_q, 0.002, CUBE_H + 0.03,
                      lat_tol=max(0.01, half - 0.012), max_dev=math.radians(15))
        m.set_table(tcp)
        return m
    # stick: ball tip
    end = s > Lmax - 0.03
    tipc = np.median(q[end, :2], 0) if end.sum() >= 2 else a_q * (Lmax - 0.012)
    cz = float(np.median(q[end, 2])) if end.sum() >= 2 else -0.004
    m = ToolModel('stick', q, tipc, a_q, 0.014, CUBE_H + 0.014 + 0.03, lat_tol=0.012,
                  max_dev=math.radians(95), contact_z=cz)
    m.set_table(tcp)
    return m


def build_hammer_model(hm, tcp, yaw):
    pts = np.concatenate([hm['slate'], hm['tan']]) if len(hm['tan']) else hm['slate']
    q = to_q(pts, tcp, yaw)
    tip = to_q(hm['tan_tip'][None], tcp, yaw)[0]
    m = ToolModel('hammer', q, tip[:2], tip[:2], 0.008, CUBE_H + 0.008 + 0.03, lat_tol=0.012,
                  max_dev=math.radians(60), contact_z=float(tip[2]), lift=0.006)
    m.set_table(tcp)
    return m


# ----------------------------------------------------------------------------- obstacles
def obstacle_grid(scene, exclude_pts=(), exclude_xy=(), excl_r=0.025):
    m = scene.elev.copy()
    pts = scene.Pimg[m]
    keep = np.ones(len(pts), bool)
    keep &= pts[:, 2] < 0.26
    for c, r in exclude_xy:
        keep &= np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) > r
    for ep in exclude_pts:
        if ep is None or len(ep) == 0: continue
        # exclude points within excl_r of any tool point (grid hashing)
        g = ObstacleGrid(ep, res=0.01)
        ij = np.floor((pts[:, :2] + g.lim) / g.res).astype(int)
        n = g.h.shape[0]; r = int(math.ceil(excl_r / g.res))
        occ = g.h > -1
        from numpy.lib.stride_tricks import sliding_window_view  # noqa
        dil = occ.copy()
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                dil |= np.roll(np.roll(occ, di, 0), dj, 1)
        ok = (ij >= 0).all(1) & (ij < n).all(1)
        hit = np.zeros(len(pts), bool); hit[ok] = dil[ij[ok, 0], ij[ok, 1]]
        keep &= ~hit
    return ObstacleGrid(pts[keep])


def reach_ok(xy):
    r = np.hypot(xy[0] - BASE_XY[0], xy[1] - BASE_XY[1])
    return REACH_MIN < r < REACH_MAX


def yaw_choice(yaw, ref):
    """Representation of yaw (mod 2pi) closest to ref inside the joint window, or None."""
    best = None
    for k in (-2, -1, 0, 1, 2):
        y = yaw + 2 * math.pi * k
        if YAW_MIN <= y <= YAW_MAX and (best is None or abs(y - ref) < abs(best - ref)):
            best = y
    return best


# ----------------------------------------------------------------------------- planning
def collides(model, tcp_xy, yaw, work_z, grid, margin, clearance=0.002):
    xy, bot = model.bottoms_world(np.r_[tcp_xy, work_z], yaw)
    h = grid.heights(xy, margin)
    return bool(np.any(h > bot - clearance)), float(np.max(h - bot)) if len(h) else -1


def plan_push(model, cube, target, grid, ref_yaw, grasp_z, log=print, extra_back=0.0, margin=0.02,
              pitches=(0.0,)):
    """Choose tool pitch, yaw and pre-contact TCP. Returns dict or None."""
    c = np.asarray(cube[:2]); T = np.asarray(target[:2])
    d = unit(T - c)
    devs = [0.0]
    step = math.radians(8)
    k = 1
    while k * step <= model.max_dev + 1e-6:
        devs += [k * step, -k * step]; k += 1
    best = None
    for phi in pitches:
        m = model if phi == 0 else model.pitched(phi)
        work_z = m.work_z(grasp_z)
        rmax = REACH_MAX - (0.04 if work_z > 0.12 else 0.0)
        for dev in devs:
            dt = R2(dev) @ d   # tool push direction
            yaw = wrap(ang(dt) - ang(m.dir_q))
            yaw = yaw_choice(yaw, ref_yaw)
            if yaw is None: continue
            C0 = c - d * (m.standoff + extra_back)
            tcp0 = m.tcp_for(C0, yaw)
            C1 = T - d * (CUBE_H + m.r)
            tcp1 = m.tcp_for(C1, yaw)
            r0 = np.hypot(*(tcp0 - BASE_XY)); r1 = np.hypot(*(tcp1 - BASE_XY))
            if not (REACH_MIN < r0 < rmax and REACH_MIN < r1 < rmax):
                continue
            bad = False
            for f in np.linspace(0, 1, 6):
                if collides(m, tcp0 + f * (tcp1 - tcp0), yaw, work_z, grid, margin)[0]:
                    bad = True; break
            if bad: continue
            cost = abs(dev) * 1.0 + 0.25 * abs(yaw - ref_yaw) + 1.5 * phi
            if best is None or cost < best['cost']:
                best = dict(yaw=yaw, dev=dev, tcp0=tcp0, tcp1=tcp1, C0=C0, d=d, cost=cost, model=m,
                            work_z=work_z, phi=phi)
    if best is None:
        log('[plan] no feasible push plan')
    return best


def pitch_height(m, tcp_xy, yaw, grid, work_z):
    """TCP height at which tipping the tool to its pitch stays 2 cm above obstacles."""
    z = work_z
    for frac in (0.5, 1.0):
        mm = m if frac == 1.0 else m.pitched_base.pitched(m.phi * frac) if hasattr(m, 'pitched_base') else m
        xy, bot = mm.bottoms_world(np.r_[tcp_xy, 0.0], yaw)
        h = grid.heights(xy, 0.03)
        z = max(z, float(np.max(h + 0.02 - bot)))
    return z


def sweep_clear(model, tcp, yaw_from, yaw_to, z, grid):
    """Max obstacle height along a yaw sweep at TCP height z (tool points)."""
    n = max(2, int(abs(yaw_to - yaw_from) / math.radians(10)) + 1)
    h = -1
    for y in np.linspace(yaw_from, yaw_to, n):
        xy, _ = model.pts_world(np.r_[tcp[:2], z], y)
        h = max(h, grid.max_height_near(xy, 0.03))
    return h


def path_clear(model, a, b, yaw, z, grid):
    h = -1
    n = max(2, int(np.linalg.norm(np.asarray(b[:2]) - a[:2]) / 0.03) + 1)
    for f in np.linspace(0, 1, n):
        p = np.asarray(a[:2]) + f * (np.asarray(b[:2]) - a[:2])
        xy, _ = model.pts_world(np.r_[p, z], yaw)
        h = max(h, grid.max_height_near(xy, 0.03))
    return h


# ----------------------------------------------------------------------------- perception during task
def look(robot):
    robot.observe()
    return Scene(robot.meta, robot.imgs, log=robot.log)


def cube_now(robot):
    sc = look(robot)
    return sc.cube, sc


# ----------------------------------------------------------------------------- grasps
def descend_grasp(robot, xy, z, yaw_u, z_above=0.12, close_n=14, label=''):
    R = robot
    # rotate the empty gripper above the current position at a safe height
    if R.pos[2] < z_above - 0.01:
        R.goto(np.r_[R.pos[:2], z_above], speed=0.8)
    R.grip = -1.0
    if abs(R.yaw_u - yaw_u) > 0.02:
        rotate_yaw(R, yaw_u, rate=0.15, label='empty')
    R.goto(np.r_[xy, z_above], speed=1.0, tol=0.006)
    R.set_gripper(-1.0, n=4)
    R.goto(np.r_[xy, z + 0.03], speed=0.6, tol=0.004)
    R.goto(np.r_[xy, z], speed=0.15, tol=0.002, max_iter=8)
    R.set_gripper(1.0, n=close_n)
    op = R.gripper_opening()
    R.log(f'[grasp {label}] tcp={np.round(R.pos,4)} yaw={math.degrees(R.yaw_u):.1f} opening={op:.4f}')
    return op


def wait_success(robot, n=60):
    R = robot
    for _ in range(max(1, n // 10)):
        R.hold(10)
    return R.task_complete


# ----------------------------------------------------------------------------- generic tool push
def run_tool_push(robot, model, scene, grasp_z, kind, log=print, budget_reserve=40):
    """The tool is held (flat model). Plan, transit, lower and push the cube to the target.
    Returns (model_used, work_z) on success."""
    R = robot
    T = scene.target
    grid = scene.grid
    pitches = (0.0, 0.45, 0.7, 0.9) if kind == 'stick' else (0.0,)
    tool_lift = 0.07  # tool bottom ~7 cm above the table in transit
    repos = 0
    cube = scene.cube
    while repos < 5:
        repos += 1
        if R.remaining < budget_reserve: raise Fail('budget')
        plan = None
        for kw in (dict(), dict(extra_back=0.03), dict(margin=0.012)):
            plan = plan_push(model, cube, T, grid, R.yaw_u, grasp_z, log=log, pitches=pitches, **kw)
            if plan is not None: break
        if plan is None:
            g0 = ObstacleGrid(np.zeros((0, 3)))
            plan = plan_push(model, cube, T, g0, R.yaw_u, grasp_z, log=log)
            if plan is None: raise Fail('no plan')
            log('[plan] using collision-relaxed plan')
        m, work_z = plan['model'], plan['work_z']
        log(f"[plan] yaw={math.degrees(plan['yaw']):.1f} dev={math.degrees(plan['dev']):.1f} phi={math.degrees(plan['phi']):.0f} "
            f"tcp0={np.round(plan['tcp0'],3)} work_z={work_z:.3f} d={np.round(plan['d'],3)}")
        # transit (flat tool)
        z_low = grasp_z + tool_lift
        z_high = max(z_low, 0.27 + grasp_z)
        p = R.pos.copy()
        h1 = sweep_clear(model, p, R.yaw_u, plan['yaw'], z_low, grid)
        h2 = path_clear(model, p, plan['tcp0'], plan['yaw'], z_low, grid)
        zt = z_low if max(h1, h2) < z_low - grasp_z - 0.01 else z_high
        zt = max(zt, p[2]) if R.pitch != 0 else zt
        log(f'[transit] z={zt:.3f} (sweep h {h1:.3f}, path h {h2:.3f})')
        if R.pitch != 0.0:
            R.goto(np.r_[p[:2], max(zt, p[2] + 0.03)], speed=0.3, tol=0.006)
            R.pitch = 0.0
            R.goto(R.pos.copy(), speed=0.3, rspeed=0.12, tol=0.006)
        R.goto(np.r_[p[:2], zt], speed=0.4, tol=0.006)
        if abs(R.yaw_u - plan['yaw']) > 0.03:
            rate = 0.03 if model.extent > 0.2 else 0.06
            rotate_yaw(R, plan['yaw'], rate=rate, label='tool')
        R.goto(np.r_[plan['tcp0'], zt], speed=0.4, tol=0.006)
        if plan['phi'] != 0.0:
            zp = pitch_height(m, plan['tcp0'], R.yaw_u, grid, work_z)
            log(f'[pitch] tipping tool to {math.degrees(plan["phi"]):.0f} deg at z={max(zp, zt):.3f}')
            if zp > R.pos[2]:
                R.goto(np.r_[plan['tcp0'], zp], speed=0.3, tol=0.006)
            R.pitch = m.pitch
            R.goto(R.pos.copy(), speed=0.3, rspeed=0.12, tol=0.006, max_iter=8)
            log(f'[pitch] done, rot err={np.linalg.norm(R.rot_err(R_goal(R.yaw_u, R.pitch))):.3f}')
        R.goto(np.r_[plan['tcp0'], work_z + 0.015], speed=0.3, tol=0.004)
        R.goto(np.r_[plan['tcp0'], work_z], speed=0.1, tol=0.002, max_iter=6)
        res = push_loop(R, m, T, work_z, log)
        if res == 'done':
            return m, work_z
        if res == 'abort':
            raise Fail('push aborted')
        cube, sc = cube_now(R)
        if cube is None: raise Fail('cube lost')
        R.goto(np.r_[R.pos[:2], R.pos[2] + 0.05], speed=0.3, tol=0.006)
    raise Fail('too many repositions')


def push_loop(R, model, T, work_z, log, max_iters=45, done_tol=0.009):
    T = np.asarray(T[:2])
    R._model = model
    last_dist = None; no_prog = 0
    for it in range(max_iters):
        cube, sc = cube_now(R)
        if cube is None:
            log('[push] cube not visible'); return 'repos'
        c = cube[:2]; e = T - c; dist = float(np.linalg.norm(e))
        C = model.contact_world(R.pos, R.yaw_u)
        d = unit(e)
        rel = C - c; along = float(rel @ d); lat = float(rel @ perp(d))
        tdir = model.dir_world(R.yaw_u)
        dev = math.degrees(abs(wrap(ang(tdir) - ang(d))))
        log(f'[push {it}] dist={dist:.4f} along={along:.4f} lat={lat:.4f} dev={dev:.1f} tcp={np.round(R.pos,3)}')
        if dist < done_tol:
            return 'done'
        if R.remaining < 30:
            return 'abort'
        contact_along = -(CUBE_H + model.r)
        if model.kind == 'hook':
            # hook pulls: the crossbar is beyond the cube; along must be <= -(CUBE_H) + small
            pass
        if abs(lat) > model.lat_tol + 0.006 or along > contact_along + 0.012 or dev > math.degrees(model.max_dev) + 10:
            log('[push] misaligned -> reposition'); return 'repos'
        if last_dist is not None and dist > last_dist - 0.002 and along > contact_along - 0.004:
            no_prog += 1
            if no_prog >= 4:
                log('[push] no progress -> reposition'); return 'repos'
        else:
            no_prog = 0
        last_dist = dist
        # next contact goal: behind the cube on the cube->target line, advanced by step
        stepd = min(0.025, dist + 0.002)
        goal_C = c + d * (contact_along + stepd)
        if along < contact_along - 0.01:
            goal_C = C + d * min(0.03, (contact_along - along) + stepd)  # approach first
            goal_C = goal_C + perp(d) * (-lat) * 0.5 + 0 * d
        tcp_goal = model.tcp_for(goal_C, R.yaw_u)
        R.goto(np.r_[tcp_goal, work_z], speed=0.25, tol=0.003, max_iter=4)
    return 'repos'


def retreat(R, model, work_z, d, log):
    """Back the tool away from the cube without disturbing it, then lift."""
    if model.kind == 'hook':
        R.goto(np.r_[R.pos[:2], work_z + 0.07], speed=0.2, tol=0.005)
    else:
        back = R.pos[:2] - np.asarray(d) * 0.03
        R.goto(np.r_[back, work_z], speed=0.15, tol=0.004, max_iter=4)
        R.goto(np.r_[back, work_z + 0.06], speed=0.2, tol=0.005)
