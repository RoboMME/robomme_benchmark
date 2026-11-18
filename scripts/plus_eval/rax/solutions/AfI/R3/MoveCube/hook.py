"""Hook manner: grasp tool handle, place crossbar behind cube, pull cube slowly to target."""
import math
import numpy as np
from motion import Robot, wrap, yaw_of, Ended
from perception import parse_reply, scene, cube_estimate, target_estimate, obstacle_points, seg_dist, world_points

# Hook geometry relative to the pink head centroid H, in (s, c) coordinates
# (s: handle->head unit vector, c: crossbar side unit vector). Measured in ep1.
POCKET = (-0.030, -0.011)     # cube centre when hooked
JUNCTION = (0.000, -0.048)    # shaft / crossbar corner
CROSS_END = (0.012, 0.024)
TIP_END = (-0.023, 0.030)
YAW_LIMIT = math.radians(150)
BASE_XY = np.array([-0.615, 0.0])
DRAG_SPEED = 0.003            # m per step during the pull
PRE_BEHIND = 0.06             # pocket starts this far behind the cube


def rot2(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def ang(v):
    return math.atan2(v[1], v[0])


class HookModel:
    """Hook points as world vectors relative to the grasp point, at grasp yaw."""

    def __init__(self, h, grasp_xy):
        H = np.asarray(h["head"][:2]); s = np.asarray(h["s"]); c = np.asarray(h["c"])
        g = np.asarray(grasp_xy)
        f = lambda sc: H + sc[0] * s + sc[1] * c - g
        self.pocket = f(POCKET); self.junction = f(JUNCTION); self.cross_end = f(CROSS_END)
        self.tip_end = f(TIP_END); self.head = H - g
        self.corner = f((JUNCTION[0] - 0.024, JUNCTION[1] + 0.025))  # cube centre nestled at shaft corner
        self.s0 = s; self.c0 = c

    def world(self, ee_xy, dyaw):
        ee = np.asarray(ee_xy)[:2]
        r = lambda v: ee + rot2(v, dyaw)
        return dict(pocket=r(self.pocket), corner=r(self.corner), junction=r(self.junction), cross_end=r(self.cross_end),
                    tip_end=r(self.tip_end), head=r(self.head), handle=ee.copy())

    def segments(self, ee_xy, dyaw):
        w = self.world(ee_xy, dyaw)
        return [(w["handle"], w["junction"]), (w["junction"], w["cross_end"]), (w["cross_end"], w["tip_end"])]


def yaw_options(phi, want_shaft):
    """(psi_g, delta) options: grasp yaw and signed hook rotation reaching want_shaft."""
    opts = []
    for psi_g in (wrap(phi), wrap(phi - math.pi)):
        d0 = wrap(want_shaft - phi)
        for delta in (d0, d0 - 2 * math.pi, d0 + 2 * math.pi):
            psi_d = psi_g + delta
            if min(psi_g, psi_d) < -YAW_LIMIT or max(psi_g, psi_d) > YAW_LIMIT:
                continue
            opts.append((abs(delta) + 0.5 * abs(psi_d) + 0.3 * abs(psi_g), psi_g, delta))
    opts.sort()
    return opts


def plan_drag(model, phi, cube, target, obst, log=print):
    """Choose the shaft angle offset theta and yaws; returns plan dict or None."""
    u = target[:2] - cube[:2]
    dist = float(np.linalg.norm(u))
    u = u / max(dist, 1e-6)
    best = None
    for th_deg in [0, 6, -6, 12, -12, 18, -18, 24, -24]:
        th = math.radians(th_deg)
        want = wrap(ang(-u) + th)
        opts = yaw_options(phi, want)
        if not opts:
            continue
        _, psi_g, delta = opts[0]
        pre_pocket = cube[:2] - u * PRE_BEHIND
        dP = rot2(model.pocket, delta)
        dE = rot2(model.corner if th_deg >= 8 else model.pocket, delta)
        ee_pre = pre_pocket - dP
        ee_end = target[:2] - dE
        cost = abs(th_deg) * 0.2 + (3.0 if th_deg < 0 else 0.0)
        # cube clearance when lowering at the pre pose
        segs = model.segments(ee_pre, delta)
        dmin = min(seg_dist(cube[None, :2], a, b)[0] for a, b in segs)
        if dmin < 0.03:
            cost += 100
        # reach
        for e in (ee_pre, ee_end):
            rr = np.linalg.norm(e - BASE_XY)
            if rr < 0.28 or rr > 0.77:
                cost += 100
            elif rr < 0.38:
                cost += (0.38 - rr) * 300
            elif rr > 0.72:
                cost += (rr - 0.72) * 400
        ncol_ee = ncol_tool = 0
        if len(obst):
            low = obst[obst[:, 2] > 0.012]
            ncol_ee = int((seg_dist(low[:, :2], ee_pre, ee_end) < 0.065).sum())
            lowtool = obst  # tall obstacles are solid down to the table
            hit = np.zeros(len(lowtool), bool)
            for t in np.linspace(0, 1, 9):
                e = ee_pre + t * (ee_end - ee_pre)
                for a, b in model.segments(e, delta):
                    hit |= seg_dist(lowtool[:, :2], a, b) < 0.018
            ncol_tool = int(hit.sum())
        cost += 2.0 * min(ncol_ee, 50) + 2.0 * min(ncol_tool, 50)
        cand = dict(theta=th_deg, psi_g=psi_g, delta=delta, ee_pre=ee_pre, ee_end=ee_end, u=u, dist=dist,
                    cost=cost, ncol_ee=ncol_ee, ncol_tool=ncol_tool, dmin=dmin)
        log(f"  drag cand th={th_deg} psi_g={math.degrees(psi_g):.0f} delta={math.degrees(delta):.0f} "
            f"cost={cost:.1f} ee_col={ncol_ee} tool_col={ncol_tool} dmin={dmin:.3f}")
        if best is None or cost < best["cost"]:
            best = cand
    return best


def choose_rot_spot(cands, obst, radius=0.36):
    best = None
    for p in cands:
        p = np.asarray(p)
        if np.linalg.norm(p - BASE_XY) < 0.25 or np.linalg.norm(p - BASE_XY) > 0.72:
            continue
        if len(obst):
            near = obst[np.linalg.norm(obst[:, :2] - p, axis=1) < radius]
            hmax = float(near[:, 2].max()) if len(near) else 0.0
        else:
            hmax = 0.0
        if best is None or hmax < best[1] - 0.01:
            best = (p, hmax)
    return best


def run_hook(rb, sc0, cube, target, obst, log=print):
    h = sc0.get("hook")
    if h is None:
        log("hook not found")
        return False
    grasp_xy = np.asarray(h["handle"][:2])
    handle_top = float(h["handle"][2])
    model = HookModel(h, grasp_xy)
    phi = ang(h["s"])
    # drop hook's own points from obstacles
    if len(obst):
        segs = model.segments(grasp_xy, 0.0)
        keep = np.ones(len(obst), bool)
        for a, b in segs:
            keep &= seg_dist(obst[:, :2], a, b) > 0.03
        obst = obst[keep]
    plan = plan_drag(model, phi, cube, target, obst, log)
    if plan is None:
        log("no drag plan")
        return False
    log(f"plan: th={plan['theta']} psi_g={math.degrees(plan['psi_g']):.1f} delta={math.degrees(plan['delta']):.1f} "
        f"ee_pre={np.round(plan['ee_pre'],3)} ee_end={np.round(plan['ee_end'],3)} dist={plan['dist']:.3f}")
    psi_g, delta = plan["psi_g"], plan["delta"]
    zg = max(0.012, min(0.02, handle_top - 0.009))
    # ---- grasp
    rb.move_to([grasp_xy[0], grasp_xy[1], 0.12], yaw=psi_g, grip=-1, tol=0.006, ytol=0.04, max_steps=90, chunk=2)
    rb.move_to([grasp_xy[0], grasp_xy[1], 0.06], yaw=psi_g, tol=0.004, ytol=0.03, max_steps=40, vmax=0.4)
    rb.move_to([grasp_xy[0], grasp_xy[1], zg], yaw=psi_g, tol=0.003, ytol=0.03, max_steps=60, vmax=0.25)
    rb.set_gripper(1.0, n=15)
    log(f"grasp width {rb.width:.4f} at {np.round(rb.pos,3)}")
    if rb.width < 0.012 or rb.width > 0.06:
        log("grasp failed (width)")
        return False
    grasp_pos = rb.pos.copy()
    grasp_yaw = yaw_of(rb.rotvec)
    # ---- lift and rotate
    tall = obst[obst[:, 2] > 0.0] if len(obst) else obst
    spot = choose_rot_spot([grasp_xy, plan["ee_pre"], (grasp_xy + plan["ee_pre"]) / 2,
                            np.array([-0.30, 0.0]), np.array([-0.30, 0.15]), np.array([-0.30, -0.15]),
                            np.array([-0.25, 0.25]), np.array([-0.25, -0.25])], tall)
    spot_xy, hmax = spot if spot else (grasp_xy, 0.0)
    z_rot = 0.20 if hmax < 0.14 else min(0.30, hmax + 0.10)
    log(f"rotation spot {np.round(spot_xy,3)} hmax={hmax:.3f} z_rot={z_rot:.2f}")
    rb.move_to([grasp_pos[0], grasp_pos[1], z_rot], yaw=grasp_yaw, tol=0.01, ytol=0.05, max_steps=80, vmax=0.35)
    rb.move_to([spot_xy[0], spot_xy[1], z_rot], yaw=grasp_yaw, tol=0.01, ytol=0.05, max_steps=80, vmax=0.35)
    n = int(abs(delta) / math.radians(15)) + 1
    for i in range(1, n + 1):
        rb.move_to([spot_xy[0], spot_xy[1], z_rot], yaw=wrap(grasp_yaw + delta * i / n), tol=0.015, ytol=0.07,
                   max_steps=20, vmax=0.3, wmax=0.3)
    final_yaw = wrap(grasp_yaw + delta)
    rb.move_to([spot_xy[0], spot_xy[1], z_rot], yaw=final_yaw, tol=0.006, ytol=0.02, max_steps=30, wmax=0.25)
    log(f"rotated: yaw {math.degrees(yaw_of(rb.rotvec)):.1f} width {rb.width:.4f}")
    # ---- move above pre pose, descend part way, verify head
    def dyaw_now():
        return wrap(yaw_of(rb.rotvec) - grasp_yaw)
    corr = np.zeros(2)
    ee_pre = plan["ee_pre"]
    rb.move_to([ee_pre[0], ee_pre[1], z_rot], yaw=final_yaw, tol=0.006, ytol=0.03, max_steps=80, vmax=0.35)
    rb.move_to([ee_pre[0], ee_pre[1], zg + 0.05], yaw=final_yaw, tol=0.004, ytol=0.02, max_steps=60, vmax=0.25)
    rb.hold(4)
    o = parse_reply(rb.observe())
    P = world_points(o, "front", "pink")
    if len(P) >= 25:
        pred = model.world(rb.pos[:2], dyaw_now())["head"]
        obs_h = P[:, :2].mean(0)
        e = obs_h - pred
        log(f"head check: observed {np.round(obs_h,3)} predicted {np.round(pred,3)} err {np.round(e,3)}")
        if 0.012 < np.linalg.norm(e) < 0.04:
            corr = e
    else:
        log(f"head check skipped (pink px {len(P)})")
    # re-estimate cube from this view (cube should still be where it was)
    c_now = cube_estimate(o)
    if c_now is not None and np.linalg.norm(c_now[:2] - cube[:2]) < 0.03:
        log(f"cube re-est {np.round(c_now,3)} (init {np.round(cube,3)})")
    u = plan["u"]
    # pocket world position = ee + rot(model.pocket, dyaw) + corr
    def pocket_of(ee_xy):
        return model.world(ee_xy, dyaw_now())["pocket"] + corr
    ee_pre = ee_pre - corr
    rb.move_to([ee_pre[0], ee_pre[1], zg + 0.05], yaw=final_yaw, tol=0.003, ytol=0.02, max_steps=30, vmax=0.2)
    z_drag = zg + 0.004
    rb.move_to([ee_pre[0], ee_pre[1], z_drag], yaw=final_yaw, tol=0.003, ytol=0.02, max_steps=60, vmax=0.12)
    log(f"lowered at {np.round(rb.pos,3)} pocket {np.round(pocket_of(rb.pos[:2]),3)}")
    # ---- pull (slow, steered toward the target from cube observations)
    tgt = target[:2]
    u0 = u.copy()
    total = PRE_BEHIND + plan["dist"]
    travelled = 0.0
    last_check = 0.0
    p = rb.pos.copy()
    cube_now = cube[:2].copy()
    cube_start = cube[:2].copy()
    misses = 0
    while travelled < total + 0.04:
        travelled += DRAG_SPEED
        p[:2] = p[:2] + u * DRAG_SPEED
        p[2] = z_drag
        rr = np.linalg.norm(p[:2] - BASE_XY)
        if rr > 0.785 or rr < 0.27:
            log(f"  drag: next EE target {np.round(p[:2],3)} outside safe reach; stopping")
            break
        row = rb.row_towards(p, grip=1.0, vmax=0.3, wmax=0.3, yaw=final_yaw)
        rb.step([row])
        if travelled - last_check > 0.02:
            last_check = travelled
            o = parse_reply(rb.observe())
            c = cube_estimate(o, cams=("front",))
            if c is None:
                c = cube_estimate(o)
            if c is None:
                misses += 1
                continue
            cube_now = c[:2]
            e = tgt - cube_now
            along = float(e @ u0)
            moved = float(np.linalg.norm(cube_now - cube_start))
            log(f"  pull {travelled:.3f}/{total:.3f} ee {np.round(rb.pos,3)} cube {np.round(cube_now,3)} "
                f"to-go {along:.3f} |e| {np.linalg.norm(e):.3f} moved {moved:.3f}")
            if along < 0.004 or np.linalg.norm(e) < 0.004:
                break
            if travelled > PRE_BEHIND + 0.06 and moved < 0.01:
                log("  cube not caught")
                break
            if moved > 0.01 and np.linalg.norm(e) > 0.015:
                un = e / np.linalg.norm(e)
                a = math.atan2(u0[0] * un[1] - u0[1] * un[0], float(u0 @ un))
                a = max(-math.radians(20), min(math.radians(20), a))
                u = rot2(u0, a)
    rb.hold(5)
    o = parse_reply(rb.observe())
    c = cube_estimate(o)
    log(f"after pull cube {None if c is None else np.round(c,3)} target {np.round(tgt,3)}")
    # lift slightly as in the demo
    p = rb.pos.copy(); p[2] += 0.03
    rb.move_to(p, yaw=final_yaw, tol=0.004, ytol=0.03, max_steps=30, vmax=0.1)
    rb.hold(10)
    return True
