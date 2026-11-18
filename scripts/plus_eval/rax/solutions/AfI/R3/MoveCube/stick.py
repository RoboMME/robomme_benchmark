"""Tool push with a symmetric (ball-tipped) stick: reproduce the demo's head/cube geometry and sweep."""
import math
import numpy as np
from motion import wrap, yaw_of, Ended
from perception import parse_reply, cube_estimate, components, seg_dist
from hook import rot2, ang, yaw_options, choose_rot_spot, BASE_XY

PRE_BACK = 0.045
SPEED = 0.003


def rel_yaw_cost(yaw, ee):
    """Penalty for gripper yaw far from the arm's azimuth (ep2 arm jump at rel -105 deg near the base)."""
    az = math.atan2(ee[1] - BASE_XY[1], ee[0] - BASE_XY[0])
    rel = abs(math.degrees(wrap(yaw - az)))
    return max(0.0, rel - 95.0) * 1.5


def reach_cost(e):
    # ep2: a low pose at r=0.36 (rel yaw -105) made the arm jump; at r=0.798 the arm froze and the env
    # burned 100 steps per row.  Keep the whole sweep (incl. overshoot) inside r ~0.44..0.72.
    rr = float(np.linalg.norm(np.asarray(e) - BASE_XY))
    if rr < 0.34 or rr > 0.775:
        return 100.0
    if rr < 0.44:
        return (0.44 - rr) * 300
    if rr > 0.72:
        return (rr - 0.72) * 400
    return 0.0


def plan_stick(tool, cube, target, rel_u, phi_rel, obst, log=print):
    grasp0 = np.asarray(tool["handle"][:2]); head0 = np.asarray(tool["head"][:2])
    v0 = head0 - grasp0
    phi0 = ang(v0)
    u = target[:2] - cube[:2]; dist = float(np.linalg.norm(u)); u = u / max(dist, 1e-6)
    best = None
    for mirror in (1, -1):
      for dev_deg in [0, 10, -10, 20, -20, 30, -30, 40, -40, 50, -50, 60, -60]:
        rel = np.array([rel_u[0], mirror * rel_u[1]])
        a_u = ang(u)
        head_contact = cube[:2] + rot2(rel, a_u)
        head_pre = head_contact - u * PRE_BACK
        want = wrap(a_u + mirror * phi_rel + math.radians(dev_deg))
        for _, psi_g, delta in yaw_options(phi0, want)[:3]:
            off = rot2(v0, delta)
            ee_pre = head_pre - off
            travel = PRE_BACK + dist
            ee_end = ee_pre + u * travel
            cost = abs(dev_deg) * 0.12 + abs(delta) * 0.5
            for e in (ee_pre, ee_end + u * 0.03, (ee_pre + ee_end) / 2):
                cost += reach_cost(e)
            yaw_f = psi_g + delta
            cost += max(rel_yaw_cost(yaw_f, ee_pre), rel_yaw_cost(yaw_f, ee_end))
            # tool must not land on the cube
            dmin = float(seg_dist(cube[None, :2], ee_pre, head_pre)[0])
            if dmin < 0.035:
                cost += 100
            # during the sweep the cube sits at the head; the shaft must not pass over it
            w = ee_pre - head_pre  # head -> EE along the shaft
            dsw = float(seg_dist(cube[None, :2], head_contact + w * 0.12, head_contact + w)[0])
            if dsw < 0.034:
                cost += (0.034 - dsw) * 3000
            ncol_ee = ncol_tool = 0
            if len(obst):
                low = obst[obst[:, 2] > 0.012]
                ncol_ee = int((seg_dist(low[:, :2], ee_pre, ee_end) < 0.065).sum())
                lt = obst  # solid below any observed surface: tall points block the stick too
                hit = np.zeros(len(lt), bool)
                for t in np.linspace(0, 1, 9):
                    e = ee_pre + t * (ee_end - ee_pre)
                    hit |= seg_dist(lt[:, :2], e, e + off) < 0.025
                ncol_tool = int(hit.sum())
            cost += 2.0 * min(ncol_ee, 50) + 2.0 * min(ncol_tool, 50)
            log(f"  stick cand m={mirror} dev={dev_deg} psi_g={math.degrees(psi_g):.0f} delta={math.degrees(delta):.0f} "
                f"ee_pre={np.round(ee_pre,3)} ee_end={np.round(ee_end,3)} cost={cost:.1f} col={ncol_ee}/{ncol_tool} dmin={dmin:.3f} dsw={dsw:.3f}")
            cand = dict(mirror=mirror, dev=dev_deg, theta=0, psi_g=psi_g, delta=delta, ee_pre=ee_pre, ee_end=ee_end, u=u,
                        dist=dist, travel=travel, off=off, cost=cost, head_pre=head_pre)
            if best is None or cand["cost"] < best["cost"]:
                best = cand
    return best


def run_stick(rb, tool, cube, target, rel_u, phi_rel, obst, log=print):
    grasp_xy = np.asarray(tool["handle"][:2])
    v0 = np.asarray(tool["head"][:2]) - grasp_xy
    # drop the tool's own points from obstacles
    if len(obst):
        obst = obst[seg_dist(obst[:, :2], grasp_xy, grasp_xy + v0 * 1.1) > 0.035]
    plan = plan_stick(tool, cube, target, rel_u, phi_rel, obst, log)
    if plan is None:
        log("stick: no plan")
        return False
    log(f"stick plan: m={plan['mirror']} dev={plan['dev']} th={plan['theta']} psi_g={math.degrees(plan['psi_g']):.1f} "
        f"delta={math.degrees(plan['delta']):.1f} ee_pre={np.round(plan['ee_pre'],3)} ee_end={np.round(plan['ee_end'],3)} cost={plan['cost']:.1f}")
    psi_g, delta = plan["psi_g"], plan["delta"]
    zg = max(0.012, min(0.02, float(tool["handle"][2]) - 0.009))
    rb.move_to([grasp_xy[0], grasp_xy[1], 0.12], yaw=psi_g, grip=-1, tol=0.006, ytol=0.04, max_steps=90, chunk=2)
    rb.move_to([grasp_xy[0], grasp_xy[1], 0.06], yaw=psi_g, tol=0.004, ytol=0.03, max_steps=40, vmax=0.4)
    rb.move_to([grasp_xy[0], grasp_xy[1], zg], yaw=psi_g, tol=0.003, ytol=0.03, max_steps=60, vmax=0.25)
    rb.set_gripper(1.0, n=15)
    log(f"stick grasp width {rb.width:.4f} at {np.round(rb.pos,3)}")
    if rb.width < 0.008 or rb.width > 0.06:
        log("stick grasp failed (width)")
        return False
    grasp_pos = rb.pos.copy(); grasp_yaw = yaw_of(rb.rotvec)
    spot = choose_rot_spot([grasp_xy, plan["ee_pre"], (grasp_xy + plan["ee_pre"]) / 2, np.array([-0.30, 0.0]),
                            np.array([-0.30, 0.15]), np.array([-0.30, -0.15])], obst, radius=0.40)
    spot_xy, hmax = spot if spot else (grasp_xy, 0.0)
    z_rot = 0.20 if hmax < 0.14 else min(0.30, hmax + 0.10)
    log(f"stick rotation spot {np.round(spot_xy,3)} hmax={hmax:.3f} z_rot={z_rot:.2f}")
    rb.move_to([grasp_pos[0], grasp_pos[1], z_rot], yaw=grasp_yaw, tol=0.01, ytol=0.05, max_steps=80, vmax=0.35)
    if abs(delta) > math.radians(20):
        rb.move_to([spot_xy[0], spot_xy[1], z_rot], yaw=grasp_yaw, tol=0.01, ytol=0.05, max_steps=80, vmax=0.35)
    n = int(abs(delta) / math.radians(15)) + 1
    for i in range(1, n + 1):
        rb.move_to(list(rb.pos[:2]) + [z_rot], yaw=wrap(grasp_yaw + delta * i / n), tol=0.02, ytol=0.07,
                   max_steps=20, vmax=0.3, wmax=0.3)
    final_yaw = wrap(grasp_yaw + delta)
    ee_pre = plan["ee_pre"]
    rb.move_to([ee_pre[0], ee_pre[1], z_rot], yaw=final_yaw, tol=0.006, ytol=0.02, max_steps=90, vmax=0.35)
    rb.move_to([ee_pre[0], ee_pre[1], zg + 0.05], yaw=final_yaw, tol=0.004, ytol=0.02, max_steps=60, vmax=0.25)
    rb.hold(4)
    dyaw = wrap(yaw_of(rb.rotvec) - grasp_yaw)
    o = parse_reply(rb.observe())
    corr = np.zeros(2)
    pred = rb.pos[:2] + rot2(v0, dyaw)
    pk = [p for p in components(o, "front", "pink", 4) if abs(p["diag"] - tool.get("head_diag", 0.045)) < 0.03]
    if pk:
        pk.sort(key=lambda p: np.linalg.norm(p["c"][:2] - pred))
        obs_h = pk[0]["P"][:, :2].mean(0)
        e = obs_h - pred
        log(f"stick head check: observed {np.round(obs_h,3)} predicted {np.round(pred,3)} err {np.round(e,3)}")
        if 0.012 < np.linalg.norm(e) < 0.04:
            corr = e
    ee_pre = ee_pre - corr
    rb.move_to([ee_pre[0], ee_pre[1], zg + 0.05], yaw=final_yaw, tol=0.003, ytol=0.02, max_steps=30, vmax=0.2)
    z_drag = zg + 0.004
    lowered = False
    for attempt in range(2):
        rb.move_to([ee_pre[0], ee_pre[1], z_drag], yaw=final_yaw, tol=0.003, ytol=0.02, max_steps=60, vmax=0.12)
        exy = float(np.linalg.norm(rb.pos[:2] - ee_pre)); ez = float(rb.pos[2] - z_drag)
        log(f"stick lowered at {np.round(rb.pos,3)} xy err {exy:.3f} z err {ez:.3f} (attempt {attempt})")
        if exy < 0.015 and ez < 0.012:
            lowered = True
            break
        rb.move_to([rb.pos[0], rb.pos[1], zg + 0.08], yaw=final_yaw, tol=0.006, ytol=0.03, max_steps=40, vmax=0.2)
        rb.move_to([ee_pre[0], ee_pre[1], zg + 0.08], yaw=final_yaw, tol=0.004, ytol=0.02, max_steps=60, vmax=0.25)
    if not lowered:
        log("stick: could not lower to sweep height")
        return False
    u0 = plan["u"].copy(); u = u0.copy()
    tgt = target[:2]
    travelled = 0.0; last = 0.0
    p = rb.pos.copy(); cube_start = cube[:2].copy()
    while travelled < plan["travel"] + 0.04:
        travelled += SPEED
        p[:2] = p[:2] + u * SPEED; p[2] = z_drag
        if np.linalg.norm(p[:2] - BASE_XY) > 0.785:
            log(f"  sweep: next EE target {np.round(p[:2],3)} beyond safe reach; stopping")
            break
        rb.step([rb.row_towards(p, grip=1.0, vmax=0.3, wmax=0.3, yaw=final_yaw)])
        if np.linalg.norm(rb.pos[:2] - p[:2]) > 0.03 or rb.pos[2] > z_drag + 0.02:
            log(f"  sweep: EE off track {np.round(rb.pos,3)} vs {np.round(p,3)}; stopping")
            break
        if travelled - last > 0.02:
            last = travelled
            o = parse_reply(rb.observe())
            c = cube_estimate(o, cams=("front",))
            if c is None:
                c = cube_estimate(o)
            if c is None:
                continue
            cn = c[:2]; e = tgt - cn
            along = float(e @ u0); moved = float(np.linalg.norm(cn - cube_start))
            log(f"  sweep {travelled:.3f}/{plan['travel']:.3f} ee {np.round(rb.pos,3)} cube {np.round(cn,3)} to-go {along:.3f} |e| {np.linalg.norm(e):.3f} moved {moved:.3f}")
            if along < 0.004 or np.linalg.norm(e) < 0.004:
                break
            if travelled > PRE_BACK + 0.06 and moved < 0.01:
                log("  cube not moving with stick")
                break
            if moved > 0.01 and np.linalg.norm(e) > 0.015:
                un = e / np.linalg.norm(e)
                a = math.atan2(u0[0] * un[1] - u0[1] * un[0], float(u0 @ un))
                a = max(-math.radians(20), min(math.radians(20), a))
                u = rot2(u0, a)
    rb.hold(5)
    o = parse_reply(rb.observe())
    c = cube_estimate(o)
    log(f"after sweep cube {None if c is None else np.round(c,3)} target {np.round(tgt,3)}")
    q = rb.pos.copy(); q[:2] -= u * 0.01; q[2] += 0.03
    rb.move_to(q, yaw=final_yaw, tol=0.004, ytol=0.03, max_steps=30, vmax=0.1)
    rb.hold(10)
    return True
