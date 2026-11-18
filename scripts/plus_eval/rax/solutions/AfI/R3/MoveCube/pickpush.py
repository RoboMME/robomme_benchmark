"""Pick-and-place and push manners."""
import math
import numpy as np
from motion import Robot, wrap, yaw_of
from perception import parse_reply, cube_estimate, world_points, seg_dist

BASE_XY = np.array([-0.615, 0.0])


def cube_yaw(o, cube):
    """Cube edge orientation (rad, mod 90deg) from top-face points of the best camera."""
    best = None
    for cam in ("wrist", "front"):
        P = world_points(o, cam, "red", zmax=0.3)
        if len(P) < 10:
            continue
        top = P[P[:, 2] > P[:, 2].max() - 0.006][:, :2]
        if best is None or len(top) > len(best):
            best = top
    if best is None or len(best) < 8:
        return 0.0
    X = best - best.mean(0)
    bestv = None
    for a in np.radians(np.arange(0, 90, 2)):
        R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        Y = X @ R.T
        area = np.ptp(Y[:, 0]) * np.ptp(Y[:, 1])
        if bestv is None or area < bestv[0]:
            bestv = (area, a)
    return bestv[1]


def grasp_yaw_for(edge_angle):
    """Yaw so the finger axis (-sin, cos) is along a cube face normal; result in [-45, 45] deg."""
    psi = edge_angle - math.pi / 2
    psi = (psi + math.pi / 4) % (math.pi / 2) - math.pi / 4
    return psi


def run_pick(rb, o0, cube, target, obst, log=print):
    yaw = grasp_yaw_for(cube_yaw(o0, cube))
    ctop = float(cube[2])
    zg = max(0.012, ctop - 0.02)
    tall = 0.0
    if len(obst):
        near = obst[seg_dist(obst[:, :2], cube[:2], target[:2]) < 0.08]
        tall = float(near[:, 2].max()) if len(near) else 0.0
    zt = max(0.15, tall + 0.10)
    log(f"pick: yaw {math.degrees(yaw):.1f} zg {zg:.3f} transport z {zt:.2f}")
    rb.move_to([cube[0], cube[1], 0.12], yaw=yaw, grip=-1, tol=0.005, ytol=0.03, max_steps=90, chunk=2)
    # refine cube position from a close top view
    rb.hold(3)
    o = parse_reply(rb.observe())
    c = cube_estimate(o, cams=("wrist",))
    if c is not None and np.linalg.norm(c[:2] - cube[:2]) < 0.03:
        cube = c
        yaw = grasp_yaw_for(cube_yaw(o, cube))
        log(f"refined cube {np.round(cube,3)} yaw {math.degrees(yaw):.1f}")
    rb.move_to([cube[0], cube[1], 0.08], yaw=yaw, tol=0.004, ytol=0.02, max_steps=40, vmax=0.4)
    rb.move_to([cube[0], cube[1], zg], yaw=yaw, tol=0.003, ytol=0.02, max_steps=60, vmax=0.2)
    rb.set_gripper(1.0, n=15)
    log(f"pick width {rb.width:.4f}")
    if rb.width < 0.01:
        log("pick: nothing grasped")
        return False
    rb.move_to([cube[0], cube[1], zt], yaw=yaw, tol=0.008, ytol=0.04, max_steps=60, vmax=0.3)
    rb.move_to([target[0], target[1], zt], yaw=yaw, tol=0.005, ytol=0.04, max_steps=90, vmax=0.35)
    rb.move_to([target[0], target[1], 0.08], yaw=yaw, tol=0.004, ytol=0.03, max_steps=60, vmax=0.3)
    rb.move_to([target[0], target[1], zg + 0.006], yaw=yaw, tol=0.003, ytol=0.03, max_steps=60, vmax=0.15)
    rb.set_gripper(-1.0, n=12)
    p = rb.pos.copy(); p[2] = 0.15
    rb.move_to(p, yaw=yaw, tol=0.01, ytol=0.05, max_steps=40, vmax=0.3)
    rb.hold(5)
    o = parse_reply(rb.observe())
    c = cube_estimate(o)
    log(f"pick done: cube {None if c is None else np.round(c,3)} target {np.round(target,3)}")
    return True


def run_push(rb, o0, cube, target, obst, closed=True, log=print):
    g = 1.0 if closed else -1.0
    tgt = np.asarray(target[:2])
    c = np.asarray(cube[:2])
    u = tgt - c; u /= max(np.linalg.norm(u), 1e-6)
    yaw = wrap(math.atan2(u[1], u[0]))
    if abs(yaw) > math.radians(110):
        yaw = wrap(yaw + math.pi)
    zp = 0.018
    back = 0.02 + 0.012
    pre = c - u * (back + 0.03)
    log(f"push: yaw {math.degrees(yaw):.1f} pre {np.round(pre,3)} closed={closed}")
    rb.move_to([pre[0], pre[1], 0.12], yaw=yaw, grip=g, tol=0.006, ytol=0.04, max_steps=90, chunk=2)
    rb.set_gripper(g, n=10)
    rb.move_to([pre[0], pre[1], zp], yaw=yaw, tol=0.003, ytol=0.03, max_steps=60, vmax=0.25)
    for it in range(60):
        o = parse_reply(rb.observe())
        cc = cube_estimate(o)
        if cc is not None:
            c = cc[:2]
        e = tgt - c
        dist = float(np.linalg.norm(e))
        log(f"  push it{it} cube {np.round(c,3)} dist {dist:.3f} ee {np.round(rb.pos,3)}")
        if dist < 0.006:
            break
        u = e / dist
        des = c - u * back            # pusher contact position
        ee = rb.pos[:2]
        lat = (ee - des) - ((ee - des) @ u) * u
        behind = float((c - ee) @ u)
        if np.linalg.norm(lat) > 0.015 or behind < 0.015:
            # reposition: lift, go behind, come down
            p = rb.pos.copy(); p[2] = 0.08
            rb.move_to(p, yaw=yaw, tol=0.006, ytol=0.05, max_steps=30, vmax=0.3)
            pre = c - u * (back + 0.03)
            rb.move_to([pre[0], pre[1], 0.08], yaw=yaw, tol=0.005, ytol=0.05, max_steps=40, vmax=0.3)
            rb.move_to([pre[0], pre[1], zp], yaw=yaw, tol=0.003, ytol=0.05, max_steps=40, vmax=0.2)
            continue
        adv = min(0.03, dist)
        goal = ee + u * adv
        goal = goal - lat * 0.5
        for _ in range(int(adv / 0.003) + 1):
            row = rb.row_towards([goal[0], goal[1], zp], yaw=yaw, vmax=0.07, wmax=0.3)
            rb.step([row])
    p = rb.pos.copy()
    p[:2] = p[:2] - u * 0.03
    rb.move_to([p[0], p[1], zp], yaw=yaw, tol=0.004, ytol=0.05, max_steps=20, vmax=0.1)
    p[2] = 0.12
    rb.move_to(p, yaw=yaw, tol=0.01, ytol=0.05, max_steps=40, vmax=0.3)
    rb.hold(5)
    return True
