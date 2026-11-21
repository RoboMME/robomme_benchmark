import numpy as np
from motion import hand_yaw, wrap
from percept import find_cubes, rect_yaw

def locate_cube_wrist(robot, xy_guess, max_dist=0.05):
    robot.observe()
    cs = find_cubes(robot.meta, robot.imgs, cam="wrist")
    best = None
    for c in cs:
        dd = np.linalg.norm(c["xyz"][:2] - xy_guess)
        if dd < max_dist and (best is None or dd < best[0]):
            best = (dd, c)
    if best is None:
        return None
    c = best[1]
    yaw, ext = rect_yaw(c["top"][:, :2])
    return dict(xyz=c["xyz"], yaw=yaw, ext=ext)

def conflicts(xy, g, neighbors):
    a = np.array([np.cos(g), np.sin(g)]); b = np.array([-a[1], a[0]])
    n = 0
    for p in neighbors:
        d = np.asarray(p) - xy
        if abs(d @ b) < 0.04 and abs(d @ a) < 0.085:
            n += 1
    return n

def pick_cube(robot, get_xy, log=print, hover=0.15, lift=0.15, place_back=True, neighbors=()):
    """Approach tracked cube, refine with wrist cam, grasp, lift, (place back), release."""
    robot.grip = -1.0
    xy = get_xy()
    robot.move_to([xy[0], xy[1], hover], chunk=8)
    info = None
    for k in range(3):
        xy = get_xy()
        info = locate_cube_wrist(robot, np.asarray(xy[:2]), max_dist=0.025)
        if info is None:
            log("wrist locate failed", k); break
        log("wrist cube", np.round(info["xyz"], 3), "yaw", np.degrees(info["yaw"]))
        xy = info["xyz"][:2]
        if np.linalg.norm(robot.state[:2] - xy) < 0.006:
            break
        robot.move_to([xy[0], xy[1], hover])
    cube_yaw = info["yaw"] if info else 0.0
    cur = hand_yaw(robot.state)
    goal = cur + wrap(cube_yaw - cur, np.pi / 2)
    alt = goal + (np.pi / 2 if goal < cur else -np.pi / 2)
    c0, c1 = conflicts(xy, goal, neighbors), conflicts(xy, alt, neighbors)
    if c0 > 0 and c1 < c0:
        log("using alternate grasp axis", c0, c1)
        goal = alt
    robot.set_down(goal)
    robot.move_to([xy[0], xy[1], hover])
    ztop = info["xyz"][2] if info else 0.04
    robot.move_to([xy[0], xy[1], ztop + 0.03], vmax=0.6)
    robot.move_to([xy[0], xy[1], max(ztop - 0.022, 0.012)], vmax=0.3, max_steps=25)
    robot.hold(10, grip=1.0)
    robot.move_to([xy[0], xy[1], lift], vmax=0.6)
    robot.hold(3)
    opening = float(robot.state[6] + robot.state[7])
    held = opening > 0.025
    log("lifted, opening %.4f held %s" % (opening, held))
    if place_back:
        robot.move_to([xy[0], xy[1], ztop - 0.015], vmax=0.5, max_steps=30)
        robot.hold(6, grip=-1.0)
        robot.move_to([xy[0], xy[1], hover])
    return held

def press_button(robot, bxy, btop, log=print, depth=0.012):
    robot.grip = 1.0
    robot.move_to([bxy[0], bxy[1], btop + 0.07], chunk=8)
    robot.move_to([bxy[0], bxy[1], btop + 0.02], vmax=0.5)
    robot.move_to([bxy[0], bxy[1], btop - depth], vmax=0.25, max_steps=20)
    robot.hold(3)
    robot.move_to([bxy[0], bxy[1], btop + 0.07], vmax=0.6)
