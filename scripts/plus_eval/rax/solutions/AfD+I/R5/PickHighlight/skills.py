import numpy as np, math
from percep import find_cube_near, find_button
from motion import wrap

FINGER_HALF = 0.045

def yaw_candidates(cur, cube_yaw):
    out = []
    for k in range(2):
        a = cube_yaw + k * math.pi / 2
        d = (wrap(a - cur) + math.pi / 2) % math.pi - math.pi / 2
        out.append(cur + d)
    return out

def finger_clearance(xy, yaw, obstacles):
    if not obstacles:
        return 1.0
    ax = np.array([math.cos(yaw), math.sin(yaw)])
    f = [np.asarray(xy) + FINGER_HALF * ax, np.asarray(xy) - FINGER_HALF * ax]
    return min(np.linalg.norm(p - np.asarray(o)[:2]) for p in f for o in obstacles)

def choose_yaw(rb, xy, cube_yaw, others, button_xy, log=print):
    best = None
    for y in yaw_candidates(rb.yaw(), cube_yaw):
        cb = finger_clearance(xy, y, [button_xy]) if button_xy is not None else 1.0
        cc = finger_clearance(xy, y, others)
        score = min(cb - 0.06, cc - 0.03)
        rot = abs(wrap(y - rb.yaw()))
        log("  yaw cand %.0f deg: clr_button %.3f clr_cubes %.3f" % (math.degrees(y), cb, cc))
        if best is None or score > best[0] + 1e-3 or (abs(score - best[0]) < 1e-3 and rot < best[2]):
            best = (score, y, rot)
    return best[1], best[0]

def press_button(rb, bxyz, log=print):
    bx, by, bz = bxyz
    rb.move_to([bx, by, bz + 0.08], grip=1.0, yaw=rb.yaw())
    rb.move_to([bx, by, bz + 0.03], vmax=0.3)
    rb.move_to([bx, by, bz - 0.02], vmax=0.2, max_iter=3, tol=0.006)
    log("press bottom z=%.3f xy=(%.3f,%.3f) step %s" % (rb.pos[2], rb.pos[0], rb.pos[1], rb.meta.get("step_id")))
    rb.hold(4)
    rb.move_to([bx, by, bz + 0.10], vmax=0.5, tol=0.01)

def reacquire(rb, track, rmax, log, tag):
    c = find_cube_near(rb.meta, rb.imgs, "wrist", track["xy"], rmax=rmax)
    if c is None:
        log("%s: no cube within %.3f of track %d at (%.3f,%.3f)" % (tag, rmax, track["id"], track["xy"][0], track["xy"][1]))
        return None
    d = float(np.linalg.norm(np.asarray(c["xy"]) - track["xy"]))
    log("%s: track %d (%.3f,%.3f) -> (%.3f,%.3f) d=%.3f ztop=%.3f yaw=%.0f" % (tag, track["id"], track["xy"][0], track["xy"][1], c["xy"][0], c["xy"][1], d, c["ztop"], math.degrees(c["yaw"])))
    track["xy"] = np.asarray(c["xy"], float)
    return c

def pick_cube(rb, tracker, tid, button_xy=None, log=print, lift_z=0.12):
    tr = tracker.get(tid)
    xy = tr["xy"].copy()
    rb.move_to([xy[0], xy[1], 0.18], grip=-1.0, tol=0.006)
    rb.hold(3)
    c = reacquire(rb, tr, 0.03, log, "acq1")
    if c is None:
        return False, None
    xy = tr["xy"].copy()
    others = [t["xy"] for t in tracker.tracks if t["id"] != tid]
    yaw, score = choose_yaw(rb, xy, c["yaw"], others, button_xy, log=log)
    log("  chosen yaw %.0f score %.3f" % (math.degrees(yaw), score))
    rb.move_to([xy[0], xy[1], 0.11], yaw=yaw, tol=0.004)
    rb.hold(2)
    c2 = reacquire(rb, tr, 0.02, log, "acq2")
    if c2 is not None:
        xy = tr["xy"].copy()
        if abs(wrap(c2["yaw"] - c["yaw"])) > 0.15 and abs(abs(wrap(c2["yaw"] - c["yaw"])) - math.pi / 2) > 0.15:
            yaw, score = choose_yaw(rb, xy, c2["yaw"], others, button_xy, log=log)
        rb.move_to([xy[0], xy[1], 0.10], yaw=yaw, tol=0.004)
        c = c2
    ztop = c["ztop"]
    zg = max(0.015, ztop - 0.02)
    rb.move_to([xy[0], xy[1], ztop + 0.03], vmax=0.4)
    rb.move_to([xy[0], xy[1], zg], vmax=0.25, max_iter=4)
    rb.hold(8, grip=1.0)
    w = rb.state[6] + rb.state[7]
    log("grasp track %d z=%.3f width=%.4f step %s" % (tid, rb.pos[2], w, rb.meta.get("step_id")))
    if rb.ended:
        return False, c
    if w < 0.01:
        log("grasp missed; reopening")
        rb.hold(5, grip=-1.0)
        rb.move_to([xy[0], xy[1], 0.15], vmax=0.4, tol=0.01)
        return False, c
    rb.move_to([xy[0], xy[1], lift_z], vmax=0.3, tol=0.01)
    w2 = rb.state[6] + rb.state[7]
    held = w2 > 0.01
    log("lifted z=%.3f width=%.4f held=%s ended=%s" % (rb.pos[2], w2, held, rb.ended))
    rb.hold(4)
    rb.move_to([xy[0], xy[1], zg + 0.008], vmax=0.3, max_iter=4)
    rb.hold(5, grip=-1.0)
    tr["xy"] = xy.copy(); tr["v"] = np.zeros(2)
    rb.move_to([xy[0], xy[1], 0.15], vmax=0.4, tol=0.01)
    return held, c
