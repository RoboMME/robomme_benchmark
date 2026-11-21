"""Pick, place and press skills built on Robot + scene perception."""
import math
import numpy as np
from scene import find_cube, find_target, find_button

HOVER = 0.13        # TCP height for looking/hovering
OPEN, CLOSE = -1.0, 1.0


def wrap_yaw(cube_yaw, ref):
    """Gripper yaw equivalent to cube_yaw mod 90deg closest to ref."""
    q = math.pi / 2
    k = round((ref - cube_yaw) / q)
    return cube_yaw + k * q


def choose_yaw(rb, cube, log=print):
    """Pick the cube-aligned yaw (two options 90deg apart) whose finger positions are free of obstacles.
    At yaw 0 the fingers close along world y."""
    from perception import world_points
    P = world_points(rb.meta, rb.imgs, "wrist").reshape(-1, 3)
    tz = cube["table_z"]
    P = P[np.isfinite(P[:, 2]) & (P[:, 2] > tz + 0.012) & (P[:, 2] < tz + 0.15)]
    c = cube["xyz"][:2]
    d = np.linalg.norm(P[:, :2] - c, axis=1)
    P = P[d > 0.035]  # drop the cube itself
    best = None
    y0 = wrap_yaw(cube["yaw"], 0.0)
    y1 = y0 + math.pi / 2 if y0 < 0 else y0 - math.pi / 2
    for yaw in (y0, y1):
        ax = np.array([-math.sin(yaw), math.cos(yaw)])
        n = 0
        for sgn in (1, -1):
            f = c + sgn * 0.055 * ax
            n += int((np.linalg.norm(P[:, :2] - f, axis=1) < 0.022).sum())
        cand = (n, abs(yaw), yaw)
        if best is None or cand < best:
            best = cand
    if best[0] > 0:
        log(f"choose_yaw: obstacle pixels near fingers {best[0]} at best yaw {math.degrees(best[2]):.0f}")
    return best[2]


def grip_width(rb):
    s = rb.state
    return float(s[6] + s[7])


def pick(rb, cube_color, guess_xy, log=print, attempt=0):
    """Hover over guess, refine with wrist camera, descend, grasp, lift. Returns True if holding."""
    rb.move_to([guess_xy[0], guess_xy[1], HOVER], grip=OPEN, tol=0.006)
    c = find_cube(rb.meta, rb.imgs, cube_color, "wrist", near=guess_xy)
    if c is None:
        c = find_cube(rb.meta, rb.imgs, cube_color, "wrist")
    if c is None:
        log("pick: cube not seen from hover")
        return False, guess_xy
    xy = c["xyz"][:2]
    yaw = wrap_yaw(c["yaw"], 0.0)
    rb.move_to([xy[0], xy[1], HOVER], yaw=yaw, grip=OPEN, tol=0.004)
    c2 = find_cube(rb.meta, rb.imgs, cube_color, "wrist", near=xy)
    if c2 is not None:
        xy = c2["xyz"][:2]
        yaw = choose_yaw(rb, c2, log)
        top = c2["top_z"]
    else:
        top = c["top_z"]
    log(f"pick: cube at {np.round(xy,4)} top={top:.4f} yaw={math.degrees(yaw):.1f}")
    gz = max(top - 0.022, c["table_z"] + 0.012)
    rb.move_to([xy[0], xy[1], gz + 0.05], yaw=yaw, grip=OPEN, tol=0.004)
    rb.move_to([xy[0], xy[1], gz], yaw=yaw, grip=OPEN, tol=0.003, max_cmd=0.3)
    rb.hold(12, grip=CLOSE)
    w = grip_width(rb)
    rb.move_to([xy[0], xy[1], HOVER], grip=CLOSE, tol=0.008)
    w2 = grip_width(rb)
    held = w2 > 0.02
    log(f"pick: width after close {w:.4f} after lift {w2:.4f} held={held}")
    return held, xy


def place(rb, target_xyz, log=print):
    tx, ty, tz = target_xyz
    rb.move_to([tx, ty, HOVER], grip=CLOSE, tol=0.005)
    # cube centre ~0.02 above surface; TCP holds cube at its centre
    rb.move_to([tx, ty, tz + 0.045], grip=CLOSE, tol=0.004)
    rb.move_to([tx, ty, tz + 0.025], grip=CLOSE, tol=0.003, max_cmd=0.2, max_steps=40)
    rb.hold(8, grip=OPEN)
    rb.move_to([tx, ty, HOVER], grip=OPEN, tol=0.008)
    return True


LEAD = 2            # steps between release decision and cube landing
REL_DIST = 0.004    # release when predicted target-cube distance below this
PASS_DIST = 0.022   # or at closest approach if within this
FRESH_AGE = 25      # max age (steps) of the newest complete detection for a moving-target release


def tcp_xy(rb):
    return rb.state[:2].copy()


def place_on_target(rb, tr, log=print, max_wait=420):
    """Carry the held cube to a point P on the target's path, hold it just above the disc,
    release as the (possibly moving) target passes under it. Returns (released, P)."""
    now = int(rb.meta["step_id"])
    sp = tr.speed(now, 30)
    P = tr.nearest_path_point(tcp_xy(rb), now)
    if P is None:
        log("place: no target history")
        return False, None
    disc_z = tr.disc_z if tr.disc_z is not None else 0.0085
    wait_z = disc_z + 0.024
    log(f"place: P={np.round(P,4)} speed={sp} disc_z={disc_z:.4f} step={now}")
    rb.move_to([P[0], P[1], HOVER], grip=CLOSE, tol=0.006)
    rb.move_to([P[0], P[1], wait_z + 0.03], grip=CLOSE, tol=0.005)
    rb.move_to([P[0], P[1], wait_z], grip=CLOSE, tol=0.003, max_cmd=0.25, max_steps=40)
    t_start = int(rb.meta["step_id"])
    prev_d = None
    retargets = 0
    while not rb.ended:
        now = int(rb.meta["step_id"])
        cxy = tcp_xy(rb)
        p2, ok = tr.predict(now, LEAD)
        p3, _ = tr.predict(now, LEAD + 1)
        if p2 is None:
            rb.hold(2, grip=CLOSE)
            continue
        d2 = float(np.linalg.norm(p2 - cxy))
        d3 = float(np.linalg.norm(p3 - cxy))
        if not tr.ever_moving:
            if d2 < 0.02:
                log(f"place: static target, release at step {now} d={d2:.4f}")
                break
            if retargets < 3:
                retargets += 1
                P = p2.copy()
                log(f"place: static target off by {d2:.4f}; re-centre to {np.round(P,4)}")
                rb.move_to([P[0], P[1], wait_z], grip=CLOSE, tol=0.003, max_cmd=0.25, max_steps=30)
                continue
            log(f"place: static target, releasing after retargets d={d2:.4f}")
            break
        fresh = tr.last_age(now) <= FRESH_AGE
        if fresh and (d2 < REL_DIST or (d2 < PASS_DIST and d3 > d2)):
            log(f"place: release at step {now} pred_d={d2:.4f} next={d3:.4f} age={tr.last_age(now)} waited={now - t_start}")
            break
        if rb.remaining < 90:
            log("place: step budget low, releasing now")
            break
        if now - t_start > max_wait:
            if retargets >= 2:
                log("place: wait timeout, releasing over predicted target anyway")
                break
            retargets += 1
            q, _ = tr.predict(now, 12)
            log(f"place: wait timeout -> retarget to {np.round(q,4)}")
            P = q.copy()
            rb.move_to([q[0], q[1], wait_z], grip=CLOSE, tol=0.004, max_steps=20)
            t_start = int(rb.meta["step_id"])
            continue
        # target far: hold 2 steps per call; near: 1 step
        rb.hold(1 if d2 < 0.08 else 2, grip=CLOSE)
        # small xy correction to stay at P
        st = rb.state
        if np.linalg.norm(st[:2] - P) > 0.006 or abs(st[2] - wait_z) > 0.006:
            rb.move_to([P[0], P[1], wait_z], grip=CLOSE, tol=0.003, max_cmd=0.25, max_steps=4, chunk=2)
    rb.hold(5, grip=OPEN)
    rb.move_to([cxy[0], cxy[1], wait_z + 0.05], grip=OPEN, tol=0.006, max_steps=30)
    return True, cxy


def verify_on_target(rb, tr, color, cube_xy, log=print):
    """After release: target should be stopped with the cube on it."""
    rb.hold(4, grip=OPEN)
    now = int(rb.meta["step_id"])
    r = tr.recent(now, 8)
    moving = None
    if len(r) >= 2:
        moving = float(np.linalg.norm(r[-1][1] - r[0][1])) > 0.006
    c = find_cube(rb.meta, rb.imgs, color, "wrist", near=cube_xy)
    last = tr.hist[-1][1] if tr.hist else None
    dist = None
    if c is not None and last is not None:
        dist = float(np.linalg.norm(c["xyz"][:2] - last))
    log(f"verify: moving={moving} cube={None if c is None else np.round(c['xyz'],4)} target={None if last is None else np.round(last,4)} dist={dist}")
    ok = (moving is not True) and (dist is None or dist < 0.03)
    return ok, (c["xyz"][:2] if c is not None else np.asarray(cube_xy))
