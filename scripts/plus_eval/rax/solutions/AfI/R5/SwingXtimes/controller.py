"""SwingXtimes controller (RAX). Observation-driven: parse count, find targets/button/cube,
grasp, alternate right(+y, image-right in front camera)->left visits N times, put down, press button."""
import os, sys, json, time, traceback
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robo import Robot, log, LOG
from percep import cube_estimate, color_mask, components
from scene import parse_count, parse_order, find_targets_button, white_blobs
from skills import DOWN, wrap90, locate_cube, CUBE_COLOR
from scene import parse_color

RIGHT_SIGN = +1.0   # right-side target = larger world y (image-right in the front camera)
Z_TRAV = 0.09
Z_LOW = 0.045
Z_VISIT = 0.075
OUT = "/workspace/output"


def save_log():
    try:
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "controller_log.txt"), "w") as f:
            f.write("\n".join(LOG))
    except Exception:
        pass


def target_lit(o, xy, cams=("front", "wrist")):
    best = 0
    for cam in cams:
        try:
            m = color_mask(o.rgb(cam), "red")
            P = o.deproject_mask(cam, m)
        except Exception:
            continue
        if len(P) == 0:
            continue
        d = np.linalg.norm(P[:, :2] - np.asarray(xy)[None], axis=1)
        best = max(best, int(np.sum((P[:, 2] < 0.013) & (P[:, 2] > -0.01) & (d < 0.045))))
    return best


def redetect_target(o, xy, other_xy, radius=0.15):
    """Nearest table-level neutral (target-like) blob to xy, from both cameras."""
    cands = []
    for cam in ("front", "wrist"):
        try:
            for b in white_blobs(o, cam):
                p = b["xyz"]
                if p[2] < 0.013 and p[2] > -0.01 and b["area"] >= 8:
                    if np.linalg.norm(p[:2] - other_xy) > 0.08 and np.linalg.norm(p[:2] - xy) < radius:
                        cands.append((b["area"], p[:2].copy(), cam))
        except Exception:
            continue
    if not cands:
        return None
    # cluster rings: weighted mean of candidates near the nearest one
    near = min(cands, key=lambda c: np.linalg.norm(c[1] - xy))[1]
    grp = [c for c in cands if np.linalg.norm(c[1] - near) < 0.04]
    w = np.array([c[0] for c in grp], float)
    return np.average([c[1] for c in grp], axis=0, weights=w)


def seg_dist(p, a, b):
    p, a, b = map(np.asarray, (p, a, b))
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def travel_xy(rb, xy, avoid, z=Z_TRAV, clear=0.07, tol=0.015):
    """Move at height z to xy, inserting a detour waypoint if the straight path passes near `avoid` points."""
    start = rb.state[:2].copy()
    if rb.state[2] < z - 0.02:
        rb.move_to([start[0], start[1], z], tol=0.02, max_steps=30, chunk=3)
    for a in avoid:
        if seg_dist(a, start, xy) < clear:
            # detour: offset perpendicular to the path, choose side away from the avoid point
            d = np.asarray(xy) - start
            n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-6)
            mid = (start + np.asarray(xy)) / 2
            if np.dot(mid - a, n) < 0:
                n = -n
            wp = a + n * (clear + 0.03)
            log("detour via", np.round(wp, 3), "avoiding", np.round(a, 3))
            rb.move_to([wp[0], wp[1], z], tol=0.025, max_steps=60, chunk=3)
            break
    rb.move_to([xy[0], xy[1], z], tol=tol, max_steps=80, chunk=3)


def grasp(rb, avoid, attempts=3):
    for k in range(attempts):
        rb.grip = -1.0
        rb.down_yaw = DOWN
        rb.hold(4, grip=-1.0)
        if rb.state[2] < 0.12:
            rb.move_to([rb.state[0], rb.state[1], 0.14], tol=0.02, max_steps=30, chunk=3)
        est, cam, o = locate_cube(rb, ("front", "wrist"))
        if est is None:
            log("grasp: cube not visible; moving up for view")
            rb.move_to([0.0, 0.0, 0.3], tol=0.03, max_steps=40, chunk=4)
            est, cam, o = locate_cube(rb, ("front", "wrist"))
            if est is None:
                continue
        c = est[0]
        log("grasp attempt", k, "coarse", cam, np.round(c, 3), round(est[1], 3))
        rb.move_to([c[0], c[1], 0.13], tol=0.006, max_steps=80, chunk=4)
        est, cam, o = locate_cube(rb, ("wrist", "front"))
        if est is None:
            continue
        c, ztop, yaw, n = est
        dyaw = wrap90(yaw) if yaw is not None else 0.0
        rb.rotate_to(R.from_rotvec([0, 0, dyaw]) * DOWN, max_steps=30)
        rb.move_to([c[0], c[1], 0.13], tol=0.004, max_steps=30, chunk=3)
        est2, _, o = locate_cube(rb, ("wrist",))
        if est2 is not None:
            c, ztop = est2[0], est2[1]
        log("grasp refine", np.round(c, 3), round(ztop, 3), "yaw", round(float(dyaw), 3))
        gz = float(np.clip(ztop - 0.02, 0.012, 0.06))
        rb.move_to([c[0], c[1], gz + 0.04], tol=0.005, max_steps=40, chunk=3)
        rb.move_to([c[0], c[1], gz], tol=0.003, max_steps=40, chunk=2, vmax=0.5)
        rb.hold(10, grip=1.0)
        rb.move_to([c[0], c[1], Z_TRAV], tol=0.015, max_steps=40, chunk=3)
        held = rb.state[6] + rb.state[7]
        log("grasp lift finger sum", round(held, 4))
        if held > 0.01:
            return True
        rb.hold(3, grip=-1.0)
    return False


def visit(rb, T, i, j, max_chase=8):
    """Bring held cube over target T[i] (live re-detected) until it lights. Updates T[i]."""
    s0 = rb.step_id
    xy, other = T[i], T[j]
    travel_xy(rb, xy, avoid=[other] if seg_dist(other, rb.state[:2], xy) < 0.07 and
              np.linalg.norm(rb.state[:2] - other) > 0.05 else [], z=Z_TRAV, tol=0.012)
    lit = 0
    for it in range(max_chase):
        o = rb.observe()
        lit = target_lit(o, xy)
        if lit > 5:
            break
        nxy = redetect_target(o, xy, other)
        if nxy is not None and np.linalg.norm(nxy - xy) > 0.008:
            log("target moved", np.round(xy, 3), "->", np.round(nxy, 3))
            xy = nxy
        rb.move_to([xy[0], xy[1], Z_VISIT], tol=0.006, max_steps=20, chunk=2)
        if rb.ended:
            break
    T[i] = np.asarray(xy)
    held = rb.state[6] + rb.state[7]
    log("visit", i, np.round(xy, 3), "EE", np.round(rb.state[:3], 3), "steps", rb.step_id - s0,
        "held", round(held, 4), "lit", lit, "remaining", rb.remaining)
    return held > 0.01, lit


def choose_drop(cube0, targets, button, others):
    cands = [np.asarray(cube0)]
    mid = (targets[0] + targets[1]) / 2
    for dx in (0.12, 0.08, -0.08):
        cands.append(mid + np.array([dx, 0.0]))
    cands.append(mid + np.array([0.15, 0.1]))
    cands.append(mid + np.array([0.15, -0.1]))
    keep = [t for t in targets] + ([button["xy"]] if button else [])
    for c in cands:
        if all(np.linalg.norm(c - k) > 0.08 for k in keep) and all(np.linalg.norm(c - k) > 0.05 for k in others):
            return c
    return cands[1]


def put_down(rb, drop, avoid, z_release=0.034):
    """Lower the held cube at `drop` (already above it) and release."""
    if np.linalg.norm(rb.state[:2] - drop) > 0.02:
        travel_xy(rb, drop, avoid=avoid)
    rb.move_to([drop[0], drop[1], z_release + 0.02], tol=0.006, max_steps=40, chunk=3)
    rb.move_to([drop[0], drop[1], z_release], tol=0.004, max_steps=20, chunk=2, vmax=0.4)
    rb.hold(8, grip=-1.0)
    rb.move_to([drop[0], drop[1], 0.12], tol=0.02, max_steps=30, chunk=3)
    log("put down at", np.round(drop, 3), "EE", np.round(rb.state[:3], 3))


def button_points(o, cam, bxy, rad=0.06):
    m = color_mask(o.rgb(cam), "white")
    P = o.deproject_mask(cam, m)
    if len(P) == 0:
        return P
    d = np.linalg.norm(P[:, :2] - bxy[None], axis=1)
    return P[(d < rad) & (P[:, 2] > 0.011) & (P[:, 2] < 0.05)]


def press_button(rb, button):
    rb.grip = 1.0
    bxy = np.asarray(button["xy"], float)
    zt = float(button.get("ztop", 0.03))
    rb.hold(4, grip=1.0)
    rb.move_to([bxy[0], bxy[1], 0.15], tol=0.008, max_steps=80, chunk=4)
    for k in range(2):
        try:
            o = rb.observe()
            P = button_points(o, "wrist", bxy)
            if len(P) >= 30:
                nb = np.median(P[:, :2], axis=0)
                if np.linalg.norm(nb - bxy) < 0.04:
                    bxy = nb
                    zt = float(np.percentile(P[:, 2], 95))
                log("button refine (wrist)", np.round(bxy, 3), round(zt, 3), len(P))
        except Exception as e:
            log("button refine failed", e)
        rb.move_to([bxy[0], bxy[1], 0.13], tol=0.004, max_steps=30, chunk=3)
    zt = float(np.clip(zt, 0.015, 0.045))
    rb.move_to([bxy[0], bxy[1], zt + 0.025], tol=0.004, max_steps=40, chunk=3)
    for i in range(40):
        if rb.ended:
            break
        rb.step([[0, 0, -0.2] + list(rb.rot_cmd()) + [1.0]])
        if rb.state[2] < zt - 0.03:
            break
    log("press: EE", np.round(rb.state[:3], 3), "ended", rb.ended, "complete", rb.complete)
    if not rb.ended:
        rb.move_to([bxy[0], bxy[1], zt + 0.05], tol=0.01, max_steps=20, chunk=3)


RULE = os.environ.get("SWING_RULE", "f")


def predict_first(targets, first, cube_xy):
    t0, t1 = np.asarray(targets[0]), np.asarray(targets[1])
    ypos = t0 if t0[1] > t1[1] else t1
    yneg = t1 if ypos is t0 else t0
    out = {"f": ypos}
    # (a) right = larger world x
    bigx = t0 if t0[0] > t1[0] else t1
    smallx = t1 if bigx is t0 else t0
    out["a"] = bigx if first == "right" else smallx
    # camera-view (+y right) and robot-view (-y right)
    out["cam"] = ypos if first == "right" else yneg
    out["robot"] = yneg if first == "right" else ypos
    if cube_xy is not None:
        c = np.asarray(cube_xy)
        out["g"] = t0 if np.linalg.norm(t0 - c) < np.linalg.norm(t1 - c) else t1
        f = (t0 + t1) / 2 - c
        rv = np.array([f[1], -f[0]])
        rt = t0 if np.dot(t0 - c, rv) > np.dot(t1 - c, rv) else t1
        lt = t1 if rt is t0 else t0
        out["b"] = rt if first == "right" else lt
    return out


def run(ctx):
    rb = Robot(ctx)
    o = rb.observe()
    instr = o.meta.get("instruction", "")
    n = parse_count(instr) or 1
    log("instruction:", instr, "| count", n, "| limit", rb.remaining)
    targets, button, cl, bl = find_targets_button(o, "front")
    if len(targets) < 2:
        t2, b2, _, _ = find_targets_button(o, "wrist")
        if len(t2) >= 2:
            targets = t2
        button = button or b2
    if len(targets) < 2:
        log("ERROR: targets not found", targets)
        return
    targets = sorted(targets, key=lambda t: t[1] * RIGHT_SIGN)
    left, right = np.asarray(targets[0]), np.asarray(targets[1])
    first = parse_order(instr)
    CUBE_COLOR[0] = parse_color(instr)
    log("cube color", CUBE_COLOR[0])
    est = cube_estimate(o, "front", CUBE_COLOR[0]) or cube_estimate(o, "wrist", CUBE_COLOR[0])
    cube_xy = est[0] if est else None
    preds = predict_first(targets, first, cube_xy)
    log("first-target predictions:", {k: np.round(v, 3).tolist() for k, v in preds.items()})
    A = np.asarray(preds[RULE])
    B = right if np.linalg.norm(A - right) > 0.01 else left
    nameA, nameB = "A", "B"
    log("right", np.round(right, 3), "left", np.round(left, 3), "first", first, "button", button and np.round(button["xy"], 3))
    cube0 = cube_xy if cube_xy is not None else (left + right) / 2 + np.array([0.12, 0])
    if not grasp(rb, avoid=[left, right]):
        log("ERROR: grasp failed")
        return
    T = [np.asarray(A, float), np.asarray(B, float)]
    travel_xy(rb, T[0], avoid=[T[1]])
    done = 0
    while done < n and not rb.ended:
        for i, j in ((0, 1), (1, 0)):
            ok = False
            for attempt in range(3):
                held, lit = visit(rb, T, i, j)
                if rb.ended:
                    break
                if not held:
                    log("cube dropped; regrasp")
                    if not grasp(rb, avoid=T):
                        return
                    continue
                ok = True
                break
            if rb.ended or not ok:
                break
        done += 1
        log("swing", done, "of", n, "step", rb.step_id, "remaining", rb.remaining)
    left, right = T[0], T[1]
    if rb.ended:
        log("episode ended during swings; complete", rb.complete)
        return
    drop = np.asarray(T[1], float)
    put_down(rb, drop, avoid=[T[0]])
    if button is None:
        o = rb.observe()
        _, button, _, _ = find_targets_button(o, "front")
    if button is None:
        log("ERROR: button not found")
        return
    press_button(rb, button)
    if not rb.ended:
        # retry once with slightly deeper press
        button["ztop"] = button.get("ztop", 0.03) - 0.01
        press_button(rb, button)
    log("final: ended", rb.ended, "complete", rb.complete, "step", rb.step_id)


def main(ctx):
    t0 = time.time()
    try:
        run(ctx)
    except Exception as e:
        log("EXCEPTION", repr(e), traceback.format_exc())
    finally:
        log("wall", round(time.time() - t0, 1))
        save_log()
