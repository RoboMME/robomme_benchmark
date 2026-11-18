"""PickXtimes controller: pick cube -> place on target, N times, then press button."""
import os, sys, time, json, math, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion import Robot
from scene import parse_instruction, find_cube, find_target, find_button
from skills import pick, place, grip_width, place_on_target, verify_on_target, HOVER, OPEN, CLOSE
from tracker import Tracker

LOGF = None


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    try:
        with open("/workspace/output/controller_log.txt", "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def overview(rb, home):
    """Go to the high overview pose (open gripper) and perceive the scene."""
    rb.move_to(home, yaw=0.0, tol=0.01)
    return rb.meta, rb.imgs


def scan(rb, color, home):
    meta, imgs = rb.meta, rb.imgs
    C = find_cube(meta, imgs, color, "wrist") or find_cube(meta, imgs, color, "front")
    T = find_target(meta, imgs, "wrist") or find_target(meta, imgs, "front")
    B = find_button(meta, imgs, "wrist") or find_button(meta, imgs, "front")
    return C, T, B


def press_button(rb, B):
    bx, by = B["base_xy"]
    top = B["xyz"][2]
    log(f"press: button at {bx:.4f},{by:.4f} cap_top={top:.4f}")
    rb.move_to([bx, by, HOVER], grip=CLOSE, tol=0.005)
    # refine with wrist view
    B2 = find_button(rb.meta, rb.imgs, "wrist")
    if B2 is not None and np.linalg.norm(B2["base_xy"] - np.array([bx, by])) < 0.05:
        bx, by = B2["base_xy"]; top = B2["xyz"][2]
        log(f"press: refined {bx:.4f},{by:.4f} cap_top={top:.4f}")
        rb.move_to([bx, by, HOVER], grip=CLOSE, tol=0.004)
    rb.hold(6, grip=CLOSE)
    rb.move_to([bx, by, top + 0.03], grip=CLOSE, tol=0.004)
    for dz in [0.01, 0.0, -0.008, -0.015, -0.02]:
        if rb.ended:
            break
        rb.move_to([bx, by, top + dz], grip=CLOSE, tol=0.003, max_cmd=0.2, max_steps=25)
        rb.hold(3)
        log(f"press: dz={dz} z={rb.state[2]:.4f} ended={rb.ended} complete={rb.task_complete}")
    if not rb.ended:
        rb.move_to([bx, by, top + 0.06], grip=CLOSE, tol=0.01, max_steps=30)
    return rb.task_complete


def run(ctx, start_count=0, press=True):
    t0 = time.time()
    rb = Robot(ctx, log=log)
    tr = Tracker(log=log)
    rb.listeners.append(tr.update)
    tr.update(rb.meta, rb.imgs)
    color, n = parse_instruction(rb.meta["instruction"])
    log(f"instruction: {rb.meta['instruction']!r} -> color={color} n={n}")
    home = rb.state[:3].copy()
    home[2] = max(home[2], 0.45)
    C, T, B = scan(rb, color, home)
    log(f"scan: cube={None if C is None else np.round(C['xyz'],4)} target={None if T is None else np.round(T['xyz'],4)} button={None if B is None else np.round(B['base_xy'],4)}")
    if C is None or T is None or B is None:
        overview(rb, home)
        C2, T2, B2 = scan(rb, color, home)
        C, T, B = C or C2, T or T2, B or B2
    if not tr.hist and T is not None:
        tr.hist.append((int(rb.meta["step_id"]), T["xyz"][:2].copy(), "scan"))
        tr.disc_z = T["xyz"][2]
    cube_xy = C["xyz"][:2] if C is not None else (T["xyz"][:2] if T is not None else home[:2])
    count = start_count
    failures = 0
    while count < n and not rb.ended and failures < 8:
        if rb.remaining < 120:
            log("low on steps; stopping cycles")
            break
        held, xy = pick(rb, color, cube_xy, log=log)
        if not held:
            failures += 1
            log(f"pick failed (failures={failures}); rescanning")
            rb.move_to([rb.state[0], rb.state[1], HOVER], grip=OPEN, tol=0.01, max_steps=30)
            overview(rb, home)
            C2 = find_cube(rb.meta, rb.imgs, color, "wrist") or find_cube(rb.meta, rb.imgs, color, "front")
            if C2 is not None:
                cube_xy = C2["xyz"][:2]
            continue
        released, rel_xy = place_on_target(rb, tr, log=log)
        if not released:
            failures += 1
            overview(rb, home)
            C2 = find_cube(rb.meta, rb.imgs, color, "wrist") or find_cube(rb.meta, rb.imgs, color, "front")
            if C2 is not None:
                cube_xy = C2["xyz"][:2]
            continue
        ok, cxy = verify_on_target(rb, tr, color, rel_xy, log=log)
        cube_xy = cxy
        if ok:
            count += 1
            log(f"placed #{count}/{n} cube={np.round(cxy,4)} step={rb.meta['step_id']} t={time.time()-t0:.0f}s")
        else:
            failures += 1
            log(f"place verify failed (failures={failures}) step={rb.meta['step_id']}")
            c = find_cube(rb.meta, rb.imgs, color, "wrist", near=rel_xy)
            if c is None:
                overview(rb, home)
                c = find_cube(rb.meta, rb.imgs, color, "wrist") or find_cube(rb.meta, rb.imgs, color, "front")
            if c is not None:
                cube_xy = c["xyz"][:2]
    if press and not rb.ended and B is not None:
        press_button(rb, B)
    log(f"done: count={count}/{n} ended={rb.ended} step={rb.meta.get('step_id')} t={time.time()-t0:.0f}s")
    return rb


def main(ctx):
    os.makedirs("/workspace/output", exist_ok=True)
    try:
        run(ctx)
    except Exception:
        log("EXCEPTION " + traceback.format_exc())
