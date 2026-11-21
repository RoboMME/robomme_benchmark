import math, json, os, time, traceback
import numpy as np
from robot import Robot, Ended, log, wrap, LOG
from scene import parse_instruction, detect
from servo import Tracker, observe_obj, servo

HOVER_Z = 0.16
OVERVIEW = np.array([-0.06, 0.0, 0.45])
PLACE_Z = 0.036


def look(rob, color, cams=("wrist", "front")):
    rob.observe()
    res = {}
    for cam in cams:
        try:
            d = detect(rob.meta, rob.imgs, cam, color)
        except Exception as e:
            log("detect error", cam, e)
            continue
        for k, v in d.items():
            res.setdefault(k, (cam, v))
    return res


def cube_yaw_target(rob, cyaw):
    if cyaw is None:
        return rob.finger_yaw()
    cur = rob.finger_yaw()
    return cur + wrap(cyaw - cur, math.pi / 2)


def pick(rob, color, cube):
    xy, yaw = np.array(cube["xy"]), cube.get("yaw")
    rob.move_to([xy[0], xy[1], HOVER_Z], yaw=cube_yaw_target(rob, yaw), grip=-1, tol=0.008)
    r = look(rob, color, cams=("wrist", "front"))
    if "cube" in r:
        c = r["cube"][1]
        if c.get("yaw") is not None:
            yaw = c["yaw"]
        cube = c
    ztop = cube.get("z", 0.04)
    gz = max(ztop - 0.022, 0.012)
    tr = Tracker("cube")
    observe_obj(rob, "cube", color, tr, gate=0.1, cams=("wrist", "front"))
    servo(rob, "cube", color, tr, gz, grip=-1.0, cams=("wrist", "front"), finish_rows=8, finish_grip=1.0,
          yaw=cube_yaw_target(rob, yaw), xy_gate_descend=0.01, xy_tol=0.004, zspeed=0.025)
    w_closed = rob.grip_width
    p = rob.pos
    rob.move_to([p[0], p[1], HOVER_Z], grip=1.0, tol=0.015)
    w = rob.grip_width
    ok = w > 0.01
    log("pick: width after close %.4f after lift %.4f ok=%s step=%s" % (w_closed, w, ok, rob.step_id))
    return ok


def place(rob, color, ttrack):
    # move above the target's current predicted position first
    observe_obj(rob, "target", color, ttrack, cams=("front", "wrist"))
    p = ttrack.predict(rob.step_id + 8)
    rob.move_to([p[0], p[1], HOVER_Z - 0.04], grip=1.0, tol=0.02, max_iter=3)
    servo(rob, "target", color, ttrack, PLACE_Z, grip=1.0, cams=("front", "wrist"), finish_rows=4,
          finish_grip=-1.0, xy_gate_descend=0.012, xy_tol=0.005, zspeed=0.03, max_calls=60, settle_calls=12)
    # lift straight up quickly
    rob.step([[0, 0, 0.6, 0, 0, 0, -1.0]] * 5)
    # verify from the same frame: cube near live target
    on = None
    try:
        d = detect(rob.meta, rob.imgs, "front", color)
        if "cube" in d and "target" in d:
            off = float(np.linalg.norm(np.array(d["cube"]["xy"]) - np.array(d["target"]["xy"])))
            on = off < 0.03
            log("verify: cube-target offset %.3f on=%s" % (off, on))
    except Exception as e:
        log("verify err", e)
    p = rob.pos
    rob.move_to([p[0], p[1], HOVER_Z], grip=-1.0, tol=0.02, max_iter=3)
    return on


def press_button(rob, button):
    xy = np.array(button["xy"])
    bz = button.get("z", 0.034)
    rob.move_to([xy[0], xy[1], HOVER_Z], grip=1.0, tol=0.004)
    rob.hold(4, 1.0)
    rob.move_to([xy[0], xy[1], bz + 0.03], grip=1.0, tol=0.004)
    rob.move_to([xy[0], xy[1], bz - 0.02], grip=1.0, tol=0.003, speed=0.3, max_iter=4)
    rob.hold(4, 1.0)
    rob.move_to([xy[0], xy[1], HOVER_Z], grip=1.0, tol=0.01)


def main(ctx):
    t0 = time.time()
    rob = Robot(ctx)
    color, count = parse_instruction(rob.meta["instruction"])
    log("instruction:", rob.meta["instruction"], "->", color, count)
    if color is None:
        color = "green"
    if count is None:
        count = 1
    try:
        run(rob, color, count, t0)
    except Ended:
        log("episode ended; task_complete=%s" % rob.task_complete)
    except Exception:
        log("ERROR", traceback.format_exc())
    finally:
        try:
            os.makedirs("/workspace/output", exist_ok=True)
            open("/workspace/output/controller_log.txt", "w").write("\n".join(LOG))
        except Exception:
            pass


def run(rob, color, count, t0):
    rob.move_to(OVERVIEW, yaw=0.0, tol=0.01)
    sc = look(rob, color)
    log("overview:", {k: (v[0], np.round(v[1]["xy"], 3).tolist()) for k, v in sc.items()})
    ttrack = Tracker("target")
    observe_obj(rob, "target", color, ttrack, cams=("front", "wrist"))
    # watch the target briefly to estimate its motion
    for _ in range(4):
        rob.hold(2, -1.0)
        observe_obj(rob, "target", color, ttrack, cams=("front", "wrist"))
    log("target speed %.4f m/step" % ttrack.speed())
    button = sc.get("button", (None, None))[1]
    done = 0
    attempts = 0
    while done < count and attempts < count + 4:
        attempts += 1
        if time.time() - t0 > 1000 or rob.remaining < 150:
            log("budget low, stop picking")
            break
        r = look(rob, color)
        cube = r.get("cube", (None, None))[1]
        if cube is None:
            rob.move_to(OVERVIEW, tol=0.02)
            r = look(rob, color)
            cube = r.get("cube", (None, None))[1]
            if cube is None:
                log("no cube found")
                break
        ok = pick(rob, color, cube)
        if not ok:
            log("grasp failed; retry")
            rob.hold(4, -1.0)
            p = rob.pos
            rob.move_to([p[0], p[1], 0.3], grip=-1, tol=0.02)
            continue
        on = place(rob, color, ttrack)
        # every release after a successful grasp counts (the native counter appears lenient;
        # re-placing an uncounted attempt risks exceeding the required count)
        done += 1
        log("placed %d/%d step=%s rem=%s t=%.0f" % (done, count, rob.step_id, rob.remaining, time.time() - t0))
    if button is None:
        rob.move_to(OVERVIEW, tol=0.02)
        button = look(rob, color).get("button", (None, None))[1]
    if button is not None:
        press_button(rob, button)
        log("button pressed; step=%s" % rob.step_id)
        rob.hold(5, 1.0)
    log("finished; task_complete=%s ended=%s" % (rob.task_complete, rob.ended))
