"""VideoRepick controller v1: track the picked object through the conditioning video,
pick & place it twice, then press the button."""
import os, sys, time, json, traceback
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np
from motion import Robot, Ended
import perception as pc
import video

VERSION = "v4"
HOVER_DZ = 0.20      # hover height above table for wrist refinement
GRASP_DZ = 0.020     # EE height above table at grasp (video expert used ~0.02)
LIFT_DZ = 0.15       # lift height above table
PLACE_DZ = 0.025     # EE height above table when releasing
MATCH_GATE = 0.05    # max live-vs-video target mismatch (m)

OUT = Path("/workspace/output")


def mklog():
    OUT.mkdir(parents=True, exist_ok=True)
    f = open(OUT / "controller_log.txt", "a")
    t0 = time.time()

    def log(msg):
        line = f"[{VERSION} {time.time()-t0:6.1f}s] {msg}"
        print(line, flush=True)
        f.write(line + "\n"); f.flush()
    return log


def pick_live_target(robot, pred_xy, log, cam="front", tz=None):
    if cam == "fused":
        B = pc.live_blobs_fused(robot, tz)
        objs = [{"xy": b.mean(0), "n": len(b)} for b in B]
    else:
        objs, tz = pc.live_objects(robot, cam)
    if not objs:
        return None, tz, objs
    d = [np.linalg.norm(o["xy"] - pred_xy) for o in objs]
    j = int(np.argmin(d))
    log(f"live[{cam}]: {len(objs)} objs, nearest to pred {np.round(pred_xy,3)} is {np.round(objs[j]['xy'],3)} d={d[j]:.3f} n={objs[j]['n']}")
    return objs[j], tz, objs


CTX = {"positions": None, "tgt": None, "unit": 42.0}


def refine_all(robot, xy, tz, log, cams):
    """Associate all known object positions to live grid blobs; return refined target xy or None."""
    P = CTX["positions"]
    if P is None:
        return None
    P = P.copy()
    P[CTX["tgt"]] = xy
    views = [pc.live_points(robot, c) for c in cams]
    zmax = float(np.clip(robot.state()[2] - tz - 0.015, 0.03, 0.09))
    B = pc.grid_blobs_from_views(views, tz, zmax)
    newP, seen = pc.associate(B, P, CTX["unit"])
    t = CTX["tgt"]
    if not seen[t]:
        log(f"refine[{'+'.join(cams)}]: target unseen among {len(B)} blobs")
        return None
    # update others that were seen (they may have been nudged)
    for j in range(len(P)):
        if seen[j]:
            P[j] = newP[j]
    CTX["positions"] = P
    log(f"refine[{'+'.join(cams)}]: {len(B)} blobs, target {np.round(xy,3)} -> {np.round(newP[t],3)}")
    return newP[t]


def wrist_update(robot, xy, tz, log, gate=MATCH_GATE):
    r = None
    try:
        r = refine_all(robot, xy, tz, log, ["wrist"])
    except Exception:
        log("refine failed: " + traceback.format_exc())
    if r is not None and np.linalg.norm(r - xy) < gate:
        return r
    o, _, _ = pick_live_target(robot, xy, log, cam="wrist")
    if o is not None and np.linalg.norm(o["xy"] - xy) < gate:
        return o["xy"]
    return xy


def grasp_cycle(robot, xy, tz, log, k):
    """Hover, refine with wrist, descend, grasp, lift. Returns (ok, xy_refined)."""
    robot.move_to([xy[0], xy[1], tz + HOVER_DZ], grip=-1)
    robot.observe()
    xy = wrist_update(robot, xy, tz, log)
    robot.move_to([xy[0], xy[1], tz + 0.08], grip=-1)
    robot.observe()
    xy = wrist_update(robot, xy, tz, log, gate=0.03)
    robot.move_to([xy[0], xy[1], tz + 0.06], grip=-1)
    robot.move_to([xy[0], xy[1], tz + GRASP_DZ], grip=-1, speed=0.3)
    robot.gripper(1, 12)
    robot.move_to([xy[0], xy[1], tz + LIFT_DZ], grip=1, speed=0.4)
    robot.hold(3, 1)
    op = robot.opening()
    ok = op > 0.004
    log(f"grasp {k}: opening after lift {op:.4f} ok={ok} ee={np.round(robot.state()[:3],3)}")
    return ok, xy


def place_cycle(robot, xy, tz, log, k):
    robot.move_to([xy[0], xy[1], tz + PLACE_DZ + 0.02], grip=1, speed=0.4)
    robot.move_to([xy[0], xy[1], tz + PLACE_DZ], grip=1, speed=0.2)
    robot.gripper(-1, 10)
    robot.move_to([xy[0], xy[1], tz + LIFT_DZ], grip=-1)
    log(f"place {k}: done ee={np.round(robot.state()[:3],3)}")


def press_button(robot, btn, tz, log):
    bx, by = btn["xy"]
    top = btn["ztop"]
    robot.move_to([bx, by, top + 0.10], grip=1)
    robot.move_to([bx, by, top + 0.03], grip=1, speed=0.3)
    # push down slowly below the top surface
    for i in range(4):
        if robot.ended:
            return
        robot.step([[0, 0, -0.25, 0, 0, 0, 1]] * 6)
        log(f"press: ee z {robot.state()[2]:.3f} top {top:.3f} complete={robot.complete}")
        if robot.state()[2] < top - 0.02:
            break
    if not robot.ended:
        robot.move_to([bx, by, top + 0.08], grip=1)


def main(ctx):
    log = mklog()
    robot = Robot(ctx, log)
    try:
        p = robot.observe()
        log(f"instruction: {p.get('instruction')} step {p.get('step_id')} limits {p.get('episode_limits')}")
        seq = p.get("frame_sequence") or {}
        man = seq.get("manifest_path")
        res = None
        try:
            res = video.analyze(man, log)
        except Exception:
            log("video analysis failed: " + traceback.format_exc())
        objs, tz = pc.live_objects(robot, "front")
        rgb, pts = pc.live_points(robot, "front")
        btn = pc.button_location(rgb, pts, tz)
        log(f"table z {tz:.4f}; live objects {[np.round(o['xy'],3).tolist() for o in objs]}; button {btn}")
        if res is not None:
            pred = res["target_xy"]
        else:
            pred = objs[0]["xy"] if objs else np.array([-0.1, 0.0])
        o, _, _ = pick_live_target(robot, pred, log, cam="fused", tz=tz)
        xy = o["xy"] if (o is not None and np.linalg.norm(o["xy"] - pred) < MATCH_GATE) else pred
        if res is not None:
            CTX["positions"] = np.array(res["final_all"], float)
            CTX["tgt"] = res["tgt"]
            CTX["unit"] = res["unit"]
            try:
                r = refine_all(robot, pred, tz, log, ["front", "wrist"])
                if r is not None and np.linalg.norm(r - pred) < MATCH_GATE:
                    xy = r
            except Exception:
                log("initial refine failed: " + traceback.format_exc())
        log(f"initial target xy {np.round(xy,3)}")
        done = 0
        attempts = 0
        while done < 2 and attempts < 5 and not robot.ended:
            attempts += 1
            ok, xy = grasp_cycle(robot, xy, tz, log, done + 1)
            if not ok:
                robot.gripper(-1, 8)
                robot.move_to([xy[0], xy[1], tz + LIFT_DZ], grip=-1)
                robot.move_to([xy[0], xy[1], tz + HOVER_DZ], grip=-1)
                robot.observe()
                xy = wrist_update(robot, xy, tz, log)
                continue
            place_cycle(robot, xy, tz, log, done + 1)
            done += 1
            if done < 2:
                robot.move_to([xy[0], xy[1], tz + HOVER_DZ], grip=-1)
                robot.observe()
                xy = wrist_update(robot, xy, tz, log)
        if not robot.ended:
            robot.move_to([xy[0], xy[1], tz + HOVER_DZ], grip=1)
            robot.observe()
            rgb, pts = pc.live_points(robot, "front")
            b2 = pc.button_location(rgb, pts, tz)
            if b2 is not None and btn is not None and np.linalg.norm(b2["xy"] - btn["xy"]) < 0.05:
                btn = b2
            if btn is None:
                btn = b2
            log(f"button {btn}")
            if btn is not None:
                press_button(robot, btn, tz, log)
        log(f"finished: ended={robot.ended} complete={robot.complete}")
    except Ended:
        log(f"episode ended complete={robot.complete}")
    except Exception:
        log("fatal: " + traceback.format_exc())
