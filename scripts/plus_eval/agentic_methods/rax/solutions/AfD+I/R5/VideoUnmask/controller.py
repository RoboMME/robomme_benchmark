"""VideoUnmask controller: remember which container covers which cube from the
conditioning video (RGB-D back-projection), then lift the containers in the
instructed order, putting non-final ones back."""
import json, os, re, sys, time, traceback
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from motion import Robot
from percept import video_cube_locations, container_tops, obs_world

OUT = "/workspace/output"
TARGET_COLORS = ["red", "green", "blue", "yellow", "orange", "purple", "cyan"]
SURVEY = [-0.06, 0.0, 0.40]

def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    try:
        with open(os.path.join(OUT, "controller_log.txt"), "a") as f: f.write(msg + "\n")
    except Exception:
        pass

def parse_instruction(text):
    """-> list of (color, all_flag) in order."""
    t = text.lower()
    t = t.split("then", 1)[1] if "then" in t else t
    parts = re.split(r",|followed by|and finally|finally|\bthen\b", t)
    plan = []
    for p in parts:
        cols = [c for c in TARGET_COLORS if re.search(r"\b" + c + r"\b", p)]
        if not cols: continue
        allf = bool(re.search(r"\ball\b", p)) or bool(re.search(r"cubes\b", p))
        plan.append((cols[0], allf))
    return plan

def survey(robot):
    robot.move_to(SURVEY, yaw=0.0, grip=-1, tol=0.012, max_iter=6)
    robot.observe()
    conts = []
    for cam in ("wrist", "front"):
        try:
            meta, rgb, Wm = obs_world(robot.raw, cam)
            Wm = Wm.copy(); bad = (Wm[..., 0] < -0.4) | (np.abs(Wm[..., 1]) > 0.45) | (Wm[..., 0] > 0.4)
            Wm[bad] = np.nan
            for c in container_tops(Wm, rgb=rgb):
                c["cam"] = cam
                if all(np.linalg.norm(c["xy"] - o["xy"]) > 0.025 for o in conts): conts.append(c)
        except Exception as e:
            log("survey error", cam, repr(e))
    log("survey containers:", [(np.round(c["xy"], 3).tolist(), c["yaw_deg"], c["cam"], c["n"]) for c in conts])
    return conts

def nearest(conts, xy, maxd=0.06):
    best = None
    for c in conts:
        d = float(np.linalg.norm(c["xy"] - xy))
        if d < maxd and (best is None or d < best[0]): best = (d, c)
    return best[1] if best else None

def main(ctx):
    try:
        os.makedirs(OUT, exist_ok=True)
    except Exception:
        pass
    try:
        _main(ctx)
    except Exception as e:
        log("FATAL", repr(e), traceback.format_exc())

def _main(ctx):
    t0 = time.time()
    robot = Robot(ctx, log=log)
    d = robot.observe()
    instr = d.get("instruction", "")
    log("instruction:", instr, "limit", d.get("episode_limits"))
    plan = parse_instruction(instr)
    log("plan:", plan)
    tracks = []
    try:
        mp = os.path.join("/workspace", d["frame_sequence"]["manifest_path"])
        tracks = video_cube_locations(mp)
    except Exception as e:
        log("video error", repr(e), traceback.format_exc())
    tracks = [t for t in tracks if t["n"] >= 3]
    log("tracks:", [(t["color"], np.round(t["xy"], 3).tolist(), t["n"], sorted(t["cams"])) for t in tracks])
    # Ordered list of cube positions to unmask
    seq = []
    for color, allf in plan:
        ts = sorted([t for t in tracks if t["color"] == color], key=lambda t: -t["n"])
        if not ts:
            log("WARNING no track for", color); continue
        chosen = ts if allf else ts[:1]
        for t in chosen: seq.append((color, np.array(t["xy"])))
    log("sequence:", [(c, np.round(x, 3).tolist()) for c, x in seq])
    conts = survey(robot)
    # assign containers (keep track of current container positions keyed by cube)
    for i, (color, cube_xy) in enumerate(seq):
        if robot.ended: break
        final = (i == len(seq) - 1)
        done = False
        for attempt in range(3):
            if robot.ended: break
            c = nearest(conts, cube_xy)
            if c is None:
                log("no container near", color, cube_xy.tolist(), "-> using cube xy")
                x, y, yaw = cube_xy[0], cube_xy[1], 0.0
            else:
                x, y, yaw = c["xy"][0], c["xy"][1], c["yaw_deg"]
            log(f"pick {i} {color} attempt {attempt} at {x:.3f},{y:.3f} yaw {yaw}")
            opening = robot.lift_container(x, y, np.deg2rad(yaw))
            log("  opening after lift", round(opening, 4), "step", robot.last.get("step_id"), "ended", robot.ended, "succ", robot.success)
            if robot.ended: break
            if opening > 0.012:
                done = True
                if not final:
                    robot.put_back(x, y, np.deg2rad(yaw))
                    cube_xy_new = np.array([x, y])
                break
            # missed: re-survey; container may have moved
            robot.gripper(-1, 6)
            robot.move_to([x, y, 0.15], yaw=np.deg2rad(yaw), grip=-1, tol=0.01, max_iter=4)
            conts = survey(robot)
        if robot.ended: break
        if not final:
            conts = survey(robot)
        if final and done and not robot.ended:
            # hold the container up; lift a bit more to make sure
            robot.move_to([x, y, 0.22], yaw=np.deg2rad(yaw), grip=1, tol=0.01, max_iter=3)
    log("finished: ended", robot.ended, "success", robot.success, "step", robot.last.get("step_id"), "elapsed", round(time.time() - t0, 1))
