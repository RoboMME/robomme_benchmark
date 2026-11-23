"""VideoUnmask controller: remember which containers cover which coloured cubes in the
conditioning video, then lift them in the instructed colour order (lift, set back, release)."""
import base64
import io
import json
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perception import (WS, backproject, cubes_from_video, detect_containers, frame_points,
                        cluster_points)
from robot import Robot, EpisodeEnded

OUT = Path("/workspace/output")
COLORS = ("red", "green", "blue")
DEFAULT_ZTOP = 0.073
GRASP_BELOW_TOP = 0.025
HOVER_ABOVE_TOP = 0.13
LIFT_ABOVE_TOP = 0.10


class Log:
    def __init__(self):
        self.lines = []
        try:
            OUT.mkdir(parents=True, exist_ok=True)
            self.f = open(OUT / "controller_log.txt", "a")
        except Exception:
            self.f = None
        self.t0 = time.time()

    def __call__(self, *a):
        s = f"[{time.time() - self.t0:7.2f}] " + " ".join(str(x) for x in a)
        print(s, flush=True)
        if self.f:
            self.f.write(s + "\n"); self.f.flush()


def parse_plural(instruction):
    """colour -> True if the instruction asks for all containers of that colour."""
    text = instruction.lower()
    res = {}
    for col in COLORS:
        if re.search(r"all containers hiding %s" % col, text) or re.search(r"%s cubes" % col, text):
            res[col] = True
        elif re.search(r"%s cube\b" % col, text):
            res[col] = False
    return res


def parse_order(instruction):
    text = instruction.lower()
    # Only the part after "then" names the pick order.
    if "then" in text:
        text = text.split("then", 1)[1]
    found = []
    for m in re.finditer(r"\b(red|green|blue)\b", text):
        if m.group(1) not in found:
            found.append(m.group(1))
    return found


def decode_obs(reply, meta):
    """Return {(camera, kind): array} from MCP image blocks."""
    out = {}
    blocks = reply.get("content", reply) if isinstance(reply, dict) else reply
    for a in meta.get("attachments", []):
        idx = a["content_index"]
        b = blocks[idx]
        data = base64.b64decode(b["data"])
        im = Image.open(io.BytesIO(data))
        arr = np.asarray(im)
        if a["kind"] == "rgb":
            arr = np.asarray(im.convert("RGB")).astype(np.float32)
        else:
            arr = arr.astype(np.float32) * meta["depth"][a["camera"]].get("meters_per_unit", 0.001)
        out[(a["camera"], a["kind"])] = arr
    return out


def live_containers(robot, cam, tz, log):
    reply = robot.ctx.call("robomme_observe")
    meta = robot._meta(reply)
    robot._update(meta)
    imgs = decode_obs(reply, meta)
    pts = backproject(imgs[(cam, "depth")], meta["camera_intrinsics"][cam]["matrix"], meta["camera_poses"][cam])
    return detect_containers(imgs[(cam, "rgb")], pts, tz, log=log)


def merge_containers(lists, radius=0.03):
    """lists ordered by priority; merge detections within radius, keep highest priority yaw/ztop."""
    merged = []
    for lst in lists:
        for c in lst:
            for m in merged:
                if np.linalg.norm(m["xy"] - c["xy"]) < radius:
                    m["votes"] += 1
                    break
            else:
                d = dict(c); d["votes"] = 1
                merged.append(d)
    return merged


def nearest(conts, xy):
    best, bd = None, 1e9
    for c in conts:
        d = np.linalg.norm(c["xy"] - np.asarray(xy[:2]))
        if d < bd:
            best, bd = c, d
    return best, bd


def plan_targets(order, cubes, conts, log, plural=None):
    targets = []
    used = []
    plural = plural or {}
    for col in order:
        ncol = 0
        for cube in cubes.get(col, []):  # sorted by detection count, most first
            if plural.get(col) is False and ncol >= 1:
                log(f"{col}: singular in instruction; extra cluster at {np.round(cube,3).tolist()} ignored")
                continue
            c, d = nearest(conts, cube)
            if c is not None and d < 0.06:
                tgt = {"color": col, "xy": np.array(c["xy"]), "ztop": c["ztop"], "yaw": c["yaw"], "src": "container", "dist": d}
            else:
                tgt = {"color": col, "xy": np.array(cube[:2]), "ztop": DEFAULT_ZTOP, "yaw": 0.0, "src": "cube", "dist": d}
            if any(np.linalg.norm(u - tgt["xy"]) < 0.02 for u in used):
                log(f"duplicate target for {col} at {tgt['xy']}, skipped")
                continue
            used.append(tgt["xy"])
            targets.append(tgt)
            ncol += 1
            log(f"target {col}: xy={np.round(tgt['xy'],3).tolist()} ztop={tgt['ztop']:.3f} yaw={np.degrees(tgt['yaw']):.1f} src={tgt['src']} d={d:.3f}")
    return targets


def refine_target(robot, tgt, tz, log):
    """From hover, re-detect containers in the wrist view and snap to the closest one."""
    try:
        conts = live_containers(robot, "wrist", tz, log=lambda *a: None)
        c, d = nearest(conts, tgt["xy"])
        if c is not None and d < 0.035:
            log(f"refine {tgt['color']}: shift {d:.3f} -> {np.round(c['xy'],3).tolist()} yaw {np.degrees(c['yaw']):.1f}")
            tgt = dict(tgt, xy=np.array(c["xy"]), ztop=c["ztop"], yaw=c["yaw"])
        else:
            log(f"refine {tgt['color']}: no close container (d={d:.3f})")
    except EpisodeEnded:
        raise
    except Exception as e:
        log("refine error", e)
    return tgt


def wrap_yaw(robot, yaw):
    """Choose the container-equivalent yaw (mod 90deg) closest to the current gripper yaw."""
    cur = robot.current_yaw()
    cands = [yaw + k * np.pi / 2 for k in range(-4, 5)]
    return min(cands, key=lambda a: abs((a - cur + np.pi) % (2 * np.pi) - np.pi))


def finger_clearance(xy, yaw, conts, reach=0.047):
    """Min distance from the two finger positions to other containers' centres."""
    ax = np.array([-np.sin(yaw), np.cos(yaw)])
    fingers = [np.asarray(xy) + reach * ax, np.asarray(xy) - reach * ax]
    best = 1e9
    for c in conts:
        if np.linalg.norm(c["xy"] - xy) < 0.02:
            continue
        for f in fingers:
            best = min(best, float(np.linalg.norm(c["xy"] - f)))
    return best


def choose_yaw(robot, tgt, conts, log):
    cands = [tgt["yaw"], tgt["yaw"] + np.pi / 2]
    scored = [(finger_clearance(tgt["xy"], a, conts), a) for a in cands]
    clr, a = max(scored)
    log(f"{tgt['color']} yaw choice: " + ", ".join(f"{np.degrees(y):.0f}deg clr={s:.3f}" for s, y in scored))
    # equivalent by 180deg symmetry: pick closest to current yaw
    cur = robot.current_yaw()
    opts = [a + k * np.pi for k in range(-2, 3)]
    return min(opts, key=lambda b: abs((b - cur + np.pi) % (2 * np.pi) - np.pi))


def pick_and_replace(robot, tgt, tz, log, conts=()):
    xy = tgt["xy"]
    hover_z = tgt["ztop"] + HOVER_ABOVE_TOP
    yaw = wrap_yaw(robot, tgt["yaw"])
    robot.move_to([xy[0], xy[1], hover_z], yaw, -1, tol=0.008, rot_tol=0.05, max_iters=70, max_lin=0.7)
    tgt = refine_target(robot, tgt, tz, log)
    xy = tgt["xy"]
    yaw = choose_yaw(robot, tgt, conts, log) if len(conts) else wrap_yaw(robot, tgt["yaw"])
    gz = tgt["ztop"] - GRASP_BELOW_TOP
    for attempt in range(3):
        if attempt == 2:
            yaw = yaw + np.pi / 2  # last try with the other face pair
        robot.move_to([xy[0], xy[1], hover_z], yaw, -1, tol=0.006, rot_tol=0.04, max_iters=25, max_lin=0.5)
        robot.move_to([xy[0], xy[1], gz], yaw, -1, tol=0.005, rot_tol=0.04, max_iters=30, max_lin=0.5)
        robot.hold(8, yaw, 1)
        w = robot.grip_open
        log(f"{tgt['color']} grasp attempt {attempt}: opening={w:.4f} pos={np.round(robot.pos,3).tolist()}")
        if w > 0.012:
            break
        # missed: open, rise, re-detect and retry
        robot.hold(5, yaw, -1)
        robot.move_to([xy[0], xy[1], hover_z], yaw, -1, tol=0.008, max_iters=20)
        tgt = refine_target(robot, tgt, tz, log)
        xy = tgt["xy"]
        yaw = choose_yaw(robot, tgt, conts, log) if len(conts) else wrap_yaw(robot, tgt["yaw"])
    lift_z = tgt["ztop"] + LIFT_ABOVE_TOP
    robot.move_to([xy[0], xy[1], lift_z], yaw, 1, tol=0.008, rot_tol=0.05, max_iters=25, max_lin=0.5)
    held = robot.grip_open > 0.012
    log(f"{tgt['color']} lifted: opening={robot.grip_open:.4f} held={held} pos={np.round(robot.pos,3).tolist()} steps_left={robot.remaining}")
    robot.hold(3, yaw, 1)
    # set it back down where it was and release
    robot.move_to([xy[0], xy[1], gz + 0.004], yaw, 1, tol=0.006, rot_tol=0.05, max_iters=25, max_lin=0.5)
    robot.hold(6, yaw, -1)
    robot.move_to([xy[0], xy[1], hover_z], yaw, -1, tol=0.01, rot_tol=0.08, max_iters=20, max_lin=0.7)
    return held


def main(ctx):
    log = Log()
    robot = Robot(ctx, log=log)
    try:
        reply = ctx.call("robomme_observe")
        meta = robot._meta(reply)
        robot._update(meta)
        instr = meta.get("instruction", "")
        log("instruction:", instr, "step", meta.get("step_id"), "remaining", robot.remaining)
        order = parse_order(instr)
        log("order:", order)
        seq = meta.get("frame_sequence") or {}
        mpath = WS / seq["manifest_path"]
        cubes, tz = cubes_from_video(mpath, log=log)
        man = json.loads(mpath.read_text())
        ex = [f for f in man["frames"] if f["phase"] == "execution"]
        last = ex[-1] if ex else man["frames"][-1]
        lists = []
        for cam in ("wrist", "front"):
            try:
                rgb, pts = frame_points(last, cam)
                log(f"containers from {cam}:")
                lists.append(detect_containers(rgb, pts, tz, log=log))
            except Exception as e:
                log("container detect error", cam, e)
        conts = merge_containers(lists)
        log(f"{len(conts)} merged containers")
        plural = parse_plural(instr)
        log("plural:", plural)
        targets = plan_targets(order, cubes, conts, log, plural=plural)
        for tgt in targets:
            if robot.remaining is not None and robot.remaining < 40:
                log("too few steps left"); break
            pick_and_replace(robot, tgt, tz, log, conts=conts)
        log("done; task_complete", robot.task_complete, "ended", robot.ended)
    except EpisodeEnded:
        log("episode ended; task_complete", robot.task_complete)
    except Exception:
        log("ERROR", traceback.format_exc())
    finally:
        try:
            if robot.last_meta is not None:
                log("final meta step", robot.last_meta.get("step_id"), "ended", robot.last_meta.get("episode_ended"),
                    "steps", json.dumps(robot.last_meta.get("steps"))[:500])
        except Exception:
            pass
