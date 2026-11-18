"""VideoUnmask controller v2 (multi-instance, front+wrist video).

1. Read the conditioning video (initial frame_sequence manifest): where each coloured cube was
   when its container covered it.
2. Parse target colours (in order) from the instruction.
3. Locate white containers in the first execution frame (front depth cloud).
4. For each target: map cube xy -> nearest container, centre the wrist camera above it, refine xy/yaw,
   grasp, lift (success is registered on lift), then put it back where it was.
"""
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import perception as P  # noqa: E402
import motion as M  # noqa: E402
import skills as S  # noqa: E402
import video as V  # noqa: E402

T0 = time.time()
WALL_LIMIT = 1050.0  # controller limit 1200 s


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def nearest_container(containers, xy, max_d=0.05, exclude=()):
    best, bd = None, 1e9
    for i, c in enumerate(containers):
        if i in exclude:
            continue
        d = float(np.linalg.norm(c["xy"] - np.asarray(xy)[:2]))
        if d < bd:
            best, bd = i, d
    return (best, bd) if bd <= max_d else (None, bd)


def order_greedy(items, start_xy):
    items = list(items)
    out = []
    cur = np.asarray(start_xy)[:2]
    while items:
        k = int(np.argmin([np.linalg.norm(it["xy"] - cur) for it in items]))
        it = items.pop(k)
        out.append(it)
        cur = it["xy"]
    return out


def build_plan(targets, inst, containers, start_xy):
    """List of dict(colour, xy, ztop, container) in execution order."""
    plan = []
    used = set()
    # containers explained by any seen cube (for guessing when a target colour is unseen)
    claimed = set()
    for col, lst in inst.items():
        for it in lst:
            i, _ = nearest_container(containers, it["xy"])
            if i is not None:
                claimed.add(i)
    cur = np.asarray(start_xy)[:2]
    for colour, plural in targets:
        lst = inst.get(colour, [])
        if not lst:
            free = [i for i in range(len(containers)) if i not in claimed and i not in used]
            if not free:
                free = [i for i in range(len(containers)) if i not in used]
            log(f"WARNING: no {colour} cube seen in video; guessing container {free[:1]}")
            if free:
                i = free[0]
                used.add(i)
                plan.append({"colour": colour, "xy": containers[i]["xy"], "ztop": containers[i]["ztop"],
                             "container": i, "guess": True})
            continue
        if not plural:
            lst = [max(lst, key=lambda t: t["n"])]
        for it in order_greedy(lst, cur):
            i, d = nearest_container(containers, it["xy"], exclude=used)
            if i is not None:
                used.add(i)
                item = {"colour": colour, "xy": containers[i]["xy"], "ztop": containers[i]["ztop"],
                        "container": i, "cube_xy": it["xy"], "match_d": d}
            else:
                item = {"colour": colour, "xy": it["xy"], "ztop": 0.073, "container": None,
                        "cube_xy": it["xy"], "match_d": d}
            plan.append(item)
            cur = item["xy"]
    return plan


def run(ctx):
    robot = M.Robot(ctx, log=log)
    obs = S.observe(robot)
    meta = obs["meta"]
    instr = meta.get("instruction", "")
    log("instruction:", instr)
    log("start state", np.round(robot.state, 3).tolist(), "limits", meta.get("episode_limits"))
    targets = V.parse_targets(instr)
    log("targets:", targets)

    # --- video: cube instances ---
    inst = {}
    try:
        mp = V.find_manifest(meta)
        log("manifest:", mp)
        if mp:
            dets, nfr = V.detections(mp, log=log)
            inst = V.instances(dets)
            log(f"video frames {nfr}, detections {len(dets)}; cube instances:")
            for col, lst in inst.items():
                for it in lst:
                    log(f"   {col:7s} xy {np.round(it['xy'], 3).tolist()} n {it['n']} frames "
                        f"{it['first_frame']}-{it['last_frame']} cams {it['cams']}")
    except Exception:
        log("video analysis failed:\n" + traceback.format_exc())

    # --- containers at execution start ---
    containers = []
    try:
        containers = P.table_containers(obs)
    except Exception:
        log("container detection failed:\n" + traceback.format_exc())
    log("containers:", [(np.round(c["xy"], 3).tolist(), round(c["ztop"], 3), sorted(c["cams"])) for c in containers])

    plan = build_plan(targets, inst, containers, robot.pos)
    log("plan:")
    for p in plan:
        log(f"   {p['colour']:7s} -> xy {np.round(p['xy'], 3).tolist()} container {p['container']} "
            f"match {p.get('match_d', -1):.3f}")
    log(f"setup done, wall {time.time() - T0:.1f}s")

    picked_xy = []
    for k, item in enumerate(plan):
        if robot.ended:
            break
        if time.time() - T0 > WALL_LIMIT:
            log("wall limit reached; stopping")
            break
        if any(np.linalg.norm(item["xy"] - q) < 0.03 for q in picked_xy):
            log(f"skip {item['colour']}: container already picked")
            continue
        t0, s0 = time.time(), robot.steps_used
        log(f"== pick {k + 1}/{len(plan)} {item['colour']} at {np.round(item['xy'], 3).tolist()} "
            f"steps_used {robot.steps_used} remaining {robot.remaining}")
        c = {"xy": item["xy"], "ztop": item["ztop"]}
        ok = pick_with_retries(robot, c)
        picked_xy.append(np.asarray(c.get("xy_refined", item["xy"])))
        if robot.ended:
            log(f"episode ended after lifting {item['colour']}")
            break
        if ok:
            S.place(robot, c.get("xy_refined", c["xy"]), c.get("ztop_refined", c["ztop"]), log=log)
        else:
            robot.set_gripper(-1.0, 8)
            robot.move_to([robot.pos[0], robot.pos[1], 0.2], speed=0.6, tol=0.01)
        log(f"   pick {k + 1} done: held={ok} steps {robot.steps_used - s0} wall {time.time() - t0:.1f}s")
    log("done. ended", robot.ended, "steps used", robot.steps_used, "remaining", robot.remaining)
    if not robot.ended:
        robot.hold(5)


def pick_with_retries(robot, c, attempts=3):
    xy = c["xy"]
    for a in range(attempts):
        if robot.ended:
            return True
        t, _ = S.refine_target(robot, xy, log=log)
        if t is None:
            t = {"xy": np.asarray(xy), "ztop": c["ztop"], "yaw": None}
        c["xy_refined"] = t["xy"]
        c["ztop_refined"] = t["ztop"]
        held = S.pick(robot, t, log=log)
        if robot.ended or held:
            return True
        log(f"pick attempt {a + 1} failed; reopening and retrying")
        robot.set_gripper(-1.0, 10)
        robot.move_to([robot.pos[0], robot.pos[1], 0.2], speed=0.6, tol=0.01)
        xy = t["xy"]
    return False


def main(ctx):
    try:
        run(ctx)
    except Exception:
        log("controller exception:\n" + traceback.format_exc())
    log("main returns")
