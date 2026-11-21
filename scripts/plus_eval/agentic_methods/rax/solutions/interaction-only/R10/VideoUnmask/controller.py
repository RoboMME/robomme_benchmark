"""VideoUnmask controller v5 (depth-first cube detection).
Conditioning video (front+wrist, every frame) -> clustered coloured cube world positions.
For each colour in instruction order (all instances if 'all ... cubes'), view from above with the wrist
camera, detect container tops from depth, grasp the container over the cube with a yaw that keeps the
fingers clear of neighbours, lift, and put it back down before the next one."""
import math, time, traceback, os
import numpy as np
import lib
from lib import log

VIEW_Z = 0.20
CAM_OFF = np.array([0.046, 0.020])  # wrist camera xy offset from TCP at yaw 0
GRASP_Z = 0.045
PLACE_Z = 0.043
LIFT_Z = 0.20
MATCH_R = 0.05


def view_and_detect(r, xy):
    r.move_to([xy[0] - CAM_OFF[0], xy[1] - CAM_OFF[1], VIEW_Z], yaw=0.0, max_steps=60, tol=0.006)
    if r.ended:
        return [], None
    r.hold(2, -1.0)
    depth = r.imgs[("wrist", "depth")]
    dets = lib.detect_containers(r.meta, depth)
    dets = [d for d in dets if 0.025 < max(d["size"]) < 0.07 and min(d["size"]) > 0.012]
    P = lib.world_points(r.meta, depth, "wrist")
    log("view at", np.round(r.state[:3], 3).tolist(), "dets",
        [(np.round(d["xy"], 3).tolist(), round(math.degrees(d["ang"]), 1), d["npts"]) for d in dets])
    return dets, P


def choose_yaw(r, P, xy, box_ang):
    cur = r.yaw()
    best = None
    for k in range(4):
        a = box_ang + k * math.pi / 2
        a = cur + lib.wrap(a - cur)
        if abs(a) > 2.4:   # keep within a comfortable wrist range
            continue
        n, zmax = lib.finger_clearance(P, xy, a) if P is not None else (0, 0)
        cost = n + 30 * abs(lib.wrap(a - cur))
        log(f"  yaw cand {math.degrees(a):.0f} obstacles {n} zmax {zmax:.3f}")
        if best is None or cost < best[0]:
            best = (cost, a)
    if best is None:
        return lib.nearest_yaw(box_ang, cur)
    return best[1]


def pick_container(r, cube_xy, picked):
    """Lift the container over cube_xy. Returns its xy if lifted, 'ended' if the episode ended, None."""
    for attempt in range(3):
        if r.ended:
            return "ended"
        dets, P = view_and_detect(r, cube_xy)
        if r.ended:
            return "ended"
        rad = MATCH_R if attempt == 0 else MATCH_R + 0.02
        cand = [d for d in dets if np.linalg.norm(d["xy"] - cube_xy) < rad]
        if not cand:
            log("no container near", np.round(cube_xy, 3).tolist(), "attempt", attempt)
            continue
        d = min(cand, key=lambda d: np.linalg.norm(d["xy"] - cube_xy))
        xy = d["xy"]
        yaw = choose_yaw(r, P, xy, d["ang"])
        log("target container", np.round(xy, 4).tolist(), "dist", round(float(np.linalg.norm(xy - cube_xy)), 3),
            "yaw", round(math.degrees(yaw), 1))
        gz = float(np.clip(d["ztop"] - 0.0275, 0.03, 0.09))
        r.move_to([xy[0], xy[1], 0.15], yaw=yaw, max_steps=70)
        r.move_to([xy[0], xy[1], gz], yaw=yaw, max_steps=60, tol=0.003)
        if r.ended:
            return "ended"
        r.hold(8, 1.0)
        if r.ended:
            return "ended"
        log("closed opening", round(r.gripper_open(), 4), "z", round(r.state[2], 4))
        r.move_to([xy[0], xy[1], LIFT_Z], yaw=yaw, grip=1.0, max_steps=40)
        if r.ended:
            return "ended"
        op = r.gripper_open()
        log("lifted opening", round(op, 4), "step", r.meta.get("step_id"))
        if op > 0.03:
            return (xy, gz)
        log("grasp failed; reopening")
        r.hold(5, -1.0)
        cube_xy = xy
    return None


def place_back(r, xy, gz):
    yaw = r.yaw()
    r.move_to([xy[0], xy[1], 0.10], yaw=yaw, grip=1.0, max_steps=40)
    r.move_to([xy[0], xy[1], gz - 0.002], yaw=yaw, grip=1.0, max_steps=40, tol=0.003)
    r.hold(6, -1.0)
    r.move_to([xy[0], xy[1], LIFT_Z], yaw=yaw, grip=-1.0, max_steps=40)


def plan(goal, clusters):
    """Ordered list of (colour, xy)."""
    seq = []
    for color, want_all in goal:
        cs = [c for c in clusters if c["color"] == color and c["n_obs"] >= 3]
        # merge same-colour clusters closer than a container width (split observations of one cube)
        merged = []
        for c in sorted(cs, key=lambda c: -c["n_obs"]):
            if any(np.linalg.norm(c["xy"] - m["xy"]) < 0.035 for m in merged):
                continue
            merged.append(c)
        cs = merged
        if not cs:
            log("WARNING colour", color, "not seen in video")
            continue
        cs.sort(key=lambda c: -c["n_obs"])
        if not want_all:
            cs = cs[:1]
        # nearest-neighbour order starting from previous target
        prev = seq[-1][1] if seq else np.array([-0.06, 0.0])
        rest = list(range(len(cs)))
        while rest:
            j = min(rest, key=lambda k: np.linalg.norm(cs[k]["xy"] - prev))
            rest.remove(j)
            nxt = cs[j]
            seq.append((color, np.array(nxt["xy"])))
            prev = nxt["xy"]
    return seq


def run(ctx):
    t0 = time.time()
    r = lib.Robot(ctx)
    instr = r.meta.get("instruction", "")
    log("instruction:", instr)
    goal = lib.parse_goal(instr)
    log("goal:", goal)
    seq_desc = r.meta.get("frame_sequence") or {}
    try:
        clusters = lib.analyze_video_v3(seq_desc["manifest_path"])
    except Exception:
        log("analyze_video_v3 failed, falling back to v2", traceback.format_exc())
        clusters = lib.analyze_video_v2(seq_desc["manifest_path"])
    for c in clusters:
        log("cube", c["color"], np.round(c["xy"], 3).tolist(), "obs", c["n_obs"], "frames", c["first"], c["last"], c["cams"])
    seq = plan(goal, clusters)
    log("plan:", [(c, np.round(xy, 3).tolist()) for c, xy in seq], f"t={time.time()-t0:.1f}")
    picked = []
    for i, (c, cube_xy) in enumerate(seq):
        if r.ended:
            break
        log(f"=== target {i} {c} at {np.round(cube_xy, 3).tolist()} step {r.meta.get('step_id')} t={time.time()-t0:.1f}")
        res = pick_container(r, cube_xy, picked)
        if isinstance(res, str) or r.ended:
            break
        if res is None:
            log("failed to pick", c)
            continue
        picked.append(res[0])
        if i < len(seq) - 1:
            place_back(r, res[0], res[1])
    if not r.ended:
        r.hold(10, 1.0)
    log("done: ended", r.ended, "step", r.meta.get("step_id"), "t", round(time.time() - t0, 1))


def main(ctx):
    try:
        run(ctx)
    except Exception:
        log("CONTROLLER EXCEPTION", traceback.format_exc())
    finally:
        try:
            os.makedirs("/workspace/output", exist_ok=True)
            with open("/workspace/output/controller_log.txt", "w") as f:
                f.write("\n".join(lib.LOG))
        except Exception:
            pass
