import sys, os, json, math, time, re, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
from percep import *
from motion import *

OUT = Path("/workspace/output"); OUT.mkdir(parents=True, exist_ok=True)
LOGF = open(OUT/"vu_log.txt", "a")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOGF.write(s+"\n"); LOGF.flush()

COLORS = ("red", "green", "blue")
HOVER_Z = 0.17
GRASP_DROP = 0.035   # grasp TCP z = container top - this
LIFT_Z = 0.20

def parse_instruction(text):
    t = text.lower()
    seq = []
    for m in re.finditer(r"(all containers hiding|the (?:container|one) hiding)[^,.]*?\b(red|green|blue)\b", t):
        seq.append((m.group(2), m.group(1).startswith("all")))
    if not seq:
        for m in re.finditer(r"\b(red|green|blue)\b", t):
            if m.group(1) not in [s[0] for s in seq]: seq.append((m.group(1), False))
    return seq

def cube_clusters(manifest):
    dets = []
    for fr in manifest["frames"]:
        if fr.get("phase") != "video_demo": continue
        try:
            for d in detect_cubes_frame(fr):
                d["frame"] = fr["frame_index"]; dets.append(d)
        except Exception as e:
            log("frame det error", fr.get("frame_index"), e)
    clusters = []
    for d in dets:
        if not (-0.45 < d["xyz"][0] < 0.35 and abs(d["xyz"][1]) < 0.45 and 0.008 < d["xyz"][2] < 0.05): continue
        xy = np.array(d["xyz"][:2])
        best = None
        for c in clusters:
            if c["color"] != d["color"]: continue
            dist = np.linalg.norm(c["xy"] - xy)
            if dist < 0.03 and (best is None or dist < best[0]): best = (dist, c)
        if best:
            c = best[1]; c["pts"].append(xy); c["frames"].add(d["frame"]); c["cams"].add(d["cam"])
            c["xy"] = np.mean(c["pts"], axis=0)
        else:
            clusters.append({"color": d["color"], "pts": [xy], "xy": xy, "frames": {d["frame"]}, "cams": {d["cam"]}})
    out = []
    for c in clusters:
        if len(c["frames"]) < 3: continue
        out.append({"color": c["color"], "xy": c["xy"].tolist(), "nframes": len(c["frames"]), "cams": sorted(c["cams"])})
    return out

def containers_from_obs(meta, imgs, cams=("front", "wrist")):
    allc = []
    for cam in cams:
        if (cam, "depth") not in imgs: continue
        cs = detect_containers_geom(imgs[(cam, "rgb")], imgs[(cam, "depth")],
                               meta["camera_intrinsics"][cam]["matrix"], meta["camera_poses"][cam])
        for c in cs:
            e = c["ext"]
            if not (0.03 <= min(e) and max(e) <= 0.052 and 0.06 < c["ztop"] < 0.09): continue
            if not (-0.45 < c["xy"][0] < 0.35): continue
            c["cam"] = cam; allc.append(c)
    merged = []
    for c in sorted(allc, key=lambda c: 0 if c["cam"] == "wrist" else 1):
        for m in merged:
            if np.linalg.norm(np.array(m["xy"]) - np.array(c["xy"])) < 0.025:
                m["seen"].append(c["cam"]); break
        else:
            c["seen"] = [c["cam"]]; merged.append(c)
    return merged

def plan_targets(seq, clusters, conts):
    plan = []; used = set()
    for color, is_all in seq:
        cand = []
        for cl in clusters:
            if cl["color"] != color: continue
            best = None
            for i, c in enumerate(conts):
                d = np.linalg.norm(np.array(c["xy"]) - np.array(cl["xy"]))
                if best is None or d < best[0]: best = (d, i)
            if best and best[0] < 0.05 and best[1] not in used:
                cand.append((best[1], best[0], cl["nframes"]))
        # dedupe
        seen = {}
        for i, d, n in cand:
            if i not in seen or seen[i][0] > d: seen[i] = (d, n)
        items = sorted(seen.items(), key=lambda kv: (-kv[1][1], kv[1][0]))
        if not is_all: items = items[:1]
        for i, _ in items:
            used.add(i); plan.append((color, i))
    return plan

def refine_container(robot, xy_guess):
    meta, imgs = robot.observe()
    cs = containers_from_obs(meta, imgs, cams=("wrist",))
    best = None
    for c in cs:
        d = np.linalg.norm(np.array(c["xy"]) - np.array(xy_guess))
        if best is None or d < best[0]: best = (d, c)
    if best and best[0] < 0.04: return best[1]
    return None

def choose_yaw(xy, cyaw, others):
    """pick the grasp yaw (square symmetry) whose finger axis has most clearance to neighbours"""
    base = wrap_q(cyaw)
    best = None
    for y in (base, base + math.pi/2 if base < 0 else base - math.pi/2):
        ax = np.array([math.sin(y), -math.cos(y)]); pe = np.array([math.cos(y), math.sin(y)])
        clear = 1.0
        for o in others:
            d = np.array(o) - xy
            if np.linalg.norm(d) < 0.02: continue
            along = abs(d @ ax); perp = abs(d @ pe)
            if perp < 0.045:
                clear = min(clear, along)
        score = clear - 0.002*abs(y)
        if best is None or score > best[0]: best = (score, y, clear)
    return best[1], best[2]

def pick_one(robot, cont, others=(), last=False):
    xy = np.array(cont["xy"]); ztop = cont["ztop"]
    yaw, clear = choose_yaw(xy, cont["yaw"], others)
    log("yaw choice", round(math.degrees(yaw), 1), "clearance", round(clear, 3))
    log("pick target", xy.round(3), round(ztop, 3), round(math.degrees(yaw), 1))
    robot.move_to([xy[0], xy[1], HOVER_Z], yaw, -1, tol=0.006, max_steps=60, vmax=0.8)
    # refine from wrist (camera is offset from TCP; compute from its pose)
    c2 = refine_container(robot, xy)
    if c2 is not None:
        log("refined", np.array(c2["xy"]).round(3), round(c2["ztop"], 3), round(math.degrees(c2["yaw"]), 1))
        xy = np.array(c2["xy"]); ztop = c2["ztop"]
        yaw, clear = choose_yaw(xy, c2["yaw"], others)
        log("yaw choice", round(math.degrees(yaw), 1), "clearance", round(clear, 3))
        robot.move_to([xy[0], xy[1], HOVER_Z], yaw, -1, tol=0.004, max_steps=30)
    zg = ztop - GRASP_DROP
    for attempt in range(2):
        robot.move_to([xy[0], xy[1], zg + 0.04], yaw, -1, tol=0.005, max_steps=40, vmax=0.8)
        robot.move_to([xy[0], xy[1], zg], yaw, -1, tol=0.003, max_steps=40, vmax=0.4)
        robot.hold(1, 8)
        fing = robot.state[6] + robot.state[7]
        log("closed fingers", round(fing, 4), "pos", robot.state[:3].round(3), "attempt", attempt)
        if fing > 0.02 or robot.ended: break
        robot.hold(-1, 5)
        robot.move_to([xy[0], xy[1], HOVER_Z], yaw, -1, tol=0.006, max_steps=40, vmax=0.8)
        c2 = refine_container(robot, xy)
        if c2 is not None:
            xy = np.array(c2["xy"]); ztop = c2["ztop"]; zg = ztop - GRASP_DROP
            yaw, clear = choose_yaw(xy, c2["yaw"], others)
            log("re-refined", xy.round(3), round(math.degrees(yaw), 1))
    robot.move_to([xy[0], xy[1], LIFT_Z], yaw, 1, tol=0.01, max_steps=50, vmax=0.6)
    fing = robot.state[6] + robot.state[7]
    log("lifted", robot.state[:3].round(3), "fingers", round(fing, 4), "complete", robot.complete, "ended", robot.ended)
    if robot.ended: return True
    robot.hold(1, 4, yaw=yaw)
    if robot.ended: return True
    # put back
    robot.move_to([xy[0], xy[1], zg + 0.006], yaw, 1, tol=0.004, max_steps=50, vmax=0.6)
    robot.hold(-1, 6)
    robot.move_to([xy[0], xy[1], HOVER_Z], yaw, -1, tol=0.01, max_steps=40, vmax=0.8)
    return fing > 0.02

def main(ctx):
    try:
        return _main(ctx)
    except Exception:
        log("FATAL", traceback.format_exc())

def _main(ctx):
    t0 = time.time()
    robot = Robot(ctx, log)
    meta, imgs = robot.observe()
    instr = meta.get("instruction", "")
    log("instruction:", instr)
    seq = parse_instruction(instr)
    log("seq", seq)
    clusters = []
    try:
        fs = meta.get("frame_sequence") or {}
        mp = fs.get("manifest_path")
        if not mp:
            import glob
            cands = []
            for f in glob.glob("/workspace/runtime/autonomous_controller/runs/*/frames/sequence_000000/manifest.json"):
                try:
                    mm = json.loads(Path(f).read_text())
                    if mm.get("transition", {}).get("episode_id") == meta.get("episode_id"): cands.append(f)
                except Exception: pass
            if cands: mp = os.path.relpath(cands[0], "/workspace")
            log("manifest fallback", mp)
        if mp:
            man = json.loads((Path("/workspace")/mp).read_text())
            clusters = cube_clusters(man)
    except Exception:
        log("manifest error", traceback.format_exc())
    for c in clusters: log("cube", c["color"], np.round(c["xy"], 3), c["nframes"], c["cams"])
    conts = containers_from_obs(meta, imgs)
    for i, c in enumerate(conts): log("cont", i, np.round(c["xy"], 3), round(c["ztop"], 3), round(math.degrees(c["yaw"]), 1), c["seen"])
    plan = plan_targets(seq, clusters, conts)
    log("plan", [(col, i, np.round(conts[i]["xy"], 3).tolist()) for col, i in plan], "t", round(time.time()-t0, 1))
    if os.environ.get("VU_DRY"): return plan
    for k, (col, i) in enumerate(plan):
        if robot.ended: break
        log("== picking", col, i, "step", robot.step_id, "remaining", robot.remaining)
        others = [c["xy"] for j, c in enumerate(conts) if j != i]
        try:
            pick_one(robot, conts[i], others, last=(k == len(plan)-1))
        except Exception:
            log("pick error", traceback.format_exc())
            if robot.ended: break
    log("done; ended", robot.ended, "complete", robot.complete, "step", robot.step_id, "t", round(time.time()-t0, 1))
