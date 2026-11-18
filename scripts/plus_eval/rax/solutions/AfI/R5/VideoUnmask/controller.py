"""VideoUnmask controller: remember where each coloured cube was covered in the
conditioning video, then lift the containers in the instructed colour order."""
import sys, os, re, json, math, time, io, base64, traceback
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from motion import Robot, packet, yaw_of, wrap, rotate_yaw_to
import perception as pc

SAFE_Z = 0.20
LOOK_Z = 0.16
GRASP_Z = 0.062
TOP_LO, TOP_HI = 0.058, 0.09
OUT = Path("/workspace/output")
LOGF = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    try:
        with open(OUT / "controller_log.txt", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {s}\n")
    except Exception:
        pass


def parse_order(instr):
    words = list(pc.COLORS.keys())
    found = []
    for m in re.finditer(r"\b(" + "|".join(words) + r")\b", instr.lower()):
        if m.group(1) not in found:
            found.append(m.group(1))
    return found


def decode_images(reply):
    p = packet(reply)
    out = {}
    for a in p.get("attachments", []):
        b = reply["content"][a["content_index"]]
        with Image.open(io.BytesIO(base64.b64decode(b["data"]))) as im:
            out[(a["camera"], a["kind"])] = np.array(im)
    return p, out


def observe_full(robot):
    rep = robot.ctx.call("robomme_observe")
    p, I = decode_images(rep)
    robot.last = p
    return p, I


def cam_points(p, I, cam):
    dep = I[(cam, "depth")].astype(np.float64) * p["depth"][cam].get("meters_per_unit", 0.001)
    P = pc.world_points(dep, p["camera_intrinsics"][cam]["matrix"], p["camera_poses"][cam])
    P[dep <= 0] = np.nan
    return P


def min_area_rect(xy):
    m = xy.mean(0)
    best = None
    for a in np.radians(np.arange(0, 90, 1.0)):
        R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        r = (xy - m) @ R.T
        lo, hi = np.percentile(r, 2, 0), np.percentile(r, 98, 0)
        ext = hi - lo
        ar = ext[0] * ext[1]
        if best is None or ar < best[0]:
            ctr = ((hi + lo) / 2) @ R + m
            best = (ar, a, ext, ctr)
    return best[3], best[1], best[2]


def locate_top_wrist(robot, est_xy, tag=""):
    """Find container top part near est_xy in the wrist depth. Returns (xy, edge_angle, ext)."""
    p, I = observe_full(robot)
    P = cam_points(p, I, "wrist")
    tops = [t for t in pc.container_tops(P, TOP_LO, TOP_HI, 60) if pc.is_container_top(t)]
    if not tops:
        log(f"[{tag}] wrist: no top component")
        return None
    t = min(tops, key=lambda t: np.linalg.norm(np.array(t["xy"]) - est_xy))
    d = float(np.linalg.norm(np.array(t["xy"]) - est_xy))
    log(f"[{tag}] wrist top: ctr={np.round(t['xy'],4)} ang={math.degrees(t['ang']):.1f} ext={np.round(t['ext'],3)} n={t['n']} dist={d:.3f}")
    if d > 0.05:
        log(f"[{tag}] wrist component too far from estimate")
        return None
    return np.array(t["xy"]), t["ang"], t["ext"]


def wrist_tops_now(robot):
    p, I = observe_full(robot)
    P = cam_points(p, I, "wrist")
    return [t for t in pc.container_tops(P, TOP_LO, TOP_HI, 30) if pc.is_container_top(t)]


def choose_yaw(ctr, ang, yaw_now, others, yaw_ref=0.0):
    """Edges at ang+k*90deg. Finger axis at gripper yaw psi is (-sin psi, cos psi).
    Pick the candidate maximising finger clearance to other containers, then small rotation."""
    best = None
    for k in range(-4, 5):
        psi = wrap(ang + k * math.pi / 2)
        rot = abs(wrap(psi - yaw_now))
        if rot > math.radians(100) or abs(wrap(psi - yaw_ref)) > math.radians(95):
            continue
        u = np.array([-math.sin(psi), math.cos(psi)])
        clear = 1.0
        for o in others:
            o = np.array(o)
            if np.linalg.norm(o - ctr) < 0.02:
                continue
            for sgn in (1, -1):
                clear = min(clear, float(np.linalg.norm(o - (ctr + sgn * 0.045 * u))))
        score = min(clear, 0.07) - 0.01 * rot
        if best is None or score > best[0]:
            best = (score, psi, clear)
    return best[1], best[2]


def front_containers(robot):
    p, I = observe_full(robot)
    img = I[("front", "rgb")][..., :3]
    dep = I[("front", "depth")].astype(np.float64) * 0.001
    cl = pc.white_clusters(img, dep, p["camera_intrinsics"]["front"]["matrix"], p["camera_poses"]["front"])
    return cl


def holding(robot):
    s = robot.state()
    w = s[6] + s[7]
    return 0.012 < w < 0.075, w


def pick_container(robot, est_xy, tag, yaw_ref, others=()):
    est_xy = np.array(est_xy, float)
    for attempt in range(3):
        if robot.ended:
            return False
        robot.move_to([est_xy[0], est_xy[1], SAFE_Z], -1, max_steps=40)
        robot.move_to([est_xy[0], est_xy[1], LOOK_Z], -1, max_steps=20)
        r = locate_top_wrist(robot, est_xy, tag)
        if r is None:
            # look from higher for a wider view
            robot.move_to([est_xy[0], est_xy[1], 0.22], -1, max_steps=20)
            r = locate_top_wrist(robot, est_xy, tag)
            if r is None:
                log(f"[{tag}] cannot see container; using estimate")
                r = (est_xy, 0.0, None)
        ctr, ang, ext = r
        # re-centre once and re-measure for accuracy
        robot.move_to([ctr[0], ctr[1], LOOK_Z], -1, max_steps=20)
        r2 = locate_top_wrist(robot, ctr, tag)
        if r2 is not None and np.linalg.norm(r2[0] - ctr) < 0.04:
            ctr, ang, ext = r2
        # choose yaw: edges at ang + k*90deg; pick nearest to yaw_ref (start yaw)
        tgt, clear = choose_yaw(ctr, ang, yaw_of(robot.state()), others, yaw_ref)
        log(f"[{tag}] finger clearance {clear:.3f}")
        robot.move_to([ctr[0], ctr[1], 0.12], -1, max_steps=20)
        y = rotate_yaw_to(robot, tgt, -1, max_iter=8)
        log(f"[{tag}] yaw target {math.degrees(tgt):.1f} achieved {math.degrees(y):.1f}")
        robot.move_to([ctr[0], ctr[1], 0.12], -1, max_steps=10)
        robot.move_to([ctr[0], ctr[1], GRASP_Z], -1, max_steps=20, tol=0.003)
        robot.step([[0, 0, 0, 0, 0, 0, 1]] * 12)
        robot.move_to([ctr[0], ctr[1], SAFE_Z], 1, max_steps=20)
        robot.observe()
        ok, w = holding(robot)
        log(f"[{tag}] attempt {attempt} lift: holding={ok} width={w:.4f} state={np.round(robot.state(),3).tolist()} success={robot.success}")
        if ok or robot.success:
            return ctr
        # failed: open, re-estimate from current front view
        robot.step([[0, 0, 0, 0, 0, 0, -1]] * 6)
        est_xy = np.array(ctr)
    return None


def place_back(robot, xy, tag):
    robot.move_to([xy[0], xy[1], GRASP_Z + 0.012], 1, max_steps=30, tol=0.004)
    robot.step([[0, 0, 0, 0, 0, 0, -1]] * 8)
    robot.move_to([xy[0], xy[1], SAFE_Z], -1, max_steps=20)
    log(f"[{tag}] placed back; state={np.round(robot.state(),3).tolist()}")


def parse_targets(instr):
    """Ordered list of (colour, plural)."""
    words = list(pc.COLORS.keys()) + ["black"]
    out = []
    for m in re.finditer(r"\b(" + "|".join(words) + r")\b(\s+cubes?)?", instr.lower()):
        col = m.group(1)
        plural = bool(m.group(2) and m.group(2).strip() == "cubes")
        if col not in [o[0] for o in out]:
            out.append((col, plural))
    return out


def plan_targets(targets, inst, tops, log):
    plan = []
    used = set()
    for col, plural in targets:
        cands = [i for i in inst if i["color"] == col and i["ndet"] >= 3]
        if not cands:
            cands = [i for i in inst if i["color"] == col]
        if not plural and len(cands) > 1:
            cands = [max(cands, key=lambda i: i["ndet"])]
        group = []
        for i in cands:
            xy = np.array(i["xy"])
            if tops:
                j = min(range(len(tops)), key=lambda j: np.linalg.norm(np.array(tops[j]["xy"]) - xy))
                d = float(np.linalg.norm(np.array(tops[j]["xy"]) - xy))
                if d < 0.045 and j not in used:
                    used.add(j)
                    group.append({"color": col, "xy": tops[j]["xy"], "cube": i["xy"], "d": d})
                    continue
                log(f"[{col}] cube {np.round(xy,3)} has no unique container (d={d:.3f}); using cube xy")
            group.append({"color": col, "xy": [float(xy[0]), float(xy[1])], "cube": i["xy"], "d": None})
        if not group:
            log(f"[{col}] NO cube instance seen in video")
        plan.append((col, group))
    return plan


def main(ctx):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    robot = Robot(ctx, log)
    try:
        p0 = robot.observe()
        instr = p0.get("instruction", "")
        targets = parse_targets(instr)
        log("instruction:", instr, "targets:", targets)
        man = None
        try:
            man = pc.load_manifest(p0["frame_sequence"]["manifest_path"])
        except Exception as e:
            log("manifest from packet failed", repr(e))
            import glob
            cands = sorted(glob.glob("/workspace/runtime/autonomous_controller/runs/*/frames/sequence_000000/manifest.json"), key=os.path.getmtime)
            if cands:
                man = json.loads(Path(cands[-1]).read_text())
        inst, _ = pc.cube_instances(man, log)
        for i in inst:
            log("cube", json.dumps(i))
        tops = wrist_tops_now(robot)
        log("containers(wrist step0):", json.dumps([[round(x, 3) for x in t["xy"]] for t in tops]))
        plan = plan_targets(targets, inst, tops, log)
        allxy = [t["xy"] for t in tops]
        yaw_ref = yaw_of(robot.state())
        seq = [g for _, grp in plan for g in grp]
        log("plan:", json.dumps(seq))
        log(f"setup t={time.time()-t0:.1f}s")
        for k, g in enumerate(seq):
            if robot.ended:
                break
            tag = f"{g['color']}#{k}"
            got = pick_container(robot, np.array(g["xy"]), tag, yaw_ref, allxy)
            if got is None:
                log(f"[{tag}] pick failed")
                continue
            if robot.ended:
                break
            if k < len(seq) - 1:
                place_back(robot, got, tag)
            log(f"[{tag}] done t={time.time()-t0:.1f}s step={robot.last.get('step_id')} rem={robot.remaining()}")
        robot.observe()
        log("final: ended", robot.ended, "success", robot.success, "step", (robot.last or {}).get("step_id"))
    except Exception as e:
        log("ERROR", repr(e), traceback.format_exc())
