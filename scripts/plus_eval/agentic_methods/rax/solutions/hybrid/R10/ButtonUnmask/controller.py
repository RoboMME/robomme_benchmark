"""ButtonUnmask controller v1.
Press button before step 90 while watching cubes appear; then lift containers hiding
the instructed colors in order (grasp, lift, place back)."""
import json, math, re, sys, time, traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lib
from lib import Robot, log
import percep, skills

OUT = Path("/workspace/output")
KNOWN = ["red", "green", "blue", "yellow", "orange", "purple", "cyan"]
ALIASES = {"violet": "purple", "magenta": "purple", "pink": "purple"}


def parse_instruction(text):
    t = text.lower()
    t = t.split("press the button", 1)[-1]
    goals = []
    for m in re.finditer(r"(all\s+(?:the\s+)?containers?\s+)?hiding\s+(?:the\s+|a\s+|an\s+)?(\w+)\s+cube(s?)", t):
        col = ALIASES.get(m.group(2), m.group(2))
        multi = bool(m.group(1)) or m.group(3) == "s"
        goals.append({"color": col, "all": multi})
    return goals


VIEW_POSE = [-0.06, 0.0, 0.52]


def watch_and_press(rb, tracker, button):
    bxy = np.array(button["xy"]) * 0.5 + np.array(button["base_xy"]) * 0.5
    ztop = button["ztop"]
    # NOTE: measured arm speed is only ~0.010-0.016 m/step; press timing below is the
    # proven v2 schedule (press registered at step ~76; deadline 89).
    rb.move_to([bxy[0] + 0.03, bxy[1], 0.30], grip=1.0, keep_frames=True, max_steps=22, tol=0.006)
    log("high hover", rb.step_id, np.round(rb.state[:3], 4))
    while rb.step_id < 36 and not rb.ended:
        rb.hold(min(5, 36 - rb.step_id), 1.0, keep_frames=True)
    hover = 0.20
    rb.move_to([bxy[0], bxy[1], hover], grip=1.0, keep_frames=True, max_steps=14, tol=0.006)
    log("hover", rb.step_id, np.round(rb.state[:3], 4))
    while rb.step_id < 54 and not rb.ended:
        rb.hold(min(5, 54 - rb.step_id), 1.0, keep_frames=True)
    rb.move_to([bxy[0], bxy[1], ztop + 0.035], grip=1.0, keep_frames=True, max_steps=16, tol=0.004, vmax=0.6, chunk=4)
    log("pre-press", rb.step_id, np.round(rb.state[:3], 4))
    zs = []
    while rb.step_id < 86 and not rb.ended:
        p = rb.state[:3]
        dx = np.clip((bxy - p[:2]) / 0.05, -0.2, 0.2)
        rb.step([[dx[0], dx[1], -0.4, 0, 0, 0, 1.0]], keep_frames=True)
        zs.append(float(rb.state[2]))
        if len(zs) >= 3 and abs(zs[-1] - zs[-3]) < 0.0015 and zs[-1] < ztop:
            break
    log("pressed", rb.step_id, np.round(rb.state[:3], 4), "zs", [round(z, 4) for z in zs])
    rb.hold(2, 1.0, keep_frames=True)
    return zs


def watch_until_covered(rb, tracker, button_xy, min_step=140, max_step=330, quiet_needed=20):
    """Go to the high view pose and watch (wrist top-down + front) until no cube is visible
    for quiet_needed steps and the container count is stable."""
    rb.move_to([rb.state[0], rb.state[1], 0.30], grip=1.0, keep_frames=True, max_steps=10, tol=0.01)
    rb.move_to(VIEW_POSE, grip=1.0, keep_frames=True, max_steps=20, tol=0.01, yaw=0.0)
    last_vis = rb.step_id
    hist = []
    while not rb.ended and rb.step_id < max_step:
        rb.hold(5, 1.0, keep_frames=True)
        o = rb.last
        vis = percep.visible_cubes(o["rgb"]["wrist"], o["depth"]["wrist"], rb.meta, "wrist", button_xy)
        visf = percep.visible_cubes(o["rgb"]["front"], o["depth"]["front"], rb.meta, "front", button_xy)
        conts = percep.detect_containers_wrist(o["rgb"]["wrist"], o["depth"]["wrist"], rb.meta)
        hist.append((rb.step_id, len(vis), len(visf), len(conts)))
        if vis or visf:
            last_vis = rb.step_id
        stable = len(hist) >= 3 and len(set(h[3] for h in hist[-3:])) == 1
        if rb.step_id >= min_step and rb.step_id - last_vis >= quiet_needed and stable:
            break
    log("watch hist", hist)
    return hist


def match_targets(goals, cubes, conts):
    """Return ordered list of (goal, cube, container)."""
    plan = []
    used = set()
    for g in goals:
        cs = [c for c in cubes if c["color"] == g["color"]]
        if not cs:
            log("WARN no cube seen for", g)
            continue
        if not g["all"]:
            cs = sorted(cs, key=lambda c: -c["n"])[:1]
        for cube in cs:
            cand = [(np.linalg.norm(np.array(k["xy"]) - np.array(cube["xyz"][:2])), i) for i, k in enumerate(conts) if i not in used]
            if not cand:
                continue
            d, i = min(cand)
            if d > 0.035:
                log("WARN nearest container far", d, "use cube xy")
                plan.append({"goal": g, "cube": cube, "cont": {"xy": list(cube["xyz"][:2]), "yaw": 0.0}, "dist": float(d)})
                continue
            used.add(i)
            plan.append({"goal": g, "cube": cube, "cont": conts[i], "dist": float(d)})
    return plan


def choose_psi(cont, neighbors, reach=0.045):
    """Pick gripper yaw (psi or psi+-90deg) whose fingers stay farthest from neighbor containers."""
    psi0 = skills.grip_yaw(cont["yaw"])
    c = np.array(cont["xy"])
    others = [np.array(n["xy"]) for n in neighbors if np.linalg.norm(np.array(n["xy"]) - c) > 0.025]
    best = None
    for psi in (psi0, psi0 + math.pi / 2 if psi0 < 0 else psi0 - math.pi / 2):
        ax = np.array([-math.sin(psi), math.cos(psi)])  # finger axis (world y at psi=0)
        fingers = [c + reach * ax, c - reach * ax]
        clr = min([np.linalg.norm(f - o) for f in fingers for o in others] or [1.0])
        if best is None or clr > best[1] + 0.008:
            best = (psi, clr)
    return best[0], best[1]


def do_pick(rb, cont, cube_xy, neighbors, attempt=0):
    psi, clr = choose_psi(cont, neighbors)
    x, y = cont["xy"]
    rb.move_to([x, y, skills.SAFE_Z], grip=-1.0, yaw=psi, max_steps=40, tol=0.006)
    rb.hold(2, -1.0)
    try:
        ref = percep.refine_container_wrist(rb.last["rgb"]["wrist"], rb.last["depth"]["wrist"], rb.meta, [x, y])
    except Exception as e:
        ref = None
        log("refine error", repr(e))
    log("refine", ref)
    if ref is not None and 0.025 < min(ref["size"]) and max(ref["size"]) < 0.06:
        cont = dict(cont, xy=ref["xy"], yaw=ref["yaw"])
    psi, clr = choose_psi(cont, neighbors)
    log("psi", round(math.degrees(psi), 1), "clearance", round(clr, 3))
    r = skills.pick_and_replace(rb, cont, psi=psi)
    cont = dict(cont, psi=psi)
    return r, cont


def main(ctx):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rb = Robot(ctx, str(OUT))
    summary = {"version": "v3.1"}
    try:
        rb.observe()
        instr = rb.meta.get("instruction", "")
        goals = parse_instruction(instr)
        log("instruction", instr, "goals", goals)
        summary["goals"] = goals
        button = percep.detect_button(rb.last["rgb"]["front"], rb.last["depth"]["front"], rb.meta)
        if button is None:
            button = {"xy": [-0.19, -0.017], "base_xy": [-0.19, -0.017], "ztop": 0.034}
            log("WARN button not found; default")
        log("button", button)
        summary["button"] = button
        tracker = percep.CubeTracker()
        # include the step-0 frame
        tracker.add_frame({"step": 0, "rgb": rb.last["rgb"], "depth": rb.last["depth"], "meta": rb.meta})
        rb.on_frames = lambda frs: [tracker.add_frame(f) for f in frs]
        zs = watch_and_press(rb, tracker, button)
        summary["press_zs"] = zs
        summary["press_step"] = rb.step_id
        summary["watch"] = watch_until_covered(rb, tracker, button["base_xy"])
        rb.on_frames = None
        rb.frames = []
        cubes = tracker.cubes(exclude_xy=button["base_xy"])
        for c in cubes:
            log("cube", c["color"], np.round(c["xyz"], 3), c["n"], c["first"], c["last"])
        summary["cubes"] = cubes
        rb.observe()
        conts = percep.detect_containers_wrist(rb.last["rgb"]["wrist"], rb.last["depth"]["wrist"], rb.meta)
        fconts = percep.detect_containers(rb.last["rgb"]["front"], rb.last["depth"]["front"], rb.meta, np.array(button["base_xy"]))
        for f in fconts:
            if all(np.linalg.norm(np.array(f["xy"]) - np.array(c["xy"])) > 0.03 for c in conts):
                f["src"] = "front"; conts.append(f)
        for k in conts:
            log("container", k.get("src"), np.round(k["xy"], 3), round(math.degrees(k["yaw"])), np.round(k["size"], 3), k["n"])
        summary["containers"] = conts
        plan = match_targets(goals, cubes, conts)
        summary["plan"] = [{"color": p["goal"]["color"], "xy": p["cont"]["xy"], "dist": p["dist"]} for p in plan]
        log("plan", summary["plan"])
        results = []
        for k, p in enumerate(plan):
            if rb.ended:
                break
            cube_xy = p["cube"]["xyz"][:2]
            # re-detect container near expected location if visible
            cont = p["cont"]
            ok = False
            for attempt in range(3):
                r, cont_used = do_pick(rb, cont, cube_xy, conts, attempt)
                log("pick", p["goal"]["color"], "attempt", attempt, r)
                results.append({"color": p["goal"]["color"], "attempt": attempt, **{kk: vv for kk, vv in r.items() if kk != "psi"}})
                if rb.ended:
                    ok = r.get("holding", False)
                    break
                if r.get("holding"):
                    ok = True
                    break
                # missed: open, lift, re-detect from front
                rb.move_to([cont_used["xy"][0], cont_used["xy"][1], skills.SAFE_Z], grip=-1.0, max_steps=30, tol=0.01)
                rb.observe()
                cs = percep.detect_containers(rb.last["rgb"]["front"], rb.last["depth"]["front"], rb.meta, np.array(button["base_xy"]))
                if cs:
                    cont = min(cs, key=lambda c: np.linalg.norm(np.array(c["xy"]) - np.array(cube_xy)))
            if rb.ended:
                break
            skills.place_back(rb, cont_used["xy"], cont_used["psi"])
        summary["results"] = results
    except Exception as e:
        log("ERROR", repr(e), traceback.format_exc())
        summary["error"] = repr(e)
    summary["ended"] = rb.ended
    summary["complete"] = rb.complete
    summary["wall_s"] = time.time() - t0
    try:
        summary["final_step"] = rb.step_id
    except Exception:
        pass
    log("SUMMARY", json.dumps(summary, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))[:4000])
    try:
        (OUT / "summary.json").write_text(json.dumps(summary, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o), indent=1))
        (OUT / "log.txt").write_text("\n".join(lib.LOG))
    except Exception:
        pass
