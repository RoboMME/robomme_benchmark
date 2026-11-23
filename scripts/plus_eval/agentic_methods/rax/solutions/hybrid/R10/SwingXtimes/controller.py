"""SwingXtimes autonomous controller.

Pick up the instructed cube, swing it N times between the two targets in the
instructed order, put it down and press the button.
"""
import json, os, re, sys, time, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robo
from motion import Robot, log

TRAVEL_Z = 0.12
SWING_Z = 0.085
GRASP_Z = 0.02
WORDS = {"once": 1, "twice": 2, "thrice": 3, "one": 1, "two": 2, "three": 3, "four": 4,
         "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12}


def parse_instruction(text):
    t = text.lower()
    color = "red"
    m = re.search(r"pick up the (\w+) cube", t)
    if m:
        color = m.group(1)
    else:
        for c in ("red", "green", "blue"):
            if c in t:
                color = c
                break
    first = "right"
    m = re.search(r"to the (left|right)[- ]side target", t)
    if m:
        first = m.group(1)
    n = None
    m = re.search(r"motion\s+(\w+)\s+times?", t) or re.search(r"(\w+)\s+times", t)
    if m:
        w = m.group(1)
        n = int(w) if w.isdigit() else WORDS.get(w)
    if n is None:
        for w, v in WORDS.items():
            if re.search(r"\b%s\b" % w, t):
                n = v
                break
    if n is None:
        n = 1
    return color, first, n


def perceive_scene(meta, rgb, depth, color):
    """Locate cube, two targets and button in the front view (world coordinates)."""
    P = robo.world_points(meta, depth, "front")
    img = rgb["front"]
    out = {}
    # cube
    best = None
    for cc in robo.components(robo.color_mask(img, color), min_px=8):
        pts = P[cc["vs"], cc["us"]]
        pts = pts[~np.isnan(pts[:, 0])]
        if len(pts) < 5:
            continue
        zt = np.max(pts[:, 2])
        if zt < 0.02 or zt > 0.09:
            continue
        top = pts[pts[:, 2] > zt - 0.008]
        best = {"xyz": top.mean(0), "ztop": zt, "uv": (cc["u"], cc["v"]), "n": cc["n"]}
        break
    out["cube"] = best
    # white things: targets (flat discs) and button (square with cap)
    whites = []
    for cc in robo.components(robo.color_mask(img, "white"), min_px=15):
        pts = P[cc["vs"], cc["us"]]
        pts = pts[~np.isnan(pts[:, 0])]
        if len(pts) < 10:
            continue
        c = np.median(pts, 0)
        # neighbourhood height (cap of button rises above)
        nb = (np.abs(P[..., 0] - c[0]) < 0.025) & (np.abs(P[..., 1] - c[1]) < 0.025)
        zmax = np.nanmax(P[..., 2][nb]) if nb.any() else c[2]
        whites.append({"xyz": c, "n": cc["n"], "zmax": float(zmax), "uv": (cc["u"], cc["v"]),
                       "pts": pts})
    # merge components closer than 4 cm (rings of one target can split)
    merged = []
    whites = [w for w in whites if -0.01 < w["xyz"][2] < 0.05]
    for w in sorted(whites, key=lambda w: -w["n"]):
        for m in merged:
            if np.linalg.norm(m["xyz"][:2] - w["xyz"][:2]) < 0.04:
                m["n"] += w["n"]
                m["zmax"] = max(m["zmax"], w["zmax"])
                break
        else:
            merged.append(dict(w))
    merged = [m for m in merged if -0.01 < m["xyz"][2] < 0.05 and -0.6 < m["xyz"][0] < 0.3 and abs(m["xyz"][1]) < 0.45]
    log("white candidates", [(m["xyz"].round(3).tolist(), m["n"], round(m["zmax"], 3)) for m in merged])
    button = None
    cands = sorted(merged, key=lambda m: -(m["xyz"][2] + 0.3 * min(m["zmax"], 0.04)))
    if cands and cands[0]["xyz"][2] > 0.012:
        button = cands[0]
        rest = [m for m in merged if m is not button]
    else:
        rest = merged
    rest = sorted(rest, key=lambda m: -m["n"])[:2]
    out["button"] = button
    if len(rest) == 2:
        a, b = sorted(rest, key=lambda m: m["xyz"][1])  # a: smaller y = robot's right
        out["right"], out["left"] = a, b
    else:
        out["right"] = out["left"] = None
    return out


def refine_cube_wrist(rob, color, guess):
    meta, rgb, dep = rob.observe()
    P = robo.world_points(meta, dep, "wrist")
    best = None
    for cc in robo.components(robo.color_mask(rgb["wrist"], color), min_px=30):
        pts = P[cc["vs"], cc["us"]]
        pts = pts[~np.isnan(pts[:, 0])]
        if len(pts) < 20:
            continue
        zt = np.max(pts[:, 2])
        top = pts[pts[:, 2] > zt - 0.006]
        c = top.mean(0)
        d = np.linalg.norm(c[:2] - guess[:2])
        if d < 0.08 and 0.02 < zt < 0.09 and (best is None or d < best[0]):
            best = (d, c, zt, top)
    if best is None:
        return None
    _, c, zt, top = best
    # yaw of the top face (mod 90 deg) from the second moments of xy points
    xy = top[:, :2] - top[:, :2].mean(0)
    yaw = 0.0
    if len(xy) > 30:
        # minimum-area rectangle via search over angles
        angs = np.radians(np.arange(0, 90, 2))
        areas = []
        for a in angs:
            R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
            q = xy @ R.T
            areas.append(np.ptp(q[:, 0]) * np.ptp(q[:, 1]))
        yaw = float(angs[int(np.argmin(areas))])
        if yaw > np.pi / 4:
            yaw -= np.pi / 2
    return {"xyz": c, "ztop": zt, "yaw": yaw}


def held(rob):
    return rob.state[6] + rob.state[7] > 0.02


def grasp_cube(rob, color, guess, tries=3):
    for k in range(tries):
        if rob.ended:
            return False
        rob.move_to([guess[0], guess[1], TRAVEL_Z], grip=-1, max_v=0.8, tol=0.006)
        rob.hold(3)
        ref = refine_cube_wrist(rob, color, guess)
        if ref is not None:
            guess = ref["xyz"]
            log("wrist refine", ref["xyz"].round(4).tolist(), "ztop", round(float(ref["ztop"]), 3))
            rob.move_to([guess[0], guess[1], TRAVEL_Z], grip=-1, max_v=0.4, tol=0.004)
        gz = max(0.012, (ref["ztop"] if ref else 0.04) - 0.02)
        rob.move_to([guess[0], guess[1], gz + 0.05], grip=-1, max_v=0.5, tol=0.005)
        rob.move_to([guess[0], guess[1], gz], grip=-1, max_v=0.25, tol=0.004, max_steps=40)
        rob.gripper(1, steps=10)
        rob.move_to([rob.state[0], rob.state[1], TRAVEL_Z], grip=1, max_v=0.4, tol=0.006)
        rob.hold(2)
        log("grasp try", k, "fingers", rob.state[6:].round(4).tolist())
        if held(rob):
            return True
        rob.gripper(-1, steps=6)
        # re-perceive from front
        meta, rgb, dep = rob.observe()
        sc = perceive_scene(meta, rgb, dep, color)
        if sc["cube"] is not None:
            guess = sc["cube"]["xyz"]
    return False


def strong_red(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (r > 140) & (g < 90) & (b < 90) & (r - g > 90)


def find_targets(meta, rgb, depth, cam, avoid):
    """Low flat target discs (achromatic or highlighted red) in one camera, world coords."""
    P = robo.world_points(meta, depth, cam)
    z = P[..., 2]
    img = rgb[cam]
    red = strong_red(img)
    mask = (robo.color_mask(img, "white") | red) & (z > -0.01) & (z < 0.02)
    out = []
    for cc in robo.components(mask, min_px=12):
        pts = P[cc["vs"], cc["us"]]
        ok = ~np.isnan(pts[:, 0])
        if ok.sum() < 10:
            continue
        c = np.median(pts[ok], 0)
        if any(np.linalg.norm(c[:2] - a[:2]) < 0.045 for a in avoid):
            continue
        hl = float(red[cc["vs"], cc["us"]].mean())
        if c[2] < 0.004:
            continue  # table-level light blob, not a raised target disc
        ext = np.ptp(pts[ok][:, :2], 0)
        out.append({"xyz": c, "n": cc["n"], "hl": hl, "cam": cam, "ext": ext,
                    "uv": (cc["u"], cc["v"]),
                    "edge": bool(cc["bbox"][0] <= 1 or cc["bbox"][1] <= 1 or cc["bbox"][2] >= 254 or cc["bbox"][3] >= 254)})
    return out


def locate_target(rob, last, other, button):
    """Observe and return (xyz, highlight, cam) of the target nearest to `last`."""
    meta, rgb, dep = rob.observe()
    avoid = [button] if button is not None else []
    best = None
    for cam in ("wrist", "front"):
        for t in find_targets(meta, rgb, dep, cam, avoid):
            d = np.linalg.norm(t["xyz"][:2] - last[:2])
            do = np.linalg.norm(t["xyz"][:2] - other[:2])
            if d > 0.15 or do < d:
                continue
            if cam == "wrist" and t["edge"]:
                continue
            score = d - (0.05 if (cam == "front" and t["n"] >= 90) else 0.0)
            if best is None or score < best[0]:
                best = (score, t)
    if best is None:
        return None
    return best[1]


def visit(rob, tracks, side, button, label, budget=150):
    other = "left" if side == "right" else "right"
    s0 = rob.step_id
    tgt = tracks[side]
    it = 0
    while not rob.ended and rob.step_id - s0 < budget:
        it += 1
        rob.move_to([tgt[0], tgt[1], SWING_Z], grip=1, max_v=0.9 if it == 1 else 0.5,
                    tol=0.008, max_steps=50)
        rob.hold(2)
        t = locate_target(rob, tgt, tracks[other], button)
        if t is None:
            log("  %s it%d: target not seen" % (label, it))
            if it > 6:
                break
            continue
        moved = np.linalg.norm(t["xyz"][:2] - tgt[:2])
        tgt = np.array(t["xyz"])
        tracks[side] = tgt
        off = np.linalg.norm(tgt[:2] - rob.state[:2])
        log("  %s it%d step %d cam %s tgt %s moved %.3f off %.3f hl %.2f n %d" % (
            label, it, rob.step_id, t["cam"], tgt.round(3).tolist(), moved, off, t["hl"], t["n"]))
        if t["hl"] > 0.12 and off < 0.03:
            rob.hold(2)
            log("visit", label, "OK step", rob.step_id)
            return True
        if off < 0.012 and it >= 3:
            # aligned but no highlight: dip a little lower
            rob.move_to([tgt[0], tgt[1], SWING_Z - 0.02], grip=1, max_v=0.3, tol=0.005, max_steps=15)
            rob.hold(2)
    log("visit", label, "NO HIGHLIGHT CONFIRMED step", rob.step_id)
    return False


def find_button_wrist(rob, guess):
    meta, rgb, dep = rob.observe()
    P = robo.world_points(meta, dep, "wrist")
    nb = (np.abs(P[..., 0] - guess[0]) < 0.05) & (np.abs(P[..., 1] - guess[1]) < 0.05) & (P[..., 2] > 0.022) & (P[..., 2] < 0.06)
    pts = P[nb]
    if len(pts) < 20:
        return None
    zt = np.percentile(pts[:, 2], 95)
    top = pts[pts[:, 2] > zt - 0.006]
    return np.array([top[:, 0].mean(), top[:, 1].mean(), float(zt)])


def press_button(rob, bxy):
    rob.gripper(1, steps=6)
    rob.move_to([bxy[0], bxy[1], 0.15], grip=1, max_v=0.8, tol=0.006)
    rob.hold(3)
    ref = find_button_wrist(rob, bxy)
    if ref is not None and np.linalg.norm(ref[:2] - bxy[:2]) < 0.04:
        log("button wrist refine", ref.round(4).tolist(), "front", np.round(bxy, 4).tolist())
        bxy = ref[:2]
    for attempt in range(3):
        if rob.ended:
            break
        off = [(0, 0), (0.008, 0), (-0.008, 0)][attempt]
        x, y = bxy[0] + off[0], bxy[1] + off[1]
        rob.move_to([x, y, 0.08], grip=1, max_v=0.6, tol=0.005)
        for z in [0.06, 0.045, 0.035, 0.025, 0.015, 0.005]:
            rob.move_to([x, y, z], grip=1, max_v=0.2, tol=0.003, max_steps=10)
            log("press z", z, rob.state[:3].round(4).tolist(), rob.step_id, rob.ended, rob.complete)
            if rob.ended:
                return
        rob.move_to([x, y, 0.10], grip=1, max_v=0.5, tol=0.006)


def main(ctx):
    try:
        _main(ctx)
    except Exception:
        log("EXCEPTION", traceback.format_exc())
    try:
        os.makedirs("/workspace/output", exist_ok=True)
        with open("/workspace/output/controller_log.txt", "w") as f:
            f.write("\n".join(__import__("motion").LOG))
    except Exception:
        pass


def _main(ctx):
    t0 = time.time()
    rob = Robot(ctx)
    meta, rgb, dep = rob.observe()
    instr = meta.get("instruction", "")
    color, first, n = parse_instruction(instr)
    log("instruction:", instr)
    log("parsed color", color, "first", first, "n", n, "limit", rob.remaining)
    sc = perceive_scene(meta, rgb, dep, color)
    log("cube", None if sc["cube"] is None else sc["cube"]["xyz"].round(3).tolist())
    log("right", None if sc["right"] is None else sc["right"]["xyz"].round(3).tolist(),
        "left", None if sc["left"] is None else sc["left"]["xyz"].round(3).tolist(),
        "button", None if sc["button"] is None else sc["button"]["xyz"].round(3).tolist())
    if sc["cube"] is None or sc["right"] is None:
        log("PERCEPTION FAILED")
        return
    button_xy = None
    if sc["button"] is not None:
        b = sc["button"]
        # merge base centroid and cap estimate
        button_xy = np.array(b["xyz"][:2])
    tracks = {"right": np.array(sc["right"]["xyz"]), "left": np.array(sc["left"]["xyz"])}
    button_xyz = None if sc["button"] is None else np.array(sc["button"]["xyz"])
    if not grasp_cube(rob, color, sc["cube"]["xyz"]):
        log("GRASP FAILED")
        return
    order = [first, "left" if first == "right" else "right"]
    seq = order * n
    rob.move_to([rob.state[0], rob.state[1], TRAVEL_Z], grip=1, max_v=0.5, tol=0.008)
    for i, side in enumerate(seq):
        if rob.ended:
            break
        if not held(rob):
            log("CUBE LOST at visit", i)
            meta, rgb, dep = rob.observe()
            sc2 = perceive_scene(meta, rgb, dep, color)
            if sc2["cube"] is None or not grasp_cube(rob, color, sc2["cube"]["xyz"]):
                log("REGRASP FAILED")
                return
        visit(rob, tracks, side, button_xyz, "%d/%d %s" % (i + 1, len(seq), side))
    # put down at the last target
    p = rob.state[:3].copy()
    rob.move_to([p[0], p[1], 0.025], grip=1, max_v=0.4, tol=0.005, max_steps=40)
    rob.gripper(-1, steps=8)
    rob.move_to([p[0], p[1], 0.12], grip=-1, max_v=0.5, tol=0.008)
    log("placed; step", rob.step_id, "elapsed", round(time.time() - t0, 1))
    if button_xy is None:
        log("NO BUTTON FOUND")
        return
    press_button(rob, button_xy)
    log("done; ended", rob.ended, "complete", rob.complete, "step", rob.step_id,
        "elapsed", round(time.time() - t0, 1))
    try:
        os.makedirs("/workspace/output", exist_ok=True)
        with open("/workspace/output/controller_log.txt", "w") as f:
            f.write("\n".join(__import__("motion").LOG))
    except Exception:
        pass
