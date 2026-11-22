import os, re, sys, json, math, time, traceback
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from robot import Robot, Ended
import vision as V
import skills as S

NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "a": 1, "an": 1}
COLORS = ["red", "green", "blue", "yellow", "orange", "cyan", "magenta", "purple"]
LOGF = None


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    try:
        with open(os.path.join(LOGDIR, "controller_log.txt"), "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


LOGDIR = "/workspace/output"


def parse_goal(instr):
    instr = instr.lower()
    goal = {}
    for m in re.finditer(r"\b(\d+|one|two|three|four|five|six|a|an)\s+(red|green|blue|yellow|orange|cyan|magenta|purple)\s+cubes?", instr):
        n = int(m.group(1)) if m.group(1).isdigit() else NUM[m.group(1)]
        goal[m.group(2)] = goal.get(m.group(2), 0) + n
    return goal


def find_bin(det, prev=None):
    bins = [b for b in det["bins"] if 0.04 < max(b["ext"]) < 0.14 and b["px"] >= 60]
    if prev is not None:
        near = [b for b in bins if math.hypot(b["xy"][0] - prev[0], b["xy"][1] - prev[1]) < 0.06]
        if near:
            return near[0]["xy"]
        return prev
    return bins[0]["xy"] if bins else None


def find_button(det, binxy):
    cands = []
    for g in det["greys"]:
        if binxy is not None and math.hypot(g["xy"][0] - binxy[0], g["xy"][1] - binxy[1]) < 0.08:
            continue
        if g["xy"][0] < -0.62:
            continue
        e = g["ext"]
        if not (0.05 < min(e) and max(e) < 0.12):
            continue
        if not (det["table_z"] + 0.018 < g["zmax"] < det["table_z"] + 0.05):
            continue
        cands.append(g)
    cands.sort(key=lambda g: -g["px"])
    return cands[0] if cands else None


def choose(cubes, need, binxy, failed, v=(0, 0), horizon=45):
    allc = cubes
    best, bs = None, 1e9
    v = np.asarray(v)
    moving = np.linalg.norm(v) > 0
    for c in cubes:
        if need.get(c["color"], 0) <= 0:
            continue
        x, y = np.array(c["xy"]) + v * horizon
        if moving and (abs(y) > 0.28 or abs(x) > 0.5):
            continue
        if binxy is not None and math.hypot(x - binxy[0], y - binxy[1]) < 0.07:
            continue
        if not (-0.45 < x < 0.35 and abs(y) < 0.45):
            continue
        nb = min([math.hypot(x - o["xy"][0], y - o["xy"][1]) for o in allc if o is not c] + [1.0])
        score = math.hypot(x - binxy[0], y - binxy[1]) if binxy else 0
        if moving:
            score = math.hypot(x + 0.06, y)
        if nb < 0.065:
            score += 0.3
        for f in failed:
            if math.hypot(x - f[0], y - f[1]) < 0.03:
                score += 0.5
        if score < bs:
            bs, best = score, c
    return best


def run(ctx, placed=None):
    r = Robot(ctx, log=log)
    r.observe()
    instr = r.meta.get("instruction", "")
    goal = parse_goal(instr)
    log("instruction:", instr, "goal:", goal)
    placed = dict(placed or {})
    binxy = None; button = None; failed = []
    t0 = time.time(); waits = 0
    while True:
        need = {k: v - placed.get(k, 0) for k, v in goal.items()}
        if all(v <= 0 for v in need.values()):
            break
        if r.remaining() < 120 or time.time() - t0 > 900:
            log("out of budget before goal; pressing button anyway")
            break
        det, vel, tdet = S.scan2(r, log=log)
        binxy = find_bin(det, binxy)
        if binxy is None:
            try:
                binxy = find_bin(V.detect(r.meta, r.imgs, "front"), None)
            except Exception:
                pass
        b = find_button(det, binxy)
        if b is not None:
            button = b
        log("step", r.meta.get("step_id"), "need", need, "bin", np.round(binxy, 3) if binxy else None,
            "cubes", [(c["color"], np.round(c["xy"], 3).tolist()) for c in det["cubes"]])
        if binxy is None:
            r.hold(10); continue
        log("belt velocity (m/step)", np.round(vel, 4))
        c = choose(det["cubes"], need, binxy, failed, vel)
        if c is None:
            waits += 1
            log("no needed cube visible; waiting")
            r.hold(20)
            continue
        log("pick", c["color"], np.round(c["xy"], 3), "yaw", round(c["yaw"], 2), "ztop", round(c["ztop"], 3))
        others = [o for o in det["cubes"] if o is not c]
        nbr = None
        if others:
            o = min(others, key=lambda o: math.hypot(o["xy"][0] - c["xy"][0], o["xy"][1] - c["xy"][1]))
            if math.hypot(o["xy"][0] - c["xy"][0], o["xy"][1] - c["xy"][1]) < 0.075:
                nbr = o["xy"]
        ok, f = S.track_pick(r, c, vel, tdet, log=log, neighbor=nbr)
        if not ok:
            failed.append(c["xy"])
            log("grasp failed")
            r.hold(3, grip=-1.0)
            continue
        held = S.place_in_bin(r, binxy, log=log)
        if not held:
            log("cube lost in transport; not counted")
            continue
        placed[c["color"]] = placed.get(c["color"], 0) + 1
        log("placed", c["color"], "total", placed)
    # press button
    det = S.scan(r)
    binxy = find_bin(det, binxy)
    b = find_button(det, binxy)
    if b is not None:
        button = b
    if button is None:
        try:
            b = find_button(V.detect(r.meta, r.imgs, "front"), binxy)
            if b is not None:
                button = b
        except Exception:
            pass
    log("button", button)
    if button is None:
        log("NO BUTTON FOUND")
        return
    S.press(r, button, log=log)
    for _ in range(5):
        r.hold(10)


def main(ctx, placed=None):
    try:
        os.makedirs(LOGDIR, exist_ok=True)
    except Exception:
        pass
    try:
        run(ctx, placed)
    except Ended:
        log("episode ended")
    except Exception:
        log("ERROR", traceback.format_exc())
