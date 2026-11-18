"""SwingXtimes controller: pick cube, alternate between targets N swings, put down, press button."""
import os, re, sys, time, json, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlib import Robot, log, LOG
from percep import find_cube, find_scene, COLORS

TRAVEL_Z = 0.10
HOVER_Z = 0.05
SERVO_Z = 0.085
GRASP_Z = 0.02
NUMS = {"one": 1, "once": 1, "two": 2, "twice": 2, "three": 3, "thrice": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def parse_instruction(text):
    t = text.lower()
    color = None
    m = re.search(r"pick up the (\w+) cube", t)
    if m:
        color = m.group(1)
    else:
        for c in COLORS:
            if c in t:
                color = c; break
    m = re.search(r"move it to the (left|right)[- ]side target", t)
    first = m.group(1) if m else None
    if first is None:
        m = re.search(r"(left|right)-to-(left|right)", t)
        first = m.group(1) if m else "right"
    n = None
    m = re.search(r"motion (\w+) times?", t) or re.search(r"(\w+) times", t)
    if m:
        w = m.group(1)
        n = int(w) if w.isdigit() else NUMS.get(w)
    if n is None:
        for w, v in NUMS.items():
            if re.search(r"\b" + w + r"\b", t):
                n = v; break
    if n is None:
        m = re.search(r"\b(\d+)\b", t)
        n = int(m.group(1)) if m else 1
    return color or "red", first, n


def perceive(r, color):
    """From current observation: cube xyz, targets {'right':xy,'left':xy}, button xy, button top z."""
    from percep import find_flat_discs
    meta, imgs = r.meta, r.imgs
    rxy = np.array(meta["state"][:2])
    cube = find_cube(meta, imgs, color, cams=("wrist",)) or find_cube(meta, imgs, color, cams=("front",))
    targets, button, whites = find_scene(meta, imgs, robot_xy=rxy)
    log("whites", [(np.round(w[0], 3).tolist(), round(w[1], 3), w[2]) for w in whites])
    if len(targets) < 2 or np.linalg.norm(targets[0][0] - targets[1][0]) < 0.08:
        flats = find_flat_discs(meta, imgs, robot_xy=rxy)
        log("flat fallback", [(np.round(w[0], 3).tolist(), w[2]) for w in flats])
        if len(flats) >= 2:
            targets = flats[:2]
    tg = None
    if len(targets) == 2:
        a, b = targets[0][0], targets[1][0]
        # world -y is the robot's right side
        tg = {"right": a, "left": b} if a[1] < b[1] else {"right": b, "left": a}
    if cube is not None and cube[1] < 15:
        log("cube cluster small", cube)
    return (cube[0] if cube else None), tg, (button[0] if button else None), (button[1] if button else None)


def lift_to(r, z, grip):
    p = r.state[:3]
    r.move_to([p[0], p[1], z], grip, tol=0.005)


def route_waypoints(p, goal, avoid, r_keep=0.075):
    """xy waypoints from p to goal that keep r_keep away from each xy in avoid."""
    wps = []
    p = np.asarray(p, float)[:2]; goal = np.asarray(goal, float)[:2]
    for o in avoid:
        if o is None:
            continue
        o = np.asarray(o, float)
        d = goal - p
        L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        t = np.clip(np.dot(o - p, d) / L ** 2, 0, 1)
        c = p + t * d
        dist = np.linalg.norm(c - o)
        if dist < r_keep and 0.02 < t < 0.98:
            if dist < 1e-4:
                n = np.array([-d[1], d[0]]) / L
            else:
                n = (c - o) / dist
            wps.append(o + n * (r_keep + 0.03))
    return wps


def travel(r, goal_xy, z, grip, avoid=()):
    p = r.state[:3]
    if p[2] < z - 0.015:
        lift_to(r, z, grip)
    for w in route_waypoints(r.state[:2], goal_xy, avoid):
        log("waypoint", np.round(w, 3))
        r.move_to([w[0], w[1], z], grip, tol=0.02)
    r.move_to([goal_xy[0], goal_xy[1], z], grip, tol=0.006)


def refine_cube(r, color, cube):
    meta, imgs = r.meta, r.imgs
    best = None
    for cams in (("wrist",), ("front",)):
        c = find_cube(meta, imgs, color, cams=cams)
        if c is not None and np.linalg.norm(c[0][:2] - cube[:2]) < 0.08:
            best = c[0]; break
    return cube if best is None else best


def pick(r, color, cube, avoid=()):
    for attempt in range(3):
        log("pick attempt", attempt, "cube", np.round(cube, 3))
        lift_to(r, max(TRAVEL_Z, r.state[2]), -1)
        r.move_to([cube[0], cube[1], TRAVEL_Z + 0.03], -1, tol=0.005)
        cube = refine_cube(r, color, cube)
        log("refined cube", np.round(cube, 3))
        r.move_to([cube[0], cube[1], GRASP_Z + 0.03], -1, tol=0.004)
        cube = refine_cube(r, color, cube)
        r.move_to([cube[0], cube[1], GRASP_Z], -1, tol=0.003)
        r.grip(1, 12)
        r.move_to([cube[0], cube[1], TRAVEL_Z], 1)
        r.grip(1, 3)
        w = r.state[6] + r.state[7]
        log("grip width", w)
        if w > 0.025:
            r.save("picked")
            return True
        r.grip(-1, 8)
        lift_to(r, 0.30, -1)
        r.observe()
        c, _, _, _ = perceive(r, color)
        if c is not None:
            cube = c
    return False


class Targets:
    def __init__(self, tg, radius):
        self.xy = {k: np.array(v, float) for k, v in tg.items()}
        self.radius = radius
        self.red = {k: 0 for k in tg}
        self.n = {k: 0 for k in tg}

    def update(self, r):
        from percep import track_target
        for k in self.xy:
            other = [v for kk, v in self.xy.items() if kk != k][0]
            try:
                t = track_target(r.meta, r.imgs, self.xy[k], radius_est=self.radius, exclude=other)
            except Exception as e:
                log("track error", e); t = None
            if t is not None:
                self.xy[k] = t[0]; self.n[k] = t[1]; self.red[k] = t[2]

    def is_red(self, k):
        return self.red[k] >= 60 and self.red[k] >= 0.15 * max(1, self.n[k])


def visit(r, T, side, max_steps=220):
    """Closed-loop: servo cube over the (possibly moving) target until its ring turns red."""
    other = [k for k in T.xy if k != side][0]
    T.update(r)
    p = r.state[:3]
    if p[2] < SERVO_Z - 0.015:
        lift_to(r, SERVO_Z, 1)
    # route around the other target
    for w in route_waypoints(r.state[:2], T.xy[side], [T.xy[other]]):
        log("waypoint", np.round(w, 3))
        r.move_to([w[0], w[1], SERVO_Z], 1, tol=0.02)
    start = r.step_id or 0
    confirmed = False
    hold = 0
    close_steps = 0
    while not r.ended and (r.step_id or 0) - start < max_steps:
        T.update(r)
        dxy = np.linalg.norm(T.xy[side] - r.state[:2])
        near = dxy < 0.035
        if T.is_red(side) and near:
            confirmed = True
            hold += 1
            if hold >= 3:
                break
        if dxy < 0.012:
            close_steps += 1
            if close_steps >= 15 and not confirmed:
                log("visit", side, "assumed (close but ring not seen red)")
                break
        tgt = T.xy[side]
        p = r.state[:3]
        err = np.array([tgt[0] - p[0], tgt[1] - p[1], SERVO_Z - p[2]])
        d = err / 0.05
        m = np.max(np.abs(d))
        if m > 1:
            d = d / m
        nrows = 2 if np.linalg.norm(err[:2]) > 0.03 else 1
        r.step([[d[0], d[1], d[2], 0, 0, 0, 1]] * nrows)
    log("visit", side, "confirmed" if confirmed else "NOT CONFIRMED", "tcp", np.round(r.state[:3], 3),
        "target", np.round(T.xy[side], 3), "red", T.red, "n", T.n, "step", r.step_id)
    return confirmed


def run(r):
    t0 = time.time()
    r.observe()
    instr = r.meta.get("instruction", "")
    color, first, n = parse_instruction(instr)
    log("instruction:", instr)
    log("parsed color", color, "first", first, "n", n, "limit", r.meta.get("episode_limits"))
    second = "left" if first == "right" else "right"
    cube, tg, button, bz = perceive(r, color)
    log("cube", cube, "targets", tg, "button", button, bz)
    r.save("start")
    if tg is None or cube is None or button is None:
        p = r.state[:3]
        r.move_to([p[0], p[1], 0.60], -1, max_steps=30)
        r.observe()
        c2, t2, b2, bz2 = perceive(r, color)
        log("retry percep", c2, t2, b2, bz2)
        cube = cube if c2 is None else c2
        tg = tg if t2 is None else t2
        if button is None:
            button, bz = b2, bz2
    if cube is None or tg is None:
        log("perception failed")
        return
    from percep import disc_radius
    rads = [disc_radius(r.meta, r.imgs, v) for v in tg.values()]
    rads = [x for x in rads if x is not None and 0.015 < x < 0.06]
    radius = float(np.median(rads)) if rads else 0.034
    log("disc radius", radius)
    T = Targets(tg, radius)
    if not pick(r, color, cube):
        log("pick failed; continuing anyway")
    last = first
    for i in range(n):
        for side in (first, second):
            if r.ended:
                return
            ok = visit(r, T, side)
            last = side
            w = r.state[6] + r.state[7]
            if w < 0.02:
                log("cube lost! width", w, "re-picking")
                r.grip(-1, 8)
                lift_to(r, 0.30, -1)
                r.observe()
                c, _, _, _ = perceive(r, color)
                if c is not None and pick(r, color, c):
                    visit(r, T, side)
        r.save(f"swing{i+1}")
        log("swing", i + 1, "done step", r.step_id, "remaining", r.remaining(), "t", round(time.time() - t0, 1))
    if r.ended:
        return
    # put down at the last target (same place, no extra visit)
    p = r.state[:3]
    r.move_to([p[0], p[1], GRASP_Z + 0.008], 1, tol=0.004, max_steps=40)
    r.grip(-1, 10)
    lift_to(r, TRAVEL_Z, -1)
    r.save("putdown")
    other = second if last == first else first
    if button is None:
        r.observe()
        _, _, button, bz = perceive(r, color)
    if button is None:
        log("no button found")
        return
    press_button(r, button, bz, avoid=[T.xy[other]])
    log("final ended", r.ended, "complete", r.complete, "step", r.step_id, "time", round(time.time() - t0, 1))


def main(ctx):
    r = Robot(ctx, outdir="/workspace/output/frames")
    try:
        run(r)
    except Exception:
        log("EXCEPTION", traceback.format_exc())
    finally:
        try:
            os.makedirs("/workspace/output", exist_ok=True)
            with open("/workspace/output/controller_log.txt", "w") as f:
                f.write("\n".join(LOG))
        except Exception:
            pass


def refine_button(r, button):
    from percep import find_whites
    try:
        ws = find_whites(r.meta, r.imgs)
    except Exception:
        return button, None
    best = None
    for xy, ztop, n in ws:
        if 0.02 <= ztop < 0.07 and np.linalg.norm(xy - button) < 0.06:
            if best is None or n > best[2]:
                best = (xy, ztop, n)
    if best is None:
        return button, None
    return best[0], best[1]


def press_button(r, button, bz, avoid=()):
    r.grip(1, 6)
    top = bz if bz is not None else 0.035
    travel(r, button, TRAVEL_Z, 1, avoid=avoid)
    b2, z2 = refine_button(r, button)
    log("button refine", np.round(button, 3), "->", np.round(b2, 3), z2)
    button = b2
    if z2 is not None:
        top = z2
    r.move_to([button[0], button[1], TRAVEL_Z], 1, tol=0.004)
    r.save("abovebutton")
    for k in range(6):
        if r.ended:
            break
        z = top - 0.01 - 0.006 * k
        r.move_to([button[0], button[1], z], 1, tol=0.003, max_steps=25)
        r.step([[0, 0, 0, 0, 0, 0, 1]] * 3)
        log("press", k, np.round(r.state[:3], 3), r.ended, r.complete)
        r.save(f"press{k}")
        if r.ended:
            break
        r.move_to([button[0], button[1], top + 0.04], 1, tol=0.005, max_steps=20)
        if r.ended:
            break
        r.observe()
        b2, z2 = refine_button(r, button)
        button = b2
