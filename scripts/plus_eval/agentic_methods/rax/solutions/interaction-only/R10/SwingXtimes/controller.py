"""SwingXtimes controller: grasp red cube, swing right->left target N times, put down, press button."""
import json, math, os, re, time
import numpy as np
try:
    from . import percept  # noqa
except Exception:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import percept

CFG = {
    # +1: "right" target is the one with smaller world y (robot's right). -1: larger y.
    "right_is_neg_y": True,
    "hover_z": 0.10,
    "visit_center_z": 0.04,   # TCP height while dwelling over a target (cube bottom ~2cm above table)
    "grasp_z": 0.018,
    "vmax_free": 1.0,
    "vmax_carry": 0.9,
    "press_z": 0.012,
    "dwell": 4,
    "lift_z": 0.16,  # lift well above 0.1 after grasp: first visit only registers after a real pick-up
}
LOG = []
OUT = "/workspace/output"


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)
    try:
        with open(os.path.join(OUT, "controller_log.txt"), "a") as f:
            f.write(s + "\n")
    except Exception:
        pass


NUMS = {"once": 1, "twice": 2, "thrice": 3, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def parse_order(instr):
    s = instr.lower()
    r, l = s.find("right"), s.find("left")
    if r < 0 and l < 0:
        return ("right", "left")
    if l >= 0 and (r < 0 or l < r):
        return ("left", "right")
    return ("right", "left")


COLORS = ("red", "green", "blue", "yellow", "purple", "magenta", "pink", "orange")


def parse_color(instr):
    s = instr.lower()
    m = re.search(r"pick up the (\w+) cube", s)
    if m and m.group(1) in COLORS:
        return m.group(1)
    best = None
    for c in COLORS:
        i = s.find(c)
        if i >= 0 and (best is None or i < best[0]):
            best = (i, c)
    return best[1] if best else "red"


def parse_count(instr):
    s = instr.lower()
    m = re.search(r"(\d+)\s*(?:times|x\b)", s)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+times?\b", s)
    if m:
        return NUMS[m.group(1)]
    for w in ("once", "twice", "thrice"):
        if re.search(r"\b" + w + r"\b", s):
            return NUMS[w]
    m = re.search(r"\b(\d+)\b", s)
    if m:
        return int(m.group(1))
    return None


# ---------- rotation helpers ----------
def rv_to_R(rv):
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def R_to_rv(R):
    c = (np.trace(R) - 1) / 2
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: axis from diagonal
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.maximum(np.diag(A), 0))
        i = int(np.argmax(k))
        k = A[:, i] / max(k[i], 1e-9)
        k = k / np.linalg.norm(k)
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = None
        self.ended = False
        self.complete = None
        self.steps = 0
        self.R0 = None
        self.observe()
        self.R_target = rv_to_R(self.state[3:6])
        self.R0 = self.R_target.copy()
        self.grip = -1.0
        self.color = "red"

    def observe(self):
        r = self.ctx.call("robomme_observe")
        self.reply = r
        self.meta = json.loads(r["content"][0]["text"])
        self._upd(self.meta)
        return r

    def _upd(self, meta):
        self.state = np.array(meta["state"], float)
        self.ended = bool(meta.get("episode_ended", False))
        lim = meta.get("episode_limits", {})
        self.remaining = lim.get("remaining_episode_steps", 9999)
        tr = meta.get("transition") or {}
        if tr.get("task_complete"):
            self.complete = True
        if tr.get("episode_ended"):
            self.ended = True
            self.complete = bool(tr.get("task_complete"))
        for s in tr.get("steps", []) or []:
            if s.get("task_complete") is not None:
                if s.get("task_complete"):
                    self.complete = True
            if s.get("episode_ended"):
                self.ended = True
                self.complete = bool(s.get("task_complete"))

    @property
    def pos(self):
        return self.state[:3]

    def rot_action(self):
        Rc = rv_to_R(self.state[3:6])
        e = R_to_rv(self.R_target @ Rc.T)
        return np.clip(e / 0.5, -1, 1)

    def step(self, rows):
        if self.ended:
            return
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        r = self.ctx.call("robomme_step", actions=rows)
        self.reply = r
        self.meta = json.loads(r["content"][0]["text"])
        self.steps += len(rows)
        self._upd(self.meta)
        if self.ended:
            log("EPISODE ENDED complete=", self.complete, "step", self.meta.get("step_id"))

    def move_to(self, target, vmax=0.8, tol=0.004, max_steps=80, gain=2.0, goal_fn=None):
        """Per-step P control. Arm realises ~40% of the commanded delta per tick.
        goal_fn(robot) may return an updated [x, y] (moving targets)."""
        target = np.asarray(target, float).copy()
        used = 0
        while used < max_steps:
            if self.ended:
                return False
            if goal_fn is not None:
                try:
                    g = goal_fn(self)
                    if g is not None:
                        target[0], target[1] = g[0], g[1]
                except Exception as e:
                    log("goal_fn error", e)
            err = target - self.pos
            d = np.linalg.norm(err)
            if d < tol:
                return True
            a = err / 0.05 * gain
            m = np.max(np.abs(a))
            if m > vmax:
                a = a * (vmax / m)
            k = 2 if d > 0.08 else 1
            rot = self.rot_action()
            rows = [list(a) + list(rot) + [self.grip]] + [list(a) + [0, 0, 0] + [self.grip]] * (k - 1)
            self.step(rows)
            used += k
        log("move_to not converged", target.tolist(), self.pos.tolist())
        return False

    def hold_track(self, n, goal_fn, z):
        """Stay over a (possibly moving) target for n steps, correcting xy each step."""
        for _ in range(n):
            if self.ended:
                return
            tgt = np.array([self.pos[0], self.pos[1], z])
            try:
                g = goal_fn(self)
                if g is not None:
                    tgt[0], tgt[1] = g[0], g[1]
            except Exception as e:
                log("goal_fn error", e)
            err = tgt - self.pos
            a = np.clip(err / 0.05 * 2.0, -0.5, 0.5)
            self.step([list(a) + list(self.rot_action()) + [self.grip]])

    def hold(self, n, grip=None):
        if grip is not None:
            self.grip = grip
        rows = []
        for j in range(n):
            rows.append([0, 0, 0] + list(self.rot_action() if j == 0 else [0, 0, 0]) + [self.grip])
        self.step(rows)

    def align_rot(self, iters=6):
        if self.ended:
            return
        for _ in range(iters):
            e = self.rot_action()
            if np.linalg.norm(e) * 0.5 < 0.02:
                return
            self.step([[0, 0, 0] + list(e) + [self.grip]])


def perceive(robot):
    r = robot.observe()
    meta, imgs = percept.decode(r)
    out = {}
    for cam in ("wrist", "front"):
        try:
            res, P = percept.analyse(meta, imgs, cam, robot.color)
            out[cam] = res
        except Exception as e:
            log("percept fail", cam, e)
    return out, meta, imgs


def find_scene(robot):
    out, meta, imgs = perceive(robot)
    for cam in ("wrist", "front"):
        if cam in out:
            log("perc", cam, json.dumps(out[cam]))
    # targets: grey comps, flat (zmax<0.02), ext ~0.06-0.10
    def classify(res):
        tg, bt = [], []
        for o in res.get("grey", []):
            if o["n"] < 30:
                continue
            if o["zmax"] < 0.02:
                tg.append(o)
            elif o["zmax"] < 0.07:
                bt.append(o)
        return tg, bt
    tg_w, bt_w = classify(out.get("wrist", {}))
    tg_f, bt_f = classify(out.get("front", {}))
    targets = tg_w if len(tg_w) >= 2 else tg_f
    targets = sorted(targets, key=lambda o: -o["n"])[:2]
    buttons = bt_w if bt_w else bt_f
    button = max(buttons, key=lambda o: o["n"]) if buttons else None
    red = out.get("wrist", {}).get("red") or out.get("front", {}).get("red")
    return targets, button, red, out


def cube_yaw(robot):
    """Estimate cube yaw (mod 90deg) from wrist top-face red pixels."""
    try:
        meta, imgs = percept.decode(robot.reply if robot.reply else robot.observe())
        rgb = imgs[("wrist", "rgb")].astype(int)
        P = percept.backproject(meta, imgs[("wrist", "depth")], "wrist")
        R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        m = percept.color_mask(rgb, robot.color) & np.isfinite(P[..., 2]) & (np.nan_to_num(P[..., 2], nan=-1) > 0.012)
        comps = percept.components(m, 3)
        if not comps:
            return None, None
        c = max(comps, key=len)
        pts = P[c[:, 0], c[:, 1]]
        ztop = np.nanmax(pts[:, 2])
        top = pts[pts[:, 2] > ztop - 0.006][:, :2]
        if len(top) < 8:
            return None, np.nanmedian(pts, 0)
        ctr = top.mean(0)
        best, ba = None, 0.0
        for a in np.radians(np.arange(0, 90, 2)):
            Rm = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
            q = (top - ctr) @ Rm.T
            area = np.ptp(q[:, 0]) * np.ptp(q[:, 1])
            if best is None or area < best:
                best, ba = area, a
        if ba > math.pi / 4:
            ba -= math.pi / 2
        c3 = np.array([ctr[0], ctr[1], ztop])
        return ba, c3
    except Exception as e:
        log("yaw fail", e)
        return None, None


def grasp_cube(robot, red_xyz, cfg):
    hz = cfg["hover_z"]
    robot.grip = -1.0
    robot.move_to([red_xyz[0], red_xyz[1], 0.16], vmax=cfg["vmax_free"], tol=0.01)
    # refine with close wrist view
    for _ in range(2):
        robot.observe()
        yaw, c3 = cube_yaw(robot)
        log("cube refine yaw", yaw, None if c3 is None else c3.tolist())
        if c3 is not None and np.linalg.norm(np.asarray(c3[:2]) - np.asarray(red_xyz[:2])) < 0.08:
            red_xyz = c3
        if yaw is not None:
            robot.R_target = Rz(yaw) @ robot.R0
        robot.move_to([red_xyz[0], red_xyz[1], 0.13], vmax=0.5, tol=0.004)
        robot.align_rot()
    ztop = red_xyz[2]
    gz = max(ztop - 0.022, 0.012)
    robot.move_to([red_xyz[0], red_xyz[1], gz + 0.03], vmax=0.6, tol=0.005)
    robot.move_to([red_xyz[0], red_xyz[1], gz], vmax=0.25, tol=0.003)
    robot.hold(8, grip=1.0)
    robot.move_to([red_xyz[0], red_xyz[1], cfg["lift_z"]], vmax=0.8, tol=0.01)
    fingers = robot.state[6:8]
    log("after grasp fingers", fingers.tolist(), "pos", robot.pos.tolist())
    return holding(robot)


def holding(robot):
    return float(robot.state[6:8].sum()) > 0.01


def overview_red(robot, cfg):
    """Go to the overview pose and re-locate the red cube."""
    robot.R_target = robot.R0.copy()
    robot.move_to([-0.06, 0.0, 0.45], vmax=1.0, tol=0.02)
    robot.align_rot()
    _, _, red, _ = find_scene(robot)
    return red


def acquire(robot, red_xyz, cfg):
    for attempt in range(4):
        if robot.ended:
            return False
        if grasp_cube(robot, red_xyz, cfg):
            return True
        log("grasp failed, retry", attempt)
        robot.grip = -1.0
        robot.hold(4)
        red = overview_red(robot, cfg)
        if red is None:
            log("red cube not found")
            return False
        red_xyz = np.array(red["xyz"])
    return False


class Tracker:
    """Tracks the two target discs (and the button) from the front camera of every reply."""
    def __init__(self, right, left, button):
        self.pos = {"right": np.array(right["xyz"][:2], float), "left": np.array(left["xyz"][:2], float)}
        self.button = None if button is None else np.array(button["xyz"][:2], float)
        self.bz = None if button is None else button["zmax"]
        self.last_step = None

    def update(self, robot):
        sid = robot.meta.get("step_id")
        if sid == self.last_step:
            return
        self.last_step = sid
        meta, imgs = percept.decode(robot.reply)
        discs = [d for d in percept.find_discs(meta, imgs, "front") if d["n"] >= 25 and min(d["ext"]) > 0.025]
        for name in ("right", "left"):
            other = self.pos["left" if name == "right" else "right"]
            best = None
            for d in discs:
                p = np.array(d["xyz"][:2])
                # use bbox centre when disc fully visible, else median
                if min(d["ext"]) > 0.065:
                    p = np.array(d["ctr"])
                dist = np.linalg.norm(p - self.pos[name])
                if dist < 0.10 and dist < np.linalg.norm(p - other) and (best is None or dist < best[0]):
                    best = (dist, p, d)
            if best is not None:
                d = best[2]
                full = min(d["ext"]) > 0.065 and d["n"] > 80
                if full:
                    self.pos[name] = best[1]
            else:
                # lost (jumped while occluded): take a full disc on this target's side that is not the other one
                cands = []
                for d in discs:
                    if not (min(d["ext"]) > 0.065 and d["n"] > 80):
                        continue
                    p = np.array(d["ctr"])
                    if np.linalg.norm(p - other) < 0.06:
                        continue
                    side_ok = (p[1] < other[1]) if name == "right" else (p[1] > other[1])
                    if side_ok:
                        cands.append((np.linalg.norm(p - self.pos[name]), p))
                if cands:
                    cands.sort(key=lambda t: t[0])
                    self.pos[name] = cands[0][1]

    def lit(self, robot, name):
        """Fraction of red pixels on the disc near target `name` (None if not visible)."""
        meta, imgs = percept.decode(robot.reply)
        best = None
        for d in percept.find_discs(meta, imgs, "front"):
            if d["n"] < 10:
                continue
            dist = np.linalg.norm(np.array(d["xyz"][:2]) - self.pos[name])
            if dist < 0.06 and (best is None or dist < best[0]):
                best = (dist, d)
        return None if best is None else best[1]["redfrac"]

    def goal(self, name):
        def fn(robot):
            self.update(robot)
            return self.pos[name]
        return fn

    def update_button(self, robot):
        meta, imgs = percept.decode(robot.reply)
        bs = percept.find_buttons(meta, imgs, "front") + percept.find_buttons(meta, imgs, "wrist")
        best = None
        for b in bs:
            p = np.array(b["xyz"][:2])
            dist = 0 if self.button is None else np.linalg.norm(p - self.button)
            if dist < 0.12 and (best is None or b["n"] > best[1]["n"]):
                best = (dist, b)
        if best is not None and best[1]["n"] >= 15:
            self.button = np.array(best[1]["xyz"][:2])
            self.bz = best[1]["zmax"]
            return True
        return False


def do_swings(robot, cfg, right, left, order, n, tracker=None):
    hz = cfg["hover_z"]
    vz = cfg["visit_center_z"]
    visits = []
    for i in range(n):
        for name in order:
            t = right if name == "right" else left
            for tries in range(3):
                if robot.ended:
                    break
                if not holding(robot):
                    log("cube lost before", name, i + 1)
                    red = overview_red(robot, cfg)
                    if red is None or not acquire(robot, np.array(red["xyz"]), cfg):
                        continue
                if tracker is None:
                    tracker = Tracker(right, left, None)
                gf = tracker.goal(name)
                robot.move_to([robot.pos[0], robot.pos[1], hz], vmax=1.0, tol=0.015)
                tracker.update(robot)
                x, y = tracker.pos[name]
                robot.move_to([x, y, hz], vmax=cfg["vmax_carry"], tol=0.012, goal_fn=gf)
                robot.move_to([x, y, vz], vmax=0.5, tol=0.005, goal_fn=gf)
                robot.hold_track(cfg["dwell"], gf, vz)
                for rep in range(3):
                    lit = tracker.lit(robot, name)
                    log("   target", name, "tracked at", [round(v, 4) for v in tracker.pos[name]], "lit", lit)
                    if lit is None or lit > 0.15 or robot.ended:
                        break
                    # not registered: back off, re-track and come down again
                    robot.move_to([robot.pos[0], robot.pos[1], vz + 0.06], vmax=0.8, tol=0.01)
                    robot.hold(2)
                    tracker.update(robot)
                    x, y = tracker.pos[name]
                    robot.move_to([x, y, vz + 0.06], vmax=0.6, tol=0.008, goal_fn=gf)
                    robot.move_to([x, y, vz], vmax=0.5, tol=0.005, goal_fn=gf)
                    robot.hold_track(cfg["dwell"], gf, vz)
                ok = holding(robot)
                log(f"swing {i+1} {name} at", [round(v, 4) for v in robot.pos], "holding", ok, "step", robot.meta.get("step_id"))
                if ok:
                    visits.append(name)
                    break
    return visits


def finish(robot, cfg, home_xy, button, tracker=None):
    hz = cfg["hover_z"]
    if holding(robot):
        robot.move_to([robot.pos[0], robot.pos[1], hz], vmax=1.0, tol=0.015)
        robot.move_to([home_xy[0], home_xy[1], hz], vmax=cfg["vmax_carry"], tol=0.012)
        robot.move_to([home_xy[0], home_xy[1], 0.024], vmax=0.5, tol=0.004)
    robot.hold(6, grip=-1.0)
    robot.move_to([robot.pos[0], robot.pos[1], hz], vmax=0.8, tol=0.01)
    if tracker is None:
        tracker = Tracker({"xyz": [0, -0.2]}, {"xyz": [0, 0.2]}, button)
    if tracker.button is not None and not robot.ended:
        robot.grip = 1.0
        # look again from above (button may have moved)
        robot.move_to([robot.pos[0], robot.pos[1], 0.25], vmax=1.0, tol=0.02)
        robot.observe()
        tracker.update_button(robot)
        log("button at", tracker.button.tolist(), tracker.bz)
        bx, by = tracker.button
        bz = tracker.bz if tracker.bz is not None else 0.034
        def bgoal(rb):
            if rb.pos[2] > bz + 0.04 and tracker.update_button(rb):
                return tracker.button
            return None
        robot.move_to([bx, by, max(hz, bz + 0.06)], vmax=cfg["vmax_free"], tol=0.006, goal_fn=bgoal)
        bx, by = tracker.button
        robot.observe()
        tracker.update_button(robot)
        bx, by = tracker.button
        log("button final", tracker.button.tolist())
        robot.move_to([bx, by, bz + 0.02], vmax=0.5, tol=0.004)
        for k in range(3):
            if robot.ended:
                break
            robot.move_to([bx, by, cfg["press_z"]], vmax=0.2, tol=0.004, max_steps=25)
            log("press pos", robot.pos.tolist(), "ended", robot.ended)
            if robot.ended:
                break
            robot.move_to([bx, by, bz + 0.05], vmax=0.5, tol=0.006)


def main(ctx):
    try:
        _main(ctx)
    except Exception as e:
        import traceback
        log("FATAL", repr(e), traceback.format_exc())


def _main(ctx):
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    cfg = dict(CFG)
    robot = Robot(ctx)
    instr = robot.meta.get("instruction", "")
    n = parse_count(instr) or 1
    order = parse_order(instr)
    robot.color = parse_color(instr)
    log("color", robot.color)
    log("instruction:", instr, "-> count", n, "order", order, "remaining", robot.remaining)
    targets, button, red, _ = find_scene(robot)
    log("targets", targets, "button", button, "red", red)
    if len(targets) < 2 or red is None:
        log("scene perception failed")
        return
    tA, tB = sorted(targets, key=lambda o: o["xyz"][1])  # A: smaller y
    if cfg["right_is_neg_y"]:
        right, left = tA, tB
    else:
        right, left = tB, tA
    red_xyz = np.array(red["xyz"])
    home_xy = red_xyz[:2].copy()
    acquire(robot, red_xyz, cfg)
    tracker = Tracker(right, left, button)
    do_swings(robot, cfg, right, left, order, n, tracker)
    finish(robot, cfg, home_xy, button, tracker)
    log("done; ended", robot.ended, "complete", robot.complete, "steps", robot.steps,
        "wall", round(time.time() - t0, 1))
