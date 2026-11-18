"""SwingXtimes controller v2.

Evidence (memory/PROGRESS.md): ep1 static targets, ep2 targets EVADE the cube (~6 cm runs when the
cube comes within ~4.5 cm xy) and coloured spotlights tint white objects. Visits are registered
natively by xy proximity (even with the cube 6 cm above the target); a wrong first visit or an
extra visit fails the episode at once. RIGHT = target with smaller world y (robot's right).

Pipeline / transition table:
  PERCEIVE -> instruction (colour, X, first side); discs + button by depth geometry (front+wrist).
  GRASP    -> hover, wrist refine, descend, close, lift; verify opening; <=3 attempts.
  SWING    -> for each planned visit: CHASE(dest) with per-step re-detection of both targets,
              path avoiding the other target; done when TCP within VISIT_TOL of a (preferably
              stationary) dest for VISIT_DWELL steps; timeout VISIT_TIMEOUT steps -> next visit.
              Cube lost -> GRASP again -> resume same visit.
  PLACE    -> free spot far from both targets/button, lower, open, lift.
  PRESS    -> button cap from fresh geometry, closed gripper, progressive descents; repeat with
              small offsets until the episode ends.
Episode end at any point -> stop and return.
"""
import math, re, sys, time, traceback, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perception as pc
import robot as rb

VERSION = "swing-v2"
Z_TRANSIT = 0.08      # TCP z when carrying (cube bottom ~0.058); ep1+ep2 visits registered at this z
Z_VISIT = 0.055       # TCP z during dwell on a target (cube bottom ~0.033 above table)
Z_HOVER = 0.10
MIN_HOLD = 0.012      # finger sum while closed on 4 cm cube = 0.0366 (ep1, ep2)
AVOID_R = 0.085       # keep cube this far (xy) from the non-destination target (> evade 0.045, reg ~0.015)
VISIT_TOL = 0.008
VISIT_DWELL = 3
VISIT_TIMEOUT = 220
PLACE_CLEAR = 0.12
DISC_MIN_N = 90
JITTER = 0.008        # per-frame estimate changes below this are smoothed (ep1 replay jitter <= 6 mm)
REACH_BOX = ((-0.40, 0.20), (-0.33, 0.33))

NUM = {"once": 1, "twice": 2, "thrice": 3, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
       "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def log(*a):
    print(f"[{VERSION} {time.strftime('%H:%M:%S')}]", *a, flush=True)


def parse_instruction(text):
    t = (text or "").lower()
    m = re.search(r"pick up the (\w+) cube", t)
    color = m.group(1) if m else "red"
    X = None
    m = re.search(r"motion (\w+) times?", t) or re.search(r"(\w+) times", t)
    if m:
        w = m.group(1)
        X = int(w) if w.isdigit() else NUM.get(w)
    if X is None:
        for w in ("once", "twice", "thrice"):
            if re.search(r"\b%s\b" % w, t):
                X = NUM[w]
    if X is None:
        X = 1
    m = re.search(r"to the (left|right)[- ]side target and then to the (left|right)", t)
    if m:
        first = m.group(1)
    else:
        m = re.search(r"(left|right)-to-(left|right)", t)
        if m:
            first = m.group(1)
        else:
            il, ir = t.find("left"), t.find("right")
            first = "left" if il != -1 and (ir == -1 or il < ir) else "right"
    return color, X, first


def seg_dist(p, a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); p = np.asarray(p, float)
    ab = b - a; L = float(ab @ ab)
    if L < 1e-12:
        return float(np.linalg.norm(p - a)), a
    t = max(0.0, min(1.0, float((p - a) @ ab) / L))
    q = a + t * ab
    return float(np.linalg.norm(p - q)), q


def next_waypoint(cur, goal, avoid, r=AVOID_R):
    """Next xy waypoint toward goal that keeps away from `avoid` points (list)."""
    cur = np.asarray(cur, float); goal = np.asarray(goal, float)
    for c in avoid:
        c = np.asarray(c, float)
        dc = np.linalg.norm(cur - c)
        if dc < r:
            # inside the circle: move radially outward (plus toward goal tangentially)
            out = (cur - c) / max(dc, 1e-6)
            return c + out * (r + 0.02)
        d, q = seg_dist(c, cur, goal)
        if d < r and np.linalg.norm(goal - c) > r:
            ab = goal - cur
            perp = np.array([-ab[1], ab[0]]) / max(np.linalg.norm(ab), 1e-9)
            side = q - c
            if np.linalg.norm(side) < 1e-4:
                side = perp if perp[0] > 0 else -perp
            side = side / np.linalg.norm(side)
            # tangent-ish waypoint beside the obstacle
            return c + side * (r + 0.03)
    return goal


class Tracker:
    """Tracks the two target discs and the button from per-step observations."""

    def __init__(self):
        self.T = [None, None]       # xy estimates (index 0 = right/smaller y, 1 = left)
        self.prev = [None, None]
        self.vel = [np.zeros(2), np.zeros(2)]
        self.last_seen = [0, 0]
        self.bias = np.zeros(2)     # correction added to front-camera estimates
        self.button = None

    def init(self, meta, imgs):
        dets = {}
        for cam in ("front", "wrist"):
            try:
                P, valid = pc.world_map(meta, imgs, cam)
                b = pc.detect_buttons(meta, imgs, cam, P, valid)
                d = pc.detect_discs(meta, imgs, cam, P=P, valid=valid)
                dets[cam] = (d, b)
            except Exception:
                log("init detect fail", cam, traceback.format_exc())
                dets[cam] = ([], [])
        buttons = dets["front"][1] + dets["wrist"][1]
        if buttons:
            bs = [buttons[0]] + [b for b in buttons[1:] if np.linalg.norm(b["center"] - buttons[0]["center"]) < 0.04]
            self.button = {"center": np.mean([b["center"] for b in bs], 0),
                           "ztop": float(np.mean([b["ztop"] for b in bs]))}
        def good(ds):
            out = []
            for d in ds:
                if d["n"] < DISC_MIN_N and not d["full"]:
                    continue
                if self.button is not None and np.linalg.norm(d["center"] - self.button["center"]) < 0.06:
                    continue
                out.append(d)
            return out
        F = good(dets["front"][0]); W = good(dets["wrist"][0])
        log("init front discs", [(np.round(d["center"], 3).tolist(), d["n"], d["full"]) for d in F])
        log("init wrist discs", [(np.round(d["center"], 3).tolist(), d["n"], d["full"]) for d in W])
        log("init button", None if self.button is None else (np.round(self.button["center"], 3).tolist(), round(self.button["ztop"], 4)))
        from_front = len(F) >= 2
        base = F[:2] if from_front else W[:2]
        if len(base) < 2:
            return False
        cents = []
        offs = []
        for d in base:
            c = d["center"]
            other = [w["center"] for w in W if np.linalg.norm(w["center"] - c) < 0.03] if from_front else []
            if other:
                offs.append(other[0] - c)
                cents.append((c + other[0]) / 2)
            else:
                cents.append(c)
        if offs:
            self.bias = np.mean(offs, 0) / 2
        cents.sort(key=lambda c: c[1])
        self.T = [np.array(cents[0]), np.array(cents[1])]
        self.prev = [self.T[0].copy(), self.T[1].copy()]
        log("targets RIGHT", np.round(self.T[0], 3).tolist(), "LEFT", np.round(self.T[1], 3).tolist(),
            "front bias", np.round(self.bias, 4).tolist())
        return True

    def update(self, meta, imgs, step):
        """Associate front-camera disc detections with the tracked targets."""
        if imgs is None or ("front", "depth") not in imgs:
            return
        try:
            d = pc.detect_discs(meta, imgs, "front")
        except Exception:
            log("track detect fail", traceback.format_exc())
            return
        cand = []
        for x in d:
            if x["n"] < 40 and not x["full"]:
                continue
            if self.button is not None and np.linalg.norm(x["center"] - self.button["center"]) < 0.06:
                continue
            cand.append(x["center"] + self.bias)
        used = set()
        for k in (0, 1):
            if self.T[k] is None:
                continue
            best, bd = None, 0.10
            for j, c in enumerate(cand):
                if j in used:
                    continue
                dd = np.linalg.norm(c - self.T[k])
                if dd < bd:
                    best, bd = j, dd
            if best is not None:
                used.add(best)
                newc = cand[best]
                if bd < JITTER:
                    newc = self.T[k] + 0.4 * (newc - self.T[k])
                self.vel[k] = 0.5 * self.vel[k] + 0.5 * (newc - self.T[k])
                self.T[k] = newc
                self.last_seen[k] = step
            else:
                self.vel[k] = 0.5 * self.vel[k]


def find_cube(R, color, cams=("wrist", "front"), on_table=True):
    for cam in cams:
        try:
            o = pc.locate_color_object(R.meta, R.imgs, cam, color, zmax=0.12,
                                       ztop_range=(0.025, 0.06) if on_table else None)
        except Exception:
            o = None
            log("locate fail", traceback.format_exc())
        if o is not None and o["n"] >= 10:
            o["cam"] = cam
            return o
    return None


def grasp(R, color, avoid, attempts=3):
    yaw = 0.0
    for att in range(attempts):
        R.observe()
        o = find_cube(R, color, cams=("front", "wrist"))
        if o is None:
            log("grasp: cube not found; observing from high pose")
            R.move_to([-0.05, 0.0, 0.30], yaw=0.0, grip=-1, tol=0.02)
            o = find_cube(R, color, cams=("front", "wrist"))
            if o is None:
                log("grasp: cube still not found")
                continue
        c = o["center"]
        log("grasp att", att, "cube", np.round(c, 3).tolist(), "yaw", round(o["yaw"], 3), o["cam"])
        # approach at hover height keeping clear of targets
        for _ in range(4):
            w = next_waypoint(R.pos[:2], c[:2], avoid)
            R.move_to([w[0], w[1], Z_HOVER + 0.05], grip=-1, tol=0.012)
            if np.linalg.norm(w - c[:2]) < 1e-6:
                break
        o2 = None
        for k in range(2):
            o2 = find_cube(R, color, cams=("wrist", "front"))
            if o2 is None:
                break
            c = o2["center"]; yaw = o2["yaw"]
            R.move_to([c[0], c[1], Z_HOVER], yaw=yaw, grip=-1, tol=0.004)
        if o2 is None:
            yaw = o["yaw"]
        log("grasp refine cube", np.round(c, 3).tolist(), "yaw", round(yaw, 3))
        zg = max(0.012, c[2] - 0.018)
        R.move_to([c[0], c[1], zg], yaw=yaw, grip=-1, speed=0.5, tol=0.003, max_steps=30)
        R.gripper(1, 12)
        R.move_to([c[0], c[1], Z_TRANSIT], yaw=yaw, grip=1, speed=0.6, tol=0.01)
        log("after lift opening", round(R.opening, 4), "pos", np.round(R.pos, 3).tolist(), "step", R.step_id)
        if R.opening > MIN_HOLD:
            return True, yaw, c[:2].copy()
        R.gripper(-1, 8)
    return False, yaw, None


def held(R):
    return R.opening > MIN_HOLD


def step_toward(R, xyz, yaw, speed=1.0):
    d = np.asarray(xyz, float) - R.pos
    dp = d / rb.POS_SCALE
    nrm = np.linalg.norm(dp)
    if nrm > speed:
        dp *= speed / nrm
    rv = R.rot_err(yaw)
    dr = rv / rb.ROT_SCALE
    rn = np.linalg.norm(dr)
    if rn > 0.6:
        dr *= 0.6 / rn
    R.step([list(dp) + list(dr) + [R.grip_cmd]])


def chase(R, trk, k, yaw):
    """Bring the held cube over target k (tracked live). Returns (ok, why)."""
    other = 1 - k
    s0 = R.step_id
    dwell = 0
    best = 9.0
    while R.step_id - s0 < VISIT_TIMEOUT:
        trk.update(R.meta, R.imgs, R.step_id)
        dest = trk.T[k]
        v = trk.vel[k]
        goal = dest + v * 1.5
        goal = np.array([min(max(goal[0], REACH_BOX[0][0]), REACH_BOX[0][1]),
                         min(max(goal[1], REACH_BOX[1][0]), REACH_BOX[1][1])])
        cur = R.pos[:2]
        d = float(np.linalg.norm(cur - dest))
        best = min(best, d)
        moving = np.linalg.norm(v) > 0.005
        if d < VISIT_TOL:
            dwell += 1
        else:
            dwell = 0
        if (dwell >= VISIT_DWELL and not moving) or dwell >= 2 * VISIT_DWELL:
            log("visit", "RL"[k], "reached d=%.4f" % d, "T", np.round(dest, 3).tolist(),
                "other", np.round(trk.T[other], 3).tolist(), "steps", R.step_id - s0, "step", R.step_id)
            return True, "reached"
        if not held(R):
            return False, "dropped"
        w = next_waypoint(cur, goal, [trk.T[other]])
        z = Z_VISIT if (d < 0.02 and np.linalg.norm(w - goal) < 1e-6) else Z_TRANSIT
        if np.linalg.norm(w - goal) > 1e-6 and R.pos[2] < Z_TRANSIT - 0.02:
            z = Z_TRANSIT
        step_toward(R, [w[0], w[1], z], yaw, speed=1.0)
        if (R.step_id - s0) % 10 == 0:
            log("  chase", "RL"[k], "d=%.3f" % d, "T", np.round(dest, 3).tolist(), "v", np.round(v, 4).tolist(),
                "tcp", np.round(R.pos, 3).tolist())
    log("visit", "RL"[k], "TIMEOUT best d=%.4f" % best)
    return best < 0.012, "timeout"


def choose_place(trk, cube_start, cur):
    obstacles = [t for t in trk.T if t is not None]
    if trk.button is not None:
        obstacles.append(trk.button["center"])
    cands = []
    if cube_start is not None:
        cands.append(np.asarray(cube_start))
    for x in np.arange(-0.25, 0.121, 0.03):
        for y in np.arange(-0.24, 0.241, 0.03):
            cands.append(np.array([x, y]))
    best, bs = None, -1e9
    for p in cands:
        m = min(np.linalg.norm(p - t) for t in obstacles) if obstacles else 1.0
        if m < PLACE_CLEAR:
            continue
        score = -np.linalg.norm(p - cur) + (0.05 if cube_start is not None and p is cands[0] else 0)
        if score > bs:
            best, bs = p, score
    if best is None:
        best = max(cands, key=lambda p: min(np.linalg.norm(p - t) for t in obstacles))
    return best


def place(R, trk, yaw, cube_start):
    p = choose_place(trk, cube_start, R.pos[:2])
    log("place at", np.round(p, 3).tolist())
    for _ in range(80):
        trk.update(R.meta, R.imgs, R.step_id)
        w = next_waypoint(R.pos[:2], p, [t for t in trk.T if t is not None])
        if np.linalg.norm(R.pos[:2] - p) < 0.01:
            break
        step_toward(R, [w[0], w[1], Z_TRANSIT], yaw)
    R.move_to([p[0], p[1], 0.026], yaw=yaw, grip=1, speed=0.5, tol=0.004, max_steps=30)
    R.gripper(-1, 10)
    R.move_to([p[0], p[1], Z_HOVER], yaw=yaw, grip=-1, speed=0.8, tol=0.015)
    log("placed; pos", np.round(R.pos, 3).tolist(), "step", R.step_id)


def detect_button_now(R, near=None):
    best = None
    for cam in ("wrist", "front"):
        try:
            bs = pc.detect_buttons(R.meta, R.imgs, cam)
        except Exception:
            bs = []
        for b in bs:
            if near is not None and np.linalg.norm(b["center"] - near) > 0.06:
                continue
            if best is None:
                best = b
    return best


def press(R, trk, offset=(0.0, 0.0)):
    if trk.button is None:
        R.observe()
        b = detect_button_now(R)
        if b is None:
            log("press: no button found")
            R.move_to([-0.1, 0.0, 0.35], yaw=0.0, grip=1, tol=0.02)
            b = detect_button_now(R)
            if b is None:
                return False
        trk.button = {"center": b["center"], "ztop": b["ztop"]}
    c = np.asarray(trk.button["center"]) + np.asarray(offset); zt = trk.button["ztop"]
    log("press: cap", np.round(c, 3).tolist(), "ztop", round(zt, 3))
    R.grip_cmd = 1.0
    R.move_to([c[0], c[1], zt + 0.07], yaw=0.0, grip=1, tol=0.008)
    b = detect_button_now(R, near=c)
    if b is not None:
        c = b["center"] + np.asarray(offset); zt = b["ztop"]
        log("press: refined cap", np.round(c, 3).tolist(), "ztop", round(zt, 3))
        R.move_to([c[0], c[1], zt + 0.04], yaw=0.0, grip=1, tol=0.005)
    for depth in (0.010, 0.018, 0.026):
        R.move_to([c[0], c[1], zt - depth], yaw=0.0, grip=1, speed=0.3, tol=0.003, max_steps=30)
        R.hold(4)
        log("press depth", depth, "pos", np.round(R.pos, 3).tolist(), "step", R.step_id)
    R.move_to([c[0], c[1], zt + 0.04], yaw=0.0, grip=1, speed=0.5, tol=0.006, max_steps=20)
    return False


def main(ctx):
    t0 = time.time()
    R = rb.Robot(ctx, log=log)
    trk = Tracker()
    try:
        meta, _ = R.observe()
        instr = meta.get("instruction", "")
        color, X, first = parse_instruction(instr)
        log("instruction:", instr, "| color", color, "X", X, "first", first, "limit", R.limit)
        ok = False
        for k in range(3):
            ok = trk.init(R.meta, R.imgs)
            if ok:
                break
            R.observe()
        if not ok:
            log("WARNING targets not found; moving up to look")
            R.move_to([-0.1, 0.0, 0.35], yaw=0.0, grip=-1, tol=0.02)
            ok = trk.init(R.meta, R.imgs)
        if not ok:
            log("FATAL no targets")
        ok, yaw, cube_start = grasp(R, color, [t for t in trk.T if t is not None])
        log("grasp ok", ok, "step", R.step_id, "t", round(time.time() - t0, 1))
        order = [0, 1] if first == "right" else [1, 0]
        plan = order * X
        i = 0; regrasps = 0; errors = 0
        while i < len(plan) and trk.T[0] is not None:
            k = plan[i]
            if not held(R):
                if regrasps >= 4:
                    log("too many regrasps; abort swings")
                    break
                regrasps += 1
                log("cube not held before visit", i + 1, "-> regrasp")
                R.move_to([R.pos[0], R.pos[1], Z_HOVER + 0.05], grip=-1, tol=0.02)
                ok, yaw, _ = grasp(R, color, [t for t in trk.T if t is not None])
                continue
            try:
                okv, why = chase(R, trk, k, yaw)
            except rb.EpisodeEnded:
                raise
            except Exception:
                errors += 1
                log("chase ERROR", errors, traceback.format_exc())
                if errors > 4:
                    raise
                R.observe()
                continue
            if why == "dropped":
                log("cube dropped during visit", i + 1)
                continue
            log("visit done", i + 1, "/", len(plan), "RL"[k], okv, why, "step", R.step_id,
                "t", round(time.time() - t0, 1))
            i += 1
        log("swings done; step", R.step_id, "t", round(time.time() - t0, 1))
        if held(R):
            place(R, trk, yaw, cube_start)
        else:
            R.gripper(-1, 5)
            R.move_to([R.pos[0], R.pos[1], Z_HOVER], grip=-1, tol=0.02)
        offs = [(0, 0), (0.006, 0), (-0.006, 0), (0, 0.006), (0, -0.006), (0.01, 0.01), (-0.01, -0.01)]
        n = 0
        while R.remaining is None or R.remaining > 25:
            press(R, trk, offs[n % len(offs)])
            n += 1
            if n == 1:
                log("first press finished without episode end; step", R.step_id)
    except rb.EpisodeEnded:
        log("episode ended; step", R.step_id, "task_complete", R.task_complete)
    except Exception:
        log("ERROR", traceback.format_exc())
    log("main returns; step", R.step_id, "ended", R.ended, "wall", round(time.time() - t0, 1))
