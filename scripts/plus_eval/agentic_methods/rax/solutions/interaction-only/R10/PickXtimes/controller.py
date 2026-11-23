"""PickXtimes controller: pick the named cube, place it on the target N times, press the button."""
import re, time, json, os, traceback
import numpy as np
from robo import Robot, all_points, color_mask, components, hsv, log, LOG
from tracking import TargetTrack, place_tracking, observe_target, detect_target, place_v3
from target_model import TargetModel

WORDS = {"once": 1, "twice": 2, "thrice": 3, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
COLORS = ["green", "red", "blue", "yellow", "purple", "orange", "cyan", "magenta", "pink"]

OPEN, CLOSE = -1.0, 1.0
SAFE_Z = 0.14
LIFT_Z = 0.18
OBS_Z = 0.26


def parse_instruction(text):
    t = text.lower()
    color = None
    m = re.search(r"pick up the (\w+) cube", t)
    if m and m.group(1) in COLORS:
        color = m.group(1)
    else:
        for c in COLORS:
            if c + " cube" in t:
                color = c; break
    n = None
    m = re.search(r"(\w+) times", t)
    if m:
        w = m.group(1)
        n = int(w) if w.isdigit() else WORDS.get(w)
    if n is None:
        for w, v in WORDS.items():
            if re.search(r"\b" + w + r"\b", t):
                n = v; break
    return color or "green", n or 1


def scene_points(R, cam="wrist"):
    meta, imgs = R.meta, R.imgs
    rgb = imgs[(cam, "rgb")]
    P = all_points(meta, cam, imgs[(cam, "depth")])
    return rgb, P


def find_cube(R, color, near=None):
    rgb, P = scene_points(R)
    comps = components(color_mask(rgb, color), 6)
    best = None
    for c in comps:
        pts = P[c["mask"]]; pts = pts[~np.isnan(pts).any(1)]
        if len(pts) < 4:
            continue
        tp = pts[pts[:, 2] > np.max(pts[:, 2]) - 0.006]
        top = np.median(tp, 0)
        if top[2] < 0.036 or top[2] > 0.12 or top[0] < -0.45:
            continue
        score = c["area"]
        if near is not None:
            score = -np.linalg.norm(top[:2] - near[:2]) + 1e-4 * c["area"]
        if best is None or score > best[0]:
            best = (score, top, c["area"], tp)
    if best is None:
        return None
    return best[1], best[2], cube_yaw(best[3])


def cube_yaw(tp):
    """Yaw (rad, in [-pi/4, pi/4)) of the cube's top face from its world xy points."""
    try:
        import cv2
        xy = (tp[:, :2] * 1000.0).astype(np.float32)
        if len(xy) < 8:
            return 0.0
        (_, _), (w, h), ang = cv2.minAreaRect(xy)
        a = np.deg2rad(ang)
        a = (a + np.pi / 4) % (np.pi / 2) - np.pi / 4
        return float(a)
    except Exception:
        return 0.0


def find_target(R, color_hint=None):
    rgb, P = scene_points(R)
    z = P[..., 2]
    h = hsv(rgb)
    flat = (z > 0.003) & (z < 0.013) & (P[..., 0] > -0.45)
    comps = components(flat, 15)
    for c in comps:
        pts = P[c["mask"]]
        return np.nanmedian(pts, 0), c["area"]
    return None


def find_button(R):
    rgb, P = scene_points(R)
    z = P[..., 2]; h = hsv(rgb)
    m = (z > 0.010) & (z < 0.06) & (h[..., 1] < 70) & (P[..., 0] > -0.45)
    comps = components(m, 20)
    for c in comps:
        pts = P[c["mask"]]; pts = pts[~np.isnan(pts).any(1)]
        cap = pts[pts[:, 2] > pts[:, 2].max() - 0.006]
        xy = np.median(pts[:, :2], 0)
        return np.array([xy[0], xy[1], pts[:, 2].max()]), c["area"]
    return None


def lift(R, grip, z=SAFE_Z):
    p = R.state[:3].copy(); p[2] = z
    R.move_to(p, grip, tol=0.01, max_steps=40)


def goto_xy_safe(R, xy, grip, z=SAFE_Z):
    if R.state[2] < z - 0.02:
        lift(R, grip, z)
    R.move_to([xy[0], xy[1], z], grip, tol=0.006, max_steps=60)


def gripper_width(R):
    return float(R.state[6] + R.state[7])


def pick(R, color, guess):
    """guess: approximate xyz of cube top. Returns True if cube appears grasped."""
    goto_xy_safe(R, guess, OPEN, OBS_Z)
    R.move_to([guess[0], guess[1], OBS_Z], OPEN, tol=0.004, max_steps=30)
    R.observe()
    f = find_cube(R, color, near=guess)
    if f is None:
        log("pick: cube not seen from obs pose; using guess"); top = np.array(guess)
    else:
        top = f[0]
    yaw = f[2] if f is not None else 0.0
    # choose the face-aligned yaw closest to the current gripper yaw target
    R.goal_yaw = yaw
    log("pick: cube top", np.round(top, 4), "yaw", round(np.rad2deg(yaw), 1))
    R.move_to([top[0], top[1], top[2] + 0.06], OPEN, tol=0.004, max_steps=50)
    gz = max(top[2] - 0.022, 0.012)
    R.move_to([top[0], top[1], gz], OPEN, tol=0.003, max_steps=40)
    R.hold(CLOSE, 10)
    lift(R, CLOSE, LIFT_Z)
    R.hold(CLOSE, 2)
    w = gripper_width(R)
    log("pick: gripper width after lift", round(w, 4))
    return w > 0.01, gz


def place(R, tgt, grasp_z):
    """grasp_z: TCP height at which the cube was grasped from its resting pose."""
    R.move_to([tgt[0], tgt[1], LIFT_Z], CLOSE, tol=0.006, max_steps=60)
    R.move_to([tgt[0], tgt[1], grasp_z + 0.03], CLOSE, tol=0.004, max_steps=50)
    R.move_to([tgt[0], tgt[1], grasp_z + 0.004], CLOSE, tol=0.003, max_steps=30)
    R.hold(OPEN, 8)
    lift(R, OPEN, SAFE_Z)


def press_button(R, btn):
    R.goal_yaw = 0.0
    R.hold(OPEN, 5)
    goto_xy_safe(R, btn, CLOSE, SAFE_Z)
    R.move_to([btn[0], btn[1], btn[2] + 0.04], CLOSE, tol=0.004, max_steps=40)
    for dz in [0.0, -0.008, -0.014, -0.02]:
        if R.ended:
            break
        R.move_to([btn[0], btn[1], btn[2] + dz], CLOSE, tol=0.003, max_steps=25)
        R.hold(CLOSE, 3)
        log("press: z", round(R.state[2], 4), "ended", R.ended, "complete", R.complete)
    if not R.ended:
        lift(R, CLOSE, SAFE_Z)


def main(ctx):
    try:
        run(ctx)
    except Exception:
        log("controller exception:", traceback.format_exc())
    finally:
        try:
            os.makedirs("/workspace/output", exist_ok=True)
            open("/workspace/output/controller_log.txt", "w").write("\n".join(LOG))
        except Exception:
            pass


def survey(R, color):
    poses = [None, (0.0, 0.0, 0.55), (-0.15, 0.0, 0.55), (0.05, 0.12, 0.5), (0.05, -0.12, 0.5)]
    cube = tgt = btn = None
    for p in poses:
        if p is None:
            R.move_to([R.state[0], R.state[1], 0.52], OPEN, tol=0.01, max_steps=10)
        else:
            R.move_to(p, OPEN, tol=0.01, max_steps=30)
        R.observe()
        cube = cube or find_cube(R, color)
        tgt = tgt or find_target(R)
        btn = btn or find_button(R)
        log("survey at", np.round(R.state[:3], 3), "cube", cube and np.round(cube[0], 3),
            "target", tgt and np.round(tgt[0], 3), "button", btn and np.round(btn[0], 3))
        if cube and tgt and btn:
            break
    return cube, tgt, btn


def wait_for_lap(R, model, max_wait=260):
    """Hold the grasped cube high until the target model has seen a full lap (or is static)."""
    if model.has_lap():
        return True
    p = R.state[:3].copy()
    R.move_to([p[0], p[1], 0.32], CLOSE, tol=0.01, max_steps=15)
    n = 0
    while n < max_wait and not R.ended:
        if n % 10 == 0 and model.has_lap():
            log("lap acquired after", n, "wait steps; samples", len(model.hist))
            return True
        R.hold(CLOSE, 2)
        n += 2
    log("lap not acquired; proceeding with local prediction")
    return False


def wait_fresh(R, model, max_wait=40):
    """Make sure the target model has fresh clean samples before a placement approach."""
    n = 0
    while not R.ended and n < max_wait:
        if model.fresh(int(R.meta.get("step_id", 0))):
            return True
        if R.state[2] < 0.30:
            p = R.state[:3].copy()
            R.move_to([p[0], p[1], 0.32], CLOSE, tol=0.01, max_steps=12)
            n += 12
        else:
            R.hold(CLOSE, 2)
            n += 2
    log("wait_fresh: no fresh target samples after", n)
    return False


def verify_place(R, model, color, rel_xy):
    """Lift high, watch the target: a counted placement freezes the target near the cube."""
    lift(R, OPEN, 0.31)
    pts = []
    for _ in range(4):
        R.hold(OPEN, 2)
        d = detect_target(R, prefer="front")
        if d is not None:
            pts.append(d[0][:2])
    f = find_cube(R, color, near=np.array([rel_xy[0], rel_xy[1], 0.04]))
    cube = None if f is None else f[0]
    moving = None
    if len(pts) >= 3:
        moving = float(np.linalg.norm(pts[-1] - pts[0]))
    dist = None if (cube is None or not pts) else float(np.linalg.norm(cube[:2] - pts[-1]))
    log("verify: cube", None if cube is None else np.round(cube, 3), "target", None if not pts else np.round(pts[-1], 3),
        "target drift", moving, "dist", dist)
    frozen = moving is not None and moving < 0.006
    return frozen, cube


def run(ctx):
    t0 = time.time()
    R = Robot(ctx)
    model = TargetModel()
    R.hooks.append(model.update)
    R.observe()
    instr = R.meta.get("instruction", "")
    color, n = parse_instruction(instr)
    log("instruction:", instr, "-> color", color, "n", n)
    cube, tgt, btn = survey(R, color)
    cube_top = cube[0] if cube else None
    btn_p = btn[0] if btn else None
    cube_h = 0.04 if cube_top is None else float(np.clip(cube_top[2], 0.02, 0.06))
    obstacles = []
    for c in COLORS:
        if c == color:
            continue
        f = find_cube(R, c)
        if f is not None:
            obstacles.append(f[0])
    if btn_p is not None:
        obstacles.append(btn_p)
    log("obstacles", [np.round(o, 3).tolist() for o in obstacles])
    done = 0
    guess = cube_top
    attempts = 0
    while done < n and not R.ended and attempts < 2 * n + 6:
        attempts += 1
        if R.remaining() < 150:
            log("low steps; stop picking"); break
        if guess is None:
            cube, _, btn2 = survey(R, color)
            guess = cube[0] if cube else None
            btn_p = btn_p if btn_p is not None else (btn2[0] if btn2 else None)
            if guess is None:
                log("cannot find cube; skip"); continue
        model.cube_xy = np.asarray(guess)[:2]
        ok, grasp_z = pick(R, color, guess)
        if ok:
            model.cube_xy = None
        if not ok:
            log("pick failed; re-survey")
            R.move_to([R.state[0], R.state[1], 0.45], OPEN, tol=0.01, max_steps=30)
            R.observe(); f = find_cube(R, color)
            guess = f[0] if f else None
            continue
        wait_for_lap(R, model)
        wait_fresh(R, model)
        released, rel_xy, tgt_xy = place_v3(R, model, grasp_z, obstacles)
        if not released:
            log("place: no release within budget; retry")
            released, rel_xy, tgt_xy = place_v3(R, model, grasp_z, obstacles)
        if not released:
            R.hold(OPEN, 6); lift(R, OPEN, SAFE_Z)
            R.observe(); f = find_cube(R, color); guess = f[0] if f else None
            continue
        model.cube_xy = np.asarray(rel_xy)[:2]
        frozen, cube_now = verify_place(R, model, color, rel_xy)
        guess = cube_now if cube_now is not None else np.array([rel_xy[0], rel_xy[1], cube_h])
        model.cube_xy = np.asarray(guess)[:2]
        if model.ever_moved and not frozen:
            log("placement NOT counted (target kept moving); redo; done", done, "step", R.meta.get("step_id"))
            continue
        done += 1
        log("cycle", done, "of", n, "frozen", frozen, "moving-episode", model.ever_moved,
            "step", R.meta.get("step_id"), "t", round(time.time() - t0, 1))
    if not R.ended and btn_p is not None:
        press_button(R, btn_p)
    log("final: ended", R.ended, "complete", R.complete, "step", R.meta.get("step_id"),
        "t", round(time.time() - t0, 1))
    try:
        os.makedirs("/workspace/output", exist_ok=True)
        json.dump({"hist": {str(k): v for k, v in model.hist.items()}},
                  open("/workspace/output/target_hist.json", "w"))
    except Exception:
        pass
