# v01 — tracking identity: the cube tracker must never accept the flat target disc (seed-2
# lock-on undercounted arrivals); fallback prompt; widening gate after misses; faster transit.
# (derived from v00) StopCube: press the button just as the moving cube reaches the target for the N-th time.
# Strategy (from the Stage 0 probe, hypotheses to verify):
#   * N is parsed from get_task_context().language.
#   * target = flat disc ("target"), button = raised cap ("round button"/"button"),
#     cube = small moving cube ("cube"), all on the front camera.
#   * The arm is walked tick-by-tick (own joint interpolation through step()) to hover just
#     above the button while the cube is tracked EVERY tick with SAM3 + depth projection.
#   * An "arrival" = cube xy centre entering the target radius (hysteresis on exit).
#   * After N-1 arrivals, on the N-th approach, predict ticks-to-closest-approach from the
#     per-tick velocity and issue the press PRESS_LEAD ticks ahead of it.
import math

PRESS_LEAD = 2          # ticks from press command to registration (hypothesis; measure)
HOVER_CLEAR = 0.02      # ee above button cap while waiting
PRESS_DEPTH = 0.02      # commanded ee below cap for the press
CAM = "front"

WORDS = {"first": 1, "once": 1, "second": 2, "twice": 2, "third": 3, "thrice": 3,
         "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
         "tenth": 10, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def parse_count(text):
    found = None
    for raw in text.lower().replace(",", " ").replace(".", " ").split():
        word = raw.strip()
        if word in WORDS:
            found = WORDS[word]
        else:
            digits = ""
            for ch in word:
                if ch in "0123456789":
                    digits = digits + ch
                else:
                    break
            if digits != "" and (word == digits or word[len(digits):] in ("st", "nd", "rd", "th")):
                found = int(digits)
    return found


def median(values):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    return ordered[len(ordered) // 2]


def quant(values, frac):
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * frac)]


ctx = get_task_context()
lang = ctx.language
N = parse_count(lang)
if N is None:
    N = 1
meta = robomme_plus.get_task_metadata()
max_steps = int(meta["max_steps"])
state0 = get_robot_state()
base = state0.base_frame
emb = state0.embodiment
home_j = [float(v) for v in state0.joint_positions["primary"]]
quat = state0.end_effector_poses["primary"].quaternion_wxyz

report = {"language": lang, "N": N, "max_steps": max_steps, "log": [], "arrivals": [],
          "events": []}


def cloud_info(cand):
    pc = mask_to_point_cloud(cand, CAM, target_frame=base)
    pts = pc.points
    if len(pts) < 8:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    zs = [float(p[2]) for p in pts]
    return {"x": median(xs), "y": median(ys), "zmax": quant(zs, 0.95), "zmin": quant(zs, 0.05),
            "sx": quant(xs, 0.95) - quant(xs, 0.05), "sy": quant(ys, 0.95) - quant(ys, 0.05),
            "n": len(pts)}


# ---------- static localization ----------
target = None
for prompt in ("target", "circle", "round target"):
    seg = segment_text(CAM, prompt)
    if not seg.ok:
        continue
    for cand in seg.segmentations[:4]:
        info = cloud_info(cand)
        if info is None:
            continue
        if 0.03 <= max(info["sx"], info["sy"]) <= 0.12 and info["zmax"] - info["zmin"] < 0.035:
            target = info
            target["prompt"] = prompt
            break
    if target is not None:
        break
if target is None:
    raise ValueError("target not found")

button = None
for prompt in ("round button", "button"):
    seg = segment_text(CAM, prompt)
    if not seg.ok:
        continue
    for cand in seg.segmentations[:4]:
        info = cloud_info(cand)
        if info is None:
            continue
        far = math.hypot(info["x"] - target["x"], info["y"] - target["y"]) > 0.08
        small = max(info["sx"], info["sy"]) <= 0.08
        raised = info["zmax"] > target["zmax"] + 0.008
        if far and small and raised:
            button = info
            button["prompt"] = prompt
            break
    if button is not None:
        break
if button is None:
    raise ValueError("button not found")

r_target = 0.5 * max(target["sx"], target["sy"])
r_in = r_target
r_out = r_target + 0.015
report["target"] = target
report["button"] = button
report["r_in"] = r_in


CUBE_MIN_TOP = target["zmax"] + 0.007   # a cube stands above the flat target disc


def track_once(prompt, prev, gate):
    seg = segment_text(CAM, prompt)
    if not seg.ok:
        return None
    best = None
    best_cost = None
    for cand in seg.segmentations[:4]:
        info = cloud_info(cand)
        if info is None:
            continue
        if max(info["sx"], info["sy"]) > 0.08:
            continue
        if math.hypot(info["x"] - button["x"], info["y"] - button["y"]) < 0.06:
            continue
        if info["zmax"] > 0.15 or info["zmax"] < CUBE_MIN_TOP:
            continue
        if prev is None:
            return info
        cost = math.hypot(info["x"] - prev["x"], info["y"] - prev["y"])
        if cost < gate and (best_cost is None or cost < best_cost):
            best = info
            best_cost = cost
    return best


def track_cube(prev, gap):
    gate = 0.05 + 0.03 * gap
    for prompt in ("cube", "small cube"):
        found = track_once(prompt, prev, gate)
        if found is not None:
            return found
    return None


# ---------- arm targets ----------
def ik_joints(pos, seed):
    ik = solve_ik(Pose(pos, quat, base), seed_joints=seed)
    if not ik.ok:
        return None
    return [float(v) for v in ik.joint_positions]


top = button["zmax"]
pre_j = ik_joints((button["x"], button["y"], top + 0.10), home_j)
hover_j = ik_joints((button["x"], button["y"], top + HOVER_CLEAR), pre_j if pre_j else home_j)
press_j = ik_joints((button["x"], button["y"], top - PRESS_DEPTH), hover_j if hover_j else home_j)
if pre_j is None or hover_j is None or press_j is None:
    raise ValueError("IK failed for button poses")

schedule = []          # list of joint targets, one per tick, for the transit
T1 = 20
T2 = 8
for i in range(1, T1 + 1):
    a = 0.5 - 0.5 * math.cos(math.pi * i / T1)
    schedule.append([home_j[j] + a * (pre_j[j] - home_j[j]) for j in range(7)])
for i in range(1, T2 + 1):
    a = 0.5 - 0.5 * math.cos(math.pi * i / T2)
    schedule.append([pre_j[j] + a * (hover_j[j] - pre_j[j]) for j in range(7)])

# ---------- main per-tick loop ----------
cube = track_cube(None, 0)
last_seen = 0
hist = []              # (tick, x, y)
count = 0
inside = False
pressed_at = None
tick = 0
ended = False
hold_after_press = 25
while tick < max_steps - 1 and not ended:
    if pressed_at is not None:
        cmd = press_j
    elif tick < len(schedule):
        cmd = schedule[tick]
    else:
        cmd = hover_j
    sr = step(RobotAction(arms={"primary": ArmCommand(
        "joint_position", cmd, gripper_position=0.0, embodiment=emb)}))
    tick += 1
    try:
        if sr.terminated or sr.truncated:
            ended = True
            report["events"].append(["ended", tick, bool(sr.terminated), bool(sr.truncated)])
            break
    except Exception:
        pass
    if pressed_at is not None and tick - pressed_at >= hold_after_press:
        break
    found = track_cube(cube, tick - last_seen)
    if found is None:
        report["log"].append([tick, None])
        continue
    if cube is None:
        prev_xy = (found["x"], found["y"])
    else:
        prev_xy = (cube["x"], cube["y"])
    cube = found
    last_seen = tick
    hist.append((tick, cube["x"], cube["y"]))
    d = math.hypot(cube["x"] - target["x"], cube["y"] - target["y"])
    report["log"].append([tick, round(cube["x"], 4), round(cube["y"], 4), round(d, 4),
                          round(cube["zmax"], 3)])
    # closest point of the segment travelled since the last fix (a fast cube can jump the disc)
    sx = cube["x"] - prev_xy[0]
    sy = cube["y"] - prev_xy[1]
    ss = sx * sx + sy * sy
    d_seg = d
    if ss > 1e-10:
        u = ((target["x"] - prev_xy[0]) * sx + (target["y"] - prev_xy[1]) * sy) / ss
        u = min(1.0, max(0.0, u))
        d_seg = min(d, math.hypot(prev_xy[0] + u * sx - target["x"], prev_xy[1] + u * sy - target["y"]))
    if not inside and d_seg < r_in:
        inside = True
        count += 1
        report["arrivals"].append([tick, round(d, 4)])
    elif inside and d > r_out:
        inside = False
    if pressed_at is not None:
        continue
    if tick < len(schedule):
        continue
    # N-th approach: predict closest approach
    want = False
    if count == N - 1 and not inside and len(hist) >= 3:
        t0, x0, y0 = hist[-3]
        t2, x2, y2 = hist[-1]
        dt = float(t2 - t0)
        if dt > 0:
            vx = (x2 - x0) / dt
            vy = (y2 - y0) / dt
            vv = vx * vx + vy * vy
            if vv > 1e-8:
                rx = cube["x"] - target["x"]
                ry = cube["y"] - target["y"]
                tau = -(rx * vx + ry * vy) / vv
                miss = math.hypot(rx + tau * vx, ry + tau * vy)
                if tau > 0 and miss < r_in and d < max(4 * r_target, 0.10) and tau <= PRESS_LEAD + 0.5:
                    want = True
                    report["events"].append(["predict", tick, round(tau, 2), round(miss, 4)])
    if count >= N and not want:
        want = True
        report["events"].append(["late_press", tick, count])
    if want:
        pressed_at = tick
        report["events"].append(["press", tick, count, round(d, 4)])

report["count"] = count
report["pressed_at"] = pressed_at
report["final_tick"] = tick
result = report
