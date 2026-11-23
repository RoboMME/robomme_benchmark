# v02 — SwingXtimes (robomme_plus task_10): v01 + closed-loop target visits. Targets can jump
# away when the held cube approaches (ep2 evidence), so every visit re-localizes the scheduled
# target from the front camera at transit height (and after landing) and re-aims before/after descent.
# Pick the language-named cube, visit right/left targets X times while holding it,
# put it down, press the button. Per-tick joint servo via step() with precomputed IK.
import math

FRAME_RIGHT_IS_NEG_Y = True   # hypothesis: "right/left" in the robot's own frame (right = -y)

report = {"version": "v02", "phases": [], "ticks": 0}
context = get_task_context()
language = str(context.language).lower()
report["language"] = language
state0 = get_robot_state()
base_frame = state0.base_frame
embodiment = state0.embodiment
home_quat = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]
terminal = {"done": False}

# ------------------------------------------------------------------ language
WORDS = {"one": 1, "once": 1, "two": 2, "twice": 2, "three": 3, "thrice": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
tokens = language.replace(",", " ").replace(".", " ").split()
count = None
for i, tok in enumerate(tokens):
    if tok in ("times", "time") and i > 0:
        prev = tokens[i - 1]
        if prev in WORDS:
            count = WORDS[prev]
        else:
            try:
                count = int(prev)
            except Exception:
                count = None
if count is None:
    for tok in tokens:
        if tok in WORDS and tok not in ("one",):
            count = WORDS[tok]
            break
if count is None:
    count = 1
report["count"] = count
pos_right = language.find("right")
pos_left = language.find("left")
first_side = "right"
if pos_left >= 0 and (pos_right < 0 or pos_left < pos_right):
    first_side = "left"
second_side = "left" if first_side == "right" else "right"
report["order"] = [first_side, second_side]

COLORS = {"red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0),
          "yellow": (1.0, 1.0, 0.0), "purple": (0.6, 0.0, 1.0), "orange": (1.0, 0.5, 0.0),
          "cyan": (0.0, 1.0, 1.0), "pink": (1.0, 0.4, 0.7), "white": (1.0, 1.0, 1.0),
          "black": (0.0, 0.0, 0.0)}
cube_color = None
for tok in tokens:
    if tok in COLORS:
        cube_color = tok
        break
report["cube_color"] = cube_color


# ------------------------------------------------------------------ helpers
def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def mean_color(pc):
    try:
        cols = pc.colors
        n = len(cols)
    except Exception:
        return None
    if n == 0:
        return None
    sr = 0.0
    sg = 0.0
    sb = 0.0
    for c in cols:
        sr += float(c[0])
        sg += float(c[1])
        sb += float(c[2])
    scale = 1.0
    if max(sr, sg, sb) / n > 1.5:
        scale = 255.0
    return (sr / n / scale, sg / n / scale, sb / n / scale)


def color_match(rgb, name):
    if rgb is None or name is None:
        return 0.0
    tgt = COLORS[name]
    total = max(1e-6, rgb[0] + rgb[1] + rgb[2])
    chrom = (rgb[0] / total, rgb[1] / total, rgb[2] / total)
    ttot = max(1e-6, tgt[0] + tgt[1] + tgt[2])
    tch = (tgt[0] / ttot, tgt[1] / ttot, tgt[2] / ttot)
    d = math.sqrt(sum((chrom[k] - tch[k]) ** 2 for k in range(3)))
    return 1.0 - d


def saturation(rgb):
    if rgb is None:
        return 1.0
    hi = max(rgb)
    lo = min(rgb)
    if hi <= 1e-6:
        return 0.0
    return (hi - lo) / hi


def candidates(prompt, camera="front"):
    out = []
    seg = segment_text(camera, prompt)
    if not seg.ok:
        return out
    for cand in seg.segmentations[:5]:
        try:
            pc = mask_to_point_cloud(cand, camera, target_frame=base_frame)
        except Exception:
            continue
        pts = pc.points
        if len(pts) < 15:
            continue
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        zs = [float(p[2]) for p in pts]
        out.append({
            "score": float(cand.score), "pts": pts, "n": len(pts),
            "x": sum(xs) / len(xs), "y": sum(ys) / len(ys),
            "zlo": quantile(zs, 0.05), "zhi": quantile(zs, 0.95), "zmed": quantile(zs, 0.5),
            "rgb": mean_color(pc), "box": [float(v) for v in cand.box_xyxy],
        })
    return out


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def yaw_quat(theta):
    return quat_mul([math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)], home_quat)


def top_face_rect(pts):
    zs = [float(p[2]) for p in pts]
    top_z = quantile(zs, 0.9)
    top = [p for p in pts if float(p[2]) >= top_z - 0.006]
    if len(top) < 8:
        top = pts
    best = None
    for k in range(30):
        a = math.radians(3.0 * k)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [float(p[0]) * ca + float(p[1]) * sa for p in top]
        vs = [-float(p[0]) * sa + float(p[1]) * ca for p in top]
        u0 = quantile(us, 0.03)
        u1 = quantile(us, 0.97)
        v0 = quantile(vs, 0.03)
        v1 = quantile(vs, 0.97)
        area = (u1 - u0) * (v1 - v0)
        if best is None or area < best[0]:
            uc = 0.5 * (u0 + u1)
            vc = 0.5 * (v0 + v1)
            best = (area, a, uc * ca - vc * sa, uc * sa + vc * ca, u1 - u0, v1 - v0)
    return {"top_z": top_z, "angle": best[1], "cx": best[2], "cy": best[3],
            "side": 0.5 * (best[4] + best[5])}


def ik_joints(xyz, quat, seed):
    pose = Pose((float(xyz[0]), float(xyz[1]), float(xyz[2])), quat, base_frame)
    res = solve_ik(pose, seed_joints=seed)
    if not res.ok:
        return None
    try:
        return [float(v) for v in res.joint_positions]
    except Exception:
        pass
    try:
        return [float(v) for v in res.joint_positions["primary"]]
    except Exception:
        return None


def measured():
    return [float(v) for v in get_robot_state().joint_positions["primary"]]


def tick(joints, grip):
    if terminal["done"]:
        return False
    res = step(RobotAction(arms={"primary": ArmCommand(
        "joint_position", [float(v) for v in joints], gripper_position=float(grip),
        embodiment=embodiment)}))
    report["ticks"] += 1
    try:
        if res.terminated or res.truncated:
            terminal["done"] = True
            report["terminal_at_tick"] = report["ticks"]
            report["terminal_flags"] = [bool(res.terminated), bool(res.truncated)]
            return False
    except Exception:
        pass
    return True


def servo(start, goal, n, grip):
    for k in range(1, n + 1):
        alpha = 0.5 - 0.5 * math.cos(math.pi * k / n)
        q = [start[j] + alpha * (goal[j] - start[j]) for j in range(7)]
        if not tick(q, grip):
            return False
    return True


def settle(goal, grip, max_ticks, tol):
    for k in range(max_ticks):
        if terminal["done"]:
            return False
        cur = measured()
        err = max(abs(cur[j] - goal[j]) for j in range(7))
        if err < tol:
            return True
        if not tick(goal, grip):
            return False
    return True


def ee_pos():
    p = get_robot_state().end_effector_poses["primary"].position
    return [round(float(v), 4) for v in p]


def phase(name):
    entry = {"name": name, "tick": report["ticks"]}
    if not terminal["done"]:
        try:
            entry["ee"] = ee_pos()
            entry["grip"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
        except Exception:
            pass
    report["phases"].append(entry)


# ------------------------------------------------------------------ perception (arm at home)
cube_cands = candidates("cube")
for extra in candidates(("%s cube" % cube_color) if cube_color else "block"):
    cube_cands.append(extra)
cube = None
best_val = None
for c in cube_cands:
    if c["zhi"] is None or c["zhi"] < 0.02 or c["zhi"] > 0.08:
        continue
    val = color_match(c["rgb"], cube_color) * 2.0 + 0.3 * c["score"]
    if best_val is None or val > best_val:
        best_val = val
        cube = c
if cube is None:
    raise ValueError("cube not found")
rect = top_face_rect(cube["pts"])
side = min(0.06, max(0.025, rect["side"]))
report["cube"] = {"x": round(rect["cx"], 4), "y": round(rect["cy"], 4), "top_z": round(rect["top_z"], 4),
                  "side": round(side, 4), "angle_deg": round(math.degrees(rect["angle"]), 1),
                  "rgb": cube["rgb"], "score": cube["score"]}

target_cands = candidates("circular target") + candidates("target")
targets = []
ranked = []
for c in target_cands:
    pos = 0
    while pos < len(ranked) and ranked[pos]["score"] >= c["score"]:
        pos += 1
    ranked.insert(pos, c)
for c in ranked:
    if c["n"] < 60 or c["zhi"] is None or c["zhi"] > 0.03 or c["zhi"] - c["zlo"] > 0.015:
        continue
    dup = False
    for t in targets:
        if math.sqrt((t["x"] - c["x"]) ** 2 + (t["y"] - c["y"]) ** 2) < 0.05:
            dup = True
    if not dup:
        targets.append(c)
if len(targets) < 2:
    raise ValueError("fewer than 2 targets found")
targets = targets[:2]
if targets[0]["y"] > targets[1]["y"]:
    targets = [targets[1], targets[0]]   # [lower y, higher y]
if FRAME_RIGHT_IS_NEG_Y:
    side_target = {"right": targets[0], "left": targets[1]}
else:
    side_target = {"right": targets[1], "left": targets[0]}
report["targets"] = {k: [round(side_target[k]["x"], 4), round(side_target[k]["y"], 4),
                         round(side_target[k]["zhi"], 4)] for k in side_target}

button = None
for c in candidates("button"):
    if c["zhi"] is None or c["zhi"] < 0.02 or c["zhi"] > 0.1:
        continue
    if saturation(c["rgb"]) > 0.45:
        continue
    too_close = False
    for t in targets:
        if math.sqrt((t["x"] - c["x"]) ** 2 + (t["y"] - c["y"]) ** 2) < 0.04:
            too_close = True
    if too_close:
        continue
    if button is None or c["score"] > button["score"]:
        button = c
if button is None:
    raise ValueError("button not found")
button_top = button["zhi"]
report["button"] = {"x": round(button["x"], 4), "y": round(button["y"], 4), "top_z": round(button_top, 4),
                    "score": button["score"], "rgb": button["rgb"]}

# ------------------------------------------------------------------ geometry helpers
TRANSIT = 0.09
CLEAR_XY = 0.07   # held cube must stay this far (xy) from a target it is not scheduled to visit


def seg_dist(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    L2 = dx * dx + dy * dy
    t = 0.0
    if L2 > 1e-9:
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    qx = ax + t * dx
    qy = ay + t * dy
    return math.sqrt((px - qx) ** 2 + (py - qy) ** 2)


def detour(a, b, avoid):
    """Return None if the straight xy path a->b clears `avoid`, else a clearing waypoint xy."""
    if seg_dist(avoid[0], avoid[1], a[0], a[1], b[0], b[1]) >= CLEAR_XY:
        return None
    best = None
    for dx in (-0.12, 0.12, -0.18, 0.18):
        w = (avoid[0] + dx, avoid[1])
        d1 = seg_dist(avoid[0], avoid[1], a[0], a[1], w[0], w[1])
        d2 = seg_dist(avoid[0], avoid[1], w[0], w[1], b[0], b[1])
        if min(d1, d2) >= CLEAR_XY and w[0] > 0.25:
            return w
        if best is None or min(d1, d2) > best[0]:
            best = (min(d1, d2), w)
    return best[1]


def wrap_angle(a):
    while a > math.pi / 4:
        a -= math.pi / 2
    while a < -math.pi / 4:
        a += math.pi / 2
    return a


def safe_ik(xyz, quat, seed):
    if terminal["done"]:
        return None
    try:
        return ik_joints(xyz, quat, seed)
    except Exception as exc:
        report["ik_error"] = str(exc)[:200]
        return None


# ------------------------------------------------------------------ stage 1: hover over cube (front estimate)
home_j = measured()
front_quat = yaw_quat(wrap_angle(rect["angle"]))
hover_z = rect["top_z"] + 0.12
hover_j = ik_joints((rect["cx"], rect["cy"], hover_z), front_quat, home_j)
if hover_j is None:
    raise ValueError("IK failed for hover")
open_gripper()
phase("start")
servo(home_j, hover_j, 30, 1.0)
settle(hover_j, 1.0, 12, 0.01)
phase("hover")

# ------------------------------------------------------------------ stage 2: wrist refinement
ref = None
wrist_prompts = ["cube"]
if cube_color:
    wrist_prompts = ["%s cube" % cube_color, "cube"]
wrist_log = []
for prompt in wrist_prompts:
    if ref is not None or terminal["done"]:
        break
    try:
        wc = candidates(prompt, "wrist")
    except Exception as exc:
        wrist_log.append(str(exc)[:120])
        wc = []
    bestv = None
    for c in wc:
        d = math.sqrt((c["x"] - rect["cx"]) ** 2 + (c["y"] - rect["cy"]) ** 2)
        span = c["zhi"] - c["zlo"]
        info = {"prompt": prompt, "n": c["n"], "d": round(d, 4), "rgb": c["rgb"], "score": round(c["score"], 3),
                "zhi": round(c["zhi"], 4)}
        wrist_log.append(info)
        if d > 0.06 or c["zhi"] < 0.02 or c["zhi"] > 0.08 or c["n"] < 30:
            continue
        if cube_color and color_match(c["rgb"], cube_color) < 0.6:
            continue
        v = c["score"] - 2.0 * d
        if bestv is None or v > bestv:
            bestv = v
            ref = c
report["wrist_log"] = wrist_log
if ref is not None:
    wrect = top_face_rect(ref["pts"])
    report["cube_wrist"] = {"x": round(wrect["cx"], 4), "y": round(wrect["cy"], 4),
                            "top_z": round(wrect["top_z"], 4), "side": round(wrect["side"], 4),
                            "angle_deg": round(math.degrees(wrect["angle"]), 1)}
    if math.sqrt((wrect["cx"] - rect["cx"]) ** 2 + (wrect["cy"] - rect["cy"]) ** 2) < 0.06:
        rect = wrect
report["cube_used"] = {"x": round(rect["cx"], 4), "y": round(rect["cy"], 4), "top_z": round(rect["top_z"], 4),
                       "side": round(rect["side"], 4), "angle_deg": round(math.degrees(rect["angle"]), 1),
                       "refined": ref is not None}

side = min(0.06, max(0.03, rect["side"]))
grasp_quat = yaw_quat(wrap_angle(rect["angle"]))
grasp_z = max(0.015, rect["top_z"] - 0.5 * side)
bottom_offset = grasp_z - max(0.0, rect["top_z"] - side)     # EE z minus held-cube bottom z
report["grasp"] = {"z": round(grasp_z, 4), "bottom_offset": round(bottom_offset, 4), "side": round(side, 4)}

cur_j = measured()
pre_j = ik_joints((rect["cx"], rect["cy"], grasp_z + 0.08), grasp_quat, cur_j)
grasp_j = ik_joints((rect["cx"], rect["cy"], grasp_z), grasp_quat, pre_j)
if pre_j is None or grasp_j is None:
    raise ValueError("IK failed for grasp")

# ------------------------------------------------------------------ closed-loop target tracking
tpos = {}
for name in ("right", "left"):
    tpos[name] = [side_target[name]["x"], side_target[name]["y"], side_target[name]["zhi"]]
other_of = {first_side: second_side, second_side: first_side}
report["relocations"] = []


def relocate(name):
    """Re-detect the scheduled target near its last known xy (front camera). None if not found."""
    if terminal["done"]:
        return None
    me = tpos[name]
    ot = tpos[other_of[name]]
    best = None
    for prompt in ("circular target", "target"):
        try:
            cs = candidates(prompt, "front")
        except Exception:
            cs = []
        for c in cs:
            if c["n"] < 30 or c["zhi"] > 0.03 or c["zhi"] - c["zlo"] > 0.015:
                continue
            dm = math.sqrt((c["x"] - me[0]) ** 2 + (c["y"] - me[1]) ** 2)
            do = math.sqrt((c["x"] - ot[0]) ** 2 + (c["y"] - ot[1]) ** 2)
            if dm > 0.2 or do < dm or do < 0.08:
                continue
            if best is None or dm < best[0]:
                best = (dm, [c["x"], c["y"], c["zhi"]], c["n"])
        if best is not None:
            break
    if best is None:
        return None
    return best[1]


def high_of(name):
    t = tpos[name]
    return (t[0], t[1], t[2] + TRANSIT + bottom_offset)


def low_of(name):
    t = tpos[name]
    return (t[0], t[1], t[2] + bottom_offset)


def track_and_aim(name, joints, label):
    """At transit height over `name`: re-detect up to 3 times and re-aim. Returns commanded joints."""
    cur = joints
    for attempt in range(3):
        if terminal["done"]:
            return cur
        for k in range(3):
            tick(cur, 0.0)
        p = relocate(name)
        if p is None:
            report["relocations"].append({"visit": label, "attempt": attempt, "found": False})
            return cur
        d = math.sqrt((p[0] - tpos[name][0]) ** 2 + (p[1] - tpos[name][1]) ** 2)
        report["relocations"].append({"visit": label, "attempt": attempt, "d": round(d, 4),
                                      "xy": [round(p[0], 4), round(p[1], 4)]})
        if d < 0.012:
            return cur
        tpos[name] = p
        nj = safe_ik(high_of(name), grasp_quat, cur)
        if nj is None:
            return cur
        servo(cur, nj, max(6, min(20, int(d / 0.006))), 0.0)
        settle(nj, 0.0, 4, 0.01)
        cur = nj
    return cur


def descend_visit(name, high_j, label):
    """Descend onto the target, then check it did not move during the descent (one re-aim)."""
    low_j = safe_ik(low_of(name), grasp_quat, high_j)
    if low_j is None:
        return high_j
    servo(high_j, low_j, 10, 0.0)
    settle(low_j, 0.0, 6, 0.01)
    for hold in range(3):
        tick(low_j, 0.0)
    p = relocate(name)
    if p is not None:
        d = math.sqrt((p[0] - tpos[name][0]) ** 2 + (p[1] - tpos[name][1]) ** 2)
        if d >= 0.015:
            report["relocations"].append({"visit": label, "after_landing": True, "d": round(d, 4)})
            servo(low_j, high_j, 8, 0.0)
            tpos[name] = p
            nh = safe_ik(high_of(name), grasp_quat, high_j)
            if nh is not None:
                servo(high_j, nh, max(6, min(20, int(d / 0.006))), 0.0)
                nh = track_and_aim(name, nh, label + "b")
                low_j = safe_ik(low_of(name), grasp_quat, nh)
                if low_j is None:
                    return nh
                servo(nh, low_j, 10, 0.0)
                settle(low_j, 0.0, 6, 0.01)
                for hold in range(3):
                    tick(low_j, 0.0)
    return low_j


# ------------------------------------------------------------------ IK for the first transit
ft = side_target[first_side]
st = side_target[second_side]
lift_z = grasp_z + TRANSIT
mz0 = max(ft["zhi"], st["zhi"]) + TRANSIT + bottom_offset
first_high = ik_joints(high_of(first_side), grasp_quat, grasp_j)
if first_high is None:
    raise ValueError("IK failed for first target")
w1 = detour((rect["cx"], rect["cy"]), (ft["x"], ft["y"]), (st["x"], st["y"]))
first_route = []
if w1 is not None:
    wj = ik_joints((w1[0], w1[1], max(lift_z, mz0)), grasp_quat, grasp_j)
    if wj is not None:
        first_route.append(wj)
report["first_detour"] = w1

# ------------------------------------------------------------------ execute: pick
servo(measured(), pre_j, 12, 1.0)
servo(pre_j, grasp_j, 12, 1.0)
settle(grasp_j, 1.0, 10, 0.01)
phase("at_grasp")
if not terminal["done"]:
    closed = close_gripper()
    try:
        report["close_steps"] = int(closed.steps_executed)
    except Exception:
        pass
phase("closed")
lift_j = safe_ik((rect["cx"], rect["cy"], lift_z), grasp_quat, grasp_j)
if lift_j is None:
    lift_j = pre_j
servo(grasp_j, lift_j, 12, 0.0)
prev = lift_j
for wj in first_route:
    servo(prev, wj, 15, 0.0)
    prev = wj
servo(prev, first_high, 22, 0.0)
settle(first_high, 0.0, 4, 0.01)
cur_high = track_and_aim(first_side, first_high, "v1")
cur_low = descend_visit(first_side, cur_high, "v1")
phase("visit_1_" + first_side)

# ------------------------------------------------------------------ swings
current = first_side
visits = 1
total_visits = 2 * count
while visits < total_visits and not terminal["done"]:
    nxt = other_of[current]
    hj = safe_ik(high_of(current), grasp_quat, cur_low)
    if hj is None:
        break
    servo(cur_low, hj, 8, 0.0)
    a = tpos[current]
    b2 = tpos[nxt]
    mid = ((a[0] + b2[0]) * 0.5, (a[1] + b2[1]) * 0.5, max(a[2], b2[2]) + TRANSIT + bottom_offset)
    mj = safe_ik(mid, grasp_quat, hj)
    nh = safe_ik(high_of(nxt), grasp_quat, mj if mj is not None else hj)
    if nh is None:
        break
    if mj is not None:
        servo(hj, mj, 12, 0.0)
        servo(mj, nh, 12, 0.0)
    else:
        servo(hj, nh, 24, 0.0)
    settle(nh, 0.0, 3, 0.015)
    label = "v%d" % (visits + 1)
    nh = track_and_aim(nxt, nh, label)
    cur_low = descend_visit(nxt, nh, label)
    visits += 1
    current = nxt
    phase("visit_%d_%s" % (visits, current))

plan = {}
if not terminal["done"]:
    plan[current + "_low"] = cur_low
    plan[current + "_high"] = safe_ik(high_of(current), grasp_quat, cur_low)
    if plan[current + "_high"] is None:
        plan[current + "_high"] = cur_low
side_target[current]["x"] = tpos[current][0]
side_target[current]["y"] = tpos[current][1]
side_target[other_of[current]]["x"] = tpos[other_of[current]][0]
side_target[other_of[current]]["y"] = tpos[other_of[current]][1]
report["tpos_final"] = {k: [round(tpos[k][0], 4), round(tpos[k][1], 4)] for k in tpos}

# ------------------------------------------------------------------ put down + button
if not terminal["done"]:
    for k in range(10):
        tick(plan[current + "_low"], 1.0)
    phase("released")
bx = button["x"]
by = button["y"]
other = side_target[first_side if current == second_side else second_side]
ct = side_target[current]
above_b = safe_ik((bx, by, button_top + 0.07), home_quat, plan.get(current + "_high"))
press_b = None
if above_b is not None:
    press_b = safe_ik((bx, by, button_top - 0.015), home_quat, above_b)
w2 = detour((ct["x"], ct["y"]), (bx, by), (other["x"], other["y"]))
report["button_detour"] = w2
w2j = None
if w2 is not None:
    w2j = safe_ik((w2[0], w2[1], button_top + 0.07), home_quat, plan.get(current + "_high"))
if not terminal["done"]:
    if above_b is None or press_b is None:
        raise ValueError("IK failed for button")
    servo(plan[current + "_low"], plan[current + "_high"], 8, 1.0)
    prev = plan[current + "_high"]
    if w2j is not None:
        servo(prev, w2j, 15, 0.0)
        prev = w2j
    servo(prev, above_b, 22, 0.0)
    settle(above_b, 0.0, 8, 0.02)
    phase("above_button")
    servo(above_b, press_b, 12, 0.0)
    for hold in range(8):
        tick(press_b, 0.0)
    phase("pressed")
    servo(press_b, above_b, 10, 0.0)
    phase("after_press")

report["visits"] = visits
report["terminal"] = terminal["done"]
if not terminal["done"]:
    try:
        md = get_task_context().metadata
        report["steps_used_md"] = str(md["episode_steps_used"])
    except Exception:
        pass
result = report
