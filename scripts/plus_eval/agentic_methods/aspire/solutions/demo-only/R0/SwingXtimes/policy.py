# robomme_plus / SwingXtimes (task 10) -- v05 button consistency (demo-only: written offline, never run).
# v05: the pre-press button re-detection must agree (<= 6 cm) with the clean start-of-episode detection
#      and must not be the just-released cube; otherwise the start detection is used.
# v04: low-score SAM candidates are not projected; a global segmentation-call cap; per-leg tick
#      accounting that drops follow/dwell/recenter refinements when the remaining horizon is short;
#      transit clearance 0.10 m so a diagonal disc-to-disc move stays above 4 cm distractor cubes.
# v03: never re-descend onto a disc already showing its red highlight; route the pick->first-disc and
#      last-disc->put-down transits around the other disc so it is never over-flown.
# v02: when two discs are seen, label them as a PAIR (left = larger base y) matched to memory;
#      nearest-to-memory only for a single detection.
# v01: post-lift check no longer filters by table-relative height and looks for the cube near the gripper.
#
# Goal template (parsed at runtime, never hardcoded):
#   "pick up the <color> cube, move it to the <A>-side target and then to the <B>-side target,
#    repeating this <A>-to-<B> swing motion <N> times, then put down the cube and press the
#    button to stop"
# Strategy: localize cube / two flat bullseye discs / button in the front camera, top-down grasp,
# 2N alternating closed-loop visits (discs can slide away, so re-localize before descending),
# hover the cube ~1 cm above each disc, put the cube back where it started, press the button.
import math

STATE = {"terminal": False, "log": [], "moves": 0, "seg_calls": 0, "tight": False}
SEG_CALL_CAP = 600

ctx = get_task_context()
goal = str(ctx.language)
report = {"goal": goal, "events": STATE["log"]}

meta = {}
try:
    meta = robomme_plus.get_task_metadata()
except Exception as exc:
    report["metadata_error"] = str(exc)


def meta_value(keys, default):
    for key in keys:
        try:
            if key in meta and meta[key] is not None:
                return meta[key]
        except Exception:
            pass
    return default


MAX_STEPS = int(meta_value(("max_steps",), 1000))
CONTROL_HZ = float(meta_value(("control_hz", "control_frequency_hz", "control_frequency", "frequency_hz"), 20.0))
state0 = get_robot_state()
BASE = state0.base_frame
EMB = state0.embodiment
report["max_steps"] = MAX_STEPS
report["control_hz"] = CONTROL_HZ

# ---------------------------------------------------------------- goal parsing
NUMBER_WORDS = {
    "once": 1, "twice": 2, "thrice": 3, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
COLOR_WORDS = ("red", "green", "blue", "yellow", "orange", "purple", "pink", "white", "black",
               "brown", "cyan", "magenta", "gray", "grey")


def clean_words(text):
    out = []
    word = ""
    for ch in text.lower():
        if ch.isalnum():
            word = word + ch
        else:
            if word:
                out.append(word)
            word = ""
    if word:
        out.append(word)
    return out


def parse_goal(text):
    words = clean_words(text)
    color = None
    for idx, word in enumerate(words):
        if word in ("cube", "block") and idx > 0:
            prev = words[idx - 1]
            if prev in COLOR_WORDS:
                color = prev
            break
    if color is None:
        for word in words:
            if word in COLOR_WORDS:
                color = word
                break
    low = text.lower()
    pos_left = low.find("left")
    pos_right = low.find("right")
    first = None
    if pos_left >= 0 and (pos_right < 0 or pos_left < pos_right):
        first = "left"
    elif pos_right >= 0:
        first = "right"
    count = None
    for idx, word in enumerate(words):
        if word in ("times", "time") and idx > 0:
            prev = words[idx - 1]
            if prev in NUMBER_WORDS:
                count = NUMBER_WORDS[prev]
            elif prev.isdigit():
                count = int(prev)
    if count is None:
        for word in words:
            if word in ("once", "twice", "thrice"):
                count = NUMBER_WORDS[word]
    second = None
    if first == "left":
        second = "right"
    elif first == "right":
        second = "left"
    return {"color": color, "first": first, "second": second, "count": count}


PLAN = parse_goal(goal)
report["plan"] = PLAN

# ---------------------------------------------------------------- small helpers


def note(kind, **fields):
    entry = {"kind": kind, "tick": now_ticks()}
    for key in fields:
        entry[key] = fields[key]
    STATE["log"].append(entry)


def now_ticks():
    try:
        return int(round(float(get_robot_state().timestamp_s) * CONTROL_HZ))
    except Exception:
        return -1


def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def is_terminal(res):
    if res is None:
        return False
    try:
        if bool(res.terminated):
            return True
    except Exception:
        pass
    try:
        if bool(res.truncated):
            return True
    except Exception:
        pass
    try:
        if bool(res.done):
            return True
    except Exception:
        pass
    try:
        status = str(res.status).lower()
        if "terminat" in status or "truncat" in status or "episode_end" in status:
            return True
    except Exception:
        pass
    return False


def check(res, label):
    ok = False
    try:
        ok = bool(res.ok)
    except Exception:
        ok = False
    status = "unknown"
    try:
        status = str(res.status)
    except Exception:
        pass
    steps = -1
    try:
        steps = int(res.steps_executed)
    except Exception:
        pass
    note("action", label=label, ok=ok, status=status, steps=steps)
    if is_terminal(res):
        STATE["terminal"] = True
        raise ValueError("EPISODE_TERMINAL after " + label)
    return ok, steps


def remaining_ticks():
    t = now_ticks()
    if t < 0:
        return MAX_STEPS
    return MAX_STEPS - t


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def quat_tool_z(q):
    w, x, y, z = q
    return (2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y))


def quat_yaw_x(q):
    w, x, y, z = q
    return math.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))


def wrap90(angle):
    quarter = math.pi / 2.0
    while angle > quarter / 2.0:
        angle = angle - quarter
    while angle < -quarter / 2.0:
        angle = angle + quarter
    return angle


def go(position, quat, label, tol=0.012, steps=150):
    target = Pose((float(position[0]), float(position[1]), float(position[2])), tuple(quat), BASE)
    STATE["moves"] = STATE["moves"] + 1
    res = move_to_pose(target, tolerance=tol, max_steps=steps)
    ok, executed = check(res, label)
    if not ok and executed == 0:
        res = move_to_pose(target, tolerance=max(tol, 0.02), max_steps=2 * steps)
        ok, executed = check(res, label + "_retry")
    return ok


def hold(ticks, grip, label):
    st = get_robot_state()
    joints = [float(v) for v in st.joint_positions["primary"]]
    for tick in range(ticks):
        res = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=joints, gripper_position=grip,
            embodiment=st.embodiment)}))
        if is_terminal(res):
            STATE["terminal"] = True
            raise ValueError("EPISODE_TERMINAL during " + label)
    note("hold", label=label, ticks=ticks)


# ---------------------------------------------------------------- perception helpers


def seg_candidates(camera, prompt, limit=6):
    out = []
    if STATE["seg_calls"] >= SEG_CALL_CAP:
        return out
    STATE["seg_calls"] = STATE["seg_calls"] + 1
    try:
        seg = segment_text(camera, prompt)
    except Exception as exc:
        note("segment_error", prompt=prompt, error=str(exc))
        return out
    try:
        if not bool(seg.ok):
            return out
    except Exception:
        pass
    items = None
    try:
        items = list(seg.segmentations)
    except Exception:
        items = None
    if items is not None:
        for cand in items[:limit]:
            entry = {"obj": cand, "mask": None, "score": 0.0, "box": None, "prompt": prompt}
            try:
                entry["score"] = float(cand.score)
            except Exception:
                pass
            try:
                entry["box"] = [float(v) for v in cand.box_xyxy]
            except Exception:
                pass
            try:
                entry["mask"] = cand.mask
            except Exception:
                pass
            if entry["score"] < 0.15 and len(out) > 0:
                continue
            out.append(entry)
        return out
    try:
        masks = list(seg.masks)
        scores = list(seg.scores)
        boxes = list(seg.boxes)
        for idx in range(min(limit, len(masks))):
            entry = {"obj": masks[idx], "mask": masks[idx], "score": float(scores[idx]),
                     "box": [float(v) for v in boxes[idx]], "prompt": prompt}
            out.append(entry)
    except Exception as exc:
        note("segment_shape_error", prompt=prompt, error=str(exc))
    return out


def cand_points(cand, camera):
    pc = None
    try:
        pc = mask_to_point_cloud(cand["obj"], camera, target_frame=BASE)
    except Exception:
        pc = None
    if pc is None and cand["mask"] is not None:
        try:
            pc = mask_to_point_cloud(cand["mask"], camera, target_frame=BASE)
        except Exception:
            pc = None
    pts = []
    if pc is None:
        return pts
    try:
        raw = pc.points
    except Exception:
        return pts
    for p in raw:
        x = float(p[0])
        y = float(p[1])
        z = float(p[2])
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
            pts.append((x, y, z))
    if len(pts) < 5:
        return pts
    mx = quantile([p[0] for p in pts], 0.5)
    my = quantile([p[1] for p in pts], 0.5)
    trimmed = []
    for p in pts:
        if abs(p[0] - mx) < 0.08 and abs(p[1] - my) < 0.08:
            trimmed.append(p)
    return trimmed


def describe(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    return {
        "x": quantile(xs, 0.5), "y": quantile(ys, 0.5),
        "z_lo": quantile(zs, 0.05), "z_mid": quantile(zs, 0.5), "z_hi": quantile(zs, 0.95),
        "span_x": quantile(xs, 0.95) - quantile(xs, 0.05),
        "span_y": quantile(ys, 0.95) - quantile(ys, 0.05),
        "n": len(pts),
    }


def box_center_u(cand):
    if cand["box"] is None:
        return None
    return 0.5 * (cand["box"][0] + cand["box"][2])


def red_fraction(cand, camera):
    """Fraction of a candidate's mask pixels that are saturated red (highlighted disc rings)."""
    if cand["mask"] is None or cand["box"] is None:
        return None
    try:
        rgb = get_observation().cameras[camera].rgb
        mask = cand["mask"]
        x1 = max(0, int(cand["box"][0]))
        y1 = max(0, int(cand["box"][1]))
        x2 = int(cand["box"][2])
        y2 = int(cand["box"][3])
        total = 0
        red = 0
        for v in range(y1, y2 + 1):
            for u in range(x1, x2 + 1):
                try:
                    inside = bool(mask[v][u])
                except Exception:
                    inside = False
                if not inside:
                    continue
                pix = rgb[v][u]
                r = int(pix[0])
                g = int(pix[1])
                b = int(pix[2])
                total = total + 1
                if r > 110 and r > 1.8 * g and r > 1.8 * b:
                    red = red + 1
        if total == 0:
            return None
        return float(red) / float(total)
    except Exception:
        return None


TARGET_PROMPTS = ("target", "bullseye target", "round white target on the table", "circular disc")
BUTTON_PROMPTS = ("button", "push button", "round button on a square base", "gray button")


def find_targets(table_z):
    """Flat round discs in the front camera. Returns list of dicts with x, y, z, u, cand."""
    found = []
    for prompt in TARGET_PROMPTS:
        for cand in seg_candidates("front", prompt):
            pts = cand_points(cand, "front")
            if len(pts) < 8:
                continue
            d = describe(pts)
            flat = (d["z_hi"] - d["z_lo"]) < 0.015
            if table_z is not None:
                flat = flat and (d["z_hi"] - table_z) < 0.02
            size_ok = 0.015 < max(d["span_x"], d["span_y"]) < 0.12
            if not (flat and size_ok):
                continue
            duplicate = False
            for other in found:
                if math.hypot(other["x"] - d["x"], other["y"] - d["y"]) < 0.03:
                    duplicate = True
            if duplicate:
                continue
            found.append({"x": d["x"], "y": d["y"], "z": d["z_hi"], "u": box_center_u(cand),
                          "score": cand["score"], "cand": cand, "n": d["n"]})
        if len(found) >= 2:
            break
    return found


def find_cube(color, table_z):
    prompts = []
    if color is not None:
        prompts = [color + " cube", color + " block", "small " + color + " cube"]
    else:
        prompts = ["cube"]
    best = None
    for prompt in prompts:
        for cand in seg_candidates("front", prompt):
            pts = cand_points(cand, "front")
            if len(pts) < 8:
                continue
            d = describe(pts)
            ref = table_z if table_z is not None else d["z_lo"]
            height = d["z_hi"] - ref
            if height < 0.015 or height > 0.10:
                continue
            if max(d["span_x"], d["span_y"]) > 0.10:
                continue
            if best is None or cand["score"] > best["score"]:
                best = {"pts": pts, "d": d, "score": cand["score"], "prompt": prompt}
        if best is not None:
            break
    return best


def held_cube_check(color, ee_xyz, start_xy, cube_bottom):
    """'lifted' if a goal-coloured cube-sized blob is under the gripper well above its start
    height; 'on_table' if it is still resting at its start location; otherwise 'unknown'."""
    prompts = [color + " cube", color + " block"] if color is not None else ["cube"]
    verdict = "unknown"
    for prompt in prompts:
        for cand in seg_candidates("front", prompt):
            pts = cand_points(cand, "front")
            if len(pts) < 8:
                continue
            d = describe(pts)
            if max(d["span_x"], d["span_y"]) > 0.10:
                continue
            near_ee = math.hypot(d["x"] - ee_xyz[0], d["y"] - ee_xyz[1]) < 0.05
            near_start = math.hypot(d["x"] - start_xy[0], d["y"] - start_xy[1]) < 0.03
            if near_ee and d["z_lo"] > cube_bottom + 0.03:
                return "lifted"
            if near_start and d["z_hi"] < cube_bottom + 0.07 and d["z_hi"] - cube_bottom > 0.015:
                verdict = "on_table"
        if verdict != "unknown":
            break
    return verdict


def find_button(table_z, avoid_xy, near_xy=None):
    best = None
    for prompt in BUTTON_PROMPTS:
        for cand in seg_candidates("front", prompt):
            pts = cand_points(cand, "front")
            if len(pts) < 8:
                continue
            d = describe(pts)
            ref = table_z if table_z is not None else d["z_lo"]
            if d["z_hi"] - ref < 0.008:
                continue
            if max(d["span_x"], d["span_y"]) > 0.14:
                continue
            if avoid_xy is not None and math.hypot(d["x"] - avoid_xy[0], d["y"] - avoid_xy[1]) < 0.04:
                continue
            if near_xy is not None and math.hypot(d["x"] - near_xy[0], d["y"] - near_xy[1]) > 0.06:
                continue
            if best is None or cand["score"] > best["score"]:
                best = {"pts": pts, "d": d, "score": cand["score"], "prompt": prompt}
        if best is not None:
            break
    return best


def top_face(pts, z_top, depth):
    layer = [p for p in pts if p[2] > z_top - depth]
    if len(layer) < 5:
        layer = pts
    return layer


def square_fit(layer):
    """Yaw and centre of a square top face: rotate, minimise the bounding area."""
    best = None
    for k in range(30):
        a = k * (math.pi / 2.0) / 30.0
        ca = math.cos(a)
        sa = math.sin(a)
        us = [p[0] * ca + p[1] * sa for p in layer]
        vs = [-p[0] * sa + p[1] * ca for p in layer]
        u_lo = quantile(us, 0.03)
        u_hi = quantile(us, 0.97)
        v_lo = quantile(vs, 0.03)
        v_hi = quantile(vs, 0.97)
        area = (u_hi - u_lo) * (v_hi - v_lo)
        if best is None or area < best[0]:
            uc = 0.5 * (u_lo + u_hi)
            vc = 0.5 * (v_lo + v_hi)
            best = (area, a, uc * ca - vc * sa, uc * sa + vc * ca)
    return {"yaw": best[1], "x": best[2], "y": best[3]}


# ---------------------------------------------------------------- main routine


def order_pair(a, b):
    """left = larger base-frame y; cross-check against the front image column (recordings:
    the robot-left disc appears on the image-right)."""
    left, right = (a, b) if a["y"] >= b["y"] else (b, a)
    if left["u"] is not None and right["u"] is not None and left["u"] < right["u"]:
        note("side_conflict", u_left=left["u"], u_right=right["u"])
        left, right = right, left
    return {"left": left, "right": right}


def assign_sides(targets, previous):
    """Return {'left': t, 'right': t} (either may be missing)."""
    if not targets:
        return {}
    ranked = sorted(targets, key=target_score_key)[:4]
    if previous is None or "left" not in previous or "right" not in previous:
        if len(ranked) < 2:
            return {}
        best = None
        for i in range(len(ranked)):
            for j in range(i + 1, len(ranked)):
                sep = math.hypot(ranked[i]["x"] - ranked[j]["x"], ranked[i]["y"] - ranked[j]["y"])
                if sep < 0.08:
                    continue
                cost = -(ranked[i]["score"] + ranked[j]["score"])
                if best is None or cost < best[0]:
                    best = (cost, ranked[i], ranked[j])
        if best is None:
            return {}
        return order_pair(best[1], best[2])
    best = None
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            sep = math.hypot(ranked[i]["x"] - ranked[j]["x"], ranked[i]["y"] - ranked[j]["y"])
            if sep < 0.08:
                continue
            pair = order_pair(ranked[i], ranked[j])
            cost = (math.hypot(pair["left"]["x"] - previous["left"]["x"], pair["left"]["y"] - previous["left"]["y"])
                    + math.hypot(pair["right"]["x"] - previous["right"]["x"], pair["right"]["y"] - previous["right"]["y"]))
            if cost > 0.30:
                continue
            if best is None or cost < best[0]:
                best = (cost, pair)
    if best is not None:
        return best[1]
    # single detection: give it to the nearer remembered disc, if plausibly that disc
    single = ranked[0]
    dl = math.hypot(single["x"] - previous["left"]["x"], single["y"] - previous["left"]["y"])
    dr = math.hypot(single["x"] - previous["right"]["x"], single["y"] - previous["right"]["y"])
    if min(dl, dr) > 0.15:
        return {}
    return {"left": single} if dl < dr else {"right": single}


def target_score_key(t):
    return -t["score"]


def locate_side(name, memory, table_z):
    targets = find_targets(table_z)
    sides = assign_sides(targets, memory)
    if name in sides:
        memory[name] = {"x": sides[name]["x"], "y": sides[name]["y"], "z": sides[name]["z"]}
        return sides[name]
    return None


def detour(from_xy, to_xy, avoid_xy, clearance):
    """Waypoint that keeps the straight from->to transit at least `clearance` (m, xy) from
    avoid_xy, or None when the straight segment already does."""
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    length2 = dx * dx + dy * dy
    if length2 < 1e-6:
        return None
    t = ((avoid_xy[0] - from_xy[0]) * dx + (avoid_xy[1] - from_xy[1]) * dy) / length2
    t = max(0.0, min(1.0, t))
    px = from_xy[0] + t * dx
    py = from_xy[1] + t * dy
    gap = math.hypot(avoid_xy[0] - px, avoid_xy[1] - py)
    if gap >= clearance or t <= 0.0 or t >= 1.0:
        return None
    length = math.sqrt(length2)
    nx = -dy / length
    ny = dx / length
    side = 1.0 if (px - avoid_xy[0]) * nx + (py - avoid_xy[1]) * ny >= 0 else -1.0
    return (avoid_xy[0] + side * nx * (clearance + 0.02), avoid_xy[1] + side * ny * (clearance + 0.02))


def transit(from_xy, to_xy, z, quat, avoid_xy, label):
    if avoid_xy is not None:
        way = detour(from_xy, to_xy, avoid_xy, 0.07)
        if way is not None:
            note("detour", label=label, way=[way[0], way[1]])
            go((way[0], way[1], z), quat, "detour_" + label)
    go((to_xy[0], to_xy[1], z), quat, label)


def visit(name, memory, table_z, quat, grasp_offset, report_legs, avoid_xy=None):
    leg = {"side": name, "start_tick": now_ticks(), "follow": 0, "highlight": None}
    tgt = locate_side(name, memory, table_z)
    if tgt is None and name not in memory:
        leg["error"] = "target not found"
        report_legs.append(leg)
        return False
    surface = memory[name]["z"]
    transit_z = surface + 0.10 + grasp_offset
    hover_z = surface + 0.01 + grasp_offset
    aim = (memory[name]["x"], memory[name]["y"])
    here = get_robot_state().end_effector_poses["primary"].position
    transit((float(here[0]), float(here[1])), aim, transit_z, quat, avoid_xy, "above_" + name)
    # follow a disc that slides away (seen in ep2): re-localize until it is still and under us
    follow_budget = 1 if STATE["tight"] else 5
    for attempt in range(follow_budget):
        hold(3, 0.0, "settle_" + name)
        tgt = locate_side(name, memory, table_z)
        if tgt is None:
            break
        drift = math.hypot(memory[name]["x"] - aim[0], memory[name]["y"] - aim[1])
        if drift < 0.012:
            break
        leg["follow"] = leg["follow"] + 1
        aim = (memory[name]["x"], memory[name]["y"])
        go((aim[0], aim[1], transit_z), quat, "follow_" + name)
        if remaining_ticks() < 220:
            break
    go((aim[0], aim[1], hover_z), quat, "hover_" + name, tol=0.008)
    if not STATE["tight"]:
        hold(2, 0.0, "dwell_" + name)
    tgt = locate_side(name, memory, table_z)
    if tgt is not None:
        frac = red_fraction(tgt["cand"], "front")
        leg["highlight"] = frac
        moved = math.hypot(memory[name]["x"] - aim[0], memory[name]["y"] - aim[1])
        leg["final_offset"] = moved
        lit = frac is not None and frac >= 0.10
        if (not lit) and (frac is not None or moved > 0.02):
            if not STATE["tight"] and remaining_ticks() > 200:
                aim = (memory[name]["x"], memory[name]["y"])
                go((aim[0], aim[1], surface + 0.004 + grasp_offset), quat, "recenter_" + name, tol=0.008)
                hold(2, 0.0, "dwell2_" + name)
                again = locate_side(name, memory, table_z)
                if again is not None:
                    leg["highlight_after_recenter"] = red_fraction(again["cand"], "front")
    leg["end_tick"] = now_ticks()
    report_legs.append(leg)
    return True


def run():
    if PLAN["color"] is None or PLAN["first"] is None or PLAN["count"] is None:
        report["parse_warning"] = "goal not fully parsed"
    count = PLAN["count"] if PLAN["count"] is not None else 1
    first = PLAN["first"] if PLAN["first"] is not None else "right"
    second = "left" if first == "right" else "right"

    home = get_robot_state().end_effector_poses["primary"]
    quat = tuple(float(v) for v in home.quaternion_wxyz)
    if quat_tool_z(quat)[2] > -0.9:
        quat = (0.0, 1.0, 0.0, 0.0)
        report["home_not_downward"] = True

    # --- clean-view perception with the arm at its start pose (no ticks)
    targets = find_targets(None)
    memory = {}
    sides = assign_sides(targets, None)
    for name in sides:
        memory[name] = {"x": sides[name]["x"], "y": sides[name]["y"], "z": sides[name]["z"]}
    table_z = None
    if memory:
        table_z = min(memory[name]["z"] for name in memory) - 0.005
    report["targets0"] = memory
    cube = find_cube(PLAN["color"], table_z)
    if cube is None:
        raise ValueError("cube not found for colour " + str(PLAN["color"]))
    if table_z is None:
        table_z = cube["d"]["z_lo"]
    if len(memory) < 2:
        targets = find_targets(table_z)
        sides = assign_sides(targets, None)
        for name in sides:
            memory[name] = {"x": sides[name]["x"], "y": sides[name]["y"], "z": sides[name]["z"]}
    if len(memory) < 2:
        raise ValueError("fewer than two targets found: " + str(len(memory)))
    button0 = find_button(table_z, (cube["d"]["x"], cube["d"]["y"]))
    report["button0"] = None if button0 is None else button0["d"]

    # --- grasp
    cube_top = cube["d"]["z_hi"]
    face = square_fit(top_face(cube["pts"], cube_top, 0.008))
    cube_bottom = min(cube["d"]["z_lo"], table_z + 0.005)
    grasp_z = max(cube_bottom + 0.012, cube_top - 0.02)
    yaw_now = quat_yaw_x(quat)
    delta = wrap90(face["yaw"] - yaw_now)
    gquat = quat_mul((math.cos(delta / 2.0), 0.0, 0.0, math.sin(delta / 2.0)), quat)
    report["cube"] = {"x": face["x"], "y": face["y"], "top": cube_top, "bottom": cube_bottom,
                      "yaw_deg": math.degrees(face["yaw"]), "prompt": cube["prompt"]}
    start_xy = (face["x"], face["y"])
    grasped = False
    for attempt in range(2):
        check(open_gripper(), "open")
        go((start_xy[0], start_xy[1], cube_top + 0.08), gquat, "pregrasp")
        go((start_xy[0], start_xy[1], grasp_z), gquat, "grasp", tol=0.008)
        check(close_gripper(), "close")
        width = float(get_robot_state().gripper_positions["primary"])
        go((start_xy[0], start_xy[1], cube_top + 0.08), gquat, "lift")
        ee_now = get_robot_state().end_effector_poses["primary"].position
        lifted = held_cube_check(PLAN["color"], ee_now, start_xy, cube_bottom)
        report["grasp_check_" + str(attempt)] = {"width": width, "lifted": lifted}
        if width > 0.05 and lifted != "on_table":
            grasped = True
            break
        check(open_gripper(), "open_after_miss")
        again = find_cube(PLAN["color"], table_z)
        if again is None:
            break
        cube_top = again["d"]["z_hi"]
        face = square_fit(top_face(again["pts"], cube_top, 0.008))
        start_xy = (face["x"], face["y"])
        delta = wrap90(face["yaw"] - yaw_now)
        gquat = quat_mul((math.cos(delta / 2.0), 0.0, 0.0, math.sin(delta / 2.0)), quat)
    report["grasped"] = grasped
    grasp_offset = grasp_z - cube_bottom

    # --- swings
    legs = []
    report["legs"] = legs
    reserve = 160
    swings_start = now_ticks()
    for swing in range(count):
        for name in (first, second):
            done_legs = len(legs)
            if done_legs > 0:
                per_leg = float(now_ticks() - swings_start) / float(done_legs)
                legs_left = 2 * count - done_legs
                STATE["tight"] = remaining_ticks() < legs_left * per_leg + reserve
                if STATE["tight"]:
                    note("tight_budget", per_leg=per_leg, legs_left=legs_left, remaining=remaining_ticks())
            avoid = None
            if swing == 0 and name == first and second in memory:
                avoid = (memory[second]["x"], memory[second]["y"])
            visit(name, memory, table_z, gquat, grasp_offset, legs, avoid)
        report["swings_done"] = swing + 1
        if remaining_ticks() < reserve:
            report["budget_warning"] = "low ticks after swing " + str(swing + 1)

    # --- put the cube down at its start location (known free table), release, retreat
    last = memory[second]
    lift_z = last["z"] + 0.10 + grasp_offset
    here = get_robot_state().end_effector_poses["primary"].position
    go((float(here[0]), float(here[1]), lift_z), gquat, "lift_from_last")
    transit((float(here[0]), float(here[1])), start_xy, lift_z, gquat,
            (memory[first]["x"], memory[first]["y"]), "to_putdown")
    go((start_xy[0], start_xy[1], cube_bottom + grasp_offset + 0.008), gquat, "putdown", tol=0.008)
    check(open_gripper(), "release")
    go((start_xy[0], start_xy[1], lift_z), gquat, "retreat")

    # --- press the button with closed fingers
    near = None if button0 is None else (button0["d"]["x"], button0["d"]["y"])
    button = find_button(table_z, start_xy, near)
    if button is None:
        button = button0
        note("button_fallback_to_start_detection")
    if button is None:
        raise ValueError("button not found")
    bd = button["d"]
    top = bd["z_hi"]
    cap = [p for p in button["pts"] if p[2] > top - 0.006]
    if len(cap) < 3:
        cap = button["pts"]
    bx = quantile([p[0] for p in cap], 0.5)
    by = quantile([p[1] for p in cap], 0.5)
    report["button"] = {"x": bx, "y": by, "top": top, "prompt": button["prompt"]}
    check(close_gripper(), "close_for_press")
    go((bx, by, top + 0.06), quat, "above_button")
    go((bx, by, top - 0.01), quat, "press", tol=0.008, steps=80)
    go((bx, by, top + 0.04), quat, "unpress")
    go((bx, by, top - 0.025), quat, "press_deeper", tol=0.008, steps=80)
    go((bx, by, top + 0.04), quat, "unpress2")


try:
    run()
    report["finished"] = "program_end"
except Exception as exc:
    report["finished"] = "stopped"
    report["stop_reason"] = str(exc)
report["terminal_seen"] = STATE["terminal"]
report["moves"] = STATE["moves"]
report["seg_calls"] = STATE["seg_calls"]
report["final_tick"] = now_ticks()
result = report
