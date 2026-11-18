# SwingXtimes v02 -- carry in the home yaw (a 45-degree grasp yaw draped the arm over the target
# in the front view), never leave an unlit target, recover an unseen target from a clear view.
# v01: targets (lowest flat white discs) vs button (raised white object) by projected top height.
# Pick the cube named in the goal, alternate it over the two table targets in the order and
# count the goal states, put it down on the last target, then press the button.
# Every object position is perceived at runtime from the front RGB-D camera.
import math

report = {"version": "v02", "moves": [], "visits": [], "events": [], "errors": []}
ended = {"v": False}

NUMBERS = {"once": 1, "twice": 2, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
           "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
COLOR_NAMES = ("red", "green", "blue", "yellow", "purple", "orange", "cyan", "pink",
               "white", "black", "gray", "grey", "brown")
CAM = "front"


def now_s():
    try:
        return float(get_robot_state().timestamp_s)
    except Exception:
        return -1.0


def quantile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def parse_goal(text):
    low = text.lower().replace(",", " ").replace(".", " ")
    words = low.replace("-", " ").split()
    count = None
    for index, word in enumerate(words):
        if word in ("times", "time") and index > 0 and count is None:
            prev = words[index - 1]
            if prev in NUMBERS:
                count = NUMBERS[prev]
            elif prev.isdigit():
                count = int(prev)
        elif word in ("once", "twice") and count is None:
            count = NUMBERS[word]
    color = None
    for index, word in enumerate(words):
        if word in ("cube", "block") and index > 0 and words[index - 1] in COLOR_NAMES:
            color = words[index - 1]
            break
    spaced = low.replace("-", " ")
    right_at = spaced.find("right side")
    left_at = spaced.find("left side")
    if right_at < 0:
        right_at = spaced.find("right")
    if left_at < 0:
        left_at = spaced.find("left")
    first = None
    if right_at >= 0 and (left_at < 0 or right_at < left_at):
        first = "right"
    elif left_at >= 0:
        first = "left"
    return count, color, first


def hue_name(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx - mn < 50:
        if mn > 150:
            return "white"
        return "gray"
    delta = float(mx - mn)
    if mx == r:
        hue = 60.0 * (((g - b) / delta) % 6.0)
    elif mx == g:
        hue = 60.0 * (((b - r) / delta) + 2.0)
    else:
        hue = 60.0 * (((r - g) / delta) + 4.0)
    if hue < 18 or hue >= 330:
        return "red"
    if hue < 40:
        return "orange"
    if hue < 70:
        return "yellow"
    if hue < 170:
        return "green"
    if hue < 200:
        return "cyan"
    if hue < 260:
        return "blue"
    return "purple"


def box_votes(rgb, box, shrink, row_from=0.0):
    height = len(rgb)
    width = len(rgb[0])
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    xa = max(0, int(x1 + bw * shrink))
    xb = min(width - 1, int(x2 - bw * shrink))
    ya = max(0, int(y1 + bh * max(shrink, row_from)))
    yb = min(height - 1, int(y2 - bh * shrink))
    votes = {}
    total = 0
    for yy in range(ya, yb + 1):
        for xx in range(xa, xb + 1):
            px = rgb[yy][xx]
            name = hue_name(int(px[0]), int(px[1]), int(px[2]))
            votes[name] = votes.get(name, 0) + 1
            total += 1
    return votes, total


def front_rgb():
    return get_observation().cameras[CAM].rgb


def red_fraction(box):
    rgb = front_rgb()
    votes, total = box_votes(rgb, box, 0.1, 0.4)
    if total == 0:
        return 0.0
    return votes.get("red", 0) / float(total)


def cloud_of(candidate):
    try:
        cloud = mask_to_point_cloud(candidate, CAM, target_frame=report["base_frame"])
    except Exception:
        cloud = mask_to_point_cloud(candidate.mask, CAM, target_frame=report["base_frame"])
    xs = []
    ys = []
    zs = []
    for point in cloud.points:
        xs.append(float(point[0]))
        ys.append(float(point[1]))
        zs.append(float(point[2]))
    if len(xs) < 6:
        return None
    zhi = quantile(zs, 0.95)
    top_x = []
    top_y = []
    for index in range(len(zs)):
        if zs[index] >= zhi - 0.008:
            top_x.append(xs[index])
            top_y.append(ys[index])
    return {
        "n": len(xs),
        "x": quantile(xs, 0.5), "y": quantile(ys, 0.5),
        "zlo": quantile(zs, 0.1), "zhi": zhi, "zmed": quantile(zs, 0.5),
        "xspan": quantile(xs, 0.95) - quantile(xs, 0.05),
        "yspan": quantile(ys, 0.95) - quantile(ys, 0.05),
        "top_x": sum(top_x) / len(top_x), "top_y": sum(top_y) / len(top_y),
    }


def candidates(prompt, min_score, limit):
    found = segment_text(CAM, prompt)
    out = []
    if not found.ok:
        return out
    for candidate in found.segmentations[:limit]:
        score = float(candidate.score)
        if score < min_score or candidate.box_xyxy is None:
            continue
        box = [float(value) for value in candidate.box_xyxy]
        stats = cloud_of(candidate)
        if stats is None:
            continue
        stats["score"] = round(score, 3)
        stats["box"] = [round(value, 1) for value in box]
        stats["prompt"] = prompt
        out.append(stats)
    return out


def rounded(stats):
    keep = {}
    for key in stats:
        value = stats[key]
        if key in ("x", "y", "zlo", "zhi", "zmed", "xspan", "yspan", "top_x", "top_y"):
            keep[key] = round(float(value), 4)
        else:
            keep[key] = value
    return keep


def xy_dist(a, b):
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)


def find_button():
    best = None
    tried = []
    for prompt in ("button", "white push button", "gray round button on a white base"):
        for stats in candidates(prompt, 0.1, 5):
            tried.append(rounded(stats))
            if stats["xspan"] > 0.15 or stats["yspan"] > 0.15:
                continue
            if best is None or stats["score"] > best["score"]:
                best = stats
        if best is not None and best["score"] >= 0.4:
            break
    report["button_candidates"] = tried
    return best


def find_cube(color):
    prompts = []
    if color is not None:
        prompts.append(color + " cube")
        prompts.append(color + " block")
    prompts.append("cube")
    rgb = front_rgb()
    best = None
    tried = []
    for prompt in prompts:
        for stats in candidates(prompt, 0.05, 8):
            if stats["xspan"] > 0.1 or stats["yspan"] > 0.1:
                continue
            votes, total = box_votes(rgb, stats["box"], 0.2)
            chromatic = 0
            for name in votes:
                if name not in ("white", "gray"):
                    chromatic += votes[name]
            match = 0.0
            if color is not None and chromatic > 0:
                match = votes.get(color, 0) / float(chromatic)
            stats["color_match"] = round(match, 3)
            tried.append(rounded(stats))
            if color is not None and match < 0.4:
                continue
            if best is None or stats["score"] > best["score"]:
                best = stats
        if best is not None:
            break
    report["cube_candidates"] = tried
    return best


def find_targets(exclude, prompts):
    pool = []
    tried = []
    for prompt in prompts:
        for stats in candidates(prompt, 0.1, 6):
            tried.append(rounded(stats))
            if stats["xspan"] > 0.14 or stats["yspan"] > 0.14:
                continue
            if stats["xspan"] < 0.01 and stats["yspan"] < 0.01:
                continue
            if stats["zhi"] - stats["zlo"] > 0.03:
                continue
            near = False
            for other in exclude:
                if other is not None and xy_dist(stats, other) < 0.035:
                    near = True
            if near:
                continue
            merged = False
            for index in range(len(pool)):
                if xy_dist(pool[index], stats) < 0.03:
                    merged = True
                    if stats["score"] > pool[index]["score"]:
                        pool[index] = stats
            if not merged:
                pool.append(stats)
    report["events"].append({"find_targets": len(tried), "pool": len(pool)})
    report["target_candidates_last"] = tried
    return sorted(pool, key=lambda item: -item["score"])


def chroma_fraction(rgb, box):
    votes, total = box_votes(rgb, box, 0.2)
    chromatic = 0
    for name in votes:
        if name not in ("white", "gray"):
            chromatic += votes[name]
    if total == 0:
        return 1.0
    return chromatic / float(total)


def find_scene(cube):
    """Flat white discs = targets (lowest tops); white object with a raised top = button."""
    rgb = front_rgb()
    pool = []
    tried = []
    prompts = ("target", "white round target", "bullseye target", "button", "white push button")
    for prompt in prompts:
        is_button_prompt = prompt.find("button") >= 0
        for stats in candidates(prompt, 0.1, 6):
            stats["chroma"] = round(chroma_fraction(rgb, stats["box"]), 3)
            tried.append(rounded(stats))
            if stats["xspan"] > 0.14 or stats["yspan"] > 0.14:
                continue
            if stats["xspan"] < 0.01 and stats["yspan"] < 0.01:
                continue
            if xy_dist(stats, cube) < 0.035:
                continue
            if stats["chroma"] > 0.35:
                continue
            stats["tscore"] = 0.0 if is_button_prompt else stats["score"]
            stats["bscore"] = stats["score"] if is_button_prompt else 0.0
            merged = False
            for index in range(len(pool)):
                if xy_dist(pool[index], stats) < 0.03:
                    merged = True
                    old = pool[index]
                    tscore = max(old["tscore"], stats["tscore"])
                    bscore = max(old["bscore"], stats["bscore"])
                    if stats["score"] > old["score"]:
                        pool[index] = stats
                    pool[index]["tscore"] = tscore
                    pool[index]["bscore"] = bscore
            if not merged:
                pool.append(stats)
    report["scene_candidates"] = tried
    report["scene_pool"] = [rounded(item) for item in pool]
    if len(pool) == 0:
        return [], None
    min_top = min(item["zhi"] for item in pool)
    flat = [item for item in pool if item["zhi"] <= min_top + 0.01]
    raised = [item for item in pool if min_top + 0.012 < item["zhi"] < min_top + 0.08]
    flat = sorted(flat, key=lambda item: -(item["tscore"] + 0.01 * item["score"]))
    raised = sorted(raised, key=lambda item: -(item["bscore"] + 0.01 * item["score"]))
    button = None
    if len(raised) > 0:
        button = raised[0]
    return flat, button


def relocalize(expected, other, button, top_limit, radius):
    """Fresh front-camera estimate of one target near its last known position (None if unseen)."""
    best = None
    for stats in candidates("target", 0.08, 6):
        if stats["zhi"] > top_limit:
            continue
        if stats["xspan"] > 0.14 or stats["yspan"] > 0.14:
            continue
        if button is not None and xy_dist(stats, button) < 0.035:
            continue
        if other is not None and xy_dist(stats, other) < 0.04:
            continue
        if xy_dist(stats, expected) > radius:
            continue
        if best is None or xy_dist(stats, expected) < xy_dist(best, expected):
            best = stats
    return best


def go(position, quat, tolerance, max_steps, label):
    if ended["v"]:
        raise Exception("episode already ended")
    t0 = now_s()
    reply = move_to_pose(Pose(list(position), list(quat), report["base_frame"]),
                         tolerance=tolerance, max_steps=max_steps)
    record = {"label": label, "ok": bool(reply.ok), "target": [round(v, 4) for v in position]}
    try:
        record["steps"] = int(reply.steps_executed)
    except Exception:
        record["steps"] = None
    try:
        record["status"] = str(reply.status)
    except Exception:
        record["status"] = None
    try:
        if bool(reply.terminated) or bool(reply.truncated):
            ended["v"] = True
            record["terminal"] = [bool(reply.terminated), bool(reply.truncated)]
    except Exception:
        record["terminal"] = "unknown"
    if not ended["v"]:
        record["t"] = round(now_s(), 2)
        try:
            pos = get_robot_state().end_effector_poses["primary"].position
            record["reached"] = [round(float(v), 4) for v in pos]
        except Exception:
            record["reached"] = None
    record["t0"] = round(t0, 2)
    report["moves"].append(record)
    return reply


def gripper(action, label):
    if ended["v"]:
        raise Exception("episode already ended")
    if action == "open":
        reply = open_gripper()
    else:
        reply = close_gripper()
    record = {"label": label, "ok": bool(reply.ok)}
    try:
        record["steps"] = int(reply.steps_executed)
    except Exception:
        record["steps"] = None
    try:
        if bool(reply.terminated) or bool(reply.truncated):
            ended["v"] = True
            record["terminal"] = [bool(reply.terminated), bool(reply.truncated)]
    except Exception:
        record["terminal"] = "unknown"
    if not ended["v"]:
        try:
            record["width"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
        except Exception:
            record["width"] = None
    report["moves"].append(record)
    return reply


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def quat_axes(q):
    w, x, y, z = q
    x_axis = [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)]
    z_axis = [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]
    return x_axis, z_axis


def footprint_yaw(stats_points):
    # min-area rectangle over 0..85 deg on the cube's xy points
    best = None
    for step_deg in range(0, 90, 5):
        angle = math.radians(step_deg)
        ca = math.cos(angle)
        sa = math.sin(angle)
        us = [ca * p[0] + sa * p[1] for p in stats_points]
        vs = [-sa * p[0] + ca * p[1] for p in stats_points]
        area = (quantile(us, 0.95) - quantile(us, 0.05)) * (quantile(vs, 0.95) - quantile(vs, 0.05))
        if best is None or area < best[0]:
            best = (area, angle)
    if best is None:
        return None
    return best[1]


def main():
    context = get_task_context()
    state = get_robot_state()
    report["goal"] = context.language
    report["base_frame"] = state.base_frame
    report["embodiment"] = state.embodiment
    try:
        meta = robomme_plus.get_task_metadata()
        keep = {}
        for key in meta:
            keep[str(key)] = str(meta[key])[:300]
        report["task_metadata"] = keep
    except Exception as exc:
        report["errors"].append("metadata: " + str(exc))
    try:
        report["video_len"] = len(robomme_plus.get_conditioning_video())
    except Exception as exc:
        report["errors"].append("video: " + str(exc))
    count, color, first = parse_goal(context.language)
    report["parsed"] = {"count": count, "color": color, "first": first}
    if count is None or first is None:
        raise ValueError("could not parse goal")

    home_pose = state.end_effector_poses["primary"]
    home_q = [float(v) for v in home_pose.quaternion_wxyz]
    x_axis, z_axis = quat_axes(home_q)
    report["home"] = {"pos": [round(float(v), 4) for v in home_pose.position],
                      "quat": [round(v, 4) for v in home_q], "z_axis": [round(v, 3) for v in z_axis]}
    down_q = home_q
    if z_axis[2] > -0.9:
        down_q = [0.0, 1.0, 0.0, 0.0]
        x_axis, z_axis = quat_axes(down_q)
    home_yaw = math.atan2(x_axis[1], x_axis[0])

    # ---------- perception of the untouched scene ----------
    cube = find_cube(color)
    if cube is None:
        raise ValueError("cube not found")
    report["cube"] = rounded(cube)
    targets, button = find_scene(cube)
    report["targets_initial"] = [rounded(t) for t in targets[:4]]
    report["button"] = None if button is None else rounded(button)
    if len(targets) < 2:
        raise ValueError("fewer than two targets found")
    pair = [targets[0], None]
    for other in targets[1:]:
        if xy_dist(other, targets[0]) > 0.06:
            pair[1] = other
            break
    if pair[1] is None:
        raise ValueError("two separated targets not found")
    top_limit = max(pair[0]["zhi"], pair[1]["zhi"]) + 0.012
    # Demonstrations: the front camera faces the robot, so the robot's right-side target is
    # the one further LEFT in the front image.
    def box_cx(t):
        return 0.5 * (t["box"][0] + t["box"][2])
    if box_cx(pair[0]) < box_cx(pair[1]):
        side_target = {"right": pair[0], "left": pair[1]}
    else:
        side_target = {"right": pair[1], "left": pair[0]}
    report["side_check"] = {"right_y": round(side_target["right"]["y"], 4),
                            "left_y": round(side_target["left"]["y"], 4),
                            "y_agrees": side_target["right"]["y"] < side_target["left"]["y"]}
    report["base_red"] = {}
    for side in ("right", "left"):
        report["base_red"][side] = round(red_fraction(side_target[side]["box"]), 3)

    # ---------- geometry ----------
    table_z = min(cube["zlo"], pair[0]["zlo"], pair[1]["zlo"])
    cube_size = max(0.015, min(0.07, cube["zhi"] - table_z))
    grasp_z = cube["zhi"] - 0.5 * cube_size
    bottom_offset = grasp_z - table_z
    transit_z = table_z + bottom_offset + 0.07
    report["geometry"] = {"cube_size": round(cube_size, 4), "table_z": round(table_z, 4),
                          "grasp_z": round(grasp_z, 4), "transit_z": round(transit_z, 4)}

    # cube yaw from its points (front camera)
    grasp_q = down_q
    try:
        found = segment_text(CAM, cube["prompt"])
        pts = []
        for candidate in found.segmentations[:8]:
            box = [float(value) for value in candidate.box_xyxy]
            if abs(box[0] - cube["box"][0]) < 3 and abs(box[1] - cube["box"][1]) < 3:
                cloud = mask_to_point_cloud(candidate, CAM, target_frame=report["base_frame"])
                for point in cloud.points:
                    if float(point[2]) > cube["zhi"] - 0.006:
                        pts.append((float(point[0]), float(point[1])))
                break
        if len(pts) >= 8:
            cube_yaw = footprint_yaw(pts)
            delta = ((cube_yaw - home_yaw + math.pi / 4) % (math.pi / 2)) - math.pi / 4
            rot = [math.cos(delta / 2), 0.0, 0.0, math.sin(delta / 2)]
            grasp_q = quat_mul(rot, down_q)
            report["yaw"] = {"cube_deg": round(math.degrees(cube_yaw), 1),
                             "delta_deg": round(math.degrees(delta), 1), "n": len(pts)}
    except Exception as exc:
        report["errors"].append("yaw: " + str(exc))

    # ---------- grasp ----------
    gripper("open", "open")
    go((cube["x"], cube["y"], grasp_z + 0.08), grasp_q, 0.015, 200, "pregrasp")
    go((cube["x"], cube["y"], grasp_z), grasp_q, 0.008, 150, "grasp")
    gripper("close", "close")
    go((cube["x"], cube["y"], transit_z), grasp_q, 0.015, 150, "lift")
    width = float(get_robot_state().gripper_positions["primary"])
    report["held_width"] = round(width, 3)

    # ---------- swings ----------
    order = ["right", "left"] if first == "right" else ["left", "right"]
    sequence = []
    for swing in range(count):
        sequence.append(order[0])
        sequence.append(order[1])
    report["sequence"] = sequence
    last = None
    red_needed = 0.2
    carry_q = down_q
    clear_view = [float(v) for v in home_pose.position]
    for index, side in enumerate(sequence):
        entry = {"i": index, "side": side, "t_start": round(now_s(), 2), "attempts": []}
        target = side_target[side]
        other = side_target["left" if side == "right" else "right"]
        lit = False
        for attempt in range(6):
            step_log = {}
            fresh = relocalize(target, other, button, top_limit, 0.15)
            if fresh is None and attempt > 0:
                # the arm may hide the target from the front camera: look from the home pose
                go(clear_view, carry_q, 0.02, 150, "clear_view")
                fresh = relocalize(target, other, button, top_limit, 0.3)
                step_log["clear_view"] = fresh is not None
            if fresh is not None:
                step_log["shift_pre"] = round(xy_dist(fresh, target), 4)
                target = fresh
            go((target["x"], target["y"], transit_z), carry_q, 0.015, 150, "hover_" + side)
            fresh = relocalize(target, other, button, top_limit, 0.15)
            if fresh is not None:
                step_log["shift_hover"] = round(xy_dist(fresh, target), 4)
                moved = xy_dist(fresh, target) > 0.012
                target = fresh
                if moved:
                    go((target["x"], target["y"], transit_z), carry_q, 0.015, 120, "rehover_" + side)
            else:
                step_log["hover_unseen"] = True
            red = red_fraction(target["box"])
            step_log["red_hover"] = round(red, 3)
            if red >= red_needed:
                lit = True
            entry["attempts"].append(step_log)
            if lit:
                break
        entry["lit"] = lit
        entry["t_end"] = round(now_s(), 2)
        entry["xy"] = [round(target["x"], 4), round(target["y"], 4)]
        report["visits"].append(entry)
        side_target[side] = target
        last = target
        if not lit:
            # moving on to the other target after a missed visit breaks the alternation
            raise ValueError("target %s not activated at visit %d" % (side, index))

    # ---------- put down on the last target, then press the button ----------
    place_z = last["zhi"] + bottom_offset + 0.012
    go((last["x"], last["y"], place_z), carry_q, 0.01, 120, "place")
    gripper("open", "release")
    go((last["x"], last["y"], transit_z + 0.02), carry_q, 0.015, 120, "retreat")
    gripper("close", "close_for_press")
    if button is None:
        raise ValueError("button not found")
    top_z = button["zhi"]
    bx = button["top_x"]
    by = button["top_y"]
    go((bx, by, top_z + 0.05), down_q, 0.012, 150, "button_hover")
    go((bx, by, top_z - 0.015), down_q, 0.01, 100, "button_press")
    if not ended["v"]:
        go((bx, by, top_z + 0.05), down_q, 0.015, 100, "button_release")
    if not ended["v"]:
        go((bx, by, top_z - 0.03), down_q, 0.01, 100, "button_press2")
    if not ended["v"]:
        go((bx, by, top_z + 0.05), down_q, 0.015, 100, "button_release2")


try:
    main()
    report["completed"] = True
except Exception as exc:
    report["completed"] = False
    report["errors"].append("main: " + str(exc))
report["ended"] = ended["v"]
try:
    report["t_final"] = round(now_s(), 2)
except Exception:
    report["t_final"] = None
result = report
