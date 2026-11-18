# PickXtimes (robomme_plus task 7) -- v04: grasp and release heights both referenced to the support
# surface (target top = table height measured once) and the cube height measured once; per-cycle
# re-observed cube tops no longer move the grasp. (v03's offset formula reduced algebraically to
# v02's, so v03 was a no-op; found by the offline mock 'biased_top_later'.)
# v02 note: colour checks on the central half of each SAM box
# (a loose box around a distractor cube on a grey table passed v01's whole-box test; AfD).
# v01 changes: strict red rule (dark wood matched the v00 red rule), button candidates rejected
# when their box is dominated by cube/target colours, point-cloud fallback on empty clouds,
# grasp height clamped inside the cube, vertical rise before traversing to the button.
# Goal pattern (parsed at runtime): "pick up the <colour> cube and place it on the target,
# repeating this pick-and-place action <N> times, then press the button to stop."
# Strategy: localise cube / target / button once from the front camera at home, then
# N x [descend+grasp, lift, lower onto target, release, rise], counting ONLY verified releases,
# then close the gripper and push the button cap down from above.
import math

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}
COLOUR_WORDS = ("red", "green", "blue", "yellow", "purple", "orange", "pink", "white", "black")

# Generic relative offsets / tolerances (metres) -- not scene coordinates.
HOVER_ABOVE_TOP = 0.10      # approach hover above an object's top face
LIFT_HEIGHT = 0.10          # how far the cube is lifted each repetition
CLEAR_HEIGHT = 0.07         # gripper rise after each release (wrist re-observation pose)
RELEASE_DROP = 0.008        # release this far above the resting grasp height
PRESS_DEPTHS = (0.012, 0.028)  # successive press targets below the button cap top
MAX_GRASP_FAILURES = 4

report = {"events": [], "placements": 0, "terminal": False}
context = get_task_context()
goal = str(context.language)
report["goal"] = goal
state0 = get_robot_state()
base_frame = state0.base_frame
t_start = float(state0.timestamp_s)

control_hz = 20.0
max_steps = None
try:
    meta = robomme_plus.get_task_metadata()
    try:
        max_steps = int(meta["max_steps"])
    except Exception:
        max_steps = None
    try:
        control_hz = float(meta["control_hz"])
    except Exception:
        control_hz = 20.0
except Exception as exc:
    report["events"].append("metadata unavailable: " + str(exc))
report["max_steps"] = max_steps


def log(message):
    report["events"].append(message)


def ticks_used():
    try:
        return int(round((float(get_robot_state().timestamp_s) - t_start) * control_hz))
    except Exception:
        return -1


# ---------------------------------------------------------------- goal parsing
def parse_count(text):
    words = text.lower().replace(",", " ").replace(".", " ").replace("-", " ").split()
    for index, word in enumerate(words):
        if word == "once":
            return 1
        if word == "twice":
            return 2
        if word in ("times", "time") and index > 0:
            previous = words[index - 1]
            if previous in NUMBER_WORDS:
                return NUMBER_WORDS[previous]
            if previous.isdigit():
                return int(previous)
    return None


def parse_object(text):
    lowered = text.lower()
    phrase = None
    key = "pick up the "
    if key in lowered:
        rest = lowered.split(key, 1)[1]
        phrase = rest.split(" and ", 1)[0].strip()
    if phrase is None or len(phrase) == 0:
        phrase = "cube"
    colour = None
    for word in phrase.split():
        if word in COLOUR_WORDS:
            colour = word
    return phrase, colour


repeat_count = parse_count(goal)
object_phrase, object_colour = parse_object(goal)
report["repeat_count"] = repeat_count
report["object_phrase"] = object_phrase
report["object_colour"] = object_colour
if repeat_count is None or repeat_count < 1:
    raise ValueError("could not parse the repetition count from the goal: " + goal)


# ---------------------------------------------------------------- small math helpers
def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def wrap_quarter(angle):
    # wrap into [-pi/4, pi/4): a cube face direction is defined modulo 90 degrees
    while angle >= math.pi / 4.0:
        angle -= math.pi / 2.0
    while angle < -math.pi / 4.0:
        angle += math.pi / 2.0
    return angle


def quat_to_matrix(q):
    w, x, y, z = (float(v) for v in q)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def matrix_to_quat(m):
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    return [w / norm, x / norm, y / norm, z / norm]


def top_down_quat(closing_angle):
    # hand z axis points down, hand y axis (finger travel) along closing_angle in the table plane
    c = math.cos(closing_angle)
    s = math.sin(closing_angle)
    m = [[-s, c, 0.0], [c, s, 0.0], [0.0, 0.0, -1.0]]
    return matrix_to_quat(m)


home_rot = quat_to_matrix(state0.end_effector_poses["primary"].quaternion_wxyz)
home_closing_angle = math.atan2(home_rot[1][1], home_rot[0][1])


def min_area_angle(points_xy):
    # angle (mod 90 deg) of the minimum-area bounding rectangle of the top-face points
    if len(points_xy) < 6:
        return None
    best_angle = None
    best_area = None
    for step_index in range(45):
        angle = step_index * (math.pi / 2.0) / 45.0
        c = math.cos(angle)
        s = math.sin(angle)
        us = [p[0] * c + p[1] * s for p in points_xy]
        vs = [-p[0] * s + p[1] * c for p in points_xy]
        width_u = quantile(us, 0.97) - quantile(us, 0.03)
        width_v = quantile(vs, 0.97) - quantile(vs, 0.03)
        area = width_u * width_v
        if best_area is None or area < best_area:
            best_area = area
            best_angle = angle
    return best_angle


# ---------------------------------------------------------------- perception helpers
def pixel_matches(r, g, b, colour):
    if colour == "green":
        return g > 110 and g > r + 50 and g > b + 50
    if colour == "red":
        return r > 140 and g < 50 and b < 50
    if colour == "blue":
        return b > 100 and b > r + 50 and b > g + 20
    if colour == "yellow":
        return r > 150 and g > 150 and b < 90 and abs(r - g) < 60
    if colour == "purple":
        return r > 120 and b > 120 and g < r - 40 and g < b - 40
    if colour == "orange":
        return r > 180 and 70 < g < 170 and b < 70
    if colour == "pink":
        return r > 180 and b > 120 and g < r - 50
    if colour == "white":
        return r > 200 and g > 200 and b > 200
    if colour == "black":
        return r < 50 and g < 50 and b < 50
    return True


def box_fraction(rgb, box, colour, grey=False):
    # colour statistics over the CENTRAL half of the box: an object fills the middle of its box,
    # while a loose box is mostly background near its border
    bx1, by1, bx2, by2 = (float(v) for v in box)
    qx = 0.25 * (bx2 - bx1)
    qy = 0.25 * (by2 - by1)
    x1 = bx1 + qx
    x2 = bx2 - qx
    y1 = by1 + qy
    y2 = by2 - qy
    height = len(rgb)
    width = len(rgb[0])
    xa = max(0, int(x1))
    xb = min(width - 1, int(x2))
    ya = max(0, int(y1))
    yb = min(height - 1, int(y2))
    stride = max(1, int(max(xb - xa, yb - ya) / 24))
    hits = 0
    total = 0
    for y in range(ya, yb + 1, stride):
        row = rgb[y]
        for x in range(xa, xb + 1, stride):
            px = row[x]
            r = int(px[0])
            g = int(px[1])
            b = int(px[2])
            total += 1
            if grey:
                if max(r, g, b) - min(r, g, b) < 40 and max(r, g, b) > 90:
                    hits += 1
            elif pixel_matches(r, g, b, colour):
                hits += 1
    if total == 0:
        return 0.0
    return hits / float(total)


def object_colour_fraction(rgb, box):
    # largest fraction of any cube/target colour inside the box (the button itself is grey/white)
    best = 0.0
    for colour in ("green", "red", "blue", "yellow", "purple"):
        best = max(best, box_fraction(rgb, box, colour))
    return best


def box_center(box):
    x1, y1, x2, y2 = (float(v) for v in box)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def cloud_points(candidate, camera_name):
    cloud = None
    try:
        cloud = mask_to_point_cloud(candidate, camera_name, target_frame=base_frame)
    except Exception:
        cloud = None
    points = []
    if cloud is not None:
        try:
            for point in cloud.points:
                points.append((float(point[0]), float(point[1]), float(point[2])))
        except Exception:
            points = []
    if len(points) == 0:
        try:
            cloud = mask_to_point_cloud(candidate.mask, camera_name, target_frame=base_frame)
            for point in cloud.points:
                points.append((float(point[0]), float(point[1]), float(point[2])))
        except Exception:
            points = []
    return points


def find_candidates(camera_name, prompts, colour=None, min_fraction=0.0, grey=False,
                    max_per_prompt=6):
    """Return accepted candidates [(score, box, candidate, prompt)] best first, colour-checked."""
    observation = get_observation()
    rgb = observation.cameras[camera_name].rgb
    accepted = []
    for prompt in prompts:
        found = segment_text(camera_name, prompt)
        if not found.ok:
            continue
        for candidate in found.segmentations[:max_per_prompt]:
            if candidate.box_xyxy is None:
                continue
            box = [float(v) for v in candidate.box_xyxy]
            fraction = 1.0
            if colour is not None or grey:
                fraction = box_fraction(rgb, box, colour, grey)
            if fraction < min_fraction:
                continue
            accepted.append((float(candidate.score), box, candidate, prompt, fraction))
        if len(accepted) > 0:
            break
    ordered = []
    while len(accepted) > 0:
        best_index = 0
        for index in range(len(accepted)):
            if accepted[index][0] > accepted[best_index][0]:
                best_index = index
        ordered.append(accepted.pop(best_index))
    return ordered


def cube_prompts():
    prompts = [object_phrase]
    if object_colour is not None:
        prompts.append(object_colour + " block")
        prompts.append(object_colour + " box")
    prompts.append("cube")
    return prompts


def top_face(points, table_z, near_xy=None, gate=None):
    """Top-face centre / height / yaw of a small box-like object from its projected points."""
    pts = []
    for p in points:
        if p[2] < table_z - 0.02 or p[2] > table_z + 0.15:
            continue
        if near_xy is not None and gate is not None:
            if math.hypot(p[0] - near_xy[0], p[1] - near_xy[1]) > gate:
                continue
        pts.append(p)
    if len(pts) < 8:
        return None
    top = quantile([p[2] for p in pts], 0.9)
    face = [p for p in pts if p[2] > top - 0.008]
    if len(face) < 4:
        return None
    cx = quantile([p[0] for p in face], 0.5)
    cy = quantile([p[1] for p in face], 0.5)
    face = [p for p in face if math.hypot(p[0] - cx, p[1] - cy) < 0.05]
    if len(face) < 4:
        return None
    cx = quantile([p[0] for p in face], 0.5)
    cy = quantile([p[1] for p in face], 0.5)
    yaw = min_area_angle([(p[0] - cx, p[1] - cy) for p in face])
    return {"x": cx, "y": cy, "top": top, "yaw": yaw, "n": len(face)}


def locate_cube(camera_name, table_z, near_xy=None, gate=None):
    candidates = find_candidates(camera_name, cube_prompts(), object_colour, 0.2)
    for entry in candidates:
        face = top_face(cloud_points(entry[2], camera_name), table_z, near_xy, gate)
        if face is not None:
            face["camera"] = camera_name
            face["score"] = round(entry[0], 3)
            face["prompt"] = entry[3]
            return face
    return None


# ---------------------------------------------------------------- motion helpers
def result_terminal(res):
    if res is None:
        return False
    for attempt in range(3):
        try:
            if attempt == 0 and bool(res.terminated):
                return True
            if attempt == 1 and bool(res.truncated):
                return True
            if attempt == 2 and "terminat" in str(res.status).lower():
                return True
        except Exception:
            pass
    return False


def check_terminal(res, label):
    if result_terminal(res):
        report["terminal"] = True
        log("terminal after " + label)
        return True
    return False


def move_ee(xyz, quat, label, tolerance=0.01, steps=150):
    target = Pose((float(xyz[0]), float(xyz[1]), float(xyz[2])), quat, base_frame)
    res = move_to_pose(target, tolerance=tolerance, max_steps=steps)
    status = "?"
    try:
        status = str(res.status)
    except Exception:
        status = "?"
    log(label + " ok=" + str(bool(res.ok)) + " status=" + status + " t=" + str(ticks_used()))
    check_terminal(res, label)
    return res


def gripper_reading():
    return float(get_robot_state().gripper_positions["primary"])


def ee_position():
    p = get_robot_state().end_effector_poses["primary"].position
    return (float(p[0]), float(p[1]), float(p[2]))


# ---------------------------------------------------------------- initial perception (arm at home)
target_entries = find_candidates("front", ("purple target", "bullseye target", "target",
                                           "purple circle"), "purple", 0.08)
target_info = None
for entry in target_entries:
    pts = cloud_points(entry[2], "front")
    if len(pts) < 5:
        continue
    zs = [p[2] for p in pts]
    tz = quantile(zs, 0.5)
    flat = [p for p in pts if abs(p[2] - tz) < 0.02]
    if len(flat) < 5:
        continue
    tx = quantile([p[0] for p in flat], 0.5)
    ty = quantile([p[1] for p in flat], 0.5)
    spread = quantile([math.hypot(p[0] - tx, p[1] - ty) for p in flat], 0.9)
    target_info = {"x": tx, "y": ty, "z": tz, "radius": spread, "box": entry[1],
                   "score": round(entry[0], 3), "prompt": entry[3]}
    break
if target_info is None:
    raise ValueError("target not localised")
report["target"] = target_info
table_z = target_info["z"]
target_radius = max(0.015, min(0.06, target_info["radius"]))

cube = locate_cube("front", table_z)
if cube is None:
    raise ValueError("cube not localised")
report["cube_initial"] = cube

button_entries = find_candidates("front", ("button", "grey button", "push button",
                                           "round button"), None, 0.3, True)
button_info = None
target_px = box_center(target_info["box"])
target_px_size = max(target_info["box"][2] - target_info["box"][0],
                     target_info["box"][3] - target_info["box"][1])
front_rgb = get_observation().cameras["front"].rgb
for entry in button_entries:
    if object_colour_fraction(front_rgb, entry[1]) > 0.2:
        continue
    centre = box_center(entry[1])
    if math.hypot(centre[0] - target_px[0], centre[1] - target_px[1]) < 0.75 * target_px_size:
        continue
    pts = [p for p in cloud_points(entry[2], "front")
           if table_z - 0.01 < p[2] < table_z + 0.10]
    if len(pts) < 5:
        continue
    btop = quantile([p[2] for p in pts], 0.92)
    cap = [p for p in pts if p[2] > btop - 0.006]
    bx = quantile([p[0] for p in cap], 0.5)
    by = quantile([p[1] for p in cap], 0.5)
    if math.hypot(bx - cube["x"], by - cube["y"]) < 0.03:
        continue
    button_info = {"x": bx, "y": by, "top": btop, "score": round(entry[0], 3),
                   "prompt": entry[3]}
    break
report["button"] = button_info

cube_height = max(0.015, min(0.08, cube["top"] - table_z))
report["cube_height_front"] = cube_height


def closing_angle_for(yaw):
    if yaw is None:
        return home_closing_angle
    return home_closing_angle + wrap_quarter(yaw - home_closing_angle)


# ---------------------------------------------------------------- refine cube from above (wrist)
open_gripper()
grasp_quat = top_down_quat(closing_angle_for(cube["yaw"]))
hover_z = cube["top"] + HOVER_ABOVE_TOP
move_ee((cube["x"], cube["y"], hover_z), grasp_quat, "hover_cube", 0.015, 200)
refined = locate_cube("wrist", table_z, (cube["x"], cube["y"]), 0.05)
if refined is not None:
    report["cube_refined"] = refined
    cube = refined
    cube_height = max(0.015, min(0.08, cube["top"] - table_z))
    grasp_quat = top_down_quat(closing_angle_for(cube["yaw"]))
report["cube_height"] = cube_height
open_reading = gripper_reading()
report["open_reading"] = open_reading

# ---------------------------------------------------------------- repetitions
placements = 0
grasp_failures = 0
current = {"x": cube["x"], "y": cube["y"], "top": cube["top"]}
lift_z = table_z + cube_height + LIFT_HEIGHT + 0.5 * cube_height
while placements < repeat_count and not report["terminal"]:
    # the cube always rests on the table or on the (thin) target, so its mid-height is
    # table_z + 0.5 * cube_height; a noisy per-cycle top estimate is used only for x/y
    grasp_offset = max(0.01, min(cube_height - 0.008, 0.5 * cube_height))
    grasp_z = table_z + grasp_offset
    move_ee((current["x"], current["y"], grasp_z), grasp_quat, "descend_grasp", 0.01, 120)
    if report["terminal"]:
        break
    close_gripper()
    held = gripper_reading()
    if held < 0.12 * max(open_reading, 0.1):
        grasp_failures += 1
        log("grasp failed reading=" + str(round(held, 3)))
        open_gripper()
        move_ee((current["x"], current["y"], grasp_z + CLEAR_HEIGHT), grasp_quat,
                "retreat_after_miss", 0.015, 120)
        if grasp_failures > MAX_GRASP_FAILURES or report["terminal"]:
            break
        again = locate_cube("wrist", table_z, (current["x"], current["y"]), 0.08)
        if again is None:
            again = locate_cube("front", table_z, (current["x"], current["y"]), 0.10)
        if again is not None:
            current = {"x": again["x"], "y": again["y"], "top": again["top"]}
            grasp_quat = top_down_quat(closing_angle_for(again["yaw"]))
        continue
    lift_res = move_ee((current["x"], current["y"], lift_z), grasp_quat, "lift", 0.015, 150)
    if report["terminal"]:
        break
    move_ee((target_info["x"], target_info["y"], lift_z), grasp_quat, "over_target", 0.015, 200)
    if report["terminal"]:
        break
    if gripper_reading() < 0.12 * max(open_reading, 0.1):
        grasp_failures += 1
        log("cube lost during transport")
        found = locate_cube("front", table_z)
        if found is None or grasp_failures > MAX_GRASP_FAILURES:
            break
        open_gripper()
        current = {"x": found["x"], "y": found["y"], "top": found["top"]}
        grasp_quat = top_down_quat(closing_angle_for(found["yaw"]))
        move_ee((current["x"], current["y"], current["top"] + HOVER_ABOVE_TOP), grasp_quat,
                "hover_recover", 0.015, 200)
        continue
    place_z = table_z + grasp_offset + RELEASE_DROP
    move_ee((target_info["x"], target_info["y"], place_z), grasp_quat, "lower_place", 0.01, 150)
    if report["terminal"]:
        break
    open_res = open_gripper()
    check_terminal(open_res, "release")
    clear_z = place_z + CLEAR_HEIGHT
    move_ee((target_info["x"], target_info["y"], clear_z), grasp_quat, "clear", 0.015, 120)
    seen = locate_cube("wrist", table_z, (target_info["x"], target_info["y"]), 0.08)
    on_target = True
    if seen is not None:
        offset = math.hypot(seen["x"] - target_info["x"], seen["y"] - target_info["y"])
        if offset > max(0.04, 1.5 * target_radius):
            on_target = False
        current = {"x": seen["x"], "y": seen["y"], "top": seen["top"]}
    else:
        current = {"x": target_info["x"], "y": target_info["y"],
                   "top": table_z + cube_height}
    if on_target:
        placements += 1
        log("placement " + str(placements) + " of " + str(repeat_count) + " t=" + str(ticks_used()))
    else:
        log("release landed off target; not counted")
    report["placements"] = placements
    if report["terminal"]:
        break

# ---------------------------------------------------------------- press the button
report["placements"] = placements
if placements == repeat_count and not report["terminal"]:
    if button_info is None:
        button_info = None
        found_buttons = find_candidates("front", ("button", "grey button"), None, 0.3, True)
        retry_rgb = get_observation().cameras["front"].rgb
        for entry in found_buttons:
            if object_colour_fraction(retry_rgb, entry[1]) > 0.2:
                continue
            pts = [p for p in cloud_points(entry[2], "front")
                   if table_z - 0.01 < p[2] < table_z + 0.10]
            if len(pts) < 5:
                continue
            btop = quantile([p[2] for p in pts], 0.92)
            cap = [p for p in pts if p[2] > btop - 0.006]
            bx = quantile([p[0] for p in cap], 0.5)
            by = quantile([p[1] for p in cap], 0.5)
            if math.hypot(bx - target_info["x"], by - target_info["y"]) < 0.04:
                continue
            button_info = {"x": bx, "y": by, "top": btop, "score": round(entry[0], 3)}
            break
        report["button_retry"] = button_info
    if button_info is None:
        log("button not localised; cannot press")
    else:
        here = ee_position()
        rise_z = max(here[2], button_info["top"] + 0.08)
        move_ee((here[0], here[1], rise_z), grasp_quat, "rise_before_button", 0.015, 120)
        close_res = close_gripper()
        check_terminal(close_res, "close_for_press")
        press_quat = grasp_quat
        above = (button_info["x"], button_info["y"], button_info["top"] + 0.08)
        move_ee(above, press_quat, "above_button", 0.015, 200)
        for depth in PRESS_DEPTHS:
            if report["terminal"]:
                break
            move_ee((button_info["x"], button_info["y"], button_info["top"] - depth),
                    press_quat, "press_" + str(depth), 0.005, 60)
        if not report["terminal"]:
            move_ee(above, press_quat, "release_button", 0.015, 100)

report["ticks_used"] = ticks_used()
report["grasp_failures"] = grasp_failures
result = report
