# v01_target_filter_button_press.py -- robomme_plus / task_10 SwingXtimes, interaction-only.
# Trace-guided repair of v00 (v00: 0/1 on episode 0; this version is UNTESTED, no start left).
# Repairs (evidence: attempt_001_b057c41b program-result.json probes + front.mp4 frames):
#   R1 target identity: v00 accepted the button cap (zext 0.0177, z50-table 0.0324) as the
#      "left" target; targets now must be flat discs (zext <= 0.008, z50-table <= 0.02), must not
#      coincide with a button candidate, and only the two best-scoring discs are kept.
#   R2 missing clause: "press the button to stop" was never executed; a "button" probe and a
#      closed-gripper top-down press after the put-down are added.
# All other phases (grasp, visit motion and heights, put-down) are unchanged from v00.
#
# This program is the scene snapshot AND the attempt. Phases:
#   A  snapshot: language, task/controller metadata, cameras, robot state
#   B  conditioning-video summary (robot-state path, gripper events, lateral turning points)
#   C  goal parsing into a clause checklist (count, colour, object, directions, put-down)
#   D  perception: SAM3 probes on front + wrist, projected clouds, cube / target / stick classes
#   E  plan: embodiment mode (gripper vs fixed stick), visit sequence, heights
#   F  execute: grasp (gripper mode), alternate target visits with dwell, put down, final hold
# Every phase is wrapped so that the program never raises. Nothing here is seed-specific:
# all positions come from runtime perception or the runtime robot state.

import math

report = {"version": "v01_target_filter_button_press", "phase": "start", "errors": [], "log": []}
result = report


def note(msg):
    text = str(msg)
    print("[v01] " + text)
    if len(report["log"]) < 400:
        report["log"].append(text[:500])


def err(where, exc):
    text = where + ": " + str(exc)[:500]
    print("[v01] ERROR " + text)
    if len(report["errors"]) < 200:
        report["errors"].append(text)


def r3(v):
    try:
        return round(float(v), 4)
    except Exception:
        return None


def vec(v):
    out = []
    try:
        for item in v:
            out.append(float(item))
    except Exception:
        pass
    return out


def flat_map(m):
    out = {}
    try:
        for k in m:
            try:
                out[str(k)] = str(m[k])[:800]
            except Exception as exc:
                out[str(k)] = "unreadable: " + str(exc)[:100]
    except Exception as exc:
        out["error"] = str(exc)[:200]
    return out


def quantile(values, frac):
    ordered = sorted(values)
    if len(ordered) == 0:
        return 0.0
    idx = int((len(ordered) - 1) * frac)
    return ordered[idx]


def dist2(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def dist3(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def quat_axes(q):
    # q = [w, x, y, z]; returns (x_axis, y_axis, z_axis) columns of R
    w, x, y, z = q[0], q[1], q[2], q[3]
    xa = [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)]
    ya = [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)]
    za = [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]
    return xa, ya, za


# ---------------------------------------------------------------- Phase A: snapshot
report["phase"] = "A_snapshot"
language = ""
base_frame = "robot_base"
embodiment = "robomme_plus"
max_steps = 0
control_hz = 20.0
gripper_controllable = True
gripper_evidence = []
init_joints = []
init_ee = [0.0, 0.0, 0.0]
init_quat = [0.0, 1.0, 0.0, 0.0]
init_gripper = 1.0
camera_names = []

try:
    ctx = get_task_context()
    language = str(ctx.language)
    report["language"] = language
    try:
        report["task_name"] = str(ctx.task_name)
        report["task_id"] = str(ctx.task_id)
        report["suite"] = str(ctx.suite)
        report["family"] = str(ctx.family)
    except Exception as exc:
        err("context fields", exc)
    try:
        report["context_metadata"] = flat_map(ctx.metadata)
    except Exception as exc:
        err("context metadata", exc)
except Exception as exc:
    err("get_task_context", exc)
note("language: " + language)

try:
    tmd = robomme_plus.get_task_metadata()
    report["task_metadata"] = flat_map(tmd)
    try:
        max_steps = int(tmd["max_steps"])
    except Exception as exc:
        err("max_steps", exc)
    for key in ("control_hz", "control_frequency", "control_freq", "frequency_hz"):
        try:
            control_hz = float(tmd[key])
            break
        except Exception:
            pass
except Exception as exc:
    err("get_task_metadata", exc)
note("max_steps " + str(max_steps) + " control_hz " + str(control_hz))

try:
    cmd = robomme_plus.get_controller_metadata()
    report["controller_metadata"] = flat_map(cmd)
    try:
        arms = cmd["controllable_gripper_arms"]
        gripper_evidence.append("controllable_gripper_arms=" + str(arms))
        if len(arms) == 0:
            gripper_controllable = False
    except Exception:
        gripper_evidence.append("no controllable_gripper_arms key")
    text = str(report["controller_metadata"]).lower()
    if "stick" in text:
        gripper_evidence.append("controller metadata mentions stick")
except Exception as exc:
    err("get_controller_metadata", exc)

try:
    state0 = get_robot_state()
    base_frame = str(state0.base_frame)
    embodiment = str(state0.embodiment)
    init_joints = vec(state0.joint_positions["primary"])
    init_ee = vec(state0.end_effector_poses["primary"].position)
    init_quat = vec(state0.end_effector_poses["primary"].quaternion_wxyz)
    init_gripper = float(state0.gripper_positions["primary"])
    report["robot_state"] = {
        "base_frame": base_frame, "embodiment": embodiment,
        "joints": [r3(v) for v in init_joints], "ee": [r3(v) for v in init_ee],
        "quat": [r3(v) for v in init_quat], "gripper": r3(init_gripper),
        "timestamp_s": r3(state0.timestamp_s),
    }
    try:
        report["robot_state"]["joint_names"] = str(state0.joint_names)[:300]
    except Exception:
        pass
except Exception as exc:
    err("get_robot_state", exc)

front_cam = "front"
wrist_cam = "wrist"
try:
    obs0 = get_observation()
    cams = {}
    for name in obs0.cameras:
        camera_names.append(str(name))
        cam = obs0.cameras[name]
        entry = {}
        try:
            entry["rgb_shape"] = str(cam.rgb.shape)
            entry["depth_shape"] = str(cam.depth_m.shape)
        except Exception as exc:
            entry["shape_error"] = str(exc)[:100]
        try:
            rows = []
            for row in cam.intrinsics:
                rows.append([r3(v) for v in row])
            entry["intrinsics"] = rows
        except Exception as exc:
            entry["intrinsics_error"] = str(exc)[:100]
        try:
            cp = cam.camera_pose
            entry["pose_position"] = [r3(v) for v in cp.position]
            entry["pose_quat"] = [r3(v) for v in cp.quaternion_wxyz]
            entry["pose_frame"] = str(cp.frame)
            axes = quat_axes(vec(cp.quaternion_wxyz))
            entry["axis_x_in_base"] = [r3(v) for v in axes[0]]
            entry["axis_y_in_base"] = [r3(v) for v in axes[1]]
            entry["axis_z_in_base"] = [r3(v) for v in axes[2]]
        except Exception as exc:
            entry["pose_error"] = str(exc)[:100]
        cams[str(name)] = entry
    report["cameras"] = cams
    if "front" not in camera_names and len(camera_names) > 0:
        front_cam = camera_names[0]
    if "wrist" not in camera_names:
        wrist_cam = None
        for name in camera_names:
            if "wrist" in name or "hand" in name:
                wrist_cam = name
except Exception as exc:
    err("get_observation", exc)
note("cameras " + str(camera_names))

# ---------------------------------------------------------------- Phase B: conditioning video
report["phase"] = "B_video"
video_info = {"length": 0}
report["video"] = video_info
try:
    video = robomme_plus.get_conditioning_video()
    path = []
    first_joints = None
    last_joints = None
    count = 0
    for fr in video:
        try:
            rs = fr.robot_state
            p = vec(rs.end_effector_poses["primary"].position)
            g = float(rs.gripper_positions["primary"])
            t = float(fr.timestamp_s)
            if first_joints is None:
                first_joints = vec(rs.joint_positions["primary"])
            last_joints = vec(rs.joint_positions["primary"])
            path.append([t, p[0], p[1], p[2], g])
        except Exception as exc:
            if count < 3:
                err("video frame " + str(count), exc)
        count += 1
    video_info["length"] = count
    if count > 0:
        video_info["t_first"] = r3(path[0][0])
        video_info["t_last"] = r3(path[-1][0])
        k = max(1, len(path) // 80)
        compact = []
        i = 0
        while i < len(path):
            compact.append([r3(v) for v in path[i]])
            i += k
        video_info["path_every_" + str(k)] = compact
        if first_joints is not None and len(init_joints) == 7:
            video_info["first_frame_vs_live_joint_maxdiff"] = r3(
                max([abs(first_joints[j] - init_joints[j]) for j in range(7)]))
            video_info["last_frame_vs_live_joint_maxdiff"] = r3(
                max([abs(last_joints[j] - init_joints[j]) for j in range(7)]))
        ys = [row[2] for row in path]
        zs = [row[3] for row in path]
        gs = [row[4] for row in path]
        video_info["y_range"] = [r3(min(ys)), r3(max(ys))]
        video_info["z_range"] = [r3(min(zs)), r3(max(zs))]
        video_info["g_range"] = [r3(min(gs)), r3(max(gs))]
        # lateral turning points with hysteresis 0.03 m
        turns = []
        direction = 0
        hi_i = 0
        lo_i = 0
        ext_i = 0
        for i in range(1, len(path)):
            y = path[i][2]
            if direction == 0:
                if y > path[hi_i][2]:
                    hi_i = i
                if y < path[lo_i][2]:
                    lo_i = i
                if path[hi_i][2] - y > 0.03:
                    direction = -1
                    ext_i = i
                elif y - path[lo_i][2] > 0.03:
                    direction = 1
                    ext_i = i
            elif direction == 1:
                if y > path[ext_i][2]:
                    ext_i = i
                elif path[ext_i][2] - y > 0.03:
                    turns.append(["max", r3(path[ext_i][0]), r3(path[ext_i][1]), r3(path[ext_i][2]), r3(path[ext_i][3]), r3(path[ext_i][4])])
                    direction = -1
                    ext_i = i
            else:
                if y < path[ext_i][2]:
                    ext_i = i
                elif y - path[ext_i][2] > 0.03:
                    turns.append(["min", r3(path[ext_i][0]), r3(path[ext_i][1]), r3(path[ext_i][2]), r3(path[ext_i][3]), r3(path[ext_i][4])])
                    direction = 1
                    ext_i = i
        video_info["lateral_turns"] = turns[:60]
        events = []
        for i in range(1, len(path)):
            if (path[i - 1][4] - 0.5) * (path[i][4] - 0.5) < 0:
                kind = "close" if path[i][4] < 0.5 else "open"
                events.append([kind, r3(path[i][0]), r3(path[i][1]), r3(path[i][2]), r3(path[i][3])])
        video_info["gripper_events"] = events[:60]
        # cheap RGB statistic on a handful of frames would need a second pass; skipped
    note("video frames " + str(count))
except Exception as exc:
    err("conditioning video", exc)

# ---------------------------------------------------------------- Phase C: goal parsing
report["phase"] = "C_goal"
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "once": 1, "twice": 2, "thrice": 3}
COLOR_WORDS = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "white", "black",
               "gray", "grey", "brown", "cyan", "magenta"]
OBJECT_WORDS = ["cube", "block", "box", "ball", "sphere", "stick", "peg", "can", "bottle"]
DIRECTION_WORDS = ["left", "right"]

goal = {"count": None, "count_source": None, "color": None, "object": None, "directions": [],
        "put_down": False, "mentions_target": False, "mentions_stick": False, "mentions_button": False, "clauses": []}
try:
    low = language.lower()
    chars = []
    for c in low:
        if c.isalnum():
            chars.append(c)
        else:
            chars.append(" ")
    tokens = "".join(chars).split()
    goal["tokens"] = tokens[:80]
    counts = []
    for i in range(len(tokens)):
        tok = tokens[i]
        if tok in ("once", "twice", "thrice"):
            counts.append([NUMBER_WORDS[tok], tok])
        if tok in ("times", "time") and i > 0:
            prev = tokens[i - 1]
            if prev.isdigit():
                counts.append([int(prev), prev + " " + tok])
            elif prev in NUMBER_WORDS:
                counts.append([NUMBER_WORDS[prev], prev + " " + tok])
    goal["count_mentions"] = counts
    if len(counts) > 0:
        goal["count"] = counts[0][0]
        goal["count_source"] = "language: " + str(counts[0][1])
    for i in range(len(tokens)):
        if tokens[i] in OBJECT_WORDS and goal["object"] is None and tokens[i] != "stick":
            goal["object"] = tokens[i]
            j = i - 1
            while j >= 0 and j >= i - 3:
                if tokens[j] in COLOR_WORDS:
                    goal["color"] = tokens[j]
                    break
                j -= 1
    if goal["color"] is None:
        for tok in tokens:
            if tok in COLOR_WORDS:
                goal["color"] = tok
                break
    for tok in tokens:
        if tok in DIRECTION_WORDS:
            goal["directions"].append(tok)
    if "target" in tokens or "targets" in tokens:
        goal["mentions_target"] = True
    if "stick" in tokens:
        goal["mentions_stick"] = True
    if "button" in tokens or "buttons" in tokens:
        goal["mentions_button"] = True
    for i in range(len(tokens)):
        if tokens[i] in ("put", "place", "set", "lower", "release", "drop") :
            window = tokens[i:i + 5]
            if "down" in window or tokens[i] in ("place", "release", "drop", "lower"):
                goal["put_down"] = True
        if tokens[i] == "down":
            goal["put_down"] = True
    # clause split on commas, periods, "then", "and then", "finally"
    pieces = []
    cur = ""
    for c in low:
        if c in ",.;":
            if cur.strip():
                pieces.append(cur.strip())
            cur = ""
        else:
            cur += c
    if cur.strip():
        pieces.append(cur.strip())
    clauses = []
    for piece in pieces:
        for sub in piece.split(" then "):
            if sub.strip():
                clauses.append(sub.strip())
    goal["clauses"] = clauses
except Exception as exc:
    err("goal parse", exc)

# direction order: first appearance order, consecutive duplicates removed
dir_cycle = []
for d in goal["directions"]:
    if len(dir_cycle) == 0 or dir_cycle[-1] != d:
        if d not in dir_cycle:
            dir_cycle.append(d)
if goal["count"] is None:
    goal["count"] = 1
    goal["count_source"] = "default 1 (no count found in language)"
report["goal"] = goal
note("goal parsed: count=" + str(goal["count"]) + " color=" + str(goal["color"]) + " object=" +
     str(goal["object"]) + " dirs=" + str(goal["directions"]) + " put_down=" + str(goal["put_down"]))

if goal["mentions_stick"]:
    gripper_evidence.append("language mentions stick")

# ---------------------------------------------------------------- Phase D: perception
report["phase"] = "D_perception"
probes = []
report["probes"] = probes
pool = []   # projected candidates: dict with stats + meta (no live objects stored in report)


def cloud_stats(points):
    xs = []
    ys = []
    zs = []
    n = len(points)
    stride = 1
    if n > 3000:
        stride = n // 3000 + 1
    i = 0
    while i < n:
        p = points[i]
        xs.append(float(p[0]))
        ys.append(float(p[1]))
        zs.append(float(p[2]))
        i += stride
    if len(xs) == 0:
        return None
    z95 = quantile(zs, 0.95)
    tx = []
    ty = []
    for j in range(len(xs)):
        if zs[j] > z95 - 0.008:
            tx.append(xs[j])
            ty.append(ys[j])
    st = {
        "n": n,
        "x50": quantile(xs, 0.5), "y50": quantile(ys, 0.5),
        "xlo": quantile(xs, 0.05), "xhi": quantile(xs, 0.95),
        "ylo": quantile(ys, 0.05), "yhi": quantile(ys, 0.95),
        "z02": quantile(zs, 0.02), "z05": quantile(zs, 0.05), "z50": quantile(zs, 0.5),
        "z95": z95, "zmax": quantile(zs, 1.0),
        "top_x": quantile(tx, 0.5) if len(tx) > 0 else quantile(xs, 0.5),
        "top_y": quantile(ty, 0.5) if len(ty) > 0 else quantile(ys, 0.5),
    }
    st["cx"] = 0.5 * (st["xlo"] + st["xhi"])
    st["cy"] = 0.5 * (st["ylo"] + st["yhi"])
    st["xext"] = st["xhi"] - st["xlo"]
    st["yext"] = st["yhi"] - st["ylo"]
    st["zext"] = st["z95"] - st["z05"]
    # low points (tip finding for a stick)
    z05 = st["z05"]
    lx = []
    ly = []
    for j in range(len(xs)):
        if zs[j] <= z05 + 0.004:
            lx.append(xs[j])
            ly.append(ys[j])
    st["low_x"] = quantile(lx, 0.5) if len(lx) > 0 else st["x50"]
    st["low_y"] = quantile(ly, 0.5) if len(ly) > 0 else st["y50"]
    return st


def short_stats(st):
    if st is None:
        return None
    out = {}
    for key in ("n", "cx", "cy", "top_x", "top_y", "xext", "yext", "z05", "z50", "z95", "zext"):
        out[key] = r3(st[key])
    return out


def project(cand, camera):
    pc = None
    try:
        pc = mask_to_point_cloud(cand, camera, target_frame=base_frame)
    except Exception as exc:
        try:
            pc = mask_to_point_cloud(cand.mask, camera, target_frame=base_frame)
        except Exception as exc2:
            err("mask_to_point_cloud " + camera, exc2)
            return None
    try:
        return cloud_stats(pc.points)
    except Exception as exc:
        err("cloud_stats", exc)
        return None


rgb_cache = {}


def sample_color(camera, box):
    # majority colour name inside the central 60% of a bbox in the current image (cached per phase)
    try:
        if camera not in rgb_cache:
            rgb_cache[camera] = get_observation().cameras[camera].rgb
        rgb = rgb_cache[camera]
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        cxp = 0.5 * (x1 + x2)
        cyp = 0.5 * (y1 + y2)
        hw = 0.3 * (x2 - x1)
        hh = 0.3 * (y2 - y1)
        votes = {}
        h = len(rgb)
        w = len(rgb[0])
        for a in range(5):
            for b in range(5):
                u = int(cxp - hw + (2 * hw) * a / 4.0)
                v = int(cyp - hh + (2 * hh) * b / 4.0)
                u = min(max(u, 0), w - 1)
                v = min(max(v, 0), h - 1)
                px = rgb[v][u]
                name = color_name(int(px[0]), int(px[1]), int(px[2]))
                votes[name] = votes.get(name, 0) + 1
        best = None
        for name in votes:
            if best is None or votes[name] > votes[best]:
                best = name
        return best, votes
    except Exception as exc:
        err("sample_color", exc)
        return None, {}


def color_name(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 50:
        return "black"
    sat = (mx - mn) / float(mx)
    if sat < 0.22:
        if mx > 190:
            return "white"
        return "gray"
    d = float(mx - mn)
    if mx == r:
        hue = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        hue = 60.0 * ((b - r) / d + 2.0)
    else:
        hue = 60.0 * ((r - g) / d + 4.0)
    if hue < 0:
        hue += 360.0
    if hue < 18 or hue >= 340:
        return "red"
    if hue < 40:
        return "orange"
    if hue < 70:
        return "yellow"
    if hue < 165:
        return "green"
    if hue < 195:
        return "cyan"
    if hue < 255:
        return "blue"
    if hue < 290:
        return "purple"
    return "pink"


def run_probe(camera, prompt, keep, tag):
    rec = {"camera": camera, "prompt": prompt, "tag": tag, "ok": False, "cands": []}
    probes.append(rec)
    try:
        seg = segment_text(camera, prompt)
        rec["ok"] = bool(seg.ok)
        if not seg.ok:
            try:
                rec["error"] = str(seg.error)[:200]
            except Exception:
                pass
            return
        cands = seg.segmentations
        rec["num"] = len(cands)
        idx = 0
        for cand in cands:
            if idx >= keep:
                break
            entry = {"rank": idx}
            try:
                entry["score"] = r3(cand.score)
            except Exception:
                entry["score"] = None
            box = None
            try:
                if cand.box_xyxy is not None:
                    box = [r3(v) for v in cand.box_xyxy]
            except Exception:
                box = None
            entry["box"] = box
            st = project(cand, camera)
            entry["stats"] = short_stats(st)
            if box is not None and camera == front_cam:
                cname, votes = sample_color(camera, box)
                entry["color"] = cname
            rec["cands"].append(entry)
            if st is not None:
                pool.append({"camera": camera, "prompt": prompt, "tag": tag, "rank": idx,
                             "score": entry["score"], "box": box, "st": st,
                             "color": entry.get("color")})
            idx += 1
    except Exception as exc:
        err("probe " + camera + " '" + prompt + "'", exc)
        rec["error"] = str(exc)[:200]


color = goal["color"]
obj_word = goal["object"] if goal["object"] is not None else "cube"
object_prompts = []
if color is not None:
    object_prompts.append(color + " " + obj_word)
object_prompts.append(obj_word)
if obj_word != "cube":
    object_prompts.append("cube")
target_prompts = ["target", "red and white target", "bullseye", "round target"]
stick_mode_hint = (not gripper_controllable) or goal["mentions_stick"]

try:
    run_probe(front_cam, "table", 1, "table")
    for prm in object_prompts:
        run_probe(front_cam, prm, 4, "object")
    for prm in target_prompts:
        run_probe(front_cam, prm, 5, "target")
    if goal.get("mentions_button", False):
        run_probe(front_cam, "button", 3, "button")
    if wrist_cam is not None:
        run_probe(wrist_cam, object_prompts[0], 3, "object")
        run_probe(wrist_cam, "target", 4, "target")
except Exception as exc:
    err("probe block", exc)

# table height
table_z = None
try:
    for item in pool:
        if item["tag"] == "table" and item["st"]["n"] >= 50 and max(item["st"]["xext"], item["st"]["yext"]) > 0.3:
            table_z = item["st"]["z50"]
            break
    if table_z is None:
        lows = []
        for item in pool:
            if item["st"]["n"] >= 15:
                lows.append(item["st"]["z05"])
        if len(lows) > 0:
            table_z = quantile(lows, 0.3)
    if table_z is None:
        table_z = 0.0
except Exception as exc:
    err("table_z", exc)
    table_z = 0.0
report["table_z"] = r3(table_z)
note("table_z " + str(r3(table_z)))


def is_robot_like(st):
    # the arm itself answers many prompts: reject tall clouds and clouds near the base origin
    if st["zext"] > 0.15:
        return True
    if math.sqrt(st["cx"] ** 2 + st["cy"] ** 2) < 0.18:
        return True
    return False


# cube candidates
cube = None
cube_cands = []
try:
    for item in pool:
        st = item["st"]
        if item["tag"] not in ("object", "target"):
            continue
        if st["n"] < 12 or is_robot_like(st):
            continue
        height = st["z95"] - table_z
        if height < 0.012 or height > 0.12:
            continue
        if st["xext"] > 0.10 or st["yext"] > 0.10:
            continue
        cube_cands.append(item)
    best = None
    best_key = None
    for item in cube_cands:
        color_match = 0
        if color is not None and item.get("color") == color:
            color_match = 1
        object_tag = 1 if item["tag"] == "object" else 0
        cam_front = 1 if item["camera"] == front_cam else 0
        sc = item["score"] if item["score"] is not None else 0.0
        key = (color_match, object_tag, cam_front, sc)
        if best_key is None or key > best_key:
            best_key = key
            best = item
    if best is not None:
        st = best["st"]
        cube = {"x": st["top_x"], "y": st["top_y"], "top": st["z95"], "camera": best["camera"],
                "prompt": best["prompt"], "score": best["score"], "color": best.get("color"),
                "key": list(best_key)}
    report["cube_candidates"] = [[c["camera"], c["prompt"], c["rank"], c["score"], c.get("color"),
                                  r3(c["st"]["top_x"]), r3(c["st"]["top_y"]), r3(c["st"]["z95"])]
                                 for c in cube_cands][:20]
except Exception as exc:
    err("cube selection", exc)
report["cube"] = None if cube is None else {"x": r3(cube["x"]), "y": r3(cube["y"]), "top": r3(cube["top"]),
                                            "camera": cube["camera"], "prompt": cube["prompt"],
                                            "score": cube["score"], "color": cube["color"], "key": cube["key"]}
note("cube " + str(report["cube"]))

# button (v01): a raised, compact object that is not the cube. Prefer the "button" probe;
# fall back to raised compact clouds returned by the target prompts (v00's false target).
button = None
button_cands = []


def button_like(st):
    if st["n"] < 12 or is_robot_like(st):
        return False
    height = st["z95"] - table_z
    if height < 0.015 or height > 0.08:
        return False
    if max(st["xext"], st["yext"]) > 0.10:
        return False
    if cube is not None and dist2([st["top_x"], st["top_y"]], [cube["x"], cube["y"]]) < 0.035:
        return False
    return True


try:
    for pass_tag in ("button", "target"):
        if button is not None:
            break
        best_sc = None
        for item in pool:
            if item["tag"] != pass_tag or not button_like(item["st"]):
                continue
            sc = item["score"] if item["score"] is not None else 0.0
            button_cands.append([item["camera"], item["prompt"], item["rank"], sc, r3(item["st"]["top_x"]),
                                 r3(item["st"]["top_y"]), r3(item["st"]["z95"])])
            if item["camera"] != front_cam:
                continue
            if best_sc is None or sc > best_sc:
                best_sc = sc
                st = item["st"]
                button = {"x": st["top_x"], "y": st["top_y"], "top": st["z95"], "prompt": item["prompt"],
                          "tag": pass_tag, "score": sc}
except Exception as exc:
    err("button selection", exc)
report["button_candidates"] = button_cands[:12]
report["button"] = None if button is None else {"x": r3(button["x"]), "y": r3(button["y"]),
                                                "top": r3(button["top"]), "prompt": button["prompt"],
                                                "tag": button["tag"], "score": button["score"]}
note("button " + str(report["button"]))

# target candidates (flat discs at table level, footprint 2.5-30 cm, not the cube, not the button)
targets = []


def collect_targets():
    found = []
    for item in pool:
        st = item["st"]
        if item["tag"] != "target":
            continue
        if st["n"] < 12 or is_robot_like(st):
            continue
        foot = max(st["xext"], st["yext"])
        if foot < 0.025 or foot > 0.30:
            continue
        if st["zext"] > 0.008:
            continue
        if st["z50"] - table_z > 0.02:
            continue
        if cube is not None and dist2([st["cx"], st["cy"]], [cube["x"], cube["y"]]) < 0.035:
            continue
        if button is not None and dist2([st["cx"], st["cy"]], [button["x"], button["y"]]) < 0.04:
            continue
        merged = False
        for t in found:
            if dist2([t["x"], t["y"]], [st["cx"], st["cy"]]) < 0.04:
                t["hits"] += 1
                sc_new = item["score"] if item["score"] is not None else 0.0
                if t["score"] is None or sc_new > t["score"]:
                    t["score"] = sc_new
                if st["n"] > t["n"] and item["camera"] == front_cam:
                    t["x"] = st["cx"]
                    t["y"] = st["cy"]
                    t["n"] = st["n"]
                    t["surface"] = st["z50"]
                merged = True
                break
        if not merged:
            found.append({"x": st["cx"], "y": st["cy"], "surface": st["z50"], "n": st["n"],
                          "foot": foot, "hits": 1, "prompt": item["prompt"], "camera": item["camera"],
                          "score": item["score"], "color": item.get("color")})
    return found


try:
    targets = collect_targets()
    if len(targets) < 2:
        note("fewer than 2 targets from primary prompts; trying fallbacks")
        for prm in ("red target", "circle", "red circle", "disc", "white circle"):
            run_probe(front_cam, prm, 5, "target")
            targets = collect_targets()
            if len(targets) >= 2:
                break
    if len(targets) > 2:
        # v01: the scene has exactly two targets; keep the two best-scoring discs
        ranked = sorted([(-(t["score"] if t["score"] is not None else 0.0), -t["hits"], ti)
                         for ti, t in enumerate(targets)])
        dropped = [targets[row[2]] for row in ranked[2:]]
        note("more than 2 flat discs; dropping " + str([[r3(t["x"]), r3(t["y"]), t["score"]] for t in dropped]))
        targets = [targets[row[2]] for row in ranked[:2]]
except Exception as exc:
    err("target selection", exc)
report["targets_found"] = [{"x": r3(t["x"]), "y": r3(t["y"]), "surface": r3(t["surface"]), "n": t["n"],
                            "foot": r3(t["foot"]), "hits": t["hits"], "prompt": t["prompt"],
                            "camera": t["camera"], "score": t["score"], "color": t["color"]} for t in targets]
note("targets " + str(report["targets_found"]))

# stick tip (only relevant for a fixed-tool embodiment)
tip_offset = None
try:
    if stick_mode_hint or cube is None:
        run_probe(front_cam, "stick", 3, "stick")
        best = None
        for item in pool:
            if item["tag"] != "stick":
                continue
            st = item["st"]
            if dist2([st["low_x"], st["low_y"]], init_ee) > 0.25:
                continue
            if st["z05"] > init_ee[2] + 0.02:
                continue
            if best is None or st["z05"] < best["z05"]:
                best = st
        if best is not None:
            tip_offset = [best["low_x"] - init_ee[0], best["low_y"] - init_ee[1], best["z05"] - init_ee[2]]
            if dist3(tip_offset, [0, 0, 0]) > 0.45:
                tip_offset = None
    report["tip_offset"] = None if tip_offset is None else [r3(v) for v in tip_offset]
except Exception as exc:
    err("stick tip", exc)

# ---------------------------------------------------------------- Phase E: plan
report["phase"] = "E_plan"
mode = "grasp"
if stick_mode_hint:
    mode = "stick"
if mode == "grasp" and cube is None:
    gripper_evidence.append("no cube localized: visiting targets with the tool point instead of grasping")
    mode = "stick"
report["mode"] = mode
report["gripper_evidence"] = gripper_evidence
note("mode " + mode + " evidence " + str(gripper_evidence))

# Lateral convention. left_sign = +1 means "left" is base +y (robot's own left).
# Decision order (logged as convention_branch):
#   1. this episode's conditioning video: its first lateral swing side + the goal's first direction
#   2. fixed fallback guess: robot base frame (+y = left)
left_sign = 1.0
convention = {"branch": "fallback: robot base frame (+y = left), unverified"}
report["convention"] = convention
try:
    fa = report["cameras"][front_cam]["axis_x_in_base"]
    # OpenCV camera convention: camera x = image right. Sign of its base-y component tells
    # whether base +y appears toward image-right or image-left in the front view.
    convention["front_axis_x_in_base"] = fa
    if fa[1] > 0.3:
        convention["base_plus_y_in_front_image"] = "right"
        convention["image_left_sign"] = -1.0
    elif fa[1] < -0.3:
        convention["base_plus_y_in_front_image"] = "left"
        convention["image_left_sign"] = 1.0
    else:
        convention["base_plus_y_in_front_image"] = "ambiguous (camera x has |y| <= 0.3)"
except Exception as exc:
    err("front axis convention", exc)

video_sides = {}
try:
    turns = video_info.get("lateral_turns", [])
    good = []
    if len(turns) >= 2:
        mid = sum([row[3] for row in turns]) / float(len(turns))
        for row in turns:
            if abs(row[3] - mid) >= 0.04:
                good.append(row)
        alternating = True
        for i in range(1, len(good)):
            if good[i][0] == good[i - 1][0]:
                alternating = False
        convention["video_turns_used"] = len(good)
        convention["video_turns_alternate"] = alternating
        if len(good) >= 2 and alternating:
            first_sign = 1.0 if good[0][0] == "max" else -1.0
            convention["video_first_swing_sign"] = first_sign
            for sgn in (1.0, -1.0):
                rows = [row for row in good if (1.0 if row[0] == "max" else -1.0) == sgn]
                if len(rows) > 0:
                    video_sides[sgn] = {
                        "x": sum([row[2] for row in rows]) / len(rows),
                        "y": sum([row[3] for row in rows]) / len(rows),
                        "z": sum([row[4] for row in rows]) / len(rows),
                        "g": sum([row[5] for row in rows]) / len(rows),
                        "n": len(rows)}
            convention["video_side_poses"] = {}
            for sgn in video_sides:
                vs = video_sides[sgn]
                convention["video_side_poses"][str(sgn)] = [r3(vs["x"]), r3(vs["y"]), r3(vs["z"]), r3(vs["g"]), vs["n"]]
            if len(dir_cycle) >= 1:
                d0 = dir_cycle[0]
                left_sign = first_sign if d0 == "left" else -first_sign
                convention["branch"] = ("video: first swing went to base sign " + str(first_sign) +
                                        " and the goal's first direction is '" + d0 + "'")
            else:
                convention["branch"] = "fallback (video swings found but the goal names no direction)"
except Exception as exc:
    err("video convention", exc)
convention["left_sign"] = left_sign
try:
    if "image_left_sign" in convention:
        convention["agrees_with"] = "image-left (front view)" if convention["image_left_sign"] == left_sign else "robot base frame"
        if convention["image_left_sign"] == 1.0:
            convention["agrees_with"] = "both (front camera looks along +x)"
except Exception as exc:
    err("convention agree", exc)
note("convention " + str(convention))

visit_plan = []
anchor = None
if cube is not None:
    anchor = [cube["x"], cube["y"]]
else:
    anchor = [init_ee[0], init_ee[1]]
try:
    tsorted = []
    for ti in range(len(targets)):
        tsorted.append((targets[ti]["y"], targets[ti]["x"], ti))
    tsorted = sorted(tsorted)
    tsorted = [(row[0], row[1], targets[row[2]]) for row in tsorted]
    left_t = None
    right_t = None
    if len(tsorted) >= 2:
        right_t = tsorted[0][2]
        left_t = tsorted[-1][2]
        if left_sign < 0:
            right_t = tsorted[-1][2]
            left_t = tsorted[0][2]
    elif len(tsorted) == 1:
        t = tsorted[0][2]
        mirror = {"x": t["x"], "y": 2 * anchor[1] - t["y"], "surface": t["surface"], "virtual": True}
        if abs(t["y"] - anchor[1]) < 0.05:
            mirror["y"] = t["y"] - 0.15
        if (t["y"] - mirror["y"]) * left_sign >= 0:
            left_t = t
            right_t = mirror
        else:
            left_t = mirror
            right_t = t
    else:
        left_t = {"x": anchor[0], "y": anchor[1] + 0.15 * left_sign, "surface": table_z, "virtual": True}
        right_t = {"x": anchor[0], "y": anchor[1] - 0.15 * left_sign, "surface": table_z, "virtual": True}
    named = {"left": left_t, "right": right_t}
    cycle = []
    for d in dir_cycle:
        if d in named:
            cycle.append(d)
    if len(cycle) == 0:
        # no side named: start with the target nearer the object, then alternate
        dl = dist2([left_t["x"], left_t["y"]], anchor)
        dr = dist2([right_t["x"], right_t["y"]], anchor)
        cycle = ["left", "right"] if dl <= dr else ["right", "left"]
        report["cycle_source"] = "no direction words; nearer target first"
    elif len(cycle) == 1:
        other = "right" if cycle[0] == "left" else "left"
        cycle = [cycle[0], other]
        report["cycle_source"] = "one direction word; alternate with the other side"
    else:
        report["cycle_source"] = "direction words in language order"
    for rep in range(int(goal["count"])):
        for d in cycle:
            visit_plan.append(d)
    report["cycle"] = cycle
    report["visit_plan"] = visit_plan
    report["named_targets"] = {}
    for key in named:
        t = named[key]
        report["named_targets"][key] = {"x": r3(t["x"]), "y": r3(t["y"]), "surface": r3(t["surface"]),
                                        "virtual": bool(t.get("virtual", False))}
except Exception as exc:
    err("plan", exc)
    named = {}
note("visit plan " + str(visit_plan))

# goal checklist mapped to phases
checklist = []
for clause in goal["clauses"]:
    phase_name = "unmapped"
    if "pick" in clause or "grasp" in clause or "grab" in clause:
        phase_name = "F1 grasp"
    if "left" in clause or "right" in clause or "target" in clause or "swing" in clause:
        phase_name = "F2 visits"
    if "time" in clause or "repeat" in clause or "once" in clause or "twice" in clause:
        phase_name = "F2 visits x count"
    if "down" in clause or "place" in clause or "release" in clause:
        phase_name = "F3 put down"
    if "button" in clause or "press" in clause:
        phase_name = (phase_name + " + " if phase_name != "unmapped" else "") + "F4 press button"
    checklist.append([clause, phase_name])
report["checklist"] = checklist
note("checklist " + str(checklist))

# ---------------------------------------------------------------- motion helpers
state_box = {"done": False, "done_reason": None, "ticks": 0, "step_failures": 0, "first_reply": None,
             "last_q": None}
down_quat = init_quat
try:
    za = quat_axes(init_quat)[2]
    report["init_tool_z_axis"] = [r3(v) for v in za]
    if za[2] > -0.9:
        down_quat = [0.0, 1.0, 0.0, 0.0]
        note("initial EE not top-down; using [0,1,0,0]")
except Exception as exc:
    err("quat", exc)
    down_quat = [0.0, 1.0, 0.0, 0.0]


def reply_terminal(reply, where):
    term = False
    flags = {}
    try:
        flags["terminated"] = bool(reply.terminated)
        if flags["terminated"]:
            term = True
    except Exception:
        pass
    try:
        flags["truncated"] = bool(reply.truncated)
        if flags["truncated"]:
            term = True
    except Exception:
        pass
    try:
        status = str(reply.status).lower()
        flags["status"] = status[:60]
        if "terminat" in status or "truncat" in status or "episode_end" in status or "done" == status:
            term = True
    except Exception:
        pass
    if state_box["first_reply"] is None:
        try:
            state_box["first_reply"] = {"where": where, "flags": flags, "repr": str(reply)[:700]}
            report["first_reply"] = state_box["first_reply"]
        except Exception:
            pass
    if term and not state_box["done"]:
        state_box["done"] = True
        state_box["done_reason"] = where + " " + str(flags)
        report["terminal"] = {"where": where, "flags": flags, "sim_time_s": r3(sim_time())}
        note("TERMINAL at " + where + " flags " + str(flags))
    return term


def sim_time():
    try:
        return float(get_robot_state().timestamp_s)
    except Exception:
        return -1.0


def joints_now():
    return vec(get_robot_state().joint_positions["primary"])


def ee_now():
    return vec(get_robot_state().end_effector_poses["primary"].position)


def send(q, g):
    if state_box["done"]:
        return False
    try:
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=[float(v) for v in q],
            gripper_position=float(g), embodiment=embodiment)}))
        state_box["ticks"] += 1
        state_box["step_failures"] = 0
        state_box["last_q"] = [float(v) for v in q]
        if reply_terminal(reply, "step"):
            return False
        try:
            if not reply.ok:
                note("step not ok: " + str(reply.error)[:200])
        except Exception:
            pass
        return True
    except Exception as exc:
        state_box["step_failures"] += 1
        err("step", exc)
        if state_box["step_failures"] >= 3:
            state_box["done"] = True
            state_box["done_reason"] = "3 consecutive step exceptions"
        return False


def servo(qt, g, rate, tol, settle):
    q0 = joints_now()
    diff = max([abs(qt[i] - q0[i]) for i in range(7)])
    nsteps = int(diff / rate) + 1
    for k in range(1, nsteps + 1):
        a = k / float(nsteps)
        q = [q0[i] + a * (qt[i] - q0[i]) for i in range(7)]
        if not send(q, g):
            return False
    for k in range(settle):
        qn = joints_now()
        e = max([abs(qt[i] - qn[i]) for i in range(7)])
        if e < tol:
            return True
        if not send(qt, g):
            return False
    return True


def ik(pos, seed):
    try:
        res = solve_ik(Pose(position=[float(v) for v in pos], quaternion_wxyz=[float(v) for v in down_quat],
                            frame=base_frame), seed_joints=seed)
        if res.ok:
            return vec(res.joint_positions)
        try:
            note("ik failed at " + str([r3(v) for v in pos]) + ": " + str(res.error)[:200])
        except Exception:
            note("ik failed at " + str([r3(v) for v in pos]))
    except Exception as exc:
        err("solve_ik", exc)
    return None


def goto(pos, g, pieces, label, rate=0.035, tol=0.01, settle=25):
    # Cartesian-straight approach: IK at evenly spaced points from the current EE position.
    if state_box["done"]:
        return False
    start = ee_now()
    seed = joints_now()
    sols = []
    for k in range(1, pieces + 1):
        a = k / float(pieces)
        p = [start[i] + a * (pos[i] - start[i]) for i in range(3)]
        q = ik(p, seed)
        if q is None:
            sols = None
            break
        sols.append(q)
        seed = q
    ok = True
    if sols is None:
        note(label + ": IK chain failed, falling back to move_to_pose")
        try:
            res = move_to_pose(Pose(position=[float(v) for v in pos], quaternion_wxyz=[float(v) for v in down_quat],
                                    frame=base_frame), tolerance=0.012, max_steps=200)
            reply_terminal(res, "move_to_pose " + label)
            ok = bool(res.ok)
            state_box["last_q"] = None
        except Exception as exc:
            err("move_to_pose " + label, exc)
            ok = False
    else:
        for k in range(len(sols)):
            last = (k == len(sols) - 1)
            if last:
                ok = servo(sols[k], g, rate, tol, settle)
            else:
                ok = servo(sols[k], g, rate, 0.04, 3)
            if not ok:
                break
    reached = ee_now()
    errv = dist3(reached, pos)
    note(label + " -> err " + str(r3(errv)) + " ee " + str([r3(v) for v in reached]) + " t=" + str(r3(sim_time())))
    return ok and errv < 0.03


def hold(g, n, label):
    q = state_box["last_q"]
    if q is None:
        q = joints_now()
    for k in range(n):
        if not send(q, g):
            return False
    return True


def gripper_reading():
    try:
        return float(get_robot_state().gripper_positions["primary"])
    except Exception:
        return -1.0


# ---------------------------------------------------------------- Phase F: execute
report["phase"] = "F_execute"
execution = {"visits": []}
report["execution"] = execution
grip_cmd = 1.0
held_dz = 0.02     # EE height above the object's bottom while holding (set from the grasp)
grasp_ee = None
stick_fallback_z = None
def wrist_refine(cx, cy, top, label):
    # re-localize the cube's top face from the wrist camera while hovering above it
    if wrist_cam is None or state_box["done"]:
        return None
    before = len(pool)
    run_probe(wrist_cam, object_prompts[0], 3, label)
    refined = None
    bestd = None
    for item in pool[before:]:
        st = item["st"]
        d = dist2([st["top_x"], st["top_y"]], [cx, cy])
        if d < 0.05 and st["n"] >= 12 and abs(st["z95"] - top) < 0.03:
            if bestd is None or d < bestd:
                bestd = d
                refined = [st["top_x"], st["top_y"], st["z95"]]
    return refined


def attempt_grasp(attempt):
    rec = {"attempt": attempt}
    execution["grasps"].append(rec)
    r = open_gripper()
    reply_terminal(r, "open_gripper")
    top = cube["top"]
    height = max(0.015, top - table_z)
    grasp_z = max(top - min(0.022, 0.5 * height), table_z + 0.013)
    rec["grasp_z"] = r3(grasp_z)
    goto([cube["x"], cube["y"], top + 0.10], 1.0, 1, "pre-grasp a" + str(attempt))
    refined = wrist_refine(cube["x"], cube["y"], top, "wrist_refine")
    rec["wrist_refine"] = None if refined is None else [r3(v) for v in refined]
    if refined is not None:
        cube["x"] = refined[0]
        cube["y"] = refined[1]
        goto([cube["x"], cube["y"], top + 0.10], 1.0, 1, "pre-grasp refined a" + str(attempt))
    goto([cube["x"], cube["y"], grasp_z + 0.04], 1.0, 2, "descend-1", rate=0.025)
    goto([cube["x"], cube["y"], grasp_z], 1.0, 1, "descend-2", rate=0.015, tol=0.006)
    r = close_gripper()
    reply_terminal(r, "close_gripper")
    hold(0.0, 3, "post-close")
    rec["gripper_after_close"] = r3(gripper_reading())
    ee = ee_now()
    rec["grasp_ee"] = [r3(v) for v in ee]
    goto([cube["x"], cube["y"], ee[2] + 0.06], 0.0, 2, "lift", rate=0.025)
    g_after_lift = gripper_reading()
    rec["gripper_after_lift"] = r3(g_after_lift)
    rec["ok"] = g_after_lift >= 0.02
    if not rec["ok"]:
        note("gripper closed on nothing after lift (reading " + str(r3(g_after_lift)) + ")")
    return rec["ok"], ee


execution["grasps"] = []
try:
    already = False
    if mode == "grasp" and cube is not None:
        dee = dist3([cube["x"], cube["y"], cube["top"]], init_ee)
        execution["cube_to_ee_at_start"] = r3(dee)
        if 0.05 < init_gripper < 0.9 and dee < 0.06:
            already = True
            grasp_ee = init_ee
            held_dz = init_ee[2] - table_z
            grip_cmd = 0.0
            execution["grasp_ok"] = True
            execution["held_dz"] = r3(held_dz)
            note("object already held at start (gripper " + str(r3(init_gripper)) + "); skipping grasp")
    if mode == "grasp" and not already:
        for attempt in range(2):
            if state_box["done"]:
                break
            okg, ee = attempt_grasp(attempt)
            if okg:
                grasp_ee = ee
                held_dz = grasp_ee[2] - table_z
                grip_cmd = 0.0
                execution["grasp_ok"] = True
                execution["held_dz"] = r3(held_dz)
                break
            execution["grasp_ok"] = False
            grip_cmd = 1.0
            if attempt == 0 and not state_box["done"]:
                r = open_gripper()
                reply_terminal(r, "open_gripper retry")
    elif mode == "stick":
        grip_cmd = init_gripper
        if tip_offset is None:
            zmin = None
            try:
                if video_info["length"] > 0:
                    zmin = float(video_info["z_range"][0])
            except Exception:
                zmin = None
            stick_fallback_z = zmin
            tip_offset = [0.0, 0.0, 0.0]
            note("stick tip not localized; EE point visits, video min EE z = " + str(zmin))
            execution["tip_offset_source"] = "default zero; video min z " + str(zmin)
        else:
            execution["tip_offset_source"] = "front stick cloud lowest points"
except Exception as exc:
    err("grasp phase", exc)

# visits
try:
    can_visit = (mode == "stick") or (mode == "grasp" and cube is not None and execution.get("grasp_ok", False))
    if mode == "grasp" and cube is not None and not execution.get("grasp_ok", False):
        note("grasp not verified; still attempting visits is pointless; skipping visits")
    if can_visit and not state_box["done"]:
        hover = 0.010
        lift_between = 0.05
        live_matches_video = False
        try:
            live_matches_video = float(video_info.get("first_frame_vs_live_joint_maxdiff", 9.0)) < 0.03
        except Exception:
            live_matches_video = False
        execution["live_matches_video_start"] = live_matches_video
        execution["visit_pose_source"] = "perception"
        for vi in range(len(visit_plan)):
            if state_box["done"]:
                break
            side = visit_plan[vi]
            t = named[side]
            if mode == "grasp":
                vz = t["surface"] + hover + held_dz
                vx = t["x"]
                vy = t["y"]
            else:
                vz = t["surface"] + hover - tip_offset[2]
                if stick_fallback_z is not None:
                    vz = max(stick_fallback_z, t["surface"] + hover)
                elif execution.get("tip_offset_source", "").startswith("default"):
                    vz = t["surface"] + 0.04
                vx = t["x"] - tip_offset[0]
                vy = t["y"] - tip_offset[1]
            sgn = left_sign if side == "left" else -left_sign
            vs = video_sides.get(sgn)
            if vs is not None and live_matches_video:
                if mode == "stick":
                    # same scene as the video: the recorded EE pose at that swing extreme
                    vx = vs["x"]
                    vy = vs["y"]
                    vz = vs["z"]
                    execution["visit_pose_source"] = "video swing extremes (stick, live start == video start)"
                elif vs["g"] < 0.9:
                    lo = t["surface"] + held_dz + 0.005
                    if lo <= vs["z"] <= lo + 0.12:
                        vz = vs["z"]
                        execution["visit_pose_source"] = "perception xy, video swing-extreme height"
            cur = ee_now()
            high = max(vz + lift_between, cur[2])
            if dist2(cur, [vx, vy]) > 0.03:
                if high - cur[2] > 0.01:
                    goto([cur[0], cur[1], high], grip_cmd, 1, "visit" + str(vi) + "-up", rate=0.035, tol=0.03, settle=4)
                goto([vx, vy, high], grip_cmd, 2, "visit" + str(vi) + "-over", rate=0.04, tol=0.02, settle=6)
            goto([vx, vy, vz], grip_cmd, 1, "visit" + str(vi) + "-down", rate=0.025, tol=0.008, settle=20)
            hold(grip_cmd, 6, "dwell")
            if vi == len(visit_plan) - 1:
                # final visit: subgoals register a few ticks after arrival; never leave right away
                hold(grip_cmd, 24, "final-visit dwell")
            e = ee_now()
            rec = {"i": vi, "side": side, "target": [r3(t["x"]), r3(t["y"]), r3(t["surface"])],
                   "cmd": [r3(vx), r3(vy), r3(vz)], "ee": [r3(v) for v in e], "t": r3(sim_time()),
                   "gripper": r3(gripper_reading())}
            execution["visits"].append(rec)
            note("visit " + str(rec))
        if not state_box["done"]:
            cur = ee_now()
            goto([cur[0], cur[1], cur[2] + 0.04], grip_cmd, 1, "post-visits-up", rate=0.03, tol=0.02, settle=4)
except Exception as exc:
    err("visit phase", exc)

# put down (gripper mode)
try:
    if mode == "grasp" and grasp_ee is not None and not state_box["done"]:
        px = cube["x"]
        py = cube["y"]
        real = [named[k] for k in named if not named[k].get("virtual", False)]
        near = False
        for t in real:
            if dist2([px, py], [t["x"], t["y"]]) < 0.06:
                near = True
        if near and len(real) >= 2:
            mx = 0.5 * (real[0]["x"] + real[1]["x"])
            my = 0.5 * (real[0]["y"] + real[1]["y"])
            if min([dist2([mx, my], [t["x"], t["y"]]) for t in real]) >= 0.06:
                px = mx
                py = my
                note("original cube spot lies on a target; putting down between the targets")
        cur = ee_now()
        goto([px, py, max(cur[2], grasp_ee[2] + 0.06)], grip_cmd, 2, "to-putdown", rate=0.04, tol=0.02, settle=6)
        goto([px, py, grasp_ee[2] + 0.004], grip_cmd, 1, "putdown-lower", rate=0.02, tol=0.006, settle=20)
        if not state_box["done"]:
            r = open_gripper()
            reply_terminal(r, "open_gripper putdown")
            grip_cmd = 1.0
        hold(1.0, 4, "release-settle")
        cur = ee_now()
        goto([cur[0], cur[1], cur[2] + 0.08], 1.0, 1, "retreat", rate=0.03, tol=0.02, settle=6)
        execution["putdown"] = [r3(px), r3(py)]
except Exception as exc:
    err("put-down phase", exc)

# press the button (v01; the goal's final clause, never executed by v00)
try:
    if goal.get("mentions_button", False) and not state_box["done"]:
        if button is None:
            note("goal names a button but no button was perceived; press skipped")
            execution["button_press"] = "skipped: no button perceived"
        else:
            bx = button["x"]
            by = button["y"]
            btop = button["top"]
            if mode == "grasp":
                r = close_gripper()
                reply_terminal(r, "close_gripper before press")
                grip_cmd = 0.0
            cur = ee_now()
            goto([bx, by, max(cur[2], btop + 0.08)], grip_cmd, 2, "to-button", rate=0.04, tol=0.02, settle=6)
            goto([bx, by, btop + 0.02], grip_cmd, 1, "button-approach", rate=0.025, tol=0.008, settle=10)
            goto([bx, by, btop - 0.010], grip_cmd, 1, "button-press", rate=0.008, tol=0.03, settle=10)
            hold(grip_cmd, 10, "press-hold")
            pe = ee_now()
            execution["button_press"] = {"button": [r3(bx), r3(by), r3(btop)], "ee": [r3(v) for v in pe],
                                         "t": r3(sim_time()), "done_after_press": state_box["done"]}
            note("button press " + str(execution["button_press"]))
            goto([bx, by, btop + 0.06], grip_cmd, 1, "button-release", rate=0.03, tol=0.02, settle=4)
except Exception as exc:
    err("button phase", exc)

# final hold until terminal (bounded)
try:
    n = 0
    while not state_box["done"] and n < 60:
        if not hold(grip_cmd, 5, "final"):
            break
        n += 5
    execution["final_hold_ticks"] = n
except Exception as exc:
    err("final hold", exc)

report["phase"] = "done"
report["ticks_sent_via_step"] = state_box["ticks"]
report["done"] = state_box["done"]
report["done_reason"] = state_box["done_reason"]
try:
    report["final_ee"] = [r3(v) for v in ee_now()]
    report["final_gripper"] = r3(gripper_reading())
    report["final_sim_time_s"] = r3(sim_time())
except Exception as exc:
    err("final state", exc)
note("finished: done=" + str(state_box["done"]) + " reason=" + str(state_box["done_reason"]))
result = report
