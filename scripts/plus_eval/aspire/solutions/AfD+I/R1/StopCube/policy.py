# v02_settle_latency -- StopCube (robomme_plus task 9), ASPIRE task worker, AfD+I, R=1.
#
# Goal (from get_task_context().language): "press the button to stop the cube just as it
# reaches the target for the <N-th> time".
#
# Strategy (hypothesis from released recordings; every task-dependent quantity is perceived):
#   1. Parse N from the language goal.
#   2. Before any control tick: segment "button", "target", "cube" on the front camera,
#      project the button mask to robot_base (button xy + top z), take the cube's colour
#      from its own mask box and the target's pixel centre from its box.
#   3. Every control tick is a single step() (joint_position hold/interpolation), so the
#      program observes the front camera after EVERY tick and never misses a cube pass.
#      A pure-Python colour tracker follows the cube in a window around its last position.
#   4. A "reach" = the cube's pixel distance to the target centre falls below a fraction of
#      the cube's own pixel size and rises again.  Count reaches from execution start.
#   5. Meanwhile move the closed gripper above the button (joint interpolation, one tick per
#      step), descend to a hover height, rehearse a short dip that stops well above the
#      button to measure the controller's lag, and precompute the press profile.
#   6. When N-1 reaches are done, predict the time to the N-th reach (template match against
#      the approach before earlier reaches, else straight-line closest approach) and start
#      the press when the predicted time-to-reach <= press latency + lead.
# Every phase is guarded; failures are recorded in result and the program falls back to
# keep stepping (it always reaches control steps).

import math

result = {
    "program": "v02_settle_latency",
    "phase": "start",
    "errors": [],
    "notes": [],
}


def note_error(where, exc):
    try:
        if len(result["errors"]) >= 80:
            result["errors_dropped"] = result.get("errors_dropped", 0) + 1
            return
        result["errors"].append(str(where) + ": " + str(exc)[:400])
    except Exception:
        pass


def note(msg):
    if len(result["notes"]) < 200:
        result["notes"].append(str(msg)[:300])


def r3(value):
    try:
        return round(float(value), 4)
    except Exception:
        return None


# ---------------------------------------------------------------- goal parsing
ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
    "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16,
    "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20,
    "once": 1, "twice": 2, "thrice": 3,
}


def parse_count(text):
    cleaned = ""
    for ch in str(text).lower():
        if ch.isalnum():
            cleaned += ch
        else:
            cleaned += " "
    words = cleaned.split()
    found = None
    for word in words:
        if word in ORDINAL_WORDS:
            found = ORDINAL_WORDS[word]
        else:
            digits = ""
            for ch in word:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            rest = word[len(digits):]
            if digits and rest in ("st", "nd", "rd", "th", ""):
                if rest != "" or "time" in words:
                    found = int(digits)
    return found


# ---------------------------------------------------------------- observation / metadata
context = None
goal = ""
try:
    context = get_task_context()
    goal = str(context.language)
except Exception as exc:
    note_error("get_task_context", exc)
result["goal"] = goal

try:
    result["task_name"] = str(context.task_name)
    result["task_id"] = str(context.task_id)
except Exception as exc:
    note_error("context fields", exc)

task_meta = {}
try:
    task_meta = dict(robomme_plus.get_task_metadata())
    compact = {}
    for key in task_meta:
        compact[str(key)] = str(task_meta[key])[:300]
    result["task_metadata"] = compact
except Exception as exc:
    note_error("get_task_metadata", exc)

try:
    ctrl_meta = dict(robomme_plus.get_controller_metadata())
    compact = {}
    for key in ctrl_meta:
        compact[str(key)] = str(ctrl_meta[key])[:300]
    result["controller_metadata"] = compact
except Exception as exc:
    note_error("get_controller_metadata", exc)

max_steps = 600
try:
    if "max_steps" in task_meta:
        max_steps = int(task_meta["max_steps"])
except Exception as exc:
    note_error("max_steps", exc)
result["max_steps"] = max_steps

target_count = parse_count(goal)
result["target_count_parsed"] = target_count
if target_count is None or target_count < 1:
    note("could not parse the ordinal from the goal; falling back to N=1")
    target_count = 1
result["target_count"] = target_count

state0 = None
base_frame = "robot_base"
embodiment = "robomme_plus"
try:
    state0 = get_robot_state()
    base_frame = state0.base_frame
    embodiment = state0.embodiment
    result["base_frame"] = str(base_frame)
    result["embodiment"] = str(embodiment)
    result["initial_joints"] = [r3(v) for v in state0.joint_positions["primary"]]
    ee0 = state0.end_effector_poses["primary"]
    result["initial_ee"] = {
        "position": [r3(v) for v in ee0.position],
        "quaternion_wxyz": [r3(v) for v in ee0.quaternion_wxyz],
        "frame": str(ee0.frame),
    }
    result["initial_gripper"] = r3(state0.gripper_positions["primary"])
    result["initial_state_timestamp_s"] = r3(state0.timestamp_s)
except Exception as exc:
    note_error("get_robot_state", exc)

obs0 = None
camera_names = []
front = "front"
wrist = "wrist"
try:
    obs0 = get_observation()
    for name in obs0.cameras:
        camera_names.append(str(name))
    result["camera_names"] = camera_names
    result["obs0_timestamp_s"] = r3(obs0.timestamp_s)
except Exception as exc:
    note_error("get_observation", exc)
if "front" not in camera_names:
    for candidate in ("agentview", "head", "top"):
        if candidate in camera_names:
            front = candidate
            break
    if front == "front" and camera_names:
        front = camera_names[0]
result["front_camera"] = front

W = 256
H = 256
try:
    rgb0 = obs0.cameras[front].rgb
    H = len(rgb0)
    W = len(rgb0[0])
    cam0 = obs0.cameras[front]
    result["front_camera_pose"] = {
        "position": [r3(v) for v in cam0.camera_pose.position],
        "quaternion_wxyz": [r3(v) for v in cam0.camera_pose.quaternion_wxyz],
    }
except Exception as exc:
    note_error("front rgb shape", exc)
result["image_wh"] = [W, H]

# Conditioning video: diagnostics only (released StopCube recordings have no video phase).
try:
    video = robomme_plus.get_conditioning_video()
    video_len = len(video)
    result["conditioning_video_frames"] = video_len
    if video_len > 0:
        result["conditioning_video_t0"] = r3(video[0].timestamp_s)
        result["conditioning_video_t1"] = r3(video[video_len - 1].timestamp_s)
except Exception as exc:
    note_error("get_conditioning_video", exc)
    result["conditioning_video_frames"] = None


# ---------------------------------------------------------------- pixel helpers
def pix(rgb, x, y):
    p = rgb[y][x]
    return (int(p[0]), int(p[1]), int(p[2]))


def norm_box(box):
    if box is None:
        return None
    vals = [float(v) for v in box]
    if max(vals) <= 1.5:
        vals = [vals[0] * W, vals[1] * H, vals[2] * W, vals[3] * H]
    return vals


def box_center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def box_iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    if ua <= 0:
        return 0.0
    return inter / ua


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return None
    if n % 2 == 1:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def inner_color(rgb, box, frac):
    cx, cy = box_center(box)
    hw = max(1.0, (box[2] - box[0]) * frac / 2.0)
    hh = max(1.0, (box[3] - box[1]) * frac / 2.0)
    rs = []
    gs = []
    bs = []
    for y in range(max(0, int(cy - hh)), min(H, int(cy + hh) + 1)):
        for x in range(max(0, int(cx - hw)), min(W, int(cx + hw) + 1)):
            p = pix(rgb, x, y)
            rs.append(p[0])
            gs.append(p[1])
            bs.append(p[2])
    if not rs:
        return None
    return (median(rs), median(gs), median(bs))


def ring_color(rgb, box, margin):
    rs = []
    gs = []
    bs = []
    x0 = int(box[0]) - margin
    x1 = int(box[2]) + margin
    y0 = int(box[1]) - margin
    y1 = int(box[3]) + margin
    for x in range(x0, x1 + 1, 2):
        for y in (y0, y1):
            if 0 <= x < W and 0 <= y < H:
                p = pix(rgb, x, y)
                rs.append(p[0])
                gs.append(p[1])
                bs.append(p[2])
    for y in range(y0, y1 + 1, 2):
        for x in (x0, x1):
            if 0 <= x < W and 0 <= y < H:
                p = pix(rgb, x, y)
                rs.append(p[0])
                gs.append(p[1])
                bs.append(p[2])
    if not rs:
        return None
    return (median(rs), median(gs), median(bs))


def cdist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


# ---------------------------------------------------------------- segmentation
result["segmentation"] = []


def seg_candidates(camera, prompts):
    out = []
    for prompt in prompts:
        rec = {"camera": camera, "prompt": prompt, "ok": False, "cands": []}
        try:
            found = segment_text(camera, prompt)
            rec["ok"] = bool(found.ok)
            if found.ok:
                for cand in found.segmentations[:5]:
                    box = None
                    try:
                        if cand.box_xyxy is not None:
                            box = norm_box(cand.box_xyxy)
                    except Exception as exc:
                        note_error("box " + prompt, exc)
                    score = 0.0
                    try:
                        score = float(cand.score)
                    except Exception:
                        score = 0.0
                    rec["cands"].append({"score": r3(score), "box": None if box is None else [round(v, 1) for v in box]})
                    if box is not None:
                        out.append({"prompt": prompt, "cand": cand, "score": score, "box": box})
        except Exception as exc:
            note_error("segment_text " + prompt, exc)
        result["segmentation"].append(rec)
        if len(out) > 0 and len(rec["cands"]) > 0:
            break
    return out


result["phase"] = "perception"
button_cands = seg_candidates(front, ("button", "push button", "white button"))
target_cands = seg_candidates(front, ("target", "round target", "bullseye", "disc"))
cube_cands = seg_candidates(front, ("cube", "small cube", "block"))

button = None
for cand in button_cands:
    bw = cand["box"][2] - cand["box"][0]
    bh = cand["box"][3] - cand["box"][1]
    if bw < 0.4 * W and bh < 0.4 * H and bw > 2 and bh > 2:
        if button is None or cand["score"] > button["score"]:
            button = cand
result["button_box"] = None if button is None else [round(v, 1) for v in button["box"]]

target = None
for cand in target_cands:
    bw = cand["box"][2] - cand["box"][0]
    bh = cand["box"][3] - cand["box"][1]
    if bw >= 0.4 * W or bh >= 0.4 * H or bw <= 2 or bh <= 2:
        continue
    if button is not None and box_iou(cand["box"], button["box"]) > 0.3:
        continue
    if target is None or cand["score"] > target["score"]:
        target = cand
result["target_box"] = None if target is None else [round(v, 1) for v in target["box"]]

cube = None
for cand in cube_cands:
    bw = cand["box"][2] - cand["box"][0]
    bh = cand["box"][3] - cand["box"][1]
    if bw >= 0.25 * W or bh >= 0.25 * H or bw <= 2 or bh <= 2:
        continue
    if button is not None and box_iou(cand["box"], button["box"]) > 0.3:
        continue
    if target is not None and box_iou(cand["box"], target["box"]) > 0.5:
        continue
    if cube is None or cand["score"] > cube["score"]:
        cube = cand
result["cube_box"] = None if cube is None else [round(v, 1) for v in cube["box"]]

# ---------------------------------------------------------------- tracker setup
track = {
    "ok": False,
    "ref": None,
    "thr": 60,
    "size": 10.0,
    "x": None,
    "y": None,
    "n0": 0,
    "source": None,
}
target_px = None
if target is not None:
    target_px = box_center(target["box"])
result["target_px"] = None if target_px is None else [round(target_px[0], 1), round(target_px[1], 1)]

try:
    if cube is not None and obs0 is not None:
        rgb0 = obs0.cameras[front].rgb
        ref = inner_color(rgb0, cube["box"], 0.5)
        bg = ring_color(rgb0, cube["box"], 3)
        track["ref"] = ref
        track["bg"] = bg
        cx, cy = box_center(cube["box"])
        track["x"] = cx
        track["y"] = cy
        track["size"] = max(4.0, ((cube["box"][2] - cube["box"][0]) + (cube["box"][3] - cube["box"][1])) / 2.0)
        track["source"] = "sam_box"
        if ref is not None and bg is not None:
            sep = cdist(ref, bg)
            track["ref_bg_sep"] = sep
            track["thr"] = max(30, min(65, int(sep * 0.45)))
            track["ok"] = True
except Exception as exc:
    note_error("tracker init", exc)


def find_cube(rgb, cx, cy, rad, stride):
    ref = track["ref"]
    thr = track["thr"]
    x0 = max(0, int(cx - rad))
    x1 = min(W, int(cx + rad) + 1)
    y0 = max(0, int(cy - rad))
    y1 = min(H, int(cy + rad) + 1)
    sx = 0.0
    sy = 0.0
    n = 0
    for y in range(y0, y1, stride):
        row = rgb[y]
        for x in range(x0, x1, stride):
            p = row[x]
            d = abs(int(p[0]) - ref[0]) + abs(int(p[1]) - ref[1]) + abs(int(p[2]) - ref[2])
            if d < thr:
                sx += x
                sy += y
                n += 1
    if n == 0:
        return None
    return (sx / n, sy / n, n)


def locate_cube(rgb):
    """Return (x, y, n, mode) or None. Window search around last position, refine, else global."""
    size = track["size"]
    min_n = max(3, int(0.15 * track["n0"])) if track["n0"] > 0 else 3
    if track["x"] is not None:
        coarse = find_cube(rgb, track["x"], track["y"], 2.5 * size + 6, 1)
        if coarse is not None and coarse[2] >= min_n:
            fine = find_cube(rgb, coarse[0], coarse[1], 1.0 * size + 2, 1)
            if fine is not None and fine[2] >= min_n:
                return (fine[0], fine[1], fine[2], "win")
            return (coarse[0], coarse[1], coarse[2], "coarse")
    # global search (stride 2) then refine at the densest match
    ref = track["ref"]
    thr = track["thr"]
    best = None
    cells = {}
    cell = max(6, int(size))
    for y in range(0, H, 2):
        row = rgb[y]
        for x in range(0, W, 2):
            p = row[x]
            d = abs(int(p[0]) - ref[0]) + abs(int(p[1]) - ref[1]) + abs(int(p[2]) - ref[2])
            if d < thr:
                key = (x // cell) * 1000 + (y // cell)
                cells[key] = cells.get(key, 0) + 1
    for key in cells:
        if best is None or cells[key] > cells[best]:
            best = key
    if best is None:
        return None
    gx = (best // 1000) * cell + cell / 2.0
    gy = (best % 1000) * cell + cell / 2.0
    fine = find_cube(rgb, gx, gy, 1.5 * size + 2, 1)
    if fine is None or fine[2] < min_n:
        return None
    return (fine[0], fine[1], fine[2], "global")


try:
    if track["ok"]:
        first = find_cube(obs0.cameras[front].rgb, track["x"], track["y"], track["size"] + 2, 1)
        if first is not None:
            track["n0"] = first[2]
            track["x"] = first[0]
            track["y"] = first[1]
        else:
            track["ok"] = False
            note("cube colour not found inside its own SAM box")
except Exception as exc:
    note_error("tracker first", exc)
    track["ok"] = False

result["tracker_init"] = {
    "ok": track["ok"],
    "ref": None if track["ref"] is None else list(track["ref"]),
    "bg": None if track.get("bg") is None else list(track["bg"]),
    "thr": track["thr"],
    "size": r3(track["size"]),
    "n0": track["n0"],
    "x": r3(track["x"]),
    "y": r3(track["y"]),
}

# ---------------------------------------------------------------- button 3D
button_xyz = None
button_top = None
try:
    if button is not None:
        cloud = mask_to_point_cloud(button["cand"], front, target_frame=base_frame)
        pts = cloud.points
        zs = sorted(float(p[2]) for p in pts)
        result["button_points"] = len(zs)
        if len(zs) >= 5:
            z_hi = zs[int((len(zs) - 1) * 0.9)]
            z_lo = zs[int((len(zs) - 1) * 0.1)]
            upper = [p for p in pts if float(p[2]) >= z_lo + 0.5 * (z_hi - z_lo)]
            if len(upper) < 3:
                upper = [p for p in pts]
            bx = median([float(p[0]) for p in upper])
            by = median([float(p[1]) for p in upper])
            button_top = z_hi
            button_xyz = (bx, by, z_hi)
            result["button_front"] = {"x": r3(bx), "y": r3(by), "z_top_q90": r3(z_hi), "z_q10": r3(z_lo)}
except Exception as exc:
    note_error("button cloud", exc)

# ---------------------------------------------------------------- per-tick machinery
TICK_LOG_LIMIT = 1200
log = {
    "t": [], "cx": [], "cy": [], "d": [], "n": [], "mode": [], "tcp_z": [], "phase": [],
}
hist = {"x": [], "y": [], "ok": []}
status = {
    "tick": 0,
    "terminal": False,
    "terminal_info": None,
    "count": 0,
    "in_zone": False,
    "zone_best_d": None,
    "zone_best_t": None,
    "reaches": [],
    "reach_min_d": [],
    "near_misses": [],
    "pressed": False,
    "press_tick": None,
    "lost": 0,
    "step_errors": 0,
    "last_obs": obs0,
}
reach = {"enter": None, "exit": None}
reach["enter"] = max(4.0, min(8.0, 0.55 * track["size"]))
reach["exit"] = reach["enter"] + 2.0
result["reach_thresholds_px"] = [r3(reach["enter"]), r3(reach["exit"])]


def tcp_z_now():
    try:
        st = get_robot_state()
        return float(st.end_effector_poses["primary"].position[2])
    except Exception:
        return None


def refine_reach_time(i):
    """Fractional tick of closest approach around history index i (pixel path vs target)."""
    best_t = float(i)
    best_d = None
    for j in (i - 1, i):
        if j < 0 or j + 1 >= len(hist["x"]):
            continue
        if not (hist["ok"][j] and hist["ok"][j + 1]):
            continue
        ax = hist["x"][j]
        ay = hist["y"][j]
        vx = hist["x"][j + 1] - ax
        vy = hist["y"][j + 1] - ay
        vv = vx * vx + vy * vy
        if vv <= 1e-6:
            continue
        s = ((target_px[0] - ax) * vx + (target_px[1] - ay) * vy) / vv
        s = max(0.0, min(1.0, s))
        px_ = ax + s * vx
        py_ = ay + s * vy
        dd = math.sqrt((px_ - target_px[0]) ** 2 + (py_ - target_px[1]) ** 2)
        if best_d is None or dd < best_d:
            best_d = dd
            best_t = j + s
    return best_t


def observe(obs):
    """Track the cube in this tick's front image, update the reach counter."""
    t = status["tick"]
    cx = None
    cy = None
    n = 0
    mode = "none"
    if track["ok"] and obs is not None:
        try:
            found = locate_cube(obs.cameras[front].rgb)
            if found is not None:
                cx, cy, n, mode = found
                track["x"] = cx
                track["y"] = cy
        except Exception as exc:
            note_error("locate_cube t=" + str(t), exc)
            if not status.get("use_get_obs"):
                # the step reply may not carry camera arrays: switch to get_observation()
                status["use_get_obs"] = True
                note("switching to get_observation() for per-tick images at t=" + str(t))
                try:
                    found = locate_cube(get_observation().cameras[front].rgb)
                    if found is not None:
                        cx, cy, n, mode = found
                        track["x"] = cx
                        track["y"] = cy
                except Exception as exc2:
                    note_error("locate_cube retry t=" + str(t), exc2)
    ok = cx is not None
    hist["x"].append(cx if ok else 0.0)
    hist["y"].append(cy if ok else 0.0)
    hist["ok"].append(ok)
    d = None
    if ok and target_px is not None:
        d = math.sqrt((cx - target_px[0]) ** 2 + (cy - target_px[1]) ** 2)
    if not ok:
        status["lost"] += 1
    if len(log["t"]) < TICK_LOG_LIMIT:
        log["t"].append(t)
        log["cx"].append(None if cx is None else round(cx, 1))
        log["cy"].append(None if cy is None else round(cy, 1))
        log["d"].append(None if d is None else round(d, 1))
        log["n"].append(n)
        log["mode"].append(mode)
    if d is None:
        return
    if not status["in_zone"]:
        if d < reach["enter"]:
            status["in_zone"] = True
            status["zone_best_d"] = d
            status["zone_best_t"] = t
            status["zone_from_start"] = (t == 0)
    else:
        if d < status["zone_best_d"]:
            status["zone_best_d"] = d
            status["zone_best_t"] = t
        if d > reach["exit"] and status.get("zone_from_start"):
            # cube was already at the target in the first observation: not an arrival
            status["in_zone"] = False
            status["zone_from_start"] = False
            note("cube started inside the target zone; that exit is not counted (t=" + str(t) + ")")
        elif d > reach["exit"]:
            status["in_zone"] = False
            status["count"] += 1
            tr = refine_reach_time(status["zone_best_t"])
            status["reaches"].append(round(tr, 2))
            status["reach_min_d"].append(round(status["zone_best_d"], 2))


def template_tau(tr, tr_prev, i, px_, py_, vx, vy, speed):
    """Time-to-reach from the approach that preceded the earlier reach at fractional tick tr.
    Only samples after the reach before it (tr_prev) are used, so a match can never come
    from an older cycle (which would add a whole period to the estimate)."""
    best = None
    k_end = int(math.floor(tr))
    k_start = max(1, int(math.floor(tr_prev)) + 1, k_end - 80)
    for s in range(k_start, min(k_end + 1, i - 2)):
        if not (hist["ok"][s] and hist["ok"][s - 1] and hist["ok"][s + 1]):
            continue
        sx = hist["x"][s]
        sy = hist["y"][s]
        svx = hist["x"][s + 1] - sx
        svy = hist["y"][s + 1] - sy
        sspeed = math.sqrt(svx * svx + svy * svy)
        if sspeed < 0.2:
            continue
        if (svx * vx + svy * vy) < 0.7 * sspeed * speed:
            continue
        dist = math.sqrt((sx - px_) ** 2 + (sy - py_) ** 2)
        if best is None or dist < best[0]:
            proj = ((px_ - sx) * svx + (py_ - sy) * svy) / (sspeed * sspeed)
            proj = max(-1.0, min(1.0, proj))
            best = (dist, tr - s - proj)
    return best


def period_eta():
    """Absolute tick of the next reach from the median interval of earlier reaches, or None."""
    reaches = status["reaches"]
    if len(reaches) < 2:
        return None
    gaps = []
    for k in range(1, len(reaches)):
        gaps.append(reaches[k] - reaches[k - 1])
    gap = median(gaps)
    eta = reaches[-1] + gap
    for guard_i in range(50):
        if eta >= status["tick"] - 1.0:
            break
        eta += gap
    return eta


def predict_tau():
    """Predicted ticks from the current tick to the next reach. Returns (tau, method, quality) or None.
    Methods: template (median over the last 3 reaches' approaches), cpa (straight line), period."""
    i = len(hist["x"]) - 1
    t = status["tick"]
    if status.get("pred_tick") == t:
        return status.get("pred_value")
    status["pred_tick"] = t
    status["pred_value"] = None
    raw = None
    if i >= 1 and hist["ok"][i] and hist["ok"][i - 1] and target_px is not None:
        px_ = hist["x"][i]
        py_ = hist["y"][i]
        vx = px_ - hist["x"][i - 1]
        vy = py_ - hist["y"][i - 1]
        speed = math.sqrt(vx * vx + vy * vy)
        if speed >= 0.2:
            taus = []
            quals = []
            n_r = len(status["reaches"])
            for idx in range(max(0, n_r - 3), n_r):
                tr = status["reaches"][idx]
                tr_prev = status["reaches"][idx - 1] if idx >= 1 else -1.0
                best = template_tau(tr, tr_prev, i, px_, py_, vx, vy, speed)
                if best is not None and best[0] < 0.5 * track["size"] + 1.0 and best[1] > -1.0:
                    taus.append(best[1])
                    quals.append(best[0])
            if taus:
                raw = (median(taus), "template", round(min(quals), 2))
            else:
                rx = target_px[0] - px_
                ry = target_px[1] - py_
                tau = (rx * vx + ry * vy) / (speed * speed)
                miss = abs(rx * vy - ry * vx) / speed
                if tau >= -1.0 and miss <= reach["enter"]:
                    raw = (tau, "cpa", round(miss, 2))
    if raw is None:
        eta = period_eta()
        if eta is not None and len(status["reaches"]) >= 2:
            raw = (eta - t, "period", None)
    if raw is None:
        return None
    # smooth: median of the absolute arrival estimates of the last 3 ticks (same method family)
    arrivals = status.setdefault("arrivals", [])
    arrivals.append((t, t + raw[0], raw[1]))
    recent = []
    for item in arrivals[-3:]:
        if t - item[0] <= 2 and item[2] == raw[1]:
            recent.append(item[1])
    if len(recent) >= 2:
        tau_s = median(recent) - t
    else:
        tau_s = raw[0]
    status["pred_value"] = (tau_s, raw[1], raw[2])
    return status["pred_value"]


def do_step(q, grip, phase):
    """One control tick holding/commanding joints q. Returns False once terminal/failed."""
    if status["terminal"]:
        return False
    if status["tick"] >= max_steps + 5:
        status["terminal"] = True
        status["terminal_info"] = "local max_steps guard"
        return False
    try:
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=[float(v) for v in q],
            gripper_position=grip, embodiment=embodiment)}))
    except Exception as exc:
        note_error("step t=" + str(status["tick"]), exc)
        status["step_errors"] += 1
        status["terminal"] = True
        status["terminal_info"] = "step raised: " + str(exc)[:200]
        return False
    status["tick"] += 1
    term = False
    trunc = False
    try:
        term = bool(reply.terminated)
    except Exception:
        term = False
    try:
        trunc = bool(reply.truncated)
    except Exception:
        trunc = False
    obs = None
    if not status.get("use_get_obs"):
        try:
            obs = reply.observation
        except Exception:
            obs = None
    if status["tick"] <= 2:
        try:
            result["step_reply_ok_first"] = str(reply.ok)
        except Exception:
            result["step_reply_ok_first"] = "n/a"
        if status["tick"] == 1:
            try:
                result["step_reply_repr"] = str(reply)[:600]
            except Exception:
                result["step_reply_repr"] = "n/a"
    try:
        err = reply.error
        if err is not None:
            note("step error t=" + str(status["tick"]) + ": " + str(err)[:200])
    except Exception:
        pass
    z = None
    try:
        z = float(obs.robot_state.end_effector_poses["primary"].position[2])
    except Exception:
        z = None
    if len(log["tcp_z"]) < TICK_LOG_LIMIT:
        log["tcp_z"].append(None if z is None else round(z, 4))
        log["phase"].append(phase)
    if term or trunc:
        status["terminal"] = True
        status["terminal_info"] = {"tick": status["tick"], "terminated": term, "truncated": trunc, "phase": phase}
        return False
    if obs is None:
        try:
            obs = get_observation()
        except Exception as exc:
            note_error("get_observation in loop", exc)
            obs = None
    status["last_obs"] = obs
    status["last_z"] = z
    observe(obs)
    return True


def current_joints():
    try:
        return [float(v) for v in get_robot_state().joint_positions["primary"]]
    except Exception as exc:
        note_error("current_joints", exc)
        return None


def interp_path(q_from, q_to, max_delta):
    biggest = 0.0
    for a, b in zip(q_from, q_to):
        biggest = max(biggest, abs(b - a))
    n = max(1, int(math.ceil(biggest / max_delta)))
    path = []
    for k in range(1, n + 1):
        f = float(k) / n
        path.append([a + f * (b - a) for a, b in zip(q_from, q_to)])
    return path


def ik(position, quat, seed):
    try:
        res = solve_ik(Pose(position=[float(v) for v in position], quaternion_wxyz=[float(v) for v in quat], frame=base_frame), seed_joints=seed)
        if res.ok:
            return [float(v) for v in res.joint_positions]
        note("ik not ok at " + str([r3(v) for v in position]) + ": " + str(res.error)[:200])
    except Exception as exc:
        note_error("solve_ik " + str([r3(v) for v in position]), exc)
    return None


GRIP = 0.0  # closed fingers act as the pressing tool (expert recordings press with a closed gripper)

# observe tick 0 (initial observation)
observe(obs0)

q_hold = None
if state0 is not None:
    q_hold = [float(v) for v in state0.joint_positions["primary"]]
quat = None
try:
    quat = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]
except Exception as exc:
    note_error("initial quaternion", exc)

# ---------------------------------------------------------------- plan approach (no ticks pass)
HOVER_ABOVE = 0.03      # tool point above perceived button top while waiting
PRE_ABOVE = 0.12        # pre-hover height above the button top
PRESS_RATE = 0.01       # metres of commanded descent per tick during the press
PRESS_BELOW = 0.03      # commanded tool depth below the perceived top at the end of the press
TIP_EXTRA = 0.01        # assumed fingertip extension below the tool point (hand + 0.1 m)
BUTTON_TRAVEL = 0.006   # assumed travel needed to register a press
REHEARSE_DEPTH = 0.012  # rehearsal dip: stays HOVER_ABOVE - REHEARSE_DEPTH above the top
LEAD = 1.5              # press registers ~[0.5, 1.5) ticks before the predicted closest approach
JOINT_STEP = 0.05       # rad per tick for transit

plan = {"pre": None, "hover": None, "press": [], "ok": False, "quat": None}
result["phase"] = "plan"
IK_SEG = 0.08           # ladder segment length (m) when a direct IK solve fails
IK_ROUNDS = 4           # executed ladder rounds before giving up on a pose
JOINT1_LIMIT = 2.7      # stay inside the Panda joint-1 range when yaw-aligning
ik_log = []


def ee_now():
    try:
        st = get_robot_state()
        p = st.end_effector_poses["primary"]
        return [float(v) for v in p.position], [float(v) for v in p.quaternion_wxyz]
    except Exception as exc:
        note_error("ee_now", exc)
        return None, None


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx_, by_, bz_ = b
    return [aw * bw - ax * bx_ - ay * by_ - az * bz_,
            aw * bx_ + ax * bw + ay * bz_ - az * by_,
            aw * by_ - ax * bz_ + ay * bw + az * bx_,
            aw * bz_ + ax * by_ - ay * bx_ + az * bw]


def yaw_quat(theta, q):
    # rotate orientation q about the base z axis by theta (closed-gripper press is yaw-agnostic)
    return quat_mul([math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)], q)


def ik_try(position, quats, seeds, tag):
    """First successful (joints, quat) over orientation x seed variants; every attempt is logged."""
    for qi in range(len(quats)):
        for si in range(len(seeds)):
            if quats[qi] is None or seeds[si] is None:
                continue
            entry = {"tag": tag, "pos": [r3(v) for v in position], "quat_i": qi, "seed_i": si, "ok": False}
            try:
                res = solve_ik(Pose(position=[float(v) for v in position], quaternion_wxyz=[float(v) for v in quats[qi]], frame=base_frame), seed_joints=[float(v) for v in seeds[si]])
                if res.ok:
                    entry["ok"] = True
                    if len(ik_log) < 120:
                        ik_log.append(entry)
                    return [float(v) for v in res.joint_positions], quats[qi]
                try:
                    entry["err"] = str(res.error.code) + " " + str(res.error.details)[:160]
                except Exception:
                    entry["err"] = str(res.error)[:200]
            except Exception as exc:
                entry["err"] = "raised " + str(exc)[:160]
            if len(ik_log) < 120:
                ik_log.append(entry)
    return None, None


def drive(q_target, max_delta, phase):
    """Joint interpolation from the measured joints to q_target, one tick per waypoint."""
    q_from = current_joints()
    if q_from is None:
        q_from = q_target
    for q in interp_path(q_from, q_target, max_delta):
        if not do_step(q, GRIP, phase):
            return False
    return True


def reach_pose(position, quats, phase, max_delta):
    """Direct IK; else an IK ladder from the current tool point, executing every solved
    prefix and re-solving from the new configuration.  Returns (joints, quat, ended)."""
    for round_i in range(IK_ROUNDS):
        q_now = current_joints()
        sol, sq = ik_try(position, quats, [q_now], phase + "/direct" + str(round_i))
        if sol is not None:
            if not drive(sol, max_delta, phase):
                return sol, sq, True
            return sol, sq, False
        p_now, q_cur = ee_now()
        if p_now is None:
            return None, None, False
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(position, p_now)))
        n_seg = max(2, int(math.ceil(dist / IK_SEG)))
        seed = q_now
        solved = []
        use_quats = [q_cur] + quats if round_i % 2 == 0 else quats + [q_cur]
        for k in range(1, n_seg):
            f = float(k) / n_seg
            wp = [a + f * (b - a) for a, b in zip(p_now, position)]
            s, sq2 = ik_try(wp, use_quats, [seed], phase + "/ladder" + str(round_i) + "." + str(k))
            if s is None:
                break
            solved.append(s)
            seed = s
        if not solved:
            note("ik ladder made no progress for " + phase + " round " + str(round_i))
            return None, None, False
        for s in solved:
            if not drive(s, max_delta, phase + "_ladder"):
                return None, None, True
    return None, None, False


press_model = {
    "lag": 1.0,
    "source": "default",
}


def press_latency():
    # commanded tool height reaches the registration height after c ticks; add measured lag.
    h = HOVER_ABOVE
    if press_model.get("hover_h") is not None:
        h = press_model["hover_h"]  # measured settled tool height above the perceived top
    c = (h - (TIP_EXTRA - BUTTON_TRAVEL)) / PRESS_RATE
    c = max(1.0, c)
    return c + press_model["lag"]


def ready_to_press_now():
    if status["pressed"] or not plan["ok"]:
        return False
    if status["count"] < target_count - 1:
        return False
    if status["in_zone"] and status["count"] == target_count - 1:
        # already inside the N-th reach zone: press immediately (late rather than never)
        return True
    pred = predict_tau()
    if pred is None:
        return False
    tau = pred[0]
    lat = press_latency()
    if status["tick"] - status.get("last_pred_log", -99) >= 1 and len(result.setdefault("predictions", [])) < 300:
        result["predictions"].append([status["tick"], round(tau, 2), pred[1], pred[2], round(lat, 2)])
        status["last_pred_log"] = status["tick"]
    return tau <= lat + LEAD


def wait_tick(q, phase):
    """Hold q for one tick unless the press should start; returns 'press', 'ok' or 'end'."""
    if ready_to_press_now() and status["arm_ready"]:
        return "press"
    if not do_step(q, GRIP, phase):
        return "end"
    return "ok"


def run_press():
    status["pressed"] = True
    status["press_tick"] = status["tick"]
    result["press_trigger"] = {
        "tick": status["tick"],
        "count": status["count"],
        "latency_model": round(press_latency(), 2),
        "lag": press_model["lag"],
        "in_zone": status["in_zone"],
        "last_pred": None if not result.get("predictions") else result["predictions"][-1],
    }
    for qk in plan["press"]:
        if not do_step(qk, GRIP, "press"):
            return
    # hold the bottom of the press for a while
    for hold in range(20):
        if not do_step(plan["press"][-1], GRIP, "press_hold"):
            return
    note("press profile finished without a terminal step")


def motion_init(rgb_a, rgb_b):
    """Fallback cube detector: pixels that changed between two frames while the arm held still."""
    info = {"ok": False}
    cell = 8
    cells = {}
    for y in range(0, H, 2):
        row_a = rgb_a[y]
        row_b = rgb_b[y]
        for x in range(0, W, 2):
            pa = row_a[x]
            pb = row_b[x]
            d = abs(int(pa[0]) - int(pb[0])) + abs(int(pa[1]) - int(pb[1])) + abs(int(pa[2]) - int(pb[2]))
            if d > 60:
                key = (x // cell) * 1000 + (y // cell)
                cells[key] = cells.get(key, 0) + 1
    info["changed_cells"] = len(cells)
    best = None
    for key in cells:
        if best is None or cells[key] > cells[best]:
            best = key
    if best is None:
        info["why"] = "no changed pixels"
        return info
    gx = (best // 1000) * cell + cell / 2.0
    gy = (best % 1000) * cell + cell / 2.0
    changed = []
    for y in range(max(0, int(gy - 16)), min(H, int(gy + 16) + 1)):
        for x in range(max(0, int(gx - 16)), min(W, int(gx + 16) + 1)):
            pa = pix(rgb_a, x, y)
            pb = pix(rgb_b, x, y)
            if cdist(pa, pb) > 60:
                changed.append((x, y, pb))
    if len(changed) < 6:
        info["why"] = "too few changed pixels"
        return info
    box = [min(c[0] for c in changed), min(c[1] for c in changed), max(c[0] for c in changed), max(c[1] for c in changed)]
    bg = ring_color(rgb_b, box, 3)
    if bg is None:
        info["why"] = "no ring"
        return info
    cube_px = [c for c in changed if cdist(c[2], bg) > 60]
    if len(cube_px) < 4:
        info["why"] = "no non-background changed pixels"
        return info
    ref = (median([c[2][0] for c in cube_px]), median([c[2][1] for c in cube_px]), median([c[2][2] for c in cube_px]))
    track["ref"] = ref
    track["bg"] = bg
    track["thr"] = max(30, min(65, int(cdist(ref, bg) * 0.45)))
    track["x"] = sum(c[0] for c in cube_px) / float(len(cube_px))
    track["y"] = sum(c[1] for c in cube_px) / float(len(cube_px))
    track["size"] = max(4.0, math.sqrt(len(cube_px)) * 1.2)
    track["n0"] = 0
    track["ok"] = True
    track["source"] = "motion"
    first = find_cube(rgb_b, track["x"], track["y"], track["size"] + 2, 1)
    if first is not None:
        track["n0"] = first[2]
        track["x"] = first[0]
        track["y"] = first[1]
    info["ok"] = True
    info["ref"] = list(ref)
    info["bg"] = list(bg)
    info["xy"] = [r3(track["x"]), r3(track["y"])]
    info["size"] = r3(track["size"])
    info["thr"] = track["thr"]
    return info


# ---------------------------------------------------------------- start-up hold + motion check
result["phase"] = "probe_hold"
PROBE_TICKS = 4
probe = {"ticks": 0, "moved_px": None, "reinit": None}
try:
    if q_hold is not None:
        for probe_i in range(PROBE_TICKS):
            if not do_step(q_hold, GRIP, "probe_hold"):
                break
            probe["ticks"] += 1
    moved = None
    if track["ok"] and hist["ok"][0]:
        for k in range(1, len(hist["ok"])):
            if hist["ok"][k]:
                dd = math.sqrt((hist["x"][k] - hist["x"][0]) ** 2 + (hist["y"][k] - hist["y"][0]) ** 2)
                moved = dd if moved is None else max(moved, dd)
    probe["moved_px"] = r3(moved)
    if (not track["ok"] or moved is None or moved < 2.0) and status["last_obs"] is not None and obs0 is not None and not status["terminal"]:
        note("cube track missing or static after the start-up hold; trying frame-difference init")
        probe["reinit"] = motion_init(obs0.cameras[front].rgb, status["last_obs"].cameras[front].rgb)
        if probe["reinit"].get("ok"):
            for k in range(len(hist["ok"])):
                hist["ok"][k] = False
            status["in_zone"] = False
            status["count"] = 0
            status["reaches"] = []
            status["reach_min_d"] = []
            reach["enter"] = max(4.0, min(8.0, 0.55 * track["size"]))
            reach["exit"] = reach["enter"] + 2.0
            result["reach_thresholds_px"] = [r3(reach["enter"]), r3(reach["exit"])]
            if len(hist["ok"]) > 0:
                hist["x"][-1] = track["x"]
                hist["y"][-1] = track["y"]
                hist["ok"][-1] = True
except Exception as exc:
    note_error("probe hold", exc)
result["probe_check"] = probe

status["arm_ready"] = False
result["phase"] = "transit"
end = False
pending_press = False
try:
    stage = {"yaw_align": None, "pre": None, "hover": None, "press_levels": 0}
    result["stage"] = stage
    if button_xyz is not None and q_hold is not None and not status["terminal"]:
        bx, by, bz = button_xyz
        # 1. yaw-align joint 1 towards the button azimuth (pure joint motion, no IK).
        #    v00 trace: a direct top-down IK from the home seed to a button at a large
        #    azimuth hit a local minimum (pos err 0.032 m, orient err 0.35 rad) -> no plan.
        p0, quat_home = ee_now()
        if quat_home is None:
            quat_home = quat
        dth = 0.0
        if p0 is not None:
            dth = math.atan2(by, bx) - math.atan2(p0[1], p0[0])
            while dth > math.pi:
                dth -= 2.0 * math.pi
            while dth < -math.pi:
                dth += 2.0 * math.pi
        q_start = current_joints() or q_hold
        if abs(dth) > 0.05:
            q_yaw = list(q_start)
            q_yaw[0] = max(-JOINT1_LIMIT, min(JOINT1_LIMIT, q_yaw[0] + dth))
            if not drive(q_yaw, JOINT_STEP, "yaw_align"):
                end = True
            for settle in range(2):
                if not end and not do_step(q_yaw, GRIP, "yaw_settle"):
                    end = True
        p1, quat_now = ee_now()
        stage["yaw_align"] = {"dtheta": r3(dth), "ee_after": None if p1 is None else [r3(v) for v in p1],
                              "quat_after": None if quat_now is None else [r3(v) for v in quat_now]}
        quats = []
        for cand in (quat_now, None if quat_home is None else yaw_quat(dth, quat_home), quat_home):
            if cand is not None:
                quats.append(cand)
        # 2. pre-hover (staged IK, executed one tick per waypoint)
        if not end and quats:
            q_pre, q_pre_quat, end = reach_pose((bx, by, bz + PRE_ABOVE), quats, "to_pre", JOINT_STEP)
            plan["pre"] = q_pre
            stage["pre"] = q_pre is not None
            if q_pre_quat is not None:
                quats = [q_pre_quat] + [v for v in quats if v is not q_pre_quat]
        # 3. hover (direct or staged from wherever the arm is, even if pre-hover failed)
        if not end and quats:
            q_hov, q_hov_quat, end = reach_pose((bx, by, bz + HOVER_ABOVE), quats, "to_hover", 0.03)
            plan["hover"] = q_hov
            stage["hover"] = q_hov is not None
            if q_hov_quat is not None:
                plan["quat"] = q_hov_quat
        # 4. press levels, solved from the hover configuration (small moves)
        if not end and plan["hover"] is not None:
            seed = plan["hover"]
            n_press = int(math.ceil((HOVER_ABOVE + PRESS_BELOW) / PRESS_RATE))
            for k in range(1, n_press + 1):
                zk = bz + HOVER_ABOVE - k * PRESS_RATE
                qk, qq = ik_try((bx, by, zk), [plan["quat"]] + [v for v in quats if v is not plan["quat"]], [seed, plan["hover"]], "press" + str(k))
                if qk is None:
                    break
                plan["press"].append(qk)
                seed = qk
            stage["press_levels"] = len(plan["press"])
            plan["ok"] = len(plan["press"]) >= 3
    result["ik_attempts"] = ik_log
    result["plan"] = {"pre_ok": plan["pre"] is not None, "hover_ok": plan["hover"] is not None,
                      "press_levels": len(plan["press"]), "ok": plan["ok"]}
    if plan["ok"] and not end:
        # settle at hover until the measured tool height is stationary (<=0.5 mm/tick)
        settle_n = 0
        if not end:
            z_prev = status.get("last_z")
            for settle in range(20):
                if not do_step(plan["hover"], GRIP, "settle"):
                    end = True
                    break
                settle_n += 1
                z_cur = status.get("last_z")
                if settle_n >= 3 and z_cur is not None and z_prev is not None and abs(z_cur - z_prev) < 0.0005:
                    break
                z_prev = z_cur
        result["settle_ticks"] = settle_n
        # one closed-loop correction: IK is accepted within 2 cm / 0.15 rad residual, so
        # compare the measured tool point with the commanded hover point and re-aim once.
        try:
            p_meas, q_meas = ee_now()
            if not end and p_meas is not None and button_xyz is not None:
                hx, hy, hz = button_xyz[0], button_xyz[1], button_xyz[2] + HOVER_ABOVE
                ex, ey, ez = hx - p_meas[0], hy - p_meas[1], hz - p_meas[2]
                corr = {"err": [r3(ex), r3(ey), r3(ez)], "applied": False}
                if math.sqrt(ex * ex + ey * ey) > 0.004 or abs(ez) > 0.006:
                    qa = [plan["quat"]] if plan["quat"] is not None else []
                    q_new, qq = ik_try((hx + ex, hy + ey, hz + ez), qa + [q_meas], [plan["hover"]], "hover_corr")
                    if q_new is not None:
                        levels = []
                        seed = q_new
                        for k in range(1, len(plan["press"]) + 1):
                            qk, qq2 = ik_try((hx + ex, hy + ey, hz + ez - k * PRESS_RATE), [qq], [seed, q_new], "press_corr" + str(k))
                            if qk is None:
                                break
                            levels.append(qk)
                            seed = qk
                        if len(levels) >= 3:
                            plan["hover"] = q_new
                            plan["press"] = levels
                            corr["applied"] = True
                            if not drive(q_new, 0.02, "hover_corr"):
                                end = True
                            n2 = 0
                            z_prev = status.get("last_z")
                            while not end and n2 < 12:
                                if not do_step(plan["hover"], GRIP, "settle2"):
                                    end = True
                                    break
                                n2 += 1
                                z_cur = status.get("last_z")
                                if n2 >= 3 and z_cur is not None and z_prev is not None and abs(z_cur - z_prev) < 0.0005:
                                    break
                                z_prev = z_cur
                            p2, q2 = ee_now()
                            if p2 is not None:
                                corr["err_after"] = [r3(hx - p2[0]), r3(hy - p2[1]), r3(hz - p2[2])]
                result["hover_correction"] = corr
        except Exception as exc:
            note_error("hover correction", exc)
        try:
            if status.get("last_z") is not None and button_top is not None:
                hh = status["last_z"] - button_top
                result["hover_height_measured"] = r3(hh)
                if 0.005 <= hh <= 0.08:
                    press_model["hover_h"] = hh
                else:
                    note("measured hover height " + str(r3(hh)) + " outside [0.005, 0.08]; latency uses HOVER_ABOVE")
        except Exception as exc:
            note_error("hover height", exc)
        hover_z = status.get("last_z")
        result["hover_measured_z"] = r3(hover_z)
        if button_top is not None:
            result["hover_target_z"] = r3(button_top + HOVER_ABOVE)
        # wrist cross-check of the button height (diagnostic only)
        if not end:
            try:
                wseg = segment_text(wrist, "button")
                if wseg.ok and len(wseg.segmentations) > 0:
                    wc = wseg.segmentations[0]
                    wcloud = mask_to_point_cloud(wc, wrist, target_frame=base_frame)
                    wz = sorted(float(p[2]) for p in wcloud.points)
                    if len(wz) >= 5:
                        result["button_wrist"] = {
                            "points": len(wz),
                            "z_top_q90": r3(wz[int((len(wz) - 1) * 0.9)]),
                            "z_q50": r3(wz[int((len(wz) - 1) * 0.5)]),
                            "x": r3(median([float(p[0]) for p in wcloud.points])),
                            "y": r3(median([float(p[1]) for p in wcloud.points])),
                            "score": r3(wc.score),
                        }
            except Exception as exc:
                note_error("wrist button check", exc)
        # 4. rehearsal dip that stays above the button (measures controller lag); aborted
        #    whenever the N-th reach could come within the rehearsal's duration.
        if not end:
            n_r = max(1, int(round(REHEARSE_DEPTH / PRESS_RATE)))
            if len(plan["press"]) > n_r:
                z_start = status.get("last_z")
                rz = []
                dip = plan["press"][:n_r] + [plan["press"][n_r - 1]] * 5
                back = interp_path(plan["press"][n_r - 1], plan["hover"], 0.02) + [plan["hover"]] * 3
                aborted = False
                for q in dip + back:
                    if status["count"] >= target_count - 1 and not aborted:
                        pred = predict_tau()
                        if pred is None or pred[0] < press_latency() + LEAD + 10:
                            if pred is not None or status["in_zone"]:
                                aborted = True
                                note("rehearsal aborted at tick " + str(status["tick"]))
                                break
                    if not do_step(q, GRIP, "rehearse"):
                        end = True
                        break
                    rz.append(status.get("last_z"))
                if aborted and not end:
                    for q in interp_path(current_joints() or plan["hover"], plan["hover"], 0.02):
                        if not do_step(q, GRIP, "rehearse_abort"):
                            end = True
                            break
                result["rehearsal"] = {"z_start": r3(z_start), "z": [r3(v) for v in rz], "n_cmd": n_r, "aborted": aborted}
                try:
                    if z_start is not None and len(rz) >= 3 and not aborted:
                        depth_cmd = n_r * PRESS_RATE
                        vals0 = [v for v in rz if v is not None]
                        depth_ref = depth_cmd
                        if vals0 and z_start - min(vals0) > 0.003:
                            depth_ref = min(depth_cmd * 1.5, z_start - min(vals0))
                        half = z_start - depth_ref / 2.0
                        t_meas = None
                        prev = z_start
                        for k in range(len(rz)):
                            if rz[k] is None:
                                break
                            if rz[k] <= half:
                                frac = 1.0
                                if prev != rz[k]:
                                    frac = (prev - half) / (prev - rz[k])
                                t_meas = k + frac
                                break
                            prev = rz[k]
                        vals = [v for v in rz if v is not None]
                        if vals:
                            result["rehearsal"]["min_z"] = r3(min(vals))
                            result["rehearsal"]["depth_measured"] = r3(z_start - min(vals))
                        if t_meas is not None:
                            lag = t_meas - n_r / 2.0
                            press_model["lag"] = max(0.0, min(4.0, lag))
                            press_model["source"] = "rehearsal"
                        result["rehearsal"]["t_half_measured"] = r3(t_meas)
                except Exception as exc:
                    note_error("rehearsal lag", exc)
            else:
                note("rehearsal skipped (short press plan)")
        status["arm_ready"] = not end
        result["arm_ready_tick"] = status["tick"]
        result["press_model"] = {"lag": r3(press_model["lag"]), "source": press_model["source"], "hover_h": r3(press_model.get("hover_h")), "latency": r3(press_latency())}
        # 5. wait at hover, observing every tick, until the press trigger fires
        result["phase"] = "wait"
        if not end:
            for wait_i in range(max_steps + 10):
                outcome = wait_tick(plan["hover"], "wait")
                if outcome == "press":
                    result["phase"] = "press"
                    run_press()
                    break
                if outcome == "end":
                    end = True
                    break
    else:
        note("no press plan (button/IK missing); holding position and tracking only")
        result["phase"] = "fallback_hold"
        q_fb = current_joints() or q_hold
        if q_fb is not None and not end and not status["terminal"]:
            for hold_i in range(max_steps + 10):
                if not do_step(q_fb, GRIP, "fallback_hold"):
                    break
except Exception as exc:
    note_error("control", exc)

# ---------------------------------------------------------------- report
result["phase_final"] = result.get("phase")
result["ticks"] = status["tick"]
result["terminal"] = status["terminal"]
result["terminal_info"] = status["terminal_info"]
result["reach_count"] = status["count"]
result["reach_times"] = status["reaches"]
result["reach_min_d"] = status["reach_min_d"]
result["lost_ticks"] = status["lost"]
result["pressed"] = status["pressed"]
result["press_tick"] = status["press_tick"]
result["tick_log"] = log
if len(status["reaches"]) >= 2:
    gaps = []
    for k in range(1, len(status["reaches"])):
        gaps.append(round(status["reaches"][k] - status["reaches"][k - 1], 2))
    result["reach_intervals"] = gaps
