# v00_initial -- PickXtimes (robomme_plus task 7), AfI, written before any scene was seen.
#
# Start 1 is a COMBINED snapshot + attempt (budget: 3 starts total for this task):
#   * records task language, task/controller metadata, cameras, conditioning-video summary,
#     and every perception candidate it considered (prompt, score, box, 3D stats, colour);
#   * then attempts the task as parsed from get_task_context().language:
#       pick the named cube, (place it on the named target / back where it was), repeat
#       N times, then press the button if the language mentions one.
# Everything task-dependent (count, colour, target, cube/target/button positions, cube yaw,
# heights) is derived at runtime from language + front/wrist RGB-D. Only generic relative
# offsets/tolerances are constants.
import math

report = {"version": "v00_initial", "phases": [], "errors": [], "probes": {}}
FLAGS = {"terminal": False, "terminal_info": None}

# ---------------------------------------------------------------- context / metadata
ctx = get_task_context()
LANG = str(ctx.language)
report["task_language"] = LANG
try:
    report["task_name"] = str(ctx.task_name)
except Exception:
    report["task_name"] = "unknown"

META = {}
try:
    META = robomme_plus.get_task_metadata()
    report["task_metadata"] = {str(k): str(META[k]) for k in META}
except Exception as exc:
    report["errors"].append("task_metadata: " + str(exc))
try:
    cmeta = robomme_plus.get_controller_metadata()
    report["controller_metadata"] = {str(k): str(cmeta[k]) for k in cmeta}
except Exception as exc:
    report["errors"].append("controller_metadata: " + str(exc))

CONTROL_HZ = 20.0
for key in ("control_hz", "control_frequency", "frequency_hz", "control_freq"):
    try:
        CONTROL_HZ = float(META[key])
        break
    except Exception:
        pass
MAX_STEPS = -1
try:
    MAX_STEPS = int(META["max_steps"])
except Exception:
    pass
report["control_hz"] = CONTROL_HZ
report["max_steps"] = MAX_STEPS

try:
    video = robomme_plus.get_conditioning_video()
    nvid = len(video)
    vinfo = {"frames": nvid}
    if nvid > 0:
        vinfo["first_t"] = float(video[0].timestamp_s)
        vinfo["last_t"] = float(video[nvid - 1].timestamp_s)
    report["conditioning_video"] = vinfo
except Exception as exc:
    report["errors"].append("conditioning_video: " + str(exc))

STATE0 = get_robot_state()
BASE = STATE0.base_frame
EMB = STATE0.embodiment
HOME_EE = STATE0.end_effector_poses["primary"]
report["base_frame"] = str(BASE)
report["embodiment"] = str(EMB)
report["home_ee"] = {"pos": [round(float(v), 4) for v in HOME_EE.position],
                     "quat": [round(float(v), 4) for v in HOME_EE.quaternion_wxyz]}
report["home_gripper"] = float(STATE0.gripper_positions["primary"])
obs0 = get_observation()
report["cameras"] = [str(name) for name in obs0.cameras]
FRONT = "front" if "front" in obs0.cameras else report["cameras"][0]
WRIST = "wrist" if "wrist" in obs0.cameras else FRONT
try:
    cp = obs0.cameras[FRONT].camera_pose
    report["front_camera_pos"] = [round(float(v), 3) for v in cp.position]
except Exception:
    pass


def now_tick():
    try:
        return int(round(float(get_robot_state().timestamp_s) * CONTROL_HZ))
    except Exception:
        return -1


report["tick_start"] = now_tick()

# ---------------------------------------------------------------- language parsing
COLOR_WORDS = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "cyan",
               "white", "black", "brown", "gray", "grey", "magenta", "violet"]
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10}
cleaned = ""
for ch in LANG.lower():
    cleaned += ch if (ch.isalnum() or ch == " ") else " "
TOKENS = [tok for tok in cleaned.split(" ") if tok]

cube_color = None
for i, tok in enumerate(TOKENS):
    if tok in ("cube", "cubes", "block", "blocks"):
        for j in range(max(0, i - 3), i):
            if TOKENS[j] in COLOR_WORDS:
                cube_color = TOKENS[j]
        if cube_color is not None:
            break
if cube_color is None:
    for tok in TOKENS:
        if tok in COLOR_WORDS:
            cube_color = tok
            break

count = None
for i, tok in enumerate(TOKENS):
    if tok == "once":
        count = 1
    elif tok == "twice":
        count = 2
    elif tok == "thrice":
        count = 3
    elif tok in ("time", "times") and i > 0:
        prev = TOKENS[i - 1]
        if prev.isdigit():
            count = int(prev)
        elif prev in NUMBER_WORDS:
            count = NUMBER_WORDS[prev]
    if count is not None:
        break
count_source = "language"
if count is None:
    count = 1
    count_source = "default"

TARGET_WORDS = ["target", "goal", "plate", "disc", "disk", "marker", "pad", "region", "zone",
                "area", "spot", "mat"]
target_word = None
for tok in TOKENS:
    if tok in TARGET_WORDS:
        target_word = tok
        break
wants_button = "button" in TOKENS
report["parsed"] = {"tokens": TOKENS, "cube_color": cube_color, "count": count,
                    "count_source": count_source, "target_word": target_word,
                    "button": wants_button}

# ---------------------------------------------------------------- small math helpers


def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[int((len(ordered) - 1) * fraction)]


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def qrot(q, v):
    w, x, y, z = q
    vx, vy, vz = v
    # R(q) v
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    return (r00 * vx + r01 * vy + r02 * vz,
            r10 * vx + r11 * vy + r12 * vz,
            r20 * vx + r21 * vy + r22 * vz)


def qz(theta):
    return (math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0))


Q_HOME = tuple(float(v) for v in HOME_EE.quaternion_wxyz)
approach_axis = qrot(Q_HOME, (0.0, 0.0, 1.0))
report["home_approach_axis"] = [round(v, 3) for v in approach_axis]
if approach_axis[2] < -0.95:
    Q_DOWN = Q_HOME
    report["down_quat_source"] = "home"
else:
    Q_DOWN = (0.0, 1.0, 0.0, 0.0)
    report["down_quat_source"] = "constructed_180_about_x"
finger_axis = qrot(Q_DOWN, (0.0, 1.0, 0.0))
FINGER_YAW = math.atan2(finger_axis[1], finger_axis[0])


def grasp_quat(cube_yaw):
    delta = cube_yaw - FINGER_YAW
    delta = ((delta + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0
    return qmul(qz(delta), Q_DOWN), delta


def hue_sat_val(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn
    if mx <= 1e-6:
        return 0.0, 0.0, 0.0
    s = d / mx
    if d < 1e-6:
        h = 0.0
    elif mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * ((b - r) / d + 2.0)
    else:
        h = 60.0 * ((r - g) / d + 4.0)
    return h, s, mx


HUES = {"red": 0.0, "orange": 30.0, "yellow": 58.0, "green": 120.0, "cyan": 185.0,
        "blue": 225.0, "purple": 275.0, "violet": 275.0, "magenta": 305.0, "pink": 330.0,
        "brown": 25.0}


def color_match(rgb, color):
    """Higher is better; None when no colour information."""
    if rgb is None or color is None:
        return None
    h, s, v = hue_sat_val(rgb[0], rgb[1], rgb[2])
    if color in ("white",):
        return (1.0 - s) + v - 1.0
    if color in ("black",):
        return 1.0 - v - 0.5 * s
    if color in ("gray", "grey"):
        return 1.0 - 2.0 * s - abs(v - 0.5)
    target_h = HUES[color]
    dh = abs(h - target_h) % 360.0
    if dh > 180.0:
        dh = 360.0 - dh
    return (1.0 - dh / 90.0) + 0.5 * s - (0.5 if s < 0.2 else 0.0)


# ---------------------------------------------------------------- perception helpers


def cloud_of(cand, camera):
    pc = None
    try:
        pc = mask_to_point_cloud(cand.mask, camera, target_frame=BASE)
    except Exception:
        pc = None
    if pc is None:
        try:
            pc = mask_to_point_cloud(cand, camera, target_frame=BASE)
        except Exception:
            pc = None
    return pc


def cloud_stats(pc):
    try:
        raw = pc.points
        n = len(raw)
    except Exception:
        return None
    if n < 5:
        return {"n": n}
    stride = max(1, n // 600)
    pts = []
    idx = 0
    while idx < n:
        p = raw[idx]
        pts.append((float(p[0]), float(p[1]), float(p[2])))
        idx += stride
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    st = {"n": n, "pts": pts,
          "x": [quantile(xs, 0.05), quantile(xs, 0.5), quantile(xs, 0.95)],
          "y": [quantile(ys, 0.05), quantile(ys, 0.5), quantile(ys, 0.95)],
          "z": [quantile(zs, 0.03), quantile(zs, 0.5), quantile(zs, 0.95)]}
    rgb = None
    try:
        cols = pc.colors
        if cols is not None and len(cols) == n:
            rs = []
            gs = []
            bs = []
            idx = 0
            while idx < n:
                c = cols[idx]
                rs.append(float(c[0]))
                gs.append(float(c[1]))
                bs.append(float(c[2]))
                idx += stride
            scale = 255.0 if max(max(rs), max(gs), max(bs)) > 1.5 else 1.0
            rgb = (quantile(rs, 0.5) / scale, quantile(gs, 0.5) / scale, quantile(bs, 0.5) / scale)
    except Exception:
        rgb = None
    st["rgb"] = rgb
    return st


def public_stats(st):
    if st is None:
        return None
    out = {"n": st["n"]}
    for key in ("x", "y", "z"):
        if key in st:
            out[key] = [round(v, 4) for v in st[key]]
    if st.get("rgb") is not None:
        out["rgb"] = [round(v, 3) for v in st["rgb"]]
    return out


def probe(prompt, camera, limit=5):
    """segment_text + per-candidate 3D stats. Returns list of candidate dicts."""
    entry = {"prompt": prompt, "camera": camera, "ok": False, "cands": []}
    found = []
    try:
        seg = segment_text(camera, prompt)
    except Exception as exc:
        entry["error"] = str(exc)
        report["probes"][camera + ":" + prompt] = entry
        return found
    entry["ok"] = bool(seg.ok)
    if seg.ok:
        k = 0
        for cand in seg.segmentations:
            if k >= limit:
                break
            k += 1
            score = 0.0
            try:
                score = float(cand.score)
            except Exception:
                pass
            box = None
            try:
                if cand.box_xyxy is not None:
                    box = [round(float(v), 1) for v in cand.box_xyxy]
            except Exception:
                pass
            st = cloud_stats(cloud_of(cand, camera))
            entry["cands"].append({"score": round(score, 3), "box": box,
                                   "stats": public_stats(st)})
            if st is not None and st["n"] >= 15 and "x" in st:
                found.append({"prompt": prompt, "camera": camera, "score": score, "box": box,
                              "st": st})
    else:
        try:
            entry["error"] = str(seg.error)
        except Exception:
            pass
    report["probes"][camera + ":" + prompt] = entry
    return found


def xy_extent(st):
    return max(st["x"][2] - st["x"][0], st["y"][2] - st["y"][0])


def z_extent(st):
    return st["z"][2] - st["z"][0]


def dist_xy(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def center_of(st):
    return (st["x"][1], st["y"][1], st["z"][1])


def cube_prompts():
    if cube_color is not None:
        return [cube_color + " cube", cube_color + " block", "cube"]
    return ["cube", "block"]


def pick_cube(cands, near=None, near_radius=None):
    """Bounded on both ends: plausible cube size; then colour, then SAM score."""
    best = None
    best_key = None
    for c in cands:
        st = c["st"]
        ext = xy_extent(st)
        if ext < 0.01 or ext > 0.10 or z_extent(st) > 0.10:
            continue
        if near is not None and dist_xy(center_of(st), near) > near_radius:
            continue
        cm = color_match(st.get("rgb"), cube_color)
        key = ((cm if cm is not None else 0.0), c["score"])
        if best_key is None or key > best_key:
            best_key = key
            best = c
    return best


def top_rect(st):
    """Cube top-face rectangle: centre xy, yaw (mod 90 deg), top z, low z."""
    pts = st["pts"]
    zs = [p[2] for p in pts]
    top = quantile(zs, 0.95)
    low = quantile(zs, 0.03)
    tp = [p for p in pts if p[2] > top - 0.012]
    if len(tp) < 8:
        tp = pts
    best = None
    for k in range(18):
        a = math.radians(5.0 * k)
        c = math.cos(a)
        s = math.sin(a)
        us = [p[0] * c + p[1] * s for p in tp]
        vs = [-p[0] * s + p[1] * c for p in tp]
        u0 = quantile(us, 0.04)
        u1 = quantile(us, 0.96)
        v0 = quantile(vs, 0.04)
        v1 = quantile(vs, 0.96)
        area = (u1 - u0) * (v1 - v0)
        if best is None or area < best[0]:
            best = (area, a, 0.5 * (u0 + u1), 0.5 * (v0 + v1), u1 - u0, v1 - v0)
    area, a, uc, vc, w, h = best
    cx = uc * math.cos(a) - vc * math.sin(a)
    cy = uc * math.sin(a) + vc * math.cos(a)
    return {"cx": cx, "cy": cy, "yaw": a, "top": top, "low": low, "w": w, "h": h,
            "n_top": len(tp)}


# ---------------------------------------------------------------- motion helpers


def check_terminal(res, label):
    try:
        if bool(res.terminated):
            FLAGS["terminal"] = True
            FLAGS["terminal_info"] = label + ":terminated"
    except Exception:
        pass
    try:
        if bool(res.truncated):
            FLAGS["terminal"] = True
            FLAGS["terminal_info"] = label + ":truncated"
    except Exception:
        pass


def record(label, res, t0, extra=None):
    entry = {"label": label, "t0": t0, "ticks": now_tick() - t0}
    try:
        entry["ok"] = bool(res.ok)
    except Exception:
        entry["ok"] = None
    try:
        entry["status"] = str(res.status)
    except Exception:
        pass
    try:
        entry["steps"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        if res.error is not None:
            entry["error"] = str(res.error)[:300]
    except Exception:
        pass
    if extra is not None:
        entry["extra"] = extra
    report["phases"].append(entry)
    check_terminal(res, label)
    return entry


def move(pos, quat, label, tol=0.015, steps=150):
    if FLAGS["terminal"]:
        return False
    t0 = now_tick()
    target = Pose((float(pos[0]), float(pos[1]), float(pos[2])), tuple(quat), BASE)
    res = move_to_pose(target, tolerance=tol, max_steps=steps)
    entry = record(label, res, t0, {"target": [round(float(v), 4) for v in pos]})
    if (not entry.get("ok")) and "exceeds max_steps" in str(entry.get("error", "")):
        t0 = now_tick()
        res = move_to_pose(target, tolerance=tol, max_steps=steps * 3)
        entry = record(label + "_retry_long", res, t0)
    ee = get_robot_state().end_effector_poses["primary"].position
    entry["ee_after"] = [round(float(v), 4) for v in ee]
    return bool(entry.get("ok"))


def gripper_now():
    return float(get_robot_state().gripper_positions["primary"])


def do_open(label):
    if FLAGS["terminal"]:
        return
    t0 = now_tick()
    res = open_gripper()
    record(label, res, t0, {"gripper": round(gripper_now(), 3)})


def do_close(label):
    if FLAGS["terminal"]:
        return 0.0
    t0 = now_tick()
    res = close_gripper()
    g = gripper_now()
    record(label, res, t0, {"gripper": round(g, 3)})
    return g


# ---------------------------------------------------------------- initial perception
cube_cands = []
for prompt in cube_prompts():
    cube_cands += probe(prompt, FRONT)
cube = pick_cube(cube_cands)
if cube is None:
    # no plausible cube: record and fall back to a probe-only run (no motion beyond a toggle)
    report["errors"].append("cube not found in front view")
    count = 0
    FLAGS["terminal"] = True
    FLAGS["terminal_info"] = "no_cube"
    cube = {"prompt": "none", "score": 0.0,
            "st": {"n": 0, "pts": [(0.0, 0.0, 0.0)], "x": [0.0, 0.0, 0.0],
                   "y": [0.0, 0.0, 0.0], "z": [0.0, 0.0, 0.0], "rgb": None}}
cube_rect = top_rect(cube["st"])
report["cube_initial"] = {"prompt": cube["prompt"], "score": round(cube["score"], 3),
                          "stats": public_stats(cube["st"]),
                          "rect": {k: round(float(cube_rect[k]), 4) for k in cube_rect}}
CUBE_HEIGHT = max(0.015, min(0.08, cube_rect["top"] - cube_rect["low"]))
TABLE_Z = cube_rect["low"]
report["cube_height_est"] = round(CUBE_HEIGHT, 4)
report["table_z_est"] = round(TABLE_Z, 4)
cube_c = (cube_rect["cx"], cube_rect["cy"], cube_rect["top"])

target = None
if target_word is not None:
    tprompts = []
    for p in (target_word, "red and white target", "round target", "circular target", "target"):
        if p not in tprompts:
            tprompts.append(p)
    tcands = []
    for p in tprompts:
        tcands += probe(p, FRONT)
    best_key = None
    for c in tcands:
        st = c["st"]
        if xy_extent(st) > 0.30 or z_extent(st) > 0.06:
            continue
        cc = center_of(st)
        # reject the cube itself being labelled as the target
        if dist_xy(cc, cube_c) < 0.03 and st["z"][2] > cube_rect["top"] - 0.01:
            continue
        key = c["score"]
        if best_key is None or key > best_key:
            best_key = key
            target = c
    if target is not None:
        tst = target["st"]
        report["target"] = {"prompt": target["prompt"], "score": round(target["score"], 3),
                            "stats": public_stats(tst)}
    else:
        report["errors"].append("target not found; placing back at pick location")

button = None
if wants_button:
    bcands = []
    for p in ("button", "red button", "round button"):
        bcands += probe(p, FRONT)
    best_key = None
    for c in bcands:
        st = c["st"]
        if xy_extent(st) > 0.15:
            continue
        cc = center_of(st)
        if dist_xy(cc, cube_c) < 0.03:
            continue
        if target is not None and dist_xy(cc, center_of(target["st"])) < 0.03:
            continue
        key = c["score"]
        if best_key is None or key > best_key:
            best_key = key
            button = c
    if button is not None:
        report["button"] = {"prompt": button["prompt"], "score": round(button["score"], 3),
                            "stats": public_stats(button["st"])}
    else:
        report["errors"].append("button not found")

# extra context probes for the analysis (no control ticks)
probe("cube", WRIST, 3)
report["tick_after_perception"] = now_tick()

# place location (fixed for all cycles)
if target is not None:
    tst = target["st"]
    PLACE_XY = (tst["x"][1], tst["y"][1])
    PLACE_SURFACE_Z = tst["z"][2]
else:
    PLACE_XY = (cube_rect["cx"], cube_rect["cy"])
    PLACE_SURFACE_Z = TABLE_Z
report["place"] = {"xy": [round(PLACE_XY[0], 4), round(PLACE_XY[1], 4)],
                   "surface_z": round(PLACE_SURFACE_Z, 4)}

# ---------------------------------------------------------------- relocalisation


def relocalize(predicted):
    """Front first, then wrist; accept only near the predicted centre; else predicted."""
    for cam in (FRONT, WRIST):
        cands = []
        for prompt in cube_prompts()[:2]:
            cands += probe(prompt, cam, 4)
        c = pick_cube(cands, near=predicted, near_radius=0.05)
        if c is not None:
            rect = top_rect(c["st"])
            rect["source"] = cam
            return rect
    return None


# ---------------------------------------------------------------- pick/place cycles
GRASP_ABOVE = 0.10   # pre-grasp stand-off above grasp point (gripper empty, open)
LIFT_ABOVE = 0.15    # lift height above grasp point (gripper holding the cube)
RETREAT_ABOVE = 0.10  # retreat after release (gripper open, empty)
cycles = []
rect = cube_rect
last_yaw = cube_rect["yaw"]
for cycle in range(count):
    if FLAGS["terminal"]:
        break
    info = {"cycle": cycle + 1, "t0": now_tick()}
    if cycle > 0:
        predicted = (PLACE_XY[0], PLACE_XY[1])
        rr = relocalize(predicted)
        if rr is None:
            rect = {"cx": predicted[0], "cy": predicted[1], "yaw": last_yaw,
                    "top": PLACE_SURFACE_Z + CUBE_HEIGHT if target is not None else TABLE_Z + CUBE_HEIGHT,
                    "low": PLACE_SURFACE_Z, "source": "predicted"}
        else:
            rect = rr
        info["rect_source"] = rect.get("source", "front")
    info["rect"] = {k: (round(float(rect[k]), 4) if k != "source" else rect[k]) for k in rect
                    if k in ("cx", "cy", "yaw", "top", "low", "source")}
    held = False
    for attempt in range(2):
        if FLAGS["terminal"]:
            break
        top_z = rect["top"]
        grasp_z = max(top_z - 0.5 * CUBE_HEIGHT, top_z - 0.025)
        grasp_z = max(grasp_z, top_z - CUBE_HEIGHT + 0.010)
        quat, delta = grasp_quat(rect["yaw"])
        info["grasp_z"] = round(grasp_z, 4)
        info["yaw_delta_deg"] = round(math.degrees(delta), 1)
        if gripper_now() < 0.9:
            do_open("c%d_open_pre" % (cycle + 1))
        move((rect["cx"], rect["cy"], grasp_z + GRASP_ABOVE), quat,
             "c%d_a%d_pregrasp" % (cycle + 1, attempt), tol=0.015, steps=200)
        move((rect["cx"], rect["cy"], grasp_z), quat,
             "c%d_a%d_descend" % (cycle + 1, attempt), tol=0.008, steps=120)
        g = do_close("c%d_a%d_close" % (cycle + 1, attempt))
        info["gripper_closed"] = round(g, 3)
        held = g > 0.08
        if held:
            break
        # missed: open, back off, re-observe from above with the wrist camera
        do_open("c%d_a%d_reopen" % (cycle + 1, attempt))
        move((rect["cx"], rect["cy"], grasp_z + GRASP_ABOVE), quat,
             "c%d_a%d_backoff" % (cycle + 1, attempt), tol=0.02, steps=150)
        rr = relocalize((rect["cx"], rect["cy"]))
        if rr is not None:
            rect = rr
    info["held"] = held
    if not held:
        info["t1"] = now_tick()
        cycles.append(info)
        report["errors"].append("cycle %d: grasp failed" % (cycle + 1))
        break
    lift_z = grasp_z + LIFT_ABOVE
    move((rect["cx"], rect["cy"], lift_z), quat, "c%d_lift" % (cycle + 1), tol=0.02, steps=150)
    info["gripper_after_lift"] = round(gripper_now(), 3)
    # place: keep the cube's bottom ~1 cm above the place surface
    place_z = PLACE_SURFACE_Z + (grasp_z - rect["low"]) + 0.012
    if dist_xy((rect["cx"], rect["cy"]), PLACE_XY) > 0.01:
        move((PLACE_XY[0], PLACE_XY[1], lift_z), quat, "c%d_transport" % (cycle + 1),
             tol=0.02, steps=200)
    move((PLACE_XY[0], PLACE_XY[1], place_z), quat, "c%d_lower" % (cycle + 1),
         tol=0.01, steps=150)
    do_open("c%d_release" % (cycle + 1))
    move((PLACE_XY[0], PLACE_XY[1], place_z + RETREAT_ABOVE), quat,
         "c%d_retreat" % (cycle + 1), tol=0.02, steps=150)
    last_yaw = rect["yaw"]
    info["t1"] = now_tick()
    cycles.append(info)
report["cycles"] = cycles

# ---------------------------------------------------------------- button press
if wants_button and button is not None and not FLAGS["terminal"]:
    bst = button["st"]
    bx = bst["x"][1]
    by = bst["y"][1]
    btop = bst["z"][2]
    report["button_press"] = {"xy": [round(bx, 4), round(by, 4)], "top": round(btop, 4)}
    do_close("button_close_gripper")
    move((bx, by, btop + 0.08), Q_DOWN, "button_above", tol=0.015, steps=200)
    move((bx, by, btop - 0.015), Q_DOWN, "button_press", tol=0.005, steps=50)
    move((bx, by, btop + 0.08), Q_DOWN, "button_release", tol=0.02, steps=100)

report["terminal"] = FLAGS["terminal_info"]
report["tick_end"] = now_tick()
result = report
