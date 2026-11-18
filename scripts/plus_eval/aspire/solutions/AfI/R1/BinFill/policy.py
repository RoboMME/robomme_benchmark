# v00_initial — BinFill (robomme_plus task 0), ASPIRE Stage 0 snapshot + first attempt.
#
# AfI, R=1: this single program is both the scene snapshot and the task attempt.
# Phase 1 (no control ticks): language, task/controller metadata, cameras, robot state,
#   conditioning-video summary, SAM3 prompt probes on both cameras.
# Phase 2: parse the runtime goal (counts, colours, container, button) from the language.
# Phase 3: localize container / cubes / button on the front camera, then a counted
#   pick -> container -> release loop with wrist-camera refinement at the hover pose.
# Phase 4: press the button only if the language asks for it.
# Every phase is wrapped; the program never raises. All poses come from perception.

import math

report = {
    "version": "v00_initial",
    "errors": [],
    "events": [],
    "phase_done": [],
}
result = report
state_flags = {"ended": False, "end_reason": "", "first_step_done": False}


def note_error(phase, exc):
    text = phase + ": " + str(exc)
    if len(text) > 400:
        text = text[:400]
    report["errors"].append(text)
    print("ERROR " + text)


def short(value, limit=300):
    text = str(value)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return 0.0
    index = int((len(ordered) - 1) * fraction)
    return ordered[index]


def r4(v):
    return round(float(v), 4)


# ------------------------------------------------------------------ quaternion helpers
def qmul(a, b):
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def qnormalize(q):
    n = math.sqrt(sum(float(v) * float(v) for v in q))
    if n < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    return (float(q[0]) / n, float(q[1]) / n, float(q[2]) / n, float(q[3]) / n)


def qrot(q, v):
    p = (0.0, float(v[0]), float(v[1]), float(v[2]))
    qc = (q[0], -q[1], -q[2], -q[3])
    out = qmul(qmul(q, p), qc)
    return (out[1], out[2], out[3])


def qz(angle):
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


# ------------------------------------------------------------------ phase 1: snapshot
base_frame = "robot_base"
control_hz = 20.0
max_steps = 0
home_joints = None
home_quat = None
home_pos = None

try:
    ctx = get_task_context()
    report["language"] = str(ctx.language)
    try:
        report["task_name"] = str(ctx.task_name)
        report["task_id"] = str(ctx.task_id)
        report["suite"] = str(ctx.suite)
        report["family"] = str(ctx.family)
    except Exception as exc:
        note_error("context-fields", exc)
    print("LANGUAGE: " + report["language"])
except Exception as exc:
    report["language"] = ""
    note_error("context", exc)

try:
    tm = robomme_plus.get_task_metadata()
    meta = {}
    for key in tm:
        meta[str(key)] = short(tm[key], 400)
    report["task_metadata"] = meta
    try:
        max_steps = int(tm["max_steps"])
    except Exception as exc:
        note_error("max_steps", exc)
    for key in tm:
        lk = str(key).lower()
        if "hz" in lk or "freq" in lk:
            try:
                control_hz = float(tm[key])
            except Exception:
                pass
except Exception as exc:
    note_error("task_metadata", exc)
report["max_steps"] = max_steps
report["control_hz"] = control_hz

try:
    cm = robomme_plus.get_controller_metadata()
    cmeta = {}
    for key in cm:
        cmeta[str(key)] = short(cm[key], 400)
    report["controller_metadata"] = cmeta
except Exception as exc:
    note_error("controller_metadata", exc)

camera_names = []
try:
    obs0 = get_observation()
    cams = {}
    for name in obs0.cameras:
        camera_names.append(str(name))
        cam = obs0.cameras[name]
        info = {}
        try:
            info["rgb_shape"] = str(cam.rgb.shape)
            info["depth_shape"] = str(cam.depth_m.shape)
        except Exception as exc:
            note_error("cam-shape-" + str(name), exc)
        try:
            info["intrinsics"] = [[r4(cam.intrinsics[i][j]) for j in range(3)] for i in range(3)]
        except Exception as exc:
            note_error("cam-intr-" + str(name), exc)
        try:
            info["pose_position"] = [r4(v) for v in cam.camera_pose.position]
            info["pose_quat_wxyz"] = [r4(v) for v in cam.camera_pose.quaternion_wxyz]
            info["pose_frame"] = str(cam.camera_pose.frame)
        except Exception as exc:
            note_error("cam-pose-" + str(name), exc)
        try:
            h = int(cam.depth_m.shape[0])
            w = int(cam.depth_m.shape[1])
            depths = []
            for yy in range(0, h, 16):
                for xx in range(0, w, 16):
                    d = float(cam.depth_m[yy][xx])
                    if d > 0.0 and d < 10.0:
                        depths.append(d)
            info["depth_q05_q50_q95"] = [r4(quantile(depths, 0.05)), r4(quantile(depths, 0.5)),
                                         r4(quantile(depths, 0.95))]
            info["depth_valid_samples"] = len(depths)
        except Exception as exc:
            note_error("cam-depth-" + str(name), exc)
        cams[str(name)] = info
    report["cameras"] = cams
except Exception as exc:
    note_error("observation", exc)
report["camera_names"] = camera_names
front_cam = "front" if "front" in camera_names else (camera_names[0] if camera_names else "front")
wrist_cam = "wrist" if "wrist" in camera_names else ""

try:
    st0 = get_robot_state()
    base_frame = str(st0.base_frame)
    home_joints = [float(v) for v in st0.joint_positions["primary"]]
    ee0 = st0.end_effector_poses["primary"]
    home_pos = [float(v) for v in ee0.position]
    home_quat = qnormalize([float(v) for v in ee0.quaternion_wxyz])
    report["robot_state0"] = {
        "base_frame": base_frame,
        "embodiment": str(st0.embodiment),
        "joints": [r4(v) for v in home_joints],
        "ee_position": [r4(v) for v in home_pos],
        "ee_quat_wxyz": [r4(v) for v in home_quat],
        "ee_frame": str(ee0.frame),
        "gripper": r4(st0.gripper_positions["primary"]),
        "timestamp_s": r4(st0.timestamp_s),
    }
except Exception as exc:
    note_error("robot_state", exc)

# Tool orientation: keep the home orientation if its approach axis points down.
down_quat = (0.0, 1.0, 0.0, 0.0)
try:
    if home_quat is not None:
        zaxis = qrot(home_quat, (0.0, 0.0, 1.0))
        report["home_tool_z_axis"] = [r4(v) for v in zaxis]
        if zaxis[2] < -0.9:
            down_quat = home_quat
            report["down_quat_source"] = "home"
        else:
            report["down_quat_source"] = "constructed (home tool axis not downward)"
except Exception as exc:
    note_error("down_quat", exc)
report["down_quat"] = [r4(v) for v in down_quat]

# Conditioning video: cheap summary only.
try:
    video = robomme_plus.get_conditioning_video()
    nvid = len(video)
    vs = {"frames": nvid}
    if nvid > 0:
        vs["t_first"] = r4(video[0].timestamp_s)
        vs["t_last"] = r4(video[nvid - 1].timestamp_s)
        samples = []
        picks = sorted(set([0, nvid - 1] + [int(k * (nvid - 1) / 7.0) for k in range(8)]))
        for idx in picks:
            fr = video[idx]
            entry = {"i": idx, "t": r4(fr.timestamp_s)}
            try:
                rgb = fr.cameras[front_cam].rgb
                hh = int(rgb.shape[0])
                ww = int(rgb.shape[1])
                acc = [0.0, 0.0, 0.0]
                cnt = 0
                for yy in range(0, hh, 32):
                    for xx in range(0, ww, 32):
                        px = rgb[yy][xx]
                        acc[0] += float(px[0])
                        acc[1] += float(px[1])
                        acc[2] += float(px[2])
                        cnt += 1
                entry["front_mean_rgb"] = [round(a / max(cnt, 1), 1) for a in acc]
            except Exception as exc:
                entry["err"] = short(exc, 120)
            try:
                entry["ee"] = [r4(v) for v in fr.robot_state.end_effector_poses["primary"].position]
                entry["grip"] = r4(fr.robot_state.gripper_positions["primary"])
            except Exception:
                pass
            samples.append(entry)
        vs["samples"] = samples
    report["conditioning_video"] = vs
    print("conditioning video frames: " + str(nvid))
except Exception as exc:
    note_error("conditioning_video", exc)


# ------------------------------------------------------------------ perception helpers
def seg(camera, prompt, limit=8):
    try:
        found = segment_text(camera, prompt)
    except Exception as exc:
        note_error("segment " + camera + " " + prompt, exc)
        return []
    try:
        if not found.ok:
            return []
        return list(found.segmentations[:limit])
    except Exception as exc:
        note_error("segment-read " + prompt, exc)
        return []


def box_of(candidate):
    try:
        b = candidate.box_xyxy
        if b is None:
            return None
        return [float(v) for v in b]
    except Exception:
        return None


def score_of(candidate):
    try:
        return float(candidate.score)
    except Exception:
        return 0.0


def cloud_points(candidate, camera, cap=1500):
    pc = None
    try:
        pc = mask_to_point_cloud(candidate, camera, target_frame=base_frame)
    except Exception as exc:
        try:
            pc = mask_to_point_cloud(candidate.mask, camera, target_frame=base_frame)
        except Exception as exc2:
            note_error("mask_to_point_cloud " + camera, exc2)
            return []
    pts = []
    try:
        raw = pc.points
        n = len(raw)
        stride = 1
        if n > cap:
            stride = int(n / cap) + 1
        for i in range(0, n, stride):
            p = raw[i]
            x = float(p[0])
            y = float(p[1])
            z = float(p[2])
            if x == x and y == y and z == z and abs(x) < 5.0 and abs(y) < 5.0 and abs(z) < 5.0:
                pts.append((x, y, z))
    except Exception as exc:
        note_error("cloud-read " + camera, exc)
    return pts


def summarize(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    s = {
        "n": len(pts),
        "x05": quantile(xs, 0.05), "x95": quantile(xs, 0.95),
        "y05": quantile(ys, 0.05), "y95": quantile(ys, 0.95),
        "z05": quantile(zs, 0.05), "z50": quantile(zs, 0.5), "z95": quantile(zs, 0.95),
    }
    s["cx"] = 0.5 * (s["x05"] + s["x95"])
    s["cy"] = 0.5 * (s["y05"] + s["y95"])
    s["ex"] = s["x95"] - s["x05"]
    s["ey"] = s["y95"] - s["y05"]
    s["ez"] = s["z95"] - s["z05"]
    return s


def current_rgb(camera):
    try:
        obs = get_observation()
        return obs.cameras[camera].rgb
    except Exception as exc:
        note_error("rgb-read " + camera, exc)
        return None


def mean_rgb_in_box(rgb, candidate):
    """Mean RGB of the mask pixels inside the box (inner region fallback)."""
    if rgb is None:
        return None
    try:
        hh = int(rgb.shape[0])
        ww = int(rgb.shape[1])
    except Exception:
        return None
    b = box_of(candidate)
    if b is None:
        return None
    x1 = max(0, int(b[0]))
    y1 = max(0, int(b[1]))
    x2 = min(ww - 1, int(b[2]))
    y2 = min(hh - 1, int(b[3]))
    if x2 < x1 or y2 < y1:
        return None
    stride = max(1, int(max(x2 - x1, y2 - y1) / 12))
    mask = None
    try:
        mask = candidate.mask
    except Exception:
        mask = None
    acc = [0.0, 0.0, 0.0]
    cnt = 0
    if mask is not None:
        try:
            for yy in range(y1, y2 + 1, stride):
                for xx in range(x1, x2 + 1, stride):
                    if bool(mask[yy][xx]):
                        px = rgb[yy][xx]
                        acc[0] += float(px[0])
                        acc[1] += float(px[1])
                        acc[2] += float(px[2])
                        cnt += 1
        except Exception:
            cnt = 0
            acc = [0.0, 0.0, 0.0]
    if cnt == 0:
        mx1 = int(x1 + 0.3 * (x2 - x1))
        mx2 = int(x1 + 0.7 * (x2 - x1))
        my1 = int(y1 + 0.3 * (y2 - y1))
        my2 = int(y1 + 0.7 * (y2 - y1))
        for yy in range(my1, my2 + 1):
            for xx in range(mx1, mx2 + 1):
                px = rgb[yy][xx]
                acc[0] += float(px[0])
                acc[1] += float(px[1])
                acc[2] += float(px[2])
                cnt += 1
    if cnt == 0:
        return None
    return [acc[0] / cnt, acc[1] / cnt, acc[2] / cnt]


def color_name(rgb):
    if rgb is None:
        return "unknown"
    r = rgb[0] / 255.0
    g = rgb[1] / 255.0
    b = rgb[2] / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 0.15:
        return "black"
    sat = (mx - mn) / mx if mx > 0 else 0.0
    if sat < 0.22:
        if mx > 0.7:
            return "white"
        return "gray"
    d = mx - mn
    if mx == r:
        hue = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        hue = 60.0 * ((b - r) / d + 2.0)
    else:
        hue = 60.0 * ((r - g) / d + 4.0)
    if hue < 15.0 or hue >= 335.0:
        return "red"
    if hue < 40.0:
        return "orange"
    if hue < 70.0:
        return "yellow"
    if hue < 165.0:
        return "green"
    if hue < 195.0:
        return "cyan"
    if hue < 255.0:
        return "blue"
    if hue < 295.0:
        return "purple"
    return "pink"


NEIGHBOR_COLORS = {
    "red": ["orange", "pink"], "orange": ["red", "yellow"], "yellow": ["orange", "green"],
    "green": ["cyan", "yellow"], "cyan": ["green", "blue"], "blue": ["cyan", "purple"],
    "purple": ["blue", "pink"], "pink": ["red", "purple"], "magenta": ["pink", "purple"],
    "grey": ["gray", "white"], "gray": ["white", "black"], "white": ["gray"], "black": ["gray"],
    "brown": ["orange", "red"],
}


def color_rank(observed, wanted):
    """0 = exact, 1 = neighbouring hue, 2 = mismatch; wanted '' accepts anything."""
    if wanted == "" or wanted is None:
        return 0
    w = wanted
    if w == "grey":
        w = "gray"
    if w == "magenta":
        w = "pink"
    if observed == w:
        return 0
    if observed in NEIGHBOR_COLORS.get(w, []):
        return 1
    return 2


# ------------------------------------------------------------------ SAM3 probes (phase 1)
PROBE_PROMPTS = ["cube", "red cube", "green cube", "blue cube", "yellow cube",
                 "bin", "box", "button"]
try:
    probes = []
    for cam_name in [front_cam, wrist_cam]:
        if cam_name == "":
            continue
        for prompt in PROBE_PROMPTS:
            cands = seg(cam_name, prompt, 5)
            entry = {"camera": cam_name, "prompt": prompt, "n": len(cands), "top": []}
            for c in cands:
                b = box_of(c)
                entry["top"].append({"score": round(score_of(c), 3),
                                     "box": [round(v, 1) for v in b] if b is not None else None})
            probes.append(entry)
    report["prompt_probes"] = probes
    print("probes done: " + str(len(probes)))
except Exception as exc:
    note_error("probes", exc)
report["phase_done"].append("snapshot")

# ------------------------------------------------------------------ phase 2: parse goal
NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "single": 1, "two": 2, "three": 3, "four": 4,
                "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "once": 1, "twice": 2, "thrice": 3}
COLOR_WORDS = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "white", "black",
               "gray", "grey", "brown", "cyan", "magenta"]
OBJECT_NOUNS = ["cube", "cubes", "block", "blocks"]
CONTAINER_NOUNS = ["bin", "box", "basket", "container", "tray", "bowl", "bucket", "crate"]

goal = {"requirements": [], "container": "", "press_button": False, "object_noun": "cube",
        "all": False}
try:
    text = report.get("language", "").lower()
    for ch in [",", ".", ";", ":", "!", "?", "(", ")", "\"", "'", "-"]:
        text = text.replace(ch, " ")
    tokens = text.split()
    report["tokens"] = tokens
    for i in range(len(tokens)):
        tok = tokens[i]
        count = None
        if tok in NUMBER_WORDS:
            count = NUMBER_WORDS[tok]
        elif tok.isdigit():
            count = int(tok)
        if count is None:
            continue
        colour = ""
        noun = ""
        for j in range(i + 1, min(i + 5, len(tokens))):
            if tokens[j] in COLOR_WORDS and colour == "":
                colour = tokens[j]
            if tokens[j] in OBJECT_NOUNS:
                noun = tokens[j]
                break
        if noun != "":
            goal["requirements"].append({"count": count, "color": colour})
            goal["object_noun"] = "block" if noun.startswith("block") else "cube"
    if "all" in tokens or "every" in tokens:
        goal["all"] = True
    for tok in tokens:
        if tok in CONTAINER_NOUNS and goal["container"] == "":
            goal["container"] = tok
    if "button" in tokens:
        goal["press_button"] = True
    if len(goal["requirements"]) == 0:
        colour = ""
        for tok in tokens:
            if tok in COLOR_WORDS:
                colour = tok
                break
        goal["requirements"].append({"count": -1, "color": colour})
        goal["count_source"] = "unparsed: fill with every visible matching cube"
    else:
        goal["count_source"] = "language"
except Exception as exc:
    note_error("parse", exc)
    goal["requirements"] = [{"count": -1, "color": ""}]
report["goal"] = goal
print("GOAL: " + str(goal))


# ------------------------------------------------------------------ control helpers
def now_state():
    try:
        return get_robot_state()
    except Exception as exc:
        note_error("get_robot_state", exc)
        return None


def steps_used():
    st = now_state()
    if st is None:
        return 0
    try:
        return int(round(float(st.timestamp_s) * control_hz))
    except Exception:
        return 0


def steps_left():
    if max_steps <= 0:
        return 100000
    return max_steps - steps_used()


def gripper_now():
    st = now_state()
    if st is None:
        return -1.0
    try:
        return float(st.gripper_positions["primary"])
    except Exception:
        return -1.0


def ee_now():
    st = now_state()
    if st is None:
        return [0.0, 0.0, 0.0]
    try:
        return [float(v) for v in st.end_effector_poses["primary"].position]
    except Exception:
        return [0.0, 0.0, 0.0]


def check_terminal(res, label):
    flags = []
    try:
        if bool(res.terminated):
            flags.append("terminated")
    except Exception:
        pass
    try:
        if bool(res.truncated):
            flags.append("truncated")
    except Exception:
        pass
    status = ""
    try:
        status = str(res.status)
    except Exception:
        status = ""
    low = status.lower()
    if "terminat" in low or "truncat" in low or "episode_end" in low or "done" == low:
        flags.append("status:" + status)
    if len(flags) > 0:
        state_flags["ended"] = True
        state_flags["end_reason"] = label + " " + ",".join(flags)
        print("EPISODE ENDED at " + label + ": " + ",".join(flags))
    return flags


def record(label, res, extra=None):
    rec = {"label": label}
    try:
        rec["ok"] = bool(res.ok)
    except Exception:
        rec["ok"] = False
    try:
        rec["status"] = str(res.status)
    except Exception:
        pass
    try:
        rec["steps"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        if res.error is not None:
            rec["error"] = short(res.error, 300)
    except Exception:
        pass
    rec["terminal"] = check_terminal(res, label)
    rec["ee"] = [r4(v) for v in ee_now()]
    rec["grip"] = r4(gripper_now())
    rec["t_steps"] = steps_used()
    if extra is not None:
        for key in extra:
            rec[key] = extra[key]
    state_flags["first_step_done"] = True
    report["events"].append(rec)
    print("EVENT " + str(rec))
    return rec


def move(label, xyz, quat, tol=0.01, limit=150):
    if state_flags["ended"]:
        return {"ok": False, "skipped": True}
    try:
        target = Pose([float(xyz[0]), float(xyz[1]), float(xyz[2])],
                      [float(v) for v in quat], base_frame)
        res = move_to_pose(target, tolerance=tol, max_steps=limit)
    except Exception as exc:
        note_error("move " + label, exc)
        rec = {"label": label, "ok": False, "exception": short(exc, 300),
               "target": [r4(v) for v in xyz]}
        report["events"].append(rec)
        return rec
    return record(label, res, {"target": [r4(v) for v in xyz]})


def grip(label, close):
    if state_flags["ended"]:
        return {"ok": False, "skipped": True}
    try:
        res = close_gripper() if close else open_gripper()
    except Exception as exc:
        note_error("gripper " + label, exc)
        return {"ok": False}
    return record(label, res)


def hold(label, ticks, grip_value):
    if state_flags["ended"]:
        return
    st = now_state()
    if st is None:
        return
    for tick in range(ticks):
        try:
            reply = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=[float(v) for v in st.joint_positions["primary"]],
                gripper_position=float(grip_value), embodiment=st.embodiment)}))
        except Exception as exc:
            note_error("hold " + label, exc)
            return
        if len(check_terminal(reply, label + "-tick" + str(tick))) > 0:
            return


def go_observe(label):
    if state_flags["ended"] or home_joints is None:
        return
    try:
        res = move_to_joints(home_joints, tolerance=0.03, max_steps=100)
        record(label, res)
    except Exception as exc:
        note_error("move_to_joints " + label, exc)


def rect_angle(xy):
    """Angle (rad, in [0, pi/2)) of the min-area bounding rectangle, and min/max area ratio."""
    if len(xy) < 12:
        return 0.0, 1.0
    if len(xy) > 300:
        stride = int(len(xy) / 300) + 1
        xy = [xy[i] for i in range(0, len(xy), stride)]
    best_a = 0.0
    best_area = None
    worst_area = None
    for k in range(18):
        a = math.radians(5.0 * k)
        c = math.cos(a)
        s = math.sin(a)
        us = [p[0] * c + p[1] * s for p in xy]
        vs = [-p[0] * s + p[1] * c for p in xy]
        area = (quantile(us, 0.95) - quantile(us, 0.05)) * (quantile(vs, 0.95) - quantile(vs, 0.05))
        if best_area is None or area < best_area:
            best_area = area
            best_a = a
        if worst_area is None or area > worst_area:
            worst_area = area
    if worst_area is None or worst_area <= 1e-12:
        return 0.0, 1.0
    return best_a, best_area / worst_area


# ------------------------------------------------------------------ scene localization
def locate_container():
    prompts = []
    if goal["container"] != "":
        prompts.append(goal["container"])
    for p in ["bin", "box", "container", "basket", "tray"]:
        if p not in prompts:
            prompts.append(p)
    tried = []
    for prompt in prompts:
        cands = seg(front_cam, prompt, 5)
        for c in cands:
            pts = cloud_points(c, front_cam)
            if len(pts) < 20:
                continue
            s = summarize(pts)
            tried.append({"prompt": prompt, "score": round(score_of(c), 3), "n": s["n"],
                          "cx": r4(s["cx"]), "cy": r4(s["cy"]), "ex": r4(s["ex"]),
                          "ey": r4(s["ey"]), "z05": r4(s["z05"]), "z95": r4(s["z95"])})
            # A container is larger than a cube but still table-sized: bound both ends.
            if s["ex"] < 0.06 and s["ey"] < 0.06:
                continue
            if s["ex"] > 0.6 or s["ey"] > 0.6:
                continue
            report["container_candidates"] = tried
            return {"prompt": prompt, "score": score_of(c), "cx": s["cx"], "cy": s["cy"],
                    "x05": s["x05"], "x95": s["x95"], "y05": s["y05"], "y95": s["y95"],
                    "rim_z": s["z95"], "floor_z": s["z05"], "n": s["n"]}
    report["container_candidates"] = tried
    return None


def in_container_xy(x, y, cont):
    if cont is None:
        return False
    return (cont["x05"] - 0.01 <= x <= cont["x95"] + 0.01) and (cont["y05"] - 0.01 <= y <= cont["y95"] + 0.01)


def scan_cubes(camera, colour, noun, label):
    """All cube-like candidates (deduplicated in 3D) with observed colour."""
    prompts = []
    if colour != "":
        prompts.append(colour + " " + noun)
    prompts.append(noun)
    if noun != "cube":
        prompts.append("cube")
    found = []
    rgb = current_rgb(camera)
    for prompt in prompts:
        cands = seg(camera, prompt, 8)
        for c in cands:
            pts = cloud_points(c, camera)
            if len(pts) < 6:
                continue
            s = summarize(pts)
            # Bound the selector at both ends: cube-sized, not the arm, not the container.
            if s["ex"] > 0.10 or s["ey"] > 0.10 or s["ez"] > 0.10:
                continue
            obs_col = color_name(mean_rgb_in_box(rgb, c))
            dup = False
            for f in found:
                if abs(f["cx"] - s["cx"]) < 0.025 and abs(f["cy"] - s["cy"]) < 0.025:
                    dup = True
                    if score_of(c) > f["score"] and prompt == f["prompt"]:
                        f["score"] = score_of(c)
                    break
            if dup:
                continue
            found.append({"prompt": prompt, "score": score_of(c), "color": obs_col,
                          "cx": s["cx"], "cy": s["cy"], "z05": s["z05"], "z95": s["z95"],
                          "ex": s["ex"], "ey": s["ey"], "n": s["n"], "pts": pts})
    return found


def cube_view(f):
    return {"prompt": f["prompt"], "score": round(f["score"], 3), "color": f["color"],
            "cx": r4(f["cx"]), "cy": r4(f["cy"]), "z05": r4(f["z05"]), "z95": r4(f["z95"]),
            "ex": r4(f["ex"]), "ey": r4(f["ey"]), "n": f["n"]}


def locate_button():
    for prompt in ["button", "red button", "push button", "round button"]:
        cands = seg(front_cam, prompt, 5)
        for c in cands:
            pts = cloud_points(c, front_cam)
            if len(pts) < 6:
                continue
            s = summarize(pts)
            if s["ex"] > 0.15 or s["ey"] > 0.15:
                continue
            return {"prompt": prompt, "score": score_of(c), "cx": s["cx"], "cy": s["cy"],
                    "top_z": s["z95"], "z05": s["z05"], "n": s["n"]}
    return None


container = None
table_z = None
try:
    container = locate_container()
    if container is not None:
        report["container"] = {"prompt": container["prompt"], "score": round(container["score"], 3),
                               "cx": r4(container["cx"]), "cy": r4(container["cy"]),
                               "x05": r4(container["x05"]), "x95": r4(container["x95"]),
                               "y05": r4(container["y05"]), "y95": r4(container["y95"]),
                               "rim_z": r4(container["rim_z"]), "floor_z": r4(container["floor_z"]),
                               "n": container["n"]}
    print("CONTAINER: " + str(report.get("container")))
except Exception as exc:
    note_error("container", exc)

try:
    tcands = seg(front_cam, "table", 3)
    if len(tcands) > 0:
        tpts = cloud_points(tcands[0], front_cam)
        if len(tpts) > 50:
            table_z = quantile([p[2] for p in tpts], 0.5)
    report["table_z_from_table_mask"] = r4(table_z) if table_z is not None else None
except Exception as exc:
    note_error("table", exc)

button = None
if goal["press_button"]:
    try:
        button = locate_button()
        if button is not None:
            report["button"] = {"prompt": button["prompt"], "score": round(button["score"], 3),
                                "cx": r4(button["cx"]), "cy": r4(button["cy"]),
                                "top_z": r4(button["top_z"]), "n": button["n"]}
        print("BUTTON: " + str(report.get("button")))
    except Exception as exc:
        note_error("button", exc)
report["phase_done"].append("localize")


# ------------------------------------------------------------------ pick-and-place loop
def choose_cube(colour, label):
    found = scan_cubes(front_cam, colour, goal["object_noun"], label)
    # Table height: lowest cube bottom seen (bounded by the table mask when available).
    zlows = [f["z05"] for f in found]
    tz = table_z
    if len(zlows) > 0:
        zmin = min(zlows)
        # The table mask is only trusted when it agrees with the lowest cube bottom.
        if tz is None or abs(zmin - tz) > 0.03:
            tz = zmin
    usable = []
    for f in found:
        f["in_container"] = in_container_xy(f["cx"], f["cy"], container)
        if f["in_container"]:
            continue
        if tz is not None and f["z05"] > tz + 0.05:
            continue  # floating: gripper/arm, not a cube resting on the table
        usable.append(f)
    scan = {"label": label, "all": [cube_view(f) for f in found], "table_z": r4(tz) if tz is not None else None}
    best = None
    best_key = None
    for f in usable:
        rank = color_rank(f["color"], colour)
        if rank >= 2:
            continue
        # isolation: distance to the nearest other detected cube
        iso = 1.0
        for g in found:
            if g is f:
                continue
            d = math.sqrt((g["cx"] - f["cx"]) ** 2 + (g["cy"] - f["cy"]) ** 2)
            if d < iso:
                iso = d
        key = (rank, -min(iso, 0.12), -f["score"])
        if best_key is None or key < best_key:
            best_key = key
            best = f
    matching = 0
    for f in usable:
        if color_rank(f["color"], colour) == 0:
            matching += 1
    scan["usable_matching_exact"] = matching
    scan["chosen"] = cube_view(best) if best is not None else None
    report.setdefault("scans", []).append(scan)
    print("SCAN " + label + ": " + str(len(found)) + " found, matching " + str(matching))
    return best, tz, matching


def refine_with_wrist(target, colour, tz):
    """At the hover pose, re-segment on the wrist camera and pick the candidate nearest the
    front-camera estimate (within 4 cm). Returns refined (x, y, top_z, yaw_angle, ratio)."""
    if wrist_cam == "":
        return None
    found = scan_cubes(wrist_cam, colour, goal["object_noun"], "wrist")
    best = None
    best_d = 0.04
    for f in found:
        if tz is not None and f["z05"] > tz + 0.05:
            continue
        if color_rank(f["color"], colour) >= 2:
            continue
        d = math.sqrt((f["cx"] - target["cx"]) ** 2 + (f["cy"] - target["cy"]) ** 2)
        if d < best_d:
            best_d = d
            best = f
    if best is None:
        report.setdefault("wrist_refine", []).append({"ok": False, "n_found": len(found),
                                                     "all": [cube_view(f) for f in found]})
        return None
    top = best["z95"]
    top_pts = [(p[0], p[1]) for p in best["pts"] if p[2] >= top - 0.01]
    if len(top_pts) < 8:
        top_pts = [(p[0], p[1]) for p in best["pts"]]
    xs = [p[0] for p in top_pts]
    ys = [p[1] for p in top_pts]
    rx = 0.5 * (quantile(xs, 0.05) + quantile(xs, 0.95))
    ry = 0.5 * (quantile(ys, 0.05) + quantile(ys, 0.95))
    ang, ratio = rect_angle(top_pts)
    out = {"ok": True, "x": rx, "y": ry, "top_z": top, "angle": ang, "ratio": ratio,
           "shift": best_d}
    report.setdefault("wrist_refine", []).append({"ok": True, "x": r4(rx), "y": r4(ry),
                                                 "top_z": r4(top), "angle_deg": round(math.degrees(ang), 1),
                                                 "ratio": round(ratio, 3), "shift": r4(best_d),
                                                 "color": best["color"], "n": best["n"]})
    return out


def yawed_quat(angle, ratio):
    """Rotate the downward tool about world z so the finger axis meets a cube edge."""
    if ratio > 0.8:
        return down_quat, 0.0
    yaxis = qrot(down_quat, (0.0, 1.0, 0.0))
    phi = math.atan2(yaxis[1], yaxis[0])
    quarter = math.pi / 2.0
    delta = ((angle - phi + quarter / 2.0) % quarter) - quarter / 2.0
    return qnormalize(qmul(qz(delta), down_quat)), delta


HOLD_THRESHOLD = 0.08  # normalized width above which the closed jaw is stopped by an object


def pick_and_place(colour, cube_index):
    label = "cube" + str(cube_index)
    target, tz, matching = choose_cube(colour, label)
    if target is None:
        return "no_target"
    if tz is None:
        tz = target["z05"]
    size = max(0.015, min(0.08, target["z95"] - tz))
    top_z = target["z95"]
    hover_z = top_z + 0.12
    grip(label + "-open", False)
    move(label + "-hover", [target["cx"], target["cy"], hover_z], down_quat, 0.015, 150)
    if state_flags["ended"]:
        return "ended"
    gx = target["cx"]
    gy = target["cy"]
    quat = down_quat
    try:
        ref = refine_with_wrist(target, colour, tz)
        if ref is not None:
            gx = ref["x"]
            gy = ref["y"]
            top_z = 0.5 * (top_z + ref["top_z"])
            quat, delta = yawed_quat(ref["angle"], ref["ratio"])
            report.setdefault("grasp_yaw_deg", []).append(round(math.degrees(delta), 1))
    except Exception as exc:
        note_error("wrist-refine", exc)
    size = max(0.015, min(0.08, top_z - tz))
    grasp_z = max(tz + 0.012, top_z - min(0.5 * size, 0.02))
    report.setdefault("grasp_targets", []).append({"x": r4(gx), "y": r4(gy), "z": r4(grasp_z),
                                                   "top_z": r4(top_z), "table_z": r4(tz),
                                                   "size": r4(size)})
    move(label + "-hover2", [gx, gy, grasp_z + 0.10], quat, 0.01, 100)
    move(label + "-descend", [gx, gy, grasp_z], quat, 0.008, 100)
    grip(label + "-close", True)
    g_closed = gripper_now()
    transit_z = grasp_z + 0.18
    if container is not None:
        transit_z = max(transit_z, container["rim_z"] + 0.15)
    move(label + "-lift", [gx, gy, transit_z], quat, 0.02, 120)
    g_lift = gripper_now()
    held = g_lift > HOLD_THRESHOLD
    report.setdefault("grasp_checks", []).append({"cube": cube_index, "closed": r4(g_closed),
                                                 "after_lift": r4(g_lift), "held": held})
    if state_flags["ended"]:
        return "ended"
    if not held:
        grip(label + "-reopen", False)
        return "missed"
    if container is None:
        return "no_container"
    bx = container["cx"]
    by = container["cy"]
    move(label + "-to-container", [bx, by, transit_z], quat, 0.02, 200)
    if state_flags["ended"]:
        return "ended"
    try:
        if wrist_cam != "":
            wc = None
            prompt = container["prompt"]
            for c in seg(wrist_cam, prompt, 5):
                pts = cloud_points(c, wrist_cam)
                if len(pts) < 20:
                    continue
                s = summarize(pts)
                if s["ex"] < 0.06 and s["ey"] < 0.06:
                    continue
                if abs(s["cx"] - bx) < 0.06 and abs(s["cy"] - by) < 0.06:
                    wc = s
                    break
            if wc is not None:
                report.setdefault("container_wrist_refine", []).append(
                    {"cx": r4(wc["cx"]), "cy": r4(wc["cy"]), "rim_z": r4(wc["z95"]),
                     "dx": r4(wc["cx"] - bx), "dy": r4(wc["cy"] - by)})
                bx = 0.5 * (bx + wc["cx"])
                by = 0.5 * (by + wc["cy"])
    except Exception as exc:
        note_error("container-wrist", exc)
    release_z = container["rim_z"] + 0.06
    move(label + "-lower", [bx, by, release_z], quat, 0.02, 120)
    g_release = gripper_now()
    grip(label + "-release", False)
    hold(label + "-settle", 5, 1.0)
    move(label + "-retreat", [bx, by, transit_z], quat, 0.02, 120)
    report.setdefault("release_checks", []).append({"cube": cube_index, "grip_before_release": r4(g_release),
                                                   "held_at_release": g_release > HOLD_THRESHOLD})
    if g_release > HOLD_THRESHOLD:
        return "placed"
    return "dropped_in_transit"


placements = []
try:
    queue = []
    for req in goal["requirements"]:
        queue.append(req)
    for req in queue:
        colour = req["color"]
        wanted = req["count"]
        if wanted < 0:
            first, tz0, matching0 = choose_cube(colour, "count-all")
            wanted = matching0 if matching0 > 0 else 1
            req["count_resolved"] = wanted
        placed = 0
        attempts = 0
        while placed < wanted and attempts < wanted + 3 and not state_flags["ended"]:
            if steps_left() < 150:
                report["stop_reason"] = "step budget"
                break
            attempts += 1
            if attempts > 1 or len(placements) > 0:
                go_observe("observe-" + str(len(placements)))
            outcome = "error"
            try:
                outcome = pick_and_place(colour, len(placements))
            except Exception as exc:
                note_error("pick_and_place", exc)
            placements.append({"color": colour, "outcome": outcome, "t_steps": steps_used()})
            print("PLACEMENT " + str(placements[-1]))
            if outcome == "placed":
                placed += 1
            if outcome == "no_target" or outcome == "no_container":
                break
        req["placed"] = placed
        req["attempts"] = attempts
except Exception as exc:
    note_error("loop", exc)
report["placements"] = placements
report["phase_done"].append("pick_place")

# ------------------------------------------------------------------ button
try:
    if goal["press_button"] and not state_flags["ended"]:
        go_observe("observe-button")
        # Re-localize from the observation pose: the arm or a placed cube may have moved it.
        fresh = locate_button()
        if fresh is not None:
            button = fresh
            report["button_fresh"] = {"prompt": button["prompt"], "cx": r4(button["cx"]),
                                      "cy": r4(button["cy"]), "top_z": r4(button["top_z"])}
        if button is not None:
            bx = button["cx"]
            by = button["cy"]
            top = button["top_z"]
            grip("button-close", True)
            move("button-hover", [bx, by, top + 0.10], down_quat, 0.01, 150)
            try:
                if wrist_cam != "":
                    for c in seg(wrist_cam, button["prompt"], 5):
                        pts = cloud_points(c, wrist_cam)
                        if len(pts) < 6:
                            continue
                        s = summarize(pts)
                        if s["ex"] > 0.15 or s["ey"] > 0.15:
                            continue
                        if abs(s["cx"] - bx) < 0.04 and abs(s["cy"] - by) < 0.04:
                            report["button_wrist"] = {"cx": r4(s["cx"]), "cy": r4(s["cy"]),
                                                      "top_z": r4(s["z95"])}
                            bx = s["cx"]
                            by = s["cy"]
                            top = 0.5 * (top + s["z95"])
                            break
            except Exception as exc:
                note_error("button-wrist", exc)
            move("button-approach", [bx, by, top + 0.02], down_quat, 0.01, 80)
            move("button-press", [bx, by, top - 0.02], down_quat, 0.005, 40)
            hold("button-hold", 5, 0.0)
            move("button-retreat", [bx, by, top + 0.10], down_quat, 0.02, 80)
        else:
            report["button_status"] = "button not found"
except Exception as exc:
    note_error("button-press", exc)
report["phase_done"].append("button")

# ------------------------------------------------------------------ wrap up
try:
    if not state_flags["ended"]:
        hold("final-settle", 3, gripper_now() if gripper_now() >= 0 else 1.0)
    report["final_steps_used"] = steps_used()
    report["ended"] = state_flags["ended"]
    report["end_reason"] = state_flags["end_reason"]
    report["final_ee"] = [r4(v) for v in ee_now()]
except Exception as exc:
    note_error("wrapup", exc)
print("DONE: placements=" + str(len(placements)) + " ended=" + str(state_flags["ended"]))
result = report
