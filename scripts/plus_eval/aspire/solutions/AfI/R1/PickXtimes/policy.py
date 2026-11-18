# PickXtimes v00 — snapshot phase + language-adaptive "pick the object X times" policy.
# AfI, single simulator start: this program is both the scene snapshot and the attempt.
# Every phase is wrapped in try/except so the program never raises.
# Only public APIs; no world coordinates, dimensions or pixel locations are hardcoded:
# every object pose comes from SAM3 masks projected through public depth + calibration.
import math

report = {"version": "v00", "phases": {}, "errors": [], "log": []}
result = report

STATE = {"over": False, "hz": 20.0, "max_steps": None, "exec_samples": 0}


def note(msg):
    text = str(msg)
    report["log"].append(text[:400])
    print("[v00] " + text[:400])


def err(where, exc):
    text = where + ": " + str(exc)[:400]
    report["errors"].append(text)
    print("[v00][ERR] " + text)


def safe_str(value, limit=300):
    try:
        return str(value)[:limit]
    except Exception:
        return "<unprintable>"


def as_float_list(seq):
    out = []
    for value in seq:
        out.append(float(value))
    return out


def quantile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    index = int((len(ordered) - 1) * fraction)
    return ordered[index]


def ticks_now():
    try:
        return int(round(float(get_robot_state().timestamp_s) * STATE["hz"]))
    except Exception:
        return -1


def is_terminal(res):
    flag = False
    for name in ("terminated", "truncated", "done", "episode_over"):
        try:
            if name == "terminated" and bool(res.terminated):
                flag = True
            if name == "truncated" and bool(res.truncated):
                flag = True
            if name == "done" and bool(res.done):
                flag = True
            if name == "episode_over" and bool(res.episode_over):
                flag = True
        except Exception:
            pass
    try:
        status = str(res.status).lower()
        if "terminal" in status or "terminated" in status or "episode_end" in status:
            flag = True
    except Exception:
        pass
    return flag


def record_exec(label, res):
    entry = {"label": label}
    try:
        entry["ok"] = bool(res.ok)
    except Exception:
        entry["ok"] = None
    try:
        entry["status"] = str(res.status)
    except Exception:
        entry["status"] = None
    try:
        entry["steps"] = int(res.steps_executed)
    except Exception:
        entry["steps"] = None
    try:
        entry["error"] = safe_str(res.error, 300)
    except Exception:
        pass
    entry["terminal"] = is_terminal(res)
    entry["tick"] = ticks_now()
    if STATE["exec_samples"] < 3:
        STATE["exec_samples"] = STATE["exec_samples"] + 1
        entry["repr"] = safe_str(res, 900)
    report["motions"].append(entry)
    note(label + " ok=" + str(entry["ok"]) + " status=" + str(entry["status"]) +
         " steps=" + str(entry["steps"]) + " tick=" + str(entry["tick"]))
    if entry["terminal"]:
        STATE["over"] = True
        note("episode reported terminal after " + label)
    return entry


report["motions"] = []

# ---------------------------------------------------------------- quaternions
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def quat_norm(q):
    n = math.sqrt(sum(v * v for v in q))
    if n < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(v / n for v in q)


def quat_matrix(q):
    w, x, y, z = quat_norm(q)
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def yaw_quat(angle):
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def wrap_quarter(angle):
    # wrap into [-pi/4, pi/4) modulo pi/2 (cube faces repeat every 90 degrees)
    quarter = math.pi / 2.0
    value = angle
    for guard in range(20):
        if value >= quarter / 2.0:
            value = value - quarter
        elif value < -quarter / 2.0:
            value = value + quarter
        else:
            break
    return value


# ---------------------------------------------------------------- phase 1: snapshot
base_frame = "robot_base"
language = ""
try:
    context = get_task_context()
    language = str(context.language)
    report["task_language"] = language
    try:
        report["task_name"] = str(context.task_name)
        report["task_id"] = safe_str(context.task_id)
        report["suite"] = safe_str(context.suite)
        report["family"] = safe_str(context.family)
    except Exception as exc:
        err("context fields", exc)
    note("language: " + language)
except Exception as exc:
    err("get_task_context", exc)

try:
    meta = robomme_plus.get_task_metadata()
    meta_out = {}
    for key in meta:
        meta_out[str(key)] = safe_str(meta[key], 400)
    report["task_metadata"] = meta_out
    try:
        STATE["max_steps"] = int(meta["max_steps"])
    except Exception as exc:
        err("max_steps", exc)
    for key in ("control_hz", "control_frequency", "frequency_hz", "control_freq"):
        try:
            STATE["hz"] = float(meta[key])
            break
        except Exception:
            pass
    if not language:
        try:
            language = str(meta["language"])
        except Exception:
            pass
except Exception as exc:
    err("get_task_metadata", exc)

try:
    cmeta = robomme_plus.get_controller_metadata()
    cm_out = {}
    for key in cmeta:
        cm_out[str(key)] = safe_str(cmeta[key], 400)
    report["controller_metadata"] = cm_out
except Exception as exc:
    err("get_controller_metadata", exc)

state0 = None
try:
    state0 = get_robot_state()
    base_frame = str(state0.base_frame)
    ee0 = state0.end_effector_poses["primary"]
    report["robot_state0"] = {
        "base_frame": base_frame,
        "embodiment": safe_str(state0.embodiment),
        "joints": as_float_list(state0.joint_positions["primary"]),
        "gripper": float(state0.gripper_positions["primary"]),
        "ee_position": as_float_list(ee0.position),
        "ee_quat_wxyz": as_float_list(ee0.quaternion_wxyz),
        "ee_frame": safe_str(ee0.frame),
        "timestamp_s": float(state0.timestamp_s),
    }
except Exception as exc:
    err("get_robot_state", exc)

camera_names = []
try:
    obs0 = get_observation()
    cams = {}
    for name in obs0.cameras:
        camera_names.append(str(name))
        cam = obs0.cameras[name]
        info = {}
        try:
            info["rgb_shape"] = [int(v) for v in cam.rgb.shape]
            info["depth_shape"] = [int(v) for v in cam.depth_m.shape]
            info["intrinsics"] = [as_float_list(row) for row in cam.intrinsics]
            info["pose_position"] = as_float_list(cam.camera_pose.position)
            info["pose_quat_wxyz"] = as_float_list(cam.camera_pose.quaternion_wxyz)
            info["pose_frame"] = safe_str(cam.camera_pose.frame)
            h = int(cam.depth_m.shape[0])
            w = int(cam.depth_m.shape[1])
            dvals = []
            for yy in range(0, h, 16):
                for xx in range(0, w, 16):
                    dvals.append(float(cam.depth_m[yy][xx]))
            info["depth_q10_q50_q90"] = [quantile(dvals, 0.1), quantile(dvals, 0.5), quantile(dvals, 0.9)]
        except Exception as exc:
            err("camera info " + str(name), exc)
        cams[str(name)] = info
    report["cameras"] = cams
    note("cameras: " + str(camera_names))
except Exception as exc:
    err("get_observation", exc)

FRONT = "front"
WRIST = "wrist"
if camera_names and FRONT not in camera_names:
    FRONT = camera_names[0]
if camera_names and WRIST not in camera_names:
    WRIST = camera_names[-1]

# conditioning video summary (cheap: robot-state series + a coarse colour grid on a few frames)
video_pick_count = None
try:
    video = list(robomme_plus.get_conditioning_video())
    nvid = len(video)
    vsum = {"frames": nvid}
    if nvid > 0:
        vsum["t_first"] = float(video[0].timestamp_s)
        vsum["t_last"] = float(video[nvid - 1].timestamp_s)
        grip_series = []
        z_series = []
        for frame in video:
            try:
                rs = frame.robot_state
                grip_series.append(round(float(rs.gripper_positions["primary"]), 3))
                z_series.append(round(float(rs.end_effector_poses["primary"].position[2]), 3))
            except Exception:
                pass
        stride = max(1, len(grip_series) // 64)
        vsum["gripper_series_subsampled"] = grip_series[::stride]
        vsum["ee_z_series_subsampled"] = z_series[::stride]
        # count close events with hysteresis (closed < 0.6, reopened > 0.8)
        closed = False
        closes = 0
        for g in grip_series:
            if (not closed) and g < 0.6:
                closed = True
                closes = closes + 1
            elif closed and g > 0.8:
                closed = False
        vsum["gripper_close_events"] = closes
        video_pick_count = closes
        grids = []
        picks = [0, nvid // 2, nvid - 1]
        for index in picks:
            try:
                rgb = video[index].cameras[FRONT].rgb
                h = int(rgb.shape[0])
                w = int(rgb.shape[1])
                acc = [0.0, 0.0, 0.0]
                count = 0
                for yy in range(0, h, 32):
                    for xx in range(0, w, 32):
                        px = rgb[yy][xx]
                        acc[0] += float(px[0])
                        acc[1] += float(px[1])
                        acc[2] += float(px[2])
                        count += 1
                grids.append({"index": index, "mean_rgb": [round(v / max(1, count), 1) for v in acc]})
            except Exception as exc:
                err("video grid", exc)
        vsum["front_mean_rgb"] = grids
    report["conditioning_video"] = vsum
    note("conditioning video frames: " + str(nvid))
except Exception as exc:
    err("get_conditioning_video", exc)

# ---------------------------------------------------------------- language parsing
COLOR_WORDS = ("red", "green", "blue", "yellow", "orange", "purple", "pink", "cyan",
               "white", "black", "gray", "grey", "brown", "magenta", "violet")
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10, "single": 1}
OBJECT_NOUNS = ("cube", "block", "box", "ball", "sphere", "peg", "can", "bottle", "object")
TARGET_NOUNS = ("target", "plate", "goal", "region", "area", "marker", "disk", "disc",
                "circle", "pad", "zone", "spot", "bin", "square", "mat", "platform")

lang = language.lower()
clean = ""
for ch in lang:
    if ch.isalnum():
        clean += ch
    else:
        clean += " "
tokens = clean.split()

count_n = None
count_source = None
for idx in range(len(tokens)):
    tok = tokens[idx]
    if tok == "once":
        count_n = 1
        count_source = "once"
        break
    if tok == "twice":
        count_n = 2
        count_source = "twice"
        break
    if tok == "thrice":
        count_n = 3
        count_source = "thrice"
        break
    if tok in ("times", "time") and idx > 0:
        prev = tokens[idx - 1]
        if prev.isdigit():
            count_n = int(prev)
            count_source = prev + " " + tok
            break
        if prev in NUMBER_WORDS:
            count_n = NUMBER_WORDS[prev]
            count_source = prev + " " + tok
            break
if count_n is None:
    for tok in tokens:
        if len(tok) >= 2 and (tok[0] == "x" and tok[1:].isdigit()):
            count_n = int(tok[1:])
            count_source = tok
            break
        if len(tok) >= 2 and (tok[-1] == "x" and tok[:-1].isdigit()):
            count_n = int(tok[:-1])
            count_source = tok
            break
if count_n is None and video_pick_count is not None and video_pick_count > 0:
    count_n = video_pick_count
    count_source = "conditioning-video gripper close events"
if count_n is None:
    count_n = 1
    count_source = "default (no count found)"
if count_n > 10:
    count_n = 10

object_noun = "cube"
object_index = -1
for idx in range(len(tokens)):
    if tokens[idx] in OBJECT_NOUNS or (tokens[idx].endswith("s") and tokens[idx][:-1] in OBJECT_NOUNS):
        object_noun = tokens[idx]
        if object_noun.endswith("s") and object_noun[:-1] in OBJECT_NOUNS:
            object_noun = object_noun[:-1]
        object_index = idx
        break

object_color = None
if object_index >= 0:
    for back in range(1, 4):
        j = object_index - back
        if j >= 0 and tokens[j] in COLOR_WORDS:
            object_color = tokens[j]
            break
if object_color is None:
    for tok in tokens:
        if tok in COLOR_WORDS:
            object_color = tok
            break
if object_color == "grey":
    object_color = "gray"

target_phrase = None
target_color = None
STOP = ("and", "then", "after", "before", "repeat", "until", "again", "times", "time", "once",
        "twice", "thrice", "press", "push", "finally", "it")
for idx in range(len(tokens) - 1):
    if tokens[idx] in ("on", "onto", "into", "in", "to", "at", "inside"):
        words = []
        for j in range(idx + 1, min(len(tokens), idx + 6)):
            if tokens[j] in STOP:
                break
            words.append(tokens[j])
        has_target = False
        for wd in words:
            if wd in TARGET_NOUNS:
                has_target = True
        if has_target:
            kept = []
            for wd in words:
                if wd not in ("the", "a", "an"):
                    kept.append(wd)
                if wd in TARGET_NOUNS:
                    break
            target_phrase = " ".join(kept)
            for wd in kept:
                if wd in COLOR_WORDS:
                    target_color = wd
            break

press_button = "button" in tokens or "buttons" in tokens
button_color = None
if press_button:
    for idx in range(len(tokens)):
        if tokens[idx] in ("button", "buttons"):
            for back in range(1, 3):
                j = idx - back
                if j >= 0 and tokens[j] in COLOR_WORDS:
                    button_color = tokens[j]
            break

plan = {"count": count_n, "count_source": count_source, "object_noun": object_noun,
        "object_color": object_color, "target_phrase": target_phrase,
        "target_color": target_color, "press_button": press_button,
        "button_color": button_color, "video_close_events": video_pick_count,
        "tokens": tokens[:60]}
report["plan"] = plan
note("plan: " + str(plan))

# ---------------------------------------------------------------- perception helpers
HUES = {"red": 0.0, "orange": 30.0, "yellow": 58.0, "green": 120.0, "cyan": 185.0,
        "blue": 225.0, "purple": 275.0, "violet": 275.0, "magenta": 305.0, "pink": 330.0,
        "brown": 25.0}


def rgb_to_hsv(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx - mn
    if diff < 1e-6:
        hue = 0.0
    elif mx == r:
        hue = (60.0 * ((g - b) / diff) + 360.0) % 360.0
    elif mx == g:
        hue = 60.0 * ((b - r) / diff) + 120.0
    else:
        hue = 60.0 * ((r - g) / diff) + 240.0
    sat = 0.0 if mx < 1e-6 else diff / mx
    return hue, sat, mx


def color_score(rgb, color):
    if color is None or rgb is None:
        return 0.5
    hue, sat, val = rgb_to_hsv(rgb[0], rgb[1], rgb[2])
    if color == "white":
        return max(0.0, min(1.0, val * (1.0 - sat)))
    if color == "black":
        return max(0.0, min(1.0, 1.0 - val))
    if color == "gray":
        return max(0.0, min(1.0, (1.0 - sat) * (1.0 - abs(val - 0.5) * 1.5)))
    if color not in HUES:
        return 0.5
    dist = abs(hue - HUES[color])
    if dist > 180.0:
        dist = 360.0 - dist
    hue_term = max(0.0, 1.0 - dist / 60.0)
    sat_term = min(1.0, sat / 0.35)
    return hue_term * sat_term


def cloud_points(pc):
    pts = []
    try:
        raw = pc.points
    except Exception:
        return pts
    for p in raw:
        try:
            x = float(p[0])
            y = float(p[1])
            z = float(p[2])
        except Exception:
            continue
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
            pts.append((x, y, z))
    return pts


def cloud_color(pc):
    raw = None
    for attempt in range(2):
        try:
            if attempt == 0:
                raw = pc.colors
            else:
                raw = pc.rgb
            if raw is not None:
                break
        except Exception:
            raw = None
    if raw is None:
        return None
    rs = []
    gs = []
    bs = []
    try:
        for c in raw:
            rs.append(float(c[0]))
            gs.append(float(c[1]))
            bs.append(float(c[2]))
    except Exception:
        return None
    if len(rs) == 0:
        return None
    r = quantile(rs, 0.5)
    g = quantile(gs, 0.5)
    b = quantile(bs, 0.5)
    scale = 255.0 if max(max(rs), max(gs), max(bs)) > 1.5 else 1.0
    return (r / scale, g / scale, b / scale)


def mask_color(cand, camera, obs):
    # fallback: median image colour inside the mask, restricted to its bounding box
    try:
        rgb = obs.cameras[camera].rgb
        mask = cand.mask
        x1, y1, x2, y2 = [int(float(v)) for v in cand.box_xyxy]
        h = int(rgb.shape[0])
        w = int(rgb.shape[1])
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)
        rs = []
        gs = []
        bs = []
        step_px = max(1, (x2 - x1) // 24)
        for yy in range(y1, y2 + 1, step_px):
            for xx in range(x1, x2 + 1, step_px):
                if bool(mask[yy][xx]):
                    px = rgb[yy][xx]
                    rs.append(float(px[0]))
                    gs.append(float(px[1]))
                    bs.append(float(px[2]))
        if len(rs) == 0:
            return None
        return (quantile(rs, 0.5) / 255.0, quantile(gs, 0.5) / 255.0, quantile(bs, 0.5) / 255.0)
    except Exception:
        return None


def project(cand, camera):
    try:
        return mask_to_point_cloud(cand, camera, target_frame=base_frame)
    except Exception as exc1:
        try:
            return mask_to_point_cloud(cand.mask, camera, target_frame=base_frame)
        except Exception as exc2:
            err("mask_to_point_cloud", exc2)
            return None


def describe(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    return {
        "n": len(pts),
        "x": [quantile(xs, 0.1), quantile(xs, 0.5), quantile(xs, 0.9)],
        "y": [quantile(ys, 0.1), quantile(ys, 0.5), quantile(ys, 0.9)],
        "z": [quantile(zs, 0.05), quantile(zs, 0.5), quantile(zs, 0.95)],
    }


def find_object(camera, prompts, color, max_xy_span, label, max_cands=5):
    """Segment every prompt, project each top candidate, reject implausible sizes,
    and rank by colour match (when a colour was named) plus SAM score."""
    best = None
    tried = []
    view = None
    try:
        view = get_observation()
    except Exception as exc:
        err("find_object observation", exc)
    for prompt in prompts:
        try:
            seg = segment_text(camera, prompt)
        except Exception as exc:
            err("segment_text " + prompt, exc)
            continue
        try:
            ok = bool(seg.ok)
        except Exception:
            ok = False
        if not ok:
            tried.append({"prompt": prompt, "ok": False})
            continue
        cands = list(seg.segmentations)[:max_cands]
        for rank in range(len(cands)):
            cand = cands[rank]
            entry = {"prompt": prompt, "rank": rank}
            try:
                entry["score"] = round(float(cand.score), 3)
            except Exception:
                entry["score"] = 0.0
            try:
                entry["box"] = [round(float(v), 1) for v in cand.box_xyxy]
            except Exception:
                entry["box"] = None
            pc = project(cand, camera)
            if pc is None:
                entry["reject"] = "no cloud"
                tried.append(entry)
                continue
            pts = cloud_points(pc)
            if len(pts) < 8:
                entry["reject"] = "few points " + str(len(pts))
                tried.append(entry)
                continue
            d = describe(pts)
            entry["geom"] = d
            span = max(d["x"][2] - d["x"][0], d["y"][2] - d["y"][0])
            if span > max_xy_span:
                entry["reject"] = "span " + str(round(span, 3))
                tried.append(entry)
                continue
            rgb = cloud_color(pc)
            if rgb is None and view is not None:
                rgb = mask_color(cand, camera, view)
            entry["rgb"] = None if rgb is None else [round(v, 3) for v in rgb]
            cscore = color_score(rgb, color)
            entry["color_score"] = round(cscore, 3)
            total = entry["score"] + (2.0 * cscore if color is not None else 0.0)
            entry["total"] = round(total, 3)
            tried.append(entry)
            if best is None or total > best["total"]:
                best = {"total": total, "pts": pts, "geom": d, "prompt": prompt, "rank": rank,
                        "color_score": cscore, "rgb": rgb, "camera": camera}
        if best is not None and (color is None or best["color_score"] > 0.6) and best["total"] > 0.5:
            break
    report["perception"].append({"label": label, "camera": camera, "candidates": tried[:12],
                                 "chosen": None if best is None else
                                 {"prompt": best["prompt"], "rank": best["rank"],
                                  "total": round(best["total"], 3), "geom": best["geom"]}})
    return best


report["perception"] = []


def top_face_fit(pts):
    """Top-face centre and a yaw of one face normal from a min-area rectangle search."""
    zs = [p[2] for p in pts]
    top = quantile(zs, 0.95)
    bottom = quantile(zs, 0.03)
    face = [p for p in pts if p[2] > top - 0.012]
    if len(face) < 6:
        face = pts
    best = None
    for deg in range(0, 90, 5):
        a = math.radians(deg)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [p[0] * ca + p[1] * sa for p in face]
        vs = [-p[0] * sa + p[1] * ca for p in face]
        u_lo = quantile(us, 0.05)
        u_hi = quantile(us, 0.95)
        v_lo = quantile(vs, 0.05)
        v_hi = quantile(vs, 0.95)
        area = (u_hi - u_lo) * (v_hi - v_lo)
        if best is None or area < best[0]:
            uc = 0.5 * (u_lo + u_hi)
            vc = 0.5 * (v_lo + v_hi)
            best = (area, a, uc * ca - vc * sa, uc * sa + vc * ca, u_hi - u_lo, v_hi - v_lo)
    return {"top_z": top, "bottom_z": bottom, "yaw": best[1], "cx": best[2], "cy": best[3],
            "face_extent": [best[4], best[5]], "face_points": len(face)}


# ---------------------------------------------------------------- probes on both cameras
probe_prompts_front = ["cube", "block"]
if object_color is not None:
    probe_prompts_front.insert(0, object_color + " " + object_noun)
for extra in ("red cube", "green cube", "blue cube", "button", "target", "table"):
    if extra not in probe_prompts_front:
        probe_prompts_front.append(extra)
probes = []
for cam_name in (FRONT, WRIST):
    plist = probe_prompts_front if cam_name == FRONT else probe_prompts_front[:4]
    for prompt in plist:
        entry = {"camera": cam_name, "prompt": prompt, "ok": False, "n": 0, "scores": [], "boxes": []}
        try:
            seg = segment_text(cam_name, prompt)
            entry["ok"] = bool(seg.ok)
            if entry["ok"]:
                cands = list(seg.segmentations)
                entry["n"] = len(cands)
                for cand in cands[:5]:
                    entry["scores"].append(round(float(cand.score), 3))
                    try:
                        entry["boxes"].append([round(float(v), 1) for v in cand.box_xyxy])
                    except Exception:
                        entry["boxes"].append(None)
            else:
                try:
                    entry["error"] = safe_str(seg.error, 200)
                except Exception:
                    pass
        except Exception as exc:
            entry["exception"] = safe_str(exc, 200)
        probes.append(entry)
report["prompt_probes"] = probes
note("probes done: " + str(len(probes)))

# ---------------------------------------------------------------- motion helpers
down_q = (0.0, 1.0, 0.0, 0.0)
try:
    q0 = tuple(as_float_list(state0.end_effector_poses["primary"].quaternion_wxyz))
    m0 = quat_matrix(q0)
    report["ee0_tool_z_axis"] = [m0[0][2], m0[1][2], m0[2][2]]
    if m0[2][2] < -0.9:
        down_q = quat_norm(q0)
        note("using measured home orientation as top-down")
    else:
        note("home tool axis not downward; using canonical (0,1,0,0)")
except Exception as exc:
    err("down quaternion", exc)


def closing_axis_yaw(q):
    m = quat_matrix(q)
    return math.atan2(m[1][1], m[0][1])


def grasp_quat(face_yaw):
    base_yaw = closing_axis_yaw(down_q)
    delta = wrap_quarter(face_yaw - base_yaw)
    return quat_norm(quat_mul(yaw_quat(delta), down_q)), delta


def go(label, pos, quat, tol=0.015, max_steps=200):
    if STATE["over"]:
        note("skip " + label + " (episode over)")
        return False
    target = Pose([float(pos[0]), float(pos[1]), float(pos[2])], [float(v) for v in quat], base_frame)
    ok = False
    try:
        res = move_to_pose(target, tolerance=tol, max_steps=max_steps)
        entry = record_exec(label, res)
        ok = bool(entry["ok"])
        if (not ok) and (not STATE["over"]) and entry["steps"] == 0:
            res = move_to_pose(target, tolerance=max(tol, 0.03), max_steps=max_steps + 150)
            entry = record_exec(label + "/retry", res)
            ok = bool(entry["ok"])
    except Exception as exc:
        err("move_to_pose " + label, exc)
    return ok


def gripper(label, opening):
    if STATE["over"]:
        return None
    try:
        if opening:
            res = open_gripper()
        else:
            res = close_gripper()
        record_exec(label, res)
    except Exception as exc:
        err(label, exc)
    try:
        return float(get_robot_state().gripper_positions["primary"])
    except Exception:
        return None


def ee_position():
    try:
        return as_float_list(get_robot_state().end_effector_poses["primary"].position)
    except Exception:
        return None


def hold_ticks(count, grip):
    for tick in range(count):
        if STATE["over"]:
            break
        try:
            st = get_robot_state()
            reply = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=st.joint_positions["primary"],
                gripper_position=grip, embodiment=st.embodiment)}))
            if is_terminal(reply):
                STATE["over"] = True
                note("terminal during hold tick " + str(tick))
        except Exception as exc:
            err("hold tick", exc)
            break


# ---------------------------------------------------------------- object localisation
obj_prompts = []
if object_color is not None:
    obj_prompts.append(object_color + " " + object_noun)
obj_prompts.append(object_noun)
if object_noun != "cube":
    obj_prompts.append("cube")
obj_prompts.append("block")


def locate_object(label, camera):
    found = find_object(camera, obj_prompts, object_color, 0.10, label)
    if found is None:
        return None
    fit = top_face_fit(found["pts"])
    fit["camera"] = camera
    fit["prompt"] = found["prompt"]
    fit["color_score"] = found["color_score"]
    return fit


target_fit = None
if target_phrase is not None:
    try:
        tprompts = [target_phrase]
        for extra in ("target", "circular target", "goal region"):
            if extra not in tprompts:
                tprompts.append(extra)
        found_t = find_object(FRONT, tprompts, target_color, 0.30, "placement target")
        if found_t is not None:
            tz = [p[2] for p in found_t["pts"]]
            target_fit = {"x": found_t["geom"]["x"][1], "y": found_t["geom"]["y"][1],
                          "top_z": quantile(tz, 0.9)}
        report["target_fit"] = target_fit
        note("target fit: " + str(target_fit))
    except Exception as exc:
        err("target localisation", exc)

button_fit = None
if press_button:
    try:
        bprompts = []
        if button_color is not None:
            bprompts.append(button_color + " button")
        bprompts.append("button")
        bprompts.append("push button")
        bprompts.append("round button")
        found_b = find_object(FRONT, bprompts, button_color, 0.20, "button")
        if found_b is not None:
            bz = [p[2] for p in found_b["pts"]]
            button_fit = {"x": found_b["geom"]["x"][1], "y": found_b["geom"]["y"][1],
                          "top_z": quantile(bz, 0.95), "base_z": quantile(bz, 0.05)}
        report["button_fit"] = button_fit
        note("button fit (pre-task): " + str(button_fit))
    except Exception as exc:
        err("button localisation", exc)

# ---------------------------------------------------------------- pick X times
cycles = []
report["cycles"] = cycles
completed = 0
table_z_est = None
LIFT_ABOVE_TABLE = 0.18
HOVER = 0.12

for cycle in range(count_n + 2):  # up to two spare cycles for missed grasps only
    if completed >= count_n or STATE["over"]:
        break
    info = {"cycle": cycle, "tick_start": ticks_now()}
    cycles.append(info)
    try:
        fit = locate_object("object front c" + str(cycle), FRONT)
        info["front_fit"] = fit
        if fit is None:
            # arm may occlude the object; retreat up and retry with both cameras
            pos = ee_position()
            if pos is not None:
                go("raise for view", (pos[0], pos[1], pos[2] + 0.10), down_q, 0.03, 150)
            fit = locate_object("object front retry c" + str(cycle), FRONT)
            if fit is None:
                fit = locate_object("object wrist retry c" + str(cycle), WRIST)
            info["front_fit_retry"] = fit
        if fit is None:
            note("object not found; stopping cycles")
            break
        if table_z_est is None:
            table_z_est = fit["bottom_z"]
        height = fit["top_z"] - table_z_est
        if height < 0.015 or height > 0.10:
            height = 0.04
            if cycle == 0 or table_z_est > fit["top_z"] - 0.015:
                table_z_est = fit["top_z"] - height
        info["height_est"] = height
        quat, delta = grasp_quat(fit["yaw"])
        hover_z = fit["top_z"] + HOVER
        go("hover c" + str(cycle), (fit["cx"], fit["cy"], hover_z), quat, 0.02, 220)
        # refine from the wrist camera, looking down from the hover pose
        wfit = locate_object("object wrist c" + str(cycle), WRIST)
        info["wrist_fit"] = wfit
        if wfit is not None:
            dist = math.sqrt((wfit["cx"] - fit["cx"]) ** 2 + (wfit["cy"] - fit["cy"]) ** 2)
            info["wrist_front_xy_dist"] = dist
            if dist < 0.06 and abs(wfit["top_z"] - fit["top_z"]) < 0.04:
                fit = {"cx": wfit["cx"], "cy": wfit["cy"], "top_z": wfit["top_z"],
                       "yaw": wfit["yaw"], "bottom_z": fit["bottom_z"]}
                quat, delta = grasp_quat(fit["yaw"])
                info["used"] = "wrist"
            else:
                info["used"] = "front (wrist disagreed)"
        else:
            info["used"] = "front"
        info["yaw_delta_deg"] = round(math.degrees(delta), 1)
        grasp_z = fit["top_z"] - min(0.02, 0.5 * height)
        info["grasp_xyz"] = [fit["cx"], fit["cy"], grasp_z]
        g_open = gripper("open c" + str(cycle), True)
        go("pregrasp c" + str(cycle), (fit["cx"], fit["cy"], fit["top_z"] + 0.07), quat, 0.015, 150)
        go("grasp c" + str(cycle), (fit["cx"], fit["cy"], grasp_z), quat, 0.01, 120)
        info["ee_at_grasp"] = ee_position()
        g_closed = gripper("close c" + str(cycle), False)
        info["gripper_closed"] = g_closed
        lift_z = table_z_est + LIFT_ABOVE_TABLE
        go("lift c" + str(cycle), (fit["cx"], fit["cy"], lift_z), quat, 0.02, 150)
        g_lift = None
        try:
            g_lift = float(get_robot_state().gripper_positions["primary"])
        except Exception:
            pass
        info["gripper_after_lift"] = g_lift
        holding = g_lift is not None and g_lift > 0.04
        info["holding"] = holding
        if not holding:
            note("grasp appears empty in cycle " + str(cycle) + " (gripper=" + str(g_lift) + ")")
            gripper("reopen after miss c" + str(cycle), True)
            continue
        hold_ticks(4, 0.0)
        # put down: on the named target if any, else back where it was picked
        if target_fit is not None:
            place_xy = (target_fit["x"], target_fit["y"])
            place_z = target_fit["top_z"] + (grasp_z - table_z_est) + 0.01
            go("transit c" + str(cycle), (place_xy[0], place_xy[1], lift_z), quat, 0.02, 220)
        else:
            place_xy = (fit["cx"], fit["cy"])
            place_z = grasp_z + 0.008
        go("lower c" + str(cycle), (place_xy[0], place_xy[1], place_z), quat, 0.012, 150)
        gripper("release c" + str(cycle), True)
        go("retreat c" + str(cycle), (place_xy[0], place_xy[1], place_z + 0.10), quat, 0.02, 120)
        completed = completed + 1
        info["completed"] = True
        info["tick_end"] = ticks_now()
        note("cycle " + str(cycle) + " done; completed=" + str(completed) + "/" + str(count_n))
    except Exception as exc:
        err("cycle " + str(cycle), exc)

report["completed_picks"] = completed

# ---------------------------------------------------------------- final button press
if press_button and not STATE["over"]:
    try:
        # re-localise now that the arm has moved (the earlier view may have been occluded)
        bprompts = []
        if button_color is not None:
            bprompts.append(button_color + " button")
        bprompts.append("button")
        bprompts.append("push button")
        found_b = find_object(FRONT, bprompts, button_color, 0.20, "button final")
        if found_b is not None:
            bz = [p[2] for p in found_b["pts"]]
            button_fit = {"x": found_b["geom"]["x"][1], "y": found_b["geom"]["y"][1],
                          "top_z": quantile(bz, 0.95), "base_z": quantile(bz, 0.05)}
        report["button_fit_final"] = button_fit
        if button_fit is not None:
            gripper("close for press", False)
            bq = down_q
            bx = button_fit["x"]
            by = button_fit["y"]
            btop = button_fit["top_z"]
            go("button hover", (bx, by, btop + 0.10), bq, 0.02, 220)
            go("button touch", (bx, by, btop + 0.01), bq, 0.012, 100)
            go("button press", (bx, by, btop - 0.02), bq, 0.01, 60)
            report["ee_at_press"] = ee_position()
            go("button release", (bx, by, btop + 0.10), bq, 0.02, 100)
        else:
            note("button not found")
    except Exception as exc:
        err("button press", exc)

# let the scene settle so the native evaluator sees the final state
try:
    hold_ticks(10, 1.0)
except Exception as exc:
    err("settle", exc)

report["final_tick"] = ticks_now()
report["episode_over_flag"] = STATE["over"]
note("finished: completed=" + str(completed) + " of " + str(count_n) +
     " tick=" + str(report["final_tick"]) + " max_steps=" + str(STATE["max_steps"]))
result = report
