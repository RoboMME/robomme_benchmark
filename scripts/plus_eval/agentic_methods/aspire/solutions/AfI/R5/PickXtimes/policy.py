# v02 -- PickXtimes: v01 + occlusion-gated cube tracking during the servo grasp. v01 ep1 failed when the
# descending hand occluded the cube (seen near the camera): partial/low-score masks and a colour-failed
# fallback mask pulled the servo 5-8 cm off the cube. v02 accepts a servo cube observation only if it is
# colour-checked (no fallback), its box keeps >= 75% of the unoccluded box height and it lies within
# 2.5 cm of the prediction (8 cm before the first fix); otherwise it dead-reckons. Finish (close) requires < 2 cm alignment.
# v01: per-tick visual servo for a MOVING target disc (and a cube riding it); static scenes keep v00.
# Everything derived at runtime from the goal text and live front-camera perception; only `import math`.
import math

report = {"version": "v02", "timeline": [], "probes": [], "cycles": [], "notes": []}
flags = {"ended": False}

COLOR_HUES = {"red": 0.0, "orange": 30.0, "yellow": 60.0, "green": 120.0, "cyan": 180.0,
              "blue": 225.0, "purple": 280.0, "violet": 280.0, "magenta": 300.0, "pink": 330.0}
ACHROMATIC = ("white", "black", "gray", "grey")
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10}
OBJECT_NOUNS = ("cube", "cubes", "block", "blocks", "box", "ball", "object")
TARGET_WORDS = ("target", "goal", "plate", "platform", "region", "area", "marker", "zone", "pad")


def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def parse_goal(text):
    lowered = str(text).lower()
    cleaned = ""
    for ch in lowered:
        if ch.isalnum() or ch == " ":
            cleaned = cleaned + ch
        else:
            cleaned = cleaned + " "
    words = cleaned.split()
    parsed = {"words": words, "color": None, "noun": "cube", "count": None,
              "button": "button" in words, "target": False, "target_word": None}
    for index, word in enumerate(words):
        if word in COLOR_HUES or word in ACHROMATIC:
            follow = words[index + 1] if index + 1 < len(words) else ""
            if follow in OBJECT_NOUNS:
                parsed["color"] = word
                parsed["noun"] = follow.rstrip("s") if follow.endswith("s") else follow
                break
    if parsed["color"] is None:
        for word in words:
            if word in COLOR_HUES or word in ACHROMATIC:
                parsed["color"] = word
                break
    for index, word in enumerate(words):
        follow = words[index + 1] if index + 1 < len(words) else ""
        value = None
        if word.isdigit():
            value = int(word)
        elif word in NUMBER_WORDS:
            value = NUMBER_WORDS[word]
        if value is not None and follow in ("time", "times"):
            parsed["count"] = value
            break
    if parsed["count"] is None:
        if "once" in words:
            parsed["count"] = 1
        elif "twice" in words:
            parsed["count"] = 2
        elif "thrice" in words:
            parsed["count"] = 3
    for word in words:
        if word in TARGET_WORDS:
            parsed["target"] = True
            parsed["target_word"] = word
            break
    return parsed


def hsv(r, g, b):
    r = float(r)
    g = float(g)
    b = float(b)
    if max(r, g, b) > 1.0:
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0
    high = max(r, g, b)
    low = min(r, g, b)
    delta = high - low
    sat = 0.0 if high <= 0.0 else delta / high
    if delta <= 1e-6:
        hue = 0.0
    elif high == r:
        hue = 60.0 * (((g - b) / delta) % 6.0)
    elif high == g:
        hue = 60.0 * (((b - r) / delta) + 2.0)
    else:
        hue = 60.0 * (((r - g) / delta) + 4.0)
    return hue, sat, high


def color_fraction(colors, name):
    if name is None or not colors:
        return None
    hits = 0
    for c in colors:
        hue, sat, val = hsv(c[0], c[1], c[2])
        if name in COLOR_HUES:
            diff = abs(hue - COLOR_HUES[name]) % 360.0
            diff = min(diff, 360.0 - diff)
            if sat > 0.35 and val > 0.15 and diff < 35.0:
                hits = hits + 1
        elif name == "white":
            if sat < 0.2 and val > 0.7:
                hits = hits + 1
        elif name == "black":
            if val < 0.25:
                hits = hits + 1
        else:
            if sat < 0.2 and 0.25 <= val <= 0.75:
                hits = hits + 1
    return float(hits) / float(len(colors))


def ended(res):
    try:
        if res.terminated or res.truncated:
            flags["ended"] = True
    except Exception:
        pass
    try:
        status = str(res.status)
        if "terminat" in status or "truncat" in status:
            flags["ended"] = True
    except Exception:
        pass
    return flags["ended"]


def now_s():
    try:
        return float(get_robot_state().timestamp_s)
    except Exception:
        return None


def record(label, res, extra=None):
    entry = {"label": label}
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
        entry["error"] = str(res.error) if res.error is not None else None
    except Exception:
        pass
    entry["ended"] = ended(res)
    if not flags["ended"]:
        entry["t"] = now_s()
        try:
            state = get_robot_state()
            entry["ee"] = [round(float(v), 4) for v in state.end_effector_poses["primary"].position]
            entry["grip"] = round(float(state.gripper_positions["primary"]), 3)
        except Exception:
            pass
    if extra is not None:
        entry["extra"] = extra
    report["timeline"].append(entry)
    return res


def cloud_points(cloud):
    try:
        return [(float(p[0]), float(p[1]), float(p[2])) for p in cloud.points]
    except Exception:
        return []


def cloud_colors(cloud):
    try:
        cols = cloud.colors
        if cols is None:
            return []
        return [(float(c[0]), float(c[1]), float(c[2])) for c in cols]
    except Exception:
        return []


def project(candidate, camera, base):
    try:
        return mask_to_point_cloud(candidate, camera, target_frame=base)
    except Exception:
        pass
    try:
        return mask_to_point_cloud(candidate.mask, camera, target_frame=base)
    except Exception:
        return None


def stats_of(points):
    if len(points) < 3:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return {"n": len(points),
            "x": quantile(xs, 0.5), "y": quantile(ys, 0.5),
            "z05": quantile(zs, 0.05), "z50": quantile(zs, 0.5), "z95": quantile(zs, 0.95),
            "dx": quantile(xs, 0.95) - quantile(xs, 0.05),
            "dy": quantile(ys, 0.95) - quantile(ys, 0.05),
            "dz": quantile(zs, 0.95) - quantile(zs, 0.05)}


def candidates(camera, prompt, base, color_name, limit=5):
    out = []
    probe = {"camera": camera, "prompt": prompt, "ok": False, "cands": []}
    try:
        found = segment_text(camera, prompt)
    except Exception as exc:
        probe["error"] = str(exc)
        report["probes"].append(probe)
        return out
    probe["ok"] = bool(found.ok)
    if found.ok:
        for cand in found.segmentations[:limit]:
            item = {"score": round(float(cand.score), 3)}
            try:
                item["box"] = [round(float(v), 1) for v in cand.box_xyxy]
            except Exception:
                item["box"] = None
            cloud = project(cand, camera, base)
            points = cloud_points(cloud) if cloud is not None else []
            item["stats"] = stats_of(points)
            if item["stats"] is not None:
                for key in ("x", "y", "z05", "z50", "z95", "dx", "dy", "dz"):
                    item["stats"][key] = round(item["stats"][key], 4)
            cols = cloud_colors(cloud) if cloud is not None else []
            item["n_colors"] = len(cols)
            if cols:
                item["mean_rgb"] = [round(sum(c[k] for c in cols) / len(cols), 3) for k in range(3)]
            item["color_frac"] = color_fraction(cols, color_name)
            probe["cands"].append(item)
            out.append({"cand": cand, "item": item, "points": points})
    if len(report["probes"]) < 80:
        report["probes"].append(probe)
    else:
        flags["probes_dropped"] = flags.get("probes_dropped", 0) + 1
    return out


def find_cube(camera, goal, base, near_xy=None, max_dist=None, strict=False):
    color = goal["color"]
    noun = goal["noun"]
    prompts = []
    for p in ((str(color) + " " + noun) if color else None, (str(color) + " cube") if color else None,
              (str(color) + " block") if color else None, noun, "cube"):
        if p is not None and p not in prompts:
            prompts.append(p)
    if strict:
        prompts = prompts[:1]
    fallback = None
    for prompt in prompts:
        best = None
        best_key = None
        for entry in candidates(camera, prompt, base, color):
            st = entry["item"]["stats"]
            if st is None or st["n"] < 8:
                continue
            if not (0.012 <= max(st["dx"], st["dy"]) <= 0.10 and st["dz"] <= 0.10):
                continue
            if near_xy is not None and max_dist is not None:
                if math.hypot(st["x"] - near_xy[0], st["y"] - near_xy[1]) > max_dist:
                    continue
            frac = entry["item"]["color_frac"]
            key = (frac if frac is not None else 0.0) * 2.0 + entry["item"]["score"]
            if frac is not None and frac < 0.3:
                if fallback is None or entry["item"]["score"] > fallback["item"]["score"]:
                    fallback = entry
                continue
            if best is None or key > best_key:
                best = entry
                best_key = key
        if best is not None:
            best["prompt"] = prompt
            return best
    if strict:
        return None
    if fallback is not None:
        fallback["prompt"] = "fallback"
        report["notes"].append("cube colour check failed; using best-scoring shape candidate")
    return fallback


def find_flat(camera, prompts, base, avoid, kind):
    for prompt in prompts:
        best = None
        for entry in candidates(camera, prompt, base, None):
            st = entry["item"]["stats"]
            if st is None or st["n"] < 8:
                continue
            span = max(st["dx"], st["dy"])
            if kind == "target":
                if not (0.03 <= span <= 0.30 and st["dz"] <= 0.04):
                    continue
            else:
                if not (0.015 <= span <= 0.20 and st["dz"] <= 0.12):
                    continue
            clash = False
            for xy in avoid:
                if math.hypot(st["x"] - xy[0], st["y"] - xy[1]) < 0.035:
                    clash = True
            if clash:
                continue
            if best is None or entry["item"]["score"] > best["item"]["score"]:
                best = entry
        if best is not None:
            best["prompt"] = prompt
            return best
    return None


def footprint_yaw(points, top_z):
    tops = [p for p in points if p[2] >= top_z - 0.008]
    use = tops if len(tops) >= 12 else points
    if len(use) < 8:
        return None
    best = None
    best_area = None
    for k in range(30):
        a = math.radians(3.0 * k)
        us = [p[0] * math.cos(a) + p[1] * math.sin(a) for p in use]
        vs = [-p[0] * math.sin(a) + p[1] * math.cos(a) for p in use]
        area = (quantile(us, 0.95) - quantile(us, 0.05)) * (quantile(vs, 0.95) - quantile(vs, 0.05))
        if best_area is None or area < best_area:
            best_area = area
            best = a
    return best


def quat_mul(a, b):
    return (a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])


def hand_axes(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    z_axis = (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
    y_axis = (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x))
    return z_axis, y_axis


def aligned_quat(q, cube_yaw):
    if cube_yaw is None:
        return q, 0.0
    z_axis, y_axis = hand_axes(q)
    close_yaw = math.atan2(y_axis[1], y_axis[0])
    delta = cube_yaw - close_yaw
    quarter = math.pi / 2.0
    while delta > quarter / 2.0:
        delta = delta - quarter
    while delta < -quarter / 2.0:
        delta = delta + quarter
    rot = (math.cos(delta / 2.0), 0.0, 0.0, math.sin(delta / 2.0))
    out = quat_mul(rot, q)
    norm = math.sqrt(sum(v * v for v in out))
    return tuple(v / norm for v in out), delta


def hold_ticks(count, gripper):
    state = get_robot_state()
    for tick in range(count):
        res = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=state.joint_positions["primary"],
            gripper_position=gripper, embodiment=state.embodiment)}))
        if ended(res):
            return res
    return None


def move(label, pos, quat, base, tol, steps):
    if flags["ended"]:
        return None
    res = move_to_pose(Pose(tuple(float(v) for v in pos), tuple(float(v) for v in quat), base),
                       tolerance=tol, max_steps=steps)
    record(label, res, {"target": [round(float(v), 4) for v in pos]})
    return res


def grip_now():
    try:
        return float(get_robot_state().gripper_positions["primary"])
    except Exception:
        return None




# ---------------------------------------------------------------- v01: moving-target tracking
DUR = {}
GEOM = {"target_rx": None}


def note_dur(kind, seconds):
    if seconds is None or seconds <= 0.0:
        return
    if kind not in DUR:
        DUR[kind] = []
    DUR[kind].append(seconds)


def timed_move(kind, label, pos, quat, base, tol, steps):
    t0 = now_s()
    res = move(label, pos, quat, base, tol, steps)
    t1 = now_s() if not flags["ended"] else None
    if t0 is not None and t1 is not None:
        note_dur(kind, t1 - t0)
    return res


def mid(values, lo, hi):
    return 0.5 * (quantile(values, lo) + quantile(values, hi))


def disc_centre(points):
    """Centre of a flat disc seen by the front camera. Anything held above the disc occludes its
    far (-x) side, never its camera-facing (+x) rim, so x = front rim - radius (radius from the
    first, unoccluded view); y = mid-range of the lateral rims."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cy = mid(ys, 0.03, 0.97)
    if GEOM["target_rx"] is None:
        return mid(xs, 0.03, 0.97), cy
    return quantile(xs, 0.97) - GEOM["target_rx"], cy


def observe_target(base, last_xy, radius):
    best = None
    for entry in candidates("front", "target", base, None, 3):
        st = entry["item"]["stats"]
        pts = entry["points"]
        if st is None or st["n"] < 8:
            continue
        if not (0.03 <= max(st["dx"], st["dy"]) <= 0.30 and st["dz"] <= 0.04):
            continue
        cx, cy = disc_centre(pts)
        if last_xy is not None and math.hypot(cx - last_xy[0], cy - last_xy[1]) > radius:
            continue
        if best is None or entry["item"]["score"] > best["score"]:
            best = {"x": cx, "y": cy, "top": st["z95"], "score": entry["item"]["score"]}
    if best is None:
        return None
    best["t"] = now_s()
    return best


def observe_cube(goal, base, near_xy, radius, strict=False):
    seen = find_cube("front", goal, base, near_xy, radius, strict)
    if seen is None:
        return None
    st = seen["item"]["stats"]
    box = seen["item"]["box"]
    box_h = None if box is None else box[3] - box[1]
    return {"t": now_s(), "x": st["x"], "y": st["y"], "top": st["z95"], "points": seen["points"],
            "box_h": box_h, "score": seen["item"]["score"]}


def velocity(hist):
    """Least-squares velocity over samples in the newest 0.6 s (>= 0.15 s span); None if unavailable."""
    if len(hist) < 2:
        return None
    t_last = hist[-1][0]
    window = [h for h in hist if t_last - h[0] <= 0.6]
    if len(window) < 2 or t_last - window[0][0] < 0.15:
        return None
    n = float(len(window))
    mt = sum(h[0] for h in window) / n
    mx = sum(h[1] for h in window) / n
    my = sum(h[2] for h in window) / n
    stt = sum((h[0] - mt) * (h[0] - mt) for h in window)
    if stt <= 1e-9:
        return None
    vx = sum((h[0] - mt) * (h[1] - mx) for h in window) / stt
    vy = sum((h[0] - mt) * (h[2] - my) for h in window) / stt
    speed = math.hypot(vx, vy)
    if speed > 0.30:
        return None
    if speed < 0.015:
        return (0.0, 0.0)
    return (vx, vy)


def predict(hist, vel, t_future):
    last = hist[-1]
    if vel is None:
        return (last[1], last[2])
    return (last[1] + vel[0] * (t_future - last[0]), last[2] + vel[1] * (t_future - last[0]))


def moving(vel):
    return vel is not None and (vel[0] != 0.0 or vel[1] != 0.0)


def ee_pos():
    p = get_robot_state().end_effector_poses["primary"].position
    return (float(p[0]), float(p[1]), float(p[2]))


def ik_joints(ik):
    try:
        if not ik.ok:
            return None
    except Exception:
        return None
    for attempt in range(3):
        try:
            if attempt == 0:
                vals = ik.joint_positions
            elif attempt == 1:
                vals = ik.solution
            else:
                vals = ik.joints
            out = [float(v) for v in vals]
            if len(out) == 7:
                return out
        except Exception:
            pass
    return None


def servo(kind, goal, base, hist, quat, z_to, carry_grip, finish_grip, finish_ticks, max_ticks, log):
    """Per-tick visual servo over a moving object ('target' disc or 'cube'): every tick observe it
    (perception does not advance the control clock), fit its velocity, command the IK joints for its
    position one tick ahead plus half the measured lag, descend <= 1.2 cm/tick to z_to, and once
    aligned (< 1 cm, or 8 ticks at height) keep tracking for finish_ticks while commanding finish_grip."""
    phase = "descend"
    at_height = 0
    fin = 0
    lag = (0.0, 0.0)
    ref_h = None
    log["servo_ticks"] = 0
    log["servo_err"] = []
    log["servo_rejects"] = 0
    for tick in range(max_ticks):
        if flags["ended"]:
            break
        guess = predict(hist, velocity(hist), now_s())
        if kind == "target":
            seen = observe_target(base, guess, 0.10)
        else:
            seen = observe_cube(goal, base, guess, 0.08 if ref_h is None else 0.025, True)
            if seen is not None and seen["box_h"] is not None:
                if ref_h is None or ref_h < seen["box_h"]:
                    ref_h = seen["box_h"]
                elif seen["box_h"] < 0.75 * ref_h:
                    seen = None
                    log["servo_rejects"] = log["servo_rejects"] + 1
        if seen is not None:
            hist.append((seen["t"], seen["x"], seen["y"]))
        vel = velocity(hist)
        t = now_s()
        obj = predict(hist, vel, t)
        ahead = predict(hist, vel, t + 0.05)
        state = get_robot_state()
        ee = state.end_effector_poses["primary"].position
        ex = float(ee[0])
        ey = float(ee[1])
        ez = float(ee[2])
        err = math.hypot(obj[0] - ex, obj[1] - ey)
        log["servo_err"].append(round(err, 4))
        lag = (obj[0] - ex, obj[1] - ey)
        cx = ahead[0] + 0.5 * lag[0]
        cy = ahead[1] + 0.5 * lag[1]
        dx = cx - ex
        dy = cy - ey
        d = math.hypot(dx, dy)
        if d > 0.04:
            cx = ex + dx * 0.04 / d
            cy = ey + dy * 0.04 / d
        cz = max(z_to, ez - 0.012)
        if phase == "descend" and ez <= z_to + 0.006:
            at_height = at_height + 1
            if err < 0.01 or (at_height >= 8 and err < 0.02):
                phase = "finish"
        grip = finish_grip if phase == "finish" else carry_grip
        current = [float(v) for v in state.joint_positions["primary"]]
        target = ik_joints(solve_ik(Pose((cx, cy, cz), quat, base), seed_joints=current))
        if target is None or max(abs(target[k] - current[k]) for k in range(7)) > 0.3:
            target = current
            log["ik_holds"] = log.get("ik_holds", 0) + 1
        res = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=target, gripper_position=grip,
            embodiment=state.embodiment)}))
        log["servo_ticks"] = log["servo_ticks"] + 1
        if ended(res):
            break
        if phase == "finish":
            fin = fin + 1
            if fin >= finish_ticks:
                break
    log["servo_phase"] = phase
    log["servo_fin"] = fin
    return fin >= finish_ticks


def target_with_velocity(base, hist, cycle):
    """Observe the target; if no usable velocity exists, hold 6 ticks (gripper closed) and re-observe."""
    for tries in range(3):
        if flags["ended"]:
            return None
        guess = predict(hist, velocity(hist), now_s()) if hist else None
        seen = observe_target(base, guess, 0.35)
        if seen is not None:
            hist.append((seen["t"], seen["x"], seen["y"]))
        vel = velocity(hist)
        if seen is not None and vel is not None:
            return vel
        hold_ticks(6, 0.0)
    cycle["notes"] = "target velocity unavailable"
    return velocity(hist)


def run():
    context = get_task_context()
    state = get_robot_state()
    base = state.base_frame
    goal_text = context.language
    report["goal"] = goal_text
    goal = parse_goal(goal_text)
    report["parsed"] = {k: goal[k] for k in ("color", "noun", "count", "button", "target", "target_word")}
    try:
        report["task_metadata"] = dict(robomme_plus.get_task_metadata())
    except Exception as exc:
        report["task_metadata_error"] = str(exc)
    obs = get_observation()
    report["cameras"] = [name for name in obs.cameras]
    start_pose = state.end_effector_poses["primary"]
    report["start_ee"] = [round(float(v), 4) for v in start_pose.position]
    quat = tuple(float(v) for v in start_pose.quaternion_wxyz)
    z_axis, y_axis = hand_axes(quat)
    if z_axis[2] > -0.9:
        quat = (0.0, 1.0, 0.0, 0.0)
        report["notes"].append("start orientation not top-down; using (0,1,0,0)")
    count = goal["count"]
    if count is None:
        count = 1
        report["notes"].append("no repetition count parsed; defaulting to 1")

    cube = find_cube("front", goal, base)
    if cube is None:
        cube = find_cube("wrist", goal, base)
    if cube is None:
        raise ValueError("target cube not found")
    cst = cube["item"]["stats"]
    report["cube0"] = {"prompt": cube["prompt"], "item": cube["item"]}
    cube_h = min(0.07, max(0.02, cst["z95"] - cst["z05"]))
    top_z = cst["z95"]
    support_z = top_z - cube_h
    report["cube_h_est"] = round(cube_h, 4)

    hist_t = []
    target0 = None
    if goal["target"]:
        tw = goal["target_word"]
        target = find_flat("front", (tw, "target", "round target", "red and white target", "disc"),
                           base, [(cst["x"], cst["y"])], "target")
        if target is not None:
            report["target"] = {"prompt": target["prompt"], "item": target["item"]}
            tp = target["points"]
            xs0 = [p[0] for p in tp]
            target0 = (mid(xs0, 0.03, 0.97), mid([p[1] for p in tp], 0.03, 0.97),
                       target["item"]["stats"]["z95"])
            GEOM["target_rx"] = 0.5 * (quantile(xs0, 0.97) - quantile(xs0, 0.03))
            report["target_rx"] = round(GEOM["target_rx"], 4)
            hist_t.append((now_s(), target0[0], target0[1]))
        else:
            report["notes"].append("target mentioned but not found; putting cube back in place")
    button = None
    if goal["button"]:
        avoid = [(cst["x"], cst["y"])]
        if target0 is not None:
            avoid.append((target0[0], target0[1]))
        button = find_flat("front", ("button", "red button", "round button", "push button"),
                           base, avoid, "button")
        if button is not None:
            report["button"] = {"prompt": button["prompt"], "item": button["item"]}
        else:
            report["notes"].append("button mentioned but not found")

    hist_c = [(now_s(), cst["x"], cst["y"])]
    cube_points = cube["points"]
    grasp_quat = quat
    target_moving = False
    on_target = False
    open_gripper()
    done = 0
    attempts = 0
    while done < count and attempts < count + 3 and not flags["ended"]:
        attempts = attempts + 1
        cycle = {"index": attempts, "t_start": now_s()}
        # ------------------------------------------------------------ pick
        yaw = footprint_yaw(cube_points, top_z) if cube_points else None
        grasp_quat, delta = aligned_quat(quat if attempts == 1 else grasp_quat, yaw)
        cycle["yaw_delta_deg"] = round(math.degrees(delta), 1)
        grasp_z = max(support_z + 0.012, top_z - 0.5 * cube_h)
        cycle["grasp_z"] = round(grasp_z, 4)
        cube_moving = target_moving and on_target
        cycle["cube_moving"] = cube_moving
        if cube_moving:
            vt = velocity(hist_t)
            aim = predict(hist_c, vt, now_s() + 0.5)
        else:
            aim = (hist_c[-1][1], hist_c[-1][2])
        cycle["cube_xy"] = [round(hist_c[-1][1], 4), round(hist_c[-1][2], 4)]
        timed_move("hover", "hover_" + str(attempts), (aim[0], aim[1], top_z + 0.10), grasp_quat, base, 0.015, 250)
        if flags["ended"]:
            break
        if grip_now() is not None and grip_now() < 0.9:
            record("reopen_" + str(attempts), open_gripper())
        if cube_moving:
            ok = servo("cube", goal, base, hist_c, grasp_quat, grasp_z, 1.0, 0.0, 7, 45, cycle)
            record("servo_grasp_" + str(attempts), None, {"done": ok})
            if not flags["ended"]:
                record("close_" + str(attempts), close_gripper())
        else:
            timed_move("descend", "descend_" + str(attempts), (aim[0], aim[1], grasp_z), grasp_quat, base, 0.01, 150)
            if flags["ended"]:
                break
            record("close_" + str(attempts), close_gripper())
        if flags["ended"]:
            report["cycles"].append(cycle)
            break
        cycle["grip_closed"] = grip_now()
        lift_z = grasp_z + 0.12
        here = ee_pos()
        timed_move("lift", "lift_" + str(attempts), (here[0], here[1], lift_z), grasp_quat, base, 0.02, 200)
        if flags["ended"]:
            report["cycles"].append(cycle)
            break
        g = grip_now()
        cycle["grip_lifted"] = g
        held = g is not None and g > 0.05
        cycle["held"] = held
        if not held:
            report["cycles"].append(cycle)
            record("open_after_miss_" + str(attempts), open_gripper())
            again = observe_cube(goal, base, (hist_c[-1][1], hist_c[-1][2]), 0.20)
            if again is not None:
                hist_c = [(again["t"], again["x"], again["y"])]
                top_z = again["top"]
                support_z = top_z - cube_h
                cube_points = again["points"]
            continue
        on_target = False
        # ------------------------------------------------------------ place
        if target0 is None:
            dest = (hist_c[-1][1], hist_c[-1][2])
            dest_support = support_z
            vt = None
        else:
            dest_support = target0[2]
            vt = target_with_velocity(base, hist_t, cycle)
            if moving(vt):
                target_moving = True
            dest = (target0[0], target0[1])
        cycle["target_vel"] = None if vt is None else [round(vt[0], 4), round(vt[1], 4)]
        release_z = dest_support + (grasp_z - support_z) + 0.006
        cycle["release_z"] = round(release_z, 4)
        if target_moving:
            aim = predict(hist_t, vt, now_s() + 0.8)
            timed_move("carry", "carry_" + str(attempts), (aim[0], aim[1], lift_z), grasp_quat, base, 0.02, 250)
            if flags["ended"]:
                report["cycles"].append(cycle)
                break
            ok = servo("target", goal, base, hist_t, grasp_quat, release_z, 0.0, 1.0, 6, 45, cycle)
            record("servo_place_" + str(attempts), None, {"done": ok})
            if flags["ended"]:
                report["cycles"].append(cycle)
                break
            if grip_now() is not None and grip_now() < 0.9:
                record("release_" + str(attempts), open_gripper())
        else:
            here = ee_pos()
            if math.hypot(dest[0] - here[0], dest[1] - here[1]) > 0.01:
                timed_move("carry", "carry_" + str(attempts), (dest[0], dest[1], lift_z), grasp_quat, base, 0.02, 250)
            timed_move("lower", "lower_" + str(attempts), (dest[0], dest[1], release_z), grasp_quat, base, 0.01, 200)
            if flags["ended"]:
                report["cycles"].append(cycle)
                break
            record("release_" + str(attempts), open_gripper())
            if not flags["ended"]:
                hold_ticks(5, 1.0)
        if flags["ended"]:
            report["cycles"].append(cycle)
            break
        on_target = target0 is not None
        done = done + 1
        cycle["done"] = done
        here = ee_pos()
        timed_move("retreat", "retreat_" + str(attempts), (here[0], here[1], release_z + 0.10), grasp_quat, base, 0.02, 200)
        cycle["t_end"] = now_s()
        report["cycles"].append(cycle)
        if flags["ended"]:
            break
        support_z = dest_support
        top_z = dest_support + cube_h
        cube_points = []
        hist_c = [(now_s(), here[0], here[1])]
        if done < count:
            guess = (here[0], here[1])
            if target_moving and len(hist_t) > 0:
                guess = predict(hist_t, velocity(hist_t), now_s())
            again = observe_cube(goal, base, guess, 0.10)
            if again is not None:
                hist_c = [(again["t"], again["x"], again["y"])]
                if abs(again["top"] - top_z) < 0.02:
                    top_z = again["top"]
                cube_points = again["points"]
                cycle["relocalized"] = [round(again["x"], 4), round(again["y"], 4), round(top_z, 4)]
    report["picks_done"] = done
    report["target_moving"] = target_moving
    report["durations"] = {k: [round(v, 3) for v in DUR[k]] for k in DUR}

    if button is not None and not flags["ended"]:
        bst = button["item"]["stats"]
        btop = bst["z95"]
        record("close_for_button", close_gripper())
        move("button_hover", (bst["x"], bst["y"], btop + 0.08), quat, base, 0.015, 250)
        move("button_press", (bst["x"], bst["y"], btop - 0.015), quat, base, 0.01, 60)
        if not flags["ended"]:
            hold_ticks(5, 0.0)
        move("button_release", (bst["x"], bst["y"], btop + 0.08), quat, base, 0.02, 150)
    if not flags["ended"]:
        hold_ticks(10, grip_now() if grip_now() is not None else 1.0)
    report["end_t"] = now_s() if not flags["ended"] else None


try:
    run()
except Exception as exc:
    report["error"] = str(exc)
report["ended"] = flags["ended"]
report["probes_dropped"] = flags.get("probes_dropped", 0)
result = report
