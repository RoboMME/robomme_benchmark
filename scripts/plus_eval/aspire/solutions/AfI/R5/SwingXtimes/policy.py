# SwingXtimes v03 -- v02 + closed-loop target tracking: before every descent the program
# hovers, holds a few ticks, re-localizes the target with the WRIST camera (twice: at hover and
# at an intermediate height) and descends on the fresh estimate. Motivation: ep1, a target slid
# ~5 cm when the cube arrived above it; v02 descended on the stale front-camera estimate, the
# visit never registered, and the episode failed on the next target.
# v02 was: v01 with ONE change: left/right named in the ROBOT frame
# (robot_base +y = robot's left; v01's viewer/front-image convention failed on ep0 with the
# button pressed), plus no API call after a terminal result (v01 crashed on a post-press read).
# v01 was: v00 + (a) button chosen by score among UNSATURATED, non-flat, non-cube
# candidates (v00 pressed the blue cube), (b) left/right named by the FRONT IMAGE column
# (viewer perspective; v00 used robot_base +y = left), (c) short bounded press.
# Records goal/metadata/scene diagnostics in `result` AND attempts the task:
#   parse goal (cube colour, repetition count, target order, put-down / button clauses)
#   -> localize cube + two flat targets in the front camera -> top-down grasp + lift
#   -> visit targets alternately (cube lowered onto each target), N repetitions
#   -> optional put-down on the cube's original spot -> optional button press.
# Everything task-dependent is derived at runtime from the goal text and live RGB-D.
import math

report = {"version": "v03", "events": [], "errors": []}
ended = {"flag": False}


def note(name, value):
    report[name] = value


def log(entry):
    report["events"].append(entry)


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def now_s():
    try:
        return float(get_robot_state().timestamp_s)
    except Exception:
        return None


def check_end(res, label):
    entry = {"call": label}
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
    term = False
    try:
        term = bool(res.terminated) or term
    except Exception:
        pass
    try:
        term = bool(res.truncated) or term
    except Exception:
        pass
    entry["terminal"] = term
    try:
        entry["error"] = str(res.error)[:200] if res.error is not None else None
    except Exception:
        pass
    log(entry)
    if term:
        ended["flag"] = True
    return entry


# ---------------------------------------------------------------- context
ctx = get_task_context()
goal = str(ctx.language)
note("goal", goal)
try:
    meta = robomme_plus.get_task_metadata()
    note("task_metadata", {k: str(meta[k]) for k in meta})
except Exception as exc:
    meta = {}
    report["errors"].append("task_metadata: " + str(exc))
try:
    cmeta = robomme_plus.get_controller_metadata()
    note("controller_metadata", {k: str(cmeta[k])[:300] for k in cmeta})
except Exception as exc:
    report["errors"].append("controller_metadata: " + str(exc))
max_steps = None
try:
    max_steps = int(meta["max_steps"])
except Exception:
    pass
note("max_steps", max_steps)
try:
    video = robomme_plus.get_conditioning_video()
    note("video_len", len(video))
    if len(video) > 0:
        note("video_t", [float(video[0].timestamp_s), float(video[len(video) - 1].timestamp_s)])
except Exception as exc:
    report["errors"].append("video: " + str(exc))

state0 = get_robot_state()
base = state0.base_frame
home_quat = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]
home_pos = [float(v) for v in state0.end_effector_poses["primary"].position]
note("home_ee", {"pos": home_pos, "quat": home_quat, "grip": float(state0.gripper_positions["primary"])})
obs0 = get_observation()
try:
    cam = obs0.cameras["front"]
    note("front_cam", {"pos": [float(v) for v in cam.camera_pose.position],
                       "quat": [float(v) for v in cam.camera_pose.quaternion_wxyz],
                       "frame": str(cam.camera_pose.frame)})
except Exception as exc:
    report["errors"].append("front_cam: " + str(exc))

# ---------------------------------------------------------------- goal parsing
text = goal.lower().replace(",", " ").replace(".", " ").replace(";", " ")
words = text.split()
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10, "once": 1, "twice": 2,
                "thrice": 3, "single": 1}
count = None
for i, w in enumerate(words):
    if w in ("times", "time") and i > 0:
        prev = words[i - 1]
        if prev.isdigit():
            count = int(prev)
        elif prev in NUMBER_WORDS:
            count = NUMBER_WORDS[prev]
    if count is None and w in ("once", "twice", "thrice"):
        count = NUMBER_WORDS[w]
if count is None:
    for w in words:
        if w.isdigit():
            count = int(w)
            break
note("parsed_count", count)
COLORS = ["red", "green", "blue", "yellow", "purple", "orange", "white", "black",
          "pink", "cyan", "brown", "gray", "grey", "magenta"]
cube_color = None
for i, w in enumerate(words):
    if w in ("cube", "block") and i > 0 and words[i - 1] in COLORS:
        cube_color = words[i - 1]
        break
if cube_color is None:
    for w in words:
        if w in COLORS:
            cube_color = w
            break
note("cube_color", cube_color)
pos_left = text.find("left")
pos_right = text.find("right")
order = None
if pos_left >= 0 and pos_right >= 0:
    order = ["left", "right"] if pos_left < pos_right else ["right", "left"]
elif pos_left >= 0:
    order = ["left", "right"]
elif pos_right >= 0:
    order = ["right", "left"]
note("target_order_words", order)
want_putdown = ("put" in words and "down" in words) or ("place" in text and "table" in text) \
    or ("release" in words) or ("drop" in words)
want_button = "button" in text
note("want_putdown", want_putdown)
note("want_button", want_button)

# ---------------------------------------------------------------- perception


def cloud_stats(cand, cam="front"):
    pc = mask_to_point_cloud(cand, cam, target_frame=base)
    pts = pc.points
    n = len(pts)
    if n < 8:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    zs = [float(p[2]) for p in pts]
    return {
        "n": n,
        "x": quantile(xs, 0.5), "y": quantile(ys, 0.5),
        "z10": quantile(zs, 0.1), "z50": quantile(zs, 0.5), "z90": quantile(zs, 0.9),
        "zmax": quantile(zs, 0.98),
        "sx": quantile(xs, 0.95) - quantile(xs, 0.05),
        "sy": quantile(ys, 0.95) - quantile(ys, 0.05),
    }


def candidates(prompt, limit=6, cam="front"):
    out = []
    seg = segment_text(cam, prompt)
    if not seg.ok:
        return out
    for cand in seg.segmentations[:limit]:
        try:
            st = cloud_stats(cand, cam)
        except Exception as exc:
            report["errors"].append("cloud " + prompt + ": " + str(exc)[:120])
            st = None
        if st is None:
            continue
        st["score"] = float(cand.score)
        st["prompt"] = prompt
        try:
            st["box"] = [round(float(v), 1) for v in cand.box_xyxy]
        except Exception:
            st["box"] = None
        out.append(st)
    return out


def neg_score(c):
    return -c["score"]


def neg_height(c):
    return -(c["zmax"] - c["z10"])


def rounded(st):
    out = {}
    for k in st:
        v = st[k]
        if k in ("prompt", "box", "n", "reject") or v is None:
            out[k] = v
        else:
            out[k] = round(v, 4)
    return out


cube_prompts = []
if cube_color is not None:
    cube_prompts.append(cube_color + " cube")
cube_prompts += ["cube", "block"]
target_prompts = ["target", "round target", "bullseye target", "flat disc on the table"]
probe = {}
cube = None
for prompt in cube_prompts:
    cs = candidates(prompt)
    probe[prompt] = [rounded(c) for c in cs]
    plausible = [c for c in cs if 0.015 <= max(c["sx"], c["sy"]) <= 0.09 and (c["zmax"] - c["z10"]) < 0.09]
    plausible = sorted(plausible, key=neg_score)
    if plausible:
        cube = plausible[0]
        break
note("cube", rounded(cube) if cube is not None else None)

target_pool = []
for prompt in target_prompts:
    cs = candidates(prompt)
    probe[prompt] = [rounded(c) for c in cs]
    for c in cs:
        flat = (c["z90"] - c["z10"]) < 0.035
        sized = 0.03 <= max(c["sx"], c["sy"]) <= 0.30
        not_cube = True
        if cube is not None:
            not_cube = math.hypot(c["x"] - cube["x"], c["y"] - cube["y"]) > 0.035
        if flat and sized and not_cube:
            target_pool.append(c)
targets = []
for c in sorted(target_pool, key=neg_score):
    dup = False
    for t in targets:
        if math.hypot(c["x"] - t["x"], c["y"] - t["y"]) < 0.05:
            dup = True
    if not dup:
        targets.append(c)
note("targets_all", [rounded(t) for t in targets])

# table height: flat targets lie on it; else cube bottom
table_z = None
if targets:
    table_z = min(t["z10"] for t in targets[:2])
elif cube is not None:
    table_z = cube["z10"]
note("table_z", table_z)

front_rgb = obs0.cameras["front"].rgb


def box_saturation(box, rgb=None):
    """Mean HSV-style saturation (max-min)/max over a 3x3 window at the box centre."""
    if box is None:
        return None
    if rgb is None:
        rgb = front_rgb
    cx = int(0.5 * (box[0] + box[2]))
    cy = int(0.5 * (box[1] + box[3]))
    total = 0.0
    k = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            px = rgb[min(len(rgb) - 1, max(0, cy + dy))][min(len(rgb[0]) - 1, max(0, cx + dx))]
            r = float(px[0])
            g = float(px[1])
            b = float(px[2])
            hi = max(r, g, b)
            lo = min(r, g, b)
            total += (hi - lo) / hi if hi > 1.0 else 0.0
            k += 1
    return total / k


button = None
button_pool = []
if want_button:
    for prompt in ["button", "push button", "grey button on a white base"]:
        cs = candidates(prompt)
        probe[prompt] = [rounded(c) for c in cs]
        for c in cs:
            c["sat"] = box_saturation(c["box"])
            c["rise"] = c["zmax"] - (table_z if table_z is not None else c["z10"])
            reason = None
            if max(c["sx"], c["sy"]) > 0.12:
                reason = "too_big"
            elif c["rise"] < 0.012:
                reason = "flat"          # targets / table patches
            elif c["sat"] is not None and c["sat"] > 0.35:
                reason = "saturated"     # coloured cubes
            elif cube is not None and math.hypot(c["x"] - cube["x"], c["y"] - cube["y"]) < 0.035:
                reason = "is_cube"
            c["reject"] = reason
            button_pool.append(rounded(c))
            if reason is None and (button is None or c["score"] > button["score"]):
                button = c
        if button is not None:
            break
    note("button_pool", button_pool)
    note("button", rounded(button) if button is not None else None)
note("probe", probe)

result = report
if cube is None or len(targets) < 2 or count is None:
    report["errors"].append("insufficient perception/parse: cube=%s targets=%d count=%s" % (cube is not None, len(targets), count))
    open_gripper()
else:
    # choose the two best targets and name them left/right in robot_base (+y = robot's left)
    pair = targets[:2]
    # robot perspective: robot_base +y is the robot's left, so the right-side target has smaller y
    right_t = pair[0] if pair[0]["y"] < pair[1]["y"] else pair[1]
    left_t = pair[1] if right_t is pair[0] else pair[0]
    note("side_assignment", {"right_xy": [round(right_t["x"], 3), round(right_t["y"], 3)],
                             "left_xy": [round(left_t["x"], 3), round(left_t["y"], 3)],
                             "convention": "robot_base y (robot perspective)"})
    named = {"left": left_t, "right": right_t}
    seq_names = order if order is not None else ["right", "left"]
    visits = []
    for rep in range(count):
        visits.append(seq_names[0])
        visits.append(seq_names[1])
    note("visit_plan", visits)

    cube_top = cube["zmax"]
    cube_h = max(0.02, min(0.06, cube_top - table_z))
    note("cube_height_est", round(cube_h, 4))
    grasp_z = cube_top - 0.5 * cube_h
    quat = home_quat
    strat = MotionStrategy()

    def go(x, y, z, label, tol=0.015, steps=150):
        if ended["flag"]:
            return None
        res = move_to_pose(Pose([x, y, z], quat, base), tolerance=tol, max_steps=steps)
        e = check_end(res, label)
        e["target"] = [round(x, 3), round(y, 3), round(z, 3)]
        if not ended["flag"]:
            e["t"] = now_s()
        return res

    def hold_ticks(n, grip):
        if ended["flag"]:
            return
        st = get_robot_state()
        for k in range(n):
            if ended["flag"]:
                return
            r = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=st.joint_positions["primary"],
                gripper_position=grip, embodiment=st.embodiment)}))
            try:
                if bool(r.terminated) or bool(r.truncated):
                    ended["flag"] = True
            except Exception:
                pass

    def relocate_target(est, label):
        """Wrist-camera re-localization of a flat target near `est` (dict with x, y).
        Accepts flat candidates lying on the table, 2-12 cm across, within 12 cm of `est`;
        returns the highest-scoring one, or None."""
        if ended["flag"]:
            return None
        best = None
        best_d = None
        seen = []
        for prompt in ("target", "round target"):
            for c in candidates(prompt, 6, "wrist"):
                d = math.hypot(c["x"] - est["x"], c["y"] - est["y"])
                ok = ((c["z90"] - c["z10"]) < 0.03 and abs(c["z50"] - table_z) < 0.02
                      and 0.02 <= max(c["sx"], c["sy"]) <= 0.12 and d < 0.12)
                seen.append([prompt, round(c["x"], 3), round(c["y"], 3), round(c["z50"], 3),
                             round(max(c["sx"], c["sy"]), 3), round(c["score"], 3), ok])
                if ok and (best is None or c["score"] > best["score"]):
                    best = c
                    best_d = d
            if best is not None:
                break
        entry = {"relocate": label, "est": [round(est["x"], 3), round(est["y"], 3)], "seen": seen[:6]}
        if best is not None:
            entry["new"] = [round(best["x"], 3), round(best["y"], 3)]
            entry["shift"] = round(best_d, 4)
        log(entry)
        return best

    held = False
    for attempt in range(2):
        if ended["flag"]:
            break
        check_end(open_gripper(), "open")
        go(cube["x"], cube["y"], grasp_z + 0.12, "pregrasp")
        go(cube["x"], cube["y"], grasp_z, "grasp", tol=0.01)
        check_end(close_gripper(), "close")
        go(cube["x"], cube["y"], grasp_z + 0.20, "lift")
        g = float(get_robot_state().gripper_positions["primary"]) if not ended["flag"] else -1.0
        log({"grip_after_lift": g})
        if g > 0.05:
            held = True
            break
        # re-localize before retry
        for prompt in cube_prompts:
            cs = [c for c in candidates(prompt, 6, "front") if 0.015 <= max(c["sx"], c["sy"]) <= 0.09]
            if cs:
                cube = sorted(cs, key=neg_score)[0]
                cube_top = cube["zmax"]
                grasp_z = cube_top - 0.5 * cube_h
                break
    note("held", held)

    hover = table_z + cube_h + 0.10   # transit height for cube bottom ~ +0.10 above table
    place_center_z = table_z + 0.5 * cube_h + 0.006  # cube resting on a thin target
    # intermediate stop: cube bottom ~6 cm above the table, still above the height at which
    # ep0 visits registered (TCP ~0.09-0.11), and low enough for a close wrist view
    mid_center_z = table_z + cube_h + 0.06 + 0.5 * cube_h
    # visit stop: cube bottom 3 cm above the target. ep0 visits registered (target turned red)
    # when the TCP passed 0.08-0.09 on the way down, so this is well inside the registering
    # zone, while a visit that misses its target no longer sets the cube down on bare table
    # (ep1 v02 lowered the cube onto bare table 5.5 cm from a moved target, then failed).
    visit_center_z = table_z + 0.03 + 0.5 * cube_h
    visits_done = 0
    if held:
        for name in visits:
            if ended["flag"]:
                break
            t = named[name]
            go(t["x"], t["y"], hover + 0.5 * cube_h, "over_" + name, tol=0.02)
            hold_ticks(6, 0.0)            # let a target that reacts to the approach finish moving
            fresh = relocate_target(t, "hover_" + name)
            if fresh is not None:
                t = {"x": fresh["x"], "y": fresh["y"]}
            go(t["x"], t["y"], mid_center_z, "mid_" + name, tol=0.015)
            fresh = relocate_target(t, "mid_" + name)
            if fresh is not None:
                t = {"x": fresh["x"], "y": fresh["y"]}
            go(t["x"], t["y"], visit_center_z, "onto_" + name, tol=0.01)
            named[name] = {"x": t["x"], "y": t["y"]}
            g = float(get_robot_state().gripper_positions["primary"]) if not ended["flag"] else -1.0
            log({"visit": name, "grip": g})
            visits_done += 1
        note("visits_done", visits_done)
        if not ended["flag"]:
            go(named[visits[-1]]["x"], named[visits[-1]]["y"], hover + 0.5 * cube_h, "rise_final", tol=0.02)
        if want_putdown and not ended["flag"]:
            go(cube["x"], cube["y"], hover + 0.5 * cube_h, "over_home_spot", tol=0.02)
            go(cube["x"], cube["y"], place_center_z, "putdown", tol=0.01)
            check_end(open_gripper(), "release")
            hold_ticks(6, 1.0)
            go(cube["x"], cube["y"], hover + 0.10, "retreat", tol=0.02)
        if want_button and button is not None and not ended["flag"]:
            check_end(close_gripper(), "close_for_press")
            btop = button["zmax"]
            go(button["x"], button["y"], btop + 0.10, "over_button", tol=0.02)
            # wrist re-check of the button (it may also have been displaced)
            if not ended["flag"]:
                wrist_rgb = get_observation(["wrist"]).cameras["wrist"].rgb
                bbest = None
                bseen = []
                for c in candidates("button", 6, "wrist"):
                    c["sat"] = box_saturation(c["box"], wrist_rgb)
                    d = math.hypot(c["x"] - button["x"], c["y"] - button["y"])
                    ok = (d < 0.08 and (c["zmax"] - table_z) > 0.012 and max(c["sx"], c["sy"]) < 0.12
                          and (c["sat"] is None or c["sat"] <= 0.35))
                    bseen.append([round(c["x"], 3), round(c["y"], 3), round(c["zmax"], 3), round(c["score"], 3), ok])
                    if ok and (bbest is None or c["score"] > bbest["score"]):
                        bbest = c
                log({"relocate": "button", "seen": bseen,
                     "new": [round(bbest["x"], 3), round(bbest["y"], 3)] if bbest is not None else None})
                if bbest is not None and math.hypot(bbest["x"] - button["x"], bbest["y"] - button["y"]) > 0.01:
                    button = bbest
                    go(button["x"], button["y"], btop + 0.10, "over_button_fixed", tol=0.02)
            go(button["x"], button["y"], btop - 0.015, "press", tol=0.008, steps=40)
            hold_ticks(5, 0.0)
            go(button["x"], button["y"], btop + 0.10, "unpress", tol=0.02)
    note("ended", ended["flag"])
    if not ended["flag"]:
        note("t_final", now_s())
result = report
