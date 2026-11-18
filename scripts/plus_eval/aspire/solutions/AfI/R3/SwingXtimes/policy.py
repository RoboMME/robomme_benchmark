# v00 -- SwingXtimes combined scene probe + first attempt (start 1 of 3).
# Records task language, metadata, cameras, conditioning-video summary, prompt probes and
# a per-motion timeline into `result`, AND attempts the task as parsed from the language:
#   pick the goal-coloured cube -> visit the two targets alternately N times (touch-hover,
#   then lift over each target) -> put the cube down where it was -> press the button.
# Every task-dependent position/size is derived at runtime from front-camera RGB-D.
import math

report = {"version": "v00", "phase": "start", "moves": [], "events": [], "errors": []}
result = report
flags = {"ended": False}

HZ = 20.0
WORD_NUM = {"one": 1, "once": 1, "two": 2, "twice": 2, "three": 3, "thrice": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
COLOUR_HUE = {"red": 0.0, "orange": 30.0, "yellow": 58.0, "green": 120.0, "cyan": 185.0,
              "blue": 225.0, "purple": 280.0, "violet": 280.0, "magenta": 305.0, "pink": 330.0}
COLOUR_NEUTRAL = ("white", "black", "gray", "grey", "silver")


def quant(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def now_tick():
    return int(round(float(get_robot_state().timestamp_s) * HZ))


def note(text):
    report["events"].append(str(now_tick()) + ": " + str(text))


def terminal_of(res):
    flag = False
    try:
        flag = bool(res.terminated)
    except Exception:
        flag = False
    try:
        flag = flag or bool(res.truncated)
    except Exception:
        pass
    return flag


def cloud_stats(pc):
    pts = pc.points
    n = len(pts)
    if n == 0:
        return None
    stride = max(1, n // 1500)
    xs = []
    ys = []
    zs = []
    for i in range(0, n, stride):
        p = pts[i]
        xs.append(float(p[0]))
        ys.append(float(p[1]))
        zs.append(float(p[2]))
    return {"n": n, "cx": quant(xs, 0.5), "cy": quant(ys, 0.5),
            "z03": quant(zs, 0.03), "z10": quant(zs, 0.10), "z50": quant(zs, 0.5),
            "z90": quant(zs, 0.90), "z97": quant(zs, 0.97),
            "xspan": quant(xs, 0.95) - quant(xs, 0.05), "yspan": quant(ys, 0.95) - quant(ys, 0.05)}


def median_rgb(rgb, cand, box):
    # Median colour over mask pixels inside the box; fallback: inner 40% of the box.
    rs = []
    gs = []
    bs = []
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(len(rgb[0]) - 1, int(box[2]))
    y2 = min(len(rgb) - 1, int(box[3]))
    used = "mask"
    try:
        mask = cand.mask
        stepy = max(1, (y2 - y1) // 24)
        stepx = max(1, (x2 - x1) // 24)
        for y in range(y1, y2 + 1, stepy):
            row = mask[y]
            for x in range(x1, x2 + 1, stepx):
                if row[x]:
                    px = rgb[y][x]
                    rs.append(int(px[0]))
                    gs.append(int(px[1]))
                    bs.append(int(px[2]))
    except Exception:
        rs = []
    if len(rs) < 4:
        used = "inner_box"
        rs = []
        gs = []
        bs = []
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        hw = max(1.0, (x2 - x1) * 0.2)
        hh = max(1.0, (y2 - y1) * 0.2)
        for y in range(int(cy - hh), int(cy + hh) + 1):
            for x in range(int(cx - hw), int(cx + hw) + 1):
                if 0 <= y < len(rgb) and 0 <= x < len(rgb[0]):
                    px = rgb[y][x]
                    rs.append(int(px[0]))
                    gs.append(int(px[1]))
                    bs.append(int(px[2]))
    if len(rs) == 0:
        return None
    return [quant(rs, 0.5), quant(gs, 0.5), quant(bs, 0.5), used]


def hsv(rgbm):
    r = rgbm[0] / 255.0
    g = rgbm[1] / 255.0
    b = rgbm[2] / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn
    s = 0.0 if mx <= 0 else d / mx
    if d <= 1e-6:
        h = 0.0
    elif mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * (((b - r) / d) + 2.0)
    else:
        h = 60.0 * (((r - g) / d) + 4.0)
    return h, s, mx


def colour_cost(word, rgbm):
    # Lower is better. None word -> 0 (no colour constraint).
    if word is None or rgbm is None:
        return 0.0
    h, s, v = hsv(rgbm)
    if word in COLOUR_HUE:
        dh = abs(h - COLOUR_HUE[word]) % 360.0
        dh = min(dh, 360.0 - dh)
        return dh / 180.0 + (1.0 if s < 0.25 else 0.0)
    if word == "white":
        return s + (1.0 - v)
    if word == "black":
        return v
    return s + abs(v - 0.5)


def collect(prompt, rgb, base_frame):
    out = []
    seg = segment_text("front", prompt)
    if not seg.ok:
        return out
    for idx, cand in enumerate(seg.segmentations):
        try:
            box = [float(value) for value in cand.box_xyxy]
        except Exception:
            box = None
        stats = None
        try:
            pc = mask_to_point_cloud(cand, "front", target_frame=base_frame)
            stats = cloud_stats(pc)
        except Exception as exc:
            report["errors"].append("cloud " + prompt + ": " + str(exc)[:160])
        if stats is None or stats["n"] < 15:
            continue
        rgbm = None
        if box is not None:
            rgbm = median_rgb(rgb, cand, box)
        out.append({"prompt": prompt, "idx": idx, "score": round(float(cand.score), 4),
                    "box": box, "stats": stats, "rgb": rgbm})
    return out


def brief(c):
    s = c["stats"]
    return {"prompt": c["prompt"], "score": c["score"],
            "box": None if c["box"] is None else [round(v, 1) for v in c["box"]],
            "xy": [round(s["cx"], 4), round(s["cy"], 4)],
            "z": [round(s["z03"], 4), round(s["z50"], 4), round(s["z97"], 4)],
            "span": [round(s["xspan"], 4), round(s["yspan"], 4)], "n": s["n"],
            "rgb": c["rgb"]}


def dist_xy(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


try:
    context = get_task_context()
    state0 = get_robot_state()
    base_frame = state0.base_frame
    embodiment = state0.embodiment
    language = str(context.language)
    report["task_language"] = language
    report["task_name"] = str(context.task_name)
    try:
        meta = robomme_plus.get_task_metadata()
        report["task_metadata"] = {str(k): str(meta[k])[:300] for k in meta}
    except Exception as exc:
        meta = {}
        report["errors"].append("task_metadata: " + str(exc)[:200])
    try:
        cmeta = robomme_plus.get_controller_metadata()
        report["controller_metadata"] = {str(k): str(cmeta[k])[:300] for k in cmeta}
    except Exception as exc:
        report["errors"].append("controller_metadata: " + str(exc)[:200])
    max_steps = 0
    try:
        max_steps = int(meta["max_steps"])
    except Exception:
        max_steps = 0
    report["max_steps"] = max_steps
    try:
        video = robomme_plus.get_conditioning_video()
        report["video_len"] = len(video)
        if len(video) > 0:
            report["video_t"] = [float(video[0].timestamp_s), float(video[len(video) - 1].timestamp_s)]
    except Exception as exc:
        report["errors"].append("video: " + str(exc)[:200])

    ee0 = state0.end_effector_poses["primary"]
    q0 = [float(v) for v in ee0.quaternion_wxyz]
    report["home_ee"] = {"pos": [round(float(v), 4) for v in ee0.position], "quat": q0}
    report["home_joints"] = [round(float(v), 4) for v in state0.joint_positions["primary"]]
    report["gripper0"] = float(state0.gripper_positions["primary"])
    zaxis_z = 1.0 - 2.0 * (q0[1] * q0[1] + q0[2] * q0[2])
    report["home_tool_z_dot_down"] = round(-zaxis_z, 4)
    down_q = q0 if zaxis_z < -0.9 else [0.0, 1.0, 0.0, 0.0]

    # ---- parse the goal ----
    low = language.lower()
    clean = ""
    for ch in low:
        clean = clean + (ch if (ch.isalnum() or ch == "-") else " ")
    tokens = clean.replace("-", " ").split()
    colour_word = None
    for tok in tokens:
        if tok in COLOUR_HUE or tok in COLOUR_NEUTRAL:
            colour_word = tok
            break
    count = None
    for i, tok in enumerate(tokens):
        if tok in ("once", "twice", "thrice") and count is None:
            count = WORD_NUM[tok]
        if tok == "times" and i > 0 and count is None:
            prev = tokens[i - 1]
            if prev.isdigit():
                count = int(prev)
            elif prev in WORD_NUM:
                count = WORD_NUM[prev]
    if count is None:
        for tok in tokens:
            if tok.isdigit():
                count = int(tok)
                break
    right_i = low.find("right")
    left_i = low.find("left")
    first_side = "right"
    if left_i >= 0 and (right_i < 0 or left_i < right_i):
        first_side = "left"
    report["parse"] = {"colour": colour_word, "count": count, "first_side": first_side,
                       "mentions_button": "button" in low, "tokens": tokens[:60]}
    n_rounds = count if count is not None else 1

    # ---- perceive ----
    obs = get_observation()
    report["cameras"] = [str(name) for name in obs.cameras]
    front = obs.cameras["front"]
    rgb = front.rgb
    try:
        cp = front.camera_pose
        report["front_camera_pose"] = {"pos": [round(float(v), 4) for v in cp.position],
                                       "quat": [round(float(v), 4) for v in cp.quaternion_wxyz]}
    except Exception as exc:
        report["errors"].append("camera_pose: " + str(exc)[:120])

    cube_prompts = ["cube"]
    if colour_word is not None:
        cube_prompts.append(colour_word + " cube")
    target_prompts = ["target", "disc"]
    button_prompts = ["round button", "button"]
    pool = []
    probes = {}
    for prompt in cube_prompts + target_prompts + button_prompts:
        got = collect(prompt, rgb, base_frame)
        probes[prompt] = [brief(c) for c in got[:6]]
        pool = pool + got
    report["probes"] = probes

    bottoms = sorted(c["stats"]["z10"] for c in pool)
    if len(bottoms) == 0:
        raise ValueError("nothing segmented")
    table_z = bottoms[0]
    # robust: lowest bottom that at least two candidates share within 1.5 cm
    for b in bottoms:
        near = 0
        for b2 in bottoms:
            if abs(b2 - b) <= 0.015:
                near += 1
        if near >= 2:
            table_z = b
            break
    report["table_z"] = round(table_z, 4)

    def height(c):
        return c["stats"]["z97"] - table_z

    # cube: tall-ish compact, resting on the table, colour closest to the goal word
    best_cube = None
    best_cost = None
    for c in pool:
        if c["prompt"] not in cube_prompts:
            continue
        s = c["stats"]
        h = height(c)
        if h < 0.015 or h > 0.09 or s["xspan"] > 0.10 or s["yspan"] > 0.10:
            continue
        if s["z03"] - table_z > 0.02:
            continue
        cost = colour_cost(colour_word, c["rgb"]) - 0.2 * c["score"]
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_cube = c
    if best_cube is None:
        raise ValueError("no cube candidate passed geometry bounds")
    report["cube"] = brief(best_cube)
    report["cube_colour_cost"] = round(best_cost, 4)
    cube_xy = [best_cube["stats"]["cx"], best_cube["stats"]["cy"]]

    # button: from button prompts, not the cube, raised above the table
    best_button = None
    for c in pool:
        if c["prompt"] not in button_prompts:
            continue
        s = c["stats"]
        h = height(c)
        if h < 0.006 or h > 0.10 or s["xspan"] > 0.15 or s["yspan"] > 0.15:
            continue
        if dist_xy([s["cx"], s["cy"]], cube_xy) < 0.035:
            continue
        if best_button is None or c["score"] > best_button["score"]:
            best_button = c
    report["button"] = None if best_button is None else brief(best_button)

    # targets: flat discs from target prompts, not cube/button, deduplicated
    flats = []
    for c in pool:
        if c["prompt"] not in target_prompts:
            continue
        s = c["stats"]
        if height(c) > 0.03 or s["xspan"] < 0.025 or s["yspan"] < 0.025:
            continue
        if s["xspan"] > 0.25 or s["yspan"] > 0.25:
            continue
        cxy = [s["cx"], s["cy"]]
        if dist_xy(cxy, cube_xy) < 0.04:
            continue
        if best_button is not None and dist_xy(cxy, [best_button["stats"]["cx"], best_button["stats"]["cy"]]) < 0.03:
            continue
        dup = False
        for f in flats:
            if dist_xy(cxy, [f["stats"]["cx"], f["stats"]["cy"]]) < 0.03:
                dup = True
                if c["score"] > f["score"]:
                    f["score"] = c["score"]
        if not dup:
            flats.append(c)
    ranked = sorted([(-f["score"], i) for i, f in enumerate(flats)])
    report["flat_candidates"] = [brief(flats[i]) for (negs, i) in ranked]
    if len(ranked) < 2:
        raise ValueError("fewer than two targets found")
    ta = flats[ranked[0][1]]
    tb = flats[ranked[1][1]]
    targets = {}
    if ta["stats"]["cy"] <= tb["stats"]["cy"]:
        targets["right"] = ta
        targets["left"] = tb
    else:
        targets["right"] = tb
        targets["left"] = ta
    track = {}
    for side in ("right", "left"):
        s = targets[side]["stats"]
        track[side] = {"xy": [s["cx"], s["cy"]], "top": s["z97"], "box": targets[side]["box"]}
    report["targets"] = {side: {"xy": [round(v, 4) for v in track[side]["xy"]],
                                "top": round(track[side]["top"], 4),
                                "box": track[side]["box"]} for side in track}

    # ---- geometry of the grasp ----
    cs = best_cube["stats"]
    cube_bottom = min(cs["z03"], table_z + 0.01)
    cube_top = cs["z97"]
    cube_h = cube_top - cube_bottom
    grasp_z = cube_bottom + 0.5 * cube_h
    if grasp_z < cube_bottom + 0.012:
        grasp_z = cube_bottom + 0.012
    ee_above_bottom = grasp_z - cube_bottom
    report["grasp_geom"] = {"bottom": round(cube_bottom, 4), "top": round(cube_top, 4),
                            "h": round(cube_h, 4), "grasp_z": round(grasp_z, 4)}
    obstacle_top = max(track["right"]["top"], track["left"]["top"])
    if best_button is not None:
        obstacle_top = max(obstacle_top, best_button["stats"]["z97"])
    transit_ee_z = obstacle_top + 0.06 + ee_above_bottom
    report["transit_ee_z"] = round(transit_ee_z, 4)

    state = {"first_str": None}

    def move(xyz, label, tol=0.01, ms=200):
        r = move_to_pose(Pose([float(xyz[0]), float(xyz[1]), float(xyz[2])], down_q, base_frame),
                         tolerance=tol, max_steps=ms)
        if state["first_str"] is None:
            state["first_str"] = str(r)[:700]
            report["first_motion_repr"] = state["first_str"]
        st = get_robot_state()
        p = st.end_effector_poses["primary"].position
        rec = {"label": label, "ok": bool(r.ok), "status": str(r.status)[:60],
               "steps": int(r.steps_executed), "tick": int(round(float(st.timestamp_s) * HZ)),
               "goal": [round(float(v), 4) for v in xyz],
               "ee": [round(float(v), 4) for v in p],
               "grip": round(float(st.gripper_positions["primary"]), 3)}
        report["moves"].append(rec)
        if terminal_of(r):
            flags["ended"] = True
            rec["terminal"] = True
            raise ValueError("EPISODE_ENDED after " + label)
        return r

    def hold(ticks, grip):
        st = get_robot_state()
        last = None
        for k in range(ticks):
            last = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=[float(v) for v in st.joint_positions["primary"]],
                gripper_position=grip, embodiment=st.embodiment)}))
            if terminal_of(last):
                flags["ended"] = True
                raise ValueError("EPISODE_ENDED during hold")
        return last

    def grip_width():
        return float(get_robot_state().gripper_positions["primary"])

    # ---- static check of the targets (8 ticks) ----
    report["phase"] = "static_check"
    hold(8, 1.0)
    obs2 = get_observation()
    again = collect("target", obs2.cameras["front"].rgb, base_frame)
    shifts = {}
    for side in ("right", "left"):
        best = None
        for c in again:
            d = dist_xy([c["stats"]["cx"], c["stats"]["cy"]], track[side]["xy"])
            if c["stats"]["z97"] - table_z <= 0.03 and (best is None or d < best):
                best = d
        shifts[side] = None if best is None else round(best, 4)
    report["target_shift_8ticks"] = shifts

    def relocate(side, max_jump):
        # Re-detect the target nearest to its last known xy; keep the old one if none close.
        o = get_observation()
        got = collect("target", o.cameras["front"].rgb, base_frame)
        best = None
        bestd = None
        other = "left" if side == "right" else "right"
        for c in got:
            s = c["stats"]
            if s["z97"] - table_z > 0.03 or s["xspan"] < 0.02 or s["yspan"] < 0.02:
                continue
            cxy = [s["cx"], s["cy"]]
            d = dist_xy(cxy, track[side]["xy"])
            if dist_xy(cxy, track[other]["xy"]) < d:
                continue
            if d <= max_jump and (bestd is None or d < bestd):
                bestd = d
                best = c
        if best is not None:
            track[side]["xy"] = [best["stats"]["cx"], best["stats"]["cy"]]
        return bestd

    # ---- pick ----
    report["phase"] = "pick"
    open_gripper()
    move([cube_xy[0], cube_xy[1], grasp_z + 0.10], "pregrasp")
    move([cube_xy[0], cube_xy[1], grasp_z], "grasp", tol=0.008, ms=150)
    close_gripper()
    hold(3, 0.0)
    report["grip_after_close"] = round(grip_width(), 3)
    move([cube_xy[0], cube_xy[1], transit_ee_z], "lift")
    report["grip_after_lift"] = round(grip_width(), 3)
    held = grip_width() > 0.12
    report["held_after_lift"] = held
    if not held:
        raise ValueError("cube not held after lift")

    # ---- swings ----
    report["phase"] = "swing"
    order = [first_side, "left" if first_side == "right" else "right"]
    visits = []
    for rnd in range(n_rounds):
        for side in order:
            v = {"round": rnd, "side": side}
            v["reloc_before"] = relocate(side, 0.12)
            xy = track[side]["xy"]
            move([xy[0], xy[1], transit_ee_z], "over_" + side + "_" + str(rnd))
            v["reloc_over"] = relocate(side, 0.06)
            xy = track[side]["xy"]
            touch_z = track[side]["top"] + ee_above_bottom + 0.006
            move([xy[0], xy[1], touch_z], "touch_" + side + "_" + str(rnd), tol=0.01, ms=120)
            hold(3, 0.0)
            v["tick_touch"] = now_tick()
            move([xy[0], xy[1], transit_ee_z], "rise_" + side + "_" + str(rnd), tol=0.01, ms=120)
            hold(2, 0.0)
            v["grip"] = round(grip_width(), 3)
            v["xy"] = [round(xy[0], 4), round(xy[1], 4)]
            visits.append(v)
            report["visits"] = visits
            if v["grip"] < 0.12:
                raise ValueError("cube dropped during swing")

    # ---- put down where it was picked ----
    report["phase"] = "put_down"
    move([cube_xy[0], cube_xy[1], transit_ee_z], "over_home_spot")
    move([cube_xy[0], cube_xy[1], table_z + ee_above_bottom + 0.006], "place", tol=0.01, ms=150)
    open_gripper()
    hold(5, 1.0)
    move([cube_xy[0], cube_xy[1], transit_ee_z], "retreat")

    # ---- press the button ----
    report["phase"] = "button"
    if best_button is None:
        raise ValueError("no button candidate")
    bxy = [best_button["stats"]["cx"], best_button["stats"]["cy"]]
    btop = best_button["stats"]["z97"]
    close_gripper()
    move([bxy[0], bxy[1], btop + 0.06], "over_button")
    move([bxy[0], bxy[1], btop + 0.025], "button_pre", tol=0.01, ms=100)
    move([bxy[0], bxy[1], btop - 0.02], "button_press", tol=0.01, ms=60)
    hold(3, 0.0)
    move([bxy[0], bxy[1], btop + 0.06], "button_release", tol=0.01, ms=100)
    hold(10, 0.0)
    report["phase"] = "done"
except Exception as exc:
    report["errors"].append("main: " + str(exc)[:300])
    if "ended" in str(exc).lower() or "EPISODE_ENDED" in str(exc):
        flags["ended"] = True

if not flags["ended"]:
    try:
        report["final_tick"] = now_tick()
    except Exception as exc:
        report["errors"].append("final_tick: " + str(exc)[:120])
result = report
