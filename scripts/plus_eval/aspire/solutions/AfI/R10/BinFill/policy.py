# v05 — BinFill: v04 with an "ambush" grasp on moving cubes (ep2: v04's jaw-along-belt descent landed a finger on
#        a 35°-rotated cube and pinned it; move timeout, miss). The jaw closing axis is now the cube face axis nearest
#        PERPENDICULAR to the belt, so the cube slides between the open fingers; the gripper waits at grasp height at
#        a downstream point and closes just before the predicted (re-measured when visible) arrival. Static branch
#        (measured speed <= 0.0005 m/tick, ep0) is unchanged from v04.
# v04 — BinFill: v03 + moving-cube interception. The cubes ride a conveyor in some episodes (ep1: ~0.65 px/tick
#        in the front image); the program measures cube velocity at run time from two localizations, predicts the
#        intercept, and aligns the jaw closing axis with the motion. Static scenes measure ~0 velocity.
# v03 — BinFill: act as soon as a hue-verified target cube rests on the table (no stability wait: the cube field
#        keeps changing); any still-needed colour may be picked next; fixed provider_unavailable code match.
# v02 — BinFill: v01 + explicit handling of provider_unavailable segmentation (bounded perception-only retry).
# v01 — BinFill: v00 + wait for the cube spawn phase to settle + hue-verified cube/bin/button selection.
# Only public ASPIRE APIs. All object locations come from SAM3 masks + front depth at runtime.
import math

report = {"version": "v05", "events": [], "provider_errors": []}


def log(name, value=None):
    report["events"].append([name, value])


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def rnd(value, digits=4):
    try:
        return round(float(value), digits)
    except Exception:
        return None


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def qrot(q, v):
    qv = (0.0, v[0], v[1], v[2])
    qc = (q[0], -q[1], -q[2], -q[3])
    r = qmul(qmul(q, qv), qc)
    return (r[1], r[2], r[3])


def qz(angle):
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


# ---------------------------------------------------------------- exploration
context = get_task_context()
goal = str(context.language)
report["goal"] = goal
state0 = get_robot_state()
base = state0.base_frame
report["base_frame"] = base
try:
    meta = robomme_plus.get_task_metadata()
    report["task_metadata"] = str(meta)
except Exception as exc:
    meta = {}
    report["task_metadata_error"] = str(exc)
try:
    report["controller_metadata"] = str(robomme_plus.get_controller_metadata())
except Exception as exc:
    report["controller_metadata_error"] = str(exc)
max_steps = 1000
try:
    max_steps = int(meta["max_steps"])
except Exception:
    pass
report["max_steps"] = max_steps

ee0 = state0.end_effector_poses["primary"]
q_home = tuple(float(v) for v in ee0.quaternion_wxyz)
report["home_ee"] = [[rnd(v) for v in ee0.position], [rnd(v) for v in q_home]]
report["home_joints"] = [rnd(v) for v in state0.joint_positions["primary"]]
report["home_gripper"] = rnd(state0.gripper_positions["primary"])
tool_z = qrot(q_home, (0.0, 0.0, 1.0))
report["home_tool_z_axis"] = [rnd(v) for v in tool_z]
q_down = q_home
if tool_z[2] > -0.9:
    q_down = (0.0, 1.0, 0.0, 0.0)
    log("home_not_downward_using_canonical", [rnd(v) for v in tool_z])


def small_image(rgb, grid):
    h = len(rgb)
    w = len(rgb[0])
    rows = []
    for gy in range(grid):
        y = int((gy + 0.5) * h / grid)
        row = []
        for gx in range(grid):
            x = int((gx + 0.5) * w / grid)
            px = rgb[y][x]
            row.append([int(px[0]), int(px[1]), int(px[2])])
        rows.append(row)
    return rows


try:
    video = robomme_plus.get_conditioning_video()
    nvid = len(video)
    report["video_len"] = nvid
    if nvid > 0:
        report["video_t0"] = rnd(video[0].timestamp_s)
        report["video_t1"] = rnd(video[nvid - 1].timestamp_s)
        frames = []
        picks = sorted(set(int(i * (nvid - 1) / 5) for i in range(6)))
        for idx in picks:
            fr = video[idx]
            frames.append({
                "index": idx,
                "t": rnd(fr.timestamp_s),
                "front32": small_image(fr.cameras["front"].rgb, 32),
                "ee": [rnd(v) for v in fr.robot_state.end_effector_poses["primary"].position],
                "grip": rnd(fr.robot_state.gripper_positions["primary"]),
            })
        report["video_frames"] = frames
except Exception as exc:
    report["video_error"] = str(exc)

obs0 = get_observation()
report["cameras"] = [str(name) for name in obs0.cameras]
try:
    front = obs0.cameras["front"]
    report["front_pose"] = [[rnd(v) for v in front.camera_pose.position],
                            [rnd(v) for v in front.camera_pose.quaternion_wxyz]]
    report["front_intrinsics"] = [[rnd(v) for v in row] for row in front.intrinsics]
    report["front32_live"] = small_image(front.rgb, 32)
except Exception as exc:
    report["front_info_error"] = str(exc)


def cloud_stats(points):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    zs = [float(p[2]) for p in points]
    return {
        "n": len(xs),
        "x": [rnd(quantile(xs, 0.05)), rnd(quantile(xs, 0.5)), rnd(quantile(xs, 0.95))],
        "y": [rnd(quantile(ys, 0.05)), rnd(quantile(ys, 0.5)), rnd(quantile(ys, 0.95))],
        "z": [rnd(quantile(zs, 0.05)), rnd(quantile(zs, 0.5)), rnd(quantile(zs, 0.95))],
    }


def project(candidate, camera):
    try:
        cloud = mask_to_point_cloud(candidate, camera, target_frame=base)
        return cloud
    except Exception:
        pass
    try:
        cloud = mask_to_point_cloud(candidate.mask, camera, target_frame=base)
        return cloud
    except Exception:
        return None


def mean_color(cloud):
    try:
        cols = cloud.colors
        n = len(cols)
        if n == 0:
            return None
        stride = max(1, n // 200)
        r = 0.0
        g = 0.0
        b = 0.0
        k = 0
        for i in range(0, n, stride):
            c = cols[i]
            r += float(c[0])
            g += float(c[1])
            b += float(c[2])
            k += 1
        return [rnd(r / k, 1), rnd(g / k, 1), rnd(b / k, 1)]
    except Exception:
        return None


def candidates(camera, prompt, limit=6):
    out = []
    found = None
    for attempt in range(3):
        # Perception does not advance the control clock, so re-asking is safe; robot actions are never retried.
        found = segment_text(camera, prompt)
        if found.ok:
            break
        code = "unknown"
        try:
            code = str(found.error.code)
        except Exception:
            pass
        if len(report["provider_errors"]) < 40:
            report["provider_errors"].append([prompt, attempt, code])
        if "unavailable" not in code.lower():
            break
    if found is None or not found.ok:
        return out
    for cand in found.segmentations[:limit]:
        entry = {"prompt": prompt, "score": rnd(cand.score, 3), "cand": cand}
        try:
            entry["box"] = [rnd(v, 1) for v in cand.box_xyxy]
        except Exception:
            entry["box"] = None
        cloud = project(cand, camera)
        if cloud is not None and len(cloud.points) >= 15:
            entry["cloud"] = cloud
            entry["stats"] = cloud_stats(cloud.points)
            entry["color"] = mean_color(cloud)
        out.append(entry)
    return out


def public(entry):
    return {k: entry[k] for k in ("prompt", "score", "box", "stats", "color") if k in entry}


probes = []
for prompt in ("cube", "red cube", "blue cube", "green cube", "yellow cube", "bin", "box",
               "button", "table"):
    try:
        cs = candidates("front", prompt, 5)
        probes.append({"prompt": prompt, "cands": [public(c) for c in cs]})
    except Exception as exc:
        probes.append({"prompt": prompt, "error": str(exc)})
report["probes"] = probes

# ---------------------------------------------------------------- goal parsing
NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "a": 1, "an": 1, "single": 1, "once": 1,
           "twice": 2, "thrice": 3}
for digit in range(1, 11):
    NUMBERS[str(digit)] = digit
COLORS = ("red", "green", "blue", "yellow", "purple", "orange", "white", "black", "pink",
          "brown", "gray", "grey", "cyan", "magenta")
NOUNS = ("cube", "cubes", "block", "blocks", "box", "boxes")

text = goal.lower()
for ch in ".,;:!?()\"'":
    text = text.replace(ch, " ")
tokens = text.split()
clauses = []
for i in range(len(tokens)):
    tok = tokens[i]
    if tok not in NUMBERS:
        continue
    color = None
    noun_found = False
    for j in range(i + 1, min(len(tokens), i + 4)):
        if tokens[j] in COLORS and color is None:
            color = tokens[j]
        if tokens[j] in NOUNS:
            noun_found = True
            break
    if noun_found:
        clauses.append([NUMBERS[tok], color])
if len(clauses) == 0:
    for c in COLORS:
        if c in tokens:
            clauses.append([1, c])
    if len(clauses) == 0:
        clauses.append([1, None])
press_button = ("button" in tokens) or ("press" in tokens)
press_first = False
if press_button:
    pi = text.find("press")
    fi = min([k for k in (text.find("put"), text.find("place"), text.find("fill"),
                          text.find("pick")) if k >= 0] + [10 ** 6])
    press_first = pi >= 0 and pi < fi
report["plan"] = {"clauses": clauses, "press_button": press_button, "press_first": press_first}


# ---------------------------------------------------------------- perception helpers
def ticks_used():
    try:
        return int(round(float(get_robot_state().timestamp_s) * 20.0))
    except Exception:
        return -1


def hue_sat_val(rgb):
    r = float(rgb[0]) / 255.0
    g = float(rgb[1]) / 255.0
    b = float(rgb[2]) / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx <= 0.0:
        return 0.0, 0.0, 0.0
    d = mx - mn
    sat = d / mx
    if d <= 1e-6:
        return 0.0, sat, mx
    if mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * ((b - r) / d + 2.0)
    else:
        h = 60.0 * ((r - g) / d + 4.0)
    return h, sat, mx


# Hue bands in degrees (lighting-independent; measured cube means are saturated > 0.9).
HUE_BANDS = {"red": ((0.0, 15.0), (345.0, 360.1)), "orange": ((15.0, 45.0),),
             "yellow": ((45.0, 75.0),), "green": ((75.0, 165.0),), "cyan": ((165.0, 200.0),),
             "blue": ((200.0, 255.0),), "purple": ((255.0, 300.0),),
             "pink": ((300.0, 345.0),), "magenta": ((300.0, 345.0),)}


def color_name(rgb):
    if rgb is None:
        return None
    h, s, v = hue_sat_val(rgb)
    if s < 0.45 or v < 0.12:
        return "neutral"
    for name in ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink"):
        for band in HUE_BANDS[name]:
            if band[0] <= h < band[1]:
                return name
    return "neutral"


def color_ok(color, rgb):
    if rgb is None:
        return False
    name = color_name(rgb)
    if color is None:
        return name != "neutral"
    if color in ("grey", "gray", "white", "black", "brown"):
        return True
    if color == "magenta":
        return name == "pink"
    return name == color


def span_of(st):
    return max(st["x"][2] - st["x"][0], st["y"][2] - st["y"][0])


def locate_bin():
    best = None
    seen = []
    for prompt in ("box", "bin", "container"):
        for entry in candidates("front", prompt, 6):
            if "stats" not in entry:
                continue
            st = entry["stats"]
            sp = span_of(st)
            name = color_name(entry.get("color"))
            seen.append([prompt, entry["score"], rnd(sp), name, st["z"][2]])
            if sp < 0.075 or sp > 0.22:
                continue
            if st["z"][2] < 0.028 or st["z"][2] > 0.09:
                continue
            if math.hypot(st["x"][1], st["y"][1]) < 0.15:
                continue
            if name != "neutral":
                continue
            key = float(entry["score"])
            if best is None or key > best[0]:
                best = (key, entry)
    log("bin_candidates", seen)
    if best is None:
        return None
    st = best[1]["stats"]
    zs = [float(p[2]) for p in best[1]["cloud"].points]
    info = {
        "prompt": best[1]["prompt"], "score": best[1]["score"],
        "cx": 0.5 * (st["x"][0] + st["x"][2]), "cy": 0.5 * (st["y"][0] + st["y"][2]),
        "x0": st["x"][0], "x1": st["x"][2], "y0": st["y"][0], "y1": st["y"][2],
        "rim": quantile(zs, 0.97), "stats": st,
    }
    return info


def in_bin(x, y, bin_info, margin):
    if bin_info is None:
        return False
    return (bin_info["x0"] - margin <= x <= bin_info["x1"] + margin
            and bin_info["y0"] - margin <= y <= bin_info["y1"] + margin)


def locate_button(bin_info):
    best = None
    seen = []
    for prompt in ("button", "round button", "push button"):
        for entry in candidates("front", prompt, 5):
            if "stats" not in entry:
                continue
            st = entry["stats"]
            sp = span_of(st)
            name = color_name(entry.get("color"))
            seen.append([prompt, entry["score"], rnd(sp), name, st["z"][2]])
            if sp > 0.08 or st["z"][2] < 0.015 or st["z"][2] > 0.09:
                continue
            if in_bin(st["x"][1], st["y"][1], bin_info, 0.0):
                continue
            key = float(entry["score"])
            if best is None or key > best[0]:
                best = (key, entry)
        if best is not None:
            break
    log("button_candidates", seen)
    if best is None:
        return None
    pts = best[1]["cloud"].points
    zs = [float(p[2]) for p in pts]
    ztop = quantile(zs, 0.95)
    top = [p for p in pts if float(p[2]) > ztop - 0.01]
    return {"prompt": best[1]["prompt"], "score": best[1]["score"],
            "x": quantile([p[0] for p in top], 0.5), "y": quantile([p[1] for p in top], 0.5),
            "ztop": ztop, "stats": best[1]["stats"]}


def cube_yaw(top_points):
    best = None
    for k in range(18):
        a = math.radians(5.0 * k)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [float(p[0]) * ca + float(p[1]) * sa for p in top_points]
        vs = [-float(p[0]) * sa + float(p[1]) * ca for p in top_points]
        area = (quantile(us, 0.95) - quantile(us, 0.05)) * (quantile(vs, 0.95) - quantile(vs, 0.05))
        if best is None or area < best[0]:
            best = (area, a)
    return best[1]


def resting_cubes(color, bin_info):
    """Hue-verified, resting (not falling), out-of-bin cube candidates of one color, deduped."""
    prompts = ["cube"]
    if color is not None:
        prompts = [color + " cube", "cube"]
    pool = []
    for prompt in prompts:
        for entry in candidates("front", prompt, 8):
            if "stats" not in entry:
                continue
            st = entry["stats"]
            sp = span_of(st)
            if sp < 0.012 or sp > 0.08:
                continue
            if st["z"][2] > 0.09 or st["z"][1] > 0.07:
                continue
            if in_bin(st["x"][1], st["y"][1], bin_info, 0.01):
                continue
            if not color_ok(color, entry.get("color")):
                continue
            dup = False
            for other in pool:
                ost = other[1]["stats"]
                if abs(ost["x"][1] - st["x"][1]) < 0.02 and abs(ost["y"][1] - st["y"][1]) < 0.02:
                    dup = True
                    if float(entry["score"]) > other[0]:
                        other[0] = float(entry["score"])
                    break
            if not dup:
                pool.append([float(entry["score"]), entry])
    return pool


def cube_from_entry(entry, n_pool):
    pts = entry["cloud"].points
    zs = [float(p[2]) for p in pts]
    ztop = quantile(zs, 0.95)
    zbot = quantile(zs, 0.05)
    top = [p for p in pts if float(p[2]) > ztop - 0.01]
    if len(top) < 8:
        top = pts
    cx = quantile([p[0] for p in top], 0.5)
    cy = quantile([p[1] for p in top], 0.5)
    yaw = cube_yaw(top)
    return {"prompt": entry["prompt"], "score": entry["score"], "color": entry.get("color"),
            "x": cx, "y": cy, "ztop": ztop, "zbot": zbot, "yaw": yaw, "stats": entry["stats"],
            "n_candidates": n_pool}


def locate_cube(color, bin_info):
    pool = resting_cubes(color, bin_info)
    if len(pool) == 0:
        return None
    pool = sorted(pool, key=lambda item: -item[0])
    return cube_from_entry(pool[0][1], len(pool))


# ---------------------------------------------------------------- motion helpers
gripper_yaw0 = None
yaxis = qrot(q_down, (0.0, 1.0, 0.0))
gripper_yaw0 = math.atan2(yaxis[1], yaxis[0])


def grasp_quat(yaw):
    delta = yaw - gripper_yaw0
    while delta > math.pi / 4:
        delta -= math.pi / 2
    while delta < -math.pi / 4:
        delta += math.pi / 2
    return qmul(qz(delta), q_down)


ended = {"flag": False}


def check_end(res, label):
    try:
        if bool(res.terminated) or bool(res.truncated):
            ended["flag"] = True
            log("episode_end", label)
    except Exception:
        pass


def move(pos, quat, label, tol=0.01, steps=150):
    if ended["flag"]:
        return None
    res = move_to_pose(Pose((float(pos[0]), float(pos[1]), float(pos[2])), quat, base),
                       tolerance=tol, max_steps=steps)
    info = {"ok": bool(res.ok), "steps": None, "status": None}
    try:
        info["steps"] = int(res.steps_executed)
        info["status"] = str(res.status)
    except Exception:
        pass
    log("move:" + label, info)
    check_end(res, label)
    return res


def hold(ticks, grip):
    for tick in range(ticks):
        if ended["flag"]:
            return
        st = get_robot_state()
        res = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=st.joint_positions["primary"],
            gripper_position=grip, embodiment=st.embodiment)}))
        check_end(res, "hold")


def gripper(open_it):
    if ended["flag"]:
        return
    res = open_gripper() if open_it else close_gripper()
    check_end(res, "open" if open_it else "close")
    log("gripper_open" if open_it else "gripper_close",
        rnd(get_robot_state().gripper_positions["primary"]))


def press(button):
    if button is None:
        log("press_skipped_no_button")
        return False
    gripper(False)
    above = (button["x"], button["y"], button["ztop"] + 0.10)
    move(above, q_down, "button_above", 0.01, 150)
    move((button["x"], button["y"], button["ztop"] + 0.03), q_down, "button_near", 0.01, 80)
    move((button["x"], button["y"], button["ztop"] - 0.015), q_down, "button_press", 0.005, 60)
    hold(5, 0.0)
    move(above, q_down, "button_lift", 0.02, 80)
    return True


def line_quat(angle):
    """Top-down grasp whose jaw closing axis (tool y) lies along a world-frame line at `angle`."""
    delta = angle - gripper_yaw0
    while delta > math.pi / 2:
        delta -= math.pi
    while delta <= -math.pi / 2:
        delta += math.pi
    return qmul(qz(delta), q_down)


def snapshot(colors, bin_info):
    """Resting, hue-verified cubes of the given colours: list of [x, y, colour_name, score, entry, wanted_colour]."""
    out = []
    for color in colors:
        for item in resting_cubes(color, bin_info):
            entry = item[1]
            pts = entry["cloud"].points
            zs = [float(p[2]) for p in pts]
            ztop = quantile(zs, 0.95)
            top = [p for p in pts if float(p[2]) > ztop - 0.01]
            if len(top) < 8:
                top = pts
            x = quantile([p[0] for p in top], 0.5)
            y = quantile([p[1] for p in top], 0.5)
            dup = False
            for o in out:
                if abs(o[0] - x) < 0.02 and abs(o[1] - y) < 0.02:
                    dup = True
            if not dup:
                out.append([x, y, color_name(entry.get("color")), item[0], entry, color])
    return out


def match_velocity(s0, t0, s1, t1, samples):
    dt = float(t1 - t0)
    if dt <= 0:
        return
    for a in s1:
        best = None
        for b in s0:
            if a[2] != b[2]:
                continue
            d = math.hypot(a[0] - b[0], a[1] - b[1])
            if d < 0.006 * dt + 0.02 and (best is None or d < best[0]):
                best = (d, b)
        if best is not None:
            b = best[1]
            samples.append([(a[0] - b[0]) / dt, (a[1] - b[1]) / dt])
    while len(samples) > 40:
        samples.pop(0)


def belt_velocity(samples):
    if len(samples) == 0:
        return (0.0, 0.0)
    return (quantile([s[0] for s in samples], 0.5), quantile([s[1] for s in samples], 0.5))


# ---------------------------------------------------------------- policy
SAFE_Z = 0.22
IDLE_TICKS = 10           # ticks held when no requested cube is visible yet (cubes arrive over time)
MEASURE_TICKS = 8         # ticks between the two localizations that give the cube velocity
AMBUSH_MARGIN = 8.0       # moving cube: planned arrival this many ticks after the descent is expected to finish
CLOSE_EARLY = 3.0         # start closing this many ticks before the cube centre reaches the jaw centre (close ~6 ticks)
placed = []
vel_samples = []
lead = {"pre": 24.0, "desc": 10.0}   # measured move durations (ticks), updated after every move
try:
    remaining = {}
    order = []
    for clause in clauses:
        key = str(clause[1])
        if key not in remaining:
            remaining[key] = 0
            order.append(clause[1])
        remaining[key] += int(clause[0])
    report["remaining_start"] = dict(remaining)

    bin_info = locate_bin()
    tries = 0
    while bin_info is None and not ended["flag"] and tries < 20:
        tries += 1
        hold(IDLE_TICKS, 1.0)
        bin_info = locate_bin()
    report["bin"] = None if bin_info is None else {k: (rnd(v) if k not in ("prompt", "stats") else v)
                                                   for k, v in bin_info.items()}
    if bin_info is None:
        raise ValueError("bin not found")
    if press_button and press_first:
        press(locate_button(bin_info))
        go_home()
    drop_z = max(SAFE_Z, bin_info["rim"] + 0.08)
    idle = 0
    misses = 0
    picks = []
    while sum(remaining[str(c)] for c in order) > 0 and not ended["flag"]:
        if ticks_used() > max_steps - 120:
            log("tick_budget_stop", ticks_used())
            break
        needed = [c for c in order if remaining[str(c)] > 0]
        t0 = ticks_used()
        s0 = snapshot(needed + [None], bin_info)
        hold(MEASURE_TICKS, 1.0)
        t1 = ticks_used()
        s1 = snapshot(needed + [None], bin_info)
        match_velocity(s0, t0, s1, t1, vel_samples)
        v = belt_velocity(vel_samples)
        speed = math.hypot(v[0], v[1])
        cands = [c for c in s1 if c[5] is not None and c[2] == c[5]]
        if len(cands) == 0:
            for c in s1:
                for want in needed:
                    if color_ok(want, c[4].get("color")) and c[5] is None:
                        c[5] = want
                        cands.append(c)
                        break
        if len(cands) == 0:
            idle += 1
            if idle <= 3 or idle % 10 == 0:
                log("no_target_visible", [t1, rnd(v[0], 5), rnd(v[1], 5), len(vel_samples)])
            hold(IDLE_TICKS - MEASURE_TICKS if IDLE_TICKS > MEASURE_TICKS else 1, 1.0)
            continue
        # most upstream cube along the motion (longest time left on the conveyor); best score if static
        if speed > 0.0005:
            cands = sorted(cands, key=lambda c: (c[0] * v[0] + c[1] * v[1]) / speed)
        else:
            cands = sorted(cands, key=lambda c: -c[3])
        target = cands[0]
        color = target[5]
        cube = cube_from_entry(target[4], len(cands))
        tx = cube["x"]
        ty = cube["y"]
        tz = cube["ztop"]
        gz = max(cube["ztop"] - 0.02, 0.5 * (cube["ztop"] + cube["zbot"]))
        if speed > 0.0005:
            # ---- moving cube: ambush (v05). Closing axis = cube face axis nearest perpendicular to the belt.
            belt = math.atan2(v[1], v[0])
            ax = cube["yaw"]
            if abs(math.sin(ax + math.pi / 2 - belt)) > abs(math.sin(ax - belt)):
                ax = ax + math.pi / 2
            quat = line_quat(ax)
            ux = v[0] / speed
            uy = v[1] / speed
            info = {"want": color, "t_seen": t1, "x": rnd(tx), "y": rnd(ty), "vx": rnd(v[0], 5), "vy": rnd(v[1], 5),
                    "nv": len(vel_samples), "lead": [rnd(lead["pre"], 1), rnd(lead["desc"], 1)],
                    "yaw_deg": rnd(math.degrees(cube["yaw"]), 1), "close_axis_deg": rnd(math.degrees(ax), 1)}
            gripper(True)
            ta = ticks_used()
            k_arr = (ta - t1) + lead["pre"] + lead["desc"] + AMBUSH_MARGIN
            amb = (tx + v[0] * k_arr, ty + v[1] * k_arr)
            info["ambush"] = [rnd(amb[0]), rnd(amb[1]), rnd(k_arr, 1)]
            move((amb[0], amb[1], tz + 0.10), quat, "pregrasp", 0.01, 150)
            tb = ticks_used()
            if tb - ta > 0:
                lead["pre"] = 0.5 * lead["pre"] + 0.5 * (tb - ta)
            res = move((amb[0], amb[1], gz), quat, "grasp", 0.006, 60)
            tc = ticks_used()
            if tc - tb > 0 and tc - tb < 40:
                lead["desc"] = 0.5 * lead["desc"] + 0.5 * (tc - tb)
            # ticks until the cube centre is CLOSE_EARLY ticks upstream of the jaw centre (predicted)
            wait = int(round((t1 + k_arr) - tc - CLOSE_EARLY))
            info["wait_pred"] = wait
            # re-measure the arrival from a fresh localization (perception does not advance the clock)
            fresh = snapshot([color], bin_info)
            best = None
            for c in fresh:
                if c[2] != c[5]:
                    continue
                across = abs((c[0] - amb[0]) * uy - (c[1] - amb[1]) * ux)
                along = (amb[0] - c[0]) * ux + (amb[1] - c[1]) * uy
                if across < 0.03 and along > -0.03 and (best is None or along < best[1]):
                    best = (across, along)
            if best is not None:
                w2 = int(round(best[1] / speed - CLOSE_EARLY))
                info["wait_seen"] = w2
                info["across_seen"] = rnd(best[0])
                if abs(w2 - wait) <= 12:
                    wait = w2
            if wait > 60:
                wait = 60
            info["wait"] = wait
            log("cube", info)
            if wait > 0:
                hold(wait, 1.0)
            gripper(False)
            gpos = (amb[0], amb[1], gz)
        else:
            quat = grasp_quat(cube["yaw"])
        if speed <= 0.0005:
          now = ticks_used()
          info = {"want": color, "t_seen": t1, "t_now": now, "x": rnd(tx), "y": rnd(ty), "vx": rnd(v[0], 5),
                  "vy": rnd(v[1], 5), "nv": len(vel_samples), "lead": [rnd(lead["pre"], 1), rnd(lead["desc"], 1)]}
          gripper(True)
          ta = ticks_used()
          k = (ta - t1) + lead["pre"] + lead["desc"]
          pre = (tx + v[0] * k, ty + v[1] * k, tz + 0.10)
          res = move(pre, quat, "pregrasp", 0.01, 150)
          tb = ticks_used()
          if tb - ta > 0:
              lead["pre"] = 0.5 * lead["pre"] + 0.5 * (tb - ta)
          k = (tb - t1) + lead["desc"] + 2.0
          gpos = (tx + v[0] * k, ty + v[1] * k, gz)
          info["pred_grasp"] = [rnd(gpos[0]), rnd(gpos[1]), rnd(k, 1)]
          log("cube", info)
          move(gpos, quat, "grasp", 0.006, 60)
          tc = ticks_used()
          if tc - tb > 0 and tc - tb < 40:
              lead["desc"] = 0.5 * lead["desc"] + 0.5 * (tc - tb)
          gripper(False)
        move((gpos[0], gpos[1], SAFE_Z), quat, "lift", 0.015, 100)
        if ended["flag"]:
            break
        width = float(get_robot_state().gripper_positions["primary"])
        log("held_width", rnd(width))
        if width < 0.05:
            misses += 1
            log("grasp_missed", color)
            gripper(True)
            go_home()
            if misses > 8:
                break
            continue
        move((bin_info["cx"], bin_info["cy"], drop_z), quat, "over_bin", 0.015, 150)
        gripper(True)
        hold(4, 1.0)
        remaining[str(color)] -= 1
        placed.append(color)
        res = go_home()
        check_end(res, "home")
    if press_button and not press_first and not ended["flag"]:
        button = locate_button(bin_info)
        report["button_late"] = None if button is None else {
            k: (rnd(v) if k not in ("prompt", "stats") else v) for k, v in button.items()}
        press(button)
        hold(10, 0.0)
except Exception as exc:
    report["error"] = str(exc)

report["placed"] = placed
report["ended"] = ended["flag"]
report["velocity_samples"] = [[rnd(s[0], 5), rnd(s[1], 5)] for s in vel_samples]
try:
    report["final_ticks"] = ticks_used()
except Exception:
    pass
result = report
