# SwingXtimes v02 — v01 + IK fallback chain for long transits and no-skip visit retry.
# (v01: v00 + closed-loop target tracking at hover height (targets can move).)
# (v00 header follows)
# SwingXtimes v00 — initial program from offline scene analysis of the released recordings.
# Pick the goal-coloured cube, alternate visits over the two bullseye targets (first side named in
# the goal) 2*N times holding the cube low over each disc, put the cube down where it started,
# then press the button with a closed gripper. Everything is derived at runtime.
import math

report = {"version": "v02", "log": []}


def note(item):
    if len(report["log"]) < 120:
        report["log"].append(item)


ctx = get_task_context()
goal = str(ctx.language).lower()
report["goal"] = goal
state0 = get_robot_state()
BASE = state0.base_frame
EMB = state0.embodiment
QDOWN = list(state0.end_effector_poses["primary"].quaternion_wxyz)
HOME_EE = list(state0.end_effector_poses["primary"].position)
report["home_ee"] = [round(float(v), 3) for v in HOME_EE]
report["qdown"] = [round(float(v), 3) for v in QDOWN]

max_steps = None
try:
    meta = robomme_plus.get_task_metadata()
    max_steps = int(meta["max_steps"])
    report["meta_keys"] = sorted([str(k) for k in meta])
except Exception as exc:
    report["meta_error"] = str(exc)[:200]
report["max_steps"] = max_steps
try:
    video = robomme_plus.get_conditioning_video()
    report["video_len"] = len(video)
except Exception as exc:
    report["video_error"] = str(exc)[:120]

# ---------------- goal parsing ----------------
NUMBERS = {"one": 1, "once": 1, "two": 2, "twice": 2, "three": 3, "thrice": 3, "four": 4,
           "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "eleven": 11, "twelve": 12}
words = goal.replace(",", " ").replace(".", " ").split()
color = None
for i, w in enumerate(words):
    if w in ("cube", "block") and i > 0:
        color = words[i - 1]
        break
first_side = None
for w in words:
    if w.startswith("right"):
        first_side = "right"
        break
    if w.startswith("left"):
        first_side = "left"
        break
count = None
for i, w in enumerate(words):
    if w in ("once", "twice", "thrice"):
        count = NUMBERS[w]
        break
    if w in ("times", "time") and i > 0:
        prev = words[i - 1]
        if prev in NUMBERS:
            count = NUMBERS[prev]
            break
        try:
            count = int(prev)
            break
        except Exception:
            pass
report["parsed"] = {"color": color, "first_side": first_side, "count": count}
if color is None or first_side is None or count is None:
    raise ValueError("could not parse goal: " + goal)
second_side = "left" if first_side == "right" else "right"

HUES = {"red": 0.0, "orange": 30.0, "yellow": 60.0, "green": 120.0, "cyan": 180.0,
        "blue": 230.0, "purple": 280.0, "violet": 280.0, "magenta": 310.0, "pink": 330.0}


# ---------------- helpers ----------------
def q(values, frac):
    s = sorted(values)
    if not s:
        return 0.0
    return s[int((len(s) - 1) * frac)]


def hue_sat(rgb):
    r, g, b = float(rgb[0]) / 255.0, float(rgb[1]) / 255.0, float(rgb[2]) / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn
    sat = 0.0 if mx <= 0 else d / mx
    if d < 1e-6:
        return 0.0, sat
    if mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * ((b - r) / d + 2.0)
    else:
        h = 60.0 * ((r - g) / d + 4.0)
    return h, sat


def hue_dist(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def cloud_info(seg, camera):
    pc = mask_to_point_cloud(seg, camera, target_frame=BASE)
    pts = [[float(p[0]), float(p[1]), float(p[2])] for p in pc.points]
    if len(pts) < 5:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    info = {"n": len(pts), "pts": pts,
            "c": [q(xs, 0.5), q(ys, 0.5), q(zs, 0.5)],
            "zlo": q(zs, 0.1), "zhi": q(zs, 0.9), "ztop": q(zs, 0.97),
            "ext": max(q(xs, 0.9) - q(xs, 0.1), q(ys, 0.9) - q(ys, 0.1))}
    col = None
    plist = None
    try:
        cols = pc.colors
        if cols is not None and len(cols) > 0:
            acc = [0.0, 0.0, 0.0]
            plist = []
            for c in cols:
                cc = [float(c[0]), float(c[1]), float(c[2])]
                plist.append(cc)
                acc[0] += cc[0]
                acc[1] += cc[1]
                acc[2] += cc[2]
            col = [acc[0] / len(cols), acc[1] / len(cols), acc[2] / len(cols)]
            if col[0] <= 1.0 and col[1] <= 1.0 and col[2] <= 1.0:
                col = [col[0] * 255.0, col[1] * 255.0, col[2] * 255.0]
                plist = [[v[0] * 255.0, v[1] * 255.0, v[2] * 255.0] for v in plist]
    except Exception:
        col = None
        plist = None
    info["col"] = col
    info["pcols"] = plist
    return info


def candidates(camera, prompt, limit=6):
    out = []
    seg = segment_text(camera, prompt)
    if not seg.ok:
        return out
    for cand in seg.segmentations[:limit]:
        try:
            info = cloud_info(cand, camera)
        except Exception as exc:
            info = None
        if info is None:
            continue
        info["score"] = float(cand.score)
        try:
            info["box"] = [float(v) for v in cand.box_xyxy]
        except Exception:
            info["box"] = None
        info["prompt"] = prompt
        out.append(info)
    return out


def brief(info):
    return {"p": info["prompt"], "s": round(info["score"], 3), "n": info["n"],
            "c": [round(v, 3) for v in info["c"]], "zlo": round(info["zlo"], 3),
            "zhi": round(info["zhi"], 3), "ext": round(info["ext"], 3),
            "col": None if info["col"] is None else [int(v) for v in info["col"]]}


def dist2(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def yaw_rot(qwxyz, ang):
    return quat_mul([math.cos(ang / 2.0), 0.0, 0.0, math.sin(ang / 2.0)], qwxyz)


def finger_axis_angle(qwxyz):
    w, x, y, z = qwxyz
    vx = 2.0 * (x * y - w * z)
    vy = 1.0 - 2.0 * (x * x + z * z)
    return math.atan2(vy, vx)


def wrap90(a):
    h = math.pi / 2.0
    while a > h / 2.0:
        a -= h
    while a < -h / 2.0:
        a += h
    return a


def edge_angle(pts):
    best = None
    for k in range(30):
        a = k * (math.pi / 2.0) / 30.0
        ca = math.cos(a)
        sa = math.sin(a)
        us = [p[0] * ca + p[1] * sa for p in pts]
        vs = [-p[0] * sa + p[1] * ca for p in pts]
        area = (q(us, 0.95) - q(us, 0.05)) * (q(vs, 0.95) - q(vs, 0.05))
        if best is None or area < best[0]:
            best = (area, a)
    return best[1]


ticks = {"n": 0}
state = {"ended": False}


def account(res, label):
    n = 0
    try:
        n = int(res.steps_executed)
    except Exception:
        n = 0
    ticks["n"] += n
    term = False
    try:
        term = bool(res.terminated) or bool(res.truncated)
    except Exception:
        term = False
    ok = False
    try:
        ok = bool(res.ok)
    except Exception:
        ok = False
    note([label, ok, n, ticks["n"]])
    if term:
        state["ended"] = True
        report["ended_at"] = label
    return ok


def goto(pos, quat, label, tol=0.012, steps=200):
    if state["ended"]:
        return False
    res = move_to_pose(Pose([float(pos[0]), float(pos[1]), float(pos[2])], quat, BASE),
                       tolerance=tol, max_steps=steps)
    ok = account(res, label)
    state["last_status"] = ""
    if not ok:
        try:
            state["last_status"] = str(res.status)
            note([label + "_err", str(res.status)[:60], str(res.error)[:120]])
        except Exception:
            pass
    return ok


def ee_pos():
    try:
        return list(get_robot_state().end_effector_poses["primary"].position)
    except Exception:
        state["ended"] = True
        return [0.0, 0.0, 0.0]


def gripper_reading():
    try:
        return float(get_robot_state().gripper_positions["primary"])
    except Exception:
        return -1.0


# ---------------- perception: targets, button, cube ----------------
try:
    cam_pose = get_observation().cameras["front"].camera_pose
    report["front_cam"] = [round(float(v), 3) for v in cam_pose.position]
except Exception as exc:
    report["front_cam_error"] = str(exc)[:100]

pool = []
for prompt in ("target", "button"):
    for info in candidates("front", prompt):
        pool.append(info)
report["pool"] = [brief(i) for i in pool]

# Targets: flat discs. Pair with the largest summed "target" score that is well separated.
flat = []
for info in pool:
    if info["prompt"] == "target" and (info["zhi"] - info["zlo"]) < 0.015 and 0.01 < info["ext"] < 0.15:
        flat.append(info)
pair = None
for i in range(len(flat)):
    for j in range(i + 1, len(flat)):
        if dist2(flat[i]["c"], flat[j]["c"]) < 0.08:
            continue
        sc = flat[i]["score"] + flat[j]["score"]
        if pair is None or sc > pair[0]:
            pair = (sc, flat[i], flat[j])
if pair is None:
    raise ValueError("two separated flat targets not found")
ta, tb = pair[1], pair[2]
if ta["c"][1] > tb["c"][1]:
    targets = {"left": ta, "right": tb}
else:
    targets = {"left": tb, "right": ta}
table_z = min(ta["zlo"], tb["zlo"])
report["targets"] = {k: brief(v) for k, v in targets.items()}
report["table_z"] = round(table_z, 4)

# Button: best "button" candidate not coincident with either target, raised above the table.
button = None
for info in pool:
    if info["prompt"] != "button":
        continue
    if dist2(info["c"], ta["c"]) < 0.04 or dist2(info["c"], tb["c"]) < 0.04:
        continue
    if info["ext"] > 0.12:
        continue
    if info["ztop"] > table_z + 0.25:
        continue
    if button is None or info["score"] > button["score"]:
        button = info
report["button"] = None if button is None else brief(button)

# Cube: goal colour; must stand above the table (rejects flat light spots) and be compact.
cube = None
cube_pool = candidates("front", color + " cube") + candidates("front", "cube")
cube_brief = []
for info in cube_pool:
    ok_geom = (info["ztop"] - table_z) > 0.015 and (info["ztop"] - table_z) < 0.10 and info["ext"] < 0.08
    ok_col = True
    if info["col"] is not None and color in HUES:
        h, s = hue_sat(info["col"])
        ok_col = s > 0.25 and hue_dist(h, HUES[color]) < 40.0
    b = brief(info)
    b["g"] = ok_geom
    b["k"] = ok_col
    cube_brief.append(b)
    if ok_geom and ok_col and (cube is None or info["score"] > cube["score"]):
        cube = info
report["cube_pool"] = cube_brief
if cube is None:
    raise ValueError("cube not found")
report["cube"] = brief(cube)

# ---------------- grasp ----------------
cube_top = cube["ztop"]
cube_xy = [cube["c"][0], cube["c"][1]]
open_res = open_gripper()
account(open_res, "open")
goto([cube_xy[0], cube_xy[1], cube_top + 0.12], QDOWN, "pre_grasp", steps=250)

# Refine centre and yaw from the wrist camera (closer, top face visible).
grasp_q = QDOWN
try:
    wc = candidates("wrist", color + " cube", limit=4)
    best = None
    for info in wc:
        if dist2(info["c"], cube_xy) > 0.05:
            continue
        if (info["ztop"] - table_z) < 0.015:
            continue
        if info["col"] is not None and color in HUES:
            h, s = hue_sat(info["col"])
            if not (s > 0.25 and hue_dist(h, HUES[color]) < 40.0):
                continue
        if best is None or info["score"] > best["score"]:
            best = info
    if best is not None:
        top_pts = [p for p in best["pts"] if p[2] > best["ztop"] - 0.006]
        if len(top_pts) >= 8:
            cube_xy = [q([p[0] for p in top_pts], 0.5), q([p[1] for p in top_pts], 0.5)]
            ea = edge_angle(top_pts)
            fa = finger_axis_angle(QDOWN)
            delta = wrap90(ea - fa)
            grasp_q = yaw_rot(QDOWN, delta)
            report["wrist_refine"] = {"n_top": len(top_pts), "xy": [round(v, 3) for v in cube_xy],
                                      "edge_deg": round(math.degrees(ea), 1),
                                      "delta_deg": round(math.degrees(delta), 1)}
        cube_top = max(cube_top, best["ztop"])
    else:
        report["wrist_refine"] = "no match"
except Exception as exc:
    report["wrist_refine"] = "error " + str(exc)[:100]

cube_h = max(0.02, cube_top - table_z)
grasp_z = table_z + 0.5 * cube_h + 0.003
report["cube_h"] = round(cube_h, 4)
goto([cube_xy[0], cube_xy[1], cube_top + 0.06], grasp_q, "pre_grasp2")
goto([cube_xy[0], cube_xy[1], grasp_z], grasp_q, "grasp_descend", tol=0.008)
account(close_gripper(), "close")
g_read = gripper_reading()
report["grip_after_close"] = round(g_read, 3)
lift_z = cube_top + 0.08
goto([cube_xy[0], cube_xy[1], lift_z], grasp_q, "lift")
report["grip_after_lift"] = round(gripper_reading(), 3)

# EE height above cube bottom while held.
hold_off = grasp_z - table_z
ori = {"q": grasp_q}
fallbacks = []


def goto_robust(pos, label, tol=0.008, steps=250):
    """move_to_pose with a fallback chain for IK failures seeded from a far configuration
    (v01 ep2: PLANNER_FAILED 'PyRoki solution exceeds the public pose residual limits' on a
    pose that had been reached earlier from another start). Failed IK costs ~0 ticks."""
    if goto(pos, ori["q"], label, tol=tol, steps=steps):
        return True
    if state["ended"] or "PLANNER" not in state.get("last_status", ""):
        return False
    try:
        cur = list(get_robot_state().end_effector_poses["primary"].position)
    except Exception:
        state["ended"] = True
        return False
    # 1) intermediate waypoint between start and goal, pulled toward the home xy, raised.
    mid = [0.25 * (cur[0] + pos[0]) + 0.5 * HOME_EE[0], 0.25 * (cur[1] + pos[1]) + 0.5 * HOME_EE[1],
           pos[2] + 0.04]
    if goto(mid, ori["q"], label + "_mid", tol=0.02, steps=250):
        if goto(pos, ori["q"], label + "_r1", tol=tol, steps=steps):
            fallbacks.append([label, "mid"])
            return True
    if state["ended"]:
        return False
    # 2) plain top-down orientation (the held cube's yaw is irrelevant while swinging).
    if goto(pos, QDOWN, label + "_qd", tol=tol, steps=steps):
        ori["q"] = QDOWN
        fallbacks.append([label, "qdown"])
        return True
    if state["ended"]:
        return False
    # 3) via the home xy with the top-down orientation.
    if goto([HOME_EE[0], HOME_EE[1], pos[2] + 0.06], QDOWN, label + "_home", tol=0.02, steps=300):
        ori["q"] = QDOWN
        if goto(pos, QDOWN, label + "_r3", tol=tol, steps=steps):
            fallbacks.append([label, "home"])
            return True
    fallbacks.append([label, "failed"])
    return False

# ---------------- swings (closed-loop: targets can move when the cube approaches) ----------------
# Observed (v00 ep1 + recordings ep1/ep2): a disc can jump several cm while the cube approaches,
# and a visit registers on xy proximity (ep0 highlight fired with the cube ~10 cm above the disc).
# So: hover at a fixed height, re-localize the current target after every move, and stop when its
# disc points turn red (highlight) with the cube above it, or when the cube has stayed over a stable
# target estimate.
sequence = []
for k in range(count):
    sequence.append(first_side)
    sequence.append(second_side)
report["sequence_len"] = len(sequence)
HOVER_BOTTOM = 0.05          # cube bottom above the table while swinging (clears button cap)
hover_z = table_z + hold_off + HOVER_BOTTOM
FLAT_TOL = 0.012
est = {"left": [targets["left"]["c"][0], targets["left"]["c"][1]],
       "right": [targets["right"]["c"][0], targets["right"]["c"][1]]}


def red_fraction(info):
    cols = info["pcols"]
    if cols is None:
        return None
    tot = 0
    red = 0
    for p, c in zip(info["pts"], cols):
        if p[2] > table_z + FLAT_TOL:
            continue
        tot += 1
        if c[0] > 150 and c[1] < 100 and c[2] < 100:
            red += 1
    if tot < 10:
        return None
    return float(red) / float(tot)


def relocate(side):
    prev = est[side]
    other = est["left" if side == "right" else "right"]
    for cam in ("front", "wrist"):
        try:
            cs = candidates(cam, "target", limit=6)
        except Exception:
            cs = []
        best = None
        for c in cs:
            if c["zhi"] > table_z + FLAT_TOL or not (0.01 < c["ext"] < 0.15):
                continue
            d = dist2(c["c"], prev)
            if d > 0.12 or dist2(c["c"], other) < d:
                continue
            if best is None or d < best[0]:
                best = (d, c)
        if best is not None:
            c = best[1]
            est[side] = [c["c"][0], c["c"][1]]
            return {"cam": cam, "jump": round(best[0], 3), "red": red_fraction(c)}
    return None


def hold(n, grip):
    try:
        st = get_robot_state()
        for k in range(n):
            sr = step(RobotAction(arms={"primary": ArmCommand(
                "joint_position", st.joint_positions["primary"],
                gripper_position=grip, embodiment=st.embodiment)}))
            ticks["n"] += 1
            if not sr.ok:
                note(["hold_err", str(sr.error)[:100]])
                break
            if bool(sr.terminated) or bool(sr.truncated):
                state["ended"] = True
                report["ended_at"] = "hold"
                break
    except Exception as exc:
        note(["hold_exc", str(exc)[:100]])
        state["ended"] = True


def ee_xy():
    try:
        p = get_robot_state().end_effector_poses["primary"].position
        return [float(p[0]), float(p[1])]
    except Exception:
        state["ended"] = True
        return None


def seg_clear(a, b, p):
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    vx, vy = bx - ax, by - ay
    L = vx * vx + vy * vy
    t = 0.0 if L < 1e-9 else max(0.0, min(1.0, ((p[0] - ax) * vx + (p[1] - ay) * vy) / L))
    return math.sqrt((ax + t * vx - p[0]) ** 2 + (ay + t * vy - p[1]) ** 2)


def route(a, b, avoid, clear=0.07):
    """Waypoints (xy) from a to b whose segments keep `clear` from every avoid point."""
    if all(seg_clear(a, b, p) >= clear for p in avoid):
        return [b]
    mx, my = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
    vx, vy = b[0] - a[0], b[1] - a[1]
    L = math.sqrt(vx * vx + vy * vy) + 1e-9
    nx, ny = -vy / L, vx / L
    for off in (0.10, -0.10, 0.16, -0.16, 0.22, -0.22):
        w = [mx + nx * off, my + ny * off]
        if all(seg_clear(a, w, p) >= clear and seg_clear(w, b, p) >= clear for p in avoid):
            return [w, b]
    return [b]


visits = []
first_move = True
for idx, side in enumerate(sequence):
    if state["ended"]:
        break
    other_side = "left" if side == "right" else "right"
    start_ticks = ticks["n"]
    rec = {"i": idx, "side": side, "it": []}
    red_at_start = None
    r0 = relocate(side)
    if r0 is not None:
        red_at_start = r0["red"]
    done = False
    stable = 0
    for it in range(14):
        if state["ended"]:
            break
        cur = ee_xy()
        if cur is None:
            break
        goal_xy = [est[side][0], est[side][1]]
        d = dist2(cur, goal_xy)
        if d <= 0.008:
            hold(3, 0.0)
        else:
            if first_move:
                wps = route(cur, goal_xy, [est[other_side]])
                first_move = False
            else:
                wps = [goal_xy]
            for w in wps:
                goto_robust([w[0], w[1], hover_z], "v%d_%d" % (idx, it), tol=0.008, steps=250)
        if state["ended"]:
            break
        r = relocate(side)
        cur = ee_xy()
        if cur is None:
            break
        d2 = dist2(cur, est[side])
        rec["it"].append([round(d, 3), None if r is None else r["jump"],
                          None if r is None else r["cam"],
                          None if (r is None or r["red"] is None) else round(r["red"], 2),
                          round(d2, 3)])
        fresh_red = (r is not None and r["red"] is not None and r["red"] > 0.3
                     and not (red_at_start is not None and red_at_start > 0.3))
        if fresh_red and d2 < 0.03:
            rec["how"] = "red"
            done = True
            break
        if d2 < 0.008:
            stable += 1
            if stable >= 2:
                rec["how"] = "stable"
                done = True
                break
        else:
            stable = 0
    rec["done"] = done
    rec["ticks"] = ticks["n"] - start_ticks
    rec["est"] = [round(v, 3) for v in est[side]]
    if len(rec["it"]) > 8:
        rec["it"] = rec["it"][:4] + rec["it"][-4:]
    visits.append(rec)
    if not done:
        # Advancing would visit the other side out of order (fatal) or miscount; stop swinging.
        report["visit_incomplete"] = idx
        break
report["visits"] = visits
report["fallbacks"] = fallbacks

# ---------------- put down ----------------
if not state["ended"]:
    place_xy = [cube_xy[0], cube_xy[1]]
    for shift in range(5):
        if min(dist2(place_xy, est["left"]), dist2(place_xy, est["right"])) >= 0.08:
            break
        place_xy = [place_xy[0] - 0.04, place_xy[1]]
    report["place_xy"] = [round(v, 3) for v in place_xy]
    cur = ee_xy()
    if cur is not None:
        last_side = sequence[-1]
        other_side = "left" if last_side == "right" else "right"
        for w in route(cur, place_xy, [est[other_side]]):
            goto_robust([w[0], w[1], hover_z], "to_place", tol=0.012, steps=250)
    goto([place_xy[0], place_xy[1], grasp_z + 0.005], ori["q"], "place_descend", tol=0.008)
    if not state["ended"]:
        account(open_gripper(), "release")
    goto([place_xy[0], place_xy[1], lift_z + 0.02], ori["q"], "retreat")

# ---------------- press button ----------------
if not state["ended"] and button is not None:
    account(close_gripper(), "close_for_press")
    btop = button["ztop"]
    goto([button["c"][0], button["c"][1], btop + 0.08], QDOWN, "over_button", steps=250)
    goto([button["c"][0], button["c"][1], btop + 0.02], QDOWN, "button_near", tol=0.008)
    goto([button["c"][0], button["c"][1], btop - 0.015], QDOWN, "press", tol=0.008)
    goto([button["c"][0], button["c"][1], btop + 0.06], QDOWN, "release_button")
elif button is None:
    report["press"] = "skipped: no button"

# Settle a few ticks if still running.
if not state["ended"]:
    try:
        st = get_robot_state()
        for settle in range(10):
            sr = step(RobotAction(arms={"primary": ArmCommand(
                "joint_position", st.joint_positions["primary"],
                gripper_position=float(st.gripper_positions["primary"]), embodiment=st.embodiment)}))
            ticks["n"] += 1
            try:
                if bool(sr.terminated) or bool(sr.truncated):
                    state["ended"] = True
                    report["ended_at"] = "settle%d" % settle
                    break
            except Exception:
                pass
            if not sr.ok:
                report["settle_err"] = str(sr.error)[:120]
                break
    except Exception as exc:
        report["settle_exc"] = str(exc)[:120]

report["ticks_total"] = ticks["n"]
report["ended"] = state["ended"]
result = report
