# SwingXtimes v03 -- v02 plus colour gate on target masks and lift-to-unocclude when the front view loses the target.
# Pick the named cube, alternate it between the two flat targets in the order and
# count parsed from the goal, put it down, then press the button.
# Every task-dependent quantity comes from the goal text and live RGB-D perception.
import math

report = {"version": "v03"}
calls = []
STATE = {"terminal": False, "ticks": 0}

ctx = get_task_context()
goal = str(ctx.language)
report["goal"] = goal
state0 = get_robot_state()
base_frame = state0.base_frame
report["base_frame"] = base_frame
try:
    meta = robomme_plus.get_task_metadata()
    report["task_metadata"] = dict(meta)
except Exception as exc:
    meta = {}
    report["task_metadata_error"] = str(exc)
try:
    report["controller_metadata"] = dict(robomme_plus.get_controller_metadata())
except Exception as exc:
    report["controller_metadata_error"] = str(exc)
try:
    report["conditioning_video_len"] = len(robomme_plus.get_conditioning_video())
except Exception as exc:
    report["conditioning_video_error"] = str(exc)
max_steps_native = None
try:
    max_steps_native = int(meta["max_steps"])
except Exception:
    max_steps_native = None
report["max_steps_native"] = max_steps_native

# ---------------------------------------------------------------- goal parsing
NUMBERS = {"once": 1, "one": 1, "twice": 2, "two": 2, "three": 3, "thrice": 3, "four": 4,
           "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "eleven": 11, "twelve": 12}


def clean_tokens(text):
    out = []
    for raw in text.lower().replace(",", " ").replace(".", " ").split():
        out.append(raw.strip())
    return out


tokens = clean_tokens(goal)
color = None
for index, tok in enumerate(tokens):
    if tok in ("cube", "block") and index > 0:
        color = tokens[index - 1]
        break
sides = []
for tok in tokens:
    if tok.startswith("left-side") or tok == "left":
        sides.append("left")
    elif tok.startswith("right-side") or tok == "right":
        sides.append("right")
first_side = sides[0] if sides else "right"
second_side = "left" if first_side == "right" else "right"
repeat = None
for index, tok in enumerate(tokens):
    if tok in ("once", "twice", "thrice"):
        repeat = NUMBERS[tok]
        break
    if tok in ("times", "time") and index > 0:
        prev = tokens[index - 1]
        if prev in NUMBERS:
            repeat = NUMBERS[prev]
        else:
            try:
                repeat = int(prev)
            except Exception:
                repeat = None
        break
if repeat is None:
    repeat = 1
    report["repeat_parse_failed"] = True
report["parsed"] = {"color": color, "first_side": first_side, "second_side": second_side,
                    "repeat": repeat}


# ---------------------------------------------------------------- helpers
def record(name, res, extra=None):
    entry = {"call": name}
    try:
        entry["ok"] = bool(res.ok)
    except Exception:
        pass
    try:
        entry["status"] = str(res.status)
    except Exception:
        pass
    try:
        entry["steps"] = int(res.steps_executed)
        STATE["ticks"] = STATE["ticks"] + int(res.steps_executed)
    except Exception:
        pass
    term = False
    try:
        if res.terminated:
            term = True
    except Exception:
        pass
    try:
        if res.truncated:
            term = True
    except Exception:
        pass
    try:
        if res.error is not None:
            entry["error"] = str(res.error)[:300]
    except Exception:
        pass
    if extra is not None:
        entry["extra"] = extra
    entry["terminal"] = term
    entry["ticks_total"] = STATE["ticks"]
    if term:
        STATE["terminal"] = True
    calls.append(entry)
    if term:
        raise Exception("EPISODE_TERMINAL after " + name)
    return res


def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
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
    p = (0.0, v[0], v[1], v[2])
    conj = (w, -x, -y, -z)
    r = qmul(qmul(q, p), conj)
    return (r[1], r[2], r[3])


def yaw_quat(angle):
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def wrap90(angle):
    # wrap to [-pi/4, pi/4)
    quarter = math.pi / 2.0
    while angle >= quarter / 2.0:
        angle = angle - quarter
    while angle < -quarter / 2.0:
        angle = angle + quarter
    return angle


def cloud_points(candidate, camera):
    pc = None
    try:
        pc = mask_to_point_cloud(candidate.mask, camera, target_frame=base_frame)
    except Exception:
        pc = mask_to_point_cloud(candidate, camera, target_frame=base_frame)
    cols = None
    try:
        cols = pc.colors
        if cols is not None and len(cols) != len(pc.points):
            cols = None
    except Exception:
        cols = None
    scale = 1.0
    if cols is not None:
        try:
            peak = 0.0
            for k in range(0, len(cols), max(1, len(cols) // 50)):
                peak = max(peak, float(cols[k][0]), float(cols[k][1]), float(cols[k][2]))
            if peak > 1.5:
                scale = 1.0 / 255.0
        except Exception:
            cols = None
    pts = []
    index = 0
    for p in pc.points:
        x = float(p[0])
        y = float(p[1])
        z = float(p[2])
        if x == x and y == y and z == z and abs(x) < 3.0 and abs(y) < 3.0 and abs(z) < 3.0:
            if cols is not None:
                c = cols[index]
                pts.append((x, y, z, float(c[0]) * scale, float(c[1]) * scale, float(c[2]) * scale))
            else:
                pts.append((x, y, z, -1.0, -1.0, -1.0))
        index = index + 1
    return pts


def summarize(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    return {
        "n": len(pts),
        "med": (quantile(xs, 0.5), quantile(ys, 0.5), quantile(zs, 0.5)),
        "z05": quantile(zs, 0.05), "z90": quantile(zs, 0.9), "z95": quantile(zs, 0.95),
        "xext": quantile(xs, 0.9) - quantile(xs, 0.1),
        "yext": quantile(ys, 0.9) - quantile(ys, 0.1),
    }


def candidates(camera, prompts, limit=6):
    found = []
    for prompt in prompts:
        try:
            seg = segment_text(camera, prompt)
        except Exception as exc:
            calls.append({"call": "segment_text", "prompt": prompt, "exception": str(exc)[:200]})
            continue
        if not seg.ok:
            calls.append({"call": "segment_text", "prompt": prompt, "ok": False})
            continue
        count = 0
        for cand in seg.segmentations[:limit]:
            count = count + 1
            entry = {"prompt": prompt, "score": float(cand.score), "cand": cand}
            try:
                box = [float(v) for v in cand.box_xyxy]
                entry["box"] = box
                entry["u"] = 0.5 * (box[0] + box[2])
                entry["v"] = 0.5 * (box[1] + box[3])
            except Exception:
                entry["box"] = None
            try:
                pts = cloud_points(cand, camera)
            except Exception as exc:
                entry["cloud_error"] = str(exc)[:200]
                pts = []
            entry["pts"] = pts
            if len(pts) >= 8:
                entry["geo"] = summarize(pts)
            found.append(entry)
    return found


def public(entry):
    out = {"prompt": entry["prompt"], "score": round(entry["score"], 3), "box": entry.get("box")}
    geo = entry.get("geo")
    if geo is not None:
        out["n"] = geo["n"]
        out["med"] = [round(v, 4) for v in geo["med"]]
        out["z05"] = round(geo["z05"], 4)
        out["z95"] = round(geo["z95"], 4)
        out["xext"] = round(geo["xext"], 4)
        out["yext"] = round(geo["yext"], 4)
    return out


def dxy(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def top_face(pts, top_z, band=0.008):
    face = [p for p in pts if p[2] >= top_z - band]
    if len(face) < 5:
        face = pts
    return face


def min_area_yaw(face):
    if len(face) < 6:
        return None
    best = None
    best_area = None
    for step_index in range(45):
        angle = step_index * (math.pi / 2.0) / 45.0
        c = math.cos(angle)
        s = math.sin(angle)
        us = [p[0] * c + p[1] * s for p in face]
        vs = [-p[0] * s + p[1] * c for p in face]
        area = (quantile(us, 0.95) - quantile(us, 0.05)) * (quantile(vs, 0.95) - quantile(vs, 0.05))
        if best_area is None or area < best_area:
            best_area = area
            best = angle
    return best


home_pose = state0.end_effector_poses["primary"]
home_quat = tuple(float(v) for v in home_pose.quaternion_wxyz)
report["home_ee"] = {"pos": [round(float(v), 4) for v in home_pose.position], "quat": [round(v, 4) for v in home_quat]}
report["home_tool_z_axis"] = [round(v, 3) for v in qrot(home_quat, (0.0, 0.0, 1.0))]
closing_axis = qrot(home_quat, (0.0, 1.0, 0.0))
closing_yaw = math.atan2(closing_axis[1], closing_axis[0])


def go(pos, quat, name, tol=0.01, max_steps=200):
    if STATE["terminal"]:
        raise Exception("EPISODE_TERMINAL")
    target = Pose((float(pos[0]), float(pos[1]), float(pos[2])), tuple(quat), base_frame)
    res = move_to_pose(target, tolerance=tol, max_steps=max_steps)
    record(name, res, {"target": [round(float(v), 4) for v in pos]})
    try:
        if (not res.ok) and int(res.steps_executed) == 0:
            res = move_to_pose(target, tolerance=tol, max_steps=max_steps * 2)
            record(name + "_retry", res)
    except Exception as exc:
        if "EPISODE_TERMINAL" in str(exc):
            raise exc
    return res


def hold(ticks, gripper, name):
    st = get_robot_state()
    executed = 0
    for tick_index in range(ticks):
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=st.joint_positions["primary"],
            gripper_position=gripper, embodiment=st.embodiment)}))
        executed = executed + 1
        term = False
        try:
            if reply.terminated or reply.truncated:
                term = True
        except Exception:
            pass
        if term:
            STATE["ticks"] = STATE["ticks"] + executed
            calls.append({"call": name, "steps": executed, "terminal": True})
            STATE["terminal"] = True
            raise Exception("EPISODE_TERMINAL after " + name)
    STATE["ticks"] = STATE["ticks"] + executed
    calls.append({"call": name, "steps": executed, "ticks_total": STATE["ticks"]})


def by_score(entry):
    return -entry["score"]


def ee_now():
    st = get_robot_state()
    pose = st.end_effector_poses["primary"]
    return [round(float(v), 4) for v in pose.position], round(float(st.gripper_positions["primary"]), 3)


# ---------------------------------------------------------------- perception (arm at home)
def run_program():
    cube_prompts = []
    if color is not None:
        cube_prompts.append(color + " cube")
        cube_prompts.append(color + " block")
    cube_prompts.append("cube")
    target_prompts = ["target", "bullseye target", "white round target"]
    button_prompts = ["button", "push button"]

    button_c = candidates("front", button_prompts)
    target_c = candidates("front", target_prompts)
    cube_c = candidates("front", cube_prompts)
    report["button_candidates"] = [public(e) for e in button_c]
    report["target_candidates"] = [public(e) for e in target_c]
    report["cube_candidates"] = [public(e) for e in cube_c]

    # Button: best-scoring compact candidate from button prompts.
    button = None
    for e in sorted(button_c, key=by_score):
        g = e.get("geo")
        if g is None:
            continue
        if g["xext"] < 0.15 and g["yext"] < 0.15:
            button = e
            break

    # Targets: flat compact candidates away from the button, deduplicated.
    flat = []
    for e in sorted(target_c, key=by_score):
        g = e.get("geo")
        if g is None:
            continue
        if g["xext"] > 0.15 or g["yext"] > 0.15 or g["xext"] < 0.01 or g["yext"] < 0.01:
            continue
        if g["z95"] - g["z05"] > 0.02:
            continue
        if button is not None and dxy(g["med"], button["geo"]["med"]) < 0.04:
            continue
        dup = False
        for f in flat:
            if dxy(f["geo"]["med"], g["med"]) < 0.04:
                dup = True
        if not dup:
            flat.append(e)
    targets = []
    for e in flat:
        if len(targets) == 0:
            targets.append(e)
        elif len(targets) == 1 and dxy(targets[0]["geo"]["med"], e["geo"]["med"]) > 0.08:
            targets.append(e)
    if len(targets) < 2:
        raise Exception("found fewer than two targets: " + str(len(targets)))
    table_z = 0.5 * (targets[0]["geo"]["med"][2] + targets[1]["geo"]["med"][2])
    report["table_z_est"] = round(table_z, 4)

    # Left/right: front camera faces the robot, so robot-left is image-right.
    t_a = targets[0]
    t_b = targets[1]
    if t_a.get("u") is not None and t_b.get("u") is not None:
        if t_a["u"] > t_b["u"]:
            left_t = t_a
            right_t = t_b
        else:
            left_t = t_b
            right_t = t_a
    else:
        if t_a["geo"]["med"][1] > t_b["geo"]["med"][1]:
            left_t = t_a
            right_t = t_b
        else:
            left_t = t_b
            right_t = t_a
    report["left_target"] = public(left_t)
    report["right_target"] = public(right_t)
    report["lr_frame_consistent"] = bool(left_t["geo"]["med"][1] > right_t["geo"]["med"][1])
    if button is not None:
        report["button"] = public(button)

    # Cube: named-colour candidate, compact, sitting on the table, not a target/button.
    cube = None
    for e in cube_c:
        g = e.get("geo")
        if g is None:
            continue
        height = g["z95"] - table_z
        if height < 0.01 or height > 0.10:
            continue
        if g["xext"] > 0.09 or g["yext"] > 0.09:
            continue
        bad = False
        for t in targets:
            if dxy(g["med"], t["geo"]["med"]) < 0.025:
                bad = True
        if button is not None and dxy(g["med"], button["geo"]["med"]) < 0.025:
            bad = True
        if bad:
            continue
        cube = e
        break
    if cube is None:
        raise Exception("cube not found")
    report["cube"] = public(cube)
    cube_pts = cube["pts"]
    zs = [p[2] for p in cube_pts]
    cube_top = quantile(zs, 0.95)
    face = top_face(cube_pts, cube_top)
    cube_xy = (sum(p[0] for p in face) / len(face), sum(p[1] for p in face) / len(face))
    cube_h = cube_top - table_z
    cube_yaw = min_area_yaw(face)
    report["cube_est_front"] = {"xy": [round(v, 4) for v in cube_xy], "top": round(cube_top, 4),
                                "h": round(cube_h, 4), "yaw": cube_yaw}

    # Obstacles for transit: other cubes (generic prompt) and the button.
    obstacles = []
    others = candidates("front", ["cube"], limit=8)
    for e in others:
        g = e.get("geo")
        if g is None:
            continue
        if dxy(g["med"], cube["geo"]["med"]) < 0.03:
            continue
        if g["xext"] > 0.12 or g["yext"] > 0.12:
            continue
        if g["z95"] > table_z + 0.12:
            continue
        near_target = False
        for t in targets:
            if dxy(g["med"], t["geo"]["med"]) < 0.04:
                near_target = True
        if near_target:
            continue
        obstacles.append((g["med"][0], g["med"][1], g["z95"]))
    if button is not None:
        obstacles.append((button["geo"]["med"][0], button["geo"]["med"][1], button["geo"]["z95"]))
    report["obstacles"] = [[round(v, 3) for v in o] for o in obstacles]

    # ------------------------------------------------------------ grasp
    def grasp_quat(yaw):
        if yaw is None:
            return home_quat
        delta = wrap90(yaw - closing_yaw)
        return qmul(yaw_quat(delta), home_quat)

    quat = grasp_quat(cube_yaw)
    open_res = open_gripper()
    record("open_gripper", open_res)
    pre_z = cube_top + 0.10
    go((cube_xy[0], cube_xy[1], pre_z), quat, "pregrasp", tol=0.01, max_steps=200)

    # Wrist refinement from directly above.
    try:
        wrist_c = candidates("wrist", cube_prompts[:1] if color is not None else ["cube"], limit=3)
        report["wrist_cube_candidates"] = [public(e) for e in wrist_c]
        best = None
        for e in wrist_c:
            g = e.get("geo")
            if g is None:
                continue
            if dxy(g["med"], cube_xy) < 0.04 and g["xext"] < 0.09 and g["yext"] < 0.09:
                best = e
                break
        if best is not None:
            wpts = best["pts"]
            wtop = quantile([p[2] for p in wpts], 0.95)
            wface = top_face(wpts, wtop)
            wxy = (sum(p[0] for p in wface) / len(wface), sum(p[1] for p in wface) / len(wface))
            wyaw = min_area_yaw(wface)
            report["cube_est_wrist"] = {"xy": [round(v, 4) for v in wxy], "top": round(wtop, 4), "yaw": wyaw}
            cube_xy = wxy
            if abs(wtop - cube_top) < 0.02:
                cube_top = wtop
                cube_h = cube_top - table_z
            if wyaw is not None:
                quat = grasp_quat(wyaw)
            go((cube_xy[0], cube_xy[1], pre_z), quat, "pregrasp_refined", tol=0.01, max_steps=200)
    except Exception as exc:
        if "EPISODE_TERMINAL" in str(exc):
            raise exc
        report["wrist_refine_error"] = str(exc)[:200]

    grasp_z = max(table_z + 0.012, cube_top - 0.5 * cube_h)
    go((cube_xy[0], cube_xy[1], grasp_z), quat, "grasp_descend", tol=0.008, max_steps=200)
    close_res = close_gripper()
    record("close_gripper", close_res)
    report["after_close"] = ee_now()
    grip_offset = grasp_z - table_z  # EE height above cube bottom

    lift_z = table_z + 0.12 + grip_offset
    go((cube_xy[0], cube_xy[1], lift_z), quat, "lift", tol=0.015, max_steps=200)
    report["after_lift"] = ee_now()

    # ------------------------------------------------------------ swings
    gap = 0.025
    order = []
    for rep_index in range(repeat):
        order.append(first_side)
        order.append(second_side)
    report["visit_order"] = order
    target_top = 0.5 * (left_t["geo"]["z90"] + right_t["geo"]["z90"])
    side_xy = {"left": (left_t["geo"]["med"][0], left_t["geo"]["med"][1]),
               "right": (right_t["geo"]["med"][0], right_t["geo"]["med"][1])}
    button_xy0 = None
    if button is not None:
        button_xy0 = (button["geo"]["med"][0], button["geo"]["med"][1])
    target_prompts_live = ["target", "white round target", "red and white target"]

    def target_obs(camera):
        # Flat ring-shaped candidates; centre from the low (surface) points only, so a
        # hovering cube merged into the mask does not bias it.
        out = []
        for e in candidates(camera, target_prompts_live, limit=5):
            pts = e["pts"]
            low = [p for p in pts if p[2] < target_top + 0.012 and p[2] > target_top - 0.03]
            if len(low) < 12:
                continue
            xs = [p[0] for p in low]
            ys = [p[1] for p in low]
            x05 = quantile(xs, 0.05)
            x95 = quantile(xs, 0.95)
            y05 = quantile(ys, 0.05)
            y95 = quantile(ys, 0.95)
            if x95 - x05 > 0.10 or y95 - y05 > 0.10 or x95 - x05 < 0.015 or y95 - y05 < 0.015:
                continue
            xy = (0.5 * (x05 + x95), 0.5 * (y05 + y95))
            if button_xy0 is not None and dxy(xy, button_xy0) < 0.04:
                continue
            red = None
            coloured = [p for p in low if p[3] >= 0.0]
            if len(coloured) >= 10:
                hits = 0
                for p in coloured:
                    if p[3] > 0.45 and p[3] > 1.8 * p[4] and p[3] > 1.8 * p[5]:
                        hits = hits + 1
                red = hits / float(len(coloured))
                # colour gate (wrist view only, where a green light arc was mis-segmented as a target): a target ring is neutral (white/grey) or highlight-red; coloured light
                # patches on the table (green arcs, wood) are rejected.
                like = 0
                for p in coloured:
                    hi = max(p[3], p[4], p[5])
                    lo = min(p[3], p[4], p[5])
                    if hi - lo < 0.2 or (p[3] > 0.45 and p[3] > 1.8 * p[4] and p[3] > 1.8 * p[5]):
                        like = like + 1
                if camera == "wrist" and like < 0.3 * len(coloured):
                    continue
            dup = False
            for o in out:
                if dxy(o["xy"], xy) < 0.025:
                    dup = True
                    if e["score"] > o["score"]:
                        o["xy"] = xy
                        o["score"] = e["score"]
                        o["red"] = red
            if not dup:
                out.append({"xy": xy, "score": e["score"], "red": red, "n": len(low)})
        return out

    def locate(side, cameras):
        pred = side_xy[side]
        other = side_xy["left" if side == "right" else "right"]
        for camera in cameras:
            best = None
            best_d = None
            for o in target_obs(camera):
                if dxy(o["xy"], other) < 0.05 and dxy(pred, other) >= 0.05:
                    continue
                d = dxy(o["xy"], pred)
                if d > 0.25:
                    continue
                if best_d is None or d < best_d:
                    best = o
                    best_d = d
            if best is not None:
                best["camera"] = camera
                best["shift"] = round(best_d, 4)
                side_xy[side] = best["xy"]
                return best
        return None

    def path_clear(a, b, z_cube_bottom):
        for o in obstacles:
            ax = b[0] - a[0]
            ay = b[1] - a[1]
            length2 = ax * ax + ay * ay
            if length2 < 1e-9:
                continue
            tpar = ((o[0] - a[0]) * ax + (o[1] - a[1]) * ay) / length2
            if tpar < 0.0 or tpar > 1.0:
                continue
            px = a[0] + tpar * ax
            py = a[1] + tpar * ay
            if math.sqrt((o[0] - px) ** 2 + (o[1] - py) ** 2) < 0.07 and o[2] + 0.02 > z_cube_bottom:
                return False
        return True

    visits = []
    prev_xy = None
    hz = target_top + gap + grip_offset
    for visit_index, side in enumerate(order):
        info = {"i": visit_index, "side": side, "tries": []}
        seen = locate(side, ("front", "wrist"))
        info["pre"] = None if seen is None else {"xy": [round(v, 4) for v in seen["xy"]], "shift": seen["shift"], "cam": seen["camera"]}
        txy = side_xy[side]
        if prev_xy is None:
            go((txy[0], txy[1], lift_z), quat, "to_first_high", tol=0.015, max_steps=200)
            seen = locate(side, ("front", "wrist"))
            txy = side_xy[side]
        elif not path_clear(prev_xy, txy, hz - grip_offset):
            top = max(o[2] for o in obstacles)
            mid_z = max(hz, top + 0.04 + grip_offset)
            go((0.5 * (prev_xy[0] + txy[0]), 0.5 * (prev_xy[1] + txy[1]), mid_z), quat,
               "transit_mid_" + str(visit_index), tol=0.02, max_steps=200)
            seen = locate(side, ("front", "wrist"))
            txy = side_xy[side]
        go((txy[0], txy[1], hz), quat, "visit_" + str(visit_index) + "_" + side, tol=0.01, max_steps=200)
        hold(3, 0.0, "dwell_" + str(visit_index))
        misses = 0
        corrections = 0
        lifts = 0
        for attempt in range(12):
            seen = locate(side, ("front", "wrist"))
            ee_xy, gnow = ee_now()
            entry = {"a": attempt, "ee": ee_xy, "g": gnow, "ticks": STATE["ticks"]}
            if seen is not None:
                entry["t"] = [round(v, 4) for v in seen["xy"]]
                entry["red"] = None if seen["red"] is None else round(seen["red"], 3)
                entry["cam"] = seen["camera"]
            info["tries"].append(entry)
            if seen is None:
                # target may be mid-relocation or occluded: wait and look again
                misses = misses + 1
                if misses >= 5:
                    break
                if ee_xy[2] < hz + 0.02 and lifts >= 2:
                    # already re-aimed from an unoccluded view twice: trust the latest estimate, dwell, go on
                    hold(8, 0.0, "blind_dwell_" + str(visit_index) + "_" + str(attempt))
                    info["blind_accept"] = True
                    break
                if misses <= 2:
                    # the arm/cube occludes the target in the front view: rise so it becomes visible
                    lifts = lifts + 1
                    go((ee_xy[0], ee_xy[1], hz + 0.06 * misses + 0.04), quat, "unocclude_" + str(visit_index) + "_" + str(attempt),
                       tol=0.02, max_steps=120)
                    hold(2, 0.0, "unocclude_wait_" + str(visit_index) + "_" + str(attempt))
                else:
                    hold(4, 0.0, "wait_" + str(visit_index) + "_" + str(attempt))
                continue
            misses = 0
            off = dxy(ee_xy, seen["xy"])
            entry["off"] = round(off, 4)
            lit = seen["red"] is not None and seen["red"] > 0.25
            if lit and off < 0.03:
                break
            if seen["red"] is None and off < 0.012 and ee_xy[2] < hz + 0.02:
                break
            if attempt == 11:
                break
            txy = seen["xy"]
            lower = 0.0
            if corrections >= 2:
                lower = 0.006
            if off < 0.012:
                # centred but not highlighted yet: settle a little lower and longer
                lower = 0.008
            go((txy[0], txy[1], hz - lower), quat, "correct_" + str(visit_index) + "_" + str(attempt),
               tol=0.008, max_steps=120)
            corrections = corrections + 1
            hold(3, 0.0, "dwell_" + str(visit_index) + "_" + str(attempt))
        info["lit"] = len(info["tries"]) > 0 and info["tries"][-1].get("red") is not None and info["tries"][-1].get("red") > 0.25
        visits.append(info)
        report["visits"] = visits
        prev_xy = side_xy[side]

    # ------------------------------------------------------------ put down where the cube is
    ee_xy, gnow = ee_now()
    go((ee_xy[0], ee_xy[1], target_top + 0.006 + grip_offset), quat, "put_down", tol=0.01, max_steps=150)
    record("release", open_gripper())
    go((ee_xy[0], ee_xy[1], target_top + 0.10 + grip_offset), quat, "retreat", tol=0.02, max_steps=150)

    # ------------------------------------------------------------ press the button (re-localised)
    if button is None:
        raise Exception("button not found")
    bpts = button["pts"]
    try:
        best = None
        best_d = None
        for e in candidates("front", ["button", "push button"], limit=5):
            g = e.get("geo")
            if g is None or g["xext"] > 0.15 or g["yext"] > 0.15:
                continue
            if g["z95"] < target_top + 0.012:
                continue
            d = dxy(g["med"], button_xy0)
            if d < 0.15 and (best_d is None or d < best_d):
                best = e
                best_d = d
        if best is not None:
            bpts = best["pts"]
            report["button_relocalised_shift"] = round(best_d, 4)
    except Exception as exc:
        if "EPISODE_TERMINAL" in str(exc):
            raise exc
        report["button_relocalise_error"] = str(exc)[:200]
    btop = quantile([p[2] for p in bpts], 0.95)
    bface = top_face(bpts, btop, band=0.01)
    bxy = (sum(p[0] for p in bface) / len(bface), sum(p[1] for p in bface) / len(bface))
    report["button_est"] = {"xy": [round(v, 4) for v in bxy], "top": round(btop, 4)}
    record("close_for_press", close_gripper())
    go((bxy[0], bxy[1], btop + 0.08), home_quat, "above_button", tol=0.015, max_steps=200)
    go((bxy[0], bxy[1], btop - 0.015), home_quat, "press", tol=0.01, max_steps=120)
    hold(5, 0.0, "press_hold")
    go((bxy[0], bxy[1], btop + 0.06), home_quat, "press_release", tol=0.02, max_steps=120)
    hold(10, 0.0, "after_press")
    go((bxy[0], bxy[1], btop - 0.03), home_quat, "press_deeper", tol=0.01, max_steps=120)
    hold(5, 0.0, "press2_hold")
    go((bxy[0], bxy[1], btop + 0.06), home_quat, "press2_release", tol=0.02, max_steps=120)
    hold(10, 0.0, "after_press2")


try:
    run_program()
    report["program_finished"] = True
except Exception as exc:
    report["stopped"] = str(exc)[:300]
report["calls"] = calls
report["ticks_total"] = STATE["ticks"]
report["terminal_seen"] = STATE["terminal"]
result = report
