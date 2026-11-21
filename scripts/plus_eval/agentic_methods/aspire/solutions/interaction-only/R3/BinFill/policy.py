# v00_initial — BinFill (robomme_plus task 0), interaction-only, combined scene probe + task attempt.
#
# Start 1 of 3. Because the budget is three simulator starts in total, this program records
# what a scene snapshot would (language, metadata, cameras, conditioning-video summary, SAM3
# prompt probes) AND attempts the task, so the one start yields both evidence and a result.
#
# Strategy (hypothesis from language only; verified by this run):
#   parse "<count> <colour> cube(s)" requests from the goal -> localize bin (front) ->
#   for each requested cube: segment "<colour> cube" on front, keep candidates whose measured
#   colour matches and which lie outside the bin, refine on the wrist camera at pre-grasp,
#   top-down grasp with yaw from a min-area rectangle of the top-face points, lift, transit
#   above the bin (rim + 0.30), release at rim + 0.10, settle -> if the goal mentions a
#   button, press it top-down with the gripper closed.
# Every object position comes from public RGB-D perception at runtime.

import math

report = {"version": "v00_initial", "phases": [], "probes": [], "cubes": [], "errors": []}
result = report

FRONT = "front"
WRIST = "wrist"

COLOR_WORDS = ("red", "green", "blue", "yellow", "purple", "orange", "pink", "white",
               "black", "gray", "grey", "brown", "cyan")
NUMBER_WORDS = {"one": 1, "a": 1, "an": 1, "single": 1, "two": 2, "three": 3, "four": 4,
                "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "once": 1, "twice": 2, "thrice": 3}


# ---------------------------------------------------------------- generic helpers
def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def now_ticks():
    st = get_robot_state()
    return round(float(st.timestamp_s) * CFG["hz"])


def ended(res):
    flag = False
    for name in ("terminated", "truncated"):
        try:
            if name == "terminated" and bool(res.terminated):
                flag = True
            if name == "truncated" and bool(res.truncated):
                flag = True
        except Exception:
            pass
    return flag


def status_of(res):
    out = {}
    try:
        out["ok"] = bool(res.ok)
    except Exception:
        pass
    try:
        out["status"] = str(res.status)
    except Exception:
        pass
    try:
        out["steps"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        out["terminated"] = bool(res.terminated)
        out["truncated"] = bool(res.truncated)
    except Exception:
        pass
    try:
        if res.error is not None:
            out["error"] = str(res.error)[:300]
    except Exception:
        pass
    return out


END_MARK = "EPISODE-ENDED:"


def act(label, res):
    info = status_of(res)
    info["label"] = label
    report["phases"].append(info)
    if ended(res):
        report["episode_ended_after"] = label
        raise ValueError(END_MARK + label)
    try:
        info["tick"] = now_ticks()
    except Exception:
        pass
    return res


def sanitize(value, depth=0):
    if value is None or value is True or value is False:
        return value
    try:
        if str(value) == value:
            return value
    except Exception:
        pass
    try:
        number = float(value)
        text = str(value)
        if text.lstrip("-").isdigit():
            return int(value)
        return number
    except Exception:
        pass
    if depth > 4:
        return str(value)[:200]
    try:
        items = value.items()
        out = {}
        for k, v in items:
            out[str(k)] = sanitize(v, depth + 1)
        return out
    except Exception:
        pass
    try:
        if len(value) <= 64:
            return [sanitize(v, depth + 1) for v in value]
    except Exception:
        pass
    return str(value)[:200]


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def quat_axes(q):
    w, x, y, z = q
    xaxis = (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
    yaxis = (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x))
    zaxis = (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
    return xaxis, yaxis, zaxis


def wrap90(angle_deg):
    a = angle_deg
    while a > 45.0:
        a -= 90.0
    while a <= -45.0:
        a += 90.0
    return a


def hue_sat_val(rgb):
    r, g, b = (float(c) / 255.0 for c in rgb)
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn
    if d < 1e-6:
        h = 0.0
    elif mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * (((b - r) / d) + 2.0)
    else:
        h = 60.0 * (((r - g) / d) + 4.0)
    s = 0.0 if mx < 1e-6 else d / mx
    return h, s, mx


def color_name(rgb):
    h, s, v = hue_sat_val(rgb)
    if s < 0.25:
        if v > 0.75:
            return "white"
        if v < 0.2:
            return "black"
        return "gray"
    if h < 20 or h >= 335:
        return "red"
    if h < 40:
        return "orange"
    if h < 70:
        return "yellow"
    if h < 170:
        return "green"
    if h < 200:
        return "cyan"
    if h < 260:
        return "blue"
    if h < 300:
        return "purple"
    return "pink"


def same_color(measured, wanted):
    if wanted is None:
        return True
    if wanted in ("gray", "grey"):
        return measured == "gray"
    if wanted == "cyan":
        return measured in ("cyan", "blue", "green")
    if wanted == "pink":
        return measured in ("pink", "red", "purple")
    if wanted == "purple":
        return measured in ("purple", "pink", "blue")
    if wanted == "orange":
        return measured in ("orange", "red", "yellow")
    return measured == wanted


# ---------------------------------------------------------------- perception helpers
def cloud_of(candidate, camera):
    pc = None
    try:
        pc = mask_to_point_cloud(candidate, camera, target_frame=CFG["base"])
    except Exception:
        pc = None
    if pc is None:
        try:
            pc = mask_to_point_cloud(candidate.mask, camera, target_frame=CFG["base"])
        except Exception:
            pc = None
    return pc


def cloud_points(pc):
    if pc is None:
        return []
    pts = []
    try:
        for p in pc.points:
            pts.append((float(p[0]), float(p[1]), float(p[2])))
    except Exception:
        return []
    return pts


def cloud_median_rgb(pc, candidate, camera):
    cols = None
    try:
        cols = pc.colors
    except Exception:
        cols = None
    rs, gs, bs = [], [], []
    if cols is not None:
        try:
            for c in cols:
                rs.append(float(c[0]))
                gs.append(float(c[1]))
                bs.append(float(c[2]))
        except Exception:
            rs, gs, bs = [], [], []
    if len(rs) > 0:
        scale = 255.0 if max(rs + gs + bs) <= 1.0 else 1.0
        return (quantile(rs, 0.5) * scale, quantile(gs, 0.5) * scale, quantile(bs, 0.5) * scale)
    # fallback: sample the image inside the candidate box where the mask is set
    try:
        rgb = get_observation(camera_names=[camera]).cameras[camera].rgb
        mask = candidate.mask
        x1, y1, x2, y2 = (int(v) for v in candidate.box_xyxy)
        for yy in range(max(0, y1), min(int(rgb.shape[0]), y2 + 1)):
            for xx in range(max(0, x1), min(int(rgb.shape[1]), x2 + 1)):
                if bool(mask[yy][xx]):
                    px = rgb[yy][xx]
                    rs.append(float(px[0]))
                    gs.append(float(px[1]))
                    bs.append(float(px[2]))
    except Exception:
        pass
    if len(rs) == 0:
        return None
    return (quantile(rs, 0.5), quantile(gs, 0.5), quantile(bs, 0.5))


def summarize_cloud(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    return {
        "n": len(pts),
        "x": [round(quantile(xs, 0.05), 4), round(quantile(xs, 0.5), 4), round(quantile(xs, 0.95), 4)],
        "y": [round(quantile(ys, 0.05), 4), round(quantile(ys, 0.5), 4), round(quantile(ys, 0.95), 4)],
        "z": [round(quantile(zs, 0.05), 4), round(quantile(zs, 0.5), 4), round(quantile(zs, 0.95), 4)],
    }


def probe(camera, prompt):
    entry = {"camera": camera, "prompt": prompt, "ok": False, "n": 0, "scores": [], "boxes": []}
    seg = segment_text(camera, prompt)
    try:
        entry["ok"] = bool(seg.ok)
        if seg.ok:
            entry["n"] = len(seg.segmentations)
            for cand in seg.segmentations[:6]:
                entry["scores"].append(round(float(cand.score), 3))
                if cand.box_xyxy is not None:
                    entry["boxes"].append([round(float(v), 1) for v in cand.box_xyxy])
    except Exception as exc:
        entry["error"] = str(exc)[:200]
    report["probes"].append(entry)
    return seg


def min_area_yaw(pts):
    """Yaw (deg) of the minimum-area XY bounding rectangle of the top-face points."""
    if len(pts) < 6:
        return None, None
    best = None
    best_angle = None
    angle = 0.0
    while angle < 90.0:
        c = math.cos(math.radians(angle))
        s = math.sin(math.radians(angle))
        us = [p[0] * c + p[1] * s for p in pts]
        vs = [-p[0] * s + p[1] * c for p in pts]
        area = (quantile(us, 0.95) - quantile(us, 0.05)) * (quantile(vs, 0.95) - quantile(vs, 0.05))
        if best is None or area < best:
            best = area
            best_angle = angle
        angle += 3.0
    return best_angle, best


def cube_estimate(pts):
    zs = [p[2] for p in pts]
    z_top = quantile(zs, 0.95)
    z_bot = quantile(zs, 0.05)
    top = [p for p in pts if p[2] >= z_top - 0.012]
    if len(top) < 6:
        top = pts
    cx = 0.5 * (quantile([p[0] for p in top], 0.05) + quantile([p[0] for p in top], 0.95))
    cy = 0.5 * (quantile([p[1] for p in top], 0.05) + quantile([p[1] for p in top], 0.95))
    yaw, area = min_area_yaw(top)
    span_x = quantile([p[0] for p in pts], 0.95) - quantile([p[0] for p in pts], 0.05)
    span_y = quantile([p[1] for p in pts], 0.95) - quantile([p[1] for p in pts], 0.05)
    return {"cx": cx, "cy": cy, "z_top": z_top, "z_bot": z_bot, "yaw": yaw,
            "span_xy": max(span_x, span_y), "n": len(pts), "n_top": len(top)}


def inside_box(x, y, box, margin):
    if box is None:
        return False
    return (box["x0"] - margin <= x <= box["x1"] + margin) and (box["y0"] - margin <= y <= box["y1"] + margin)


def by_score(entry):
    return -entry["score"]


def by_near(entry):
    return entry["dist_near"]


def find_cubes(camera, color, bin_box, near=None, max_n=8):
    """All cube candidates of the wanted colour outside the bin, best first."""
    prompts = []
    if color is not None:
        prompts.append(color + " cube")
    prompts.append("cube")
    found = []
    for prompt in prompts:
        seg = probe(camera, prompt)
        if not seg.ok:
            continue
        for cand in seg.segmentations[:max_n]:
            pc = cloud_of(cand, camera)
            pts = cloud_points(pc)
            if len(pts) < 8:
                continue
            est = cube_estimate(pts)
            med = cloud_median_rgb(pc, cand, camera)
            cname = color_name(med) if med is not None else None
            est["score"] = round(float(cand.score), 3)
            est["prompt"] = prompt
            est["color"] = cname
            est["rgb"] = None if med is None else [round(v) for v in med]
            if est["span_xy"] > 0.09:
                est["reject"] = "too large"
            elif inside_box(est["cx"], est["cy"], bin_box, 0.01):
                est["reject"] = "inside bin"
            elif cname is not None and not same_color(cname, color):
                est["reject"] = "colour " + str(cname)
            if near is not None:
                est["dist_near"] = math.sqrt((est["cx"] - near[0]) ** 2 + (est["cy"] - near[1]) ** 2)
            found.append(est)
        if len([e for e in found if "reject" not in e]) > 0:
            break
    good = [e for e in found if "reject" not in e]
    # de-duplicate (two masks of the same cube)
    uniq = []
    for e in sorted(good, key=by_score):
        dup = False
        for u in uniq:
            if math.sqrt((u["cx"] - e["cx"]) ** 2 + (u["cy"] - e["cy"]) ** 2) < 0.025:
                dup = True
        if not dup:
            uniq.append(e)
    if near is not None:
        uniq = sorted(uniq, key=by_near)
    return uniq, found


def locate_bin():
    for prompt in ("bin", "box", "container", "basket", "tray"):
        seg = probe(FRONT, prompt)
        if not seg.ok or len(seg.segmentations) == 0:
            continue
        best = None
        for cand in seg.segmentations[:4]:
            pts = cloud_points(cloud_of(cand, FRONT))
            if len(pts) < 30:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [p[2] for p in pts]
            box = {"x0": quantile(xs, 0.03), "x1": quantile(xs, 0.97),
                   "y0": quantile(ys, 0.03), "y1": quantile(ys, 0.97),
                   "rim_z": quantile(zs, 0.95), "floor_z": quantile(zs, 0.05),
                   "score": round(float(cand.score), 3), "prompt": prompt, "n": len(pts)}
            box["cx"] = 0.5 * (box["x0"] + box["x1"])
            box["cy"] = 0.5 * (box["y0"] + box["y1"])
            box["span"] = max(box["x1"] - box["x0"], box["y1"] - box["y0"])
            if box["span"] < 0.06:
                continue  # too small to be a bin
            if best is None or box["score"] > best["score"]:
                best = box
        if best is not None:
            return best
    return None


def locate_button():
    for prompt in ("button", "round button", "push button"):
        seg = probe(FRONT, prompt)
        if not seg.ok or len(seg.segmentations) == 0:
            continue
        for cand in seg.segmentations[:3]:
            pts = cloud_points(cloud_of(cand, FRONT))
            if len(pts) < 8:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [p[2] for p in pts]
            return {"cx": quantile(xs, 0.5), "cy": quantile(ys, 0.5), "z_top": quantile(zs, 0.95),
                    "z_bot": quantile(zs, 0.05), "score": round(float(cand.score), 3),
                    "prompt": prompt, "n": len(pts),
                    "span": max(quantile(xs, 0.95) - quantile(xs, 0.05), quantile(ys, 0.95) - quantile(ys, 0.05))}
    return None


# ---------------------------------------------------------------- language
def parse_goal(text):
    words = []
    for raw in text.lower().replace(",", " ").replace(".", " ").replace(";", " ").split():
        words.append(raw.strip())
    requests = []
    for i, w in enumerate(words):
        if w in COLOR_WORDS:
            count = None
            for back in (1, 2):
                if i - back >= 0:
                    tok = words[i - back]
                    if tok in NUMBER_WORDS:
                        count = NUMBER_WORDS[tok]
                        break
                    if tok.isdigit():
                        count = int(tok)
                        break
                    if tok == "all":
                        count = -1
                        break
            nxt = words[i + 1] if i + 1 < len(words) else ""
            if nxt.startswith("cube") or nxt.startswith("block") or count is not None:
                requests.append({"color": "gray" if w == "grey" else w,
                                 "count": 1 if count is None else count})
    if len(requests) == 0:
        for i, w in enumerate(words):
            if w.startswith("cube") or w.startswith("block"):
                count = 1
                if i - 1 >= 0 and words[i - 1] in NUMBER_WORDS:
                    count = NUMBER_WORDS[words[i - 1]]
                elif i - 1 >= 0 and words[i - 1].isdigit():
                    count = int(words[i - 1])
                requests.append({"color": None, "count": count})
                break
    return {"requests": requests, "button": "button" in words, "words": words}


# ---------------------------------------------------------------- motion helpers
def hold(ticks, gripper):
    st = get_robot_state()
    last = None
    for tick in range(ticks):
        last = step(RobotAction(arms={"primary": ArmCommand(
            "joint_position", st.joint_positions["primary"],
            gripper_position=gripper, embodiment=st.embodiment)}))
        if ended(last):
            report["episode_ended_after"] = "hold"
            raise ValueError(END_MARK + "hold")
    return last


def top_down_quat(cube_yaw_deg):
    """Downward quaternion whose finger-closing axis is aligned with a cube face."""
    base_q = CFG["home_q"] if CFG["home_down"] else (0.0, 1.0, 0.0, 0.0)
    if cube_yaw_deg is None:
        return base_q
    xa, ya, za = quat_axes(base_q)
    close_yaw = math.degrees(math.atan2(ya[1], ya[0]))
    delta = wrap90(cube_yaw_deg - close_yaw)
    half = math.radians(delta) / 2.0
    return quat_mul((math.cos(half), 0.0, 0.0, math.sin(half)), base_q)


def goto(label, xyz, quat, tol=0.015, max_steps=200, strategy=None):
    pose = Pose((float(xyz[0]), float(xyz[1]), float(xyz[2])), quat, CFG["base"])
    if strategy is None:
        res = move_to_pose(pose, tolerance=tol, max_steps=max_steps)
    else:
        res = move_to_pose(pose, tolerance=tol, max_steps=max_steps, strategy=strategy)
    act(label, res)
    return res


def gripper_width():
    return float(get_robot_state().gripper_positions["primary"])


# ---------------------------------------------------------------- main
def main():
    ctx = get_task_context()
    st = get_robot_state()
    CFG["base"] = st.base_frame
    report["language"] = ctx.language
    report["task_name"] = ctx.task_name
    try:
        report["context_metadata"] = sanitize(ctx.metadata)
    except Exception:
        pass
    meta = {}
    try:
        meta = robomme_plus.get_task_metadata()
        report["task_metadata"] = sanitize(meta)
    except Exception as exc:
        report["errors"].append("task_metadata: " + str(exc)[:200])
    try:
        report["controller_metadata"] = sanitize(robomme_plus.get_controller_metadata())
    except Exception as exc:
        report["errors"].append("controller_metadata: " + str(exc)[:200])
    for key in ("control_hz", "control_frequency", "control_freq", "frequency_hz"):
        try:
            CFG["hz"] = float(meta[key])
            break
        except Exception:
            pass
    report["control_hz"] = CFG["hz"]
    try:
        report["max_steps"] = int(meta["max_steps"])
    except Exception:
        pass

    home = st.end_effector_poses["primary"]
    CFG["home_q"] = tuple(float(v) for v in home.quaternion_wxyz)
    xa, ya, za = quat_axes(CFG["home_q"])
    CFG["home_down"] = za[2] < -0.9
    report["home"] = {"pos": [round(float(v), 4) for v in home.position], "quat": [round(v, 4) for v in CFG["home_q"]],
                      "z_axis": [round(v, 3) for v in za], "y_axis": [round(v, 3) for v in ya], "down": CFG["home_down"],
                      "joints": [round(float(v), 4) for v in st.joint_positions["primary"]],
                      "gripper": float(st.gripper_positions["primary"]), "t": float(st.timestamp_s)}

    obs = get_observation()
    cams = {}
    for name in obs.cameras:
        cam = obs.cameras[name]
        info = {"shape": [int(v) for v in cam.rgb.shape]}
        try:
            info["pose"] = [round(float(v), 4) for v in cam.camera_pose.position]
            info["quat"] = [round(float(v), 4) for v in cam.camera_pose.quaternion_wxyz]
        except Exception:
            pass
        cams[name] = info
    report["cameras"] = cams

    # conditioning video: summary only (count and timestamps)
    try:
        video = robomme_plus.get_conditioning_video()
        report["video_frames"] = len(video)
        if len(video) > 0:
            report["video_t"] = [float(video[0].timestamp_s), float(video[len(video) - 1].timestamp_s)]
    except Exception as exc:
        report["errors"].append("video: " + str(exc)[:200])

    goal = parse_goal(ctx.language)
    report["goal"] = {"requests": goal["requests"], "button": goal["button"]}

    # scene probes (no control time)
    for prompt in ("cube", "red cube", "green cube", "blue cube"):
        probe(FRONT, prompt)
    bin_box = locate_bin()
    report["bin"] = bin_box
    button = locate_button()
    report["button"] = button
    report["t_first_control"] = float(get_robot_state().timestamp_s)

    if bin_box is None:
        raise ValueError("bin not found")

    act("open_initial", open_gripper())
    transit_z = bin_box["rim_z"] + 0.30
    release_z = bin_box["rim_z"] + 0.10

    placed_total = 0
    for req in goal["requests"]:
        want = req["count"]
        color = req["color"]
        placed = 0
        attempts = 0
        while (want < 0 or placed < want) and attempts < (want + 3 if want > 0 else 8):
            attempts += 1
            entry = {"color": color, "attempt": attempts, "tick_start": now_ticks()}
            report["cubes"].append(entry)
            cands, raw = find_cubes(FRONT, color, bin_box)
            entry["front_candidates"] = [
                {k: (round(v, 4) if k in ("cx", "cy", "z_top", "z_bot", "span_xy") and v is not None else v)
                 for k, v in e.items()} for e in raw[:8]]
            if len(cands) == 0:
                entry["result"] = "no candidate"
                if want < 0:
                    break
                continue
            tgt = cands[0]
            q = top_down_quat(tgt["yaw"])
            pre_z = max(tgt["z_top"] + 0.12, bin_box["rim_z"] + 0.12)
            goto("pregrasp", (tgt["cx"], tgt["cy"], pre_z), q, tol=0.02, max_steps=250)
            # refine on the wrist camera from directly above
            wc, wraw = find_cubes(WRIST, color, bin_box, near=(tgt["cx"], tgt["cy"]))
            entry["wrist_candidates"] = [
                {k: (round(v, 4) if k in ("cx", "cy", "z_top", "z_bot", "span_xy", "dist_near") and v is not None else v)
                 for k, v in e.items()} for e in wraw[:6]]
            if len(wc) > 0 and wc[0]["dist_near"] < 0.04 and wc[0]["n"] >= 15:
                tgt = wc[0]
                entry["refined"] = True
                q = top_down_quat(tgt["yaw"])
                goto("pregrasp_refined", (tgt["cx"], tgt["cy"], pre_z), q, tol=0.02, max_steps=150)
            height = tgt["z_top"] - tgt["z_bot"]
            grasp_z = min(tgt["z_top"] - 0.012, tgt["z_bot"] + 0.5 * max(height, 0.0))
            if height < 0.01:
                grasp_z = tgt["z_top"] - 0.015
            grasp_z = max(grasp_z, tgt["z_top"] - 0.025)
            entry["target"] = {"x": round(tgt["cx"], 4), "y": round(tgt["cy"], 4), "z_top": round(tgt["z_top"], 4),
                               "z_bot": round(tgt["z_bot"], 4), "grasp_z": round(grasp_z, 4), "yaw": tgt["yaw"]}
            goto("grasp", (tgt["cx"], tgt["cy"], grasp_z), q, tol=0.01, max_steps=150)
            act("close", close_gripper())
            entry["width_closed"] = round(gripper_width(), 3)
            goto("lift", (tgt["cx"], tgt["cy"], transit_z), q, tol=0.02, max_steps=250)
            entry["width_lifted"] = round(gripper_width(), 3)
            if entry["width_lifted"] < 0.05:
                entry["result"] = "missed (gripper closed on nothing)"
                act("open_after_miss", open_gripper())
                continue
            goto("above_bin", (bin_box["cx"], bin_box["cy"], transit_z), q, tol=0.02, max_steps=300)
            goto("release_pose", (bin_box["cx"], bin_box["cy"], release_z), q, tol=0.02, max_steps=200)
            entry["width_at_release"] = round(gripper_width(), 3)
            held = entry["width_at_release"] >= 0.05
            act("release", open_gripper())
            hold(8, 1.0)
            goto("retreat", (bin_box["cx"], bin_box["cy"], transit_z), q, tol=0.03, max_steps=200)
            entry["tick_end"] = now_ticks()
            if held:
                placed += 1
                placed_total += 1
                entry["result"] = "released over bin"
            else:
                entry["result"] = "lost during transport"
        report["placed_" + str(color)] = placed

    report["placed_total"] = placed_total
    report["bin_after"] = locate_bin()

    if goal["button"]:
        btn = locate_button()
        if btn is None:
            btn = button
        report["button_press_target"] = btn
        if btn is None:
            raise ValueError("button not found")
        act("close_for_press", close_gripper())
        qd = CFG["home_q"] if CFG["home_down"] else (0.0, 1.0, 0.0, 0.0)
        goto("above_button", (btn["cx"], btn["cy"], btn["z_top"] + 0.10), qd, tol=0.02, max_steps=250)
        goto("press_approach", (btn["cx"], btn["cy"], btn["z_top"] + 0.03), qd, tol=0.015, max_steps=120)
        goto("press", (btn["cx"], btn["cy"], btn["z_top"] - 0.015), qd, tol=0.01, max_steps=80)
        hold(5, 0.0)
        goto("press_release", (btn["cx"], btn["cy"], btn["z_top"] + 0.08), qd, tol=0.02, max_steps=120)
        hold(10, 0.0)
    else:
        hold(20, 1.0)
    report["final_tick"] = now_ticks()


CFG = {"base": "robot_base", "hz": 20.0, "home_q": (0.0, 1.0, 0.0, 0.0), "home_down": False}
try:
    main()
    report["completed"] = True
except Exception as exc:
    report["completed"] = False
    msg = str(exc)
    if msg.startswith(END_MARK):
        report["stopped"] = "episode ended after " + msg[len(END_MARK):]
    else:
        report["errors"].append("main: " + msg[:400])
result = report
