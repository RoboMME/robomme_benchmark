# v01 -- BinFill (RoboMME-Plus, interaction-only). Perception repair after v00 (episode 0) dropped every cube
# on the BUTTON: v00 picked the bin by SAM score, and "bin" scores ~0.08 on this scene while
# "box" scores the button and the cubes almost as high as the bin. v01 selects the bin by
# geometry (hollow interior + largest footprint), checks cube/button colour from RGB pixels,
# re-observes from the home pose before each pick, and lowers the release height.
# (v00 header follows.)
# v00 -- BinFill initial program (RoboMME-Plus, interaction-only).
# Folds Stage 0 scene exploration into the first attempt: records goal, metadata,
# conditioning-video summary, segmentation probes and geometry, then attempts the task:
# parse "<count> <color> cube(s)" from the goal, pick each cube top-down, drop it above the
# observed bin, and press an observed button if the goal mentions one.
# Every task-dependent quantity is derived at runtime from goal text and live perception.
import math

report = {"version": "v01", "events": [], "errors": []}
result = report

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a": 1, "an": 1,
                "single": 1, "once": 1, "twice": 2}
COLOR_WORDS = ("red", "green", "blue", "yellow", "purple", "orange", "white", "black",
               "pink", "brown", "gray", "grey", "cyan", "magenta")
ENDED = {"value": False}


def log(name, value):
    report["events"].append([name, value])


def now_s():
    try:
        return round(float(get_robot_state().timestamp_s), 3)
    except Exception:
        return None


def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def qmat(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def qz(angle):
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]


def exec_summary(res):
    out = {}
    for key in ("ok", "status", "steps_executed", "terminated", "truncated"):
        try:
            if key == "ok":
                out[key] = bool(res.ok)
            elif key == "status":
                out[key] = str(res.status)
            elif key == "steps_executed":
                out[key] = int(res.steps_executed)
            elif key == "terminated":
                out[key] = bool(res.terminated)
            else:
                out[key] = bool(res.truncated)
        except Exception:
            out[key] = None
    try:
        if res.error is not None:
            out["error"] = str(res.error)[:300]
    except Exception:
        pass
    if out.get("terminated") or out.get("truncated"):
        ENDED["value"] = True
    return out


def do(name, res):
    summary = exec_summary(res)
    summary["t"] = now_s() if not ENDED["value"] else None
    log(name, summary)
    return summary


def gripper_reading():
    try:
        return round(float(get_robot_state().gripper_positions["primary"]), 4)
    except Exception:
        return None


def parse_goal(text):
    lowered = text.lower()
    cleaned = ""
    for ch in lowered:
        if ch.isalnum():
            cleaned += ch
        else:
            cleaned += " "
    tokens = cleaned.split()
    requests = []
    for index, token in enumerate(tokens):
        if token.startswith("cube") or token.startswith("block"):
            count = None
            color = None
            back = index - 1
            while back >= 0 and back >= index - 4:
                word = tokens[back]
                if color is None and word in COLOR_WORDS:
                    color = word
                elif count is None and word.isdigit():
                    count = int(word)
                elif count is None and word in NUMBER_WORDS:
                    count = NUMBER_WORDS[word]
                back -= 1
            requests.append({"color": color, "count": count, "token_index": index})
    button_index = None
    bin_index = None
    for index, token in enumerate(tokens):
        if token.startswith("button") and button_index is None:
            button_index = index
        if token in ("bin", "bins", "box", "basket", "container") and bin_index is None:
            bin_index = index
    return {"tokens": tokens, "requests": requests, "button_index": button_index,
            "bin_index": bin_index}


def seg_summary(found, limit=6):
    entry = {"ok": False, "n": 0, "scores": [], "boxes": []}
    try:
        entry["ok"] = bool(found.ok)
        entry["n"] = len(found.segmentations)
        for cand in found.segmentations[:limit]:
            entry["scores"].append(round(float(cand.score), 3))
            if cand.box_xyxy is not None:
                entry["boxes"].append([round(float(v), 1) for v in cand.box_xyxy])
    except Exception as exc:
        entry["error"] = str(exc)[:200]
    return entry


def cloud_of(cand, camera, base_frame):
    try:
        return mask_to_point_cloud(cand, camera, target_frame=base_frame)
    except Exception as exc:
        report["errors"].append("mask_to_point_cloud(cand): " + str(exc)[:200])
    return mask_to_point_cloud(cand.mask, camera, target_frame=base_frame)


def cloud_stats(points):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    zs = [float(p[2]) for p in points]
    if len(zs) < 5:
        return None
    return {
        "n": len(zs),
        "x": [round(quantile(xs, 0.05), 4), round(quantile(xs, 0.5), 4), round(quantile(xs, 0.95), 4)],
        "y": [round(quantile(ys, 0.05), 4), round(quantile(ys, 0.5), 4), round(quantile(ys, 0.95), 4)],
        "z": [round(quantile(zs, 0.05), 4), round(quantile(zs, 0.5), 4), round(quantile(zs, 0.95), 4)],
    }


def rgb_samples(camera, box):
    """RGB samples on a 5x5 grid over the central 40% of a pixel box (current image)."""
    out = []
    if box is None or len(box) < 4:
        return out
    try:
        image = get_observation().cameras[camera].rgb
        height = int(image.shape[0])
        width = int(image.shape[1])
        x1, y1, x2, y2 = [float(v) for v in box]
        for i in range(5):
            for j in range(5):
                px = int(x1 + (x2 - x1) * (0.3 + 0.1 * i))
                py = int(y1 + (y2 - y1) * (0.3 + 0.1 * j))
                px = min(max(px, 0), width - 1)
                py = min(max(py, 0), height - 1)
                pixel = image[py][px]
                out.append((int(pixel[0]), int(pixel[1]), int(pixel[2])))
    except Exception as exc:
        report["errors"].append("rgb_samples: " + str(exc)[:200])
    return out


def classify_pixel(rgb):
    r, g, b = rgb
    top = max(r, g, b)
    if top < 25:
        return "dark"
    if (top - min(r, g, b)) < 0.3 * top:
        return "gray"
    if r > 2.5 * g and r > 2.5 * b:
        return "red"
    if b > 1.3 * r and b > 1.1 * g:
        return "blue"
    if g > 1.3 * r and g > 1.2 * b:
        return "green"
    if r > 1.5 * b and g > 1.5 * b and g > 0.7 * r:
        return "yellow"
    return "other"


def color_votes(samples):
    votes = {}
    for sample in samples:
        label = classify_pixel(sample)
        votes[label] = votes.get(label, 0) + 1
    return votes


def candidates(camera, prompts, base_frame, min_score):
    """All candidates across prompts with stats, de-duplicated by xy proximity."""
    found_all = []
    for prompt in prompts:
        found = segment_text(camera, prompt)
        log("seg:" + camera + ":" + prompt, seg_summary(found))
        if not found.ok:
            continue
        projected = 0
        for cand in found.segmentations[:10]:
            score = float(cand.score)
            if score < min_score:
                continue
            try:
                cloud = cloud_of(cand, camera, base_frame)
                stats = cloud_stats(cloud.points)
            except Exception as exc:
                report["errors"].append("cloud " + prompt + ": " + str(exc)[:200])
                stats = None
            if stats is None:
                continue
            projected += 1
            duplicate = False
            for other in found_all:
                dx = other["stats"]["x"][1] - stats["x"][1]
                dy = other["stats"]["y"][1] - stats["y"][1]
                if math.sqrt(dx * dx + dy * dy) < 0.02:
                    duplicate = True
            if not duplicate:
                box = []
                if cand.box_xyxy is not None:
                    box = [round(float(v), 1) for v in cand.box_xyxy]
                found_all.append({"prompt": prompt, "score": round(score, 3), "box": box,
                                  "votes": color_votes(rgb_samples(camera, box)),
                                  "stats": stats, "points": cloud.points})
        if projected == 0 and len(found.segmentations) > 0:
            # fallback: top-1 through the composed helper
            try:
                loc = localize_object(prompt, camera_name=camera, target_frame=base_frame)
                if loc.ok:
                    stats = cloud_stats(loc.point_cloud.points)
                    if stats is not None:
                        found_all.append({"prompt": prompt + " (localize)",
                                          "score": round(float(found.segmentations[0].score), 3),
                                          "box": [], "stats": stats,
                                          "points": loc.point_cloud.points})
            except Exception as exc:
                report["errors"].append("localize fallback: " + str(exc)[:200])
    return found_all


def top_yaw(points, top_z):
    """Yaw (rad) of the minimum-area xy rectangle of the top-face points, in [0, pi/2)."""
    top = [(float(p[0]), float(p[1])) for p in points if float(p[2]) > top_z - 0.012]
    if len(top) < 6:
        return None
    best = None
    best_area = None
    for k in range(18):
        angle = k * (math.pi / 2.0) / 18.0
        c = math.cos(angle)
        s = math.sin(angle)
        us = [x * c + y * s for (x, y) in top]
        vs = [-x * s + y * c for (x, y) in top]
        area = (max(us) - min(us)) * (max(vs) - min(vs))
        if best_area is None or area < best_area:
            best_area = area
            best = angle
    return best


def grasp_quaternion(down_q, cube_yaw):
    if cube_yaw is None:
        return down_q
    rot = qmat(down_q)
    hand_y_yaw = math.atan2(rot[1][1], rot[0][1])
    delta = cube_yaw - hand_y_yaw
    while delta > math.pi / 4.0:
        delta -= math.pi / 2.0
    while delta < -math.pi / 4.0:
        delta += math.pi / 2.0
    return qmul(qz(delta), down_q)


def move(name, position, quat, base_frame, tolerance=0.015, max_steps=200):
    pose = Pose([float(position[0]), float(position[1]), float(position[2])], quat, base_frame)
    res = move_to_pose(pose, tolerance=tolerance, max_steps=max_steps)
    summary = do(name, res)
    if not ENDED["value"]:
        try:
            ee = get_robot_state().end_effector_poses["primary"].position
            summary["ee"] = [round(float(v), 4) for v in ee]
        except Exception:
            pass
    return summary


def settle(ticks, gripper):
    state = get_robot_state()
    count = 0
    for tick in range(ticks):
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=state.joint_positions["primary"],
            gripper_position=gripper, embodiment=state.embodiment)}))
        count += 1
        try:
            if reply.terminated or reply.truncated:
                ENDED["value"] = True
                break
        except Exception:
            pass
    return count


try:
    context = get_task_context()
    goal = str(context.language)
    report["goal"] = goal
    report["task_name"] = str(context.task_name)
    try:
        meta = robomme_plus.get_task_metadata()
        report["task_metadata"] = {str(k): str(meta[k])[:300] for k in meta}
    except Exception as exc:
        report["errors"].append("task_metadata: " + str(exc)[:200])
    try:
        cmeta = robomme_plus.get_controller_metadata()
        report["controller_metadata"] = {str(k): str(cmeta[k])[:400] for k in cmeta}
    except Exception as exc:
        report["errors"].append("controller_metadata: " + str(exc)[:200])
    try:
        video = robomme_plus.get_conditioning_video()
        report["video_len"] = len(video)
        if len(video) > 0:
            report["video_t"] = [float(video[0].timestamp_s), float(video[len(video) - 1].timestamp_s)]
    except Exception as exc:
        report["errors"].append("video: " + str(exc)[:200])

    state = get_robot_state()
    base_frame = state.base_frame
    report["base_frame"] = str(base_frame)
    report["embodiment"] = str(state.embodiment)
    report["t0"] = float(state.timestamp_s)
    home_joints = [float(v) for v in state.joint_positions["primary"]]
    report["home_joints"] = [round(v, 4) for v in home_joints]
    ee0 = state.end_effector_poses["primary"]
    down_q = [float(v) for v in ee0.quaternion_wxyz]
    report["ee0"] = {"p": [round(float(v), 4) for v in ee0.position],
                     "q": [round(v, 4) for v in down_q]}
    rot0 = qmat(down_q)
    report["ee0_z_axis"] = [round(rot0[0][2], 3), round(rot0[1][2], 3), round(rot0[2][2], 3)]
    if rot0[2][2] > -0.9:
        down_q = [0.0, 1.0, 0.0, 0.0]
        report["down_q_fallback"] = True
    report["gripper0"] = gripper_reading()

    obs = get_observation()
    cams = {}
    for name in obs.cameras:
        cam = obs.cameras[name]
        entry = {"shape": [int(v) for v in cam.rgb.shape]}
        try:
            entry["pose"] = [round(float(v), 4) for v in cam.camera_pose.position]
            entry["q"] = [round(float(v), 4) for v in cam.camera_pose.quaternion_wxyz]
            entry["K"] = [[round(float(v), 2) for v in row] for row in cam.intrinsics]
        except Exception as exc:
            entry["error"] = str(exc)[:200]
        cams[str(name)] = entry
    report["cameras"] = cams

    parsed = parse_goal(goal)
    report["parsed"] = {"requests": parsed["requests"], "button_index": parsed["button_index"],
                        "bin_index": parsed["bin_index"]}

    # ---- Scene probes (front camera) ----
    for prompt in ("cube", "bin", "button"):
        log("probe_front:" + prompt, seg_summary(segment_text("front", prompt)))
    for prompt in ("cube", "bin"):
        log("probe_wrist:" + prompt, seg_summary(segment_text("wrist", prompt)))

    # ---- Bin: chosen by geometry, not SAM score ----
    # Every candidate of every prompt (score >= 0.03) is projected; the bin is the candidate
    # whose footprint centre sits clearly below its own rim (hollow) and whose footprint is the
    # largest. A cube or button is solid: its centre is its highest part.
    bins = []
    for prompt in ("bin", "box", "container", "open box"):
        found = segment_text("front", prompt)
        log("seg:front:" + prompt, seg_summary(found))
        if not found.ok:
            continue
        for cand in found.segmentations[:8]:
            if float(cand.score) < 0.03:
                continue
            try:
                pts = cloud_of(cand, "front", base_frame).points
                st = cloud_stats(pts)
            except Exception as exc:
                report["errors"].append("bin cloud: " + str(exc)[:200])
                st = None
            if st is None:
                continue
            span = max(st["x"][2] - st["x"][0], st["y"][2] - st["y"][0])
            cx = 0.5 * (st["x"][0] + st["x"][2])
            cy = 0.5 * (st["y"][0] + st["y"][2])
            hx = 0.25 * (st["x"][2] - st["x"][0])
            hy = 0.25 * (st["y"][2] - st["y"][0])
            central = [float(p[2]) for p in pts
                       if abs(float(p[0]) - cx) < hx and abs(float(p[1]) - cy) < hy]
            hollow = None
            if len(central) >= 5:
                hollow = st["z"][2] - quantile(central, 0.5)
            box = []
            if cand.box_xyxy is not None:
                box = [round(float(v), 1) for v in cand.box_xyxy]
            bins.append({"prompt": prompt, "score": round(float(cand.score), 3), "box": box,
                         "stats": st, "points": pts, "span": round(span, 4),
                         "hollow": None if hollow is None else round(hollow, 4),
                         "center": [round(cx, 4), round(cy, 4)]})
    # table height = lowest 5% z over all projected candidates (they all rest on the table);
    # candidates that do not reach the table (e.g. the gripper seen at home) are rejected.
    table_z = None
    for cand in bins:
        if table_z is None or cand["stats"]["z"][0] < table_z:
            table_z = cand["stats"]["z"][0]
    report["table_z"] = table_z
    for cand in bins:
        cand["on_table"] = (cand["stats"]["z"][0] < table_z + 0.03
                            and cand["stats"]["z"][2] < table_z + 0.25)
    bin_choice = None
    for cand in bins:
        if cand["span"] < 0.06 or cand["span"] > 0.6 or not cand["on_table"]:
            continue
        if cand["hollow"] is None or cand["hollow"] < 0.015:
            continue
        if bin_choice is None or cand["span"] > bin_choice["span"]:
            bin_choice = cand
    if bin_choice is None:
        report["errors"].append("no hollow bin candidate; using largest footprint")
        for cand in bins:
            if cand["span"] < 0.06 or cand["span"] > 0.6 or not cand["on_table"]:
                continue
            if bin_choice is None or cand["span"] > bin_choice["span"]:
                bin_choice = cand
    report["bin_candidates"] = [{"prompt": c["prompt"], "score": c["score"], "box": c["box"],
                                 "stats": c["stats"], "span": c["span"], "hollow": c["hollow"],
                                 "on_table": c["on_table"]}
                                for c in bins]
    if bin_choice is None:
        raise ValueError("no bin candidate")
    bst = bin_choice["stats"]
    bin_xy = bin_choice["center"]
    zs_bin = [float(p[2]) for p in bin_choice["points"]]
    bin_top = quantile(zs_bin, 0.97)
    report["bin"] = {"xy": bin_xy, "top": round(bin_top, 4), "x": bst["x"], "y": bst["y"],
                     "prompt": bin_choice["prompt"], "score": bin_choice["score"],
                     "hollow": bin_choice["hollow"], "span": bin_choice["span"]}

    # ---- Requests ----
    requests = []
    for req in parsed["requests"]:
        if req["count"] is not None or req["color"] is not None:
            requests.append(req)
    if len(requests) == 0:
        requests = [{"color": None, "count": 1}]
    report["plan"] = requests

    def in_bin(stats):
        return (bst["x"][0] - 0.01 <= stats["x"][1] <= bst["x"][2] + 0.01
                and bst["y"][0] - 0.01 <= stats["y"][1] <= bst["y"][2] + 0.01)

    def pile_top():
        """Highest top (z95) of any cube seen inside the bin footprint, or None."""
        best = None
        found = segment_text("front", "cube")
        if not found.ok:
            return None
        for cand in found.segmentations[:10]:
            if float(cand.score) < 0.25:
                continue
            try:
                st = cloud_stats(cloud_of(cand, "front", base_frame).points)
            except Exception as exc:
                report["errors"].append("pile cloud: " + str(exc)[:200])
                st = None
            if st is None or not in_bin(st):
                continue
            if max(st["x"][2] - st["x"][0], st["y"][2] - st["y"][0]) > 0.09:
                continue
            if best is None or st["z"][2] > best:
                best = st["z"][2]
        return best

    def find_cubes(color):
        prompts = ("cube",) if color is None else (color + " cube", color + " block")
        found = candidates("front", prompts, base_frame, 0.25)
        good = []
        for cand in found:
            st = cand["stats"]
            span = max(st["x"][2] - st["x"][0], st["y"][2] - st["y"][0])
            height = st["z"][2] - st["z"][0]
            cand["span"] = round(span, 4)
            if span > 0.09 or height > 0.09:
                continue
            if st["z"][2] > table_z + 0.12:
                continue
            if in_bin(st):
                continue
            if color is not None and color in ("red", "green", "blue", "yellow"):
                votes = cand["votes"]
                mine = votes.get(color, 0)
                other = 0
                for label in ("red", "green", "blue", "yellow"):
                    if label != color:
                        other = max(other, votes.get(label, 0))
                if other > mine:
                    continue
            good.append(cand)
        log("cubes:" + str(color), [{"score": c["score"], "box": c["box"], "stats": c["stats"],
                                     "span": c["span"], "votes": c["votes"]} for c in found])
        return good

    def locate_button():
        """Button = small, solid, gray (unsaturated) candidate outside the bin footprint."""
        found_all = []
        for prompt in ("button", "white button", "gray button", "push button"):
            found = segment_text("front", prompt)
            log("seg:front:" + prompt, seg_summary(found))
            if not found.ok:
                continue
            for cand in found.segmentations[:8]:
                if float(cand.score) < 0.03:
                    continue
                try:
                    pts = cloud_of(cand, "front", base_frame).points
                    st = cloud_stats(pts)
                except Exception as exc:
                    report["errors"].append("button cloud: " + str(exc)[:200])
                    st = None
                if st is None:
                    continue
                box = []
                if cand.box_xyxy is not None:
                    box = [round(float(v), 1) for v in cand.box_xyxy]
                span = max(st["x"][2] - st["x"][0], st["y"][2] - st["y"][0])
                found_all.append({"prompt": prompt, "score": round(float(cand.score), 3),
                                  "box": box, "stats": st, "points": pts,
                                  "span": round(span, 4),
                                  "votes": color_votes(rgb_samples("front", box))})
        log("buttons", [{"prompt": c["prompt"], "score": c["score"], "box": c["box"],
                         "stats": c["stats"], "span": c["span"], "votes": c["votes"]}
                        for c in found_all])
        best = None
        for cand in found_all:
            st = cand["stats"]
            if cand["span"] < 0.015 or cand["span"] > 0.10:
                continue
            if st["z"][0] > table_z + 0.03:
                continue
            if in_bin(st):
                continue
            votes = cand["votes"]
            neutral = votes.get("gray", 0) + votes.get("dark", 0)
            colored = 0
            for label in ("red", "green", "blue", "yellow"):
                colored = max(colored, votes.get(label, 0))
            if colored >= neutral:
                continue
            if best is None or cand["score"] > best["score"]:
                best = cand
        if best is None:
            report["errors"].append("no button candidate")
            return None
        st = best["stats"]
        top = quantile([float(p[2]) for p in best["points"]], 0.95)
        bxy = [0.5 * (st["x"][0] + st["x"][2]), 0.5 * (st["y"][0] + st["y"][2])]
        return {"xy": [round(bxy[0], 4), round(bxy[1], 4)], "top": round(top, 4),
                "prompt": best["prompt"], "score": best["score"], "span": best["span"]}

    def press_button(button):
        if button is None:
            return False
        bxy = button["xy"]
        top = button["top"]
        do("button_close", close_gripper())
        if ENDED["value"]:
            return True
        move("button_above", [bxy[0], bxy[1], top + 0.10], down_q, base_frame)
        if ENDED["value"]:
            return True
        for depth in (0.02, 0.035):
            move("button_press_" + str(depth), [bxy[0], bxy[1], top - depth], down_q, base_frame,
                 tolerance=0.01, max_steps=120)
            if ENDED["value"]:
                return True
            settle(5, 0.0)
            if ENDED["value"]:
                return True
            move("button_release", [bxy[0], bxy[1], top + 0.06], down_q, base_frame)
            if ENDED["value"]:
                return True
        return True

    button = None
    if parsed["button_index"] is not None:
        button = locate_button()
    report["button"] = button

    button_first = (parsed["button_index"] is not None and len(parsed["requests"]) > 0
                    and parsed["button_index"] < parsed["requests"][0]["token_index"])
    report["button_first"] = button_first
    if button_first:
        press_button(button)

    placed = []
    attempts = 0
    for req in requests:
        need = req["count"] if req["count"] is not None else 1
        done_for_req = 0
        req_attempts = 0
        waits = 0
        while done_for_req < need and req_attempts < need + 3 and not ENDED["value"]:
            req_attempts += 1
            attempts += 1
            if attempts > 1:
                do("observe_home", move_to_joints(home_joints, tolerance=0.02, max_steps=150))
                if ENDED["value"]:
                    break
            cubes = find_cubes(req["color"])
            pile = pile_top()
            log("pile_top", pile)
            if len(cubes) == 0:
                # cubes appear on the table over time (observed in v00): wait at home and look again
                waits += 1
                report["errors"].append("no cube for " + str(req["color"]) + " (wait " + str(waits) + ")")
                if waits > 8:
                    break
                req_attempts -= 1
                settle(20, 1.0)
                continue
            # nearest to the bin first keeps transport short
            best = None
            best_d = None
            for cand in cubes:
                d = math.sqrt((cand["stats"]["x"][1] - bin_xy[0]) ** 2 + (cand["stats"]["y"][1] - bin_xy[1]) ** 2)
                if best_d is None or d < best_d:
                    best_d = d
                    best = cand
            st = best["stats"]
            zs = [float(p[2]) for p in best["points"]]
            top = quantile(zs, 0.95)
            low = quantile(zs, 0.05)
            height = min(max(top - low, 0.015), 0.06)
            grasp_z = top - 0.5 * height
            yaw = top_yaw(best["points"], top)
            gq = grasp_quaternion(down_q, yaw)
            cx = st["x"][1]
            cy = st["y"][1]
            pick = {"xy": [cx, cy], "top": round(top, 4), "low": round(low, 4),
                    "grasp_z": round(grasp_z, 4), "yaw": yaw, "score": best["score"]}
            log("pick_target", pick)
            do("open", open_gripper())
            if ENDED["value"]:
                break
            move("pre_grasp", [cx, cy, grasp_z + 0.10], gq, base_frame)
            if ENDED["value"]:
                break
            move("grasp", [cx, cy, grasp_z], gq, base_frame, tolerance=0.01)
            if ENDED["value"]:
                break
            do("close", close_gripper())
            if ENDED["value"]:
                break
            g_closed = gripper_reading()
            move("lift", [cx, cy, grasp_z + 0.15], gq, base_frame)
            if ENDED["value"]:
                break
            g_lift = gripper_reading()
            log("gripper_after_close_lift", [g_closed, g_lift])
            if g_lift is not None and g_lift < 0.05:
                report["errors"].append("grasp missed (gripper fully closed)")
                do("open_after_miss", open_gripper())
                continue
            release_base = bin_top
            if pile is not None and pile > release_base:
                release_base = pile
            transit_z = max(bin_top + 0.15, grasp_z + 0.15, release_base + 0.12)
            move("above_bin_high", [bin_xy[0], bin_xy[1], transit_z], gq, base_frame)
            if ENDED["value"]:
                break
            move("above_bin_release", [bin_xy[0], bin_xy[1], release_base + 0.05], gq, base_frame)
            if ENDED["value"]:
                break
            g_release = gripper_reading()
            do("release", open_gripper())
            if ENDED["value"]:
                break
            settle(8, 1.0)
            if ENDED["value"]:
                break
            move("retreat", [bin_xy[0], bin_xy[1], transit_z], gq, base_frame)
            placed.append({"color": req["color"], "g_release": g_release})
            if g_release is not None and g_release >= 0.05:
                done_for_req += 1
        report["placed"] = placed

    if not ENDED["value"] and parsed["button_index"] is not None and not button_first:
        press_button(button)
    if not ENDED["value"]:
        settle(10, gripper_reading() if gripper_reading() is not None else 1.0)
    report["t_end"] = now_s() if not ENDED["value"] else None
except Exception as exc:
    report["errors"].append("fatal: " + str(exc)[:400])

report["ended"] = ENDED["value"]
result = report
