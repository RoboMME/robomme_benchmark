# VideoUnmask v02 -- goal parsing with plurals ("all containers hiding blue cubes" -> every
# instance), hidden cubes detected in BOTH conditioning-video cameras (front + wrist) and merged
# as 3D clusters, grasp centre = cube position refined by a wrist-camera view at the hover
# (no dependence on SAM3 listing every container), yaw from the wrist top face.
# v01 -- v00 + generalization hardening (colour gate derived from the video's own
# background, hue bins that keep orange/yellow apart from the wooden table, denser video sampling,
# container fallback via segment_points / video cube position, grasp check with one retry).
# v00 header follows.
# Records Stage 0 diagnostics in `result` AND attempts the task:
#   1. goal text -> target cube colour(s)
#   2. conditioning video (front camera, pure Python) -> where the target cube was before it
#      was covered (pixel + depth), and which coloured blobs disappear by the end of the video
#   3. live front image -> container candidates (SAM3 text prompts), 3D tops via public depth
#   4. container covering the target cube = mask containing the cube pixel / nearest in 3D
#   5. top-down grasp of that container, lift, hold.
# Only public APIs; no constants copied from any scene.
import math

report = {"version": "v02", "stage": "start", "errors": []}


def note_error(where, exc):
    report["errors"].append(where + ": " + str(exc)[:300])


def rnd(value, digits=4):
    try:
        return round(float(value), digits)
    except Exception:
        return None


def vec(values, digits=4):
    out = []
    for value in values:
        out.append(rnd(value, digits))
    return out


def as_rows(array):
    try:
        return array.tolist()
    except Exception:
        rows = []
        for row in array:
            rows.append(list(row))
        return rows


def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


# ---------------------------------------------------------------- quaternions
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def rotmat(q):
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def rotate(matrix, v):
    return (
        matrix[0][0] * v[0] + matrix[0][1] * v[1] + matrix[0][2] * v[2],
        matrix[1][0] * v[0] + matrix[1][1] * v[1] + matrix[1][2] * v[2],
        matrix[2][0] * v[0] + matrix[2][1] * v[1] + matrix[2][2] * v[2],
    )


def wrap90(angle):
    while angle > math.pi / 2:
        angle -= math.pi
    while angle <= -math.pi / 2:
        angle += math.pi
    return angle


# ---------------------------------------------------------------- colour
COLOR_WORDS = {
    "red": "red", "orange": "orange", "yellow": "yellow", "green": "green",
    "cyan": "cyan", "blue": "blue", "purple": "purple", "violet": "purple",
    "pink": "pink", "magenta": "pink",
}


SAT_GATE = [0.45]
CLASS_GATES = {}


def saturation(r, g, b):
    mx = max(int(r), int(g), int(b))
    if mx <= 0:
        return 0.0
    return (mx - min(int(r), int(g), int(b))) / float(mx)


def hue_class(r, g, b):
    r = int(r)
    g = int(g)
    b = int(b)
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 50:
        return None
    d = float(mx - mn)
    if d < 40:
        return None
    if mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * ((b - r) / d + 2.0)
    else:
        h = 60.0 * ((r - g) / d + 4.0)
    if h < 15 or h >= 340:
        return "red"
    if h < 25:
        return None  # red-orange boundary (wood-table hue): ambiguous, ignore
    if h < 50:
        return "orange"
    if h < 75:
        return "yellow"
    if h < 165:
        return "green"
    if h < 195:
        return "cyan"
    if h < 255:
        return "blue"
    if h < 300:
        return "purple"
    return "pink"


def calibrate_gate(camera):
    """Per-hue-class saturation gate from a (mostly background) frame: a class the background
    occupies (>= 1% of samples) gets gate = its background saturation p99 + 0.08 (max 0.9);
    every other class keeps the base gate. Returns the overall background saturation p99."""
    rows = as_rows(camera.rgb)
    sats = []
    per = {}
    total = 0
    for v in range(0, len(rows), 4):
        row = rows[v]
        for u in range(0, len(row), 4):
            p = row[u]
            total += 1
            if max(int(p[0]), int(p[1]), int(p[2])) < 50:
                continue
            sat = saturation(p[0], p[1], p[2])
            sats.append(sat)
            cls = hue_class(p[0], p[1], p[2])
            if cls is not None:
                if cls not in per:
                    per[cls] = []
                per[cls].append(sat)
    CLASS_GATES.clear()
    for cls in per:
        if len(per[cls]) >= 0.01 * max(1, total):
            CLASS_GATES[cls] = max(SAT_GATE[0], min(0.9, quantile(per[cls], 0.99) + 0.08))
    return quantile(sats, 0.99) if sats else None


def classify(r, g, b):
    cls = hue_class(r, g, b)
    if cls is None:
        return None
    gate = CLASS_GATES[cls] if cls in CLASS_GATES else SAT_GATE[0]
    if saturation(r, g, b) < gate:
        return None
    return cls


def parse_targets(text):
    """Colour mentions in goal order -> [(colour, plural)]. Plural when the colour is followed by
    'cubes'/'blocks' or preceded (within 4 words) by 'all'/'every'/'each'."""
    words = []
    for raw in str(text).lower().replace(",", " ").replace(".", " ").replace(";", " ").split():
        words.append(raw.strip("'\"()!?:"))
    near_cube = []
    anywhere = []
    for index, word in enumerate(words):
        if word in COLOR_WORDS:
            cls = COLOR_WORDS[word]
            window = words[index + 1:index + 4]
            before = words[max(0, index - 4):index]
            plural = ("cubes" in window or "blocks" in window or "all" in before
                      or "every" in before or "each" in before)
            anywhere.append((cls, plural))
            if "cube" in window or "cubes" in window or "block" in window or "blocks" in window:
                near_cube.append((cls, plural))
    chosen = near_cube if near_cube else anywhere
    ordered = []
    seen = []
    for cls, plural in chosen:
        if cls not in seen:
            seen.append(cls)
            ordered.append((cls, plural))
    return ordered, words


def class_pixels(camera, stride):
    rows = as_rows(camera.rgb)
    found = {}
    height = len(rows)
    width = len(rows[0]) if height else 0
    for v in range(0, height, stride):
        row = rows[v]
        for u in range(0, width, stride):
            p = row[u]
            cls = classify(p[0], p[1], p[2])
            if cls is not None:
                if cls not in found:
                    found[cls] = []
                found[cls].append((u, v))
    return found, width, height


def components(pixels, stride, reach=2):
    cells = set()
    for u, v in pixels:
        cells.add((u // stride, v // stride))
    seen = set()
    comps = []
    for cell in sorted(cells):
        if cell in seen:
            continue
        seen.add(cell)
        stack = [cell]
        members = []
        while stack:
            cu, cv = stack.pop()
            members.append((cu, cv))
            for du in range(-reach, reach + 1):
                for dv in range(-reach, reach + 1):
                    nb = (cu + du, cv + dv)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        us = [m[0] * stride for m in members]
        vs = [m[1] * stride for m in members]
        comps.append({
            "size": len(members),
            "u": sum(us) / float(len(us)),
            "v": sum(vs) / float(len(vs)),
            "box": [min(us), min(vs), max(us), max(vs)],
        })
    comps.sort(key=lambda c: -c["size"])
    return comps


# ---------------------------------------------------------------- projection
def project_pixel(u, v, depth, intrinsics, cam_pose, convention):
    fx = float(intrinsics[0][0])
    fy = float(intrinsics[1][1])
    cx = float(intrinsics[0][2])
    cy = float(intrinsics[1][2])
    xn = (u - cx) * depth / fx
    yn = (v - cy) * depth / fy
    if convention == "opencv":
        pc = (xn, yn, depth)
    elif convention == "opengl":
        pc = (xn, -yn, -depth)
    else:  # x forward, y left, z up
        pc = (depth, -xn, -yn)
    rot = rotmat(cam_pose.quaternion_wxyz)
    pr = rotate(rot, pc)
    t = cam_pose.position
    return (pr[0] + float(t[0]), pr[1] + float(t[1]), pr[2] + float(t[2]))


CONVENTIONS = ("opencv", "opengl", "xforward")


def camera_summary(camera):
    out = {}
    try:
        out["pose_pos"] = vec(camera.camera_pose.position)
        out["pose_quat"] = vec(camera.camera_pose.quaternion_wxyz)
        out["pose_frame"] = str(camera.camera_pose.frame)
    except Exception as exc:
        note_error("camera_pose", exc)
    try:
        out["intrinsics"] = [vec(row) for row in as_rows(camera.intrinsics)]
    except Exception as exc:
        note_error("intrinsics", exc)
    return out


def exec_summary(res):
    out = {}
    for name in ("ok", "status", "steps_executed", "terminated", "truncated", "error"):
        try:
            if name == "ok":
                out[name] = bool(res.ok)
            elif name == "status":
                out[name] = str(res.status)
            elif name == "steps_executed":
                out[name] = int(res.steps_executed)
            elif name == "terminated":
                out[name] = bool(res.terminated)
            elif name == "truncated":
                out[name] = bool(res.truncated)
            else:
                out[name] = str(res.error)[:200] if res.error is not None else None
        except Exception:
            out[name] = "n/a"
    return out


def ended(summary):
    return summary.get("terminated") is True or summary.get("truncated") is True


def robot_snapshot():
    try:
        st = get_robot_state()
        return {
            "t": rnd(st.timestamp_s, 3),
            "ee": vec(st.end_effector_poses["primary"].position),
            "grip": rnd(st.gripper_positions["primary"], 3),
        }
    except Exception as exc:
        return {"error": str(exc)[:200]}


motions = []


def cloud_info(pts, prompt, score):
    if len(pts) < 20:
        return None
    zs = [p[2] for p in pts]
    top = quantile(zs, 0.97)
    bottom = quantile(zs, 0.03)
    top_pts = [p for p in pts if p[2] >= top - 0.012]
    return {
        "prompt": prompt, "score": score, "box": [0.0, 0.0, 0.0, 0.0], "area": 0.0, "n_pts": len(pts),
        "top": top, "bottom": bottom,
        "cx": sum(p[0] for p in top_pts) / len(top_pts), "cy": sum(p[1] for p in top_pts) / len(top_pts),
        "mean": (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts), sum(zs) / len(zs)),
        "xspan": quantile([p[0] for p in pts], 0.95) - quantile([p[0] for p in pts], 0.05),
        "yspan": quantile([p[1] for p in pts], 0.95) - quantile([p[1] for p in pts], 0.05),
        "pts": pts, "top_pts": top_pts, "also": [prompt],
    }


def do(name, fn):
    res = fn()
    summary = exec_summary(res)
    summary["name"] = name
    summary["after"] = robot_snapshot() if not ended(summary) else {}
    motions.append(summary)
    return res, summary


# =============================================================== main
def main():
    context = get_task_context()
    goal = str(context.language)
    report["goal"] = goal
    report["task_name"] = str(context.task_name)
    try:
        meta = robomme_plus.get_task_metadata()
        report["task_metadata"] = {str(k): str(meta[k])[:300] for k in meta}
    except Exception as exc:
        note_error("task_metadata", exc)
    try:
        cmeta = robomme_plus.get_controller_metadata()
        report["controller_metadata"] = {str(k): str(cmeta[k])[:300] for k in cmeta}
    except Exception as exc:
        note_error("controller_metadata", exc)

    state0 = get_robot_state()
    base_frame = state0.base_frame
    home_pose = state0.end_effector_poses["primary"]
    report["state0"] = {
        "base_frame": str(base_frame),
        "embodiment": str(state0.embodiment),
        "ee_pos": vec(home_pose.position),
        "ee_quat": vec(home_pose.quaternion_wxyz),
        "joints": vec(state0.joint_positions["primary"]),
        "grip": rnd(state0.gripper_positions["primary"], 3),
        "t": rnd(state0.timestamp_s, 3),
    }

    targets, words = parse_targets(goal)
    report["targets"] = targets

    # ------------------------------------------------ live observation
    live = get_observation()
    live_front = live.cameras["front"]
    report["live_front_camera"] = camera_summary(live_front)
    report["cameras"] = [str(name) for name in live.cameras]
    video = robomme_plus.get_conditioning_video()
    n_frames = len(video)
    try:
        bg_cam = video[0].cameras["front"] if n_frames > 0 else live_front
        report["background_sat_p99"] = rnd(calibrate_gate(bg_cam), 3)
    except Exception as exc:
        note_error("gate", exc)
    report["sat_gates_front"] = {k: rnd(CLASS_GATES[k], 3) for k in CLASS_GATES}
    live_pixels, width, height = class_pixels(live_front, 2)
    report["image_size"] = [width, height]
    report["live_colors"] = {}
    for cls in live_pixels:
        entry = {"n": len(live_pixels[cls])}
        if len(live_pixels[cls]) < 2500:
            comps = components(live_pixels[cls], 2)
            entry["comps"] = [[c["size"], rnd(c["u"], 1), rnd(c["v"], 1)] for c in comps[:4]]
        report["live_colors"][cls] = entry

    # ------------------------------------------------ container candidates (live)
    live_depth_rows = as_rows(live_front.depth_m)
    live_intr = as_rows(live_front.intrinsics)
    prompts = ("box", "container", "bin", "cup", "white box")
    probes = []
    candidates = []
    for prompt in prompts:
        probe = {"prompt": prompt, "ok": False, "n": 0, "scores": [], "boxes": []}
        try:
            seg = segment_text("front", prompt)
        except Exception as exc:
            note_error("segment " + prompt, exc)
            probes.append(probe)
            continue
        probe["ok"] = bool(seg.ok)
        if seg.ok:
            probe["n"] = len(seg.segmentations)
            for cand in seg.segmentations[:6]:
                probe["scores"].append(rnd(cand.score, 3))
                box = None
                if cand.box_xyxy is not None:
                    box = [float(value) for value in cand.box_xyxy]
                probe["boxes"].append(vec(box, 1) if box else None)
                if box is None:
                    continue
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area < 20 or area > 0.4 * width * height:
                    continue
                info = {"prompt": prompt, "score": float(cand.score), "box": box, "area": area}
                cloud = None
                try:
                    cloud = mask_to_point_cloud(cand, "front", target_frame=base_frame)
                except Exception as exc:
                    try:
                        cloud = mask_to_point_cloud(cand.mask, "front", target_frame=base_frame)
                    except Exception as exc2:
                        note_error("cloud " + prompt, exc2)
                if cloud is None:
                    continue
                pts = as_rows(cloud.points)
                info["n_pts"] = len(pts)
                if len(pts) < 20:
                    continue
                zs = [p[2] for p in pts]
                top = quantile(zs, 0.97)
                bottom = quantile(zs, 0.03)
                top_pts = [p for p in pts if p[2] >= top - 0.012]
                info["top"] = top
                info["bottom"] = bottom
                info["cx"] = sum(p[0] for p in top_pts) / len(top_pts)
                info["cy"] = sum(p[1] for p in top_pts) / len(top_pts)
                info["mean"] = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts), sum(zs) / len(zs))
                info["xspan"] = quantile([p[0] for p in pts], 0.95) - quantile([p[0] for p in pts], 0.05)
                info["yspan"] = quantile([p[1] for p in pts], 0.95) - quantile([p[1] for p in pts], 0.05)
                info["pts"] = pts
                info["top_pts"] = top_pts
                # mask pixels (for pixel-containment test and projection calibration)
                try:
                    mrows = as_rows(cand.mask)
                    mpix = []
                    for v in range(0, height, 2):
                        mrow = mrows[v]
                        for u in range(0, width, 2):
                            if mrow[u]:
                                mpix.append((u, v))
                    info["mask_pix"] = mpix
                except Exception as exc:
                    note_error("mask pixels", exc)
                candidates.append(info)
        probes.append(probe)
    report["prompt_probes"] = probes

    # projection-convention calibration on live candidates (harness cloud is ground truth)
    calib = {}
    for conv in CONVENTIONS:
        errs = []
        for info in candidates:
            mpix = info.get("mask_pix")
            if not mpix:
                continue
            acc = [0.0, 0.0, 0.0]
            count = 0
            stride_pix = max(1, len(mpix) // 300)
            for u, v in mpix[::stride_pix]:
                d = float(live_depth_rows[v][u])
                if d <= 0.05 or d > 5.0:
                    continue
                p = project_pixel(u, v, d, live_intr, live_front.camera_pose, conv)
                acc[0] += p[0]
                acc[1] += p[1]
                acc[2] += p[2]
                count += 1
            if count < 10:
                continue
            m = info["mean"]
            errs.append(math.sqrt((acc[0] / count - m[0]) ** 2 + (acc[1] / count - m[1]) ** 2 + (acc[2] / count - m[2]) ** 2))
        if errs:
            calib[conv] = quantile(errs, 0.5)
    report["projection_calibration"] = {k: rnd(calib[k]) for k in calib}
    best_conv = None
    for conv in calib:
        if best_conv is None or calib[conv] < calib[best_conv]:
            best_conv = conv
    report["best_convention"] = best_conv

    # plausible containers, de-duplicated by top centre
    plausible = []
    for info in sorted(candidates, key=lambda i: -i["score"]):
        height_m = info["top"] - info["bottom"]
        radial = math.sqrt(info["cx"] ** 2 + info["cy"] ** 2)
        info["ok_shape"] = (
            0.015 <= height_m <= 0.3 and info["xspan"] < 0.2 and info["yspan"] < 0.2 and radial > 0.2
        )
        if not info["ok_shape"]:
            continue
        dup = False
        for other in plausible:
            if abs(other["cx"] - info["cx"]) < 0.025 and abs(other["cy"] - info["cy"]) < 0.025:
                dup = True
                other["also"].append(info["prompt"])
        if not dup:
            info["also"] = [info["prompt"]]
            plausible.append(info)
    report["containers"] = [
        {
            "prompt": i["prompt"], "also": i["also"], "score": rnd(i["score"], 3), "box": vec(i["box"], 1),
            "top": rnd(i["top"]), "bottom": rnd(i["bottom"]), "cx": rnd(i["cx"]), "cy": rnd(i["cy"]),
            "xspan": rnd(i["xspan"]), "yspan": rnd(i["yspan"]), "n_pts": i["n_pts"],
        }
        for i in plausible
    ]
    report["all_candidates"] = [
        {"prompt": i["prompt"], "score": rnd(i["score"], 3), "box": vec(i["box"], 1), "n_pts": i.get("n_pts"),
         "top": rnd(i.get("top")), "bottom": rnd(i.get("bottom")), "cx": rnd(i.get("cx")), "cy": rnd(i.get("cy")),
         "xspan": rnd(i.get("xspan")), "yspan": rnd(i.get("yspan")), "ok_shape": i.get("ok_shape")}
        for i in candidates
    ]
    report["stage"] = "containers_done"

    # ------------------------------------------------ hidden cubes: both video cameras, 3D clusters
    conv = best_conv if (best_conv is not None and calib[best_conv] < 0.02) else "opencv"
    report["video_convention"] = conv
    sample_count = min(n_frames, 120)
    sample_idx = []
    if n_frames > 0:
        for k in range(sample_count):
            idx = int(round(k * (n_frames - 1) / float(max(1, sample_count - 1))))
            if idx not in sample_idx:
                sample_idx.append(idx)
    target_classes = [entry[0] for entry in targets]
    detections = []
    timeline = []
    gates = {}
    for cam_name in ("front", "wrist"):
        try:
            gates[cam_name] = rnd(calibrate_gate(video[0].cameras[cam_name]), 3)
        except Exception as exc:
            note_error("gate " + cam_name, exc)
            continue
        gates[cam_name + "_class_gates"] = {k: rnd(CLASS_GATES[k], 3) for k in CLASS_GATES}
        for idx in sample_idx:
            cam = video[idx].cameras[cam_name]
            pixels, fw, fh = class_pixels(cam, 2)
            depth_rows = None
            intr = None
            seen = {}
            for cls in pixels:
                pts = pixels[cls]
                seen[cls] = len(pts)
                if cls not in target_classes or len(pts) > 3000:
                    continue
                for comp in components(pts, 2):
                    if comp["size"] < 4:
                        continue
                    if depth_rows is None:
                        depth_rows = as_rows(cam.depth_m)
                        intr = as_rows(cam.intrinsics)
                    du = int(round(comp["u"]))
                    dv = int(round(comp["v"]))
                    ds = []
                    for yy in range(max(0, dv - 2), min(fh, dv + 3)):
                        for xx in range(max(0, du - 2), min(fw, du + 3)):
                            val = float(depth_rows[yy][xx])
                            if 0.02 < val < 5.0:
                                ds.append(val)
                    if not ds:
                        continue
                    xyz = project_pixel(comp["u"], comp["v"], quantile(ds, 0.5), intr, cam.camera_pose, conv)
                    if xyz[2] < -0.03 or xyz[2] > 0.15:
                        continue
                    detections.append({"cls": cls, "xyz": xyz, "cam": cam_name, "i": idx, "size": comp["size"]})
            if seen and (sample_idx.index(idx) % 6 == 0):
                timeline.append([cam_name, idx, seen])
    report["video_frames"] = n_frames
    report["video_gates"] = gates
    report["video_timeline"] = timeline
    clusters = []
    for det in detections:
        home_cluster = None
        for cl in clusters:
            if cl["cls"] != det["cls"]:
                continue
            m = cl["xyz"]
            if math.sqrt((m[0] - det["xyz"][0]) ** 2 + (m[1] - det["xyz"][1]) ** 2) < 0.04:
                home_cluster = cl
                break
        if home_cluster is None:
            clusters.append({"cls": det["cls"], "xyz": list(det["xyz"]), "n": 1, "cams": [det["cam"]],
                             "first": det["i"], "last": det["i"], "max_size": det["size"]})
        else:
            n = home_cluster["n"]
            for axis in range(3):
                home_cluster["xyz"][axis] = (home_cluster["xyz"][axis] * n + det["xyz"][axis]) / (n + 1)
            home_cluster["n"] = n + 1
            if det["cam"] not in home_cluster["cams"]:
                home_cluster["cams"].append(det["cam"])
            home_cluster["last"] = max(home_cluster["last"], det["i"])
            home_cluster["first"] = min(home_cluster["first"], det["i"])
            home_cluster["max_size"] = max(home_cluster["max_size"], det["size"])
    # a hidden cube: seen repeatedly, a real blob, and no longer visible at the end of the video
    end_idx = sample_idx[-2] if len(sample_idx) >= 2 else -1
    for cl in clusters:
        cl["hidden_at_end"] = cl["last"] < end_idx
        cl["valid"] = cl["n"] >= 3 and cl["max_size"] >= 6 and cl["hidden_at_end"]
    report["cube_clusters"] = [
        {"cls": c["cls"], "xyz": vec(c["xyz"]), "n": c["n"], "cams": c["cams"], "first": c["first"], "last": c["last"],
         "max_size": c["max_size"], "hidden_at_end": c["hidden_at_end"], "valid": c["valid"]}
        for c in clusters
    ]
    report["stage"] = "video_done"

    # ------------------------------------------------ container reference geometry (live)
    full = [i for i in plausible if i["bottom"] < 0.03 and i["top"] < 0.2]
    ref_top = quantile([i["top"] for i in full], 0.5) if full else None
    ref_height = quantile([i["top"] - i["bottom"] for i in full], 0.5) if full else None
    report["ref_container"] = {"n_full": len(full), "top": rnd(ref_top), "height": rnd(ref_height)}

    # ------------------------------------------------ ordered pick list
    plan = []
    for cls, plural in targets:
        inst = [c for c in clusters if c["cls"] == cls and c["valid"]]
        inst.sort(key=lambda c: -c["n"])
        if not inst:
            plan.append({"cls": cls, "plural": plural, "missing": True})
            continue
        if not plural:
            inst = inst[:1]
        # nearest-first ordering within one colour group, starting from the last pick (or home)
        ordered = []
        cur = plan[-1]["xy"] if plan and "xy" in plan[-1] else (float(home_pose.position[0]), float(home_pose.position[1]))
        remaining = inst[:]
        while remaining:
            remaining.sort(key=lambda c: (c["xyz"][0] - cur[0]) ** 2 + (c["xyz"][1] - cur[1]) ** 2)
            nxt = remaining.pop(0)
            ordered.append(nxt)
            cur = (nxt["xyz"][0], nxt["xyz"][1])
        for c in ordered:
            item = {"cls": cls, "plural": plural, "xy": (c["xyz"][0], c["xyz"][1]), "n": c["n"]}
            near = None
            for index, info in enumerate(plausible):
                if info["top"] > 0.2:
                    continue
                dist = math.sqrt((info["cx"] - c["xyz"][0]) ** 2 + (info["cy"] - c["xyz"][1]) ** 2)
                if dist < 0.035 and (near is None or dist < near[0]):
                    near = (dist, index)
            if near is not None:
                item["container"] = near[1]
                item["container_dist"] = rnd(near[0])
            plan.append(item)
    report["plan"] = [{k: (vec(v) if k == "xy" else v) for k, v in item.items()} for item in plan]
    plan = [item for item in plan if not item.get("missing")]
    report["stage"] = "plan_done"
    if not plan:
        report["stage"] = "no_plan"
        return

    # ------------------------------------------------ grasp orientation basis
    home_q = tuple(float(value) for value in home_pose.quaternion_wxyz)
    approach = rotate(rotmat(home_q), (0.0, 0.0, 1.0))
    base_q = home_q if approach[2] < -0.9 else (0.0, 1.0, 0.0, 0.0)
    finger = rotate(rotmat(base_q), (0.0, 1.0, 0.0))
    finger_angle = math.atan2(finger[1], finger[0])
    report["grasp_basis"] = {"home_approach": vec(approach), "base_q": vec(base_q), "finger_angle_deg": rnd(math.degrees(finger_angle), 1)}

    def yaw_quat(top_pts):
        if not top_pts or len(top_pts) < 5:
            return base_q, None
        best = None
        for k in range(18):
            theta = math.radians(5.0 * k)
            c = math.cos(theta)
            s = math.sin(theta)
            xs = [p[0] * c + p[1] * s for p in top_pts]
            ys = [-p[0] * s + p[1] * c for p in top_pts]
            lx = quantile(xs, 0.95) - quantile(xs, 0.05)
            ly = quantile(ys, 0.95) - quantile(ys, 0.05)
            if best is None or lx * ly < best[0]:
                best = (lx * ly, theta, lx, ly)
        theta = best[1]
        if max(best[2], best[3]) / max(1e-6, min(best[2], best[3])) < 1.2:
            options = [theta, theta + math.pi / 2]
        elif best[2] < best[3]:
            options = [theta]
        else:
            options = [theta + math.pi / 2]
        delta = None
        for axis in options:
            cand = wrap90(axis - finger_angle)
            if delta is None or abs(cand) < abs(delta):
                delta = cand
        return qmul((math.cos(delta / 2), 0.0, 0.0, math.sin(delta / 2)), base_q), [rnd(best[2]), rnd(best[3]), rnd(math.degrees(delta), 1)]

    def wrist_container(cx, cy, top_guess, radius):
        """Nearest container in the current wrist view (public SAM3 + depth); no control ticks."""
        found = None
        listing = []
        for prompt in ("box", "white box"):
            try:
                wseg = segment_text("wrist", prompt)
            except Exception as exc:
                note_error("wrist seg", exc)
                continue
            if not wseg.ok:
                continue
            for cand in wseg.segmentations[:6]:
                try:
                    ci = cloud_info(as_rows(mask_to_point_cloud(cand, "wrist", target_frame=base_frame).points), prompt, float(cand.score))
                except Exception as exc:
                    note_error("wrist cloud", exc)
                    continue
                if ci is None:
                    continue
                dist = math.sqrt((ci["cx"] - cx) ** 2 + (ci["cy"] - cy) ** 2)
                listing.append([prompt, rnd(ci["cx"]), rnd(ci["cy"]), rnd(ci["top"]), rnd(dist)])
                if top_guess is not None and abs(ci["top"] - top_guess) > 0.03:
                    continue
                if ci["xspan"] > 0.12 or ci["yspan"] > 0.12:
                    continue
                if dist < radius and (found is None or dist < found[0]):
                    found = (dist, ci)
            if found is not None:
                break
        return found, listing

    for step_index, item in enumerate(plan):
        attempt = {"cls": item["cls"], "cube_xy": vec(item["xy"])}
        report["attempt_" + str(step_index)] = attempt
        cx = item["xy"][0]
        cy = item["xy"][1]
        top = ref_top
        top_pts = None
        if "container" in item:
            info = plausible[item["container"]]
            cx = info["cx"]
            cy = info["cy"]
            top = info["top"]
            top_pts = info["top_pts"]
        if top is None:
            top = 0.07
            attempt["top_default"] = True
        height_ref = ref_height if ref_height is not None else top
        grasp_q, yaw_info = yaw_quat(top_pts)
        attempt["front_center"] = [rnd(cx), rnd(cy)]
        attempt["front_yaw"] = yaw_info

        res, summ = do("open", lambda: open_gripper())
        if ended(summ):
            return
        pre = Pose((cx, cy, top + 0.10), grasp_q, base_frame)
        res, summ = do("pre_grasp", lambda: move_to_pose(pre, tolerance=0.01, max_steps=200))
        if ended(summ):
            return
        found, listing = wrist_container(item["xy"][0], item["xy"][1], top, 0.035)
        attempt["wrist_listing"] = listing[:10]
        if found is not None:
            ci = found[1]
            cx = ci["cx"]
            cy = ci["cy"]
            top = ci["top"]
            q2, yaw2 = yaw_quat(ci["top_pts"])
            grasp_q = q2
            attempt["wrist_center"] = [rnd(cx), rnd(cy), rnd(top)]
            attempt["wrist_yaw"] = yaw2
        grasp_z = top - min(0.025, 0.4 * height_ref)
        attempt["grasp_z"] = rnd(grasp_z)
        held = False
        for try_index in range(2):
            pre2 = Pose((cx, cy, top + 0.06), grasp_q, base_frame)
            grasp_pose = Pose((cx, cy, grasp_z), grasp_q, base_frame)
            res, summ = do("pre_grasp_refined", lambda: move_to_pose(pre2, tolerance=0.01, max_steps=150))
            if ended(summ):
                return
            res, summ = do("descend", lambda: move_to_pose(grasp_pose, tolerance=0.008, max_steps=150))
            if ended(summ):
                return
            res, summ = do("close", lambda: close_gripper())
            if ended(summ):
                return
            lift = Pose((cx, cy, grasp_z + 0.15), grasp_q, base_frame)
            res, summ = do("lift", lambda: move_to_pose(lift, tolerance=0.01, max_steps=150))
            if ended(summ):
                return
            st = get_robot_state()
            grip = float(st.gripper_positions["primary"])
            attempt["grip_after_lift_" + str(try_index)] = rnd(grip, 3)
            if grip > 0.1:
                held = True
                break
            # empty jaw (0 = closed): re-centre from the wrist view at the hover and retry once
            res, summ = do("reopen", lambda: open_gripper())
            if ended(summ):
                return
            res, summ = do("re_hover", lambda: move_to_pose(Pose((cx, cy, top + 0.10), grasp_q, base_frame), tolerance=0.01, max_steps=150))
            if ended(summ):
                return
            found, listing = wrist_container(cx, cy, top, 0.04)
            if found is not None:
                cx = found[1]["cx"]
                cy = found[1]["cy"]
                attempt["retry_center"] = [rnd(cx), rnd(cy)]
        attempt["held"] = held
        st = get_robot_state()
        for tick in range(6):
            reply = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=st.joint_positions["primary"],
                gripper_position=0.0, embodiment=st.embodiment)}))
            flags = {}
            try:
                flags = {"terminated": bool(reply.terminated), "truncated": bool(reply.truncated)}
            except Exception:
                flags = {}
            if flags.get("terminated") or flags.get("truncated"):
                motions.append({"name": "hold", "ended": True, "tick": tick, "flags": flags})
                return
        motions.append({"name": "hold", "ticks": 6, "after": robot_snapshot()})
        if step_index + 1 < len(plan):
            # put this container back where it was before the next pick
            back = Pose((cx, cy, grasp_z + 0.005), grasp_q, base_frame)
            res, summ = do("lower_back", lambda: move_to_pose(back, tolerance=0.01, max_steps=150))
            if ended(summ):
                return
            res, summ = do("release", lambda: open_gripper())
            if ended(summ):
                return
            up = Pose((cx, cy, top + 0.10), grasp_q, base_frame)
            res, summ = do("retreat", lambda: move_to_pose(up, tolerance=0.015, max_steps=150))
            if ended(summ):
                return
    report["stage"] = "all_done"


try:
    main()
except Exception as exc:
    note_error("main", exc)
report["motions"] = motions
report["final"] = robot_snapshot()
result = report
