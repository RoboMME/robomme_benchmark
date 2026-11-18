# PatternLock v00 -- combined scene snapshot + first attempt (start 1 of 3).
#
# Hypothesis (to be verified by this start's recording): PatternLock is an imitation task; the
# conditioning video shows the robot tracing a pattern over targets with a held stick, and the
# live execution must retrace it.  This program
#   1. records goal, task/controller metadata, the conditioning video's full robot-state timeline,
#      coarse thumbnails of sampled video frames and of the live front view, and SAM prompt probes;
#   2. compares video frame 0 with the live scene (pixel difference + colour-blob registration of
#      target-like objects, colour learned from live SAM candidates);
#   3. replays the video's joint trajectory over its moving segment ("replay"), or, if the target
#      layout is translated relative to the video, replays the video EE path shifted by the
#      measured translation through IK ("shifted").
# Nothing here is seed-specific: every position comes from the video robot state or from
# perception of the current episode.

import math

report = {"version": "v00", "phase": "start", "errors": []}
result = report


def note_error(where, exc):
    report["errors"].append(where + ": " + str(exc)[:300])


def r4(value):
    return round(float(value), 4)


def vec(values):
    return [r4(v) for v in values]


def is_terminal(res):
    try:
        if res.terminated:
            return True
    except Exception:
        pass
    try:
        if res.truncated:
            return True
    except Exception:
        pass
    try:
        text = str(res.status).lower()
        if "terminat" in text or "truncat" in text:
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------- context and metadata
ctx = get_task_context()
report["language"] = ctx.language
try:
    report["task_name"] = ctx.task_name
except Exception as exc:
    note_error("task_name", exc)
task_meta = {}
try:
    task_meta = dict(robomme_plus.get_task_metadata())
    report["task_metadata"] = task_meta
except Exception as exc:
    note_error("task_metadata", exc)
try:
    report["controller_metadata"] = dict(robomme_plus.get_controller_metadata())
except Exception as exc:
    note_error("controller_metadata", exc)

hz = 20.0
for key in ("control_hz", "control_frequency", "control_freq", "frequency_hz"):
    try:
        if task_meta.get(key) is not None:
            hz = float(task_meta.get(key))
            break
    except Exception:
        pass
max_steps = 0
try:
    max_steps = int(task_meta.get("max_steps"))
except Exception:
    max_steps = 0
report["hz"] = hz
report["max_steps"] = max_steps

state0 = get_robot_state()
base_frame = state0.base_frame
embodiment = state0.embodiment
live_q0 = [float(v) for v in state0.joint_positions["primary"]]
report["live_start"] = {
    "joints": vec(live_q0),
    "ee": vec(state0.end_effector_poses["primary"].position),
    "quat": vec(state0.end_effector_poses["primary"].quaternion_wxyz),
    "gripper": r4(state0.gripper_positions["primary"]),
    "embodiment": embodiment,
    "base_frame": base_frame,
    "timestamp_s": r4(state0.timestamp_s),
}

# ---------------------------------------------------------------- conditioning video timeline
video = []
try:
    video = robomme_plus.get_conditioning_video()
except Exception as exc:
    note_error("conditioning_video", exc)
n_video = len(video)
report["video_frames"] = n_video

v_t = []
v_q = []
v_ee = []
v_quat = []
v_grip = []
for index in range(n_video):
    frame = video[index]
    rs = frame.robot_state
    v_t.append(float(frame.timestamp_s))
    v_q.append([float(v) for v in rs.joint_positions["primary"]])
    pose = rs.end_effector_poses["primary"]
    v_ee.append([float(v) for v in pose.position])
    v_quat.append([float(v) for v in pose.quaternion_wxyz])
    v_grip.append(float(rs.gripper_positions["primary"]))
report["video_timeline"] = {
    "t": [r4(v) for v in v_t],
    "ee": [vec(p) for p in v_ee],
    "quat": [vec(p) for p in v_quat],
    "q": [vec(p) for p in v_q],
    "grip": [r4(v) for v in v_grip],
}
report["phase"] = "video_timeline_recorded"


# ---------------------------------------------------------------- image helpers
def as_lists(array):
    try:
        return array.tolist()
    except Exception:
        return [[array[row][col] for col in range(len(array[0]))] for row in range(len(array))]


def thumbnail(rgb_rows, stride):
    flat = []
    for row in range(0, len(rgb_rows), stride):
        line = rgb_rows[row]
        for col in range(0, len(line), stride):
            pixel = line[col]
            flat.append(int(pixel[0]))
            flat.append(int(pixel[1]))
            flat.append(int(pixel[2]))
    return flat


def diff_fraction(rows_a, rows_b, stride, threshold):
    changed = 0
    total = 0
    for row in range(0, len(rows_a), stride):
        line_a = rows_a[row]
        line_b = rows_b[row]
        for col in range(0, len(line_a), stride):
            pa = line_a[col]
            pb = line_b[col]
            total += 1
            if abs(int(pa[0]) - int(pb[0])) + abs(int(pa[1]) - int(pb[1])) + abs(int(pa[2]) - int(pb[2])) > threshold:
                changed += 1
    return changed / max(1, total)


def quat_to_matrix(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def camera_model(cam):
    intr = as_lists(cam.intrinsics)
    pose = cam.camera_pose
    return {
        "fx": float(intr[0][0]), "fy": float(intr[1][1]),
        "cx": float(intr[0][2]), "cy": float(intr[1][2]),
        "R": quat_to_matrix(pose.quaternion_wxyz),
        "t": [float(v) for v in pose.position],
    }


def deproject(model, u, v, depth, convention):
    xn = (u - model["cx"]) / model["fx"] * depth
    yn = (v - model["cy"]) / model["fy"] * depth
    if convention == "opencv":
        pc = (xn, yn, depth)
    elif convention == "opengl":
        pc = (xn, -yn, -depth)
    else:  # body frame: x forward, y left, z up
        pc = (depth, -xn, -yn)
    rot = model["R"]
    tr = model["t"]
    return (
        rot[0][0] * pc[0] + rot[0][1] * pc[1] + rot[0][2] * pc[2] + tr[0],
        rot[1][0] * pc[0] + rot[1][1] * pc[1] + rot[1][2] * pc[2] + tr[1],
        rot[2][0] * pc[0] + rot[2][1] * pc[1] + rot[2][2] * pc[2] + tr[2],
    )


def median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[len(ordered) // 2]


# ---------------------------------------------------------------- live observation + probes
obs = get_observation()
live_cam = obs.cameras["front"]
live_rgb = as_lists(live_cam.rgb)
live_depth = as_lists(live_cam.depth_m)
live_model = camera_model(live_cam)
report["live_front_camera"] = {
    "fx": r4(live_model["fx"]), "fy": r4(live_model["fy"]),
    "cx": r4(live_model["cx"]), "cy": r4(live_model["cy"]),
    "position": vec(live_model["t"]),
    "quat": vec(live_cam.camera_pose.quaternion_wxyz),
    "shape": [len(live_rgb), len(live_rgb[0])],
}
report["live_front_thumb64"] = thumbnail(live_rgb, 4)
depth_thumb = []
for row in range(0, len(live_depth), 4):
    for col in range(0, len(live_depth[row]), 4):
        dval = float(live_depth[row][col])
        depth_thumb.append(int(1000 * dval) if (dval > 0 and dval < 20) else -1)
report["live_front_depth_thumb64_mm"] = depth_thumb
try:
    wrist_rgb = as_lists(obs.cameras["wrist"].rgb)
    report["live_wrist_thumb64"] = thumbnail(wrist_rgb, 4)
except Exception as exc:
    note_error("wrist_thumb", exc)

PROMPTS = ["target", "disc", "button", "circle", "dot", "stick", "grid", "cube"]
probes = []
target_like = []
for prompt in PROMPTS:
    entry = {"prompt": prompt, "ok": False, "candidates": []}
    try:
        seg = segment_text("front", prompt)
        entry["ok"] = bool(seg.ok)
        if seg.ok:
            for cand in seg.segmentations[:5]:
                info = {"score": r4(cand.score)}
                try:
                    info["label"] = str(cand.label)
                except Exception:
                    pass
                box = None
                if cand.box_xyxy is not None:
                    box = [float(v) for v in cand.box_xyxy]
                    info["box"] = [round(v, 1) for v in box]
                try:
                    cloud = mask_to_point_cloud(cand, "front", target_frame=base_frame)
                    pts = cloud.points
                    info["n_points"] = len(pts)
                    if len(pts) > 0:
                        xs = [float(p[0]) for p in pts]
                        ys = [float(p[1]) for p in pts]
                        zs = [float(p[2]) for p in pts]
                        info["median"] = [r4(median(xs)), r4(median(ys)), r4(median(zs))]
                        info["min"] = [r4(min(xs)), r4(min(ys)), r4(min(zs))]
                        info["max"] = [r4(max(xs)), r4(max(ys)), r4(max(zs))]
                except Exception as exc:
                    info["cloud_error"] = str(exc)[:200]
                if box is not None:
                    uc = int((box[0] + box[2]) / 2.0)
                    vc = int((box[1] + box[3]) / 2.0)
                    uc = min(max(uc, 1), len(live_rgb[0]) - 2)
                    vc = min(max(vc, 1), len(live_rgb) - 2)
                    rlist = []
                    glist = []
                    blist = []
                    for dv in (-1, 0, 1):
                        for du in (-1, 0, 1):
                            px = live_rgb[vc + dv][uc + du]
                            rlist.append(int(px[0]))
                            glist.append(int(px[1]))
                            blist.append(int(px[2]))
                    info["center_px"] = [uc, vc]
                    info["center_rgb"] = [median(rlist), median(glist), median(blist)]
                    info["center_depth"] = r4(live_depth[vc][uc])
                entry["candidates"].append(info)
                if prompt in ("target", "disc", "button", "circle", "dot") and "max" in info and "center_rgb" in info:
                    span_x = info["max"][0] - info["min"][0]
                    span_y = info["max"][1] - info["min"][1]
                    span_z = info["max"][2] - info["min"][2]
                    if span_z <= 0.03 and 0.008 <= max(span_x, span_y) <= 0.12:
                        target_like.append(info)
    except Exception as exc:
        entry["error"] = str(exc)[:300]
    probes.append(entry)
report["prompt_probes"] = probes
report["phase"] = "probes_recorded"

# ---------------------------------------------------------------- deprojection convention check
convention = None
conv_errors = {}
for name in ("opencv", "opengl", "body"):
    errs = []
    for info in target_like:
        uc, vc = info["center_px"]
        d = float(live_depth[vc][uc])
        if not (d > 0 and d < 20):
            continue
        p = deproject(live_model, uc, vc, d, name)
        m = info["median"]
        errs.append(math.sqrt((p[0] - m[0]) ** 2 + (p[1] - m[1]) ** 2 + (p[2] - m[2]) ** 2))
    if errs:
        conv_errors[name] = r4(median(errs))
best_err = None
for name in conv_errors:
    if best_err is None or conv_errors[name] < best_err:
        best_err = conv_errors[name]
        convention = name
if best_err is not None and best_err > 0.03:
    convention = None
report["deprojection"] = {"errors": conv_errors, "convention": convention}

# reference colours from target-like candidates (dedupe within 40 RGB units)
ref_colors = []
for info in target_like:
    col = info["center_rgb"]
    dup = False
    for ref in ref_colors:
        if abs(ref[0] - col[0]) + abs(ref[1] - col[1]) + abs(ref[2] - col[2]) < 40:
            dup = True
    if not dup:
        ref_colors.append(col)
z_band = None
if target_like:
    zs_med = [info["median"][2] for info in target_like]
    z_band = [min(zs_med) - 0.02, max(zs_med) + 0.02]
report["ref_colors"] = ref_colors
report["target_z_band"] = z_band


def blob_size(blob):
    return blob["n"]


def detect_blobs(rgb_rows, depth_rows, model, stride, tol):
    height = len(rgb_rows)
    width = len(rgb_rows[0])
    hit = {}
    for row in range(0, height, stride):
        for col in range(0, width, stride):
            px = rgb_rows[row][col]
            pr, pg, pb = int(px[0]), int(px[1]), int(px[2])
            match = False
            for ref in ref_colors:
                dist = math.sqrt((pr - ref[0]) ** 2 + (pg - ref[1]) ** 2 + (pb - ref[2]) ** 2)
                if dist <= tol:
                    match = True
                    break
            if not match:
                continue
            d = float(depth_rows[row][col])
            if not (d > 0 and d < 20):
                continue
            p = deproject(model, col, row, d, convention)
            if z_band is not None and (p[2] < z_band[0] or p[2] > z_band[1]):
                continue
            hit[(row, col)] = p
    seen = set()
    blobs = []
    for key in hit:
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        members = []
        while stack:
            cur = stack.pop()
            members.append(cur)
            for dr in (-stride, 0, stride):
                for dc in (-stride, 0, stride):
                    nb = (cur[0] + dr, cur[1] + dc)
                    if nb in hit and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        if len(members) < 4:
            continue
        sx = sum(hit[m][0] for m in members) / len(members)
        sy = sum(hit[m][1] for m in members) / len(members)
        sz = sum(hit[m][2] for m in members) / len(members)
        su = sum(m[1] for m in members) / len(members)
        sv = sum(m[0] for m in members) / len(members)
        blobs.append({"xyz": [r4(sx), r4(sy), r4(sz)], "px": [round(su, 1), round(sv, 1)], "n": len(members)})
    blobs = sorted(blobs, key=blob_size, reverse=True)
    return blobs[:25]


# ---------------------------------------------------------------- video frames: thumbnails + comparison
video_thumbs = []
v0_rgb = None
v0_depth = None
v0_model = None
if n_video > 0:
    sample_idx = list(range(0, n_video, 16))
    if (n_video - 1) not in sample_idx:
        sample_idx.append(n_video - 1)
    for index in sample_idx:
        try:
            cam = video[index].cameras["front"]
            rows = as_lists(cam.rgb)
            entry = {"i": index, "t": r4(v_t[index]), "thumb32": thumbnail(rows, 8),
                     "diff_vs_live": r4(diff_fraction(rows, live_rgb, 2, 60))}
            if index == 0:
                v0_rgb = rows
                v0_depth = as_lists(cam.depth_m)
                v0_model = camera_model(cam)
                entry["thumb64"] = thumbnail(rows, 4)
                entry["camera_position"] = vec(v0_model["t"])
                entry["camera_quat"] = vec(cam.camera_pose.quaternion_wxyz)
            if index == n_video - 1:
                entry["thumb64"] = thumbnail(rows, 4)
            video_thumbs.append(entry)
        except Exception as exc:
            note_error("video_frame_" + str(index), exc)
report["video_thumbs"] = video_thumbs
report["phase"] = "video_images_recorded"

# ---------------------------------------------------------------- layout comparison
pixel_diff0 = None
if video_thumbs and video_thumbs[0]["i"] == 0:
    pixel_diff0 = video_thumbs[0]["diff_vs_live"]
shift = None
reg = {"pixel_diff_frame0": pixel_diff0}
if convention is not None and ref_colors and v0_rgb is not None:
    try:
        live_blobs = detect_blobs(live_rgb, live_depth, live_model, 2, 45)
        video_blobs = detect_blobs(v0_rgb, v0_depth, v0_model, 2, 45)
        reg["live_blobs"] = live_blobs
        reg["video_blobs"] = video_blobs
        best = None
        for vb in video_blobs:
            for lb in live_blobs:
                dx = lb["xyz"][0] - vb["xyz"][0]
                dy = lb["xyz"][1] - vb["xyz"][1]
                inl = 0
                res_sum = 0.0
                for vb2 in video_blobs:
                    near = None
                    for lb2 in live_blobs:
                        dd = math.sqrt((lb2["xyz"][0] - vb2["xyz"][0] - dx) ** 2 + (lb2["xyz"][1] - vb2["xyz"][1] - dy) ** 2)
                        if near is None or dd < near:
                            near = dd
                    if near is not None and near < 0.012:
                        inl += 1
                        res_sum += near
                if best is None or inl > best[0] or (inl == best[0] and res_sum < best[1]):
                    best = (inl, res_sum, dx, dy)
        if best is not None:
            inl, res_sum, dx, dy = best
            mx = []
            my = []
            for vb2 in video_blobs:
                near = None
                pick = None
                for lb2 in live_blobs:
                    dd = math.sqrt((lb2["xyz"][0] - vb2["xyz"][0] - dx) ** 2 + (lb2["xyz"][1] - vb2["xyz"][1] - dy) ** 2)
                    if near is None or dd < near:
                        near = dd
                        pick = lb2
                if near is not None and near < 0.012:
                    mx.append(pick["xyz"][0] - vb2["xyz"][0])
                    my.append(pick["xyz"][1] - vb2["xyz"][1])
            if mx:
                dx = sum(mx) / len(mx)
                dy = sum(my) / len(my)
            reg["inliers"] = inl
            reg["shift_xy"] = [r4(dx), r4(dy)]
            need = max(2, int(0.6 * min(len(video_blobs), len(live_blobs)) + 0.5))
            reg["inliers_needed"] = need
            if inl >= need:
                shift = (dx, dy)
    except Exception as exc:
        note_error("registration", exc)
report["registration"] = reg

mode = "replay"
if pixel_diff0 is not None and pixel_diff0 < 0.005:
    mode = "replay"
elif shift is not None and math.sqrt(shift[0] ** 2 + shift[1] ** 2) >= 0.008:
    mode = "shifted"
report["mode"] = mode
report["phase"] = "analysis_done"

# ---------------------------------------------------------------- moving segment of the video
moving = []
for index in range(1, n_video):
    dq = max(abs(v_q[index][j] - v_q[index - 1][j]) for j in range(7))
    moving.append(dq)
i0 = None
i1 = None
for index in range(len(moving)):
    if moving[index] > 0.002:
        if i0 is None:
            i0 = index
        i1 = index + 1
report["video_motion_range"] = [i0, i1]
report["video_max_dq"] = r4(max(moving)) if moving else None

# ---------------------------------------------------------------- execute
exec_log = {"ticks": 0, "terminal": False, "samples": []}
report["execution"] = exec_log


def hold_command(q, grip):
    return RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=[float(v) for v in q],
        gripper_position=grip, embodiment=embodiment)})


try:
    if i0 is None:
        raise ValueError("no robot motion found in conditioning video")
    # joint targets per video frame
    targets = []
    if mode == "shifted":
        seed = v_q[i0]
        ik_fail = 0
        for index in range(i0, i1 + 1):
            pos = v_ee[index]
            q = v_quat[index]
            qn = math.sqrt(sum(c * c for c in q))
            goal = Pose([pos[0] + shift[0], pos[1] + shift[1], pos[2]], [c / qn for c in q], base_frame)
            ik = solve_ik(goal, seed_joints=seed)
            if ik.ok:
                sol = [float(v) for v in ik.joint_positions]
                seed = sol
                targets.append(sol)
            else:
                ik_fail += 1
                targets.append(None)
        exec_log["ik_failures"] = ik_fail
        filled = []
        last = None
        for tq in targets:
            if tq is not None:
                last = tq
            filled.append(last)
        first_valid = None
        for tq in filled:
            if tq is not None:
                first_valid = tq
                break
        targets = [tq if tq is not None else first_valid for tq in filled]
        if first_valid is None:
            raise ValueError("shifted replay: every IK failed")
    else:
        for index in range(i0, i1 + 1):
            targets.append(v_q[index])

    ticks_per = [1]
    for index in range(i0 + 1, i1 + 1):
        ticks_per.append(max(1, int(round((v_t[index] - v_t[index - 1]) * hz))))
    total_ticks = sum(ticks_per)
    budget = max_steps - 220 if max_steps > 0 else total_ticks
    scale = 1.0
    if total_ticks > budget > 0:
        scale = budget / total_ticks
    exec_log["planned_ticks"] = total_ticks
    exec_log["scale"] = r4(scale)

    grip = float(state0.gripper_positions["primary"])
    approach = move_to_joints(targets[0], tolerance=0.01, max_steps=150)
    approach_info = {"ok": bool(approach.ok)}
    try:
        approach_info["status"] = str(approach.status)
        approach_info["steps"] = int(approach.steps_executed)
        exec_log["ticks"] += int(approach.steps_executed)
    except Exception as exc:
        note_error("approach_fields", exc)
    exec_log["approach"] = approach_info
    if is_terminal(approach):
        exec_log["terminal"] = True
    report["phase"] = "replaying"
    carry = 0.0
    stop = exec_log["terminal"]
    for k in range(1, len(targets)):
        if stop:
            break
        carry += ticks_per[k] * scale
        n_sub = int(carry)
        carry -= n_sub
        if n_sub <= 0:
            continue
        qa = targets[k - 1]
        qb = targets[k]
        for s in range(1, n_sub + 1):
            frac = s / n_sub
            q = [qa[j] + (qb[j] - qa[j]) * frac for j in range(7)]
            res = step(hold_command(q, grip))
            exec_log["ticks"] += 1
            if is_terminal(res):
                exec_log["terminal"] = True
                exec_log["terminal_at_frame"] = i0 + k
                stop = True
                break
        if (not stop) and k % 5 == 0:
            st = get_robot_state()
            exec_log["samples"].append([i0 + k, exec_log["ticks"]] + vec(st.end_effector_poses["primary"].position))
    report["phase"] = "replay_done"
    if not exec_log["terminal"]:
        final_q = targets[len(targets) - 1]
        for tick in range(15):
            res = step(hold_command(final_q, grip))
            exec_log["ticks"] += 1
            if is_terminal(res):
                exec_log["terminal"] = True
                exec_log["terminal_in_hold"] = tick
                break
    if not exec_log["terminal"]:
        st = get_robot_state()
        exec_log["final_ee"] = vec(st.end_effector_poses["primary"].position)
        exec_log["final_joints"] = vec(st.joint_positions["primary"])
    report["phase"] = "done"
except Exception as exc:
    note_error("execute", exc)

result = report
