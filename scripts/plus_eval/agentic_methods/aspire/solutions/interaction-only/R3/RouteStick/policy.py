# RouteStick v00 -- combined scene snapshot + first attempt (one simulator start).
#
# Strategy (hypothesis, interaction-only, no recordings): the episode's own conditioning video shows the
# route; every video frame carries the robot state. Re-execute the video's joint path in the
# live episode, one native control tick per step(), resampled on the video's own timestamps.
# Everything a snapshot would record (goal, metadata, calibration, video state timeline,
# subsampled RGB/depth grids, SAM probes, per-tick live EE path) goes into `result`, because
# R3 runs may retain only results and logs.
import math

HEX = "0123456789abcdef"
DT_DEFAULT = 0.05
LOOKAHEAD_TICKS = 1.0     # command the video state this many ticks ahead (controller lag)
NEAR_START_RAD = 0.08     # live joints within this of the video start: servo there directly
LIFT_M = 0.15             # transit lift when the live start is far from the video start
IDLE_RAD = 0.003          # joint motion below this counts as idle (trim video ends)
HOLD_TICKS = 15
PROBE_PROMPTS = ["stick", "obstacle", "pillar", "pole", "cylinder", "block", "target", "peg"]
CLOUD_PROMPTS = ["stick", "obstacle", "target"]

report = {"version": "v00_initial", "errors": [], "notes": [], "phases": []}
result = report
state_box = {"terminal": False}


def err(tag, exc):
    report["errors"].append(tag + ": " + str(exc)[:500])


def tolist(arr):
    try:
        return arr.tolist()
    except Exception:
        return [float(value) for value in arr]


def flist(arr, nd=4):
    return [round(float(value), nd) for value in tolist(arr)]


def sub_grid(arr, stride):
    try:
        return arr[::stride, ::stride].tolist()
    except Exception:
        rows = []
        h = len(arr)
        w = len(arr[0])
        for i in range(0, h, stride):
            row = []
            for j in range(0, w, stride):
                row.append(tolist(arr[i][j]) if len(arr.shape) == 3 else float(arr[i][j]))
            rows.append(row)
        return rows


def rgb_hex(rgb, stride):
    grid = sub_grid(rgb, stride)
    out = []
    for row in grid:
        text = ""
        for px in row:
            for channel in px[:3]:
                value = int(channel)
                text = text + HEX[value // 16] + HEX[value % 16]
        out.append(text)
    return out


def depth_grid(depth, stride):
    grid = sub_grid(depth, stride)
    return [[round(float(value), 3) for value in row] for row in grid]


def rgb_diff(a_grid, b_grid):
    total = 0.0
    count = 0
    for row_a, row_b in zip(a_grid, b_grid):
        for px_a, px_b in zip(row_a, row_b):
            for ca, cb in zip(px_a[:3], px_b[:3]):
                total += abs(int(ca) - int(cb))
                count += 1
    if count == 0:
        return None
    return round(total / count, 2)


def camera_info(cam):
    info = {}
    try:
        info["intrinsics"] = [flist(row, 3) for row in tolist(cam.intrinsics)]
    except Exception as exc:
        err("intrinsics", exc)
    try:
        info["pose_position"] = flist(cam.camera_pose.position, 4)
        info["pose_quat_wxyz"] = flist(cam.camera_pose.quaternion_wxyz, 5)
        info["pose_frame"] = str(cam.camera_pose.frame)
    except Exception as exc:
        err("camera_pose", exc)
    try:
        info["shape"] = [int(value) for value in cam.rgb.shape]
    except Exception as exc:
        err("rgb shape", exc)
    return info


def is_terminal(reply):
    flags = {}
    for name in ("terminated", "truncated", "done", "terminal"):
        try:
            if name == "terminated":
                flags[name] = bool(reply.terminated)
            elif name == "truncated":
                flags[name] = bool(reply.truncated)
            elif name == "done":
                flags[name] = bool(reply.done)
            else:
                flags[name] = bool(reply.terminal)
        except Exception:
            pass
    for value in flags.values():
        if value:
            return True
    return False


def exec_summary(tag, res):
    entry = {"phase": tag}
    try:
        entry["ok"] = bool(res.ok)
    except Exception:
        pass
    try:
        entry["status"] = str(res.status)
    except Exception:
        pass
    try:
        entry["steps_executed"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        entry["error"] = str(res.error)[:300]
    except Exception:
        pass
    entry["terminal"] = is_terminal(res)
    if entry["terminal"]:
        state_box["terminal"] = True
    report["phases"].append(entry)
    if "repr_execution" not in report:
        report["repr_execution"] = str(res)[:1500]
    return entry


def interp_q(times, joints, t, hint):
    index = hint
    n = len(times)
    while index + 1 < n - 1 and times[index + 1] <= t:
        index += 1
    if t <= times[0]:
        return list(joints[0]), 0
    if t >= times[n - 1]:
        return list(joints[n - 1]), n - 2
    t0 = times[index]
    t1 = times[index + 1]
    alpha = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
    alpha = max(0.0, min(1.0, alpha))
    q = [a + alpha * (b - a) for a, b in zip(joints[index], joints[index + 1])]
    return q, index


try:
    context = get_task_context()
    report["task_language"] = str(context.language)
    report["task_name"] = str(context.task_name)
    report["task_id"] = str(context.task_id)
    try:
        report["context_metadata"] = dict(context.metadata)
    except Exception as exc:
        err("context metadata", exc)
except Exception as exc:
    err("context", exc)

meta = {}
try:
    meta = dict(robomme_plus.get_task_metadata())
    report["task_metadata"] = meta
except Exception as exc:
    err("task metadata", exc)
try:
    report["controller_metadata"] = dict(robomme_plus.get_controller_metadata())
except Exception as exc:
    err("controller metadata", exc)

dt = DT_DEFAULT
max_steps = None
for key in meta:
    lowered = str(key).lower()
    try:
        if "max_steps" in lowered:
            max_steps = int(meta[key])
        if ("hz" in lowered or "freq" in lowered) and float(meta[key]) > 1.0:
            dt = 1.0 / float(meta[key])
    except Exception:
        pass
report["dt"] = dt
report["max_steps"] = max_steps

state = get_robot_state()
base_frame = state.base_frame
embodiment = state.embodiment
report["base_frame"] = str(base_frame)
report["embodiment"] = str(embodiment)
report["repr_state"] = str(state)[:2000]
q_live = [float(value) for value in tolist(state.joint_positions["primary"])]
grip_live = float(state.gripper_positions["primary"])
ee_live = state.end_effector_poses["primary"]
report["live_start"] = {
    "q": [round(value, 4) for value in q_live],
    "gripper": round(grip_live, 4),
    "ee_pos": flist(ee_live.position),
    "ee_quat": flist(ee_live.quaternion_wxyz, 5),
    "timestamp_s": float(state.timestamp_s),
}

# ---------------- live observation ----------------
live_front_grid32 = None
try:
    obs = get_observation()
    report["cameras"] = [str(name) for name in obs.cameras]
    report["camera_info"] = {}
    for name in obs.cameras:
        report["camera_info"][str(name)] = camera_info(obs.cameras[name])
    front = obs.cameras["front"]
    wrist = obs.cameras["wrist"]
    live_front_grid32 = sub_grid(front.rgb, 8)
    report["live_front_rgb64"] = rgb_hex(front.rgb, 4)
    report["live_front_depth64"] = depth_grid(front.depth_m, 4)
    report["live_wrist_rgb64"] = rgb_hex(wrist.rgb, 4)
    report["live_wrist_depth64"] = depth_grid(wrist.depth_m, 4)
except Exception as exc:
    err("live observation", exc)

# ---------------- conditioning video ----------------
vt = []
vq = []
vp = []
frames = []
try:
    video = robomme_plus.get_conditioning_video()
    frames = list(video)
    report["video_len"] = len(frames)
    if len(frames) > 0:
        report["repr_video_frame0_state"] = str(frames[0].robot_state)[:1500]
    for fr in frames:
        rs = fr.robot_state
        vt.append(float(fr.timestamp_s))
        vq.append([float(value) for value in tolist(rs.joint_positions["primary"])])
        vp.append([float(value) for value in tolist(rs.end_effector_poses["primary"].position)])
    report["video_t"] = [round(value, 3) for value in vt]
    report["video_q"] = [[round(value, 4) for value in q] for q in vq]
    report["video_ee"] = [[round(value, 4) for value in p] for p in vp]
    vg = []
    for fr in frames:
        try:
            vg.append(round(float(fr.robot_state.gripper_positions["primary"]), 3))
        except Exception:
            vg.append(None)
    report["video_gripper"] = vg
    if len(frames) > 0:
        quat0 = frames[0].robot_state.end_effector_poses["primary"].quaternion_wxyz
        report["video_ee_quat0"] = flist(quat0, 5)
        report["video_ee_quat_last"] = flist(
            frames[len(frames) - 1].robot_state.end_effector_poses["primary"].quaternion_wxyz, 5)
        report["video_camera_info0"] = {}
        for name in frames[0].cameras:
            report["video_camera_info0"][str(name)] = camera_info(frames[0].cameras[name])
except Exception as exc:
    err("video", exc)

nvid = len(vt)
try:
    if nvid > 0:
        picks = [0, nvid // 2, nvid - 1]
        report["video_front_rgb64"] = {}
        report["video_front_depth64"] = {}
        report["video_wrist_rgb64"] = {}
        for index in picks:
            cam = frames[index].cameras["front"]
            report["video_front_rgb64"][str(index)] = rgb_hex(cam.rgb, 4)
            report["video_front_depth64"][str(index)] = depth_grid(cam.depth_m, 4)
            report["video_wrist_rgb64"][str(index)] = rgb_hex(frames[index].cameras["wrist"].rgb, 4)
        stride = max(1, nvid // 32)
        series = []
        for index in range(0, nvid, stride):
            fr = frames[index]
            grid = sub_grid(fr.cameras["front"].rgb, 8)
            item = {"i": index, "t": round(vt[index], 3)}
            if live_front_grid32 is not None:
                item["diff_vs_live"] = rgb_diff(grid, live_front_grid32)
            item["front_rgb32"] = rgb_hex(fr.cameras["front"].rgb, 8)
            if index % (stride * 4) == 0:
                item["wrist_rgb32"] = rgb_hex(fr.cameras["wrist"].rgb, 8)
            series.append(item)
        report["video_series"] = series
except Exception as exc:
    err("video images", exc)

# ---------------- SAM probes on the live front view ----------------
report["prompt_probes"] = []
for prompt in PROBE_PROMPTS:
    probe = {"prompt": prompt, "ok": False, "n": 0, "scores": [], "boxes": []}
    try:
        seg = segment_text("front", prompt)
        probe["ok"] = bool(seg.ok)
        if seg.ok:
            probe["n"] = len(seg.segmentations)
            for cand in seg.segmentations[:5]:
                probe["scores"].append(round(float(cand.score), 3))
                if cand.box_xyxy is not None:
                    probe["boxes"].append([round(float(value), 1) for value in cand.box_xyxy])
            if prompt in CLOUD_PROMPTS:
                clouds = []
                for cand in seg.segmentations[:3]:
                    cloud = mask_to_point_cloud(cand, "front", target_frame=base_frame)
                    pts = tolist(cloud.points)
                    if len(pts) == 0:
                        clouds.append({"n": 0})
                        continue
                    xs = sorted(float(p[0]) for p in pts)
                    ys = sorted(float(p[1]) for p in pts)
                    zs = sorted(float(p[2]) for p in pts)
                    m = len(pts)
                    clouds.append({
                        "n": m,
                        "med": [round(xs[m // 2], 4), round(ys[m // 2], 4), round(zs[m // 2], 4)],
                        "lo": [round(xs[m // 20], 4), round(ys[m // 20], 4), round(zs[m // 20], 4)],
                        "hi": [round(xs[(19 * m) // 20], 4), round(ys[(19 * m) // 20], 4),
                               round(zs[(19 * m) // 20], 4)],
                    })
                probe["clouds"] = clouds
    except Exception as exc:
        probe["error"] = str(exc)[:300]
    report["prompt_probes"].append(probe)

# ---------------- attempt: replay the video's joint path ----------------
live_path = []


def send(q, tag_tick):
    reply = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=[float(value) for value in q],
        gripper_position=grip_live, embodiment=embodiment)}))
    if "repr_step" not in report:
        report["repr_step"] = str(reply)[:1500]
    if is_terminal(reply):
        state_box["terminal"] = True
    if tag_tick % 2 == 0 or state_box["terminal"]:
        try:
            rs = reply.observation.robot_state
            live_path.append([tag_tick, round(float(rs.timestamp_s), 3)] +
                             flist(rs.end_effector_poses["primary"].position))
        except Exception as exc:
            if "path_err" not in report:
                report["path_err"] = str(exc)[:300]
    return reply


try:
    if nvid < 2:
        report["notes"].append("no conditioning video: nothing to replay")
    else:
        moving = [i for i in range(nvid)
                  if max(abs(a - b) for a, b in zip(vq[i], vq[0])) > IDLE_RAD]
        i_start = max(0, moving[0] - 1) if moving else 0
        moving_end = [i for i in range(nvid)
                      if max(abs(a - b) for a, b in zip(vq[i], vq[nvid - 1])) > IDLE_RAD]
        i_end = min(nvid - 1, moving_end[len(moving_end) - 1] + 1) if moving_end else nvid - 1
        report["replay_window"] = [i_start, i_end, round(vt[i_start], 3), round(vt[i_end], 3)]
        d_start = max(abs(a - b) for a, b in zip(q_live, vq[i_start]))
        d_end = max(abs(a - b) for a, b in zip(q_live, vq[nvid - 1]))
        d_first = max(abs(a - b) for a, b in zip(q_live, vq[0]))
        report["live_vs_video"] = {"d_first": round(d_first, 4), "d_start": round(d_start, 4),
                                   "d_last": round(d_end, 4)}
        # approach the video's start configuration
        if d_start <= NEAR_START_RAD:
            res = move_to_joints(vq[i_start], tolerance=0.01, max_steps=60)
            exec_summary("approach_servo", res)
        else:
            p_live = [float(value) for value in tolist(ee_live.position)]
            quat_live = [float(value) for value in tolist(ee_live.quaternion_wxyz)]
            p_goal = vp[i_start]
            z_transit = max(p_live[2], p_goal[2]) + LIFT_M
            up = Pose([p_live[0], p_live[1], z_transit], quat_live, base_frame)
            exec_summary("approach_lift", move_to_pose(up, tolerance=0.02, max_steps=200))
            if not state_box["terminal"]:
                over = Pose([p_goal[0], p_goal[1], z_transit], quat_live, base_frame)
                exec_summary("approach_transit", move_to_pose(over, tolerance=0.02, max_steps=250))
            if not state_box["terminal"]:
                exec_summary("approach_descend",
                             move_to_joints(vq[i_start], tolerance=0.01, max_steps=200))
        used = 0
        for entry in report["phases"]:
            used += entry.get("steps_executed", 0)
        duration = vt[i_end] - vt[i_start]
        speed = 1.0
        n_ticks = int(math.ceil(duration / dt))
        if max_steps is not None:
            allowance = int(0.85 * (max_steps - used - HOLD_TICKS))
            if allowance > 0 and n_ticks > allowance:
                speed = float(n_ticks) / float(allowance)
                n_ticks = allowance
        report["replay_plan"] = {"duration_s": round(duration, 3), "ticks": n_ticks,
                                 "speed": round(speed, 3), "used_before": used}
        hint = i_start
        ticks_done = 0
        sub_t = vt[i_start:i_end + 1]
        sub_q = vq[i_start:i_end + 1]
        hint = 0
        for k in range(1, n_ticks + 1):
            if state_box["terminal"]:
                break
            t = vt[i_start] + min(duration, (k + LOOKAHEAD_TICKS) * dt * speed)
            q, hint = interp_q(sub_t, sub_q, t, hint)
            send(q, k)
            ticks_done = k
        report["replay_ticks_done"] = ticks_done
        for k in range(HOLD_TICKS):
            if state_box["terminal"]:
                break
            send(vq[i_end], n_ticks + 1 + k)
        report["terminal_seen"] = state_box["terminal"]
except Exception as exc:
    err("replay", exc)

report["live_path"] = live_path
try:
    if not state_box["terminal"]:
        final = get_robot_state()
        report["live_end"] = {
            "q": flist(final.joint_positions["primary"]),
            "ee_pos": flist(final.end_effector_poses["primary"].position),
            "timestamp_s": float(final.timestamp_s),
        }
        fobs = get_observation()
        report["final_front_rgb64"] = rgb_hex(fobs.cameras["front"].rgb, 4)
except Exception as exc:
    err("final", exc)

result = report
