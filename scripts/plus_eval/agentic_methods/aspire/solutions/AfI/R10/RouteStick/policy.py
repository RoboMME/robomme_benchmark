import math

# v00 — RouteStick initial program (AfI, scene exploration folded into the first real program).
# 1. Records goal, task/controller metadata, live robot state and SAM3 prompt probes (front).
# 2. Summarizes the per-episode conditioning video: timestamps, recorded robot joint/EE
#    trajectory, and downsampled RGB of evenly spaced frames (for offline inspection).
# 3. Policy hypothesis: the conditioning video shows the route to imitate in this episode;
#    reproduce it by replaying the video's recorded joint trajectory at its own timing.
#    (The live joints are first servoed to the video's first recorded joint state.)
# Nothing here uses hidden state: video frames are part of the live episode input.

report = {"version": "v00", "errors": []}


def note_error(where, exc):
    report["errors"].append(where + ": " + str(exc)[:300])


def r3(values):
    return [round(float(v), 4) for v in values]


def downsample_rgb(rgb, stride):
    try:
        return rgb[::stride, ::stride].tolist()
    except Exception as exc:
        note_error("downsample slice", exc)
    rows = []
    height = len(rgb)
    width = len(rgb[0])
    for y in range(0, height, stride):
        row = []
        for x in range(0, width, stride):
            px = rgb[y][x]
            row.append([int(px[0]), int(px[1]), int(px[2])])
        rows.append(row)
    return rows


context = get_task_context()
report["language"] = context.language
report["task_name"] = context.task_name
try:
    report["context_metadata"] = dict(context.metadata)
except Exception as exc:
    note_error("context metadata", exc)
try:
    report["task_metadata"] = dict(robomme_plus.get_task_metadata())
except Exception as exc:
    note_error("task metadata", exc)
try:
    report["controller_metadata"] = dict(robomme_plus.get_controller_metadata())
except Exception as exc:
    note_error("controller metadata", exc)

max_steps = 1000
try:
    max_steps = int(report["task_metadata"]["max_steps"])
except Exception as exc:
    note_error("max_steps", exc)
report["max_steps_used"] = max_steps

state = get_robot_state()
base_frame = state.base_frame
live_q0 = [float(v) for v in state.joint_positions["primary"]]
report["live_initial"] = {
    "joints": r3(live_q0),
    "ee": r3(state.end_effector_poses["primary"].position),
    "ee_quat": r3(state.end_effector_poses["primary"].quaternion_wxyz),
    "gripper": float(state.gripper_positions["primary"]),
    "timestamp_s": float(state.timestamp_s),
    "embodiment": state.embodiment,
    "base_frame": base_frame,
}

live = get_observation()
cams = {}
for name in live.cameras:
    cam = live.cameras[name]
    info = {"shape": [int(v) for v in cam.rgb.shape]}
    try:
        info["intrinsics"] = [r3(row) for row in cam.intrinsics]
        info["pose_pos"] = r3(cam.camera_pose.position)
        info["pose_quat"] = r3(cam.camera_pose.quaternion_wxyz)
    except Exception as exc:
        note_error("camera calib " + name, exc)
    cams[name] = info
report["cameras"] = cams
try:
    report["live_front_rgb_ds4"] = downsample_rgb(live.cameras["front"].rgb, 4)
except Exception as exc:
    note_error("live front rgb", exc)

# --- SAM3 probes on the live front view, with projected geometry ---
probes = []
for prompt in ("stick", "pole", "peg", "cylinder", "obstacle", "block", "cube", "bar",
               "target", "wall", "gate", "barrier"):
    entry = {"prompt": prompt, "ok": False, "cands": []}
    try:
        seg = segment_text("front", prompt)
        entry["ok"] = bool(seg.ok)
        if seg.ok:
            for cand in seg.segmentations[:5]:
                c = {"score": round(float(cand.score), 3)}
                if cand.box_xyxy is not None:
                    c["box"] = [round(float(v), 1) for v in cand.box_xyxy]
                try:
                    pc = mask_to_point_cloud(cand, "front", target_frame=base_frame)
                    c["n"] = len(pc.points)
                    if len(pc.points) >= 10:
                        xs = sorted(float(p[0]) for p in pc.points)
                        ys = sorted(float(p[1]) for p in pc.points)
                        zs = sorted(float(p[2]) for p in pc.points)
                        m = len(xs)
                        c["x"] = [round(xs[m // 20], 3), round(xs[m // 2], 3), round(xs[(19 * m) // 20], 3)]
                        c["y"] = [round(ys[m // 20], 3), round(ys[m // 2], 3), round(ys[(19 * m) // 20], 3)]
                        c["z"] = [round(zs[m // 20], 3), round(zs[m // 2], 3), round(zs[(19 * m) // 20], 3)]
                except Exception as exc:
                    c["pc_error"] = str(exc)[:120]
                entry["cands"].append(c)
    except Exception as exc:
        entry["error"] = str(exc)[:200]
    probes.append(entry)
report["probes"] = probes

# --- Conditioning video summary ---
video = []
try:
    video = robomme_plus.get_conditioning_video()
except Exception as exc:
    note_error("conditioning video", exc)
n_video = len(video)
report["video_len"] = n_video
video_traj = []
video_summary = []
for index in range(n_video):
    frame = video[index]
    try:
        rs = frame.robot_state
        q = [float(v) for v in rs.joint_positions["primary"]]
        video_traj.append((float(frame.timestamp_s), q))
        video_summary.append({
            "i": index,
            "t": round(float(frame.timestamp_s), 3),
            "q": r3(q),
            "ee": r3(rs.end_effector_poses["primary"].position),
            "g": round(float(rs.gripper_positions["primary"]), 3),
        })
    except Exception as exc:
        note_error("video frame %d" % index, exc)
report["video_traj"] = video_summary

frames_rgb = []
if n_video > 0:
    picks = []
    count = 12
    for k in range(count):
        picks.append(int(round(k * (n_video - 1) / float(count - 1))))
    for index in picks:
        try:
            frames_rgb.append({
                "i": index,
                "front": downsample_rgb(video[index].cameras["front"].rgb, 4),
            })
        except Exception as exc:
            note_error("video rgb %d" % index, exc)
    try:
        report["video_first_pose_front"] = r3(video[0].cameras["front"].camera_pose.position)
    except Exception as exc:
        note_error("video cam pose", exc)
report["video_rgb"] = frames_rgb

# --- Policy: replay the conditioning video's joint trajectory ---
exec_log = {"moved_to_start": None, "ticks": 0, "terminated": False, "truncated": False}
ended = False


def hold_command(q):
    return RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=list(q),
        gripper_position=float(get_robot_state().gripper_positions["primary"]),
        embodiment=state.embodiment)})


if len(video_traj) >= 2:
    first_q = video_traj[0][1]
    gap = max(abs(a - b) for a, b in zip(first_q, live_q0))
    exec_log["start_gap_rad"] = round(gap, 4)
    if gap > 0.02:
        moved = move_to_joints(first_q, tolerance=0.02, max_steps=150)
        exec_log["moved_to_start"] = {"ok": bool(moved.ok), "status": str(moved.status),
                                      "steps": int(moved.steps_executed)}
        try:
            if moved.terminated or moved.truncated:
                ended = True
        except Exception:
            pass
    budget = max_steps - 20 - int(get_robot_state().timestamp_s * 20.0 + 0.5)
    gripper_now = float(get_robot_state().gripper_positions["primary"])
    ticks = 0
    for k in range(1, len(video_traj)):
        if ended or ticks >= budget:
            break
        t_prev, q_prev = video_traj[k - 1]
        t_next, q_next = video_traj[k]
        sub = int(round((t_next - t_prev) / 0.05))
        if sub < 1:
            sub = 1
        for j in range(1, sub + 1):
            if ticks >= budget:
                break
            alpha = j / float(sub)
            target = [a + alpha * (b - a) for a, b in zip(q_prev, q_next)]
            reply = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=target, gripper_position=gripper_now,
                embodiment=state.embodiment)}))
            ticks += 1
            if reply.terminated or reply.truncated:
                exec_log["terminated"] = bool(reply.terminated)
                exec_log["truncated"] = bool(reply.truncated)
                exec_log["ended_at_video_index"] = k
                ended = True
                break
    exec_log["ticks"] = ticks
    exec_log["budget"] = budget
    if not ended:
        try:
            final_q = video_traj[-1][1]
            for settle in range(10):
                reply = step(RobotAction(arms={"primary": ArmCommand(
                    mode="joint_position", target=final_q, gripper_position=gripper_now,
                    embodiment=state.embodiment)}))
                if reply.terminated or reply.truncated:
                    exec_log["terminated"] = bool(reply.terminated)
                    exec_log["truncated"] = bool(reply.truncated)
                    ended = True
                    break
        except Exception as exc:
            note_error("settle", exc)
else:
    # no video: one benign hold tick for keyframes
    try:
        step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=live_q0,
            gripper_position=float(state.gripper_positions["primary"]),
            embodiment=state.embodiment)}))
    except Exception as exc:
        note_error("hold", exc)

if not ended:
    try:
        final_state = get_robot_state()
        exec_log["final_ee"] = r3(final_state.end_effector_poses["primary"].position)
        exec_log["final_t"] = float(final_state.timestamp_s)
    except Exception as exc:
        note_error("final state", exc)
report["exec"] = exec_log

result = report
