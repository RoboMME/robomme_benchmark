# InsertPeg v00 -- conditioning-video replay with live layout gate.
#
# Strategy (all episode-dependent data derived at runtime):
#   1. Read the goal and the conditioning video (robomme_plus.get_conditioning_video()).
#      Each video frame carries the recorded robot state (7 joints + normalized gripper),
#      so the video's own grasp ("same peg, same end") and insertion ("same side") are the
#      recorded arm motion itself.
#   2. Gate: compare the live front depth with video frame 0's front depth (depth is
#      lighting-invariant). Released recordings show the execution starts from the video's
#      initial layout; the gate records whether that holds in this episode.
#   3. Replay: stream the recorded joint targets one control tick at a time (target = the
#      next recorded joint state), with the gripper command derived from the recorded
#      gripper trace (closing onset -> 0, opening onset -> 1). Before every gripper event and
#      at the end, servo until the arm converges on the recorded joints so the grasp happens
#      at the recorded hand pose.
#   4. Stop as soon as a step reports terminated/truncated.
import math

report = {"version": "v00"}
ctx = get_task_context()
goal = ctx.language
report["goal"] = goal
meta = robomme_plus.get_task_metadata()
max_steps = 0
try:
    max_steps = int(meta["max_steps"])
except Exception:
    max_steps = 0
report["max_steps"] = max_steps
try:
    report["meta_keys"] = sorted([str(key) for key in meta])
    report["video_sampling"] = str(meta["conditioning_video_sampling"])
except Exception:
    report["video_sampling"] = "unavailable"
try:
    controller = robomme_plus.get_controller_metadata()
    report["controller"] = str(dict(controller))[:1500]
except Exception as error:
    report["controller"] = "error " + str(error)

state0 = get_robot_state()
embodiment = state0.embodiment
live_q0 = [float(value) for value in state0.joint_positions["primary"]]
report["live_q0"] = [round(value, 4) for value in live_q0]
report["live_g0"] = float(state0.gripper_positions["primary"])

video = robomme_plus.get_conditioning_video()
frame_count = len(video)
report["video_frames"] = frame_count
if frame_count < 2:
    raise ValueError("no conditioning video; replay strategy has nothing to follow")

joints = []
grips = []
times = []
ee_positions = []
for frame in video:
    rs = frame.robot_state
    joints.append([float(value) for value in rs.joint_positions["primary"]])
    grips.append(float(rs.gripper_positions["primary"]))
    times.append(float(frame.timestamp_s))
    ee_positions.append([float(value) for value in rs.end_effector_poses["primary"].position])

dts = [times[index + 1] - times[index] for index in range(frame_count - 1)]
report["video_dt_min_max"] = [round(min(dts), 4), round(max(dts), 4)]
report["video_duration_s"] = round(times[-1] - times[0], 3)
report["video_grip_min_max"] = [round(min(grips), 4), round(max(grips), 4)]

# ---- layout gate: live front depth vs video frame-0 front depth ----
live_obs = get_observation()
gate = {"checked": False}
try:
    live_depth = live_obs.cameras["front"].depth_m
    video_depth = video[0].cameras["front"].depth_m
    rows = len(live_depth)
    cols = len(live_depth[0])
    sampled = 0
    changed = 0
    total_abs = 0.0
    for row in range(0, rows, 4):
        for col in range(0, cols, 4):
            a = float(live_depth[row][col])
            b = float(video_depth[row][col])
            if a > 0.0 and b > 0.0 and a < 5.0 and b < 5.0:
                sampled += 1
                diff = abs(a - b)
                total_abs += diff
                if diff > 0.01:
                    changed += 1
    gate["checked"] = True
    gate["sampled"] = sampled
    gate["changed_px"] = changed
    gate["changed_frac"] = round(changed / max(1, sampled), 5)
    gate["mean_abs_m"] = round(total_abs / max(1, sampled), 5)
    live_cam = [float(value) for value in live_obs.cameras["front"].camera_pose.position]
    video_cam = [float(value) for value in video[0].cameras["front"].camera_pose.position]
    gate["camera_shift_m"] = round(math.sqrt(sum((live_cam[k] - video_cam[k]) ** 2 for k in range(3))), 5)
except Exception as error:
    gate["error"] = str(error)
q0_err = max(abs(live_q0[k] - joints[0][k]) for k in range(7))
gate["q0_max_err_rad"] = round(q0_err, 5)
gate["same_layout"] = bool(gate["checked"]) and gate.get("changed_frac", 1.0) < 0.02
report["gate"] = gate

# ---- gripper command per frame from the recorded gripper trace ----
grip_open_level = max(grips)
closed_cmd = []
events = []
closed = grips[0] < 0.5 * grip_open_level
for index in range(frame_count):
    if index + 1 < frame_count:
        delta = grips[index + 1] - grips[index]
        if (not closed) and delta < -0.01:
            closed = True
            events.append(["close", index])
        elif closed and delta > 0.01:
            closed = False
            events.append(["open", index])
    closed_cmd.append(closed)
report["gripper_events"] = events
for event in events:
    frame_index = event[1]
    report["video_ee_at_" + event[0] + "_" + str(frame_index)] = [round(value, 4) for value in ee_positions[frame_index]]
report["video_ee_final"] = [round(value, 4) for value in ee_positions[-1]]

# ---- expand frames to control ticks (videos > 512 frames are uniformly subsampled) ----
control_dt = 0.05
try:
    control_dt = 1.0 / float(meta["control_hz"])
except Exception:
    control_dt = 0.05
event_set = set([event[1] for event in events])
tick_joints = [joints[0]]
tick_closed = [closed_cmd[0]]
tick_frame = [0]
tick_events = set()
for index in range(1, frame_count):
    span = max(1, int(round((times[index] - times[index - 1]) / control_dt)))
    for sub in range(1, span + 1):
        alpha = float(sub) / float(span)
        tick_joints.append([joints[index - 1][k] + alpha * (joints[index][k] - joints[index - 1][k]) for k in range(7)])
        tick_closed.append(closed_cmd[index - 1] if sub < span else closed_cmd[index])
        tick_frame.append(index)
    if index in event_set:
        tick_events.add(len(tick_joints) - 1)
if 0 in event_set:
    tick_events.add(0)
tick_count = len(tick_joints)
report["replay_tick_targets"] = tick_count

# ---- replay ----
episode_over = [False]
ticks = [0]
last_reply = [None]


def send(target, grip_closed):
    if episode_over[0]:
        return None
    reply = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=[float(value) for value in target],
        gripper_position=(0.0 if grip_closed else 1.0), embodiment=embodiment)}))
    ticks[0] += 1
    last_reply[0] = reply
    try:
        if bool(reply.terminated) or bool(reply.truncated):
            episode_over[0] = True
    except Exception:
        episode_over[0] = False
    return reply


def joint_error(target):
    current = [float(value) for value in get_robot_state().joint_positions["primary"]]
    return max(abs(current[k] - target[k]) for k in range(7))


def converge(target, grip_closed, tolerance, max_ticks):
    used = 0
    error = joint_error(target)
    while (not episode_over[0]) and error > tolerance and used < max_ticks:
        send(target, grip_closed)
        used += 1
        if not episode_over[0]:
            error = joint_error(target)
    return [used, round(error, 5)]


slack = max_steps - tick_count if max_steps > 0 else 200
hold_budget = max(0, min(25, (slack - 20) // max(1, len(events) + 2)))
report["hold_budget_per_event"] = hold_budget


def ee_error_to(frame_index):
    try:
        current = [float(value) for value in get_robot_state().end_effector_poses["primary"].position]
        return round(math.sqrt(sum((current[k] - ee_positions[frame_index][k]) ** 2 for k in range(3))), 5)
    except Exception:
        return -1.0


convergence = []
tracking = []
if q0_err > 0.01:
    convergence.append(["start"] + converge(tick_joints[0], tick_closed[0], 0.005, 60))

for index in range(1, tick_count):
    if episode_over[0]:
        break
    if (index - 1) in tick_events and hold_budget > 0:
        # the arm should sit at the recorded pose before the gripper changes state
        previous_closed = tick_closed[index - 2] if index >= 2 else tick_closed[0]
        outcome = converge(tick_joints[index - 1], previous_closed, 0.004, hold_budget)
        convergence.append([index - 1] + outcome + [ee_error_to(tick_frame[index - 1])])
    send(tick_joints[index], tick_closed[index - 1])
    if index % 25 == 0 and not episode_over[0]:
        tracking.append([index, round(joint_error(tick_joints[index]), 4), ee_error_to(tick_frame[index])])

report["tracking_every25"] = tracking
report["replay_ticks"] = ticks[0]
report["episode_over_during_replay"] = episode_over[0]
if not episode_over[0]:
    final_hold = max(0, min(40, max_steps - ticks[0] - 2)) if max_steps > 0 else 40
    convergence.append(["final"] + converge(tick_joints[-1], tick_closed[-1], 0.002, final_hold))
    # a few more ticks of holding in case the insertion predicate needs settling
    extra = 0
    while (not episode_over[0]) and extra < 10 and (max_steps <= 0 or ticks[0] < max_steps - 2):
        send(tick_joints[-1], tick_closed[-1])
        extra += 1
report["convergence"] = convergence
report["total_ticks"] = ticks[0]
report["episode_over"] = episode_over[0]
try:
    reply = last_reply[0]
    report["last_terminated"] = bool(reply.terminated)
    report["last_truncated"] = bool(reply.truncated)
except Exception:
    report["last_terminated"] = "unknown"
if not episode_over[0]:
    try:
        final_state = get_robot_state()
        final_ee = [float(value) for value in final_state.end_effector_poses["primary"].position]
        report["live_ee_final"] = [round(value, 4) for value in final_ee]
        report["ee_final_err_m"] = round(math.sqrt(sum((final_ee[k] - ee_positions[-1][k]) ** 2 for k in range(3))), 5)
        report["live_grip_final"] = float(final_state.gripper_positions["primary"])
    except Exception as error:
        report["final_state_error"] = str(error)

result = report
