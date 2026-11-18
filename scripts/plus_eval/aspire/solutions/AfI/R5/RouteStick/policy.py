# RouteStick v00 -- exploration folded into a first attempt.
# 1. Records goal, task/controller metadata, conditioning-video summary (timestamps, recorded
#    robot joint/EE path, a few small front-camera thumbnails), live-vs-video scene difference,
#    and SAM3 probe geometry of the live front image.
# 2. Attempts the task by re-executing the robot joint trajectory recorded in THIS episode's
#    conditioning video (derived at runtime; nothing copied from any other episode), at the
#    video's own timing (time-compressed only if it would not fit the task's max_steps).
import math

result = {"version": "v00_initial", "stage": "start", "errors": []}
flags = {"ended": False, "end_tick": None, "end_source": None}
HEX = "0123456789abcdef"


def note_error(where, err):
    if len(result["errors"]) < 30:
        result["errors"].append(where + ": " + str(err)[:300])


def as_list(arr):
    return [float(v) for v in arr]


def maxabs_diff(a, b):
    best = 0.0
    for i in range(min(len(a), len(b))):
        d = abs(float(a[i]) - float(b[i]))
        if d > best:
            best = d
    return best


def check_end(reply, source, tick):
    hit = False
    try:
        if bool(reply.terminated):
            hit = True
    except Exception:
        pass
    try:
        if bool(reply.truncated):
            hit = True
    except Exception:
        pass
    if hit and not flags["ended"]:
        flags["ended"] = True
        flags["end_tick"] = tick
        flags["end_source"] = source
    return hit


def thumb_hex(rgb, size):
    h = len(rgb)
    w = len(rgb[0])
    out = []
    for yi in range(size):
        row = rgb[int((yi + 0.5) * h / size)]
        for xi in range(size):
            px = row[int((xi + 0.5) * w / size)]
            for c in range(3):
                v = int(px[c])
                out.append(HEX[v // 16] + HEX[v % 16])
    return "".join(out)


def grid_diff(rgb_a, rgb_b, size):
    h = len(rgb_a)
    w = len(rgb_a[0])
    total = 0.0
    big = 0
    count = 0
    for yi in range(size):
        y = int((yi + 0.5) * h / size)
        ra = rgb_a[y]
        rb = rgb_b[y]
        for xi in range(size):
            x = int((xi + 0.5) * w / size)
            pa = ra[x]
            pb = rb[x]
            d = 0.0
            for c in range(3):
                d += abs(float(pa[c]) - float(pb[c]))
            d = d / 3.0
            total += d
            count += 1
            if d > 30.0:
                big += 1
    return {"mean_abs": round(total / max(1, count), 2), "frac_gt30": round(big / max(1, count), 4)}


def pose_summary(pose):
    try:
        return {"p": [round(float(v), 4) for v in pose.position],
                "q": [round(float(v), 4) for v in pose.quaternion_wxyz]}
    except Exception as err:
        return {"err": str(err)[:100]}


# ---------------------------------------------------------------- context / metadata
context = get_task_context()
result["goal"] = str(context.language)
try:
    result["task_name"] = str(context.task_name)
except Exception as err:
    note_error("task_name", err)

max_steps = 600
control_hz = 20.0
try:
    meta = robomme_plus.get_task_metadata()
    md = {}
    for key in meta:
        md[str(key)] = str(meta[key])[:400]
    result["task_metadata"] = md
    max_steps = int(meta["max_steps"])
    try:
        control_hz = float(meta["control_hz"])
    except Exception as err:
        note_error("control_hz", err)
except Exception as err:
    note_error("task_metadata", err)
result["max_steps"] = max_steps
result["control_hz"] = control_hz

try:
    cmeta = robomme_plus.get_controller_metadata()
    cm = {}
    for key in cmeta:
        cm[str(key)] = str(cmeta[key])[:300]
    result["controller_metadata"] = cm
except Exception as err:
    note_error("controller_metadata", err)

state0 = get_robot_state()
base_frame = state0.base_frame
live_q0 = as_list(state0.joint_positions["primary"])
result["live_start"] = {
    "q": [round(v, 4) for v in live_q0],
    "ee": pose_summary(state0.end_effector_poses["primary"]),
    "gripper": float(state0.gripper_positions["primary"]),
    "t": float(state0.timestamp_s),
    "embodiment": str(state0.embodiment),
    "base_frame": str(base_frame),
}

obs0 = get_observation()
live_front = obs0.cameras["front"]
result["live_cameras"] = [str(name) for name in obs0.cameras]
result["live_front_pose"] = pose_summary(live_front.camera_pose)
try:
    result["live_front_thumb40"] = thumb_hex(live_front.rgb, 40)
except Exception as err:
    note_error("live_thumb", err)

# ---------------------------------------------------------------- conditioning video
times = []
joints = []
ee_path = []
grip_path = []
video = []
try:
    video = robomme_plus.get_conditioning_video()
except Exception as err:
    note_error("get_conditioning_video", err)
n_video = len(video)
result["video_len"] = n_video
for i in range(n_video):
    frame = video[i]
    try:
        times.append(float(frame.timestamp_s))
        rs = frame.robot_state
        joints.append(as_list(rs.joint_positions["primary"]))
        p = rs.end_effector_poses["primary"].position
        ee_path.append([round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)])
        grip_path.append(round(float(rs.gripper_positions["primary"]), 3))
    except Exception as err:
        note_error("video_frame_" + str(i), err)

if n_video > 0 and len(times) == n_video:
    result["video_t_first"] = times[0]
    result["video_t_last"] = times[-1]
    dts = [times[i + 1] - times[i] for i in range(n_video - 1)]
    if dts:
        result["video_dt_min"] = round(min(dts), 4)
        result["video_dt_max"] = round(max(dts), 4)
    result["video_ee_path"] = ee_path
    result["video_grip_first_last"] = [grip_path[0], grip_path[-1]]
    result["video_q_first"] = [round(v, 4) for v in joints[0]]
    result["video_q_last"] = [round(v, 4) for v in joints[-1]]
    result["live_vs_video_q_first"] = round(maxabs_diff(live_q0, joints[0]), 4)
    result["live_vs_video_q_last"] = round(maxabs_diff(live_q0, joints[-1]), 4)
    # per-frame joint motion (max abs change to next frame), coarse
    motion = []
    for i in range(n_video - 1):
        motion.append(round(maxabs_diff(joints[i], joints[i + 1]), 4))
    result["video_joint_motion"] = motion
    try:
        vf0 = video[0].cameras["front"]
        vfl = video[n_video - 1].cameras["front"]
        result["video_front_pose_first"] = pose_summary(vf0.camera_pose)
        result["video_front_pose_last"] = pose_summary(vfl.camera_pose)
        result["diff_live_vs_video_first"] = grid_diff(live_front.rgb, vf0.rgb, 48)
        result["diff_live_vs_video_last"] = grid_diff(live_front.rgb, vfl.rgb, 48)
        result["diff_video_first_vs_last"] = grid_diff(vf0.rgb, vfl.rgb, 48)
    except Exception as err:
        note_error("video_diff", err)
    thumbs = []
    n_thumbs = min(10, n_video)
    for k in range(n_thumbs):
        idx = int(round(k * (n_video - 1) / max(1, n_thumbs - 1)))
        try:
            thumbs.append({"i": idx, "t": round(times[idx], 3),
                           "front40": thumb_hex(video[idx].cameras["front"].rgb, 40)})
        except Exception as err:
            note_error("thumb_" + str(idx), err)
    result["video_thumbs"] = thumbs

# ---------------------------------------------------------------- live perception probes
probes = []
for prompt in ("stick", "pole", "cylinder", "obstacle", "block", "target"):
    entry = {"prompt": prompt, "ok": False, "cands": []}
    try:
        seg = segment_text("front", prompt)
        entry["ok"] = bool(seg.ok)
        if seg.ok:
            entry["n"] = len(seg.segmentations)
            rank = 0
            for cand in seg.segmentations[:4]:
                c = {"score": round(float(cand.score), 3)}
                try:
                    c["box"] = [round(float(v), 1) for v in cand.box_xyxy]
                except Exception:
                    pass
                if rank < 3:
                    try:
                        pc = mask_to_point_cloud(cand, "front", target_frame=base_frame)
                        geo = estimate_geometry(pc)
                        c["center"] = [round(float(v), 4) for v in geo.pose.position]
                        c["extents"] = [round(float(v), 4) for v in geo.extents]
                        c["npts"] = int(geo.point_count)
                        zs = [float(pt[2]) for pt in pc.points]
                        if zs:
                            c["zmin"] = round(min(zs), 4)
                            c["zmax"] = round(max(zs), 4)
                    except Exception as err:
                        c["geo_err"] = str(err)[:120]
                entry["cands"].append(c)
                rank += 1
    except Exception as err:
        entry["err"] = str(err)[:200]
    probes.append(entry)
result["probes"] = probes

# ---------------------------------------------------------------- replay attempt
result["stage"] = "replay_setup"
replay = {"done": False}
if n_video >= 2 and len(joints) == n_video:
    thr = 0.003
    i_start = 0
    for i in range(n_video):
        if maxabs_diff(joints[i], joints[0]) > thr:
            i_start = max(0, i - 1)
            break
    i_end = n_video - 1
    for i in range(n_video - 1, -1, -1):
        if maxabs_diff(joints[i], joints[n_video - 1]) > thr:
            i_end = min(n_video - 1, i + 1)
            break
    seg_t = times[i_start:i_end + 1]
    seg_q = joints[i_start:i_end + 1]
    dur_ticks = (seg_t[-1] - seg_t[0]) * control_hz
    budget = max_steps - 160
    speed = 1.0
    if budget > 50 and dur_ticks > budget:
        speed = dur_ticks / budget
    n_out = int(dur_ticks / speed) + 1
    replay.update({"i_start": i_start, "i_end": i_end, "dur_ticks": round(dur_ticks, 1),
                   "speed": round(speed, 3), "n_out": n_out})
    result["replay"] = replay

    # approach the first recorded configuration
    result["stage"] = "approach"
    try:
        appr = move_to_joints(seg_q[0], tolerance=0.01, max_steps=100)
        replay["approach_ok"] = bool(appr.ok)
        replay["approach_steps"] = int(appr.steps_executed)
        check_end(appr, "approach", -1)
    except Exception as err:
        note_error("approach", err)

    result["stage"] = "replay"
    track = []
    tick = 0
    j = 0
    max_err = 0.0
    if not flags["ended"]:
        st = get_robot_state()
        emb = st.embodiment
        grip = float(st.gripper_positions["primary"])
        for k in range(1, n_out + 1):
            t = seg_t[0] + k * speed / control_hz
            while j < len(seg_t) - 2 and seg_t[j + 1] < t:
                j += 1
            span = seg_t[j + 1] - seg_t[j]
            alpha = 1.0
            if span > 1e-9:
                alpha = (t - seg_t[j]) / span
            alpha = max(0.0, min(1.0, alpha))
            qa = seg_q[j]
            qb = seg_q[j + 1]
            target = [qa[d] + alpha * (qb[d] - qa[d]) for d in range(len(qa))]
            try:
                reply = step(RobotAction(arms={"primary": ArmCommand(
                    mode="joint_position", target=target,
                    gripper_position=grip, embodiment=emb)}))
            except Exception as err:
                note_error("step_" + str(k), err)
                break
            tick = k
            if check_end(reply, "replay", k):
                break
            if k % 10 == 0:
                try:
                    rs = reply.observation.robot_state
                    qm = as_list(rs.joint_positions["primary"])
                    e = maxabs_diff(qm, target)
                    if e > max_err:
                        max_err = e
                    p = rs.end_effector_poses["primary"].position
                    track.append([k, round(float(p[0]), 4), round(float(p[1]), 4),
                                  round(float(p[2]), 4), round(e, 4)])
                except Exception as err:
                    note_error("track_" + str(k), err)
        replay["ticks"] = tick
        replay["max_track_err"] = round(max_err, 4)
        replay["track"] = track
        # hold final configuration to converge
        if not flags["ended"]:
            result["stage"] = "hold"
            last = seg_q[-1]
            for h in range(30):
                try:
                    reply = step(RobotAction(arms={"primary": ArmCommand(
                        mode="joint_position", target=last,
                        gripper_position=grip, embodiment=emb)}))
                except Exception as err:
                    note_error("hold", err)
                    break
                if check_end(reply, "hold", h):
                    break
            replay["hold_done"] = True
    replay["done"] = True
else:
    result["replay"] = {"skipped": "no usable conditioning video"}

result["ended"] = flags
if not flags["ended"]:
    try:
        stf = get_robot_state()
        result["final"] = {"t": float(stf.timestamp_s),
                           "ee": pose_summary(stf.end_effector_poses["primary"])}
    except Exception as err:
        note_error("final_state", err)
result["stage"] = "done"
