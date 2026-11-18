# v01_final_dwell.py (from v00_initial.py) -- RouteStick (robomme_plus task 8), ASPIRE Stage 0/1, AfI, R=1.
#
# One program is both the scene snapshot and the attempt (single start for this task).
# Phase A  records language, metadata, cameras, robot state.
# Phase B  summarizes the conditioning video: per-frame timestamps, recorded joints and
#          end-effector path (the episode's own public input), cheap RGB statistics.
# Phase C  SAM3 prompt probes on both live cameras (+ geometry of top candidates).
# Phase D  decides the execution plan from the runtime language and from where the live
#          robot sits relative to the video's recorded start/end configuration.
# Phase E  executes: reproduce the demonstrated stick route by tracking the video's
#          recorded joint path with one-tick joint-position commands, stopping on any
#          terminal status.
# No phase may raise: every phase is wrapped and errors are appended to result["errors"].
import math

report = {
    "version": "v01_final_dwell",
    "errors": [],
    "phases_done": [],
    "notes": [],
}
result = report

EMB = "robomme_plus"
BASE = "robot_base"
HZ = 20.0
MAX_STEPS = 0
S = {"over": False, "ticks": 0}
FINAL_HOLD_MIN = 40   # ticks reserved for the final dwell (v00 evidence: registration ~3-10 ticks after arrival)
FINAL_HOLD_MAX = 80   # upper bound on the final dwell; stops earlier on a terminal status


def note(text):
    print("[v01] " + str(text))
    report["notes"].append(str(text))


def err(phase, exc):
    text = phase + ": " + str(exc)
    print("[v01][ERR] " + text)
    report["errors"].append(text)


def fl(values):
    out = []
    for value in values:
        out.append(float(value))
    return out


def r4(values):
    out = []
    for value in values:
        out.append(round(float(value), 4))
    return out


def max_abs_diff(a, b):
    best = 0.0
    for i in range(min(len(a), len(b))):
        d = abs(float(a[i]) - float(b[i]))
        if d > best:
            best = d
    return best


def dist3(a, b):
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2
                     + (float(a[2]) - float(b[2])) ** 2)


def reply_flags(reply):
    info = {}
    try:
        info["terminated"] = bool(reply.terminated)
    except Exception:
        pass
    try:
        info["truncated"] = bool(reply.truncated)
    except Exception:
        pass
    try:
        info["done"] = bool(reply.done)
    except Exception:
        pass
    try:
        info["terminal"] = bool(reply.terminal)
    except Exception:
        pass
    try:
        info["ok"] = bool(reply.ok)
    except Exception:
        pass
    try:
        info["status"] = str(reply.status)
    except Exception:
        pass
    try:
        if reply.error is not None:
            info["error"] = str(reply.error)[:300]
    except Exception:
        pass
    return info


def is_terminal(info):
    return bool(info.get("terminated", False) or info.get("truncated", False)
                or info.get("done", False) or info.get("terminal", False))


def grid_rgb(rgb, stride):
    vals = []
    try:
        h = int(rgb.shape[0])
        w = int(rgb.shape[1])
    except Exception:
        h = len(rgb)
        w = len(rgb[0])
    y = stride // 2
    while y < h:
        row = rgb[y]
        x = stride // 2
        while x < w:
            px = row[x]
            vals.append((int(px[0]), int(px[1]), int(px[2])))
            x += stride
        y += stride
    return vals


def grid_diff(ga, gb):
    if len(ga) == 0 or len(ga) != len(gb):
        return None
    total = 0.0
    changed = 0
    for i in range(len(ga)):
        d = abs(ga[i][0] - gb[i][0]) + abs(ga[i][1] - gb[i][1]) + abs(ga[i][2] - gb[i][2])
        total += d
        if d > 60:
            changed += 1
    return {"mean_abs_sum": round(total / len(ga), 2), "changed_frac": round(changed / float(len(ga)), 4)}


# ---------------------------------------------------------------- Phase A: context
goal = ""
try:
    context = get_task_context()
    goal = str(context.language)
    report["task_language"] = goal
    try:
        report["task_name"] = str(context.task_name)
        report["task_id"] = str(context.task_id)
        report["suite"] = str(context.suite)
    except Exception as exc:
        err("A.context_fields", exc)
    print("[v01] goal: " + goal)
except Exception as exc:
    err("A.context", exc)

try:
    md = robomme_plus.get_task_metadata()
    mdd = {}
    for key in md:
        mdd[str(key)] = str(md[key])[:400]
    report["task_metadata"] = mdd
    try:
        MAX_STEPS = int(md["max_steps"])
    except Exception as exc:
        err("A.max_steps", exc)
    for key in md:
        lk = str(key).lower()
        if ("hz" in lk or "freq" in lk) and "video" not in lk:
            try:
                value = float(md[key])
                if value > 1.0 and value < 1000.0:
                    HZ = value
            except Exception:
                pass
except Exception as exc:
    err("A.task_metadata", exc)
report["max_steps"] = MAX_STEPS
report["control_hz"] = HZ

try:
    cm = robomme_plus.get_controller_metadata()
    cmd = {}
    for key in cm:
        cmd[str(key)] = str(cm[key])[:400]
    report["controller_metadata"] = cmd
except Exception as exc:
    err("A.controller_metadata", exc)

live_q = None
live_g = 1.0
live_ee = None
try:
    state = get_robot_state()
    try:
        BASE = str(state.base_frame)
    except Exception:
        pass
    try:
        EMB = str(state.embodiment)
    except Exception:
        pass
    live_q = fl(state.joint_positions["primary"])
    try:
        live_g = float(state.gripper_positions["primary"])
    except Exception:
        live_g = 1.0
    live_ee = state.end_effector_poses["primary"]
    report["robot_state"] = {
        "base_frame": BASE,
        "embodiment": EMB,
        "joints": r4(live_q),
        "gripper": round(live_g, 4),
        "ee_position": r4(live_ee.position),
        "ee_quat_wxyz": r4(live_ee.quaternion_wxyz),
        "timestamp_s": float(state.timestamp_s),
    }
except Exception as exc:
    err("A.robot_state", exc)

live_obs = None
live_front_grid = None
try:
    live_obs = get_observation()
    cams = {}
    for name in live_obs.cameras:
        cam = live_obs.cameras[name]
        entry = {}
        try:
            entry["rgb_shape"] = [int(v) for v in cam.rgb.shape]
            entry["depth_shape"] = [int(v) for v in cam.depth_m.shape]
        except Exception as exc:
            entry["shape_error"] = str(exc)
        try:
            K = cam.intrinsics
            entry["intrinsics"] = [r4(K[0]), r4(K[1]), r4(K[2])]
        except Exception as exc:
            entry["intrinsics_error"] = str(exc)
        try:
            entry["pose_position"] = r4(cam.camera_pose.position)
            entry["pose_quat_wxyz"] = r4(cam.camera_pose.quaternion_wxyz)
            entry["pose_frame"] = str(cam.camera_pose.frame)
        except Exception as exc:
            entry["pose_error"] = str(exc)
        try:
            dep = cam.depth_m
            dvals = []
            yy = 8
            while yy < int(dep.shape[0]):
                xx = 8
                while xx < int(dep.shape[1]):
                    dvals.append(float(dep[yy][xx]))
                    xx += 16
                yy += 16
            dvals = sorted(dvals)
            entry["depth_q05_q50_q95"] = [round(dvals[len(dvals) // 20], 3), round(dvals[len(dvals) // 2], 3),
                                          round(dvals[(19 * len(dvals)) // 20], 3)]
        except Exception as exc:
            entry["depth_error"] = str(exc)
        cams[str(name)] = entry
    report["cameras"] = cams
    try:
        live_front_grid = grid_rgb(live_obs.cameras["front"].rgb, 8)
    except Exception as exc:
        err("A.front_grid", exc)
except Exception as exc:
    err("A.observation", exc)
report["phases_done"].append("A")

# ---------------------------------------------------------------- Phase A2: goal checklist
clauses = []
try:
    text = goal.replace(";", ",").replace(" then ", ",").replace(" and then ", ",").replace(". ", ",")
    for part in text.split(","):
        part = part.strip().strip(".")
        if len(part) > 0:
            clauses.append(part)
except Exception as exc:
    err("A2.clauses", exc)
lower_goal = goal.lower()
keywords = {}
for word in ("video", "same", "repeat", "follow", "imitate", "reverse", "opposite", "backward",
             "back", "return", "clockwise", "counterclockwise", "counter-clockwise", "left", "right",
             "around", "obstacle", "route", "path", "stick", "through", "between", "under", "over",
             "touch", "avoid", "without", "first", "finally", "once", "twice"):
    if word in lower_goal:
        keywords[word] = True
report["goal_clauses"] = clauses
report["goal_keywords"] = sorted(keywords)
reverse_mode = False
if ("reverse" in lower_goal) or ("opposite direction" in lower_goal) or ("backward" in lower_goal):
    reverse_mode = True
report["reverse_mode_from_language"] = reverse_mode
report["clause_to_phase"] = {}
for clause in clauses:
    report["clause_to_phase"][clause] = "E (replay of the demonstrated route; see plan)"
report["phases_done"].append("A2")

# ---------------------------------------------------------------- Phase B: conditioning video
vid_q = []
vid_t = []
vid_ee = []
vid_g = []
video_len = 0
try:
    video = robomme_plus.get_conditioning_video()
    frames = list(video)
    video_len = len(frames)
    report["video_len"] = video_len
    print("[v01] conditioning video frames: " + str(video_len))
    for idx in range(video_len):
        frame = frames[idx]
        try:
            rs = frame.robot_state
            vid_q.append(fl(rs.joint_positions["primary"]))
            vid_t.append(float(frame.timestamp_s))
            try:
                vid_ee.append(fl(rs.end_effector_poses["primary"].position))
            except Exception:
                vid_ee.append(None)
            try:
                vid_g.append(float(rs.gripper_positions["primary"]))
            except Exception:
                vid_g.append(None)
        except Exception as exc:
            if len(report["errors"]) < 40:
                err("B.frame" + str(idx), exc)
    vsum = {}
    if len(vid_q) > 0:
        vsum["t_first_last"] = [round(vid_t[0], 3), round(vid_t[-1], 3)]
        dts = []
        for i in range(1, len(vid_t)):
            dts.append(vid_t[i] - vid_t[i - 1])
        if len(dts) > 0:
            vsum["dt_min_max"] = [round(min(dts), 4), round(max(dts), 4)]
        ranges = []
        for j in range(len(vid_q[0])):
            col = [q[j] for q in vid_q]
            ranges.append(round(max(col) - min(col), 4))
        vsum["joint_range_per_joint"] = ranges
        vsum["q_first"] = r4(vid_q[0])
        vsum["q_last"] = r4(vid_q[-1])
        moving = []
        for i in range(1, len(vid_q)):
            moving.append(max_abs_diff(vid_q[i], vid_q[i - 1]) > 0.002)
        first_move = None
        last_move = None
        for i in range(len(moving)):
            if moving[i]:
                if first_move is None:
                    first_move = i
                last_move = i + 1
        vsum["first_motion_frame"] = first_move
        vsum["last_motion_frame"] = last_move
        if first_move is not None:
            vsum["motion_t"] = [round(vid_t[first_move], 3), round(vid_t[last_move], 3)]
        ees = [p for p in vid_ee if p is not None]
        if len(ees) > 0:
            vsum["ee_first"] = r4(ees[0])
            vsum["ee_last"] = r4(ees[-1])
            vsum["ee_min"] = [round(min(p[k] for p in ees), 4) for k in range(3)]
            vsum["ee_max"] = [round(max(p[k] for p in ees), 4) for k in range(3)]
            path_len = 0.0
            for i in range(1, len(ees)):
                path_len += dist3(ees[i], ees[i - 1])
            vsum["ee_path_len_m"] = round(path_len, 4)
            stride = max(1, len(vid_ee) // 64)
            samples = []
            for i in range(0, len(vid_ee), stride):
                if vid_ee[i] is not None:
                    samples.append([round(vid_t[i], 2)] + r4(vid_ee[i]))
            if vid_ee[-1] is not None:
                samples.append([round(vid_t[-1], 2)] + r4(vid_ee[-1]))
            vsum["ee_path_samples_t_xyz"] = samples
        gs = [g for g in vid_g if g is not None]
        if len(gs) > 0:
            vsum["gripper_min_max"] = [round(min(gs), 3), round(max(gs), 3)]
        try:
            ee0 = frames[0].robot_state.end_effector_poses["primary"]
            vsum["ee_quat_first"] = r4(ee0.quaternion_wxyz)
            eeN = frames[-1].robot_state.end_effector_poses["primary"]
            vsum["ee_quat_last"] = r4(eeN.quaternion_wxyz)
        except Exception as exc:
            err("B.quat", exc)
        if live_q is not None:
            vsum["live_vs_first_maxdiff_rad"] = round(max_abs_diff(live_q, vid_q[0]), 4)
            vsum["live_vs_last_maxdiff_rad"] = round(max_abs_diff(live_q, vid_q[-1]), 4)
            best_i = 0
            best_d = None
            for i in range(len(vid_q)):
                d = max_abs_diff(live_q, vid_q[i])
                if best_d is None or d < best_d:
                    best_d = d
                    best_i = i
            vsum["live_nearest_frame"] = [best_i, round(best_d, 4)]
    report["video_summary"] = vsum
    # cheap RGB statistics on a subset of frames (front camera)
    try:
        if video_len > 0:
            picks = []
            count = min(24, video_len)
            for k in range(count):
                picks.append(int(round(k * (video_len - 1) / float(max(1, count - 1)))))
            grids = {}
            for i in picks:
                try:
                    grids[i] = grid_rgb(frames[i].cameras["front"].rgb, 16)
                except Exception as exc:
                    err("B.grid" + str(i), exc)
            energy = []
            prev = None
            for i in picks:
                if i in grids:
                    if prev is not None:
                        energy.append([i, grid_diff(grids[prev], grids[i])])
                    prev = i
            report["video_front_motion_energy"] = energy
            if live_obs is not None:
                live16 = grid_rgb(live_obs.cameras["front"].rgb, 16)
                if picks[0] in grids:
                    report["live_vs_video_first_front"] = grid_diff(live16, grids[picks[0]])
                if picks[-1] in grids:
                    report["live_vs_video_last_front"] = grid_diff(live16, grids[picks[-1]])
            try:
                report["video_front_camera_pose_first"] = r4(frames[0].cameras["front"].camera_pose.position)
            except Exception:
                pass
    except Exception as exc:
        err("B.rgb_stats", exc)
except Exception as exc:
    err("B.video", exc)
report["phases_done"].append("B")

# ---------------------------------------------------------------- Phase C: SAM3 probes
probes = []
FRONT_PROMPTS = ("stick", "obstacle", "pillar", "cylinder", "peg", "block", "target")
WRIST_PROMPTS = ("stick", "obstacle", "cylinder")
GEOM_PROMPTS = ("stick", "obstacle", "cylinder")
for camera_name, prompts in (("front", FRONT_PROMPTS), ("wrist", WRIST_PROMPTS)):
    for prompt in prompts:
        probe = {"camera": camera_name, "prompt": prompt, "ok": False, "n": 0, "cands": []}
        try:
            seg = segment_text(camera_name, prompt)
            probe["ok"] = bool(seg.ok)
            if seg.ok:
                probe["n"] = len(seg.segmentations)
                rank = 0
                for cand in seg.segmentations[:4]:
                    c = {"score": round(float(cand.score), 3)}
                    try:
                        if cand.box_xyxy is not None:
                            c["box"] = [round(float(v), 1) for v in cand.box_xyxy]
                    except Exception:
                        pass
                    if camera_name == "front" and prompt in GEOM_PROMPTS and rank < 3:
                        try:
                            pc = mask_to_point_cloud(cand, camera_name, target_frame=BASE)
                            npts = len(pc.points)
                            c["npts"] = npts
                            if npts >= 10:
                                geo = estimate_geometry(pc)
                                c["center"] = r4(geo.pose.position)
                                c["extents"] = r4(geo.extents)
                                zs = sorted(float(p[2]) for p in pc.points)
                                c["z_q10_q90"] = [round(zs[len(zs) // 10], 4), round(zs[(9 * len(zs)) // 10], 4)]
                        except Exception as exc:
                            c["geom_error"] = str(exc)[:200]
                    probe["cands"].append(c)
                    rank += 1
        except Exception as exc:
            probe["error"] = str(exc)[:200]
        probes.append(probe)
report["prompt_probes"] = probes
report["phases_done"].append("C")

# ---------------------------------------------------------------- Phase D: plan
plan = {"mode": "none", "reason": ""}
replay = []  # list of joint targets, one per control tick
try:
    if len(vid_q) < 2:
        plan["reason"] = "no conditioning-video robot states; nothing to imitate"
    else:
        ranges = report["video_summary"].get("joint_range_per_joint", [0.0])
        if max(ranges) < 0.05:
            plan["reason"] = "video robot did not move (max joint range < 0.05 rad)"
        else:
            order = list(range(len(vid_q)))
            if reverse_mode:
                order = list(reversed(order))
            seq_q = [vid_q[i] for i in order]
            seq_t = [vid_t[i] for i in order]
            # motion ticks at original timing (skip pauses)
            seg_ticks = []
            motion_ticks = 0
            for i in range(1, len(seq_q)):
                dq = max_abs_diff(seq_q[i], seq_q[i - 1])
                if dq < 0.0015:
                    seg_ticks.append(0)
                    continue
                n = int(round(abs(seq_t[i] - seq_t[i - 1]) * HZ))
                if n < 1:
                    n = 1
                seg_ticks.append(n)
                motion_ticks += n
            plan["motion_ticks_original"] = motion_ticks
            # v01: reserve ticks for the final dwell (the last target registers only after the
            # stick has dwelt on it for a few ticks; v00 ended 6 ticks after arriving and the
            # last of 11 visits never registered).  Budget the ACTUAL replay length, which the
            # per-segment rounding makes longer than slow * motion_ticks (1026 vs 882 on ep 0).
            d_live = max_abs_diff(live_q, seq_q[0]) if live_q is not None else 1.0
            approach_reserve = 10 if d_live <= 0.05 else 120
            horizon = MAX_STEPS if MAX_STEPS > 0 else 1000
            reserve = approach_reserve + FINAL_HOLD_MIN + 5
            slow = 1.5
            for attempt in range(30):
                replay = []
                for i in range(1, len(seq_q)):
                    n0 = seg_ticks[i - 1]
                    if n0 == 0:
                        continue
                    n = int(round(n0 * slow))
                    if n < 1:
                        n = 1
                    a = seq_q[i - 1]
                    b = seq_q[i]
                    for k in range(1, n + 1):
                        f = k / float(n)
                        replay.append([a[j] + (b[j] - a[j]) * f for j in range(len(a))])
                if len(replay) + reserve <= horizon or slow <= 0.5:
                    break
                slow = slow * 0.92
                if slow < 0.5:
                    slow = 0.5
            plan["slow_factor"] = round(slow, 3)
            plan["reserve_ticks"] = reserve
            plan["replay_ticks"] = len(replay)
            plan["start_q"] = r4(seq_q[0])
            plan["end_q"] = r4(seq_q[-1])
            d_start = max_abs_diff(live_q, seq_q[0]) if live_q is not None else None
            plan["live_vs_route_start_rad"] = None if d_start is None else round(d_start, 4)
            plan["mode"] = "reverse_replay" if reverse_mode else "replay"
            plan["reason"] = "imitate the demonstrated route from the episode's conditioning video"
except Exception as exc:
    err("D.plan", exc)
    plan["mode"] = "none"
report["plan"] = plan
report["phases_done"].append("D")
print("[v01] plan: " + str(plan.get("mode")) + " ticks=" + str(plan.get("replay_ticks")))


# ---------------------------------------------------------------- Phase E: execute
exec_log = {"ticks": 0, "track_err_samples": [], "ee_samples": [], "terminal": None,
            "step_errors": 0}
report["execution"] = exec_log


def current_q():
    try:
        return fl(get_robot_state().joint_positions["primary"])
    except Exception:
        return None


def send(q):
    if S["over"]:
        return None
    if MAX_STEPS > 0 and S["ticks"] >= MAX_STEPS - 2:
        note("tick budget reached; stopping")
        S["over"] = True
        return None
    try:
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=[float(v) for v in q],
            gripper_position=live_g, embodiment=EMB)}))
    except Exception as exc:
        exec_log["step_errors"] += 1
        if exec_log["step_errors"] <= 5:
            err("E.step", exc)
        if exec_log["step_errors"] >= 3:
            S["over"] = True
        return None
    S["ticks"] += 1
    exec_log["ticks"] = S["ticks"]
    info = reply_flags(reply)
    if is_terminal(info):
        S["over"] = True
        exec_log["terminal"] = {"tick": S["ticks"], "flags": info}
        note("terminal status at tick " + str(S["ticks"]) + ": " + str(info))
    elif info.get("ok", True) is False and exec_log["step_errors"] < 5:
        exec_log["step_errors"] += 1
        err("E.step_not_ok", info)
    return reply


def reply_q(reply):
    try:
        return fl(reply.observation.robot_state.joint_positions["primary"])
    except Exception:
        return None


def reply_ee(reply):
    try:
        return fl(reply.observation.robot_state.end_effector_poses["primary"].position)
    except Exception:
        return None


def servo_to(target, max_rate, max_ticks, tol):
    """Joint-space straight line from the measured joints to target, max_rate rad/tick."""
    q0 = current_q()
    if q0 is None:
        return False
    d = max_abs_diff(q0, target)
    n = int(math.ceil(d / max_rate))
    if n < 1:
        n = 1
    if n > max_ticks:
        n = max_ticks
    for k in range(1, n + 1):
        f = k / float(n)
        send([q0[j] + (target[j] - q0[j]) * f for j in range(len(q0))])
        if S["over"]:
            return False
    for hold in range(25):
        q = current_q()
        if q is not None and max_abs_diff(q, target) < tol:
            return True
        send(target)
        if S["over"]:
            return False
    q = current_q()
    return q is not None and max_abs_diff(q, target) < tol * 3


try:
    if plan["mode"] in ("replay", "reverse_replay") and len(replay) > 0 and live_q is not None:
        seq_first = vid_q[-1] if reverse_mode else vid_q[0]
        d_start = max_abs_diff(live_q, seq_first)
        exec_log["approach"] = {"d_start_rad": round(d_start, 4)}
        if d_start > 0.05:
            # Far from the demonstrated start: lift, transit above the route start, then descend
            # in joint space.  Only reached if the live robot does not begin where the video began.
            note("live robot is " + str(round(d_start, 3)) + " rad from route start; lifted transit")
            try:
                st = get_robot_state()
                ee = st.end_effector_poses["primary"]
                lift_z = float(ee.position[2]) + 0.12
                try:
                    vz = report["video_summary"]["ee_max"][2] + 0.05
                    if vz > lift_z:
                        lift_z = vz
                except Exception:
                    pass
                exec_log["approach"]["lift_z"] = round(lift_z, 4)
                up = Pose([float(ee.position[0]), float(ee.position[1]), lift_z],
                          fl(ee.quaternion_wxyz), BASE)
                r1 = move_to_pose(up, tolerance=0.02, max_steps=60)
                exec_log["approach"]["lift"] = reply_flags(r1)
                try:
                    S["ticks"] += int(r1.steps_executed)
                except Exception:
                    pass
                if is_terminal(reply_flags(r1)):
                    S["over"] = True
                if not S["over"]:
                    src = frames[-1] if reverse_mode else frames[0]
                    see = src.robot_state.end_effector_poses["primary"]
                    above = Pose([float(see.position[0]), float(see.position[1]),
                                  max(lift_z, float(see.position[2]) + 0.12)],
                                 fl(see.quaternion_wxyz), BASE)
                    r2 = move_to_pose(above, tolerance=0.02, max_steps=80)
                    exec_log["approach"]["above_start"] = reply_flags(r2)
                    try:
                        S["ticks"] += int(r2.steps_executed)
                    except Exception:
                        pass
                    if is_terminal(reply_flags(r2)):
                        S["over"] = True
            except Exception as exc:
                err("E.lifted_transit", exc)
        if not S["over"]:
            ok0 = servo_to(seq_first, 0.03, 80, 0.01)
            exec_log["approach"]["reached_start"] = bool(ok0)
            q = current_q()
            if q is not None:
                exec_log["approach"]["residual_rad"] = round(max_abs_diff(q, seq_first), 4)
        # replay the demonstrated route, one control tick per interpolated target
        max_err = 0.0
        sample_every = max(1, len(replay) // 60)
        for i in range(len(replay)):
            if S["over"]:
                break
            reply = send(replay[i])
            if reply is None:
                continue
            if i % sample_every == 0 or i == len(replay) - 1:
                q = reply_q(reply)
                if q is None:
                    q = current_q()
                if q is not None:
                    e = max_abs_diff(q, replay[i])
                    if e > max_err:
                        max_err = e
                    exec_log["track_err_samples"].append([i, round(e, 4)])
                p = reply_ee(reply)
                if p is not None:
                    exec_log["ee_samples"].append([i] + r4(p))
        exec_log["replay_max_track_err_rad"] = round(max_err, 4)
        exec_log["replay_ticks_sent"] = S["ticks"]
        report["phases_done"].append("E.replay")
        # settle on the final demonstrated configuration
        if not S["over"]:
            # v01: dwell on the final demonstrated configuration until the episode reports a
            # terminal status (success ends the native episode) or FINAL_HOLD_MAX ticks pass.
            # No early exit on small joint residual: in v00 that exit cost the last visit.
            final_q = replay[-1]
            hold_ticks = 0
            for hold in range(FINAL_HOLD_MAX):
                send(final_q)
                hold_ticks += 1
                if S["over"]:
                    break
            exec_log["final_hold_ticks"] = hold_ticks
            q = current_q()
            if q is not None:
                exec_log["final_residual_rad"] = round(max_abs_diff(q, final_q), 4)
        try:
            st = get_robot_state()
            exec_log["final_ee"] = r4(st.end_effector_poses["primary"].position)
            if len(vid_ee) > 0:
                ref = vid_ee[0] if reverse_mode else vid_ee[-1]
                if ref is not None:
                    exec_log["final_ee_vs_video_end_m"] = round(dist3(exec_log["final_ee"], ref), 4)
        except Exception as exc:
            err("E.final_state", exc)
    else:
        note("no executable plan: " + str(plan.get("reason")))
        # keyframe-producing benign hold so the run still records the scene
        if live_q is not None:
            send(live_q)
except Exception as exc:
    err("E.execute", exc)

report["ticks_used"] = S["ticks"]
report["episode_over_seen"] = S["over"]
report["phases_done"].append("E")
print("[v01] done; ticks=" + str(S["ticks"]) + " errors=" + str(len(report["errors"])))
result = report
