# PatternLock v00 -- snapshot + attempt in one start (R=1, interaction-only).
#
# Phase A  record: language, task/controller metadata, cameras, robot state.
# Phase B  conditioning video: per-frame timestamps, joints, EE, gripper; coarse
#          front-image change statistics; waypoint (low-speed) frames.
# Phase C  SAM3 probes on both cameras + 3D geometry of the top candidates,
#          cross-checked against the video's waypoint EE positions (log only).
# Phase D  attempt: replay the conditioning video's own joint trajectory at the
#          native control rate (timestamp-interpolated), dwell briefly at the
#          video's low-speed waypoints, then hold until a terminal status.
# Every phase is wrapped; the program never raises.

import math

result = {"version": "v00_initial", "errors": [], "phases": {}}
report = result

DWELL_TICKS = 3          # extra hold ticks at a video waypoint the video itself does not pause on
FINAL_HOLD_TICKS = 40    # bounded hold after the last replayed frame, until terminal status
MOTION_EPS_RAD = 0.003   # joint change that counts as "the robot moved" in the video
ALIGN_TOL_RAD = 0.02     # live-vs-video start joints mismatch that triggers an alignment move


def note_error(phase, exc):
    msg = str(phase) + ": " + str(exc)
    report["errors"].append(msg[:600])
    print("ERROR " + msg[:300])


def flist(values, nd):
    out = []
    for v in values:
        out.append(round(float(v), nd))
    return out


def maxdiff(a, b):
    best = 0.0
    for j in range(min(len(a), len(b))):
        d = abs(a[j] - b[j])
        if d > best:
            best = d
    return best


def dist3(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def median(values):
    ordered = sorted(values)
    if len(ordered) == 0:
        return 0.0
    return ordered[len(ordered) // 2]


def read_terminal(reply):
    term = False
    trunc = False
    try:
        term = bool(reply.terminated)
    except Exception:
        pass
    try:
        trunc = bool(reply.truncated)
    except Exception:
        pass
    try:
        if bool(reply.done):
            term = True
    except Exception:
        pass
    try:
        if bool(reply.terminal):
            term = True
    except Exception:
        pass
    return term, trunc


def reply_state(reply):
    try:
        return reply.observation.robot_state
    except Exception:
        pass
    try:
        return reply.final_observation.robot_state
    except Exception:
        pass
    return get_robot_state()


def grid_rgb(rgb, n):
    # n x n grid of RGB samples (pure Python, cheap)
    h = len(rgb)
    w = len(rgb[0])
    cells = []
    for gy in range(n):
        y = int((gy + 0.5) * h / n)
        row = rgb[y]
        for gx in range(n):
            x = int((gx + 0.5) * w / n)
            p = row[x]
            cells.append((int(p[0]), int(p[1]), int(p[2])))
    return cells


def grid_diff(a, b):
    total = 0.0
    per = []
    for k in range(min(len(a), len(b))):
        d = (abs(a[k][0] - b[k][0]) + abs(a[k][1] - b[k][1]) + abs(a[k][2] - b[k][2])) / 3.0
        per.append(d)
        total += d
    mean = total / max(1, len(per))
    return mean, per


# ---------------------------------------------------------------- Phase A
state0 = None
base_frame = "robot_base"
embodiment = "robomme_plus"
control_hz = 20.0
max_steps = None
language = ""
try:
    context = get_task_context()
    language = str(context.language)
    report["task_language"] = language
    try:
        report["task_name"] = str(context.task_name)
        report["task_id"] = str(context.task_id)
    except Exception as exc:
        note_error("A.context_ids", exc)
    print("language: " + language)
except Exception as exc:
    note_error("A.context", exc)

try:
    meta = robomme_plus.get_task_metadata()
    md = {}
    for key in meta:
        md[str(key)] = str(meta[key])[:400]
    report["task_metadata"] = md
    try:
        max_steps = int(meta["max_steps"])
    except Exception as exc:
        note_error("A.max_steps", exc)
    try:
        control_hz = float(meta["control_hz"])
    except Exception:
        try:
            control_hz = float(meta["control_frequency"])
        except Exception:
            pass
    if language == "":
        try:
            language = str(meta["language"])
            report["task_language"] = language
        except Exception:
            pass
except Exception as exc:
    note_error("A.task_metadata", exc)

try:
    cmeta = robomme_plus.get_controller_metadata()
    cd = {}
    for key in cmeta:
        cd[str(key)] = str(cmeta[key])[:400]
    report["controller_metadata"] = cd
except Exception as exc:
    note_error("A.controller_metadata", exc)

report["max_steps"] = max_steps
report["control_hz"] = control_hz
dt = 1.0 / control_hz if control_hz > 0 else 0.05

live_obs = None
try:
    state0 = get_robot_state()
    base_frame = str(state0.base_frame)
    embodiment = str(state0.embodiment)
    ee0 = state0.end_effector_poses["primary"]
    report["robot_state0"] = {
        "base_frame": base_frame,
        "embodiment": embodiment,
        "timestamp_s": float(state0.timestamp_s),
        "joints": flist(state0.joint_positions["primary"], 4),
        "gripper": float(state0.gripper_positions["primary"]),
        "ee_position": flist(ee0.position, 4),
        "ee_quat_wxyz": flist(ee0.quaternion_wxyz, 4),
    }
except Exception as exc:
    note_error("A.robot_state", exc)

try:
    live_obs = get_observation()
    cams = {}
    for name in live_obs.cameras:
        cam = live_obs.cameras[name]
        info = {}
        try:
            info["rgb_hw"] = [len(cam.rgb), len(cam.rgb[0])]
        except Exception as exc:
            note_error("A.cam_rgb_" + str(name), exc)
        try:
            info["intrinsics"] = [flist(row, 3) for row in cam.intrinsics]
        except Exception as exc:
            note_error("A.cam_K_" + str(name), exc)
        try:
            info["pose_position"] = flist(cam.camera_pose.position, 4)
            info["pose_quat_wxyz"] = flist(cam.camera_pose.quaternion_wxyz, 4)
            info["pose_frame"] = str(cam.camera_pose.frame)
        except Exception as exc:
            note_error("A.cam_pose_" + str(name), exc)
        cams[str(name)] = info
    report["cameras"] = cams
    report["live_timestamp_s"] = float(live_obs.timestamp_s)
except Exception as exc:
    note_error("A.observation", exc)

# ---------------------------------------------------------------- Phase B
frames = []
vq = []      # joints per frame
vt = []      # timestamps per frame
vg = []      # gripper per frame
vee = []     # EE position per frame
vstate_ok = False
try:
    raw = robomme_plus.get_conditioning_video()
    for fr in raw:
        frames.append(fr)
    report["video_len"] = len(frames)
    print("conditioning video frames: " + str(len(frames)))
    if len(frames) == 0:
        report["video_note"] = "conditioning video has 0 frames"
except Exception as exc:
    note_error("B.get_video", exc)

try:
    for i in range(len(frames)):
        fr = frames[i]
        rs = fr.robot_state
        vq.append([float(v) for v in rs.joint_positions["primary"]])
        vt.append(float(fr.timestamp_s))
        try:
            vg.append(float(rs.gripper_positions["primary"]))
        except Exception:
            vg.append(None)
        try:
            vee.append([float(v) for v in rs.end_effector_poses["primary"].position])
        except Exception:
            vee.append(None)
    vstate_ok = len(vq) > 1
    if len(vt) > 0:
        gaps = [vt[i + 1] - vt[i] for i in range(len(vt) - 1)]
        report["video_time"] = {
            "first": vt[0], "last": vt[-1],
            "gap_min": min(gaps) if gaps else None,
            "gap_max": max(gaps) if gaps else None,
            "gap_median": median(gaps) if gaps else None,
        }
except Exception as exc:
    note_error("B.video_state", exc)

# compact per-frame logs (joints 3 dp, EE mm)
try:
    report["video_joints"] = [flist(q, 3) for q in vq]
    report["video_ee"] = [flist(p, 3) if p is not None else None for p in vee]
    report["video_gripper"] = [round(g, 3) if g is not None else None for g in vg]
except Exception as exc:
    note_error("B.video_log", exc)

# motion range, waypoints
start_i = 0
end_i = max(0, len(vq) - 1)
video_moved = False
waypoints = []
try:
    if vstate_ok:
        n = len(vq)
        biggest = 0.0
        for i in range(n):
            d = maxdiff(vq[i], vq[0])
            if d > biggest:
                biggest = d
        video_moved = biggest > 0.02
        report["video_max_joint_excursion"] = round(biggest, 4)
        for i in range(n):
            if maxdiff(vq[i], vq[0]) > MOTION_EPS_RAD:
                start_i = max(0, i - 1)
                break
        for i in range(n - 1, -1, -1):
            if maxdiff(vq[i], vq[n - 1]) > MOTION_EPS_RAD:
                end_i = min(n - 1, i + 1)
                break
        report["video_motion_range"] = [start_i, end_i]
        # joint speed per frame (rad/s, max over joints)
        speeds = {}
        for i in range(start_i + 1, end_i):
            span = vt[i + 1] - vt[i - 1]
            if span < dt:
                span = 2 * dt
            speeds[i] = maxdiff(vq[i + 1], vq[i - 1]) / span
        moving = [s for s in speeds.values() if s > 1e-3]
        med = median(moving)
        thresh = 0.3 * med
        report["video_speed_median"] = round(med, 4)
        region = []
        regions = []
        for i in range(start_i + 1, end_i):
            if speeds[i] < thresh:
                region.append(i)
            else:
                if region:
                    regions.append(region)
                region = []
        if region:
            regions.append(region)
        for reg in regions:
            best = reg[0]
            for i in reg:
                if speeds[i] < speeds[best]:
                    best = i
            span_ticks = int(round((vt[reg[-1]] - vt[reg[0]]) / dt)) + 1
            wp = {"frame": best, "t": round(vt[best], 3), "region": [reg[0], reg[-1]],
                  "pause_ticks": span_ticks,
                  "ee": flist(vee[best], 4) if vee[best] is not None else None}
            waypoints.append(wp)
        report["video_waypoints"] = waypoints
        print("video moved=" + str(video_moved) + " range=" + str([start_i, end_i]) +
              " waypoints=" + str(len(waypoints)))
        # lowest EE heights in the video (touch candidates)
        zs = [(vee[i][2], i) for i in range(n) if vee[i] is not None]
        if zs:
            zmin = min(zs)[0]
            report["video_ee_zmin"] = round(zmin, 4)
            low = []
            for i in range(1, n - 1):
                if vee[i] is None or vee[i - 1] is None or vee[i + 1] is None:
                    continue
                z = vee[i][2]
                if z <= vee[i - 1][2] and z <= vee[i + 1][2] and z < zmin + 0.02:
                    low.append({"frame": i, "t": round(vt[i], 3), "ee": flist(vee[i], 4)})
            report["video_low_points"] = low[:60]
except Exception as exc:
    note_error("B.waypoints", exc)

# coarse front-image statistics over the video, and live vs video
try:
    if len(frames) > 0 and live_obs is not None:
        g0 = grid_rgb(frames[0].cameras["front"].rgb, 32)
        glast = grid_rgb(frames[-1].cameras["front"].rgb, 32)
        glive = grid_rgb(live_obs.cameras["front"].rgb, 32)
        m0, per0 = grid_diff(glive, g0)
        ml, perl = grid_diff(glive, glast)
        report["live_vs_video_front"] = {
            "vs_frame0_mean": round(m0, 2),
            "vs_frame0_cells_gt30": len([d for d in per0 if d > 30]),
            "vs_last_mean": round(ml, 2),
            "vs_last_cells_gt30": len([d for d in perl if d > 30]),
        }
        stats = []
        count = len(frames)
        picks = []
        nsamp = 32
        for k in range(nsamp):
            idx = int(round(k * (count - 1) / max(1, nsamp - 1)))
            if idx not in picks:
                picks.append(idx)
        for idx in picks:
            g = grid_rgb(frames[idx].cameras["front"].rgb, 32)
            m, per = grid_diff(g, g0)
            ranked = sorted([(-per[c], c) for c in range(len(per))])
            order = [pair[1] for pair in ranked[:4]]
            stats.append({"frame": idx, "t": round(vt[idx], 3) if idx < len(vt) else None,
                          "mean": round(m, 2),
                          "cells_gt40": len([d for d in per if d > 40]),
                          "top": [[c // 32, c % 32, round(per[c], 1), list(g[c])] for c in order]})
        report["video_front_change"] = stats
except Exception as exc:
    note_error("B.image_stats", exc)

# ---------------------------------------------------------------- Phase C
targets3d = []
probes = []
try:
    plan_probes = [("front", "target", 5), ("front", "disc", 5), ("front", "circle", 5),
                   ("front", "button", 5), ("front", "dot", 3), ("front", "stick", 3),
                   ("wrist", "target", 3), ("wrist", "disc", 3)]
    for camera_name, prompt, nproj in plan_probes:
        probe = {"camera": camera_name, "prompt": prompt, "ok": False, "cands": []}
        try:
            seg = segment_text(camera_name, prompt)
            probe["ok"] = bool(seg.ok)
            if seg.ok:
                probe["num"] = len(seg.segmentations)
                rank = 0
                for cand in seg.segmentations[:nproj]:
                    entry = {"rank": rank, "score": round(float(cand.score), 3)}
                    rank += 1
                    try:
                        if cand.box_xyxy is not None:
                            entry["box"] = flist(cand.box_xyxy, 1)
                    except Exception:
                        pass
                    if camera_name == "front":
                        try:
                            cloud = mask_to_point_cloud(cand, camera_name, target_frame=base_frame)
                            pts = cloud.points
                            npts = len(pts)
                            entry["npts"] = npts
                            if npts >= 5:
                                stride = max(1, npts // 300)
                                xs = []
                                ys = []
                                zs = []
                                for pi in range(0, npts, stride):
                                    p = pts[pi]
                                    xs.append(float(p[0]))
                                    ys.append(float(p[1]))
                                    zs.append(float(p[2]))
                                xs.sort()
                                ys.sort()
                                zs.sort()
                                m = len(xs)
                                entry["med"] = [round(xs[m // 2], 4), round(ys[m // 2], 4), round(zs[m // 2], 4)]
                                lo = int(0.05 * (m - 1))
                                hi = int(0.95 * (m - 1))
                                entry["ext"] = [round(xs[hi] - xs[lo], 4), round(ys[hi] - ys[lo], 4),
                                                round(zs[hi] - zs[lo], 4)]
                                entry["zmax"] = round(zs[hi], 4)
                                if prompt in ("target", "disc", "circle", "button", "dot"):
                                    targets3d.append({"prompt": prompt, "score": entry["score"],
                                                      "med": entry["med"], "ext": entry["ext"]})
                        except Exception as exc:
                            entry["proj_error"] = str(exc)[:200]
                    probe["cands"].append(entry)
            else:
                try:
                    probe["error"] = str(seg.error)[:200]
                except Exception:
                    pass
        except Exception as exc:
            probe["error"] = str(exc)[:200]
        probes.append(probe)
        print("probe " + camera_name + " '" + prompt + "' ok=" + str(probe["ok"]) +
              " n=" + str(len(probe["cands"])))
    report["prompt_probes"] = probes
except Exception as exc:
    note_error("C.probes", exc)

# cross-check: video waypoint / low-point EE xy vs live candidate targets (log only)
try:
    checks = []
    pts = []
    for wp in waypoints:
        if wp["ee"] is not None:
            pts.append(("wp", wp["frame"], wp["ee"]))
    for lp in report.get("video_low_points", []):
        pts.append(("low", lp["frame"], lp["ee"]))
    for kind, frame_i, ee in pts:
        best = None
        for tg in targets3d:
            d = math.sqrt((tg["med"][0] - ee[0]) ** 2 + (tg["med"][1] - ee[1]) ** 2)
            if best is None or d < best[0]:
                best = (d, tg)
        if best is not None:
            checks.append({"kind": kind, "frame": frame_i, "xy_dist": round(best[0], 4),
                           "target_med": best[1]["med"], "prompt": best[1]["prompt"]})
    report["video_vs_live_targets"] = checks
except Exception as exc:
    note_error("C.crosscheck", exc)

# ---------------------------------------------------------------- Phase D
exec_log = {"steps": 0, "terminated": False, "truncated": False, "stopped_reason": None,
            "step_errors": [], "aligned": None}
report["execution"] = exec_log
live_ee_log = []
track_err = []
wp_checks = []


def do_step(q, g):
    # one native tick holding/commanding joints q; returns (continue?, state)
    cmd = ArmCommand(mode="joint_position", target=list(q), gripper_position=g,
                     embodiment=embodiment)
    try:
        reply = step(RobotAction(arms={"primary": cmd}))
    except Exception as exc:
        exec_log["step_errors"].append(str(exc)[:200])
        exec_log["stopped_reason"] = "step raised: " + str(exc)[:200]
        return False, None
    exec_log["steps"] += 1
    ok = True
    try:
        ok = bool(reply.ok)
    except Exception:
        ok = True
    if not ok:
        try:
            exec_log["step_errors"].append(str(reply.error)[:200])
        except Exception:
            exec_log["step_errors"].append("step not ok")
    term, trunc = read_terminal(reply)
    st = None
    try:
        st = reply_state(reply)
    except Exception:
        st = None
    if term or trunc:
        exec_log["terminated"] = term
        exec_log["truncated"] = trunc
        exec_log["stopped_reason"] = "terminal status at step " + str(exec_log["steps"])
        return False, st
    if len(exec_log["step_errors"]) >= 5 and not ok:
        exec_log["stopped_reason"] = "repeated step errors"
        return False, st
    return True, st


try:
    if not vstate_ok or not video_moved:
        exec_log["stopped_reason"] = "no usable robot motion in conditioning video (len=" + \
            str(len(vq)) + ", moved=" + str(video_moved) + "); holding only"
        print(exec_log["stopped_reason"])
        hold_state = get_robot_state()
        hq = [float(v) for v in hold_state.joint_positions["primary"]]
        hg = float(hold_state.gripper_positions["primary"])
        for tick in range(5):
            cont, st = do_step(hq, hg)
            if not cont:
                break
    else:
        # build per-tick plan from timestamps
        plan = []        # list of (q, g, frame_index_or_None)
        wp_frames = {}
        all_wp_frames = {}
        for wp in waypoints:
            all_wp_frames[wp["frame"]] = True
        for wp in waypoints:
            if wp["pause_ticks"] < DWELL_TICKS:
                wp_frames[wp["frame"]] = True
        for i in range(start_i, end_i):
            gap = vt[i + 1] - vt[i]
            k = int(round(gap / dt)) if gap > 0 else 1
            if k < 1:
                k = 1
            ga = vg[i] if vg[i] is not None else 1.0
            gb = vg[i + 1] if vg[i + 1] is not None else ga
            for s in range(1, k + 1):
                f = s / float(k)
                q = [vq[i][j] + (vq[i + 1][j] - vq[i][j]) * f for j in range(len(vq[i]))]
                g = ga + (gb - ga) * f
                plan.append((q, g, i + 1 if s == k else None))
            if (i + 1) in wp_frames:
                for d in range(DWELL_TICKS):
                    plan.append((vq[i + 1], gb, None))
        budget = (max_steps if max_steps is not None else 100000) - 10
        report["plan_ticks"] = len(plan)
        report["plan_budget"] = budget
        if len(plan) > budget and budget > 10:
            # drop dwell ticks first, then subsample uniformly
            exec_log["plan_compressed"] = True
            stride = len(plan) / float(budget)
            newplan = []
            pos = 0.0
            while int(pos) < len(plan):
                newplan.append(plan[int(pos)])
                pos += stride
            if newplan[-1] is not plan[-1]:
                newplan.append(plan[-1])
            plan = newplan
        # align live start with the video's start frame if needed
        live = get_robot_state()
        lq = [float(v) for v in live.joint_positions["primary"]]
        start_gap = maxdiff(lq, vq[start_i])
        report["live_vs_video_start_joint_gap"] = round(start_gap, 4)
        report["live_vs_video_frame0_joint_gap"] = round(maxdiff(lq, vq[0]), 4)
        if start_gap > ALIGN_TOL_RAD:
            try:
                al = move_to_joints(vq[start_i], tolerance=0.01, max_steps=150)
                exec_log["aligned"] = {"ok": bool(al.ok), "steps": int(al.steps_executed)}
            except Exception as exc:
                exec_log["aligned"] = {"error": str(exc)[:200]}
        else:
            exec_log["aligned"] = "not needed"
        print("replay plan ticks=" + str(len(plan)) + " start_gap=" + str(round(start_gap, 4)))
        cont = True
        last_q = vq[start_i]
        last_g = vg[start_i] if vg[start_i] is not None else 1.0
        for idx in range(len(plan)):
            q, g, fi = plan[idx]
            cont, st = do_step(q, g)
            last_q = q
            last_g = g
            if st is not None:
                try:
                    if idx % 2 == 0 or fi in all_wp_frames:
                        live_ee_log.append([exec_log["steps"]] +
                                           flist(st.end_effector_poses["primary"].position, 4))
                    mq = [float(v) for v in st.joint_positions["primary"]]
                    track_err.append(maxdiff(mq, q))
                    if fi is not None and fi in all_wp_frames and vee[fi] is not None:
                        lee = [float(v) for v in st.end_effector_poses["primary"].position]
                        wp_checks.append({"frame": fi, "step": exec_log["steps"],
                                          "live_ee": flist(lee, 4),
                                          "video_ee": flist(vee[fi], 4),
                                          "err_m": round(dist3(lee, vee[fi]), 4)})
                except Exception as exc:
                    if len(exec_log["step_errors"]) < 20:
                        exec_log["step_errors"].append("log: " + str(exc)[:120])
            if not cont:
                break
        exec_log["replay_complete"] = cont
        print("replay done steps=" + str(exec_log["steps"]) + " cont=" + str(cont))
        # bounded hold at the final frame until a terminal status
        if cont:
            remaining = (max_steps if max_steps is not None else 100000) - exec_log["steps"] - 2
            nhold = min(FINAL_HOLD_TICKS, max(0, remaining))
            held = 0
            for tick in range(nhold):
                cont, st = do_step(vq[end_i], last_g)
                held += 1
                if not cont:
                    break
            exec_log["final_hold_ticks"] = held
except Exception as exc:
    note_error("D.execute", exc)

try:
    report["live_ee_log"] = live_ee_log
    if track_err:
        report["tracking_err_rad"] = {"max": round(max(track_err), 4),
                                      "mean": round(sum(track_err) / len(track_err), 4)}
    report["waypoint_tracking"] = wp_checks
    final_state = get_robot_state()
    report["final_state"] = {"timestamp_s": float(final_state.timestamp_s),
                             "joints": flist(final_state.joint_positions["primary"], 4),
                             "ee": flist(final_state.end_effector_poses["primary"].position, 4)}
except Exception as exc:
    note_error("D.final_log", exc)

# ---------------------------------------------------------------- per-clause record + self-check
try:
    text = language
    for sep in [", then ", " then ", "; ", ". ", ", and ", " finally ", ", "]:
        text = text.replace(sep, "|")
    clauses = []
    for part in text.split("|"):
        part = part.strip()
        if part:
            clauses.append({"clause": part,
                            "executed_by": "video joint replay" if video_moved else "none",
                            "replay_complete": exec_log.get("replay_complete", False)})
    report["clauses"] = clauses
    report["self_check"] = {
        "video_frames": len(vq),
        "video_moved": video_moved,
        "replay_complete": exec_log.get("replay_complete", False),
        "terminal_seen": bool(exec_log["terminated"] or exec_log["truncated"]),
        "steps": exec_log["steps"],
        "errors": len(report["errors"]),
    }
    print("self_check " + str(report["self_check"]))
except Exception as exc:
    note_error("E.self_check", exc)
