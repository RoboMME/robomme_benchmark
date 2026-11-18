# PatternLock v00 — initial program (Stage 0 exploration folded in).
# Hypothesis: the conditioning video shows the stick-holding arm tracing the
# pattern in this episode's own scene; the video's per-frame robot_state carries
# the joint trajectory. v00 records scene/video diagnostics and replays the
# video's joint trajectory tick by tick (time-resampled at the control rate).
# Everything is derived at runtime from this episode's video and live state.
import math

report = {"version": "v00", "phase": "start"}
ended = False


def is_ended(reply):
    term = False
    try:
        term = bool(reply.terminated) or bool(reply.truncated)
    except Exception:
        term = False
    return term


def flist(values, nd=4):
    return [round(float(v), nd) for v in values]


ctx = get_task_context()
report["goal"] = ctx.language
meta = {}
try:
    meta = dict(robomme_plus.get_task_metadata())
except Exception as exc:
    report["meta_error"] = str(exc)
report["meta"] = {}
for key in meta:
    try:
        report["meta"][str(key)] = str(meta[key])[:300]
    except Exception:
        report["meta"][str(key)] = "unprintable"
try:
    cmeta = dict(robomme_plus.get_controller_metadata())
    report["controller_meta"] = {str(k): str(cmeta[k])[:300] for k in cmeta}
except Exception as exc:
    report["controller_meta_error"] = str(exc)

hz = 20.0
try:
    hz = float(meta["control_hz"])
except Exception:
    hz = 20.0
max_steps = 0
try:
    max_steps = int(meta["max_steps"])
except Exception:
    max_steps = 0

state = get_robot_state()
emb = state.embodiment
q_live = [float(v) for v in state.joint_positions["primary"]]
grip_live = float(state.gripper_positions["primary"])
ee_live = state.end_effector_poses["primary"]
report["live_start"] = {
    "q": flist(q_live),
    "ee": flist(ee_live.position),
    "quat": flist(ee_live.quaternion_wxyz),
    "grip": round(grip_live, 4),
    "t": float(state.timestamp_s),
    "frame": state.base_frame,
}

# ---------------- conditioning video ----------------
video = list(robomme_plus.get_conditioning_video())
n_video = len(video)
report["video_len"] = n_video
vq = []
vt = []
vee = []
for frame in video:
    rs = frame.robot_state
    vq.append([float(v) for v in rs.joint_positions["primary"]])
    vt.append(float(frame.timestamp_s))
    vee.append([float(v) for v in rs.end_effector_poses["primary"].position])
if n_video > 0:
    report["video_t"] = [round(vt[0], 3), round(vt[-1], 3)]
    report["video_ee"] = [flist(p, 4) for p in vee]
    report["video_ts"] = [round(t, 3) for t in vt]
    report["video_q_sub"] = [flist(vq[i], 4) for i in range(0, n_video, 4)]
    report["video_q_last"] = flist(vq[-1], 4)
    zs = [p[2] for p in vee]
    report["video_ee_zrange"] = [round(min(zs), 4), round(max(zs), 4)]
    d0 = max(abs(a - b) for a, b in zip(vq[0], q_live))
    d1 = max(abs(a - b) for a, b in zip(vq[-1], q_live))
    report["live_vs_video_q"] = {"first": round(d0, 4), "last": round(d1, 4)}

# Layout check: compare live front RGB with video first / last front RGB on a
# coarse pixel grid (8x8 blocks of mean abs diff).
live_obs = get_observation()


def block_diff(rgb_a, rgb_b, stride=8, blocks=8):
    h = len(rgb_a)
    w = len(rgb_a[0])
    bh = h // blocks
    bw = w // blocks
    grid = []
    total = 0.0
    count = 0
    for by in range(blocks):
        row = []
        for bx in range(blocks):
            acc = 0.0
            n = 0
            for y in range(by * bh, (by + 1) * bh, stride):
                ra = rgb_a[y]
                rb = rgb_b[y]
                for x in range(bx * bw, (bx + 1) * bw, stride):
                    pa = ra[x]
                    pb = rb[x]
                    acc += abs(int(pa[0]) - int(pb[0])) + abs(int(pa[1]) - int(pb[1])) + abs(int(pa[2]) - int(pb[2]))
                    n += 1
            val = acc / max(1, n) / 3.0
            row.append(round(val, 1))
            total += acc
            count += n
        grid.append(row)
    return {"mean": round(total / max(1, count) / 3.0, 2), "grid": grid}


try:
    live_front = live_obs.cameras["front"].rgb
    report["front_shape"] = [len(live_front), len(live_front[0])]
    if n_video > 0:
        report["diff_live_vs_v0"] = block_diff(live_front, video[0].cameras["front"].rgb)
        report["diff_live_vs_vlast"] = block_diff(live_front, video[n_video - 1].cameras["front"].rgb)
        mid = n_video // 2
        report["diff_v0_vs_vmid"] = block_diff(video[0].cameras["front"].rgb, video[mid].cameras["front"].rgb)
        try:
            p0 = video[0].cameras["front"].camera_pose
            p1 = live_obs.cameras["front"].camera_pose
            report["front_pose_video"] = flist(p0.position) + flist(p0.quaternion_wxyz)
            report["front_pose_live"] = flist(p1.position) + flist(p1.quaternion_wxyz)
        except Exception as exc:
            report["pose_err"] = str(exc)
except Exception as exc:
    report["diff_error"] = str(exc)

# Live segmentation probes (scene understanding for later versions).
probes = []
for prompt in ("target", "button", "circle", "stick", "grid"):
    entry = {"prompt": prompt}
    try:
        seg = segment_text("front", prompt)
        entry["ok"] = bool(seg.ok)
        cands = []
        if seg.ok:
            for cand in seg.segmentations[:8]:
                box = None
                if cand.box_xyxy is not None:
                    box = [round(float(v), 1) for v in cand.box_xyxy]
                cands.append({"score": round(float(cand.score), 3), "box": box})
        entry["cands"] = cands
    except Exception as exc:
        entry["error"] = str(exc)[:200]
    probes.append(entry)
report["probes"] = probes

# ---------------- replay ----------------
report["phase"] = "pre-replay"


def hold_ticks(target, n):
    """Hold a joint target for n ticks; returns (ticks_done, ended_flag)."""
    done = 0
    stop = False
    for k in range(n):
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=list(target),
            gripper_position=grip_live, embodiment=emb)}))
        done += 1
        if is_ended(reply):
            stop = True
            break
    return done, stop


ticks_used = 0
if n_video >= 2:
    # 1) Reach the video's first configuration (lift first if the stick is low).
    d0 = max(abs(a - b) for a, b in zip(vq[0], q_live))
    if d0 > 0.03:
        zmin = min(p[2] for p in vee)
        if ee_live.position[2] < zmin + 0.04:
            lift = Pose([float(ee_live.position[0]), float(ee_live.position[1]),
                         float(ee_live.position[2]) + 0.08],
                        [float(v) for v in ee_live.quaternion_wxyz], state.base_frame)
            res = move_to_pose(lift, tolerance=0.02, max_steps=60)
            ticks_used += int(res.steps_executed)
            report["lift"] = {"ok": bool(res.ok), "steps": int(res.steps_executed)}
            if is_ended(res):
                ended = True
        if not ended:
            res = move_to_joints(vq[0], tolerance=0.01, max_steps=150)
            ticks_used += int(res.steps_executed)
            report["to_start"] = {"ok": bool(res.ok), "steps": int(res.steps_executed),
                                  "status": str(res.status)}
            if is_ended(res):
                ended = True

    # 2) Time-resample the video joints at the control rate.
    t0 = vt[0]
    total_t = vt[-1] - t0
    n_ticks = int(round(total_t * hz))
    targets = []
    seg_i = 0
    for k in range(1, n_ticks + 1):
        tau = t0 + k / hz
        while seg_i < n_video - 2 and vt[seg_i + 1] < tau:
            seg_i += 1
        ta = vt[seg_i]
        tb = vt[seg_i + 1]
        a = 0.0
        if tb > ta:
            a = (tau - ta) / (tb - ta)
        a = max(0.0, min(1.0, a))
        qa = vq[seg_i]
        qb = vq[seg_i + 1]
        targets.append([qa[j] + a * (qb[j] - qa[j]) for j in range(7)])
    report["replay_ticks_planned"] = len(targets)

    # 3) If the horizon is too short, compress static holds (keep <=5 ticks each).
    budget = max_steps - ticks_used - 25 if max_steps > 0 else len(targets)
    if len(targets) > budget:
        compressed = []
        still = 0
        prev = None
        for tq in targets:
            moving = True
            if prev is not None:
                moving = max(abs(a - b) for a, b in zip(tq, prev)) > 0.002
            if moving:
                still = 0
            else:
                still += 1
            if moving or still <= 5:
                compressed.append(tq)
            prev = tq
        targets = compressed
        report["replay_compressed_to"] = len(targets)
    if budget > 10 and len(targets) > budget:
        # Still too long: uniform time compression (speed-up) as a last resort.
        sped = []
        for i in range(budget):
            sped.append(targets[int(i * (len(targets) - 1) / max(1, budget - 1))])
        targets = sped
        report["replay_sped_to"] = len(targets)

    # 4) Replay tick by tick.
    track = []
    k = 0
    for tq in targets:
        if ended:
            break
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=tq,
            gripper_position=grip_live, embodiment=emb)}))
        ticks_used += 1
        if is_ended(reply):
            ended = True
            report["ended_at_replay_tick"] = k
            break
        if k % 20 == 0:
            try:
                qs = [float(v) for v in reply.observation.robot_state.joint_positions["primary"]]
                track.append(round(max(abs(a - b) for a, b in zip(qs, tq)), 4))
            except Exception:
                track.append(-1)
        k += 1
    report["replay_done_ticks"] = k
    report["track_err_every20"] = track
    # 5) Settle on the final configuration.
    if not ended:
        held, stop = hold_ticks(targets[-1], 15)
        ticks_used += held
        ended = ended or stop

report["ticks_used_est"] = ticks_used
report["ended"] = ended
if not ended:
    try:
        st = get_robot_state()
        report["final_ee"] = flist(st.end_effector_poses["primary"].position)
        report["final_t"] = float(st.timestamp_s)
    except Exception as exc:
        report["final_err"] = str(exc)[:200]
report["phase"] = "done"
result = report
