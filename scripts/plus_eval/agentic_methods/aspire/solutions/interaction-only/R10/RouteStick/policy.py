# v00_initial — RouteStick: reproduce the conditioning video's drawn path.
#
# Strategy (all quantities derived at runtime from this episode's inputs):
#  1. Read the conditioning video's per-frame robot state (joints + end-effector pose) and the
#     original control-step index of each retained frame.
#  2. Find the drawing phase (frames near the lowest end-effector height) and its dwell points
#     (frames where the tool rests, i.e. the route's keypoints).
#  3. Localize the small route markers in the LIVE front image; snap every video keypoint to its
#     nearest live marker. If the live layout matches the video (residual below a tolerance) the
#     recorded joints are replayed unchanged; otherwise each segment of the path is mapped with
#     the 2-D similarity that carries its two video keypoints onto the two live markers and is
#     re-solved with IK (seeded by the recorded joints).
#  4. Execute one joint target per native control tick at the video's original pacing.

import math

CAM = "front"
MARKER_PROMPTS = ("button", "small white marker", "white disc")
DWELL_RADIUS = 0.004       # m: xy motion allowed inside a dwell
DWELL_MIN_S = 0.3          # s: minimum dwell duration
DRAW_BAND = 0.01           # m: frames within this of the lowest EE height are "drawing"
SNAP_MAX = 0.08            # m: a live marker farther than this from a keypoint is not its match
SNAP_TOL = 0.015           # m: residual below this is treated as perception noise (no correction)
LEAD_TICKS = 1             # command the recorded state this many ticks ahead (PD lag)
FINAL_HOLD = 20            # ticks to hold the last pose

report = {"version": "v00_initial"}
ctx = get_task_context()
report["language"] = ctx.language
state0 = get_robot_state()
base = state0.base_frame
emb = state0.embodiment
grip_cmd = float(state0.gripper_positions["primary"])

# ---------------- 1. conditioning video ----------------
video = robomme_plus.get_conditioning_video()
n = len(video)
report["video_frames"] = n
if n < 2:
    raise ValueError("no conditioning video")

src = None
try:
    src = [int(round(float(v))) for v in ctx.metadata["conditioning_video_sampling"]["source_frame_indices"]]
    if len(src) != n:
        src = None
except Exception:
    src = None
freq = 20.0
try:
    freq = float(robomme_plus.get_task_metadata()["control_frequency_hz"])
except Exception:
    freq = 20.0

J = []
POS = []
QUAT = []
TS = []
for index in range(n):
    frame = video[index]
    rs = frame.robot_state
    J.append([float(v) for v in rs.joint_positions["primary"]])
    pose = rs.end_effector_poses["primary"]
    POS.append([float(v) for v in pose.position])
    QUAT.append([float(v) for v in pose.quaternion_wxyz])
    TS.append(float(frame.timestamp_s))
if src is None:
    src = [int(round((t - TS[0]) * freq)) for t in TS]
report["video_ticks"] = src[n - 1] - src[0]

# ---------------- 2. drawing phase and dwell keypoints ----------------
zmin = min(p[2] for p in POS)
drawing = [p[2] <= zmin + DRAW_BAND for p in POS]
dwells = []
i = 0
while i < n:
    if not drawing[i]:
        i += 1
        continue
    j = i
    while j + 1 < n and drawing[j + 1] and math.hypot(POS[j + 1][0] - POS[i][0], POS[j + 1][1] - POS[i][1]) < DWELL_RADIUS:
        j += 1
    if TS[j] - TS[i] >= DWELL_MIN_S:
        xs = sorted(POS[k][0] for k in range(i, j + 1))
        ys = sorted(POS[k][1] for k in range(i, j + 1))
        dwells.append({"start": i, "end": j, "xy": [xs[len(xs) // 2], ys[len(ys) // 2]]})
    i = j + 1
report["draw_z"] = round(zmin, 4)
report["keypoints"] = [{"start": d["start"], "end": d["end"], "xy": [round(d["xy"][0], 4), round(d["xy"][1], 4)]} for d in dwells]

# ---------------- 3. live markers ----------------
markers = []
marker_log = []
for prompt in MARKER_PROMPTS:
    seg = segment_text(CAM, prompt)
    if not seg.ok:
        marker_log.append({"prompt": prompt, "ok": False})
        continue
    for cand in seg.segmentations[:5]:
        entry = {"prompt": prompt, "score": round(float(cand.score), 3)}
        try:
            cloud = mask_to_point_cloud(cand, CAM, target_frame=base)
            pts = cloud.points
            if len(pts) < 5:
                entry["reject"] = "few points"
                marker_log.append(entry)
                continue
            xs = sorted(float(p[0]) for p in pts)
            ys = sorted(float(p[1]) for p in pts)
            zs = sorted(float(p[2]) for p in pts)
            m = len(xs)
            lo = m // 10
            hi = (9 * m) // 10
            span_xy = max(xs[hi] - xs[lo], ys[hi] - ys[lo])
            span_z = zs[hi] - zs[lo]
            center = [xs[m // 2], ys[m // 2], zs[m // 2]]
            entry["center"] = [round(v, 4) for v in center]
            entry["span_xy"] = round(span_xy, 4)
            entry["span_z"] = round(span_z, 4)
            if span_xy > 0.06 or span_z > 0.04:
                entry["reject"] = "too large"
                marker_log.append(entry)
                continue
            dup = False
            for mk in markers:
                if math.hypot(mk[0] - center[0], mk[1] - center[1]) < 0.02:
                    dup = True
            if not dup:
                markers.append(center)
            marker_log.append(entry)
        except Exception as exc:
            entry["error"] = str(exc)
            marker_log.append(entry)
report["markers"] = [[round(v, 4) for v in mk] for mk in markers]
report["marker_log"] = marker_log

deltas = []
for d in dwells:
    best = None
    best_dist = None
    for mk in markers:
        dist = math.hypot(mk[0] - d["xy"][0], mk[1] - d["xy"][1])
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = mk
    if best is not None and best_dist <= SNAP_MAX and best_dist > SNAP_TOL:
        deltas.append([best[0] - d["xy"][0], best[1] - d["xy"][1]])
    else:
        deltas.append([0.0, 0.0])
    d["residual"] = None if best_dist is None else round(best_dist, 4)
report["keypoint_residuals"] = [d["residual"] for d in dwells]
# Layout match is a global property: if the median keypoint residual is perception-noise sized,
# the live scene is the recorded one and the recorded joints are replayed unchanged.
known = sorted(d["residual"] for d in dwells if d["residual"] is not None)
layout_same = len(known) == 0 or known[len(known) // 2] <= SNAP_TOL
report["layout_same"] = layout_same
if layout_same:
    deltas = [[0.0, 0.0] for d in dwells]
corrected = any(abs(dl[0]) + abs(dl[1]) > 0.0 for dl in deltas)
report["corrected"] = corrected


def similarity_map(a, b, a2, b2, p):
    # 2-D similarity mapping a->a2 and b->b2, applied to p (xy only)
    vx = b[0] - a[0]
    vy = b[1] - a[1]
    wx = b2[0] - a2[0]
    wy = b2[1] - a2[1]
    den = vx * vx + vy * vy
    if den < 1e-8:
        return [p[0] + a2[0] - a[0], p[1] + a2[1] - a[1]]
    c = (vx * wx + vy * wy) / den
    s = (vx * wy - vy * wx) / den
    px = p[0] - a[0]
    py = p[1] - a[1]
    return [a2[0] + c * px - s * py, a2[1] + s * px + c * py]


targets = [list(q) for q in J]
ik_fail = 0
if corrected and len(dwells) > 0:
    zhome = POS[0][2]
    for k in range(n):
        p = POS[k]
        if k <= dwells[0]["end"]:
            dl = deltas[0]
            w = 1.0
            if zhome - zmin > 1e-6:
                w = min(1.0, max(0.0, (zhome - p[2]) / (zhome - zmin)))
            new_xy = [p[0] + w * dl[0], p[1] + w * dl[1]]
        elif k >= dwells[len(dwells) - 1]["start"]:
            dl = deltas[len(dwells) - 1]
            new_xy = [p[0] + dl[0], p[1] + dl[1]]
        else:
            seg_i = 0
            for s_i in range(len(dwells) - 1):
                if dwells[s_i]["start"] <= k and k <= dwells[s_i + 1]["end"]:
                    seg_i = s_i
            a = dwells[seg_i]["xy"]
            b = dwells[seg_i + 1]["xy"]
            a2 = [a[0] + deltas[seg_i][0], a[1] + deltas[seg_i][1]]
            b2 = [b[0] + deltas[seg_i + 1][0], b[1] + deltas[seg_i + 1][1]]
            new_xy = similarity_map(a, b, a2, b2, p)
        if math.hypot(new_xy[0] - p[0], new_xy[1] - p[1]) > 0.001:
            ik = solve_ik(Pose((new_xy[0], new_xy[1], p[2]), QUAT[k], base), seed_joints=J[k])
            if ik.ok:
                targets[k] = [float(v) for v in ik.joint_positions]
            else:
                ik_fail += 1
report["ik_fail"] = ik_fail

# ---------------- 4. execute at the recorded pacing ----------------
live_j = [float(v) for v in get_robot_state().joint_positions["primary"]]
start_gap = max(abs(live_j[q] - targets[0][q]) for q in range(7))
report["start_gap_rad"] = round(start_gap, 4)
if start_gap > 0.02:
    move_to_joints(targets[0], tolerance=0.01, max_steps=150)

tick_targets = []
for k in range(n - 1):
    gap = max(1, src[k + 1] - src[k])
    for g in range(gap):
        f = g / float(gap)
        tick_targets.append([targets[k][q] + f * (targets[k + 1][q] - targets[k][q]) for q in range(7)])
for h in range(FINAL_HOLD + LEAD_TICKS):
    tick_targets.append(list(targets[n - 1]))

# ticks at which to sample tracking error: the end of each dwell
check_ticks = {}
for di in range(len(dwells)):
    check_ticks[src[dwells[di]["end"]] - src[0]] = di
tracking = []
ended = False
steps_done = 0
total = len(tick_targets) - LEAD_TICKS
for t in range(total):
    cmd = tick_targets[t + LEAD_TICKS]
    res = step(RobotAction(arms={"primary": ArmCommand(
        "joint_position", cmd, gripper_position=grip_cmd, embodiment=emb)}))
    steps_done += 1
    term = False
    try:
        term = bool(res.terminated) or bool(res.truncated)
    except Exception:
        term = False
    if term:
        ended = True
        report["ended_at_tick"] = t
        break
    if t in check_ticks:
        try:
            ee = get_robot_state().end_effector_poses["primary"].position
            di = check_ticks[t]
            tracking.append({"kp": di, "ee": [round(float(v), 4) for v in ee]})
        except Exception:
            pass
report["steps_done"] = steps_done
report["ended"] = ended
report["tracking"] = tracking
result = report
