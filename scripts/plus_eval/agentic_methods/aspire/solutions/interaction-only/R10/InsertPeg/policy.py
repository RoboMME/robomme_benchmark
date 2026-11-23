# v06 — InsertPeg: v05 with the side views REMOVED (v05 seeds 1,2: the view shifted +2 cm
# picked up non-floor points up to +3.7 cm / z 0.06, so the union centre was wrong on both).
# Hole centre = MID-POINT of the lateral extent (q03..q97) of the visible hole-floor points in
# the single centred wrist view (the median is biased by the peg occluding one side); fine
# 1.5 mm lateral grid first, then +-1.5 mm and +3/+4.5 mm heights. Keeps v05's deep push.
# v05 — InsertPeg: v04 + two failure-mode fixes (seed 1: "entered but no native success";
# seed 2: "every offset stalls on the face").
#  (1) Insertion depth: once the peg tip has entered the hole (progress clearly beyond the face
#      stall), keep the guarded push going PAST the video's push depth until it stalls or the
#      episode ends, then hold. v04 stopped 4 mm short of the video depth.
#  (2) Hole centre from an unoccluded view: the held peg hides part of the hole floor in the
#      wrist view, so the visible-floor median is biased away from the peg line. v05 takes three
#      wrist-depth views (peg retracted, centred and shifted to either side), unions the
#      world-frame hole points and uses the MID-POINT of their lateral extent as the hole centre;
#      floor height as before. Search around it on a fine 1.5 mm grid with short retracts.
# All offsets / steps are generic tolerances; every pose comes from the video or live depth.
# v04 — InsertPeg: v03 + program fix: the native episode can end (success) inside a push; every
# state read after a step is now guarded and the push loop returns as soon as a step reports
# termination, so the program result is not lost.
# v03 — (kept) v02 plus the HOLE HEIGHT: the push line is lowered/raised so the peg's
# underside meets the perceived hole floor (EE z = hole floor - table + EE grasp height).
# v02 — (kept) live hole perception: when the push stalls on the box face, back off, read the wrist depth, and take the
# lateral position of the hole from the depth points that lie BEHIND the box face plane
# (the hole interior), then re-push on that line with a small search around it.
#
# The goal says: grasp the same peg at the same end and insert it into the same side of the box
# as in the video. All look-alike pegs are identified by the video itself, so the program reads
# the video's own per-frame robot state at runtime:
#   * grasp frame  = last frame before the video gripper starts closing (EE at the grasp pose)
#   * insertion    = the video's post-grasp EE path (lift, transit, descend, side push)
# Then it measures from front depth whether the layout moved between video frame 0 and the live
# scene (centroids of the peg region around the video grasp point and of the box region around
# the video's final insertion point, same method on both images).
#   * unchanged layout -> follow the video joint states tick by tick (same peg, same end, same
#     side, same IK branch)
#   * moved layout     -> shift the video EE path by the measured peg / box displacement and solve
#     IK per waypoint, seeded with the video joints of that frame.
# No scene coordinate, size or colour is a constant here; every pose comes from this episode's
# video or live depth.

report = {"version": "v06"}
context = get_task_context()
report["task"] = context.language
state0 = get_robot_state()
base_frame = state0.base_frame
embodiment = state0.embodiment


def quat_to_matrix(q):
    w = float(q[0]); x = float(q[1]); y = float(q[2]); z = float(q[3])
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def camera_model(cam):
    k = cam.intrinsics
    return {
        "fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]), "cy": float(k[1][2]),
        "R": quat_to_matrix(cam.camera_pose.quaternion_wxyz),
        "p": [float(v) for v in cam.camera_pose.position],
        "depth": cam.depth_m,
        "h": int(cam.depth_m.shape[0]), "w": int(cam.depth_m.shape[1]),
    }


def project(model, point):
    R = model["R"]; p = model["p"]
    d = [float(point[0]) - p[0], float(point[1]) - p[1], float(point[2]) - p[2]]
    cxyz = [R[0][0] * d[0] + R[1][0] * d[1] + R[2][0] * d[2],
            R[0][1] * d[0] + R[1][1] * d[1] + R[2][1] * d[2],
            R[0][2] * d[0] + R[1][2] * d[1] + R[2][2] * d[2]]
    if cxyz[2] <= 1e-6:
        return None
    return (model["fx"] * cxyz[0] / cxyz[2] + model["cx"], model["fy"] * cxyz[1] / cxyz[2] + model["cy"])


def deproject(model, u, v):
    z = float(model["depth"][v][u])
    if not (z > 0.0) or z > 5.0:
        return None
    xc = (u - model["cx"]) / model["fx"] * z
    yc = (v - model["cy"]) / model["fy"] * z
    R = model["R"]; p = model["p"]
    return (R[0][0] * xc + R[0][1] * yc + R[0][2] * z + p[0],
            R[1][0] * xc + R[1][1] * yc + R[1][2] * z + p[1],
            R[2][0] * xc + R[2][1] * yc + R[2][2] * z + p[2])


def region_centroid(model, ref, radius, z_low, z_high, window):
    """Mean of depth points within `radius` (XY) of ref and inside the z band."""
    uv = project(model, ref)
    if uv is None:
        return None
    u0 = int(round(uv[0])); v0 = int(round(uv[1]))
    sx = 0.0; sy = 0.0; sz = 0.0; count = 0
    for v in range(max(0, v0 - window), min(model["h"], v0 + window + 1)):
        for u in range(max(0, u0 - window), min(model["w"], u0 + window + 1)):
            q = deproject(model, u, v)
            if q is None:
                continue
            if q[2] < z_low or q[2] > z_high:
                continue
            dx = q[0] - ref[0]; dy = q[1] - ref[1]
            if dx * dx + dy * dy > radius * radius:
                continue
            sx += q[0]; sy += q[1]; sz += q[2]; count += 1
    if count == 0:
        return None
    return (sx / count, sy / count, sz / count, count)


# ---------------- Read the conditioning video ----------------
video = robomme_plus.get_conditioning_video()
n = len(video)
report["video_frames"] = n
if n < 10:
    raise ValueError("conditioning video missing or too short")

joints = []
ee_pos = []
ee_quat = []
grip = []
for index in range(n):
    rs = video[index].robot_state
    joints.append([float(v) for v in rs.joint_positions["primary"]])
    pose = rs.end_effector_poses["primary"]
    ee_pos.append([float(v) for v in pose.position])
    ee_quat.append([float(v) for v in pose.quaternion_wxyz])
    grip.append(float(rs.gripper_positions["primary"]))

open_level = grip[0]
close_index = None
for index in range(1, n):
    if grip[index] < open_level - 0.1:
        close_index = index
        break
if close_index is None:
    raise ValueError("video gripper never closes; cannot find the grasp frame")
grasp_index = close_index - 1
# The commanded close precedes the measured drop by a few ticks; the EE is stationary there.
report["video_grasp_index"] = grasp_index
report["video_grasp_pose"] = {"position": [round(v, 4) for v in ee_pos[grasp_index]],
                              "quaternion_wxyz": [round(v, 4) for v in ee_quat[grasp_index]]}
final_index = n - 1
report["video_final_pose"] = {"position": [round(v, 4) for v in ee_pos[final_index]],
                              "quaternion_wxyz": [round(v, 4) for v in ee_quat[final_index]]}

lift_top = grasp_index
for index in range(grasp_index, n):
    if ee_pos[index][2] > ee_pos[lift_top][2] + 1e-4:
        lift_top = index
descend_mid = final_index
mid_z = 0.5 * (ee_pos[lift_top][2] + ee_pos[final_index][2])
for index in range(lift_top, n):
    if ee_pos[index][2] < mid_z:
        descend_mid = index
        break
report["video_lift_top_index"] = lift_top
report["video_descend_mid_index"] = descend_mid

# ---------------- Did the layout move between video frame 0 and now? ----------------
live = get_observation()
live_front = camera_model(live.cameras["front"])
video_front = camera_model(video[0].cameras["front"])

grasp_ref = ee_pos[grasp_index]
final_ref = ee_pos[final_index]
peg_band = (0.002, max(0.01, grasp_ref[2] + 0.03))       # table-level band around the grasp height
box_band = (grasp_ref[2] + 0.03, final_ref[2] + 0.15)     # above lying pegs, below the arm at home

peg_video = region_centroid(video_front, grasp_ref, 0.06, peg_band[0], peg_band[1], 30)
peg_live = region_centroid(live_front, grasp_ref, 0.06, peg_band[0], peg_band[1], 30)
box_video = region_centroid(video_front, final_ref, 0.15, box_band[0], box_band[1], 45)
box_live = region_centroid(live_front, final_ref, 0.15, box_band[0], box_band[1], 45)
report["peg_region"] = {"video": peg_video, "live": peg_live}
report["box_region"] = {"video": box_video, "live": box_live}

peg_delta = (0.0, 0.0)
box_delta = (0.0, 0.0)
if peg_video is not None and peg_live is not None:
    peg_delta = (peg_live[0] - peg_video[0], peg_live[1] - peg_video[1])
if box_video is not None and box_live is not None:
    box_delta = (box_live[0] - box_video[0], box_live[1] - box_video[1])
peg_shift = (peg_delta[0] ** 2 + peg_delta[1] ** 2) ** 0.5
box_shift = (box_delta[0] ** 2 + box_delta[1] ** 2) ** 0.5
report["peg_delta"] = [round(peg_delta[0], 4), round(peg_delta[1], 4)]
report["box_delta"] = [round(box_delta[0], 4), round(box_delta[1], 4)]

# Global depth agreement (diagnostic): fraction of pixels whose depth changed by > 1 cm.
changed = 0
total = 0
for v in range(0, live_front["h"], 2):
    live_row = live_front["depth"][v]
    video_row = video_front["depth"][v]
    for u in range(0, live_front["w"], 2):
        total += 1
        if abs(float(live_row[u]) - float(video_row[u])) > 0.01:
            changed += 1
report["depth_changed_fraction"] = round(changed / max(1, total), 4)

# Live perception sanity check: is a peg actually under the video grasp point?
peg_seg = segment_text("front", "peg")
nearest_peg = None
if peg_seg.ok:
    for candidate in peg_seg.segmentations[:5]:
        cloud = mask_to_point_cloud(candidate, "front", target_frame=base_frame)
        for point in cloud.points:
            dx = float(point[0]) - grasp_ref[0]; dy = float(point[1]) - grasp_ref[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if nearest_peg is None or dist < nearest_peg:
                nearest_peg = dist
report["live_peg_nearest_to_video_grasp_m"] = None if nearest_peg is None else round(nearest_peg, 4)

SHIFT_TOL = 0.004
layout_unchanged = (peg_video is not None and peg_live is not None and box_video is not None
                    and box_live is not None and peg_shift < SHIFT_TOL and box_shift < SHIFT_TOL)
report["mode"] = "video_joint_replay" if layout_unchanged else "shifted_pose_replay"

# ---------------- Build per-frame joint targets ----------------
targets = []
if layout_unchanged:
    targets = joints
else:
    def frame_delta(index):
        if index <= grasp_index:
            a = float(index) / max(1, grasp_index)
            return (peg_delta[0] * a, peg_delta[1] * a)
        if index <= lift_top:
            return peg_delta
        if index >= descend_mid:
            return box_delta
        a = float(index - lift_top) / max(1, descend_mid - lift_top)
        return (peg_delta[0] * (1 - a) + box_delta[0] * a, peg_delta[1] * (1 - a) + box_delta[1] * a)

    samples = list(range(0, n, 4))
    if grasp_index not in samples:
        samples.append(grasp_index)
    if final_index not in samples:
        samples.append(final_index)
    samples = sorted(samples)
    solved = {}
    previous = None
    ik_failures = 0
    for index in samples:
        d = frame_delta(index)
        pose = Pose((ee_pos[index][0] + d[0], ee_pos[index][1] + d[1], ee_pos[index][2]),
                    ee_quat[index], base_frame)
        ik = solve_ik(pose, seed_joints=joints[index])
        if ik.ok:
            solved[index] = [float(v) for v in ik.joint_positions]
        else:
            ik_failures += 1
            solved[index] = None
    report["ik_failures"] = ik_failures
    known = [index for index in samples if solved[index] is not None]
    if len(known) < 2:
        raise ValueError("shifted replay: IK failed on nearly every waypoint")
    for index in range(n):
        lower = None
        upper = None
        for k in known:
            if k <= index:
                lower = k
            if k >= index and upper is None:
                upper = k
        if lower is None:
            lower = upper
        if upper is None:
            upper = lower
        if upper == lower:
            targets.append(solved[lower])
        else:
            a = float(index - lower) / float(upper - lower)
            targets.append([solved[lower][j] * (1 - a) + solved[upper][j] * a for j in range(7)])




# ---------------- Push geometry from the video ----------------
try:
    max_steps = int(robomme_plus.get_task_metadata()["max_steps"])
except Exception:
    max_steps = 800
final_z = ee_pos[final_index][2]
push_start = final_index
for index in range(descend_mid, n):
    if ee_pos[index][2] <= final_z + 0.003:
        push_start = index
        break
use_delta = box_delta if not layout_unchanged else (0.0, 0.0)
P0 = [ee_pos[push_start][0] + use_delta[0], ee_pos[push_start][1] + use_delta[1], ee_pos[push_start][2]]
PF = [ee_pos[final_index][0] + use_delta[0], ee_pos[final_index][1] + use_delta[1], ee_pos[final_index][2]]
axis = [PF[0] - P0[0], PF[1] - P0[1], PF[2] - P0[2]]
s_final = (axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2) ** 0.5
if s_final < 0.01:
    raise ValueError("video has no side push segment")
axis = [axis[0] / s_final, axis[1] / s_final, axis[2] / s_final]
lat_norm = (axis[0] ** 2 + axis[1] ** 2) ** 0.5
lateral = [-axis[1] / lat_norm, axis[0] / lat_norm, 0.0]   # horizontal, perpendicular to the push
push_quat = ee_quat[final_index]
report["video_push_start_index"] = push_start
report["push_axis"] = [round(v, 4) for v in axis]
report["push_length"] = round(s_final, 4)


# ---------------- Execute ----------------
ended = {"flag": False}
counter = {"ticks": 0}


def tick(target, gripper):
    res = step(RobotAction(arms={"primary": ArmCommand(
        "joint_position", target, gripper_position=gripper, embodiment=embodiment)}))
    counter["ticks"] += 1
    try:
        if bool(res.terminated) or bool(res.truncated):
            ended["flag"] = True
    except Exception:
        pass
    return res


last_known = {"joints": [float(v) for v in state0.joint_positions["primary"]],
              "ee": [float(v) for v in state0.end_effector_poses["primary"].position]}


def read_state():
    if ended["flag"]:
        return None
    try:
        st = get_robot_state()
    except Exception as exc:
        ended["flag"] = True
        report["state_read_error"] = str(exc)
        return None
    last_known["joints"] = [float(v) for v in st.joint_positions["primary"]]
    last_known["ee"] = [float(v) for v in st.end_effector_poses["primary"].position]
    return st


def current_joints():
    read_state()
    return list(last_known["joints"])


def joint_error(target):
    js = current_joints()
    return max(abs(js[j] - float(target[j])) for j in range(7))


def ee_now():
    read_state()
    return list(last_known["ee"])


def progress(point):
    return sum((point[k] - P0[k]) * axis[k] for k in range(3))


def push_pose(s, du, dz):
    return Pose((P0[0] + axis[0] * s + lateral[0] * du,
                 P0[1] + axis[1] * s + lateral[1] * du,
                 P0[2] + axis[2] * s + dz), push_quat, base_frame)


def ik_for(s, du, dz):
    seed = current_joints()
    ik = solve_ik(push_pose(s, du, dz), seed_joints=seed)
    if not ik.ok:
        return None
    solution = [float(v) for v in ik.joint_positions]
    if max(abs(solution[j] - seed[j]) for j in range(7)) > 0.3:
        return None   # branch jump: refuse rather than swing the arm through the box
    return solution


def glide(s, du, dz, ticks_n):
    """Joint-interpolate from the current joints to the IK solution of a push pose."""
    goal = ik_for(s, du, dz)
    if goal is None:
        return False
    start = current_joints()
    for k in range(1, ticks_n + 1):
        a = float(k) / ticks_n
        tick([start[j] * (1 - a) + goal[j] * a for j in range(7)], 0.0)
        if ended["flag"]:
            return True
    if not ended["flag"]:
        tick(goal, 0.0)
    return True
def guarded_push(du, dz, lead, budget_ticks, s_cap):
    """Push along the axis, commanding `lead` ahead of measured progress (never beyond s_cap);
    stop on stall, on episode end, or when the cap is reached."""
    history = []
    used = 0
    while used < budget_ticks and not ended["flag"]:
        s_now = progress(ee_now())
        if s_now >= s_cap - 0.001:
            return ("cap", s_now)
        target = ik_for(min(s_now + lead, s_cap), du, dz)
        if target is None:
            return ("ik_failed", s_now)
        tick(target, 0.0)
        used += 1
        if ended["flag"]:
            return ("episode_ended", s_now)
        s_after = progress(ee_now())
        history.append(s_after)
        if len(history) >= 6 and history[-1] - history[-6] < 0.0008:
            return ("stalled", s_after)
    return ("budget", progress(ee_now()))


    return ("budget", progress(ee_now()))


# Phase A: approach the video grasp pose with the gripper open.
for index in range(0, grasp_index + 1):
    tick(targets[index], 1.0)
    if ended["flag"]:
        break
if not ended["flag"]:
    for settle in range(25):
        if joint_error(targets[grasp_index]) < 0.004:
            break
        tick(targets[grasp_index], 1.0)
        if ended["flag"]:
            break
    report["grasp_joint_error"] = round(joint_error(targets[grasp_index]), 4)
    report["ee_at_grasp"] = [round(v, 4) for v in ee_now()]

# Phase B: close.
if not ended["flag"]:
    closed = close_gripper()
    try:
        counter["ticks"] += int(closed.steps_executed)
        if bool(closed.terminated) or bool(closed.truncated):
            ended["flag"] = True
    except Exception:
        pass
    if not ended["flag"]:
        st = read_state()
        if st is not None:
            report["gripper_after_close"] = round(float(st.gripper_positions["primary"]), 3)

# Phase C: follow the video's lift/transit/descent up to the start of the push, skipping
# video frames where the joints barely move (saves ticks for the guarded push).
if not ended["flag"]:
    last_sent = targets[grasp_index]
    for index in range(grasp_index + 1, push_start + 1):
        change = max(abs(float(targets[index][j]) - float(last_sent[j])) for j in range(7))
        if change < 0.002 and index != push_start:
            continue
        tick(targets[index], 0.0)
        last_sent = targets[index]
        if ended["flag"]:
            break
    for settle in range(10):
        if ended["flag"] or joint_error(targets[push_start]) < 0.003:
            break
        tick(targets[push_start], 0.0)
    if not ended["flag"]:
        st = read_state()
        if st is not None:
            report["gripper_at_push_start"] = round(float(st.gripper_positions["primary"]), 3)
        report["ticks_at_push_start"] = counter["ticks"]

# Box face plane and top along the push axis, from live front depth around the video insertion
# point (same region as the layout check).
box_s = []
box_z = []
box_lat = []
uvb = project(live_front, final_ref)
if uvb is not None:
    ub = int(round(uvb[0])); vb = int(round(uvb[1]))
    for v in range(max(0, vb - 45), min(live_front["h"], vb + 46)):
        for u in range(max(0, ub - 45), min(live_front["w"], ub + 46)):
            q = deproject(live_front, u, v)
            if q is None or q[2] < box_band[0] or q[2] > box_band[1]:
                continue
            dx = q[0] - final_ref[0]; dy = q[1] - final_ref[1]
            if dx * dx + dy * dy > 0.15 * 0.15:
                continue
            box_s.append(progress(q))
            box_z.append(q[2])
            box_lat.append(sum((q[k] - P0[k]) * lateral[k] for k in range(3)))
face_s = None
box_top = None
if len(box_s) >= 30:
    ordered = sorted(box_s)
    face_s = ordered[len(ordered) // 50]
    ordered_z = sorted(box_z)
    box_top = ordered_z[(95 * len(ordered_z)) // 100]
    ordered_lat = sorted(box_lat)
    report["box_lateral_range"] = [round(ordered_lat[len(ordered_lat) // 50], 4),
                                   round(ordered_lat[(49 * len(ordered_lat)) // 50], 4)]
report["box_face_s"] = None if face_s is None else round(face_s, 4)
report["box_top_z"] = None if box_top is None else round(box_top, 4)


def hole_points():
    """Wrist-depth points of the hole interior: behind the face plane, below the box top, above
    the table-level pegs, laterally within the box. Returns (lats, zs) or None."""
    if face_s is None or ended["flag"]:
        return None
    try:
        obs = get_observation()
    except Exception as exc:
        ended["flag"] = True
        report["observation_error"] = str(exc)
        return None
    wrist = camera_model(obs.cameras["wrist"])
    lats = []
    zs = []
    for v in range(0, wrist["h"]):
        for u in range(0, wrist["w"]):
            q = deproject(wrist, u, v)
            if q is None:
                continue
            if q[2] > box_top - 0.004 or q[2] < grasp_ref[2] + 0.015:
                continue
            sq = progress(q)
            if sq < face_s + 0.006 or sq > face_s + 0.10:
                continue
            lat_q = sum((q[k] - P0[k]) * lateral[k] for k in range(3))
            if abs(lat_q) > 0.08:
                continue
            lats.append(lat_q)
            zs.append(q[2])
    return (lats, zs)


def quantile(values, frac):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(frac * len(ordered))))]


# Table height near the peg (live front depth, points below the lying pegs' band).
table_region = region_centroid(live_front, grasp_ref, 0.10, -0.03, peg_band[0], 30)
table_z = 0.0 if table_region is None else table_region[2]
report["table_z"] = round(table_z, 4)

RETRACT_VIEW = 0.015   # back-off from the face stall before looking
RETRACT_TRY = 0.005    # back-off between insertion tries
ENTRY_GAIN = 0.010     # progress beyond the face stall that counts as "tip inside the hole"
OVERSHOOT = 0.03       # allowed push beyond the video's push depth once inside
GRID = 0.0015          # search step around the hole estimate


def entered(s_value, face_stall):
    return s_value > face_stall + ENTRY_GAIN or s_value >= s_final - 0.004


def finish_insertion(du, dz):
    """Tip is inside: keep pushing past the video depth until stall / episode end, then hold."""
    deeper = guarded_push(du, dz, 0.004, 80, s_final + OVERSHOOT)
    attempts.append({"du": round(du, 4), "dz": round(dz, 4), "result": "deep_" + deeper[0],
                     "s": round(deeper[1], 4)})
    if not ended["flag"]:
        hold = current_joints()
        for settle in range(20):
            if counter["ticks"] >= max_steps - 2:
                break
            tick(hold, 0.0)
            if ended["flag"]:
                break


# Phase D: guarded push on the video line; on a face stall, look for the hole and search.
attempts = []
inserted = False
if not ended["flag"]:
    outcome = guarded_push(0.0, 0.0, 0.004, 120, s_final + OVERSHOOT)
    attempts.append({"du": 0.0, "dz": 0.0, "result": outcome[0], "s": round(outcome[1], 4)})
    stall_s = outcome[1]
    if outcome[0] in ("stalled", "cap", "budget") and entered(stall_s, stall_s) and stall_s >= s_final - 0.004:
        inserted = True
        finish_insertion(0.0, 0.0)
    hole_du = 0.0
    hole_dz = 0.0
    median_du = None
    if not inserted and not ended["flag"]:
        glide(stall_s - RETRACT_VIEW, 0.0, 0.0, 4)
        pts = hole_points()
        view = {"du": 0.0}
        if pts is not None:
            view["count"] = len(pts[0])
        if pts is not None and len(pts[0]) >= 10:
            lo = quantile(pts[0], 0.03)
            hi = quantile(pts[0], 0.97)
            median_du = quantile(pts[0], 0.5)
            hole_du = 0.5 * (lo + hi)
            floor_z = quantile(pts[1], 0.1)
            # Peg underside rested on the table when grasped at the video grasp height; align it
            # with the hole floor plus a little clearance.
            target_z = floor_z - table_z + grasp_ref[2] + 0.0015
            hole_dz = target_z - (P0[2] + axis[2] * stall_s)
            report["hole_estimate"] = {"lat_lo": round(lo, 4), "lat_hi": round(hi, 4), "centre": round(hole_du, 4),
                                       "median": round(median_du, 4), "floor_z": round(floor_z, 4),
                                       "z_q90": round(quantile(pts[1], 0.9), 4), "count": len(pts[0])}
            report["hole_target_ee_z"] = round(target_z, 4)
        report["hole_views"] = [view]
    report["hole_offset"] = [round(hole_du, 4), round(hole_dz, 4)]

    offsets = []
    for k in (0, 1, -1, 2, -2, 3, -3, 4, -4):
        offsets.append((hole_du + GRID * k, hole_dz))
    for kz in (1, -1):
        for k in (0, 1, -1, 2, -2, 3, -3):
            offsets.append((hole_du + GRID * k, hole_dz + GRID * kz))
    for kz in (2, 3):
        for k in (0, 2, -2):
            offsets.append((hole_du + GRID * k, hole_dz + GRID * kz))
    if median_du is not None:
        offsets.append((median_du, hole_dz))
    offsets.append((0.0, hole_dz))

    current = (0.0, 0.0)
    for offset in offsets:
        if inserted or ended["flag"]:
            break
        if counter["ticks"] > max_steps - 30:
            break
        back_s = stall_s - RETRACT_TRY
        if not glide(back_s, current[0], current[1], 2):
            attempts.append({"du": round(offset[0], 4), "dz": round(offset[1], 4), "result": "ik_failed_retract"})
            continue
        if not glide(back_s, offset[0], offset[1], 3):
            attempts.append({"du": round(offset[0], 4), "dz": round(offset[1], 4), "result": "ik_failed_shift"})
            continue
        current = offset
        remaining = max_steps - 20 - counter["ticks"]
        outcome = guarded_push(offset[0], offset[1], 0.004, min(60, max(0, remaining)), s_final + OVERSHOOT)
        attempts.append({"du": round(offset[0], 4), "dz": round(offset[1], 4), "result": outcome[0],
                         "s": round(outcome[1], 4)})
        if outcome[0] == "episode_ended":
            inserted = True
        elif entered(outcome[1], stall_s):
            inserted = True
            finish_insertion(offset[0], offset[1])
report["push_attempts"] = attempts
report["inserted"] = inserted

if not ended["flag"]:
    hold = current_joints()
    for settle in range(8):
        if counter["ticks"] >= max_steps - 2:
            break
        tick(hold, 0.0)
        if ended["flag"]:
            break

report["ticks"] = counter["ticks"]
report["episode_ended"] = ended["flag"]
if not ended["flag"]:
    try:
        final_state = get_robot_state()
        report["final_ee"] = [round(float(v), 4) for v in final_state.end_effector_poses["primary"].position]
        report["final_gripper"] = round(float(final_state.gripper_positions["primary"]), 3)
        report["final_progress"] = round(progress(report["final_ee"]), 4)
    except Exception as exc:
        report["final_read_error"] = str(exc)

result = report
