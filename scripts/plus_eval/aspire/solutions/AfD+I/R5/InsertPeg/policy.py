# InsertPeg v04 -- v02's grasp/carry/PCA correction + wrist RGB-D hole localisation at contact.
#
# Changes vs v02/v03 (every slow approach stalled on the box face at the same depth; momentum
# entries wedged the tilted peg ~1.3 cm in and later impacts slid it through the grasp. Offline
# ray-casting of the v03 wrist frame at the stall onto the face plane put the visible opening
# about 1 cm to the side of, and below, the live peg tip, outside v01's +-4 mm search):
#   1. No momentum pushes. At the first stall (tip touching the face), back-project the wrist
#      depth; pixels seen through the face plane, below the box-top level, are the opening.
#      Their face-plane footprint gives the opening edges and centre (the edge hidden by the peg
#      comes from the opening height); the offset of the peg centre from the opening centre is
#      added to every later target. Re-measure up to 3 times if it stalls again.
#   2. Slow 5 mm-step insertion with stall detection, then a small slow spiral around the
#      estimate (+-3/6 mm) as fallback; hold with a gentle press at the deepest pose.
#
# --- v02 header kept below for provenance ---
# InsertPeg v02 -- video-guided insertion, PCA in-hand peg correction, momentum + wiggle insertion.
#
# Changes vs v01 (v01 ep0: every slow, small-perturbation retry stalled at the hole entrance,
# EE y=0.0499; only a deep-target push launched from 2 cm back (arm moving fast) passed the
# entrance and entered ~1.2 cm before wedging, with the hand yaw drifting back to the video's):
#   1. In-hand axis from the 3D principal component of the wrist-depth peg points (v01's
#      tip-minus-mean axis disagreed with the image-moment angle by ~2 deg). Full rotation
#      correction (yaw and pitch) maps the live peg axis onto the video's, plus a translation that
#      keeps the live tip on the video tip path. Every perturbation rotates about the peg tip.
#   2. Insertion = repeated "momentum" pushes: back off 2 cm, then drive toward a target 1 cm past
#      the video's end pose so the arm arrives with speed; while it is still making progress,
#      wiggle +-0.6 deg in yaw and pitch about the tip with deep targets; otherwise back off and
#      try the next perturbation (yaw, pitch, z, lateral).
#
# --- v01 header kept below for provenance ---
# InsertPeg v01 -- video-guided insertion + in-hand peg correction + compliant hole search.
#
# Changes vs v00 (failure mode "insertion jam": v00 ep0 reached the hole entrance at the video's
# end-effector pose (<1 mm) but the peg jammed on the box face for 480 ticks):
#   1. Pre-grasp: when the live front depth matches the video's first frame, servo to the video's
#      recorded joint configurations directly (exact expert configuration); IK otherwise.
#   2. After lifting, measure the peg tip and axis in the end-effector frame from the WRIST depth
#      image, live and in the video frame recorded at the same waypoint.  The difference (in-hand
#      yaw about the gripper axis + tip translation) corrects every later waypoint so the live tip
#      follows the video's tip path rather than the video's hand path.
#   3. Fine insertion zone: short servo calls with stall detection; on a stall, back off 2 cm and
#      retry with small perpendicular / vertical / yaw perturbations while ticks remain.
# All quantities come from the goal text, the conditioning video and live observations.

import math

report = {"version": "v04"}
context = get_task_context()
report["goal"] = context.language
state0 = get_robot_state()
base_frame = state0.base_frame

max_steps_native = 800
try:
    meta = robomme_plus.get_task_metadata()
    max_steps_native = int(meta["max_steps"])
    report["max_steps"] = max_steps_native
except Exception as exc:
    report["metadata_error"] = str(exc)


def vec(p):
    return [float(p[0]), float(p[1]), float(p[2])]


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def rot(q):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def mat_vec(m, v):
    return [m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]]


def mat_t_vec(m, v):
    return [m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
            m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
            m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2]]


def qmul(a, b):
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]


def qz(angle):
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]


def qy(angle):
    return [math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0]


def qnorm(q):
    n = math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    return [q[0] / n, q[1] / n, q[2] / n, q[3] / n]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def is_done(res):
    flag = False
    try:
        if res.terminated:
            flag = True
    except Exception:
        pass
    try:
        if res.truncated:
            flag = True
    except Exception:
        pass
    return flag


def steps_used():
    try:
        return int(round(float(get_robot_state().timestamp_s) * 20.0))
    except Exception:
        return 0


# ---------------------------------------------------------------- conditioning video
video = robomme_plus.get_conditioning_video()
n_video = len(video)
report["video_frames"] = n_video
if n_video < 10:
    raise ValueError("conditioning video missing; cannot identify peg/end/side")

positions = []
quats = []
joints = []
grips = []
for frame in video:
    rs = frame.robot_state
    pose = rs.end_effector_poses["primary"]
    positions.append(vec(pose.position))
    quats.append([float(v) for v in pose.quaternion_wxyz])
    joints.append([float(v) for v in rs.joint_positions["primary"]])
    grips.append(float(rs.gripper_positions["primary"]))

g_max = grips[0]
close_idx = None
for i in range(n_video):
    if grips[i] > g_max:
        g_max = grips[i]
    if grips[i] < g_max - 0.03:
        close_idx = i
        break
if close_idx is None:
    raise ValueError("no grasp event found in conditioning video")
grasp_idx = max(0, close_idx - 1)
g_min = grips[close_idx]
open_idx = None
for i in range(close_idx, n_video):
    if grips[i] < g_min:
        g_min = grips[i]
    if grips[i] > g_min + 0.08:
        open_idx = i
        break
end_idx = n_video - 1 if open_idx is None else max(close_idx, open_idx - 1)
report["video_events"] = {"close_idx": close_idx, "open_idx": open_idx, "end_idx": end_idx,
                          "grasp_pos": [round(v, 4) for v in positions[grasp_idx]],
                          "end_pos": [round(v, 4) for v in positions[end_idx]]}


def sample_path(start, stop, coarse, fine, fine_zone):
    remaining = [0.0] * (stop - start + 1)
    acc = 0.0
    for j in range(stop - 1, start - 1, -1):
        acc = acc + dist(positions[j], positions[j + 1])
        remaining[j - start] = acc
    picked = []
    last = None
    for j in range(start, stop + 1):
        spacing = fine if remaining[j - start] <= fine_zone else coarse
        if last is None:
            if j > start:
                picked.append(j)
                last = j
            continue
        if dist(positions[j], positions[last]) >= spacing:
            picked.append(j)
            last = j
    if not picked or picked[-1] != stop:
        picked.append(stop)
    return picked


# ---------------------------------------------------------------- live layout check
live = get_observation()
same_layout = False
layout = {}
try:
    vd = video[0].cameras["front"].depth_m
    ld = live.cameras["front"].depth_m
    h = len(ld)
    w = len(ld[0])
    changed = 0
    total = 0
    for v in range(0, h, 2):
        vrow = vd[v]
        lrow = ld[v]
        for u in range(0, w, 2):
            a = float(vrow[u])
            b = float(lrow[u])
            if a > 0.0 and b > 0.0 and a < 5.0 and b < 5.0:
                total = total + 1
                if abs(a - b) > 0.004:
                    changed = changed + 1
    layout["checked"] = total
    layout["changed"] = changed
    same_layout = total > 0 and changed <= max(3, total // 1000)
except Exception as exc:
    layout["error"] = str(exc)
layout["same_layout"] = same_layout
report["layout_check"] = layout


# ---------------------------------------------------------------- wrist in-hand measurement
def measure_peg(obs_cam, ee_pos, ee_quat):
    """Peg tip and principal axis in the end-effector frame from a wrist depth image."""
    depth = obs_cam.depth_m
    K = obs_cam.intrinsics
    fx = float(K[0][0])
    fy = float(K[1][1])
    cx = float(K[0][2])
    cy = float(K[1][2])
    cam_r = rot([float(v) for v in obs_cam.camera_pose.quaternion_wxyz])
    cam_t = vec(obs_cam.camera_pose.position)
    hh = len(depth)
    ww = len(depth[0])
    vmax = int(0.70 * hh)
    pix = []
    dmin = None
    for v in range(0, vmax, 2):
        row = depth[v]
        for u in range(0, ww, 2):
            d = float(row[u])
            if d > 0.01 and d < 3.0:
                pix.append((u, v, d))
                if dmin is None or d < dmin:
                    dmin = d
    if dmin is None:
        return None
    thr = dmin + 0.035
    sel = [p for p in pix if p[2] < thr]
    if len(sel) < 30:
        return None
    n = float(len(sel))
    mu = sum(p[0] for p in sel) / n
    mv = sum(p[1] for p in sel) / n
    suu = sum((p[0] - mu) ** 2 for p in sel) / n
    svv = sum((p[1] - mv) ** 2 for p in sel) / n
    suv = sum((p[0] - mu) * (p[1] - mv) for p in sel) / n
    theta = 0.5 * math.atan2(2.0 * suv, suu - svv)
    eu = math.cos(theta)
    ev = math.sin(theta)
    if ev > 0:
        eu = -eu
        ev = -ev
    proj = [(p[0] - mu) * eu + (p[1] - mv) * ev for p in sel]
    smax = max(proj)
    tip_pts = []
    all_pts = []
    for k in range(len(sel)):
        p = sel[k]
        pc = [(p[0] - cx) / fx * p[2], (p[1] - cy) / fy * p[2], p[2]]
        pw = mat_vec(cam_r, pc)
        pw = [pw[0] + cam_t[0], pw[1] + cam_t[1], pw[2] + cam_t[2]]
        all_pts.append(pw)
        if proj[k] >= smax - 5.0:
            tip_pts.append(pw)
    tip = [sum(p[i] for p in tip_pts) / len(tip_pts) for i in range(3)]
    mid = [sum(p[i] for p in all_pts) / len(all_pts) for i in range(3)]
    guess = [tip[0] - mid[0], tip[1] - mid[1], tip[2] - mid[2]]
    gn = math.sqrt(dot(guess, guess))
    if gn < 1e-6:
        return None
    # 3D principal axis (power iteration on the covariance), oriented toward the tip
    cov = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for pw in all_pts:
        d0 = [pw[0] - mid[0], pw[1] - mid[1], pw[2] - mid[2]]
        for a in range(3):
            for b in range(3):
                cov[a][b] = cov[a][b] + d0[a] * d0[b]
    axis = [guess[0] / gn, guess[1] / gn, guess[2] / gn]
    for it in range(40):
        nxt = mat_vec(cov, axis)
        nn = math.sqrt(dot(nxt, nxt))
        if nn < 1e-12:
            break
        axis = [nxt[0] / nn, nxt[1] / nn, nxt[2] / nn]
    if dot(axis, guess) < 0:
        axis = [-axis[0], -axis[1], -axis[2]]
    ee_r = rot(ee_quat)
    rel = [tip[0] - ee_pos[0], tip[1] - ee_pos[1], tip[2] - ee_pos[2]]
    ee_cam = mat_t_vec(cam_r, [ee_pos[0] - cam_t[0], ee_pos[1] - cam_t[1], ee_pos[2] - cam_t[2]])
    # peg width: spread across the axis in the horizontal plane (uniform -> std * sqrt(12))
    e = cross(axis, [0.0, 0.0, 1.0])
    en = math.sqrt(dot(e, e))
    width = 0.0
    if en > 1e-6:
        e = [e[0] / en, e[1] / en, e[2] / en]
        var = sum(dot([pw[0] - mid[0], pw[1] - mid[1], pw[2] - mid[2]], e) ** 2 for pw in all_pts) / len(all_pts)
        width = math.sqrt(12.0 * var)
    return {"tip_ee": mat_t_vec(ee_r, rel), "width": round(width, 4), "axis_ee": mat_t_vec(ee_r, axis),
            "axis_tipmid_ee": mat_t_vec(ee_r, [guess[0] / gn, guess[1] / gn, guess[2] / gn]),
            "n": len(sel), "theta_deg": round(math.degrees(theta), 2), "dmin": round(dmin, 4),
            "ee_cam_z": round(ee_cam[2], 4), "tip_world": [round(x, 4) for x in tip]}


correction = {"valid": False, "qm": [1.0, 0.0, 0.0, 0.0], "r_v": [0.0, 0.0, 0.0], "r_l": [0.0, 0.0, 0.0]}


def corrected_pose(idx, dpos, dyaw, dpitch):
    """Video pose at idx, corrected so the live peg tip/axis follow the video's; dyaw/dpitch
    rotate about the peg tip (EE z / EE y axes); dpos is a world translation."""
    p = positions[idx]
    q = quats[idx]
    q2 = qnorm(qmul(qmul(qmul(q, correction["qm"]), qz(dyaw)), qy(dpitch)))
    a = mat_vec(rot(q), correction["r_v"])
    b = mat_vec(rot(q2), correction["r_l"])
    p2 = [p[i] + a[i] - b[i] + dpos[i] for i in range(3)]
    return p2, q2


# ---------------------------------------------------------------- motion helpers
log = []
terminal = {"done": False}


def servo(joint_target, tol, max_steps, entry):
    res = move_to_joints(joint_target, tolerance=tol, max_steps=max_steps)
    try:
        entry["ok"] = bool(res.ok)
        entry["steps"] = int(res.steps_executed)
    except Exception:
        pass
    if is_done(res):
        terminal["done"] = True
        entry["terminal"] = True
    log.append(entry)
    return res


def goto_video(idx, tol, max_steps, tag):
    entry = {"tag": tag, "idx": idx, "mode": "video_joints"}
    return servo(joints[idx], tol, max_steps, entry)


def ik_for(pos, quat, seed):
    try:
        ik = solve_ik(Pose(pos, quat, base_frame), seed_joints=seed)
        if ik.ok:
            return [float(v) for v in ik.joint_positions]
    except Exception:
        pass
    return None


def ee_now():
    return vec(get_robot_state().end_effector_poses["primary"].position)


def goto_pose(idx, dpos, dyaw, dpitch, tol, max_steps, tag):
    pos, quat = corrected_pose(idx, dpos, dyaw, dpitch)
    entry = {"tag": tag, "idx": idx}
    target = ik_for(pos, quat, joints[idx])
    entry["mode"] = "ik"
    if target is None:
        target = joints[idx]
        entry["mode"] = "video_joints_fallback"
    servo(target, tol, max_steps, entry)
    try:
        if not terminal["done"]:
            actual = ee_now()
            entry["err_mm"] = round(1000.0 * dist(actual, pos), 1)
            return dist(actual, pos)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------- execute
DEG = math.pi / 180.0
try:
    open_gripper()
    pre_path = sample_path(0, grasp_idx, 0.04, 0.01, 0.06)
    for idx in pre_path:
        if terminal["done"]:
            break
        if same_layout:
            goto_video(idx, 0.005 if idx == grasp_idx else 0.02, 120, "approach")
        else:
            goto_pose(idx, [0.0, 0.0, 0.0], 0.0, 0.0, 0.005 if idx == grasp_idx else 0.02, 120, "approach")

    if not terminal["done"]:
        cg = close_gripper()
        log.append({"tag": "close", "grip": round(float(get_robot_state().gripper_positions["primary"]), 3)})
        if is_done(cg):
            terminal["done"] = True

    post_path = sample_path(grasp_idx, end_idx, 0.04, 0.005, 0.07)
    fine_zone = 0.07
    measure_at = None
    for idx in post_path:
        if positions[idx][2] >= positions[grasp_idx][2] + 0.08:
            measure_at = idx
            break
    report["measure_idx"] = measure_at

    fine_list = [idx for idx in post_path if dist(positions[idx], positions[end_idx]) <= fine_zone]
    first_fine = fine_list[0] if fine_list else end_idx
    coarse_list = [idx for idx in post_path if idx < first_fine]

    for idx in coarse_list:
        if terminal["done"]:
            break
        goto_pose(idx, [0.0, 0.0, 0.0], 0.0, 0.0, 0.02, 120, "carry")
        if measure_at is not None and idx == measure_at and not terminal["done"]:
            try:
                st = get_robot_state()
                live_m = measure_peg(get_observation(["wrist"]).cameras["wrist"],
                                     vec(st.end_effector_poses["primary"].position),
                                     [float(v) for v in st.end_effector_poses["primary"].quaternion_wxyz])
                vid_m = measure_peg(video[idx].cameras["wrist"], positions[idx], quats[idx])
                report["inhand_live"] = live_m
                report["inhand_video"] = vid_m
                if live_m is not None and vid_m is not None:
                    al = live_m["axis_ee"]
                    av = vid_m["axis_ee"]
                    cosang = max(-1.0, min(1.0, dot(al, av)))
                    ang = math.acos(cosang)
                    shift = dist(live_m["tip_ee"], vid_m["tip_ee"])
                    report["inhand_axis_angle_deg"] = round(math.degrees(ang), 2)
                    report["inhand_yaw_deg"] = round(math.degrees(math.atan2(al[1], al[0]) - math.atan2(av[1], av[0])), 2)
                    report["inhand_pitch_deg"] = round(math.degrees(math.asin(max(-1.0, min(1.0, al[2]))) - math.asin(max(-1.0, min(1.0, av[2])))), 2)
                    report["inhand_theta_diff_deg"] = round(live_m["theta_deg"] - vid_m["theta_deg"], 2)
                    report["inhand_tip_shift_mm"] = round(1000.0 * shift, 1)
                    if ang < 8.0 * DEG and shift < 0.025 and live_m["ee_cam_z"] > 0 and vid_m["ee_cam_z"] > 0:
                        c = cross(al, av)
                        correction["qm"] = qnorm([1.0 + cosang, c[0], c[1], c[2]])
                        correction["r_v"] = vid_m["tip_ee"]
                        correction["r_l"] = live_m["tip_ee"]
                        correction["valid"] = True
                    elif vid_m is not None:
                        correction["r_v"] = vid_m["tip_ee"]
                        correction["r_l"] = vid_m["tip_ee"]
            except Exception as exc:
                report["measure_error"] = str(exc)
    report["correction"] = {"valid": correction["valid"], "qm": [round(x, 5) for x in correction["qm"]],
                            "r_v": [round(x, 4) for x in correction["r_v"]],
                            "r_l": [round(x, 4) for x in correction["r_l"]]}

    # insertion axis (video's final approach) and lateral direction
    a0 = positions[first_fine]
    a1 = positions[end_idx]
    ax = [a1[0] - a0[0], a1[1] - a0[1], a1[2] - a0[2]]
    an = math.sqrt(dot(ax, ax))
    ax = [ax[0] / an, ax[1] / an, ax[2] / an] if an > 1e-6 else [1.0, 0.0, 0.0]
    lat = [-ax[1], ax[0], 0.0]
    ln = math.sqrt(dot(lat, lat))
    lat = [lat[0] / ln, lat[1] / ln, 0.0] if ln > 1e-6 else [0.0, 1.0, 0.0]
    p_end0 = corrected_pose(end_idx, [0.0, 0.0, 0.0], 0.0, 0.0)[0]

    def s_now():
        e = ee_now()
        return dot([e[0] - p_end0[0], e[1] - p_end0[1], e[2] - p_end0[2]], ax)

    def push_to(s_off, pert, wy, wp, tol, max_steps, tag):
        pl, pz, pyaw, ppitch = pert
        dpos = [ax[i] * s_off + lat[i] * pl for i in range(3)]
        dpos[2] = dpos[2] + pz
        pos, quat = corrected_pose(end_idx, dpos, pyaw + wy, ppitch + wp)
        tgt = ik_for(pos, quat, joints[end_idx])
        entry = {"tag": tag, "s_cmd": round(s_off, 3)}
        if tgt is None:
            entry["ik"] = "fail"
            log.append(entry)
            return None
        servo(tgt, tol, max_steps, entry)
        if terminal["done"]:
            return None
        sv = s_now()
        entry["s"] = round(sv, 4)
        return sv

    # Phase A: follow the video's fine insertion waypoints until the first stall
    s_stall = None
    for k in range(len(fine_list)):
        if terminal["done"]:
            break
        err = goto_pose(fine_list[k], [0.0, 0.0, 0.0], 0.0, 0.0, 0.004, 30, "insert")
        if err > 0.006:
            s_stall = s_now()
            break
    report["s_stall"] = None if s_stall is None else round(s_stall, 4)
    best = {"s": -1.0 if s_stall is None else s_stall, "pert": (0.0, 0.0, 0.0, 0.0)}
    if not terminal["done"]:
        cur = s_now()
        if cur > best["s"]:
            best["s"] = cur
    s_entr = s_stall if s_stall is not None else -0.036
    finished = (not terminal["done"]) and best["s"] >= -0.003

    # ------------------------------------------------------------ hole localisation (wrist RGB-D)
    peg_w = 0.02
    try:
        if report["inhand_live"] is not None and report["inhand_live"]["width"] > 0.005:
            peg_w = float(report["inhand_live"]["width"])
    except Exception:
        pass
    r_tip = correction["r_l"]

    def tip_world_now():
        st = get_robot_state()
        p = vec(st.end_effector_poses["primary"].position)
        q = [float(v) for v in st.end_effector_poses["primary"].quaternion_wxyz]
        a = mat_vec(rot(q), r_tip)
        return [p[0] + a[0], p[1] + a[1], p[2] + a[2]]

    def locate_hole():
        """At contact, back-project the wrist depth. Pixels seen *through* the face plane (deeper
        than the plane through the touching peg tip, below the box-top level) are the opening.
        Their face-plane footprint gives the opening edges; the edge hidden by the peg is
        inferred from the opening height (square opening)."""
        tipw = tip_world_now()
        cam = get_observation(["wrist"]).cameras["wrist"]
        depth = cam.depth_m
        K = cam.intrinsics
        fx = float(K[0][0])
        fy = float(K[1][1])
        cx = float(K[0][2])
        cy = float(K[1][2])
        cr = rot([float(v) for v in cam.camera_pose.quaternion_wxyz])
        ct = vec(cam.camera_pose.position)
        hh = len(depth)
        ww = len(depth[0])
        nd = [ax[0], ax[1], 0.0]
        nn = math.sqrt(nd[0] ** 2 + nd[1] ** 2)
        nd = [nd[0] / nn, nd[1] / nn, 0.0]
        side = [-nd[1], nd[0], 0.0]
        face_c = dot(tipw, nd)
        pts = []
        zmin_all = None
        for v in range(0, hh, 2):
            row = depth[v]
            for u in range(0, ww, 2):
                d = float(row[u])
                if d <= 0.01 or d > 2.0:
                    continue
                ray = mat_vec(cr, [(u - cx) / fx, (v - cy) / fy, 1.0])
                pw = [ct[0] + ray[0] * d, ct[1] + ray[1] * d, ct[2] + ray[2] * d]
                if zmin_all is None or pw[2] < zmin_all:
                    zmin_all = pw[2]
                pts.append((u, v, d, ray, pw))
        if not pts:
            return None
        # box-top level: highest points just behind the face plane, near the tip laterally
        tops = []
        for (u, v, d, ray, pw) in pts:
            depth_in = dot(pw, nd) - face_c
            if depth_in > 0.004 and depth_in < 0.08 and abs(dot(pw, side) - dot(tipw, side)) < 0.06:
                tops.append(pw[2])
        if len(tops) < 10:
            return None
        tops = sorted(tops)
        z_top = tops[int(0.9 * (len(tops) - 1))]
        # refine the face plane: median offset of the dominant surface around the tip
        offs = []
        for (u, v, d, ray, pw) in pts:
            depth_in = dot(pw, nd) - face_c
            if depth_in > -0.015 and depth_in < 0.025 and pw[2] > zmin_all + 0.01 and pw[2] < z_top - 0.01:
                if abs(dot(pw, side) - dot(tipw, side)) < 0.04:
                    offs.append(depth_in)
        face_shift = 0.0
        if len(offs) > 20:
            offs = sorted(offs)
            face_shift = offs[len(offs) // 2]
            if abs(face_shift) < 0.015:
                face_c = face_c + face_shift
        fx_list = []
        fz_list = []
        pz_list = []
        peg_side = []
        for (u, v, d, ray, pw) in pts:
            depth_in = dot(pw, nd) - face_c
            rd = dot(ray, nd)
            if abs(rd) < 1e-6:
                continue
            t = (face_c - dot(ct, nd)) / rd
            if t <= 0:
                continue
            fp = [ct[0] + ray[0] * t, ct[1] + ray[1] * t, ct[2] + ray[2] * t]
            lat_f = dot(fp, side) - dot(tipw, side)
            if abs(lat_f) > 0.04 or fp[2] > z_top - 0.003 or fp[2] < tipw[2] - 0.05:
                continue
            if depth_in < -0.004:
                # in front of the face: peg / gripper pixels (occluders) near the tip height
                if abs(pw[2] - (tipw[2] - peg_w / 2.0)) < peg_w and depth_in > -0.06 and d < t:
                    peg_side.append(lat_f)
                continue
            if depth_in > 0.008 and pw[2] < z_top - 0.008 and pw[2] > zmin_all + 0.006:
                fx_list.append(lat_f)
                fz_list.append(fp[2])
                pz_list.append(pw[2])
        info = {"n_open": len(fx_list), "z_top": round(z_top, 4), "tip": [round(x, 4) for x in tipw],
                "face_shift": round(face_shift, 4), "n_face": len(offs)}
        if len(fx_list) < 6:
            return info
        fx_s = sorted(fx_list)
        fz_s = sorted(fz_list)
        k5 = int(0.05 * (len(fx_s) - 1))
        k95 = int(0.95 * (len(fx_s) - 1))
        lo = fx_s[k5]
        hi = fx_s[k95]
        pz_s = sorted(pz_list)
        # rays through the upper part of the opening give its top edge on the face plane; the
        # lowest interior points seen through it lie on the opening's floor
        zl = pz_s[k5]
        zh = fz_s[k95]
        size = zh - zl
        info.update({"lat_lo": round(lo, 4), "lat_hi": round(hi, 4), "z_lo": round(zl, 4), "z_hi": round(zh, 4)})
        if size < peg_w * 0.8 or size > 0.06:
            size = max(peg_w * 1.2, min(0.06, max(size, hi - lo)))
        p_lo = min(peg_side) if peg_side else 0.0
        p_hi = max(peg_side) if peg_side else 0.0
        info.update({"peg_lo": round(p_lo, 4), "peg_hi": round(p_hi, 4), "size": round(size, 4)})
        width_seen = hi - lo
        # the peg (centred on lateral 0, closer to the camera) hides the opening edge nearer to it
        if width_seen < size - 0.004:
            if abs(hi) < abs(lo):
                lat_c = lo + size / 2.0
            else:
                lat_c = hi - size / 2.0
        else:
            lat_c = (lo + hi) / 2.0
        z_c = (zl + zh) / 2.0
        # peg tip reference is the top-surface tip; peg centre is half a width below it
        d_lat = lat_c
        d_z = z_c + peg_w / 2.0 - tipw[2]
        info["d_lat"] = round(d_lat, 4)
        info["d_z"] = round(d_z, 4)
        info["ok"] = abs(d_lat) < 0.025 and abs(d_z) < 0.025
        info["side"] = [round(x, 4) for x in side]
        return info

    def target_s(s_off, dw):
        return [ax[i] * s_off + dw[i] for i in range(3)]

    def go_s(s_off, dw, tol, max_steps, tag):
        pos, quat = corrected_pose(end_idx, target_s(s_off, dw), 0.0, 0.0)
        tgt = ik_for(pos, quat, joints[end_idx])
        entry = {"tag": tag, "s_cmd": round(s_off, 4)}
        if tgt is None:
            entry["ik"] = "fail"
            log.append(entry)
            return None
        servo(tgt, tol, max_steps, entry)
        if terminal["done"]:
            return None
        sv = s_now()
        entry["s"] = round(sv, 4)
        return sv

    def slow_insert(dw, s_from, s_goal):
        """5 mm steps along the insertion axis; returns (reached_s, stalled)."""
        s_cmd = s_from
        sv = s_now()
        while (not terminal["done"]) and steps_used() < max_steps_native - 15:
            s_cmd = min(s_goal, s_cmd + 0.005)
            r = go_s(s_cmd, dw, 0.003, 12, "slow")
            if r is None:
                return sv, True
            sv = r
            if sv < s_cmd - 0.006:
                return sv, True
            if s_cmd >= s_goal:
                return sv, False
        return sv, True

    holes = []
    dw = [0.0, 0.0, 0.0]
    s_goal = 0.004
    inserted = False
    if (not finished) and (not terminal["done"]):
        for rnd in range(3):
            if terminal["done"] or steps_used() > max_steps_native - 60:
                break
            info = None
            try:
                info = locate_hole()
            except Exception as exc:
                info = {"error": str(exc)}
            holes.append(info)
            if info is None or not info.get("ok", False):
                break
            sd = info["side"]
            ndw = [dw[0] + sd[0] * info["d_lat"], dw[1] + sd[1] * info["d_lat"], dw[2] + info["d_z"]]
            if math.sqrt(ndw[0] ** 2 + ndw[1] ** 2) > 0.025 or abs(ndw[2]) > 0.025:
                info["rejected"] = "cumulative offset too large"
                break
            dw = ndw
            go_s(s_entr - 0.012, [dw[0], dw[1], dw[2] + 0.003], 0.003, 15, "retreat_shift")
            go_s(s_entr - 0.012, dw, 0.003, 10, "align")
            sv, stalled = slow_insert(dw, s_entr - 0.012, s_goal)
            if sv > best["s"]:
                best["s"] = sv
                best["pert"] = (dw[0], dw[1], dw[2], 0.0)
            if not stalled or sv >= s_goal - 0.004:
                inserted = True
                break
            if sv > s_entr + 0.006:
                # entered but stuck deeper: short re-seat then continue
                go_s(sv - 0.004, dw, 0.003, 8, "reseat")
                sv, stalled = slow_insert(dw, sv - 0.004, s_goal)
                best["s"] = max(best["s"], sv)
                if not stalled:
                    inserted = True
                break
            s_entr = max(s_entr, sv) if sv is not None else s_entr
    report["holes"] = holes
    report["hole_offset_world"] = [round(x, 4) for x in dw]

    # fallback: small spiral around the current estimate, slow approach only
    spiral = [(0.003, 0.0), (-0.003, 0.0), (0.0, -0.003), (0.0, 0.003), (0.003, -0.003),
              (-0.003, -0.003), (0.003, 0.003), (-0.003, 0.003), (0.006, 0.0), (-0.006, 0.0)]
    probes = []
    if (not inserted) and (not finished) and (not terminal["done"]):
        side_v = [-ax[1], ax[0], 0.0]
        sn = math.sqrt(side_v[0] ** 2 + side_v[1] ** 2)
        side_v = [side_v[0] / sn, side_v[1] / sn, 0.0]
        for (pl, pz) in spiral:
            if terminal["done"] or steps_used() > max_steps_native - 40:
                break
            dwp = [dw[0] + side_v[0] * pl, dw[1] + side_v[1] * pl, dw[2] + pz]
            go_s(s_entr - 0.008, dwp, 0.003, 10, "probe_back")
            sv = go_s(s_entr + 0.008, dwp, 0.003, 12, "probe_in")
            if sv is None:
                break
            probes.append([pl, pz, round(sv, 4)])
            if sv > s_entr + 0.004:
                sv2, stalled = slow_insert(dwp, sv, s_goal)
                best["s"] = max(best["s"], sv2)
                best["pert"] = (dwp[0], dwp[1], dwp[2], 0.0)
                if not stalled:
                    inserted = True
                break
    report["probes"] = probes
    report["inserted_flag"] = inserted
    report["best_s"] = round(best["s"], 4)

    # Hold: press gently at the deepest reached pose for the remaining ticks
    if not terminal["done"]:
        st = get_robot_state()
        dwb = [best["pert"][0], best["pert"][1], best["pert"][2]]
        pos, quat = corrected_pose(end_idx, target_s(min(best["s"] + 0.004, s_goal), dwb), 0.0, 0.0)
        tgt = ik_for(pos, quat, joints[end_idx])
        if tgt is None:
            tgt = st.joint_positions["primary"]
        n_hold = max(3, min(15, max_steps_native - steps_used() - 2))
        for tick in range(n_hold):
            sr = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=tgt, gripper_position=0.0, embodiment=st.embodiment)}))
            if is_done(sr):
                terminal["done"] = True
                break
        if not terminal["done"]:
            report["s_final"] = round(s_now(), 4)
except Exception as exc:
    report["error"] = str(exc)

report["motion_log"] = log
report["terminal_seen"] = terminal["done"]
try:
    if not terminal["done"]:
        fin = get_robot_state()
        report["final_ee"] = [round(v, 4) for v in vec(fin.end_effector_poses["primary"].position)]
        report["steps_used"] = steps_used()
except Exception:
    pass

result = report
