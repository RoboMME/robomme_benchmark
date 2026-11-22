# v01 — MoveCube: imitate the manner shown in the conditioning video.
# v01 vs v00 (task-predicate mode: v00 left the cube ~1 cm short of the target centre with the
# hook still pressed against it): closed-loop residual pull measured from a fresh cube
# localization, then lift the tool clear (as the video's demonstrator does) and hold still.
#
# 1. Classify the video's manner from its recorded robot states only
#    (gripper closing / holding width / low-height sweeps while holding):
#      "tool"       : grasp something, lift, later sweep along the table while holding it
#      "pick_place" : grasp, carry at height, release
#      "push"       : never hold an object (gripper open throughout or closed empty)
# 2. Localize cube and target (and the tool) in the live front camera.
# 3. Execute the same manner. Tool manner: grasp the tool handle, rotate the tool so its shaft
#    points from the target past the cube, lower the hook arm behind the cube, pull straight
#    to the target.
import math

report = {"version": "v01", "events": []}
BASE = get_robot_state().base_frame
CAM = "front"


def note(msg):
    report["events"].append(str(msg))


def r4(value):
    return round(float(value), 4)


# ==== GEOMETRY START ====
def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def wrap_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def principal_axis_xy(points):
    n = len(points)
    mx = sum(float(p[0]) for p in points) / n
    my = sum(float(p[1]) for p in points) / n
    sxx = sum((float(p[0]) - mx) ** 2 for p in points) / n
    syy = sum((float(p[1]) - my) ** 2 for p in points) / n
    sxy = sum((float(p[0]) - mx) * (float(p[1]) - my) for p in points) / n
    angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    return (mx, my), (math.cos(angle), math.sin(angle))


def tool_model(points):
    """Shaft axis, handle end, hook-arm side/width/position from a tool point cloud (base frame).

    Returns a dict with: origin (xy of handle end on shaft line), axis (unit, handle->hook),
    normal (unit, toward the hook arm side), length, arm_u, arm_w, handle_top_z, grasp_u.
    """
    pts = [(float(p[0]), float(p[1]), float(p[2])) for p in points]
    center, axis = principal_axis_xy(pts)
    normal = (-axis[1], axis[0])
    us = [(p[0] - center[0]) * axis[0] + (p[1] - center[1]) * axis[1] for p in pts]
    vs = [(p[0] - center[0]) * normal[0] + (p[1] - center[1]) * normal[1] for p in pts]
    u_lo = quantile(us, 0.01)
    u_hi = quantile(us, 0.99)
    length = u_hi - u_lo
    end_span = 0.2 * length
    low_idx = [i for i in range(len(pts)) if us[i] < u_lo + end_span]
    high_idx = [i for i in range(len(pts)) if us[i] > u_hi - end_span]
    mid_idx = [i for i in range(len(pts)) if u_lo + 0.35 * length < us[i] < u_lo + 0.75 * length]
    v_mid = quantile([vs[i] for i in mid_idx], 0.5) if len(mid_idx) > 5 else 0.0
    def end_stats(idx):
        if len(idx) == 0:
            return {"z90": 0.0, "spread": 0.0, "n": 0}
        z90 = quantile([pts[i][2] for i in idx], 0.9)
        dv = [abs(vs[i] - v_mid) for i in idx]
        return {"z90": z90, "spread": quantile(dv, 0.9), "n": len(idx)}
    low = end_stats(low_idx)
    high = end_stats(high_idx)
    # Handle = thicker (taller) end; hook end = wider lateral spread (the L arm).
    vote = 0.0
    vote += (low["z90"] - high["z90"]) / 0.004
    vote += (high["spread"] - low["spread"]) / 0.01
    handle_is_low = vote >= 0.0
    if not handle_is_low:
        axis = (-axis[0], -axis[1])
        normal = (-normal[0], -normal[1])
        us = [-u for u in us]
        vs = [-v for v in vs]
        v_mid = -v_mid
        u_lo, u_hi = -u_hi, -u_lo
    # tool coordinates: u from handle end, v from the shaft line
    tu = [u - u_lo for u in us]
    tv = [v - v_mid for v in vs]
    hook_idx = [i for i in range(len(pts)) if tu[i] > length - 0.07]
    off = [tv[i] for i in hook_idx if abs(tv[i]) > 0.012]
    side = 1.0
    if len(off) > 0 and quantile(off, 0.5) < 0.0:
        side = -1.0
    arm_pts = [i for i in hook_idx if side * tv[i] > 0.012]
    arm_w = quantile([side * tv[i] for i in arm_pts], 0.95) if len(arm_pts) > 2 else 0.04
    arm_u = quantile([tu[i] for i in arm_pts], 0.5) if len(arm_pts) > 2 else length - 0.008
    handle_idx = [i for i in range(len(pts)) if tu[i] < 0.07]
    handle_top = quantile([pts[i][2] for i in handle_idx], 0.95) if len(handle_idx) > 0 else 0.03
    # world xy of tool coordinate (u, v):  center + axis*(u + u_lo) + normal*(v + v_mid)
    # (u_lo/v_mid are in the possibly-flipped centered frame)
    model = {
        "center": center, "axis": axis, "normal": (normal[0] * side, normal[1] * side),
        "u_lo": u_lo, "v_mid": v_mid, "side": side, "length": length,
        "arm_u": arm_u, "arm_w": arm_w, "handle_top_z": handle_top,
        "grasp_u": min(0.035, 0.12 * length), "low": low, "high": high,
        "handle_is_low": handle_is_low, "n": len(pts),
    }
    return model


def tool_point_world(model, u, v):
    """World xy of tool coordinate (u from handle end along axis, v toward hook-arm side)."""
    cx, cy = model["center"]
    ax, ay = model["axis"]
    nx, ny = model["normal"]
    side = model["side"]
    uu = u + model["u_lo"]
    # normal stored already multiplied by side; v_mid lives in the unsigned normal frame
    vv = v + side * model["v_mid"]
    return (cx + ax * uu + nx * vv, cy + ay * uu + ny * vv)


def plan_hook_pull(model, cube_xy, cube_half, target_xy, phi=0.0, back_margin=0.03, gap=0.006):
    """Gripper start/end xy and yaws for pulling the cube to the target with the hook arm.

    phi tilts the shaft away from the pull line (radians); phi=0 pulls along the shaft.
    """
    dx = target_xy[0] - cube_xy[0]
    dy = target_xy[1] - cube_xy[1]
    dist = math.sqrt(dx * dx + dy * dy)
    d = (dx / dist, dy / dist)
    a = model["axis"]
    theta = wrap_angle(math.atan2(-d[1], -d[0]) + phi - math.atan2(a[1], a[0]))
    c, s = math.cos(theta), math.sin(theta)
    n = model["normal"]
    a_rot = (c * a[0] - s * a[1], s * a[0] + c * a[1])
    n_rot = (c * n[0] - s * n[1], s * n[0] + c * n[1])
    m_v = d[0] * n_rot[0] + d[1] * n_rot[1]
    u_g = model["grasp_u"]
    v_g = 0.0
    arm_half_thick = 0.005
    u_contact = model["arm_u"] - arm_half_thick - cube_half
    u_c = u_contact - gap
    v_c = max(0.5 * model["arm_w"], cube_half + 0.014)
    v_end = v_c if abs(phi) < 1e-6 else cube_half + 0.006
    def gripper_for(cube_u, cube_v, world_xy):
        return (world_xy[0] - a_rot[0] * (cube_u - u_g) - n_rot[0] * (cube_v - v_g),
                world_xy[1] - a_rot[1] * (cube_u - u_g) - n_rot[1] * (cube_v - v_g))
    s0 = gripper_for(u_c, v_c, cube_xy)
    start = (s0[0] - d[0] * back_margin, s0[1] - d[1] * back_margin)
    end = gripper_for(u_contact, v_end, target_xy)
    axis_angle = math.atan2(a[1], a[0])
    best = None
    for psi0 in (axis_angle, axis_angle + math.pi):
        p0 = wrap_angle(psi0)
        p1 = p0 + theta
        cost = max(abs(p0), abs(p1))
        if best is None or cost < best[0]:
            best = (cost, p0, p1)
    grasp_xy = tool_point_world(model, u_g, v_g)
    return {"start": start, "end": end, "theta": theta, "yaw_grasp": best[1], "yaw_pull": best[2],
            "dir": d, "dist": dist, "grasp_xy": grasp_xy, "v_c": v_c, "u_c": u_c, "m_v": m_v, "phi": phi}


def topdown_quat(yaw):
    return (0.0, math.cos(0.5 * yaw), math.sin(0.5 * yaw), 0.0)


def classify_video(states):
    """states: list of (t, x, y, z, g). Returns (manner, info)."""
    info = {"frames": len(states)}
    if len(states) < 3:
        return "unknown", info
    i_close = None
    for i in range(1, len(states)):
        if states[i][4] < 0.9 and states[i - 1][4] >= 0.9:
            i_close = i
            break
    if i_close is None:
        info["reason"] = "gripper never closed"
        return "push", info
    i_release = None
    for i in range(i_close + 1, len(states)):
        if states[i][4] > 0.9:
            i_release = i
            break
    end = i_release if i_release is not None else len(states)
    held = [states[i][4] for i in range(i_close, end)]
    hold_g = quantile(held, 0.5)
    z_grasp = states[i_close][3]
    info.update({"i_close": i_close, "i_release": i_release, "hold_g": r4(hold_g),
                 "z_grasp": r4(z_grasp), "grasp_xy": [r4(states[i_close][1]), r4(states[i_close][2])]})
    if hold_g < 0.06:
        info["reason"] = "closed on nothing"
        return "push", info
    lifted = False
    sweep = 0.0
    for i in range(i_close + 1, end):
        if states[i][3] > z_grasp + 0.05:
            lifted = True
        if lifted and states[i][3] < z_grasp + 0.03 and states[i - 1][3] < z_grasp + 0.03:
            sweep += math.sqrt((states[i][1] - states[i - 1][1]) ** 2 + (states[i][2] - states[i - 1][2]) ** 2)
    info["low_sweep_m"] = r4(sweep)
    if sweep > 0.08:
        info["reason"] = "held object swept along the table after a lift"
        return "tool", info
    info["reason"] = "held object carried and released"
    return "pick_place", info
# ==== GEOMETRY END ====


def top_down(xyz, yaw):
    return {"position": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
            "rpy_deg": [180.0, 0.0, math.degrees(yaw)], "frame": BASE}


ENDED = {"flag": False}


def go(xyz, yaw, label, tol=0.01, steps=150):
    if ENDED["flag"]:
        return None
    res = move_to_pose(top_down(xyz, yaw), tolerance=tol, max_steps=steps)
    entry = label + ": ok=" + str(res.ok) + " status=" + str(res.status) + " steps=" + str(res.steps_executed)
    try:
        if res.terminated or res.truncated:
            ENDED["flag"] = True
            entry += " TERMINAL"
    except Exception:
        pass
    note(entry)
    return res


def gripper(close, label):
    if ENDED["flag"]:
        return None
    res = close_gripper() if close else open_gripper()
    entry = label + ": ok=" + str(res.ok) + " status=" + str(res.status) + " steps=" + str(res.steps_executed)
    try:
        if res.terminated or res.truncated:
            ENDED["flag"] = True
            entry += " TERMINAL"
    except Exception:
        pass
    note(entry)
    return res


def candidates(prompts, min_score=0.02, top=5):
    out = []
    for prompt in prompts:
        seg = segment_text(CAM, prompt)
        if not seg.ok:
            continue
        for cand in seg.segmentations[:top]:
            score = float(cand.score)
            if score < min_score:
                continue
            cloud = mask_to_point_cloud(cand, CAM, target_frame=BASE)
            pts = cloud.points
            if len(pts) < 15:
                continue
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            zs = [float(p[2]) for p in pts]
            out.append({
                "prompt": prompt, "score": score, "pts": pts,
                "x": quantile(xs, 0.5), "y": quantile(ys, 0.5),
                "z95": quantile(zs, 0.95), "z05": quantile(zs, 0.05),
                "ext_x": quantile(xs, 0.97) - quantile(xs, 0.03),
                "ext_y": quantile(ys, 0.97) - quantile(ys, 0.03),
                "box": [round(float(b), 1) for b in cand.box_xyxy] if cand.box_xyxy is not None else None,
            })
    return out


def best_score(cands):
    best = None
    for c in cands:
        if best is None or c["score"] > best["score"]:
            best = c
    return best


def summary(c):
    return {"prompt": c["prompt"], "score": r4(c["score"]), "x": r4(c["x"]), "y": r4(c["y"]),
            "z95": r4(c["z95"]), "ext": [r4(c["ext_x"]), r4(c["ext_y"])], "box": c["box"], "n": len(c["pts"])}


def locate_cube_near(ref_xy):
    found = [c for c in candidates(["small cube", "cube"])
             if 0.015 < c["z95"] < 0.09 and c["ext_x"] < 0.09 and c["ext_y"] < 0.09]
    best = None
    for c in found:
        top = quantile([float(p[2]) for p in c["pts"]], 0.95)
        tp = [p for p in c["pts"] if float(p[2]) > top - 0.008]
        xy = (quantile([float(p[0]) for p in tp], 0.5), quantile([float(p[1]) for p in tp], 0.5))
        dd = math.sqrt((xy[0] - ref_xy[0]) ** 2 + (xy[1] - ref_xy[1]) ** 2)
        if dd < 0.08 and (best is None or c["score"] > best[0]):
            best = (c["score"], xy)
    if best is None:
        return None
    return best[1]


def hold_still(ticks, gripper_cmd):
    if ENDED["flag"]:
        return
    st = get_robot_state()
    joints = [float(v) for v in st.joint_positions["primary"]]
    done = 0
    for tick in range(ticks):
        try:
            cmd = ArmCommand("joint_position", joints, gripper_position=gripper_cmd, embodiment=st.embodiment)
        except TypeError:
            cmd = ArmCommand("joint_position", joints, gripper_position=gripper_cmd)
        res = step(RobotAction(arms={"primary": cmd}))
        done += 1
        try:
            if res.terminated or res.truncated:
                ENDED["flag"] = True
                break
        except Exception:
            pass
    note("hold_still ticks=" + str(done))


# ---------------- 1. classify the video ----------------
context = get_task_context()
report["language"] = context.language
video = robomme_plus.get_conditioning_video()
states = []
for index in range(len(video)):
    st = video[index].robot_state
    pos = st.end_effector_poses["primary"].position
    states.append((float(video[index].timestamp_s), float(pos[0]), float(pos[1]), float(pos[2]),
                   float(st.gripper_positions["primary"])))
manner, manner_info = classify_video(states)
report["manner"] = manner
report["manner_info"] = manner_info

# ---------------- 2. localize cube and target ----------------
cube_cands = [c for c in candidates(["small cube", "cube"])
              if 0.015 < c["z95"] < 0.09 and c["ext_x"] < 0.09 and c["ext_y"] < 0.09]
report["cube_candidates"] = [summary(c) for c in cube_cands]
if len(cube_cands) == 0:
    raise ValueError("cube not found")
cube = best_score(cube_cands)
cube_top = quantile([float(p[2]) for p in cube["pts"]], 0.95)
top_pts = [p for p in cube["pts"] if float(p[2]) > cube_top - 0.008]
cube_xy = (quantile([float(p[0]) for p in top_pts], 0.5), quantile([float(p[1]) for p in top_pts], 0.5))
cube_half = min(0.04, max(0.01, 0.5 * cube_top))
report["cube"] = {"xy": [r4(cube_xy[0]), r4(cube_xy[1])], "top": r4(cube_top), "half": r4(cube_half)}

target_cands = []
for c in candidates(["target", "bullseye target", "round target marker on the table", "circle on the table"], min_score=0.01):
    if c["z95"] < 0.03 and 0.01 < c["ext_x"] < 0.15 and 0.01 < c["ext_y"] < 0.15:
        if math.sqrt((c["x"] - cube_xy[0]) ** 2 + (c["y"] - cube_xy[1]) ** 2) > 0.03:
            target_cands.append(c)
report["target_candidates"] = [summary(c) for c in target_cands]
if len(target_cands) == 0:
    raise ValueError("target not found")
target = best_score(target_cands)
target_xy = (quantile([float(p[0]) for p in target["pts"]], 0.5), quantile([float(p[1]) for p in target["pts"]], 0.5))
report["target"] = {"xy": [r4(target_xy[0]), r4(target_xy[1])], "prompt": target["prompt"]}

dx = target_xy[0] - cube_xy[0]
dy = target_xy[1] - cube_xy[1]
dist = math.sqrt(dx * dx + dy * dy)
d = (dx / dist, dy / dist)
report["move"] = {"dist": r4(dist), "dir": [r4(d[0]), r4(d[1])]}


def yaw_mod(angle, period):
    a = wrap_angle(angle)
    while a > period / 2:
        a -= period
    while a < -period / 2:
        a += period
    return a


# ---------------- 3. execute ----------------
if manner == "tool":
    tool_cands = []
    for c in candidates(["hook", "metal hook tool", "long stick with a handle", "stick", "peg"], min_score=0.005):
        length = math.sqrt(c["ext_x"] ** 2 + c["ext_y"] ** 2)
        if 0.15 < length < 0.7 and c["z95"] < 0.07:
            if math.sqrt((c["x"] - cube_xy[0]) ** 2 + (c["y"] - cube_xy[1]) ** 2) > 0.06:
                tool_cands.append(c)
    report["tool_candidates"] = [summary(c) for c in tool_cands]
    if len(tool_cands) == 0:
        raise ValueError("tool not found")
    tool = tool_cands[0]  # first prompt in priority order, highest-ranked candidate
    model = tool_model(tool["pts"])
    grasp_z = min(0.04, max(0.015, model["handle_top_z"] - 0.012))
    plan = None
    report["phi_trials"] = []
    for phi_deg in (0, 15, -15, 30, -30):
        trial = plan_hook_pull(model, cube_xy, cube_half, target_xy, phi=math.radians(phi_deg))
        if phi_deg != 0 and trial["m_v"] < 0.0:
            report["phi_trials"].append([phi_deg, "corner-escaping tilt"])
            continue
        feasible = True
        for key in ("start", "end"):
            pose = Pose((trial[key][0], trial[key][1], grasp_z + 0.004), topdown_quat(trial["yaw_pull"]), BASE)
            ik = solve_ik(pose)
            if not ik.ok:
                feasible = False
        gpose = Pose((trial["grasp_xy"][0], trial["grasp_xy"][1], grasp_z), topdown_quat(trial["yaw_grasp"]), BASE)
        if not solve_ik(gpose).ok:
            feasible = False
        report["phi_trials"].append([phi_deg, feasible])
        if feasible:
            plan = trial
            break
    if plan is None:
        note("no IK-feasible hook plan; using phi=0")
        plan = plan_hook_pull(model, cube_xy, cube_half, target_xy)
    report["tool_model"] = {
        "prompt": tool["prompt"], "length": r4(model["length"]), "arm_u": r4(model["arm_u"]),
        "arm_w": r4(model["arm_w"]), "handle_top": r4(model["handle_top_z"]),
        "axis": [r4(model["axis"][0]), r4(model["axis"][1])], "side": model["side"],
        "low": {k: r4(v) for k, v in model["low"].items()}, "high": {k: r4(v) for k, v in model["high"].items()},
        "handle_is_low": model["handle_is_low"],
    }
    report["plan"] = {"start": [r4(v) for v in plan["start"]], "end": [r4(v) for v in plan["end"]],
                      "theta_deg": r4(math.degrees(plan["theta"])), "yaw_grasp_deg": r4(math.degrees(plan["yaw_grasp"])),
                      "yaw_pull_deg": r4(math.degrees(plan["yaw_pull"])), "phi_deg": r4(math.degrees(plan["phi"])), "grasp_xy": [r4(v) for v in plan["grasp_xy"]]}
    gx, gy = plan["grasp_xy"]
    yaw_g = plan["yaw_grasp"]
    yaw_p = plan["yaw_pull"]
    transit_z = 0.16
    gripper(False, "open")
    go((gx, gy, grasp_z + 0.12), yaw_g, "pre-grasp tool", tol=0.02)
    go((gx, gy, grasp_z), yaw_g, "descend to handle", tol=0.008)
    gripper(True, "close on handle")
    held = float(get_robot_state().gripper_positions["primary"]) if not ENDED["flag"] else -1.0
    report["held_width"] = r4(held)
    go((gx, gy, transit_z), yaw_g, "lift tool", tol=0.02)
    sx, sy = plan["start"]
    ex, ey = plan["end"]
    # rotate gradually while translating to the start pose
    for k in range(1, 4):
        f = k / 3.0
        yaw_k = yaw_g + f * (yaw_p - yaw_g)
        go((gx + f * (sx - gx), gy + f * (sy - gy), transit_z), yaw_k, "transit " + str(k), tol=0.02)
    go((sx, sy, grasp_z + 0.04), yaw_p, "lower above table", tol=0.015)
    go((sx, sy, grasp_z + 0.004), yaw_p, "tool on table", tol=0.01)
    pull_len = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
    n_seg = max(1, int(math.ceil(pull_len / 0.025)))
    for k in range(1, n_seg + 1):
        f = k / float(n_seg)
        go((sx + f * (ex - sx), sy + f * (ey - sy), grasp_z + 0.004), yaw_p, "pull " + str(k), tol=0.01, steps=80)
    # closed-loop residual: re-localize the cube and pull the remaining distance along d
    cur = (ex, ey)
    report["residuals"] = []
    for attempt in range(2):
        if ENDED["flag"]:
            break
        now = locate_cube_near(target_xy)
        if now is None:
            note("residual: cube not re-localized")
            break
        rx = target_xy[0] - now[0]
        ry = target_xy[1] - now[1]
        along = rx * d[0] + ry * d[1]
        lateral = -rx * d[1] + ry * d[0]
        report["residuals"].append([r4(now[0]), r4(now[1]), r4(along), r4(lateral)])
        if along < 0.004 or along > 0.08:
            break
        nxt = (cur[0] + d[0] * along, cur[1] + d[1] * along)
        n_seg = max(1, int(math.ceil(along / 0.02)))
        for k in range(1, n_seg + 1):
            f = k / float(n_seg)
            go((cur[0] + f * (nxt[0] - cur[0]), cur[1] + f * (nxt[1] - cur[1]), grasp_z + 0.004), yaw_p,
               "residual pull " + str(attempt) + "." + str(k), tol=0.008, steps=60)
        cur = nxt
    # lift the tool clear of the cube (as in the video) and hold still
    go((cur[0], cur[1], grasp_z + 0.045), yaw_p, "lift tool clear", tol=0.01, steps=60)
    hold_still(30, 0.0)

elif manner == "pick_place":
    best_yaw = 0.0
    best_area = None
    for deg in range(0, 90, 5):
        a = math.radians(deg)
        us = [float(p[0]) * math.cos(a) + float(p[1]) * math.sin(a) for p in top_pts]
        vs = [-float(p[0]) * math.sin(a) + float(p[1]) * math.cos(a) for p in top_pts]
        area = (max(us) - min(us)) * (max(vs) - min(vs))
        if best_area is None or area < best_area:
            best_area = area
            best_yaw = a
    yaw = yaw_mod(best_yaw, math.pi / 2)
    grasp_z = max(0.015, cube_half)
    gripper(False, "open")
    go((cube_xy[0], cube_xy[1], grasp_z + 0.12), yaw, "pre-grasp cube", tol=0.02)
    go((cube_xy[0], cube_xy[1], grasp_z), yaw, "descend to cube", tol=0.008)
    gripper(True, "close on cube")
    go((cube_xy[0], cube_xy[1], grasp_z + 0.12), yaw, "lift cube", tol=0.02)
    go((target_xy[0], target_xy[1], grasp_z + 0.12), yaw, "above target", tol=0.02)
    go((target_xy[0], target_xy[1], grasp_z + 0.008), yaw, "lower onto target", tol=0.01)
    gripper(False, "release")
    go((target_xy[0], target_xy[1], grasp_z + 0.12), yaw, "retreat", tol=0.02)
    hold_still(20, 1.0)

else:  # push with the gripper
    yaw = yaw_mod(math.atan2(d[1], d[0]), math.pi)
    push_z = 0.02
    back = cube_half + 0.035
    bx, by = cube_xy[0] - d[0] * back, cube_xy[1] - d[1] * back
    fx, fy = target_xy[0] - d[0] * (cube_half + 0.012), target_xy[1] - d[1] * (cube_half + 0.012)
    gripper(True, "close (push)")
    go((bx, by, push_z + 0.10), yaw, "above push start", tol=0.02)
    go((bx, by, push_z), yaw, "push start", tol=0.01)
    plen = math.sqrt((fx - bx) ** 2 + (fy - by) ** 2)
    n_seg = max(1, int(math.ceil(plen / 0.025)))
    for k in range(1, n_seg + 1):
        f = k / float(n_seg)
        go((bx + f * (fx - bx), by + f * (fy - by), push_z), yaw, "push " + str(k), tol=0.01, steps=80)
    go((fx, fy, push_z + 0.08), yaw, "retreat", tol=0.02)
    hold_still(20, 0.0)

# ---------------- 4. final check (guarded) ----------------
if not ENDED["flag"]:
    try:
        final = candidates(["small cube", "cube"])
        final = [c for c in final if 0.015 < c["z95"] < 0.09 and c["ext_x"] < 0.09 and c["ext_y"] < 0.09]
        if len(final) > 0:
            fc = best_score(final)
            report["final_cube_xy"] = [r4(fc["x"]), r4(fc["y"])]
            report["final_cube_to_target"] = r4(math.sqrt((fc["x"] - target_xy[0]) ** 2 + (fc["y"] - target_xy[1]) ** 2))
    except Exception as exc:
        report["final_check_error"] = str(exc)
report["ended_by_terminal"] = ENDED["flag"]
result = report
