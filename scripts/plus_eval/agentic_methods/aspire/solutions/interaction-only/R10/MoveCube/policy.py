# MoveCube v00 -- initial program from scene analysis (interaction-only, no prior run of this task).
# Exploration is folded in: task/controller metadata, conditioning-video robot-state timeline,
# coarse video thumbnails, live front-camera prompt probes are all recorded into `result`.
# Strategy: infer the demonstrated manner (pick-and-place vs push) from the conditioning
# video's gripper/EE timeline, localize cube + target in the live front camera, execute.
import math

report = {"version": "v04", "events": [], "notes": []}
ended = {"flag": False}


def short(value, limit=400):
    text = str(value)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def exec_info(res, label):
    info = {"label": label}
    try:
        info["ok"] = bool(res.ok)
    except Exception:
        info["ok"] = None
    try:
        info["status"] = str(res.status)
    except Exception:
        pass
    try:
        info["steps"] = int(res.steps_executed)
    except Exception:
        pass
    term = False
    try:
        if res.terminated:
            term = True
    except Exception:
        pass
    try:
        if res.truncated:
            term = True
    except Exception:
        pass
    info["terminal"] = term
    try:
        if res.error is not None:
            info["error"] = short(res.error, 300)
    except Exception:
        pass
    if term:
        ended["flag"] = True
    try:
        st = get_robot_state()
        p = st.end_effector_poses["primary"].position
        info["ee"] = [round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)]
        info["grip"] = round(float(st.gripper_positions["primary"]), 3)
        info["t"] = round(float(st.timestamp_s), 2)
    except Exception:
        pass
    report["events"].append(info)
    return info


def quantile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


# ---------------------------------------------------------------- metadata
ctx = get_task_context()
report["language"] = ctx.language
report["task_name"] = ctx.task_name
try:
    meta = robomme_plus.get_task_metadata()
    report["task_metadata"] = {str(k): short(meta[k], 600) for k in meta}
except Exception as exc:
    report["task_metadata_error"] = short(exc)
try:
    cmeta = robomme_plus.get_controller_metadata()
    report["controller_metadata"] = {str(k): short(cmeta[k], 600) for k in cmeta}
except Exception as exc:
    report["controller_metadata_error"] = short(exc)

state0 = get_robot_state()
base = state0.base_frame
home_pose = state0.end_effector_poses["primary"]
down_q = [float(v) for v in home_pose.quaternion_wxyz]
report["home_ee"] = [float(v) for v in home_pose.position]
report["home_q"] = down_q
report["home_joints"] = [round(float(v), 4) for v in state0.joint_positions["primary"]]
report["home_grip"] = float(state0.gripper_positions["primary"])

# ---------------------------------------------------------------- conditioning video
timeline = []
thumbs = []
try:
    video = robomme_plus.get_conditioning_video()
    nvid = len(video)
    report["video_frames"] = nvid
    thumb_idx = []
    if nvid > 0:
        for k in range(6):
            thumb_idx.append(int(round(k * (nvid - 1) / 5.0)))
    for i in range(nvid):
        fr = video[i]
        rs = fr.robot_state
        p = rs.end_effector_poses["primary"].position
        q = rs.end_effector_poses["primary"].quaternion_wxyz
        g = float(rs.gripper_positions["primary"])
        timeline.append([round(float(fr.timestamp_s), 3), round(float(p[0]), 4),
                         round(float(p[1]), 4), round(float(p[2]), 4), round(g, 3),
                         [round(float(v), 3) for v in q]])
except Exception as exc:
    report["video_error"] = short(exc)
report["video_thumbs"] = thumbs
stride = 1
if len(timeline) > 128:
    stride = int(len(timeline) / 128) + 1
report["video_timeline"] = [timeline[i] for i in range(0, len(timeline), stride)]

# manner features from the video robot-state timeline
manner = "unknown"
feat = {}
if len(timeline) > 2:
    gs = [row[4] for row in timeline]
    zs = [row[3] for row in timeline]
    g_open = max(gs)
    zmin = min(zs)
    closed = [g < 0.6 * g_open for g in gs]
    segments = []
    i = 0
    while i < len(timeline):
        if closed[i]:
            j = i
            while j + 1 < len(timeline) and closed[j + 1]:
                j += 1
            seg_g = sorted(gs[i:j + 1])
            seg_z = zs[i:j + 1]
            travel = math.hypot(timeline[j][1] - timeline[i][1], timeline[j][2] - timeline[i][2])
            segments.append({"start": i, "end": j, "t0": timeline[i][0], "t1": timeline[j][0],
                             "grip_median": seg_g[len(seg_g) // 2],
                             "z_start": seg_z[0], "z_max": max(seg_z), "z_min": min(seg_z),
                             "z_end": seg_z[-1], "xy_travel": round(travel, 4),
                             "xy_start": [timeline[i][1], timeline[i][2]],
                             "xy_end": [timeline[j][1], timeline[j][2]]})
            i = j + 1
        else:
            i += 1
    feat = {"g_open": g_open, "g_min": min(gs), "z_min": zmin, "z_max": max(zs),
            "closed_segments": segments}
    holding_lift = False
    for seg in segments:
        if seg["grip_median"] > 0.12 and seg["z_max"] - seg["z_start"] > 0.05:
            holding_lift = True
    low_travel = 0.0
    for k in range(1, len(timeline)):
        if timeline[k][3] < zmin + 0.03 and timeline[k - 1][3] < zmin + 0.03:
            low_travel += math.hypot(timeline[k][1] - timeline[k - 1][1],
                                     timeline[k][2] - timeline[k - 1][2])
    feat["low_xy_travel"] = round(low_travel, 4)
    feat["holding_lift"] = holding_lift
    # tool use: object held + lifted, then dragged along the table while still held
    tool_like = False
    pick_like = False
    for seg in segments:
        if not (seg["grip_median"] > 0.12 and seg["z_max"] - seg["z_start"] > 0.05):
            continue
        peak = seg["start"]
        for k in range(seg["start"], seg["end"] + 1):
            if timeline[k][3] > timeline[peak][3]:
                peak = k
        held_low = 0.0
        for k in range(peak + 1, seg["end"] + 1):
            if timeline[k][3] < zmin + 0.03 and timeline[k - 1][3] < zmin + 0.03:
                held_low += math.hypot(timeline[k][1] - timeline[k - 1][1],
                                       timeline[k][2] - timeline[k - 1][2])
        seg["held_low_travel_after_peak"] = round(held_low, 4)
        released = seg["end"] < len(timeline) - 3
        seg["released_before_end"] = released
        if held_low > 0.08 or not released:
            tool_like = True
        else:
            pick_like = True
    feat["tool_like"] = tool_like
    feat["pick_like"] = pick_like
    if tool_like:
        manner = "tool"
    elif pick_like:
        manner = "pick"
    elif low_travel > 0.04:
        manner = "push"
report["video_features"] = feat
report["manner_from_video"] = manner


# ---------------------------------------------------------------- live perception
cam = "front"
probes = {}
cands_by_prompt = {}


def seg_prompt(prompt):
    if prompt in cands_by_prompt:
        return cands_by_prompt[prompt]
    entry = {"ok": False, "n": 0, "cands": []}
    kept = []
    try:
        seg = segment_text(cam, prompt)
        if seg.ok:
            entry["ok"] = True
            entry["n"] = len(seg.segmentations)
            for c in seg.segmentations[:5]:
                box = None
                if c.box_xyxy is not None:
                    box = [round(float(v), 1) for v in c.box_xyxy]
                entry["cands"].append({"score": round(float(c.score), 3), "box": box})
                kept.append(c)
    except Exception as exc:
        entry["error"] = short(exc)
    probes[prompt] = entry
    cands_by_prompt[prompt] = kept
    return kept


def cloud_of(cand):
    pc = None
    try:
        pc = mask_to_point_cloud(cand, cam, target_frame=base)
    except Exception:
        pc = None
    if pc is None:
        try:
            pc = mask_to_point_cloud(cand.mask, cam, target_frame=base)
        except Exception as exc:
            report["notes"].append("mask_to_point_cloud failed: " + short(exc))
            return None
    pts = []
    for p in pc.points:
        pts.append((float(p[0]), float(p[1]), float(p[2])))
    return pts


def summarize(pts):
    if pts is None or len(pts) < 15:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    z90 = quantile(zs, 0.9)
    # top-face points give an unbiased xy centre (the front face is seen from +x)
    top = [p for p in pts if p[2] > z90 - 0.008]
    if len(top) < 5:
        top = pts
    return {"n": len(pts), "cx": quantile([p[0] for p in top], 0.5),
            "cy": quantile([p[1] for p in top], 0.5),
            "mx": quantile(xs, 0.5), "my": quantile(ys, 0.5),
            "z10": quantile(zs, 0.1), "z50": quantile(zs, 0.5), "z90": z90,
            "zmin": min(zs), "zmax": max(zs),
            "xspan": quantile(xs, 0.95) - quantile(xs, 0.05),
            "yspan": quantile(ys, 0.95) - quantile(ys, 0.05)}


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = box_area(a) + box_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


img_area = 256.0 * 256.0

# cube: highest-score compact candidate whose cloud is a small block standing on the table
cube = None
cube_box = None
for prompt in ["red cube", "cube", "small block"]:
    best = None
    for c in seg_prompt(prompt):
        if c.box_xyxy is None:
            continue
        box = [float(v) for v in c.box_xyxy]
        if box_area(box) > 0.03 * img_area:
            continue
        if best is None or float(c.score) > float(best.score):
            best = c
    if best is not None and float(best.score) > 0.3:
        s = summarize(cloud_of(best))
        if s is not None and s["xspan"] < 0.08 and s["yspan"] < 0.08:
            cube = s
            cube_box = [float(v) for v in best.box_xyxy]
            cube["prompt"] = prompt
            cube["score"] = round(float(best.score), 3)
            cube["box"] = cube_box
            break
report["cube"] = cube

# target: flat, small marker on the table (not the cube)
target = None
best_t = None
for prompt in ["purple target", "circular target on the table", "target", "purple and white target"]:
    for c in seg_prompt(prompt):
        if c.box_xyxy is None or float(c.score) < 0.05:
            continue
        box = [float(v) for v in c.box_xyxy]
        if box_area(box) > 0.02 * img_area:
            continue
        if cube_box is not None and iou(box, cube_box) > 0.2:
            continue
        if best_t is None or float(c.score) > best_t[0]:
            best_t = (float(c.score), c, prompt, box)
    if best_t is not None and best_t[0] > 0.2:
        break
if best_t is not None:
    s = summarize(cloud_of(best_t[1]))
    if s is not None:
        target = s
        target["cx"] = s["mx"]
        target["cy"] = s["my"]
        target["prompt"] = best_t[2]
        target["score"] = round(best_t[0], 3)
        target["box"] = best_t[3]
report["target"] = target


# hook tool: long, flat object with a thicker handle at one end and a perpendicular bar at the other
def analyse_tool(pts):
    if pts is None or len(pts) < 40:
        return None
    n = float(len(pts))
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts) / n
    syy = sum((p[1] - my) ** 2 for p in pts) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts) / n
    ang = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    ux, uy = math.cos(ang), math.sin(ang)
    nx, ny = -uy, ux
    ts = [(p[0] - mx) * ux + (p[1] - my) * uy for p in pts]
    ss = [(p[0] - mx) * nx + (p[1] - my) * ny for p in pts]
    tlo = quantile(ts, 0.01)
    thi = quantile(ts, 0.99)
    length = thi - tlo
    lo_pts = [i for i in range(len(pts)) if ts[i] < tlo + 0.045]
    hi_pts = [i for i in range(len(pts)) if ts[i] > thi - 0.045]
    if len(lo_pts) < 5 or len(hi_pts) < 5:
        return None
    zlo = quantile([pts[i][2] for i in lo_pts], 0.8)
    zhi = quantile([pts[i][2] for i in hi_pts], 0.8)
    mid = [i for i in range(len(pts)) if tlo + 0.3 * length < ts[i] < thi - 0.3 * length]
    s0 = quantile([ss[i] for i in mid], 0.5) if len(mid) > 3 else 0.0
    if zlo >= zhi:
        handle_idx, hook_idx, sign, t_hook = lo_pts, hi_pts, 1.0, thi
    else:
        handle_idx, hook_idx, sign, t_hook = hi_pts, lo_pts, -1.0, tlo
    hx = quantile([pts[i][0] for i in handle_idx], 0.5)
    hy = quantile([pts[i][1] for i in handle_idx], 0.5)
    hz = quantile([pts[i][2] for i in handle_idx], 0.9)
    lat = [ss[i] - s0 for i in hook_idx]
    lat_pos = quantile(lat, 0.97)
    lat_neg = quantile(lat, 0.03)
    if lat_pos >= -lat_neg:
        bar_sign, bar_len = 1.0, lat_pos
    else:
        bar_sign, bar_len = -1.0, -lat_neg
    ax, ay = sign * ux, sign * uy
    kx = mx + ux * t_hook + nx * s0
    ky = my + uy * t_hook + ny * s0
    return {"length": round(length, 4), "handle": [hx, hy, hz], "corner": [kx, ky],
            "axis": [ax, ay], "bar_dir": [bar_sign * nx, bar_sign * ny],
            "bar_len": round(bar_len, 4), "z_handle_end": round(max(zlo, zhi), 4),
            "z_hook_end": round(min(zlo, zhi), 4), "n": len(pts)}


tool = None
if manner == "tool":
    tool_trials = []
    best_tool = None
    for prompt in ["hook", "L-shaped hook", "stick", "peg", "long tool with a handle"]:
        for c in seg_prompt(prompt)[:5]:
            if c.box_xyxy is None or float(c.score) < 0.005:
                continue
            box = [float(v) for v in c.box_xyxy]
            if box_area(box) > 0.12 * img_area:
                continue
            if cube_box is not None and iou(box, cube_box) > 0.3:
                continue
            info = analyse_tool(cloud_of(c))
            trial = {"prompt": prompt, "score": round(float(c.score), 3), "box": box,
                     "analysis": info}
            tool_trials.append(trial)
            if info is None:
                continue
            valid = (0.18 < info["length"] < 0.5 and 0.025 < info["bar_len"] < 0.12
                     and info["z_handle_end"] - info["z_hook_end"] > 0.006
                     and info["z_handle_end"] < 0.05 and info["z_hook_end"] < 0.03)
            trial["valid"] = valid
            if valid and (best_tool is None or float(c.score) > best_tool["score"]):
                best_tool = trial
        if best_tool is not None and best_tool["score"] > 0.2:
            break
    report["tool_trials"] = tool_trials
    if best_tool is not None:
        tool = best_tool["analysis"]
        tool["prompt"] = best_tool["prompt"]
        tool["score"] = best_tool["score"]
report["tool"] = tool
report["probes"] = probes

# ---------------------------------------------------------------- act


def yaw_quat(psi):
    # world-frame yaw psi applied to the observed home (top-down) orientation
    cw, sz = math.cos(psi / 2.0), math.sin(psi / 2.0)
    w, x, y, z = down_q
    return [cw * w - sz * z, cw * x - sz * y, cw * y + sz * x, cw * z + sz * w]


def move(xyz, label, tol=0.01, steps=150, quat=None):
    if ended["flag"]:
        return {"ok": False, "terminal": True}
    q = down_q if quat is None else quat
    pose = Pose((float(xyz[0]), float(xyz[1]), float(xyz[2])), q, base)
    try:
        res = move_to_pose(pose, tolerance=tol, max_steps=steps)
    except Exception as exc:
        report["events"].append({"label": label, "exception": short(exc)})
        return {"ok": False, "terminal": False}
    return exec_info(res, label)


def grip(open_it, label):
    if ended["flag"]:
        return None
    try:
        res = open_gripper() if open_it else close_gripper()
    except Exception as exc:
        report["events"].append({"label": label, "exception": short(exc)})
        return None
    return exec_info(res, label)


def ik_ok(xyz, quat):
    try:
        r = solve_ik(Pose((float(xyz[0]), float(xyz[1]), float(xyz[2])), quat, base))
        return bool(r.ok)
    except Exception:
        return False


def wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def locate_cube_now():
    # re-localize the cube (top-face xy) for closed-loop dragging; None when not found
    for prompt in ["red cube", "cube"]:
        try:
            seg = segment_text(cam, prompt)
        except Exception:
            continue
        if not seg.ok:
            continue
        for c in seg.segmentations[:3]:
            if c.box_xyxy is None or float(c.score) < 0.3:
                continue
            box = [float(v) for v in c.box_xyxy]
            if box_area(box) > 0.03 * img_area:
                continue
            sm = summarize(cloud_of(c))
            if sm is not None and sm["xspan"] < 0.08 and sm["yspan"] < 0.08 and sm["z90"] < 0.08:
                return sm
    return None


mode = manner if manner in ("pick", "push", "tool") else "pick"
if mode == "tool" and tool is None:
    report["notes"].append("video shows tool use but no hook tool localized; falling back to push")
    mode = "push"
report["executed_mode"] = mode

if cube is None or target is None:
    report["notes"].append("localization incomplete; no manipulation attempted")
else:
    table_z = min(cube["z10"], target["z50"])
    report["table_z"] = table_z
    height = max(0.02, min(0.08, cube["z90"] - table_z))
    report["cube_height_est"] = height
    half = 0.5 * height
    cx, cy = cube["cx"], cube["cy"]
    tx, ty = target["cx"], target["cy"]
    dist = math.hypot(tx - cx, ty - cy)
    report["cube_to_target"] = dist
    dx = (tx - cx) / max(dist, 1e-6)
    dy = (ty - cy) / max(dist, 1e-6)
    if mode == "pick":
        grip(True, "open")
        top = table_z + height
        move((cx, cy, top + 0.10), "pre_grasp", tol=0.015)
        move((cx, cy, table_z + 0.55 * height), "grasp", tol=0.01)
        grip(False, "close")
        move((cx, cy, top + 0.12), "lift", tol=0.02)
        move((tx, ty, top + 0.12), "above_target", tol=0.02)
        move((tx, ty, table_z + 0.6 * height + 0.01), "lower", tol=0.01)
        grip(True, "release")
        move((tx, ty, top + 0.10), "retreat", tol=0.02)
    elif mode == "push":
        back = half + 0.04
        push_z = table_z + 0.5 * height
        grip(False, "close_for_push")
        move((cx - dx * back, cy - dy * back, push_z + 0.10), "pre_push_high", tol=0.015)
        move((cx - dx * back, cy - dy * back, push_z), "pre_push_low", tol=0.01)
        end_d = dist - half - 0.005
        nseg = max(1, int(math.ceil((end_d + back) / 0.03)))
        for k in range(1, nseg + 1):
            d = -back + (end_d + back) * k / nseg
            move((cx + dx * d, cy + dy * d, push_z), "push_" + str(k), tol=0.01, steps=60)
        move((cx + dx * (end_d - 0.03), cy + dy * (end_d - 0.03), push_z + 0.10),
             "push_retreat", tol=0.02)
    else:
        # hook pull: grasp the handle, swing the shaft so it points away from the target,
        # drop the bar behind the cube, drag the hand along cube->target.
        gx, gy, gz = tool["handle"]
        ax, ay = tool["axis"]
        kx, ky = tool["corner"]
        bx, by = tool["bar_dir"]
        theta_tool = math.atan2(ay, ax)
        theta_des = math.atan2(-dy, -dx)
        delta = wrap(theta_des - theta_tool)
        grasp_z = max(gz - 0.003, table_z + 0.012)
        lift_z = grasp_z + 0.13
        cdel, sdel = math.cos(delta), math.sin(delta)
        vx0, vy0 = kx - gx, ky - gy
        vx, vy = cdel * vx0 - sdel * vy0, cdel * vy0 + sdel * vx0
        bx2, by2 = cdel * bx - sdel * by, cdel * by + sdel * bx
        a2x, a2y = -dx, -dy
        lat = max(half + 0.012, min(tool["bar_len"] - 0.012, 1.41 * half + 0.012))
        behind = half + 0.06   # v02: extra margin; the held shaft droops and shortens
        kdx = cx + a2x * behind - bx2 * lat
        kdy = cy + a2y * behind - by2 * lat
        px, py = kdx - vx, kdy - vy
        psi0 = math.atan2(ay, ax)
        options = []
        for kk in range(-3, 4):
            psi = psi0 + kk * math.pi
            fin = psi + delta
            if abs(psi) <= 3.7 and abs(fin) <= 3.7:
                options.append((max(abs(psi), abs(fin)), psi))
        options = sorted(options)
        chosen = None
        tried = []
        for opt in options:
            psi = opt[1]
            ok1 = ik_ok((gx, gy, grasp_z), yaw_quat(psi))
            ok2 = ik_ok((px, py, grasp_z + 0.004), yaw_quat(psi + delta))
            tried.append([round(psi, 3), ok1, ok2])
            if ok1 and ok2:
                chosen = psi
                break
        if chosen is None and len(options) > 0:
            chosen = options[0][1]
        report["hook_plan"] = {"delta": round(delta, 3), "psi_options": tried,
                               "psi": None if chosen is None else round(chosen, 3),
                               "place_hand": [px, py], "corner_des": [kdx, kdy],
                               "grasp": [gx, gy, grasp_z], "lat": lat}
        if chosen is not None:
            qg = yaw_quat(chosen)
            qd = yaw_quat(chosen + delta)
            grip(True, "open")
            move((gx, gy, lift_z), "above_handle", tol=0.015, steps=200, quat=qg)
            move((gx, gy, grasp_z + 0.04), "pre_handle", tol=0.01, quat=qg)
            move((gx, gy, grasp_z), "handle", tol=0.008, quat=qg)
            grip(False, "close_on_handle")
            move((gx, gy, lift_z), "lift_tool", tol=0.015, quat=qg)
            move((gx, gy, lift_z), "rotate_tool", tol=0.02, steps=200, quat=qd)
            move((px, py, lift_z), "over_cube", tol=0.015, steps=200, quat=qd)
            move((px, py, grasp_z + 0.03), "lower_tool_high", tol=0.01, quat=qd)
            move((px, py, grasp_z + 0.004), "lower_tool", tol=0.01, quat=qd)
            # v03: nominal free travel before the bar meets the cube, then closed-loop on the cube
            free = behind - half - 0.005
            drag_z = grasp_z + 0.004
            hx, hy = px, py

            def drag_by(sx, sy, length, tag):
                nseg = max(1, int(math.ceil(length / 0.03)))
                for k in range(1, nseg + 1):
                    f = length * k / nseg
                    move((sx + dx * f, sy + dy * f, drag_z), tag + "_" + str(k), tol=0.01,
                         steps=60, quat=qd)
                return sx + dx * length, sy + dy * length

            first = max(0.0, free + dist - 0.04)
            hx, hy = drag_by(hx, hy, first, "drag")
            drag_log = [{"stage": 0, "length": first}]
            for stage in range(1, 4):
                if ended["flag"]:
                    break
                now = locate_cube_now()
                if now is None:
                    drag_log.append({"stage": stage, "cube": None})
                    break
                ex, ey = tx - now["cx"], ty - now["cy"]
                rem = ex * dx + ey * dy
                latb = ex * bx2 + ey * by2
                drag_log.append({"stage": stage, "cube": [now["cx"], now["cy"]],
                                 "remaining": rem, "lateral_bar": latb})
                if rem > 0.12:
                    break
                acted = False
                # v04: the shaft lies on the -bar side of the cube, so it can push the cube
                # toward +bar; a residual on the -bar side cannot be corrected by this tool pose.
                if latb > 0.006:
                    move((hx + bx2 * (latb + 0.004), hy + by2 * (latb + 0.004), drag_z),
                         "lateral_s" + str(stage), tol=0.004, steps=60, quat=qd)
                    hx, hy = hx + bx2 * (latb + 0.004), hy + by2 * (latb + 0.004)
                    acted = True
                if rem > 0.008:
                    hx, hy = drag_by(hx, hy, rem, "drag_s" + str(stage))
                    acted = True
                if not acted:
                    break
            report["drag_log"] = drag_log
            # back the bar off the cube before lifting so the lift does not drag it
            # v04: back off 3 cm along the shaft and 1.2 cm off the cube side with a tight
            # tolerance (v03's 1 cm tolerance stopped after 0.5 cm and the lift tipped the cube)
            rx, ry = hx - dx * 0.03 - bx2 * 0.012, hy - dy * 0.03 - by2 * 0.012
            move((rx, ry, drag_z), "release_bar", tol=0.004, steps=60, quat=qd)
            move((rx, ry, grasp_z + 0.06), "lift_after_drag", tol=0.01, quat=qd)

# ---------------------------------------------------------------- settle + final diagnostics
if not ended["flag"]:
    try:
        st = get_robot_state()
        for tick in range(10):
            r = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=st.joint_positions["primary"],
                gripper_position=st.gripper_positions["primary"],
                embodiment=st.embodiment)}))
            try:
                if r.terminated or r.truncated:
                    ended["flag"] = True
                    report["notes"].append("terminal during settle tick " + str(tick))
                    break
            except Exception:
                pass
    except Exception as exc:
        report["notes"].append("settle failed: " + short(exc))
if not ended["flag"]:
    try:
        final_seg = segment_text(cam, "red cube")
        if final_seg.ok and len(final_seg.segmentations) > 0:
            report["final_cube"] = summarize(cloud_of(final_seg.segmentations[0]))
    except Exception as exc:
        report["notes"].append("final cube read failed: " + short(exc))
report["ended"] = ended["flag"]
result = report
