# MoveCube v03 -- v02 plus: transit at z>=0.32 (v02's held shaft hit the 0.185 m tall blue box
# during a low, IK-confused rotation and the episode terminated), rotation of the held tool done
# in <=0.6 rad joint-continuous stages at transit height, pull 1 cm higher.
# v02 notes -- tool manner: grasp the hook tool by its thick HANDLE block (v01 grasped the
# thin shaft, which slipped out at descent; the empty gripper then touched the cube and the
# episode terminated). Rotate the held tool so its shaft points from the hand through the cube
# away from the target, lower the hook bar beyond the cube, and PULL along the shaft (the
# manner seen in the ep0 conditioning video).
# 1. Records goal, metadata and the conditioning video robot-state trajectory
#    (v01: no RGB frame dump; video robot-state trajectory only).
# 2. Classifies the manner shown in the video (pick / push / peg) from the recorded
#    gripper width + end-effector height (pure Python over public Observation objects).
# 3. Localizes cube / target / peg in the live front image (SAM3 + depth), then executes
#    the chosen manner with staged Cartesian moves.
import math

report = {"version": "v03", "errors": [], "moves": []}
flags = {"ended": False}


def note_err(where, exc):
    report["errors"].append(where + ": " + str(exc))


def quant(values, frac):
    ordered = sorted(values)
    if len(ordered) == 0:
        return 0.0
    idx = int((len(ordered) - 1) * frac)
    return ordered[idx]


def r4(v):
    return round(float(v), 4)


# ---------------------------------------------------------------- context / metadata
ctx = get_task_context()
goal = str(ctx.language)
report["goal"] = goal
try:
    report["task_name"] = str(ctx.task_name)
except Exception as exc:
    note_err("task_name", exc)

max_steps = 0
try:
    md = robomme_plus.get_task_metadata()
    meta = {}
    for key in md:
        meta[str(key)] = str(md[key])
    report["task_metadata"] = meta
    max_steps = int(md["max_steps"])
except Exception as exc:
    note_err("task_metadata", exc)
try:
    cmd = robomme_plus.get_controller_metadata()
    cmeta = {}
    for key in cmd:
        cmeta[str(key)] = str(cmd[key])
    report["controller_metadata"] = cmeta
except Exception as exc:
    note_err("controller_metadata", exc)

state0 = get_robot_state()
base_frame = state0.base_frame
home_pose = state0.end_effector_poses["primary"]
home_pos = [float(v) for v in home_pose.position]
home_quat = [float(v) for v in home_pose.quaternion_wxyz]
report["home"] = {
    "position": [r4(v) for v in home_pos],
    "quat": [r4(v) for v in home_quat],
    "joints": [r4(v) for v in state0.joint_positions["primary"]],
    "gripper": r4(state0.gripper_positions["primary"]),
    "base_frame": str(base_frame),
    "embodiment": str(state0.embodiment),
    "timestamp_s": r4(state0.timestamp_s),
}


# ---------------------------------------------------------------- quaternion helpers
def qmul(a, b):
    return [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]


def qrot(q, v):
    w, x, y, z = q[0], q[1], q[2], q[3]
    vq = [0.0, v[0], v[1], v[2]]
    qc = [w, -x, -y, -z]
    out = qmul(qmul(q, vq), qc)
    return [out[1], out[2], out[3]]


def closing_angle(q):
    axis = qrot(q, [0.0, 1.0, 0.0])
    return math.atan2(axis[1], axis[0])


def wrap_half_pi(a):
    # wrap to [-pi/2, pi/2) using pi periodicity (gripper is symmetric)
    while a >= math.pi / 2:
        a -= math.pi
    while a < -math.pi / 2:
        a += math.pi
    return a


home_close_angle = closing_angle(home_quat)
report["home"]["closing_axis_deg"] = round(math.degrees(home_close_angle), 1)


def quat_for_closing(theta):
    delta = wrap_half_pi(theta - home_close_angle)
    rz = [math.cos(delta / 2.0), 0.0, 0.0, math.sin(delta / 2.0)]
    return qmul(rz, home_quat)


# ---------------------------------------------------------------- conditioning video
traj = []
video_len = 0
try:
    video = robomme_plus.get_conditioning_video()
    video_len = len(video)
    for i in range(video_len):
        fr = video[i]
        rs = fr.robot_state
        eep = rs.end_effector_poses["primary"]
        p = eep.position
        q = [float(v) for v in eep.quaternion_wxyz]
        traj.append([
            round(float(fr.timestamp_s), 3), r4(p[0]), r4(p[1]), r4(p[2]),
            round(float(rs.gripper_positions["primary"]), 3),
            round(math.degrees(closing_angle(q)), 1),
        ])
except Exception as exc:
    note_err("video", exc)
report["video_len"] = video_len
report["video_traj"] = traj


def classify_manner(rows):
    info = {}
    if len(rows) < 5:
        return "unknown", info
    zs = [r[3] for r in rows]
    gs = [r[4] for r in rows]
    zmin = min(zs)
    info["zmin"] = r4(zmin)
    info["g_min"] = r4(min(gs))
    info["g_max"] = r4(max(gs))
    held = [i for i in range(len(rows)) if 0.12 < gs[i] < 0.85]
    empty_closed = [i for i in range(len(rows)) if gs[i] <= 0.12]
    info["n_held"] = len(held)
    info["n_empty_closed"] = len(empty_closed)
    low_travel_free = 0.0
    low_travel_held = 0.0
    high_travel_held = 0.0
    for i in range(1, len(rows)):
        dxy = math.sqrt((rows[i][1] - rows[i - 1][1]) ** 2 + (rows[i][2] - rows[i - 1][2]) ** 2)
        low = zs[i] < zmin + 0.03
        is_held = 0.12 < gs[i] < 0.85
        if low and is_held:
            low_travel_held += dxy
        elif low:
            low_travel_free += dxy
        if is_held and zs[i] > zmin + 0.06:
            high_travel_held += dxy
    info["low_travel_free"] = r4(low_travel_free)
    info["low_travel_held"] = r4(low_travel_held)
    info["high_travel_held"] = r4(high_travel_held)
    if len(held) >= max(3, len(rows) // 20):
        held_g = quant([gs[i] for i in held], 0.5)
        info["held_g_median"] = r4(held_g)
        if low_travel_held > 0.08:
            return "peg", info
        return "pick", info
    return "push", info


manner, manner_info = classify_manner(traj)
report["manner_video"] = manner
report["manner_info"] = manner_info
gl = goal.lower()
report["goal_words"] = {
    "push": ("push" in gl), "pick": ("pick" in gl or "grasp" in gl),
    "peg": ("peg" in gl or "stick" in gl or "tool" in gl),
}

# ---------------------------------------------------------------- live perception
CAM = "front"


def cloud_points(cand):
    pc = None
    try:
        pc = mask_to_point_cloud(cand, CAM, target_frame=base_frame)
    except Exception as exc:
        note_err("mask_to_point_cloud(cand)", exc)
        pc = mask_to_point_cloud(cand.mask, CAM, target_frame=base_frame)
    pts = []
    for p in pc.points:
        pts.append((float(p[0]), float(p[1]), float(p[2])))
    if len(pts) > 600:
        stride = len(pts) // 600 + 1
        pts = [pts[i] for i in range(0, len(pts), stride)]
    return pts


def rect_fit(pts2):
    best = None
    for k in range(18):
        th = math.radians(5.0 * k)
        c = math.cos(th)
        s = math.sin(th)
        us = [p[0] * c + p[1] * s for p in pts2]
        vs = [-p[0] * s + p[1] * c for p in pts2]
        u0 = quant(us, 0.04)
        u1 = quant(us, 0.96)
        v0 = quant(vs, 0.04)
        v1 = quant(vs, 0.96)
        area = (u1 - u0) * (v1 - v0)
        if best is None or area < best[0]:
            uc = (u0 + u1) / 2.0
            vc = (v0 + v1) / 2.0
            best = (area, th, u1 - u0, v1 - v0, uc * c - vc * s, uc * s + vc * c)
    return best


def pca2(pts2):
    n = len(pts2)
    mx = sum(p[0] for p in pts2) / n
    my = sum(p[1] for p in pts2) / n
    cxx = sum((p[0] - mx) ** 2 for p in pts2) / n
    cyy = sum((p[1] - my) ** 2 for p in pts2) / n
    cxy = sum((p[0] - mx) * (p[1] - my) for p in pts2) / n
    ang = 0.5 * math.atan2(2.0 * cxy, cxx - cyy)
    tr = cxx + cyy
    det = cxx * cyy - cxy * cxy
    disc = math.sqrt(max(tr * tr / 4.0 - det, 0.0))
    l1 = tr / 2.0 + disc
    l2 = max(tr / 2.0 - disc, 1e-9)
    return ang, math.sqrt(l1 / l2), mx, my


def features(pts):
    f = {"n": len(pts)}
    if len(pts) < 8:
        return f
    zs = [p[2] for p in pts]
    f["z05"] = quant(zs, 0.05)
    f["z50"] = quant(zs, 0.5)
    f["z95"] = quant(zs, 0.95)
    top = [p for p in pts if p[2] > f["z95"] - 0.01]
    if len(top) < 5:
        top = pts
    rf = rect_fit([(p[0], p[1]) for p in top])
    f["top_yaw"] = rf[1]
    f["top_a"] = rf[2]
    f["top_b"] = rf[3]
    f["top_cx"] = rf[4]
    f["top_cy"] = rf[5]
    ang, ratio, mx, my = pca2([(p[0], p[1]) for p in pts])
    f["pca_ang"] = ang
    f["pca_ratio"] = ratio
    f["cx"] = quant([p[0] for p in pts], 0.5)
    f["cy"] = quant([p[1] for p in pts], 0.5)
    rfa = rect_fit([(p[0], p[1]) for p in pts])
    f["all_a"] = rfa[2]
    f["all_b"] = rfa[3]
    f["all_cx"] = rfa[4]
    f["all_cy"] = rfa[5]
    return f


def rounded(f):
    out = {}
    for k in f:
        out[k] = r4(f[k])
    return out


probe_log = []


def candidates(prompt, limit=5):
    out = []
    try:
        seg = segment_text(CAM, prompt)
    except Exception as exc:
        note_err("segment_text " + prompt, exc)
        return out
    entry = {"prompt": prompt, "ok": bool(seg.ok), "cands": []}
    if seg.ok:
        k = 0
        for cand in seg.segmentations[:limit]:
            item = {"score": round(float(cand.score), 4)}
            try:
                if cand.box_xyxy is not None:
                    item["box"] = [round(float(v), 1) for v in cand.box_xyxy]
            except Exception as exc:
                note_err("box", exc)
            pts = []
            try:
                pts = cloud_points(cand)
                f = features(pts)
            except Exception as exc:
                note_err("cloud " + prompt, exc)
                f = {"n": 0}
            item["f"] = rounded(f)
            entry["cands"].append(item)
            out.append((prompt, k, float(cand.score), f, cand, pts))
            k += 1
    probe_log.append(entry)
    return out


# cube
cube = None
for prompt in ("cube", "small cube", "block"):
    for c in candidates(prompt):
        f = c[3]
        if f["n"] < 15:
            continue
        a = f["top_a"]
        b = f["top_b"]
        if 0.015 <= a <= 0.08 and 0.015 <= b <= 0.08 and max(a, b) / min(a, b) < 1.5:
            if f["z95"] < home_pos[2] - 0.05:
                cube = c
                break
    if cube is not None:
        break

table_z = None
cube_side = 0.04
if cube is not None:
    cf = cube[3]
    cube_side = (cf["top_a"] + cf["top_b"]) / 2.0
    table_z = cf["z95"] - cube_side
    report["cube"] = {"prompt": cube[0], "rank": cube[1], "f": rounded(cf),
                      "side": r4(cube_side), "table_z": r4(table_z)}


def far_from_cube(f, dist):
    if cube is None:
        return True
    dx = f["cx"] - cube[3]["top_cx"]
    dy = f["cy"] - cube[3]["top_cy"]
    return math.sqrt(dx * dx + dy * dy) > dist


target = None
if cube is not None:
    for prompt in ("target", "goal", "target area", "bullseye", "circle", "marker", "disk"):
        for c in candidates(prompt):
            f = c[3]
            if f["n"] < 15:
                continue
            flat = f["z95"] - table_z < 0.02
            size_ok = 0.02 <= max(f["all_a"], f["all_b"]) <= 0.35
            if flat and size_ok and far_from_cube(f, 0.04):
                target = c
                break
        if target is not None:
            break
if target is not None:
    report["target"] = {"prompt": target[0], "rank": target[1], "f": rounded(target[3])}
    tz = target[3]["z50"]
    if abs(tz - table_z) < 0.025:
        table_z = tz
        cube_side = cube[3]["z95"] - table_z
    report["table_z_used"] = r4(table_z)
    report["cube_side_used"] = r4(cube_side)

def tool_model(pts):
    """Planar model of a long hand tool from its cloud: axis handle->hook, handle grasp point,
    hook-bar side and inner-face position (tool coords: t along axis, l lateral = rot90(axis))."""
    n = len(pts)
    ang, ratio, mx, my = pca2([(p[0], p[1]) for p in pts])
    ax = (math.cos(ang), math.sin(ang))
    for flip in range(2):
        nx = (-ax[1], ax[0])
        al = [(p[0] - mx) * ax[0] + (p[1] - my) * ax[1] for p in pts]
        lt = [(p[0] - mx) * nx[0] + (p[1] - my) * nx[1] for p in pts]
        a0 = quant(al, 0.02)
        a1 = quant(al, 0.98)
        end0 = [i for i in range(n) if al[i] < a0 + 0.06]
        end1 = [i for i in range(n) if al[i] > a1 - 0.06]
        z0 = quant([pts[i][2] for i in end0], 0.9)
        z1 = quant([pts[i][2] for i in end1], 0.9)
        if z0 >= z1 or flip == 1:
            break
        ax = (-ax[0], -ax[1])
    length = a1 - a0
    mid = [lt[i] for i in range(n) if a0 + 0.35 * length < al[i] < a0 + 0.65 * length]
    stick_l = quant(mid, 0.5) if len(mid) > 0 else 0.0
    handle = end0
    t_g = quant([al[i] for i in handle], 0.5)
    l_g = quant([lt[i] for i in handle], 0.5)
    handle_top = quant([pts[i][2] for i in handle], 0.9)
    hook = end1
    rel = [lt[i] - stick_l for i in hook]
    pos_ext = quant(rel, 0.98)
    neg_ext = -quant(rel, 0.02)
    sigma = 1.0 if pos_ext >= neg_ext else -1.0
    bar_len = max(pos_ext, neg_ext)
    bar = [i for i in hook if (lt[i] - stick_l) * sigma > 0.02]
    if len(bar) >= 3:
        t_in = quant([al[i] for i in bar], 0.1)
    else:
        t_in = a1 - 0.012
    gx = mx + ax[0] * t_g + nx[0] * l_g
    gy = my + ax[1] * t_g + nx[1] * l_g
    return {"axis_ang": math.atan2(ax[1], ax[0]), "length": length, "ratio": ratio,
            "z_handle": z0, "z_hook": z1, "stick_l": stick_l, "t_g": t_g, "l_g": l_g,
            "handle_top": handle_top, "grasp_x": gx, "grasp_y": gy, "sigma": sigma,
            "bar_len": bar_len, "bar_other": min(pos_ext, neg_ext), "t_in": t_in,
            "n_bar": len(bar), "n": n}


peg = None
tool = None
tool_pts = None
tool_log = []
if cube is not None:
    for prompt in ("hook", "stick", "peg", "tool"):
        for c in candidates(prompt):
            f = c[3]
            if f["n"] < 30:
                continue
            length = max(f["all_a"], f["all_b"])
            low = f["z95"] - table_z < 0.05
            if not (f["pca_ratio"] > 3.0 and 0.18 <= length <= 0.5 and low and far_from_cube(f, 0.04)):
                continue
            try:
                tm = tool_model(c[5])
            except Exception as exc:
                note_err("tool_model", exc)
                continue
            tool_log.append({"prompt": prompt, "rank": c[1], "m": rounded(tm)})
            hook_like = tm["bar_len"] > 0.03 and tm["bar_len"] > 2.0 * max(tm["bar_other"], 0.008)
            if hook_like and tm["z_handle"] > tm["z_hook"]:
                peg = c
                tool = tm
                break
        if peg is not None:
            break
report["tool_candidates"] = tool_log
if peg is not None:
    report["peg"] = {"prompt": peg[0], "rank": peg[1], "f": rounded(peg[3]), "model": rounded(tool)}

# extra probes for offline analysis
for prompt in ("button", "table", "robot gripper"):
    candidates(prompt, limit=3)
report["probes"] = probe_log


# ---------------------------------------------------------------- motion helpers
def check_end(res):
    try:
        if bool(res.terminated) or bool(res.truncated):
            flags["ended"] = True
    except Exception as exc:
        note_err("term_flags", exc)


def go(pos, quat, label, tol=0.01, ms=150):
    if flags["ended"]:
        return None
    try:
        res = move_to_pose(Pose((float(pos[0]), float(pos[1]), float(pos[2])),
                                (quat[0], quat[1], quat[2], quat[3]), base_frame),
                           tolerance=tol, max_steps=ms)
    except Exception as exc:
        note_err("move " + label, exc)
        return None
    entry = {"label": label, "target": [r4(v) for v in pos]}
    try:
        entry["ok"] = bool(res.ok)
        entry["status"] = str(res.status)
        entry["steps"] = int(res.steps_executed)
    except Exception as exc:
        note_err("move fields", exc)
    check_end(res)
    entry["ended"] = flags["ended"]
    report["moves"].append(entry)
    return res


def grip(close, label):
    if flags["ended"]:
        return None
    try:
        res = close_gripper() if close else open_gripper()
    except Exception as exc:
        note_err("grip " + label, exc)
        return None
    entry = {"label": label}
    try:
        entry["ok"] = bool(res.ok)
        entry["steps"] = int(res.steps_executed)
    except Exception as exc:
        note_err("grip fields", exc)
    check_end(res)
    entry["ended"] = flags["ended"]
    report["moves"].append(entry)
    return res


def hold(ticks, gpos):
    if flags["ended"]:
        return
    st = get_robot_state()
    for tick in range(ticks):
        if flags["ended"]:
            return
        try:
            reply = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=st.joint_positions["primary"],
                gripper_position=gpos, embodiment=st.embodiment)}))
        except Exception as exc:
            note_err("hold", exc)
            return
        check_end(reply)


def ee_now():
    st = get_robot_state()
    p = st.end_effector_poses["primary"].position
    return [float(p[0]), float(p[1]), float(p[2])], float(st.gripper_positions["primary"])


def relocate_cube(expect_xy):
    best = None
    for c in candidates("cube"):
        f = c[3]
        if f["n"] < 10:
            continue
        a = f["top_a"]
        b = f["top_b"]
        if not (0.012 <= a <= 0.09 and 0.012 <= b <= 0.09):
            continue
        if f["z95"] > table_z + cube_side + 0.02:
            continue
        d = math.sqrt((f["top_cx"] - expect_xy[0]) ** 2 + (f["top_cy"] - expect_xy[1]) ** 2)
        if d < 0.08 and (best is None or d < best[0]):
            best = (d, f)
    if best is None:
        return None
    return (best[1]["top_cx"], best[1]["top_cy"])


def push_quat(theta):
    return quat_for_closing(theta + math.pi / 2.0)


def ticks_used():
    try:
        return float(get_robot_state().timestamp_s) * 20.0
    except Exception as exc:
        note_err("ticks", exc)
        return 0.0


def push_loop(tool_offset, push_z, quat_fn, rounds=4):
    """Push the cube toward the target; tool_offset = distance from EE xy to cube centre at contact."""
    cxy = (cube[3]["top_cx"], cube[3]["top_cy"])
    txy = (target[3]["cx"], target[3]["cy"])
    log = []
    for rnd in range(rounds):
        if flags["ended"]:
            break
        dx = txy[0] - cxy[0]
        dy = txy[1] - cxy[1]
        dist = math.sqrt(dx * dx + dy * dy)
        log.append({"round": rnd, "cube": [r4(cxy[0]), r4(cxy[1])], "dist": r4(dist)})
        if dist < 0.012:
            break
        if max_steps > 0 and ticks_used() > 0.8 * max_steps:
            log.append({"stop": "tick budget", "ticks": r4(ticks_used())})
            break
        ux = dx / dist
        uy = dy / dist
        q = quat_fn(math.atan2(uy, ux))
        pre = (cxy[0] - ux * (tool_offset + 0.03), cxy[1] - uy * (tool_offset + 0.03))
        go((pre[0], pre[1], push_z + 0.07), q, "push_above_" + str(rnd), tol=0.015)
        go((pre[0], pre[1], push_z), q, "push_down_" + str(rnd), tol=0.01)
        travel = dist + 0.03
        nseg = int(travel / 0.025) + 1
        for k in range(1, nseg + 1):
            frac = k / float(nseg)
            px = pre[0] + ux * travel * frac
            py = pre[1] + uy * travel * frac
            go((px, py, push_z), q, "push_" + str(rnd) + "_" + str(k), tol=0.01, ms=60)
        # back off and lift so the front camera sees the cube
        endp = (pre[0] + ux * travel, pre[1] + uy * travel)
        go((endp[0] - ux * 0.03, endp[1] - uy * 0.03, push_z + 0.0), q, "push_back_" + str(rnd), tol=0.015, ms=60)
        go((endp[0] - ux * 0.03, endp[1] - uy * 0.03, push_z + 0.12), q, "push_lift_" + str(rnd), tol=0.02)
        if flags["ended"]:
            break
        expect = (endp[0] + ux * tool_offset, endp[1] + uy * tool_offset)
        found = relocate_cube(expect)
        log.append({"relocated": None if found is None else [r4(found[0]), r4(found[1])]})
        cxy = found if found is not None else expect
    report["push_log"] = log


Q7_HOME = 0.785
Q7_LIMIT = 2.7
Q7_GRASP_UNC = 0.15
Q7_PULL_UNC = 0.85


def wrap_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


def q7_estimate(x, y, closing):
    # joint-7 estimate: home value + base yaw - hand yaw offset (measured sign on this arm)
    return wrap_pi(Q7_HOME + math.atan2(y, x) - wrap_pi(closing - home_close_angle))


def quat_closing_exact(theta):
    delta = wrap_pi(theta - home_close_angle)
    rz = [math.cos(delta / 2.0), 0.0, 0.0, math.sin(delta / 2.0)]
    return qmul(rz, home_quat)


def hook_pull():
    tm = tool
    cxy = (cube[3]["top_cx"], cube[3]["top_cy"])
    txy = (target[3]["cx"], target[3]["cy"])
    dx = txy[0] - cxy[0]
    dy = txy[1] - cxy[1]
    dist = math.sqrt(dx * dx + dy * dy)
    d = (dx / dist, dy / dist)
    ax_ang = tm["axis_ang"]
    # desired shaft direction (handle->hook) after rotation: from hand through cube, away from target
    want = math.atan2(-d[1], -d[0])
    delta = wrap_pi(want - ax_ang)
    clear = 0.018
    s_half = cube_side / 2.0
    off = min(s_half + 0.02, tm["bar_len"] - s_half - 0.002)
    off = max(off, s_half + 0.012)
    r_t = (tm["t_in"] - s_half - clear) - tm["t_g"]
    r_l = (tm["stick_l"] + tm["sigma"] * off) - tm["l_g"]
    ap = (-d[0], -d[1])
    npv = (-ap[1], ap[0])
    gx = cxy[0] - ap[0] * r_t - npv[0] * r_l
    gy = cxy[1] - ap[1] * r_t - npv[1] * r_l
    # grasp yaw: closing axis perpendicular to the shaft; pick the option whose grasp and
    # final joint-7 estimates both stay inside the limit
    # The planar estimate is accurate at transit height, but at the LOW pull pose near the
    # base the wrist folds and joint 7 measured ~0.8 rad away from it (ep0 v03: est 1.93,
    # actual 2.72 -> saturated at 2.75 during the pull, hand drifted 1.4 cm sideways).
    # Score each grasp-yaw option by the worst case of that uncertainty.
    best = None
    opt_log = []
    for opt in (ax_ang + math.pi / 2.0, ax_ang - math.pi / 2.0):
        q7g = q7_estimate(tm["grasp_x"], tm["grasp_y"], opt)
        q7f = q7_estimate(gx, gy, opt + delta)
        margin = Q7_LIMIT - max(abs(q7g) + Q7_GRASP_UNC, abs(q7f) + Q7_PULL_UNC)
        opt_log.append([r4(q7g), r4(q7f), r4(margin)])
        if best is None or margin > best[0]:
            best = (margin, opt, q7g, q7f)
    report["q7_options"] = opt_log
    close_g = best[1]
    close_f = close_g + delta
    plan = {"d": [r4(d[0]), r4(d[1])], "dist": r4(dist), "delta_deg": round(math.degrees(delta), 1),
            "grasp_closing_deg": round(math.degrees(wrap_pi(close_g)), 1),
            "final_closing_deg": round(math.degrees(wrap_pi(close_f)), 1),
            "q7_grasp_est": r4(best[2]), "q7_final_est": r4(best[3]), "margin": r4(best[0]),
            "pull_start": [r4(gx), r4(gy)], "r_t": r4(r_t), "r_l": r4(r_l), "off": r4(off)}
    report["hook_plan"] = plan
    qg = quat_closing_exact(close_g)
    qf = quat_closing_exact(close_f)
    grasp_z = max(tm["handle_top"] - 0.006, table_z + 0.014)
    pull_z = grasp_z + 0.010
    transit_z = max(0.32, table_z + 0.31)
    hx = tm["grasp_x"]
    hy = tm["grasp_y"]
    grip(False, "open")
    go((hx, hy, grasp_z + 0.10), qg, "tool_pre", tol=0.015, ms=200)
    go((hx, hy, grasp_z), qg, "tool_grasp", tol=0.008)
    grip(True, "tool_close")
    if flags["ended"]:
        return
    pos, g = ee_now()
    report["tool_grip_reading"] = r4(g)
    if g < 0.15:
        report["abort"] = "tool grasp missed (reading " + str(r4(g)) + ")"
        go((hx, hy, grasp_z + 0.10), qg, "tool_missed_lift", tol=0.02)
        return
    go((hx, hy, transit_z), qg, "tool_lift", tol=0.02, ms=200)
    off_g = wrap_pi(close_g - home_close_angle)
    off_f = wrap_pi(close_f - home_close_angle)
    nrot = int(abs(off_f - off_g) / 0.6) + 1
    rot_log = []
    for k in range(1, nrot + 1):
        frac = k / float(nrot)
        theta = home_close_angle + off_g + (off_f - off_g) * frac
        px = hx + (gx - hx) * frac
        py = hy + (gy - hy) * frac
        res = go((px, py, transit_z), quat_closing_exact(theta), "tool_rotate_" + str(k), tol=0.02, ms=200)
        ok = False
        try:
            ok = bool(res.ok)
        except Exception as exc:
            note_err("rot ok", exc)
        rot_log.append(ok)
        if flags["ended"]:
            return
    report["rotation_steps_ok"] = rot_log
    if not rot_log[-1]:
        report["abort"] = "final tool orientation not reached at transit height"
        return
    go((gx, gy, pull_z + 0.05), qf, "tool_lower", tol=0.015, ms=200)
    go((gx, gy, pull_z), qf, "tool_down", tol=0.01)
    if flags["ended"]:
        return
    pos, g = ee_now()
    report["tool_reading_at_pull"] = r4(g)
    # stage 1: close the gap, then drag most of the way
    travel1 = clear + 0.75 * dist
    done = 0.0
    nseg = int(travel1 / 0.025) + 1
    for k in range(1, nseg + 1):
        done = travel1 * k / float(nseg)
        go((gx + d[0] * done, gy + d[1] * done, pull_z), qf, "pull1_" + str(k), tol=0.01, ms=60)
    if flags["ended"]:
        return
    expect = (cxy[0] + d[0] * (done - clear), cxy[1] + d[1] * (done - clear))
    found = relocate_cube(expect)
    remaining = dist - (done - clear)
    if found is not None:
        remaining = (txy[0] - found[0]) * d[0] + (txy[1] - found[1]) * d[1]
    report["pull_check"] = {"expect": [r4(expect[0]), r4(expect[1])],
                            "found": None if found is None else [r4(found[0]), r4(found[1])],
                            "remaining": r4(remaining)}
    # ep0 v03: the bar gap closed ~0.6 cm during pull2 (cube overshot); bias slightly short
    remaining = max(0.0, min(remaining - 0.005, 0.10))
    if remaining > 0.006:
        nseg = int(remaining / 0.025) + 1
        start = done
        for k in range(1, nseg + 1):
            done = start + remaining * k / float(nseg)
            go((gx + d[0] * done, gy + d[1] * done, pull_z), qf, "pull2_" + str(k), tol=0.01, ms=60)
    hold(5, 0.0)
    if not flags["ended"]:
        found = relocate_cube((txy[0], txy[1]))
        report["final_cube"] = None if found is None else [r4(found[0]), r4(found[1])]


# ---------------------------------------------------------------- execute
report["manner_used"] = manner
report["max_steps"] = max_steps
try:
    if cube is None or target is None:
        report["abort"] = "missing cube or target"
        grip(False, "open_for_keyframe")
    elif manner == "pick":
        cf = cube[3]
        yaw_cube = cf["top_yaw"]
        q = quat_for_closing(yaw_cube + math.pi / 2.0)
        cx = cf["top_cx"]
        cy = cf["top_cy"]
        grasp_z = cf["z95"] - min(0.02, cube_side / 2.0)
        grip(False, "open")
        go((cx, cy, grasp_z + 0.10), q, "pre_grasp", tol=0.015)
        go((cx, cy, grasp_z), q, "grasp", tol=0.01)
        grip(True, "close")
        go((cx, cy, grasp_z + 0.15), q, "lift", tol=0.02)
        if not flags["ended"]:
            pos, g = ee_now()
            report["held_gripper"] = r4(g)
        tx = target[3]["cx"]
        ty = target[3]["cy"]
        go((tx, ty, grasp_z + 0.15), q, "above_target", tol=0.02)
        go((tx, ty, table_z + cube_side / 2.0 + 0.012), q, "place", tol=0.01)
        grip(False, "release")
        hold(6, 1.0)
        go((tx, ty, table_z + cube_side + 0.12), q, "retreat", tol=0.02)
    elif manner == "peg" and peg is not None:
        hook_pull()
    else:
        if manner == "peg":
            report["fallback"] = "peg not found -> gripper push"
        grip(True, "close_for_push")

        push_loop(cube_side / 2.0 + 0.012, table_z + 0.018, push_quat)
    if manner != "peg":
        hold(5, 1.0 if manner == "pick" else 0.0)
except Exception as exc:
    note_err("execute", exc)

if not flags["ended"]:
    try:
        pos, g = ee_now()
        report["final_ee"] = [r4(v) for v in pos]
        report["final_gripper"] = r4(g)
        report["final_t"] = r4(get_robot_state().timestamp_s)
    except Exception as exc:
        note_err("final", exc)
report["ended"] = flags["ended"]
result = report
