# v01_hook_pull — robomme_plus/task_4 MoveCube (AfI, R=1). UNTESTED trace-guided repair of v00.
# v00 failed episode 0 (0/1): (A) tool identity — took the hammer handle ("wooden stick") instead of
# the L-hook the video used; (B) grasp retention — centre grasp on that handle slipped on lift;
# (C) manner violation — fell back to a gripper push; the episode terminated as a failure at tick 87.
# v01 changes ONLY the tool branch: part/object grouping of every front candidate, tool chosen by
# matching the demonstrated grip width (video g_hold * finger span), handle grasp at the video grasp
# height with closing axis across the handle, rigid-tool planner that places the tool's contact
# edge behind the cube (obstacle-, reach- and yaw-checked) and pulls/pushes along cube->target,
# re-look and correction passes, and never touches the cube with the gripper in tool manner.
# Snapshot AND attempt in one start. Phases:
#   P0 snapshot (language, metadata, cameras, robot state)
#   P1 conditioning-video summary + manner classification from recorded robot_state
#   P2 SAM3 prompt probes on front + wrist, geometry of every top candidate
#   P3 select cube / target / peg / table by projected geometry
#   P4 decide manner (language first, then video, then default "pick")
#   P5 execute manner (pick | push | peg) with wrist refinement and verification
#   P6 final verification + settle ticks
# Never raises: every phase is wrapped; `result` is assigned early and mutated.
import math

report = {
    "version": "v01_hook_pull",
    "errors": [],
    "log": [],
    "snapshot": {},
    "video": {},
    "probes": [],
    "selection": {},
    "decision": {},
    "motions": [],
    "execution": {},
    "final": {},
}
result = report
flags = {"done": False, "hz": 20.0, "max_steps": 300}


def note(msg):
    text = str(msg)
    print("[v01] " + text)
    if len(report["log"]) < 400:
        report["log"].append(text)


def err(phase, exc):
    text = phase + ": " + str(exc)
    print("[v01][ERR] " + text)
    if len(report["errors"]) < 200:
        report["errors"].append(text)


def r4(value):
    try:
        return round(float(value), 4)
    except Exception:
        return None


def vec(values):
    out = []
    for value in values:
        out.append(r4(value))
    return out


def quantile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    index = int((len(ordered) - 1) * fraction)
    return ordered[index]


def median(values):
    return quantile(values, 0.5)


# ---------------------------------------------------------------- quaternion helpers (wxyz)
def qmul(a, b):
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def qnorm(q):
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n < 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    return [q[0] / n, q[1] / n, q[2] / n, q[3] / n]


def qmat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def qyaw(deg):
    half = math.radians(deg) / 2.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def wrap90(deg):
    value = float(deg)
    for guard in range(8):
        if value > 90.0:
            value -= 180.0
        elif value <= -90.0:
            value += 180.0
    return value


def closing_axis_deg(q):
    m = qmat(q)
    return math.degrees(math.atan2(m[1][1], m[0][1]))


# ---------------------------------------------------------------- state / time helpers
def ee_now():
    try:
        st = get_robot_state()
        pose = st.end_effector_poses["primary"]
        return [float(pose.position[0]), float(pose.position[1]), float(pose.position[2])], \
            [float(v) for v in pose.quaternion_wxyz]
    except Exception as exc:
        err("ee_now", exc)
        return None, None


def grip_now():
    try:
        return float(get_robot_state().gripper_positions["primary"])
    except Exception as exc:
        err("grip_now", exc)
        return None


def ticks_now():
    try:
        return int(round(float(get_robot_state().timestamp_s) * flags["hz"]))
    except Exception:
        return -1


def budget_left():
    used = ticks_now()
    if used < 0:
        return flags["max_steps"]
    return flags["max_steps"] - used


def is_terminal(res):
    term = False
    if res is None:
        return False
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
    try:
        status = str(res.status).lower()
        if "terminat" in status or "truncat" in status:
            term = True
    except Exception:
        pass
    return term


def res_summary(res):
    out = {}
    if res is None:
        return {"none": True}
    try:
        out["ok"] = bool(res.ok)
    except Exception:
        out["ok"] = None
    try:
        out["status"] = str(res.status)
    except Exception:
        pass
    try:
        out["steps"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        if res.error is not None:
            out["error"] = str(res.error)[:300]
    except Exception:
        pass
    try:
        out["terminated"] = bool(res.terminated)
    except Exception:
        pass
    try:
        out["truncated"] = bool(res.truncated)
    except Exception:
        pass
    return out


def after_action(res, label):
    rec = res_summary(res)
    rec["label"] = label
    rec["tick"] = ticks_now()
    pos, quat = ee_now()
    if pos is not None:
        rec["ee"] = vec(pos)
    rec["grip"] = r4(grip_now())
    if is_terminal(res):
        flags["done"] = True
        rec["terminal"] = True
        note("terminal after " + label)
    if len(report["motions"]) < 300:
        report["motions"].append(rec)
    return rec


def move(pos, quat, label, tol=0.01, max_steps=150):
    if flags["done"]:
        return None
    if budget_left() < 8:
        note("budget exhausted before " + label)
        flags["done"] = True
        return None
    target = Pose([float(pos[0]), float(pos[1]), float(pos[2])], [float(v) for v in quat],
                  report["snapshot"].get("base_frame", "robot_base"))
    res = None
    try:
        res = move_to_pose(target, tolerance=tol, max_steps=max_steps)
    except Exception as exc:
        err("move " + label, exc)
        res = None
    rec = after_action(res, label)
    rec["target"] = vec(pos)
    retry = False
    if res is None:
        retry = True
    else:
        try:
            if not res.ok and not flags["done"]:
                retry = True
        except Exception:
            retry = True
    if retry and not flags["done"] and budget_left() > 8:
        try:
            res = move_to_pose(target, tolerance=max(tol, 0.02), max_steps=max_steps * 2)
        except Exception as exc:
            err("move-retry " + label, exc)
            res = None
        rec2 = after_action(res, label + "/retry")
        rec2["target"] = vec(pos)
    return res


def gripper(action, label):
    if flags["done"]:
        return None
    res = None
    try:
        if action == "open":
            res = open_gripper()
        else:
            res = close_gripper()
    except Exception as exc:
        err("gripper " + label, exc)
    after_action(res, label)
    return res


def hold(ticks, grip_value, label):
    count = 0
    for tick in range(ticks):
        if flags["done"] or budget_left() < 2:
            break
        try:
            st = get_robot_state()
            reply = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=st.joint_positions["primary"],
                gripper_position=grip_value, embodiment=st.embodiment)}))
            count += 1
            if is_terminal(reply):
                flags["done"] = True
                note("terminal during hold " + label)
                break
        except Exception as exc:
            err("hold " + label, exc)
            break
    report["motions"].append({"label": label, "hold_ticks": count, "tick": ticks_now()})


# ---------------------------------------------------------------- perception helpers
def cloud_list(pc):
    try:
        pts = pc.points.tolist()
    except Exception:
        pts = []
        try:
            for p in pc.points:
                pts.append([float(p[0]), float(p[1]), float(p[2])])
        except Exception as exc:
            err("cloud_list", exc)
    if len(pts) > 2000:
        stride = int(len(pts) / 2000) + 1
        pts = pts[::stride]
    return pts


def npoints(pc):
    try:
        return len(pc.points)
    except Exception:
        return 0


def cand_cloud(cam, cand):
    base = report["snapshot"].get("base_frame", "robot_base")
    pc = None
    try:
        pc = mask_to_point_cloud(cand, cam, target_frame=base)
    except Exception as exc1:
        if len(report["errors"]) < 20:
            err("mask_to_point_cloud(candidate) " + cam, str(exc1)[:160])
    if pc is None or npoints(pc) < 6:
        try:
            pc2 = mask_to_point_cloud(cand.mask, cam, target_frame=base)
            if pc is None or npoints(pc2) > npoints(pc):
                pc = pc2
        except Exception as exc2:
            if len(report["errors"]) < 20:
                err("mask_to_point_cloud(candidate.mask) " + cam, str(exc2)[:160])
    return pc


def geom_from_points(pts):
    if len(pts) < 6:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    zs = [float(p[2]) for p in pts]
    z_hi = quantile(zs, 0.9)
    z_lo = quantile(zs, 0.1)
    top = [p for p in pts if float(p[2]) >= z_hi - 0.008]
    if len(top) < 3:
        top = pts
    tx = median([float(p[0]) for p in top])
    ty = median([float(p[1]) for p in top])
    mx = median(xs)
    my = median(ys)
    # robust PCA in xy on the 10-90% box
    x10, x90 = quantile(xs, 0.1), quantile(xs, 0.9)
    y10, y90 = quantile(ys, 0.1), quantile(ys, 0.9)
    inner = [p for p in pts if x10 <= p[0] <= x90 and y10 <= p[1] <= y90]
    if len(inner) < 6:
        inner = pts
    n = float(len(inner))
    ax = sum(float(p[0]) for p in inner) / n
    ay = sum(float(p[1]) for p in inner) / n
    sxx = sum((float(p[0]) - ax) ** 2 for p in inner) / n
    syy = sum((float(p[1]) - ay) ** 2 for p in inner) / n
    sxy = sum((float(p[0]) - ax) * (float(p[1]) - ay) for p in inner) / n
    theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    ux, uy = math.cos(theta), math.sin(theta)
    major = [(float(p[0]) - ax) * ux + (float(p[1]) - ay) * uy for p in pts]
    minor = [-(float(p[0]) - ax) * uy + (float(p[1]) - ay) * ux for p in pts]
    major_ext = quantile(major, 0.95) - quantile(major, 0.05)
    minor_ext = quantile(minor, 0.95) - quantile(minor, 0.05)
    return {
        "n": len(pts),
        "med": [r4(mx), r4(my), r4(median(zs))],
        "top_xy": [r4(tx), r4(ty)],
        "z_lo": r4(z_lo),
        "z_hi": r4(z_hi),
        "x_ext": r4(x90 - x10),
        "y_ext": r4(y90 - y10),
        "major_ext": r4(major_ext),
        "minor_ext": r4(minor_ext),
        "axis_deg": r4(math.degrees(theta)),
        "pca_center": [r4(ax), r4(ay)],
    }


def segment(cam, prompt, keep=4, with_geom=True):
    entry = {"camera": cam, "prompt": prompt, "ok": False, "cands": []}
    try:
        seg = segment_text(cam, prompt)
    except Exception as exc:
        err("segment_text " + cam + " " + prompt, exc)
        entry["error"] = str(exc)[:200]
        return entry, []
    try:
        entry["ok"] = bool(seg.ok)
    except Exception:
        pass
    found = []
    try:
        cands = list(seg.segmentations)
    except Exception:
        cands = []
    try:
        if not entry["ok"] and seg.error is not None:
            entry["error"] = str(seg.error)[:200]
    except Exception:
        pass
    entry["num"] = len(cands)
    for index, cand in enumerate(cands[:keep]):
        item = {"rank": index}
        try:
            item["score"] = r4(cand.score)
        except Exception:
            pass
        try:
            if cand.box_xyxy is not None:
                item["box"] = [round(float(v), 1) for v in cand.box_xyxy]
        except Exception:
            pass
        geom = None
        pts = None
        if with_geom:
            pc = cand_cloud(cam, cand)
            if pc is not None:
                pts = cloud_list(pc)
                geom = geom_from_points(pts)
        item["geom"] = geom
        entry["cands"].append(item)
        found.append({"camera": cam, "prompt": prompt, "rank": index,
                      "score": item.get("score", 0.0), "box": item.get("box"), "geom": geom,
                      "pts": thin(pts, 400) if pts else None})
    return entry, found


def size_of(geom, table_z):
    if geom is None:
        return None
    return float(geom["z_hi"]) - float(table_z)


def is_cube(geom, table_z):
    if geom is None or geom["n"] < 8:
        return False
    h = size_of(geom, table_z)
    ext = max(float(geom["major_ext"]), float(geom["minor_ext"]))
    if h is None or h < 0.012 or h > 0.10:
        return False
    if ext < 0.012 or ext > 0.10:
        return False
    if float(geom["minor_ext"]) > 1e-4 and float(geom["major_ext"]) / float(geom["minor_ext"]) > 3.0:
        return False
    return True


def is_flat_target(geom, table_z):
    if geom is None or geom["n"] < 8:
        return False
    h = size_of(geom, table_z)
    ext = max(float(geom["major_ext"]), float(geom["minor_ext"]))
    return h is not None and h < 0.015 and 0.025 <= ext <= 0.40


def is_peg(geom, table_z):
    if geom is None or geom["n"] < 8:
        return False
    h = size_of(geom, table_z)
    major = float(geom["major_ext"])
    minor = max(float(geom["minor_ext"]), 1e-4)
    return h is not None and h < 0.08 and 0.06 <= major <= 0.5 and major / minor >= 2.5


def dist2(a, b):
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)


def best_by_score(cands):
    best = None
    for cand in cands:
        if best is None or float(cand.get("score") or 0.0) > float(best.get("score") or 0.0):
            best = cand
    return best


def wrap180(deg):
    value = float(deg)
    for guard in range(8):
        if value > 180.0:
            value -= 360.0
        elif value <= -180.0:
            value += 360.0
    return value


def thin(pts, cap):
    if pts is None:
        return None
    if len(pts) <= cap:
        return pts
    stride = int(len(pts) / cap) + 1
    return pts[::stride]


def box_area(b):
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def box_iou(a, b):
    ix = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    iy = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    inter = ix * iy
    union = box_area(a) + box_area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def box_inside(inner, outer, tol):
    return (float(inner[0]) >= float(outer[0]) - tol and float(inner[1]) >= float(outer[1]) - tol
            and float(inner[2]) <= float(outer[2]) + tol and float(inner[3]) <= float(outer[3]) + tol)


def min_xy_dist(pts, xy):
    best = None
    for p in pts:
        d = (float(p[0]) - float(xy[0])) ** 2 + (float(p[1]) - float(xy[1])) ** 2
        if best is None or d < best:
            best = d
    if best is None:
        return None
    return math.sqrt(best)


# ---------------------------------------------------------------- image helpers
def thumb(rgb, stride):
    try:
        return rgb[::stride, ::stride].tolist()
    except Exception:
        pass
    out = []
    try:
        h = len(rgb)
        w = len(rgb[0])
        for y in range(0, h, stride):
            row = []
            for x in range(0, w, stride):
                px = rgb[y][x]
                row.append([int(px[0]), int(px[1]), int(px[2])])
            out.append(row)
    except Exception as exc:
        err("thumb", exc)
    return out


def thumb_diff(a, b):
    try:
        total = 0.0
        count = 0
        cells = []
        for y in range(min(len(a), len(b))):
            for x in range(min(len(a[y]), len(b[y]))):
                d = abs(int(a[y][x][0]) - int(b[y][x][0])) + abs(int(a[y][x][1]) - int(b[y][x][1])) \
                    + abs(int(a[y][x][2]) - int(b[y][x][2]))
                total += d
                count += 1
                if d > 60:
                    cells.append([d, y, x])
        cells = sorted(cells, reverse=True)[:24]
        return {"mean_abs": r4(total / max(count, 1)), "hot_cells_d_y_x": cells}
    except Exception as exc:
        err("thumb_diff", exc)
        return None


# =============================================================================== P0 snapshot
try:
    ctx = get_task_context()
    language = str(ctx.language)
    report["snapshot"]["language"] = language
    note("language: " + language)
    try:
        report["snapshot"]["task_name"] = str(ctx.task_name)
        report["snapshot"]["task_id"] = str(ctx.task_id)
        report["snapshot"]["suite"] = str(ctx.suite)
        report["snapshot"]["family"] = str(ctx.family)
    except Exception as exc:
        err("ctx fields", exc)
    try:
        report["snapshot"]["ctx_metadata"] = str(dict(ctx.metadata))[:2000]
    except Exception as exc:
        err("ctx metadata", exc)
except Exception as exc:
    language = ""
    err("P0 context", exc)

try:
    tmeta = robomme_plus.get_task_metadata()
    tmd = {}
    for key in tmeta:
        tmd[str(key)] = str(tmeta[key])[:400]
    report["snapshot"]["task_metadata"] = tmd
    try:
        flags["max_steps"] = int(tmeta["max_steps"])
    except Exception as exc:
        err("max_steps", exc)
    for key in ("control_hz", "control_frequency", "control_freq", "frequency_hz", "hz"):
        try:
            flags["hz"] = float(tmeta[key])
            break
        except Exception:
            pass
except Exception as exc:
    err("P0 task metadata", exc)

try:
    cmeta = robomme_plus.get_controller_metadata()
    cmd = {}
    for key in cmeta:
        cmd[str(key)] = str(cmeta[key])[:400]
    report["snapshot"]["controller_metadata"] = cmd
except Exception as exc:
    err("P0 controller metadata", exc)
report["snapshot"]["max_steps"] = flags["max_steps"]
report["snapshot"]["hz"] = flags["hz"]
note("max_steps=" + str(flags["max_steps"]) + " hz=" + str(flags["hz"]))

live_thumbs = {}
camera_names = []
try:
    state0 = get_robot_state()
    report["snapshot"]["base_frame"] = str(state0.base_frame)
    report["snapshot"]["embodiment"] = str(state0.embodiment)
    report["snapshot"]["joints"] = vec(state0.joint_positions["primary"])
    report["snapshot"]["grip"] = r4(state0.gripper_positions["primary"])
    p0 = state0.end_effector_poses["primary"]
    report["snapshot"]["ee_pos"] = vec(p0.position)
    report["snapshot"]["ee_quat"] = vec(p0.quaternion_wxyz)
    report["snapshot"]["ee_frame"] = str(p0.frame)
    report["snapshot"]["timestamp_s"] = r4(state0.timestamp_s)
except Exception as exc:
    err("P0 robot state", exc)
    report["snapshot"]["base_frame"] = "robot_base"

try:
    obs0 = get_observation()
    cams = {}
    for name in obs0.cameras:
        camera_names.append(str(name))
        cam = obs0.cameras[name]
        info = {}
        try:
            info["rgb_shape"] = [int(v) for v in cam.rgb.shape]
        except Exception:
            pass
        try:
            info["depth_shape"] = [int(v) for v in cam.depth_m.shape]
        except Exception:
            pass
        try:
            info["intrinsics"] = [vec(row) for row in cam.intrinsics]
        except Exception:
            pass
        try:
            info["pose_pos"] = vec(cam.camera_pose.position)
            info["pose_quat"] = vec(cam.camera_pose.quaternion_wxyz)
            info["pose_frame"] = str(cam.camera_pose.frame)
        except Exception:
            pass
        try:
            dep = cam.depth_m[::16, ::16].tolist()
            flat = [float(v) for row in dep for v in row]
            info["depth_q"] = [r4(quantile(flat, f)) for f in (0.05, 0.5, 0.95)]
        except Exception:
            pass
        live_thumbs[str(name)] = thumb(cam.rgb, 8)
        cams[str(name)] = info
    report["snapshot"]["cameras"] = cams
    note("cameras: " + str(camera_names))
except Exception as exc:
    err("P0 observation", exc)
if "front" not in camera_names:
    camera_names.append("front")
FRONT = "front"
WRIST = "wrist" if "wrist" in camera_names else None

# home orientation (downward) and its closing axis
home_pos, home_quat = ee_now()
q_down = home_quat
try:
    m0 = qmat(home_quat)
    report["snapshot"]["ee_z_axis"] = vec([m0[0][2], m0[1][2], m0[2][2]])
    if m0[2][2] > -0.9:
        yaw0 = math.degrees(math.atan2(m0[1][0], m0[0][0]))
        q_down = qnorm(qmul(qyaw(yaw0), [0.0, 1.0, 0.0, 0.0]))
        note("home orientation not downward; using constructed downward quaternion")
    report["snapshot"]["q_down"] = vec(q_down)
    report["snapshot"]["closing_axis_deg"] = r4(closing_axis_deg(q_down))
except Exception as exc:
    err("P0 orientation", exc)
    q_down = [0.0, 1.0, 0.0, 0.0]

# =============================================================================== P1 video
video_info = report["video"]
traj = []
vthumbs = {}
try:
    video = robomme_plus.get_conditioning_video()
    try:
        frames = list(video)
    except Exception:
        frames = []
        for fr in video:
            frames.append(fr)
    nvid = len(frames)
    video_info["n_frames"] = nvid
    note("conditioning video frames: " + str(nvid))
    pick_idx = []
    if nvid > 0:
        for k in range(6):
            pick_idx.append(int(round(k * (nvid - 1) / 5.0)))
    for index, fr in enumerate(frames):
        try:
            rs = fr.robot_state
            pos = rs.end_effector_poses["primary"].position
            g = float(rs.gripper_positions["primary"])
            traj.append([float(fr.timestamp_s), float(pos[0]), float(pos[1]), float(pos[2]), g])
        except Exception as exc:
            if len(report["errors"]) < 50:
                err("video frame " + str(index), exc)
        if index in pick_idx:
            try:
                vthumbs[index] = thumb(fr.cameras["front"].rgb, 8)
            except Exception as exc:
                err("video thumb", exc)
    if nvid > 0:
        video_info["t_first"] = r4(traj[0][0]) if traj else None
        video_info["t_last"] = r4(traj[-1][0]) if traj else None
    # compact trajectory (every k-th frame)
    stride = max(1, int(len(traj) / 120))
    video_info["traj_t_x_y_z_g"] = [vec(row) for row in traj[::stride]]
    video_info["traj_stride"] = stride
    # thumbnails: front 32x32 for 6 frames, plus diffs
    video_info["thumb_idx"] = pick_idx
    video_info["thumbs_front_32"] = {}
    for index in pick_idx:
        if index in vthumbs:
            video_info["thumbs_front_32"][str(index)] = vthumbs[index]
    if "front" in live_thumbs:
        video_info["live_thumb_front_32"] = live_thumbs["front"]
        if len(pick_idx) > 0 and pick_idx[0] in vthumbs:
            video_info["diff_live_vs_first"] = thumb_diff(live_thumbs["front"], vthumbs[pick_idx[0]])
        if len(pick_idx) > 0 and pick_idx[-1] in vthumbs:
            video_info["diff_live_vs_last"] = thumb_diff(live_thumbs["front"], vthumbs[pick_idx[-1]])
        if len(pick_idx) > 1 and pick_idx[0] in vthumbs and pick_idx[-1] in vthumbs:
            video_info["diff_first_vs_last"] = thumb_diff(vthumbs[pick_idx[0]], vthumbs[pick_idx[-1]])
    frames = None
except Exception as exc:
    err("P1 video", exc)

# manner classification from the recorded robot state
video_cls = {"manner": None, "reasons": []}
try:
    if len(traj) >= 5:
        gs = [row[4] for row in traj]
        zs = [row[3] for row in traj]
        head = gs[: max(3, int(len(gs) / 10))]
        g_open = max(max(head), max(gs))
        g_min = min(gs)
        z_min = min(zs)
        video_cls["g_open"] = r4(g_open)
        video_cls["g_min"] = r4(g_min)
        video_cls["z_min"] = r4(z_min)
        video_cls["z_max"] = r4(max(zs))
        closed = [g < min(0.8, g_open - 0.15) for g in gs]
        segs = []
        start = None
        for index in range(len(closed)):
            if closed[index] and start is None:
                start = index
            if (not closed[index]) and start is not None:
                segs.append([start, index - 1])
                start = None
        if start is not None:
            segs.append([start, len(closed) - 1])
        segs = [s for s in segs if s[1] - s[0] >= 2]
        video_cls["closed_segments"] = segs
        # low horizontal path anywhere (z within 3 cm of the minimum)
        low_path = 0.0
        for index in range(1, len(traj)):
            if traj[index][3] < z_min + 0.03 and traj[index - 1][3] < z_min + 0.03:
                low_path += dist2(traj[index][1:3], traj[index - 1][1:3])
        video_cls["low_path_m"] = r4(low_path)
        last_low = None
        for index in range(1, len(traj)):
            if traj[index][3] < z_min + 0.03 and traj[index - 1][3] < z_min + 0.03:
                if dist2(traj[index][1:3], traj[index - 1][1:3]) > 0.002:
                    last_low = index
        if last_low is not None:
            back = max(0, last_low - 5)
            ddx = traj[last_low][1] - traj[back][1]
            ddy = traj[last_low][2] - traj[back][2]
            dn = math.sqrt(ddx * ddx + ddy * ddy)
            video_cls["last_low_xy"] = vec(traj[last_low][1:3])
            if dn > 1e-4:
                video_cls["last_low_dir"] = [r4(ddx / dn), r4(ddy / dn)]
        seg_info = []
        for s in segs:
            i0, i1 = s[0], s[1]
            # hold value: median over the stable part (skip the closing transient)
            mid = gs[i0 + int((i1 - i0) / 4): i1 + 1]
            g_hold = median(mid) if mid else gs[i0]
            z0 = traj[i0][3]
            zmax_i = i0
            for index in range(i0, i1 + 1):
                if traj[index][3] > traj[zmax_i][3]:
                    zmax_i = index
            lift = traj[zmax_i][3] - z0
            # first lift event: first frame after the grasp more than 3 cm above the grasp height
            lift_i = zmax_i
            for index in range(i0, i1 + 1):
                if traj[index][3] > z0 + 0.03:
                    lift_i = index
                    break
            sweep_after = 0.0
            low_z = []
            for index in range(lift_i + 1, i1 + 1):
                if traj[index][3] < z0 + 0.03 and traj[index - 1][3] < z0 + 0.03:
                    sweep_after += dist2(traj[index][1:3], traj[index - 1][1:3])
                    low_z.append(traj[index][3])
            sweep_before = 0.0
            for index in range(i0 + 1, lift_i + 1):
                if traj[index][3] < z0 + 0.03 and traj[index - 1][3] < z0 + 0.03:
                    sweep_before += dist2(traj[index][1:3], traj[index - 1][1:3])
            seg_info.append({
                "i0": i0, "i1": i1, "g_hold": r4(g_hold), "z0": r4(z0), "lift": r4(lift), "lift_i": lift_i,
                "sweep_after_lift": r4(sweep_after), "sweep_before_lift": r4(sweep_before),
                "grasp_xy": vec(traj[i0][1:3]), "release_xyz": vec(traj[i1][1:4]),
                "released": i1 < len(traj) - 1,
                "sweep_z_med": r4(median(low_z)) if low_z else None,
            })
        video_cls["segments"] = seg_info
        main = None
        for info in seg_info:
            if main is None or (info["i1"] - info["i0"]) > (main["i1"] - main["i0"]):
                main = info
        if main is None:
            if low_path > 0.05:
                video_cls["manner"] = "push"
                video_cls["reasons"].append("no closed segment, low horizontal path " + str(r4(low_path)))
            else:
                video_cls["reasons"].append("no closed segment and no low sweep")
        elif float(main["g_hold"]) < 0.06:
            if low_path > 0.04:
                video_cls["manner"] = "push"
                video_cls["reasons"].append("gripper closed on nothing, low sweep " + str(r4(low_path)))
            else:
                video_cls["reasons"].append("closed on nothing, no low sweep")
        else:
            if float(main["lift"]) > 0.03 and float(main["sweep_after_lift"]) > 0.05:
                video_cls["manner"] = "peg"
                video_cls["reasons"].append("held object, lifted, then low sweep while holding")
            elif float(main["lift"]) > 0.03:
                video_cls["manner"] = "pick"
                video_cls["reasons"].append("held object, lifted, released without low sweep")
            elif float(main["sweep_before_lift"]) > 0.05:
                video_cls["manner"] = "push"
                video_cls["reasons"].append("held something but no lift; low drag")
            else:
                video_cls["reasons"].append("held object but no lift and no sweep")
        video_cls["main_segment"] = main
    else:
        video_cls["reasons"].append("video trajectory too short: " + str(len(traj)))
except Exception as exc:
    err("P1 classify", exc)
video_info["classification"] = video_cls
note("video manner: " + str(video_cls.get("manner")) + " " + str(video_cls.get("reasons")))

# =============================================================================== P2 probes
CUBE_PROMPTS = ["cube", "small cube", "block", "red cube", "colored cube"]
lang_color = None
try:
    low_lang = language.lower()
    for color in ("red", "green", "blue", "yellow", "purple", "orange", "pink", "white", "black",
                  "brown", "gray", "grey", "cyan"):
        if (color + " cube") in low_lang or (color + " block") in low_lang:
            lang_color = color
            break
    if lang_color is not None and (lang_color + " cube") not in CUBE_PROMPTS:
        CUBE_PROMPTS.insert(0, lang_color + " cube")
except Exception as exc:
    err("P2 colour parse", exc)
report["selection"]["language_cube_color"] = lang_color
TARGET_PROMPTS = ["target", "red and white target", "circular target", "goal region",
                  "bullseye", "red circle", "white circle", "disc", "goal"]
PEG_PROMPTS = ["peg", "stick", "wooden stick", "rod"]
# v01: hook/handle prompts (never probed in v00; the L-hook's whole mask only appeared at low rank)
TOOL_PROMPTS = ["hook", "L-shaped hook", "hook tool", "L-shaped tool", "tool", "handle", "tool handle"]
pool = []
table_z = None
try:
    entry, found = segment(FRONT, "table", keep=1, with_geom=True)
    report["probes"].append(entry)
    if found and found[0]["geom"] is not None:
        table_z = float(found[0]["geom"]["med"][2])
        report["selection"]["table_z_from_table_mask"] = r4(table_z)
except Exception as exc:
    err("P2 table", exc)

for prompt in CUBE_PROMPTS + TARGET_PROMPTS + PEG_PROMPTS + TOOL_PROMPTS:
    try:
        entry, found = segment(FRONT, prompt, keep=4, with_geom=True)
        report["probes"].append(entry)
        for item in found:
            if prompt in CUBE_PROMPTS:
                item["role"] = "cube"
            elif prompt in TARGET_PROMPTS:
                item["role"] = "target"
            elif prompt in PEG_PROMPTS:
                item["role"] = "peg"
            else:
                item["role"] = "tool"
            pool.append(item)
    except Exception as exc:
        err("P2 probe " + prompt, exc)
if WRIST is not None:
    for prompt in ["cube", "target", "red and white target", "peg", "stick", "table"]:
        try:
            entry, found = segment(WRIST, prompt, keep=3, with_geom=True)
            report["probes"].append(entry)
        except Exception as exc:
            err("P2 wrist probe " + prompt, exc)
note("probes done: " + str(len(report["probes"])) + " pool " + str(len(pool)))

# =============================================================================== P3 selection
sel = report["selection"]
cube = None
target = None
peg = None
try:
    if table_z is None:
        lows = []
        for item in pool:
            if item["geom"] is not None:
                lows.append(float(item["geom"]["z_lo"]))
        if lows:
            table_z = quantile(lows, 0.2)
            sel["table_z_source"] = "low quantile of candidate clouds"
    if table_z is None:
        table_z = 0.0
        sel["table_z_source"] = "default 0"
    sel["table_z_initial"] = r4(table_z)
    cubes = [c for c in pool if c["role"] == "cube" and is_cube(c["geom"], table_z)]
    cube = best_by_score(cubes)
    if lang_color is not None:
        coloured = [c for c in cubes if c["prompt"] == lang_color + " cube"]
        if coloured:
            cube = best_by_score(coloured)
            sel["cube_rule"] = "language colour prompt"
    flats = []
    for c in pool:
        if c["role"] == "target" and is_flat_target(c["geom"], table_z):
            if cube is None or dist2(c["geom"]["med"], cube["geom"]["top_xy"]) > 0.03:
                flats.append(c)
    target = best_by_score(flats)
    pegs = [c for c in pool if c["role"] == "peg" and is_peg(c["geom"], table_z)]
    peg = best_by_score(pegs)
    # refine table z with the flat target (it lies on the table)
    if target is not None and sel.get("table_z_from_table_mask") is None:
        table_z = float(target["geom"]["z_lo"])
        sel["table_z_source"] = "target z_lo"
    sel["table_z"] = r4(table_z)
    if target is not None:
        sel["target_z_lo_vs_table"] = r4(float(target["geom"]["z_lo"]) - table_z)
    if cube is not None:
        sel["cube_z_lo_vs_table"] = r4(float(cube["geom"]["z_lo"]) - table_z)
    sel["n_cube_like"] = len(cubes)
    sel["n_flat"] = len(flats)
    sel["n_peg_like"] = len(pegs)
    sel["peg_note"] = "peg = v00 best-score elongated rule, logged only; v01 executes the P3b tool"
    for name, item in (("cube", cube), ("target", target), ("peg", peg)):
        if item is not None:
            sel[name] = {"prompt": item["prompt"], "rank": item["rank"], "score": item["score"],
                         "box": item["box"], "geom": item["geom"]}
        else:
            sel[name] = None
    note("selected cube=" + str(sel["cube"] is not None) + " target=" + str(sel["target"] is not None)
         + " peg=" + str(sel["peg"] is not None) + " table_z=" + str(r4(table_z)))
except Exception as exc:
    err("P3 selection", exc)
    if table_z is None:
        table_z = 0.0

# fallback: localize_object (top-1 mask) when the candidate pool gave nothing usable
def localize_fallback(prompts, kind):
    for prompt in prompts:
        try:
            found = localize_object(prompt, camera_name=FRONT,
                                    target_frame=report["snapshot"].get("base_frame", "robot_base"))
            if not found.ok:
                continue
            geom = geom_from_points(cloud_list(found.point_cloud))
            item = {"camera": FRONT, "prompt": prompt, "rank": 0, "score": 0.0, "box": None,
                    "geom": geom, "role": kind, "via": "localize_object"}
            sel.setdefault("localize_fallback", []).append({"prompt": prompt, "geom": geom})
            if geom is None:
                continue
            if kind == "cube" and is_cube(geom, table_z):
                return item
            if kind == "target" and is_flat_target(geom, table_z):
                return item
        except Exception as exc:
            err("localize_fallback " + prompt, exc)
    return None


try:
    if cube is None:
        cube = localize_fallback(CUBE_PROMPTS[:3], "cube")
        if cube is not None:
            sel["cube"] = {"prompt": cube["prompt"], "via": "localize_object", "geom": cube["geom"]}
    if target is None:
        target = localize_fallback(TARGET_PROMPTS[:4], "target")
        if target is not None:
            sel["target"] = {"prompt": target["prompt"], "via": "localize_object", "geom": target["geom"]}
except Exception as exc:
    err("P3 fallback", exc)

cube_xy = None
cube_size = 0.04
target_xy = None
try:
    # keep the table plane consistent with the chosen cube's own lowest visible points
    if cube is not None and cube["geom"] is not None:
        cz_lo = float(cube["geom"]["z_lo"])
        if table_z < cz_lo - 0.015 or table_z > cz_lo + 0.005:
            sel["table_z_adjusted_from"] = r4(table_z)
            table_z = cz_lo - 0.004
            sel["table_z"] = r4(table_z)
            note("table_z adjusted to cube bottom: " + str(r4(table_z)))
except Exception as exc:
    err("P3 table consistency", exc)

try:
    if cube is not None:
        cube_xy = [float(cube["geom"]["top_xy"][0]), float(cube["geom"]["top_xy"][1])]
        cube_size = min(max(float(cube["geom"]["z_hi"]) - table_z, 0.02), 0.07)
    if target is not None:
        target_xy = [float(target["geom"]["med"][0]), float(target["geom"]["med"][1])]
    sel["cube_xy_front"] = vec(cube_xy) if cube_xy else None
    sel["cube_size"] = r4(cube_size)
    sel["target_xy_front"] = vec(target_xy) if target_xy else None
except Exception as exc:
    err("P3 derive", exc)

# video-derived cross-checks (same episode; scene may differ, so only a fallback)
try:
    main = video_cls.get("main_segment")
    vgoal = None
    if main is not None and main.get("released") and video_cls.get("manner") == "pick":
        vgoal = [float(main["release_xyz"][0]), float(main["release_xyz"][1])]
    sel["video_goal_xy_estimate"] = vec(vgoal) if vgoal else None
    if main is not None:
        gxy = main["grasp_xy"]
        if cube_xy is not None:
            sel["video_grasp_to_live_cube_m"] = r4(dist2(gxy, cube_xy))
        if peg is not None:
            sel["video_grasp_to_live_peg_m"] = r4(dist2(gxy, peg["geom"]["med"]))
    if vgoal is not None and target_xy is not None:
        sel["video_goal_to_live_target_m"] = r4(dist2(vgoal, target_xy))
    if vgoal is None and video_cls.get("manner") == "push" and video_cls.get("last_low_dir"):
        lx = video_cls["last_low_xy"]
        ld = video_cls["last_low_dir"]
        vgoal = [float(lx[0]) + float(ld[0]) * (0.5 * cube_size + 0.012),
                 float(lx[1]) + float(ld[1]) * (0.5 * cube_size + 0.012)]
        sel["video_goal_xy_estimate"] = vec(vgoal)
    if vgoal is not None and target_xy is not None:
        sel["video_goal_to_live_target_m"] = r4(dist2(vgoal, target_xy))
    if target_xy is None and vgoal is not None:
        target_xy = vgoal
        sel["target_source"] = "video release point (no live flat target found)"
except Exception as exc:
    err("P3 video cross-check", exc)

# =============================================================================== P3b tool selection (v01)
# Mode A (episode 0): v00 took the best-scoring "wooden stick" (a hammer handle, 1.3 cm wide) while
# the video's tool was an L-hook whose teal handle is as wide as the demonstrator's grip.
# v01 groups every front candidate into parts, keeps maximal low elongated objects as tool
# candidates, enumerates graspable regions (contained compact parts + both ends + middle) and picks
# the region whose width matches the demonstrated grip width w_video = g_hold * GRIP_SPAN.
GRIP_SPAN = 0.08   # Panda finger opening at normalized gripper 1.0 (robot spec, not a scene value)
HAND_R = 0.045     # generic clearance radius around the TCP for fingers and hand
OBS_R = 0.012      # generic clearance between a tool point and an obstacle point
tsel = report.setdefault("tool_selection", {})
img_area = 65536.0
try:
    shp = report["snapshot"]["cameras"][FRONT]["rgb_shape"]
    img_area = float(shp[0]) * float(shp[1])
except Exception as exc:
    err("P3b image size", exc)


def build_parts(items):
    parts = []
    for item in items:
        if item.get("camera") != FRONT or item.get("geom") is None or item.get("pts") is None:
            continue
        if item.get("box") is None or len(item["pts"]) < 8:
            continue
        dup = None
        for p in parts:
            if box_iou(p["box"], item["box"]) > 0.85:
                dup = p
                break
        sc = float(item.get("score") or 0.0)
        if dup is not None:
            if len(dup["prompts"]) < 12:
                dup["prompts"].append([item["prompt"], r4(sc)])
            if sc > dup["score"]:
                dup["score"] = sc
            continue
        parts.append({"box": item["box"], "geom": item["geom"], "pts": item["pts"], "score": sc,
                      "prompt": item["prompt"], "prompts": [[item["prompt"], r4(sc)]]})
    return parts


def tool_regions(tool, parts):
    regions = []
    tb = tool["box"]
    for p in parts:
        if p is tool or not box_inside(p["box"], tb, 3.0):
            continue
        g = p["geom"]
        if g["n"] < 15 or float(g["major_ext"]) > 0.10 or float(g["minor_ext"]) < 0.006:
            continue
        if float(g["z_hi"]) - table_z < 0.006:
            continue
        dmin = min_xy_dist(tool["pts"], g["med"])
        if dmin is None or dmin > 0.015:
            continue
        regions.append({"src": "part", "prompt": p["prompt"], "score": r4(p["score"]),
                        "xy": [float(g["med"][0]), float(g["med"][1])], "w": float(g["minor_ext"]),
                        "len": float(g["major_ext"]), "axis": float(g["axis_deg"]),
                        "h": float(g["z_hi"]) - table_z})
    tg = tool["geom"]
    th = math.radians(float(tg["axis_deg"]))
    ux, uy = math.cos(th), math.sin(th)
    cx, cy = float(tg["pca_center"][0]), float(tg["pca_center"][1])
    pts = tool["pts"]
    proj = [(float(p[0]) - cx) * ux + (float(p[1]) - cy) * uy for p in pts]
    lo = quantile(proj, 0.03)
    hi = quantile(proj, 0.97)
    if lo is None or hi is None:
        return regions
    mid = 0.5 * (lo + hi)
    for name, a0, a1 in (("end_lo", lo, lo + 0.06), ("end_hi", hi - 0.06, hi), ("middle", mid - 0.03, mid + 0.03)):
        sub = [p for p, s in zip(pts, proj) if a0 <= s <= a1]
        g = geom_from_points(sub)
        if g is None or g["n"] < 8:
            continue
        regions.append({"src": name, "prompt": tool["prompt"], "score": r4(tool["score"]),
                        "xy": [float(g["med"][0]), float(g["med"][1])], "w": float(g["minor_ext"]),
                        "len": float(g["major_ext"]), "axis": float(g["axis_deg"]),
                        "h": float(g["z_hi"]) - table_z})
    return regions


def select_tool(items, cxy, w_video):
    parts = build_parts(items)
    tools = []
    for p in parts:
        g = p["geom"]
        h = float(g["z_hi"]) - table_z
        major = float(g["major_ext"])
        if major < 0.12 or major > 0.50 or h > 0.06 or h < 0.004:
            continue
        if box_area(p["box"]) > 0.2 * img_area:
            continue
        if cxy is not None:
            dmin = min_xy_dist(p["pts"], cxy)
            if dmin is not None and dmin < 0.5 * cube_size + 0.01:
                continue
        near_base = None
        for q in p["pts"]:
            rr = math.sqrt(float(q[0]) ** 2 + float(q[1]) ** 2)
            if near_base is None or rr < near_base:
                near_base = rr
        if near_base is not None and near_base < 0.12:
            continue
        tools.append(p)
    maximal = []
    for t in tools:
        inside = False
        for o in tools:
            if o is not t and box_inside(t["box"], o["box"], 3.0) and o["geom"]["n"] > t["geom"]["n"]:
                inside = True
                break
        if not inside:
            maximal.append(t)
    best = None
    best_cost = None
    log = []
    for t in maximal:
        hook_bonus = 0.0
        for pr in t["prompts"]:
            if "hook" in str(pr[0]) or "tool" in str(pr[0]):
                hook_bonus = max(hook_bonus, float(pr[1] or 0.0))
        for r in tool_regions(t, parts):
            if r["w"] > 0.07 or r["w"] < 0.006:
                continue
            if w_video is not None:
                cost = abs(r["w"] - w_video)
            else:
                cost = 0.02 - 0.01 * min(1.0, float(r["score"] or 0.0))
            if r["src"] != "part":
                cost += 0.003
            cost -= 0.002 * min(1.0, hook_bonus)
            if len(log) < 60:
                log.append({"tool_box": t["box"], "tool_prompt": t["prompt"], "src": r["src"],
                            "prompt": r["prompt"], "xy": vec(r["xy"]), "w": r4(r["w"]), "h": r4(r["h"]),
                            "axis": r4(r["axis"]), "cost": r4(cost)})
            if best is None or cost < best_cost:
                best = {"tool": t, "region": r}
                best_cost = cost
    return best, best_cost, parts, maximal, log


def make_tool(found, cost):
    t = found["tool"]
    r = found["region"]
    return {"pts": t["pts"], "geom": t["geom"], "box": t["box"], "prompt": t["prompt"],
            "prompts": t["prompts"], "region": r, "cost": cost}


tool = None
tool_parts = []
w_video = None
try:
    main_seg = video_cls.get("main_segment")
    if main_seg is not None and video_cls.get("manner") == "peg":
        w_video = float(main_seg["g_hold"]) * GRIP_SPAN
    tsel["w_video"] = r4(w_video)
    found, cost, tool_parts, maximal, tlog = select_tool(pool, cube_xy, w_video)
    tsel["n_parts"] = len(tool_parts)
    tsel["n_tools"] = len(maximal)
    tsel["tools"] = [{"box": t["box"], "prompts": t["prompts"][:6], "geom": t["geom"]} for t in maximal[:8]]
    order = sorted([[float(e["cost"] if e["cost"] is not None else 9.0), k] for k, e in enumerate(tlog)])
    tsel["regions_by_cost"] = [tlog[pair[1]] for pair in order[:12]]
    if found is not None:
        tool = make_tool(found, cost)
        reg = tool["region"]
        tsel["chosen"] = {"tool_box": tool["box"], "tool_prompts": tool["prompts"][:6],
                          "tool_geom": tool["geom"], "region": {"src": reg["src"], "prompt": reg["prompt"],
                                                                "xy": vec(reg["xy"]), "w": r4(reg["w"]),
                                                                "len": r4(reg["len"]), "axis": r4(reg["axis"]),
                                                                "h": r4(reg["h"])},
                          "cost": r4(cost)}
        if w_video is not None and cost is not None and cost > 0.012:
            tsel["width_mismatch"] = True
            note("tool region width does not match the demonstrated grip (cost " + str(r4(cost)) + ")")
        note("tool: box " + str(tool["box"]) + " region " + reg["src"] + "/" + reg["prompt"]
             + " w=" + str(r4(reg["w"])) + " vs video " + str(r4(w_video)))
    else:
        note("no tool candidate found")
except Exception as exc:
    err("P3b tool selection", exc)
    tool = None

# =============================================================================== P4 decision
dec = report["decision"]
manner = None
try:
    low = language.lower()
    words = set()
    for w in ("push", "slide", "nudge", "shove", "drag"):
        if w in low:
            words.add("push")
    for w in ("pick", "grasp", "lift", "carry", "grab"):
        if w in low:
            words.add("pick")
    for w in ("peg", "stick", "tool", "hook", "rod"):
        if w in low:
            words.add("peg")
    dec["language_manner_words"] = sorted(words)
    if len(words) == 1:
        manner = sorted(words)[0]
        dec["source"] = "language"
    elif "peg" in words and "push" in words and "pick" not in words:
        manner = "peg"
        dec["source"] = "language (tool + push)"
    vm = video_cls.get("manner")
    dec["video_manner"] = vm
    # identity cross-check of the grasped object in the video (pick vs peg)
    same_scene = False
    try:
        same_scene = float(video_info["diff_live_vs_first"]["mean_abs"]) < 8.0
    except Exception:
        same_scene = False
    dec["video_scene_matches_live"] = same_scene
    if vm in ("pick", "peg") and same_scene:
        dc = sel.get("video_grasp_to_live_cube_m")
        dp = sel.get("video_grasp_to_live_peg_m")
        if dc is not None and dp is not None:
            if dc < 0.05 and dp > 0.08:
                vm = "pick"
            elif dp < 0.05 and dc > 0.08:
                vm = "peg"
        dec["video_manner_after_identity_check"] = vm
    if manner is None and vm is not None:
        manner = vm
        dec["source"] = "video"
    if manner is None:
        manner = "pick"
        dec["source"] = "default"
    if manner == "peg" and tool is None:
        # v01: mode C — never substitute a gripper push for a demonstrated tool manner
        dec["tool_missing"] = True
        note("tool manner but no tool selected; will not touch the cube with the gripper")
    dec["manner"] = manner
    note("MANNER=" + manner + " via " + dec["source"])
except Exception as exc:
    err("P4 decision", exc)
    manner = "pick"
    dec["manner"] = manner


# =============================================================================== P5 execution
ex = report["execution"]
SAFE_DZ = 0.12


def relocalize(camera, prompts, near_xy, want, radius):
    """Return (xy, geom) of the best candidate of kind `want` within radius of near_xy."""
    best = None
    best_key = None
    tries = []
    for prompt in prompts:
        entry, found = segment(camera, prompt, keep=4, with_geom=True)
        tries.append({"prompt": prompt, "n": entry.get("num"),
                      "cands": [[c.get("score"), c["geom"]["top_xy"] if c["geom"] else None,
                                 c["geom"]["z_hi"] if c["geom"] else None] for c in found]})
        for c in found:
            g = c["geom"]
            if g is None:
                continue
            ok = is_cube(g, table_z) if want == "cube" else is_flat_target(g, table_z)
            if not ok:
                continue
            xy = g["top_xy"] if want == "cube" else g["med"]
            d = dist2(xy, near_xy) if near_xy is not None else 0.0
            if near_xy is not None and d > radius:
                continue
            key = d - 0.01 * float(c.get("score") or 0.0)
            if best is None or key < best_key:
                best = [[float(xy[0]), float(xy[1])], g]
                best_key = key
        if best is not None:
            break
    if len(ex.setdefault("relocalize", [])) < 30:
        ex["relocalize"].append({"camera": camera, "want": want, "tries": tries,
                                 "chosen": vec(best[0]) if best else None})
    return best


def refine_cube_from_wrist(xy):
    if WRIST is None or xy is None or flags["done"]:
        return xy
    move([xy[0], xy[1], table_z + 0.22], q_down, "hover above cube for wrist look", tol=0.015)
    if flags["done"]:
        return xy
    found = relocalize(WRIST, ["cube", "small cube", "block"], xy, "cube", 0.06)
    if found is not None:
        ex.setdefault("wrist_refine", []).append({"front": vec(xy), "wrist": vec(found[0])})
        try:
            global_size = float(found[1]["z_hi"]) - table_z
            ex["wrist_cube_height"] = r4(global_size)
        except Exception:
            pass
        return found[0]
    return xy


def look_cube_and_target(near_cube, near_target):
    """Hover between cube and target, look with wrist then front."""
    out = {"cube": None, "target": None}
    if flags["done"]:
        return out
    if WRIST is not None and near_cube is not None:
        mid = near_cube
        if near_target is not None:
            mid = [(near_cube[0] + near_target[0]) / 2.0, (near_cube[1] + near_target[1]) / 2.0]
        move([mid[0], mid[1], table_z + 0.28], q_down, "hover for verification look", tol=0.02)
        if flags["done"]:
            return out
        c = relocalize(WRIST, ["cube", "small cube"], near_cube, "cube", 0.15)
        if c is not None:
            out["cube"] = c[0]
        if near_target is not None:
            t = relocalize(WRIST, ["target", "red and white target", "circular target"], near_target, "target", 0.08)
            if t is not None:
                out["target"] = t[0]
    if out["cube"] is None:
        c = relocalize(FRONT, ["cube", "small cube", "block"], near_cube, "cube", 0.20)
        if c is not None:
            out["cube"] = c[0]
    return out


def do_pick(cxy, txy):
    size = cube_size
    gz = table_z + max(0.5 * size - 0.003, 0.012)
    gripper("open", "open before pick")
    cxy = refine_cube_from_wrist(cxy)
    ex["pick_cube_xy"] = vec(cxy)
    move([cxy[0], cxy[1], gz + SAFE_DZ], q_down, "pre-grasp", tol=0.01)
    move([cxy[0], cxy[1], gz], q_down, "grasp descend", tol=0.008)
    gripper("close", "close on cube")
    g = grip_now()
    ex["grip_after_close"] = r4(g)
    move([cxy[0], cxy[1], gz + SAFE_DZ], q_down, "lift", tol=0.015)
    g2 = grip_now()
    ex["grip_after_lift"] = r4(g2)
    if g2 is not None and g2 < 0.03:
        note("gripper nearly closed after lift: grasp likely missed")
        ex["grasp_missed"] = True
        return False
    move([txy[0], txy[1], gz + SAFE_DZ], q_down, "transport above target", tol=0.015)
    move([txy[0], txy[1], table_z + 0.5 * size + 0.012], q_down, "lower to target", tol=0.01)
    gripper("open", "release on target")
    hold(6, 1.0, "settle after release")
    move([txy[0], txy[1], table_z + size + 0.10], q_down, "retreat up", tol=0.02)
    return True


def push_line(cxy, txy, size, ee_z, lead, label):
    """Push from behind cxy along cxy->txy until cube centre reaches txy. lead = distance from
    EE point to the contact face of whatever pushes (fingers or tool)."""
    dx = txy[0] - cxy[0]
    dy = txy[1] - cxy[1]
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 0.008:
        return 0.0
    ux, uy = dx / dist, dy / dist
    half = 0.5 * size
    start = [cxy[0] - ux * (half + lead + 0.025), cxy[1] - uy * (half + lead + 0.025)]
    end = [txy[0] - ux * (half + lead), txy[1] - uy * (half + lead)]
    move([start[0], start[1], ee_z + 0.08], q_now["q"], label + " above stand-off", tol=0.012)
    move([start[0], start[1], ee_z], q_now["q"], label + " descend stand-off", tol=0.008)
    total = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    nstep = max(1, int(math.ceil(total / 0.015)))
    for k in range(1, nstep + 1):
        if flags["done"] or budget_left() < 25:
            break
        f = float(k) / nstep
        move([start[0] + (end[0] - start[0]) * f, start[1] + (end[1] - start[1]) * f, ee_z],
             q_now["q"], label + " push " + str(k) + "/" + str(nstep), tol=0.008, max_steps=60)
    # back off along -u and lift
    move([end[0] - ux * 0.03, end[1] - uy * 0.03, ee_z + 0.06], q_now["q"], label + " back off", tol=0.02)
    return dist


q_now = {"q": q_down}


def do_push(cxy, txy):
    size = cube_size
    gripper("close", "close fingers for push")
    cxy = refine_cube_from_wrist(cxy)
    ex["push_cube_xy"] = vec(cxy)
    ee_z = table_z + max(0.4 * size, 0.012)
    cur = cxy
    tgt = txy
    for attempt in range(3):
        if flags["done"] or budget_left() < 60:
            break
        d = push_line(cur, tgt, size, ee_z, 0.012, "push" + str(attempt))
        ex.setdefault("push_lengths", []).append(r4(d))
        seen = look_cube_and_target(tgt, tgt)
        ex.setdefault("push_checks", []).append({"cube": vec(seen["cube"]) if seen["cube"] else None,
                                                 "target": vec(seen["target"]) if seen["target"] else None})
        if seen["target"] is not None and dist2(seen["target"], txy) < 0.05:
            tgt = seen["target"]
        if seen["cube"] is None:
            note("cube not re-found after push " + str(attempt))
            break
        err_m = dist2(seen["cube"], tgt)
        ex.setdefault("push_errors_m", []).append(r4(err_m))
        if err_m < 0.015:
            break
        cur = seen["cube"]
    return True


def relocate_front_cube(near_xy, radius):
    """Front-camera cube re-localization that ignores anything lifted off the table (the held tool)."""
    best = None
    best_key = None
    tries = []
    prompts = []
    for pr in CUBE_PROMPTS:
        if pr != "block" and pr not in prompts:
            prompts.append(pr)
    for prompt in prompts[:3]:
        entry, found = segment(FRONT, prompt, keep=4, with_geom=True)
        tries.append({"prompt": prompt, "n": entry.get("num")})
        for c in found:
            g = c["geom"]
            if g is None or not is_cube(g, table_z):
                continue
            if float(g["z_lo"]) - table_z > 0.03 or float(g["z_hi"]) - table_z > 0.08:
                continue
            xy = g["top_xy"]
            d = dist2(xy, near_xy) if near_xy is not None else 0.0
            if near_xy is not None and d > radius:
                continue
            key = d - 0.02 * float(c.get("score") or 0.0)
            if best is None or key < best_key:
                best = [float(xy[0]), float(xy[1])]
                best_key = key
        if best is not None:
            break
    if len(ex.setdefault("front_relocate", [])) < 20:
        ex["front_relocate"].append({"near": vec(near_xy) if near_xy else None, "tries": tries,
                                     "chosen": vec(best) if best else None})
    return best


def obstacle_list(parts, tool_box, cxy):
    """Points (xy) of every other above-table object, for the tool's descent and sweep checks."""
    obs = []
    max_h = 0.0
    for p in parts:
        g = p["geom"]
        h = float(g["z_hi"]) - table_z
        if h < 0.012 or box_area(p["box"]) > 0.2 * img_area:
            continue
        if tool_box is not None and (box_inside(p["box"], tool_box, 3.0) or box_inside(tool_box, p["box"], 3.0)):
            continue
        if cxy is not None:
            dmin = min_xy_dist(p["pts"], cxy)
            if dmin is not None and dmin < 0.5 * cube_size + 0.005:
                continue
        if math.sqrt(float(g["med"][0]) ** 2 + float(g["med"][1]) ** 2) < 0.15:
            continue
        for q in thin(p["pts"], 40):
            dz = float(q[2]) - table_z
            if 0.008 < dz < 0.20:
                obs.append([float(q[0]), float(q[1])])
                max_h = max(max_h, dz)
    return thin(obs, 300), max_h


def plan_pull(rel, psi_opts, cxy, txy, obs, relax, fixed_delta):
    """Rigid-tool placement. rel: tool xy points minus the TCP xy at grasp yaw. Rotating the hand by
    delta about world z rotates rel by delta. Finds (delta, lateral offset) so that every tool point
    in the cube's lateral band lies behind the cube's rear face (w.r.t. cube->target), the foremost
    such points (the contact edge) straddle the cube centre, the hand never meets the cube, and the
    TCP/tool path stays reachable and clear of obstacle points. Prefers the hand AHEAD of the cube
    (a pull, as demonstrated), wide contact edges and small wrist rotations."""
    dx = float(txy[0]) - float(cxy[0])
    dy = float(txy[1]) - float(cxy[1])
    dd = math.sqrt(dx * dx + dy * dy)
    if dd < 0.006:
        return None
    ux, uy = dx / dd, dy / dd
    vx, vy = -uy, ux
    half = 0.5 * cube_size
    clr = 0.015 if relax == 0 else 0.010
    band = half + clr
    yaw_lim = 120.0 if relax == 0 else 150.0
    travel = clr + dd
    oa = []
    ol = []
    for o in obs:
        ox = float(o[0]) - float(cxy[0])
        oy = float(o[1]) - float(cxy[1])
        if ox * ox + oy * oy > (travel + 0.55) ** 2:
            continue
        oa.append(ox * ux + oy * uy)
        ol.append(ox * vx + oy * vy)
    if fixed_delta is not None:
        deltas = [float(fixed_delta)]
    else:
        deltas = [-180.0 + 10.0 * k for k in range(36)]
    best = None
    n = len(rel)
    for delta in deltas:
        opt = None
        for oi in range(len(psi_opts)):
            pg = wrap180(psi_opts[oi])
            rf = wrap180(pg + delta)
            if fixed_delta is None and (abs(pg) > yaw_lim or abs(rf) > yaw_lim):
                continue
            if opt is None or abs(rf) < abs(opt[1]):
                opt = [oi, rf]
        if opt is None:
            continue
        cd = math.cos(math.radians(delta))
        sd = math.sin(math.radians(delta))
        pa = []
        pl = []
        for p in rel:
            rx = cd * p[0] - sd * p[1]
            ry = sd * p[0] + cd * p[1]
            pa.append(rx * ux + ry * uy)
            pl.append(rx * vx + ry * vy)
        for j in range(71):
            l_h = -0.35 + 0.01 * j
            a_max = None
            for i in range(n):
                lat = l_h + pl[i]
                if -band < lat < band:
                    if a_max is None or pa[i] > a_max:
                        a_max = pa[i]
            if a_max is None:
                continue
            if abs(l_h) < band + HAND_R:
                if not (a_max > HAND_R + 0.01 or a_max + 2.0 * half + clr < -HAND_R - 0.01):
                    continue
            lmin = None
            lmax = None
            nf = 0
            for i in range(n):
                lat = l_h + pl[i]
                if -half <= lat <= half and pa[i] >= a_max - 0.012:
                    nf += 1
                    if lmin is None or lat < lmin:
                        lmin = lat
                    if lmax is None or lat > lmax:
                        lmax = lat
            if nf < 2 or lmin > 0.35 * half or lmax < -0.35 * half:
                continue
            cov = min(1.0, (lmax - lmin) / (1.5 * half))
            if cov < 0.2:
                continue
            a_h = -half - clr - a_max
            score = cov - 0.5 * abs(0.5 * (lmin + lmax)) / half - 0.001 * abs(delta)
            if a_h > 0:
                score += 0.2
            if best is not None and score <= best["score"]:
                continue
            h0 = [float(cxy[0]) + a_h * ux + l_h * vx, float(cxy[1]) + a_h * uy + l_h * vy]
            h1 = [h0[0] + travel * ux, h0[1] + travel * uy]
            r0 = math.sqrt(h0[0] ** 2 + h0[1] ** 2)
            r1 = math.sqrt(h1[0] ** 2 + h1[1] ** 2)
            tproj = min(max(-(h0[0] * ux + h0[1] * uy), 0.0), travel)
            rmin = math.sqrt((h0[0] + tproj * ux) ** 2 + (h0[1] + tproj * uy) ** 2)
            if min(r0, r1, rmin) < 0.25 or max(r0, r1) > 0.80:
                continue
            bad = False
            for i in range(0, n, 3):
                wx = float(cxy[0]) + (a_h + pa[i]) * ux + (l_h + pl[i]) * vx
                wy = float(cxy[1]) + (a_h + pa[i]) * uy + (l_h + pl[i]) * vy
                if wx * wx + wy * wy < 0.0144 or (wx + travel * ux) ** 2 + (wy + travel * uy) ** 2 < 0.0144:
                    bad = True
                    break
            if not bad:
                for k in range(len(oa)):
                    ao = oa[k] - a_h
                    lob = ol[k] - l_h
                    if abs(lob) < HAND_R + 0.01 and -HAND_R - 0.01 <= ao <= travel + HAND_R + 0.01:
                        bad = True
                        break
                    for i in range(0, n, 2):
                        if abs(lob - pl[i]) < OBS_R and pa[i] - OBS_R <= ao <= pa[i] + travel + OBS_R:
                            bad = True
                            break
                    if bad:
                        break
            if bad:
                continue
            best = {"delta": delta, "opt": opt[0], "final_yaw_rel": opt[1], "l_h": l_h, "a_h": a_h,
                    "a_max": a_max, "h0": h0, "h1": h1, "travel": travel, "cov": cov, "score": score,
                    "u": [ux, uy], "n_front": nf, "clr": clr}
    return best


def plan_with_relax(rel, opts, cxy, txy, obs, fixed_delta, tag):
    for relax in (0, 1):
        plan = plan_pull(rel, opts, cxy, txy, obs, relax, fixed_delta)
        if plan is not None:
            plan["relax"] = relax
            plan["obstacles"] = len(obs)
            return plan
    plan = plan_pull(rel, opts, cxy, txy, [], 1, fixed_delta)
    if plan is not None:
        plan["relax"] = 2
        plan["obstacles"] = 0
        note(tag + ": placement found only after ignoring obstacles")
    return plan


def plan_summary(plan):
    if plan is None:
        return None
    out = {}
    for key in plan:
        value = plan[key]
        if key in ("h0", "h1", "u"):
            out[key] = vec(value)
        else:
            out[key] = r4(value)
    return out


def tool_rel(pts, gxy):
    out = []
    for p in thin(pts, 150):
        out.append([float(p[0]) - float(gxy[0]), float(p[1]) - float(gxy[1])])
    return out


def grasp_axis_options(tl):
    reg = tl["region"]
    phi0 = closing_axis_deg(q_down)
    gaxis = float(reg["axis"])
    if float(reg["len"]) < 1.3 * float(reg["w"]):
        gaxis = float(tl["geom"]["axis_deg"])
    d1 = wrap90((gaxis + 90.0) - phi0)
    other = d1 - 180.0 if d1 > 0 else d1 + 180.0
    return gaxis, [d1, other]


def grasp_height(h_region):
    hr = max(float(h_region), 0.008)
    lo_z = 0.4 * hr
    hi_z = max(hr - 0.004, lo_z)
    main_seg = video_cls.get("main_segment")
    try:
        if main_seg is not None and main_seg.get("z0") is not None:
            demo = float(main_seg["z0"]) - table_z
            return table_z + min(max(demo, lo_z), hi_z), "video grasp height clamped to region"
    except Exception as exc:
        err("grasp_height", exc)
    return table_z + max(0.5 * hr, min(0.012, hi_z)), "half region height"


def sweep_dz():
    main_seg = video_cls.get("main_segment")
    try:
        if main_seg is not None and main_seg.get("sweep_z_med") is not None:
            return min(max(float(main_seg["sweep_z_med"]) - float(main_seg["z0"]), 0.003), 0.012)
    except Exception as exc:
        err("sweep_dz", exc)
    return 0.005


def reperceive_tool(cxy):
    items = []
    prompts = []
    for pr in [tool["prompt"], tool["region"]["prompt"]] + TOOL_PROMPTS[:3] + ["stick", "block", "rod"]:
        if pr not in prompts:
            prompts.append(pr)
    for pr in prompts:
        entry, found = segment(FRONT, pr, keep=4, with_geom=True)
        if len(report["probes"]) < 80:
            report["probes"].append(entry)
        for item in found:
            items.append(item)
    found, cost, parts, maximal, log = select_tool(items, cxy, w_video)
    ex["tool"].setdefault("reperceive", []).append({"prompts": prompts, "n_tools": len(maximal),
                                                     "chosen": vec(found["region"]["xy"]) if found else None,
                                                     "cost": r4(cost)})
    if found is None:
        return None, parts
    return make_tool(found, cost), parts


def tool_pass(plan, rel_a, psi_a, quat_g, z_t, z_s, cur, txy, obs, label):
    rec = {"plan": plan_summary(plan)}
    q_cmd = qnorm(qmul(qyaw(plan["delta"]), quat_g))
    pos, quat = ee_now()
    if pos is not None and quat is not None:
        move([pos[0], pos[1], z_t], quat, label + " raise tool", tol=0.015)
    move([plan["h0"][0], plan["h0"][1], z_t], q_cmd, label + " carry tool above stand-off", tol=0.015,
         max_steps=200)
    if flags["done"]:
        return rec
    pos, quat = ee_now()
    if quat is not None:
        d_act = wrap180(closing_axis_deg(quat) - closing_axis_deg(quat_g))
        rec["delta_actual"] = r4(d_act)
        if abs(wrap180(d_act - plan["delta"])) > 4.0:
            fix = plan_with_relax(rel_a, [psi_a], cur, txy, obs, d_act, label + " yaw fix")
            if fix is not None:
                plan = fix
                q_cmd = quat
                rec["plan_yaw_fixed"] = plan_summary(fix)
                move([plan["h0"][0], plan["h0"][1], z_t], q_cmd, label + " re-centre after yaw fix", tol=0.015)
    h0 = plan["h0"]
    h1 = plan["h1"]
    move([h0[0], h0[1], z_s + 0.03], q_cmd, label + " lower tool", tol=0.008)
    move([h0[0], h0[1], z_s], q_cmd, label + " tool at sweep height", tol=0.006)
    nstep = max(1, int(math.ceil(float(plan["travel"]) / 0.02)))
    for k in range(1, nstep + 1):
        if flags["done"] or budget_left() < 20:
            break
        f = float(k) / nstep
        move([h0[0] + (h1[0] - h0[0]) * f, h0[1] + (h1[1] - h0[1]) * f, z_s], q_cmd,
             label + " sweep " + str(k) + "/" + str(nstep), tol=0.006, max_steps=40)
    move([h1[0], h1[1], z_s + 0.07], q_cmd, label + " lift tool", tol=0.015)
    rec["grip_end"] = r4(grip_now())
    rec["tick_end"] = ticks_now()
    return rec


def do_tool(cxy, txy):
    tex = ex.setdefault("tool", {})
    tl = tool
    # empty-hand wrist look at cube and target (v00's look; refined the cube by 2.7 mm on episode 0)
    seen = look_cube_and_target(cxy, txy)
    if seen.get("cube") is not None and dist2(seen["cube"], cxy) < 0.05:
        cxy = seen["cube"]
    if seen.get("target") is not None and dist2(seen["target"], txy) < 0.05:
        txy = seen["target"]
    tex["cube_xy"] = vec(cxy)
    tex["target_xy"] = vec(txy)
    if flags["done"]:
        return False
    phi0 = closing_axis_deg(q_down)
    obs, max_h = obstacle_list(tool_parts, tl["box"], cxy)
    tex["n_obstacle_pts"] = len(obs)
    tex["obstacle_max_h"] = r4(max_h)
    grasped = False
    pos_g = None
    quat_g = None
    for attempt in range(2):
        if flags["done"] or budget_left() < 150:
            break
        reg = tl["region"]
        gxy = [float(reg["xy"][0]), float(reg["xy"][1])]
        gaxis, opts = grasp_axis_options(tl)
        rel = tool_rel(tl["pts"], gxy)
        pre = plan_with_relax(rel, opts, cxy, txy, obs, None, "pre-plan")
        tex.setdefault("pre_plans", []).append(plan_summary(pre))
        if pre is None:
            note("no feasible tool placement for this tool; not touching the cube")
            return False
        yaw = opts[int(pre["opt"])]
        q_grasp = qnorm(qmul(qyaw(yaw), q_down))
        gz, gz_src = grasp_height(reg["h"])
        gz = max(gz - 0.003 * attempt, table_z + 0.006)
        rec = {"attempt": attempt, "xy": vec(gxy), "axis": r4(gaxis), "yaw": r4(yaw), "z": r4(gz),
               "z_src": gz_src, "w": r4(reg["w"]), "expect_g": r4(float(reg["w"]) / GRIP_SPAN)}
        tex.setdefault("grasps", []).append(rec)
        gripper("open", "open before tool grasp")
        move([gxy[0], gxy[1], gz + SAFE_DZ], q_grasp, "tool pre-grasp", tol=0.01)
        move([gxy[0], gxy[1], gz], q_grasp, "tool grasp descend", tol=0.006)
        gripper("close", "close on tool handle")
        gc = grip_now()
        pos_g, quat_g = ee_now()
        rec["g_close"] = r4(gc)
        rec["tcp_close"] = vec(pos_g) if pos_g is not None else None
        held = gc is not None and gc >= 0.02 and pos_g is not None and quat_g is not None
        if held:
            move([pos_g[0], pos_g[1], pos_g[2] + SAFE_DZ], quat_g, "tool lift", tol=0.015)
            g2 = grip_now()
            rec["g_after_lift"] = r4(g2)
            held = g2 is not None and g2 >= 0.02
        if held:
            grasped = True
            break
        if flags["done"]:
            break
        note("tool grasp lost on attempt " + str(attempt) + "; re-perceiving the tool")
        gripper("open", "reopen after tool miss")
        move([gxy[0], gxy[1], table_z + 0.25], q_down, "rise for tool re-look", tol=0.02)
        if attempt == 0:
            new_tool, new_parts = reperceive_tool(cxy)
            if new_tool is None:
                break
            tl = new_tool
            obs, max_h = obstacle_list(new_parts, tl["box"], cxy)
    tex["grasped"] = grasped
    if not grasped or flags["done"]:
        note("tool not held; stopping without touching the cube")
        return False
    rel_a = tool_rel(tl["pts"], pos_g)
    psi_a = wrap180(closing_axis_deg(quat_g) - phi0)
    z_t = table_z + min(max(max_h + 0.05 + (pos_g[2] - table_z), 0.15), 0.32)
    z_s = pos_g[2] + sweep_dz()
    tex["z_transport"] = r4(z_t)
    tex["z_sweep"] = r4(z_s)
    cur = cxy
    for pidx in range(3):
        if flags["done"] or budget_left() < 90:
            break
        plan = plan_with_relax(rel_a, [psi_a], cur, txy, obs, None, "pass" + str(pidx))
        if plan is None:
            note("no feasible placement for pass " + str(pidx))
            tex.setdefault("passes", []).append({"plan": None, "cube": vec(cur)})
            break
        prec = tool_pass(plan, rel_a, psi_a, quat_g, z_t, z_s, cur, txy, obs, "pass" + str(pidx))
        prec["cube_before"] = vec(cur)
        tex.setdefault("passes", []).append(prec)
        if flags["done"]:
            break
        g = grip_now()
        if g is not None and g < 0.02:
            note("tool dropped during pass " + str(pidx))
            break
        now = relocate_front_cube(txy, 0.15)
        if now is None:
            now = relocate_front_cube(cur, 0.25)
        prec["cube_after"] = vec(now) if now else None
        if now is None:
            break
        e = dist2(now, txy)
        prec["err_after_m"] = r4(e)
        if e < 0.015:
            break
        cur = now
    return True


try:
    if cube_xy is None:
        note("no cube found; cannot attempt task")
        ex["skipped"] = "no cube"
    elif target_xy is None:
        note("no target found; cannot attempt task")
        ex["skipped"] = "no target"
    else:
        ex["start_tick"] = ticks_now()
        ex["cube_to_target_m"] = r4(dist2(cube_xy, target_xy))
        if manner == "push":
            do_push(cube_xy, target_xy)
        elif manner == "peg":
            if tool is None:
                ex["skipped"] = "tool manner without a selected tool (no gripper-push fallback)"
            else:
                do_tool(cube_xy, target_xy)
        else:
            ok = do_pick(cube_xy, target_xy)
            if not ok and not flags["done"] and budget_left() > 120:
                note("retrying pick after re-localization")
                gripper("open", "reopen after miss")
                move([cube_xy[0], cube_xy[1], table_z + 0.25], q_down, "retreat for re-look", tol=0.02)
                again = relocalize(FRONT, ["cube", "small cube", "block"], cube_xy, "cube", 0.15)
                if again is not None:
                    do_pick(again[0], target_xy)
        ex["end_tick"] = ticks_now()
except Exception as exc:
    err("P5 execution", exc)

# =============================================================================== P6 final
fin = report["final"]
try:
    if not flags["done"]:
        if manner == "peg":
            hold(5, 0.0, "final settle (tool clamped)")
        else:
            hold(5, grip_now() if grip_now() is not None else 1.0, "final settle")
    if not flags["done"] and target_xy is not None and manner == "peg":
        fxy = relocate_front_cube(target_xy, 0.25)
        seen = {"cube": fxy}
    elif not flags["done"] and target_xy is not None:
        seen = look_cube_and_target(target_xy, target_xy) if cube_xy is not None else {"cube": None}
        fin["cube_final_xy"] = vec(seen["cube"]) if seen.get("cube") else None
        if seen.get("cube") is not None:
            fin["cube_to_target_m"] = r4(dist2(seen["cube"], target_xy))
    fin["terminal_seen"] = flags["done"]
    fin["ticks_used"] = ticks_now()
    fin["n_errors"] = len(report["errors"])
except Exception as exc:
    err("P6 final", exc)
note("finished: manner=" + str(report["decision"].get("manner")) + " ticks=" + str(ticks_now()))
result = report
