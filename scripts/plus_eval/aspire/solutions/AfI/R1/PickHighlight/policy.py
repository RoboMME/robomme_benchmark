# v01_live_highlight_depth -- robomme_plus/task_6 PickHighlight (AfI, R=1).  UNTESTED repair of v00.
#
# Trace-guided repair of v00 (episode 0: program_ok, task_success=false, terminated on lifting a
# NON-highlighted cube at tick 269).  Evidence (debug/EVIDENCE.md):
#  * the episode has NO conditioning video; the highlight is LIVE: white discs (~0.10 m wide)
#    appear on the table around the highlighted cubes at execution tick ~10 and vanish at tick ~55,
#    independent of the button (v00 had not touched it yet).  v00 watched only after its press
#    (ticks 180-220) and picked cubes whose appearance changed for another reason.
#  * SAM3 segment_text returns at most 5 candidates per prompt; 9 cubes were present, so text
#    prompts alone cannot enumerate the cubes.  Front depth + public intrinsics/camera_pose
#    (OpenCV pinhole, verified offline against mask_to_point_cloud) finds all 9 + the button.
#  * a non-highlighted cube slid ~3 cm during v00's 126-tick button-press timeout, and the wrist
#    refinement (3 cm acceptance) then locked onto it.
# Changes: (1) hold the arm still from tick 0 and watch the front camera every tick for a changed
# region; on onset freeze (no stepping) and score each depth-detected cube by changed table-level
# pixels in its 3D Voronoi cell;  (2) cubes from elevated depth clusters, re-detected before each
# pick;  (3) press capped at 40 ticks; wrist acceptance 2 cm;  (4) final button press ("finally
# press the button to stop").  Every phase wrapped; never raises; public APIs only.

import math

report = {
    "program": "v01_live_highlight_depth",
    "errors": [],
    "log": [],
    "phases": {},
    "motions": [],
}
result = report

ctl = {
    "hz": 20.0,
    "max_steps": None,
    "t0": 0.0,
    "terminal": False,
    "terminal_label": None,
}


def note(msg):
    text = str(msg)
    print("[v01] " + text)
    if len(report["log"]) < 500:
        report["log"].append(text[:400])


def err(where, exc):
    text = str(where) + ": " + str(exc)
    print("[v01][error] " + text)
    if len(report["errors"]) < 200:
        report["errors"].append(text[:600])


def num(value, digits=4):
    try:
        return round(float(value), digits)
    except Exception:
        return None


def clamp(value, low, high):
    return max(low, min(high, value))


def quantile(values, fraction):
    if len(values) == 0:
        return None
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * fraction)]


def as_list(array):
    try:
        return array.tolist()
    except Exception:
        return array


def pose_info(pose):
    out = {}
    try:
        out["position"] = [num(v) for v in pose.position]
    except Exception as exc:
        out["position_error"] = str(exc)[:200]
    try:
        out["quaternion_wxyz"] = [num(v) for v in pose.quaternion_wxyz]
    except Exception as exc:
        out["quaternion_error"] = str(exc)[:200]
    try:
        out["frame"] = str(pose.frame)
    except Exception:
        pass
    return out


def safe_map(mapping, limit=800):
    out = {}
    try:
        for key in mapping:
            try:
                out[str(key)] = str(mapping[key])[:limit]
            except Exception as exc:
                out[str(key)] = "unreadable: " + str(exc)[:100]
    except Exception as exc:
        out["error"] = str(exc)[:200]
    return out


def elapsed_ticks():
    try:
        state = get_robot_state()
        return (float(state.timestamp_s) - ctl["t0"]) * ctl["hz"]
    except Exception:
        return -1.0


def ee_pos():
    try:
        state = get_robot_state()
        return [float(v) for v in state.end_effector_poses["primary"].position]
    except Exception:
        return None


def gripper_now():
    try:
        return float(get_robot_state().gripper_positions["primary"])
    except Exception:
        return None


# ---------------------------------------------------------------- hex thumbnails
HEX_DIGITS = "0123456789abcdef"
HEX_TABLE = []
for hex_value in range(256):
    HEX_TABLE.append(HEX_DIGITS[hex_value // 16] + HEX_DIGITS[hex_value % 16])


def thumbnail(img, x0, y0, x1, y1, stride):
    parts = []
    rows = 0
    cols = 0
    for y in range(y0, y1, stride):
        row = img[y]
        rows += 1
        cols = 0
        for x in range(x0, x1, stride):
            pixel = row[x]
            parts.append(HEX_TABLE[int(pixel[0])] + HEX_TABLE[int(pixel[1])] + HEX_TABLE[int(pixel[2])])
            cols += 1
    return {"x0": x0, "y0": y0, "stride": stride, "w": cols, "h": rows, "rgb_hex": "".join(parts)}


# ---------------------------------------------------------------- execution records
def record_exec(res, label):
    rec = {"label": label, "ok": False, "status": ""}
    try:
        rec["ok"] = bool(res.ok)
    except Exception:
        pass
    try:
        rec["status"] = str(res.status)
    except Exception:
        pass
    try:
        rec["steps"] = int(res.steps_executed)
    except Exception:
        pass
    terminal = False
    try:
        if bool(res.terminated):
            terminal = True
            rec["terminated"] = True
    except Exception:
        pass
    try:
        if bool(res.truncated):
            terminal = True
            rec["truncated"] = True
    except Exception:
        pass
    try:
        if res.error is not None:
            rec["error"] = str(res.error)[:400]
    except Exception:
        pass
    low = rec["status"].lower()
    if ("terminat" in low) or ("truncat" in low):
        terminal = True
    rec["tick"] = num(elapsed_ticks(), 1)
    if terminal:
        ctl["terminal"] = True
        ctl["terminal_label"] = label
        rec["episode_terminal"] = True
    if len(report["motions"]) < 400:
        report["motions"].append(rec)
    note("exec " + label + " ok=" + str(rec["ok"]) + " status=" + rec["status"]
         + " steps=" + str(rec.get("steps")) + " tick=" + str(rec["tick"]))
    return rec


def move(pos, quat, label, tol=0.01, steps=200):
    if ctl["terminal"]:
        return False, None
    try:
        target = Pose([float(pos[0]), float(pos[1]), float(pos[2])],
                      [float(v) for v in quat], base_frame)
        res = move_to_pose(target, tolerance=tol, max_steps=steps)
    except Exception as exc:
        err("move_to_pose " + label, exc)
        return False, None
    rec = record_exec(res, label)
    rec["target"] = [num(v) for v in pos]
    here = ee_pos()
    dist = None
    if here is not None:
        dist = math.sqrt((here[0] - pos[0]) ** 2 + (here[1] - pos[1]) ** 2 + (here[2] - pos[2]) ** 2)
        rec["final_pos"] = [num(v) for v in here]
        rec["final_err"] = num(dist)
    return rec["ok"], dist


def gripper_cmd(close, label):
    if ctl["terminal"]:
        return None
    try:
        if close:
            res = close_gripper()
        else:
            res = open_gripper()
    except Exception as exc:
        err("gripper " + label, exc)
        return None
    rec = record_exec(res, label)
    width = gripper_now()
    rec["gripper_after"] = num(width)
    return width


def hold(ticks, label, grip_value):
    count = 0
    for tick_index in range(ticks):
        if ctl["terminal"]:
            break
        try:
            state = get_robot_state()
            res = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=state.joint_positions["primary"],
                gripper_position=float(grip_value), embodiment=state.embodiment)}))
        except Exception as exc:
            err("step " + label, exc)
            break
        count += 1
        terminal = False
        try:
            if bool(res.terminated):
                terminal = True
        except Exception:
            pass
        try:
            if bool(res.truncated):
                terminal = True
        except Exception:
            pass
        if terminal:
            ctl["terminal"] = True
            ctl["terminal_label"] = label + " (step)"
            note("episode terminal during " + label)
            break
    if len(report["motions"]) < 400:
        report["motions"].append({"label": label, "hold_ticks": count, "tick": num(elapsed_ticks(), 1)})
    return count


# ---------------------------------------------------------------- perception helpers
def probe(camera, prompt):
    rec = {"camera": camera, "prompt": prompt, "ok": False, "n": 0, "scores": [], "boxes": []}
    try:
        seg = segment_text(camera, prompt)
        rec["ok"] = bool(seg.ok)
        if seg.ok:
            cands = seg.segmentations
            rec["n"] = len(cands)
            for cand in cands[:6]:
                rec["scores"].append(num(cand.score, 3))
                try:
                    rec["boxes"].append([round(float(v), 1) for v in cand.box_xyxy])
                except Exception:
                    rec["boxes"].append(None)
        else:
            try:
                rec["error"] = str(seg.error)[:300]
            except Exception:
                pass
        return rec, seg
    except Exception as exc:
        rec["error"] = str(exc)[:300]
        return rec, None


def cloud_points(cand, camera):
    cloud = None
    try:
        cloud = mask_to_point_cloud(cand, camera, target_frame=base_frame)
    except Exception as first_exc:
        try:
            cloud = mask_to_point_cloud(cand.mask, camera, target_frame=base_frame)
        except Exception as exc:
            err("mask_to_point_cloud " + camera, str(first_exc)[:200] + " | " + str(exc)[:200])
            return []
    try:
        raw = as_list(cloud.points)
    except Exception as exc:
        err("point_cloud.points", exc)
        return []
    total = len(raw)
    stride = 1
    if total > 800:
        stride = total // 800 + 1
    out = []
    for k in range(0, total, stride):
        point = raw[k]
        try:
            x = float(point[0])
            y = float(point[1])
            z = float(point[2])
        except Exception:
            continue
        if x != x or y != y or z != z:
            continue
        if abs(x) > 3.0 or abs(y) > 3.0 or abs(z) > 3.0:
            continue
        out.append((x, y, z))
    return out


def cloud_stats(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return {
        "n": len(points),
        "x05": quantile(xs, 0.05), "x50": quantile(xs, 0.5), "x95": quantile(xs, 0.95),
        "y05": quantile(ys, 0.05), "y50": quantile(ys, 0.5), "y95": quantile(ys, 0.95),
        "z05": quantile(zs, 0.05), "z50": quantile(zs, 0.5), "z95": quantile(zs, 0.95),
    }


def top_face(points, z_top):
    chosen = [p for p in points if p[2] >= z_top - 0.008]
    if len(chosen) < 4:
        chosen = points
    cx = quantile([p[0] for p in chosen], 0.5)
    cy = quantile([p[1] for p in chosen], 0.5)
    return cx, cy, chosen


def yaw_from_points(points):
    """Angle in [0, pi/2) of the minimum-area XY rectangle of the points (cube face direction)."""
    if len(points) < 6:
        return None
    mx = sum(p[0] for p in points) / len(points)
    my = sum(p[1] for p in points) / len(points)
    best = None
    for k in range(18):
        angle = k * math.pi / 36.0
        ca = math.cos(angle)
        sa = math.sin(angle)
        us = [(p[0] - mx) * ca + (p[1] - my) * sa for p in points]
        vs = [-(p[0] - mx) * sa + (p[1] - my) * ca for p in points]
        area = (quantile(us, 0.95) - quantile(us, 0.05)) * (quantile(vs, 0.95) - quantile(vs, 0.05))
        if best is None or area < best[0]:
            best = (area, angle)
    return best[1]


# ---------------------------------------------------------------- quaternion helpers
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def qz(angle):
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]


def z_axis_z(q):
    w, x, y, z = q
    return 1.0 - 2.0 * (x * x + y * y)


def finger_angle(q):
    w, x, y, z = q
    fx = 2.0 * (x * y - w * z)
    fy = 1.0 - 2.0 * (x * x + z * z)
    return math.atan2(fy, fx)


def grasp_quats(theta, target_xy, neighbors):
    """Top-down orientations aligned to the cube faces, least neighbour conflict first."""
    base = home_quat
    if z_axis_z(base) > -0.85:
        base = [0.0, 1.0, 0.0, 0.0]
    phi = finger_angle(base)
    if theta is None:
        d0 = 0.0
    else:
        d0 = theta - phi
        while d0 > math.pi / 4.0:
            d0 -= math.pi / 2.0
        while d0 <= -math.pi / 4.0:
            d0 += math.pi / 2.0
    if d0 <= 0.0:
        d1 = d0 + math.pi / 2.0
    else:
        d1 = d0 - math.pi / 2.0
    scored = []
    for delta in (d0, d1):
        angle = phi + delta
        fx = math.cos(angle)
        fy = math.sin(angle)
        conflicts = 0
        for other in neighbors:
            dx = other[0] - target_xy[0]
            dy = other[1] - target_xy[1]
            along = abs(dx * fx + dy * fy)
            perp = abs(-dx * fy + dy * fx)
            if along < 0.085 and perp < 0.045:
                conflicts += 1
        scored.append((conflicts, abs(delta), delta))
    scored = sorted(scored)
    out = []
    for item in scored:
        out.append({"delta_deg": num(math.degrees(item[2]), 1), "conflicts": item[0],
                    "quat": qmul(qz(item[2]), base)})
    return out


# ================================================================ Phase 1: snapshot
lang = ""
base_frame = "robot_base"
home_quat = [0.0, 1.0, 0.0, 0.0]
home_pos = None
live_joints = None
front_name = "front"
wrist_name = "wrist"
live_front = None
H = 256
W = 256
try:
    try:
        ctx = get_task_context()
        lang = str(ctx.language)
        report["language"] = lang
        try:
            report["task_name"] = str(ctx.task_name)
            report["task_id"] = str(ctx.task_id)
            report["family"] = str(ctx.family)
        except Exception as exc:
            err("context fields", exc)
        try:
            report["context_metadata"] = safe_map(ctx.metadata)
        except Exception as exc:
            err("context metadata", exc)
    except Exception as exc:
        err("get_task_context", exc)
    note("language: " + lang)

    try:
        meta = robomme_plus.get_task_metadata()
        report["task_metadata"] = safe_map(meta)
        try:
            ctl["max_steps"] = int(meta["max_steps"])
        except Exception as exc:
            err("max_steps", exc)
        for hz_key in ("control_hz", "control_frequency", "control_freq", "frequency_hz"):
            try:
                ctl["hz"] = float(meta[hz_key])
                break
            except Exception:
                pass
    except Exception as exc:
        err("get_task_metadata", exc)
    try:
        report["controller_metadata"] = safe_map(robomme_plus.get_controller_metadata())
    except Exception as exc:
        err("get_controller_metadata", exc)

    try:
        st0 = get_robot_state()
        base_frame = str(st0.base_frame)
        ctl["t0"] = float(st0.timestamp_s)
        live_joints = [float(v) for v in st0.joint_positions["primary"]]
        ee0 = st0.end_effector_poses["primary"]
        home_quat = [float(v) for v in ee0.quaternion_wxyz]
        home_pos = [float(v) for v in ee0.position]
        report["robot_state"] = {
            "base_frame": base_frame,
            "embodiment": str(st0.embodiment),
            "timestamp_s": num(st0.timestamp_s),
            "joints": [num(v) for v in live_joints],
            "gripper": num(st0.gripper_positions["primary"]),
            "ee_pose": pose_info(ee0),
            "ee_z_axis_z": num(z_axis_z(home_quat)),
            "ee_finger_angle_deg": num(math.degrees(finger_angle(home_quat)), 1),
        }
    except Exception as exc:
        err("get_robot_state", exc)

    try:
        obs0 = get_observation()
        cam_names = [str(name) for name in obs0.cameras]
        report["cameras"] = {}
        for name in cam_names:
            cam = obs0.cameras[name]
            info = {}
            try:
                info["rgb_shape"] = [int(v) for v in cam.rgb.shape]
            except Exception as exc:
                info["rgb_error"] = str(exc)[:200]
            try:
                info["intrinsics"] = [[num(v) for v in row] for row in as_list(cam.intrinsics)]
            except Exception as exc:
                info["intrinsics_error"] = str(exc)[:200]
            try:
                info["camera_pose"] = pose_info(cam.camera_pose)
            except Exception as exc:
                info["pose_error"] = str(exc)[:200]
            try:
                depth = as_list(cam.depth_m)
                vals = []
                for y in range(0, len(depth), 16):
                    for x in range(0, len(depth[0]), 16):
                        vals.append(float(depth[y][x]))
                info["depth_sampled"] = {"min": num(min(vals)), "median": num(quantile(vals, 0.5)),
                                         "max": num(max(vals))}
            except Exception as exc:
                info["depth_error"] = str(exc)[:200]
            report["cameras"][name] = info
        if "front" in cam_names:
            front_name = "front"
        elif len(cam_names) > 0:
            front_name = cam_names[0]
        if "wrist" in cam_names:
            wrist_name = "wrist"
        elif len(cam_names) > 1:
            wrist_name = cam_names[1]
        live_front = as_list(obs0.cameras[front_name].rgb)
        H = len(live_front)
        W = len(live_front[0])
        report["thumb_live_front"] = thumbnail(live_front, 0, 0, W, H, 4)
        try:
            live_wrist = as_list(obs0.cameras[wrist_name].rgb)
            report["thumb_live_wrist"] = thumbnail(live_wrist, 0, 0, len(live_wrist[0]), len(live_wrist), 4)
        except Exception as exc:
            err("wrist thumbnail", exc)
    except Exception as exc:
        err("get_observation", exc)
    report["phases"]["snapshot"] = "done"
except Exception as exc:
    err("phase snapshot", exc)
    report["phases"]["snapshot"] = "error"
note("snapshot done; max_steps=" + str(ctl["max_steps"]) + " hz=" + str(ctl["hz"]))

lang_low = lang.lower()

# ================================================================ Phase 2: SAM probes (button + calibration)
front_segs = {}
report["probes"] = []
try:
    for prompt in ("button", "cube"):
        rec, seg = probe(front_name, prompt)
        report["probes"].append(rec)
        front_segs[prompt] = seg
    report["phases"]["probes"] = "done"
except Exception as exc:
    err("phase probes", exc)
    report["phases"]["probes"] = "error"


# ================================================================ front-camera depth geometry
def quat_matrix(q):
    w, x, y, z = q
    return [[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)]]


def front_view(obs):
    """Public front-camera arrays + calibration of one Observation (camera_pose is camera->base)."""
    cam = obs.cameras[front_name]
    k = as_list(cam.intrinsics)
    pose = cam.camera_pose
    return {"rgb": as_list(cam.rgb), "depth": as_list(cam.depth_m),
            "fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]), "cy": float(k[1][2]),
            "R": quat_matrix([float(v) for v in pose.quaternion_wxyz]),
            "t": [float(v) for v in pose.position], "frame": str(pose.frame)}


def project(view, u, v):
    """OpenCV pinhole: depth_m is z-depth along the optical axis, x right, y down."""
    d = float(view["depth"][v][u])
    if d != d or d <= 0.0 or d > 5.0:
        return None
    a = (u - view["cx"]) / view["fx"] * d
    b = (v - view["cy"]) / view["fy"] * d
    r = view["R"]
    t = view["t"]
    return (r[0][0] * a + r[0][1] * b + r[0][2] * d + t[0],
            r[1][0] * a + r[1][1] * b + r[1][2] * d + t[1],
            r[2][0] * a + r[2][1] * b + r[2][2] * d + t[2])


def table_height(view):
    """Mode of projected z (5 mm bins) over a 4-px grid: the table plane."""
    bins = {}
    for v in range(0, H, 4):
        for u in range(0, W, 4):
            p = project(view, u, v)
            if p is None or p[0] < 0.10 or abs(p[1]) > 0.8 or abs(p[2]) > 0.5:
                continue
            key = int(math.floor(p[2] / 0.005))
            if key not in bins:
                bins[key] = []
            bins[key].append(p[2])
    best = None
    for key in bins:
        if best is None or len(bins[key]) > len(bins[best]):
            best = key
    if best is None:
        return None
    return quantile(bins[best], 0.5)


def elevated_clusters(view, tz):
    """Objects standing 1.2-7 cm above the table, single-linkage (1.2 cm) in base-frame XY."""
    pts = []
    for v in range(0, H, 2):
        for u in range(0, W, 2):
            p = project(view, u, v)
            if p is None or p[0] < 0.12 or abs(p[1]) > 0.8:
                continue
            if p[2] < tz + 0.012 or p[2] > tz + 0.07:
                continue
            pts.append((p[0], p[1], p[2], u, v))
    link = 0.012
    grid = {}
    for k in range(len(pts)):
        key = (int(math.floor(pts[k][0] / link)), int(math.floor(pts[k][1] / link)))
        if key not in grid:
            grid[key] = []
        grid[key].append(k)
    label = [-1] * len(pts)
    groups = []
    for i in range(len(pts)):
        if label[i] >= 0:
            continue
        label[i] = len(groups)
        members = [i]
        stack = [i]
        while len(stack) > 0:
            j = stack.pop()
            kx = int(math.floor(pts[j][0] / link))
            ky = int(math.floor(pts[j][1] / link))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for m in grid.get((kx + dx, ky + dy), []):
                        if label[m] < 0 and math.hypot(pts[m][0] - pts[j][0], pts[m][1] - pts[j][1]) < link:
                            label[m] = label[i]
                            members.append(m)
                            stack.append(m)
        groups.append(members)
    clusters = []
    for members in groups:
        if len(members) < 4:
            continue
        cp = [(pts[m][0], pts[m][1], pts[m][2]) for m in members]
        st = cloud_stats(cp)
        z_top = st["z95"]
        cx, cy, top_pts = top_face(cp, z_top)
        clusters.append({"cx": cx, "cy": cy, "z_top": z_top, "n": len(members),
                         "ext_x": st["x95"] - st["x05"], "ext_y": st["y95"] - st["y05"],
                         "u": sum(pts[m][3] for m in members) / len(members),
                         "v": sum(pts[m][4] for m in members) / len(members),
                         "theta": yaw_from_points(top_pts)})
    return clusters


def cube_like(c, tz):
    return (c["z_top"] - tz >= 0.025 and c["ext_x"] <= 0.075 and c["ext_y"] <= 0.075
            and c["ext_x"] >= 0.008 and c["ext_y"] >= 0.008)


def cluster_info(c, tz):
    return {"xy": [num(c["cx"]), num(c["cy"])], "height": num(c["z_top"] - tz), "n": c["n"],
            "ext": [num(c["ext_x"]), num(c["ext_y"])], "uv": [num(c["u"], 1), num(c["v"], 1)],
            "yaw_deg": None if c["theta"] is None else num(math.degrees(c["theta"]), 1)}


def count_changed(rgb_now, rgb_ref, stride, threshold):
    count = 0
    for v in range(0, H, stride):
        row_now = rgb_now[v]
        row_ref = rgb_ref[v]
        for u in range(0, W, stride):
            a = row_now[u]
            b = row_ref[u]
            if abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) + abs(int(a[2]) - int(b[2])) > threshold:
                count += 1
    return count


def highlight_scores(view_now, rgb_ref, cubes_list, tz, threshold):
    """Changed pixels (stride 1) projected to base XY and assigned to the nearest cube <= 6.5 cm."""
    table_score = [0] * len(cubes_list)
    raised_score = [0] * len(cubes_list)
    unassigned = 0
    for v in range(0, H):
        row_now = view_now["rgb"][v]
        row_ref = rgb_ref[v]
        for u in range(0, W):
            a = row_now[u]
            b = row_ref[u]
            if abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) + abs(int(a[2]) - int(b[2])) <= threshold:
                continue
            p = project(view_now, u, v)
            if p is None or p[0] < 0.12 or p[2] > tz + 0.08:
                continue
            best = None
            for i in range(len(cubes_list)):
                c = cubes_list[i]
                dist = math.hypot(p[0] - c["cx"], p[1] - c["cy"])
                if dist < 0.065 and (best is None or dist < best[0]):
                    best = (dist, i)
            if best is None:
                unassigned += 1
                continue
            if p[2] < tz + 0.012:
                table_score[best[1]] += 1
            else:
                raised_score[best[1]] += 1
    return table_score, raised_score, unassigned


def choose_targets(scores):
    top = max(scores) if len(scores) > 0 else 0
    if top < 20:
        return [], top
    floor = max(10, 0.2 * top)
    return [i for i in range(len(scores)) if scores[i] >= floor], top


# ================================================================ Phase 3: scene geometry at tick 0
ref_view = None
table_z = None
cubes = []
button = None
try:
    obs_ref = get_observation()
    ref_view = front_view(obs_ref)
    report["front_pose_frame"] = ref_view["frame"]
    table_z = table_height(ref_view)
    report["table_z"] = num(table_z)
    if table_z is not None:
        all_clusters = elevated_clusters(ref_view, table_z)
        report["clusters_tick0"] = [cluster_info(c, table_z) for c in all_clusters]
        cubes = [c for c in all_clusters if cube_like(c, table_z)]
    # button: SAM "button" candidate whose projected cloud is compact
    seg = front_segs.get("button")
    if seg is not None and seg.ok:
        for cand in seg.segmentations[:5]:
            pts = cloud_points(cand, front_name)
            if len(pts) < 8:
                continue
            st = cloud_stats(pts)
            if st["x95"] - st["x05"] > 0.12 or st["y95"] - st["y05"] > 0.12:
                continue
            bx, by, top_pts = top_face(pts, st["z95"])
            button = {"x": bx, "y": by, "z_top": st["z95"], "score": num(cand.score, 3)}
            break
    if button is not None:
        report["button"] = {"xy": [num(button["x"]), num(button["y"])], "z_top": num(button["z_top"]),
                            "score": button["score"]}
        kept = []
        for c in cubes:
            if math.hypot(c["cx"] - button["x"], c["cy"] - button["y"]) < 0.04:
                report["notes"] = "cluster at button removed from cubes"
                continue
            kept.append(c)
        cubes = kept
    for k, c in enumerate(cubes):
        c["id"] = k
    report["cubes_tick0"] = [cluster_info(c, table_z) for c in cubes]
    # calibration check: SAM "cube" clouds (mask_to_point_cloud) vs my projection clusters
    residuals = []
    seg = front_segs.get("cube")
    if seg is not None and seg.ok and len(cubes) > 0:
        for cand in seg.segmentations[:5]:
            pts = cloud_points(cand, front_name)
            if len(pts) < 8:
                continue
            st = cloud_stats(pts)
            sx, sy, top_pts = top_face(pts, st["z95"])
            residuals.append(num(min(math.hypot(c["cx"] - sx, c["cy"] - sy) for c in cubes)))
    report["calibration_residuals_m"] = residuals
    report["phases"]["geometry"] = "done: " + str(len(cubes)) + " cubes"
except Exception as exc:
    err("phase geometry", exc)
    report["phases"]["geometry"] = "error"
note("table_z=" + str(report.get("table_z")) + " cubes=" + str(len(cubes)) + " button=" + str(report.get("button")))


# ================================================================ Phase 4: live highlight watch
def step_and_view(label, grip_value):
    count = hold(1, label, grip_value)
    if count == 0:
        return None
    try:
        return front_view(get_observation())
    except Exception as exc:
        err("watch observation", exc)
        return None


def watch_for_highlight(ref_rgb, max_ticks, label):
    """Hold still, compare each tick's front RGB with ref_rgb; return the view at onset (+2 ticks)."""
    series = []
    consecutive = 0
    for tick_index in range(max_ticks):
        if ctl["terminal"]:
            return None, series
        view = step_and_view(label, 1.0)
        if view is None:
            return None, series
        changed = count_changed(view["rgb"], ref_rgb, 2, 60)
        series.append(changed)
        if changed >= 10:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= 2:
            for extra in range(2):
                later = step_and_view(label + " settle", 1.0)
                if later is not None:
                    view = later
                    series.append(count_changed(view["rgb"], ref_rgb, 2, 60))
            return view, series
    return None, series


def analyse_highlight(view, ref_rgb, source):
    info = {"source": source, "tick": num(elapsed_ticks(), 1)}
    table_score, raised_score, unassigned = highlight_scores(view, ref_rgb, cubes, table_z, 60)
    info["table_score"] = table_score
    info["raised_score"] = raised_score
    info["unassigned_changed_px"] = unassigned
    chosen, top = choose_targets(table_score)
    info["metric"] = "table_changed_px"
    if len(chosen) == 0:
        chosen, top = choose_targets(raised_score)
        info["metric"] = "raised_changed_px"
    info["selected"] = chosen
    info["top_score"] = top
    try:
        info["thumb_front"] = thumbnail(view["rgb"], 0, 0, W, H, 4)
    except Exception as exc:
        err("highlight thumb", exc)
    for prompt in ("white circle", "highlighted cube"):
        rec, seg = probe(front_name, prompt)
        rec["purpose"] = "record only, at highlight frame"
        report["probes"].append(rec)
    return info


targets = []
try:
    if ref_view is not None and len(cubes) > 0:
        view_h, series = watch_for_highlight(ref_view["rgb"], 120, "watch")
        report["watch_series_changed_px_stride2"] = series
        if view_h is not None:
            info = analyse_highlight(view_h, ref_view["rgb"], "hold from tick 0")
            report["highlight"] = info
            targets = list(info["selected"])
        if len(targets) == 0 and not ctl["terminal"]:
            note("no highlight while holding; pressing the button, then watching again")
            report["highlight_fallback"] = "press then watch"
    report["phases"]["watch"] = "done"
except Exception as exc:
    err("phase watch", exc)
    report["phases"]["watch"] = "error"
note("highlight targets: " + str(targets))


# ================================================================ Phase 5: button press
def down_quat():
    if z_axis_z(home_quat) > -0.85:
        return [0.0, 1.0, 0.0, 0.0]
    return home_quat


def press_button(label):
    if button is None or ctl["terminal"]:
        return False
    q = down_quat()
    bx = button["x"]
    by = button["y"]
    bz = button["z_top"]
    gripper_cmd(True, label + " close fingers")
    move((bx, by, bz + 0.10), q, label + " hover")
    move((bx, by, bz - 0.012), q, label + " press", tol=0.02, steps=40)
    hold(4, label + " dwell", 0.0)
    move((bx, by, bz + 0.12), q, label + " release")
    return True


try:
    report["button_pressed_first"] = press_button("button1")
    if len(targets) == 0 and len(cubes) > 0 and not ctl["terminal"]:
        try:
            record_exec(go_home(), "home before second watch")
        except Exception as exc:
            err("go_home", exc)
        ref2 = front_view(get_observation())
        view_h, series = watch_for_highlight(ref2["rgb"], 150, "watch2")
        report["watch2_series_changed_px_stride2"] = series
        if view_h is not None:
            info = analyse_highlight(view_h, ref2["rgb"], "after button press")
            report["highlight2"] = info
            targets = list(info["selected"])
    if len(targets) == 0:
        report["target_source"] = "none: no highlight observed; nothing is picked (a guess would pick a wrong cube)"
    else:
        report["target_source"] = "live highlight"
    report["phases"]["button"] = "done"
except Exception as exc:
    err("phase button", exc)
    report["phases"]["button"] = "error"


# ================================================================ Phase 6: pick each target
def wrist_refine(expect_xy):
    best = None
    for prompt in ("cube", "block"):
        rec, seg = probe(wrist_name, prompt)
        rec["purpose"] = "wrist refine"
        report["wrist_probes"].append(rec)
        if seg is None:
            continue
        try:
            if not seg.ok:
                continue
            cands = seg.segmentations[:8]
        except Exception:
            continue
        for cand in cands:
            pts = cloud_points(cand, wrist_name)
            if len(pts) < 8:
                continue
            st = cloud_stats(pts)
            if st["x95"] - st["x05"] > 0.09 or st["y95"] - st["y05"] > 0.09:
                continue
            if table_z is not None and st["z95"] > table_z + 0.12:
                continue
            tx, ty, top_pts = top_face(pts, st["z95"])
            dist = math.hypot(tx - expect_xy[0], ty - expect_xy[1])
            if dist < 0.02 and (best is None or dist < best["dist"]):
                best = {"dist": dist, "xy": (tx, ty), "z95": st["z95"], "top": top_pts, "prompt": prompt}
        if best is not None:
            break
    return best


def redetect(target):
    """Re-find the target among fresh depth clusters (cubes can be displaced); None if ambiguous."""
    try:
        view = front_view(get_observation())
        fresh = [c for c in elevated_clusters(view, table_z) if cube_like(c, table_z)]
    except Exception as exc:
        err("redetect", exc)
        return None
    best = None
    for c in fresh:
        dist = math.hypot(c["cx"] - target["cx"], c["cy"] - target["cy"])
        if dist < 0.04 and (best is None or dist < best[0]):
            best = (dist, c)
    if best is None:
        return None
    return best


report["wrist_probes"] = []
report["picks"] = []
try:
    remaining = list(targets)
    here_xy = (button["x"], button["y"]) if button is not None else (0.5, 0.0)
    order = []
    while len(remaining) > 0:
        nearest = None
        for tid in remaining:
            c = cubes[tid]
            dist = math.hypot(c["cx"] - here_xy[0], c["cy"] - here_xy[1])
            if nearest is None or dist < nearest[0]:
                nearest = (dist, tid)
        order.append(nearest[1])
        remaining = [t for t in remaining if t != nearest[1]]
        here_xy = (cubes[nearest[1]]["cx"], cubes[nearest[1]]["cy"])
    report["pick_order"] = order
    for order_index, target_id in enumerate(order):
        if ctl["terminal"]:
            break
        cube = cubes[target_id]
        rec = {"cube": target_id, "order": order_index, "tick_start": num(elapsed_ticks(), 1)}
        report["picks"].append(rec)
        note("pick cube " + str(target_id))
        found = redetect(cube)
        tx = cube["cx"]
        ty = cube["cy"]
        z_top = cube["z_top"]
        theta = cube["theta"]
        if found is not None:
            rec["redetect_shift_m"] = num(found[0])
            tx = found[1]["cx"]
            ty = found[1]["cy"]
            z_top = found[1]["z_top"]
            theta = found[1]["theta"]
        else:
            rec["redetect"] = "no fresh cluster within 4 cm; using tick-0 position"
        neighbors = [(c["cx"], c["cy"]) for c in cubes if c["id"] != target_id]
        quats = grasp_quats(theta, (tx, ty), neighbors)
        quat = quats[0]["quat"]
        rec["yaw_options"] = [{"delta_deg": q["delta_deg"], "conflicts": q["conflicts"]} for q in quats]
        gripper_cmd(False, "open " + str(target_id))
        ok, dist = move((tx, ty, z_top + 0.15), quat, "hover " + str(target_id))
        if (not ok) and (dist is None or dist > 0.03) and not ctl["terminal"]:
            quat = quats[1]["quat"]
            rec["used_alt_yaw"] = True
            ok, dist = move((tx, ty, z_top + 0.15), quat, "hover alt " + str(target_id))
        refined = None
        if not ctl["terminal"]:
            refined = wrist_refine((tx, ty))
        if refined is not None:
            rec["wrist_shift_m"] = num(refined["dist"])
            tx = refined["xy"][0]
            ty = refined["xy"][1]
            wtheta = yaw_from_points(refined["top"])
            if wtheta is not None:
                rec["wrist_yaw_deg"] = num(math.degrees(wtheta), 1)
                quats = grasp_quats(wtheta, (tx, ty), neighbors)
                if rec.get("used_alt_yaw"):
                    quat = quats[1]["quat"]
                else:
                    quat = quats[0]["quat"]
        height = clamp(z_top - table_z, 0.015, 0.08)
        grasp_z = table_z + max(0.5 * height, 0.012)
        rec["grasp_xyz"] = [num(tx), num(ty), num(grasp_z)]
        move((tx, ty, grasp_z + 0.07), quat, "pregrasp " + str(target_id))
        move((tx, ty, grasp_z), quat, "grasp " + str(target_id), tol=0.008, steps=150)
        width = gripper_cmd(True, "close " + str(target_id))
        rec["width_after_close"] = num(width)
        if width is not None and width < 0.02 and not ctl["terminal"]:
            rec["retry"] = True
            gripper_cmd(False, "reopen " + str(target_id))
            move((tx, ty, z_top + 0.15), quat, "rehover " + str(target_id))
            refined = None
            if not ctl["terminal"]:
                refined = wrist_refine((tx, ty))
            if refined is not None:
                tx = refined["xy"][0]
                ty = refined["xy"][1]
                rec["retry_shift_m"] = num(refined["dist"])
            move((tx, ty, grasp_z + 0.07), quat, "repregrasp " + str(target_id))
            move((tx, ty, grasp_z), quat, "regrasp " + str(target_id), tol=0.008, steps=150)
            width = gripper_cmd(True, "reclose " + str(target_id))
            rec["width_after_reclose"] = num(width)
        move((tx, ty, grasp_z + 0.15), quat, "lift " + str(target_id))
        hold(6, "hold lifted " + str(target_id), 0.0)
        rec["width_lifted"] = num(gripper_now())
        rec["held_after_lift"] = (rec["width_lifted"] is not None and rec["width_lifted"] > 0.02)
        move((tx, ty, grasp_z + 0.008), quat, "lower " + str(target_id))
        gripper_cmd(False, "release " + str(target_id))
        move((tx, ty, grasp_z + 0.12), quat, "retreat " + str(target_id))
        rec["tick_end"] = num(elapsed_ticks(), 1)
    report["phases"]["pick"] = "done"
except Exception as exc:
    err("phase pick", exc)
    report["phases"]["pick"] = "error"

# ================================================================ Phase 7: final button press
try:
    if len(targets) > 0 and not ctl["terminal"]:
        report["button_pressed_final"] = press_button("button2")
        hold(10, "final settle", 0.0)
    report["phases"]["final_press"] = "done"
except Exception as exc:
    err("phase final press", exc)
    report["phases"]["final_press"] = "error"

report["terminal"] = ctl["terminal"]
report["terminal_label"] = ctl["terminal_label"]
report["final_tick"] = num(elapsed_ticks(), 1)
note("finished; terminal=" + str(ctl["terminal"]) + " at " + str(ctl["terminal_label"]))
result = report
