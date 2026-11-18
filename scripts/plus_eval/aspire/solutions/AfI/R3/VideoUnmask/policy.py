# v01 — VideoUnmask: full ordered multi-container sequence.
#
# Evidence from v00 (ep 0): the goal lists up to three cube colours in order
# ("pick up the container hiding the blue cube, followed by the one hiding the
# red cube, and finally the one hiding the green cube"). The conditioning video
# (static robot, static front camera identical to live) shows groups of cubes
# briefly uncovered; each cube sits under one of five white containers.
# v00 lifted the correct first (blue) container (held width 0.73) but did not
# continue, so the episode never succeeded.
# Changes (task-predicate mode: incomplete subgoal sequence):
#   - parse every colour in goal order and execute them all;
#   - cube 3D point = OpenCV deprojection of the video front depth at the cube's
#     last well-visible pixel (v00 check: within 2 mm xy of the container cloud);
#   - container = live candidate (union of prompts "box", "container", "bin")
#     whose top-centre is nearest that 3D point (pixel-box containment fallback);
#   - between picks put the container back down where it came from, open,
#     retreat to transit height; after the last lift hold and watch termination.
import math


report = {"version": "v01", "stage": "start"}
result = report

COLOR_WORDS = ["red", "green", "blue", "yellow", "orange", "purple", "pink",
               "magenta", "cyan"]
STRIDE = 2
MAX_VIDEO_FRAMES = 128


def hsv_of(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn
    if mx <= 0:
        return 0.0, 0.0, 0.0
    s = d / float(mx)
    if d == 0:
        h = 0.0
    elif mx == r:
        h = (60.0 * (g - b) / float(d)) % 360.0
    elif mx == g:
        h = 60.0 * (b - r) / float(d) + 120.0
    else:
        h = 60.0 * (r - g) / float(d) + 240.0
    return h, s, mx / 255.0


def color_class(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx - mn < 45 or mx < 50:
        return None
    h, s, v = hsv_of(r, g, b)
    if s < 0.45:
        return None
    if h < 15.0 or h >= 345.0:
        return "red"
    if h < 40.0:
        return "orange"
    if h < 70.0:
        return "yellow"
    if h < 165.0:
        return "green"
    if h < 195.0:
        return "cyan"
    if h < 260.0:
        return "blue"
    if h < 300.0:
        return "purple"
    return "magenta"


def image_rows(rgb):
    try:
        return rgb.tolist(), True
    except Exception:
        return None, False


def classify_image(rgb, stride):
    """Return dict colour -> list of (gx, gy) grid cells (pixel = cell * stride)."""
    cells = {}
    rows, fast = image_rows(rgb)
    height = len(rgb)
    width = len(rgb[0])
    if not fast:
        stride = max(stride, 4)
    for y in range(0, height, stride):
        row = rows[y] if fast else None
        for x in range(0, width, stride):
            if fast:
                p = row[x]
                r, g, b = p[0], p[1], p[2]
            else:
                p = rgb[y][x]
                r, g, b = int(p[0]), int(p[1]), int(p[2])
            name = color_class(r, g, b)
            if name is None:
                continue
            if name not in cells:
                cells[name] = []
            cells[name].append((x // stride, y // stride))
    return cells, stride


def components(cell_list, max_cells):
    """Connected components (8-neighbour) on grid cells; biggest first."""
    if len(cell_list) > max_cells:
        return None
    remaining = set(cell_list)
    comps = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            cx, cy = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cx + dx, cy + dy)
                    if nb in remaining:
                        remaining.discard(nb)
                        stack.append(nb)
                        comp.append(nb)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def blob_summary(comp, stride):
    xs = [c[0] * stride for c in comp]
    ys = [c[1] * stride for c in comp]
    return {"n": len(comp),
            "u": round(sum(xs) / float(len(xs)), 1),
            "v": round(sum(ys) / float(len(ys)), 1),
            "box": [min(xs), min(ys), max(xs) + stride, max(ys) + stride]}


def pose_list(pose):
    try:
        return {"p": [round(float(v), 4) for v in pose.position],
                "q": [round(float(v), 4) for v in pose.quaternion_wxyz],
                "frame": str(pose.frame)}
    except Exception as exc:
        return {"error": str(exc)}


def quat_to_rot(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def deproject(cam, u, v):
    """Pixel -> base frame under OpenCV and OpenGL camera conventions (diagnostic)."""
    try:
        d = float(cam.depth_m[int(v)][int(u)])
        k = cam.intrinsics
        fx, fy, cx, cy = float(k[0][0]), float(k[1][1]), float(k[0][2]), float(k[1][2])
        xc = (u - cx) / fx * d
        yc = (v - cy) / fy * d
        rot = quat_to_rot(cam.camera_pose.quaternion_wxyz)
        t = [float(a) for a in cam.camera_pose.position]
        out = {"depth": round(d, 4)}
        for name, pc in (("opencv", (xc, yc, d)), ("opengl", (xc, -yc, -d))):
            out[name] = [round(t[i] + rot[i][0] * pc[0] + rot[i][1] * pc[1]
                               + rot[i][2] * pc[2], 4) for i in range(3)]
        return out
    except Exception as exc:
        return {"error": str(exc)}


def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    return ordered[int((len(ordered) - 1) * fraction)]


def flags_of(res):
    out = {}
    for key in ("terminated", "truncated"):
        try:
            if key == "terminated":
                out[key] = bool(res.terminated)
            else:
                out[key] = bool(res.truncated)
        except Exception:
            pass
    try:
        out["status"] = str(res.status)
    except Exception:
        pass
    try:
        out["steps"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        out["ok"] = bool(res.ok)
    except Exception:
        pass
    return out


def ended(flags):
    return bool(flags.get("terminated")) or bool(flags.get("truncated"))


log = []
report["log"] = log
run_state = {"over": False}


def record(name, res):
    f = flags_of(res)
    f["call"] = name
    try:
        f["t"] = round(float(get_robot_state().timestamp_s), 3) if not ended(f) else None
    except Exception:
        pass
    log.append(f)
    if ended(f):
        run_state["over"] = True
    return f


def hold(ticks, gripper):
    state = get_robot_state()
    for tick in range(ticks):
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=state.joint_positions["primary"],
            gripper_position=gripper, embodiment=state.embodiment)}))
        f = flags_of(reply)
        if ended(f):
            f["call"] = "hold"
            f["tick"] = tick
            log.append(f)
            run_state["over"] = True
            return False
    return True


# ---------------------------------------------------------------- 1. context
context = get_task_context()
goal = str(context.language)
report["goal"] = goal
report["task_name"] = str(context.task_name)
try:
    report["task_metadata"] = {str(k): str(v)[:300] for k, v in dict(robomme_plus.get_task_metadata()).items()}
except Exception as exc:
    report["task_metadata"] = {"error": str(exc)}
try:
    report["controller_metadata"] = {str(k): str(v)[:300] for k, v in dict(robomme_plus.get_controller_metadata()).items()}
except Exception as exc:
    report["controller_metadata"] = {"error": str(exc)}

state0 = get_robot_state()
report["ee_start"] = pose_list(state0.end_effector_poses["primary"])
report["gripper_start"] = float(state0.gripper_positions["primary"])
report["joints_start"] = [round(float(v), 4) for v in state0.joint_positions["primary"]]
base_frame = state0.base_frame

live = get_observation()
report["cameras"] = {}
for cam_name in live.cameras:
    cam = live.cameras[cam_name]
    report["cameras"][cam_name] = {"pose": pose_list(cam.camera_pose),
                                   "shape": [len(cam.rgb), len(cam.rgb[0])]}
    try:
        report["cameras"][cam_name]["K"] = [[round(float(v), 3) for v in row] for row in cam.intrinsics]
    except Exception:
        pass

lower_goal = goal.lower()
cleaned = ""
for ch in lower_goal:
    cleaned += ch if ("a" <= ch <= "z") else " "
targets = [word for word in cleaned.split() if word in COLOR_WORDS]   # whole words, goal order
report["goal_colors"] = targets
report["stage"] = "context"

# ---------------------------------------------------------- 2. conditioning video
video = list(robomme_plus.get_conditioning_video())
n_video = len(video)
report["video_frames"] = n_video
timeline = []
tracks = {}
if n_video > 0:
    if n_video <= MAX_VIDEO_FRAMES:
        indices = list(range(n_video))
    else:
        indices = sorted(set(int(round(i * (n_video - 1) / float(MAX_VIDEO_FRAMES - 1)))
                             for i in range(MAX_VIDEO_FRAMES)))
    report["video_indices_used"] = len(indices)
    first_front = video[0].cameras["front"]
    last_front = video[n_video - 1].cameras["front"]
    report["video_first_front_pose"] = pose_list(first_front.camera_pose)
    report["video_last_front_pose"] = pose_list(last_front.camera_pose)
    report["video_t_first"] = float(video[0].timestamp_s)
    report["video_t_last"] = float(video[n_video - 1].timestamp_s)
    try:
        for fi in indices:
            frame = video[fi]
            cam = frame.cameras["front"]
            cells, used_stride = classify_image(cam.rgb, STRIDE)
            entry = {"i": fi, "t": round(float(frame.timestamp_s), 3)}
            try:
                entry["ee"] = [round(float(v), 3) for v in frame.robot_state.end_effector_poses["primary"].position]
                entry["grip"] = round(float(frame.robot_state.gripper_positions["primary"]), 3)
            except Exception:
                pass
            blobs = {}
            for name in cells:
                comps = components(cells[name], 2500)
                if comps is None:
                    blobs[name] = {"n_total": len(cells[name]), "too_many": True}
                    continue
                top = [blob_summary(c, used_stride) for c in comps[:3] if len(c) >= 3]
                if top:
                    blobs[name] = {"n_total": len(cells[name]), "blobs": top}
                    if name not in tracks:
                        tracks[name] = []
                    tracks[name].append((fi, top[0]))
            entry["blobs"] = blobs
            timeline.append(entry)
    except Exception as exc:
        report["video_error"] = str(exc)[:500]
report["stage"] = "video"


# compact timeline for the result: only frames where the set of visible colours changes
compact = []
previous_key = None
for entry in timeline:
    key = sorted(name for name in entry["blobs"] if "blobs" in entry["blobs"][name])
    if key != previous_key:
        compact.append({"i": entry["i"], "t": entry["t"], "visible": {
            name: entry["blobs"][name]["blobs"][0] for name in key}})
        previous_key = key
report["video_changes"] = compact

# ---------------------------------------------------------- 3. targets from the video
choice = {}
report["choice"] = choice


def track_target(name):
    track = tracks.get(name, [])
    if not track:
        return None
    max_n = max(item[1]["n"] for item in track)
    good = [item for item in track if item[1]["n"] >= 0.6 * max_n]
    last_good = good[-1]
    final_n = 0
    if timeline:
        final = timeline[-1]["blobs"].get(name, {})
        if "blobs" in final:
            final_n = final["blobs"][0]["n"]
    return {"color": name, "max_n": max_n, "last_good_frame": last_good[0],
            "u": last_good[1]["u"], "v": last_good[1]["v"], "box": last_good[1]["box"],
            "final_n": final_n}


def cube_point(cam, box):
    """Nearest-depth pixel inside the cube's blob box -> base frame (OpenCV convention)."""
    best = None
    for y in range(int(box[1]), int(box[3]) + 1):
        for x in range(int(box[0]), int(box[2]) + 1):
            if 0 <= y < len(cam.depth_m) and 0 <= x < len(cam.depth_m[0]):
                d = float(cam.depth_m[y][x])
                if d > 0.05 and (best is None or d < best[0]):
                    best = (d, x, y)
    if best is None:
        return None
    # median depth of the closest 25 % of box pixels would be more robust; use the
    # blob centroid ray at the cube-top depth instead
    return best


summaries = {}
for name in tracks:
    summaries[name] = track_target(name)
choice["tracks"] = summaries

plan = []
for name in targets:
    if name in summaries and summaries[name] is not None and summaries[name]["max_n"] >= 4:
        if not plan or plan[-1] != name:
            plan.append(name)
if not plan:
    best = None
    for name in summaries:
        s = summaries[name]
        if s["final_n"] < 0.3 * s["max_n"] and (best is None or s["max_n"] > summaries[best]["max_n"]):
            best = name
    if best is not None:
        plan.append(best)
    choice["fallback"] = True
choice["plan"] = plan

cube_xyz = {}
for name in plan:
    s = summaries[name]
    cam = video[s["last_good_frame"]].cameras["front"]
    near = cube_point(cam, s["box"])
    info = {"pixel": [s["u"], s["v"]]}
    if near is not None:
        k = cam.intrinsics
        fx, fy, cx0, cy0 = float(k[0][0]), float(k[1][1]), float(k[0][2]), float(k[1][2])
        d = near[0]
        pc = ((s["u"] - cx0) / fx * d, (s["v"] - cy0) / fy * d, d)
        rot = quat_to_rot(cam.camera_pose.quaternion_wxyz)
        t = [float(a) for a in cam.camera_pose.position]
        xyz = [t[i] + rot[i][0] * pc[0] + rot[i][1] * pc[1] + rot[i][2] * pc[2] for i in range(3)]
        info["xyz"] = [round(a, 4) for a in xyz]
        cube_xyz[name] = xyz
    choice[name] = info
report["stage"] = "targets"

# ---------------------------------------------------------- 4. live containers (arm at home)
probes = []
report["prompt_probes"] = probes
raw = []
for prompt in ("box", "container", "bin"):
    seg = segment_text("front", prompt)
    probe = {"prompt": prompt, "ok": bool(seg.ok), "n": 0}
    if seg.ok:
        probe["n"] = len(seg.segmentations)
        for cand in seg.segmentations[:8]:
            if cand.box_xyxy is not None:
                raw.append((prompt, cand, [float(v) for v in cand.box_xyxy], float(cand.score)))
    probes.append(probe)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


unique = []
for item in raw:
    if all(iou(item[2], other[2]) < 0.6 for other in unique):
        unique.append(item)

containers = []
report["containers"] = containers
for prompt, cand, box, score in unique:
    entry = {"prompt": prompt, "box": [round(v, 1) for v in box], "score": round(score, 3)}
    area = (box[2] - box[0]) * (box[3] - box[1])
    if area > 0.25 * 256 * 256:
        entry["reject"] = "box too large"
        containers.append(entry)
        continue
    try:
        cloud = mask_to_point_cloud(cand, "front", target_frame=base_frame)
        pts = cloud.points
    except Exception as exc:
        entry["reject"] = "cloud error " + str(exc)[:100]
        containers.append(entry)
        continue
    entry["points"] = len(pts)
    if len(pts) < 20:
        entry["reject"] = "too few points"
        containers.append(entry)
        continue
    top_z = quantile((p[2] for p in pts), 0.97)
    band = [p for p in pts if float(p[2]) >= top_z - 0.03]
    entry["top_z"] = round(top_z, 4)
    entry["center"] = [round(quantile((p[0] for p in band), 0.5), 4),
                       round(quantile((p[1] for p in band), 0.5), 4)]
    # a container stands on the table: bounded height, not the robot (base/arm)
    if top_z < 0.02 or top_z > 0.20:
        entry["reject"] = "implausible height"
    containers.append(entry)

valid = [c for c in containers if "reject" not in c]
assigned = []
picks = []
choice["picks"] = picks
for name in plan:
    best = None
    best_d = None
    if name in cube_xyz:
        xyz = cube_xyz[name]
        for c in valid:
            if c in assigned:
                continue
            d = ((c["center"][0] - xyz[0]) ** 2 + (c["center"][1] - xyz[1]) ** 2) ** 0.5
            if best_d is None or d < best_d:
                best_d = d
                best = c
        if best_d is not None and best_d > 0.06:
            best = None
    how = "nearest_3d"
    if best is None:
        how = "pixel_box"
        u, v = choice[name]["pixel"]
        for c in valid:
            if c in assigned:
                continue
            b = c["box"]
            if b[0] - 2 <= u <= b[2] + 2 and b[1] - 2 <= v <= b[3] + 2:
                d = ((u - (b[0] + b[2]) / 2.0) ** 2 + (v - (b[1] + b[3]) / 2.0) ** 2) ** 0.5
                if best_d is None or how != "pixel_box_set" or d < best_d:
                    best_d = d
                    best = c
                    how = "pixel_box_set"
    if best is not None:
        assigned.append(best)
        picks.append({"color": name, "how": how, "dist": round(float(best_d), 4),
                      "center": best["center"], "top_z": best["top_z"], "box": best["box"]})
    else:
        picks.append({"color": name, "how": "none"})
report["stage"] = "assigned"

# ---------------------------------------------------------- 5. ordered pick sequence
q0 = state0.end_effector_poses["primary"].quaternion_wxyz
if quat_to_rot(q0)[2][2] < -0.9:
    down = [float(v) for v in q0]
else:
    down = [0.0, 1.0, 0.0, 0.0]
choice["grasp_quat"] = [round(v, 4) for v in down]
GRASP_BELOW_TOP = 0.018   # grasp.md / ButtonUnmask evidence; v00 held at width 0.73
LIFT = 0.15
TRANSIT = 0.12


def go(name, x, y, z, tol, steps):
    r = move_to_pose(Pose((x, y, z), down, base_frame), tolerance=tol, max_steps=steps)
    f = record(name, r)
    return f


try:
    for index in range(len(picks)):
        pick = picks[index]
        if run_state["over"] or pick["how"] == "none":
            break
        cx, cy = pick["center"]
        gz = pick["top_z"] - GRASP_BELOW_TOP
        pick["grasp_z"] = round(gz, 4)
        record("open_" + pick["color"], open_gripper())
        if not run_state["over"]:
            go("pre_" + pick["color"], cx, cy, gz + TRANSIT, 0.015, 250)
        if not run_state["over"]:
            go("descend_" + pick["color"], cx, cy, gz, 0.012, 200)
        if not run_state["over"]:
            record("close_" + pick["color"], close_gripper())
        if not run_state["over"]:
            pick["held_width"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
            go("lift_" + pick["color"], cx, cy, gz + LIFT, 0.02, 250)
        if not run_state["over"]:
            hold(10, 0.0)
        if not run_state["over"]:
            st = get_robot_state()
            pick["lifted_width"] = round(float(st.gripper_positions["primary"]), 3)
            pick["lifted_ee_z"] = round(float(st.end_effector_poses["primary"].position[2]), 4)
            pick["lifted_t"] = round(float(st.timestamp_s), 3)
        last = index == len(picks) - 1
        if run_state["over"]:
            break
        if last:
            hold(40, 0.0)
            break
        # put the container back where it came from, release, retreat to transit height
        go("lower_" + pick["color"], cx, cy, gz + 0.01, 0.015, 250)
        if not run_state["over"]:
            record("release_" + pick["color"], open_gripper())
        if not run_state["over"]:
            hold(5, 1.0)
        if not run_state["over"]:
            go("retreat_" + pick["color"], cx, cy, gz + TRANSIT, 0.02, 200)
except Exception as exc:
    report["action_error"] = str(exc)[:500]
    run_state["over"] = True

report["stage"] = "done" if not run_state["over"] else "episode_ended"
report["episode_over"] = run_state["over"]
result = report
