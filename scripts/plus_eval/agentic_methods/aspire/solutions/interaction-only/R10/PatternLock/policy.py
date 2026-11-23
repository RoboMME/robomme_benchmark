# v01 — PatternLock: v00 plus explicit button identification and a short press (dip) at each
# pattern button. v00 replayed the video's EE path to <1 mm but the first press never registered.
#
# Strategy (all values derived at runtime from this episode's own inputs):
#  1. Read the conditioning video's recorded EE path; z_min = its lowest EE height; frames
#     within PRESS_BAND of z_min are "at the buttons".
#  2. Buttons: bright low-saturation pixels in the front image (video frame 0 and live),
#     projected with depth + calibration, lowest band kept, clustered in XY. Video->live
#     registration = centroid difference of the button pixels.
#  3. Pattern = ordered distinct video buttons within NODE_R of the tool at pressing height.
#  4. Execute: move_to_pose above the first button, then per-tick IK lines at the video's
#     travel height between buttons, with a short dip DIP_BELOW under z_min at each button.

import math

STEP_M = 0.005           # m of tool travel per control tick while tracing
BUTTON_BAND = 0.03       # m above the lowest bright-gray point kept as grid buttons
MAX_SHIFT = 0.15         # m; larger registration shifts are treated as unreliable
CLUSTER_R = 0.02         # m, XY radius for grouping button pixels into one button
MIN_CLUSTER_PTS = 8      # pixels needed for a button cluster
PRESS_BAND = 0.004       # m above the video's lowest EE z that counts as "at a button"
NODE_R = 0.02            # m, XY distance from a button centre that counts as visiting it
DIP_BELOW = 0.004        # m below the video's lowest EE z for the press
DIP_TICKS = 3            # ticks down, held, and up for each press
HOVER = 0.04             # m above travel height for the first approach

report = {"task": get_task_context().language}
state0 = get_robot_state()
base_frame = state0.base_frame
embodiment = state0.embodiment
home_quat = state0.end_effector_poses["primary"].quaternion_wxyz
grip_cmd = float(state0.gripper_positions["primary"])


def quat_to_matrix(q):
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def button_points(cam):
    """Bright low-saturation pixels projected to the base frame (pinhole, z-depth)."""
    rgb = cam.rgb
    depth = cam.depth_m
    K = cam.intrinsics
    fx = float(K[0][0]); fy = float(K[1][1]); cx = float(K[0][2]); cy = float(K[1][2])
    R = quat_to_matrix(cam.camera_pose.quaternion_wxyz)
    t = [float(v) for v in cam.camera_pose.position]
    h = len(rgb)
    w = len(rgb[0])
    pts = []
    for v in range(h):
        row = rgb[v]
        drow = depth[v]
        for u in range(w):
            px = row[u]
            r = int(px[0]); g = int(px[1]); b = int(px[2])
            hi = max(r, g, b)
            lo = min(r, g, b)
            if hi - lo < 40 and hi > 120:
                d = float(drow[u])
                if d <= 0.0 or d != d:
                    continue
                xc = (u + 0.5 - cx) / fx * d
                yc = (v + 0.5 - cy) / fy * d
                zc = d
                pts.append((R[0][0] * xc + R[0][1] * yc + R[0][2] * zc + t[0],
                            R[1][0] * xc + R[1][1] * yc + R[1][2] * zc + t[1],
                            R[2][0] * xc + R[2][1] * yc + R[2][2] * zc + t[2]))
    if len(pts) < 20:
        return None
    zs = sorted(p[2] for p in pts)
    z_floor = zs[len(zs) // 20]
    kept = [p for p in pts if p[2] <= z_floor + BUTTON_BAND]
    if len(kept) < 20:
        return None
    return kept


def centroid(pts):
    n = float(len(pts))
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n, sum(p[2] for p in pts) / n)


def cluster_buttons(pts):
    clusters = []   # each: [sx, sy, sz, n]
    for p in pts:
        hit = None
        for c in clusters:
            cx = c[0] / c[3]
            cy = c[1] / c[3]
            if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 <= CLUSTER_R * CLUSTER_R:
                hit = c
                break
        if hit is None:
            clusters.append([p[0], p[1], p[2], 1])
        else:
            hit[0] += p[0]
            hit[1] += p[1]
            hit[2] += p[2]
            hit[3] += 1
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                a = clusters[i]
                b = clusters[j]
                if (a[0] / a[3] - b[0] / b[3]) ** 2 + (a[1] / a[3] - b[1] / b[3]) ** 2 <= CLUSTER_R * CLUSTER_R:
                    clusters[i] = [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
                    clusters.pop(j)
                    merged = True
                    break
            if merged:
                break
    out = []
    for c in clusters:
        if c[3] >= MIN_CLUSTER_PTS:
            out.append((c[0] / c[3], c[1] / c[3], c[2] / c[3]))
    return out


def nearest(centers, x, y, radius):
    best = None
    best_d = None
    for i in range(len(centers)):
        d = math.sqrt((centers[i][0] - x) ** 2 + (centers[i][1] - y) ** 2)
        if d <= radius and (best_d is None or d < best_d):
            best = i
            best_d = d
    return best


# ---- 1. Video path ----
video = robomme_plus.get_conditioning_video()
if len(video) == 0:
    raise ValueError("no conditioning video")
path = []
for frame in video:
    p = frame.robot_state.end_effector_poses["primary"].position
    path.append((float(p[0]), float(p[1]), float(p[2])))
z_min = min(p[2] for p in path)
press_frames = [i for i in range(len(path)) if path[i][2] <= z_min + PRESS_BAND]
travel_zs = sorted(path[i][2] for i in press_frames)
travel_z = travel_zs[len(travel_zs) // 2]
dip_z = z_min - DIP_BELOW
report["video_frames"] = len(video)
report["video_z_min"] = round(z_min, 4)
report["travel_z"] = round(travel_z, 4)

# ---- 2. Buttons in video frame 0 and live; registration ----
vid_pts = button_points(video[0].cameras["front"])
live_pts = button_points(get_observation().cameras["front"])
if vid_pts is None or live_pts is None:
    raise ValueError("button pixels not found")
vid_buttons = cluster_buttons(vid_pts)
live_buttons = cluster_buttons(live_pts)
cv = centroid(vid_pts)
cl = centroid(live_pts)
shift = (cl[0] - cv[0], cl[1] - cv[1])
if math.sqrt(shift[0] ** 2 + shift[1] ** 2) > MAX_SHIFT:
    shift = (0.0, 0.0)
report["buttons"] = {"video": len(vid_buttons), "live": len(live_buttons),
                     "shift": [round(shift[0], 4), round(shift[1], 4)]}

# ---- 3. Pattern = ordered distinct buttons the video's tool visited at pressing height ----
seq = []
runs = []          # frame indices assigned to each visit
for i in press_frames:
    b = nearest(vid_buttons, path[i][0], path[i][1], NODE_R)
    if b is None:
        continue
    if len(seq) == 0 or seq[len(seq) - 1] != b:
        seq.append(b)
        runs.append([i])
    else:
        runs[len(runs) - 1].append(i)
if len(seq) == 0:
    raise ValueError("no pattern buttons identified in the video")
nodes = []
for k in range(len(seq)):
    b = seq[k]
    c = vid_buttons[b]
    dists = [math.sqrt((path[i][0] - c[0]) ** 2 + (path[i][1] - c[1]) ** 2) for i in runs[k]]
    dmin = min(dists)
    close = [runs[k][j] for j in range(len(dists)) if dists[j] <= dmin + 0.003]
    tx = sum(path[i][0] for i in close) / len(close)
    ty = sum(path[i][1] for i in close) / len(close)
    # per-button registration: offset between matched live and video button centres
    lb = nearest(live_buttons, c[0] + shift[0], c[1] + shift[1], 0.03)
    if lb is not None:
        ox = live_buttons[lb][0] - c[0]
        oy = live_buttons[lb][1] - c[1]
    else:
        ox = shift[0]
        oy = shift[1]
    nodes.append((tx + ox, ty + oy))
report["video_pattern"] = [[round(vid_buttons[b][0], 4), round(vid_buttons[b][1], 4)] for b in seq]
report["live_nodes"] = [[round(n[0], 4), round(n[1], 4)] for n in nodes]

# ---- 4. Execute ----
run_state = {"ended": False, "ticks": 0, "ik_fail": 0, "seed": None}


def hold_tick(joints):
    res = step(RobotAction(arms={"primary": ArmCommand(
        "joint_position", joints, gripper_position=grip_cmd, embodiment=embodiment)}))
    run_state["ticks"] += 1
    try:
        if res.terminated or res.truncated:
            run_state["ended"] = True
    except Exception:
        pass
    return res


def go_line(a, b, n=None):
    if n is None:
        seg = math.sqrt(sum((b[i] - a[i]) ** 2 for i in range(3)))
        n = max(1, int(math.ceil(seg / STEP_M)))
    for k in range(1, n + 1):
        if run_state["ended"]:
            return
        s = float(k) / n
        p = (a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]), a[2] + s * (b[2] - a[2]))
        ik = solve_ik(Pose(p, home_quat, base_frame), seed_joints=run_state["seed"])
        if not ik.ok:
            run_state["ik_fail"] += 1
            continue
        run_state["seed"] = ik.joint_positions
        hold_tick(run_state["seed"])


def hold(n):
    for k in range(n):
        if run_state["ended"] or run_state["seed"] is None:
            return
        hold_tick(run_state["seed"])


hover = (nodes[0][0], nodes[0][1], travel_z + HOVER)
mv = move_to_pose(Pose(hover, home_quat, base_frame), tolerance=0.01, max_steps=150)
report["approach_ok"] = bool(mv.ok)
try:
    report["approach_steps"] = int(mv.steps_executed)
    if mv.terminated or mv.truncated:
        run_state["ended"] = True
except Exception:
    pass
if not run_state["ended"]:
    run_state["seed"] = get_robot_state().joint_positions["primary"]

cur = hover
pressed = 0
for n in nodes:
    if run_state["ended"]:
        break
    above = (n[0], n[1], travel_z)
    go_line(cur, above)
    go_line(above, (n[0], n[1], dip_z), DIP_TICKS)
    hold(DIP_TICKS)
    go_line((n[0], n[1], dip_z), above, DIP_TICKS)
    pressed += 1
    cur = above

report["pressed"] = pressed
report["trace_ticks"] = run_state["ticks"]
report["ik_failures"] = run_state["ik_fail"]
report["ended"] = run_state["ended"]
if not run_state["ended"]:
    hold(5)
    try:
        fin = get_robot_state()
        report["final_ee"] = [round(float(v), 4) for v in fin.end_effector_poses["primary"].position]
    except Exception as exc:
        report["final_error"] = str(exc)

result = report
