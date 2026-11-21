# ButtonUnmaskSwap -- v00 initial program (ASPIRE CaP, RoboMME-Plus, demo-only: never run in simulator)
#
# Strategy (derived offline from the three released recordings, see task_analysis.md):
#   1. Tick 0: cubes are visible. Self-calibrate the RGB-D unprojection convention from table
#      flatness, find the table height and the 3D position of every goal-colour cube.
#   2. Hold the arm while the white containers land on the cubes; initialise one XY track per
#      container from the container-top height band; bind goal colours to tracks.
#   3. Press both buttons with a closed gripper using per-tick joint servoing (one step() per
#      tick) so every tick's front+wrist RGB-D updates the container tracks (Lloyd assignment).
#   4. Return to the start pose and keep tracking until all containers stop moving.
#   5. For each goal colour in goal order: top-down grasp of its tracked container, lift (the
#      pick registers), lower it back, release, rise.
# Every episode-dependent quantity (colours, order, positions, heights, counts) is derived at
# runtime from the goal string and live RGB-D. Constants below are generic tolerances.
import math

report = {"version": "v00_initial", "notes": [], "phase": "start"}


def note(msg):
    if len(report["notes"]) < 80:
        report["notes"].append(str(msg))


# ----------------------------------------------------------------------------- utilities
def to_list(arr, depth=0):
    # numpy arrays -> .tolist(); tuples/lists/sequences (e.g. Pose.position) -> recursive list
    try:
        return arr.tolist()
    except Exception:
        pass
    if depth > 3:
        return arr
    try:
        items = list(arr)
    except Exception:
        return arr
    return [to_list(item, depth + 1) for item in items]


def floats(vec):
    return [float(v) for v in to_list(vec)]


def median(values):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    return ordered[len(ordered) // 2]


def quantile(values, frac):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * frac)]


def quat_to_mat(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def transpose(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]


def wrap_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a <= -math.pi:
        a += 2 * math.pi
    return a


# ----------------------------------------------------------------------------- goal parsing
COLOR_WORDS = {
    "red": "red", "green": "green", "blue": "blue", "yellow": "yellow", "purple": "purple",
    "violet": "purple", "orange": "orange", "black": "black", "white": "white", "pink": "pink",
    "magenta": "pink", "cyan": "cyan", "gray": "gray", "grey": "gray",
}
NUMBER_WORDS = {"once": 1, "twice": 2, "thrice": 3, "one": 1, "two": 2, "three": 3, "four": 4}


def parse_goal(text):
    low = str(text).lower()
    for ch in ",.;:!?()":
        low = low.replace(ch, " ")
    words = low.split()
    order = []
    for i, w in enumerate(words):
        if w in COLOR_WORDS:
            nxt = words[i + 1:i + 3]
            if "cube" in nxt or "cubes" in nxt or "block" in nxt or "blocks" in nxt:
                order.append(COLOR_WORDS[w])
    press_mentioned = ("press" in words) or ("button" in words) or ("buttons" in words)
    all_buttons = ("both" in words) or ("buttons" in words) or ("all" in words)
    times = 1
    for i, w in enumerate(words):
        if w in ("once", "twice", "thrice"):
            times = NUMBER_WORDS[w]
        if w == "times" and i > 0 and words[i - 1] in NUMBER_WORDS:
            times = NUMBER_WORDS[words[i - 1]]
    return {"order": order, "press": press_mentioned, "all_buttons": all_buttons, "times": times}


# ----------------------------------------------------------------------------- pixel colours
def classify(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 60:
        return "black"
    sat = (mx - mn) / float(mx)
    if sat < 0.13:
        if mn >= 195:
            return "white"
        return "gray"
    if sat < 0.35:
        return None
    d = float(mx - mn)
    if mx == r:
        h = (60.0 * (g - b) / d) % 360.0
    elif mx == g:
        h = 60.0 * (b - r) / d + 120.0
    else:
        h = 60.0 * (r - g) / d + 240.0
    if h < 12 or h >= 335:
        return "red"
    if h < 42:
        return "orange"
    if h < 72:
        return "yellow"
    if h < 165:
        return "green"
    if h < 195:
        return "cyan"
    if h < 255:
        return "blue"
    if h < 300:
        return "purple"
    return "pink"


# ----------------------------------------------------------------------------- camera models
HYPS = ("cv", "gl", "link", "cv_inv", "gl_inv", "link_inv")
AXIS_MAPS = {
    "cv": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "gl": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
    "link": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
}


def cam_model(cam, hyp):
    K = to_list(cam.intrinsics)
    pose = cam.camera_pose
    t = floats(pose.position)
    R = quat_to_mat(floats(pose.quaternion_wxyz))
    if hyp.endswith("_inv"):
        Rt = transpose(R)
        t = [-(Rt[i][0] * t[0] + Rt[i][1] * t[1] + Rt[i][2] * t[2]) for i in range(3)]
        R = Rt
    A = matmul(R, AXIS_MAPS[hyp.replace("_inv", "")])
    return {"A": A, "t": t, "fx": float(K[0][0]), "fy": float(K[1][1]),
            "cx": float(K[0][2]), "cy": float(K[1][2])}


LAST_ARR = {}


def project(m, x, y, z):
    """robot_base point -> (u, v, depth) under the calibrated camera model, or None if behind."""
    A = m["A"]
    t = m["t"]
    px = x - t[0]
    py = y - t[1]
    pz = z - t[2]
    a = A[0][0] * px + A[1][0] * py + A[2][0] * pz
    b = A[0][1] * px + A[1][1] * py + A[2][1] * pz
    d = A[0][2] * px + A[1][2] * py + A[2][2] * pz
    if d <= 0.02:
        return None
    return (m["fx"] * a / d + m["cx"], m["fy"] * b / d + m["cy"], d)


def unproject(m, u, v, d):
    a = (u - m["cx"]) / m["fx"] * d
    b = (v - m["cy"]) / m["fy"] * d
    A = m["A"]
    t = m["t"]
    return (A[0][0] * a + A[0][1] * b + A[0][2] * d + t[0],
            A[1][0] * a + A[1][1] * b + A[1][2] * d + t[1],
            A[2][0] * a + A[2][1] * b + A[2][2] * d + t[2])


def camera_arrays(cam):
    return to_list(cam.rgb), to_list(cam.depth_m)


def table_samples(rgb, depth, stride):
    samples = []
    for v in range(0, len(rgb), stride):
        row = rgb[v]
        drow = depth[v]
        for u in range(0, len(row), stride):
            p = row[u]
            d = float(drow[u])
            if d <= 0.05 or d > 4.0:
                continue
            if classify(int(p[0]), int(p[1]), int(p[2])) == "orange":
                samples.append((u, v, d))
    return samples


def calibrate(obs, cam_names):
    """Choose the camera-frame convention under which the table is flat and below the camera."""
    scores = {}
    data = {}
    for name in cam_names:
        rgb, depth = camera_arrays(obs.cameras[name])
        data[name] = table_samples(rgb, depth, 4)
    best = None
    for hyp in HYPS:
        total = 0.0
        table = {}
        valid = True
        for name in cam_names:
            samples = data[name]
            if len(samples) < 30:
                continue
            m = cam_model(obs.cameras[name], hyp)
            zs = [unproject(m, u, v, d)[2] for (u, v, d) in samples]
            zmed = median(zs)
            mad = median([abs(z - zmed) for z in zs])
            if zmed > m["t"][2] - 0.05:
                valid = False
            total += mad
            table[name] = zmed
        if len(table) == 0:
            valid = False
        scores[hyp] = [round(total, 4), valid]
        if valid and (best is None or total < best[0]):
            best = (total, hyp, table)
    report["calibration_scores"] = scores
    if best is None:
        note("calibration failed; defaulting to cv")
        return "cv", None
    tz = [best[2][k] for k in best[2]]
    report["calibration"] = {"hyp": best[1], "mad": round(best[0], 4),
                             "table_z": {k: round(best[2][k], 4) for k in best[2]}}
    return best[1], median(tz)


# ----------------------------------------------------------------------------- task setup
ctx = get_task_context()
goal = ctx.language
plan = parse_goal(goal)
report["goal"] = goal
report["plan"] = plan
meta = {}
try:
    meta = dict(robomme_plus.get_task_metadata())
except Exception as exc:
    note("metadata error " + str(exc))
max_steps = None
for key in ("max_steps", "max_episode_steps", "horizon"):
    if key in meta and max_steps is None:
        try:
            max_steps = int(meta[key])
        except Exception:
            max_steps = None
control_hz = None
for key in ("control_hz", "control_frequency", "control_freq", "frequency_hz"):
    if key in meta and control_hz is None:
        try:
            control_hz = float(meta[key])
        except Exception:
            control_hz = None
if control_hz is None:
    control_hz = 20.0
report["max_steps"] = max_steps

state0 = get_robot_state()
EMB = state0.embodiment
BASE = state0.base_frame
HOME_Q = floats(state0.joint_positions["primary"])
HOME_POSE = state0.end_effector_poses["primary"]
HOME_POS = floats(HOME_POSE.position)
HOME_QUAT = floats(HOME_POSE.quaternion_wxyz)

obs0 = get_observation()
CAMS = [name for name in ("front", "wrist") if name in obs0.cameras]
if len(CAMS) == 0:
    CAMS = [name for name in obs0.cameras]
report["cameras"] = CAMS
HYP, TABLE_Z = calibrate(obs0, CAMS)
if TABLE_Z is None:
    TABLE_Z = 0.0

counters = {"ticks": 0, "terminal": False, "obs": obs0}
BASE_EXCL_R = 0.12      # ignore points this close (XY) to the robot base origin


def elapsed_ticks():
    try:
        return int(round(float(get_robot_state().timestamp_s) * control_hz))
    except Exception:
        return counters["ticks"]


def ticks_left():
    if max_steps is None:
        return 100000
    return max_steps - max(elapsed_ticks(), counters["ticks"])


# ----------------------------------------------------------------------------- point extraction
def frame_points(obs, mode, ee_xyz):
    """mode 'band': bright-white points in the container-top band. 'bright': bright-white
    elevated points (any height up to 0.25 m above table). 'colour': labelled elevated points."""
    out = []
    for name in CAMS:
        cam = obs.cameras[name]
        m = cam_model(cam, HYP)
        rgb, depth = camera_arrays(cam)
        LAST_ARR[name] = {"m": m, "depth": depth}
        stride = 1 if (mode == "colour" or name == "front") else 2
        for v in range(0, len(rgb), stride):
            row = rgb[v]
            drow = depth[v]
            for u in range(0, len(row), stride):
                p = row[u]
                g = int(p[1])
                if mode != "colour":
                    if g < 195:
                        continue
                    r = int(p[0])
                    b = int(p[2])
                    if r < 195 or b < 195 or max(r, g, b) - min(r, g, b) > 28:
                        continue
                    label = "white"
                else:
                    label = classify(int(p[0]), g, int(p[2]))
                    if label is None:
                        continue
                d = float(drow[u])
                if d <= 0.05 or d > 4.0:
                    continue
                x, y, z = unproject(m, u, v, d)
                e = z - TABLE_Z
                if x * x + y * y < BASE_EXCL_R * BASE_EXCL_R:
                    continue
                if mode == "band":
                    if e < BAND["lo"] or e > BAND["hi"]:
                        continue
                    if ee_xyz is not None and ee_xyz[2] - TABLE_Z < BAND["hi"] + 0.15:
                        dx = x - ee_xyz[0]
                        dy = y - ee_xyz[1]
                        if dx * dx + dy * dy < EE_EXCL_R * EE_EXCL_R:
                            continue
                    out.append((x, y, e, name))
                elif mode == "bright":
                    if e < 0.02 or e > 0.25:
                        continue
                    out.append((x, y, e))
                else:
                    if e < 0.003 or e > 0.30:
                        continue
                    out.append((x, y, e, label, name))
    return out


def grid_components(points, cell):
    cells = {}
    for p in points:
        key = (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)))
        if key in cells:
            cells[key].append(p)
        else:
            cells[key] = [p]
    seen = set()
    comps = []
    for key in cells:
        if key in seen:
            continue
        queue = [key]
        seen.add(key)
        members = []
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            members.append(cur)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cur[0] + dx, cur[1] + dy)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        queue.append(nb)
        pts = []
        for c in members:
            pts.extend(cells[c])
        comps.append({"cells": len(members), "points": pts,
                      "cx": sum(p[0] for p in pts) / len(pts),
                      "cy": sum(p[1] for p in pts) / len(pts)})
    return comps


def kmeans_split(points, k):
    xs = sorted(points, key=lambda p: p[0])
    centers = [[xs[int((i + 0.5) * len(xs) / k)][0], xs[int((i + 0.5) * len(xs) / k)][1]] for i in range(k)]
    for it in range(8):
        sums = [[0.0, 0.0, 0] for i in range(k)]
        for p in points:
            best = 0
            bd = None
            for j in range(k):
                dd = (p[0] - centers[j][0]) ** 2 + (p[1] - centers[j][1]) ** 2
                if bd is None or dd < bd:
                    bd = dd
                    best = j
            sums[best][0] += p[0]
            sums[best][1] += p[1]
            sums[best][2] += 1
        for j in range(k):
            if sums[j][2] > 0:
                centers[j] = [sums[j][0] / sums[j][2], sums[j][1] / sums[j][2]]
    return centers


# ----------------------------------------------------------------------------- tick-0 analysis
colour_pts = frame_points(obs0, "colour", None)
cube_seeds = {}
for colour in plan["order"]:
    pts = [p for p in colour_pts if p[3] == colour and 0.004 <= p[2] <= 0.09]
    comps = grid_components(pts, 0.012)
    comps = sorted(comps, key=lambda c: -len(c["points"]))
    if len(comps) > 0 and len(comps[0]["points"]) >= 3:
        cube_seeds[colour] = (comps[0]["cx"], comps[0]["cy"])
        if len(comps) > 1 and len(comps[1]["points"]) >= 0.5 * len(comps[0]["points"]):
            note("ambiguous colour " + colour + ": " + str(len(comps)) + " clusters")
    else:
        note("cube colour not found at tick 0: " + colour)
report["cube_seeds"] = {k: [round(cube_seeds[k][0], 3), round(cube_seeds[k][1], 3)] for k in cube_seeds}

# Count every cube at tick 0 (chromatic or black elevated blobs) to guide container splitting.
chromatic = [p for p in colour_pts if p[3] not in ("white", "gray") and 0.004 <= p[2] <= 0.09]
cube_comps = []
for lab in sorted(set(p[3] for p in chromatic)):
    for c in grid_components([p for p in chromatic if p[3] == lab], 0.012):
        if len(c["points"]) >= 3:
            c["label"] = lab
            cube_comps.append(c)
if len(cube_comps) > 0:
    med_cube_cells = median([c["cells"] for c in cube_comps])
    split_comps = []
    for c in cube_comps:
        k = int(round(c["cells"] / float(max(1, med_cube_cells))))
        if k >= 2 and c["cells"] >= 2.2 * med_cube_cells:
            for cc in kmeans_split(c["points"], k):
                split_comps.append({"cx": cc[0], "cy": cc[1], "cells": med_cube_cells, "label": c["label"],
                                    "points": []})
        else:
            split_comps.append(c)
    cube_comps = split_comps
report["cube_blobs_t0"] = [[c["label"], round(c["cx"], 3), round(c["cy"], 3)] for c in cube_comps]

# Button candidates at tick 0: low white/gray elevated blobs that are not part of tall objects.
neutral = [p for p in colour_pts if p[3] in ("white", "gray") and 0.003 <= p[2] <= 0.30]
neutral_comps = grid_components([p for p in neutral if p[2] <= 0.06], 0.01)
tall = [p for p in neutral if p[2] > 0.08]
button_cands = []
for c in neutral_comps:
    if len(c["points"]) < 6:
        continue
    xs = [p[0] for p in c["points"]]
    ys = [p[1] for p in c["points"]]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    if span < 0.015 or span > 0.10:
        continue
    near_tall = 0
    for p in tall:
        if (p[0] - c["cx"]) ** 2 + (p[1] - c["cy"]) ** 2 < (0.6 * span + 0.01) ** 2:
            near_tall += 1
    if near_tall > 5:
        continue
    gray_pts = [p for p in c["points"] if p[3] == "gray"]
    if len(gray_pts) >= 5:
        bx = median([p[0] for p in gray_pts])
        by = median([p[1] for p in gray_pts])
    else:
        bx, by = c["cx"], c["cy"]
    es = [p[2] for p in c["points"]]
    button_cands.append({"x": bx, "y": by, "top": quantile(es, 0.95), "base": quantile(es, 0.3),
                         "n": len(c["points"]), "span": span})
report["button_candidates_t0"] = [[round(b["x"], 3), round(b["y"], 3), round(b["top"], 3), b["n"]]
                                  for b in button_cands]
# White/grey cubes have no chromatic blob: add every cube-sized neutral low blob as a container seed.
# Seeds that receive no container-top points after landing (e.g. the buttons) are pruned later.
container_seeds = [[c["cx"], c["cy"]] for c in cube_comps]
for b in button_cands:
    dup = False
    for cc in container_seeds:
        if (cc[0] - b["x"]) ** 2 + (cc[1] - b["y"]) ** 2 < 0.02 ** 2:
            dup = True
    if not dup:
        container_seeds.append([b["x"], b["y"]])

# ----------------------------------------------------------------------------- per-tick control
BAND = {"lo": 0.025, "hi": 0.25}
EE_EXCL_R = 0.06
tracks = []
trk = {"Ra": 0.045, "still": 0, "log": [], "reacq": [], "orphans": 0}


def reply_obs(reply):
    try:
        o = reply.observation
        if o is not None:
            return o
    except Exception:
        pass
    return get_observation()


def is_terminal(res):
    for probe in ("terminated", "truncated", "done"):
        try:
            if probe == "terminated" and bool(res.terminated):
                return True
            if probe == "truncated" and bool(res.truncated):
                return True
            if probe == "done" and bool(res.done):
                return True
        except Exception:
            pass
    return False


def current_q():
    return floats(get_robot_state().joint_positions["primary"])


def ee_position(obs):
    try:
        return floats(obs.robot_state.end_effector_poses["primary"].position)
    except Exception:
        return floats(get_robot_state().end_effector_poses["primary"].position)


DISC = []
for i in range(-3, 4):
    for j in range(-3, 4):
        if i * i + j * j <= 10:
            DISC.append((i / 3.0, j / 3.0))


def visible_disc(cam_name, centre, ee):
    """Fraction and centroid of a disc (area of the measured container top) around `centre` that the
    camera can see unoccluded in the last frame; None if the camera frame is unavailable."""
    arr = LAST_ARR.get(cam_name)
    if arr is None:
        return None
    m = arr["m"]
    depth = arr["depth"]
    h = len(depth)
    w = len(depth[0]) if h > 0 else 0
    r = 0.56 * trk.get("side", 0.05)
    z = TABLE_Z + 0.5 * (BAND["lo"] + BAND["hi"])
    n_vis = 0
    sx = 0.0
    sy = 0.0
    for (du, dv) in DISC:
        x = centre[0] + r * du
        y = centre[1] + r * dv
        if x * x + y * y < BASE_EXCL_R * BASE_EXCL_R:
            continue
        if ee is not None and ee[2] - TABLE_Z < BAND["hi"] + 0.15:
            if (x - ee[0]) ** 2 + (y - ee[1]) ** 2 < EE_EXCL_R * EE_EXCL_R:
                continue
        pr = project(m, x, y, z)
        if pr is None:
            continue
        u = int(round(pr[0]))
        v = int(round(pr[1]))
        if u < 0 or v < 0 or u >= w or v >= h:
            continue
        dd = float(depth[v][u])
        if dd <= 0.05 or dd < pr[2] - 0.015:
            continue
        n_vis += 1
        sx += x
        sy += y
    if n_vis == 0:
        return (0.0, centre[0], centre[1])
    return (n_vis / float(len(DISC)), sx / n_vis, sy / n_vis)


def wrist_usable(ee):
    """The wrist camera sees whole container tops only from near the home height."""
    if ee is None:
        return True
    return ee[2] - TABLE_Z > 0.6 * (HOME_POS[2] - TABLE_Z)


def update_tracks(obs):
    if len(tracks) == 0:
        return
    ee = ee_position(obs)
    # partial views (low wrist camera, occluders) are handled by visible_disc, so every camera is used
    pts = frame_points(obs, "band", ee)
    pred = [[t["c"][0] + t["v"][0], t["c"][1] + t["v"][1]] for t in tracks]
    ra2 = trk["Ra"] * trk["Ra"]
    good = [False for t in tracks]
    for it in range(2):
        sums = [{} for t in tracks]
        for p in pts:
            best = -1
            bd = ra2
            for k in range(len(pred)):
                dx = p[0] - pred[k][0]
                dy = p[1] - pred[k][1]
                dd = dx * dx + dy * dy
                if dd < bd:
                    bd = dd
                    best = k
            if best >= 0:
                acc = sums[best].get(p[3])
                if acc is None:
                    acc = [0.0, 0.0, 0]
                    sums[best][p[3]] = acc
                acc[0] += p[0]
                acc[1] += p[1]
                acc[2] += 1
        for k in range(len(tracks)):
            ex = 0.0
            ey = 0.0
            wsum = 0.0
            for cam_name in sums[k]:
                acc = sums[k][cam_name]
                if acc[2] < 4:
                    continue
                vis = visible_disc(cam_name, pred[k], ee)
                if vis is None:
                    continue
                frac, vx, vy = vis
                if frac < 0.35:
                    continue
                # observed centroid minus the centroid of the predicted top's visible part
                ex += frac * (pred[k][0] + acc[0] / acc[2] - vx)
                ey += frac * (pred[k][1] + acc[1] / acc[2] - vy)
                wsum += frac
            good[k] = wsum > 0
            if wsum > 0:
                pred[k] = [ex / wsum, ey / wsum]
    max_speed = 0.0
    n_seen = 0
    for k in range(len(tracks)):
        t = tracks[k]
        if good[k]:
            t["streak"] = 0
            vx = pred[k][0] - t["c"][0]
            vy = pred[k][1] - t["c"][1]
            t["v"] = [0.5 * t["v"][0] + 0.5 * vx, 0.5 * t["v"][1] + 0.5 * vy]
            t["c"] = pred[k]
            t["seen"] += 1
            n_seen += 1
            sp = math.sqrt(t["v"][0] ** 2 + t["v"][1] ** 2)
            if sp > max_speed:
                max_speed = sp
        else:
            t["c"] = [t["c"][0] + 0.8 * t["v"][0], t["c"][1] + 0.8 * t["v"][1]]
            t["v"] = [0.8 * t["v"][0], 0.8 * t["v"][1]]
            t["missed"] += 1
            t["streak"] = t.get("streak", 0) + 1
    # Re-acquire lost tracks: a container-sized blob explained by no track is given to the lost track
    # whose last estimate is nearest (unambiguous when one container emerges from an occluder).
    orphans = []
    lost = [k for k in range(len(tracks)) if tracks[k].get("streak", 0) >= 3]
    if len(lost) > 0 or counters["ticks"] % 5 == 0:
        ra2b = (0.75 * trk.get("side", 0.05) + 0.01) ** 2
        free_pts = []
        for p in pts:
            near = False
            for t in tracks:
                if (p[0] - t["c"][0]) ** 2 + (p[1] - t["c"][1]) ** 2 < ra2b:
                    near = True
                    break
            if not near:
                free_pts.append(p)
        for c in grid_components(free_pts, 0.01):
            if c["cells"] >= 0.5 * trk.get("cells", 9):
                orphans.append(c)
    trk["orphans"] = len(orphans)
    for c in sorted(orphans, key=lambda c: -c["cells"]):
        if len(lost) == 0:
            break
        k = sorted(lost, key=lambda k: (tracks[k]["c"][0] - c["cx"]) ** 2 + (tracks[k]["c"][1] - c["cy"]) ** 2)[0]
        lost = [j for j in lost if j != k]
        t = tracks[k]
        if len(trk["reacq"]) < 60:
            trk["reacq"].append([counters["ticks"], t["id"], round(c["cx"], 3), round(c["cy"], 3),
                                 len([j for j in range(len(tracks)) if tracks[j].get("streak", 0) >= 3])])
        t["c"] = [c["cx"], c["cy"]]
        t["v"] = [0.0, 0.0]
        t["streak"] = 0
    trk["last_speed"] = max_speed
    trk["last_seen"] = n_seen
    if n_seen >= max(1, len(tracks) // 2) and max_speed < 0.0012 and trk["orphans"] == 0:
        trk["still"] += 1
    else:
        trk["still"] = 0
    if counters["ticks"] % 10 == 0 and len(trk["log"]) < 120:
        trk["log"].append([counters["ticks"], n_seen, round(max_speed, 4)] +
                          [[round(t["c"][0], 3), round(t["c"][1], 3)] for t in tracks if t["target"]])


def tick(target_q, grip):
    if counters["terminal"]:
        return True
    st = get_robot_state()
    reply = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=[float(v) for v in target_q],
        gripper_position=float(grip), embodiment=st.embodiment)}))
    counters["ticks"] += 1
    if is_terminal(reply):
        counters["terminal"] = True
    obs = reply_obs(reply)
    counters["obs"] = obs
    update_tracks(obs)
    return counters["terminal"]


DQ_MAX = 0.04


def servo(goal_q, grip, settle_max, tol):
    q0 = current_q()
    dq = max(abs(goal_q[j] - q0[j]) for j in range(len(q0)))
    n = max(2, int(math.ceil(dq / DQ_MAX)))
    for i in range(1, n + 1):
        tgt = [q0[j] + (goal_q[j] - q0[j]) * i / float(n) for j in range(len(q0))]
        if tick(tgt, grip):
            return False
    for k in range(settle_max):
        q = current_q()
        err = max(abs(goal_q[j] - q[j]) for j in range(len(q)))
        if err < tol:
            break
        if tick(goal_q, grip):
            return False
    return True


def ik_joints(position, quat, seed):
    res = solve_ik(Pose([float(v) for v in position], [float(v) for v in quat], BASE),
                   seed_joints=seed)
    if not res.ok:
        return None
    try:
        return floats(res.joint_positions)
    except Exception:
        pass
    try:
        return floats(res.solution)
    except Exception:
        return None


# ----------------------------------------------------------------------------- orientation
R_home = quat_to_mat(HOME_QUAT)
z_axis = [R_home[0][2], R_home[1][2], R_home[2][2]]
if z_axis[2] < -0.9:
    DOWN_QUAT = tuple(HOME_QUAT)
else:
    DOWN_QUAT = (0.0, 1.0, 0.0, 0.0)
    note("home EE not top-down; using canonical top-down quaternion")
R_down = quat_to_mat(DOWN_QUAT)
FINGER_YAW0 = math.atan2(R_down[1][1], R_down[0][1])   # hand y axis = finger closing axis


def yawed_quat(yaw):
    q = quat_mul((math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)), DOWN_QUAT)
    return [q[0], q[1], q[2], q[3]]


# ----------------------------------------------------------------------------- conditioning video
try:
    video = robomme_plus.get_conditioning_video()
    n_video = len(video)
except Exception as exc:
    video = ()
    n_video = 0
    note("conditioning video error " + str(exc))
report["conditioning_video_frames"] = n_video

# ----------------------------------------------------------------------------- phase 1: landing
report["phase"] = "landing"
bright0 = len(frame_points(obs0, "bright", None))
init_done = False
prev_count = None
stable = 0
hold_grip = 1.0
try:
    hold_grip = float(state0.gripper_positions["primary"])
except Exception:
    hold_grip = 1.0
for hold in range(40):
    if tick(HOME_Q, hold_grip):
        break
    pts = frame_points(counters["obs"], "bright", None)
    cnt = len(pts)
    if cnt > 3 * bright0 + 30 and prev_count is not None and abs(cnt - prev_count) < 0.15 * cnt:
        stable += 1
    else:
        stable = 0
    prev_count = cnt
    if stable >= 2:
        # container-top band = modal elevation of bright elevated points
        hist = {}
        for p in pts:
            key = int(p[2] / 0.005)
            hist[key] = hist.get(key, 0) + 1
        mode_key = max(hist, key=lambda k: hist[k])
        top_e = (mode_key + 0.5) * 0.005
        BAND["lo"] = max(0.015, top_e - 0.012)
        BAND["hi"] = top_e + 0.012
        band_pts = [p for p in pts if BAND["lo"] <= p[2] <= BAND["hi"]]
        if len(band_pts) < 10:
            continue
        # Containers are dropped onto the cubes: seed one track per tick-0 cube blob, refine by Lloyd.
        centers = [[cc[0], cc[1]] for cc in container_seeds]
        cap2 = 0.06 * 0.06
        members = [[] for cc in centers]
        for it in range(8):
            if it == 4:
                # second pass: shrink the capture radius to the measured container size
                cp = [len(set((int(math.floor(p[0] / 0.01)), int(math.floor(p[1] / 0.01))) for p in m))
                      for m in members if len(m) >= 4]
                if len(cp) > 0:
                    cap = max(0.02, min(0.06, 0.8 * math.sqrt(median(cp)) * 0.01))
                    cap2 = cap * cap
                centers = [[cc[0], cc[1]] for cc in container_seeds]
            members = [[] for cc in centers]
            for p in band_pts:
                best = -1
                bd = cap2
                for k in range(len(centers)):
                    dd = (p[0] - centers[k][0]) ** 2 + (p[1] - centers[k][1]) ** 2
                    if dd < bd:
                        bd = dd
                        best = k
                if best >= 0:
                    members[best].append(p)
            for k in range(len(centers)):
                if len(members[k]) >= 4:
                    centers[k] = [sum(p[0] for p in members[k]) / len(members[k]),
                                  sum(p[1] for p in members[k]) / len(members[k])]
        cells_per = [len(set((int(math.floor(p[0] / 0.01)), int(math.floor(p[1] / 0.01))) for p in m))
                     for m in members if len(m) >= 4]
        med_cells = median(cells_per) if len(cells_per) > 0 else 16
        side = math.sqrt(med_cells) * 0.01
        # band blobs not explained by any seed (e.g. container over a white/grey cube) -> extra tracks
        leftover = []
        for p in band_pts:
            near = False
            for cc in centers:
                if (p[0] - cc[0]) ** 2 + (p[1] - cc[1]) ** 2 < (0.75 * side + 0.01) ** 2:
                    near = True
                    break
            if not near:
                leftover.append(p)
        n_extra = 0
        for c in grid_components(leftover, 0.01):
            if c["cells"] < 0.5 * med_cells:
                continue
            k = max(1, int(round(c["cells"] / float(med_cells))))
            for cc in (kmeans_split(c["points"], k) if k >= 2 else [[c["cx"], c["cy"]]]):
                centers.append(cc)
                n_extra += 1
        report["init_extra_tracks"] = n_extra
        report["init_seed_support"] = [len(m) for m in members]
        # drop unsupported non-goal seeds (spurious tick-0 blobs) so they cannot steal points later
        goal_xy = [cube_seeds[c] for c in cube_seeds]
        kept = []
        for k in range(len(centers)):
            if k >= len(members) or len(members[k]) >= 4:
                kept.append(centers[k])
                continue
            for gx, gy in goal_xy:
                if (centers[k][0] - gx) ** 2 + (centers[k][1] - gy) ** 2 < 0.01 ** 2:
                    kept.append(centers[k])
                    break
        centers = kept
        trk["Ra"] = max(0.03, min(0.05, 0.75 * side + 0.008))
        trk["side"] = side
        trk["cells"] = med_cells
        EE_EXCL_R = max(0.05, side + 0.02)
        for i, cc in enumerate(centers):
            tracks.append({"id": i, "c": [cc[0], cc[1]], "v": [0.0, 0.0], "seen": 1, "missed": 0,
                           "target": None})
        init_done = True
        report["init"] = {"tick": counters["ticks"], "band": [round(BAND["lo"], 3), round(BAND["hi"], 3)],
                          "n_tracks": len(tracks), "n_cube_blobs": len(cube_comps),
                          "side_m": round(side, 3), "Ra": round(trk["Ra"], 3)}
        break
if not init_done:
    note("container initialisation failed")

# bind goal colours to tracks (greedy, exclusive, nearest first)
pairs = []
for colour in cube_seeds:
    sx, sy = cube_seeds[colour]
    for t in tracks:
        pairs.append(((t["c"][0] - sx) ** 2 + (t["c"][1] - sy) ** 2, colour, t["id"]))
pairs = sorted(pairs)
bound = {}
used = set()
for dd, colour, tid in pairs:
    if colour in bound or tid in used:
        continue
    bound[colour] = tid
    used.add(tid)
    tracks[tid]["target"] = colour
report["binding"] = {k: [bound[k], round(math.sqrt(min(p[0] for p in pairs if p[1] == k)), 3)]
                     for k in bound}

# Buttons: tick-0 candidates not covered by a container now.
buttons = []
for b in button_cands:
    covered = False
    for t in tracks:
        if (t["c"][0] - b["x"]) ** 2 + (t["c"][1] - b["y"]) ** 2 < 0.035 ** 2:
            covered = True
    if not covered:
        buttons.append(b)
buttons = sorted(buttons, key=lambda b: b["x"] * b["x"] + b["y"] * b["y"])
if len(buttons) < 2 and plan["press"]:
    note("geometric button detection found " + str(len(buttons)) + "; trying localize_object")
    for prompt in ("button", "white button", "round button", "push button"):
        if len(buttons) >= 2:
            break
        for camera in ("wrist", "front"):
            if camera not in CAMS or len(buttons) >= 2:
                continue
            try:
                seg = segment_text(camera, prompt)
                if not seg.ok:
                    continue
                for cand in seg.segmentations[:4]:
                    pc = mask_to_point_cloud(cand, camera, target_frame=BASE)
                    pts = to_list(pc.points)
                    if len(pts) < 10:
                        continue
                    cx = median([float(p[0]) for p in pts])
                    cy = median([float(p[1]) for p in pts])
                    top = quantile([float(p[2]) for p in pts], 0.95) - TABLE_Z
                    if top > 0.08:
                        continue
                    dup = False
                    for b in buttons:
                        if (b["x"] - cx) ** 2 + (b["y"] - cy) ** 2 < 0.03 ** 2:
                            dup = True
                    if not dup:
                        buttons.append({"x": cx, "y": cy, "top": top, "base": top - 0.01, "n": len(pts),
                                        "span": 0.0})
            except Exception as exc:
                note("segment error " + str(exc))
if plan["all_buttons"]:
    buttons = buttons[:2]
else:
    buttons = buttons[:1]
report["buttons"] = [[round(b["x"], 3), round(b["y"], 3), round(b["top"], 3)] for b in buttons]


# ----------------------------------------------------------------------------- phase 2: presses
def press_button(b):
    top_z = TABLE_Z + b["top"]
    press_z = max(TABLE_Z + b["base"] + 0.003, top_z - 0.012)
    seq = [(b["x"], b["y"], top_z + 0.06), (b["x"], b["y"], top_z + 0.02), (b["x"], b["y"], press_z)]
    seed = current_q()
    qs = []
    for pos in seq:
        q = ik_joints(pos, DOWN_QUAT, seed)
        if q is None:
            note("press IK failed at " + str([round(v, 3) for v in pos]))
            return False
        qs.append(q)
        seed = q
    for q in qs:
        if not servo(q, 0.0, 12, 0.01):
            return False
    for hold in range(4):
        if tick(qs[-1], 0.0):
            return False
    if not servo(qs[1], 0.0, 4, 0.02):
        return False
    return servo(qs[0], 0.0, 4, 0.03)


report["phase"] = "press"
press_log = []
if plan["press"] and init_done:
    for rep in range(plan["times"]):
        for b in buttons:
            ok = press_button(b)
            press_log.append({"button": [round(b["x"], 3), round(b["y"], 3)], "ok": ok,
                              "tick": counters["ticks"], "elapsed": elapsed_ticks()})
            if counters["terminal"]:
                break
report["presses"] = press_log

# ----------------------------------------------------------------------------- phase 3: settle
report["phase"] = "track_until_still"
if not counters["terminal"]:
    servo(HOME_Q, 1.0, 10, 0.02)
STILL_TICKS = 40
wait = 0
while not counters["terminal"] and trk["still"] < STILL_TICKS and wait < 520:
    if ticks_left() < 150 * max(1, len(plan["order"])) + 50:
        note("stopping wait early to keep pick budget")
        break
    tick(HOME_Q, 1.0)
    wait += 1
report["settle"] = {"wait_ticks": wait, "still": trk["still"], "tick": counters["ticks"]}


# ----------------------------------------------------------------------------- phase 4: picks
def refine_target(tid):
    """Re-measure the target container from a fresh observation (no ticks)."""
    obs = get_observation()
    pts = frame_points(obs, "band", ee_position(obs))
    t = tracks[tid]
    ra2 = trk["Ra"] * trk["Ra"]
    mine = []
    for p in pts:
        best = -1
        bd = ra2
        for k in range(len(tracks)):
            dd = (p[0] - tracks[k]["c"][0]) ** 2 + (p[1] - tracks[k]["c"][1]) ** 2
            if dd < bd:
                bd = dd
                best = k
        if best == tid:
            mine.append(p)
    if len(mine) >= 6:
        cx = median([p[0] for p in mine])
        cy = median([p[1] for p in mine])
        if (cx - t["c"][0]) ** 2 + (cy - t["c"][1]) ** 2 < 0.03 ** 2:
            t["c"] = [cx, cy]
    return mine


def face_angle(points, cx, cy):
    """Square-top orientation (mod 90 deg) minimising the bounding-box area of the top points."""
    if len(points) < 8:
        return None
    best = None
    for k in range(18):
        a = k * (math.pi / 2) / 18.0
        ca = math.cos(a)
        sa = math.sin(a)
        us = [(p[0] - cx) * ca + (p[1] - cy) * sa for p in points]
        vs = [-(p[0] - cx) * sa + (p[1] - cy) * ca for p in points]
        area = (quantile(us, 0.95) - quantile(us, 0.05)) * (quantile(vs, 0.95) - quantile(vs, 0.05))
        if best is None or area < best[0]:
            best = (area, a)
    return best[1]


def choose_yaw(tid, angle):
    t = tracks[tid]
    nearest = None
    for k in range(len(tracks)):
        if k == tid:
            continue
        dd = (tracks[k]["c"][0] - t["c"][0]) ** 2 + (tracks[k]["c"][1] - t["c"][1]) ** 2
        if nearest is None or dd < nearest[0]:
            nearest = (dd, k)
    if angle is None:
        angle = 0.0
    options = []
    for k in range(4):
        finger_dir = angle + k * math.pi / 2
        yaw = wrap_angle(finger_dir - FINGER_YAW0)
        if abs(yaw) > math.pi / 2 + 1e-6:
            continue
        clear = 0.0
        if nearest is not None:
            nb = tracks[nearest[1]]
            ndir = math.atan2(nb["c"][1] - t["c"][1], nb["c"][0] - t["c"][0])
            clear = abs(math.sin(ndir - finger_dir))   # 1 = neighbour off the finger axis
        options.append((-(clear), abs(yaw), yaw))
    if len(options) == 0:
        return 0.0
    return sorted(options)[0][2]


def gripper_now():
    try:
        return float(get_robot_state().gripper_positions["primary"])
    except Exception:
        return None


def exec_ok(res, label):
    ok = False
    try:
        ok = bool(res.ok)
    except Exception:
        ok = False
    if is_terminal(res):
        counters["terminal"] = True
    if not ok:
        status = "?"
        try:
            status = str(res.status)
        except Exception:
            status = "?"
        note(label + " not ok: " + status)
    return ok


def pick_container(colour, tid):
    entry = {"colour": colour, "track": tid}
    mine = refine_target(tid)
    t = tracks[tid]
    cx, cy = t["c"]
    top_e = median([p[2] for p in mine]) if len(mine) >= 6 else 0.5 * (BAND["lo"] + BAND["hi"])
    angle = face_angle(mine, cx, cy)
    yaw = choose_yaw(tid, angle)
    quat = yawed_quat(yaw)
    top_z = TABLE_Z + top_e
    grasp_z = max(TABLE_Z + 0.012, top_z - min(0.035, 0.55 * top_e))
    entry["xy"] = [round(cx, 3), round(cy, 3)]
    entry["top_e"] = round(top_e, 3)
    entry["yaw_deg"] = round(yaw * 180 / math.pi, 1)
    entry["n_pts"] = len(mine)
    open_gripper()
    pre = Pose([cx, cy, top_z + 0.10], quat, BASE)
    if not exec_ok(move_to_pose(pre, tolerance=0.01, max_steps=300), "pre-grasp"):
        entry["result"] = "pre-grasp failed"
        return entry
    if counters["terminal"]:
        return entry
    grasp = Pose([cx, cy, grasp_z], quat, BASE)
    exec_ok(move_to_pose(grasp, tolerance=0.008, max_steps=200), "grasp descent")
    close_gripper()
    width = gripper_now()
    entry["grip_after_close"] = width
    lift = Pose([cx, cy, grasp_z + 0.12], quat, BASE)
    lift_res = move_to_pose(lift, tolerance=0.015, max_steps=200)
    exec_ok(lift_res, "lift")
    if counters["terminal"]:
        entry["result"] = "terminal after lift"
        return entry
    width = gripper_now()
    entry["grip_after_lift"] = width
    held = width is not None and width > 0.04
    # put it back where it was, release, rise
    exec_ok(move_to_pose(Pose([cx, cy, grasp_z + 0.006], quat, BASE), tolerance=0.01, max_steps=200),
            "lower back")
    open_gripper()
    exec_ok(move_to_pose(Pose([cx, cy, top_z + 0.10], quat, BASE), tolerance=0.015, max_steps=200),
            "retreat")
    entry["result"] = "lifted" if held else "grasp not confirmed"
    return entry


report["phase"] = "pick"
picks = []
for colour in plan["order"]:
    if counters["terminal"]:
        break
    if colour not in bound:
        # fall back: the unbound track nearest the colour's cube seed, else skip
        note("no track bound for " + colour)
        if colour not in cube_seeds:
            continue
        free = [t for t in tracks if t["id"] not in used]
        if len(free) == 0:
            continue
        sx, sy = cube_seeds[colour]
        pick_t = sorted(free, key=lambda t: (t["c"][0] - sx) ** 2 + (t["c"][1] - sy) ** 2)[0]
        bound[colour] = pick_t["id"]
        used.add(pick_t["id"])
    tid = bound[colour]
    if not counters["terminal"]:
        servo(HOME_Q, 1.0, 10, 0.02)
    entry = pick_container(colour, tid)
    if entry.get("result") == "grasp not confirmed" and not counters["terminal"]:
        servo(HOME_Q, 1.0, 10, 0.02)
        entry["retry"] = pick_container(colour, tid)
    picks.append(entry)
report["picks"] = picks
report["tracker_log"] = trk["log"]
report["reacquisitions"] = trk["reacq"]
report["ticks"] = counters["ticks"]
report["terminal"] = counters["terminal"]
report["phase"] = "done"
result = report
