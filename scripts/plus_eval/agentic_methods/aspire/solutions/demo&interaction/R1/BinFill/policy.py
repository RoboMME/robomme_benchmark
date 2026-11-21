# BinFill v00 -- initial program from offline scene analysis (released recordings only).
#
# Task family (from the three released recordings): coloured cubes appear on the table over
# time -- in two of three recordings they ride a conveyor belt (spawned at one end, carried
# along), in the third they appear on a static table. The goal text names counts per colour
# ("put three red cubes and three blue cubes into the bin, then press the button to stop").
#
# Strategy (everything task-dependent is perceived at runtime):
#   1. Parse required colour counts from get_task_context().language.
#   2. Calibrate pixel->robot_base projection from the public intrinsics + camera_pose
#      (camera-axis convention decided from the pose itself, cross-checked against
#      mask_to_point_cloud on a SAM mask).
#   3. Localize bin and button (SAM3 text prompts, validated by colour / geometry).
#   4. Loop: detect coloured cube blobs by hue in the front RGB, project with depth, estimate the
#      common belt velocity by matching blobs between observations (0 on a static table),
#      choose a reachable cube of a still-needed colour, hover above its predicted position,
#      refine with the wrist camera, then run a per-tick tracking descend/close/lift trajectory
#      (IK per tick, step()) that moves with the cube; verify the grip; drop into the bin.
#   5. When all counts are delivered (or the step budget is nearly spent) press the button.
# Heavily instrumented: this single run is also the scene probe.

import math

report = {"version": "v00_initial", "phases": [], "log": [], "errors": []}
LOG_LIMIT = 900
KIND_LIMIT = 160
log_counts = {}


def note(kind, data):
    log_counts[kind] = log_counts.get(kind, 0) + 1
    if len(report["log"]) < LOG_LIMIT and log_counts[kind] <= KIND_LIMIT:
        entry = {"k": kind}
        for key in data:
            entry[key] = data[key]
        report["log"].append(entry)


def rnd(value, digits=4):
    try:
        return round(float(value), digits)
    except Exception:
        return None


def rlist(values, digits=4):
    out = []
    for value in values:
        out.append(rnd(value, digits))
    return out


def to_list(array):
    try:
        return array.tolist()
    except Exception:
        rows = []
        for row in array:
            rows.append([value for value in row])
        return rows


# ---------------------------------------------------------------- quaternion helpers
def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def quat_norm(q):
    n = math.sqrt(sum(float(v) * float(v) for v in q))
    if n < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(float(v) / n for v in q)


def quat_to_rot(q):
    w, x, y, z = quat_norm(q)
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def rot_apply(rot, vec):
    return [rot[0][0] * vec[0] + rot[0][1] * vec[1] + rot[0][2] * vec[2],
            rot[1][0] * vec[0] + rot[1][1] * vec[1] + rot[1][2] * vec[2],
            rot[2][0] * vec[0] + rot[2][1] * vec[1] + rot[2][2] * vec[2]]


def yaw_quat(theta):
    return (math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0))


def wrap_half_turn(angle):
    # finger axis is symmetric under 180 deg: map to (-90, 90]
    while angle > math.pi / 2:
        angle -= math.pi
    while angle <= -math.pi / 2:
        angle += math.pi
    return angle


def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def median(values):
    return quantile(values, 0.5)


# ---------------------------------------------------------------- result/flag helpers
def res_ok(res):
    try:
        return bool(res.ok)
    except Exception:
        return False


def res_steps(res):
    try:
        return int(res.steps_executed)
    except Exception:
        return None


def res_status(res):
    try:
        return str(res.status)
    except Exception:
        return "?"


def res_error(res):
    try:
        err = res.error
        if err is None:
            return None
        return str(err)[:300]
    except Exception:
        return None


def is_terminal(res):
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


episode = {"terminal": False, "terminal_where": None}


def mark_terminal(res, where):
    if is_terminal(res):
        episode["terminal"] = True
        episode["terminal_where"] = where
        note("terminal", {"where": where, "t": now_t()})
        return True
    return False


def now_t():
    try:
        return float(get_robot_state().timestamp_s)
    except Exception:
        return None


# ---------------------------------------------------------------- task context / metadata
context = get_task_context()
goal = str(context.language)
report["goal"] = goal
try:
    report["task_name"] = str(context.task_name)
except Exception:
    pass

task_meta = {}
try:
    task_meta = dict(robomme_plus.get_task_metadata())
except Exception as exc:
    report["errors"].append("task_metadata: " + str(exc)[:200])
meta_out = {}
for key in task_meta:
    try:
        meta_out[str(key)] = str(task_meta[key])[:300]
    except Exception:
        pass
report["task_metadata"] = meta_out
try:
    ctrl_meta = dict(robomme_plus.get_controller_metadata())
    ctrl_out = {}
    for key in ctrl_meta:
        ctrl_out[str(key)] = str(ctrl_meta[key])[:300]
    report["controller_metadata"] = ctrl_out
except Exception as exc:
    report["errors"].append("controller_metadata: " + str(exc)[:200])

CONTROL_HZ = 20.0
try:
    CONTROL_HZ = float(task_meta["control_hz"])
except Exception:
    try:
        CONTROL_HZ = float(task_meta["control_frequency"])
    except Exception:
        pass
DT = 1.0 / CONTROL_HZ
MAX_STEPS = None
try:
    MAX_STEPS = int(task_meta["max_steps"])
except Exception:
    pass
report["control_hz"] = CONTROL_HZ
report["max_steps"] = MAX_STEPS

try:
    video = robomme_plus.get_conditioning_video()
    report["conditioning_video_frames"] = len(video)
except Exception as exc:
    report["conditioning_video_frames"] = "error: " + str(exc)[:200]


def ticks_used():
    t = now_t()
    if t is None:
        return 0
    return int(round(t * CONTROL_HZ))


def ticks_left():
    if MAX_STEPS is None:
        return 100000
    return MAX_STEPS - ticks_used()


# ---------------------------------------------------------------- goal parsing
NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "single": 1}
COLOR_WORDS = {"red": "red", "green": "green", "blue": "blue", "yellow": "yellow",
               "orange": "orange", "cyan": "cyan", "magenta": "magenta", "purple": "magenta",
               "pink": "magenta", "violet": "magenta"}


def parse_counts(text):
    cleaned = text.lower()
    for ch in ",.;:!":
        cleaned = cleaned.replace(ch, " ")
    words = [w for w in cleaned.split(" ") if w]
    counts = {}
    index = 0
    while index < len(words):
        word = words[index]
        number = None
        if word in NUMBER_WORDS:
            number = NUMBER_WORDS[word]
        else:
            try:
                number = int(word)
            except Exception:
                number = None
        if number is not None and index + 1 < len(words) and words[index + 1] in COLOR_WORDS:
            color = COLOR_WORDS[words[index + 1]]
            counts[color] = counts.get(color, 0) + number
            index += 2
            continue
        index += 1
    return counts


required = parse_counts(goal)
report["required_counts"] = dict(required)
delivered = {}
for color_name in required:
    delivered[color_name] = 0


def still_needed():
    out = []
    for color_name in required:
        if delivered[color_name] < required[color_name]:
            out.append(color_name)
    return out


# ---------------------------------------------------------------- robot basics
state0 = get_robot_state()
BASE = state0.base_frame
EMB = state0.embodiment
report["base_frame"] = str(BASE)
report["embodiment"] = str(EMB)
home_joints = [float(v) for v in state0.joint_positions["primary"]]
home_pose = state0.end_effector_poses["primary"]
report["home_joints"] = rlist(home_joints)
report["home_ee"] = {"p": rlist(home_pose.position), "q": rlist(home_pose.quaternion_wxyz)}
report["home_gripper"] = rnd(state0.gripper_positions["primary"])

home_q = quat_norm([float(v) for v in home_pose.quaternion_wxyz])
home_rot = quat_to_rot(home_q)
home_z_axis = [home_rot[0][2], home_rot[1][2], home_rot[2][2]]
if home_z_axis[2] < -0.95:
    BASE_Q = home_q
else:
    BASE_Q = (0.0, 1.0, 0.0, 0.0)  # tool z pointing down
report["base_q"] = rlist(BASE_Q)
report["home_tool_z_axis"] = rlist(home_z_axis)


def finger_angle(q):
    rot = quat_to_rot(q)
    return math.atan2(rot[1][1], rot[0][1])


def grasp_quat(direction_xy):
    """Top-down quaternion whose finger (closing) axis is along direction_xy (or base yaw)."""
    if direction_xy is None:
        return BASE_Q
    desired = math.atan2(direction_xy[1], direction_xy[0])
    delta = wrap_half_turn(desired - finger_angle(BASE_Q))
    return quat_norm(quat_mul(yaw_quat(delta), BASE_Q))


def current_joints():
    return [float(v) for v in get_robot_state().joint_positions["primary"]]


def current_ee():
    return [float(v) for v in get_robot_state().end_effector_poses["primary"].position]


def gripper_reading():
    try:
        return float(get_robot_state().gripper_positions["primary"])
    except Exception:
        return None


step_info = {"first_step_repr": None, "steps": 0}


def hold(ticks, gripper):
    """Hold the measured joints for `ticks` control ticks (advances time)."""
    joints = current_joints()
    for tick in range(ticks):
        res = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=joints, gripper_position=gripper,
            embodiment=EMB)}))
        step_info["steps"] += 1
        if step_info["first_step_repr"] is None:
            step_info["first_step_repr"] = str(res)[:600]
        if mark_terminal(res, "hold"):
            return False
    return True


def ik_joints(ik):
    try:
        return [float(v) for v in ik.joint_positions]
    except Exception:
        pass
    try:
        return [float(v) for v in ik.joints]
    except Exception:
        pass
    try:
        return [float(v) for v in ik.solution]
    except Exception:
        return None


def solve(position, quat, seed):
    pose = Pose([float(position[0]), float(position[1]), float(position[2])], list(quat), BASE)
    try:
        ik = solve_ik(pose, seed_joints=seed)
    except Exception as exc:
        note("ik_exc", {"e": str(exc)[:200]})
        return None
    if not res_ok(ik):
        return None
    return ik_joints(ik)


def move_pose(position, quat, label, max_steps=300, tolerance=0.01):
    pose = Pose([float(position[0]), float(position[1]), float(position[2])], list(quat), BASE)
    t0 = now_t()
    try:
        res = move_to_pose(pose, tolerance=tolerance, max_steps=max_steps)
    except Exception as exc:
        note("move_exc", {"label": label, "e": str(exc)[:200]})
        return False, 0
    t1 = now_t()
    ticks = 0
    if t0 is not None and t1 is not None:
        ticks = int(round((t1 - t0) * CONTROL_HZ))
    ee = current_ee()
    note("move", {"label": label, "ok": res_ok(res), "status": res_status(res),
                  "steps": res_steps(res), "ticks": ticks, "target": rlist(position),
                  "ee": rlist(ee), "err": res_error(res)})
    mark_terminal(res, "move:" + label)
    return res_ok(res), ticks


# ---------------------------------------------------------------- camera geometry
CONV = {"forced": None}

def cam_geom(cam):
    k = to_list(cam.intrinsics)
    pose = cam.camera_pose
    rot = quat_to_rot([float(v) for v in pose.quaternion_wxyz])
    pos = [float(v) for v in pose.position]
    # optical axis sign decides the convention: OpenCV looks along +z, OpenGL along -z.
    conv = "opencv"
    if rot[2][2] > 0:
        conv = "opengl"
    if CONV["forced"] is not None:
        conv = CONV["forced"]
    return {"fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]),
            "cy": float(k[1][2]), "rot": rot, "pos": pos, "conv": conv,
            "frame": str(pose.frame)}


def project(geom, u, v, depth):
    x = (u - geom["cx"]) * depth / geom["fx"]
    y = (v - geom["cy"]) * depth / geom["fy"]
    if geom["conv"] == "opencv":
        local = [x, y, depth]
    elif geom["conv"] == "opengl":
        local = [x, -y, -depth]
    else:  # "sapien": x forward, y left, z up
        local = [depth, -x, -y]
    world = rot_apply(geom["rot"], local)
    return [world[0] + geom["pos"][0], world[1] + geom["pos"][1], world[2] + geom["pos"][2]]


# ---------------------------------------------------------------- colour classification
def classify(r, g, b):
    mx = max(r, g, b)
    if mx < 35:
        return None
    mn = min(r, g, b)
    span = mx - mn
    if span < 0.45 * mx:
        return None
    if mx == r:
        hue = 60.0 * ((g - b) / span)
    elif mx == g:
        hue = 60.0 * ((b - r) / span + 2.0)
    else:
        hue = 60.0 * ((r - g) / span + 4.0)
    if hue < 0:
        hue += 360.0
    if hue >= 340.0 or hue < 8.0:
        return "red"
    if 28.0 <= hue < 50.0:
        return "orange"
    if 50.0 <= hue < 72.0:
        return "yellow"
    if 85.0 <= hue < 155.0:
        return "green"
    if 165.0 <= hue < 200.0:
        return "cyan"
    if 210.0 <= hue < 262.0:
        return "blue"
    if 275.0 <= hue < 335.0:
        return "magenta"
    return None


def detect_blobs(cam, stride, min_cells, max_cells, colors=None):
    """Connected components of hue-classified pixels; returns blobs with 3D summaries."""
    geom = cam_geom(cam)
    rgb = to_list(cam.rgb)
    dep = to_list(cam.depth_m)
    height = len(rgb)
    width = len(rgb[0])
    labels = {}
    for row in range(0, height, stride):
        line = rgb[row]
        for col in range(0, width, stride):
            px = line[col]
            lab = classify(int(px[0]), int(px[1]), int(px[2]))
            if lab is not None and (colors is None or lab in colors):
                labels[(row, col)] = lab
    seen = set()
    blobs = []
    for key in labels:
        if key in seen:
            continue
        lab = labels[key]
        stack = [key]
        seen.add(key)
        cells = []
        while stack:
            cur = stack.pop()
            cells.append(cur)
            for dr, dc in ((stride, 0), (-stride, 0), (0, stride), (0, -stride)):
                nb = (cur[0] + dr, cur[1] + dc)
                if nb in labels and nb not in seen and labels[nb] == lab:
                    seen.add(nb)
                    stack.append(nb)
        if len(cells) < min_cells or len(cells) > max_cells:
            continue
        rows = [c[0] for c in cells]
        cols = [c[1] for c in cells]
        r0 = max(0, min(rows) - stride)
        r1 = min(height - 1, max(rows) + stride)
        c0 = max(0, min(cols) - stride)
        c1 = min(width - 1, max(cols) + stride)
        pts = []
        pix_r = 0.0
        pix_c = 0.0
        npx = 0
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                px = rgb[row][col]
                if classify(int(px[0]), int(px[1]), int(px[2])) != lab:
                    continue
                npx += 1
                pix_r += row
                pix_c += col
                d = float(dep[row][col])
                if d > 0.03 and d < 5.0:
                    pts.append(project(geom, col, row, d))
        if len(pts) < 4:
            continue
        zs = [p[2] for p in pts]
        top_z = quantile(zs, 0.85)
        top = [p for p in pts if p[2] >= top_z - 0.008]
        if len(top) < 2:
            top = pts
        cx = sum(p[0] for p in top) / len(top)
        cy = sum(p[1] for p in top) / len(top)
        ring = []
        pad = 4
        for row in range(max(0, r0 - pad), min(height, r1 + pad + 1)):
            for col in (max(0, c0 - pad), min(width - 1, c1 + pad)):
                px = rgb[row][col]
                if classify(int(px[0]), int(px[1]), int(px[2])) is None:
                    d = float(dep[row][col])
                    if d > 0.03 and d < 5.0:
                        ring.append(project(geom, col, row, d)[2])
        for col in range(max(0, c0 - pad), min(width, c1 + pad + 1)):
            for row in (max(0, r0 - pad), min(height - 1, r1 + pad)):
                px = rgb[row][col]
                if classify(int(px[0]), int(px[1]), int(px[2])) is None:
                    d = float(dep[row][col])
                    if d > 0.03 and d < 5.0:
                        ring.append(project(geom, col, row, d)[2])
        surface = quantile(ring, 0.7) if ring else None
        blobs.append({"color": lab, "px": [pix_c / max(1, npx), pix_r / max(1, npx)],
                      "npx": npx, "box": [c0, r0, c1, r1], "xy": [cx, cy], "top": top_z,
                      "zmin": min(zs), "surface": surface, "npts": len(pts)})
    return blobs


def gray_points(cam, stride):
    """3D points of low-saturation mid-grey pixels (the conveyor belt surface, if any)."""
    geom = cam_geom(cam)
    rgb = to_list(cam.rgb)
    dep = to_list(cam.depth_m)
    pts = []
    for row in range(0, len(rgb), stride):
        for col in range(0, len(rgb[0]), stride):
            px = rgb[row][col]
            r = int(px[0])
            g = int(px[1])
            b = int(px[2])
            mx = max(r, g, b)
            mn = min(r, g, b)
            if mx < 50 or mx > 150 or (mx - mn) > 0.18 * mx:
                continue
            d = float(dep[row][col])
            if d > 0.03 and d < 5.0:
                p = project(geom, col, row, d)
                if p[2] < 0.15 and math.sqrt(p[0] * p[0] + p[1] * p[1]) > 0.2:
                    pts.append(p)
    return pts


def dark_blob_center(cam, stride, max_value, max_z):
    """Largest near-black component (bin interior) -> 3D centroid and pixel box."""
    geom = cam_geom(cam)
    rgb = to_list(cam.rgb)
    dep = to_list(cam.depth_m)
    marks = set()
    for row in range(0, len(rgb), stride):
        for col in range(0, len(rgb[0]), stride):
            px = rgb[row][col]
            if max(int(px[0]), int(px[1]), int(px[2])) <= max_value:
                marks.add((row, col))
    seen = set()
    best = None
    pts = []
    for key in marks:
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        cells = []
        while stack:
            cur = stack.pop()
            cells.append(cur)
            for dr, dc in ((stride, 0), (-stride, 0), (0, stride), (0, -stride)):
                nb = (cur[0] + dr, cur[1] + dc)
                if nb in marks and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if len(cells) < 3:
            continue
        cpts = []
        for cell in cells:
            d = float(dep[cell[0]][cell[1]])
            if d > 0.03 and d < 5.0:
                cpts.append(project(geom, cell[1], cell[0], d))
        if not cpts or median([p[2] for p in cpts]) > max_z:
            continue
        if best is None or len(cells) > len(best):
            best = cells
            pts = cpts
    if best is None:
        return None
    rows = [c[0] for c in best]
    cols = [c[1] for c in best]
    return {"xy": [sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)],
            "zmed": median([p[2] for p in pts]), "cells": len(best),
            "box": [min(cols), min(rows), max(cols), max(rows)]}


def cloud_points(cloud):
    out = []
    try:
        for p in cloud.points:
            out.append([float(p[0]), float(p[1]), float(p[2])])
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- initial observation
obs0 = get_observation()
cam_names = [name for name in obs0.cameras]
report["cameras"] = cam_names
FRONT = "front" if "front" in cam_names else cam_names[0]
WRIST = "wrist" if "wrist" in cam_names else None
geoms = {}
for name in cam_names:
    try:
        g = cam_geom(obs0.cameras[name])
        geoms[name] = {"fx": rnd(g["fx"]), "fy": rnd(g["fy"]), "cx": rnd(g["cx"]),
                       "cy": rnd(g["cy"]), "pos": rlist(g["pos"]), "conv": g["conv"],
                       "frame": g["frame"],
                       "optical_axis_world": rlist([g["rot"][0][2], g["rot"][1][2],
                                                    g["rot"][2][2]]),
                       "shape": [len(to_list(obs0.cameras[name].depth_m)),
                                 len(to_list(obs0.cameras[name].depth_m)[0])]}
    except Exception as exc:
        geoms[name] = "error: " + str(exc)[:200]
report["camera_geometry"] = geoms
report["t0"] = rnd(obs0.timestamp_s)


def seg_candidates(camera, prompt):
    try:
        found = segment_text(camera, prompt)
    except Exception as exc:
        note("seg_exc", {"prompt": prompt, "e": str(exc)[:200]})
        return []
    if not res_ok(found):
        return []
    out = []
    for cand in found.segmentations[:5]:
        try:
            box = [float(v) for v in cand.box_xyxy]
        except Exception:
            box = None
        out.append({"cand": cand, "score": rnd(cand.score, 3), "box": box})
    return out


def box_dark_fraction(rgb_rows, box, max_value):
    if box is None:
        return 0.0
    c0 = max(0, int(box[0]))
    r0 = max(0, int(box[1]))
    c1 = min(len(rgb_rows[0]) - 1, int(box[2]))
    r1 = min(len(rgb_rows) - 1, int(box[3]))
    total = 0
    dark = 0
    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            px = rgb_rows[row][col]
            total += 1
            if max(int(px[0]), int(px[1]), int(px[2])) <= max_value:
                dark += 1
    if total == 0:
        return 0.0
    return dark / total


def top_band_center(points, band):
    zs = [p[2] for p in points]
    top = quantile(zs, 0.95)
    sel = [p for p in points if p[2] >= top - band]
    if not sel:
        return None
    return {"xy": [sum(p[0] for p in sel) / len(sel), sum(p[1] for p in sel) / len(sel)],
            "top": top, "n": len(points),
            "ext": [max(p[0] for p in points) - min(p[0] for p in points),
                    max(p[1] for p in points) - min(p[1] for p in points),
                    max(zs) - min(zs)]}


front_rgb0 = to_list(obs0.cameras[FRONT].rgb)
scene = {"bin": None, "button": None}


def calibrate_convention():
    """Choose the camera-axis convention whose projection of a SAM box matches the provider's
    own mask_to_point_cloud (public calibration only)."""
    cam = obs0.cameras[FRONT]
    dep = to_list(cam.depth_m)
    out = {"heuristic": cam_geom(cam)["conv"], "trials": []}
    for prompt in ("bin", "button"):
        cands = seg_candidates(FRONT, prompt)
        if not cands or cands[0]["box"] is None:
            continue
        pts = cloud_points(mask_to_point_cloud(cands[0]["cand"], FRONT, target_frame=BASE))
        if len(pts) < 5:
            continue
        ref = [median([p[0] for p in pts]), median([p[1] for p in pts]),
               median([p[2] for p in pts])]
        box = cands[0]["box"]
        c0 = int(box[0] + 0.25 * (box[2] - box[0]))
        c1 = int(box[2] - 0.25 * (box[2] - box[0]))
        r0 = int(box[1] + 0.25 * (box[3] - box[1]))
        r1 = int(box[3] - 0.25 * (box[3] - box[1]))
        errors = {}
        for conv in ("opencv", "opengl", "sapien"):
            CONV["forced"] = conv
            geom = cam_geom(cam)
            proj = []
            for row in range(max(0, r0), min(len(dep), r1 + 1)):
                for col in range(max(0, c0), min(len(dep[0]), c1 + 1)):
                    d = float(dep[row][col])
                    if d > 0.03 and d < 5.0:
                        proj.append(project(geom, col, row, d))
            if not proj:
                continue
            mine = [median([p[0] for p in proj]), median([p[1] for p in proj]),
                    median([p[2] for p in proj])]
            errors[conv] = math.sqrt(sum((mine[i] - ref[i]) ** 2 for i in range(3)))
        CONV["forced"] = None
        trial = {"prompt": prompt, "ref": rlist(ref), "errors": {}}
        for key in errors:
            trial["errors"][key] = rnd(errors[key])
        out["trials"].append(trial)
        if errors:
            best = min(errors, key=lambda key: errors[key])
            if errors[best] < 0.08:
                out["chosen"] = best
                CONV["forced"] = best
                return out
    out["chosen"] = "heuristic:" + out["heuristic"]
    return out


try:
    report["projection_calibration"] = calibrate_convention()
except Exception as exc:
    CONV["forced"] = None
    report["errors"].append("calibration: " + str(exc)[:200])

# --- bin: SAM prompt validated by near-black interior, fallback to dark blob
bin_dark = None
try:
    bin_dark = dark_blob_center(obs0.cameras[FRONT], 1, 28, 0.15)
except Exception as exc:
    report["errors"].append("dark_blob: " + str(exc)[:200])
report["bin_dark_blob"] = None if bin_dark is None else {
    "xy": rlist(bin_dark["xy"]), "zmed": rnd(bin_dark["zmed"]), "cells": bin_dark["cells"],
    "box": bin_dark["box"]}
bin_probe = []
for prompt in ("bin", "box", "square container"):
    cands = seg_candidates(FRONT, prompt)
    for c in cands:
        frac = box_dark_fraction(front_rgb0, c["box"], 28)
        entry = {"prompt": prompt, "score": c["score"], "box": c["box"], "dark": rnd(frac, 3)}
        if frac > 0.08 and scene["bin"] is None:
            try:
                cloud = mask_to_point_cloud(c["cand"], FRONT, target_frame=BASE)
                pts = cloud_points(cloud)
                center = top_band_center(pts, 0.012) if len(pts) >= 10 else None
                if center is not None and center["ext"][0] < 0.35 and center["ext"][1] < 0.35:
                    scene["bin"] = {"xy": center["xy"], "rim": center["top"], "src": prompt,
                                    "ext": center["ext"], "n": center["n"]}
                    entry["chosen"] = True
                    # projection cross-check: provider cloud vs our own projection of dark blob
            except Exception as exc:
                entry["e"] = str(exc)[:200]
        bin_probe.append(entry)
    if scene["bin"] is not None:
        break
report["bin_probe"] = bin_probe
if scene["bin"] is None:
    for prompt in ("bin", "box"):
        for c in seg_candidates(FRONT, prompt)[:2]:
            if scene["bin"] is not None:
                break
            try:
                pts = cloud_points(mask_to_point_cloud(c["cand"], FRONT, target_frame=BASE))
                center = top_band_center(pts, 0.012) if len(pts) >= 10 else None
                if center is not None and center["ext"][0] < 0.25 and center["ext"][1] < 0.25:
                    near_dark = True
                    if bin_dark is not None:
                        near_dark = math.sqrt((center["xy"][0] - bin_dark["xy"][0]) ** 2 +
                                              (center["xy"][1] - bin_dark["xy"][1]) ** 2) < 0.08
                    if near_dark:
                        scene["bin"] = {"xy": center["xy"], "rim": center["top"],
                                        "src": prompt + "_undarkened", "ext": center["ext"]}
            except Exception as exc:
                report["errors"].append("bin_fallback: " + str(exc)[:200])
if scene["bin"] is None and bin_dark is not None:
    scene["bin"] = {"xy": bin_dark["xy"], "rim": bin_dark["zmed"] + 0.04, "src": "dark_blob"}
report["bin"] = None if scene["bin"] is None else {
    "xy": rlist(scene["bin"]["xy"]), "rim": rnd(scene["bin"]["rim"]), "src": scene["bin"]["src"]}
if scene["bin"] is not None and bin_dark is not None:
    report["bin_sam_vs_dark_xy_m"] = rlist([scene["bin"]["xy"][0] - bin_dark["xy"][0],
                                            scene["bin"]["xy"][1] - bin_dark["xy"][1]])

# --- button: SAM prompt validated by small extent
button_probe = []
for prompt in ("button", "round button", "white button"):
    cands = seg_candidates(FRONT, prompt)
    for c in cands:
        entry = {"prompt": prompt, "score": c["score"], "box": c["box"]}
        if scene["button"] is None:
            try:
                cloud = mask_to_point_cloud(c["cand"], FRONT, target_frame=BASE)
                pts = cloud_points(cloud)
                center = top_band_center(pts, 0.006) if len(pts) >= 5 else None
                if center is not None:
                    entry["ext"] = rlist(center["ext"])
                    entry["xy"] = rlist(center["xy"])
                    entry["top"] = rnd(center["top"])
                    far_from_bin = True
                    if scene["bin"] is not None:
                        dx = center["xy"][0] - scene["bin"]["xy"][0]
                        dy = center["xy"][1] - scene["bin"]["xy"][1]
                        far_from_bin = math.sqrt(dx * dx + dy * dy) > 0.05
                    if center["ext"][0] < 0.15 and center["ext"][1] < 0.15 and far_from_bin:
                        scene["button"] = {"xy": center["xy"], "top": center["top"],
                                           "src": prompt}
                        entry["chosen"] = True
            except Exception as exc:
                entry["e"] = str(exc)[:200]
        button_probe.append(entry)
    if scene["button"] is not None:
        break
report["button_probe"] = button_probe
report["button"] = None if scene["button"] is None else {
    "xy": rlist(scene["button"]["xy"]), "top": rnd(scene["button"]["top"]),
    "src": scene["button"]["src"]}

# --- belt extent (grey surface points)
belt_pts = []
try:
    belt_pts = gray_points(obs0.cameras[FRONT], 4)
except Exception as exc:
    report["errors"].append("gray_points: " + str(exc)[:200])
report["belt_points"] = len(belt_pts)
if len(belt_pts) > 20:
    report["belt_bounds"] = {
        "x": [rnd(quantile([p[0] for p in belt_pts], 0.03)),
              rnd(quantile([p[0] for p in belt_pts], 0.97))],
        "y": [rnd(quantile([p[1] for p in belt_pts], 0.03)),
              rnd(quantile([p[1] for p in belt_pts], 0.97))],
        "z": rnd(median([p[2] for p in belt_pts]))}


# ---------------------------------------------------------------- velocity tracking
vel = {"v": [0.0, 0.0], "n": 0, "last": None, "last_t": None, "history": []}


def velocity_ready():
    return vel["n"] >= 2


def summarize_blobs(blobs):
    out = []
    for b in blobs[:12]:
        out.append({"c": b["color"], "xy": rlist(b["xy"], 3), "top": rnd(b["top"], 3),
                    "s": rnd(b["surface"], 3), "n": b["npx"]})
    return out


def grounded(blob):
    if blob["surface"] is None:
        return True
    return blob["top"] - blob["surface"] < 0.08


def update_velocity(blobs, t):
    if vel["last"] is not None and t is not None and vel["last_t"] is not None:
        dt = t - vel["last_t"]
        if dt >= 0.25:
            disp = []
            for b in blobs:
                if not grounded(b):
                    continue
                best = None
                best_d = None
                for a in vel["last"]:
                    if a["color"] != b["color"]:
                        continue
                    px = a["xy"][0] + vel["v"][0] * dt
                    py = a["xy"][1] + vel["v"][1] * dt
                    d = math.sqrt((b["xy"][0] - px) ** 2 + (b["xy"][1] - py) ** 2)
                    if best_d is None or d < best_d:
                        best_d = d
                        best = a
                gate = 0.02 + (0.12 * dt if vel["n"] == 0 else 0.04 * dt)
                if best is not None and best_d <= gate:
                    disp.append([(b["xy"][0] - best["xy"][0]) / dt,
                                 (b["xy"][1] - best["xy"][1]) / dt])
            if len(disp) >= 1:
                vx = median([d[0] for d in disp])
                vy = median([d[1] for d in disp])
                weight = min(0.7, dt / 1.5) if vel["n"] > 0 else 1.0
                vel["v"] = [(1 - weight) * vel["v"][0] + weight * vx,
                            (1 - weight) * vel["v"][1] + weight * vy]
                vel["n"] += 1
                if len(vel["history"]) < 60:
                    vel["history"].append({"t": rnd(t, 2), "dt": rnd(dt, 2), "pairs": len(disp),
                                           "meas": [rnd(vx), rnd(vy)],
                                           "v": rlist(vel["v"])})
            vel["last"] = [b for b in blobs if grounded(b)]
            vel["last_t"] = t
    else:
        vel["last"] = [b for b in blobs if grounded(b)]
        vel["last_t"] = t


def speed():
    return math.sqrt(vel["v"][0] ** 2 + vel["v"][1] ** 2)


def vdir():
    s = speed()
    if s < 0.006:
        return None
    return [vel["v"][0] / s, vel["v"][1] / s]


def observe_front(tag):
    obs = get_observation()
    t = float(obs.timestamp_s)
    try:
        blobs = detect_blobs(obs.cameras[FRONT], 2, 2, 400)
    except Exception as exc:
        report["errors"].append("detect_front: " + str(exc)[:200])
        blobs = []
    update_velocity(blobs, t)
    counts = {}
    for b in blobs:
        counts[b["color"]] = counts.get(b["color"], 0) + 1
    note("front", {"tag": tag, "t": rnd(t, 2), "counts": counts, "v": rlist(vel["v"]),
                   "blobs": summarize_blobs([b for b in blobs if b["color"] in required])})
    return blobs, t


# ---------------------------------------------------------------- projection cross-check
def projection_check(blobs):
    """Compare our own projection with mask_to_point_cloud on a SAM point-prompt mask."""
    out = []
    for b in blobs[:2]:
        entry = {"color": b["color"], "px": rlist(b["px"], 1), "ours_xy": rlist(b["xy"]),
                 "ours_top": rnd(b["top"])}
        for order in ("xy", "yx"):
            point = [int(round(b["px"][0])), int(round(b["px"][1]))] if order == "xy" else \
                [int(round(b["px"][1])), int(round(b["px"][0]))]
            try:
                seg = segment_points(FRONT, [point])
                if not res_ok(seg) or len(seg.segmentations) == 0:
                    entry[order] = "no_mask"
                    continue
                cand = seg.segmentations[0]
                box = [float(v) for v in cand.box_xyxy]
                inside = box[0] - 2 <= b["px"][0] <= box[2] + 2 and \
                    box[1] - 2 <= b["px"][1] <= box[3] + 2
                pts = cloud_points(mask_to_point_cloud(cand, FRONT, target_frame=BASE))
                if pts:
                    c = top_band_center(pts, 0.008)
                    entry[order] = {"box": rlist(box, 1), "inside": inside,
                                    "xy": rlist(c["xy"]), "top": rnd(c["top"]),
                                    "n": len(pts), "ext": rlist(c["ext"], 3)}
                if inside:
                    break
            except Exception as exc:
                entry[order] = "error: " + str(exc)[:200]
        out.append(entry)
    return out


# ---------------------------------------------------------------- picking
timing = {"hover_ticks": 40}
BLEND = 6
DESCEND = 14
CLOSE = 12
LIFT = 12
LAG = 1
REACH_MIN = 0.22
REACH_MAX = 0.80
HOVER_DZ = 0.10


def predict(blob_xy, t_ref, t):
    return [blob_xy[0] + vel["v"][0] * (t - t_ref), blob_xy[1] + vel["v"][1] * (t - t_ref)]


def belt_end_ok(xy):
    direction = vdir()
    if direction is None or len(belt_pts) < 20:
        return True
    proj = [p[0] * direction[0] + p[1] * direction[1] for p in belt_pts]
    s_end = quantile(proj, 0.97)
    return xy[0] * direction[0] + xy[1] * direction[1] < s_end - 0.04


def reachable(xy):
    r = math.sqrt(xy[0] * xy[0] + xy[1] * xy[1])
    return REACH_MIN <= r <= REACH_MAX


def choose_target(blobs, t):
    needed = still_needed()
    ee = current_ee()
    best = None
    best_score = None
    lead = (timing["hover_ticks"] + BLEND + DESCEND + CLOSE) * DT
    for b in blobs:
        if b["color"] not in needed or not grounded(b):
            continue
        p_grasp = predict(b["xy"], t, t + lead)
        if not reachable(p_grasp) or not belt_end_ok(p_grasp):
            continue
        if scene["bin"] is not None:
            dbx = p_grasp[0] - scene["bin"]["xy"][0]
            dby = p_grasp[1] - scene["bin"]["xy"][1]
            if math.sqrt(dbx * dbx + dby * dby) < 0.06:
                continue  # inside/at the bin: already delivered
        p_hover = predict(b["xy"], t, t + timing["hover_ticks"] * DT)
        score = math.sqrt((p_hover[0] - ee[0]) ** 2 + (p_hover[1] - ee[1]) ** 2)
        if best_score is None or score < best_score:
            best_score = score
            best = b
    return best


def wrist_refine(color, predicted_xy, expected_top):
    if WRIST is None:
        return None
    obs = get_observation()
    t = float(obs.timestamp_s)
    try:
        blobs = detect_blobs(obs.cameras[WRIST], 2, 3, 5000, colors=[color])
    except Exception as exc:
        report["errors"].append("detect_wrist: " + str(exc)[:200])
        return None
    best = None
    best_d = None
    for b in blobs:
        d = math.sqrt((b["xy"][0] - predicted_xy[0]) ** 2 + (b["xy"][1] - predicted_xy[1]) ** 2)
        if best_d is None or d < best_d:
            best_d = d
            best = b
    info = {"t": rnd(t, 2), "n": len(blobs), "pred": rlist(predicted_xy)}
    if best is not None:
        info["xy"] = rlist(best["xy"])
        info["top"] = rnd(best["top"])
        info["d"] = rnd(best_d)
        info["npx"] = best["npx"]
    ok = best is not None and best_d < 0.05 and abs(best["top"] - expected_top) < 0.03
    info["used"] = ok
    note("wrist_refine", info)
    if not ok:
        return None
    return {"xy": best["xy"], "top": best["top"], "t": t}


def pick_attempt(target, t_obs, attempt_no):
    color = target["color"]
    rec = {"attempt": attempt_no, "color": color, "obs_xy": rlist(target["xy"]),
           "top": rnd(target["top"]), "surface": rnd(target["surface"]), "v": rlist(vel["v"]),
           "t_obs": rnd(t_obs, 2)}
    surface = target["surface"] if target["surface"] is not None else target["top"] - 0.03
    belt = report.get("belt_bounds")
    if belt is not None and belt["x"][0] + 0.01 <= target["xy"][0] <= belt["x"][1] - 0.01 and \
            belt["y"][0] + 0.01 <= target["xy"][1] <= belt["y"][1] - 0.01 and \
            belt["z"] < target["top"] - 0.01:
        rec["surface_ring"] = rnd(surface)
        surface = belt["z"]
    height = max(0.015, min(0.06, target["top"] - surface))
    rec["height"] = rnd(height)
    grasp_z = min(target["top"] - 0.008, surface + max(0.010, 0.35 * height))
    hover_z = target["top"] + HOVER_DZ
    quat = grasp_quat(vdir())
    rec["quat"] = rlist(quat)
    t_now = now_t()
    lead_ticks = timing["hover_ticks"] + 8
    hover_xy = predict(target["xy"], t_obs, t_now + lead_ticks * DT)
    if vdir() is not None:
        hover_xy = [hover_xy[0] + 0.01 * vdir()[0], hover_xy[1] + 0.01 * vdir()[1]]
    rec["hover_xy"] = rlist(hover_xy)
    ok, ticks = move_pose([hover_xy[0], hover_xy[1], hover_z], quat, "hover" + str(attempt_no))
    rec["hover_ok"] = ok
    rec["hover_ticks"] = ticks
    ee_check = current_ee()
    rec["hover_miss"] = rnd(math.sqrt((ee_check[0] - hover_xy[0]) ** 2 +
                                      (ee_check[1] - hover_xy[1]) ** 2 +
                                      (ee_check[2] - hover_z) ** 2))
    if not ok and rec["hover_miss"] > 0.03:
        rec["abort"] = "hover not reached"
        report["attempts"].append(rec)
        return False
    if ticks > 5:
        timing["hover_ticks"] = int(0.5 * timing["hover_ticks"] + 0.5 * ticks)
    if episode["terminal"]:
        report["attempts"].append(rec)
        return False
    ref_xy = target["xy"]
    ref_t = t_obs
    ref_top = target["top"]
    t_now = now_t()
    pred = predict(ref_xy, ref_t, t_now)
    refined = wrist_refine(color, pred, ref_top)
    if refined is not None:
        ref_xy = refined["xy"]
        ref_t = refined["t"]
        rec["wrist_offset"] = rlist([refined["xy"][0] - pred[0], refined["xy"][1] - pred[1]])
    # wait until the cube arrives under the hover point (moving belt only)
    direction = vdir()
    ee = current_ee()
    if direction is not None:
        t_now = now_t()
        p_now = predict(ref_xy, ref_t, t_now)
        along = (p_now[0] - ee[0]) * direction[0] + (p_now[1] - ee[1]) * direction[1]
        rec["along_at_hover"] = rnd(along)
        if along > 0.06:
            rec["abort"] = "cube already past hover point"
            report["attempts"].append(rec)
            return False
        wait_ticks = 0
        if along < -0.005:
            wait_ticks = int((-along - 0.005) / (speed() * DT)) - BLEND
        wait_ticks = max(0, min(160, wait_ticks))
        rec["wait_ticks"] = wait_ticks
        if wait_ticks > 0:
            if not hold(wait_ticks, 1.0):
                report["attempts"].append(rec)
                return False
            if wait_ticks > 10:
                t_now = now_t()
                refined = wrist_refine(color, predict(ref_xy, ref_t, t_now), ref_top)
                if refined is not None:
                    ref_xy = refined["xy"]
                    ref_t = refined["t"]
    # build per-tick tracking trajectory
    t_start = now_t()
    ee = current_ee()
    catch_xy = predict(ref_xy, ref_t, t_start + (BLEND + LAG) * DT)
    catch_gap = math.sqrt((catch_xy[0] - ee[0]) ** 2 + (catch_xy[1] - ee[1]) ** 2)
    rec["catch_gap"] = rnd(catch_gap)
    if catch_gap > 0.05:
        rec["abort"] = "cube not under the gripper at tracking start"
        report["attempts"].append(rec)
        return False
    seed = current_joints()
    plan = []
    total = BLEND + DESCEND + CLOSE + LIFT
    for k in range(1, total + 1):
        t_k = t_start + (k + LAG) * DT
        cube_xy = predict(ref_xy, ref_t, t_k)
        if k <= BLEND:
            a = k / float(BLEND)
            x = (1 - a) * ee[0] + a * cube_xy[0]
            y = (1 - a) * ee[1] + a * cube_xy[1]
            z = ee[2]
            g = 1.0
        elif k <= BLEND + DESCEND:
            a = (k - BLEND) / float(DESCEND)
            x = cube_xy[0]
            y = cube_xy[1]
            z = (1 - a) * ee[2] + a * grasp_z
            g = 1.0
        elif k <= BLEND + DESCEND + CLOSE:
            x = cube_xy[0]
            y = cube_xy[1]
            z = grasp_z
            g = 0.0
        else:
            a = (k - BLEND - DESCEND - CLOSE) / float(LIFT)
            x = cube_xy[0]
            y = cube_xy[1]
            z = grasp_z + a * 0.12
            g = 0.0
        joints = solve([x, y, z], quat, seed)
        if joints is None:
            rec["ik_fail_at"] = k
            rec["ik_fail_pose"] = rlist([x, y, z])
            break
        plan.append((joints, g, [x, y, z]))
        seed = joints
    rec["grasp_z"] = rnd(grasp_z)
    rec["plan_len"] = len(plan)
    if len(plan) < BLEND + DESCEND + CLOSE:
        rec["abort"] = "tracking IK incomplete"
        report["attempts"].append(rec)
        return False
    grip_trace = []
    ee_trace = []
    for index, item in enumerate(plan):
        res = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=item[0], gripper_position=item[1], embodiment=EMB)}))
        step_info["steps"] += 1
        if step_info["first_step_repr"] is None:
            step_info["first_step_repr"] = str(res)[:600]
        if index >= BLEND + DESCEND - 2 or index % 4 == 0:
            st = get_robot_state()
            grip_trace.append(rnd(st.gripper_positions["primary"], 3))
            pos = [float(v) for v in st.end_effector_poses["primary"].position]
            ee_trace.append([index] + rlist([pos[0] - item[2][0], pos[1] - item[2][1],
                                             pos[2] - item[2][2]], 4))
        if mark_terminal(res, "track"):
            break
    rec["grip_trace"] = grip_trace
    rec["ee_err_trace"] = ee_trace[:30]
    reading = gripper_reading()
    rec["grip_after_lift"] = rnd(reading, 3)
    held = reading is not None and reading > 0.05
    # wrist evidence (diagnostic only): target-colour pixels in the lower half of the wrist view
    try:
        obs = get_observation()
        wb = detect_blobs(obs.cameras[WRIST], 4, 2, 5000, colors=[color]) if WRIST else []
        rec["wrist_blob_after_lift"] = [[b["npx"], rlist(b["px"], 0)] for b in wb[:4]]
    except Exception as exc:
        rec["wrist_blob_after_lift"] = "error: " + str(exc)[:120]
    rec["held"] = held
    report["attempts"].append(rec)
    return held


def deliver(color, attempt_no):
    rec = {"attempt": attempt_no, "color": color}
    if scene["bin"] is None:
        rec["abort"] = "no bin"
        report["drops"].append(rec)
        return False
    quat = get_robot_state().end_effector_poses["primary"].quaternion_wxyz
    quat = [float(v) for v in quat]
    bx = scene["bin"]["xy"][0]
    by = scene["bin"]["xy"][1]
    rim = scene["bin"]["rim"]
    ok, ticks = move_pose([bx, by, rim + 0.15], quat, "above_bin" + str(attempt_no))
    rec["above_ok"] = ok
    rec["above_ticks"] = ticks
    if episode["terminal"]:
        report["drops"].append(rec)
        return False
    # wrist refinement of the bin opening (dark interior seen from above)
    try:
        obs = get_observation()
        dark = dark_blob_center(obs.cameras[WRIST], 2, 28, rim + 0.02) if WRIST else None
        if dark is not None:
            off = [dark["xy"][0] - bx, dark["xy"][1] - by]
            rec["wrist_bin_offset"] = rlist(off)
            rec["wrist_bin_cells"] = dark["cells"]
            if math.sqrt(off[0] ** 2 + off[1] ** 2) < 0.04 and dark["cells"] > 20:
                bx = dark["xy"][0]
                by = dark["xy"][1]
                rec["bin_refined"] = True
    except Exception as exc:
        rec["wrist_bin_error"] = str(exc)[:160]
    ok, ticks = move_pose([bx, by, rim + 0.07], quat, "release" + str(attempt_no))
    rec["release_ok"] = ok
    rec["grip_before_open"] = rnd(gripper_reading(), 3)
    try:
        res = open_gripper()
        rec["open_steps"] = res_steps(res)
        mark_terminal(res, "open_gripper")
    except Exception as exc:
        rec["open_error"] = str(exc)[:160]
    if not episode["terminal"]:
        hold(4, 1.0)
    rec["grip_after_open"] = rnd(gripper_reading(), 3)
    # diagnostic: colour pixels inside the bin as seen from the wrist after release
    try:
        obs = get_observation()
        wb = detect_blobs(obs.cameras[WRIST], 2, 2, 5000, colors=[color]) if WRIST else []
        rec["wrist_color_blobs_after"] = [[b["npx"], rlist(b["xy"], 3)] for b in wb[:4]]
    except Exception as exc:
        rec["wrist_after_error"] = str(exc)[:120]
    rec["t"] = rnd(now_t(), 2)
    report["drops"].append(rec)
    return True


def press_button():
    rec = {"t_start": rnd(now_t(), 2)}
    report["button_press"] = rec
    if scene["button"] is None:
        rec["abort"] = "no button"
        return
    try:
        res = close_gripper()
        rec["close_steps"] = res_steps(res)
        mark_terminal(res, "close_for_button")
    except Exception as exc:
        rec["close_error"] = str(exc)[:160]
    bx = scene["button"]["xy"][0]
    by = scene["button"]["xy"][1]
    top = scene["button"]["top"]
    ok, ticks = move_pose([bx, by, top + 0.10], BASE_Q, "above_button")
    rec["above_ok"] = ok
    if episode["terminal"]:
        return
    for depth in (0.02, 0.035):
        seed = current_joints()
        start = current_ee()
        plan = []
        for k in range(1, 16):
            a = k / 15.0
            z = (1 - a) * start[2] + a * (top - depth)
            joints = solve([bx, by, z], BASE_Q, seed)
            if joints is None:
                rec["ik_fail_depth"] = depth
                break
            plan.append(joints)
            seed = joints
        press_seq = (plan + [plan[-1]] * 6) if plan else []
        for joints in press_seq:
            res = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=joints, gripper_position=0.0,
                embodiment=EMB)}))
            step_info["steps"] += 1
            if mark_terminal(res, "press"):
                break
        rec["ee_at_press_" + str(depth)] = rlist(current_ee())
        if episode["terminal"]:
            rec["terminal_at_depth"] = depth
            return
        ok, ticks = move_pose([bx, by, top + 0.08], BASE_Q, "after_press")
        if episode["terminal"]:
            return
    rec["no_terminal"] = True


# ---------------------------------------------------------------- main loop
report["attempts"] = []
report["drops"] = []
try:
    first, t_first = observe_front("init")
    try:
        report["projection_check"] = projection_check(
            sorted(first, key=lambda b: -b["npx"]))
    except Exception as exc:
        report["errors"].append("projection_check: " + str(exc)[:200])
    open_gripper()
    hold(8, 1.0)
    attempts = 0
    idle_rounds = 0
    while still_needed() and not episode["terminal"] and attempts < 30:
        if ticks_left() < 110:
            report["stop_reason"] = "step budget nearly spent"
            break
        blobs, t_obs = observe_front("loop")
        target = choose_target(blobs, t_obs) if velocity_ready() else None
        if target is None:
            idle_rounds += 1
            if not hold(8, 1.0):
                break
            if idle_rounds > 200:
                report["stop_reason"] = "no target for 200 rounds"
                break
            continue
        if report.get("first_projection_check_done") is None and len(blobs) > 0 and \
                report.get("projection_check") == []:
            report["projection_check"] = projection_check(blobs)
            report["first_projection_check_done"] = True
        attempts += 1
        held = pick_attempt(target, t_obs, attempts)
        if episode["terminal"]:
            break
        if held:
            if deliver(target["color"], attempts):
                delivered[target["color"]] += 1
        else:
            try:
                open_gripper()
            except Exception:
                pass
            ee = current_ee()
            move_pose([ee[0], ee[1], min(ee[2] + 0.08, 0.40)], BASE_Q, "retreat" + str(attempts))
    report["phases"].append({"phase": "picking_done", "t": rnd(now_t(), 2),
                             "delivered": dict(delivered), "attempts": attempts})
    if not episode["terminal"]:
        press_button()
except Exception as exc:
    report["errors"].append("main: " + str(exc)[:400])

report["delivered"] = dict(delivered)
report["velocity_history"] = vel["history"]
report["final_v"] = rlist(vel["v"])
report["timing"] = dict(timing)
report["step_info"] = step_info
report["episode"] = episode
report["t_end"] = rnd(now_t(), 2)
report["ticks_used"] = ticks_used()
report["log_counts"] = dict(log_counts)
result = report
