# v00_initial — SwingXtimes (robomme_plus task 10)
# Pick the named cube, alternate it between the two targets the instructed number of times
# (starting side from the instruction), put it down, press the button.
# Everything task-dependent is derived at runtime from get_task_context().language and live
# front/wrist RGB-D perception. Targets may hop while approached (seen in released recordings),
# so every visit is closed-loop: re-segment, follow, and confirm with the target's own
# "latched red" visual state (observation only).
import math

HZ = 20.0
LOG = []
DIAG = {"segments": [], "visits": [], "moves": []}
STATE = {"ended": False, "end_reason": None}


def log(msg):
    try:
        t = round(float(get_robot_state().timestamp_s), 2)
    except Exception:
        t = None
    LOG.append([t, msg])


def tick_now():
    try:
        return float(get_robot_state().timestamp_s) * HZ
    except Exception:
        return None


def check_end(res, label):
    term = False
    trunc = False
    try:
        term = bool(res.terminated)
    except Exception:
        term = False
    try:
        trunc = bool(res.truncated)
    except Exception:
        trunc = False
    if term or trunc:
        STATE["ended"] = True
        STATE["end_reason"] = label + (" terminated" if term else " truncated")
    return STATE["ended"]


def res_summary(res):
    out = {}
    try:
        out["ok"] = bool(res.ok)
    except Exception:
        out["ok"] = None
    try:
        out["status"] = str(res.status)
    except Exception:
        out["status"] = None
    try:
        out["steps"] = int(res.steps_executed)
    except Exception:
        out["steps"] = None
    try:
        if res.error is not None:
            out["error"] = str(res.error)[:200]
    except Exception:
        pass
    return out


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


# ---------------- quaternion helpers (wxyz) ----------------
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def qnorm(q):
    n = math.sqrt(sum(v * v for v in q))
    return tuple(v / n for v in q)


def qrot_axes(q):
    w, x, y, z = q
    xaxis = (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
    zaxis = (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
    return xaxis, zaxis


def qz(angle):
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def wrap90(a):
    # wrap an angle to [-pi/4, pi/4) modulo pi/2 (cube symmetry)
    q = math.pi / 2.0
    while a >= q / 2.0:
        a -= q
    while a < -q / 2.0:
        a += q
    return a


# ---------------- perception helpers ----------------
def cand_cloud(cand, camera, base_frame):
    pc = None
    try:
        pc = mask_to_point_cloud(cand, camera, target_frame=base_frame)
    except Exception:
        pc = None
    if pc is None:
        try:
            pc = mask_to_point_cloud(cand.mask, camera, target_frame=base_frame)
        except Exception:
            pc = None
    return pc


def cloud_stats(pc):
    pts = []
    try:
        for p in pc.points:
            pts.append((float(p[0]), float(p[1]), float(p[2])))
    except Exception:
        pts = []
    if len(pts) < 5:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    return {
        "n": len(pts),
        "cx": quantile(xs, 0.5), "cy": quantile(ys, 0.5),
        "z10": quantile(zs, 0.1), "z50": quantile(zs, 0.5), "z90": quantile(zs, 0.9),
        "z98": quantile(zs, 0.98),
        "xr": quantile(xs, 0.95) - quantile(xs, 0.05),
        "yr": quantile(ys, 0.95) - quantile(ys, 0.05),
        "pts": pts,
    }


def box_of(cand):
    try:
        if cand.box_xyxy is None:
            return None
        return [float(v) for v in cand.box_xyxy]
    except Exception:
        return None


def segment_all(camera, prompts, base_frame, tag, max_per_prompt=6):
    """Run several prompts; return de-duplicated candidates with 3D stats."""
    found = []
    for prompt in prompts:
        try:
            seg = segment_text(camera, prompt)
        except Exception as exc:
            DIAG["segments"].append({"tag": tag, "prompt": prompt, "error": str(exc)[:120]})
            continue
        entry = {"tag": tag, "prompt": prompt, "t": tick_now(), "cands": []}
        ok = False
        try:
            ok = bool(seg.ok)
        except Exception:
            ok = False
        if ok:
            for cand in seg.segmentations[:max_per_prompt]:
                score = 0.0
                try:
                    score = float(cand.score)
                except Exception:
                    score = 0.0
                box = box_of(cand)
                pc = cand_cloud(cand, camera, base_frame)
                st = cloud_stats(pc) if pc is not None else None
                rec = {"prompt": prompt, "score": round(score, 3), "box": box}
                if st is not None:
                    rec["c"] = [round(st["cx"], 4), round(st["cy"], 4), round(st["z50"], 4)]
                    rec["zr"] = [round(st["z10"], 4), round(st["z90"], 4)]
                    rec["xyr"] = [round(st["xr"], 4), round(st["yr"], 4)]
                    rec["n"] = st["n"]
                entry["cands"].append(rec)
                if st is None:
                    continue
                dup = False
                for f in found:
                    if (f["st"]["cx"] - st["cx"]) ** 2 + (f["st"]["cy"] - st["cy"]) ** 2 < 0.02 ** 2:
                        dup = True
                        if score > f["score"]:
                            f["score"] = score
                        break
                if not dup:
                    found.append({"prompt": prompt, "score": score, "box": box, "st": st,
                                  "cand": cand})
        DIAG["segments"].append(entry)
    return found


def rgb_at(rgb, x, y):
    p = rgb[y][x]
    return int(p[0]), int(p[1]), int(p[2])


def mean_color(rgb, box):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2.0)
    cy = int((y1 + y2) / 2.0)
    acc = [0, 0, 0]
    n = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            try:
                r, g, b = rgb_at(rgb, cx + dx, cy + dy)
            except Exception:
                continue
            acc[0] += r
            acc[1] += g
            acc[2] += b
            n += 1
    if n == 0:
        return None
    return [acc[0] / n, acc[1] / n, acc[2] / n]


def color_match(rgb_mean, color):
    if rgb_mean is None:
        return 0.0
    r, g, b = rgb_mean
    if color == "red":
        return (r - max(g, b)) / 255.0
    if color == "green":
        return (g - max(r, b)) / 255.0
    if color == "blue":
        return (b - max(r, g)) / 255.0
    if color == "yellow":
        return (min(r, g) - b) / 255.0
    return 0.0


def ring_counts(rgb, box):
    """Count latched-red ring pixels and grey ring pixels in the lower 60% of a target box."""
    if box is None:
        return None
    x1, y1, x2, y2 = box
    h = len(rgb)
    w = len(rgb[0])
    ys = int(y1 + 0.4 * (y2 - y1))
    red = 0
    grey = 0
    tot = 0
    for y in range(max(0, ys), min(h, int(y2) + 2)):
        for x in range(max(0, int(x1) - 1), min(w, int(x2) + 2)):
            r, g, b = rgb_at(rgb, x, y)
            tot += 1
            if r > 90 and g < 50 and b < 55 and r > 2.5 * g:
                red += 1
            elif max(r, g, b) - min(r, g, b) < 18 and 80 < max(r, g, b) < 205:
                grey += 1
    return {"red": red, "grey": grey, "tot": tot}


# ---------------- motion helpers ----------------
def move(pos, quat, base_frame, label, tol=0.01, max_steps=300):
    if STATE["ended"]:
        return None
    t0 = tick_now()
    res = None
    try:
        res = move_to_pose(Pose(tuple(pos), tuple(quat), base_frame), tolerance=tol,
                           max_steps=max_steps)
    except Exception as exc:
        DIAG["moves"].append({"label": label, "exception": str(exc)[:200]})
        return None
    rec = res_summary(res)
    rec["label"] = label
    rec["target"] = [round(v, 4) for v in pos]
    rec["t0"] = t0
    rec["t1"] = tick_now()
    DIAG["moves"].append(rec)
    check_end(res, label)
    return res


def hold_ticks(n, gripper):
    """Hold current joints for n control ticks (advances time)."""
    for k in range(n):
        if STATE["ended"]:
            return
        st = get_robot_state()
        try:
            reply = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=st.joint_positions["primary"],
                gripper_position=gripper, embodiment=st.embodiment)}))
        except Exception as exc:
            log("hold step exception " + str(exc)[:120])
            return
        check_end(reply, "hold")


def ee_pos():
    st = get_robot_state()
    p = st.end_effector_poses["primary"].position
    return [float(p[0]), float(p[1]), float(p[2])]


# ---------------- language ----------------
NUMBERS = {"once": 1, "twice": 2, "thrice": 3, "one": 1, "two": 2, "three": 3, "four": 4,
           "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "eleven": 11, "twelve": 12}


def parse_goal(text):
    t = text.lower().replace(",", " ").replace(".", " ")
    words = t.split()
    color = None
    for i in range(len(words) - 1):
        if words[i + 1] in ("cube", "block") and color is None:
            color = words[i]
    first = None
    k = t.find("move it to the")
    seg = t[k:] if k >= 0 else t
    il = seg.find("left")
    ir = seg.find("right")
    if il >= 0 and (ir < 0 or il < ir):
        first = "left"
    elif ir >= 0:
        first = "right"
    count = None
    for i in range(len(words)):
        wv = words[i]
        if wv in ("once", "twice", "thrice"):
            count = NUMBERS[wv]
        if wv == "times" and i > 0:
            prev = words[i - 1]
            if prev in NUMBERS:
                count = NUMBERS[prev]
            else:
                try:
                    count = int(prev)
                except Exception:
                    pass
    return color, first, count


# =====================================================================
context = get_task_context()
goal = str(context.language)
state0 = get_robot_state()
base_frame = state0.base_frame
report = {"goal": goal, "base_frame": base_frame, "log": LOG, "diag": DIAG}

meta = {}
try:
    meta = dict(robomme_plus.get_task_metadata())
except Exception as exc:
    report["meta_error"] = str(exc)[:200]
try:
    report["meta"] = {str(k): str(meta[k])[:300] for k in meta}
except Exception:
    report["meta"] = None
for key in ("control_hz", "control_frequency_hz", "control_freq"):
    try:
        if key in meta:
            HZ = float(meta[key])
    except Exception:
        pass
max_steps_total = None
try:
    max_steps_total = int(meta["max_steps"])
except Exception:
    max_steps_total = None
report["max_steps"] = max_steps_total
try:
    report["controller_meta"] = {str(k): str(v)[:300]
                                 for k, v in dict(robomme_plus.get_controller_metadata()).items()}
except Exception as exc:
    report["controller_meta_error"] = str(exc)[:200]
try:
    video = robomme_plus.get_conditioning_video()
    report["conditioning_video_len"] = len(video)
except Exception as exc:
    report["conditioning_video_error"] = str(exc)[:200]

color, first_side, count = parse_goal(goal)
if color is None:
    color = "red"
if first_side is None:
    first_side = "right"
if count is None:
    count = 1
second_side = "left" if first_side == "right" else "right"
sequence = []
for k in range(count):
    sequence.append(first_side)
    sequence.append(second_side)
report["parsed"] = {"color": color, "first": first_side, "count": count,
                    "n_visits": len(sequence)}

home_pose = state0.end_effector_poses["primary"]
home_pos = [float(v) for v in home_pose.position]
home_q = tuple(float(v) for v in home_pose.quaternion_wxyz)
xa, za = qrot_axes(home_q)
report["home"] = {"pos": home_pos, "quat": list(home_q), "zaxis": list(za)}
down_q = home_q
if za[2] > -0.9:
    down_q = (0.0, 1.0, 0.0, 0.0)
    report["home_not_down"] = True

# ---------------- initial scene perception (front camera, arm at home) ----------------
obs0 = get_observation()
front_rgb = obs0.cameras["front"].rgb
img_w = len(front_rgb[0])

cube_prompts = [color + " cube", color + " block"]
cubes = segment_all("front", cube_prompts, base_frame, "cube0")
best_cube = None
best_val = None
for c in cubes:
    mc = mean_color(front_rgb, c["box"])
    cm = color_match(mc, color)
    c["color_match"] = cm
    if c["st"]["xr"] > 0.12 or c["st"]["yr"] > 0.12:
        continue
    val = c["score"] + 2.0 * cm
    if cm < 0.05:
        continue
    if best_val is None or val > best_val:
        best_val = val
        best_cube = c
if best_cube is None:
    for c in cubes:
        if best_val is None or c["score"] > best_val:
            best_val = c["score"]
            best_cube = c
if best_cube is None:
    result = report
    raise ValueError("cube not found")
cube_st = best_cube["st"]
report["cube_front"] = {"c": [cube_st["cx"], cube_st["cy"], cube_st["z50"]],
                        "z98": cube_st["z98"], "box": best_cube["box"],
                        "color_match": best_cube["color_match"]}

target_prompts = ["round target", "bullseye target", "target"]
tcands = segment_all("front", target_prompts, base_frame, "targets0")
button_cands = segment_all("front", ["button", "white square button"], base_frame, "button0")

# Table height estimate: lowest robust surface among flat target-like candidates and cube bottom.
table_candidates = []
for c in tcands:
    table_candidates.append(c["st"]["z10"])
cube_top = cube_st["z98"]
flat = []
for c in tcands:
    st = c["st"]
    if st["xr"] > 0.15 or st["yr"] > 0.15:
        continue
    if st["z90"] - st["z10"] > 0.02:
        continue
    if ((st["cx"] - cube_st["cx"]) ** 2 + (st["cy"] - cube_st["cy"]) ** 2) < 0.03 ** 2:
        continue
    flat.append(c)
table_z = None
if len(flat) > 0:
    table_z = quantile([c["st"]["z50"] for c in flat], 0.0)
if table_z is None:
    table_z = cube_st["z10"]
report["table_z"] = table_z

# Button: candidate standing clearly above the table, compact.
button = None
bval = None
for c in button_cands + tcands:
    st = c["st"]
    if st["xr"] > 0.15 or st["yr"] > 0.15:
        continue
    if st["z98"] - table_z < 0.012:
        continue
    if ((st["cx"] - cube_st["cx"]) ** 2 + (st["cy"] - cube_st["cy"]) ** 2) < 0.04 ** 2:
        continue
    v = c["score"] + (0.5 if c["prompt"] in ("button", "white square button") else 0.0)
    if bval is None or v > bval:
        bval = v
        button = c
if button is not None:
    report["button0"] = {"c": [button["st"]["cx"], button["st"]["cy"]], "top": button["st"]["z98"],
                         "box": button["box"]}

# Targets: flat, not the button, two laterally separated.
targets = []
flat_sorted = []
remaining_flat = list(flat)
while len(remaining_flat) > 0:
    bi = 0
    for k in range(len(remaining_flat)):
        if remaining_flat[k]["score"] > remaining_flat[bi]["score"]:
            bi = k
    flat_sorted.append(remaining_flat[bi])
    remaining_flat = remaining_flat[:bi] + remaining_flat[bi + 1:]
for c in flat_sorted:
    st = c["st"]
    if button is not None:
        if (st["cx"] - button["st"]["cx"]) ** 2 + (st["cy"] - button["st"]["cy"]) ** 2 < 0.04 ** 2:
            continue
    if st["z90"] - table_z > 0.02:
        continue
    clash = False
    for t in targets:
        if (t["st"]["cx"] - st["cx"]) ** 2 + (t["st"]["cy"] - st["cy"]) ** 2 < 0.08 ** 2:
            clash = True
    if not clash:
        targets.append(c)
    if len(targets) == 2:
        break
if len(targets) < 2:
    result = report
    raise ValueError("fewer than two targets found: " + str(len(targets)))


def box_cx(c):
    if c["box"] is None:
        return None
    return (c["box"][0] + c["box"][2]) / 2.0


ta, tb = targets[0], targets[1]
# Language left/right is the robot's: in the front camera (which faces the robot) the robot's
# LEFT target is on the IMAGE RIGHT (released recordings: "left" visits go to image right).
side_by_image = None
if box_cx(ta) is not None and box_cx(tb) is not None:
    side_by_image = "a_left" if box_cx(ta) > box_cx(tb) else "b_left"
side_by_y = "a_left" if ta["st"]["cy"] > tb["st"]["cy"] else "b_left"
report["side_check"] = {"by_image": side_by_image, "by_base_y": side_by_y}
decision = side_by_image if side_by_image is not None else side_by_y
if decision == "a_left":
    left_t, right_t = ta, tb
else:
    left_t, right_t = tb, ta
TRACK = {
    "left": {"x": left_t["st"]["cx"], "y": left_t["st"]["cy"], "top": left_t["st"]["z90"],
             "box": left_t["box"]},
    "right": {"x": right_t["st"]["cx"], "y": right_t["st"]["cy"], "top": right_t["st"]["z90"],
              "box": right_t["box"]},
}
report["targets0"] = {k: {"x": TRACK[k]["x"], "y": TRACK[k]["y"], "top": TRACK[k]["top"],
                          "box": TRACK[k]["box"]} for k in TRACK}


def find_target(side):
    """Re-segment targets in the front camera and update TRACK[side] (handles hops)."""
    other = "left" if side == "right" else "right"
    cands = segment_all("front", ["round target", "target"], base_frame, "track_" + side)
    best = None
    bestd = None
    try:
        ee = ee_pos()
    except Exception:
        ee = None
    for c in cands:
        st = c["st"]
        if st["xr"] > 0.15 or st["yr"] > 0.15:
            continue
        if st["z90"] - table_z > 0.025:
            continue
        if button is not None:
            if (st["cx"] - button["st"]["cx"]) ** 2 + (st["cy"] - button["st"]["cy"]) ** 2 < 0.04 ** 2:
                continue
        ds = (st["cx"] - TRACK[side]["x"]) ** 2 + (st["cy"] - TRACK[side]["y"]) ** 2
        do = (st["cx"] - TRACK[other]["x"]) ** 2 + (st["cy"] - TRACK[other]["y"]) ** 2
        if do < ds:
            continue
        if ds > 0.20 ** 2:
            continue
        if bestd is None or ds < bestd:
            bestd = ds
            best = c
    if best is None:
        return False, None
    moved = math.sqrt(bestd)
    TRACK[side]["x"] = best["st"]["cx"]
    TRACK[side]["y"] = best["st"]["cy"]
    TRACK[side]["top"] = min(best["st"]["z90"], table_z + 0.02)
    TRACK[side]["box"] = best["box"]
    return True, moved


def target_visited(side, baseline):
    try:
        rgb = get_observation(camera_names=["front"]).cameras["front"].rgb
    except Exception:
        try:
            rgb = get_observation().cameras["front"].rgb
        except Exception:
            return False, None
    rc = ring_counts(rgb, TRACK[side]["box"])
    if rc is None:
        return False, None
    vis = False
    if rc["tot"] > 0:
        red_gain = rc["red"] - (baseline["red"] if baseline is not None else 0)
        grey_drop = True
        if baseline is not None and baseline["grey"] >= 6:
            grey_drop = rc["grey"] <= 0.5 * baseline["grey"]
        vis = red_gain >= max(4, 0.08 * rc["tot"]) and grey_drop
    return vis, rc


# ---------------- grasp the cube ----------------
open_gripper()
cube_x, cube_y = cube_st["cx"], cube_st["cy"]
pre_z = cube_top + 0.12
res = move([cube_x, cube_y, pre_z], down_q, base_frame, "pre_grasp_front", tol=0.015)

# refine with wrist camera (top-down view)
grasp_q = down_q
side_len = None
wr = segment_all("wrist", cube_prompts, base_frame, "cube_wrist")
best_w = None
bw = None
try:
    wrist_rgb = get_observation().cameras["wrist"].rgb
except Exception:
    wrist_rgb = None
for c in wr:
    st = c["st"]
    if st["xr"] > 0.12 or st["yr"] > 0.12:
        continue
    d = (st["cx"] - cube_x) ** 2 + (st["cy"] - cube_y) ** 2
    if d > 0.06 ** 2:
        continue
    cm = color_match(mean_color(wrist_rgb, c["box"]), color) if wrist_rgb is not None else 0.0
    v = c["score"] + 2.0 * cm - 10.0 * math.sqrt(d)
    if bw is None or v > bw:
        bw = v
        best_w = c
yaw_info = {}
if best_w is not None:
    pts = best_w["st"]["pts"]
    ztop = best_w["st"]["z98"]
    top_pts = [p for p in pts if p[2] > ztop - 0.012]
    if len(top_pts) < 15:
        top_pts = pts
    best_ang = None
    best_area = None
    best_dims = None
    for k in range(30):
        ang = k * (math.pi / 2.0) / 30.0
        ca = math.cos(ang)
        sa = math.sin(ang)
        us = [p[0] * ca + p[1] * sa for p in top_pts]
        vs = [-p[0] * sa + p[1] * ca for p in top_pts]
        u0, u1 = quantile(us, 0.03), quantile(us, 0.97)
        v0, v1 = quantile(vs, 0.03), quantile(vs, 0.97)
        area = (u1 - u0) * (v1 - v0)
        if best_area is None or area < best_area:
            best_area = area
            best_ang = ang
            best_dims = (u1 - u0, v1 - v0, (u0 + u1) / 2.0, (v0 + v1) / 2.0)
    du, dv, uc, vc = best_dims
    ca = math.cos(best_ang)
    sa = math.sin(best_ang)
    cube_x = uc * ca - vc * sa
    cube_y = uc * sa + vc * ca
    cube_top = ztop
    side_len = max(du, dv)
    gx, gz = qrot_axes(down_q)
    grip_yaw = math.atan2(gx[1], gx[0])
    delta = wrap90(best_ang - grip_yaw)
    grasp_q = qnorm(qmul(qz(delta), down_q))
    yaw_info = {"cube_yaw_deg": round(math.degrees(best_ang), 1),
                "grip_yaw_deg": round(math.degrees(grip_yaw), 1),
                "delta_deg": round(math.degrees(delta), 1),
                "dims": [round(du, 4), round(dv, 4)], "n_top": len(top_pts)}
report["cube_wrist"] = {"found": best_w is not None, "x": cube_x, "y": cube_y, "top": cube_top,
                        "yaw": yaw_info}

if side_len is None or side_len < 0.02 or side_len > 0.07:
    side_len = max(0.02, min(0.06, cube_top - table_z))
cube_h = max(0.02, min(0.07, cube_top - table_z))
report["cube_dims"] = {"side_xy": side_len, "height": cube_h}
grasp_z = max(table_z + 0.012, cube_top - min(0.5 * cube_h, 0.025))
cube_bottom = cube_top - cube_h
bottom_offset = grasp_z - cube_bottom  # EE z minus cube bottom z while held

move([cube_x, cube_y, cube_top + 0.10], grasp_q, base_frame, "pre_grasp_aligned", tol=0.012)
move([cube_x, cube_y, grasp_z], grasp_q, base_frame, "grasp_descend", tol=0.008)
cg = None
try:
    cg = close_gripper()
    check_end(cg, "close_gripper")
except Exception as exc:
    log("close exception " + str(exc)[:100])
lift_z = table_z + bottom_offset + 0.09
move([cube_x, cube_y, lift_z], grasp_q, base_frame, "lift", tol=0.015)
grip_after = None
try:
    grip_after = float(get_robot_state().gripper_positions["primary"])
except Exception:
    grip_after = None
report["grip_after_lift"] = grip_after
report["grasp_ok_guess"] = (grip_after is not None and grip_after > 0.05)

# One re-grasp attempt if the jaw closed on nothing.
if (not STATE["ended"]) and grip_after is not None and grip_after <= 0.05:
    log("regrasp: jaw closed on nothing")
    open_gripper()
    re = segment_all("front", cube_prompts, base_frame, "cube_regrasp")
    for c in re:
        if color_match(mean_color(get_observation().cameras["front"].rgb, c["box"]), color) > 0.05:
            cube_x, cube_y = c["st"]["cx"], c["st"]["cy"]
            break
    move([cube_x, cube_y, cube_top + 0.10], grasp_q, base_frame, "regrasp_pre", tol=0.012)
    move([cube_x, cube_y, grasp_z], grasp_q, base_frame, "regrasp_descend", tol=0.008)
    try:
        cg = close_gripper()
        check_end(cg, "close_gripper2")
    except Exception:
        pass
    move([cube_x, cube_y, lift_z], grasp_q, base_frame, "relift", tol=0.015)
    try:
        report["grip_after_regrasp"] = float(get_robot_state().gripper_positions["primary"])
    except Exception:
        pass

# ---------------- swings ----------------
HOVER_CLEAR = 0.02     # cube bottom above target top while visiting
TRANSIT_CLEAR = 0.06   # cube bottom above target top on arrival from the other side
visits_done = 0
for vi in range(len(sequence)):
    if STATE["ended"]:
        break
    side = sequence[vi]
    vrec = {"i": vi, "side": side, "t_start": tick_now(), "hops": [], "polls": []}
    # Re-observe the destination before leaving (arm is over the other side / centre).
    ok, moved = find_target(side)
    vrec["pre_seen"] = ok
    vrec["pre_moved"] = moved
    baseline = None
    try:
        baseline = ring_counts(get_observation().cameras["front"].rgb, TRACK[side]["box"])
    except Exception:
        baseline = None
    vrec["baseline"] = baseline
    # Tick budget: keep enough control time for the remaining visits, put-down and button.
    max_polls = 8
    allow_lower = True
    tnow = tick_now()
    if max_steps_total is not None and tnow is not None:
        per_visit = (max_steps_total - tnow - 160.0) / float(len(sequence) - vi)
        vrec["per_visit_allow"] = round(per_visit, 1)
        if per_visit < 70:
            max_polls = 3
            allow_lower = False
        if per_visit < 45:
            max_polls = 1
    tgt = TRACK[side]
    move([tgt["x"], tgt["y"], tgt["top"] + bottom_offset + TRANSIT_CLEAR], grasp_q, base_frame,
         "v%d_%s_transit" % (vi, side), tol=0.015)
    if STATE["ended"]:
        break
    ok, moved = find_target(side)
    if ok and moved is not None and moved > 0.015:
        vrec["hops"].append(["after_transit", round(moved, 4)])
    tgt = TRACK[side]
    clear = HOVER_CLEAR
    move([tgt["x"], tgt["y"], tgt["top"] + bottom_offset + clear], grasp_q, base_frame,
         "v%d_%s_hover" % (vi, side), tol=0.01)
    visited = False
    lowered = False
    for poll in range(max_polls):
        if STATE["ended"]:
            break
        vis, rc = target_visited(side, baseline)
        vrec["polls"].append([poll, tick_now(), rc, vis])
        if vis:
            visited = True
            break
        ok, moved = find_target(side)
        if ok and moved is not None and moved > 0.015:
            vrec["hops"].append([poll, round(moved, 4)])
            tgt = TRACK[side]
            move([tgt["x"], tgt["y"], tgt["top"] + bottom_offset + clear], grasp_q, base_frame,
                 "v%d_%s_follow%d" % (vi, side, poll), tol=0.01)
            continue
        if poll >= 2 and not lowered and allow_lower:
            lowered = True
            clear = 0.008
            move([tgt["x"], tgt["y"], tgt["top"] + bottom_offset + clear], grasp_q, base_frame,
                 "v%d_%s_lower" % (vi, side), tol=0.008)
            continue
        hold_ticks(3, 0.0)
    vrec["visited_seen"] = visited
    vrec["track"] = [round(TRACK[side]["x"], 4), round(TRACK[side]["y"], 4),
                     round(TRACK[side]["top"], 4)]
    vrec["lowered"] = lowered
    vrec["t_end"] = tick_now()
    DIAG["visits"].append(vrec)
    visits_done += 1
report["visits_done"] = visits_done

# ---------------- put down ----------------
if not STATE["ended"]:
    p = ee_pos()
    move([p[0], p[1], table_z + bottom_offset + 0.012], grasp_q, base_frame, "putdown", tol=0.01)
    if not STATE["ended"]:
        try:
            r = open_gripper()
            check_end(r, "release")
        except Exception:
            pass
    move([p[0], p[1], table_z + bottom_offset + 0.10], grasp_q, base_frame, "retreat", tol=0.02)

# ---------------- press button ----------------
if not STATE["ended"]:
    bc = segment_all("front", ["button", "white square button"], base_frame, "button1")
    bsel = None
    bv = None
    for c in bc:
        st = c["st"]
        if st["xr"] > 0.15 or st["yr"] > 0.15:
            continue
        if st["z98"] - table_z < 0.008:
            continue
        v = c["score"]
        if button is not None:
            d = math.sqrt((st["cx"] - button["st"]["cx"]) ** 2 + (st["cy"] - button["st"]["cy"]) ** 2)
            v -= 5.0 * d
        if bv is None or v > bv:
            bv = v
            bsel = c
    if bsel is None:
        bsel = button
    if bsel is not None:
        bst = bsel["st"]
        btop = bst["z98"]
        cap = [p for p in bst["pts"] if p[2] > btop - 0.006]
        if len(cap) < 5:
            cap = bst["pts"]
        bx = quantile([p[0] for p in cap], 0.5)
        by = quantile([p[1] for p in cap], 0.5)
        report["button_press"] = {"x": bx, "y": by, "top": btop, "n_cap": len(cap)}
        try:
            r = close_gripper()
            check_end(r, "close_for_button")
        except Exception:
            pass
        move([bx, by, btop + 0.07], grasp_q, base_frame, "button_above", tol=0.015)
        move([bx, by, btop - 0.012], grasp_q, base_frame, "button_press1", tol=0.01, max_steps=120)
        if not STATE["ended"]:
            hold_ticks(3, 0.0)
        if not STATE["ended"]:
            move([bx, by, btop - 0.025], grasp_q, base_frame, "button_press2", tol=0.01,
                 max_steps=120)
        if not STATE["ended"]:
            move([bx, by, btop + 0.07], grasp_q, base_frame, "button_release", tol=0.02)
    else:
        report["button_press"] = "button not found"

report["ended"] = STATE["ended"]
report["end_reason"] = STATE["end_reason"]
report["final_tick"] = tick_now()
# drop bulky point lists before reporting
result = report
