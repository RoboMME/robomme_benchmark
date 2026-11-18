# StopCube v02 -- press the button just as the moving cube reaches the target for the Nth time.
# v00 -> v01 (perception mode, ep1): identify objects by GEOMETRY + MOTION instead of prompt rank:
#   target = largest FLAT candidate (point-cloud height < 6 mm) over several prompts (incl. "button",
#            whose top-1 was the target disc on ep1); cube = the candidate that MOVES over the first
#            ticks (arm held at home); button = a non-flat, STATIC, low candidate of "button"
#            (fallback: static non-flat "cube" candidates). Cube tracking gate tied to cube speed;
#            pass counting uses the distance from the target to the segment between consecutive
#            detections (fast passes fall between samples).
# v01 -> v02 (time-to-ready ablation): approach 22 ticks (was 30) to a pre-hover one press-jump
#   above the hover; the free-air descent onto the hover doubles as the latency calibration
#   (removes the separate mid pose and 8-tick descend phase). Hover ready ~tick 37 instead of ~56-60.
import math

ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
    "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "once": 1, "twice": 2, "thrice": 3,
}

report = {"version": "v02", "errors": []}
ctx = get_task_context()
goal = str(ctx.language)
report["goal"] = goal


def parse_count(text):
    words = text.lower().replace(",", " ").replace(".", " ").split()
    for word in words:
        if word in ORDINALS:
            return ORDINALS[word]
    for word in words:
        digits = ""
        for ch in word:
            if ch in "0123456789":
                digits = digits + ch
            else:
                break
        if digits != "" and len(word) > len(digits) and word[len(digits):] in ("st", "nd", "rd", "th"):
            return int(digits)
    return None


N = parse_count(goal)
report["N"] = N
if N is None:
    N = 1
    report["errors"].append("could not parse ordinal; defaulting to 1")

max_steps = 400
try:
    md = robomme_plus.get_task_metadata()
    max_steps = int(md["max_steps"])
    report["max_steps"] = max_steps
except Exception as exc:
    report["errors"].append("metadata: " + str(exc))
try:
    video = robomme_plus.get_conditioning_video()
    report["conditioning_video_len"] = len(video)
except Exception as exc:
    report["errors"].append("video: " + str(exc))

state0 = get_robot_state()
emb = state0.embodiment
base_frame = state0.base_frame
q_home = [float(v) for v in state0.joint_positions["primary"]]
ee0 = state0.end_effector_poses["primary"]
quat = [float(v) for v in ee0.quaternion_wxyz]
report["home_ee"] = [round(float(v), 4) for v in ee0.position]


def box_of(cand):
    b = cand.box_xyxy
    return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]


def box_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = box_area(a) + box_area(b) - inter
    if ua <= 0:
        return 0.0
    return inter / ua


def overlap_frac(a, b):
    """Fraction of box a covered by box b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    if box_area(a) <= 0:
        return 0.0
    return ix * iy / box_area(a)


def quantile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def cloud_stats(cand):
    pc = mask_to_point_cloud(cand, "front", target_frame=base_frame)
    pts = pc.points
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    zs = [float(p[2]) for p in pts]
    if len(zs) < 5:
        return None
    return {"n": len(zs), "x": quantile(xs, 0.5), "y": quantile(ys, 0.5),
            "z10": quantile(zs, 0.1), "z50": quantile(zs, 0.5), "z95": quantile(zs, 0.95),
            "h": quantile(zs, 0.95) - quantile(zs, 0.1), "xs": xs, "ys": ys, "zs": zs}


def describe(prompt, cand, st):
    b = box_of(cand)
    d = {"prompt": prompt, "box": [round(v, 1) for v in b], "score": round(float(cand.score), 3)}
    if st is not None:
        d["h"] = round(st["h"], 4)
        d["z95"] = round(st["z95"], 4)
        d["xy"] = [round(st["x"], 4), round(st["y"], 4)]
    return d


def seg_boxes(prompt):
    seg = segment_text("front", prompt)
    out = []
    if seg.ok:
        for cand in seg.segmentations[:5]:
            out.append(box_of(cand))
    return out


FLAT_H = 0.006
HIGH_Z = 0.07

# ---------------- tick 0 perception (arm at home) ----------------
cands0 = []          # (prompt, cand, stats)
for prompt in ("button", "target", "bullseye", "disc", "cube"):
    seg = segment_text("front", prompt)
    if not seg.ok:
        continue
    for cand in seg.segmentations[:5]:
        st = cloud_stats(cand)
        cands0.append((prompt, cand, st))
report["cands0"] = [describe(p, c, s) for (p, c, s) in cands0]

# target: flat candidates; best-scored flat one, then the largest flat box overlapping it
flat = []
for (p, c, s) in cands0:
    if s is not None and s["h"] < FLAT_H and s["z95"] < HIGH_Z:
        flat.append((p, c, s))
if len(flat) == 0:
    raise ValueError("no flat target candidate")
best_flat = None
for item in flat:
    if item[0] == "cube":
        continue
    if best_flat is None or float(item[1].score) > float(best_flat[1].score):
        best_flat = item
if best_flat is None:
    best_flat = flat[0]
target_box = box_of(best_flat[1])
seed_box = target_box
for item in flat:
    b = box_of(item[1])
    if overlap_frac(seed_box, b) > 0.8 and box_area(b) > box_area(target_box) and box_area(b) < 9.0 * box_area(seed_box) \
            and (b[2] - b[0]) < 60 and (b[3] - b[1]) < 60:
        target_box = b
target_c = box_center(target_box)
target_w = max(target_box[2] - target_box[0], target_box[3] - target_box[1])
report["target_box"] = [round(v, 1) for v in target_box]
report["target_xyz"] = [round(best_flat[2]["x"], 4), round(best_flat[2]["y"], 4), round(best_flat[2]["z50"], 4)]

# solid low candidates (cube or button), not the target
solid0 = []
for (p, c, s) in cands0:
    if s is None or s["h"] < FLAT_H or s["z95"] > HIGH_Z:
        continue
    b = box_of(c)
    if (b[2] - b[0]) > 3.0 * target_w or (b[3] - b[1]) > 3.0 * target_w:
        continue
    if overlap_frac(b, target_box) > 0.85 and overlap_frac(target_box, b) > 0.85:
        continue
    solid0.append((p, c, s, b))

# ---------------- motion identification: hold the arm, re-segment "cube" ----------------
hold_lists = [[box_of(c) for (p, c, s) in cands0 if p == "cube"]]
tick = 0
cube_box = None
terminated = False
ident_disp = None
while tick < 10 and cube_box is None:
    reply = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=q_home, gripper_position=0.0, embodiment=emb)}))
    tick = tick + 1
    hold_lists.append(seg_boxes("cube"))
    if tick >= 3:
        best = None
        for b in hold_lists[tick]:
            c = box_center(b)
            if (b[2] - b[0]) > 3.0 * target_w or (b[3] - b[1]) > 3.0 * target_w:
                continue
            disp = None
            for b0 in hold_lists[0]:
                c0 = box_center(b0)
                dd = math.hypot(c[0] - c0[0], c[1] - c0[1])
                if disp is None or dd < disp:
                    disp = dd
            if disp is None:
                continue
            if disp >= 2.0 and disp <= 15.0 * tick:
                if best is None or disp > best[0]:
                    best = (disp, b)
        if best is not None:
            cube_box = best[1]
            ident_disp = best[0]
report["ident_tick"] = tick
report["ident_disp"] = None if ident_disp is None else round(ident_disp, 2)
if cube_box is None:
    raise ValueError("no moving cube candidate found")
cube_w = max(cube_box[2] - cube_box[0], cube_box[3] - cube_box[1])
report["cube_box_ident"] = [round(v, 1) for v in cube_box]

# backtrack the cube through the hold ticks for pass counting
path = [box_center(cube_box)]
for k in range(tick - 1, -1, -1):
    last = path[0]
    best = None
    for b in hold_lists[k]:
        c = box_center(b)
        dd = math.hypot(c[0] - last[0], c[1] - last[1])
        if dd <= 2.0 * cube_w and (best is None or dd < best[0]):
            best = (dd, c)
    path.insert(0, last if best is None else best[1])

# button: non-flat, static, low candidates of "button" not overlapping the cube's hold path
button_choice = None
for prompt_pref in ("button", "cube"):
    for (p, c, s, b) in solid0:
        if p != prompt_pref:
            continue
        near_cube = False
        for pt in path:
            if abs(box_center(b)[0] - pt[0]) < 0.5 * cube_w + 2 and abs(box_center(b)[1] - pt[1]) < 0.5 * cube_w + 2:
                near_cube = True
        if near_cube:
            continue
        # static check against the latest hold segmentation
        static = False
        for bl in hold_lists[tick]:
            cb = box_center(bl)
            if math.hypot(cb[0] - box_center(b)[0], cb[1] - box_center(b)[1]) < 2.0:
                static = True
        if p == "cube" and not static:
            continue
        if button_choice is None or float(c.score) > float(button_choice[1].score):
            button_choice = (p, c, s, b)
    if button_choice is not None:
        break
if button_choice is None:
    raise ValueError("button not identified")
button_box = button_choice[3]
bstats = button_choice[2]
cap_top = bstats["z95"]
cap_x = []
cap_y = []
for i in range(len(bstats["zs"])):
    if bstats["zs"][i] >= cap_top - 0.004:
        cap_x.append(bstats["xs"][i])
        cap_y.append(bstats["ys"][i])
cap_center = [sum(cap_x) / len(cap_x), sum(cap_y) / len(cap_y)]
report["button"] = {"prompt": button_choice[0], "box": [round(v, 1) for v in button_box],
                    "score": round(float(button_choice[1].score), 3), "cap_top": round(cap_top, 4),
                    "cap_xy": [round(v, 4) for v in cap_center], "h": round(bstats["h"], 4)}

thr_reach = 0.45 * target_w
thr_exit = max(2.0 * thr_reach, thr_reach + 6.0)
report["thr_reach_px"] = round(thr_reach, 2)
report["cube_w"] = round(cube_w, 1)


def seg_dist(p0, p1, q):
    """Distance from point q to segment p0-p1 and the segment parameter of the closest point."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    l2 = dx * dx + dy * dy
    if l2 < 1e-9:
        return math.hypot(q[0] - p1[0], q[1] - p1[1]), 1.0
    s = ((q[0] - p0[0]) * dx + (q[1] - p0[1]) * dy) / l2
    s = max(0.0, min(1.0, s))
    return math.hypot(p0[0] + s * dx - q[0], p0[1] + s * dy - q[1]), s


vis = {"reaches": 0, "in_visit": False, "min_d": None}
reach_ticks = []
visit_log = []
d_start = math.hypot(path[0][0] - target_c[0], path[0][1] - target_c[1])
if d_start <= thr_reach:
    vis["in_visit"] = True
    vis["reaches"] = 1
    vis["min_d"] = d_start
    reach_ticks.append(0)


def update_visits(p0, p1, t):
    dseg, s = seg_dist(p0, p1, target_c)
    d1 = math.hypot(p1[0] - target_c[0], p1[1] - target_c[1])
    if vis["in_visit"]:
        if vis["min_d"] is None or dseg < vis["min_d"]:
            vis["min_d"] = dseg
        if d1 > thr_exit:
            vis["in_visit"] = False
            visit_log.append([t, round(vis["min_d"], 1)])
            vis["min_d"] = None
    elif dseg <= thr_reach:
        vis["in_visit"] = True
        vis["reaches"] = vis["reaches"] + 1
        reach_ticks.append(t)
        vis["min_d"] = dseg
    return d1, dseg


track = []
for k in range(1, len(path)):
    d1, dseg = update_visits(path[k - 1], path[k], k)
    track.append([k, round(path[k][0], 1), round(path[k][1], 1), round(d1, 1), round(dseg, 1), 0, "id", 0.0])

cube_c = path[-1]
prev_c = path[-2] if len(path) >= 2 else path[-1]
prev2_c = path[-3] if len(path) >= 3 else prev_c
missing = 0


def detect_cube(pred, gate):
    seg = segment_text("front", "cube")
    best = None
    if not seg.ok:
        return None
    for cand in seg.segmentations[:5]:
        b = box_of(cand)
        if iou(b, button_box) > 0.3:
            continue
        if (b[2] - b[0]) > 2.0 * cube_w + 4 or (b[3] - b[1]) > 2.0 * cube_w + 4:
            continue
        c = box_center(b)
        dist = math.hypot(c[0] - pred[0], c[1] - pred[1])
        if iou(b, target_box) > 0.75:
            dist = dist + 0.3 * cube_w
        if dist > gate:
            continue
        if best is None or dist < best[0]:
            best = (dist, c)
    if best is None:
        return None
    return best[1]


# ---------------- arm targets ----------------
def ik(pos, seed):
    res = solve_ik(Pose(position=[float(pos[0]), float(pos[1]), float(pos[2])],
                        quaternion_wxyz=quat, frame=base_frame), seed_joints=seed)
    if not res.ok:
        return None
    return [float(v) for v in res.joint_positions]


HOVER_GAP = 0.015
PRESS_DEPTH = 0.02
# v02: the pre-hover sits one press-jump (HOVER_GAP + PRESS_DEPTH) above the hover, so the
# descent to the hover IS the latency calibration (no separate mid pose / descend phase).
PRE_GAP = HOVER_GAP + (HOVER_GAP + PRESS_DEPTH)
q_pre = ik([cap_center[0], cap_center[1], cap_top + PRE_GAP], q_home)
if q_pre is None:
    raise ValueError("IK pre-hover failed")
q_hover = ik([cap_center[0], cap_center[1], cap_top + HOVER_GAP], q_pre)
q_press = ik([cap_center[0], cap_center[1], cap_top - PRESS_DEPTH], q_hover if q_hover is not None else q_pre)
if q_hover is None or q_press is None:
    raise ValueError("IK failed for hover/press")


def lerp(a, b, s):
    return [a[i] + (b[i] - a[i]) * s for i in range(len(a))]


def smooth(s):
    s = max(0.0, min(1.0, s))
    return s * s * (3.0 - 2.0 * s)


# ---------------- main control loop ----------------
APPROACH_TICKS = 22
phase = "approach"
phase_tick = 0
calib_z = []
calib_z0 = None
lead = 3.0
fire_info = None
z_now = float(ee0.position[2])
q_start = q_home

while tick < max_steps - 2:
    if phase == "approach":
        q_cmd = lerp(q_start, q_pre, smooth((phase_tick + 1) / float(APPROACH_TICKS)))
    elif phase == "settle_pre":
        q_cmd = q_pre
    elif phase == "calib":
        q_cmd = q_hover
    elif phase == "wait":
        q_cmd = q_hover
    else:
        q_cmd = q_press
    reply = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=q_cmd, gripper_position=0.0, embodiment=emb)}))
    tick = tick + 1
    phase_tick = phase_tick + 1
    try:
        if bool(reply.terminated) or bool(reply.truncated):
            terminated = True
    except Exception:
        terminated = False
    if terminated:
        report["terminated_tick"] = tick
        report["terminated_phase"] = phase
        break
    try:
        rs = reply.observation.robot_state
        z_now = float(rs.end_effector_poses["primary"].position[2])
        q_now = [float(v) for v in rs.joint_positions["primary"]]
    except Exception:
        rs = get_robot_state()
        z_now = float(rs.end_effector_poses["primary"].position[2])
        q_now = [float(v) for v in rs.joint_positions["primary"]]

    # ---- track cube ----
    vx = 0.5 * (cube_c[0] - prev2_c[0])
    vy = 0.5 * (cube_c[1] - prev2_c[1])
    speed = math.hypot(vx, vy)
    pred = (cube_c[0] + vx, cube_c[1] + vy)
    det = detect_cube(pred, 6.0 + 2.0 * speed + 4.0 * missing)
    if det is None:
        missing = missing + 1
        new_c = pred
    else:
        missing = 0
        new_c = det
    prev2_c = prev_c
    prev_c = cube_c
    cube_c = new_c
    if missing == 0:
        d, dseg = update_visits(prev_c, cube_c, tick)
    else:
        d = math.hypot(cube_c[0] - target_c[0], cube_c[1] - target_c[1])
        dseg = d
    track.append([tick, round(cube_c[0], 1), round(cube_c[1], 1), round(d, 1), round(dseg, 1), missing,
                  phase[0:2], round(z_now, 4)])

    # ---- phase transitions ----
    if phase == "approach" and phase_tick >= APPROACH_TICKS:
        phase = "settle_pre"
        phase_tick = 0
    elif phase == "settle_pre":
        err = max([abs(q_now[i] - q_pre[i]) for i in range(7)])
        if err < 0.01 or phase_tick >= 8:
            phase = "calib"
            phase_tick = 0
            calib_z0 = z_now
            calib_z = []
    elif phase == "calib":
        # free-air jump of exactly the press size, straight down onto the hover pose
        calib_z.append(round(z_now, 4))
        err = max([abs(q_now[i] - q_hover[i]) for i in range(7)])
        if (phase_tick >= 6 and err < 0.01) or phase_tick >= 12:
            jump = calib_z0 - (cap_top + HOVER_GAP)
            frac = (HOVER_GAP + 0.007) / (HOVER_GAP + PRESS_DEPTH)
            lead_found = None
            for k in range(len(calib_z)):
                if calib_z0 - calib_z[k] >= frac * jump:
                    lead_found = k + 1
                    break
            if lead_found is not None:
                lead = float(lead_found)
            report["calib"] = {"z0": round(calib_z0, 4), "z": calib_z, "jump": round(jump, 4), "lead": lead}
            phase = "wait"
            phase_tick = 0
            report["wait_start_tick"] = tick
            report["hover_z"] = round(z_now, 4)
    elif phase == "wait":
        nvx = 0.5 * (cube_c[0] - prev2_c[0])
        nvy = 0.5 * (cube_c[1] - prev2_c[1])
        sp2 = nvx * nvx + nvy * nvy
        rx = cube_c[0] - target_c[0]
        ry = cube_c[1] - target_c[1]
        fire = False
        tca = None
        dca = None
        if sp2 > 0.25 and missing == 0:
            tca = -(rx * nvx + ry * nvy) / sp2
            dca = math.hypot(rx + nvx * tca, ry + nvy * tca)
            if vis["reaches"] == N - 1 and not vis["in_visit"] and tca >= -0.5 and tca <= lead and dca <= thr_reach:
                fire = True
            if vis["reaches"] == N and vis["in_visit"] and tca <= lead and d <= thr_exit:
                fire = True
        if fire:
            phase = "press"
            phase_tick = 0
            fire_info = {"tick": tick, "reaches": vis["reaches"], "d": round(d, 1), "tca": round(tca, 2),
                         "dca": round(dca, 1), "v": [round(nvx, 2), round(nvy, 2)], "lead": lead}
    elif phase == "press":
        if phase_tick >= 40:
            report["press_timeout"] = True
            break

report["track"] = track
report["reaches"] = vis["reaches"]
report["reach_ticks"] = reach_ticks
report["visit_log"] = visit_log
report["fire"] = fire_info
report["final_phase"] = phase
report["ticks"] = tick
report["terminated"] = terminated
report["target_c"] = [round(target_c[0], 1), round(target_c[1], 1)]
result = report
