# SwingXtimes v05 — execution/closed-loop: targets can MOVE during the episode (ep1/ep2: the due target
# jumped ~5-8 cm while the cube approached it), so every visit re-localizes its target from the front
# camera before descending and re-checks after the dwell. Built on v04 (language order, xy route,
# termination-safe stepping).
# History: v03 (right = robot's right = -y) succeeded on ep0; ep1/ep2 say "left ... then right"
# and terminated (failure) as soon as the cube reached the right target first.
import math
# Goal (from get_task_context().language): pick up the <color> cube, move it to the right-side
# target then the left-side target, repeat the right-to-left swing N times, put the cube down,
# press the button.
# All object locations come from SAM3 + depth at runtime. Motion: PyRoki IK + own joint-space
# interpolation through step() so the tick cost of every segment is controlled.

RIGHT_IS_POSITIVE_Y = False  # confirmed on ep0 (v02 vs v03): "right" = robot's right = -y in robot_base
MAX_JOINT_STEP = 0.05        # rad per control tick during interpolation
HOVER_GAP = 0.006            # held-cube bottom above target top at each visit
DWELL_TICKS = 3
RELOC_WAIT_TICKS = 4        # ticks held above a target before re-localizing it
TRANSIT_ABOVE_HOVER = 0.13  # v03 visits registered with the tool ~0.09 m above hover; stay above that

report = {"phases": [], "events": []}

def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]

def r3(seq):
    return [round(float(v), 4) for v in seq]

def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)

def yaw_quat(theta, base_quat):
    return qmul((math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)), base_quat)

# ---------------- language parsing ----------------
context = get_task_context()
language = str(context.language).lower()
report["language"] = language
NUMBER_WORDS = {"once": 1, "twice": 2, "thrice": 3, "one": 1, "two": 2, "three": 3, "four": 4,
                "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12}
tokens = language.replace(",", " ").replace(".", " ").split()
swings = None
for index, token in enumerate(tokens):
    if token in ("times", "time") and index > 0:
        prev = tokens[index - 1]
        if prev in NUMBER_WORDS:
            swings = NUMBER_WORDS[prev]
        elif prev.isdigit():
            swings = int(prev)
    if token in ("once", "twice", "thrice") and swings is None:
        swings = NUMBER_WORDS[token]
if swings is None:
    raise ValueError("could not parse swing count from language")
report["swings"] = swings
COLORS = ("red", "green", "blue", "yellow", "purple", "orange", "pink", "white", "black")
cube_color = None
for index, token in enumerate(tokens):
    if token in COLORS and index + 1 < len(tokens) and tokens[index + 1] in ("cube", "block"):
        cube_color = token
        break
report["cube_color"] = cube_color

def color_matches(rgb_px, color):
    r, g, b = float(rgb_px[0]), float(rgb_px[1]), float(rgb_px[2])
    if color is None:
        return True
    if color == "red":
        return r > 80 and r > 1.8 * g and r > 1.8 * b
    if color == "green":
        return g > 80 and g > 1.5 * r and g > 1.5 * b
    if color == "blue":
        return b > 80 and b > 1.5 * r and b > 1.5 * g
    if color == "yellow":
        return r > 100 and g > 100 and b < 0.6 * min(r, g)
    return True

# ---------------- perception ----------------
state0 = get_robot_state()
base = state0.base_frame
down = tuple(float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz)
obs0 = get_observation()
rgb = obs0.cameras["front"].rgb

def box_color(box):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    acc = [0.0, 0.0, 0.0]
    count = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy = min(255, max(0, cy + dy))
            xx = min(255, max(0, cx + dx))
            px = rgb[yy][xx]
            acc[0] += float(px[0])
            acc[1] += float(px[1])
            acc[2] += float(px[2])
            count += 1
    return [acc[0] / count, acc[1] / count, acc[2] / count]

def candidates(prompt, limit=6):
    out = []
    seg = segment_text("front", prompt)
    if not seg.ok:
        return out
    for cand in seg.segmentations[:limit]:
        if cand.box_xyxy is None:
            continue
        box = [float(v) for v in cand.box_xyxy]
        try:
            pc = mask_to_point_cloud(cand, "front", target_frame=base)
            pts = [(float(p[0]), float(p[1]), float(p[2])) for p in pc.points]
        except Exception:
            pts = []
        if len(pts) < 15:
            continue
        zs = [p[2] for p in pts]
        out.append({"prompt": prompt, "score": float(cand.score), "box": box,
                    "color": box_color(box), "pts": pts,
                    "z_hi": quantile(zs, 0.97), "z_lo": quantile(zs, 0.03),
                    "x_med": quantile([p[0] for p in pts], 0.5),
                    "y_med": quantile([p[1] for p in pts], 0.5)})
    return out

def top_face(pts, z_hi, band=0.006):
    top = [p for p in pts if p[2] >= z_hi - band]
    if len(top) < 8:
        top = pts
    return top

def fit_yaw(points):
    """Angle in [0, pi/2) minimizing the xy bounding-box area of the points."""
    best = None
    for k in range(18):
        a = k * (math.pi / 2.0) / 18.0
        ca = math.cos(a)
        sa = math.sin(a)
        us = [p[0] * ca + p[1] * sa for p in points]
        vs = [-p[0] * sa + p[1] * ca for p in points]
        span_u = quantile(us, 0.97) - quantile(us, 0.03)
        span_v = quantile(vs, 0.97) - quantile(vs, 0.03)
        area = span_u * span_v
        if best is None or area < best[0]:
            best = (area, a, max(span_u, span_v))
    return best[1], best[2]

# --- cube ---
cube = None
cube_pool = []
prompts = ["cube"]
if cube_color is not None:
    prompts = [cube_color + " cube", "cube"]
for prompt in prompts:
    cube_pool.extend(candidates(prompt))
for cand in cube_pool:
    height = cand["z_hi"] - cand["z_lo"]
    if not color_matches(cand["color"], cube_color):
        continue
    if cand["z_hi"] < 0.02 or cand["z_hi"] > 0.09:
        continue
    if cube is None or cand["score"] > cube["score"]:
        cube = cand
if cube is None:
    raise ValueError("cube not found")
cube_top = top_face(cube["pts"], cube["z_hi"])
cube_x = quantile([p[0] for p in cube_top], 0.5)
cube_y = quantile([p[1] for p in cube_top], 0.5)
cube_alpha, cube_width = fit_yaw(cube_top)
cube_height = max(cube_width, cube["z_hi"] - cube["z_lo"])
cube_height = min(0.06, max(0.03, cube_height))
cube_top_z = cube["z_hi"]
report["cube"] = {"xy": r3((cube_x, cube_y)), "top_z": round(cube_top_z, 4),
                  "alpha_deg": round(cube_alpha * 180 / math.pi, 1), "width": round(cube_width, 4), "height": round(cube_height, 4),
                  "score": round(cube["score"], 3), "color": r3(cube["color"]), "prompt": cube["prompt"]}

# --- targets (flat discs) ---
target_pool = candidates("circular target") + candidates("target")
targets = []
ordered_pool = []
remaining = list(target_pool)
while remaining:
    best_i = 0
    for i in range(len(remaining)):
        if remaining[i]["score"] > remaining[best_i]["score"]:
            best_i = i
    ordered_pool.append(remaining.pop(best_i))
for cand in ordered_pool:
    if cand["z_hi"] > 0.025 or cand["score"] < 0.2 or len(cand["pts"]) < 40:
        continue
    xy_span = (quantile([p[1] for p in cand["pts"]], 0.97) - quantile([p[1] for p in cand["pts"]], 0.03))
    if xy_span > 0.15:
        continue
    duplicate = False
    for kept in targets:
        if abs(kept["x_med"] - cand["x_med"]) + abs(kept["y_med"] - cand["y_med"]) < 0.05:
            duplicate = True
    if not duplicate:
        targets.append(cand)
if len(targets) < 2:
    raise ValueError("fewer than two targets found")
targets = targets[:2]
def target_center(cand):
    xs = [p[0] for p in cand["pts"]]
    ys = [p[1] for p in cand["pts"]]
    return ((quantile(xs, 0.03) + quantile(xs, 0.97)) / 2.0,
            (quantile(ys, 0.03) + quantile(ys, 0.97)) / 2.0, cand["z_hi"])
t_a = target_center(targets[0])
t_b = target_center(targets[1])
if (t_a[1] > t_b[1]) == RIGHT_IS_POSITIVE_Y:
    right_t, left_t = t_a, t_b
else:
    right_t, left_t = t_b, t_a
report["targets"] = {"right": r3(right_t), "left": r3(left_t)}
pos_left = language.find("left")
pos_right = language.find("right")
if pos_left < 0 and pos_right < 0:
    raise ValueError("no left/right side named in the language")
left_first = pos_left >= 0 and (pos_right < 0 or pos_left < pos_right)
if left_first:
    first_t, second_t = left_t, right_t
else:
    first_t, second_t = right_t, left_t
report["order"] = "left-first" if left_first else "right-first"
disc_radius = 0.0
for cand in targets:
    ys = [p[1] for p in cand["pts"]]
    disc_radius = max(disc_radius, (quantile(ys, 0.97) - quantile(ys, 0.03)) / 2.0)
report["disc_radius"] = round(disc_radius, 4)

# --- button ---
button = None
button_pool = candidates("button") + candidates("push button")
for cand in button_pool:
    if cand["z_hi"] < 0.02 or cand["z_hi"] > 0.12:
        continue
    col = cand["color"]
    if max(col) - min(col) > 40:   # skip coloured cubes; the button is grey/white
        continue
    if abs(cand["x_med"] - cube_x) + abs(cand["y_med"] - cube_y) < 0.03:
        continue
    if button is None or cand["score"] > button["score"]:
        button = cand
if button is not None:
    b_top = top_face(button["pts"], button["z_hi"], band=0.005)
    button_xyz = (quantile([p[0] for p in b_top], 0.5), quantile([p[1] for p in b_top], 0.5),
                  button["z_hi"])
    report["button"] = {"xyz": r3(button_xyz), "score": round(button["score"], 3)}
else:
    button_xyz = None
    report["button"] = None

# ---------------- motion helpers ----------------
gripper_cmd = [1.0]
ended = [False]
tick_count = [0]

def joints_now():
    return [float(v) for v in get_robot_state().joint_positions["primary"]]

def send(q):
    if ended[0]:
        return None
    state = get_robot_state()
    reply = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=list(q), gripper_position=gripper_cmd[0],
        embodiment=state.embodiment)}))
    tick_count[0] += 1
    term = False
    try:
        term = bool(reply.terminated) or bool(reply.truncated)
    except Exception:
        term = False
    if term:
        ended[0] = True
    return reply

def servo(q_goal, settle_max=12, tol=0.01, step_rad=MAX_JOINT_STEP, min_ticks=2):
    if ended[0]:
        return False
    q0 = joints_now()
    dist = max(abs(a - b) for a, b in zip(q_goal, q0))
    n = max(min_ticks, int(math.ceil(dist / step_rad)))
    for i in range(1, n + 1):
        s = float(i) / n
        s = 0.5 - 0.5 * math.cos(math.pi * s)
        send([a + (b - a) * s for a, b in zip(q0, q_goal)])
        if ended[0]:
            return False
    for k in range(settle_max):
        err = max(abs(a - b) for a, b in zip(q_goal, joints_now()))
        if err < tol:
            return True
        send(q_goal)
        if ended[0]:
            return False
    return True

seed_q = [joints_now()]
def ik(position, quat):
    res = solve_ik(Pose(tuple(float(v) for v in position), tuple(quat), base), seed_joints=seed_q[0])
    if not res.ok:
        res = solve_ik(Pose(tuple(float(v) for v in position), tuple(quat), base))
    if not res.ok:
        raise ValueError("IK failed at " + str(r3(position)))
    q = [float(v) for v in res.joint_positions]
    seed_q[0] = q
    return q

def phase(name):
    try:
        t = float(get_robot_state().timestamp_s)
    except Exception:
        t = -1.0
    report["phases"].append({"name": name, "ticks": tick_count[0], "t": round(t, 2)})

def seg_point_dist(a, b, p):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    den = dx * dx + dy * dy
    t = 0.0 if den < 1e-12 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / den))
    cx = a[0] + t * dx - p[0]
    cy = a[1] + t * dy - p[1]
    return math.sqrt(cx * cx + cy * cy)

def route_xy(start, goal, avoid, clearance):
    """Intermediate xy waypoints so the carried cube never passes over an `avoid` target."""
    options = [[]]
    for c in avoid:
        vx = goal[0] - start[0]
        vy = goal[1] - start[1]
        norm = math.sqrt(vx * vx + vy * vy) + 1e-9
        nx, ny = -vy / norm, vx / norm
        for side in (1.0, -1.0):
            for scale in (1.0, 1.5, 2.0):
                off = clearance * scale * side
                options.append([(c[0] + nx * off, c[1] + ny * off)])
    for wps in options:
        pts = [start] + wps + [goal]
        clear = True
        for i in range(len(pts) - 1):
            for c in avoid:
                if seg_point_dist(pts[i], pts[i + 1], c) < clearance:
                    clear = False
        if clear:
            return wps
    return []

def xy_dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)

def relocate(est, other):
    """Re-segment the flat discs on the front camera; return the one matching `est` (or None)."""
    best = None
    for prompt in ("circular target", "target"):
        seg = segment_text("front", prompt)
        if not seg.ok:
            continue
        for cand in seg.segmentations[:6]:
            try:
                pc = mask_to_point_cloud(cand, "front", target_frame=base)
            except Exception:
                continue
            low = [(float(p[0]), float(p[1]), float(p[2])) for p in pc.points if float(p[2]) < 0.03]
            if len(low) < 25:
                continue
            xs = [p[0] for p in low]
            ys = [p[1] for p in low]
            if quantile(ys, 0.97) - quantile(ys, 0.03) > 0.15:
                continue
            c = ((quantile(xs, 0.03) + quantile(xs, 0.97)) / 2.0, (quantile(ys, 0.03) + quantile(ys, 0.97)) / 2.0)
            d = xy_dist(c, est)
            d_other = xy_dist(c, other)
            if d < 0.15 and d < d_other and (best is None or d < best[0]):
                best = (d, c)
        if best is not None:
            break
    if best is None:
        return None
    return best[1]

def run_motion():
    # ---------------- grasp ----------------
    theta = cube_alpha
    while theta > math.pi / 4:
        theta -= math.pi / 2
    while theta <= -math.pi / 4:
        theta += math.pi / 2
    grasp_quat = yaw_quat(theta, down)
    grasp_z = cube_top_z - 0.02
    q_pre = ik((cube_x, cube_y, cube_top_z + 0.08), grasp_quat)
    q_grasp = ik((cube_x, cube_y, grasp_z), grasp_quat)
    phase("start")
    servo(q_pre)
    servo(q_grasp, step_rad=0.03)
    phase("at_grasp")
    close_res = close_gripper()
    gripper_cmd[0] = 0.0
    try:
        report["close"] = {"ok": bool(close_res.ok), "steps": int(close_res.steps_executed)}
        tick_count[0] += int(close_res.steps_executed)
    except Exception:
        report["close"] = "unknown"

    # held-cube geometry: cube bottom sits this far below the tool point
    hold_drop = grasp_z - (cube_top_z - cube_height)
    disc_z = max(first_t[2], second_t[2])
    hover_z = disc_z + HOVER_GAP + hold_drop
    transit_z = hover_z + TRANSIT_ABOVE_HOVER
    ests = [(first_t[0], first_t[1]), (second_t[0], second_t[1])]

    servo(ik((cube_x, cube_y, transit_z), grasp_quat), step_rad=0.04)
    g_after = float(get_robot_state().gripper_positions["primary"]) if not ended[0] else -1.0
    report["gripper_after_lift"] = round(g_after, 3)
    phase("lifted")
    for wp in route_xy((cube_x, cube_y), ests[0], [ests[1]], disc_radius + 0.07):
        servo(ik((wp[0], wp[1], transit_z), grasp_quat), settle_max=2, tol=0.05)
        report["events"].append({"route_wp": r3(wp), "ticks": tick_count[0]})

    def go_high(xy):
        servo(ik((xy[0], xy[1], transit_z), grasp_quat), settle_max=6, tol=0.02)

    def visit(idx, n, final):
        est = ests[idx]
        other = ests[1 - idx]
        log = {"visit": "first" if idx == 0 else "second", "n": n, "moves": []}
        for attempt in range(2):
            for k in range(4):
                if ended[0]:
                    break
                go_high(est)
                for t in range(RELOC_WAIT_TICKS):
                    send(seed_q[0])
                new = relocate(est, other)
                if new is None:
                    log["moves"].append("none")
                    break
                moved = xy_dist(new, est)
                log["moves"].append(round(moved, 3))
                est = new
                if moved < 0.012:
                    break
            ests[idx] = est
            if ended[0]:
                break
            q_low = ik((est[0], est[1], hover_z), grasp_quat)
            servo(q_low, step_rad=0.04)
            for k in range(DWELL_TICKS):
                send(q_low)
            post = relocate(est, other)
            if post is not None and xy_dist(post, est) > 0.03 and not ended[0]:
                log["moves"].append("low:" + str(round(xy_dist(post, est), 3)))
                est = post
                ests[idx] = est
                go_high(est)
                continue
            break
        log["xy"] = r3(est)
        log["ticks"] = tick_count[0]
        log["ended"] = ended[0]
        report["events"].append(log)
        if not final:
            go_high(est)

    total = 2 * swings
    for v in range(total):
        if ended[0]:
            break
        idx = v % 2
        if v > 0:
            mid = ((ests[0][0] + ests[1][0]) / 2.0, (ests[0][1] + ests[1][1]) / 2.0)
            servo(ik((mid[0], mid[1], transit_z), grasp_quat), settle_max=2, tol=0.05)
        visit(idx, v // 2 + 1, v == total - 1)
    phase("swings_done")
    last = ests[(total - 1) % 2]

    # ---------------- put down on the spot of the last visit ----------------
    if not ended[0]:
        q_place = ik((last[0], last[1], disc_z + hold_drop + 0.002), grasp_quat)
        servo(q_place, step_rad=0.02)
        gripper_cmd[0] = 1.0
        open_res = open_gripper()
        try:
            report["open"] = {"ok": bool(open_res.ok), "steps": int(open_res.steps_executed)}
            tick_count[0] += int(open_res.steps_executed)
        except Exception:
            report["open"] = "unknown"
        for k in range(4):
            send(q_place)
        phase("released")
        servo(ik((last[0], last[1], transit_z), grasp_quat), step_rad=0.04)
        if not ended[0]:
            report["gripper_after_release_lift"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
        gripper_cmd[0] = 0.0          # close the empty hand during transit, to press with fingertips

    # ---------------- press button (re-detected near its first estimate) ----------------
    bxyz = button_xyz
    if not ended[0] and bxyz is not None:
        try:
            seg = segment_text("front", "button")
            best = None
            if seg.ok:
                for cand in seg.segmentations[:6]:
                    pc = mask_to_point_cloud(cand, "front", target_frame=base)
                    pts = [(float(p[0]), float(p[1]), float(p[2])) for p in pc.points]
                    if len(pts) < 15:
                        continue
                    zh = quantile([p[2] for p in pts], 0.97)
                    if zh < 0.02 or zh > 0.12:
                        continue
                    top = top_face(pts, zh, band=0.005)
                    c = (quantile([p[0] for p in top], 0.5), quantile([p[1] for p in top], 0.5), zh)
                    d = xy_dist(c, bxyz)
                    if d < 0.08 and xy_dist(c, last) > 0.05 and (best is None or d < best[0]):
                        best = (d, c)
            if best is not None:
                report["button_redetect"] = {"xyz": r3(best[1]), "shift": round(best[0], 3)}
                bxyz = best[1]
        except Exception as exc:
            report["button_redetect"] = "error: " + str(exc)
    if not ended[0] and bxyz is not None:
        q_btn_up = ik((bxyz[0], bxyz[1], bxyz[2] + 0.08), down)
        servo(q_btn_up)
        for k in range(6):
            send(q_btn_up)
        phase("above_button")
        depth = 0.0
        while depth < 0.04 and not ended[0]:
            depth += 0.01
            q_press = ik((bxyz[0], bxyz[1], bxyz[2] + 0.02 - depth), down)
            servo(q_press, step_rad=0.02, settle_max=6)
            report["events"].append({"press_depth": round(depth, 3), "ticks": tick_count[0],
                                     "ended": ended[0]})
        phase("pressed")
        if not ended[0]:
            servo(q_btn_up, step_rad=0.03)
            for k in range(10):
                if ended[0]:
                    break
                send(q_btn_up)

try:
    run_motion()
except Exception as exc:
    report["motion_error"] = str(exc)
report["ended"] = ended[0]
report["ticks"] = tick_count[0]
try:
    report["final_ee"] = r3(get_robot_state().end_effector_poses["primary"].position)
except Exception:
    report["final_ee"] = "unavailable"
result = report
