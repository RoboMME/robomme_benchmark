# BinFill v02 -- v01 + FIFO (most downstream catchable needed cube), belt-end estimate, faster cycle.
# Strategy: parse the goal into {colour: count}; localise bin and button once from the
# front camera; measure the belt velocity from two front segmentations; then repeatedly
# pick a cube of a still-needed colour (colour verified from mask pixels by hue), with
# motion prediction, hover re-observation from the wrist camera, top-down yaw-aligned
# grasp, drop above the bin; finally press the button.
# Every task-dependent quantity is derived at runtime (goal text, RGB-D, robot state).
import math

FRONT = "front"
WRIST = "wrist"
report = {"version": "v02", "events": [], "picks": []}


def log(msg):
    if len(report["events"]) < 400:
        report["events"].append(msg)


# ---------------------------------------------------------------- metadata / goal
ctx = get_task_context()
goal = str(ctx.language)
report["goal"] = goal
max_steps = 1000
try:
    meta = robomme_plus.get_task_metadata()
    report["task_metadata"] = {}
    for key in meta:
        try:
            report["task_metadata"][str(key)] = str(meta[key])[:300]
        except Exception:
            pass
    try:
        max_steps = int(meta["max_steps"])
    except Exception:
        pass
except Exception as exc:
    report["task_metadata_error"] = str(exc)[:300]
try:
    cmeta = robomme_plus.get_controller_metadata()
    report["controller_metadata"] = {}
    for key in cmeta:
        report["controller_metadata"][str(key)] = str(cmeta[key])[:300]
except Exception as exc:
    report["controller_metadata_error"] = str(exc)[:300]
report["max_steps"] = max_steps

NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
           "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
COLOR_WORDS = {"red": "red", "orange": "orange", "yellow": "yellow", "green": "green",
               "cyan": "cyan", "blue": "blue", "purple": "magenta", "magenta": "magenta",
               "pink": "magenta", "violet": "magenta"}
# hue windows in degrees (measured offline on recording pixels; table wood is 10-29 deg)
HUE_WINDOWS = {"red": [(340.0, 360.0), (0.0, 9.0)], "orange": [(33.0, 50.0)],
               "yellow": [(50.0, 78.0)], "green": [(88.0, 152.0)], "cyan": [(160.0, 205.0)],
               "blue": [(212.0, 262.0)], "magenta": [(272.0, 330.0)]}


def parse_goal(text):
    clean = ""
    for ch in text.lower():
        if ch.isalnum():
            clean += ch
        else:
            clean += " "
    words = clean.split()
    cut = len(words)
    for i in range(len(words)):
        if words[i] == "into" or words[i] == "then":
            cut = i
            break
    need = {}
    order = []
    i = 0
    while i < cut:
        w = words[i]
        count = None
        if w in NUMBERS:
            count = NUMBERS[w]
        elif w.isdigit():
            count = int(w)
        if count is not None:
            j = i + 1
            while j < cut and j <= i + 3:
                if words[j] in COLOR_WORDS:
                    col = COLOR_WORDS[words[j]]
                    need[col] = need.get(col, 0) + count
                    if col not in order:
                        order.append(col)
                    break
                j += 1
        i += 1
    return need, order


need, color_order = parse_goal(goal)
report["need"] = dict(need)
if len(need) == 0:
    log("goal parse found no colour counts")


# ---------------------------------------------------------------- small helpers
def now_s():
    try:
        return float(get_robot_state().timestamp_s)
    except Exception:
        return -1.0


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


ended = {"flag": False, "why": ""}


def mark(res, why):
    if is_terminal(res):
        ended["flag"] = True
        ended["why"] = why
    return res


def hold(ticks, grip=None):
    state = get_robot_state()
    g = state.gripper_positions["primary"] if grip is None else grip
    for k in range(ticks):
        res = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=state.joint_positions["primary"],
            gripper_position=g, embodiment=state.embodiment)}))
        mark(res, "hold")
        if ended["flag"]:
            return


def quantile(values, frac):
    ordered = sorted(values)
    if len(ordered) == 0:
        return 0.0
    return ordered[int((len(ordered) - 1) * frac)]


def rgb_to_hue_sat_val(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    v = mx / 255.0
    if mx <= 0:
        return 0.0, 0.0, 0.0
    s = (mx - mn) / float(mx)
    if mx == mn:
        return 0.0, s, v
    d = float(mx - mn)
    if mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * (((b - r) / d) + 2.0)
    else:
        h = 60.0 * (((r - g) / d) + 4.0)
    if h < 0:
        h += 360.0
    return h, s, v


def hue_class(h):
    for name in HUE_WINDOWS:
        for win in HUE_WINDOWS[name]:
            if h >= win[0] and h < win[1]:
                return name
    return None


def mask_color_votes(cand, rgb):
    """Per-pixel hue classification inside the candidate bbox+mask."""
    votes = {}
    total = 0
    try:
        box = [float(v) for v in cand.box_xyxy]
    except Exception:
        return votes, 0
    mask = None
    try:
        mask = cand.mask
    except Exception:
        mask = None
    if mask is None:
        # no mask exposed: use the central 60% of the box
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        box = [box[0] + 0.2 * bw, box[1] + 0.2 * bh, box[2] - 0.2 * bw, box[3] - 0.2 * bh]
    h_img = len(rgb)
    w_img = len(rgb[0])
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(w_img - 1, int(box[2]) + 1)
    y2 = min(h_img - 1, int(box[3]) + 1)
    stride = 1
    if (x2 - x1) * (y2 - y1) > 6000:
        stride = 3
    for y in range(y1, y2 + 1, stride):
        for x in range(x1, x2 + 1, stride):
            inside = True
            if mask is not None:
                try:
                    inside = bool(mask[y][x])
                except Exception:
                    inside = True
            if not inside:
                continue
            px = rgb[y][x]
            r = int(px[0])
            g = int(px[1])
            b = int(px[2])
            h, s, v = rgb_to_hue_sat_val(r, g, b)
            total += 1
            if s < 0.45 or v < 0.12:
                continue
            c = hue_class(h)
            if c is not None:
                votes[c] = votes.get(c, 0) + 1
    return votes, total


def dominant_color(votes, total):
    best = None
    best_n = 0
    for c in votes:
        if votes[c] > best_n:
            best_n = votes[c]
            best = c
    if best is None or total <= 0:
        return None, 0.0
    return best, best_n / float(total)


def cloud_points(cand, cam):
    base = get_robot_state().base_frame
    pc = None
    try:
        pc = mask_to_point_cloud(cand, cam, target_frame=base)
    except Exception:
        try:
            pc = mask_to_point_cloud(cand.mask, cam, target_frame=base)
        except Exception:
            pc = None
    if pc is None:
        return []
    pts = []
    try:
        for p in pc.points:
            pts.append((float(p[0]), float(p[1]), float(p[2])))
    except Exception:
        return []
    return pts


def cube_from_points(pts):
    """Top-face centre, top z, estimated height and yaw (mod 90 deg) of a cube cloud."""
    zs = [p[2] for p in pts]
    z_hi = quantile(zs, 0.92)
    z_lo = quantile(zs, 0.05)
    top = [p for p in pts if p[2] >= z_hi - 0.012]
    if len(top) < 4:
        top = pts
    cx = quantile([p[0] for p in top], 0.5)
    cy = quantile([p[1] for p in top], 0.5)
    height = z_hi - z_lo
    yaw = 0.0
    best_area = None
    if len(top) >= 8:
        for k in range(18):
            a = math.radians(5.0 * k)
            ca = math.cos(a)
            sa = math.sin(a)
            us = [(p[0] - cx) * ca + (p[1] - cy) * sa for p in top]
            vs = [-(p[0] - cx) * sa + (p[1] - cy) * ca for p in top]
            area = (quantile(us, 0.98) - quantile(us, 0.02)) * (quantile(vs, 0.98) - quantile(vs, 0.02))
            if best_area is None or area < best_area:
                best_area = area
                yaw = a
    ex = quantile([p[0] for p in pts], 0.95) - quantile([p[0] for p in pts], 0.05)
    ey = quantile([p[1] for p in pts], 0.95) - quantile([p[1] for p in pts], 0.05)
    return {"x": cx, "y": cy, "z_top": z_hi, "z_low": z_lo, "height": height,
            "yaw": yaw, "n": len(pts), "ext_xy": max(ex, ey)}


def find_cubes(cam, color, max_cands=10):
    """Candidates of a colour on one camera; colour verified by pixel hue."""
    obs = get_observation()
    t_obs = float(obs.timestamp_s)
    rgb = obs.cameras[cam].rgb
    found = []
    prompts = []
    if color is None:
        prompts = ["cube"]
    else:
        prompts = [color + " cube", "cube"]
    seen_boxes = []
    for prompt in prompts:
        seg = segment_text(cam, prompt)
        if not seg.ok:
            continue
        k = 0
        for cand in seg.segmentations:
            if k >= max_cands:
                break
            k += 1
            try:
                box = [float(v) for v in cand.box_xyxy]
            except Exception:
                continue
            dup = False
            for sb in seen_boxes:
                if abs(sb[0] - box[0]) < 3 and abs(sb[1] - box[1]) < 3 and abs(sb[2] - box[2]) < 3:
                    dup = True
            if dup:
                continue
            seen_boxes.append(box)
            votes, total = mask_color_votes(cand, rgb)
            dom, frac = dominant_color(votes, total)
            if color is not None and dom != color:
                continue
            if color is not None and frac < 0.2:
                continue
            pts = cloud_points(cand, cam)
            if len(pts) < 5:
                continue
            cube = cube_from_points(pts)
            if cube["ext_xy"] > 0.12:
                continue
            cube["color"] = dom
            cube["frac"] = round(frac, 2)
            cube["score"] = round(float(cand.score), 3)
            cube["t"] = t_obs
            cube["box"] = [round(v, 1) for v in box]
            found.append(cube)
        if len(found) > 0 and color is not None:
            break
    return found, t_obs


def locate_static(prompts, max_ext, max_zext):
    best = None
    for prompt in prompts:
        seg = segment_text(FRONT, prompt)
        if not seg.ok:
            continue
        k = 0
        for cand in seg.segmentations:
            if k >= 6:
                break
            k += 1
            pts = cloud_points(cand, FRONT)
            if len(pts) < 10:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [p[2] for p in pts]
            ex = quantile(xs, 0.97) - quantile(xs, 0.03)
            ey = quantile(ys, 0.97) - quantile(ys, 0.03)
            ez = quantile(zs, 0.97) - quantile(zs, 0.03)
            if ex > max_ext or ey > max_ext or ez > max_zext:
                continue
            z_top = quantile(zs, 0.95)
            top = [p for p in pts if p[2] >= z_top - 0.01]
            item = {"prompt": prompt, "score": float(cand.score),
                    "x": 0.5 * (quantile(xs, 0.05) + quantile(xs, 0.95)),
                    "y": 0.5 * (quantile(ys, 0.05) + quantile(ys, 0.95)),
                    "top_x": quantile([p[0] for p in top], 0.5),
                    "top_y": quantile([p[1] for p in top], 0.5),
                    "z_top": z_top, "z_low": quantile(zs, 0.03),
                    "ext": [round(ex, 3), round(ey, 3), round(ez, 3)], "n": len(pts)}
            if best is None or item["score"] > best["score"]:
                best = item
        if best is not None:
            return best
    return best


# ---------------------------------------------------------------- orientation helpers
def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def rotate_vec(q, v):
    w, x, y, z = q
    qv = (0.0, v[0], v[1], v[2])
    conj = (w, -x, -y, -z)
    r = quat_mul(quat_mul(q, qv), conj)
    return (r[1], r[2], r[3])


def downward_quat(jaw_yaw):
    # 180 deg about x gives a straight-down hand whose jaw (local y) points to world -y
    # (yaw -90 deg); add a world-z rotation so that the jaw axis points along jaw_yaw.
    psi = jaw_yaw + math.pi / 2.0
    qz = (math.cos(psi / 2.0), 0.0, 0.0, math.sin(psi / 2.0))
    return quat_mul(qz, (0.0, 1.0, 0.0, 0.0))


def wrap_half_pi(a):
    # wrap to (-pi/4, pi/4] modulo pi/2 (cube symmetry)
    while a > math.pi / 4.0:
        a -= math.pi / 2.0
    while a <= -math.pi / 4.0:
        a += math.pi / 2.0
    return a


home_state = get_robot_state()
base = home_state.base_frame
home_pose = home_state.end_effector_poses["primary"]
home_q = tuple(float(v) for v in home_pose.quaternion_wxyz)
home_jaw = rotate_vec(home_q, (0.0, 1.0, 0.0))
home_down = rotate_vec(home_q, (0.0, 0.0, 1.0))
home_jaw_yaw = math.atan2(home_jaw[1], home_jaw[0])
report["home"] = {"pos": [round(float(v), 4) for v in home_pose.position],
                  "quat": [round(v, 4) for v in home_q],
                  "hand_z_world": [round(v, 3) for v in home_down],
                  "jaw_yaw_deg": round(math.degrees(home_jaw_yaw), 1),
                  "gripper": float(home_state.gripper_positions["primary"]),
                  "t": float(home_state.timestamp_s)}


def grasp_quat(cube_yaw):
    # choose the jaw yaw equivalent to the cube face direction closest to the home jaw yaw
    delta = wrap_half_pi(cube_yaw - home_jaw_yaw)
    return downward_quat(home_jaw_yaw + delta)


DURATIONS = {"hover": 1.05, "descend": 0.45, "close": 0.15}


def timed_move(pos, quat, tol, max_steps_move, label):
    t0 = now_s()
    res = move_to_pose(Pose(position=[pos[0], pos[1], pos[2]], quaternion_wxyz=list(quat),
                            frame=base), tolerance=tol, max_steps=max_steps_move)
    mark(res, label)
    t1 = now_s()
    info = {"label": label, "ok": bool(res.ok), "dt": round(t1 - t0, 3),
            "target": [round(pos[0], 4), round(pos[1], 4), round(pos[2], 4)]}
    try:
        info["steps"] = int(res.steps_executed)
    except Exception:
        pass
    if not res.ok:
        try:
            info["error"] = str(res.error)[:160]
        except Exception:
            pass
    try:
        p = get_robot_state().end_effector_poses["primary"].position
        info["reached"] = [round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)]
    except Exception:
        pass
    return res, info, t1 - t0


# ---------------------------------------------------------------- scene: bin + button
open_gripper()
bin_info = locate_static(["bin", "open box", "container"], 0.30, 0.20)
button_info = locate_static(["button", "white button", "round button"], 0.12, 0.08)
report["bin"] = bin_info
report["button"] = button_info


def locate_belt():
    """Largest flat segment under the cubes: kept as a sparse xy sample for the belt end."""
    best = None
    for prompt in ["conveyor belt", "gray mat", "gray board"]:
        seg = segment_text(FRONT, prompt)
        if not seg.ok:
            continue
        k = 0
        for cand in seg.segmentations:
            if k >= 4:
                break
            k += 1
            pts = cloud_points(cand, FRONT)
            if len(pts) < 50:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [p[2] for p in pts]
            ex = quantile(xs, 0.98) - quantile(xs, 0.02)
            ey = quantile(ys, 0.98) - quantile(ys, 0.02)
            ez = quantile(zs, 0.98) - quantile(zs, 0.02)
            if max(ex, ey) < 0.25 or ez > 0.05:
                continue
            if best is None or ex * ey > best["area"]:
                stride = max(1, len(pts) // 400)
                sample = []
                for i in range(0, len(pts), stride):
                    sample.append((pts[i][0], pts[i][1]))
                best = {"prompt": prompt, "area": ex * ey, "ext": [round(ex, 3), round(ey, 3), round(ez, 3)],
                        "z": quantile(zs, 0.5), "sample": sample}
        if best is not None:
            return best
    return None


belt_info = locate_belt()
if belt_info is not None:
    report["belt"] = {"prompt": belt_info["prompt"], "ext": belt_info["ext"], "z": round(belt_info["z"], 4)}
else:
    report["belt"] = None
log("bin/button/belt located at t=" + str(now_s()))


# ---------------------------------------------------------------- belt velocity model
# Velocity samples (m/s) come from (a) paired front observations of the target colour a
# known number of ticks apart and (b) each pick's front->wrist observation pair of the same
# cube. The model is the component-wise median of the last samples (a static table gives ~0).
vel_state = {"samples": [], "v": (0.0, 0.0)}
report["velocity_samples"] = []


def add_velocity_sample(vx, vy, source):
    if abs(vx) > 0.25 or abs(vy) > 0.25:
        return
    vel_state["samples"].append((vx, vy))
    recent = vel_state["samples"][-7:]
    vel_state["v"] = (quantile([s[0] for s in recent], 0.5), quantile([s[1] for s in recent], 0.5))
    report["velocity_samples"].append({"v": [round(vx, 4), round(vy, 4)], "src": source,
                                       "model": [round(vel_state["v"][0], 4), round(vel_state["v"][1], 4)]})


def match_displacements(a, b, dt, radius):
    dxs = []
    dys = []
    for ca in a:
        best = None
        bd = None
        for cb in b:
            d = math.sqrt((ca["x"] - cb["x"]) ** 2 + (ca["y"] - cb["y"]) ** 2)
            if bd is None or d < bd:
                bd = d
                best = cb
        if best is not None and bd < radius:
            dxs.append((best["x"] - ca["x"]) / dt)
            dys.append((best["y"] - ca["y"]) / dt)
    return dxs, dys


# ---------------------------------------------------------------- pick loop
remaining = dict(need)
placed = {}
fails = {}
attempt = 0
RESERVE_TICKS = 110
TRANSIT_CLEAR = 0.14
MOVING_SPEED = 0.01


def ticks_used():
    return int(round(now_s() * 20.0))


def predict(cube, t_future):
    dt = t_future - cube["t"]
    vel = vel_state["v"]
    return (cube["x"] + vel[0] * dt, cube["y"] + vel[1] * dt)


def jaw_quat(cube_yaw):
    vel = vel_state["v"]
    speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
    if speed > MOVING_SPEED:
        # close ACROSS the motion: the moving cube slides between the open fingers
        want = math.atan2(vel[1], vel[0]) + math.pi / 2.0
        d = want - home_jaw_yaw
        while d > math.pi / 2.0:
            d -= math.pi
        while d <= -math.pi / 2.0:
            d += math.pi
        return downward_quat(home_jaw_yaw + d)
    return grasp_quat(cube_yaw)


def in_bin(c):
    if bin_info is None:
        return False
    return abs(c["x"] - bin_info["x"]) < 0.08 and abs(c["y"] - bin_info["y"]) < 0.08


def pick_target():
    """Choose the most upstream, isolated, resting cube of a still-needed colour."""
    others, t_all = find_cubes(FRONT, None, 16)
    pool = []
    for col in color_order:
        if remaining.get(col, 0) <= 0:
            continue
        cands, t_obs = find_cubes(FRONT, col, 10)
        for c in cands:
            if in_bin(c):
                continue
            pool.append(c)
    if len(pool) == 0:
        return None, 0
    tops = sorted([c["z_top"] for c in pool])
    z_ref = tops[len(tops) // 2]
    if len(vel_state["samples"]) == 0:
        # first measurement of motion: re-observe the same colour 10 ticks later
        hold(10)
        if ended["flag"]:
            return None, 0
        later = []
        for col in color_order:
            if remaining.get(col, 0) <= 0:
                continue
            cands, t_obs = find_cubes(FRONT, col, 10)
            for c in cands:
                if not in_bin(c):
                    later.append(c)
        if len(later) > 0:
            dt = later[0]["t"] - pool[0]["t"]
            if dt > 0:
                dxs, dys = match_displacements(pool, later, dt, 0.05)
                if len(dxs) > 0:
                    add_velocity_sample(quantile(dxs, 0.5), quantile(dys, 0.5), "front_pair")
            pool = later
    vel = vel_state["v"]
    speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
    ee = get_robot_state().end_effector_poses["primary"].position
    end_proj = None
    if speed > MOVING_SPEED:
        ux = vel[0] / speed
        uy = vel[1] / speed
        # furthest downstream point at which a cube has been observed on the belt (lower bound of the end)
        for c in pool + others:
            pr = c["x"] * ux + c["y"] * uy
            if vel_state.get("max_seen") is None or pr > vel_state["max_seen"]:
                vel_state["max_seen"] = pr
        seen = vel_state.get("max_seen")
        if belt_info is not None:
            end_proj = quantile([p[0] * ux + p[1] * uy for p in belt_info["sample"]], 0.98)
            if seen is not None and end_proj < seen:
                end_proj = seen
        elif seen is not None:
            end_proj = seen + 0.045
        report["belt_end_proj"] = end_proj
        report["max_seen_proj"] = seen
    lead_t = DURATIONS["hover"] + DURATIONS["descend"] + 0.4
    best = None
    best_score = None
    for c in pool:
        if c["z_top"] > z_ref + 0.03:
            continue  # still falling / stacked
        crowded = 0
        for o in others:
            dx = abs(o["x"] - c["x"])
            dy = abs(o["y"] - c["y"])
            if dx < 0.012 and dy < 0.012:
                continue  # the same cube
            if speed > MOVING_SPEED:
                # fingers sit across the motion direction; neighbours in that lane collide
                across = abs((o["x"] - c["x"]) * (-uy) + (o["y"] - c["y"]) * ux)
                along = abs((o["x"] - c["x"]) * ux + (o["y"] - c["y"]) * uy)
                if across < 0.075 and along < 0.045:
                    crowded += 1
            else:
                if dx < 0.07 and dy < 0.07:
                    crowded += 1
        if speed > MOVING_SPEED:
            px, py = predict(c, now_s() + lead_t)
            down = px * ux + py * uy
            if end_proj is not None and down > end_proj - 0.045:
                continue  # would leave the belt before the grasp closes
            score = -down + 0.10 * crowded
        else:
            score = math.sqrt((c["x"] - float(ee[0])) ** 2 + (c["y"] - float(ee[1])) ** 2) + 0.10 * crowded
        if best_score is None or score < best_score:
            best_score = score
            best = c
    if best is not None:
        best["crowded_score"] = round(best_score, 3)
    return best, len(pool)


idle_rounds = 0
while not ended["flag"]:
    left = 0
    for col in remaining:
        left += max(0, remaining[col])
    if left <= 0:
        break
    if ticks_used() > max_steps - RESERVE_TICKS:
        log("tick budget reserve reached; stop picking")
        break
    if attempt >= 18:
        log("attempt cap reached")
        break
    target, n_pool = pick_target()
    if ended["flag"]:
        break
    if target is None:
        idle_rounds += 1
        if idle_rounds > 40:
            log("no candidate for 40 rounds")
            break
        hold(8)
        continue
    attempt += 1
    rec = {"attempt": attempt, "color": target["color"], "t_start": now_s(), "pool": n_pool,
           "v_model": [round(vel_state["v"][0], 4), round(vel_state["v"][1], 4)],
           "obs": {"x": round(target["x"], 4), "y": round(target["y"], 4),
                   "z_top": round(target["z_top"], 4), "h": round(target["height"], 4),
                   "yaw_deg": round(math.degrees(target["yaw"]), 1), "score": target["score"],
                   "frac": target["frac"], "t": target["t"]}, "moves": []}
    report["picks"].append(rec)
    height = target["height"]
    if height < 0.025 or height > 0.06:
        height = 0.04
    z_top = target["z_top"]
    hover_z = z_top + 0.09
    q = jaw_quat(target["yaw"])
    hx, hy = predict(target, now_s() + DURATIONS["hover"])
    res, info, dur = timed_move((hx, hy, hover_z), q, 0.015, 200, "hover")
    rec["moves"].append(info)
    if ended["flag"]:
        break
    if info["ok"] and dur > 0:
        DURATIONS["hover"] = 0.5 * DURATIONS["hover"] + 0.5 * dur
    # re-observe from the wrist camera at hover; the candidate nearest the prediction
    cands, t_w = find_cubes(WRIST, target["color"], 8)
    px, py = predict(target, t_w)
    chosen = None
    cd = None
    for c in cands:
        d = math.sqrt((c["x"] - px) ** 2 + (c["y"] - py) ** 2)
        if cd is None or d < cd:
            cd = d
            chosen = c
    gate = 0.04
    if len(vel_state["samples"]) < 2:
        gate = 0.07
    if chosen is not None and cd < gate:
        rec["wrist"] = {"x": round(chosen["x"], 4), "y": round(chosen["y"], 4),
                        "z_top": round(chosen["z_top"], 4), "yaw_deg": round(math.degrees(chosen["yaw"]), 1),
                        "d_pred": round(cd, 4), "n": chosen["n"], "t": t_w}
        dtw = t_w - target["t"]
        if dtw > 0.3:
            add_velocity_sample((chosen["x"] - target["x"]) / dtw, (chosen["y"] - target["y"]) / dtw,
                                "front_wrist")
        ref = chosen
        z_top_use = chosen["z_top"]
        if abs(z_top_use - z_top) > 0.025:
            z_top_use = z_top
        q = jaw_quat(chosen["yaw"])
    else:
        rec["wrist"] = {"n_cands": len(cands), "d": cd}
        ref = target
        z_top_use = z_top
    grasp_z = z_top_use - 0.5 * height
    t_grasp = now_s() + DURATIONS["descend"] + 0.5 * DURATIONS["close"]
    gx, gy = predict(ref, t_grasp)
    rec["grasp_target"] = [round(gx, 4), round(gy, 4), round(grasp_z, 4)]
    res, info, dur = timed_move((gx, gy, grasp_z), q, 0.01, 30, "descend")
    rec["moves"].append(info)
    if ended["flag"]:
        break
    if info["ok"] and dur > 0:
        DURATIONS["descend"] = 0.5 * DURATIONS["descend"] + 0.5 * dur
    blocked = False
    try:
        if info["reached"][2] > grasp_z + 0.015:
            blocked = True
    except Exception:
        pass
    rec["blocked"] = blocked
    if blocked:
        fails[target["color"]] = fails.get(target["color"], 0) + 1
        res, info, dur = timed_move((gx, gy, hover_z), q, 0.02, 60, "retreat")
        rec["moves"].append(info)
        continue
    t0 = now_s()
    mark(close_gripper(), "close")
    if ended["flag"]:
        break
    DURATIONS["close"] = 0.5 * DURATIONS["close"] + 0.5 * max(0.0, now_s() - t0)
    lift_z = max(grasp_z + TRANSIT_CLEAR, hover_z)
    res, info, dur = timed_move((gx, gy, lift_z), q, 0.02, 150, "lift")
    rec["moves"].append(info)
    if ended["flag"]:
        break
    grip = float(get_robot_state().gripper_positions["primary"])
    rec["grip_after_lift"] = round(grip, 3)
    held = grip > 0.08 and grip < 0.95
    rec["held"] = held
    if not held:
        fails[target["color"]] = fails.get(target["color"], 0) + 1
        mark(open_gripper(), "reopen")
        continue
    if bin_info is None:
        log("no bin localised; cannot place")
        mark(open_gripper(), "release-nobin")
        continue
    bx = bin_info["x"]
    by = bin_info["y"]
    rim = bin_info["z_top"]
    drop_z = rim + 0.05 + 0.5 * height
    res, info, dur = timed_move((bx, by, drop_z), q, 0.015, 200, "to_bin")
    rec["moves"].append(info)
    if ended["flag"]:
        break
    grip2 = float(get_robot_state().gripper_positions["primary"])
    rec["grip_at_bin"] = round(grip2, 3)
    mark(open_gripper(), "release")
    if ended["flag"]:
        break
    if grip2 > 0.08:
        remaining[target["color"]] = remaining.get(target["color"], 0) - 1
        placed[target["color"]] = placed.get(target["color"], 0) + 1
    else:
        fails[target["color"]] = fails.get(target["color"], 0) + 1
        rec["lost_in_transit"] = True
    rec["t_done"] = now_s()

report["placed"] = placed
report["remaining"] = remaining
report["fails"] = fails

# ---------------------------------------------------------------- press the button
if not ended["flag"] and button_info is not None:
    mark(close_gripper(), "close-for-button")
    bxp = button_info["top_x"]
    byp = button_info["top_y"]
    btop = button_info["z_top"]
    qb = downward_quat(home_jaw_yaw)
    press = {"moves": []}
    report["press"] = press
    if not ended["flag"]:
        res, info, dur = timed_move((bxp, byp, btop + 0.10), qb, 0.015, 200, "button_above")
        press["moves"].append(info)
    if not ended["flag"]:
        res, info, dur = timed_move((bxp, byp, btop - 0.02), qb, 0.01, 40, "button_press")
        press["moves"].append(info)
    if not ended["flag"]:
        hold(5)
    if not ended["flag"]:
        res, info, dur = timed_move((bxp, byp, btop + 0.08), qb, 0.02, 80, "button_up")
        press["moves"].append(info)
    if not ended["flag"]:
        hold(10)

report["ended"] = dict(ended)
try:
    report["t_end"] = now_s()
except Exception:
    pass
result = report
