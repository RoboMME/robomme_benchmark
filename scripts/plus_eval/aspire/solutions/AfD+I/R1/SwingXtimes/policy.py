# SwingXtimes v01 -- trace-guided repair of v00 (episode 1 run: visit 2 missed an evasive
# target; the program then entered the other target out of order and the episode ended at
# step 277).  Changes vs v00 (grasp, transit and button logic unchanged):
#   * no velocity lead: the targets hop ~6 cm when the cube approaches; extrapolating the
#     hops overshot by 2.4 cm.  Each chase step goes to the latest observed target centre.
#   * chase loop: re-localize the target (front camera, wrist-camera fallback) and re-centre
#     until the target is seen lit, up to MAX_CHASE steps, bounded by the remaining budget;
#     never advance to the other target while budget remains and the visit is unregistered.
#   * highlight test = fraction of strongly red pixels (r - max(g, b) > LIT_RED) in a disk
#     box projected at the tracked target, excluding the projected held-cube box (the cube
#     can itself be red).  v00 also required white > 175 pixels, which never occur.
#   * episode-end check happens before any further API call after a move or step.
# Strategy otherwise as v00: localize cube / two flat targets / button in the front camera;
# refine the cube in the wrist camera; top-down grasp; 2N alternating visits; put down at the
# last target; press the button.
import math

CAM_F = "front"
CAM_W = "wrist"
HOVER_CLEAR = 0.035      # cube bottom above target top during a visit (m)
TRANSIT_CLEAR = 0.10     # cube bottom above table at the swing mid-waypoint (m)
SETTLE_TICKS = 2
DWELL_TICKS = 5
MAX_CHASE = 14           # re-centre steps per visit (bounded further by the tick budget)
LOWER_STEP = 0.01        # lower 1 cm when the target is under the cube but not lit
LIT_RED = 60             # r - max(g, b) above this counts as a lit-target pixel
LIT_FRAC = 0.3           # lit if this fraction of the (cube-excluded) disk box is lit-red
LIT_MIN_N = 20           # minimum visible pixels for a highlight decision
TARGET_BOX_R = 0.03      # half-size of the projected target box (m)
OTHER_STATIC_R = 0.04    # inactive target: candidate within this of its track (m)
ACTIVE_JUMP_R = 0.15     # active target: nearest other candidate within this of its track (m)
SIZE_RATIO = (0.6, 1.6)  # accepted candidate radius relative to the target's radius at home
TARGET_PROMPTS = ["target", "bullseye target", "white round target", "round flat disc"]
BUTTON_PROMPTS = ["button", "white push button", "grey round button"]
NUMBER_WORDS = {"once": 1, "twice": 2, "thrice": 3, "one": 1, "two": 2, "three": 3, "four": 4,
                "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12}
COLOR_WORDS = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "white",
               "black", "cyan", "magenta", "brown", "gray", "grey"]

R = {"phase": "start", "log": [], "errors": [], "visits": []}
S = {"terminal": False}


def rnd(v, n=4):
    try:
        return round(float(v), n)
    except Exception:
        return None


def rv(vec, n=4):
    out = []
    for x in vec:
        out.append(rnd(x, n))
    return out


def note(kind, fields):
    item = {"k": kind, "tick": cur_tick()}
    for key in fields:
        item[key] = fields[key]
    R["log"].append(item)
    return item


def err(where, e):
    R["errors"].append({"where": where, "error": str(e)[:400], "tick": cur_tick()})


def cur_tick():
    try:
        return int(round(float(get_robot_state().timestamp_s) * S.get("hz", 20.0)))
    except Exception:
        return -1


def quantile(values, frac):
    vals = sorted(values)
    if len(vals) == 0:
        return None
    idx = int((len(vals) - 1) * frac)
    return vals[idx]


def check_terminal(res, where):
    flags = {}
    for name in ("terminated", "truncated", "done"):
        try:
            if name == "terminated":
                val = res.terminated
            elif name == "truncated":
                val = res.truncated
            else:
                val = res.done
            flags[name] = bool(val)
        except Exception:
            pass
    if flags.get("terminated") or flags.get("truncated") or flags.get("done"):
        S["terminal"] = True
        note("terminal", {"where": where, "flags": flags})
        raise Exception("TERMINAL at " + where)
    return flags


# ---------------- quaternion helpers (wxyz) ----------------
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def qrot(q, v):
    w, x, y, z = q
    r00 = 1 - 2 * (y * y + z * z); r01 = 2 * (x * y - z * w); r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w); r11 = 1 - 2 * (x * x + z * z); r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w); r21 = 2 * (y * z + x * w); r22 = 1 - 2 * (x * x + y * y)
    return (r00 * v[0] + r01 * v[1] + r02 * v[2],
            r10 * v[0] + r11 * v[1] + r12 * v[2],
            r20 * v[0] + r21 * v[1] + r22 * v[2])


def qinv_rot(q, v):
    w, x, y, z = q
    return qrot((w, -x, -y, -z), v)


def qz(angle):
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def wrap90(a):
    # wrap an angle into [-pi/4, pi/4) modulo pi/2
    q = math.pi / 2.0
    while a >= q / 2.0:
        a -= q
    while a < -q / 2.0:
        a += q
    return a


# ---------------- perception helpers ----------------
def cloud_of(cand, cam):
    base = S["base"]
    pc = None
    try:
        pc = mask_to_point_cloud(cand, cam, target_frame=base)
    except Exception as e:
        try:
            pc = mask_to_point_cloud(cand.mask, cam, target_frame=base)
        except Exception as e2:
            err("mask_to_point_cloud", str(e) + " | " + str(e2))
            return []
    pts = []
    try:
        raw = pc.points
        n = len(raw)
        stride = 1
        if n > 1500:
            stride = n // 1500 + 1
        i = 0
        while i < n:
            p = raw[i]
            x = float(p[0]); y = float(p[1]); z = float(p[2])
            if x == x and y == y and z == z:
                pts.append((x, y, z))
            i += stride
    except Exception as e:
        err("cloud_points", e)
    return pts


def summarize(pts):
    if len(pts) < 5:
        return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
    ztop = quantile(zs, 0.9)
    top = [p for p in pts if p[2] >= ztop - 0.008]
    if len(top) < 3:
        top = pts
    return {
        "n": len(pts),
        "cx": quantile([p[0] for p in top], 0.5), "cy": quantile([p[1] for p in top], 0.5),
        "mx": quantile(xs, 0.5), "my": quantile(ys, 0.5), "mz": quantile(zs, 0.5),
        "z10": quantile(zs, 0.1), "z90": ztop, "z05": quantile(zs, 0.05),
        "sx": quantile(xs, 0.9) - quantile(xs, 0.1), "sy": quantile(ys, 0.9) - quantile(ys, 0.1),
        "sz": ztop - quantile(zs, 0.1),
    }


def box_of(cand):
    try:
        b = cand.box_xyxy
        if b is None:
            return None
        return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    except Exception:
        return None


def score_of(cand):
    try:
        return float(cand.score)
    except Exception:
        return 0.0


def segment(cam, prompt, limit=8):
    out = []
    try:
        seg = segment_text(cam, prompt)
    except Exception as e:
        err("segment_text " + cam + " " + prompt, e)
        return out
    try:
        if not seg.ok:
            return out
        cands = seg.segmentations
    except Exception as e:
        err("segment_text fields", e)
        return out
    i = 0
    for cand in cands:
        if i >= limit:
            break
        i += 1
        out.append(cand)
    return out


def pixel_stats(cam, box, lower_frac=0.0, excl=None):
    # count red / white / lit-red / colour means in the inner part of a box on the current
    # image, skipping pixels inside the optional exclusion box (projected held cube)
    res = {"red": 0, "white": 0, "lit": 0, "n": 0, "r": 0.0, "g": 0.0, "b": 0.0}
    if box is None:
        return res
    try:
        rgb = get_observation().cameras[cam].rgb
        h = int(rgb.shape[0]); w = int(rgb.shape[1])
    except Exception as e:
        err("pixel_stats rgb", e)
        return res
    x1, y1, x2, y2 = box
    bw = x2 - x1; bh = y2 - y1
    u1 = int(max(0, x1 + 0.15 * bw)); u2 = int(min(w - 1, x2 - 0.15 * bw))
    v1 = int(max(0, y1 + max(0.15, lower_frac) * bh)); v2 = int(min(h - 1, y2 - 0.1 * bh))
    step_u = max(1, (u2 - u1) // 20); step_v = max(1, (v2 - v1) // 20)
    v = v1
    sr = 0.0; sg = 0.0; sb = 0.0
    while v <= v2:
        u = u1
        while u <= u2:
            if excl is not None and excl[0] <= u <= excl[2] and excl[1] <= v <= excl[3]:
                u += step_u
                continue
            px = rgb[v][u]
            r = int(px[0]); g = int(px[1]); b = int(px[2])
            sr += r; sg += g; sb += b
            res["n"] += 1
            if r > 150 and g < 90 and b < 90:
                res["red"] += 1
            if r > 175 and g > 175 and b > 175:
                res["white"] += 1
            if r - max(g, b) > LIT_RED:
                res["lit"] += 1
            u += step_u
        v += step_v
    if res["n"] > 0:
        res["r"] = sr / res["n"]; res["g"] = sg / res["n"]; res["b"] = sb / res["n"]
    return res


def color_score(name, r, g, b):
    if name == "red":
        return r - max(g, b)
    if name == "green":
        return g - max(r, b)
    if name == "blue":
        return b - max(r, g)
    if name == "yellow":
        return min(r, g) - b
    if name in ("purple", "magenta"):
        return min(r, b) - g
    if name == "orange":
        return r - b - abs(g - 0.5 * (r + b))
    if name == "cyan":
        return min(g, b) - r
    if name == "pink":
        return r - g
    if name in ("white",):
        return min(r, g, b) - 150
    if name in ("black",):
        return 80 - max(r, g, b)
    return 0.0


# ---------------- camera projection (convention selected from data) ----------------
def project(cam_obs, p, conv):
    K = cam_obs.intrinsics
    fx = float(K[0][0]); fy = float(K[1][1]); cx = float(K[0][2]); cy = float(K[1][2])
    pose = cam_obs.camera_pose
    t = pose.position; q = pose.quaternion_wxyz
    q = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    d = (p[0] - float(t[0]), p[1] - float(t[1]), p[2] - float(t[2]))
    c = qinv_rot(q, d)
    if conv == "opencv":
        if c[2] <= 1e-6:
            return None
        return (fx * c[0] / c[2] + cx, fy * c[1] / c[2] + cy)
    if conv == "opengl":
        if -c[2] <= 1e-6:
            return None
        return (fx * c[0] / (-c[2]) + cx, -fy * c[1] / (-c[2]) + cy)
    if conv == "sapien":
        if c[0] <= 1e-6:
            return None
        return (fx * (-c[1]) / c[0] + cx, fy * (-c[2]) / c[0] + cy)
    return None


def choose_projection(samples):
    # samples: list of (xyz, box)
    best = None
    table = {}
    try:
        cam_obs = get_observation().cameras[CAM_F]
    except Exception as e:
        err("choose_projection", e)
        return None
    for conv in ("opencv", "opengl", "sapien"):
        tot = 0.0; cnt = 0
        for item in samples:
            uv = project(cam_obs, item[0], conv)
            b = item[1]
            if uv is None or b is None:
                tot += 1000.0; cnt += 1
                continue
            du = uv[0] - 0.5 * (b[0] + b[2]); dv = uv[1] - 0.5 * (b[1] + b[3])
            tot += math.sqrt(du * du + dv * dv); cnt += 1
        if cnt > 0:
            table[conv] = rnd(tot / cnt, 2)
            if best is None or tot / cnt < table[best]:
                best = conv
    R["projection_errors_px"] = table
    if best is not None and table[best] < 12.0:
        R["projection"] = best
        return best
    R["projection"] = None
    return None


def box_from_projection(p, radius_m):
    conv = S.get("proj")
    if conv is None:
        return None
    try:
        cam_obs = get_observation().cameras[CAM_F]
        c = project(cam_obs, p, conv)
        e1 = project(cam_obs, (p[0], p[1] + radius_m, p[2]), conv)
        e2 = project(cam_obs, (p[0] + radius_m, p[1], p[2]), conv)
    except Exception as e:
        err("box_from_projection", e)
        return None
    if c is None or e1 is None or e2 is None:
        return None
    ru = max(abs(e1[0] - c[0]), abs(e2[0] - c[0]), 2.0)
    rvv = max(abs(e1[1] - c[1]), abs(e2[1] - c[1]), 2.0)
    return [c[0] - ru, c[1] - rvv, c[0] + ru, c[1] + rvv]


# ---------------- scene-object finders ----------------
def find_targets(prompts, record_key=None, exclude_button=True, limit=8, cam=CAM_F):
    """Flat disc-like candidates near table height in one camera. Returns list of dicts."""
    found = []
    raw = []
    table_z = S.get("table_z")
    for prompt in prompts:
        for cand in segment(cam, prompt, limit):
            box = box_of(cand)
            if cam != CAM_F:
                box = None
            sm = summarize(cloud_of(cand, cam))
            entry = {"prompt": prompt, "score": rnd(score_of(cand), 3), "box": rv(box or [], 1)}
            if sm is None:
                entry["rej"] = "few points"
                raw.append(entry)
                continue
            entry["c"] = [rnd(sm["mx"]), rnd(sm["my"]), rnd(sm["mz"])]
            entry["span"] = [rnd(sm["sx"], 3), rnd(sm["sy"], 3), rnd(sm["sz"], 3)]
            rej = None
            if sm["sz"] > 0.02:
                rej = "not flat"
            elif max(sm["sx"], sm["sy"]) > 0.16 or max(sm["sx"], sm["sy"]) < 0.015:
                rej = "size"
            elif table_z is not None and (sm["mz"] > table_z + 0.03 or sm["mz"] < table_z - 0.05):
                rej = "height"
            btn = S.get("button")
            if rej is None and btn is not None and exclude_button:
                dxb = sm["mx"] - btn["x"]; dyb = sm["my"] - btn["y"]
                if math.sqrt(dxb * dxb + dyb * dyb) < 0.035:
                    rej = "button"
            cube = S.get("cube0")
            if rej is None and cube is not None and not S.get("holding"):
                dxc = sm["mx"] - cube["x"]; dyc = sm["my"] - cube["y"]
                if math.sqrt(dxc * dxc + dyc * dyc) < 0.03:
                    rej = "cube"
            if rej is not None:
                entry["rej"] = rej
                raw.append(entry)
                continue
            raw.append(entry)
            dup = False
            for f in found:
                dx = f["x"] - sm["mx"]; dy = f["y"] - sm["my"]
                if math.sqrt(dx * dx + dy * dy) < 0.03:
                    dup = True
                    if score_of(cand) > f["score"]:
                        f["x"] = sm["mx"]; f["y"] = sm["my"]; f["z"] = sm["mz"]
                        f["score"] = score_of(cand); f["box"] = box
            if not dup:
                found.append({"x": sm["mx"], "y": sm["my"], "z": sm["mz"], "score": score_of(cand),
                              "box": box, "prompt": prompt,
                              "r": 0.5 * max(sm["sx"], sm["sy"])})
        if len(found) >= 2 and record_key is None:
            break
    if record_key is not None:
        R[record_key] = raw
    return found


def find_cube(color, cam, near=None, record_key=None):
    prompts = []
    if color is not None:
        prompts.append(color + " cube")
        prompts.append(color + " block")
    prompts.append("cube")
    best = None
    best_colour_rej = None
    raw = []
    for prompt in prompts:
        for cand in segment(cam, prompt, 6):
            box = box_of(cand)
            pts = cloud_of(cand, cam)
            sm = summarize(pts)
            px = pixel_stats(cam, box, 0.0)
            cs = color_score(color, px["r"], px["g"], px["b"]) if color is not None else 0.0
            entry = {"prompt": prompt, "score": rnd(score_of(cand), 3), "box": rv(box or [], 1),
                     "rgb": [rnd(px["r"], 1), rnd(px["g"], 1), rnd(px["b"], 1)], "cscore": rnd(cs, 1)}
            if sm is None:
                entry["rej"] = "few points"
                raw.append(entry)
                continue
            entry["c"] = [rnd(sm["cx"]), rnd(sm["cy"]), rnd(sm["z90"])]
            entry["span"] = [rnd(sm["sx"], 3), rnd(sm["sy"], 3), rnd(sm["sz"], 3)]
            rej = None
            if max(sm["sx"], sm["sy"]) > 0.10:
                rej = "too large"
            elif cam == CAM_F and sm["sz"] < 0.01:
                rej = "flat (light blob / target?)"
            elif color is not None and cs < 25:
                rej = "colour"
                if near is None or math.sqrt((sm["cx"] - near[0]) ** 2 + (sm["cy"] - near[1]) ** 2) < 0.07:
                    if best_colour_rej is None or cs > best_colour_rej["key"]:
                        best_colour_rej = {"key": cs, "cand": cand, "pts": pts, "sm": sm, "box": box, "prompt": prompt + " (colour-fallback)"}
            elif near is not None:
                dx = sm["cx"] - near[0]; dy = sm["cy"] - near[1]
                if math.sqrt(dx * dx + dy * dy) > 0.07:
                    rej = "far from prior"
            if rej is not None:
                entry["rej"] = rej
                raw.append(entry)
                continue
            raw.append(entry)
            key = cs + 40.0 * score_of(cand)
            if best is None or key > best["key"]:
                best = {"key": key, "cand": cand, "pts": pts, "sm": sm, "box": box, "prompt": prompt}
        if best is not None:
            break
    if best is None and best_colour_rej is not None:
        best = best_colour_rej
    if record_key is not None:
        R[record_key] = raw
    return best


def find_button(cam, near=None, record_key=None):
    best = None
    raw = []
    for prompt in BUTTON_PROMPTS:
        for cand in segment(cam, prompt, 6):
            box = box_of(cand)
            pts = cloud_of(cand, cam)
            sm = summarize(pts)
            entry = {"prompt": prompt, "score": rnd(score_of(cand), 3), "box": rv(box or [], 1)}
            if sm is None:
                entry["rej"] = "few points"
                raw.append(entry)
                continue
            entry["c"] = [rnd(sm["cx"]), rnd(sm["cy"]), rnd(sm["z90"])]
            entry["span"] = [rnd(sm["sx"], 3), rnd(sm["sy"], 3), rnd(sm["sz"], 3)]
            rej = None
            if max(sm["sx"], sm["sy"]) > 0.12:
                rej = "too large"
            elif cam == CAM_F and sm["sz"] < 0.008:
                rej = "flat (target-like)"
            elif near is not None:
                dx = sm["cx"] - near[0]; dy = sm["cy"] - near[1]
                if math.sqrt(dx * dx + dy * dy) > 0.06:
                    rej = "far from prior"
            if rej is not None:
                entry["rej"] = rej
                raw.append(entry)
                continue
            raw.append(entry)
            key = score_of(cand) + 2.0 * sm["sz"]
            if best is None or key > best["key"]:
                best = {"key": key, "sm": sm, "box": box, "prompt": prompt}
        if best is not None:
            break
    if record_key is not None:
        R[record_key] = raw
    return best


def estimate_yaw(pts):
    """Yaw (rad) of a square top face from its xy points: minimise the bounding area."""
    if len(pts) < 20:
        return None, None
    stride = max(1, len(pts) // 400)
    sub = []
    i = 0
    while i < len(pts):
        sub.append(pts[i])
        i += stride
    best = None
    for k in range(18):
        a = k * math.pi / 36.0
        ca = math.cos(a); sa = math.sin(a)
        us = [p[0] * ca + p[1] * sa for p in sub]
        vs = [-p[0] * sa + p[1] * ca for p in sub]
        su = quantile(us, 0.97) - quantile(us, 0.03)
        sv = quantile(vs, 0.97) - quantile(vs, 0.03)
        area = su * sv
        if best is None or area < best[0]:
            best = (area, a, su, sv)
    return best[1], (best[2], best[3])


# ---------------- motion helpers ----------------
def ee_pos():
    p = get_robot_state().end_effector_poses["primary"].position
    return (float(p[0]), float(p[1]), float(p[2]))


def move(xyz, label, tol=0.01, max_steps=200, quat=None):
    if quat is None:
        quat = S["quat"]
    t0 = cur_tick()
    pose = Pose([float(xyz[0]), float(xyz[1]), float(xyz[2])], [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])], S["base"])
    res = None
    ok = False
    status = None
    steps = None
    try:
        res = move_to_pose(pose, tolerance=tol, max_steps=max_steps)
        ok = bool(res.ok)
        if not ok:
            msg = ""
            try:
                msg = str(res.error) + " " + str(res.status)
            except Exception:
                msg = ""
            if "exceeds" in msg and cur_tick() == t0:
                note("move_retry", {"label": label, "reason": msg[:200]})
                res = move_to_pose(pose, tolerance=tol, max_steps=max_steps * 2)
                ok = bool(res.ok)
        try:
            status = str(res.status)
        except Exception:
            status = None
        try:
            steps = int(res.steps_executed)
        except Exception:
            steps = None
    except Exception as e:
        err("move " + label, e)
        if "Episode has ended" in str(e):
            S["terminal"] = True
            raise Exception("TERMINAL (closed) at move " + label)
    if res is not None:
        check_terminal(res, "move " + label)
    reached = ee_pos()
    dx = reached[0] - xyz[0]; dy = reached[1] - xyz[1]; dz = reached[2] - xyz[2]
    entry = note("move", {"label": label, "ok": ok, "status": status, "steps": steps, "dt": cur_tick() - t0, "goal": rv(xyz), "reached": rv(reached), "err_m": rnd(math.sqrt(dx * dx + dy * dy + dz * dz))})
    if res is not None and not ok:
        try:
            entry["error"] = str(res.error)[:300]
        except Exception:
            pass
    return ok


def hold(n, grip, label):
    st = get_robot_state()
    joints = [float(v) for v in st.joint_positions["primary"]]
    done = 0
    for tick_i in range(n):
        try:
            reply = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=joints, gripper_position=grip,
                embodiment=st.embodiment)}))
        except Exception as e:
            err("hold " + label, e)
            if "Episode has ended" in str(e):
                S["terminal"] = True
                raise Exception("TERMINAL (closed) at hold " + label)
            break
        done += 1
        check_terminal(reply, "hold " + label)
    return done


def gripper_reading():
    try:
        return float(get_robot_state().gripper_positions["primary"])
    except Exception:
        return None


def ticks_left():
    return S["max_steps"] - cur_tick()


# =====================================================================
# Phase 0: context, metadata, goal parsing
# =====================================================================
try:
    ctx = get_task_context()
    goal = str(ctx.language)
    R["goal"] = goal
    try:
        R["task_name"] = str(ctx.task_name)
    except Exception:
        pass
    state0 = get_robot_state()
    S["base"] = state0.base_frame
    R["base_frame"] = str(state0.base_frame)
    R["embodiment"] = str(state0.embodiment)
    meta = {}
    try:
        md = robomme_plus.get_task_metadata()
        for key in md:
            meta[str(key)] = str(md[key])[:300]
        try:
            S["max_steps"] = int(md["max_steps"])
        except Exception:
            S["max_steps"] = 1000
        try:
            S["hz"] = float(md["control_hz"])
        except Exception:
            try:
                S["hz"] = float(md["control_frequency_hz"])
            except Exception:
                S["hz"] = 20.0
    except Exception as e:
        err("task_metadata", e)
        S["max_steps"] = 1000
    if "max_steps" not in S:
        S["max_steps"] = 1000
    R["task_metadata"] = meta
    R["max_steps_used"] = S["max_steps"]
    try:
        cm = robomme_plus.get_controller_metadata()
        cmeta = {}
        for key in cm:
            cmeta[str(key)] = str(cm[key])[:300]
        R["controller_metadata"] = cmeta
    except Exception as e:
        err("controller_metadata", e)
    try:
        video = robomme_plus.get_conditioning_video()
        R["conditioning_video_frames"] = len(video)
        if len(video) > 0:
            R["conditioning_video_t"] = [rnd(video[0].timestamp_s, 3), rnd(video[len(video) - 1].timestamp_s, 3)]
    except Exception as e:
        err("conditioning_video", e)
    obs0 = get_observation()
    cams = {}
    for name in obs0.cameras:
        c = obs0.cameras[name]
        info = {}
        try:
            info["shape"] = [int(v) for v in c.rgb.shape]
            info["pose_p"] = rv(c.camera_pose.position)
            info["pose_q"] = rv(c.camera_pose.quaternion_wxyz)
            info["pose_frame"] = str(c.camera_pose.frame)
            K = c.intrinsics
            info["K"] = [rnd(K[0][0], 2), rnd(K[1][1], 2), rnd(K[0][2], 2), rnd(K[1][2], 2)]
        except Exception as e:
            err("camera info " + str(name), e)
        cams[str(name)] = info
    R["cameras"] = cams
    ee0 = state0.end_effector_poses["primary"]
    R["ee0"] = {"p": rv(ee0.position), "q": rv(ee0.quaternion_wxyz), "frame": str(ee0.frame)}
    R["joints0"] = rv(state0.joint_positions["primary"])
    R["gripper0"] = gripper_reading()

    # --- parse the goal ---
    low = goal.lower().replace(",", " ").replace(".", " ")
    words = low.split()
    color = None
    for i in range(len(words)):
        if words[i] in ("cube", "block") and i > 0 and words[i - 1] in COLOR_WORDS:
            color = words[i - 1]
            break
    if color is None:
        for w in words:
            if w in COLOR_WORDS:
                color = w
                break
    count = None
    for i in range(len(words)):
        if words[i] == "times" and i > 0:
            prev = words[i - 1]
            if prev in NUMBER_WORDS:
                count = NUMBER_WORDS[prev]
            else:
                try:
                    count = int(prev)
                except Exception:
                    count = None
    if count is None:
        for w in words:
            if w in ("once", "twice", "thrice"):
                count = NUMBER_WORDS[w]
    ir = low.find("right")
    il = low.find("left")
    if ir >= 0 and (il < 0 or ir < il):
        first = "right"
    else:
        first = "left"
    second = "left" if first == "right" else "right"
    if count is None:
        count = 1
        R["errors"].append({"where": "parse", "error": "count not found; defaulting to 1"})
    S["color"] = color; S["count"] = count; S["first"] = first; S["second"] = second
    R["parsed"] = {"color": color, "count": count, "first": first, "second": second}
except Exception as e:
    err("phase0", e)

# =====================================================================
# Phase 1: initial perception (front camera, arm at home)
# =====================================================================
try:
    R["phase"] = "perceive"
    q0 = get_robot_state().end_effector_poses["primary"].quaternion_wxyz
    q0 = (float(q0[0]), float(q0[1]), float(q0[2]), float(q0[3]))
    zaxis = qrot(q0, (0.0, 0.0, 1.0))
    R["ee0_zaxis"] = rv(zaxis)
    if zaxis[2] < -0.95:
        S["quat"] = q0
        R["down_quat_source"] = "home"
    else:
        S["quat"] = (0.0, 1.0, 0.0, 0.0)
        R["down_quat_source"] = "fixed 180deg about x (home not pointing down)"

    cube = find_cube(S.get("color"), CAM_F, None, "cube_front_candidates")
    if cube is None:
        raise ValueError("cube not found in front camera")
    sm = cube["sm"]
    S["cube0"] = {"x": sm["cx"], "y": sm["cy"], "top": sm["z90"]}
    S["table_z"] = sm["z05"]
    R["cube_front"] = {"x": rnd(sm["cx"]), "y": rnd(sm["cy"]), "top": rnd(sm["z90"]),
                       "z05": rnd(sm["z05"]), "span": [rnd(sm["sx"], 3), rnd(sm["sy"], 3), rnd(sm["sz"], 3)],
                       "prompt": cube["prompt"]}

    btn = find_button(CAM_F, None, "button_front_candidates")
    if btn is not None:
        bsm = btn["sm"]
        S["button"] = {"x": bsm["cx"], "y": bsm["cy"], "top": bsm["z90"]}
        R["button_front"] = {"x": rnd(bsm["cx"]), "y": rnd(bsm["cy"]), "top": rnd(bsm["z90"]),
                             "span": [rnd(bsm["sx"], 3), rnd(bsm["sy"], 3), rnd(bsm["sz"], 3)],
                             "prompt": btn["prompt"]}

    tgs = find_targets(TARGET_PROMPTS, "target_front_candidates")
    if len(tgs) < 2:
        tgs = find_targets(TARGET_PROMPTS, "target_front_candidates_nobutton", False)
        R["targets_without_button_exclusion"] = True
    if len(tgs) < 2:
        raise ValueError("fewer than two targets found: " + str(len(tgs)))
    # keep the two highest-scoring distinct targets
    ordered = []
    for t in tgs:
        ordered.append((-t["score"], len(ordered), t))
    ordered = sorted(ordered)
    two = [ordered[0][2], ordered[1][2]]
    if two[0]["y"] < two[1]["y"]:
        right_t = two[0]; left_t = two[1]
    else:
        right_t = two[1]; left_t = two[0]
    t_now = float(get_robot_state().timestamp_s)
    S["targets"] = {"right": {"x": right_t["x"], "y": right_t["y"], "z": right_t["z"], "r": right_t["r"],
                              "hist": [(t_now, right_t["x"], right_t["y"])]},
                    "left": {"x": left_t["x"], "y": left_t["y"], "z": left_t["z"], "r": left_t["r"],
                             "hist": [(t_now, left_t["x"], left_t["y"])]}}
    R["targets0"] = {"right": [rnd(right_t["x"]), rnd(right_t["y"]), rnd(right_t["z"])],
                     "left": [rnd(left_t["x"]), rnd(left_t["y"]), rnd(left_t["z"])],
                     "right_box": rv(right_t["box"] or [], 1), "left_box": rv(left_t["box"] or [], 1)}
    # sanity: image-left should be robot-right (u smaller) if base +y appears on image right
    if right_t["box"] is not None and left_t["box"] is not None:
        R["side_check_right_u_less"] = (right_t["box"][0] + right_t["box"][2]) < (left_t["box"][0] + left_t["box"][2])
    # table height: targets lie on the table; prefer their mean when consistent with the cube
    tz = 0.5 * (right_t["z"] + left_t["z"])
    R["table_z_from_cube"] = rnd(S["table_z"])
    R["table_z_from_targets"] = rnd(tz)
    S["target_z"] = tz
    # projection convention for highlight checks
    S["proj"] = choose_projection([((right_t["x"], right_t["y"], right_t["z"]), right_t["box"]),
                                   ((left_t["x"], left_t["y"], left_t["z"]), left_t["box"])])
    R["highlight0"] = {"right": pixel_stats(CAM_F, right_t["box"], 0.3),
                       "left": pixel_stats(CAM_F, left_t["box"], 0.3)}
except Exception as e:
    err("phase1", e)


def relocate_targets(tag, force=False):
    """Re-localize both targets and update the per-side tracks. Returns the update dict."""
    t_now = float(get_robot_state().timestamp_s)
    if not force and S.get("last_reloc_t") is not None and t_now - S["last_reloc_t"] < 0.2:
        note("targets", {"tag": tag, "cached": True})
        return {}
    S["last_reloc_t"] = t_now
    tgs = find_targets(TARGET_PROMPTS[:2], None, True, 5)
    upd = {}
    for side in ("right", "left"):
        tr = S["targets"][side]
        other = S["targets"]["left" if side == "right" else "right"]
        best = None
        for t in tgs:
            dx = t["x"] - tr["x"]; dy = t["y"] - tr["y"]
            d = math.sqrt(dx * dx + dy * dy)
            dxo = t["x"] - other["x"]; dyo = t["y"] - other["y"]
            do = math.sqrt(dxo * dxo + dyo * dyo)
            if d < 0.12 and d <= do and (best is None or d < best[0]):
                best = (d, t)
        if best is not None:
            t = best[1]
            tr["x"] = t["x"]; tr["y"] = t["y"]; tr["z"] = t["z"]; tr["box"] = t["box"]
            tr["hist"].append((t_now, t["x"], t["y"]))
            if len(tr["hist"]) > 12:
                tr["hist"] = tr["hist"][len(tr["hist"]) - 12:]
            upd[side] = [rnd(t["x"]), rnd(t["y"]), rnd(best[0], 3)]
        else:
            tr["box"] = None
            upd[side] = None
    note("targets", {"tag": tag, "n": len(tgs), "upd": upd})
    return upd


def assign_active(side, tgs, cam):
    """Assign candidates during a visit to <side>.  The inactive target was static during
    visits in the v00 trace, so a candidate within OTHER_STATIC_R of its track is the inactive
    one; the active target is the nearest remaining candidate within ACTIVE_JUMP_R of its
    track (the active target hops away from the approaching cube) whose radius is within
    SIZE_RATIO of the radius this target had at home (rejects large flat table regions)."""
    tr = S["targets"][side]
    other = S["targets"]["left" if side == "right" else "right"]
    o_best = None
    a_best = None
    for t in tgs:
        dxo = t["x"] - other["x"]; dyo = t["y"] - other["y"]
        do = math.sqrt(dxo * dxo + dyo * dyo)
        if do < OTHER_STATIC_R:
            if o_best is None or do < o_best[0]:
                o_best = (do, t)
            continue
        if t["r"] < SIZE_RATIO[0] * tr["r"] or t["r"] > SIZE_RATIO[1] * tr["r"]:
            continue
        dx = t["x"] - tr["x"]; dy = t["y"] - tr["y"]
        d = math.sqrt(dx * dx + dy * dy)
        if d < ACTIVE_JUMP_R and (a_best is None or d < a_best[0]):
            a_best = (d, t)
    t_now = float(get_robot_state().timestamp_s)
    if o_best is not None and cam == CAM_F:
        other["box"] = o_best[1]["box"]
    if a_best is None:
        if cam == CAM_F:
            tr["box"] = None
        return None
    t = a_best[1]
    tr["x"] = t["x"]; tr["y"] = t["y"]
    tr["box"] = t["box"] if cam == CAM_F else None
    tr["hist"].append((t_now, t["x"], t["y"]))
    if len(tr["hist"]) > 12:
        tr["hist"] = tr["hist"][len(tr["hist"]) - 12:]
    return [rnd(t["x"]), rnd(t["y"]), rnd(a_best[0], 3)]


def relocate_side(side, tag):
    """Front-camera relocation of the active target; wrist-camera fallback. Returns the
    source camera name or None when the active target was not found."""
    tgs = find_targets(TARGET_PROMPTS[:2], None, True, 5)
    upd = assign_active(side, tgs, CAM_F)
    note("targets", {"tag": tag, "side": side, "cam": CAM_F, "n": len(tgs), "upd": upd})
    if upd is not None:
        return "front"
    tgs = find_targets(TARGET_PROMPTS[:1], None, True, 5, CAM_W)
    upd = assign_active(side, tgs, CAM_W)
    note("targets", {"tag": tag, "side": side, "cam": CAM_W, "n": len(tgs), "upd": upd})
    if upd is not None:
        return "wrist"
    return None


def cube_box_front():
    """Front-image box of the held cube (from the EE pose), used to exclude its pixels."""
    conv = S.get("proj")
    if conv is None or not S.get("holding"):
        return None
    try:
        cam_obs = get_observation().cameras[CAM_F]
        ee = ee_pos()
    except Exception as e:
        err("cube_box_front", e)
        return None
    half = 0.5 * S.get("size", 0.04) + 0.006
    zb = ee[2] - S["d"] - 0.004
    zt = ee[2] - S["d"] + S.get("size", 0.04) + 0.01
    us = []; vs = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for z in (zb, zt):
                uv = project(cam_obs, (ee[0] + sx * half, ee[1] + sy * half, z), conv)
                if uv is None:
                    return None
                us.append(uv[0]); vs.append(uv[1])
    return [min(us), min(vs), max(us), max(vs)]


def lit_fraction(side, excl):
    tr = S["targets"][side]
    box = box_from_projection((tr["x"], tr["y"], tr["z"]), TARGET_BOX_R)
    src = "proj"
    if box is None:
        box = tr.get("box")
        src = "seg"
    if box is None:
        return None, {"src": "none"}
    px = pixel_stats(CAM_F, box, 0.0, excl)
    if px["n"] < LIT_MIN_N:
        return None, {"src": src, "n": px["n"], "lit": px["lit"]}
    return px["lit"] / float(px["n"]), {"src": src, "n": px["n"], "lit": px["lit"]}


def highlighted(side):
    excl = cube_box_front()
    f, info = lit_fraction(side, excl)
    other = "left" if side == "right" else "right"
    fo, info_o = lit_fraction(other, excl)
    info["frac"] = rnd(f, 3)
    info["other_frac"] = rnd(fo, 3)
    hl = f is not None and f >= LIT_FRAC and (fo is None or f >= fo + 0.1)
    return hl, info


# =====================================================================
# Phase 2: grasp the cube (top-down, wrist refinement, yaw alignment)
# =====================================================================
try:
    R["phase"] = "grasp"
    if "targets" not in S or "cube0" not in S:
        raise ValueError("perception incomplete; skipping grasp")
    open_gripper()
    c0 = S["cube0"]
    pre_z = c0["top"] + 0.14
    move((c0["x"], c0["y"], pre_z), "pregrasp_front", tol=0.015, max_steps=200)
    wc = find_cube(S.get("color"), CAM_W, (c0["x"], c0["y"]), "cube_wrist_candidates")
    cx = c0["x"]; cy = c0["y"]; ctop = c0["top"]; size = 0.04
    yaw = None
    if wc is not None:
        wsm = wc["sm"]
        cx = wsm["cx"]; cy = wsm["cy"]; ctop = wsm["z90"]
        top_pts = [p for p in wc["pts"] if p[2] >= wsm["z90"] - 0.008]
        yaw, ext = estimate_yaw(top_pts)
        if ext is not None:
            size = max(0.02, min(0.07, 0.5 * (ext[0] + ext[1])))
        R["cube_wrist"] = {"x": rnd(cx), "y": rnd(cy), "top": rnd(ctop), "yaw_deg": rnd(yaw * 180.0 / math.pi if yaw is not None else None, 1),
                           "ext": rv(ext or [], 3), "n_top": len(top_pts)}
    R["cube_size_est"] = rnd(size, 3)
    S["size"] = size
    S["table_z_grasp"] = ctop - size
    # yaw alignment of the finger closing axis (tool y) with a cube face normal
    if yaw is not None:
        yax = qrot(S["quat"], (0.0, 1.0, 0.0))
        phi = math.atan2(yax[1], yax[0])
        delta = wrap90(yaw - phi)
        S["quat"] = qmul(qz(delta), S["quat"])
        R["grasp_yaw_delta_deg"] = rnd(delta * 180.0 / math.pi, 1)
    grasp_z = ctop - 0.5 * size
    grasp_z = max(grasp_z, ctop - size + 0.012)
    move((cx, cy, ctop + 0.08), "pregrasp_aligned", tol=0.012, max_steps=200)
    move((cx, cy, grasp_z), "grasp_descend", tol=0.008, max_steps=150)
    gres = close_gripper()
    try:
        check_terminal(gres, "close_gripper")
    except Exception as e:
        raise e
    S["holding"] = True
    S["d"] = grasp_z - (ctop - size)  # EE z minus cube bottom z
    R["ee_minus_cube_bottom"] = rnd(S["d"], 4)
    note("grasp", {"grip": gripper_reading(), "cx": rnd(cx), "cy": rnd(cy), "grasp_z": rnd(grasp_z)})
    lift_z = S["target_z"] + TRANSIT_CLEAR + S["d"]
    move((cx, cy, lift_z), "lift", tol=0.015, max_steps=200)
    g = gripper_reading()
    R["grip_after_lift"] = g
    if g is not None and g < 0.05:
        # fingers fully closed: nothing between them -> one re-grasp attempt from the wrist view
        note("regrasp", {"reason": "gripper fully closed after lift", "grip": g})
        open_gripper()
        wc2 = find_cube(S.get("color"), CAM_W, (cx, cy), "cube_wrist_candidates_retry")
        if wc2 is not None:
            cx = wc2["sm"]["cx"]; cy = wc2["sm"]["cy"]; ctop = wc2["sm"]["z90"]
            grasp_z = max(ctop - 0.5 * size, ctop - size + 0.012)
            move((cx, cy, ctop + 0.06), "regrasp_above", tol=0.012, max_steps=200)
            move((cx, cy, grasp_z), "regrasp_descend", tol=0.008, max_steps=150)
            close_gripper()
            move((cx, cy, lift_z), "regrasp_lift", tol=0.015, max_steps=200)
            R["grip_after_regrasp"] = gripper_reading()
except Exception as e:
    err("phase2", e)

# =====================================================================
# Phase 3: swing visits
# =====================================================================
try:
    R["phase"] = "swing"
    if not S.get("holding"):
        raise ValueError("no grasp; skipping swings")
    seq = []
    for i in range(S["count"]):
        seq.append(S["first"])
        seq.append(S["second"])
    R["visit_plan"] = seq
    total = len(seq)
    for k in range(total):
        if S["terminal"]:
            break
        side = seq[k]
        vt0 = cur_tick()
        visit = {"k": k, "side": side, "t0": vt0}
        reserve = 160 + 45 * (total - k - 1)
        relocate_side(side, "pre_visit_" + str(k))
        tr = S["targets"][side]
        tz = tr["z"]
        hover_z = tz + HOVER_CLEAR + S["d"]
        floor_z = tz + 0.012 + S["d"]
        transit_z = S["target_z"] + TRANSIT_CLEAR + S["d"]
        here = ee_pos()
        ddx = tr["x"] - here[0]; ddy = tr["y"] - here[1]
        dist = math.sqrt(ddx * ddx + ddy * ddy)
        visit["dist"] = rnd(dist, 3)
        if dist > 0.08:
            mid = (0.5 * (here[0] + tr["x"]), 0.5 * (here[1] + tr["y"]), max(transit_z, hover_z))
            move(mid, "mid_" + str(k), tol=0.03, max_steps=150)
        move((tr["x"], tr["y"], hover_z), "hover_" + str(k), tol=0.01, max_steps=200)
        hold(SETTLE_TICKS, 0.0, "settle_" + str(k))
        checks = []
        lit = False
        low = 0
        misses = 0
        for attempt in range(MAX_CHASE):
            srcname = relocate_side(side, "chase_" + str(k) + "_" + str(attempt))
            hl, hinfo = highlighted(side)
            here = ee_pos()
            tr = S["targets"][side]
            ex = tr["x"] - here[0]; ey = tr["y"] - here[1]
            e_xy = math.sqrt(ex * ex + ey * ey)
            checks.append({"attempt": attempt, "hl": hl, "hinfo": hinfo, "src": srcname,
                           "err_xy": rnd(e_xy, 4), "ee": rv(here),
                           "tgt": [rnd(tr["x"]), rnd(tr["y"])], "tick": cur_tick()})
            if hl:
                lit = True
                hold(DWELL_TICKS, 0.0, "dwell_" + str(k))
                break
            if ticks_left() < reserve + 15:
                note("chase_budget_stop", {"k": k, "attempt": attempt, "ticks_left": ticks_left()})
                break
            if srcname is None:
                misses += 1
                if misses >= 2:
                    # target not seen twice in a row: look from higher up, then re-centre
                    move((here[0], here[1], hover_z + 0.06), "lookup_" + str(k) + "_" + str(attempt), tol=0.015, max_steps=80)
                    misses = 0
                    continue
            else:
                misses = 0
            if e_xy < 0.012:
                low += 1
            z = max(hover_z - LOWER_STEP * low, floor_z)
            move((tr["x"], tr["y"], z), "chase_" + str(k) + "_" + str(attempt), tol=0.008, max_steps=100)
            hold(SETTLE_TICKS, 0.0, "settle_c_" + str(k))
        visit["lit"] = lit
        visit["checks"] = checks
        visit["ticks"] = cur_tick() - vt0
        visit["grip"] = gripper_reading()
        R["visits"].append(visit)
        if not lit:
            note("visit_unregistered", {"k": k, "side": side})
        if visit["grip"] is not None and visit["grip"] < 0.03:
            note("cube_lost", {"k": k, "grip": visit["grip"]})
except Exception as e:
    err("phase3", e)

# =====================================================================
# Phase 4: put the cube down at the last target, then press the button
# =====================================================================
try:
    R["phase"] = "putdown"
    if S["terminal"]:
        raise ValueError("terminal before put-down")
    if S.get("holding"):
        here = ee_pos()
        place_z = S["target_z"] + 0.012 + S["d"]
        move((here[0], here[1], place_z), "place_descend", tol=0.008, max_steps=120)
        ores = open_gripper()
        check_terminal(ores, "open_gripper")
        hold(3, 1.0, "release_settle")
        S["holding"] = False
        here = ee_pos()
        move((here[0], here[1], S["target_z"] + 0.12), "retreat_up", tol=0.02, max_steps=150)
    R["phase"] = "button"
    btn = S.get("button")
    if btn is None:
        b2 = find_button(CAM_F, None, "button_front_candidates_late")
        if b2 is not None:
            btn = {"x": b2["sm"]["cx"], "y": b2["sm"]["cy"], "top": b2["sm"]["z90"]}
    if btn is None:
        raise ValueError("button not found")
    cres = close_gripper()
    check_terminal(cres, "close_gripper_button")
    move((btn["x"], btn["y"], btn["top"] + 0.10), "above_button", tol=0.015, max_steps=200)
    wb = find_button(CAM_W, (btn["x"], btn["y"]), "button_wrist_candidates")
    bx = btn["x"]; by = btn["y"]; btop = btn["top"]
    if wb is not None:
        bx = wb["sm"]["cx"]; by = wb["sm"]["cy"]; btop = wb["sm"]["z90"]
        R["button_wrist"] = {"x": rnd(bx), "y": rnd(by), "top": rnd(btop)}
    move((bx, by, btop + 0.03), "button_approach", tol=0.01, max_steps=120)
    move((bx, by, btop - 0.012), "button_press", tol=0.005, max_steps=80)
    hold(4, 0.0, "press_hold")
    note("pressed", {"ee": rv(ee_pos())})
    move((bx, by, btop + 0.08), "button_release", tol=0.02, max_steps=100)
    hold(5, 0.0, "final_hold")
    R["phase"] = "done"
except Exception as e:
    err("phase4", e)

R["final_tick"] = cur_tick()
R["terminal"] = S["terminal"]
result = R
