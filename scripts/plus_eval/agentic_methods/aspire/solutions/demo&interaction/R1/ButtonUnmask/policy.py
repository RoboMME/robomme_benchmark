# ButtonUnmask v00_initial -- ASPIRE task worker, robomme_plus task_1.
# Strategy (from offline analysis of the released recordings; nothing scene-specific is hardcoded):
#   1. t=0: read goal/metadata, calibrate pixel->robot_base projection for both cameras
#      (hypotheses checked against mask_to_point_cloud and table flatness), localize the button.
#   2. Hold at home and colour-scan BOTH cameras every control tick; every saturated colour blob
#      becomes a 3-D cube track (colour, xy). Cubes appear and are covered by containers while the
#      episode runs, so this memory is the only source of "which container hides which colour".
#   3. Leave home as late as the press deadline allows (press must register before step 90),
#      press the button with a closed gripper, keep scanning while moving.
#   4. For each colour phrase of the goal (in order; "all containers hiding X" -> every X track):
#      hover, measure the container from depth, grasp top-down aligned to its faces, lift,
#      check the episode status, put it back, continue.
import math

result = {"version": "v00_initial", "errors": [], "events": [], "phases": {}}
R = result
ST = {"ended": False, "tick": 0, "ts": None, "hz": 20.0, "steps_called": 0, "emb": "robomme_plus",
      "base": "robot_base", "last_obs": None, "hyp": None, "table_z": None,
      "max_steps": None, "grip_partial_ok": True}
TRACKS = []
COLOR_NAMES = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "black",
               "white", "gray", "grey", "brown"]


def note_error(where, exc):
    text = str(exc)
    if len(R["errors"]) < 60:
        R["errors"].append({"where": where, "tick": ST["tick"], "error": text[:400]})
    if "Episode has ended" in text or "execution is closing" in text:
        ST["ended"] = True


def note(kind, data):
    if len(R["events"]) < 400:
        entry = {"kind": kind, "tick": ST["tick"]}
        entry["data"] = data
        R["events"].append(entry)


def r4(value):
    try:
        return round(float(value), 4)
    except Exception:
        return None


def vec(values):
    out = []
    try:
        for value in values:
            out.append(float(value))
    except Exception as exc:
        note_error("vec", exc)
    return out


def rows_of(array):
    try:
        return array.tolist()
    except Exception:
        pass
    out = []
    for row in array:
        out.append([value for value in row])
    return out


def terminal_of(res):
    flag = False
    try:
        flag = flag or bool(res.terminated)
    except Exception:
        pass
    try:
        flag = flag or bool(res.truncated)
    except Exception:
        pass
    return flag


def update_tick(ts):
    try:
        ST["ts"] = float(ts)
        value = int(round(float(ts) * ST["hz"]))
        if value > ST["tick"]:
            ST["tick"] = value
    except Exception:
        pass


def robot_state():
    if ST["ended"]:
        return None
    try:
        state = get_robot_state()
        update_tick(state.timestamp_s)
        return state
    except Exception as exc:
        note_error("get_robot_state", exc)
        return None


def joints_now():
    state = robot_state()
    if state is None:
        return None
    return vec(state.joint_positions["primary"])


def grip_now():
    state = robot_state()
    if state is None:
        return None
    try:
        return float(state.gripper_positions["primary"])
    except Exception as exc:
        note_error("grip_now", exc)
        return None


def observe():
    if ST["ended"]:
        return None
    try:
        obs = get_observation()
        update_tick(obs.timestamp_s)
        return obs
    except Exception as exc:
        note_error("get_observation", exc)
        return None


def hold_step(joints, grip):
    """One control tick toward absolute joints. Returns the observation (or None)."""
    if ST["ended"]:
        return None
    try:
        cmd = ArmCommand(mode="joint_position", target=[float(v) for v in joints],
                         gripper_position=float(grip), embodiment=ST["emb"])
        res = step(RobotAction(arms={"primary": cmd}))
    except Exception as exc:
        note_error("step", exc)
        return None
    ST["steps_called"] += 1
    if ST["tick"] < ST["steps_called"]:
        ST["tick"] = ST["steps_called"]
    try:
        if not bool(res.ok):
            note("step_not_ok", str(res.error)[:200])
    except Exception:
        pass
    if terminal_of(res):
        ST["ended"] = True
        note("terminal_after_step", {"steps_called": ST["steps_called"]})
    obs = None
    try:
        obs = res.observation
        update_tick(obs.timestamp_s)
    except Exception:
        obs = None
    ST["last_obs"] = obs
    return obs


def check_exec(res, label):
    info = {"label": label}
    try:
        info["ok"] = bool(res.ok)
    except Exception:
        info["ok"] = None
    try:
        info["status"] = str(res.status)
    except Exception:
        pass
    try:
        info["steps"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        if res.error is not None:
            info["error"] = str(res.error)[:200]
    except Exception:
        pass
    if terminal_of(res):
        ST["ended"] = True
        info["terminal"] = True
    robot_state()
    info["tick"] = ST["tick"]
    return info


# ---------------------------------------------------------------- quaternion / geometry
def quat_mat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 0:
        n = 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def mat_vec(m, v):
    return [m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]]


def topdown_quat(theta):
    # R = Rz(theta) * Rx(pi): tool z points down, hand x = (cos t, sin t, 0), fingers close along
    # hand y = (sin t, -cos t, 0).
    return [0.0, math.cos(theta / 2.0), math.sin(theta / 2.0), 0.0]


def wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def quantile(values, frac):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * frac)]


# ---------------------------------------------------------------- camera projection
HYPS = []
for conv in ("cv", "gl", "ros"):
    for dtype in ("z", "range"):
        for direction in ("c2b", "b2c"):
            HYPS.append(conv + "|" + dtype + "|" + direction)


def cam_data(cobs):
    K = rows_of(cobs.intrinsics)
    pose = cobs.camera_pose
    pos = vec(pose.position)
    quat = vec(pose.quaternion_wxyz)
    m = quat_mat(quat)
    return {"fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]),
            "cy": float(K[1][2]), "R": m, "t": pos, "q": quat,
            "frame": str(pose.frame)}


def project(cd, u, v, d, hyp):
    parts = hyp.split("|")
    xn = (u - cd["cx"]) / cd["fx"]
    yn = (v - cd["cy"]) / cd["fy"]
    if parts[1] == "range":
        n = math.sqrt(xn * xn + yn * yn + 1.0)
        z = d / n
    else:
        z = d
    x = xn * z
    y = yn * z
    if parts[0] == "cv":
        pc = [x, y, z]
    elif parts[0] == "gl":
        pc = [x, -y, -z]
    else:
        pc = [z, -x, -y]
    m = cd["R"]
    t = cd["t"]
    if parts[2] == "c2b":
        p = mat_vec(m, pc)
        return [p[0] + t[0], p[1] + t[1], p[2] + t[2]]
    dv = [pc[0] - t[0], pc[1] - t[1], pc[2] - t[2]]
    return [m[0][0] * dv[0] + m[1][0] * dv[1] + m[2][0] * dv[2],
            m[0][1] * dv[0] + m[1][1] * dv[1] + m[2][1] * dv[2],
            m[0][2] * dv[0] + m[1][2] * dv[1] + m[2][2] * dv[2]]


def valid_depth(d):
    return d == d and d > 0.02 and d < 4.0


def hue_sv(r, g, b):
    mx = r
    if g > mx:
        mx = g
    if b > mx:
        mx = b
    mn = r
    if g < mn:
        mn = g
    if b < mn:
        mn = b
    if mx <= 0:
        return 0.0, 0.0, 0.0
    dlt = mx - mn
    s = dlt / mx
    if dlt == 0:
        return 0.0, s, mx
    if mx == r:
        h = 60.0 * ((g - b) / dlt)
    elif mx == g:
        h = 60.0 * ((b - r) / dlt + 2.0)
    else:
        h = 60.0 * ((r - g) / dlt + 4.0)
    if h < 0:
        h += 360.0
    return h, s, mx


def classify(r, g, b, want_black):
    h, s, mx = hue_sv(r, g, b)
    if want_black and mx < 40 and mx > 3:
        return "black"
    if s < 0.38 or mx < 60:
        return None
    if 9.0 <= h < 30.0:
        return None  # wood table band
    if h < 9.0 or h >= 340.0:
        if s >= 0.65 and mx >= 80:
            return "red"
        return None
    if h < 46.0:
        if s >= 0.72 and mx >= 110:
            return "orange"
        return None
    if h < 75.0:
        if s >= 0.65 and mx >= 100:
            return "yellow"
        return None
    if h < 170.0:
        if s >= 0.55 and mx >= 70:
            return "green"
        return None
    if h < 205.0:
        if s >= 0.55 and mx >= 80:
            return "cyan"
        return None
    if h < 215.0:
        return None  # navy button base band
    if h < 262.0:
        if s >= 0.8 and mx >= 70:
            return "blue"
        return None
    if h < 330.0:
        if s >= 0.38 and mx >= 90:
            return "purple"
        return None
    if s >= 0.5 and mx >= 100:
        return "pink"
    return None


def wood_like(r, g, b):
    h, s, mx = hue_sv(r, g, b)
    return 9.0 <= h < 30.0 and 0.45 <= s <= 0.8 and mx >= 90


# ---------------------------------------------------------------- calibration
def calibrate(obs):
    report = {"hyp_scores": {}, "api_checks": []}
    base = ST["base"]
    per_cam = {}
    for cam in ("front", "wrist"):
        try:
            cobs = obs.cameras[cam]
        except Exception as exc:
            note_error("calib_cam_" + cam, exc)
            continue
        cd = cam_data(cobs)
        rgb = rows_of(cobs.rgb)
        dep = rows_of(cobs.depth_m)
        H = len(dep)
        W = len(dep[0])
        report[cam + "_shape"] = [H, W]
        report[cam + "_K"] = [r4(cd["fx"]), r4(cd["fy"]), r4(cd["cx"]), r4(cd["cy"])]
        report[cam + "_pose"] = [[r4(v) for v in cd["t"]], [r4(v) for v in cd["q"]], cd["frame"]]
        wood = []
        for v in range(4, H, 8):
            for u in range(4, W, 8):
                px = rgb[v][u]
                d = float(dep[v][u])
                if valid_depth(d) and wood_like(int(px[0]), int(px[1]), int(px[2])):
                    wood.append((u, v, d))
        per_cam[cam] = {"cd": cd, "wood": wood, "dep": dep, "W": W, "H": H}
        report[cam + "_wood_samples"] = len(wood)
    # API check: own mask patches projected by mask_to_point_cloud
    api_means = []
    for cam in per_cam:
        info = per_cam[cam]
        W = info["W"]
        H = info["H"]
        centers = [(int(W * 0.3), int(H * 0.62)), (int(W * 0.72), int(H * 0.45))]
        for (cu, cv) in centers:
            pix = []
            mask = []
            for v in range(H):
                row = []
                for u in range(W):
                    inside = abs(u - cu) <= 3 and abs(v - cv) <= 3
                    row.append(inside)
                    if inside:
                        d = float(info["dep"][v][u])
                        if valid_depth(d):
                            pix.append((u, v, d))
                mask.append(row)
            entry = {"cam": cam, "center": [cu, cv], "n_pix": len(pix)}
            try:
                pc = mask_to_point_cloud(mask, cam, target_frame=base)
                pts = rows_of(pc.points)
                entry["api_n"] = len(pts)
                if len(pts) > 0:
                    mean = [sum(p[k] for p in pts) / len(pts) for k in range(3)]
                    entry["api_mean"] = [r4(x) for x in mean]
                    api_means.append((cam, pix, mean))
            except Exception as exc:
                entry["api_error"] = str(exc)[:200]
            report["api_checks"].append(entry)
    best = None
    best_score = None
    for hyp in HYPS:
        score = {}
        flat_total = 0.0
        zs_all = []
        for cam in per_cam:
            info = per_cam[cam]
            zs = []
            for (u, v, d) in info["wood"]:
                zs.append(project(info["cd"], u, v, d, hyp)[2])
            if len(zs) >= 10:
                spread = quantile(zs, 0.9) - quantile(zs, 0.1)
                score[cam + "_flat"] = r4(spread)
                score[cam + "_zmed"] = r4(quantile(zs, 0.5))
                flat_total += spread
                zs_all.append(quantile(zs, 0.5))
            else:
                flat_total += 1.0
        if len(zs_all) == 2:
            score["cam_z_gap"] = r4(abs(zs_all[0] - zs_all[1]))
            flat_total += abs(zs_all[0] - zs_all[1])
        api_err = None
        for (cam, pix, mean) in api_means:
            if not pix:
                continue
            own = [0.0, 0.0, 0.0]
            for (u, v, d) in pix:
                p = project(per_cam[cam]["cd"], u, v, d, hyp)
                own = [own[0] + p[0], own[1] + p[1], own[2] + p[2]]
            own = [own[k] / len(pix) for k in range(3)]
            err = math.sqrt(sum((own[k] - mean[k]) ** 2 for k in range(3)))
            api_err = err if api_err is None else max(api_err, err)
        score["api_err"] = r4(api_err) if api_err is not None else None
        score["flat_total"] = r4(flat_total)
        report["hyp_scores"][hyp] = score
        if api_err is not None:
            metric = api_err
        else:
            metric = 10.0 + flat_total
        if best_score is None or metric < best_score:
            best_score = metric
            best = hyp
    report["chosen"] = best
    report["chosen_metric"] = r4(best_score)
    ok = False
    if best is not None:
        sc = report["hyp_scores"][best]
        if sc["api_err"] is not None and sc["api_err"] < 0.02:
            ok = True
        elif sc["api_err"] is None and sc["flat_total"] is not None and sc["flat_total"] < 0.04:
            ok = True
    report["ok"] = ok
    if ok:
        ST["hyp"] = best
        zmeds = []
        for cam in per_cam:
            zm = report["hyp_scores"][best].get(cam + "_zmed")
            if zm is not None:
                zmeds.append(zm)
        if zmeds:
            ST["table_z"] = sum(zmeds) / len(zmeds)
    report["table_z"] = r4(ST["table_z"]) if ST["table_z"] is not None else None
    return report


# ---------------------------------------------------------------- colour tracking
def scan_obs(obs, cams, allow_new, label, want_black):
    """Colour-scan the given cameras; update TRACKS. Returns per-colour cluster counts."""
    summary = {}
    if obs is None or ST["hyp"] is None or ST["table_z"] is None:
        return summary
    tz = ST["table_z"]
    for cam in cams:
        try:
            cobs = obs.cameras[cam]
            cd = cam_data(cobs)
            rgb = rows_of(cobs.rgb)
            dep = rows_of(cobs.depth_m)
        except Exception as exc:
            note_error("scan_" + cam, exc)
            continue
        H = len(dep)
        W = len(dep[0])
        by_color = {}
        for v in range(0, H, 2):
            rrow = rgb[v]
            drow = dep[v]
            for u in range(0, W, 2):
                px = rrow[u]
                r = int(px[0])
                g = int(px[1])
                b = int(px[2])
                if not want_black:
                    if r > g and g > b and r - b > 40:
                        if (g - b) * 2 < (r - b) and (g - b) * 6 > (r - b):
                            continue  # fast wood reject (hue 10-30 deg)
                name = classify(r, g, b, want_black)
                if name is None:
                    continue
                d = float(drow[u])
                if not valid_depth(d):
                    continue
                p = project(cd, u, v, d, ST["hyp"])
                if p[2] < tz + 0.002 or p[2] > tz + 0.075:
                    continue
                if name not in by_color:
                    by_color[name] = []
                by_color[name].append((p, u, v))
        for name in by_color:
            clusters = []
            for (p, u, v) in by_color[name]:
                placed = False
                for c in clusters:
                    cx = c["sx"] / c["n"]
                    cy = c["sy"] / c["n"]
                    if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 < 0.025 ** 2:
                        c["sx"] += p[0]
                        c["sy"] += p[1]
                        c["sz"] += p[2]
                        c["n"] += 1
                        if p[2] > c["zmax"]:
                            c["zmax"] = p[2]
                        c["su"] += u
                        c["sv"] += v
                        placed = True
                        break
                if not placed:
                    clusters.append({"sx": p[0], "sy": p[1], "sz": p[2], "n": 1, "zmax": p[2],
                                     "su": u, "sv": v})
            good = 0
            for c in clusters:
                if c["n"] < 3 or c["zmax"] < tz + 0.008:
                    continue
                good += 1
                cx = c["sx"] / c["n"]
                cy = c["sy"] / c["n"]
                matched = None
                for t in TRACKS:
                    if t["color"] == name and (t["x"] - cx) ** 2 + (t["y"] - cy) ** 2 < 0.035 ** 2:
                        matched = t
                        break
                if matched is not None:
                    w = matched["w"] + c["n"]
                    matched["x"] = (matched["x"] * matched["w"] + cx * c["n"]) / w
                    matched["y"] = (matched["y"] * matched["w"] + cy * c["n"]) / w
                    matched["w"] = w
                    if matched["last_tick"] != ST["tick"]:
                        matched["ticks"] += 1
                    matched["last_tick"] = ST["tick"]
                    if cam not in matched["cams"]:
                        matched["cams"].append(cam)
                    if label == "postcover":
                        matched["seen_postcover"] += c["n"]
                elif allow_new:
                    TRACKS.append({"id": len(TRACKS), "color": name, "x": cx, "y": cy,
                                   "z": c["sz"] / c["n"], "w": c["n"], "ticks": 1,
                                   "first_tick": ST["tick"], "last_tick": ST["tick"],
                                   "cams": [cam], "seen_postcover": 0,
                                   "first_px": [cam, int(c["su"] / c["n"]), int(c["sv"] / c["n"])]})
            summary[cam + ":" + name] = good
    return summary


TIMELINE = []


def scan_tick(obs, phase, allow_new, want_black):
    if obs is None:
        return
    s = scan_obs(obs, ("front", "wrist"), allow_new, phase, want_black)
    key = str(sorted(s.items()))
    if not TIMELINE or TIMELINE[-1]["key"] != key:
        if len(TIMELINE) < 120:
            TIMELINE.append({"tick": ST["tick"], "phase": phase, "key": key, "counts": s,
                             "n_tracks": len(TRACKS)})


# ---------------------------------------------------------------- motion helpers
def solve(pos, quat, seed, label):
    info = {"label": label, "pos": [r4(v) for v in pos]}
    try:
        ik = solve_ik(Pose(list(pos), list(quat), ST["base"]), seed_joints=seed)
    except Exception as exc:
        note_error("solve_ik_" + label, exc)
        info["ok"] = False
        return None, info
    ok = False
    try:
        ok = bool(ik.ok)
    except Exception:
        ok = False
    info["ok"] = ok
    if not ok:
        try:
            info["error"] = str(ik.error)[:200]
        except Exception:
            pass
        return None, info
    joints = None
    try:
        joints = vec(ik.joint_positions)
    except Exception:
        joints = None
    if joints is None or len(joints) != 7:
        try:
            joints = vec(ik.joint_positions["primary"])
        except Exception as exc:
            note_error("ik_joints_" + label, exc)
            joints = None
    info["joints"] = [r4(v) for v in joints] if joints else None
    return joints, info


def servo(target, grip, rate, settle_max, tol, scan_phase, want_black, deadline_tick):
    """Joint-space linear interpolation by step(), then settle. Returns info dict."""
    info = {"start_tick": ST["tick"]}
    cur = joints_now()
    if cur is None or target is None:
        info["skipped"] = True
        return info
    delta = max(abs(target[k] - cur[k]) for k in range(7))
    n = int(math.ceil(delta / rate))
    if n < 1:
        n = 1
    info["n"] = n
    info["delta"] = r4(delta)
    for i in range(1, n + 1):
        if ST["ended"]:
            break
        if deadline_tick is not None and ST["tick"] >= deadline_tick:
            info["deadline_cut"] = ST["tick"]
            break
        frac = float(i) / n
        q = [cur[k] + (target[k] - cur[k]) * frac for k in range(7)]
        obs = hold_step(q, grip)
        if scan_phase is not None:
            scan_tick(obs, scan_phase, True, want_black)
    settle = 0
    err = None
    while settle < settle_max and not ST["ended"]:
        now = joints_now()
        if now is None:
            break
        err = max(abs(target[k] - now[k]) for k in range(7))
        if err < tol:
            break
        if deadline_tick is not None and ST["tick"] >= deadline_tick:
            info["deadline_cut"] = ST["tick"]
            break
        obs = hold_step(target, grip)
        if scan_phase is not None:
            scan_tick(obs, scan_phase, True, want_black)
        settle += 1
    now = joints_now()
    if now is not None:
        err = max(abs(target[k] - now[k]) for k in range(7))
    info["settle"] = settle
    info["final_err"] = r4(err)
    info["end_tick"] = ST["tick"]
    return info


def ee_pos():
    state = robot_state()
    if state is None:
        return None
    try:
        return vec(state.end_effector_poses["primary"].position)
    except Exception as exc:
        note_error("ee_pos", exc)
        return None


def move_pose(pos, quat, grip, label, rate, settle_max, tol, log):
    seed = joints_now()
    joints, ik_info = solve(pos, quat, seed, label)
    entry = {"ik": ik_info}
    if joints is not None:
        entry["servo"] = servo(joints, grip, rate, settle_max, tol, None, False, None)
        p = ee_pos()
        entry["ee_after"] = [r4(v) for v in p] if p else None
        if p:
            entry["pos_err"] = r4(math.sqrt(sum((p[k] - pos[k]) ** 2 for k in range(3))))
    log.append(entry)
    return joints is not None, entry


# ---------------------------------------------------------------- goal parsing
def parse_goal(text):
    low = text.lower()
    for ch in ",.;:!?()":
        low = low.replace(ch, " ")
    words = low.split()
    start = 0
    for i in range(len(words)):
        if words[i] == "pick":
            start = i
            break
    plan = []
    all_flag = False
    for i in range(start, len(words)):
        w = words[i]
        if w == "all" or w == "every" or w == "each":
            all_flag = True
        if w in COLOR_NAMES:
            name = "gray" if w == "grey" else w
            nxt = words[i + 1] if i + 1 < len(words) else ""
            plural = nxt.endswith("s") and nxt != "is"
            plan.append({"color": name, "all": all_flag or plural})
            all_flag = False
    return plan


# ---------------------------------------------------------------- button localization
def cloud_stats(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return {"n": len(points),
            "q05": [r4(quantile(xs, 0.05)), r4(quantile(ys, 0.05)), r4(quantile(zs, 0.05))],
            "q50": [r4(quantile(xs, 0.5)), r4(quantile(ys, 0.5)), r4(quantile(zs, 0.5))],
            "q95": [r4(quantile(xs, 0.95)), r4(quantile(ys, 0.95)), r4(quantile(zs, 0.95))]}


def locate_button():
    rep = {"candidates": []}
    tz = ST["table_z"]
    best = None
    for cam in ("wrist", "front"):
        for prompt in ("button", "round button"):
            try:
                seg = segment_text(cam, prompt)
            except Exception as exc:
                note_error("segment_button_" + cam, exc)
                continue
            try:
                if not bool(seg.ok):
                    rep["candidates"].append({"cam": cam, "prompt": prompt, "ok": False})
                    continue
                cands = seg.segmentations
            except Exception as exc:
                note_error("seg_fields", exc)
                continue
            k = 0
            for cand in cands:
                if k >= 4:
                    break
                k += 1
                entry = {"cam": cam, "prompt": prompt}
                try:
                    entry["score"] = r4(cand.score)
                    entry["box"] = [r4(v) for v in cand.box_xyxy]
                except Exception:
                    pass
                try:
                    pc = mask_to_point_cloud(cand, cam, target_frame=ST["base"])
                    pts = rows_of(pc.points)
                except Exception as exc:
                    entry["pc_error"] = str(exc)[:160]
                    rep["candidates"].append(entry)
                    continue
                if len(pts) < 10:
                    entry["n"] = len(pts)
                    rep["candidates"].append(entry)
                    continue
                st = cloud_stats(pts)
                entry["stats"] = st
                ext_x = st["q95"][0] - st["q05"][0]
                ext_y = st["q95"][1] - st["q05"][1]
                ztop = st["q95"][2]
                plausible = ext_x < 0.12 and ext_y < 0.12 and ext_x > 0.01 and ext_y > 0.01
                if tz is not None:
                    plausible = plausible and ztop > tz + 0.004 and ztop < tz + 0.09
                entry["plausible"] = plausible
                rep["candidates"].append(entry)
                if plausible:
                    zs = sorted(p[2] for p in pts)
                    zt = zs[int(0.95 * (len(zs) - 1))]
                    top = [p for p in pts if p[2] >= zt - 0.006]
                    cx = sum(p[0] for p in top) / len(top)
                    cy = sum(p[1] for p in top) / len(top)
                    cand_score = 0.0
                    try:
                        cand_score = float(cand.score)
                    except Exception:
                        pass
                    rank = cand_score + (0.5 if cam == "wrist" else 0.0)
                    if best is None or rank > best["rank"]:
                        best = {"x": cx, "y": cy, "ztop": zt, "rank": rank, "cam": cam,
                                "prompt": prompt, "n": len(pts)}
            if best is not None and best["cam"] == cam:
                break
    rep["chosen"] = None if best is None else {"x": r4(best["x"]), "y": r4(best["y"]),
                                               "ztop": r4(best["ztop"]), "cam": best["cam"],
                                               "prompt": best["prompt"]}
    return best, rep


def locate_button_depth(obs):
    """Fallback: compact elevated cluster from own projection (no segmentation)."""
    if obs is None or ST["hyp"] is None or ST["table_z"] is None:
        return None
    tz = ST["table_z"]
    pts = []
    for cam in ("wrist", "front"):
        try:
            cobs = obs.cameras[cam]
            cd = cam_data(cobs)
            dep = rows_of(cobs.depth_m)
        except Exception as exc:
            note_error("btn_depth_" + cam, exc)
            continue
        for v in range(0, len(dep), 3):
            for u in range(0, len(dep[0]), 3):
                d = float(dep[v][u])
                if not valid_depth(d):
                    continue
                p = project(cd, u, v, d, ST["hyp"])
                if tz + 0.006 < p[2] < tz + 0.07:
                    pts.append(p)
    clusters = []
    for p in pts:
        placed = False
        for c in clusters:
            if (p[0] - c["x"]) ** 2 + (p[1] - c["y"]) ** 2 < 0.05 ** 2:
                c["pts"].append(p)
                placed = True
                break
        if not placed:
            clusters.append({"x": p[0], "y": p[1], "pts": [p]})
    best = None
    for c in clusters:
        if len(c["pts"]) < 15:
            continue
        xs = [p[0] for p in c["pts"]]
        ys = [p[1] for p in c["pts"]]
        ext = max(quantile(xs, 0.95) - quantile(xs, 0.05), quantile(ys, 0.95) - quantile(ys, 0.05))
        if ext > 0.1:
            continue
        if best is None or len(c["pts"]) > len(best["pts"]):
            best = c
    if best is None:
        return None
    zs = sorted(p[2] for p in best["pts"])
    zt = zs[int(0.95 * (len(zs) - 1))]
    top = [p for p in best["pts"] if p[2] >= zt - 0.006]
    return {"x": sum(p[0] for p in top) / len(top), "y": sum(p[1] for p in top) / len(top),
            "ztop": zt, "cam": "depth", "prompt": None, "n": len(best["pts"])}


# ---------------------------------------------------------------- container measurement
def container_points(obs, cx, cy, radius, cams):
    tz = ST["table_z"]
    zcap = tz + 0.40
    ee = ee_pos()
    if ee is not None and ee[2] - 0.04 < zcap:
        zcap = ee[2] - 0.04  # exclude the robot's own fingers/hand
    pts = []
    for cam in cams:
        try:
            cobs = obs.cameras[cam]
            cd = cam_data(cobs)
            dep = rows_of(cobs.depth_m)
        except Exception as exc:
            note_error("cont_pts_" + cam, exc)
            continue
        for v in range(0, len(dep), 2):
            drow = dep[v]
            for u in range(0, len(drow), 2):
                d = float(drow[u])
                if not valid_depth(d):
                    continue
                p = project(cd, u, v, d, ST["hyp"])
                if p[2] < tz + 0.01 or p[2] > zcap:
                    continue
                if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 > radius * radius:
                    continue
                pts.append(p)
    return pts


def measure_container(obs, tx, ty, cams):
    out = {"cams": list(cams)}
    if obs is None or ST["hyp"] is None:
        out["ok"] = False
        return out
    pts = container_points(obs, tx, ty, 0.07, cams)
    return analyze_container(pts, tx, ty, out)


def analyze_container(pts, tx, ty, out):
    out["n_near"] = len(pts)
    if len(pts) < 12 or ST["table_z"] is None:
        out["ok"] = False
        return out
    # drop tall structures (e.g. an occluding pillar): 1.5 cm cells holding any point above
    # table + 0.13 m, plus their 8 neighbours
    tall = {}
    for p in pts:
        if p[2] > ST["table_z"] + 0.13:
            ix = int(math.floor(p[0] / 0.015))
            iy = int(math.floor(p[1] / 0.015))
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    tall[str(ix + ddx) + "," + str(iy + ddy)] = True
    if tall:
        keep = []
        for p in pts:
            key = str(int(math.floor(p[0] / 0.015))) + "," + str(int(math.floor(p[1] / 0.015)))
            if key not in tall and p[2] <= ST["table_z"] + 0.13:
                keep.append(p)
        out["n_tall_removed"] = len(pts) - len(keep)
        pts = keep
        if len(pts) < 12:
            out["ok"] = False
            return out
    cx, cy = tx, ty
    top = []
    zt = None
    for it in range(3):
        sel = [p for p in pts if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 < 0.038 ** 2]
        if len(sel) < 8:
            break
        zt = quantile([p[2] for p in sel], 0.9)
        top = [p for p in sel if p[2] > zt - 0.01]
        if len(top) < 5:
            break
        cx = sum(p[0] for p in top) / len(top)
        cy = sum(p[1] for p in top) / len(top)
    if zt is None or len(top) < 5:
        out["ok"] = False
        return out
    best_a = 0.0
    best_area = None
    best_ext = None
    for k in range(30):
        a = k * (math.pi / 2.0) / 30.0
        ca = math.cos(a)
        sa = math.sin(a)
        p1 = [(p[0] - cx) * ca + (p[1] - cy) * sa for p in top]
        p2 = [-(p[0] - cx) * sa + (p[1] - cy) * ca for p in top]
        e1 = quantile(p1, 0.97) - quantile(p1, 0.03)
        e2 = quantile(p2, 0.97) - quantile(p2, 0.03)
        if best_area is None or e1 * e2 < best_area:
            best_area = e1 * e2
            best_a = a
            best_ext = [e1, e2]
    out.update({"ok": True, "cx": cx, "cy": cy, "ztop": zt, "yaw": best_a,
                "ext": [r4(best_ext[0]), r4(best_ext[1])], "n_top": len(top),
                "height": r4(zt - ST["table_z"])})
    # obstacle points around the two candidate finger axes (full-open fingers ~ +-0.045)
    clear = []
    for axis in (best_a, best_a + math.pi / 2.0):
        ca = math.cos(axis)
        sa = math.sin(axis)
        count = 0
        for sign in (1.0, -1.0):
            fx = cx + sign * 0.047 * ca
            fy = cy + sign * 0.047 * sa
            for p in pts:
                if p[2] < ST["table_z"] + 0.012:
                    continue
                if (p[0] - fx) ** 2 + (p[1] - fy) ** 2 < 0.013 ** 2:
                    count += 1
        clear.append(count)
    out["finger_obstacles"] = clear
    return out


def measure_container_api(tx, ty, cams):
    """Fallback without own projection: SAM text masks of containers -> mask_to_point_cloud."""
    out = {"cams": list(cams), "api": True, "cands": []}
    best = None
    for cam in cams:
        for prompt in ("white box", "white container"):
            try:
                seg = segment_text(cam, prompt)
                if not bool(seg.ok):
                    continue
                cands = seg.segmentations
            except Exception as exc:
                note_error("cont_api_seg", exc)
                continue
            k = 0
            for cand in cands:
                if k >= 12:
                    break
                k += 1
                try:
                    pts = rows_of(mask_to_point_cloud(cand, cam, target_frame=ST["base"]).points)
                except Exception as exc:
                    note_error("cont_api_pc", exc)
                    continue
                if len(pts) < 12:
                    continue
                mx = quantile([p[0] for p in pts], 0.5)
                my = quantile([p[1] for p in pts], 0.5)
                dist = math.sqrt((mx - tx) ** 2 + (my - ty) ** 2)
                out["cands"].append([cam, prompt, r4(mx), r4(my), r4(dist), len(pts)])
                if dist < 0.05 and (best is None or dist < best[0]):
                    best = [dist, pts]
            if best is not None:
                break
        if best is not None:
            break
    if best is None:
        out["ok"] = False
        return out
    if ST["table_z"] is None:
        ST["table_z"] = quantile([p[2] for p in best[1]], 0.03)
        note("table_z_from_container_cloud", r4(ST["table_z"]))
    return analyze_container(best[1], tx, ty, out)


def sam_color_probe(colors, phase):
    """Independent API-only cube localization: text masks per goal colour on both cameras."""
    found = []
    for cam in ("front", "wrist"):
        for color in colors:
            prompt = color + " cube"
            try:
                seg = segment_text(cam, prompt)
                if not bool(seg.ok):
                    continue
                cands = seg.segmentations
            except Exception as exc:
                note_error("sam_probe", exc)
                continue
            k = 0
            for cand in cands:
                if k >= 4:
                    break
                k += 1
                entry = {"cam": cam, "color": color, "phase": phase, "tick": ST["tick"]}
                try:
                    entry["score"] = r4(cand.score)
                except Exception:
                    pass
                try:
                    pts = rows_of(mask_to_point_cloud(cand, cam, target_frame=ST["base"]).points)
                except Exception as exc:
                    entry["pc_error"] = str(exc)[:120]
                    found.append(entry)
                    continue
                if len(pts) < 5:
                    entry["n"] = len(pts)
                    found.append(entry)
                    continue
                st = cloud_stats(pts)
                entry["stats"] = st
                ext = max(st["q95"][0] - st["q05"][0], st["q95"][1] - st["q05"][1])
                entry["small"] = ext < 0.06
                found.append(entry)
    return found


def color_near(obs, color, x, y, radius):
    """Count pixels of the colour within radius of (x, y) (revealed-cube check)."""
    if obs is None or ST["hyp"] is None:
        return None
    tz = ST["table_z"]
    total = 0
    for cam in ("front", "wrist"):
        try:
            cobs = obs.cameras[cam]
            cd = cam_data(cobs)
            rgb = rows_of(cobs.rgb)
            dep = rows_of(cobs.depth_m)
        except Exception as exc:
            note_error("color_near", exc)
            continue
        for v in range(0, len(dep), 2):
            for u in range(0, len(dep[0]), 2):
                px = rgb[v][u]
                name = classify(int(px[0]), int(px[1]), int(px[2]), color == "black")
                if name != color:
                    continue
                d = float(dep[v][u])
                if not valid_depth(d):
                    continue
                p = project(cd, u, v, d, ST["hyp"])
                if p[2] < tz - 0.01 or p[2] > tz + 0.08:
                    continue
                if (p[0] - x) ** 2 + (p[1] - y) ** 2 < radius * radius:
                    total += 1
    return total


# ================================================================ main
try:
    ctx = get_task_context()
    goal = str(ctx.language)
    R["goal"] = goal
    try:
        R["task_name"] = str(ctx.task_name)
    except Exception:
        pass
    md = None
    try:
        md = robomme_plus.get_task_metadata()
        R["task_metadata"] = str(md)[:3000]
    except Exception as exc:
        note_error("get_task_metadata", exc)
    try:
        cm = robomme_plus.get_controller_metadata()
        R["controller_metadata"] = str(cm)[:3000]
    except Exception as exc:
        note_error("get_controller_metadata", exc)
    if md is not None:
        for key in ("max_steps", "max_episode_steps", "horizon"):
            try:
                ST["max_steps"] = int(md[key])
                R["max_steps"] = ST["max_steps"]
                break
            except Exception:
                pass
        for key in ("control_hz", "control_freq", "control_frequency", "frequency_hz"):
            try:
                ST["hz"] = float(md[key])
                R["control_hz"] = ST["hz"]
                break
            except Exception:
                pass
    plan = parse_goal(goal)
    R["plan"] = plan
    want_black = False
    for item in plan:
        if item["color"] == "black":
            want_black = True
    state0 = get_robot_state()
    ST["emb"] = str(state0.embodiment)
    ST["base"] = str(state0.base_frame)
    update_tick(state0.timestamp_s)
    home = vec(state0.joint_positions["primary"])
    grip0 = float(state0.gripper_positions["primary"])
    ee0 = state0.end_effector_poses["primary"]
    ee0_pos = vec(ee0.position)
    ee0_q = vec(ee0.quaternion_wxyz)
    m0 = quat_mat(ee0_q)
    tool_z = [m0[0][2], m0[1][2], m0[2][2]]
    home_yaw = math.atan2(m0[1][0], m0[0][0])
    R["home"] = {"joints": [r4(v) for v in home], "grip": r4(grip0),
                 "ee_pos": [r4(v) for v in ee0_pos], "ee_q": [r4(v) for v in ee0_q],
                 "tool_z_axis": [r4(v) for v in tool_z], "home_yaw_deg": r4(math.degrees(home_yaw)),
                 "embodiment": ST["emb"], "base_frame": ST["base"], "tick0": ST["tick"]}
    obs0 = get_observation()
    R["cameras"] = [str(name) for name in obs0.cameras]
    try:
        video = robomme_plus.get_conditioning_video()
        R["video_len"] = len(video)
        if len(video) > 0:
            R["video_ts"] = [r4(video[0].timestamp_s), r4(video[len(video) - 1].timestamp_s)]
    except Exception as exc:
        video = []
        note_error("get_conditioning_video", exc)
    calib = calibrate(obs0)
    R["phases"]["calibration"] = calib
    if ST["table_z"] is None:
        api_z = [e["api_mean"][2] for e in calib["api_checks"] if e.get("api_mean")]
        if api_z:
            ST["table_z"] = quantile(api_z, 0.5)
            note("table_z_from_api_patches", r4(ST["table_z"]))
    goal_colors = []
    for item in plan:
        if item["color"] not in goal_colors:
            goal_colors.append(item["color"])
    R["sam_probe"] = []
    # conditioning video (expected empty for this task): scan a bounded subsample
    if ST["hyp"] is not None and len(video) > 0:
        stride = int(math.ceil(len(video) / 96.0))
        k = 0
        while k < len(video):
            scan_tick(video[k], "video", True, want_black)
            k += stride
    # press orientation: home orientation if it points down, else synthetic top-down
    down_ok = tool_z[2] < -0.94
    press_q = ee0_q if down_ok else topdown_quat(home_yaw)
    R["home"]["tool_points_down"] = down_ok
    button, brep = locate_button()
    R["phases"]["button"] = brep
    if button is None:
        button = locate_button_depth(obs0)
        R["phases"]["button_depth_fallback"] = None if button is None else {
            "x": r4(button["x"]), "y": r4(button["y"]), "ztop": r4(button["ztop"]), "n": button["n"]}
    press_plan = {}
    pre_j = None
    press_j = None
    if button is not None:
        bx, by, bz = button["x"], button["y"], button["ztop"]
        pre_pos = [bx, by, bz + 0.05]
        press_pos = [bx, by, bz - 0.012]
        deep_pos = [bx, by, bz - 0.022]
        pre_j, i1 = solve(pre_pos, press_q, home, "pre_press")
        press_j, i2 = solve(press_pos, press_q, pre_j if pre_j else home, "press")
        deep_j, i3 = solve(deep_pos, press_q, press_j if press_j else home, "press_deep")
        press_plan = {"ik_pre": i1, "ik_press": i2, "ik_deep": i3}
    R["phases"]["press_plan"] = press_plan

    # ---------------- phase A: hold at home, scan every tick until departure
    RATE_FAST = 0.07
    need = 30
    if pre_j is not None and press_j is not None:
        d1 = max(abs(pre_j[k] - home[k]) for k in range(7))
        d2 = max(abs(press_j[k] - pre_j[k]) for k in range(7))
        need = int(math.ceil(d1 / RATE_FAST)) + int(math.ceil(d2 / 0.05)) + 12
    depart = 84 - need
    if depart > 59:
        depart = 59
    if depart < 1:
        depart = 1
    R["phases"]["schedule"] = {"need_ticks": need, "depart_tick": depart}
    scan_tick(obs0, "home", True, want_black)
    hold_grip = grip0
    probed_mid = False
    while not ST["ended"] and ST["tick"] < depart:
        obs = hold_step(home, hold_grip)
        if obs is None:
            obs = observe()
        scan_tick(obs, "home", True, want_black)
        if not probed_mid and ST["tick"] >= 30 and depart > 32:
            probed_mid = True
            R["sam_probe"] = R["sam_probe"] + sam_color_probe(goal_colors, "home_mid")
        if ST["steps_called"] > 120:
            break
    if not ST["ended"]:
        R["sam_probe"] = R["sam_probe"] + sam_color_probe(goal_colors, "pre_depart")
    R["phases"]["home_wait_end_tick"] = ST["tick"]
    R["phases"]["tick_check"] = {"tick": ST["tick"], "steps_called": ST["steps_called"],
                                 "ts": r4(ST["ts"]) if ST["ts"] is not None else None}
    if ST["hyp"] is None:
        # API-only fallback memory: small SAM colour-cube clouds become tracks
        for e in R["sam_probe"]:
            if not e.get("small"):
                continue
            ex = e["stats"]["q50"][0]
            ey = e["stats"]["q50"][1]
            hit = None
            for t in TRACKS:
                if t["color"] == e["color"] and (t["x"] - ex) ** 2 + (t["y"] - ey) ** 2 < 0.035 ** 2:
                    hit = t
            if hit is None:
                TRACKS.append({"id": len(TRACKS), "color": e["color"], "x": ex, "y": ey, "z": 0.0,
                               "w": 8, "ticks": 2, "first_tick": e["tick"], "last_tick": e["tick"],
                               "cams": [e["cam"]], "seen_postcover": 0, "first_px": None,
                               "from_sam": True})

    # ---------------- phase B: press the button before step 90
    press_rep = {"depart_tick": ST["tick"]}
    if pre_j is not None and press_j is not None and not ST["ended"]:
        press_rep["to_pre"] = servo(pre_j, 0.0, RATE_FAST, 4, 0.03, "move", want_black, 84)
        press_rep["ee_at_pre"] = [r4(v) for v in (ee_pos() or [])]
        press_rep["to_press"] = servo(press_j, 0.0, 0.05, 4, 0.01, "press", want_black, 87)
        p = ee_pos()
        press_rep["ee_at_press"] = [r4(v) for v in (p or [])]
        press_rep["tick_at_press"] = ST["tick"]
        if p is not None and button is not None and p[2] > button["ztop"] - 0.004 and deep_j is not None \
                and ST["tick"] < 86:
            press_rep["deep"] = servo(deep_j, 0.0, 0.05, 2, 0.01, "press", want_black, 88)
            p = ee_pos()
            press_rep["ee_at_deep"] = [r4(v) for v in (p or [])]
        holds = 0
        hold_target = press_j
        while not ST["ended"] and (ST["tick"] < 89 or holds < 4) and holds < 20:
            obs = hold_step(hold_target, 0.0)
            scan_tick(obs, "press_hold", ST["tick"] < 88, want_black)
            holds += 1
        press_rep["hold_ticks"] = holds
        press_rep["ee_min_z_hint"] = r4((ee_pos() or [0, 0, 0])[2])
        press_rep["rise"] = servo(pre_j, 0.0, 0.05, 3, 0.03, None, False, None)
    else:
        press_rep["skipped"] = True
    press_rep["end_tick"] = ST["tick"]
    R["phases"]["press"] = press_rep

    # ---------------- phase C: return home for an overview after covering
    over = {}
    if not ST["ended"]:
        over["home"] = servo(home, 0.0, 0.06, 6, 0.02, None, False, None)
        while not ST["ended"] and ST["tick"] < 96:
            hold_step(home, 0.0)
        obs = observe()
        scan_tick(obs, "postcover", False, want_black)
        over["tick"] = ST["tick"]
        try:
            for prompt in ("white box", "container"):
                seg = segment_text("front", prompt)
                entry = {"prompt": prompt, "ok": bool(seg.ok)}
                if seg.ok:
                    entry["n"] = len(seg.segmentations)
                    entry["scores"] = [r4(c.score) for c in seg.segmentations[:8]]
                over["seg_" + prompt.replace(" ", "_")] = entry
        except Exception as exc:
            note_error("probe_containers", exc)
        nmeas = 0
        for t in TRACKS:
            if (t["ticks"] >= 2 or t["w"] >= 8) and nmeas < 16 and ST["hyp"] is not None:
                nmeas += 1
                mc = measure_container(obs, t["x"], t["y"], ("front", "wrist"))
                t["container0"] = mc
    R["phases"]["overview"] = over

    # merge duplicate tracks of the same colour
    merged = []
    for t in TRACKS:
        if t["ticks"] < 2 and t["w"] < 8:
            continue
        hit = None
        for m in merged:
            if m["color"] == t["color"] and (m["x"] - t["x"]) ** 2 + (m["y"] - t["y"]) ** 2 < 0.03 ** 2:
                hit = m
                break
        if hit is None:
            merged.append(t)
        else:
            w = hit["w"] + t["w"]
            hit["x"] = (hit["x"] * hit["w"] + t["x"] * t["w"]) / w
            hit["y"] = (hit["y"] * hit["w"] + t["y"] * t["w"]) / w
            hit["w"] = w
            hit["ticks"] += t["ticks"]
    R["tracks"] = [{"id": t["id"], "color": t["color"], "x": r4(t["x"]), "y": r4(t["y"]),
                    "w": t["w"], "ticks": t["ticks"], "first": t["first_tick"],
                    "last": t["last_tick"], "cams": t["cams"], "postcover": t["seen_postcover"],
                    "px": t["first_px"],
                    "c0": [r4((t.get("container0") or {}).get(key)) for key in ("cx", "cy", "ztop", "yaw")],
                    "c0_ext": (t.get("container0") or {}).get("ext")} for t in TRACKS]
    R["merged_track_ids"] = [t["id"] for t in merged]

    # ---------------- target selection
    targets = []
    used = []
    for item in plan:
        cands = [t for t in merged if t["color"] == item["color"] and t["id"] not in used]
        hidden = [t for t in cands if t["seen_postcover"] < 3]
        if hidden:
            cands = hidden
        ordered = sorted([(-t["w"], t["id"]) for t in cands])
        chosen_ids = [pair[1] for pair in ordered]
        if not item["all"]:
            chosen_ids = chosen_ids[:1]
        for tid in chosen_ids:
            for t in merged:
                if t["id"] == tid:
                    targets.append(t)
                    used.append(tid)
        if not chosen_ids:
            note("no_track_for_color", item["color"])
    R["targets"] = [{"id": t["id"], "color": t["color"]} for t in targets]

    # ---------------- phase D: unmask -- grasp, lift, put back
    picks = []
    tz = ST["table_z"] if ST["table_z"] is not None else 0.0
    hover_z = tz + 0.24
    for t in targets:
        if ST["ended"]:
            break
        if ST["max_steps"] is not None and ST["tick"] > ST["max_steps"] - 60:
            note("out_of_ticks", ST["tick"])
            break
        pk = {"id": t["id"], "color": t["color"], "start_tick": ST["tick"], "moves": []}
        picks.append(pk)
        try:
            # hover above the remembered cube position
            hov_q = topdown_quat(home_yaw)
            ok, e = move_pose([t["x"], t["y"], hover_z], hov_q, 1.0, "hover", 0.06, 8, 0.02, pk["moves"])
            if ST["ended"]:
                break
            obs = observe()
            mc = measure_container(obs, t["x"], t["y"], ("wrist", "front"))
            pk["measure_hover"] = {k2: (r4(v2) if k2 in ("cx", "cy", "ztop", "yaw") else v2)
                                   for k2, v2 in mc.items()}
            if not mc.get("ok"):
                mc = t.get("container0") or {"ok": False}
                pk["used_initial_measure"] = True
            if not mc.get("ok"):
                mc = measure_container_api(t["x"], t["y"], ("wrist", "front"))
                pk["measure_api"] = {k2: (r4(v2) if k2 in ("cx", "cy", "ztop", "yaw") else v2)
                                     for k2, v2 in mc.items()}
            if mc.get("ok"):
                cx, cy, ztop, yaw = mc["cx"], mc["cy"], mc["ztop"], mc["yaw"]
                ext = mc["ext"]
                clear = mc.get("finger_obstacles", [0, 0])
            else:
                cx, cy, ztop, yaw = t["x"], t["y"], tz + 0.05, 0.0
                ext = [0.05, 0.05]
                clear = [0, 0]
                pk["blind"] = True
            height = ztop - tz
            depth_in = 0.02 if height > 0.05 else 0.4 * height
            grasp_z = ztop - depth_in
            # finger closing axis: one of the two face normals (yaw, yaw+90); prefer clearance,
            # and fall back to the other face pair when IK rejects the first
            axes = [yaw, yaw + math.pi / 2.0]
            first = 0 if clear[0] <= clear[1] else 1
            order = [first, 1 - first]
            gq = None
            pk["ik_checks"] = []
            for idx in order:
                alpha = axes[idx]
                best_theta = None
                for kq in range(-2, 3):
                    th = alpha + math.pi / 2.0 + kq * math.pi
                    if best_theta is None or abs(wrap_pi(th - home_yaw)) < abs(wrap_pi(best_theta - home_yaw)):
                        best_theta = th
                theta_try = home_yaw + wrap_pi(best_theta - home_yaw)
                q_try = topdown_quat(theta_try)
                seed_now = joints_now()
                j_pre, info_pre = solve([cx, cy, ztop + 0.08], q_try, seed_now, "check_pre")
                j_gr, info_gr = solve([cx, cy, grasp_z], q_try, j_pre if j_pre else seed_now, "check_grasp")
                pk["ik_checks"].append([idx, info_pre["ok"], info_gr["ok"]])
                if j_pre is not None and j_gr is not None:
                    gq = q_try
                    theta = theta_try
                    width_along = ext[idx] if ext[idx] is not None else 0.05
                    clear_used = clear[idx]
                    break
            if gq is None:
                pk["skipped"] = "no IK for either face pair"
                pk["end_tick"] = ST["tick"]
                continue
            pk["grasp"] = {"cx": r4(cx), "cy": r4(cy), "ztop": r4(ztop), "grasp_z": r4(grasp_z),
                           "theta_deg": r4(math.degrees(theta)), "width_along": width_along,
                           "clear": clear, "clear_used": clear_used}
            open_cmd = 1.0
            if clear_used > 0 and width_along is not None and ST["grip_partial_ok"]:
                open_cmd = min(1.0, max(0.55, (width_along + 0.03) / 0.08))
            pk["open_cmd"] = r4(open_cmd)
            attempt = 0
            held = False
            while attempt < 2 and not held and not ST["ended"]:
                attempt += 1
                move_pose([cx, cy, ztop + 0.08], gq, open_cmd, "pregrasp", 0.06, 8, 0.015, pk["moves"])
                pk["grip_after_open_" + str(attempt)] = r4(grip_now())
                move_pose([cx, cy, grasp_z], gq, open_cmd, "grasp", 0.04, 10, 0.008, pk["moves"])
                if ST["ended"]:
                    break
                try:
                    cres = close_gripper()
                    pk["close_" + str(attempt)] = check_exec(cres, "close")
                except Exception as exc:
                    note_error("close_gripper", exc)
                g = grip_now()
                pk["grip_closed_" + str(attempt)] = r4(g)
                held = g is not None and g > 0.04
                if not held and not ST["ended"]:
                    # missed: open, back off, re-measure from above, retry once
                    move_pose([cx, cy, ztop + 0.08], gq, 1.0, "backoff", 0.06, 6, 0.02, pk["moves"])
                    obs = observe()
                    mc2 = measure_container(obs, cx, cy, ("wrist", "front"))
                    pk["remeasure"] = {k2: (r4(v2) if k2 in ("cx", "cy", "ztop", "yaw") else v2)
                                       for k2, v2 in mc2.items()}
                    if mc2.get("ok"):
                        cx, cy, ztop = mc2["cx"], mc2["cy"], mc2["ztop"]
                        grasp_z = ztop - depth_in
                    open_cmd = 1.0
            pk["held"] = held
            if ST["ended"]:
                break
            # lift with closed gripper commands
            move_pose([cx, cy, ztop + 0.10], gq, 0.0, "lift", 0.04, 8, 0.015, pk["moves"])
            pk["grip_after_lift"] = r4(grip_now())
            pk["tick_lifted"] = ST["tick"]
            if ST["ended"]:
                pk["ended_after_lift"] = True
                break
            obs = observe()
            pk["revealed_" + t["color"]] = color_near(obs, t["color"], t["x"], t["y"], 0.04)
            # put the container back where it was and release
            move_pose([cx, cy, grasp_z + 0.004], gq, 0.0, "lower", 0.04, 8, 0.01, pk["moves"])
            if ST["ended"]:
                break
            try:
                ores = open_gripper()
                pk["open_after"] = check_exec(ores, "open")
            except Exception as exc:
                note_error("open_gripper", exc)
            move_pose([cx, cy, ztop + 0.10], gq, 1.0, "retreat", 0.06, 6, 0.02, pk["moves"])
        except Exception as exc:
            note_error("pick_" + str(t["id"]), exc)
        pk["end_tick"] = ST["tick"]
    R["picks"] = picks
except Exception as exc:
    note_error("main", exc)

R["timeline"] = TIMELINE
R["final_tick"] = ST["tick"]
R["ended_flag"] = ST["ended"]
R["steps_called"] = ST["steps_called"]
result = R
