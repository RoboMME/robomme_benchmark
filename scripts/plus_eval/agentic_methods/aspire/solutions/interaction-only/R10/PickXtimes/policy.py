# v02 — PickXtimes: moving-target intercept with global phase re-sync after each lift and a
#        "target stopped under the cube" placement check (the target pauses while the cube rests on it).
# Goal (from language): pick the <colour> cube, place it on the target N times, then press button.
import math

CAM = "front"
COLOR_CHROMA = {
    "red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0),
    "yellow": (0.5, 0.5, 0.0), "purple": (0.45, 0.1, 0.45), "magenta": (0.5, 0.0, 0.5),
    "pink": (0.5, 0.2, 0.3), "cyan": (0.0, 0.5, 0.5), "orange": (0.65, 0.3, 0.05),
}
NUMBER_WORDS = {
    "once": 1, "twice": 2, "thrice": 3, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

context = get_task_context()
language = str(context.language)
state0 = get_robot_state()
BASE = state0.base_frame
report = {"language": language, "events": []}


def log(kind, info):
    entry = {"kind": kind}
    for key in info:
        entry[key] = info[key]
    report["events"].append(entry)


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def parse_goal(text):
    words = text.lower().replace(",", " ").replace(".", " ").split()
    color = None
    for i in range(len(words)):
        if words[i] in ("cube", "block") and i > 0 and words[i - 1] in COLOR_CHROMA:
            color = words[i - 1]
            break
    if color is None:
        for w in words:
            if w in COLOR_CHROMA:
                color = w
                break
    count = None
    for i in range(len(words)):
        w = words[i]
        if w in NUMBER_WORDS and (w in ("once", "twice", "thrice") or
                                  (i + 1 < len(words) and words[i + 1] in ("times", "time"))):
            count = NUMBER_WORDS[w]
            break
        if w.isdigit() and i + 1 < len(words) and words[i + 1] in ("times", "time"):
            count = int(w)
            break
    return color, count


def chroma(rgb):
    total = float(rgb[0]) + float(rgb[1]) + float(rgb[2]) + 1e-6
    return (float(rgb[0]) / total, float(rgb[1]) / total, float(rgb[2]) / total)


def chroma_dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def cloud_stats(cand, camera):
    pc = mask_to_point_cloud(cand, camera, target_frame=BASE)
    pts = pc.points
    if len(pts) < 15:
        return None
    zs = [float(p[2]) for p in pts]
    stats = {"pts": pts, "n": len(pts), "z95": quantile(zs, 0.95), "z05": quantile(zs, 0.05)}
    stats["xspan"] = quantile((p[0] for p in pts), 0.95) - quantile((p[0] for p in pts), 0.05)
    stats["yspan"] = quantile((p[1] for p in pts), 0.95) - quantile((p[1] for p in pts), 0.05)
    rgb = None
    try:
        cols = pc.colors
        if cols is not None and len(cols) > 0:
            rgb = (quantile((c[0] for c in cols), 0.5), quantile((c[1] for c in cols), 0.5),
                   quantile((c[2] for c in cols), 0.5))
    except Exception:
        rgb = None
    stats["rgb"] = rgb
    return stats


def rect_fit(xy):
    """Minimum-area rectangle over 2-degree steps; robust 3/97% quantile extents."""
    best = None
    for k in range(45):
        a = math.radians(2.0 * k)
        c = math.cos(a)
        s = math.sin(a)
        us = [p[0] * c + p[1] * s for p in xy]
        vs = [-p[0] * s + p[1] * c for p in xy]
        u0 = quantile(us, 0.03)
        u1 = quantile(us, 0.97)
        v0 = quantile(vs, 0.03)
        v1 = quantile(vs, 0.97)
        area = (u1 - u0) * (v1 - v0)
        if best is None or area < best[0]:
            uc = 0.5 * (u0 + u1)
            vc = 0.5 * (v0 + v1)
            best = (area, a, uc * c - vc * s, uc * s + vc * c, u1 - u0, v1 - v0)
    yaw = best[1]
    if yaw >= math.pi / 4:
        yaw -= math.pi / 2
    return {"yaw": yaw, "x": best[2], "y": best[3], "side": max(best[4], best[5])}


def top_face(stats, depth=0.006):
    ztop = stats["z95"]
    return [(float(p[0]), float(p[1])) for p in stats["pts"] if float(p[2]) >= ztop - depth], ztop


def find_cube(color):
    ref = COLOR_CHROMA.get(color)
    prompts = []
    if color is not None:
        prompts.append(color + " cube")
    prompts.append("cube")
    best = None
    for prompt in prompts:
        seg = segment_text(CAM, prompt)
        if not seg.ok:
            continue
        for cand in seg.segmentations[:6]:
            st = cloud_stats(cand, CAM)
            if st is None or st["rgb"] is None:
                continue
            if not (0.015 <= st["z95"] <= 0.12 and 0.012 <= st["xspan"] <= 0.09 and 0.012 <= st["yspan"] <= 0.09):
                continue
            d = chroma_dist(chroma(st["rgb"]), ref) if ref is not None else 0.0
            if d > 0.3:
                continue
            key = st["n"] - 300.0 * d
            if best is None or key > best[0]:
                best = (key, st, prompt, float(cand.score))
        if best is not None:
            break
    if best is None:
        return None
    st = best[1]
    xy, ztop = top_face(st)
    if len(xy) < 8:
        return None
    fit = rect_fit(xy)
    fit["ztop"] = ztop
    fit["prompt"] = best[2]
    fit["score"] = best[3]
    fit["n"] = st["n"]
    return fit


def now_tick():
    return int(round(float(get_robot_state().timestamp_s) / DT))


def make_cmd(q, g):
    try:
        return ArmCommand("joint_position", q, gripper_position=g, embodiment=EMB)
    except TypeError:
        return ArmCommand("joint_position", q, gripper_position=g)


def hold(n, g):
    """Advance n control ticks holding the measured joints and gripper target g."""
    q = [float(v) for v in get_robot_state().joint_positions["primary"]]
    for k in range(n):
        res = step(RobotAction(arms={"primary": make_cmd(q, g)}))
        report["ticks"] = report.get("ticks", 0) + 1
        term = False
        try:
            term = bool(res.terminated) or bool(res.truncated)
        except Exception:
            term = False
        if term:
            report["ended_during"] = "hold"
            raise ValueError("episode ended during hold")


def interp(track, u):
    if u <= track[0][0]:
        return (track[0][1], track[0][2])
    for i in range(1, len(track)):
        if track[i][0] >= u:
            a = track[i - 1]
            b = track[i]
            w = (u - a[0]) / max(1e-9, (b[0] - a[0]))
            return (a[1] + w * (b[1] - a[1]), a[2] + w * (b[2] - a[2]))
    return (track[-1][1], track[-1][2])


def dist2(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


MOTION = {"mode": "static", "track": [], "T": None, "off": 0.0, "p": None, "v": (0.0, 0.0), "t": 0}


def predict(tq):
    m = MOTION
    if m["mode"] == "periodic":
        tr = m["track"]
        ts = tr[0][0]
        u = tq + m["off"] - ts
        u = u - m["T"] * math.floor(u / m["T"])
        return interp(tr, ts + u)
    if m["mode"] == "linear":
        dt = tq - m["t"]
        return (m["p"][0] + m["v"][0] * dt, m["p"][1] + m["v"][1] * dt)
    return m["p"]


def period_rms(track, T):
    errs = []
    for s in track:
        if s[0] + T <= track[-1][0]:
            q = interp(track, s[0] + T)
            errs.append((q[0] - s[1]) ** 2 + (q[1] - s[2]) ** 2)
    if len(errs) < 4:
        return None
    return math.sqrt(sum(errs) / len(errs))


def observe_motion(max_ticks, every):
    """Hold still and sample the target to classify static / periodic / linear motion."""
    track = []
    nref = []
    t0 = now_tick()
    first = find_target()
    if first is None:
        return False
    track.append((float(first["t"]), first["x"], first["y"]))
    nref.append(first["n"])
    dmax = 0.0
    closure = None
    while now_tick() - t0 < max_ticks:
        hold(every, 1.0)
        ob = find_target(prev=(track[-1][1], track[-1][2]), max_jump=0.08)
        if ob is None:
            continue
        nref.append(ob["n"])
        ns = sorted(nref)
        if ob["n"] < 0.6 * ns[len(ns) // 2]:
            continue
        track.append((float(ob["t"]), ob["x"], ob["y"]))
        d0 = dist2((ob["x"], ob["y"]), (track[0][1], track[0][2]))
        dmax = max(dmax, d0)
        el = ob["t"] - t0
        if el >= 30 and dmax < 0.004:
            MOTION["mode"] = "static"
            MOTION["p"] = (quantile((s[1] for s in track), 0.5), quantile((s[2] for s in track), 0.5))
            MOTION["track"] = track
            return True
        if closure is None and dmax > 0.05 and d0 < 0.02:
            closure = ob["t"]
        if closure is not None and ob["t"] - closure >= 18:
            best = None
            k = -24.0
            while k <= 24.0:
                T = (closure - t0) + k
                if T > 20:
                    r = period_rms(track, T)
                    if r is not None and (best is None or r < best[0]):
                        best = (r, T)
                k += 0.25
            if best is not None and best[0] < 0.008:
                MOTION["mode"] = "periodic"
                MOTION["track"] = track
                MOTION["T"] = best[1]
                MOTION["off"] = 0.0
                MOTION["rms"] = best[0]
                plen = 0.0
                for i in range(1, len(track)):
                    if track[i][0] <= track[0][0] + best[1]:
                        plen += dist2((track[i][1], track[i][2]), (track[i - 1][1], track[i - 1][2]))
                MOTION["speed"] = plen / best[1]
                return True
            closure = None
    # No closure: linear extrapolation from the last samples.
    MOTION["track"] = track
    if dmax < 0.01:
        MOTION["mode"] = "static"
        MOTION["p"] = (track[-1][1], track[-1][2])
        return True
    tail = track[-6:]
    dt = tail[-1][0] - tail[0][0]
    MOTION["mode"] = "linear"
    MOTION["p"] = (tail[-1][1], tail[-1][2])
    MOTION["t"] = tail[-1][0]
    MOTION["v"] = ((tail[-1][1] - tail[0][1]) / max(dt, 1.0), (tail[-1][2] - tail[0][2]) / max(dt, 1.0))
    MOTION["speed"] = math.sqrt(MOTION["v"][0] ** 2 + MOTION["v"][1] ** 2)
    return True


def total_err(obs, off):
    MOTION["off"] = off
    e = 0.0
    for o in obs:
        e += dist2(predict(o[0]), (o[1], o[2]))
    return e / len(obs)


def correct_phase(obs):
    """Re-synchronise the motion model from fresh target observations [(t, x, y), ...].
    Periodic mode searches the whole period, because the target pauses while the cube rests on it."""
    m = MOTION
    if not obs:
        return None
    last = obs[-1]
    if m["mode"] == "static":
        m["p"] = (last[1], last[2])
        return 0.0
    if m["mode"] == "linear":
        if len(obs) >= 2 and obs[-1][0] - obs[0][0] >= 2:
            dt = obs[-1][0] - obs[0][0]
            m["v"] = ((obs[-1][1] - obs[0][1]) / dt, (obs[-1][2] - obs[0][2]) / dt)
        m["p"] = (last[1], last[2])
        m["t"] = last[0]
        return 0.0
    base = m["off"]
    best = None
    k = 0.0
    while k < m["T"]:
        e = total_err(obs, base + k)
        if best is None or e < best[0]:
            best = (e, base + k)
        k += 0.5
    center = best[1]
    k = -0.5
    while k <= 0.5:
        e = total_err(obs, center + k)
        if e < best[0]:
            best = (e, center + k)
        k += 0.1
    if best[0] < 0.02:
        m["off"] = best[1] - m["T"] * math.floor((best[1]) / m["T"])
    else:
        m["off"] = base
    return round(best[0], 4)


def target_obs(n_obs, gap, g, min_frac):
    """Collect up to n_obs target observations gap ticks apart (holding gripper target g)."""
    out = []
    for i in range(n_obs):
        if i > 0:
            hold(gap, g)
        ob = find_target()
        if ob is not None and ob["n"] >= min_frac * MOTION.get("nref", 0):
            out.append((float(ob["t"]), ob["x"], ob["y"]))
    return out


def find_target(prev=None, max_jump=None):
    """Flat 'target' candidate; returns centre of its top surface, point count, timestamp."""
    seg = segment_text(CAM, "target")
    tick = now_tick()
    if not seg.ok:
        return None
    best = None
    for cand in seg.segmentations[:5]:
        st = cloud_stats(cand, CAM)
        if st is None:
            continue
        if st["z95"] > 0.03 or max(st["xspan"], st["yspan"]) < 0.02:
            continue
        xy, ztop = top_face(st, 0.004)
        if len(xy) < 10:
            continue
        cx = quantile((p[0] for p in xy), 0.5)
        cy = quantile((p[1] for p in xy), 0.5)
        if prev is not None and max_jump is not None:
            if math.sqrt((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2) > max_jump:
                continue
        sc = float(cand.score)
        if best is None or sc > best[0]:
            best = (sc, {"x": cx, "y": cy, "ztop": ztop, "n": len(xy), "score": sc, "t": tick,
                         "radius": 0.25 * (st["xspan"] + st["yspan"])})
    if best is None:
        return None
    return best[1]


def find_button():
    best = None
    for prompt in ("button", "white button", "push button"):
        seg = segment_text(CAM, prompt)
        if not seg.ok:
            continue
        for cand in seg.segmentations[:5]:
            st = cloud_stats(cand, CAM)
            if st is None or st["rgb"] is None:
                continue
            ch = chroma(st["rgb"])
            sat = max(ch) - min(ch)
            if sat > 0.12:
                continue
            if not (0.012 <= st["z95"] <= 0.09 and max(st["xspan"], st["yspan"]) <= 0.10):
                continue
            key = float(cand.score)
            if best is None or key > best[0]:
                best = (key, st, prompt)
        if best is not None:
            break
    if best is None:
        return None
    st = best[1]
    xy, ztop = top_face(st, 0.004)
    return {"x": quantile((p[0] for p in xy), 0.5), "y": quantile((p[1] for p in xy), 0.5),
            "ztop": ztop, "score": best[0], "prompt": best[2]}


def down_quat(yaw):
    return (0.0, math.cos(yaw / 2.0), math.sin(yaw / 2.0), 0.0)


def ended(res):
    try:
        return bool(res.terminated) or bool(res.truncated)
    except Exception:
        return False


def go(x, y, z, yaw, tol=0.01, steps=200, tag="move"):
    res = move_to_pose(Pose((x, y, z), down_quat(yaw), BASE), tolerance=tol, max_steps=steps)
    report["ticks"] = report.get("ticks", 0) + int(res.steps_executed)
    if not res.ok:
        log("move_fail", {"tag": tag, "status": str(res.status), "err": str(res.error)})
    if ended(res):
        report["ended_during"] = tag
        raise ValueError("episode ended during " + tag)
    return res


def grip(close):
    res = close_gripper() if close else open_gripper()
    report["ticks"] = report.get("ticks", 0) + int(res.steps_executed)
    if ended(res):
        report["ended_during"] = "close" if close else "open"
        raise ValueError("episode ended during gripper")
    return float(get_robot_state().gripper_positions["primary"])


color, count = parse_goal(language)
report["color"] = color
report["count"] = count
if count is None:
    count = 1
DT = 0.05
try:
    DT = 1.0 / float(robomme_plus.get_task_metadata()["control_frequency_hz"])
except Exception:
    DT = 0.05
EMB = state0.embodiment

target = find_target()
button = find_button()
cube = find_cube(color)
report["target"] = target
report["button"] = button
report["cube0"] = cube
if target is None or cube is None:
    raise ValueError("initial localization failed")
MOTION["nref"] = target["n"]
obstacles = []
if button is not None:
    obstacles.append((button["x"], button["y"], 0.08))
seg_all = segment_text(CAM, "cube")
if seg_all.ok:
    for cand in seg_all.segmentations[:6]:
        st = cloud_stats(cand, CAM)
        if st is None:
            continue
        cx = quantile((p[0] for p in st["pts"]), 0.5)
        cy = quantile((p[1] for p in st["pts"]), 0.5)
        if dist2((cx, cy), (cube["x"], cube["y"])) < 0.03:
            continue
        if 0.015 <= st["z95"] <= 0.12 and max(st["xspan"], st["yspan"]) <= 0.09:
            obstacles.append((cx, cy, 0.06))
report["obstacles"] = [[round(o[0], 3), round(o[1], 3)] for o in obstacles]

placed = 0
attempts = 0
grip(False)
try:
    observe_motion(330, 3)
    report["motion"] = {"mode": MOTION["mode"], "T": MOTION["T"], "rms": MOTION.get("rms"), "speed": MOTION.get("speed"),
                        "n_samples": len(MOTION["track"]), "ticks": now_tick()}
    while placed < count and attempts < count + 4:
        attempts += 1
        if cube is None:
            cube = find_cube(color)
            if cube is None:
                log("cube_lost", {"attempt": attempts})
                break
        side = min(max(cube["side"], 0.02), 0.07)
        grasp_z = cube["ztop"] - min(0.5 * side, 0.025)
        transit_z = max(cube["ztop"], target["ztop"] + side) + 0.10
        yaw = cube["yaw"]
        go(cube["x"], cube["y"], transit_z, yaw, tag="above_cube")
        go(cube["x"], cube["y"], grasp_z, yaw, tol=0.008, tag="descend_cube")
        width = grip(True)
        go(cube["x"], cube["y"], transit_z, yaw, tag="lift")
        width_after = float(get_robot_state().gripper_positions["primary"])
        if width_after < 0.1:
            log("grasp_miss", {"attempt": attempts, "width": width, "width_after": width_after})
            grip(False)
            cube = find_cube(color)
            continue
        release_z = target["ztop"] + (grasp_z - (cube["ztop"] - side)) + 0.006
        sync_obs = target_obs(2, 3, 0.0, 0.5)
        sync_try = 0
        while len(sync_obs) == 0 and sync_try < 3 and MOTION["mode"] != "static":
            sync_try += 1
            hold(3, 0.0)
            sync_obs = target_obs(2, 3, 0.0, 0.3)
        perr = correct_phase(sync_obs)
        late = 0
        t_drop = now_tick()
        p = predict(t_drop)
        for try_plan in range(3):
            # Choose the earliest reachable intercept point clear of obstacles.
            t_now = now_tick()
            here = (float(get_robot_state().end_effector_poses["primary"].position[0]),
                    float(get_robot_state().end_effector_poses["primary"].position[1]))
            plan = None
            horizon = 160
            if MOTION["mode"] == "periodic":
                horizon = int(MOTION["T"]) + 40
            for lead in range(0, horizon):
                p = predict(t_now + 30 + lead)
                need = 12 + int(30.0 * dist2(here, p)) + 12 + 4
                if MOTION["mode"] != "static" and 30 + lead < need:
                    continue
                r = math.sqrt(p[0] ** 2 + p[1] ** 2)
                if r < 0.3 or r > 0.78:
                    continue
                clear = True
                for ob in obstacles:
                    if dist2(p, (ob[0], ob[1])) < ob[2]:
                        clear = False
                if not clear:
                    continue
                plan = (t_now + 30 + lead, p)
                break
            if plan is None:
                plan = (t_now + 30, predict(t_now + 30))
            t_drop = plan[0]
            p = plan[1]
            go(p[0], p[1], transit_z, yaw, tag="above_drop")
            go(p[0], p[1], release_z, yaw, tol=0.008, tag="lower_drop")
            if MOTION["mode"] == "static":
                late = 0
                break
            late = now_tick() - (t_drop - 2)
            if late <= 2:
                while now_tick() < t_drop - 2:
                    hold(1, 0.0)
                break
            log("late", {"attempt": attempts, "late": late, "try": try_plan})
            go(p[0], p[1], transit_z, yaw, tag="relift_late")
        report.setdefault("drops", []).append({"t_drop": t_drop, "late": late, "p": [round(p[0], 3), round(p[1], 3)],
                                              "phase_err": perr})
        grip(False)
        go(p[0], p[1], transit_z, yaw, tag="retreat")
        cube = find_cube(color)
        on_target = False
        verdict = "unknown"
        stops = target_obs(2, 6, 1.0, 0.2)
        if len(stops) < 2:
            stops = stops + target_obs(2, 6, 1.0, 0.2)
        if cube is not None:
            dxy = dist2((cube["x"], cube["y"]), predict(t_drop))
            if MOTION["mode"] == "static":
                if stops:
                    dxy = dist2((cube["x"], cube["y"]), (stops[-1][1], stops[-1][2]))
                on_target = dxy < 0.03
                verdict = "static"
            elif len(stops) >= 2:
                disp = dist2((stops[0][1], stops[0][2]), (stops[-1][1], stops[-1][2]))
                gapt = max(1.0, stops[-1][0] - stops[0][0])
                near = dist2((cube["x"], cube["y"]), (stops[-1][1], stops[-1][2]))
                expect = MOTION.get("speed", 0.0) * gapt
                on_target = disp < max(0.006, 0.3 * expect) and near < 0.04
                verdict = ("stopped" if on_target else "moving_or_far") + ":" + str(round(disp, 4)) + "/" + str(round(expect, 4))
                dxy = near
            else:
                on_target = dxy < 0.015 and late <= 2
                verdict = "predicted"
            log("placed", {"attempt": attempts, "dxy": round(dxy, 4), "on_target": on_target, "late": late,
                           "verdict": verdict, "n_stop_obs": len(stops),
                           "width_held": round(width_after, 3), "ticks": now_tick()})
        if on_target:
            placed += 1
    report["placed"] = placed
    # Press the button.
    if button is not None and placed >= count:
        grip(True)
        bz = button["ztop"]
        go(button["x"], button["y"], bz + 0.08, 0.0, tag="above_button")
        go(button["x"], button["y"], bz + 0.01, 0.0, tol=0.008, tag="touch_button")
        res = go(button["x"], button["y"], bz - 0.015, 0.0, tol=0.005, steps=60, tag="press_button")
        log("press", {"status": str(res.status), "ticks": report.get("ticks", 0)})
        res = go(button["x"], button["y"], bz - 0.025, 0.0, tol=0.005, steps=60, tag="press_deeper")
        log("press2", {"status": str(res.status), "ticks": report.get("ticks", 0)})
        go(button["x"], button["y"], bz + 0.08, 0.0, tag="leave_button")
except ValueError as exc:
    report["stopped"] = str(exc)

report["placed"] = placed
report["motion_final"] = {"mode": MOTION["mode"], "T": MOTION["T"], "off": MOTION["off"]}
result = report
