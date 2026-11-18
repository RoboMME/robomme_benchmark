# PickXtimes v02 -- colour-verified cube selection (on top of v01 moving-target interception).
# Evidence (episode 2, v01): SAM3 'green cube' (score 0.92) segmented the YELLOW distractor cube;
# grasping a distractor terminates the episode (failure at close1).  v02 checks the mean mask RGB of
# every segmentation candidate against the colour word parsed from the goal (front and wrist), and
# falls back to the generic noun with the same colour check.
# Everything episode-dependent comes from get_task_context().language and live perception.
import math

LIFT_DZ = 0.12        # pick lift above the placement height (relative)
RETREAT_DZ = 0.06     # retreat above placement height after each release
HOVER_DZ = 0.16       # observation / transit height above the table surface
RELEASE_GAP = 0.012   # cube bottom height above target surface at release
MOVE_STEPS = 220

log = []
report = {"log": log}


def now_t():
    try:
        return float(get_robot_state().timestamp_s)
    except Exception:
        return -1.0


def is_terminal(res):
    term = False
    try:
        term = bool(res.terminated)
    except Exception:
        term = False
    try:
        term = term or bool(res.truncated)
    except Exception:
        pass
    return term


def record(name, res):
    entry = {"name": name}
    try:
        entry["ok"] = bool(res.ok)
    except Exception:
        entry["ok"] = None
    try:
        entry["status"] = str(res.status)
    except Exception:
        pass
    try:
        entry["steps"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        if res.error is not None:
            entry["error"] = str(res.error)[:300]
    except Exception:
        pass
    entry["terminal"] = is_terminal(res)
    entry["t"] = now_t()
    log.append(entry)
    return entry


END_MARK = "EPISODE_ENDED"


def check_end(res):
    if is_terminal(res):
        raise ValueError(END_MARK)


# ---------------- quaternion helpers (wxyz) ----------------
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def qrot(q, v):
    qv = (0.0, v[0], v[1], v[2])
    qc = (q[0], -q[1], -q[2], -q[3])
    r = qmul(qmul(q, qv), qc)
    return (r[1], r[2], r[3])


def qyaw(angle):
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


# ---------------- language parsing ----------------
NUMBER_WORDS = {
    "once": 1, "twice": 2, "thrice": 3,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


def parse_count(text):
    words = text.replace(",", " ").replace(".", " ").split()
    for i, word in enumerate(words):
        if word in ("times", "time") and i > 0:
            prev = words[i - 1]
            if prev in NUMBER_WORDS:
                return NUMBER_WORDS[prev]
            try:
                return int(prev)
            except Exception:
                pass
        if word in ("once", "twice", "thrice"):
            return NUMBER_WORDS[word]
    return None


def parse_object(text):
    key = "pick up the "
    i = text.find(key)
    if i < 0:
        return "cube"
    rest = text[i + len(key):]
    for stop in (" and ", ",", " then "):
        j = rest.find(stop)
        if j > 0:
            rest = rest[:j]
    return rest.strip()


# ---------------- perception helpers ----------------
def points_of(loc):
    try:
        pts = loc.point_cloud.points
        return [(float(p[0]), float(p[1]), float(p[2])) for p in pts]
    except Exception:
        return []


def locate(queries, camera, min_points=8):
    tried = []
    for query in queries:
        try:
            found = localize_object(query, camera_name=camera, target_frame=BASE)
        except Exception as exc:
            tried.append({"q": query, "exc": str(exc)[:200]})
            continue
        pts = points_of(found) if found.ok else []
        info = {"q": query, "ok": bool(found.ok), "n": len(pts)}
        try:
            segs = found.segmentation.segmentations
            info["scores"] = [round(float(s.score), 3) for s in segs[:3]]
        except Exception:
            try:
                info["score"] = round(float(found.segmentation.score), 3)
            except Exception:
                pass
        tried.append(info)
        if found.ok and len(pts) >= min_points:
            return pts, tried
    return None, tried


def summarize(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    return {
        "x": quantile(xs, 0.5), "y": quantile(ys, 0.5), "z": quantile(zs, 0.5),
        "zlo": quantile(zs, 0.1), "zhi": quantile(zs, 0.95), "n": len(pts),
    }


def top_face(pts):
    zhi = quantile([p[2] for p in pts], 0.95)
    top = [p for p in pts if p[2] >= zhi - 0.008]
    return top, zhi


def square_yaw(pts):
    """Yaw (rad, in [0, pi/2)) minimizing the 1.5%-trimmed bounding-box perimeter of xy points."""
    sample = pts
    if len(sample) > 400:
        stride = len(sample) // 400 + 1
        sample = sample[::stride]
    best = None
    for k in range(60):
        a = k * (math.pi / 2.0) / 60.0
        c = math.cos(a)
        s = math.sin(a)
        u = [p[0] * c + p[1] * s for p in sample]
        v = [-p[0] * s + p[1] * c for p in sample]
        eu = quantile(u, 0.985) - quantile(u, 0.015)
        ev = quantile(v, 0.985) - quantile(v, 0.015)
        area = eu + ev
        if best is None or area < best[0]:
            best = (area, a, eu, ev)
    return best[1], best[2], best[3]


def wrap_quarter(angle):
    while angle > math.pi / 4.0:
        angle -= math.pi / 2.0
    while angle < -math.pi / 4.0:
        angle += math.pi / 2.0
    return angle


# ---------------- motion helpers ----------------
def go(name, xyz, quat, tol=0.01, steps=MOVE_STEPS):
    pose = Pose((xyz[0], xyz[1], xyz[2]), quat, BASE)
    res = move_to_pose(pose, tolerance=tol, max_steps=steps)
    entry = record(name, res)
    check_end(res)
    if not entry["ok"]:
        res = move_to_pose(pose, tolerance=max(tol, 0.02), max_steps=steps)
        entry = record(name + "_retry", res)
        check_end(res)
    try:
        ee = get_robot_state().end_effector_poses["primary"].position
        entry["ee"] = [round(float(ee[0]), 4), round(float(ee[1]), 4), round(float(ee[2]), 4)]
    except Exception:
        pass
    return entry["ok"]


def gripper_reading():
    try:
        return float(get_robot_state().gripper_positions["primary"])
    except Exception:
        return -1.0


def do_open(name):
    res = open_gripper()
    record(name, res)
    check_end(res)


def do_close(name):
    res = close_gripper()
    entry = record(name, res)
    entry["grip"] = round(gripper_reading(), 4)
    check_end(res)
    return entry["grip"]




# ---------------- colour-verified object selection (v02) ----------------
def color_match(word, rgb):
    """True if a mean RGB triple is consistent with a colour word (None = no colour constraint)."""
    r, g, b = rgb
    if word is None:
        return True
    if word == "red":
        return r > 2.0 * max(g, b) + 20
    if word == "green":
        return g > 1.3 * max(r, b) + 15
    if word == "blue":
        return b > 1.3 * max(r, g) + 15
    if word == "yellow":
        return min(r, g) > 1.8 * b + 30 and abs(r - g) < 0.3 * max(r, g)
    if word in ("purple", "violet", "magenta"):
        return min(r, b) > 1.3 * g + 15
    if word == "orange":
        return r > 1.3 * g + 15 and g > 1.3 * b + 10
    return True


COLOR_WORDS = ("red", "green", "blue", "yellow", "purple", "violet", "magenta", "orange")


def mask_mean_rgb(cand, rgb):
    x1, y1, x2, y2 = (int(v) for v in cand.box_xyxy)
    msk = cand.mask
    acc = [0.0, 0.0, 0.0]
    npx = 0
    hgt = len(rgb)
    wid = len(rgb[0])
    for v in range(max(0, y1), min(hgt, y2 + 1)):
        row = msk[v]
        for u in range(max(0, x1), min(wid, x2 + 1)):
            if row[u]:
                px = rgb[v][u]
                acc[0] += float(px[0])
                acc[1] += float(px[1])
                acc[2] += float(px[2])
                npx += 1
    if npx == 0:
        return None
    return [a / npx for a in acc]


def locate_colored(queries, camera, color, min_points=8, near=None, near_tol=None):
    """Segment each query, keep candidates whose mask colour matches `color`, return their points."""
    tried = []
    rgb = get_observation().cameras[camera].rgb
    for query in queries:
        try:
            seg = segment_text(camera, query)
        except Exception as exc:
            tried.append({"q": query, "exc": str(exc)[:160]})
            continue
        if not seg.ok:
            tried.append({"q": query, "ok": False})
            continue
        for cand in seg.segmentations[:8]:
            info = {"q": query, "score": round(float(cand.score), 3)}
            try:
                mean = mask_mean_rgb(cand, rgb)
            except Exception as exc:
                mean = None
                info["rgb_exc"] = str(exc)[:120]
            info["rgb"] = [round(a) for a in mean] if mean is not None else None
            ok_color = mean is not None and color_match(color, mean)
            info["color_ok"] = ok_color
            if not ok_color:
                tried.append(info)
                continue
            try:
                pc = mask_to_point_cloud(cand, camera, target_frame=BASE)
                pts = [(float(p[0]), float(p[1]), float(p[2])) for p in pc.points]
            except Exception as exc:
                info["pc_exc"] = str(exc)[:120]
                tried.append(info)
                continue
            info["n"] = len(pts)
            if len(pts) < min_points:
                tried.append(info)
                continue
            sm = summarize(pts)
            info["xy"] = [round(sm["x"], 4), round(sm["y"], 4)]
            if near is not None and math.sqrt((sm["x"] - near[0]) ** 2 + (sm["y"] - near[1]) ** 2) > near_tol:
                info["far"] = True
                tried.append(info)
                continue
            tried.append(info)
            return pts, tried
    return None, tried


# ---------------- time / waiting helpers ----------------
def tick_now():
    return int(round(get_robot_state().timestamp_s * 20.0))


def hold(n, grip):
    """Hold the measured joints for n control ticks with an explicit gripper command."""
    st = get_robot_state()
    q = [float(v) for v in st.joint_positions["primary"]]
    for i in range(n):
        r = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=q, gripper_position=grip,
            embodiment=st.embodiment)}))
        if is_terminal(r):
            raise ValueError(END_MARK)


def hold_until(t_goal, grip, chunk=1000):
    now = tick_now()
    n = t_goal - now
    if n > 0:
        hold(min(n, chunk), grip)
    return tick_now()


# ---------------- moving-target model ----------------
def observe_target():
    """Front-camera target centre (x, y, z) or None; only flat, low masks are accepted."""
    for query in ("target", "purple and white target"):
        try:
            found = localize_object(query, camera_name="front", target_frame=BASE)
        except Exception:
            continue
        if not found.ok:
            continue
        pts = [p for p in points_of(found) if p[2] < 0.03]
        if len(pts) >= 12:
            return (quantile([p[0] for p in pts], 0.5), quantile([p[1] for p in pts], 0.5),
                    quantile([p[2] for p in pts], 0.5), len(pts))
    return None


def interp(samples, t):
    """Linear interpolation of (tick, x, y) samples (sorted by tick) at tick t."""
    if t <= samples[0][0]:
        return samples[0][1], samples[0][2]
    for i in range(1, len(samples)):
        if samples[i][0] >= t:
            a = samples[i - 1]
            b = samples[i]
            span = b[0] - a[0]
            f = (t - a[0]) / span if span > 0 else 0.0
            return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])
    return samples[-1][1], samples[-1][2]


def mismatch(samples, lag):
    t_last = samples[-1][0]
    total = 0.0
    count_used = 0
    for s in samples:
        if s[0] + lag > t_last:
            break
        px, py = interp(samples, s[0] + lag)
        total += math.sqrt((px - s[1]) ** 2 + (py - s[2]) ** 2)
        count_used += 1
    if count_used < 4:
        return None
    return total / count_used


def dist2(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def match_phase(path, period, obs, prior=None, window=None):
    """Phase (integer tick offset into path) at obs[0]'s tick that best explains obs."""
    best = None
    t_first = obs[0][0]
    for phi in range(period):
        if prior is not None and window is not None:
            d = (phi - prior) % period
            if d > window and period - d > window:
                continue
        cost = 0.0
        for o in obs:
            p = path[(phi + o[0] - t_first) % period]
            cost += (p[0] - o[1]) ** 2 + (p[1] - o[2]) ** 2
        if best is None or cost < best[0]:
            best = (cost, phi)
    if best is None:
        return None, None
    return best[1], math.sqrt(best[0] / len(obs))


# ======================= program =======================
context = get_task_context()
state0 = get_robot_state()
BASE = state0.base_frame
goal = context.language.lower()
report["goal"] = context.language
max_steps = None
try:
    meta = robomme_plus.get_task_metadata()
    max_steps = int(meta["max_steps"])
except Exception as exc:
    report["meta_error"] = str(exc)[:200]
report["max_steps"] = max_steps

count = parse_count(goal)
obj_phrase = parse_object(goal)
report["count"] = count
report["object"] = obj_phrase
if count is None:
    count = 1
    report["count_defaulted"] = True

home_pose = state0.end_effector_poses["primary"]
home_q = tuple(float(v) for v in home_pose.quaternion_wxyz)
z_axis = qrot(home_q, (0.0, 0.0, 1.0))
down_q = home_q if z_axis[2] < -0.95 else (0.0, 1.0, 0.0, 0.0)

CUBE_PROMPTS = (obj_phrase, obj_phrase.split()[-1] if obj_phrase else "cube")
CUBE_COLOR = None
for word in (obj_phrase or "").split():
    if word in COLOR_WORDS:
        CUBE_COLOR = word
report["cube_color"] = CUBE_COLOR
CUBE_SEG_PROMPTS = (obj_phrase, obj_phrase.split()[-1] if obj_phrase else "cube", "cube", "block")
TARGET_PROMPTS = ("target", "purple and white target", "bullseye", "round target")
BUTTON_PROMPTS = ("button", "white button", "round button", "push button")

LEARN_STRIDE = 4          # ticks between target observations while learning the motion
LEARN_MAX = 360           # tick cap for learning one period
STILL_TOL = 0.025         # mean-of-3 displacement (m) over >=24 ticks below which the target is static
OPEN_LEAD = 3             # ticks between open command and expected landing
WAIT_RELEASE_GAP = 0.015  # cube bottom above target top while waiting for the target
FAR_FOR_OBS = 0.07        # only trust target observations this far from the held cube

phase_log = []
report["phase_log"] = phase_log

try:
    # ---- front-camera localization (arm at home) ----
    cube_pts, tr = locate_colored(CUBE_SEG_PROMPTS, "front", CUBE_COLOR)
    report["front_cube_try"] = tr
    if cube_pts is None:
        raise ValueError("front cube localization failed")
    cube_f = summarize(cube_pts)
    report["front_cube"] = cube_f

    # button: prefer a low-saturation (white/grey) candidate standing above the table
    button_f = None
    seg = segment_text("front", "button")
    obs0 = get_observation()
    rgb = obs0.cameras["front"].rgb
    btn_cands = []
    if seg.ok:
        for cand in seg.segmentations[:5]:
            info = {"score": round(float(cand.score), 3)}
            try:
                pc = mask_to_point_cloud(cand, "front", target_frame=BASE)
                pts = [(float(p[0]), float(p[1]), float(p[2])) for p in pc.points]
            except Exception as exc:
                info["exc"] = str(exc)[:120]
                pts = []
            if len(pts) >= 8:
                sm = summarize(pts)
                info["xyz"] = [round(sm["x"], 4), round(sm["y"], 4), round(sm["zhi"], 4)]
                sat = None
                try:
                    x1, y1, x2, y2 = (int(v) for v in cand.box_xyxy)
                    msk = cand.mask
                    acc = [0.0, 0.0, 0.0]
                    npx = 0
                    for v in range(max(0, y1), min(256, y2 + 1)):
                        for u in range(max(0, x1), min(256, x2 + 1)):
                            if msk[v][u]:
                                px = rgb[v][u]
                                acc[0] += float(px[0])
                                acc[1] += float(px[1])
                                acc[2] += float(px[2])
                                npx += 1
                    if npx > 0:
                        mean_rgb = [a / npx for a in acc]
                        sat = max(mean_rgb) - min(mean_rgb)
                        info["rgb"] = [round(a) for a in mean_rgb]
                except Exception as exc:
                    info["rgb_exc"] = str(exc)[:120]
                info["sat"] = sat
                plausible = 0.015 < sm["zhi"] < 0.07 and (sat is None or sat < 60)
                info["plausible"] = plausible
                if plausible and button_f is None:
                    button_f = sm
            btn_cands.append(info)
    report["button_candidates"] = btn_cands
    if button_f is None:
        bp, tr = locate(BUTTON_PROMPTS, "front")
        report["button_fallback_try"] = tr
        if bp is not None:
            button_f = summarize(bp)
    report["front_button"] = button_f

    # ---- learn the target's motion from the front camera (arm at home, gripper open) ----
    t0 = tick_now()
    samples = []
    first = observe_target()
    for retry in range(3):
        if first is not None:
            break
        hold(2, 1.0)
        t0 = tick_now()
        first = observe_target()
    if first is None:
        raise ValueError("front target localization failed")
    tz = first[2]
    samples.append((t0, first[0], first[1]))
    moving = None
    far = 0.0
    returned_at = None
    while True:
        hold(LEARN_STRIDE, 1.0)
        t = tick_now()
        o = observe_target()
        if o is not None:
            if dist2((o[0], o[1]), (samples[-1][1], samples[-1][2])) < 0.012 * (t - samples[-1][0]) + 0.02:
                samples.append((t, o[0], o[1]))
        d0 = dist2((samples[-1][1], samples[-1][2]), (samples[0][1], samples[0][2]))
        far = max(far, d0)
        if moving is None and t - t0 >= 24 and len(samples) >= 6:
            ax = sum(sm[1] for sm in samples[:3]) / 3.0
            ay = sum(sm[2] for sm in samples[:3]) / 3.0
            bx = sum(sm[1] for sm in samples[-3:]) / 3.0
            by = sum(sm[2] for sm in samples[-3:]) / 3.0
            report["still_test"] = dist2((ax, ay), (bx, by))
            moving = report["still_test"] > STILL_TOL
            if not moving:
                break
        if moving and returned_at is None and t - t0 >= 40 and far > 0.05 and d0 < 0.02:
            returned_at = t
        if returned_at is not None and t - returned_at >= 32:
            break
        if t - t0 >= LEARN_MAX:
            break
    report["learn_samples"] = [[s[0], round(s[1], 4), round(s[2], 4)] for s in samples]
    report["moving"] = moving

    period = None
    path = None
    if moving:
        lo_lag = 30
        hi_lag = samples[-1][0] - samples[0][0] - 8
        best = None
        for lag in range(lo_lag, hi_lag + 1, 2):
            m = mismatch(samples, lag)
            if m is not None and (best is None or m < best[0]):
                best = (m, lag)
        if best is not None:
            for lag in range(best[1] - 2, best[1] + 3):
                m = mismatch(samples, lag)
                if m is not None and m < best[0]:
                    best = (m, lag)
        report["period_fit"] = best
        if best is None or best[0] > 0.02:
            raise ValueError("could not identify target period")
        period = best[1]
        if period >= LEARN_MAX - 40:
            raise ValueError("period not observed within the learning window")
        path = [interp(samples, samples[0][0] + k) for k in range(period)]
        # phase model: phase(t) = ref_phase + (t - ref_tick) while the target is free
        ref_tick = samples[0][0]
        ref_phase = 0
        tx, ty = None, None
    else:
        tx = quantile([s[1] for s in samples], 0.5)
        ty = quantile([s[2] for s in samples], 0.5)
    report["period"] = period

    # ---- refine cube from the wrist camera ----
    do_open("open0")
    hover_z = tz + HOVER_DZ
    go("hover_cube", (cube_f["x"], cube_f["y"], hover_z), down_q, tol=0.015)
    cx, cy = cube_f["x"], cube_f["y"]
    cube_top = cube_f["zhi"]
    yaw = None
    edge = None
    wc, tr = locate_colored(CUBE_SEG_PROMPTS, "wrist", CUBE_COLOR, min_points=30,
                            near=(cube_f["x"], cube_f["y"]), near_tol=0.05)
    report["wrist_cube_try"] = tr
    if wc is not None:
        top, zhi = top_face(wc)
        if len(top) >= 15:
            yaw, eu, ev = square_yaw(top)
            tops = summarize(top)
            report["wrist_cube_top"] = tops
            if math.sqrt((tops["x"] - cx) ** 2 + (tops["y"] - cy) ** 2) < 0.05:
                cx, cy = tops["x"], tops["y"]
                cube_top = tops["z"]
                edge = (eu + ev) / 2.0 / 0.97
    height = cube_top - tz
    if edge is None or edge < 0.015 or edge > 0.08:
        edge = height
    if edge < 0.015 or edge > 0.08:
        edge = 0.04 if edge > 0.08 else max(edge, 0.02)
        report["edge_clamped"] = True
    report["edge_used"] = edge

    x_axis = qrot(down_q, (1.0, 0.0, 0.0))
    phi_ee = math.atan2(x_axis[1], x_axis[0])
    delta = wrap_quarter((yaw if yaw is not None else phi_ee) - phi_ee)
    grasp_q = qmul(qyaw(delta), down_q)
    report["yaw_delta_deg"] = delta * 180.0 / math.pi

    grasp_z = cube_top - edge / 2.0
    place_z = tz + edge / 2.0 + RELEASE_GAP
    wait_z = tz + edge / 2.0 + WAIT_RELEASE_GAP
    regrasp_z = tz + edge / 2.0 + 0.002
    lift_z = place_z + LIFT_DZ
    report["z_plan"] = {"grasp": grasp_z, "place": place_z, "wait": wait_z, "regrasp": regrasp_z}

    # ---- first pick ----
    go("pre_grasp", (cx, cy, grasp_z + 0.08), grasp_q, tol=0.01)
    go("grasp", (cx, cy, grasp_z), grasp_q, tol=0.008)
    report["grip_closed_1"] = do_close("close1")
    go("lift1", (cx, cy, lift_z), grasp_q, tol=0.015)
    report["grip_after_lift_1"] = gripper_reading()

    places = 0
    picks = 1
    qx, qy, q_phase = None, None, None
    misses = 0
    stop_phase = None
    cycle_ticks = []
    while places < count:
        k = picks
        if moving:
            # observations of the free-moving target since its last resume refine the phase
            obs = []
            now = tick_now()
            ee = get_robot_state().end_effector_poses["primary"].position
            o = observe_target()
            if o is not None and (qx is None or dist2((o[0], o[1]), (qx, qy)) > FAR_FOR_OBS):
                obs.append((now, o[0], o[1]))
            pred = (ref_phase + now - ref_tick) % period
            if obs:
                ph, res = match_phase(path, period, obs, prior=pred, window=30)
                if ph is not None and res < 0.025:
                    ref_tick, ref_phase = now, ph
                phase_log.append({"k": k, "t": now, "pred": pred, "meas": ph, "res": res})
            else:
                phase_log.append({"k": k, "t": now, "pred": pred, "meas": None})
            now = tick_now()
            cur = (ref_phase + now - ref_tick) % period
            if qx is None:
                # choose the intercept point: earliest path point we can reach in time
                for d in range(25, 25 + period):
                    p = path[(cur + d) % period]
                    need = 20 + int(dist2((float(ee[0]), float(ee[1])), p) / 0.010) + 20
                    if d >= need:
                        q_phase = (cur + d) % period
                        qx, qy = p[0], p[1]
                        break
                if qx is None:
                    q_phase = (cur + period // 2) % period
                    qx, qy = path[q_phase][0], path[q_phase][1]
                report["intercept"] = [qx, qy, q_phase]
            go("over_q%d" % k, (qx, qy, lift_z), grasp_q, tol=0.015)
            go("wait_low%d" % k, (qx, qy, wait_z), grasp_q, tol=0.008)
            # wait for the target, refining its phase whenever it is predicted far from Q
            last_obs = -100
            while True:
                now = tick_now()
                cur = (ref_phase + now - ref_tick) % period
                d = (q_phase - cur) % period
                if d < OPEN_LEAD + 1:
                    d += period
                if d - OPEN_LEAD <= 18:
                    break
                if dist2(path[cur], (qx, qy)) > FAR_FOR_OBS + 0.01 and now - last_obs >= 6:
                    last_obs = now
                    o = observe_target()
                    if o is not None and dist2((o[0], o[1]), (qx, qy)) > FAR_FOR_OBS:
                        ph, res = match_phase(path, period, [(now, o[0], o[1])], prior=cur, window=25)
                        phase_log.append({"k": k, "t": now, "pred": cur, "meas": ph, "res": res})
                        if ph is not None and res < 0.025:
                            ref_tick, ref_phase = now, ph
                            continue
                hold(min(8, d - OPEN_LEAD - 18), 0.0)
            now = tick_now()
            cur = (ref_phase + now - ref_tick) % period
            d = (q_phase - cur) % period
            if d < OPEN_LEAD + 1:
                d += period
            t_open = now + d - OPEN_LEAD
            hold_until(t_open, 0.0)
            report["t_open_%d" % k] = [t_open, tick_now()]
            do_open("release%d" % k)
            stop_phase = (ref_phase + t_open + OPEN_LEAD - ref_tick) % period
            go("retreat%d" % k, (qx, qy, place_z + RETREAT_DZ), grasp_q, tol=0.015)
            # verify the target stopped under the cube; a moving target means a missed place
            o = observe_target()
            t_chk = tick_now()
            if o is not None:
                miss_d = dist2((o[0], o[1]), (qx, qy))
                phase_log.append({"k": k, "t": t_chk, "check_dist": miss_d})
                if miss_d > 0.045 and misses < 2:
                    misses += 1
                    report["miss_%d" % misses] = {"k": k, "dist": miss_d}
                    ph, res = match_phase(path, period, [(t_chk, o[0], o[1])])
                    if ph is not None and res < 0.025:
                        ref_tick, ref_phase = t_chk, ph
                    go("regrasp_miss%d" % misses, (qx, qy, regrasp_z), grasp_q, tol=0.008)
                    do_close("close_miss%d" % misses)
                    go("lift_miss%d" % misses, (qx, qy, lift_z), grasp_q, tol=0.015)
                    continue
        else:
            go("over_target%d" % k, (tx, ty, lift_z), grasp_q, tol=0.015)
            go("place%d" % k, (tx, ty, place_z), grasp_q, tol=0.008)
            do_open("release%d" % k)
            go("retreat%d" % k, (tx, ty, place_z + RETREAT_DZ), grasp_q, tol=0.015)
            qx, qy = tx, ty
        places += 1
        cycle_ticks.append(tick_now())
        if places >= count:
            break
        # ---- re-pick from the target ----
        k = picks + 1
        go("regrasp%d" % k, (qx, qy, regrasp_z), grasp_q, tol=0.008)
        report["grip_closed_%d" % k] = do_close("close%d" % k)
        t_lift = tick_now()
        go("lift%d" % k, (qx, qy, lift_z), grasp_q, tol=0.015)
        report["grip_after_lift_%d" % k] = gripper_reading()
        picks = k
        if moving and stop_phase is not None:
            # the target paused under the cube; it resumes about when the cube leaves it
            ref_tick, ref_phase = t_lift, stop_phase
    report["picks"] = picks
    report["places"] = places
    report["cycle_ticks"] = cycle_ticks

    # ---- press the button ----
    if button_f is None:
        raise ValueError("button not localized")
    bx, by, btop = button_f["x"], button_f["y"], button_f["zhi"]
    report["button_used"] = [bx, by, btop]
    go("hover_button", (bx, by, tz + HOVER_DZ), grasp_q, tol=0.015)
    do_close("close_for_press")
    go("pre_press", (bx, by, btop + 0.04), grasp_q, tol=0.01)
    pressed = False
    for depth in (0.01, 0.02, 0.03):
        res = move_to_pose(Pose((bx, by, btop - depth), grasp_q, BASE), tolerance=0.005, max_steps=120)
        entry = record("press_%d" % int(depth * 1000), res)
        if entry["terminal"]:
            pressed = True
            break
    report["press_terminal"] = pressed
    if not pressed:
        go("press_release", (bx, by, btop + 0.04), grasp_q, tol=0.01)
except Exception as exc:
    if str(exc) == END_MARK:
        report["ended"] = True
    else:
        report["exception"] = str(exc)[:400]

try:
    report["t_end"] = tick_now()
except Exception:
    report["t_end"] = None
result = report
