# v04 — BinFill. Over v03 (urgency target choice, no home trip, result survives terminal):
#  neighbour-collision fix on the crowded belt (v03 seed 1 cycle 3: the descent landed on an
#  adjacent cube, 124 ticks blocked, the target passed, a needed cube later left the belt).
#  * all cubes on the belt (any colour) are observed as obstacles each cycle;
#  * the jaw yaw is chosen among perpendicular-to-motion +-20/40 deg to minimise predicted
#    finger/neighbour overlaps at the intercept (relative belt configuration is fixed, so only
#    the yaw can be changed, not the timing);
#  * the arm hovers ABOVE the cubes at the intercept and descends just before the predicted
#    arrival, so fingers sit in the lanes for ~15 ticks instead of up to 60;
#  * leads adapt to the measured pre-grasp ticks (home and bin starts separately).
#  Static scenes (speed < MOVING) keep the v00 behaviour.
import math

CAMS = ["wrist", "front"]
NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
           "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a": 1, "an": 1}
COLOURS = ["red", "blue", "green", "yellow", "orange", "purple", "white", "black"]
WAIT_TICKS = 10          # ticks held per empty observation
VEL_TICKS = 15           # ticks between the two observations used for velocity
LEAD_TICKS = 50          # intercept lead: >= pre-grasp (~25) + descent (~12) ticks measured in v00/v01
MOVING = 0.0006           # m/tick; below this a cube is treated as static (belt in v01 ~0.002)
BUTTON_RESERVE = 120     # ticks kept for the button press
OBST_PROMPTS = ["cube", "red cube", "blue cube", "green cube", "yellow cube", "orange cube",
                "purple cube", "cyan cube", "pink cube"]
FINGER_OFF = 0.045       # finger centre offset from the jaw centre when open (m)
PERP_CLEAR = 0.04        # neighbour half-size (rotated) + finger half-thickness
ALONG_LO = -0.06         # along-motion window (relative to finger) swept while fingers are low
ALONG_HI = 0.055
YAW_OFFSETS = [0.0, 20.0, -20.0, 40.0, -40.0]

base_frame = get_robot_state().base_frame
home_state = get_robot_state()
q_down = [float(v) for v in home_state.end_effector_poses["primary"].quaternion_wxyz]
language = get_task_context().language
report = {"language": language, "events": [], "cycles": [], "waits": []}
ended = {"flag": False}


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    return ordered[int((len(ordered) - 1) * fraction)]


def meta_int(key, default):
    try:
        return int(get_task_context().metadata[key])
    except Exception:
        return default


def steps_used():
    return meta_int("episode_steps_used", -1)


MAX_STEPS = meta_int("episode_max_steps", meta_int("max_steps", 1000))


def note(kind, res):
    entry = {"kind": kind}
    try:
        entry["ok"] = bool(res.ok)
        entry["steps"] = int(res.steps_executed)
        if res.terminated or res.truncated:
            ended["flag"] = True
            entry["terminal"] = [bool(res.terminated), bool(res.truncated)]
    except Exception as exc:
        entry["err"] = str(exc)
    entry["t"] = steps_used()
    report["events"].append(entry)
    return res


def gripper():
    try:
        return float(get_robot_state().gripper_positions["primary"])
    except Exception:
        return -1.0


def hold(n):
    st = get_robot_state()
    done = 0
    for k in range(n):
        r = step(RobotAction(arms={"primary": ArmCommand(
            "joint_position", st.joint_positions["primary"], gripper_position=1.0,
            embodiment=st.embodiment)}))
        done += 1
        try:
            if r.terminated or r.truncated:
                ended["flag"] = True
                break
        except Exception:
            pass
    return done


def parse_goal(text):
    words = text.lower().replace(",", " ").replace(".", " ").split()
    goal = []
    for i in range(1, len(words)):
        if words[i] in COLOURS and i + 1 < len(words) and words[i + 1].startswith("cube"):
            prev = words[i - 1]
            count = 1
            if prev in NUMBERS:
                count = NUMBERS[prev]
            else:
                try:
                    count = int(prev)
                except Exception:
                    count = 1
            goal.append([words[i], count])
    return goal


def colour_stats(pc):
    try:
        cols = pc.colors
        if cols is None or len(cols) == 0:
            return None, 0.0
        sr = 0.0
        sg = 0.0
        sb = 0.0
        dark = 0
        for col in cols:
            r = float(col[0])
            g = float(col[1])
            b = float(col[2])
            sr += r
            sg += g
            sb += b
            if max(r, g, b) < 30.0:
                dark += 1
        m = float(len(cols))
        return [sr / m, sg / m, sb / m], dark / m
    except Exception:
        return None, 0.0


def colour_matches(colour, rgb):
    if rgb is None:
        return True
    r, g, b = rgb
    eps = 1.0
    if colour == "red":
        return r > 2.0 * (max(g, b) + eps)
    if colour == "green":
        return g > 2.0 * (max(r, b) + eps)
    if colour == "blue":
        return b > 2.0 * (max(r, g) + eps)
    if colour == "yellow":
        return r > 2.0 * (b + eps) and g > 2.0 * (b + eps)
    return True


def describe(cand, cam):
    pc = mask_to_point_cloud(cand, cam, target_frame=base_frame)
    pts = pc.points
    if len(pts) < 12:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    zs = [float(p[2]) for p in pts]
    x0 = quantile(xs, 0.1)
    x1 = quantile(xs, 0.9)
    y0 = quantile(ys, 0.1)
    y1 = quantile(ys, 0.9)
    rgb, dark = colour_stats(pc)
    return {
        "cam": cam, "score": float(cand.score),
        "cx": 0.5 * (x0 + x1), "cy": 0.5 * (y0 + y1),
        "ex": x1 - x0, "ey": y1 - y0,
        "zlo": quantile(zs, 0.1), "ztop": quantile(zs, 0.9),
        "npts": len(pts), "rgb": rgb, "dark": dark,
    }


def brief(d):
    out = {}
    for k in d:
        v = d[k]
        try:
            out[k] = round(float(v), 3)
        except Exception:
            if k == "rgb" and v is not None:
                out[k] = [round(float(c), 1) for c in v]
            else:
                out[k] = v
    return out


def candidates(prompt, cams=None, limit=5):
    out = []
    use = cams if cams is not None else CAMS
    for cam in use:
        try:
            seg = segment_text(cam, prompt)
        except Exception:
            continue
        if not seg.ok:
            continue
        for cand in seg.segmentations[:limit]:
            try:
                d = describe(cand, cam)
            except Exception:
                d = None
            if d is not None:
                d["prompt"] = prompt
                out.append(d)
    return out


def far_from_base(d):
    return d["cx"] * d["cx"] + d["cy"] * d["cy"] > 0.2 * 0.2


def find_bin(log):
    best = None
    best_s = None
    for prompt in ("bin", "box", "container"):
        for d in candidates(prompt):
            valid = (0.05 <= d["ex"] <= 0.25 and 0.05 <= d["ey"] <= 0.25 and
                     0.03 <= d["ztop"] <= 0.2 and far_from_base(d))
            s = d["score"]
            if d["dark"] >= 0.15:
                s += 0.5
            if d["ztop"] >= 0.042:
                s += 0.2
            d["bin_s"] = s
            d["valid"] = valid
            log.append(brief(d))
            if valid and (best_s is None or s > best_s):
                best = d
                best_s = s
    return best


def in_bin(d, bin_d, margin=0.015):
    if bin_d is None:
        return False
    return (abs(d["cx"] - bin_d["cx"]) < 0.5 * bin_d["ex"] + margin and
            abs(d["cy"] - bin_d["cy"]) < 0.5 * bin_d["ey"] + margin)


def greyish(rgb):
    if rgb is None:
        return True
    return (max(rgb) - min(rgb)) <= 0.3 * max(rgb) + 5.0


def find_button(bin_d, log):
    valid_list = []
    for d in candidates("button"):
        valid = far_from_base(d) and d["dark"] < 0.1 and greyish(d["rgb"])
        valid = valid and 0.005 <= d["ztop"] <= 0.07 and d["ex"] <= 0.15 and d["ey"] <= 0.15
        if bin_d is not None:
            dx = d["cx"] - bin_d["cx"]
            dy = d["cy"] - bin_d["cy"]
            if dx * dx + dy * dy < 0.07 * 0.07:
                valid = False
        d["valid"] = valid
        log.append(brief(d))
        if valid:
            valid_list.append(d)
    best = None
    for d in valid_list:
        if best is None or d["score"] > best["score"]:
            best = d
    if best is None:
        return None
    top = best["ztop"]
    for d in valid_list:
        dx = d["cx"] - best["cx"]
        dy = d["cy"] - best["cy"]
        if dx * dx + dy * dy < 0.03 * 0.03 and d["ztop"] > top:
            top = d["ztop"]
    best["press_top"] = top
    return best


def cube_list(colours, bin_d, button_d, seen, cams):
    out = []
    for colour in colours:
        for d in candidates(colour + " cube", cams=cams):
            ok = (0.015 <= d["ex"] <= 0.09 and 0.015 <= d["ey"] <= 0.09 and
                  0.015 <= d["ztop"] <= 0.12 and far_from_base(d))
            if in_bin(d, bin_d):
                ok = False
            if button_d is not None:
                dx = d["cx"] - button_d["cx"]
                dy = d["cy"] - button_d["cy"]
                if dx * dx + dy * dy < 0.04 * 0.04:
                    ok = False
            if not colour_matches(colour, d["rgb"]):
                ok = False
            d["colour"] = colour
            if seen is not None:
                seen.append([colour, d["cam"], round(d["score"], 3), round(d["cx"], 3),
                             round(d["cy"], 3), round(d["ztop"], 3), ok])
            if not ok:
                continue
            dup = False
            for e in out:
                if e["cam"] == d["cam"] and e["colour"] == colour:
                    ddx = e["cx"] - d["cx"]
                    ddy = e["cy"] - d["cy"]
                    if ddx * ddx + ddy * ddy < 0.015 * 0.015:
                        dup = True
            if not dup:
                out.append(d)
    return out


def median(values):
    if len(values) == 0:
        return 0.0
    return quantile(values, 0.5)


def estimate_velocity(first, second, ticks):
    """Median displacement per tick of same-colour, same-camera nearest matches."""
    dxs = []
    dys = []
    for d in second:
        best = None
        best_r = None
        for e in first:
            if e["cam"] != d["cam"] or e["colour"] != d["colour"]:
                continue
            ddx = d["cx"] - e["cx"]
            ddy = d["cy"] - e["cy"]
            r = ddx * ddx + ddy * ddy
            if best_r is None or r < best_r:
                best_r = r
                best = [ddx, ddy]
        if best is not None and best_r < 0.06 * 0.06:
            dxs.append(best[0])
            dys.append(best[1])
    if len(dxs) == 0:
        return None, 0
    return [median(dxs) / ticks, median(dys) / ticks], len(dxs)


def find_belt(log):
    best = None
    for prompt in ("conveyor belt", "grey mat"):
        for d in candidates(prompt):
            ok = (0.15 <= d["ex"] <= 0.8 and 0.15 <= d["ey"] <= 0.8 and 0.005 <= d["ztop"] <= 0.05)
            log.append([prompt, d["cam"], round(d["score"], 3), round(d["cx"], 3), round(d["cy"], 3),
                        round(d["ex"], 3), round(d["ey"], 3), round(d["ztop"], 3), ok])
            if not ok:
                continue
            if best is None or d["ex"] * d["ey"] > best["ex"] * best["ey"]:
                best = d
    return best


def inside_belt_ahead(px, py, vel, belt_d, margin):
    """True if (px, py) is not beyond the downstream end of the belt (minus margin)."""
    if belt_d is None or vel is None:
        return True
    speed = math.sqrt(vel[0] * vel[0] + vel[1] * vel[1])
    if speed < MOVING:
        return True
    ux = vel[0] / speed
    uy = vel[1] / speed
    along = (px - belt_d["cx"]) * ux + (py - belt_d["cy"]) * uy
    half = 0.5 * (abs(ux) * belt_d["ex"] + abs(uy) * belt_d["ey"])
    return along < half - margin


def qmul(a, b):
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]


def yawed(theta):
    """Home top-down orientation rotated by theta about the base z axis."""
    qz = [math.cos(0.5 * theta), 0.0, 0.0, math.sin(0.5 * theta)]
    return qmul(qz, q_down)


def jaw_yaw(vel):
    """Yaw so the jaws close perpendicular to vel. At home yaw the jaws close along base y
    (Panda fingers move along the hand y axis; home quaternion maps it to -y)."""
    if vel is None:
        return 0.0
    speed = math.sqrt(vel[0] * vel[0] + vel[1] * vel[1])
    if speed < MOVING:
        return 0.0
    # jaw axis at yaw t is (sin t, -cos t); we want it parallel to the motion normal (-vy, vx)
    theta = math.atan2(vel[0], -vel[1]) + 0.5 * math.pi
    while theta > 0.5 * math.pi:
        theta -= math.pi
    while theta < -0.5 * math.pi:
        theta += math.pi
    return theta


def hold_current(n, grip):
    st = get_robot_state()
    done = 0
    for k in range(n):
        r = step(RobotAction(arms={"primary": ArmCommand(
            "joint_position", st.joint_positions["primary"], gripper_position=grip,
            embodiment=st.embodiment)}))
        done += 1
        try:
            if r.terminated or r.truncated:
                ended["flag"] = True
                break
        except Exception:
            pass
    return done


def pose(x, y, z, q=None):
    return Pose((x, y, z), q if q is not None else q_down, base_frame)


def obstacles(cams):
    out = []
    for prompt in OBST_PROMPTS:
        for d in candidates(prompt, cams=cams):
            if not (0.015 <= d["ex"] <= 0.09 and 0.015 <= d["ey"] <= 0.09 and
                    0.03 <= d["ztop"] <= 0.09 and far_from_base(d)):
                continue
            if in_bin(d, bin_d):
                continue
            dup = False
            for e in out:
                ddx = e[0] - d["cx"]
                ddy = e[1] - d["cy"]
                if ddx * ddx + ddy * ddy < 0.015 * 0.015:
                    dup = True
            if not dup:
                out.append([d["cx"], d["cy"]])
    return out


def wrap_half(theta):
    while theta > 0.5 * math.pi:
        theta -= math.pi
    while theta < -0.5 * math.pi:
        theta += math.pi
    return theta


def collisions(px, py, theta, vel, lead, obst):
    speed = math.sqrt(vel[0] * vel[0] + vel[1] * vel[1])
    ux = vel[0] / speed
    uy = vel[1] / speed
    nx = -uy
    ny = ux
    ax = math.sin(theta)
    ay = -math.cos(theta)
    hits = 0
    for o in obst:
        ox = o[0] + vel[0] * lead
        oy = o[1] + vel[1] * lead
        if (ox - px) * (ox - px) + (oy - py) * (oy - py) < 0.025 * 0.025:
            continue
        for sgn in (1.0, -1.0):
            fx = px + sgn * FINGER_OFF * ax
            fy = py + sgn * FINGER_OFF * ay
            al = (ox - fx) * ux + (oy - fy) * uy
            pe = (ox - fx) * nx + (oy - fy) * ny
            if abs(pe) < PERP_CLEAR and ALONG_LO < al < ALONG_HI:
                hits += 1
    return hits


def best_yaw(px, py, vel, lead, obst):
    base = jaw_yaw(vel)
    best = None
    for off in YAW_OFFSETS:
        th = wrap_half(base + off * math.pi / 180.0)
        n = collisions(px, py, th, vel, lead, obst)
        if best is None or n < best[1]:
            best = [th, n, off]
    return best


goal = parse_goal(language)
report["goal"] = goal
remaining = {}
order = []
for g in goal:
    remaining[g[0]] = remaining.get(g[0], 0) + g[1]
    if g[0] not in order:
        order.append(g[0])
state = {"vel": None, "lead": LEAD_TICKS, "lead_bin": LEAD_TICKS + 20, "at_home": True,
         "misses": 0, "waits": 0, "s_max": None, "approach": [], "approach_bin": [],
         "desc": [10], "obst": []}


def along(d, vel):
    speed = math.sqrt(vel[0] * vel[0] + vel[1] * vel[1])
    return (d["cx"] * vel[0] + d["cy"] * vel[1]) / speed


def choose_target(cands, vel, belt_d):
    """Static: highest score. Moving: most downstream cube whose intercept is still on the belt."""
    moving = vel is not None and math.sqrt(vel[0] * vel[0] + vel[1] * vel[1]) >= MOVING
    lead = 0
    if moving:
        lead = state["lead"] if state["at_home"] else state["lead_bin"]
    best = None
    best_key = None
    for d in cands:
        px = d["cx"]
        py = d["cy"]
        if moving:
            px = px + vel[0] * lead
            py = py + vel[1] * lead
            s_here = along(d, vel)
            if state["s_max"] is None or s_here > state["s_max"]:
                state["s_max"] = s_here
            if belt_d is not None:
                if not inside_belt_ahead(px, py, vel, belt_d, 0.05):
                    continue
            key = s_here
        else:
            key = d["score"]
        if best_key is None or key > best_key:
            best = d
            best_key = key
            best["px"] = px
            best["py"] = py
    return best, lead


def run_cycle(cube, lead, t_obs, vel):
    colour = cube["colour"]
    theta = 0.0
    hits = None
    if lead > 0:
        yb = best_yaw(cube["px"], cube["py"], vel, lead, state["obst"])
        theta = yb[0]
        hits = [yb[1], yb[2], collisions(cube["px"], cube["py"], jaw_yaw(vel), vel, lead, state["obst"])]
    qg = yawed(theta)
    cyc = {"colour": colour, "t_obs": t_obs, "lead": lead, "from_home": state["at_home"],
           "yaw_deg": round(theta * 180.0 / math.pi, 1), "hits_best_off_perp": hits,
           "n_obst": len(state["obst"]),
           "cube": [cube["cam"], round(cube["cx"], 3), round(cube["cy"], 3), round(cube["ztop"], 3)],
           "intercept": [round(cube["px"], 3), round(cube["py"], 3)]}
    report["cycles"].append(cyc)
    half = 0.5 * min(cube["ex"], cube["ey"])
    depth = min(max(half, 0.012), 0.03)
    grasp_z = max(cube["ztop"] - depth, 0.01)
    cyc["grasp_z"] = round(grasp_z, 3)
    note("open", open_gripper())
    if ended["flag"]:
        return
    note("pre", move_to_pose(pose(cube["px"], cube["py"], cube["ztop"] + 0.10, qg),
                             tolerance=0.02, max_steps=250))
    if ended["flag"]:
        return
    state["at_home"] = False
    now = steps_used()
    desc_t = max(state["desc"])
    if lead > 0 and now >= 0 and t_obs >= 0:
        if cyc["from_home"]:
            state["approach"].append(now - t_obs)
        else:
            state["approach_bin"].append(now - t_obs)
        hover = (t_obs + lead - desc_t - 1) - now
        cyc["hover_wait"] = hover
        if hover > 0:
            hold_current(min(hover, 60), 1.0)
            if ended["flag"]:
                return
    t_d = steps_used()
    gx = cube["px"]
    gy = cube["py"]
    if lead > 0 and t_d >= 0 and t_obs >= 0:
        late = (t_d + desc_t) - (t_obs + lead)
        if late > 0:
            # behind schedule: aim the descent where the cube will be when it ends
            late = min(late, 40)
            gx = gx + vel[0] * late
            gy = gy + vel[1] * late
            cyc["late"] = late
    r_desc = note("descend", move_to_pose(pose(gx, gy, grasp_z, qg),
                                          tolerance=0.01, max_steps=60))
    if ended["flag"]:
        return
    now = steps_used()
    try:
        if r_desc.ok and now > t_d >= 0:
            state["desc"].append(now - t_d)
    except Exception:
        pass
    if lead > 0 and now >= 0 and t_obs >= 0:
        wait = (t_obs + lead) - now
        cyc["arrive_wait"] = wait
        if wait > 1:
            hold_current(min(wait - 1, 30), 1.0)
            if ended["flag"]:
                return
    note("close", close_gripper())
    if ended["flag"]:
        return
    cyc["g_close"] = round(gripper(), 3)
    note("lift", move_to_pose(pose(gx, gy, transit_z, qg),
                              tolerance=0.02, max_steps=200))
    if ended["flag"]:
        return
    g_lift = gripper()
    cyc["g_lift"] = round(g_lift, 3)
    if g_lift < 0.08:
        cyc["result"] = "grasp miss"
        state["misses"] += 1
        note("open_miss", open_gripper())
        if ended["flag"]:
            return
        note("home_miss", go_home())
        state["at_home"] = True
        return
    note("over_bin", move_to_pose(pose(bin_d["cx"], bin_d["cy"], release_z, qg),
                                  tolerance=0.02, max_steps=250))
    if ended["flag"]:
        return
    note("release", open_gripper())
    remaining[colour] = remaining[colour] - 1
    cyc["result"] = "deposited"
    cyc["t_end"] = steps_used()
    if ended["flag"]:
        return
    moving = lead > 0
    if not moving:
        note("home", go_home())
        state["at_home"] = True


def adapt_lead():
    desc_t = max(state["desc"])
    if len(state["approach"]) > 0:
        state["lead"] = max(40, max(state["approach"]) + desc_t + 6)
    if len(state["approach_bin"]) > 0:
        state["lead_bin"] = max(50, max(state["approach_bin"]) + desc_t + 6)


def main_loop():
    while not ended["flag"]:
        needed = []
        for c in order:
            if remaining[c] > 0:
                needed.append(c)
        if len(needed) == 0:
            return
        used = steps_used()
        if used >= 0 and MAX_STEPS - used < BUTTON_RESERVE:
            report["stopped"] = "tick reserve reached"
            return
        if state["waits"] > 160:
            report["stopped"] = "wait cap reached"
            return
        if state["misses"] >= 5:
            report["stopped"] = "too many misses"
            return
        vel = state["vel"]
        if not state["at_home"]:
            # Over the bin after a release on a moving belt: try the front camera first.
            seen = []
            cands = cube_list(needed, bin_d, button_d, seen, ["front"])
            t_obs = steps_used()
            if vel is not None and math.sqrt(vel[0] * vel[0] + vel[1] * vel[1]) >= MOVING:
                state["obst"] = obstacles(["front"])
            cube, lead = choose_target(cands, vel, belt_d)
            if cube is None:
                note("home", go_home())
                state["at_home"] = True
                continue
            run_cycle(cube, lead, t_obs, vel)
            adapt_lead()
            continue
        first = cube_list(needed, bin_d, button_d, None, None)
        t_first = steps_used()
        if len(first) == 0:
            state["waits"] += 1
            hold(WAIT_TICKS)
            continue
        if vel is None:
            hold(VEL_TICKS)
            if ended["flag"]:
                return
            t2 = steps_used()
            second = cube_list(needed, bin_d, button_d, None, None)
            dt = t2 - t_first if (t_first >= 0 and t2 > t_first) else VEL_TICKS
            est, nmatch = estimate_velocity(first, second, dt)
            report["velocity"] = {"est": est, "nmatch": nmatch, "dt": dt}
            if est is None:
                state["waits"] += 1
                continue
            sp = math.sqrt(est[0] * est[0] + est[1] * est[1])
            if sp < MOVING:
                est = [0.0, 0.0]
            state["vel"] = est
            vel = est
            cands = second
            t_obs = t2
        else:
            cands = first
            t_obs = t_first
        if math.sqrt(vel[0] * vel[0] + vel[1] * vel[1]) >= MOVING:
            state["obst"] = obstacles(None)
        cube, lead = choose_target(cands, vel, belt_d)
        if cube is None:
            state["waits"] += 1
            hold(WAIT_TICKS)
            continue
        run_cycle(cube, lead, t_obs, vel)
        adapt_lead()


try:
    bin_d = None
    bin_tries = 0
    while bin_d is None and bin_tries < 4 and not ended["flag"]:
        log = []
        bin_d = find_bin(log)
        report["bin_log_" + str(bin_tries)] = log
        bin_tries += 1
        if bin_d is None:
            hold(WAIT_TICKS)
    report["bin"] = brief(bin_d) if bin_d is not None else None
    if bin_d is None:
        raise ValueError("bin not found")
    blog = []
    button_d = find_button(bin_d, blog)
    report["button_log"] = blog
    report["button"] = brief(button_d) if button_d is not None else None
    belt_log = []
    belt_d = find_belt(belt_log)
    report["belt_log"] = belt_log
    report["belt"] = brief(belt_d) if belt_d is not None else None
    rim_top = bin_d["ztop"]
    release_z = rim_top + 0.10
    transit_z = rim_top + 0.14
    main_loop()
    report["remaining"] = dict(remaining)
    report["lead_final"] = state["lead"]
    report["approach_ticks"] = state["approach"]
    report["approach_bin_ticks"] = state["approach_bin"]
    report["lead_bin_final"] = state["lead_bin"]
    report["desc_ticks"] = state["desc"]
    # Press the button.
    if not ended["flag"]:
        if not state["at_home"]:
            note("home_before_button", go_home())
            state["at_home"] = True
    if not ended["flag"]:
        blog2 = []
        b2 = find_button(bin_d, blog2)
        if b2 is not None:
            button_d = b2
        report["button_final"] = brief(button_d) if button_d is not None else None
        if button_d is not None:
            note("close_for_press", close_gripper())
            bx = button_d["cx"]
            by = button_d["cy"]
            top = button_d["press_top"]
            if not ended["flag"]:
                note("above_button", move_to_pose(pose(bx, by, top + 0.08), tolerance=0.02, max_steps=250))
            if not ended["flag"]:
                note("press", move_to_pose(pose(bx, by, top - 0.012), tolerance=0.01, max_steps=120))
            if not ended["flag"]:
                note("raise", move_to_pose(pose(bx, by, top + 0.06), tolerance=0.02, max_steps=120))
        else:
            report["button_log_final"] = blog2
except Exception as exc:
    report["exception"] = str(exc)
    report["remaining_at_exception"] = dict(remaining)

report["steps_used_final"] = steps_used()
result = report
