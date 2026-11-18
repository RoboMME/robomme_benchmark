# v04_wait_still (from v03 <- v02 <- v01 <- v00) -- ButtonUnmaskSwap (robomme_plus task_2)
# Strategy (from offline analysis of released recordings; all quantities derived at runtime):
#  1. At tick 0 (before any action) the cubes are uncovered: detect coloured cubes in front+wrist
#     RGB-D, project to robot_base, and create one track per cube.
#  2. Containers drop over the cubes and then swap continuously. Every control tick is stepped
#     manually; white container-top points from BOTH cameras are projected to robot_base and
#     associated to the nearest predicted track (Voronoi + gate), so identities follow the cubes.
#  3. Press both buttons (detected at tick 0 as the two largest low white blobs) before the
#     published deadline, then for each goal colour (in goal order) servo onto its container's
#     live track, grasp, lift, put back, release.
import math

COLOR_WORDS = ["red", "green", "blue", "yellow", "purple", "orange", "black", "white", "pink", "cyan"]
LOG = {"events": [], "hist": []}
S = {"tick": 0, "terminal": False, "conv": None, "table_z": None, "hc": None,
     "containers_seen": False, "appear_tick": None, "hc_samples": [], "static": None, "anon": 0}


def ev(msg, data=None):
    if len(LOG["events"]) < 400:
        LOG["events"].append([S["tick"], msg, data])


def classify(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 60:
        return "black"
    if mn > 170 and mx - mn < 35:
        return "white"
    if r > 150 and g < 75 and b < 75 and r - g > 100:
        return "red"
    if g > 120 and r < 110 and b < 110 and g - max(r, b) > 50:
        return "green"
    if b > 120 and r < 90 and g < 110 and b - max(r, g) > 50:
        return "blue"
    if r > 170 and g > 150 and b < 90 and abs(r - g) < 60:
        return "yellow"
    if r > 100 and b > 120 and g < 100 and min(r, b) - g > 40:
        return "purple"
    if r > 200 and 90 < g < 190 and b < 50:
        return "orange"
    return None


def is_white(r, g, b):
    mn = min(r, g, b)
    return mn > 165 and max(r, g, b) - mn < 40


def is_wood(r, g, b):
    return r > g > b and r - b > 55 and r > 90


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def qrot(q):
    w, x, y, z = q
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))


def cam_frame(cam):
    K = cam.intrinsics
    fx = float(K[0][0]); fy = float(K[1][1]); cx = float(K[0][2]); cy = float(K[1][2])
    pose = cam.camera_pose
    t = [float(v) for v in pose.position]
    q = [float(v) for v in pose.quaternion_wxyz]
    return fx, fy, cx, cy, qrot(q), t


def to_cam(conv, xc, yc, zc):
    if conv == "cv":
        return xc, yc, zc
    if conv == "gl":
        return xc, -yc, -zc
    return zc, -xc, -yc  # "ros": x forward, y left, z up


def project(cf, conv, u, v, d):
    fx, fy, cx, cy, R, t = cf
    xc = (u - cx) / fx * d
    yc = (v - cy) / fy * d
    a, b, c = to_cam(conv, xc, yc, d)
    return (R[0][0] * a + R[0][1] * b + R[0][2] * c + t[0],
            R[1][0] * a + R[1][1] * b + R[1][2] * c + t[1],
            R[2][0] * a + R[2][1] * b + R[2][2] * c + t[2])


def arrays(cam):
    try:
        rgb = cam.rgb.tolist()
    except Exception:
        rgb = cam.rgb
    try:
        dep = cam.depth_m.tolist()
    except Exception:
        dep = cam.depth_m
    return rgb, dep


def median(vals):
    s = sorted(vals)
    if len(s) == 0:
        return None
    return s[len(s) // 2]


def quant(vals, f):
    s = sorted(vals)
    if len(s) == 0:
        return None
    return s[int((len(s) - 1) * f)]


def calibrate(obs):
    """Pick the camera-axis convention that makes the wood table flat and below the camera."""
    best = None
    info = {}
    for conv in ("cv", "gl", "ros"):
        score = 0.0
        zs_all = []
        okc = True
        for name in ("front", "wrist"):
            cam = obs.cameras[name]
            rgb, dep = arrays(cam)
            cf = cam_frame(cam)
            H = len(rgb); W = len(rgb[0])
            zs = []
            for v in range(0, H, 8):
                for u in range(0, W, 8):
                    r, g, b = rgb[v][u]
                    d = float(dep[v][u])
                    if d <= 0.05 or d > 5.0 or not is_wood(r, g, b):
                        continue
                    p = project(cf, conv, u, v, d)
                    zs.append(p[2])
            if len(zs) < 20:
                okc = False
                continue
            m = median(zs)
            spread = quant(zs, 0.9) - quant(zs, 0.1)
            if m > cf[5][2]:
                spread += 10.0
            score += spread
            zs_all.extend(zs)
            info[conv + "_" + name] = [round(m, 4), round(spread, 4), len(zs)]
        if okc and (best is None or score < best[0]):
            best = (score, conv, median(zs_all))
    S["conv"] = best[1]
    S["table_z"] = best[2]
    ev("calibrate", {"conv": best[1], "table_z": round(best[2], 4), "info": info})


def scan(obs, step_px, want_colors, z_lo, z_hi, ee=None):
    """Return list of (x, y, z, label, cam) for pixels whose class is in want_colors."""
    out = []
    for name in ("front", "wrist"):
        cam = obs.cameras[name]
        rgb, dep = arrays(cam)
        cf = cam_frame(cam)
        H = len(rgb); W = len(rgb[0])
        wrist = name == "wrist"
        vfing = int(H * 0.70)
        ul = int(W * 0.45); ur = int(W * 0.80)
        for v in range(0, H, step_px):
            row = rgb[v]
            drow = dep[v]
            for u in range(0, W, step_px):
                if wrist and v > vfing and (u < ul or u > ur):
                    continue
                r, g, b = row[u]
                if want_colors == "white":
                    if not is_white(r, g, b):
                        continue
                    lab = "white"
                else:
                    lab = classify(r, g, b)
                    if lab is None or lab not in want_colors:
                        continue
                d = float(drow[u])
                if d <= 0.05 or d > 5.0:
                    continue
                p = project(cf, S["conv"], u, v, d)
                dz = p[2] - S["table_z"]
                if dz < z_lo or dz > z_hi:
                    continue
                if want_colors == "white" and S["static"] is not None:
                    key = (int(math.floor(p[0] / 0.02)), int(math.floor(p[1] / 0.02)))
                    if key in S["static"]:
                        continue
                if ee is not None:
                    dx = p[0] - ee[0]; dy = p[1] - ee[1]
                    if dx * dx + dy * dy < 0.09 * 0.09 and p[2] > ee[2] + 0.035:
                        continue
                out.append((p[0], p[1], p[2], lab, name))
    return out


def grid_components(pts, cell):
    cells = {}
    for i, p in enumerate(pts):
        key = (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)))
        if key not in cells:
            cells[key] = []
        cells[key].append(i)
    seen = set()
    comps = []
    for key in cells:
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        members = []
        while len(stack) > 0:
            k = stack.pop()
            members.extend(cells[k])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nk = (k[0] + dx, k[1] + dy)
                    if nk in cells and nk not in seen:
                        seen.add(nk)
                        stack.append(nk)
        comps.append(members)
    return comps


def centroid(pts, idx):
    n = float(len(idx))
    return (sum(pts[i][0] for i in idx) / n, sum(pts[i][1] for i in idx) / n,
            max(pts[i][2] for i in idx))


# ------------------------------------------------------------------ tracking
TRACKS = []


def init_scene(obs):
    calibrate(obs)
    # cubes
    cols = [c for c in COLOR_WORDS if c != "white"] + ["white"]
    pts = scan(obs, 1, cols, -0.01, 0.07)
    cubes = []
    buttons = []
    for c in cols:
        sub = [p for p in pts if p[3] == c]
        for comp in grid_components(sub, 0.012):
            if len(comp) < 6:
                continue
            cx, cy, cz = centroid(sub, comp)
            xs = [sub[i][0] for i in comp]; ys = [sub[i][1] for i in comp]
            ext = max(max(xs) - min(xs), max(ys) - min(ys))
            cubes.append({"color": c, "x": cx, "y": cy, "top": cz - S["table_z"], "n": len(comp),
                          "ext": ext})
    # buttons: the two most populated white low components of button size (robot base is larger)
    # v03: buttons are FLAT: their measured height is well below the coloured cubes'
    # (runtime-measured median cube height), which rejects white cubes (ep1 run3).
    ctops = sorted([c["top"] for c in cubes if c["color"] != "white" and c["n"] >= 6 and c["ext"] < 0.06])
    cube_top = ctops[len(ctops) // 2] if len(ctops) > 0 else 0.034
    S["cube_top"] = cube_top
    allw = [c for c in cubes if c["color"] == "white"]
    ev("white_comps", [[round(c["x"], 3), round(c["y"], 3), round(c["top"], 3), c["n"], round(c["ext"], 3)]
                       for c in allw])
    # dedicated low-band scan: only points below 0.6 x cube height (button ring/cap level)
    low = scan(obs, 1, "white", -0.01, 0.6 * cube_top)
    lows = []
    for comp in grid_components(low, 0.01):
        if len(comp) < 15:
            continue
        cx, cy, cz = centroid(low, comp)
        xs = [low[i][0] for i in comp]; ys = [low[i][1] for i in comp]
        ext = max(max(xs) - min(xs), max(ys) - min(ys))
        lows.append({"color": "white", "x": cx, "y": cy, "top": cz - S["table_z"], "n": len(comp), "ext": ext})
    ev("low_white_comps", [[round(c["x"], 3), round(c["y"], 3), round(c["top"], 3), c["n"], round(c["ext"], 3)]
                           for c in lows])
    whites = [c for c in lows if 0.02 < c["ext"] < 0.11]
    whites.sort(key=lambda c: -c["n"])
    if len(whites) < 2:
        extra = [c for c in allw if 0.03 < c["ext"] < 0.09 and c["top"] < 0.7 * cube_top]
        extra.sort(key=lambda c: -c["n"])
        for c in extra:
            if len(whites) < 2 and all(math.sqrt((c["x"] - w["x"]) ** 2 + (c["y"] - w["y"]) ** 2) > 0.05
                                       for w in whites):
                whites.append(c)
    for c in whites[:2]:
        buttons.append(c)
    # static tall white structure (robot base, occluders) -> excluded grid cells
    tall = scan(obs, 2, "white", 0.025, 0.30)
    static = set()
    for comp in grid_components(tall, 0.02):
        xs = [tall[i][0] for i in comp]; ys = [tall[i][1] for i in comp]
        ext = max(max(xs) - min(xs), max(ys) - min(ys))
        if ext > 0.06:
            for i in comp:
                kx = int(math.floor(tall[i][0] / 0.02)); ky = int(math.floor(tall[i][1] / 0.02))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        static.add((kx + dx, ky + dy))
    S["static"] = static
    ev("static_cells", len(static))
    cubes = [c for c in cubes if not (c["color"] == "white" and any(
        math.sqrt((c["x"] - bt["x"]) ** 2 + (c["y"] - bt["y"]) ** 2) < 0.04 for bt in buttons))]
    cubes = [c for c in cubes if c["ext"] < 0.06 and c["n"] >= 6]
    for i, c in enumerate(cubes):
        TRACKS.append({"id": i, "color": c["color"], "x": c["x"], "y": c["y"], "vx": 0.0, "vy": 0.0,
                       "miss": 0, "n": 0, "top": None, "pts": []})
    ev("cubes", [[c["color"], round(c["x"], 3), round(c["y"], 3), round(c["top"], 3), c["n"],
                  round(c["ext"], 3)] for c in cubes])
    ev("buttons", [[round(b["x"], 3), round(b["y"], 3), round(b["top"], 3), b["n"], round(b["ext"], 3)]
                   for b in buttons])
    return buttons


def update_tracks(obs, ee):
    hc = S["hc"]
    if hc is None:
        z_lo, z_hi = 0.03, 0.25
    else:
        z_lo, z_hi = hc - 0.025, hc + 0.02
    pts = scan(obs, 2, "white", z_lo, z_hi, ee)
    if S.get("baseline") is None:
        S["baseline"] = len(pts)
        return len(pts)
    if not S["containers_seen"]:
        if len(pts) > S["baseline"] + 150:
            S["containers_seen"] = True
            S["appear_tick"] = S["tick"]
            ev("containers_appear", len(pts))
        else:
            return len(pts)
    if hc is None:
        zs = [p[2] - S["table_z"] for p in pts]
        if len(zs) > 100 and S["tick"] - S["appear_tick"] >= 8:
            S["hc_samples"].append(quant(zs, 0.95))
        if len(S["hc_samples"]) >= 4:
            S["hc"] = min(0.12, max(0.04, median(S["hc_samples"])))
            ev("hc", [round(S["hc"], 4), [round(x, 4) for x in S["hc_samples"]]])
    gate = 0.045
    assign = {}
    free = []
    for p in pts:
        best = None
        bd = gate * gate
        for t in TRACKS:
            px = t["x"] + t["vx"]; py = t["y"] + t["vy"]
            d = (p[0] - px) ** 2 + (p[1] - py) ** 2
            if d < bd:
                bd = d
                best = t["id"]
        if best is None:
            free.append(p)
        else:
            if best not in assign:
                assign[best] = []
            assign[best].append(p)
    for t in TRACKS:
        a = assign.get(t["id"], [])
        if len(a) >= 3:
            nx = sum(p[0] for p in a) / len(a)
            ny = sum(p[1] for p in a) / len(a)
            vx = nx - t["x"]; vy = ny - t["y"]
            t["vx"] = 0.6 * t["vx"] + 0.4 * vx
            t["vy"] = 0.6 * t["vy"] + 0.4 * vy
            t["x"] = nx; t["y"] = ny
            t["miss"] = 0
            t["n"] = len(a)
            t["top"] = quant([p[2] for p in a], 0.8)
            t["pts"] = a
        else:
            # v04: coast with strongly damped velocity (bounded drift ~1.5 v), so an occluded
            # container's track stays near where it was last seen (run4: red drifted 10 cm)
            t["x"] += t["vx"]; t["y"] += t["vy"]
            t["vx"] *= 0.6; t["vy"] *= 0.6
            t["miss"] += 1
            t["n"] = len(a)
            t["pts"] = a
    # v04: scene motion = max smoothed speed of well-observed tracks (m/tick)
    mv = 0.0
    for t in TRACKS:
        if t["miss"] == 0 and t["n"] >= 15:
            sp = math.sqrt(t["vx"] ** 2 + t["vy"] ** 2)
            if sp > mv:
                mv = sp
    S["maxv"] = mv
    if mv < 0.0015:
        S["still"] = S.get("still", 0) + 1
    else:
        S["still"] = 0
    # spawn anonymous tracks for unexplained container blobs
    if len(free) >= 8 and S["tick"] - S["appear_tick"] >= 2:
        for comp in grid_components(free, 0.015):
            if len(comp) >= 10 and S["anon"] < 12:
                cx, cy, cz = centroid(free, comp)
                if (cx - ee[0]) ** 2 + (cy - ee[1]) ** 2 < 0.08 * 0.08:
                    continue
                S["anon"] += 1
                TRACKS.append({"id": len(TRACKS), "color": "anon", "x": cx, "y": cy, "vx": 0.0, "vy": 0.0,
                               "miss": 0, "n": len(comp), "top": cz, "pts": []})
                ev("spawn", [round(cx, 3), round(cy, 3), len(comp)])
    return len(pts)


# ------------------------------------------------------------------ control
ST = {"q": None, "home_quat": None, "p_cmd": None, "grip": 1.0, "emb": None, "max_steps": 480,
      "last_obs": None}


def ee_pose(obs):
    pose = obs.robot_state.end_effector_poses["primary"]
    return [float(v) for v in pose.position], [float(v) for v in pose.quaternion_wxyz]


def ik(pos, quat):
    pose = Pose(tuple(pos), tuple(quat), "robot_base")
    try:
        res = solve_ik(pose, seed_joints=ST["q"])
    except Exception as e:
        ev("ik_exc", str(e)[:200])
        return None
    if not res.ok:
        return None
    sol = None
    try:
        sol = res.joint_positions
    except Exception:
        sol = None
    if sol is None:
        try:
            sol = res.solution
        except Exception:
            sol = None
    if sol is None:
        try:
            sol = res.joints
        except Exception:
            sol = None
    if sol is None:
        return None
    try:
        sol = sol["primary"]
    except Exception:
        pass
    return [float(v) for v in sol]


def do_step(joints):
    if S["terminal"]:
        return None
    act = RobotAction(arms={"primary": ArmCommand(mode="joint_position", target=list(joints),
                                                  gripper_position=ST["grip"], embodiment=ST["emb"])})
    rep = step(act)
    S["tick"] += 1
    term = False
    try:
        term = bool(rep.terminated) or bool(rep.truncated)
    except Exception:
        term = False
    if term:
        S["terminal"] = True
        ev("terminal", None)
        return None
    obs = None
    try:
        obs = rep.observation
    except Exception:
        obs = None
    if obs is None:
        obs = get_observation()
    ST["last_obs"] = obs
    try:
        ST["q"] = [float(v) for v in obs.robot_state.joint_positions["primary"]]
    except Exception:
        pass
    pos, qq = ee_pose(obs)
    npts = update_tracks(obs, pos)
    if S["tick"] % 2 == 0:
        LOG["hist"].append([S["tick"], npts, [[t["id"], round(t["x"], 3), round(t["y"], 3), t["n"]]
                                              for t in TRACKS if t["color"] in GOAL_COLORS],
                            [round(pos[0], 3), round(pos[1], 3), round(pos[2], 3)], ST["grip"],
                            round(S.get("maxv", 0.0), 4)])
    return obs


def ticks_left():
    return ST["max_steps"] - S["tick"] - 2


def servo(goal_fn, quat, max_ticks, tol=0.01, speed=0.012, settle=0):
    """Cartesian servo: each tick move the commanded point toward goal_fn() by <= speed."""
    n = 0
    stable = 0
    while n < max_ticks and not S["terminal"] and ticks_left() > 0:
        goal = goal_fn()
        p = ST["p_cmd"]
        d = [goal[i] - p[i] for i in range(3)]
        dist = math.sqrt(sum(x * x for x in d))
        if dist > speed:
            p = [p[i] + d[i] * speed / dist for i in range(3)]
        else:
            p = list(goal)
        sol = ik(p, quat)
        if sol is None:
            ev("ik_fail", [round(x, 3) for x in p])
            sol = ST["q"]
        else:
            ST["p_cmd"] = p
        obs = do_step(sol)
        n += 1
        if obs is None:
            break
        meas, mq = ee_pose(obs)
        err = math.sqrt(sum((meas[i] - goal[i]) ** 2 for i in range(3)))
        if err < tol:
            stable += 1
            if stable > settle:
                return True, n
        else:
            stable = 0
    return False, n


def hold(ticks):
    for k in range(ticks):
        if S["terminal"] or ticks_left() <= 0:
            return
        sol = ik(ST["p_cmd"], ST["cur_quat"])
        if sol is None:
            sol = ST["q"]
        do_step(sol)


def wait_still(cap_tick, need):
    """v04: hold at the high home pose (wrist sees the whole table from above) until the
    containers have stopped moving for `need` ticks, or until cap_tick."""
    t0 = S["tick"]
    S["still"] = 0
    while not S["terminal"] and ticks_left() > 0 and S["tick"] < cap_tick:
        if S.get("still", 0) >= need:
            break
        hold(1)
    ev("wait_still", {"from": t0, "to": S["tick"], "still": S.get("still", 0),
                      "maxv": round(S.get("maxv", 0.0), 4)})
    return S.get("still", 0) >= need


def yaw_quat(yaw):
    h = ST["home_quat"]
    rz = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
    return qmul(rz, h)


def container_yaw(t):
    pts = t["pts"]
    if len(pts) < 8:
        return 0.0
    cx = sum(p[0] for p in pts) / len(pts); cy = sum(p[1] for p in pts) / len(pts)
    best = None
    for k in range(18):
        a = math.radians(-45 + 5 * k)
        ca = math.cos(a); sa = math.sin(a)
        us = [(p[0] - cx) * ca + (p[1] - cy) * sa for p in pts]
        vs = [-(p[0] - cx) * sa + (p[1] - cy) * ca for p in pts]
        area = (quant(us, 0.95) - quant(us, 0.05)) * (quant(vs, 0.95) - quant(vs, 0.05))
        if best is None or area < best[0]:
            best = (area, a)
    return best[1]


def track_by_id(i):
    for t in TRACKS:
        if t["id"] == i:
            return t
    return None


def nearest_container(x, y):
    best = None
    for t in TRACKS:
        if t["miss"] > 30:
            continue
        for k in (0, 8):
            d = math.sqrt((t["x"] + k * t["vx"] - x) ** 2 + (t["y"] + k * t["vy"] - y) ** 2)
            if best is None or d < best:
                best = d
    return best


def press(bt, start_after, force_at):
    tz = S["table_z"]
    hc = S["hc"] if S["hc"] is not None else 0.07
    ST["grip"] = 0.0
    q = ST["home_quat"]
    ST["cur_quat"] = q
    # watch from home until the scheduled start (keeps the wrist camera's full view for tracking)
    while S["tick"] < start_after and not S["terminal"]:
        hold(1)
    above = [bt["x"], bt["y"], tz + hc + 0.07]
    ok1, n1 = servo(lambda: above, q, 45, tol=0.015, speed=0.016)
    waited = 0
    clear = None
    while not S["terminal"] and S["tick"] < force_at:
        clear = nearest_container(bt["x"], bt["y"])
        if clear is None or clear > 0.085:
            break
        hold(1)
        waited += 1
    down = [bt["x"], bt["y"], tz + 0.006]
    ok2, n2 = servo(lambda: down, q, 14, tol=0.006, speed=0.015)
    hold(2)
    ok3, n3 = servo(lambda: above, q, 12, tol=0.02, speed=0.014)
    ev("press", {"btn": [round(bt["x"], 3), round(bt["y"], 3)], "ok": [ok1, ok2, ok3], "n": [n1, n2, n3],
                 "waited": waited, "clear": None if clear is None else round(clear, 3)})


def gripper_now():
    try:
        return float(ST["last_obs"].robot_state.gripper_positions["primary"])
    except Exception:
        return None


def ee_now():
    try:
        p, qq = ee_pose(ST["last_obs"])
        return p
    except Exception:
        return ST["p_cmd"]


def pick(color):
    cands = [t for t in TRACKS if t["color"] == color]
    if len(cands) == 0:
        ev("no_track", color)
        return False
    t = cands[0]
    tz = S["table_z"]
    hc = S["hc"] if S["hc"] is not None else 0.07
    info = {"color": color, "track": t["id"], "tries": []}
    for attempt in range(3):
        if S["terminal"] or ticks_left() < 40:
            break
        ST["grip"] = 1.0
        q = ST["home_quat"]
        ST["cur_quat"] = q

        def above_goal():
            return [t["x"] + 2 * t["vx"], t["y"] + 2 * t["vy"], tz + hc + 0.08]

        ok1, n1 = servo(above_goal, q, 60, tol=0.012, speed=0.014, settle=1)
        top = [p for p in t["pts"] if t["top"] is None or p[2] > t["top"] - 0.012]
        yaw = container_yaw({"pts": top})
        q2 = yaw_quat(yaw)
        ST["cur_quat"] = q2
        # v02: grasp the container's top knob just below its measured top (the
        # successful v01 holds closed at ~top-0.009); do not command far below the
        # contact height, which pushed/tipped the container in v01 try 1.
        grip_z = max(tz + 0.015, tz + hc - 0.012)
        pre_z = tz + hc + 0.03

        def pre_goal():
            return [t["x"] + 2 * t["vx"], t["y"] + 2 * t["vy"], pre_z]

        def down_goal():
            return [t["x"] + 2 * t["vx"], t["y"] + 2 * t["vy"], grip_z]

        okp, npre = servo(pre_goal, q2, 25, tol=0.006, speed=0.01, settle=2)
        ok2, n2 = servo(down_goal, q2, 20, tol=0.006, speed=0.006, settle=1)
        # final alignment check against the live track before closing
        extra = 0
        while extra < 10 and not S["terminal"]:
            e = ee_now()
            off = math.sqrt((e[0] - t["x"]) ** 2 + (e[1] - t["y"]) ** 2)
            if off < 0.01:
                break
            servo(down_goal, q2, 1, tol=0.004, speed=0.01)
            extra += 1
        e = ee_now()
        pos0 = [round(t["x"], 3), round(t["y"], 3), round(e[0], 3), round(e[1], 3), round(e[2] - tz, 3)]
        ST["grip"] = 0.0
        hold(8)
        gw = gripper_now()
        lift = [ST["p_cmd"][0], ST["p_cmd"][1], grip_z + 0.12]
        ok3, n3 = servo(lambda: lift, q2, 16, tol=0.02, speed=0.012)
        hold(3)
        gw2 = gripper_now()
        e2 = ee_now()
        held = gw2 is not None and gw2 > 0.08
        place = [lift[0], lift[1], grip_z + 0.005]
        n4 = 0
        if held:
            ok4, n4 = servo(lambda: place, q2, 16, tol=0.01, speed=0.012)
        ST["grip"] = 1.0
        hold(5)
        up = [ST["p_cmd"][0], ST["p_cmd"][1], tz + hc + 0.08]
        ok5, n5 = servo(lambda: up, q2, 12, tol=0.02, speed=0.014)
        info["tries"].append({"yaw": round(math.degrees(yaw), 1), "at": pos0, "ok": [ok1, ok2],
                              "n": [n1, npre, n2, extra, n3, n4, n5], "grip_closed": gw, "grip_lifted": gw2,
                              "lift_z": round(e2[2] - tz, 3), "held": held, "tick": S["tick"]})
        if held:
            break
    ev("pick", info)
    return True


GOAL_COLORS = []
summary = {}
try:
    ctx = get_task_context()
    lang = str(ctx.language)
    summary["goal"] = lang
    low = lang.lower()
    found = []
    for c in COLOR_WORDS:
        k = low.find(c + " cube")
        if k >= 0:
            found.append((k, c))
    found.sort()
    GOAL_COLORS = [c for k, c in found]
    summary["goal_colors"] = GOAL_COLORS
    try:
        md = robomme_plus.get_task_metadata()
        summary["metadata"] = {k: str(md[k])[:200] for k in md}
        try:
            ST["max_steps"] = int(md["max_steps"])
        except Exception:
            pass
    except Exception as e:
        summary["metadata_error"] = str(e)[:200]
    try:
        cm = robomme_plus.get_controller_metadata()
        summary["controller"] = {k: str(cm[k])[:200] for k in cm}
    except Exception as e:
        summary["controller_error"] = str(e)[:200]
    try:
        vid = robomme_plus.get_conditioning_video()
        summary["video_len"] = len(vid)
    except Exception as e:
        summary["video_error"] = str(e)[:200]
    obs0 = get_observation()
    ST["last_obs"] = obs0
    st = obs0.robot_state
    ST["emb"] = st.embodiment
    ST["q"] = [float(v) for v in st.joint_positions["primary"]]
    pos, qq = ee_pose(obs0)
    ST["home_quat"] = qq
    ST["home_pos"] = pos
    ST["p_cmd"] = list(pos)
    ST["cur_quat"] = qq
    summary["home"] = [[round(v, 4) for v in pos], [round(v, 4) for v in qq]]
    for name in ("front", "wrist"):
        cam = obs0.cameras[name]
        summary["cam_" + name] = {"pose": [[round(float(v), 4) for v in cam.camera_pose.position],
                                           [round(float(v), 4) for v in cam.camera_pose.quaternion_wxyz]],
                                  "K": [[round(float(v), 2) for v in row] for row in cam.intrinsics]}
    buttons = init_scene(obs0)
    goal_tracks = [t for t in TRACKS if t["color"] in GOAL_COLORS]
    summary["goal_tracks_init"] = [[t["id"], t["color"], round(t["x"], 3), round(t["y"], 3)] for t in goal_tracks]
    # press buttons nearest-first
    # all three expert recordings press the +y (robot-left) button first, then the -y button,
    # at roughly ticks 85 and 170; follow that order and schedule (press deadline: tick 180)
    order = sorted(buttons, key=lambda b: -b["y"])
    hp = ST["home_pos"]
    if len(order) >= 1:
        press(order[0], 40, 100)
        summary["press1_done"] = S["tick"]
        # watch pose: high above the button pair so the wrist camera sees the whole table
        mx = sum(b["x"] for b in order) / len(order)
        watch = [mx + 0.08, 0.0, S["table_z"] + 0.36]
        ST["cur_quat"] = ST["home_quat"]
        servo(lambda: watch, ST["home_quat"], 25, tol=0.02, speed=0.018)
    if len(order) >= 2:
        press(order[1], 132, 158)
        summary["press2_done"] = S["tick"]
    summary["tick_after_press"] = S["tick"]
    # return high over the scene so the wrist camera re-observes everything
    ST["cur_quat"] = ST["home_quat"]
    hp = ST["home_pos"]
    servo(lambda: hp, ST["home_quat"], 30, tol=0.02, speed=0.016)
    hold(4)
    summary["tick_after_home"] = S["tick"]
    # v04: pick only once the swaps have stopped (or after a bounded wait), observing from home
    budget_wait = max(0, min(160, ticks_left() - 200 * len(GOAL_COLORS)))
    summary["still_after_press"] = wait_still(S["tick"] + budget_wait, 10)
    summary["tick_pick_start"] = S["tick"]
    for k, c in enumerate(GOAL_COLORS):
        if S["terminal"] or ticks_left() < 20:
            break
        if k > 0:
            ST["cur_quat"] = ST["home_quat"]
            servo(lambda: hp, ST["home_quat"], 40, tol=0.02, speed=0.018)
            budget_wait = max(0, min(40, ticks_left() - 200 * (len(GOAL_COLORS) - k)))
            wait_still(S["tick"] + budget_wait, 6)
        pick(c)
    summary["tick_end"] = S["tick"]
except Exception as e:
    summary["error"] = str(e)[:500]
summary["terminal"] = S["terminal"]
summary["conv"] = S["conv"]
summary["table_z"] = S["table_z"]
summary["hc"] = S["hc"]
summary["appear_tick"] = S["appear_tick"]
summary["final_tracks"] = [[t["id"], t["color"], round(t["x"], 3), round(t["y"], 3), t["miss"], t["n"]]
                           for t in TRACKS]
summary["events"] = LOG["events"]
summary["hist"] = LOG["hist"]
result = summary
