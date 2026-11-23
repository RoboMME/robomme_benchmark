# v00_initial — ButtonUnmaskSwap
# Goal (from get_task_context().language): press every button once (before the published
# press deadline), then pick up the containers hiding the named cubes, in the stated order.
# Strategy:
#   1. Tick 0 (no time passes for perception): localize the named cubes and the buttons.
#   2. Drive the arm only with one-tick joint steps so the front camera can be read after
#      EVERY tick; a pure-Python 3D tracker follows every white container on the table
#      (front RGB-D -> robot_base points, height band, Voronoi/k-means update seeded by
#      constant-velocity predictions). Each named cube is bound to the container that lands
#      on it.
#   3. Press the buttons, park at home, keep tracking until containers stop moving.
#   4. Pick each target container (yaw from its top footprint), lift, put it back, next.
# Every pose/geometry is derived from perception at runtime; constants are generic
# tolerances, bands, and step limits.
import math

COLORS = ["red", "green", "blue", "yellow", "purple", "orange", "black", "white",
          "pink", "brown", "gray", "grey", "cyan"]
CAM = "front"
WHITE_MIN = 140          # container pixels: bright ...
WHITE_SPREAD = 45        # ... and unsaturated (wood is orange, cubes are saturated)
BAND_LO = 0.045          # above button caps / cubes / table
BAND_HI = 0.16           # below the parked/transiting arm links
ROI_STRIDE = 2
GATE = 0.06              # metres; a point joins a track only within this radius
EE_EXCLUDE = 0.12        # ignore points this close (XY) to a low end effector
X_MIN = 0.22             # in front of the robot pedestal (pedestal itself is in the band)
X_MAX = 0.95
Y_ABS = 0.45
MAX_DQ = 0.06            # rad per tick for self-generated joint interpolation

report = {"events": [], "errors": []}
S = {"ended": False, "tick": 0, "grip": 1.0, "tracks": [], "track_ok": False,
     "last_pts": [], "speed_hist": [], "obs": None}


def log(msg):
    report["events"].append([S["tick"], msg])


# ---------------------------------------------------------------- language
context = get_task_context()
lang = str(context.language)
report["language"] = lang
low = lang.lower().replace(",", " ").replace(".", " ")
words = low.split()
pick_part = low
if "then" in words:
    pick_part = low[low.index("then"):]
order = []
for w in pick_part.split():
    if w in COLORS and w not in order:
        order.append(w)
n_buttons = 1
if "both" in words or "two" in words:
    n_buttons = 2
if "three" in words:
    n_buttons = 3
report["order"] = order
report["n_buttons"] = n_buttons

state0 = get_robot_state()
base_frame = state0.base_frame
embodiment = state0.embodiment
home_q = [float(v) for v in state0.joint_positions["primary"]]
home_quat = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]
try:
    max_steps_total = int(robomme_plus.get_task_metadata()["max_steps"])
except Exception:
    max_steps_total = 1000
report["max_steps"] = max_steps_total


# ---------------------------------------------------------------- geometry helpers
def quat_to_rot(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


obs0 = get_observation()
cam0 = obs0.cameras[CAM]
K = cam0.intrinsics
FX, FY, CX, CY = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
RC = quat_to_rot([float(v) for v in cam0.camera_pose.quaternion_wxyz])
TC = [float(v) for v in cam0.camera_pose.position]
IMG_H = len(cam0.rgb)
IMG_W = len(cam0.rgb[0])
report["camera_frame"] = str(cam0.camera_pose.frame)


def row_list(arr, v):
    row = arr[v]
    try:
        return row.tolist()
    except Exception:
        return [row[i] for i in range(len(row))]


def container_points(obs):
    """White, unsaturated front-camera pixels projected to robot_base, kept in the height band."""
    cam = obs.cameras[CAM]
    rgb = cam.rgb
    dep = cam.depth_m
    ee = obs.robot_state.end_effector_poses["primary"].position
    ex, ey, ez = float(ee[0]), float(ee[1]), float(ee[2])
    exclude_ee = ez < 0.30
    pts = []
    v = 0
    while v < IMG_H:
        rrow = row_list(rgb, v)
        drow = None
        u = 0
        while u < IMG_W:
            p = rrow[u]
            r, g, b = p[0], p[1], p[2]
            mn = min(r, g, b)
            if mn > WHITE_MIN and max(r, g, b) - mn < WHITE_SPREAD:
                if drow is None:
                    drow = row_list(dep, v)
                zc = float(drow[u])
                if zc > 0.05:
                    xc = (u - CX) * zc / FX
                    yc = (v - CY) * zc / FY
                    X = RC[0][0] * xc + RC[0][1] * yc + RC[0][2] * zc + TC[0]
                    Y = RC[1][0] * xc + RC[1][1] * yc + RC[1][2] * zc + TC[1]
                    Z = RC[2][0] * xc + RC[2][1] * yc + RC[2][2] * zc + TC[2]
                    if BAND_LO < Z < BAND_HI and X_MIN < X < X_MAX and -Y_ABS < Y < Y_ABS:
                        if not (exclude_ee and (X - ex) ** 2 + (Y - ey) ** 2 < EE_EXCLUDE ** 2):
                            pts.append((X, Y, Z))
            u += ROI_STRIDE
        v += ROI_STRIDE
    return pts


def grid_clusters(pts, cell=0.01, min_pts=6):
    """Connected components over a 1 cm XY occupancy grid."""
    cells = {}
    for p in pts:
        key = (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)))
        if key in cells:
            cells[key].append(p)
        else:
            cells[key] = [p]
    seen = set()
    clusters = []
    for key in cells:
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        members = []
        while stack:
            k = stack.pop()
            members.extend(cells[k])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (k[0] + dx, k[1] + dy)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        if len(members) >= min_pts:
            sx = sum(m[0] for m in members) / len(members)
            sy = sum(m[1] for m in members) / len(members)
            clusters.append({"x": sx, "y": sy, "n": len(members)})
    return clusters


def track_update(pts):
    tracks = S["tracks"]
    if not tracks:
        return
    preds = []
    for t in tracks:
        preds.append((t["x"] + t["vx"], t["y"] + t["vy"]))
    sums = [[0.0, 0.0, 0] for t in tracks]
    assign = []
    g2 = GATE * GATE
    for p in pts:
        best = -1
        bd = g2
        for i in range(len(preds)):
            dx = p[0] - preds[i][0]
            dy = p[1] - preds[i][1]
            d = dx * dx + dy * dy
            if d < bd:
                bd = d
                best = i
        assign.append(best)
        if best >= 0:
            s = sums[best]
            s[0] += p[0]
            s[1] += p[1]
            s[2] += 1
    for i in range(len(tracks)):
        t = tracks[i]
        s = sums[i]
        if s[2] >= 6:
            nx = s[0] / s[2]
            ny = s[1] / s[2]
            t["vx"] = 0.6 * t["vx"] + 0.4 * (nx - t["x"])
            t["vy"] = 0.6 * t["vy"] + 0.4 * (ny - t["y"])
            t["x"] = nx
            t["y"] = ny
            t["n"] = s[2]
            t["miss"] = 0
        else:
            t["x"] = preds[i][0]
            t["y"] = preds[i][1]
            t["vx"] *= 0.8
            t["vy"] *= 0.8
            t["n"] = s[2]
            t["miss"] += 1
    S["last_pts"] = pts
    S["last_assign"] = assign
    # Late-landing containers: unassigned points that form a compact cluster become a new
    # track, but only early in the episode (later, unassigned points are arm/noise).
    if S["tick"] < 40:
        free = [pts[i] for i in range(len(pts)) if assign[i] < 0]
        if len(free) >= 15:
            for c in grid_clusters(free, min_pts=15):
                tracks.append({"x": c["x"], "y": c["y"], "vx": 0.0, "vy": 0.0, "n": c["n"], "miss": 0})
                log("spawned track %d at %.3f %.3f" % (len(tracks) - 1, c["x"], c["y"]))


def observe(obs):
    """Called after every tick with the step's observation."""
    S["obs"] = obs
    if S.get("skip_tracking"):
        return
    pts = container_points(obs)
    if not S["track_ok"]:
        init_tracks(pts)
        return
    track_update(pts)
    sp = 0.0
    for t in S["tracks"]:
        sp = max(sp, math.sqrt(t["vx"] ** 2 + t["vy"] ** 2))
    S["speed_hist"].append(round(sp, 5))
    if S["tick"] % 10 == 0:
        snap = [[round(t["x"], 3), round(t["y"], 3), t["n"]] for t in S["tracks"]]
        report.setdefault("track_log", []).append([S["tick"], snap])


def init_tracks(pts):
    cl = grid_clusters(pts)
    hist = S.setdefault("init_hist", [])
    hist.append(len(cl))
    report.setdefault("init_counts", []).append([S["tick"], len(cl), len(pts)])
    # Containers have landed when the cluster count is non-zero and unchanged for 3 ticks.
    if len(cl) >= max(1, len(order)) and len(hist) >= 3 and hist[-1] == hist[-2] == hist[-3]:
        S["tracks"] = [{"x": c["x"], "y": c["y"], "vx": 0.0, "vy": 0.0, "n": c["n"], "miss": 0}
                       for c in cl]
        S["track_ok"] = True
        bind_targets()


def bind_targets():
    bind = {}
    for color in order:
        cube = cubes.get(color)
        if cube is None:
            continue
        best = None
        bd = None
        for i in range(len(S["tracks"])):
            t = S["tracks"][i]
            d = (t["x"] - cube[0]) ** 2 + (t["y"] - cube[1]) ** 2
            if bd is None or d < bd:
                bd = d
                best = i
        bind[color] = best
        log("bind %s -> track %s dist %.3f" % (color, str(best), math.sqrt(bd) if bd is not None else -1))
    S["bind"] = bind
    report["bind"] = bind
    report["tracks_init"] = [[round(t["x"], 3), round(t["y"], 3), t["n"]] for t in S["tracks"]]


# ---------------------------------------------------------------- stepping
def tick(q, grip):
    if S["ended"]:
        return False
    sr = step(RobotAction(arms={"primary": ArmCommand(
        "joint_position", [float(v) for v in q], gripper_position=float(grip),
        embodiment=embodiment)}))
    S["tick"] += 1
    term = False
    try:
        term = bool(sr.terminated) or bool(sr.truncated)
    except Exception:
        term = False
    if term:
        S["ended"] = True
        log("episode ended")
        return False
    try:
        obs = sr.observation
        if obs is None:
            obs = get_observation()
    except Exception:
        obs = get_observation()
    try:
        observe(obs)
    except Exception as exc:
        report["errors"].append("observe@%d: %s" % (S["tick"], str(exc)[:200]))
    return True


def current_q():
    obs = S["obs"]
    if obs is not None:
        return [float(v) for v in obs.robot_state.joint_positions["primary"]]
    return [float(v) for v in get_robot_state().joint_positions["primary"]]


def goto_joints(target, grip, max_ticks, tol=0.02, max_dq=MAX_DQ):
    """Interpolate toward joint target one tick at a time (tracking after every tick)."""
    cmd = current_q()
    for k in range(max_ticks):
        if S["ended"]:
            return False
        nxt = []
        done_cmd = True
        for j in range(7):
            d = target[j] - cmd[j]
            if d > max_dq:
                d = max_dq
                done_cmd = False
            elif d < -max_dq:
                d = -max_dq
                done_cmd = False
            nxt.append(cmd[j] + d)
        cmd = nxt
        if not tick(cmd, grip):
            return False
        if done_cmd:
            q = current_q()
            err = max(abs(q[j] - target[j]) for j in range(7))
            if err < tol:
                return True
    return False


def goto_pose(pos, quat, grip, max_ticks, tol=0.02, max_dq=MAX_DQ):
    seed = current_q()
    ik = solve_ik(Pose((float(pos[0]), float(pos[1]), float(pos[2])), quat, base_frame),
                  seed_joints=seed)
    if not ik.ok:
        log("ik failed for %s" % str([round(float(v), 3) for v in pos]))
        return False
    target = [float(v) for v in ik.joint_positions]
    return goto_joints(target, grip, max_ticks, tol, max_dq)


def hold(n, grip):
    q = current_q()
    for k in range(n):
        if not tick(q, grip):
            return False
    return True


def ee_now():
    obs = S["obs"]
    st = obs.robot_state if obs is not None else get_robot_state()
    return [float(v) for v in st.end_effector_poses["primary"].position]


# ---------------------------------------------------------------- tick-0 perception
cubes = {}
for color in order:
    try:
        seg = segment_text(CAM, color + " cube")
        if seg.ok and len(seg.segmentations) > 0:
            cand = seg.segmentations[0]
            pc = mask_to_point_cloud(cand, CAM, target_frame=base_frame)
            if len(pc.points) >= 10:
                cubes[color] = (quantile([p[0] for p in pc.points], 0.5),
                                quantile([p[1] for p in pc.points], 0.5),
                                float(cand.score))
    except Exception as exc:
        report["errors"].append("cube %s: %s" % (color, str(exc)[:200]))
report["cubes"] = {c: [round(v, 4) for v in cubes[c]] for c in cubes}

buttons = []
try:
    seg = segment_text(CAM, "button")
    cands = []
    if seg.ok:
        for cand in seg.segmentations[:5]:
            pc = mask_to_point_cloud(cand, CAM, target_frame=base_frame)
            if len(pc.points) < 10:
                continue
            bx = quantile([p[0] for p in pc.points], 0.5)
            by = quantile([p[1] for p in pc.points], 0.5)
            ztop = quantile([p[2] for p in pc.points], 0.95)
            zlo = quantile([p[2] for p in pc.points], 0.05)
            cands.append({"x": bx, "y": by, "top": ztop, "zlo": zlo, "score": float(cand.score)})
    report["button_cands"] = [[round(c["x"], 3), round(c["y"], 3), round(c["top"], 3),
                               round(c["zlo"], 3), round(c["score"], 3)] for c in cands]
    # Keep the highest-scoring, mutually distinct candidates that are not one of the named cubes.
    for c in sorted(cands, key=lambda c: -c["score"]):
        if len(buttons) >= n_buttons:
            break
        near_cube = False
        for col in cubes:
            if (c["x"] - cubes[col][0]) ** 2 + (c["y"] - cubes[col][1]) ** 2 < 0.02 ** 2:
                near_cube = True
        dup = False
        for b in buttons:
            if (c["x"] - b["x"]) ** 2 + (c["y"] - b["y"]) ** 2 < 0.04 ** 2:
                dup = True
        if not near_cube and not dup:
            buttons.append(c)
except Exception as exc:
    report["errors"].append("buttons: %s" % str(exc)[:200])
report["buttons"] = [[round(b["x"], 3), round(b["y"], 3), round(b["top"], 3)] for b in buttons]

# initialise tracking state from the tick-0 observation
S["obs"] = obs0
try:
    observe(obs0)
except Exception as exc:
    report["errors"].append("observe0: %s" % str(exc)[:200])

# wait (holding still, tracking every tick) until the containers have landed and been bound
while not S["ended"] and not S["track_ok"] and S["tick"] < 40:
    hold(1, 1.0)
report["tick_track_init"] = S["tick"]

# ---------------------------------------------------------------- 1. press buttons
grip_closed = 0.0
ee0 = ee_now()
buttons_sorted = sorted(buttons, key=lambda b: (b["x"] - ee0[0]) ** 2 + (b["y"] - ee0[1]) ** 2)
press_log = []
for b in buttons_sorted:
    if S["ended"]:
        break
    above = (b["x"], b["y"], b["top"] + 0.06)
    ok1 = goto_pose(above, home_quat, grip_closed, 45, tol=0.03, max_dq=0.08)
    down = (b["x"], b["y"], b["top"] - 0.012)
    ok2 = goto_pose(down, home_quat, grip_closed, 14, tol=0.01, max_dq=0.04)
    eed = ee_now()
    hold(2, grip_closed)
    ok3 = goto_pose(above, home_quat, grip_closed, 14, tol=0.03, max_dq=0.06)
    press_log.append({"ok": [ok1, ok2, ok3], "ee_down": [round(v, 4) for v in eed], "tick": S["tick"]})
report["press"] = press_log
report["tick_after_press"] = S["tick"]

# ---------------------------------------------------------------- 2. park and wait for stillness
goto_joints(home_q, 1.0, 60, tol=0.03, max_dq=0.08)
still = 0
wait_cap = min(max_steps_total - 400, 700)
while not S["ended"] and S["tick"] < wait_cap:
    if not hold(1, 1.0):
        break
    if S["track_ok"] and S["speed_hist"] and S["speed_hist"][-1] < 0.0006:
        still += 1
    else:
        still = 0
    if still >= 25 and S["tick"] > 60:
        break
report["tick_still"] = S["tick"]
report["still_reached"] = still >= 25


# ---------------------------------------------------------------- 3. picks
def target_geometry(ti):
    """Top-face footprint of track ti from the latest assigned points: centre, yaw, top z."""
    pts = S["last_pts"]
    assign = S.get("last_assign", [])
    mine = [pts[i] for i in range(min(len(pts), len(assign))) if assign[i] == ti]
    if len(mine) < 10:
        return None
    ztop = quantile([p[2] for p in mine], 0.97)
    top = [p for p in mine if p[2] > ztop - 0.012]
    if len(top) < 5:
        top = mine
    cx = sum(p[0] for p in top) / len(top)
    cy = sum(p[1] for p in top) / len(top)
    best_a = 0.0
    best_area = None
    for deg in range(0, 90, 3):
        a = math.radians(deg)
        ca, sa = math.cos(a), math.sin(a)
        us = [p[0] * ca + p[1] * sa for p in top]
        vs = [-p[0] * sa + p[1] * ca for p in top]
        area = (quantile(us, 0.95) - quantile(us, 0.05)) * (quantile(vs, 0.95) - quantile(vs, 0.05))
        if best_area is None or area < best_area:
            best_area = area
            best_a = a
            best_ext = [quantile(us, 0.95) - quantile(us, 0.05), quantile(vs, 0.95) - quantile(vs, 0.05)]
    # recentre on the oriented box mid-point (robust to partial side views)
    ca, sa = math.cos(best_a), math.sin(best_a)
    us = [p[0] * ca + p[1] * sa for p in top]
    vs = [-p[0] * sa + p[1] * ca for p in top]
    mu = 0.5 * (quantile(us, 0.05) + quantile(us, 0.95))
    mv = 0.5 * (quantile(vs, 0.05) + quantile(vs, 0.95))
    cx2 = mu * ca - mv * sa
    cy2 = mu * sa + mv * ca
    zs = sorted(p[2] for p in mine)
    return {"x": cx2, "y": cy2, "cx_mean": cx, "cy_mean": cy, "top": ztop, "yaw": best_a,
            "ext": best_ext, "n": len(mine), "ntop": len(top),
            "zq": [round(zs[int((len(zs) - 1) * f)], 3) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]}


def yaw_quat(yaw_mod90):
    d = yaw_mod90
    while d > math.pi / 4:
        d -= math.pi / 2
    while d <= -math.pi / 4:
        d += math.pi / 2
    qz = [math.cos(d / 2), 0.0, 0.0, math.sin(d / 2)]
    return quat_mul(qz, home_quat), d


picks = []
for color in order:
    if S["ended"]:
        break
    ti = S.get("bind", {}).get(color)
    entry = {"color": color, "track": ti}
    picks.append(entry)
    if ti is None:
        entry["error"] = "unbound"
        continue
    geo = target_geometry(ti)
    if geo is None:
        t = S["tracks"][ti]
        entry["geo_fallback"] = True
        geo = {"x": t["x"], "y": t["y"], "top": 0.08, "yaw": 0.0}
    entry["geo"] = {k: (round(v, 4) if k in ("x", "y", "top", "yaw", "cx_mean", "cy_mean") else v)
                    for k, v in geo.items()}
    quat, dyaw = yaw_quat(geo["yaw"])
    entry["dyaw_deg"] = round(math.degrees(dyaw), 1)
    grasp_z = geo["top"] - 0.02
    S["skip_tracking"] = True
    above = (geo["x"], geo["y"], geo["top"] + 0.10)
    entry["above_ok"] = goto_pose(above, quat, 1.0, 60, tol=0.02, max_dq=0.08)
    entry["descend_ok"] = goto_pose((geo["x"], geo["y"], grasp_z), quat, 1.0, 40, tol=0.01, max_dq=0.04)
    entry["ee_grasp"] = [round(v, 4) for v in ee_now()]
    hold(12, 0.0)
    try:
        entry["grip_closed"] = round(float(S["obs"].robot_state.gripper_positions["primary"]), 3)
    except Exception:
        pass
    entry["lift_ok"] = goto_pose((geo["x"], geo["y"], geo["top"] + 0.12), quat, 0.0, 40, tol=0.02, max_dq=0.05)
    hold(5, 0.0)
    try:
        entry["grip_lifted"] = round(float(S["obs"].robot_state.gripper_positions["primary"]), 3)
        entry["ee_lifted"] = [round(v, 4) for v in ee_now()]
    except Exception:
        pass
    entry["tick_lifted"] = S["tick"]
    # put the container back where it was and release
    goto_pose((geo["x"], geo["y"], grasp_z + 0.008), quat, 0.0, 40, tol=0.015, max_dq=0.05)
    hold(6, 1.0)
    goto_pose((geo["x"], geo["y"], geo["top"] + 0.10), quat, 1.0, 30, tol=0.03, max_dq=0.08)
    entry["tick_done"] = S["tick"]
    S["skip_tracking"] = False
    # return home (tracking on) so the next target is seen unoccluded
    if not S["ended"]:
        goto_joints(home_q, 1.0, 60, tol=0.03, max_dq=0.08)
        hold(2, 1.0)

report["picks"] = picks
report["final_tick"] = S["tick"]
report["ended"] = S["ended"]
report["speed_tail"] = S["speed_hist"][-40:]
report["speed_every10"] = S["speed_hist"][::10]
result = report
