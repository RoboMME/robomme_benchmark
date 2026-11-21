# VideoUnmaskSwap (robomme_plus task 15) -- v04 = v03 + goal-parser robustness (the text before the
# first "then" is dropped only when it names no colour; "each"/"both"/number words mean all-of).
# v03 (from v01; v02's coalescence guard rejected):
# grasp-point re-centring by two-pass mean shift (an UNTRACKED neighbour no longer biases the
# box centre) and one re-measure + regrasp when the gripper closes on nothing.
#
# Strategy (everything episode-dependent is derived at runtime):
#   1. Parse the ordered colour targets from get_task_context().language.
#   2. Self-calibrate the camera projection convention on the first conditioning-video frame
#      (table pixels from front and wrist must land on one horizontal plane in robot_base).
#   3. In the conditioning video: find the last frame before the white containers drop, detect
#      the coloured cubes (both cameras, background-subtracted against video frame 0), and seed
#      one container track per cube at its robot-base XY.
#   4. Track every container through the video by gated, temporally seeded k-means over the
#      robot-base XY of 'container-white' points from both cameras (upper part of the container
#      only, via a z band), then keep tracking on the live cameras one held tick at a time until
#      every container has stopped moving.
#   5. Pick up the containers in goal order (top-down grasp aligned to the container faces,
#      refined from the wrist camera at the pre-grasp), lift to reveal, put back, release.
import math

FRONT = "front"
WRIST = "wrist"
COLOUR_WORDS = {
    "red": "red", "green": "green", "blue": "blue", "yellow": "yellow", "orange": "orange",
    "purple": "purple", "violet": "purple", "pink": "purple", "magenta": "purple",
    "black": "black", "white": "white", "gray": "white", "grey": "white",
}
report = {"version": "v04", "notes": []}


def note(text):
    report["notes"].append(str(text))


# ---------------------------------------------------------------- goal parsing
def is_alpha(ch):
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def find_word(text, word):
    start = 0
    while True:
        j = text.find(word, start)
        if j < 0:
            return -1
        before = text[j - 1] if j > 0 else " "
        after = text[j + len(word)] if j + len(word) < len(text) else " "
        if not is_alpha(before) and not is_alpha(after):
            return j
        start = j + 1


def parse_goal(text):
    low = text.lower()
    k = find_word(low, "then")
    body = low
    if k >= 0:
        prefix = low[:k]
        named = False
        for word in COLOUR_WORDS:
            if find_word(prefix, word) >= 0:
                named = True
        body = low if named else low[k + 4:]
    for sep in (", followed by ", " followed by ", ", and finally ", " and finally ", ", finally ",
                " finally ", ", and then ", " and then ", ", then ", " then ", "; ", ", next ",
                " next ", ", after that ", " after that "):
        body = body.replace(sep, "|")
    items = []
    for clause in body.split("|"):
        found = None
        pos = 1000000
        for word in COLOUR_WORDS:
            j = find_word(clause, word)
            if 0 <= j < pos:
                pos = j
                found = word
        if found is None:
            continue
        plural = (found + " cubes") in clause
        for word in ("all", "every", "each", "both", "containers", "two", "three", "four", "five"):
            if find_word(clause, word) >= 0:
                plural = True
        items.append({"colour": COLOUR_WORDS[found], "all": plural, "clause": clause.strip()})
    return items


# ---------------------------------------------------------------- small math helpers
def to_list(arr):
    try:
        return arr.tolist()
    except Exception:
        return [list(row) for row in arr]


def quantile(values, frac):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * frac)]


def quat_to_r(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def wrap_half_pi(a):
    # wrap an axis angle (period pi) into (-pi/2, pi/2]
    while a > math.pi / 2:
        a -= math.pi
    while a <= -math.pi / 2:
        a += math.pi
    return a


def hue_class(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 75:
        return "black"
    s = (mx - mn) / float(mx)
    if s < 0.28:
        return "white" if mx > 120 else "black"
    d = float(mx - mn)
    if mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * ((b - r) / d + 2.0)
    else:
        h = 60.0 * ((r - g) / d + 4.0)
    if h < 15 or h >= 335:
        return "red"
    if h < 47:
        return "orange"
    if h < 75:
        return "yellow"
    if h < 170:
        return "green"
    if h < 265:
        return "blue"
    return "purple"


# ---------------------------------------------------------------- camera models
def cam_model(cam, conv, rng, inv):
    K = cam.intrinsics
    fx, fy = float(K[0][0]), float(K[1][1])
    cx, cy = float(K[0][2]), float(K[1][2])
    pose = cam.camera_pose
    R = quat_to_r(pose.quaternion_wxyz)
    t = [float(pose.position[0]), float(pose.position[1]), float(pose.position[2])]
    if inv:
        Rt = [[R[0][0], R[1][0], R[2][0]], [R[0][1], R[1][1], R[2][1]], [R[0][2], R[1][2], R[2][2]]]
        t = [-(Rt[i][0] * t[0] + Rt[i][1] * t[1] + Rt[i][2] * t[2]) for i in range(3)]
        R = Rt
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "R": R, "t": t, "conv": conv, "rng": rng,
            "pos": [float(pose.position[0]), float(pose.position[1]), float(pose.position[2])]}


def project(m, u, v, d):
    xn = (u + 0.5 - m["cx"]) / m["fx"]
    yn = (v + 0.5 - m["cy"]) / m["fy"]
    if m["rng"]:
        d = d / math.sqrt(1.0 + xn * xn + yn * yn)
    conv = m["conv"]
    if conv == 0:
        a, b, c = xn * d, yn * d, d
    elif conv == 1:
        a, b, c = xn * d, -yn * d, -d
    else:
        a, b, c = d, -xn * d, -yn * d
    R = m["R"]
    t = m["t"]
    return (R[0][0] * a + R[0][1] * b + R[0][2] * c + t[0],
            R[1][0] * a + R[1][1] * b + R[1][2] * c + t[1],
            R[2][0] * a + R[2][1] * b + R[2][2] * c + t[2])


def table_samples(cam, stride):
    rgb = to_list(cam.rgb)
    dep = to_list(cam.depth_m)
    out = []
    for v in range(0, len(rgb), stride):
        row = rgb[v]
        drow = dep[v]
        for u in range(0, len(row), stride):
            p = row[u]
            r, g, b = p[0], p[1], p[2]
            d = float(drow[u])
            if r > 90 and r > g + 25 and g > b + 10 and 0.0 < d < 10.0:
                out.append((u, v, d))
    return out


def spread(values):
    if len(values) < 10:
        return None, None
    lo = quantile(values, 0.1)
    hi = quantile(values, 0.9)
    return quantile(values, 0.5), hi - lo


def calibrate(obs):
    """Pick the camera-frame convention under which table pixels of BOTH cameras land on one
    horizontal plane.  Returns (conv, rng, inv, table_z, score) or None."""
    samples = {}
    for name in (FRONT, WRIST):
        if name in obs.cameras:
            samples[name] = table_samples(obs.cameras[name], 6)
    if FRONT not in samples or len(samples[FRONT]) < 20:
        return None
    best = None
    for conv in (0, 1, 2):
        for rng in (False, True):
            for inv in (False, True):
                score = 0.0
                meds = []
                okc = True
                for name in samples:
                    if len(samples[name]) < 20:
                        continue
                    m = cam_model(obs.cameras[name], conv, rng, inv)
                    zs = [project(m, u, v, d)[2] for (u, v, d) in samples[name]]
                    med, spr = spread(zs)
                    if med is None:
                        okc = False
                        break
                    score += spr
                    meds.append(med)
                if not okc or len(meds) == 0:
                    continue
                if len(meds) == 2:
                    score += abs(meds[0] - meds[1])
                if best is None or score < best[4]:
                    best = (conv, rng, inv, sum(meds) / len(meds), score)
    return best


# ---------------------------------------------------------------- pixel extraction
def is_container_white(r, g, b):
    if r <= 110 or g <= 110 or b <= 110:
        return False
    mx = r if r > g else g
    mx = mx if mx > b else b
    mn = r if r < g else g
    mn = mn if mn < b else b
    return (mx - mn) < 45 and (b - r) < 12


def white_points(cam, model, bg, stride, zlo, zhi, roi=None):
    rgb = to_list(cam.rgb)
    dep = to_list(cam.depth_m)
    hgt = len(rgb)
    wid = len(rgb[0])
    u0, v0, u1, v1 = 0, 0, wid, hgt
    if roi is not None:
        u0, v0, u1, v1 = max(0, roi[0]), max(0, roi[1]), min(wid, roi[2]), min(hgt, roi[3])
    pts = []
    fx, fy, cx, cy = model["fx"], model["fy"], model["cx"], model["cy"]
    for v in range(v0, v1, stride):
        row = rgb[v]
        drow = dep[v]
        brow = bg[v] if bg is not None else None
        for u in range(u0, u1, stride):
            p = row[u]
            r, g, b = p[0], p[1], p[2]
            if not is_container_white(r, g, b):
                continue
            if brow is not None:
                q = brow[u]
                if abs(r - q[0]) + abs(g - q[1]) + abs(b - q[2]) <= 60:
                    continue
            d = float(drow[u])
            if not (0.0 < d < 10.0):
                continue
            x, y, z = project(model, u, v, d)
            if zlo <= z <= zhi:
                pts.append((x, y, z))
    return pts


def colour_components(cam, model, bg, stride, table_z, min_samples):
    """Connected components (same hue class, 4-neighbour on the sampled grid) of pixels that
    differ from the background.  Returns [(class, x, y, z, n)] in robot base."""
    rgb = to_list(cam.rgb)
    dep = to_list(cam.depth_m)
    hgt = len(rgb)
    wid = len(rgb[0])
    grid = {}
    for v in range(0, hgt, stride):
        row = rgb[v]
        brow = bg[v]
        for u in range(0, wid, stride):
            p = row[u]
            q = brow[u]
            if abs(p[0] - q[0]) + abs(p[1] - q[1]) + abs(p[2] - q[2]) <= 60:
                continue
            grid[(u, v)] = hue_class(p[0], p[1], p[2])
    seen = set()
    comps = []
    for key in grid:
        if key in seen:
            continue
        cls = grid[key]
        stack = [key]
        seen.add(key)
        members = []
        while len(stack) > 0:
            cur = stack.pop()
            members.append(cur)
            for nb in ((cur[0] + stride, cur[1]), (cur[0] - stride, cur[1]),
                       (cur[0], cur[1] + stride), (cur[0], cur[1] - stride)):
                if nb in grid and nb not in seen and grid[nb] == cls:
                    seen.add(nb)
                    stack.append(nb)
        if len(members) < min_samples:
            continue
        xs, ys, zs = [], [], []
        for (u, v) in members:
            d = float(dep[v][u])
            if 0.0 < d < 10.0:
                x, y, z = project(model, u, v, d)
                xs.append(x)
                ys.append(y)
                zs.append(z)
        if len(xs) < max(2, min_samples // 2):
            continue
        zc = quantile(zs, 0.5)
        if zc < table_z - 0.02 or zc > table_z + 0.12:
            continue
        mxc = sum(xs) / len(xs)
        myc = sum(ys) / len(ys)
        rms = math.sqrt(sum((xs[i] - mxc) ** 2 + (ys[i] - myc) ** 2 for i in range(len(xs))) / len(xs))
        comps.append((cls, quantile(xs, 0.5), quantile(ys, 0.5), zc, len(members), 2.45 * rms))
    return comps


def white_count(cam, bg, stride):
    rgb = to_list(cam.rgb)
    n = 0
    for v in range(0, len(rgb), stride):
        row = rgb[v]
        brow = bg[v]
        for u in range(0, len(row), stride):
            p = row[u]
            if is_container_white(p[0], p[1], p[2]):
                q = brow[u]
                if abs(p[0] - q[0]) + abs(p[1] - q[1]) + abs(p[2] - q[2]) > 60:
                    n += 1
    return n


# ---------------------------------------------------------------- tracking
def kmeans_update(tracks, pts, gate, iters, min_pts):
    n = len(tracks)
    g2 = gate * gate
    counts = [0] * n
    for it in range(iters):
        sx = [0.0] * n
        sy = [0.0] * n
        counts = [0] * n
        cx = [t["x"] for t in tracks]
        cy = [t["y"] for t in tracks]
        for p in pts:
            px = p[0]
            py = p[1]
            best = -1
            bd = g2
            for k in range(n):
                dx = px - cx[k]
                dy = py - cy[k]
                dd = dx * dx + dy * dy
                if dd < bd:
                    bd = dd
                    best = k
            if best >= 0:
                sx[best] += px
                sy[best] += py
                counts[best] += 1
        for k in range(n):
            if counts[k] >= min_pts:
                nx = sx[k] / counts[k]
                ny = sy[k] / counts[k]
                dx = nx - tracks[k]["x"]
                dy = ny - tracks[k]["y"]
                step = math.sqrt(dx * dx + dy * dy)
                cap = 0.5 * gate
                if step > cap:
                    nx = tracks[k]["x"] + dx * cap / step
                    ny = tracks[k]["y"] + dy * cap / step
                tracks[k]["x"] = nx
                tracks[k]["y"] = ny
    return counts


def subsample(pts, limit):
    if len(pts) <= limit:
        return pts
    step = len(pts) // limit + 1
    return pts[::step]


def frame_points(obs, models, bgs, strides, zlo, zhi, rois):
    pts = []
    for name in models:
        if name not in obs.cameras:
            continue
        cam = obs.cameras[name]
        bg = bgs.get(name)
        pts.extend(white_points(cam, models[name](cam), bg, strides[name], zlo, zhi, rois.get(name)))
    return pts


# ---------------------------------------------------------------- robot helpers
def is_terminal(res):
    flags = []
    for attr in ("terminated", "truncated", "done"):
        try:
            if attr == "terminated":
                flags.append(bool(res.terminated))
            elif attr == "truncated":
                flags.append(bool(res.truncated))
            else:
                flags.append(bool(res.done))
        except Exception:
            pass
    return any(flags)


def hold_tick():
    st = get_robot_state()
    res = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=[float(a) for a in st.joint_positions["primary"]],
        gripper_position=float(st.gripper_positions["primary"]), embodiment=st.embodiment)}))
    return res


def ok_of(res):
    try:
        return bool(res.ok)
    except Exception:
        return False


# ================================================================ main program
ctx = get_task_context()
goal = ctx.language
report["goal"] = goal
targets = parse_goal(goal)
report["targets"] = targets
meta = robomme_plus.get_task_metadata()
max_steps = None
control_hz = 20.0
try:
    max_steps = int(meta["max_steps"])
except Exception:
    note("max_steps unavailable")
try:
    control_hz = float(meta["control_hz"])
except Exception:
    try:
        control_hz = float(meta["control_frequency"])
    except Exception:
        pass
report["max_steps"] = max_steps
state0 = get_robot_state()
base_frame = state0.base_frame
video = robomme_plus.get_conditioning_video()
report["video_frames"] = len(video)
live0 = get_observation()

calib_src = video[0] if len(video) > 0 else live0
cal = calibrate(calib_src)
report["calibration"] = None if cal is None else [cal[0], cal[1], cal[2], round(cal[3], 4), round(cal[4], 4)]
if cal is None or cal[4] > 0.06:
    note("projection self-calibration failed; tracking quality unknown")
CONV, RNG, INV = (0, False, False) if cal is None else (cal[0], cal[1], cal[2])
table_z = live0.robot_state.end_effector_poses["primary"].position[2] - 0.3 if cal is None else cal[3]


def model_of(cam):
    return cam_model(cam, CONV, RNG, INV)


def same_pose(cam_a, cam_b):
    pa = cam_a.camera_pose.position
    pb = cam_b.camera_pose.position
    qa = cam_a.camera_pose.quaternion_wxyz
    qb = cam_b.camera_pose.quaternion_wxyz
    dp = sum(abs(float(pa[i]) - float(pb[i])) for i in range(3))
    dq = min(sum(abs(float(qa[i]) - float(qb[i])) for i in range(4)),
             sum(abs(float(qa[i]) + float(qb[i])) for i in range(4)))
    return dp < 0.005 and dq < 0.01


tracks = []
cont_size = 0.05
cont_h = 0.05
models = {FRONT: model_of, WRIST: model_of}
strides = {FRONT: 2, WRIST: 3}

if len(video) >= 3:
    bg_obs = video[0]
    bgs_video = {}
    for name in (FRONT, WRIST):
        if name in bg_obs.cameras:
            bgs_video[name] = to_list(bg_obs.cameras[name].rgb)
    # ---- find the drop frame: first frame whose white foreground jumps above everything before
    counts = []
    drop = None
    running_max = 0
    for idx in range(1, len(video)):
        c = white_count(video[idx].cameras[FRONT], bgs_video[FRONT], 3)
        counts.append(c)
        if idx >= 2 and c > 3 * running_max + 30:
            drop = idx
            break
        running_max = max(running_max, c)
    report["drop_frame"] = drop
    if drop is None or drop < 2:
        note("container drop not found in the video")
    else:
        # ---- cubes in the last frame before the drop (both cameras)
        cube_obs = video[drop - 1]
        comps = []
        for name in (FRONT, WRIST):
            if name not in cube_obs.cameras or name not in bgs_video:
                continue
            cam = cube_obs.cameras[name]
            if name == WRIST and not same_pose(cam, bg_obs.cameras[WRIST]):
                continue
            comps.extend([(name,) + c for c in colour_components(
                cam, model_of(cam), bgs_video[name], 2, table_z, 3 if name == FRONT else 6)])
        cubes = []
        for c in sorted(comps, key=lambda e: -e[5]):
            merged = False
            for cube in cubes:
                # two detections of one cube (front vs wrist centroid bias) lie within ~one cube
                # width of each other; two real cubes of one colour sit under two containers
                radius = max(0.02, 1.5 * max(cube["w"], c[6]))
                if cube["label"] == c[1] and math.hypot(cube["x"] - c[2], cube["y"] - c[3]) < radius:
                    merged = True
                    cube["support"] += c[5]
                    break
            if not merged:
                cubes.append({"label": c[1], "x": c[2], "y": c[3], "src": c[0], "w": c[6], "support": c[5]})
        report["cubes"] = [[c["label"], round(c["x"], 3), round(c["y"], 3), c["src"]] for c in cubes]
        tracks = [{"label": c["label"], "x": c["x"], "y": c["y"], "support": c["support"]} for c in cubes]
        if len(tracks) == 0:
            note("no cubes detected before the drop")
        # ---- container size/height from SETTLED frames: the drop frame can still show the
        # containers falling, so take the per-frame top over ~2 s after the drop and use a low
        # quantile across frames (falling frames are higher, settled frames agree)
        t_drop = float(video[drop].timestamp_s)
        tops = []
        for idx in range(drop, len(video)):
            if float(video[idx].timestamp_s) > t_drop + 2.0:
                break
            fo = video[idx]
            ptsf = []
            for name in (FRONT, WRIST):
                if name in fo.cameras and name in bgs_video:
                    cam = fo.cameras[name]
                    bgw = bgs_video[name] if same_pose(cam, bg_obs.cameras[name]) else None
                    ptsf.extend(white_points(cam, model_of(cam), bgw, 3, table_z + 0.004, table_z + 0.3))
            if len(ptsf) > 30:
                tops.append((quantile([p[2] for p in ptsf], 0.95), idx, ptsf))
        pts0 = []
        top = None
        if len(tops) > 0:
            ordered = sorted(tops, key=lambda e: e[0])
            pick = ordered[len(ordered) // 4]
            top = pick[0] - table_z
            pts0 = pick[2]
            report["settled_frame"] = pick[1]
        if len(pts0) > 30 and len(tracks) > 0:
            if 0.01 < top < 0.25:
                cont_h = top
            spreads = []
            tmp = [{"x": t["x"], "y": t["y"]} for t in tracks]
            kmeans_update(tmp, subsample(pts0, 1500), 0.06, 3, 5)
            for t in tmp:
                mine = [p for p in pts0 if math.hypot(p[0] - t["x"], p[1] - t["y"]) < 0.06]
                own = []
                for p in mine:
                    dbest = None
                    for o in tmp:
                        dd = math.hypot(p[0] - o["x"], p[1] - o["y"])
                        if dbest is None or dd < dbest:
                            dbest = dd
                    if abs(dbest - math.hypot(p[0] - t["x"], p[1] - t["y"])) < 1e-9:
                        own.append(p)
                if len(own) >= 10:
                    mx = sum(p[0] for p in own) / len(own)
                    my = sum(p[1] for p in own) / len(own)
                    var = sum((p[0] - mx) ** 2 + (p[1] - my) ** 2 for p in own) / len(own)
                    spreads.append(2.45 * math.sqrt(var))
            if len(spreads) > 0:
                s = quantile(spreads, 0.5)
                if 0.015 < s < 0.15:
                    cont_size = s
        report["container_size_m"] = round(cont_size, 4)
        report["container_height_m"] = round(cont_h, 4)
        gate = max(0.015, min(0.06, 0.75 * cont_size))
        zlo = table_z + 0.45 * cont_h
        zhi = table_z + 1.15 * cont_h + 0.005
        report["gate_m"] = round(gate, 4)
        # ---- track through the video
        for idx in range(drop, len(video)):
            fo = video[idx]
            pts = []
            for name in (FRONT, WRIST):
                if name not in fo.cameras:
                    continue
                cam = fo.cameras[name]
                bgw = None
                if name in bgs_video and same_pose(cam, bg_obs.cameras[name]):
                    bgw = bgs_video[name]
                pts.extend(white_points(cam, model_of(cam), bgw, strides[name], zlo, zhi))
            if len(tracks) > 0 and len(pts) > 0:
                kmeans_update(tracks, subsample(pts, 700), gate, 2, 5)
        report["tracks_end_of_video"] = [[t["label"], round(t["x"], 3), round(t["y"], 3)] for t in tracks]
else:
    note("no conditioning video: container identities unknown")

gate = max(0.015, min(0.06, 0.75 * cont_size))
zlo = table_z + 0.45 * cont_h
zhi = table_z + 1.15 * cont_h + 0.005

# ---- live tracking while the shuffle finishes (hold the arm one tick at a time)
terminal = False
live_ticks = 0
still = 0
cap_ticks = 300 if max_steps is None else max(40, min(400, max_steps // 3))
bg_front_live = None
if len(video) > 0 and FRONT in video[0].cameras and same_pose(live0.cameras[FRONT], video[0].cameras[FRONT]):
    bg_front_live = to_list(video[0].cameras[FRONT].rgb)
obs = live0
if len(tracks) > 0:
    for tick in range(cap_ticks):
        prev = [(t["x"], t["y"]) for t in tracks]
        pts = []
        for name in (FRONT, WRIST):
            if name in obs.cameras:
                cam = obs.cameras[name]
                bgw = bg_front_live if name == FRONT else None
                pts.extend(white_points(cam, model_of(cam), bgw, strides[name], zlo, zhi))
        if len(pts) > 0:
            kmeans_update(tracks, subsample(pts, 900), gate, 2, 5)
        moved = max(math.hypot(tracks[k]["x"] - prev[k][0], tracks[k]["y"] - prev[k][1]) for k in range(len(tracks)))
        still = still + 1 if (moved < 0.002 and tick > 0) else 0
        if still >= 8:
            break
        res = hold_tick()
        live_ticks += 1
        if is_terminal(res):
            terminal = True
            break
        obs = get_observation()
report["live_tracking_ticks"] = live_ticks
report["tracks_final"] = [[t["label"], round(t["x"], 3), round(t["y"], 3)] for t in tracks]


# ---------------------------------------------------------------- picking
def measure(expect_x, expect_y, others):
    """Re-localize one container near (expect_x, expect_y) from the current cameras.
    Returns dict(x, y, top, yaw, n) or None."""
    ob = get_observation()
    pts = []
    for name in (WRIST, FRONT):
        if name in ob.cameras:
            cam = ob.cameras[name]
            bgw = bg_front_live if (name == FRONT and bg_front_live is not None
                                    and same_pose(cam, video[0].cameras[FRONT])) else None
            pts.extend(white_points(cam, model_of(cam), bgw, 2, table_z + 0.3 * cont_h,
                                    table_z + 1.3 * cont_h + 0.01))
    # two-pass mean shift on top-face points: a container's own top face lies within ~0.71 of its
    # width from its centre; points of an untracked neighbour start beyond ~0.5 widths
    cxm, cym = expect_x, expect_y
    mine = []
    for radius in (0.85, 0.62, 0.62):
        mine = []
        for p in pts:
            d0 = math.hypot(p[0] - cxm, p[1] - cym)
            if d0 > radius * cont_size:
                continue
            closer = False
            for o in others:
                if math.hypot(p[0] - o[0], p[1] - o[1]) < d0:
                    closer = True
                    break
            if not closer:
                mine.append(p)
        if len(mine) < 8:
            return None
        ztop = quantile([p[2] for p in mine], 0.9)
        face = [p for p in mine if p[2] >= ztop - 0.25 * cont_h]
        if len(face) < 8:
            face = mine
        cxm = quantile([p[0] for p in face], 0.5)
        cym = quantile([p[1] for p in face], 0.5)
    top = quantile([p[2] for p in mine], 0.9)
    topp = [p for p in mine if p[2] >= top - 0.35 * cont_h]
    if len(topp) < 8:
        topp = mine
    best = None
    for k in range(18):
        a = k * math.pi / 36.0
        ca, sa = math.cos(a), math.sin(a)
        us = [p[0] * ca + p[1] * sa for p in topp]
        vs = [-p[0] * sa + p[1] * ca for p in topp]
        u_lo, u_hi = quantile(us, 0.03), quantile(us, 0.97)
        v_lo, v_hi = quantile(vs, 0.03), quantile(vs, 0.97)
        area = (u_hi - u_lo) * (v_hi - v_lo)
        if best is None or area < best[0]:
            uc, vc = 0.5 * (u_lo + u_hi), 0.5 * (v_lo + v_hi)
            best = (area, a, uc * ca - vc * sa, uc * sa + vc * ca)
    return {"x": best[2], "y": best[3], "top": top, "yaw": best[1], "n": len(mine)}


q_start = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]
Rs = quat_to_r(q_start)
if Rs[2][2] < -0.9:
    q_down = q_start
else:
    q_down = [0.0, 1.0, 0.0, 0.0]
    note("start EE not pointing down; using canonical downward quaternion")
Rd = quat_to_r(q_down)
finger_angle0 = math.atan2(Rd[1][1], Rd[0][1])   # world angle of the hand's local y (finger) axis


def grasp_quat(face_angle, neighbours, cx, cy):
    best = None
    for cand in (face_angle, face_angle + math.pi / 2):
        delta = wrap_half_pi(cand - finger_angle0)
        ax = math.cos(finger_angle0 + delta)
        ay = math.sin(finger_angle0 + delta)
        clear = 1.0
        for sgn in (1.0, -1.0):
            fxp = cx + sgn * ax * (0.5 * cont_size + 0.015)
            fyp = cy + sgn * ay * (0.5 * cont_size + 0.015)
            for o in neighbours:
                clear = min(clear, math.hypot(fxp - o[0], fyp - o[1]) - 0.5 * cont_size)
        key = (round(clear, 3), -abs(delta))
        if best is None or key > best[0]:
            best = (key, delta)
    delta = best[1]
    qz = [math.cos(delta / 2), 0.0, 0.0, math.sin(delta / 2)]
    return qmul(qz, q_down), delta


def go(x, y, z, quat, steps=150, tol=0.01):
    res = move_to_pose(Pose([x, y, z], quat, base_frame), tolerance=tol, max_steps=steps)
    return res


def elapsed_ticks():
    try:
        return int(round(float(get_robot_state().timestamp_s) * control_hz))
    except Exception:
        return None


def revealed_colour(cx, cy):
    ob = get_observation()
    votes = {}
    for name in (FRONT, WRIST):
        if name not in ob.cameras:
            continue
        cam = ob.cameras[name]
        m = model_of(cam)
        rgb = to_list(cam.rgb)
        dep = to_list(cam.depth_m)
        for v in range(0, len(rgb), 2):
            row = rgb[v]
            drow = dep[v]
            for u in range(0, len(row), 2):
                p = row[u]
                cls = hue_class(p[0], p[1], p[2])
                if cls in ("white", "black"):
                    continue
                if p[0] > 90 and p[0] > p[1] + 25 and p[1] > p[2] + 10 and (max(p) - min(p)) / float(max(p)) < 0.75:
                    continue   # table-like orange
                d = float(drow[u])
                if not (0.0 < d < 10.0):
                    continue
                x, y, z = project(m, u, v, d)
                if math.hypot(x - cx, y - cy) < 0.6 * cont_size and z < table_z + 0.8 * cont_h:
                    votes[cls] = votes.get(cls, 0) + 1
    best = None
    for k in votes:
        if best is None or votes[k] > votes[best]:
            best = k
    return best, votes


picks = []
picked_ids = set()
if not terminal and len(tracks) > 0:
    open_gripper()
    for item in targets:
        if terminal:
            break
        cands = [k for k in range(len(tracks)) if tracks[k]["label"] == item["colour"] and k not in picked_ids]
        if len(cands) == 0:
            picks.append({"colour": item["colour"], "status": "no track with this colour"})
            continue
        if not item["all"] and len(cands) > 1:
            # a singular target: keep the best-supported detection of that colour
            cands = [max(cands, key=lambda j: tracks[j]["support"])]
            note("several tracks labelled %s for a singular target; using best supported" % item["colour"])
        order = []
        ee = get_robot_state().end_effector_poses["primary"].position
        remaining = list(cands)
        px, py = float(ee[0]), float(ee[1])
        while len(remaining) > 0:
            kbest = None
            for k in remaining:
                dd = math.hypot(tracks[k]["x"] - px, tracks[k]["y"] - py)
                if kbest is None or dd < kbest[0]:
                    kbest = (dd, k)
            order.append(kbest[1])
            remaining.remove(kbest[1])
            px, py = tracks[kbest[1]]["x"], tracks[kbest[1]]["y"]
        if not item["all"]:
            order = order[:1]
        for k in order:
            if terminal:
                break
            rec = {"colour": item["colour"], "track": k, "start_tick": elapsed_ticks()}
            others = [(tracks[j]["x"], tracks[j]["y"]) for j in range(len(tracks)) if j != k]
            tx, ty = tracks[k]["x"], tracks[k]["y"]
            top = table_z + cont_h
            meas = measure(tx, ty, others)
            yaw = 0.0
            if meas is not None:
                tx, ty, top, yaw = meas["x"], meas["y"], meas["top"], meas["yaw"]
            quat, delta = grasp_quat(yaw, others, tx, ty)
            hover = top + 0.10
            r1 = go(tx, ty, hover, quat, 200, 0.015)
            terminal = terminal or is_terminal(r1)
            if terminal:
                rec["status"] = "terminal during transit"
                picks.append(rec)
                break
            meas2 = measure(tx, ty, others)
            if meas2 is not None and math.hypot(meas2["x"] - tx, meas2["y"] - ty) < 0.5 * cont_size:
                tx, ty, top = meas2["x"], meas2["y"], meas2["top"]
                quat, delta = grasp_quat(meas2["yaw"], others, tx, ty)
                hover = top + 0.10
                r1b = go(tx, ty, hover, quat, 120, 0.01)
                terminal = terminal or is_terminal(r1b)
            rec["refined"] = meas2 is not None
            grip_depth = max(0.01, min(0.03, 0.45 * cont_h))
            gz = max(table_z + 0.008, top - grip_depth)
            if not terminal:
                r2 = go(tx, ty, top + 0.03, quat, 100, 0.01)
                terminal = terminal or is_terminal(r2)
            if not terminal:
                r3 = go(tx, ty, gz, quat, 80, 0.008)
                terminal = terminal or is_terminal(r3)
            if not terminal:
                rc = close_gripper()
                terminal = terminal or is_terminal(rc)
            gpos = float(get_robot_state().gripper_positions["primary"])
            rec["gripper_after_close"] = round(gpos, 3)
            if not terminal and gpos < 0.03:
                # closed on nothing: open, back to hover, re-measure, one more attempt
                rec["regrasp"] = True
                ro = open_gripper()
                terminal = terminal or is_terminal(ro)
                if not terminal:
                    rh = go(tx, ty, hover, quat, 120, 0.015)
                    terminal = terminal or is_terminal(rh)
                meas3 = None if terminal else measure(tx, ty, others)
                if meas3 is not None and math.hypot(meas3["x"] - tx, meas3["y"] - ty) < 0.6 * cont_size:
                    tx, ty, top = meas3["x"], meas3["y"], meas3["top"]
                    quat, delta = grasp_quat(meas3["yaw"], others, tx, ty)
                    gz = max(table_z + 0.008, top - grip_depth)
                if not terminal:
                    r2b = go(tx, ty, top + 0.03, quat, 120, 0.01)
                    terminal = terminal or is_terminal(r2b)
                if not terminal:
                    r3b = go(tx, ty, gz, quat, 80, 0.008)
                    terminal = terminal or is_terminal(r3b)
                if not terminal:
                    rc2 = close_gripper()
                    terminal = terminal or is_terminal(rc2)
                gpos = float(get_robot_state().gripper_positions["primary"])
                rec["gripper_after_regrasp"] = round(gpos, 3)
            lift_z = gz + cont_h + 0.05
            if not terminal:
                r4 = go(tx, ty, lift_z, quat, 100, 0.015)
                terminal = terminal or is_terminal(r4)
            if not terminal:
                col, votes = revealed_colour(tx, ty)
                rec["revealed"] = col
                rec["revealed_votes"] = votes
            if not terminal:
                r5 = go(tx, ty, gz + 0.008, quat, 100, 0.01)
                terminal = terminal or is_terminal(r5)
            if not terminal:
                r6 = open_gripper()
                terminal = terminal or is_terminal(r6)
            if not terminal:
                r7 = go(tx, ty, hover, quat, 100, 0.02)
                terminal = terminal or is_terminal(r7)
            tracks[k]["x"], tracks[k]["y"] = tx, ty
            picked_ids.add(k)
            rec["end_tick"] = elapsed_ticks()
            rec["status"] = "terminal" if terminal else "done"
            picks.append(rec)

report["picks"] = picks
report["terminal"] = terminal
result = report
