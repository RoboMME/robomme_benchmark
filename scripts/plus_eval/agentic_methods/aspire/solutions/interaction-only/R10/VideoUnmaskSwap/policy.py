import math

# v03 - v02 + merge duplicate cube detections within 0.035 m (one cube split by an occluder
#       produced two seeds -> two tracks on one container). Tracks that receive no points hold
#       position (a constant-velocity variant was tried offline on an occluded replay and did not
#       help, so it is disabled); per-track lost-frame counts are reported.
# v02 - v01 + (a) tall-object exclusion: points whose XY footprint also has points 0.11-0.17 m
#       above the table (a standing pillar/obstacle) are not containers or cubes, and do not
#       trigger "containers appeared"; (b) front + wrist camera fusion for cubes, tracking and
#       top-face refinement (the front view can be occluded by a tall object); (c) containers
#       with no cube under them at appearance get an "unknown" track; (d) raise the approach
#       height near tall obstacles; (e) compact visual diagnostics of the video.
# All positions come from the current episode's depth images; nothing is copied from a run.

report = {"version": "v03", "errors": [], "motions": []}
result = report

COLOR_WORDS = ("red", "green", "blue", "yellow", "orange", "purple", "cyan", "white", "pink", "black", "brown", "gray", "grey")
TOP_LO = 0.045
TOP_HI = 0.11
TALL_LO = 0.11
TALL_HI = 0.17
CUBE_LO = 0.012
CUBE_HI = 0.042
X_MIN = 0.2
GATE = 0.06
CELL = 0.02
ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def enc_rgb(rgb, stride):
    rows = to_rows(rgb, stride)
    parts = []
    for row in rows:
        for pixel in row:
            parts.append(ALPH[int(pixel[0]) // 4])
            parts.append(ALPH[int(pixel[1]) // 4])
            parts.append(ALPH[int(pixel[2]) // 4])
    return "".join(parts)


def to_rows(array, stride):
    try:
        return array[::stride, ::stride].tolist()
    except Exception:
        rows = []
        for y in range(0, len(array), stride):
            row = []
            for x in range(0, len(array[0]), stride):
                row.append(array[y][x])
            rows.append(row)
        return rows


def mat_rows(m):
    try:
        return m.tolist()
    except Exception:
        return [[m[i][j] for j in range(3)] for i in range(3)]


def qmat(q):
    w = float(q[0]); x = float(q[1]); y = float(q[2]); z = float(q[3])
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def qmul(a, b):
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]


def cam_model(cam):
    intr = mat_rows(cam.intrinsics)
    pos = [float(v) for v in cam.camera_pose.position]
    quat = [float(v) for v in cam.camera_pose.quaternion_wxyz]
    return {"fx": float(intr[0][0]), "fy": float(intr[1][1]), "cx": float(intr[0][2]), "cy": float(intr[1][2]),
            "t": pos, "R": qmat(quat), "h": len(cam.rgb), "w": len(cam.rgb[0])}


def ray_table(model, stride):
    R = model["R"]
    rows = []
    for v in range(0, model["h"], stride):
        row = []
        for u in range(0, model["w"], stride):
            a = (u - model["cx"]) / model["fx"]
            b = (v - model["cy"]) / model["fy"]
            row.append((R[0][0] * a + R[0][1] * b + R[0][2],
                        R[1][0] * a + R[1][1] * b + R[1][2],
                        R[2][0] * a + R[2][1] * b + R[2][2]))
        rows.append(row)
    return {"rows": rows, "t": model["t"]}


def get_rays(cache, name, cam, stride):
    key = name + "/" + str(stride)
    pos = [float(v) for v in cam.camera_pose.position]
    quat = [float(v) for v in cam.camera_pose.quaternion_wxyz]
    entry = cache.get(key)
    if entry is not None:
        same = True
        for i in range(3):
            if abs(entry["pos"][i] - pos[i]) > 0.0005:
                same = False
        for i in range(4):
            if abs(entry["quat"][i] - quat[i]) > 0.0005:
                same = False
        if same:
            return entry["rays"]
    model = cam_model(cam)
    rays = ray_table(model, stride)
    cache[key] = {"pos": pos, "quat": quat, "rays": rays, "model": model}
    return rays


def scan(depth_rows, rays, rgb_rows=None):
    """One pass: top-band points, tall points (footprint of standing obstacles), cube-band points."""
    tx = rays["t"][0]; ty = rays["t"][1]; tz = rays["t"][2]
    top = []
    tall = []
    low = []
    iy = 0
    for drow in depth_rows:
        rrow = rays["rows"][iy]
        ix = 0
        for d in drow:
            if d > 0.05 and d < 4.0:
                k = rrow[ix]
                z = tz + d * k[2]
                if z > CUBE_LO and z < TALL_HI:
                    x = tx + d * k[0]
                    if x > X_MIN:
                        y = ty + d * k[1]
                        if z > TOP_LO and z < TOP_HI:
                            top.append([x, y, z])
                        if z > TALL_LO:
                            tall.append([x, y, z])
                        if rgb_rows is not None and z < CUBE_HI:
                            low.append([x, y, z, rgb_rows[iy][ix]])
            ix += 1
        iy += 1
    return top, tall, low


def tall_cells(tall):
    cells = {}
    for p in tall:
        key = (int(math.floor(p[0] / CELL)), int(math.floor(p[1] / CELL)))
        cells[key] = cells.get(key, 0) + 1
    keep = {}
    for key in cells:
        if cells[key] >= 2:
            keep[key] = True
    return keep


def exclude(points, cells):
    if not cells:
        return points
    out = []
    for p in points:
        cx = int(math.floor(p[0] / CELL)); cy = int(math.floor(p[1] / CELL))
        hit = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (cx + dx, cy + dy) in cells:
                    hit = True
        if not hit:
            out.append(p)
    return out


def fused_points(frame, cache, stride, want_low=False):
    tops = []
    talls = []
    lows = []
    for name in ("front", "wrist"):
        try:
            cam = frame.cameras[name]
        except Exception:
            continue
        rays = get_rays(cache, name, cam, stride)
        rgb_rows = to_rows(cam.rgb, stride) if want_low else None
        top, tall, low = scan(to_rows(cam.depth_m, stride), rays, rgb_rows)
        tops.extend(top)
        talls.extend(tall)
        lows.extend(low)
    cells = tall_cells(talls)
    return exclude(tops, cells), exclude(lows, cells), cells, len(talls)


def cluster_xy(points, radius):
    clusters = []
    used = [False] * len(points)
    r2 = radius * radius
    for i in range(len(points)):
        if used[i]:
            continue
        used[i] = True
        members = [points[i]]
        frontier = [i]
        while frontier:
            j = frontier.pop()
            pj = points[j]
            for k in range(len(points)):
                if not used[k]:
                    dx = points[k][0] - pj[0]
                    dy = points[k][1] - pj[1]
                    if dx * dx + dy * dy < r2:
                        used[k] = True
                        members.append(points[k])
                        frontier.append(k)
        clusters.append(members)
    return clusters


def classify_color(rgb_list):
    if not rgb_list:
        return "unknown"
    r = sum(float(p[0]) for p in rgb_list) / len(rgb_list)
    g = sum(float(p[1]) for p in rgb_list) / len(rgb_list)
    b = sum(float(p[2]) for p in rgb_list) / len(rgb_list)
    if max(r, g, b) < 40:
        return "unknown"
    if r > 1.8 * b and g > 1.8 * b:
        ratio = g / max(r, 1.0)
        if ratio > 0.85 and ratio < 1.18:
            return "yellow"
        if ratio <= 0.85 and ratio > 0.4:
            return "orange"
    if r > 2.0 * g and r > 2.0 * b:
        return "red"
    if g > 1.8 * r and g > 1.8 * b:
        return "green"
    if b > 1.8 * r and b > 1.5 * g:
        return "blue"
    if r > 1.8 * g and b > 1.8 * g:
        return "purple"
    if g > 1.8 * r and b > 1.8 * r:
        return "cyan"
    if r > 150 and g > 150 and b > 150:
        return "white"
    return "unknown"


def track_step(tracks, points, gate):
    sums = [[0.0, 0.0, 0, 0.0] for t in tracks]
    g2 = gate * gate
    for p in points:
        best = -1
        bestd = g2
        for ti in range(len(tracks)):
            dx = p[0] - tracks[ti]["x"]
            dy = p[1] - tracks[ti]["y"]
            dd = dx * dx + dy * dy
            if dd < bestd:
                bestd = dd
                best = ti
        if best >= 0:
            sums[best][0] += p[0]
            sums[best][1] += p[1]
            sums[best][2] += 1
            sums[best][3] += p[2]
    for ti in range(len(tracks)):
        tracks[ti]["n"] = sums[ti][2]
        if sums[ti][2] >= 2:
            tracks[ti]["x"] = sums[ti][0] / sums[ti][2]
            tracks[ti]["y"] = sums[ti][1] / sums[ti][2]
            tracks[ti]["z"] = sums[ti][3] / sums[ti][2]


def track_frame(tracks, points, dt=0.05):
    before = [(t["x"], t["y"]) for t in tracks]
    for it in range(3):
        track_step(tracks, points, GATE)
    moved = 0.0
    for ti in range(len(tracks)):
        t = tracks[ti]
        vx = t.get("vx", 0.0); vy = t.get("vy", 0.0)
        if t.get("n", 0) >= 2:
            if dt > 0.0:
                t["vx"] = 0.5 * vx + 0.5 * (t["x"] - before[ti][0]) / dt
                t["vy"] = 0.5 * vy + 0.5 * (t["y"] - before[ti][1]) / dt
            t["lost"] = 0
        else:
            t["x"] = before[ti][0]
            t["y"] = before[ti][1]
            t["lost"] = t.get("lost", 0) + 1
            if t["lost"] > t.get("maxlost", 0):
                t["maxlost"] = t["lost"]
        dd = math.sqrt((t["x"] - before[ti][0]) ** 2 + (t["y"] - before[ti][1]) ** 2)
        if dd > t.get("maxstep", 0.0):
            t["maxstep"] = dd
        if dd > moved:
            moved = dd
    return moved


def min_separation(tracks):
    best = 9.0
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            dd = math.sqrt((tracks[i]["x"] - tracks[j]["x"]) ** 2 + (tracks[i]["y"] - tracks[j]["y"]) ** 2)
            if dd < best:
                best = dd
    return best


def snapshot(tracks):
    return [[t["color"], round(t["x"], 4), round(t["y"], 4), t.get("n", 0), t.get("maxlost", 0)] for t in tracks]


def detect_cubes(frame, cache):
    top, low, cells, ntall = fused_points(frame, cache, 2, True)
    groups = []
    for c in cluster_xy(low, 0.012):
        if len(c) < 4:
            continue
        groups.append({"pts": c, "x": sum(p[0] for p in c) / len(c), "y": sum(p[1] for p in c) / len(c)})
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if (groups[i]["x"] - groups[j]["x"]) ** 2 + (groups[i]["y"] - groups[j]["y"]) ** 2 < 0.035 * 0.035:
                    pts = groups[i]["pts"] + groups[j]["pts"]
                    groups[i] = {"pts": pts, "x": sum(p[0] for p in pts) / len(pts), "y": sum(p[1] for p in pts) / len(pts)}
                    groups.pop(j)
                    merged = True
                    break
            if merged:
                break
    cubes = []
    for g in groups:
        cubes.append({"x": g["x"], "y": g["y"], "color": classify_color([p[3] for p in g["pts"]]), "npix": len(g["pts"])})
    return cubes


def parse_goal(language):
    words = []
    for raw in language.lower().replace(",", " ").replace(".", " ").replace(";", " ").split():
        words.append(raw)
    order = []
    for i in range(len(words)):
        if words[i] in COLOR_WORDS:
            plural = False
            if i + 1 < len(words) and words[i + 1] == "cubes":
                plural = True
            for j in range(max(0, i - 4), i):
                if words[j] == "all":
                    plural = True
            order.append([words[i], plural])
    return order


def refine_container(model, depth, tx, ty, tz_hint, others):
    """Top face of the container nearest (tx, ty): centre, top z, yaw (deg) from full-res depth."""
    R = model["R"]; t = model["t"]
    px = tx - t[0]; py = ty - t[1]; pz = tz_hint - t[2]
    xc = R[0][0] * px + R[1][0] * py + R[2][0] * pz
    yc = R[0][1] * px + R[1][1] * py + R[2][1] * pz
    zc = R[0][2] * px + R[1][2] * py + R[2][2] * pz
    if zc <= 0.01:
        return None
    u0 = int(model["fx"] * xc / zc + model["cx"])
    v0 = int(model["fy"] * yc / zc + model["cy"])
    half = 22
    ua = max(0, u0 - half); ub = min(model["w"], u0 + half)
    va = max(0, v0 - half); vb = min(model["h"], v0 + half)
    if ua >= ub or va >= vb:
        return None
    try:
        window = depth[va:vb, ua:ub].tolist()
    except Exception:
        window = [[depth[v][u] for u in range(ua, ub)] for v in range(va, vb)]
    pts = []
    for iv in range(len(window)):
        v = va + iv
        b = (v - model["cy"]) / model["fy"]
        for iu in range(len(window[iv])):
            d = window[iv][iu]
            if not (d > 0.05 and d < 4.0):
                continue
            a = (ua + iu - model["cx"]) / model["fx"]
            z = t[2] + d * (R[2][0] * a + R[2][1] * b + R[2][2])
            if z < 0.04 or z > TOP_HI:
                continue
            x = t[0] + d * (R[0][0] * a + R[0][1] * b + R[0][2])
            y = t[1] + d * (R[1][0] * a + R[1][1] * b + R[1][2])
            d0 = (x - tx) ** 2 + (y - ty) ** 2
            if d0 < 0.035 * 0.035:
                nearest = True
                for o in others:
                    if (x - o[0]) ** 2 + (y - o[1]) ** 2 < d0:
                        nearest = False
                if nearest:
                    pts.append([x, y, z])
    if len(pts) < 8:
        return None
    zs = sorted(p[2] for p in pts)
    top_z = zs[int(0.9 * (len(zs) - 1))]
    face = [p for p in pts if p[2] > top_z - 0.006]
    if len(face) < 5:
        face = pts
    cx = sum(p[0] for p in face) / len(face)
    cy = sum(p[1] for p in face) / len(face)
    best_yaw = 0.0
    best_area = None
    for deg in range(0, 90, 5):
        c = math.cos(math.radians(deg)); s = math.sin(math.radians(deg))
        amin = 9.0; amax = -9.0; bmin = 9.0; bmax = -9.0
        for p in face:
            qa = (p[0] - cx) * c + (p[1] - cy) * s
            qb = -(p[0] - cx) * s + (p[1] - cy) * c
            amin = min(amin, qa); amax = max(amax, qa); bmin = min(bmin, qb); bmax = max(bmax, qb)
        area = (amax - amin) * (bmax - bmin)
        if best_area is None or area < best_area:
            best_area = area
            best_yaw = float(deg)
            best_ext = [amax - amin, bmax - bmin]
    if best_yaw > 45.0:
        best_yaw -= 90.0
    return {"x": cx, "y": cy, "top_z": top_z, "yaw_deg": best_yaw, "npts": len(pts), "nface": len(face),
            "ext": [round(best_ext[0], 4), round(best_ext[1], 4)]}


def ticks_now():
    try:
        return int(round(float(get_robot_state().timestamp_s) * 20.0))
    except Exception:
        return -1


def motion(label, res):
    entry = [label, bool(res.ok), str(res.status), int(res.steps_executed), ticks_now()]
    try:
        if res.terminated or res.truncated:
            entry.append("TERMINAL")
    except Exception:
        pass
    report["motions"].append(entry)
    try:
        return bool(res.terminated or res.truncated)
    except Exception:
        return False


ended = False

# ---------------- context ----------------
context = get_task_context()
language = context.language
report["language"] = language
goal = parse_goal(language)
report["goal_order"] = goal
max_steps = 1370
try:
    meta = robomme_plus.get_task_metadata()
    max_steps = int(meta["max_steps"])
except Exception as exc:
    report["errors"].append("meta: " + str(exc))
report["max_steps"] = max_steps

# ---------------- conditioning video tracking ----------------
tracks = None
cubes = []
video_snaps = []
thumbs = {}
vcache = {}
try:
    video = robomme_plus.get_conditioning_video()
    n = len(video)
    report["video_len"] = n
    if n > 0:
        try:
            report["video_wrist_pose0"] = [round(float(v), 4) for v in video[0].cameras["wrist"].camera_pose.position]
            report["video_ee0"] = [round(float(v), 4) for v in video[0].robot_state.end_effector_poses["primary"].position]
        except Exception as exc:
            report["errors"].append("video pose: " + str(exc))
        last_pre = -1
        max_move = 0.0
        min_sep = 9.0
        top_counts = []
        for index in range(n):
            frame = video[index]
            top, low, cells, ntall = fused_points(frame, vcache, 4, False)
            if index == 0:
                report["video_tall_cells0"] = len(cells)
                report["video_ntall0"] = ntall
            if index % 32 == 0:
                top_counts.append([index, len(top)])
            if tracks is None:
                if len(top) < 6:
                    last_pre = index
                    continue
                report["containers_appear_frame"] = index
                report["containers_appear_npts"] = len(top)
                if last_pre >= 0:
                    cubes = detect_cubes(video[last_pre], vcache)
                report["cubes"] = [[c["color"], round(c["x"], 4), round(c["y"], 4), c["npix"]] for c in cubes]
                tracks = []
                for c in cubes:
                    tracks.append({"x": c["x"], "y": c["y"], "color": c["color"]})
                extra = 0
                for cl in cluster_xy(top, 0.02):
                    if len(cl) < 4:
                        continue
                    mx = sum(p[0] for p in cl) / len(cl)
                    my = sum(p[1] for p in cl) / len(cl)
                    near = False
                    for t in tracks:
                        if (t["x"] - mx) ** 2 + (t["y"] - my) ** 2 < 0.035 * 0.035:
                            near = True
                    if not near:
                        tracks.append({"x": mx, "y": my, "color": "unknown"})
                        extra += 1
                report["unknown_tracks_added"] = extra
                if not tracks:
                    break
                for ti in (last_pre, index, index + 40):
                    if ti >= 0 and ti < n:
                        try:
                            thumbs[str(ti)] = [enc_rgb(video[ti].cameras["front"].rgb, 4), enc_rgb(video[ti].cameras["wrist"].rgb, 4)]
                        except Exception:
                            pass
            frame_dt = 0.05
            if index > 0:
                frame_dt = float(frame.timestamp_s) - float(video[index - 1].timestamp_s)
            moved = track_frame(tracks, top, frame_dt)
            if moved > max_move:
                max_move = moved
            sep = min_separation(tracks)
            if sep < min_sep:
                min_sep = sep
            if index % 64 == 0 or index == n - 1:
                video_snaps.append([index, round(float(frame.timestamp_s), 3), snapshot(tracks)])
        report["video_top_counts"] = top_counts
        report["video_max_step"] = round(max_move, 4)
        report["video_min_separation"] = round(min_sep, 4)
        for ti in (0, n - 1):
            try:
                thumbs[str(ti)] = [enc_rgb(video[ti].cameras["front"].rgb, 4), enc_rgb(video[ti].cameras["wrist"].rgb, 4)]
            except Exception:
                pass
except Exception as exc:
    report["errors"].append("video: " + str(exc))
report["video_snaps"] = video_snaps
report["video_thumbs_s4"] = thumbs

# ---------------- live tracking until still ----------------
live_log = []
lcache = {}
obstacle = []
try:
    state = get_robot_state()
    base_frame = state.base_frame
    embodiment = state.embodiment
    home_joints = [float(v) for v in state.joint_positions["primary"]]
    q_down = [float(v) for v in state.end_effector_poses["primary"].quaternion_wxyz]
    obs = get_observation()
    if tracks is not None:
        top, low, cells, ntall = fused_points(obs, lcache, 4, False)
        jump = track_frame(tracks, top)
        report["live_first_jump"] = round(jump, 4)
        live_log.append([0, snapshot(tracks)])
        still_count = 0
        history = [[(t["x"], t["y"]) for t in tracks]]
        tick = 0
        cap = 500
        while tick < cap:
            reply = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=home_joints, gripper_position=1.0, embodiment=embodiment)}))
            tick += 1
            if reply.terminated or reply.truncated:
                ended = True
                report["ended_in_wait"] = tick
                break
            obs = get_observation()
            top, low, cells, ntall = fused_points(obs, lcache, 4, False)
            track_frame(tracks, top)
            history.append([(t["x"], t["y"]) for t in tracks])
            if len(history) > 6:
                old = history[len(history) - 6]
                drift = 0.0
                for ti in range(len(tracks)):
                    dd = math.sqrt((tracks[ti]["x"] - old[ti][0]) ** 2 + (tracks[ti]["y"] - old[ti][1]) ** 2)
                    if dd > drift:
                        drift = dd
                if drift < 0.003:
                    still_count += 1
                else:
                    still_count = 0
            if tick % 10 == 0:
                live_log.append([tick, snapshot(tracks)])
            if still_count >= 10:
                break
        report["live_wait_ticks"] = tick
        report["live_still"] = still_count >= 10
        report["live_min_separation"] = round(min_separation(tracks), 4)
        report["live_final"] = snapshot(tracks)
        # standing obstacles (anything 0.11-0.45 m above the table away from the robot base)
        for name in ("front", "wrist"):
            cam = obs.cameras[name]
            model = cam_model(cam)
            rays = ray_table(model, 4)
            tx = rays["t"][0]; ty = rays["t"][1]; tz = rays["t"][2]
            drows = to_rows(cam.depth_m, 4)
            for iy in range(len(drows)):
                for ix in range(len(drows[iy])):
                    d = drows[iy][ix]
                    if d > 0.05 and d < 4.0:
                        k = rays["rows"][iy][ix]
                        z = tz + d * k[2]
                        if z > TALL_LO and z < 0.45:
                            x = tx + d * k[0]
                            if x > X_MIN:
                                obstacle.append([x, ty + d * k[1], z])
        report["obstacle_points"] = len(obstacle)
        if obstacle:
            report["obstacle_ztop"] = round(max(p[2] for p in obstacle), 4)
except Exception as exc:
    report["errors"].append("live: " + str(exc))
report["live_log"] = live_log

# ---------------- plan and execute ordered lifts ----------------
picks = []
try:
    if tracks is not None and not ended:
        obs = get_observation()
        for ti in range(len(tracks)):
            t = tracks[ti]
            others = [(o["x"], o["y"]) for oj, o in enumerate(tracks) if oj != ti]
            ref = None
            for name in ("wrist", "front"):
                cam = obs.cameras[name]
                try:
                    ref = refine_container(cam_model(cam), cam.depth_m, t["x"], t["y"], t.get("z", (TOP_LO + TOP_HI) / 2.0), others)
                except Exception as exc:
                    report["errors"].append("refine " + name + ": " + str(exc))
                    ref = None
                if ref is not None:
                    ref["cam"] = name
                    break
            t["ref"] = ref
        report["refined"] = [[t["color"], t["ref"]] for t in tracks]
        taken = [False] * len(tracks)
        for item in goal:
            color = item[0]
            cands = []
            for ti in range(len(tracks)):
                if not taken[ti] and tracks[ti]["color"] == color:
                    cands.append(ti)
            if not item[1]:
                cands = cands[:1]
            for ti in cands:
                taken[ti] = True
                picks.append(ti)
        report["pick_plan"] = [[tracks[ti]["color"], round(tracks[ti]["x"], 4), round(tracks[ti]["y"], 4)] for ti in picks]
except Exception as exc:
    report["errors"].append("plan: " + str(exc))

lifts = []
report["lifts"] = lifts
try:
    for ti in picks:
        if ended:
            break
        t = tracks[ti]
        ref = t.get("ref")
        if ref is None:
            ref = {"x": t["x"], "y": t["y"], "top_z": t.get("z", (TOP_LO + TOP_HI) / 2.0), "yaw_deg": 0.0}
        gx = ref["x"]; gy = ref["y"]; top_z = ref["top_z"]
        half = math.radians(ref["yaw_deg"]) / 2.0
        quat = qmul([math.cos(half), 0.0, 0.0, math.sin(half)], q_down)
        grasp_z = top_z - 0.028
        hover_z = top_z + 0.07
        near_top = None
        for p in obstacle:
            if (p[0] - gx) ** 2 + (p[1] - gy) ** 2 < 0.12 * 0.12:
                if near_top is None or p[2] > near_top:
                    near_top = p[2]
        if near_top is not None and near_top + 0.05 > hover_z:
            hover_z = near_top + 0.05
        info = {"color": t["color"], "target": [round(gx, 4), round(gy, 4), round(top_z, 4), ref["yaw_deg"]], "hover_z": round(hover_z, 4)}
        lifts.append(info)
        ended = motion("above_" + t["color"], move_to_pose(Pose((gx, gy, hover_z), quat, base_frame), tolerance=0.01, max_steps=200))
        if ended:
            break
        ended = motion("descend", move_to_pose(Pose((gx, gy, grasp_z), quat, base_frame), tolerance=0.008, max_steps=150))
        if ended:
            break
        ended = motion("close", close_gripper())
        info["grip_closed"] = round(float(get_robot_state().gripper_positions["primary"]), 4)
        if ended:
            break
        ended = motion("lift", move_to_pose(Pose((gx, gy, grasp_z + 0.12), quat, base_frame), tolerance=0.015, max_steps=150))
        if ended:
            break
        st = get_robot_state()
        info["grip_lifted"] = round(float(st.gripper_positions["primary"]), 4)
        info["ee_lifted"] = [round(float(v), 4) for v in st.end_effector_poses["primary"].position]
        ended = motion("lower", move_to_pose(Pose((gx, gy, grasp_z + 0.006), quat, base_frame), tolerance=0.01, max_steps=150))
        if ended:
            break
        ended = motion("release", open_gripper())
        if ended:
            break
        ended = motion("retreat", move_to_pose(Pose((gx, gy, hover_z), quat, base_frame), tolerance=0.015, max_steps=150))
        if ended:
            break
except Exception as exc:
    report["errors"].append("lifts: " + str(exc))

report["ended"] = ended
report["final_tick"] = ticks_now() if not ended else "ended"
result = report
