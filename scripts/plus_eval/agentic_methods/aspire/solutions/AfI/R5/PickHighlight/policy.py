# PickHighlight v03 — v02 + shuffle timing only from starts SEEN at home (epoch-tagged, re-anchored),
#   joint-step press (no 120-tick final-waypoint timeout), faster pick motions.
# v02 evidence (ep0): picks 0,1 OK; shuffles at 196/476 happened while the arm was away, so the
#   learned period was 2x (281) -> pick 1 overlapped a whole shuffle -> identity lost -> pick 2 wrong.
# (v02 header follows)
# PickHighlight v02 — continuous cube tracking across PERIODIC shuffles; one pick per static window.
# Goal (ep0): "first press the button, then pick up all highlighted cubes one by one,
#              finally press the button to stop".  No conditioning video.
# v01 evidence (ep0): highlight ticks 10-54, then the cubes shuffle for ~40 ticks every ~140 ticks
# (starts 56, 196, 336, 476) regardless of the arm; v01 picked from positions frozen at tick 123,
# two picks were right by luck, the third lifted a non-highlighted cube -> terminated.
# v02:
#   A. hold at home, per-tick colour-blob tracker; highlight = owned new white pixels / cube area;
#      gripper closes meanwhile (needed for the press); stop after the first shuffle ends.
#   B. press the button (above -> push, capped), back to home joints.
#   C. keep tracking at home through shuffles; learn the shuffle period from two observed starts;
#      in each static window with enough ticks left, pick ONE highlighted cube from its current
#      tracked position (wrist re-centre, yaw-aligned), lift, set it back, return home.
#   D. final press.
import math

MAX_WATCH = 700
NO_SHUFFLE_TICKS = 80
PICK_TICKS = 90
SLACK = 8
PUSH_TICKS = 12
LIFT = 0.10
report = {"version": "v03", "errors": [], "events": []}
flags = {"ended": False}


def note_terminal(res, label):
    term = False
    trunc = False
    try:
        term = bool(res.terminated)
    except Exception:
        term = False
    try:
        trunc = bool(res.truncated)
    except Exception:
        trunc = False
    if term or trunc:
        flags["ended"] = True
        report["events"].append({"ended_at": label, "terminated": term, "truncated": trunc})
    return res


def status_of(res):
    out = {}
    try:
        out["ok"] = bool(res.ok)
    except Exception:
        out["ok"] = None
    try:
        out["status"] = str(res.status)
    except Exception:
        out["status"] = None
    try:
        out["steps"] = int(res.steps_executed)
    except Exception:
        out["steps"] = None
    return out


def to_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        return arr


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    return ordered[int((len(ordered) - 1) * fraction)]


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def quat_rotate(q, v):
    w, x, y, z = q
    return ((1 - 2 * (y * y + z * z)) * v[0] + 2 * (x * y - z * w) * v[1] + 2 * (x * z + y * w) * v[2],
            2 * (x * y + z * w) * v[0] + (1 - 2 * (x * x + z * z)) * v[1] + 2 * (y * z - x * w) * v[2],
            2 * (x * z - y * w) * v[0] + 2 * (y * z + x * w) * v[1] + (1 - 2 * (x * x + y * y)) * v[2])


def cam_model(cam):
    k = to_rows(cam.intrinsics)
    pose = cam.camera_pose
    return {"fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]), "cy": float(k[1][2]),
            "q": [float(c) for c in pose.quaternion_wxyz], "t": [float(c) for c in pose.position]}


def pix_to_base(model, depth_rows, u, v):
    d = float(depth_rows[int(v)][int(u)])
    p = ((u - model["cx"]) / model["fx"] * d, (v - model["cy"]) / model["fy"] * d, d)
    r = quat_rotate(model["q"], p)
    return [r[0] + model["t"][0], r[1] + model["t"][1], r[2] + model["t"][2]]


def is_cube_px(px):
    return px[1] - px[0] > 15 and px[1] > 80


def is_white_px(px):
    return px[0] > 212 and px[1] > 212 and px[2] > 212


# ---------------------------------------------------------------- blob detection
def find_blobs(rows):
    h = len(rows)
    w = len(rows[0])
    seen = set()
    blobs = []
    for y in range(h):
        row = rows[y]
        for x in range(w):
            if (y * w + x) in seen:
                continue
            if not is_cube_px(row[x]):
                continue
            stack = [(x, y)]
            seen.add(y * w + x)
            pts = []
            while stack:
                cx, cy = stack.pop()
                pts.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if nx < 0 or ny < 0 or nx >= w or ny >= h:
                        continue
                    key = ny * w + nx
                    if key in seen:
                        continue
                    if is_cube_px(rows[ny][nx]):
                        seen.add(key)
                        stack.append((nx, ny))
            if len(pts) >= 25:
                blobs.append(pts)
    return blobs


def centroid(pts):
    n = float(len(pts))
    return [sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n]


def split_k(pts, k):
    # k-means seeded along the principal spread, for blobs holding k touching cubes
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    horiz = max(xs) - min(xs) >= max(ys) - min(ys)
    order = sorted(pts) if horiz else [(q[1], q[0]) for q in sorted((p[1], p[0]) for p in pts)]
    cents = []
    for j in range(k):
        cents.append(list(order[int((j + 0.5) * len(order) / k)]))
    groups = []
    for it in range(10):
        groups = [[] for j in range(k)]
        for p in pts:
            best = 0
            bestd = None
            for j in range(k):
                d = (p[0] - cents[j][0]) ** 2 + (p[1] - cents[j][1]) ** 2
                if bestd is None or d < bestd:
                    best = j
                    bestd = d
            groups[best].append(p)
        for j in range(k):
            if groups[j]:
                cents[j] = centroid(groups[j])
    return [g for g in groups if g]


def init_tracks(rows):
    blobs = find_blobs(rows)
    areas = sorted(len(b) for b in blobs)
    med = areas[len(areas) // 2] if areas else 0
    tracks = []
    for b in blobs:
        parts = [b]
        if med > 0 and len(b) > 1.45 * med:
            parts = split_k(b, max(2, int(round(len(b) / float(med)))))
        for p in parts:
            if len(p) < 15:
                continue
            c = centroid(p)
            rad = math.sqrt(len(p) / math.pi)
            tracks.append({"u": c[0], "v": c[1], "u0": c[0], "v0": c[1], "area": len(p),
                           "rad": rad, "white0": None, "hl": 0.0, "hl_ticks": 0, "path": 0.0})
    return tracks


def update_tracks(rows, tracks):
    # assign green pixels near the tracks to the nearest track; move each track to its pixels' mean
    h = len(rows)
    w = len(rows[0])
    umin = max(0, int(min(t["u"] for t in tracks) - 30))
    umax = min(w - 1, int(max(t["u"] for t in tracks) + 30))
    vmin = max(0, int(min(t["v"] for t in tracks) - 30))
    vmax = min(h - 1, int(max(t["v"] for t in tracks) + 30))
    acc = [[0.0, 0.0, 0] for t in tracks]
    for y in range(vmin, vmax + 1):
        row = rows[y]
        for x in range(umin, umax + 1):
            if not is_cube_px(row[x]):
                continue
            best = -1
            bestd = 0.0
            for ti in range(len(tracks)):
                t = tracks[ti]
                d = (x - t["u"]) ** 2 + (y - t["v"]) ** 2
                lim = (t["rad"] * 1.8 + 3.0) ** 2
                if d <= lim and (best < 0 or d < bestd):
                    best = ti
                    bestd = d
            if best >= 0:
                a = acc[best]
                a[0] += x
                a[1] += y
                a[2] += 1
    moved = 0.0
    for ti in range(len(tracks)):
        t = tracks[ti]
        a = acc[ti]
        if a[2] >= 8:
            nu = a[0] / a[2]
            nv = a[1] / a[2]
            dd = math.hypot(nu - t["u"], nv - t["v"])
            t["path"] += dd
            moved = max(moved, dd)
            t["u"] = nu
            t["v"] = nv
            t["seen"] = a[2]
        else:
            t["seen"] = a[2]
    return moved


def white_counts(rows, tracks):
    # global white count (stride 2) and, per track, white pixels whose NEAREST track it is
    h = len(rows)
    w = len(rows[0])
    total = 0
    for y in range(0, h, 2):
        row = rows[y]
        for x in range(0, w, 2):
            if is_white_px(row[x]):
                total += 1
    per = [0 for t in tracks]
    umin = max(0, int(min(t["u"] for t in tracks) - 40))
    umax = min(w - 1, int(max(t["u"] for t in tracks) + 40))
    vmin = max(0, int(min(t["v"] for t in tracks) - 40))
    vmax = min(h - 1, int(max(t["v"] for t in tracks) + 40))
    for y in range(vmin, vmax + 1):
        row = rows[y]
        for x in range(umin, umax + 1):
            if not is_white_px(row[x]):
                continue
            best = -1
            bestd = 0.0
            for ti in range(len(tracks)):
                t = tracks[ti]
                d = (x - t["u"]) ** 2 + (y - t["v"]) ** 2
                lim = (t["rad"] * 2.6 + 3.0) ** 2
                if d <= lim and (best < 0 or d < bestd):
                    best = ti
                    bestd = d
            if best >= 0:
                per[best] += 1
    return total, per




# ---------------------------------------------------------------- context
context = get_task_context()
goal = str(context.language)
report["goal"] = goal
meta = {}
try:
    meta = robomme_plus.get_task_metadata()
    report["task_metadata"] = {str(k): str(meta[k]) for k in meta}
except Exception as exc:
    report["errors"].append("task_metadata: " + str(exc))
max_steps = 0
try:
    max_steps = int(meta["max_steps"])
except Exception:
    max_steps = 0
hz = 20.0
try:
    hz = float(meta["control_frequency_hz"])
except Exception:
    hz = 20.0
report["max_steps"] = max_steps
video_len = -1
try:
    video_len = len(robomme_plus.get_conditioning_video())
except Exception as exc:
    report["errors"].append("video: " + str(exc))
report["video_len"] = video_len

state0 = get_robot_state()
base_frame = state0.base_frame
embodiment = state0.embodiment
home_q = list(state0.joint_positions["primary"])
down_q = [float(c) for c in state0.end_effector_poses["primary"].quaternion_wxyz]

obs0 = get_observation()
front0 = obs0.cameras["front"]
fmodel = cam_model(front0)
rows0 = to_rows(front0.rgb)
tracks = init_tracks(rows0)
report["n_tracks0"] = len(tracks)
report["tracks0"] = [[round(t["u"], 1), round(t["v"], 1), t["area"]] for t in tracks]


def now_tick():
    return int(round(float(get_robot_state().timestamp_s) * hz))


# ---------------------------------------------------------------- per-tick watcher (arm at home)
W = {"grip": 0.0, "white_base": None, "hl_seen": False, "hl_over": False, "hl_start": -1, "hl_end": -1,
     "moving": False, "move_run": 0, "static_run": 0, "starts": [], "ends": [],
     "white_series": [], "moved_series": [], "hist": [], "last_obs": obs0, "ticks": 0,
     "fresh": 0, "epoch": 0, "vstarts": [], "away": []}


def leave_home():
    # the arm leaves the watch pose: frames are no longer tracked until it is back
    W["epoch"] += 1
    W["fresh"] = 0
    W["move_run"] = 0
    W["static_run"] = 0
    W["away"].append(now_tick())


def watch_tick():
    r = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=home_q, gripper_position=W["grip"], embodiment=embodiment)}))
    note_terminal(r, "watch")
    W["ticks"] += 1
    if flags["ended"]:
        return
    try:
        obs = r.observation
        rows = to_rows(obs.cameras["front"].rgb)
    except Exception:
        obs = get_observation()
        rows = to_rows(obs.cameras["front"].rgb)
    W["last_obs"] = obs
    W["fresh"] += 1
    tk = int(round(float(obs.timestamp_s) * hz))
    moved = update_tracks(rows, tracks)
    W["moved_series"].append([tk, round(moved, 2)])
    if tk % 2 == 0:
        W["hist"].append([tk] + [[round(t["u"], 1), round(t["v"], 1)] for t in tracks])
    if not W["hl_over"]:
        total, per = white_counts(rows, tracks)
        W["white_series"].append(total)
        if W["white_base"] is None:
            W["white_base"] = total
            for ti in range(len(tracks)):
                tracks[ti]["white0"] = per[ti]
        if total - W["white_base"] > 40:
            if not W["hl_seen"]:
                W["hl_start"] = tk
            W["hl_seen"] = True
            for ti in range(len(tracks)):
                gain = per[ti] - tracks[ti]["white0"]
                tracks[ti]["hl"] = max(tracks[ti]["hl"], float(gain))
                if gain > 10:
                    tracks[ti]["hl_ticks"] += 1
        elif W["hl_seen"]:
            W["hl_over"] = True
            W["hl_end"] = tk
    # shuffle segmentation: >=3 consecutive moving ticks start a shuffle, >=4 static ticks end it
    if moved >= 0.35:
        W["move_run"] += 1
        W["static_run"] = 0
    else:
        W["static_run"] += 1
        W["move_run"] = 0
    if not W["moving"] and W["move_run"] >= 3 and W["hl_over"]:
        W["moving"] = True
        W["starts"].append(tk - 2)
        if W["fresh"] - W["move_run"] >= 4:
            # motion began while we were watching a static scene: a true start time
            W["vstarts"].append([tk - 2, W["epoch"]])
    if W["moving"] and W["static_run"] >= 4:
        W["moving"] = False
        W["ends"].append(tk - 3)


def watch_until_shuffle_end(max_ticks):
    # hold at home and track until the next shuffle has started and ended (or max_ticks)
    n_end = len(W["ends"])
    begin = W["ticks"]
    while not flags["ended"] and W["ticks"] - begin < max_ticks:
        watch_tick()
        if len(W["ends"]) > n_end:
            return True
    return False


# Phase A: highlight, then the first shuffle (or a long static period when nothing moves)
if tracks:
    begin_a = W["ticks"]
    while not flags["ended"] and W["ticks"] - begin_a < MAX_WATCH:
        watch_tick()
        if W["ends"]:
            break
        if W["hl_over"] and not W["moving"] and not W["starts"] and W["static_run"] >= NO_SHUFFLE_TICKS:
            break

# ---------------------------------------------------------------- choose highlighted tracks
for t in tracks:
    t["ratio"] = t["hl"] / max(1.0, float(t["area"]))
ratios = sorted([t["ratio"] for t in tracks], reverse=True)
top = ratios[0] if ratios else 0.0
selected = []
for ti in range(len(tracks)):
    t = tracks[ti]
    if top > 0.3 and t["ratio"] >= max(0.3, 0.3 * top) and t["hl_ticks"] >= 3:
        selected.append(ti)
report["selected"] = selected
report["highlight"] = [W["hl_start"], W["hl_end"]]


def target_from_front(ti):
    lf = None
    try:
        lf = W["last_obs"].cameras["front"]
        depth_rows = to_rows(lf.depth_m)
        rgb_rows = to_rows(lf.rgb)
    except Exception:
        lf = get_observation().cameras["front"]
        depth_rows = to_rows(lf.depth_m)
        rgb_rows = to_rows(lf.rgb)
    model = cam_model(lf)
    t = tracks[ti]
    rad = t["rad"] * 1.4
    pts = []
    for y in range(max(0, int(t["v"] - rad)), min(len(rgb_rows), int(t["v"] + rad) + 1)):
        for x in range(max(0, int(t["u"] - rad)), min(len(rgb_rows[0]), int(t["u"] + rad) + 1)):
            if (x - t["u"]) ** 2 + (y - t["v"]) ** 2 <= rad * rad and is_cube_px(rgb_rows[y][x]):
                pts.append(pix_to_base(model, depth_rows, x, y))
    if len(pts) < 10:
        return None
    zt = quantile([p[2] for p in pts], 0.95)
    top_pts = [p for p in pts if p[2] > zt - 0.012]
    return {"track": ti, "x": quantile([p[0] for p in top_pts], 0.5),
            "y": quantile([p[1] for p in top_pts], 0.5), "ztop": zt}


# ---------------------------------------------------------------- motion helpers
def move(xyz, label, quat=None, tol=0.012, steps=250):
    if flags["ended"]:
        return None
    q = down_q if quat is None else quat
    r = move_to_pose(Pose(tuple(xyz), q, base_frame), tolerance=tol, max_steps=steps)
    note_terminal(r, label)
    report["events"].append({label: status_of(r)})
    return r


def hold(nticks, grip, label):
    if flags["ended"]:
        return
    st = get_robot_state()
    q = st.joint_positions["primary"]
    for k in range(nticks):
        r = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=q, gripper_position=grip, embodiment=embodiment)}))
        note_terminal(r, label)
        if flags["ended"]:
            return


def grip_call(close, label):
    if flags["ended"]:
        return
    r = close_gripper() if close else open_gripper()
    note_terminal(r, label)


def cloud_stats(pts):
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    zs = [float(p[2]) for p in pts]
    return xs, ys, zs


def yaw_of(xs, ys):
    # min-area rectangle angle over 0..87 deg (cube faces), returned in [-45, 45) deg
    best = None
    best_a = 0.0
    for k in range(30):
        a = math.radians(k * 3.0)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [x * ca + y * sa for x, y in zip(xs, ys)]
        vs = [-x * sa + y * ca for x, y in zip(xs, ys)]
        area = (quantile(us, 0.99) - quantile(us, 0.01)) * (quantile(vs, 0.99) - quantile(vs, 0.01))
        if best is None or area < best:
            best = area
            best_a = k * 3.0
    if best_a >= 45.0:
        best_a -= 90.0
    return best_a


def find_button():
    best = None
    for prompt in ("button", "white button", "gray button"):
        seg = segment_text("front", prompt)
        if not seg.ok:
            continue
        for cand in seg.segmentations[:5]:
            if cand.box_xyxy is None:
                continue
            box = [float(v) for v in cand.box_xyxy]
            if box[2] - box[0] > 50 or box[3] - box[1] > 50:
                continue
            # colour test: a button is grey/white, cubes are green
            sat = 0.0
            n = 0
            for y in range(int(box[1]), int(box[3]) + 1):
                for x in range(int(box[0]), int(box[2]) + 1):
                    px = rows0[min(len(rows0) - 1, y)][min(len(rows0[0]) - 1, x)]
                    sat += max(px) - min(px)
                    n += 1
            green = 0
            for y in range(int(box[1]), int(box[3]) + 1):
                for x in range(int(box[0]), int(box[2]) + 1):
                    if is_cube_px(rows0[min(len(rows0) - 1, y)][min(len(rows0[0]) - 1, x)]):
                        green += 1
            if n == 0 or green > 0.15 * n:
                continue
            try:
                cloud = mask_to_point_cloud(cand, "front", target_frame=base_frame)
            except Exception:
                continue
            if len(cloud.points) < 10:
                continue
            xs, ys, zs = cloud_stats(cloud.points)
            zt = quantile(zs, 0.97)
            cap = [(x, y) for x, y, z in zip(xs, ys, zs) if z > zt - 0.01]
            c = {"prompt": prompt, "score": float(cand.score), "box": box, "sat": sat / n,
                 "x": quantile([p[0] for p in cap], 0.5), "y": quantile([p[1] for p in cap], 0.5),
                 "ztop": zt, "span": quantile(xs, 0.95) - quantile(xs, 0.05)}
            if c["span"] > 0.12:
                continue
            if best is None or c["sat"] < best["sat"]:
                best = c
        if best is not None:
            break
    return best


def press(button, label):
    if flags["ended"] or button is None:
        return
    if float(get_robot_state().gripper_positions["primary"]) > 0.1:
        grip_call(True, label + "_close")
    bx = button["x"]
    by = button["y"]
    bz = button["ztop"]
    leave_home()
    move((bx, by, bz + 0.05), label + "_above", tol=0.015)
    if flags["ended"]:
        return
    ik = solve_ik(Pose((bx, by, bz - 0.02), down_q, base_frame))
    if ik.ok:
        q = list(ik.joint_positions)
        for k in range(PUSH_TICKS):
            r = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=q, gripper_position=0.0, embodiment=embodiment)}))
            note_terminal(r, label + "_push")
            if flags["ended"]:
                return
        report["events"].append({label + "_push": "ik-step %d" % PUSH_TICKS})
        try:
            report[label + "_push_ee_z"] = round(float(get_robot_state().end_effector_poses["primary"].position[2]), 4)
        except Exception:
            pass
    else:
        move((bx, by, bz - 0.02), label + "_push", tol=0.01, steps=20)
    move((bx, by, bz + 0.05), label + "_up", tol=0.03, steps=30)


def wrist_recentre(target):
    # at hover the wrist camera sees the cube unoccluded; pick the cube candidate nearest the target
    best = None
    seg = segment_text("wrist", "cube")
    if not seg.ok:
        return None
    for cand in seg.segmentations[:5]:
        try:
            cloud = mask_to_point_cloud(cand, "wrist", target_frame=base_frame)
        except Exception:
            continue
        if len(cloud.points) < 15:
            continue
        xs, ys, zs = cloud_stats(cloud.points)
        zt = quantile(zs, 0.95)
        if zt > 0.09:
            continue
        top_idx = [i for i in range(len(zs)) if zs[i] > zt - 0.012]
        tx = [xs[i] for i in top_idx]
        ty = [ys[i] for i in top_idx]
        cx = quantile(tx, 0.5)
        cy = quantile(ty, 0.5)
        sx = quantile(xs, 0.95) - quantile(xs, 0.05)
        sy = quantile(ys, 0.95) - quantile(ys, 0.05)
        if sx > 0.08 or sy > 0.08:
            continue
        d = math.hypot(cx - target["x"], cy - target["y"])
        if d > 0.03:
            continue
        if best is None or d < best["d"]:
            best = {"x": cx, "y": cy, "ztop": zt, "d": d, "yaw": yaw_of(tx, ty),
                    "score": float(cand.score)}
    return best




def go_home_joints(label):
    if flags["ended"]:
        return
    r = move_to_joints(home_q, tolerance=0.02, max_steps=80)
    note_terminal(r, label)
    report["events"].append({label: status_of(r)})


def real_starts():
    # starts seen at home whose shuffle lasted >= 10 ticks (or is still running); short blips dropped
    out = []
    for vs in W["vstarts"]:
        dur = None
        for e in W["ends"]:
            if e > vs[0]:
                dur = e - vs[0]
                break
        if dur is None or dur >= 10:
            out.append([vs[0], vs[1], dur])
    return out


def shuffle_period():
    # only pairs of consecutive real starts observed while continuously at home (same epoch)
    best = None
    rs = real_starts()
    for i in range(1, len(rs)):
        if rs[i][1] == rs[i - 1][1]:
            d = rs[i][0] - rs[i - 1][0]
            dur = rs[i - 1][2] if rs[i - 1][2] is not None else 0
            # a period must contain the whole earlier shuffle plus a real static gap
            if d > dur + 20 and (best is None or d < best):
                best = d
    return best


def window_left():
    # ticks until the next predicted shuffle start; None when no period is known yet
    period = shuffle_period()
    rs = real_starts()
    if period is None or not rs:
        return None
    nxt = rs[-1][0] + period
    tk = now_tick()
    while nxt <= tk:
        nxt += period
    dur_typ = 0
    for r in rs:
        if r[2] is not None and r[2] > dur_typ:
            dur_typ = r[2]
    if tk - (nxt - period) < dur_typ + 3 or W["moving"]:
        # a predicted shuffle is (probably) still running, e.g. it began while the arm was away
        return 0
    return nxt - tk


def pick_and_replace(ti, order):
    rec = {"track": ti, "t_start": now_tick()}
    tg = target_from_front(ti)
    if tg is None:
        rec["error"] = "no 3-D target"
        return rec
    rec["front_xy"] = [round(tg["x"], 4), round(tg["y"], 4)]
    leave_home()
    if float(get_robot_state().gripper_positions["primary"]) < 0.9:
        grip_call(False, "open%d" % order)
    move((tg["x"], tg["y"], tg["ztop"] + 0.08), "pre%d" % order, tol=0.025)
    if flags["ended"]:
        return rec
    wr = None
    try:
        wr = wrist_recentre(tg)
    except Exception as exc:
        report["errors"].append("wrist: " + str(exc))
    gx = tg["x"]
    gy = tg["y"]
    gz = tg["ztop"]
    yaw = 0.0
    if wr is not None:
        gx = wr["x"]
        gy = wr["y"]
        gz = wr["ztop"]
        yaw = wr["yaw"]
        rec["wrist"] = {"xy": [round(gx, 4), round(gy, 4)], "ztop": round(gz, 4), "yaw": yaw,
                        "d": round(wr["d"], 4)}
    half = math.radians(yaw) / 2.0
    gq = quat_mul([math.cos(half), 0.0, 0.0, math.sin(half)], down_q)
    grasp_z = gz - 0.02
    move((gx, gy, grasp_z), "grasp%d" % order, quat=gq, tol=0.01, steps=60)
    if flags["ended"]:
        return rec
    grip_call(True, "close%d" % order)
    move((gx, gy, grasp_z + LIFT), "lift%d" % order, quat=gq, tol=0.02, steps=60)
    hold(1, 0.0, "liftHold%d" % order)
    if flags["ended"]:
        return rec
    st = get_robot_state()
    rec["grip_after_lift"] = round(float(st.gripper_positions["primary"]), 3)
    move((gx, gy, grasp_z + 0.008), "down%d" % order, quat=gq, tol=0.02, steps=60)
    if flags["ended"]:
        return rec
    grip_call(False, "release%d" % order)
    move((gx, gy, grasp_z + 0.05), "up%d" % order, quat=gq, tol=0.03, steps=40)
    go_home_joints("home%d" % order)
    rec["t_end"] = now_tick() if not flags["ended"] else None
    return rec


# ---------------------------------------------------------------- Phase B: first press
button = None
if not flags["ended"]:
    button = find_button()
report["button"] = None if button is None else {k: (round(v, 4) if k in ("x", "y", "ztop", "sat", "span", "score") else v) for k, v in button.items()}
if button is None:
    report["errors"].append("button not found")
t_press = now_tick() if not flags["ended"] else None
press(button, "press1")
go_home_joints("home_press1")
W["grip"] = 1.0
report["press1_ticks"] = [t_press, now_tick() if not flags["ended"] else None]

# ---------------------------------------------------------------- Phase C: one pick per static window
picks = []
need = PICK_TICKS
durations = []
pending = list(selected)
waits = 0
decisions = []
while pending and not flags["ended"] and waits < 14:
    left = window_left()
    shuffled = len(W["starts"]) > 0
    decisions.append([now_tick(), left, need, shuffle_period()])
    at_window_start = False
    if W["ends"] and W["starts"] and W["fresh"] >= 4 and (now_tick() - W["ends"][-1]) <= 8:
        # just after the end of a full-length shuffle: the whole static window is ahead of us
        at_window_start = (W["ends"][-1] - W["starts"][-1]) >= 10
    if shuffled and (left is None or (left < need - SLACK and not at_window_start)):
        # no period known yet, or not enough time before the next shuffle: wait it out at home
        watch_until_shuffle_end(400)
        waits += 1
        continue
    ti = pending.pop(0)
    rec = pick_and_replace(ti, len(picks))
    rec["window_left"] = left
    picks.append(rec)
    if rec.get("t_end") is not None:
        durations.append(rec["t_end"] - rec["t_start"])
        need = int(round(sum(durations) / float(len(durations)))) + 2
    if not flags["ended"] and pending:
        watch_tick()
report["picks"] = picks
report["need_final"] = need
report["decisions"] = decisions
report["period"] = shuffle_period()
report["vstarts"] = W["vstarts"]
report["away"] = W["away"]

# ---------------------------------------------------------------- Phase D: final press
press(button, "press2")
if not flags["ended"]:
    hold(10, 0.0, "final_hold")
report["shuffle_starts"] = W["starts"]
report["shuffle_ends"] = W["ends"]
report["watch_ticks"] = W["ticks"]
report["white_series"] = W["white_series"]
report["moved_series"] = W["moved_series"]
report["track_hist"] = W["hist"]
report["tracks"] = [{"u": round(t["u"], 1), "v": round(t["v"], 1), "u0": round(t["u0"], 1),
                     "v0": round(t["v0"], 1), "area": t["area"], "hl": t["hl"],
                     "hl_ticks": t["hl_ticks"], "ratio": round(t["ratio"], 3),
                     "path": round(t["path"], 1), "seen": t.get("seen", 0)} for t in tracks]
if not flags["ended"]:
    try:
        report["final_t"] = float(get_robot_state().timestamp_s)
    except Exception:
        pass
report["ended"] = flags["ended"]
result = report
