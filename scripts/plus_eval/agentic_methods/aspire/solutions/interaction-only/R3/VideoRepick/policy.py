# VideoRepick v01 -- perception/memory fix: track the picked object through the
# post-pick shuffle in the conditioning video; then pick+place it N times and
# press the button (goal: "... identify the object that was picked up, then pick up
# and place down the same object twice, finally press the button to stop").
#
# v00 (ep 0) diagnosis: the video robot picks one of six identical green objects and
# puts it back; the objects then move around the table for ~35 s while the robot is
# still.  v00 matched the RELEASE location to the nearest live object, grasped a
# different object, and the episode terminated during the lift (task_success=false).
#
# Method (all derived at runtime from public data):
#  1. grasp event from the video's recorded robot state (gripper close -> lift -> reopen);
#  2. release location projected into the front image (calibrated pinhole, cv convention
#     self-checked against live SAM boxes);
#  3. reference object colour = chromaticity of pixels near that point that differ from
#     the surrounding table, sampled once the arm has moved away;
#  4. every retained video frame: object-coloured samples (stride 2) in windows around the
#     predicted track positions, assigned to the nearest track (k-means style), so merged
#     blobs split between their tracks; one extra step on the live front image;
#  5. the target track's final pixel -> live object (SAM box containing it, else a point
#     prompt) -> top-face centre and height from its point cloud;
#  6. pick (pre-grasp +0.10, descend to top-0.02, close, lift 0.12) and place back, N times;
#  7. press the button: non-object-coloured, low, compact "button" candidate; press at the
#     highest points' centre, closed gripper, from +0.06 down to top-0.015.
import math

report = {"version": "v01", "stage": "start"}
result = report

FRONT = "front"
LIFT_DZ = 0.12
PRE_DZ = 0.10
STRIDE = 2           # pixel sampling stride for video tracking
WIN = 14             # half window (px) around each predicted track position
CHROMA_TOL = 0.07    # chromaticity tolerance for object-coloured pixels


def vec(v):
    return [round(float(x), 4) for x in v]


def terminal_of(res):
    flags = {}
    try:
        flags["terminated"] = bool(res.terminated)
    except Exception:
        pass
    try:
        flags["truncated"] = bool(res.truncated)
    except Exception:
        pass
    return flags


def is_terminal(res):
    flags = terminal_of(res)
    return any(flags[k] for k in flags)


log = []
report["log"] = log
ended = {"flag": False}


def record(name, res):
    entry = {"call": name}
    try:
        entry["ok"] = bool(res.ok)
    except Exception:
        pass
    try:
        entry["status"] = str(res.status)
    except Exception:
        pass
    try:
        entry["steps"] = int(res.steps_executed)
    except Exception:
        pass
    entry.update(terminal_of(res))
    try:
        entry["error"] = str(res.error)[:200] if res.error is not None else None
    except Exception:
        pass
    if is_terminal(res):
        ended["flag"] = True
    else:
        try:
            st = get_robot_state()
            entry["t"] = round(float(st.timestamp_s), 3)
            entry["grip"] = round(float(st.gripper_positions["primary"]), 3)
            entry["ee"] = vec(st.end_effector_poses["primary"].position)
        except Exception:
            pass
    log.append(entry)
    return entry


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    return ordered[int((len(ordered) - 1) * fraction)]


def as_rows(rgb):
    try:
        return rgb.tolist()
    except Exception:
        return rgb


# ------------------------------------------------------------------ context
ctx = get_task_context()
goal = str(ctx.language)
report["goal"] = goal
try:
    meta = robomme_plus.get_task_metadata()
    report["max_steps"] = int(meta["max_steps"])
except Exception:
    report["max_steps"] = None
state0 = get_robot_state()
base_frame = state0.base_frame
home_q = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]
report["live_start_ee"] = vec(state0.end_effector_poses["primary"].position)

# ------------------------------------------------------------------ goal parse
words = goal.lower().replace(",", " ").replace(".", " ").split()
NUMBERS = {"once": 1, "one": 1, "twice": 2, "two": 2, "thrice": 3, "three": 3,
           "four": 4, "five": 5, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
repeat_count = None
for i, w in enumerate(words):
    if w in ("once", "twice", "thrice"):
        repeat_count = NUMBERS[w]
    if w in ("time", "times") and i > 0 and words[i - 1] in NUMBERS:
        repeat_count = NUMBERS[words[i - 1]]
put_down = any(w in ("put", "place", "down", "release", "back", "times", "twice", "thrice")
               for w in words)
press_button = "button" in words
report["goal_parse"] = {"repeat_count": repeat_count, "put_down": put_down, "press_button": press_button}

# ------------------------------------------------------------------ projection
def quat_to_rot(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def project(point, cam, convention):
    pose = cam.camera_pose
    R = quat_to_rot([float(v) for v in pose.quaternion_wxyz])
    c = [float(v) for v in pose.position]
    d = [point[i] - c[i] for i in range(3)]
    pc = [sum(R[r][i] * d[r] for r in range(3)) for i in range(3)]
    if convention == "gl":
        pc = [pc[0], -pc[1], -pc[2]]
    if pc[2] <= 1e-6:
        return None
    K = cam.intrinsics
    return [float(K[0][0]) * pc[0] / pc[2] + float(K[0][2]),
            float(K[1][1]) * pc[1] / pc[2] + float(K[1][2])]


# ------------------------------------------------------------------ video: grasp events
try:
    video = robomme_plus.get_conditioning_video()
    n_frames = len(video)
except Exception as exc:
    video = []
    n_frames = 0
    report["video_error"] = str(exc)[:200]
report["video_frames"] = n_frames
frames = []
for idx in range(n_frames):
    fr = video[idx]
    rs = fr.robot_state
    pose = rs.end_effector_poses["primary"]
    frames.append({"i": idx, "t": float(fr.timestamp_s), "g": float(rs.gripper_positions["primary"]),
                   "p": [float(x) for x in pose.position], "q": [float(x) for x in pose.quaternion_wxyz]})

events = []
open_state = True
cur = None
prev = None
for f in frames:
    if open_state and f["g"] < 0.8:
        open_state = False
        cz = f["p"][2]
        if prev is not None:
            cz = min(cz, prev["p"][2])
        cur = {"close_i": f["i"], "close_p": [f["p"][0], f["p"][1], cz], "close_q": f["q"],
               "min_g": f["g"], "max_z": f["p"][2]}
    elif (not open_state) and cur is not None:
        if f["g"] > 0.85:
            open_state = True
            cur["open_i"] = f["i"]
            cur["open_p"] = prev["p"]
            cur["open_q"] = prev["q"]
            events.append(cur)
            cur = None
        else:
            cur["min_g"] = min(cur["min_g"], f["g"])
            cur["max_z"] = max(cur["max_z"], f["p"][2])
    prev = f
picks = [e for e in events if e["max_z"] - e["close_p"][2] > 0.03 and e["min_g"] > 0.05]
report["video_picks"] = [{"close_i": e["close_i"], "open_i": e["open_i"], "close_p": vec(e["close_p"]),
                          "open_p": vec(e["open_p"])} for e in picks]
pick = picks[-1] if picks else None   # the (last) object the video robot picked up

# ------------------------------------------------------------------ live observation
obs = get_observation()
front = obs.cameras[FRONT]
H = len(front.rgb)
W = len(front.rgb[0])

# projection convention self-check against live SAM boxes
live_cands = []
for prompt in ("block", "cube"):
    try:
        seg = segment_text(FRONT, prompt)
    except Exception:
        continue
    if not seg.ok:
        continue
    for c in seg.segmentations[:5]:
        if c.box_xyxy is None:
            continue
        try:
            cloud = mask_to_point_cloud(c, FRONT, target_frame=base_frame)
        except Exception:
            continue
        pts = cloud.points
        if len(pts) < 15:
            continue
        zs = [float(p[2]) for p in pts]
        top = quantile(zs, 0.9)
        face = [p for p in pts if float(p[2]) >= top - 0.008]
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        live_cands.append({"prompt": prompt, "score": round(float(c.score), 3),
                           "box": [float(b) for b in c.box_xyxy],
                           "xy": [quantile([p[0] for p in face], 0.5), quantile([p[1] for p in face], 0.5)],
                           "top_z": top, "bot_z": quantile(zs, 0.05), "zmax": quantile(zs, 0.98),
                           "span": max(quantile(xs, 0.95) - quantile(xs, 0.05),
                                       quantile(ys, 0.95) - quantile(ys, 0.05))})
votes = {"cv": 0, "gl": 0}
for cand in live_cands:
    for conv in ("cv", "gl"):
        uv = project([cand["xy"][0], cand["xy"][1], cand["top_z"]], front, conv)
        b = cand["box"]
        if uv is not None and b[0] - 3 <= uv[0] <= b[2] + 3 and b[1] - 3 <= uv[1] <= b[3] + 3:
            votes[conv] += 1
convention = "cv" if votes["cv"] >= votes["gl"] else "gl"
report["projection_votes"] = votes
report["stage"] = "live_candidates"

# ------------------------------------------------------------------ tracker
def is_obj(px, ref):
    r = px[0]
    g = px[1]
    b = px[2]
    s = r + g + b
    if s < 60:
        return False
    return abs(r / s - ref[0]) < CHROMA_TOL and abs(g / s - ref[1]) < CHROMA_TOL


def components(points, step):
    remaining = set(points)
    comps = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            cu, cv = stack.pop()
            for du in (-step, 0, step):
                for dv in (-step, 0, step):
                    nb = (cu + du, cv + dv)
                    if nb in remaining:
                        remaining.remove(nb)
                        stack.append(nb)
                        comp.append(nb)
        comps.append(comp)
    return comps


def kmeans_split(pts, k):
    cs = [pts[0]]
    while len(cs) < k:
        best = None
        bd = -1
        for p in pts:
            d = min((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 for c in cs)
            if d > bd:
                bd = d
                best = p
        cs.append(best)
    cs = [[float(c[0]), float(c[1])] for c in cs]
    for it in range(5):
        acc = [[0.0, 0.0, 0] for c in cs]
        for p in pts:
            j = 0
            bd = None
            for ci in range(len(cs)):
                d = (p[0] - cs[ci][0]) ** 2 + (p[1] - cs[ci][1]) ** 2
                if bd is None or d < bd:
                    bd = d
                    j = ci
            acc[j][0] += p[0]
            acc[j][1] += p[1]
            acc[j][2] += 1
        for ci in range(len(cs)):
            if acc[ci][2] > 0:
                cs[ci] = [acc[ci][0] / acc[ci][2], acc[ci][1] / acc[ci][2]]
    return cs


def init_tracks(rgb, ref, h, w, step, min_pts):
    pts = []
    for v in range(0, h, step):
        row = rgb[v]
        for u in range(0, w, step):
            if is_obj(row[u], ref):
                pts.append((u, v))
    comps = [c for c in components(pts, step) if len(c) >= min_pts]
    if not comps:
        return [], 0
    sizes = sorted(len(c) for c in comps)
    single = sizes[len(sizes) // 2]
    tracks = []
    for c in comps:
        k = max(1, int(round(len(c) / float(single))))
        if k == 1:
            tracks.append([sum(p[0] for p in c) / len(c), sum(p[1] for p in c) / len(c)])
        else:
            for cc in kmeans_split(c, k):
                tracks.append(cc)
    return tracks, single


def track_step(rgb, ref, tracks, vel, h, w, step, win):
    preds = [[tracks[i][0] + vel[i][0], tracks[i][1] + vel[i][1]] for i in range(len(tracks))]
    seen = set()
    pts = []
    for p in preds:
        u0 = max(0, int(p[0]) - win)
        u1 = min(w, int(p[0]) + win + 1)
        v0 = max(0, int(p[1]) - win)
        v1 = min(h, int(p[1]) + win + 1)
        u0 -= u0 % step
        v0 -= v0 % step
        for v in range(v0, v1, step):
            row = rgb[v]
            for u in range(u0, u1, step):
                if (u, v) in seen:
                    continue
                seen.add((u, v))
                if is_obj(row[u], ref):
                    pts.append((u, v))
    cs = [[p[0], p[1]] for p in preds]
    counts = [0 for p in preds]
    for it in range(3):
        acc = [[0.0, 0.0, 0] for c in cs]
        for p in pts:
            j = -1
            bd = None
            for ci in range(len(cs)):
                d = (p[0] - cs[ci][0]) ** 2 + (p[1] - cs[ci][1]) ** 2
                if bd is None or d < bd:
                    bd = d
                    j = ci
            if bd is not None and bd <= win * win:
                acc[j][0] += p[0]
                acc[j][1] += p[1]
                acc[j][2] += 1
        for ci in range(len(cs)):
            if acc[ci][2] > 0:
                cs[ci] = [acc[ci][0] / acc[ci][2], acc[ci][1] / acc[ci][2]]
            counts[ci] = acc[ci][2]
    newvel = []
    for i in range(len(tracks)):
        if counts[i] > 0:
            newvel.append([0.5 * vel[i][0] + 0.5 * (cs[i][0] - tracks[i][0]),
                           0.5 * vel[i][1] + 0.5 * (cs[i][1] - tracks[i][1])])
        else:
            newvel.append([vel[i][0], vel[i][1]])
    return cs, newvel, counts


target_px = None
ref = None
track_info = {}
if pick is not None:
    rel = pick["open_p"]
    # start once the arm is clear of the released object (>= 6 cm away in xy, or 15 cm above)
    start_i = None
    for f in frames[pick["open_i"]:]:
        dxy = math.hypot(f["p"][0] - rel[0], f["p"][1] - rel[1])
        if dxy > 0.06 or f["p"][2] > rel[2] + 0.15:
            start_i = f["i"]
            break
    if start_i is None:
        start_i = n_frames - 1
    track_info["start_i"] = start_i
    cam_s = video[start_i].cameras[FRONT]
    rel_uv = project([rel[0], rel[1], rel[2]], cam_s, convention)
    track_info["release_uv"] = vec(rel_uv) if rel_uv else None
    rgb_s = as_rows(cam_s.rgb)
    if rel_uv is not None:
        # reference colour: pixels near the release point that differ from the window border
        cu = int(round(rel_uv[0]))
        cv = int(round(rel_uv[1]))
        half = 10
        border = []
        inner = []
        for v in range(max(0, cv - half), min(H, cv + half + 1)):
            for u in range(max(0, cu - half), min(W, cu + half + 1)):
                px = rgb_s[v][u]
                trip = [int(px[0]), int(px[1]), int(px[2])]
                if abs(u - cu) == half or abs(v - cv) == half:
                    border.append(trip)
                elif abs(u - cu) <= 6 and abs(v - cv) <= 6:
                    inner.append(trip)
        if border:
            tb = [quantile([b[ch] for b in border], 0.5) for ch in range(3)]
            obj = [p for p in inner if abs(p[0] - tb[0]) + abs(p[1] - tb[1]) + abs(p[2] - tb[2]) > 60
                   and sum(p) >= 60]
            track_info["table_rgb"] = vec(tb)
            track_info["ref_samples"] = len(obj)
            if len(obj) >= 5:
                ref = [quantile([p[0] / float(sum(p)) for p in obj], 0.5),
                       quantile([p[1] / float(sum(p)) for p in obj], 0.5)]
                track_info["ref_chroma"] = vec(ref)
    if ref is not None:
        tracks, single = init_tracks(rgb_s, ref, H, W, STRIDE, 4)
        track_info["n_tracks"] = len(tracks)
        track_info["single_size"] = single
        track_info["init_tracks"] = [vec(t) for t in tracks]
        if tracks:
            tgt = 0
            bd = None
            for j in range(len(tracks)):
                d = (tracks[j][0] - rel_uv[0]) ** 2 + (tracks[j][1] - rel_uv[1]) ** 2
                if bd is None or d < bd:
                    bd = d
                    tgt = j
            track_info["target_track"] = tgt
            track_info["target_init_dist_px"] = round(math.sqrt(bd), 2)
            vel = [[0.0, 0.0] for t in tracks]
            path = []
            lost = 0
            close_calls = 0
            report["stage"] = "tracking"
            for i in range(start_i + 1, n_frames + 1):
                if i < n_frames:
                    rgb = as_rows(video[i].cameras[FRONT].rgb)
                else:
                    rgb = as_rows(front.rgb)       # one more step onto the live image
                tracks, vel, counts = track_step(rgb, ref, tracks, vel, H, W, STRIDE, WIN)
                if counts[tgt] == 0:
                    lost += 1
                dmin = None
                for j in range(len(tracks)):
                    if j != tgt:
                        d = math.hypot(tracks[j][0] - tracks[tgt][0], tracks[j][1] - tracks[tgt][1])
                        if dmin is None or d < dmin:
                            dmin = d
                if dmin is not None and dmin < 8:
                    close_calls += 1
                if (i - start_i) % 16 == 0 or i == n_frames:
                    path.append([i, round(tracks[tgt][0], 1), round(tracks[tgt][1], 1), counts[tgt]])
            target_px = [tracks[tgt][0], tracks[tgt][1]]
            track_info["target_path"] = path
            track_info["target_lost_frames"] = lost
            track_info["target_close_calls_lt8px"] = close_calls
            track_info["final_tracks"] = [vec(t) for t in tracks]
            track_info["target_final_px"] = vec(target_px)
report["tracking"] = track_info
report["stage"] = "tracked"

# ------------------------------------------------------------------ choose live target
chosen = None
if target_px is not None:
    best = None
    for cand in live_cands:
        b = cand["box"]
        if b[0] - 1 <= target_px[0] <= b[2] + 1 and b[1] - 1 <= target_px[1] <= b[3] + 1 \
                and cand["span"] < 0.08 and cand["top_z"] - cand["bot_z"] < 0.09:
            cu = (b[0] + b[2]) / 2.0
            cv = (b[1] + b[3]) / 2.0
            d = math.hypot(cu - target_px[0], cv - target_px[1])
            if best is None or d < best[0]:
                best = [d, cand]
    if best is not None:
        chosen = best[1]
        report["choice"] = {"how": "sam_box_contains_track", "box_center_dist_px": round(best[0], 2)}
    else:
        try:
            seg = segment_points(FRONT, [[target_px[0], target_px[1]]])
            seg_ok = bool(seg.ok) and len(seg.segmentations) > 0
        except Exception:
            seg_ok = False
        if seg_ok:
            try:
                cloud = mask_to_point_cloud(seg.segmentations[0], FRONT, target_frame=base_frame)
                pts = cloud.points
            except Exception:
                pts = []
            if len(pts) >= 15:
                zs = [float(p[2]) for p in pts]
                top = quantile(zs, 0.9)
                face = [p for p in pts if float(p[2]) >= top - 0.008]
                chosen = {"prompt": "point", "xy": [quantile([p[0] for p in face], 0.5),
                                                    quantile([p[1] for p in face], 0.5)],
                          "top_z": top, "bot_z": quantile(zs, 0.05)}
                report["choice"] = {"how": "point_prompt"}
if chosen is None and pick is not None:
    rel = pick["open_p"]
    chosen = {"prompt": "video_release", "xy": [rel[0], rel[1]], "top_z": pick["close_p"][2] + 0.018,
              "bot_z": pick["close_p"][2] - 0.02}
    report["choice"] = {"how": "video_release_fallback"}
if chosen is not None:
    report["chosen"] = {"xy": vec(chosen["xy"]), "top_z": round(chosen["top_z"], 4),
                        "bot_z": round(chosen["bot_z"], 4), "prompt": chosen["prompt"]}
report["stage"] = "chosen"

# ------------------------------------------------------------------ button (before moving: clean view)
button = None
if press_button:
    bcands = []
    for prompt in ("button", "round button"):
        try:
            seg = segment_text(FRONT, prompt)
        except Exception:
            continue
        if not seg.ok:
            continue
        for c in seg.segmentations[:5]:
            try:
                cloud = mask_to_point_cloud(c, FRONT, target_frame=base_frame)
                pts = cloud.points
            except Exception:
                continue
            if len(pts) < 15:
                continue
            zs = [float(p[2]) for p in pts]
            zmax = quantile(zs, 0.98)
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            span = max(quantile(xs, 0.95) - quantile(xs, 0.05), quantile(ys, 0.95) - quantile(ys, 0.05))
            capf = [p for p in pts if float(p[2]) >= zmax - 0.006]
            bx = quantile([p[0] for p in capf], 0.5)
            by = quantile([p[1] for p in capf], 0.5)
            colour_ok = True
            if ref is not None and c.box_xyxy is not None:
                b = [float(v) for v in c.box_xyxy]
                n_obj = 0
                n_all = 0
                for v in range(int(b[1]), int(b[3]) + 1, 2):
                    for u in range(int(b[0]), int(b[2]) + 1, 2):
                        if 0 <= u < W and 0 <= v < H:
                            n_all += 1
                            px = front.rgb[v][u]
                            if is_obj([int(px[0]), int(px[1]), int(px[2])], ref):
                                n_obj += 1
                colour_ok = n_all > 0 and n_obj < 0.3 * n_all
            entry = {"prompt": prompt, "score": round(float(c.score), 3), "zmax": round(zmax, 4),
                     "span": round(span, 4), "xy": [round(bx, 4), round(by, 4)], "colour_ok": colour_ok}
            far = True
            if chosen is not None and math.hypot(bx - chosen["xy"][0], by - chosen["xy"][1]) < 0.03:
                far = False
            entry["far_from_target"] = far
            entry["accepted"] = colour_ok and far and span <= 0.10 and 0.003 <= zmax <= 0.08
            bcands.append(entry)
    report["button_candidates"] = bcands
    best_b = None
    for entry in bcands:
        if entry["accepted"] and (best_b is None or entry["score"] > best_b["score"]):
            best_b = entry
    # the button is a low base with a raised cap; SAM may return either part. Among accepted
    # candidates within 4 cm of the best-scored one, press the highest (the cap).
    if best_b is not None:
        button = best_b
        for entry in bcands:
            if entry["accepted"] and math.hypot(entry["xy"][0] - best_b["xy"][0],
                                                entry["xy"][1] - best_b["xy"][1]) < 0.04 \
                    and entry["zmax"] > button["zmax"]:
                button = entry
    report["button"] = button
report["stage"] = "button_located"

# ------------------------------------------------------------------ act
def hold(ticks, grip):
    st = get_robot_state()
    for tick in range(ticks):
        r = step(RobotAction(arms={"primary": ArmCommand(
            "joint_position", st.joint_positions["primary"],
            gripper_position=grip, embodiment=st.embodiment)}))
        if is_terminal(r):
            ended["flag"] = True
            log.append({"call": "hold", "tick": tick, "terminal": terminal_of(r)})
            return False
    return True


if chosen is None:
    report["stage"] = "no_target"
else:
    grasp_q = home_q
    if pick is not None:
        grasp_q = [float(v) for v in pick["open_q"]]   # the video robot's yaw for this object
    cx = float(chosen["xy"][0])
    cy = float(chosen["xy"][1])
    top = float(chosen["top_z"])
    height = max(0.02, top - float(chosen["bot_z"]))
    grasp_z = top - min(0.02, 0.5 * height)
    report["grasp"] = {"xy": vec([cx, cy]), "z": round(grasp_z, 4), "q": vec(grasp_q)}
    cycles = repeat_count if repeat_count is not None else 1
    report["held_widths"] = []
    for cycle in range(cycles):
        if ended["flag"]:
            break
        record("open", open_gripper())
        if ended["flag"]:
            break
        record("pre_grasp", move_to_pose(Pose((cx, cy, top + PRE_DZ), grasp_q, base_frame),
                                         tolerance=0.015, max_steps=200))
        if ended["flag"]:
            break
        record("descend", move_to_pose(Pose((cx, cy, grasp_z), grasp_q, base_frame),
                                       tolerance=0.012, max_steps=150))
        if ended["flag"]:
            break
        e = record("close", close_gripper())
        report["held_widths"].append(e.get("grip"))
        if ended["flag"]:
            break
        record("lift", move_to_pose(Pose((cx, cy, grasp_z + LIFT_DZ), grasp_q, base_frame),
                                    tolerance=0.015, max_steps=150))
        if ended["flag"]:
            break
        if not hold(5, 0.0):
            break
        report["stage"] = "lifted_%d" % cycle
        if not put_down:
            break
        record("lower", move_to_pose(Pose((cx, cy, grasp_z + 0.008), grasp_q, base_frame),
                                     tolerance=0.012, max_steps=150))
        if ended["flag"]:
            break
        record("release", open_gripper())
        if ended["flag"]:
            break
        if not hold(5, 1.0):
            break
        record("retreat", move_to_pose(Pose((cx, cy, top + PRE_DZ), grasp_q, base_frame),
                                       tolerance=0.015, max_steps=150))
        if ended["flag"]:
            break
        report["stage"] = "cycle_done_%d" % cycle

    if press_button and button is not None and not ended["flag"]:
        bx = button["xy"][0]
        by = button["xy"][1]
        bz = button["zmax"]
        record("close_for_press", close_gripper())
        if not ended["flag"]:
            record("press_hover", move_to_pose(Pose((bx, by, bz + 0.06), home_q, base_frame),
                                               tolerance=0.015, max_steps=200))
        if not ended["flag"]:
            record("press_down", move_to_pose(Pose((bx, by, bz - 0.015), home_q, base_frame),
                                              tolerance=0.01, max_steps=60))
        if not ended["flag"]:
            hold(5, 0.0)
        if not ended["flag"]:
            record("press_up", move_to_pose(Pose((bx, by, bz + 0.06), home_q, base_frame),
                                            tolerance=0.015, max_steps=100))
        report["stage"] = "pressed"
    report["ended_by_env"] = ended["flag"]

result = report
