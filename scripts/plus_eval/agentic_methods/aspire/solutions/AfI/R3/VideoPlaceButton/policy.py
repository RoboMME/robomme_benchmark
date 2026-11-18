# v03_ring_blobs -- VideoPlaceButton (robomme_plus task 11).  UNTESTED (0 simulator runs).
# Minimal repair of v02 (diagnosed in v02's ep-0 trace: swap_tracking.live_rule0 = [1 slot
# found, 1 off-slot blob] -> rule None -> fallback to the static slot -> wrong target again).
# On lossless frames each target is concentric pink (201,96,132) / white (220,215,216) rings
# with a darker pink rim (128,61,84); at stride 2 the pink rings are separated by white rings,
# so 4-connected blobs fragment below 8 cells. Repair: (1) pink test r>g+30, b>g+12, b>70
# (keeps the dark rim); (2) connect pink cells within 2 grid cells (4 px) so the rings of one
# target join into one blob. Nothing else changed.
#
# v02 header follows.
# Change from v01 (diagnosed in v01's ep-0 rollout): the four identical targets are SWAPPED
# pairwise many times during the conditioning video (each swap ~14 sampled frames, the two
# targets orbit their midpoint). The referenced target is a physical object, so its identity
# is followed from the referenced release to the end of the video: slots = live target
# pixel centres; a swap event = a run of frames with a pink blob off every slot; the swapped
# pair = the two slots most often vacant during the event. Final slot -> live target -> place.
#
# v01 header follows.
# Changes from v00 (each addresses a failure diagnosed in v00's ep-0 trace):
#   * identity: pick the cube whose observed colour matches the colour word in the goal
#     (v00 took the top "cube" score = blue cube; lifting it terminated the episode);
#   * video events: a button press is any EE dip within 4 cm (xy) of the live button,
#     whatever the gripper does (v00 required "no gripper transition"; the press closed the
#     empty jaw first, so it was missed and the choice fell back to the last release);
#   * placements are releases whose xy lies within 4 cm of a live target (the final
#     put-back to the cube's start spot is not a placement); "first"/"last" + before/after;
#   * grasp yaw aligned to the cube's top face (min-area rectangle of its top points).
#
# v00 header follows.
#
# Start 1 of 3 (AfI). Records what a scene snapshot would (goal, metadata, calibration,
# conditioning-video robot-state timeline, video thumbnails, live prompt probes with 3-D
# statistics) AND attempts the task with a hypothesis-driven policy:
#   * conditioning video: detect gripper close/open transitions and EE-height dips from the
#     recorded robot state of each video frame (public Observation.robot_state);
#   * opening transitions = cube placements (EE xy/z at release = target location/height);
#     dips with no gripper transition near the live-perceived button = button presses;
#   * resolve the referenced placement from the goal language ("before"/"after" the button,
#     ordinals), snap it to a live-perceived target, pick the live cube and place it there.
# Every task-dependent position comes from this episode's perception or its conditioning video.

import math

report = {"version": "v03_ring_blobs", "log": [], "motions": [], "errors": []}
result = report


def log(msg):
    report["log"].append(str(msg))


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def r4(v):
    return round(float(v), 4)


def vec(v):
    return [r4(a) for a in v]


def is_terminal(res):
    flag = False
    try:
        flag = bool(res.terminated)
    except Exception:
        flag = False
    try:
        flag = flag or bool(res.truncated)
    except Exception:
        flag = flag
    return flag


# ---------------------------------------------------------------- context / metadata
ctx = get_task_context()
lang = str(ctx.language)
report["task_language"] = lang
report["task_name"] = str(ctx.task_name)
max_steps = None
try:
    tmeta = robomme_plus.get_task_metadata()
    report["task_metadata"] = str(dict(tmeta))
    try:
        max_steps = int(tmeta["max_steps"])
    except Exception:
        max_steps = None
except Exception as exc:
    report["errors"].append("task_metadata: " + str(exc))
report["max_steps"] = max_steps
try:
    report["controller_metadata"] = str(dict(robomme_plus.get_controller_metadata()))
except Exception as exc:
    report["errors"].append("controller_metadata: " + str(exc))

state0 = get_robot_state()
bf = state0.base_frame
emb = state0.embodiment
home_pos = [float(v) for v in state0.end_effector_poses["primary"].position]
home_q = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]
report["base_frame"] = str(bf)
report["embodiment"] = str(emb)
report["home_ee"] = {"pos": vec(home_pos), "q": vec(home_q),
                     "gripper": r4(state0.gripper_positions["primary"]),
                     "joints": vec(state0.joint_positions["primary"])}

obs0 = get_observation()
report["cameras"] = {}
for cam_name in obs0.cameras:
    cam = obs0.cameras[cam_name]
    entry = {}
    try:
        entry["shape"] = [len(cam.rgb), len(cam.rgb[0])]
        entry["intrinsics"] = [[r4(v) for v in row] for row in cam.intrinsics]
        entry["pose_pos"] = vec(cam.camera_pose.position)
        entry["pose_q"] = vec(cam.camera_pose.quaternion_wxyz)
        entry["pose_frame"] = str(cam.camera_pose.frame)
    except Exception as exc:
        entry["error"] = str(exc)
    report["cameras"][cam_name] = entry


def thumb(rgb, stride):
    try:
        sub = rgb[::stride, ::stride].tolist()
    except Exception:
        sub = []
        for yy in range(0, len(rgb), stride):
            row = []
            for xx in range(0, len(rgb[0]), stride):
                row.append([int(c) for c in rgb[yy][xx]])
            sub.append(row)
    flat = []
    for row in sub:
        for px in row:
            for c in px:
                flat.append(int(c))
    return flat


def region_mean(rgb, box, stride):
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = max(x1 + 1, int(box[2]))
    y2 = max(y1 + 1, int(box[3]))
    try:
        sub = rgb[y1:y2:stride, x1:x2:stride].tolist()
    except Exception:
        sub = []
        for yy in range(y1, y2, stride):
            sub.append([[int(c) for c in rgb[yy][xx]] for xx in range(x1, x2, stride)])
    tot = [0.0, 0.0, 0.0]
    cnt = 0
    for row in sub:
        for px in row:
            tot[0] += px[0]
            tot[1] += px[1]
            tot[2] += px[2]
            cnt += 1
    if cnt == 0:
        return [0, 0, 0]
    return [int(tot[0] / cnt), int(tot[1] / cnt), int(tot[2] / cnt)]


report["live_front_thumb16"] = thumb(obs0.cameras["front"].rgb, 4)   # 64x64

# ---------------------------------------------------------------- conditioning video
frames = []
try:
    video = robomme_plus.get_conditioning_video()
    for fr in video:
        frames.append(fr)
except Exception as exc:
    report["errors"].append("video: " + str(exc))
nv = len(frames)
report["video_len"] = nv
vt = []
vx = []
vy = []
vz = []
vg = []
vq = []
for fr in frames:
    rs = fr.robot_state
    p = rs.end_effector_poses["primary"].position
    vt.append(float(fr.timestamp_s))
    vx.append(float(p[0]))
    vy.append(float(p[1]))
    vz.append(float(p[2]))
    vg.append(float(rs.gripper_positions["primary"]))
    vq.append([float(a) for a in rs.end_effector_poses["primary"].quaternion_wxyz])
report["video_timeline"] = {"t": [round(v, 3) for v in vt], "x": [r4(v) for v in vx],
                            "y": [r4(v) for v in vy], "z": [r4(v) for v in vz],
                            "g": [r4(v) for v in vg]}
if nv > 0:
    try:
        c0 = frames[0].cameras["front"]
        report["video_front_pose0"] = {"pos": vec(c0.camera_pose.position),
                                       "q": vec(c0.camera_pose.quaternion_wxyz)}
    except Exception as exc:
        report["errors"].append("video cam pose: " + str(exc))

# motion energy per frame on a stride-16 grid
energy = []
prev = None
for fr in frames:
    cur = thumb(fr.cameras["front"].rgb, 16)
    if prev is None:
        energy.append(0.0)
    else:
        s = 0
        for k in range(len(cur)):
            s += abs(cur[k] - prev[k])
        energy.append(round(s / max(1, len(cur)), 2))
    prev = cur
report["video_energy"] = energy

# ---------------------------------------------------------------- video events
def idx_window(i, t_lo, t_hi):
    out = []
    for j in range(nv):
        if vt[j] >= vt[i] + t_lo and vt[j] <= vt[i] + t_hi:
            out.append(j)
    return out


transitions = []
if nv > 1:
    gmax = max(vg)
    gmin = min(vg)
    grange = gmax - gmin
    report["video_gripper_range"] = [r4(gmin), r4(gmax)]
    if grange > 0.1:
        close_thr = gmax - 0.25 * grange
        open_thr = gmax - 0.08 * grange
        is_open = vg[0] > close_thr
        for i in range(1, nv):
            if is_open and vg[i] < close_thr:
                is_open = False
                transitions.append({"kind": "close", "i": i})
            elif (not is_open) and vg[i] > open_thr:
                is_open = True
                transitions.append({"kind": "open", "i": i})
for tr in transitions:
    i = tr["i"]
    # the EE is near the bottom of its dwell when the jaw moves; take the lowest EE frame
    # within +-1 s as the event pose
    win = idx_window(i, -1.0, 1.0)
    jbest = i
    for j in win:
        if vz[j] < vz[jbest]:
            jbest = j
    tr["t"] = round(vt[i], 3)
    tr["pos"] = [r4(vx[jbest]), r4(vy[jbest]), r4(vz[jbest])]
    tr["q"] = vec(vq[jbest])
    post = [vg[j] for j in idx_window(i, 0.3, 1.5)]
    tr["g_after"] = r4(quantile(post, 0.5)) if len(post) > 0 else None

dips = []
for i in range(nv):
    near = idx_window(i, -0.75, 0.75)
    if min(vz[j] for j in near) < vz[i]:
        continue
    before = [vz[j] for j in idx_window(i, -4.0, 0.0)]
    after = [vz[j] for j in idx_window(i, 0.0, 4.0)]
    if max(before) - vz[i] < 0.025 or max(after) - vz[i] < 0.025:
        continue
    if len(dips) > 0 and vt[i] - dips[-1]["t"] < 1.0:
        if vz[i] < dips[-1]["pos"][2]:
            dips[-1] = {"i": i, "t": round(vt[i], 3), "pos": [r4(vx[i]), r4(vy[i]), r4(vz[i])]}
        continue
    dips.append({"i": i, "t": round(vt[i], 3), "pos": [r4(vx[i]), r4(vy[i]), r4(vz[i])]})
for dp in dips:
    dp["g"] = r4(vg[dp["i"]])
    dp["transition"] = None
    for tr in transitions:
        if abs(tr["t"] - dp["t"]) <= 1.5:
            dp["transition"] = tr["kind"]
report["video_transitions"] = transitions
report["video_dips"] = dips

# thumbnails of the video (32x32) at event frames and a uniform sample
thumb_idx = []
if nv > 0:
    stride_f = max(1, nv // 24)
    for i in range(0, nv, stride_f):
        thumb_idx.append(i)
    thumb_idx.append(nv - 1)
    for ev in transitions + dips:
        thumb_idx.append(ev["i"])
thumb_idx = sorted(set(thumb_idx))
report["video_thumbs"] = []
for i in thumb_idx:
    report["video_thumbs"].append({"i": i, "t": round(vt[i], 3),
                                   "front": thumb(frames[i].cameras["front"].rgb, 8)})

# ---------------------------------------------------------------- live perception
def cand_mask(cand):
    try:
        return cand.mask
    except Exception:
        return cand


def cloud_points(pc):
    try:
        pts = pc.points.tolist()
    except Exception:
        pts = [[float(a) for a in p] for p in pc.points]
    return pts


def cloud_stats(pts):
    if len(pts) < 8:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    return {"n": len(pts), "cx": r4(quantile(xs, 0.5)), "cy": r4(quantile(ys, 0.5)),
            "z05": r4(quantile(zs, 0.05)), "z50": r4(quantile(zs, 0.5)),
            "z95": r4(quantile(zs, 0.95)),
            "sx": r4(quantile(xs, 0.95) - quantile(xs, 0.05)),
            "sy": r4(quantile(ys, 0.95) - quantile(ys, 0.05))}


def cloud_color(pc):
    try:
        cols = pc.colors.tolist()
    except Exception:
        return None
    if len(cols) == 0:
        return None
    step_c = max(1, len(cols) // 200)
    tot = [0.0, 0.0, 0.0]
    cnt = 0
    for k in range(0, len(cols), step_c):
        tot[0] += float(cols[k][0])
        tot[1] += float(cols[k][1])
        tot[2] += float(cols[k][2])
        cnt += 1
    return [round(tot[0] / cnt, 3), round(tot[1] / cnt, 3), round(tot[2] / cnt, 3)]


def probe(prompt, cam_name):
    out = {"prompt": prompt, "camera": cam_name, "cands": []}
    seg = segment_text(cam_name, prompt)
    out["ok"] = bool(seg.ok)
    if not seg.ok:
        return out
    for cand in seg.segmentations[:5]:
        c = {"score": r4(cand.score)}
        try:
            c["box"] = [r4(v) for v in cand.box_xyxy]
        except Exception:
            c["box"] = None
        try:
            pc = mask_to_point_cloud(cand_mask(cand), cam_name, target_frame=bf)
            pts = cloud_points(pc)
            c["stats"] = cloud_stats(pts)
            c["color"] = cloud_color(pc)
            if prompt == "cube" and c["stats"] is not None:
                ztop = c["stats"]["z95"]
                top = [[r4(p[0]), r4(p[1])] for p in pts if p[2] > ztop - 0.006]
                c["top_xy"] = top[:300]
        except Exception as exc:
            c["stats"] = None
            c["err"] = str(exc)
        out["cands"].append(c)
    return out


PROMPTS = ["cube", "button", "target", "disc", "round button", "table"]
probes = []
for pr in PROMPTS:
    try:
        probes.append(probe(pr, "front"))
    except Exception as exc:
        report["errors"].append("probe " + pr + ": " + str(exc))
report["probes"] = probes

table_z = None
for pb in probes:
    if pb["prompt"] == "table":
        for c in pb["cands"]:
            if c.get("stats") is not None and c["stats"]["n"] > 200:
                table_z = c["stats"]["z50"]
                break
if table_z is None:
    lows = []
    for pb in probes:
        for c in pb["cands"]:
            if c.get("stats") is not None:
                lows.append(c["stats"]["z05"])
    table_z = min(lows) if len(lows) > 0 else None
report["table_z"] = table_z

# group candidates from object prompts by xy proximity
groups = []
for pb in probes:
    if pb["prompt"] == "table":
        continue
    for c in pb["cands"]:
        st = c.get("stats")
        if st is None:
            continue
        if st["sx"] > 0.15 or st["sy"] > 0.15:
            continue
        hit = None
        for gr in groups:
            if math.hypot(gr["cx"] - st["cx"], gr["cy"] - st["cy"]) < 0.025:
                hit = gr
                break
        if hit is None:
            hit = {"cx": st["cx"], "cy": st["cy"], "z95": st["z95"], "z05": st["z05"],
                   "sx": st["sx"], "sy": st["sy"], "scores": {}, "box": c.get("box"),
                   "color": c.get("color"), "top_xy": None}
            groups.append(hit)
        if hit["top_xy"] is None and c.get("top_xy") is not None:
            hit["top_xy"] = c["top_xy"]
        old = hit["scores"].get(pb["prompt"], 0.0)
        if c["score"] > old:
            hit["scores"][pb["prompt"]] = c["score"]
for gr in groups:
    gr["height"] = r4(gr["z95"] - table_z) if table_z is not None else None
report["groups"] = groups


def best_group(prompt_list, hmin, hmax, exclude):
    best = None
    best_s = 0.0
    for gr in groups:
        if gr in exclude:
            continue
        if gr["height"] is not None and (gr["height"] < hmin or gr["height"] > hmax):
            continue
        s = 0.0
        for pr in prompt_list:
            s = max(s, gr["scores"].get(pr, 0.0))
        if s > best_s:
            best_s = s
            best = gr
    return best


COLOURS = {"red": [1.0, 0.0, 0.0], "green": [0.0, 1.0, 0.0], "blue": [0.0, 0.0, 1.0],
           "yellow": [1.0, 1.0, 0.0], "purple": [0.6, 0.0, 1.0], "orange": [1.0, 0.5, 0.0],
           "cyan": [0.0, 1.0, 1.0], "pink": [1.0, 0.4, 0.7], "magenta": [1.0, 0.0, 1.0]}


def chroma(col):
    # normalise a mean colour by its max channel (brightness-free); None if grey/unknown
    if col is None:
        return None
    m = max(col[0], col[1], col[2])
    n = min(col[0], col[1], col[2])
    if m <= 0 or (m - n) < 0.35 * m:
        return None
    return [col[0] / m, col[1] / m, col[2] / m]


low = lang.lower()
colour_word = None
cpos = low.find("cube")
best_pos = -1
for cname in COLOURS:
    p = low.rfind(cname, 0, cpos if cpos >= 0 else len(low))
    if p > best_pos:
        best_pos = p
        colour_word = cname
report["colour_word"] = colour_word

cube_cands = []
for gr in groups:
    if gr["scores"].get("cube", 0.0) <= 0.3:
        continue
    if gr["height"] is not None and (gr["height"] < 0.025 or gr["height"] > 0.07):
        continue
    if gr["sx"] > 0.06 or gr["sy"] > 0.06:
        continue
    if chroma(gr["color"]) is None:
        continue                      # cubes are saturated; the grey button is not
    cube_cands.append(gr)
cube_g = None
best_d = None
for gr in cube_cands:
    ch = chroma(gr["color"])
    gr["chroma"] = ch
    if colour_word is None:
        d = -gr["scores"]["cube"]
    elif ch is None:
        continue
    else:
        ref = COLOURS[colour_word]
        d = math.sqrt(sum((ch[k] - ref[k]) ** 2 for k in range(3)))
        if d > 0.5:
            continue
    if best_d is None or d < best_d:
        best_d = d
        cube_g = gr
report["cube_colour_distance"] = best_d

button_g = None
bs = 0.0
for gr in groups:
    if gr is cube_g or gr in cube_cands:
        continue
    if gr["height"] is not None and (gr["height"] < 0.012 or gr["height"] > 0.08):
        continue
    s = max(gr["scores"].get("button", 0.0), gr["scores"].get("round button", 0.0))
    if s > bs:
        bs = s
        button_g = gr
targets = []
for gr in groups:
    if gr is cube_g or gr is button_g or gr in cube_cands:
        continue
    if gr["height"] is not None and gr["height"] > 0.02:
        continue
    if gr["scores"].get("target", 0.0) > 0 or gr["scores"].get("disc", 0.0) > 0:
        targets.append(gr)
report["cube_group"] = cube_g
report["button_group"] = button_g
report["target_groups"] = targets

# bbox colour timelines through the video (button and targets): does the button light up?
report["video_box_colors"] = []
watch = []
if button_g is not None and button_g.get("box") is not None:
    watch.append(("button", button_g["box"]))
for k in range(len(targets)):
    if targets[k].get("box") is not None:
        watch.append(("target" + str(k), targets[k]["box"]))
if cube_g is not None and cube_g.get("box") is not None:
    watch.append(("cube", cube_g["box"]))
for wname, box in watch:
    series = []
    for fr in frames:
        series.append(region_mean(fr.cameras["front"].rgb, box, 2))
    report["video_box_colors"].append({"name": wname, "box": box, "rgb": series})

# ---------------------------------------------------------------- resolve the referenced target
def nearest_target(xy):
    best = None
    bd = 1.0
    for gr in targets:
        d = math.hypot(gr["cx"] - xy[0], gr["cy"] - xy[1])
        if d < bd:
            bd = d
            best = gr
    return best, bd


all_releases = [tr for tr in transitions if tr["kind"] == "open"]
grasps = [tr for tr in transitions if tr["kind"] == "close"]
presses = []
for dp in dips:
    if button_g is not None:
        if math.hypot(dp["pos"][0] - button_g["cx"], dp["pos"][1] - button_g["cy"]) < 0.04:
            presses.append(dp)
    elif dp["transition"] is None:
        presses.append(dp)
releases = []
for tr in all_releases:
    if button_g is not None and math.hypot(tr["pos"][0] - button_g["cx"],
                                           tr["pos"][1] - button_g["cy"]) < 0.04:
        continue
    ng, nd = nearest_target(tr["pos"])
    tr["target_dist"] = r4(nd)
    if ng is not None and nd < 0.04:
        releases.append(tr)
report["video_all_releases"] = all_releases
report["video_releases"] = releases
report["video_presses"] = presses

rel_word = None
bpos = low.find("button")
if bpos >= 0:
    pre = low[max(0, bpos - 60):bpos]
    if pre.rfind("after") > pre.rfind("before"):
        rel_word = "after"
    elif pre.rfind("before") > pre.rfind("after"):
        rel_word = "before"
report["relation_word"] = rel_word
want_last = low.find("last") >= 0 and low.find("first") < 0
report["want_last"] = want_last

choice = None
reason = ""
if len(releases) > 0:
    if len(presses) > 0:
        tp = presses[0]["t"]
        before_l = [r for r in releases if r["t"] < tp]
        after_l = [r for r in releases if r["t"] > tp]
        if rel_word == "after" and len(after_l) > 0:
            choice = after_l[-1] if want_last else after_l[0]
            reason = "release after first press (last=" + str(want_last) + ")"
        elif rel_word == "before" and len(before_l) > 0:
            choice = before_l[0] if (low.find("first") >= 0) else before_l[-1]
            reason = "release before first press"
        elif len(before_l) > 0:
            choice = before_l[-1]
            reason = "last release before first press"
        elif len(after_l) > 0:
            choice = after_l[0]
            reason = "fallback: first release after press"
    if choice is None:
        choice = releases[-1]
        reason = "fallback: last release"
report["choice"] = choice
report["choice_reason"] = reason

target_xy = None
place_z = None
if choice is not None:
    target_xy = [choice["pos"][0], choice["pos"][1]]
    place_z = choice["pos"][2] + 0.01
    nearest, nd = nearest_target(target_xy)
    report["snap_distance"] = r4(nd)
    if nearest is not None and nd < 0.05:
        target_xy = [nearest["cx"], nearest["cy"]]
        report["snapped_to"] = nearest
report["target_xy_static"] = target_xy


# ---------------------------------------------------------------- follow target swaps (v02)
def pink(px, rule):
    rr = px[0]
    gg = px[1]
    bb = px[2]
    if rule == 0:
        return rr > gg + 30 and bb > gg + 12 and bb > 70
    return rr > gg + 25 and bb > gg + 8 and bb > 60


RING_NEIGHBOURS = [[dy, dx] for dy in (-2, -1, 0, 1, 2) for dx in (-2, -1, 0, 1, 2)
                   if not (dy == 0 and dx == 0)]


def frame_blobs(rgb, region, rule):
    x0 = region[0]
    y0 = region[1]
    sub = rgb[y0:region[3]:2, x0:region[2]:2].tolist()
    hh = len(sub)
    ww = len(sub[0]) if hh > 0 else 0
    mask = [[pink(sub[yy][xx], rule) for xx in range(ww)] for yy in range(hh)]
    seen = [[False] * ww for yy in range(hh)]
    out = []
    for yy in range(hh):
        for xx in range(ww):
            if mask[yy][xx] and not seen[yy][xx]:
                stack = [[yy, xx]]
                seen[yy][xx] = True
                sx = 0
                sy = 0
                n = 0
                while len(stack) > 0:
                    cell = stack.pop()
                    cy = cell[0]
                    cx = cell[1]
                    sx += cx
                    sy += cy
                    n += 1
                    for d in RING_NEIGHBOURS:
                        ny = cy + d[0]
                        nx = cx + d[1]
                        if ny >= 0 and ny < hh and nx >= 0 and nx < ww and mask[ny][nx] and not seen[ny][nx]:
                            seen[ny][nx] = True
                            stack.append([ny, nx])
                if n >= 8:
                    out.append([x0 + 2.0 * sx / n, y0 + 2.0 * sy / n, n])
    return out


def slot_hits(bl, slots, near):
    present = []
    off = 0
    for b in bl:
        dmin = min(math.hypot(b[0] - q[0], b[1] - q[1]) for q in slots)
        if dmin > near:
            off += 1
    for q in slots:
        hit = False
        for b in bl:
            if math.hypot(b[0] - q[0], b[1] - q[1]) <= near:
                hit = True
        present.append(hit)
    return present, off


track = {"ok": False}
report["swap_tracking"] = track
try:
    slots = []
    for gr in targets:
        bx = gr.get("box")
        slots.append([(bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0])
    track["slots_px"] = [[round(q[0], 1), round(q[1], 1)] for q in slots]
    if choice is not None and len(slots) >= 2 and nv > 0:
        near = 6.0
        rx0 = max(0, int(min(q[0] for q in slots)) - 30)
        ry0 = max(0, int(min(q[1] for q in slots)) - 30)
        rx1 = min(256, int(max(q[0] for q in slots)) + 30)
        ry1 = min(256, int(max(q[1] for q in slots)) + 30)
        rx0 = rx0 - rx0 % 2
        ry0 = ry0 - ry0 % 2
        region = [rx0, ry0, rx1, ry1]
        track["region"] = region
        # pick the colour rule that finds every slot (and nothing else) in the live image
        rule = None
        live_rgb = obs0.cameras["front"].rgb
        for cand_rule in (0, 1):
            pres, off = slot_hits(frame_blobs(live_rgb, region, cand_rule), slots, near)
            track["live_rule" + str(cand_rule)] = [sum(1 for p in pres if p), off]
            if rule is None and all(pres) and off == 0:
                rule = cand_rule
        track["rule"] = rule
        k0 = None
        bdist = 1.0
        for k in range(len(targets)):
            d = math.hypot(targets[k]["cx"] - choice["pos"][0], targets[k]["cy"] - choice["pos"][1])
            if d < bdist:
                bdist = d
                k0 = k
        track["start_slot"] = k0
        if rule is not None and k0 is not None and bdist < 0.04:
            i_rel = choice["i"]
            per = []
            for i in range(i_rel + 1, nv):
                pres, off = slot_hits(frame_blobs(frames[i].cameras["front"].rgb, region, rule), slots, near)
                per.append([i, off > 0, [k for k in range(len(slots)) if not pres[k]]])
            evs = []
            cur = None
            last = -10
            for row in per:
                if row[1]:
                    if cur is not None and row[0] - last <= 3:
                        cur["end"] = row[0]
                    else:
                        if cur is not None:
                            evs.append(cur)
                        cur = {"start": row[0], "end": row[0]}
                    last = row[0]
            if cur is not None:
                evs.append(cur)
            swaps = []
            for ev in evs:
                if ev["end"] - ev["start"] + 1 < 3:
                    continue
                cnt = [0] * len(slots)
                for row in per:
                    if row[0] >= ev["start"] and row[0] <= ev["end"]:
                        for k in row[2]:
                            cnt[k] += 1
                order = [pk[1] for pk in sorted([[-cnt[k], k] for k in range(len(slots))])]
                evlen = ev["end"] - ev["start"] + 1
                if cnt[order[1]] < max(3, 0.4 * evlen):
                    continue          # one vacant slot only: arm occlusion, not a swap
                swaps.append({"start": ev["start"], "end": ev["end"], "counts": cnt,
                              "pair": [order[0], order[1]]})
            chain = [k0]
            curk = k0
            for sw in swaps:
                if curk == sw["pair"][0]:
                    curk = sw["pair"][1]
                    chain.append(curk)
                elif curk == sw["pair"][1]:
                    curk = sw["pair"][0]
                    chain.append(curk)
            track["swaps"] = swaps
            track["chain"] = chain
            track["final_slot"] = curk
            track["ok"] = True
            target_xy = [targets[curk]["cx"], targets[curk]["cy"]]
except Exception as exc:
    report["errors"].append("swap tracking: " + str(exc))
    track["ok"] = False
report["target_xy"] = target_xy
report["place_z"] = place_z


# ---------------------------------------------------------------- motion
def is_down(q):
    w = q[0]
    x = q[1]
    y = q[2]
    zq = q[3]
    return 1.0 - 2.0 * (x * x + y * y) < -0.9


down_q = home_q
if not is_down(home_q):
    if len(grasps) > 0 and is_down(grasps[0]["q"]):
        down_q = grasps[0]["q"]
    else:
        down_q = [0.0, 1.0, 0.0, 0.0]


def qmul(a, b):
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]


def top_yaw(pts):
    # face-normal yaw of a square top face: angle in [-45, 45) deg minimising the
    # bounding-rectangle area of the top-face points
    if pts is None or len(pts) < 15:
        return None
    best = None
    best_a = None
    for k in range(30):
        ang = math.radians(-45.0 + 3.0 * k)
        ca = math.cos(ang)
        sa = math.sin(ang)
        us = [p[0] * ca + p[1] * sa for p in pts]
        vs = [-p[0] * sa + p[1] * ca for p in pts]
        area = (quantile(us, 0.98) - quantile(us, 0.02)) * (quantile(vs, 0.98) - quantile(vs, 0.02))
        if best_a is None or area < best_a:
            best_a = area
            best = ang
    return best


cube_yaw = None
if cube_g is not None:
    cube_yaw = top_yaw(cube_g.get("top_xy"))
report["cube_yaw_deg"] = round(math.degrees(cube_yaw), 1) if cube_yaw is not None else None
if cube_yaw is not None:
    down_q = qmul([math.cos(cube_yaw / 2.0), 0.0, 0.0, math.sin(cube_yaw / 2.0)], down_q)
report["down_q"] = vec(down_q)


def ee_now():
    st = get_robot_state()
    return vec(st.end_effector_poses["primary"].position), r4(st.gripper_positions["primary"])


def mv(name, pos, tol, steps):
    res = move_to_pose(Pose([float(pos[0]), float(pos[1]), float(pos[2])], down_q, bf),
                       tolerance=tol, max_steps=steps)
    entry = {"name": name, "target": vec(pos), "ok": bool(res.ok)}
    try:
        entry["status"] = str(res.status)
        entry["steps"] = int(res.steps_executed)
    except Exception:
        entry["status"] = "?"
    entry["terminal"] = is_terminal(res)
    if not entry["terminal"]:
        now_pos, now_g = ee_now()
        entry["ee"] = now_pos
        entry["g"] = now_g
    report["motions"].append(entry)
    return res


def grip(name, closing):
    res = close_gripper() if closing else open_gripper()
    entry = {"name": name, "ok": bool(res.ok)}
    try:
        entry["status"] = str(res.status)
        entry["steps"] = int(res.steps_executed)
    except Exception:
        entry["status"] = "?"
    entry["terminal"] = is_terminal(res)
    if not entry["terminal"]:
        now_pos, now_g = ee_now()
        entry["ee"] = now_pos
        entry["g"] = now_g
    report["motions"].append(entry)
    return res


def hold(ticks, gpos):
    st = get_robot_state()
    jt = [float(v) for v in st.joint_positions["primary"]]
    for k in range(ticks):
        r = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=jt, gripper_position=gpos,
            embodiment=st.embodiment)}))
        if is_terminal(r):
            report["motions"].append({"name": "hold", "terminal": True, "tick": k})
            return True
    return False


def attempt():
    if cube_g is None:
        report["errors"].append("no cube found")
        return
    if target_xy is None:
        report["errors"].append("no target resolved")
        return
    cube_top = cube_g["z95"]
    grasp_z = cube_top - 0.02
    if table_z is not None:
        grasp_z = max(grasp_z, table_z + 0.012)
    report["grasp_z"] = r4(grasp_z)
    pz = place_z
    if pz is None:
        pz = cube_top + 0.01
    transit = max(cube_top + 0.10, pz + 0.10)
    report["transit_z"] = r4(transit)
    if is_terminal(grip("open", False)):
        return
    if is_terminal(mv("above_cube", [cube_g["cx"], cube_g["cy"], cube_top + 0.10], 0.01, 250)):
        return
    if is_terminal(mv("grasp", [cube_g["cx"], cube_g["cy"], grasp_z], 0.008, 200)):
        return
    if is_terminal(grip("close", True)):
        return
    if is_terminal(mv("lift", [cube_g["cx"], cube_g["cy"], transit], 0.015, 250)):
        return
    report["held_width"] = ee_now()[1]
    if is_terminal(mv("above_target", [target_xy[0], target_xy[1], transit], 0.01, 300)):
        return
    if is_terminal(mv("place", [target_xy[0], target_xy[1], pz], 0.008, 200)):
        return
    if is_terminal(grip("release", False)):
        return
    if hold(10, 1.0):
        return
    if is_terminal(mv("retreat", [target_xy[0], target_xy[1], pz + 0.10], 0.015, 200)):
        return
    hold(10, 1.0)


try:
    attempt()
except Exception as exc:
    report["errors"].append("attempt: " + str(exc))

try:
    fin_pos, fin_g = ee_now()
    report["final_ee"] = fin_pos
    report["final_g"] = fin_g
    report["final_t"] = r4(get_robot_state().timestamp_s)
except Exception as exc:
    report["errors"].append("final: " + str(exc))

result = report
