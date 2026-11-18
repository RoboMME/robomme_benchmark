import math

# VideoPlaceOrder v03 -- the targets SWAP places during the conditioning video: track target identities
# through the swaps (pixel blobs of the target colour, swap = two blobs point-symmetric about the midpoint
# of two vacated slots), map the ordinal placement to an identity, place on that identity's final slot.
# v02: v01 + unit-normalised quaternions (unrounded video pose), flat-target height gate.
# v01: colour-gated cube, held low releases as placements, video rest pose for the grasp.
# Strategy: read the episode's own conditioning video (robot state per frame) to find the
# ordered placement events (gripper close -> open transitions, fallback: low dwell points
# while closed), parse the ordinal from the language goal, localize the cube live with SAM3
# + depth, pick it top-down, and place it at the selected event's location (snapped to a
# live-segmented target when one is near).  Everything is derived at runtime.

report = {"version": "v04", "stage": "start", "errors": []}


def note_error(where, exc):
    report["errors"].append(where + ": " + str(exc)[:300])


def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def rnd(value, digits=4):
    return round(float(value), digits)


def is_over(res):
    flag = False
    try:
        flag = bool(res.terminated) or bool(res.truncated)
    except Exception:
        flag = False
    return flag


def res_summary(res):
    out = {}
    try:
        out["ok"] = bool(res.ok)
    except Exception:
        pass
    try:
        out["status"] = str(res.status)
    except Exception:
        pass
    try:
        out["steps"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        out["terminated"] = bool(res.terminated)
        out["truncated"] = bool(res.truncated)
    except Exception:
        pass
    try:
        if res.error is not None:
            out["error"] = str(res.error)[:300]
    except Exception:
        pass
    return out


motion_log = []
episode_over = [False]


def log_motion(label, res):
    entry = res_summary(res)
    entry["label"] = label
    try:
        entry["t"] = rnd(get_robot_state().timestamp_s, 3)
    except Exception:
        pass
    motion_log.append(entry)
    if is_over(res):
        episode_over[0] = True
    return res


def thumb(rgb, size):
    h = len(rgb)
    w = len(rgb[0])
    sy = max(1, h // size)
    sx = max(1, w // size)
    rows = []
    for yy in range(0, h, sy):
        row = []
        line = rgb[yy]
        for xx in range(0, w, sx):
            px = line[xx]
            row.append([int(px[0]), int(px[1]), int(px[2])])
        rows.append(row)
    return rows


# ---------------------------------------------------------------- context / metadata
context = get_task_context()
goal = str(context.language)
report["goal"] = goal
try:
    meta = robomme_plus.get_task_metadata()
    report["task_metadata"] = dict(meta)
except Exception as exc:
    meta = {}
    note_error("task_metadata", exc)
try:
    report["controller_metadata"] = dict(robomme_plus.get_controller_metadata())
except Exception as exc:
    note_error("controller_metadata", exc)

state0 = get_robot_state()
base_frame = state0.base_frame
home_pose = state0.end_effector_poses["primary"]
home_quat = [float(v) for v in home_pose.quaternion_wxyz]
report["home_ee"] = [rnd(v) for v in home_pose.position]
report["home_quat"] = [rnd(v) for v in home_quat]
report["start_gripper"] = rnd(state0.gripper_positions["primary"])
report["start_t"] = rnd(state0.timestamp_s, 3)

# ---------------------------------------------------------------- parse ordinal
ORDINALS = [
    ("first", 1), ("second", 2), ("third", 3), ("fourth", 4), ("fifth", 5),
    ("sixth", 6), ("seventh", 7), ("eighth", 8), ("1st", 1), ("2nd", 2), ("3rd", 3),
    ("4th", 4), ("5th", 5), ("6th", 6),
]
words = goal.lower().replace(",", " ").replace(".", " ").replace("-", " ").split()
ordinal = None
from_end = False
for index, word in enumerate(words):
    for name, value in ORDINALS:
        if word == name and ordinal is None:
            ordinal = value
            tail = " ".join(words[index + 1:index + 4])
            if tail.startswith("to last") or tail.startswith("from last") or tail.startswith("last"):
                from_end = True
if ordinal is None:
    for word in words:
        if word in ("last", "final", "latest"):
            ordinal = 1
            from_end = True
report["ordinal"] = ordinal
report["from_end"] = from_end

# ---------------------------------------------------------------- conditioning video
video = []
try:
    video = robomme_plus.get_conditioning_video()
except Exception as exc:
    note_error("conditioning_video", exc)
n_frames = len(video)
report["video_frames"] = n_frames

timeline = []
for index in range(n_frames):
    frame = video[index]
    rs = frame.robot_state
    p = rs.end_effector_poses["primary"].position
    timeline.append([
        index, rnd(frame.timestamp_s, 3), rnd(p[0]), rnd(p[1]), rnd(p[2]),
        rnd(rs.gripper_positions["primary"], 3),
    ])
report["video_timeline"] = timeline

releases = []
grasps = []
dwell_points = []
carry_runs = []
if n_frames > 2:
    gs = [row[5] for row in timeline]
    gmax = max(gs)
    gmin = min(gs)
    report["video_gripper_range"] = [gmin, gmax]
    closed = [False for g in gs]
    if gmax - gmin > 0.15:
        mid = 0.5 * (gmax + gmin)
        closed = [g < mid for g in gs]
    run_start = None
    for index in range(n_frames + 1):
        is_closed = index < n_frames and closed[index]
        if is_closed and run_start is None:
            run_start = index
        if (not is_closed) and run_start is not None:
            run_end = index - 1
            levels = [timeline[j][5] for j in range(run_start, run_end + 1)]
            level = quantile(levels, 0.5)
            carry_runs.append({"start": run_start, "end": run_end, "level": rnd(level, 3),
                               "complete": index < n_frames})
            run_start = None
    # a carry run holds something between the fingers (partially closed reading); a run that
    # closes fully (reading near 0) is an empty-gripper action such as a button press.
    for run in carry_runs:
        if run["level"] <= 0.15:
            continue
        s = timeline[run["start"]]
        grasps.append({"frame": run["start"], "t": s[1], "xyz": [s[2], s[3], s[4]],
                       "quat": [rnd(v) for v in video[run["start"]].robot_state.end_effector_poses["primary"].quaternion_wxyz]})
        if run["complete"]:
            e = timeline[run["end"]]
            releases.append({"frame": run["end"], "t": e[1], "xyz": [e[2], e[3], e[4]],
                             "quat": [rnd(v) for v in video[run["end"]].robot_state.end_effector_poses["primary"].quaternion_wxyz]})
report["video_carry_runs"] = carry_runs
report["video_grasps"] = grasps
report["video_releases"] = releases
# low releases only: put down near the pick height rather than dropped from the air
if len(grasps) > 0:
    pick_z = grasps[0]["xyz"][2]
    low = []
    for ev in releases:
        if ev["xyz"][2] <= pick_z + 0.05:
            low.append(ev)
        else:
            ev["high"] = True
    releases = low

# thumbnails of the video front camera (for offline inspection) at events and evenly spaced
thumb_frames = []
if n_frames > 0:
    for k in range(8):
        thumb_frames.append(int(round(k * (n_frames - 1) / 7.0)))
    for ev in releases + grasps:
        thumb_frames.append(ev["frame"])
thumbs = {}
for index in sorted(set(thumb_frames))[:24]:
    try:
        thumbs[str(index)] = thumb(video[index].cameras["front"].rgb, 64)
    except Exception as exc:
        note_error("thumb", exc)
report["video_front_thumbs"] = thumbs

# ---------------------------------------------------------------- live perception
live = get_observation()
try:
    report["live_front_thumb"] = thumb(live.cameras["front"].rgb, 64)
except Exception as exc:
    note_error("live_thumb", exc)


def mean_color(cloud):
    try:
        cols = cloud.colors
    except Exception:
        return None
    if cols is None or len(cols) == 0:
        return None
    sr = 0.0
    sg = 0.0
    sb = 0.0
    for c in cols:
        sr += float(c[0])
        sg += float(c[1])
        sb += float(c[2])
    n = float(len(cols))
    mr = sr / n
    mg = sg / n
    mb = sb / n
    top = max(mr, mg, mb, 1e-6)
    scale = 255.0 if top <= 1.0 else 1.0
    return [rnd(mr * scale, 1), rnd(mg * scale, 1), rnd(mb * scale, 1)]


def saturated_color(cloud):
    try:
        cols = cloud.colors
    except Exception:
        return None
    if cols is None or len(cols) == 0:
        return None
    top = 0.0
    for c in cols[:50]:
        top = max(top, float(c[0]), float(c[1]), float(c[2]))
    scale = 255.0 if top <= 1.0 else 1.0
    sr = 0.0
    sg = 0.0
    sb = 0.0
    n = 0
    for c in cols:
        r = float(c[0]) * scale
        g = float(c[1]) * scale
        b = float(c[2]) * scale
        if max(r, g, b) - min(r, g, b) > 60:
            sr += r
            sg += g
            sb += b
            n += 1
    if n < 10:
        return None
    return [rnd(sr / n, 1), rnd(sg / n, 1), rnd(sb / n, 1), n]


def color_matches(word, rgb):
    if rgb is None:
        return None
    r, g, b = rgb
    if word == "red":
        return r > 1.5 * g and r > 1.5 * b
    if word == "green":
        return g > 1.3 * r and g > 1.3 * b
    if word == "blue":
        return b > 1.3 * r and b > 1.3 * g
    if word == "yellow":
        return r > 1.4 * b and g > 1.4 * b
    if word == "purple":
        return r > 1.2 * g and b > 1.2 * g
    return None


def candidates_for(prompts, camera_name):
    found = []
    for prompt in prompts:
        seg = segment_text(camera_name, prompt)
        if not seg.ok:
            continue
        for candidate in seg.segmentations[:5]:
            try:
                cloud = mask_to_point_cloud(candidate, camera_name, target_frame=base_frame)
                pts = cloud.points
            except Exception as exc:
                note_error("cloud " + prompt, exc)
                continue
            if len(pts) < 15:
                continue
            xs = [float(pt[0]) for pt in pts]
            ys = [float(pt[1]) for pt in pts]
            zs = [float(pt[2]) for pt in pts]
            z_top = quantile(zs, 0.95)
            top_pts = [pt for pt in pts if float(pt[2]) >= z_top - 0.008]
            entry = {
                "prompt": prompt,
                "score": rnd(candidate.score, 3),
                "box": [int(v) for v in candidate.box_xyxy] if candidate.box_xyxy is not None else None,
                "n": len(pts),
                "center": [rnd(quantile(xs, 0.5)), rnd(quantile(ys, 0.5)), rnd(quantile(zs, 0.5))],
                "top_center": [rnd(quantile((float(pt[0]) for pt in top_pts), 0.5)),
                               rnd(quantile((float(pt[1]) for pt in top_pts), 0.5))],
                "x_span": rnd(quantile(xs, 0.95) - quantile(xs, 0.05)),
                "y_span": rnd(quantile(ys, 0.95) - quantile(ys, 0.05)),
                "z_lo": rnd(quantile(zs, 0.05)),
                "z_hi": rnd(z_top),
                "rgb": mean_color(cloud),
                "sat_rgb": saturated_color(cloud),
                "points": pts,
            }
            found.append(entry)
    return found


def strip(entries):
    out = []
    for entry in entries:
        copy = {}
        for key in entry:
            if key != "points":
                copy[key] = entry[key]
        out.append(copy)
    return out


color_words = []
for word in words:
    if word in ("red", "green", "blue", "yellow", "purple", "orange", "white", "black", "pink", "brown", "gray", "grey"):
        if word not in color_words:
            color_words.append(word)
cube_color = color_words[0] if len(color_words) > 0 else None
report["cube_color"] = cube_color
if cube_color is not None:
    cube_prompts = [cube_color + " cube", cube_color + " block"]
else:
    cube_prompts = ["cube", "small block"]
cube_cands = candidates_for(cube_prompts, "front")
target_cands = candidates_for(["target", "circular target", "purple and white target"], "front")
report["cube_candidates"] = strip(cube_cands)
report["target_candidates"] = strip(target_cands)

# flat live targets, de-duplicated by xy
live_targets = []
live_target_px = []
live_target_box = []
live_target_rgb = []
flat_z = None
for entry in target_cands:
    if entry["z_hi"] - entry["z_lo"] <= 0.03 and (flat_z is None or entry["center"][2] < flat_z):
        flat_z = entry["center"][2]
for entry in target_cands:
    if entry["z_hi"] - entry["z_lo"] > 0.03 or entry["x_span"] < 0.03 or entry["x_span"] > 0.12:
        continue
    if entry["center"][2] > flat_z + 0.012:
        continue
    dup = False
    for other in live_targets:
        if math.hypot(other[0] - entry["center"][0], other[1] - entry["center"][1]) < 0.03:
            dup = True
    if not dup:
        live_targets.append([entry["center"][0], entry["center"][1], entry["center"][2]])
        box = entry["box"]
        live_target_px.append([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])] if box is not None else None)
        live_target_box.append(box)
        live_target_rgb.append(entry["sat_rgb"])
report["live_targets"] = live_targets
report["live_target_px"] = live_target_px
report["live_target_rgb"] = live_target_rgb

# cube gate: compact, a few cm in every horizontal span, not tall; colour must not contradict
good_cubes = []
for entry in cube_cands:
    if not (0.015 <= entry["x_span"] <= 0.08 and 0.015 <= entry["y_span"] <= 0.08 and entry["z_hi"] - entry["z_lo"] <= 0.09):
        continue
    match = color_matches(cube_color, entry["rgb"]) if cube_color is not None else None
    entry["color_ok"] = match
    if match is False:
        continue
    good_cubes.append(entry)
cube = None
for entry in good_cubes:
    if cube is None or entry["score"] > cube["score"]:
        cube = entry
report["cube_selected"] = strip([cube])[0] if cube is not None else None

# ---------------------------------------------------------------- choose the placement event
origin = grasps[0]["xyz"] if len(grasps) > 0 else None
for ev in releases:
    ev["at_origin"] = origin is not None and math.hypot(ev["xyz"][0] - origin[0], ev["xyz"][1] - origin[1]) < 0.03
    near = None
    for tgt in live_targets:
        d = math.hypot(ev["xyz"][0] - tgt[0], ev["xyz"][1] - tgt[1])
        if d < 0.045 and (near is None or d < near):
            near = d
    ev["on_live_target"] = near is not None
on_target = [ev for ev in releases if ev["on_live_target"]]
not_origin = [ev for ev in releases if not ev["at_origin"]]
events = releases
event_kind = "all_low_releases"
if ordinal is not None and len(on_target) >= ordinal:
    events = on_target
    event_kind = "releases_on_live_targets"
elif ordinal is not None and len(not_origin) >= ordinal:
    events = not_origin
    event_kind = "releases_not_at_origin"
report["placements"] = events
report["event_kind"] = event_kind
chosen = None
if ordinal is not None and len(events) >= ordinal:
    if from_end:
        chosen = events[len(events) - ordinal]
    else:
        chosen = events[ordinal - 1]
report["chosen_event"] = chosen

place_xy = None
place_z = None
if chosen is not None:
    place_xy = [chosen["xyz"][0], chosen["xyz"][1]]
    place_z = chosen["xyz"][2]
    best = None
    best_d = None
    for tgt in live_targets:
        d = math.hypot(tgt[0] - place_xy[0], tgt[1] - place_xy[1])
        if d < 0.05 and (best_d is None or d < best_d):
            best = tgt
            best_d = d
    if best is not None:
        report["snapped_target"] = {"center": best, "d": rnd(best_d)}
        place_xy = [best[0], best[1]]
report["place_xy"] = place_xy
report["place_z_video"] = place_z

# ---------------------------------------------------------------- target identity tracking
# The targets look identical and exchange places during the conditioning video.  Track which
# target identity sits on each slot (a slot = a target position in the first video frame),
# read which identity each placement landed on, and place on that identity's final slot.
track = {"ok": False}
report["location_choice_xy"] = place_xy
try:
    ref_rgb = None
    for col in live_target_rgb:
        if col is not None and (ref_rgb is None or col[3] > ref_rgb[3]):
            ref_rgb = col
    track["ref_rgb"] = ref_rgb
    width = len(live.cameras["front"].rgb[0])
    height = len(live.cameras["front"].rgb)
    scale = width / 256.0
    stride = 2
    margin = int(50 * scale)
    xs_box = [b[0] for b in live_target_box if b is not None] + [b[2] for b in live_target_box if b is not None]
    ys_box = [b[1] for b in live_target_box if b is not None] + [b[3] for b in live_target_box if b is not None]
    if len(xs_box) == 0 or n_frames < 2:
        raise ValueError("no live target boxes or no video")
    x0 = max(0, int(min(xs_box)) - margin)
    x1 = min(width, int(max(xs_box)) + margin)
    y0 = max(0, int(min(ys_box)) - margin)
    y1 = min(height, int(max(ys_box)) + margin)
    track["roi"] = [x0, y0, x1, y1]

    def target_cells(rgb):
        try:
            rows = rgb.tolist()
        except Exception:
            rows = rgb
        cells = {}
        for y in range(y0 - y0 % stride, y1, stride):
            line = rows[y]
            for x in range(x0 - x0 % stride, x1, stride):
                p = line[x]
                r = int(p[0])
                g = int(p[1])
                b = int(p[2])
                if ref_rgb is not None:
                    hit = abs(r - ref_rgb[0]) + abs(g - ref_rgb[1]) + abs(b - ref_rgb[2]) < 110 and max(r, g, b) - min(r, g, b) > 50
                else:
                    hit = r > 100 and b > g + 40 and r > g + 25
                if hit:
                    cells[(y // stride, x // stride)] = 1
        return cells

    def components(cells, min_cells, merge_px):
        seen = {}
        comps = []
        for key in cells:
            if key in seen:
                continue
            stack = [key]
            seen[key] = 1
            sx = 0.0
            sy = 0.0
            n = 0
            while len(stack) > 0:
                cy, cx = stack.pop()
                sx += cx
                sy += cy
                n += 1
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1),
                               (cy + 1, cx + 1), (cy - 1, cx - 1), (cy + 1, cx - 1), (cy - 1, cx + 1)):
                    nk = (ny, nx)
                    if nk in cells and nk not in seen:
                        seen[nk] = 1
                        stack.append(nk)
            comps.append([sx / n * stride, sy / n * stride, n])
        merged = []
        order = sorted([[-c[2], c[0], c[1]] for c in comps])
        for item in order:
            comp = [item[1], item[2], -item[0]]
            home = None
            for m in merged:
                if math.hypot(m[0] - comp[0], m[1] - comp[1]) < merge_px:
                    home = m
                    break
            if home is None:
                merged.append([comp[0], comp[1], comp[2]])
            else:
                tot = home[2] + comp[2]
                home[0] = (home[0] * home[2] + comp[0] * comp[2]) / tot
                home[1] = (home[1] * home[2] + comp[1] * comp[2]) / tot
                home[2] = tot
        out = []
        for m in merged:
            if m[2] >= min_cells:
                out.append(m)
        return out

    min_cells = max(3, int(48 * scale * scale / (stride * stride)))
    frames_blobs = []
    for index in range(n_frames):
        cells = target_cells(video[index].cameras["front"].rgb)
        frames_blobs.append(components(cells, min_cells, 9.0 * scale))
    track["blob_counts"] = [len(b) for b in frames_blobs]

    # slots: the first frame whose blobs match every live target pixel position
    slots = None
    slot_frame = None
    for index in range(min(n_frames, 60)):
        blobs = frames_blobs[index]
        if len(blobs) != len(live_targets):
            continue
        okm = True
        for b in blobs:
            dmin = None
            for px in live_target_px:
                if px is None:
                    continue
                d = math.hypot(px[0] - b[0], px[1] - b[1])
                if dmin is None or d < dmin:
                    dmin = d
            if dmin is None or dmin > 8.0 * scale:
                okm = False
        if okm:
            slots = [[b[0], b[1]] for b in blobs]
            slot_frame = index
            break
    if slots is None:
        raise ValueError("no video frame whose target blobs match the live targets")
    track["slot_frame"] = slot_frame
    track["slots_px"] = [[rnd(v, 1) for v in sl] for sl in slots]
    slot_xyz = []
    for sl in slots:
        best = None
        best_d = None
        for k in range(len(live_target_px)):
            px = live_target_px[k]
            if px is None:
                continue
            d = math.hypot(px[0] - sl[0], px[1] - sl[1])
            if best_d is None or d < best_d:
                best = k
                best_d = d
        slot_xyz.append(live_targets[best])
    track["slots_xyz"] = slot_xyz

    # swap episodes
    move_px = 10.0 * scale
    sym_px = 6.0 * scale
    moving_frames = []
    for fi in range(n_frames):
        moving = []
        for blob in frames_blobs[fi]:
            dmin = None
            for sl in slots:
                d = math.hypot(blob[0] - sl[0], blob[1] - sl[1])
                if dmin is None or d < dmin:
                    dmin = d
            if dmin > move_px:
                moving.append(blob)
        moving_frames.append(moving)
    episodes = []
    cur = None
    gap = 0
    for fi in range(slot_frame, n_frames):
        if len(moving_frames[fi]) > 0:
            if cur is None:
                cur = [fi, fi]
            cur[1] = fi
            gap = 0
        elif cur is not None:
            gap += 1
            # a still frame with every slot occupied means the swap is complete: close the
            # episode at once so back-to-back swaps are not merged (v04)
            all_home = True
            for sl in slots:
                near = False
                for blob in frames_blobs[fi]:
                    if math.hypot(blob[0] - sl[0], blob[1] - sl[1]) < 5.0 * scale:
                        near = True
                if not near:
                    all_home = False
            if gap > 4 or all_home:
                episodes.append(cur)
                cur = None
                gap = 0
    if cur is not None:
        episodes.append(cur)
    nslot = len(slots)
    swaps = []
    for ep in episodes:
        vacant = []
        for k in range(nslot):
            empty = 0
            for fi in range(ep[0], ep[1] + 1):
                hit = False
                for blob in frames_blobs[fi]:
                    if math.hypot(blob[0] - slots[k][0], blob[1] - slots[k][1]) < 5.0 * scale:
                        hit = True
                if not hit:
                    empty += 1
            vacant.append(empty / float(ep[1] - ep[0] + 1))
        scores = {}
        for i in range(nslot):
            for j in range(i + 1, nslot):
                if vacant[i] < 0.3 or vacant[j] < 0.3:
                    continue
                mx = 0.5 * (slots[i][0] + slots[j][0])
                my = 0.5 * (slots[i][1] + slots[j][1])
                rad = 0.5 * math.hypot(slots[i][0] - slots[j][0], slots[i][1] - slots[j][1])
                count = 0
                for fi in range(ep[0], ep[1] + 1):
                    mv = moving_frames[fi]
                    hit = False
                    for ia in range(len(mv)):
                        a = mv[ia]
                        ra = math.hypot(a[0] - mx, a[1] - my)
                        if ra < 0.4 * rad or ra > 3.0 * rad:
                            continue
                        for ib in range(len(mv)):
                            if ib == ia:
                                continue
                            b = mv[ib]
                            rb = math.hypot(b[0] - mx, b[1] - my)
                            if rb < 0.4 * rad or rb > 3.0 * rad:
                                continue
                            if math.hypot(0.5 * (a[0] + b[0]) - mx, 0.5 * (a[1] + b[1]) - my) < sym_px:
                                hit = True
                    if hit:
                        count += 1
                if count > 0:
                    scores[(i, j)] = count
        ranked = sorted([[-scores[key], key[0], key[1]] for key in scores])
        busy = {}
        pairs = []
        for item in ranked:
            cnt = -item[0]
            i = item[1]
            j = item[2]
            if cnt < 2:
                break
            if i in busy or j in busy:
                continue
            busy[i] = 1
            busy[j] = 1
            pairs.append([i, j])
        swaps.append({"frames": ep, "pairs": pairs, "vacant": [rnd(v, 2) for v in vacant],
                      "scores": [[key[0], key[1], scores[key]] for key in scores]})
    track["swaps"] = swaps

    # replay: identity k starts on slot k; releases read the identity under the cube
    content = [k for k in range(nslot)]
    timeline_events = []
    for sw in swaps:
        timeline_events.append([sw["frames"][1], 0, sw["pairs"]])
    for ev in releases:
        timeline_events.append([ev["frame"], 1, ev])
    ordered = sorted([[item[0], item[1], k] for k, item in enumerate(timeline_events)])
    id_placements = []
    for item in ordered:
        kind = item[1]
        payload = timeline_events[item[2]][2]
        if kind == 0:
            for pair in payload:
                i = pair[0]
                j = pair[1]
                tmp = content[i]
                content[i] = content[j]
                content[j] = tmp
        else:
            best = None
            best_d = None
            for k in range(nslot):
                d = math.hypot(slot_xyz[k][0] - payload["xyz"][0], slot_xyz[k][1] - payload["xyz"][1])
                if best_d is None or d < best_d:
                    best = k
                    best_d = d
            if best_d is not None and best_d < 0.045:
                id_placements.append({"frame": payload["frame"], "slot": best, "identity": content[best],
                                      "d": rnd(best_d)})
    track["final_content"] = list(content)
    track["id_placements"] = id_placements
    if ordinal is None or len(id_placements) < ordinal:
        raise ValueError("ordinal %s but %d identity placements" % (str(ordinal), len(id_placements)))
    if from_end:
        pick = id_placements[len(id_placements) - ordinal]
    else:
        pick = id_placements[ordinal - 1]
    final_slot = content.index(pick["identity"])
    track["chosen"] = pick
    track["final_slot"] = final_slot
    track["ok"] = True
    place_xy = [slot_xyz[final_slot][0], slot_xyz[final_slot][1]]
    report["placement_basis"] = "identity_tracking"
except Exception as exc:
    note_error("tracking", exc)
    report["placement_basis"] = "location_fallback"
report["track"] = track
report["place_xy"] = place_xy

# ---------------------------------------------------------------- grasp geometry
def yaw_of_finger_axis(quat):
    w, x, y, z = quat
    # second column of the rotation matrix: hand y axis (finger closing direction)
    ry_x = 2.0 * (x * y - w * z)
    ry_y = 1.0 - 2.0 * (x * x + z * z)
    return math.atan2(ry_y, ry_x)


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def cube_edge_yaw(points, top_z):
    top = [(float(pt[0]), float(pt[1])) for pt in points if float(pt[2]) >= top_z - 0.012]
    if len(top) < 10:
        return None
    best_angle = None
    best_area = None
    for k in range(18):
        ang = math.radians(5.0 * k)
        ca = math.cos(ang)
        sa = math.sin(ang)
        us = [px * ca + py * sa for px, py in top]
        vs = [-px * sa + py * ca for px, py in top]
        area = (quantile(us, 0.95) - quantile(us, 0.05)) * (quantile(vs, 0.95) - quantile(vs, 0.05))
        if best_area is None or area < best_area:
            best_area = area
            best_angle = ang
    return best_angle


def wrap_quarter(angle):
    while angle > math.pi / 4.0:
        angle -= math.pi / 2.0
    while angle < -math.pi / 4.0:
        angle += math.pi / 2.0
    return angle


def downward_with_yaw(delta):
    qz = [math.cos(delta / 2.0), 0.0, 0.0, math.sin(delta / 2.0)]
    return quat_mul(qz, home_quat)


def move(label, xyz, quat, tol=0.012, steps=200):
    norm = math.sqrt(sum(float(v) * float(v) for v in quat))
    quat = [float(v) / norm for v in quat]
    res = move_to_pose(Pose((float(xyz[0]), float(xyz[1]), float(xyz[2])), quat, base_frame),
                       tolerance=tol, max_steps=steps)
    return log_motion(label, res)


def hold_ticks(count, gripper):
    st = get_robot_state()
    last = None
    for tick in range(count):
        last = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=st.joint_positions["primary"],
            gripper_position=gripper, embodiment=st.embodiment)}))
        try:
            if bool(last.terminated) or bool(last.truncated):
                episode_over[0] = True
                break
        except Exception:
            pass
    return last


try:
    if cube is None:
        raise ValueError("no cube candidate passed the geometry gate")
    if place_xy is None:
        raise ValueError("no placement event chosen (ordinal %s, releases %d, dwell %d)" % (
            str(ordinal), len(releases), len(dwell_points)))
    pts = cube["points"]
    zs = [float(pt[2]) for pt in pts]
    z_top = quantile(zs, 0.95)
    z_low = quantile(zs, 0.05)
    cx = cube["top_center"][0]
    cy = cube["top_center"][1]
    # the cube rests where the video last released it: reuse that release's xy and orientation
    video_rest = None
    all_rel = report["video_releases"]
    for ev in all_rel:
        if math.hypot(ev["xyz"][0] - cx, ev["xyz"][1] - cy) < 0.035:
            video_rest = ev
    report["cube_video_rest"] = video_rest
    grasp_z = max(z_top - 0.02, 0.5 * (z_top + z_low))
    report["grasp_z_perception"] = rnd(grasp_z)
    if len(grasps) > 0:
        report["grasp_z_video"] = grasps[0]["xyz"][2]
    edge = cube_edge_yaw(pts, z_top)
    finger = yaw_of_finger_axis(home_quat)
    delta = 0.0
    if edge is not None:
        delta = wrap_quarter(edge - finger)
    report["grasp_yaw_delta_deg"] = rnd(math.degrees(delta), 1)
    grasp_quat = downward_with_yaw(delta)
    if video_rest is not None:
        cx = video_rest["xyz"][0]
        cy = video_rest["xyz"][1]
        qv = video[video_rest["frame"]].robot_state.end_effector_poses["primary"].quaternion_wxyz
        grasp_quat = [float(v) for v in qv]
    qn = math.sqrt(sum(v * v for v in grasp_quat))
    grasp_quat = [v / qn for v in grasp_quat]
    report["grasp_xy"] = [rnd(cx), rnd(cy)]
    report["grasp_quat"] = [rnd(v) for v in grasp_quat]

    report["stage"] = "pick"
    log_motion("open", open_gripper())
    move("pre_grasp", (cx, cy, z_top + 0.10), grasp_quat)
    move("grasp", (cx, cy, grasp_z), grasp_quat, tol=0.008)
    log_motion("close", close_gripper())
    report["gripper_after_close"] = rnd(get_robot_state().gripper_positions["primary"], 3)
    if report["gripper_after_close"] < 0.12 and not episode_over[0]:
        # empty close: back off, re-open and retry once from the same observation
        move("regrasp_up", (cx, cy, z_top + 0.06), grasp_quat)
        log_motion("reopen", open_gripper())
        move("regrasp", (cx, cy, grasp_z - 0.005), grasp_quat, tol=0.008)
        log_motion("reclose", close_gripper())
        report["gripper_after_reclose"] = rnd(get_robot_state().gripper_positions["primary"], 3)
    move("lift", (cx, cy, z_top + 0.12), grasp_quat)
    report["gripper_after_lift"] = rnd(get_robot_state().gripper_positions["primary"], 3)

    report["stage"] = "place"
    transit_z = max(z_top + 0.12, place_z + 0.10)
    move("above_target", (place_xy[0], place_xy[1], transit_z), grasp_quat, steps=250)
    release_z = place_z + 0.012
    move("lower", (place_xy[0], place_xy[1], release_z), grasp_quat, tol=0.008)
    log_motion("release", open_gripper())
    if not episode_over[0]:
        hold_ticks(5, 1.0)
    if not episode_over[0]:
        move("retreat", (place_xy[0], place_xy[1], release_z + 0.10), grasp_quat)
    if not episode_over[0]:
        hold_ticks(10, 1.0)
    report["stage"] = "done"
except Exception as exc:
    note_error("execute", exc)

try:
    final = get_robot_state()
    report["final_t"] = rnd(final.timestamp_s, 3)
    report["final_ee"] = [rnd(v) for v in final.end_effector_poses["primary"].position]
except Exception as exc:
    note_error("final_state", exc)
report["motions"] = motion_log
report["episode_over"] = episode_over[0]
result = report
