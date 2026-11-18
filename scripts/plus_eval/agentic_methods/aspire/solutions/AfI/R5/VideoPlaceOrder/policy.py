# VideoPlaceOrder v03 (from v02; adds target-identity tracking through video swaps). Original header: v00 -- initial program from scene analysis (folds Stage 0 exploration in).
# Strategy (hypothesis, from the task name only; verified against the run's own evidence):
#   1. Read the goal and metadata; parse an ordinal ("first", "second", ..., "last").
#   2. Read the conditioning video. Use the RECORDED robot state of each frame (end-effector
#      position + gripper reading) to find the low "touch-down" segments; a segment entered with
#      the gripper holding something is a placement. The k-th placement's XY is the target.
#   3. Live: grasp the cube (unless already held), carry it to the selected placement XY
#      (refined by a live target segmentation only if a candidate lies very close), release, settle.
# Everything task-dependent is derived at runtime from the goal, video robot state and live RGB-D.
import math

report = {"version": "v03", "log": [], "errors": []}
ended = {"flag": False}


def note(name, res):
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
    try:
        entry["terminated"] = bool(res.terminated)
        entry["truncated"] = bool(res.truncated)
        if res.terminated or res.truncated:
            ended["flag"] = True
    except Exception:
        pass
    try:
        if res.error is not None:
            entry["error"] = str(res.error)[:300]
    except Exception:
        pass
    report["log"].append(entry)
    return res


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def vec(p):
    return [float(p[0]), float(p[1]), float(p[2])]


# ---------------------------------------------------------------- goal + metadata
context = get_task_context()
goal = str(context.language)
report["goal"] = goal
meta = {}
try:
    md = robomme_plus.get_task_metadata()
    for key in md:
        meta[str(key)] = str(md[key])[:300]
except Exception as exc:
    report["errors"].append("metadata: " + str(exc)[:200])
report["task_metadata"] = meta
try:
    cmd = robomme_plus.get_controller_metadata()
    report["controller_metadata"] = {str(k): str(cmd[k])[:200] for k in cmd}
except Exception as exc:
    report["errors"].append("controller metadata: " + str(exc)[:200])

ORDINALS = [
    ("first", 1), ("1st", 1), ("second", 2), ("2nd", 2), ("third", 3), ("3rd", 3),
    ("fourth", 4), ("4th", 4), ("fifth", 5), ("5th", 5), ("sixth", 6), ("6th", 6),
    ("seventh", 7), ("eighth", 8),
]
low_goal = goal.lower()
ordinal = None
ordinal_word = None
best_pos = None
for word, number in ORDINALS:
    pos = low_goal.find(word)
    if pos >= 0 and (best_pos is None or pos < best_pos):
        best_pos = pos
        ordinal = number
        ordinal_word = word
from_last = ("last" in low_goal) or ("final" in low_goal)
if from_last and ordinal is None:
    ordinal = -1
    ordinal_word = "last"
elif from_last and ordinal is not None and ("to last" in low_goal or "from last" in low_goal or "from the end" in low_goal):
    ordinal = -ordinal
report["ordinal"] = ordinal
report["ordinal_word"] = ordinal_word
sequential = ("same order" in low_goal) or ("in order" in low_goal and ordinal is None) or ("sequence" in low_goal and ordinal is None)
report["sequential_mode"] = sequential

COLORS = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "white", "black", "brown", "gray", "grey", "cyan"]
cube_color = None
cube_pos = low_goal.find("cube")
best_gap = None
for color in COLORS:
    cpos = low_goal.find(color)
    if cpos >= 0 and cube_pos >= 0 and cpos < cube_pos:
        gap = cube_pos - cpos
        if best_gap is None or gap < best_gap:
            best_gap = gap
            cube_color = color
report["cube_color"] = cube_color

# ---------------------------------------------------------------- conditioning video (robot state)
video = []
try:
    video = robomme_plus.get_conditioning_video()
except Exception as exc:
    report["errors"].append("video: " + str(exc)[:200])
n_frames = len(video)
report["video_frames"] = n_frames
series = []
for index in range(n_frames):
    frame = video[index]
    row = {"i": index}
    try:
        row["t"] = round(float(frame.timestamp_s), 3)
    except Exception:
        row["t"] = None
    try:
        rs = frame.robot_state
        pos = rs.end_effector_poses["primary"].position
        row["p"] = [round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)]
        row["g"] = round(float(rs.gripper_positions["primary"]), 3)
    except Exception as exc:
        row["err"] = str(exc)[:100]
    series.append(row)
report["video_series"] = series

valid = [r for r in series if "p" in r]
events = []
if len(valid) > 3:
    zs = [r["p"][2] for r in valid]
    zmin = min(zs)
    zmax = max(zs)
    low_band = zmin + max(0.035, 0.15 * (zmax - zmin))
    report["video_z"] = {"min": zmin, "max": zmax, "low_band": low_band}
    segment = []
    segments = []
    for r in valid:
        if r["p"][2] <= low_band:
            segment.append(r)
        else:
            if len(segment) > 0:
                segments.append(segment)
            segment = []
    if len(segment) > 0:
        segments.append(segment)
    for seg in segments:
        lowest = seg[0]
        for r in seg:
            if r["p"][2] < lowest["p"][2]:
                lowest = r
        # gripper just before entering and just after leaving the segment
        first_i = seg[0]["i"]
        last_i = seg[-1]["i"]
        g_in = seg[0].get("g")
        g_out = seg[-1].get("g")
        for r in valid:
            if r["i"] < first_i:
                g_in = r.get("g", g_in)
            if r["i"] > last_i:
                g_out = r.get("g", g_out)
                break
        g_values = [r.get("g") for r in seg if r.get("g") is not None]
        kind = "unknown"
        if g_in is not None and g_out is not None:
            if 0.1 < g_in < 0.85 and (g_out > 0.85 or (len(g_values) > 0 and max(g_values) > 0.85)):
                kind = "place"
            elif g_in < 0.85:
                kind = "touch_closed"
            elif g_out < 0.85:
                kind = "pick"
            else:
                kind = "touch_open"
        events.append({
            "first_i": first_i, "last_i": last_i, "min_i": lowest["i"],
            "t": lowest.get("t"), "xyz": lowest["p"], "g_in": g_in, "g_out": g_out,
            "g_min": min(g_values) if len(g_values) > 0 else None,
            "g_max": max(g_values) if len(g_values) > 0 else None,
            "kind": kind,
        })
report["video_events"] = events
placements = [e for e in events if e["kind"] == "place"]
report["placements"] = placements

# small RGB thumbnails (stride 8) of the event frames for offline inspection
thumbs = []
for e in events[:10]:
    try:
        rgb = video[e["min_i"]].cameras["front"].rgb
        h = len(rgb)
        w = len(rgb[0])
        small = []
        for y in range(0, h, 8):
            rowpix = []
            for x in range(0, w, 8):
                px = rgb[y][x]
                rowpix.append([int(px[0]), int(px[1]), int(px[2])])
            small.append(rowpix)
        thumbs.append({"i": e["min_i"], "rgb8": small})
    except Exception as exc:
        report["errors"].append("thumb: " + str(exc)[:120])
report["event_thumbs"] = thumbs

# ---------------------------------------------------------------- live scene
state = get_robot_state()
base = state.base_frame
start_g = float(state.gripper_positions["primary"])
report["start_gripper"] = start_g
report["start_ee"] = vec(state.end_effector_poses["primary"].position)
holding = 0.1 < start_g < 0.85
report["start_holding_guess"] = holding

if not holding:
    note("open_gripper", open_gripper())
    if not ended["flag"]:
        note("go_home", go_home())

state = get_robot_state()
q = state.end_effector_poses["primary"].quaternion_wxyz
qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
if r22 < -0.9:
    down = (qw, qx, qy, qz)
else:
    down = (0.0, 1.0, 0.0, 0.0)
report["down_quat"] = list(down)
report["home_ee"] = vec(state.end_effector_poses["primary"].position)


def cloud_summary(cloud):
    pts = cloud.points
    if len(pts) < 15:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    zs = [float(p[2]) for p in pts]
    ztop = quantile(zs, 0.95)
    tx = [xs[k] for k in range(len(zs)) if zs[k] >= ztop - 0.006]
    ty = [ys[k] for k in range(len(zs)) if zs[k] >= ztop - 0.006]
    return {
        "top_c": [quantile(tx, 0.5), quantile(ty, 0.5)],
        "n": len(pts),
        "c": [quantile(xs, 0.5), quantile(ys, 0.5), quantile(zs, 0.5)],
        "zlo": quantile(zs, 0.05), "zhi": quantile(zs, 0.95),
        "span": [quantile(xs, 0.95) - quantile(xs, 0.05), quantile(ys, 0.95) - quantile(ys, 0.05)],
    }


def candidates(prompt, limit=6):
    out = []
    seg = segment_text("front", prompt)
    if not seg.ok:
        return out
    for cand in seg.segmentations[:limit]:
        item = {"prompt": prompt, "score": round(float(cand.score), 3), "cand": cand}
        try:
            item["box"] = [round(float(v), 1) for v in cand.box_xyxy]
        except Exception:
            item["box"] = None
        try:
            pc = mask_to_point_cloud(cand, "front", target_frame=base)
            item["geo"] = cloud_summary(pc)
        except Exception as exc:
            item["geo"] = None
            item["err"] = str(exc)[:100]
        out.append(item)
    return out


def public(items):
    return [{k: v for k, v in it.items() if k != "cand"} for it in items]


# live target candidates (diagnostic + refinement)
target_cands = []
for prompt in ("target", "bullseye target", "round flat target"):
    try:
        target_cands = target_cands + candidates(prompt)
    except Exception as exc:
        report["errors"].append("target seg: " + str(exc)[:150])
report["live_target_candidates"] = public(target_cands)

plaus_targets = []
for it in target_cands:
    g = it.get("geo")
    if g is None:
        continue
    # flat disc: thin in z, a few cm across; drop duplicates from several prompts
    if 0.03 < g["span"][0] < 0.12 and 0.03 < g["span"][1] < 0.12 and g["zhi"] - g["zlo"] < 0.01:
        dup = False
        for other in plaus_targets:
            if math.sqrt((other["c"][0] - g["c"][0]) ** 2 + (other["c"][1] - g["c"][1]) ** 2) < 0.03:
                dup = True
        if not dup:
            plaus_targets.append(g)
report["flat_targets"] = [[round(g["c"][0], 4), round(g["c"][1], 4), round(g["zhi"], 4)] for g in plaus_targets]

# ---------------------------------------------------------------- target choice
# keep only video releases that landed on a live flat target (drops the cube's return to its
# start spot and button presses); without live targets, drop releases at the first pick spot.
first_pick = None
for e in events:
    if e["kind"] == "pick":
        first_pick = e
        break
on_target = []
for e in placements:
    near = None
    for g in plaus_targets:
        d = math.sqrt((g["c"][0] - e["xyz"][0]) ** 2 + (g["c"][1] - e["xyz"][1]) ** 2)
        if d < 0.045 and (near is None or d < near):
            near = d
    if len(plaus_targets) > 0:
        if near is not None:
            on_target.append(e)
    else:
        if first_pick is None or math.sqrt((first_pick["xyz"][0] - e["xyz"][0]) ** 2 + (first_pick["xyz"][1] - e["xyz"][1]) ** 2) > 0.03:
            on_target.append(e)
report["target_placements"] = [[e["min_i"], e["xyz"]] for e in on_target]
chosen = []
if len(on_target) > 0:
    if sequential:
        chosen = on_target
    elif ordinal is not None:
        k = ordinal
        if k > 0:
            index = min(k, len(on_target)) - 1
        else:
            index = max(0, len(on_target) + k)
        chosen = [on_target[index]]
        report["chosen_index"] = index
        report["ordinal_clipped"] = (k > 0 and k > len(on_target)) or (k < 0 and -k > len(on_target))
report["chosen_targets"] = [c["xyz"] for c in chosen]

# ---------------------------------------------------------------- target identity through swaps
# The identical targets exchange slots during the conditioning video (pairs travel along arcs
# about their midpoint, passing over other slots). The target that received the chosen placement
# is followed through every later exchange; the cube goes to the slot it occupies at video end.
# Slots = live flat targets (final layout). Each video frame: sample pixels of each slot disc
# projected through the front camera; a slot is VACANT when its disc shows the local table
# colour (not target colour, not an occluding arm/cube). A run of frames with vacant slots is
# one exchange event; two vacated slots = a swap; four = two swaps paired by an arc fit.
swap_info = {"windows": [], "path": []}


def rot_of(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def make_cam(cam):
    k = cam.intrinsics
    p = cam.camera_pose.position
    return {"R": rot_of(cam.camera_pose.quaternion_wxyz), "t": [float(p[0]), float(p[1]), float(p[2])],
            "fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]), "cy": float(k[1][2]),
            "h": len(cam.rgb), "w": len(cam.rgb[0])}


def project(cm, x, y, z):
    d = [x - cm["t"][0], y - cm["t"][1], z - cm["t"][2]]
    R = cm["R"]
    pc = [R[0][i] * d[0] + R[1][i] * d[1] + R[2][i] * d[2] for i in range(3)]
    if pc[2] <= 0.01:
        return None
    u = int(round(cm["fx"] * pc[0] / pc[2] + cm["cx"]))
    v = int(round(cm["fy"] * pc[1] / pc[2] + cm["cy"]))
    if u < 0 or v < 0 or u >= cm["w"] or v >= cm["h"]:
        return None
    return (v, u)


def disc_pixels(cm, cx, cy, z, r0, r1, step):
    out = []
    n = int(r1 / step) + 1
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            dx = i * step
            dy = j * step
            rr = math.sqrt(dx * dx + dy * dy)
            if r0 <= rr <= r1:
                pix = project(cm, cx + dx, cy + dy, z)
                if pix is not None and pix not in out:
                    out.append(pix)
    return out


def read_pixels(rgb, pixels, cache):
    vals = []
    for v, u in pixels:
        row = cache.get(v)
        if row is None:
            try:
                row = rgb[v].tolist()
            except Exception:
                row = [[int(c[0]), int(c[1]), int(c[2])] for c in rgb[v]]
            cache[v] = row
        c = row[u]
        vals.append((int(c[0]), int(c[1]), int(c[2])))
    return vals


def cdist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def near_any(c, palette, tol):
    for p in palette:
        if cdist(c, p) < tol:
            return True
    return False


def thin(colors, limit, tol):
    out = []
    for c in colors:
        if not near_any(c, out, tol):
            out.append(c)
        if len(out) >= limit:
            break
    return out


def track_swaps(place_event):
    slots = [(g["c"][0], g["c"][1], g["zhi"]) for g in plaus_targets]
    ns = len(slots)
    if ns < 2 or n_frames < 10:
        swap_info["error"] = "too few slots/frames"
        return None
    spans = [0.5 * (g["span"][0] + g["span"][1]) for g in plaus_targets]
    disc_r = 0.45 * sorted(spans)[len(spans) // 2]
    # camera: live front camera pose or the video's own front pose, whichever shows targets at the
    # slots in the LAST video frame (final layout == live layout)
    cams = []
    try:
        cams.append(("video", make_cam(video[n_frames - 1].cameras["front"])))
    except Exception as exc:
        report["errors"].append("video cam: " + str(exc)[:120])
    try:
        live_obs = get_observation(camera_names=["front"])
        cams.append(("live", make_cam(live_obs.cameras["front"])))
    except Exception as exc:
        report["errors"].append("live cam: " + str(exc)[:120])
    best = None
    last_rgb = video[n_frames - 1].cameras["front"].rgb
    for name, cm in cams:
        slot_pix = [disc_pixels(cm, s[0], s[1], s[2], 0.0, disc_r, 0.005) for s in slots]
        ring_pix = []
        for k in range(ns):
            ring = disc_pixels(cm, slots[k][0], slots[k][1], slots[k][2], 2.0 * disc_r, 3.2 * disc_r, 0.008)
            keep = []
            for pix in ring:
                keep.append(pix)
            ring_pix.append(keep)
        cache = {}
        # local table colour from each slot's ring (outliers such as cubes removed)
        bg = []
        for k in range(ns):
            cols = read_pixels(last_rgb, ring_pix[k], cache)
            if len(cols) < 5:
                bg.append([])
                continue
            med = (quantile([c[0] for c in cols], 0.5), quantile([c[1] for c in cols], 0.5), quantile([c[2] for c in cols], 0.5))
            dists = sorted(cdist(c, med) for c in cols)
            lim = max(30.0, 1.5 * dists[int(0.6 * (len(dists) - 1))])
            bg.append(thin([c for c in cols if cdist(c, med) < lim], 40, 8.0))
        # target colours: slot-disc pixels in the last frame that are not table-coloured
        tcols = []
        for k in range(ns):
            for c in read_pixels(last_rgb, slot_pix[k], cache):
                if len(bg[k]) > 0 and not near_any(c, bg[k], 30.0):
                    tcols.append(c)
        score = len(tcols) / float(max(1, sum(len(p) for p in slot_pix)))
        report.setdefault("swap_cam_scores", {})
        report["swap_cam_scores"][name] = round(score, 3)
        if best is None or score > best[0]:
            best = (score, name, cm, slot_pix, bg, thin(tcols, 60, 10.0))
    if best is None or best[0] < 0.3:
        swap_info["error"] = "slots not visible as targets in last video frame"
        return None
    score, cam_name, cm, slot_pix, bg, tpal = best
    swap_info["camera"] = cam_name
    swap_info["target_palette"] = len(tpal)
    # per-frame slot states: 1 vacant (table colour), 0 occupied (target colour), -1 occluded/other
    states = []
    for n in range(n_frames):
        rgb = video[n].cameras["front"].rgb
        cache = {}
        row = []
        for k in range(ns):
            cols = read_pixels(rgb, slot_pix[k], cache)
            m = float(max(1, len(cols)))
            nb = 0
            nt = 0
            for c in cols:
                if near_any(c, bg[k], 26.0):
                    nb += 1
                elif near_any(c, tpal, 30.0):
                    nt += 1
            if nt / m >= 0.3:
                row.append(0)
            elif nb / m >= 0.6:
                row.append(1)
            else:
                row.append(-1)
        states.append(row)
    swap_info["states"] = ["".join(("." if s == 0 else ("v" if s == 1 else "x")) for s in row) for row in states]
    # vacancy runs (gaps of up to 2 frames bridged)
    windows = []
    cur = None
    gap = 0
    for n in range(n_frames):
        if 1 in states[n]:
            if cur is None:
                cur = [n, n]
            cur[1] = n
            gap = 0
        elif cur is not None:
            gap += 1
            if gap > 2:
                windows.append(cur)
                cur = None
                gap = 0
    if cur is not None:
        windows.append(cur)
    events_sw = []
    for w0, w1 in windows:
        vac = [k for k in range(ns) if sum(1 for n in range(w0, w1 + 1) if states[n][k] == 1) >= 3]
        item = {"frames": [w0, w1], "vacated": vac}
        if len(vac) == 2:
            item["pairs"] = [[vac[0], vac[1]]]
        elif len(vac) >= 3:
            item["pairs"] = pair_by_arcs(w0, w1, vac, slots, cm, bg, tpal)
        else:
            item["pairs"] = []
        events_sw.append(item)
    swap_info["windows"] = events_sw
    # follow the chosen target
    ex, ey = place_event["xyz"][0], place_event["xyz"][1]
    start = None
    for k in range(ns):
        d = math.sqrt((slots[k][0] - ex) ** 2 + (slots[k][1] - ey) ** 2)
        if start is None or d < start[0]:
            start = (d, k)
    if start is None or start[0] > 0.05:
        swap_info["error"] = "placement not on a slot"
        return None
    slot = start[1]
    swap_info["path"].append([place_event["min_i"], slot])
    for item in events_sw:
        if item["frames"][0] <= place_event["min_i"]:
            continue
        for a, b in item["pairs"]:
            if slot == a:
                slot = b
                swap_info["path"].append([item["frames"][0], slot])
                break
            if slot == b:
                slot = a
                swap_info["path"].append([item["frames"][0], slot])
                break
    swap_info["final_slot"] = slot
    return slots[slot]


def matchings(items):
    if len(items) < 2:
        return [[]]
    out = []
    first = items[0]
    for j in range(1, len(items)):
        rest = [items[k] for k in range(1, len(items)) if k != j]
        for m in matchings(rest):
            out.append([[first, items[j]]] + m)
    if len(items) % 2 == 1:
        for m in matchings(items[1:]):
            out.append(m)
    return out


def pair_by_arcs(w0, w1, vac, slots, cm, bg, tpal):
    # moving target pixels on a table-plane grid; a swap pair rotates about its midpoint, so its
    # moving targets stay ~half the slot distance from that midpoint
    xs = [s[0] for s in slots]
    ys = [s[1] for s in slots]
    z = sorted(s[2] for s in slots)[len(slots) // 2]
    grid = []
    gx = min(xs) - 0.1
    while gx <= max(xs) + 0.1:
        gy = min(ys) - 0.1
        while gy <= max(ys) + 0.1:
            pix = project(cm, gx, gy, z)
            if pix is not None:
                grid.append((gx, gy, pix))
            gy += 0.008
        gx += 0.008
    allbg = []
    for b in bg:
        allbg = allbg + b
    pts = []
    for n in range(w0, w1 + 1):
        rgb = video[n].cameras["front"].rgb
        cache = {}
        cols = read_pixels(rgb, [g[2] for g in grid], cache)
        for k in range(len(grid)):
            c = cols[k]
            if near_any(c, tpal, 30.0) and not near_any(c, allbg, 26.0):
                gx, gy = grid[k][0], grid[k][1]
                home = False
                for s in slots:
                    if math.sqrt((s[0] - gx) ** 2 + (s[1] - gy) ** 2) < 0.04:
                        home = True
                if not home:
                    pts.append((gx, gy))
    best = None
    scores = []
    for m in matchings(vac):
        if len(m) == 0:
            continue
        tot = 0.0
        for gx, gy in pts:
            e = None
            for a, b in m:
                mx = 0.5 * (slots[a][0] + slots[b][0])
                my = 0.5 * (slots[a][1] + slots[b][1])
                r = 0.5 * math.sqrt((slots[a][0] - slots[b][0]) ** 2 + (slots[a][1] - slots[b][1]) ** 2)
                d = abs(math.sqrt((gx - mx) ** 2 + (gy - my) ** 2) - r)
                if e is None or d < e:
                    e = d
            tot += min(e, 0.08)
        # unmatched vacated slots are penalised
        used = sum(2 for p in m)
        tot += 0.08 * 20 * (len(vac) - used)
        scores.append([m, round(tot, 3)])
        if best is None or tot < best[0]:
            best = (tot, m)
    swap_info.setdefault("arc_fits", [])
    swap_info["arc_fits"].append({"frames": [w0, w1], "moving_px": len(pts), "scores": scores})
    if best is None:
        return []
    return best[1]


final_target = None
if len(chosen) == 1 and not sequential:
    try:
        final_target = track_swaps(chosen[0])
    except Exception as exc:
        report["errors"].append("swap tracking: " + str(exc)[:300])
report["swap_tracking"] = swap_info
report["final_target"] = final_target


def find_cube():
    prompts = []
    if cube_color is not None:
        prompts.append(cube_color + " cube")
    prompts.append("cube")
    prompts.append("small block")
    best = None
    seen = []
    for prompt in prompts:
        try:
            items = candidates(prompt)
        except Exception as exc:
            report["errors"].append("cube seg: " + str(exc)[:150])
            continue
        seen = seen + items
        for it in items:
            g = it.get("geo")
            if g is None:
                continue
            height = g["zhi"] - g["zlo"]
            if g["span"][0] > 0.09 or g["span"][1] > 0.09 or height < 0.01 or height > 0.09:
                continue
            if best is None or it["score"] > best["score"]:
                best = it
        if best is not None:
            break
    report.setdefault("cube_searches", [])
    report["cube_searches"].append(public(seen))
    return best


def qmul(a, b):
    return (a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])


# same top-down approach, wrist turned 90 deg about the base z axis (fingers close along the other axis)
down90 = qmul((math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)), down)
report["down90_quat"] = list(down90)
grip_quat = {"q": down}


def move(name, x, y, z, tol=0.015, steps=250):
    if ended["flag"]:
        return None
    return note(name, move_to_pose(Pose((x, y, z), grip_quat["q"], base), tolerance=tol, max_steps=steps))


def hold_ticks(n, grip):
    st = get_robot_state()
    for tick in range(n):
        if ended["flag"]:
            break
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=st.joint_positions["primary"],
            gripper_position=grip, embodiment=st.embodiment)}))
        try:
            if reply.terminated or reply.truncated:
                ended["flag"] = True
                report["log"].append({"call": "hold", "terminated": bool(reply.terminated), "truncated": bool(reply.truncated)})
        except Exception:
            pass


grasp_info = {"cube_bottom": None, "grasp_z": None}


def grasp_cube():
    cube = find_cube()
    if cube is None:
        report["errors"].append("cube not found")
        return False
    g = cube["geo"]
    # top-face centre: the front camera also sees the cube's camera-facing side, which biased
    # the full-cloud median ~2.5 cm toward the camera in v00 (cube held off-centre, landed off-target)
    cx, cy = g["top_c"][0], g["top_c"][1]
    top = g["zhi"]
    bottom = g["zlo"]
    grasp_z = max(top - 0.02, bottom + 0.012)
    grasp_info["cube_bottom"] = bottom
    grasp_info["grasp_z"] = grasp_z
    report.setdefault("grasps", [])
    report["grasps"].append({"xy": [cx, cy], "top": top, "bottom": bottom, "grasp_z": grasp_z})
    note("open_gripper", open_gripper())
    move("pre_grasp", cx, cy, grasp_z + 0.12, 0.02)
    move("grasp", cx, cy, grasp_z, 0.012)
    if ended["flag"]:
        return False
    # Centre the cube in the hand on BOTH axes, independent of perception bias: a closing jaw
    # centres the cube along its closing axis. Close (axis 1), open, rise 3 cm, turn 90 deg,
    # descend, close again (axis 2). v01's top-face estimate still left the cube ~1.5 cm off-centre.
    grip_quat["q"] = down
    note("centre_close_1", close_gripper())
    note("centre_open_1", open_gripper())
    move("centre_rise", cx, cy, grasp_z + 0.03, 0.012)
    grip_quat["q"] = down90
    move("centre_turn", cx, cy, grasp_z + 0.03, 0.012)
    move("grasp_2", cx, cy, grasp_z, 0.012)
    if ended["flag"]:
        return False
    note("close_gripper", close_gripper())
    move("lift", cx, cy, grasp_z + 0.20, 0.02)
    reading = float(get_robot_state().gripper_positions["primary"])
    report["grasps"][-1]["held_reading"] = reading
    return reading > 0.1


def place_at(target_xyz, label):
    tx, ty = float(target_xyz[0]), float(target_xyz[1])
    refined = None
    for g in plaus_targets:
        d = math.sqrt((g["c"][0] - tx) ** 2 + (g["c"][1] - ty) ** 2)
        if d < 0.04 and (refined is None or d < refined[0]):
            refined = (d, g)
    # the slot's earlier video releases were onto OTHER targets (targets swap), so the live
    # disc centre alone defines the release point
    surface = None
    if refined is not None:
        tx = refined[1]["c"][0]
        ty = refined[1]["c"][1]
        surface = refined[1]["zhi"]
    # release height: recorded touch-down TCP height, or our own grasp geometry
    rel_z = float(target_xyz[2]) + 0.015
    if surface is not None and grasp_info["grasp_z"] is not None and grasp_info["cube_bottom"] is not None:
        rel_z = surface + (grasp_info["grasp_z"] - grasp_info["cube_bottom"]) + 0.006
    report.setdefault("places", [])
    report["places"].append({"label": label, "xy": [tx, ty], "rel_z": rel_z,
                             "refined": refined[1] if refined is not None else None,
                             "recorded": list(target_xyz)})
    cur = get_robot_state().end_effector_poses["primary"].position
    transit = max(float(cur[2]), rel_z + 0.15)
    move("transit_" + label, tx, ty, transit, 0.02, 300)
    move("descend_" + label, tx, ty, rel_z, 0.012)
    if ended["flag"]:
        return
    note("release_" + label, open_gripper())
    hold_ticks(8, 1.0)
    move("retreat_" + label, tx, ty, rel_z + 0.12, 0.02)
    hold_ticks(5, 1.0)


try:
    if len(chosen) == 0:
        report["errors"].append("no target chosen from video")
    for idx, target in enumerate(chosen):
        if ended["flag"]:
            break
        held = holding and idx == 0
        tries = 0
        while not held and tries < 2 and not ended["flag"]:
            held = grasp_cube()
            tries += 1
        if not held:
            report["errors"].append("grasp failed")
            break
        if final_target is not None and not sequential:
            place_at((final_target[0], final_target[1], target["xyz"][2]), str(idx))
        else:
            place_at(target["xyz"], str(idx))
except Exception as exc:
    report["errors"].append("main: " + str(exc)[:300])

try:
    fin = get_robot_state()
    report["final_ee"] = vec(fin.end_effector_poses["primary"].position)
    report["final_t"] = float(fin.timestamp_s)
except Exception as exc:
    report["errors"].append("final: " + str(exc)[:100])
report["ended"] = ended["flag"]

result = report
