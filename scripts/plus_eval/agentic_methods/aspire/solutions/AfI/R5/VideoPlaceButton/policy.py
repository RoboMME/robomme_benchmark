# v03 — VideoPlaceButton. Perception fix over v02 (v02 passed episode 0 but failed episode 1: the
# targets there are PURPLE, v02's hard-coded pink pixel rule found no target, identity tracking was
# skipped and the cube went to the chosen release POSITION, which a later swap had vacated).
# v03 LEARNS the target colour from this episode's live front image instead of hard-coding it:
#   - candidate pixels: saturated (max-min >= 60) and not wood-like (r >= g >= b excluded);
#   - colours quantized to 8-level bins; each frequent bin's mean is a seed; pixels within L1 45 of a
#     seed form 8-connected components, and components sharing a centroid (<= 5 px: the rings of
#     one bullseye) are merged into one object (v02's greedy 14-px clustering merged two targets);
#   - an object counts only if it is FLAT on the table (deprojected depth z within 1.5 cm of the
#     lowest candidate object) - cubes' top faces sit a cube-height higher;
#   - target colour = mean RGB of the seed with the most flat objects (>= 2; ties -> more pixels);
#     rest centres = those flat objects (>= 15 px); a pixel is "target" if L1 <= 45 to the colour.
# Everything else (video event parsing, swap tracking, motion) is unchanged from v02.
# Everything is derived at runtime from this episode's goal, video and live images.
import math

report = {"version": "v03", "stage": "start", "notes": []}


def fl(values):
    return [float(v) for v in values]


def r4(values):
    return [round(float(v), 4) for v in values]


def unit(q):
    n = math.sqrt(sum(float(v) * float(v) for v in q))
    if n < 1e-9:
        return None
    return [float(v) / n for v in q]


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def thumb(rgb, stride):
    try:
        return rgb[::stride, ::stride].tolist()
    except Exception:
        return None


# ---------------------------------------------------------------- context
context = get_task_context()
goal = str(context.language)
words = goal.lower()
report["goal"] = goal
try:
    meta = dict(robomme_plus.get_task_metadata())
    report["max_steps"] = meta.get("max_steps")
except Exception as exc:
    report["task_metadata_error"] = str(exc)

state0 = get_robot_state()
base_frame = state0.base_frame
home_quat = unit(fl(state0.end_effector_poses["primary"].quaternion_wxyz))
report["home_ee"] = r4(state0.end_effector_poses["primary"].position)
start_grip = float(state0.gripper_positions["primary"])
report["start_gripper"] = start_grip

# ---------------------------------------------------------------- conditioning video (robot state)
video = robomme_plus.get_conditioning_video()
n_frames = len(video)
report["video_frames"] = n_frames
ts = []
xs = []
ys = []
zs = []
gs = []
quats = []
for frame in video:
    rs = frame.robot_state
    p = rs.end_effector_poses["primary"]
    pos = fl(p.position)
    ts.append(float(frame.timestamp_s))
    xs.append(pos[0])
    ys.append(pos[1])
    zs.append(pos[2])
    gs.append(float(rs.gripper_positions["primary"]))
    quats.append(fl(p.quaternion_wxyz))
report["stage"] = "video_read"


def grip_label(g):
    if g > 0.85:
        return "O"
    if g < 0.12:
        return "C"
    return "H"


labels = [grip_label(g) for g in gs]
runs = []
i = 0
while i < n_frames:
    if labels[i] == "H":
        j = i
        while j + 1 < n_frames and labels[j + 1] == "H":
            j += 1
        if j - i + 1 >= 3:
            runs.append([i, j])
        i = j + 1
    else:
        i += 1


def lowest(lo, hi):
    lo = max(0, lo)
    hi = min(n_frames - 1, hi)
    best = lo
    for k in range(lo, hi + 1):
        if zs[k] < zs[best]:
            best = k
    return best


grasps = []
releases = []
for run in runs:
    a = run[0]
    b = run[1]
    k = lowest(a - 6, a + 2)
    grasps.append({"frame": k, "run": [a, b], "pos": [xs[k], ys[k], zs[k]], "quat": quats[k]})
    if b + 1 < n_frames:
        k = lowest(b - 2, b + 6)
        releases.append({"frame": k, "run": [a, b], "pos": [xs[k], ys[k], zs[k]], "quat": quats[k]})

troughs = []
if n_frames > 0:
    hyst = 0.03
    mode = "down"
    ext_i = 0
    for k in range(1, n_frames):
        if mode == "down":
            if zs[k] < zs[ext_i]:
                ext_i = k
            elif zs[k] > zs[ext_i] + hyst:
                troughs.append(ext_i)
                mode = "up"
                ext_i = k
        else:
            if zs[k] > zs[ext_i]:
                ext_i = k
            elif zs[k] < zs[ext_i] - hyst:
                mode = "down"
                ext_i = k
    if mode == "down" and zs[ext_i] < zs[0] - hyst:
        troughs.append(ext_i)
dips = []
for k in troughs:
    kind = "press"
    for gr in grasps:
        if abs(gr["frame"] - k) <= 8 or (gr["run"][0] - 8 <= k <= gr["run"][0] + 2):
            kind = "pick"
    for rl in releases:
        if abs(rl["frame"] - k) <= 8 or (rl["run"][1] - 2 <= k <= rl["run"][1] + 8):
            kind = "place"
    dips.append({"frame": k, "t": round(ts[k], 2), "pos": r4([xs[k], ys[k], zs[k]]), "g": round(gs[k], 3),
                 "kind": kind})
presses = [d for d in dips if d["kind"] == "press"]
report["video_grasps"] = [{"frame": g["frame"], "pos": r4(g["pos"])} for g in grasps]
report["video_releases"] = [{"frame": g["frame"], "pos": r4(g["pos"])} for g in releases]
report["video_dips"] = dips

# transit height: highest EE z while holding the cube in the video (carry height)
carry_z = None
for run in runs:
    for k in range(run[0], run[1] + 1):
        if carry_z is None or zs[k] > carry_z:
            carry_z = zs[k]
report["video_carry_z"] = carry_z
table_grasp_z = quantile([g["pos"][2] for g in grasps], 0.5)
report["video_grasp_z_median"] = table_grasp_z

# ---------------------------------------------------------------- choose the target release
ordinals = [("first", 0), ("second", 1), ("third", 2), ("fourth", 3), ("fifth", 4)]
k_index = 0
for word, idx in ordinals:
    if (word + " placed") in words or (word + " put") in words or (word + " time") in words:
        k_index = idx
direction = "after"
if "before" in words and "after" not in words:
    direction = "before"
report["parse"] = {"direction": direction, "k_index": k_index, "n_presses": len(presses),
                   "n_releases": len(releases)}
chosen = None
if len(presses) > 0 and len(releases) > 0:
    press_frame = presses[0]["frame"]
    report["press_frame_used"] = press_frame
    if direction == "after":
        after = [rl for rl in releases if rl["frame"] > press_frame]
        if len(after) > k_index:
            chosen = after[k_index]
        elif len(after) > 0:
            chosen = after[0]
    else:
        before = [rl for rl in releases if rl["frame"] < press_frame]
        if len(before) > k_index:
            chosen = before[len(before) - 1 - k_index]
        elif len(before) > 0:
            chosen = before[len(before) - 1]
if chosen is None and len(releases) > 0:
    chosen = releases[-1]
    report["notes"].append("fallback: last release")
if chosen is not None:
    report["chosen_release"] = {"frame": chosen["frame"], "pos": r4(chosen["pos"])}
try:
    if chosen is not None:
        report["thumb_chosen"] = thumb(video[chosen["frame"]].cameras["front"].rgb, 8)
    if len(presses) > 0:
        report["thumb_press"] = thumb(video[presses[0]["frame"]].cameras["front"].rgb, 8)
except Exception as exc:
    report["thumb_error"] = str(exc)
report["stage"] = "video_parsed"

# ---------------------------------------------------------------- live perception: the cube
colour = None
for c in ("red", "green", "blue", "yellow", "purple", "orange", "white", "black"):
    if (c + " cube") in words or (c + " block") in words:
        colour = c
cube_word = "cube"
if "block" in words and "cube" not in words:
    cube_word = "block"
prompts = []
if colour is not None:
    prompts.append(colour + " " + cube_word)
prompts.append(cube_word)
report["colour"] = colour

live = get_observation()
live_rgb = live.cameras["front"].rgb


def box_colour(box):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])
    cx1 = x1 + (x2 - x1) // 4
    cx2 = x2 - (x2 - x1) // 4
    cy1 = y1 + (y2 - y1) // 4
    cy2 = y2 - (y2 - y1) // 4
    sr = 0.0
    sg = 0.0
    sb = 0.0
    n = 0
    for y in range(max(0, cy1), min(255, cy2) + 1):
        for x in range(max(0, cx1), min(255, cx2) + 1):
            px = live_rgb[y][x]
            sr += float(px[0])
            sg += float(px[1])
            sb += float(px[2])
            n += 1
    if n == 0:
        return None
    return [sr / n, sg / n, sb / n]


def colour_ok(rgb, name):
    if rgb is None or name is None:
        return True
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    if name == "red":
        return r > g + 40 and r > b + 40
    if name == "green":
        return g > r + 30 and g > b + 20
    if name == "blue":
        return b > r + 30 and b > g + 20
    if name == "yellow":
        return r > b + 50 and g > b + 50
    return True


ref_xy = None
if len(releases) > 0:
    ref_xy = [releases[-1]["pos"][0], releases[-1]["pos"][1]]
cands = []
for prompt in prompts:
    seg = segment_text("front", prompt)
    if not seg.ok:
        continue
    for cand in seg.segmentations[:5]:
        if cand.box_xyxy is None:
            continue
        entry = {"prompt": prompt, "score": round(float(cand.score), 3),
                 "box": [round(float(v), 1) for v in cand.box_xyxy]}
        entry["rgb"] = box_colour(cand.box_xyxy)
        entry["colour_ok"] = colour_ok(entry["rgb"], colour)
        try:
            pc = mask_to_point_cloud(cand, "front", target_frame=base_frame)
            pts = pc.points
            entry["n"] = len(pts)
            if len(pts) >= 10:
                px = [float(p[0]) for p in pts]
                py = [float(p[1]) for p in pts]
                pz = [float(p[2]) for p in pts]
                entry["med"] = [quantile(px, 0.5), quantile(py, 0.5), quantile(pz, 0.5)]
                entry["span"] = [quantile(px, 0.95) - quantile(px, 0.05), quantile(py, 0.95) - quantile(py, 0.05),
                                 quantile(pz, 0.95) - quantile(pz, 0.05)]
                entry["top"] = quantile(pz, 0.95)
        except Exception as exc:
            entry["pc_error"] = str(exc)
        cands.append(entry)
    if colour is not None and any(e["colour_ok"] and "med" in e for e in cands):
        break
cube = None
best_cost = None
for e in cands:
    if "med" not in e or not e["colour_ok"]:
        continue
    if e["span"][0] > 0.09 or e["span"][1] > 0.09 or e["span"][2] > 0.09:
        continue
    cost = -e["score"]
    if ref_xy is not None:
        d = math.sqrt((e["med"][0] - ref_xy[0]) ** 2 + (e["med"][1] - ref_xy[1]) ** 2)
        e["d_last_release"] = round(d, 4)
        cost = cost + 2.0 * d
    if best_cost is None or cost < best_cost:
        best_cost = cost
        cube = e
if cube is not None:
    report["cube_selected"] = {"prompt": cube["prompt"], "score": cube["score"], "med": r4(cube["med"]), "top": round(cube["top"], 4)}
report["cube_cands"] = [{k: (r4(v) if k in ("med", "span", "rgb") and v is not None else v) for k, v in e.items()} for e in cands]
report["stage"] = "perceived"

# ---------------------------------------------------------------- target identity through swaps
colour_model = [0.0, 0.0, 0.0]
colour_tol = 45.0


def is_target_px(px):
    if abs(px[0] - colour_model[0]) > colour_tol:
        return False
    return abs(px[0] - colour_model[0]) + abs(px[1] - colour_model[1]) + abs(px[2] - colour_model[2]) <= colour_tol


def quat_to_rot(q):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def cam_model(cam):
    k = cam.intrinsics
    return {"fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]), "cy": float(k[1][2]),
            "R": quat_to_rot(unit(fl(cam.camera_pose.quaternion_wxyz))), "t": fl(cam.camera_pose.position)}


def project(model, p):
    R = model["R"]
    t = model["t"]
    d = [p[0] - t[0], p[1] - t[1], p[2] - t[2]]
    xc = R[0][0] * d[0] + R[1][0] * d[1] + R[2][0] * d[2]
    yc = R[0][1] * d[0] + R[1][1] * d[1] + R[2][1] * d[2]
    zc = R[0][2] * d[0] + R[1][2] * d[1] + R[2][2] * d[2]
    if zc <= 1e-6:
        return None
    return [model["fx"] * xc / zc + model["cx"], model["fy"] * yc / zc + model["cy"]]


def deproject(model, u, v, depth):
    xc = (u - model["cx"]) * depth / model["fx"]
    yc = (v - model["cy"]) * depth / model["fy"]
    R = model["R"]
    t = model["t"]
    return [R[0][0] * xc + R[0][1] * yc + R[0][2] * depth + t[0],
            R[1][0] * xc + R[1][1] * yc + R[1][2] * depth + t[1],
            R[2][0] * xc + R[2][1] * yc + R[2][2] * depth + t[2]]


def rows_of(arr, y0, y1, x0, x1, stride):
    try:
        return arr[y0:y1:stride, x0:x1:stride].tolist()
    except Exception:
        out = []
        for y in range(y0, y1, stride):
            row = []
            for x in range(x0, x1, stride):
                px = arr[y][x]
                row.append([int(px[0]), int(px[1]), int(px[2])])
            out.append(row)
        return out


live_cam = live.cameras["front"]
live_model = cam_model(live_cam)
img_h = len(live_rgb)
img_w = len(live_rgb[0])
full = rows_of(live_rgb, 0, img_h, 0, img_w, 1)
# ---- learn the target colour from the live image
def components(pts):
    # 8-connected components of a pixel set
    member = {}
    for p in pts:
        member[p[1] * img_w + p[0]] = p
    seen = {}
    comps = []
    for p in pts:
        key0 = p[1] * img_w + p[0]
        if key0 in seen:
            continue
        seen[key0] = True
        stack = [p]
        comp = {"sx": 0.0, "sy": 0.0, "n": 0, "pts": []}
        while len(stack) > 0:
            q = stack.pop()
            comp["sx"] += q[0]
            comp["sy"] += q[1]
            comp["n"] += 1
            comp["pts"].append(q)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    key = (q[1] + dy) * img_w + (q[0] + dx)
                    if key in member and key not in seen and 0 <= q[0] + dx < img_w:
                        seen[key] = True
                        stack.append(member[key])
        comps.append(comp)
    return comps


def concentric_groups(comps):
    # the rings of one bullseye are separate components sharing (almost) one centroid
    groups = []
    ordered = sorted([[c["n"], ci] for ci, c in enumerate(comps)], reverse=True)
    for item in ordered:
        c = comps[item[1]]
        cu = c["sx"] / c["n"]
        cv = c["sy"] / c["n"]
        hit = None
        for g in groups:
            if math.sqrt((g["sx"] / g["n"] - cu) ** 2 + (g["sy"] / g["n"] - cv) ** 2) <= 5.0:
                hit = g
                break
        if hit is None:
            groups.append({"sx": c["sx"], "sy": c["sy"], "n": c["n"], "pts": list(c["pts"])})
        else:
            hit["sx"] += c["sx"]
            hit["sy"] += c["sy"]
            hit["n"] += c["n"]
            hit["pts"] = hit["pts"] + c["pts"]
    return groups


def group_z(g):
    u = g["sx"] / g["n"]
    v = g["sy"] / g["n"]
    try:
        dep = float(live_cam.depth_m[int(round(v))][int(round(u))])
        return deproject(live_model, u, v, dep)[2]
    except Exception:
        return None


cand_px = []
bins = {}
for y in range(img_h):
    row = full[y]
    for x in range(img_w):
        px = row[x]
        r = px[0]
        g = px[1]
        b = px[2]
        if max(r, g, b) - min(r, g, b) < 60:
            continue
        if r >= g and g >= b:
            continue
        cand_px.append([x, y, r, g, b])
        key = (r // 8, g // 8, b // 8)
        if key not in bins:
            bins[key] = {"n": 0, "s": [0.0, 0.0, 0.0]}
        e = bins[key]
        e["n"] += 1
        e["s"][0] += r
        e["s"][1] += g
        e["s"][2] += b
ranked_bins = sorted([[bins[key]["n"], key] for key in bins], reverse=True)[:10]
seeds = []
for item in ranked_bins:
    if item[0] >= 20:
        e = bins[item[1]]
        seeds.append([e["s"][0] / e["n"], e["s"][1] / e["n"], e["s"][2] / e["n"]])
seed_info = []
for seed in seeds:
    pts = []
    sr = 0.0
    sg = 0.0
    sb = 0.0
    for cp in cand_px:
        if abs(cp[2] - seed[0]) + abs(cp[3] - seed[1]) + abs(cp[4] - seed[2]) <= colour_tol:
            pts.append([cp[0], cp[1]])
            sr += cp[2]
            sg += cp[3]
            sb += cp[4]
    groups = []
    for g in concentric_groups(components(pts)):
        if g["n"] >= 10:
            g["z"] = group_z(g)
            groups.append(g)
    if len(pts) > 0:
        seed_info.append({"seed": seed, "n": len(pts), "rgb": [sr / len(pts), sg / len(pts), sb / len(pts)],
                          "groups": groups})
all_z = []
for si in seed_info:
    for g in si["groups"]:
        if g["z"] is not None:
            all_z.append(g["z"])
z_floor = None
if len(all_z) > 0:
    z_floor = min(all_z)
best_seed = None
for si in seed_info:
    flat = []
    for g in si["groups"]:
        if g["z"] is None or z_floor is None or g["z"] <= z_floor + 0.015:
            flat.append(g)
    si["flat"] = flat
    if len(flat) >= 2:
        if best_seed is None or len(flat) > len(best_seed["flat"]) or (len(flat) == len(best_seed["flat"]) and si["n"] > best_seed["n"]):
            best_seed = si
report["colour_seeds"] = [{"seed": r4(si["seed"]), "n": si["n"], "flat": len(si["flat"]), "groups": len(si["groups"]),
                           "z": [None if g["z"] is None else round(g["z"], 4) for g in si["groups"]][:8]} for si in seed_info]
rests = []
if best_seed is not None:
    colour_model = best_seed["rgb"]
    report["target_colour_model"] = r4(colour_model)
    for g in best_seed["flat"]:
        if g["n"] >= 15:
            cx = g["sx"] / g["n"]
            cy = g["sy"] / g["n"]
            rad = 0.0
            for p in g["pts"]:
                dd = math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2)
                if dd > rad:
                    rad = dd
            rests.append({"u": cx, "v": cy, "n": g["n"], "rad": rad})
else:
    colour_model = [-999.0, -999.0, -999.0]
    report["notes"].append("no target colour learned")
report["rest_targets_px"] = [[round(r["u"], 1), round(r["v"], 1), r["n"], round(r["rad"], 1)] for r in rests]

swaps = []
track_ok = len(rests) >= 2 and n_frames > 0
if track_ok:
    umin = int(max(0, min(r["u"] for r in rests) - 40))
    umax = int(min(img_w, max(r["u"] for r in rests) + 40))
    vmin = int(max(0, min(r["v"] for r in rests) - 40))
    vmax = int(min(img_h, max(r["v"] for r in rests) + 40))
    report["track_roi"] = [umin, vmin, umax, vmax]
    per_frame = []
    for k in range(n_frames):
        rgbk = video[k].cameras["front"].rgb
        grid = rows_of(rgbk, vmin, vmax, umin, umax, 2)
        off = 0
        counts = [0 for r in rests]
        for gy in range(len(grid)):
            row = grid[gy]
            y = vmin + 2 * gy
            for gx in range(len(row)):
                px = row[gx]
                if is_target_px(px):
                    x = umin + 2 * gx
                    best = None
                    best_d = None
                    for ri in range(len(rests)):
                        dd = math.sqrt((x - rests[ri]["u"]) ** 2 + (y - rests[ri]["v"]) ** 2)
                        if best_d is None or dd < best_d:
                            best_d = dd
                            best = ri
                    if best_d > rests[best]["rad"] + 3.0:
                        off += 1
                        if best_d <= 35.0:
                            counts[best] += 1
        per_frame.append([off, counts])
    report["track_off_counts"] = [pf[0] for pf in per_frame]
    moving = [pf[0] >= 4 for pf in per_frame]
    k = 0
    while k < n_frames:
        if moving[k]:
            a = k
            b = k
            j = k + 1
            while j < n_frames and (moving[j] or (j + 1 < n_frames and moving[j + 1]) or (j + 2 < n_frames and moving[j + 2])):
                if moving[j]:
                    b = j
                j += 1
            if b - a + 1 >= 4:
                tot = [0 for r in rests]
                for f in list(range(a, min(a + 3, b + 1))) + list(range(max(a, b - 2), b + 1)):
                    for ri in range(len(rests)):
                        tot[ri] += per_frame[f][1][ri]
                ranked = []
                for ri in range(len(rests)):
                    ranked.append([tot[ri], ri])
                ranked = sorted(ranked, reverse=True)
                entry = {"start": a, "end": b, "counts": tot}
                if len(ranked) >= 2 and ranked[1][0] >= 2:
                    entry["pair"] = [ranked[0][1], ranked[1][1]]
                swaps.append(entry)
            k = b + 1
        else:
            k += 1
report["swaps"] = swaps


def nearest_rest(u, v):
    best = None
    best_d = None
    for ri in range(len(rests)):
        dd = math.sqrt((u - rests[ri]["u"]) ** 2 + (v - rests[ri]["v"]) ** 2)
        if best_d is None or dd < best_d:
            best_d = dd
            best = ri
    return best, best_d


def table_point(pos):
    # the cube centre at release sits ~half a cube above the target surface
    return [pos[0], pos[1], pos[2] - 0.02]


target_xy = None
if chosen is not None and track_ok:
    vmodel = cam_model(video[chosen["frame"]].cameras["front"])
    uv = project(vmodel, table_point(chosen["pos"]))
    if uv is not None:
        ri, dd = nearest_rest(uv[0], uv[1])
        report["chosen_rest"] = {"uv": r4(uv), "rest": ri, "dist_px": round(dd, 1)}
        if dd <= 15.0:
            cur = ri
            history = [ri]
            for sw in swaps:
                if sw["start"] > chosen["frame"] and "pair" in sw:
                    if sw["pair"][0] == cur:
                        cur = sw["pair"][1]
                        history.append(cur)
                    elif sw["pair"][1] == cur:
                        cur = sw["pair"][0]
                        history.append(cur)
            report["identity_history"] = history
            # 3-D position of the final rest centre: a video release made at that rest position, else depth
            for rl in releases:
                ruv = project(cam_model(video[rl["frame"]].cameras["front"]), table_point(rl["pos"]))
                if ruv is not None:
                    rj, dj = nearest_rest(ruv[0], ruv[1])
                    if rj == cur and dj <= 15.0:
                        target_xy = [rl["pos"][0], rl["pos"][1]]
                        report["target_source"] = "video release at frame %d" % rl["frame"]
            uu = int(round(rests[cur]["u"]))
            vv = int(round(rests[cur]["v"]))
            try:
                depth = float(live_cam.depth_m[vv][uu])
                dp = deproject(live_model, rests[cur]["u"], rests[cur]["v"], depth)
                report["target_deprojected"] = r4(dp)
                if target_xy is None:
                    target_xy = [dp[0], dp[1]]
                    report["target_source"] = "live depth deprojection"
            except Exception as exc:
                report["deproject_error"] = str(exc)
            report["target_rest_final"] = cur
if target_xy is None and chosen is not None:
    target_xy = [chosen["pos"][0], chosen["pos"][1]]
    report["notes"].append("identity tracking unavailable: using chosen release position")
if target_xy is not None:
    report["target_xy"] = r4(target_xy)


def live_off_pixels():
    try:
        obs = get_observation()
        grid = rows_of(obs.cameras["front"].rgb, vmin, vmax, umin, umax, 2)
        off = 0
        for gy in range(len(grid)):
            row = grid[gy]
            y = vmin + 2 * gy
            for gx in range(len(row)):
                px = row[gx]
                if is_target_px(px):
                    x = umin + 2 * gx
                    ri, dd = nearest_rest(x, y)
                    if dd > rests[ri]["rad"] + 3.0:
                        off += 1
        return off
    except Exception:
        return -1


report["stage"] = "tracked"

# ---------------------------------------------------------------- act
motions = []


def go(pos, quat, label, tol=0.015, steps=200):
    res = move_to_pose(Pose(tuple(pos), tuple(quat), base_frame), tolerance=tol, max_steps=steps)
    entry = {"label": label, "target": r4(pos), "ok": bool(res.ok)}
    try:
        entry["status"] = str(res.status)
        entry["steps"] = int(res.steps_executed)
    except Exception:
        pass
    motions.append(entry)
    try:
        if (not bool(res.ok)) and int(res.steps_executed) == 0 and not (bool(res.terminated) or bool(res.truncated)):
            res = move_to_pose(Pose(tuple(pos), tuple(quat), base_frame), tolerance=tol, max_steps=2 * steps)
            motions.append({"label": label + "_retry", "ok": bool(res.ok), "status": str(res.status),
                            "steps": int(res.steps_executed)})
    except Exception:
        pass
    return res


def ended(res):
    try:
        return bool(res.terminated) or bool(res.truncated)
    except Exception:
        return False


def hold(ticks, grip):
    st = get_robot_state()
    for tick in range(ticks):
        last = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=st.joint_positions["primary"],
            gripper_position=grip, embodiment=st.embodiment)}))
        try:
            if bool(last.terminated) or bool(last.truncated):
                return True
        except Exception:
            pass
    return False


finished = False
holding_now = 0.15 < start_grip < 0.85
report["holding_at_start"] = holding_now
grasp_quat = home_quat
if len(releases) > 0:
    grasp_quat = unit(releases[-1]["quat"])
if chosen is None:
    report["notes"].append("no release found in video; no target")
elif (not holding_now) and cube is None and ref_xy is None:
    report["notes"].append("no cube found")
else:
    transit = 0.15
    if carry_z is not None:
        transit = carry_z
    if not holding_now:
        if cube is not None:
            cx = cube["med"][0]
            cy = cube["med"][1]
            grasp_z = cube["top"] - 0.02
            if ref_xy is not None and math.sqrt((cx - ref_xy[0]) ** 2 + (cy - ref_xy[1]) ** 2) < 0.025:
                # the cube has not moved since the video's last release: use that release centre
                cx = ref_xy[0]
                cy = ref_xy[1]
                report["notes"].append("cube xy from last video release (live cloud within 2.5 cm)")
        else:
            cx = ref_xy[0]
            cy = ref_xy[1]
            grasp_z = releases[-1]["pos"][2]
            report["notes"].append("cube from video fallback")
        if table_grasp_z is not None:
            grasp_z = max(table_grasp_z - 0.01, min(table_grasp_z + 0.01, grasp_z))
        report["grasp_xyz"] = r4([cx, cy, grasp_z])
        transit = max(transit, grasp_z + 0.10)
        res = go([cx, cy, transit], grasp_quat, "above_cube")
        finished = ended(res)
        if not finished:
            res = go([cx, cy, grasp_z], grasp_quat, "grasp", tol=0.01)
            finished = ended(res)
        if not finished:
            cres = close_gripper()
            finished = ended(cres)
            try:
                report["close_steps"] = int(cres.steps_executed)
            except Exception:
                pass
            report["grip_after_close"] = float(get_robot_state().gripper_positions["primary"])
        if not finished:
            res = go([cx, cy, transit], grasp_quat, "lift")
            finished = ended(res)
            if not finished:
                report["grip_after_lift"] = float(get_robot_state().gripper_positions["primary"])
    if not finished:
        tx = target_xy[0]
        ty = target_xy[1]
        tz = chosen["pos"][2]
        res = go([tx, ty, transit], grasp_quat, "above_target")
        finished = ended(res)
        if not finished and track_ok:
            report["live_off_before_release"] = live_off_pixels()
        if not finished:
            res = go([tx, ty, tz + 0.005], grasp_quat, "release_pose", tol=0.01)
            finished = ended(res)
        if not finished:
            ores = open_gripper()
            finished = ended(ores)
            try:
                report["open_steps"] = int(ores.steps_executed)
            except Exception:
                pass
        if not finished:
            finished = hold(6, 1.0)
        if not finished:
            res = go([tx, ty, transit], grasp_quat, "retreat")
            finished = ended(res)
        if not finished:
            finished = hold(10, 1.0)

report["motions"] = motions
report["finished_by_env"] = finished
try:
    st = get_robot_state()
    report["final_ee"] = r4(st.end_effector_poses["primary"].position)
    report["final_t"] = float(st.timestamp_s)
except Exception as exc:
    report["final_state_error"] = str(exc)
report["stage"] = "done"
result = report
