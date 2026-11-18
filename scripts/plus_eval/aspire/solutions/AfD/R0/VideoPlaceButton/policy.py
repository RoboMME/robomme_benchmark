# VideoPlaceButton -- v05 slot-hold tracking: a target parked on its slot holds that slot while another target passes over it (on top of v03; v04 discarded) (ASPIRE Stage 0, AfD: written offline, never run in simulator).
#
# Goal form (from get_task_context().language, parsed at runtime):
#   "watch the video carefully, then place the <colour> cube on the target where it was
#    first placed after the button was pressed"
#
# Strategy (derived from the released training recordings' STRUCTURE, no recorded coordinates):
#   1. Read the episode's own conditioning video (robomme_plus.get_conditioning_video()).
#   2. Find the first button press (EE pose near the live-localised button; image fallback:
#      the button's white pixels in the front view are occluded by the arm).
#   3. After the press, find the first time the goal-coloured cube MOVES and then comes to
#      rest on a target slot -> that target object is X.
#   4. Targets later swap positions (identical-looking bullseyes rotate about their midpoint),
#      so track every target blob frame-to-frame from that moment to the end of the video and
#      report where X ends up.
#   5. In the live scene: pick the cube, place it on the target at X's final slot.
# Everything episode-dependent (cube colour, button, target slots, press time, placement, swaps)
# is derived at runtime. Only generic tolerances/offsets are constants.

import math

report = {"version": "v05_slot_hold_tracking"}

# ---------------------------------------------------------------- goal parsing
context = get_task_context()
goal = str(context.language)
report["goal"] = goal
goal_l = goal.lower()
COLOR_NAMES = ("red", "blue", "green", "yellow", "purple", "orange", "pink", "white", "black", "cyan")
cube_color = None
words = goal_l.replace(",", " ").replace(".", " ").split()
for wi in range(len(words) - 1):
    if words[wi + 1] in ("cube", "block") and words[wi] in COLOR_NAMES:
        cube_color = words[wi]
        break
if cube_color is None:
    for name in COLOR_NAMES:
        if name in words:
            cube_color = name
            break
if cube_color is None:
    raise ValueError("could not parse cube colour from goal: " + goal)
report["cube_color"] = cube_color

try:
    task_meta = robomme_plus.get_task_metadata()
    report["max_steps"] = task_meta["max_steps"]
except Exception as exc:
    task_meta = {}
    report["metadata_error"] = str(exc)

state0 = get_robot_state()
base_frame = state0.base_frame
CAM = "front"
live = get_observation()
if CAM not in live.cameras:
    raise ValueError("front camera missing: " + str(sorted(live.cameras)))


# ---------------------------------------------------------------- pixel helpers (pure Python)
STRIDE = 2   # analyse every 2nd pixel in x and y


def image_rows(rgb):
    """Return the image as nested Python lists (fast path), or the array itself."""
    try:
        return rgb.tolist()
    except Exception:
        return rgb


def is_cube_px(r, g, b, color):
    if color == "red":
        return r > 140 and g < 80 and b < 80
    if color == "blue":
        return b > 120 and r < 70 and g < 70
    if color == "green":
        return g > 130 and r < 100 and b < 100
    if color == "yellow":
        return r > 160 and g > 160 and b < 90
    if color == "purple":
        return r > 90 and b > 120 and g < 70
    if color == "orange":
        return r > 200 and 90 < g < 170 and b < 60
    if color == "pink":
        return r > 200 and b > 140 and g < 140
    if color == "white":
        return r > 215 and g > 215 and b > 215
    if color == "black":
        return r < 40 and g < 40 and b < 40
    if color == "cyan":
        return g > 150 and b > 150 and r < 90
    return False


ring_ref = None   # (r, g, b) reference colour of target rings, set from the live scene
wood_ref = None   # (r, g, b) dominant table colour, set from the live scene


def is_magenta_px(r, g, b):
    # seed rule for the ring colour when no SAM mask is readable (pink/purple family)
    return r > 120 and r - g > 40 and b - g > 10 and b > 100


def is_target_px(r, g, b):
    if ring_ref is None:
        return is_magenta_px(r, g, b)
    dr = r - ring_ref[0]
    dg = g - ring_ref[1]
    db = b - ring_ref[2]
    d2 = dr * dr + dg * dg + db * db
    if d2 > 45 * 45 or abs(dg) >= 25:
        return False
    if max(r, g, b) - min(r, g, b) <= 30:
        return False
    if wood_ref is not None:
        wr = r - wood_ref[0]
        wg = g - wood_ref[1]
        wb = b - wood_ref[2]
        if d2 >= 0.36 * (wr * wr + wg * wg + wb * wb):
            return False
    return True


def is_white_px(r, g, b):
    return r > 215 and g > 215 and b > 215


def classify(rows, height, width, want_target, want_cube, button_box):
    """One pass over a frame: target cells, cube cells and button-white count."""
    tcells = []
    ccells = []
    white = 0
    bx1 = by1 = bx2 = by2 = -1
    if button_box is not None:
        bx1, by1, bx2, by2 = button_box
    for y in range(0, height, STRIDE):
        row = rows[y]
        for x in range(0, width, STRIDE):
            px = row[x]
            r = int(px[0])
            g = int(px[1])
            b = int(px[2])
            if want_cube and is_cube_px(r, g, b, cube_color):
                ccells.append((x // STRIDE, y // STRIDE))
            elif want_target and is_target_px(r, g, b):
                tcells.append((x // STRIDE, y // STRIDE))
            if bx1 <= x <= bx2 and by1 <= y <= by2 and is_white_px(r, g, b):
                white += 1
    return tcells, ccells, white


def components(cells, link):
    """Connected components on the cell grid; cells within `link` cells are joined
    (so the concentric rings of one bullseye form one component)."""
    remaining = set(cells)
    comps = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        members = [seed]
        while stack:
            cx, cy = stack.pop()
            for dx in range(-link, link + 1):
                for dy in range(-link, link + 1):
                    nb = (cx + dx, cy + dy)
                    if nb in remaining:
                        remaining.discard(nb)
                        stack.append(nb)
                        members.append(nb)
        sx = 0.0
        sy = 0.0
        for m in members:
            sx += m[0]
            sy += m[1]
        n = len(members)
        comps.append((STRIDE * sx / n + 0.5, STRIDE * sy / n + 0.5, n))
    return comps


def dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


# ---------------------------------------------------------------- live scene: references
front_live = live.cameras["front"]
live_rows = image_rows(front_live.rgb)
H = len(live_rows)
W = len(live_rows[0])
report["image_hw"] = [H, W]


def seg_candidates(prompt):
    out = []
    try:
        found = segment_text(CAM, prompt)
    except Exception as exc:
        report.setdefault("seg_errors", []).append(prompt + ": " + str(exc))
        return out
    try:
        if not found.ok:
            return out
        for cand in found.segmentations[:10]:
            try:
                box = [float(v) for v in cand.box_xyxy]
            except Exception:
                continue
            try:
                score = float(cand.score)
            except Exception:
                score = 0.0
            out.append((cand, box, score))
    except Exception as exc:
        report.setdefault("seg_errors", []).append(prompt + " parse: " + str(exc))
    return out


def cloud_points(cand):
    try:
        cloud = mask_to_point_cloud(cand, CAM, target_frame=base_frame)
        pts = cloud.points
        if len(pts) >= 10:
            return cloud, pts
    except Exception as exc:
        report.setdefault("cloud_errors", []).append(str(exc))
    return None, None


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    return ordered[int((len(ordered) - 1) * fraction)]


# ring colour from SAM target masks (falls back to the generic magenta rule)
target_boxes = []
for prompt in ("bullseye target", "target", "round target on the table"):
    cands = seg_candidates(prompt)
    for cand, box, score in cands:
        if box[2] - box[0] < W * 0.25 and box[3] - box[1] < H * 0.25 and score >= 0.2:
            target_boxes.append((cand, box, score))
    if len(target_boxes) >= 2:
        report["target_prompt"] = prompt
        break
samples = []
for cand, box, score in target_boxes:
    try:
        mask = cand.mask
        for y in range(int(box[1]), int(box[3]) + 1):
            for x in range(int(box[0]), int(box[2]) + 1):
                if 0 <= y < H and 0 <= x < W and bool(mask[y][x]):
                    px = live_rows[y][x]
                    r = int(px[0])
                    g = int(px[1])
                    b = int(px[2])
                    if max(r, g, b) - min(r, g, b) > 40 and not is_white_px(r, g, b):
                        samples.append((r, g, b))
    except Exception as exc:
        report["ring_mask_error"] = str(exc)
        break
ring_source = "sam_mask"
if len(samples) < 30:
    # no readable SAM mask: seed the ring colour from the magenta rule inside SAM target boxes
    # (or the whole image when SAM returned no target boxes)
    ring_source = "magenta_seed"
    samples = []
    regions = [b for c, b, s in target_boxes] if target_boxes else [[0, 0, W - 1, H - 1]]
    for box in regions:
        for y in range(max(0, int(box[1])), min(H, int(box[3]) + 1)):
            for x in range(max(0, int(box[0])), min(W, int(box[2]) + 1)):
                px = live_rows[y][x]
                r = int(px[0])
                g = int(px[1])
                b = int(px[2])
                if is_magenta_px(r, g, b) and max(r, g, b) - min(r, g, b) > 40 and not is_white_px(r, g, b):
                    samples.append((r, g, b))
if len(samples) >= 30:
    ring_ref = (quantile([s[0] for s in samples], 0.5),
                quantile([s[1] for s in samples], 0.5),
                quantile([s[2] for s in samples], 0.5))
else:
    ring_source = "magenta_rule"
wood_samples = []
for y in range(0, H, 4):
    for x in range(0, W, 4):
        px = live_rows[y][x]
        wood_samples.append((int(px[0]), int(px[1]), int(px[2])))
wood_ref = (quantile([s[0] for s in wood_samples], 0.5),
            quantile([s[1] for s in wood_samples], 0.5),
            quantile([s[2] for s in wood_samples], 0.5))
report["ring_ref"] = ring_ref
report["ring_source"] = ring_source
report["wood_ref"] = wood_ref

# button: SAM box (small, high score) and 3D centre
button_box = None
button_xyz = None
best = None
for prompt in ("white button", "button", "white square button"):
    for cand, box, score in seg_candidates(prompt):
        if box[2] - box[0] < W * 0.3 and box[3] - box[1] < H * 0.3:
            if best is None or score > best[2]:
                best = (cand, box, score)
    if best is not None and best[2] >= 0.3:
        break
if best is not None:
    button_box = (int(best[1][0]), int(best[1][1]), int(best[1][2]), int(best[1][3]))
    cloud, pts = cloud_points(best[0])
    if pts is not None:
        button_xyz = (quantile([p[0] for p in pts], 0.5), quantile([p[1] for p in pts], 0.5),
                      quantile([p[2] for p in pts], 0.9))
report["button_box"] = button_box
report["button_xyz"] = button_xyz

# live target slots (targets only ever exchange positions)
tcells, ccells, live_white = classify(live_rows, H, W, True, True, button_box)
raw_slots = [c for c in components(tcells, 2) if c[2] >= 6]
slot_sizes = sorted([c[2] for c in raw_slots], reverse=True)
min_slot = 6
if slot_sizes:
    min_slot = max(6, 0.4 * slot_sizes[(len(slot_sizes) - 1) // 2])
slots = [c for c in raw_slots if c[2] >= min_slot]
report["live_slots"] = [[round(s[0], 1), round(s[1], 1), s[2]] for s in slots]
report["live_cube_px"] = None
if ccells:
    biggest = None
    for c in components(ccells, 1):
        if biggest is None or c[2] > biggest[2]:
            biggest = c
    report["live_cube_px"] = [round(biggest[0], 1), round(biggest[1], 1), biggest[2]]

# ---------------------------------------------------------------- conditioning video analysis
video = robomme_plus.get_conditioning_video()
nframes = len(video)
report["video_frames"] = nframes
if nframes < 2:
    raise ValueError("no conditioning video; cannot resolve which target")

frames_info = []   # per frame: (t, cube_px or None, targets, white_count, ee_xyz or None)
for fi in range(nframes):
    fr = video[fi]
    try:
        t = float(fr.timestamp_s)
    except Exception:
        t = float(fi)
    rows = image_rows(fr.cameras["front"].rgb)
    tc, cc, wc = classify(rows, len(rows), len(rows[0]), True, True, button_box)
    targets = [c for c in components(tc, 2) if c[2] >= 5]
    cube = None
    if len(cc) >= 3:
        for c in components(cc, 1):
            if cube is None or c[2] > cube[2]:
                cube = c
    ee = None
    try:
        p = fr.robot_state.end_effector_poses["primary"].position
        ee = (float(p[0]), float(p[1]), float(p[2]))
    except Exception:
        ee = None
    frames_info.append((t, cube, targets, wc, ee))

# slots: union of live slots and the first video frame's targets (dedupe within 6 px)
for c in frames_info[0][2]:
    if c[2] >= min_slot:
        near = False
        for s in slots:
            if dist2(s, c) < 36:
                near = True
        if not near:
            slots.append(c)
if len(slots) < 2:
    raise ValueError("fewer than two target slots detected")

# ---- 2. first button press
press_idx = None
press_method = None
if button_xyz is not None:
    for fi in range(nframes):
        ee = frames_info[fi][4]
        if ee is None:
            continue
        dxy = math.sqrt((ee[0] - button_xyz[0]) ** 2 + (ee[1] - button_xyz[1]) ** 2)
        if dxy < 0.04 and ee[2] < button_xyz[2] + 0.06:
            press_idx = fi
            press_method = "ee_near_button"
            break
if press_idx is None and button_box is not None:
    baseline = max(live_white, frames_info[0][3])
    if baseline >= 8:
        for fi in range(nframes):
            if frames_info[fi][3] < 0.8 * baseline:
                press_idx = fi
                press_method = "button_occluded_front"
                break
if press_idx is None:
    raise ValueError("button press not found in conditioning video")
report["press"] = {"index": press_idx, "t": frames_info[press_idx][0], "method": press_method}

# ---- 3. first placement of the cube on a target after the press
ref = None
for fi in range(press_idx, -1, -1):
    if frames_info[fi][1] is not None and frames_info[fi][1][2] >= 6:
        ref = frames_info[fi][1]
        break
moved = ref is None
place_idx = None
x_slot = None
REST_PX2 = 2.5 * 2.5
REST_S = 1.5        # a carried cube can pause ~0.5 s; a placed cube rests for seconds
PLACE_SLOT_PX = 10.0
for fi in range(press_idx + 1, nframes):
    cube = frames_info[fi][1]
    if cube is None:
        continue
    if not moved:
        if dist2(cube, ref) > 8 * 8:
            moved = True
        continue
    # rest: detected and still over the following >= 0.5 s
    j = fi + 1
    still = True
    while j < nframes and frames_info[j][0] - frames_info[fi][0] < REST_S:
        cj = frames_info[j][1]
        if cj is None or dist2(cj, cube) > REST_PX2:
            still = False
            break
        j += 1
    if not still or j >= nframes:
        continue
    nearest = None
    for si in range(len(slots)):
        d = dist2(slots[si], cube)
        if nearest is None or d < nearest[0]:
            nearest = (d, si)
    if nearest is not None and nearest[0] <= PLACE_SLOT_PX * PLACE_SLOT_PX:
        place_idx = fi
        x_slot = nearest[1]
        break
if place_idx is None:
    raise ValueError("no post-press placement on a target found")
report["placement"] = {"index": place_idx, "t": frames_info[place_idx][0],
                       "cube_px": [round(frames_info[place_idx][1][0], 1), round(frames_info[place_idx][1][1], 1)],
                       "slot_px": [round(slots[x_slot][0], 1), round(slots[x_slot][1], 1)]}

# ---- 4. track all targets from the placement to the end of the video
# Targets only exchange slots. Tracks are velocity-gated; whenever every slot is occupied by
# exactly one detection and nothing is off-slot ("rest frame"), tracks are snapped back onto
# slots, so a track lost mid-swap cannot stay stranded and later steal another target.
SLOT_R2 = 6.0 * 6.0
HOLD_R2 = 8.0 * 8.0      # a parked target's own detection stays within this of its slot
MERGE_FACTOR = 1.5       # a blob this much larger than the slot's own size is two targets touching
holds = 0
tracks = [(s[0], s[1]) for s in slots]
vels = [(0.0, 0.0) for s in slots]
lost = [0 for s in slots]
x_track = x_slot
rest_frames = 0
snaps = 0


def nearest_slot(pt):
    best_i = None
    best_d = None
    for si in range(len(slots)):
        d = dist2(slots[si], pt)
        if best_d is None or d < best_d:
            best_d = d
            best_i = si
    return best_i, best_d


for fi in range(place_idx + 1, nframes):
    dt = max(frames_info[fi][0] - frames_info[fi - 1][0], 0.02)
    max_jump = 8.0 + 100.0 * dt
    dets = frames_info[fi][2]
    preds = []
    for ti in range(len(tracks)):
        if lost[ti] <= 5:
            preds.append((tracks[ti][0] + vels[ti][0] * dt, tracks[ti][1] + vels[ti][1] * dt))
        else:
            preds.append(tracks[ti])
    # slot hold: a track parked on a slot keeps it while a detection still covers that slot
    # (its own blob, or a merged blob when another target passes over it); a passing target
    # is then coasted as "lost" instead of stealing the parked identity.
    held_t = set()
    held_d = set()
    for ti in range(len(tracks)):
        if lost[ti] != 0:
            continue
        tsi, tsd = nearest_slot(tracks[ti])
        if tsd > SLOT_R2:
            continue
        best_di = None
        best_dd = None
        for di in range(len(dets)):
            dd = dist2(slots[tsi], dets[di])
            if best_dd is None or dd < best_dd:
                best_dd = dd
                best_di = di
        if best_di is None:
            continue
        merged = dets[best_di][2] >= MERGE_FACTOR * slots[tsi][2] and best_dd <= max_jump * max_jump
        if best_dd <= HOLD_R2 or merged:
            held_t.add(ti)
            held_d.add(best_di)
            tracks[ti] = (slots[tsi][0], slots[tsi][1])
            vels[ti] = (0.0, 0.0)
            if merged and best_dd > HOLD_R2:
                holds += 1
    pairs = []
    for ti in range(len(tracks)):
        if ti in held_t:
            continue
        gate = max_jump * (1.0 + 0.5 * min(lost[ti], 4))
        for di in range(len(dets)):
            if di in held_d:
                continue
            d = dist2(preds[ti], dets[di])
            if d <= gate * gate:
                pairs.append((d, ti, di))
    pairs = sorted(pairs)
    used_t = set()
    used_d = set()
    for d, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        nx = dets[di][0]
        ny = dets[di][1]
        steps_gone = lost[ti] + 1
        vx = (nx - tracks[ti][0]) / (dt * steps_gone)
        vy = (ny - tracks[ti][1]) / (dt * steps_gone)
        vels[ti] = (0.5 * vels[ti][0] + 0.5 * vx, 0.5 * vels[ti][1] + 0.5 * vy)
        tracks[ti] = (nx, ny)
        lost[ti] = 0
    for ti in range(len(tracks)):
        if ti not in used_t and ti not in held_t:
            lost[ti] += 1
            vels[ti] = (0.7 * vels[ti][0], 0.7 * vels[ti][1])
    # rest frame?
    occupied = [0 for s in slots]
    off_slot = 0
    for det in dets:
        si, d = nearest_slot(det)
        if d <= SLOT_R2:
            occupied[si] += 1
        else:
            off_slot += 1
    if off_slot == 0 and min(occupied) == 1 and max(occupied) == 1:
        rest_frames += 1
        owner = [[] for s in slots]
        off_tracks = []
        for ti in range(len(tracks)):
            si, d = nearest_slot(tracks[ti])
            if d <= SLOT_R2:
                owner[si].append(ti)
            else:
                off_tracks.append(ti)
        for si in range(len(slots)):
            if len(owner[si]) > 1:
                keep = owner[si][0]
                for ti in owner[si]:
                    if lost[ti] < lost[keep]:
                        keep = ti
                for ti in owner[si]:
                    if ti != keep:
                        off_tracks.append(ti)
                owner[si] = [keep]
        free = [si for si in range(len(slots)) if len(owner[si]) == 0]
        if off_tracks:
            snaps += 1
            cand = []
            for ti in off_tracks:
                for si in free:
                    cand.append((dist2(tracks[ti], slots[si]), ti, si))
            cand = sorted(cand)
            done_t = set()
            done_s = set()
            for d, ti, si in cand:
                if ti in done_t or si in done_s:
                    continue
                done_t.add(ti)
                done_s.add(si)
                owner[si] = [ti]
        for si in range(len(slots)):
            for ti in owner[si]:
                tracks[ti] = (slots[si][0], slots[si][1])
                vels[ti] = (0.0, 0.0)
                lost[ti] = 0
report["tracking"] = {"rest_frames": rest_frames, "snaps": snaps, "merge_holds": holds}

final_px = tracks[x_track]
best_slot = None
for si in range(len(slots)):
    d = dist2(slots[si], final_px)
    if best_slot is None or d < best_slot[0]:
        best_slot = (d, si)
goal_slot = slots[best_slot[1]]
report["x_final_track_px"] = [round(final_px[0], 1), round(final_px[1], 1)]
report["target_px"] = [round(goal_slot[0], 1), round(goal_slot[1], 1)]
report["target_slot_residual_px"] = round(math.sqrt(best_slot[0]), 1)

# ---------------------------------------------------------------- live 3D localisation
tgt_cand = None
for cand, box, score in target_boxes:
    cx = 0.5 * (box[0] + box[2])
    cy = 0.5 * (box[1] + box[3])
    d = (cx - goal_slot[0]) ** 2 + (cy - goal_slot[1]) ** 2
    if d <= 15 * 15 and (tgt_cand is None or d < tgt_cand[0]):
        tgt_cand = (d, cand)
tgt_pts = None
if tgt_cand is not None:
    cloud, tgt_pts = cloud_points(tgt_cand[1])
if tgt_pts is None:
    try:
        sp = segment_points(CAM, [[int(goal_slot[0]), int(goal_slot[1])]])
        if sp.ok and len(sp.segmentations) > 0:
            cloud, tgt_pts = cloud_points(sp.segmentations[0])
    except Exception as exc:
        report["segment_points_error"] = str(exc)
if tgt_pts is None:
    raise ValueError("could not localise the chosen target in 3D")
target_xyz = (quantile([p[0] for p in tgt_pts], 0.5), quantile([p[1] for p in tgt_pts], 0.5),
              quantile([p[2] for p in tgt_pts], 0.9))
report["target_xyz"] = [round(v, 4) for v in target_xyz]

cube_loc = None
for prompt in (cube_color + " cube", cube_color + " block"):
    found = localize_object(prompt, camera_name=CAM, target_frame=base_frame)
    if found.ok and len(found.point_cloud.points) >= 20:
        cube_loc = found
        break
if cube_loc is None:
    raise ValueError("cube not localised")
cpts = cube_loc.point_cloud.points
try:
    cpos = cube_loc.geometry.pose.position
    cube_xy = (float(cpos[0]), float(cpos[1]))
except Exception:
    cube_xy = (quantile([p[0] for p in cpts], 0.5), quantile([p[1] for p in cpts], 0.5))
cube_top = quantile([p[2] for p in cpts], 0.95)
cube_low = quantile([p[2] for p in cpts], 0.05)
cube_h = max(0.02, min(0.06, cube_top - cube_low))
report["cube_xyz"] = [round(cube_xy[0], 4), round(cube_xy[1], 4), round(cube_top, 4)]

# ---------------------------------------------------------------- execution
down = state0.end_effector_poses["primary"].quaternion_wxyz
EXEC = {}


def go(xyz, label, tol=0.015, steps=200):
    res = move_to_pose(Pose((xyz[0], xyz[1], xyz[2]), down, base_frame), tolerance=tol, max_steps=steps)
    EXEC[label] = str(res.status) if not res.ok else "ok"
    try:
        if res.terminated or res.truncated:
            EXEC[label + "_terminal"] = True
            return None
    except Exception:
        pass
    return res


open_gripper()
grasp_z = cube_top - 0.5 * cube_h
if go((cube_xy[0], cube_xy[1], cube_top + 0.10), "pre_grasp") is None:
    raise ValueError("terminal before grasp")
go((cube_xy[0], cube_xy[1], grasp_z), "grasp", tol=0.01)
close_gripper()
go((cube_xy[0], cube_xy[1], cube_top + 0.12), "lift")
go((target_xyz[0], target_xyz[1], target_xyz[2] + cube_h + 0.12), "above_target")
go((target_xyz[0], target_xyz[1], target_xyz[2] + 0.5 * cube_h + 0.015), "place", tol=0.01)
open_gripper()
go((target_xyz[0], target_xyz[1], target_xyz[2] + cube_h + 0.10), "retreat")
report["exec"] = EXEC
result = report
