# VideoPlaceOrder (robomme_plus task 12) -- v02 (demo-only, written offline).
# v02 vs v00: slots from the clean live frame; swap = two off-slot blobs symmetric about the
# midpoint of two vacant slots (no circle/separation model, no single-blob evidence); >=2 frames.
#
# Strategy (derived from the released recordings, see task_analysis.md):
#   1. Parse cube colour and ordinal N from get_task_context().language.
#   2. In the episode's own conditioning video (front camera), find the target "slots"
#      (purple/white discs at rest), detect pairwise target SWAPS (two targets rotate about
#      their midpoint and exchange slots), and detect PLACEMENTS of the goal-colour cube
#      (cube resting on a slot for a while).  Track target identities through the swaps.
#   3. The N-th placement names a target identity; follow it to its slot at the end of the
#      video, which is where it sits at the start of execution.
#   4. Pick the cube top-down (yaw aligned to its top face) and place it on that target.
# Everything episode-dependent comes from the goal string, the conditioning video and the
# live observation.  Only generic tolerances / offsets are constants.
import math

report = {"version": "v02"}

# ---------------------------------------------------------------- goal parsing
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "1st": 1, "2nd": 2,
            "3rd": 3, "4th": 4, "5th": 5, "6th": 6, "7th": 7, "8th": 8, "9th": 9, "10th": 10,
            "last": -1, "final": -1, "once": 1, "twice": 2}
COLOR_CODES = {"red": 0, "green": 1, "blue": 2, "yellow": 3}


def parse_goal(text):
    words = []
    for raw in str(text).lower().replace(",", " ").replace(".", " ").split():
        words.append(raw.strip())
    color = None
    for w in words:
        if w in COLOR_CODES:
            color = w
            break
    ordinal = None
    for w in words:
        if w in ORDINALS:
            ordinal = ORDINALS[w]
            break
    return color, ordinal


goal = get_task_context().language
color, ordinal = parse_goal(goal)
report["goal"] = goal
report["color"] = color
report["ordinal"] = ordinal
if color is None:
    color = "red"
    report["color_fallback"] = True
if ordinal is None:
    ordinal = 1
    report["ordinal_fallback"] = True
CC = COLOR_CODES[color]

meta = {}
try:
    meta = robomme_plus.get_task_metadata()
except Exception as exc:
    report["meta_error"] = str(exc)
max_steps = 0
control_hz = 20.0
try:
    max_steps = int(meta["max_steps"])
except Exception:
    max_steps = 0
try:
    control_hz = float(meta["control_hz"])
except Exception:
    control_hz = 20.0
report["max_steps"] = max_steps


# ---------------------------------------------------------------- pixel helpers
def to_rows(rgb):
    try:
        return rgb.tolist()
    except Exception:
        return [[[int(rgb[y][x][0]), int(rgb[y][x][1]), int(rgb[y][x][2])]
                 for x in range(len(rgb[0]))] for y in range(len(rgb))]


def is_target(r, g, b):
    # purple ring of a target disc; tolerant to coloured scene lighting
    if b <= g + 20 or b <= 80 or r * 2 <= b:
        return False
    mx = max(r, b)
    mn = min(r, g)
    return (mx - mn) * 10 > mx * 3


def is_cube(r, g, b):
    if CC == 0:
        return r > 100 and g * 10 < r * 3 and b * 10 < r * 3
    if CC == 1:
        return g > 90 and r * 10 < g * 5 and b * 10 < g * 5
    if CC == 2:
        return b > 90 and r * 10 < b * 4 and g * 10 < b * 4
    return r > 120 and g > 120 and b * 10 < min(r, g) * 4 and abs(r - g) * 10 < max(r, g) * 3


def scan(rows, stride):
    tcells = []
    ccells = []
    h = len(rows)
    w = len(rows[0])
    for y in range(0, h, stride):
        row = rows[y]
        gy = y // stride
        for x in range(0, w, stride):
            p = row[x]
            r = p[0]
            g = p[1]
            b = p[2]
            if b > g + 20:
                if is_target(r, g, b):
                    tcells.append((x // stride, gy))
                    continue
            if is_cube(r, g, b):
                ccells.append((x // stride, gy))
    return tcells, ccells


def blobs(cells, stride, min_cells):
    remaining = set(cells)
    out = []
    while len(remaining) > 0:
        seed = remaining.pop()
        stack = [seed]
        members = [seed]
        while len(stack) > 0:
            cur = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (cur[0] + dx, cur[1] + dy)
                    if q in remaining:
                        remaining.remove(q)
                        stack.append(q)
                        members.append(q)
        if len(members) >= min_cells:
            sx = 0.0
            sy = 0.0
            x0 = members[0][0]
            x1 = x0
            for m in members:
                sx += m[0]
                sy += m[1]
                if m[0] < x0:
                    x0 = m[0]
                if m[0] > x1:
                    x1 = m[0]
            n = len(members)
            out.append([sx / n * stride, sy / n * stride, n, (x1 - x0 + 1) * stride / 2.0])
    return out


def merge_close(bl, radius):
    # ring fragments of one disc share a centroid; merge them
    bl = sorted(bl, key=lambda b: -b[2])
    merged = []
    for b in bl:
        joined = False
        for m in merged:
            if math.hypot(m[0] - b[0], m[1] - b[1]) <= radius:
                n = m[2] + b[2]
                m[0] = (m[0] * m[2] + b[0] * b[2]) / n
                m[1] = (m[1] * m[2] + b[1] * b[2]) / n
                m[2] = n
                m[3] = max(m[3], b[3])
                joined = True
                break
        if not joined:
            merged.append([b[0], b[1], b[2], b[3]])
    return merged


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------------------------------------------------- video analysis
STRIDE = 2
video = ()
try:
    video = robomme_plus.get_conditioning_video()
except Exception as exc:
    report["video_error"] = str(exc)
report["video_frames"] = len(video)

times = []
frame_targets = []
frame_cube = []
frame_grip = []
for fi in range(len(video)):
    fr = video[fi]
    try:
        ts = float(fr.timestamp_s)
    except Exception:
        ts = fi * 0.05
    rows = to_rows(fr.cameras["front"].rgb)
    tc, cc = scan(rows, STRIDE)
    tb = merge_close(blobs(tc, STRIDE, 3), 6.0)
    cb = blobs(cc, STRIDE, 4)
    best = None
    for b in cb:
        if best is None or b[2] > best[2]:
            best = b
    times.append(ts)
    frame_targets.append(tb)
    frame_cube.append(best)
    gval = None
    try:
        gval = float(fr.robot_state.gripper_positions["primary"])
    except Exception:
        gval = None
    frame_grip.append(gval)
NF = len(times)

# --- slots: target discs at rest in the clean live frame (arm at home); video vote as fallback
obs = get_observation()
live_rows = to_rows(obs.cameras["front"].rgb)
lt, lc = scan(live_rows, 1)
live_targets = [b for b in merge_close(blobs(lt, 1, 6), 9.0) if b[2] >= 40]
report["live_targets_px"] = [[round(b[0], 1), round(b[1], 1), b[2]] for b in live_targets]
slots = []
for b in live_targets:
    slots.append([b[0], b[1], max(6.0, b[3], math.sqrt(b[2] / (0.5 * math.pi)))])
if len(slots) < 2:
    report["slots_from_video_vote"] = True
    slots = []
    pts = []
    for fi in range(NF):
        for b in frame_targets[fi]:
            if b[2] >= 5:
                pts.append(b)
    while len(pts) > 0 and len(slots) < 16:
        bins = {}
        for p in pts:
            key = (int(p[0] // 4), int(p[1] // 4))
            bins[key] = bins.get(key, 0) + 1
        bestkey = None
        for key in bins:
            if bestkey is None or bins[key] > bins[bestkey]:
                bestkey = key
        cx = bestkey[0] * 4 + 2.0
        cy = bestkey[1] * 4 + 2.0
        members = []
        for it in range(3):
            members = [p for p in pts if math.hypot(p[0] - cx, p[1] - cy) <= 6.0]
            if len(members) == 0:
                break
            cx = sum(p[0] for p in members) / len(members)
            cy = sum(p[1] for p in members) / len(members)
        if len(members) < max(3, int(0.25 * NF)):
            break
        radii = sorted(p[3] for p in members)
        slots.append([cx, cy, max(6.0, radii[len(radii) // 2])])
        pts = [p for p in pts if math.hypot(p[0] - cx, p[1] - cy) > 6.0]
NS = len(slots)
report["slots_px"] = [[round(s[0], 1), round(s[1], 1), round(s[2], 1)] for s in slots]


def slot_gate(s):
    return max(5.0, 0.6 * slots[s][2])


# --- per-frame occupancy and off-slot blobs
occ = []
offs = []
for fi in range(NF):
    o = [False] * NS
    off = []
    for b in frame_targets[fi]:
        near = -1
        for s in range(NS):
            if dist(b, slots[s]) <= slot_gate(s):
                near = s
                break
        if near >= 0:
            o[near] = True
        elif b[2] >= 6:
            off.append(b)
    occ.append(o)
    offs.append(off)

# --- swap evidence: two targets rotating about the midpoint of two vacant slots
pair_evidence = {}
for fi in range(NF):
    off = offs[fi]
    if len(off) == 0:
        continue
    o = occ[fi]
    cands = []
    for i in range(NS):
        if o[i]:
            continue
        for j in range(i + 1, NS):
            if o[j]:
                continue
            mx = (slots[i][0] + slots[j][0]) / 2.0
            my = (slots[i][1] + slots[j][1]) / 2.0
            d = dist(slots[i], slots[j])
            for a in range(len(off)):
                for c in range(a + 1, len(off)):
                    ma = ((off[a][0] + off[c][0]) / 2.0, (off[a][1] + off[c][1]) / 2.0)
                    e1 = math.hypot(ma[0] - mx, ma[1] - my)
                    e2 = abs(dist(off[a], off[c]) - d)
                    sep = dist(off[a], off[c])
                    if e1 <= max(4.0, 0.15 * d) and sep >= 0.3 * d and sep <= 3.0 * d + 10.0:
                        cands.append((e1 / max(4.0, 0.15 * d), i, j, a, c))
    cands = sorted(cands)
    used_s = set()
    used_b = set()
    for cand in cands:
        if cand[1] in used_s or cand[2] in used_s or cand[3] in used_b or cand[4] in used_b:
            continue
        used_s.add(cand[1])
        used_s.add(cand[2])
        used_b.add(cand[3])
        used_b.add(cand[4])
        pair_evidence.setdefault((cand[1], cand[2]), []).append((fi, 2))

events = []
for key in pair_evidence:
    ev = sorted(pair_evidence[key])
    group = []
    for item in ev:
        if len(group) > 0:
            last = group[-1][0]
            split = times[item[0]] - times[last] > 2.5
            if not split:
                for k in range(last + 1, item[0]):
                    if occ[k][key[0]] and occ[k][key[1]]:
                        split = True
                        break
            if split:
                events.append((key, group))
                group = []
        if len(group) == 0 or group[-1][0] != item[0]:
            group.append(item)
    if len(group) > 0:
        events.append((key, group))
# minimum evidence frames scales with the conditioning-video sampling interval:
# a swap lasts ~1-2 s, so sparse sampling (long videos) yields fewer frames per swap
gaps = sorted([times[k + 1] - times[k] for k in range(len(times) - 1)])
sample_dt = gaps[len(gaps) // 2] if len(gaps) > 0 else 0.05
min_strong = 2 if sample_dt < 0.15 else 1
swaps = []
for key, group in events:
    strong = sum(1 for g in group if g[1] == 2)
    weak = sum(1 for g in group if g[1] == 1)
    if strong >= min_strong:
        t0 = times[group[0][0]]
        t1 = times[group[-1][0]]
        swaps.append([(t0 + t1) / 2.0, key[0], key[1], t0, t1, 2 * strong + weak])
swaps = sorted(swaps)
# conflicts: overlapping events sharing a slot -> keep the better-supported one
kept = []
for sw in swaps:
    clash = -1
    for k in range(len(kept)):
        other = kept[k]
        share = sw[1] in (other[1], other[2]) or sw[2] in (other[1], other[2])
        if share and sw[3] <= other[4] + 0.2 and other[3] <= sw[4] + 0.2:
            clash = k
            break
    if clash < 0:
        kept.append(sw)
    elif sw[5] > kept[clash][5]:
        kept[clash] = sw
swaps = sorted(kept)
report["swaps"] = [[round(s[0], 2), s[1], s[2], s[5]] for s in swaps]

# --- placements: goal cube resting on a slot
def cube_slot(c):
    best = -1
    bestd = None
    for s in range(NS):
        r = slots[s][2]
        dd = math.hypot(c[0] - slots[s][0], (c[1] - slots[s][1]) + 0.5 * r)
        if dd <= 1.6 * r + 3.0 and (bestd is None or dd < bestd):
            best = s
            bestd = dd
    return best


runs = []
cur = None
for fi in range(NF):
    c = frame_cube[fi]
    if c is None:
        if cur is not None and times[fi] - times[cur["last"]] > 1.5:
            runs.append(cur)
            cur = None
        continue
    s = cube_slot(c)
    if cur is not None:
        if s == cur["slot"] and dist(c, cur["anchor"]) <= 3.0:
            cur["last"] = fi
            continue
        runs.append(cur)
        cur = None
    if s >= 0:
        cur = {"slot": s, "first": fi, "last": fi, "anchor": (c[0], c[1])}
if cur is not None:
    runs.append(cur)
rests = [r for r in runs if times[r["last"]] - times[r["first"]] >= 0.8]

# optional gripper check (video robot state): a placement releases the cube
grip_known = [g for g in frame_grip if g is not None]
use_grip = len(grip_known) > 0 and max(grip_known) - min(grip_known) > 0.3
gopen = max(grip_known) - 0.15 if use_grip else 0.0


def released(r):
    if not use_grip:
        return True
    for k in range(r["first"], min(NF, r["last"] + 1)):
        if frame_grip[k] is not None and frame_grip[k] >= gopen:
            return True
    return False


rests_g = [r for r in rests if released(r)]
if len(rests_g) >= max(1, ordinal):
    rests = rests_g
else:
    report["grip_check_dropped"] = True

placements = []
for r in rests:
    if len(placements) > 0 and placements[-1]["slot"] == r["slot"]:
        placements[-1]["last"] = r["last"]
        continue
    placements.append({"slot": r["slot"], "first": r["first"], "last": r["last"], "anchor": r["anchor"]})

# --- identity bookkeeping: perm[slot] = target identity
perm = list(range(NS))
place_ids = []
si = 0
for p in placements:
    tp = times[p["first"]]
    while si < len(swaps) and swaps[si][0] < tp:
        a = swaps[si][1]
        b = swaps[si][2]
        perm[a], perm[b] = perm[b], perm[a]
        si += 1
    place_ids.append(perm[p["slot"]])
while si < len(swaps):
    a = swaps[si][1]
    b = swaps[si][2]
    perm[a], perm[b] = perm[b], perm[a]
    si += 1
report["placements"] = [[round(times[p["first"]], 2), p["slot"], place_ids[k]] for k, p in enumerate(placements)]

chosen_slot = None
if len(place_ids) > 0:
    if ordinal == -1 or ordinal > len(place_ids):
        if ordinal != -1:
            report["ordinal_beyond_placements"] = True
        pick_id = place_ids[-1]
    else:
        pick_id = place_ids[ordinal - 1]
    for s in range(NS):
        if perm[s] == pick_id:
            chosen_slot = s
report["chosen_slot"] = chosen_slot

# ---------------------------------------------------------------- live scene
state0 = get_robot_state()
base = state0.base_frame
emb = state0.embodiment
t_start = float(state0.timestamp_s)

if chosen_slot is not None:
    goal_px = (slots[chosen_slot][0], slots[chosen_slot][1])
    goal_r = slots[chosen_slot][2]
else:
    # no evidence from the video: fall back to the first live target (recorded as such)
    report["no_video_evidence"] = True
    if len(live_targets) == 0:
        raise ValueError("no target visible and no video evidence")
    goal_px = (live_targets[0][0], live_targets[0][1])
    goal_r = live_targets[0][3]
best = None
for b in live_targets:
    if dist(b, goal_px) <= max(8.0, goal_r) and (best is None or dist(b, goal_px) < dist(best, goal_px)):
        best = b
if best is not None:
    goal_px = (best[0], best[1])
    goal_r = max(goal_r, best[3])
else:
    report["live_target_unconfirmed"] = True
report["goal_px"] = [round(goal_px[0], 1), round(goal_px[1], 1)]


def disc_mask(rows_img, center, radius, want_cube):
    m = []
    cnt = 0
    for y in range(len(rows_img)):
        row = rows_img[y]
        line = []
        for x in range(len(row)):
            ok = False
            if (x - center[0]) * (x - center[0]) + (y - center[1]) * (y - center[1]) <= radius * radius:
                p = row[x]
                if want_cube:
                    ok = is_cube(p[0], p[1], p[2])
                else:
                    ok = is_target(p[0], p[1], p[2])
            if ok:
                cnt += 1
            line.append(ok)
        m.append(line)
    return m, cnt


def cloud_points(mask_obj):
    pc = mask_to_point_cloud(mask_obj, "front", target_frame=base)
    return [(float(p[0]), float(p[1]), float(p[2])) for p in pc.points]


def seg_points_cloud(px):
    seg = segment_points("front", [[int(round(px[0])), int(round(px[1]))]])
    if not seg.ok:
        return []
    cand = seg.segmentations[0]
    try:
        return cloud_points(cand)
    except Exception:
        return cloud_points(cand.mask)


def median(vals):
    v = sorted(vals)
    return v[len(v) // 2]


def quant(vals, q):
    v = sorted(vals)
    return v[int((len(v) - 1) * q)]


# target 3D
tpts = []
try:
    tm, tcount = disc_mask(live_rows, goal_px, 1.4 * goal_r + 2.0, False)
    if tcount >= 8:
        tpts = cloud_points(tm)
except Exception as exc:
    report["target_mask_error"] = str(exc)
if len(tpts) < 8:
    try:
        tpts = seg_points_cloud(goal_px)
        report["target_cloud_source"] = "segment_points"
    except Exception as exc:
        report["target_seg_error"] = str(exc)
if len(tpts) < 5:
    raise ValueError("could not localize chosen target in 3D")
tx = median([p[0] for p in tpts])
ty = median([p[1] for p in tpts])
tz = median([p[2] for p in tpts])
report["target_xyz"] = [round(tx, 4), round(ty, 4), round(tz, 4)]


def locate_cube():
    rows_now = to_rows(get_observation().cameras["front"].rgb)
    ltn, lcn = scan(rows_now, 1)
    cubes = blobs(lcn, 1, 15)
    bestc = None
    for b in cubes:
        if bestc is None or b[2] > bestc[2]:
            bestc = b
    pts3 = []
    if bestc is not None:
        try:
            cm, ccount = disc_mask(rows_now, (bestc[0], bestc[1]), 2.0 * bestc[3] + 3.0, True)
            if ccount >= 8:
                pts3 = cloud_points(cm)
        except Exception as exc:
            report["cube_mask_error"] = str(exc)
    if len(pts3) < 8:
        seg = segment_text("front", color + " cube")
        if seg.ok and len(seg.segmentations) > 0:
            cand = seg.segmentations[0]
            try:
                pts3 = cloud_points(cand)
            except Exception:
                pts3 = cloud_points(cand.mask)
            report["cube_cloud_source"] = "segment_text"
    return pts3


cpts = locate_cube()
if len(cpts) < 5:
    raise ValueError("could not localize the " + color + " cube")


def cube_pose(points):
    ztop = quant([p[2] for p in points], 0.95)
    top = [p for p in points if p[2] >= ztop - 0.008]
    if len(top) < 4:
        top = points
    cxx = median([p[0] for p in top])
    cyy = median([p[1] for p in top])
    best_a = 0.0
    best_area = None
    for k in range(30):
        a = math.radians(3.0 * k)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [p[0] * ca + p[1] * sa for p in top]
        vs = [-p[0] * sa + p[1] * ca for p in top]
        area = (max(us) - min(us)) * (max(vs) - min(vs))
        if best_area is None or area < best_area:
            best_area = area
            best_a = a
    return cxx, cyy, ztop, best_a


cx3, cy3, ctop, cyaw = cube_pose(cpts)
cube_h = min(0.08, max(0.02, ctop - tz))
report["cube_xyz_top"] = [round(cx3, 4), round(cy3, 4), round(ctop, 4)]
report["cube_h"] = round(cube_h, 4)


# ---------------------------------------------------------------- orientation helpers
def qmul(a, b):
    return (a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])


def qrot(q, v):
    qv = (0.0, v[0], v[1], v[2])
    qc = (q[0], -q[1], -q[2], -q[3])
    r = qmul(qmul(q, qv), qc)
    return (r[1], r[2], r[3])


q_now = state0.end_effector_poses["primary"].quaternion_wxyz
q_now = (float(q_now[0]), float(q_now[1]), float(q_now[2]), float(q_now[3]))
if qrot(q_now, (0.0, 0.0, 1.0))[2] > -0.9:
    q_now = (0.0, 1.0, 0.0, 0.0)
    report["orientation_fallback"] = True
fax = qrot(q_now, (0.0, 1.0, 0.0))
finger_yaw = math.atan2(fax[1], fax[0])


def grasp_quat(face_yaw):
    d = face_yaw - finger_yaw
    while d > math.pi / 4.0:
        d -= math.pi / 2.0
    while d <= -math.pi / 4.0:
        d += math.pi / 2.0
    qz = (math.cos(d / 2.0), 0.0, 0.0, math.sin(d / 2.0))
    return qmul(qz, q_now)


# ---------------------------------------------------------------- motion helpers
def ticks_used():
    try:
        return (float(get_robot_state().timestamp_s) - t_start) * control_hz
    except Exception:
        return 0.0


def budget(want):
    if max_steps <= 0:
        return want
    left = max_steps - ticks_used() - 5
    return int(max(10, min(want, left)))


def terminal(res):
    for name in ("terminated", "truncated"):
        try:
            if name == "terminated" and bool(res.terminated):
                return True
            if name == "truncated" and bool(res.truncated):
                return True
        except Exception:
            pass
    return False


log = []


def go(xyz, q, tol, steps, label):
    pose = Pose((float(xyz[0]), float(xyz[1]), float(xyz[2])), q, base)
    res = move_to_pose(pose, tolerance=tol, max_steps=budget(steps))
    ok = bool(res.ok)
    try:
        n = int(res.steps_executed)
    except Exception:
        n = -1
    log.append([label, ok, n])
    if terminal(res):
        report["log"] = log
        report["terminal_at"] = label
        raise ValueError("episode terminal during " + label)
    if not ok and n == 0:
        res = move_to_pose(pose, tolerance=tol * 2.0, max_steps=budget(steps),
                           strategy=MotionStrategy(trajectory_planner="curobo"))
        ok = bool(res.ok)
        log.append([label + "_retry_curobo", ok])
        if terminal(res):
            report["log"] = log
            raise ValueError("episode terminal during " + label)
    return ok


def grip_value():
    try:
        return float(get_robot_state().gripper_positions["primary"])
    except Exception:
        return -1.0


# ---------------------------------------------------------------- pick
qg = grasp_quat(cyaw)
hover_z = tz + 0.15
grasp_z = max(ctop - 0.5 * cube_h, tz + 0.012)
open_gripper()
held = False
for attempt in range(2):
    go((cx3, cy3, ctop + 0.10), qg, 0.01, 150, "pregrasp")
    go((cx3, cy3, grasp_z), qg, 0.008, 100, "grasp")
    close_gripper()
    go((cx3, cy3, hover_z), qg, 0.015, 100, "lift")
    gv = grip_value()
    log.append(["grip_after_lift", round(gv, 3)])
    held = gv > 0.05 or gv < 0.0
    if held:
        break
    if max_steps > 0 and ticks_used() > 0.6 * max_steps:
        break
    open_gripper()
    cpts = locate_cube()
    if len(cpts) < 5:
        break
    cx3, cy3, ctop, cyaw = cube_pose(cpts)
    qg = grasp_quat(cyaw)
    grasp_z = max(ctop - 0.5 * cube_h, tz + 0.012)
report["held"] = held

# ---------------------------------------------------------------- place
place_z = tz + 0.5 * cube_h + 0.012
go((tx, ty, hover_z), qg, 0.015, 150, "above_target")
go((tx, ty, place_z), qg, 0.008, 100, "place")
open_gripper()
if max_steps <= 0 or ticks_used() < max_steps - 30:
    go((tx, ty, place_z + 0.08), qg, 0.02, 60, "retreat")
report["log"] = log
report["ticks_used"] = round(ticks_used(), 1)
result = report
