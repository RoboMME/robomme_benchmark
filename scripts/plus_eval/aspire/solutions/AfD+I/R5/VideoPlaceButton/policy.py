# VideoPlaceButton v01 -- v00 + dip classification from the gripper inside the low band (press = closed-empty).
#
# Strategy (all quantities derived at runtime):
#   1. Read the goal colour word ("red"/"blue"/"green" cube) from get_task_context().language.
#   2. Conditioning video robot_state: segment EE descents, classify each by the gripper change
#      (grasp: open->closed, release: closed->open, press: no change). First press = button.
#   3. First release after the press whose resting cube (goal colour blob in the front image)
#      sits on a target slot -> that slot holds the answer target at that time.
#   4. Targets later swap places pairwise (180-degree rotation about the pair midpoint). Detect
#      swaps from frames after the release (two vacant slots + two off-slot target blobs whose
#      image midpoint matches the slot pair midpoint) and follow the answer target to its final
#      slot.
#   5. Live: localize goal cube + answer target (segment_points on colour-blob pixels),
#      pick top-down, place on the target, release, retreat.

import math

report = {"version": "v01"}

# ---------------------------------------------------------------- helpers (pure python)


def d2(ax, ay, bx, by):
    return (ax - bx) * (ax - bx) + (ay - by) * (ay - by)


def to_rows(rgb):
    try:
        return rgb.tolist()
    except Exception:
        return None


def pix(rows, rgb, y, x):
    if rows is not None:
        p = rows[y][x]
        return int(p[0]), int(p[1]), int(p[2])
    p = rgb[y][x]
    return int(p[0]), int(p[1]), int(p[2])


def is_cube_colour(r, g, b, colour):
    if colour == "red":
        return r > 100 and g < 60 and b < 60
    if colour == "green":
        return g > 100 and r < 90 and b < 90
    if colour == "blue":
        return b > 100 and r < 70 and g < 70
    return False


def components(grid, gx, gy, merge_px, min_cells):
    h = len(gy)
    w = len(gx)
    seen = [[False] * w for k in range(h)]
    blobs = []
    for i in range(h):
        for j in range(w):
            if grid[i][j] and not seen[i][j]:
                stack = [(i, j)]
                seen[i][j] = True
                sx = 0.0
                sy = 0.0
                n = 0
                while stack:
                    cell = stack.pop()
                    ci = cell[0]
                    cj = cell[1]
                    sx += gx[cj]
                    sy += gy[ci]
                    n += 1
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            ni = ci + di
                            nj = cj + dj
                            if 0 <= ni < h and 0 <= nj < w and grid[ni][nj] and not seen[ni][nj]:
                                seen[ni][nj] = True
                                stack.append((ni, nj))
                blobs.append([sx / n, sy / n, n])
    merged = True
    while merged:
        merged = False
        for a in range(len(blobs)):
            for c in range(a + 1, len(blobs)):
                if d2(blobs[a][0], blobs[a][1], blobs[c][0], blobs[c][1]) < merge_px * merge_px:
                    na = blobs[a][2]
                    nc = blobs[c][2]
                    blobs[a] = [(blobs[a][0] * na + blobs[c][0] * nc) / (na + nc),
                                (blobs[a][1] * na + blobs[c][1] * nc) / (na + nc), na + nc]
                    blobs.pop(c)
                    merged = True
                    break
            if merged:
                break
    return [b for b in blobs if b[2] >= min_cells]


def detect(rgb, y0, y1, x0, x1, stride, goal_colour):
    """Target blobs (pink/violet discs: B>G, R>G, bright R), any-cube blobs, goal-cube blobs."""
    rows = to_rows(rgb)
    gy = list(range(y0, y1, stride))
    gx = list(range(x0, x1, stride))
    grid = []
    sat = []
    goal = []
    for y in gy:
        line = []
        sline = []
        gline = []
        if rows is not None:
            row = rows[y]
        for x in gx:
            if rows is not None:
                p = row[x]
                r = p[0]
                g = p[1]
                b = p[2]
            else:
                q = pix(None, rgb, y, x)
                r = q[0]
                g = q[1]
                b = q[2]
            line.append((b - g > 12) and (r > 120) and (r - g > 20))
            red = r > 100 and g < 60 and b < 60
            green = g > 100 and r < 90 and b < 90
            blue = b > 100 and r < 70 and g < 70
            sline.append(red or green or blue)
            gline.append(is_cube_colour(r, g, b, goal_colour))
        grid.append(line)
        sat.append(sline)
        goal.append(gline)
    h = len(gy)
    w = len(gx)
    for i in range(h):
        for j in range(w):
            if sat[i][j]:
                for ni in range(max(0, i - 2), min(h, i + 3)):
                    for nj in range(max(0, j - 2), min(w, j + 3)):
                        grid[ni][nj] = False
    targets = components(grid, gx, gy, 8.0, 3)
    cubes = components(sat, gx, gy, 6.0, 3)
    goals = components(goal, gx, gy, 6.0, 2)
    return targets, cubes, goals


def frame_evidence(slots, targets, cubes, park, mid_tol):
    n = len(slots)
    seen = [False] * n
    at_slot = set()
    for s in range(n):
        for bi in range(len(targets)):
            dd = d2(targets[bi][0], targets[bi][1], slots[s][0], slots[s][1])
            if dd <= park * park:
                seen[s] = True
                at_slot.add(bi)
            elif dd <= 16.0 * 16.0 and targets[bi][2] >= 1.35 * slots[s][2]:
                seen[s] = True
        for c in cubes:
            if d2(c[0], c[1], slots[s][0], slots[s][1]) <= 13.0 * 13.0:
                seen[s] = True
    off = [targets[bi] for bi in range(len(targets)) if bi not in at_slot]
    pairs = []
    for a in range(n):
        if seen[a]:
            continue
        for b in range(a + 1, n):
            if seen[b]:
                continue
            mx = 0.5 * (slots[a][0] + slots[b][0])
            my = 0.5 * (slots[a][1] + slots[b][1])
            sep = math.sqrt(d2(slots[a][0], slots[a][1], slots[b][0], slots[b][1]))
            hit = False
            for i in range(len(off)):
                for j in range(i + 1, len(off)):
                    p = off[i]
                    q = off[j]
                    if d2(0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]), mx, my) > mid_tol * mid_tol:
                        continue
                    pq = math.sqrt(d2(p[0], p[1], q[0], q[1]))
                    if 0.45 * sep <= pq <= 1.6 * sep:
                        hit = True
            if hit:
                pairs.append((a, b))
    return seen, pairs


def group_swaps(records, min_hits):
    events = []
    open_runs = {}
    for rec in records:
        t = rec[0]
        seen = rec[1]
        pairs = rec[2]
        for key in list(open_runs.keys()):
            run = open_runs[key]
            if seen[key[0]] and seen[key[1]]:
                if run["hits"] >= min_hits:
                    events.append((run["start"], run["end"], key[0], key[1], run["hits"]))
                open_runs.pop(key)
        for key in pairs:
            if key in open_runs:
                open_runs[key]["end"] = t
                open_runs[key]["hits"] = open_runs[key]["hits"] + 1
            else:
                open_runs[key] = {"start": t, "end": t, "hits": 1}
    for key in open_runs:
        run = open_runs[key]
        if run["hits"] >= min_hits:
            events.append((run["start"], run["end"], key[0], key[1], run["hits"]))
    events.sort()
    return events


def resolve_conflicts(events):
    clusters = []
    for ev in events:
        placed = False
        for cl in clusters:
            for other in cl:
                if ev[0] <= other[1] and other[0] <= ev[1]:
                    cl.append(ev)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            clusters.append([ev])
    kept = []
    for cl in clusters:
        ranked = sorted([(-e[4], e) for e in cl])
        cand = [item[1] for item in ranked][:10]
        best = None
        for mask in range(1, 2 ** len(cand)):
            used = set()
            score = 0
            ok = True
            chosen = []
            for k in range(len(cand)):
                if (mask >> k) & 1:
                    e = cand[k]
                    if e[2] in used or e[3] in used:
                        ok = False
                        break
                    used.add(e[2])
                    used.add(e[3])
                    score += e[4]
                    chosen.append(e)
            if ok and (best is None or score > best[0]):
                best = (score, chosen)
        if best is not None:
            for e in best[1]:
                kept.append(e)
    kept.sort()
    return kept


def follow(start_slot, events, after_t):
    s = start_slot
    for ev in events:
        if ev[1] <= after_t:
            continue
        if ev[2] == s:
            s = ev[3]
        elif ev[3] == s:
            s = ev[2]
    return s


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def cloud_from_pixel(u, v):
    seg = segment_points("front", [[int(round(u)), int(round(v))]])
    if not seg.ok or len(seg.segmentations) == 0:
        return None
    cloud = mask_to_point_cloud(seg.segmentations[0], "front", target_frame=base_frame)
    pts = cloud.points
    if len(pts) < 5:
        return None
    return pts


def finished(res):
    try:
        if res.terminated or res.truncated:
            return True
    except Exception:
        return False
    return False


# ---------------------------------------------------------------- task context

ctx = get_task_context()
goal_text = str(ctx.language)
report["goal"] = goal_text
try:
    meta = robomme_plus.get_task_metadata()
    report["meta"] = {"max_steps": meta["max_steps"], "control_hz": meta["control_frequency_hz"] if "control_frequency_hz" in meta else None}
    try:
        report["meta"]["sampling"] = dict(meta["conditioning_video_sampling"])
    except Exception:
        report["meta"]["sampling"] = None
    report["meta_keys"] = sorted([str(k) for k in meta.keys()])
except Exception as exc:
    report["meta_error"] = str(exc)

goal_colour = None
low = goal_text.lower()
best_pos = None
for word in ("red", "green", "blue"):
    pos = low.find(word + " cube")
    if pos >= 0 and (best_pos is None or pos < best_pos):
        best_pos = pos
        goal_colour = word
report["goal_colour"] = goal_colour

state0 = get_robot_state()
base_frame = state0.base_frame
home_pose = state0.end_effector_poses["primary"]
down_q = [float(v) for v in home_pose.quaternion_wxyz]
report["home_ee"] = {"pos": [round(float(v), 4) for v in home_pose.position], "q": [round(v, 4) for v in down_q],
                     "gripper": float(state0.gripper_positions["primary"])}

# ---------------------------------------------------------------- conditioning video

video = robomme_plus.get_conditioning_video()
nv = len(video)
report["video_frames"] = nv

times = []
zs = []
xs = []
ys = []
gs = []
for k in range(nv):
    fr = video[k]
    times.append(float(fr.timestamp_s))
    try:
        rs = fr.robot_state
        p = rs.end_effector_poses["primary"].position
        xs.append(float(p[0]))
        ys.append(float(p[1]))
        zs.append(float(p[2]))
        gs.append(float(rs.gripper_positions["primary"]))
    except Exception:
        xs.append(None)
        ys.append(None)
        zs.append(None)
        gs.append(None)

valid_z = [z for z in zs if z is not None]
report["video_state_ok"] = len(valid_z) == nv and nv > 0
# compact series for diagnosis (every 4th frame)
report["series"] = [[round(times[k], 2), None if zs[k] is None else round(zs[k], 3),
                     None if gs[k] is None else round(gs[k], 2),
                     None if xs[k] is None else round(xs[k], 3), None if ys[k] is None else round(ys[k], 3)]
                    for k in range(0, nv, 4)]

segments = []
press_t = None
PROM = 0.04
if len(valid_z) > 0 and len(valid_z) == nv:
    zmin = min(valid_z)
    zmax = max(valid_z)
    gvalid = [g for g in gs if g is not None]
    gmin = min(gvalid)
    gmax = max(gvalid)
    gthr = max(0.15, 0.3 * (gmax - gmin))
    gmid = 0.5 * (gmax + gmin)
    report["z_range"] = [round(zmin, 4), round(zmax, 4)]
    report["g_range"] = [round(gmin, 3), round(gmax, 3)]
    # hysteresis dip detection: a dip = descent of >= PROM below the running max, then rise of >= PROM
    hi_idx = 0
    lo_idx = None
    for k in range(1, nv):
        if lo_idx is None:
            if zs[k] > zs[hi_idx]:
                hi_idx = k
            elif zs[hi_idx] - zs[k] >= PROM:
                lo_idx = k
        else:
            if zs[k] < zs[lo_idx]:
                lo_idx = k
            elif zs[k] - zs[lo_idx] >= PROM:
                # classify by the gripper inside the low band of this dip (z within 0.03 of the dip minimum)
                band = [q for q in range(hi_idx, k + 1) if zs[q] <= zs[lo_idx] + 0.03]
                before = gs[band[0]]
                after = gs[band[-1]]
                gmean = sum(gs[q] for q in band) / len(band)
                kind = "other"
                if after - before > gthr:
                    kind = "release"
                elif before - after > gthr:
                    kind = "grasp"
                elif gmean < gmin + 0.2 * (gmax - gmin):
                    kind = "press"
                elif gmean > gmid:
                    kind = "open_dip"
                segments.append({"kind": kind, "start": hi_idx, "low": lo_idx, "end": k,
                                 "t0": round(times[hi_idx], 2), "tl": round(times[lo_idx], 2),
                                 "t1": round(times[k], 2), "g_before": round(before, 3), "g_after": round(after, 3),
                                 "xyz": [round(xs[lo_idx], 3), round(ys[lo_idx], 3), round(zs[lo_idx], 3)]})
                hi_idx = k
                lo_idx = None
report["segments"] = segments
for sg in segments:
    if sg["kind"] == "press" and press_t is None:
        press_t = times[sg["low"]]
if press_t is None:
    # fallback: the dip whose xy is farthest from every grasp/release dip
    far = None
    for sg in segments:
        if sg["kind"] in ("grasp", "release"):
            continue
        dmin = None
        for other in segments:
            if other["kind"] in ("grasp", "release"):
                dd = d2(sg["xyz"][0], sg["xyz"][1], other["xyz"][0], other["xyz"][1])
                if dmin is None or dd < dmin:
                    dmin = dd
        if dmin is not None and (far is None or dmin > far[0]):
            far = (dmin, sg)
    if far is not None:
        press_t = times[far[1]["low"]]
        report["press_fallback"] = True
report["press_t"] = press_t

# ---------------------------------------------------------------- slots + answer target

Y0 = 8
Y1 = 230
STRIDE = 2
first = detect(video[0].cameras["front"].rgb, Y0, Y1, 0, 256, STRIDE, goal_colour) if nv > 0 else ([], [], [])
last = detect(video[nv - 1].cameras["front"].rgb, Y0, Y1, 0, 256, STRIDE, goal_colour) if nv > 0 else ([], [], [])
slots = [[b[0], b[1], b[2]] for b in first[0]]
for b in last[0]:
    near = False
    for s in slots:
        if d2(b[0], b[1], s[0], s[1]) < 12.0 * 12.0:
            near = True
    if not near:
        slots.append([b[0], b[1], b[2]])
report["slots"] = [[round(s[0], 1), round(s[1], 1), s[2]] for s in slots]

answer_slot = None
release_t = None
release_info = []
for sg in segments:
    if sg["kind"] != "release":
        continue
    if press_t is None or times[sg["start"]] < press_t:
        continue
    # look for the resting goal cube in the 2.5 s after release (before any later grasp)
    t_rel = times[sg["end"]]
    best = None
    q = sg["end"]
    while q < nv and times[q] <= t_rel + 2.5:
        det = detect(video[q].cameras["front"].rgb, Y0, Y1, 0, 256, STRIDE, goal_colour)
        for gb in det[2]:
            if best is None or gb[2] > best[2]:
                best = [gb[0], gb[1], gb[2], q]
        q += 3
    info = {"t": round(t_rel, 2), "cube": None if best is None else [round(best[0], 1), round(best[1], 1), best[2]]}
    if best is not None:
        bs = None
        for s in range(len(slots)):
            dd = d2(best[0], best[1], slots[s][0], slots[s][1] - 3.0)
            if bs is None or dd < bs[0]:
                bs = (dd, s)
        info["slot"] = bs[1] if bs is not None else None
        info["slot_px_dist"] = None if bs is None else round(math.sqrt(bs[0]), 1)
        if bs is not None and bs[0] <= 16.0 * 16.0:
            answer_slot = bs[1]
            release_t = t_rel
            release_info.append(info)
            break
    release_info.append(info)
report["release_info"] = release_info
report["answer_slot_at_release"] = answer_slot
report["release_t"] = release_t

events = []
final_slot = answer_slot
if answer_slot is not None:
    records = []
    for k in range(nv):
        if times[k] < release_t - 0.2:
            continue
        det = detect(video[k].cameras["front"].rgb, Y0, Y1, 0, 256, STRIDE, goal_colour)
        seen, pairs = frame_evidence(slots, det[0], det[1], 7.0, 9.0)
        records.append((times[k], seen, pairs))
    raw = group_swaps(records, 2)
    events = resolve_conflicts(raw)
    report["swaps_raw"] = [[round(e[0], 2), round(e[1], 2), e[2], e[3], e[4]] for e in raw]
    report["swaps"] = [[round(e[0], 2), round(e[1], 2), e[2], e[3], e[4]] for e in events]
    final_slot = follow(answer_slot, events, release_t)
report["final_slot"] = final_slot

# ---------------------------------------------------------------- live scene

obs = get_observation()
live = detect(obs.cameras["front"].rgb, Y0, Y1, 0, 256, STRIDE, goal_colour)
report["live_targets"] = [[round(b[0], 1), round(b[1], 1), b[2]] for b in live[0]]
report["live_goal_cubes"] = [[round(b[0], 1), round(b[1], 1), b[2]] for b in live[2]]

target_px = None
if final_slot is not None:
    sx = slots[final_slot][0]
    sy = slots[final_slot][1]
    target_px = [sx, sy]
    bestd = None
    for b in live[0]:
        dd = d2(b[0], b[1], sx, sy)
        if dd <= 14.0 * 14.0 and (bestd is None or dd < bestd):
            bestd = dd
            target_px = [b[0], b[1]]
report["target_px"] = target_px

cube_px = None
if len(live[2]) > 0:
    bigc = live[2][0]
    for b in live[2]:
        if b[2] > bigc[2]:
            bigc = b
    cube_px = [bigc[0], bigc[1]]
report["cube_px"] = cube_px

target_xyz = None
table_z = None
if target_px is not None:
    pts = cloud_from_pixel(target_px[0], target_px[1])
    if pts is not None:
        tx = quantile([p[0] for p in pts], 0.5)
        ty = quantile([p[1] for p in pts], 0.5)
        tz = quantile([p[2] for p in pts], 0.5)
        ext = [quantile([p[0] for p in pts], 0.95) - quantile([p[0] for p in pts], 0.05),
               quantile([p[1] for p in pts], 0.95) - quantile([p[1] for p in pts], 0.05)]
        report["target_cloud"] = {"n": len(pts), "center": [round(tx, 4), round(ty, 4), round(tz, 4)],
                                  "ext": [round(e, 4) for e in ext]}
        target_xyz = [tx, ty, tz]
        table_z = tz

cube_xyz = None
cube_top = None
if cube_px is not None:
    pts = cloud_from_pixel(cube_px[0], cube_px[1])
    if pts is not None:
        cx = quantile([p[0] for p in pts], 0.5)
        cy = quantile([p[1] for p in pts], 0.5)
        cz = quantile([p[2] for p in pts], 0.5)
        cube_top = quantile([p[2] for p in pts], 0.95)
        top_pts = [p for p in pts if p[2] >= cube_top - 0.008]
        if len(top_pts) >= 5:
            cx = quantile([p[0] for p in top_pts], 0.5)
            cy = quantile([p[1] for p in top_pts], 0.5)
            report["cube_top_n"] = len(top_pts)
        report["cube_cloud"] = {"n": len(pts), "center": [round(cx, 4), round(cy, 4), round(cz, 4)],
                                "top": round(cube_top, 4)}
        cube_xyz = [cx, cy, cz]
if cube_xyz is None and goal_colour is not None:
    loc = localize_object(goal_colour + " cube", camera_name="front", target_frame=base_frame)
    if loc.ok and len(loc.point_cloud.points) >= 5:
        pts = loc.point_cloud.points
        cube_xyz = [quantile([p[0] for p in pts], 0.5), quantile([p[1] for p in pts], 0.5),
                    quantile([p[2] for p in pts], 0.5)]
        cube_top = quantile([p[2] for p in pts], 0.95)
        report["cube_cloud_fallback"] = [round(v, 4) for v in cube_xyz]

# ---------------------------------------------------------------- manipulation

steps_log = []


def go(pos, label, max_steps=200, tol=0.01):
    res = move_to_pose(Pose([pos[0], pos[1], pos[2]], down_q, base_frame), tolerance=tol, max_steps=max_steps)
    entry = {"label": label, "ok": bool(res.ok)}
    try:
        entry["steps"] = int(res.steps_executed)
        entry["status"] = str(res.status)
    except Exception:
        entry["steps"] = None
    steps_log.append(entry)
    return res


done = False
if cube_xyz is not None and target_xyz is not None:
    if table_z is None:
        table_z = cube_xyz[2] - 0.02
    height = 0.04
    if cube_top is not None:
        height = max(0.02, min(0.07, cube_top - table_z))
    grasp_z = max(table_z + 0.012, table_z + 0.5 * height)
    report["plan"] = {"table_z": round(table_z, 4), "cube_h": round(height, 4), "grasp_z": round(grasp_z, 4)}
    r = open_gripper()
    if not finished(r):
        r = go([cube_xyz[0], cube_xyz[1], grasp_z + 0.10], "pre_grasp")
    if not finished(r):
        r = go([cube_xyz[0], cube_xyz[1], grasp_z], "grasp", 150, 0.008)
    if not finished(r):
        r = close_gripper()
        steps_log.append({"label": "close", "ok": bool(r.ok)})
    if not finished(r):
        r = go([cube_xyz[0], cube_xyz[1], grasp_z + 0.12], "lift")
        try:
            report["grip_after_lift"] = float(get_robot_state().gripper_positions["primary"])
        except Exception:
            report["grip_after_lift"] = None
    if not finished(r):
        r = go([target_xyz[0], target_xyz[1], grasp_z + 0.12], "above_target", 250)
    if not finished(r):
        r = go([target_xyz[0], target_xyz[1], table_z + 0.5 * height + 0.02], "place", 150, 0.008)
    if not finished(r):
        r = open_gripper()
        steps_log.append({"label": "open", "ok": bool(r.ok)})
    if not finished(r):
        r = go([target_xyz[0], target_xyz[1], table_z + height + 0.08], "retreat", 80)
    done = finished(r)
report["motion"] = steps_log
report["terminal"] = done

result = report
