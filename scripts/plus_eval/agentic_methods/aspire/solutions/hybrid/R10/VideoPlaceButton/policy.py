"""VideoPlaceButton v05 (= v04 with the target colour model learned at runtime from segment_text("front","target") boxes instead of recording-derived ring thresholds). Based on v00: track target swaps in the conditioning video, find the first
cube placement after the button press (press time from recorded robot state), follow that
target identity to the end of the video, then pick the cube and place it there."""
import math

# BEGIN_VIDLIB
def is_cube_px(color, r, g, b):
    if color == "blue":
        return b > 120 and r < 70 and g < 80
    if color == "green":
        return g > 120 and r < 90 and b < 90
    if color == "yellow":
        return r > 150 and g > 150 and b < 70
    return r > 140 and g < 70 and b < 70

def frame_scan(rows, step, color):
    # rows: nested list [y][x] -> [r,g,b]. Returns (target blobs, cube centroid/count)
    # v02: clean renders have neutral-white rings (b-g<=12); white cells join a blob only when
    # they touch a coloured ring cell (keeps robot arm / button plate out).
    height = len(rows)
    width = len(rows[0])
    colored = set()
    whites = set()
    cx = 0.0
    cy = 0.0
    cn = 0
    for y in range(0, height, step):
        row = rows[y]
        for x in range(0, width, step):
            px = row[x]
            r = px[0]
            g = px[1]
            b = px[2]
            if r > 170 and g > 150 and b > 160:
                whites.add((y // step, x // step))
            elif r > 120 and b - g > 12 and b > 90:
                colored.add((y // step, x // step))
            elif is_cube_px(color, r, g, b):
                cx += x
                cy += y
                cn += 1
    grid = {}
    for key in colored:
        grid[key] = 0
    for key in whites:
        touch = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (key[0] + dy, key[1] + dx) in colored:
                    touch = True
        if touch:
            grid[key] = 1
    seen = set()
    blobs = []
    for key in grid:
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        sx = 0.0
        sy = 0.0
        n = 0
        nw = 0
        while stack:
            cur = stack.pop()
            sy += cur[0]
            sx += cur[1]
            n += 1
            nw += grid[cur]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nb = (cur[0] + dy, cur[1] + dx)
                    if nb in grid and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        if n >= 6 and nw >= 2 and n - nw >= 3:
            blobs.append((sx / n * step, sy / n * step, n))
    cube = None
    if cn > 0:
        cube = (cx / cn, cy / cn, cn)
    return blobs, cube

# BEGIN_PAL
def color_bin(r, g, b):
    return (int(r) // 32) * 64 + (int(g) // 32) * 8 + (int(b) // 32)

def learn_target_palette(rows, boxes, min_box_px=3, min_boxes=2, min_in_frac=0.5):
    # boxes: [(x1,y1,x2,y2)] of segment_text("front","target") candidates in the live image.
    # A colour bin is a target colour when it appears (>= min_box_px px) inside at least
    # max(min_boxes, half of the boxes) boxes AND most of its image pixels lie inside boxes.
    height = len(rows)
    width = len(rows[0])
    inside = {}
    total = {}
    per_box = []
    for bx in boxes:
        per_box.append({})
    for y in range(height):
        row = rows[y]
        for x in range(width):
            px = row[x]
            k = color_bin(px[0], px[1], px[2])
            total[k] = total.get(k, 0) + 1
            hit = False
            for bi in range(len(boxes)):
                bx = boxes[bi]
                if bx[0] <= x <= bx[2] and bx[1] <= y <= bx[3]:
                    per_box[bi][k] = per_box[bi].get(k, 0) + 1
                    hit = True
            if hit:
                inside[k] = inside.get(k, 0) + 1
    need = max(min_boxes, (len(boxes) + 1) // 2)
    pal = set()
    for k in inside:
        nbox = 0
        for pb in per_box:
            if pb.get(k, 0) >= min_box_px:
                nbox += 1
        if nbox >= need and inside[k] >= min_in_frac * total[k]:
            pal.add(k)
    return pal

def frame_scan_pal(rows, step, color, pal, merge_px):
    # target cells = pixels whose colour bin is in the runtime palette; connected components
    # (8-neighbour on the stride grid) whose centroids lie within merge_px are merged
    # (concentric rings of one target), then components with >= 6 cells are blobs.
    height = len(rows)
    width = len(rows[0])
    grid = set()
    cx = 0.0
    cy = 0.0
    cn = 0
    for y in range(0, height, step):
        row = rows[y]
        for x in range(0, width, step):
            px = row[x]
            r = px[0]
            g = px[1]
            b = px[2]
            if is_cube_px(color, r, g, b):
                cx += x
                cy += y
                cn += 1
            elif color_bin(r, g, b) in pal:
                grid.add((y // step, x // step))
    seen = set()
    comps = []
    for key in grid:
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        sx = 0.0
        sy = 0.0
        n = 0
        while stack:
            cur = stack.pop()
            sy += cur[0]
            sx += cur[1]
            n += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nb = (cur[0] + dy, cur[1] + dx)
                    if nb in grid and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        comps.append([sx * step, sy * step, n])
    order = sorted([(-c[2], i) for i, c in enumerate(comps)])
    merged = []
    for item in order:
        c = comps[item[1]]
        hit = None
        for m in merged:
            if ((m[0] / m[2] - c[0] / c[2]) ** 2 + (m[1] / m[2] - c[1] / c[2]) ** 2) ** 0.5 < merge_px:
                hit = m
                break
        if hit is None:
            merged.append([c[0], c[1], c[2]])
        else:
            hit[0] += c[0]
            hit[1] += c[1]
            hit[2] += c[2]
    blobs = []
    for m in merged:
        if m[2] >= 6:
            blobs.append((m[0] / m[2], m[1] / m[2], m[2]))
    cube = None
    if cn > 0:
        cube = (cx / cn, cy / cn, cn)
    return blobs, cube
# END_PAL

def dist2(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def rest_positions(all_blobs, frac=0.3, tol=3.0, merge=7.0):
    cand = []
    for blobs in all_blobs:
        for b in blobs:
            hit = None
            for c in cand:
                if abs(c[0] - b[0]) < tol and abs(c[1] - b[1]) < tol:
                    hit = c
                    break
            if hit is None:
                cand.append([b[0], b[1], 1])
            else:
                hit[2] += 1
    order = sorted([(-c[2], i) for i, c in enumerate(cand)])
    cand = [cand[o[1]] for o in order]
    rest = []
    for c in cand:
        if c[2] < frac * len(all_blobs):
            continue
        if any(dist2(c, r) < merge for r in rest):
            continue
        rest.append((c[0], c[1], c[2]))
    return [(r[0], r[1]) for r in rest]

def swap_hits(blobs, cube, rest, vac_tol=5.0):
    occupied = [any(dist2(b, r) < vac_tol for b in blobs) for r in rest]
    off = []
    for b in blobs:
        if any(dist2(b, r) < vac_tol for r in rest):
            continue
        if cube is not None and dist2(b, cube) < 8.0:
            continue
        off.append(b)
    scored = []
    for i in range(len(rest)):
        if occupied[i]:
            continue
        for j in range(i + 1, len(rest)):
            if occupied[j]:
                continue
            A = rest[i]
            B = rest[j]
            M = ((A[0] + B[0]) / 2.0, (A[1] + B[1]) / 2.0)
            R = dist2(A, B) / 2.0
            best = None
            for a in range(len(off)):
                for c in range(a + 1, len(off)):
                    p = off[a]
                    q = off[c]
                    mid = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)
                    e = dist2(mid, M)
                    dp = dist2(p, M)
                    dq = dist2(q, M)
                    if e < 0.2 * R + 3.0 and 0.5 * R < dp < 1.5 * R and 0.5 * R < dq < 1.5 * R:
                        s = e + abs(dp - R) + abs(dq - R)
                        if best is None or s < best:
                            best = s
            if best is not None:
                scored.append((best, i, j))
    scored = sorted(scored)
    used = set()
    hits = []
    for s, i, j in scored:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        hits.append((i, j))
    return hits

def swap_events(hit_list, times, max_gap_s=1.0, min_hits=2, min_span_s=0.6):
    open_ev = {}
    events = []
    for t in range(len(times)):
        for k in hit_list[t]:
            if k in open_ev and times[t] - open_ev[k][1] <= max_gap_s:
                open_ev[k][1] = times[t]
                open_ev[k][2] += 1
            else:
                if k in open_ev:
                    events.append((open_ev[k][0], open_ev[k][1], k, open_ev[k][2]))
                open_ev[k] = [times[t], times[t], 1]
    for k in open_ev:
        events.append((open_ev[k][0], open_ev[k][1], k, open_ev[k][2]))
    good = [e for e in events if e[3] >= min_hits and e[1] - e[0] >= min_span_s]
    return sorted(good)

def cube_states(cubes, rest, times, near=10.0, min_dwell_s=1.0):
    # stable on-target segments: list of (t_start, t_end, k)
    counts = sorted(c[2] for c in cubes if c is not None)
    typical = counts[len(counts) // 2] if counts else 1
    lab = []
    for c in cubes:
        k = None
        if c is not None and 0.5 * typical <= c[2] <= 1.6 * typical:
            best = None
            for i in range(len(rest)):
                d = dist2(c, rest[i])
                if d < near and (best is None or d < best):
                    best = d
                    k = i
        lab.append(k)
    segs = []
    start = None
    for t in range(len(lab) + 1):
        cur = lab[t] if t < len(lab) else "end"
        if start is not None and (cur != lab[start]):
            if lab[start] is not None and times[t - 1] - times[start] >= min_dwell_s:
                segs.append((times[start], times[t - 1], lab[start]))
            start = None
        if start is None and t < len(lab):
            start = t
    return segs, lab

def first_place_after(segs, t_press):
    prev = None
    for s in segs:
        if s[2] != prev and s[0] > t_press:
            return s
        prev = s[2]
    return None

def follow_identity(k, t0, events):
    cur = k
    for e in events:
        if e[0] <= t0:
            continue
        i, j = e[2]
        if cur == i:
            cur = j
        elif cur == j:
            cur = i
    return cur

# ---------------------------------------------------------------- live program
def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]

def to_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        h = len(arr)
        w = len(arr[0])
        return [[[int(arr[y][x][0]), int(arr[y][x][1]), int(arr[y][x][2])] for x in range(w)] for y in range(h)]

def pos3(p):
    return (float(p[0]), float(p[1]), float(p[2]))

report = {"version": "v05"}
try:
    ctx = get_task_context()
    goal = str(ctx.language)
    report["goal"] = goal
    words = goal.lower().replace(",", " ").split()
    color = "red"
    for cw in ("red", "blue", "green", "yellow"):
        if cw in words:
            idx_c = words.index(cw)
            if idx_c + 1 < len(words) and words[idx_c + 1] == "cube":
                color = cw
                break
    report["cube_color"] = color
    try:
        meta = robomme_plus.get_task_metadata()
        report["max_steps"] = meta["max_steps"]
        try:
            report["video_sampling"] = str(meta["conditioning_video_sampling"])
        except Exception:
            pass
    except Exception as exc:
        report["meta_error"] = str(exc)

    state0 = get_robot_state()
    base_frame = state0.base_frame
    home_pose = state0.end_effector_poses["primary"]
    report["home_ee"] = [list(pos3(home_pose.position)), [float(v) for v in home_pose.quaternion_wxyz]]

    # --- live scene: button, cube, targets (robot at home, clean view)
    live = get_observation()
    live_rows = to_rows(live.cameras["front"].rgb)
    # v05: target appearance is learned at runtime from the goal noun's segmentation
    tboxes = []
    for q in ("target", "bullseye target"):
        tseg = segment_text("front", q)
        if not tseg.ok:
            continue
        for cand in tseg.segmentations[:20]:
            if cand.box_xyxy is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in cand.box_xyxy)
            area = (x2 - x1) * (y2 - y1)
            if 20.0 < area < 4000.0:
                tboxes.append((x1, y1, x2, y2))
        if len(tboxes) >= 2:
            report["palette_prompt"] = q
            break
        tboxes = []
    if len(tboxes) < 2:
        raise ValueError("fewer than 2 target boxes for the runtime palette")
    pal = learn_target_palette(live_rows, tboxes)
    half_w = sorted((b[2] - b[0]) / 2.0 for b in tboxes)
    merge_px = 0.5 * half_w[len(half_w) // 2]
    report["palette"] = [len(tboxes), len(pal), round(merge_px, 2)]
    if len(pal) == 0:
        raise ValueError("empty runtime target palette")
    live_blobs, live_cube_px = frame_scan_pal(live_rows, 2, color, pal, merge_px)
    report["live_blobs"] = [[round(b[0], 1), round(b[1], 1), b[2]] for b in live_blobs]

    button = None
    for q in ("button", "white button", "push button"):
        found = localize_object(q, camera_name="front", target_frame=base_frame)
        if found.ok and len(found.point_cloud.points) >= 10:
            button = found
            report["button_prompt"] = q
            break
    button_xyz = None
    if button is not None:
        bpts = button.point_cloud.points
        button_xyz = (quantile([p[0] for p in bpts], 0.5), quantile([p[1] for p in bpts], 0.5), quantile([p[2] for p in bpts], 0.9))
        report["button_xyz"] = [round(v, 4) for v in button_xyz]

    # --- conditioning video analysis
    video = robomme_plus.get_conditioning_video()
    report["video_len"] = len(video)
    all_blobs = []
    cubes = []
    times = []
    ee_track = []
    for frame in video:
        rows = to_rows(frame.cameras["front"].rgb)
        fb, fc = frame_scan_pal(rows, 2, color, pal, merge_px)
        all_blobs.append(fb)
        cubes.append(fc)
        times.append(float(frame.timestamp_s))
        try:
            rs = frame.robot_state
            ee_track.append((pos3(rs.end_effector_poses["primary"].position), float(rs.gripper_positions["primary"])))
        except Exception:
            ee_track.append(None)
    if len(video) > 0:
        try:
            report["video_cam_pose"] = list(pos3(video[0].cameras["front"].camera_pose.position))
            report["live_cam_pose"] = list(pos3(live.cameras["front"].camera_pose.position))
        except Exception:
            pass

    diag = []
    for t in range(0, len(times), 16):
        e = ee_track[t]
        diag.append([round(times[t], 2), len(all_blobs[t]), None if cubes[t] is None else [round(cubes[t][0]), round(cubes[t][1]), cubes[t][2]], None if e is None else [round(e[0][0], 3), round(e[0][1], 3), round(e[0][2], 3), round(e[1], 2)]])
    report["video_diag"] = diag
    if times:
        report["video_times"] = [times[0], times[-1], times[1] if len(times) > 1 else None]
    rs_times = []
    for frame in video:
        try:
            rs_times.append(float(frame.robot_state.timestamp_s))
        except Exception:
            rs_times.append(None)
    report["video_rs_times"] = rs_times[:3]
    increasing = len(times) > 1 and all(times[i + 1] > times[i] for i in range(len(times) - 1))
    if not increasing:
        # v04: no recording-derived timing constants; fall back only to the frames' own
        # robot-state timestamps, otherwise fail honestly
        rs_ok = len(rs_times) > 1 and all(v is not None for v in rs_times)
        if rs_ok:
            rs_ok = all(rs_times[i + 1] > rs_times[i] for i in range(len(rs_times) - 1))
        if not rs_ok:
            raise ValueError("conditioning video timestamps unusable")
        report["time_fallback"] = "robot_state timestamps"
        times = list(rs_times)
    print("TIMES", report.get("video_times"), rs_times[:3], report.get("time_fallback"))
    print("VIDEO_DIAG", diag)
    rest = rest_positions(all_blobs)
    report["rest_px"] = [[round(r[0], 1), round(r[1], 1)] for r in rest]
    hits = [swap_hits(all_blobs[t], cubes[t], rest) for t in range(len(all_blobs))]
    events = swap_events(hits, times)
    report["swap_events"] = [[round(e[0], 2), round(e[1], 2), list(e[2]), e[3]] for e in events]
    segs, labels = cube_states(cubes, rest, times)
    report["cube_segments"] = [[round(s[0], 2), round(s[1], 2), s[2]] for s in segs]

    # --- button press time from recorded robot state
    press_t = None
    press_frames = []
    best_d = None
    if button_xyz is not None:
        for t in range(len(ee_track)):
            if ee_track[t] is None:
                continue
            e = ee_track[t][0]
            dxy = ((e[0] - button_xyz[0]) ** 2 + (e[1] - button_xyz[1]) ** 2) ** 0.5
            dz = e[2] - button_xyz[2]
            score = dxy + max(0.0, dz)
            if best_d is None or score < best_d[0]:
                best_d = (score, t, dxy, dz)
            if dxy < 0.05 and dz < 0.03:
                press_frames.append(t)
        if press_frames:
            first = press_frames[0]
            last = first
            for t in press_frames:
                if t - last <= 3:
                    last = t
            press_t = times[last]
            report["press_window"] = [times[first], times[last]]
        elif best_d is not None:
            press_t = times[best_d[1]]
            report["press_fallback"] = [round(v, 4) for v in (best_d[0], best_d[2], best_d[3])]
    report["press_t"] = press_t
    if best_d is not None:
        report["press_best"] = [round(best_d[0], 4), times[best_d[1]], round(best_d[2], 4), round(best_d[3], 4)]

    print("PRESS", report.get("press_window"), report.get("press_best"), report.get("press_fallback"))
    print("REST", report["rest_px"], "SEGS", report["cube_segments"], "EVENTS", report["swap_events"])
    placed = None
    if press_t is not None:
        placed = first_place_after(segs, press_t)
    report["first_place_after_press"] = placed
    if placed is None:
        # fallback guess: most recent placement onto a new target
        report["placement_fallback"] = True
        prev_k = None
        for sg in segs:
            if sg[2] != prev_k:
                placed = sg
            prev_k = sg[2]
    if placed is None:
        raise ValueError("no cube placement found at all")
    final_k = follow_identity(placed[2], placed[0], events)
    goal_px = rest[final_k]
    report["final_rest_index"] = final_k
    report["goal_px"] = [round(goal_px[0], 1), round(goal_px[1], 1)]

    # --- 3D of the chosen target in the live scene
    best_live = None
    for b in live_blobs:
        d = dist2(b, goal_px)
        if best_live is None or d < best_live[0]:
            best_live = (d, b)
    seed_px = goal_px
    if best_live is not None and best_live[0] < 10.0:
        seed_px = (best_live[1][0], best_live[1][1])
    report["live_target_px"] = [round(seed_px[0], 1), round(seed_px[1], 1), None if best_live is None else round(best_live[0], 1)]

    target_cloud = None
    for q in ("target", "bullseye target", "round target"):
        seg = segment_text("front", q)
        if not seg.ok:
            continue
        pick_c = None
        for cand in seg.segmentations[:20]:
            if cand.box_xyxy is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in cand.box_xyxy)
            if x1 - 2 <= seed_px[0] <= x2 + 2 and y1 - 2 <= seed_px[1] <= y2 + 2:
                area = (x2 - x1) * (y2 - y1)
                if area < 4000 and (pick_c is None or area < pick_c[0]):
                    pick_c = (area, cand)
        if pick_c is not None:
            pc = mask_to_point_cloud(pick_c[1], "front", target_frame=base_frame)
            if len(pc.points) >= 10:
                target_cloud = pc
                report["target_prompt"] = q
                break
    if target_cloud is None:
        seg = segment_points("front", [[int(round(seed_px[0])), int(round(seed_px[1]))]])
        if seg.ok and len(seg.segmentations) > 0:
            pc = mask_to_point_cloud(seg.segmentations[0], "front", target_frame=base_frame)
            if len(pc.points) >= 5:
                target_cloud = pc
                report["target_prompt"] = "points"
    if target_cloud is None:
        raise ValueError("target 3D localization failed")
    tp = target_cloud.points
    target_xyz = (quantile([p[0] for p in tp], 0.5), quantile([p[1] for p in tp], 0.5), quantile([p[2] for p in tp], 0.9))
    report["target_xyz"] = [round(v, 4) for v in target_xyz]

    # --- pick the cube
    cube_loc = None
    for q in (color + " cube", color + " block"):
        found = localize_object(q, camera_name="front", target_frame=base_frame)
        if found.ok and len(found.point_cloud.points) >= 10:
            cube_loc = found
            report["cube_prompt"] = q
            break
    if cube_loc is None:
        raise ValueError("cube not found")
    cp = cube_loc.point_cloud.points
    cube_xy = (quantile([p[0] for p in cp], 0.5), quantile([p[1] for p in cp], 0.5))
    cube_top = quantile([p[2] for p in cp], 0.95)
    table_z = target_xyz[2]
    cube_h = max(0.015, min(0.08, cube_top - table_z))
    report["cube_xyz"] = [round(cube_xy[0], 4), round(cube_xy[1], 4), round(cube_top, 4), round(cube_h, 4)]

    # yaw of cube from top-face points (min-area bounding square search)
    top_pts = [p for p in cp if float(p[2]) > cube_top - 0.012]
    best_yaw = None
    for deg in range(0, 90, 3):
        a = math.radians(deg)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [float(p[0]) * ca + float(p[1]) * sa for p in top_pts]
        vs = [-float(p[0]) * sa + float(p[1]) * ca for p in top_pts]
        if not us:
            break
        area = (max(us) - min(us)) * (max(vs) - min(vs))
        if best_yaw is None or area < best_yaw[0]:
            best_yaw = (area, a)
    hq = [float(v) for v in home_pose.quaternion_wxyz]
    home_yaw = 2.0 * math.atan2(hq[2], hq[1])
    yaw = home_yaw
    if best_yaw is not None:
        cand_yaw = best_yaw[1]
        bestc = None
        for k in range(-4, 5):
            y = cand_yaw + k * math.pi / 2.0
            dd = abs(math.atan2(math.sin(y - home_yaw), math.cos(y - home_yaw)))
            if bestc is None or dd < bestc[0]:
                bestc = (dd, y)
        yaw = bestc[1]
    report["yaw"] = [round(home_yaw, 3), round(yaw, 3)]
    down_q = [0.0, math.cos(yaw / 2.0), math.sin(yaw / 2.0), 0.0]

    def go(xyz, tol=0.01, steps=200):
        r = move_to_pose(Pose((xyz[0], xyz[1], xyz[2]), down_q, base_frame), tolerance=tol, max_steps=steps)
        entry = [round(xyz[0], 3), round(xyz[1], 3), round(xyz[2], 3), bool(r.ok)]
        try:
            entry.append(str(r.status))
            entry.append(int(r.steps_executed))
        except Exception:
            pass
        report.setdefault("moves", []).append(entry)
        try:
            if r.terminated or r.truncated:
                report["ended_during_move"] = True
        except Exception:
            pass
        return r

    grasp_z = cube_top - 0.5 * cube_h
    open_gripper()
    go((cube_xy[0], cube_xy[1], cube_top + 0.10), 0.02)
    go((cube_xy[0], cube_xy[1], grasp_z), 0.008)
    cr = close_gripper()
    report["close"] = [bool(cr.ok), str(cr.status)]
    go((cube_xy[0], cube_xy[1], cube_top + 0.12), 0.02)
    try:
        report["grip_after_lift"] = float(get_robot_state().gripper_positions["primary"])
    except Exception:
        pass
    place_z = table_z + 0.5 * cube_h + 0.012
    go((target_xyz[0], target_xyz[1], place_z + 0.10), 0.02)
    go((target_xyz[0], target_xyz[1], place_z), 0.008)
    orr = open_gripper()
    report["open"] = [bool(orr.ok), str(orr.status)]
    go((target_xyz[0], target_xyz[1], place_z + 0.10), 0.02)
except Exception as exc:
    report["error"] = str(exc)
    print("ERROR", str(exc))
    print("REPORT", report)
result = report
