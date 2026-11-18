import math

# ---- VideoPlaceButton core (pure Python, sandbox-safe: no imports except math, no underscores) ----
STRIDE = 2


def is_target_px(r, g, b):
    # concentric ring targets: magenta/pink/purple hue (r and b both above g)
    if (b > g + 8) and (r > 120) and (r - g > 25):
        return True
    # flat-shaded simulator render: the rings' white bands are slightly pink (r > b >= g);
    # the button base and the arm are neutral or bluish grey (g >= r) and are excluded
    return r > 190 and b > 185 and r - g >= 3 and r - g < 20 and b >= g and r >= b


def cube_px(r, g, b, color):
    # saturated primary colours; the wooden table (r/g <= ~2.6) is excluded by the ratios
    if color == "red":
        return r > 90 and r > 3 * g + 20 and r > 3 * b + 20
    if color == "blue":
        return b > 90 and b > 2 * r + 30 and b > 2 * g + 30
    if color == "green":
        return g > 90 and g > 2 * r + 20 and g > 2 * b + 20
    return False


def frame_blobs(rows, color, slots=None, stride=STRIDE, min_cells=4):
    """rows: H x W x 3 nested list. Returns (target blobs [[x, y, n]], cube (x, y, n) or None,
    per-slot occupancy: True when the slot centre disc is target-coloured or covered by the cube)."""
    height = len(rows)
    width = len(rows[0])
    gh = (height + stride - 1) // stride
    gw = (width + stride - 1) // stride
    mask = [[False] * gw for row_index in range(gh)]
    cubecells = set()
    tablecells = set()
    whitecells = set()
    cx = 0.0
    cy = 0.0
    cn = 0
    for gy in range(gh):
        row = rows[gy * stride]
        mrow = mask[gy]
        for gx in range(gw):
            px = row[gx * stride]
            r = px[0]
            g = px[1]
            b = px[2]
            if is_target_px(r, g, b):
                mrow[gx] = True
            elif r > 185 and g > 185 and b > 185 and r >= g and r + 2 >= b:
                # neutral/pinkish white (ring bands in the flat render; also the button base, which
                # never sits on a slot centre) - counts only for slot occupancy, not for blobs
                whitecells.add((gy, gx))
            elif r > g + 15 and g + 10 >= b and r > 3 * g + 20 and r > 3 * b + 20 and color != "red":
                pass
            elif r > g + 15 and g + 10 >= b and not cube_px(r, g, b, color):
                tablecells.add((gy, gx))
            elif cube_px(r, g, b, color):
                cubecells.add((gy, gx))
                cx += gx
                cy += gy
                cn += 1
    seen = [[False] * gw for row_index in range(gh)]
    blobs = []
    for gy in range(gh):
        for gx in range(gw):
            if mask[gy][gx] and not seen[gy][gx]:
                stack = [(gy, gx)]
                seen[gy][gx] = True
                sx = 0.0
                sy = 0.0
                n = 0
                while stack:
                    cell = stack.pop()
                    yy = cell[0]
                    xx = cell[1]
                    sx += xx
                    sy += yy
                    n += 1
                    for dy in (-1, 0, 1):
                        ny = yy + dy
                        if ny < 0 or ny >= gh:
                            continue
                        for dx in (-1, 0, 1):
                            nx = xx + dx
                            if nx < 0 or nx >= gw:
                                continue
                            if mask[ny][nx] and not seen[ny][nx]:
                                seen[ny][nx] = True
                                stack.append((ny, nx))
                if n >= min_cells:
                    blobs.append([sx / n * stride, sy / n * stride, n])
    # merge concentric fragments of one ring (centroids within 7 px)
    merged = []
    order = sorted([(-blob[2], index) for index, blob in enumerate(blobs)])
    for entry in order:
        blob = blobs[entry[1]]
        joined = False
        for other in merged:
            if (other[0] - blob[0]) ** 2 + (other[1] - blob[1]) ** 2 < 49.0:
                total = other[2] + blob[2]
                other[0] = (other[0] * other[2] + blob[0] * blob[2]) / total
                other[1] = (other[1] * other[2] + blob[1] * blob[2]) / total
                other[2] = total
                joined = True
                break
        if not joined:
            merged.append([blob[0], blob[1], blob[2]])
    cube = None
    if cn >= 3:
        cube = (cx / cn * stride, cy / cn * stride, cn)
    occ = []
    if slots is not None:
        for slot in slots:
            sgx = int(round(slot[0] / stride))
            sgy = int(round(slot[1] / stride))
            hit = 0
            table = 0
            total = 0
            for dy in (-2, -1, 0, 1, 2):
                for dx in (-2, -1, 0, 1, 2):
                    if dx * dx + dy * dy > 5:
                        continue
                    yy = sgy + dy
                    xx = sgx + dx
                    if yy < 0 or yy >= gh or xx < 0 or xx >= gw:
                        continue
                    total += 1
                    if mask[yy][xx] or ((yy, xx) in cubecells) or ((yy, xx) in whitecells):
                        hit += 1
                    elif (yy, xx) in tablecells:
                        table += 1
            # 1 = target (or the cube) on the slot, 0 = bare table, 2 = occluded/unknown
            if total > 0 and hit >= 0.6 * total:
                occ.append(1)
            elif total > 0 and table >= 0.6 * total:
                occ.append(0)
            else:
                occ.append(2)
    return merged, cube, occ


def dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def nearest_slot(point, slots):
    best = None
    best_d = None
    for index, slot in enumerate(slots):
        d = dist2(point, slot)
        if best_d is None or d < best_d:
            best_d = d
            best = index
    return best, best_d


def track_identity(frames_blobs, frames_occ, slots, start_index, slot_k, gate=30.0):
    """Shell-game tracker over target identities, all anchored on their slots at start_index.
    Anchored identities stay while their slot centre is occupied; when it empties they follow
    the nearest unclaimed off-slot blob. Moving identities claim blobs first (nearest to the
    predicted position) and re-anchor on a free, occupied slot when they slow down near it."""
    count = len(slots)
    pos = [[s[0], s[1]] for s in slots]
    vel = [[0.0, 0.0] for s in slots]
    anchored = [True for s in slots]
    home = [i for i in range(count)]
    missing = [0 for s in slots]
    log = []
    for fi in range(start_index, len(frames_blobs)):
        blobs = frames_blobs[fi]
        occ = frames_occ[fi]
        claimed = set()
        movers = [i for i in range(count) if not anchored[i]]
        # blobs centred on a visibly occupied, anchored slot are static targets; movers skip them
        static = set()
        for i in range(count):
            if anchored[i] and occ[home[i]] == 1:
                for j in range(len(blobs)):
                    if dist2(blobs[j], slots[home[i]]) < 4.5 * 4.5:
                        static.add(j)
        pairs = []
        for i in movers:
            px = pos[i][0] + vel[i][0]
            py = pos[i][1] + vel[i][1]
            for j in range(len(blobs)):
                if j in static:
                    continue
                d = (px - blobs[j][0]) ** 2 + (py - blobs[j][1]) ** 2
                if d < gate * gate:
                    pairs.append((d, i, j))
        pairs.sort()
        done = set()
        for item in pairs:
            i = item[1]
            j = item[2]
            if i in done or j in claimed:
                continue
            done.add(i)
            claimed.add(j)
            nx = blobs[j][0]
            ny = blobs[j][1]
            vel[i] = [0.6 * (nx - pos[i][0]) + 0.4 * vel[i][0], 0.6 * (ny - pos[i][1]) + 0.4 * vel[i][1]]
            pos[i] = [nx, ny]
            missing[i] = 0
        for i in movers:
            if i not in done:
                missing[i] += 1
                if missing[i] <= 2:
                    pos[i] = [pos[i][0] + vel[i][0], pos[i][1] + vel[i][1]]
                else:
                    vel[i] = [0.0, 0.0]
        held = set(home[i] for i in range(count) if anchored[i])
        # re-anchor slow movers on a free occupied slot
        for i in movers:
            speed2 = vel[i][0] ** 2 + vel[i][1] ** 2
            best = None
            best_d = None
            for k in range(count):
                if k in held or occ[k] != 1:
                    continue
                d = dist2(pos[i], slots[k])
                if d < 100.0 and (best_d is None or d < best_d):
                    best = k
                    best_d = d
            if best is not None and speed2 < 9.0:
                anchored[i] = True
                home[i] = best
                held.add(best)
                pos[i] = [slots[best][0], slots[best][1]]
                vel[i] = [0.0, 0.0]
        for i in range(count):
            if not anchored[i] or i in movers:
                continue
            if occ[home[i]] != 0:
                continue
            slot = slots[home[i]]
            best = None
            best_d = None
            for j in range(len(blobs)):
                if j in claimed:
                    continue
                d = dist2(blobs[j], slot)
                if d < 22.0 * 22.0 and (best_d is None or d < best_d):
                    best = j
                    best_d = d
            if best is not None:
                claimed.add(best)
                anchored[i] = False
                vel[i] = [blobs[best][0] - slot[0], blobs[best][1] - slot[1]]
                pos[i] = [blobs[best][0], blobs[best][1]]
        state = home[slot_k] if anchored[slot_k] else -1
        log.append((fi, round(pos[slot_k][0], 1), round(pos[slot_k][1], 1), state, ''.join(str(home[i]) if anchored[i] else 'm' for i in range(count))))
    if anchored[slot_k]:
        final_slot = home[slot_k]
    else:
        held = set(home[i] for i in range(count) if anchored[i] and i != slot_k)
        final_slot = None
        best_d = None
        for k in range(count):
            if k in held:
                continue
            d = dist2(pos[slot_k], slots[k])
            if best_d is None or d < best_d:
                best_d = d
                final_slot = k
    return final_slot, pos[slot_k], log


def first_placement_after(cubes, slots, press_end, leave_px=12.0, near_px=8.0, stable=4, still_px=1.6):
    """cubes: per-frame cube centroid or None. Return (frame index, slot index) of the first
    stationary rest of the cube near a slot after press_end, once the cube has left the spot it
    occupied at press time (a cube already resting on a target during the press does not count)."""
    ref = None
    for fi in range(press_end, -1, -1):
        if cubes[fi] is not None:
            ref = cubes[fi]
            break
    left = ref is None
    run_slot = None
    run_len = 0
    run_start = None
    prev = None
    for fi in range(press_end + 1, len(cubes)):
        c = cubes[fi]
        if c is None:
            run_slot = None
            run_len = 0
            prev = None
            continue
        if not left:
            if dist2(c, ref) > leave_px * leave_px:
                left = True
            else:
                prev = c
                continue
        k, d = nearest_slot(c, slots)
        moving = prev is None or dist2(c, prev) > still_px * still_px
        prev = c
        if d < near_px * near_px and not moving:
            if run_slot == k:
                run_len += 1
            else:
                run_slot = k
                run_len = 1
                run_start = fi
            if run_len >= stable:
                return run_start, k
        else:
            run_slot = None
            run_len = 0
    return None, None

# ---- VideoPlaceButton v00 main ----
report = {"version": "v02", "stage": "start", "notes": []}


def hold(ticks):
    """Hold the measured joints/gripper for a number of control ticks; returns ticks executed."""
    done = 0
    for tick_index in range(ticks):
        st = get_robot_state()
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position",
            target=[float(v) for v in st.joint_positions["primary"]],
            gripper_position=float(st.gripper_positions["primary"]),
            embodiment=st.embodiment)}))
        done += 1
        try:
            if reply.terminated or reply.truncated:
                break
        except Exception:
            pass
    return done


def to_rows(rgb):
    try:
        return rgb.tolist()
    except Exception:
        rows = []
        height = len(rgb)
        for y in range(height):
            src = rgb[y]
            rows.append([[int(src[x][0]), int(src[x][1]), int(src[x][2])] for x in range(len(src))])
        return rows


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def cloud_of_pixel(camera, px, py, prompt=None):
    """3D points of the object under pixel (px, py): text candidates whose box contains the
    pixel first (if a prompt is given), else a point-prompt segmentation."""
    base = get_robot_state().base_frame
    chosen = None
    how = None
    if prompt is not None:
        found = segment_text(camera, prompt)
        if found.ok:
            best_area = None
            for cand in found.segmentations[:10]:
                if cand.box_xyxy is None:
                    continue
                x1, y1, x2, y2 = (float(v) for v in cand.box_xyxy)
                if x1 - 2 <= px <= x2 + 2 and y1 - 2 <= py <= y2 + 2:
                    area = (x2 - x1) * (y2 - y1)
                    if best_area is None or area < best_area:
                        best_area = area
                        chosen = cand
                        how = "text:" + prompt
    if chosen is None:
        found = segment_points(camera, [[float(px), float(py)]])
        if found.ok and len(found.segmentations) > 0:
            chosen = found.segmentations[0]
            how = "points"
    if chosen is None:
        return None, None
    cloud = mask_to_point_cloud(chosen, camera, target_frame=base)
    pts = [[float(p[0]), float(p[1]), float(p[2])] for p in cloud.points]
    return pts, how


def quat_mul(a, b):
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]


def rotate_vec(q, v):
    qv = [0.0, v[0], v[1], v[2]]
    qc = [q[0], -q[1], -q[2], -q[3]]
    r = quat_mul(quat_mul(q, qv), qc)
    return [r[1], r[2], r[3]]


def square_yaw(points_xy):
    """Yaw (rad, in [0, pi/2)) of the minimum-area bounding square of xy points."""
    best = None
    best_area = None
    for deg in range(0, 90, 3):
        a = math.radians(deg)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [p[0] * ca + p[1] * sa for p in points_xy]
        vs = [-p[0] * sa + p[1] * ca for p in points_xy]
        area = (max(us) - min(us)) * (max(vs) - min(vs))
        if best_area is None or area < best_area:
            best_area = area
            best = a
    return best


def deproject(cam, px, py, convention):
    """Pixel + depth -> robot_base point for one camera-axis convention (calibrated at runtime)."""
    depth = cam.depth_m
    values = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy = int(round(py)) + dy
            xx = int(round(px)) + dx
            if 0 <= yy < len(depth) and 0 <= xx < len(depth[0]):
                d = float(depth[yy][xx])
                if d > 0.05 and d < 5.0:
                    values.append(d)
    if len(values) == 0:
        return None
    d = sorted(values)[len(values) // 2]
    k = cam.intrinsics
    fx = float(k[0][0])
    fy = float(k[1][1])
    cx = float(k[0][2])
    cy = float(k[1][2])
    a = (px - cx) / fx
    b = (py - cy) / fy
    if convention == "opencv":
        v = [a * d, b * d, d]
    elif convention == "ros":
        v = [d, -a * d, -b * d]
    else:
        v = [a * d, -b * d, -d]
    pose = cam.camera_pose
    q = [float(x) for x in pose.quaternion_wxyz]
    r = rotate_vec(q, v)
    t = [float(x) for x in pose.position]
    return [r[0] + t[0], r[1] + t[1], r[2] + t[2]]


goal = get_task_context().language
report["goal"] = goal
goal_l = goal.lower()
color = None
for word in ("red", "blue", "green"):
    if color is None and (word + " cube") in goal_l:
        color = word
if color is None:
    for word in ("red", "blue", "green"):
        if color is None and word in goal_l:
            color = word
report["color"] = color
try:
    meta = robomme_plus.get_task_metadata()
    report["max_steps"] = meta["max_steps"]
    try:
        report["sampling"] = str(meta["conditioning_video_sampling"])
    except Exception:
        pass
except Exception as exc:
    report["notes"].append("metadata: " + str(exc))

obs0 = get_observation()
state0 = obs0.robot_state
base = state0.base_frame
live_rows = to_rows(obs0.cameras["front"].rgb)
live_blobs, live_cube, live_occ = frame_blobs(live_rows, color)
slots = [b for b in live_blobs if b[2] >= 15]
report["slots_px"] = [[round(s[0], 1), round(s[1], 1), s[2]] for s in slots]
report["live_cube_px"] = None if live_cube is None else [round(live_cube[0], 1), round(live_cube[1], 1), live_cube[2]]
report["home_ee"] = [round(float(v), 4) for v in state0.end_effector_poses["primary"].position]
report["home_quat"] = [round(float(v), 4) for v in state0.end_effector_poses["primary"].quaternion_wxyz]
report["home_grip"] = float(state0.gripper_positions["primary"])

# ---- conditioning video: per-frame target blobs, cube centroid, slot occupancy, robot state ----
report["stage"] = "video"
video = robomme_plus.get_conditioning_video()
nvid = len(video)
report["video_frames"] = nvid
frames_blobs = []
frames_occ = []
cubes = []
ee = []
grip = []
times = []
hold_ticks = 0
for vi in range(nvid):
    frame = video[vi]
    rows = to_rows(frame.cameras["front"].rgb)
    blobs, cube, occ = frame_blobs(rows, color, slots)
    frames_blobs.append(blobs)
    frames_occ.append(occ)
    cubes.append(cube)
    try:
        rs = frame.robot_state
        p = rs.end_effector_poses["primary"].position
        ee.append([float(p[0]), float(p[1]), float(p[2])])
        grip.append(float(rs.gripper_positions["primary"]))
    except Exception:
        ee.append(None)
        grip.append(None)
    try:
        times.append(float(frame.timestamp_s))
    except Exception:
        times.append(None)
    if vi % 48 == 47:
        hold_ticks += hold(1)
report["hold_ticks"] = hold_ticks
report["video_t_end"] = times[-1] if nvid > 0 else None

# compact diagnostics of the robot-state stream (every 8th frame)
diag = []
for vi in range(0, nvid, 8):
    e = ee[vi]
    c = cubes[vi]
    diag.append([vi, None if times[vi] is None else round(times[vi], 2),
                 None if e is None else [round(e[0], 3), round(e[1], 3), round(e[2], 3)],
                 None if grip[vi] is None else round(grip[vi], 2),
                 None if c is None else [round(c[0]), round(c[1])],
                 ''.join(str(x) for x in frames_occ[vi])])
report["video_diag"] = diag
report["video_blobs"] = [[[round(b[0]), round(b[1]), b[2]] for b in fb] for fb in frames_blobs]
report["video_cubes"] = [None if c is None else [round(c[0], 1), round(c[1], 1), c[2]] for c in cubes]
report["video_occ"] = [''.join(str(x) for x in o) for o in frames_occ]
report["video_ee_z"] = [None if e is None else round(e[2], 3) for e in ee]
report["video_grip"] = [None if g is None else round(g, 2) for g in grip]
result = report

motions = []


def go(x, y, z, quat, label, steps=200, tol=0.01):
    res = move_to_pose(Pose([x, y, z], quat, base), tolerance=tol, max_steps=steps)
    entry = [label, bool(res.ok)]
    try:
        entry.append(str(res.status))
        entry.append(int(res.steps_executed))
    except Exception:
        pass
    motions.append(entry)
    report["motions"] = motions
    try:
        if res.terminated or res.truncated:
            report["terminated_at"] = label
            return res, True
    except Exception:
        pass
    return res, False


try:
    # ---- press detection: lowest closed-empty-gripper dip of the end effector ----
    report["stage"] = "press"
    press_index = None
    valid = [i for i in range(nvid) if ee[i] is not None and grip[i] is not None]
    if len(valid) > 0:
        zmin_all = min(ee[i][2] for i in valid)
        segments = []
        current = []
        for i in range(nvid):
            if ee[i] is not None and grip[i] is not None and grip[i] < 0.2:
                current.append(i)
            else:
                if len(current) > 0:
                    segments.append(current)
                current = []
        if len(current) > 0:
            segments.append(current)
        seg_info = []
        best_seg = None
        best_z = None
        for seg in segments:
            low = min(ee[i][2] for i in seg)
            seg_info.append([seg[0], seg[-1], round(low, 3)])
            if best_z is None or low < best_z:
                best_z = low
                best_seg = seg
        report["closed_segments"] = seg_info
        report["zmin_all"] = round(zmin_all, 3)
        if best_seg is not None:
            bottom = None
            for i in best_seg:
                if bottom is None or ee[i][2] < ee[bottom][2]:
                    bottom = i
            press_index = bottom
            last_low = bottom
            for i in best_seg:
                if i > bottom and ee[i][2] < ee[bottom][2] + 0.015:
                    last_low = i
            press_index = last_low
            report["press_bottom_index"] = bottom
            report["press_ee"] = [round(v, 3) for v in ee[bottom]]
    report["press_index"] = press_index
    if press_index is None:
        report["notes"].append("no press found; using video midpoint")
        press_index = nvid // 2

    # ---- first placement after the press, then follow that target through the swaps ----
    report["stage"] = "placement"
    place_index, place_slot = first_placement_after(cubes, slots, press_index)
    report["place_index"] = place_index
    report["place_slot"] = place_slot
    final_slot = None
    if place_index is not None:
        final_slot, final_pos, track_log = track_identity(frames_blobs, frames_occ, slots, place_index, place_slot)
        changes = []
        for li in range(len(track_log)):
            entry = track_log[li]
            if li == 0 or entry[3] != track_log[li - 1][3]:
                changes.append([entry[0], entry[3], entry[4]])
        report["identity_path"] = changes[:60]
    report["final_slot"] = final_slot
    if final_slot is None:
        raise ValueError("could not determine the target slot")
    target_px = slots[final_slot]
    report["target_px"] = [round(target_px[0], 1), round(target_px[1], 1)]

    # ---- 3D: cube and target from the live front camera ----
    report["stage"] = "localize"
    if live_cube is None:
        raise ValueError("goal cube not visible in live front image")
    cube_pts, cube_how = cloud_of_pixel("front", live_cube[0], live_cube[1], color + " cube")
    report["cube_how"] = cube_how
    if cube_pts is None or len(cube_pts) < 10:
        raise ValueError("cube point cloud empty")
    cube_x = quantile([p[0] for p in cube_pts], 0.5)
    cube_y = quantile([p[1] for p in cube_pts], 0.5)
    cube_top = quantile([p[2] for p in cube_pts], 0.95)
    cube_mid_z = quantile([p[2] for p in cube_pts], 0.5)
    # calibrate the camera-axis convention on the cube: deprojected cube pixel vs its segmented cloud
    front_cam = obs0.cameras["front"]
    convention = None
    conv_err = {}
    for name in ("opencv", "ros", "opengl"):
        try:
            guess = deproject(front_cam, live_cube[0], live_cube[1], name)
        except Exception as exc:
            guess = None
            report["notes"].append("deproject " + name + ": " + str(exc))
        if guess is None:
            continue
        err = math.sqrt((guess[0] - cube_x) ** 2 + (guess[1] - cube_y) ** 2 + (guess[2] - cube_mid_z) ** 2)
        conv_err[name] = round(err, 4)
        if err < 0.04 and (convention is None or err < conv_err[convention]):
            convention = name
    report["convention_err"] = conv_err
    report["convention"] = convention
    tgt_dep = None
    if convention is not None:
        tgt_dep = deproject(front_cam, target_px[0], target_px[1], convention)
    report["target_deproject"] = None if tgt_dep is None else [round(v, 4) for v in tgt_dep]
    tgt_seg = None
    try:
        tgt_pts, tgt_how = cloud_of_pixel("front", target_px[0], target_px[1], None)
        report["target_how"] = tgt_how
        if tgt_pts is not None and len(tgt_pts) >= 10:
            xs = [p[0] for p in tgt_pts]
            ys = [p[1] for p in tgt_pts]
            spread = max(quantile(xs, 0.95) - quantile(xs, 0.05), quantile(ys, 0.95) - quantile(ys, 0.05))
            report["target_seg_spread"] = round(spread, 4)
            tgt_seg = [quantile(xs, 0.5), quantile(ys, 0.5), quantile([p[2] for p in tgt_pts], 0.5)]
            report["target_seg"] = [round(v, 4) for v in tgt_seg]
            report["target_points"] = len(tgt_pts)
            if spread > 0.12:
                tgt_seg = None
    except Exception as exc:
        report["notes"].append("target seg: " + str(exc))
    if tgt_dep is not None:
        tgt_x, tgt_y, tgt_z = tgt_dep[0], tgt_dep[1], tgt_dep[2]
        report["target_source"] = "deproject"
    elif tgt_seg is not None:
        tgt_x, tgt_y, tgt_z = tgt_seg[0], tgt_seg[1], tgt_seg[2]
        report["target_source"] = "segment_points"
    else:
        raise ValueError("no 3D estimate for the target slot")
    table_z = tgt_z
    cube_h = cube_top - table_z
    report["cube_xyz"] = [round(cube_x, 4), round(cube_y, 4), round(cube_top, 4)]
    report["cube_points"] = len(cube_pts)
    report["target_xyz"] = [round(tgt_x, 4), round(tgt_y, 4), round(tgt_z, 4)]
    report["target_points"] = len(tgt_pts)
    report["cube_h"] = round(cube_h, 4)
    if cube_h < 0.01 or cube_h > 0.08:
        report["notes"].append("implausible cube height; using 0.04")
        cube_h = 0.04
    # cube xy centre from its top face (points within 1 cm of the top)
    top_pts = [p for p in cube_pts if p[2] > cube_top - 0.01]
    if len(top_pts) >= 5:
        cube_x = quantile([p[0] for p in top_pts], 0.5)
        cube_y = quantile([p[1] for p in top_pts], 0.5)
        yaw_c = square_yaw([[p[0] - cube_x, p[1] - cube_y] for p in top_pts])
    else:
        yaw_c = 0.0
    report["cube_top_center"] = [round(cube_x, 4), round(cube_y, 4)]
    report["cube_yaw_deg"] = round(math.degrees(yaw_c), 1)

    q0 = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]
    finger_axis = rotate_vec(q0, [0.0, 1.0, 0.0])
    phi0 = math.atan2(finger_axis[1], finger_axis[0])
    delta = yaw_c - phi0
    while delta > math.pi / 4:
        delta -= math.pi / 2
    while delta < -math.pi / 4:
        delta += math.pi / 2
    qz = [math.cos(delta / 2), 0.0, 0.0, math.sin(delta / 2)]
    q_grasp = quat_mul(qz, q0)
    report["grasp_delta_deg"] = round(math.degrees(delta), 1)

    # ---- pick ----
    report["stage"] = "pick"
    motions = []




    result = report
    try:
        grasp_z = table_z + 0.5 * cube_h
        open_gripper()
        res, ended = go(cube_x, cube_y, grasp_z + 0.10, q_grasp, "pre_grasp", 250, 0.015)
        if not ended:
            res, ended = go(cube_x, cube_y, grasp_z, q_grasp, "grasp", 150, 0.008)
        if not ended:
            close_gripper()
            report["grip_after_close"] = float(get_robot_state().gripper_positions["primary"])
            res, ended = go(cube_x, cube_y, grasp_z + 0.10, q_grasp, "lift", 150, 0.015)
        if not ended:
            report["grip_after_lift"] = float(get_robot_state().gripper_positions["primary"])
            report["stage"] = "place"
            res, ended = go(tgt_x, tgt_y, grasp_z + 0.10, q_grasp, "above_target", 250, 0.012)
        if not ended:
            res, ended = go(tgt_x, tgt_y, table_z + 0.5 * cube_h + 0.012, q_grasp, "lower", 150, 0.008)
        if not ended:
            open_gripper()
            res, ended = go(tgt_x, tgt_y, table_z + 0.5 * cube_h + 0.08, q_grasp, "retreat", 120, 0.02)
        if not ended:
            hold(10)
    except Exception as exc:
        report["motion_error"] = str(exc)
    report["stage"] = "done"
    try:
        report["final_ee"] = [round(float(v), 4) for v in get_robot_state().end_effector_poses["primary"].position]
    except Exception:
        pass
    result = report
except Exception as exc:
    report["error"] = str(exc)
result = report
