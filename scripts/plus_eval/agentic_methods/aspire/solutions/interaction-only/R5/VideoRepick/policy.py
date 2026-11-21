# v02 -- VideoRepick (RoboMME-Plus, interaction-only). v01 + robust per-object area (merged blobs split).
#
# Goal (ep0): "watch the video carefully, identify the object that was picked up, then pick up
# and place down the same object twice, finally press the button to stop".
#
# Strategy (all derived at runtime from this episode's conditioning video + live views):
#  1. Video robot state: the first gripper close that holds something (reading stays > 0.05)
#     is the pick; the following open is the put-down. The put-down point (EE xy at release,
#     EE z at grasp) projected into the video front camera marks the target object.
#  2. Background = per-pixel median of frames sampled after the robot left; the target's
#     foreground blob at the init frame gives its reference chromaticity.
#  3. All look-alike objects are tracked through the shuffle with pixel k-means (one centre per
#     object, constant-velocity prediction, coasting when a track loses its pixels).
#  4. Live: assign live object pixels to the final track centres; the target's pixels give a
#     point prompt for segment_points -> point cloud -> top-down grasp (yaw chosen to keep the
#     fingers away from neighbouring objects).
#  5. Pick, lift, put back down, twice; then press the button (segment_text "button").
import math

report = {"version": "v05"}
context = get_task_context()
goal = str(context.language)
report["goal"] = goal

try:
    meta = robomme_plus.get_task_metadata()
    report["max_steps"] = int(meta["max_steps"])
except Exception as exc:
    report["meta_error"] = str(exc)

state0 = get_robot_state()
base_frame = state0.base_frame
downward = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]
report["home_ee"] = [round(float(v), 4) for v in state0.end_effector_poses["primary"].position]

STRIDE = 2


# ---------------------------------------------------------------- generic helpers
def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 0:
        return 0.0
    return ordered[int((len(ordered) - 1) * fraction)]


def vec(p):
    return [float(p[0]), float(p[1]), float(p[2])]


def dist2(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def quat_to_matrix(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def cam_params(cam):
    rot = quat_to_matrix(cam.camera_pose.quaternion_wxyz)
    t = vec(cam.camera_pose.position)
    k = cam.intrinsics
    return {"R": rot, "t": t, "fx": float(k[0][0]), "fy": float(k[1][1]),
            "cx": float(k[0][2]), "cy": float(k[1][2])}


def project(point, cp):
    """robot_base point -> pixel (OpenCV convention: +z forward, +y down)."""
    rot = cp["R"]
    t = cp["t"]
    d = [point[0] - t[0], point[1] - t[1], point[2] - t[2]]
    xc = rot[0][0] * d[0] + rot[1][0] * d[1] + rot[2][0] * d[2]
    yc = rot[0][1] * d[0] + rot[1][1] * d[1] + rot[2][1] * d[2]
    zc = rot[0][2] * d[0] + rot[1][2] * d[1] + rot[2][2] * d[2]
    if zc <= 1e-6:
        return None
    return [cp["fx"] * xc / zc + cp["cx"], cp["fy"] * yc / zc + cp["cy"]]


def backproject(u, v, depth, cp):
    xc = (u - cp["cx"]) / cp["fx"] * depth
    yc = (v - cp["cy"]) / cp["fy"] * depth
    zc = depth
    rot = cp["R"]
    t = cp["t"]
    return [rot[0][0] * xc + rot[0][1] * yc + rot[0][2] * zc + t[0],
            rot[1][0] * xc + rot[1][1] * yc + rot[1][2] * zc + t[1],
            rot[2][0] * xc + rot[2][1] * yc + rot[2][2] * zc + t[2]]


def to_rows(arr):
    """RGB array -> nested lists. Fallback (if tolist is unavailable) fills only the even
    rows/columns the stride-2 grid reads and duplicates them into their odd neighbours."""
    try:
        return arr.tolist()
    except Exception:
        rows = []
        h = len(arr)
        w = len(arr[0])
        prev = None
        for y in range(h):
            if y % 2 == 1 and prev is not None:
                rows.append(prev)
                continue
            row = arr[y]
            out = []
            for x in range(0, w, 2):
                px = row[x]
                val = [int(px[0]), int(px[1]), int(px[2])]
                out.append(val)
                out.append(val)
            out = out[:w]
            rows.append(out)
            prev = out
        return rows


def depth_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        rows = []
        for row in arr:
            out = []
            for val in row:
                out.append(float(val))
            rows.append(out)
        return rows


def chroma(px):
    s = float(px[0]) + float(px[1]) + float(px[2]) + 1.0
    return [px[0] / s, px[1] / s, px[2] / s]


def summarize(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return {"cx": 0.5 * (quantile(xs, 0.1) + quantile(xs, 0.9)),
            "cy": 0.5 * (quantile(ys, 0.1) + quantile(ys, 0.9)),
            "ztop": quantile(zs, 0.9), "zlow": quantile(zs, 0.1),
            "span": max(quantile(xs, 0.9) - quantile(xs, 0.1), quantile(ys, 0.9) - quantile(ys, 0.1)),
            "n": len(points)}


def is_done(res):
    try:
        if bool(res.terminated):
            return True
    except Exception:
        pass
    try:
        if bool(res.truncated):
            return True
    except Exception:
        pass
    return False


def res_summary(res):
    out = {}
    try:
        out["ok"] = bool(res.ok)
    except Exception:
        out["ok"] = None
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


def components(cells):
    """8-connected components over a set of (gx, gy) grid cells."""
    remaining = set(cells)
    comps = []
    while len(remaining) > 0:
        seed = remaining.pop()
        stack = [seed]
        comp = [seed]
        while len(stack) > 0:
            cx, cy = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cx + dx, cy + dy)
                    if nb in remaining:
                        remaining.remove(nb)
                        stack.append(nb)
                        comp.append(nb)
        comps.append(comp)
    return comps


def centroid(cells):
    sx = 0.0
    sy = 0.0
    for c in cells:
        sx += c[0]
        sy += c[1]
    return [sx / len(cells), sy / len(cells)]


def kmeans_split(cells, k):
    """Split one blob into k centres (farthest-point seeding, 6 Lloyd iterations)."""
    centres = [list(centroid(cells))]
    while len(centres) < k:
        best = None
        for c in cells:
            dmin = None
            for m in centres:
                d = (c[0] - m[0]) ** 2 + (c[1] - m[1]) ** 2
                if dmin is None or d < dmin:
                    dmin = d
            if best is None or dmin > best[0]:
                best = [dmin, c]
        centres.append([float(best[1][0]), float(best[1][1])])
    for it in range(6):
        sums = [[0.0, 0.0, 0] for m in centres]
        for c in cells:
            bi = 0
            bd = None
            for i in range(len(centres)):
                d = (c[0] - centres[i][0]) ** 2 + (c[1] - centres[i][1]) ** 2
                if bd is None or d < bd:
                    bd = d
                    bi = i
            sums[bi][0] += c[0]
            sums[bi][1] += c[1]
            sums[bi][2] += 1
        for i in range(len(centres)):
            if sums[i][2] > 0:
                centres[i] = [sums[i][0] / sums[i][2], sums[i][1] / sums[i][2]]
    return centres


def object_cells(rows, ref, thr, bright_min, box):
    """Grid cells whose chromaticity is within thr (L1) of the reference."""
    x0, y0, x1, y1 = box
    cells = []
    rr, rg, rb = ref[0], ref[1], ref[2]
    for gy in range(y0, y1 + 1):
        row = rows[gy * STRIDE]
        for gx in range(x0, x1 + 1):
            px = row[gx * STRIDE]
            s = px[0] + px[1] + px[2] + 1.0
            if s < bright_min:
                continue
            d = abs(px[0] / s - rr) + abs(px[1] / s - rg) + abs(px[2] / s - rb)
            if d < thr:
                cells.append((gx, gy))
    return cells


def assign_track(cells, centres, rmax, iters):
    """Pixel k-means against existing centres; returns (new centres, counts, members)."""
    cur = [list(c) for c in centres]
    members = [[] for c in cur]
    r2 = rmax * rmax
    for it in range(iters):
        members = [[] for c in cur]
        for c in cells:
            bi = -1
            bd = r2
            for i in range(len(cur)):
                d = (c[0] - cur[i][0]) ** 2 + (c[1] - cur[i][1]) ** 2
                if d < bd:
                    bd = d
                    bi = i
            if bi >= 0:
                members[bi].append(c)
        for i in range(len(cur)):
            if len(members[i]) > 0:
                cur[i] = centroid(members[i])
    counts = [len(m) for m in members]
    return cur, counts, members


def run_all():
    # ---------------------------------------------------------------- conditioning video
    video = robomme_plus.get_conditioning_video()
    nframes = len(video)
    vinfo = {"frames": nframes}
    trace = []
    for i in range(nframes):
        rs = video[i].robot_state
        p = vec(rs.end_effector_poses["primary"].position)
        trace.append([float(video[i].timestamp_s), float(rs.gripper_positions["primary"]), p])

    # grasp / release events (hysteresis on the normalized gripper reading)
    events = []
    closed = trace[0][1] < 0.75 if nframes > 0 else False
    for i in range(nframes):
        g = trace[i][1]
        if (not closed) and g < 0.75:
            closed = True
            events.append(["close", i])
        elif closed and g > 0.85:
            closed = False
            events.append(["open", i])
    grasp_frame = None
    release_frame = None
    for ei in range(len(events)):
        kind, fi = events[ei]
        if kind == "close" and grasp_frame is None:
            later = trace[min(nframes - 1, fi + 8)][1]
            if later > 0.05:
                grasp_frame = fi
                for ej in range(ei + 1, len(events)):
                    if events[ej][0] == "open":
                        release_frame = events[ej][1]
                        break
    vinfo["events"] = events
    vinfo["grasp_frame"] = grasp_frame
    vinfo["release_frame"] = release_frame
    if grasp_frame is None:
        raise ValueError("no holding grasp found in the conditioning video robot state")

    lo = max(0, grasp_frame - 10)
    grasp_z = min(trace[j][2][2] for j in range(lo, grasp_frame + 1))
    if release_frame is not None:
        rel = trace[release_frame - 1][2]
        target_world = [rel[0], rel[1], grasp_z]
        # init frame: robot has risen well clear of the put-down point, or has stopped moving
        init_frame = None
        for j in range(release_frame, nframes):
            ee = trace[j][2]
            if ee[2] - rel[2] > 0.08:
                init_frame = j
                break
            if j + 5 < nframes:
                moved = 0.0
                for k in range(j, j + 5):
                    moved += dist2(trace[k][2], trace[k + 1][2]) + abs(trace[k][2][2] - trace[k + 1][2][2])
                if moved < 0.002:
                    init_frame = j
                    break
        if init_frame is None:
            init_frame = min(nframes - 1, release_frame + 20)
    else:
        # video ends while holding: the object stays with the gripper
        target_world = trace[nframes - 1][2]
        init_frame = nframes - 1
    vinfo["target_world_putdown"] = [round(v, 4) for v in target_world]
    vinfo["init_frame"] = init_frame

    cam_v = cam_params(video[init_frame].cameras["front"])
    H = len(video[0].cameras["front"].rgb)
    W = len(video[0].cameras["front"].rgb[0])
    GH = (H - 1) // STRIDE
    GW = (W - 1) // STRIDE
    target_px = project(target_world, cam_v)
    vinfo["target_px_init"] = target_px

    # background: per-cell median over frames sampled from init..end
    samples = []
    nsamp = 9
    for k in range(nsamp):
        j = init_frame + int(round(k * (nframes - 1 - init_frame) / (nsamp - 1)))
        samples.append(to_rows(video[j].cameras["front"].rgb))
    background = []
    for gy in range(GH + 1):
        brow = []
        for gx in range(GW + 1):
            cs = [[], [], []]
            for srows in samples:
                px = srows[gy * STRIDE][gx * STRIDE]
                cs[0].append(px[0])
                cs[1].append(px[1])
                cs[2].append(px[2])
            brow.append([sorted(cs[0])[nsamp // 2], sorted(cs[1])[nsamp // 2], sorted(cs[2])[nsamp // 2]])
        background.append(brow)
    samples = None

    init_rows = to_rows(video[init_frame].cameras["front"].rgb)
    fg = []
    for gy in range(GH + 1):
        row = init_rows[gy * STRIDE]
        for gx in range(GW + 1):
            px = row[gx * STRIDE]
            b = background[gy][gx]
            if abs(px[0] - b[0]) + abs(px[1] - b[1]) + abs(px[2] - b[2]) > 60:
                fg.append((gx, gy))
    fg_comps = components(fg)
    tg = [target_px[0] / STRIDE, target_px[1] / STRIDE]
    best = None
    for comp in fg_comps:
        if len(comp) < 4:
            continue
        for c in comp:
            d = (c[0] - tg[0]) ** 2 + (c[1] - tg[1]) ** 2
            if best is None or d < best[0]:
                best = [d, comp]
    if best is None:
        raise ValueError("no foreground blob near the video put-down point")
    vinfo["target_fg_dist_cells"] = round(math.sqrt(best[0]), 2)
    tcomp = best[1]
    chs = [[], [], []]
    bright = []
    for c in tcomp:
        px = init_rows[c[1] * STRIDE][c[0] * STRIDE]
        ch = chroma(px)
        chs[0].append(ch[0])
        chs[1].append(ch[1])
        chs[2].append(ch[2])
        bright.append(px[0] + px[1] + px[2])
    ref = [quantile(chs[0], 0.5), quantile(chs[1], 0.5), quantile(chs[2], 0.5)]
    bgc = [[], [], []]
    for gy in range(0, GH + 1, 4):
        for gx in range(0, GW + 1, 4):
            ch = chroma(background[gy][gx])
            bgc[0].append(ch[0])
            bgc[1].append(ch[1])
            bgc[2].append(ch[2])
    bg_ref = [quantile(bgc[0], 0.5), quantile(bgc[1], 0.5), quantile(bgc[2], 0.5)]
    sep = abs(ref[0] - bg_ref[0]) + abs(ref[1] - bg_ref[1]) + abs(ref[2] - bg_ref[2])
    thr = max(0.05, min(0.25, 0.45 * sep))
    bright_min = max(30.0, 0.4 * quantile(bright, 0.1))
    vinfo["ref_chroma"] = [round(v, 3) for v in ref]
    vinfo["bg_chroma"] = [round(v, 3) for v in bg_ref]
    vinfo["chroma_thr"] = round(thr, 3)
    vinfo["bright_min"] = round(bright_min, 1)

    full_box = [0, 0, GW, GH]
    cells0 = object_cells(init_rows, ref, thr, bright_min, full_box)
    # Guard: objects cover a small part of the image; if the colour mask is too large the
    # threshold admitted background, so tighten it until the mask is plausible.
    grid_n = (GW + 1) * (GH + 1)
    tighten = 0
    while len(cells0) > 0.12 * grid_n and tighten < 6:
        thr = thr * 0.7
        tighten += 1
        cells0 = object_cells(init_rows, ref, thr, bright_min, full_box)
    vinfo["thr_tighten"] = tighten
    vinfo["chroma_thr_final"] = round(thr, 3)
    vinfo["init_cells"] = len(cells0)
    comps0 = components(cells0)
    # Per-object area: the median colour-blob area is a single object as long as most objects
    # are apart; take the smallest such median over three frames (pre-pick, init, last) so one
    # frame with several touching pairs cannot inflate it. Merged blobs are split by k-means.

    def median_area(comps):
        areas = sorted(len(c) for c in comps if len(c) >= 3)
        if len(areas) == 0:
            return None
        return areas[len(areas) // 2]

    area_votes = []
    a_init = median_area(comps0)
    if a_init is not None:
        area_votes.append(a_init)
    for fj in (0, nframes - 1):
        comps_j = components(object_cells(to_rows(video[fj].cameras["front"].rgb), ref, thr, bright_min, full_box))
        a_j = median_area(comps_j)
        if a_j is not None:
            area_votes.append(a_j)
    vinfo["area_votes"] = area_votes
    if len(area_votes) == 0:
        raise ValueError("no object blobs found")
    target_area = min(area_votes)
    min_area = max(3, int(0.3 * target_area))
    centres = []
    blob_k = []
    for comp in comps0:
        if len(comp) < min_area:
            continue
        k = max(1, int(round(len(comp) / float(target_area))))
        blob_k.append([len(comp), k])
        if k == 1:
            centres.append(centroid(comp))
        else:
            for c in kmeans_split(comp, k):
                centres.append(c)
    vinfo["init_blobs_area_k"] = blob_k
    if len(centres) == 0:
        raise ValueError("no objects found at init frame")
    tidx = 0
    bd = None
    for i in range(len(centres)):
        d = (centres[i][0] - tg[0]) ** 2 + (centres[i][1] - tg[1]) ** 2
        if bd is None or d < bd:
            bd = d
            tidx = i
    vinfo["n_objects"] = len(centres)
    vinfo["target_area_cells"] = target_area
    vinfo["init_centres"] = [[round(c[0] * STRIDE, 1), round(c[1] * STRIDE, 1)] for c in centres]
    vinfo["target_index"] = tidx

    r_obj = math.sqrt(target_area / math.pi)
    rmax = max(3.0, 2.2 * r_obj)
    vel = [[0.0, 0.0] for c in centres]
    coast_total = [0 for c in centres]
    target_path = []
    all_path = []
    margin = int(rmax * 3) + 4
    for j in range(init_frame + 1, nframes):
        rows = to_rows(video[j].cameras["front"].rgb)
        pred = []
        for i in range(len(centres)):
            pred.append([centres[i][0] + vel[i][0], centres[i][1] + vel[i][1]])
        xs = [p[0] for p in pred]
        ys = [p[1] for p in pred]
        box = [max(0, int(min(xs)) - margin), max(0, int(min(ys)) - margin),
               min(GW, int(max(xs)) + margin), min(GH, int(max(ys)) + margin)]
        cells = object_cells(rows, ref, thr, bright_min, box)
        newc, counts, members = assign_track(cells, pred, rmax, 3)
        for i in range(len(centres)):
            if counts[i] >= 0.25 * target_area:
                v = [newc[i][0] - centres[i][0], newc[i][1] - centres[i][1]]
                vel[i] = [0.5 * vel[i][0] + 0.5 * v[0], 0.5 * vel[i][1] + 0.5 * v[1]]
                centres[i] = newc[i]
            else:
                centres[i] = pred[i]
                vel[i] = [0.6 * vel[i][0], 0.6 * vel[i][1]]
                coast_total[i] += 1
        if (j - init_frame) % 4 == 0 or j == nframes - 1:
            target_path.append([j, round(centres[tidx][0] * STRIDE, 1), round(centres[tidx][1] * STRIDE, 1),
                                counts[tidx]])
            all_path.append([j] + [[round(c[0] * STRIDE, 1), round(c[1] * STRIDE, 1)] for c in centres] + [counts])
    vinfo["target_path"] = target_path
    vinfo["all_path"] = all_path
    vinfo["coast_frames"] = coast_total
    vinfo["final_centres"] = [[round(c[0] * STRIDE, 1), round(c[1] * STRIDE, 1)] for c in centres]
    report["video"] = vinfo

    # ---------------------------------------------------------------- live localisation
    obs = get_observation()
    front = obs.cameras["front"]
    cam_l = cam_params(front)
    report["live_cam_shift"] = round(dist2(cam_l["t"], cam_v["t"]) + abs(cam_l["t"][2] - cam_v["t"][2]), 5)
    live_rows = to_rows(front.rgb)
    cells_l = object_cells(live_rows, ref, thr, bright_min, full_box)
    live_c, live_counts, live_members = assign_track(cells_l, centres, rmax, 4)
    report["live_counts"] = live_counts
    report["live_centres"] = [[round(c[0] * STRIDE, 1), round(c[1] * STRIDE, 1)] for c in live_c]
    tmem = live_members[tidx]
    if len(tmem) < 3:
        raise ValueError("target track has no live pixels")


    def ray_to_plane(u, v, plane_z, cp):
        """Intersect the camera ray through pixel (u, v) with the horizontal plane z = plane_z."""
        dc = [(u - cp["cx"]) / cp["fx"], (v - cp["cy"]) / cp["fy"], 1.0]
        rot = cp["R"]
        t = cp["t"]
        dw = [rot[0][0] * dc[0] + rot[0][1] * dc[1] + rot[0][2] * dc[2],
              rot[1][0] * dc[0] + rot[1][1] * dc[1] + rot[1][2] * dc[2],
              rot[2][0] * dc[0] + rot[2][1] * dc[1] + rot[2][2] * dc[2]]
        if abs(dw[2]) < 1e-6:
            return None
        s = (plane_z - t[2]) / dw[2]
        return [t[0] + s * dw[0], t[1] + s * dw[1], plane_z]


    # Object pixel centroid ray-cast onto the plane at the video's own grasp height (the expert's
    # grasp point projects onto the visible-pixel centroid of the object it grasped).
    tcen = centroid(tmem)
    tpx = [tcen[0] * STRIDE, tcen[1] * STRIDE]
    report["target_live_px"] = [round(tpx[0], 1), round(tpx[1], 1)]
    tpos = ray_to_plane(tpx[0], tpx[1], grasp_z, cam_l)
    report["target_xy_raycast"] = [round(tpos[0], 4), round(tpos[1], 4)]
    neighbours = []
    for i in range(len(live_c)):
        if i == tidx or len(live_members[i]) < 3:
            continue
        nc = centroid(live_members[i])
        npos = ray_to_plane(nc[0] * STRIDE, nc[1] * STRIDE, grasp_z, cam_l)
        if npos is not None:
            neighbours.append([npos[0], npos[1]])
    report["neighbours"] = [[round(n[0], 4), round(n[1], 4)] for n in neighbours]

    # Diagnostic only: SAM point prompt on the target pixel nearest its centre
    best_cell = tmem[0]
    bd = None
    for c in tmem:
        d = (c[0] - tcen[0]) ** 2 + (c[1] - tcen[1]) ** 2
        if bd is None or d < bd:
            bd = d
            best_cell = c
    prompt_px = [best_cell[0] * STRIDE, best_cell[1] * STRIDE]
    report["prompt_px"] = prompt_px
    try:
        seg = segment_points("front", [prompt_px])
        if seg.ok and len(seg.segmentations) > 0:
            cand = seg.segmentations[0]
            cloud = mask_to_point_cloud(cand, "front", target_frame=base_frame)
            pts = [vec(p) for p in cloud.points]
            if len(pts) >= 10:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                zs = [p[2] for p in pts]
                report["target_sam"] = {"cx": round(quantile(xs, 0.5), 4), "cy": round(quantile(ys, 0.5), 4),
                                        "ztop": round(quantile(zs, 0.9), 4), "zlow": round(quantile(zs, 0.1), 4),
                                        "n": len(pts)}
    except Exception as exc:
        report["sam_error"] = str(exc)[:200]

    # yaw: keep both fingertips (+-finger_reach along the closing axis) away from neighbours


    def finger_clearance(yaw, centre):
        ax = [math.sin(yaw), -math.cos(yaw)]
        clear = 1.0
        for sign in (-1.0, 1.0):
            fx = centre[0] + sign * finger_reach * ax[0]
            fy = centre[1] + sign * finger_reach * ax[1]
            for n in neighbours:
                clear = min(clear, dist2([fx, fy], n))
        return clear


    finger_reach = 0.05
    best_yaw = 0.0
    best_clear = finger_clearance(0.0, tpos)
    if best_clear < 0.035:
        for k in range(12):
            yaw = -math.pi / 2 + k * math.pi / 12
            c = finger_clearance(yaw, tpos)
            if c > best_clear + 1e-4:
                best_clear = c
                best_yaw = yaw
    report["grasp_yaw_deg"] = round(best_yaw * 180.0 / math.pi, 1)
    report["grasp_clearance"] = round(best_clear, 4)
    qyaw = [math.cos(best_yaw / 2.0), 0.0, 0.0, math.sin(best_yaw / 2.0)]
    grasp_quat = quat_mul(qyaw, downward)
    target_geom = {"cx": tpos[0], "cy": tpos[1]}

    # ---------------------------------------------------------------- button
    button = None
    button_info = []
    for prompt in ("button", "white button"):
        if button is not None:
            break
        try:
            segb = segment_text("front", prompt)
        except Exception as exc:
            button_info.append({"prompt": prompt, "error": str(exc)[:200]})
            continue
        if not segb.ok:
            button_info.append({"prompt": prompt, "ok": False})
            continue
        for cand in segb.segmentations[:5]:
            entry = {"prompt": prompt, "score": round(float(cand.score), 3)}
            try:
                cl = mask_to_point_cloud(cand, "front", target_frame=base_frame)
                if len(cl.points) >= 10:
                    s = summarize([vec(p) for p in cl.points])
                    entry["geom"] = s
                    near_object = dist2([s["cx"], s["cy"]], [target_geom["cx"], target_geom["cy"]]) < 0.03
                    for n in neighbours:
                        if dist2([s["cx"], s["cy"]], n) < 0.03:
                            near_object = True
                    object_coloured = False
                    if cand.box_xyxy is not None:
                        bb = [float(v) for v in cand.box_xyxy]
                        px = live_rows[int((bb[1] + bb[3]) / 2.0)][int((bb[0] + bb[2]) / 2.0)]
                        ch = chroma(px)
                        entry["center_rgb"] = px
                        if abs(ch[0] - ref[0]) + abs(ch[1] - ref[1]) + abs(ch[2] - ref[2]) < thr:
                            object_coloured = True
                    entry["near_object"] = near_object
                    entry["object_coloured"] = object_coloured
                    if (s["span"] > 0.015 and s["span"] < 0.12 and (not near_object) and (not object_coloured)
                            and s["ztop"] < grasp_z + 0.08):
                        if button is None:
                            button = s
                            entry["chosen"] = True
            except Exception as exc:
                entry["error"] = str(exc)[:200]
            button_info.append(entry)
    report["button_probe"] = button_info
    report["button"] = button

    # ---------------------------------------------------------------- act
    actions = []
    done = False
    gx = target_geom["cx"]
    gy = target_geom["cy"]
    gz = grasp_z
    report["grasp_point"] = [round(gx, 4), round(gy, 4), round(gz, 4)]


    def mv(pos, quat, tol, steps, label):
        res = move_to_pose(Pose((pos[0], pos[1], pos[2]), quat, base_frame), tolerance=tol, max_steps=steps)
        entry = res_summary(res)
        entry["label"] = label
        try:
            entry["t"] = round(float(get_robot_state().timestamp_s), 2)
        except Exception:
            pass
        actions.append(entry)
        return res


    def act(fn_result, label):
        entry = res_summary(fn_result)
        entry["label"] = label
        actions.append(entry)
        return fn_result


    def kasa_fit(pts):
        """Least-squares circle through xy points; returns (cx, cy, r) or None."""
        n = len(pts)
        if n < 8:
            return None
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        sxx = sxy = syy = sx = sy = sxz = syz = sz = 0.0
        for p in pts:
            x = p[0] - mx
            y = p[1] - my
            z = x * x + y * y
            sxx += x * x
            sxy += x * y
            syy += y * y
            sx += x
            sy += y
            sxz += x * z
            syz += y * z
            sz += z
        m = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, float(n)]]
        rhs = [-sxz, -syz, -sz]

        def det3(q):
            return (q[0][0] * (q[1][1] * q[2][2] - q[1][2] * q[2][1])
                    - q[0][1] * (q[1][0] * q[2][2] - q[1][2] * q[2][0])
                    + q[0][2] * (q[1][0] * q[2][1] - q[1][1] * q[2][0]))
        dm = det3(m)
        if abs(dm) < 1e-18:
            return None
        sol = []
        for col in range(3):
            q = [list(row) for row in m]
            for row in range(3):
                q[row][col] = rhs[row]
            sol.append(det3(q) / dm)
        cx = -sol[0] / 2.0
        cy = -sol[1] / 2.0
        r2 = cx * cx + cy * cy - sol[2]
        if r2 <= 0:
            return None
        return [cx + mx, cy + my, math.sqrt(r2)]


    def wrist_refine(cx0, cy0, tag):
        """From the hover pose, look straight down with the wrist camera: segment the target ring,
        fit its top-rim circle (centre) and find the open sector of the C (slot direction)."""
        info = {"tag": tag}
        try:
            wobs = get_observation()
            wc = wobs.cameras["wrist"]
            cpw = cam_params(wc)
            rows_w = to_rows(wc.rgb)
            hw = len(rows_w)
            ww = len(rows_w[0])
            pc = project([cx0, cy0, grasp_z + 0.02], cpw)
            if pc is None:
                info["error"] = "projection"
                return info
            best = None
            for vv in range(max(0, int(pc[1]) - 60), min(hw, int(pc[1]) + 61), 2):
                row = rows_w[vv]
                for uu in range(max(0, int(pc[0]) - 60), min(ww, int(pc[0]) + 61), 2):
                    px = row[uu]
                    s = px[0] + px[1] + px[2] + 1.0
                    if s < bright_min:
                        continue
                    d = abs(px[0] / s - ref[0]) + abs(px[1] / s - ref[1]) + abs(px[2] / s - ref[2])
                    if d < thr:
                        dd = (uu - pc[0]) ** 2 + (vv - pc[1]) ** 2
                        if best is None or dd < best[0]:
                            best = [dd, uu, vv]
            info["proj_px"] = [round(pc[0], 1), round(pc[1], 1)]
            if best is None:
                info["error"] = "no colour pixel near projection"
                return info
            info["prompt_px"] = [best[1], best[2]]
            segw = segment_points("wrist", [[best[1], best[2]]])
            if (not segw.ok) or len(segw.segmentations) == 0:
                info["error"] = "segment_points failed"
                return info
            cl = mask_to_point_cloud(segw.segmentations[0], "wrist", target_frame=base_frame)
            pts = [vec(p) for p in cl.points]
            pts = [p for p in pts if dist2(p, [cx0, cy0]) < 0.04]
            info["n_pts"] = len(pts)
            if len(pts) < 20:
                info["error"] = "too few points"
                return info
            ztop = quantile([p[2] for p in pts], 0.9)
            top = [p for p in pts if p[2] > ztop - 0.01]
            info["ztop"] = round(ztop, 4)
            info["n_top"] = len(top)
            fit = kasa_fit(top)
            if fit is None:
                info["error"] = "fit failed"
                return info
            sel = [p for p in top if abs(dist2(p, fit) - fit[2]) < 0.012]
            fit2 = kasa_fit(sel)
            if fit2 is not None:
                fit = fit2
            info["fit"] = [round(fit[0], 4), round(fit[1], 4), round(fit[2], 4)]
            if fit[2] < 0.008 or fit[2] > 0.035 or dist2(fit, [cx0, cy0]) > 0.025:
                info["error"] = "implausible circle"
                return info
            nb = 24
            bins = [0] * nb
            for p in top:
                ang = math.atan2(p[1] - fit[1], p[0] - fit[0])
                k = int((ang + math.pi) / (2 * math.pi) * nb) % nb
                bins[k] += 1
            info["bins"] = bins
            level = 0.25 * quantile([float(v) for v in bins], 0.5)
            empty = [1 if bins[k] <= level else 0 for k in range(nb)]
            best_run = 0
            best_mid = None
            for start in range(nb):
                if empty[start] == 0 or empty[(start - 1) % nb] == 1:
                    continue
                ln = 0
                while ln < nb and empty[(start + ln) % nb] == 1:
                    ln += 1
                if ln > best_run:
                    best_run = ln
                    best_mid = start + (ln - 1) / 2.0
            info["slot_bins"] = best_run
            info["centre"] = [fit[0], fit[1]]
            if best_mid is not None and best_run >= 1 and best_run <= 10:
                phi = -math.pi + (best_mid + 0.5) * 2 * math.pi / nb
                info["slot_deg"] = round(phi * 180.0 / math.pi, 1)
                info["slot"] = phi
        except Exception as exc:
            info["error"] = str(exc)[:200]
        return info


    def choose_yaw(centre, slot):
        """Closing axis perpendicular to the slot (yaw = slot angle mod pi), within +-30 deg if a
        neighbour is too close to a fingertip; without a slot, the clearance-only rule."""
        cands = []
        if slot is not None:
            for k in (0, 1, -1, 2, -2):
                cands.append(slot + k * math.pi / 12)
        else:
            cands = [0.0] + [-math.pi / 2 + k * math.pi / 12 for k in range(12)]
        best_y = None
        best_c = -1.0
        for yv in cands:
            while yv > math.pi / 2:
                yv -= math.pi
            while yv < -math.pi / 2:
                yv += math.pi
            c = finger_clearance(yv, centre)
            if best_y is None:
                best_y = yv
                best_c = c
            if best_c >= 0.035:
                break
            if c > best_c + 1e-4:
                best_y = yv
                best_c = c
        return best_y, best_c


    def yaw_quat(yv):
        return quat_mul([math.cos(yv / 2.0), 0.0, 0.0, math.sin(yv / 2.0)], downward)


    refine_log = []
    picks_done = 0
    r = act(open_gripper(), "open")
    done = is_done(r)
    if not done:
        r = mv([gx, gy, gz + 0.10], grasp_quat, 0.02, 200, "pregrasp")
        done = is_done(r)
    for rep in range(2):
        if done:
            break
        info = wrist_refine(gx, gy, "pick%d" % rep)
        if "centre" in info:
            gx = info["centre"][0]
            gy = info["centre"][1]
        yaw_r, clear_r = choose_yaw([gx, gy], info.get("slot"))
        info["yaw_deg"] = round(yaw_r * 180.0 / math.pi, 1)
        info["clearance"] = round(clear_r, 4)
        info.pop("slot", None)
        info.pop("centre", None)
        info["grasp_xy"] = [round(gx, 4), round(gy, 4)]
        refine_log.append(info)
        grasp_quat = yaw_quat(yaw_r)
        r = mv([gx, gy, gz + 0.10], grasp_quat, 0.02, 150, "hover_aligned")
        if is_done(r):
            done = True
            break
        r = mv([gx, gy, gz], grasp_quat, 0.01, 150, "grasp")
        if is_done(r):
            done = True
            break
        r = act(close_gripper(), "close")
        if is_done(r):
            done = True
            break
        r = mv([gx, gy, gz + 0.10], grasp_quat, 0.02, 150, "lift")
        try:
            held = float(get_robot_state().gripper_positions["primary"])
        except Exception:
            held = None
        actions.append({"label": "held_reading", "g": held})
        if is_done(r):
            done = True
            break
        r = mv([gx, gy, gz + 0.005], grasp_quat, 0.01, 150, "lower")
        if is_done(r):
            done = True
            break
        r = act(open_gripper(), "release")
        if is_done(r):
            done = True
            break
        st = get_robot_state()
        for tick in range(4):
            sr = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=st.joint_positions["primary"],
                gripper_position=1.0, embodiment=st.embodiment)}))
            if is_done(sr):
                done = True
                break
        if done:
            break
        r = mv([gx, gy, gz + 0.10], grasp_quat, 0.02, 150, "retreat")
        picks_done += 1
        if is_done(r):
            done = True
            break
        # Diagnostic only: is the object still on the table at its put-down pixel?
        try:
            rows_r = to_rows(get_observation().cameras["front"].rgb)
            pp = project([gx, gy, gz], cam_l)
            cgx = int(pp[0] / STRIDE)
            cgy = int(pp[1] / STRIDE)
            box_r = [max(0, cgx - 6), max(0, cgy - 6), min(GW, cgx + 6), min(GH, cgy + 6)]
            actions.append({"label": "left_on_table_cells", "n": len(object_cells(rows_r, ref, thr, bright_min, box_r)),
                            "area": target_area})
        except Exception as exc_d:
            actions.append({"label": "left_on_table_cells", "error": str(exc_d)[:120]})
    report["wrist_refine"] = refine_log
    report["picks_done"] = picks_done

    if (not done) and button is not None:
        bx = button["cx"]
        by = button["cy"]
        btop = button["ztop"]
        r = act(close_gripper(), "close_for_press")
        done = is_done(r)
        if not done:
            r = mv([bx, by, btop + 0.08], downward, 0.02, 200, "above_button")
            done = is_done(r)
        if not done:
            r = mv([bx, by, btop - 0.015], downward, 0.01, 80, "press")
            done = is_done(r)
        if not done:
            r = mv([bx, by, btop + 0.08], downward, 0.02, 100, "release_button")
            done = is_done(r)

    report["actions"] = actions
    report["done_flag"] = done
    try:
        report["final_t"] = float(get_robot_state().timestamp_s)
    except Exception:
        pass

try:
    run_all()
except Exception as exc:
    report["fatal_error"] = str(exc)[:500]

result = report
