# VideoUnmaskSwap v03 -- track every container from the cover moment of the conditioning
# video through the swaps (front + wrist RGB-D fused on the table plane), keep tracking
# live until the containers stop, then lift the containers in the order the goal names.
# Only public APIs; every position/colour/count is derived from the current episode.
import math

STEP_F = 2          # front grid stride (pixels)
STEP_W = 3          # wrist grid stride (pixels)
CELL = 0.01         # point aggregation cell (m)
VMAX_M = 0.012      # max track speed per frame (m) -- generic clamp

COLOR_WORDS = ("red", "green", "blue", "yellow", "orange", "purple", "pink", "black", "white")


def whitish(px):
    r = px[0]; g = px[1]; b = px[2]
    mn = min(r, g, b); mx = max(r, g, b)
    return mn >= 125 and mx - mn <= 30


def color_of(px):
    r = px[0]; g = px[1]; b = px[2]
    if r - g > 140 and g < 70 and b < 90:
        return "red"
    if g - r > 40 and g - b > 40:
        return "green"
    if b - r > 40 and b - g > 20:
        return "blue"
    if r > 150 and g > 150 and b < 70 and abs(r - g) < 60:
        return "yellow"
    if r > 100 and b > 100 and g < 0.7 * min(r, b):
        return "purple"
    if max(r, g, b) < 50:
        return "black"
    return None


def parse_goal(text):
    low = text.lower()
    words = []
    for token in low.replace(",", " ").replace(".", " ").split():
        words.append(token)
    order = []
    for idx, token in enumerate(words):
        if token in COLOR_WORDS:
            prev = " ".join(words[max(0, idx - 4):idx])
            plural = ("all" in prev.split()) or (idx + 1 < len(words) and words[idx + 1] in ("cubes",))
            order.append((token, plural))
    return order


def quat_to_mat(q):
    w = float(q[0]); x = float(q[1]); y = float(q[2]); z = float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w /= n; x /= n; y /= n; z /= n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def to_list(arr):
    try:
        return arr.tolist()
    except Exception:
        return [list(row) for row in arr]


def cam_model(camobs, conv):
    K = to_list(camobs.intrinsics)
    return {"fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]), "cy": float(K[1][2]),
            "R": quat_to_mat(camobs.camera_pose.quaternion_wxyz),
            "t": [float(v) for v in camobs.camera_pose.position], "conv": conv}


def backproject(c, u, v, d):
    xc = (u - c["cx"]) * d / c["fx"]
    yc = (v - c["cy"]) * d / c["fy"]
    zc = d
    if c["conv"] == 1:       # OpenGL: x right, y up, z back
        yc = -yc; zc = -zc
    elif c["conv"] == 2:     # ROS link: x forward, y left, z up
        xc, yc, zc = zc, -xc, -yc
    R = c["R"]; t = c["t"]
    return (R[0][0] * xc + R[0][1] * yc + R[0][2] * zc + t[0],
            R[1][0] * xc + R[1][1] * yc + R[1][2] * zc + t[1],
            R[2][0] * xc + R[2][1] * yc + R[2][2] * zc + t[2])


def project(c, p):
    R = c["R"]; t = c["t"]
    d0 = p[0] - t[0]; d1 = p[1] - t[1]; d2 = p[2] - t[2]
    xc = R[0][0] * d0 + R[1][0] * d1 + R[2][0] * d2
    yc = R[0][1] * d0 + R[1][1] * d1 + R[2][1] * d2
    zc = R[0][2] * d0 + R[1][2] * d1 + R[2][2] * d2
    if c["conv"] == 1:
        yc = -yc; zc = -zc
    elif c["conv"] == 2:
        xc, yc, zc = -yc, -zc, xc
    if zc <= 1e-6:
        return None
    return (c["fx"] * xc / zc + c["cx"], c["fy"] * yc / zc + c["cy"])


def choose_convention(camobs, depth):
    """Pick the camera-axis convention under which non-white table pixels are most planar."""
    best = None
    H = len(depth); W = len(depth[0])
    for conv in (0, 1, 2):
        c = cam_model(camobs, conv)
        zs = []
        for v in range(H // 2, H, 8):
            for u in range(0, W, 8):
                d = float(depth[v][u])
                if d > 0.05 and d < 5.0:
                    zs.append(backproject(c, u, v, d)[2])
        if len(zs) < 10:
            continue
        zs.sort()
        spread = zs[(9 * len(zs)) // 10] - zs[len(zs) // 10]
        below = zs[len(zs) // 2] < c["t"][2]
        score = spread + (0.0 if below else 10.0)
        if best is None or score < best[0]:
            best = (score, conv, zs[len(zs) // 2])
    return best


def fg_cells(rgb, depth, cam, static, step, zlo, zhi, cells=None, weight=1.0):
    """Aggregate non-static whitish pixels in [zlo,zhi] into XY cells: key -> [sx, sy, n]."""
    if cells is None:
        cells = {}
    H = len(rgb); W = len(rgb[0])
    for v in range(0, H, step):
        row = rgb[v]; drow = depth[v]; srow = static[v // step] if static is not None else None
        for u in range(0, W, step):
            px = row[u]
            r = px[0]; g = px[1]; b = px[2]
            mn = min(r, g, b)
            if mn < 125 or max(r, g, b) - mn > 30:
                continue
            if srow is not None and srow[u // step]:
                continue
            d = float(drow[u])
            if d <= 0.05 or d > 5.0:
                continue
            p = backproject(cam, u, v, d)
            if p[2] < zlo or p[2] > zhi:
                continue
            key = (int(math.floor(p[0] / CELL)), int(math.floor(p[1] / CELL)))
            s = cells.get(key)
            if s is None:
                cells[key] = [p[0] * weight, p[1] * weight, weight]
            else:
                s[0] += p[0] * weight; s[1] += p[1] * weight; s[2] += weight
    return cells


def cell_list(cells):
    out = []
    for key in cells:
        s = cells[key]
        out.append((s[0] / s[2], s[1] / s[2], s[2]))
    return out


def track_update(tracks, pts, radius, min_w, dt):
    preds = []
    for t in tracks:
        preds.append((t["x"] + t["vx"] * dt, t["y"] + t["vy"] * dt))
    cur = list(preds)
    r2 = radius * radius
    wsum = [0.0] * len(tracks)
    for it in range(3):
        sums = [[0.0, 0.0, 0.0] for t in tracks]
        for p in pts:
            best = -1; bd = r2
            for j in range(len(cur)):
                dx = p[0] - cur[j][0]; dy = p[1] - cur[j][1]
                dd = dx * dx + dy * dy
                if dd < bd:
                    bd = dd; best = j
            if best >= 0:
                s = sums[best]; s[0] += p[0] * p[2]; s[1] += p[1] * p[2]; s[2] += p[2]
        new = []
        for j in range(len(sums)):
            s = sums[j]
            if s[2] >= min_w:
                new.append((s[0] / s[2], s[1] / s[2]))
            else:
                new.append(preds[j])
            wsum[j] = s[2]
        cur = new
    for j in range(len(tracks)):
        t = tracks[j]
        if wsum[j] >= min_w:
            nvx = (cur[j][0] - t["x"]) / dt; nvy = (cur[j][1] - t["y"]) / dt
            t["vx"] = 0.5 * t["vx"] + 0.5 * nvx; t["vy"] = 0.5 * t["vy"] + 0.5 * nvy
        else:
            t["vx"] = 0.7 * t["vx"]; t["vy"] = 0.7 * t["vy"]
            t["lost"] = t["lost"] + 1
        sp = math.sqrt(t["vx"] * t["vx"] + t["vy"] * t["vy"])
        lim = VMAX_M
        if sp > lim:
            t["vx"] = t["vx"] * lim / sp; t["vy"] = t["vy"] * lim / sp
        t["x"] = cur[j][0]; t["y"] = cur[j][1]; t["w"] = wsum[j]
    return tracks


def components_xy(pts, cell):
    occ = {}
    for i in range(len(pts)):
        p = pts[i]
        key = (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)))
        if key in occ:
            occ[key].append(i)
        else:
            occ[key] = [i]
    seen = set(); comps = []
    for key in occ:
        if key in seen:
            continue
        stack = [key]; seen.add(key); members = []
        while stack:
            k = stack.pop(); members.extend(occ[k])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (k[0] + dx, k[1] + dy)
                    if nb in occ and nb not in seen:
                        seen.add(nb); stack.append(nb)
        comps.append([pts[i] for i in members])
    return comps


def static_mask(frames_rgb, step):
    H = len(frames_rgb[0]); W = len(frames_rgb[0][0])
    counts = [[0] * ((W + step - 1) // step) for v in range((H + step - 1) // step)]
    for rgb in frames_rgb:
        for v in range(0, H, step):
            row = rgb[v]; crow = counts[v // step]
            for u in range(0, W, step):
                if whitish(row[u]):
                    crow[u // step] += 1
    need = int(0.85 * len(frames_rgb))
    out = []
    for crow in counts:
        out.append([c >= need for c in crow])
    # dilate by one cell
    Hh = len(out); Ww = len(out[0])
    dil = [[False] * Ww for v in range(Hh)]
    for a in range(Hh):
        for b in range(Ww):
            if out[a][b]:
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        aa = a + da; bb = b + db
                        if 0 <= aa < Hh and 0 <= bb < Ww:
                            dil[aa][bb] = True
    return dil

def kmeans_xy(points, k, iters=8):
    srt = sorted(points, key=lambda p: (p[0], p[1]))
    cs = [(srt[int((i + 0.5) * len(srt) / k)][0], srt[int((i + 0.5) * len(srt) / k)][1]) for i in range(k)]
    groups = []
    for it in range(iters):
        groups = [[] for i in range(k)]
        for p in points:
            bj = 0; bd = None
            for j in range(k):
                d = (p[0] - cs[j][0]) ** 2 + (p[1] - cs[j][1]) ** 2
                if bd is None or d < bd:
                    bd = d; bj = j
            groups[bj].append(p)
        ncs = []
        for j in range(k):
            g = groups[j]
            if len(g) > 0:
                ncs.append((sum(p[0] for p in g) / len(g), sum(p[1] for p in g) / len(g)))
            else:
                ncs.append(cs[j])
        cs = ncs
    return [g for g in groups if len(g) > 0]


MERGE_D = 0.04       # same-colour fragments closer than this are one cube
SIZE_SPLIT = False   # size-based splitting is unreliable under perspective; off in v01


def blob_color(c):
    cols = {}
    for p in c:
        if p[3] is not None:
            cols[p[3]] = cols.get(p[3], 0) + 1
    best = None
    for k in cols:
        if best is None or cols[k] > cols[best]:
            best = k
    return best


def blob_area(c):
    """Metric footprint of a blob: number of distinct 5 mm XY cells (independent of camera distance)."""
    return len(set((int(math.floor(p[0] / 0.005)), int(math.floor(p[1] / 0.005))) for p in c))


def split_blobs(blobs):
    """Split blobs that hold two cubes: by major colour first, then by size against the median."""
    out = []
    for c in blobs:
        cols = {}
        for p in c:
            if p[3] is not None:
                cols[p[3]] = cols.get(p[3], 0) + 1
        major = [k for k in cols if cols[k] >= 0.15 * len(c) and cols[k] >= 8]
        if len(major) >= 2:
            cents = []
            for k in major:
                pk = [p for p in c if p[3] == k]
                cents.append((sum(p[0] for p in pk) / len(pk), sum(p[1] for p in pk) / len(pk)))
            groups = [[] for k in major]
            for p in c:
                bj = 0; bd = None
                for j in range(len(cents)):
                    d = (p[0] - cents[j][0]) ** 2 + (p[1] - cents[j][1]) ** 2
                    if bd is None or d < bd:
                        bd = d; bj = j
                groups[bj].append(p)
            out.extend([g for g in groups if len(g) > 0])
        else:
            out.append(c)
    # merge same-colour fragments closer than a cube (distinct cubes sit under distinct containers)
    merged = True
    while merged:
        merged = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                ci = blob_color(out[i]); cj = blob_color(out[j])
                if ci is None or ci != cj:
                    continue
                xi = sum(p[0] for p in out[i]) / len(out[i]); yi = sum(p[1] for p in out[i]) / len(out[i])
                xj = sum(p[0] for p in out[j]) / len(out[j]); yj = sum(p[1] for p in out[j]) / len(out[j])
                if (xi - xj) ** 2 + (yi - yj) ** 2 < MERGE_D * MERGE_D:
                    out[i] = out[i] + out[j]
                    out.pop(j)
                    merged = True
                    break
            if merged:
                break
    if SIZE_SPLIT and len(out) >= 3:
        areas = [blob_area(c) for c in out]
        sizes = sorted(areas)
        med = sizes[len(sizes) // 2]
        final = []
        for i in range(len(out)):
            c = out[i]
            k = int(round(areas[i] / float(med)))
            if areas[i] > 1.7 * med and k >= 2:
                final.extend(kmeans_xy(c, k))
            else:
                final.append(c)
        out = final
    return out

# ==== END OF PURE FUNCTIONS ====

def nsteps(res):
    try:
        return int(res.steps_executed)
    except Exception:
        return -1


def ended(res):
    try:
        if res.terminated or res.truncated:
            return True
    except Exception:
        pass
    return False


# v02: the expert raises each container well clear of the others (wrist-view scale in the
# recording: camera ~2x further from the table at the top of the lift than v01's 8 cm lift)
LIFT_H = 0.20
HOLD_TICKS = 10     # keep the container raised briefly before putting it back
# v03: association radius for tracking through the swaps is wider than the grasp-refinement
# radius (crowded scenes: a 0.55*side gate let tracks lose their container and jump to a neighbour)
TRACK_RADF = 0.9
report = {"version": "v03"}
DUMP = []
result = report


def main():
    ctx = get_task_context()
    goal = ctx.language
    report["goal"] = goal
    order = parse_goal(goal)
    report["order"] = order
    try:
        meta = robomme_plus.get_task_metadata()
        report["meta"] = {"max_steps": meta.get("max_steps"), "control_hz": meta.get("control_hz"),
                          "sampling": meta.get("conditioning_video_sampling")}
    except Exception as exc:
        report["meta_error"] = str(exc)

    video = robomme_plus.get_conditioning_video()
    nvid = len(video)
    report["video_frames"] = nvid
    live0 = get_observation()
    state0 = get_robot_state()
    emb = state0.embodiment
    report["cams"] = [name for name in live0.cameras]
    if nvid < 10:
        raise ValueError("conditioning video missing")

    fconv = choose_convention(video[0].cameras["front"], to_list(video[0].cameras["front"].depth_m))
    wconv = choose_convention(video[0].cameras["wrist"], to_list(video[0].cameras["wrist"].depth_m))
    report["conv_front"] = fconv
    report["conv_wrist"] = wconv
    table_z = fconv[2]
    report["table_z"] = table_z
    TS = [float(video[k].timestamp_s) for k in range(nvid)]
    report["video_t"] = [TS[0], TS[-1]]


    def vframe(k, camname):
        cam = video[k].cameras[camname]
        conv = fconv[1] if camname == "front" else wconv[1]
        return to_list(cam.rgb), to_list(cam.depth_m), cam_model(cam, conv)


    # robot motion in the video (wrist static mask is only valid if the arm is still)
    j0 = [float(v) for v in video[0].robot_state.joint_positions["primary"]]
    jmax = 0.0
    for k in range(0, nvid, max(1, nvid // 20)):
        jk = [float(v) for v in video[k].robot_state.joint_positions["primary"]]
        jmax = max(jmax, max(abs(jk[i] - j0[i]) for i in range(7)))
    report["video_joint_motion"] = jmax
    use_wrist = jmax < 0.01

    ZLO = table_z + 0.004
    ZHI = table_z + 0.20

    # static masks from frames spread over the later video
    samp = [int(nvid * 0.3 + i * (nvid * 0.7 - 1) / 15.0) for i in range(16)]
    FST = static_mask([to_list(video[i].cameras["front"].rgb) for i in samp], STEP_F)
    WST = static_mask([to_list(video[i].cameras["wrist"].rgb) for i in samp], STEP_W)




    def enc(cells):
        parts = []
        for key in cells:
            s = cells[key]
            parts.append("%d,%d,%d" % (int(round(1000 * s[0] / s[2])), int(round(1000 * s[1] / s[2])), int(round(2 * s[2]))))
        return ";".join(parts)


    def frame_cells(k, dump=False):
        rgb, dep, cm = vframe(k, "front")
        cells = fg_cells(rgb, dep, cm, FST, STEP_F, ZLO, ZHI)
        wcells = {}
        if use_wrist:
            rgb, dep, cm = vframe(k, "wrist")
            wcells = fg_cells(rgb, dep, cm, WST, STEP_W, ZLO, ZHI, None, 0.5)
        if dump:
            DUMP.append([k, round(TS[k], 3), enc(cells), enc(wcells)])
        for key in wcells:
            s = wcells[key]
            c = cells.get(key)
            if c is None:
                cells[key] = [s[0], s[1], s[2]]
            else:
                c[0] += s[0]; c[1] += s[1]; c[2] += s[2]
        return cells


    # cover detection: the largest jump of the container foreground in the first 45% of the video
    counts = []
    kmax = max(4, int(nvid * 0.45))
    for k in range(kmax):
        counts.append(len(frame_cells(k)))
    report["counts_head"] = counts[:60]
    cover = None
    best = 0
    for k in range(2, kmax):
        jump = counts[k] - counts[k - 1]
        if jump > best and counts[k] > 2 * counts[k - 1] + 10:
            best = jump
            cover = k
    report["cover"] = cover
    print("counts_head", counts[:60], "cover", cover)
    if cover is None:
        raise ValueError("no cover event found in conditioning video")
    pre = cover - 1

    # cubes: coloured, raised pixels in the pre-cover frame that the containers cover at the cover frame
    cube_pts = []
    rej = {"diff": 0, "depth": 0, "z": 0, "color": 0}
    for camname in ("front", "wrist"):
        if camname == "wrist" and not use_wrist:
            continue
        R2 = to_list(video[cover].cameras[camname].rgb)
        R1, D1, cm = vframe(pre, camname)
        H = len(R1); W = len(R1[0])
        st = 1 if camname == "front" else 2
        for v in range(0, H, st):
            r1 = R1[v]; r2 = R2[v]; d1 = D1[v]
            for u in range(0, W, st):
                a = r1[u]; b = r2[u]
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]) <= 60:
                    rej["diff"] += 1
                    continue
                col = color_of(a)
                if col is None:
                    rej["color"] += 1
                    continue
                d = float(d1[u])
                if d <= 0.05 or d > 5.0:
                    rej["depth"] += 1
                    continue
                p = backproject(cm, u, v, d)
                if p[2] < table_z - 0.01 or p[2] > table_z + 0.08:
                    rej["z"] += 1
                    continue
                cube_pts.append((p[0], p[1], 1.0, col, camname))
    report["cube_rejects"] = rej
    blobs = split_blobs([c for c in components_xy(cube_pts, 0.006) if len(c) >= 6])
    cubes = []
    for c in blobs:
        cols = {}
        for p in c:
            if p[3] is not None:
                cols[p[3]] = cols.get(p[3], 0) + 1
        col = None
        if len(cols) > 0:
            bestc = None
            for kk in cols:
                if bestc is None or cols[kk] > cols[bestc]:
                    bestc = kk
            if cols[bestc] >= 0.25 * len(c):
                col = bestc
        cams = sorted(set(p[4] for p in c))
        cubes.append({"x": sum(p[0] for p in c) / len(c), "y": sum(p[1] for p in c) / len(c),
                      "n": len(c), "color": col, "cols": cols, "cams": cams})
    report["cubes"] = [{"x": round(c["x"], 4), "y": round(c["y"], 4), "n": c["n"], "color": c["color"],
                        "cols": c["cols"], "cams": c["cams"]} for c in cubes]
    if len(cubes) == 0:
        raise ValueError("no cubes found before cover")

    # container scale from the cover frame
    cov_cells = frame_cells(cover, True)
    cov_pts = cell_list(cov_cells)
    area = len(cov_cells) * CELL * CELL
    side = math.sqrt(area / len(cubes))
    side = max(0.025, min(0.07, side))
    radius = 0.55 * side
    track_r = TRACK_RADF * side
    report["track_r"] = track_r
    report["side_est"] = side

    tracks = []
    for c in cubes:
        tracks.append({"x": c["x"], "y": c["y"], "vx": 0.0, "vy": 0.0, "lost": 0, "w": 0.0,
                       "color": c["color"]})
    track_update(tracks, cov_pts, radius, 1.0, 1.0)
    # containers whose cube colour was not recognised still need a track (otherwise they steal
    # assignments from the tracked ones): add one per uncovered container component
    comps = components_xy([(q[0], q[1], q[2], None) for q in cov_pts], CELL)
    extra = []
    wts = sorted(sum(q[2] for q in c) for c in comps)
    wmed = wts[len(wts) // 2] if len(wts) > 0 else 0.0
    for c in comps:
        wsum = sum(q[2] for q in c)
        if wsum < 0.3 * wmed or len(c) < 3:
            continue
        cx = sum(q[0] * q[2] for q in c) / wsum
        cy = sum(q[1] * q[2] for q in c) / wsum
        near = min([math.sqrt((t["x"] - cx) ** 2 + (t["y"] - cy) ** 2) for t in tracks] + [9.0])
        if near > 1.3 * radius:
            tracks.append({"x": cx, "y": cy, "vx": 0.0, "vy": 0.0, "lost": 0, "w": 0.0, "color": None})
            extra.append([round(cx, 4), round(cy, 4), round(wsum, 1)])
    report["extra_tracks"] = extra
    print("cubes", [[round(c["x"], 3), round(c["y"], 3), c["color"]] for c in cubes], "extra", extra, "side", side)
    track_update(tracks, cov_pts, radius, 1.0, 1.0)
    for t in tracks:
        t["vx"] = 0.0; t["vy"] = 0.0; t["lost"] = 0
    report["tracks_cover"] = [[round(t["x"], 4), round(t["y"], 4), t["color"]] for t in tracks]

    last_t = TS[cover]
    snap = []
    for k in range(cover + 1, nvid):
        dt = max(0.2, (TS[k] - last_t) / 0.05)    # in 20 Hz ticks
        track_update(tracks, cell_list(frame_cells(k, (k - cover) % 2 == 0)), track_r, 1.0, dt)
        last_t = TS[k]
        if k % 64 == 0:
            snap.append([k, [[round(t["x"], 3), round(t["y"], 3)] for t in tracks]])
    report["track_snapshots"] = snap
    def px_of(t):
        q = project(FC0, (t["x"], t["y"], table_z + 0.5 * side))
        return None if q is None else [round(q[0], 1), round(q[1], 1)]


    FC0 = cam_model(video[0].cameras["front"], fconv[1])
    report["front_cam0"] = FC0
    report["wrist_cam0"] = cam_model(video[0].cameras["wrist"], wconv[1])
    report["tracks_video_end_px"] = [px_of(t) for t in tracks]
    report["tracks_video_end"] = [[round(t["x"], 4), round(t["y"], 4), t["color"], t["lost"]] for t in tracks]

    # ---- live tracking: hold joints, observe the front camera each tick until containers stop ----
    lf = live0.cameras["front"]
    LFC = cam_model(lf, fconv[1])
    ldep = to_list(lf.depth_m)
    ref = to_list(video[samp[-1]].cameras["front"].depth_m)
    LST = []
    for a in range(len(FST)):
        rowm = []
        for b in range(len(FST[0])):
            v = a * STEP_F; u = b * STEP_F
            same = False
            if FST[a][b] and v < len(ldep) and u < len(ldep[0]):
                same = abs(float(ldep[v][u]) - float(ref[v][u])) < 0.01
            rowm.append(same)
        LST.append(rowm)
    hold = [float(v) for v in state0.joint_positions["primary"]]
    still = 0
    ticks = 0
    terminated = False
    prev_pos = [(t["x"], t["y"]) for t in tracks]
    moves = []
    obs = live0
    while ticks < 220:
        cells = fg_cells(to_list(obs.cameras["front"].rgb), to_list(obs.cameras["front"].depth_m), LFC, LST,
                         STEP_F, ZLO, ZHI)
        DUMP.append(["L%d" % ticks, 0.0, enc(cells), ""])
        track_update(tracks, cell_list(cells), track_r, 1.0, 1.0)
        mv = 0.0
        for j in range(len(tracks)):
            mv = max(mv, math.sqrt((tracks[j]["x"] - prev_pos[j][0]) ** 2 + (tracks[j]["y"] - prev_pos[j][1]) ** 2))
        prev_pos = [(t["x"], t["y"]) for t in tracks]
        moves.append(round(mv, 4))
        if mv < 0.0015:
            still += 1
        else:
            still = 0
        if still >= 12 and ticks >= 20:
            break
        reply = step(RobotAction(arms={"primary": ArmCommand(mode="joint_position", target=hold,
                                                             gripper_position=1.0, embodiment=emb)}))
        ticks += 1
        if ended(reply):
            terminated = True
            break
        try:
            obs = reply.observation
            test = obs.cameras["front"]
        except Exception:
            obs = get_observation()
    report["live_ticks"] = ticks
    report["live_moves"] = moves
    report["tracks_final_px"] = [px_of(t) for t in tracks]
    report["tracks_final"] = [[round(t["x"], 4), round(t["y"], 4), t["color"], t["lost"], round(t["w"], 1)]
                              for t in tracks]

    # ---- pick order from the goal ----
    picks = []
    used = set()
    for item in order:
        col = item[0]; plural = item[1]
        cand = [j for j in range(len(tracks)) if tracks[j]["color"] == col and j not in used]
        if not plural:
            cand = cand[:1]
        for j in cand:
            picks.append(j); used.add(j)
    report["picks"] = picks
    report["pick_colors"] = [tracks[j]["color"] for j in picks]

    # ---- manipulation ----
    st = get_robot_state()
    q_cur = [float(v) for v in st.end_effector_poses["primary"].quaternion_wxyz]
    Rm = quat_to_mat(q_cur)
    if Rm[2][2] < -0.9:
        q_down = (q_cur[0], q_cur[1], q_cur[2], q_cur[3])
    else:
        q_down = (0.0, 1.0, 0.0, 0.0)
    Rd = quat_to_mat(q_down)
    phi0 = math.atan2(Rd[1][1], Rd[0][1])     # base-frame angle of the finger closing axis at yaw 0
    report["phi0"] = phi0
    report["q_start"] = q_cur
    report["ee_start"] = [float(v) for v in st.end_effector_poses["primary"].position]
    base = st.base_frame


    def yaw_quat(yaw):
        return quat_mul((math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)), q_down)


    def wrist_points(x, y, rad):
        o = get_observation()
        wc = o.cameras["wrist"]
        cm = cam_model(wc, wconv[1])
        rgb = to_list(wc.rgb); dep = to_list(wc.depth_m)
        pts = []
        for v in range(0, len(rgb), 2):
            row = rgb[v]; drow = dep[v]
            for u in range(0, len(row), 2):
                if not whitish(row[u]):
                    continue
                d = float(drow[u])
                if d <= 0.03 or d > 3.0:
                    continue
                p = backproject(cm, u, v, d)
                if p[2] < ZLO or p[2] > table_z + 0.15:
                    continue
                if (p[0] - x) ** 2 + (p[1] - y) ** 2 < rad * rad:
                    pts.append(p)
        return pts


    def refine(x, y):
        allp = wrist_points(x, y, radius * 2.5)
        pts = []
        for it in range(4):
            pts = [p for p in allp if (p[0] - x) ** 2 + (p[1] - y) ** 2 < (radius * 1.1) ** 2]
            if len(pts) < 10:
                return x, y, None, None, len(pts)
            zs = sorted(p[2] for p in pts)
            ztop = zs[(9 * len(zs)) // 10]
            top = [p for p in pts if p[2] > ztop - 0.012]
            x = sum(p[0] for p in top) / len(top)
            y = sum(p[1] for p in top) / len(top)
        zs = sorted(p[2] for p in pts)
        ztop = zs[(9 * len(zs)) // 10]
        top = [p for p in pts if p[2] > ztop - 0.012]
        best = None
        for deg in range(0, 90, 5):
            a = math.radians(deg)
            ca = math.cos(a); sa = math.sin(a)
            us = sorted((p[0] - x) * ca + (p[1] - y) * sa for p in top)
            vs = sorted(-(p[0] - x) * sa + (p[1] - y) * ca for p in top)
            n = len(us)
            ext = (us[(95 * n) // 100] - us[(5 * n) // 100]) * (vs[(95 * n) // 100] - vs[(5 * n) // 100])
            if best is None or ext < best[0]:
                best = (ext, a)
        return x, y, ztop, best[1], len(pts)


    pick_log = []
    try:
        if not terminated:
            r = open_gripper()
            terminated = ended(r)
        for j in picks:
            if terminated:
                break
            t = tracks[j]
            entry = {"track": j, "color": t["color"], "xy_track": [round(t["x"], 4), round(t["y"], 4)]}
            pick_log.append(entry)
            hover_z = table_z + 0.22
            r = move_to_pose(Pose((t["x"], t["y"], hover_z), yaw_quat(0.0), base), tolerance=0.01, max_steps=200)
            entry["hover_ok"] = [bool(r.ok), nsteps(r)]
            if ended(r):
                terminated = True; break
            x, y, ztop, yaw, npts = refine(t["x"], t["y"])
            entry["refined"] = [round(x, 4), round(y, 4), ztop, yaw, npts]
            if ztop is None:
                x = t["x"]; y = t["y"]
                ztop = table_z + side
            if yaw is None:
                yaw = phi0
            rot = yaw - phi0
            while rot > math.pi / 4:
                rot -= math.pi / 2
            while rot < -math.pi / 4:
                rot += math.pi / 2
            yaw = rot
            entry["yaw_rot"] = round(yaw, 3)
            height = max(0.02, ztop - table_z)
            gz = ztop - min(0.02, 0.45 * height)
            entry["gz"] = gz
            q = yaw_quat(yaw)
            seq = [("pre", (x, y, ztop + 0.08), 0.01), ("down", (x, y, gz), 0.008)]
            for name, pos, tol in seq:
                r = move_to_pose(Pose(pos, q, base), tolerance=tol, max_steps=150)
                entry[name + "_ok"] = [bool(r.ok), nsteps(r)]
                if ended(r):
                    terminated = True; break
            if terminated:
                break
            r = close_gripper()
            entry["close_ok"] = [bool(r.ok), nsteps(r)]
            if ended(r):
                terminated = True; break
            r = move_to_pose(Pose((x, y, gz + LIFT_H), q, base), tolerance=0.01, max_steps=200)
            entry["lift_ok"] = [bool(r.ok), nsteps(r)]
            if ended(r):
                terminated = True; break
            try:
                sl = get_robot_state()
                entry["grip_after_lift"] = float(sl.gripper_positions["primary"])
                entry["ee_z_lift"] = round(float(sl.end_effector_poses["primary"].position[2]), 4)
                jl = [float(v) for v in sl.joint_positions["primary"]]
                for hh in range(HOLD_TICKS):
                    r = step(RobotAction(arms={"primary": ArmCommand(mode="joint_position", target=jl,
                                                                     gripper_position=0.0, embodiment=emb)}))
                    if ended(r):
                        terminated = True
                        break
            except Exception:
                pass
            if terminated:
                break
            r = move_to_pose(Pose((x, y, gz + 0.003), q, base), tolerance=0.01, max_steps=200)
            entry["lower_ok"] = [bool(r.ok), nsteps(r)]
            if ended(r):
                terminated = True; break
            r = open_gripper()
            entry["open_ok"] = [bool(r.ok), nsteps(r)]
            if ended(r):
                terminated = True; break
            r = move_to_pose(Pose((x, y, ztop + 0.10), q, base), tolerance=0.015, max_steps=150)
            entry["up_ok"] = [bool(r.ok), nsteps(r)]
            try:
                entry["t_end"] = float(get_robot_state().timestamp_s)
            except Exception:
                pass
            if ended(r):
                terminated = True; break
    except Exception as exc:
        report["pick_error"] = str(exc)
    report["pick_log"] = pick_log
    report["cube_pts"] = ";".join("%d,%d,%s,%s" % (int(round(1000 * p[0])), int(round(1000 * p[1])), p[3], p[4][0]) for p in cube_pts[::2])
    report["dump"] = DUMP
    report["terminated"] = terminated


try:
    main()
except Exception as exc:
    report["fatal"] = str(exc)
summ = {}
for key in ("goal", "order", "cover", "conv_front", "table_z", "cubes", "cube_rejects", "live_ticks", "picks",
            "pick_colors", "pick_error", "fatal", "terminated"):
    if key in report:
        summ[key] = report[key]
print("SUMMARY", summ)
report["dump"] = DUMP
result = report
