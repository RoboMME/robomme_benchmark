# VideoUnmask v00_initial -- ASPIRE task worker, robomme_plus task 14.
#
# Strategy (all task-dependent quantities perceived at runtime):
#   1. Read goal text; parse the ordered colour clauses ("the one" vs "all").
#   2. Self-calibrate the RGB-D back-projection convention from the live front
#      camera (table-plane flatness) and a SAM cloud cross-check.
#   3. Scan the conditioning video (front + wrist, both RGB-D with per-frame
#      calibration): detect saturated colour blobs, back-project to robot_base,
#      cluster per colour across frames => remembered cube positions.
#   4. Live: back-project front+wrist depth, find container components above the
#      table, match each remembered target cube to the container covering it.
#   5. For each clause in order: hover above container, refine from wrist view,
#      top-down grasp aligned to the container edges, lift, check gripper, put it
#      back, release, retreat.
# Heavily instrumented: everything lands in `result`.

import math

report = {"version": "v00_initial", "errors": [], "phases": [], "picks": []}
result = report
state = {"ended": False, "end_reason": None}

COLOR_WORDS = ("red", "green", "blue", "yellow", "orange", "purple", "pink", "cyan")


def note_error(where, exc):
    text = where + ": " + str(exc)
    report["errors"].append(text[:400])
    if "Episode has ended" in text or "execution is closing" in text:
        state["ended"] = True
        state["end_reason"] = "exception:" + where


def ffloat(value, default=None):
    try:
        out = float(value)
        if math.isfinite(out):
            return out
        return default
    except Exception:
        return default


def rnd(value, digits=4):
    try:
        return round(float(value), digits)
    except Exception:
        return None


# ---------------------------------------------------------------- geometry

def quat_to_mat(q):
    w = float(q[0])
    x = float(q[1])
    y = float(q[2])
    z = float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    w = w / n
    x = x / n
    y = y / n
    z = z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def cam_params(cam):
    k = cam.intrinsics
    fx = float(k[0][0])
    fy = float(k[1][1])
    cx = float(k[0][2])
    cy = float(k[1][2])
    pose = cam.camera_pose
    rot = quat_to_mat(pose.quaternion_wxyz)
    pos = [float(pose.position[0]), float(pose.position[1]), float(pose.position[2])]
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "R": rot, "t": pos}


COMBOS = []
for axes_name in ("cv", "gl", "ros"):
    for depth_mode in ("z", "range"):
        for inverse_pose in (False, True):
            COMBOS.append((axes_name, depth_mode, inverse_pose))


def project(u, v, d, cp, combo):
    xn = (u - cp["cx"]) / cp["fx"]
    yn = (v - cp["cy"]) / cp["fy"]
    if combo[1] == "range":
        s = d / math.sqrt(xn * xn + yn * yn + 1.0)
    else:
        s = d
    ox = xn * s
    oy = yn * s
    oz = s
    if combo[0] == "cv":
        bx, by, bz = ox, oy, oz
    elif combo[0] == "gl":
        bx, by, bz = ox, -oy, -oz
    else:
        bx, by, bz = oz, -ox, -oy
    r = cp["R"]
    t = cp["t"]
    if not combo[2]:
        return (
            r[0][0] * bx + r[0][1] * by + r[0][2] * bz + t[0],
            r[1][0] * bx + r[1][1] * by + r[1][2] * bz + t[1],
            r[2][0] * bx + r[2][1] * by + r[2][2] * bz + t[2],
        )
    px = bx - t[0]
    py = by - t[1]
    pz = bz - t[2]
    return (
        r[0][0] * px + r[1][0] * py + r[2][0] * pz,
        r[0][1] * px + r[1][1] * py + r[2][1] * pz,
        r[0][2] * px + r[1][2] * py + r[2][2] * pz,
    )


def world_to_pixel(p, cp, combo):
    r = cp["R"]
    t = cp["t"]
    if not combo[2]:
        dx = p[0] - t[0]
        dy = p[1] - t[1]
        dz = p[2] - t[2]
        bx = r[0][0] * dx + r[1][0] * dy + r[2][0] * dz
        by = r[0][1] * dx + r[1][1] * dy + r[2][1] * dz
        bz = r[0][2] * dx + r[1][2] * dy + r[2][2] * dz
    else:
        bx = r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0]
        by = r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1]
        bz = r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2]
    if combo[0] == "cv":
        ox, oy, oz = bx, by, bz
    elif combo[0] == "gl":
        ox, oy, oz = bx, -by, -bz
    else:
        ox, oy, oz = -by, -bz, bx
    if oz <= 1e-6:
        return None
    return (cp["fx"] * ox / oz + cp["cx"], cp["fy"] * oy / oz + cp["cy"], oz)


def to_rows(array, as_int):
    try:
        return array.tolist()
    except Exception:
        rows = []
        for row in array:
            if as_int:
                rows.append([[int(c) for c in px] for px in row])
            else:
                rows.append([float(px) for px in row])
        return rows


def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    idx = int((len(ordered) - 1) * fraction)
    return ordered[idx]


def median(values):
    return quantile(values, 0.5)


# ---------------------------------------------------------------- colour blobs

def classify_rgb(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 45:
        return None
    sat = (mx - mn) / float(mx)
    if sat < 0.80 or (mx - mn) < 40:
        return None
    if mx == r:
        hue = 60.0 * (((g - b) / float(mx - mn)) % 6.0)
    elif mx == g:
        hue = 60.0 * (((b - r) / float(mx - mn)) + 2.0)
    else:
        hue = 60.0 * (((r - g) / float(mx - mn)) + 4.0)
    if hue < 12.0 or hue >= 340.0:
        return "red"
    if hue < 45.0:
        return "orange"
    if hue < 75.0:
        return "yellow"
    if hue < 165.0:
        return "green"
    if hue < 195.0:
        return "cyan"
    if hue < 265.0:
        return "blue"
    if hue < 315.0:
        return "purple"
    return "pink"


def find_blobs(rows, width, height, stride, min_count, cell):
    cells = {}
    v = 0
    while v < height:
        row = rows[v]
        u = 0
        while u < width:
            px = row[u]
            name = classify_rgb(px[0], px[1], px[2])
            if name is not None:
                key = (u // cell, v // cell)
                if key in cells:
                    cells[key].append((u, v, name))
                else:
                    cells[key] = [(u, v, name)]
            u += stride
        v += stride
    seen = set()
    blobs = []
    for key in cells:
        if key in seen:
            continue
        seen.add(key)
        stack = [key]
        members = []
        while stack:
            cur = stack.pop()
            members.extend(cells[cur])
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    nb = (cur[0] + du, cur[1] + dv)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        if len(members) < min_count:
            continue
        votes = {}
        for m in members:
            votes[m[2]] = votes.get(m[2], 0) + 1
        best_name = None
        best_votes = -1
        for name in votes:
            if votes[name] > best_votes:
                best_votes = votes[name]
                best_name = name
        if best_votes < 0.6 * len(members):
            continue
        pix = [(m[0], m[1]) for m in members if m[2] == best_name]
        su = sum(p[0] for p in pix) / float(len(pix))
        sv = sum(p[1] for p in pix) / float(len(pix))
        blobs.append({"color": best_name, "u": su, "v": sv, "n": len(pix), "pixels": pix})
    return blobs


def blob_world(blob, cam, cp, combo):
    pix = blob["pixels"]
    step_n = max(1, len(pix) // 12)
    xs = []
    ys = []
    zs = []
    depth = cam.depth_m
    i = 0
    while i < len(pix):
        u = pix[i][0]
        v = pix[i][1]
        d = ffloat(depth[v][u])
        if d is not None and 0.05 < d < 5.0:
            p = project(u + 0.5, v + 0.5, d, cp, combo)
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])
        i += step_n
    if not xs:
        return None
    return (median(xs), median(ys), median(zs))


# ---------------------------------------------------------------- calibration

def table_plane_score(cam, combo, stride):
    cp = cam_params(cam)
    drows = to_rows(cam.depth_m, False)
    height = len(drows)
    width = len(drows[0])
    zs = []
    v = stride // 2
    while v < height:
        u = stride // 2
        while u < width:
            d = ffloat(drows[v][u])
            if d is not None and 0.05 < d < 5.0:
                zs.append(project(u + 0.5, v + 0.5, d, cp, combo)[2])
            u += stride
        v += stride
    if len(zs) < 20:
        return (0.0, None)
    bins = {}
    for z in zs:
        key = int(math.floor(z / 0.005))
        bins[key] = bins.get(key, 0) + 1
    best_key = None
    best_count = -1
    for key in bins:
        c = bins.get(key - 1, 0) + bins[key] + bins.get(key + 1, 0)
        if c > best_count:
            best_count = c
            best_key = key
    inside = [z for z in zs if abs(z - (best_key + 0.5) * 0.005) <= 0.0075]
    return (best_count / float(len(zs)), median(inside))


def calibrate(live_obs):
    info = {"scores": []}
    front = live_obs.cameras["front"]
    wrist = live_obs.cameras["wrist"]
    front_cam_z = cam_params(front)["t"][2]
    best = None
    best_score = -1.0
    best_z = None
    for combo in COMBOS:
        score, zmode = table_plane_score(front, combo, 8)
        wscore = 0.0
        wz = None
        try:
            wscore, wz = table_plane_score(wrist, combo, 16)
        except Exception as exc:
            note_error("calibrate.wrist_plane", exc)
        # Physical priors: the table lies below the front camera, and both cameras
        # must see the same table height (a mirrored convention also yields a flat
        # plane, but on the wrong side of the camera and at inconsistent heights).
        plausible = zmode is not None and zmode < front_cam_z - 0.05
        if plausible and wz is not None and abs(wz - zmode) > 0.03:
            plausible = False
        info["scores"].append([combo[0], combo[1], combo[2], rnd(score, 3), rnd(zmode), rnd(wscore, 3), rnd(wz), plausible])
        eff = score if plausible else 0.25 * score
        if eff > best_score + 0.02:
            best_score = eff
            best = combo
            best_z = zmode
    info["plane_choice"] = [best[0], best[1], best[2], rnd(best_score, 3), rnd(best_z)]
    # SAM cross-check: cloud from the harness vs our projection of the box centre.
    cross = []
    try:
        base_frame = get_robot_state().base_frame
        loc = localize_object("white box", camera_name="front", target_frame=base_frame)
        info["sam_ok"] = bool(loc.ok)
        if loc.ok:
            pts = loc.point_cloud.points
            npts = len(pts)
            stp = max(1, npts // 1500)
            xs = []
            ys = []
            zs = []
            j = 0
            while j < npts:
                xs.append(float(pts[j][0]))
                ys.append(float(pts[j][1]))
                zs.append(float(pts[j][2]))
                j += stp
            cloud_med = (median(xs), median(ys), median(zs))
            info["sam_cloud_median"] = [rnd(c) for c in cloud_med]
            info["sam_cloud_points"] = npts
            box = None
            try:
                box = loc.segmentation.box_xyxy
            except Exception:
                try:
                    box = loc.segmentation.segmentations[0].box_xyxy
                except Exception:
                    box = None
            if box is not None:
                bu = 0.5 * (float(box[0]) + float(box[2]))
                bv = 0.5 * (float(box[1]) + float(box[3]))
                info["sam_box"] = [rnd(b, 1) for b in box]
                cp = cam_params(front)
                drows = to_rows(front.depth_m, False)
                ds = []
                for dv in (-2, -1, 0, 1, 2):
                    for du in (-2, -1, 0, 1, 2):
                        vv = min(len(drows) - 1, max(0, int(bv) + dv))
                        uu = min(len(drows[0]) - 1, max(0, int(bu) + du))
                        d = ffloat(drows[vv][uu])
                        if d is not None and 0.05 < d < 5.0:
                            ds.append(d)
                if ds:
                    dmed = median(ds)
                    for combo in COMBOS:
                        p = project(bu, bv, dmed, cp, combo)
                        dist = math.sqrt((p[0] - cloud_med[0]) ** 2 + (p[1] - cloud_med[1]) ** 2 + (p[2] - cloud_med[2]) ** 2)
                        cross.append((dist, combo))
    except Exception as exc:
        note_error("calibrate.sam", exc)
    info["cross"] = [[c[1][0], c[1][1], c[1][2], rnd(c[0], 3)] for c in cross]
    chosen = best
    if cross:
        # Prefer the flattest-table combo unless the SAM check clearly refutes it.
        best_cross = None
        for c in cross:
            if best_cross is None or c[0] < best_cross[0]:
                best_cross = c
        plane_cross = None
        for c in cross:
            if c[1] == best:
                plane_cross = c[0]
        info["plane_choice_cross_dist"] = rnd(plane_cross, 3)
        if plane_cross is not None and plane_cross > 0.08 and best_cross[0] < 0.04:
            # check that the cross winner is also reasonably flat
            for row in info["scores"]:
                if (row[0], row[1], row[2]) == best_cross[1] and row[3] is not None and row[3] > 0.5 * best_score:
                    chosen = best_cross[1]
                    info["override_by_sam"] = True
    table_z = None
    for row in info["scores"]:
        if (row[0], row[1], row[2]) == chosen:
            table_z = row[4]
    info["chosen"] = [chosen[0], chosen[1], chosen[2]]
    info["table_z_front"] = table_z
    try:
        wscore, wz = table_plane_score(live_obs.cameras["wrist"], chosen, 8)
        info["wrist_plane"] = [rnd(wscore, 3), rnd(wz)]
    except Exception as exc:
        note_error("calibrate.wrist", exc)
    return chosen, table_z, info


# ---------------------------------------------------------------- goal parsing

def parse_goal(text):
    low = " " + text.lower().replace(",", " , ").replace(".", " . ") + " "
    start = low.find(" then ")
    if start < 0:
        start = 0
    hits = []
    for color in COLOR_WORDS:
        pos = low.find(" " + color + " ", start)
        while pos >= 0:
            hits.append((pos + 1, color))
            pos = low.find(" " + color + " ", pos + 1)
    hits = sorted(hits)
    clauses = []
    prev = start
    for hit in hits:
        pos = hit[0]
        color = hit[1]
        seg = low[prev:pos + len(color) + 8]
        is_all = (" all " in seg) or (" every " in seg) or (" each " in seg) or ((color + " cubes") in seg)
        clauses.append({"color": color, "all": bool(is_all), "text": seg.strip()[:80]})
        prev = pos + len(color)
    return clauses


# ---------------------------------------------------------------- video memory

def scan_video(video, combo, table_z, goal_colors):
    n = len(video)
    fstride = max(1, (n + 119) // 120)
    log = []
    clusters = []
    processed = 0
    for index, frame in enumerate(video):
        if index % fstride != 0 and index != n - 1:
            continue
        processed += 1
        for cam_name in ("front", "wrist"):
            try:
                cam = frame.cameras[cam_name]
                rows = to_rows(cam.rgb, True)
                height = len(rows)
                width = len(rows[0])
                blobs = find_blobs(rows, width, height, 2, 3, 4)
            except Exception as exc:
                note_error("scan." + cam_name + "." + str(index), exc)
                continue
            if not blobs:
                continue
            primary = [b for b in blobs if b["color"] in goal_colors]
            others = sorted([(-b["n"], k) for k, b in enumerate(blobs) if b["color"] not in goal_colors])
            chosen = list(primary)
            for item in others[:6]:
                chosen.append(blobs[item[1]])
            cp = cam_params(cam)
            for b in chosen:
                p = blob_world(b, cam, cp, combo)
                entry = [index, cam_name, b["color"], round(b["u"], 1), round(b["v"], 1), b["n"]]
                if p is None:
                    entry.append(None)
                else:
                    entry.append([rnd(p[0]), rnd(p[1]), rnd(p[2])])
                if len(log) < 260:
                    log.append(entry)
                if p is None:
                    continue
                if table_z is not None and (p[2] < table_z - 0.03 or p[2] > table_z + 0.08):
                    continue
                placed = False
                for cl in clusters:
                    if cl["color"] != b["color"]:
                        continue
                    cxm = cl["sx"] / cl["n"]
                    cym = cl["sy"] / cl["n"]
                    if math.sqrt((p[0] - cxm) ** 2 + (p[1] - cym) ** 2) < 0.03:
                        cl["sx"] += p[0]
                        cl["sy"] += p[1]
                        cl["sz"] += p[2]
                        cl["n"] += 1
                        cl["cams"][cam_name] = cl["cams"].get(cam_name, 0) + 1
                        cl["per_cam"].append([cam_name, rnd(p[0]), rnd(p[1]), rnd(p[2])])
                        cl["last"] = index
                        placed = True
                        break
                if not placed:
                    clusters.append({"color": b["color"], "sx": p[0], "sy": p[1], "sz": p[2], "n": 1,
                                     "cams": {cam_name: 1}, "first": index, "last": index,
                                     "per_cam": [[cam_name, rnd(p[0]), rnd(p[1]), rnd(p[2])]]})
    # merge same-colour clusters closer than 3.5 cm
    merged = []
    for cl in clusters:
        joined = False
        for m in merged:
            if m["color"] != cl["color"]:
                continue
            d = math.sqrt((m["sx"] / m["n"] - cl["sx"] / cl["n"]) ** 2 + (m["sy"] / m["n"] - cl["sy"] / cl["n"]) ** 2)
            if d < 0.035:
                m["sx"] += cl["sx"]
                m["sy"] += cl["sy"]
                m["sz"] += cl["sz"]
                m["n"] += cl["n"]
                for k in cl["cams"]:
                    m["cams"][k] = m["cams"].get(k, 0) + cl["cams"][k]
                m["per_cam"].extend(cl["per_cam"])
                m["first"] = min(m["first"], cl["first"])
                m["last"] = max(m["last"], cl["last"])
                joined = True
                break
        if not joined:
            merged.append(cl)
    cubes = []
    best_support = {}
    for m in merged:
        best_support[m["color"]] = max(best_support.get(m["color"], 0), m["n"])
    for m in merged:
        if m["n"] < 2 or m["n"] < 0.2 * best_support[m["color"]]:
            continue
        spread = {}
        for row in m["per_cam"]:
            if row[0] not in spread:
                spread[row[0]] = []
            spread[row[0]].append(row)
        cam_means = {}
        for k in spread:
            rows_k = spread[k]
            cam_means[k] = [rnd(median([r[1] for r in rows_k])), rnd(median([r[2] for r in rows_k])),
                            rnd(median([r[3] for r in rows_k]))]
        m["cam_means"] = cam_means
        cubes.append({"color": m["color"], "x": m["sx"] / m["n"], "y": m["sy"] / m["n"], "z": m["sz"] / m["n"],
                      "support": m["n"], "cams": m["cams"], "first": m["first"], "last": m["last"],
                      "cam_means": m["cam_means"]})
    return cubes, log, {"frames": n, "fstride": fstride, "processed": processed}


# ---------------------------------------------------------------- live containers

def high_points(cam, combo, table_z, stride, zlo, zhi, center, radius):
    cp = cam_params(cam)
    drows = to_rows(cam.depth_m, False)
    height = len(drows)
    width = len(drows[0])
    pts = []
    v = 0
    while v < height:
        row = drows[v]
        u = 0
        while u < width:
            d = ffloat(row[u])
            if d is not None and 0.05 < d < 5.0:
                p = project(u + 0.5, v + 0.5, d, cp, combo)
                if table_z + zlo < p[2] < table_z + zhi:
                    if center is None or math.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2) < radius:
                        pts.append(p)
            u += stride
        v += stride
    return pts


def components(points, cell):
    grid = {}
    for p in points:
        key = (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)))
        if key in grid:
            grid[key].append(p)
        else:
            grid[key] = [p]
    seen = set()
    comps = []
    for key in grid:
        if key in seen:
            continue
        seen.add(key)
        stack = [key]
        members = []
        while stack:
            cur = stack.pop()
            members.extend(grid[cur])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cur[0] + dx, cur[1] + dy)
                    if nb in grid and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        xs = [m[0] for m in members]
        ys = [m[1] for m in members]
        zs = [m[2] for m in members]
        comps.append({"n": len(members), "x": sum(xs) / len(xs), "y": sum(ys) / len(ys),
                      "top": quantile(zs, 0.9), "xmin": min(xs), "xmax": max(xs),
                      "ymin": min(ys), "ymax": max(ys), "points": members})
    return comps


def square_fit(points_xy):
    # Minimum-area bounding rectangle over angles 0..87 deg (3 deg steps).
    best = None
    for k in range(30):
        ang = math.radians(3.0 * k)
        ca = math.cos(ang)
        sa = math.sin(ang)
        a_vals = [p[0] * ca + p[1] * sa for p in points_xy]
        b_vals = [-p[0] * sa + p[1] * ca for p in points_xy]
        a_lo = quantile(a_vals, 0.03)
        a_hi = quantile(a_vals, 0.97)
        b_lo = quantile(b_vals, 0.03)
        b_hi = quantile(b_vals, 0.97)
        area = (a_hi - a_lo) * (b_hi - b_lo)
        if best is None or area < best[0]:
            ac = 0.5 * (a_lo + a_hi)
            bc = 0.5 * (b_lo + b_hi)
            best = (area, 3.0 * k, ac * ca - bc * sa, ac * sa + bc * ca, a_hi - a_lo, b_hi - b_lo)
    return {"yaw_deg": best[1], "cx": best[2], "cy": best[3], "ext_a": best[4], "ext_b": best[5]}


def describe_container(points, center, table_z):
    near = [p for p in points if math.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2) < 0.05]
    core = [p[2] for p in near if math.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2) < 0.02]
    if len(near) < 6 or len(core) < 3:
        return None
    top = quantile(core, 0.9)
    face = [p for p in near if abs(p[2] - top) < 0.012]
    pieces = components(face, 0.005)
    chosen = None
    for piece in pieces:
        d = math.sqrt((piece["x"] - center[0]) ** 2 + (piece["y"] - center[1]) ** 2)
        if piece["n"] >= 4 and (chosen is None or d < chosen[0]):
            chosen = (d, piece)
    if chosen is None:
        return None
    face_xy = [(p[0], p[1]) for p in chosen[1]["points"]]
    fit = square_fit(face_xy)
    fit["merged"] = False
    if max(fit["ext_a"], fit["ext_b"]) > 0.07:
        # touching neighbours merged into one face: keep the prior centre, fit the
        # orientation only from the face points close to it
        inner = [(p[0], p[1]) for p in face if math.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2) < 0.03]
        if len(inner) >= 5:
            fit = square_fit(inner)
        fit["cx"] = center[0]
        fit["cy"] = center[1]
        fit["merged"] = True
    fit["top"] = top
    fit["height"] = top - table_z
    fit["n"] = len(face_xy)
    fit["pieces"] = len(pieces)
    return fit


# ---------------------------------------------------------------- robot actions

def check_terminal(res, label):
    term = False
    trunc = False
    try:
        term = bool(res.terminated)
    except Exception:
        term = False
    try:
        trunc = bool(res.truncated)
    except Exception:
        trunc = False
    if term or trunc:
        state["ended"] = True
        state["end_reason"] = label + (":terminated" if term else ":truncated")
    return term or trunc


def summarize(res, label):
    out = {"label": label}
    try:
        out["ok"] = bool(res.ok)
    except Exception:
        out["ok"] = None
    try:
        out["status"] = str(res.status)
    except Exception:
        out["status"] = None
    try:
        out["steps"] = int(res.steps_executed)
    except Exception:
        out["steps"] = None
    try:
        if res.error is not None:
            out["error"] = str(res.error)[:300]
    except Exception:
        out["error"] = None
    out["terminal"] = check_terminal(res, label)
    return out


def now_tick():
    try:
        return rnd(get_robot_state().timestamp_s, 3)
    except Exception as exc:
        note_error("now_tick", exc)
        return None


def do_move(position, quat, base_frame, tol, max_steps, label):
    if state["ended"]:
        return None
    try:
        res = move_to_pose(Pose([position[0], position[1], position[2]], quat, base_frame),
                           tolerance=tol, max_steps=max_steps)
        info = summarize(res, label)
    except Exception as exc:
        note_error("move." + label, exc)
        return None
    info["target"] = [rnd(c) for c in position]
    if not state["ended"]:
        try:
            ee = get_robot_state().end_effector_poses["primary"].position
            info["ee"] = [rnd(ee[0]), rnd(ee[1]), rnd(ee[2])]
            info["t"] = now_tick()
        except Exception as exc:
            note_error("move.state." + label, exc)
    report["phases"].append(info)
    return info


def do_gripper(kind, label):
    if state["ended"]:
        return None
    try:
        if kind == "open":
            res = open_gripper()
        else:
            res = close_gripper()
        info = summarize(res, label)
    except Exception as exc:
        note_error("gripper." + label, exc)
        return None
    if not state["ended"]:
        try:
            info["gripper"] = rnd(get_robot_state().gripper_positions["primary"], 3)
            info["t"] = now_tick()
        except Exception as exc:
            note_error("gripper.state." + label, exc)
    report["phases"].append(info)
    return info


def topdown_quat(psi_deg):
    h = math.radians(psi_deg) / 2.0
    return [0.0, math.cos(h), math.sin(h), 0.0]


def wrap180(a):
    while a > 90.0:
        a -= 180.0
    while a <= -90.0:
        a += 180.0
    return a


def choose_psi(fit_yaw, center, others, psi_home):
    # two closing-axis options aligned with the container edges; pick the one whose
    # finger corridor is clearer of neighbouring containers.
    best = None
    for option in (fit_yaw, fit_yaw + 90.0):
        a = math.radians(option)
        ax = math.cos(a)
        ay = math.sin(a)
        cost = 0.0
        for o in others:
            dx = o[0] - center[0]
            dy = o[1] - center[1]
            along = abs(dx * ax + dy * ay)
            perp = abs(-dx * ay + dy * ax)
            if perp < 0.05 and along < 0.11:
                cost += 0.11 - along
        psi = psi_home + wrap180(option + 90.0 - psi_home)
        if best is None or cost < best[0] - 1e-6:
            best = (cost, psi, option)
    return best


def reveal_check(cube_xyz, combo, label):
    # After lifting, look at the remembered cube location in the front image.
    out = {"label": label}
    try:
        obs = get_observation()
        cam = obs.cameras["front"]
        cp = cam_params(cam)
        pix = world_to_pixel(cube_xyz, cp, combo)
        if pix is None:
            out["pixel"] = None
            return out
        out["pixel"] = [rnd(pix[0], 1), rnd(pix[1], 1)]
        rows = to_rows(cam.rgb, True)
        votes = {}
        for dv in range(-5, 6):
            for du in range(-5, 6):
                vv = int(pix[1]) + dv
                uu = int(pix[0]) + du
                if 0 <= vv < len(rows) and 0 <= uu < len(rows[0]):
                    px = rows[vv][uu]
                    name = classify_rgb(px[0], px[1], px[2])
                    if name is not None:
                        votes[name] = votes.get(name, 0) + 1
        out["votes"] = votes
        try:
            wcam = obs.cameras["wrist"]
            wrows = to_rows(wcam.rgb, True)
            wb = find_blobs(wrows, len(wrows[0]), len(wrows), 3, 3, 6)
            out["wrist_blobs"] = [[b["color"], round(b["u"]), round(b["v"]), b["n"]] for b in wb[:8]]
        except Exception as exc:
            note_error("reveal.wrist", exc)
    except Exception as exc:
        note_error("reveal." + label, exc)
    return out


# ================================================================== main
try:
    ctx = get_task_context()
    goal = str(ctx.language)
    report["goal"] = goal
    try:
        report["task_name"] = str(ctx.task_name)
    except Exception:
        pass
    meta = {}
    try:
        tm = robomme_plus.get_task_metadata()
        for k in tm:
            try:
                val = tm[k]
                if val is None or val is True or val is False:
                    meta[k] = val
                else:
                    meta[k] = str(val)[:300]
            except Exception:
                meta[k] = "?"
    except Exception as exc:
        note_error("task_metadata", exc)
    report["task_metadata"] = meta
    try:
        cm = robomme_plus.get_controller_metadata()
        report["controller_metadata"] = dict((str(k), str(cm[k])[:300]) for k in cm)
    except Exception as exc:
        note_error("controller_metadata", exc)
    max_steps = None
    try:
        max_steps = int(robomme_plus.get_task_metadata()["max_steps"])
    except Exception as exc:
        note_error("max_steps", exc)
    report["max_steps"] = max_steps
    control_hz = 20.0
    try:
        control_hz = float(robomme_plus.get_task_metadata()["control_hz"])
    except Exception:
        control_hz = 20.0

    rs = get_robot_state()
    base_frame = rs.base_frame
    report["base_frame"] = str(base_frame)
    report["embodiment"] = str(rs.embodiment)
    home_pose = rs.end_effector_poses["primary"]
    home_q = [float(c) for c in home_pose.quaternion_wxyz]
    home_p = [float(c) for c in home_pose.position]
    report["home_ee"] = {"p": [rnd(c) for c in home_p], "q": [rnd(c) for c in home_q],
                         "gripper": rnd(rs.gripper_positions["primary"], 3), "t": rnd(rs.timestamp_s, 3)}
    hm = quat_to_mat(home_q)
    alpha_home = math.degrees(math.atan2(hm[1][1], hm[0][1]))
    psi_home = alpha_home + 90.0
    report["home_axes"] = {"closing_axis_deg": rnd(alpha_home, 1), "psi_home": rnd(psi_home, 1),
                           "tool_z_world": [rnd(hm[0][2], 3), rnd(hm[1][2], 3), rnd(hm[2][2], 3)]}

    clauses = parse_goal(goal)
    report["clauses"] = clauses
    goal_colors = set(c["color"] for c in clauses)

    live = get_observation()
    report["live_cameras"] = [str(n) for n in live.cameras]
    cam_info = {}
    for cam_name in live.cameras:
        try:
            cpi = cam_params(live.cameras[cam_name])
            cam_info[str(cam_name)] = {"fx": rnd(cpi["fx"], 2), "fy": rnd(cpi["fy"], 2), "cx": rnd(cpi["cx"], 2),
                                       "cy": rnd(cpi["cy"], 2), "t": [rnd(c) for c in cpi["t"]],
                                       "q": [rnd(c) for c in live.cameras[cam_name].camera_pose.quaternion_wxyz],
                                       "frame": str(live.cameras[cam_name].camera_pose.frame),
                                       "shape": [len(live.cameras[cam_name].rgb), len(live.cameras[cam_name].rgb[0])]}
        except Exception as exc:
            note_error("cam_info." + str(cam_name), exc)
    report["live_cam_info"] = cam_info

    combo, table_z, calib = calibrate(live)
    report["calibration"] = calib
    if table_z is None:
        table_z = 0.0

    # ---------------- conditioning video memory
    video = []
    try:
        video = list(robomme_plus.get_conditioning_video())
    except Exception as exc:
        note_error("conditioning_video", exc)
    vinfo = {"len": len(video)}
    if len(video) > 0:
        vinfo["t_first"] = rnd(video[0].timestamp_s, 3)
        vinfo["t_last"] = rnd(video[len(video) - 1].timestamp_s, 3)
        try:
            c0 = cam_params(video[0].cameras["front"])
            vinfo["front_pose_first"] = [rnd(c) for c in c0["t"]]
            c1 = cam_params(video[0].cameras["wrist"])
            vinfo["wrist_pose_first"] = [rnd(c) for c in c1["t"]]
            c2 = cam_params(video[len(video) - 1].cameras["wrist"])
            vinfo["wrist_pose_last"] = [rnd(c) for c in c2["t"]]
        except Exception as exc:
            note_error("video.poses", exc)
    report["video"] = vinfo
    cubes = []
    if len(video) > 0:
        cubes, vlog, vstats = scan_video(video, combo, table_z, goal_colors)
        report["video_scan"] = vstats
        report["video_blob_log"] = vlog
    report["cubes"] = [{"color": c["color"], "xyz": [rnd(c["x"]), rnd(c["y"]), rnd(c["z"])],
                        "support": c["support"], "cams": c["cams"], "first": c["first"], "last": c["last"],
                        "cam_means": c["cam_means"]}
                       for c in cubes]

    # ---------------- live containers
    live_pts = []
    for cam_name in ("wrist", "front"):
        try:
            got = high_points(live.cameras[cam_name], combo, table_z, 2, 0.012, 0.12, None, None)
            report["live_points_" + cam_name] = len(got)
            live_pts.extend(got)
        except Exception as exc:
            note_error("live_points." + cam_name, exc)
    comps = components(live_pts, 0.006)
    containers = []
    for cmp in comps:
        ex = max(cmp["xmax"] - cmp["xmin"], cmp["ymax"] - cmp["ymin"])
        if cmp["n"] >= 8 and 0.02 <= ex <= 0.12:
            containers.append(cmp)
    report["live_components"] = [[rnd(c["x"]), rnd(c["y"]), rnd(c["top"]), c["n"],
                                  rnd(max(c["xmax"] - c["xmin"], c["ymax"] - c["ymin"]), 3)] for c in comps[:40]]
    report["live_containers"] = len(containers)

    # Live front-camera SAM probe of containers (instrumentation only).
    try:
        seg = segment_text("front", "white box")
        probe = {"ok": bool(seg.ok), "n": 0, "scores": [], "boxes": []}
        if seg.ok:
            probe["n"] = len(seg.segmentations)
            for cand in seg.segmentations[:8]:
                probe["scores"].append(rnd(cand.score, 3))
                if cand.box_xyxy is not None:
                    probe["boxes"].append([rnd(b, 1) for b in cand.box_xyxy])
        report["sam_front_white_box"] = probe
    except Exception as exc:
        note_error("sam_probe", exc)

    # match cubes -> containers
    for cube in cubes:
        best = None
        for idx, cmp in enumerate(containers):
            d = math.sqrt((cmp["x"] - cube["x"]) ** 2 + (cmp["y"] - cube["y"]) ** 2)
            if best is None or d < best[0]:
                best = (d, idx)
        if best is not None and best[0] < 0.05:
            cube["container"] = best[1]
            cube["container_dist"] = best[0]
            cmp = containers[best[1]]
            cube["container_clean"] = best[0] < 0.02 and max(cmp["xmax"] - cmp["xmin"], cmp["ymax"] - cmp["ymin"]) < 0.075
        else:
            cube["container"] = None
            cube["container_dist"] = None if best is None else best[0]
    report["cube_matches"] = [[c["color"], c["container"], rnd(c["container_dist"], 3), c.get("container_clean")] for c in cubes]

    # ---------------- plan target sequence
    plan = []
    used = set()
    for clause in clauses:
        cands = [c for c in cubes if c["color"] == clause["color"]]
        cands = sorted([(-c["support"], k) for k, c in enumerate(cands)])
        cands = [[c for c in cubes if c["color"] == clause["color"]][item[1]] for item in cands]
        if not clause["all"]:
            cands = cands[:1]
        else:
            # greedy nearest-first order from the previous target (saves control ticks)
            ordered = []
            last_xy = (home_p[0], home_p[1])
            for prev in plan:
                if prev.get("xy") is not None:
                    last_xy = prev["xy"]
            remaining = list(cands)
            while remaining:
                best_k = 0
                best_d = None
                for k, c in enumerate(remaining):
                    d = math.sqrt((c["x"] - last_xy[0]) ** 2 + (c["y"] - last_xy[1]) ** 2)
                    if best_d is None or d < best_d:
                        best_d = d
                        best_k = k
                nxt = remaining.pop(best_k)
                ordered.append(nxt)
                last_xy = (nxt["x"], nxt["y"])
            cands = ordered
        if not cands:
            # unknown-content fallback: a live container no remembered cube maps to
            matched = set(c["container"] for c in cubes if c["container"] is not None)
            unknown = [k for k in range(len(containers)) if k not in matched and k not in used]
            if len(unknown) >= 1:
                k = unknown[0]
                plan.append({"color": clause["color"], "xy": (containers[k]["x"], containers[k]["y"]),
                             "cube_z": table_z + 0.01, "source": "unknown-container-guess", "container": k})
                used.add(k)
            else:
                plan.append({"color": clause["color"], "xy": None, "source": "not-found"})
            continue
        for cube in cands:
            key = cube["container"]
            if key is not None and key in used:
                continue
            if key is not None:
                used.add(key)
            # the remembered cube centre is the prior for the container centre; the
            # close-up wrist refinement at the hover pose corrects it
            xy = (cube["x"], cube["y"])
            plan.append({"color": clause["color"], "xy": xy, "cube_xy": (cube["x"], cube["y"]),
                         "cube_z": cube["z"], "source": "video-cube", "container": key})
    report["plan"] = [{"color": p["color"], "xy": None if p["xy"] is None else [rnd(p["xy"][0]), rnd(p["xy"][1])],
                       "source": p["source"], "container": p.get("container")} for p in plan]
    report["pre_control_tick"] = now_tick()

    # ---------------- execute
    neighbour_xy = [(c["x"], c["y"]) for c in containers] + [(c["x"], c["y"]) for c in cubes]
    first = do_gripper("open", "initial_open")
    for pick_index, target in enumerate(plan):
        if state["ended"]:
            break
        pick = {"index": pick_index, "color": target["color"], "source": target["source"]}
        report["picks"].append(pick)
        if target["xy"] is None:
            pick["skipped"] = "no target"
            continue
        cx = target["xy"][0]
        cy = target["xy"][1]
        desc = describe_container(live_pts, (cx, cy), table_z)
        if desc is None:
            desc = {"yaw_deg": 0.0, "cx": cx, "cy": cy, "top": table_z + 0.05, "height": 0.05, "n": 0}
        else:
            # keep only yaw/top from the far home view; centre stays at the prior
            desc["cx"] = cx
            desc["cy"] = cy
        pick["coarse"] = {"cx": rnd(desc["cx"]), "cy": rnd(desc["cy"]), "top": rnd(desc["top"]),
                          "yaw": desc["yaw_deg"], "h": rnd(desc["height"]), "n": desc["n"]}
        others = [o for o in neighbour_xy if math.sqrt((o[0] - desc["cx"]) ** 2 + (o[1] - desc["cy"]) ** 2) > 0.025]
        choice = choose_psi(desc["yaw_deg"], (desc["cx"], desc["cy"]), others, psi_home)
        psi = choice[1]
        quat = topdown_quat(psi)
        hover_z = desc["top"] + 0.10
        pick["psi"] = rnd(psi, 1)
        hov = do_move((desc["cx"], desc["cy"], hover_z), quat, base_frame, 0.015, 300, "hover_" + str(pick_index))
        if hov is not None and not hov.get("ok") and (hov.get("steps") == 0 or hov.get("steps") is None) and not state["ended"]:
            psi = psi + 90.0 if psi < psi_home else psi - 90.0
            quat = topdown_quat(psi)
            pick["psi_alt"] = rnd(psi, 1)
            hov = do_move((desc["cx"], desc["cy"], hover_z), quat, base_frame, 0.02, 300, "hover_alt_" + str(pick_index))
        if state["ended"]:
            break
        # refine from the wrist view at the hover pose
        try:
            wobs = get_observation()
            near = high_points(wobs.cameras["wrist"], combo, table_z, 2, 0.012, 0.12, (desc["cx"], desc["cy"]), 0.06)
            fine = describe_container(near, (desc["cx"], desc["cy"]), table_z)
            if fine is not None:
                pick["fine"] = {"cx": rnd(fine["cx"]), "cy": rnd(fine["cy"]), "top": rnd(fine["top"]),
                                "yaw": fine["yaw_deg"], "h": rnd(fine["height"]), "n": fine["n"],
                                "ext": [rnd(fine["ext_a"], 3), rnd(fine["ext_b"], 3)], "merged": fine["merged"]}
                shift = math.sqrt((fine["cx"] - desc["cx"]) ** 2 + (fine["cy"] - desc["cy"]) ** 2)
                if shift < 0.025 and fine["n"] >= 12 and not fine["merged"]:
                    desc["cx"] = fine["cx"]
                    desc["cy"] = fine["cy"]
                    desc["top"] = fine["top"]
                    pick["fine_used"] = True
                    desc["height"] = fine["height"]
                    newchoice = choose_psi(fine["yaw_deg"], (desc["cx"], desc["cy"]), others, psi_home)
                    psi = newchoice[1]
                    quat = topdown_quat(psi)
                    pick["psi_fine"] = rnd(psi, 1)
        except Exception as exc:
            note_error("refine." + str(pick_index), exc)
        height = max(0.015, min(0.12, desc["height"]))
        grasp_z = max(table_z + 0.008, desc["top"] - min(0.025, 0.5 * height))
        pick["grasp_z"] = rnd(grasp_z)
        dsc = do_move((desc["cx"], desc["cy"], grasp_z), quat, base_frame, 0.01, 200, "descend_" + str(pick_index))
        if state["ended"]:
            break
        cl = do_gripper("close", "close_" + str(pick_index))
        if state["ended"]:
            break
        held = cl is not None and cl.get("gripper") is not None and cl["gripper"] > 0.03
        pick["held_after_close"] = held
        if not held:
            # confirmed miss: reopen, go up, retry once slightly lower
            do_gripper("open", "reopen_" + str(pick_index))
            grasp_z = max(table_z + 0.008, grasp_z - 0.01)
            do_move((desc["cx"], desc["cy"], grasp_z), quat, base_frame, 0.01, 200, "descend_retry_" + str(pick_index))
            cl = do_gripper("close", "close_retry_" + str(pick_index))
            if state["ended"]:
                break
            held = cl is not None and cl.get("gripper") is not None and cl["gripper"] > 0.03
            pick["held_after_retry"] = held
        lift = do_move((desc["cx"], desc["cy"], grasp_z + 0.10), quat, base_frame, 0.015, 200, "lift_" + str(pick_index))
        if state["ended"]:
            break
        try:
            pick["gripper_after_lift"] = rnd(get_robot_state().gripper_positions["primary"], 3)
        except Exception as exc:
            note_error("lift.state", exc)
        if "cube_xy" in target:
            pick["reveal"] = reveal_check((target["cube_xy"][0], target["cube_xy"][1], target["cube_z"]), combo,
                                          "reveal_" + str(pick_index))
        if state["ended"]:
            break
        do_move((desc["cx"], desc["cy"], grasp_z + 0.006), quat, base_frame, 0.01, 200, "lower_" + str(pick_index))
        if state["ended"]:
            break
        do_gripper("open", "release_" + str(pick_index))
        if state["ended"]:
            break
        do_move((desc["cx"], desc["cy"], desc["top"] + 0.10), quat, base_frame, 0.02, 200, "retreat_" + str(pick_index))
        pick["t_end"] = now_tick() if not state["ended"] else None
except Exception as exc:
    note_error("main", exc)

report["ended"] = state["ended"]
report["end_reason"] = state["end_reason"]
result = report
