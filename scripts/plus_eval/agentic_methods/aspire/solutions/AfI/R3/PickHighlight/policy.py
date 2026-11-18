# v03_midaction_tracking -- PickHighlight (robomme_plus task 6), AfI.
# Goal (ep 0 language): "first press the button, then pick up all highlighted cubes one by one,
# finally press the button to stop".
# Diagnosed in v01 (ep 0, start 2): highlight detection (5/9 white-disc cubes) and the button press
# worked and pick 1 lifted a highlighted cube, but the cubes PERMUTE their positions every ~140
# ticks (swap motions of ~35 ticks: 57-95, 200-232, 335-369, 477-509, 616-651 in ep 0).
# Remembered slots became stale: pick 2's descent ran into a swap (blocked at z 0.051, 141-tick
# timeout) and pick 3 lifted a non-highlighted cube that had moved into a remembered slot
# (episode terminated). The press also wasted 122 ticks in a timeout (target 2 cm below the cap).
#
# v02: (1) as v01: chroma cube detection, white-disc ring test, bounded SAM button;
#      (2) NEW: every control tick is stepped by the program (plan_motion waypoints through step)
#          and the front image is tracked: each cube keeps its highlight label through swaps;
#      (3) NEW: swaps are detected from track motion while the arm is out of the front view; the
#          program acts only in the still interval after a swap, when the measured swap period
#          leaves enough ticks for the action, and parks out of the camera view in between;
#      (4) NEW: target pose is re-measured from its current tracked blob before each grasp;
#          short lift (+8 cm), lower and release; shallow button press (cap - 1.2 cm, few ticks).
# v03 (minimal repair, UNTESTED): v02 ep 0 picked 5 highlighted cubes, but pick 3 took 181 ticks
#      (IK solutions near the robot jumped up to 2.1 rad) and swap 5 (615-649) happened while
#      tracking was paused; re-association by position then marked a picked cube as unpicked, so
#      one highlighted cube was picked twice and one never. v03 resumes tracking of the other
#      cubes inside an action once the predicted swap time is reached, does not let lost tracks
#      adopt blobs during an action, and predicts swaps only from swaps seen while parked.

import math

report = {"version": "v03", "stage": "start", "log": [], "errors": [], "actions": [], "swaps": []}
result = report

CHROMA_MIN = 0.07      # chroma L1 distance from table median that marks a cube pixel
SAT_MIN = 25           # max-min channel spread; neutral (white/grey) pixels are not cubes
WHITE_MIN = 190        # white pixel: every channel >= WHITE_MIN ...
WHITE_SPREAD = 30      # ... and channel spread <= WHITE_SPREAD
GS = 2                 # tracking grid stride (px)
GATE = 6.0             # tracking gate (px) from predicted position to a blob's box
MOVE_PX = 1.0          # per-tick track displacement that counts as motion
SWAP_ON = 3            # consecutive moving ticks that start a swap
SWAP_OFF = 6           # consecutive still ticks that end a swap
MARGIN = 8             # ticks kept free before a predicted swap
DEFAULT_PAUSE = 90     # assumed still interval before a swap period has been measured
def note(entry):
    report["log"].append(entry)


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def to_rows(rgb):
    try:
        return rgb.tolist()
    except Exception:
        rows = []
        for y in range(len(rgb)):
            row = []
            for x in range(len(rgb[y])):
                px = rgb[y][x]
                row.append([int(px[0]), int(px[1]), int(px[2])])
            rows.append(row)
        return rows


def ended(res):
    flag = False
    try:
        flag = flag or bool(res.terminated)
    except Exception:
        pass
    try:
        flag = flag or bool(res.truncated)
    except Exception:
        pass
    return flag


def res_summary(name, res):
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
    except Exception:
        pass
    try:
        if res.error is not None:
            entry["error"] = str(res.error)[:300]
    except Exception:
        pass
    try:
        st = get_robot_state()
        entry["t"] = round(float(st.timestamp_s), 3)
        entry["grip"] = round(float(st.gripper_positions["primary"]), 3)
        entry["ee"] = [round(float(v), 4) for v in st.end_effector_poses["primary"].position]
    except Exception:
        pass
    note(entry)
    return entry


def is_white(px):
    lo = min(px[0], px[1], px[2])
    hi = max(px[0], px[1], px[2])
    return lo >= WHITE_MIN and hi - lo <= WHITE_SPREAD


def table_chroma(rows, h, w):
    rs = []
    gs = []
    bs = []
    for y in range(0, h, 4):
        for x in range(0, w, 4):
            px = rows[y][x]
            s = px[0] + px[1] + px[2] + 1.0
            rs.append(px[0] / s)
            gs.append(px[1] / s)
            bs.append(px[2] / s)
    return (quantile(rs, 0.5), quantile(gs, 0.5), quantile(bs, 0.5))


def cube_pixel_mask(rows, h, w, tc):
    mask = []
    for y in range(h):
        line = []
        row = rows[y]
        for x in range(w):
            px = row[x]
            hi = max(px[0], px[1], px[2])
            lo = min(px[0], px[1], px[2])
            if hi - lo < SAT_MIN:
                line.append(False)
                continue
            s = px[0] + px[1] + px[2] + 1.0
            d = abs(px[0] / s - tc[0]) + abs(px[1] / s - tc[1]) + abs(px[2] / s - tc[2])
            line.append(d > CHROMA_MIN)
        mask.append(line)
    return mask


def components(mask, h, w, min_area):
    seen = [[False] * w for y in range(h)]
    comps = []
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0][x0] or seen[y0][x0]:
                continue
            stack = [(y0, x0)]
            seen[y0][x0] = True
            pix = []
            while len(stack) > 0:
                y, x = stack.pop()
                pix.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            if len(pix) >= min_area:
                comps.append(pix)
    return comps


def split_component(pix, parts):
    """k-means in image space, used only when a blob is several cube areas large."""
    pix_sorted = sorted(pix)
    centers = []
    for k in range(parts):
        p = pix_sorted[int((k + 0.5) * len(pix_sorted) / parts)]
        centers.append([float(p[0]), float(p[1])])
    groups = []
    for it in range(8):
        groups = [[] for k in range(parts)]
        for p in pix:
            best = 0
            bd = None
            for k in range(parts):
                d = (p[0] - centers[k][0]) ** 2 + (p[1] - centers[k][1]) ** 2
                if bd is None or d < bd:
                    bd = d
                    best = k
            groups[best].append(p)
        for k in range(parts):
            if len(groups[k]) > 0:
                centers[k] = [sum(p[0] for p in groups[k]) / len(groups[k]),
                              sum(p[1] for p in groups[k]) / len(groups[k])]
    return [g for g in groups if len(g) > 0]


def detect_cubes(rows, h, w):
    tc = table_chroma(rows, h, w)
    report["table_chroma"] = [round(v, 4) for v in tc]
    mask = cube_pixel_mask(rows, h, w, tc)
    comps = components(mask, h, w, 25)
    areas = sorted(len(c) for c in comps)
    report["blob_areas"] = areas
    if len(areas) == 0:
        return []
    typical = areas[len(areas) // 2]
    blobs = []
    for c in comps:
        if len(c) > 1.7 * typical:
            parts = int(len(c) / typical + 0.5)
            for g in split_component(c, max(2, parts)):
                blobs.append(g)
        else:
            blobs.append(c)
    cubes = []
    for pix in blobs:
        ys = [p[0] for p in pix]
        xs = [p[1] for p in pix]
        cubes.append({"pix": pix, "u": sum(xs) / len(xs), "v": sum(ys) / len(ys),
                      "box": [min(xs), min(ys), max(xs), max(ys)], "area": len(pix)})
    return cubes


def cube_geometry(cube, camera, h, w, base_frame):
    mask = [[False] * w for y in range(h)]
    for p in cube["pix"]:
        mask[p[0]][p[1]] = True
    pc = None
    try:
        pc = mask_to_point_cloud(mask, camera, target_frame=base_frame)
    except Exception as exc:
        report["errors"].append("mask_to_point_cloud(list): " + str(exc)[:200])
        seg = segment_points(camera, [[int(cube["u"]), int(cube["v"])]])
        if seg.ok and len(seg.segmentations) > 0:
            pc = mask_to_point_cloud(seg.segmentations[0].mask, camera, target_frame=base_frame)
    if pc is None or len(pc.points) < 8:
        return False
    pts = pc.points
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    zs = [float(p[2]) for p in pts]
    top = quantile(zs, 0.9)
    bottom = quantile(zs, 0.05)
    tx = []
    ty = []
    for i in range(len(zs)):
        if zs[i] >= top - 0.008:
            tx.append(xs[i])
            ty.append(ys[i])
    cx = quantile(tx, 0.5)
    cy = quantile(ty, 0.5)
    # face orientation: angle (0..85 deg) minimising the top-face bounding rectangle
    best_a = 0
    best_area = None
    for k in range(18):
        a = math.radians(5 * k)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [ca * (tx[i] - cx) + sa * (ty[i] - cy) for i in range(len(tx))]
        vs = [-sa * (tx[i] - cx) + ca * (ty[i] - cy) for i in range(len(tx))]
        area = (quantile(us, 0.95) - quantile(us, 0.05)) * (quantile(vs, 0.95) - quantile(vs, 0.05))
        if best_area is None or area < best_area:
            best_area = area
            best_a = 5 * k
    cube["center"] = [round(cx, 4), round(cy, 4)]
    cube["top_z"] = round(top, 4)
    cube["bottom_z"] = round(bottom, 4)
    cube["face_deg"] = best_a
    cube["span"] = [round(quantile(xs, 0.95) - quantile(xs, 0.05), 4),
                    round(quantile(ys, 0.95) - quantile(ys, 0.05), 4)]
    cube["n"] = len(zs)
    return True


def white_grid(rows, h, w):
    g = []
    for y in range(0, h, 2):
        line = []
        for x in range(0, w, 2):
            line.append(is_white(rows[y][x]))
        g.append(line)
    return g


def new_white_count(rows, ref_white, h, w):
    n = 0
    for y in range(0, h, 2):
        ry = ref_white[y // 2]
        row = rows[y]
        for x in range(0, w, 2):
            if not ry[x // 2] and is_white(row[x]):
                n += 1
    return n


def ring_score(cube, rows, ref_rows):
    """Fraction of ring samples (around the cube blob) that turned white, and quadrant coverage."""
    r0 = (cube["area"] / 3.14159) ** 0.5
    hits = 0
    total = 0
    quads = [0, 0, 0, 0]
    h = len(rows)
    w = len(rows[0])
    for ri in range(4):
        rad = r0 * (1.25 + 0.25 * ri)
        for k in range(24):
            a = 2 * 3.14159 * k / 24.0
            x = int(cube["u"] + rad * math.cos(a) + 0.5)
            y = int(cube["v"] + rad * math.sin(a) + 0.5)
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            total += 1
            if is_white(rows[y][x]) and not is_white(ref_rows[y][x]):
                hits += 1
                quads[k // 6] += 1
    frac = hits / max(1, total)
    return frac, sum(1 for q in quads if q >= 3), quads


def yaw_quat(down_q, yaw):
    # rotate the home (pointing-down) orientation about base z by yaw: q = qz(yaw) * q0
    c = math.cos(yaw / 2.0)
    s = math.sin(yaw / 2.0)
    w0, x0, y0, z0 = [float(v) for v in down_q]
    return (c * w0 - s * z0, c * x0 - s * y0, c * y0 + s * x0, c * z0 + s * w0)


def neutral_button(base_frame):
    """Fallback: largest neutral (white/grey) blob resting on the table, bounded in size."""
    rows = to_rows(get_observation().cameras["front"].rgb)
    h = len(rows)
    w = len(rows[0])
    mask = []
    for y in range(h):
        line = []
        for x in range(w):
            px = rows[y][x]
            hi = max(px[0], px[1], px[2])
            lo = min(px[0], px[1], px[2])
            line.append(hi - lo <= 15 and lo >= 90)
        mask.append(line)
    best = None
    tried = []
    for pix in components(mask, h, w, 30):
        if len(pix) > 1500:
            continue
        m = [[False] * w for y in range(h)]
        for p in pix:
            m[p[0]][p[1]] = True
        pc = mask_to_point_cloud(m, "front", target_frame=base_frame)
        if len(pc.points) < 8:
            continue
        xs = [float(p[0]) for p in pc.points]
        ys = [float(p[1]) for p in pc.points]
        zs = [float(p[2]) for p in pc.points]
        top = quantile(zs, 0.95)
        entry = {"area": len(pix), "top_z": round(top, 4), "bottom_z": round(quantile(zs, 0.05), 4),
                 "span": [round(quantile(xs, 0.95) - quantile(xs, 0.05), 4),
                          round(quantile(ys, 0.95) - quantile(ys, 0.05), 4)]}
        tx = [xs[i] for i in range(len(zs)) if zs[i] >= top - 0.005]
        ty = [ys[i] for i in range(len(zs)) if zs[i] >= top - 0.005]
        entry["center"] = [round(quantile(tx, 0.5), 4), round(quantile(ty, 0.5), 4)]
        tried.append(entry)
        if entry["span"][0] > 0.12 or entry["span"][1] > 0.12 or entry["bottom_z"] > 0.04 or top > 0.10:
            continue
        if best is None or entry["area"] > best["area"]:
            best = entry
    report["neutral_button_candidates"] = tried
    if best is not None:
        best["source"] = "neutral_blob"
    return best


def coloured_fraction(mask, box, rows, tc):
    """Fraction of mask pixels that look cube-coloured (chroma far from the table, saturated)."""
    x0, y0, x1, y1 = [int(v) for v in box]
    hits = 0
    total = 0
    for y in range(max(0, y0), min(len(rows), y1 + 1)):
        for x in range(max(0, x0), min(len(rows[0]), x1 + 1)):
            if not mask[y][x]:
                continue
            px = rows[y][x]
            total += 1
            hi = max(px[0], px[1], px[2])
            lo = min(px[0], px[1], px[2])
            if hi - lo < SAT_MIN:
                continue
            s = px[0] + px[1] + px[2] + 1.0
            if abs(px[0] / s - tc[0]) + abs(px[1] / s - tc[1]) + abs(px[2] / s - tc[2]) > CHROMA_MIN:
                hits += 1
    return hits / max(1, total)


def find_button(base_frame, rows, tc):
    """SAM 'button' candidates, bounded: small, on the table, not a cube-coloured blob."""
    found = []
    for prompt in ("button", "round button"):
        seg = segment_text("front", prompt)
        if not seg.ok:
            continue
        for cand in seg.segmentations[:5]:
            entry = {"prompt": prompt, "score": round(float(cand.score), 3)}
            if cand.box_xyxy is not None:
                entry["box"] = [round(float(v), 1) for v in cand.box_xyxy]
            pc = mask_to_point_cloud(cand.mask, "front", target_frame=base_frame)
            pts = pc.points
            entry["n"] = len(pts)
            if len(pts) < 8:
                entry["rejected"] = "few points"
                found.append(entry)
                continue
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            zs = [float(p[2]) for p in pts]
            top = quantile(zs, 0.95)
            tx = [xs[i] for i in range(len(zs)) if zs[i] >= top - 0.005]
            ty = [ys[i] for i in range(len(zs)) if zs[i] >= top - 0.005]
            entry["center"] = [round(quantile(tx, 0.5), 4), round(quantile(ty, 0.5), 4)]
            entry["top_z"] = round(top, 4)
            entry["bottom_z"] = round(quantile(zs, 0.05), 4)
            sx = quantile(xs, 0.95) - quantile(xs, 0.05)
            sy = quantile(ys, 0.95) - quantile(ys, 0.05)
            entry["span"] = [round(sx, 4), round(sy, 4)]
            if sx > 0.12 or sy > 0.12 or entry["bottom_z"] > 0.04 or top > 0.10:
                entry["rejected"] = "size/height out of bounds"
            if cand.box_xyxy is not None:
                try:
                    cf = coloured_fraction(cand.mask, cand.box_xyxy, rows, tc)
                    entry["coloured_frac"] = round(cf, 3)
                    if cf > 0.5:
                        entry["rejected"] = "cube-coloured"
                except Exception as exc:
                    report["errors"].append("coloured_fraction: " + str(exc)[:200])
            found.append(entry)
        if len([e for e in found if "rejected" not in e]) > 0:
            break
    report["button_candidates"] = found
    good = [e for e in found if "rejected" not in e]
    if len(good) == 0:
        return neutral_button(base_frame)
    best = good[0]
    for e in good:
        if e["score"] > best["score"]:
            best = e
    return best


def grid_mask(rows, tc):
    """rows: subsampled RGB rows (already stride GS). Returns bool grid of cube-coloured samples."""
    out = []
    for row in rows:
        line = []
        for px in row:
            r = px[0]
            g = px[1]
            b = px[2]
            hi = max(r, g, b)
            lo = min(r, g, b)
            if hi - lo < SAT_MIN:
                line.append(False)
                continue
            s = r + g + b + 1.0
            line.append(abs(r / s - tc[0]) + abs(g / s - tc[1]) + abs(b / s - tc[2]) > CHROMA_MIN)
        out.append(line)
    return out


def grid_blobs(mask, min_area):
    h = len(mask)
    w = len(mask[0]) if h > 0 else 0
    seen = [[False] * w for y in range(h)]
    blobs = []
    for y0 in range(h):
        my = mask[y0]
        for x0 in range(w):
            if not my[x0] or seen[y0][x0]:
                continue
            stack = [(y0, x0)]
            seen[y0][x0] = True
            pix = []
            while len(stack) > 0:
                y, x = stack.pop()
                pix.append((x, y))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            if len(pix) >= min_area:
                blobs.append(pix)
    return blobs


def kmeans_split(pix, seeds, iters):
    cents = [[float(s[0]), float(s[1])] for s in seeds]
    groups = []
    for it in range(iters):
        groups = [[] for c in cents]
        for p in pix:
            best = 0
            bd = None
            for k in range(len(cents)):
                d = (p[0] - cents[k][0]) ** 2 + (p[1] - cents[k][1]) ** 2
                if bd is None or d < bd:
                    bd = d
                    best = k
            groups[best].append(p)
        for k in range(len(cents)):
            if len(groups[k]) > 0:
                cents[k] = [sum(p[0] for p in groups[k]) / len(groups[k]),
                            sum(p[1] for p in groups[k]) / len(groups[k])]
    return cents, [len(g) for g in groups]


def track_update(tracks, blobs, ox, oy, gate):
    """tracks: list of dicts with u, v, vu, vv (full-res px), frozen flag. blobs: grid pixel lists.
    ox, oy: full-res offset of grid origin. gate: max full-res distance from prediction to blob."""
    binfo = []
    for pix in blobs:
        us = [ox + GS * p[0] for p in pix]
        vs = [oy + GS * p[1] for p in pix]
        binfo.append({"pix": pix, "u": sum(us) / len(us), "v": sum(vs) / len(vs),
                      "x0": min(us), "x1": max(us), "y0": min(vs), "y1": max(vs), "tracks": []})
    for ti in range(len(tracks)):
        t = tracks[ti]
        if t["frozen"]:
            continue
        pu = t["u"] + t["vu"]
        pv = t["v"] + t["vv"]
        best = None
        bd = None
        for bi in range(len(binfo)):
            b = binfo[bi]
            du = max(b["x0"] - pu, 0.0, pu - b["x1"])
            dv = max(b["y0"] - pv, 0.0, pv - b["y1"])
            d = (du * du + dv * dv) ** 0.5
            if d <= gate and (bd is None or d < bd):
                bd = d
                best = bi
        t["pred"] = (pu, pv)
        if best is None:
            t["lost"] = t["lost"] + 1
            t["vu"] = 0.0
            t["vv"] = 0.0
        else:
            binfo[best]["tracks"].append(ti)
    for b in binfo:
        n = len(b["tracks"])
        if n == 0:
            continue
        if n == 1:
            cents = [[b["u"], b["v"]]]
        else:
            seeds = [((tracks[ti]["pred"][0] - ox) / GS, (tracks[ti]["pred"][1] - oy) / GS) for ti in b["tracks"]]
            gc, sizes = kmeans_split(b["pix"], seeds, 4)
            cents = [[ox + GS * c[0], oy + GS * c[1]] for c in gc]
        for k in range(n):
            t = tracks[b["tracks"][k]]
            nu = cents[k][0]
            nv = cents[k][1]
            t["vu"] = 0.5 * t["vu"] + 0.5 * (nu - t["u"])
            t["vv"] = 0.5 * t["vv"] + 0.5 * (nv - t["v"])
            t["u"] = nu
            t["v"] = nv
            t["lost"] = 0
            t["shared"] = n
    return binfo


# ---------------------------------------------------------------- camera geometry (public pose)
def quat_matrix(q):
    w, x, y, z = [float(v) for v in q]
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def project(p):
    """robot_base point -> (u, v, depth) in the front image (OpenCV camera, z forward)."""
    c = S["cam_pos"]
    R = S["cam_R"]
    K = S["cam_K"]
    d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]]
    pc = [R[0][i] * d[0] + R[1][i] * d[1] + R[2][i] * d[2] for i in range(3)]
    if pc[2] <= 1e-6:
        return (None, None, pc[2])
    return (K[0][2] + K[0][0] * pc[0] / pc[2], K[1][2] + K[1][1] * pc[1] / pc[2], pc[2])


def pixel_to_plane(u, v, zplane):
    """front-image pixel -> robot_base (x, y) on the horizontal plane z = zplane."""
    K = S["cam_K"]
    R = S["cam_R"]
    c = S["cam_pos"]
    dc = [(u - K[0][2]) / K[0][0], (v - K[1][2]) / K[1][1], 1.0]
    dw = [R[i][0] * dc[0] + R[i][1] * dc[1] + R[i][2] * dc[2] for i in range(3)]
    if abs(dw[2]) < 1e-9:
        return None
    t = (zplane - c[2]) / dw[2]
    return (c[0] + t * dw[0], c[1] + t * dw[1])


def out_of_view(p):
    """True when the tool point and a box around the hand project above/outside the image."""
    for off in ((0, 0, 0), (0.06, 0, 0), (-0.06, 0, 0), (0, 0.06, 0), (0, -0.06, 0), (0, 0, 0.12)):
        u, v, d = project((p[0] + off[0], p[1] + off[1], p[2] + off[2]))
        if u is None:
            continue
        if -4 <= u <= S["w"] + 4 and v >= -4:
            return False
    return True


# ---------------------------------------------------------------- per-tick stepping + tracking
def grid_rows(rgb):
    x0, x1, y0, y1 = S["region"]
    try:
        return rgb[y0:y1:GS, x0:x1:GS].tolist()
    except Exception:
        rows = to_rows(rgb)
        return [rows[y][x0:x1:GS] for y in range(y0, y1, GS)]


def track_motion(old):
    moved = 0
    for k in range(len(S["tracks"])):
        t = S["tracks"][k]
        if t["frozen"] or t["lost"] > 0:
            continue
        d = ((t["u"] - old[k][0]) ** 2 + (t["v"] - old[k][1]) ** 2) ** 0.5
        if d > MOVE_PX:
            moved += 1
    return moved


def adopt_orphans(gate):
    """A lost track takes over the nearest blob that no track claims (e.g. a cube put back a bit off)."""
    if S["in_action"]:
        return
    for ti in range(len(S["tracks"])):
        t = S["tracks"][ti]
        if t["frozen"] or t["lost"] == 0:
            continue
        best = None
        bd = None
        for b in S["blobs"]:
            if len(b["tracks"]) > 0:
                continue
            d = ((b["u"] - t["u"]) ** 2 + (b["v"] - t["v"]) ** 2) ** 0.5
            if d <= gate and (bd is None or d < bd):
                bd = d
                best = b
        if best is not None:
            t["u"] = best["u"]
            t["v"] = best["v"]
            t["lost"] = 0
            best["tracks"].append(ti)
            note({"call": "adopt", "track": ti, "dist": round(bd, 1), "t": S["tick"]})


def observe(obs):
    """Update tracks from one front image. Swap detection only while tracking is live."""
    S["obs"] = obs
    if S["track_paused"]:
        return
    rgb = obs.cameras["front"].rgb
    rows = grid_rows(rgb)
    blobs = grid_blobs(grid_mask(rows, S["tc"]), 6)
    old = [(t["u"], t["v"]) for t in S["tracks"]]
    S["blobs"] = track_update(S["tracks"], blobs, S["region"][0], S["region"][2], GATE)
    adopt_orphans(20.0)
    moved = track_motion(old)
    if moved > 0:
        S["mrun"] += 1
        S["srun"] = 0
    else:
        S["srun"] += 1
        S["mrun"] = 0
    if not S["in_swap"] and S["mrun"] >= SWAP_ON:
        S["in_swap"] = True
        S["swap_start"] = S["tick"] - SWAP_ON + 1
    elif S["in_swap"] and S["srun"] >= SWAP_OFF:
        S["in_swap"] = False
        end = S["tick"] - SWAP_OFF
        report["swaps"].append([S["swap_start"], end, 1 if S["in_action"] else 0])
        S["last_end"] = S["tick"]


def tick(target, grip):
    """One control tick to an absolute joint target, then observe. False when the episode ended."""
    if not S["alive"]:
        return False
    r = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=target,
        gripper_position=float(grip), embodiment=S["emb"])}))
    S["tick"] += 1
    if ended(r):
        S["alive"] = False
        note({"call": "step", "ended_at_tick": S["tick"]})
        return False
    if S["track_paused"] and S["pred_next"] is not None and S["tick"] >= S["pred_next"] - 2 * MARGIN:
        mid_action_resume()
    obs = None
    try:
        obs = r.observation
    except Exception:
        obs = None
    try:
        if obs is None:
            obs = get_observation()
        S["joints"] = [float(v) for v in obs.robot_state.joint_positions["primary"]]
        S["grip"] = float(obs.robot_state.gripper_positions["primary"])
        S["ee"] = [float(v) for v in obs.robot_state.end_effector_poses["primary"].position]
        observe(obs)
    except Exception as exc:
        if S["obs_errors"] < 3:
            report["errors"].append("step observation: " + str(exc)[:200])
        S["obs_errors"] += 1
        obs = get_observation()
        st = get_robot_state()
        S["joints"] = [float(v) for v in st.joint_positions["primary"]]
        S["grip"] = float(st.gripper_positions["primary"])
        S["ee"] = [float(v) for v in st.end_effector_poses["primary"].position]
        observe(obs)
    return True


def hold_ticks(n, grip):
    target = S["hold_target"]
    for k in range(n):
        if not tick(target, grip):
            return False
    return True


def go(pos, q, grip, settle, tol, label):
    """Plan to a pose and execute its waypoints one tick at a time (observing every tick)."""
    if not S["alive"]:
        return None
    t0 = S["tick"]
    plan = plan_motion(Pose(tuple(float(v) for v in pos), q, S["base"]))
    if not plan.ok:
        note({"call": "plan", "label": label, "error": str(plan.error)[:200], "pos": [round(v, 4) for v in pos]})
        r = move_to_pose(Pose(tuple(float(v) for v in pos), q, S["base"]), tolerance=tol, max_steps=150)
        res_summary(label + "_fallback", r)
        try:
            S["tick"] += int(r.steps_executed)
        except Exception:
            pass
        if ended(r):
            S["alive"] = False
            return None
        st = get_robot_state()
        S["joints"] = [float(v) for v in st.joint_positions["primary"]]
        S["hold_target"] = st.joint_positions["primary"]
        S["ee"] = [float(v) for v in st.end_effector_poses["primary"].position]
        S["grip"] = float(st.gripper_positions["primary"])
        observe(get_observation())
    else:
        last = None
        for row in plan.trajectory.joint_positions:
            last = row
            S["hold_target"] = row
            if not tick(last, grip):
                return None
        for k in range(settle):
            err = sum((S["ee"][i] - pos[i]) ** 2 for i in range(3)) ** 0.5
            if err <= tol or last is None:
                break
            if not tick(last, grip):
                return None
    err = sum((S["ee"][i] - pos[i]) ** 2 for i in range(3)) ** 0.5
    note({"call": label, "ticks": S["tick"] - t0, "err": round(err, 4), "ee": [round(v, 4) for v in S["ee"]],
          "grip": round(S["grip"], 3), "t": S["tick"]})
    return err


# ---------------------------------------------------------------- scheduling around swaps
def parked_swaps():
    return [s for s in report["swaps"] if s[2] == 0]


def period_estimate():
    starts = [s[0] for s in parked_swaps()]
    if len(starts) < 2:
        return None
    diffs = sorted(starts[i + 1] - starts[i] for i in range(len(starts) - 1))
    return diffs[0]


def window_left():
    if S["in_swap"] or S["mrun"] > 0:
        return 0
    if len(report["swaps"]) == 0:
        if S["tick"] - S["watch_end"] > 160:
            return 100000          # no swap seen for a long time: treat the scene as static
        return 0                   # wait for the first swap (it followed the highlight in ep 0)
    p = period_estimate()
    if p is None or len(parked_swaps()) == 0:
        S["pred_next"] = S["last_end"] + DEFAULT_PAUSE
        return DEFAULT_PAUSE - (S["tick"] - S["last_end"])
    last_start = parked_swaps()[-1][0]
    nxt = last_start + p
    while nxt < S["tick"] - 2 * SWAP_OFF:
        nxt += p                   # a predicted swap did not happen (e.g. motion missed)
    S["pred_next"] = nxt
    return nxt - S["tick"] - MARGIN


def wait_for_window(need, label):
    t0 = S["tick"]
    while S["alive"]:
        left = window_left()
        if left >= need:
            break
        if S["tick"] - t0 > 320:
            note({"call": "wait_timeout", "label": label})
            break
        if S["tick"] > S["max_steps"] - 60:
            break
        if not hold_ticks(1, S["grip_hold"]):
            break
    report["actions"].append({"label": label, "wait_from": t0, "start": S["tick"], "need": need,
                              "window_left": window_left()})


def park_pose():
    return S["park"]


def pause_tracking():
    S["track_paused"] = True
    S["in_action"] = True


def mid_action_resume():
    """v03: the action reached the predicted swap time -> track the other cubes again now
    (v02 kept tracking paused and missed swap 5 during a 181-tick pick)."""
    S["track_paused"] = False
    for t in S["tracks"]:
        t["vu"] = 0.0
        t["vv"] = 0.0
    note({"call": "mid_action_resume", "t": S["tick"], "pred_next": S["pred_next"]})


def resume_tracking():
    """Re-associate every track to the current blobs (no swap happens inside an action window)."""
    S["track_paused"] = False
    S["in_action"] = False
    for t in S["tracks"]:
        t["vu"] = 0.0
        t["vv"] = 0.0
        t["frozen"] = False
    obs = S["obs"]
    rows = grid_rows(obs.cameras["front"].rgb)
    blobs = grid_blobs(grid_mask(rows, S["tc"]), 6)
    S["blobs"] = track_update(S["tracks"], blobs, S["region"][0], S["region"][2], 10.0)
    adopt_orphans(20.0)
    lost = [k for k in range(len(S["tracks"])) if S["tracks"][k]["lost"] > 0]
    if len(lost) > 0:
        note({"call": "reassoc_lost", "tracks": lost, "t": S["tick"]})
    S["mrun"] = 0
    S["srun"] = 0


# ---------------------------------------------------------------- measuring a tracked cube
def measure_track(ti):
    """3D centre/top/face of a tracked cube from its current blob (full-resolution pixels)."""
    t = S["tracks"][ti]
    blob = None
    for b in S["blobs"]:
        if ti in b["tracks"]:
            blob = b
    if blob is None:
        return None
    rows = to_rows(S["obs"].cameras["front"].rgb)
    tc = S["tc"]
    others = [S["tracks"][k] for k in blob["tracks"] if k != ti]
    pix = []
    for y in range(max(0, int(blob["y0"]) - 2), min(S["h"], int(blob["y1"]) + 3)):
        for x in range(max(0, int(blob["x0"]) - 2), min(S["w"], int(blob["x1"]) + 3)):
            px = rows[y][x]
            hi = max(px[0], px[1], px[2])
            lo = min(px[0], px[1], px[2])
            if hi - lo < SAT_MIN:
                continue
            s = px[0] + px[1] + px[2] + 1.0
            if abs(px[0] / s - tc[0]) + abs(px[1] / s - tc[1]) + abs(px[2] / s - tc[2]) <= CHROMA_MIN:
                continue
            d0 = (x - t["u"]) ** 2 + (y - t["v"]) ** 2
            mine = True
            for o in others:
                if (x - o["u"]) ** 2 + (y - o["v"]) ** 2 < d0:
                    mine = False
            if mine:
                pix.append((y, x))
    if len(pix) < 20:
        return None
    cube = {"pix": pix, "u": t["u"], "v": t["v"], "area": len(pix)}
    if not cube_geometry(cube, "front", S["h"], S["w"], S["base"]):
        return None
    cube["shared"] = len(blob["tracks"])
    return cube


def approx_xy(t):
    p = pixel_to_plane(t["u"], t["v"], 0.02)
    if p is None:
        return (0.0, 0.0)
    return p


def choose_yaw2(target, ti):
    """Finger axis along a cube face normal; prefer |yaw| <= 45 deg unless clearance is poor."""
    others = []
    for k in range(len(S["tracks"])):
        if k != ti:
            others.append(approx_xy(S["tracks"][k]))
    best = None
    for k in range(2):
        normal = math.radians(target["face_deg"] + 90 * k)
        dx = math.cos(normal)
        dy = math.sin(normal)
        clear = 1.0
        for o in others:
            for sgn in (-1.0, 1.0):
                fx = target["center"][0] + sgn * 0.05 * dx
                fy = target["center"][1] + sgn * 0.05 * dy
                clear = min(clear, ((o[0] - fx) ** 2 + (o[1] - fy) ** 2) ** 0.5)
        yaw = normal - math.pi / 2.0
        while yaw > math.pi / 2.0:
            yaw -= math.pi
        while yaw < -math.pi / 2.0:
            yaw += math.pi
        score = min(clear, 0.045) - (0.01 if abs(yaw) > math.pi / 4.0 + 0.01 else 0.0)
        if best is None or score > best[0]:
            best = (score, yaw, clear)
    return best


# ---------------------------------------------------------------- actions
def press(button, label):
    t0 = S["tick"]
    bx, by = button["center"]
    top = button["top_z"]
    q = S["down_q"]
    pause_tracking()
    S["grip_hold"] = 0.0
    out = "pressed"
    if go((bx, by, top + 0.05), q, 0.0, 3, 0.01, label + "_above") is None:
        out = "ended"
    elif go((bx, by, top + 0.005), q, 0.0, 3, 0.005, label + "_approach") is None:
        out = "ended"
    elif go((bx, by, top - 0.012), q, 0.0, 2, 0.003, label + "_press") is None:
        out = "ended"
    elif not hold_ticks(4, 0.0):
        out = "ended"
    else:
        note({"call": label + "_bottom", "ee_z": round(S["ee"][2], 4), "cap_top": top, "t": S["tick"]})
        if go((bx, by, top + 0.06), q, 0.0, 0, 0.02, label + "_release") is None:
            out = "ended"
        elif go(park_pose(), q, 0.0, 0, 0.03, label + "_park") is None:
            out = "ended"
    if S["alive"]:
        resume_tracking()
    report["actions"][-1]["end"] = S["tick"]
    report["actions"][-1]["ticks"] = S["tick"] - t0
    report["actions"][-1]["outcome"] = out
    return out


def pick(ti):
    t0 = S["tick"]
    t = S["tracks"][ti]
    cube = measure_track(ti)
    entry = report["actions"][-1]
    entry["track"] = ti
    entry["uv"] = [round(t["u"], 1), round(t["v"], 1)]
    if cube is None:
        entry["outcome"] = "measure_failed"
        return "measure_failed"
    entry["center"] = cube["center"]
    entry["top_z"] = cube["top_z"]
    entry["face_deg"] = cube["face_deg"]
    entry["shared"] = cube["shared"]
    if not (0.02 <= cube["top_z"] <= 0.08):
        entry["outcome"] = "implausible_top"
        return "implausible_top"
    score, yaw, clear = choose_yaw2(cube, ti)
    entry["yaw_deg"] = round(math.degrees(yaw), 1)
    entry["clear"] = round(clear, 3)
    q = yaw_quat(S["down_q"], yaw)
    cx, cy = cube["center"]
    top = cube["top_z"]
    height = top - cube["bottom_z"]
    grasp_z = top - max(0.015, min(0.025, 0.5 * height))
    pause_tracking()
    S["tracks"][ti]["frozen"] = True
    out = "ended"
    held = False
    S["grip_hold"] = 1.0
    while True:
        if go((cx, cy, top + 0.06), q, 1.0, 2, 0.01, "pre_grasp") is None:
            break
        err = go((cx, cy, grasp_z), q, 1.0, 4, 0.005, "descend")
        if err is None:
            break
        if S["ee"][2] > grasp_z + 0.015:
            note({"call": "descend_blocked", "ee_z": round(S["ee"][2], 4), "grasp_z": round(grasp_z, 4)})
            if go((cx, cy, top + 0.08), q, 1.0, 0, 0.03, "abort_up") is None:
                break
            out = "blocked"
            break
        if not hold_ticks(6, 0.0):
            break
        S["grip_hold"] = 0.0
        if go((cx, cy, grasp_z + 0.08), q, 0.0, 0, 0.02, "lift") is None:
            break
        if not hold_ticks(2, 0.0):
            break
        held = S["grip"] > 0.1
        entry["grip_lifted"] = round(S["grip"], 3)
        if go((cx, cy, grasp_z + 0.008), q, 0.0, 2, 0.006, "lower") is None:
            break
        if not hold_ticks(4, 1.0):
            break
        S["grip_hold"] = 1.0
        if go((cx, cy, top + 0.06), q, 1.0, 0, 0.03, "retreat") is None:
            break
        out = "picked" if held else "missed"
        break
    if S["alive"]:
        if go(park_pose(), S["down_q"], 1.0, 0, 0.03, "park") is None:
            out = "ended"
    if S["alive"]:
        resume_tracking()
    entry["end"] = S["tick"]
    entry["ticks"] = S["tick"] - t0
    entry["outcome"] = out
    return out


def snapshot():
    return [[k, round(t["u"]), round(t["v"]), 1 if t["H"] else 0, t["lost"]] for k, t in enumerate(S["tracks"])]


S = {"tick": 0, "alive": True, "track_paused": False, "tracks": [], "blobs": [], "mrun": 0, "srun": 0,
     "in_swap": False, "swap_start": None, "last_end": 0, "watch_end": 0, "grip_hold": 0.03, "obs_errors": 0,
     "in_action": False, "pred_next": None}


try:
    context = get_task_context()
    state = get_robot_state()
    S["base"] = state.base_frame
    S["emb"] = state.embodiment
    S["joints"] = [float(v) for v in state.joint_positions["primary"]]
    S["home_joints"] = list(S["joints"])
    S["hold_target"] = state.joint_positions["primary"]
    S["ee"] = [float(v) for v in state.end_effector_poses["primary"].position]
    S["grip"] = float(state.gripper_positions["primary"])
    S["down_q"] = state.end_effector_poses["primary"].quaternion_wxyz
    base_frame = S["base"]
    report["task_language"] = str(context.language)
    meta = robomme_plus.get_task_metadata()
    S["max_steps"] = int(meta["max_steps"])
    report["max_steps"] = S["max_steps"]
    try:
        report["video_len"] = len(robomme_plus.get_conditioning_video())
    except Exception as exc:
        report["errors"].append("video: " + str(exc)[:200])

    live = get_observation()
    S["obs"] = live
    front = live.cameras["front"]
    h = len(front.rgb)
    w = len(front.rgb[0])
    S["h"] = h
    S["w"] = w
    S["cam_K"] = [[float(v) for v in row] for row in front.intrinsics]
    S["cam_pos"] = [float(v) for v in front.camera_pose.position]
    S["cam_R"] = quat_matrix(front.camera_pose.quaternion_wxyz)
    ref_rows = to_rows(front.rgb)

    # ---- cubes at tick 0 (same as v01)
    cubes = detect_cubes(ref_rows, h, w)
    S["tc"] = report["table_chroma"]
    measured = [c for c in cubes if cube_geometry(c, "front", h, w, base_frame)]
    if len(measured) > 0:
        spans = sorted(max(c["span"]) for c in measured)
        areas = sorted(c["area"] for c in measured)
        typical_span = spans[len(spans) // 2]
        typical_area = areas[len(areas) // 2]
        report["typical_span"] = typical_span
        split = []
        for c in measured:
            if max(c["span"]) > 1.5 * typical_span:
                parts = max(2, int(c["area"] / typical_area + 0.5))
                for g in split_component(c["pix"], parts):
                    ys = [p[0] for p in g]
                    xs = [p[1] for p in g]
                    part = {"pix": g, "u": sum(xs) / len(xs), "v": sum(ys) / len(ys),
                            "box": [min(xs), min(ys), max(xs), max(ys)], "area": len(g), "split_from": c["area"]}
                    if cube_geometry(part, "front", h, w, base_frame):
                        split.append(part)
            else:
                split.append(c)
        cubes = split
    good = []
    for c in cubes:
        if c["span"][0] <= 0.09 and c["span"][1] <= 0.09 and c["bottom_z"] < 0.03 and c["top_z"] < 0.09:
            good.append(c)
        else:
            c["rejected"] = "size/height"
    report["cubes_detected"] = len(cubes)
    button = find_button(base_frame, ref_rows, S["tc"])
    report["button"] = button
    report["stage"] = "scene_found"

    # ---- tracking region (cube boxes + margin) and tracks
    if len(good) > 0:
        bx0 = min(c["box"][0] for c in good)
        by0 = min(c["box"][1] for c in good)
        bx1 = max(c["box"][2] for c in good)
        by1 = max(c["box"][3] for c in good)
        x0 = max(0, int(bx0) - 36)
        x1 = min(w, int(bx1) + 37)
        y0 = max(0, int(by0) - 36)
        y1 = min(h, int(by1) + 37)
        x0 = x0 - (x0 % GS)
        y0 = y0 - (y0 % GS)
        S["region"] = (x0, x1, y0, y1)
    else:
        S["region"] = (0, w, 0, h)
    report["region"] = list(S["region"])
    for c in good:
        S["tracks"].append({"u": c["u"], "v": c["v"], "vu": 0.0, "vv": 0.0, "frozen": False, "lost": 0,
                            "H": False, "xy0": c["center"]})

    # ---- park pose: above the cube cluster, high enough to leave the front view
    if len(good) > 0:
        mx = sum(c["center"][0] for c in good) / len(good)
        my = sum(c["center"][1] for c in good) / len(good)
    else:
        mx = S["ee"][0]
        my = S["ee"][1]
    park = None
    for k in range(10):
        z = 0.24 + 0.02 * k
        if out_of_view((mx, my, z)):
            park = (mx, my, z)
            break
    if park is None:
        park = (S["ee"][0], S["ee"][1], S["ee"][2])
    S["park"] = park
    report["park"] = [round(v, 3) for v in park]

    # ---- watch the highlight at home (tracking runs every tick)
    ref_white = white_grid(ref_rows, h, w)
    counts = []
    best_count = 0
    best_rows = None
    best_tick = None
    on_tick = None
    off_run = 0
    for k in range(200):
        if not hold_ticks(1, S["grip"]):
            break
        rows = to_rows(S["obs"].cameras["front"].rgb)
        n = new_white_count(rows, ref_white, h, w)
        counts.append(n)
        if n > best_count:
            best_count = n
            best_rows = rows
            best_tick = S["tick"]
        if on_tick is None and n >= 25:
            on_tick = S["tick"]
        if on_tick is not None:
            if n < 0.3 * best_count:
                off_run += 1
            else:
                off_run = 0
            if off_run >= 3:
                break
        if on_tick is None and k >= 150:
            break
    S["watch_end"] = S["tick"]
    report["watch_counts"] = counts
    report["highlight_on_tick"] = on_tick
    report["highlight_best_tick"] = best_tick
    report["highlight_best_count"] = best_count
    if best_rows is not None and best_count >= 25:
        for k in range(len(good)):
            c = good[k]
            frac, nq, quads = ring_score(c, best_rows, ref_rows)
            c["ring_frac"] = round(frac, 3)
            c["ring_quads"] = quads
            S["tracks"][k]["H"] = frac >= 0.08 and nq >= 2
    best_rows = None
    report["cubes"] = [{kk: c[kk] for kk in c if kk != "pix"} for c in cubes]
    report["highlighted"] = [good[k]["center"] for k in range(len(good)) if S["tracks"][k]["H"]]
    report["tracks_start"] = snapshot()
    report["stage"] = "watched"

    # ---- press 1 in the first still interval
    S["grip_hold"] = S["grip"]
    if S["alive"] and button is not None:
        wait_for_window(70, "press1")
        report["press1"] = press(button, "press1")
    report["stage"] = "pressed1"

    # ---- picks, one per still interval that is long enough
    pick_need = 80
    tries = {}
    done = []
    while S["alive"]:
        cand = [k for k in range(len(S["tracks"])) if S["tracks"][k]["H"] and k not in done
                and tries.get(k, 0) < 2]
        if len(cand) == 0:
            break
        if S["tick"] > S["max_steps"] - 180:
            note({"call": "step_budget_stop", "t": S["tick"]})
            break
        wait_for_window(pick_need, "pick")
        if not S["alive"]:
            break
        cand = [k for k in cand if S["tracks"][k]["lost"] == 0] or cand
        best = cand[0]
        bd = None
        for k in cand:
            xy = approx_xy(S["tracks"][k])
            d = (xy[0] - S["ee"][0]) ** 2 + (xy[1] - S["ee"][1]) ** 2
            if bd is None or d < bd:
                bd = d
                best = k
        report["actions"][-1]["tracks"] = snapshot()
        out = pick(best)
        tries[best] = tries.get(best, 0) + 1
        if out == "picked":
            done.append(best)
            pick_need = max(60, min(95, report["actions"][-1]["ticks"] + 5))
        report["pick_outcomes"] = [[a.get("track"), a.get("outcome"), a.get("ticks")] for a in report["actions"]
                                   if a["label"] == "pick"]
        if out == "ended":
            break
    report["stage"] = "picked"

    # ---- final press
    if S["alive"] and button is not None:
        wait_for_window(60, "press2")
        report["press2"] = press(button, "press2")
        if S["alive"]:
            hold_ticks(10, 0.0)
    report["tracks_end"] = snapshot()
    report["stage"] = "done"
except Exception as exc:
    report["errors"].append("fatal: " + str(exc)[:500])

report["ticks"] = S["tick"]
result = report
