# v05 — v04 + physical de-duplication: same-colour cube clusters closer than one container footprint
#       are one cube; container tracks closer than 3 cm are one container; never pick the same container twice
# v04 — v03 + front+wrist depth fusion, static-background voxel rejection (occluding pillar),
#       occlusion-aware track re-acquisition, fused cube detection, hidden-cube fallback label
# v03 — v02 + live hold-and-track: the shuffle continues during execution; track per tick until still, then pick
# v02 — v01 with the whole body guarded by try/except (no uncaught exception can reach the worker)
#       and diagnostic dumps encoded as strings (no ragged nested lists in the recorded result)
# v01 — (v00 resubmitted without `import math`, `**` and sorted(reverse=); v00 start crashed the worker before any artifact)
# v00 — robomme_plus VideoUnmaskSwap: depth-tracked container shuffle, then ordered picks.
# Allowed APIs only. Everything task-dependent (cube colours, container positions, heights,
# yaw) is derived at runtime from the conditioning video and the live front RGB-D camera.
PI = 3.141592653589793


def mradians(d):
    return d * PI / 180.0


def msin(x):
    for rr in range(20):
        if x > PI:
            x = x - 2.0 * PI
        elif x < -PI:
            x = x + 2.0 * PI
    term = x
    total = x
    x2 = x * x
    for n in range(1, 10):
        term = -term * x2 / ((2 * n) * (2 * n + 1))
        total += term
    return total


def mcos(x):
    return msin(x + PI / 2.0)


def msqrt(v):
    if v <= 0:
        return 0.0
    g = v if v > 1.0 else 1.0
    for n in range(40):
        g = 0.5 * (g + v / g)
    return g


CAM = "front"
STRIDE = 2              # pixel stride for per-frame video processing
WIDE_ZLO = 0.03         # wide band used to detect container appearance (metres above table)
WIDE_ZHI = 0.15         # below the parked gripper in the video
TRACK_RADIUS = 0.035    # max association radius per frame (container footprint ~5 cm)
MIN_PTS = 3
TRANSIT_ABOVE_TOP = 0.10
LIFT_ABOVE_TOP = 0.10
MAX_WAIT_TICKS = 700     # hold-and-track cap (native horizon 1370)
STILL_EPS = 0.0015       # per-tick displacement treated as stationary
STILL_TICKS = 12         # consecutive stationary ticks before picking
WRIST = "wrist"
WRIST_STRIDE = 4         # wrist pixels per video frame (dense close-up view)
BG_VOX = 0.01           # static-background voxel size (m)
REACQ_GROW = 0.006       # re-acquisition radius growth per lost frame
REACQ_MAX = 0.12

report = {"version": "v01", "errors": [], "calls": []}
report["version"] = "v05"
report["stage"] = "start"
try:
    context = get_task_context()
    language = context.language
    report["language"] = language
    state0 = get_robot_state()
    base_frame = state0.base_frame
    home_quat = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]
    try:
        meta = robomme_plus.get_task_metadata()
        max_steps_total = int(meta["max_steps"])
    except Exception as exc:
        max_steps_total = 1000
        report["errors"].append("meta: " + str(exc))
    report["max_steps"] = max_steps_total

    # ---------------------------------------------------------------- language -> colour order
    COLOR_WORDS = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "white", "black"]
    low = language.lower()
    mentions = []
    for cw in COLOR_WORDS:
        pos = low.find(cw)
        if pos >= 0:
            mentions.append([pos, cw])
    mentions = sorted(mentions)
    color_order = [m[1] for m in mentions]
    report["color_order"] = color_order


    def hue_name(r, g, b):
        mx = max(r, g, b)
        mn = min(r, g, b)
        if mx - mn < 50:
            return "none"
        d = float(mx - mn)
        if mx == r:
            h = 60.0 * ((g - b) / d)
        elif mx == g:
            h = 60.0 * ((b - r) / d) + 120.0
        else:
            h = 60.0 * ((r - g) / d) + 240.0
        if h < 0:
            h += 360.0
        if h < 15 or h >= 340:
            return "red"
        if h < 45:
            return "orange"
        if h < 70:
            return "yellow"
        if h < 170:
            return "green"
        if h < 200:
            return "cyan"
        if h < 260:
            return "blue"
        return "purple"


    def quat_to_rot(q):
        w = q[0]
        x = q[1]
        y = q[2]
        z = q[3]
        return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


    def to_rows(arr):
        try:
            return arr.tolist()
        except Exception:
            out = []
            for rr in range(int(arr.shape[0])):
                out.append([float(arr[rr, cc]) for cc in range(int(arr.shape[1]))])
            return out


    def build_samples(cam, stride):
        # Per sampled pixel the base-frame ray scaled for z-depth: point = cam_pos + depth * dir.
        K = to_rows(cam.intrinsics)
        fx = float(K[0][0])
        fy = float(K[1][1])
        cxp = float(K[0][2])
        cyp = float(K[1][2])
        pose = cam.camera_pose
        Rm = quat_to_rot([float(v) for v in pose.quaternion_wxyz])
        cpos = [float(v) for v in pose.position]
        h = int(cam.rgb.shape[0])
        w = int(cam.rgb.shape[1])
        rows = []
        for r in range(0, h, stride):
            yc = (r - cyp) / fy
            line = []
            for c in range(0, w, stride):
                xc = (c - cxp) / fx
                dx = Rm[0][0] * xc + Rm[0][1] * yc + Rm[0][2]
                dy = Rm[1][0] * xc + Rm[1][1] * yc + Rm[1][2]
                dz = Rm[2][0] * xc + Rm[2][1] * yc + Rm[2][2]
                if dz >= -1e-6:
                    continue  # ray does not hit the table plane
                s = -cpos[2] / dz
                tx = cpos[0] + s * dx
                ty = cpos[1] + s * dy
                if tx < 0.1 or tx > 0.9 or ty < -0.5 or ty > 0.5:
                    continue
                line.append((c, dx, dy, dz))
            if len(line) > 0:
                rows.append((r, line))
        return {"rows": rows, "pos": cpos}


    def band_points(depth_rows, samples, zlo, zhi):
        cpx = samples["pos"][0]
        cpy = samples["pos"][1]
        cpz = samples["pos"][2]
        pts = []
        for entry in samples["rows"]:
            drow = depth_rows[entry[0]]
            for s in entry[1]:
                dd = drow[s[0]]
                z = cpz + dd * s[3]
                if z > zlo and z < zhi:
                    pts.append((cpx + dd * s[1], cpy + dd * s[2], z))
        return pts


    def track_step(centers, prev, pts, radius, iters, min_pts):
        k = len(centers)
        cur = []
        for j in range(k):
            vx = centers[j][0] - prev[j][0]
            vy = centers[j][1] - prev[j][1]
            cur.append([centers[j][0] + vx, centers[j][1] + vy])
        r2 = radius * radius
        counts = [0] * k
        for it in range(iters):
            sx = [0.0] * k
            sy = [0.0] * k
            counts = [0] * k
            for p in pts:
                px = p[0]
                py = p[1]
                best = -1
                bd = r2
                for j in range(k):
                    ddx = px - cur[j][0]
                    ddy = py - cur[j][1]
                    dd = ddx * ddx + ddy * ddy
                    if dd < bd:
                        bd = dd
                        best = j
                if best >= 0:
                    sx[best] += px
                    sy[best] += py
                    counts[best] += 1
            for j in range(k):
                if counts[j] >= min_pts:
                    cur[j] = [sx[j] / counts[j], sy[j] / counts[j]]
        for j in range(k):
            if counts[j] < min_pts:
                cur[j] = [centers[j][0], centers[j][1]]
        return cur, counts


    def greedy_clusters(pts, radius, min_size):
        clusters = []
        for p in pts:
            placed = False
            for cl in clusters:
                ddx = p[0] - cl["x"]
                ddy = p[1] - cl["y"]
                if ddx * ddx + ddy * ddy < radius * radius:
                    cl["pts"].append(p)
                    n = len(cl["pts"])
                    cl["x"] += (p[0] - cl["x"]) / n
                    cl["y"] += (p[1] - cl["y"]) / n
                    placed = True
                    break
            if not placed:
                clusters.append({"x": p[0], "y": p[1], "pts": [p]})
        return [cl for cl in clusters if len(cl["pts"]) >= min_size]


    def track_step2(centers, prev, lost, pts, radius, min_pts, grow, rmax):
        # pass 1: visible tracks, constant-velocity prediction + Voronoi mean
        k = len(centers)
        cur = []
        for j in range(k):
            if lost[j] == 0:
                cur.append([2 * centers[j][0] - prev[j][0], 2 * centers[j][1] - prev[j][1]])
            else:
                cur.append([centers[j][0], centers[j][1]])
        act = [lost[j] == 0 for j in range(k)]
        r2 = radius * radius
        counts = [0] * k
        owner = [-1] * len(pts)
        for it in range(3):
            sx = [0.0] * k
            sy = [0.0] * k
            counts = [0] * k
            for i in range(len(pts)):
                p = pts[i]
                best = -1
                bd = r2
                for j in range(k):
                    ddx = p[0] - cur[j][0]
                    ddy = p[1] - cur[j][1]
                    dd = ddx * ddx + ddy * ddy
                    if dd < bd:
                        bd = dd
                        best = j
                owner[i] = best
                if best >= 0:
                    sx[best] += p[0]
                    sy[best] += p[1]
                    counts[best] += 1
            for j in range(k):
                if counts[j] >= min_pts:
                    cur[j] = [sx[j] / counts[j], sy[j] / counts[j]]
        new = []
        new_lost = []
        for j in range(k):
            if counts[j] >= min_pts:
                new.append(cur[j])
                new_lost.append(0)
            else:
                new.append([centers[j][0], centers[j][1]])
                new_lost.append(lost[j] + 1)
        # pass 2: unclaimed points -> clusters -> recapture by lost tracks
        left = []
        for i in range(len(pts)):
            j = owner[i]
            if j < 0 or counts[j] < min_pts:
                left.append(pts[i])
        if len(left) >= min_pts:
            cls = greedy_clusters(left, 0.03, max(min_pts, 6))
            used = [False] * len(cls)
            order = sorted([[new_lost[j], j] for j in range(k) if new_lost[j] > 0])
            for o in order:
                j = o[1]
                rr = radius + grow * lost[j]
                if rr > rmax:
                    rr = rmax
                best = -1
                bd = rr * rr
                for c in range(len(cls)):
                    if used[c]:
                        continue
                    ddx = cls[c]["x"] - new[j][0]
                    ddy = cls[c]["y"] - new[j][1]
                    dd = ddx * ddx + ddy * ddy
                    if dd < bd:
                        bd = dd
                        best = c
                if best >= 0:
                    used[best] = True
                    new[j] = [cls[best]["x"], cls[best]["y"]]
                    new_lost[j] = 0
                    counts[j] = len(cls[best]["pts"])
        return new, new_lost, counts


    def quantile(vals, q):
        o = sorted(vals)
        if len(o) == 0:
            return None
        return o[int((len(o) - 1) * q)]


    # ---------------------------------------------------------------- conditioning video
    report["stage"] = "video"
    video = robomme_plus.get_conditioning_video()
    nv = len(video)
    report["video_len"] = nv
    if nv < 3:
        raise ValueError("no conditioning video")
    samples = build_samples(video[0].cameras[CAM], STRIDE)
    wide = []
    counts_wide = []
    for i in range(nv):
        pts = band_points(to_rows(video[i].cameras[CAM].depth_m), samples, WIDE_ZLO, WIDE_ZHI)
        wide.append(pts)
        counts_wide.append(len(pts))
    # wrist view of the parked arm (top-down, sees behind front-view occluders)
    wrist_ok = True
    wide_w = []
    try:
        wsamples = build_samples(video[0].cameras[WRIST], WRIST_STRIDE)
        for i in range(nv):
            wide_w.append(band_points(to_rows(video[i].cameras[WRIST].depth_m), wsamples, WIDE_ZLO, WIDE_ZHI))
    except Exception as exc:
        wrist_ok = False
        wide_w = [[] for i in range(nv)]
        report["errors"].append("wrist video: " + str(exc))
    report["wrist_ok"] = wrist_ok
    report["counts_wrist"] = [len(w) for w in wide_w]
    report["stage"] = "after_wide"
    report["counts_wide"] = counts_wide
    baseline = counts_wide[0]
    first = -1
    for i in range(1, nv):
        if counts_wide[i] > baseline + 40:
            first = i
            break
    report["first_container_frame"] = first
    if first < 1:
        raise ValueError("container appearance not found")
    zs_first = [p[2] for p in wide[first]]
    top_z = quantile(zs_first, 0.9)
    zlo = 0.6 * top_z
    zhi = top_z + 0.03
    report["top_z"] = top_z
    report["band"] = [zlo, zhi]

    # static background (e.g. an occluding pillar): band points of the pre-cover frame 0,
    # dilated by one voxel; any later band point inside is rejected (video and live)
    def vkey(x, y, z):
        return str(int(x / BG_VOX + 1000.0)) + "," + str(int(y / BG_VOX + 1000.0)) + "," + str(int(z / BG_VOX + 1000.0))

    bg = {}
    for p in wide[0] + wide_w[0]:
        if p[2] > zlo and p[2] < zhi:
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    for oz in (-1, 0, 1):
                        bg[vkey(p[0] + ox * BG_VOX, p[1] + oy * BG_VOX, p[2] + oz * BG_VOX)] = 1
    report["bg_voxels"] = len(bg)

    # cubes in the frame before the containers appear (front stride 1 + wrist stride 2, colour from RGB)
    cube_pts = []
    low_pts = []
    cam_list = [[CAM, 1]]
    if wrist_ok:
        cam_list.append([WRIST, 2])
    for cs in cam_list:
        ccam = video[first - 1].cameras[cs[0]]
        cube_samples = build_samples(ccam, cs[1])
        cdepth = to_rows(ccam.depth_m)
        crgb = to_rows(ccam.rgb)
        cpx = cube_samples["pos"][0]
        cpy = cube_samples["pos"][1]
        cpz = cube_samples["pos"][2]
        for entry in cube_samples["rows"]:
            r = entry[0]
            drow = cdepth[r]
            crow = crgb[r]
            for s in entry[1]:
                dd = drow[s[0]]
                z = cpz + dd * s[3]
                if z > 0.004 and z < zlo:
                    px = crow[s[0]]
                    sat = max(px[0], px[1], px[2]) - min(px[0], px[1], px[2])
                    if sat >= 70:
                        cube_pts.append((cpx + dd * s[1], cpy + dd * s[2], z, px[0], px[1], px[2]))
                    elif z > 0.008 and cs[0] == CAM:
                        low_pts.append((cpx + dd * s[1], cpy + dd * s[2], z))
    report["stage"] = "cubes"
    cube_clusters = greedy_clusters(cube_pts, 0.025, 8)
    merged = []
    order_list = sorted([[len(cube_clusters[i]["pts"]), i] for i in range(len(cube_clusters))])
    order_list.reverse()
    for cl in order_list:
        cand = cube_clusters[cl[1]]
        dup = False
        for mcl in merged:
            ddx = cand["x"] - mcl["x"]
            ddy = cand["y"] - mcl["y"]
            if ddx * ddx + ddy * ddy < 0.03 * 0.03:
                dup = True
        if not dup:
            merged.append(cand)
    cube_clusters = merged
    cubes = []
    for cl in cube_clusters:
        rr = quantile([p[3] for p in cl["pts"]], 0.5)
        gg = quantile([p[4] for p in cl["pts"]], 0.5)
        bb = quantile([p[5] for p in cl["pts"]], 0.5)
        cubes.append({"x": cl["x"], "y": cl["y"], "rgb": [rr, gg, bb], "color": hue_name(rr, gg, bb),
                      "n": len(cl["pts"])})
    # a cube seen partly from two views can split into two same-colour clusters; two cubes under
    # different containers are at least one container footprint apart, so merge closer pairs
    FOOTPRINT = 0.045
    dedup = []
    n_merged = 0
    for cb in cubes:
        hit = -1
        for m in range(len(dedup)):
            ddx = cb["x"] - dedup[m]["x"]
            ddy = cb["y"] - dedup[m]["y"]
            if dedup[m]["color"] == cb["color"] and ddx * ddx + ddy * ddy < FOOTPRINT * FOOTPRINT:
                hit = m
        if hit >= 0:
            tot = dedup[hit]["n"] + cb["n"]
            dedup[hit]["x"] = (dedup[hit]["x"] * dedup[hit]["n"] + cb["x"] * cb["n"]) / tot
            dedup[hit]["y"] = (dedup[hit]["y"] * dedup[hit]["n"] + cb["y"] * cb["n"]) / tot
            dedup[hit]["n"] = tot
            n_merged += 1
        else:
            dedup.append(cb)
    cubes = dedup
    report["cubes_merged"] = n_merged
    report["cubes"] = cubes

    # initialise container tracks at the first container frame
    def band_filter(pts):
        out = []
        for p in pts:
            if p[2] > zlo and p[2] < zhi:
                if vkey(p[0], p[1], p[2]) not in bg:
                    out.append(p)
        return out

    pts0 = band_filter(wide[first] + wide_w[first])
    centers = [[c["x"], c["y"]] for c in cubes]
    centers, cnt0 = track_step(centers, centers, pts0, 0.05, 6, MIN_PTS)
    labels = []
    kept = []
    kcnt = []
    for j in range(len(centers)):
        if cnt0[j] >= MIN_PTS:
            kept.append(centers[j])
            labels.append(cubes[j]["color"])
            kcnt.append(cnt0[j])
    centers = kept
    med = 20
    if len(kcnt) > 0:
        med = sorted(kcnt)[int(len(kcnt) / 2)]
    report["median_container_pts"] = med
    # containers not over a detected cube
    left = []
    for p in pts0:
        near = False
        for c in centers:
            ddx = p[0] - c[0]
            ddy = p[1] - c[1]
            if ddx * ddx + ddy * ddy < 0.045 * 0.045:
                near = True
                break
        if not near:
            left.append(p)
    min_extra = int(0.4 * med)
    if min_extra < 8:
        min_extra = 8
    for cl in greedy_clusters(left, 0.035, min_extra):
        centers.append([cl["x"], cl["y"]])
        labels.append("unknown")
    centers, cnt0 = track_step(centers, centers, pts0, 0.05, 4, MIN_PTS)
    # two tracks on one container (closer than 3 cm) are one container: keep the labelled / better supported one
    keep_c = []
    keep_l = []
    keep_n = []
    n_dup_tracks = 0
    for j in range(len(centers)):
        hit = -1
        for m in range(len(keep_c)):
            ddx = centers[j][0] - keep_c[m][0]
            ddy = centers[j][1] - keep_c[m][1]
            if ddx * ddx + ddy * ddy < 0.03 * 0.03:
                hit = m
        if hit < 0:
            keep_c.append(centers[j])
            keep_l.append(labels[j])
            keep_n.append(cnt0[j])
        else:
            n_dup_tracks += 1
            if keep_l[hit] == "unknown" and labels[j] != "unknown":
                keep_c[hit] = centers[j]
                keep_l[hit] = labels[j]
                keep_n[hit] = cnt0[j]
    centers = keep_c
    labels = keep_l
    cnt0 = keep_n
    report["dup_tracks_removed"] = n_dup_tracks
    # unknown containers: is there an unsaturated (white/black) cube under it, or nothing seen (hidden)?
    hidden = []
    for j in range(len(centers)):
        if labels[j] == "unknown":
            n_low = 0
            for p in low_pts:
                ddx = p[0] - centers[j][0]
                ddy = p[1] - centers[j][1]
                if ddx * ddx + ddy * ddy < 0.03 * 0.03:
                    n_low += 1
            if n_low < 8:
                hidden.append(j)
    report["hidden_unknown"] = hidden
    report["labels"] = labels
    report["init_centers"] = [[round(c[0], 4), round(c[1], 4)] for c in centers]

    report["stage"] = "track"
    prev = [c[:] for c in centers]
    lost = [0] * len(centers)
    track_hist = [[[round(c[0], 4), round(c[1], 4)] for c in centers]]
    lost_frames = 0
    max_lost = [0] * len(centers)
    for i in range(first + 1, nv):
        pts = band_filter(wide[i] + wide_w[i])
        new, lost, cnt = track_step2(centers, prev, lost, pts, TRACK_RADIUS, MIN_PTS, REACQ_GROW, REACQ_MAX)
        if max(lost) > 0:
            lost_frames += 1
        for j in range(len(lost)):
            if lost[j] > max_lost[j]:
                max_lost[j] = lost[j]
        prev = centers
        centers = new
        track_hist.append([[round(c[0], 4), round(c[1], 4)] for c in centers])
    report["lost_frames"] = lost_frames
    report["max_lost"] = max_lost
    report["final_video_centers"] = track_hist[-1]
    # compact dump of band points for offline replay (mm), encoded as strings so the
    # recorded result never contains ragged nested arrays
    dump = []
    for i in range(nv):
        dump.append(";".join([str(int(p[0] * 1000)) + "," + str(int(p[1] * 1000)) + "," + str(int(p[2] * 1000)) for p in band_filter(wide[i] + wide_w[i])]))
    report["dump_wide_mm"] = dump
    report["dump_cube_pts"] = ";".join([str(int(p[0] * 1000)) + "," + str(int(p[1] * 1000)) + "," + str(int(p[2] * 1000)) + "," + str(p[3]) + "," + str(p[4]) + "," + str(p[5]) for p in cube_pts])
    report["track_hist"] = [";".join([str(c[0]) + "," + str(c[1]) for c in h]) for h in track_hist]

    # ---------------------------------------------------------------- live scene
    def obs_points(obs, fstride, wstride):
        out = []
        cam = obs.cameras[CAM]
        out = out + band_points(to_rows(cam.depth_m), build_samples(cam, fstride), zlo, zhi)
        if wrist_ok:
            try:
                wc = obs.cameras[WRIST]
                out = out + band_points(to_rows(wc.depth_m), build_samples(wc, wstride), zlo, zhi)
            except Exception as exc:
                report["errors"].append("live wrist: " + str(exc))
        return band_filter(out)


    def live_points(stride, lo, hi):
        return obs_points(get_observation(), stride, 2 * stride)


    def refine_live(centers_in, stride):
        pts = live_points(stride, zlo, zhi)
        cur, cnt = track_step(centers_in, centers_in, pts, 0.045, 4, MIN_PTS)
        groups = [[] for c in cur]
        for p in pts:
            best = -1
            bd = 0.04 * 0.04
            for j in range(len(cur)):
                ddx = p[0] - cur[j][0]
                ddy = p[1] - cur[j][1]
                dd = ddx * ddx + ddy * ddy
                if dd < bd:
                    bd = dd
                    best = j
            if best >= 0:
                groups[best].append(p)
        return cur, cnt, groups


    def square_yaw(points):
        # min-area bounding square orientation (degrees in [0, 90)) from top-face points
        best_a = 0
        best_area = None
        for a in range(0, 90, 5):
            ca = mcos(mradians(a))
            sa = msin(mradians(a))
            us = [p[0] * ca + p[1] * sa for p in points]
            vs = [-p[0] * sa + p[1] * ca for p in points]
            area = (quantile(us, 0.95) - quantile(us, 0.05)) * (quantile(vs, 0.95) - quantile(vs, 0.05))
            if best_area is None or area < best_area:
                best_area = area
                best_a = a
        return best_a


    def yaw_quat(theta_deg):
        th = mradians(theta_deg)
        cw = mcos(th / 2.0)
        sz = msin(th / 2.0)
        hw = home_quat[0]
        hx = home_quat[1]
        hy = home_quat[2]
        hz = home_quat[3]
        return [cw * hw - sz * hz, cw * hx - sz * hy, cw * hy + sz * hx, cw * hz + sz * hw]


    steps_used = [0]
    terminal = [False]


    def log_call(name, res):
        entry = {"call": name}
        try:
            entry["ok"] = bool(res.ok)
            entry["status"] = str(res.status)
            entry["steps"] = int(res.steps_executed)
            steps_used[0] += int(res.steps_executed)
        except Exception as exc:
            entry["err"] = str(exc)
        try:
            if res.terminated or res.truncated:
                terminal[0] = True
                entry["terminal"] = [bool(res.terminated), bool(res.truncated)]
        except Exception:
            pass
        report["calls"].append(entry)
        return res


    def move(x, y, z, quat, name, tol):
        res = move_to_pose(Pose((x, y, z), quat, base_frame), tolerance=tol, max_steps=150)
        return log_call(name, res)


    report["stage"] = "live_wait"
    # The shuffle CONTINUES during execution (v02 evidence): hold the arm still, track every
    # tick from the step observation, and only pick once all containers have stopped.
    hold_state = get_robot_state()
    hold_joints = [float(v) for v in hold_state.joint_positions["primary"]]
    obs_l0 = get_observation()
    live_samples = build_samples(obs_l0.cameras[CAM], STRIDE)
    live_wsamples = None
    if wrist_ok:
        try:
            live_wsamples = build_samples(obs_l0.cameras[WRIST], WRIST_STRIDE)
        except Exception as exc:
            report["errors"].append("live wrist samples: " + str(exc))
    lost = [0] * len(centers)
    still = 0
    wait_ticks = 0
    live_hist = []
    disp_hist = []
    for tick in range(MAX_WAIT_TICKS):
        try:
            cmd = ArmCommand("joint_position", hold_joints, gripper_position=1.0, embodiment=hold_state.embodiment)
        except Exception:
            cmd = ArmCommand("joint_position", hold_joints, gripper_position=1.0)
        sres = step(RobotAction(arms={"primary": cmd}))
        wait_ticks += 1
        steps_used[0] += 1
        try:
            if sres.terminated or sres.truncated:
                terminal[0] = True
                break
        except Exception:
            pass
        try:
            obs = sres.observation
            dcam = obs.cameras[CAM]
        except Exception:
            obs = get_observation()
            dcam = obs.cameras[CAM]
        pts = band_points(to_rows(dcam.depth_m), live_samples, zlo, zhi)
        if live_wsamples is not None:
            try:
                pts = pts + band_points(to_rows(obs.cameras[WRIST].depth_m), live_wsamples, zlo, zhi)
            except Exception:
                pass
        pts = band_filter(pts)
        new, lost, cnt = track_step2(centers, prev, lost, pts, TRACK_RADIUS, MIN_PTS, REACQ_GROW, REACQ_MAX)
        dmax = 0.0
        for j in range(len(new)):
            ddx = new[j][0] - centers[j][0]
            ddy = new[j][1] - centers[j][1]
            dd = msqrt(ddx * ddx + ddy * ddy)
            if dd > dmax:
                dmax = dd
        prev = centers
        centers = new
        disp_hist.append(int(dmax * 10000))
        live_hist.append(";".join([str(round(c[0], 4)) + "," + str(round(c[1], 4)) for c in centers]))
        if dmax < STILL_EPS:
            still += 1
        else:
            still = 0
        if still >= STILL_TICKS:
            break
    report["wait_ticks"] = wait_ticks
    report["still_reached"] = still >= STILL_TICKS
    report["live_disp_1e4"] = disp_hist
    report["live_track_hist"] = live_hist

    report["stage"] = "live"
    live_c, live_cnt, groups = refine_live(centers, 1)
    report["live_centers"] = [[round(c[0], 4), round(c[1], 4)] for c in live_c]
    report["live_counts"] = live_cnt
    shift = []
    for j in range(len(centers)):
        ddx = live_c[j][0] - centers[j][0]
        ddy = live_c[j][1] - centers[j][1]
        shift.append(round(msqrt(ddx * ddx + ddy * ddy), 4))
    report["video_to_live_shift"] = shift

    plan = []
    fallback = []
    for cw in color_order:
        found = 0
        for j in range(len(labels)):
            if labels[j] == cw:
                plan.append(j)
                found += 1
        if found == 0 and len(hidden) > 0:
            # the requested cube was never visible: it is under a container that no visible cube explains
            j = hidden[0]
            hidden = hidden[1:]
            labels[j] = cw + "?"
            plan.append(j)
            fallback.append(cw)
    report["fallback_colors"] = fallback
    report["plan"] = plan
    report["plan_colors"] = [labels[j] for j in plan]
    if len(plan) == 0:
        raise ValueError("no container matches the requested colours")

    report["stage"] = "motion"
    positions = [c[:] for c in live_c]
    picked = []
    skipped = []
    for j in plan:
        if terminal[0]:
            break
        again = False
        for pk in picked:
            ddx = positions[j][0] - pk["grasp_xy"][0]
            ddy = positions[j][1] - pk["grasp_xy"][1]
            if ddx * ddx + ddy * ddy < 0.025 * 0.025:
                again = True
        if again:
            skipped.append(j)
            continue
        # re-observe: refresh every container position from the current live view
        try:
            cur, cnt, groups = refine_live(positions, 1)
            positions = cur
        except Exception as exc:
            report["errors"].append("refine: " + str(exc))
            groups = [[] for c in positions]
        tops = [p for p in groups[j] if p[2] > top_z - 0.008]
        if len(tops) >= 8:
            gx = quantile([p[0] for p in tops], 0.5)
            gy = quantile([p[1] for p in tops], 0.5)
            alpha = square_yaw(tops)
        else:
            gx = positions[j][0]
            gy = positions[j][1]
            alpha = 0
        # choose the jaw axis (two square-aligned options) with more clearance from neighbours
        best_theta = None
        best_clear = None
        for theta in (alpha, alpha - 90):
            th = mradians(theta)
            ax = -msin(th)
            ay = mcos(th)
            clear = 1.0
            for k in range(len(positions)):
                if k == j:
                    continue
                for sgn in (1.0, -1.0):
                    fx = gx + sgn * 0.045 * ax
                    fy = gy + sgn * 0.045 * ay
                    ddx = fx - positions[k][0]
                    ddy = fy - positions[k][1]
                    dd = msqrt(ddx * ddx + ddy * ddy)
                    if dd < clear:
                        clear = dd
            if best_clear is None or clear > best_clear:
                best_clear = clear
                best_theta = theta
        quat = yaw_quat(best_theta)
        grasp_z = 0.6 * top_z
        transit_z = top_z + TRANSIT_ABOVE_TOP
        info = {"index": j, "color": labels[j], "grasp_xy": [round(gx, 4), round(gy, 4)],
                "alpha": alpha, "theta": best_theta, "clear": round(best_clear, 4), "n_top": len(tops),
                "steps_before": steps_used[0]}
        log_call("preopen_" + str(j), open_gripper())
        move(gx, gy, transit_z, quat, "above_" + str(j), 0.015)
        if terminal[0]:
            break
        move(gx, gy, grasp_z, quat, "descend_" + str(j), 0.01)
        if terminal[0]:
            break
        log_call("close_" + str(j), close_gripper())
        if terminal[0]:
            break
        try:
            info["grip_after_close"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
        except Exception:
            pass
        move(gx, gy, top_z + LIFT_ABOVE_TOP, quat, "lift_" + str(j), 0.015)
        if terminal[0]:
            picked.append(info)
            break
        try:
            info["grip_after_lift"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
        except Exception:
            pass
        move(gx, gy, grasp_z + 0.01, quat, "lower_" + str(j), 0.015)
        if terminal[0]:
            picked.append(info)
            break
        log_call("open_" + str(j), open_gripper())
        if terminal[0]:
            picked.append(info)
            break
        move(gx, gy, transit_z, quat, "retreat_" + str(j), 0.02)
        info["steps_after"] = steps_used[0]
        picked.append(info)

    report["picked"] = picked
    report["skipped_same_container"] = skipped
    report["steps_used"] = steps_used[0]
    report["terminal"] = terminal[0]
except Exception as exc:
    report["fatal"] = str(exc)
result = report
