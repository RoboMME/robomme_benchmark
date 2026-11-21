# VideoRepick v00 -- track the object picked in the conditioning video through the shuffle,
# then pick it up and place it down twice and press the button.
# Every episode-dependent quantity is derived at runtime from the conditioning video
# (RGB-D + calibration + recorded robot state), live observations and public perception.
import math

report = {"version": "v02"}
try:
    goal = get_task_context().language
    report["goal"] = goal
    meta = robomme_plus.get_task_metadata()
    try:
        max_steps_total = int(meta["max_steps"])
    except Exception:
        max_steps_total = -1
    report["max_steps"] = max_steps_total
    try:
        report["video_sampling"] = str(meta["conditioning_video_sampling"])
    except Exception:
        report["video_sampling"] = None

    state0 = get_robot_state()
    base_frame = state0.base_frame
    home_quat = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]


    # ---------------------------------------------------------------- geometry helpers
    def qmat(q):
        w = float(q[0]); x = float(q[1]); y = float(q[2]); z = float(q[3])
        n = math.sqrt(w * w + x * x + y * y + z * z)
        w = w / n; x = x / n; y = y / n; z = z / n
        return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


    def cam_calib(cam):
        K = cam.intrinsics
        fx = float(K[0][0]); fy = float(K[1][1]); cx = float(K[0][2]); cy = float(K[1][2])
        pose = cam.camera_pose
        R = qmat(pose.quaternion_wxyz)
        t = [float(v) for v in pose.position]
        return (fx, fy, cx, cy, R, t)


    def body_from_cv(X, Y, Z, conv):
        if conv[0] == "c":
            return (X, Y, Z)
        if conv[0] == "g":
            return (X, -Y, -Z)
        return (Z, -X, -Y)  # "ros": x forward, y left, z up


    def cv_from_body(a, b, c, conv):
        if conv[0] == "c":
            return (a, b, c)
        if conv[0] == "g":
            return (a, -b, -c)
        return (-b, -c, a)


    def unproject(calib, u, v, d, conv):
        fx, fy, cx, cy, R, t = calib
        xn = (u - cx) / fx
        yn = (v - cy) / fy
        if conv[-1] == "r":
            d = d / math.sqrt(xn * xn + yn * yn + 1.0)
        X = xn * d
        Y = yn * d
        a, b, c = body_from_cv(X, Y, d, conv)
        return (R[0][0] * a + R[0][1] * b + R[0][2] * c + t[0],
                R[1][0] * a + R[1][1] * b + R[1][2] * c + t[1],
                R[2][0] * a + R[2][1] * b + R[2][2] * c + t[2])


    def project(calib, p, conv):
        fx, fy, cx, cy, R, t = calib
        dx = p[0] - t[0]; dy = p[1] - t[1]; dz = p[2] - t[2]
        a = R[0][0] * dx + R[1][0] * dy + R[2][0] * dz
        b = R[0][1] * dx + R[1][1] * dy + R[2][1] * dz
        c = R[0][2] * dx + R[1][2] * dy + R[2][2] * dz
        X, Y, Z = cv_from_body(a, b, c, conv)
        if Z <= 1e-6:
            return None
        return (fx * X / Z + cx, fy * Y / Z + cy)


    def as_rows(arr):
        try:
            return arr.tolist()
        except Exception:
            return [list(row) for row in arr]


    def quant(values, frac):
        s = sorted(values)
        if len(s) == 0:
            return None
        return s[int((len(s) - 1) * frac)]


    def hue_sat(r, g, b):
        mx = max(r, g, b); mn = min(r, g, b)
        sat = mx - mn
        if sat == 0:
            return (0.0, 0, mx)
        if mx == r:
            h = 60.0 * ((g - b) / sat)
        elif mx == g:
            h = 60.0 * ((b - r) / sat) + 120.0
        else:
            h = 60.0 * ((r - g) / sat) + 240.0
        if h < 0:
            h = h + 360.0
        return (h, sat, mx)


    # ---------------------------------------------------------------- camera convention
    live = get_observation()
    cam_names = [name for name in live.cameras]
    report["cameras"] = cam_names
    front_name = "front" if "front" in cam_names else cam_names[0]
    wrist_name = "wrist" if "wrist" in cam_names else cam_names[-1]


    def table_stats(cam, conv):
        calib = cam_calib(cam)
        drows = as_rows(cam.depth_m)
        H = len(drows); W = len(drows[0])
        zs = []
        for v in range(4, H, 12):
            for u in range(4, W, 12):
                d = float(drows[v][u])
                if d > 0.05 and d < 5.0:
                    zs.append(unproject(calib, u, v, d, conv)[2])
        if len(zs) < 10:
            return None
        q25 = quant(zs, 0.25); q50 = quant(zs, 0.5); q75 = quant(zs, 0.75)
        return (q50, q75 - q25, calib[5][2])


    conv_scores = {}
    best_conv = "cv"
    best_score = None
    for conv in ("cv", "gl", "ros", "cv_r", "gl_r", "ros_r"):
        fs = table_stats(live.cameras[front_name], conv)
        ws = table_stats(live.cameras[wrist_name], conv)
        if fs is None:
            continue
        score = fs[1]
        if fs[0] > fs[2] - 0.05:          # table must lie below the camera
            score = score + 10.0
        if ws is not None:
            score = score + 0.5 * abs(ws[0] - fs[0])
        conv_scores[conv] = [round(fs[0], 4), round(fs[1], 4), round(ws[0], 4) if ws else None, round(score, 4)]
        if best_score is None or score < best_score:
            best_score = score
            best_conv = conv
    CONV = best_conv
    report["conv_scores"] = conv_scores
    report["conv"] = CONV
    table_z = table_stats(live.cameras[front_name], CONV)[0]
    report["table_z"] = round(table_z, 4)

    # ---------------------------------------------------------------- button (live)
    # v01: "button" top-1 was a tall occluder on episode 2; keep only candidates whose projected
    # geometry is button-like (small footprint, low top) and not strongly coloured.
    def find_button():
        tried = []
        for cname in (front_name, wrist_name):
            try:
                seg = segment_text(cname, "button")
            except Exception as exc:
                tried.append([cname, "seg_error", str(exc)])
                continue
            if not seg.ok:
                tried.append([cname, "seg_not_ok"])
                continue
            best = None
            for cand in seg.segmentations[:8]:
                pc = None
                try:
                    pc = mask_to_point_cloud(cand, cname, target_frame=base_frame)
                except Exception:
                    try:
                        pc = mask_to_point_cloud(cand.mask, cname, target_frame=base_frame)
                    except Exception as exc:
                        tried.append([cname, "pc_error", str(exc)[:80]])
                        pc = None
                if pc is None:
                    continue
                pts = pc.points
                if len(pts) < 8:
                    tried.append([cname, "few_points", len(pts)])
                    continue
                xs = [float(q[0]) for q in pts]; ys = [float(q[1]) for q in pts]; zs = [float(q[2]) for q in pts]
                ext = max(quant(xs, 0.95) - quant(xs, 0.05), quant(ys, 0.95) - quant(ys, 0.05))
                top = quant(zs, 0.9)
                sat = None
                try:
                    cols = pc.colors
                    if cols is not None and len(cols) > 0:
                        ssum = 0.0
                        for c in cols[: min(len(cols), 400)]:
                            cr = float(c[0]); cg = float(c[1]); cb = float(c[2])
                            if max(cr, cg, cb) <= 1.0:
                                cr = cr * 255; cg = cg * 255; cb = cb * 255
                            ssum += max(cr, cg, cb) - min(cr, cg, cb)
                        sat = ssum / min(len(cols), 400)
                except Exception:
                    sat = None
                info = [cname, round(float(cand.score), 3), round(ext, 3), round(top - table_z, 3), round(sat, 1) if sat is not None else None]
                tried.append(info)
                okgeom = ext <= 0.07 and top > table_z + 0.004 and top < table_z + 0.08
                if sat is not None and sat > 70:
                    okgeom = False
                if okgeom and (best is None or float(cand.score) > best[0]):
                    best = (float(cand.score), (quant(xs, 0.5), quant(ys, 0.5)), top)
            if best is not None:
                return best, tried
        return None, tried


    button_xy = None
    button_top = None
    try:
        bres, btried = find_button()
        report["button_candidates"] = btried
        if bres is not None:
            button_xy = bres[1]
            button_top = bres[2]
    except Exception as exc:
        report["button_error"] = str(exc)
    report["button_xy"] = [round(v, 4) for v in button_xy] if button_xy else None
    report["button_top"] = round(button_top, 4) if button_top is not None else None

    # ---------------------------------------------------------------- conditioning video
    video = robomme_plus.get_conditioning_video()
    nf = len(video)
    report["video_frames"] = nf
    if nf == 0:
        raise ValueError("no conditioning video")

    grip = []
    ee = []
    for fr in video:
        rs = fr.robot_state
        grip.append(float(rs.gripper_positions["primary"]))
        pe = rs.end_effector_poses["primary"]
        ee.append(([float(v) for v in pe.position], [float(v) for v in pe.quaternion_wxyz]))
    report["grip_series"] = [round(grip[i], 3) for i in range(0, nf, max(1, nf // 60))]
    report["ee_z_series"] = [round(ee[i][0][2], 3) for i in range(0, nf, max(1, nf // 60))]

    gmax = max(grip)
    zmin_all = min(e[0][2] for e in ee)
    grasp_idx = None
    for i in range(1, nf):
        if grasp_idx is None and grip[i] < gmax - 0.15 and grip[i - 1] >= gmax - 0.15 and ee[i][0][2] < zmin_all + 0.05:
            grasp_idx = i
    release_idx = None
    if grasp_idx is not None:
        for i in range(grasp_idx, nf):
            if release_idx is None and grip[i] > gmax - 0.05:
                release_idx = i
    if grasp_idx is None:
        report["grasp_fallback"] = "min_z"
        zmin = None
        for i in range(nf // 2):
            if zmin is None or ee[i][0][2] < zmin:
                zmin = ee[i][0][2]
                grasp_idx = i
        release_idx = grasp_idx
    if release_idx is None:
        release_idx = grasp_idx
    grasp_pos = ee[grasp_idx][0]
    grasp_quat = ee[grasp_idx][1]
    release_pos = ee[release_idx][0]
    report["grasp_idx"] = grasp_idx
    report["release_idx"] = release_idx
    report["grasp_ee"] = [round(v, 4) for v in grasp_pos]
    report["release_ee"] = [round(v, 4) for v in release_pos]
    report["grasp_quat"] = [round(v, 4) for v in grasp_quat]

    # ---------------------------------------------------------------- object colour model
    # v01: learn the object hue only from views in which the grasp point is VISIBLE (depth agrees
    # with the projected distance), voting over several pre-grasp frames of both cameras.
    NB = 18


    def hue_hist(rows, stride):
        hist = [0] * NB
        n = 0
        for v in range(0, len(rows), stride):
            row = rows[v]
            for u in range(0, len(row), stride):
                p = row[u]
                h, s, mx = hue_sat(int(p[0]), int(p[1]), int(p[2]))
                n = n + 1
                if s > 22 and mx > 35:
                    hist[int(h // 20) % NB] += 1
        return hist, n


    def cam_depth_of(calib, P):
        fx, fy, cx, cy, R, t = calib
        dx = P[0] - t[0]; dy = P[1] - t[1]; dz = P[2] - t[2]
        a = R[0][0] * dx + R[1][0] * dy + R[2][0] * dz
        b = R[0][1] * dx + R[1][1] * dy + R[2][1] * dz
        c = R[0][2] * dx + R[1][2] * dy + R[2][2] * dz
        X, Y, Z = cv_from_body(a, b, c, CONV)
        if CONV[-1] == "r":
            return math.sqrt(X * X + Y * Y + Z * Z), Z
        return Z, Z


    f0 = video[0].cameras[front_name]
    rows0 = as_rows(f0.rgb)
    glob, total = hue_hist(rows0, 3)
    P_obj = (grasp_pos[0], grasp_pos[1], table_z + 0.6 * max(0.01, min(0.06, grasp_pos[2] - table_z)))
    votes = [0.0] * NB
    colour_views = []
    cand = []
    stepf = max(1, grasp_idx // 10)
    for i in range(0, max(1, grasp_idx), stepf):
        cand.append((i, front_name))
        cand.append((i, wrist_name))
    for ci in cand:
        i = ci[0]; cname = ci[1]
        cam = video[i].cameras[cname]
        calib = cam_calib(cam)
        pix = project(calib, P_obj, CONV)
        if pix is None:
            continue
        pu = int(round(pix[0])); pv = int(round(pix[1]))
        try:
            dmap = cam.depth_m
            Hh = len(dmap); Ww = len(dmap[0])
        except Exception:
            continue
        if pu < 0 or pv < 0 or pu >= Ww or pv >= Hh:
            continue
        exp_d, Zc = cam_depth_of(calib, P_obj)
        meas = float(dmap[pv][pu])
        if meas < exp_d - 0.05 or meas > exp_d + 0.02:   # v02: visible surface may lie up to 5 cm nearer
            colour_views.append([i, cname, "occluded", round(meas, 3), round(exp_d, 3)])
            continue
        half = int(calib[0] * 0.012 / max(Zc, 0.05))
        half = max(3, min(25, half))
        rows = as_rows(cam.rgb)
        gh, gn = hue_hist(rows, 4)
        win = [0] * NB
        npx = 0
        for v in range(max(0, pv - half), min(Hh, pv + half + 1)):
            for u in range(max(0, pu - half), min(Ww, pu + half + 1)):
                p = rows[v][u]
                npx = npx + 1
                h, s, mx = hue_sat(int(p[0]), int(p[1]), int(p[2]))
                if s > 22 and mx > 35:
                    win[int(h // 20) % NB] += 1
        bk = None
        be = 0.0
        for k in range(NB):
            if win[k] >= max(4, int(0.1 * npx)):
                enr = win[k] / (gh[k] / float(gn) * npx + 1.0)
                if enr > be:
                    be = enr
                    bk = k
        colour_views.append([i, cname, bk, round(be, 2), half])
        if bk is not None and be > 1.5:
            votes[bk] += min(be, 20.0)
    report["colour_views"] = colour_views
    report["colour_votes"] = [round(v, 1) for v in votes]
    obj_bin = None
    best_enr = 0.0
    for k in range(NB):
        if votes[k] > best_enr:
            best_enr = votes[k]
            obj_bin = k
    bg_bins = set([k for k in range(NB) if glob[k] > 0.03 * total and k != obj_bin])
    if obj_bin is None:
        for k in range(NB):
            if k not in bg_bins and glob[k] > 0 and (obj_bin is None or glob[k] > glob[obj_bin]):
                obj_bin = k
    if obj_bin is None:
        obj_bin = 4
    report["hue_enrichment"] = round(best_enr, 2)
    band = set([k % NB for k in (obj_bin - 1, obj_bin, obj_bin + 1) if (k % NB) not in bg_bins])
    band.add(obj_bin)
    report["hue_global"] = glob
    report["bg_bins"] = sorted(bg_bins)
    report["obj_band"] = sorted(band)

    lut = [False] * 32768
    for qr in range(32):
        for qg in range(32):
            for qb in range(32):
                h, s, mx = hue_sat(qr * 8 + 4, qg * 8 + 4, qb * 8 + 4)
                if s > 22 and mx > 35 and (int(h // 20) % NB) in band:
                    lut[qr * 1024 + qg * 32 + qb] = True

    Z_LO = table_z + 0.004
    Z_HI = table_z + 0.10


    def detect(cam, stride):
        calib = cam_calib(cam)
        rows = as_rows(cam.rgb)
        drows = as_rows(cam.depth_m)
        H = len(rows); W = len(rows[0])
        fx, fy, cx, cy, R, t = calib
        out = []
        for v in range(0, H, stride):
            row = rows[v]; drow = drows[v]
            for u in range(0, W, stride):
                p = row[u]
                if lut[(p[0] // 8) * 1024 + (p[1] // 8) * 32 + (p[2] // 8)]:
                    d = drow[u]
                    if d > 0.03 and d < 5.0:
                        q = unproject(calib, u, v, d, CONV)
                        if q[2] > Z_LO and q[2] < Z_HI:
                            if button_xy is None or (q[0] - button_xy[0]) ** 2 + (q[1] - button_xy[1]) ** 2 > 0.02 * 0.02:
                                out.append((q[0], q[1], q[2]))
        return out


    def frame_points(obs, stride_front, stride_wrist):
        pts = detect(obs.cameras[front_name], stride_front)
        pts = pts + detect(obs.cameras[wrist_name], stride_wrist)
        return pts


    CELL = 0.006


    def components(pts, min_pts):
        cells = {}
        for p in pts:
            key = (int(math.floor(p[0] / CELL)), int(math.floor(p[1] / CELL)))
            if key in cells:
                cells[key].append(p)
            else:
                cells[key] = [p]
        seen = set()
        comps = []
        for key in cells:
            if key in seen:
                continue
            stack = [key]
            seen.add(key)
            members = []
            while len(stack) > 0:
                k = stack.pop()
                members = members + cells[k]
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nk = (k[0] + dx, k[1] + dy)
                        if nk in cells and nk not in seen:
                            seen.add(nk)
                            stack.append(nk)
            if len(members) >= min_pts:
                sx = sum(m[0] for m in members) / len(members)
                sy = sum(m[1] for m in members) / len(members)
                comps.append([sx, sy, len(members)])
        return comps


    def kmeans_update(centers, pts, gate, iters):
        cur = [list(c) for c in centers]
        counts = [0] * len(cur)
        for it in range(iters):
            acc = [[0.0, 0.0, 0] for c in cur]
            for p in pts:
                best = -1
                bd = gate * gate
                for j in range(len(cur)):
                    dd = (p[0] - cur[j][0]) ** 2 + (p[1] - cur[j][1]) ** 2
                    if dd < bd:
                        bd = dd
                        best = j
                if best >= 0:
                    acc[best][0] += p[0]
                    acc[best][1] += p[1]
                    acc[best][2] += 1
            for j in range(len(cur)):
                counts[j] = acc[j][2]
                if acc[j][2] >= 3:
                    cur[j] = [acc[j][0] / acc[j][2], acc[j][1] / acc[j][2]]
        return cur, counts


    # ---------------------------------------------------------------- tracking through the shuffle
    start_idx = release_idx
    for i in range(release_idx, nf):
        if ee[i][0][2] > release_pos[2] + 0.08:
            start_idx = i
            break
    report["track_start_idx"] = start_idx

    best_init = None
    for i in range(start_idx, min(nf, start_idx + 8)):
        pts = frame_points(video[i], 2, 2)
        comps = components(pts, 6)
        if best_init is None or len(comps) > len(best_init[1]):
            best_init = (i, comps, len(pts))
    init_idx = best_init[0]
    tracks = [[c[0], c[1]] for c in best_init[1]]
    report["init_idx"] = init_idx
    report["n_objects"] = len(tracks)
    report["init_tracks"] = [[round(c[0], 4), round(c[1], 4), c[2]] for c in best_init[1]]
    if len(tracks) == 0:
        report["init_fallback"] = "release_xy"
        tracks = [[release_pos[0], release_pos[1]]]
    target = 0
    bd = None
    for j in range(len(tracks)):
        dd = (tracks[j][0] - release_pos[0]) ** 2 + (tracks[j][1] - release_pos[1]) ** 2
        if bd is None or dd < bd:
            bd = dd
            target = j
    report["target_init"] = target
    report["target_init_dist"] = round(math.sqrt(bd), 4)

    vel = [[0.0, 0.0] for c in tracks]
    history = []
    min_sep = []
    coast = 0
    for i in range(init_idx + 1, nf):
        pts = frame_points(video[i], 2, 2)
        pred = [[tracks[j][0] + vel[j][0], tracks[j][1] + vel[j][1]] for j in range(len(tracks))]
        newc, counts = kmeans_update(pred, pts, 0.045, 2)
        for j in range(len(tracks)):
            if counts[j] >= 3:
                nv = [newc[j][0] - tracks[j][0], newc[j][1] - tracks[j][1]]
                vel[j] = [0.5 * vel[j][0] + 0.5 * nv[0], 0.5 * vel[j][1] + 0.5 * nv[1]]
                tracks[j] = newc[j]
            else:
                vel[j] = [0.5 * vel[j][0], 0.5 * vel[j][1]]
                tracks[j] = pred[j]
                if j == target:
                    coast = coast + 1
        sep = None
        for j in range(len(tracks)):
            if j != target:
                dd = math.sqrt((tracks[j][0] - tracks[target][0]) ** 2 + (tracks[j][1] - tracks[target][1]) ** 2)
                if sep is None or dd < sep:
                    sep = dd
        if sep is not None:
            min_sep.append(sep)
        if (i - init_idx) % 16 == 0 or i == nf - 1:
            history.append([i, len(pts), [round(tracks[target][0], 3), round(tracks[target][1], 3)], counts[target], round(sep, 3) if sep is not None else None, [[round(c[0], 3), round(c[1], 3)] for c in tracks], counts])
    report["track_history"] = history
    report["target_coast_frames"] = coast
    report["min_sep_overall"] = round(min(min_sep), 4) if min_sep else None
    report["close_calls_lt_25mm"] = len([s for s in min_sep if s < 0.025])
    report["final_tracks"] = [[round(c[0], 4), round(c[1], 4)] for c in tracks]
    report["target_final"] = [round(tracks[target][0], 4), round(tracks[target][1], 4)]

    # ---------------------------------------------------------------- live localisation of the target
    def live_target(all_tracks, tidx, gate):
        obs = get_observation()
        pts = frame_points(obs, 1, 1)
        comps = components(pts, 6)
        newc, counts = kmeans_update(all_tracks, pts, gate, 3)
        c = newc[tidx]
        if counts[tidx] < 3:
            return None, comps, pts, newc
        mine = []
        for p in pts:
            bj = -1
            bd2 = gate * gate
            for j in range(len(newc)):
                dd = (p[0] - newc[j][0]) ** 2 + (p[1] - newc[j][1]) ** 2
                if dd < bd2:
                    bd2 = dd
                    bj = j
            if bj == tidx:
                mine.append(p)
        top = quant([p[2] for p in mine], 0.9)
        return (c[0], c[1], top, len(mine)), comps, pts, newc


    tgt, live_comps, live_pts, live_centers = live_target(tracks, target, 0.045)
    report["live_components"] = [[round(c[0], 4), round(c[1], 4), c[2]] for c in live_comps]
    if tgt is None:
        tgt = (tracks[target][0], tracks[target][1], table_z + 0.03, 0)
    report["live_target"] = [round(tgt[0], 4), round(tgt[1], 4), round(tgt[2], 4), tgt[3]]
    report["live_centers"] = [[round(c[0], 4), round(c[1], 4)] for c in live_centers]

    # grasp height: the video's own grasp height above the table
    grasp_dz = grasp_pos[2] - table_z
    if grasp_dz < 0.008 or grasp_dz > 0.08:
        grasp_dz = 0.02
    report["grasp_dz"] = round(grasp_dz, 4)
    quat = grasp_quat


    def pose(x, y, z):
        return Pose((x, y, z), quat, base_frame)


    def ended(res):
        try:
            return bool(res.terminated) or bool(res.truncated)
        except Exception:
            return False


    log = []


    def mv(x, y, z, tol, steps, label):
        res = move_to_pose(pose(x, y, z), tolerance=tol, max_steps=steps)
        entry = [label, bool(res.ok)]
        try:
            entry.append(int(res.steps_executed))
        except Exception:
            entry.append(None)
        try:
            entry.append(str(res.status))
        except Exception:
            entry.append(None)
        log.append(entry)
        return res


    done = False
    cur = (tgt[0], tgt[1])
    for cycle in range(2):
        if done:
            break
        if cycle > 0:
            trk2 = [list(c) for c in live_centers]
            trk2[target] = [cur[0], cur[1]]
            again, c2, p2, live_centers = live_target(trk2, target, 0.04)
            if again is not None and (again[0] - cur[0]) ** 2 + (again[1] - cur[1]) ** 2 < 0.03 ** 2:
                cur = (again[0], again[1])
            report["recheck_%d" % cycle] = [round(cur[0], 4), round(cur[1], 4)] if again else None
        gz = table_z + grasp_dz
        r = open_gripper()
        if ended(r):
            done = True; break
        r = mv(cur[0], cur[1], gz + 0.10, 0.02, 200, "pre%d" % cycle)
        if ended(r):
            done = True; break
        if not r.ok and quat != home_quat:
            quat = home_quat
            report["quat_fallback"] = True
            r = mv(cur[0], cur[1], gz + 0.10, 0.02, 200, "pre%d_home" % cycle)
            if ended(r):
                done = True; break
        r = mv(cur[0], cur[1], gz, 0.01, 150, "grasp%d" % cycle)
        if ended(r):
            done = True; break
        r = close_gripper()
        if ended(r):
            done = True; break
        try:
            report["grip_after_close_%d" % cycle] = round(float(get_robot_state().gripper_positions["primary"]), 3)
        except Exception:
            pass
        r = mv(cur[0], cur[1], gz + 0.10, 0.02, 150, "lift%d" % cycle)
        if ended(r):
            done = True; break
        try:
            report["grip_after_lift_%d" % cycle] = round(float(get_robot_state().gripper_positions["primary"]), 3)
        except Exception:
            pass
        r = mv(cur[0], cur[1], gz + 0.008, 0.01, 150, "place%d" % cycle)
        if ended(r):
            done = True; break
        r = open_gripper()
        if ended(r):
            done = True; break
        r = mv(cur[0], cur[1], gz + 0.10, 0.02, 150, "retreat%d" % cycle)
        if ended(r):
            done = True; break

    if not done:
        if button_xy is None:
            bres, btried = find_button()
            if bres is not None:
                button_xy = bres[1]
                button_top = bres[2]
        if button_xy is not None:
            r = close_gripper()
            if not ended(r):
                r = mv(button_xy[0], button_xy[1], button_top + 0.08, 0.02, 200, "btn_above")
            if not ended(r):
                r = mv(button_xy[0], button_xy[1], button_top - 0.015, 0.01, 120, "btn_press")
            if not ended(r):
                r = mv(button_xy[0], button_xy[1], button_top + 0.06, 0.02, 100, "btn_up")
            done = ended(r)
    report["motion_log"] = log
    report["episode_ended"] = done
except Exception as exc:
    report["error"] = str(exc)[:500]
result = report
