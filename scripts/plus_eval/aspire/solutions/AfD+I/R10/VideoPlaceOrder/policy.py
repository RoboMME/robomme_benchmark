# VideoPlaceOrder v03 -- v02 with the swap-model table-plane height derived at runtime (median z of live target clouds)
# instead of a constant; 2D ellipse fallback if no target cloud. v02: lighting-robust occupancy, saturated scores, penalty matching.
# Goal form: "watch the video carefully, then place the <colour> cube on the <ordinal> target
# it was previously placed on". In the conditioning video the robot places the named cube on a
# sequence of purple bullseye targets while target PAIRS swap places (each pair rotates ~180 deg
# about its midpoint). The answer is the identity of the k-th placement tracked through all
# later swaps to its final position at the end of the video.
# Everything episode-dependent comes from get_task_context().language,
# robomme_plus.get_conditioning_video() and live observations.
import math

report = {"version": "v03_plane_z"}

# ------------------------------------------------------------------ perception helpers
def is_purple(p):
    r = p[0]; g = p[1]; b = p[2]
    return r >= 90 and b >= 100 and b - g >= 18

def is_color(p, name):
    r = p[0]; g = p[1]; b = p[2]
    if name == "red":
        return r > 150 and g < 70 and b < 70
    if name == "green":
        return g > 120 and r < 90 and b < 90
    if name == "blue":
        return b > 120 and r < 60 and g < 60
    return False

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

def detect(img, roi, stride, cube_color, cell):
    x0 = roi[0]; y0 = roi[1]; x1 = roi[2]; y1 = roi[3]
    cells = {}
    cx = 0.0; cy = 0.0; cn = 0
    for y in range(y0, min(y1, len(img)), stride):
        row = img[y]
        for x in range(x0, min(x1, len(row)), stride):
            p = row[x]
            if is_purple(p):
                key = (x // cell, y // cell)
                c = cells.get(key)
                if c is None:
                    cells[key] = [x, y, 1]
                else:
                    c[0] += x; c[1] += y; c[2] += 1
            elif is_color(p, cube_color):
                cx += x; cy += y; cn += 1
    seen = set()
    blobs = []
    for key in cells:
        if key in seen:
            continue
        stack = [key]; seen.add(key)
        sx = 0; sy = 0; sn = 0
        while stack:
            k = stack.pop()
            c = cells[k]
            sx += c[0]; sy += c[1]; sn += c[2]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nk = (k[0] + dx, k[1] + dy)
                    if nk in cells and nk not in seen:
                        seen.add(nk); stack.append(nk)
        blobs.append([sx / sn, sy / sn, sn])
    cube = None
    if cn > 0:
        cube = [cx / cn, cy / cn, cn]
    return blobs, cube

def nearest(S, p):
    best = -1; bd = None
    for i in range(len(S)):
        d = ((S[i][0] - p[0]) ** 2 + (S[i][1] - p[1]) ** 2) ** 0.5
        if bd is None or d < bd:
            bd = d; best = i
    return best, bd

def frame_info_density(img, S, r, thr):
    occ = []
    for s in S:
        occ.append(purple_density(img, s[0], s[1], r, 1) >= thr)
    return occ, []

def frame_info(S, blobs, static_px):
    occ = [False] * len(S)
    moving = []
    for b in blobs:
        i, d = nearest(S, b)
        if d is not None and d < static_px:
            occ[i] = True
        else:
            moving.append(b)
    return occ, moving

def matchings(items):
    if len(items) < 2:
        return [[]]
    first = items[0]
    rest = items[1:]
    out = []
    for m in matchings(rest):
        out.append(m)
    for t in range(len(rest)):
        others = rest[:t] + rest[t + 1:]
        for m in matchings(others):
            out.append([(first, rest[t])] + m)
    return out

def purple_density(img, x, y, r, step):
    n = 0; t = 0
    yi0 = int(y - r); yi1 = int(y + r)
    xi0 = int(x - r); xi1 = int(x + r)
    for yy in range(yi0, yi1 + 1, step):
        if yy < 0 or yy >= len(img):
            continue
        row = img[yy]
        for xx in range(xi0, xi1 + 1, step):
            if xx < 0 or xx >= len(row):
                continue
            if (xx - x) ** 2 + (yy - y) ** 2 > r * r:
                continue
            t += 1
            if is_purple(row[xx]):
                n += 1
    if t == 0:
        return 0.0
    return n / t

COS = [1.0, 0.9659, 0.8660, 0.7071, 0.5, 0.2588, 0.0, -0.2588, -0.5, -0.7071, -0.8660, -0.9659]
SIN = [0.0, 0.2588, 0.5, 0.7071, 0.8660, 0.9659, 1.0, 0.9659, 0.8660, 0.7071, 0.5, 0.2588]

def theta_table(img, S, occ, i, j, yscale, r, excl):
    si = S[i]; sj = S[j]
    mx = 0.5 * (si[0] + sj[0]); my = 0.5 * (si[1] + sj[1])
    vx = si[0] - mx; vy = (si[1] - my) / yscale
    tabs = []
    for sense in (1.0, -1.0):
        row = []
        for k in range(1, len(COS)):
            c = COS[k]; s = SIN[k] * sense
            px = mx + c * vx - s * vy; py = my + (s * vx + c * vy) * yscale
            qx = 2 * mx - px; qy = 2 * my - py
            bad = False
            for t in range(len(S)):
                if t != i and t != j and occ[t]:
                    if (S[t][0] - px) ** 2 + (S[t][1] - py) ** 2 < excl * excl or (S[t][0] - qx) ** 2 + (S[t][1] - qy) ** 2 < excl * excl:
                        bad = True
            if bad:
                row.append(None)
            else:
                row.append(min(purple_density(img, px, py, r, 1), purple_density(img, qx, qy, r, 1)))
        tabs.append(row)
    return tabs

def cam_model(cam):
    K = cam.intrinsics
    pose = cam.camera_pose
    q = [float(v) for v in pose.quaternion_wxyz]
    t = [float(v) for v in pose.position]
    w = q[0]; x = q[1]; y = q[2]; z = q[3]
    R = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    return {"fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]), "cy": float(K[1][2]), "R": R, "t": t}

def pix_to_plane(cm, u, v, z0):
    dc = ((u - cm["cx"]) / cm["fx"], (v - cm["cy"]) / cm["fy"], 1.0)
    R = cm["R"]; t = cm["t"]
    dw = [R[a][0] * dc[0] + R[a][1] * dc[1] + R[a][2] * dc[2] for a in range(3)]
    if abs(dw[2]) < 1e-9:
        return None
    s = (z0 - t[2]) / dw[2]
    return (t[0] + s * dw[0], t[1] + s * dw[1])

def plane_to_pix(cm, X, Y, z0):
    R = cm["R"]; t = cm["t"]
    d = (X - t[0], Y - t[1], z0 - t[2])
    c = [R[0][a] * d[0] + R[1][a] * d[1] + R[2][a] * d[2] for a in range(3)]
    if c[2] <= 1e-6:
        return None
    return (cm["cx"] + cm["fx"] * c[0] / c[2], cm["cy"] + cm["fy"] * c[1] / c[2])

def pair_paths(cm, S, i, j, z0, nsteps):
    # projected antipodal pixel points for rotation angles 180*k/nsteps, k=1..nsteps-1, both senses
    A = pix_to_plane(cm, S[i][0], S[i][1], z0); B = pix_to_plane(cm, S[j][0], S[j][1], z0)
    mx = 0.5 * (A[0] + B[0]); my = 0.5 * (A[1] + B[1])
    vx = A[0] - mx; vy = A[1] - my
    tabs = []
    for sense in (1.0, -1.0):
        row = []
        for k in range(1, nsteps):
            th = math.pi * k / nsteps
            c = math.cos(th); s = math.sin(th) * sense
            px = mx + c * vx - s * vy; py = my + s * vx + c * vy
            qx = 2 * mx - px; qy = 2 * my - py
            row.append((plane_to_pix(cm, px, py, z0), plane_to_pix(cm, qx, qy, z0)))
        tabs.append(row)
    return tabs

def theta_table3(img, S, occ, i, j, paths, r, excl, sat):
    tabs = []
    for row in paths:
        out = []
        for pq in row:
            p = pq[0]; q = pq[1]
            if p is None or q is None:
                out.append(None)
                continue
            bad = False
            for t in range(len(S)):
                if t != i and t != j and occ[t]:
                    if (S[t][0] - p[0]) ** 2 + (S[t][1] - p[1]) ** 2 < excl * excl or (S[t][0] - q[0]) ** 2 + (S[t][1] - q[1]) ** 2 < excl * excl:
                        bad = True
            if bad:
                out.append(None)
            else:
                out.append(min(1.0, min(purple_density(img, p[0], p[1], r, 1), purple_density(img, q[0], q[1], r, 1)) / sat))
        tabs.append(out)
    return tabs

def monotone_path_score(rows):
    K = len(rows[0]) if rows else 0
    best = [0.0] * K
    for row in rows:
        new = []
        run = None
        for k in range(K):
            if run is None or best[k] > run:
                run = best[k]
            v = row[k]
            new.append(run + (v if v is not None else 0.0))
        best = new
    return max(best) if best else 0.0

def placements(cube_hist, S, times, rest_px, rest_s, near_px, start_px):
    segs = []
    cur = None
    for i in range(len(cube_hist)):
        c = cube_hist[i]
        if c is not None and c[2] >= 8 and cur is not None and abs(c[0] - cur[2]) < rest_px and abs(c[1] - cur[3]) < rest_px:
            cur[1] = i
            continue
        if cur is not None:
            segs.append(cur)
        cur = None if (c is None or c[2] < 8) else [i, i, c[0], c[1]]
    if cur is not None:
        segs.append(cur)
    segs = [s for s in segs if times[s[1]] - times[s[0]] >= rest_s]
    first = None
    for c in cube_hist:
        if c is not None and c[2] >= 8:
            first = c
            break
    events = []
    for s in segs:
        if first is not None and abs(s[2] - first[0]) + abs(s[3] - first[1]) < start_px:
            events.append(("start", s[0], s[1], None, None))
            continue
        best, bd = nearest(S, (s[2], s[3]))
        if bd is not None and bd < near_px:
            if events and events[-1][0] == "place" and events[-1][3] == best:
                continue
            events.append(("place", s[0], s[1], best, round(bd, 1)))
        else:
            events.append(("rest_off_target", s[0], s[1], best, None if bd is None else round(bd, 1)))
    return events

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
            "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6}

def parse_goal(text):
    words = text.lower().replace(",", " ").replace(".", " ").split()
    colour = None
    for w in words:
        if colour is None and w in ("red", "green", "blue"):
            colour = w
    k = None
    for w in words:
        if k is None and w in ORDINALS:
            k = ORDINALS[w]
    return colour, k

def analyze_video(video, colour, k, params):
    # Returns dict with S (pixel positions), events, placements, identities, answer index.
    out = {}
    roi = (0, 0, 256, 200)
    n = len(video)
    out["n_frames"] = n
    imgs = []
    times = []
    for fi in range(n):
        frame = video[fi]
        imgs.append(to_rows(frame.cameras["front"].rgb))
        try:
            times.append(float(frame.timestamp_s))
        except Exception:
            times.append(fi * 0.05)
    out["t_first"] = times[0] if times else None
    out["t_last"] = times[-1] if times else None
    b0, c0 = detect(imgs[0], roi, 2, colour, 2)
    S = [(b[0], b[1]) for b in b0 if b[2] >= 20]
    out["S"] = [(round(s[0], 1), round(s[1], 1)) for s in S]
    cm = None
    if params.get("model3d"):
        try:
            cm = cam_model(video[0].cameras["front"])
        except Exception as exc:
            out["cam_error"] = str(exc)[:200]
    out["model3d"] = cm is not None
    infos = []
    cubes = []
    dens = []
    othr = []
    if params.get("occ_density"):
        for fi in range(n):
            dens.append([purple_density(imgs[fi], s[0], s[1], 5.0, 1) for s in S])
        for t in range(len(S)):
            col = sorted([dens[fi][t] for fi in range(n)])
            othr.append(max(params["occ_thr"], params["occ_rel"] * col[len(col) // 2]))
        out["occ_thr_pos"] = [round(v, 3) for v in othr]
    for fi in range(n):
        blobs, cube = detect(imgs[fi], roi, 2, colour, 2)
        blobs = [b for b in blobs if b[2] >= 5]
        info = frame_info(S, blobs, 7.0)
        if params.get("occ_density"):
            dv = [dens[fi][t] >= othr[t] for t in range(len(S))]
            if params.get("occ_or"):
                info = ([info[0][t] or dv[t] for t in range(len(S))], info[1])
            else:
                info = (dv, info[1])
        infos.append(info)
        cubes.append(cube)
    runs = []
    cur = None
    for fi in range(n):
        nv = len([o for o in infos[fi][0] if not o])
        if nv >= 2:
            if cur is not None and fi - cur[1] <= 2:
                cur[1] = fi
            else:
                if cur is not None:
                    runs.append(cur)
                cur = [fi, fi]
    if cur is not None:
        runs.append(cur)
    perm_events = []
    ev_log = []
    YS = params["yscale"]; THR = params["thr"]
    for run in runs:
        a = run[0]; b = run[1]
        fr = list(range(a, b + 1))
        vc = [len([fi for fi in fr if not infos[fi][0][i]]) for i in range(len(S))]
        V = [i for i in range(len(S)) if vc[i] >= 0.5 * len(fr)]
        if len(V) > 6:
            V = V[:6]
        scores = {}
        for x in range(len(V)):
            for y in range(x + 1, len(V)):
                i = V[x]; j = V[y]
                rp = []; rm = []
                paths = None
                if cm is not None:
                    paths = pair_paths(cm, S, i, j, params["z0"], params["nsteps"])
                for fi in fr:
                    if paths is not None:
                        tabs = theta_table3(imgs[fi], S, infos[fi][0], i, j, paths, params["r"], 9.0, params["sat"])
                    else:
                        tabs = theta_table(imgs[fi], S, infos[fi][0], i, j, YS, 3.0, 9.0)
                    rp.append(tabs[0]); rm.append(tabs[1])
                scores[(i, j)] = max(monotone_path_score(rp), monotone_path_score(rm))
        best = []
        bs = 0.0
        CP = params.get("cpen", 0.0)
        for mt in matchings(V):
            okm = True
            for pr in mt:
                if scores[pr] < (CP if CP > 0 else THR):
                    okm = False
            if not okm:
                continue
            s = sum([scores[pr] - CP for pr in mt])
            if s > bs:
                bs = s; best = mt
        dur = times[b] - times[a]
        ev_log.append((round(times[a], 2), round(dur, 2), V, best, dict([(str(kk[0]) + "-" + str(kk[1]), round(vv, 2)) for kk, vv in scores.items()])))
        if best and dur <= params["dmax"]:
            perm_events.append((b, best))
    out["events"] = ev_log
    ev = placements(cubes, S, times, params["rest_px"], params["rest_s"], 14.0, 10.0)
    places = [x for x in ev if x[0] == "place"]
    out["placements"] = [(x[3], round(times[x[1]], 2)) for x in places]
    out["cube_events"] = [(x[0], round(times[x[1]], 2), x[3]) for x in ev]

    def ident_at(limit):
        idn = list(range(len(S)))
        for pe in perm_events:
            if pe[0] > limit:
                break
            new = idn[:]
            for pr in pe[1]:
                new[pr[0]] = idn[pr[1]]
                new[pr[1]] = idn[pr[0]]
            idn = new
        return idn

    place_ids = [ident_at(x[1])[x[3]] for x in places]
    final = ident_at(10 ** 9)
    out["placed_ids"] = place_ids
    out["final_ident"] = final
    out["ids_unique"] = len(set(place_ids)) == len(place_ids)
    answer = None
    if k is not None and len(place_ids) >= k:
        answer = final.index(place_ids[k - 1])
        out["answer_mode"] = "tracked"
    elif places:
        answer = places[-1][3]
        out["answer_mode"] = "fallback_last_place"
    out["answer"] = answer
    if answer is not None:
        out["answer_px"] = (round(S[answer][0], 1), round(S[answer][1], 1))
    imgs = None
    return out

PARAMS = {"yscale": 0.9, "thr": 1.5, "dmax": 1.0e9, "model3d": True, "z0": None, "nsteps": 18, "r": 3.0, "sat": 0.25, "occ_density": True, "occ_thr": 0.06, "occ_rel": 0.5, "occ_or": True, "rest_px": 3.0, "rest_s": 0.8, "cpen": 0.8}
# ---- CONTROL ----

# ------------------------------------------------------------------ geometry helpers
def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    return ordered[int((len(ordered) - 1) * fraction)]

def qmul(a, b):
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]

def qrot_x_axis(q):
    w = q[0]; x = q[1]; y = q[2]; z = q[3]
    return (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))

def rect_yaw(points):
    # min-area bounding rectangle angle (deg, 0..90) of xy points
    pts = [(float(p[0]), float(p[1])) for p in points]
    best = 0.0; ba = None
    for step_i in range(18):
        ang = step_i * 5.0
        c = math.cos(math.radians(ang)); s = math.sin(math.radians(ang))
        us = [c * p[0] + s * p[1] for p in pts]
        vs = [-s * p[0] + c * p[1] for p in pts]
        area = (quantile(us, 0.95) - quantile(us, 0.05)) * (quantile(vs, 0.95) - quantile(vs, 0.05))
        if ba is None or area < ba:
            ba = area; best = ang
    return best

def cloud_of(candidate):
    try:
        return mask_to_point_cloud(candidate.mask, "front", target_frame=base_frame)
    except Exception:
        return mask_to_point_cloud(candidate, "front", target_frame=base_frame)

def safe_status(res):
    d = {}
    try:
        d["ok"] = bool(res.ok)
    except Exception:
        d["ok"] = None
    try:
        d["status"] = str(res.status)
    except Exception:
        pass
    try:
        d["steps"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        d["term"] = bool(res.terminated)
        d["trunc"] = bool(res.truncated)
    except Exception:
        pass
    try:
        if res.error is not None:
            d["error"] = str(res.error)[:200]
    except Exception:
        pass
    return d

def ended(res):
    try:
        return bool(res.terminated) or bool(res.truncated)
    except Exception:
        return False

motions = []
episode_over = False

def move(name, pos, quat, tol, steps):
    res = move_to_pose(Pose((pos[0], pos[1], pos[2]), quat, base_frame), tolerance=tol, max_steps=steps)
    st = safe_status(res)
    st["name"] = name
    st["target"] = [round(v, 4) for v in pos]
    motions.append(st)
    return res

# ------------------------------------------------------------------ main
goal = get_task_context().language
report["goal"] = goal
colour, k_ord = parse_goal(goal)
report["colour"] = colour
report["ordinal"] = k_ord
try:
    report["task_meta"] = str(robomme_plus.get_task_metadata())[:600]
except Exception as exc:
    report["task_meta_error"] = str(exc)[:200]

def box_of(cand):
    try:
        b = cand.box_xyxy
        return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    except Exception:
        return None

def candidates(prompt):
    found = segment_text("front", prompt)
    out = []
    try:
        if found.ok:
            for cand in found.segmentations[:12]:
                out.append(cand)
    except Exception:
        pass
    return out

base_frame = get_robot_state().base_frame
# ---- table-plane height for the swap rotation model, derived from the live target clouds
tcands = candidates("purple target")
tcands = tcands + candidates("target")
plane_zs = []
for cand in tcands:
    b = box_of(cand)
    if b is None or (b[2] - b[0]) * (b[3] - b[1]) >= 2500:
        continue
    try:
        pc = cloud_of(cand).points
        if len(pc) >= 10:
            plane_zs.append(quantile([p[2] for p in pc], 0.5))
    except Exception:
        pass
run_params = dict(PARAMS)
if plane_zs:
    run_params["z0"] = quantile(plane_zs, 0.5)
else:
    run_params["model3d"] = False
report["plane_z"] = None if not plane_zs else round(run_params["z0"], 4)
report["plane_n"] = len(plane_zs)

analysis = None
try:
    video = robomme_plus.get_conditioning_video()
    analysis = analyze_video(video, colour, k_ord, run_params)
    video = None
except Exception as exc:
    report["analysis_error"] = str(exc)[:300]
report["analysis"] = analysis

obs = get_observation()
live_img = to_rows(obs.cameras["front"].rgb)
live_blobs, live_cube = detect(live_img, (0, 0, 256, 256), 2, colour, 2)
live_blobs = [b for b in live_blobs if b[2] >= 5]
report["live_blobs"] = [(round(b[0], 1), round(b[1], 1), b[2]) for b in live_blobs]
report["live_cube_px"] = None if live_cube is None else (round(live_cube[0], 1), round(live_cube[1], 1), live_cube[2])

# ---- target pixel
target_px = None
if analysis is not None and analysis.get("answer_px") is not None:
    ap = analysis["answer_px"]
    target_px = (ap[0], ap[1])
    bi, bd = nearest([(b[0], b[1]) for b in live_blobs], ap)
    if bi >= 0 and bd is not None and bd < 10.0:
        target_px = (live_blobs[bi][0], live_blobs[bi][1])
    report["live_match_dist"] = None if bd is None else round(bd, 1)
report["target_px"] = target_px
if target_px is None:
    raise ValueError("no target answer: " + str(report)[:1500])

# ---- localize target in 3D
target_xyz = None
tinfo = {}
bestc = None; barea = None
for cand in tcands:
    b = box_of(cand)
    if b is None:
        continue
    if b[0] - 2 <= target_px[0] <= b[2] + 2 and b[1] - 2 <= target_px[1] <= b[3] + 2:
        area = (b[2] - b[0]) * (b[3] - b[1])
        if area < 2500 and (barea is None or area < barea):
            barea = area; bestc = cand
if bestc is None:
    pts_found = segment_points("front", [[int(round(target_px[0])), int(round(target_px[1]))]])
    tinfo["via"] = "points"
    try:
        if pts_found.ok:
            bestc = pts_found.segmentations[0]
    except Exception:
        bestc = None
else:
    tinfo["via"] = "text"
if bestc is not None:
    tinfo["box"] = box_of(bestc)
    tcloud = cloud_of(bestc)
    tp = tcloud.points
    if len(tp) >= 10:
        target_xyz = (quantile([p[0] for p in tp], 0.5), quantile([p[1] for p in tp], 0.5), quantile([p[2] for p in tp], 0.5))
        tinfo["extent_xy"] = (round(quantile([p[0] for p in tp], 0.95) - quantile([p[0] for p in tp], 0.05), 3),
                              round(quantile([p[1] for p in tp], 0.95) - quantile([p[1] for p in tp], 0.05), 3))
        tinfo["n"] = len(tp)
report["target_info"] = tinfo
report["target_xyz"] = None if target_xyz is None else [round(v, 4) for v in target_xyz]
if target_xyz is None:
    raise ValueError("target localisation failed: " + str(report)[:1500])

# ---- localize cube
cube_prompt = colour + " cube"
ccands = candidates(cube_prompt)
cinfo = {"n_cands": len(ccands)}
cube_c = None; carea = None
for cand in ccands:
    b = box_of(cand)
    if b is None:
        continue
    cnt = 0
    for yy in range(max(0, int(b[1])), min(256, int(b[3]) + 1)):
        row = live_img[yy]
        for xx in range(max(0, int(b[0])), min(256, int(b[2]) + 1)):
            if is_color(row[xx], colour):
                cnt += 1
    area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    if cnt >= 6 and cnt / area > 0.15 and (carea is None or area < carea):
        carea = area; cube_c = cand
if cube_c is None:
    loc = localize_object(cube_prompt, camera_name="front", target_frame=base_frame)
    cinfo["via"] = "localize_object"
    cpts = loc.point_cloud.points
else:
    cinfo["via"] = "colour_box"
    cinfo["box"] = box_of(cube_c)
    cpts = cloud_of(cube_c).points
if len(cpts) < 10:
    raise ValueError("cube localisation failed: " + str(report)[:1500])
cx = quantile([p[0] for p in cpts], 0.5); cy = quantile([p[1] for p in cpts], 0.5)
ztop = quantile([p[2] for p in cpts], 0.9); zlow = quantile([p[2] for p in cpts], 0.05)
cube_h = max(0.02, min(0.06, ztop - zlow))
cinfo["xyz"] = [round(cx, 4), round(cy, 4), round(ztop, 4)]
cinfo["height"] = round(cube_h, 4)
try:
    yaw_c = rect_yaw(cpts)
except Exception:
    yaw_c = 0.0
cinfo["yaw_deg"] = yaw_c
report["cube_info"] = cinfo

home_state = get_robot_state()
q_home = [float(v) for v in home_state.end_effector_poses["primary"].quaternion_wxyz]
xa = qrot_x_axis(q_home)
psi = math.degrees(math.atan2(xa[1], xa[0]))
delta = yaw_c - psi
while delta > 45.0:
    delta -= 90.0
while delta < -45.0:
    delta += 90.0
half = math.radians(delta) / 2.0
q_grasp = qmul([math.cos(half), 0.0, 0.0, math.sin(half)], q_home)
report["yaw"] = {"psi": round(psi, 1), "delta": round(delta, 1)}

# ---- pick
grasp_z = max(ztop - 0.02, zlow + 0.5 * cube_h)
open_gripper()
r = move("pregrasp", (cx, cy, grasp_z + 0.12), q_grasp, 0.02, 250)
if ended(r):
    episode_over = True
if not episode_over:
    r = move("grasp", (cx, cy, grasp_z), q_grasp, 0.015, 250)
    episode_over = ended(r)
if not episode_over:
    rc = close_gripper()
    motions.append({"name": "close", "res": safe_status(rc)})
    episode_over = ended(rc)
if not episode_over:
    r = move("lift", (cx, cy, grasp_z + 0.15), q_grasp, 0.02, 250)
    episode_over = ended(r)
if not episode_over:
    try:
        report["grip_after_lift"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
    except Exception:
        pass
# ---- transport and place
tx = target_xyz[0]; ty = target_xyz[1]; tz = target_xyz[2]
if not episode_over:
    r = move("above_target", (tx, ty, grasp_z + 0.15), q_grasp, 0.02, 300)
    episode_over = ended(r)
if not episode_over:
    r = move("release_pose", (tx, ty, tz + cube_h + 0.02), q_grasp, 0.015, 250)
    episode_over = ended(r)
if not episode_over:
    ro = open_gripper()
    motions.append({"name": "open", "res": safe_status(ro)})
    episode_over = ended(ro)
if not episode_over:
    try:
        state = get_robot_state()
        for settle_tick in range(10):
            sr = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=state.joint_positions["primary"],
                gripper_position=1.0, embodiment=state.embodiment)}))
            if ended(sr):
                episode_over = True
                motions.append({"name": "settle_end", "tick": settle_tick, "res": safe_status(sr)})
                break
    except Exception as exc:
        report["settle_error"] = str(exc)[:200]
if not episode_over:
    try:
        r = move("retreat", (tx, ty, tz + 0.15), q_grasp, 0.03, 200)
        episode_over = ended(r)
    except Exception as exc:
        report["retreat_error"] = str(exc)[:200]
if not episode_over:
    try:
        fobs = get_observation()
        fimg = to_rows(fobs.cameras["front"].rgb)
        fb, fc = detect(fimg, (0, 0, 256, 256), 2, colour, 2)
        report["final_cube_px"] = None if fc is None else (round(fc[0], 1), round(fc[1], 1), fc[2])
    except Exception as exc:
        report["final_obs_error"] = str(exc)[:200]
report["motions"] = motions
report["episode_over"] = episode_over
result = report
