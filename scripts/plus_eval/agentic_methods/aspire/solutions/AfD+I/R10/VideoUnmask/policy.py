# VideoUnmask v01 -- v00 + point-prompt container localization at projected cube positions,
# yaw restricted to +-45 deg of the start wrist yaw, IK/hover fallbacks, empty-grasp retry.
# (v00 header follows)
# Strategy: scan the conditioning video (front + wrist RGB-D) for saturated coloured cubes,
# deproject them into robot_base, cluster per colour; localize the white containers live;
# match each goal cube to the nearest container; pick up containers in goal order
# (lift, and put back down before the next one).
import math

report = {"version": "v01", "errors": []}
goal = get_task_context().language
report["goal"] = goal
try:
    meta = robomme_plus.get_task_metadata()
    report["max_steps"] = meta["max_steps"]
    try:
        report["video_sampling"] = meta["conditioning_video_sampling"]
    except Exception:
        report["video_sampling"] = None
except Exception as exc:
    report["errors"].append("metadata: " + str(exc))

state0 = get_robot_state()
base_frame = state0.base_frame
embodiment = state0.embodiment

COLOR_WORDS = ["red", "orange", "yellow", "green", "blue", "purple"]


# ---------------------------------------------------------------- math helpers
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def qrot(q, v):
    w, x, y, z = q
    r00 = 1 - 2 * (y * y + z * z); r01 = 2 * (x * y - w * z); r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z); r11 = 1 - 2 * (x * x + z * z); r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y); r21 = 2 * (y * z + w * x); r22 = 1 - 2 * (x * x + y * y)
    return (r00 * v[0] + r01 * v[1] + r02 * v[2],
            r10 * v[0] + r11 * v[1] + r12 * v[2],
            r20 * v[0] + r21 * v[1] + r22 * v[2])


def qrot_inv(q, v):
    return qrot((q[0], -q[1], -q[2], -q[3]), v)


def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def wrap_deg(a):
    while a > 180.0:
        a -= 360.0
    while a <= -180.0:
        a += 360.0
    return a


# camera conventions: map OpenCV-style camera coords (x right, y down, z forward) <-> pose frame
def cv_to_cam(conv, p):
    if conv == "opencv":
        return p
    if conv == "opengl":
        return (p[0], -p[1], -p[2])
    return (p[2], -p[0], -p[1])  # ros: x forward, y left, z up


def cam_to_cv(conv, p):
    if conv == "opencv":
        return p
    if conv == "opengl":
        return (p[0], -p[1], -p[2])
    return (-p[1], -p[2], p[0])


def intr(K):
    return float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])


def deproject(u, v, d, K, pose, conv):
    fx, fy, cx, cy = intr(K)
    pc = ((u - cx) / fx * d, (v - cy) / fy * d, d)
    pcam = cv_to_cam(conv, pc)
    q = [float(val) for val in pose.quaternion_wxyz]
    t = [float(val) for val in pose.position]
    w = qrot(q, pcam)
    return (w[0] + t[0], w[1] + t[1], w[2] + t[2])


def project(p, K, pose, conv):
    fx, fy, cx, cy = intr(K)
    q = [float(val) for val in pose.quaternion_wxyz]
    t = [float(val) for val in pose.position]
    rel = (p[0] - t[0], p[1] - t[1], p[2] - t[2])
    pc = cam_to_cv(conv, qrot_inv(q, rel))
    if pc[2] <= 1e-4:
        return None
    return (fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy)


def to_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        return [[arr[y][x] for x in range(len(arr[0]))] for y in range(len(arr))]


# ---------------------------------------------------------------- colour classifier
def classify(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 50:
        return None
    if (mx - mn) < 0.72 * mx:
        return None
    dlt = float(mx - mn)
    if mx == r:
        h = 60.0 * (((g - b) / dlt) % 6.0)
    elif mx == g:
        h = 60.0 * ((b - r) / dlt + 2.0)
    else:
        h = 60.0 * ((r - g) / dlt + 4.0)
    if h < 12.0 or h >= 340.0:
        return "red"
    if h < 42.0:
        return "orange"
    if h < 72.0:
        return "yellow"
    if h < 165.0:
        return "green"
    if h < 255.0:
        return "blue"
    return "purple"


# ---------------------------------------------------------------- goal parsing
def parse_goal(text):
    low = text.lower()
    start = low.find("then")
    if start < 0:
        start = 0
    seq = []
    pos = start
    while True:
        best = None
        for c in COLOR_WORDS:
            k = low.find(c, pos)
            if k >= 0 and (best is None or k < best[0]):
                best = (k, c)
        if best is None:
            break
        k, c = best
        window = low[max(start, k - 40):k]
        cut = max(window.rfind(","), window.rfind("followed"), window.rfind("then"))
        if cut >= 0:
            window = window[cut:]
        is_all = ("all " in window) or (low[k:k + 20].find("cubes") >= 0 and low[k:k + 20].find("cubes") < 12)
        seq.append({"color": c, "all": bool(is_all)})
        pos = k + len(c)
    return seq


goal_seq = parse_goal(goal)
report["goal_seq"] = goal_seq
wanted_colors = set(item["color"] for item in goal_seq)

# ---------------------------------------------------------------- 1. scan conditioning video
FRAME_STRIDE = 2
PIX_STRIDE = 2
video = robomme_plus.get_conditioning_video()
n_video = len(video)
report["video_frames"] = n_video
raw_blobs = []  # (color, cam, frame_index, u, v, d, npix)
cams_seen = {}
fi = 0
while fi < n_video:
    frame = video[fi]
    for cam in ("front", "wrist"):
        try:
            camobs = frame.cameras[cam]
        except Exception:
            continue
        cams_seen[cam] = cams_seen.get(cam, 0) + 1
        rows = to_rows(camobs.rgb)
        hits = {}
        yy = 0
        nrow = len(rows)
        while yy < nrow:
            row = rows[yy]
            xx = 0
            ncol = len(row)
            while xx < ncol:
                px = row[xx]
                r = px[0]; g = px[1]; b = px[2]
                mx = max(r, g, b)
                if mx >= 50 and (mx - min(r, g, b)) >= 0.72 * mx:
                    c = classify(r, g, b)
                    if c in wanted_colors:
                        if c not in hits:
                            hits[c] = []
                        hits[c].append((xx, yy))
                xx += PIX_STRIDE
            yy += PIX_STRIDE
        if len(hits) == 0:
            continue
        drows = None
        for c in hits:
            clusters = []  # [sum_u, sum_v, n, pts]
            for p in hits[c]:
                placed = False
                for cl in clusters:
                    mu = cl[0] / cl[2]; mv = cl[1] / cl[2]
                    if abs(p[0] - mu) <= 12 and abs(p[1] - mv) <= 12:
                        cl[0] += p[0]; cl[1] += p[1]; cl[2] += 1; cl[3].append(p)
                        placed = True
                        break
                if not placed:
                    clusters.append([p[0], p[1], 1, [p]])
            for cl in clusters:
                if cl[2] < 3:
                    continue
                if drows is None:
                    drows = to_rows(camobs.depth_m)
                us = sorted(pt[0] for pt in cl[3]); vs = sorted(pt[1] for pt in cl[3])
                um = us[len(us) // 2]; vm = vs[len(vs) // 2]
                ds = sorted(float(drows[pt[1]][pt[0]]) for pt in cl[3])
                dm = ds[len(ds) // 2]
                raw_blobs.append((c, cam, fi, float(um), float(vm), dm, cl[2]))
    fi += FRAME_STRIDE
report["raw_blob_count"] = len(raw_blobs)
report["cams_seen"] = cams_seen

# ---------------------------------------------------------------- 2. live containers
obs = get_observation()
live_cams = [name for name in obs.cameras]
report["live_cams"] = live_cams
CONTAINER_PROMPTS = ["white box", "box"]


def container_candidates(cam):
    out = []
    for prompt in CONTAINER_PROMPTS:
        seg = segment_text(cam, prompt)
        if not seg.ok:
            continue
        for cand in seg.segmentations:
            try:
                pc = mask_to_point_cloud(cand, cam, target_frame=base_frame)
            except Exception:
                continue
            pts = [(float(p[0]), float(p[1]), float(p[2])) for p in pc.points]
            if len(pts) < 15:
                continue
            box = None
            if cand.box_xyxy is not None:
                box = [float(val) for val in cand.box_xyxy]
            out.append({"prompt": prompt, "score": float(cand.score), "pts": pts, "box": box, "cam": cam})
        if len(out) > 0:
            break
    return out


def describe(pts):
    top = quantile([p[2] for p in pts], 0.92)
    bottom = quantile([p[2] for p in pts], 0.03)
    topp = [p for p in pts if p[2] >= top - 0.012]
    if len(topp) < 6:
        topp = pts
    best = None
    for k in range(30):
        ang = math.radians(3.0 * k)
        ca = math.cos(ang); sa = math.sin(ang)
        a = [p[0] * ca + p[1] * sa for p in topp]
        bb = [-p[0] * sa + p[1] * ca for p in topp]
        a0 = quantile(a, 0.03); a1 = quantile(a, 0.97)
        b0 = quantile(bb, 0.03); b1 = quantile(bb, 0.97)
        area = (a1 - a0) * (b1 - b0)
        if best is None or area < best[0]:
            best = (area, 3.0 * k, a0, a1, b0, b1)
    area, theta, a0, a1, b0, b1 = best
    ang = math.radians(theta)
    am = 0.5 * (a0 + a1); bm = 0.5 * (b0 + b1)
    cx = am * math.cos(ang) - bm * math.sin(ang)
    cy = am * math.sin(ang) + bm * math.cos(ang)
    return {"x": cx, "y": cy, "top": top, "bottom": bottom, "theta": theta,
            "wa": a1 - a0, "wb": b1 - b0, "n": len(pts), "ntop": len(topp)}


all_cands = []
for cam in ("wrist", "front"):
    if cam in live_cams:
        try:
            all_cands.extend(container_candidates(cam))
        except Exception as exc:
            report["errors"].append("segment " + cam + ": " + str(exc))

# convention calibration: project the live clouds back into their image and test the box
calib = {}
for cam in ("front", "wrist"):
    if cam not in live_cams:
        continue
    camobs = obs.cameras[cam]
    scores = {}
    for conv in ("opencv", "opengl", "ros"):
        inside = 0; total = 0
        for cand in all_cands:
            if cand["cam"] != cam or cand["box"] is None:
                continue
            x1, y1, x2, y2 = cand["box"]
            step_n = max(1, len(cand["pts"]) // 40)
            for p in cand["pts"][::step_n]:
                uv = project(p, camobs.intrinsics, camobs.camera_pose, conv)
                total += 1
                if uv is not None and x1 - 3 <= uv[0] <= x2 + 3 and y1 - 3 <= uv[1] <= y2 + 3:
                    inside += 1
        scores[conv] = (inside / float(total)) if total > 0 else 0.0
    bestconv = "opencv"
    for conv in scores:
        if scores[conv] > scores[bestconv]:
            bestconv = conv
    calib[cam] = {"conv": bestconv, "scores": scores}
report["calibration"] = calib

containers = []
for cand in all_cands:
    dsc = describe(cand["pts"])
    span = max(dsc["wa"], dsc["wb"])
    if span > 0.12 or span < 0.015:
        continue
    if dsc["top"] - dsc["bottom"] > 0.20:
        continue
    merged = False
    for ct in containers:
        if math.hypot(ct["x"] - dsc["x"], ct["y"] - dsc["y"]) < 0.025:
            merged = True
            if ct["cam"] != "wrist" and cand["cam"] == "wrist":
                pass
            break
    if not merged:
        dsc["cam"] = cand["cam"]
        dsc["score"] = cand["score"]
        containers.append(dsc)
report["containers"] = [{"x": round(c["x"], 4), "y": round(c["y"], 4), "top": round(c["top"], 4),
                         "bottom": round(c["bottom"], 4), "theta": c["theta"], "wa": round(c["wa"], 4),
                         "wb": round(c["wb"], 4), "cam": c["cam"], "n": c["n"]} for c in containers]

# ---------------------------------------------------------------- 2b. point-prompt localization
def locate_by_point(cam, xyz, obs_now, max_dist=0.035):
    """Project an expected container-top point into `cam`, segment at that pixel, return describe() or None."""
    try:
        camobs = obs_now.cameras[cam]
    except Exception:
        return None, "no-cam"
    conv = calib[cam]["conv"] if cam in calib else "opencv"
    uv = project(xyz, camobs.intrinsics, camobs.camera_pose, conv)
    if uv is None:
        return None, "behind"
    h = len(camobs.depth_m)
    w = len(camobs.depth_m[0])
    if uv[0] < 4 or uv[1] < 4 or uv[0] > w - 5 or uv[1] > h - 5:
        return None, "off-image %.0f,%.0f" % (uv[0], uv[1])
    seg = segment_points(cam, [[int(round(uv[0])), int(round(uv[1]))]])
    if not seg.ok:
        return None, "seg-failed"
    best = None
    for cand in seg.segmentations[:3]:
        try:
            pc = mask_to_point_cloud(cand, cam, target_frame=base_frame)
        except Exception:
            continue
        pts = [(float(p[0]), float(p[1]), float(p[2])) for p in pc.points]
        if len(pts) < 15:
            continue
        dsc = describe(pts)
        span = max(dsc["wa"], dsc["wb"])
        if span > 0.08 or span < 0.02:
            continue
        dist = math.hypot(dsc["x"] - xyz[0], dsc["y"] - xyz[1])
        if dist <= max_dist and (best is None or dist < best[0]):
            best = (dist, dsc)
    if best is None:
        return None, "no-match"
    best[1]["cam"] = cam + "-point"
    return best[1], "ok %.3f" % best[0]


# ---------------------------------------------------------------- 3. deproject video blobs, cluster cubes
dets = []
for blob in raw_blobs:
    c, cam, fidx, u, v, d, npix = blob
    if d <= 0.05 or d > 3.0:
        continue
    conv = "opencv"
    if cam in calib:
        conv = calib[cam]["conv"]
    camobs = video[fidx].cameras[cam]
    p = deproject(u, v, d, camobs.intrinsics, camobs.camera_pose, conv)
    dets.append((c, p, fidx, cam, npix))
zs = [d[1][2] for d in dets]
z_ref = quantile(zs, 0.5) if len(zs) > 0 else None
report["det_count"] = len(dets)
report["det_z_ref"] = z_ref
cube_clusters = []
for det in dets:
    c, p, fidx, cam, npix = det
    if z_ref is not None and abs(p[2] - z_ref) > 0.05:
        continue
    placed = False
    for cl in cube_clusters:
        if cl["color"] == c and math.hypot(cl["x"] - p[0], cl["y"] - p[1]) < 0.03:
            n = cl["n"]
            cl["x"] = (cl["x"] * n + p[0]) / (n + 1)
            cl["y"] = (cl["y"] * n + p[1]) / (n + 1)
            cl["z"] = (cl["z"] * n + p[2]) / (n + 1)
            cl["n"] = n + 1
            cl["frames"].add(fidx)
            cl["cams"].add(cam)
            placed = True
            break
    if not placed:
        cube_clusters.append({"color": c, "x": p[0], "y": p[1], "z": p[2], "n": 1,
                              "frames": set([fidx]), "cams": set([cam])})
cubes = [cl for cl in cube_clusters if len(cl["frames"]) >= 3]
report["cubes"] = [{"color": cl["color"], "x": round(cl["x"], 4), "y": round(cl["y"], 4),
                    "z": round(cl["z"], 4), "n": cl["n"], "nframes": len(cl["frames"]),
                    "first_frame": min(cl["frames"]), "cams": sorted(cl["cams"])} for cl in cube_clusters]

# ---------------------------------------------------------------- 4. build pick plan
used = set()
plan = []
med_top = quantile([ct["top"] for ct in containers], 0.5) if len(containers) > 0 else None
med_bottom = quantile([ct["bottom"] for ct in containers], 0.5) if len(containers) > 0 else None
med_w = quantile([0.5 * (ct["wa"] + ct["wb"]) for ct in containers], 0.5) if len(containers) > 0 else 0.05
for item in goal_seq:
    cands = [cl for cl in cubes if cl["color"] == item["color"]]
    cands = sorted(cands, key=lambda cl: -len(cl["frames"]))
    if not item["all"]:
        cands = cands[:1]
    for cl in cands:
        best = None
        for idx in range(len(containers)):
            ct = containers[idx]
            dist = math.hypot(ct["x"] - cl["x"], ct["y"] - cl["y"])
            if best is None or dist < best[0]:
                best = (dist, idx)
        if best is not None and best[0] <= 0.045:
            if best[1] in used:
                continue  # duplicate cluster of a cube already planned
            used.add(best[1])
            plan.append({"color": cl["color"], "container": best[1], "cube": (cl["x"], cl["y"]),
                         "match_dist": round(best[0], 4)})
            continue
        top_guess = med_top if med_top is not None else (cl["z"] + 0.045)
        found = None
        tries = []
        for cam in ("wrist", "front"):
            if found is None and cam in live_cams:
                try:
                    found, msg = locate_by_point(cam, (cl["x"], cl["y"], top_guess), obs, 0.04)
                except Exception as exc:
                    found, msg = None, "error " + str(exc)
                tries.append(cam + ":" + msg)
        if found is not None:
            containers.append(found)
            used.add(len(containers) - 1)
            plan.append({"color": cl["color"], "container": len(containers) - 1, "cube": (cl["x"], cl["y"]),
                         "match_dist": round(math.hypot(found["x"] - cl["x"], found["y"] - cl["y"]), 4),
                         "point_loc": tries})
            continue
        if med_top is None:
            plan.append({"color": cl["color"], "container": None, "cube": (cl["x"], cl["y"]), "match_dist": None,
                         "point_loc": tries})
            continue
        containers.append({"x": cl["x"], "y": cl["y"], "top": med_top, "bottom": med_bottom, "theta": 0.0,
                           "wa": med_w, "wb": med_w, "n": 0, "ntop": 0, "cam": "pseudo-from-cube", "score": 0.0})
        used.add(len(containers) - 1)
        plan.append({"color": cl["color"], "container": len(containers) - 1, "cube": (cl["x"], cl["y"]),
                     "match_dist": None if best is None else round(best[0], 4), "pseudo": True, "point_loc": tries})
report["plan"] = [{"color": p["color"], "container": p["container"], "match_dist": p["match_dist"],
                   "src": (containers[p["container"]]["cam"] if p["container"] is not None else None),
                   "cube": [round(p["cube"][0], 4), round(p["cube"][1], 4)]} for p in plan]

# ---------------------------------------------------------------- 5. motion helpers
Q_DOWN = (0.0, 1.0, 0.0, 0.0)


def q_for_yaw(psi_deg):
    h = math.radians(psi_deg) / 2.0
    return qmul((math.cos(h), 0.0, 0.0, math.sin(h)), Q_DOWN)


q0 = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]
xa = qrot(q0, (1.0, 0.0, 0.0))
za = qrot(q0, (0.0, 0.0, 1.0))
psi_start = math.degrees(math.atan2(xa[1], xa[0]))
report["start_ee"] = {"pos": [round(float(v), 4) for v in state0.end_effector_poses["primary"].position],
                      "psi": round(psi_start, 1), "z_axis_z": round(za[2], 3),
                      "gripper": float(state0.gripper_positions["primary"])}


def ended(res):
    try:
        return bool(res.terminated) or bool(res.truncated)
    except Exception:
        return False


log = []
status = {"ended": False}


def note(tag, res):
    entry = {"tag": tag}
    try:
        entry["ok"] = bool(res.ok)
        entry["status"] = str(res.status)
        entry["steps"] = int(res.steps_executed)
    except Exception:
        pass
    if ended(res):
        entry["ended"] = True
        status["ended"] = True
    log.append(entry)
    return res


def go(pos, psi, tag, tol=0.01, steps=200):
    res = move_to_pose(Pose((pos[0], pos[1], pos[2]), q_for_yaw(psi), base_frame), tolerance=tol, max_steps=steps)
    return note(tag, res)


def choose_psi(ct, others, offset=0.0):
    """Square container: pick the face-aligned yaw closest to the start yaw (+offset)."""
    ref = psi_start + offset
    best = None
    for k in range(4):
        axis = ct["theta"] + 90.0 * k  # finger closing axis (deg)
        psi = axis + 90.0
        while psi - ref > 45.0:
            psi -= 90.0
        while psi - ref <= -45.0:
            psi += 90.0
        width = ct["wa"] if k % 2 == 0 else ct["wb"]
        ax = math.cos(math.radians(psi - 90.0)); ay = math.sin(math.radians(psi - 90.0))
        clear = 1.0
        for o in others:
            for sgn in (-1.0, 1.0):
                fx = ct["x"] + sgn * ax * (0.5 * width + 0.02)
                fy = ct["y"] + sgn * ay * (0.5 * width + 0.02)
                clear = min(clear, math.hypot(o["x"] - fx, o["y"] - fy))
        key = -abs(psi - ref)
        if best is None or key > best[0]:
            best = (key, psi, width, clear)
    return best[1], best[2], best[3]


def refine(ct):
    try:
        obs_now = get_observation()
        dsc, msg = locate_by_point("wrist", (ct["x"], ct["y"], ct["top"]), obs_now, 0.03)
        if dsc is not None:
            return dsc, "point " + msg
    except Exception as exc:
        msg = "point-error " + str(exc)
    try:
        cands = container_candidates("wrist")
    except Exception:
        return ct, "refine-error"
    best = None
    for cand in cands:
        dsc = describe(cand["pts"])
        if max(dsc["wa"], dsc["wb"]) > 0.12:
            continue
        dist = math.hypot(dsc["x"] - ct["x"], dsc["y"] - ct["y"])
        if dist < 0.03 and (best is None or dist < best[0]):
            best = (dist, dsc)
    if best is None:
        return ct, "refine-none"
    dsc = best[1]
    dsc["cam"] = "wrist-refined"
    return dsc, "refined %.3f" % best[0]


# ---------------------------------------------------------------- 6. execute picks
def hover_to(ct, others, k):
    """Try hover poses: two heights x yaw offsets; return (ok, psi)."""
    for offset in (0.0, 90.0, -90.0):
        psi, width, clear = choose_psi(ct, others, offset)
        for dz in (0.15, 0.10):
            res = go((ct["x"], ct["y"], ct["top"] + dz), psi, "hover%d_%d_%d" % (k, int(offset), int(dz * 100)),
                     tol=0.015, steps=250)
            if status["ended"]:
                return False, psi
            if res.ok:
                return True, psi
    return False, None


def grasp_once(ct, psi, k, tag):
    height = ct["top"] - ct["bottom"]
    depth = min(0.035, max(0.02, 0.4 * height))
    grasp_z = ct["top"] - depth
    r1 = go((ct["x"], ct["y"], ct["top"] + 0.05), psi, "pre%d%s" % (k, tag), tol=0.01, steps=200)
    if status["ended"] or not r1.ok:
        return "pre-failed", grasp_z
    r2 = go((ct["x"], ct["y"], grasp_z), psi, "grasp%d%s" % (k, tag), tol=0.008, steps=150)
    if status["ended"]:
        return "ended", grasp_z
    if not r2.ok:
        go((ct["x"], ct["y"], ct["top"] + 0.05), psi, "backoff%d%s" % (k, tag), tol=0.015, steps=150)
        return "descent-failed", grasp_z
    note("close%d%s" % (k, tag), close_gripper())
    if status["ended"]:
        return "ended", grasp_z
    g = 1.0
    try:
        g = float(get_robot_state().gripper_positions["primary"])
    except Exception:
        pass
    if g < 0.15:
        note("reopen%d%s" % (k, tag), open_gripper())
        go((ct["x"], ct["y"], ct["top"] + 0.05), psi, "backoff%d%s" % (k, tag), tol=0.015, steps=150)
        return "empty %.3f" % g, grasp_z
    return "held %.3f" % g, grasp_z


picks = []
open_res = note("open", open_gripper())
n_plan = len(plan)
for k in range(n_plan):
    if status["ended"]:
        break
    item = plan[k]
    final = (k == n_plan - 1)
    entry = {"color": item["color"], "container": item["container"], "attempts": []}
    picks.append(entry)
    if item["container"] is None:
        entry["skipped"] = "no container match"
        report["errors"].append("no container for " + item["color"])
        continue
    ct = containers[item["container"]]
    others = [containers[j] for j in range(len(containers)) if j != item["container"]]
    held = False
    for attempt in range(2):
        ok, psi = hover_to(ct, others, k)
        if status["ended"]:
            break
        if not ok:
            entry["attempts"].append("hover-failed")
            report["errors"].append("hover failed for pick %d" % k)
            break
        ct2, rmsg = refine(ct)
        entry["attempts"].append("refine " + rmsg)
        if ct2 is not ct:
            ct2["bottom"] = min(ct["bottom"], ct2["bottom"])
            ct = ct2
        psi2, width, clear = choose_psi(ct, others, 0.0)
        if abs(psi2 - psi) <= 46.0:
            psi = psi2
        entry["target"] = [round(ct["x"], 4), round(ct["y"], 4), round(ct["top"], 4)]
        entry["psi"] = round(psi, 1); entry["width"] = round(width, 4); entry["clear"] = round(clear, 4)
        outcome, grasp_z = grasp_once(ct, psi, k, "a%d" % attempt)
        entry["attempts"].append(outcome)
        entry["grasp_z"] = round(grasp_z, 4)
        if status["ended"]:
            break
        if outcome.startswith("held"):
            held = True
            break
    if status["ended"] or not held:
        if not held and not status["ended"]:
            report["errors"].append("pick %d not held" % k)
        continue
    go((ct["x"], ct["y"], grasp_z + 0.12), psi, "lift%d" % k, tol=0.015, steps=200)
    if status["ended"]:
        entry["ended_after_lift"] = True
        break
    try:
        entry["grip_after_lift"] = round(float(get_robot_state().gripper_positions["primary"]), 4)
    except Exception:
        pass
    if not final:
        go((ct["x"], ct["y"], grasp_z + 0.004), psi, "lower%d" % k, tol=0.01, steps=200)
        if status["ended"]:
            break
        note("release%d" % k, open_gripper())
        if status["ended"]:
            break
        go((ct["x"], ct["y"], ct["top"] + 0.10), psi, "retreat%d" % k, tol=0.015, steps=200)

report["picks"] = picks
report["motion_log"] = log
report["ended"] = status["ended"]
try:
    if not status["ended"]:
        report["final_time_s"] = float(get_robot_state().timestamp_s)
except Exception:
    pass
result = report
