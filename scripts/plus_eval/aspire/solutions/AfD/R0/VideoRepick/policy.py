# VideoRepick v01 -- point-ownership tracker (v00 + tracker change); initial program from offline analysis of the released recordings (AfD).
# Strategy (everything episode-dependent is derived at runtime):
#   1. Parse the repetition count from get_task_context().language ("once"/"twice"/"N times").
#   2. Calibrate the camera back-projection convention against mask_to_point_cloud on the live
#      front image, then back-project depth of every conditioning-video frame (front + wrist) into
#      robot_base; above-table connected components are object detections (colour-free, so it
#      works for green rings, purple stars under moving pink light, etc.).
#   3. Identify the object the video robot picked: gripper close/open events in the video's own
#      robot_state give the grasp and release end-effector xy; fallback = most displaced detection
#      between the first frame and the last pre-shuffle frame.
#   4. Track all detections in robot_base xy (greedy nearest-neighbour with constant-velocity
#      prediction and coasting through occlusion) to the end of the video, then to the live frame.
#   5. Top-down pick and place-down of that object the parsed number of times, re-localising it
#      from the wrist camera before every grasp; then press the button located by segmentation.
import math

report = {"version": "v01", "events": []}


def note(key, value):
    report[key] = value


def log(msg):
    if len(report["events"]) < 200:
        report["events"].append(str(msg))


# ----------------------------------------------------------------------------- goal parsing
def parse_repetitions(text):
    words = text.lower().replace(",", " ").replace(".", " ").split()
    names = {"once": 1, "twice": 2, "thrice": 3}
    numbers = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
    for i, w in enumerate(words):
        if w in names:
            return names[w]
        if w == "times" and i > 0 and words[i - 1] in numbers:
            return numbers[words[i - 1]]
    return 1


# ----------------------------------------------------------------------------- geometry helpers
def quat_to_matrix(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def to_rows(array):
    try:
        return array.tolist()
    except Exception:
        return array


def camera_model(cam, convention):
    k = to_rows(cam.intrinsics)
    pose = cam.camera_pose
    return {"fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]), "cy": float(k[1][2]),
            "R": quat_to_matrix(pose.quaternion_wxyz), "t": [float(v) for v in pose.position],
            "conv": convention}


def backproject(model, u, v, d):
    a = (u - model["cx"]) / model["fx"] * d
    b = (v - model["cy"]) / model["fy"] * d
    if model["conv"] == "opencv":
        pc = (a, b, d)
    else:  # opengl: x right, y up, camera looks along -z
        pc = (a, -b, -d)
    R = model["R"]
    t = model["t"]
    return (R[0][0] * pc[0] + R[0][1] * pc[1] + R[0][2] * pc[2] + t[0],
            R[1][0] * pc[0] + R[1][1] * pc[1] + R[1][2] * pc[2] + t[1],
            R[2][0] * pc[0] + R[2][1] * pc[1] + R[2][2] * pc[2] + t[2])


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def dist2(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def seg_list(found):
    try:
        if found.ok:
            return list(found.segmentations)
    except Exception:
        pass
    return []


def cloud_of(segmentation, camera_name, base_frame):
    try:
        cloud = mask_to_point_cloud(segmentation, camera_name, target_frame=base_frame)
        pts = list(cloud.points)
        if len(pts) > 0:
            return pts
    except Exception:
        pass
    try:
        cloud = mask_to_point_cloud(segmentation.mask, camera_name, target_frame=base_frame)
        return list(cloud.points)
    except Exception:
        return []


# ----------------------------------------------------------------------------- calibration
def calibrate_convention(camera_name, base_frame):
    """Pick the back-projection convention whose box-centre projection matches the public
    mask_to_point_cloud() median.  Falls back to 'the optical axis points at the table'."""
    obs = get_observation()
    if camera_name not in obs.cameras:
        return None
    cam = obs.cameras[camera_name]
    depth = to_rows(cam.depth_m)
    errors = {"opencv": [], "opengl": []}
    for prompt in ("table", "object", "button"):
        for seg in seg_list(segment_text(camera_name, prompt))[:3]:
            pts = cloud_of(seg, camera_name, base_frame)
            if len(pts) < 10 or seg.box_xyxy is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in seg.box_xyxy)
            u = int((x1 + x2) / 2.0)
            v = int((y1 + y2) / 2.0)
            if v < 0 or u < 0 or v >= len(depth) or u >= len(depth[0]):
                continue
            d = float(depth[v][u])
            if not d > 0.0:
                continue
            ref = (quantile([p[0] for p in pts], 0.5), quantile([p[1] for p in pts], 0.5),
                   quantile([p[2] for p in pts], 0.5))
            for conv in ("opencv", "opengl"):
                p = backproject(camera_model(cam, conv), u, v, d)
                errors[conv].append(math.sqrt((p[0] - ref[0]) ** 2 + (p[1] - ref[1]) ** 2 + (p[2] - ref[2]) ** 2))
    if len(errors["opencv"]) > 0:
        e_cv = quantile(errors["opencv"], 0.5)
        e_gl = quantile(errors["opengl"], 0.5)
        log("calib " + camera_name + " opencv=" + str(round(e_cv, 3)) + " opengl=" + str(round(e_gl, 3)))
        return "opencv" if e_cv <= e_gl else "opengl"
    R = quat_to_matrix(cam.camera_pose.quaternion_wxyz)
    log("calib " + camera_name + " fallback axis test")
    return "opencv" if R[2][2] < 0.0 else "opengl"


# ----------------------------------------------------------------------------- detection
DET = {"stride": {"front": 2, "wrist": 3}, "min_h": 0.012, "max_h": 0.12, "max_extent": 0.09,
       "min_pts": 4, "merge": 0.025}


def foreground(observation, camera_name, convention):
    """Above-table pixels of one camera: dict (u, v) -> robot_base xyz, plus the table z."""
    if convention is None or camera_name not in observation.cameras:
        return {}, None
    cam = observation.cameras[camera_name]
    model = camera_model(cam, convention)
    depth = to_rows(cam.depth_m)
    stride = DET["stride"][camera_name]
    pts = {}
    zs = []
    for v in range(0, len(depth), stride):
        row = depth[v]
        for u in range(0, len(row), stride):
            d = float(row[u])
            if d > 0.0 and d < 5.0:
                p = backproject(model, u, v, d)
                pts[(u, v)] = p
                zs.append(p[2])
    if len(zs) < 50:
        return {}, None
    table_z = quantile(zs, 0.4)
    fg = {}
    for key in pts:
        dz = pts[key][2] - table_z
        if dz > DET["min_h"] and dz < DET["max_h"]:
            fg[key] = pts[key]
    return fg, table_z


def components(fg, stride, table_z, camera_name):
    seen = set()
    comps = []
    for key in fg:
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        members = []
        while len(stack) > 0:
            cur = stack.pop()
            members.append(fg[cur])
            for du in (-stride, 0, stride):
                for dv in (-stride, 0, stride):
                    nb = (cur[0] + du, cur[1] + dv)
                    if nb in fg and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        if len(members) < DET["min_pts"]:
            continue
        xs = [m[0] for m in members]
        ys = [m[1] for m in members]
        ext = max(max(xs) - min(xs), max(ys) - min(ys))
        if ext > DET["max_extent"]:
            continue
        comps.append({"xy": (sum(xs) / len(xs), sum(ys) / len(ys)), "top": max(m[2] for m in members),
                      "n": len(members), "ext": ext, "table_z": table_z, "cam": camera_name})
    return comps


def detect(observation, camera_name, convention):
    fg, table_z = foreground(observation, camera_name, convention)
    if table_z is None:
        return []
    return components(fg, DET["stride"][camera_name], table_z, camera_name)


def detect_all(observation, conventions):
    dets = []
    for cam_name in ("front", "wrist"):
        for det in detect(observation, cam_name, conventions.get(cam_name)):
            merged = False
            for other in dets:
                if dist2(other["xy"], det["xy"]) < DET["merge"]:
                    other["xy"] = ((other["xy"][0] + det["xy"][0]) / 2.0, (other["xy"][1] + det["xy"][1]) / 2.0)
                    other["top"] = max(other["top"], det["top"])
                    merged = True
                    break
            if not merged:
                dets.append(det)
    return dets


def fg_points_all(observation, conventions):
    """Pooled above-table xy points of both cameras, minus very large blobs (robot arm, fixtures)."""
    pts = []
    for cam_name in ("front", "wrist"):
        fg, table_z = foreground(observation, cam_name, conventions.get(cam_name))
        if table_z is None:
            continue
        stride = DET["stride"][cam_name]
        seen = set()
        for key in fg:
            if key in seen:
                continue
            stack = [key]
            seen.add(key)
            members = []
            while len(stack) > 0:
                cur = stack.pop()
                members.append(fg[cur])
                for du in (-stride, 0, stride):
                    for dv in (-stride, 0, stride):
                        nb = (cur[0] + du, cur[1] + dv)
                        if nb in fg and nb not in seen:
                            seen.add(nb)
                            stack.append(nb)
            xs = [m[0] for m in members]
            ys = [m[1] for m in members]
            # keep blobs up to ~3 touching objects; drop arm-sized blobs
            if max(max(xs) - min(xs), max(ys) - min(ys)) <= 2.5 * DET["max_extent"]:
                for m in members:
                    pts.append((m[0], m[1]))
    return pts


# ----------------------------------------------------------------------------- tracker
# Point-ownership tracker: every track owns the pooled foreground points nearer to its predicted
# centre than to any other track's (within radius), new centre = mean of owned points, 3
# iterations.  Unlike nearest-blob assignment it keeps identities through touching/merged blobs
# (offline: blob-nearest swapped identity on ep0, ownership did not).
TRK = {"radius_scale": 0.8, "min_frac": 0.15, "iters": 3}


def make_tracks(dets):
    return [{"xy": d["xy"], "vel": (0.0, 0.0), "coast": 0, "xy0": d["xy"], "coast_total": 0} for d in dets]


def track_params(dets):
    exts = [d["ext"] for d in dets if "ext" in d]
    ns = [d["n"] for d in dets]
    diam = quantile(exts, 0.5) if len(exts) > 0 else 0.04
    return {"radius": max(0.012, TRK["radius_scale"] * diam), "diam": diam}


def update_tracks(tracks, pts, params):
    if len(tracks) == 0:
        return tracks
    pred = [(t["xy"][0] + t["vel"][0], t["xy"][1] + t["vel"][1]) for t in tracks]
    centres = list(pred)
    counts = [0] * len(tracks)
    expected = params.get("expected_pts", 0)
    for it in range(TRK["iters"]):
        sums = [[0.0, 0.0, 0] for t in tracks]
        for p in pts:
            best = -1
            best_d = 1e9
            for i in range(len(centres)):
                c = centres[i]
                dd = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2
                if dd < best_d:
                    best_d = dd
                    best = i
            lim = params["radius"] + 0.004 * min(tracks[best]["coast"], 10)
            if best >= 0 and best_d <= lim * lim:
                sums[best][0] = sums[best][0] + p[0]
                sums[best][1] = sums[best][1] + p[1]
                sums[best][2] = sums[best][2] + 1
        new_centres = []
        for i in range(len(tracks)):
            counts[i] = sums[i][2]
            if counts[i] >= max(2, int(TRK["min_frac"] * expected)):
                new_centres.append((sums[i][0] / counts[i], sums[i][1] / counts[i]))
            else:
                new_centres.append(pred[i])
        centres = new_centres
    for i, t in enumerate(tracks):
        if counts[i] >= max(2, int(TRK["min_frac"] * expected)):
            step_v = (centres[i][0] - t["xy"][0], centres[i][1] - t["xy"][1])
            t["vel"] = (0.5 * t["vel"][0] + 0.5 * step_v[0], 0.5 * t["vel"][1] + 0.5 * step_v[1])
            t["coast"] = 0
        else:
            t["vel"] = (0.6 * t["vel"][0], 0.6 * t["vel"][1])
            t["coast"] = t["coast"] + 1
            t["coast_total"] = t["coast_total"] + 1
        t["xy"] = centres[i]
    return tracks


def init_tracker(observation, conventions):
    dets = detect_all(observation, conventions)
    params = track_params(dets)
    tracks = make_tracks(dets)
    pts = fg_points_all(observation, conventions)
    params["expected_pts"] = len(pts) / max(1, len(tracks))
    return tracks, params


# ----------------------------------------------------------------------------- video analysis
def gripper_events(video):
    """Return (grasp_index, release_index) from the video's own robot_state, or None."""
    g = []
    for frame in video:
        try:
            g.append(float(frame.robot_state.gripper_positions["primary"]))
        except Exception:
            g.append(None)
    vals = [x for x in g if x is not None]
    if len(vals) < 3:
        return None
    g_open = max(vals)
    if g_open - min(vals) < 0.15:
        return None
    grasp = None
    release = None
    for i, x in enumerate(g):
        if x is None:
            continue
        if grasp is None and x < g_open - 0.15:
            grasp = i
        elif grasp is not None and x > g_open - 0.05:
            release = i
            break
    if grasp is None or release is None:
        return None
    return (grasp, release)


def ee_xy(frame):
    try:
        p = frame.robot_state.end_effector_poses["primary"].position
        return (float(p[0]), float(p[1]), float(p[2]))
    except Exception:
        return None


def nearest_track(tracks, xy):
    best = None
    for k in range(len(tracks)):
        if best is None or dist2(tracks[k]["xy"], xy) < dist2(tracks[best]["xy"], xy):
            best = k
    return best


def identify_and_track(video, conventions, live_obs):
    n = len(video)
    info = {"frames": n}
    events = gripper_events(video)
    info["gripper_events"] = events
    target = None
    start = 0
    tracks = []
    params = {}
    if events is not None:
        ge = ee_xy(video[events[0]])
        re = ee_xy(video[events[1]])
        if ge is not None and re is not None:
            info["grasp_xy"] = [round(ge[0], 4), round(ge[1], 4)]
            info["place_xy"] = [round(re[0], 4), round(re[1], 4)]
            start = events[1]
            for i in range(events[1], n):
                e = ee_xy(video[i])
                if e is not None and (e[2] - re[2] > 0.08 or dist2(e, re) > 0.10):
                    start = i
                    break
            tracks, params = init_tracker(video[start], conventions)
            if len(tracks) > 0:
                target = nearest_track(tracks, (re[0], re[1]))
                info["target_init_err"] = round(dist2(tracks[target]["xy"], re), 4)
                if info["target_init_err"] > max(0.05, 1.5 * params["diam"]):
                    target = None
    if target is None:
        # Fallback without gripper events: track from frame 0; the picked object is the track that
        # coasted (was lifted/hidden) longest before >=2 tracks moved by > 1 diameter (shuffle).
        tracks, params = init_tracker(video[0], conventions)
        if len(tracks) == 0:
            info["identify"] = "failed"
            return None, info
        start = n - 1
        for i in range(1, n):
            tracks = update_tracks(tracks, fg_points_all(video[i], conventions), params)
            moved = [t for t in tracks if t["coast"] == 0 and dist2(t["xy"], t["xy0"]) > params["diam"]]
            if len(moved) >= 2:
                start = i
                break
        target = 0
        for k in range(len(tracks)):
            if tracks[k]["coast_total"] > tracks[target]["coast_total"]:
                target = k
        info["identify"] = "coast-fallback@" + str(start) + " coast=" + str(tracks[target]["coast_total"])
    else:
        info["identify"] = "gripper-events@" + str(start)
    info["n_tracks"] = len(tracks)
    info["radius"] = round(params["radius"], 4)
    for i in range(start + 1, n):
        tracks = update_tracks(tracks, fg_points_all(video[i], conventions), params)
    tracks = update_tracks(tracks, fg_points_all(live_obs, conventions), params)
    info["target_coast"] = tracks[target]["coast"]
    info["target_xy"] = [round(v, 4) for v in tracks[target]["xy"]]
    info["all_xy"] = [[round(v, 3) for v in t["xy"]] for t in tracks]
    return tracks[target]["xy"], info

# ----------------------------------------------------------------------------- motion helpers
def terminal(res):
    try:
        if res.terminated or res.truncated:
            return True
    except Exception:
        pass
    return False


def down_quat():
    q = get_robot_state().end_effector_poses["primary"].quaternion_wxyz
    R = quat_to_matrix(q)
    if R[2][2] < -0.9:
        return [float(v) for v in q]
    return [0.0, 1.0, 0.0, 0.0]


def go(xyz, quat, base_frame, tol=0.01, steps=200):
    res = move_to_pose(Pose([float(xyz[0]), float(xyz[1]), float(xyz[2])], quat, base_frame),
                       tolerance=tol, max_steps=steps)
    ok = False
    try:
        ok = bool(res.ok)
    except Exception:
        pass
    if terminal(res):
        state["done"] = True
    return ok


def relocalize(xy, conventions, radius):
    obs = get_observation()
    best = None
    for cam_name in ("wrist", "front"):
        for d in detect(obs, cam_name, conventions.get(cam_name)):
            dd = dist2(d["xy"], xy)
            if dd <= radius and (best is None or dd < best[0]):
                best = (dd, d)
        if best is not None:
            return best[1]
    return None


def find_button(base_frame, avoid_xy):
    for prompt in ("button", "white button", "gray round button"):
        for seg in seg_list(segment_text("front", prompt))[:4]:
            pts = cloud_of(seg, "front", base_frame)
            if len(pts) < 10:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if quantile(xs, 0.95) - quantile(xs, 0.05) > 0.12 or quantile(ys, 0.95) - quantile(ys, 0.05) > 0.12:
                continue
            c = (quantile(xs, 0.5), quantile(ys, 0.5))
            if avoid_xy is not None and dist2(c, avoid_xy) < 0.03:
                continue
            return (c[0], c[1], quantile([p[2] for p in pts], 0.95)), prompt
    return None, None


# ----------------------------------------------------------------------------- main
state = {"done": False}
goal = get_task_context().language
note("goal", goal)
reps = parse_repetitions(goal)
note("repetitions", reps)
try:
    meta = robomme_plus.get_task_metadata()
    note("max_steps", meta["max_steps"])
except Exception:
    note("max_steps", None)
base = get_robot_state().base_frame
live = get_observation()
conventions = {}
for cam_name in ("front", "wrist"):
    conventions[cam_name] = calibrate_convention(cam_name, base)
note("conventions", conventions)

video = robomme_plus.get_conditioning_video()
target_xy = None
if len(video) > 0:
    target_xy, info = identify_and_track(video, conventions, live)
    note("video", info)
else:
    note("video", {"frames": 0})

quat = down_quat()
cycles = 0
if target_xy is None:
    log("no target identified; falling back to nearest detection to the workspace centroid")
    dets = detect_all(live, conventions)
    if len(dets) > 0:
        cx = sum(d["xy"][0] for d in dets) / len(dets)
        cy = sum(d["xy"][1] for d in dets) / len(dets)
        best_det = dets[0]
        for d in dets:
            if dist2(d["xy"], (cx, cy)) < dist2(best_det["xy"], (cx, cy)):
                best_det = d
        target_xy = best_det["xy"]

if target_xy is not None:
    obj = relocalize(target_xy, conventions, 0.05)
    for rep in range(reps):
        if state["done"] or obj is None:
            break
        xy = obj["xy"]
        open_gripper()
        go((xy[0], xy[1], obj["top"] + 0.12), quat, base, tol=0.015, steps=250)
        refined = relocalize(xy, conventions, 0.04)
        if refined is not None:
            obj = refined
            xy = obj["xy"]
        grasp_z = max(obj["top"] - 0.02, obj["table_z"] + 0.012)
        go((xy[0], xy[1], grasp_z + 0.06), quat, base, tol=0.01, steps=150)
        go((xy[0], xy[1], grasp_z), quat, base, tol=0.008, steps=120)
        close_gripper()
        go((xy[0], xy[1], grasp_z + 0.10), quat, base, tol=0.015, steps=150)
        held = float(get_robot_state().gripper_positions["primary"])
        log("rep " + str(rep) + " gripper after lift " + str(round(held, 3)))
        if state["done"]:
            break
        go((xy[0], xy[1], grasp_z + 0.005), quat, base, tol=0.01, steps=150)
        open_gripper()
        go((xy[0], xy[1], grasp_z + 0.12), quat, base, tol=0.015, steps=150)
        cycles = cycles + 1
        obj = relocalize(xy, conventions, 0.05)

note("pick_place_cycles", cycles)
if not state["done"]:
    button, prompt = find_button(base, target_xy)
    note("button", None if button is None else [round(v, 4) for v in button])
    if button is not None:
        close_gripper()
        go((button[0], button[1], button[2] + 0.08), quat, base, tol=0.015, steps=200)
        go((button[0], button[1], button[2] - 0.015), quat, base, tol=0.01, steps=80)
        go((button[0], button[1], button[2] + 0.08), quat, base, tol=0.02, steps=80)
note("terminal_seen", state["done"])
result = report
