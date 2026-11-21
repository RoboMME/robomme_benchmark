# v02_multi_instance.py -- RoboMME-Plus VideoUnmaskSwap (task 15). UNTESTED Stage 1 repair (v01 + 1).
#
# v02 change (mode "perception: several cubes share a colour"): the v00 run's goal asks for "all
# containers hiding green cubes"; three green cubes were visible and v00's single-centroid colour
# estimate returned None for green, so no green container was planned. v02 finds every cube instance
# per colour, assigns instances to containers one-to-one, and plans all instances of a plural colour.
#
# Inherited v01 change:
# v01 change (mode "dynamics: shuffle continues live"): the v00 run showed the containers still
# moving in the live episode from tick 1 on, so v00's tick-0 localisation was stale when it grasped
# at tick 44. v01 keeps tracking every container in the live front view (arm held still) until the
# scene is static (or a tick budget is reached), then picks the container at its tracked position.
# Everything else is unchanged from v00.
#
# Snapshot AND attempt (R=1: one simulator start). Phases:
#   P0  context / metadata / cameras / robot-state snapshot
#   P1  SAM3 prompt probes on both cameras (diagnostic only)
#   P2  depth-projection convention self-calibration on the live front camera
#   P3  conditioning-video analysis (pure Python, subsampled):
#         - per frame: object points above the table (depth + calibrated camera pose)
#         - cubes of the goal colours: last position before they disappear
#         - containers: tracked backward from the live scene (and forward as a check)
#   P4  map each goal cube -> the container that covered it -> its live position
#   P5  top-down grasp + lift of each target container (in the language's order)
# Every phase is wrapped in try/except; the program never raises. Nothing below uses
# simulator state: every position comes from public RGB-D, calibration and robot state.
import math

report = {
    "version": "v02_multi_instance",
    "phase_status": {},
    "errors": [],
    "log": [],
}
result = report


def log(msg):
    text = str(msg)
    print("[v02] " + text)
    if len(report["log"]) < 400:
        report["log"].append(text[:400])


def record_error(phase, exc):
    text = phase + ": " + str(exc)
    print("[v02][ERROR] " + text)
    if len(report["errors"]) < 100:
        report["errors"].append(text[:600])


def rnd(value, digits=4):
    try:
        return round(float(value), digits)
    except Exception:
        return None


def vec(values, digits=4):
    out = []
    try:
        for value in values:
            out.append(rnd(value, digits))
    except Exception:
        return None
    return out


def as_list(arr):
    try:
        return arr.tolist()
    except Exception:
        pass
    try:
        return [list(row) for row in arr]
    except Exception:
        return None


def sub_grid(arr, stride):
    """Return a subsampled nested python list of a (H, W[, C]) array."""
    try:
        return arr[::stride, ::stride].tolist()
    except Exception:
        pass
    full = as_list(arr)
    if full is None:
        return None
    rows = []
    for row in full[::stride]:
        rows.append(row[::stride])
    return rows


def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = int((len(ordered) - 1) * fraction)
    return ordered[index]


def median(values):
    return quantile(values, 0.5)


# ----------------------------------------------------------------- quaternion helpers
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def quat_to_rot(q):
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def rot_apply(rot, v):
    return [
        rot[0][0] * v[0] + rot[0][1] * v[1] + rot[0][2] * v[2],
        rot[1][0] * v[0] + rot[1][1] * v[1] + rot[1][2] * v[2],
        rot[2][0] * v[0] + rot[2][1] * v[1] + rot[2][2] * v[2],
    ]


def wrap_half_pi(angle):
    """Wrap to [-pi/2, pi/2): a parallel jaw is symmetric under a pi rotation."""
    while angle >= math.pi / 2:
        angle -= math.pi
    while angle < -math.pi / 2:
        angle += math.pi
    return angle


# ----------------------------------------------------------------- P0 snapshot
state0 = None
base_frame = "robot_base"
language = ""
control_hz = 20.0
max_steps = None
embodiment = "robomme_plus"
try:
    ctx = get_task_context()
    language = str(ctx.language)
    report["task_language"] = language
    try:
        report["task_name"] = str(ctx.task_name)
        report["task_id"] = str(ctx.task_id)
        report["suite"] = str(ctx.suite)
        report["family"] = str(ctx.family)
    except Exception as exc:
        record_error("P0.ctx", exc)
    try:
        meta = robomme_plus.get_task_metadata()
        meta_out = {}
        for key in meta:
            meta_out[str(key)] = str(meta[key])[:300]
        report["task_metadata"] = meta_out
        try:
            max_steps = int(meta["max_steps"])
        except Exception as exc:
            record_error("P0.max_steps", exc)
        for key in ("control_hz", "control_frequency_hz", "control_freq", "frequency_hz"):
            try:
                control_hz = float(meta[key])
                break
            except Exception:
                pass
    except Exception as exc:
        record_error("P0.task_metadata", exc)
    try:
        cmeta = robomme_plus.get_controller_metadata()
        cm_out = {}
        for key in cmeta:
            cm_out[str(key)] = str(cmeta[key])[:300]
        report["controller_metadata"] = cm_out
    except Exception as exc:
        record_error("P0.controller_metadata", exc)
    state0 = get_robot_state()
    base_frame = str(state0.base_frame)
    embodiment = str(state0.embodiment)
    ee0 = state0.end_effector_poses["primary"]
    report["robot_state0"] = {
        "base_frame": base_frame,
        "embodiment": embodiment,
        "timestamp_s": rnd(state0.timestamp_s),
        "joints": vec(state0.joint_positions["primary"]),
        "gripper": rnd(state0.gripper_positions["primary"]),
        "ee_position": vec(ee0.position),
        "ee_quat_wxyz": vec(ee0.quaternion_wxyz),
        "ee_frame": str(ee0.frame),
    }
    report["max_steps"] = max_steps
    report["control_hz"] = control_hz
    log("language: " + language)
    log("max_steps=" + str(max_steps) + " hz=" + str(control_hz))
    report["phase_status"]["P0"] = "ok"
except Exception as exc:
    record_error("P0", exc)
    report["phase_status"]["P0"] = "failed"


def camera_summary(cam):
    out = {}
    try:
        out["rgb_shape"] = [int(v) for v in cam.rgb.shape]
    except Exception as exc:
        out["rgb_shape_error"] = str(exc)[:100]
    try:
        out["depth_shape"] = [int(v) for v in cam.depth_m.shape]
    except Exception as exc:
        out["depth_shape_error"] = str(exc)[:100]
    try:
        out["intrinsics"] = [vec(row) for row in as_list(cam.intrinsics)]
    except Exception as exc:
        out["intrinsics_error"] = str(exc)[:100]
    try:
        out["pose_position"] = vec(cam.camera_pose.position)
        out["pose_quat_wxyz"] = vec(cam.camera_pose.quaternion_wxyz)
        out["pose_frame"] = str(cam.camera_pose.frame)
    except Exception as exc:
        out["pose_error"] = str(exc)[:100]
    return out


def thumbnail_hex(cam, size):
    """Tiny RGB thumbnail as a hex string (evidence when no media is retained)."""
    try:
        h = int(cam.rgb.shape[0])
        stride = max(1, h // size)
        grid = sub_grid(cam.rgb, stride)
        digits = "0123456789abcdef"
        parts = []
        for row in grid[:size]:
            for px in row[:size]:
                for channel in px[:3]:
                    value = int(channel)
                    parts.append(digits[value // 16] + digits[value % 16])
        return {"size": size, "stride": stride, "hex": "".join(parts)}
    except Exception as exc:
        return {"error": str(exc)[:200]}


obs0 = None
camera_names = []
try:
    obs0 = get_observation()
    for name in obs0.cameras:
        camera_names.append(str(name))
    report["cameras"] = camera_names
    cams = {}
    for name in camera_names:
        cams[name] = camera_summary(obs0.cameras[name])
    report["camera_info"] = cams
    report["live_thumbs"] = {}
    for name in camera_names:
        report["live_thumbs"][name + "_start"] = thumbnail_hex(obs0.cameras[name], 48)
    report["phase_status"]["P0_obs"] = "ok"
except Exception as exc:
    record_error("P0.obs", exc)
    report["phase_status"]["P0_obs"] = "failed"

FRONT = "front"
WRIST = "wrist"
if camera_names:
    if FRONT not in camera_names:
        FRONT = camera_names[0]
    if WRIST not in camera_names:
        WRIST = camera_names[-1]


# ----------------------------------------------------------------- language parsing
COLOR_WORDS = ("red", "green", "blue", "yellow", "purple", "orange", "pink", "black",
               "white", "gray", "grey", "brown", "cyan", "magenta", "violet")
OBJECT_WORDS = ("cube", "cubes", "block", "blocks", "one", "box", "boxes")
target_colors = []
plural_colors = []
try:
    cleaned = ""
    for ch in language.lower():
        if ch.isalpha():
            cleaned += ch
        else:
            cleaned += " "
    tokens = cleaned.split()
    strict = []
    loose = []
    for index, token in enumerate(tokens):
        if token in COLOR_WORDS:
            word = token
            if word == "grey":
                word = "gray"
            if word == "violet":
                word = "purple"
            if word not in loose:
                loose.append(word)
            following = tokens[index + 1:index + 3]
            if any(f in OBJECT_WORDS for f in following) and word not in strict:
                strict.append(word)
    target_colors = strict if strict else loose
    # plural requests ("all containers hiding green cubes") -> every instance of that colour
    for index, token in enumerate(tokens):
        word = {"grey": "gray", "violet": "purple"}.get(token, token)
        if word in target_colors:
            before = tokens[max(0, index - 5):index]
            after = tokens[index + 1:index + 2]
            if "all" in before or "every" in before or "containers" in before or \
                    any(a in ("cubes", "blocks", "boxes") for a in after):
                if word not in plural_colors:
                    plural_colors.append(word)
    report["plural_colors"] = plural_colors
    report["language_tokens"] = tokens[:80]
    report["target_colors"] = target_colors
    report["target_colors_rule"] = "color+object-word" if strict else "any-color-word"
    put_down_words = [t for t in tokens if t in ("put", "place", "down", "return", "back")]
    report["put_down_words"] = put_down_words
    log("target colours: " + str(target_colors))
except Exception as exc:
    record_error("lang", exc)


# ----------------------------------------------------------------- P1 SAM3 probes
PROBE_PROMPTS = ["container", "cup", "bin", "box", "cube", "table"]
for color in target_colors:
    PROBE_PROMPTS.append(color + " cube")
report["prompt_probes"] = []
probe_cache = {}
for cam_name in (FRONT, WRIST):
    if cam_name not in camera_names:
        continue
    for prompt in PROBE_PROMPTS:
        probe = {"camera": cam_name, "prompt": prompt, "ok": False, "n": 0,
                 "scores": [], "boxes": []}
        try:
            seg = segment_text(cam_name, prompt)
            probe["ok"] = bool(seg.ok)
            if seg.ok:
                probe["n"] = len(seg.segmentations)
                for cand in seg.segmentations[:5]:
                    probe["scores"].append(rnd(cand.score, 3))
                    try:
                        probe["boxes"].append(vec(cand.box_xyxy, 1))
                    except Exception:
                        probe["boxes"].append(None)
                probe_cache[cam_name + "|" + prompt] = seg
        except Exception as exc:
            probe["error"] = str(exc)[:200]
        report["prompt_probes"].append(probe)
report["phase_status"]["P1"] = "done"


# ----------------------------------------------------------------- P2 projection model
CONVENTIONS = ("cv", "gl", "link")


def camera_model(cam):
    k = as_list(cam.intrinsics)
    pose = cam.camera_pose
    return {
        "fx": float(k[0][0]), "fy": float(k[1][1]),
        "cx": float(k[0][2]), "cy": float(k[1][2]),
        "pos": [float(v) for v in pose.position],
        "rot": quat_to_rot([float(v) for v in pose.quaternion_wxyz]),
        "h": int(cam.depth_m.shape[0]), "w": int(cam.depth_m.shape[1]),
    }


def cam_ray(conv, a, b):
    """Camera-frame ray with unit optical-axis component (z-depth scaling)."""
    if conv == "cv":
        return [a, b, 1.0]
    if conv == "gl":
        return [a, -b, -1.0]
    return [1.0, -a, -b]  # "link": x forward, y left, z up


def build_rays(model, conv, stride, use_range):
    rays = []
    for iy in range(0, model["h"], stride):
        row = []
        b = (iy + 0.5 - model["cy"]) / model["fy"]
        for ix in range(0, model["w"], stride):
            a = (ix + 0.5 - model["cx"]) / model["fx"]
            r = cam_ray(conv, a, b)
            if use_range:
                n = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
                r = [r[0] / n, r[1] / n, r[2] / n]
            row.append(rot_apply(model["rot"], r))
        rays.append(row)
    return rays


def project_to_pixel(model, conv, use_range, point):
    """World (base-frame) point -> (u, v) pixel, or None if behind the camera."""
    p = model["pos"]
    rot = model["rot"]
    d = [point[0] - p[0], point[1] - p[1], point[2] - p[2]]
    c = [rot[0][0] * d[0] + rot[1][0] * d[1] + rot[2][0] * d[2],
         rot[0][1] * d[0] + rot[1][1] * d[1] + rot[2][1] * d[2],
         rot[0][2] * d[0] + rot[1][2] * d[1] + rot[2][2] * d[2]]
    if conv == "cv":
        zf, a, b = c[2], c[0], c[1]
    elif conv == "gl":
        zf, a, b = -c[2], c[0], -c[1]
    else:
        zf, a, b = c[0], -c[1], -c[2]
    if zf <= 1e-6:
        return None
    a = a / zf
    b = b / zf
    return [a * model["fx"] + model["cx"] - 0.5, b * model["fy"] + model["cy"] - 0.5]


calib = {"conv": "cv", "use_range": False, "table_z": 0.0}
front_model = None
try:
    front_cam = obs0.cameras[FRONT]
    front_model = camera_model(front_cam)
    depth_grid = sub_grid(front_cam.depth_m, 8)
    scores = []
    for conv in CONVENTIONS:
        for use_range in (False, True):
            rays = build_rays(front_model, conv, 8, use_range)
            zs = []
            below = 0
            total = 0
            for iy in range(len(rays)):
                for ix in range(len(rays[iy])):
                    d = float(depth_grid[iy][ix])
                    if not (d > 0.05 and d < 5.0):
                        continue
                    z = front_model["pos"][2] + d * rays[iy][ix][2]
                    zs.append(z)
                    total += 1
                    if z < front_model["pos"][2]:
                        below += 1
            bins = {}
            for z in zs:
                key = int(math.floor(z / 0.005))
                bins[key] = bins.get(key, 0) + 1
            best_key = None
            best_count = 0
            for key in bins:
                count = bins[key] + bins.get(key - 1, 0) + bins.get(key + 1, 0)
                if count > best_count:
                    best_count = count
                    best_key = key
            frac = best_count / max(1, total)
            # the support surface: among strong horizontal planes (>= 40% of the mode),
            # prefer the one nearest the robot base height (a floor can out-count a table)
            table_key = best_key
            for key in bins:
                count = bins[key] + bins.get(key - 1, 0) + bins.get(key + 1, 0)
                if count >= 0.4 * best_count and abs(key) < abs(table_key):
                    table_key = key
            best_key = table_key
            below_frac = below / max(1, total)
            table_z = (best_key + 0.5) * 0.005 if best_key is not None else 0.0
            # refine the plane height: median of points within 1 cm of the mode
            near = [z for z in zs if abs(z - table_z) < 0.01]
            if near:
                table_z = median(near)
            scores.append({"conv": conv, "use_range": use_range, "plane_frac": rnd(frac, 3),
                           "below_frac": rnd(below_frac, 3), "table_z": rnd(table_z)})
    best = None
    for entry in scores:
        value = entry["plane_frac"] + (0.0 if entry["below_frac"] > 0.6 else -1.0)
        if best is None or value > best[0]:
            best = (value, entry)
    calib["conv"] = best[1]["conv"]
    calib["use_range"] = best[1]["use_range"]
    calib["table_z"] = best[1]["table_z"]
    report["calibration"] = {"candidates": scores, "chosen": best[1]}
    log("calibration chosen: " + str(best[1]))
    report["phase_status"]["P2"] = "ok"
except Exception as exc:
    record_error("P2", exc)
    report["phase_status"]["P2"] = "failed"

TABLE_Z = calib["table_z"]


# ----------------------------------------------------------------- pixel colour classes
def color_label(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 55:
        return "black"
    sat = (mx - mn) / mx
    if sat < 0.2:
        if mn >= 170:
            return "white"
        return "gray"
    if sat < 0.33:
        return "dull"
    delta = float(mx - mn)
    if mx == r:
        hue = 60.0 * (((g - b) / delta) % 6.0)
    elif mx == g:
        hue = 60.0 * (((b - r) / delta) + 2.0)
    else:
        hue = 60.0 * (((r - g) / delta) + 4.0)
    if hue < 0:
        hue += 360.0
    if hue < 18 or hue >= 335:
        return "red"
    if hue < 40:
        if mx < 150:
            return "brown"
        return "orange"
    if hue < 70:
        return "yellow"
    if hue < 170:
        return "green"
    if hue < 200:
        return "cyan"
    if hue < 255:
        return "blue"
    if hue < 295:
        return "purple"
    return "pink"


def color_matches(word, label):
    if word == label:
        return True
    if word == "magenta" and label in ("pink", "purple"):
        return True
    if word == "pink" and label == "purple":
        return True
    if word == "purple" and label == "pink":
        return True
    if word == "orange" and label == "brown":
        return True
    if word == "brown" and label == "orange":
        return True
    return False


# ----------------------------------------------------------------- per-frame point extraction
STRIDE = 3
H_OBJ_MIN = 0.008
H_OBJ_MAX = 0.40
H_CUBE_MAX = 0.10
H_TALL = 0.075
ray_cache = {}


def rays_for(model):
    key = str(vec(model["pos"], 3)) + str(vec(model["rot"][0], 3)) + str(vec(model["rot"][2], 3))
    if key not in ray_cache:
        ray_cache[key] = build_rays(model, calib["conv"], STRIDE, calib["use_range"])
    return ray_cache[key]


def extract_points(cam, ee_pos, want_colors):
    """Object points above the table in a camera frame.

    Returns (points, cube_hits, label_counts). points: list of (x, y, h, ix, iy, label)
    with h = height above the table plane. Robot hand region (above the measured TCP)
    and the robot base column are excluded.
    """
    model = camera_model(cam)
    rays = rays_for(model)
    depth = sub_grid(cam.depth_m, STRIDE)
    rgb = sub_grid(cam.rgb, STRIDE)
    px, py, pz = model["pos"]
    tz = TABLE_Z
    ex = ee_pos[0]
    ey = ee_pos[1]
    ez = ee_pos[2]
    points = []
    hits = {}
    tall_hits = {}
    for word in want_colors:
        hits[word] = []
        tall_hits[word] = []
    counts = {}
    ny = min(len(rays), len(depth))
    for iy in range(ny):
        drow = depth[iy]
        rrow = rays[iy]
        crow = rgb[iy]
        nx = min(len(rrow), len(drow))
        for ix in range(nx):
            d = drow[ix]
            if not (d > 0.05 and d < 5.0):
                continue
            ray = rrow[ix]
            h = pz + d * ray[2] - tz
            if h < H_OBJ_MIN or h > H_OBJ_MAX:
                continue
            x = px + d * ray[0]
            y = py + d * ray[1]
            if x * x + y * y < 0.0144:
                continue
            rx = x - ex
            ry = y - ey
            if h + tz > ez - 0.03 and rx * rx + ry * ry < 0.0144:
                continue
            c = crow[ix]
            label = color_label(c[0], c[1], c[2])
            points.append((x, y, h, ix, iy, label))
            if h <= H_CUBE_MAX:
                counts[label] = counts.get(label, 0) + 1
            for word in want_colors:
                if color_matches(word, label):
                    if h <= H_CUBE_MAX:
                        hits[word].append((x, y, h))
                    if h > H_TALL:
                        tall_hits[word].append((x, y, h))
    # colour hits next to a TALL object of the same colour belong to that object
    # (e.g. a container sharing the cube's colour), not to a cube
    for word in want_colors:
        if tall_hits[word] and hits[word]:
            keep = []
            for p in hits[word]:
                near_tall = False
                for q in tall_hits[word]:
                    if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < 0.0012:
                        near_tall = True
                        break
                if not near_tall:
                    keep.append(p)
            hits[word] = keep
    return points, hits, counts


def cube_estimate(hit_list):
    """Robust centroid of colour hits: median, then mean of hits within 4 cm."""
    if len(hit_list) < 2:
        return None
    mx = median([p[0] for p in hit_list])
    my = median([p[1] for p in hit_list])
    near = [p for p in hit_list if (p[0] - mx) ** 2 + (p[1] - my) ** 2 < 0.0016]
    if len(near) < 2:
        return None
    sx = sum(p[0] for p in near) / len(near)
    sy = sum(p[1] for p in near) / len(near)
    span = 0.0
    for p in near:
        span = max(span, math.sqrt((p[0] - sx) ** 2 + (p[1] - sy) ** 2))
    return {"x": sx, "y": sy, "n": len(near), "maxh": max(p[2] for p in near),
            "span": span}


def cluster_points(points, h_min):
    """Grid-adjacency + metric-proximity clustering (union-find) of object points."""
    sel = [p for p in points if p[2] >= h_min]
    index_of = {}
    for i, p in enumerate(sel):
        index_of[(p[3], p[4])] = i
    parent = list(range(len(sel)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, p in enumerate(sel):
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            j = index_of.get((p[3] + dx, p[4] + dy))
            if j is None:
                continue
            q = sel[j]
            if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < 0.0009 and abs(p[2] - q[2]) < 0.05:
                a = find(i)
                b = find(j)
                if a != b:
                    parent[a] = b
    groups = {}
    for i in range(len(sel)):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(sel[i])
    clusters = []
    for root in groups:
        pts = groups[root]
        if len(pts) < 4:
            continue
        clusters.append(summarize_cluster(pts))
    return clusters


def summarize_cluster(pts):
    n = len(pts)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    maxh = max(p[2] for p in pts)
    top = [p for p in pts if p[2] >= maxh - 0.012]
    tx = sum(p[0] for p in top) / len(top)
    ty = sum(p[1] for p in top) / len(top)
    extent = 0.0
    for p in pts:
        extent = max(extent, math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2))
    labels = {}
    for p in pts:
        if p[5]:
            labels[p[5]] = labels.get(p[5], 0) + 1
    uv = [median([p[3] for p in top]) * STRIDE, median([p[4] for p in top]) * STRIDE]
    return {"n": n, "x": cx, "y": cy, "top_x": tx, "top_y": ty, "maxh": maxh,
            "extent": extent, "labels": labels, "uv": uv, "pts": pts}


def split_cluster(cluster):
    """2-means split of a cluster in xy (used when two containers touch)."""
    pts = cluster["pts"]
    far = None
    best = -1.0
    for p in pts:
        dd = (p[0] - cluster["x"]) ** 2 + (p[1] - cluster["y"]) ** 2
        if dd > best:
            best = dd
            far = p
    far2 = None
    best = -1.0
    for p in pts:
        dd = (p[0] - far[0]) ** 2 + (p[1] - far[1]) ** 2
        if dd > best:
            best = dd
            far2 = p
    ca = [far[0], far[1]]
    cb = [far2[0], far2[1]]
    ga = []
    gb = []
    for iteration in range(8):
        ga = []
        gb = []
        for p in pts:
            da = (p[0] - ca[0]) ** 2 + (p[1] - ca[1]) ** 2
            db = (p[0] - cb[0]) ** 2 + (p[1] - cb[1]) ** 2
            if da <= db:
                ga.append(p)
            else:
                gb.append(p)
        if not ga or not gb:
            return [cluster]
        ca = [sum(p[0] for p in ga) / len(ga), sum(p[1] for p in ga) / len(ga)]
        cb = [sum(p[0] for p in gb) / len(gb), sum(p[1] for p in gb) / len(gb)]
    if len(ga) < 4 or len(gb) < 4:
        return [cluster]
    return [summarize_cluster(ga), summarize_cluster(gb)]


def cluster_brief(c):
    return {"n": c["n"], "x": rnd(c["x"]), "y": rnd(c["y"]), "top_x": rnd(c["top_x"]),
            "top_y": rnd(c["top_y"]), "maxh": rnd(c["maxh"]), "extent": rnd(c["extent"]),
            "uv": vec(c["uv"], 1), "labels": c["labels"]}


# ----------------------------------------------------------------- live containers
live_points = []
containers = []
container_h = 0.06
try:
    ee_live = [float(v) for v in state0.end_effector_poses["primary"].position]
    live_points, live_hits, live_counts = extract_points(obs0.cameras[FRONT], ee_live,
                                                         target_colors)
    raw_clusters = cluster_points(live_points, 0.012)
    report["live_clusters_raw"] = [cluster_brief(c) for c in raw_clusters]
    report["live_label_counts"] = live_counts
    plausible = [c for c in raw_clusters if 0.02 <= c["maxh"] <= 0.25 and c["extent"] < 0.15]
    # identical containers: group clusters of similar height, keep the largest group
    # (ties -> the taller group)
    best_group = []
    best_score = 0.0
    for c in plausible:
        group = [d for d in plausible if abs(d["maxh"] - c["maxh"]) <= 0.25 * c["maxh"]]
        score = len(group) * median([d["maxh"] for d in group])
        if score > best_score:
            best_score = score
            best_group = group
    cands = best_group
    if cands:
        biggest = max(c["n"] for c in cands)
        cands = [c for c in cands if c["n"] >= 0.2 * biggest]
    if len(cands) >= 2:
        med_extent = median([c["extent"] for c in cands])
        split = []
        for c in cands:
            if c["extent"] > 1.7 * med_extent:
                split.extend(split_cluster(c))
            else:
                split.append(c)
        cands = split
    containers = cands
    if containers:
        container_h = median([c["maxh"] for c in containers])
    report["live_containers"] = [cluster_brief(c) for c in containers]
    report["container_height_est"] = rnd(container_h)
    live_cubes = {}
    for word in target_colors:
        est = cube_estimate(live_hits[word])
        live_cubes[word] = None if est is None else {k: rnd(est[k]) for k in est}
    report["live_cube_hits"] = live_cubes
    log("live containers: " + str(len(containers)) + " height~" + str(rnd(container_h)))
    report["phase_status"]["P3_live"] = "ok"
except Exception as exc:
    record_error("P3.live", exc)
    report["phase_status"]["P3_live"] = "failed"


# ----------------------------------------------------------------- P3 conditioning video
video = []
n_video = 0
frame_points = []     # per processed frame: list of tracking points (x, y, h)
frame_cubes = []      # per processed frame: {color: estimate or None}
frame_times = []
frame_hits = []       # per processed frame: {color: [(x, y, h), ...]} (capped)
try:
    video = robomme_plus.get_conditioning_video()
    n_video = len(video)
    report["video_len"] = n_video
    log("conditioning video frames: " + str(n_video))
    if n_video > 0:
        try:
            first = video[0]
            last = video[n_video - 1]
            report["video_first_ts"] = rnd(first.timestamp_s)
            report["video_last_ts"] = rnd(last.timestamp_s)
            report["video_front_first"] = camera_summary(first.cameras[FRONT])
            report["video_thumbs"] = {"first": thumbnail_hex(first.cameras[FRONT], 32),
                                      "last": thumbnail_hex(last.cameras[FRONT], 32)}
        except Exception as exc:
            record_error("P3.video_meta", exc)
    report["phase_status"]["P3_video_fetch"] = "ok"
except Exception as exc:
    record_error("P3.video_fetch", exc)
    report["phase_status"]["P3_video_fetch"] = "failed"

track_h_min = max(0.012, min(0.05, 0.45 * container_h))
report["track_h_min"] = rnd(track_h_min)
video_summary = []
try:
    for fi in range(n_video):
        frame = video[fi]
        try:
            ts = float(frame.timestamp_s)
        except Exception:
            ts = float(fi)
        try:
            ee = [float(v) for v in frame.robot_state.end_effector_poses["primary"].position]
        except Exception:
            ee = [0.0, 0.0, 10.0]
        pts, hits, counts = extract_points(frame.cameras[FRONT], ee, target_colors)
        tpts = [(p[0], p[1], p[2]) for p in pts if p[2] >= track_h_min]
        cubes = {}
        for word in target_colors:
            cubes[word] = cube_estimate(hits[word])
        frame_points.append(tpts)
        frame_cubes.append(cubes)
        frame_hits.append({word: hits[word][:300] for word in target_colors})
        frame_times.append(ts)
        if fi % 8 == 0 or fi == n_video - 1:
            entry = {"i": fi, "t": rnd(ts, 3), "n_obj": len(pts), "n_track": len(tpts),
                     "ee": vec(ee, 3), "labels": counts}
            for word in target_colors:
                est = cubes[word]
                entry["cube_" + word] = None if est is None else [rnd(est["x"]), rnd(est["y"]),
                                                                  est["n"], rnd(est["maxh"])]
            video_summary.append(entry)
        if fi % 64 == 0:
            log("video frame " + str(fi) + "/" + str(n_video) + " pts=" + str(len(pts)))
    report["video_summary"] = video_summary
    try:
        for qi in (1, 2, 3):
            if n_video == 0:
                break
            fi = (n_video - 1) * qi // 4
            report["video_thumbs"]["q" + str(qi) + "_" + str(fi)] = thumbnail_hex(
                video[fi].cameras[FRONT], 32)
    except Exception as exc:
        record_error("P3.thumbs", exc)
    report["phase_status"]["P3_video_points"] = "ok"
except Exception as exc:
    record_error("P3.video_points", exc)
    report["phase_status"]["P3_video_points"] = "failed"

n_proc = len(frame_points)

# cube last-seen events
cube_events = {}
try:
    for word in target_colors:
        counts = [0 if c[word] is None else c[word]["n"] for c in frame_cubes]
        peak = max(counts) if counts else 0
        thresh = max(2, int(0.35 * peak))
        seen = [i for i in range(n_proc) if counts[i] >= thresh]
        if not seen:
            cube_events[word] = None
            continue
        last_seen = seen[-1]
        window = [i for i in seen if i >= last_seen - 6]
        xs = [frame_cubes[i][word]["x"] for i in window]
        ys = [frame_cubes[i][word]["y"] for i in window]
        cube_events[word] = {"first_seen": seen[0], "last_seen": last_seen, "peak": peak,
                             "n_seen": len(seen), "x": sum(xs) / len(xs), "y": sum(ys) / len(ys),
                             "visible_at_end": last_seen >= n_proc - 2}
    report["cube_events"] = {}
    for word in cube_events:
        ev = cube_events[word]
        report["cube_events"][word] = None if ev is None else {
            k: (rnd(ev[k]) if k in ("x", "y") else ev[k]) for k in ev}
    log("cube events: " + str(report["cube_events"]))
except Exception as exc:
    record_error("P3.cube_events", exc)


def hit_instances(hit_list, link=0.025, min_hits=3):
    """Single-linkage clusters of colour hits (one per cube); oversized clusters split once."""
    n = len(hit_list)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a = hit_list[i]
            b = hit_list[j]
            if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 < link * link:
                ra = find(i)
                rb = find(j)
                if ra != rb:
                    parent[ra] = rb
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(hit_list[i])
    out = []
    for key in groups:
        g = groups[key]
        if len(g) < min_hits:
            continue
        cx = sum(p[0] for p in g) / len(g)
        cy = sum(p[1] for p in g) / len(g)
        ext = max(math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) for p in g)
        parts = [g]
        if ext > 0.045 and len(g) >= 2 * min_hits:
            fake = {"pts": [(p[0], p[1], p[2], 0, 0, "") for p in g], "x": cx, "y": cy}
            halves = split_cluster_pts(fake)
            if halves is not None:
                parts = halves
        for part in parts:
            px = sum(p[0] for p in part) / len(part)
            py = sum(p[1] for p in part) / len(part)
            out.append({"x": px, "y": py, "n": len(part)})
    return out


def split_cluster_pts(cluster):
    pts = cluster["pts"]
    far = None
    best = -1.0
    for q in pts:
        dd = (q[0] - cluster["x"]) ** 2 + (q[1] - cluster["y"]) ** 2
        if dd > best:
            best = dd
            far = q
    far2 = None
    best = -1.0
    for q in pts:
        dd = (q[0] - far[0]) ** 2 + (q[1] - far[1]) ** 2
        if dd > best:
            best = dd
            far2 = q
    ca = [far[0], far[1]]
    cb = [far2[0], far2[1]]
    ga = []
    gb = []
    for iteration in range(8):
        ga = [q for q in pts if (q[0] - ca[0]) ** 2 + (q[1] - ca[1]) ** 2 <=
              (q[0] - cb[0]) ** 2 + (q[1] - cb[1]) ** 2]
        gb = [q for q in pts if (q[0] - ca[0]) ** 2 + (q[1] - ca[1]) ** 2 >
              (q[0] - cb[0]) ** 2 + (q[1] - cb[1]) ** 2]
        if not ga or not gb:
            return None
        ca = [sum(q[0] for q in ga) / len(ga), sum(q[1] for q in ga) / len(ga)]
        cb = [sum(q[0] for q in gb) / len(gb), sum(q[1] for q in gb) / len(gb)]
    return [ga, gb]


# per-instance cube events: instances found in the first frames where the colour is visible
# (cubes are static before the cover), then each instance's own last-seen frame
cube_instances = []   # [{"key", "color", "x", "y", "n", "last_seen", ...}]
try:
    for word in target_colors:
        counts = [len(h[word]) for h in frame_hits]
        peak = max(counts) if counts else 0
        seen = [i for i in range(len(counts)) if counts[i] >= max(2, int(0.35 * peak))]
        if not seen:
            continue
        pool = []
        for fi in seen[:6]:
            pool.extend(frame_hits[fi][word])
        insts = hit_instances(pool, 0.025, max(3, min(6, len(seen[:6]))))
        order_i = sorted([(-insts[i]["n"], i) for i in range(len(insts))])
        insts = [insts[i] for neg, i in order_i]
        for k, inst in enumerate(insts):
            per = []
            for fi in range(len(frame_hits)):
                c = 0
                for p in frame_hits[fi][word]:
                    if (p[0] - inst["x"]) ** 2 + (p[1] - inst["y"]) ** 2 < 0.0009:
                        c += 1
                per.append(c)
            ipeak = max(per)
            iseen = [i for i in range(len(per)) if per[i] >= max(1, int(0.35 * ipeak))]
            if not iseen:
                continue
            last = iseen[-1]
            # refine the position from the frames just before it vanished
            near = []
            for fi in iseen:
                if fi >= last - 6:
                    for p in frame_hits[fi][word]:
                        if (p[0] - inst["x"]) ** 2 + (p[1] - inst["y"]) ** 2 < 0.0009:
                            near.append(p)
            if near:
                inst["x"] = sum(p[0] for p in near) / len(near)
                inst["y"] = sum(p[1] for p in near) / len(near)
            cube_instances.append({"key": word + "#" + str(k), "color": word, "x": inst["x"],
                                   "y": inst["y"], "n": inst["n"], "peak": ipeak,
                                   "first_seen": iseen[0], "last_seen": last,
                                   "visible_at_end": last >= len(per) - 2})
    report["cube_instances"] = [{k: (rnd(q[k]) if k in ("x", "y") else q[k]) for k in q}
                                for q in cube_instances]
    log("cube instances: " + str(report["cube_instances"]))
except Exception as exc:
    record_error("P3.cube_instances", exc)


# ----------------------------------------------------------------- tracking
GATE = 0.07
CONFLICT = 0.045


def tracker_init(init_positions):
    k = len(init_positions)
    return {"k": k, "pos": [[p[0], p[1], p[2]] for p in init_positions],
            "vel": [[0.0, 0.0, 0.0] for i in range(k)],
            "recent": [[[p[0], p[1], p[2]]] for p in init_positions],
            "history": [[] for i in range(k)], "coast": [0] * k, "close_calls": 0}


def tracker_step(state, pts):
    """One constant-velocity association step on one frame's tracking points.

    Association: each point goes to the nearest predicted track in (x, y, h) within GATE.
    When two predicted tracks are closer than CONFLICT in xy, both coast on their
    pre-conflict velocity (measurements there are ambiguous), and resume afterwards.
    """
    k = state["k"]
    pos = state["pos"]
    vel = state["vel"]
    recent = state["recent"]
    history = state["history"]
    coast = state["coast"]
    pred = [[pos[i][0] + vel[i][0], pos[i][1] + vel[i][1], pos[i][2] + vel[i][2]]
            for i in range(k)]
    in_conflict = [False] * k
    for i in range(k):
        for j in range(i + 1, k):
            dd = (pred[i][0] - pred[j][0]) ** 2 + (pred[i][1] - pred[j][1]) ** 2
            if dd < CONFLICT * CONFLICT:
                in_conflict[i] = True
                in_conflict[j] = True
    for i in range(k):
        if in_conflict[i]:
            # coast in xy only: a vertical hop is not constant-velocity
            pred[i][2] = pos[i][2]
            vel[i][2] = 0.0
    if any(in_conflict):
        state["close_calls"] += 1
    sx = [0.0] * k
    sy = [0.0] * k
    sh = [0.0] * k
    sn = [0] * k
    for p in pts:
        best = -1
        best_d = GATE * GATE
        for i in range(k):
            dx = p[0] - pred[i][0]
            dy = p[1] - pred[i][1]
            dh = 0.5 * (p[2] - pred[i][2])
            dd = dx * dx + dy * dy + dh * dh
            if dd < best_d:
                best_d = dd
                best = i
        if best >= 0:
            sx[best] += p[0]
            sy[best] += p[1]
            sh[best] += p[2]
            sn[best] += 1
    for i in range(k):
        if sn[i] >= 3 and not in_conflict[i]:
            pos[i] = [sx[i] / sn[i], sy[i] / sn[i], sh[i] / sn[i]]
            recent[i].append([pos[i][0], pos[i][1], pos[i][2]])
            if len(recent[i]) > 5:
                recent[i] = recent[i][-5:]
            span = len(recent[i]) - 1
            if span >= 1:
                first = recent[i][0]
                vel[i] = [(pos[i][0] - first[0]) / span, (pos[i][1] - first[1]) / span,
                          (pos[i][2] - first[2]) / span]
        else:
            coast[i] += 1
            pos[i] = pred[i]
            if not in_conflict[i]:
                vel[i] = [0.85 * vel[i][0], 0.85 * vel[i][1], 0.85 * vel[i][2]]
                recent[i] = [[pos[i][0], pos[i][1], pos[i][2]]]
        history[i].append([pos[i][0], pos[i][1], sn[i], pos[i][2]])
    return state


def run_tracker(order, init_positions):
    """Run the tracker over video frames in `order` (frame indices)."""
    state = tracker_init(init_positions)
    for fi in order:
        tracker_step(state, frame_points[fi])
    return {"history": state["history"], "coast": state["coast"],
            "close_calls": state["close_calls"], "state": state}


def track_mean_h(c):
    hs = [p[2] for p in c["pts"] if p[2] >= track_h_min]
    if not hs:
        return c["maxh"]
    return sum(hs) / len(hs)


backward = None
forward = None
assignment = {}
try:
    if n_proc > 0 and containers:
        init = [[c["x"], c["y"], track_mean_h(c)] for c in containers]
        order_b = list(range(n_proc - 1, -1, -1))
        backward = run_tracker(order_b, init)
        # backward["history"][i][j] is the position at frame order_b[j] = n_proc-1-j
        report["track_backward"] = {"coast": backward["coast"],
                                    "close_calls": backward["close_calls"]}
        traj = []
        for i in range(len(containers)):
            seq = []
            for j in range(0, n_proc, 8):
                fi = n_proc - 1 - j
                h = backward["history"][i][j]
                seq.append([fi, rnd(h[0], 3), rnd(h[1], 3), h[2]])
            traj.append(seq)
        report["track_backward_traj"] = traj

        def back_pos(i, fi):
            h = backward["history"][i][n_proc - 1 - fi]
            return h

        def separated(fi):
            """All tracks pairwise apart (no swap in progress) at frame fi."""
            k = len(containers)
            for i in range(k):
                for j in range(i + 1, k):
                    a = back_pos(i, fi)
                    b = back_pos(j, fi)
                    if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 < (CONFLICT + 0.015) ** 2:
                        return False
            return True

        # cube -> container: median distance of each track to the cube's last position
        # over the frames right after it vanished, stopping before any swap begins
        latest_cover = 0
        inst_rank = {}
        for ev in cube_instances:
            word = ev["key"]
            t0 = ev["last_seen"]
            t1 = min(n_proc - 1, t0 + 1)
            window = [t1]
            for fi in range(t0 + 2, min(n_proc, t0 + 11)):
                if not separated(fi):
                    break
                window.append(fi)
            best = None
            ranking = []
            for i in range(len(containers)):
                dists = []
                for fi in window:
                    h = back_pos(i, fi)
                    dists.append(math.sqrt((h[0] - ev["x"]) ** 2 + (h[1] - ev["y"]) ** 2))
                dd = median(dists)
                ranking.append([i, rnd(dd)])
                if best is None or dd < best[0]:
                    best = (dd, i)
            margin = None
            others = sorted(r[1] for r in ranking if r[0] != best[1])
            if others:
                margin = rnd(others[0] - best[0])
            assignment[word] = {"track": best[1], "dist": rnd(best[0]), "window": [window[0],
                                window[-1]], "ranking": ranking, "margin": margin,
                                "color": ev["color"]}
            inst_rank[word] = ranking
            latest_cover = max(latest_cover, window[-1])
        # one container per cube: greedy on the global distance order
        pairs = []
        for word in inst_rank:
            for r in inst_rank[word]:
                pairs.append((r[1], word, r[0]))
        pairs = sorted(pairs)
        used_tracks = set()
        done_inst = set()
        for dd, word, track in pairs:
            if word in done_inst or track in used_tracks:
                continue
            if assignment[word]["track"] != track:
                assignment[word]["track_nearest"] = assignment[word]["track"]
                assignment[word]["track"] = track
                assignment[word]["dist"] = dd
                assignment[word]["exclusive_reassigned"] = True
            done_inst.add(word)
            used_tracks.add(track)
        for word in target_colors:
            if not any(ev["color"] == word for ev in cube_instances):
                assignment[word] = {"track": None, "why": "cube never seen in video",
                                    "color": word}
        # forward consistency check, started from a frame where all tracks are apart
        start_f = None
        for fi in range(latest_cover, n_proc - 1):
            if separated(fi):
                start_f = fi
                break
        if start_f is not None:
            init_f = []
            for i in range(len(containers)):
                h = back_pos(i, start_f)
                init_f.append([h[0], h[1], h[3]])
            order_f = list(range(start_f + 1, n_proc))
            forward = run_tracker(order_f, init_f)
            perm = []
            for i in range(len(containers)):
                end = forward["history"][i][-1]
                best = None
                for j in range(len(containers)):
                    dd = (end[0] - containers[j]["x"]) ** 2 + (end[1] - containers[j]["y"]) ** 2
                    if best is None or dd < best[0]:
                        best = (dd, j)
                perm.append(best[1])
            valid_perm = len(set(perm)) == len(perm)
            report["track_forward"] = {"coast": forward["coast"],
                                       "close_calls": forward["close_calls"],
                                       "perm": perm, "valid_perm": valid_perm,
                                       "start_frame": start_f}
            consistent = all(perm[i] == i for i in range(len(perm)))
            report["track_forward"]["consistent"] = consistent
            if not consistent:
                # backward starts from the live scene (well localized); forward is used only
                # when it is a valid one-to-one mapping AND coasted clearly less
                fb = sum(backward["coast"])
                ff = sum(forward["coast"])
                use_forward = valid_perm and ff * 2 < fb
                report["track_forward"]["use_forward"] = use_forward
                log("WARNING: forward/backward tracking disagree; use_forward=" +
                    str(use_forward))
                if use_forward:
                    for word in assignment:
                        if assignment[word].get("track") is not None:
                            assignment[word]["track_backward"] = assignment[word]["track"]
                            assignment[word]["track"] = perm[assignment[word]["track"]]
        report["assignment"] = assignment
        log("assignment: " + str(assignment))
    report["phase_status"]["P4"] = "ok"
except Exception as exc:
    record_error("P4", exc)
    report["phase_status"]["P4"] = "failed"


# ----------------------------------------------------------------- execution helpers
def ticks_now():
    try:
        return int(round(float(get_robot_state().timestamp_s) * control_hz))
    except Exception:
        return 0


def ticks_left():
    if max_steps is None:
        return 10000
    return max_steps - ticks_now()


terminal = {"done": False, "why": ""}


def check_terminal(res, label):
    flags = {}
    try:
        flags["terminated"] = bool(res.terminated)
    except Exception:
        pass
    try:
        flags["truncated"] = bool(res.truncated)
    except Exception:
        pass
    try:
        flags["done"] = bool(res.done)
    except Exception:
        pass
    if any(flags[k] for k in flags):
        terminal["done"] = True
        terminal["why"] = label + " " + str(flags)
        log("TERMINAL after " + label + ": " + str(flags))
    return flags


def exec_summary(res, label):
    out = {"label": label}
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
        if res.error is not None:
            out["error"] = str(res.error)[:300]
    except Exception:
        pass
    out["flags"] = check_terminal(res, label)
    out["tick"] = ticks_now()
    try:
        st = get_robot_state()
        out["ee"] = vec(st.end_effector_poses["primary"].position)
        out["gripper"] = rnd(st.gripper_positions["primary"])
    except Exception:
        pass
    report["motions"].append(out)
    log("motion " + label + ": " + str(out))
    return out


report["motions"] = []


def goto(position, quat, label, tolerance=0.01, steps=150):
    if terminal["done"]:
        return {"ok": False, "status": "skipped-terminal"}
    try:
        pose = Pose([float(position[0]), float(position[1]), float(position[2])],
                    [float(q) for q in quat], base_frame)
        res = move_to_pose(pose, tolerance=tolerance, max_steps=steps)
        out = exec_summary(res, label)
        text = str(out.get("status", "")) + " " + str(out.get("error", ""))
        if not out.get("ok") and "max_steps" in text and out.get("steps", 0) == 0:
            # the plan was longer than the cap and nothing moved: re-issue with a larger cap
            res = move_to_pose(pose, tolerance=tolerance, max_steps=int(steps * 2.5))
            out = exec_summary(res, label + "_retry")
        return out
    except Exception as exc:
        record_error("goto." + label, exc)
        return {"ok": False, "status": "exception"}


def gripper_call(kind):
    if terminal["done"]:
        return {"ok": False}
    try:
        if kind == "close":
            res = close_gripper()
        else:
            res = open_gripper()
        return exec_summary(res, kind + "_gripper")
    except Exception as exc:
        record_error("gripper." + kind, exc)
        return {"ok": False}


def hold_ticks(n, grip):
    for tick in range(n):
        if terminal["done"]:
            return
        try:
            st = get_robot_state()
            reply = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=st.joint_positions["primary"],
                gripper_position=grip, embodiment=st.embodiment)}))
            check_terminal(reply, "hold")
        except Exception as exc:
            record_error("hold", exc)
            return


def cloud_points(cloud, limit):
    pts = as_list(cloud.points)
    if pts is None:
        return []
    if len(pts) > limit:
        stride = int(math.ceil(len(pts) / float(limit)))
        pts = pts[::stride]
    return [[float(p[0]), float(p[1]), float(p[2])] for p in pts]


def top_face(points):
    """Top-face analysis: top z, min-area rectangle centre, narrow-axis angle, widths."""
    zs = [p[2] for p in points]
    top_z = quantile(zs, 0.97)
    top = [p for p in points if p[2] >= top_z - 0.01]
    if len(top) < 5:
        top = points
    best = None
    for step_i in range(18):
        ang = math.radians(5.0 * step_i)
        c = math.cos(ang)
        s = math.sin(ang)
        us = [p[0] * c + p[1] * s for p in top]
        ws = [-p[0] * s + p[1] * c for p in top]
        u0 = quantile(us, 0.03)
        u1 = quantile(us, 0.97)
        w0 = quantile(ws, 0.03)
        w1 = quantile(ws, 0.97)
        area = (u1 - u0) * (w1 - w0)
        if best is None or area < best[0]:
            best = (area, ang, u0, u1, w0, w1)
    area, ang, u0, u1, w0, w1 = best
    um = 0.5 * (u0 + u1)
    wm = 0.5 * (w0 + w1)
    c = math.cos(ang)
    s = math.sin(ang)
    cx = um * c - wm * s
    cy = um * s + wm * c
    wu = u1 - u0
    ww = w1 - w0
    if wu <= ww:
        narrow_ang = ang
        narrow = wu
        wide = ww
    else:
        narrow_ang = ang + math.pi / 2
        narrow = ww
        wide = wu
    return {"top_z": top_z, "cx": cx, "cy": cy, "narrow_angle": narrow_ang,
            "narrow": narrow, "wide": wide, "n_top": len(top), "n": len(points)}


def segment_cloud(cam_name, cand):
    try:
        return mask_to_point_cloud(cand, cam_name, target_frame=base_frame)
    except Exception:
        return mask_to_point_cloud(cand.mask, cam_name, target_frame=base_frame)


def localize_front_container(cluster_xy, uv):
    """segment_points at the tracked container's top pixel -> metric cloud -> top face."""
    out = {"source": None}
    tries = [[uv[0], uv[1]], [uv[1], uv[0]]]
    for attempt, pt in enumerate(tries):
        try:
            seg = segment_points(FRONT, [[float(pt[0]), float(pt[1])]])
            if not seg.ok or len(seg.segmentations) == 0:
                continue
            cand = seg.segmentations[0]
            cloud = segment_cloud(FRONT, cand)
            pts = cloud_points(cloud, 600)
            pts = [p for p in pts if p[2] > TABLE_Z + 0.004]
            if len(pts) < 10:
                continue
            mx = sum(p[0] for p in pts) / len(pts)
            my = sum(p[1] for p in pts) / len(pts)
            dist = math.sqrt((mx - cluster_xy[0]) ** 2 + (my - cluster_xy[1]) ** 2)
            out["attempt_" + str(attempt)] = {"n": len(pts), "dist": rnd(dist),
                                              "score": rnd(cand.score, 3)}
            if dist < 0.035:
                face = top_face(pts)
                out["source"] = "segment_points"
                out["face"] = face
                out["cand"] = cand
                return out
        except Exception as exc:
            out["error_" + str(attempt)] = str(exc)[:200]
    return out


def localize_wrist_container(target_xy, target_top):
    """At the hover pose: wrist segmentation, candidate nearest the expected centre."""
    out = {"source": None, "cands": []}
    best = None
    try:
        obs = get_observation()
        wmodel = camera_model(obs.cameras[WRIST])
        uv = project_to_pixel(wmodel, calib["conv"], calib["use_range"],
                              [target_xy[0], target_xy[1], target_top])
        out["proj_uv"] = vec(uv, 1) if uv is not None else None
        segs = []
        if uv is not None and 0 <= uv[0] < wmodel["w"] and 0 <= uv[1] < wmodel["h"]:
            try:
                segs.append(("points", segment_points(WRIST, [[float(uv[0]), float(uv[1])]])))
            except Exception as exc:
                out["points_error"] = str(exc)[:200]
        for prompt in ("container", "cup"):
            try:
                segs.append((prompt, segment_text(WRIST, prompt)))
            except Exception as exc:
                out["text_error"] = str(exc)[:200]
        for tag, seg in segs:
            if not seg.ok:
                continue
            for cand in seg.segmentations[:4]:
                try:
                    pts = cloud_points(segment_cloud(WRIST, cand), 600)
                    pts = [p for p in pts if p[2] > TABLE_Z + 0.01]
                    if len(pts) < 10:
                        continue
                    face = top_face(pts)
                    dist = math.sqrt((face["cx"] - target_xy[0]) ** 2 +
                                     (face["cy"] - target_xy[1]) ** 2)
                    entry = {"tag": tag, "score": rnd(cand.score, 3), "dist": rnd(dist),
                             "narrow": rnd(face["narrow"]), "wide": rnd(face["wide"]),
                             "top_z": rnd(face["top_z"])}
                    out["cands"].append(entry)
                    plausible = face["wide"] < 0.2 and abs(face["top_z"] - target_top) < 0.03
                    if dist < 0.03 and plausible and (best is None or dist < best[0]):
                        best = (dist, face, tag)
                except Exception as exc:
                    out["cand_error"] = str(exc)[:200]
    except Exception as exc:
        out["error"] = str(exc)[:200]
    if best is not None:
        out["source"] = best[2]
        out["face"] = best[1]
    return out


def face_brief(face):
    if face is None:
        return None
    return {"top_z": rnd(face["top_z"]), "cx": rnd(face["cx"]), "cy": rnd(face["cy"]),
            "narrow_angle_deg": rnd(math.degrees(face["narrow_angle"]), 1),
            "narrow": rnd(face["narrow"]), "wide": rnd(face["wide"]),
            "n_top": face["n_top"], "n": face["n"]}


# ----------------------------------------------------------------- P5 execute
def relocate_container(prev_xy):
    """Re-detect the container nearest prev_xy in the current front view."""
    try:
        obs = get_observation()
        ee = [float(v) for v in obs.robot_state.end_effector_poses["primary"].position]
        pts, hits, counts = extract_points(obs.cameras[FRONT], ee, [])
        best = None
        for c in cluster_points(pts, 0.012):
            dd = math.sqrt((c["x"] - prev_xy[0]) ** 2 + (c["y"] - prev_xy[1]) ** 2)
            if c["maxh"] >= 0.6 * container_h and (best is None or dd < best[0]):
                best = (dd, c)
        if best is not None and best[0] < 0.08:
            return best[1]
    except Exception as exc:
        record_error("relocate", exc)
    return None


def graspnet_pose(seg_cand):
    try:
        grasps = generate_grasps(FRONT, seg_cand, backend="contact-graspnet")
        if not grasps.ok:
            return None
        g = select_grasp(grasps, strategy="top_down")
        if g is None:
            return None
        return [[float(v) for v in g.pose.position], [float(v) for v in g.pose.quaternion_wxyz]]
    except Exception as exc:
        record_error("graspnet", exc)
        return None


def attempt_pick(word, cont, pick, attempt, q_down, jaw_angle):
    """One localize -> hover -> wrist refine -> descend -> close -> lift attempt."""
    tag = word + "_a" + str(attempt)
    info = {"attempt": attempt}
    pick["attempts"].append(info)
    loc = localize_front_container([cont["x"], cont["y"]], cont["uv"])
    info["front_loc"] = {k: loc[k] for k in loc if k not in ("face", "cand")}
    face = loc.get("face")
    info["front_face"] = face_brief(face)
    if face is None:
        face = {"top_z": TABLE_Z + cont["maxh"], "cx": cont["top_x"], "cy": cont["top_y"],
                "narrow_angle": jaw_angle, "narrow": 0.05, "wide": 0.05, "n_top": 0, "n": 0}
        info["front_face_fallback"] = True
    target_xy = [face["cx"], face["cy"]]
    top_z = face["top_z"]
    height = top_z - TABLE_Z
    info["height"] = rnd(height)

    def yaw_quat(narrow_angle):
        delta = wrap_half_pi(narrow_angle - jaw_angle)
        qz = [math.cos(delta / 2.0), 0.0, 0.0, math.sin(delta / 2.0)]
        return quat_mul(qz, q_down), delta

    quat, delta = yaw_quat(face["narrow_angle"])
    hover_z = top_z + 0.12
    gripper_call("open")
    hv = goto([target_xy[0], target_xy[1], hover_z], quat, "hover_" + tag, 0.012, 150)
    if not hv.get("ok") and not terminal["done"]:
        quat, delta = q_down, 0.0
        info["hover_yaw_dropped"] = True
        goto([target_xy[0], target_xy[1], hover_z], quat, "hover_plain_" + tag, 0.012, 150)
    if not live["static"] and not terminal["done"]:
        moved = relocate_container(target_xy)
        if moved is not None:
            shift = math.sqrt((moved["top_x"] - target_xy[0]) ** 2 + (moved["top_y"] - target_xy[1]) ** 2)
            info["moving_rehover_shift"] = rnd(shift)
            if shift > 0.015:
                target_xy = [moved["top_x"], moved["top_y"]]
                goto([target_xy[0], target_xy[1], hover_z], quat, "rehover_" + tag, 0.012, 80)
    wloc = localize_wrist_container(target_xy, top_z)
    info["wrist_loc"] = {k: wloc[k] for k in wloc if k != "face"}
    wface = wloc.get("face")
    info["wrist_face"] = face_brief(wface)
    width = face["narrow"]
    if wface is not None:
        target_xy = [wface["cx"], wface["cy"]]
        if abs(wface["narrow"] - wface["wide"]) > 0.008:
            quat, delta = yaw_quat(wface["narrow_angle"])
        width = wface["narrow"]
    info["grasp_xy"] = vec(target_xy)
    info["yaw_delta_deg"] = rnd(math.degrees(delta), 1)
    info["width_est"] = rnd(width)
    depth_in = min(0.03, max(0.012, 0.45 * height))
    grasp_z = max(TABLE_Z + 0.008, top_z - depth_in)
    info["grasp_z"] = rnd(grasp_z)
    grasp_pos = [target_xy[0], target_xy[1], grasp_z]
    pre_pos = [target_xy[0], target_xy[1], top_z + 0.03]
    if width > 0.076 and loc.get("cand") is not None:
        gp = graspnet_pose(loc["cand"])
        info["too_wide"] = True
        info["graspnet"] = None if gp is None else [vec(gp[0]), vec(gp[1])]
        if gp is not None:
            quat = gp[1]
            grasp_pos = gp[0]
            appr = rot_apply(quat_to_rot(quat), [0.0, 0.0, 1.0])
            pre_pos = [grasp_pos[0] - 0.08 * appr[0], grasp_pos[1] - 0.08 * appr[1],
                       grasp_pos[2] - 0.08 * appr[2]]
            goto(pre_pos, quat, "graspnet_pre_" + tag, 0.012, 120)
    else:
        goto([target_xy[0], target_xy[1], hover_z], quat, "hover_refined_" + tag, 0.01, 80)
    goto(pre_pos, quat, "pregrasp_" + tag, 0.01, 80)
    goto(grasp_pos, quat, "grasp_" + tag, 0.008, 80)
    gripper_call("close")
    try:
        g_closed = float(get_robot_state().gripper_positions["primary"])
    except Exception:
        g_closed = -1.0
    info["gripper_after_close"] = rnd(g_closed)
    lift_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.15]
    goto(lift_pos, quat, "lift_" + tag, 0.015, 100)
    g_lift = -1.0
    try:
        g_lift = float(get_robot_state().gripper_positions["primary"])
    except Exception:
        pass
    info["gripper_after_lift"] = rnd(g_lift)
    hold_ticks(10, 0.0)
    holding = g_lift > 0.05
    info["holding"] = holding
    try:
        obs_v = get_observation()
        report["live_thumbs"]["front_after_lift_" + tag] = thumbnail_hex(obs_v.cameras[FRONT], 48)
        pts_c, hits_c, counts_c = extract_points(obs_v.cameras[FRONT], [0.0, 0.0, 10.0],
                                                 target_colors)
        near = [p for p in pts_c if (p[0] - target_xy[0]) ** 2 +
                (p[1] - target_xy[1]) ** 2 < 0.0016]
        info["verify_points_near"] = len(near)
        info["verify_max_h_near"] = rnd(max([p[2] for p in near])) if near else None
        cubes_v = {}
        for w2 in target_colors:
            est = cube_estimate(hits_c[w2])
            cubes_v[w2] = None if est is None else [rnd(est["x"]), rnd(est["y"]), est["n"]]
        info["verify_cubes_visible"] = cubes_v
    except Exception as exc:
        record_error("verify", exc)
    return {"holding": holding, "grasp_pos": grasp_pos, "quat": quat, "hover_z": hover_z,
            "target_xy": target_xy}


# ----------------------------------------------------------------- P4b live watch
# v00 evidence (episode 0): the live front view kept changing from tick 1 on while the arm was
# still far above the table -- the shuffle continues after the conditioning video. Track every
# container live (same association as the video tracker) with the arm held still, until static.
LIVE_HOLD = 2           # ticks between live tracker updates (~1 cm of container motion)
STILL_EPS = 0.006       # m: max per-track displacement over the stillness window
STILL_UPDATES = 10      # updates (= 20 ticks) the scene must stay still
live = {"tracker": None, "static": False, "updates": 0, "pts": None, "obs": None}
containers_now = list(containers)


def live_observe():
    obs = get_observation()
    try:
        ee = [float(v) for v in obs.robot_state.end_effector_poses["primary"].position]
    except Exception:
        ee = [float(v) for v in get_robot_state().end_effector_poses["primary"].position]
    pts, hits, counts = extract_points(obs.cameras[FRONT], ee, [])
    live["obs"] = obs
    live["pts"] = pts
    return [(p[0], p[1], p[2]) for p in pts if p[2] >= track_h_min]


def live_update(n_hold, grip):
    hold_ticks(n_hold, grip)
    if terminal["done"]:
        return False
    tpts = live_observe()
    tracker_step(live["tracker"], tpts)
    live["updates"] += 1
    return True


def tracks_to_clusters():
    """Current cluster for every track (nearest same-height cluster within 3 cm)."""
    out = []
    clusters = []
    try:
        clusters = [c for c in cluster_points(live["pts"], 0.012)
                    if c["maxh"] >= 0.6 * container_h and c["maxh"] <= 1.6 * container_h]
    except Exception as exc:
        record_error("P4b.clusters", exc)
    model = None
    try:
        model = camera_model(live["obs"].cameras[FRONT])
    except Exception:
        model = None
    for i in range(live["tracker"]["k"]):
        p = live["tracker"]["pos"][i]
        best = None
        for c in clusters:
            dd = math.sqrt((c["x"] - p[0]) ** 2 + (c["y"] - p[1]) ** 2)
            if dd < 0.03 and (best is None or dd < best[0]):
                best = (dd, c)
        if best is not None:
            out.append(best[1])
            continue
        uv = None
        if model is not None:
            uv = project_to_pixel(model, calib["conv"], calib["use_range"],
                                  [p[0], p[1], TABLE_Z + container_h])
        if uv is None:
            uv = containers[i]["uv"]
        out.append({"n": 0, "x": p[0], "y": p[1], "top_x": p[0], "top_y": p[1],
                    "maxh": container_h, "extent": 0.0, "labels": {}, "uv": uv, "pts": [],
                    "synthetic": True})
    return out


report["live_watch"] = {}
try:
    n_targets = 0
    for word in assignment:
        a = assignment.get(word)
        if a is not None and a.get("track") is not None:
            n_targets += 1
    reserve = 80 + 150 * max(1, n_targets)
    wait_until = (max_steps - reserve) if max_steps is not None else 600
    wait_until = max(40, wait_until)
    report["live_watch"]["wait_until_tick"] = wait_until
    if containers:
        live["tracker"] = tracker_init([[c["x"], c["y"], track_mean_h(c)] for c in containers])
        snaps = []
        disp_log = []
        while not terminal["done"] and ticks_now() < wait_until:
            if not live_update(LIVE_HOLD, 1.0):
                break
            snaps.append([[p[0], p[1]] for p in live["tracker"]["pos"]])
            if len(snaps) > STILL_UPDATES:
                ref = snaps[-1 - STILL_UPDATES]
                cur = snaps[-1]
                disp = max(math.sqrt((cur[i][0] - ref[i][0]) ** 2 + (cur[i][1] - ref[i][1]) ** 2)
                           for i in range(len(cur)))
                if live["updates"] % 5 == 0:
                    disp_log.append([ticks_now(), rnd(disp)])
                if disp < STILL_EPS:
                    live["static"] = True
                    break
            if len(snaps) > 3 * STILL_UPDATES:
                snaps = snaps[-(STILL_UPDATES + 1):]
        containers_now = tracks_to_clusters()
        drift = []
        for i in range(len(containers)):
            p = live["tracker"]["pos"][i]
            drift.append(rnd(math.sqrt((p[0] - containers[i]["x"]) ** 2 +
                                       (p[1] - containers[i]["y"]) ** 2)))
        report["live_watch"].update({
            "updates": live["updates"], "static": live["static"], "tick": ticks_now(),
            "close_calls": live["tracker"]["close_calls"], "coast": live["tracker"]["coast"],
            "disp_log": disp_log[-60:], "drift_from_tick0": drift,
            "positions": [vec(p[:2], 4) for p in live["tracker"]["pos"]],
            "now": [cluster_brief(c) for c in containers_now]})
        try:
            report["live_thumbs"]["front_after_watch"] = thumbnail_hex(live["obs"].cameras[FRONT], 48)
        except Exception as exc:
            record_error("P4b.thumb", exc)
        log("live watch: static=" + str(live["static"]) + " updates=" + str(live["updates"]) +
            " tick=" + str(ticks_now()) + " drift=" + str(drift))
    report["phase_status"]["P4b"] = "ok"
except Exception as exc:
    record_error("P4b", exc)
    report["phase_status"]["P4b"] = "failed"
    containers_now = list(containers)


report["picks"] = []
try:
    q_down = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]
    approach = rot_apply(quat_to_rot(q_down), [0.0, 0.0, 1.0])
    report["home_approach_axis"] = vec(approach)
    if approach[2] > -0.9:
        q_down = [0.0, 1.0, 0.0, 0.0]
        log("home orientation is not top-down; using [0,1,0,0]")
    jaw_axis = rot_apply(quat_to_rot(q_down), [0.0, 1.0, 0.0])
    jaw_angle = math.atan2(jaw_axis[1], jaw_axis[0])

    plan = []
    for word in target_colors:
        keys = [ev["key"] for ev in cube_instances if ev["color"] == word and
                assignment.get(ev["key"], {}).get("track") is not None]
        if not keys:
            continue
        if word not in plural_colors:
            keys = keys[:1]     # instances are sorted by hit count: the clearest cube
        for key in keys:
            plan.append((key, assignment[key]["track"]))
    if not plan and containers:
        # no usable video evidence: nothing principled to choose -- recorded as a guess
        plan.append(("unresolved", 0))
        log("WARNING: no cube->container assignment; falling back to container 0 (guess)")
    report["pick_plan"] = [[w, t] for w, t in plan]

    for pick_index, item in enumerate(plan):
        word, track = item
        if terminal["done"]:
            break
        pick = {"color": word, "track": track, "attempts": []}
        report["picks"].append(pick)
        cont = containers_now[track]
        if pick_index > 0:
            moved = relocate_container([cont["x"], cont["y"]])
            if moved is not None:
                cont = moved
        if ticks_left() < 60:
            pick["skipped"] = "tick budget"
            break
        outcome = attempt_pick(word, cont, pick, 0, q_down, jaw_angle)
        if not outcome["holding"] and not terminal["done"] and ticks_left() > 150:
            # missed or slipped: release, back off, re-detect the same container, retry once
            gripper_call("open")
            goto([outcome["target_xy"][0], outcome["target_xy"][1], outcome["hover_z"]],
                 outcome["quat"], "backoff_" + word, 0.015, 80)
            moved = relocate_container(outcome["target_xy"])
            pick["retry_relocated"] = None if moved is None else cluster_brief(moved)
            if moved is not None:
                cont = moved
            outcome = attempt_pick(word, cont, pick, 1, q_down, jaw_angle)
        pick["holding"] = outcome["holding"]
        more = pick_index < len(plan) - 1
        if more and not terminal["done"]:
            # put it back where it was, release, retreat
            gp = outcome["grasp_pos"]
            goto([gp[0], gp[1], gp[2] + 0.006], outcome["quat"], "putdown_" + word, 0.012, 100)
            gripper_call("open")
            goto([gp[0], gp[1], outcome["hover_z"]], outcome["quat"], "retreat_" + word, 0.015, 80)
    if not terminal["done"]:
        hold_ticks(5, 0.0)
    report["phase_status"]["P5"] = "done"
except Exception as exc:
    record_error("P5", exc)
    report["phase_status"]["P5"] = "failed"

report["terminal"] = terminal
report["final_tick"] = ticks_now()
try:
    report["live_thumbs"]["front_end"] = thumbnail_hex(get_observation().cameras[FRONT], 48)
except Exception as exc:
    record_error("final_thumb", exc)
log("done; terminal=" + str(terminal) + " tick=" + str(report["final_tick"]))
result = report
