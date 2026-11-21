# StopCube v05 — readiness: if the N-th arrival becomes imminent while the arm is still moving to,
# settling above, or calibrating over the button, skip the remaining preparation and press at once
# (urgency judged from the measured arm speed and the distance to the registration point).
# Builds on v04 (mask-based role assignment) and v03 (RANSAC table plane).
#
# Strategy: every control tick is a single `step`, so the front camera is observed on every tick.
# A colour-independent depth tracker fits the table plane in the front camera, keeps pixels whose
# height above the plane lies in the cube-top band, projects them onto the plane and measures the
# cube footprint's distance to the target centre in metres.  Visits of the target are counted with
# hysteresis.  The arm closes the gripper, moves above the button (refined with the wrist camera),
# calibrates its own vertical step response in free air, hovers, and starts the press when the
# predicted time to the N-th closest approach equals the calibrated press latency.
import math

report = {"version": "v05", "notes": [], "visits": [], "stages": []}

# ----------------------------------------------------------------------------- goal parsing
ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
    "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20,
}
CARDINAL_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "once": 1, "twice": 2, "thrice": 3,
}


def parse_count(text):
    words = []
    cleaned = ""
    for ch in str(text).lower():
        if ch.isalnum():
            cleaned = cleaned + ch
        else:
            cleaned = cleaned + " "
    words = cleaned.split()
    for word in words:
        if word in ORDINAL_WORDS:
            return ORDINAL_WORDS[word]
    for word in words:
        digits = ""
        for ch in word:
            if ch.isdigit():
                digits = digits + ch
            else:
                break
        if digits != "" and (word[len(digits):] in ("st", "nd", "rd", "th", "")):
            return int(digits)
    for index, word in enumerate(words):
        if word in CARDINAL_WORDS:
            return CARDINAL_WORDS[word]
    return None


context = get_task_context()
goal = str(context.language)
report["goal"] = goal
target_count = parse_count(goal)
report["target_count"] = target_count
if target_count is None:
    target_count = 1
    report["notes"].append("count not parsed; defaulting to 1")

max_steps = None
control_hz = 20.0
try:
    task_meta = robomme_plus.get_task_metadata()
    try:
        max_steps = int(task_meta["max_steps"])
    except Exception:
        max_steps = None
    try:
        control_hz = float(task_meta["control_hz"])
    except Exception:
        try:
            control_hz = float(task_meta["control_frequency"])
        except Exception:
            control_hz = 20.0
except Exception as exc:
    report["notes"].append("task metadata unavailable: " + str(exc))
report["max_steps"] = max_steps
report["control_hz"] = control_hz

try:
    video = robomme_plus.get_conditioning_video()
    report["conditioning_video_frames"] = len(video)
except Exception as exc:
    report["conditioning_video_frames"] = "unavailable: " + str(exc)

state0 = get_robot_state()
base_frame = state0.base_frame
embodiment = state0.embodiment

# ----------------------------------------------------------------------------- small helpers


def quantile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def vsub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def vadd(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def vscale(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vcross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def vnorm(a):
    return math.sqrt(vdot(a, a))


def depth_rows_of(camera):
    depth = camera.depth_m
    try:
        return depth.tolist()
    except Exception:
        rows = []
        for v in range(len(depth)):
            rows.append([float(z) for z in depth[v]])
        return rows


def intrinsics_of(camera):
    k = camera.intrinsics
    return (float(k[0][0]), float(k[1][1]), float(k[0][2]), float(k[1][2]))


def front_observation():
    try:
        obs = get_observation(camera_names=["front"])
        cam = obs.cameras["front"]
    except Exception:
        obs = get_observation()
        cam = obs.cameras["front"]
    try:
        stamp = float(obs.timestamp_s)
    except Exception:
        stamp = float(get_robot_state().timestamp_s)
    return cam, stamp


def candidates_for(camera_name, prompts, limit):
    found = []
    for prompt in prompts:
        try:
            seg = segment_text(camera_name, prompt)
        except Exception as exc:
            report["notes"].append("segment_text failed " + prompt + ": " + str(exc))
            continue
        ok = True
        try:
            ok = bool(seg.ok)
        except Exception:
            ok = True
        if not ok:
            continue
        documented = []
        try:
            boxes = seg.boxes
            scores = seg.scores
            masks = seg.masks
            for index in range(min(len(boxes), limit)):
                documented.append({"box": [float(value) for value in boxes[index]],
                                   "score": float(scores[index]), "item": masks[index],
                                   "prompt": prompt})
        except Exception:
            documented = []
        if len(documented) > 0:
            found.extend(documented)
            continue
        items = None
        try:
            items = list(seg.segmentations)
        except Exception:
            items = None
        if items is not None:
            for item in items[:limit]:
                try:
                    box = [float(value) for value in item.box_xyxy]
                except Exception:
                    continue
                try:
                    score = float(item.score)
                except Exception:
                    score = 0.0
                found.append({"box": box, "score": score, "item": item, "prompt": prompt})
        else:
            try:
                boxes = seg.boxes
                scores = seg.scores
                masks = seg.masks
                for index in range(min(len(boxes), limit)):
                    found.append({"box": [float(value) for value in boxes[index]],
                                  "score": float(scores[index]), "item": masks[index],
                                  "prompt": prompt})
            except Exception:
                pass
    return found


def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    if area_a + area_b - inter <= 0.0:
        return 0.0
    return inter / (area_a + area_b - inter)


# ----------------------------------------------------------------------------- table plane (front camera)
# Pinhole model in the camera's own pixel/depth convention.  Plane: n . P = 1, so the plane's inverse
# depth at pixel (u, v) is a*u + b*v + c with n = (a*fx, b*fy, c + a*cx + b*cy).


def solve3(m, r):
    det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
           - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
           + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    if abs(det) < 1e-18:
        return None
    out = []
    for col in range(3):
        mm = [list(m[0]), list(m[1]), list(m[2])]
        for row in range(3):
            mm[row][col] = r[row]
        d = (mm[0][0] * (mm[1][1] * mm[2][2] - mm[1][2] * mm[2][1])
             - mm[0][1] * (mm[1][0] * mm[2][2] - mm[1][2] * mm[2][0])
             + mm[0][2] * (mm[1][0] * mm[2][1] - mm[1][1] * mm[2][0]))
        out.append(d / det)
    return out


def lsq_plane(keep):
    suu = suv = su = svv = sv = sn = 0.0
    sur = svr = sr = 0.0
    for sample in keep:
        u, v, r = sample[0], sample[1], sample[2]
        suu += u * u
        suv += u * v
        su += u
        svv += v * v
        sv += v
        sn += 1.0
        sur += u * r
        svr += v * r
        sr += r
    return solve3([[suu, suv, su], [suv, svv, sv], [su, sv, sn]], [sur, svr, sr])


def plane_height(coeffs, intr, sample):
    fx, fy, cx, cy = intr
    a, b, c = coeffs
    nn = vnorm([a * fx, b * fy, c + a * cx + b * cy])
    return (1.0 - sample[3] * (a * sample[0] + b * sample[1] + c)) / nn


def fit_plane(rows, intr, stride):
    # Deterministic RANSAC in inverse-depth space (a plane's inverse depth is affine in u, v),
    # then least-squares refinement on the inliers.  The table is the dominant plane; walls, the
    # robot and objects are outliers.
    fx, fy, cx, cy = intr
    height = len(rows)
    width = len(rows[0])
    samples = []
    for v in range(0, height, stride):
        row = rows[v]
        for u in range(0, width, stride):
            z = row[u]
            if z > 0.0 and z < 50.0:
                samples.append((float(u), float(v), 1.0 / z, z))
    if len(samples) < 30:
        return None
    probe = samples[::max(1, len(samples) // 600)]
    seed = 12345
    best = None
    best_count = -1
    for hypothesis in range(150):
        picks = []
        for pick in range(3):
            seed = (seed * 1103515245 + 12345) % 2147483648
            picks.append(samples[seed % len(samples)])
        p0, p1, p2 = picks
        area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
        if abs(area) < 100.0:
            continue
        coeffs = solve3([[p0[0], p0[1], 1.0], [p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0]], [p0[2], p1[2], p2[2]])
        if coeffs is None:
            continue
        count = 0
        for sample in probe:
            if abs(plane_height(coeffs, intr, sample)) < 0.003:
                count += 1
        if count > best_count:
            best_count = count
            best = coeffs
    if best is None:
        return None
    coeffs = best
    keep = samples
    for limit in (0.004, 0.002):
        keep = [sample for sample in samples if abs(plane_height(coeffs, intr, sample)) < limit]
        if len(keep) < 30:
            break
        refined = lsq_plane(keep)
        if refined is None:
            break
        coeffs = refined
    a, b, c = coeffs
    n = [a * fx, b * fy, c + a * cx + b * cy]
    nn = vnorm(n)
    unit = vscale(n, 1.0 / nn)
    ex = [1.0, 0.0, 0.0]
    e1 = vsub(ex, vscale(unit, vdot(ex, unit)))
    e1 = vscale(e1, 1.0 / vnorm(e1))
    e2 = vcross(unit, e1)
    return {"a": a, "b": b, "c": c, "nn": nn, "unit": unit, "e1": e1, "e2": e2,
            "intr": intr, "inliers": len(keep), "samples": len(samples)}


def pixel_point(plane, u, v, z):
    fx, fy, cx, cy = plane["intr"]
    return [(u - cx) * z / fx, (v - cy) * z / fy, z]


def height_of(plane, u, v, z):
    return (1.0 - z * (plane["a"] * u + plane["b"] * v + plane["c"])) / plane["nn"]


def footprint(plane, u, v, z):
    point = pixel_point(plane, u, v, z)
    h = (1.0 - vdot(point, vscale(plane["unit"], plane["nn"]))) / plane["nn"]
    return vadd(point, vscale(plane["unit"], h)), h


def box_samples(rows, plane, box, stride):
    height = len(rows)
    width = len(rows[0])
    u0 = max(0, int(box[0]))
    v0 = max(0, int(box[1]))
    u1 = min(width - 1, int(box[2] + 0.999))
    v1 = min(height - 1, int(box[3] + 0.999))
    out = []
    for v in range(v0, v1 + 1, stride):
        row = rows[v]
        for u in range(u0, u1 + 1, stride):
            z = row[u]
            if z > 0.0 and z < 50.0:
                foot, h = footprint(plane, float(u), float(v), z)
                out.append((h, foot, u, v))
    return out


# ----------------------------------------------------------------------------- initial perception (zero ticks)
cam0, stamp0 = front_observation()
intr = intrinsics_of(cam0)
rows0 = depth_rows_of(cam0)
img_h = len(rows0)
img_w = len(rows0[0])
plane = fit_plane(rows0, intr, 4)
if plane is None:
    raise ValueError("table plane fit failed in front camera")
report["plane_inliers"] = [plane["inliers"], plane["samples"]]
report["camera_to_table_m"] = round(1.0 / plane["nn"], 4)

BUTTON_PROMPTS = ("button", "push button", "white button")
TARGET_PROMPTS = ("target", "round target", "bullseye target", "circular target", "disc")
CUBE_PROMPTS = ("cube", "small cube", "block")

button_cands = candidates_for("front", BUTTON_PROMPTS, 5)
target_cands = candidates_for("front", TARGET_PROMPTS, 5)
cube_cands = candidates_for("front", CUBE_PROMPTS, 5)
report["candidate_counts"] = [len(button_cands), len(target_cands), len(cube_cands)]


def mask_grid(item):
    """Return an indexable [v][u] mask for a candidate, or None (box-only fallback)."""
    for attempt in range(2):
        try:
            grid = item.mask if attempt == 0 else item
            probe = grid[0][0]
            if len(grid) == img_h and len(grid[0]) == img_w:
                return grid
        except Exception:
            pass
    return None


def cand_samples(cand, rows):
    samples = box_samples(rows, plane, cand["box"], 1)
    grid = cand.get("grid")
    if grid is None:
        return samples
    inside = [s for s in samples if bool(grid[s[3]][s[2]])]
    if len(inside) >= 4:
        return inside
    return samples


def change_fraction(cand, rows_a, rows_b):
    # Fraction of the candidate's pixels whose height above the table changed by > 3 mm.
    first = cand_samples(cand, rows_a)
    later = {}
    for s in box_samples(rows_b, plane, cand["box"], 1):
        later[(s[2], s[3])] = s[0]
    changed = 0
    counted = 0
    for s in first:
        key = (s[2], s[3])
        if key in later:
            counted += 1
            if abs(later[key] - s[0]) > 0.003:
                changed += 1
    if counted == 0:
        return None
    return changed / counted


def describe(cand, rows):
    samples = cand_samples(cand, rows)
    heights = [s[0] for s in samples]
    raised = [s for s in samples if s[0] > 0.006]
    info = {"n": len(samples), "h50": quantile(heights, 0.5), "h90": quantile(heights, 0.9),
            "raised": len(raised)}
    if len(raised) > 0:
        info["foot"] = [sum(s[1][i] for s in raised) / len(raised) for i in range(3)]
    else:
        info["foot"] = None
    return info


for cand in button_cands + target_cands + cube_cands:
    cand["grid"] = mask_grid(cand["item"])
    cand["info"] = describe(cand, rows0)
report["masks_available"] = sum(1 for c in button_cands + target_cands + cube_cands if c["grid"] is not None)

# Hold the arm for two ticks so the moving cube can be told apart from the static button.
hold_q = [float(q) for q in state0.joint_positions["primary"]]
pending_frames = [(0, stamp0, rows0)]
terminal = {"flag": False, "why": ""}
clock = {"ticks": 0, "stamps": []}


def terminal_from(reply):
    for name in ("terminated", "truncated", "done", "terminal"):
        try:
            if name == "terminated" and bool(reply.terminated):
                return "terminated"
            if name == "truncated" and bool(reply.truncated):
                return "truncated"
            if name == "done" and bool(reply.done):
                return "done"
            if name == "terminal" and bool(reply.terminal):
                return "terminal"
        except Exception:
            pass
    return ""


def raw_step(q_target, grip):
    reply = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=[float(q) for q in q_target],
        gripper_position=grip, embodiment=embodiment)}))
    clock["ticks"] += 1
    why = terminal_from(reply)
    if why != "":
        terminal["flag"] = True
        terminal["why"] = why
    return reply


for hold_tick in range(2):
    raw_step(hold_q, 0.0)
    cam_now, stamp_now = front_observation()
    pending_frames.append((clock["ticks"], stamp_now, depth_rows_of(cam_now)))
    if terminal["flag"]:
        break

rows_last = pending_frames[-1][2]
for cand in button_cands + cube_cands + target_cands:
    cand["change"] = change_fraction(cand, rows0, rows_last)
    cand["moved"] = cand["change"]

# Cube first: the candidate whose own pixels changed most between the frames (the cube moves
# continuously); fall back to the best raised cube-prompt candidate.
cube = None
movers = [c for c in cube_cands if c["change"] is not None and c["change"] > 0.15
          and c["info"]["raised"] >= 4]
if len(movers) > 0:
    cube = sorted(movers, key=lambda c: -c["change"])[0]
cube_change = cube["change"] if cube is not None else 0.0

# Button: raised candidate that is clearly more static than the cube and is not the cube.
button = None
for cand in sorted(button_cands, key=lambda c: -c["score"]):
    info = cand["info"]
    if info["raised"] < 4:
        continue
    if cube is not None and box_iou(cand["box"], cube["box"]) > 0.3:
        continue
    if cand["change"] is not None and cand["change"] > max(0.15, 0.5 * cube_change):
        continue
    button = cand
    break
if button is None:
    for cand in sorted(button_cands, key=lambda c: -c["score"]):
        if cube is None or box_iou(cand["box"], cube["box"]) <= 0.3:
            if cand["info"]["raised"] >= 4:
                button = cand
                report["notes"].append("button fallback: raised non-cube candidate")
                break
if button is None and len(button_cands) > 0:
    button = sorted(button_cands, key=lambda c: -c["score"])[0]
    report["notes"].append("button fallback: top-score candidate")
if button is None:
    raise ValueError("button not found in front camera")

if cube is None:
    for cand in sorted(cube_cands, key=lambda c: -c["score"]):
        if cand["info"]["raised"] >= 4 and box_iou(cand["box"], button["box"]) < 0.3:
            cube = cand
            report["notes"].append("cube fallback: no moving candidate")
            break
report["roles"] = {"button": [round(x) for x in button["box"]] + [button["change"]],
                   "cube": None if cube is None else [round(x) for x in cube["box"]] + [cube["change"]]}
cube_height = None
if cube is not None:
    raised_h = [s[0] for s in cand_samples(cube, rows0) if s[0] > 0.006]
    cube_height = quantile(raised_h, 0.9)
report["cube_height_m"] = cube_height

# Target: flat candidate (not the button, not the cube), best score first.
target = None
flat_limit = 0.4 * cube_height if cube_height is not None else None
for cand in sorted(target_cands, key=lambda c: -c["score"]):
    if box_iou(cand["box"], button["box"]) > 0.3:
        continue
    if cube is not None and box_iou(cand["box"], cube["box"]) > 0.5:
        continue
    if flat_limit is not None and cand["info"]["h50"] is not None and cand["info"]["h50"] > flat_limit:
        continue
    target = cand
    break
if target is None:
    raise ValueError("target not found in front camera")


def target_geometry(rows, box, band_low):
    samples = box_samples(rows, plane, box, 1)
    flat = [s for s in samples if band_low is None or s[0] < band_low]
    if len(flat) < 4:
        flat = samples
    centre = [sum(s[1][i] for s in flat) / len(flat) for i in range(3)]
    along = [vdot(vsub(s[1], centre), plane["e1"]) for s in flat]
    radius = 0.5 * (max(along) - min(along))
    thickness = quantile([s[0] for s in flat], 0.75)
    return centre, radius, thickness


band_guess = 0.5 * cube_height if cube_height is not None else None
target_centre, target_radius, target_thick = target_geometry(rows0, target["box"], band_guess)
report["target_box"] = [round(x, 1) for x in target["box"]]
report["target_radius_m"] = round(target_radius, 4)
report["target_thickness_m"] = round(target_thick, 4)

# Button footprint (camera/plane frame) for excluding the button and the hovering hand.
button_samples = [s for s in cand_samples(button, rows0) if s[0] > 0.003]
if len(button_samples) > 0:
    button_foot = [sum(s[1][i] for s in button_samples) / len(button_samples) for i in range(3)]
    button_reach = quantile([vnorm(vsub(s[1], button_foot)) for s in button_samples], 0.9)
    report["button_exclusion_m"] = round(button_reach + 0.03, 4)
else:
    button_foot = None
    button_reach = 0.0

# ----------------------------------------------------------------------------- tracker
tw = target["box"][2] - target["box"][0]
roi_half = max(4.0 * tw, 40.0)
tcu = 0.5 * (target["box"][0] + target["box"][2])
tcv = 0.5 * (target["box"][1] + target["box"][3])
roi = (max(0, int(tcu - roi_half)), max(0, int(tcv - roi_half)),
       min(img_w - 1, int(tcu + roi_half)), min(img_h - 1, int(tcv + roi_half)))
tracker = {"band_low": None, "band_high": None, "hist": [], "visits": [], "inside": False,
           "dmin": None, "kmin": None, "missing": 0, "frames": 0, "last_d": None}


def set_band(height):
    low = 0.6 * height
    if target_thick is not None and target_thick + 0.004 > low:
        low = target_thick + 0.5 * max(0.0, height - target_thick)
    tracker["band_low"] = low
    tracker["band_high"] = 1.3 * height + 0.01
    tracker["cube_h"] = height
    tracker["thr_in"] = target_radius + 0.4 * height
    tracker["thr_out"] = target_radius + 0.7 * height


if cube_height is not None and cube_height > 0.005:
    set_band(cube_height)


def scan(rows):
    plane_a = plane["a"]
    plane_b = plane["b"]
    plane_c = plane["c"]
    nn = plane["nn"]
    if tracker["band_low"] is None:
        low = 0.008
        high = 0.2
    else:
        low = tracker["band_low"]
        high = tracker["band_high"]
    pts = []
    for v in range(roi[1], roi[3] + 1, 2):
        row = rows[v]
        row_term = plane_b * v + plane_c
        for u in range(roi[0], roi[2] + 1, 2):
            z = row[u]
            if z <= 0.0 or z > 50.0:
                continue
            h = (1.0 - z * (plane_a * u + row_term)) / nn
            if h < low or h > high:
                continue
            foot, hh = footprint(plane, float(u), float(v), z)
            if button_foot is not None and vnorm(vsub(foot, button_foot)) < button_reach + 0.03:
                continue
            pts.append((h, foot))
    return pts


def track(tick, stamp, rows):
    # Each `step` is exactly one control tick, so the tick index is the number of steps issued;
    # the observation timestamp is recorded alongside as a cross-check.
    if len(clock["stamps"]) < 4 or tick % 50 == 0:
        clock["stamps"].append((tick, round(stamp, 3)))
    pts = scan(rows)
    tracker["frames"] += 1
    if tracker["band_low"] is None:
        if len(pts) >= 6:
            set_band(quantile([p[0] for p in pts], 0.9))
            pts = [p for p in pts if p[0] >= tracker["band_low"] and p[0] <= tracker["band_high"]]
        else:
            tracker["missing"] += 1
            return None
    if len(pts) < 3:
        tracker["missing"] += 1
        return None
    mx = quantile([vdot(vsub(p[1], target_centre), plane["e1"]) for p in pts], 0.5)
    my = quantile([vdot(vsub(p[1], target_centre), plane["e2"]) for p in pts], 0.5)
    near = []
    for p in pts:
        q = vsub(p[1], target_centre)
        px = vdot(q, plane["e1"])
        py = vdot(q, plane["e2"])
        if math.sqrt((px - mx) ** 2 + (py - my) ** 2) < 1.5 * tracker["cube_h"] + 0.01:
            near.append((px, py))
    if len(near) < 3:
        tracker["missing"] += 1
        return None
    px = sum(p[0] for p in near) / len(near)
    py = sum(p[1] for p in near) / len(near)
    d = math.sqrt(px * px + py * py)
    tracker["hist"].append((tick, px, py))
    tracker["last_d"] = d
    if not tracker["inside"]:
        if d < tracker["thr_in"] and tracker["frames"] > 1:
            tracker["inside"] = True
            tracker["dmin"] = d
            tracker["kmin"] = tick
            tracker["visits"].append({"enter": tick, "ca": tick, "dmin": d, "exit": None})
        elif d < tracker["thr_in"]:
            tracker["inside"] = True
            tracker["dmin"] = d
            tracker["kmin"] = tick
            tracker["start_inside"] = True
    else:
        if d < tracker["dmin"]:
            tracker["dmin"] = d
            tracker["kmin"] = tick
            if len(tracker["visits"]) > 0 and tracker["visits"][-1]["exit"] is None:
                tracker["visits"][-1]["ca"] = tick
                tracker["visits"][-1]["dmin"] = d
        if d > tracker["thr_out"]:
            tracker["inside"] = False
            if len(tracker["visits"]) > 0 and tracker["visits"][-1]["exit"] is None:
                tracker["visits"][-1]["exit"] = tick
    return d


for frame in pending_frames:
    track(frame[0], frame[1], frame[2])

report["tracker_roi"] = list(roi)
report["thresholds"] = {"thr_in": tracker.get("thr_in"), "thr_out": tracker.get("thr_out"),
                        "band_low": tracker["band_low"], "band_high": tracker["band_high"]}


def predict_ca():
    """Ticks from the newest observation to the next closest approach, or None."""
    hist = tracker["hist"]
    if len(hist) < 2 or tracker["band_low"] is None:
        return None, "none"
    k1, x1, y1 = hist[-1]
    k0, x0, y0 = hist[-2]
    dk = k1 - k0
    if dk <= 0 or dk > 3:
        return None, "none"
    vx = (x1 - x0) / dk
    vy = (y1 - y0) / dk
    best = None
    closed = [vis for vis in tracker["visits"] if vis["exit"] is not None]
    if len(closed) > 0:
        for index in range(1, len(hist)):
            ka, xa, ya = hist[index]
            kb, xb, yb = hist[index - 1]
            if ka - kb != dk or ka >= k1 - 2:
                continue
            nxt = None
            for vis in closed:
                if vis["ca"] >= ka:
                    nxt = vis
                    break
            if nxt is None or nxt["ca"] - ka > 20:
                continue
            wx = (xa - xb) / dk
            wy = (ya - yb) / dk
            err = math.sqrt((xa - x1) ** 2 + (ya - y1) ** 2) + 2.0 * math.sqrt((wx - vx) ** 2 + (wy - vy) ** 2)
            if best is None or err < best[0]:
                best = (err, nxt["ca"] - ka)
    if best is not None and best[0] < 0.25 * tracker["thr_in"]:
        return float(best[1]), "analogue"
    speed2 = vx * vx + vy * vy
    if speed2 <= 1e-10:
        return None, "none"
    dot = x1 * vx + y1 * vy
    if dot >= 0.0:
        return None, "receding"
    tau = -dot / speed2
    miss = math.sqrt((x1 + tau * vx) ** 2 + (y1 + tau * vy) ** 2)
    if miss > tracker["thr_in"]:
        return None, "miss"
    return tau, "linear"


# ----------------------------------------------------------------------------- arm helpers
def ee_pose():
    st = get_robot_state()
    return st.end_effector_poses["primary"]


def current_q():
    return [float(q) for q in get_robot_state().joint_positions["primary"]]


home_pose = ee_pose()
quat = [float(x) for x in home_pose.quaternion_wxyz]
qw, qx, qy, qz = quat
tool_z = [2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)]
if tool_z[2] > -0.9:
    quat = [0.0, 1.0, 0.0, 0.0]
    report["notes"].append("home tool axis not downward; using a downward quaternion")


def ik_for(xyz, seed):
    pose = Pose([float(xyz[0]), float(xyz[1]), float(xyz[2])], quat, base_frame)
    try:
        res = solve_ik(pose, seed_joints=seed)
    except Exception:
        res = solve_ik(pose)
    if not res.ok:
        res = solve_ik(pose)
    if not res.ok:
        return None
    try:
        return [float(q) for q in res.joint_positions]
    except Exception:
        return [float(q) for q in res.joints]


def observe_tick():
    cam, stamp = front_observation()
    return track(clock["ticks"], stamp, depth_rows_of(cam)), stamp


prep = {"active": False, "urgent": False, "vpeak": 0.0, "prev": None, "why": ""}


def prep_check():
    # During preparation (stages A-C): is the N-th arrival due before the remaining preparation ends?
    pos = [float(x) for x in ee_pose().position]
    if prep["prev"] is not None:
        prep["vpeak"] = max(prep["vpeak"], vnorm(vsub(pos, prep["prev"])))
    prep["prev"] = pos
    entered = len(tracker["visits"])
    if entered == target_count and tracker["inside"]:
        prep["urgent"] = True
        prep["why"] = "target visit started during preparation"
    elif entered == target_count - 1 and not tracker["inside"]:
        tau, method = predict_ca()
        if tau is not None:
            reg = [cap[0], cap[1], cap[2] - REG_DEPTH]
            speed = max(prep["vpeak"], 0.004)
            need = 2.0 + math.ceil(vnorm(vsub(pos, reg)) / speed)
            if tau <= need + 0.5:
                prep["urgent"] = True
                prep["why"] = method + " tau=" + str(round(tau, 2)) + " need=" + str(need)
    if prep["urgent"]:
        report["urgent"] = {"tick": clock["ticks"], "why": prep["why"], "vpeak": round(prep["vpeak"], 4)}


def stop_now():
    return terminal["flag"] or prep["urgent"]


def tick_to(q_target, grip):
    raw_step(q_target, grip)
    out = observe_tick()
    if prep["active"] and not prep["urgent"] and not terminal["flag"]:
        prep_check()
    return out


def glide(q_goal, max_delta, settle_xyz, settle_ticks):
    q0 = current_q()
    span = max(abs(q_goal[i] - q0[i]) for i in range(len(q0)))
    count = max(2, int(math.ceil(span / max_delta)))
    for index in range(1, count + 1):
        if stop_now():
            return
        frac = float(index) / count
        tick_to([q0[i] + frac * (q_goal[i] - q0[i]) for i in range(len(q0))], 0.0)
    for index in range(settle_ticks):
        if stop_now():
            return
        pos = [float(x) for x in ee_pose().position]
        if settle_xyz is not None and vnorm(vsub(pos, settle_xyz)) < 0.004:
            return
        tick_to(q_goal, 0.0)


# ----------------------------------------------------------------------------- button geometry in base frame
def button_cloud(camera_name, cand):
    pts = None
    if cand is not None:
        try:
            cloud = mask_to_point_cloud(cand["item"], camera_name, target_frame=base_frame)
            pts = [[float(c) for c in p] for p in cloud.points]
        except Exception:
            pts = None
    if pts is None or len(pts) < 20:
        for prompt in BUTTON_PROMPTS:
            try:
                loc = localize_object(prompt, camera_name=camera_name, target_frame=base_frame)
            except Exception:
                continue
            if loc.ok and len(loc.point_cloud.points) >= 20:
                pts = [[float(c) for c in p] for p in loc.point_cloud.points]
                break
    return pts


def cap_from(points):
    zs = [p[2] for p in points]
    top = quantile(zs, 0.95)
    cap = [p for p in points if p[2] > top - 0.006]
    cx = sum(p[0] for p in cap) / len(cap)
    cy = sum(p[1] for p in cap) / len(cap)
    return [cx, cy, top]


front_pts = button_cloud("front", button)
if front_pts is None or len(front_pts) < 20:
    raise ValueError("button point cloud unavailable")
cap = cap_from(front_pts)
report["cap_front"] = [round(x, 4) for x in cap]

GAP = 0.02          # fingertip clearance above the cap while waiting
PRESS_DEPTH = 0.02  # commanded depth below the cap top
REG_DEPTH = 0.004   # assumed travel into the cap before the press registers
TRANSIT_UP = 0.07   # transit height above the front-camera cap estimate

# Stage A: close the gripper and move above the button.
report["stages"].append(["transit_start", tracker["hist"][-1][0] if tracker["hist"] else None])
above = [cap[0], cap[1], cap[2] + TRANSIT_UP]
prep["active"] = True
q_above = ik_for(above, current_q())
if q_above is None:
    raise ValueError("IK failed above the button")
glide(q_above, 0.06, above, 15)


def settle_at(q_goal, max_ticks):
    # Hold the joint target until the end effector is at rest (two consecutive ticks moving < 0.5 mm).
    prev = [float(x) for x in ee_pose().position]
    still = 0
    for index in range(max_ticks):
        if stop_now():
            return index
        tick_to(q_goal, 0.0)
        pos = [float(x) for x in ee_pose().position]
        if vnorm(vsub(pos, prev)) < 0.0005:
            still += 1
            if still >= 2:
                return index + 1
        else:
            still = 0
        prev = pos
    return max_ticks


report["settle_ticks_above"] = settle_at(q_above, 30)

# Stage B: refine the cap from the wrist camera.
wrist_pts = None
if not stop_now():
    wrist_pts = button_cloud("wrist", None)
if wrist_pts is not None and len(wrist_pts) >= 20:
    cap_w = cap_from(wrist_pts)
    if vnorm([cap_w[0] - cap[0], cap_w[1] - cap[1], 0.0]) < 0.05 and abs(cap_w[2] - cap[2]) < 0.04:
        cap = cap_w
        report["cap_wrist"] = [round(x, 4) for x in cap_w]
    else:
        report["notes"].append("wrist cap estimate rejected: " + str([round(x, 4) for x in cap_w]))

hover_xyz = [cap[0], cap[1], cap[2] + GAP]
press_xyz = [cap[0], cap[1], cap[2] - PRESS_DEPTH]
q_hover = ik_for(hover_xyz, current_q())
if q_hover is None:
    raise ValueError("IK failed at hover")
q_press = ik_for(press_xyz, q_hover)
if q_press is None:
    raise ValueError("IK failed at press")

# Stage C: calibrate the vertical step response with a single free-air jump to the hover target.
latency = 3
if not stop_now():
    z_start = float(ee_pose().position[2])
    profile = []
    still = 0
    for cal_tick in range(30):
        tick_to(q_hover, 0.0)
        if stop_now():
            break
        z_now = float(ee_pose().position[2])
        profile.append(z_now)
        if len(profile) >= 2 and abs(profile[-1] - profile[-2]) < 0.0005:
            still += 1
            if still >= 2:
                break
        else:
            still = 0
    calibrated = False
    if len(profile) > 0:
        jump = z_start - profile[-1]
        if jump > 0.01:
            # Linear step-response scaling: the press jump (GAP + PRESS_DEPTH) must cover the
            # distance from the measured hover height to the registration level.
            needed = (profile[-1] - (cap[2] - REG_DEPTH)) / (GAP + PRESS_DEPTH)
            for index in range(len(profile)):
                if (z_start - profile[index]) / jump >= needed:
                    latency = index + 1
                    calibrated = True
                    break
        report["calibration"] = {"z_start": round(z_start, 4), "profile": [round(z, 4) for z in profile],
                                 "latency_ticks": latency, "calibrated": calibrated}
    if not calibrated:
        report["notes"].append("press latency not calibrated; using default " + str(latency))
report["latency_ticks"] = latency
prep["active"] = False

# Stage D: hover, count visits, trigger the press for the N-th visit.
pressed = False
press_info = {}
wait_ticks = 0
while not stop_now():
    hist = tracker["hist"]
    k_now = hist[-1][0] if len(hist) > 0 else None
    entered = len(tracker["visits"])
    target_upcoming = (entered == target_count - 1 and not tracker["inside"])
    target_current = (entered == target_count and tracker["inside"])
    fire = False
    reason = ""
    if target_upcoming or target_current:
        tau, method = predict_ca()
        if tau is not None and tau <= latency + 0.5:
            fire = True
            reason = method + " tau=" + str(round(tau, 2))
        elif target_current and method == "receding":
            fire = True
            reason = "inside and receding"
    elif entered >= target_count and not tracker["inside"]:
        fire = True
        reason = "target visit already passed (late)"
    if max_steps is not None and k_now is not None and k_now >= max_steps - latency - 2:
        fire = True
        reason = "horizon guard"
    if fire:
        press_info = {"tick": k_now, "reason": reason, "visits_entered": entered,
                      "d": tracker["last_d"]}
        break
    tick_to(q_hover, 0.0)
    wait_ticks += 1
    if wait_ticks > 5000:
        break

if prep["urgent"] and not terminal["flag"]:
    press_info = {"tick": clock["ticks"], "reason": "urgent: " + prep["why"],
                  "visits_entered": len(tracker["visits"]), "d": tracker["last_d"]}
    q_now_press = ik_for(press_xyz, current_q())
    if q_now_press is not None:
        q_press = q_now_press
if not terminal["flag"]:
    pressed = True
    for press_tick in range(40 if prep["urgent"] else 12):
        target_q = q_press
        if press_tick >= 6:
            deeper = ik_for([cap[0], cap[1], cap[2] - 2.0 * PRESS_DEPTH], q_press)
            if deeper is not None:
                target_q = deeper
        tick_to(target_q, 0.0)
        if terminal["flag"]:
            press_info["ticks_to_terminal"] = press_tick + 1
            break

report["pressed"] = pressed
report["press"] = press_info
report["terminal"] = terminal["why"]
report["visits"] = [{"enter": v["enter"], "ca": v["ca"], "exit": v["exit"], "dmin": round(v["dmin"], 4)}
                    for v in tracker["visits"]]
report["tracked_frames"] = tracker["frames"]
report["ticks_issued"] = clock["ticks"]
report["clock_check"] = clock["stamps"]
report["missing_frames"] = tracker["missing"]
result = report
