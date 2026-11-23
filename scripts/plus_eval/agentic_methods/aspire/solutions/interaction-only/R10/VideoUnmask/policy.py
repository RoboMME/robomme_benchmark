# v02 — VideoUnmask: v01 + height-banded grasp-clearance score.
#
# Single change from v01 (execution/clearance defect seen in the v01 seed-0 trace): the
# finger/hand clearance score counted every observed point above the grasp height,
# including the robot's own arm and hand at the home pose (hundreds of points), so the
# yaw choice could be decided by the robot rather than by neighbouring containers or
# occluders. Points are now counted only inside the height band the hand actually
# sweeps during the vertical descent from hover: z <= top_z + 0.22 m.
#
# Changes from v00 (failure modes seen on dev seeds 1 and 2):
#   perception/video : cubes occluded in the front view (tall occluders) were missed
#                      -> scan BOTH conditioning cameras (front + wrist), project each
#                      blob with that frame's own depth + calibration, merge in 3D.
#   task predicate   : "all containers hiding blue cubes" -> every cube of a colour.
#   perception/live  : SAM3 returns at most 5 masks and scored containers poorly when
#                      there are 10+ of them -> fit each target container directly from
#                      live RGB-D (front + wrist) around the remembered cube position.
#   execution        : neighbouring containers / occluders -> choose the grasp yaw
#                      (theta or theta+90) whose fingers and hand hit fewer observed
#                      points; escalate hover height if a transit move fails.
import math

COLOR_HUES = {
    "red": 0.0, "orange": 38.0, "yellow": 60.0, "green": 120.0, "cyan": 180.0,
    "blue": 240.0, "purple": 275.0, "violet": 275.0, "magenta": 300.0, "pink": 330.0,
}
SCAN_STRIDE = 4
VIVID_S = 0.85
VIVID_V = 0.2
TRANSIENT_DIFF = 60
HUE_TOL = 20.0
NAME_HUE_TOL = 30.0
CLUSTER_PX = 20

report = {"motions": [], "notes": []}
episode_over = {"flag": False}


def note(msg):
    report["notes"].append(str(msg))


def sub_rows(arr, stride):
    try:
        return arr[::stride, ::stride].tolist()
    except Exception:
        rows = arr.tolist()
        out = []
        for y in range(0, len(rows), stride):
            out.append(rows[y][::stride])
        return out


def rgb_hsv(p):
    r = float(p[0])
    g = float(p[1])
    b = float(p[2])
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx <= 0.0:
        return 0.0, 0.0, 0.0
    d = mx - mn
    s = d / mx
    v = mx / 255.0
    if d <= 0.0:
        return 0.0, s, v
    if mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * (((b - r) / d) + 2.0)
    else:
        h = 60.0 * (((r - g) / d) + 4.0)
    return h, s, v


def hue_dist(a, b):
    d = abs(a - b) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return d


def quantile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def quat_to_rot(q):
    w, x, y, z = (float(v) for v in q)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def quat_mul(a, b):
    a1, b1, c1, d1 = a
    a2, b2, c2, d2 = b
    return [
        a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
        a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
        a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
        a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
    ]


def cam_model(cam):
    K = cam.intrinsics.tolist()
    return {
        "fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]), "cy": float(K[1][2]),
        "R": quat_to_rot(cam.camera_pose.quaternion_wxyz),
        "t": [float(val) for val in cam.camera_pose.position],
    }


def pix_to_base(m, u, v, depth):
    xc = (u - m["cx"]) / m["fx"] * depth
    yc = (v - m["cy"]) / m["fy"] * depth
    R = m["R"]
    t = m["t"]
    return [
        R[0][0] * xc + R[0][1] * yc + R[0][2] * depth + t[0],
        R[1][0] * xc + R[1][1] * yc + R[1][2] * depth + t[1],
        R[2][0] * xc + R[2][1] * yc + R[2][2] * depth + t[2],
    ]


def base_to_pix(m, p):
    R = m["R"]
    t = m["t"]
    d0 = p[0] - t[0]
    d1 = p[1] - t[1]
    d2 = p[2] - t[2]
    xc = R[0][0] * d0 + R[1][0] * d1 + R[2][0] * d2
    yc = R[0][1] * d0 + R[1][1] * d1 + R[2][1] * d2
    zc = R[0][2] * d0 + R[1][2] * d1 + R[2][2] * d2
    if zc <= 0.01:
        return None
    return [m["fx"] * xc / zc + m["cx"], m["fy"] * yc / zc + m["cy"], zc]


def parse_targets(language):
    text = ""
    for ch in language.lower():
        if ch.isalpha():
            text = text + ch
        else:
            text = text + " "
    order = []
    for tok in text.split():
        if tok in COLOR_HUES and tok not in order:
            order.append(tok)
    return order


def nearest_name(hue):
    best = None
    best_d = None
    for name in COLOR_HUES:
        d = hue_dist(hue, COLOR_HUES[name])
        if best_d is None or d < best_d:
            best_d = d
            best = name
    if best_d is not None and best_d <= NAME_HUE_TOL:
        if best == "violet":
            return "purple"
        return best
    return None


# ---------------------------------------------------------------- 1. goal
context = get_task_context()
language = context.language
targets = parse_targets(language)
report["language"] = language
report["targets"] = targets
state0 = get_robot_state()
base_frame = state0.base_frame
home_quat = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]

# ---------------------------------------------------------------- 2. video cubes
video = robomme_plus.get_conditioning_video()
n_frames = len(video)
report["video_frames"] = n_frames


def scan_camera(cam_name):
    clusters = []
    if n_frames == 0:
        return []
    first_sub = sub_rows(video[0].cameras[cam_name].rgb, SCAN_STRIDE)
    last_sub = sub_rows(video[n_frames - 1].cameras[cam_name].rgb, SCAN_STRIDE)
    for fi in range(n_frames):
        rows = sub_rows(video[fi].cameras[cam_name].rgb, SCAN_STRIDE)
        for yi in range(len(rows)):
            row = rows[yi]
            for xi in range(len(row)):
                p = row[xi]
                mx = max(p[0], p[1], p[2])
                mn = min(p[0], p[1], p[2])
                if mx < 51 or (mx - mn) < VIVID_S * mx:
                    continue
                pl = last_sub[yi][xi]
                pf = first_sub[yi][xi]
                dl = abs(p[0] - pl[0]) + abs(p[1] - pl[1]) + abs(p[2] - pl[2])
                df = abs(p[0] - pf[0]) + abs(p[1] - pf[1]) + abs(p[2] - pf[2])
                if dl < TRANSIENT_DIFF and df < TRANSIENT_DIFF:
                    continue
                h, s, v = rgb_hsv(p)
                u = xi * SCAN_STRIDE
                vv = yi * SCAN_STRIDE
                found = None
                for c in clusters:
                    if hue_dist(c["hue"], h) <= HUE_TOL:
                        if abs(c["su"] / c["n"] - u) <= CLUSTER_PX and abs(c["sv"] / c["n"] - vv) <= CLUSTER_PX:
                            found = c
                            break
                if found is None:
                    found = {"hue": h, "n": 0, "su": 0.0, "sv": 0.0, "fc": {}, "hs": 0.0, "hc": 0.0}
                    clusters.append(found)
                found["n"] += 1
                found["su"] += u
                found["sv"] += vv
                found["hs"] += math.sin(math.radians(h))
                found["hc"] += math.cos(math.radians(h))
                found["hue"] = math.degrees(math.atan2(found["hs"], found["hc"])) % 360.0
                key = str(fi)
                if key in found["fc"]:
                    found["fc"][key] += 1
                else:
                    found["fc"][key] = 1
    cubes = []
    for c in clusters:
        frames = sorted(int(k) for k in c["fc"])
        if c["n"] < 3 or len(frames) < 2:
            continue
        best_frame = frames[0]
        for f in frames:
            if c["fc"][str(f)] >= c["fc"][str(best_frame)]:
                best_frame = f
        cam = video[best_frame].cameras[cam_name]
        m = cam_model(cam)
        cu = c["su"] / c["n"]
        cv = c["sv"] / c["n"]
        y0 = max(0, int(cv) - 26)
        x0 = max(0, int(cu) - 26)
        try:
            rgb_win = cam.rgb[y0:int(cv) + 27, x0:int(cu) + 27].tolist()
            dep_win = cam.depth_m[y0:int(cv) + 27, x0:int(cu) + 27].tolist()
        except Exception:
            rgb_all = cam.rgb.tolist()
            dep_all = cam.depth_m.tolist()
            rgb_win = [row[x0:int(cu) + 27] for row in rgb_all[y0:int(cv) + 27]]
            dep_win = [row[x0:int(cu) + 27] for row in dep_all[y0:int(cv) + 27]]
        us = []
        vs = []
        ds = []
        for yy in range(len(rgb_win)):
            for xx in range(len(rgb_win[yy])):
                h, s, v = rgb_hsv(rgb_win[yy][xx])
                if s >= VIVID_S and v >= VIVID_V and hue_dist(h, c["hue"]) <= HUE_TOL:
                    d = float(dep_win[yy][xx])
                    if d > 0.0:
                        us.append(x0 + xx)
                        vs.append(y0 + yy)
                        ds.append(d)
        if len(us) < 3:
            continue
        pu = sum(us) / len(us)
        pv = sum(vs) / len(vs)
        xyz = pix_to_base(m, pu, pv, quantile(ds, 0.5))
        cubes.append({
            "cam": cam_name, "hue": round(c["hue"], 1), "hits": c["n"],
            "first_frame": frames[0], "best_frame": best_frame, "npx": len(us),
            "px": [round(pu, 1), round(pv, 1)], "xyz": [round(val, 4) for val in xyz],
        })
    return cubes


raw_cubes = []
for cam_name in ("front", "wrist"):
    try:
        raw_cubes = raw_cubes + scan_camera(cam_name)
    except Exception as exc:
        note("scan " + cam_name + " failed: " + str(exc))
report["raw_cubes"] = raw_cubes

# merge detections of the same cube across cameras (same hue, close in XY)
cubes = []
for rc in raw_cubes:
    merged = None
    for c in cubes:
        dx = c["xy"][0] - rc["xyz"][0]
        dy = c["xy"][1] - rc["xyz"][1]
        if hue_dist(c["hue"], rc["hue"]) <= HUE_TOL and math.sqrt(dx * dx + dy * dy) <= 0.045:
            merged = c
            break
    if merged is None:
        cubes.append({"hue": rc["hue"], "xy": [rc["xyz"][0], rc["xyz"][1]], "z": rc["xyz"][2],
                      "w": rc["npx"], "first_frame": rc["first_frame"], "cams": [rc["cam"]]})
    else:
        w = merged["w"] + rc["npx"]
        merged["xy"] = [
            (merged["xy"][0] * merged["w"] + rc["xyz"][0] * rc["npx"]) / w,
            (merged["xy"][1] * merged["w"] + rc["xyz"][1] * rc["npx"]) / w,
        ]
        merged["w"] = w
        merged["first_frame"] = min(merged["first_frame"], rc["first_frame"])
        merged["cams"].append(rc["cam"])
for c in cubes:
    c["name"] = nearest_name(c["hue"])
report["cubes"] = [
    {"name": c["name"], "hue": c["hue"], "xy": [round(c["xy"][0], 4), round(c["xy"][1], 4)],
     "z": c["z"], "first_frame": c["first_frame"], "cams": c["cams"]}
    for c in cubes
]

# ---------------------------------------------------------------- 3. live scene cloud
obs = get_observation()
live = {}
for cam_name in ("front", "wrist"):
    try:
        cam = obs.cameras[cam_name]
        live[cam_name] = {"m": cam_model(cam), "depth": cam.depth_m.tolist()}
    except Exception as exc:
        note("live camera " + cam_name + ": " + str(exc))

scene = []
for cam_name in live:
    m = live[cam_name]["m"]
    drows = live[cam_name]["depth"]
    for v in range(0, len(drows), 2):
        row = drows[v]
        for u in range(0, len(row), 2):
            d = float(row[u])
            if d <= 0.0 or d > 3.0:
                continue
            scene.append(pix_to_base(m, u, v, d))

table_z = 0.0
front_z = []
if "front" in live:
    m = live["front"]["m"]
    drows = live["front"]["depth"]
    for v in range(0, len(drows), 4):
        for u in range(0, len(drows[v]), 4):
            d = float(drows[v][u])
            if d > 0.0 and d < 3.0:
                front_z.append(pix_to_base(m, u, v, d)[2])
if len(front_z) > 0:
    table_z = quantile(front_z, 0.5)
report["table_z"] = round(table_z, 4)

# container-top height = most populated 5 mm bin of elevated points
hist = {}
for p in scene:
    dz = p[2] - table_z
    if dz > 0.03 and dz < 0.25:
        key = int(dz / 0.005)
        if key in hist:
            hist[key] += 1
        else:
            hist[key] = 1
top_ref = None
best_count = 0
for key in hist:
    if hist[key] > best_count:
        best_count = hist[key]
        top_ref = table_z + (key + 0.5) * 0.005
report["top_ref"] = None if top_ref is None else round(top_ref, 4)


def local_points(center_xy, radius):
    pts = []
    for cam_name in live:
        m = live[cam_name]["m"]
        drows = live[cam_name]["depth"]
        guess_z = top_ref if top_ref is not None else table_z + 0.05
        pix = base_to_pix(m, [center_xy[0], center_xy[1], guess_z])
        if pix is None:
            continue
        hw = int(m["fx"] * (radius + 0.02) / pix[2]) + 3
        u0 = max(0, int(pix[0]) - hw)
        u1 = min(len(drows[0]), int(pix[0]) + hw + 1)
        v0 = max(0, int(pix[1]) - hw)
        v1 = min(len(drows), int(pix[1]) + hw + 1)
        for v in range(v0, v1):
            row = drows[v]
            for u in range(u0, u1):
                d = float(row[u])
                if d <= 0.0 or d > 3.0:
                    continue
                p = pix_to_base(m, u, v, d)
                dx = p[0] - center_xy[0]
                dy = p[1] - center_xy[1]
                if dx * dx + dy * dy <= radius * radius:
                    pts.append(p)
    return pts


def rect_fit(top):
    best = None
    for deg in range(0, 90, 3):
        t = math.radians(deg)
        ca = math.cos(t)
        sa = math.sin(t)
        us = [p[0] * ca + p[1] * sa for p in top]
        vs = [-p[0] * sa + p[1] * ca for p in top]
        u_lo = quantile(us, 0.05)
        u_hi = quantile(us, 0.95)
        v_lo = quantile(vs, 0.05)
        v_hi = quantile(vs, 0.95)
        area = (u_hi - u_lo) * (v_hi - v_lo)
        if best is None or area < best["area"]:
            uc = 0.5 * (u_lo + u_hi)
            vc = 0.5 * (v_lo + v_hi)
            best = {"area": area, "deg": deg, "span": [u_hi - u_lo, v_hi - v_lo],
                    "center": [uc * ca - vc * sa, uc * sa + vc * ca]}
    return best


def fit_container(cube_xy):
    if top_ref is None:
        return None
    c = [cube_xy[0], cube_xy[1]]
    pts = local_points(c, 0.07)
    top_pts = [p for p in pts if abs(p[2] - top_ref) <= 0.015]
    sel = []
    for it in range(8):
        sel = []
        for p in top_pts:
            dx = p[0] - c[0]
            dy = p[1] - c[1]
            if dx * dx + dy * dy <= 0.03 * 0.03:
                sel.append(p)
        if len(sel) < 5:
            return None
        nc = [sum(p[0] for p in sel) / len(sel), sum(p[1] for p in sel) / len(sel)]
        moved = math.sqrt((nc[0] - c[0]) ** 2 + (nc[1] - c[1]) ** 2)
        c = nc
        if moved < 0.001:
            break
    fit = rect_fit(sel)
    zs = [p[2] for p in sel]
    return {
        "center": fit["center"], "yaw_deg": fit["deg"], "span": fit["span"],
        "top_z": quantile(zs, 0.9), "npts": len(sel), "shift": round(math.sqrt(
            (fit["center"][0] - cube_xy[0]) ** 2 + (fit["center"][1] - cube_xy[1]) ** 2), 4),
    }


def clearance_score(center, top_z, yaw_deg):
    grasp_z = top_z - 0.025
    t = math.radians(yaw_deg)
    # closing axis for a top-down hand rotated by yaw about +z (see quat below)
    ax = [math.sin(t), -math.cos(t)]
    bx = [math.cos(t), math.sin(t)]
    finger = 0
    hand = 0
    for p in scene:
        dx = p[0] - center[0]
        dy = p[1] - center[1]
        if dx * dx + dy * dy < 0.03 * 0.03:
            continue
        if p[2] > top_z + 0.22:
            continue
        a = dx * ax[0] + dy * ax[1]
        b = dx * bx[0] + dy * bx[1]
        if abs(b) < 0.015 and abs(abs(a) - 0.045) < 0.015 and p[2] > grasp_z - 0.012:
            finger += 1
        if abs(a) < 0.11 and abs(b) < 0.035 and p[2] > grasp_z + 0.04:
            hand += 1
    return 3 * finger + hand, finger, hand


# ---------------------------------------------------------------- 4. plan
plan = []
for name in targets:
    group = []
    for c in cubes:
        if c["name"] == name:
            pos = len(group)
            while pos > 0 and group[pos - 1]["first_frame"] > c["first_frame"]:
                pos -= 1
            group.insert(pos, c)
    if len(group) == 0:
        plan.append({"color": name, "container": None, "why": "no video cube of this colour"})
        continue
    for c in group:
        fit = None
        try:
            fit = fit_container(c["xy"])
        except Exception as exc:
            note("fit failed: " + str(exc))
        if fit is None:
            fit = {"center": [c["xy"][0], c["xy"][1]], "yaw_deg": 0, "span": None,
                   "top_z": top_ref if top_ref is not None else table_z + 0.07, "npts": 0,
                   "shift": 0.0, "fallback": True}
        dup = False
        for item in plan:
            if item.get("container") is not None:
                ox = item["container"]["center"][0] - fit["center"][0]
                oy = item["container"]["center"][1] - fit["center"][1]
                if math.sqrt(ox * ox + oy * oy) < 0.02:
                    dup = True
        if dup:
            note("duplicate container for " + name + " cube at " + str(c["xy"]))
            continue
        yaw = float(fit["yaw_deg"])
        while yaw >= 45.0:
            yaw -= 90.0
        while yaw < -45.0:
            yaw += 90.0
        options = []
        for cand in (yaw, yaw + 90.0 if yaw < 0.0 else yaw - 90.0):
            sc, fh, hh = clearance_score(fit["center"], fit["top_z"], cand)
            options.append({"yaw": cand, "score": sc, "finger": fh, "hand": hh})
        chosen = options[0]
        if options[1]["score"] < options[0]["score"]:
            chosen = options[1]
        plan.append({"color": name, "cube_xy": [round(c["xy"][0], 4), round(c["xy"][1], 4)],
                     "container": fit, "yaw": chosen["yaw"], "options": options})
report["plan"] = plan

# ---------------------------------------------------------------- 5. act
def ended(res):
    try:
        if bool(res.terminated) or bool(res.truncated):
            episode_over["flag"] = True
            return True
    except Exception:
        pass
    return False


def move(label, xyz, quat, tol=0.01, steps=150):
    if episode_over["flag"]:
        return None
    target = Pose((xyz[0], xyz[1], xyz[2]), quat, base_frame)
    res = move_to_pose(target, tolerance=tol, max_steps=steps)
    entry = {"label": label, "ok": bool(res.ok), "steps": int(res.steps_executed)}
    try:
        entry["status"] = str(res.status)
    except Exception:
        pass
    ended(res)
    if not episode_over["flag"]:
        try:
            ee = get_robot_state().end_effector_poses["primary"].position
            entry["ee"] = [round(float(val), 4) for val in ee]
        except Exception:
            pass
    entry["episode_over"] = episode_over["flag"]
    report["motions"].append(entry)
    return res


def grip(label, closing):
    if episode_over["flag"]:
        return None
    if closing:
        res = close_gripper()
    else:
        res = open_gripper()
    entry = {"label": label, "ok": bool(res.ok), "steps": int(res.steps_executed)}
    ended(res)
    if not episode_over["flag"]:
        try:
            entry["gripper"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
        except Exception:
            pass
    entry["episode_over"] = episode_over["flag"]
    report["motions"].append(entry)
    return res


picked = []
for item in plan:
    if episode_over["flag"]:
        break
    if item["container"] is None:
        note("skipping " + item["color"] + ": " + item["why"])
        continue
    box = item["container"]
    half = math.radians(item["yaw"]) / 2.0
    quat = quat_mul([math.cos(half), 0.0, 0.0, math.sin(half)], home_quat)
    cx = box["center"][0]
    cy = box["center"][1]
    top_z = box["top_z"]
    grasp_z = top_z - 0.025
    hover_z = top_z + 0.10
    tag = item["color"] + "@" + str(round(cx, 3)) + "," + str(round(cy, 3))
    grip("open " + tag, False)
    res = move("hover " + tag, (cx, cy, hover_z), quat, 0.015, 150)
    if res is not None and not res.ok and not episode_over["flag"]:
        st = get_robot_state().end_effector_poses["primary"].position
        move("escape-up " + tag, (float(st[0]), float(st[1]), hover_z + 0.12), quat, 0.02, 150)
        move("high-transit " + tag, (cx, cy, hover_z + 0.12), quat, 0.02, 150)
        move("hover-retry " + tag, (cx, cy, hover_z), quat, 0.015, 150)
    move("descend " + tag, (cx, cy, grasp_z), quat, 0.01, 120)
    grip("close " + tag, True)
    move("lift " + tag, (cx, cy, grasp_z + 0.12), quat, 0.015, 120)
    picked.append(tag)
    if episode_over["flag"]:
        break
    move("lower " + tag, (cx, cy, grasp_z + 0.005), quat, 0.01, 120)
    grip("release " + tag, False)
    move("retreat " + tag, (cx, cy, hover_z), quat, 0.015, 120)

report["picked"] = picked
report["episode_over"] = episode_over["flag"]
report["total_steps"] = sum(m["steps"] for m in report["motions"])
result = report
