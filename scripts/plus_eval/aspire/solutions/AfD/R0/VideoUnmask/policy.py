# VideoUnmask (robomme_plus task 14) -- v04: v01 + split of top faces merged with a touching neighbour (AfD).
# (v02/v03 tall-column detours were not carried forward: no offline benefit, see CHANGELOG.)
# Strategy: remember where each goal-coloured cube was last seen in the conditioning video (fused
# front+wrist RGB-D -> robot_base), then, in goal order, lift the container now standing at each
# remembered location (top-down, yaw from its top-face depth), put it back, release.
# Everything episode-dependent comes from the goal string, the conditioning video and live RGB-D.
import math

report = {"version": "v04", "events": []}


def note(msg):
    report["events"].append(str(msg))


# ---------------------------------------------------------------- small math helpers
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def qmat(q):
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))


def qyaw(angle):
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def quantile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def wrap(angle, period):
    # wrap into [-period/2, period/2)
    half = period / 2.0
    return ((angle + half) % period) - half


# ---------------------------------------------------------------- camera geometry
def as_rows(array):
    try:
        return array.tolist()
    except Exception:
        rows = []
        for v in range(len(array)):
            rows.append(list(array[v]))
        return rows


def cam_params(cam):
    k = cam.intrinsics
    pose = cam.camera_pose
    pos = pose.position
    return {"fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]), "cy": float(k[1][2]),
            "R": qmat(pose.quaternion_wxyz),
            "t": (float(pos[0]), float(pos[1]), float(pos[2])), "frame": str(pose.frame)}


def unproject(p, u, v, d, conv):
    xn = (u - p["cx"]) / p["fx"]
    yn = (v - p["cy"]) / p["fy"]
    z = d
    if conv[1] == "ray":
        z = d / math.sqrt(1.0 + xn * xn + yn * yn)
    x = xn * z
    y = yn * z
    if conv[0] == "cv":
        c = (x, y, z)
    elif conv[0] == "gl":
        c = (x, -y, -z)
    else:  # sapien / ROS camera link: x forward, y left, z up
        c = (z, -x, -y)
    r = p["R"]
    t = p["t"]
    return (r[0][0] * c[0] + r[0][1] * c[1] + r[0][2] * c[2] + t[0],
            r[1][0] * c[0] + r[1][1] * c[1] + r[1][2] * c[2] + t[1],
            r[2][0] * c[0] + r[2][1] * c[1] + r[2][2] * c[2] + t[2])


def project(p, point, conv):
    r = p["R"]
    t = p["t"]
    q = (point[0] - t[0], point[1] - t[1], point[2] - t[2])
    c = (r[0][0] * q[0] + r[1][0] * q[1] + r[2][0] * q[2],
         r[0][1] * q[0] + r[1][1] * q[1] + r[2][1] * q[2],
         r[0][2] * q[0] + r[1][2] * q[1] + r[2][2] * q[2])
    if conv[0] == "cv":
        x, y, z = c
    elif conv[0] == "gl":
        x, y, z = c[0], -c[1], -c[2]
    else:
        x, y, z = -c[1], -c[2], c[0]
    if z <= 1e-6:
        return None
    return (p["fx"] * x / z + p["cx"], p["fy"] * y / z + p["cy"])


def valid_depth(d):
    return d == d and 0.03 < d < 6.0


CONVENTIONS = (("cv", "z"), ("cv", "ray"), ("gl", "z"), ("gl", "ray"), ("sapien", "z"), ("sapien", "ray"))


def choose_convention(obs):
    """Pick the camera axis / depth convention under which the table is flattest (zero ticks)."""
    scores = []
    for conv in CONVENTIONS:
        total = 0.0
        medians = []
        usable = True
        for name in ("front", "wrist"):
            if name not in obs.cameras:
                continue
            cam = obs.cameras[name]
            p = cam_params(cam)
            depth = as_rows(cam.depth_m)
            h = len(depth)
            w = len(depth[0])
            zs = []
            for v in range(4, h, 12):
                for u in range(4, w, 12):
                    d = float(depth[v][u])
                    if valid_depth(d):
                        zs.append(unproject(p, u, v, d, conv)[2])
            if len(zs) < 20:
                usable = False
                continue
            lo = quantile(zs, 0.25)
            hi = quantile(zs, 0.75)
            med = quantile(zs, 0.5)
            total += hi - lo
            if med > p["t"][2]:
                total += 10.0  # table above the camera: impossible
            medians.append(med)
        if len(medians) == 2:
            total += abs(medians[0] - medians[1])
        if not usable or len(medians) == 0:
            total += 100.0
        scores.append((total, conv, medians))
    scores.sort(key=lambda s: s[0])
    return scores


# ---------------------------------------------------------------- goal parsing
COLOR_NAMES = ("red", "green", "blue", "yellow", "purple", "orange", "pink", "cyan")


def parse_goal(goal):
    text = " " + goal.lower().replace(",", " ,").replace(".", " .") + " "
    found = []
    for color in COLOR_NAMES + ("black", "white", "gray", "grey", "brown"):
        start = 0
        while True:
            pos = text.find(" " + color + " cube", start)
            if pos < 0:
                break
            after = pos + len(color) + 6
            plural = text[after:after + 1] == "s"
            seg_start = max(text.rfind(",", 0, pos), text.rfind(" then ", 0, pos),
                            text.rfind("followed by", 0, pos))
            if seg_start < 0:
                seg_start = 0
            if " all " in text[seg_start:pos + 1] or " every " in text[seg_start:pos + 1]:
                plural = True
            found.append((pos, color, plural))
            start = pos + 1
    found.sort(key=lambda f: f[0])
    return [{"color": f[1], "all": f[2]} for f in found]


# ---------------------------------------------------------------- colour detection
def is_color(color, r, g, b):
    if color == "red":
        return r >= 60 and g * 10 <= r * 3 and b * 10 <= r * 3
    if color == "green":
        return g >= 60 and r * 20 <= g * 9 and b * 20 <= g * 9
    if color == "blue":
        return b >= 60 and r * 20 <= b * 7 and g * 20 <= b * 7
    if color == "yellow":
        return r >= 100 and g * 5 >= r * 4 and r * 5 >= g * 4 and b * 10 <= min(r, g) * 3
    if color == "purple":
        return r >= 50 and b >= 50 and g * 2 <= min(r, b) and r * 2 >= b and b * 2 >= r
    if color == "orange":
        return r >= 120 and g * 10 >= r * 4 and g * 4 <= r * 3 and b * 8 <= r
    if color == "pink":
        return r >= 150 and b * 2 >= r and g * 10 <= b * 7 and b <= r
    if color == "cyan":
        return g >= 80 and b >= 80 and r * 20 <= min(g, b) * 7 and g * 4 >= b * 3 and b * 4 >= g * 3
    return False


def color_blobs(rgb_rows, colors, stride):
    """Return {color: [list of (u, v) pixels]} for 8-connected same-colour components."""
    pix = {}
    for color in colors:
        pix[color] = set()
    h = len(rgb_rows)
    w = len(rgb_rows[0])
    for v in range(0, h, stride):
        row = rgb_rows[v]
        for u in range(0, w, stride):
            px = row[u]
            r = px[0]
            g = px[1]
            b = px[2]
            for color in colors:
                if is_color(color, r, g, b):
                    pix[color].add((u, v))
                    break
    out = {}
    for color in colors:
        remaining = pix[color]
        blobs = []
        while len(remaining) > 0 and len(blobs) < 40:
            seed = remaining.pop()
            blob = [seed]
            frontier = [seed]
            while len(frontier) > 0:
                cu, cv = frontier.pop()
                for du in (-stride, 0, stride):
                    for dv in (-stride, 0, stride):
                        nb = (cu + du, cv + dv)
                        if nb in remaining:
                            remaining.discard(nb)
                            blob.append(nb)
                            frontier.append(nb)
            blobs.append(blob)
        out[color] = blobs
    return out


def blob_point(blob, depth_rows, p, conv, stride):
    """Median 3D point of a pixel blob plus its physical area (m^2); None if depth unusable."""
    step = max(1, len(blob) // 25)
    xs = []
    ys = []
    zs = []
    ds = []
    for i in range(0, len(blob), step):
        u, v = blob[i]
        d = float(depth_rows[v][u])
        if valid_depth(d):
            q = unproject(p, u, v, d, conv)
            xs.append(q[0])
            ys.append(q[1])
            zs.append(q[2])
            ds.append(d)
    if len(xs) < 2:
        return None
    dmed = quantile(ds, 0.5)
    area = len(blob) * stride * stride * (dmed / p["fx"]) * (dmed / p["fy"])
    return ((quantile(xs, 0.5), quantile(ys, 0.5), quantile(zs, 0.5)), area)


def remember_cubes(video, colors, conv, table_z):
    """Cluster goal-coloured cube detections across the whole conditioning video in robot_base."""
    clusters = []
    if len(video) == 0 or len(colors) == 0:
        return clusters
    last_t = None
    used = 0
    for index in range(len(video)):
        frame = video[index]
        t = float(frame.timestamp_s)
        if last_t is not None and t - last_t < 0.14 and index != len(video) - 1:
            continue
        last_t = t
        used += 1
        for name, stride in (("front", 1), ("wrist", 2)):
            if name not in frame.cameras:
                continue
            cam = frame.cameras[name]
            p = cam_params(cam)
            rgb_rows = as_rows(cam.rgb)
            depth_rows = None
            blobs = color_blobs(rgb_rows, colors, stride)
            for color in colors:
                for blob in blobs[color]:
                    if len(blob) * stride * stride < 5:
                        continue
                    if depth_rows is None:
                        depth_rows = as_rows(cam.depth_m)
                    found = blob_point(blob, depth_rows, p, conv, stride)
                    if found is None:
                        continue
                    point, area = found
                    if area < 0.5e-4 or area > 6e-3:
                        continue
                    if point[2] < table_z - 0.02 or point[2] > table_z + 0.08:
                        continue
                    best = None
                    best_d = None
                    for cl in clusters:
                        if cl["color"] != color:
                            continue
                        dxy = math.sqrt((cl["x"] - point[0]) ** 2 + (cl["y"] - point[1]) ** 2)
                        if dxy <= 0.03 and (best_d is None or dxy < best_d):
                            best = cl
                            best_d = dxy
                    if best is None:
                        best = {"color": color, "sx": 0.0, "sy": 0.0, "sz": 0.0, "n": 0,
                                "frames": set(), "cams": set(), "first_t": t, "last_t": t}
                        clusters.append(best)
                    best["sx"] += point[0]
                    best["sy"] += point[1]
                    best["sz"] += point[2]
                    best["n"] += 1
                    best["x"] = best["sx"] / best["n"]
                    best["y"] = best["sy"] / best["n"]
                    best["z"] = best["sz"] / best["n"]
                    best["frames"].add(index)
                    best["cams"].add(name)
                    best["last_t"] = t
    report["video_frames_used"] = used
    return clusters


# ---------------------------------------------------------------- live container geometry
def min_area_yaw(points):
    """Yaw (rad, in [0, pi/2)) of the minimum-area XY rectangle, its extents and centre."""
    best = None
    for step_deg in range(0, 90, 3):
        a = math.radians(step_deg)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [pt[0] * ca + pt[1] * sa for pt in points]
        vs = [-pt[0] * sa + pt[1] * ca for pt in points]
        u_lo = quantile(us, 0.03)
        u_hi = quantile(us, 0.97)
        v_lo = quantile(vs, 0.03)
        v_hi = quantile(vs, 0.97)
        area = (u_hi - u_lo) * (v_hi - v_lo)
        if best is None or area < best[0]:
            uc = (u_lo + u_hi) / 2.0
            vc = (v_lo + v_hi) / 2.0
            best = (area, a, u_hi - u_lo, v_hi - v_lo, (uc * ca - vc * sa, uc * sa + vc * ca))
    return best


def container_at(target, camera_name, conv, table_z):
    """Top face of the container standing over the remembered cube, from live depth (zero ticks)."""
    obs = get_observation()
    if camera_name not in obs.cameras:
        return None
    cam = obs.cameras[camera_name]
    p = cam_params(cam)
    depth = as_rows(cam.depth_m)
    h = len(depth)
    w = len(depth[0])
    pr = project(p, (target["x"], target["y"], target["z"] + 0.03), conv)
    if pr is None:
        return None
    pu = int(round(pr[0]))
    pv = int(round(pr[1]))
    if pu < 0 or pv < 0 or pu >= w or pv >= h:
        return None
    seed = None
    for v in range(max(0, pv - 14), min(h, pv + 15)):
        for u in range(max(0, pu - 14), min(w, pu + 15)):
            d = float(depth[v][u])
            if not valid_depth(d):
                continue
            q = unproject(p, u, v, d, conv)
            if math.sqrt((q[0] - target["x"]) ** 2 + (q[1] - target["y"]) ** 2) > 0.03:
                continue
            # bounded at both ends: above the remembered cube top, below a plausible container top
            if q[2] < target["z"] + 0.005 or q[2] > table_z + 0.16:
                continue
            if seed is None or q[2] > seed[2][2]:
                seed = ((u, v), d, q)
    if seed is None:
        return None
    top_z0 = seed[2][2]
    face = [seed[2]]
    visited = set([seed[0]])
    frontier = [seed[0]]
    while len(frontier) > 0 and len(face) < 5000:
        cu, cv = frontier.pop()
        for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nu = cu + du
            nv = cv + dv
            if nu < 0 or nv < 0 or nu >= w or nv >= h or (nu, nv) in visited:
                continue
            visited.add((nu, nv))
            d = float(depth[nv][nu])
            if not valid_depth(d):
                continue
            q = unproject(p, nu, nv, d, conv)
            if abs(q[2] - top_z0) > 0.012:
                continue
            if math.sqrt((q[0] - seed[2][0]) ** 2 + (q[1] - seed[2][1]) ** 2) > 0.07:
                continue
            face.append(q)
            frontier.append((nu, nv))
    if len(face) < 8:
        return None
    rect = min_area_yaw(face)
    split = False
    if max(rect[2], rect[3]) > 0.072:
        # wider than the open gripper can hold: the face grew into a touching neighbour at the same
        # height. Keep the part around the remembered cube, sized by the face's own narrow extent.
        r_clip = max(0.018, min(0.04, 0.5 * min(rect[2], rect[3]) * 1.42))
        centre = (target["x"], target["y"])
        for iteration in range(3):
            sub = [q for q in face if math.sqrt((q[0] - centre[0]) ** 2 + (q[1] - centre[1]) ** 2) <= r_clip]
            if len(sub) < 8:
                break
            rect_sub = min_area_yaw(sub)
            centre = rect_sub[4]
            rect = rect_sub
            split = True
        if split:
            face = [q for q in face if math.sqrt((q[0] - centre[0]) ** 2 + (q[1] - centre[1]) ** 2) <= r_clip]
    top_z = quantile([q[2] for q in face], 0.5)
    # neighbourhood cloud (everything within 0.10 m XY of the face centre) for finger clearance
    radius_px = int(0.10 * p["fx"] / max(0.05, seed[1])) + 2
    near = []
    for v in range(max(0, seed[0][1] - radius_px), min(h, seed[0][1] + radius_px + 1)):
        for u in range(max(0, seed[0][0] - radius_px), min(w, seed[0][0] + radius_px + 1)):
            d = float(depth[v][u])
            if not valid_depth(d):
                continue
            q = unproject(p, u, v, d, conv)
            if math.sqrt((q[0] - rect[4][0]) ** 2 + (q[1] - rect[4][1]) ** 2) <= 0.10:
                near.append(q)
    return {"camera": camera_name, "x": rect[4][0], "y": rect[4][1], "top_z": top_z, "yaw": rect[1],
            "ext_a": rect[2], "ext_b": rect[3], "points": len(face), "near": near, "split": split,
            "offset_xy": math.sqrt((rect[4][0] - target["x"]) ** 2 + (rect[4][1] - target["y"]) ** 2)}


def finger_blockers(box, axis, grasp_z):
    """Count observed points inside the two finger landing zones for a closing axis (zero ticks).
    Open Panda fingers: inner faces ~0.04 m from the grasp centre, ~0.02 m thick, ~0.02 m wide."""
    ca = math.cos(axis)
    sa = math.sin(axis)
    count = 0
    for q in box["near"]:
        if q[2] < grasp_z - 0.005:
            continue
        dx = q[0] - box["x"]
        dy = q[1] - box["y"]
        along = abs(dx * ca + dy * sa)
        across = abs(-dx * sa + dy * ca)
        if 0.038 <= along <= 0.066 and across <= 0.014:
            count += 1
    return count


# ---------------------------------------------------------------- motion helpers
state0 = get_robot_state()
base_frame = state0.base_frame
stop = {"terminal": False}


def terminal_of(res):
    flag = False
    for name in ("terminated", "truncated"):
        try:
            if name == "terminated" and bool(res.terminated):
                flag = True
            if name == "truncated" and bool(res.truncated):
                flag = True
        except Exception:
            pass
    return flag


def run(label, res):
    ok = False
    try:
        ok = bool(res.ok)
    except Exception:
        ok = False
    status = "?"
    try:
        status = str(res.status)
    except Exception:
        pass
    steps = -1
    try:
        steps = int(res.steps_executed)
    except Exception:
        pass
    if terminal_of(res):
        stop["terminal"] = True
    note(label + " ok=" + str(ok) + " status=" + status + " steps=" + str(steps))
    return ok


def goto(label, xyz, quat, tolerance=0.01, max_steps=200):
    if stop["terminal"]:
        return False
    return run(label, move_to_pose(Pose((float(xyz[0]), float(xyz[1]), float(xyz[2])), quat, base_frame),
                                   tolerance=tolerance, max_steps=max_steps))


def gripper_now():
    return float(get_robot_state().gripper_positions["primary"])


# ---------------------------------------------------------------- main
goal = str(get_task_context().language)
report["goal"] = goal
groups = parse_goal(goal)
report["groups"] = groups
meta = {}
try:
    meta = robomme_plus.get_task_metadata()
    report["max_steps"] = meta["max_steps"]
except Exception as exc:
    note("metadata: " + str(exc))
hz = 20.0
try:
    hz = float(meta["control_hz"])
except Exception:
    try:
        hz = float(meta["control_frequency"])
    except Exception:
        pass
t_start = float(state0.timestamp_s)

obs0 = get_observation()
conv_scores = choose_convention(obs0)
conv = conv_scores[0][1]
report["convention"] = list(conv)
report["convention_scores"] = [[round(s[0], 4), list(s[1]), [round(m, 4) for m in s[2]]] for s in conv_scores]
table_z = conv_scores[0][2][0] if len(conv_scores[0][2]) > 0 else 0.0
report["table_z"] = table_z

colors = []
for grp in groups:
    if grp["color"] in COLOR_NAMES and grp["color"] not in colors:
        colors.append(grp["color"])
    if grp["color"] not in COLOR_NAMES:
        note("unsupported colour in goal: " + grp["color"])

video = robomme_plus.get_conditioning_video()
report["video_len"] = len(video)
clusters = remember_cubes(video, colors, conv, table_z)
cubes = [cl for cl in clusters if len(cl["frames"]) >= 2]
report["cubes"] = [{"color": cl["color"], "xyz": [round(cl["x"], 4), round(cl["y"], 4), round(cl["z"], 4)],
                    "frames": len(cl["frames"]), "cams": sorted(cl["cams"]),
                    "last_t": round(cl["last_t"], 2)} for cl in clusters]

# top-down orientation: keep the start orientation if it points down, else a canonical one
q_start = state0.end_effector_poses["primary"].quaternion_wxyz
q_start = (float(q_start[0]), float(q_start[1]), float(q_start[2]), float(q_start[3]))
if qmat(q_start)[2][2] < -0.9:
    q_down = q_start
else:
    q_down = (0.0, 1.0, 0.0, 0.0)
r_down = qmat(q_down)
close_yaw0 = math.atan2(r_down[1][1], r_down[0][1])

hover_z = table_z + 0.22
picked = []
outcomes = []
open_gripper()
for grp in groups:
    if stop["terminal"]:
        break
    cands = [cl for cl in cubes if cl["color"] == grp["color"]]
    cands.sort(key=lambda cl: -len(cl["frames"]))
    if not grp["all"]:
        cands = cands[:3]  # strongest first; later ones only as fallbacks
    done_in_group = 0
    remaining = list(cands)
    while len(remaining) > 0 and not stop["terminal"]:
        if not grp["all"] and done_in_group >= 1:
            break
        ee = get_robot_state().end_effector_poses["primary"].position
        if grp["all"]:
            remaining.sort(key=lambda cl: (cl["x"] - float(ee[0])) ** 2 + (cl["y"] - float(ee[1])) ** 2)
        target = remaining.pop(0)
        rec = {"color": grp["color"], "cube": [round(target["x"], 4), round(target["y"], 4)],
               "t_begin": round(float(get_robot_state().timestamp_s) - t_start, 2)}
        outcomes.append(rec)
        if not goto("hover", (target["x"], target["y"], hover_z), q_down, 0.015, 250):
            rec["result"] = "hover failed"
        box = container_at(target, "wrist", conv, table_z)
        if box is None:
            box = container_at(target, "front", conv, table_z)
        if box is None:
            rec["result"] = "no container found at remembered location"
            continue
        rec["box"] = {"cam": box["camera"], "xy": [round(box["x"], 4), round(box["y"], 4)],
                      "top_z": round(box["top_z"], 4), "yaw_deg": round(math.degrees(box["yaw"]), 1),
                      "ext": [round(box["ext_a"], 4), round(box["ext_b"], 4)], "pts": box["points"],
                      "split": box["split"],
                      "offset": round(box["offset_xy"], 4)}
        duplicate = False
        for prev in picked:
            if math.sqrt((prev[0] - box["x"]) ** 2 + (prev[1] - box["y"]) ** 2) < 0.025:
                duplicate = True
        if duplicate:
            rec["result"] = "duplicate of an already lifted container"
            continue
        height = box["top_z"] - table_z
        grasp_z = max(box["top_z"] - min(0.025, 0.5 * height), table_z + 0.01)
        # closing axis: either rectangle side whose extent fits the opening; prefer the side whose finger
        # landing zones are free of neighbouring containers, then the narrower side, then less wrist turn
        options = []
        for axis, width in ((box["yaw"], box["ext_a"]), (box["yaw"] + math.pi / 2.0, box["ext_b"])):
            dpsi_opt = wrap(axis - close_yaw0, math.pi)
            blockers = finger_blockers(box, axis, grasp_z)
            fits = 0 if width <= 0.072 else 1
            narrower = 0 if width <= min(box["ext_a"], box["ext_b"]) + 0.01 else 1
            options.append((fits, 0 if blockers <= 2 else 1, narrower, blockers, abs(dpsi_opt), dpsi_opt, width))
        options.sort()
        chosen = options[0]
        dpsi = chosen[5]
        width = chosen[6]
        q_grasp = qmul(qyaw(dpsi), q_down)
        rec["grasp"] = {"z": round(grasp_z, 4), "dpsi_deg": round(math.degrees(dpsi), 1),
                        "width": round(width, 4), "blockers": [o[3] for o in options]}
        cx = box["x"]
        cy = box["y"]
        pre_z = box["top_z"] + 0.06
        goto("pre", (cx, cy, pre_z), q_grasp, 0.01, 200)
        if not goto("descend", (cx, cy, grasp_z), q_grasp, 0.008, 150):
            note("descend not converged; continuing to close")
        if stop["terminal"]:
            break
        run("close", close_gripper())
        held = gripper_now()
        rec["gripper_after_close"] = round(held, 3)
        if held < 0.03:
            rec["result"] = "closed on nothing"
            run("reopen", open_gripper())
            goto("retreat", (cx, cy, pre_z), q_grasp, 0.015, 150)
            continue
        goto("lift", (cx, cy, grasp_z + 0.09), q_grasp, 0.012, 150)
        rec["gripper_after_lift"] = round(gripper_now(), 3)
        goto("lower", (cx, cy, grasp_z + 0.004), q_grasp, 0.01, 150)
        if not stop["terminal"]:
            run("release", open_gripper())
        goto("retreat", (cx, cy, pre_z), q_grasp, 0.015, 150)
        picked.append((cx, cy))
        done_in_group += 1
        rec["result"] = "lifted and replaced"
        rec["t_end"] = round(float(get_robot_state().timestamp_s) - t_start, 2)
    if done_in_group == 0:
        note("group " + grp["color"] + " had no liftable container")

report["outcomes"] = outcomes
report["ticks_used"] = round((float(get_robot_state().timestamp_s) - t_start) * hz)
report["terminal"] = stop["terminal"]
result = report
