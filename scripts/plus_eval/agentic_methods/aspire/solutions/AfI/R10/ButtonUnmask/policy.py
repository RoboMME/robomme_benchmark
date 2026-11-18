import math

# ButtonUnmask v03 -- two-camera, per-instance cube memory; button found in either camera.
# Changes vs v02 (episode 1 failure modes):
#   * button: candidates from BOTH cameras (front can be occluded by a tall distractor), score
#     floor 0.1, accepted only with button-like 3D geometry; missing button => press skipped =>
#     native termination at control step 89.
#   * memory: connected colour components per colour (several cubes can share a colour, and the goal
#     can say "all containers hiding green cubes"), from BOTH cameras every 2 ticks, each component
#     projected to robot_base with its own pixels' depth, then fused into 3D instances.
#   * goal parsing: ordered colour clauses; "all"/"<colour> cubes" selects every instance.

report = {"errors": []}
END = {"ended": False}
TRACK_EVERY = 2
SETTLE_TICKS = 25
MAX_WAIT_TICK = 320
MIN_BLOB = 6           # sampled pixels at stride 2 (~24 image pixels)
SAME_INSTANCE_M = 0.03
CAMS = ("front", "wrist")

SAT_COLORS = ("red", "green", "blue", "purple", "yellow", "orange", "pink", "cyan")
COLOR_WORDS = SAT_COLORS + ("white", "black", "gray", "grey", "brown")


def err(where, exc):
    if len(report["errors"]) < 40:
        report["errors"].append(where + ": " + str(exc)[:300])


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def fl(seq):
    return [float(v) for v in seq]


def r3(seq):
    return [round(float(v), 4) for v in seq]


def dist2(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def hue_sat(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 60 or mx == mn:
        return None, 0.0
    d = float(mx - mn)
    sat = d / float(mx)
    if mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * (((b - r) / d) + 2.0)
    else:
        h = 60.0 * (((r - g) / d) + 4.0)
    return h, sat


def cube_color(r, g, b):
    """Saturated cube colours; the wooden table (hue ~18-26, sat <= 0.75) maps to None."""
    h, sat = hue_sat(r, g, b)
    if h is None or sat < 0.6:
        return None
    if h < 10 or h >= 345:
        return "red"
    if 30 <= h < 45:
        return "orange" if sat > 0.78 else None
    if 45 <= h < 72:
        return "yellow"
    if 85 <= h < 160:
        return "green"
    if 170 <= h < 200:
        return "cyan"
    if 205 <= h < 255:
        return "blue"
    if 262 <= h < 305:
        return "purple"
    if 305 <= h < 345:
        return "pink"
    return None


def color_components(rgb, stride=2):
    """Connected components (8-neighbour on the stride grid) per saturated colour."""
    try:
        rows = rgb.tolist()
    except Exception:
        rows = [[list(px) for px in row] for row in rgb]
    labels = {}
    for y in range(0, len(rows), stride):
        row = rows[y]
        for x in range(0, len(row), stride):
            px = row[x]
            c = cube_color(int(px[0]), int(px[1]), int(px[2]))
            if c is not None:
                labels[(x, y)] = c
    seen = set()
    comps = []
    for key in labels:
        if key in seen:
            continue
        c = labels[key]
        stack = [key]
        seen.add(key)
        pix = []
        while stack:
            p = stack.pop()
            pix.append(p)
            for dx in (-stride, 0, stride):
                for dy in (-stride, 0, stride):
                    q = (p[0] + dx, p[1] + dy)
                    if q in labels and q not in seen and labels[q] == c:
                        seen.add(q)
                        stack.append(q)
        if len(pix) >= MIN_BLOB:
            comps.append({"color": c, "pix": pix})
    return comps


def quat_matrix(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def pixels_to_base(cam, pix):
    """Median robot_base point of the given pixels (public depth, intrinsics, camera_pose; OpenCV axes)."""
    depth = cam.depth_m
    k = cam.intrinsics
    fx = float(k[0][0])
    fy = float(k[1][1])
    cx = float(k[0][2])
    cy = float(k[1][2])
    rm = quat_matrix(fl(cam.camera_pose.quaternion_wxyz))
    pos = fl(cam.camera_pose.position)
    step_n = max(1, len(pix) // 30)
    xs = []
    ys = []
    zs = []
    for i in range(0, len(pix), step_n):
        u = pix[i][0]
        v = pix[i][1]
        d = float(depth[v][u])
        if d < 0.05 or d > 3.0:
            continue
        pc = [(u - cx) * d / fx, (v - cy) * d / fy, d]
        xs.append(pos[0] + rm[0][0] * pc[0] + rm[0][1] * pc[1] + rm[0][2] * pc[2])
        ys.append(pos[1] + rm[1][0] * pc[0] + rm[1][1] * pc[1] + rm[1][2] * pc[2])
        zs.append(pos[2] + rm[2][0] * pc[0] + rm[2][1] * pc[1] + rm[2][2] * pc[2])
    if len(xs) < 3:
        return None
    return [quantile(xs, 0.5), quantile(ys, 0.5), quantile(zs, 0.5)]


# ------------------------------------------------------------------ memory of cube sightings
TRACK = {"tick": 0, "last_any": None, "log": []}
INST = []   # {"color", "first", "last", "hits", "pts": [[x,y,z],...], "cams": {cam: hits}}


def inst_xyz(inst):
    pts = inst["pts"]
    return [quantile([p[0] for p in pts], 0.5), quantile([p[1] for p in pts], 0.5),
            quantile([p[2] for p in pts], 0.5)]


def track(force=False):
    tick = TRACK["tick"]
    if not force and tick % TRACK_EVERY != 0:
        return
    try:
        ob = get_observation()
    except Exception as exc:
        err("track obs", exc)
        return
    logged = {}
    for camname in CAMS:
        try:
            cam = ob.cameras[camname]
            comps = color_components(cam.rgb, 2)
        except Exception as exc:
            err("track " + camname, exc)
            continue
        for comp in comps:
            TRACK["last_any"] = tick
            try:
                xyz = pixels_to_base(cam, comp["pix"])
            except Exception as exc:
                err("project", exc)
                xyz = None
            if xyz is None:
                continue
            if xyz[2] < -0.01 or xyz[2] > 0.07:
                continue
            best = None
            for inst in INST:
                if inst["color"] != comp["color"]:
                    continue
                d = dist2(inst_xyz(inst), xyz)
                if d < SAME_INSTANCE_M and (best is None or d < best[0]):
                    best = [d, inst]
            if best is None:
                inst = {"color": comp["color"], "first": tick, "last": tick, "hits": 0, "pts": [],
                        "cams": {}}
                INST.append(inst)
            else:
                inst = best[1]
            inst["last"] = tick
            inst["hits"] = inst["hits"] + 1
            inst["cams"][camname] = inst["cams"].get(camname, 0) + 1
            inst["pts"].append(xyz)
            if len(inst["pts"]) > 40:
                inst["pts"] = inst["pts"][-40:]
            logged[camname + ":" + comp["color"]] = logged.get(camname + ":" + comp["color"], 0) + 1
    if tick % 10 == 0 or force:
        TRACK["log"].append({"tick": tick, "vis": logged})


def is_ended(res):
    flag = False
    try:
        flag = bool(res.terminated) or bool(res.truncated)
    except Exception:
        pass
    if flag:
        END["ended"] = True
    return flag


def tick_to(target, grip):
    st = get_robot_state()
    res = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=list(target), gripper_position=grip,
        embodiment=st.embodiment)}))
    TRACK["tick"] = TRACK["tick"] + 1
    if is_ended(res):
        return False
    track()
    return True


def joint_path(q_from, q_to, max_delta):
    biggest = max(abs(q_to[i] - q_from[i]) for i in range(7))
    n = max(1, int(math.ceil(biggest / max_delta)))
    path = []
    for k in range(1, n + 1):
        t = k / float(n)
        s = 0.5 - 0.5 * math.cos(math.pi * t)
        path.append([q_from[i] + s * (q_to[i] - q_from[i]) for i in range(7)])
    return path


def run_path(path, grip):
    for q in path:
        if not tick_to(q, grip):
            return False
    return True


def current_q():
    return fl(get_robot_state().joint_positions["primary"])


def yaw_quat(yaw):
    # Rz(yaw) * q_down where q_down = (0, -1, 0, 0) (wxyz) is the start pose's top-down hand.
    return (0.0, -math.cos(yaw / 2.0), -math.sin(yaw / 2.0), 0.0)


# ------------------------------------------------------------------ Phase 0: read the episode
ctx = get_task_context()
goal = str(ctx.language)
report["goal"] = goal
try:
    meta = dict(robomme_plus.get_task_metadata())
    report["max_steps"] = meta.get("max_steps")
except Exception as exc:
    err("metadata", exc)
try:
    report["video_len"] = len(robomme_plus.get_conditioning_video())
except Exception as exc:
    err("video", exc)


def parse_goal(text):
    """Ordered [colour, take_all] requests from the goal sentence."""
    low = text.lower()
    hits = []
    for c in COLOR_WORDS:
        start = 0
        while True:
            pos = low.find(c, start)
            if pos < 0:
                break
            before = low[pos - 1] if pos > 0 else " "
            after = low[pos + len(c)] if pos + len(c) < len(low) else " "
            if not before.isalpha() and not after.isalpha():
                hits.append([pos, c])
            start = pos + len(c)
    hits = sorted(hits)
    out = []
    prev_end = 0
    for h in hits:
        pos = h[0]
        c = h[1]
        clause = low[prev_end:pos]
        tail = low[pos + len(c):pos + len(c) + 7]
        take_all = (" all " in (" " + clause + " ")) or tail.startswith(" cubes")
        out.append([c, take_all])
        prev_end = pos + len(c)
    return out


REQUESTS = parse_goal(goal)
report["requests"] = REQUESTS

state0 = get_robot_state()
q_home = fl(state0.joint_positions["primary"])
down_quat = fl(state0.end_effector_poses["primary"].quaternion_wxyz)
base = state0.base_frame
report["start"] = {"ee": r3(state0.end_effector_poses["primary"].position), "quat": r3(down_quat)}


# ------------------------------------------------------------------ Phase A: press the button
def button_candidates(camera):
    out = []
    found = segment_text(camera, "button")
    if not found.ok:
        return out
    for cand in found.segmentations[:4]:
        if float(cand.score) < 0.1:
            continue
        cloud = mask_to_point_cloud(cand, camera, target_frame=base)
        pts = cloud.points
        if len(pts) < 10:
            continue
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        zs = [float(p[2]) for p in pts]
        top = quantile(zs, 0.95)
        span = max(quantile(xs, 0.95) - quantile(xs, 0.05), quantile(ys, 0.95) - quantile(ys, 0.05))
        entry = {"cam": camera, "score": round(float(cand.score), 3), "top": round(top, 4),
                 "span": round(span, 4), "n": len(pts)}
        if span > 0.08 or top < 0.012 or top > 0.07:
            entry["rejected"] = True
            out.append(entry)
            continue
        try:
            box = cand.box_xyxy
            if box is not None:
                bu = int(round((float(box[0]) + float(box[2])) / 2.0))
                bv = int(round((float(box[1]) + float(box[3])) / 2.0))
                cam = get_observation().cameras[camera]
                mine = pixels_to_base(cam, [(bu + du, bv + dv) for du in (-1, 0, 1) for dv in (-1, 0, 1)])
                if mine is not None:
                    entry["proj_err"] = round(dist2(mine, (quantile(xs, 0.5), quantile(ys, 0.5))), 4)
        except Exception as exc:
            err("proj check", exc)
        topx = [xs[i] for i in range(len(zs)) if zs[i] >= top - 0.012]
        topy = [ys[i] for i in range(len(zs)) if zs[i] >= top - 0.012]
        entry["x"] = quantile(topx, 0.5)
        entry["y"] = quantile(topy, 0.5)
        out.append(entry)
    return out


press = {}
button = None
bcands = []
for camname in CAMS:
    try:
        bcands = bcands + button_candidates(camname)
    except Exception as exc:
        err("button " + camname, exc)
for c in bcands:
    if c.get("rejected"):
        continue
    if button is None or c["score"] > button["score"]:
        button = c
press["candidates"] = bcands
press["button"] = button
# My pixel projection is trusted per camera only if it agrees with the harness point cloud
# of the same button mask (front was checked in episode 0; wrist is checked here, every episode).
wrist_ok = False
for c in bcands:
    if c["cam"] == "wrist" and c.get("proj_err") is not None and c["proj_err"] < 0.03:
        wrist_ok = True
CAMS = ("front", "wrist") if wrist_ok else ("front",)
report["tracking_cams"] = list(CAMS)
track(force=True)

if button is not None:
    try:
        pre_pose = Pose((button["x"], button["y"], button["top"] + 0.07), tuple(down_quat), base)
        press_pose = Pose((button["x"], button["y"], button["top"] - 0.025), tuple(down_quat), base)
        ik_pre = solve_ik(pre_pose, seed_joints=q_home)
        ik_press = solve_ik(press_pose, seed_joints=fl(ik_pre.joint_positions)) if ik_pre.ok else None
        press["ik"] = [bool(ik_pre.ok), bool(ik_press.ok) if ik_press is not None else False]
        if ik_press is not None and ik_press.ok:
            q_pre = fl(ik_pre.joint_positions)
            q_press = fl(ik_press.joint_positions)
            ok = run_path(joint_path(q_home, q_pre, 0.07), 0.0)
            if ok:
                ok = run_path(joint_path(current_q(), q_press, 0.03), 0.0)
            if ok:
                ok = run_path([q_press for k in range(5)], 0.0)
                st = get_robot_state()
                press["ee_at_press"] = r3(st.end_effector_poses["primary"].position)
                press["tick_at_press"] = TRACK["tick"]
            if ok:
                ok = run_path(joint_path(current_q(), q_pre, 0.05), 0.0)
            if ok:
                ok = run_path(joint_path(current_q(), q_home, 0.07), 0.0)
            press["tick_home"] = TRACK["tick"]
    except Exception as exc:
        err("press", exc)
report["press"] = press

# ------------------------------------------------------------------ Phase B: wait for covering
wait = {"start": TRACK["tick"]}
try:
    while not END["ended"] and TRACK["tick"] < MAX_WAIT_TICK:
        wanted = [r[0] for r in REQUESTS if r[0] in SAT_COLORS]
        seen_all = all(any(i["color"] == c and i["hits"] >= 2 for i in INST) for c in wanted)
        last_any = TRACK["last_any"]
        quiet = last_any is not None and TRACK["tick"] - last_any >= SETTLE_TICKS
        if seen_all and quiet:
            break
        if not tick_to(q_home, 1.0):
            break
except Exception as exc:
    err("wait", exc)
wait["end"] = TRACK["tick"]
report["wait"] = wait
MEMORY = [i for i in INST if i["hits"] >= 2]
report["instances"] = [{"color": i["color"], "xyz": r3(inst_xyz(i)), "first": i["first"], "last": i["last"],
                        "hits": i["hits"], "cams": i["cams"]} for i in INST]
report["track_log"] = TRACK["log"]


# ------------------------------------------------------------------ Phase C: find containers
def rect_yaw(xs, ys):
    """Yaw (rad, in [0, pi/2)) of the min-area bounding rectangle of the top-face points."""
    best = None
    for k in range(30):
        a = k * (math.pi / 2.0) / 30.0
        ca = math.cos(a)
        sa = math.sin(a)
        us = [xs[i] * ca + ys[i] * sa for i in range(len(xs))]
        vs = [-xs[i] * sa + ys[i] * ca for i in range(len(xs))]
        du = quantile(us, 0.99) - quantile(us, 0.01)
        dv = quantile(vs, 0.99) - quantile(vs, 0.01)
        area = du * dv
        if best is None or area < best[0]:
            best = [area, a, du, dv]
    return best[1], best[2], best[3]


def find_containers(camera):
    out = []
    for prompt in ("white box", "cube", "container"):
        try:
            found = segment_text(camera, prompt)
        except Exception as exc:
            err("seg " + prompt, exc)
            continue
        if not found.ok:
            continue
        for cand in found.segmentations[:16]:
            if float(cand.score) < 0.15 or cand.box_xyxy is None:
                continue
            box = [float(v) for v in cand.box_xyxy]
            if (box[2] - box[0]) > 70 or (box[3] - box[1]) > 70:
                continue
            dup = False
            for o in out:
                ob = o["box"]
                ix = max(0.0, min(box[2], ob[2]) - max(box[0], ob[0]))
                iy = max(0.0, min(box[3], ob[3]) - max(box[1], ob[1]))
                inter = ix * iy
                area = (box[2] - box[0]) * (box[3] - box[1])
                if inter > 0.5 * area:
                    dup = True
                    break
            if dup:
                continue
            try:
                cloud = mask_to_point_cloud(cand, camera, target_frame=base)
                pts = cloud.points
                if len(pts) < 20:
                    continue
                zs = [float(p[2]) for p in pts]
                top = quantile(zs, 0.95)
                txs = [float(pts[i][0]) for i in range(len(pts)) if zs[i] >= top - 0.008]
                tys = [float(pts[i][1]) for i in range(len(pts)) if zs[i] >= top - 0.008]
                if len(txs) < 10:
                    continue
                yaw, du, dv = rect_yaw(txs, tys)
                cx = (quantile(txs, 0.03) + quantile(txs, 0.97)) / 2.0
                cy = (quantile(tys, 0.03) + quantile(tys, 0.97)) / 2.0
                out.append({"box": box, "score": round(float(cand.score), 3), "prompt": prompt,
                            "x": cx, "y": cy, "top": top, "yaw": yaw, "du": du, "dv": dv, "n": len(pts)})
            except Exception as exc:
                err("container cloud", exc)
    return out


def depth_containers(camname):
    """Containers from the wrist depth alone (SAM returns at most five masks per prompt, and there
    can be eight containers). Points above the button/cube height are bucketed on a 0.8 cm XY grid,
    4-connected cells form one object; tall objects (distractor pillars) are rejected."""
    cam = get_observation().cameras[camname]
    try:
        depth = cam.depth_m.tolist()
    except Exception:
        depth = cam.depth_m
    k = cam.intrinsics
    fx = float(k[0][0])
    fy = float(k[1][1])
    cx0 = float(k[0][2])
    cy0 = float(k[1][2])
    rm = quat_matrix(fl(cam.camera_pose.quaternion_wxyz))
    pos = fl(cam.camera_pose.position)
    cell = 0.008
    cells = {}
    for v in range(0, len(depth)):
        row = depth[v]
        for u in range(0, len(row)):
            d = float(row[u])
            if d < 0.05 or d > 3.0:
                continue
            a = (u - cx0) * d / fx
            b = (v - cy0) * d / fy
            z = pos[2] + rm[2][0] * a + rm[2][1] * b + rm[2][2] * d
            if z < CONTAINER_MIN_Z or z > 0.35:
                continue
            x = pos[0] + rm[0][0] * a + rm[0][1] * b + rm[0][2] * d
            y = pos[1] + rm[1][0] * a + rm[1][1] * b + rm[1][2] * d
            key = (int(math.floor(x / cell)), int(math.floor(y / cell)))
            lst = cells.get(key)
            if lst is None:
                lst = []
                cells[key] = lst
            lst.append((x, y, z))
    seen = set()
    out = []
    rejected = []
    for key in cells:
        if key in seen:
            continue
        seen.add(key)
        stack = [key]
        pts = []
        while stack:
            kk = stack.pop()
            pts = pts + cells[kk]
            for nb in ((kk[0] + 1, kk[1]), (kk[0] - 1, kk[1]), (kk[0], kk[1] + 1), (kk[0], kk[1] - 1)):
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if len(pts) < 40:
            continue
        zs = [p[2] for p in pts]
        zmax = quantile(zs, 0.98)
        top = quantile(zs, 0.9)
        txs = [p[0] for p in pts if p[2] >= top - 0.008]
        tys = [p[1] for p in pts if p[2] >= top - 0.008]
        if len(txs) < 25:
            continue
        yaw, du, dv = rect_yaw(txs, tys)
        c = {"x": (quantile(txs, 0.03) + quantile(txs, 0.97)) / 2.0,
             "y": (quantile(tys, 0.03) + quantile(tys, 0.97)) / 2.0,
             "top": top, "yaw": yaw, "du": du, "dv": dv, "n": len(pts), "score": 1.0,
             "prompt": "depth", "box": [0.0, 0.0, 0.0, 0.0], "zmax": zmax}
        if zmax > 0.11 or not (0.025 < max(du, dv) < 0.09):
            rejected.append({"x": round(c["x"], 3), "y": round(c["y"], 3), "zmax": round(zmax, 3),
                             "du": round(du, 3), "dv": round(dv, 3), "n": len(pts)})
            continue
        out.append(c)
    report["depth_rejected"] = rejected[:12]
    return out


CONTAINER_MIN_Z = 0.045
containers = []
if not END["ended"]:
    raw = []
    try:
        raw = depth_containers("wrist")
    except Exception as exc:
        err("depth containers", exc)
    report["n_depth_containers"] = len(raw)
    for c in raw:
        c["cam"] = "wrist-depth"
    for c in find_containers("wrist"):
        c["cam"] = "wrist"
        near = False
        for o in raw:
            if math.sqrt((c["x"] - o["x"]) ** 2 + (c["y"] - o["y"]) ** 2) < 0.03:
                near = True
                break
        if not near:
            raw.append(c)
    for c in find_containers("front"):
        c["cam"] = "front"
        near = False
        for o in raw:
            if math.sqrt((c["x"] - o["x"]) ** 2 + (c["y"] - o["y"]) ** 2) < 0.03:
                near = True
                break
        if not near:
            raw.append(c)
    containers = raw
    if button is not None:
        containers = [c for c in containers
                      if math.sqrt((c["x"] - button["x"]) ** 2 + (c["y"] - button["y"]) ** 2) > 0.035]
    containers = [c for c in containers
                  if 0.025 < max(c["du"], c["dv"]) < 0.11 and 0.03 < c["top"] < 0.12]
report["containers"] = [{"x": round(c["x"], 4), "y": round(c["y"], 4), "top": round(c["top"], 4),
                         "yaw_deg": round(math.degrees(c["yaw"]), 1), "du": round(c["du"], 4),
                         "dv": round(c["dv"], 4), "score": c["score"], "prompt": c["prompt"], "cam": c["cam"],
                         "box": [round(v, 1) for v in c["box"]]} for c in containers]


# ------------------------------------------------------------------ Phase D: lift in goal order
def choose_yaw(c, others):
    best = None
    for base_yaw in (c["yaw"], c["yaw"] + math.pi / 2.0):
        yaw = base_yaw
        while yaw > math.pi / 2.0:
            yaw -= math.pi
        while yaw <= -math.pi / 2.0:
            yaw += math.pi
        ax = -math.sin(yaw)
        ay = math.cos(yaw)
        clearance = 1.0
        for o in others:
            for sgn in (1.0, -1.0):
                fx = c["x"] + sgn * 0.055 * ax
                fy = c["y"] + sgn * 0.055 * ay
                d = math.sqrt((fx - o["x"]) ** 2 + (fy - o["y"]) ** 2)
                if d < clearance:
                    clearance = d
        score = clearance - 0.01 * abs(yaw)
        if best is None or score > best[0]:
            best = [score, yaw, clearance]
    return best[1], best[2]


def move(pose_xyz, yaw, tol, steps):
    res = move_to_pose(Pose(tuple(pose_xyz), yaw_quat(yaw), base), tolerance=tol, max_steps=steps)
    is_ended(res)
    return res


# Build the ordered target list: one instance per singular request, every instance for "all".
targets = []
taken = []
for req in REQUESTS:
    color = req[0]
    cands = [i for i in MEMORY if i["color"] == color and not (i in taken)]
    if not cands:
        targets.append({"color": color, "xyz": None})
        continue
    if req[1]:
        for i in cands:
            taken.append(i)
            targets.append({"color": color, "xyz": inst_xyz(i)})
    else:
        best = None
        for i in cands:
            if best is None or i["hits"] > best["hits"]:
                best = i
        taken.append(best)
        targets.append({"color": color, "xyz": inst_xyz(best)})
report["targets"] = [{"color": t["color"], "xyz": r3(t["xyz"]) if t["xyz"] else None} for t in targets]

picks = []
used = []
if not END["ended"] and containers:
    try:
        r0 = open_gripper()
        is_ended(r0)
    except Exception as exc:
        err("open", exc)
    for gi in range(len(targets)):
        if END["ended"]:
            break
        tgt = targets[gi]
        last = gi == len(targets) - 1
        entry = {"color": tgt["color"]}
        picks.append(entry)
        if tgt["xyz"] is None:
            entry["skip"] = "no remembered position"
            continue
        best = None
        for ci in range(len(containers)):
            if ci in used:
                continue
            d = dist2((containers[ci]["x"], containers[ci]["y"]), tgt["xyz"])
            if best is None or d < best[0]:
                best = [d, ci]
        if best is None:
            entry["skip"] = "no container left"
            continue
        used.append(best[1])
        c = containers[best[1]]
        others = [containers[k] for k in range(len(containers)) if k != best[1]]
        yaw, clearance = choose_yaw(c, others)
        entry["container"] = best[1]
        entry["match_dist"] = round(best[0], 4)
        entry["yaw_deg"] = round(math.degrees(yaw), 1)
        entry["clearance"] = round(clearance, 4)
        grasp_z = max(c["top"] - 0.025, 0.015)
        try:
            res = move((c["x"], c["y"], c["top"] + 0.09), yaw, 0.015, 150)
            entry["above"] = [bool(res.ok), int(res.steps_executed)]
            if END["ended"]:
                break
            res = move((c["x"], c["y"], grasp_z), yaw, 0.01, 80)
            entry["descend"] = [bool(res.ok), int(res.steps_executed)]
            if END["ended"]:
                break
            res = close_gripper()
            is_ended(res)
            entry["close"] = [bool(res.ok), int(res.steps_executed)]
            entry["grip_closed"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
            if END["ended"]:
                break
            res = move((c["x"], c["y"], c["top"] + 0.10), yaw, 0.015, 100)
            entry["lift"] = [bool(res.ok), int(res.steps_executed)]
            if END["ended"]:
                break
            st = get_robot_state()
            entry["grip_lifted"] = round(float(st.gripper_positions["primary"]), 3)
            entry["ee_lifted"] = r3(st.end_effector_poses["primary"].position)
            q = current_q()
            run_path([q for k in range(6)], 0.0)
            if END["ended"] or last:
                break
            res = move((c["x"], c["y"], grasp_z + 0.005), yaw, 0.015, 100)
            entry["lower"] = [bool(res.ok), int(res.steps_executed)]
            if END["ended"]:
                break
            res = open_gripper()
            is_ended(res)
            if END["ended"]:
                break
            res = move((c["x"], c["y"], c["top"] + 0.09), yaw, 0.02, 80)
            entry["retreat"] = [bool(res.ok), int(res.steps_executed)]
        except Exception as exc:
            err("pick " + tgt["color"], exc)
report["picks"] = picks
report["ended"] = END["ended"]
report["ticks"] = TRACK["tick"]
try:
    report["final_t"] = float(get_robot_state().timestamp_s)
except Exception:
    pass

result = report
