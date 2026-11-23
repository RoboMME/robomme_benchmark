# ButtonUnmask v00 — initial program from offline recording analysis.
# Observe cube batches live (front + wrist), press the button between batches, remember the
# 3D position of every goal-colour cube, then lift the containers in goal order.
import math

report = {"version": "v00", "events": [], "errors": []}
ctx = get_task_context()
goal = str(ctx.language)
report["goal"] = goal
try:
    meta = robomme_plus.get_task_metadata()
    report["max_steps"] = meta["max_steps"]
except Exception as exc:
    meta = {}
    report["errors"].append("meta: " + str(exc))
try:
    cmeta = robomme_plus.get_controller_metadata()
    report["controller_meta_keys"] = sorted([str(k) for k in cmeta])
except Exception as exc:
    report["errors"].append("cmeta: " + str(exc))

state0 = get_robot_state()
base_frame = state0.base_frame
emb = state0.embodiment
home_joints = [float(v) for v in state0.joint_positions["primary"]]
q_down = [float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz]
home_pos = [float(v) for v in state0.end_effector_poses["primary"].position]
report["home_pos"] = home_pos
report["q_down"] = q_down

S = {"tick": 0, "ended": False, "cmd_mode": None, "status": "running", "phase": "home"}

COLORS = ["red", "green", "blue", "yellow", "orange", "purple"]


# ---------------------------------------------------------------- math helpers
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def qrot(q, v):
    w, x, y, z = q
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    return [r00 * v[0] + r01 * v[1] + r02 * v[2],
            r10 * v[0] + r11 * v[1] + r12 * v[2],
            r20 * v[0] + r21 * v[1] + r22 * v[2]]


def qz(angle):
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]


def quant(values, frac):
    vals = sorted(values)
    if not vals:
        return None
    return vals[int((len(vals) - 1) * frac)]


def to_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        return [[[int(c) for c in px] for px in row] for row in arr]


# ---------------------------------------------------------------- colour classifier
def classify(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    c = mx - mn
    if mx < 90 or c < 64:
        return None
    s = c / float(mx)
    if mx == r:
        h = ((g - b) / float(c)) % 6.0
    elif mx == g:
        h = (b - r) / float(c) + 2.0
    else:
        h = (r - g) / float(c) + 4.0
    h = h * 60.0
    if (h >= 340.0 or h < 12.0) and s > 0.8:
        return "red"
    if 22.0 <= h < 46.0 and s > 0.86:
        return "orange"
    if 46.0 <= h < 75.0 and s > 0.75:
        return "yellow"
    if 85.0 <= h < 160.0 and s > 0.6:
        return "green"
    if 215.0 <= h < 260.0 and s > 0.8 and mx > 110:
        return "blue"
    if 262.0 <= h < 320.0 and s > 0.45 and mx > 120:
        return "purple"
    return None


def quick_count(obs):
    total = 0
    for cam in ("front", "wrist"):
        try:
            sub = obs.cameras[cam].rgb[::4, ::4].tolist()
        except Exception:
            rows = to_rows(obs.cameras[cam].rgb)
            sub = [row[::4] for row in rows[::4]]
        for row in sub:
            for px in row:
                if classify(px[0], px[1], px[2]) is not None:
                    total += 1
    return total


def find_blobs(obs, cam, min_px):
    rows = to_rows(obs.cameras[cam].rgb)
    labels = {}
    for v in range(len(rows)):
        row = rows[v]
        for u in range(len(row)):
            px = row[u]
            col = classify(px[0], px[1], px[2])
            if col is not None:
                labels[(u, v)] = col
    seen = set()
    blobs = []
    for key in labels:
        if key in seen:
            continue
        col = labels[key]
        stack = [key]
        seen.add(key)
        pts = []
        while stack:
            cur = stack.pop()
            pts.append(cur)
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    nb = (cur[0] + du, cur[1] + dv)
                    if nb in labels and nb not in seen and labels[nb] == col:
                        seen.add(nb)
                        stack.append(nb)
        if len(pts) >= min_px:
            mu = sum(p[0] for p in pts) / float(len(pts))
            mv = sum(p[1] for p in pts) / float(len(pts))
            best = pts[0]
            bestd = None
            for p in pts:
                dd = (p[0] - mu) ** 2 + (p[1] - mv) ** 2
                if bestd is None or dd < bestd:
                    bestd = dd
                    best = p
            blobs.append({"color": col, "cam": cam, "n": len(pts), "u": int(best[0]), "v": int(best[1]),
                          "umin": min(p[0] for p in pts), "umax": max(p[0] for p in pts),
                          "vmin": min(p[1] for p in pts), "vmax": max(p[1] for p in pts)})
    return blobs


def deproject(obs, cam, u, v):
    camo = obs.cameras[cam]
    try:
        d = float(camo.depth_m[v][u])
    except Exception:
        return None
    K = camo.intrinsics
    fx = float(K[0][0])
    fy = float(K[1][1])
    cx = float(K[0][2])
    cy = float(K[1][2])
    x = (u - cx) / fx
    y = (v - cy) / fy
    pose = camo.camera_pose
    q = [float(a) for a in pose.quaternion_wxyz]
    p = [float(a) for a in pose.position]
    out = {}
    for name, pc in (("cv", [x * d, y * d, d]), ("gl", [x * d, -y * d, -d])):
        w = qrot(q, pc)
        out[name] = [w[0] + p[0], w[1] + p[1], w[2] + p[2]]
    out["cam_z"] = p[2]
    out["depth"] = d
    return out


def cloud_points(seg_candidate, cam):
    pc = mask_to_point_cloud(seg_candidate, cam, target_frame=base_frame)
    pts = [[float(a) for a in p] for p in pc.points]
    return pts


def localize_blob(cam, blob):
    info = {"sam": None, "sam_n": 0}
    try:
        seg = segment_points(cam, [[blob["u"], blob["v"]]])
        if seg.ok and len(seg.segmentations) > 0:
            pts = cloud_points(seg.segmentations[0], cam)
            info["sam_n"] = len(pts)
            if 3 <= len(pts) <= max(400, 12 * blob["n"]):
                info["sam"] = [quant([p[0] for p in pts], 0.5), quant([p[1] for p in pts], 0.5),
                               quant([p[2] for p in pts], 0.5)]
    except Exception as exc:
        info["sam_err"] = str(exc)[:200]
    return info


# ---------------------------------------------------------------- goal parsing
def parse_goal(text):
    low = text.lower()
    start = low.find("pick")
    if start < 0:
        start = 0
    hits = []
    for col in COLORS:
        idx = low.find(col, start)
        while idx >= 0:
            hits.append((idx, col))
            idx = low.find(col, idx + 1)
    hits = sorted(hits)
    steps = []
    prev = start
    for idx, col in hits:
        seg = low[prev:idx]
        steps.append({"color": col, "all": ("all " in seg) or ("every" in seg)})
        prev = idx + len(col)
    return steps


plan_steps = parse_goal(goal)
report["plan_steps"] = plan_steps
target_colors = set(s["color"] for s in plan_steps)


# ---------------------------------------------------------------- low-level control
def is_ended(res):
    flag = False
    try:
        flag = flag or bool(res.terminated)
    except Exception:
        pass
    try:
        flag = flag or bool(res.truncated)
    except Exception:
        pass
    return flag


def make_action(joints, grip):
    if S["cmd_mode"] != "plain":
        try:
            act = RobotAction({"primary": ArmCommand("joint_position", list(joints), gripper_position=grip,
                                                      embodiment=emb)})
            S["cmd_mode"] = "emb"
            return act
        except Exception:
            S["cmd_mode"] = "plain"
    return RobotAction({"primary": ArmCommand("joint_position", list(joints), gripper_position=grip)})


def tick(joints, grip):
    res = step(make_action(joints, grip))
    S["tick"] += 1
    if is_ended(res):
        S["ended"] = True
    try:
        return res.observation
    except Exception:
        return get_observation()


def add_steps(res):
    try:
        S["tick"] += int(res.steps_executed)
    except Exception:
        pass
    if is_ended(res):
        S["ended"] = True


# ---------------------------------------------------------------- observation memory
detections = []
batches = []
watch = {"visible": False, "baseline": None, "last_cover": None, "last_appear": None, "counts": []}


def analyse_batch(obs):
    found = []
    for cam in ("front", "wrist"):
        try:
            blobs = find_blobs(obs, cam, 8)
        except Exception as exc:
            report["errors"].append("blobs " + cam + ": " + str(exc)[:200])
            blobs = []
        for blob in blobs:
            blob["tick"] = S["tick"]
            dp = deproject(obs, cam, blob["u"], blob["v"])
            blob["deproj"] = dp
            if blob["color"] in target_colors:
                info = localize_blob(cam, blob)
                blob["sam"] = info["sam"]
                blob["sam_n"] = info["sam_n"]
                if "sam_err" in info:
                    blob["sam_err"] = info["sam_err"]
            found.append(blob)
    batches.append({"tick": S["tick"], "n": len(found), "at_home": S["phase"] == "home",
                    "summary": [[b["cam"], b["color"], b["n"], b["u"], b["v"]] for b in found]})
    for b in found:
        detections.append(b)


def poll(obs):
    try:
        cnt = quick_count(obs)
    except Exception as exc:
        report["errors"].append("poll: " + str(exc)[:200])
        return
    if watch["baseline"] is None:
        watch["baseline"] = cnt
    watch["counts"].append([S["tick"], cnt])
    thr = watch["baseline"] + 4
    if not watch["visible"] and cnt >= thr:
        watch["visible"] = True
        watch["last_appear"] = S["tick"]
        report["events"].append(["appear", S["tick"], cnt])
        analyse_batch(obs)
    elif watch["visible"] and cnt < thr:
        watch["visible"] = False
        watch["last_cover"] = S["tick"]
        report["events"].append(["cover", S["tick"], cnt])


# ---------------------------------------------------------------- button
button = None
try:
    loc = localize_object("button", camera_name="front", target_frame=base_frame)
    if loc.ok and len(loc.point_cloud.points) >= 5:
        pts = [[float(a) for a in p] for p in loc.point_cloud.points]
        top = quant([p[2] for p in pts], 0.9)
        tops = [p for p in pts if p[2] >= top - 0.01]
        button = [quant([p[0] for p in tops], 0.5), quant([p[1] for p in tops], 0.5), top]
        report["button_npts"] = len(pts)
except Exception as exc:
    report["errors"].append("button loc: " + str(exc)[:200])
if button is None:
    try:
        loc = localize_object("button", camera_name="wrist", target_frame=base_frame)
        if loc.ok and len(loc.point_cloud.points) >= 5:
            pts = [[float(a) for a in p] for p in loc.point_cloud.points]
            top = quant([p[2] for p in pts], 0.9)
            tops = [p for p in pts if p[2] >= top - 0.01]
            button = [quant([p[0] for p in tops], 0.5), quant([p[1] for p in tops], 0.5), top]
    except Exception as exc:
        report["errors"].append("button loc wrist: " + str(exc)[:200])
report["button"] = button

press_plan = None
if button is not None:
    hover_b = Pose([button[0], button[1], button[2] + 0.06], q_down, base_frame)
    press_b = Pose([button[0], button[1], button[2] - 0.015], q_down, base_frame)
    ik_h = solve_ik(hover_b, seed_joints=home_joints)
    if ik_h.ok:
        jh = [float(a) for a in ik_h.joint_positions]
        ik_p = solve_ik(press_b, seed_joints=jh)
        if ik_p.ok:
            press_plan = {"hover": jh, "press": [float(a) for a in ik_p.joint_positions]}
    report["press_ik"] = press_plan is not None


def servo(target, n, grip, settle):
    cur = [float(a) for a in get_robot_state().joint_positions["primary"]]
    obs = None
    for k in range(1, n + 1):
        frac = k / float(n)
        cmd = [cur[j] + (target[j] - cur[j]) * frac for j in range(7)]
        obs = tick(cmd, grip)
        poll(obs)
        if S["ended"]:
            return obs
    for k in range(settle):
        obs = tick(target, grip)
        poll(obs)
        if S["ended"]:
            return obs
        now = [float(a) for a in obs.robot_state.joint_positions["primary"]]
        if max(abs(now[j] - target[j]) for j in range(7)) < 0.01:
            break
    return obs


# ---------------------------------------------------------------- phase 1: watch + press
pressed = False
obs = get_observation()
poll(obs)
while not S["ended"] and S["tick"] < 400:
    if not pressed and press_plan is None:
        pressed = True
        report["errors"].append("no press plan; button not pressed")
    if not pressed and (len(batches) >= 1 or S["tick"] >= 25):
        S["phase"] = "moving"
        report["events"].append(["press_start", S["tick"]])
        servo(press_plan["hover"], 8, 0.0, 6)
        servo(press_plan["press"], 5, 0.0, 4)
        for k in range(2):
            o2 = tick(press_plan["press"], 0.0)
            poll(o2)
        try:
            report["press_ee"] = [float(a) for a in get_robot_state().end_effector_poses["primary"].position]
        except Exception:
            pass
        report["events"].append(["press_bottom", S["tick"]])
        servo(press_plan["hover"], 4, 0.0, 2)
        servo(home_joints, 8, 0.0, 6)
        report["events"].append(["press_done", S["tick"]])
        pressed = True
        S["phase"] = "home"
        continue
    obs = tick(home_joints, 0.0)
    poll(obs)
    if watch["visible"] and len(batches) > 0 and not batches[-1]["at_home"] and S["phase"] == "home":
        report["events"].append(["reanalyse_home", S["tick"]])
        analyse_batch(obs)
    if pressed and not watch["visible"] and watch["last_cover"] is not None:
        if S["tick"] - watch["last_cover"] > 30 and S["tick"] - watch["last_appear"] > 50:
            break
    if pressed and watch["last_cover"] is None and S["tick"] > 220:
        break
report["watch_end_tick"] = S["tick"]
report["pressed"] = pressed
report["batches"] = batches
report["counts"] = watch["counts"]

# ---------------------------------------------------------------- choose convention + merge
conv_err = {"cv": [], "gl": []}
for d in detections:
    if d.get("sam") is not None and d.get("deproj") is not None:
        for name in ("cv", "gl"):
            w = d["deproj"][name]
            conv_err[name].append(math.sqrt(sum((w[i] - d["sam"][i]) ** 2 for i in range(3))))
conv = "cv"
if conv_err["cv"] and conv_err["gl"]:
    if quant(conv_err["gl"], 0.5) < quant(conv_err["cv"], 0.5):
        conv = "gl"
else:
    below = 0
    for d in detections:
        if d.get("deproj") is not None:
            if d["deproj"]["cv"][2] < d["deproj"]["cam_z"]:
                below += 1
            else:
                below -= 1
    if below < 0:
        conv = "gl"
report["conv"] = conv
report["conv_err_med"] = {k: quant(v, 0.5) for k, v in conv_err.items()}

cubes = []
for d in detections:
    if d["color"] not in target_colors:
        continue
    pos = d.get("sam")
    src = "sam"
    if pos is None and d.get("deproj") is not None:
        pos = d["deproj"][conv]
        src = "deproj"
    if pos is None:
        continue
    merged = False
    for c in cubes:
        if c["color"] == d["color"] and math.sqrt((c["pos"][0] - pos[0]) ** 2 + (c["pos"][1] - pos[1]) ** 2) < 0.035:
            c["obs"].append([d["cam"], src, pos])
            c["support"] += d["n"]
            if src == "sam" and c["src"] != "sam":
                c["pos"] = pos
                c["src"] = "sam"
            merged = True
            break
    if not merged:
        cubes.append({"color": d["color"], "pos": pos, "src": src, "support": d["n"], "obs": [[d["cam"], src, pos]]})
report["cubes"] = cubes

# ---------------------------------------------------------------- phase 2: lift containers
finger0 = qrot(q_down, [0.0, 1.0, 0.0])
phi0 = math.atan2(finger0[1], finger0[0])


def min_area_yaw(pts):
    best = None
    for deg in range(0, 90, 3):
        a = math.radians(deg)
        ca = math.cos(a)
        sa = math.sin(a)
        xs = [p[0] * ca + p[1] * sa for p in pts]
        ys = [-p[0] * sa + p[1] * ca for p in pts]
        x0 = quant(xs, 0.03)
        x1 = quant(xs, 0.97)
        y0 = quant(ys, 0.03)
        y1 = quant(ys, 0.97)
        area = (x1 - x0) * (y1 - y0)
        if best is None or area < best[0]:
            mx = (x0 + x1) / 2.0
            my = (y0 + y1) / 2.0
            best = [area, a, mx * ca - my * sa, mx * sa + my * ca, x1 - x0, y1 - y0]
    return best


def find_container(target_xy, cam):
    for prompt in ("white box", "white container", "box"):
        try:
            seg = segment_text(cam, prompt)
        except Exception as exc:
            report["errors"].append("seg " + prompt + ": " + str(exc)[:120])
            continue
        if not seg.ok:
            continue
        best = None
        for cand in seg.segmentations[:12]:
            try:
                pts = cloud_points(cand, cam)
            except Exception:
                continue
            if len(pts) < 15:
                continue
            mxy = [quant([p[0] for p in pts], 0.5), quant([p[1] for p in pts], 0.5)]
            dist = math.sqrt((mxy[0] - target_xy[0]) ** 2 + (mxy[1] - target_xy[1]) ** 2)
            if dist < 0.05 and (best is None or dist < best[0]):
                best = [dist, pts, prompt]
        if best is not None:
            return best
    return None


def move(pos, quat, steps=200, tol=0.01):
    res = move_to_pose(Pose(list(pos), list(quat), base_frame), tolerance=tol, max_steps=steps)
    add_steps(res)
    return res


picks = []
order = []
for st in plan_steps:
    cands = [c for c in cubes if c["color"] == st["color"] and not c.get("used")]
    if not cands:
        order.append({"color": st["color"], "cube": None})
        continue
    if st["all"]:
        for c in cands:
            c["used"] = True
            order.append({"color": st["color"], "cube": c})
    else:
        c = cands[0]
        for cc in cands:
            if cc["support"] > c["support"]:
                c = cc
        c["used"] = True
        order.append({"color": st["color"], "cube": c})
report["order"] = [[o["color"], o["cube"]["pos"] if o["cube"] else None] for o in order]

for item in order:
    if S["ended"]:
        break
    rec = {"color": item["color"]}
    picks.append(rec)
    if item["cube"] is None:
        rec["error"] = "no memory of cube"
        continue
    cp = item["cube"]["pos"]
    hover_z = cp[2] + 0.22
    r1 = move([cp[0], cp[1], hover_z], q_down, 250, 0.015)
    rec["hover_ok"] = bool(r1.ok)
    if S["ended"]:
        break
    found = find_container([cp[0], cp[1]], "wrist")
    if found is None:
        found = find_container([cp[0], cp[1]], "front")
    if found is None:
        rec["container"] = None
        cx, cy = cp[0], cp[1]
        top_z = cp[2] + 0.04
        yaw = phi0
        rec["fallback"] = True
    else:
        pts = found[1]
        top_z = quant([p[2] for p in pts], 0.95)
        tops = [p for p in pts if p[2] >= top_z - 0.01]
        if len(tops) < 8:
            tops = pts
        mar = min_area_yaw(tops)
        cx, cy = mar[2], mar[3]
        yaw = mar[1]
        rec["container"] = {"dist": found[0], "prompt": found[2], "npts": len(pts), "top_z": top_z,
                            "side": [mar[4], mar[5]], "center": [cx, cy], "yaw_deg": math.degrees(yaw)}
    rot = yaw - phi0
    while rot > math.pi / 4:
        rot -= math.pi / 2
    while rot < -math.pi / 4:
        rot += math.pi / 2
    qg = qmul(qz(rot), q_down)
    height = max(0.02, top_z - (cp[2] - 0.01))
    grip_z = top_z - min(0.025, 0.45 * height)
    rec["grip_z"] = grip_z
    rec["rot_deg"] = math.degrees(rot)
    res = open_gripper()
    add_steps(res)
    move([cx, cy, top_z + 0.08], qg, 200, 0.01)
    if S["ended"]:
        break
    r2 = move([cx, cy, grip_z], qg, 150, 0.008)
    rec["descend_ok"] = bool(r2.ok)
    if S["ended"]:
        break
    res = close_gripper()
    add_steps(res)
    if S["ended"]:
        break
    r3 = move([cx, cy, grip_z + 0.12], qg, 150, 0.015)
    rec["lift_ok"] = bool(r3.ok)
    try:
        rec["grip_after_lift"] = float(get_robot_state().gripper_positions["primary"])
    except Exception:
        pass
    if S["ended"]:
        rec["ended_after_lift"] = True
        break
    for k in range(5):
        st_now = get_robot_state()
        tick([float(a) for a in st_now.joint_positions["primary"]], 0.0)
        if S["ended"]:
            break
    if S["ended"]:
        rec["ended_after_hold"] = True
        break
    move([cx, cy, grip_z + 0.005], qg, 150, 0.01)
    if S["ended"]:
        break
    res = open_gripper()
    add_steps(res)
    if S["ended"]:
        break
    move([cx, cy, top_z + 0.10], qg, 150, 0.015)
    rec["tick_done"] = S["tick"]

report["picks"] = picks
report["final_tick"] = S["tick"]
report["ended"] = S["ended"]
report["cmd_mode"] = S["cmd_mode"]
result = report
