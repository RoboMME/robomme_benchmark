# PickHighlight v05 - v04 + tracker exclusivity, picked cubes stay tracked as blockers, cube-top z band in trk_3d.
# Highlight = white discs under a subset of cubes (read live from the wrist view, never from colour).
# Cubes then swap places autonomously; every control tick is stepped by this program (IK + joint
# interpolation) so the front image can be tracked each tick and each target re-located before grasp.
import math

LOG = []
STATE = {"done": False, "ticks": 0}
report = {"version": "v05"}


def note(item):
    if len(LOG) < 160:
        LOG.append(item)


def quant(values, frac):
    vals = sorted(values)
    if len(vals) == 0:
        return None
    return vals[int((len(vals) - 1) * frac)]


def track(res, label):
    entry = [label]
    try:
        entry.append(bool(res.ok))
    except Exception:
        entry.append(None)
    try:
        entry.append(int(res.steps_executed))
        STATE["ticks"] += int(res.steps_executed)
    except Exception:
        entry.append(None)
    try:
        entry.append(str(res.status))
    except Exception:
        entry.append("")
    term = False
    try:
        term = bool(res.terminated) or bool(res.truncated)
    except Exception:
        term = False
    if term:
        STATE["done"] = True
        entry.append("TERMINAL")
    note(entry)
    return res


def rows_of(cam):
    try:
        return cam.rgb.tolist()
    except Exception:
        img = cam.rgb
        out = []
        for y in range(len(img)):
            row = []
            for x in range(len(img[0])):
                row.append([int(img[y][x][0]), int(img[y][x][1]), int(img[y][x][2])])
            out.append(row)
        return out


def is_white(px):
    lo = min(px[0], px[1], px[2])
    hi = max(px[0], px[1], px[2])
    return lo > 200 and hi - lo < 35


def white_map(rows):
    out = []
    for row in rows:
        out.append([is_white(px) for px in row])
    return out


def count_new_white(rows, base_white, stride):
    n = 0
    for y in range(0, len(rows), stride):
        row = rows[y]
        brow = base_white[y]
        for x in range(0, len(row), stride):
            if (not brow[x]) and is_white(row[x]):
                n += 1
    return n


def hold_tick(grip):
    st = get_robot_state()
    sr = step(RobotAction(arms={"primary": ArmCommand(
        "joint_position", st.joint_positions["primary"], gripper_position=grip,
        embodiment=st.embodiment)}))
    ok = False
    try:
        ok = bool(sr.ok)
    except Exception:
        ok = False
    STATE["ticks"] += 1
    try:
        if bool(sr.terminated) or bool(sr.truncated):
            STATE["done"] = True
    except Exception:
        pass
    return ok


def cloud_points(cand, cam, base):
    try:
        pc = mask_to_point_cloud(cand, cam, target_frame=base)
    except Exception:
        return []
    try:
        pts = pc.points.tolist()
    except Exception:
        try:
            pts = [list(p) for p in pc.points]
        except Exception:
            pts = []
    return pts


def box_of(cand):
    try:
        b = cand.box_xyxy
        if b is None:
            return None
        return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    except Exception:
        return None


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    if ua <= 0:
        return 0.0
    return inter / ua


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def yaw_quat(q0, yaw):
    qz = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
    return qmul(qz, q0)


def finger_axis(q):
    w, x, y, z = q
    ax = 2.0 * (x * y - w * z)
    ay = 1.0 - 2.0 * (x * x + z * z)
    n = math.sqrt(ax * ax + ay * ay)
    if n < 1e-6:
        return (0.0, 1.0)
    return (ax / n, ay / n)


# ---------------------------------------------------------------- setup
ctx = get_task_context()
goal = str(ctx.language)
report["goal"] = goal
try:
    meta = robomme_plus.get_task_metadata()
    report["max_steps"] = meta["max_steps"]
except Exception:
    report["max_steps"] = None
state0 = get_robot_state()
base = state0.base_frame
emb = state0.embodiment
home_q = tuple(float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz)
home_p = [float(v) for v in state0.end_effector_poses["primary"].position]
report["home_ee"] = [round(v, 3) for v in home_p]
grip0 = float(state0.gripper_positions["primary"])

obs0 = get_observation()
rows0 = rows_of(obs0.cameras["front"])
white0 = white_map(rows0)
H = len(rows0)
W = len(rows0[0])
wrows0 = rows_of(obs0.cameras["wrist"])
wwhite0 = white_map(wrows0)
WH = len(wrows0)
WW = len(wrows0[0])

# ---------------------------------------------------------------- objects at tick 0
cubes = []
cube_prompts = ["cube", "block"]
seg_counts = {}
for prompt in cube_prompts:
    seg = segment_text("front", prompt)
    if not seg.ok:
        seg_counts[prompt] = 0
        continue
    seg_counts[prompt] = len(seg.segmentations)
    for cand in seg.segmentations:
        b = box_of(cand)
        if b is None:
            continue
        dup = False
        for c in cubes:
            if iou(c["box"], b) > 0.5:
                dup = True
        if dup:
            continue
        cubes.append({"box": b, "cand": cand, "score": float(cand.score), "prompt": prompt})
report["seg_counts"] = seg_counts

button = None
bseg = segment_text("front", "button")
bcands = []
if bseg.ok:
    for cand in bseg.segmentations[:5]:
        b = box_of(cand)
        if b is not None:
            bcands.append([round(float(cand.score), 3)] + [round(v) for v in b])
    if len(bseg.segmentations) > 0:
        bc = bseg.segmentations[0]
        bpts = cloud_points(bc, "front", base)
        if len(bpts) >= 10:
            button = {"box": box_of(bc),
                      "x": quant([p[0] for p in bpts], 0.5),
                      "y": quant([p[1] for p in bpts], 0.5),
                      "top": quant([p[2] for p in bpts], 0.95),
                      "n": len(bpts)}
report["button_cands"] = bcands
report["button"] = None if button is None else [round(button["x"], 3), round(button["y"], 3), round(button["top"], 3)]

# 3D for each cube candidate
kept = []
for c in cubes:
    if button is not None and button["box"] is not None and iou(c["box"], button["box"]) > 0.3:
        continue
    pts = cloud_points(c["cand"], "front", base)
    if len(pts) < 8:
        continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    c["x"] = quant(xs, 0.5)
    c["y"] = quant(ys, 0.5)
    c["top"] = quant(zs, 0.9)
    c["bot"] = quant(zs, 0.05)
    c["size"] = max(quant(xs, 0.9) - quant(xs, 0.1), quant(ys, 0.9) - quant(ys, 0.1))
    c["n"] = len(pts)
    kept.append(c)
table_z = quant([c["bot"] for c in kept], 0.5) if len(kept) > 0 else None
cube_list = []
for c in kept:
    if table_z is not None and c["top"] - table_z > 0.09:
        continue
    if c["size"] > 0.09:
        continue
    c["id"] = len(cube_list)
    cube_list.append(c)
report["table_z"] = None if table_z is None else round(table_z, 4)
report["n_cubes"] = len(cube_list)


def find_holes(rows, white_base, R):
    hh = len(rows)
    ww = len(rows[0])
    nw = []
    hw = []
    for y in range(hh):
        nrow = []
        hrow = []
        for x in range(ww):
            wnow = is_white(rows[y][x])
            hrow.append(wnow)
            nrow.append(wnow and not white_base[y][x])
        nw.append(nrow)
        hw.append(hrow)
    big = 100000
    left = []
    right = []
    for y in range(hh):
        lrow = [big] * ww
        last = -big
        for x in range(ww):
            if nw[y][x]:
                last = x
            lrow[x] = x - last
        rrow = [big] * ww
        last = big
        for x in range(ww - 1, -1, -1):
            if nw[y][x]:
                last = x
            rrow[x] = last - x
        left.append(lrow)
        right.append(rrow)
    up = [[big] * ww for y in range(hh)]
    down = [[big] * ww for y in range(hh)]
    for x in range(ww):
        last = -big
        for y in range(hh):
            if nw[y][x]:
                last = y
            up[y][x] = y - last
        last = big
        for y in range(hh - 1, -1, -1):
            if nw[y][x]:
                last = y
            down[y][x] = last - y
    enc = set()
    for y in range(hh):
        for x in range(ww):
            if hw[y][x]:
                continue
            if left[y][x] <= R and right[y][x] <= R and up[y][x] <= R and down[y][x] <= R:
                enc.add((y, x))
    comps = []
    while len(enc) > 0:
        seed = enc.pop()
        stack = [seed]
        comp = [seed]
        while len(stack) > 0:
            py, px = stack.pop()
            for q in ((py + 1, px), (py - 1, px), (py, px + 1), (py, px - 1)):
                if q in enc:
                    enc.remove(q)
                    stack.append(q)
                    comp.append(q)
        comps.append(comp)
    holes = []
    for comp in comps:
        if len(comp) < 6:
            continue
        cy = sum(p[0] for p in comp) / float(len(comp))
        cx = sum(p[1] for p in comp) / float(len(comp))
        holes.append([cx, cy, len(comp)])
    # merge holes whose centres are close (one cube split by a dark edge)
    out = []
    for hcand in holes:
        joined = False
        for o in out:
            if (o[0] - hcand[0]) ** 2 + (o[1] - hcand[1]) ** 2 < 100.0:
                tot = o[2] + hcand[2]
                o[0] = (o[0] * o[2] + hcand[0] * hcand[2]) / tot
                o[1] = (o[1] * o[2] + hcand[1] * hcand[2]) / tot
                o[2] = tot
                joined = True
                break
        if not joined:
            out.append([hcand[0], hcand[1], hcand[2]])
    final = []
    for o in out:
        if o[2] < 50 or o[2] > 900:
            continue
        x0 = max(0, int(o[0]) - R)
        x1 = min(ww - 1, int(o[0]) + R)
        y0 = max(0, int(o[1]) - R)
        y1 = min(hh - 1, int(o[1]) + R)
        nnew = 0
        nbase = 0
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if nw[y][x]:
                    nnew += 1
                if white_base[y][x]:
                    nbase += 1
        o.append(nnew)
        o.append(nbase)
        if nnew >= 40 and nbase < nnew:
            final.append(o)
    return final

# ---------------------------------------------------------------- camera geometry (convention self-calibrated at runtime)
def as_list(a):
    try:
        return a.tolist()
    except Exception:
        return [list(r) for r in a]


def cam_model(cam):
    K = as_list(cam.intrinsics)
    pose = cam.camera_pose
    t = [float(v) for v in pose.position]
    w, x, y, z = [float(v) for v in pose.quaternion_wxyz]
    R = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    return {"fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]), "cy": float(K[1][2]), "R": R, "t": t}


def project(cm, p, conv):
    d = [p[0] - cm["t"][0], p[1] - cm["t"][1], p[2] - cm["t"][2]]
    R = cm["R"]
    X = R[0][0] * d[0] + R[1][0] * d[1] + R[2][0] * d[2]
    Y = R[0][1] * d[0] + R[1][1] * d[1] + R[2][1] * d[2]
    Z = R[0][2] * d[0] + R[1][2] * d[1] + R[2][2] * d[2]
    if conv == 0:
        if Z <= 1e-6:
            return None
        return (cm["fx"] * X / Z + cm["cx"], cm["fy"] * Y / Z + cm["cy"])
    if conv == 1:
        if -Z <= 1e-6:
            return None
        return (cm["fx"] * X / (-Z) + cm["cx"], cm["cy"] - cm["fy"] * Y / (-Z))
    if X <= 1e-6:
        return None
    return (cm["cx"] - cm["fx"] * Y / X, cm["cy"] - cm["fy"] * Z / X)


def backproject(cm, u, v, dep, conv):
    a = (u - cm["cx"]) / cm["fx"] * dep
    b = (v - cm["cy"]) / cm["fy"] * dep
    if conv == 0:
        pc = (a, b, dep)
    elif conv == 1:
        pc = (a, -b, -dep)
    else:
        pc = (dep, -a, -b)
    R = cm["R"]
    t = cm["t"]
    return [R[0][0] * pc[0] + R[0][1] * pc[1] + R[0][2] * pc[2] + t[0],
            R[1][0] * pc[0] + R[1][1] * pc[1] + R[1][2] * pc[2] + t[1],
            R[2][0] * pc[0] + R[2][1] * pc[1] + R[2][2] * pc[2] + t[2]]


def depth_at(cam, u, v, r):
    try:
        dm = cam.depth_m
        hh = len(dm)
        ww = len(dm[0])
        x0 = max(0, int(u) - r)
        x1 = min(ww - 1, int(u) + r)
        y0 = max(0, int(v) - r)
        y1 = min(hh - 1, int(v) + r)
        vals = []
        for row in dm[y0:y1 + 1, x0:x1 + 1].tolist():
            for d in row:
                if d > 0.05 and d < 5.0:
                    vals.append(d)
        return quant(vals, 0.5)
    except Exception:
        return None


def calibrate(cam_name, cands, cam):
    # cands: list of (box, [x,y,z] centroid)
    cm = cam_model(cam)
    res = []
    for conv in (0, 1, 2):
        errs = []
        for b, p in cands:
            pr = project(cm, p, conv)
            if pr is None:
                errs.append(999.0)
                continue
            errs.append(math.sqrt((pr[0] - (b[0] + b[2]) / 2.0) ** 2 + (pr[1] - (b[1] + b[3]) / 2.0) ** 2))
        res.append((quant(errs, 0.5) if len(errs) > 0 else 999.0, conv))
    res.sort()
    note(["calib", cam_name, [[round(e, 1), c] for e, c in res]])
    if res[0][0] < 8.0:
        return res[0][1]
    return None


def cloud_stats(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    return {"x": quant(xs, 0.5), "y": quant(ys, 0.5), "top": quant(zs, 0.9), "bot": quant(zs, 0.05),
            "zm": quant(zs, 0.5),
            "size": max(quant(xs, 0.9) - quant(xs, 0.1), quant(ys, 0.9) - quant(ys, 0.1)), "n": len(pts)}


front_cal = [(c["box"], [c["x"], c["y"], 0.5 * (c["top"] + c["bot"])]) for c in cube_list]
front_conv = calibrate("front", front_cal, obs0.cameras["front"])
wrist_cal = []
wseg0 = segment_text("wrist", "cube")
if wseg0.ok:
    for cand in wseg0.segmentations[:5]:
        b = box_of(cand)
        if b is None:
            continue
        pts = cloud_points(cand, "wrist", base)
        if len(pts) < 8:
            continue
        cs = cloud_stats(pts)
        if cs["size"] > 0.07:
            continue
        wrist_cal.append((b, [cs["x"], cs["y"], cs["zm"]]))
wrist_conv = calibrate("wrist", wrist_cal, obs0.cameras["wrist"])
report["conv"] = [front_conv, wrist_conv]

# ---------------------------------------------------------------- front-view template trackers
TR = 3


def patch_at(img, u, v, r):
    hh = len(img)
    ww = len(img[0])
    if u - r < 0 or v - r < 0 or u + r >= ww or v + r >= hh:
        return None
    return img[v - r:v + r + 1, u - r:u + r + 1].tolist()


def track_one(trk, img, others):
    r = TR
    S = trk["S"]
    hh = len(img)
    ww = len(img[0])
    pu = int(round(trk["u"] + trk["vu"]))
    pv = int(round(trk["v"] + trk["vv"]))
    x0 = max(0, pu - r - S)
    y0 = max(0, pv - r - S)
    x1 = min(ww - 1, pu + r + S)
    y1 = min(hh - 1, pv + r + S)
    reg = img[y0:y1 + 1, x0:x1 + 1].tolist()
    T = trk["T"]
    n = (2 * r + 1) * (2 * r + 1) * 3
    best = None
    for cv in range(y0 + r, y1 - r + 1):
        for cu in range(x0 + r, x1 - r + 1):
            blocked = False
            for o in others:
                if (cu - o[0]) ** 2 + (cv - o[1]) ** 2 < 16:
                    blocked = True
            if blocked:
                continue
            pen = 0.6 * ((cu - pu) ** 2 + (cv - pv) ** 2)
            lim = None if best is None else (best[0] - pen) * n
            s = 0
            stop = False
            for j in range(2 * r + 1):
                rr = reg[cv - r + j - y0]
                tr = T[j]
                for i in range(2 * r + 1):
                    a = rr[cu - r + i - x0]
                    b = tr[i]
                    s += abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
                if lim is not None and s >= lim:
                    stop = True
                    break
            if stop:
                continue
            sc = s / n + pen
            if best is None or sc < best[0]:
                best = (sc, cu, cv, s / n)
    if best is None:
        trk["lost"] += 1
        return
    trk["err"] = best[3]
    if best[3] < trk["acc"]:
        nu = best[1]
        nv = best[2]
        k = 1.0 + trk["lost"]
        trk["vu"] = 0.5 * trk["vu"] + 0.5 * (nu - trk["u"]) / k
        trk["vv"] = 0.5 * trk["vv"] + 0.5 * (nv - trk["v"]) / k
        if abs(trk["vu"]) > 2.5 or abs(trk["vv"]) > 2.5:
            trk["vu"] = 0.0
            trk["vv"] = 0.0
        if abs(nu - trk["u0"]) + abs(nv - trk["v0"]) > 0:
            trk["moved"] = max(trk["moved"], abs(nu - trk["u0"]) + abs(nv - trk["v0"]))
        trk["u"] = nu
        trk["v"] = nv
        if trk["lost"] > 0:
            trk["events"] += 1
        trk["lost"] = 0
        trk["S"] = 5
        if best[3] < trk["upd"]:
            p = patch_at(img, nu, nv, r)
            if p is not None:
                trk["T"] = p
    else:
        trk["lost"] += 1
        trk["maxlost"] = max(trk["maxlost"], trk["lost"])
        if trk["lost"] <= 4:
            trk["u"] = trk["u"] + trk["vu"]
            trk["v"] = trk["v"] + trk["vv"]
        else:
            trk["vu"] = 0.0
            trk["vv"] = 0.0
        trk["S"] = min(9, 5 + trk["lost"] // 2)


def new_tracker(img, u, v, name):
    p = patch_at(img, int(round(u)), int(round(v)), TR)
    if p is None:
        return None
    return {"name": name, "u": int(round(u)), "v": int(round(v)), "u0": int(round(u)), "v0": int(round(v)),
            "vu": 0.0, "vv": 0.0, "T": p, "S": 5, "lost": 0, "err": 0.0, "acc": 32.0, "upd": 26.0,
            "maxlost": 0, "moved": 0, "events": 0, "active": True, "todo": True, "p3": None}


TRK = []
STATE["obs"] = obs0
STATE["bad_steps"] = 0


def update_trackers(obs):
    try:
        img = obs.cameras["front"].rgb
    except Exception:
        return
    for trk in TRK:
        if trk["active"]:
            others = [(o["u"], o["v"]) for o in TRK if o["name"] != trk["name"]]
            track_one(trk, img, others)


def apply_cmd(joints, g):
    sr = step(RobotAction(arms={"primary": ArmCommand(
        "joint_position", joints, gripper_position=g, embodiment=emb)}))
    STATE["ticks"] += 1
    ok = False
    try:
        ok = bool(sr.ok)
    except Exception:
        ok = False
    try:
        if bool(sr.terminated) or bool(sr.truncated):
            STATE["done"] = True
            note(["TERMINAL", STATE["ticks"]])
            return False
    except Exception:
        pass
    if not ok:
        STATE["bad_steps"] += 1
    ob = None
    try:
        ob = sr.observation
        ob.cameras["front"]
    except Exception:
        ob = None
    if ob is None:
        try:
            ob = get_observation()
        except Exception:
            ob = None
    if ob is not None:
        STATE["obs"] = ob
        update_trackers(ob)
    return ok


def cur_joints():
    return [float(v) for v in get_robot_state().joint_positions["primary"]]


def hold(g, n):
    if STATE["done"]:
        return
    j = cur_joints()
    for k in range(n):
        if STATE["done"]:
            return
        apply_cmd(j, g)


def ee_pos():
    return [float(v) for v in get_robot_state().end_effector_poses["primary"].position]


def go(xyz, q, g, tol, tag, maxv=0.05, settle=25):
    if STATE["done"]:
        return False
    j0 = cur_joints()
    ik = None
    try:
        ik = solve_ik(Pose((xyz[0], xyz[1], xyz[2]), q, base), seed_joints=j0)
    except Exception:
        ik = None
    if ik is None or not ik.ok:
        note([tag, "ik_fail_fallback"])
        track(move_to_pose(Pose((xyz[0], xyz[1], xyz[2]), q, base), tolerance=tol, max_steps=150), tag + ":mtp")
        return not STATE["done"]
    j1 = [float(v) for v in ik.joint_positions]
    dmax = max(abs(j1[i] - j0[i]) for i in range(len(j1)))
    n = max(3, int(dmax / maxv) + 1)
    t0 = STATE["ticks"]
    for i in range(1, n + 1):
        if STATE["done"]:
            return False
        f = float(i) / n
        apply_cmd([j0[k] + (j1[k] - j0[k]) * f for k in range(len(j1))], g)
    err = None
    for k in range(settle):
        if STATE["done"]:
            return False
        p = ee_pos()
        err = math.sqrt((p[0] - xyz[0]) ** 2 + (p[1] - xyz[1]) ** 2 + (p[2] - xyz[2]) ** 2)
        if err < tol:
            break
        apply_cmd(j1, g)
    note([tag, STATE["ticks"] - t0, None if err is None else round(err, 3)])
    return True


def set_grip(g, tag, nmax=14):
    if STATE["done"]:
        return
    j = cur_joints()
    last = None
    still = 0
    for k in range(nmax):
        if STATE["done"]:
            return
        apply_cmd(j, g)
        if STATE["done"]:
            return
        r = float(get_robot_state().gripper_positions["primary"])
        if last is not None and abs(r - last) < 0.003:
            still += 1
            if still >= 2 and k >= 3:
                break
        else:
            still = 0
        last = r


def seg_point_hole(u, v):
    # segment_points masks come back with box_xyxy=None: validate by the 3D cloud instead
    try:
        sp = segment_points("wrist", [[int(round(u)), int(round(v))]])
    except Exception:
        return None
    if not sp.ok:
        return None
    bestc = None
    for cand in sp.segmentations[:3]:
        pts = cloud_points(cand, "wrist", base)
        if len(pts) < 8:
            continue
        cs = cloud_stats(pts)
        if cs["size"] > 0.06:
            continue
        if bestc is None or cs["n"] > bestc["n"]:
            bestc = cs
    if bestc is None:
        return None
    return [bestc["x"], bestc["y"], bestc["top"]]


# ---------------------------------------------------------------- wait for highlight (wrist view, arm held still), start trackers
hold_j = cur_joints()
whl_rows = None
whl_obs = None
detect_tick = None
peak = 0
counts = []
clean_tick = None
for t in range(95):
    if STATE["done"]:
        break
    apply_cmd(hold_j, grip0)
    ob = STATE["obs"]
    wr = rows_of(ob.cameras["wrist"])
    n = count_new_white(wr, wwhite0, 2)
    counts.append(n)
    if n >= 25 and detect_tick is None:
        detect_tick = t + 1
    if detect_tick is not None:
        if n > peak:
            peak = n
        if t + 1 == detect_tick + 2:
            whl_rows = wr
            whl_obs = ob
            # resolve holes now and start front trackers on this very frame
            holes = find_holes(whl_rows, wwhite0, 20)
            report["wrist_holes"] = [[round(h[0]), round(h[1]), h[2], h[3], h[4]] for h in holes]
            wcm = cam_model(ob.cameras["wrist"])
            fcm = cam_model(ob.cameras["front"])
            fimg = ob.cameras["front"].rgb
            for hcand in holes:
                p3 = None
                src = None
                if wrist_conv is not None:
                    dep = depth_at(ob.cameras["wrist"], hcand[0], hcand[1], 2)
                    if dep is not None:
                        p3 = backproject(wcm, hcand[0], hcand[1], dep, wrist_conv)
                        src = "wdepth"
                if p3 is None or table_z is None or not (table_z + 0.012 < p3[2] < table_z + 0.09):
                    p3 = seg_point_hole(hcand[0], hcand[1])
                    src = "wpoint"
                if p3 is None or table_z is None or not (table_z + 0.012 < p3[2] < table_z + 0.09):
                    note(["hole_bad3d", round(hcand[0]), round(hcand[1]), None if p3 is None else [round(v, 3) for v in p3]])
                    continue
                if front_conv is None:
                    note(["no_front_conv"])
                    continue
                pr = project(fcm, p3, front_conv)
                if pr is None:
                    continue
                trk = new_tracker(fimg, pr[0], pr[1], "h" + str(len(TRK)))
                if trk is None:
                    continue
                trk["p3"] = p3
                trk["top"] = p3[2]
                trk["src"] = src
                TRK.append(trk)
        if whl_rows is not None and n < 0.15 * peak:
            clean_tick = t + 1
            break
report["white_counts"] = counts[:12] + ["..."] + counts[-3:]
report["detect_tick"] = detect_tick
report["clean_tick"] = clean_tick
report["targets0"] = [[t["name"], [round(v, 3) for v in t["p3"]], t["u"], t["v"]] for t in TRK]


def trk_3d(trk):
    # back-project a window around the tracked pixel; keep only cube-top-surface points
    ob = STATE["obs"]
    if trk["lost"] > 0 or front_conv is None or table_z is None:
        return trk["p3"]
    try:
        cam = ob.cameras["front"]
        cm = cam_model(cam)
        dm = cam.depth_m
        r = 4
        u0 = int(trk["u"])
        v0 = int(trk["v"])
        rows = dm[v0 - r:v0 + r + 1, u0 - r:u0 + r + 1].tolist()
    except Exception:
        return trk["p3"]
    pts = []
    for j in range(len(rows)):
        for i in range(len(rows[j])):
            d = rows[j][i]
            if d <= 0.05 or d >= 5.0:
                continue
            p = backproject(cm, u0 - r + i, v0 - r + j, d, front_conv)
            if table_z + 0.02 < p[2] < trk["top"] + 0.012:
                pts.append(p)
    if len(pts) < 3:
        return trk["p3"]
    zmax = quant([p[2] for p in pts], 0.9)
    topp = [p for p in pts if p[2] > zmax - 0.012]
    if len(topp) < 2:
        return trk["p3"]
    trk["p3"] = [quant([p[0] for p in topp], 0.5), quant([p[1] for p in topp], 0.5), zmax]
    return trk["p3"]


# ---------------------------------------------------------------- motion helpers
hover_z = (table_z if table_z is not None else 0.0) + 0.14


def press_button(tag):
    if button is None or STATE["done"]:
        note([tag, "no button"])
        return
    set_grip(0.0, tag + ":close", 8)
    above = (button["x"], button["y"], button["top"] + 0.08)
    go(above, home_q, 0.0, 0.015, tag + ":above")
    go((button["x"], button["y"], button["top"] - 0.02), home_q, 0.0, 0.006, tag + ":press", 0.03, 10)
    hold(0.0, 4)
    go(above, home_q, 0.0, 0.02, tag + ":rise")


def best_yaw(c, obstacles):
    best = None
    for k in range(-6, 7):
        yaw = k * math.pi / 12.0
        q = yaw_quat(home_q, yaw)
        ax = finger_axis(q)
        clear = 1.0
        for sign in (1.0, -1.0):
            fx = c["x"] + sign * 0.05 * ax[0]
            fy = c["y"] + sign * 0.05 * ax[1]
            for o in obstacles:
                if math.sqrt((o["x"] - c["x"]) ** 2 + (o["y"] - c["y"]) ** 2) < 0.015:
                    continue
                d = math.sqrt((fx - o["x"]) ** 2 + (fy - o["y"]) ** 2) - 0.5 * o["size"]
                if d < clear:
                    clear = d
        if button is not None:
            for sign in (1.0, -1.0):
                fx = c["x"] + sign * 0.05 * ax[0]
                fy = c["y"] + sign * 0.05 * ax[1]
                d = math.sqrt((fx - button["x"]) ** 2 + (fy - button["y"]) ** 2) - 0.03
                if d < clear:
                    clear = d
        key = (round(clear, 3), -abs(k))
        if best is None or key > best[0]:
            best = (key, yaw, q)
    return best


def wrist_point_refine(p3):
    # cube-height-validated wrist candidates (point prompt at the projected position + text "cube")
    cands = []
    if wrist_conv is not None:
        try:
            cam = get_observation().cameras["wrist"]
            pr = project(cam_model(cam), p3, wrist_conv)
            if pr is not None:
                sp = segment_points("wrist", [[int(round(pr[0])), int(round(pr[1]))]])
                if sp.ok:
                    cands = cands + [c for c in sp.segmentations[:3]]
        except Exception:
            pass
    try:
        st = segment_text("wrist", "cube")
        if st.ok:
            cands = cands + [c for c in st.segmentations[:5]]
    except Exception:
        pass
    bestc = None
    for cand in cands:
        pts = cloud_points(cand, "wrist", base)
        if len(pts) < 8:
            continue
        cs = cloud_stats(pts)
        if cs["size"] > 0.06 or not (table_z + 0.025 < cs["top"] < table_z + 0.075):
            continue
        d = math.sqrt((cs["x"] - p3[0]) ** 2 + (cs["y"] - p3[1]) ** 2)
        if d < 0.025 and (bestc is None or d < bestc[0]):
            bestc = (d, cs)
    return bestc


def wrist_neighbours():
    seen = []
    try:
        seg = segment_text("wrist", "cube")
    except Exception:
        return seen
    if not seg.ok:
        return seen
    for cand in seg.segmentations[:5]:
        pts = cloud_points(cand, "wrist", base)
        if len(pts) < 8:
            continue
        cs = cloud_stats(pts)
        if cs["size"] < 0.07:
            seen.append({"x": cs["x"], "y": cs["y"], "size": max(cs["size"], 0.03)})
    return seen


def grip_now():
    return float(get_robot_state().gripper_positions["primary"])


def pick_one(trk, idx):
    tag = "c" + str(idx)
    p3 = trk_3d(trk)
    note([tag, trk["name"], "target", [round(v, 3) for v in p3], trk["lost"], trk["moved"]])
    set_grip(1.0, tag + ":open", 10)
    go((p3[0], p3[1], hover_z), home_q, 1.0, 0.015, tag + ":hover")
    if STATE["done"]:
        return False
    p3 = trk_3d(trk)
    ref = wrist_point_refine(p3)
    gx = p3[0]
    gy = p3[1]
    top = trk["top"]
    if ref is not None:
        note([tag, "wref", round(ref[0], 3), round(ref[1]["x"], 3), round(ref[1]["y"], 3), round(ref[1]["top"], 3)])
        gx = ref[1]["x"]
        gy = ref[1]["y"]
        if table_z + 0.02 < ref[1]["top"] < table_z + 0.08:
            top = ref[1]["top"]
    obstacles = wrist_neighbours() + [o for o in cube_list]
    for o in TRK:
        if o["name"] != trk["name"] and o["p3"] is not None:
            obstacles.append({"x": o["p3"][0], "y": o["p3"][1], "size": 0.04})
    yb = best_yaw({"x": gx, "y": gy}, obstacles)
    q = yb[2]
    note([tag, "yaw", round(yb[1], 2), "clear", yb[0][0]])
    height = top - table_z
    gz = table_z + max(0.5 * height, 0.012)
    go((gx, gy, gz + 0.05), q, 1.0, 0.01, tag + ":pre", 0.04)
    if STATE["done"]:
        return False
    pn = trk_3d(trk)
    if trk["lost"] == 0 and math.sqrt((pn[0] - gx) ** 2 + (pn[1] - gy) ** 2) > 0.025:
        note([tag, "target_moved", [round(v, 3) for v in pn]])
        go((gx, gy, hover_z), q, 1.0, 0.02, tag + ":back_up")
        return False
    go((gx, gy, gz), q, 1.0, 0.008, tag + ":down", 0.03)
    if STATE["done"]:
        return False
    trk["active"] = False
    set_grip(0.0, tag + ":close", 14)
    if STATE["done"]:
        return False
    if STATE["done"]:
        return False
    g = grip_now()
    note([tag, "grip", round(g, 3)])
    if g <= 0.03:
        set_grip(1.0, tag + ":reopen", 10)
        go((gx, gy, hover_z), q, 1.0, 0.02, tag + ":abort_up")
        trk["active"] = True
        return False
    go((gx, gy, gz + 0.10), q, 0.0, 0.015, tag + ":lift", 0.04)
    if STATE["done"]:
        return True
    hold(0.0, 3)
    g2 = grip_now() if not STATE["done"] else g
    note([tag, "grip_lifted", round(g2, 3)])
    go((gx, gy, gz + 0.008), q, 0.0, 0.01, tag + ":lower", 0.04)
    if STATE["done"]:
        return g2 > 0.03
    set_grip(1.0, tag + ":release", 10)
    go((gx, gy, hover_z), q, 1.0, 0.02, tag + ":up")
    # keep the put-back cube tracked so no other tracker can jump onto it
    if not STATE["done"] and front_conv is not None:
        try:
            cam = STATE["obs"].cameras["front"]
            pr = project(cam_model(cam), [gx, gy, trk["top"]], front_conv)
            if pr is not None:
                nt = new_tracker(cam.rgb, pr[0], pr[1], trk["name"])
                if nt is not None:
                    trk["u"] = nt["u"]
                    trk["v"] = nt["v"]
                    trk["T"] = nt["T"]
                    trk["vu"] = 0.0
                    trk["vv"] = 0.0
                    trk["lost"] = 0
                    trk["active"] = True
        except Exception:
            pass
    return g2 > 0.03


# ---------------------------------------------------------------- execute
picked = []
try:
    press_button("press1")
    picked = []
    idx = 0
    attempts = {}
    while not STATE["done"]:
        rem = [t for t in TRK if t["todo"]]
        if len(rem) == 0:
            break
        # prefer trackers currently locked; among them the far side (smallest x) first so the arm occludes done cubes only
        bestt = None
        for t in rem:
            p = trk_3d(t)
            key = (1 if t["lost"] > 0 else 0, p[0])
            if bestt is None or key < bestt[0]:
                bestt = (key, t)
        t = bestt[1]
        attempts[t["name"]] = attempts.get(t["name"], 0) + 1
        ok = pick_one(t, idx)
        idx += 1
        if ok:
            picked.append(t["name"])
            t["todo"] = False
        elif attempts[t["name"]] >= 2:
            t["todo"] = False
            note([t["name"], "gave_up"])
    report["picked"] = picked
    if not STATE["done"]:
        press_button("press2")
    if not STATE["done"]:
        hold(0.0, 5)
except Exception as exc:
    note(["exec_exception", str(exc)[:200]])
    report["picked"] = picked

report["ticks"] = STATE["ticks"]
report["bad_steps"] = STATE["bad_steps"]
report["done"] = STATE["done"]
report["trackers"] = [[t["name"], t["u"], t["v"], t["lost"], t["maxlost"], t["moved"], t["events"], round(t["err"], 1)] for t in TRK]
report["log"] = LOG
result = report
