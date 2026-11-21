import math

# v05 -- from v04 (ep1 fail at the red lift: while the pressing hand hid the red container,
#  its LOST track coasted next to another container and captured it by k-means; the true red
#  container re-emerged as an unclaimed orphan). v05: (a) lost tracks never take points by
#  k-means -- they re-attach only through orphan-cluster recovery (cap 0.15 m); (b) during
#  the presses the hand exclusion is a 0.12 x 0.24 m box aligned with the finger axis
#  instead of a 0.13 m disc, so nearby containers stay visible; (c) faster high transits.
# v04 -- from v03 (ep1 still failed: a container hidden behind the front-view pillar at the
#  drop was never tracked, and recovery jumps of up to 0.3 m swapped identities). v04 fuses
#  the top-face points of BOTH cameras each tick: the wrist camera, held at the home pose,
#  looks straight down, so the pillar cannot hide a container top; each camera is
#  back-projected with its own live pose, so the tracker stays in one metric frame even
#  while the wrist moves during the presses. Recovery jumps are capped at 0.12 m.
# v03 -- from v02 (ep0 success, ep1 fail). ep1 had a tall gray occluding pillar: a static
#  track latched onto it and containers passing behind it / under the pressing hand lost
#  identity (green track captured by the hand). v03 adds: (a) static tall white structure
#  observed at t0 (before the containers drop) excluded by xy footprint; (b) hand exclusion
#  radius 0.13 m (Panda hand is ~0.2 m wide); (c) lost-track recovery: a track without
#  points is re-attached to the nearest orphan top-face cluster (identity by elimination);
#  (d) two tracks closer than 3 cm -> the weaker is declared lost.
# v02 -- ButtonUnmaskSwap. Fixes v01's two failure modes on ep0:
#  (tracking) 2D white-blob k-means let tracks merge onto one container and get captured
#  by the white robot arm -> v02 tracks containers in METRIC table coordinates using only
#  the back-projected TOP-FACE points of white pixels (live depth); top faces of
#  equal-height containers never overlap, and the arm/buttons fall outside the height band.
#  Tracks coast with their velocity while occluded, and a track never takes points
#  nearer to another track.
#  (grasp) v01 blue grasp closed on air (gripper 0.0) at a centroid between two
#  containers -> grasp at the tracked top-face centroid, fingers yawed to the
#  container faces (4-fold orientation of top-face points), verify the gripper did not
#  close fully, retry once from a fresh observation.

R = {"errors": [], "events": [], "trk_log": [], "recoveries": []}
result = R
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "white", "black",
          "brown", "gray", "grey", "cyan", "magenta"]
TICKS = {"n": 0}
STATE = {"ended": False}
PRESS_AT = 100          # start pressing at this tick (v01: presses at 129/149 registered fine)
QUIET_TICKS = 20
STRIDE = 2


def err(tag, e):
    R["errors"].append("[t%d] %s: %s" % (TICKS["n"], tag, str(e)[:300]))


def ev(msg):
    R["events"].append("[t%d] %s" % (TICKS["n"], msg))


def is_term(r):
    t = False
    try:
        t = bool(r.terminated)
    except Exception:
        pass
    try:
        t = t or bool(r.truncated)
    except Exception:
        pass
    return t


def qrot(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def qmul(a, b):
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]


def white(px):
    mn = min(px[0], px[1], px[2])
    mx = max(px[0], px[1], px[2])
    return mn > 100 and mx - mn < 45


try:
    ctx = get_task_context()
    lang = str(ctx.language)
    R["language"] = lang
    tm = robomme_plus.get_task_metadata()
    max_steps = 480
    try:
        max_steps = int(tm["max_steps"])
    except Exception as e:
        err("max_steps", e)
    R["max_steps"] = max_steps

    words = lang.lower().replace(",", " ").replace(".", " ").split()
    order = []
    for i in range(1, len(words)):
        if words[i] in ("cube", "cubes", "block") and words[i - 1] in COLORS:
            if words[i - 1] not in order:
                order.append(words[i - 1])
    n_buttons = 2 if ("both" in words or "two" in words or "buttons" in words) else 1
    R["order"] = order
    R["n_buttons_goal"] = n_buttons

    obs = get_observation()
    st = get_robot_state()
    base = st.base_frame
    EMB = st.embodiment
    fcam = obs.cameras["front"]
    K = [[float(v) for v in row] for row in fcam.intrinsics.tolist()]
    FX, FY, CX, CY = K[0][0], K[1][1], K[0][2], K[1][2]
    CP = [float(v) for v in fcam.camera_pose.position]
    RM = qrot(fcam.camera_pose.quaternion_wxyz)
    rgb0 = fcam.rgb.tolist()
    dep0 = fcam.depth_m.tolist()
    H = len(rgb0)
    W = len(rgb0[0])

    def backproject(u, v, dep):
        xc = (u - CX) * dep / FX
        yc = (v - CY) * dep / FY
        zc = dep
        return [CP[0] + RM[0][0] * xc + RM[0][1] * yc + RM[0][2] * zc,
                CP[1] + RM[1][0] * xc + RM[1][1] * yc + RM[1][2] * zc,
                CP[2] + RM[2][0] * xc + RM[2][1] * yc + RM[2][2] * zc]

    q0 = [float(v) for v in st.joint_positions["primary"]]
    quat0 = [float(v) for v in st.end_effector_poses["primary"].quaternion_wxyz]
    zax = qrot(quat0)
    down_quat = quat0 if zax[2][2] < -0.9 else [0.0, 1.0, 0.0, 0.0]
    hand_y = qrot(down_quat)
    FINGER_YAW = math.atan2(hand_y[1][1], hand_y[0][1])   # finger closing axis in base xy
    R["finger_yaw_deg"] = round(math.degrees(FINGER_YAW), 1)

    def rgb_color_ok(px, color):
        r, g, b = px[0], px[1], px[2]
        if color == "red":
            return r > 120 and r > g + 50 and r > b + 50
        if color == "green":
            return g > 90 and g > r + 30 and g > b + 30
        if color == "blue":
            return b > 90 and b > r + 30 and b > g + 20
        if color == "yellow":
            return r > 120 and g > 120 and b < min(r, g) - 50
        return True

    # ---- cubes named in the goal (t0, before the containers drop) ----
    cubes = {}
    for color in order:
        found = None
        try:
            s = segment_text("front", color + " cube")
            if s.ok:
                for c in s.segmentations[:5]:
                    if c.box_xyxy is None:
                        continue
                    b = [float(v) for v in c.box_xyxy]
                    if b[2] - b[0] > 40 or b[3] - b[1] > 40:
                        continue
                    u = int((b[0] + b[2]) / 2.0)
                    v = int((b[1] + b[3]) / 2.0)
                    if rgb_color_ok(rgb0[v][u], color):
                        p = backproject(u + 0.5, v + 0.5, float(dep0[v][u]))
                        found = {"px": [u, v], "xyz": [round(p[0], 4), round(p[1], 4), round(p[2], 4)],
                                 "score": round(float(c.score), 3)}
                        break
        except Exception as e:
            err("cube " + color, e)
        cubes[color] = found
    R["cubes"] = cubes

    # ---- buttons (3D gate: raised base plate, small footprint) ----
    buttons = []
    try:
        s = segment_text("front", "button")
        if s.ok:
            for c in s.segmentations[:5]:
                try:
                    pc = mask_to_point_cloud(c, "front", target_frame=base)
                    pts = pc.points.tolist()
                    if len(pts) < 10:
                        continue
                    n = len(pts)
                    xs = sorted([p[0] for p in pts])
                    ys = sorted([p[1] for p in pts])
                    zs = sorted([p[2] for p in pts])
                    xsp = xs[int(0.95 * (n - 1))] - xs[int(0.05 * (n - 1))]
                    ysp = ys[int(0.95 * (n - 1))] - ys[int(0.05 * (n - 1))]
                    zlo = zs[int(0.05 * (n - 1))]
                    zhi = zs[int(0.95 * (n - 1))]
                    info = {"xyz": [xs[n // 2], ys[n // 2]], "top": zhi, "zlo": zlo,
                            "score": round(float(c.score), 3), "span": [round(xsp, 3), round(ysp, 3)]}
                    if xsp > 0.08 or ysp > 0.08 or zlo < 0.010 or zhi - zlo > 0.06:
                        continue
                    dup = False
                    for bb in buttons:
                        if abs(bb["xyz"][0] - info["xyz"][0]) < 0.03 and abs(bb["xyz"][1] - info["xyz"][1]) < 0.03:
                            dup = True
                    if not dup:
                        buttons.append(info)
                except Exception as e:
                    err("button cand", e)
    except Exception as e:
        err("buttons", e)
    R["buttons"] = buttons

    # ---- metric top-face tracker ----
    STATIC = {}
    TR = {"init": False, "ztop": None, "last_motion": -1, "frozen": False, "init_tick": -1}
    tracks = []
    ARM = {"xy": None, "box": False}

    def cam_params(c):
        k = [[float(v) for v in row] for row in c.intrinsics.tolist()]
        return {"fx": k[0][0], "fy": k[1][1], "cx": k[0][2], "cy": k[1][2],
                "p": [float(v) for v in c.camera_pose.position], "R": qrot(c.camera_pose.quaternion_wxyz)}

    FRONT0 = cam_params(fcam)

    def white_points(rgb, dep, cp=None):
        if cp is None:
            cp = FRONT0
        cfx, cfy, ccx, ccy, cpp, crm = cp["fx"], cp["fy"], cp["cx"], cp["cy"], cp["p"], cp["R"]
        pts = []
        for v in range(0, H, STRIDE):
            row = rgb[v]
            drow = dep[v]
            for u in range(0, W, STRIDE):
                if not white(row[u]):
                    continue
                d = float(drow[u])
                if d < 0.05 or d > 3.0:
                    continue
                xc = (u + 0.5 - ccx) * d / cfx
                yc = (v + 0.5 - ccy) * d / cfy
                p = [cpp[0] + crm[0][0] * xc + crm[0][1] * yc + crm[0][2] * d,
                     cpp[1] + crm[1][0] * xc + crm[1][1] * yc + crm[1][2] * d,
                     cpp[2] + crm[2][0] * xc + crm[2][1] * yc + crm[2][2] * d]
                if p[2] < 0.02 or p[2] > 0.15:
                    continue
                if p[0] * p[0] + p[1] * p[1] < 0.2 * 0.2:
                    continue   # robot's own base/links near the base-frame origin
                near_b = False
                for bb in buttons:
                    if (p[0] - bb["xyz"][0]) ** 2 + (p[1] - bb["xyz"][1]) ** 2 < 0.03 * 0.03:
                        near_b = True
                if near_b:
                    continue
                if ARM["xy"] is not None:
                    if ARM["box"]:
                        # pressing: hand axis-aligned (fingers along base y) -> box footprint
                        if abs(p[0] - ARM["xy"][0]) < 0.06 and abs(p[1] - ARM["xy"][1]) < 0.12:
                            continue
                    elif (p[0] - ARM["xy"][0]) ** 2 + (p[1] - ARM["xy"][1]) ** 2 < 0.13 * 0.13:
                        continue
                if (int(math.floor(p[0] / 0.01)), int(math.floor(p[1] / 0.01))) in STATIC:
                    continue
                pts.append(p)
        return pts

    def all_points(ob):
        pts = []
        for cn in ("front", "wrist"):
            try:
                c = ob.cameras[cn]
                pts = pts + white_points(c.rgb.tolist(), c.depth_m.tolist(), cam_params(c))
            except Exception as e:
                err("pts " + cn, e)
        return pts

    # static tall white structure at t0 (no containers yet): xy cells holding white points
    # higher than 5 cm (cubes/buttons are ~3.5 cm), dilated by one 1-cm cell
    try:
        raw0 = all_points(obs)
        cells = {}
        for p in raw0:
            if p[2] > 0.05:
                cells[(int(math.floor(p[0] / 0.01)), int(math.floor(p[1] / 0.01)))] = True
        for k in cells:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    STATIC[(k[0] + dx, k[1] + dy)] = True
        R["static_cells"] = len(STATIC)
        R["white_t0_pts"] = len(raw0)
    except Exception as e:
        err("static map", e)

    def top_band(pts):
        zt = TR["ztop"]
        return [p for p in pts if zt - 0.016 <= p[2] <= zt + 0.02]

    def grid_components(pts, cell):
        occ = {}
        for p in pts:
            k = (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)))
            if k not in occ:
                occ[k] = []
            occ[k].append(p)
        seen = {}
        comps = []
        for k in occ:
            if k in seen:
                continue
            seen[k] = True
            stack = [k]
            comp = []
            while len(stack) > 0:
                cur = stack.pop()
                comp = comp + occ[cur]
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nk = (cur[0] + dx, cur[1] + dy)
                        if nk in occ and nk not in seen:
                            seen[nk] = True
                            stack.append(nk)
            comps.append(comp)
        return comps

    def kmeans_xy(pts, cents, iters):
        for it in range(iters):
            acc = [[0.0, 0.0, 0] for c in cents]
            for p in pts:
                bi = 0
                bd = None
                for i in range(len(cents)):
                    d = (p[0] - cents[i][0]) ** 2 + (p[1] - cents[i][1]) ** 2
                    if bd is None or d < bd:
                        bd = d
                        bi = i
                acc[bi][0] += p[0]
                acc[bi][1] += p[1]
                acc[bi][2] += 1
            for i in range(len(cents)):
                if acc[i][2] > 0:
                    cents[i] = [acc[i][0] / acc[i][2], acc[i][1] / acc[i][2]]
        return cents

    def init_tracks(pts):
        zs = sorted([p[2] for p in pts])
        TR["ztop"] = zs[int(0.93 * (len(zs) - 1))]
        top = top_band(pts)
        comps = [c for c in grid_components(top, 0.01) if len(c) >= 6]
        if len(comps) == 0:
            return False
        sizes = sorted([len(c) for c in comps])
        unit = sizes[len(sizes) // 2]
        for c in comps:
            k = max(1, int(round(len(c) / float(unit))))
            mx = sum([p[0] for p in c]) / len(c)
            my = sum([p[1] for p in c]) / len(c)
            if k == 1:
                cents = [[mx, my]]
            else:
                xs = sorted([p[0] for p in c])
                ys = sorted([p[1] for p in c])
                cents = []
                for i in range(k):
                    f = (i + 0.5) / k
                    if xs[-1] - xs[0] >= ys[-1] - ys[0]:
                        cents.append([xs[int(f * (len(xs) - 1))], my])
                    else:
                        cents.append([mx, ys[int(f * (len(ys) - 1))]])
                cents = kmeans_xy(c, cents, 8)
            for cc in cents:
                tracks.append({"id": len(tracks), "c": cc, "v": [0.0, 0.0], "n": 0, "lost": 0})
        R["ztop"] = round(TR["ztop"], 4)
        R["init_unit"] = unit
        R["init_tracks"] = [[t["id"], round(t["c"][0], 4), round(t["c"][1], 4)] for t in tracks]
        return True

    def update_tracks(top):
        pred = [[t["c"][0] + t["v"][0], t["c"][1] + t["v"][1]] for t in tracks]
        cents = [[p[0], p[1]] for p in pred]
        cnt = [0 for t in tracks]
        live = [t["lost"] < 3 for t in tracks]   # v05: lost tracks never capture points by k-means
        for it in range(2):
            acc = [[0.0, 0.0, 0] for t in tracks]
            for p in top:
                bi = -1
                bd = 0.032 * 0.032
                for i in range(len(cents)):
                    if not live[i]:
                        continue
                    d = (p[0] - cents[i][0]) ** 2 + (p[1] - cents[i][1]) ** 2
                    if d < bd:
                        bd = d
                        bi = i
                if bi >= 0:
                    acc[bi][0] += p[0]
                    acc[bi][1] += p[1]
                    acc[bi][2] += 1
            for i in range(len(tracks)):
                cnt[i] = acc[i][2]
                if acc[i][2] >= 5:
                    cents[i] = [acc[i][0] / acc[i][2], acc[i][1] / acc[i][2]]
                else:
                    cents[i] = pred[i]
        mv = 0.0
        for i in range(len(tracks)):
            t = tracks[i]
            nc = cents[i]
            dx = nc[0] - t["c"][0]
            dy = nc[1] - t["c"][1]
            dd = math.sqrt(dx * dx + dy * dy)
            if dd > 0.015:
                dx = dx * 0.015 / dd
                dy = dy * 0.015 / dd
            if cnt[i] >= 5:
                t["v"] = [0.5 * t["v"][0] + 0.5 * dx, 0.5 * t["v"][1] + 0.5 * dy]
                mv = max(mv, math.sqrt(dx * dx + dy * dy))
            else:
                t["v"] = [0.85 * t["v"][0], 0.85 * t["v"][1]]
                dx, dy = t["v"][0], t["v"][1]
            t["c"] = [t["c"][0] + dx, t["c"][1] + dy]
            t["n"] = cnt[i]
            t["lost"] = t["lost"] + 1 if cnt[i] < 5 else 0
        # merge prevention: two tracks within 3 cm -> the weaker one is lost
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                a = tracks[i]
                b = tracks[j]
                if (a["c"][0] - b["c"][0]) ** 2 + (a["c"][1] - b["c"][1]) ** 2 < 0.03 * 0.03:
                    weak = a if a["n"] < b["n"] else b
                    weak["lost"] = max(weak["lost"], 3)
                    weak["n"] = 0
                    TR["merges"] = TR.get("merges", 0) + 1
        # recovery: lost tracks (>= 3 ticks) re-attach to orphan clusters
        lost = [t for t in tracks if t["lost"] >= 3]
        if len(lost) > 0:
            active = [t for t in tracks if t["lost"] < 3]
            orphans = []
            for p in top:
                far = True
                for t in active:
                    if (p[0] - t["c"][0]) ** 2 + (p[1] - t["c"][1]) ** 2 < 0.04 * 0.04:
                        far = False
                        break
                if far:
                    orphans.append(p)
            clusters = []
            for comp in grid_components(orphans, 0.01):
                if len(comp) >= 15:
                    clusters.append([sum([p[0] for p in comp]) / len(comp), sum([p[1] for p in comp]) / len(comp)])
            pairs = []
            for li in range(len(lost)):
                for ci in range(len(clusters)):
                    d = math.sqrt((lost[li]["c"][0] - clusters[ci][0]) ** 2 + (lost[li]["c"][1] - clusters[ci][1]) ** 2)
                    if d < 0.15:
                        pairs.append([d, li, ci])
            pairs = sorted(pairs, key=lambda q: q[0])
            ul = set()
            uc = set()
            for pr in pairs:
                if pr[1] in ul or pr[2] in uc:
                    continue
                ul.add(pr[1])
                uc.add(pr[2])
                t = lost[pr[1]]
                t["c"] = [clusters[pr[2]][0], clusters[pr[2]][1]]
                t["v"] = [0.0, 0.0]
                t["lost"] = 0
                t["n"] = 5
                R["recoveries"].append([TICKS["n"], t["id"], round(pr[0], 3)])
        return mv

    def track_tick():
        try:
            ob = get_observation()
        except Exception as e:
            err("obs", e)
            return None
        pts = all_points(ob)
        rec = {"t": TICKS["n"], "w": len(pts)}
        if not TR["init"]:
            if len(pts) > 250:
                if init_tracks(pts):
                    TR["init"] = True
                    TR["init_tick"] = TICKS["n"]
                    TR["last_motion"] = TICKS["n"]
                    ev("tracks initialised: %d, ztop %.3f" % (len(tracks), TR["ztop"]))
        elif not TR["frozen"]:
            mv = update_tracks(top_band(pts))
            rec["mv"] = round(mv * 1000, 1)
            if mv > 0.0015:
                TR["last_motion"] = TICKS["n"]
        if TR["init"] and TICKS["n"] % 2 == 0:
            rec["tr"] = [[round(t["c"][0], 3), round(t["c"][1], 3), t["n"], t["lost"]] for t in tracks]
        R["trk_log"].append(rec)
        return pts

    # ---- motion helpers ----
    def joints_now():
        return [float(v) for v in get_robot_state().joint_positions["primary"]]

    def update_arm():
        try:
            p = [float(v) for v in get_robot_state().end_effector_poses["primary"].position]
            ARM["xy"] = [p[0], p[1]] if p[2] < 0.25 else None
        except Exception as e:
            err("arm", e)

    def step_hold(q, grip, track):
        r = step(RobotAction(arms={"primary": ArmCommand(mode="joint_position", target=list(q),
                                                         gripper_position=grip, embodiment=EMB)}))
        TICKS["n"] += 1
        if is_term(r):
            STATE["ended"] = True
            ev("episode terminal flag on step")
            return r
        if track:
            update_arm()
            track_tick()
        return r

    def step_to(qt, grip, min_n, track, vmax):
        qs = joints_now()
        md = 0.0
        for k in range(7):
            md = max(md, abs(qt[k] - qs[k]))
        n = max(min_n, int(math.ceil(md / vmax)))
        for i in range(1, n + 1):
            if STATE["ended"]:
                return
            a = i / float(n)
            step_hold([qs[k] + (qt[k] - qs[k]) * a for k in range(7)], grip, track)

    def ik(pos, quat, seed):
        res = solve_ik(Pose(position=list(pos), quaternion_wxyz=list(quat), frame=base), seed_joints=list(seed))
        if not res.ok:
            raise ValueError("IK failed for %s: %s" % (str([round(v, 3) for v in pos]), str(res.error)[:200]))
        return [float(v) for v in res.joint_positions]

    press_plan = []
    bl = sorted(buttons, key=lambda b: b["xyz"][1])
    for bb in bl[:n_buttons]:
        qp = ik([bb["xyz"][0], bb["xyz"][1], bb["top"] + 0.06], down_quat, q0)
        qd = ik([bb["xyz"][0], bb["xyz"][1], bb["top"] - 0.012], down_quat, qp)
        press_plan.append([qp, qd, bb])
    if len(press_plan) < n_buttons:
        ev("only %d buttons found" % len(press_plan))

    # ---- phase W: watch from home until PRESS_AT ----
    track_tick()
    while not STATE["ended"] and TICKS["n"] < PRESS_AT:
        step_hold(q0, 0.0, True)

    # ---- phase P: press buttons ----
    ARM["box"] = True
    for pp in press_plan:
        if STATE["ended"]:
            break
        step_to(pp[0], 0.0, 8, True, 0.10)
        step_to(pp[1], 0.0, 5, True, 0.05)
        for k in range(3):
            if not STATE["ended"]:
                step_hold(pp[1], 0.0, True)
        ev("pressed button at %.3f %.3f" % (pp[2]["xyz"][0], pp[2]["xyz"][1]))
        step_to(pp[0], 0.0, 4, True, 0.10)
    R["press_end"] = TICKS["n"]
    step_to(q0, 0.0, 10, True, 0.10)
    ARM["xy"] = None
    ARM["box"] = False
    t_after = TICKS["n"]
    while not STATE["ended"]:
        step_hold(q0, 0.0, True)
        if STATE["ended"]:
            break
        if TICKS["n"] - TR["last_motion"] >= QUIET_TICKS and TICKS["n"] > t_after + 10:
            ev("containers quiet")
            break
        if TICKS["n"] > max_steps - 420:
            ev("post-press watch cap")
            break
    R["final_tracks"] = [[t["id"], round(t["c"][0], 4), round(t["c"][1], 4), t["n"]] for t in tracks]

    # ---- map cubes -> container tracks at initialisation ----
    init_pos = {}
    for row in R.get("init_tracks", []):
        init_pos[row[0]] = [row[1], row[2]]
    assign = {}
    used = set()
    for color in order:
        cb = cubes.get(color)
        if cb is None:
            continue
        best = None
        for tid in init_pos:
            if tid in used:
                continue
            d = math.sqrt((init_pos[tid][0] - cb["xyz"][0]) ** 2 + (init_pos[tid][1] - cb["xyz"][1]) ** 2)
            if best is None or d < best[0]:
                best = [d, tid]
        if best is not None and best[0] < 0.04:
            assign[color] = best[1]
            used.add(best[1])
        R["assign_d_" + color] = None if best is None else round(best[0], 4)
    R["assign"] = assign

    # ---- phase G: pick each target container in order ----
    def container_now(tid):
        # refresh tracks for a few ticks with the arm at home, then measure this track
        pts = None
        for k in range(3):
            if not STATE["ended"]:
                step_hold(q0, 1.0, True)
        try:
            ob = get_observation()
            pts = all_points(ob)
        except Exception as e:
            err("obs pick", e)
            return None
        top = top_band(pts)
        me = None
        for t in tracks:
            if t["id"] == tid:
                me = t
        mine = []
        for p in top:
            dme = (p[0] - me["c"][0]) ** 2 + (p[1] - me["c"][1]) ** 2
            if dme > 0.032 * 0.032:
                continue
            ok = True
            for t in tracks:
                if t["id"] != tid and (p[0] - t["c"][0]) ** 2 + (p[1] - t["c"][1]) ** 2 < dme:
                    ok = False
                    break
            if ok:
                mine.append(p)
        if len(mine) < 8:
            return {"c": me["c"], "top": TR["ztop"], "yaw": FINGER_YAW, "n": len(mine)}
        cx = sum([p[0] for p in mine]) / len(mine)
        cy = sum([p[1] for p in mine]) / len(mine)
        zs = sorted([p[2] for p in mine])
        topz = zs[int(0.8 * (len(zs) - 1))]
        # 4-fold orientation of the square top face: corners are the farthest points
        s4 = 0.0
        c4 = 0.0
        for p in mine:
            dx = p[0] - cx
            dy = p[1] - cy
            r2 = dx * dx + dy * dy
            a = math.atan2(dy, dx)
            s4 += r2 * r2 * math.sin(4 * a)
            c4 += r2 * r2 * math.cos(4 * a)
        corner = math.atan2(s4, c4) / 4.0
        face = corner + math.pi / 4.0
        return {"c": [cx, cy], "top": topz, "yaw": face, "n": len(mine)}

    def yawed_quat(face_yaw):
        d = face_yaw - FINGER_YAW
        while d > math.pi / 4:
            d -= math.pi / 2
        while d < -math.pi / 4:
            d += math.pi / 2
        rz = [math.cos(d / 2), 0.0, 0.0, math.sin(d / 2)]
        return qmul(rz, down_quat), d

    R["picks"] = []
    TR["frozen"] = False
    for color in order:
        if STATE["ended"]:
            break
        if color not in assign:
            ev("no container for %s" % color)
            continue
        for attempt in range(2):
            if STATE["ended"]:
                break
            info = container_now(assign[color])
            if info is None:
                break
            gq, dyaw = yawed_quat(info["yaw"])
            cx, cy, top = info["c"][0], info["c"][1], info["top"]
            rec = {"color": color, "track": assign[color], "attempt": attempt, "c": [round(cx, 4), round(cy, 4)],
                   "top": round(top, 4), "n": info["n"], "dyaw_deg": round(math.degrees(dyaw), 1)}
            R["picks"].append(rec)
            gz = max(top - 0.022, 0.012)
            try:
                qa = ik([cx, cy, top + 0.07], gq, q0)
                qg = ik([cx, cy, gz], gq, qa)
                ql = ik([cx, cy, top + 0.12], gq, qg)
            except Exception as e:
                err("pick ik", e)
                gq = down_quat
                qa = ik([cx, cy, top + 0.07], gq, q0)
                qg = ik([cx, cy, gz], gq, qa)
                ql = ik([cx, cy, top + 0.12], gq, qg)
            ev("pick %s try %d at %.3f %.3f top %.3f dyaw %.0f" % (color, attempt, cx, cy, top, math.degrees(dyaw)))
            TR["frozen"] = True
            step_to(qa, 1.0, 10, False, 0.08)
            step_to(qg, 1.0, 6, False, 0.04)
            for k in range(10):
                if not STATE["ended"]:
                    step_hold(qg, 0.0, False)
            g = 0.0
            try:
                g = float(get_robot_state().gripper_positions["primary"])
            except Exception as e:
                err("grip", e)
            rec["grip_closed"] = round(g, 3)
            if g < 0.03 and attempt == 0:
                ev("grasp closed on nothing; re-observe and retry")
                step_to(qa, 1.0, 6, False, 0.06)
                step_to(q0, 1.0, 10, False, 0.08)
                TR["frozen"] = False
                continue
            step_to(ql, 0.0, 8, False, 0.04)
            for k in range(8):
                if not STATE["ended"]:
                    step_hold(ql, 0.0, False)
            if STATE["ended"]:
                break
            step_to(qg, 0.0, 8, False, 0.04)
            for k in range(5):
                if not STATE["ended"]:
                    step_hold(qg, 1.0, False)
            step_to(qa, 1.0, 6, False, 0.06)
            step_to(q0, 1.0, 10, False, 0.08)
            TR["frozen"] = False
            ev("placed back %s" % color)
            break
except Exception as e:
    err("main", e)

R["ticks"] = TICKS["n"]
result = R
