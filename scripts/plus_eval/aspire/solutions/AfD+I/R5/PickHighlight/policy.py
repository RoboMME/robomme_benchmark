# v04_period -- PickHighlight (robomme_plus task 6)
# v04 vs v03 (ep0: truncated at 2096 after only 1 pick): one 12-tick false window (the arm releasing
# a cube) made the period estimate min(diffs)=69 instead of ~140, so every pick waited its 220-tick
# limit. v04: windows shorter than 22 ticks are noise; period = median start spacing; descents and
# lifts run faster; a lifted cube is held through a window unless >=25 free ticks remain; track
# workspace = targets/button box +-15 cm (v03 also tracked the robot base).
# v03 vs v02 (ep0 run: no failure terminal, 3/5 grasp attempts, 1 closed on air, 2 targets given up):
#  - track on cube TOP faces only (z >= table+2.8 cm) with a 2.6 cm gate: v02's full-height band let
#    one track absorb two neighbouring cubes (n=70) and the grasp went between them;
#  - shuffle detection ignores tracks near the arm / occluded tracks and needs 3 consecutive ticks
#    (v02 aborted 12 approaches on arm-induced centroid jitter);
#  - at plan time the tracked cube is re-segmented with segment_points at its own top pixel and the
#    grasp centre/yaw/top come from that single-cube top face;
#  - Cartesian moves are split into <=2.5 cm IK waypoints seeded in a chain (v02 descents jumped
#    up to 1.2 rad in joint space for an 8 cm vertical move).
# Root cause of v01's failure (environment terminated, success=false, while lifting the 2nd cube):
# the cubes are SHUFFLED in pairs during periodic windows (~40 ticks, every ~100-141 ticks; the
# first begins when the highlight discs vanish). v01 grasped the position where a highlighted cube
# had been at tick 13; offline tracking of its own front video shows a non-highlighted cube had
# been swapped into that spot, so lifting it ended the episode.
# v02 keeps v01's highlight detection, then tracks EVERY above-table blob in robot_base x/y
# every control tick from front depth (calibrated camera pose + intrinsics) with competitive
# nearest-centroid assignment; targets keep identity through swaps. All motion is executed
# tick-by-tick (solve_ik seeded from the current joints + joint interpolation via step) so the
# tracker never goes blind. Picks start only while nothing moves and a window is not imminent;
# the arm parks high (no occlusion) while it waits. Grasp pose comes from the tracked top face.
import math

report = {"version": "v04_period", "events": [], "errors": []}


def log(name, value):
    report["events"].append([name, value])


def now_tick():
    try:
        return int(round(float(get_robot_state().timestamp_s) * 20.0))
    except Exception:
        return -1


def is_terminal(res):
    flag = False
    try:
        if res.terminated:
            flag = True
    except Exception:
        pass
    try:
        if res.truncated:
            flag = True
    except Exception:
        pass
    return flag


def Done(label):
    return Exception("TERMINAL:" + str(label))


def check(res, label):
    entry = {"label": label, "tick": -1}
    try:
        entry["ok"] = bool(res.ok)
    except Exception:
        entry["ok"] = None
    try:
        entry["status"] = str(res.status)
    except Exception:
        pass
    try:
        entry["steps"] = int(res.steps_executed)
    except Exception:
        pass
    term = is_terminal(res)
    entry["terminal"] = term
    if not term:
        entry["tick"] = now_tick()
    log("motion", entry)
    if term:
        raise Done(label)
    return entry


def rgb_rows(camera):
    obs = get_observation()
    arr = obs.cameras[camera].rgb
    try:
        return arr.tolist()
    except Exception:
        rows = []
        for y in range(len(arr)):
            row = []
            for x in range(len(arr[y])):
                px = arr[y][x]
                row.append([int(px[0]), int(px[1]), int(px[2])])
            rows.append(row)
        return rows


# >>> highlight helpers
def white_set(rows, y0):
    out = set()
    h = len(rows)
    for y in range(y0, h):
        row = rows[y]
        for x in range(len(row)):
            p = row[x]
            lo = min(p[0], p[1], p[2])
            if lo > 200 and max(p[0], p[1], p[2]) - lo < 35:
                out.add((x, y))
    return out


def components(pixels, diagonal=True):
    remaining = set(pixels)
    comps = []
    if diagonal:
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            cx, cy = stack.pop()
            for dxy in nbrs:
                q = (cx + dxy[0], cy + dxy[1])
                if q in remaining:
                    remaining.discard(q)
                    stack.append(q)
                    comp.append(q)
        comps.append(comp)
    return sorted(comps, key=len, reverse=True)


def highlight_points(diff, rows):
    """Cube pixels enclosed by a NEW white disc: per disc blob, pixels with blob pixels
    left+right in their row and above+below in their column; one point per enclosed component."""
    found = []
    for comp in components(diff):
        if len(comp) < 30:
            break
        cset = set(comp)
        rmin = {}
        rmax = {}
        cmin = {}
        cmax = {}
        for (x, y) in comp:
            rmin[y] = min(rmin.get(y, x), x)
            rmax[y] = max(rmax.get(y, x), x)
            cmin[x] = min(cmin.get(x, y), y)
            cmax[x] = max(cmax.get(x, y), y)
        inside = set()
        for y in rmin:
            for x in range(rmin[y] + 1, rmax[y]):
                if (x, y) in cset or (x, y) in diff:
                    continue
                if x in cmax and y < cmax[x]:
                    inside.add((x, y))
        for part in components(inside, diagonal=False):
            if len(part) < 8:
                break
            u = sum(p[0] for p in part) / float(len(part))
            v = sum(p[1] for p in part) / float(len(part))
            quads = [0, 0, 0, 0]
            for (x, y) in comp:
                if (x - u) ** 2 + (y - v) ** 2 <= 256.0:
                    quads[(1 if x >= u else 0) + (2 if y >= v else 0)] += 1
            nq = len([q for q in quads if q >= 3])
            # representative pixel: interior pixel closest to the centroid (always on the part)
            best = None
            bd = None
            for (x, y) in part:
                dd = (x - u) ** 2 + (y - v) ** 2
                if bd is None or dd < bd:
                    bd = dd
                    best = (x, y)
            found.append([u, v, len(part), nq, best[0], best[1], len(comp)])
    return found
# <<< highlight helpers


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def cloud_points(seg, camera):
    try:
        pc = mask_to_point_cloud(seg, camera, target_frame=base_frame)
        pts = [[float(p[0]), float(p[1]), float(p[2])] for p in pc.points]
        return pts
    except Exception as exc:
        report["errors"].append("cloud " + str(exc)[:200])
        return []


def describe_cloud(pts):
    if len(pts) < 5:
        return None
    zs = [p[2] for p in pts]
    z_low = quantile(zs, 0.05)
    z_top = quantile(zs, 0.92)
    body = [p for p in pts if p[2] > z_low + 0.008]
    if len(body) < 5:
        body = pts
    cx = quantile([p[0] for p in body], 0.5)
    cy = quantile([p[1] for p in body], 0.5)
    # footprint yaw: angle minimising the bounding square of the body xy points
    best_a = 0.0
    best_s = None
    for k in range(18):
        a = math.radians(5.0 * k)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [p[0] * ca + p[1] * sa for p in body]
        vs = [-p[0] * sa + p[1] * ca for p in body]
        s = max(quantile(us, 0.95) - quantile(us, 0.05), quantile(vs, 0.95) - quantile(vs, 0.05))
        if best_s is None or s < best_s:
            best_s = s
            best_a = a
    return {"x": cx, "y": cy, "z_top": z_top, "z_low": z_low, "yaw": best_a,
            "side": best_s, "n": len(pts)}


def box_of(seg):
    try:
        b = seg.box_xyxy
        if b is None:
            return None
        return [float(v) for v in b]
    except Exception:
        return None


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def qrot(q, v):
    w, x, y, z = q
    vq = [0.0, v[0], v[1], v[2]]
    conj = [w, -x, -y, -z]
    r = qmul(qmul(q, vq), conj)
    return [r[1], r[2], r[3]]


def yaw_quat(base_q, theta):
    qz = [math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)]
    q = qmul(qz, base_q)
    n = math.sqrt(sum(c * c for c in q))
    return [c / n for c in q]


def wrap_half_pi(a):
    while a > math.pi / 4.0:
        a -= math.pi / 2.0
    while a <= -math.pi / 4.0:
        a += math.pi / 2.0
    return a




def grasp_yaw(cube, others):
    # two face-aligned jaw directions; choose the one whose jaws are furthest from neighbours
    options = []
    for extra in (0.0, math.pi / 2.0):
        jaw_dir = cube["yaw"] + extra
        theta = wrap_half_pi(jaw_dir - jaw0)
        # the gripper's jaw direction after rotating by theta (mod pi)
        d = [math.cos(jaw0 + theta), math.sin(jaw0 + theta)]
        clear = 1.0
        for o in others:
            dx = o["x"] - cube["x"]
            dy = o["y"] - cube["y"]
            along = abs(dx * d[0] + dy * d[1])
            across = abs(-dx * d[1] + dy * d[0])
            if across < 0.04:
                clear = min(clear, along)
        options.append([clear, -abs(theta), theta])
    options.sort()
    return options[-1][2], options[-1][0]



def seg_point_cloud(u, v):
    sp = segment_points("front", [[int(u), int(v)]])
    if not sp.ok or len(sp.segmentations) == 0:
        return None
    return describe_cloud(cloud_points(sp.segmentations[0], "front"))


def is_cube(info):
    if info is None:
        return False
    h = info["z_top"] - info["z_low"]
    return 0.012 <= h <= 0.08 and info["side"] < 0.07


base_frame = get_robot_state().base_frame


# ---------------------------------------------------------------- v02: geometry + tracking
def quat_mat(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def to_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        return [[float(v) for v in row] for row in arr]


def setup_geometry(obs):
    cam = obs.cameras["front"]
    K = to_rows(cam.intrinsics)
    pose = cam.camera_pose
    pos = [float(c) for c in pose.position]
    R = quat_mat([float(c) for c in pose.quaternion_wxyz])
    dep = to_rows(cam.depth_m)
    H = len(dep)
    W = len(dep[0])
    fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
    models = {}
    for mode in ("zdepth", "raydist"):
        RX = []
        RY = []
        RZ = []
        for v in range(H):
            rx = []
            ry = []
            rz = []
            for u in range(W):
                xn = (u - cx) / fx
                a = [xn, (v - cy) / fy, 1.0]
                if mode == "raydist":
                    nrm = math.sqrt(a[0] * a[0] + a[1] * a[1] + 1.0)
                    a = [a[0] / nrm, a[1] / nrm, a[2] / nrm]
                rx.append(R[0][0] * a[0] + R[0][1] * a[1] + R[0][2] * a[2])
                ry.append(R[1][0] * a[0] + R[1][1] * a[1] + R[1][2] * a[2])
                rz.append(R[2][0] * a[0] + R[2][1] * a[1] + R[2][2] * a[2])
            RX.append(rx)
            RY.append(ry)
            RZ.append(rz)
        zs = []
        for v in range(H // 2, H, 6):
            for u in range(0, W, 6):
                d = dep[v][u]
                if d > 0:
                    zs.append(pos[2] + d * RZ[v][u])
        spread = quantile(zs, 0.75) - quantile(zs, 0.25)
        models[mode] = {"RX": RX, "RY": RY, "RZ": RZ, "spread": spread, "tz": quantile(zs, 0.5)}
    mode = "zdepth" if models["zdepth"]["spread"] <= models["raydist"]["spread"] else "raydist"
    m = models[mode]
    report["depth_model"] = {"mode": mode, "spread_z": round(models["zdepth"]["spread"], 5),
                             "spread_r": round(models["raydist"]["spread"], 5), "table_z": round(m["tz"], 4)}
    return {"c": pos, "RX": m["RX"], "RY": m["RY"], "RZ": m["RZ"], "tz": m["tz"], "W": W, "H": H,
            "roi": [0, 0, W, H], "ws": [-10.0, -10.0, 10.0, 10.0]}


def front_depth():
    if obs_mode[0] == 0:
        try:
            obs = get_observation(camera_names=["front"])
            return to_rows(obs.cameras["front"].depth_m)
        except Exception as exc:
            obs_mode[0] = 1
            report["errors"].append("obs_filter " + str(exc)[:120])
    obs = get_observation()
    return to_rows(obs.cameras["front"].depth_m)


def fg_points(dep, excl):
    # points above the table in the cube band, in robot_base, from the front depth image
    c = G["c"]
    zlo = G["tz"] + 0.028
    zhi = G["tz"] + 0.065
    roi = G["roi"]
    pts = []
    cx0, cy0, cz0 = c[0], c[1], c[2]
    for v in range(roi[1], roi[3]):
        row = dep[v]
        RXv = G["RX"][v]
        RYv = G["RY"][v]
        RZv = G["RZ"][v]
        for u in range(roi[0], roi[2]):
            d = row[u]
            if d <= 0.0:
                continue
            z = cz0 + d * RZv[u]
            if z > zlo and z < zhi:
                x = cx0 + d * RXv[u]
                y = cy0 + d * RYv[u]
                if x < G["ws"][0] or x > G["ws"][2] or y < G["ws"][1] or y > G["ws"][3]:
                    continue
                if excl is not None:
                    if (x - excl[0]) ** 2 + (y - excl[1]) ** 2 < excl[2] * excl[2]:
                        continue
                pts.append([x, y, z, u, v])
    return pts


def seed_tracks(pts, seeds):
    tracks = []
    for s in seeds:
        tracks.append({"x": s[0], "y": s[1], "vx": 0.0, "vy": 0.0, "n0": 0, "n": 0, "target": s[2],
                       "done": False, "held": False, "miss": 0, "speed": 0.0, "pts": [], "id": len(tracks),
                       "tag": s[3]})
    rad = 0.022
    covered = [False] * len(pts)
    for i in range(len(pts)):
        p = pts[i]
        for t in tracks:
            if (p[0] - t["x"]) ** 2 + (p[1] - t["y"]) ** 2 < rad * rad:
                covered[i] = True
                break
    for rep in range(40):
        first = -1
        for i in range(len(pts)):
            if not covered[i]:
                first = i
                break
        if first < 0:
            break
        cxy = [pts[first][0], pts[first][1]]
        near = []
        for it in range(4):
            near = [p for k, p in enumerate(pts) if not covered[k] and
                    (p[0] - cxy[0]) ** 2 + (p[1] - cxy[1]) ** 2 < rad * rad]
            if not near:
                break
            cxy = [sum(p[0] for p in near) / len(near), sum(p[1] for p in near) / len(near)]
        for k in range(len(pts)):
            p = pts[k]
            if (p[0] - cxy[0]) ** 2 + (p[1] - cxy[1]) ** 2 < rad * rad:
                covered[k] = True
        covered[first] = True
        if len(near) >= 4:
            tracks.append({"x": cxy[0], "y": cxy[1], "vx": 0.0, "vy": 0.0, "n0": 0, "n": 0,
                           "target": False, "done": False, "held": False, "miss": 0, "speed": 0.0,
                           "pts": [], "id": len(tracks), "tag": "other"})
    # k-means refinement (targets keep their identity because they were seeded first)
    for it in range(4):
        groups = assign(pts, tracks, 0.026, False)
        for k in range(len(tracks)):
            g = groups[k]
            if len(g) >= 4:
                tracks[k]["x"] = sum(p[0] for p in g) / len(g)
                tracks[k]["y"] = sum(p[1] for p in g) / len(g)
                tracks[k]["n0"] = len(g)
                tracks[k]["n"] = len(g)
                tracks[k]["pts"] = g
    return tracks


def assign(pts, tracks, radius, predict):
    groups = [[] for t in tracks]
    cents = []
    for t in tracks:
        if predict:
            cents.append([t["x"] + t["vx"], t["y"] + t["vy"], t["held"]])
        else:
            cents.append([t["x"], t["y"], t["held"]])
    r2 = radius * radius
    for p in pts:
        best = -1
        bd = r2
        for k in range(len(cents)):
            c = cents[k]
            if c[2]:
                continue
            d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2
            if d < bd:
                bd = d
                best = k
        if best >= 0:
            groups[best].append(p)
    return groups


def update_tracks(pts, ee):
    groups = assign(pts, tracks, 0.026, True)
    fast = 0
    for k in range(len(tracks)):
        t = tracks[k]
        if t["held"]:
            continue
        g = groups[k]
        need = max(4, int(0.35 * max(t["n0"], 1)))
        if len(g) >= need:
            nx = sum(p[0] for p in g) / len(g)
            ny = sum(p[1] for p in g) / len(g)
            dx = nx - t["x"]
            dy = ny - t["y"]
            t["vx"] = 0.5 * t["vx"] + 0.5 * dx
            t["vy"] = 0.5 * t["vy"] + 0.5 * dy
            t["x"] = nx
            t["y"] = ny
            t["n"] = len(g)
            t["pts"] = g
            t["miss"] = 0
            sp = math.sqrt(dx * dx + dy * dy)
            t["speed"] = 0.5 * t["speed"] + 0.5 * sp
            if len(g) >= 0.7 * t["n0"]:
                t["n0"] = int(0.9 * t["n0"] + 0.1 * len(g) + 0.5)
        else:
            t["x"] += 0.5 * t["vx"]
            t["y"] += 0.5 * t["vy"]
            t["vx"] *= 0.5
            t["vy"] *= 0.5
            t["miss"] += 1
            t["speed"] *= 0.5
        near_arm = ee[2] < G["tz"] + 0.25 and (t["x"] - ee[0]) ** 2 + (t["y"] - ee[1]) ** 2 < 0.12 * 0.12
        if t["tag"] != "button" and t["speed"] > 0.0015 and t["miss"] == 0 and not near_arm:
            if len(g) >= 0.6 * max(t["n0"], 1):
                fast += 1
    return fast


def observe(tick_now):
    # one tracking update per control tick
    st = get_robot_state()
    ee = [float(c) for c in st.end_effector_poses["primary"].position]
    excl = None
    if ee[2] < G["tz"] + 0.11:
        excl = [ee[0], ee[1], 0.045]
    for t in tracks:
        if t["held"]:
            excl = [ee[0], ee[1], 0.05]
    pts = fg_points(front_depth(), excl)
    fast = update_tracks(pts, ee)
    if fast >= 2:
        sched["fast_run"] += 1
    else:
        sched["fast_run"] = 0
    moving = sched["fast_run"] >= 3 or (sched["moving"] and fast >= 1)
    if moving:
        sched["last_move"] = tick_now
        if not sched["moving"]:
            sched["moving"] = True
            sched["starts"].append(tick_now)
        sched["still"] = 0
    else:
        sched["still"] += 1
        if sched["moving"] and sched["still"] >= 4:
            sched["moving"] = False
            if sched["starts"] and tick_now - 4 - sched["starts"][-1] < 22:
                sched["noise"].append(sched["starts"].pop())
            else:
                sched["ends"].append(tick_now - 4)
                log("window", [sched["starts"][-1] if sched["starts"] else -1, tick_now - 4,
                               [[t["id"], round(t["x"], 3), round(t["y"], 3), t["miss"]] for t in tracks if t["target"]]])
    return st


def period_est():
    S = sched["starts"]
    if len(S) >= 2:
        diffs = sorted([S[i + 1] - S[i] for i in range(len(S) - 1) if S[i + 1] - S[i] > 60])
        if diffs:
            return diffs[(len(diffs) - 1) // 2]
    return 95


def free_ticks(tick_now):
    # ticks until the next predicted shuffle window (0 while one is active)
    if sched["moving"]:
        return 0
    S = sched["starts"]
    if not S:
        return 10 ** 6
    nxt = S[-1] + period_est()
    while nxt + 60 < tick_now:
        nxt += period_est()
    return nxt - tick_now


def joints_now():
    st = get_robot_state()
    return [float(c) for c in st.joint_positions["primary"]], st


def tick(target, grip):
    reply = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=target, gripper_position=grip, embodiment=EMB)}))
    if is_terminal(reply):
        raise Done("step:" + cur_label[0])
    clock[0] += 1
    observe(clock[0])
    return reply


def solve(pos, quat, seed):
    target = Pose([float(pos[0]), float(pos[1]), float(pos[2])], quat, base_frame)
    best = None
    for s in (seed, home_joints):
        try:
            ik = solve_ik(target, seed_joints=s)
        except Exception as exc:
            report["errors"].append("ik " + str(exc)[:120])
            continue
        if not ik.ok:
            continue
        q = [float(c) for c in ik.joint_positions]
        jump = max(abs(q[i] - seed[i]) for i in range(7))
        if best is None or jump < best[1]:
            best = [q, jump]
        if jump < 1.2:
            break
    return best


def go(pos, quat, grip, label, speed=0.05, settle=15, tol=0.006, abort_moving=False):
    cur_label[0] = label
    q0, st = joints_now()
    want = [pos[0] + corr[0], pos[1] + corr[1], pos[2] + corr[2]]
    ee0 = [float(c) for c in st.end_effector_poses["primary"].position]
    dist = math.sqrt(sum((want[i] - ee0[i]) ** 2 for i in range(3)))
    nseg = 1
    if dist < 0.2:
        nseg = int(dist / 0.025) + 1
    chain = []
    seed = q0
    jump_max = 0.0
    for k in range(1, nseg + 1):
        f = float(k) / nseg
        wp = [ee0[i] + (want[i] - ee0[i]) * f for i in range(3)]
        if k == nseg:
            wp = want
        sol = solve(wp, quat, seed)
        if sol is None:
            log("ik_fail", [label, k])
            return "ik_fail"
        chain.append(sol[0])
        jump_max = max(jump_max, sol[1])
        seed = sol[0]
    q1 = chain[-1]
    t0 = clock[0]
    prev = q0
    for qk in chain:
        n = int(max(abs(qk[i] - prev[i]) for i in range(7)) / speed) + 1
        for i in range(1, n + 1):
            f = float(i) / n
            tick([prev[j] + (qk[j] - prev[j]) * f for j in range(7)], grip)
            if abort_moving and sched["moving"]:
                log("abort", [label, clock[0]])
                return "abort"
        prev = qk
    sol = [q1, jump_max]
    for s in range(settle):
        qm, st = joints_now()
        if max(abs(qm[j] - q1[j]) for j in range(7)) < tol:
            break
        tick(q1, grip)
        if abort_moving and sched["moving"]:
            log("abort", [label, clock[0]])
            return "abort"
    ee = [float(c) for c in get_robot_state().end_effector_poses["primary"].position]
    err = math.sqrt(sum((ee[i] - pos[i]) ** 2 for i in range(3)))
    log("go", [label, clock[0] - t0, round(err, 4), round(sol[1], 2)])
    if not corr_done[0]:
        corr_done[0] = True
        horiz = math.sqrt((ee[0] - pos[0]) ** 2 + (ee[1] - pos[1]) ** 2)
        if err > 0.05 and horiz < 0.02:
            for i in range(3):
                corr[i] += pos[i] - ee[i]
            log("tool_offset_correction", [round(c, 4) for c in corr])
            return go(pos, quat, grip, label + ":redo", speed, settle, tol, abort_moving)
    return "ok"


def hold(grip, n):
    q, st = joints_now()
    for i in range(n):
        tick(q, grip)


def gripper_to(grip, n, label):
    cur_label[0] = label
    hold(grip, n)
    g = float(get_robot_state().gripper_positions["primary"])
    log(label, [clock[0], round(g, 3)])
    return g


def wait_until(cond_free, label, limit):
    # park: hold still and keep tracking until not moving and enough free ticks
    cur_label[0] = label
    q, st = joints_now()
    w = 0
    while w < limit:
        if (not sched["moving"]) and sched["still"] >= 3 and free_ticks(clock[0]) >= cond_free:
            break
        tick(q, grip_state[0])
        w += 1
    if w > 0:
        log("wait", [label, w, clock[0]])
    return w


def track_cube(t):
    # cube description from the tracked points: top face -> centre, top height, yaw
    pts = t["pts"]
    if len(pts) < 4:
        return None
    ztop = quantile([p[2] for p in pts], 0.9)
    top = [p for p in pts if p[2] > ztop - 0.006]
    if len(top) < 4:
        top = pts
    info = describe_cloud([[p[0], p[1], p[2]] for p in top])
    if info is None:
        return None
    info["z_top"] = ztop
    info["z_low"] = G["tz"]
    return info


def press(label):
    if button is None:
        log("press_skipped", label)
        return
    bx, by, bz = button["x"], button["y"], button["z_top"]
    grip_state[0] = 0.0
    gripper_to(0.0, 8, label + ":close")
    go([bx, by, bz + 0.07], down_q, 0.0, label + ":above", speed=0.06)
    go([bx, by, bz + 0.02], down_q, 0.0, label + ":approach", speed=0.03, settle=8)
    go([bx, by, bz - 0.015], down_q, 0.0, label + ":press", speed=0.02, settle=6)
    hold(0.0, 4)
    go([bx, by, bz + 0.08], down_q, 0.0, label + ":retreat", speed=0.05, settle=6)
    log(label + "_tick", clock[0])


def nearest_other(px, py, me):
    best = 1.0
    for o in tracks:
        if o is me:
            continue
        best = min(best, math.sqrt((o["x"] - px) ** 2 + (o["y"] - py) ** 2))
    return best


def free_spot(x, y, me):
    # put the cube back where it was unless a shuffled cube now occupies that spot
    if nearest_other(x, y, me) >= 0.032:
        return [x, y]
    xs = [o["x"] for o in tracks]
    ys = [o["y"] for o in tracks]
    cand = [x, y, nearest_other(x, y, me)]
    for r in (0.06, 0.08, 0.1):
        for k in range(12):
            a = k * math.pi / 6.0
            px = x + r * math.cos(a)
            py = y + r * math.sin(a)
            if px < min(xs) - 0.02 or px > max(xs) + 0.02 or py < min(ys) - 0.02 or py > max(ys) + 0.02:
                continue
            dn = nearest_other(px, py, me)
            if dn > cand[2] + 0.005:
                cand = [px, py, dn]
        if cand[2] >= 0.05:
            break
    return [cand[0], cand[1]]


def sam_refine(t):
    # re-segment the tracked cube at its own top pixel; single-cube top face -> centre, yaw, top
    pts = t["pts"]
    if len(pts) < 3:
        return None
    best = None
    bd = None
    for p in pts:
        d = (p[0] - t["x"]) ** 2 + (p[1] - t["y"]) ** 2
        if bd is None or d < bd:
            bd = d
            best = p
    try:
        sp = segment_points("front", [[int(best[3]), int(best[4])]])
    except Exception as exc:
        report["errors"].append("sam " + str(exc)[:120])
        return None
    if not sp.ok or len(sp.segmentations) == 0:
        return None
    cp = cloud_points(sp.segmentations[0], "front")
    if len(cp) < 6:
        return None
    ztop = quantile([p[2] for p in cp], 0.9)
    top = [p for p in cp if p[2] > ztop - 0.006]
    info = describe_cloud(top) if len(top) >= 5 else None
    full = describe_cloud(cp)
    if info is None or full is None:
        return None
    if ztop - G["tz"] < 0.015 or ztop - G["tz"] > 0.08 or full["side"] > 0.07:
        return None
    if math.sqrt((info["x"] - t["x"]) ** 2 + (info["y"] - t["y"]) ** 2) > 0.03:
        return None
    info["z_top"] = ztop
    info["z_low"] = G["tz"]
    return info


def pick(t, idx):
    label = "cube%d" % idx
    for attempt in range(4):
        wait_until(45, label + ":wait", 220)
        info = track_cube(t)
        if info is None:
            log(label + "_lost", [round(t["x"], 3), round(t["y"], 3), t["miss"]])
            return False
        ref = sam_refine(t)
        if ref is not None:
            log(label + "_sam", [round(ref["x"] - info["x"], 4), round(ref["y"] - info["y"], 4),
                                 round(math.degrees(ref["yaw"]), 1), round(math.degrees(info["yaw"]), 1), t["n"]])
            info = ref
        else:
            log(label + "_sam", "none")
        others = [{"x": o["x"], "y": o["y"]} for o in tracks if o is not t and o["tag"] != "button"]
        theta, clear = grasp_yaw(info, others)
        q = yaw_quat(down_q, theta)
        grip_state[0] = 1.0
        r = go([info["x"], info["y"], info["z_top"] + 0.08], q, 1.0, label + ":hover", speed=0.06,
               abort_moving=True)
        if r == "abort":
            go([t["x"], t["y"], park_z], down_q, 1.0, label + ":abort_up", speed=0.06)
            continue
        if sched["moving"] or free_ticks(clock[0]) < 25:
            continue
        info2 = track_cube(t)
        if info2 is not None:
            shift = math.sqrt((info2["x"] - info["x"]) ** 2 + (info2["y"] - info["y"]) ** 2)
            if shift > 0.012:
                info = info2
                theta, clear = grasp_yaw(info, others)
                q = yaw_quat(down_q, theta)
        height = max(0.02, info["z_top"] - G["tz"])
        grasp_z = info["z_top"] - 0.5 * height
        log(label + "_plan", {"tick": clock[0], "xy": [round(info["x"], 4), round(info["y"], 4)],
                              "z_top": round(info["z_top"], 4), "grasp_z": round(grasp_z, 4),
                              "theta_deg": round(math.degrees(theta), 1), "clear": round(clear, 3),
                              "free": free_ticks(clock[0]), "n": t["n"]})
        r = go([info["x"], info["y"], grasp_z], q, 1.0, label + ":descend", speed=0.05, settle=6,
               tol=0.003, abort_moving=True)
        if r == "abort":
            go([info["x"], info["y"], info["z_top"] + 0.1], q, 1.0, label + ":abort_up", speed=0.05)
            continue
        t["held"] = True
        grip_state[0] = 0.0
        g = gripper_to(0.0, 8, label + ":close")
        go([info["x"], info["y"], grasp_z + 0.06], q, 0.0, label + ":lift", speed=0.05, settle=4)
        g2 = float(get_robot_state().gripper_positions["primary"])
        log(label + "_grip", [round(g, 3), round(g2, 3)])
        wait_until(25, label + ":hold", 160)
        spot = free_spot(info["x"], info["y"], t)
        if spot[0] != info["x"] or spot[1] != info["y"]:
            log(label + "_respot", [round(spot[0], 4), round(spot[1], 4)])
            go([spot[0], spot[1], grasp_z + 0.06], q, 0.0, label + ":shift", speed=0.04, settle=6)
            info["x"] = spot[0]
            info["y"] = spot[1]
        go([info["x"], info["y"], grasp_z + 0.008], q, 0.0, label + ":lower", speed=0.035, settle=6)
        grip_state[0] = 1.0
        gripper_to(1.0, 6, label + ":release")
        t["held"] = False
        t["done"] = True
        t["vx"] = 0.0
        t["vy"] = 0.0
        t["x"] = info["x"]
        t["y"] = info["y"]
        go([info["x"], info["y"], park_z], q, 1.0, label + ":up", speed=0.06, settle=4, tol=0.01)
        return True
    log(label + "_gave_up", clock[0])
    return False


base_frame = get_robot_state().base_frame
button = None
tracks = []
sched = {"moving": False, "noise": [], "fast_run": 0, "still": 0, "starts": [], "ends": [], "last_move": -1}
clock = [0]
cur_label = ["start"]
grip_state = [1.0]
corr = [0.0, 0.0, 0.0]
corr_done = [False]
obs_mode = [0]
G = None
try:
    ctx = get_task_context()
    report["language"] = ctx.language
    try:
        meta = robomme_plus.get_task_metadata()
        report["metadata"] = dict(meta)
    except Exception as exc:
        report["errors"].append("meta " + str(exc)[:200])
    try:
        video = robomme_plus.get_conditioning_video()
        report["conditioning_video_len"] = len(video)
    except Exception as exc:
        report["errors"].append("video " + str(exc)[:200])

    st0 = get_robot_state()
    EMB = st0.embodiment
    home_joints = [float(c) for c in st0.joint_positions["primary"]]
    home_pose = st0.end_effector_poses["primary"]
    home_q = [float(c) for c in home_pose.quaternion_wxyz]
    report["home_pose"] = {"p": [float(c) for c in home_pose.position], "q": home_q}
    zaxis = qrot(home_q, [0.0, 0.0, 1.0])
    if zaxis[2] < -0.95:
        down_q = home_q
    else:
        down_q = [0.0, 1.0, 0.0, 0.0]
    jaw_vec = qrot(down_q, [0.0, 1.0, 0.0])
    jaw0 = math.atan2(jaw_vec[1], jaw_vec[0])
    report["down_q"] = down_q
    report["jaw0_deg"] = round(math.degrees(jaw0), 1)

    obs0 = get_observation()
    G = setup_geometry(obs0)
    park_z = G["tz"] + 0.28
    p0 = fg_points(to_rows(obs0.cameras["front"].depth_m), None)
    if p0:
        us = [p[3] for p in p0]
        vs = [p[4] for p in p0]
        G["roi"] = [max(0, min(us) - 30), max(0, min(vs) - 30), min(G["W"], max(us) + 31),
                    min(G["H"], max(vs) + 31)]
    report["roi"] = G["roi"]
    report["fg_points_t0"] = len(p0)

    base_rows = rgb_rows("front")
    img_h = len(base_rows)
    img_w = len(base_rows[0])
    report["front_size"] = [img_w, img_h]
    y0 = int(0.12 * img_h)
    base_white = white_set(base_rows, y0)
    base_comps = components(base_white)
    report["baseline_white_comps"] = [len(c) for c in base_comps[:5]]

    # --- wait for highlight discs (hold joints) ---
    counts = []
    hl_diff = None
    hl_rows = None
    hl_dep = None
    best_count = 0
    stable = 0
    for tick_i in range(45):
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=home_joints, gripper_position=1.0, embodiment=EMB)}))
        if is_terminal(reply):
            raise Done("highlight_wait")
        rows = rgb_rows("front")
        diff = white_set(rows, y0) - base_white
        n = len(diff)
        counts.append(n)
        if n > 60:
            if n >= best_count:
                best_count = n
                hl_rows = rows
                hl_diff = diff
                hl_dep = front_depth()
            stable += 1
            if stable >= 3:
                break
    report["highlight_wait_counts"] = counts
    clock[0] = now_tick()
    report["highlight_tick"] = clock[0]

    # --- button from the baseline white blob (perception is done before any motion) ---
    if base_comps:
        bc = base_comps[0]
        bu = sum(p[0] for p in bc) / float(len(bc))
        bv = sum(p[1] for p in bc) / float(len(bc))
        report["button_px"] = [round(bu, 1), round(bv, 1), len(bc)]
        bseg = None
        bs = segment_text("front", "button")
        if bs.ok:
            for c in bs.segmentations:
                b = box_of(c)
                if b is not None and b[0] - 2 <= bu <= b[2] + 2 and b[1] - 2 <= bv <= b[3] + 2:
                    bseg = c
                    break
        if bseg is None:
            sp = segment_points("front", [[int(round(bu)), int(round(bv))]])
            if sp.ok and len(sp.segmentations) > 0:
                bseg = sp.segmentations[0]
                log("button_from", "segment_points")
        else:
            log("button_from", "segment_text")
        if bseg is not None:
            button = describe_cloud(cloud_points(bseg, "front"))
    report["button"] = button

    # --- highlighted cubes from disc interiors ---
    targets = []
    probes = []
    if hl_diff is not None:
        found = highlight_points(hl_diff, hl_rows)
        explained = []
        for f in found:
            info = seg_point_cloud(f[4], f[5])
            rec = {"px": [f[4], f[5]], "inside": f[2], "quads": f[3], "blob": f[6]}
            if info is not None:
                rec["xyz"] = [round(info["x"], 4), round(info["y"], 4), round(info["z_top"], 4)]
                rec["h"] = round(info["z_top"] - info["z_low"], 4)
                rec["side"] = round(info["side"], 4)
            if is_cube(info) and f[3] >= 2:
                rec["accept"] = True
                targets.append(info)
                explained.append([f[0], f[1]])
            probes.append(rec)
        # disc arcs with no enclosed cube: group nearby arcs, probe the union-box centre
        arcs = []
        for comp in components(hl_diff):
            if len(comp) < 25:
                break
            xs = [p[0] for p in comp]
            ys = [p[1] for p in comp]
            box = [min(xs), min(ys), max(xs), max(ys)]
            hit = False
            for e in explained:
                if box[0] - 4 <= e[0] <= box[2] + 4 and box[1] - 4 <= e[1] <= box[3] + 4:
                    hit = True
            if not hit:
                arcs.append(box)
        groups = []
        for box in arcs:
            placed = False
            for g in groups:
                gb = g["box"]
                gap_x = max(0, max(box[0], gb[0]) - min(box[2], gb[2]))
                gap_y = max(0, max(box[1], gb[1]) - min(box[3], gb[3]))
                if gap_x <= 14 and gap_y <= 6:
                    g["box"] = [min(box[0], gb[0]), min(box[1], gb[1]), max(box[2], gb[2]), max(box[3], gb[3])]
                    g["n"] += 1
                    placed = True
                    break
            if not placed:
                groups.append({"box": list(box), "n": 1})
        for g in groups:
            gb = g["box"]
            u = 0.5 * (gb[0] + gb[2])
            v = 0.5 * (gb[1] + gb[3])
            info = seg_point_cloud(u, v)
            rec = {"px": [round(u, 1), round(v, 1)], "arc_group": g["n"], "box": gb}
            if info is not None:
                rec["xyz"] = [round(info["x"], 4), round(info["y"], 4), round(info["z_top"], 4)]
                rec["h"] = round(info["z_top"] - info["z_low"], 4)
                rec["side"] = round(info["side"], 4)
            if is_cube(info):
                rec["accept"] = True
                targets.append(info)
            probes.append(rec)
    report["probes"] = probes
    uniq = []
    for c in targets:
        if all(math.sqrt((c["x"] - u["x"]) ** 2 + (c["y"] - u["y"]) ** 2) > 0.025 for u in uniq):
            uniq.append(c)
    targets = uniq
    report["n_targets"] = len(targets)
    report["targets"] = [[round(c["x"], 3), round(c["y"], 3), round(c["z_top"], 3)] for c in targets]


    # --- tracker: every above-table blob in the highlight frame; targets seeded first ---
    seeds = [[c["x"], c["y"], True, "target"] for c in targets]
    if button is not None:
        seeds.append([button["x"], button["y"], False, "button"])
    if hl_dep is None:
        hl_dep = front_depth()
    tracks.extend(seed_tracks(fg_points(hl_dep, None), seeds))
    wsx = [t["x"] for t in tracks if t["tag"] != "other"]
    wsy = [t["y"] for t in tracks if t["tag"] != "other"]
    if wsx:
        G["ws"] = [min(wsx) - 0.15, min(wsy) - 0.15, max(wsx) + 0.15, max(wsy) + 0.15]
        keep = []
        for t in tracks:
            if G["ws"][0] <= t["x"] <= G["ws"][2] and G["ws"][1] <= t["y"] <= G["ws"][3]:
                keep.append(t)
        tracks.clear()
        tracks.extend(keep)
    report["workspace_xy"] = [round(c, 3) for c in G["ws"]]
    report["tracks_init"] = [[t["id"], t["tag"], round(t["x"], 3), round(t["y"], 3), t["n0"]] for t in tracks]

    # --- let the first shuffle (it starts when the highlight disappears) pass, parked at home ---
    cur_label[0] = "first_shuffle"
    w = 0
    vanish = -1
    while w < 150:
        if vanish >= 0:
            late = [e for e in sched["ends"] if e > vanish]
            if late:
                break
            if not sched["moving"] and clock[0] - vanish > 70:
                break
        tick(home_joints, 1.0)
        w += 1
        if vanish < 0:
            nd = len(white_set(rgb_rows("front"), y0) - base_white)
            if nd < 0.3 * max(best_count, 1):
                vanish = clock[0]
                # windows "seen" before the discs vanished are tracker noise
                sched["starts"] = [x for x in sched["starts"] if x >= vanish - 3]
                sched["ends"] = [x for x in sched["ends"] if x >= vanish - 3]
                if sched["moving"] and not sched["starts"]:
                    sched["starts"].append(vanish)
    report["highlight_vanish_tick"] = vanish
    log("first_shuffle", [w, list(sched["starts"]), list(sched["ends"]), clock[0]])

    # --- first button press (outside a shuffle) ---
    wait_until(50, "press1:wait", 200)
    press("press1")
    go([button["x"] if button else home_pose.position[0], button["y"] if button else 0.0, park_z],
       down_q, 1.0, "park", speed=0.06, settle=4, tol=0.01)
    grip_state[0] = 1.0

    # --- pick each highlighted cube (tracked identity), nearest-first ---
    idx = 0
    while True:
        remaining = [t for t in tracks if t["target"] and not t["done"]]
        if not remaining or idx >= 12:
            break
        ee = [float(c) for c in get_robot_state().end_effector_poses["primary"].position]
        best = None
        bd = None
        for t in remaining:
            d = (t["x"] - ee[0]) ** 2 + (t["y"] - ee[1]) ** 2
            if bd is None or d < bd:
                bd = d
                best = t
        ok = pick(best, idx)
        if not ok:
            best["done"] = True
            best["failed"] = True
        idx += 1

    # --- final button press ---
    press("press2")
    hold(0.0, 5)
    report["final_tick"] = clock[0]
except Exception as exc:
    if str(exc).startswith("TERMINAL:"):
        report["terminated_at"] = str(exc)
    else:
        report["errors"].append("fatal " + str(exc)[:400])

report["clock"] = clock[0]
report["windows"] = {"starts": sched["starts"], "ends": sched["ends"], "noise": sched["noise"]}
report["tracks_final"] = [[t["id"], t["tag"], round(t["x"], 3), round(t["y"], 3), t["done"], t["miss"]]
                          for t in tracks]
result = report
