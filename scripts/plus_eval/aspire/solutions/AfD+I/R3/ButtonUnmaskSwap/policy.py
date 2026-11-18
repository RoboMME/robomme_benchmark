# ButtonUnmaskSwap v01 -- v00 + colour-blob cube candidates (no missing tracks, targets never
# mapped to buttons), deeper button press, and per-pick top-face refinement of the container
# centre and yaw (gripper aligned with container faces) before a Cartesian-waypoint descent.
# Strategy (all task-dependent quantities measured at runtime):
#  1. step 0: parse goal colour order; self-calibrate the RGB-D deprojection convention
#     (table-plane flatness); find small objects on the table (cubes + buttons) in 3D from
#     both cameras; locate the goal-colour cubes.
#  2. hold still until the containers drop; objects that stay uncovered are the buttons,
#     covered ones become container tracks (initialised at their cube positions).
#  3. every control tick: fuse white, container-height RGB-D points from front + wrist into
#     robot_base xy and update all container tracks (warm-start k-means + velocity).
#  4. press both buttons early (deadline step 180) with per-tick joint stepping while tracking.
#  5. go home, keep tracking until containers settle (bounded wait), then pick up the tracked
#     containers in goal order (lift, put back for all but the last), re-targeting each tick.
import math

report = {"version": "v01", "errors": [], "events": []}
STRIDE = 3
G = {"n": 0, "done": False, "obs": None, "term": False}


def ev(msg):
    report["events"].append("%d:%s" % (G["n"], msg))


def tolist2(arr):
    try:
        return arr.tolist()
    except Exception:
        return [list(row) for row in arr]


def qmat(q):
    w = float(q[0]); x = float(q[1]); y = float(q[2]); z = float(q[3])
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def quantile(vals, frac):
    s = sorted(vals)
    if len(s) == 0:
        return None
    return s[int((len(s) - 1) * frac)]


def cam_model(cam):
    K = tolist2(cam.intrinsics)
    pose = cam.camera_pose
    return {"fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]), "cy": float(K[1][2]),
            "R": qmat(list(pose.quaternion_wxyz)), "t": [float(v) for v in pose.position]}


def deproject(model, u, v, d, sgn):
    xc = (u - model["cx"]) / model["fx"] * d
    yc = (v - model["cy"]) / model["fy"] * d * sgn
    zc = d * sgn
    R = model["R"]; t = model["t"]
    return (R[0][0] * xc + R[0][1] * yc + R[0][2] * zc + t[0],
            R[1][0] * xc + R[1][1] * yc + R[1][2] * zc + t[1],
            R[2][0] * xc + R[2][1] * yc + R[2][2] * zc + t[2])


def choose_convention(cam):
    """Pick the camera-axis convention (OpenCV z-forward vs OpenGL z-backward) under which the
    dominant table plane lies BELOW the camera (the mirrored convention puts it above)."""
    model = cam_model(cam)
    dep = tolist2(cam.depth_m)
    h = len(dep); w = len(dep[0])
    best = None
    spreads = {}
    for name, sgn in (("cv", 1.0), ("gl", -1.0)):
        zs = []
        for v in range(0, h, 8):
            for u in range(0, w, 8):
                d = float(dep[v][u])
                if d > 0.01 and d < 5.0:
                    zs.append(deproject(model, u, v, d, sgn)[2])
        if len(zs) < 20:
            continue
        med = quantile(zs, 0.5)
        near = [z for z in zs if abs(z - med) < 0.03]
        spreads[name] = [round(med, 4), round(len(near) / float(len(zs)), 3), round(model["t"][2], 4)]
        if best is None or med < best[0]:
            best = (med, sgn, name)
    return best, spreads


def cam_points(cam, sgn, stride, roi, mode, zlo, zhi, box, hand):
    """mode 'all': every valid point (x,y,z,r,g,b); mode 'white': container-coloured points
    inside the z band and workspace box, minus points belonging to the robot hand."""
    model = cam_model(cam)
    rgb = tolist2(cam.rgb)
    dep = tolist2(cam.depth_m)
    h = len(dep); w = len(dep[0])
    if roi is None:
        u0, u1, v0, v1 = 0, w, 0, h
    else:
        u0, u1, v0, v1 = roi
    fx = model["fx"]; fy = model["fy"]; cx = model["cx"]; cy = model["cy"]
    R = model["R"]; t = model["t"]
    out = []
    for v in range(v0, v1, stride):
        row = rgb[v]; drow = dep[v]
        ry = (v - cy) / fy * sgn
        for u in range(u0, u1, stride):
            px = row[u]
            r = px[0]; g = px[1]; b = px[2]
            if mode == "white":
                mn = min(r, g, b)
                if mn < 140 or max(r, g, b) - mn > 40:
                    continue
            d = float(drow[u])
            if not (d > 0.01 and d < 5.0):
                continue
            xc = (u - cx) / fx * d
            yc = ry * d
            zc = d * sgn
            X = R[0][0] * xc + R[0][1] * yc + R[0][2] * zc + t[0]
            Y = R[1][0] * xc + R[1][1] * yc + R[1][2] * zc + t[1]
            Z = R[2][0] * xc + R[2][1] * yc + R[2][2] * zc + t[2]
            if mode == "white":
                if Z < zlo or Z > zhi:
                    continue
                if X < box[0] or X > box[1] or Y < box[2] or Y > box[3]:
                    continue
                if hand is not None:
                    dx = X - hand[0]; dy = Y - hand[1]
                    if dx * dx + dy * dy < hand[3] * hand[3] and Z > hand[2] - 0.01:
                        continue
                out.append((X, Y, Z))
            else:
                out.append((X, Y, Z, r, g, b, u, v))
    return out


def colour_of(name, r, g, b):
    if name == "red":
        return r > 110 and g < 70 and b < 70 and r > 2 * g
    if name == "green":
        return g > 100 and g > r + 40 and g > b + 40
    if name == "blue":
        return b > 90 and b > r + 40 and b > g + 30
    if name == "yellow":
        return r > 150 and g > 150 and b < 90
    if name == "purple":
        return r > 90 and b > 110 and g < 90
    if name == "orange":
        return r > 170 and g > 90 and g < 170 and b < 45 and r - g > 50
    if name == "black":
        return max(r, g, b) < 60
    if name == "white":
        return min(r, g, b) > 200 and max(r, g, b) - min(r, g, b) < 25
    return False


def grid_clusters(pts, cell):
    cells = {}
    for i in range(len(pts)):
        key = (int(math.floor(pts[i][0] / cell)), int(math.floor(pts[i][1] / cell)))
        if key in cells:
            cells[key].append(i)
        else:
            cells[key] = [i]
    seen = set()
    clusters = []
    for key in cells:
        if key in seen:
            continue
        stack = [key]; seen.add(key); members = []
        while len(stack) > 0:
            k = stack.pop()
            members.extend(cells[k])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nk = (k[0] + dx, k[1] + dy)
                    if nk in cells and nk not in seen:
                        seen.add(nk); stack.append(nk)
        clusters.append(members)
    return clusters


def jlist(state):
    return [float(v) for v in state.joint_positions["primary"]]


def ee_pos(state):
    return [float(v) for v in state.end_effector_poses["primary"].position]


# ----------------------------------------------------------------------------------------
def main():
    ctx = get_task_context()
    goal = str(ctx.language)
    report["goal"] = goal
    try:
        meta = robomme_plus.get_task_metadata()
        max_steps = int(meta["max_steps"])
    except Exception as exc:
        max_steps = None
        report["errors"].append("metadata: " + str(exc))
    report["max_steps"] = max_steps
    try:
        video = robomme_plus.get_conditioning_video()
        report["video_len"] = len(video)
    except Exception as exc:
        report["errors"].append("video: " + str(exc))

    low = goal.lower()
    cut = low.find("pick")
    seg = low[cut:] if cut >= 0 else low
    found = []
    for c in ("red", "green", "blue", "yellow", "purple", "orange", "black", "white"):
        start = 0
        while True:
            p = seg.find(c + " cube", start)
            if p < 0:
                break
            found.append((p, c)); start = p + 1
    order = [c for p, c in sorted(found)]
    report["order"] = order
    if len(order) == 0:
        raise ValueError("no colour order parsed from goal")

    obs = get_observation()
    state = obs.robot_state
    emb = state.embodiment
    home = jlist(state)
    home_pose = state.end_effector_poses["primary"]
    base_frame = state.base_frame
    report["home_ee"] = [round(v, 4) for v in ee_pos(state)]
    report["home_quat"] = [round(float(v), 4) for v in home_pose.quaternion_wxyz]
    report["cameras"] = sorted(list(obs.cameras.keys()))

    conv = {}
    for name in ("front", "wrist"):
        best, spreads = choose_convention(obs.cameras[name])
        conv[name] = best[1]
        report["conv_" + name] = [best[2], spreads]

    # ---------------- step-0 scene analysis (both cameras) ----------------
    allpts = []
    for name in ("front", "wrist"):
        allpts.extend(cam_points(obs.cameras[name], conv[name], 2, None, "all", 0, 0, None, None))
    zs = [p[2] for p in allpts]
    med = quantile(zs, 0.5)
    table_z = quantile([z for z in zs if abs(z - med) < 0.02], 0.5)
    report["table_z"] = round(table_z, 4)
    objs = [p for p in allpts if p[2] > table_z + 0.006 and (p[0] * p[0] + p[1] * p[1]) > 0.15 * 0.15]
    clusters = []
    for members in grid_clusters(objs, 0.01):
        if len(members) < 6:
            continue
        xs = [objs[i][0] for i in members]; ys = [objs[i][1] for i in members]
        zz = [objs[i][2] for i in members]
        top = quantile(zz, 0.9)
        ext = max(max(xs) - min(xs), max(ys) - min(ys))
        if top > table_z + 0.12 or ext > 0.08:
            continue
        clusters.append({"x": quantile(xs, 0.5), "y": quantile(ys, 0.5), "top": top, "n": len(members), "ext": ext})
    report["clusters0"] = [[round(c["x"], 3), round(c["y"], 3), round(c["top"] - table_z, 3), c["n"]] for c in clusters]
    if len(clusters) < 3:
        raise ValueError("too few table objects found at step 0")
    # v01: every distinctly coloured cube becomes a candidate from its own colour-pixel blob
    # (largest 1 cm-connected component, below container height); blobs not matched by a
    # geometric cluster are added as new candidates so that no cube/track is missing.
    lowobjs = [p for p in objs if p[2] < table_z + 0.10]
    colour_pos = {}
    for c in ("red", "green", "blue", "yellow", "purple", "orange", "black", "white"):
        if c == "white" and c not in order:
            continue
        cp = [p for p in lowobjs if colour_of(c, p[3], p[4], p[5])]
        if len(cp) < 4:
            continue
        comps = grid_clusters(cp, 0.01)
        big = max(comps, key=lambda m: len(m))
        if len(big) < 4:
            continue
        mx = quantile([cp[i][0] for i in big], 0.5); my = quantile([cp[i][1] for i in big], 0.5)
        colour_pos[c] = (mx, my, len(big))
    report["colour_pos"] = dict((c, [round(v[0], 3), round(v[1], 3), v[2]]) for c, v in colour_pos.items())
    for c in colour_pos:
        mx, my, npx = colour_pos[c]
        best = None
        for i in range(len(clusters)):
            d = math.hypot(clusters[i]["x"] - mx, clusters[i]["y"] - my)
            if best is None or d < best[0]:
                best = (d, i)
        if best is not None and best[0] < 0.03 and clusters[best[1]].get("colour") is None:
            clusters[best[1]]["colour"] = c
        else:
            clusters.append({"x": mx, "y": my, "top": table_z + 0.03, "n": npx, "ext": 0.0, "colour": c,
                             "added": True})
            ev("added colour candidate %s at (%.3f, %.3f)" % (c, mx, my))
    targets = {}
    for c in order:
        ks = [i for i in range(len(clusters)) if clusters[i].get("colour") == c]
        if len(ks) == 0:
            report["errors"].append("colour %s not seen" % c)
            continue
        targets[c] = ks[0]
        report["events"].append("target %s -> cluster %d" % (c, ks[0]))
    report["clusters0b"] = [[round(c["x"], 3), round(c["y"], 3), c.get("colour")] for c in clusters]
    low_top = max(c["top"] for c in clusters if not c.get("added"))
    xs = [c["x"] for c in clusters]; ys = [c["y"] for c in clusters]
    box = [min(xs) - 0.10, max(xs) + 0.10, min(ys) - 0.10, max(ys) + 0.10]
    report["box"] = [round(v, 3) for v in box]
    # front-camera ROI from workspace pixels

    tr = {"pos": [[c["x"], c["y"]] for c in clusters], "vel": [[0.0, 0.0] for c in clusters],
          "cnt": [0 for c in clusters], "active": [True for c in clusters]}
    P = {"zlo": low_top + 0.01, "zhi": table_z + 0.15, "gate": 0.04, "cont_top": None,
         "box": box, "hand": None, "frozen": -1}
    hist = []

    def get_points(o):
        st = o.robot_state
        e = ee_pos(st)
        hand = (e[0], e[1], e[2], 0.07)
        pts = []
        for name in ("front", "wrist"):
            try:
                pts.extend(cam_points(o.cameras[name], conv[name], STRIDE, None, "white", P["zlo"], P["zhi"], P["box"], hand))
            except Exception as exc:
                report["errors"].append("pts %s: %s" % (name, str(exc)))
        return pts

    def update_tracks(pts):
        K = len(tr["pos"])
        pred = []
        for k in range(K):
            pred.append([tr["pos"][k][0] + tr["vel"][k][0], tr["pos"][k][1] + tr["vel"][k][1]])
        cen = [list(p) for p in pred]
        gate2 = P["gate"] * P["gate"]
        cnts = [0] * K
        for it in range(2):
            sx = [0.0] * K; sy = [0.0] * K; cnts = [0] * K
            for p in pts:
                bk = -1; bd = gate2
                for k in range(K):
                    if not tr["active"][k]:
                        continue
                    dx = p[0] - cen[k][0]; dy = p[1] - cen[k][1]
                    dd = dx * dx + dy * dy
                    if dd < bd:
                        bd = dd; bk = k
                if bk >= 0:
                    sx[bk] += p[0]; sy[bk] += p[1]; cnts[bk] += 1
            for k in range(K):
                if cnts[k] >= 6:
                    cen[k] = [sx[k] / cnts[k], sy[k] / cnts[k]]
                else:
                    cen[k] = list(pred[k])
        for k in range(K):
            if not tr["active"][k]:
                continue
            if k == P["frozen"]:
                tr["vel"][k] = [0.0, 0.0]
                continue
            close = False
            for j in range(K):
                if j != k and tr["active"][j]:
                    if math.hypot(pred[j][0] - pred[k][0], pred[j][1] - pred[k][1]) < 1.2 * P["gate"]:
                        close = True
            new = cen[k]
            if close and cnts[k] >= 6:
                new = [0.5 * new[0] + 0.5 * pred[k][0], 0.5 * new[1] + 0.5 * pred[k][1]]
            vx = new[0] - tr["pos"][k][0]; vy = new[1] - tr["pos"][k][1]
            if cnts[k] >= 6:
                nvx = 0.6 * tr["vel"][k][0] + 0.4 * vx; nvy = 0.6 * tr["vel"][k][1] + 0.4 * vy
            else:
                nvx = 0.9 * tr["vel"][k][0]; nvy = 0.9 * tr["vel"][k][1]
            sp = math.hypot(nvx, nvy)
            if sp > 0.01:
                nvx = nvx * 0.01 / sp; nvy = nvy * 0.01 / sp
            tr["vel"][k] = [nvx, nvy]
            tr["pos"][k] = new
            tr["cnt"][k] = cnts[k]

    def observe_and_track(o):
        if P["cont_top"] is None:
            return
        pts = get_points(o)
        update_tracks(pts)
        if G["n"] % 5 == 0:
            rec = [G["n"], len(pts)]
            for c in order:
                if c in targets:
                    k = targets[c]
                    rec.append([round(tr["pos"][k][0], 3), round(tr["pos"][k][1], 3), tr["cnt"][k],
                                round(1000 * math.hypot(tr["vel"][k][0], tr["vel"][k][1]), 2)])
            hist.append(rec)

    def refine_face(k, cx0, cy0, nticks, stride=2):
        """Accumulate top-face container points around (cx0, cy0) for a few ticks (both
        cameras, Voronoi-restricted against other tracks) and fit a minimum-area rectangle:
        returns (cx, cy, yaw offset in (-pi/4, pi/4], side_a, side_b, npts)."""
        ct = P["cont_top"]
        acc = []
        for t in range(nticks):
            o = do_step(jlist(G["obs"].robot_state), 1.0)
            if o is None:
                return None
            e = ee_pos(o.robot_state)
            hand = (e[0], e[1], e[2], 0.07)
            for name in ("front", "wrist"):
                try:
                    acc.extend(cam_points(o.cameras[name], conv[name], stride, None, "white", ct - 0.010, ct + 0.010,
                                          P["box"], hand))
                except Exception as exc:
                    report["errors"].append("face pts: " + str(exc))
        cx = cx0; cy = cy0
        sel = []
        for it in range(3):
            sel = []
            for p in acc:
                dx = p[0] - cx; dy = p[1] - cy
                dd = dx * dx + dy * dy
                if dd > 0.045 * 0.045:
                    continue
                mine = True
                for j in range(len(tr["pos"])):
                    if j != k and tr["active"][j]:
                        ex = p[0] - tr["pos"][j][0]; ey = p[1] - tr["pos"][j][1]
                        if ex * ex + ey * ey < dd:
                            mine = False
                            break
                if mine:
                    sel.append(p)
            if len(sel) < 15:
                return None
            cx = sum(p[0] for p in sel) / len(sel); cy = sum(p[1] for p in sel) / len(sel)
        best = None
        cands = [a * math.pi / 180.0 for a in range(0, 90, 3)]
        for rnd in range(2):
            for a in cands:
                ca = math.cos(a); sa = math.sin(a)
                us = [(p[0] - cx) * ca + (p[1] - cy) * sa for p in sel]
                ws = [-(p[0] - cx) * sa + (p[1] - cy) * ca for p in sel]
                u0 = quantile(us, 0.03); u1 = quantile(us, 0.97)
                w0 = quantile(ws, 0.03); w1 = quantile(ws, 0.97)
                area = (u1 - u0) * (w1 - w0)
                if best is None or area < best[0]:
                    mu = 0.5 * (u0 + u1); mw = 0.5 * (w0 + w1)
                    best = (area, a, cx + mu * ca - mw * sa, cy + mu * sa + mw * ca, u1 - u0, w1 - w0)
            cands = [best[1] + d * math.pi / 180.0 for d in (-2, -1, 1, 2)]
        a = best[1]
        while a > math.pi / 4:
            a -= math.pi / 2
        while a <= -math.pi / 4:
            a += math.pi / 2
        return (best[2], best[3], a, best[4], best[5], len(sel))

    def do_step(q, grip):
        if G["term"]:
            return None
        st = G["obs"].robot_state
        try:
            res = step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=[float(v) for v in q],
                gripper_position=float(grip), embodiment=st.embodiment)}))
        except Exception as exc:
            report["errors"].append("step: " + str(exc))
            G["term"] = True
            return None
        G["n"] += 1
        try:
            if bool(res.terminated) or bool(res.truncated):
                G["term"] = True
                ev("terminated=%s truncated=%s" % (str(res.terminated), str(res.truncated)))
                return None
        except Exception:
            pass
        o = None
        try:
            o = res.observation
            o.robot_state
        except Exception:
            o = None
        if o is None:
            try:
                o = get_observation()
            except Exception as exc:
                report["errors"].append("obs: " + str(exc))
                G["term"] = True
                return None
        G["obs"] = o
        observe_and_track(o)
        return o

    def drive(target, grip, max_ticks, max_delta, tol, hold=0):
        st = G["obs"].robot_state
        sp = jlist(st)
        for i in range(max_ticks):
            if G["term"]:
                return False
            nsp = []
            for j in range(7):
                dlt = target[j] - sp[j]
                if dlt > max_delta:
                    dlt = max_delta
                if dlt < -max_delta:
                    dlt = -max_delta
                nsp.append(sp[j] + dlt)
            sp = nsp
            o = do_step(sp, grip)
            if o is None:
                return False
            q = jlist(o.robot_state)
            err = max(abs(q[j] - target[j]) for j in range(7))
            reached = max(abs(sp[j] - target[j]) for j in range(7)) < 1e-6
            if reached and err < tol:
                for hh in range(hold):
                    if do_step(sp, grip) is None:
                        return False
                return True
        return False

    def topdown(yaw):
        return [0.0, math.cos(yaw / 2.0), math.sin(yaw / 2.0), 0.0]

    hq = [float(v) for v in home_pose.quaternion_wxyz]
    hR = qmat(hq)
    home_yaw = math.atan2(hR[1][0], hR[0][0])
    report["home_zaxis"] = [round(hR[0][2], 3), round(hR[1][2], 3), round(hR[2][2], 3)]

    def ik(pos, yaw):
        seed = jlist(G["obs"].robot_state)
        try:
            r = solve_ik(Pose(position=[float(v) for v in pos], quaternion_wxyz=topdown(yaw), frame=base_frame),
                         seed_joints=seed)
        except Exception as exc:
            report["errors"].append("ik exc: " + str(exc))
            return None
        if not r.ok:
            if len(report["errors"]) < 60:
                report["errors"].append("ik not ok at %s" % str([round(v, 3) for v in pos]))
            return None
        try:
            return [float(v) for v in r.joint_positions]
        except Exception:
            pass
        try:
            return [float(v) for v in r.solution]
        except Exception as exc:
            report["errors"].append("ik attr: " + str(exc))
            return None

    # ---------------- wait for the containers to drop ----------------
    G["obs"] = obs

    def drop_points(o):
        e = ee_pos(o.robot_state)
        out = []
        for name in ("front", "wrist"):
            out.extend(cam_points(o.cameras[name], conv[name], STRIDE, None, "white", low_top + 0.01,
                                  table_z + 0.15, box, (e[0], e[1], e[2], 0.07)))
        return out

    baseline = len(drop_points(obs))
    report["drop_baseline"] = baseline
    drop_seen = 0
    for i in range(60):
        o = do_step(home, 1.0)
        if o is None:
            break
        pts = drop_points(o)
        if len(pts) > 15 * len(clusters) and len(pts) > 3 * baseline:
            drop_seen += 1
        if drop_seen >= 2:
            cont_top = quantile([p[2] for p in pts], 0.95)
            P["cont_top"] = cont_top
            # classify clusters: covered = cube under container, uncovered = button
            cover = []
            for c in clusters:
                n = 0
                for p in pts:
                    if (p[0] - c["x"]) ** 2 + (p[1] - c["y"]) ** 2 < 0.025 * 0.025:
                        n += 1
                cover.append(n)
            report["cover"] = cover
            report["cont_top_rel"] = round(cont_top - table_z, 4)
            ev("drop detected npts=%d" % len(pts))
            idx = sorted(range(len(clusters)), key=lambda k: cover[k])
            unc = [k for k in idx if cover[k] < 3 and clusters[k].get("colour") is None]
            if len(unc) < 2:
                unc = [k for k in idx if cover[k] < 3]
            if len(unc) >= 2:
                unc = sorted(unc, key=lambda k: clusters[k]["x"] ** 2 + clusters[k]["y"] ** 2)[:2]
            else:
                unc = idx[:2]
            buttons = [clusters[k] for k in unc]
            for k in unc:
                tr["active"][k] = False
            report["buttons"] = [[round(b["x"], 3), round(b["y"], 3), round(b["top"] - table_z, 3)] for b in buttons]
            # container size -> gate
            dists = []
            for p in pts:
                bd = None
                for k in range(len(clusters)):
                    if tr["active"][k]:
                        d = math.hypot(p[0] - clusters[k]["x"], p[1] - clusters[k]["y"])
                        if bd is None or d < bd:
                            bd = d
                if bd is not None and bd < 0.06:
                    dists.append(bd)
            r_ext = quantile(dists, 0.9)
            P["gate"] = max(0.02, 1.4 * r_ext)
            P["r_ext"] = r_ext
            report["r_ext"] = round(r_ext, 4)
            P["zlo"] = max(low_top + 0.005, table_z + 0.5 * (cont_top - table_z))
            P["zhi"] = cont_top + 0.015
            update_tracks(get_points(o))
            break
    if P["cont_top"] is None:
        raise ValueError("container drop never detected")
    for c in order:
        if c in targets and not tr["active"][targets[c]]:
            report["errors"].append("target %s was classified as a button" % c)

    # ---------------- press both buttons ----------------
    press_log = []
    bsorted = sorted(buttons, key=lambda b: b["y"])
    for b in bsorted:
        above = ik([b["x"], b["y"], b["top"] + 0.06], home_yaw)
        down = ik([b["x"], b["y"], b["top"] - 0.03], home_yaw)
        if above is None or down is None:
            press_log.append("ik fail")
            continue
        ok1 = drive(above, 0.0, 40, 0.06, 0.03)
        ok2 = drive(down, 0.0, 25, 0.03, 0.01, hold=6)
        ez = ee_pos(G["obs"].robot_state)[2] if not G["term"] else None
        ok3 = drive(above, 0.0, 15, 0.05, 0.03)
        press_log.append([ok1, ok2, ok3, G["n"], None if ez is None else round(ez - table_z, 4)])
        if G["term"]:
            break
    report["press"] = press_log
    ev("presses done")
    drive(home, 1.0, 40, 0.06, 0.02)

    # ---------------- wait for containers to settle ----------------
    if max_steps is not None:
        wait_until = min(500, max_steps - 360)
    else:
        wait_until = 500
    calm = 0
    while not G["term"] and G["n"] < wait_until:
        o = do_step(home, 1.0)
        if o is None:
            break
        spmax = 0.0
        for k in range(len(tr["pos"])):
            if tr["active"][k]:
                spmax = max(spmax, math.hypot(tr["vel"][k][0], tr["vel"][k][1]))
        if spmax < 0.0004:
            calm += 1
        else:
            calm = 0
        if calm >= 15:
            ev("settled")
            break
    report["settle_step"] = G["n"]

    # ---------------- pick in order ----------------
    cont_top = P["cont_top"]
    grasp_z = cont_top - 0.4 * (cont_top - table_z)
    hover_z = cont_top + 0.11
    picks = []
    report["fits"] = []
    for pi in range(len(order)):
        c = order[pi]
        if G["term"]:
            break
        if c not in targets:
            picks.append([c, "no target"])
            continue
        k = targets[c]
        log = [c]
        yaw = home_yaw
        # v01: from the home pose (full unoccluded view of the scene) fit the target container's
        # top face: centre + yaw. Go home first for every pick after the first.
        if pi > 0:
            drive(home, 1.0, 60, 0.06, 0.02, hold=3)
        fit = refine_face(k, tr["pos"][k][0], tr["pos"][k][1], 2, 1)
        report["fits"].append([c, G["n"]] + ([round(v, 4) for v in fit] if fit is not None else [None]))
        static = math.hypot(tr["vel"][k][0], tr["vel"][k][1]) < 0.0015
        good = fit is not None and min(fit[3], fit[4]) > 0.75 * max(fit[3], fit[4]) \
            and math.hypot(fit[0] - tr["pos"][k][0], fit[1] - tr["pos"][k][1]) < 0.03
        if good:
            tr["pos"][k] = [fit[0], fit[1]]
            yaw = home_yaw + fit[2]
            if static:
                P["frozen"] = k
        log.append(["fit", good, static, round(yaw, 3)])
        # hover servo
        reached = False
        sp = jlist(G["obs"].robot_state)
        qlast = None
        for i in range(90):
            if G["term"]:
                break
            tx = tr["pos"][k][0] + 3 * tr["vel"][k][0]; ty = tr["pos"][k][1] + 3 * tr["vel"][k][1]
            if i % 3 == 0 or qlast is None:
                qn = ik([tx, ty, hover_z], yaw)
                if qn is not None:
                    qlast = qn
            q = qlast if qlast is not None else sp
            nsp = []
            for j in range(7):
                dlt = max(-0.05, min(0.05, q[j] - sp[j]))
                nsp.append(sp[j] + dlt)
            sp = nsp
            o = do_step(sp, 1.0)
            if o is None:
                break
            e = ee_pos(o.robot_state)
            qe = jlist(o.robot_state)
            if math.hypot(e[0] - tr["pos"][k][0], e[1] - tr["pos"][k][1]) < 0.006 and abs(e[2] - hover_z) < 0.02 \
                    and math.hypot(tr["vel"][k][0], tr["vel"][k][1]) < 0.0015 \
                    and max(abs(qe[j] - q[j]) for j in range(7)) < 0.02:
                reached = True
                break
        log.append(["hover", reached, G["n"], [round(tr["pos"][k][0], 3), round(tr["pos"][k][1], 3)]])
        if G["term"]:
            picks.append(log)
            break
        P["frozen"] = k
        gx = tr["pos"][k][0]; gy = tr["pos"][k][1]
        qd = ik([gx, gy, grasp_z], yaw)
        if qd is None:
            log.append("grasp ik fail")
            picks.append(log)
            P["frozen"] = -1
            continue
        nwp = 4
        for wi in range(1, nwp):
            zw = hover_z + (grasp_z - hover_z) * wi / float(nwp)
            qw = ik([gx, gy, zw], yaw)
            if qw is not None:
                drive(qw, 1.0, 15, 0.03, 0.02)
        drive(qd, 1.0, 30, 0.03, 0.01)
        e = ee_pos(G["obs"].robot_state)
        log.append(["at-grasp", G["n"], round(e[0] - gx, 4), round(e[1] - gy, 4), round(e[2] - table_z, 4),
                    round(yaw, 3)])
        for i in range(12):
            if do_step(qd, 0.0) is None:
                break
        qu = ik([gx, gy, cont_top + 0.10], yaw)
        if qu is not None:
            drive(qu, 0.0, 30, 0.03, 0.02, hold=5)
        if G["term"]:
            log.append(["lifted-terminal", G["n"]])
            picks.append(log)
            break
        st = G["obs"].robot_state
        log.append(["lift", G["n"], round(float(st.gripper_positions["primary"]), 3), round(ee_pos(st)[2] - table_z, 3)])
        if pi < len(order) - 1:
            qp = ik([gx, gy, grasp_z + 0.005], yaw)
            if qp is not None:
                drive(qp, 0.0, 30, 0.03, 0.015)
            for i in range(8):
                if do_step(jlist(G["obs"].robot_state), 1.0) is None:
                    break
            qh = ik([gx, gy, hover_z], yaw)
            if qh is not None:
                drive(qh, 1.0, 25, 0.04, 0.03)
            # re-seed the put-back track at the release point
            tr["pos"][k] = [gx, gy]; tr["vel"][k] = [0.0, 0.0]
        P["frozen"] = -1
        picks.append(log)
    report["picks"] = picks
    report["track_hist"] = hist
    report["final_tracks"] = [[round(p[0], 3), round(p[1], 3)] for p in tr["pos"]]
    report["steps"] = G["n"]


try:
    main()
except Exception as exc:
    report["errors"].append("fatal: " + str(exc))
    report["steps"] = G["n"]
result = report
