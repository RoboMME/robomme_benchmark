# MoveCube v04 -- generated CaP program (ASPIRE Stage 1).
import math

# Pure-python perception helpers, written to be pasted into the sandboxed program.
# (No imports, no underscore names, no getattr.)

def pix_class(r, g, b):
    mx = max(r, g, b); mn = min(r, g, b)
    v = mx / 255.0
    if mx == 0:
        return 'dark'
    s = (mx - mn) / float(mx)
    if s < 0.18:
        if v < 0.45:
            return 'dark'
        return 'grey'
    d = float(mx - mn)
    if mx == r:
        h = ((g - b) / d) / 6.0
    elif mx == g:
        h = (2.0 + (b - r) / d) / 6.0
    else:
        h = (4.0 + (r - g) / d) / 6.0
    if h < 0:
        h += 1.0
    if (h < 0.025 or h > 0.97) and s > 0.7:
        return 'red'
    if 0.42 <= h < 0.55:
        return 'teal'
    if 0.55 <= h < 0.66:
        if v > 0.45:
            return 'blue'
        return 'dark'
    if 0.11 <= h < 0.2 and s > 0.45:
        return 'yellow'
    if 0.7 <= h < 0.95:
        return 'purple'
    if (h < 0.07 or h > 0.95) and 0.22 <= s < 0.45 and v > 0.55:
        return 'pink'
    if (h < 0.07 or h > 0.95) and s < 0.22:
        return 'grey'
    if 0.075 <= h < 0.14 and s < 0.5:
        return 'tan'
    if 0.02 <= h < 0.11 and s >= 0.45:
        return 'wood'
    return 'other'


TOOL_CLASSES = ('teal', 'grey', 'pink', 'tan', 'dark', 'yellow')


def class_grid(rgb_rows, height, width):
    grid = []
    for v in range(height):
        row = rgb_rows[v]
        out = []
        for u in range(width):
            p = row[u]
            out.append(pix_class(int(p[0]), int(p[1]), int(p[2])))
        grid.append(out)
    return grid


def components(grid, height, width, top_margin=10, min_size=25):
    """Connected components (8-neighbour) of tool-coloured pixels, excluding pixels next to blue."""
    seen = [[False] * width for k in range(height)]
    comps = []
    for v0 in range(top_margin, height):
        for u0 in range(width):
            if seen[v0][u0] or grid[v0][u0] not in TOOL_CLASSES:
                continue
            stack = [(v0, u0)]
            seen[v0][u0] = True
            pix = []
            touches_top = False
            while stack:
                v, u = stack.pop()
                pix.append((v, u))
                if v <= top_margin:
                    touches_top = True
                for dv in (-1, 0, 1):
                    for du in (-1, 0, 1):
                        vv = v + dv; uu = u + du
                        if vv < top_margin or vv >= height or uu < 0 or uu >= width:
                            continue
                        if seen[vv][uu] or grid[vv][uu] not in TOOL_CLASSES:
                            continue
                        seen[vv][uu] = True
                        stack.append((vv, uu))
            if len(pix) < min_size:
                continue
            near_blue = 0
            for (v, u) in pix:
                hit = False
                for dv in (-2, 0, 2):
                    for du in (-2, 0, 2):
                        vv = v + dv; uu = u + du
                        if 0 <= vv < height and 0 <= uu < width and grid[vv][uu] == 'blue':
                            hit = True
                if hit:
                    near_blue += 1
            if near_blue > 0.3 * len(pix) and comp_tan(grid, pix) < 40:
                continue
            comp = describe(grid, pix)
            comp['near_blue'] = near_blue
            comp['touches_top'] = touches_top
            if comp['kind'] == 'other':
                continue
            comps.append(comp)
    return comps


def comp_tan(grid, pix):
    k = 0
    for (v, u) in pix:
        if grid[v][u] == 'tan':
            k += 1
    return k


def describe(grid, pix):
    counts = {}
    for (v, u) in pix:
        c = grid[v][u]
        counts[c] = counts.get(c, 0) + 1
    def stats(cname):
        sel = [(v, u) for (v, u) in pix if grid[v][u] == cname]
        if not sel:
            return None
        vs = [p[0] for p in sel]; us = [p[1] for p in sel]
        return {'n': len(sel), 'cu': sum(us) / float(len(us)), 'cv': sum(vs) / float(len(vs)),
                'bbox': [min(us), min(vs), max(us), max(vs)]}
    vs = [p[0] for p in pix]; us = [p[1] for p in pix]
    # blue-adjacent (ramp texture) count
    comp = {'n': len(pix), 'counts': counts, 'bbox': [min(us), min(vs), max(us), max(vs)],
            'cu': sum(us) / float(len(us)), 'cv': sum(vs) / float(len(vs)),
            'teal': stats('teal'), 'pink': stats('pink'), 'yellow': stats('yellow'),
            'tan': stats('tan'), 'dark': stats('dark'), 'pixels': pix}
    comp['shape'] = shape_features(comp)
    comp['kind'] = tool_kind(comp)
    return comp


def shape_features(comp):
    ts = comp['teal']
    if ts is None:
        return None
    tu = ts['cu']; tv = ts['cv']
    far = None; far_d = -1.0
    for (v, u) in comp['pixels']:
        d = ((u - tu) ** 2 + (v - tv) ** 2) ** 0.5
        if d > far_d:
            far_d = d; far = (u, v)
    if far_d < 3:
        return None
    au = (far[0] - tu) / far_d; av = (far[1] - tv) / far_d
    perp = []
    for (v, u) in comp['pixels']:
        along = (u - tu) * au + (v - tv) * av
        if along > 0.75 * far_d:
            perp.append((u - tu) * (-av) + (v - tv) * au)
    width = (max(perp) - min(perp) + 1.0) if perp else 0.0
    density = len(perp) / max(1.0, width * 0.25 * far_d)
    return {'length': far_d, 'end_width': width, 'ratio': width / far_d, 'density': density,
            'end_uv': [far[0], far[1]], 'handle_uv': [tu, tv], 'axis_uv': [au, av]}


def tool_kind(comp):
    c = comp['counts']
    n = float(comp['n'])
    yel = c.get('yellow', 0); teal = c.get('teal', 0); pink = c.get('pink', 0)
    tan = c.get('tan', 0); dark = c.get('dark', 0)
    if yel >= 0.5 * n:
        return 'ball'
    if tan >= 40 and tan >= 0.12 * n:
        return 'hammer'
    if teal >= 6:
        sh = comp['shape']
        if sh is None or sh['length'] < 12:
            return 'teal_unknown'
        if sh['ratio'] < 0.17:
            return 'stick'
        if sh['density'] >= 0.5:
            return 'spatula'
        return 'hook'
    return 'other'


def moved_fraction(comp, grid_late):
    changed = 0
    for (v, u) in comp['pixels']:
        if grid_late[v][u] != 'wood':
            pass
        if grid_late[v][u] == 'wood':
            changed += 1
    return changed / float(comp['n'])


# ---------------------------------------------------------------------------
# MoveCube program body (v04; v00 header kept for history).  Strategy:
#  1. From the conditioning video, find which tool-like object was moved
#     (connected colour component in the first frame that has become table
#     coloured by the end of the video), classify it (hook / stick / spatula /
#     hammer / ball).
#  2. In the live front image, pick the component of the same kind.
#  3. Tools: grasp the anchor (teal handle, or dark head for a hammer) top-down,
#     lift, rotate so the tool points along cube->target, lower behind the cube
#     and push the cube onto the target with the tool's far end.
#     Ball: grasp, release on the high edge of the blue ramp.
# ---------------------------------------------------------------------------

report = {"version": "v04", "notes": []}
FRONT = "front"


def note(msg):
    report["notes"].append(str(msg))


def to_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        return [[arr[v][u] for u in range(len(arr[0]))] for v in range(len(arr))]


def quat_to_mat(q):
    w = float(q[0]); x = float(q[1]); y = float(q[2]); z = float(q[3])
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def yaw_quat(angle):
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]


def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def cam_model(cam, depth_rows, convention):
    K = to_rows(cam.intrinsics)
    pose = cam.camera_pose
    m = {"fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]), "cy": float(K[1][2]),
         "R": quat_to_mat(pose.quaternion_wxyz), "t": [float(c) for c in pose.position],
         "depth": depth_rows, "conv": convention}
    return m


def deproject(m, u, v):
    d = float(m["depth"][v][u])
    if not (d > 0.0) or d > 10.0:
        return None
    x = (u - m["cx"]) / m["fx"] * d
    y = (v - m["cy"]) / m["fy"] * d
    if m["conv"] == "opencv":
        pc = (x, y, d)
    else:
        pc = (x, -y, -d)
    R = m["R"]; t = m["t"]
    return [R[0][0] * pc[0] + R[0][1] * pc[1] + R[0][2] * pc[2] + t[0],
            R[1][0] * pc[0] + R[1][1] * pc[1] + R[1][2] * pc[2] + t[1],
            R[2][0] * pc[0] + R[2][1] * pc[1] + R[2][2] * pc[2] + t[2]]


def deproject_many(m, pix_uv):
    pts = []
    for (u, v) in pix_uv:
        p = deproject(m, u, v)
        if p is not None:
            pts.append(p)
    return pts


def median(vals):
    s = sorted(vals)
    if not s:
        return None
    return s[len(s) // 2]


def quant(vals, q):
    s = sorted(vals)
    if not s:
        return None
    return s[int((len(s) - 1) * q)]


def centroid(pts):
    n = float(len(pts))
    return [sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n, sum(p[2] for p in pts) / n]


def class_pixels(grid, cname, top_margin=10):
    out = []
    for v in range(top_margin, len(grid)):
        row = grid[v]
        for u in range(len(row)):
            if row[u] == cname:
                out.append((u, v))
    return out


def largest_blob(grid, cname, top_margin=10):
    """Largest 8-connected blob of one colour class, as (u, v) list."""
    h = len(grid); w = len(grid[0])
    seen = {}
    best = []
    for (u0, v0) in class_pixels(grid, cname, top_margin):
        if (u0, v0) in seen:
            continue
        stack = [(u0, v0)]
        seen[(u0, v0)] = True
        blob = []
        while stack:
            u, v = stack.pop()
            blob.append((u, v))
            for dv in (-1, 0, 1):
                for du in (-1, 0, 1):
                    uu = u + du; vv = v + dv
                    if 0 <= uu < w and top_margin <= vv < h and (uu, vv) not in seen and grid[vv][uu] == cname:
                        seen[(uu, vv)] = True
                        stack.append((uu, vv))
        if len(blob) > len(best):
            best = blob
    return best


def summarize(comp):
    out = {"kind": comp["kind"], "n": comp["n"], "cu": round(comp["cu"], 1), "cv": round(comp["cv"], 1),
           "counts": comp["counts"]}
    if comp["shape"] is not None:
        sh = comp["shape"]
        out["shape"] = {"length": round(sh["length"], 1), "end_width": round(sh["end_width"], 1),
                        "ratio": round(sh["ratio"], 3), "density": round(sh["density"], 3)}
    return out


def ex_flags(res):
    info = {}
    try:
        info["ok"] = bool(res.ok)
    except Exception:
        info["ok"] = None
    for key in ("status", "steps_executed", "terminated", "truncated"):
        try:
            if key == "status":
                info[key] = str(res.status)
            elif key == "steps_executed":
                info[key] = int(res.steps_executed)
            elif key == "terminated":
                info[key] = bool(res.terminated)
            else:
                info[key] = bool(res.truncated)
        except Exception:
            pass
    try:
        if res.error is not None:
            info["error"] = str(res.error)[:300]
    except Exception:
        pass
    return info


motion_log = []


def track(label, res):
    info = ex_flags(res)
    info["label"] = label
    motion_log.append(info)
    if info.get("terminated") or info.get("truncated"):
        raise ValueError("EPISODE_OVER:" + label)
    return info


def goto(label, pos, quat, tol=0.01, steps=150):
    target = Pose([float(pos[0]), float(pos[1]), float(pos[2])], [float(c) for c in quat], base_frame)
    res = move_to_pose(target, tolerance=tol, max_steps=steps)
    info = track(label, res)
    return info


def hold(ticks, grip):
    st = get_robot_state()
    last = None
    for k in range(ticks):
        last = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=st.joint_positions["primary"],
            gripper_position=grip, embodiment=st.embodiment)}))
        over = False
        try:
            over = bool(last.terminated) or bool(last.truncated)
        except Exception:
            over = False
        if over:
            raise ValueError("EPISODE_OVER:hold")
    return last


def pick_convention(camx, gridx, depthx):
    woodx = class_pixels(gridx, "wood")
    samplex = woodx[::max(1, len(woodx) // 400)]
    scores = {}
    for cv in ("opencv", "opengl"):
        mm = cam_model(camx, depthx, cv)
        zz = [p[2] for p in deproject_many(mm, samplex)]
        if len(zz) > 10:
            scores[cv] = [quant(zz, 0.9) - quant(zz, 0.1), median(zz)]
    cz = float(camx.camera_pose.position[2])
    pickc = "opencv"
    belowc = [c for c in ("opencv", "opengl") if c in scores and scores[c][1] < cz - 0.05]
    if len(belowc) == 1:
        pickc = belowc[0]
    mm = cam_model(camx, depthx, pickc)
    tz = median([p[2] for p in deproject_many(mm, samplex)])
    return pickc, mm, tz, scores


def cube_target(gridx, mm):
    redx = largest_blob(gridx, "red")
    purx = largest_blob(gridx, "purple")
    cp = deproject_many(mm, redx)
    tp = deproject_many(mm, purx)
    if len(cp) < 5 or len(tp) < 3:
        return None
    cxy = [median([p[0] for p in cp]), median([p[1] for p in cp])]
    half = 0.5 * max(quant([p[0] for p in cp], 0.95) - quant([p[0] for p in cp], 0.05),
                     quant([p[1] for p in cp], 0.95) - quant([p[1] for p in cp], 0.05))
    half = min(0.035, max(0.012, half))
    txy = [median([p[0] for p in tp]), median([p[1] for p in tp])]
    return {"cube": cxy, "half": half, "target": txy, "cube_top": quant([p[2] for p in cp], 0.9)}


def tool_geom(gridx, mm, comp, kindx, tz):
    pix = [(u, v) for (v, u) in comp["pixels"]]
    if kindx == "hammer":
        apx = [p for p in pix if gridx[p[1]][p[0]] == "dark"]
    elif kindx == "ball":
        apx = [p for p in pix if gridx[p[1]][p[0]] == "yellow"]
    else:
        apx = [p for p in pix if gridx[p[1]][p[0]] == "teal"]
    apts = deproject_many(mm, apx)
    allp = deproject_many(mm, pix)
    axy = [median([p[0] for p in apts]), median([p[1] for p in apts])]
    atop = quant([p[2] for p in apts], 0.9)
    farp = None; fd = -1.0
    for p in allp:
        dd = ((p[0] - axy[0]) ** 2 + (p[1] - axy[1]) ** 2) ** 0.5
        if dd > fd and dd < 0.6:
            fd = dd; farp = p
    out = {"anchor": axy, "top": atop, "n3d": len(allp)}
    if farp is None or fd < 0.02:
        out["axis"] = 0.0; out["tip"] = [0.0, 0.0]; out["tip_len"] = 0.0
        return out
    au = (farp[0] - axy[0]) / fd; av = (farp[1] - axy[1]) / fd
    endreg = []
    for p in allp:
        along = (p[0] - axy[0]) * au + (p[1] - axy[1]) * av
        if along > 0.8 * fd:
            endreg.append((along, (p[0] - axy[0]) * (-av) + (p[1] - axy[1]) * au))
    lead = max(e[0] for e in endreg)
    lat = sum(e[1] for e in endreg) / float(len(endreg))
    ev = [au * lead - av * lat, av * lead + au * lat]
    out["axis"] = math.atan2(av, au)
    out["tip"] = ev
    out["tip_len"] = (ev[0] ** 2 + ev[1] ** 2) ** 0.5
    return out


def yaw_of(q):
    Rq = quat_to_mat(q)
    return math.atan2(Rq[1][0], Rq[0][0])


def to_frame(p, origin, ang):
    dx = p[0] - origin[0]; dy = p[1] - origin[1]
    c = math.cos(ang); s = math.sin(ang)
    return [c * dx + s * dy, -s * dx + c * dy]


def from_frame(r, origin, ang):
    c = math.cos(ang); s = math.sin(ang)
    return [origin[0] + c * r[0] - s * r[1], origin[1] + s * r[0] + c * r[1]]


# ------------------------------------------------------------------ setup
ctx = get_task_context()
report["language"] = ctx.language
try:
    report["task_metadata"] = dict(robomme_plus.get_task_metadata())
except Exception as exc:
    report["task_metadata_error"] = str(exc)
state0 = get_robot_state()
base_frame = state0.base_frame
home_pose = state0.end_effector_poses["primary"]
report["home_ee"] = {"pos": [float(c) for c in home_pose.position],
                     "quat": [float(c) for c in home_pose.quaternion_wxyz]}

# ------------------------------------------------------------------ video analysis
video_kind = None
video_comp = None
try:
    video = robomme_plus.get_conditioning_video()
    nvid = len(video)
    report["video_frames"] = nvid
    if nvid > 2:
        f0 = video[0].cameras[FRONT]
        H = len(f0.rgb); W = len(f0.rgb[0])
        g0 = class_grid(to_rows(f0.rgb), H, W)
        vcomps = components(g0, H, W)
        late_idx = [int(nvid * 0.6), int(nvid * 0.8), nvid - 1]
        lates = [class_grid(to_rows(video[k].cameras[FRONT].rgb), H, W) for k in late_idx]
        vsum = []
        best = None; best_mv = 0.0
        for c in vcomps:
            mv = max(moved_fraction(c, gl) for gl in lates)
            s = summarize(c); s["moved"] = round(mv, 3)
            vsum.append(s)
            if mv > best_mv:
                best_mv = mv; best = c
        report["video_components"] = vsum
        if best is not None and best_mv >= 0.3:
            video_comp = best
            video_kind = best["kind"]
        # ---- manner transfer: expert grasp + contact phase relative to tool / cube / target
        if video_comp is not None:
            vconv, Mv, vtable, vscores = pick_convention(f0, g0, to_rows(f0.depth_m))
            vct = cube_target(g0, Mv)
            vtool = tool_geom(g0, Mv, video_comp, video_kind, vtable)
            states = []
            for k in range(nvid):
                rs = video[k].robot_state
                pe = rs.end_effector_poses["primary"]
                states.append([float(rs.gripper_positions["primary"]), [float(c) for c in pe.position],
                               yaw_of(pe.quaternion_wxyz)])
            g_open = states[0][0]
            kg = None
            for k in range(1, nvid):
                if states[k][0] < 0.75 * g_open and states[k][1][2] < 0.12:
                    kg = k
                    break
            vplan = {"conv": vconv, "table_z": vtable, "ct": vct, "tool": vtool, "grasp_index": kg}
            if kg is not None and vct is not None:
                gz = states[kg][1][2]
                klift = None
                for k in range(kg, nvid):
                    if states[k][1][2] > gz + 0.08:
                        klift = k
                        break
                low = []
                if klift is not None:
                    started = False
                    gap = 0
                    for k in range(klift, nvid):
                        lowk = states[k][1][2] < gz + 0.03 and states[k][0] < 0.75 * g_open
                        if lowk:
                            low.append(k); started = True; gap = 0
                        elif started:
                            gap += 1
                            if gap > 3:
                                break
                vplan["lift_index"] = klift
                vplan["low_indices"] = [low[0], low[-1]] if low else None
                if len(low) >= 3:
                    cv0 = vct["cube"]; tv0 = vct["target"]
                    uang_v = math.atan2(tv0[1] - cv0[1], tv0[0] - cv0[0])
                    dist_v = ((tv0[0] - cv0[0]) ** 2 + (tv0[1] - cv0[1]) ** 2) ** 0.5
                    # grasp relative to tool frame
                    gp = states[kg][1]
                    vplan["grasp_rel"] = to_frame(gp, vtool["anchor"], vtool["axis"])
                    vplan["grasp_dz"] = gp[2] - vtable
                    vplan["grasp_yaw_rel"] = wrap(states[kg][2] - vtool["axis"])
                    nwp = min(10, len(low))
                    wps = []
                    for j in range(nwp):
                        k = low[int(round(j * (len(low) - 1) / float(max(1, nwp - 1))))]
                        frac = j / float(max(1, nwp - 1))
                        rel = to_frame(states[k][1], cv0, uang_v)
                        rel[0] -= frac * dist_v          # progress measured in units of the push distance
                        wps.append({"rel": rel, "frac": frac, "dz": states[k][1][2] - vtable,
                                    "yaw_rel": wrap(states[k][2] - uang_v)})
                    vplan["waypoints"] = wps
                    vplan["dist_v"] = dist_v
            report["video_plan"] = vplan
        # gripper-state trace of the video (for diagnosis)
        gtrace = []
        for k in range(0, nvid, max(1, nvid // 40)):
            rs = video[k].robot_state
            gtrace.append([round(float(video[k].timestamp_s), 2), round(float(rs.gripper_positions["primary"]), 3),
                           [round(float(c), 3) for c in rs.end_effector_poses["primary"].position]])
        report["video_gripper_trace"] = gtrace
except Exception as exc:
    report["video_error"] = str(exc)[:300]
report["video_kind"] = video_kind
note("video kind: " + str(video_kind))

# ------------------------------------------------------------------ live scene
obs = get_observation()
cam = obs.cameras[FRONT]
rgb_rows = to_rows(cam.rgb)
H = len(rgb_rows); W = len(rgb_rows[0])
grid = class_grid(rgb_rows, H, W)
depth_rows = to_rows(cam.depth_m)

# camera convention: choose the one that makes the table horizontal
wood = class_pixels(grid, "wood")
wood_sample = wood[::max(1, len(wood) // 400)]
conv_scores = {}
for conv in ("opencv", "opengl"):
    m = cam_model(cam, depth_rows, conv)
    pts = deproject_many(m, wood_sample)
    zs = [p[2] for p in pts]
    if len(zs) > 10:
        conv_scores[conv] = [quant(zs, 0.9) - quant(zs, 0.1), median(zs)]
report["convention_scores"] = conv_scores
cam_z = float(cam.camera_pose.position[2])
report["camera_z"] = cam_z
conv = "opencv"
below = [c for c in ("opencv", "opengl") if c in conv_scores and conv_scores[c][1] < cam_z - 0.05]
if len(below) == 1:
    conv = below[0]
elif len(below) == 2:
    if conv_scores["opengl"][0] < conv_scores["opencv"][0]:
        conv = "opengl"
report["convention"] = conv
M = cam_model(cam, depth_rows, conv)
table_z = median([p[2] for p in deproject_many(M, wood_sample)])
report["table_z"] = table_z

# cube and target
red = largest_blob(grid, "red")
purple = largest_blob(grid, "purple")
report["red_px"] = len(red); report["purple_px"] = len(purple)
cube_pts = deproject_many(M, red)
targ_pts = deproject_many(M, purple)
if len(cube_pts) < 5 or len(targ_pts) < 3:
    raise ValueError("cube or target not visible")
cube_xy = [median([p[0] for p in cube_pts]), median([p[1] for p in cube_pts])]
cube_top = quant([p[2] for p in cube_pts], 0.9)
cube_half = 0.5 * max(quant([p[0] for p in cube_pts], 0.95) - quant([p[0] for p in cube_pts], 0.05),
                      quant([p[1] for p in cube_pts], 0.95) - quant([p[1] for p in cube_pts], 0.05))
cube_half = min(0.035, max(0.012, cube_half))
targ_xy = [median([p[0] for p in targ_pts]), median([p[1] for p in targ_pts])]
report["cube"] = {"xy": cube_xy, "top": cube_top, "half": cube_half}
report["target"] = {"xy": targ_xy}
# cross-check convention with SAM localisation of the cube (diagnostic only)
try:
    loc = localize_object("red cube", camera_name=FRONT, target_frame=base_frame)
    if loc.ok:
        report["sam_cube_center"] = [float(c) for c in loc.geometry.pose.position]
except Exception as exc:
    report["sam_cube_error"] = str(exc)[:200]

live_comps = components(grid, H, W)
report["live_components"] = [summarize(c) for c in live_comps]

chosen = None
TEAL_FAMILY = ("stick", "hook", "spatula", "teal_unknown")
if video_kind in TEAL_FAMILY and video_comp["shape"] is not None:
    best_d = None
    vs = video_comp["shape"]
    for c in live_comps:
        if c["kind"] not in TEAL_FAMILY or c["shape"] is None:
            continue
        d = 2.0 * abs(c["shape"]["ratio"] - vs["ratio"]) + abs(c["shape"]["density"] - vs["density"])
        if best_d is None or d < best_d:
            best_d = d; chosen = c
    report["teal_match_distance"] = best_d
if chosen is None and video_kind is not None:
    same = [c for c in live_comps if c["kind"] == video_kind]
    if same:
        for c in same:
            if chosen is None or c["n"] > chosen["n"]:
                chosen = c
if chosen is None and video_comp is not None and video_comp["shape"] is not None:
    best_d = None
    for c in live_comps:
        if c["shape"] is None:
            continue
        d = abs(c["shape"]["ratio"] - video_comp["shape"]["ratio"]) + abs(c["shape"]["density"] - video_comp["shape"]["density"])
        if best_d is None or d < best_d:
            best_d = d; chosen = c
if chosen is None:
    teal_tools = [c for c in live_comps if c["kind"] in ("stick", "hook", "spatula")]
    if teal_tools:
        for c in teal_tools:
            if chosen is None or c["n"] > chosen["n"]:
                chosen = c
if chosen is None:
    raise ValueError("no tool candidate in live scene")
kind = chosen["kind"]
report["chosen"] = summarize(chosen)

# ------------------------------------------------------------------ orientation helpers
Rh = quat_to_mat(home_pose.quaternion_wxyz)
approach_z = Rh[2][2]
if approach_z < -0.9:
    down_q = [float(c) for c in home_pose.quaternion_wxyz]
else:
    down_q = [0.0, 1.0, 0.0, 0.0]
Rd = quat_to_mat(down_q)
finger_angle = math.atan2(Rd[1][1], Rd[0][1])   # hand y axis (finger closing) in world xy
report["down_q"] = down_q
report["finger_angle_deg"] = round(math.degrees(finger_angle), 1)


def grasp_quat(axis_angle, flip):
    # fingers perpendicular to the tool axis
    delta = wrap(axis_angle + math.pi / 2.0 - finger_angle)
    if delta > math.pi / 2:
        delta -= math.pi
    if delta < -math.pi / 2:
        delta += math.pi
    if flip:
        delta = wrap(delta + math.pi)
    return yaw_quat(delta), delta


def pts_of(pix_uv):
    return deproject_many(M, pix_uv)



# ------------------------------------------------------------------ v04 helpers
# v04 changes (from v03 ep0 evidence, published attempt_004):
#  * in-hand tool measurement uses background subtraction against the frame-0 depth
#    (v03's lift measurement was contaminated by the static ramp's white texture -> wrong
#    direction, wrong height, z_low 0.093 so the hook passed over the cube);
#  * an obstacle height map from the frame-0 RGB-D (everything except table, cube, target,
#    the chosen tool and the high robot) is used to reject sweeps whose measured tool
#    points or hand would hit an obstacle (v03 pass 1 lowered the hand onto the ramp);
#  * a sweep may be split into two legs through an intermediate cube position when no
#    straight sweep is reachable and collision-free; each leg is re-planned from the
#    re-observed cube;
#  * a sweep is aborted when its lowering or two consecutive segments fail (v03 lost
#    ~850 steps to repeated 120-step timeouts).

DOWN_YAW = yaw_of(down_q)
R_MIN = 0.33
R_MAX = 0.76
SHAFT_H = 0.030
used_steps = [0]
depth0 = depth_rows
plan_log = []


def qyaw(psi):
    return quat_mul(yaw_quat(wrap(psi - DOWN_YAW)), down_q)


def ee_now():
    st = get_robot_state()
    pe = st.end_effector_poses["primary"]
    return [float(c) for c in pe.position], yaw_of(pe.quaternion_wxyz), float(st.gripper_positions["primary"])


def gomove(label, pos, quat, tol=0.01, steps=150):
    info = goto(label, pos, quat, tol, steps)
    used_steps[0] += int(info.get("steps_executed", 0) or 0)
    return info


def ik_ok(pos, quat):
    try:
        ik = solve_ik(Pose([float(pos[0]), float(pos[1]), float(pos[2])], [float(c) for c in quat], base_frame))
        return bool(ik.ok)
    except Exception:
        return False


# ---- obstacle height map (1 cm cells) from the frame-0 observation
tool_pix = {}
for (v, u) in chosen["pixels"]:
    for dv in (-2, -1, 0, 1, 2):
        for du in (-2, -1, 0, 1, 2):
            tool_pix[(u + du, v + dv)] = True
obst = {}
for v in range(0, H, 1):
    row = grid[v]
    for u in range(0, W, 1):
        c = row[u]
        if c in ("wood", "red", "purple") or (u, v) in tool_pix:
            continue
        p = deproject(M, u, v)
        if p is None:
            continue
        if p[2] < table_z + 0.012 or p[2] > table_z + 0.25:
            continue
        if abs(p[0] - cube_xy[0]) < cube_half + 0.015 and abs(p[1] - cube_xy[1]) < cube_half + 0.015:
            continue
        key = (int(math.floor(p[0] * 100.0)), int(math.floor(p[1] * 100.0)))
        if obst.get(key, -1.0) < p[2]:
            obst[key] = p[2]
report["obstacle_cells"] = len(obst)
coarse = {}
for key in obst:
    ck = "%d,%d" % (key[0] // 2, key[1] // 2)
    hv = round(obst[key] - table_z, 3)
    if coarse.get(ck, -1.0) < hv:
        coarse[ck] = hv
report["obstacle_map_2cm"] = coarse


def obst_h(x, y, rad_cells=1):
    kx = int(math.floor(x * 100.0)); ky = int(math.floor(y * 100.0))
    h = -1.0
    for dx in range(-rad_cells, rad_cells + 1):
        for dy in range(-rad_cells, rad_cells + 1):
            hv = obst.get((kx + dx, ky + dy))
            if hv is not None and hv > h:
                h = hv
    return h


def observe_cube():
    o = get_observation()
    cm = o.cameras[FRONT]
    g = class_grid(to_rows(cm.rgb), H, W)
    mm = cam_model(cm, to_rows(cm.depth_m), conv)
    rp = deproject_many(mm, largest_blob(g, "red"))
    rp = [p for p in rp if p[2] < table_z + 0.08]
    if len(rp) < 5:
        return None
    return [median([p[0] for p in rp]), median([p[1] for p in rp])]


def measure_tool(kindx):
    """Held tool measured in the live front RGB-D relative to the EE (pixels that moved since frame 0)."""
    ee, yaw, grip = ee_now()
    o = get_observation()
    cm = o.cameras[FRONT]
    g = class_grid(to_rows(cm.rgb), H, W)
    drows = to_rows(cm.depth_m)
    mm = cam_model(cm, drows, conv)
    if kindx == "hammer":
        cls = ("tan", "dark")
    else:
        cls = ("teal", "grey", "pink")
    pix = []
    for v in range(H):
        row = g[v]
        drow = drows[v]
        d0row = depth0[v]
        for u in range(W):
            if row[u] in cls:
                dn = float(drow[u]); d0 = float(d0row[u])
                if dn > 0.0 and (d0 <= 0.0 or dn < d0 - 0.02):
                    pix.append((u, v))
    pts = []
    for p in deproject_many(mm, pix):
        dxy = ((p[0] - ee[0]) ** 2 + (p[1] - ee[1]) ** 2) ** 0.5
        if 0.05 <= dxy <= 0.50 and ee[2] - 0.10 <= p[2] <= ee[2] + 0.03:
            pts.append(p)
    out = {"ee": ee, "yaw": yaw, "grip": grip, "n": len(pts)}
    if len(pts) < 15:
        return out
    n = float(len(pts))
    mx = sum(p[0] for p in pts) / n; my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts) / n
    syy = sum((p[1] - my) ** 2 for p in pts) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts) / n
    th = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    dx = math.cos(th); dy = math.sin(th)
    if (mx - ee[0]) * dx + (my - ee[1]) * dy < 0:
        dx = -dx; dy = -dy
    th = math.atan2(dy, dx)
    al = []
    for p in pts:
        a = (p[0] - ee[0]) * dx + (p[1] - ee[1]) * dy
        l = -(p[0] - ee[0]) * dy + (p[1] - ee[1]) * dx
        al.append([a, l, p[2] - ee[2]])
    amax = quant([e[0] for e in al], 0.98)
    mid = [e for e in al if 0.04 <= e[0] <= 0.75 * amax]
    if len(mid) < 5:
        mid = al
    out["dir"] = th
    out["rel"] = wrap(th - yaw)
    out["len"] = amax
    out["lat"] = median([e[1] for e in mid])
    out["dz_min"] = quant([e[2] for e in al], 0.03)
    ma = sum(e[0] for e in mid) / float(len(mid)); mz = sum(e[2] for e in mid) / float(len(mid))
    vaa = sum((e[0] - ma) ** 2 for e in mid)
    slope = 0.0
    if vaa > 1e-6:
        slope = sum((e[0] - ma) * (e[2] - mz) for e in mid) / vaa
    slope = max(-0.3, min(0.3, slope))
    out["dz_a0"] = mz - slope * ma
    out["dz_slope"] = slope
    # tool model in the shaft frame (along, lateral, dz) for collision checks
    stepk = max(1, len(al) // 70)
    out["model"] = [[round(e[0], 4), round(e[1], 4), round(e[2], 4)] for e in al[::stepk] if e[0] <= amax + 0.01]
    return out


def default_model(meas):
    pts = []
    k = 0.03
    while k <= meas["len"]:
        pts.append([k, meas["lat"], meas["dz_a0"] + meas["dz_slope"] * k])
        k += 0.02
    return pts


def rotate_to(label, pos, psi_target, max_step_deg=30.0):
    ee, yaw, grip = ee_now()
    delta = wrap(psi_target - yaw)
    nrot = max(1, int(abs(math.degrees(delta)) / max_step_deg + 0.999))
    for k in range(1, nrot + 1):
        gomove("%s_%d" % (label, k), pos, qyaw(yaw + delta * k / float(nrot)), 0.02, 120)
    return delta


def low_z(meas, a):
    dz_a = meas["dz_a0"] + meas["dz_slope"] * a
    z = table_z + SHAFT_H - dz_a
    z = max(z, table_z + 0.006 - meas["dz_min"])
    z = max(z, table_z + 0.014)
    return z


def leg_collisions(meas, dang, sx, sy, ux, uy, travel, zl):
    """Count obstacle hits of the measured tool points and the hand along a straight sweep."""
    model = meas.get("model") or default_model(meas)
    dx = math.cos(dang); dy = math.sin(dang)
    hits = 0
    nt = max(1, int(travel / 0.03 + 0.999))
    for i in range(nt + 1):
        f = travel * i / float(nt)
        ex = sx + ux * f; ey = sy + uy * f
        for e in model:
            px = ex + dx * e[0] - dy * e[1]
            py = ey + dy * e[0] + dx * e[1]
            pz = zl + e[2]
            if obst_h(px, py, 1) > pz - 0.008:
                hits += 1
        # hand / fingers footprint
        for (ox, oy) in ((0.0, 0.0), (0.035, 0.0), (-0.035, 0.0), (0.0, 0.035), (0.0, -0.035)):
            if obst_h(ex + ox, ey + oy, 1) > zl - 0.012:
                hits += 1
    return hits


def leg_candidates(meas, cxy, txy):
    ux = txy[0] - cxy[0]; uy = txy[1] - cxy[1]
    dist = (ux * ux + uy * uy) ** 0.5
    ux /= max(dist, 1e-6); uy /= max(dist, 1e-6)
    pang = math.atan2(uy, ux)
    L = meas["len"]
    out = []
    for side in (1.0, -1.0):
        for skew_deg in (0, 12, -12, 24, -24):
            dang = pang + side * math.pi / 2.0 + math.radians(skew_deg)
            dx = math.cos(dang); dy = math.sin(dang)
            cosang = abs(dx * ux + dy * uy)
            back = cube_half + 0.03 + cube_half * cosang
            cx = cxy[0] - ux * back; cy = cxy[1] - uy * back
            travel = dist + 0.020 + cube_half * cosang
            a = 0.08
            while a <= L - 0.05:
                sx = cx - (a * dx - meas["lat"] * dy)
                sy = cy - (a * dy + meas["lat"] * dx)
                ex = sx + ux * travel; ey = sy + uy * travel
                rs = (sx * sx + sy * sy) ** 0.5
                re = (ex * ex + ey * ey) ** 0.5
                margin = min(rs - R_MIN, R_MAX - rs, re - R_MIN, R_MAX - re)
                if margin >= 0.0:
                    psi = wrap(dang - meas["rel"])
                    zl = low_z(meas, a)
                    hits = leg_collisions(meas, dang, sx, sy, ux, uy, travel, zl)
                    out.append({"side": side, "skew": skew_deg, "a": a, "dang": dang, "psi": psi, "start": [sx, sy],
                                "end": [ex, ey], "margin": margin, "hits": hits, "zl": zl, "u": [ux, uy],
                                "dist": dist, "travel": travel, "rot": abs(wrap(psi - meas["yaw"]))})
                a += 0.03
    return out


def cand_cost(c):
    return 10.0 * c["hits"] + abs(c["skew"]) / 12.0 + 0.8 * c["rot"] + 2.0 * max(0.0, 0.05 - c["margin"])


def best_leg(meas, cxy, txy, need_ik=True):
    cands = leg_candidates(meas, cxy, txy)
    free = [c for c in cands if c["hits"] == 0]
    ranked = sorted(free, key=cand_cost)
    tried = 0
    for c in ranked:
        if not need_ik:
            return c
        tried += 1
        if tried > 8:
            break
        q = qyaw(c["psi"])
        if ik_ok([c["start"][0], c["start"][1], c["zl"] + 0.10], q) and ik_ok([c["start"][0], c["start"][1], c["zl"]], q) \
                and ik_ok([c["end"][0], c["end"][1], c["zl"]], q):
            return c
    return None


def plan_route(meas, cxy, txy):
    """Straight sweep if possible, else two legs through an intermediate cube position."""
    direct = best_leg(meas, cxy, txy)
    log = {"direct": direct is not None}
    if direct is not None:
        log["choice"] = "direct"
        plan_log.append(log)
        return [txy], direct
    ux = txy[0] - cxy[0]; uy = txy[1] - cxy[1]
    dist = (ux * ux + uy * uy) ** 0.5
    pang = math.atan2(uy, ux)
    best = None
    tried = []
    for dd in (30, -30, 45, -45, 60, -60):
        dl = math.radians(dd)
        l1 = dist * math.cos(dl)
        P = [cxy[0] + l1 * math.cos(pang + dl), cxy[1] + l1 * math.sin(pang + dl)]
        if obst_h(P[0], P[1], 2) > table_z + 0.012:
            tried.append([dd, "P blocked"])
            continue
        c1 = best_leg(meas, cxy, P, False)
        if c1 is None:
            tried.append([dd, "leg1 none"])
            continue
        # leg 2 evaluated with the tool rotated as needed (same in-hand geometry)
        m2 = dict(meas)
        m2["yaw"] = c1["psi"]
        c2 = best_leg(m2, P, txy, False)
        if c2 is None:
            tried.append([dd, "leg2 none"])
            continue
        q1 = qyaw(c1["psi"])
        ok1 = ik_ok([c1["start"][0], c1["start"][1], c1["zl"] + 0.10], q1) and ik_ok([c1["end"][0], c1["end"][1], c1["zl"]], q1)
        q2 = qyaw(c2["psi"])
        ok2 = ik_ok([c2["start"][0], c2["start"][1], c2["zl"]], q2) and ik_ok([c2["end"][0], c2["end"][1], c2["zl"]], q2)
        cost = cand_cost(c1) + cand_cost(c2) + abs(dd) / 30.0
        tried.append([dd, round(cost, 2), ok1, ok2])
        if ok1 and ok2 and (best is None or cost < best[0]):
            best = [cost, P, c1]
    log["two_leg"] = tried
    if best is not None:
        log["choice"] = "two_leg"
        log["P"] = best[1]
        plan_log.append(log)
        return [best[1], txy], best[2]
    # last resort: least-colliding straight sweep
    cands = leg_candidates(meas, cxy, txy)
    if not cands:
        log["choice"] = "none"
        plan_log.append(log)
        return [txy], None
    c = sorted(cands, key=cand_cost)[0]
    log["choice"] = "least_colliding"
    log["hits"] = c["hits"]
    plan_log.append(log)
    return [txy], c


def run_sweep(label, c, q):
    sx, sy = c["start"]
    ux, uy = c["u"]
    zl = c["zl"]
    r1 = gomove(label + "_above", [sx, sy, zl + 0.10], q, 0.02, 200)
    r2 = gomove(label + "_lower1", [sx, sy, zl + 0.04], q, 0.012, 90)
    r3 = gomove(label + "_lower2", [sx, sy, zl], q, 0.01, 90)
    if not r3.get("ok"):
        ee, yaw, grip = ee_now()
        if abs(ee[2] - zl) > 0.02:
            gomove(label + "_abort_up", [sx, sy, zl + 0.12], q, 0.03, 100)
            return "lower_failed"
    nseg = max(1, int(c["travel"] / 0.02 + 0.999))
    fails = 0
    for k in range(1, nseg + 1):
        f = c["travel"] * k / float(nseg)
        rk = gomove("%s_%d" % (label, k), [sx + ux * f, sy + uy * f, zl], q, 0.01, 50)
        if rk.get("ok"):
            fails = 0
        else:
            fails += 1
            if fails >= 2:
                break
    ee, yaw, grip = ee_now()
    gomove(label + "_back", [ee[0] - ux * 0.03, ee[1] - uy * 0.03, zl + 0.004], q, 0.015, 50)
    gomove(label + "_up", [ee[0] - ux * 0.03, ee[1] - uy * 0.03, zl + 0.12], q, 0.03, 120)
    if fails >= 2:
        return "segments_failed"
    return "done"


def grasp_tool():
    """Expert-transferred grasp (tool frame) if IK-valid, else top-down grasp across the anchor."""
    ltool = tool_geom(grid, M, chosen, kind, table_z)
    vplan = report.get("video_plan")
    plans = []
    if vplan is not None and vplan.get("grasp_rel") is not None and (video_kind == kind or (kind in TEAL_FAMILY and video_kind in TEAL_FAMILY)):
        g_xy = from_frame(vplan["grasp_rel"], ltool["anchor"], ltool["axis"])
        g_z = max(table_z + 0.006, table_z + vplan["grasp_dz"])
        for flip in (0.0, math.pi):
            plans.append(["transfer", g_xy, g_z, ltool["axis"] + vplan["grasp_yaw_rel"] + flip])
    a_top = ltool["top"]
    g_z2 = max(table_z + 0.008, a_top - 0.012)
    # fingers (hand y axis, finger_angle from the EE yaw) across the handle; along the axis for the hammer head
    base_yaw = ltool["axis"] + math.pi / 2.0 - finger_angle
    if kind == "hammer":
        base_yaw = ltool["axis"] - finger_angle
    for flip in (0.0, math.pi):
        plans.append(["geometric", ltool["anchor"], g_z2, base_yaw + flip])
    glog = []
    pick = None
    for pl in plans:
        q = qyaw(pl[3])
        ok1 = ik_ok([pl[1][0], pl[1][1], pl[2] + 0.12], q)
        ok2 = ik_ok([pl[1][0], pl[1][1], pl[2]], q)
        glog.append([pl[0], round(math.degrees(wrap(pl[3])), 1), ok1, ok2])
        if ok1 and ok2:
            pick = pl
            break
    report["grasp_candidates"] = glog
    if pick is None:
        raise ValueError("no IK-valid grasp")
    q = qyaw(pick[3])
    report["grasp"] = {"kind": pick[0], "xy": pick[1], "z": pick[2], "yaw_deg": round(math.degrees(wrap(pick[3])), 1)}
    open_gripper()
    gomove("g_pre", [pick[1][0], pick[1][1], pick[2] + 0.12], q, 0.015, 200)
    gomove("g_grasp", [pick[1][0], pick[1][1], pick[2]], q, 0.008, 150)
    track("g_close", close_gripper())
    ee, yaw, grip = ee_now()
    report["grip_after_close"] = grip
    gomove("g_lift", [pick[1][0], pick[1][1], table_z + 0.22], q, 0.02, 200)
    return ltool


def fallback_meas(ltool):
    ee, yaw, grip = ee_now()
    tv = ltool["tip"]
    m = {"ee": ee, "yaw": yaw, "grip": grip, "n": 0, "dir": math.atan2(tv[1], tv[0]),
         "len": max(0.15, ltool["tip_len"] * 0.95), "lat": 0.0, "dz_min": -0.02, "dz_a0": -0.007, "dz_slope": 0.0}
    m["rel"] = wrap(m["dir"] - yaw)
    note("tool not re-observed; using frame-0 geometry")
    return m


def floor_len(m, full_len):
    if m.get("dir") is not None and m["len"] < 0.7 * full_len:
        m["len"] = 0.95 * full_len
        m["model"] = (m.get("model") or []) + default_model(m)
    return m


def strip(m):
    return {k: m[k] for k in m if k != "model"}


steps_info = {}
try:
    if kind == "ball":
        bpts = pts_of([p for p in [(u, v) for (v, u) in chosen["pixels"]] if grid[p[1]][p[0]] == "yellow"])
        bxy = [median([p[0] for p in bpts]), median([p[1] for p in bpts])]
        btop = quant([p[2] for p in bpts], 0.95)
        brad = max(0.01, min(0.04, 0.5 * (btop - table_z)))
        bc_z = btop - brad
        ramp = largest_blob(grid, "blue")
        rpts = pts_of(ramp)
        rz_hi = quant([p[2] for p in rpts], 0.92)
        top_pts = [p for p in rpts if p[2] >= rz_hi]
        rc_all = centroid(rpts)
        rtop = centroid(top_pts)
        report["ball"] = {"xy": bxy, "top": btop, "rad": brad}
        report["ramp"] = {"top": rtop, "centroid": rc_all, "n": len(rpts)}
        q = quat_mul(yaw_quat(0.0), down_q)
        open_gripper()
        goto("ball_pre", [bxy[0], bxy[1], btop + 0.12], q, 0.015, 200)
        goto("ball_grasp", [bxy[0], bxy[1], bc_z], q, 0.01, 150)
        track("ball_close", close_gripper())
        goto("ball_lift", [bxy[0], bxy[1], btop + 0.20], q, 0.02, 200)
        dx = rc_all[0] - rtop[0]; dy = rc_all[1] - rtop[1]
        dn = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        rel = [rtop[0] + 0.02 * dx / dn, rtop[1] + 0.02 * dy / dn]
        goto("ramp_above", [rel[0], rel[1], rtop[2] + 0.15], q, 0.02, 250)
        goto("ramp_release", [rel[0], rel[1], rtop[2] + brad + 0.02], q, 0.015, 200)
        track("ball_open", open_gripper())
        hold(80, 1.0)
        goto("ramp_retreat", [rel[0], rel[1], rtop[2] + 0.20], q, 0.03, 150)
        hold(60, 1.0)
    else:
        ltool = grasp_tool()
        meas = measure_tool(kind)
        report["meas_lift"] = strip(meas)
        full_len = ltool["tip_len"]
        if meas.get("dir") is None or meas["len"] < 0.7 * full_len:
            # tool (partly) outside the front view: turn it towards the cube, then re-measure
            guess = fallback_meas(ltool)
            rel_guess = meas["rel"] if meas.get("dir") is not None else guess["rel"]
            ee, yaw, grip = ee_now()
            want = math.atan2(cube_xy[1] - ee[1], cube_xy[0] - ee[0])
            rotate_to("view", [ee[0], ee[1], ee[2]], wrap(want - rel_guess))
            m1 = measure_tool(kind)
            report["meas_view"] = strip(m1)
            if m1.get("dir") is not None:
                meas = m1
            elif meas.get("dir") is None:
                meas = fallback_meas(ltool)
                meas["rel"] = rel_guess
                meas["dir"] = wrap(meas["yaw"] + rel_guess)
        if meas["len"] < 0.7 * full_len:
            note("measured tool length %.3f < frame-0 length %.3f; using frame-0 length" % (meas["len"], full_len))
            meas["len"] = 0.95 * full_len
            meas["model"] = (meas.get("model") or []) + default_model(meas)
        cxy = cube_xy
        sweeps = []
        nleg = 0
        for rnd in range(3):
            goals, c = plan_route(meas, cxy, targ_xy)
            report["plans"] = plan_log
            if c is None:
                note("no sweep plan")
                break
            planned_dir = c["dang"]
            hpos = [meas["ee"][0], meas["ee"][1], max(meas["ee"][2], table_z + 0.20)]
            rotate_to("rot%d" % nleg, hpos, c["psi"])
            m2 = floor_len(measure_tool(kind), full_len)
            fix = None
            if m2.get("dir") is not None:
                meas = m2
                err = wrap(planned_dir - meas["dir"])
                if abs(err) > math.radians(10.0):
                    fix = round(math.degrees(err), 1)
                    rotate_to("rfix%d" % nleg, hpos, wrap(meas["yaw"] + err))
                    m3 = floor_len(measure_tool(kind), full_len)
                    if m3.get("dir") is not None:
                        meas = m3
            else:
                ee, yaw, grip = ee_now()
                meas["yaw"] = yaw; meas["dir"] = wrap(yaw + meas["rel"]); meas["ee"] = ee
            # re-solve this leg's start for the measured tool, keeping side / contact point
            leg_goal = goals[0]
            cands = leg_candidates(meas, cxy, leg_goal)
            pick = None
            for k in cands:
                if k["side"] == c["side"] and abs(wrap(k["dang"] - meas["dir"])) < math.radians(7.0):
                    if pick is None or (k["hits"], abs(k["a"] - c["a"])) < (pick["hits"], abs(pick["a"] - c["a"])):
                        pick = k
            if pick is None:
                # translate the planned candidate using the measured direction
                pick = dict(c)
            pick = dict(pick)
            # use exactly the measured direction: recompute start for the achieved yaw
            dxm = math.cos(meas["dir"]); dym = math.sin(meas["dir"])
            ux, uy = pick["u"]
            cosang = abs(dxm * ux + dym * uy)
            back = cube_half + 0.03 + cube_half * cosang
            cx = cxy[0] - ux * back; cy = cxy[1] - uy * back
            pick["start"] = [cx - (pick["a"] * dxm - meas["lat"] * dym), cy - (pick["a"] * dym + meas["lat"] * dxm)]
            pick["travel"] = pick["dist"] + 0.020 + cube_half * cosang
            pick["zl"] = low_z(meas, pick["a"])
            pick["hits"] = leg_collisions(meas, meas["dir"], pick["start"][0], pick["start"][1], ux, uy, pick["travel"], pick["zl"])
            ee, yaw, grip = ee_now()
            q = qyaw(yaw)
            rec = {"leg": nleg, "round": rnd, "cube": cxy, "goal": leg_goal, "n_goals": len(goals), "side": pick["side"],
                   "skew": pick["skew"], "a": pick["a"], "start": pick["start"], "travel": pick["travel"],
                   "z_low": pick["zl"], "hits": pick["hits"], "rot_fix_deg": fix,
                   "shaft_vs_push_deg": round(math.degrees(math.acos(min(1.0, cosang))), 1), "meas": strip(meas)}
            sweeps.append(rec)
            report["sweeps"] = sweeps
            status = run_sweep("sw%d" % nleg, pick, q)
            nleg += 1
            rec["status"] = status
            newc = observe_cube()
            rec["cube_after"] = newc
            report["sweeps"] = sweeps
            if newc is None:
                break
            dres = ((newc[0] - targ_xy[0]) ** 2 + (newc[1] - targ_xy[1]) ** 2) ** 0.5
            rec["residual"] = dres
            if dres < 0.012 or used_steps[0] > 1000:
                break
            cxy = newc
            m4 = floor_len(measure_tool(kind), full_len)
            if m4.get("dir") is not None:
                meas = m4
            else:
                ee, yaw, grip = ee_now()
                meas["yaw"] = yaw; meas["dir"] = wrap(yaw + meas["rel"]); meas["ee"] = ee
        hold(20, 0.0)
except Exception as exc:
    if str(exc).startswith("EPISODE_OVER"):
        note("episode ended during " + str(exc))
        report["episode_over"] = True
    else:
        report["error"] = str(exc)[:400]

report["motions"] = motion_log
report["used_steps"] = used_steps[0]
# final diagnostics (guarded)
try:
    if report.get("episode_over"):
        raise ValueError("skip final observation after terminal episode")
    fobs = get_observation()
    fgrid = class_grid(to_rows(fobs.cameras[FRONT].rgb), H, W)
    fred = largest_blob(fgrid, "red")
    fM = cam_model(fobs.cameras[FRONT], to_rows(fobs.cameras[FRONT].depth_m), conv)
    fpts = deproject_many(fM, fred)
    if fpts:
        fxy = [median([p[0] for p in fpts]), median([p[1] for p in fpts])]
        report["final_cube_xy"] = fxy
        report["final_cube_to_target_m"] = ((fxy[0] - targ_xy[0]) ** 2 + (fxy[1] - targ_xy[1]) ** 2) ** 0.5
except Exception as exc:
    report["final_obs_error"] = str(exc)[:200]

result = report
