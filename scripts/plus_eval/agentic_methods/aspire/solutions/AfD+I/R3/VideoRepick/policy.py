import math

# VideoRepick v02 (v01 + camera-convention/box-less fixes) -- identify the object picked in the conditioning video (robot-state gripper
# close event at the end-effector xy), track every object through the shuffle in world xy with a
# fixed-N k-means over depth+colour object cells from BOTH cameras, then pick/place the tracked
# object twice and press the button.
# All task-dependent values are derived at runtime from the episode's goal, conditioning video
# and live observations.

report = {"version": "v02", "stage": "start", "errors": []}


def note_error(where, exc):
    report["errors"].append(where + ": " + str(exc)[:300])


def as_list(arr):
    try:
        return arr.tolist()
    except Exception:
        return [list(row) for row in arr]


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


# ---------------------------------------------------------------- quaternion / camera helpers
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def quat_rot(q, v):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    r00 = 1 - 2 * (y * y + z * z); r01 = 2 * (x * y - w * z); r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z); r11 = 1 - 2 * (x * x + z * z); r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y); r21 = 2 * (y * z + w * x); r22 = 1 - 2 * (x * x + y * y)
    return [r00 * v[0] + r01 * v[1] + r02 * v[2],
            r10 * v[0] + r11 * v[1] + r12 * v[2],
            r20 * v[0] + r21 * v[1] + r22 * v[2]]


def quat_conj(q):
    return [q[0], -q[1], -q[2], -q[3]]


def cam_params(cam):
    k = as_list(cam.intrinsics)
    pose = cam.camera_pose
    return {"fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]), "cy": float(k[1][2]),
            "t": [float(c) for c in pose.position], "q": [float(c) for c in pose.quaternion_wxyz]}


def project(cp, p, conv):
    rel = [p[0] - cp["t"][0], p[1] - cp["t"][1], p[2] - cp["t"][2]]
    c = quat_rot(quat_conj(cp["q"]), rel)
    if conv == "opengl":
        d = -c[2]
        if d <= 1e-6:
            return None
        return [cp["cx"] + cp["fx"] * c[0] / d, cp["cy"] - cp["fy"] * c[1] / d]
    if conv == "opencv":
        d = c[2]
        if d <= 1e-6:
            return None
        return [cp["cx"] + cp["fx"] * c[0] / d, cp["cy"] + cp["fy"] * c[1] / d]
    d = c[0]  # ros camera-link: x forward, y left, z up
    if d <= 1e-6:
        return None
    return [cp["cx"] - cp["fx"] * c[1] / d, cp["cy"] - cp["fy"] * c[2] / d]


def backproject(cp, u, v, depth, conv):
    xn = (u - cp["cx"]) / cp["fx"]
    yn = (v - cp["cy"]) / cp["fy"]
    if conv == "opengl":
        c = [xn * depth, -yn * depth, -depth]
    elif conv == "opencv":
        c = [xn * depth, yn * depth, depth]
    else:
        c = [depth, -xn * depth, -yn * depth]
    w = quat_rot(cp["q"], c)
    return [w[0] + cp["t"][0], w[1] + cp["t"][1], w[2] + cp["t"][2]]


# ---------------------------------------------------------------- table plane + blob detection
def solve3(m, r):
    def det(a):
        return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    d0 = det(m)
    if abs(d0) < 1e-18:
        return None
    out = []
    for col in range(3):
        mm = [list(row) for row in m]
        for row in range(3):
            mm[row][col] = r[row]
        out.append(det(mm) / d0)
    return out


def fit_table_plane(depth):
    """Fit inverse depth 1/d = a*u + b*v + c over the dominant (table) surface."""
    h = len(depth)
    w = len(depth[0])
    pts = []
    for v in range(4, h, 8):
        row = depth[v]
        for u in range(4, w, 8):
            d = float(row[u])
            if d > 0.05 and d < 10.0:
                pts.append((u, v, d))
    keep = pts
    coef = None
    for iteration in range(5):
        s = [[0.0] * 3 for i in range(3)]
        r = [0.0, 0.0, 0.0]
        for (u, v, d) in keep:
            vec = (u, v, 1.0)
            inv = 1.0 / d
            for i in range(3):
                r[i] += vec[i] * inv
                for j in range(3):
                    s[i][j] += vec[i] * vec[j]
        coef = solve3(s, r)
        if coef is None:
            return None
        res = []
        for (u, v, d) in pts:
            pd = 1.0 / max(1e-6, coef[0] * u + coef[1] * v + coef[2])
            res.append(abs(pd - d))
        med = quantile(res, 0.5)
        thr = max(0.004, 3.0 * med)
        keep = [pts[i] for i in range(len(pts)) if res[i] < thr]
        if len(keep) < 30:
            break
    return {"coef": coef, "inliers": len(keep), "samples": len(pts)}



DD_MIN = 0.008     # minimum depth step above the fitted table (generic, sensor-level)
DD_MAX = 0.10      # objects on the table; excludes a hovering arm
SAT_MIN_PX = 26      # colourful objects; excludes grey/white robot, button and fixtures


def detect_blobs(rgb, depth, plane):
    coef = plane["coef"]
    h = len(depth)
    w = len(depth[0])
    cells = {}
    for v in range(0, h, 2):
        drow = depth[v]
        crow = rgb[v]
        base = coef[1] * v + coef[2]
        for u in range(0, w, 2):
            d = drow[u]
            inv = coef[0] * u + base
            if inv <= 0:
                continue
            dd = 1.0 / inv - d
            if dd < DD_MIN or dd > DD_MAX:
                continue
            px = crow[u]
            mx = max(px[0], px[1], px[2])
            mn = min(px[0], px[1], px[2])
            if mx - mn < SAT_MIN_PX:
                continue
            cells[(u // 2, v // 2)] = 1
    blobs = []
    seen = {}
    for key in cells:
        if key in seen:
            continue
        stack = [key]
        seen[key] = 1
        members = []
        while len(stack) > 0:
            cur = stack.pop()
            members.append(cur)
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    nb = (cur[0] + du, cur[1] + dv)
                    if nb in cells and nb not in seen:
                        seen[nb] = 1
                        stack.append(nb)
        if len(members) < 3:
            continue
        su = sum(m[0] for m in members) * 2
        sv = sum(m[1] for m in members) * 2
        n = len(members)
        blobs.append({"u": su / n, "v": sv / n, "n": n,
                      "umin": min(m[0] for m in members) * 2,
                      "umax": max(m[0] for m in members) * 2,
                      "vmin": min(m[1] for m in members) * 2,
                      "vmax": max(m[1] for m in members) * 2})
    return blobs


# ---------------------------------------------------------------- world-space object cells (both cameras)
Z_MIN = 0.008      # cell must rise this far above the fitted table plane (sensor-level threshold)
Z_MAX = 0.10       # objects resting on / held just above the table; excludes the hovering arm
SAT_MIN = 26       # colourful objects; grey/white robot, button, fixtures measure sat < 20
STRIDE = 2
RAY_CACHE = {}


def ray_table(cam_name, cp, h, w, conv):
    """World-frame ray direction (per unit depth) for every sampled pixel, cached per camera pose."""
    key = cam_name + ":" + ",".join(str(round(c, 5)) for c in cp["t"] + cp["q"])
    if key in RAY_CACHE:
        return RAY_CACHE[key]
    rows = []
    for v in range(0, h, STRIDE):
        row = []
        yn = (v - cp["cy"]) / cp["fy"]
        for u in range(0, w, STRIDE):
            xn = (u - cp["cx"]) / cp["fx"]
            if conv == "opengl":
                c = [xn, -yn, -1.0]
            elif conv == "opencv":
                c = [xn, yn, 1.0]
            else:
                c = [1.0, -xn, -yn]
            row.append(quat_rot(cp["q"], c))
        rows.append(row)
    if len(RAY_CACHE) > 64:
        for k in list(RAY_CACHE.keys()):
            RAY_CACHE.pop(k)
    RAY_CACHE[key] = rows
    return rows


def object_cells(obs, cams, table_z, conv):
    """Return [(x, y, z, cam_index)] for saturated pixels standing Z_MIN..Z_MAX above the table."""
    out = []
    for ci in range(len(cams)):
        cam = obs.cameras[cams[ci]]
        cp = cam_params(cam)
        rgb = as_list(cam.rgb)
        depth = as_list(cam.depth_m)
        h = len(depth)
        w = len(depth[0])
        rays = ray_table(cams[ci], cp, h, w, conv)
        tx, ty, tz = cp["t"]
        zlo = table_z + Z_MIN
        zhi = table_z + Z_MAX
        for vi in range(len(rays)):
            v = vi * STRIDE
            drow = depth[v]
            crow = rgb[v]
            rrow = rays[vi]
            for ui in range(len(rrow)):
                u = ui * STRIDE
                d = drow[u]
                if d <= 0.02 or d > 5.0:
                    continue
                r = rrow[ui]
                z = tz + d * r[2]
                if z < zlo or z > zhi:
                    continue
                px = crow[u]
                if max(px[0], px[1], px[2]) - min(px[0], px[1], px[2]) < SAT_MIN:
                    continue
                out.append((tx + d * r[0], ty + d * r[1], z, ci))
    return out


def grid_components(cells, res):
    grid = {}
    for c in cells:
        key = (int(math.floor(c[0] / res)), int(math.floor(c[1] / res)))
        if key in grid:
            grid[key].append(c)
        else:
            grid[key] = [c]
    seen = {}
    comps = []
    for key in grid:
        if key in seen:
            continue
        seen[key] = 1
        stack = [key]
        members = []
        while len(stack) > 0:
            cur = stack.pop()
            members.extend(grid[cur])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cur[0] + dx, cur[1] + dy)
                    if nb in grid and nb not in seen:
                        seen[nb] = 1
                        stack.append(nb)
        if len(members) < 4:
            continue
        xs = [m[0] for m in members]
        ys = [m[1] for m in members]
        zs = [m[2] for m in members]
        x0, x1 = quantile(xs, 0.05), quantile(xs, 0.95)
        y0, y1 = quantile(ys, 0.05), quantile(ys, 0.95)
        comps.append({"x": 0.5 * (x0 + x1), "y": 0.5 * (y0 + y1), "n": len(members),
                      "ext": max(x1 - x0, y1 - y0), "ztop": quantile(zs, 0.95),
                      "cams": sorted(set(m[3] for m in members)), "members": members})
    return comps


# ---------------------------------------------------------------- multi-object tracker (world xy)
def dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def track_step_world(tracks, cells, r_obj, max_comp_ext):
    """Fixed-N k-means on world-xy object cells seeded at each track's constant-velocity
    prediction. Touching objects stay separate tracks; occluded tracks hold position."""
    est = [[t["x"] + t["vx"], t["y"] + t["vy"]] for t in tracks]
    r_assign2 = (2.0 * r_obj) ** 2
    counts = [0] * len(tracks)
    for iteration in range(4):
        sx = [0.0] * len(tracks)
        sy = [0.0] * len(tracks)
        counts = [0] * len(tracks)
        for c in cells:
            best = -1
            bd = r_assign2
            for ti in range(len(est)):
                d = (c[0] - est[ti][0]) ** 2 + (c[1] - est[ti][1]) ** 2
                if d < bd:
                    bd = d
                    best = ti
            if best >= 0:
                sx[best] += c[0]
                sy[best] += c[1]
                counts[best] += 1
        for ti in range(len(est)):
            if counts[ti] >= 3:
                est[ti] = [sx[ti] / counts[ti], sy[ti] / counts[ti]]
    for ti in range(len(tracks)):
        t = tracks[ti]
        if counts[ti] >= 3:
            nx, ny = est[ti]
            t["vx"] = 0.5 * t["vx"] + 0.5 * (nx - t["x"])
            t["vy"] = 0.5 * t["vy"] + 0.5 * (ny - t["y"])
            t["x"], t["y"] = nx, ny
            t["miss"] = 0
            t["n"] = counts[ti]
        else:
            t["miss"] += 1
            t["vx"] *= 0.5
            t["vy"] *= 0.5
            t["n"] = counts[ti]


def nearest_track(tracks, pt):
    best = None
    bd = None
    for ti in range(len(tracks)):
        d = dist2((tracks[ti]["x"], tracks[ti]["y"]), pt)
        if bd is None or d < bd:
            bd = d
            best = ti
    return best, (math.sqrt(bd) if bd is not None else None)



# ---------------------------------------------------------------- execution helpers
def exec_info(res):
    info = {}
    try:
        info["ok"] = bool(res.ok)
    except Exception:
        info["ok"] = None
    for name in ("status", "steps_executed", "terminated", "truncated"):
        try:
            if name == "status":
                info[name] = str(res.status)
            elif name == "steps_executed":
                info[name] = int(res.steps_executed)
            elif name == "terminated":
                info[name] = bool(res.terminated)
            else:
                info[name] = bool(res.truncated)
        except Exception:
            pass
    try:
        if res.error is not None:
            info["error"] = str(res.error)[:200]
    except Exception:
        pass
    return info


ENDED = {"flag": False}
MOTION_LOG = []


def run_motion(label, fn):
    if ENDED["flag"]:
        MOTION_LOG.append({"label": label, "skipped": "episode_ended"})
        return None
    try:
        res = fn()
    except Exception as exc:
        MOTION_LOG.append({"label": label, "exception": str(exc)[:200]})
        if "ended" in str(exc).lower():
            ENDED["flag"] = True
        return None
    info = exec_info(res)
    info["label"] = label
    try:
        info["t"] = float(res.final_observation.timestamp_s)
    except Exception:
        pass
    try:
        info["grip"] = float(res.final_observation.robot_state.gripper_positions["primary"])
    except Exception:
        pass
    MOTION_LOG.append(info)
    if info.get("terminated") or info.get("truncated"):
        ENDED["flag"] = True
    return res


def yaw_quat(psi):
    return [math.cos(psi / 2.0), 0.0, 0.0, math.sin(psi / 2.0)]


def min_area_yaw(xy):
    best = None
    for k in range(18):
        a = k * math.pi / 36.0
        ca, sa = math.cos(a), math.sin(a)
        ps = [p[0] * ca + p[1] * sa for p in xy]
        qs = [-p[0] * sa + p[1] * ca for p in xy]
        ext = (quantile(ps, 0.97) - quantile(ps, 0.03)) * (quantile(qs, 0.97) - quantile(qs, 0.03))
        if best is None or ext < best[0]:
            best = (ext, a)
    return best[1]


def wrap_near(angle, ref, period):
    while angle - ref > period / 2.0:
        angle -= period
    while angle - ref < -period / 2.0:
        angle += period
    return angle


# ================================================================ main
def main():
    base_frame = get_robot_state().base_frame
    ctx = get_task_context()
    report["goal"] = ctx.language
    try:
        meta = robomme_plus.get_task_metadata()
        report["meta"] = {"max_steps": meta.get("max_steps"), "control_hz": meta.get("control_hz"),
                          "sampling": str(meta.get("conditioning_video_sampling"))[:300]}
    except Exception as exc:
        note_error("meta", exc)

    live = get_observation()
    live_front = live.cameras["front"]
    live_cp = cam_params(live_front)
    live_rgb = as_list(live_front.rgb)
    live_depth = as_list(live_front.depth_m)
    live_plane = fit_table_plane(live_depth)
    report["live_plane"] = {"inliers": live_plane["inliers"], "samples": live_plane["samples"]}
    live_blobs = detect_blobs(live_rgb, live_depth, live_plane)
    report["live_front_blobs"] = [[round(b["u"], 1), round(b["v"], 1), b["n"]] for b in live_blobs]
    home_state = get_robot_state()
    home_pose = home_state.end_effector_poses["primary"]
    report["home_ee"] = [round(float(c), 4) for c in home_pose.position]
    report["home_q"] = [round(float(c), 4) for c in home_pose.quaternion_wxyz]

    # ------------------------------------------------ camera convention calibration (public API)
    calib = {}
    conv = "opencv"
    calib_err = {}
    if len(live_blobs) > 0:
        cb = sorted(live_blobs, key=lambda b: -b["n"])[0]
        calib["blob_px"] = [round(cb["u"], 1), round(cb["v"], 1)]
        try:
            seg = segment_points("front", [[int(round(cb["u"])), int(round(cb["v"]))]])
            if seg.ok and len(seg.segmentations) > 0:
                cand = seg.segmentations[0]
                cloud = mask_to_point_cloud(cand, "front", target_frame=base_frame)
                pts = cloud.points
                cen = [quantile([p[i] for p in pts], 0.5) for i in range(3)]
                calib["centroid3d"] = [round(c, 4) for c in cen]
                calib["api_z_q05"] = round(quantile([p[2] for p in pts], 0.05), 4)
                for cv in ("opencv", "opengl", "roslink"):
                    pr = project(live_cp, cen, cv)
                    if pr is None:
                        calib[cv] = None
                        continue
                    err = math.sqrt(dist2(pr, (cb["u"], cb["v"])))
                    calib[cv] = [round(pr[0], 1), round(pr[1], 1), round(err, 1)]
                    calib_err[cv] = err
        except Exception as exc:
            note_error("calib", exc)
    if len(calib_err) > 0:
        best_cv = sorted(calib_err.keys(), key=lambda k: calib_err[k])[0]
        if calib_err[best_cv] < 20.0:
            conv = best_cv
    calib["chosen"] = conv
    report["calib"] = calib

    # table height: back-project fitted-plane inliers of the live front image
    coef = live_plane["coef"]
    tz_list = []
    for v in range(8, 256, 16):
        for u in range(8, 256, 16):
            d = float(live_depth[v][u])
            pd = 1.0 / (coef[0] * u + coef[1] * v + coef[2])
            if abs(pd - d) < 0.003:
                tz_list.append(backproject(live_cp, u, v, d, conv)[2])
    table_z = quantile(tz_list, 0.5)
    report["table_z"] = round(table_z, 4)
    report["table_z_spread"] = [round(quantile(tz_list, 0.1), 4), round(quantile(tz_list, 0.9), 4)]
    cams = ["front", "wrist"]

    # ------------------------------------------------ conditioning video: pick event from robot state
    video = robomme_plus.get_conditioning_video()
    n_frames = len(video)
    report["video_frames"] = n_frames
    if n_frames == 0:
        report["stage"] = "no_video"
        raise ValueError("no conditioning video; cannot identify the object")
    grips = []
    ees = []
    times = []
    for fi in range(n_frames):
        rs = video[fi].robot_state
        grips.append(float(rs.gripper_positions["primary"]))
        ees.append([float(c) for c in rs.end_effector_poses["primary"].position])
        times.append(float(video[fi].timestamp_s))
    open_level = max(grips)
    report["grip_open_level"] = round(open_level, 3)
    events = []
    closed = False
    for fi in range(n_frames):
        if not closed and grips[fi] < open_level - 0.25:
            closed = True
            events.append(["close", fi])
        elif closed and grips[fi] > open_level - 0.1:
            closed = False
            events.append(["open", fi])
    report["grip_events"] = [[e[0], e[1], round(times[e[1]], 2), round(grips[e[1]], 3),
                              [round(c, 3) for c in ees[e[1]]]] for e in events]
    close_fi = None
    open_fi = None
    for e in events:
        if e[0] == "close" and close_fi is None:
            close_fi = e[1]
        if e[0] == "open" and close_fi is not None and open_fi is None and e[1] > close_fi:
            open_fi = e[1]
    if close_fi is None:
        report["stage"] = "no_pick_event"
        raise ValueError("no gripper-close event in conditioning video")
    # sanity: at the gripper-close event the tool is at object height, so the table must lie just below it
    ee_close_z = ees[close_fi][2]
    report["ee_close_z"] = round(ee_close_z, 4)
    if not (ee_close_z - 0.10 < table_z < ee_close_z + 0.01):
        for cv in ("opencv", "opengl", "roslink"):
            tzl = []
            for v in range(8, 256, 16):
                for u in range(8, 256, 16):
                    d = float(live_depth[v][u])
                    pd = 1.0 / (coef[0] * u + coef[1] * v + coef[2])
                    if abs(pd - d) < 0.003:
                        tzl.append(backproject(live_cp, u, v, d, cv)[2])
            tzc = quantile(tzl, 0.5)
            if tzc is not None and ee_close_z - 0.10 < tzc < ee_close_z + 0.01:
                report["errors"].append("convention switched by table check: " + conv + " -> " + cv)
                conv = cv
                table_z = tzc
                break
        report["table_z_checked"] = round(table_z, 4)
        report["conv_final"] = conv
    v0_cp = cam_params(video[0].cameras["front"])
    report["front_pose_delta_m"] = round(math.sqrt(sum((v0_cp["t"][i] - live_cp["t"][i]) ** 2 for i in range(3))), 5)

    # ------------------------------------------------ initialise tracks from the first frames
    # grid resolution from the sampled-pixel spacing on the table (stride * depth / focal length)
    mid_depths = sorted(float(live_depth[v][u]) for v in range(64, 192, 16) for u in range(64, 192, 16))
    grid_res = 1.5 * STRIDE * mid_depths[len(mid_depths) // 2] / live_cp["fx"]
    report["grid_res_m"] = round(grid_res, 4)
    init_comps = None
    init_fi = 0
    for fi in range(min(6, n_frames)):
        comps = grid_components(object_cells(video[fi], cams, table_z, conv), grid_res)
        if init_comps is None or len(comps) > len(init_comps):
            init_comps = comps
            init_fi = fi
    exts = sorted(c["ext"] for c in init_comps)
    med_ext = exts[len(exts) // 2] if len(exts) > 0 else None
    if med_ext is None or med_ext <= 0:
        report["stage"] = "no_objects"
        raise ValueError("no objects detected in conditioning video")
    init_comps = [c for c in init_comps if c["ext"] <= 3.0 * med_ext and c["ext"] >= 0.3 * med_ext]
    # merge duplicates (one object seen by both cameras with small calibration offsets)
    merged = []
    for c in sorted(init_comps, key=lambda c: -c["n"]):
        dup = False
        for m in merged:
            if math.sqrt(dist2((c["x"], c["y"]), (m["x"], m["y"]))) < 0.8 * med_ext:
                dup = True
        if not dup:
            merged.append(c)
    r_obj = 0.5 * med_ext
    tracks = [{"x": c["x"], "y": c["y"], "vx": 0.0, "vy": 0.0, "miss": 0, "n": c["n"]} for c in merged]
    report["init"] = {"frame": init_fi, "n_objects": len(tracks), "med_ext_m": round(med_ext, 4),
                      "objects": [[round(c["x"], 3), round(c["y"], 3), c["n"], c["cams"]] for c in merged]}

    close_xy = ees[close_fi][:2]
    release_xy = ees[open_fi][:2] if open_fi is not None else None
    target = None
    target_log = []
    count_log = []
    for fi in range(init_fi + 1, n_frames):
        cells = object_cells(video[fi], cams, table_z, conv)
        track_step_world(tracks, cells, r_obj, 3.0 * med_ext)
        if fi % 16 == 0:
            count_log.append([fi, len(cells), sum(1 for t in tracks if t["miss"] == 0)])
        if fi == close_fi:
            ti, d = nearest_track(tracks, close_xy)
            target = ti
            target_log.append(["close", fi, ti, round(d, 4)])
        if target is not None and release_xy is not None and fi == min(n_frames - 1, open_fi + 2):
            ti, d = nearest_track(tracks, release_xy)
            target_log.append(["release", fi, target, ti, round(d, 4)])
            if d < 2.0 * r_obj:
                target = ti
        if target is not None and fi % 24 == 0:
            t = tracks[target]
            target_log.append(["pos", fi, round(t["x"], 3), round(t["y"], 3), t["miss"], t["n"]])
    report["count_log"] = count_log
    report["target_log"] = target_log
    if target is None:
        report["stage"] = "no_target"
        raise ValueError("target track not identified")
    tgt_xy = [tracks[target]["x"], tracks[target]["y"]]
    report["final_target_xy"] = [round(tgt_xy[0], 4), round(tgt_xy[1], 4)]
    report["final_tracks"] = [[round(t["x"], 3), round(t["y"], 3), t["miss"]] for t in tracks]

    # ------------------------------------------------ live target localization
    live_comps = grid_components(object_cells(get_observation(), cams, table_z, conv), grid_res)
    live_comps = [c for c in live_comps if c["ext"] <= 3.0 * med_ext and c["ext"] >= 0.3 * med_ext]
    report["live_objects"] = [[round(c["x"], 3), round(c["y"], 3), c["n"], c["cams"]] for c in live_comps]
    best = None
    for c in live_comps:
        d = math.sqrt(dist2((c["x"], c["y"]), tgt_xy))
        if best is None or d < best[0]:
            best = (d, c)
    if best is None:
        report["stage"] = "no_live_target"
        raise ValueError("no live object component")
    comp = best[1]
    report["live_target"] = {"x": round(comp["x"], 4), "y": round(comp["y"], 4), "d_to_track": round(best[0], 4),
                             "ztop": round(comp["ztop"], 4)}
    gx, gy, z_top = comp["x"], comp["y"], comp["ztop"]
    top_xy = [[m[0], m[1]] for m in comp["members"] if m[2] > z_top - 0.01]
    # refine with the public segmentation + projection when it agrees
    try:
        pr = project(live_cp, [gx, gy, z_top], conv)
        if pr is not None:
            seg = segment_points("front", [[int(round(pr[0])), int(round(pr[1]))]])
            refine_log = []
            best_ref = None
            if seg.ok:
                for ci in range(min(5, len(seg.segmentations))):
                    cloud = mask_to_point_cloud(seg.segmentations[ci], "front", target_frame=base_frame)
                    pts = [[float(p[0]), float(p[1]), float(p[2])] for p in cloud.points]
                    if len(pts) < 10:
                        continue
                    zt = quantile([p[2] for p in pts], 0.95)
                    tp = [p for p in pts if p[2] > zt - 0.01]
                    xs = [p[0] for p in tp]
                    ys = [p[1] for p in tp]
                    ax = 0.5 * (quantile(xs, 0.05) + quantile(xs, 0.95))
                    ay = 0.5 * (quantile(ys, 0.05) + quantile(ys, 0.95))
                    ext = max(quantile(xs, 0.95) - quantile(xs, 0.05), quantile(ys, 0.95) - quantile(ys, 0.05))
                    agree = math.sqrt(dist2((ax, ay), (gx, gy)))
                    refine_log.append([ci, round(ax, 4), round(ay, 4), round(zt, 4), round(ext, 4), round(agree, 4), len(pts)])
                    if agree < 0.6 * med_ext and ext < 1.6 * med_ext and ext > 0.4 * med_ext:
                        if best_ref is None or agree < best_ref[0]:
                            best_ref = (agree, ax, ay, zt, tp)
            report["api_refine"] = {"px": [round(pr[0], 1), round(pr[1], 1)], "cands": refine_log,
                                    "used": best_ref is not None}
            if best_ref is not None:
                gx, gy, z_top = best_ref[1], best_ref[2], best_ref[3]
                top_xy = [[p[0], p[1]] for p in best_ref[4]]
    except Exception as exc:
        note_error("api_refine", exc)
    if len(top_xy) < 6:
        top_xy = [[m[0], m[1]] for m in comp["members"]]
    obj_yaw = min_area_yaw(top_xy)
    obj_h = z_top - table_z

    # ------------------------------------------------ button localization
    button = None
    button_log = []
    for prompt in ("button", "white button", "round button"):
        try:
            seg = segment_text("front", prompt)
        except Exception as exc:
            note_error("button_seg", exc)
            continue
        if not seg.ok:
            button_log.append([prompt, "not_ok"])
            continue
        cands = []
        for ci in range(min(5, len(seg.segmentations))):
            cand = seg.segmentations[ci]
            try:
                score = float(seg.scores[ci])
            except Exception:
                try:
                    score = float(cand.score)
                except Exception:
                    score = 0.0
            try:
                cloud = mask_to_point_cloud(cand, "front", target_frame=base_frame)
                pts = [[float(p[0]), float(p[1]), float(p[2])] for p in cloud.points]
            except Exception as exc:
                note_error("button_cloud", exc)
                continue
            if len(pts) < 8:
                button_log.append([prompt, ci, "few_points", len(pts)])
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ext = max(quantile(xs, 0.95) - quantile(xs, 0.05), quantile(ys, 0.95) - quantile(ys, 0.05))
            bxw = 0.5 * (quantile(xs, 0.05) + quantile(xs, 0.95))
            byw = 0.5 * (quantile(ys, 0.05) + quantile(ys, 0.95))
            bzt = quantile([p[2] for p in pts], 0.9)
            near_obj = min([math.sqrt(dist2((bxw, byw), (c["x"], c["y"]))) for c in live_comps] + [9.0])
            button_log.append([prompt, ci, round(score, 3), round(bxw, 4), round(byw, 4), round(bzt, 4),
                               round(ext, 4), len(pts), round(near_obj, 4)])
            if near_obj < 0.8 * med_ext or ext > 0.12:
                continue
            cands.append((-score, ci, {"x": bxw, "y": byw, "z_top": bzt, "ext": ext,
                                       "prompt": prompt, "n": len(pts)}))
        if len(cands) > 0:
            button = sorted(cands, key=lambda c: c[0])[0][2]
            break
    report["button_log"] = button_log
    report["button"] = button

    # ------------------------------------------------ grasp pose
    hq = [float(c) for c in home_pose.quaternion_wxyz]
    x_axis = quat_rot(hq, [1.0, 0.0, 0.0])
    report["home_z_axis"] = [round(c, 3) for c in quat_rot(hq, [0.0, 0.0, 1.0])]
    home_yaw = math.atan2(x_axis[1], x_axis[0])
    down = [0.0, 1.0, 0.0, 0.0]
    psi = wrap_near(obj_yaw, home_yaw, math.pi / 2.0)
    grasp_q = quat_mul(yaw_quat(psi), down)
    grasp_z = table_z + 0.55 * obj_h
    grasp_z = max(grasp_z, table_z + 0.012)
    grasp_z = min(grasp_z, z_top - 0.006)
    report["grasp"] = {"x": round(gx, 4), "y": round(gy, 4), "z": round(grasp_z, 4), "z_top": round(z_top, 4),
                       "obj_h": round(obj_h, 4), "obj_yaw_deg": round(obj_yaw * 180 / math.pi, 1),
                       "psi_deg": round(psi * 180 / math.pi, 1), "home_yaw_deg": round(home_yaw * 180 / math.pi, 1)}
    report["stage"] = "analysis_done"

    def pose_at(x, y, z, q):
        return Pose([x, y, z], q, base_frame)

    above_z = z_top + 0.10
    pick_log = []
    for cycle in range(2):
        if ENDED["flag"]:
            break
        if cycle == 0:
            run_motion("open0", lambda: open_gripper())
            run_motion("pre_grasp", lambda: move_to_pose(pose_at(gx, gy, above_z, grasp_q), tolerance=0.01, max_steps=150))
        run_motion("descend%d" % cycle, lambda: move_to_pose(pose_at(gx, gy, grasp_z, grasp_q), tolerance=0.008, max_steps=80))
        run_motion("close%d" % cycle, lambda: close_gripper())
        g_after = None
        try:
            g_after = float(get_robot_state().gripper_positions["primary"])
        except Exception as exc:
            note_error("grip_read", exc)
        run_motion("lift%d" % cycle, lambda: move_to_pose(pose_at(gx, gy, grasp_z + 0.08, grasp_q), tolerance=0.015, max_steps=60))
        g_lift = None
        try:
            g_lift = float(get_robot_state().gripper_positions["primary"])
        except Exception as exc:
            note_error("grip_read2", exc)
        pick_log.append({"cycle": cycle, "grip_after_close": g_after, "grip_after_lift": g_lift})
        report["pick_log"] = pick_log
        run_motion("lower%d" % cycle, lambda: move_to_pose(pose_at(gx, gy, grasp_z + 0.005, grasp_q), tolerance=0.01, max_steps=60))
        run_motion("release%d" % cycle, lambda: open_gripper())
        run_motion("retreat%d" % cycle, lambda: move_to_pose(pose_at(gx, gy, grasp_z + 0.06, grasp_q), tolerance=0.015, max_steps=50))

    if button is not None and not ENDED["flag"]:
        bq = quat_mul(yaw_quat(home_yaw), down)
        bx_, by_, bz_ = button["x"], button["y"], button["z_top"]
        run_motion("close_for_press", lambda: close_gripper())
        run_motion("above_button", lambda: move_to_pose(pose_at(bx_, by_, bz_ + 0.08, bq), tolerance=0.01, max_steps=120))
        run_motion("press", lambda: move_to_pose(pose_at(bx_, by_, bz_ - 0.015, bq), tolerance=0.008, max_steps=50))
        run_motion("unpress", lambda: move_to_pose(pose_at(bx_, by_, bz_ + 0.06, bq), tolerance=0.02, max_steps=40))
    elif button is None:
        report["errors"].append("button not localized; skipped press")
    report["stage"] = "done"


try:
    main()
except Exception as exc:
    note_error("main", exc)
report["motions"] = MOTION_LOG
report["ended"] = ENDED["flag"]
result = report
