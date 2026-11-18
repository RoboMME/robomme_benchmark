# v02 - VideoUnmaskSwap: video tracking (as v01) + LIVE tracking while the shuffle continues
# (v01 ep0 rollout: containers kept moving for ~62 steps after the video ended, so v01 grasped
# stale positions). Hold at home, track every tick until the scene is static, then per target:
# re-observe from home, footprint-extent centre of top points, wrist RGB-D refinement at the
# pre-grasp pose, cap grasp, lift, put back, open_gripper release, return home.
# v03 (one change class vs v02, "pick registration"): after the verified 10 cm lift, lift
# further to grasp_z+0.20 and hold still (15 ticks per intermediate pick, up to 80 for the
# final pick), watching terminated/truncated. v02 ep0 picked the right containers in the
# right order (revealed red, green x3, blue) but success was never observed. If the final
# hold does not terminate, the last container is also put back and the arm idles 20 ticks.
import math

report = {"version": "v03", "stage": "start"}

HEXD = "0123456789abcdef"
HEX2 = [HEXD[i // 16] + HEXD[i % 16] for i in range(256)]
MIN_SEP = 0.04
COLORS = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink"]


def grid(arr, s):
    try:
        return arr[::s, ::s].tolist()
    except Exception:
        out = []
        h = len(arr)
        w = len(arr[0])
        for y in range(0, h, s):
            row = arr[y]
            out.append([row[x] for x in range(0, w, s)])
        return out


def thumb_hex(g):
    parts = []
    for row in g:
        for px in row:
            parts.append(HEX2[int(px[0])] + HEX2[int(px[1])] + HEX2[int(px[2])])
    return "".join(parts)


def cube_class(r, g, b):
    # saturated, bright-enough pixels only (cubes s>=0.95; wooden table s<=0.70)
    mx = max(r, g, b)
    mn = min(r, g, b)
    c = mx - mn
    if mx < 70 or c < 0.82 * mx:
        return None
    if mx == r:
        h = 60.0 * ((g - b) / c)
        if h < 0:
            h = h + 360.0
    elif mx == g:
        h = 60.0 * ((b - r) / c + 2.0)
    else:
        h = 60.0 * ((r - g) / c + 4.0)
    if h < 20 or h >= 340:
        return "red"
    if h < 50:
        return "orange"
    if h < 75:
        return "yellow"
    if h < 165:
        return "green"
    if h < 195:
        return "cyan"
    if h < 255:
        return "blue"
    if h < 290:
        return "purple"
    return "pink"


def is_white(px):
    r = px[0]
    g = px[1]
    b = px[2]
    mn = min(r, g, b)
    return mn > 115 and max(r, g, b) - mn < 45


def changed(px, bg):
    return abs(px[0] - bg[0]) + abs(px[1] - bg[1]) + abs(px[2] - bg[2]) > 60


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def qmat(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def cam_params(cam):
    K = cam.intrinsics
    pose = cam.camera_pose
    p = [float(v) for v in pose.position]
    q = [float(v) for v in pose.quaternion_wxyz]
    return {"fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]), "cy": float(K[1][2]),
            "p": p, "R": qmat(q), "q": q}


def ray(cp, u, v):
    # OpenCV convention (calibrated on v00 ep0 live front camera: 0.62 px median error)
    a = (u - cp["cx"]) / cp["fx"]
    b = (v - cp["cy"]) / cp["fy"]
    R = cp["R"]
    return (R[0][0] * a + R[0][1] * b + R[0][2], R[1][0] * a + R[1][1] * b + R[1][2],
            R[2][0] * a + R[2][1] * b + R[2][2])


def project(cp, X):
    d = [X[0] - cp["p"][0], X[1] - cp["p"][1], X[2] - cp["p"][2]]
    R = cp["R"]
    xc = R[0][0] * d[0] + R[1][0] * d[1] + R[2][0] * d[2]
    yc = R[0][1] * d[0] + R[1][1] * d[1] + R[2][1] * d[2]
    zc = R[0][2] * d[0] + R[1][2] * d[1] + R[2][2] * d[2]
    if zc <= 1e-4:
        return None
    return (cp["fx"] * xc / zc + cp["cx"], cp["fy"] * yc / zc + cp["cy"])


def quant(vals, f):
    o = sorted(vals)
    if not o:
        return None
    return o[int((len(o) - 1) * f)]


def safe_flag(res, name):
    try:
        if name == "terminated":
            return bool(res.terminated)
        return bool(res.truncated)
    except Exception:
        return False


def ended(res):
    return safe_flag(res, "terminated") or safe_flag(res, "truncated")


def exec_info(res):
    info = {}
    try:
        info["ok"] = bool(res.ok)
    except Exception:
        pass
    try:
        info["status"] = str(res.status)
    except Exception:
        pass
    try:
        info["steps"] = int(res.steps_executed)
    except Exception:
        pass
    info["terminated"] = safe_flag(res, "terminated")
    info["truncated"] = safe_flag(res, "truncated")
    try:
        if res.error is not None:
            info["error"] = str(res.error)[:300]
    except Exception:
        pass
    return info


def components(cmap, min_size):
    h = len(cmap)
    w = len(cmap[0])
    seen = [[False] * w for yy in range(h)]
    comps = []
    for y0 in range(h):
        for x0 in range(w):
            k = cmap[y0][x0]
            if k is None or seen[y0][x0]:
                continue
            stack = [(y0, x0)]
            seen[y0][x0] = True
            pix = []
            while stack:
                cy, cx = stack.pop()
                pix.append((cy, cx))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and cmap[ny][nx] == k:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            if len(pix) >= min_size:
                comps.append({"color": k, "pix": pix})
    return comps


def kmeans_step(pts, centers, gate):
    k = len(centers)
    sx = [0.0] * k
    sy = [0.0] * k
    sn = [0] * k
    for p in pts:
        best = -1
        bd = gate * gate
        for i in range(k):
            dx = p[0] - centers[i][0]
            dy = p[1] - centers[i][1]
            d = dx * dx + dy * dy
            if d < bd:
                bd = d
                best = i
        if best >= 0:
            sx[best] = sx[best] + p[0]
            sy[best] = sy[best] + p[1]
            sn[best] = sn[best] + 1
    out = []
    for i in range(k):
        if sn[i] >= 4:
            out.append([sx[i] / sn[i], sy[i] / sn[i]])
        else:
            out.append([centers[i][0], centers[i][1]])
    return out, sn


def separate(centers, min_d):
    # identical rigid containers cannot interpenetrate: push apart centres closer than min_d
    k = len(centers)
    for it in range(3):
        for i in range(k):
            for j in range(i + 1, k):
                dx = centers[j][0] - centers[i][0]
                dy = centers[j][1] - centers[i][1]
                d = math.sqrt(dx * dx + dy * dy)
                if d >= min_d:
                    continue
                if d < 1e-6:
                    ux = 1.0
                    uy = 0.0
                else:
                    ux = dx / d
                    uy = dy / d
                push = 0.5 * (min_d - d)
                centers[i][0] = centers[i][0] - ux * push
                centers[i][1] = centers[i][1] - uy * push
                centers[j][0] = centers[j][0] + ux * push
                centers[j][1] = centers[j][1] + uy * push
    return centers


def parse_goal(goal):
    low = goal.lower()
    found = []
    for cname in COLORS:
        start = 0
        while True:
            idx = low.find(cname, start)
            if idx < 0:
                break
            found.append([idx, cname])
            start = idx + 1
    found = sorted(found)
    seq = []
    for f in found:
        if f[1] not in seq:
            seq.append(f[1])
    return seq


actions = []
report["actions"] = actions
try:
    ctx = get_task_context()
    goal = str(ctx.language)
    report["goal"] = goal
    try:
        report["task_metadata"] = dict(robomme_plus.get_task_metadata())
    except Exception as exc:
        report["task_metadata_error"] = str(exc)
    targets = parse_goal(goal)
    report["target_colors"] = targets

    state0 = get_robot_state()
    base_frame = state0.base_frame
    home_pose = state0.end_effector_poses["primary"]
    home_q = [float(v) for v in home_pose.quaternion_wxyz]
    report["home_ee"] = [round(float(v), 4) for v in home_pose.position]

    live = get_observation()
    lcp = cam_params(live.cameras["front"])

    report["stage"] = "video"
    video = robomme_plus.get_conditioning_video()
    nvid = len(video)
    report["video_len"] = nvid
    if nvid < 10:
        raise ValueError("conditioning video too short")
    ts = []
    for i in range(nvid):
        ts.append(round(float(video[i].timestamp_s), 3))
    report["video_ts_first_last"] = [ts[0], ts[-1]]
    vcp = cam_params(video[0].cameras["front"])
    vcpl = cam_params(video[nvid - 1].cameras["front"])
    cam_moved = max(abs(vcp["p"][i] - vcpl["p"][i]) for i in range(3)) + max(abs(vcp["q"][i] - vcpl["q"][i]) for i in range(4))
    report["video_cam_motion"] = round(cam_moved, 6)

    # ---- 1. choose cube frame: max saturated pixel count in first 25% ----
    best_f = 0
    best_n = -1
    sat_counts = []
    for fi in range(0, max(1, nvid // 4), 2):
        g = grid(video[fi].cameras["front"].rgb, 4)
        n = 0
        for row in g:
            for px in row:
                if cube_class(px[0], px[1], px[2]) is not None:
                    n = n + 1
        sat_counts.append([fi, n])
        if n > best_n:
            best_n = n
            best_f = fi
    report["cube_frame"] = best_f
    report["sat_counts_early"] = sat_counts

    full = video[best_f].cameras["front"].rgb.tolist()
    dep0 = video[best_f].cameras["front"].depth_m.tolist()
    H = len(full)
    W = len(full[0])
    cmap = []
    for y in range(H):
        row = full[y]
        cmap.append([cube_class(px[0], px[1], px[2]) for px in row])
    comps = components(cmap, 6)
    cubes = []
    for c in comps:
        us = [p[1] for p in c["pix"]]
        vs = [p[0] for p in c["pix"]]
        u = sum(us) / len(us)
        v = sum(vs) / len(vs)
        ds = sorted(float(dep0[p[0]][p[1]]) for p in c["pix"])
        d = ds[len(ds) // 2]
        rr = ray(vcp, u + 0.5, v + 0.5)
        X = [vcp["p"][0] + d * rr[0], vcp["p"][1] + d * rr[1], vcp["p"][2] + d * rr[2]]
        cubes.append({"color": c["color"], "n": len(c["pix"]), "uv": [round(u, 1), round(v, 1)],
                      "xyz": [round(X[0], 4), round(X[1], 4), round(X[2], 4)],
                      "box": [min(us), min(vs), max(us), max(vs)]})
    # merge same-colour components whose 3D centres are within 3 cm (lit/shaded faces)
    merged = []
    for c in sorted([[-c["n"], i] for i, c in enumerate(cubes)]):
        cb = cubes[c[1]]
        dup = False
        for m in merged:
            if m["color"] == cb["color"] and math.sqrt((m["xyz"][0] - cb["xyz"][0]) ** 2 + (m["xyz"][1] - cb["xyz"][1]) ** 2) < 0.03:
                dup = True
                m["n"] = m["n"] + cb["n"]
                break
        if not dup:
            merged.append(cb)
    cubes = merged
    report["cubes"] = cubes
    if not cubes:
        raise ValueError("no cubes detected in the video")

    # ROI (stride-2 grid) around cubes, generous margin
    umin = max(0, min(c["box"][0] for c in cubes) - 70)
    umax = min(W - 1, max(c["box"][2] for c in cubes) + 70)
    vmin = max(0, min(c["box"][1] for c in cubes) - 50)
    vmax = min(H - 1, max(c["box"][3] for c in cubes) + 70)
    S = 2
    us_grid = list(range(umin - umin % S, umax + 1, S))
    vs_grid = list(range(vmin - vmin % S, vmax + 1, S))
    report["roi"] = [umin, vmin, umax, vmax]
    rays = {}
    for v in vs_grid:
        for u in us_grid:
            rays[v * 1000 + u] = ray(vcp, u + 0.5, v + 0.5)
    bg = {}
    for v in vs_grid:
        for u in us_grid:
            bg[v * 1000 + u] = full[v][u]

    def container_points(rgb_full_rows, depth_rows, cp_rays, use_bg):
        pts = []
        for v in vs_grid:
            rrow = rgb_full_rows[v]
            drow = depth_rows[v]
            for u in us_grid:
                px = rrow[u]
                if not is_white(px):
                    continue
                key = v * 1000 + u
                if use_bg and not changed(px, bg[key]):
                    continue
                d = float(drow[u])
                rr = cp_rays[key]
                z = vcp["p"][2] + d * rr[2]
                if z < 0.004 or z > 0.16:
                    continue
                pts.append((vcp["p"][0] + d * rr[0], vcp["p"][1] + d * rr[1], z))
        return pts

    # ---- 2. find the frame where all cubes are hidden ----
    report["stage"] = "cover"
    peak = None
    f_hidden = None
    hid_counts = []
    for fi in range(best_f, nvid):
        rgb = video[fi].cameras["front"].rgb
        n = 0
        for c in cubes:
            b = c["box"]
            sub = rgb[max(0, b[1] - 1):b[3] + 2, max(0, b[0] - 1):b[2] + 2].tolist()
            for row in sub:
                for px in row:
                    if cube_class(px[0], px[1], px[2]) == c["color"]:
                        n = n + 1
        hid_counts.append([fi, n])
        if peak is None:
            peak = n
        if n <= max(1, 0.03 * peak):
            f_hidden = fi
            break
    report["hidden_counts"] = hid_counts[-40:]
    report["f_hidden"] = f_hidden
    if f_hidden is None:
        raise ValueError("cubes never fully hidden")

    # ---- 3. track ----
    report["stage"] = "track"
    k = len(cubes)
    centers = [[c["xyz"][0], c["xyz"][1]] for c in cubes]
    vel = [[0.0, 0.0] for i in range(k)]
    f_start = min(nvid - 1, f_hidden + 1)
    track_log = []
    cells_log = []
    count_log = []
    for fi in range(f_start, nvid):
        fr = video[fi].cameras["front"]
        rgbrows = fr.rgb.tolist()
        deprows = fr.depth_m.tolist()
        pts = container_points(rgbrows, deprows, rays, True)
        if fi == f_start:
            pred = [[centers[i][0], centers[i][1]] for i in range(k)]
            for it in range(4):
                pred, sn = kmeans_step(pts, pred, 0.05)
            newc = pred
        else:
            pred = [[centers[i][0] + vel[i][0], centers[i][1] + vel[i][1]] for i in range(k)]
            newc, sn = kmeans_step(pts, pred, 0.04)
            newc, sn = kmeans_step(pts, newc, 0.04)
            newc = separate(newc, MIN_SEP)
        for i in range(k):
            if fi > f_start:
                vel[i] = [0.5 * vel[i][0] + 0.5 * (newc[i][0] - centers[i][0]),
                          0.5 * vel[i][1] + 0.5 * (newc[i][1] - centers[i][1])]
                if sn[i] < 4:
                    vel[i] = [0.7 * vel[i][0], 0.7 * vel[i][1]]
        centers = newc
        track_log.append([fi] + [[round(c[0], 3), round(c[1], 3)] for c in centers])
        count_log.append(sn)
        cs = set()
        for p in pts:
            ix = int((p[0] + 0.2) / 0.005)
            iy = int((p[1] + 0.64) / 0.005)
            if 0 <= ix < 256 and 0 <= iy < 256:
                cs.add(ix * 256 + iy)
        cells_log.append("".join(HEX2[c // 256] + HEX2[c % 256] for c in sorted(cs)))
    report["tracks"] = track_log
    report["track_counts"] = count_log
    report["cells"] = cells_log
    report["cells_origin"] = [-0.2, -0.64, 0.005]

    report["ts"] = ts

    # ---- 4. LIVE tracking: the shuffle continues after the video ends; hold and track ----
    report["stage"] = "live_wait"
    st = get_robot_state()
    hold_q = st.joint_positions["primary"]
    emb = st.embodiment
    g_hold = float(st.gripper_positions["primary"])
    obs = live
    prev_grid = None
    quiet = 0
    wait_log = []
    done = False
    n_wait = 0
    for tick in range(320):
        fr = obs.cameras["front"]
        rgbrows = fr.rgb.tolist()
        deprows = fr.depth_m.tolist()
        pts = container_points(rgbrows, deprows, rays, True)
        pred = [[centers[i][0] + vel[i][0], centers[i][1] + vel[i][1]] for i in range(k)]
        newc, sn = kmeans_step(pts, pred, 0.04)
        newc, sn = kmeans_step(pts, newc, 0.04)
        newc = separate(newc, MIN_SEP)
        mv = 0.0
        for i in range(k):
            dx = newc[i][0] - centers[i][0]
            dy = newc[i][1] - centers[i][1]
            mv = max(mv, math.sqrt(dx * dx + dy * dy))
            vel[i] = [0.5 * vel[i][0] + 0.5 * dx, 0.5 * vel[i][1] + 0.5 * dy]
            if sn[i] < 4:
                vel[i] = [0.7 * vel[i][0], 0.7 * vel[i][1]]
        centers = newc
        diff = 0
        cur_grid = []
        for v in vs_grid:
            row = rgbrows[v]
            for u in us_grid:
                cur_grid.append(row[u])
        if prev_grid is not None:
            for j in range(len(cur_grid)):
                a = cur_grid[j]
                b = prev_grid[j]
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]) > 60:
                    diff = diff + 1
        else:
            diff = 9999
        prev_grid = cur_grid
        wait_log.append([tick, diff, round(mv, 4)] + [[round(c[0], 3), round(c[1], 3)] for c in centers])
        if diff <= 3 and mv < 0.0015:
            quiet = quiet + 1
        else:
            quiet = 0
        if quiet >= 10:
            break
        sr = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=hold_q, gripper_position=g_hold, embodiment=emb)}))
        n_wait = n_wait + 1
        if ended(sr):
            done = True
            break
        obs = get_observation()
    report["wait_steps"] = n_wait
    report["wait_log"] = wait_log
    report["static_centers"] = [[round(c[0], 4), round(c[1], 4)] for c in centers]

    def live_points(cam, cp, stride, box, zmax):
        rgbr = cam.rgb.tolist()
        depr = cam.depth_m.tolist()
        hh = len(rgbr)
        ww = len(rgbr[0])
        out = []
        for v in range(max(0, box[1]), min(hh - 1, box[3]) + 1, stride):
            rrow = rgbr[v]
            drow = depr[v]
            for u in range(max(0, box[0]), min(ww - 1, box[2]) + 1, stride):
                px = rrow[u]
                if not is_white(px):
                    continue
                d = float(drow[u])
                if d <= 0.0 or d > 3.0:
                    continue
                rr = ray(cp, u + 0.5, v + 0.5)
                X = (cp["p"][0] + d * rr[0], cp["p"][1] + d * rr[1], cp["p"][2] + d * rr[2])
                if X[2] > 0.004 and X[2] < zmax:
                    out.append(X)
        return out

    def footprint(pts, c0, gates, zt_fixed):
        c = [c0[0], c0[1]]
        res = None
        for gate in gates:
            near = [p for p in pts if (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 < gate * gate]
            if len(near) < 8:
                return res
            if zt_fixed is None:
                zt = quant([p[2] for p in near], 0.92)
            else:
                zt = zt_fixed
            top = [p for p in near if p[2] >= zt - 0.007 and p[2] <= zt + 0.012]
            if len(top) < 6:
                return res
            best = None
            for adeg in range(0, 90, 5):
                a = math.radians(adeg)
                ca = math.cos(a)
                sa = math.sin(a)
                pu = [ca * p[0] + sa * p[1] for p in top]
                pv = [-sa * p[0] + ca * p[1] for p in top]
                u0 = quant(pu, 0.04)
                u1 = quant(pu, 0.96)
                v0 = quant(pv, 0.04)
                v1 = quant(pv, 0.96)
                area = (u1 - u0) * (v1 - v0)
                if best is None or area < best[0]:
                    best = [area, a, 0.5 * (u0 + u1), 0.5 * (v0 + v1), u1 - u0, v1 - v0]
            a = best[1]
            c = [math.cos(a) * best[2] - math.sin(a) * best[3], math.sin(a) * best[2] + math.cos(a) * best[3]]
            res = {"center": c, "ztop": zt, "yaw": a, "ext": [best[4], best[5]], "n": len(top), "n_near": len(near)}
        return res

    def wrist_refine(c, zt, tag, rec):
        info = {"tag": tag}
        rec.setdefault("wrist", []).append(info)
        try:
            ob = get_observation()
            wc = ob.cameras["wrist"]
            wcp = cam_params(wc)
            rgbr = wc.rgb.tolist()
            depr = wc.depth_m.tolist()
            hh = len(rgbr)
            ww = len(rgbr[0])
            pts = []
            tz = []
            for v in range(0, hh, 1):
                rrow = rgbr[v]
                drow = depr[v]
                for u in range(0, ww, 1):
                    d = float(drow[u])
                    if d <= 0.0 or d > 2.0:
                        continue
                    px = rrow[u]
                    white = is_white(px)
                    wood = px[0] > px[1] + 15 and px[1] > px[2] and cube_class(px[0], px[1], px[2]) is None
                    if not white and not (wood and u % 3 == 0 and v % 3 == 0):
                        continue
                    rr = ray(wcp, u + 0.5, v + 0.5)
                    X = (wcp["p"][0] + d * rr[0], wcp["p"][1] + d * rr[1], wcp["p"][2] + d * rr[2])
                    if white:
                        if abs(X[2] - zt) < 0.02:
                            pts.append(X)
                    else:
                        tz.append(X[2])
            info["n_white_band"] = len(pts)
            tmed = quant(tz, 0.5)
            info["table_z"] = None if tmed is None else round(tmed, 4)
            if tmed is None or abs(tmed) > 0.015:
                info["reject"] = "table plane check failed (wrist calibration?)"
                return None
            wz = quant([p[2] for p in pts if (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 < 0.03 * 0.03], 0.9)
            if wz is None or abs(wz - zt) > 0.012:
                info["reject"] = "cap top height mismatch"
                info["wz"] = None if wz is None else round(wz, 4)
                return None
            fp = footprint(pts, c, [0.032, 0.027, 0.026], wz)
            if fp is None or fp["n"] < 25:
                info["reject"] = "too few wrist top points"
                return None
            sh = math.sqrt((fp["center"][0] - c[0]) ** 2 + (fp["center"][1] - c[1]) ** 2)
            info.update({"center": [round(fp["center"][0], 4), round(fp["center"][1], 4)], "wz": round(wz, 4),
                         "ext": [round(fp["ext"][0], 4), round(fp["ext"][1], 4)], "yaw_deg": round(math.degrees(fp["yaw"]), 1),
                         "n": fp["n"], "shift": round(sh, 4)})
            if sh > 0.025:
                info["reject"] = "shift too large"
                return None
            return fp
        except Exception as exc:
            info["error"] = str(exc)[:200]
            return None

    plan = []
    for tcol in targets:
        for i in range(k):
            if cubes[i]["color"] == tcol:
                plan.append(i)
    report["plan"] = [[i, cubes[i]["color"]] for i in plan]

    # ---- 5. act ----
    report["stage"] = "act"
    Rh = qmat(home_q)
    alpha0 = math.atan2(Rh[1][1], Rh[0][1])  # hand y axis (finger closing axis) in base XY
    home_pos = [float(v) for v in home_pose.position]
    cur = [[c[0], c[1]] for c in centers]
    box_all = [us_grid[0], vs_grid[0], us_grid[-1], vs_grid[-1]]
    for pi in range(len(plan)):
        if done:
            break
        ti = plan[pi]
        rec = {"track": ti, "color": cubes[ti]["color"]}
        actions.append(rec)
        if pi > 0:
            r = move_to_pose(Pose(home_pos, home_q, base_frame), tolerance=0.02, max_steps=250)
            rec["to_home"] = exec_info(r)
            if ended(r):
                done = True
                break
        obs = get_observation()
        fcam = obs.cameras["front"]
        allp = live_points(fcam, lcp, 2, box_all, 0.10)
        for it in range(3):
            cur, csn = kmeans_step(allp, cur, 0.035)
        cur = separate(cur, MIN_SEP)
        rec["reassoc"] = [[round(c[0], 4), round(c[1], 4)] for c in cur]
        uv = project(lcp, (cur[ti][0], cur[ti][1], 0.04))
        if uv is None:
            rec["error"] = "projection failed"
            continue
        win = [int(uv[0]) - 30, int(uv[1]) - 32, int(uv[0]) + 30, int(uv[1]) + 26]
        p1 = live_points(fcam, lcp, 1, win, 0.10)
        fp = footprint(p1, cur[ti], [0.036, 0.03, 0.027, 0.027], None)
        if fp is None:
            rec["error"] = "front footprint failed"
            rec["n_points"] = len(p1)
            continue
        cx = fp["center"][0]
        cy = fp["center"][1]
        ztop = fp["ztop"]
        rec["front"] = {"center": [round(cx, 4), round(cy, 4)], "ztop": round(ztop, 4), "n": fp["n"],
                        "ext": [round(fp["ext"][0], 4), round(fp["ext"][1], 4)], "yaw_deg": round(math.degrees(fp["yaw"]), 1),
                        "track_center": [round(cur[ti][0], 4), round(cur[ti][1], 4)]}
        yaw = fp["yaw"]
        ext = fp["ext"]
        held = False
        for attempt in range(2):
            rec["attempt"] = attempt
            # closing axis: cap side normal with max clearance to neighbour tracks
            best_theta = yaw
            best_clear = None
            best_w = ext[0]
            for ii in range(2):
                theta = yaw + ii * math.pi / 2
                clear = 1.0
                for j in range(k):
                    if j == ti:
                        continue
                    for sgn in (-1.0, 1.0):
                        fx = cx + sgn * 0.035 * math.cos(theta)
                        fy = cy + sgn * 0.035 * math.sin(theta)
                        clear = min(clear, math.sqrt((fx - cur[j][0]) ** 2 + (fy - cur[j][1]) ** 2))
                if best_clear is None or clear > best_clear:
                    best_clear = clear
                    best_theta = theta
                    best_w = ext[ii]
            delta = best_theta - alpha0
            while delta > math.pi / 2:
                delta = delta - math.pi
            while delta < -math.pi / 2:
                delta = delta + math.pi
            qg = qmul((math.cos(delta / 2), 0.0, 0.0, math.sin(delta / 2)), home_q)
            if best_w < 0.02 or best_w > 0.06:
                best_w = 0.045
            gopen = min(1.0, max(0.6, (best_w + 0.026) / 0.08))
            gz = ztop - 0.012
            a_rec = {"center": [round(cx, 4), round(cy, 4)], "yaw_delta_deg": round(math.degrees(delta), 1),
                     "clearance": round(best_clear, 4), "grasp_z": round(gz, 4), "gopen": round(gopen, 3), "w": round(best_w, 4)}
            rec["a" + str(attempt)] = a_rec
            r = open_gripper()
            a_rec["open"] = exec_info(r)
            if ended(r):
                done = True
                break
            r = set_gripper(gopen)
            a_rec["preopen"] = exec_info(r)
            if ended(r):
                done = True
                break
            r = move_to_pose(Pose((cx, cy, ztop + 0.10), qg, base_frame), tolerance=0.012, max_steps=250)
            a_rec["pre"] = exec_info(r)
            if ended(r):
                done = True
                break
            for rr_i in range(2):
                wf = wrist_refine([cx, cy], ztop, "pre" + str(rr_i), rec)
                if wf is None:
                    break
                ncx = wf["center"][0]
                ncy = wf["center"][1]
                yaw = wf["yaw"]
                ext = wf["ext"]
                if math.sqrt((ncx - cx) ** 2 + (ncy - cy) ** 2) < 0.002:
                    cx = ncx
                    cy = ncy
                    break
                cx = ncx
                cy = ncy
                r = move_to_pose(Pose((cx, cy, ztop + 0.10), qg, base_frame), tolerance=0.008, max_steps=120)
                a_rec["adjust" + str(rr_i)] = exec_info(r)
                if ended(r):
                    done = True
                    break
            if done:
                break
            a_rec["center_final"] = [round(cx, 4), round(cy, 4)]
            r = move_to_pose(Pose((cx, cy, gz), qg, base_frame), tolerance=0.008, max_steps=150)
            a_rec["down"] = exec_info(r)
            if ended(r):
                done = True
                break
            r = close_gripper()
            a_rec["close"] = exec_info(r)
            if ended(r):
                done = True
                break
            a_rec["grip_closed"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
            r = move_to_pose(Pose((cx, cy, gz + 0.10), qg, base_frame), tolerance=0.015, max_steps=150)
            a_rec["lift"] = exec_info(r)
            if ended(r):
                done = True
                break
            gl = float(get_robot_state().gripper_positions["primary"])
            a_rec["grip_lifted"] = round(gl, 3)
            if gl > 0.03:
                held = True
                break
        if done:
            break
        rec["held"] = held
        if not held:
            r = open_gripper()
            rec["giveup_open"] = exec_info(r)
            if ended(r):
                done = True
                break
            continue
        r = move_to_pose(Pose((cx, cy, gz + 0.20), qg, base_frame), tolerance=0.015, max_steps=150)
        rec["lift_high"] = exec_info(r)
        if ended(r):
            done = True
            break
        rec["grip_high"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
        n_hold = 15
        if pi == len(plan) - 1:
            n_hold = 80
        try:
            st = get_robot_state()
            for tick in range(n_hold):
                sr = step(RobotAction(arms={"primary": ArmCommand(
                    mode="joint_position", target=st.joint_positions["primary"],
                    gripper_position=0.0, embodiment=st.embodiment)}))
                if ended(sr):
                    done = True
                    rec["hold_end_tick"] = tick
                    break
        except Exception as exc:
            rec["hold_err"] = str(exc)[:200]
        if done:
            break
        try:
            ob2 = get_observation()
            rg2 = ob2.cameras["front"].rgb.tolist()
            uvc = project(lcp, (cx, cy, 0.02))
            cnts = {}
            if uvc is not None:
                for v in range(max(0, int(uvc[1]) - 12), min(H, int(uvc[1]) + 12)):
                    for u in range(max(0, int(uvc[0]) - 12), min(W, int(uvc[0]) + 12)):
                        px = rg2[v][u]
                        kk = cube_class(px[0], px[1], px[2])
                        if kk is not None:
                            cnts[kk] = cnts.get(kk, 0) + 1
            rec["revealed"] = cnts
        except Exception as exc:
            rec["reveal_err"] = str(exc)[:200]
        r = move_to_pose(Pose((cx, cy, gz + 0.004), qg, base_frame), tolerance=0.01, max_steps=150)
        rec["putback"] = exec_info(r)
        if ended(r):
            done = True
            break
        r = open_gripper()
        rec["release"] = exec_info(r)
        if ended(r):
            done = True
            break
        r = move_to_pose(Pose((cx, cy, ztop + 0.10), qg, base_frame), tolerance=0.015, max_steps=150)
        rec["retreat"] = exec_info(r)
        if ended(r):
            done = True
            break
    if not done:
        # final fallback: last container was held 80 ticks without termination and then put
        # back like the others; idle a little so a place-based predicate can be observed.
        try:
            st = get_robot_state()
            for tick in range(20):
                sr = step(RobotAction(arms={"primary": ArmCommand(
                    mode="joint_position", target=st.joint_positions["primary"],
                    gripper_position=1.0, embodiment=st.embodiment)}))
                if ended(sr):
                    done = True
                    report["final_idle_end_tick"] = tick
                    break
        except Exception as exc:
            report["final_idle_err"] = str(exc)[:200]
    report["stage"] = "done"
    report["ended"] = done
except Exception as exc:
    report["error"] = str(exc)[:1000]

result = report
