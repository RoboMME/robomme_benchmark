import math

# PatternLock v00: decode the stick pattern from the episode's conditioning video
# (recorded end-effector track + button lattice detected from front RGB-D), map it onto the
# live button lattice (detected the same way) by grid index, and redraw it with Cartesian
# straight-line stick motions at the video's drawing height.  All task-dependent values are
# measured at runtime from this episode's video and live observation.

report = {"version": "v01_dwell", "notes": [], "errors": []}


def note(msg):
    report["notes"].append(str(msg))


def err(msg):
    report["errors"].append(str(msg))


def fl(values):
    return [float(v) for v in values]


def median(values):
    vals = sorted(float(v) for v in values)
    if len(vals) == 0:
        return None
    mid = len(vals) // 2
    if len(vals) % 2 == 1:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def quat_to_r(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def as_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        rows = []
        for v in range(len(arr)):
            rows.append([arr[v][u] for u in range(len(arr[v]))])
        return rows


def cam_info(obs, name):
    cam = obs.cameras[name]
    k = cam.intrinsics
    pose = cam.camera_pose
    return {
        "fx": float(k[0][0]), "fy": float(k[1][1]), "cx": float(k[0][2]), "cy": float(k[1][2]),
        "t": fl(pose.position), "r": quat_to_r(fl(pose.quaternion_wxyz)),
    }


def pix_to_cam(conv, u, v, d, ci):
    a = (u - ci["cx"]) / ci["fx"]
    b = (v - ci["cy"]) / ci["fy"]
    if conv[1] == "ray":
        d = d / math.sqrt(1.0 + a * a + b * b)
    if conv[0] == "cv":
        return (a * d, b * d, d)
    if conv[0] == "gl":
        return (a * d, -b * d, -d)
    return (d, -a * d, -b * d)  # ros: x forward, y left, z up


def cam_to_world(pc, ci):
    r = ci["r"]
    t = ci["t"]
    return (
        r[0][0] * pc[0] + r[0][1] * pc[1] + r[0][2] * pc[2] + t[0],
        r[1][0] * pc[0] + r[1][1] * pc[1] + r[1][2] * pc[2] + t[1],
        r[2][0] * pc[0] + r[2][1] * pc[1] + r[2][2] * pc[2] + t[2],
    )


def world_to_pix(conv, w, ci):
    r = ci["r"]
    t = ci["t"]
    dx, dy, dz = w[0] - t[0], w[1] - t[1], w[2] - t[2]
    pc = (
        r[0][0] * dx + r[1][0] * dy + r[2][0] * dz,
        r[0][1] * dx + r[1][1] * dy + r[2][1] * dz,
        r[0][2] * dx + r[1][2] * dy + r[2][2] * dz,
    )
    if conv[0] == "cv":
        zc, xc, yc = pc[2], pc[0], pc[1]
    elif conv[0] == "gl":
        zc, xc, yc = -pc[2], pc[0], -pc[1]
    else:
        zc, xc, yc = pc[0], -pc[1], -pc[2]
    if zc <= 1e-6:
        return None
    return (ci["fx"] * xc / zc + ci["cx"], ci["fy"] * yc / zc + ci["cy"])


def solve3(m, b):
    def det(a):
        return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    dm = det(m)
    if abs(dm) < 1e-12:
        return None
    out = []
    for col in range(3):
        mm = [list(row) for row in m]
        for row in range(3):
            mm[row][col] = b[row]
        out.append(det(mm) / dm)
    return out


def fit_plane(points):
    # z = a x + b y + c
    sxx = sxy = sx = syy = sy = n = sxz = syz = sz = 0.0
    for p in points:
        x, y, z = p[0], p[1], p[2]
        sxx += x * x; sxy += x * y; sx += x; syy += y * y; sy += y; n += 1.0
        sxz += x * z; syz += y * z; sz += z
    sol = solve3([[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]], [sxz, syz, sz])
    if sol is None:
        return None
    se = 0.0
    for p in points:
        e = p[2] - (sol[0] * p[0] + sol[1] * p[1] + sol[2])
        se += e * e
    return {"a": sol[0], "b": sol[1], "c": sol[2], "rms": math.sqrt(se / max(1.0, n))}


CONVS = (("cv", "z"), ("gl", "z"), ("ros", "z"), ("cv", "ray"), ("gl", "ray"), ("ros", "ray"))


def choose_convention(depth, ci):
    h = len(depth)
    w = len(depth[0])
    scores = {}
    best = None
    for conv in CONVS:
        pts = []
        for v in range(int(0.65 * h), h, 4):
            for u in range(0, w, 4):
                d = float(depth[v][u])
                if d > 0.05 and d < 10.0:
                    pts.append(cam_to_world(pix_to_cam(conv, u, v, d, ci), ci))
        if len(pts) < 20:
            continue
        pl = fit_plane(pts)
        if pl is None:
            continue
        tilt = math.sqrt(pl["a"] ** 2 + pl["b"] ** 2)
        if pl["c"] > ci["t"][2] - 0.05:
            # the table cannot be above the camera: mirrored convention
            scores[conv[0] + "_" + conv[1]] = ["above_camera", round(pl["c"], 4)]
            continue
        score = tilt + 20.0 * pl["rms"]
        scores[conv[0] + "_" + conv[1]] = [round(tilt, 5), round(pl["rms"], 6), round(pl["c"], 4)]
        if best is None or score < best[0]:
            best = (score, conv, pl)
    return best, scores


def detect_buttons(obs, conv, plane_hint, tag):
    """Raised small blobs above the table plane in the front camera -> list of button dicts."""
    ci = cam_info(obs, "front")
    depth = as_rows(obs.cameras["front"].depth_m)
    h = len(depth)
    w = len(depth[0])
    if plane_hint is None:
        pts = []
        for v in range(int(0.65 * h), h, 4):
            for u in range(0, w, 4):
                d = float(depth[v][u])
                if d > 0.05 and d < 10.0:
                    pts.append(cam_to_world(pix_to_cam(conv, u, v, d, ci), ci))
        plane = fit_plane(pts)
    else:
        plane = plane_hint
    world = [None] * (h * w)
    hgt = [0.0] * (h * w)
    hist = {}
    for v in range(h):
        row = depth[v]
        for u in range(w):
            d = float(row[u])
            if d <= 0.05 or d >= 10.0:
                continue
            if not (d == d):
                continue
            p = cam_to_world(pix_to_cam(conv, u, v, d, ci), ci)
            if not (p[2] == p[2]):
                continue
            world[v * w + u] = p
            hh = p[2] - (plane["a"] * p[0] + plane["b"] * p[1] + plane["c"])
            hgt[v * w + u] = hh
            key = int(math.floor(hh / 0.002))
            if key >= -5 and key <= 40:
                hist[key] = hist.get(key, 0) + 1
    hmin = max(0.0015, 4.0 * plane["rms"])
    hmax = 0.05
    seen = [False] * (h * w)
    comps = []
    for start in range(h * w):
        if seen[start] or world[start] is None:
            continue
        if not (hgt[start] > hmin and hgt[start] < hmax):
            continue
        stack = [start]
        seen[start] = True
        members = []
        while len(stack) > 0:
            idx = stack.pop()
            members.append(idx)
            vv = idx // w
            uu = idx - vv * w
            for nb in ((vv - 1, uu), (vv + 1, uu), (vv, uu - 1), (vv, uu + 1)):
                if nb[0] < 0 or nb[0] >= h or nb[1] < 0 or nb[1] >= w:
                    continue
                j = nb[0] * w + nb[1]
                if seen[j] or world[j] is None:
                    continue
                if hgt[j] > hmin and hgt[j] < hmax:
                    seen[j] = True
                    stack.append(j)
        if len(members) < 3:
            continue
        maxh = max(hgt[i] for i in members)
        top = [i for i in members if hgt[i] >= 0.7 * maxh]
        xs = [world[i][0] for i in members]
        ys = [world[i][1] for i in members]
        ext = max(max(xs) - min(xs), max(ys) - min(ys))
        cx = sum(world[i][0] for i in top) / len(top)
        cy = sum(world[i][1] for i in top) / len(top)
        cz = sum(world[i][2] for i in top) / len(top)
        pu = sum((i - (i // w) * w) for i in members) / len(members)
        pv = sum((i // w) for i in members) / len(members)
        comps.append({"x": cx, "y": cy, "z": cz, "n": len(members), "ext": ext, "maxh": maxh,
                      "u": pu, "v": pv})
    raw_count = len(comps)
    comps = [c for c in comps if c["ext"] < 0.08]
    if len(comps) > 0:
        mn = median([c["n"] for c in comps])
        comps = [c for c in comps if c["n"] >= 0.3 * mn and c["n"] <= 3.0 * mn]
    info = {
        "plane": {k: round(plane[k], 6) for k in plane},
        "hmin": round(hmin, 5),
        "height_hist_2mm": {str(k): hist[k] for k in sorted(hist.keys())},
        "raw_components": raw_count,
        "kept_components": len(comps),
        "components": [[round(c["x"], 4), round(c["y"], 4), round(c["z"], 4), c["n"],
                        round(c["maxh"], 4), round(c["u"], 1), round(c["v"], 1)] for c in comps[:60]],
    }
    report["detect_" + tag] = info
    return comps, plane, ci


def fit_lattice(comps, tag):
    n = len(comps)
    if n < 4:
        err(tag + ": too few button blobs " + str(n))
        return None
    nn = []
    for i in range(n):
        best = None
        for j in range(n):
            if i == j:
                continue
            dd = math.hypot(comps[i]["x"] - comps[j]["x"], comps[i]["y"] - comps[j]["y"])
            if best is None or dd < best:
                best = dd
        nn.append(best)
    d0 = median(nn)
    pts = [comps[i] for i in range(n) if nn[i] >= 0.6 * d0 and nn[i] <= 1.5 * d0]
    s4 = 0.0
    c4 = 0.0
    pair_d = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = pts[j]["x"] - pts[i]["x"]
            dy = pts[j]["y"] - pts[i]["y"]
            dd = math.hypot(dx, dy)
            if dd >= 0.75 * d0 and dd <= 1.25 * d0:
                phi = math.atan2(dy, dx)
                s4 += math.sin(4 * phi)
                c4 += math.cos(4 * phi)
                pair_d.append(dd)
    if len(pair_d) == 0:
        err(tag + ": no lattice neighbour pairs")
        return None
    theta = math.atan2(s4, c4) / 4.0
    d = sum(pair_d) / len(pair_d)
    e1 = (math.cos(theta), math.sin(theta))
    e2 = (-math.sin(theta), math.cos(theta))
    mx = sum(p["x"] for p in pts) / len(pts)
    my = sum(p["y"] for p in pts) / len(pts)
    ref = None
    for p in pts:
        dd = math.hypot(p["x"] - mx, p["y"] - my)
        if ref is None or dd < ref[0]:
            ref = (dd, p)
    ref = ref[1]
    cells = {}
    for p in pts:
        du = ((p["x"] - ref["x"]) * e1[0] + (p["y"] - ref["y"]) * e1[1]) / d
        dv = ((p["x"] - ref["x"]) * e2[0] + (p["y"] - ref["y"]) * e2[1]) / d
        i = int(round(du))
        j = int(round(dv))
        res = math.hypot(du - i, dv - j)
        if res > 0.3:
            continue
        key = (i, j)
        if key not in cells or res < cells[key][0]:
            cells[key] = (res, p)
    if len(cells) < 4:
        err(tag + ": lattice assignment failed")
        return None
    imin = min(k[0] for k in cells)
    jmin = min(k[1] for k in cells)
    grid = {}
    for k in cells:
        grid[(k[0] - imin, k[1] - jmin)] = cells[k][1]
    # affine least squares c = o + i*A + j*B, per coordinate
    sol = []
    for coord in ("x", "y", "z"):
        m = [[0.0] * 3 for r in range(3)]
        b = [0.0, 0.0, 0.0]
        for k in grid:
            row = (1.0, float(k[0]), float(k[1]))
            val = grid[k][coord]
            for r in range(3):
                b[r] += row[r] * val
                for c in range(3):
                    m[r][c] += row[r] * row[c]
        sol.append(solve3(m, b))
    ni = max(k[0] for k in grid) + 1
    nj = max(k[1] for k in grid) + 1
    resid = []
    for k in grid:
        if sol[0] is None or sol[1] is None:
            break
        px = sol[0][0] + sol[0][1] * k[0] + sol[0][2] * k[1]
        py = sol[1][0] + sol[1][1] * k[0] + sol[1][2] * k[1]
        resid.append(math.hypot(px - grid[k]["x"], py - grid[k]["y"]))
    lat = {"grid": grid, "d": d, "theta": theta, "dims": (ni, nj), "affine": sol,
           "e1": e1, "e2": e2}
    report["lattice_" + tag] = {
        "spacing_m": round(d, 5), "theta_deg": round(math.degrees(theta), 3),
        "dims": [ni, nj], "cells": len(grid), "rejected_blobs": n - len(grid),
        "max_resid_m": round(max(resid), 5) if len(resid) > 0 else None,
        "cells_xyz": [[k[0], k[1], round(grid[k]["x"], 4), round(grid[k]["y"], 4),
                       round(grid[k]["z"], 4)] for k in sorted(grid.keys())],
    }
    return lat


def lattice_point(lat, key):
    if key in lat["grid"]:
        g = lat["grid"][key]
        return (g["x"], g["y"], g["z"])
    s = lat["affine"]
    if s[0] is None or s[1] is None or s[2] is None:
        return None
    return (s[0][0] + s[0][1] * key[0] + s[0][2] * key[1],
            s[1][0] + s[1][1] * key[0] + s[1][2] * key[1],
            s[2][0] + s[2][1] * key[0] + s[2][2] * key[1])


def nearest_cell(lat, x, y):
    best = None
    for k in lat["grid"]:
        g = lat["grid"][k]
        dd = math.hypot(x - g["x"], y - g["y"])
        if best is None or dd < best[0]:
            best = (dd, k)
    return best


def red_count(rgb, pu, pv, half):
    if pu is None:
        return -1
    h = len(rgb)
    w = len(rgb[0])
    cnt = 0
    for v in range(int(pv) - half, int(pv) + half + 1):
        if v < 0 or v >= h:
            continue
        for u in range(int(pu) - half, int(pu) + half + 1):
            if u < 0 or u >= w:
                continue
            px = rgb[v][u]
            r = int(px[0]); g = int(px[1]); b = int(px[2])
            if r > 150 and g < 50 and b < 50:
                cnt += 1
    return cnt


# ----------------------------------------------------------------------------------------
def main():
    ctx = get_task_context()
    report["goal"] = str(ctx.language)
    max_steps = None
    try:
        meta = robomme_plus.get_task_metadata()
        md = {}
        for key in meta:
            md[str(key)] = str(meta[key])[:400]
        report["task_metadata"] = md
        max_steps = int(meta["max_steps"])
    except Exception as exc:
        err("task metadata: " + str(exc))
    try:
        cmeta = robomme_plus.get_controller_metadata()
        cm = {}
        for key in cmeta:
            cm[str(key)] = str(cmeta[key])[:400]
        report["controller_metadata"] = cm
    except Exception as exc:
        err("controller metadata: " + str(exc))

    state0 = get_robot_state()
    base_frame = state0.base_frame
    emb = state0.embodiment
    q_home = fl(state0.joint_positions["primary"])
    ee0 = state0.end_effector_poses["primary"]
    report["live_start"] = {"joints": [round(v, 4) for v in q_home],
                            "ee_pos": [round(v, 4) for v in fl(ee0.position)],
                            "ee_quat": [round(v, 4) for v in fl(ee0.quaternion_wxyz)],
                            "t": float(state0.timestamp_s), "base_frame": str(base_frame)}

    video = robomme_plus.get_conditioning_video()
    nvid = len(video)
    report["video_frames"] = nvid
    if nvid == 0:
        raise ValueError("no conditioning video: pattern cannot be observed")

    # --- camera convention from the live front view (table must be horizontal) -------------
    live = get_observation()
    live_ci = cam_info(live, "front")
    live_depth = as_rows(live.cameras["front"].depth_m)
    best_conv, conv_scores = choose_convention(live_depth, live_ci)
    report["convention_scores"] = conv_scores
    if best_conv is None:
        raise ValueError("could not determine camera convention")
    conv = best_conv[1]
    report["convention"] = conv[0] + "_" + conv[1]

    # --- video lattice -----------------------------------------------------------------------
    v0 = video[0]
    vcomps, vplane, vci = detect_buttons(v0, conv, None, "video0")
    vlat = fit_lattice(vcomps, "video0")
    if vlat is None:
        raise ValueError("video button lattice not found")

    # --- video end-effector track ------------------------------------------------------------
    track = []
    for fi in range(nvid):
        fr = video[fi]
        rs = fr.robot_state
        pose = rs.end_effector_poses["primary"]
        p = fl(pose.position)
        track.append({"i": fi, "t": float(fr.timestamp_s), "p": p, "q": fl(pose.quaternion_wxyz),
                      "j": fl(rs.joint_positions["primary"])})
    zs = [f["p"][2] for f in track]
    zmin = min(zs)
    report["video_track"] = {
        "t0": track[0]["t"], "t1": track[-1]["t"], "zmin": round(zmin, 4), "zmax": round(max(zs), 4),
        "start_pos": [round(v, 4) for v in track[0]["p"]],
        "start_joints": [round(v, 4) for v in track[0]["j"]],
        "z_profile": [[round(f["t"], 2), round(f["p"][0], 4), round(f["p"][1], 4), round(f["p"][2], 4)]
                      for f in track[::max(1, nvid // 120)]],
    }
    low_tol = 0.02
    low = [f for f in track if f["p"][2] <= zmin + low_tol]
    d = vlat["d"]
    # tip xy offset relative to the end-effector xy, from closest approaches
    offs = []
    for f in low:
        nc = nearest_cell(vlat, f["p"][0], f["p"][1])
        if nc is not None and nc[0] < 0.5 * d:
            g = vlat["grid"][nc[1]]
            offs.append((f["p"][0] - g["x"], f["p"][1] - g["y"]))
    if len(offs) == 0:
        raise ValueError("video track never comes near a button")
    off_x = median([o[0] for o in offs])
    off_y = median([o[1] for o in offs])
    # per-frame planar speed from neighbouring recorded frames (timestamps, not indices)
    for k in range(nvid):
        a = track[max(0, k - 1)]
        b = track[min(nvid - 1, k + 1)]
        dt = b["t"] - a["t"]
        if dt > 1e-6:
            track[k]["v"] = math.sqrt(sum((b["p"][i] - a["p"][i]) ** 2 for i in range(3))) / dt
        else:
            track[k]["v"] = 0.0
    # a registration in the video is a DWELL: the stick rests (near-zero speed) on a button
    dwell_v = 0.015
    dwells = []
    cur = None
    for f in track:
        is_rest = f["v"] < dwell_v and f["p"][2] <= zmin + 0.01
        if is_rest:
            if cur is not None and f["i"] <= cur["last_i"] + 2:
                cur["frames"].append(f)
                cur["last_i"] = f["i"]
            else:
                cur = {"frames": [f], "last_i": f["i"]}
                dwells.append(cur)
    visits = []
    rejected_dwells = []
    for dw in dwells:
        fr_list = dw["frames"]
        dur = fr_list[-1]["t"] - fr_list[0]["t"]
        mx = sum(f["p"][0] for f in fr_list) / len(fr_list) - off_x
        my = sum(f["p"][1] for f in fr_list) / len(fr_list) - off_y
        nc = nearest_cell(vlat, mx, my)
        if nc is None or nc[0] >= 0.3 * d or len(fr_list) < 2:
            rejected_dwells.append([round(fr_list[0]["t"], 2), round(dur, 2), len(fr_list),
                                    round(nc[0] / d, 3) if nc is not None else None])
            continue
        best = None
        for f in fr_list:
            dd = math.hypot(f["p"][0] - off_x - vlat["grid"][nc[1]]["x"],
                            f["p"][1] - off_y - vlat["grid"][nc[1]]["y"])
            if best is None or dd < best[0]:
                best = (dd, f)
        visits.append({"cell": nc[1], "first_i": fr_list[0]["i"], "last_i": fr_list[-1]["i"],
                       "t0": fr_list[0]["t"], "t1": fr_list[-1]["t"], "dmin": nc[0],
                       "best": best[1], "n": len(fr_list)})
    report["video_dwells_rejected"] = rejected_dwells
    # merge consecutive duplicates separated by short gaps
    seq = []
    for vv in visits:
        if len(seq) > 0 and seq[-1]["cell"] == vv["cell"]:
            seq[-1]["last_i"] = vv["last_i"]
            seq[-1]["t1"] = vv["t1"]
            if vv["dmin"] < seq[-1]["dmin"]:
                seq[-1]["dmin"] = vv["dmin"]
                seq[-1]["best"] = vv["best"]
        else:
            seq.append(vv)
    if len(seq) == 0:
        raise ValueError("no touches decoded from the video")
    dzs = []
    for vv in seq:
        g = vlat["grid"][vv["cell"]]
        dzs.append(vv["best"]["p"][2] - g["z"])
    dz_touch = median(dzs)
    q_touch = seq[0]["best"]["q"]
    # transit lift and speed between consecutive touches
    for k in range(1, len(seq)):
        a = seq[k - 1]
        b = seq[k]
        seg = [f for f in track if f["i"] >= a["last_i"] and f["i"] <= b["first_i"]]
        b["transit_zmax_rel"] = max(f["p"][2] for f in seg) - zmin if len(seg) > 0 else 0.0
    low_speeds = []
    for k in range(1, nvid):
        f0 = track[k - 1]
        f1 = track[k]
        dt = f1["t"] - f0["t"]
        if dt > 1e-6 and f0["p"][2] <= zmin + low_tol and f1["p"][2] <= zmin + low_tol:
            low_speeds.append(math.hypot(f1["p"][0] - f0["p"][0], f1["p"][1] - f0["p"][1]) / dt)
    vid_speed = None
    if len(low_speeds) > 0:
        ls = sorted(low_speeds)
        vid_speed = ls[int(0.75 * (len(ls) - 1))]
    report["decoded"] = {
        "offset_xy": [round(off_x, 5), round(off_y, 5)], "n_offsets": len(offs),
        "dz_touch_ee_minus_button_top": round(dz_touch, 5),
        "dz_list": [round(v, 4) for v in dzs],
        "q_touch": [round(v, 4) for v in q_touch],
        "video_low_speed_mps": round(vid_speed, 4) if vid_speed is not None else None,
        "sequence": [[s["cell"][0], s["cell"][1], round(s["t0"], 2), round(s["t1"], 2),
                      round(s["dmin"] / d, 3), round(s.get("transit_zmax_rel", 0.0), 4)] for s in seq],
    }

    # --- red-onset cross-check in the video (diagnostic only) --------------------------------
    try:
        pix = {}
        for k in vlat["grid"]:
            g = vlat["grid"][k]
            pix[k] = world_to_pix(conv, (g["x"], g["y"], g["z"]), vci)
        red_state = {}
        onsets = []
        for fi in range(nvid):
            rgb = video[fi].cameras["front"].rgb
            for k in pix:
                pp = pix[k]
                if pp is None:
                    continue
                c = red_count(rgb, pp[0], pp[1], 2)
                on = c >= 3
                if on and not red_state.get(k, False):
                    onsets.append([k[0], k[1], round(float(video[fi].timestamp_s), 2)])
                red_state[k] = on
        report["video_red_onsets"] = onsets
    except Exception as exc:
        err("red onset diagnostic: " + str(exc))

    # --- live lattice --------------------------------------------------------------------------
    lcomps, lplane, lci = detect_buttons(live, conv, None, "live")
    llat = fit_lattice(lcomps, "live")
    use_video_positions = False
    if llat is None:
        err("live lattice failed; falling back to the video lattice positions")
        llat = vlat
        use_video_positions = True
    if llat["dims"] != vlat["dims"]:
        err("lattice dims differ: video " + str(vlat["dims"]) + " live " + str(llat["dims"]))
    targets = []
    for s in seq:
        lp = lattice_point(llat, s["cell"])
        if lp is None:
            raise ValueError("cell missing in live lattice " + str(s["cell"]))
        targets.append({"cell": s["cell"],
                        "pos": (lp[0] + off_x, lp[1] + off_y, lp[2] + dz_touch),
                        "lift": s.get("transit_zmax_rel", 0.0)})
    report["targets"] = [[t["cell"][0], t["cell"][1]] + [round(v, 4) for v in t["pos"]] for t in targets]

    # --- motion helpers ------------------------------------------------------------------------
    exec_log = []
    status = {"terminated": False, "truncated": False, "ticks": 0, "ik_fail": 0}
    motion = {"q": list(q_home)}
    grip_cmd = float(state0.gripper_positions["primary"])


    def solve(pos, seed):
        res = solve_ik(Pose(list(pos), list(q_touch), base_frame), seed_joints=list(seed))
        if not res.ok:
            return None
        try:
            return fl(res.joint_positions)
        except Exception:
            pass
        try:
            return fl(res.solution)
        except Exception:
            pass
        return fl(res.joints)


    def send(q):
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=list(q), gripper_position=grip_cmd, embodiment=emb)}))
        status["ticks"] += 1
        motion["q"] = list(q)
        term = False
        trunc = False
        try:
            term = bool(reply.terminated)
        except Exception:
            term = False
        try:
            trunc = bool(reply.truncated)
        except Exception:
            trunc = False
        if term or trunc:
            status["terminated"] = term
            status["truncated"] = trunc
            return False
        return True


    def joint_move(q_to, rad_per_tick):
        q_from = list(motion["q"])
        diff = max(abs(q_to[i] - q_from[i]) for i in range(7))
        n = max(1, int(math.ceil(diff / rad_per_tick)))
        for s in range(1, n + 1):
            a = s / float(n)
            if not send([q_from[i] + a * (q_to[i] - q_from[i]) for i in range(7)]):
                return False
        return True


    def cart_line(p_from, p_to, speed):
        length = math.sqrt(sum((p_to[i] - p_from[i]) ** 2 for i in range(3)))
        nsub = max(1, int(math.ceil(length / 0.01)))
        ticks_per_sub = max(1, int(round((length / nsub) / (speed / 20.0))))
        for s in range(1, nsub + 1):
            a = s / float(nsub)
            p = [p_from[i] + a * (p_to[i] - p_from[i]) for i in range(3)]
            q = solve(p, motion["q"])
            if q is None:
                status["ik_fail"] += 1
                continue
            q_from = list(motion["q"])
            for tk in range(1, ticks_per_sub + 1):
                b = tk / float(ticks_per_sub)
                if not send([q_from[i] + b * (q[i] - q_from[i]) for i in range(7)]):
                    return False
        return True


    def hold(n):
        for tk in range(n):
            if not send(motion["q"]):
                return False
        return True


    def ee_now():
        return fl(get_robot_state().end_effector_poses["primary"].position)

    def settle(need, extra, max_ticks):
        """Hold the commanded joints until the measured tip has been at rest for `need`
        consecutive ticks (planar+vertical speed < 4 mm/s), then `extra` more ticks."""
        prev = ee_now()
        count = 0
        used = 0
        speeds = []
        while used < max_ticks:
            if not send(motion["q"]):
                return False, used, speeds
            used += 1
            p = ee_now()
            sp = math.sqrt(sum((p[i] - prev[i]) ** 2 for i in range(3))) / 0.05
            speeds.append(round(sp, 4))
            prev = p
            if sp < 0.004:
                count += 1
            else:
                count = 0
            if count >= need:
                break
        for tk in range(extra):
            if not send(motion["q"]):
                return False, used, speeds
            used += 1
        return True, used, speeds

    def touch_check(cell):
        out = {}
        try:
            ob = get_observation()
            lp = lattice_point(llat, cell)
            for cam_name in ("front", "wrist"):
                ci = cam_info(ob, cam_name)
                pp = world_to_pix(conv, lp, ci)
                rgb = ob.cameras[cam_name].rgb
                half = 2 if cam_name == "front" else 8
                out[cam_name] = red_count(rgb, pp[0], pp[1], half) if pp is not None else None
            out["t"] = float(ob.timestamp_s)
            ee = ob.robot_state.end_effector_poses["primary"]
            out["ee"] = [round(v, 4) for v in fl(ee.position)]
        except Exception as exc:
            out["error"] = str(exc)
        return out


    speed = 0.05
    if vid_speed is not None:
        speed = min(0.10, max(0.04, vid_speed))
    # budget ticks against the native horizon
    path_len = 0.0
    for k in range(1, len(targets)):
        path_len += math.sqrt(sum((targets[k]["pos"][i] - targets[k - 1]["pos"][i]) ** 2 for i in range(3)))
    if max_steps is not None:
        est = path_len / (speed / 20.0) + 14 * len(targets) + 120
        if est > 0.8 * max_steps:
            speed = speed * est / (0.8 * max_steps)
    report["plan"] = {"speed_mps": round(speed, 4), "path_len_m": round(path_len, 4),
                      "max_steps": max_steps, "n_targets": len(targets)}

    # --- execute -------------------------------------------------------------------------------
    alive = True
    try:
        first = targets[0]["pos"]
        pre = (first[0], first[1], first[2] + 0.08)
        seed = seq[0]["best"]["j"]
        q_pre = solve(pre, seed)
        if q_pre is None:
            q_pre = solve(pre, motion["q"])
        if q_pre is None:
            raise ValueError("IK failed for pre-touch pose")
        alive = joint_move(q_pre, 0.03)
        exec_log.append({"seg": "approach", "ticks": status["ticks"]})
        if alive:
            alive = cart_line(pre, first, 0.04)
        st = None
        if alive:
            st = settle(4, 2, 25)
            alive = st[0]
        exec_log.append({"seg": "touch0", "cell": list(targets[0]["cell"]), "ticks": status["ticks"],
                         "settle": [st[1], st[2]] if st is not None else None,
                         "check": touch_check(targets[0]["cell"]) if alive else None})
        for k in range(1, len(targets)):
            if not alive:
                break
            a = targets[k - 1]["pos"]
            b = targets[k]["pos"]
            lift = targets[k]["lift"]
            if lift > 0.02:
                hgt = min(lift, 0.06)
                a2 = (a[0], a[1], a[2] + hgt)
                b2 = (b[0], b[1], b[2] + hgt)
                alive = cart_line(a, a2, speed)
                if alive:
                    alive = cart_line(a2, b2, speed)
                if alive:
                    alive = cart_line(b2, b, speed)
            else:
                alive = cart_line(a, b, speed)
            st = None
            if alive:
                st = settle(4, 2, 25)
                alive = st[0]
            exec_log.append({"seg": k, "cell": list(targets[k]["cell"]), "ticks": status["ticks"],
                             "settle": [st[1], st[2]] if st is not None else None,
                             "check": touch_check(targets[k]["cell"]) if alive else None})
        if alive:
            alive = hold(10)
            exec_log.append({"seg": "final_hold", "ticks": status["ticks"]})
        if alive:
            last = targets[-1]["pos"]
            alive = cart_line(last, (last[0], last[1], last[2] + 0.05), 0.04)
            if alive:
                alive = hold(10)
            exec_log.append({"seg": "lift", "ticks": status["ticks"]})
    except Exception as exc:
        err("execution: " + str(exc))

    report["exec_log"] = exec_log
    report["status"] = status
    if not status["terminated"] and not status["truncated"]:
        try:
            report["final_t"] = float(get_robot_state().timestamp_s)
        except Exception as exc:
            err("final state: " + str(exc))


try:
    main()
except Exception as exc:
    err("fatal: " + str(exc))
result = report
