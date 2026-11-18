import math

# MoveCube v02 (from v01) -- pull manner: measure the HELD tool from the front camera while hovering
# (depth that became closer than the initial scene, in the tool's height band), plan the hand yaw and the
# cube placement in the tool's free space so that the tool caught the cube's trailing face and caged it
# laterally (cube footprint = min-area square of its segmented cloud), verify once more, then pull.
# v01 placed the hook from the rest-pose cloud, which was truncated at the image border, and landed on the cube.
# MoveCube v00 -- "move the cube to the target in the same manner shown in the video".
# Strategy (all task-dependent quantities derived at runtime):
#  1. Calibrate the front-camera back-projection convention from the live depth image
#     (choose the convention under which the table is a horizontal plane).
#  2. Segment above-table objects in the live front view and in conditioning-video frame 0
#     by height above the table (connected components of the RGB-D cloud).
#  3. From the video robot_state, find when the gripper first closed and where the
#     end-effector was: the video object nearest that point is the tool that was used.
#  4. Match that tool to a live object by appearance (colour histogram) and size; grasp it
#     at the same relative position along its long axis.
#  5. Rotate the held tool so its far end points along cube->target and push the cube onto
#     the target with that far end (compact tools, e.g. a ball, push with the object itself).

report = {"version": "v02", "phase": "start", "events": []}
result = report


def note(name, value):
    report["events"].append([name, value])


def fnum(value, digits=4):
    try:
        return round(float(value), digits)
    except Exception:
        return None


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    index = int((len(ordered) - 1) * fraction)
    return ordered[index]


def quat_to_matrix(q):
    w = float(q[0]); x = float(q[1]); y = float(q[2]); z = float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w = w / n; x = x / n; y = y / n; z = z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def quat_mul(a, b):
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def yaw_quat(angle):
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]


def wrap(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def to_list(array):
    try:
        return array.tolist()
    except Exception:
        rows = []
        for row in array:
            rows.append(list(row))
        return rows


def camera_data(cam):
    rgb = to_list(cam.rgb)
    depth = to_list(cam.depth_m)
    k = to_list(cam.intrinsics)
    pose = cam.camera_pose
    rot = quat_to_matrix(pose.quaternion_wxyz)
    pos = [float(v) for v in pose.position]
    return {"rgb": rgb, "depth": depth, "fx": float(k[0][0]), "fy": float(k[1][1]),
            "cx": float(k[0][2]), "cy": float(k[1][2]), "rot": rot, "pos": pos,
            "h": len(depth), "w": len(depth[0])}


def backproject(cd, u, v, convention):
    d = cd["depth"][v][u]
    try:
        d = float(d)
    except Exception:
        return None
    if not (d > 0.01 and d < 10.0):
        return None
    xn = (u + 0.5 - cd["cx"]) / cd["fx"]
    yn = (v + 0.5 - cd["cy"]) / cd["fy"]
    if convention[1] == "ray":
        scale = d / math.sqrt(xn * xn + yn * yn + 1.0)
    else:
        scale = d
    if convention[0] == "cv":
        pc = [xn * scale, yn * scale, scale]
    else:
        pc = [xn * scale, -yn * scale, -scale]
    r = cd["rot"]; t = cd["pos"]
    return [r[0][0] * pc[0] + r[0][1] * pc[1] + r[0][2] * pc[2] + t[0],
            r[1][0] * pc[0] + r[1][1] * pc[1] + r[1][2] * pc[2] + t[1],
            r[2][0] * pc[0] + r[2][1] * pc[1] + r[2][2] * pc[2] + t[2]]


def calibrate_convention(cd):
    """Pick the back-projection under which most pixels form one horizontal plane."""
    best = None
    scores = []
    for convention in (("cv", "z"), ("cv", "ray"), ("gl", "z"), ("gl", "ray")):
        zs = []
        for v in range(4, cd["h"], 6):
            for u in range(4, cd["w"], 6):
                p = backproject(cd, u, v, convention)
                if p is not None:
                    zs.append(p[2])
        if len(zs) < 50:
            continue
        zm = quantile(zs, 0.5)
        inliers = sum(1 for z in zs if abs(z - zm) < 0.006)
        frac = inliers / float(len(zs))
        if zm > cd["pos"][2] - 0.1:
            frac = frac - 1.0  # a table above the camera is the mirrored convention
        scores.append([convention[0] + "/" + convention[1], round(frac, 3), fnum(zm)])
        if best is None or frac > best[0]:
            best = [frac, convention, zm]
    note("convention_scores", scores)
    return best


def table_height(cd, convention):
    zs = []
    for v in range(2, cd["h"], 4):
        for u in range(2, cd["w"], 4):
            p = backproject(cd, u, v, convention)
            if p is not None:
                zs.append(p[2])
    zm = quantile(zs, 0.5)
    near = [z for z in zs if abs(z - zm) < 0.01]
    return quantile(near, 0.5)


def color_bin(rgb):
    return (int(rgb[0]) * 3 // 256) * 9 + (int(rgb[1]) * 3 // 256) * 3 + (int(rgb[2]) * 3 // 256)


def components(cd, convention, table_z, stride, min_h, max_h, min_points):
    """Connected above-table regions (grid stride in pixels) with xyz + colour per cell."""
    h = cd["h"]; w = cd["w"]
    gh = (h + stride - 1) // stride
    gw = (w + stride - 1) // stride
    cells = {}
    for gv in range(gh):
        v = gv * stride
        for gu in range(gw):
            u = gu * stride
            p = backproject(cd, u, v, convention)
            if p is None:
                continue
            dz = p[2] - table_z
            if dz < min_h or dz > max_h:
                continue
            if math.sqrt(p[0] * p[0] + p[1] * p[1]) < 0.15:
                continue
            cells[gv * gw + gu] = [p, cd["rgb"][v][u], u, v]
    seen = set()
    comps = []
    for key in cells:
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        members = []
        while len(stack) > 0:
            cur = stack.pop()
            members.append(cur)
            cv = cur // gw
            cu = cur % gw
            pc = cells[cur][0]
            for dv, du in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                nv = cv + dv
                nu = cu + du
                if nv < 0 or nu < 0 or nv >= gh or nu >= gw:
                    continue
                nk = nv * gw + nu
                if nk in seen or nk not in cells:
                    continue
                pn = cells[nk][0]
                dist = math.sqrt((pn[0] - pc[0]) ** 2 + (pn[1] - pc[1]) ** 2 + (pn[2] - pc[2]) ** 2)
                if dist < 0.025:
                    seen.add(nk)
                    stack.append(nk)
        if len(members) >= min_points:
            comps.append([cells[m] for m in members])
    return comps


def describe(comp, table_z):
    xs = [c[0][0] for c in comp]; ys = [c[0][1] for c in comp]; zs = [c[0][2] for c in comp]
    n = float(len(comp))
    mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs) / n
    syy = sum((y - my) ** 2 for y in ys) / n
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(len(xs))) / n
    angle = 0.5 * math.atan2(2 * sxy, sxx - syy)
    ax = [math.cos(angle), math.sin(angle)]
    perp = [-ax[1], ax[0]]
    along = [(xs[i] - mx) * ax[0] + (ys[i] - my) * ax[1] for i in range(len(xs))]
    across = [(xs[i] - mx) * perp[0] + (ys[i] - my) * perp[1] for i in range(len(xs))]
    length = quantile(along, 0.98) - quantile(along, 0.02)
    width = quantile(across, 0.98) - quantile(across, 0.02)
    hist = [0.0] * 27
    hist_pos = [0.0] * 27
    hist_neg = [0.0] * 27
    rsum = [0.0, 0.0, 0.0]
    for i in range(len(comp)):
        b = color_bin(comp[i][1])
        hist[b] += 1.0 / n
        if along[i] >= 0:
            hist_pos[b] += 1.0
        else:
            hist_neg[b] += 1.0
        for j in range(3):
            rsum[j] += float(comp[i][1][j]) / n
    sp = max(1.0, sum(hist_pos)); sn = max(1.0, sum(hist_neg))
    hist_pos = [v / sp for v in hist_pos]
    hist_neg = [v / sn for v in hist_neg]
    return {"n": len(comp), "center": [mx, my], "axis": ax, "perp": perp, "length": length,
            "width": width, "height": quantile(zs, 0.98) - table_z, "hist": hist,
            "hist_pos": hist_pos, "hist_neg": hist_neg, "mean_rgb": rsum,
            "along": along, "across": across, "points": [c[0] for c in comp],
            "pixel": [sum(c[2] for c in comp) / n, sum(c[3] for c in comp) / n],
            "uv": [[c[2], c[3]] for c in comp]}


def summary(desc):
    return {"n": desc["n"], "center": [fnum(v) for v in desc["center"]],
            "axis": [fnum(v, 3) for v in desc["axis"]], "length": fnum(desc["length"]),
            "width": fnum(desc["width"]), "height": fnum(desc["height"]),
            "mean_rgb": [int(v) for v in desc["mean_rgb"]],
            "pixel": [int(v) for v in desc["pixel"]]}


def intersection(a, b):
    return sum(min(a[i], b[i]) for i in range(len(a)))


def size_sim(a, b):
    return min(a, b) / max(a, b, 1e-6)


def cloud_center(points):
    return [quantile([p[0] for p in points], 0.5), quantile([p[1] for p in points], 0.5),
            quantile([p[2] for p in points], 0.5)]


def localize_points(prompts, base_frame):
    for prompt in prompts:
        try:
            found = segment_text("front", prompt)
        except Exception as error:
            note("segment_error", [prompt, str(error)[:200]])
            continue
        if not found.ok or len(found.segmentations) == 0:
            note("segment_none", prompt)
            continue
        cand = found.segmentations[0]
        try:
            cloud = mask_to_point_cloud(cand, "front", target_frame=base_frame)
        except Exception as error:
            note("cloud_error", [prompt, str(error)[:200]])
            continue
        pts = [[float(p[0]), float(p[1]), float(p[2])] for p in cloud.points]
        note("segment_ok", [prompt, fnum(cand.score, 3), len(pts),
                            [fnum(v) for v in cand.box_xyxy] if cand.box_xyxy is not None else None])
        if len(pts) >= 10:
            return pts
    return None


def move(position, quat, base_frame, label, tol=0.01, steps=250):
    pose = Pose([float(position[0]), float(position[1]), float(position[2])], quat, base_frame)
    res = move_to_pose(pose, tolerance=tol, max_steps=steps)
    info = {"label": label, "ok": bool(res.ok), "target": [fnum(v) for v in position]}
    try:
        info["status"] = str(res.status)
        info["steps"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        info["terminated"] = bool(res.terminated)
        info["truncated"] = bool(res.truncated)
    except Exception:
        pass
    try:
        if not res.ok:
            info["error"] = str(res.error)[:300]
    except Exception:
        pass
    note("move", info)
    ended = False
    try:
        ended = bool(res.terminated) or bool(res.truncated)
    except Exception:
        ended = False
    return [res, ended]


def ik_ok(position, quat, base_frame):
    try:
        ik = solve_ik(Pose([float(position[0]), float(position[1]), float(position[2])], quat, base_frame))
        return bool(ik.ok)
    except Exception:
        return False


def finished(flag):
    report["ended_by_env"] = flag


def main():
    ctx = get_task_context()
    report["goal"] = ctx.language
    try:
        meta = robomme_plus.get_task_metadata()
        report["metadata"] = {k: str(meta[k])[:200] for k in meta}
    except Exception as error:
        note("metadata_error", str(error)[:200])
    state = get_robot_state()
    base = state.base_frame
    home_pose = state.end_effector_poses["primary"]
    home_q = [float(v) for v in home_pose.quaternion_wxyz]
    home_p = [float(v) for v in home_pose.position]
    report["home"] = {"p": [fnum(v) for v in home_p], "q": [fnum(v) for v in home_q],
                      "gripper": fnum(state.gripper_positions["primary"])}
    hand_rot = quat_to_matrix(home_q)
    home_closing_yaw = math.atan2(hand_rot[1][1], hand_rot[0][1])
    report["home_hand_axes"] = [[fnum(hand_rot[r][c], 3) for r in range(3)] for c in range(3)]

    # ---------- live scene ----------
    obs = get_observation()
    live = camera_data(obs.cameras["front"])
    best = calibrate_convention(live)
    if best is None:
        report["phase"] = "no_depth"
        return
    convention = best[1]
    report["convention"] = convention[0] + "/" + convention[1]
    table_z = table_height(live, convention)
    report["table_z"] = fnum(table_z)

    cube_pts = localize_points(("red cube", "red block", "cube"), base)
    target_pts = localize_points(("purple and white target", "bullseye target", "purple target", "target"), base)
    if cube_pts is None or target_pts is None:
        report["phase"] = "cube_or_target_missing"
        return
    cube_c = cloud_center(cube_pts)
    target_c = cloud_center(target_pts)
    cube_half = 0.25 * ((quantile([p[0] for p in cube_pts], 0.95) - quantile([p[0] for p in cube_pts], 0.05)) +
                        (quantile([p[1] for p in cube_pts], 0.95) - quantile([p[1] for p in cube_pts], 0.05)))
    cube_sq = cube_square(cube_pts)
    cube_c = [cube_sq["c"][0], cube_sq["c"][1], cube_c[2]]
    report["cube_square"] = {"c": [fnum(v) for v in cube_sq["c"]], "yaw_deg": fnum(math.degrees(cube_sq["yaw"]), 1),
                             "h": fnum(cube_sq["h"]), "top": fnum(cube_sq["top"])}
    report["cube"] = [fnum(v) for v in cube_c]
    report["cube_half"] = fnum(cube_half)
    report["target"] = [fnum(v) for v in target_c]

    live_comps = components(live, convention, table_z, 2, 0.004, 0.2, 6)
    live_desc = []
    for comp in live_comps:
        d = describe(comp, table_z)
        dc = math.sqrt((d["center"][0] - cube_c[0]) ** 2 + (d["center"][1] - cube_c[1]) ** 2)
        d["cube_dist"] = dc
        live_desc.append(d)
    report["live_components"] = [summary(d) for d in live_desc]

    # ---------- conditioning video ----------
    video = robomme_plus.get_conditioning_video()
    report["video_frames"] = len(video)
    if len(video) == 0:
        report["phase"] = "no_video"
        return
    grip = []
    for i in range(len(video)):
        rs = video[i].robot_state
        grip.append(float(rs.gripper_positions["primary"]))
    grasp_index = None
    running = grip[0]
    for i in range(1, len(grip)):
        if grip[i] > running:
            running = grip[i]
        if running > 0.5 and grip[i] < running - 0.2:
            grasp_index = i
            break
    report["video_gripper_samples"] = [fnum(grip[i], 3) for i in range(0, len(grip), max(1, len(grip) // 40))]
    traj = []
    for i in range(0, len(video), max(1, len(video) // 60)):
        try:
            pe = video[i].robot_state.end_effector_poses["primary"]
            traj.append([i, fnum(video[i].timestamp_s, 2), [fnum(v, 3) for v in pe.position],
                         [fnum(v, 3) for v in pe.quaternion_wxyz], fnum(grip[i], 2)])
        except Exception:
            pass
    report["video_ee_samples"] = traj
    if grasp_index is None:
        report["phase"] = "video_no_grasp"
        return
    gpose = video[grasp_index].robot_state.end_effector_poses["primary"]
    gpt = [float(v) for v in gpose.position]
    report["video_grasp"] = {"index": grasp_index, "t": fnum(video[grasp_index].timestamp_s, 2),
                             "ee": [fnum(v) for v in gpt]}
    v0 = camera_data(video[0].cameras["front"])
    vbest = calibrate_convention(v0)
    vconv = vbest[1] if vbest is not None else convention
    vtable = table_height(v0, vconv)
    report["video_table_z"] = fnum(vtable)
    vcomps = components(v0, vconv, vtable, 2, 0.004, 0.2, 6)
    vdesc = [describe(c, vtable) for c in vcomps]
    report["video_components"] = [summary(d) for d in vdesc]
    tool_v = None
    tool_dist = None
    for d in vdesc:
        dmin = min(math.sqrt((p[0] - gpt[0]) ** 2 + (p[1] - gpt[1]) ** 2) for p in d["points"])
        if tool_dist is None or dmin < tool_dist:
            tool_dist = dmin
            tool_v = d
    if tool_v is None or tool_dist > 0.05:
        report["phase"] = "video_tool_not_found"
        report["video_tool_dist"] = fnum(tool_dist)
        return
    # orient the video tool axis so that the grasp lies on the negative side
    sg = (gpt[0] - tool_v["center"][0]) * tool_v["axis"][0] + (gpt[1] - tool_v["center"][1]) * tool_v["axis"][1]
    hist_grasp = tool_v["hist_neg"]; hist_far = tool_v["hist_pos"]
    if sg > 0:
        sg = -sg
        hist_grasp = tool_v["hist_pos"]; hist_far = tool_v["hist_neg"]
    elongated_v = tool_v["length"] > 1.8 * tool_v["width"]
    manner = "push"
    try:
        vax = [tool_v["axis"][0], tool_v["axis"][1]]
        sgv = (gpt[0] - tool_v["center"][0]) * vax[0] + (gpt[1] - tool_v["center"][1]) * vax[1]
        if sgv > 0:
            vax = [-vax[0], -vax[1]]
        valong = [(p[0] - tool_v["center"][0]) * vax[0] + (p[1] - tool_v["center"][1]) * vax[1] for p in tool_v["points"]]
        vfar = quantile(valong, 0.98)
        vtip = [tool_v["center"][0] + vax[0] * vfar, tool_v["center"][1] + vax[1] * vfar]
        vo = [vtip[0] - gpt[0], vtip[1] - gpt[1]]
        vol = math.sqrt(vo[0] ** 2 + vo[1] ** 2)
        rg = quat_to_matrix(video[grasp_index].robot_state.end_effector_poses["primary"].quaternion_wxyz)
        yaw_g = math.atan2(rg[1][1], rg[0][1])
        along_motion = 0.0
        low_motion = 0.0
        prev = None
        for i in range(grasp_index, len(video)):
            if grip[i] > 0.5 * (grip[0] + grip[grasp_index]) and grip[0] > grip[grasp_index]:
                break
            pe = video[i].robot_state.end_effector_poses["primary"]
            pp = [float(v) for v in pe.position]
            if prev is not None and pp[2] < gpt[2] + 0.035 and prev[2] < gpt[2] + 0.035:
                ri = quat_to_matrix(pe.quaternion_wxyz)
                yi = math.atan2(ri[1][1], ri[0][1])
                ang = math.atan2(vo[1], vo[0]) + wrap(yi - yaw_g)
                dxv = pp[0] - prev[0]; dyv = pp[1] - prev[1]
                along_motion += dxv * math.cos(ang) + dyv * math.sin(ang)
                low_motion += math.sqrt(dxv * dxv + dyv * dyv)
            prev = pp
        report["video_manner"] = {"along_motion": fnum(along_motion), "low_motion": fnum(low_motion),
                                  "tool_offset_len": fnum(vol)}
        if vol > 0.05 and along_motion < -0.02 and along_motion < -0.3 * low_motion:
            manner = "pull"
    except Exception as error:
        note("manner_error", str(error)[:200])
    report["manner"] = manner
    manner_video = manner
    rel = sg / max(0.5 * tool_v["length"], 1e-3)
    report["video_tool"] = summary(tool_v)
    report["video_tool"]["grasp_rel"] = fnum(rel, 3)
    report["video_tool"]["elongated"] = elongated_v
    report["video_tool"]["grasp_dist"] = fnum(tool_dist)

    # ---------- match the tool in the live scene ----------
    scored = []
    for d in live_desc:
        if d["cube_dist"] < 0.03:
            continue
        s_hist = intersection(d["hist"], tool_v["hist"])
        s_size = (size_sim(d["length"], tool_v["length"]) + size_sim(d["width"], tool_v["width"]) +
                  size_sim(d["height"], tool_v["height"])) / 3.0
        scored.append([s_hist + s_size, s_hist, s_size, d])
    scored = sorted(scored, key=score_key, reverse=True)
    report["live_match_scores"] = [[fnum(s[0], 3), fnum(s[1], 3), fnum(s[2], 3), summary(s[3])["center"]] for s in scored[:6]]
    if len(scored) == 0:
        report["phase"] = "live_tool_not_found"
        return
    tool = scored[0][3]
    axis = [tool["axis"][0], tool["axis"][1]]
    if elongated_v:
        keep = intersection(hist_grasp, tool["hist_neg"]) + intersection(hist_far, tool["hist_pos"])
        flip = intersection(hist_grasp, tool["hist_pos"]) + intersection(hist_far, tool["hist_neg"])
        if flip > keep:
            axis = [-axis[0], -axis[1]]
            along = [-a for a in tool["along"]]
        else:
            along = tool["along"]
        report["orientation_scores"] = [fnum(keep, 3), fnum(flip, 3)]
        lo = quantile(along, 0.02); hi = quantile(along, 0.98)
        grasp_s = rel * 0.5 * (hi - lo)
        far_s = hi
        gx = tool["center"][0] + axis[0] * grasp_s
        gy = tool["center"][1] + axis[1] * grasp_s
        # tip: median perpendicular offset of the far 10% of points
        far_pts = [tool["points"][i] for i in range(len(along)) if along[i] > hi - 0.1 * (hi - lo)]
        tx = quantile([p[0] for p in far_pts], 0.5)
        ty = quantile([p[1] for p in far_pts], 0.5)
        tip_extra = 0.0
    else:
        gx = tool["center"][0]; gy = tool["center"][1]
        tx = gx; ty = gy
        tip_extra = 0.5 * max(tool["width"], tool["length"])
    hook = None
    tool_rest = [[p[0], p[1]] for p in tool["points"]]
    if elongated_v:
        try:
            dense = dense_points(live, convention, table_z, tool)
            report["tool_dense_points"] = len(dense)
            if len(dense) >= 20:
                tool_rest = [[p[0], p[1]] for p in dense]
            pax = [axis[0], axis[1]]; ppx = [-axis[1], axis[0]]
            al = [(p[0] - gx) * pax[0] + (p[1] - gy) * pax[1] for p in dense]
            ac = [(p[0] - gx) * ppx[0] + (p[1] - gy) * ppx[1] for p in dense]
            far_al = quantile(al, 0.99)
            mid = [ac[i] for i in range(len(al)) if al[i] > 0.3 * far_al and al[i] < 0.7 * far_al]
            shaft_c = quantile(mid, 0.5) if len(mid) > 3 else 0.0
            shaft_half = 0.5 * (quantile(mid, 0.9) - quantile(mid, 0.1)) if len(mid) > 3 else 0.005
            farc = [ac[i] - shaft_c for i in range(len(al)) if al[i] > far_al - 0.05]
            cpos = quantile(farc, 0.97); cneg = quantile(farc, 0.03)
            cbar = cpos if cpos > -cneg else cneg
            report["hook_geometry"] = {"far_al": fnum(far_al), "shaft_c": fnum(shaft_c), "shaft_half": fnum(shaft_half),
                                       "cpos": fnum(cpos), "cneg": fnum(cneg), "n_far": len(farc)}
            if abs(cbar) > 2.0 * cube_half + 0.015:
                hook = {"far": far_al, "shaft_c": shaft_c, "shaft_half": max(shaft_half, 0.003), "c": cbar}
            tx = gx + pax[0] * far_al + ppx[0] * shaft_c
            ty = gy + pax[1] * far_al + ppx[1] * shaft_c
        except Exception as error:
            note("dense_error", str(error)[:200])
    if manner == "pull" and hook is None:
        note("pull_without_crossbar", "falling back to push")
        manner = "push"
    report["manner_used"] = manner
    near = [p[2] for p in tool["points"] if math.sqrt((p[0] - gx) ** 2 + (p[1] - gy) ** 2) < 0.02]
    if len(near) < 2:
        near = [p[2] for p in tool["points"]]
    local_h = quantile(near, 0.95) - table_z
    grasp_z = table_z + max(0.5 * local_h, 0.01)
    report["plan_tool"] = {"grasp": [fnum(gx), fnum(gy), fnum(grasp_z)], "tip": [fnum(tx), fnum(ty)],
                           "local_h": fnum(local_h), "axis": [fnum(axis[0], 3), fnum(axis[1], 3)],
                           "tip_extra": fnum(tip_extra)}

    # ---------- obstacles: every other above-table object ----------
    obstacles = []
    max_other_h = 0.0
    for d in live_desc:
        if d is tool:
            continue
        if d["cube_dist"] < 0.03 or d["height"] < 0.012:
            continue
        max_other_h = max(max_other_h, d["height"])
        for i in range(0, len(d["points"]), 2):
            obstacles.append(d["points"][i])
    report["obstacle_points"] = len(obstacles)
    report["max_other_height"] = fnum(max_other_h)
    transit_z = table_z + max(0.12, max_other_h + 0.07)
    push_z = grasp_z + 0.004
    ox = tx - gx; oy = ty - gy
    olen = math.sqrt(ox * ox + oy * oy)
    tool_angle = math.atan2(axis[1], axis[0])
    perp_angle = tool_angle + math.pi / 2.0
    ctx_plan = {"manner": manner, "hook": hook, "cube_half": cube_half, "tip_extra": tip_extra, "push_z": push_z, "transit_z": transit_z,
                "base": base, "home_q": home_q, "obstacles": obstacles, "cube_pts": cube_pts, "target": target_c}

    options = []
    for flipk in (0, 1):
        g_yaw = wrap(perp_angle + flipk * math.pi - home_closing_yaw)
        gq = quat_mul(yaw_quat(g_yaw), home_q)
        g_ok = ik_ok([gx, gy, grasp_z], gq, base) and ik_ok([gx, gy, transit_z], gq, base)
        cands = plan_candidates(ctx_plan, [cube_c[0], cube_c[1]], olen, math.atan2(oy, ox), g_yaw)
        best_c = cands[0] if len(cands) > 0 else None
        rank = 2
        if g_ok and best_c is not None:
            rank = best_c["rank"]
        options.append([rank, abs(g_yaw) + (abs(best_c["yaw"]) if best_c is not None else 0.0), g_yaw, gq, g_ok, best_c, len(cands)])
    options = sorted(options, key=option_key)
    choice = options[0]
    report["plan_options"] = [[o[0], fnum(o[2], 3), o[4], o[6], cand_summary(o[5])] for o in options]
    if choice[5] is None or choice[0] >= 2:
        report["phase"] = "no_feasible_push"
        return
    gq = choice[3]
    grasp_yaw = choice[2]

    # ---------- execute grasp ----------
    report["phase"] = "grasp"
    open_gripper()
    out = move([gx, gy, transit_z], gq, base, "pregrasp", 0.015, 400)
    if out[1]:
        finished("pregrasp"); return
    out = move([gx, gy, grasp_z + 0.04], gq, base, "grasp_mid", 0.01, 200)
    if out[1]:
        finished("grasp_mid"); return
    out = move([gx, gy, grasp_z], gq, base, "grasp", 0.008, 200)
    if out[1]:
        finished("grasp"); return
    close_gripper()
    try:
        report["after_close_gripper"] = fnum(get_robot_state().gripper_positions["primary"], 3)
    except Exception:
        pass
    out = move([gx, gy, grasp_z + 0.04], gq, base, "lift_low", 0.015, 200)
    if out[1]:
        finished("lift_low"); return
    out = move([gx, gy, transit_z], gq, base, "lift", 0.015, 250)
    if out[1]:
        finished("lift"); return

    # ---------- measure the held tool's far end (front camera, cloud connected to the gripper) ----------
    try:
        st = get_robot_state()
        report["after_lift_gripper"] = fnum(st.gripper_positions["primary"], 3)
        ee_now = [float(v) for v in st.end_effector_poses["primary"].position]
        lifted = camera_data(get_observation().cameras["front"])
        lcomps = components(lifted, convention, table_z, 2, 0.004, transit_z - table_z + 0.06, 4)
        best_far = None
        for comp in lcomps:
            touches = False
            for c in comp:
                p = c[0]
                if abs(p[2] - ee_now[2]) < 0.04 and math.sqrt((p[0] - ee_now[0]) ** 2 + (p[1] - ee_now[1]) ** 2) < 0.04:
                    touches = True
                    break
            if not touches:
                continue
            for c in comp:
                p = c[0]
                if p[2] < ee_now[2] - 0.035 or p[2] > ee_now[2] + 0.02:
                    continue
                dxy = math.sqrt((p[0] - ee_now[0]) ** 2 + (p[1] - ee_now[1]) ** 2)
                if best_far is None or dxy > best_far[0]:
                    best_far = [dxy, p]
        if best_far is not None:
            report["lifted_far_point"] = {"dist": fnum(best_far[0]), "p": [fnum(v) for v in best_far[1]],
                                          "ee": [fnum(v) for v in ee_now], "planned_olen": fnum(olen)}
            mo = [best_far[1][0] - ee_now[0], best_far[1][1] - ee_now[1]]
            mlen = math.sqrt(mo[0] ** 2 + mo[1] ** 2)
            pa = math.atan2(oy, ox) if olen > 0.01 else 0.0
            ma = math.atan2(mo[1], mo[0])
            if olen > 0.05 and abs(mlen - olen) < 0.3 * olen and abs(wrap(ma - pa)) < 0.5:
                report["offset_source"] = "measured"
                ox = mo[0]; oy = mo[1]; olen = mlen
            else:
                report["offset_source"] = "planned"
        else:
            report["offset_source"] = "planned_no_cloud"
    except Exception as error:
        note("lift_measure_error", str(error)[:200])

    # ---------- v02: pull with the measured held tool (cage planner) ----------
    if manner_video == "pull":
        rest_r = [rot2(-grasp_yaw, p[0] - gx, p[1] - gy) for p in dedupe(tool_rest, 0.004)]
        P = {"live": live, "convention": convention, "ghgt": grasp_z - table_z, "local_h": local_h,
             "target": [target_c[0], target_c[1]], "home_q": home_q, "base": base, "transit_z": transit_z,
             "push_z": push_z, "hover_z": push_z + max(0.03, cube_sq["top"] - table_z) + 0.03,
             "obstacles": [[p[0], p[1]] for p in dedupe(obstacles, 0.01)], "cube": cube_sq, "tool_rest": rest_r}
        report["pull_hover_z"] = fnum(P["hover_z"])
        outcome = pull_flow(P, rest_r, grasp_yaw)
        report["pull_outcome"] = outcome
        if outcome != "no_plan":
            if report["phase"] == "pull":
                report["phase"] = "pull_" + outcome
            return
        note("pull_fallback", "no cage plan from the rest-pose tool cloud; falling back to v01 push/pull candidates")
        ctx_plan["manner"] = "push"

    # ---------- push (closed loop: up to 2 passes) ----------
    report["phase"] = "push"
    cur_cube = [cube_c[0], cube_c[1]]
    off_angle = math.atan2(oy, ox) if olen > 0.01 else 0.0
    for attempt in range(2):
        dist = math.sqrt((target_c[0] - cur_cube[0]) ** 2 + (target_c[1] - cur_cube[1]) ** 2)
        if dist < 0.012:
            break
        cands = plan_candidates(ctx_plan, cur_cube, olen, off_angle, grasp_yaw)
        note("push_candidates", {"attempt": attempt, "n": len(cands), "top": [cand_summary(c) for c in cands[:4]]})
        if len(cands) == 0 or cands[0]["rank"] >= 2:
            report["phase"] = "no_feasible_push_" + str(attempt)
            break
        c = cands[0]
        fq = quat_mul(yaw_quat(c["yaw"]), home_q)
        sx = c["start"][0]; sy = c["start"][1]; fx = c["final"][0]; fy = c["final"][1]
        ux = c["u"][0]; uy = c["u"][1]
        out = move([sx, sy, transit_z], fq, base, "above_start", 0.015, 400)
        if out[1]:
            finished("above_start"); return
        out = move([sx, sy, push_z + 0.04], fq, base, "start_mid", 0.01, 200)
        if out[1]:
            finished("start_mid"); return
        out = move([sx, sy, push_z], fq, base, "start", 0.01, 200)
        if out[1]:
            finished("start"); return
        segments = max(2, int(math.sqrt((fx - sx) ** 2 + (fy - sy) ** 2) / 0.04) + 1)
        for k in range(1, segments + 1):
            px = sx + (fx - sx) * k / float(segments)
            py = sy + (fy - sy) * k / float(segments)
            out = move([px, py, push_z], fq, base, "push_" + str(attempt) + "_" + str(k), 0.01, 200)
            if out[1]:
                finished("push_" + str(k)); return
        out = move([fx - ux * 0.03, fy - uy * 0.03, push_z + 0.01], fq, base, "back_off", 0.015, 150)
        if out[1]:
            finished("back_off"); return
        out = move([fx - ux * 0.03, fy - uy * 0.03, transit_z], fq, base, "retreat_up", 0.02, 200)
        if out[1]:
            finished("retreat_up"); return
        grasp_yaw = c["yaw"]
        off_angle = c["dir_angle"]
        after = localize_points(("red cube", "red block"), base)
        if after is None:
            break
        ac = cloud_center(after)
        cur_cube = [ac[0], ac[1]]
        note("cube_after_push", {"attempt": attempt, "cube": [fnum(v) for v in ac],
                                 "to_target": fnum(math.sqrt((ac[0] - target_c[0]) ** 2 + (ac[1] - target_c[1]) ** 2))})
    if report["phase"] == "push":
        report["phase"] = "done"


def seg_dist(p, a, b):
    abx = b[0] - a[0]; aby = b[1] - a[1]
    den = abx * abx + aby * aby
    t = 0.0
    if den > 1e-9:
        t = ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / den
        t = max(0.0, min(1.0, t))
    qx = a[0] + t * abx; qy = a[1] + t * aby
    return math.sqrt((p[0] - qx) ** 2 + (p[1] - qy) ** 2)


def plan_candidates(cp, cube_xy, olen, off_angle, cur_yaw):
    """Tool orientations around the push direction u; contact with the tip (|a|<=60 deg) or
    the shaft side (90 deg sweep). Ranked by IK feasibility, clearance from other objects, |a|."""
    tx = cp["target"][0] - cube_xy[0]; ty = cp["target"][1] - cube_xy[1]
    dist = math.sqrt(tx * tx + ty * ty)
    if dist < 1e-4:
        return []
    ux = tx / dist; uy = ty / dist
    ua = math.atan2(uy, ux)
    pulls = []
    if cp["manner"] == "pull":
        pulls = pull_candidates(cp, cube_xy, olen, off_angle, cur_yaw, ux, uy)
        if len(pulls) > 0 and pulls[0]["rank"] < 2:
            return pulls
    specs = []
    if olen < 0.01:
        specs.append([0.0, 0.0, cp["cube_half"] + cp["tip_extra"]])
    else:
        for deg in (0, 20, -20, 40, -40, 60, -60):
            specs.append([math.radians(deg), olen, cp["cube_half"] + cp["tip_extra"]])
        for deg in (90, -90):
            for frac in (0.75, 0.55):
                specs.append([math.radians(deg), olen * frac, cp["cube_half"] + 0.008])
    out = []
    for spec in specs:
        a = spec[0]; s = spec[1]; back = spec[2]
        da = ua + a
        dx = math.cos(da); dy = math.sin(da)
        csx = cube_xy[0] - ux * (back + 0.045); csy = cube_xy[1] - uy * (back + 0.045)
        cfx = cp["target"][0] - ux * (back - 0.003); cfy = cp["target"][1] - uy * (back - 0.003)
        sx = csx - s * dx; sy = csy - s * dy
        fx = cfx - s * dx; fy = cfy - s * dy
        yaw = wrap(cur_yaw + wrap(da - off_angle)) if olen >= 0.01 else cur_yaw
        q = quat_mul(yaw_quat(yaw), cp["home_q"])
        ik = [ik_ok([sx, sy, cp["transit_z"]], q, cp["base"]), ik_ok([sx, sy, cp["push_z"] + 0.04], q, cp["base"]),
              ik_ok([sx, sy, cp["push_z"]], q, cp["base"]), ik_ok([fx, fy, cp["push_z"]], q, cp["base"])]
        clear = 1.0
        for k in range(6):
            fr = k / 5.0
            ex = sx + (fx - sx) * fr; ey = sy + (fy - sy) * fr
            tipx = ex + olen * dx; tipy = ey + olen * dy
            for p in cp["obstacles"]:
                if p[2] > cp["push_z"] + 0.15:
                    continue
                d1 = seg_dist(p, [ex, ey], [tipx, tipy]) - cp["tip_extra"]
                d2 = math.sqrt((p[0] - ex) ** 2 + (p[1] - ey) ** 2) - 0.05
                clear = min(clear, d1, d2)
        # the tool must start clear of the cube itself
        cube_clear = 1.0
        for p in cp["cube_pts"]:
            d1 = seg_dist(p, [sx, sy], [sx + olen * dx, sy + olen * dy]) - cp["tip_extra"]
            cube_clear = min(cube_clear, d1)
        feasible = ik[0] and ik[1] and ik[2] and ik[3]
        rank = 2
        if feasible and clear >= 0.025 and cube_clear >= 0.015:
            rank = 0
        elif feasible and cube_clear >= 0.0:
            rank = 1
        out.append({"rank": rank, "a": a, "s": s, "yaw": yaw, "dir_angle": da, "u": [ux, uy],
                    "start": [sx, sy], "final": [fx, fy], "ik": ik, "clear": clear, "cube_clear": cube_clear})
    out = sorted(out, key=cand_key)
    if len(pulls) > 0:
        report["pull_infeasible_fallback_push"] = True
    return out


def dense_points(cd, convention, table_z, desc):
    us = [c for c in desc["pixel"]]
    pts = desc["points"]
    umin = 999; umax = -1; vmin = 999; vmax = -1
    for p in desc["uv"]:
        umin = min(umin, p[0]); umax = max(umax, p[0]); vmin = min(vmin, p[1]); vmax = max(vmax, p[1])
    out = []
    for v in range(max(0, vmin - 3), min(cd["h"], vmax + 4)):
        for u in range(max(0, umin - 3), min(cd["w"], umax + 4)):
            p = backproject(cd, u, v, convention)
            if p is None or p[2] - table_z < 0.004 or p[2] - table_z > 0.2:
                continue
            ok = False
            for q in pts:
                if abs(q[0] - p[0]) < 0.012 and abs(q[1] - p[1]) < 0.012 and abs(q[2] - p[2]) < 0.02:
                    ok = True
                    break
            if ok:
                out.append(p)
    return out


def pull_candidates(cp, cube_xy, olen, off_angle, cur_yaw, ux, uy):
    """Hook pull: tool points away from the target (a ~ -u), the crossbar lands beyond the
    cube's far side, the shaft passes beside the cube, and the EE pulls along u."""
    hk = cp["hook"]
    side = 1.0 if hk["c"] > 0 else -1.0
    across = side * (hk["shaft_half"] + 0.5 * (abs(hk["c"]) - hk["shaft_half"]))
    inner = hk["far"] - 0.012 - cp["cube_half"]
    out = []
    ua = math.atan2(-uy, -ux)
    for deg in (0, 15, -15, 30, -30, 45, -45):
        da = ua + math.radians(deg)
        ax = math.cos(da); ay = math.sin(da)
        px = -ay; py = ax
        sx = cube_xy[0] - ax * (inner - 0.045) - px * across - px * hk["shaft_c"]
        sy = cube_xy[1] - ay * (inner - 0.045) - py * across - py * hk["shaft_c"]
        fx = cp["target"][0] - ax * (inner + 0.003) - px * across - px * hk["shaft_c"]
        fy = cp["target"][1] - ay * (inner + 0.003) - py * across - py * hk["shaft_c"]
        yaw = wrap(cur_yaw + wrap(da - off_angle))
        q = quat_mul(yaw_quat(yaw), cp["home_q"])
        ik = [ik_ok([sx, sy, cp["transit_z"]], q, cp["base"]), ik_ok([sx, sy, cp["push_z"] + 0.04], q, cp["base"]),
              ik_ok([sx, sy, cp["push_z"]], q, cp["base"]), ik_ok([fx, fy, cp["push_z"]], q, cp["base"])]
        clear = 1.0
        for k in range(6):
            fr = k / 5.0
            ex = sx + (fx - sx) * fr; ey = sy + (fy - sy) * fr
            sh0 = [ex + px * hk["shaft_c"], ey + py * hk["shaft_c"]]
            tip = [sh0[0] + ax * hk["far"], sh0[1] + ay * hk["far"]]
            bar = [tip[0] + px * hk["c"], tip[1] + py * hk["c"]]
            for p in cp["obstacles"]:
                if p[2] > cp["push_z"] + 0.15:
                    continue
                d1 = seg_dist(p, sh0, tip)
                d2 = seg_dist(p, tip, bar)
                d3 = math.sqrt((p[0] - ex) ** 2 + (p[1] - ey) ** 2) - 0.05
                clear = min(clear, d1, d2, d3)
        sh0 = [sx + px * hk["shaft_c"], sy + py * hk["shaft_c"]]
        tip = [sh0[0] + ax * hk["far"], sh0[1] + ay * hk["far"]]
        bar = [tip[0] + px * hk["c"], tip[1] + py * hk["c"]]
        cube_clear = 1.0
        for p in cp["cube_pts"]:
            cube_clear = min(cube_clear, seg_dist(p, sh0, tip) - hk["shaft_half"], seg_dist(p, tip, bar) - 0.006)
        feasible = ik[0] and ik[1] and ik[2] and ik[3]
        rank = 2
        if feasible and clear >= 0.025 and cube_clear >= 0.003:
            rank = 0
        elif feasible and cube_clear >= -0.002:
            rank = 1
        out.append({"rank": rank, "a": math.radians(deg), "s": olen, "yaw": yaw, "dir_angle": da, "u": [ux, uy],
                    "start": [sx, sy], "final": [fx, fy], "ik": ik, "clear": clear, "cube_clear": cube_clear,
                    "mode": "pull"})
    out = sorted(out, key=cand_key)
    return out


def cand_key(c):
    rmin = min(math.sqrt(c["start"][0] ** 2 + c["start"][1] ** 2), math.sqrt(c["final"][0] ** 2 + c["final"][1] ** 2))
    if c["rank"] == 0:
        return [0, max(0.0, 0.33 - rmin), abs(c["a"]), -c["clear"]]
    return [c["rank"], -c["clear"], abs(c["a"])]


def cand_summary(c):
    if c is None:
        return None
    return {"rank": c["rank"], "a_deg": int(round(math.degrees(c["a"]))), "s": fnum(c["s"], 3),
            "yaw": fnum(c["yaw"], 3), "start": [fnum(v) for v in c["start"]], "final": [fnum(v) for v in c["final"]],
            "ik": c["ik"], "clear": fnum(c["clear"], 3), "cube_clear": fnum(c["cube_clear"], 3)}


def score_key(item):
    return item[0]


def option_key(item):
    return [item[0], item[1]]


# ======================= v02: measured in-hand tool + cage/pull planner =======================
def rot2(angle, x, y):
    c = math.cos(angle); s = math.sin(angle)
    return [c * x - s * y, s * x + c * y]


def cube_square(points):
    """Footprint square of the cube from its segmented cloud: minimum-area rectangle over yaw
    (a square repeats every 90 deg). The top face covers the whole footprint in x-y."""
    best = None
    for deg in range(0, 90, 2):
        th = math.radians(deg)
        ca = math.cos(th); sa = math.sin(th)
        pa = [p[0] * ca + p[1] * sa for p in points]
        pb = [-p[0] * sa + p[1] * ca for p in points]
        a0 = quantile(pa, 0.02); a1 = quantile(pa, 0.98); b0 = quantile(pb, 0.02); b1 = quantile(pb, 0.98)
        area = (a1 - a0) * (b1 - b0)
        if best is None or area < best[0]:
            best = [area, th, a0, a1, b0, b1]
    th = best[1]
    ma = 0.5 * (best[2] + best[3]); mb = 0.5 * (best[4] + best[5])
    cx = ma * math.cos(th) - mb * math.sin(th)
    cy = ma * math.sin(th) + mb * math.cos(th)
    h = 0.25 * ((best[3] - best[2]) + (best[5] - best[4]))
    top = quantile([p[2] for p in points], 0.95)
    return {"c": [cx, cy], "yaw": th, "h": h, "top": top}


def hand_yaw(q, home_q):
    r = quat_to_matrix(q); r0 = quat_to_matrix(home_q)
    m10 = r[1][0] * r0[0][0] + r[1][1] * r0[0][1] + r[1][2] * r0[0][2]
    m00 = r[0][0] * r0[0][0] + r[0][1] * r0[0][1] + r[0][2] * r0[0][2]
    return math.atan2(m10, m00)


def dedupe(points, cell):
    seen = set()
    out = []
    for p in points:
        key = (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)))
        if key in seen:
            continue
        seen.add(key)
        out.append([p[0], p[1]])
    return out


def measure_held_tool(P, ee, yaw):
    """Front-camera points of the held tool while hovering: pixels that became closer to the
    camera than in the initial scene, inside the tool's height band, away from the fingers.
    Returned in the hand frame (rotated by -yaw about the EE)."""
    cd = camera_data(get_observation().cameras["front"])
    base_cd = P["live"]
    zlo = ee[2] - P["ghgt"] - 0.03  # allow the far end of a long tool to droop in the grasp
    zhi = ee[2] + max(P["local_h"] - P["ghgt"], 0.0) + 0.012
    pts = []
    for v in range(cd["h"]):
        row = cd["depth"][v]; brow = base_cd["depth"][v]
        for u in range(cd["w"]):
            try:
                dn = float(row[u]); db = float(brow[u])
            except Exception:
                continue
            if not (dn < db - 0.008):
                continue
            p = backproject(cd, u, v, P["convention"])
            if p is None or p[2] < zlo or p[2] > zhi:
                continue
            dx = p[0] - ee[0]; dy = p[1] - ee[1]
            dd = math.sqrt(dx * dx + dy * dy)
            if dd < 0.045 or dd > 0.55:
                continue
            pts.append(rot2(-yaw, dx, dy))
    return dedupe(pts, 0.004)


def tool_extent(tool_r):
    far = 0.0
    for r in tool_r:
        far = max(far, math.sqrt(r[0] * r[0] + r[1] * r[1]))
    return far


def plan_cage_relaxed(P, tool_r, cube, psis, cur_yaw, near=None):
    """Try the strict landing/sliding margins first, then relax them (measured walls are pixel-quantised)."""
    best = None
    for level in ((0.01, 0.003), (0.008, 0.0015), (0.006, 0.0)):
        plans = plan_cage(P, tool_r, cube, psis, cur_yaw, near, level[0], level[1])
        if plans is not None and len(plans) > 0:
            for c in plans:
                c["margins"] = [level[0], level[1]]
            if plans[0]["rank"] == 0:
                return plans
            if best is None or plans[0]["rank"] < best[0]["rank"]:
                best = plans
    return best


def plan_cage(P, tool_r, cube, psis, cur_yaw, near=None, m=0.01, m_side=0.003):
    """Search hand yaw psi and cube placement (s0, t0) relative to the EE (u = pull direction
    cube->target, n = left normal) such that the cube footprint (+margin) is free of tool points,
    moving the tool along u makes contact with the cube's trailing face within a short gap,
    the contact spans both sides of the cube centre, and preferably tool walls cage it laterally."""
    tgt = P["target"]
    tx = tgt[0] - cube["c"][0]; ty = tgt[1] - cube["c"][1]
    dist = math.sqrt(tx * tx + ty * ty)
    if dist < 1e-4:
        return None
    ux = tx / dist; uy = ty / dist
    nx = -uy; ny = ux
    phi = cube["yaw"] - math.atan2(uy, ux)
    e = cube["h"] * (abs(math.cos(phi)) + abs(math.sin(phi)))
    # m: landing margin (no tool point may come down within m of the cube footprint)
    # m_side: sliding margin past the tool's walls while the cube travels into contact
    gmax = 0.075    # longest travel before contact (lets the tool land before a short mouth)
    step = 0.0025
    found = []
    for psi in psis:
        S = []; T = []
        for r in tool_r:
            w = rot2(psi, r[0], r[1])
            S.append(w[0] * ux + w[1] * uy); T.append(w[0] * nx + w[1] * ny)
        if len(S) < 10:
            continue
        cols = {}
        for i in range(len(S)):
            j = int(math.floor(T[i] / step))
            if j not in cols:
                cols[j] = []
            cols[j].append([S[i], T[i]])
        smin = min(S); smax = max(S); tmin = min(T); tmax = max(T)
        cands = []
        nt = int((tmax - tmin + 2 * e) / step) + 1
        ns = int((smax - smin + 2 * e) / step) + 1
        for it in range(nt):
            t0 = tmin - e + it * step
            if near is not None and abs(t0 - near[1]) > 0.02:
                continue
            band = []
            for j in range(int(math.floor((t0 - e - 0.036) / step)), int(math.floor((t0 + e + 0.036) / step)) + 1):
                if j in cols:
                    band.extend(cols[j])
            inner = [b for b in band if abs(b[1] - t0) < e + m]
            front = [b for b in band if abs(b[1] - t0) <= e - 0.002]
            rub = [b for b in band if abs(b[1] - t0) > e - 0.002 and abs(b[1] - t0) < e + m_side]
            left = [b for b in band if b[1] - t0 >= e + m_side and b[1] - t0 <= e + 0.035]
            right = [b for b in band if t0 - b[1] >= e + m_side and t0 - b[1] <= e + 0.035]
            for isx in range(ns):
                s0 = smin + isx * step
                if near is not None and abs(s0 - near[0]) > 0.02:
                    continue
                if math.sqrt(s0 * s0 + t0 * t0) < e + 0.055:
                    continue
                free = True
                for b in inner:
                    if abs(b[0] - s0) < e + m:
                        free = False
                        break
                if not free:
                    continue
                face = s0 - e
                gaps = []
                for b in front:
                    if b[0] < face:
                        gaps.append([face - b[0], b[1]])
                if len(gaps) == 0:
                    continue
                gap = min(g[0] for g in gaps)
                if gap > gmax:
                    continue
                cts = [g[1] for g in gaps if g[0] <= gap + 0.006]
                if not (min(cts) < t0 - 0.006 and max(cts) > t0 + 0.006):
                    continue
                rubbed = False
                for b in rub:
                    if b[0] > face - gap + 0.004 and b[0] < s0 + e:
                        rubbed = True
                        break
                if rubbed:
                    continue
                cage = 0
                for side in (left, right):
                    for b in side:
                        if b[0] > face - gap - 0.003 and b[0] < s0 + e - gap:
                            cage += 1
                            break
                flat_bins = set()
                for g in gaps:
                    if g[0] <= gap + 0.005:
                        flat_bins.add(int(math.floor((g[1] - t0) / 0.004)))
                flat = min(1.0, len(flat_bins) * 0.004 / (2.0 * e))
                cands.append([-cage, gap + 0.08 * (1.0 - flat), s0, t0, gap, flat])
        cands = sorted(cands, key=option_key)
        kept = []
        for c in cands:
            if len(kept) >= 4:
                break
            dup = False
            for k in kept:
                if abs(k[2] - c[2]) < 0.012 and abs(k[3] - c[3]) < 0.012:
                    dup = True
            if not dup:
                kept.append(c)
        for c in kept[:2]:
            found.append([c[0], c[1] + 0.01 * abs(wrap(psi - cur_yaw)), psi, c])
    found = sorted(found, key=option_key)
    out = []
    n_ok = 0
    for f in found[:40]:
        if n_ok >= 3:
            break
        f = [f[2], f[3]]
        psi = f[0]; c = f[1]
        s0 = c[2]; t0 = c[3]; gap = c[4]
        sx = cube["c"][0] - (s0 * ux + t0 * nx); sy = cube["c"][1] - (s0 * uy + t0 * ny)
        fx = tgt[0] - ((s0 - gap) * ux + t0 * nx); fy = tgt[1] - ((s0 - gap) * uy + t0 * ny)
        q = quat_mul(yaw_quat(psi), P["home_q"])
        ik = [ik_ok([sx, sy, P["transit_z"]], q, P["base"]), ik_ok([sx, sy, P["hover_z"]], q, P["base"]),
              ik_ok([sx, sy, P["push_z"]], q, P["base"]), ik_ok([fx, fy, P["push_z"]], q, P["base"])]
        clear = 1.0
        wpts = []
        for r in dedupe(tool_r, 0.01):
            wpts.append(rot2(psi, r[0], r[1]))
        reach = tool_extent(tool_r) + 0.06
        for k in range(6):
            fr = k / 5.0
            ex = sx + (fx - sx) * fr; ey = sy + (fy - sy) * fr
            for p in P["obstacles"]:
                dh = math.sqrt((p[0] - ex) ** 2 + (p[1] - ey) ** 2)
                clear = min(clear, dh - 0.05)
                if dh > reach:
                    continue
                for w in wpts:
                    d = math.sqrt((p[0] - ex - w[0]) ** 2 + (p[1] - ey - w[1]) ** 2)
                    if d < clear:
                        clear = d
        feasible = ik[0] and ik[1] and ik[2] and ik[3]
        rank = 2
        if feasible and clear >= 0.015:
            rank = 0 if (c[0] <= -2 and c[5] >= 0.5) else 1
        if rank == 0:
            n_ok += 1
        out.append({"rank": rank, "psi": psi, "cage": -c[0], "gap": gap, "flat": c[5], "score": c[1], "s0": s0, "t0": t0, "e": e,
                    "start": [sx, sy], "final": [fx, fy], "u": [ux, uy], "ik": ik, "clear": clear,
                    "dpsi": abs(wrap(psi - cur_yaw))})
    out = sorted(out, key=cage_key)
    return out


def cage_key(c):
    return [c["rank"], c["score"] + 0.01 * c["dpsi"], -c["clear"]]


def cage_summary(c):
    if c is None:
        return None
    return {"rank": c["rank"], "psi": fnum(c["psi"], 3), "cage": c["cage"], "gap": fnum(c["gap"]), "flat": fnum(c["flat"], 2), "s0": fnum(c["s0"]),
            "t0": fnum(c["t0"]), "e": fnum(c["e"]), "start": [fnum(v) for v in c["start"]],
            "final": [fnum(v) for v in c["final"]], "ik": c["ik"], "clear": fnum(c["clear"], 3),
            "margins": c["margins"] if "margins" in c else None}


def psi_list(tool_r, cube, target, cur_yaw):
    far = tool_extent(tool_r)
    fx = 0.0; fy = 0.0
    for r in tool_r:
        if math.sqrt(r[0] * r[0] + r[1] * r[1]) > 0.8 * far:
            fx += r[0]; fy += r[1]
    ang_r = math.atan2(fy, fx)
    ua = math.atan2(target[1] - cube["c"][1], target[0] - cube["c"][0])
    out = []
    for k in range(37):
        out.append(wrap(ua + math.pi - ang_r + math.radians(-45.0 + 2.5 * k)))
    return out


def current_pose(P):
    st = get_robot_state()
    pe = st.end_effector_poses["primary"]
    return [[float(v) for v in pe.position], hand_yaw([float(v) for v in pe.quaternion_wxyz], P["home_q"])]


def pull_flow(P, tool_r, cur_yaw):
    report["phase"] = "pull"
    cube = P["cube"]
    source = "rest"
    for attempt in range(2):
        plans = plan_cage_relaxed(P, tool_r, cube, psi_list(tool_r, cube, P["target"], cur_yaw), cur_yaw)
        note("cage_plans", {"attempt": attempt, "source": source, "n": 0 if plans is None else len(plans),
                            "top": [] if plans is None else [cage_summary(c) for c in plans[:4]]})
        if plans is None or len(plans) == 0 or plans[0]["rank"] >= 2:
            return "no_plan" if attempt == 0 else "no_plan_" + str(attempt)
        c = plans[0]
        q = quat_mul(yaw_quat(c["psi"]), P["home_q"])
        out = move([c["start"][0], c["start"][1], P["transit_z"]], q, P["base"], "cage_above_" + str(attempt), 0.015, 400)
        if out[1]:
            finished("cage_above"); return "ended"
        out = move([c["start"][0], c["start"][1], P["hover_z"]], q, P["base"], "cage_hover_" + str(attempt), 0.008, 250)
        if out[1]:
            finished("cage_hover"); return "ended"
        # closed-loop placement: measure the held tool, re-plan, verify once more
        for k in range(2):
            pose = current_pose(P)
            try:
                meas = measure_held_tool(P, pose[0], pose[1])
            except Exception as error:
                note("measure_error", str(error)[:200])
                meas = []
            ext = tool_extent(meas) if len(meas) > 0 else 0.0
            note("held_tool_measure", {"attempt": attempt, "k": k, "n": len(meas), "extent": fnum(ext),
                                       "rest_extent": fnum(tool_extent(tool_r)), "ee": [fnum(v) for v in pose[0]],
                                       "yaw": fnum(pose[1], 3)})
            if len(meas) < 30 or ext < 0.5 * tool_extent(P["tool_rest"]):
                break
            tool_r = meas
            source = "measured"
            cur_yaw = pose[1]
            if k == 0:
                psis = psi_list(tool_r, cube, P["target"], cur_yaw)
                psis = [cur_yaw] + psis
                near = None
            else:
                psis = [cur_yaw]
                near = [c["s0"], c["t0"]]
            replans = plan_cage_relaxed(P, tool_r, cube, psis, cur_yaw, near)
            note("cage_replan", {"k": k, "n": 0 if replans is None else len(replans),
                                 "top": [] if replans is None else [cage_summary(x) for x in replans[:3]]})
            if replans is None or len(replans) == 0 or replans[0]["rank"] >= 2:
                if k == 0:
                    return "no_measured_plan"
                break
            c = replans[0]
            moved = math.sqrt((c["start"][0] - pose[0][0]) ** 2 + (c["start"][1] - pose[0][1]) ** 2)
            if moved < 0.003 and abs(wrap(c["psi"] - pose[1])) < 0.03:
                break
            q = quat_mul(yaw_quat(c["psi"]), P["home_q"])
            if moved > 0.05 or abs(wrap(c["psi"] - pose[1])) > 0.12:
                out = move([pose[0][0], pose[0][1], P["transit_z"]], q, P["base"], "cage_up_" + str(k), 0.015, 300)
                if out[1]:
                    finished("cage_up"); return "ended"
                out = move([c["start"][0], c["start"][1], P["transit_z"]], q, P["base"], "cage_above2_" + str(k), 0.015, 300)
                if out[1]:
                    finished("cage_above2"); return "ended"
            out = move([c["start"][0], c["start"][1], P["hover_z"]], q, P["base"], "cage_adjust_" + str(k), 0.006, 250)
            if out[1]:
                finished("cage_adjust"); return "ended"
        report["cage_used"] = cage_summary(c)
        report["tool_source"] = source
        q = quat_mul(yaw_quat(c["psi"]), P["home_q"])
        sx = c["start"][0]; sy = c["start"][1]; fx = c["final"][0]; fy = c["final"][1]
        out = move([sx, sy, P["push_z"] + 0.02], q, P["base"], "cage_low_" + str(attempt), 0.008, 200)
        if out[1]:
            finished("cage_low"); return "ended"
        out = move([sx, sy, P["push_z"]], q, P["base"], "cage_start_" + str(attempt), 0.008, 200)
        if out[1]:
            finished("cage_start"); return "ended"
        segments = max(2, int(math.sqrt((fx - sx) ** 2 + (fy - sy) ** 2) / 0.03) + 1)
        for k in range(1, segments + 1):
            px = sx + (fx - sx) * k / float(segments)
            py = sy + (fy - sy) * k / float(segments)
            out = move([px, py, P["push_z"]], q, P["base"], "pull_" + str(attempt) + "_" + str(k), 0.008, 200)
            if out[1]:
                finished("pull_" + str(k)); return "ended"
        ux = c["u"][0]; uy = c["u"][1]
        back = min(c["gap"], 0.012)
        out = move([fx - ux * back, fy - uy * back, P["push_z"]], q, P["base"], "pull_back_off", 0.01, 150)
        if out[1]:
            finished("pull_back_off"); return "ended"
        out = move([fx - ux * back, fy - uy * back, P["transit_z"]], q, P["base"], "pull_retreat_up", 0.02, 250)
        if out[1]:
            finished("pull_retreat_up"); return "ended"
        cur_yaw = c["psi"]
        after = localize_points(("red cube", "red block"), P["base"])
        if after is None:
            return "cube_lost"
        cube = cube_square(after)
        dist = math.sqrt((cube["c"][0] - P["target"][0]) ** 2 + (cube["c"][1] - P["target"][1]) ** 2)
        note("cube_after_pull", {"attempt": attempt, "cube": [fnum(v) for v in cube["c"]], "yaw": fnum(cube["yaw"], 3),
                                 "to_target": fnum(dist)})
        if dist < 0.02:
            return "done"
    return "done_" + str(attempt)


try:
    main()
except Exception as error:
    report["exception"] = str(error)[:500]
    report["exception_phase"] = report["phase"]
try:
    final_state = get_robot_state()
    report["final_time_s"] = fnum(final_state.timestamp_s, 2)
    report["final_ee"] = [fnum(v) for v in final_state.end_effector_poses["primary"].position]
    report["final_gripper"] = fnum(final_state.gripper_positions["primary"], 3)
except Exception as error:
    report["final_state_error"] = str(error)[:200]

result = report
