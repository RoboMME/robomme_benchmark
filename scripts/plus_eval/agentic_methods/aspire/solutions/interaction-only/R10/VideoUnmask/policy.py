import math

# VideoUnmask v01 -- conditioning-video colour memory in 3D from BOTH cameras (wrist first),
# every instance of each target colour (plural goals: "all containers hiding blue cubes"),
# live container recovered with a point prompt on the wrist view at the re-projected memory.
# All task-dependent quantities come from the goal text, the video and live RGB-D at runtime.

report = {"version": "v01", "stages": []}

COLOR_WORDS = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "cyan",
               "magenta", "white", "black", "brown", "gray", "grey"]
CONTAINER_PROMPTS = ["container", "bin", "box", "cup", "bowl"]


def note(msg):
    report["stages"].append(str(msg))


def flag(res, name):
    try:
        return bool(res.terminated) if name == "terminated" else bool(res.truncated)
    except Exception:
        return False


def ended(res):
    return flag(res, "terminated") or flag(res, "truncated")


def exec_summary(res):
    out = {}
    try:
        out["ok"] = bool(res.ok)
    except Exception:
        out["ok"] = None
    try:
        out["status"] = str(res.status)
    except Exception:
        pass
    try:
        out["steps"] = int(res.steps_executed)
    except Exception:
        pass
    out["terminated"] = flag(res, "terminated")
    out["truncated"] = flag(res, "truncated")
    try:
        if res.error is not None:
            out["error"] = str(res.error)[:300]
    except Exception:
        pass
    return out


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def classify(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 40:
        return "black"
    sat = (mx - mn) / float(mx)
    if sat < 0.35:
        if mx > 200:
            return "white"
        return "gray"
    d = float(mx - mn)
    if mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * (((b - r) / d) + 2.0)
    else:
        h = 60.0 * (((r - g) / d) + 4.0)
    if mx < 70:
        return "dark"
    if h < 15 or h >= 345:
        return "red"
    if h < 40:
        return "orange"
    if h < 70:
        return "yellow"
    if h < 165:
        return "green"
    if h < 195:
        return "cyan"
    if h < 255:
        return "blue"
    if h < 290:
        return "purple"
    return "magenta"


def color_alias(word):
    if word == "pink":
        return "magenta"
    if word == "grey":
        return "gray"
    return word


def to_list(arr):
    try:
        return arr.tolist()
    except Exception:
        out = []
        for row in arr:
            out.append([[int(c) for c in px] for px in row])
        return out


def color_stats(rgb_list, stride):
    """Per colour class: count, sum u, sum v, pixel list (subsampled)."""
    stats = {}
    h = len(rgb_list)
    w = len(rgb_list[0])
    for v in range(0, h, stride):
        row = rgb_list[v]
        for u in range(0, w, stride):
            px = row[u]
            c = classify(int(px[0]), int(px[1]), int(px[2]))
            if c in ("gray", "white", "black", "dark"):
                continue
            s = stats.get(c)
            if s is None:
                s = {"n": 0, "su": 0.0, "sv": 0.0, "px": []}
                stats[c] = s
            s["n"] += 1
            s["su"] += u
            s["sv"] += v
            s["px"].append((u, v))
    return stats


def blobs(pixels, stride):
    """Connected components on the subsampled grid (4-neighbour)."""
    remaining = set(pixels)
    comps = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            cu, cv = stack.pop()
            for du, dv in ((stride, 0), (-stride, 0), (0, stride), (0, -stride)):
                nb = (cu + du, cv + dv)
                if nb in remaining:
                    remaining.discard(nb)
                    stack.append(nb)
                    comp.append(nb)
        su = sum(p[0] for p in comp)
        sv = sum(p[1] for p in comp)
        us = [p[0] for p in comp]
        vs = [p[1] for p in comp]
        comps.append({"n": len(comp), "u": su / len(comp), "v": sv / len(comp),
                      "box": [min(us), min(vs), max(us), max(vs)]})
    comps.sort(key=lambda c: -c["n"])
    return comps


def quat_rotate(q, vec):
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    vx, vy, vz = vec
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx))


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def project(u, v, d, intr, pose, convention):
    fx = float(intr[0][0])
    fy = float(intr[1][1])
    cx = float(intr[0][2])
    cy = float(intr[1][2])
    a = (u - cx) * d / fx
    b = (v - cy) * d / fy
    if convention == "opencv":
        cam = (a, b, d)
    elif convention == "opengl":
        cam = (a, -b, -d)
    else:  # ros / x-forward
        cam = (d, -a, -b)
    r = quat_rotate(pose.quaternion_wxyz, cam)
    p = pose.position
    return (r[0] + float(p[0]), r[1] + float(p[1]), r[2] + float(p[2]))


def depth_at(depth, u, v, radius):
    vals = []
    h = len(depth)
    w = len(depth[0])
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            uu = int(u) + du
            vv = int(v) + dv
            if 0 <= uu < w and 0 <= vv < h:
                d = float(depth[vv][uu])
                if d > 0.0 and d < 10.0:
                    vals.append(d)
    return quantile(vals, 0.5)


def pose_summary(pose):
    return {"p": [round(float(x), 4) for x in pose.position],
            "q": [round(float(x), 4) for x in pose.quaternion_wxyz]}


def thumb(rgb_list, stride):
    out = []
    for v in range(0, len(rgb_list), stride):
        row = rgb_list[v]
        out.append([[int(row[u][0]), int(row[u][1]), int(row[u][2])]
                    for u in range(0, len(row), stride)])
    return out




def unproject(xyz, intr, pose):
    """robot_base point -> (u, v, depth) in an OpenCV-convention camera (convention verified
    against mask_to_point_cloud in v00: opencv error 5 mm vs >0.5 m for the others)."""
    q = pose.quaternion_wxyz
    qc = (float(q[0]), -float(q[1]), -float(q[2]), -float(q[3]))
    p = pose.position
    rel = (xyz[0] - float(p[0]), xyz[1] - float(p[1]), xyz[2] - float(p[2]))
    c = quat_rotate(qc, rel)
    if c[2] <= 1e-6:
        return None
    u = float(intr[0][0]) * c[0] / c[2] + float(intr[0][2])
    v = float(intr[1][1]) * c[1] / c[2] + float(intr[1][2])
    return (u, v, c[2])


def dist2(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# ---------------------------------------------------------------- 1. context
context = get_task_context()
goal = str(context.language)
report["goal"] = goal
try:
    report["task_metadata"] = dict(robomme_plus.get_task_metadata())
except Exception as exc:
    report["task_metadata_error"] = str(exc)[:200]

words = goal.lower().replace(",", " ").replace(".", " ").split()
targets = []   # list of (colour, plural)
for wi in range(len(words)):
    wd = words[wi]
    if wd in COLOR_WORDS:
        nxt = words[wi + 1] if wi + 1 < len(words) else ""
        prev = words[max(0, wi - 4):wi]
        plural = nxt.endswith("s") or ("all" in prev)
        targets.append((color_alias(wd), plural))
report["targets"] = targets
note("targets %s" % targets)

state0 = get_robot_state()
base_frame = state0.base_frame
home_q = tuple(float(x) for x in state0.end_effector_poses["primary"].quaternion_wxyz)
report["home_ee"] = pose_summary(state0.end_effector_poses["primary"])

# ---------------------------------------------------------------- 2. video memory
video = robomme_plus.get_conditioning_video()
nvid = len(video)
report["video_len"] = nvid
target_set = set(t[0] for t in targets)
obs_by_cam = {"wrist": [], "front": []}    # (colour, frame, n, u, v, xyz)
vinfo = {}
if nvid > 0:
    vinfo["t1"] = float(video[nvid - 1].timestamp_s)
    try:
        vinfo["ee_first"] = pose_summary(video[0].robot_state.end_effector_poses["primary"])
        vinfo["ee_last"] = pose_summary(video[nvid - 1].robot_state.end_effector_poses["primary"])
    except Exception:
        pass
    nsample = min(nvid, 64)
    idxs = sorted(set(int(round(i * (nvid - 1) / float(max(1, nsample - 1)))) for i in range(nsample)))
    thumb_idx = set(idxs[int(k * (len(idxs) - 1) / 7.0)] for k in range(8))
    thumbs = {}
    for fi in idxs:
        fr = video[fi]
        for cam in ("wrist", "front"):
            co = fr.cameras[cam]
            rgbl = to_list(co.rgb)
            if cam == "wrist" and fi in thumb_idx:
                thumbs[str(fi)] = thumb(rgbl, 4)
            st = color_stats(rgbl, 2)
            dep = None
            for c in target_set:
                s = st.get(c)
                if s is None or s["n"] < 4:
                    continue
                for bl in blobs(s["px"], 2):
                    if bl["n"] < 4:
                        continue
                    if dep is None:
                        dep = to_list(co.depth_m)
                    d = depth_at(dep, bl["u"], bl["v"], 1)
                    if d is None:
                        continue
                    xyz = project(bl["u"], bl["v"], d, co.intrinsics, co.camera_pose, "opencv")
                    obs_by_cam[cam].append((c, fi, bl["n"], bl["u"], bl["v"], xyz))
    report["video_thumbs_wrist_stride4"] = thumbs
report["video_info"] = vinfo


def cluster(rows):
    """Group blob sightings of one colour into distinct cube instances by 3D xy distance."""
    inst = []
    for row in sorted(rows, key=lambda r: -r[2]):
        placed = False
        for it in inst:
            if dist2(it["xyz"], row[5]) < 0.03:
                it["hits"] += 1
                it["frames"].append(row[1])
                placed = True
                break
        if not placed:
            inst.append({"xyz": row[5], "n": row[2], "uv": [round(row[3], 1), round(row[4], 1)],
                         "frame": row[1], "hits": 1, "frames": [row[1]]})
    return inst


memory = {}
mem_log = {}
for c in target_set:
    per = {}
    for cam in ("wrist", "front"):
        rows = [r for r in obs_by_cam[cam] if r[0] == c]
        inst = cluster(rows)
        if inst:
            maxn = max(it["n"] for it in inst)
            # a real cube is seen in several sampled frames and is not a speck
            inst = [it for it in inst if it["n"] >= max(6, 0.3 * maxn) and it["hits"] >= 2]
        per[cam] = inst
    chosen = per["wrist"]
    src = "wrist"
    # add front-only instances not already seen by the wrist camera
    for it in per["front"]:
        if all(dist2(it["xyz"], w["xyz"]) > 0.04 for w in chosen):
            chosen = chosen + [dict(it, src="front")]
    if not per["wrist"]:
        src = "front"
    memory[c] = chosen
    mem_log[c] = {"src": src, "wrist": [[round(x, 4) for x in it["xyz"]] + [it["n"], it["hits"], it["uv"]] for it in per["wrist"]],
                  "front": [[round(x, 4) for x in it["xyz"]] + [it["n"], it["hits"], it["uv"]] for it in per["front"]]}
report["memory"] = mem_log
note("video scanned: %d frames" % nvid)

# ---------------------------------------------------------------- 3. live containers
obs = get_observation()
wr = obs.cameras["wrist"]
report["live_wrist_cam"] = pose_summary(wr.camera_pose)
report["thumb_live_wrist"] = thumb(to_list(wr.rgb), 2)
live_wdepth = to_list(wr.depth_m)

# text candidates on the wrist view (fallback only; the 5-mask cap makes them non-exhaustive)
text_cands = []
for prompt in ("box", "container"):
    seg = segment_text("wrist", prompt)
    if seg.ok:
        for cand in seg.segmentations[:10]:
            text_cands.append(cand)


def cloud_geom(pts):
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    zs = [float(p[2]) for p in pts]
    return {"med": [quantile(xs, 0.5), quantile(ys, 0.5), quantile(zs, 0.5)],
            "span": max(quantile(xs, 0.97) - quantile(xs, 0.03), quantile(ys, 0.97) - quantile(ys, 0.03)),
            "top": quantile(zs, 0.95), "low": quantile(zs, 0.03)}


def plausible(g):
    # container-sized: raised above the table, compact footprint (bounded at both ends)
    return g["top"] > 0.02 and 0.015 < g["span"] < 0.10


def container_for(xyz):
    out = {"mem": [round(x, 4) for x in xyz]}
    pix = unproject(xyz, wr.intrinsics, wr.camera_pose)
    best = None
    if pix is not None:
        out["uv"] = [round(pix[0], 1), round(pix[1], 1)]
        # the cube sits under its container, so the overhead wrist pixel of the remembered
        # cube position lies on that container's top face
        seg = segment_points("wrist", [[int(pix[0]), int(pix[1])]])
        if seg.ok and len(seg.segmentations) > 0:
            for cand in seg.segmentations[:3]:
                pc = mask_to_point_cloud(cand, "wrist", target_frame=base_frame)
                if len(pc.points) < 20:
                    continue
                g = cloud_geom(pc.points)
                ok = plausible(g) and dist2(g["med"], xyz) < 0.04
                out.setdefault("point_cands", []).append(
                    [round(g["med"][0], 4), round(g["med"][1], 4), round(g["top"], 4), round(g["span"], 4), ok])
                if ok:
                    best = (pc.points, g, "point")
                    break
    if best is None:
        bd = None
        for cand in text_cands:
            pc = mask_to_point_cloud(cand, "wrist", target_frame=base_frame)
            if len(pc.points) < 20:
                continue
            g = cloud_geom(pc.points)
            dd = dist2(g["med"], xyz)
            if plausible(g) and dd < 0.04 and (bd is None or dd < bd):
                bd = dd
                best = (pc.points, g, "text")
    if best is not None:
        out["how"] = best[2]
    return best, out


plan = []
plan_log = []
for (c, plural) in targets:
    inst = memory.get(c, [])
    if not inst:
        plan_log.append({"color": c, "error": "no video memory"})
        continue
    use = inst if plural else inst[:1]
    for it in use:
        found, info = container_for(it["xyz"])
        info["color"] = c
        plan_log.append(info)
        if found is None:
            # last resort: grasp straight above the remembered cube using live depth there
            plan.append({"color": c, "pts": None, "xyz": it["xyz"]})
        else:
            plan.append({"color": c, "pts": found[0], "xyz": it["xyz"]})
report["plan"] = plan_log

# ---------------------------------------------------------------- 4. grasp + lift
def min_area_yaw(points):
    best = None
    for k in range(0, 90, 5):
        a = math.radians(k)
        ca = math.cos(a)
        sa = math.sin(a)
        us = [p[0] * ca + p[1] * sa for p in points]
        vs = [-p[0] * sa + p[1] * ca for p in points]
        w1 = quantile(us, 0.97) - quantile(us, 0.03)
        w2 = quantile(vs, 0.97) - quantile(vs, 0.03)
        if best is None or w1 * w2 < best[0]:
            best = (w1 * w2, a, w1, w2)
    return best


def grasp_orientation(axis_yaw):
    cyv = quat_rotate(home_q, (0.0, 1.0, 0.0))
    cur = math.atan2(cyv[1], cyv[0])
    delta = axis_yaw - cur
    while delta > math.pi / 2:
        delta -= math.pi
    while delta < -math.pi / 2:
        delta += math.pi
    return quat_mul((math.cos(delta / 2.0), 0.0, 0.0, math.sin(delta / 2.0)), home_q), delta


episode_over = False
lifts = []
for idx in range(len(plan)):
    item = plan[idx]
    if episode_over:
        break
    info = {"color": item["color"]}
    lifts.append(info)
    pts = item["pts"]
    if pts is not None:
        top_z = quantile([float(p[2]) for p in pts], 0.95)
        top_pts = [(float(p[0]), float(p[1])) for p in pts if float(p[2]) > top_z - 0.012]
        if len(top_pts) < 10:
            top_pts = [(float(p[0]), float(p[1])) for p in pts]
        cx = quantile([p[0] for p in top_pts], 0.5)
        cy = quantile([p[1] for p in top_pts], 0.5)
        mar = min_area_yaw(top_pts)
        axis = mar[1] if mar[2] <= mar[3] else mar[1] + math.pi / 2
        info["dims"] = [round(mar[2], 4), round(mar[3], 4)]
    else:
        # no container mask: use remembered cube xy and the live wrist depth above it
        cx = item["xyz"][0]
        cy = item["xyz"][1]
        pix = unproject(item["xyz"], wr.intrinsics, wr.camera_pose)
        top_z = item["xyz"][2] + 0.04
        if pix is not None:
            d = depth_at(live_wdepth, pix[0], pix[1], 2)
            if d is not None:
                top_z = project(pix[0], pix[1], d, wr.intrinsics, wr.camera_pose, "opencv")[2]
        axis = 0.0
        info["fallback"] = True
    q, delta = grasp_orientation(axis)
    grasp_z = top_z - 0.02
    info["center"] = [round(cx, 4), round(cy, 4)]
    info["top_z"] = round(top_z, 4)
    info["yaw_delta_deg"] = round(math.degrees(delta), 1)
    note("lift %s at %s" % (item["color"], info["center"]))
    r = open_gripper()
    if ended(r):
        episode_over = True
        break
    r = move_to_pose(Pose((cx, cy, top_z + 0.12), q, base_frame), tolerance=0.02, max_steps=250)
    info["pre"] = exec_summary(r)
    if ended(r):
        episode_over = True
        break
    r = move_to_pose(Pose((cx, cy, grasp_z), q, base_frame), tolerance=0.012, max_steps=200)
    info["descend"] = exec_summary(r)
    if ended(r):
        episode_over = True
        break
    r = close_gripper()
    info["close"] = exec_summary(r)
    try:
        info["grip_after_close"] = round(float(get_robot_state().gripper_positions["primary"]), 4)
    except Exception:
        pass
    if ended(r):
        episode_over = True
        break
    r = move_to_pose(Pose((cx, cy, grasp_z + 0.15), q, base_frame), tolerance=0.02, max_steps=250)
    info["lift"] = exec_summary(r)
    if ended(r):
        episode_over = True
        break
    st = get_robot_state()
    for tick in range(10):
        sr = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=st.joint_positions["primary"],
            gripper_position=0.0, embodiment=st.embodiment)}))
        if ended(sr):
            episode_over = True
            break
    if episode_over:
        break
    if idx + 1 < len(plan):
        r = move_to_pose(Pose((cx, cy, grasp_z + 0.005), q, base_frame), tolerance=0.015, max_steps=200)
        info["lower"] = exec_summary(r)
        if ended(r):
            episode_over = True
            break
        r = open_gripper()
        if ended(r):
            episode_over = True
            break
        r = move_to_pose(Pose((cx, cy, top_z + 0.12), q, base_frame), tolerance=0.02, max_steps=200)
        if ended(r):
            episode_over = True
            break

report["lifts"] = lifts
report["episode_over"] = episode_over
result = report
