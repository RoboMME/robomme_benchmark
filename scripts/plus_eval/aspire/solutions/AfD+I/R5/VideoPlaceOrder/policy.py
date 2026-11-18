import math

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "1st": 1, "2nd": 2, "3rd": 3,
            "4th": 4, "5th": 5, "6th": 6, "7th": 7, "8th": 8, "9th": 9, "10th": 10}


COLORS = ("red", "green", "blue", "yellow")


STRIDE = 2


def parse_goal(text):
    words = text.lower().replace(",", " ").replace(".", " ").split()
    color = None
    order = None
    for idx, word in enumerate(words):
        if color is None and word in COLORS and idx + 1 < len(words) and words[idx + 1] == "cube":
            color = word
        if order is None and word in ORDINALS:
            order = ORDINALS[word]
    return color, order


def is_purple_v00(p):
    r = p[0]
    g = p[1]
    b = p[2]
    return r > 80 and b > 100 and b - g > 30


def is_purple(p):
    """Lighting-tolerant target-ring colour (v01): magenta/purple under white, yellow or coloured light."""
    r = p[0]
    g = p[1]
    b = p[2]
    return r > 80 and b > 70 and b - g > 12


def is_cube(p, color):
    r = p[0]
    g = p[1]
    b = p[2]
    if color == "red":
        return r > 100 and g < 45 and b < 45 and r > 3 * g
    if color == "green":
        return g > 100 and r < 80 and b < 80
    if color == "blue":
        return b > 90 and r < 60 and g < 60
    if color == "yellow":
        return r > 150 and g > 150 and b < 70
    return False


def components(cells):
    """cells: set of (gx, gy) grid cells. 8-connected components -> list of (cx, cy, n) in pixels."""
    seen = set()
    out = []
    for cell in cells:
        if cell in seen:
            continue
        stack = [cell]
        seen.add(cell)
        sx = 0
        sy = 0
        n = 0
        while stack:
            cx, cy = stack.pop()
            sx += cx
            sy += cy
            n += 1
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cx + dx, cy + dy)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        out.append((STRIDE * sx / n, STRIDE * sy / n, n))
    return out


def analyse_rows(rows, color, y0, y1):
    purple = set()
    cx = 0.0
    cy = 0.0
    cn = 0
    for y in range(y0, y1, STRIDE):
        row = rows[y]
        gy = y // STRIDE
        for x in range(0, len(row), STRIDE):
            p = row[x]
            if is_purple(p):
                purple.add((x // STRIDE, gy))
            elif is_cube(p, color):
                cx += x
                cy += y
                cn += 1
    blobs = [b for b in components(purple) if b[2] >= 6]
    cube = None
    if cn >= 3:
        cube = (cx / cn, cy / cn, cn)
    return blobs, cube, purple


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def nearest_spot(p, spots):
    best = None
    bestd = None
    for si in range(len(spots)):
        d = dist(p, spots[si])
        if bestd is None or d < bestd:
            bestd = d
            best = si
    return best, bestd




def spot_counts(cells, spots, radius_cells):
    """Number of purple grid cells within radius (grid units) of each spot."""
    out = []
    r2 = radius_cells * radius_cells
    for s in spots:
        gx = s[0] / STRIDE
        gy = s[1] / STRIDE
        n = 0
        for c in cells:
            dx = c[0] - gx
            dy = c[1] - gy
            if dx * dx + dy * dy <= r2:
                n += 1
        out.append(n)
    return out


def vacancy_runs(flags, times, gap_s):
    """flags[i] True = vacant. Returns runs [start_i, end_i(first occupied after), ts, te] with short occupied gaps filled."""
    runs = []
    i = 0
    n = len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for r in runs:
        if merged and r[0] < n and times[r[0]] - times[merged[-1][1] - 1] <= gap_s + 1e-6 and r[0] - merged[-1][1] <= 2:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    out = []
    for r in merged:
        te = times[r[1]] if r[1] < n else times[-1]
        out.append([r[0], r[1], times[r[0]], te])
    return out


def sym_votes(per, spots, a, b, i0, i1, at_spot_px, mid_px):
    """Count samples in [i0, i1) where two off-spot blobs are point-symmetric about the a-b midpoint."""
    mx = (spots[a][0] + spots[b][0]) / 2.0
    my = (spots[a][1] + spots[b][1]) / 2.0
    sep = dist(spots[a], spots[b])
    votes = 0
    for i in range(i0, i1):
        moving = []
        for bl in per[i]:
            si, d = nearest_spot(bl, spots)
            if d > at_spot_px:
                moving.append(bl)
        hit = False
        for p in range(len(moving)):
            for q in range(p + 1, len(moving)):
                u = moving[p]
                v = moving[q]
                if dist(((u[0] + v[0]) / 2.0, (u[1] + v[1]) / 2.0), (mx, my)) <= mid_px and dist(u, v) >= 0.4 * sep:
                    hit = True
        if hit:
            votes += 1
    return votes


def solve_vac(per, cubes, order, times, cells, radius_cells=4.0, occ_frac=0.3, gap_s=0.3, start_tol_s=0.3,
              end_tol_s=1.0, min_swap_s=0.6, max_swap_s=3.5, at_spot_px=5.0, mid_px=7.0, rest_px=1.5,
              near_px=9.0, rest_s=0.75, min_votes=1, tight_s=0.3):
    report = {}
    spots = select_spots(per)
    nsp = len(spots)
    report["spots"] = [(round(s[0], 1), round(s[1], 1)) for s in spots]
    counts = [spot_counts(cells[i], spots, radius_cells) for i in range(len(per))]
    base = []
    for s in range(nsp):
        vals = sorted(counts[i][s] for i in range(len(per)))
        base.append(vals[int(0.8 * (len(vals) - 1))])
    report["base_counts"] = base
    runs = []
    for s in range(nsp):
        flags = [counts[i][s] < max(2.0, occ_frac * base[s]) for i in range(len(per))]
        for r in vacancy_runs(flags, times, gap_s):
            dur = r[3] - r[2]
            if min_swap_s * 0.5 <= dur <= max_swap_s + 1.0:
                runs.append([s] + r)
    # candidate pairings
    cands = []
    for x in range(len(runs)):
        for y in range(x + 1, len(runs)):
            ra = runs[x]
            rb = runs[y]
            if ra[0] == rb[0]:
                continue
            ds = abs(ra[3] - rb[3])
            de = abs(ra[4] - rb[4])
            if min(ds, de) > start_tol_s or max(ds, de) > end_tol_s:
                continue
            span = max(ra[4], rb[4]) - min(ra[3], rb[3])
            if span < min_swap_s or span > max_swap_s:
                continue
            a = min(ra[0], rb[0])
            b = max(ra[0], rb[0])
            v = sym_votes(per, spots, a, b, min(ra[1], rb[1]), max(ra[2], rb[2]), at_spot_px, mid_px)
            if v < min_votes and ds + de > tight_s:
                continue
            cands.append([-v, ds + de, -v, x, y, a, b])
    cands.sort()
    used = set()
    swaps = []
    for c in cands:
        if c[3] in used or c[4] in used:
            continue
        used.add(c[3])
        used.add(c[4])
        ra = runs[c[3]]
        rb = runs[c[4]]
        swaps.append([min(ra[1], rb[1]), max(ra[2], rb[2]), -c[2], [c[5], c[6]], round(c[1], 2)])
    swaps.sort()
    report["swaps"] = swaps
    report["unpaired_runs"] = [[r[0], r[1], r[2]] for k, r in enumerate(runs) if k not in used]
    # replay identities; cube rest events
    labels = list(range(nsp))
    events = []
    cube0 = None
    for c in cubes[:5]:
        if c is not None:
            cube0 = c
            break
    still_start = None
    si_ptr = 0
    pending = []
    for i in range(len(per)):
        while si_ptr < len(swaps) and swaps[si_ptr][1] <= i:
            a, b = swaps[si_ptr][3]
            la = labels[a]
            labels[a] = labels[b]
            labels[b] = la
            si_ptr += 1
        busy = set()
        for sw in swaps:
            if sw[0] <= i < sw[1]:
                busy.add(sw[3][0])
                busy.add(sw[3][1])
        if i > 0:
            a0 = cubes[i - 1]
            c = cubes[i]
            if a0 is not None and c is not None and dist(a0, c) <= rest_px:
                if still_start is None:
                    still_start = i - 1
            else:
                still_start = None
            if still_start is not None and times[i] - times[still_start] >= rest_s:
                if cube0 is None or dist(c, cube0) >= near_px:
                    si, d = nearest_spot((c[0], c[1] + 3.0), spots)
                    if si is not None and d <= near_px and si not in busy:
                        tid = labels[si]
                        if events and events[-1][1] == tid:
                            events[-1][2] = i
                        else:
                            events.append([i, tid, i, si, round(c[0], 1), round(c[1], 1), round(d, 1)])
    report["events"] = events
    report["labels_end"] = labels
    if order is None or order < 1 or order > len(events):
        report["chosen"] = None
        return report
    ev = events[order - 1]
    report["chosen"] = ev
    final_spot = labels.index(ev[1])
    report["final_spot"] = final_spot
    report["final_px"] = spots[final_spot]
    return report


def select_spots(per, near=5.0, min_frac=0.1):
    """Rest spots: blobs of the first/last frame that persist in >= min_frac of all frames."""
    cands = [(b[0], b[1]) for b in per[0]] + [(b[0], b[1]) for b in per[-1]]
    spots = []
    for c in cands:
        dup = False
        for s in spots:
            if dist(c, s) < 2.0 * near:
                dup = True
        if dup:
            continue
        hits = 0
        for blobs in per:
            for b in blobs:
                if dist(b, c) <= near:
                    hits += 1
                    break
        if hits >= min_frac * len(per):
            spots.append(c)
    return spots


ENSEMBLE = [{}, {"radius_cells": 3.0}, {"radius_cells": 5.0}, {"occ_frac": 0.2}, {"occ_frac": 0.4},
            {"start_tol_s": 0.2}, {"start_tol_s": 0.45}, {"mid_px": 5.0}, {"mid_px": 9.0}, {"gap_s": 0.15}]


def solve_ensemble(per, cubes, order, times, cells):
    """Run solve_vac under perturbed parameters; majority vote on the final rest spot."""
    votes = {}
    first = None
    runs = []
    for cfg in ENSEMBLE:
        r = solve_vac(per, cubes, order, times, cells, **cfg)
        if first is None:
            first = r
        fs = r.get("final_spot")
        runs.append(fs)
        if fs is not None:
            votes[fs] = votes.get(fs, 0) + 1
    best = None
    for k in votes:
        if best is None or votes[k] > votes[best]:
            best = k
    first["ensemble_final_spots"] = runs
    if best is not None:
        first["final_spot"] = best
        first["final_px"] = select_spots(per)[best]
        first["ensemble_support"] = votes[best]
    return first


# ---------------- runtime program (VideoPlaceOrder) ----------------
AT_SPOT_PX = 5.0
MID_PX = 6.0
Y0 = 24
Y1 = 200

report = {"version": "v01"}


def to_rows(rgb):
    try:
        return rgb.tolist()
    except Exception:
        rows = []
        for y in range(len(rgb)):
            row = []
            for x in range(len(rgb[y])):
                p = rgb[y][x]
                row.append((int(p[0]), int(p[1]), int(p[2])))
            rows.append(row)
        return rows


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def cloud_of_pixel(camera, u, v):
    seg = segment_points(camera, [[int(round(u)), int(round(v))]])
    if not seg.ok or len(seg.segmentations) == 0:
        return None
    pc = mask_to_point_cloud(seg.segmentations[0], camera, target_frame=base_frame)
    pts = list(pc.points)
    if len(pts) < 5:
        return None
    return pts


def span_xy(pts):
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return max(quantile(xs, 0.95) - quantile(xs, 0.05), quantile(ys, 0.95) - quantile(ys, 0.05))


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def rotate_vec(q, v):
    w, x, y, z = q
    qv = (0.0, v[0], v[1], v[2])
    qc = (w, -x, -y, -z)
    r = quat_mul(quat_mul(q, qv), qc)
    return (r[1], r[2], r[3])


def edge_angle(pts):
    """Orientation (rad, in [0, pi/2)) of a square footprint: minimum-area bounding box search."""
    best = None
    best_area = None
    for k in range(18):
        th = k * (math.pi / 2.0) / 18.0
        c = math.cos(th)
        s = math.sin(th)
        us = [c * float(p[0]) + s * float(p[1]) for p in pts]
        vs = [-s * float(p[0]) + c * float(p[1]) for p in pts]
        area = (quantile(us, 0.95) - quantile(us, 0.05)) * (quantile(vs, 0.95) - quantile(vs, 0.05))
        if best_area is None or area < best_area:
            best_area = area
            best = th
    return best


def hold_tick():
    st = get_robot_state()
    sr = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=st.joint_positions["primary"],
        gripper_position=st.gripper_positions["primary"], embodiment=st.embodiment)}))
    return sr


def project_pixel(cam, u, v, gl):
    """Pinhole back-projection of pixel (u, v) using public depth, intrinsics and camera pose."""
    d = float(cam.depth_m[int(round(v))][int(round(u))])
    k = cam.intrinsics
    fx = float(k[0][0])
    fy = float(k[1][1])
    ox = float(k[0][2])
    oy = float(k[1][2])
    xc = (u - ox) * d / fx
    yc = (v - oy) * d / fy
    if gl:
        pc = (xc, -yc, -d)
    else:
        pc = (xc, yc, d)
    q = tuple(float(a) for a in cam.camera_pose.quaternion_wxyz)
    w = rotate_vec(q, pc)
    t = cam.camera_pose.position
    return (w[0] + float(t[0]), w[1] + float(t[1]), w[2] + float(t[2]))


def finished(res):
    try:
        if res.terminated or res.truncated:
            return True
    except Exception:
        pass
    return False


def move(pos, quat, label, max_steps=150, tol=0.01):
    res = move_to_pose(Pose((pos[0], pos[1], pos[2]), quat, base_frame), tolerance=tol, max_steps=max_steps)
    entry = {"label": label, "ok": bool(res.ok), "target": [round(pos[0], 4), round(pos[1], 4), round(pos[2], 4)]}
    try:
        entry["status"] = str(res.status)
        entry["steps"] = int(res.steps_executed)
    except Exception:
        pass
    report.setdefault("motions", []).append(entry)
    return res


state0 = get_robot_state()
base_frame = state0.base_frame
goal = get_task_context().language
report["goal"] = goal
try:
    meta = robomme_plus.get_task_metadata()
    report["max_steps"] = meta["max_steps"]
    report["sampling"] = meta["conditioning_video_sampling"]
except Exception as exc:
    report["meta_error"] = str(exc)
color, order = parse_goal(goal)
report["color"] = color
report["order"] = order

# ---------- 1. read the conditioning video ----------
video = robomme_plus.get_conditioning_video()
report["video_frames"] = len(video)
per = []
cubes = []
times = []
cells = []
vstate = []
for vi in range(len(video)):
    frame = video[vi]
    if vi % 48 == 47:
        hold_tick()
        report["hold_ticks"] = report.get("hold_ticks", 0) + 1
    rows = to_rows(frame.cameras["front"].rgb)
    blobs, cube, cellset = analyse_rows(rows, color, Y0, Y1)
    per.append(blobs)
    cells.append(cellset)
    cubes.append(cube)
    times.append(float(frame.timestamp_s))
    if vi % 8 == 0:
        try:
            rs = frame.robot_state
            ee = rs.end_effector_poses["primary"].position
            vstate.append([vi, round(float(frame.timestamp_s), 2), round(float(ee[0]), 3), round(float(ee[1]), 3),
                           round(float(ee[2]), 3), round(float(rs.gripper_positions["primary"]), 3)])
        except Exception:
            pass
report["video_state_every8"] = vstate
if len(times) > 0:
    report["video_t"] = [times[0], times[-1]]

solution = None
if len(per) > 0:
    solution = solve_ensemble(per, cubes, order, times, cells)
    report["ensemble_final_spots"] = solution.get("ensemble_final_spots")
    report["ensemble_support"] = solution.get("ensemble_support")
    report["base_counts"] = solution.get("base_counts")
    report["unpaired_runs"] = solution.get("unpaired_runs")
    report["spots"] = solution.get("spots")
    report["swaps"] = solution.get("swaps")
    report["events"] = solution.get("events")
    report["chosen"] = solution.get("chosen")
    report["final_px"] = solution.get("final_px")

# ---------- 2. live scene ----------
obs = get_observation()
live_rows = to_rows(obs.cameras["front"].rgb)
live_blobs, live_cube, live_cells = analyse_rows(live_rows, color, Y0, Y1)
report["live_blobs"] = [[round(b[0], 1), round(b[1], 1), b[2]] for b in live_blobs]
report["live_cube_px"] = live_cube and [round(live_cube[0], 1), round(live_cube[1], 1), live_cube[2]]

target_px = None
if solution is not None and solution.get("final_px") is not None:
    fp = solution["final_px"]
    target_px = (fp[0], fp[1])
    near, d = nearest_spot(target_px, [(b[0], b[1]) for b in live_blobs])
    if near is not None and d <= 8.0:
        target_px = (live_blobs[near][0], live_blobs[near][1])
    report["target_live_match_px"] = d
report["target_px"] = target_px

if target_px is None or live_cube is None:
    report["abort"] = "no target or cube"
else:
    # ---------- 3. 3D geometry from public segmentation + depth ----------
    tpts = cloud_of_pixel("front", target_px[0], target_px[1])
    if tpts is None or span_xy(tpts) > 0.15:
        report["target_cloud_issue"] = "point prompt failed or too large"
    cpts = cloud_of_pixel("front", live_cube[0], live_cube[1])
    if cpts is None or span_xy(cpts) > 0.12:
        loc = localize_object(color + " cube", camera_name="front", target_frame=base_frame)
        if loc.ok and len(loc.point_cloud.points) >= 5:
            cpts = list(loc.point_cloud.points)
            report["cube_cloud_source"] = "localize_object"
    cam = obs.cameras["front"]
    proj = None
    if cpts is not None:
        try:
            cmx = quantile([p[0] for p in cpts], 0.5)
            cmy = quantile([p[1] for p in cpts], 0.5)
            cmz = quantile([p[2] for p in cpts], 0.5)
            errs = []
            for gl in (False, True):
                pp = project_pixel(cam, live_cube[0], live_cube[1], gl)
                errs.append(((pp[0] - cmx) ** 2 + (pp[1] - cmy) ** 2 + (pp[2] - cmz) ** 2) ** 0.5)
            report["projection_err_cv_gl"] = [round(e, 4) for e in errs]
            gl = errs[1] < errs[0]
            if min(errs) < 0.04:
                proj = project_pixel(cam, target_px[0], target_px[1], gl)
                report["target_proj"] = [round(a, 4) for a in proj]
        except Exception as exc:
            report["projection_error"] = str(exc)
    if (tpts is None or span_xy(tpts) > 0.15) and proj is not None:
        tpts = [proj for k in range(5)]
        report["target_source"] = "depth_projection"
    if tpts is None or cpts is None:
        report["abort"] = "missing cloud"
    else:
        tx = quantile([p[0] for p in tpts], 0.5)
        ty = quantile([p[1] for p in tpts], 0.5)
        table_z = quantile([p[2] for p in tpts], 0.5)
        cx = quantile([p[0] for p in cpts], 0.5)
        cy = quantile([p[1] for p in cpts], 0.5)
        top_z = quantile([p[2] for p in cpts], 0.9)
        height = top_z - table_z
        report["target_xyz"] = [round(tx, 4), round(ty, 4), round(table_z, 4)]
        report["cube_xyz"] = [round(cx, 4), round(cy, 4), round(top_z, 4)]
        report["cube_height"] = round(height, 4)
        top_pts = [p for p in cpts if float(p[2]) >= top_z - 0.4 * height]
        if len(top_pts) < 8:
            top_pts = cpts
        # recentre on the top face
        cx = quantile([p[0] for p in top_pts], 0.5)
        cy = quantile([p[1] for p in top_pts], 0.5)
        theta = edge_angle(top_pts)
        q_home = tuple(float(v) for v in state0.end_effector_poses["primary"].quaternion_wxyz)
        finger_axis = rotate_vec(q_home, (0.0, 1.0, 0.0))
        psi = math.atan2(finger_axis[1], finger_axis[0])
        delta = theta - psi
        while delta > math.pi / 4.0:
            delta -= math.pi / 2.0
        while delta < -math.pi / 4.0:
            delta += math.pi / 2.0
        qz = (math.cos(delta / 2.0), 0.0, 0.0, math.sin(delta / 2.0))
        q_grasp = quat_mul(qz, q_home)
        report["yaw"] = {"cube_edge_deg": round(math.degrees(theta), 1), "finger_deg": round(math.degrees(psi), 1),
                         "delta_deg": round(math.degrees(delta), 1)}
        grasp_z = table_z + 0.5 * height
        hover = top_z + 0.10
        done = False
        open_gripper()
        r = move((cx, cy, hover), q_grasp, "pre_grasp", max_steps=200)
        done = finished(r)
        if not done:
            r = move((cx, cy, grasp_z), q_grasp, "grasp", max_steps=120, tol=0.008)
            done = finished(r)
        if not done:
            g = close_gripper()
            done = finished(g)
            try:
                report["gripper_after_close"] = round(float(get_robot_state().gripper_positions["primary"]), 3)
            except Exception:
                pass
        if not done:
            r = move((cx, cy, hover), q_grasp, "lift", max_steps=150)
            done = finished(r)
        if not done:
            r = move((tx, ty, hover), q_grasp, "above_target", max_steps=200)
            done = finished(r)
        if not done:
            r = move((tx, ty, table_z + 0.5 * height + 0.04), q_grasp, "pre_place", max_steps=120)
            done = finished(r)
        if not done:
            r = move((tx, ty, table_z + 0.5 * height + 0.008), q_grasp, "place", max_steps=100, tol=0.008)
            done = finished(r)
        if not done:
            g = open_gripper()
            done = finished(g)
        if not done:
            st = get_robot_state()
            for tick in range(10):
                sr = step(RobotAction(arms={"primary": ArmCommand(
                    mode="joint_position", target=st.joint_positions["primary"],
                    gripper_position=1.0, embodiment=st.embodiment)}))
                try:
                    if sr.terminated or sr.truncated:
                        done = True
                        break
                except Exception:
                    pass
        if not done:
            r = move((tx, ty, hover), q_grasp, "retreat", max_steps=120)
            done = finished(r)
        report["ended_by_terminal"] = done

result = report
