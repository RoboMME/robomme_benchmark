# v01_track_target_identity -- VideoPlaceOrder (robomme_plus task 12), start 2 of 3.
#
# Repair of start 1 (v00 ep0: placed on the SLOT of the 4th video placement -> task failure).
# Finding: the four identical targets are shuffled between their fixed slots many times during
# the conditioning video, so the goal "place the red cube on the fourth target it was previously
# placed on" refers to a PHYSICAL target whose slot changes.  v01:
#   1. placements = video releases (from the video's own recorded robot state) whose
#      end-effector projects onto a target slot;
#   2. slots = live SAM "target" boxes with purple pixel support (+ uncovered purple blobs);
#   3. every video frame: purple pixels in a slot ROI -> blobs + per-slot counts;
#      shuffle episodes = runs of frames with full-size purple blobs away from all slots;
#      per episode, slots whose purple count stays intact (or that hold the cube, from robot
#      state) are static; two moving slots => forced swap, more => tracked assignment;
#   4. the ordinal-th placement's target is followed through every later episode to its
#      end-of-video slot (= live slot), and the cube is placed there with v00's pick/place.
# Everything is derived at runtime from live perception or the episode's own conditioning video.

import math

report = {"version": "v01_track_target_identity", "stage": "start", "log": [], "motions": []}
result = report


def note(msg):
    report["log"].append(str(msg))


def as_list(a):
    try:
        return a.tolist()
    except Exception:
        return list(a)


def quantile(values, fraction):
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 0:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def r4(v):
    return round(float(v), 4)


def ended(res):
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


def record_motion(name, res):
    rec = {"name": name}
    try:
        rec["ok"] = bool(res.ok)
    except Exception:
        pass
    try:
        rec["status"] = str(res.status)
    except Exception:
        pass
    try:
        rec["steps"] = int(res.steps_executed)
    except Exception:
        pass
    try:
        if res.error is not None:
            rec["error"] = str(res.error)
    except Exception:
        pass
    rec["ended"] = ended(res)
    if not rec["ended"]:
        try:
            st = get_robot_state()
            p = st.end_effector_poses["primary"].position
            rec["ee"] = [r4(p[0]), r4(p[1]), r4(p[2])]
            rec["grip"] = r4(st.gripper_positions["primary"])
            rec["t"] = r4(st.timestamp_s)
        except Exception as exc:
            rec["state_error"] = str(exc)
    report["motions"].append(rec)
    if rec["ended"]:
        raise ValueError("EPISODE_ENDED after " + name)
    return rec


HEXD = "0123456789abcdef"
HEX = []
for hex_i in range(256):
    HEX.append(HEXD[hex_i // 16] + HEXD[hex_i % 16])


def dump_rgb(rgb, stride):
    rows = as_list(rgb)
    out = []
    for y in range(0, len(rows), stride):
        row = rows[y]
        parts = []
        for x in range(0, len(row), stride):
            px = row[x]
            parts.append(HEX[int(px[0])] + HEX[int(px[1])] + HEX[int(px[2])])
        out.append("".join(parts))
    return out


def dump_depth_mm(depth, stride):
    rows = as_list(depth)
    out = []
    for y in range(0, len(rows), stride):
        row = rows[y]
        vals = []
        for x in range(0, len(row), stride):
            d = float(row[x])
            if d != d or d > 60.0:
                vals.append(-1)
            else:
                vals.append(int(d * 1000.0))
        out.append(vals)
    return out


def calib(cam):
    out = {}
    try:
        out["intrinsics"] = [[r4(v) for v in row] for row in as_list(cam.intrinsics)]
    except Exception as exc:
        out["intrinsics_error"] = str(exc)
    try:
        out["position"] = [r4(v) for v in as_list(cam.camera_pose.position)]
        out["quaternion_wxyz"] = [r4(v) for v in as_list(cam.camera_pose.quaternion_wxyz)]
        out["frame"] = str(cam.camera_pose.frame)
    except Exception as exc:
        out["pose_error"] = str(exc)
    return out


def hsv(r, g, b):
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn
    if d <= 1e-6:
        h = 0.0
    elif mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * (((b - r) / d) + 2.0)
    else:
        h = 60.0 * (((r - g) / d) + 4.0)
    s = 0.0 if mx <= 1e-6 else d / mx
    return h, s, mx


COLOR_HUES = {"red": 0.0, "orange": 30.0, "yellow": 58.0, "green": 120.0, "cyan": 185.0,
              "blue": 225.0, "purple": 275.0, "violet": 275.0, "magenta": 305.0, "pink": 330.0}
ACHROMATIC = ("white", "black", "gray", "grey", "silver")


def region_colour(rows, x1, y1, x2, y2, shrink, stride=1):
    """Mean colour stats of pixels inside the inner part of a box. rows = rgb.tolist()."""
    w = x2 - x1
    h = y2 - y1
    xa = int(x1 + w * shrink)
    xb = int(x2 - w * shrink)
    ya = int(y1 + h * shrink)
    yb = int(y2 - h * shrink)
    if xb <= xa:
        xb = xa + 1
    if yb <= ya:
        yb = ya + 1
    sx = 0.0
    sy = 0.0
    n_sat = 0
    n_all = 0
    mr = 0.0
    mg = 0.0
    mb = 0.0
    vsum = 0.0
    for y in range(max(0, ya), min(len(rows), yb), stride):
        row = rows[y]
        for x in range(max(0, xa), min(len(row), xb), stride):
            px = row[x]
            r = float(px[0])
            g = float(px[1])
            b = float(px[2])
            mr += r
            mg += g
            mb += b
            n_all += 1
            hh, ss, vv = hsv(r, g, b)
            vsum += vv
            if ss > 0.35 and vv > 0.2:
                sx += math.cos(math.radians(hh))
                sy += math.sin(math.radians(hh))
                n_sat += 1
    out = {"n": n_all, "sat_frac": 0.0, "hue": None}
    if n_all > 0:
        out["mean_rgb"] = [round(mr / n_all, 1), round(mg / n_all, 1), round(mb / n_all, 1)]
        out["mean_v"] = round(vsum / n_all, 3)
        out["sat_frac"] = round(n_sat / float(n_all), 3)
    if n_sat > 0:
        hue = math.degrees(math.atan2(sy, sx)) % 360.0
        out["hue"] = round(hue, 1)
    return out


def hue_dist(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def name_hue(hue):
    best = None
    bestd = None
    for name in COLOR_HUES:
        dd = hue_dist(hue, COLOR_HUES[name])
        if bestd is None or dd < bestd:
            bestd = dd
            best = name
    return best


def cloud_stats(pc):
    pts = as_list(pc.points)
    n = len(pts)
    st = {"n": n}
    if n < 5:
        return st, pts
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    zs = [float(p[2]) for p in pts]
    ztop = quantile(zs, 0.95)
    zlow = quantile(zs, 0.05)
    top = [p for p in pts if float(p[2]) >= ztop - 0.008]
    if len(top) < 3:
        top = pts
    st["center"] = [r4(quantile(xs, 0.5)), r4(quantile(ys, 0.5)), r4(quantile(zs, 0.5))]
    st["top_xy"] = [r4(quantile([float(p[0]) for p in top], 0.5)),
                    r4(quantile([float(p[1]) for p in top], 0.5))]
    st["ztop"] = r4(ztop)
    st["zlow"] = r4(zlow)
    st["zrange"] = r4(ztop - zlow)
    st["xspan"] = r4(quantile(xs, 0.95) - quantile(xs, 0.05))
    st["yspan"] = r4(quantile(ys, 0.95) - quantile(ys, 0.05))
    return st, pts


def project_candidates(prompt, camera, base_frame, rows, max_n):
    out = []
    seg = segment_text(camera, prompt)
    if not seg.ok:
        return out
    k = 0
    for cand in seg.segmentations:
        if k >= max_n:
            break
        k += 1
        rec = {"prompt": prompt, "score": round(float(cand.score), 4)}
        try:
            box = [float(v) for v in cand.box_xyxy]
            rec["box"] = [round(v, 1) for v in box]
            rec["colour"] = region_colour(rows, box[0], box[1], box[2], box[3], 0.25)
        except Exception as exc:
            rec["box_error"] = str(exc)
        pc = None
        try:
            pc = mask_to_point_cloud(cand, camera, target_frame=base_frame)
        except Exception as exc:
            rec["pc_error1"] = str(exc)
            try:
                pc = mask_to_point_cloud(cand.mask, camera, target_frame=base_frame)
            except Exception as exc2:
                rec["pc_error2"] = str(exc2)
        if pc is not None:
            try:
                st, pts = cloud_stats(pc)
                rec["cloud"] = st
            except Exception as exc:
                rec["cloud_error"] = str(exc)
        out.append(rec)
    return out


def hold_ticks(n, grip):
    st = get_robot_state()
    last = None
    for tick in range(n):
        last = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=st.joint_positions["primary"],
            gripper_position=grip, embodiment=st.embodiment)}))
        if ended(last):
            report["motions"].append({"name": "hold", "ended": True, "tick": tick})
            raise ValueError("EPISODE_ENDED during hold")
    return last


def parse_ordinal(text):
    words = text.lower()
    for ch in ",.;:!?()'\"":
        words = words.replace(ch, " ")
    toks = words.split()
    table = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
             "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "sixth": 6, "6th": 6,
             "last": -1, "final": -1, "once": 1, "twice": 2}
    for tok in toks:
        if tok in table:
            return table[tok], tok
    return None, None


def goal_colours(text):
    words = text.lower()
    for ch in ",.;:!?()'\"":
        words = words.replace(ch, " ")
    found = []
    for tok in words.split():
        if tok in COLOR_HUES or tok in ACHROMATIC:
            found.append(tok)
    return found



# ---------------------------------------------------------------- shuffle analysis core


def purple_points(rows, x0, y0, stride, bg=40, rg=25):
    pts = []
    yy = y0
    for row in rows:
        xx = x0
        for p in row:
            g = p[1]
            if p[2] - g > bg and p[0] - g > rg:
                pts.append((xx, yy))
            xx += stride
        yy += stride
    return pts


def cluster_points(pts, stride, link_cells):
    cells = {}
    for p in pts:
        cells[(p[0] // stride, p[1] // stride)] = p
    seen = set()
    blobs = []
    for key in cells:
        if key in seen:
            continue
        seen.add(key)
        stack = [key]
        sx = 0.0
        sy = 0.0
        n = 0
        while len(stack) > 0:
            c = stack.pop()
            p = cells[c]
            sx += p[0]
            sy += p[1]
            n += 1
            for dx in range(-link_cells, link_cells + 1):
                for dy in range(-link_cells, link_cells + 1):
                    nb = (c[0] + dx, c[1] + dy)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        blobs.append([sx / n, sy / n, n])
    return blobs


def cluster_cells(pts, cell):
    # bin points into cell x cell pixel cells, then 8-connected flood fill over occupied cells
    cells = {}
    for p in pts:
        key = (p[0] // cell, p[1] // cell)
        acc = cells.get(key)
        if acc is None:
            cells[key] = [p[0], p[1], 1]
        else:
            acc[0] += p[0]
            acc[1] += p[1]
            acc[2] += 1
    seen = set()
    blobs = []
    for key in cells:
        if key in seen:
            continue
        seen.add(key)
        stack = [key]
        sx = 0.0
        sy = 0.0
        n = 0
        while len(stack) > 0:
            c = stack.pop()
            acc = cells[c]
            sx += acc[0]
            sy += acc[1]
            n += acc[2]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (c[0] + dx, c[1] + dy)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        blobs.append([sx / n, sy / n, n])
    return blobs


def slot_counts(pts, slots, radius):
    r2 = radius * radius
    out = [0] * len(slots)
    for p in pts:
        for i in range(len(slots)):
            s = slots[i]
            dx = p[0] - s[0]
            dy = p[1] - s[1]
            if dx * dx + dy * dy <= r2:
                out[i] += 1
    return out


def nearest_slot(x, y, slots):
    best = -1
    bd = None
    for i in range(len(slots)):
        d = math.hypot(x - slots[i][0], y - slots[i][1])
        if bd is None or d < bd:
            bd = d
            best = i
    return best, bd


def off_slot_blobs(blobs, slots, min_n, off_dist):
    out = []
    for b in blobs:
        if b[2] >= min_n:
            k, d = nearest_slot(b[0], b[1], slots)
            if d > off_dist:
                out.append(b)
    return out


def is_rest_between(frame_counts, ref, k0, k1, frac):
    # True if some frame in [k0, k1] has every slot count back to >= frac x its reference count
    for k in range(k0, k1 + 1):
        ok = True
        for s in range(len(frame_counts[k])):
            if frame_counts[k][s] < frac * frame_counts[ref][s]:
                ok = False
                break
        if ok:
            return True
    return False


def find_episodes(off_flags, frame_counts, merge_gap, min_frames):
    eps = []
    cur = None
    for k in range(len(off_flags)):
        if off_flags[k]:
            if cur is not None and k - cur[1] <= merge_gap and not (
                    k - cur[1] >= 3 and is_rest_between(frame_counts, max(0, cur[0] - 2), cur[1] + 1, k - 1, 0.6)):
                cur[1] = k
                cur[2] += 1
            else:
                if cur is not None:
                    eps.append(cur)
                cur = [k, k, 1]
    if cur is not None:
        eps.append(cur)
    return [e for e in eps if e[2] >= min_frames]


def median(vals):
    s = sorted(vals)
    if len(s) == 0:
        return 0.0
    return s[len(s) // 2]


def all_assignments(nt, nb):
    # every map of nt tracks to blob index or -1 (miss)
    out = [[]]
    for i in range(nt):
        nxt = []
        for a in out:
            for j in range(-1, nb):
                nxt.append(a + [j])
        out = nxt
    return out


def permutations(items):
    if len(items) <= 1:
        return [list(items)]
    out = []
    for i in range(len(items)):
        rest = items[:i] + items[i + 1:]
        for p in permutations(rest):
            out.append([items[i]] + p)
    return out


def top_by_n(blobs, k):
    rest = list(blobs)
    out = []
    while len(rest) > 0 and len(out) < k:
        bi = 0
        for i in range(1, len(rest)):
            if rest[i][2] > rest[bi][2]:
                bi = i
        out.append(rest[bi])
        rest = rest[:bi] + rest[bi + 1:]
    return out


def track_moving(tracks, blobs, single_n, params):
    gate2 = params["gate"] ** 2
    big = top_by_n([b for b in blobs if b[2] >= params["min_n"]], 5)
    nt = len(tracks)
    nb = len(big)
    pred = []
    for t in tracks:
        pred.append([t["p"][0] + t["v"][0], t["p"][1] + t["v"][1]])
    d2 = []
    for i in range(nt):
        row = []
        for b in big:
            row.append((pred[i][0] - b[0]) ** 2 + (pred[i][1] - b[1]) ** 2)
        d2.append(row)
    best_cost = None
    best = None
    for cand in all_assignments(nt, nb):
        cost = 0.0
        ok = True
        counts = [0] * nb
        for i in range(nt):
            j = cand[i]
            if j < 0:
                cost += params["miss"]
            else:
                if d2[i][j] > gate2:
                    ok = False
                    break
                cost += d2[i][j]
                counts[j] += 1
        if not ok:
            continue
        for j in range(nb):
            if counts[j] >= 2:
                if big[j][2] >= 1.25 * single_n:
                    cost += params["share"] * 0.25 * (counts[j] - 1)
                else:
                    cost += params["share"] * (counts[j] - 1)
            elif counts[j] == 0 and big[j][2] >= 0.5 * single_n:
                cost += params["uncl"]
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best = cand
    claim = {}
    for i in range(nt):
        if best[i] >= 0:
            claim[best[i]] = claim.get(best[i], 0) + 1
    for i in range(nt):
        t = tracks[i]
        j = best[i]
        if j >= 0 and claim[j] == 1:
            q = [big[j][0], big[j][1]]
            t["miss"] = 0
        elif j >= 0:
            q = pred[i]
            t["miss"] = 0
        else:
            t["miss"] = t.get("miss", 0) + 1
            if t["miss"] <= 2:
                q = pred[i]
            else:
                q = t["p"]
        vx = q[0] - t["p"][0]
        vy = q[1] - t["p"][1]
        if j < 0:
            t["v"] = [0.5 * t["v"][0], 0.5 * t["v"][1]]
        else:
            t["v"] = [0.7 * vx + 0.3 * t["v"][0], 0.7 * vy + 0.3 * t["v"][1]]
        t["p"] = [q[0], q[1]]
    return best


def residual_blobs(blobs, static_slots, slots, ref_n, min_n):
    out = []
    for b in blobs:
        if b[2] < min_n:
            continue
        keep = [b[0], b[1], b[2]]
        for s in static_slots:
            if keep is None:
                break
            sx = slots[s][0]
            sy = slots[s][1]
            d = math.hypot(keep[0] - sx, keep[1] - sy)
            rn = ref_n[s]
            if rn <= 0:
                continue
            if d <= 6.0 and keep[2] <= 1.4 * rn:
                keep = None
            elif d <= 16.0 and keep[2] > 1.4 * rn:
                nm = keep[2] - rn
                keep = [(keep[2] * keep[0] - rn * sx) / nm, (keep[2] * keep[1] - rn * sy) / nm, nm]
        if keep is not None:
            out.append(keep)
    return out


def solve_episode(ep, frame_blobs, frame_counts, base_counts, slots, single_n, params, pinned):
    nf = len(frame_blobs)
    e0 = ep[0]
    e1 = ep[1]
    r = max(0, e0 - 2)
    ref_c = frame_counts[r]
    ref_n = []
    for s in range(len(slots)):
        bn = 0
        for b in frame_blobs[r]:
            if math.hypot(b[0] - slots[s][0], b[1] - slots[s][1]) <= 8.0 and b[2] > bn:
                bn = b[2]
        ref_n.append(bn)
    lo = e0
    hi = min(nf - 1, e1 + 3)
    static = []
    moving = []
    for s in range(len(slots)):
        ok_ref = ref_c[s] >= max(3.0, 0.15 * base_counts[s]) and ref_n[s] > 0
        mn = min(frame_counts[k][s] for k in range(lo, hi + 1))
        if s in pinned or (ok_ref and mn >= 0.5 * ref_c[s]):
            static.append(s)
        else:
            moving.append(s)
    info = {"frames": [e0, e1], "ref_frame": r, "pinned": list(pinned), "ref_counts": ref_c, "static": static,
            "moving": moving}
    if len(moving) < 2:
        info["perm"] = None
        info["method"] = "no-move"
        return None, info
    tracks = []
    for s in moving:
        tracks.append({"p": [slots[s][0], slots[s][1]], "v": [0.0, 0.0], "slot": s})
    path = []
    for k in range(max(0, e0 - 1), hi + 1):
        rb = residual_blobs(frame_blobs[k], static, slots, ref_n, params["min_n"])
        track_moving(tracks, rb, single_n, params)
        path.append([k] + [[round(t["p"][0], 1), round(t["p"][1], 1)] for t in tracks])
    best = None
    bc = None
    for perm in permutations(moving):
        c = 0.0
        for i in range(len(tracks)):
            c += (tracks[i]["p"][0] - slots[perm[i]][0]) ** 2 + (tracks[i]["p"][1] - slots[perm[i]][1]) ** 2
        if bc is None or c < bc:
            bc = c
            best = perm
    tracked = {}
    for i in range(len(tracks)):
        tracked[moving[i]] = best[i]
    info["tracked_perm"] = tracked
    info["tracked_cost"] = round(bc, 1)
    info["end_positions"] = [[round(t["p"][0], 1), round(t["p"][1], 1)] for t in tracks]
    info["path_tail"] = path[-3:]
    if len(moving) == 2:
        perm = {moving[0]: moving[1], moving[1]: moving[0]}
        info["method"] = "forced-swap"
        info["tracker_agrees"] = tracked == perm
    else:
        perm = tracked
        info["method"] = "tracked"
    info["perm"] = perm
    return perm, info


def apply_perm(occ, perm):
    # occ[slot] = target id; perm maps old slot -> new slot for the moving targets
    if perm is None:
        return list(occ)
    new = list(occ)
    for s in perm:
        new[perm[s]] = occ[s]
    return new


def quat_rot(q):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def project_point(calib, p):
    # OpenCV convention: p_cam = R^T (p - C)
    rot = quat_rot(calib["quaternion_wxyz"])
    c = calib["position"]
    d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]]
    pc = [rot[0][i] * d[0] + rot[1][i] * d[1] + rot[2][i] * d[2] for i in range(3)]
    if pc[2] <= 1e-6:
        return None
    k = calib["intrinsics"]
    return [k[0][0] * pc[0] / pc[2] + k[0][2], k[1][1] * pc[1] / pc[2] + k[1][2]]


def placement_events(events, slots, calib, surface_z, max_px):
    """Releases whose end-effector projects onto a slot = placements on a target.
    Also returns cube-resting intervals [release_frame, next_grasp_frame, slot]."""
    places = []
    rests = []
    for i in range(len(events)):
        ev = events[i]
        if ev["kind"] != "release":
            continue
        uv = project_point(calib, [ev["ee"][0], ev["ee"][1], surface_z])
        if uv is None:
            continue
        s, d = nearest_slot(uv[0], uv[1], slots)
        if d > max_px:
            continue
        nxt = None
        for j in range(i + 1, len(events)):
            if events[j]["kind"] == "grasp":
                nxt = events[j]["frame"]
                break
        places.append({"frame": ev["frame"], "slot": s, "px_dist": round(d, 1), "regrasp": nxt})
        rests.append([ev["frame"], nxt, s])
    return places, rests


def pinned_for(ep, rests, nframes):
    out = []
    for r in rests:
        end = r[1]
        if end is None:
            end = nframes
        if r[0] <= ep[0] - 1 and end >= ep[1] + 1:
            out.append(r[2])
    return out


def pixel_to_plane(calib, u, v, z):
    rot = quat_rot(calib["quaternion_wxyz"])
    c = calib["position"]
    k = calib["intrinsics"]
    dc = [(u - k[0][2]) / k[0][0], (v - k[1][2]) / k[1][1], 1.0]
    d = [rot[i][0] * dc[0] + rot[i][1] * dc[1] + rot[i][2] * dc[2] for i in range(3)]
    if abs(d[2]) < 1e-9:
        return None
    t = (z - c[2]) / d[2]
    return [c[0] + t * d[0], c[1] + t * d[1], z]


def sub_rows(rgb, roi, stride):
    x0 = roi[0]
    y0 = roi[1]
    x1 = roi[2]
    y1 = roi[3]
    try:
        return rgb[y0:y1:stride, x0:x1:stride].tolist()
    except Exception:
        rows = as_list(rgb)
        out = []
        for y in range(y0, y1, stride):
            out.append(list(rows[y])[x0:x1:stride])
        return out


def frame_points(rgb, roi, stride, mode):
    if mode[0] == "np":
        try:
            a = rgb[roi[1]:roi[3]:stride, roi[0]:roi[2]:stride].astype("int32")
            g = a[:, :, 1]
            m = ((a[:, :, 2] - g) > 40) & ((a[:, :, 0] - g) > 25)
            nz = m.nonzero()
            ys = nz[0].tolist()
            xs = nz[1].tolist()
            return [(roi[0] + stride * xs[i], roi[1] + stride * ys[i]) for i in range(len(xs))]
        except Exception as exc:
            mode[0] = "py"
            mode.append("np path failed: " + str(exc)[:200])
    return purple_points(sub_rows(rgb, roi, stride), roi[0], roi[1], stride)


def follow_slot(slot, frame, ep_perms):
    cur = slot
    chain = [slot]
    for item in ep_perms:
        ep = item[0]
        perm = item[1]
        if (ep[0] + ep[1]) / 2.0 > frame and perm is not None and cur in perm:
            cur = perm[cur]
            chain.append(cur)
    return cur, chain


try:
    # ------------------------------------------------------------------ context
    context = get_task_context()
    goal = str(context.language)
    report["goal"] = goal
    report["task_name"] = str(context.task_name)
    try:
        report["task_metadata"] = dict(robomme_plus.get_task_metadata())
    except Exception as exc:
        report["task_metadata_error"] = str(exc)
    state0 = get_robot_state()
    base_frame = state0.base_frame
    home_pose = state0.end_effector_poses["primary"]
    down_q = [float(v) for v in as_list(home_pose.quaternion_wxyz)]
    report["base_frame"] = str(base_frame)
    report["home_ee"] = [r4(v) for v in as_list(home_pose.position)]
    report["start_grip"] = r4(state0.gripper_positions["primary"])
    ordinal, ordinal_word = parse_ordinal(goal)
    colours = goal_colours(goal)
    report["ordinal"] = ordinal
    report["ordinal_word"] = ordinal_word
    report["goal_colours"] = colours
    report["stage"] = "context"

    # ------------------------------------------------------------------ conditioning video: robot state
    video = robomme_plus.get_conditioning_video()
    frames = list(video)
    nf = len(frames)
    report["video_frames"] = nf
    timeline = []
    for i in range(nf):
        fr = frames[i]
        try:
            rs = fr.robot_state
            p = rs.end_effector_poses["primary"].position
            g = float(rs.gripper_positions["primary"])
            timeline.append([round(float(fr.timestamp_s), 3), r4(p[0]), r4(p[1]), r4(p[2]),
                             round(g, 3)])
        except Exception as exc:
            timeline.append([round(float(fr.timestamp_s), 3), None, None, None, None])
            if len(report["log"]) < 20:
                note("video state error frame " + str(i) + ": " + str(exc))
    report["video_timeline"] = timeline

    events = []
    grip_vals = [row[4] for row in timeline if row[4] is not None]
    if len(grip_vals) > 0:
        gmax = max(grip_vals)
        gmin = min(grip_vals)
        report["video_grip_range"] = [gmin, gmax]
        if gmax - gmin > 0.2:
            lo = gmax - 0.25
            hi = gmax - 0.08
            closed = False
            if timeline[0][4] is not None and timeline[0][4] < lo:
                closed = True
            last_open_i = 0
            last_closed_i = 0
            for i in range(nf):
                g = timeline[i][4]
                if g is None:
                    continue
                if g > hi:
                    if closed:
                        j = last_closed_i
                        events.append({"kind": "release", "frame": j, "open_frame": i,
                                       "t": timeline[j][0], "ee": timeline[j][1:4],
                                       "grip_before": timeline[j][4]})
                    closed = False
                    last_open_i = i
                elif g < lo:
                    if not closed:
                        j = last_open_i
                        events.append({"kind": "grasp", "frame": i, "open_frame": j,
                                       "t": timeline[i][0], "ee": timeline[i][1:4],
                                       "grip_after": g})
                    closed = True
                    last_closed_i = i
    report["video_events"] = events
    releases = [e for e in events if e["kind"] == "release"]
    report["n_releases"] = len(releases)
    report["stage"] = "video_events"

    # ------------------------------------------------------------------ live perception
    obs = get_observation()
    live_cal = calib(obs.cameras["front"])
    report["live_front_calib"] = live_cal
    vcal = calib(frames[0].cameras["front"])
    report["video_front_calib"] = vcal
    rows_live = as_list(obs.cameras["front"].rgb)
    img_h = len(rows_live)
    img_w = len(rows_live[0])
    probes = {}
    for prompt in ("cube", "target", "disc"):
        try:
            probes[prompt] = project_candidates(prompt, "front", base_frame, rows_live, 6)
        except Exception as exc:
            probes[prompt] = [{"error": str(exc)}]
    report["probes"] = probes
    report["stage"] = "probes"

    # flat target candidates with purple support at the box centre (live frame)
    live_pts = purple_points(rows_live, 0, 0, 1)
    live_blobs = cluster_cells(live_pts, 4)
    targets = []
    for prompt in ("target", "disc"):
        for rec in probes.get(prompt, []):
            cl = rec.get("cloud")
            if cl is None or cl.get("n", 0) < 10 or "center" not in cl or "box" not in rec:
                continue
            if cl["zrange"] > 0.03 or cl["xspan"] > 0.2 or cl["yspan"] > 0.2:
                continue
            b = rec["box"]
            uv = [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]
            support = slot_counts(live_pts, [uv], 7.0)[0]
            dup = False
            for t in targets:
                if math.hypot(t["uv"][0] - uv[0], t["uv"][1] - uv[1]) < 8.0:
                    dup = True
            if dup:
                continue
            targets.append({"xy": [cl["center"][0], cl["center"][1]], "ztop": cl["ztop"],
                            "prompt": prompt, "score": rec["score"], "uv": [round(uv[0], 1), round(uv[1], 1)],
                            "purple_support": support})
    report["live_target_candidates"] = targets
    sup = [t["purple_support"] for t in targets]
    sup_ref = max(sup) if len(sup) > 0 else 0
    slots_t = [t for t in targets if t["purple_support"] >= max(5, 0.3 * sup_ref)]
    # live purple blobs not covered by a SAM target become extra slots (ray-plane world position)
    zt = [t["ztop"] for t in slots_t]
    plane_z = median(zt) if len(zt) > 0 else 0.009
    nb_live = [b[2] for b in live_blobs if b[2] >= 20]
    ref_live = median(nb_live) if len(nb_live) > 0 else 0
    for b in live_blobs:
        if ref_live <= 0 or b[2] < 0.5 * ref_live:
            continue
        near = False
        for t in slots_t:
            if math.hypot(t["uv"][0] - b[0], t["uv"][1] - b[1]) < 10.0:
                near = True
        if not near:
            w = pixel_to_plane(live_cal, b[0], b[1], plane_z)
            if w is not None:
                slots_t.append({"xy": [r4(w[0]), r4(w[1])], "ztop": plane_z, "prompt": "purple-blob",
                                "score": 0.0, "uv": [round(b[0], 1), round(b[1], 1)],
                                "purple_support": b[2]})
    report["slots"] = slots_t
    if len(slots_t) < 2:
        raise ValueError("fewer than two target slots found")
    slots = [t["uv"] for t in slots_t]

    # cubes
    cubes = []
    for rec in probes.get("cube", []):
        cl = rec.get("cloud")
        if cl is None or cl.get("n", 0) < 10 or "center" not in cl:
            continue
        if cl["zrange"] < 0.012 or cl["xspan"] > 0.10 or cl["yspan"] > 0.10:
            continue
        cubes.append(rec)
    report["n_cube_candidates"] = len(cubes)
    chosen_cube = None
    chromatic = [c for c in colours if c in COLOR_HUES]
    if len(chromatic) > 0 and len(cubes) > 0:
        want = COLOR_HUES[chromatic[0]]
        best_d = None
        for rec in cubes:
            col = rec.get("colour", {})
            if col.get("hue") is None or col.get("sat_frac", 0.0) < 0.15:
                continue
            dd = hue_dist(col["hue"], want)
            rec["hue_dist"] = round(dd, 1)
            if best_d is None or dd < best_d:
                best_d = dd
                chosen_cube = rec
        report["cube_colour_match_dist"] = best_d
    if chosen_cube is None and len(cubes) > 0:
        chosen_cube = cubes[0]
        note("cube chosen by score (no colour match)")
    if chosen_cube is None:
        raise ValueError("no cube candidate passed the geometry bounds")
    report["chosen_cube"] = chosen_cube
    report["stage"] = "cube_chosen"

    # ------------------------------------------------------------------ video: target identity through shuffles
    margin = 40
    stride = 2
    x0 = int(max(0, min(s[0] for s in slots) - margin))
    x0 -= x0 % stride
    y0 = int(max(0, min(s[1] for s in slots) - margin))
    y0 -= y0 % stride
    x1 = int(min(img_w, max(s[0] for s in slots) + margin))
    y1 = int(min(img_h, max(s[1] for s in slots) + margin))
    roi = [x0, y0, x1, y1]
    sh = {"roi": roi, "stride": stride}
    report["shuffle"] = sh
    mode = ["np"]
    rgb0 = frames[0].cameras["front"].rgb
    p_np = frame_points(rgb0, roi, stride, mode)
    p_py = purple_points(sub_rows(rgb0, roi, stride), x0, y0, stride)
    if mode[0] == "np" and set(p_np) != set(p_py):
        mode[0] = "py"
        mode.append("np/py mismatch " + str(len(p_np)) + "/" + str(len(p_py)))
    sh["pixel_mode"] = list(mode)
    sh["frame0_points"] = len(p_py)
    report["stage"] = "shuffle_scan"
    FB = []
    FC = []
    for k in range(nf):
        pts = frame_points(frames[k].cameras["front"].rgb, roi, stride, mode)
        FB.append(cluster_cells(pts, 4))
        FC.append(slot_counts(pts, slots, 7.0))
    sh["pixel_mode_end"] = list(mode)
    n0 = [b[2] for b in FB[0] if nearest_slot(b[0], b[1], slots)[1] <= 8.0]
    single_n = median(n0) if len(n0) > 0 else 20
    sh["single_n"] = single_n
    OFF = []
    for k in range(nf):
        OFF.append(len(off_slot_blobs(FB[k], slots, 0.6 * single_n, 8.0)) > 0)
    rest_frames = [k for k in range(nf) if not OFF[k]]
    base = []
    for s in range(len(slots)):
        base.append(median([FC[k][s] for k in rest_frames]))
    sh["base_counts"] = base
    eps = find_episodes(OFF, FC, 6, 3)
    places, rests = placement_events(events, slots, vcal, plane_z, 12.0)
    sh["placements"] = places
    params = {"gate": 30.0, "miss": 500.0, "share": 400.0, "uncl": 300.0, "min_n": 2}
    ep_perms = []
    infos = []
    for e in eps:
        perm, info = solve_episode(e, FB, FC, base, slots, single_n, params, pinned_for(e, rests, nf))
        ep_perms.append([e, perm])
        if perm is not None:
            info["perm"] = [[a, perm[a]] for a in perm]
        if "tracked_perm" in info:
            info["tracked_perm"] = [[a, info["tracked_perm"][a]] for a in info["tracked_perm"]]
        infos.append(info)
    sh["episodes"] = infos
    last_blobs = [[round(b[0], 1), round(b[1], 1), b[2]] for b in FB[nf - 1] if b[2] >= 0.4 * single_n]
    sh["last_frame_blobs"] = last_blobs
    sh["last_frame_counts"] = FC[nf - 1]
    report["stage"] = "shuffle_solved"

    # ------------------------------------------------------------------ choose the target
    target = None
    if len(places) > 0 and ordinal is not None:
        if ordinal == -1:
            idx = len(places) - 1
        else:
            idx = ordinal - 1
        ends = []
        for pl in places:
            end, chain = follow_slot(pl["slot"], pl["frame"], ep_perms)
            pl["end_slot"] = end
            pl["chain"] = chain
            ends.append(end)
        sh["placement_end_slots"] = ends
        sh["placement_targets_distinct"] = len(set(ends)) == len(ends)
        if 0 <= idx < len(places):
            report["placement_index_used"] = idx
            target = slots_t[places[idx]["end_slot"]]
            report["target_source"] = "tracked identity of placement " + str(idx + 1)
        else:
            note("ordinal " + str(ordinal) + " exceeds placement count " + str(len(places)))
    if target is None and len(releases) > 0:
        idx = len(releases) - 1 if (ordinal is None or ordinal == -1) else min(ordinal - 1, len(releases) - 1)
        ev = releases[idx]
        best_d = None
        for t in slots_t:
            dd = math.hypot(t["xy"][0] - ev["ee"][0], t["xy"][1] - ev["ee"][1])
            if best_d is None or dd < best_d:
                best_d = dd
                target = t
        report["target_source"] = "fallback: live slot nearest video release " + str(idx + 1)
    if target is None:
        raise ValueError("no target could be chosen")
    target_xy = target["xy"]
    target_top = target["ztop"]
    report["target"] = target
    report["stage"] = "target_chosen"

    # ------------------------------------------------------------------ pick (v00, worked on ep 0)
    cl = chosen_cube["cloud"]
    cube_xy = cl["top_xy"]
    cube_top = cl["ztop"]
    cube_low = cl["zlow"]
    grasp_z = max(cube_top - 0.02, cube_low + 0.01)
    report["grasp"] = [cube_xy[0], cube_xy[1], r4(grasp_z)]
    start_grip = float(state0.gripper_positions["primary"])
    if start_grip < 0.95:
        record_motion("open_initial", open_gripper())
    pre = Pose([cube_xy[0], cube_xy[1], grasp_z + 0.12], down_q, base_frame)
    record_motion("pre_grasp", move_to_pose(pre, tolerance=0.015, max_steps=200))
    gp = Pose([cube_xy[0], cube_xy[1], grasp_z], down_q, base_frame)
    record_motion("descend_grasp", move_to_pose(gp, tolerance=0.01, max_steps=150))
    record_motion("close", close_gripper())
    hold_ticks(3, 0.0)
    lift_z = grasp_z + 0.15
    record_motion("lift", move_to_pose(Pose([cube_xy[0], cube_xy[1], lift_z], down_q, base_frame),
                                       tolerance=0.02, max_steps=150))
    held = float(get_robot_state().gripper_positions["primary"])
    report["held_width"] = r4(held)
    report["stage"] = "lifted"

    # ------------------------------------------------------------------ place
    release_z = target_top + (grasp_z - cube_low) + 0.012
    report["release_z"] = r4(release_z)
    above_z = max(lift_z, release_z + 0.12)
    record_motion("above_target", move_to_pose(Pose([target_xy[0], target_xy[1], above_z], down_q,
                                                    base_frame), tolerance=0.015, max_steps=200))
    record_motion("descend_place", move_to_pose(Pose([target_xy[0], target_xy[1], release_z], down_q,
                                                     base_frame), tolerance=0.01, max_steps=150))
    record_motion("open_release", open_gripper())
    hold_ticks(10, 1.0)
    report["stage"] = "released"
    record_motion("retreat", move_to_pose(Pose([target_xy[0], target_xy[1], release_z + 0.10], down_q,
                                               base_frame), tolerance=0.02, max_steps=120))
    hold_ticks(10, 1.0)
    report["stage"] = "done"
except Exception as exc:
    msg = str(exc)
    if msg.startswith("EPISODE_ENDED"):
        report["episode_ended"] = msg
    else:
        report["error"] = msg

result = report
