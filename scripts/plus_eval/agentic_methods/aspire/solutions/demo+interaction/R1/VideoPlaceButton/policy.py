"""VideoPlaceButton v01_terminal_guard — robomme_plus/task_11 (UNTESTED in simulator; NOT selected).

v01 = v00 + terminal-status guard: no tool call after a motion reports terminated/truncated,
and every post-motion state read is guarded so `result` survives episode end.


Strategy (from offline analysis of the released recordings, see task_analysis.md):
the conditioning video shows identical target rings that keep swapping places pairwise
(each swap = ~1.3 s half-turn of two rings about their midpoint), the robot moving the goal
cube between rings, and one button press. The answer is the ring on which the goal cube
was first placed after the press, followed through every later swap to its slot in the
current scene. The program perceives everything at runtime (goal text, live front image,
conditioning-video frames), then picks the goal cube and places it on that ring.
"""
import math

# ---- VideoPlaceButton video-analysis core (pure Python, sandbox-safe: no imports
# ---- besides math, no underscores, no getattr/isinstance). Shared by v00 and the
# ---- offline replay harness (tools/replay_full.py). Image coordinates = front camera px.

def vpb_rows(rgb, stride):
    """Sub-sampled nested list of [r,g,b] rows; falls back to element access."""
    try:
        return rgb[::stride, ::stride].tolist()
    except Exception:
        pass
    try:
        full = rgb.tolist()
    except Exception:
        full = [[list(px) for px in row] for row in rgb]
    out = []
    for yy in range(0, len(full), stride):
        out.append(full[yy][::stride])
    return out

def vpb_is_target(r, g, b):
    # target rings are pink/purple/white; wood has b < g, cubes/arm/button fail one test
    return b > g + 12 and r > 100

def vpb_is_color(r, g, b, color):
    if color == "red":
        return r > 120 and g < 80 and b < 80 and r > 2 * g + 20
    if color == "blue":
        return b > 100 and r < 80 and g < 100 and b > 2 * r + 20
    if color == "green":
        return g > 100 and r < 100 and b < 100 and g > r + 40
    if color == "yellow":
        return r > 150 and g > 150 and b < 90
    return False

def vpb_frame_features(rows, stride, color, y_min):
    """One pass: target-ring connected components + goal-cube centroid."""
    h = len(rows)
    cells = {}
    cx = 0.0
    cy = 0.0
    cn = 0
    for gy in range(h):
        y = gy * stride
        if y < y_min:
            continue
        row = rows[gy]
        for gx in range(len(row)):
            px = row[gx]
            r = px[0]
            g = px[1]
            b = px[2]
            if b > g + 12 and r > 100:
                cells[(gx, gy)] = 1
            elif vpb_is_color(r, g, b, color):
                cx += gx * stride
                cy += y
                cn += 1
    comps = []
    seen = {}
    for key in cells:
        if key in seen:
            continue
        stack = [key]
        seen[key] = 1
        sx = 0.0
        sy = 0.0
        n = 0
        while stack:
            cur = stack.pop()
            sx += cur[0]
            sy += cur[1]
            n += 1
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cur[0] + dx, cur[1] + dy)
                    if nb in cells and nb not in seen:
                        seen[nb] = 1
                        stack.append(nb)
        comps.append([sx / n * stride, sy / n * stride, n])
    cube = None
    if cn > 0:
        cube = [cx / cn, cy / cn, cn]
    return {"comps": comps, "cube": cube}

def vpb_cube_rests(feats, times, min_cells, still_px, min_dur_s):
    """Segments where the goal cube centroid is still for >= min_dur_s."""
    rests = []
    start = None
    last = None
    for fi in range(len(feats)):
        cube = feats[fi]["cube"]
        ok = cube is not None and cube[2] >= min_cells
        if ok and last is not None and math.sqrt((cube[0] - last[0]) ** 2 + (cube[1] - last[1]) ** 2) <= still_px:
            if start is None:
                start = fi - 1
        else:
            if start is not None:
                rests.append([start, fi - 1])
            start = None
        last = cube if ok else None
    if start is not None:
        rests.append([start, len(feats) - 1])
    out = []
    for seg in rests:
        if times[seg[1]] - times[seg[0]] >= min_dur_s:
            xs = sorted(feats[k]["cube"][0] for k in range(seg[0], seg[1] + 1))
            ys = sorted(feats[k]["cube"][1] for k in range(seg[0], seg[1] + 1))
            out.append({"start": seg[0], "end": seg[1], "t0": times[seg[0]], "t1": times[seg[1]],
                        "x": xs[len(xs) // 2], "y": ys[len(ys) // 2]})
    # merge consecutive rests at the same place (arm occlusion splits a rest)
    merged = []
    for seg in out:
        if merged:
            prev = merged[-1]
            dist = math.sqrt((prev["x"] - seg["x"]) ** 2 + (prev["y"] - seg["y"]) ** 2)
            moved = False
            for k in range(prev["end"] + 1, seg["start"]):
                cube = feats[k]["cube"]
                if cube is not None and cube[2] >= min_cells:
                    if math.sqrt((cube[0] - prev["x"]) ** 2 + (cube[1] - prev["y"]) ** 2) > 12.0:
                        moved = True
            if dist <= 8.0 and not moved:
                prev["end"] = seg["end"]
                prev["t1"] = seg["t1"]
                continue
        merged.append(seg)
    return merged

def vpb_key_neg_n(e):
    return -e["n"]

def vpb_key_tc(e):
    return e["tc"]

def vpb_key_neg_score(e):
    return -e["score"]

def vpb_slots(comps, min_cells):
    return [[c[0], c[1], c[2]] for c in comps if c[2] >= min_cells]

def vpb_swap_events(feats, times, slots, occ_px, mid_frac, min_frames, gap_s, swap_s, min_span_s):
    """Detect pairwise slot swaps. Per frame, every pair of off-slot rings is attributed to the
    ONE slot pair whose midpoint it matches best (rings rotate about that midpoint); rings are
    used at most once per frame; a slot pair needs at least one vacant slot."""
    ns = len(slots)
    pairs_ab = []
    for a in range(ns):
        for b in range(a + 1, ns):
            mx = 0.5 * (slots[a][0] + slots[b][0])
            my = 0.5 * (slots[a][1] + slots[b][1])
            sep = math.sqrt((slots[a][0] - slots[b][0]) ** 2 + (slots[a][1] - slots[b][1]) ** 2)
            pairs_ab.append([a, b, mx, my, sep])
    evid = {}
    for fi in range(len(feats)):
        off = []
        occupied = {}
        for c in feats[fi]["comps"]:
            if c[2] < 3:
                continue
            dmin = None
            kmin = -1
            for k in range(ns):
                s = slots[k]
                d = math.sqrt((c[0] - s[0]) ** 2 + (c[1] - s[1]) ** 2)
                if dmin is None or d < dmin:
                    dmin = d
                    kmin = k
            if dmin is not None and dmin > occ_px:
                off.append(c)
            elif kmin >= 0 and c[2] >= 6:
                occupied[kmin] = 1
        if len(off) < 2:
            continue
        cands = []
        for i in range(len(off)):
            for j in range(i + 1, len(off)):
                p = off[i]
                q = off[j]
                pq = math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
                hx = 0.5 * (p[0] + q[0])
                hy = 0.5 * (p[1] + q[1])
                best = None
                for item in pairs_ab:
                    a = item[0]
                    b = item[1]
                    sep = item[4]
                    if a in occupied and b in occupied:
                        continue
                    if pq < 0.45 * sep or pq > 1.6 * sep:
                        continue
                    dm = math.sqrt((hx - item[2]) ** 2 + (hy - item[3]) ** 2)
                    tol = max(5.0, mid_frac * sep)
                    if dm > tol:
                        continue
                    err = dm / tol
                    if best is None or err < best[0]:
                        best = [err, a, b]
                if best is not None:
                    cands.append([best[0], i, j, best[1], best[2]])
        cands.sort()
        used = {}
        for cnd in cands:
            if cnd[1] in used or cnd[2] in used:
                continue
            used[cnd[1]] = 1
            used[cnd[2]] = 1
            key = (cnd[3], cnd[4])
            if key not in evid:
                evid[key] = []
            evid[key].append(fi)
    # group evidence frames per slot pair into runs; one swap lasts about swap_s seconds
    events = []
    for key in evid:
        frames = evid[key]
        run = [frames[0]]
        for fi in frames[1:]:
            if times[fi] - times[run[-1]] <= gap_s or times[fi] - times[run[0]] <= swap_s:
                run.append(fi)
            else:
                events.append([key, run])
                run = [fi]
        events.append([key, run])
    out = []
    for item in events:
        key = item[0]
        run = item[1]
        span = times[run[-1]] - times[run[0]]
        if len(run) >= min_frames and span >= min_span_s:
            out.append({"a": key[0], "b": key[1], "f0": run[0], "f1": run[-1], "n": len(run),
                        "t0": times[run[0]], "t1": times[run[-1]],
                        "tc": 0.5 * (times[run[0]] + times[run[-1]])})
    out.sort(key=vpb_key_neg_n)
    kept = []
    for e in out:
        clash = False
        for k in kept:
            share = e["a"] in (k["a"], k["b"]) or e["b"] in (k["a"], k["b"])
            overlap = not (e["t1"] < k["t0"] - gap_s or e["t0"] > k["t1"] + gap_s)
            if share and overlap:
                clash = True
        if not clash:
            kept.append(e)
    kept.sort(key=vpb_key_tc)
    return kept

def vpb_follow_token(slot, events, t_from):
    """Apply, in time order, every swap whose centre is after t_from to a token at slot."""
    trail = [slot]
    for e in events:
        if e["tc"] <= t_from:
            continue
        if slot == e["a"]:
            slot = e["b"]
            trail.append(slot)
        elif slot == e["b"]:
            slot = e["a"]
            trail.append(slot)
    return slot, trail

def vpb_find_button(rows, stride, slots, y_min):
    """Button = largest compact near-white blob (min channel > 200) away from every target ring."""
    cells = {}
    for gy in range(len(rows)):
        if gy * stride < y_min:
            continue
        row = rows[gy]
        for gx in range(len(row)):
            px = row[gx]
            if min(px[0], px[1], px[2]) > 200:
                cells[(gx, gy)] = 1
    best = None
    seen = {}
    for key in cells:
        if key in seen:
            continue
        stack = [key]
        seen[key] = 1
        pts = []
        while stack:
            cur = stack.pop()
            pts.append(cur)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cur[0] + dx, cur[1] + dy)
                    if nb in cells and nb not in seen:
                        seen[nb] = 1
                        stack.append(nb)
        n = len(pts)
        x = sum([p[0] for p in pts]) * stride / n
        y = sum([p[1] for p in pts]) * stride / n
        w = (max([p[0] for p in pts]) - min([p[0] for p in pts]) + 1) * stride
        hh = (max([p[1] for p in pts]) - min([p[1] for p in pts]) + 1) * stride
        near = min([math.sqrt((x - s[0]) ** 2 + (y - s[1]) ** 2) for s in slots] + [999.0])
        if near < 10.0 or w > 40 or hh > 40:
            continue
        if best is None or n > best[2]:
            best = [x, y, n, w, hh]
    return best

def vpb_patch(rows, stride, bx, by, half):
    """Flattened RGB patch of grid cells within +-half px of (bx, by)."""
    out = []
    g0x = int((bx - half) / stride)
    g1x = int((bx + half) / stride)
    g0y = int((by - half) / stride)
    g1y = int((by + half) / stride)
    for gy in range(max(0, g0y), min(len(rows), g1y + 1)):
        row = rows[gy]
        for gx in range(max(0, g0x), min(len(row), g1x + 1)):
            out.append(row[gx])
    return out

def vpb_patch_change(patch, ref, thr):
    """Fraction of patch cells whose summed |RGB difference| to the reference exceeds thr."""
    n = min(len(patch), len(ref))
    if n == 0:
        return 0.0
    c = 0
    for i in range(n):
        p = patch[i]
        q = ref[i]
        if abs(p[0] - q[0]) + abs(p[1] - q[1]) + abs(p[2] - q[2]) > thr:
            c += 1
    return c / n

def vpb_runs(flags, times, min_s):
    """Contiguous True runs lasting at least min_s seconds -> list of [t0, t1, i0, i1]."""
    runs = []
    s = None
    for k in range(len(flags)):
        if flags[k] and s is None:
            s = k
        if (not flags[k]) and s is not None:
            runs.append([times[s], times[k - 1], s, k - 1])
            s = None
    if s is not None:
        runs.append([times[s], times[len(flags) - 1], s, len(flags) - 1])
    return [r for r in runs if r[1] - r[0] >= min_s]

def vpb_press_from_change(change, times, frac, min_s, merge_s):
    """Image press detector: runs where the button patch differs from the reference; merged; strongest first."""
    runs = vpb_runs([c >= frac for c in change], times, 0.0)
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= merge_s:
            merged[-1][1] = r[1]
            merged[-1][3] = r[3]
        else:
            merged.append([r[0], r[1], r[2], r[3]])
    out = []
    for r in merged:
        if r[1] - r[0] >= min_s:
            peak = max(change[r[2]:r[3] + 1])
            out.append({"t0": r[0], "t1": r[1], "peak": peak, "score": peak * (r[1] - r[0] + 0.05)})
    out.sort(key=vpb_key_neg_score)
    return out

def vpb_answer(slots, events, rests, press_t1, max_slot_px, y_off):
    """First cube rest starting after the press that lies on a slot; follow that ring through later swaps."""
    for r in rests:
        if r["t0"] < press_t1:
            continue
        ds = [math.sqrt((s[0] - r["x"]) ** 2 + (s[1] - r["y"] - y_off) ** 2) for s in slots]
        k = 0
        for i in range(len(ds)):
            if ds[i] < ds[k]:
                k = i
        if ds[k] > max_slot_px:
            continue
        fin, trail = vpb_follow_token(k, events, r["t0"])
        return {"rest": r, "place_slot": k, "place_dist": ds[k], "final_slot": fin, "trail": trail}
    return None

# =====================================================================================
# v00 main program. Everything task-dependent is perceived at runtime:
#   goal colour  <- get_task_context().language
#   slots/button <- live front RGB (identical scene to the last conditioning frame)
#   swaps/press/placement <- robomme_plus.get_conditioning_video() front RGB (+ robot state)
#   3D targets   <- mask_to_point_cloud on live front depth
# Numeric constants below are generic pixel/second/metre tolerances, not scene answers.
# =====================================================================================

report = {"version": "v01_terminal_guard", "phase": "start", "errors": [], "phases": {}}
result = report


def note_error(where, exc):
    report["errors"].append(where + ": " + str(exc)[:400])


def r3(v):
    try:
        return [round(float(v[0]), 4), round(float(v[1]), 4), round(float(v[2]), 4)]
    except Exception:
        return None


def exec_summary(res):
    out = {}
    try:
        out["ok"] = bool(res.ok)
    except Exception:
        out["ok"] = None
    for name in ("status", "steps_executed", "terminated", "truncated", "error"):
        try:
            if name == "status":
                out[name] = str(res.status)
            elif name == "steps_executed":
                out[name] = res.steps_executed
            elif name == "terminated":
                out[name] = bool(res.terminated)
            elif name == "truncated":
                out[name] = bool(res.truncated)
            else:
                out[name] = str(res.error)[:300] if res.error is not None else None
        except Exception:
            pass
    return out


def is_terminal(summary):
    return bool(summary.get("terminated")) or bool(summary.get("truncated"))


def state_snapshot():
    out = {"t": None, "ee": None, "grip": None}
    try:
        st = get_robot_state()
    except Exception as exc:
        out["error"] = str(exc)[:200]
        return out
    try:
        out["t"] = float(st.timestamp_s)
    except Exception:
        pass
    try:
        out["ee"] = r3(st.end_effector_poses["primary"].position)
    except Exception:
        pass
    try:
        out["grip"] = round(float(st.gripper_positions["primary"]), 4)
    except Exception:
        pass
    return out


def quantile(values, fraction):
    ordered = sorted([float(v) for v in values])
    if not ordered:
        return None
    return ordered[int((len(ordered) - 1) * fraction)]


def cloud_points(pc):
    try:
        pts = pc.points.tolist()
    except Exception:
        pts = [[float(p[0]), float(p[1]), float(p[2])] for p in pc.points]
    return pts


def mask_from_pixels(pixels, h, w):
    grid = [[False] * w for k in range(h)]
    for p in pixels:
        if 0 <= p[1] < h and 0 <= p[0] < w:
            grid[p[1]][p[0]] = True
    return grid


def components_full(rows, test, y_min):
    """8-connected components of full-resolution pixels where test(r,g,b) holds."""
    cells = {}
    for y in range(y_min, len(rows)):
        row = rows[y]
        for x in range(len(row)):
            px = row[x]
            if test(px[0], px[1], px[2]):
                cells[(x, y)] = 1
    comps = []
    seen = {}
    for key in cells:
        if key in seen:
            continue
        stack = [key]
        seen[key] = 1
        pts = []
        while stack:
            cur = stack.pop()
            pts.append(cur)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cur[0] + dx, cur[1] + dy)
                    if nb in cells and nb not in seen:
                        seen[nb] = 1
                        stack.append(nb)
        n = len(pts)
        comps.append({"pts": pts, "n": n, "x": sum([p[0] for p in pts]) / n, "y": sum([p[1] for p in pts]) / n})
    return comps


def mask_centroid(mask):
    try:
        rows = mask.tolist()
    except Exception:
        rows = [list(r) for r in mask]
    sx = 0.0
    sy = 0.0
    n = 0
    for y in range(len(rows)):
        row = rows[y]
        for x in range(len(row)):
            if row[x]:
                sx += x
                sy += y
                n += 1
    if n == 0:
        return None
    return [sx / n, sy / n, n]


def project_pixels(pixels, cam, base_frame, h, w, label, text_prompt=None):
    """Own colour mask -> mask_to_point_cloud; fallbacks: segment_points at the pixel centroid,
    then segment_text(text_prompt) choosing the mask whose centroid is nearest that pixel."""
    info = {"label": label, "n_px": len(pixels)}
    pts = []
    try:
        pc = mask_to_point_cloud(mask_from_pixels(pixels, h, w), cam, target_frame=base_frame)
        pts = cloud_points(pc)
        info["method"] = "own_mask"
    except Exception as exc:
        note_error("mask_to_point_cloud own mask " + label, exc)
    if len(pts) < 5 and len(pixels) > 0:
        try:
            cx = int(round(sum([p[0] for p in pixels]) / len(pixels)))
            cy = int(round(sum([p[1] for p in pixels]) / len(pixels)))
            seg = segment_points(cam, [[cx, cy]])
            info["segment_points_ok"] = bool(seg.ok)
            if seg.ok and len(seg.masks) > 0:
                pc = mask_to_point_cloud(seg.masks[0], cam, target_frame=base_frame)
                pts = cloud_points(pc)
                info["method"] = "segment_points"
        except Exception as exc:
            note_error("segment_points fallback " + label, exc)
    if len(pts) < 5 and len(pixels) > 0 and text_prompt is not None:
        try:
            cx = sum([p[0] for p in pixels]) / len(pixels)
            cy = sum([p[1] for p in pixels]) / len(pixels)
            seg = segment_text(cam, text_prompt)
            info["segment_text_ok"] = bool(seg.ok)
            best = None
            if seg.ok:
                for mi in range(len(seg.masks)):
                    mc = mask_centroid(seg.masks[mi])
                    if mc is None:
                        continue
                    dd = math.sqrt((mc[0] - cx) ** 2 + (mc[1] - cy) ** 2)
                    if best is None or dd < best[0]:
                        best = [dd, mi]
            if best is not None and best[0] < 12:
                pc = mask_to_point_cloud(seg.masks[best[1]], cam, target_frame=base_frame)
                pts = cloud_points(pc)
                info["method"] = "segment_text"
                info["text_mask_px_dist"] = round(best[0], 2)
        except Exception as exc:
            note_error("segment_text fallback " + label, exc)
    info["n_pts"] = len(pts)
    if len(pts) < 5:
        return None, info
    good = []
    for p in pts:
        if p[0] == p[0] and p[1] == p[1] and p[2] == p[2] and abs(p[0]) < 5 and abs(p[1]) < 5 and abs(p[2]) < 5:
            good.append(p)
    if len(good) < 5:
        return None, info
    c = [quantile([p[0] for p in good], 0.5), quantile([p[1] for p in good], 0.5), quantile([p[2] for p in good], 0.5)]
    info["center"] = r3(c)
    info["z_lo"] = round(quantile([p[2] for p in good], 0.1), 4)
    info["z_hi"] = round(quantile([p[2] for p in good], 0.9), 4)
    info["x_ext"] = round(quantile([p[0] for p in good], 0.95) - quantile([p[0] for p in good], 0.05), 4)
    info["y_ext"] = round(quantile([p[1] for p in good], 0.95) - quantile([p[1] for p in good], 0.05), 4)
    return {"center": c, "z_lo": info["z_lo"], "z_hi": info["z_hi"], "pts": good}, info


def wide_camera(obs):
    names = [name for name in obs.cameras]
    for candidate in ("front", "agentview", "head", "top"):
        if candidate in names:
            return candidate, names
    return names[0], names


def target_test(r, g, b):
    return vpb_is_target(r, g, b)


def white_test(r, g, b):
    return min(r, g, b) > 200


# ---------------- Phase 0: metadata, goal, live scene ----------------
color = None
cam = "front"
base_frame = "robot_base"
live_rows = None
H = 256
W = 256
max_steps = None
try:
    ctx = get_task_context()
    goal = str(ctx.language)
    report["goal"] = goal
    try:
        report["task_name"] = str(ctx.task_name)
    except Exception:
        pass
    low = goal.lower()
    for c in ("red", "blue", "green", "yellow"):
        if c in low and color is None:
            color = c
    # the cube colour is the word right before "cube" when several colours appear
    words = low.replace(",", " ").replace(".", " ").split()
    for i in range(1, len(words)):
        if words[i] == "cube" and words[i - 1] in ("red", "blue", "green", "yellow"):
            color = words[i - 1]
            break
    report["goal_color"] = color
except Exception as exc:
    note_error("task context", exc)
try:
    md = robomme_plus.get_task_metadata()
    report["task_metadata"] = {str(k): str(md[k])[:300] for k in md}
    try:
        max_steps = int(md["max_steps"])
    except Exception:
        max_steps = None
except Exception as exc:
    note_error("task metadata", exc)
try:
    cmd = robomme_plus.get_controller_metadata()
    report["controller_metadata"] = {str(k): str(cmd[k])[:300] for k in cmd}
except Exception as exc:
    note_error("controller metadata", exc)
try:
    st0 = get_robot_state()
    base_frame = st0.base_frame
    report["embodiment"] = str(st0.embodiment)
    report["start_state"] = state_snapshot()
    q0 = st0.end_effector_poses["primary"].quaternion_wxyz
    report["start_quat"] = [round(float(v), 4) for v in q0]
    obs0 = get_observation()
    cam, cam_names = wide_camera(obs0)
    report["cameras"] = cam_names
    rgb0 = obs0.cameras[cam].rgb
    live_rows = rgb0.tolist()
    H = len(live_rows)
    W = len(live_rows[0])
    report["image_hw"] = [H, W]
    try:
        report["front_camera_pose"] = {"pos": r3(obs0.cameras[cam].camera_pose.position),
                                       "frame": str(obs0.cameras[cam].camera_pose.frame)}
    except Exception:
        pass
except Exception as exc:
    note_error("live observation", exc)
report["phase"] = "live_scene"

slots = []
button = None
ref_patch = None
live_feat = None
try:
    live_s2 = [row[::2] for row in live_rows[::2]]
    live_feat = vpb_frame_features(live_s2, 2, color, 16)
    slots = vpb_slots(live_feat["comps"], 6)
    report["slots_px"] = [[round(s[0], 1), round(s[1], 1), s[2]] for s in slots]
    report["live_cube_px"] = live_feat["cube"]
    button = vpb_find_button(live_s2, 2, slots, 30)
    report["button_px"] = button
    if button is not None:
        ref_patch = vpb_patch(live_s2, 2, button[0], button[1], 6)
except Exception as exc:
    note_error("live slots/button", exc)

button_3d = None
try:
    if button is not None:
        bcomp = components_full(live_rows, white_test, 30)
        best = None
        for c in bcomp:
            d = math.sqrt((c["x"] - button[0]) ** 2 + (c["y"] - button[1]) ** 2)
            if d < 8 and (best is None or c["n"] > best["n"]):
                best = c
        if best is not None:
            bres, binfo = project_pixels(best["pts"], cam, base_frame, H, W, "button")
            report["button_3d"] = binfo
            if bres is not None:
                button_3d = bres
except Exception as exc:
    note_error("button 3d", exc)

# ---------------- Phase 1: conditioning video analysis ----------------
feats = []
times = []
change = []
ee_track = []
grip_track = []
n_video = 0
try:
    video = robomme_plus.get_conditioning_video()
    for frame in video:
        n_video += 1
        try:
            rows = vpb_rows(frame.cameras[cam].rgb, 2)
            feats.append(vpb_frame_features(rows, 2, color, 16))
            times.append(float(frame.timestamp_s))
            if ref_patch is not None:
                change.append(vpb_patch_change(vpb_patch(rows, 2, button[0], button[1], 6), ref_patch, 90))
            else:
                change.append(0.0)
        except Exception as exc:
            if len(report["errors"]) < 30:
                note_error("video frame " + str(n_video), exc)
            continue
        try:
            rs = frame.robot_state
            ee_track.append(r3(rs.end_effector_poses["primary"].position))
            grip_track.append(round(float(rs.gripper_positions["primary"]), 3))
        except Exception:
            ee_track.append(None)
            grip_track.append(None)
    report["video"] = {"n_frames": n_video, "n_used": len(feats),
                       "t_first": times[0] if times else None, "t_last": times[-1] if times else None}
except Exception as exc:
    note_error("conditioning video", exc)
report["phase"] = "video_read"

answer = None
target_slot = None
try:
    dt = (times[-1] - times[0]) / max(1, len(times) - 1)
    min_span = max(0.3, 0.75 - dt)
    still_px = 1.5 + 4.0 * dt
    report["video"]["dt"] = round(dt, 4)
    report["video"]["min_span_s"] = round(min_span, 3)
    report["video"]["still_px"] = round(still_px, 3)
    events = vpb_swap_events(feats, times, slots, 7.0, 0.15, 2, 0.5, 1.6, min_span)
    report["swaps"] = [[e["a"], e["b"], round(e["t0"], 2), round(e["t1"], 2), e["n"]] for e in events]
    rests = vpb_cube_rests(feats, times, 4, still_px, 1.5)
    rest_rows = []
    for rr in rests:
        gs = [grip_track[k] for k in range(rr["start"], rr["end"] + 1) if k < len(grip_track) and grip_track[k] is not None]
        rest_rows.append([round(rr["t0"], 2), round(rr["t1"], 2), round(rr["x"], 1), round(rr["y"], 1),
                          round(sum(gs) / len(gs), 3) if gs else None])
    report["cube_rests"] = rest_rows
    press = vpb_press_from_change(change, times, 0.1, 0.1, 0.5)
    report["press_image"] = [[round(p["t0"], 2), round(p["t1"], 2), round(p["peak"], 3)] for p in press]
    # robot-state cross-check: EE close above the button in 3D (recorded; fallback only)
    press_ee = []
    if button_3d is not None:
        bc = button_3d["center"]
        flags = []
        dmin = None
        for k in range(len(ee_track)):
            p = ee_track[k]
            if p is None:
                flags.append(False)
                continue
            dxy = math.sqrt((p[0] - bc[0]) ** 2 + (p[1] - bc[1]) ** 2)
            if dmin is None or dxy < dmin[0]:
                dmin = [round(dxy, 4), round(p[2] - button_3d["z_hi"], 4), times[k] if k < len(times) else None]
            flags.append(dxy < 0.04 and p[2] < button_3d["z_hi"] + 0.05)
        report["press_ee_closest"] = dmin
        if len(flags) == len(times):
            for r in vpb_runs(flags, times, 0.0):
                press_ee.append([round(r[0], 2), round(r[1], 2)])
    report["press_ee"] = press_ee
    press_t1 = None
    if press:
        press_t1 = press[0]["t1"]
        report["press_source"] = "image"
    elif press_ee:
        press_t1 = press_ee[0][1]
        report["press_source"] = "ee"
    report["press_t1"] = press_t1
    if press_t1 is not None:
        answer = vpb_answer(slots, events, rests, press_t1, 12.0, 4.0)
    if answer is not None:
        target_slot = answer["final_slot"]
        report["answer"] = {"place_slot": answer["place_slot"], "place_t0": round(answer["rest"]["t0"], 2),
                            "place_px": [round(answer["rest"]["x"], 1), round(answer["rest"]["y"], 1)],
                            "place_dist": round(answer["place_dist"], 2), "trail": answer["trail"],
                            "final_slot": target_slot,
                            "final_px": [round(slots[target_slot][0], 1), round(slots[target_slot][1], 1)]}
        report["answer"]["literal_no_swap_slot"] = answer["place_slot"]
    # alternatives for diagnosis only
    alts = {}
    if press_ee:
        a2 = vpb_answer(slots, events, rests, press_ee[0][1], 12.0, 4.0)
        alts["ee_press"] = a2["final_slot"] if a2 is not None else None
    if len(press) > 1:
        a3 = vpb_answer(slots, events, rests, press[1]["t1"], 12.0, 4.0)
        alts["second_image_press"] = a3["final_slot"] if a3 is not None else None
    report["alternatives"] = alts
    stride_c = max(1, len(feats) // 128)
    report["cube_track"] = [[round(times[k], 2)] + ([round(feats[k]["cube"][0]), round(feats[k]["cube"][1]), feats[k]["cube"][2]] if feats[k]["cube"] else [])
                            for k in range(0, len(feats), stride_c)]
    report["press_change_trace"] = [[round(times[k], 2), round(change[k], 3)] for k in range(len(change)) if change[k] >= 0.05]
except Exception as exc:
    note_error("video analysis", exc)
report["phase"] = "video_analysed"

# ---------------- Phase 2: 3D target and cube from live depth ----------------
target_3d = None
cube_3d = None
table_z = None
try:
    if target_slot is None:
        raise ValueError("no target slot from video analysis")
    sx = slots[target_slot][0]
    sy = slots[target_slot][1]
    rings = components_full(live_rows, target_test, 16)
    best = None
    for c in rings:
        d = math.sqrt((c["x"] - sx) ** 2 + (c["y"] - sy) ** 2)
        if c["n"] >= 10 and (best is None or d < best[0]):
            best = [d, c]
    if best is None or best[0] > 10:
        raise ValueError("no ring component near target slot")
    tres, tinfo = project_pixels(best[1]["pts"], cam, base_frame, H, W, "target_ring", "target")
    tinfo["px_dist"] = round(best[0], 2)
    report["target_3d"] = tinfo
    if tres is None:
        raise ValueError("target ring projection failed")
    target_3d = tres["center"]
    table_z = tres["center"][2]
except Exception as exc:
    note_error("target 3d", exc)
try:
    cells = []
    for y in range(16, H):
        row = live_rows[y]
        for x in range(W):
            px = row[x]
            if vpb_is_color(px[0], px[1], px[2], color):
                cells.append((x, y))
    cset = {}
    for p in cells:
        cset[p] = 1
    comps = []
    seen = {}
    for key in cset:
        if key in seen:
            continue
        stack = [key]
        seen[key] = 1
        pts = []
        while stack:
            cur = stack.pop()
            pts.append(cur)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cur[0] + dx, cur[1] + dy)
                    if nb in cset and nb not in seen:
                        seen[nb] = 1
                        stack.append(nb)
        comps.append(pts)
    biggest = None
    for pts in comps:
        if biggest is None or len(pts) > len(biggest):
            biggest = pts
    report["cube_components"] = sorted([len(p) for p in comps], reverse=True)[:5]
    if biggest is None or len(biggest) < 15:
        raise ValueError("goal cube not visible in live front image")
    cres, cinfo = project_pixels(biggest, cam, base_frame, H, W, "goal_cube", str(color) + " cube")
    report["cube_3d"] = cinfo
    if cres is None:
        raise ValueError("cube projection failed")
    cube_3d = cres
except Exception as exc:
    note_error("cube 3d", exc)
report["phase"] = "perceived"

# ---------------- Phase 3: pick (staged top-down template, grasp.md) ----------------
held = False
terminal = False
moves = []


def do_move(name, pos, quat, tol, steps):
    if terminal:
        return {"name": name, "skipped": "episode already terminal", "terminated": True, "after": {"t": None, "ee": None, "grip": None}}
    res = move_to_pose(Pose((float(pos[0]), float(pos[1]), float(pos[2])), quat, base_frame),
                       tolerance=tol, max_steps=steps)
    s = exec_summary(res)
    s["name"] = name
    s["target"] = r3(pos)
    s["after"] = {"t": None, "ee": None, "grip": None} if is_terminal(s) else state_snapshot()
    moves.append(s)
    return s


def do_grip(name, opening):
    if terminal:
        return {"name": name, "skipped": "episode already terminal", "terminated": True, "after": {"t": None, "ee": None, "grip": None}}
    res = open_gripper() if opening else close_gripper()
    s = exec_summary(res)
    s["name"] = name
    s["after"] = {"t": None, "ee": None, "grip": None} if is_terminal(s) else state_snapshot()
    moves.append(s)
    return s


report["moves"] = moves
quat = None
try:
    quat = get_robot_state().end_effector_poses["primary"].quaternion_wxyz
    qw = float(quat[0])
    qx = float(quat[1])
    qy = float(quat[2])
    qz = float(quat[3])
    report["tool_z_axis"] = [round(2 * (qx * qz + qw * qy), 3), round(2 * (qy * qz - qw * qx), 3), round(1 - 2 * (qx * qx + qy * qy), 3)]
    if 1 - 2 * (qx * qx + qy * qy) > -0.8:
        # start pose not pointing down: use the generic top-down convention (180 deg about x)
        quat = [0.0, 1.0, 0.0, 0.0]
        report["quat_override"] = True
except Exception as exc:
    note_error("orientation", exc)

cube_h = None
grasp_z = None
lift_z = None
try:
    if cube_3d is None or target_3d is None:
        raise ValueError("missing cube or target 3d; not moving")
    tz = table_z
    cube_h = cube_3d["z_hi"] - tz
    if cube_h < 0.02 or cube_h > 0.08:
        report["cube_h_raw"] = round(cube_h, 4)
        cube_h = min(0.08, max(0.02, cube_h))
    report["cube_h"] = round(cube_h, 4)
    cx = cube_3d["center"][0]
    cy = cube_3d["center"][1]
    grasp_z = tz + 0.5 * cube_h
    lift_z = grasp_z + 0.15
    attempt = 0
    while attempt < 2 and not held and not terminal:
        attempt += 1
        s = do_grip("open_" + str(attempt), True)
        terminal = terminal or is_terminal(s)
        s = do_move("pre_grasp_" + str(attempt), (cx, cy, grasp_z + 0.12), quat, 0.02, 250)
        terminal = terminal or is_terminal(s)
        s = do_move("grasp_" + str(attempt), (cx, cy, grasp_z), quat, 0.015, 250)
        terminal = terminal or is_terminal(s)
        s = do_grip("close_" + str(attempt), False)
        terminal = terminal or is_terminal(s)
        s = do_move("lift_" + str(attempt), (cx, cy, lift_z), quat, 0.02, 250)
        terminal = terminal or is_terminal(s)
        g = s["after"]["grip"]
        held = g is not None and 0.05 < g < 0.95
        report["held_" + str(attempt)] = {"grip": g, "held": held}
        if not held and attempt < 2 and not terminal:
            # re-perceive the cube from the live image before the single retry
            try:
                rows2 = get_observation().cameras[cam].rgb.tolist()
                pix = []
                for y in range(16, H):
                    for x in range(W):
                        px = rows2[y][x]
                        if vpb_is_color(px[0], px[1], px[2], color):
                            pix.append((x, y))
                if len(pix) >= 15:
                    c2, i2 = project_pixels(pix, cam, base_frame, H, W, "goal_cube_retry")
                    report["cube_3d_retry"] = i2
                    if c2 is not None:
                        cx = c2["center"][0]
                        cy = c2["center"][1]
            except Exception as exc:
                note_error("retry perception", exc)
except Exception as exc:
    note_error("pick", exc)
report["phase"] = "picked" if held else "pick_failed"

# ---------------- Phase 4: transport and place on the answer ring ----------------
try:
    if not held:
        raise ValueError("cube not held; skipping place")
    if terminal:
        raise ValueError("episode terminal before place")
    tx = target_3d[0]
    ty = target_3d[1]
    s = do_move("above_target", (tx, ty, lift_z), quat, 0.02, 300)
    terminal = terminal or is_terminal(s)
    s = do_move("lower_target", (tx, ty, table_z + 0.5 * cube_h + 0.012), quat, 0.015, 250)
    terminal = terminal or is_terminal(s)
    s = do_grip("release", True)
    terminal = terminal or is_terminal(s)
    s = do_move("retreat", (tx, ty, table_z + 0.5 * cube_h + 0.12), quat, 0.02, 200)
    terminal = terminal or is_terminal(s)
    report["phase"] = "placed"
except Exception as exc:
    note_error("place", exc)

# ---------------- Phase 5: hold a few ticks so the evaluator sees the settled scene ----------------
try:
    holds = []
    k = 0
    while k < 10 and not terminal:
        st = get_robot_state()
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=st.joint_positions["primary"],
            gripper_position=st.gripper_positions["primary"], embodiment=st.embodiment)}))
        s = exec_summary(reply)
        holds.append(s)
        terminal = is_terminal(s)
        k += 1
    report["hold"] = holds[-1] if holds else None
    report["hold_ticks"] = k
except Exception as exc:
    note_error("hold", exc)
if not terminal:
    report["final_state"] = state_snapshot()
report["terminal_seen"] = terminal
result = report
