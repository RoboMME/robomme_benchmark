# VideoPlaceOrder v01_no_delete -- v00 with every construct the sandbox validator may reject removed
# (v00 was rejected: "Delete is not allowed in generated programs"). Logic unchanged.
# Strategy: track every identical target through the conditioning video (front camera), record
# the targets the goal-coloured cube rested on, follow the ordinal-th one through all later swaps
# to its final position, then pick the cube and place it on that live target.
# All task-dependent values (goal colour, ordinal, target count, positions, swaps) are perceived
# at runtime from get_task_context(), robomme_plus.get_conditioning_video() and the live cameras.
import math

# Pure-Python core of the VideoPlaceOrder video analysis (sandbox-compatible: no imports,
# no underscores, only whitelisted builtins). This block is pasted verbatim into the program.

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
            "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6, "7th": 7, "8th": 8}
CUBE_COLORS = ("red", "green", "blue", "yellow")
PARAMS = {"sig_base": 4.0, "sig_speed": 0.5, "sig_miss": 3.0, "recover_gate": 90.0, "cap2": 1.12, "still_hold": 1.12,
          "still_recover_pen": 0.0, "still_depart_pen": 0.0, "resync": 1, "still_move": 3.0, "core_frac10": 2,
          "rest_speed": 20.0, "dwell_s": 1.5, "share_r": 4.0, "share_v": 1.0}


def parse_goal(text):
    words = []
    cur = ""
    for ch in str(text).lower():
        if ch.isalnum():
            cur += ch
        else:
            if cur:
                words.append(cur)
            cur = ""
    if cur:
        words.append(cur)
    color = None
    for k in range(len(words) - 1):
        if words[k] in CUBE_COLORS and words[k + 1] in ("cube", "block"):
            color = words[k]
            break
    if color is None:
        for w in words:
            if w in CUBE_COLORS:
                color = w
                break
    ordinal = None
    last = False
    for k in range(len(words)):
        w = words[k]
        if w in ORDINALS and k + 1 < len(words) and words[k + 1] in ("target", "targets", "time", "one"):
            ordinal = ORDINALS[w]
            break
    if ordinal is None:
        for w in words:
            if w in ORDINALS:
                ordinal = ORDINALS[w]
                break
            if w.isdigit():
                ordinal = int(w)
                break
    if ordinal is None and ("last" in words or "final" in words or "latest" in words):
        last = True
    return color, ordinal, last


def is_cube_px(color, r, g, b):
    if color == "red":
        return r >= 90 and g * 10 < r * 4 and b * 10 < r * 4
    if color == "green":
        return g >= 90 and r * 10 < g * 6 and b * 10 < g * 6
    if color == "blue":
        return b >= 80 and r * 10 < b * 5 and g * 10 < b * 5
    if color == "yellow":
        return r >= 120 and g >= 110 and b * 10 < g * 4 and g * 10 > r * 7
    return False


def any_cube_px(r, g, b):
    return is_cube_px("red", r, g, b) or is_cube_px("green", r, g, b) or is_cube_px("blue", r, g, b)


def components(cells, radius):
    """cells: dict (gx, gy) -> value. Groups cells within Chebyshev distance `radius`."""
    seen = set()
    out = []
    for start in cells:
        if start in seen:
            continue
        seen.add(start)
        stack = [start]
        members = []
        for guard in range(len(cells) + 1):
            if not stack:
                break
            c = stack.pop()
            members.append(c)
            cx = c[0]
            cy = c[1]
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nb = (cx + dx, cy + dy)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        out.append(members)
    return out


def scan_frame(rows, roi, step, cube_color, depth_rows=None, bg=None, depth_tol=0.02):
    """Returns (disc_blobs, cube_blob). disc blob = (x_px, y_px, n_cells, purple_cells)."""
    x0 = roi[0]
    y0 = roi[1]
    x1 = roi[2]
    y1 = roi[3]
    tcells = {}
    ccells = {}
    for y in range(y0, y1, step):
        row = rows[y]
        drow = None
        brow = None
        if depth_rows is not None and bg is not None:
            drow = depth_rows[y]
            brow = bg.get(y)
        gy = y // step
        for x in range(x0, x1, step):
            px = row[x]
            r = px[0]
            g = px[1]
            b = px[2]
            if any_cube_px(r, g, b):
                if is_cube_px(cube_color, r, g, b):
                    ccells[(x // step, gy)] = 1
                continue
            purple = (b - g) >= 25 and (r - g) >= 10
            if purple or b >= g - 8:
                if brow is not None:
                    bd = brow.get(x)
                    dv = drow[x]
                    if bd is not None and not (dv >= bd - depth_tol):
                        continue
                tcells[(x // step, gy)] = 1 if purple else 0
    tb = []
    for members in components(tcells, 1):
        n = len(members)
        if n < 3:
            continue
        sx = 0
        sy = 0
        pc = 0
        for c in members:
            sx += c[0]
            sy += c[1]
            pc += tcells[c]
        if pc < 3:
            continue
        if pc * 10 < n * PARAMS["core_frac10"]:
            # a target touching a neutral object (robot, gripper): keep only its purple core
            sub = {}
            for c in members:
                if tcells[c]:
                    sub[c] = 1
            for sm in components(sub, 1):
                m = len(sm)
                if m < 3:
                    continue
                ax = 0
                ay = 0
                for c in sm:
                    ax += c[0]
                    ay += c[1]
                tb.append((step * ax / m + step / 2.0, step * ay / m + step / 2.0, int(m / 0.9), m))
            continue
        tb.append((step * sx / n + step / 2.0, step * sy / n + step / 2.0, n, pc))
    cube = None
    for members in components(ccells, 1):
        n = len(members)
        if n < 2:
            continue
        if cube is None or n > cube[2]:
            sx = 0
            sy = 0
            for c in members:
                sx += c[0]
                sy += c[1]
            cube = (step * sx / n + step / 2.0, step * sy / n + step / 2.0, n)
    return tb, cube


def second_item(e):
    return e[1]


def median(values):
    s = sorted(values)
    if not s:
        return None
    return s[len(s) // 2]


def area_model(blobs):
    """Least-squares fit area = a + b * y (perspective: nearer rows look bigger)."""
    n = len(blobs)
    my = sum(b[1] for b in blobs) / n
    ma = sum(b[2] for b in blobs) / n
    sxx = sum((b[1] - my) * (b[1] - my) for b in blobs)
    sxy = sum((b[1] - my) * (b[2] - ma) for b in blobs)
    slope = sxy / sxx if sxx > 1e-6 else 0.0
    if slope < 0:
        slope = 0.0
    return (ma - slope * my, slope)


def expected_area(st, y):
    a = st["amodel"][0] + st["amodel"][1] * y
    lo = 0.5 * min(st["area0"])
    return a if a > lo else lo


def tracker_new(init_blobs):
    return {"pos": [(b[0], b[1]) for b in init_blobs],
            "vel": [(0.0, 0.0) for b in init_blobs],
            "miss": [0 for b in init_blobs],
            "still": [0 for b in init_blobs],
            "area": [b[2] for b in init_blobs],
            "area0": [b[2] for b in init_blobs],
            "slots": [(b[0], b[1]) for b in init_blobs],
            "amodel": area_model(init_blobs),
            "radius": (median([b[2] for b in init_blobs]) * 4.0 / 3.14159) ** 0.5,
            "n": len(init_blobs),
            "recover_gate": PARAMS["recover_gate"],
            "events": {"merged": 0, "missing": 0, "rejected_big": 0, "snapped": 0, "recovered": 0}}


def group_assign(tracks, cands, sig, dets, st, cap2, still):
    """Exhaustive assignment for one conflict group.
    tracks: list of track ids; cands[i]: list of (det index, distance); returns dict i -> j or None."""
    best = [None, None]
    order = list(tracks)
    choice = {}
    use = {}

    def rec(k, cost):
        if best[0] is not None and cost >= best[0]:
            return
        if k == len(order):
            best[0] = cost
            best[1] = dict(choice)
            return
        i = order[k]
        for jd in cands[i]:
            j = jd[0]
            d = jd[1]
            u = use.get(j, 0)
            if u >= 2 or (u == 1 and not cap2[j]):
                continue
            if u == 1:
                rr = st["radius"]
                dd = d - rr if d > rr else 0.0
                c = (dd / sig[i]) ** 2 + 0.5
            else:
                c = (d / sig[i]) ** 2
            if still[i] and d > 0.6 * st["radius"]:
                c += PARAMS["still_depart_pen"]
            use[j] = u + 1
            choice[i] = j
            rec(k + 1, cost + c)
            use[j] = u
        choice[i] = None
        rec(k + 1, cost + 9.0)
        choice.pop(i, None)

    rec(0, 0.0)
    return best[1]


def is_still(st, i):
    if st["still"][i] < 2:
        return False
    p = st["pos"][i]
    for q in st["slots"]:
        if abs(q[0] - p[0]) + abs(q[1] - p[1]) <= 7.0:
            return True
    return False


def tracker_update(st, blobs, dt_scale):
    n = st["n"]
    pos = st["pos"]
    vel = st["vel"]
    miss = st["miss"]
    dets = []
    bigs = []
    for b in blobs:
        ea = expected_area(st, b[1])
        if b[2] > 3.2 * ea:
            st["events"]["rejected_big"] += 1
            bigs.append(b)
            continue
        if b[2] < 0.3 * ea:
            continue
        dets.append(b)
    cap2 = [d[2] >= PARAMS["cap2"] * expected_area(st, d[1]) for d in dets]
    preds = []
    sig = []
    cands = []
    for i in range(n):
        preds.append((pos[i][0] + vel[i][0] * dt_scale, pos[i][1] + vel[i][1] * dt_scale))
        sp = (vel[i][0] * vel[i][0] + vel[i][1] * vel[i][1]) ** 0.5 * dt_scale
        sg = PARAMS["sig_base"] + PARAMS["sig_speed"] * sp + PARAMS["sig_miss"] * min(miss[i], 4)
        sig.append(sg)
        lst = []
        for j in range(len(dets)):
            dx = dets[j][0] - preds[i][0]
            dy = dets[j][1] - preds[i][1]
            d = (dx * dx + dy * dy) ** 0.5
            if d <= 3.0 * sg + (st["radius"] if cap2[j] else 0.0):
                lst.append((j, d))
        lst.sort(key=second_item)
        cands.append(lst[:3])
    stillv = [is_still(st, i) for i in range(n)]
    # conflict groups: tracks linked through shared candidate detections
    parent = list(range(n))

    def root(a):
        for guard in range(len(parent) + 1):
            if parent[a] == a:
                break
            a = parent[a]
        return a
    owner = {}
    for i in range(n):
        for jd in cands[i]:
            j = jd[0]
            if j in owner:
                ra = root(owner[j])
                rb = root(i)
                if ra != rb:
                    parent[ra] = rb
            else:
                owner[j] = i
    groups = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    assign = {}
    for g in groups.values():
        if len(g) > 6:
            def best_cand_dist(i):
                return cands[i][0][1] if cands[i] else 1e9
            g.sort(key=best_cand_dist)
            for k in range(0, len(g), 5):
                sub = g[k:k + 5]
                res = group_assign(sub, cands, sig, dets, st, cap2, stillv)
                for i in sub:
                    assign[i] = res.get(i) if res else None
        else:
            res = group_assign(g, cands, sig, dets, st, cap2, stillv)
            for i in g:
                assign[i] = res.get(i) if res else None
    # recovery: lost tracks take unclaimed normal-size detections (target count is constant)
    claimed = set(assign.get(i) for i in range(n) if assign.get(i) is not None)
    free = [j for j in range(len(dets)) if j not in claimed]
    lost = [i for i in range(n) if assign.get(i) is None]
    if free and lost:
        pairs = []
        for i in lost:
            for j in free:
                dx = dets[j][0] - pos[i][0]
                dy = dets[j][1] - pos[i][1]
                # resting targets hidden by the cube or arm are usually still there: they recover
                # a distant detection only if no moving track can
                pen = PARAMS["still_recover_pen"] if is_still(st, i) else 0.0
                pairs.append(((dx * dx + dy * dy) ** 0.5 + pen, i, j))
        pairs.sort()
        ti = set()
        tj = set()
        for e in pairs:
            if e[0] > st["recover_gate"]:
                break
            if e[1] in ti or e[2] in tj:
                continue
            ti.add(e[1])
            tj.add(e[2])
            assign[e[1]] = e[2]
            st["events"]["recovered"] += 1
    users = {}
    for i in range(n):
        j = assign.get(i)
        if j is not None:
            users.setdefault(j, []).append(i)
    # a resting target that a moving target passes over: share the blob instead of dragging
    # the resting one along (the moving track would otherwise be lost)
    for j in list(users.keys()):
        if len(users[j]) != 1:
            continue
        i = users[j][0]
        if not is_still(st, i):
            continue
        d = dets[j]
        if abs(d[0] - pos[i][0]) + abs(d[1] - pos[i][1]) <= 2.0:
            continue
        rr = st["radius"]
        bestu = None
        for u in range(n):
            if u == i or assign.get(u) is not None or is_still(st, u):
                continue
            if abs(vel[u][0]) + abs(vel[u][1]) < PARAMS["share_v"]:
                continue
            du = ((preds[u][0] - d[0]) ** 2 + (preds[u][1] - d[1]) ** 2) ** 0.5
            du2 = ((pos[u][0] - d[0]) ** 2 + (pos[u][1] - d[1]) ** 2) ** 0.5
            if du2 < du:
                du = du2
            if du < PARAMS["share_r"] * rr and (bestu is None or du < bestu[0]):
                bestu = (du, u)
        if bestu is not None:
            assign[bestu[1]] = j
            users[j].append(bestu[1])
            st["events"]["shared"] = st["events"].get("shared", 0) + 1
    newpos = list(pos)
    newvel = list(vel)
    newmiss = list(miss)
    for i in range(n):
        j = assign.get(i)
        if j is None:
            sp = abs(vel[i][0]) + abs(vel[i][1])
            occluded = False
            for b in bigs:
                if abs(b[0] - pos[i][0]) + abs(b[1] - pos[i][1]) < 80:
                    occluded = True
            if miss[i] < 3 and sp > 0.8:
                newpos[i] = preds[i]
            newvel[i] = (vel[i][0] * 0.6, vel[i][1] * 0.6) if (miss[i] >= 1 or occluded) else vel[i]
            newmiss[i] = miss[i] + 1
            st["events"]["missing"] += 1
        elif len(users[j]) >= 2:
            us = users[j]
            still = [u for u in us if is_still(st, u)]
            moving = [u for u in us if u not in still]
            st["events"]["merged"] += 1
            newmiss[i] = 0
            if i in still:
                newpos[i] = pos[i]
                newvel[i] = (0.0, 0.0)
            elif len(moving) == 1:
                # blob centroid = area-weighted mean of members; stationary members are known
                cx = dets[j][0]
                cy = dets[j][1]
                wm = expected_area(st, preds[i][1])
                wsum = wm
                sx = 0.0
                sy = 0.0
                for u in still:
                    wu = expected_area(st, pos[u][1])
                    wsum += wu
                    sx += wu * pos[u][0]
                    sy += wu * pos[u][1]
                mx = (cx * wsum - sx) / wm
                my = (cy * wsum - sy) / wm
                # blend with prediction (occlusion makes the centroid noisy)
                newpos[i] = (0.6 * mx + 0.4 * preds[i][0], 0.6 * my + 0.4 * preds[i][1])
                mv = ((newpos[i][0] - pos[i][0]) / dt_scale, (newpos[i][1] - pos[i][1]) / dt_scale)
                newvel[i] = (0.5 * mv[0] + 0.5 * vel[i][0], 0.5 * mv[1] + 0.5 * vel[i][1])
            else:
                mx = sum(preds[u][0] for u in moving) / len(moving)
                my = sum(preds[u][1] for u in moving) / len(moving)
                tw = 0.0
                tx = 0.0
                ty = 0.0
                for u in us:
                    wu = expected_area(st, preds[u][1])
                    tw += wu
                    tx += wu * (pos[u][0] if u in still else preds[u][0])
                    ty += wu * (pos[u][1] if u in still else preds[u][1])
                ox = (dets[j][0] - tx / tw) * len(us) / len(moving)
                oy = (dets[j][1] - ty / tw) * len(us) / len(moving)
                newpos[i] = (preds[i][0] + 0.5 * ox, preds[i][1] + 0.5 * oy)
                newvel[i] = vel[i]
        else:
            d = dets[j]
            ddev = abs(d[0] - pos[i][0]) + abs(d[1] - pos[i][1])
            if is_still(st, i) and ddev > 2.0 and d[2] >= PARAMS["still_hold"] * expected_area(st, d[1]):
                # a resting target under a passing one: the oversized blob's shift is the other target
                newpos[i] = pos[i]
                newvel[i] = (0.0, 0.0)
                newmiss[i] = 0
                st["events"]["held"] = st["events"].get("held", 0) + 1
                continue
            if cap2[j]:
                # big blob explained by one track (partner unseen): trust prediction more
                newpos[i] = (0.5 * (d[0] + preds[i][0]), 0.5 * (d[1] + preds[i][1]))
            else:
                newpos[i] = (d[0], d[1])
            mv = ((newpos[i][0] - pos[i][0]) / dt_scale, (newpos[i][1] - pos[i][1]) / dt_scale)
            newvel[i] = (0.7 * mv[0] + 0.3 * vel[i][0], 0.7 * mv[1] + 0.3 * vel[i][1])
            newmiss[i] = 0
    for i in range(n):
        sp = abs(newvel[i][0]) + abs(newvel[i][1])
        if sp <= 1.0 and newmiss[i] > 0:
            best = None
            for k in range(len(st["slots"])):
                sx = st["slots"][k][0] - newpos[i][0]
                sy = st["slots"][k][1] - newpos[i][1]
                d2 = sx * sx + sy * sy
                if d2 <= 64.0 and (best is None or d2 < best[0]):
                    best = (d2, k)
            if best is not None:
                newpos[i] = st["slots"][best[1]]
                newvel[i] = (0.0, 0.0)
                st["events"]["snapped"] += 1
    for i in range(n):
        dx = newpos[i][0] - pos[i][0]
        dy = newpos[i][1] - pos[i][1]
        if abs(dx) + abs(dy) <= PARAMS["still_move"]:
            st["still"][i] = st["still"][i] + 1
        else:
            st["still"][i] = 0
    rb = st.get("roi")
    if rb is not None:
        for i in range(n):
            x = min(max(newpos[i][0], rb[0]), rb[2])
            y = min(max(newpos[i][1], rb[1]), rb[3])
            if x != newpos[i][0] or y != newpos[i][1]:
                newpos[i] = (x, y)
                newvel[i] = (0.0, 0.0)
    st["pos"] = newpos
    st["vel"] = newvel
    st["miss"] = newmiss


def resync(st, blobs, cube):
    """At a clean rest state (every slot visibly occupied, or the only empty one is under the cube),
    re-impose a one-to-one track/slot assignment so errors cannot propagate between motion events."""
    slots = st["slots"]
    n = st["n"]
    occ = set()
    for b in blobs:
        ea = expected_area(st, b[1])
        if b[2] >= 1.35 * ea:
            return False
        if b[2] < 0.3 * ea:
            continue
        hit = None
        for k in range(len(slots)):
            if abs(b[0] - slots[k][0]) + abs(b[1] - slots[k][1]) <= 5.0:
                hit = k
        if hit is None:
            return False
        occ.add(hit)
    if len(occ) < n:
        if len(occ) == n - 1 and cube is not None:
            miss = [k for k in range(n) if k not in occ][0]
            if abs(cube[0] - slots[miss][0]) <= 12 and -22 <= cube[1] - slots[miss][1] <= 6:
                occ.add(miss)
            else:
                return False
        else:
            return False
    pos = st["pos"]
    cur = []
    for i in range(n):
        best = None
        for k in range(n):
            d = abs(pos[i][0] - slots[k][0]) + abs(pos[i][1] - slots[k][1])
            if best is None or d < best[0]:
                best = (d, k)
        cur.append(best)
    if len(set(c[1] for c in cur)) == n and max(c[0] for c in cur) <= 6.0:
        for i in range(n):
            st["pos"][i] = slots[cur[i][1]]
            st["vel"][i] = (0.0, 0.0)
        return True
    pairs = []
    for i in range(n):
        for k in range(n):
            dx = pos[i][0] - slots[k][0]
            dy = pos[i][1] - slots[k][1]
            pairs.append((dx * dx + dy * dy, i, k))
    pairs.sort()
    ti = set()
    tk = set()
    for e in pairs:
        if e[1] in ti or e[2] in tk:
            continue
        ti.add(e[1])
        tk.add(e[2])
        st["pos"][e[1]] = slots[e[2]]
        st["vel"][e[1]] = (0.0, 0.0)
        st["miss"][e[1]] = 0
    st["events"]["resync_fix"] = st["events"].get("resync_fix", 0) + 1
    return True


def find_placements(cube_track):
    """cube_track: list of (t, cube_blob or None, track positions). A placement = the cube resting
    (speed < rest_speed px/s) within 13 px of a target for >= dwell_s, away from the cube's start."""
    home = None
    for e in cube_track:
        if e[1] is not None:
            home = (e[1][0], e[1][1])
            break
    rest = []
    for k in range(1, len(cube_track)):
        ts = cube_track[k][0]
        cb = cube_track[k][1]
        poss = cube_track[k][2]
        cb0 = cube_track[k - 1][1]
        if cb is None or cb0 is None:
            continue
        dt = ts - cube_track[k - 1][0]
        if dt <= 0:
            continue
        if (abs(cb[0] - cb0[0]) + abs(cb[1] - cb0[1])) / dt > PARAMS["rest_speed"]:
            continue
        if home is not None and abs(cb[0] - home[0]) + abs(cb[1] - home[1]) <= 12.0:
            continue
        best = None
        for i in range(len(poss)):
            dx = cb[0] - poss[i][0]
            dy = cb[1] - poss[i][1]
            d2 = dx * dx + dy * dy
            if best is None or d2 < best[0]:
                best = (d2, i)
        if best is not None and best[0] <= 13.0 * 13.0:
            rest.append((ts, best[1], cb, k))
    # group by cube location (the identity under a resting cube can only be read reliably at the
    # start of the rest, before the cube hides the target)
    segs = []
    for ts, ident, cb, k in rest:
        same = False
        if segs and abs(cb[0] - segs[-1]["cube_px"][0]) + abs(cb[1] - segs[-1]["cube_px"][1]) <= 6.0:
            if ts - segs[-1]["t1"] <= 1.5:
                same = True
            else:
                # a longer gap is the same placement when the cube was never seen anywhere else
                # meanwhile (it was only hidden, e.g. by the arm moving a distractor)
                same = True
                for kk in range(segs[-1]["k1"] + 1, k):
                    c2 = cube_track[kk][1]
                    if c2 is not None and abs(c2[0] - cb[0]) + abs(c2[1] - cb[1]) > 8.0:
                        same = False
                        break
        if same:
            segs[-1]["k1"] = k
            segs[-1]["t1"] = ts
            segs[-1]["n"] += 1
            segs[-1]["ids"][ident] = segs[-1]["ids"].get(ident, 0) + 1
        else:
            segs.append({"id": ident, "t0": ts, "t1": ts, "k1": k, "n": 1, "ids": {ident: 1},
                         "cube_px": (round(cb[0], 1), round(cb[1], 1))})
    placements = [dict(sg) for sg in segs if sg["t1"] - sg["t0"] >= PARAMS["dwell_s"]]
    return placements, segs, home


def analyse_video(times, get_rows, cube_color, step=2, get_depth=None, bg=None, keep_history=False, on_frame=None):
    """times: timestamps; get_rows(k) -> nested-list RGB rows, converted lazily (memory);
    get_depth(k) -> nested-list depth rows or None."""
    nfr = len(times)
    if nfr <= 0:
        return {"ok": False, "why": "no frames"}
    f0 = get_rows(0)
    h = len(f0)
    w = len(f0[0])
    full = (0, 0, w, h)
    counts = []
    first = []
    for k in range(min(40, nfr)):
        tb, cb = scan_frame(get_rows(k), full, step, cube_color,
                            None if get_depth is None else get_depth(k), bg)
        a = median([b[2] for b in tb]) if tb else 0
        tb = [b for b in tb if b[2] >= 0.4 * a and b[2] <= 2.0 * a]
        counts.append(len(tb))
        first.append(tb)
    # target count: the largest count seen in >= 3 early frames (targets may already be moving,
    # merging or occluded at the very start, which only lowers the count)
    best_n = None
    for c in sorted(set(counts), reverse=True):
        if counts.count(c) >= 3:
            best_n = c
            break
    if best_n is None:
        best_n = max(counts)
    k0 = counts.index(best_n)
    init = first[k0]
    if best_n < 2:
        return {"ok": False, "why": "fewer than 2 targets", "counts": counts}
    xs = [b[0] for b in init]
    ys = [b[1] for b in init]
    margin = 70
    roi = (max(0, int(min(xs)) - margin), max(0, int(min(ys)) - margin),
           min(w, int(max(xs)) + margin), min(h, int(max(ys)) + margin))
    tr = tracker_new(init)
    tr["roi"] = roi
    t_prev = times[k0]
    dts = [times[k + 1] - times[k] for k in range(nfr - 1)]
    dt_nom = median(dts) if dts else 0.1
    if not dt_nom or dt_nom <= 0:
        dt_nom = 0.1
    cube_track = []
    history = []
    for k in range(k0, nfr):
        if on_frame is not None:
            on_frame(k)
        ts = times[k]
        rows = get_rows(k)
        tb, cb = scan_frame(rows, roi, step, cube_color,
                            None if get_depth is None else get_depth(k), bg)
        if k > k0:
            tracker_update(tr, tb, max(0.25, (ts - t_prev) / dt_nom))
            if PARAMS["resync"]:
                resync(tr, tb, cb)
        t_prev = ts
        cube_track.append((ts, cb, list(tr["pos"])))
        if keep_history:
            history.append((k, ts, list(tr["pos"]), tb, cb))
    placements, segs, cube_home = find_placements(cube_track)
    final = cube_track[-1][2]
    out = {"ok": True, "n_targets": best_n, "counts": counts, "slots": tr["slots"], "roi": roi,
           "placements": placements, "raw_segments": segs, "final_positions": final, "cube_home": cube_home,
           "events": tr["events"], "dt_nom": dt_nom, "n_frames": nfr - k0}
    if keep_history:
        out["history"] = history
    return out


# =============================================================================================
# Runtime part (v00): perceive the answer from the conditioning video, then pick the goal cube
# and place it on the live target that carries the answer identity. Every phase is guarded and
# instrumented into `result`; nothing task-dependent is a constant.
# =============================================================================================

result = {"version": "v01_no_delete", "phase_status": {}, "errors": [], "keepalive_ticks": 0,
          "motions": []}
STATE = {"ended": False}


def r4(v):
    return round(float(v), 4)


def note(phase, status):
    result["phase_status"][phase] = status


def fail(phase, err):
    result["errors"].append(phase + ": " + str(err))
    note(phase, "error")
    if "Episode has ended" in str(err):
        STATE["ended"] = True


def is_terminal(res):
    flag = False
    for key in ("terminated", "truncated"):
        try:
            if key == "terminated" and res.terminated:
                flag = True
            if key == "truncated" and res.truncated:
                flag = True
        except Exception:
            pass
    return flag


def describe(res):
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
    try:
        out["terminated"] = bool(res.terminated)
        out["truncated"] = bool(res.truncated)
    except Exception:
        pass
    try:
        out["error"] = str(res.error)[:300]
    except Exception:
        pass
    return out


def hold_tick():
    """One control tick holding the measured joints and gripper (keeps the progress timer alive)."""
    if STATE["ended"] or STATE.get("tick_broken"):
        return False
    try:
        st = get_robot_state()
        reply = step(RobotAction(arms={"primary": ArmCommand(
            mode="joint_position", target=st.joint_positions["primary"],
            gripper_position=st.gripper_positions["primary"], embodiment=st.embodiment)}))
        result["keepalive_ticks"] += 1
        if result["keepalive_ticks"] == 1:
            result["first_tick"] = describe(reply)
        if is_terminal(reply):
            STATE["ended"] = True
    except Exception as e:
        STATE["tick_broken"] = True
        result["errors"].append("hold_tick: " + str(e)[:300])
        if "Episode has ended" in str(e):
            STATE["ended"] = True
        return False
    return True


def ee_now():
    st = get_robot_state()
    p = st.end_effector_poses["primary"]
    return [r4(v) for v in p.position], [r4(v) for v in p.quaternion_wxyz], float(st.gripper_positions["primary"])


def move(label, xyz, quat, frame, tol=0.012, max_steps=150):
    if STATE["ended"]:
        result["motions"].append({"label": label, "skipped": "episode ended"})
        return False
    target = Pose([float(xyz[0]), float(xyz[1]), float(xyz[2])], quat, frame)
    res = move_to_pose(target, tolerance=tol, max_steps=max_steps)
    rec = describe(res)
    if rec.get("ok") is not True and rec.get("steps") == 0 and "max_steps" in str(rec.get("error")) \
            and not is_terminal(res):
        # the plan was refused before any motion (planned waypoints exceed the cap): safe to re-plan
        rec["first_attempt"] = dict(rec)
        res = move_to_pose(target, tolerance=tol, max_steps=2 * max_steps)
        first = rec["first_attempt"]
        rec = describe(res)
        rec["first_attempt"] = first
    rec["label"] = label
    rec["target"] = [r4(v) for v in xyz]
    try:
        pos, q, g = ee_now()
        rec["ee_after"] = pos
        rec["gripper_after"] = r4(g)
    except Exception as e:
        rec["state_error"] = str(e)[:200]
        if "Episode has ended" in str(e):
            STATE["ended"] = True
    result["motions"].append(rec)
    if is_terminal(res):
        STATE["ended"] = True
    return rec.get("ok") is True


def grip(label, closing):
    if STATE["ended"]:
        result["motions"].append({"label": label, "skipped": "episode ended"})
        return None
    res = close_gripper() if closing else open_gripper()
    rec = describe(res)
    rec["label"] = label
    g = None
    try:
        g = float(get_robot_state().gripper_positions["primary"])
        rec["gripper_after"] = r4(g)
    except Exception as e:
        rec["state_error"] = str(e)[:200]
        if "Episode has ended" in str(e):
            STATE["ended"] = True
    result["motions"].append(rec)
    if is_terminal(res):
        STATE["ended"] = True
    return g


def quat_matrix(q):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w = w / n
    x = x / n
    y = y / n
    z = z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def backproject(u, v, d, cam, conv):
    """Pixel (u, v) with depth d [m] -> robot_base point, for a camera-frame convention:
    cv = x right / y down / z forward (z-depth); gl = x right / y up / z backward (z-depth);
    *_ray = depth measured along the ray."""
    K = cam["K"]
    xn = (u - K[0][2]) / K[0][0]
    yn = (v - K[1][2]) / K[1][1]
    z = d
    if conv == "cv_ray" or conv == "gl_ray":
        z = d / (1.0 + xn * xn + yn * yn) ** 0.5
    if conv == "cv" or conv == "cv_ray":
        p = (xn * z, yn * z, z)
    else:
        p = (xn * z, -yn * z, -z)
    R = cam["R"]
    t = cam["t"]
    return (R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2] + t[0],
            R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2] + t[1],
            R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2] + t[2])


def depth_at(depth, u, v, rad=2):
    vals = []
    h = len(depth)
    w = len(depth[0])
    for y in range(int(v) - rad, int(v) + rad + 1):
        for x in range(int(u) - rad, int(u) + rad + 1):
            if 0 <= y < h and 0 <= x < w:
                d = depth[y][x]
                if d == d and d > 0.05 and d < 5.0:
                    vals.append(d)
    return median(vals)


def cube_pixels(rows, color, cx, cy, rad=14):
    pts = []
    h = len(rows)
    w = len(rows[0])
    for y in range(max(0, int(cy) - rad), min(h, int(cy) + rad + 1)):
        for x in range(max(0, int(cx) - rad), min(w, int(cx) + rad + 1)):
            px = rows[y][x]
            if is_cube_px(color, px[0], px[1], px[2]):
                pts.append((x, y))
    return pts


def qtile(arr, f):
    return arr[int((len(arr) - 1) * f)]


def cloud_stats(points):
    xs = sorted(float(p[0]) for p in points)
    ys = sorted(float(p[1]) for p in points)
    zs = sorted(float(p[2]) for p in points)
    n = len(xs)
    if n == 0:
        return None
    return {"n": n, "med": [r4(qtile(xs, 0.5)), r4(qtile(ys, 0.5)), r4(qtile(zs, 0.5))],
            "lo": [r4(qtile(xs, 0.05)), r4(qtile(ys, 0.05)), r4(qtile(zs, 0.05))],
            "hi": [r4(qtile(xs, 0.95)), r4(qtile(ys, 0.95)), r4(qtile(zs, 0.95))]}


def provider_cloud(label, u, v):
    """SAM point prompt at (u, v) on the live front image -> projected cloud stats."""
    rec = {"prompt_px": [r4(u), r4(v)]}
    seg = segment_points("front", [[float(u), float(v)]])
    rec["seg_ok"] = bool(seg.ok)
    if not seg.ok:
        try:
            rec["seg_error"] = str(seg.error)[:200]
        except Exception:
            pass
        result["provider_" + label] = rec
        return None
    cand = None
    try:
        cand = seg.segmentations[0]
        rec["mask_source"] = "segmentations[0]"
    except Exception:
        cand = None
    if cand is None:
        try:
            cand = seg.masks[0]
            rec["mask_source"] = "masks[0]"
        except Exception:
            cand = None
    try:
        rec["score"] = r4(cand.score)
    except Exception:
        try:
            rec["score"] = r4(seg.scores[0])
        except Exception:
            pass
    try:
        rec["box"] = [r4(b) for b in cand.box_xyxy]
    except Exception:
        pass
    if cand is None:
        result["provider_" + label] = rec
        return None
    pc = mask_to_point_cloud(cand, "front", target_frame=get_robot_state().base_frame)
    pts = pc.points
    st = cloud_stats(pts) if len(pts) > 0 else None
    rec["cloud"] = st
    result["provider_" + label] = rec
    return st


# ---------------------------------------------------------------------------- phase 1: goal
goal = None
color = None
ordinal = None
want_last = False
try:
    ctx = get_task_context()
    goal = str(ctx.language)
    result["goal"] = goal
    color, ordinal, want_last = parse_goal(goal)
    result["parsed_goal"] = {"color": color, "ordinal": ordinal, "last": want_last}
    note("goal", "ok" if color is not None and (ordinal is not None or want_last) else "unparsed")
except Exception as e:
    fail("goal", e)

# ---------------------------------------------------------------------------- phase 2: metadata
try:
    md = robomme_plus.get_task_metadata()
    result["task_metadata"] = str(md)[:1500]
    try:
        result["max_steps"] = int(md["max_steps"])
    except Exception:
        pass
    try:
        result["video_sampling"] = str(md["conditioning_video_sampling"])[:300]
    except Exception:
        pass
    cm = robomme_plus.get_controller_metadata()
    result["controller_metadata"] = str(cm)[:1500]
    obs0 = get_observation()
    result["camera_names"] = [str(n) for n in obs0.cameras]
    pos0, q0, g0 = ee_now()
    st0 = get_robot_state()
    base_frame = st0.base_frame
    result["start_ee"] = {"pos": pos0, "quat": q0, "gripper": r4(g0), "base_frame": str(base_frame),
                          "embodiment": str(st0.embodiment)}
    note("metadata", "ok")
except Exception as e:
    fail("metadata", e)

# ---------------------------------------------------------------------------- phase 3: video
analysis = None
video = None
try:
    video = robomme_plus.get_conditioning_video()
    nv = len(video)
    result["video_frames"] = nv
    times = [float(video[k].timestamp_s) for k in range(nv)]
    result["video_t"] = [r4(times[0]), r4(times[-1])] if nv else []
    if nv:
        cp = video[0].cameras["front"].camera_pose
        result["video_front_pose0"] = [[r4(v) for v in cp.position], [r4(v) for v in cp.quaternion_wxyz]]
        sh = video[0].cameras["front"].rgb.shape
        result["video_rgb_shape"] = [int(v) for v in sh]
    hold_tick()
    cache = {}

    def get_rows(k):
        if k not in cache:
            cache.clear()
            rgb = video[k].cameras["front"].rgb
            try:
                cache[k] = rgb.tolist()
            except Exception:
                cache[k] = [[[int(c) for c in px] for px in row] for row in rgb]
        return cache[k]

    def on_frame(k):
        if k % 48 == 0:
            hold_tick()

    analysis = analyse_video(times, get_rows, color, on_frame=on_frame)
    cache.clear()
    hold_tick()
    a = analysis
    summary = {"ok": a.get("ok"), "why": a.get("why"), "n_targets": a.get("n_targets"),
               "counts": a.get("counts"), "roi": a.get("roi"), "events": a.get("events"),
               "dt_nom": a.get("dt_nom"), "n_frames": a.get("n_frames"), "cube_home": a.get("cube_home")}
    if a.get("ok"):
        summary["slots"] = [[round(p[0], 1), round(p[1], 1)] for p in a["slots"]]
        summary["final_positions"] = [[round(p[0], 1), round(p[1], 1)] for p in a["final_positions"]]
        summary["placements"] = [{"id": s["id"], "t0": r4(s["t0"]), "t1": r4(s["t1"]), "n": s["n"],
                                  "cube_px": s["cube_px"], "ids": [[int(i), int(c)] for i, c in s["ids"].items()]}
                                 for s in a["placements"]]
        summary["raw_segments"] = [{"id": s["id"], "t0": r4(s["t0"]), "t1": r4(s["t1"]), "n": s["n"],
                                    "cube_px": s["cube_px"]} for s in a["raw_segments"]][:40]
    result["video_analysis"] = summary
    note("video", "ok" if a.get("ok") else "failed")
except Exception as e:
    fail("video", e)

# ---------------------------------------------------------------------------- phase 4: answer
answer_px = None
try:
    pl = analysis["placements"]
    if want_last and pl:
        idx = len(pl) - 1
    else:
        idx = ordinal - 1
    if idx < 0 or idx >= len(pl):
        raise ValueError("ordinal %s but only %d placements detected" % (str(ordinal), len(pl)))
    ident = pl[idx]["id"]
    fp = analysis["final_positions"][ident]
    answer_px = (fp[0], fp[1])
    result["answer"] = {"placement_index": idx, "identity": ident, "final_px": [round(fp[0], 1), round(fp[1], 1)]}
    note("answer", "ok")
except Exception as e:
    fail("answer", e)

# ---------------------------------------------------------------------------- phase 5: live scene
cam = None
target_px = None
cube_px = None
live_rows = None
live_depth = None
try:
    obs = get_observation()
    fc = obs.cameras["front"]
    live_rows = fc.rgb.tolist()
    live_depth = fc.depth_m.tolist()
    Kl = fc.intrinsics.tolist()
    cpose = fc.camera_pose
    cam = {"K": Kl, "R": quat_matrix([float(v) for v in cpose.quaternion_wxyz]),
           "t": [float(v) for v in cpose.position]}
    result["live_front"] = {"K": [[r4(v) for v in row] for row in Kl],
                            "pose": [[r4(v) for v in cpose.position], [r4(v) for v in cpose.quaternion_wxyz]],
                            "pose_frame": str(cpose.frame)}
    full = (0, 0, len(live_rows[0]), len(live_rows))
    tb, cb = scan_frame(live_rows, full, 1, color)
    am = median([b[2] for b in tb]) if tb else 0
    tbf = [b for b in tb if b[2] >= 0.35 * am and b[2] <= 2.5 * am]
    result["live_targets"] = [[round(b[0], 1), round(b[1], 1), b[2]] for b in tbf]
    result["live_cube_blob"] = None if cb is None else [round(cb[0], 1), round(cb[1], 1), cb[2]]
    if cb is None:
        raise ValueError("goal-colour cube not visible in live front image")
    cube_px = (cb[0], cb[1])
    if answer_px is not None and tbf:
        best = None
        for b in tbf:
            d = ((b[0] - answer_px[0]) ** 2 + (b[1] - answer_px[1]) ** 2) ** 0.5
            if best is None or d < best[0]:
                best = (d, b)
        result["live_match"] = {"dist_px": round(best[0], 2), "blob": [round(best[1][0], 1), round(best[1][1], 1)]}
        target_px = (best[1][0], best[1][1]) if best[0] <= 20.0 else answer_px
    else:
        target_px = answer_px
    note("live_scene", "ok" if target_px is not None else "no target")
except Exception as e:
    fail("live_scene", e)

# ---------------------------------------------------------------------------- phase 6: 3-D
conv = "cv"
cube_xyz = None
cube_top = None
table_z = None
target_xyz = None
try:
    pix = cube_pixels(live_rows, color, cube_px[0], cube_px[1])
    result["cube_pixel_count"] = len(pix)
    prov = None
    try:
        prov = provider_cloud("cube", cube_px[0], cube_px[1])
    except Exception as e:
        result["errors"].append("provider_cube: " + str(e)[:200])
    errs = {}
    mine = {}
    for cv in ("cv", "gl", "cv_ray", "gl_ray"):
        acc = [0.0, 0.0, 0.0]
        cnt = 0
        for (x, y) in pix:
            d = live_depth[y][x]
            if d == d and d > 0.05 and d < 5.0:
                p = backproject(x, y, d, cam, cv)
                acc[0] += p[0]
                acc[1] += p[1]
                acc[2] += p[2]
                cnt += 1
        if cnt:
            m = (acc[0] / cnt, acc[1] / cnt, acc[2] / cnt)
            mine[cv] = [r4(v) for v in m]
            if prov is not None:
                errs[cv] = r4(((m[0] - prov["med"][0]) ** 2 + (m[1] - prov["med"][1]) ** 2 +
                               (m[2] - prov["med"][2]) ** 2) ** 0.5)
    result["calibration"] = {"mine": mine, "err_vs_provider": errs}
    if errs:
        conv = None
        for cname in errs:
            if conv is None or errs[cname] < errs[conv]:
                conv = cname
    result["calibration"]["convention"] = conv
    calib_ok = bool(errs) and errs[conv] < 0.02
    result["calibration"]["ok"] = calib_ok
    # table height from depth just around the target pixel (flat disc on the table)
    dt = depth_at(live_depth, target_px[0], target_px[1], 2)
    tp = backproject(target_px[0] - 0.5, target_px[1] - 0.5, dt, cam, conv)
    result["target_backproject"] = [r4(v) for v in tp]
    tprov = None
    try:
        tprov = provider_cloud("target", target_px[0], target_px[1])
    except Exception as e:
        result["errors"].append("provider_target: " + str(e)[:200])
    if calib_ok or tprov is None:
        target_xyz = tp
    else:
        ext = max(tprov["hi"][0] - tprov["lo"][0], tprov["hi"][1] - tprov["lo"][1])
        target_xyz = tuple(tprov["med"]) if ext < 0.12 else tp
    table_z = target_xyz[2]
    # cube: centre from my calibrated projection (provider as cross-check), top from high z quantile
    zs = []
    xs = []
    ys = []
    for (x, y) in pix:
        d = live_depth[y][x]
        if d == d and d > 0.05 and d < 5.0:
            p = backproject(x, y, d, cam, conv)
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])
    xs.sort()
    ys.sort()
    zs.sort()
    if not zs:
        raise ValueError("no valid cube depth")
    cube_top = zs[int(0.9 * (len(zs) - 1))]
    if prov is not None and not calib_ok:
        cube_xyz = (prov["med"][0], prov["med"][1], prov["hi"][2])
        cube_top = prov["hi"][2]
    else:
        cube_xyz = (xs[len(xs) // 2], ys[len(ys) // 2], cube_top)
    result["cube_xyz"] = [r4(v) for v in cube_xyz]
    result["cube_top_z"] = r4(cube_top)
    result["table_z"] = r4(table_z)
    result["target_xyz"] = [r4(v) for v in target_xyz]
    note("geometry", "ok")
except Exception as e:
    fail("geometry", e)

# ---------------------------------------------------------------------------- phase 7: pick & place
try:
    if cube_xyz is None or target_xyz is None:
        raise ValueError("missing geometry; manipulation skipped")
    pos0, quat, g0 = ee_now()
    frame = get_robot_state().base_frame
    h = cube_top - table_z
    if h < 0.015 or h > 0.08:
        result["cube_height_suspicious"] = r4(h)
        h = 0.04
    result["cube_height_used"] = r4(h)
    grasp_z = table_z + 0.5 * h
    hover = cube_top + 0.10
    grip("open_initial", False)
    move("pre_grasp", (cube_xyz[0], cube_xyz[1], hover), quat, frame, 0.015, 150)
    move("grasp", (cube_xyz[0], cube_xyz[1], grasp_z), quat, frame, 0.008, 100)
    g = grip("close", True)
    move("lift", (cube_xyz[0], cube_xyz[1], hover), quat, frame, 0.015, 100)
    try:
        g = float(get_robot_state().gripper_positions["primary"])
    except Exception:
        pass
    held = g is not None and g > 0.1
    result["held_after_lift"] = {"gripper": None if g is None else r4(g), "held": held}
    if not held and not STATE["ended"]:
        # one re-grasp after a confirmed miss: re-perceive the cube first
        grip("open_retry", False)
        o2 = get_observation().cameras["front"]
        rows2 = o2.rgb.tolist()
        depth2 = o2.depth_m.tolist()
        tb2, cb2 = scan_frame(rows2, (0, 0, len(rows2[0]), len(rows2)), 1, color)
        result["retry_cube_blob"] = None if cb2 is None else [round(cb2[0], 1), round(cb2[1], 1), cb2[2]]
        if cb2 is not None:
            pix2 = cube_pixels(rows2, color, cb2[0], cb2[1])
            ps = []
            for (x, y) in pix2:
                d = depth2[y][x]
                if d == d and d > 0.05 and d < 5.0:
                    ps.append(backproject(x, y, d, cam, conv))
            if ps:
                c2 = cloud_stats(ps)
                result["retry_cube_cloud"] = c2
                cx2 = c2["med"][0]
                cy2 = c2["med"][1]
                move("pre_grasp_retry", (cx2, cy2, hover), quat, frame, 0.015, 120)
                move("grasp_retry", (cx2, cy2, grasp_z - 0.005), quat, frame, 0.008, 100)
                grip("close_retry", True)
                move("lift_retry", (cx2, cy2, hover), quat, frame, 0.015, 100)
                try:
                    g = float(get_robot_state().gripper_positions["primary"])
                except Exception:
                    pass
                held = g is not None and g > 0.1
                result["held_after_retry"] = {"gripper": None if g is None else r4(g), "held": held}
    carry_z = table_z + h + 0.12
    move("transit", (target_xyz[0], target_xyz[1], carry_z), quat, frame, 0.015, 150)
    release_z = table_z + 0.5 * h + 0.015
    move("place_descend", (target_xyz[0], target_xyz[1], release_z), quat, frame, 0.008, 100)
    grip("release", False)
    for settle in range(6):
        if STATE["ended"]:
            break
        hold_tick()
    move("retreat", (target_xyz[0], target_xyz[1], carry_z), quat, frame, 0.02, 100)
    note("manipulation", "done")
except Exception as e:
    fail("manipulation", e)

# ---------------------------------------------------------------------------- phase 8: check
try:
    if not STATE["ended"]:
        o3 = get_observation().cameras["front"]
        rows3 = o3.rgb.tolist()
        tb3, cb3 = scan_frame(rows3, (0, 0, len(rows3[0]), len(rows3)), 1, color)
        result["final_cube_blob"] = None if cb3 is None else [round(cb3[0], 1), round(cb3[1], 1), cb3[2]]
        if cb3 is not None and target_px is not None:
            result["final_cube_to_target_px"] = round(((cb3[0] - target_px[0]) ** 2 + (cb3[1] - target_px[1]) ** 2) ** 0.5, 2)
        st = get_robot_state()
        result["final_time_s"] = r4(st.timestamp_s)
    result["episode_ended_during_program"] = STATE["ended"]
    note("check", "ok")
except Exception as e:
    fail("check", e)
