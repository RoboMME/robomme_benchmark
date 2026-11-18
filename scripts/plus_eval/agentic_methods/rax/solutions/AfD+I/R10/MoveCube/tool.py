import math
import numpy as np
from percep import *

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def Rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])

def yaw_of(state):
    """gripper finger-axis yaw: angle of gripper y-axis in world xy."""
    rv = np.array(state[3:6], float)
    th = np.linalg.norm(rv)
    if th < 1e-9:
        Rm = np.eye(3)
    else:
        k = rv / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Rm = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
    y = Rm[:, 1]
    return math.atan2(y[1], y[0])

def rotate_yaw(R, angle, rate=0.12, log=print):
    """rotate gripper about world z by angle (rad), closed loop on measured yaw."""
    y0 = yaw_of(R.state())
    goal = y0 + angle
    for _ in range(6):
        cur = yaw_of(R.state())
        rem = wrap(goal - cur)
        if abs(rem) < 0.02 or R.ended:
            break
        n = max(1, int(math.ceil(abs(rem) / rate)))
        a = rem / n / 0.5
        R.step([[0, 0, 0, 0, 0, a, R.grip]] * n)
    log("yaw", y0, "->", yaw_of(R.state()), "wanted", goal)
    return wrap(yaw_of(R.state()) - y0)

def find_cube(R, approx=None, maxjump=0.12):
    """cube centre xy (and top z) from front camera red blob (fallback wrist)."""
    from controller import current_arrays
    m, A = current_arrays(R)
    best = None
    for cam in ("front", "wrist"):
        P = backproject(A[(cam, "depth")], m["camera_intrinsics"][cam]["matrix"], m["camera_poses"][cam])
        for n, u, v, pix in blobs(mask_red(A[(cam, "rgb")]), min_px=6):
            pts = P[pix[:, 0], pix[:, 1]]
            pts = pts[~np.isnan(pts).any(1)]
            if len(pts) < 4:
                continue
            c = np.median(pts, 0)
            if c[2] > 0.08:
                continue
            if approx is not None and np.linalg.norm(c[:2] - approx[:2]) > maxjump:
                continue
            best = c; break
        if best is not None:
            break
    if best is None:
        return None
    # surface points are on the camera-facing side; nudge back toward centre
    return best

def find_target(R):
    from controller import current_arrays
    m, A = current_arrays(R)
    P = backproject(A[("front", "depth")], m["camera_intrinsics"]["front"]["matrix"], m["camera_poses"]["front"])
    bl = blobs(mask_purple(A[("front", "rgb")]), min_px=3)
    if not bl:
        return None
    pix = np.concatenate([b[3] for b in bl if b[0] >= 3 and np.hypot(b[1] - bl[0][1], b[2] - bl[0][2]) < 12])
    pts = P[pix[:, 0], pix[:, 1]]
    pts = pts[~np.isnan(pts).any(1)]
    return np.median(pts, 0)

def push_cube(R, offset, z_push, target, cube, half_along, mode_dir, log=print,
              max_iter=45, stop_tol=0.006):
    """Tool already held & rotated. offset: head centre - TCP (xy, world).
    Move head behind cube, lower, push toward target closed loop on the seen cube."""
    u = (target[:2] - cube[:2]); dist = np.linalg.norm(u); u = u / max(dist, 1e-6)
    gap = 0.02
    stand = 0.02 + half_along + gap
    hp = cube[:2] - u * stand
    g = hp - offset
    zt = max(R.state()[2], z_push + 0.06)
    R.move_to([R.state()[0], R.state()[1], zt], tol=0.006)
    R.move_to([g[0], g[1], zt], tol=0.004)
    R.move_to([g[0], g[1], z_push], tol=0.003, speed=0.5)
    log("prepush tcp", np.round(R.state(), 3).tolist(), "head", (R.state()[:2] + offset).round(3).tolist())
    c0 = cube[:2].copy(); last_c = cube[:2].copy(); travelled = 0.0; seen_ok = 0
    for it in range(max_iter):
        if R.ended or R.complete:
            break
        st = R.state()
        if getattr(R, "tool_held", False) and st[6] + st[7] < 0.003:
            log("tool lost during push"); return None
        c = find_cube(R, approx=np.r_[last_c, 0], maxjump=0.06)
        head = R.state()[:2] + offset
        if c is not None:
            cxy = c[:2]
        else:
            # dead reckoning: cube sits just ahead of the pusher once in contact
            ahead = head + u * (half_along + 0.02)
            cxy = last_c if np.dot(ahead - last_c, u) < 0 else ahead
        last_c = cxy
        d = target[:2] - cxy
        nd = np.linalg.norm(d)
        moved = np.linalg.norm(cxy - c0)
        log("iter", it, "cube", np.round(cxy, 3).tolist(), "seen", c is not None, "rem", round(float(nd), 4), "trav", round(travelled, 3), "moved", round(float(moved), 3))
        if nd < stop_tol or np.dot(d, u) < 0.002:
            break
        if travelled > gap + 0.07 and moved < 0.01:
            log("stall: cube not moving")
            return None
        if travelled > dist + gap + 0.12:
            log("travel limit")
            break
        dirn = d / nd
        # keep mostly along nominal push axis to stay in contact
        dirn = 0.6 * u + 0.4 * dirn; dirn /= np.linalg.norm(dirn)
        stepl = min(0.025, max(0.004, float(np.dot(d, dirn))))
        p = R.state()[:3]
        R.move_to([p[0] + dirn[0] * stepl, p[1] + dirn[1] * stepl, z_push], tol=0.002, max_steps=12, speed=0.25)
        travelled += stepl
    # back off and lift
    p = R.state()[:3]
    R.move_to([p[0] - u[0] * 0.03, p[1] - u[1] * 0.03, z_push + 0.005], tol=0.004, max_steps=10, speed=0.3)
    R.move_to([R.state()[0], R.state()[1], z_push + 0.10], tol=0.01, max_steps=20)
    return last_c

def scene_now(R):
    from controller import current_arrays
    m, A = current_arrays(R)
    out = {}
    for cam in ("front", "wrist"):
        P = backproject(A[(cam, "depth")], m["camera_intrinsics"][cam]["matrix"], m["camera_poses"][cam])
        out[cam] = (P, A[(cam, "rgb")])
    return out

def mask_gray(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (np.abs(r - g) < 25) & (np.abs(g - b) < 25) & (r > 85)

def line_frac(mask, p0, p1, skip=0.2):
    n = 40; hits = 0; cnt = 0
    for i in range(n + 1):
        t = i / n
        if t < skip or t > 1 - skip:
            continue
        u = p0[0] + (p1[0] - p0[0]) * t; v = p0[1] + (p1[1] - p0[1]) * t
        ui, vi = int(round(u)), int(round(v))
        win = mask[max(0, vi - 2):vi + 3, max(0, ui - 2):ui + 3]
        cnt += 1; hits += bool(win.any())
    return hits / max(cnt, 1)

def detect_tools(P, rgb, log=print):
    """teal-handled tools with salmon heads joined by a gray shaft."""
    handles = []
    for n, u, v, pix in blobs(mask_teal(rgb), min_px=12):
        pts = P[pix[:, 0], pix[:, 1]]; pts = pts[~np.isnan(pts).any(1)]
        if len(pts) < 8 or n > 600:
            continue
        if np.percentile(pts[:, 2], 90) > 0.06:
            continue
        handles.append((np.median(pts, 0), (u, v)))
    heads = []
    for n, u, v, pix in blobs(mask_salmon(rgb), min_px=8):
        pts = P[pix[:, 0], pix[:, 1]]; pts = pts[~np.isnan(pts).any(1)]
        if len(pts) < 6 or np.percentile(pts[:, 2], 90) > 0.07:
            continue
        heads.append((n, np.median(pts, 0), pts, (u, v)))
    gm = mask_gray(rgb)
    pairs = []
    for i, (h, hp) in enumerate(handles):
        for j, (n, c, pts, cp) in enumerate(heads):
            d = np.linalg.norm(c[:2] - h[:2])
            if not (0.05 < d < 0.45):
                continue
            f = line_frac(gm, hp, cp)
            pairs.append((f, i, j))
    pairs.sort(reverse=True)
    tools = []; ui = set(); uj = set()
    for f, i, j in pairs:
        if f < 0.5 or i in ui or j in uj:
            continue
        ui.add(i); uj.add(j)
        h, hp = handles[i]; n, c, pts, cp = heads[j]
        ax = c[:2] - h[:2]; L = np.linalg.norm(ax); ax = ax / L
        perp = np.array([-ax[1], ax[0]])
        rel = pts[:, :2] - c[:2]
        tools.append(dict(handle=h, head=c, head_n=n, L=L, heading=math.atan2(ax[1], ax[0]),
                          half_w=float(np.percentile(np.abs(rel @ perp), 95)),
                          half_l=float(np.percentile(np.abs(rel @ ax), 95)),
                          tip=float(np.max(rel @ ax)), shaft=f, hpix=hp, cpix=cp))
    return tools

def use_tool(R, tool, cube, target, mode="side", log=print, z_grasp=None):
    """grasp tool handle, rotate so the head can push cube to target, push."""
    H = tool["handle"]; phi = tool["heading"]
    # finger axis must be perpendicular to the shaft: finger yaw = phi + pi/2, reduce to near -pi/2 (default)
    fy = phi + math.pi / 2
    cands = [fy + k * math.pi for k in (-2, -1, 0, 1, 2)]
    fy = min(cands, key=lambda a: abs(a - (-math.pi / 2)))
    R.grip = -1
    R.move_to([R.state()[0], R.state()[1], max(R.state()[2], 0.15)], tol=0.01)
    R.move_to([H[0], H[1], 0.15], tol=0.006)
    R.orient(fy)
    R.move_to([H[0], H[1], 0.06], tol=0.004)
    zg = z_grasp if z_grasp is not None else max(0.012, H[2] - 0.010)
    R.move_to([H[0], H[1], zg], tol=0.003, speed=0.5)
    R.gripper(1, 14)
    st = R.state()
    log("grasped", st.tolist())
    if st[6] + st[7] < 0.004:
        log("grasp missed (fingers closed)")
        return "miss"
    g0 = st[:2].copy()
    off = tool["head"][:2] - g0
    R.move_to([st[0], st[1], zg + 0.07], tol=0.005, speed=0.6)
    # choose head heading
    u = target[:2] - cube[:2]; au = math.atan2(u[1], u[0])
    if mode == "side":
        want = [au + math.pi / 2, au - math.pi / 2]
    else:
        want = [au]
    best = None
    for w in want:
        th = wrap(w - phi)
        newfy = fy + th
        cost = abs(th) + 2.0 * max(0.0, abs(newfy - (-math.pi / 2)) - 2.2)
        if best is None or cost < best[0]:
            best = (cost, th, newfy)
    _, th, newfy = best
    log("rotate tool by", th, "finger yaw", newfy)
    # rotate gradually to keep IK happy
    nseg = max(1, int(math.ceil(abs(th) / 0.3)))
    for k in range(1, nseg + 1):
        R.orient(fy + th * k / nseg, iters=4)
    off2 = Rz(th) @ off
    # head half extent along push direction
    half = tool["half_w"] if mode == "side" else tool["tip"]
    res = push_cube(R, off2, zg + 0.004, target, cube, half, None, log=log)
    return res

def detect_hammer(P, rgb, log=print):
    """tan shaft + dark slate upright post. returns dict(grasp xyz, tip xyz, heading)."""
    tb = [b for b in blobs(mask_tan(rgb), min_px=10)]
    sb = [b for b in blobs(mask_slate(rgb), min_px=10)]
    best = None
    for n, u, v, pix in sb:
        pts = P[pix[:, 0], pix[:, 1]]; pts = pts[~np.isnan(pts).any(1)]
        if len(pts) < 8:
            continue
        hi = pts[pts[:, 2] > 0.05]
        if len(hi) < 5:
            continue
        post = np.median(hi, 0); top = float(np.percentile(hi[:, 2], 95))
        for n2, u2, v2, pix2 in tb:
            tp = P[pix2[:, 0], pix2[:, 1]]; tp = tp[~np.isnan(tp).any(1)]
            tp = tp[tp[:, 2] < 0.06]
            if len(tp) < 8:
                continue
            dd = np.linalg.norm(tp[:, :2] - post[:2], axis=1)
            if dd.min() > 0.08:
                continue
            tip = tp[np.argmax(dd)]
            L = dd.max()
            if L < 0.12:
                continue
            if best is None or n2 > best["n"]:
                ax = (tip[:2] - post[:2]) / L
                best = dict(grasp=post, top=top, tip=tip, L=L, heading=math.atan2(ax[1], ax[0]), n=n2,
                            shaft_z=float(np.median(tp[:, 2])))
    return best

def grasp_and_push(R, grasp_xy, zg, phi, contact_xy, half_along, cube, target, log=print, lift=0.07, finger_along=False, push_heading=None, obstacles=None, deltas=(0.0,)):
    """generic: grasp tool at grasp_xy/zg with fingers perpendicular to shaft heading phi,
    rotate tool so shaft points along push dir (tip first), push with contact point."""
    fy = phi if finger_along else phi + math.pi / 2
    u = target[:2] - cube[:2]; au = math.atan2(u[1], u[0])
    if push_heading is None:
        push_heading = choose_heading(au, u / np.linalg.norm(u), cube, np.linalg.norm(np.asarray(contact_xy[:2]) - np.asarray(grasp_xy[:2])),
                                      half_along, obstacles, phi, log=log, deltas=deltas)
    th = wrap(push_heading - phi)
    fy = min([fy + k * math.pi for k in (-2, -1, 0, 1, 2)],
             key=lambda a: max(abs(a + math.pi / 2), abs(a + th + math.pi / 2)))
    log("fy", fy, "th", th)
    R.grip = -1
    R.hold_yaw = -math.pi / 2
    R.move_to([R.state()[0], R.state()[1], max(R.state()[2], zg + 0.12)], tol=0.01)
    R.move_to([grasp_xy[0], grasp_xy[1], zg + 0.12], tol=0.006)
    R.orient(fy)
    R.move_to([grasp_xy[0], grasp_xy[1], zg + 0.04], tol=0.004)
    R.move_to([grasp_xy[0], grasp_xy[1], zg], tol=0.003, speed=0.4)
    R.gripper(1, 14)
    st = R.state()
    log("grasped", np.round(st, 4).tolist())
    R.tool_held = True
    if st[6] + st[7] < 0.003:
        log("grasp missed")
        return "miss"
    off = np.asarray(contact_xy[:2]) - st[:2]
    R.move_to([st[0], st[1], zg + lift], tol=0.005, speed=0.5)
    st2 = R.state()
    if st2[6] + st2[7] < 0.003:
        log("tool slipped on lift")
        return "miss"
    log("rotate tool by", th)
    nseg = max(1, int(math.ceil(abs(th) / 0.25)))
    for k in range(1, nseg + 1):
        R.orient(fy + th * k / nseg, iters=4)
    got = wrap(yaw_of(R.state()) - fy)
    off2 = Rz(got) @ off
    log("rot got", got, "off2", off2.round(3).tolist())
    res = push_cube(R, off2, zg + 0.003, target, cube, half_along, None, log=log)
    e = np.array([math.cos(yaw_of(R.state()) - fy + phi), math.sin(yaw_of(R.state()) - fy + phi)])
    for k in range(3):
        if R.ended or R.complete or res is None or R.remaining() < 120:
            break
        c = view_cube(R, res, off2, obstacles, log)
        if c is None:
            log("correction: cube not visible"); break
        err = target[:2] - c[:2]; ne = np.linalg.norm(err)
        log("correction", k, "cube", np.round(c, 3).tolist(), "err", round(float(ne), 4))
        if ne < 0.008:
            break
        u2 = err / ne
        if np.dot(e, u2) < -0.3:
            log("correction dir blocked by shaft"); break
        res = push_cube(R, off2, zg + 0.003, target, c, half_along, None, log=log, stop_tol=0.004)
    return res

def view_cube(R, approx, off2, obstacles, log=print):
    c = find_cube(R, approx=np.r_[approx[:2], 0], maxjump=0.08)
    if c is not None:
        return c
    p0 = R.state()[:3].copy()
    for dx, dy in ((0, -0.12), (0, 0.12), (-0.12, 0), (-0.12, -0.12), (-0.12, 0.12), (0, -0.22), (0, 0.22)):
        q = p0[:2] + np.array([dx, dy])
        if not reach_ok(q) or np.linalg.norm(q - BASE) > 0.75:
            continue
        pts = [q, q + off2, q + 0.5 * off2]
        if obstacles is not None and len(obstacles) and min(np.min(np.linalg.norm(obstacles - x, axis=1)) for x in pts) < 0.05:
            continue
        R.move_to([q[0], q[1], p0[2]], tol=0.008, max_steps=40, speed=0.6)
        c = find_cube(R, approx=np.r_[approx[:2], 0], maxjump=0.08)
        log("view try", [dx, dy], "seen", c is not None)
        if c is not None:
            return c
    return None

def tool_type(t):
    if t["L"] < 0.17:
        return "scoop"
    if t["head"][2] > 0.016:
        return "ballstick"
    if t["half_w"] > 0.025 and t["head_n"] > 120:
        return "scoop"
    return "hook"

def classify_tools(frames, log=print):
    """which teal tool moves in the conditioning video. returns (type, info)."""
    import json as _j
    vid = [f for f in frames if f.get("phase") == "video_demo"]
    if len(vid) < 5:
        return None, {}
    def tools_at(f):
        A = load_frame(f["files"])
        P = world_points(f, A, "front")
        ts = detect_tools(P, A[("front", "rgb")])
        return [(tool_type(t), t["handle"][:2]) for t in ts]
    t0 = tools_at(vid[0])
    if not t0:
        return None, {"t0": []}
    idx = list(range(len(vid) // 3, len(vid), max(1, len(vid) // 25)))
    miss = [0] * len(t0); n = 0
    for i in idx:
        cur = tools_at(vid[i]); n += 1
        for k, (ty, h) in enumerate(t0):
            if not any(ty2 == ty and np.linalg.norm(h2 - h) < 0.03 for ty2, h2 in cur):
                miss[k] += 1
    fr = [m / max(n, 1) for m in miss]
    info = {"t0": [(ty, np.round(h, 3).tolist()) for ty, h in t0], "miss": fr}
    k = int(np.argmax(fr))
    if fr[k] >= 0.5:
        return t0[k][0], info
    return None, info


def shaft_axis(P, rgb, tool):
    """refine shaft direction with gray pixels between handle and head."""
    gm = mask_gray(rgb)
    pts = P[gm]; pts = pts[~np.isnan(pts).any(1)]
    h = tool["handle"][:2]; c = tool["head"][:2]
    ax = (c - h) / np.linalg.norm(c - h); n = np.array([-ax[1], ax[0]])
    rel = pts[:, :2] - h
    al = rel @ ax; pe = rel @ n
    sel = (al > 0.03) & (al < tool["L"] - 0.02) & (np.abs(pe) < 0.03) & (pts[:, 2] < 0.05)
    if sel.sum() < 10:
        return ax
    q = pts[sel, :2]
    q0 = q.mean(0)
    w, V = np.linalg.eigh(np.cov((q - q0).T))
    d = V[:, -1]
    if d @ ax < 0:
        d = -d
    return d

def hook_geometry(P, rgb, tool):
    s = shaft_axis(P, rgb, tool)
    n = np.array([-s[1], s[0]])
    h = tool["handle"][:2]
    bl = blobs(mask_salmon(rgb), min_px=5)
    pts = P[mask_salmon(rgb)]; pts = pts[~np.isnan(pts).any(1)]
    rel = pts[:, :2] - h
    al = rel @ s; pe = rel @ n
    sel = (al > tool["L"] - 0.08) & (al < tool["L"] + 0.08) & (np.abs(pe) < 0.09) & (pts[:, 2] < 0.04)
    al, pe = al[sel], pe[sel]
    if len(al) < 5:
        return None
    side = 1.0 if np.mean(pe) > 0 else -1.0
    la = float(np.percentile(np.abs(pe), 95))
    amax = float(np.percentile(al, 95))
    # arm points: largest |perp|; their mean 'along' gives arm line position
    arm = np.abs(pe) > 0.5 * la
    arm_al = float(np.median(al[arm])) if arm.sum() > 2 else amax
    return dict(s=s, n=n, side=side, la=la, arm_al=arm_al, amax=amax)


BASE = np.array([-0.615, 0.0])
RMIN = 0.29

def reach_ok(p):
    r = np.linalg.norm(np.asarray(p[:2]) - BASE)
    return RMIN <= r <= 0.82 and abs(p[1]) < 0.55

def choose_heading(au, u, cube, L, half_along, obstacles, phi, log=print, deltas=(0.0,)):
    """pick shaft heading (tip-first) near the push direction that keeps the shaft
    clear of obstacles and the gripper reachable over the whole push."""
    best = None
    dist = 0.2
    for d in deltas:
        h = au + d
        e = np.array([math.cos(h), math.sin(h)])
        tip0 = cube[:2] - u * (0.02 + half_along + 0.02)
        g0 = tip0 - e * L
        g1 = g0 + u * dist
        cost = abs(d) + 0.3 * abs(wrap(h - phi))
        if not (reach_ok(g0) and reach_ok(g1)):
            cost += 10
        if obstacles is not None and len(obstacles):
            for t in np.linspace(0.08, 0.95, 12):
                q = g0 + e * L * t
                for q2 in (q, q + u * dist):
                    if np.min(np.linalg.norm(obstacles - q2, axis=1)) < 0.02:
                        cost += 5
                        break
        if best is None or cost < best[0]:
            best = (cost, h)
    log("heading choice", best)
    return best[1]

def obstacle_points(P, rgb):
    bl = blobs(mask_blue(rgb), min_px=300)
    if not bl:
        return np.zeros((0, 2))
    pix = bl[0][3]
    pts = P[pix[:, 0], pix[:, 1]]; pts = pts[~np.isnan(pts).any(1)]
    return pts[::3, :2]
