# PatternLock v01 — replay the conditioning-video stick path, mapped onto the live target lattice.
# Public APIs only. Everything task-dependent is derived at runtime from the conditioning video
# (RGB-D + calibration + robot_state per frame) and the live observation.
import math

result = {"version": "v02"}
diag = {}
result["diag"] = diag

# ---------------------------------------------------------------- generic helpers
def qmat(q):
    w = float(q[0]); x = float(q[1]); y = float(q[2]); z = float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w = w / n; x = x / n; y = y / n; z = z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def mv(R, v):
    return [R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
            R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
            R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]]


def mtv(R, v):
    return [R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2],
            R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2],
            R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2]]


def r4(v):
    return [round(float(a), 4) for a in v]


def median(vals):
    s = sorted(vals)
    if len(s) == 0:
        return None
    return s[len(s) // 2]


def cam_basic(cam):
    K = cam.intrinsics
    pose = cam.camera_pose
    return {"fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]), "cy": float(K[1][2]),
            "R": qmat(pose.quaternion_wxyz), "t": [float(a) for a in pose.position]}


def cam_dir(m, conv, ray, u, v):
    a = (u - m["cx"]) / m["fx"]
    b = (v - m["cy"]) / m["fy"]
    s = 1.0
    if ray:
        s = 1.0 / math.sqrt(a * a + b * b + 1.0)
    xc = a * s; yc = b * s; zc = s
    if conv == "gl":
        c = [xc, -yc, -zc]
    elif conv == "ros":
        c = [zc, -xc, -yc]
    else:
        c = [xc, yc, zc]
    return mv(m["R"], c)


def world_to_pix(m, conv, p):
    d = [p[0] - m["t"][0], p[1] - m["t"][1], p[2] - m["t"][2]]
    c = mtv(m["R"], d)
    if conv == "gl":
        xc = c[0]; yc = -c[1]; zc = -c[2]
    elif conv == "ros":
        xc = -c[1]; yc = -c[2]; zc = c[0]
    else:
        xc = c[0]; yc = c[1]; zc = c[2]
    if zc <= 1e-6:
        return None
    return [m["fx"] * xc / zc + m["cx"], m["fy"] * yc / zc + m["cy"]]


def choose_conv(cam, m):
    depth = cam.depth_m
    H = int(depth.shape[0]); W = int(depth.shape[1])
    best = None
    scores = {}
    for conv in ("cv", "gl", "ros"):
        for ray in (False, True):
            zs = []
            for v in range(4, H, 12):
                for u in range(4, W, 12):
                    d = float(depth[v][u])
                    if not (d > 0.05 and d < 5.0):
                        continue
                    dv = cam_dir(m, conv, ray, u, v)
                    zs.append(m["t"][2] + d * dv[2])
            hist = {}
            for z in zs:
                k = int(math.floor(z / 0.004))
                hist[k] = hist.get(k, 0) + 1
                hist[k + 1] = hist.get(k + 1, 0) + 1
            top = 0
            topk = 0
            for k in hist:
                if hist[k] > top:
                    top = hist[k]
                    topk = k
            score = top / max(1, len(zs))
            if topk * 0.004 > m["t"][2] - 0.05:
                score = score * 0.01
            scores[conv + ("_ray" if ray else "_z")] = round(score, 3)
            if best is None or score > best[0]:
                best = (score, conv, ray)
    return best[1], best[2], scores


def red_px(px):
    r = int(px[0]); g = int(px[1]); b = int(px[2])
    return r > 120 and g < 60 and b < 60 and r > 2.2 * g


def detect_targets(cam, m, conv, ray, label):
    depth = cam.depth_m.tolist()
    H = len(depth); W = len(depth[0])
    pts = []
    hist = {}
    for v in range(H):
        row = depth[v]
        for u in range(W):
            d = row[u]
            if not (d > 0.05 and d < 5.0):
                continue
            dv = cam_dir(m, conv, ray, u, v)
            p = [m["t"][0] + d * dv[0], m["t"][1] + d * dv[1], m["t"][2] + d * dv[2]]
            pts.append((p[0], p[1], p[2], u, v))
            k = int(math.floor(p[2] / 0.002))
            hist[k] = hist.get(k, 0) + 1
    tk = None
    for k in hist:
        if tk is None or hist[k] > hist[tk]:
            tk = k
    tz0 = (tk + 0.5) * 0.002
    near = [p[2] for p in pts if abs(p[2] - tz0) < 0.003]
    table_z = sum(near) / max(1, len(near))
    elev = [p for p in pts if p[2] > table_z + 0.003 and p[2] < table_z + 0.08]
    # height histogram above table (diagnostic)
    hh = {}
    for p in elev:
        k = int(math.floor((p[2] - table_z) / 0.002))
        hh[k] = hh.get(k, 0) + 1
    # connected components of elevated pixels in IMAGE space (8-connectivity)
    pixmap = {}
    for idx in range(len(elev)):
        pixmap[(elev[idx][3], elev[idx][4])] = idx
    seen = set()
    comps = []
    for key in pixmap:
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        members = []
        while len(stack) > 0:
            k = stack.pop()
            members.append(pixmap[k])
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    nk = (k[0] + du, k[1] + dv)
                    if nk in pixmap and nk not in seen:
                        seen.add(nk)
                        stack.append(nk)
        comps.append(members)
    blobs = []
    rejected = 0
    for members in comps:
        if len(members) < 3:
            rejected += 1
            continue
        xs = [elev[i][0] for i in members]
        ys = [elev[i][1] for i in members]
        zs = sorted([elev[i][2] for i in members])
        ext = max(max(xs) - min(xs), max(ys) - min(ys))
        if ext > 0.06:
            rejected += 1
            continue
        top = zs[max(0, len(zs) - 2)]
        topp = [elev[i] for i in members if elev[i][2] >= top - 0.003]
        cx = sum([p[0] for p in topp]) / len(topp)
        cy = sum([p[1] for p in topp]) / len(topp)
        pix = [(elev[i][3], elev[i][4]) for i in members]
        blobs.append({"x": cx, "y": cy, "top": top, "n": len(members), "ext": ext, "pix": pix})
    diag[label + "_table_z"] = round(table_z, 4)
    diag[label + "_n_valid"] = len(pts)
    diag[label + "_n_elev"] = len(elev)
    diag[label + "_height_hist_2mm"] = [[k, hh[k]] for k in sorted(hh)][:40]
    diag[label + "_n_blobs"] = len(blobs)
    diag[label + "_rejected"] = rejected
    return blobs, table_z


def solve3(A, b):
    # Gaussian elimination 3x3
    M = [[A[i][0], A[i][1], A[i][2], b[i]] for i in range(3)]
    for c in range(3):
        piv = c
        for r in range(c + 1, 3):
            if abs(M[r][c]) > abs(M[piv][c]):
                piv = r
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-12:
            return None
        for r in range(3):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(4):
                    M[r][k] -= f * M[c][k]
    return [M[i][3] / M[i][i] for i in range(3)]


def lattice_index(O, a, b, x, y):
    det = a[0] * b[1] - a[1] * b[0]
    dx = x - O[0]; dy = y - O[1]
    i = (dx * b[1] - dy * b[0]) / det
    j = (a[0] * dy - a[1] * dx) / det
    return i, j


def fit_lattice(blobs, label):
    n = len(blobs)
    if n < 4:
        return None
    nn = []
    for i in range(n):
        best = None
        for j in range(n):
            if i == j:
                continue
            d = math.hypot(blobs[j]["x"] - blobs[i]["x"], blobs[j]["y"] - blobs[i]["y"])
            if best is None or d < best:
                best = d
        nn.append(best)
    s0 = median(nn)
    ax = [0.0, 0.0]; bx = [0.0, 0.0]; na = 0; nb = 0
    for i in range(n):
        for j in range(i + 1, n):
            vx = blobs[j]["x"] - blobs[i]["x"]; vy = blobs[j]["y"] - blobs[i]["y"]
            L = math.hypot(vx, vy)
            if L < 0.75 * s0 or L > 1.3 * s0:
                continue
            if abs(vx) >= abs(vy):
                if vx < 0:
                    vx = -vx; vy = -vy
                ax[0] += vx; ax[1] += vy; na += 1
            else:
                if vy < 0:
                    vx = -vx; vy = -vy
                bx[0] += vx; bx[1] += vy; nb += 1
    if na == 0 or nb == 0:
        return None
    a = [ax[0] / na, ax[1] / na]
    b = [bx[0] / nb, bx[1] / nb]
    mx = sum([q["x"] for q in blobs]) / n
    my = sum([q["y"] for q in blobs]) / n
    ref = None
    for q in blobs:
        d = math.hypot(q["x"] - mx, q["y"] - my)
        if ref is None or d < ref[0]:
            ref = (d, q)
    O = [ref[1]["x"], ref[1]["y"]]
    inl = []
    for it in range(3):
        cand = {}
        for q in blobs:
            fi, fj = lattice_index(O, a, b, q["x"], q["y"])
            ri = int(round(fi)); rj = int(round(fj))
            res = math.hypot(fi - ri, fj - rj)
            if res < 0.3:
                key = (ri, rj)
                if key not in cand or res < cand[key][0]:
                    cand[key] = (res, q)
        inl = [(k[0], k[1], cand[k][1]) for k in cand]
        if len(inl) < 4:
            return None
        # least squares x = Ox + i ax + j bx ; same for y
        A = [[0.0] * 3 for r in range(3)]
        bxv = [0.0] * 3; byv = [0.0] * 3
        for e in inl:
            reg = [1.0, float(e[0]), float(e[1])]
            for r in range(3):
                for c in range(3):
                    A[r][c] += reg[r] * reg[c]
                bxv[r] += reg[r] * e[2]["x"]
                byv[r] += reg[r] * e[2]["y"]
        sx = solve3(A, bxv); sy = solve3(A, byv)
        if sx is None or sy is None:
            break
        O = [sx[0], sy[0]]; a = [sx[1], sy[1]]; b = [sx[2], sy[2]]
    # prune isolated extreme rows/cols
    changed = True
    while changed and len(inl) > 4:
        changed = False
        ci = {}; cj = {}
        for e in inl:
            ci[e[0]] = ci.get(e[0], 0) + 1
            cj[e[1]] = cj.get(e[1], 0) + 1
        mi = max([ci[k] for k in ci]); mj = max([cj[k] for k in cj])
        lo_i = min([k for k in ci]); hi_i = max([k for k in ci])
        lo_j = min([k for k in cj]); hi_j = max([k for k in cj])
        drop = set()
        for k in (lo_i, hi_i):
            if ci[k] == 1 and mi >= 3:
                drop.add(("i", k))
        for k in (lo_j, hi_j):
            if cj[k] == 1 and mj >= 3:
                drop.add(("j", k))
        if len(drop) > 0:
            inl = [e for e in inl if ("i", e[0]) not in drop and ("j", e[1]) not in drop]
            changed = True
    mi0 = min([e[0] for e in inl]); mj0 = min([e[1] for e in inl])
    O = [O[0] + mi0 * a[0] + mj0 * b[0], O[1] + mi0 * a[1] + mj0 * b[1]]
    cells = {}
    for e in inl:
        cells[(e[0] - mi0, e[1] - mj0)] = e[2]
    ni = max([k[0] for k in cells]) + 1
    nj = max([k[1] for k in cells]) + 1
    tops = [cells[k]["top"] for k in cells]
    lat = {"O": O, "a": a, "b": b, "cells": cells, "ni": ni, "nj": nj, "top": median(tops),
           "spacing": math.sqrt((math.hypot(a[0], a[1]) ** 2 + math.hypot(b[0], b[1]) ** 2) / 2.0)}
    diag[label + "_lattice"] = {"O": r4(O), "a": r4(a), "b": r4(b), "ni": ni, "nj": nj,
                                "n_cells": len(cells), "top": round(lat["top"], 4),
                                "angle_a_deg": round(math.degrees(math.atan2(a[1], a[0])), 2),
                                "angle_b_deg": round(math.degrees(math.atan2(b[1], b[0])), 2)}
    diag[label + "_cells"] = [[k[0], k[1], round(cells[k]["x"], 4), round(cells[k]["y"], 4),
                               round(cells[k]["top"], 4), cells[k]["n"]] for k in sorted(cells)]
    return lat


def lattice_pos(lat, i, j):
    key = (i, j)
    if key in lat["cells"]:
        c = lat["cells"][key]
        return [c["x"], c["y"], c["top"]], True
    O = lat["O"]; a = lat["a"]; b = lat["b"]
    return [O[0] + i * a[0] + j * b[0], O[1] + i * a[1] + j * b[1], lat["top"]], False


def exec_info(r):
    info = {}
    try:
        info["ok"] = bool(r.ok)
    except Exception:
        info["ok"] = None
    try:
        info["status"] = str(r.status)
    except Exception:
        pass
    try:
        info["steps"] = int(r.steps_executed)
    except Exception:
        pass
    term = False
    try:
        term = bool(r.terminated) or bool(r.truncated)
    except Exception:
        pass
    info["terminal"] = term
    try:
        if r.error is not None:
            info["error"] = str(r.error)[:300]
    except Exception:
        pass
    return info


# ---------------------------------------------------------------- context
ctx = get_task_context()
result["goal"] = ctx.language
hz = 20.0
max_steps = None
try:
    md = robomme_plus.get_task_metadata()
    mdd = {}
    for k in md:
        mdd[str(k)] = str(md[k])[:400]
    diag["task_metadata"] = mdd
    try:
        max_steps = int(md["max_steps"])
    except Exception:
        max_steps = None
    try:
        hz = float(md["control_hz"])
    except Exception:
        pass
except Exception as e:
    diag["task_metadata_error"] = str(e)[:300]
try:
    cm = robomme_plus.get_controller_metadata()
    cmd = {}
    for k in cm:
        cmd[str(k)] = str(cm[k])[:400]
    diag["controller_metadata"] = cmd
except Exception as e:
    diag["controller_metadata_error"] = str(e)[:300]
result["max_steps"] = max_steps

state0 = get_robot_state()
base = state0.base_frame
ee0 = state0.end_effector_poses["primary"]
diag["live_ee0"] = r4(ee0.position) + r4(ee0.quaternion_wxyz)
diag["live_gripper0"] = float(state0.gripper_positions["primary"])
diag["live_t0"] = float(state0.timestamp_s)

# ---------------------------------------------------------------- conditioning video
video = list(robomme_plus.get_conditioning_video())
nv = len(video)
result["video_frames"] = nv
ee_traj = []
for fr in video:
    p = fr.robot_state.end_effector_poses["primary"]
    ee_traj.append((float(fr.timestamp_s), [float(a) for a in p.position], [float(a) for a in p.quaternion_wxyz]))
diag["video_ee_sub"] = [[round(e[0], 2)] + r4(e[1]) for e in ee_traj[::4]]
if nv > 0:
    diag["video_q0"] = r4(ee_traj[0][2])

# ---------------------------------------------------------------- video lattice
vcam = video[0].cameras["front"]
vm = cam_basic(vcam)
vconv, vray, vscores = choose_conv(vcam, vm)
diag["video_conv"] = [vconv, vray, vscores]
diag["video_cam"] = {"fx": vm["fx"], "fy": vm["fy"], "cx": vm["cx"], "cy": vm["cy"], "t": r4(vm["t"])}
vblobs, vtable = detect_targets(vcam, vm, vconv, vray, "video")
vlat = fit_lattice(vblobs, "video")
if vlat is None:
    result["error"] = "video lattice not found"

# ---------------------------------------------------------------- pattern from red onsets
events = []
if vlat is not None:
    keys = sorted(vlat["cells"])
    samp = {}
    for k in keys:
        pix = vlat["cells"][k]["pix"]
        stepn = max(1, len(pix) // 30)
        samp[k] = pix[::stepn]
    last_red = {}
    red_state = {}
    onset_log = []
    for fi in range(nv):
        rgb = video[fi].cameras["front"].rgb
        t = ee_traj[fi][0]
        for k in keys:
            cnt = 0
            for uv in samp[k]:
                if red_px(rgb[uv[1]][uv[0]]):
                    cnt += 1
            isred = cnt >= max(2, int(0.15 * len(samp[k])))
            if isred:
                prev = last_red.get(k)
                if prev is None or t - prev > 1.0:
                    onset_log.append((t, fi, k, cnt))
                last_red[k] = t
    onset_log = sorted(onset_log)
    for e in onset_log:
        events.append({"t": e[0], "frame": e[1], "ij": [e[2][0], e[2][1]], "red": e[3]})
diag["red_events"] = [[round(e["t"], 2), e["frame"], e["ij"][0], e["ij"][1], e["red"]] for e in events]

# backup: EE z minima near targets
if vlat is not None and len(events) < 2:
    zmin = min([e[1][2] for e in ee_traj])
    seq = []
    for fi in range(nv):
        p = ee_traj[fi][1]
        if p[2] > zmin + 0.005:
            continue
        fi_, fj_ = lattice_index(vlat["O"], vlat["a"], vlat["b"], p[0], p[1])
        ri = int(round(fi_)); rj = int(round(fj_))
        if math.hypot(fi_ - ri, fj_ - rj) < 0.3:
            if len(seq) == 0 or seq[len(seq) - 1]["ij"] != [ri, rj]:
                seq.append({"t": ee_traj[fi][0], "frame": fi, "ij": [ri, rj], "red": 0})
    diag["ee_fallback_events"] = [[round(e["t"], 2), e["ij"][0], e["ij"][1]] for e in seq]
    events = seq
result["pattern"] = [e["ij"] for e in events]

# ---------------------------------------------------------------- tool offset from video contacts
d_locals = []
contacts = []
for e in events:
    best = None
    for fi in range(nv):
        t = ee_traj[fi][0]
        if t < e["t"] - 0.6 or t > e["t"] + 0.6:
            continue
        if best is None or ee_traj[fi][1][2] < ee_traj[best][1][2]:
            best = fi
    if best is None:
        continue
    T, det = lattice_pos(vlat, e["ij"][0], e["ij"][1])
    p = ee_traj[best][1]
    R = qmat(ee_traj[best][2])
    dl = mtv(R, [T[0] - p[0], T[1] - p[1], T[2] - p[2]])
    d_locals.append(dl)
    contacts.append({"frame": best, "z": p[2], "q": ee_traj[best][2], "T": T, "p": p})
d_local = [0.0, 0.0, 0.0]
if len(d_locals) > 0:
    for c in range(3):
        d_local[c] = median([d[c] for d in d_locals])
diag["d_locals"] = [r4(d) for d in d_locals]
diag["d_local"] = r4(d_local)
diag["contacts"] = [[c["frame"], round(c["z"], 4), r4(c["T"]), r4(c["p"])] for c in contacts]
# transit clearance in the video
lifts = []
for k in range(len(contacts) - 1):
    f0 = contacts[k]["frame"]; f1 = contacts[k + 1]["frame"]
    if f1 <= f0 + 1:
        continue
    zmax = max([ee_traj[fi][1][2] for fi in range(f0, f1 + 1)])
    lifts.append(zmax - 0.5 * (contacts[k]["z"] + contacts[k + 1]["z"]))
lift_video = median(lifts)
diag["video_lifts"] = [round(v, 4) for v in lifts]
LIFT = 0.02
if lift_video is not None:
    LIFT = min(0.06, max(0.02, lift_video))
diag["LIFT"] = LIFT

# ---------------------------------------------------------------- live lattice
obs = get_observation()
lcam = obs.cameras["front"]
lm = cam_basic(lcam)
lconv, lray, lscores = choose_conv(lcam, lm)
diag["live_conv"] = [lconv, lray, lscores]
lblobs, ltable = detect_targets(lcam, lm, lconv, lray, "live")
llat = fit_lattice(lblobs, "live")
if llat is None:
    result["error"] = "live lattice not found"
if llat is not None and vlat is not None:
    diag["lattice_shape_match"] = [vlat["ni"], vlat["nj"], llat["ni"], llat["nj"]]

# ---------------------------------------------------------------- v01: replay the video's stick path
# The expert in the video drags the stick at target-top height and dwells ~0.5 s on each vertex;
# v00's hover/descend/lift touches did not register (no red) and the episode failed on approach
# to the 2nd vertex.  v01 reproduces the video's own motion (timing, approach, dwell, drag height)
# with every tip position mapped from the video lattice into the live lattice.
flags = {"terminal": False}
# v02: ArmCommand defaults to embodiment 'robosuite', which the adapter rejects (v01 ran 0 ticks).
EMB = "robomme_plus"
try:
    EMB = str(state0.embodiment)
except Exception:
    pass
diag["embodiment"] = EMB
stepinfo = {"bad": 0}


def do_step(js):
    try:
        cmd = ArmCommand("joint_position", js, gripper_position=0.0, embodiment=EMB)
    except TypeError:
        cmd = ArmCommand("joint_position", js, 0.0, EMB)
    sr = step(RobotAction({"primary": cmd}))
    okv = True
    try:
        okv = bool(sr.ok)
    except Exception:
        okv = True
    if not okv:
        stepinfo["bad"] += 1
        if stepinfo["bad"] <= 2:
            try:
                diag["step_err_" + str(stepinfo["bad"])] = str(sr.error)[:300]
            except Exception:
                pass
        if stepinfo["bad"] >= 3:
            flags["terminal"] = True
            diag["step_abort"] = True
            return False
    try:
        if bool(sr.terminated) or bool(sr.truncated):
            flags["terminal"] = True
            return False
    except Exception:
        pass
    return True


def ticks_now():
    try:
        return int(round(get_robot_state().timestamp_s * hz))
    except Exception:
        return -1


def map_xy(x, y, w):
    fi, fj = lattice_index(vlat["O"], vlat["a"], vlat["b"], x, y)
    O = llat["O"]; a = llat["a"]; b = llat["b"]
    mx = O[0] + fi * a[0] + fj * b[0]
    my = O[1] + fi * a[1] + fj * b[1]
    return [x + w * (mx - x), y + w * (my - y)]


def ik_joints(r):
    try:
        return [float(v) for v in r.joint_positions]
    except Exception:
        pass
    try:
        return [float(v) for v in r.joint_positions["primary"]]
    except Exception:
        return None


ok_inputs = llat is not None and vlat is not None and nv > 1
if ok_inputs:
    t_first = None
    if len(events) > 0:
        t_first = events[0]["t"]
    else:
        zmin = min([e[1][2] for e in ee_traj])
        for e in ee_traj:
            if e[1][2] < zmin + 0.003:
                t_first = e[0]
                break
    t_blend = max(0.2, t_first - 1.0)
    dz = llat["top"] - vlat["top"]
    diag["t_first"] = round(t_first, 3)
    diag["t_blend"] = round(t_blend, 3)
    diag["dz"] = round(dz, 4)
    # IK for every video frame
    seed = [float(v) for v in state0.joint_positions["primary"]]
    sol = []
    nfail = 0
    targets_dbg = []
    for i in range(nv):
        t, p, q = ee_traj[i]
        s = min(1.0, t / t_blend)
        w = s * s * (3.0 - 2.0 * s)
        R = qmat(q)
        off = mv(R, d_local)
        tip = [p[0] + off[0], p[1] + off[1], p[2] + off[2]]
        mxy = map_xy(tip[0], tip[1], w)
        tgt = [mxy[0] - off[0], mxy[1] - off[1], p[2] + w * dz]
        r = None
        try:
            r = solve_ik(Pose(tgt, q, base), seed_joints=seed)
        except Exception as ex:
            if nfail < 3:
                diag["ik_exc_" + str(nfail)] = str(ex)[:200]
            nfail += 1
            sol.append(None)
            continue
        js = None
        okr = False
        try:
            okr = bool(r.ok)
        except Exception:
            okr = False
        if okr:
            js = ik_joints(r)
        if js is None:
            nfail += 1
            sol.append(None)
            if nfail <= 3:
                try:
                    diag["ik_fail_" + str(nfail)] = str(r.error)[:200]
                except Exception:
                    pass
            continue
        seed = js
        sol.append(js)
        if i % 16 == 0:
            targets_dbg.append([round(t, 2)] + r4(tgt))
    diag["ik_fail"] = nfail
    diag["replay_targets_sub"] = targets_dbg
    # fill gaps
    good = [i for i in range(nv) if sol[i] is not None]
    if len(good) == 0:
        result["error"] = "no IK solutions"
        ok_inputs = False

if ok_inputs:
    for i in range(nv):
        if sol[i] is None:
            prev = None; nxt = None
            for g in good:
                if g < i:
                    prev = g
                if g > i and nxt is None:
                    nxt = g
            if prev is None:
                sol[i] = sol[nxt]
            elif nxt is None:
                sol[i] = sol[prev]
            else:
                f = (i - prev) / float(nxt - prev)
                sol[i] = [sol[prev][k] + f * (sol[nxt][k] - sol[prev][k]) for k in range(7)]
    times = [e[0] for e in ee_traj]
    T_end = times[nv - 1]
    HOLD = 30
    t_a = max(0.0, t_first - 1.5)
    cA = 2.0
    ticks_A = int(math.ceil(t_a * hz / cA))
    budget = 433
    if max_steps is not None:
        budget = max_steps
    avail = budget - ticks_A - HOLD - 15
    need_B = (T_end - t_a) * hz
    cB = 1.0
    if avail > 10 and need_B > avail:
        cB = need_B / avail
    ticks_B = int(math.ceil(need_B / cB))
    diag["schedule"] = {"T_end": round(T_end, 3), "t_a": round(t_a, 3), "ticks_A": ticks_A, "cA": cA,
                        "ticks_B": ticks_B, "cB": round(cB, 3), "budget": budget}
    sched = []
    for k in range(1, ticks_A + 1):
        sched.append(k * cA / hz)
    for k in range(1, ticks_B + 1):
        sched.append(min(T_end, t_a + k * cB / hz))
    jidx = 0
    nstep = 0
    for tau in sched:
        while jidx < nv - 2 and times[jidx + 1] < tau:
            jidx += 1
        t0 = times[jidx]; t1 = times[jidx + 1]
        f = 0.0
        if t1 > t0:
            f = min(1.0, max(0.0, (tau - t0) / (t1 - t0)))
        js = [sol[jidx][k] + f * (sol[jidx + 1][k] - sol[jidx][k]) for k in range(7)]
        try:
            cont = do_step(js)
        except Exception as ex:
            diag["step_exc"] = str(ex)[:200]
            flags["terminal"] = True
            break
        nstep += 1
        if not cont:
            break
    diag["replay_steps"] = nstep
    # hold the final pose so the last vertex gets its dwell
    if not flags["terminal"]:
        last = sol[nv - 1]
        for h in range(HOLD):
            try:
                cont = do_step(last)
            except Exception as ex:
                diag["hold_exc"] = str(ex)[:200]
                flags["terminal"] = True
                break
            nstep += 1
            if not cont:
                break
    diag["total_steps"] = nstep
    diag["bad_steps"] = stepinfo["bad"]
    # final diagnostics (guarded)
    if not flags["terminal"]:
        try:
            ob = get_observation()
            rgb = ob.cameras["front"].rgb
            reds = []
            for k in sorted(llat["cells"]):
                cnt = 0
                for uv in llat["cells"][k]["pix"]:
                    if red_px(rgb[uv[1]][uv[0]]):
                        cnt += 1
                if cnt > 0:
                    reds.append([k[0], k[1], cnt])
            diag["final_front_red"] = reds
            diag["final_ee"] = r4(ob.robot_state.end_effector_poses["primary"].position)
        except Exception as ex:
            diag["final_exc"] = str(ex)[:200]
    result["stage"] = "done"

result["terminal"] = flags["terminal"]
result["final_tick"] = ticks_now()
