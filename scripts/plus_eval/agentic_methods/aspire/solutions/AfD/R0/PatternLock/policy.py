# PatternLock v03 -- calibration: fix improper (mirror) similarity fit; the front camera faces the robot,
# so image->base is expected to be a reflection (AfD, untested in simulator).
#
# Strategy (all episode-dependent quantities derived at runtime):
#   1. Conditioning video, first frame, front camera: detect the button grid (whitish blobs),
#      index it as a lattice (BFS growth, axes nearest image x / y) and fit a homography
#      image -> video-lattice coordinates.
#   2. Every conditioning-video frame: count red pixels in a window at each button; a button's
#      red ONSET (not red during the previous second) is a touch -> ordered pattern.
#   3. The video's own robot_state gives the end-effector pose at each touch; a 2-D similarity
#      from video-lattice coords to robot-base xy is fitted from those touches (absorbs the
#      stick-tip offset and the camera geometry; no depth or camera convention needed).
#   4. Live front image: detect buttons, map through the same homography (front camera is
#      static), fit a rotated square lattice, index with the minimal rotation.
#   5. Touch the pattern's cells in the live grid in order, at the video's touch height and
#      orientation, with the video's inter-touch lift.
import math

goal = get_task_context().language
meta = robomme_plus.get_task_metadata()
try:
    MAX_STEPS = int(meta["max_steps"])
except Exception:
    MAX_STEPS = -1
try:
    HZ = float(meta["control_hz"])
except Exception:
    HZ = 20.0

diag = {"goal": goal, "max_steps": MAX_STEPS}


def to_rows(arr):
    try:
        return arr.tolist()
    except Exception:
        out = []
        for y in range(len(arr)):
            row = arr[y]
            out.append([[int(row[x][0]), int(row[x][1]), int(row[x][2])] for x in range(len(row))])
        return out


def white_blobs(rows):
    H = len(rows)
    W = len(rows[0])
    mask = [[False] * W for yy in range(H)]
    for y in range(H):
        row = rows[y]
        mrow = mask[y]
        for x in range(W):
            p = row[x]
            r = int(p[0]); g = int(p[1]); b = int(p[2])
            lo = min(r, g, b)
            if lo > 150 and max(r, g, b) - lo < 40:
                mrow[x] = True
    amin = max(4, int(0.00015 * W * H))
    amax = int(0.0046 * W * H)
    seen = [[False] * W for yy in range(H)]
    blobs = []
    for y in range(H):
        for x in range(W):
            if not mask[y][x] or seen[y][x]:
                continue
            stack = [(y, x)]
            seen[y][x] = True
            n = 0; sx = 0.0; sy = 0.0
            while stack:
                cy, cx = stack.pop()
                n += 1; sx += cx; sy += cy
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny = cy + dy; nx = cx + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            if amin <= n <= amax:
                blobs.append([sx / n, sy / n, n])
    return blobs


def argbest(items, scores, highest):
    best = None
    bs = None
    for it, sc in zip(items, scores):
        if bs is None or (sc > bs if highest else sc < bs):
            best = it
            bs = sc
    return best


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def prune_isolated(blobs):
    if len(blobs) < 5:
        return blobs
    nn = []
    for i in range(len(blobs)):
        nn.append(min(dist(blobs[i], blobs[j]) for j in range(len(blobs)) if j != i))
    med = sorted(nn)[len(nn) // 2]
    return [blobs[i] for i in range(len(blobs)) if nn[i] <= 1.6 * med]


def solve_linear(A, b):
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        piv = argbest(list(range(c, n)), [abs(M[i][c]) for i in range(c, n)], True)
        if abs(M[piv][c]) < 1e-12:
            raise ValueError("singular system")
        M[c], M[piv] = M[piv], M[c]
        for i in range(n):
            if i != c:
                f = M[i][c] / M[c][c]
                if f != 0.0:
                    for k in range(c, n + 1):
                        M[i][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def least_squares(rows_a, vals):
    n = len(rows_a[0])
    AtA = [[0.0] * n for i in range(n)]
    Atb = [0.0] * n
    for a, v in zip(rows_a, vals):
        for i in range(n):
            Atb[i] += a[i] * v
            for j in range(n):
                AtA[i][j] += a[i] * a[j]
    return solve_linear(AtA, Atb)


def index_lattice_image(blobs):
    """BFS lattice growth in image space. Returns dict (col,row)->[x,y,n]; col axis is the
    neighbour direction nearest image +x, row axis nearest image +y."""
    pts = [b for b in blobs]
    xs = sorted(p[0] for p in pts)
    ys = sorted(p[1] for p in pts)
    cx = xs[len(xs) // 2]
    cy = ys[len(ys) // 2]
    # v02: seed = an interior lattice point (4 near-equal nearest neighbours) closest to the
    # coordinate-wise median; non-button blobs (robot body) neither attract the median nor
    # have a square neighbourhood.
    scores = []
    for i in range(len(pts)):
        dd = sorted(dist(pts[i], pts[j]) for j in range(len(pts)) if j != i)
        sq = dd[3] / (dd[0] + 1e-9) if len(dd) >= 4 else 99.0
        pen = 0.0 if sq < 1.35 else 1e6
        scores.append(pen + (pts[i][0] - cx) ** 2 + (pts[i][1] - cy) ** 2)
    seed = argbest(list(range(len(pts))), scores, False)
    others = sorted([[dist(pts[j], pts[seed]), j] for j in range(len(pts)) if j != seed])
    near = [o[1] for o in others[:4]]
    vecs = [[pts[j][0] - pts[seed][0], pts[j][1] - pts[seed][1]] for j in near]
    a = argbest(vecs, [v[0] / (math.sqrt(v[0] ** 2 + v[1] ** 2) + 1e-9) for v in vecs], True)
    b = argbest(vecs, [v[1] / (math.sqrt(v[0] ** 2 + v[1] ** 2) + 1e-9) for v in vecs], True)
    idx = {seed: (0, 0)}
    cell_of = {(0, 0): seed}
    frontier = [seed]
    while frontier:
        i = frontier.pop(0)
        ci, ri = idx[i]
        # local basis from already-indexed neighbours when available
        la = a; lb = b
        if (ci + 1, ri) in cell_of:
            k = cell_of[(ci + 1, ri)]; la = [pts[k][0] - pts[i][0], pts[k][1] - pts[i][1]]
        elif (ci - 1, ri) in cell_of:
            k = cell_of[(ci - 1, ri)]; la = [pts[i][0] - pts[k][0], pts[i][1] - pts[k][1]]
        if (ci, ri + 1) in cell_of:
            k = cell_of[(ci, ri + 1)]; lb = [pts[k][0] - pts[i][0], pts[k][1] - pts[i][1]]
        elif (ci, ri - 1) in cell_of:
            k = cell_of[(ci, ri - 1)]; lb = [pts[i][0] - pts[k][0], pts[i][1] - pts[k][1]]
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cell = (ci + dc, ri + dr)
            if cell in cell_of:
                continue
            px = pts[i][0] + dc * la[0] + dr * lb[0]
            py = pts[i][1] + dc * la[1] + dr * lb[1]
            tol = 0.35 * min(math.sqrt(la[0] ** 2 + la[1] ** 2), math.sqrt(lb[0] ** 2 + lb[1] ** 2))
            best = None; bd = None
            for j in range(len(pts)):
                if j in idx:
                    continue
                d = math.sqrt((pts[j][0] - px) ** 2 + (pts[j][1] - py) ** 2)
                if d < tol and (bd is None or d < bd):
                    best = j; bd = d
            if best is not None:
                idx[best] = cell
                cell_of[cell] = best
                frontier.append(best)
    c0 = min(c for c, r in cell_of)
    r0 = min(r for c, r in cell_of)
    out = {}
    for cell in cell_of:
        out[(cell[0] - c0, cell[1] - r0)] = pts[cell_of[cell]]
    return out


def fit_homography(src, dst):
    # src: image px, dst: plane; h33 = 1; normalised pixel coordinates
    s = 1.0 / 256.0
    A = []; v = []
    for (x, y), (u, w) in zip(src, dst):
        x = x * s; y = y * s
        A.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y]); v.append(u)
        A.append([0.0, 0.0, 0.0, x, y, 1.0, -w * x, -w * y]); v.append(w)
    h = least_squares(A, v)
    return [h[0] * s, h[1] * s, h[2], h[3] * s, h[4] * s, h[5], h[6] * s, h[7] * s, 1.0]


def apply_h(h, x, y):
    d = h[6] * x + h[7] * y + h[8]
    return [(h[0] * x + h[1] * y + h[2]) / d, (h[3] * x + h[4] * y + h[5]) / d]


def invert_h(h):
    a, b, c, d, e, f, g, k, i = h
    A = e * i - f * k; B = -(d * i - f * g); C = d * k - e * g
    D = -(b * i - c * k); E = a * i - c * g; F = -(a * k - b * g)
    G = b * f - c * e; Hh = -(a * f - c * d); I = a * e - b * d
    return [A, D, G, B, E, Hh, C, F, I]


def fit_plane_lattice(q):
    """q: list of plane points (video-lattice units). Returns (T, M, idx list) with
    q ~ T + M * idx, M = s * R(theta), theta in (-45, 45] deg."""
    n = len(q)
    s4 = 0.0; c4 = 0.0; nnd = []
    for i in range(n):
        ks = [k for k in range(n) if k != i]
        j = argbest(ks, [dist(q[k], q[i]) for k in ks], False)
        vx = q[j][0] - q[i][0]; vy = q[j][1] - q[i][1]
        ang = math.atan2(vy, vx)
        s4 += math.sin(4 * ang); c4 += math.cos(4 * ang)
        nnd.append(math.sqrt(vx * vx + vy * vy))
    th = math.atan2(s4, c4) / 4.0
    sp = sorted(nnd)[n // 2]
    ct = math.cos(th); st = math.sin(th)
    w = [[(ct * p[0] + st * p[1]) / sp, (-st * p[0] + ct * p[1]) / sp] for p in q]
    offs = []
    for ax in range(2):
        ss = sum(math.sin(2 * math.pi * p[ax]) for p in w)
        cc = sum(math.cos(2 * math.pi * p[ax]) for p in w)
        offs.append(math.atan2(ss, cc) / (2 * math.pi))
    idx = [[int(round(p[0] - offs[0])), int(round(p[1] - offs[1]))] for p in w]
    m0 = min(i[0] for i in idx); m1 = min(i[1] for i in idx)
    idx = [[i[0] - m0, i[1] - m1] for i in idx]
    # refine similarity q = T + [a -b; b a] idx
    A = []; v = []
    for p, i in zip(q, idx):
        A.append([i[0], -i[1], 1.0, 0.0]); v.append(p[0])
        A.append([i[1], i[0], 0.0, 1.0]); v.append(p[1])
    a, b, tx, ty = least_squares(A, v)
    resid = max(math.sqrt((tx + a * i[0] - b * i[1] - p[0]) ** 2 + (ty + b * i[0] + a * i[1] - p[1]) ** 2)
                for p, i in zip(q, idx))
    return {"a": a, "b": b, "tx": tx, "ty": ty, "idx": idx, "theta_deg": math.degrees(math.atan2(b, a)),
            "spacing": math.sqrt(a * a + b * b), "resid": resid}


def fit_similarity(cells, xy, proper):
    A = []; v = []
    for c, p in zip(cells, xy):
        if proper:
            A.append([c[0], -c[1], 1.0, 0.0]); v.append(p[0])
            A.append([c[1], c[0], 0.0, 1.0]); v.append(p[1])
        else:
            # model x = tx + a*c + b*r ; y = ty + b*c - a*r  (v03: y-row coefficients fixed)
            A.append([c[0], c[1], 1.0, 0.0]); v.append(p[0])
            A.append([-c[1], c[0], 0.0, 1.0]); v.append(p[1])
    a, b, tx, ty = least_squares(A, v)
    def f(c):
        if proper:
            return [tx + a * c[0] - b * c[1], ty + b * c[0] + a * c[1]]
        return [tx + a * c[0] + b * c[1], ty + b * c[0] - a * c[1]]
    res = max(dist(f(c), p) for c, p in zip(cells, xy))
    return {"f": f, "resid": res, "scale": math.sqrt(a * a + b * b)}


def is_red(p):
    r = int(p[0]); g = int(p[1]); b = int(p[2])
    return r > 100 and g < 0.3 * r and b < 0.3 * r


def is_terminal(res):
    for name in ("terminated", "truncated", "done", "terminal"):
        try:
            if name == "terminated" and bool(res.terminated):
                return True
            if name == "truncated" and bool(res.truncated):
                return True
            if name == "done" and bool(res.done):
                return True
            if name == "terminal" and bool(res.terminal):
                return True
        except Exception:
            pass
    return False


result = {"ok": False, "diag": diag}

video = robomme_plus.get_conditioning_video()
diag["video_frames"] = len(video)
if len(video) == 0:
    raise ValueError("PatternLock needs the conditioning video; none supplied")

# ---- 1. video grid -----------------------------------------------------------
vgrid = None
for vi in range(min(5, len(video))):
    vb = prune_isolated(white_blobs(to_rows(video[vi].cameras["front"].rgb)))
    if len(vb) >= 9:
        cand = index_lattice_image(vb)
        if vgrid is None or len(cand) > len(vgrid):
            vgrid = cand
    if vgrid is not None:
        vc = max(c for c, r in vgrid) + 1
        vr = max(r for c, r in vgrid) + 1
        if len(vgrid) == vc * vr:
            break
if vgrid is None or len(vgrid) < 9:
    raise ValueError("could not detect the button grid in the conditioning video")
ncol = max(c for c, r in vgrid) + 1
nrow = max(r for c, r in vgrid) + 1
diag["video_grid"] = {"buttons": len(vgrid), "cols": ncol, "rows": nrow}
vcells = sorted(vgrid)
Hv = fit_homography([vgrid[c][:2] for c in vcells], [[float(c[0]), float(c[1])] for c in vcells])
Hinv = invert_h(Hv)
diag["video_h_resid"] = max(dist(apply_h(Hv, vgrid[c][0], vgrid[c][1]), c) for c in vcells)

# pixel window per cell (all cells of the lattice, missing ones predicted via Hinv)
win = {}
for c in range(ncol):
    for r in range(nrow):
        px = apply_h(Hinv, float(c), float(r))
        nb = []
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if 0 <= c + dc < ncol and 0 <= r + dr < nrow:
                nb.append(dist(px, apply_h(Hinv, float(c + dc), float(r + dr))))
        h = max(2, int(round(0.3 * min(nb))))
        area = vgrid[(c, r)][2] if (c, r) in vgrid else (2 * h) ** 2 * 0.3
        win[(c, r)] = [int(round(px[0])), int(round(px[1])), h, area]

# ---- 2. red onsets -> pattern --------------------------------------------------
last_red = {}
onsets = []
for fi in range(len(video)):
    fr = video[fi]
    t = float(fr.timestamp_s)
    img = fr.cameras["front"].rgb
    Himg = len(img); Wimg = len(img[0])
    for cell in win:
        x0, y0, h, area = win[cell]
        cnt = 0
        for y in range(max(0, y0 - h), min(Himg, y0 + h + 1)):
            row = img[y]
            for x in range(max(0, x0 - h), min(Wimg, x0 + h + 1)):
                if is_red(row[x]):
                    cnt += 1
        if cnt >= max(4, 0.3 * area):
            if cell not in last_red or t - last_red[cell] > 1.0:
                onsets.append([t, fi, cell, cnt])
            last_red[cell] = t
onsets = [o[5:] for o in sorted([[x[0], -x[3], x[1], x[2][0], x[2][1]] + x for x in onsets])]
pattern = []
for o in onsets:
    if pattern and pattern[-1][2] == o[2]:
        continue
    pattern.append(o)
diag["pattern_video"] = [[round(o[0], 2), o[2][0], o[2][1]] for o in pattern]
if len(pattern) == 0:
    raise ValueError("no touch detected in the conditioning video")

# ---- 3. touch poses from the video's robot state ---------------------------------
def ee_of(fr):
    p = fr.robot_state.end_effector_poses["primary"]
    return [float(p.position[0]), float(p.position[1]), float(p.position[2])], [float(q) for q in p.quaternion_wxyz]

vt = [float(f.timestamp_s) for f in video]
touch = []
for k in range(len(pattern)):
    t = pattern[k][0]
    lo = t - 0.5; hi = t + 0.5
    if k > 0:
        lo = max(lo, 0.5 * (pattern[k - 1][0] + t))
    if k + 1 < len(pattern):
        hi = min(hi, 0.5 * (t + pattern[k + 1][0]))
    best = None
    for fi in range(len(video)):
        if lo <= vt[fi] <= hi:
            pos, quat = ee_of(video[fi])
            if best is None or pos[2] < best[0][2]:
                best = [pos, quat, fi]
    if best is None:
        pos, quat = ee_of(video[pattern[k][1]])
        best = [pos, quat, pattern[k][1]]
    touch.append(best)

zs = sorted(tp[0][2] for tp in touch)
z_touch = zs[len(zs) // 2]

cells = [[float(o[2][0]), float(o[2][1])] for o in pattern]
xy = [[tp[0][0], tp[0][1]] for tp in touch]
distinct = []
for c in cells:
    if c not in distinct:
        distinct.append(c)
if len(distinct) < 2:
    raise ValueError("pattern has fewer than two distinct buttons; cannot calibrate")
simP = fit_similarity(cells, xy, True)
simI = fit_similarity(cells, xy, False)
S = simP if simP["resid"] <= simI["resid"] else simI
diag["sim_resid_proper"] = simP["resid"]
diag["sim_resid_improper"] = simI["resid"]
diag["spacing_m_video"] = S["scale"]

# end-of-video check: the video may end at the moment of the final touch, before it turns red
endpos, endquat = ee_of(video[len(video) - 1])
if endpos[2] < z_touch + 0.006:
    best = None; bd = None
    for c in range(ncol):
        for r in range(nrow):
            d = dist(S["f"]([float(c), float(r)]), endpos)
            if bd is None or d < bd:
                best = (c, r); bd = d
    if bd is not None and bd < 0.3 * S["scale"] and best != pattern[-1][2]:
        pattern.append([vt[len(video) - 1], len(video) - 1, best, 0])
        touch.append([endpos, endquat, len(video) - 1])
        diag["appended_final_touch"] = [best[0], best[1]]

# inter-touch lift from the video's EE height
lifts = []
for k in range(len(touch) - 1):
    f0 = touch[k][2]; f1 = touch[k + 1][2]
    zmax = z_touch
    for fi in range(f0, f1 + 1):
        zmax = max(zmax, ee_of(video[fi])[0][2])
    lifts.append(min(0.10, max(0.0, zmax - z_touch)))
diag["lifts"] = [round(l, 4) for l in lifts]
quat = touch[0][1]

# ---- 4. live grid -----------------------------------------------------------------
# v01: keep only blobs that are lattice-connected (BFS in image space), then keep the
# ncol x nrow index window (dimensions from the VIDEO grid) holding the most blobs, drop
# lattice outliers and refit.  v00 let robot-body blobs at the image top into the fit.
obs = get_observation()
lb = prune_isolated(white_blobs(to_rows(obs.cameras["front"].rgb)))
if len(lb) < 9:
    raise ValueError("could not detect the live button grid")
# v02: lattice membership is decided in the video-homography plane, where the grid is a
# (rotated) square lattice; in the image, perspective + rotation broke the BFS (ep2 kept 5).
qall = [apply_h(Hv, b[0], b[1]) + [b[2]] for b in lb]
lgrid = index_lattice_image(qall)
q = [lgrid[c][:2] for c in sorted(lgrid)]
L = fit_plane_lattice(q)
for rnd in range(2):
    sc = max(i[0] for i in L["idx"]) + 1
    sr = max(i[1] for i in L["idx"]) + 1
    if sc > ncol or sr > nrow:
        bestw = None; bestn = -1
        for oc in range(0, max(1, sc - ncol + 1)):
            for orr in range(0, max(1, sr - nrow + 1)):
                n = 0
                for i in L["idx"]:
                    if oc <= i[0] < oc + ncol and orr <= i[1] < orr + nrow:
                        n += 1
                if n > bestn:
                    bestn = n; bestw = [oc, orr]
        q = [p for p, i in zip(q, L["idx"]) if bestw[0] <= i[0] < bestw[0] + ncol and bestw[1] <= i[1] < bestw[1] + nrow]
        L = fit_plane_lattice(q)
    keep = []
    for p, i in zip(q, L["idx"]):
        px = L["tx"] + L["a"] * i[0] - L["b"] * i[1]
        py = L["ty"] + L["b"] * i[0] + L["a"] * i[1]
        if dist([px, py], p) <= 0.3 * L["spacing"]:
            keep.append(p)
    if len(keep) < len(q) and len(keep) >= 9:
        q = keep
        L = fit_plane_lattice(q)
lspan = [max(i[0] for i in L["idx"]) + 1, max(i[1] for i in L["idx"]) + 1]
shift = [0, 0]
if lspan[0] < ncol:
    shift[0] = (ncol - lspan[0]) // 2
if lspan[1] < nrow:
    shift[1] = (nrow - lspan[1]) // 2
if shift != [0, 0]:
    # a whole edge row/column is missing: centre the detected part (ambiguous; recorded)
    L["tx"] = L["tx"] - (L["a"] * shift[0] - L["b"] * shift[1])
    L["ty"] = L["ty"] - (L["b"] * shift[0] + L["a"] * shift[1])
diag["live_grid"] = {"buttons": len(q), "span": lspan, "shift": shift, "theta_deg": L["theta_deg"],
                     "spacing_rel": L["spacing"], "resid": L["resid"]}
try:
    vp = video[0].cameras["front"].camera_pose.position
    lp = obs.cameras["front"].camera_pose.position
    diag["front_camera_shift_m"] = dist([float(vp[0]), float(vp[1])], [float(lp[0]), float(lp[1])])
except Exception:
    diag["front_camera_shift_m"] = "unavailable"


def live_xy(cell):
    c = cell[0]; r = cell[1]
    qx = L["tx"] + L["a"] * c - L["b"] * r
    qy = L["ty"] + L["b"] * c + L["a"] * r
    return S["f"]([qx, qy])


base = get_robot_state().base_frame
press = 0.002
targets = [live_xy(o[2]) for o in pattern]
diag["pattern_live_xy"] = [[round(p[0], 4), round(p[1], 4)] for p in targets]

# ---- 5. execute -----------------------------------------------------------------
log = []


def go(p, label):
    res = move_to_pose(Pose(position=[p[0], p[1], p[2]], quaternion_wxyz=quat, frame=base))
    entry = {"label": label, "ok": bool(res.ok)}
    try:
        entry["status"] = str(res.status)
    except Exception:
        pass
    try:
        entry["steps"] = int(res.steps_executed)
    except Exception:
        pass
    log.append(entry)
    return res


hover = z_touch + max(0.04, max(lifts) if lifts else 0.0)
done = False
res = go([targets[0][0], targets[0][1], hover], "hover0")
done = is_terminal(res)
if not done:
    res = go([targets[0][0], targets[0][1], z_touch - press], "touch0")
    done = is_terminal(res)
for k in range(1, len(targets)):
    if done:
        break
    lift = lifts[k - 1] if k - 1 < len(lifts) else 0.0
    if lift > 0.006:
        mid = [0.5 * (targets[k - 1][0] + targets[k][0]), 0.5 * (targets[k - 1][1] + targets[k][1]), z_touch + lift]
        res = go(mid, "mid" + str(k))
        done = is_terminal(res)
        if done:
            break
    res = go([targets[k][0], targets[k][1], z_touch - press], "touch" + str(k))
    done = is_terminal(res)

try:
    diag["final_t"] = float(get_robot_state().timestamp_s)
except Exception:
    pass
result = {"ok": True, "terminal": done, "moves": log, "diag": diag,
          "pattern": [[o[2][0], o[2][1]] for o in pattern]}
