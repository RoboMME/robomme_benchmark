"""Helpers for VideoPlaceOrder: observation decoding, video parsing, perception, motion."""
import base64, io, json, re, math, time
from pathlib import Path
import numpy as np
from PIL import Image

WS = Path("/workspace")
LOG = []

def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)

def parse(reply):
    meta = None
    for b in reply.get("content", []):
        if b.get("type") == "text":
            try:
                meta = json.loads(b["text"]); break
            except Exception:
                pass
    imgs = {}
    if meta and "attachments" in meta:
        blocks = reply["content"]
        for att in meta["attachments"]:
            blk = blocks[att["content_index"]]
            data = base64.b64decode(blk["data"])
            im = Image.open(io.BytesIO(data))
            if att["kind"] == "rgb":
                imgs[(att["camera"], "rgb")] = np.asarray(im.convert("RGB"))
            else:
                imgs[(att["camera"], "depth")] = np.asarray(im).astype(np.float32) * 0.001
    return meta, imgs

def observe(ctx):
    return parse(ctx.call("robomme_observe"))

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def backproject(meta, cam, us, vs, depth):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    us = np.asarray(us, float); vs = np.asarray(vs, float)
    d = depth[vs.astype(int), us.astype(int)]
    pc = np.stack([(us-cx)*d/fx, -(vs-cy)*d/fy, -d], -1)
    return pc @ R.T + t, d

def project(meta, cam, P):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    pc = (np.asarray(P) - t) @ R
    d = -pc[..., 2]
    u = K[0, 2] + K[0, 0]*pc[..., 0]/d
    v = K[1, 2] - K[1, 1]*pc[..., 1]/d
    return u, v

def world_points(meta, imgs, cam, mask):
    vs, us = np.nonzero(mask)
    if len(us) == 0:
        return np.zeros((0, 3))
    P, d = backproject(meta, cam, us, vs, imgs[(cam, "depth")])
    return P[d > 0]

def red_mask(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (r > 120) & (g < 60) & (b < 60) & (r - g > 90)

def target_mask(rgb):
    # purple/magenta rings: r and b high, g lower
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (b > 150) & (r > 120) & (g < 0.8*b) & (b - g > 50)

def components(mask, min_px=3):
    from collections import deque
    H, W = mask.shape
    seen = np.zeros_like(mask, bool); comps = []
    for v0, u0 in zip(*np.nonzero(mask)):
        if seen[v0, u0]:
            continue
        q = deque([(v0, u0)]); seen[v0, u0] = True; pts = []
        while q:
            v, u = q.popleft(); pts.append((v, u))
            for dv in (-1, 0, 1):
                for du in (-1, 0, 1):
                    vv, uu = v+dv, u+du
                    if 0 <= vv < H and 0 <= uu < W and mask[vv, uu] and not seen[vv, uu]:
                        seen[vv, uu] = True; q.append((vv, uu))
        if len(pts) >= min_px:
            comps.append(np.array(pts))
    return comps

def table_points(meta, imgs, cam, mask, zmax=0.1):
    P = world_points(meta, imgs, cam, mask)
    return P[(P[:, 2] < zmax) & (P[:, 2] > -0.05)] if len(P) else P

def find_cube(meta, imgs, cam="front"):
    P = table_points(meta, imgs, cam, red_mask(imgs[(cam, "rgb")]))
    if len(P) < 3:
        return None
    c = np.median(P, 0)
    return c

def find_targets(meta, imgs, cam="front", min_px=6):
    m = target_mask(imgs[(cam, "rgb")])
    out = []
    # grow mask slightly to merge rings (white between purple rings)
    from PIL import ImageFilter
    mg = np.asarray(Image.fromarray((m*255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))) > 0
    for comp in components(mg, min_px):
        vs, us = comp[:, 0], comp[:, 1]
        sel = m[vs, us]
        if sel.sum() < min_px:
            continue
        P, d = backproject(meta, cam, us[sel], vs[sel], imgs[(cam, "depth")])
        P = P[(d > 0) & (P[:, 2] < 0.05) & (P[:, 2] > -0.05)]
        if len(P) < 3:
            continue
        out.append(dict(xyz=P.mean(0), n=len(P), uv=(float(us.mean()), float(vs.mean()))))
    return out

# ---------------- video parsing -----------------
def load_video(meta):
    seq = meta.get("frame_sequence") or {}
    mp = seq.get("manifest_path")
    if not mp:
        return None
    return json.loads((WS / mp).read_text())

def video_events(man):
    """Return list of grasp and release events from video states."""
    frames = [f for f in man["frames"] if f["phase"] == "video_demo"]
    frames.sort(key=lambda f: f["frame_index"])
    ev = []
    prev = None
    for f in frames:
        s = f["state"]; w = s[6] + s[7]
        st = "open" if w > 0.065 else ("hold" if w > 0.01 else "shut")
        if prev is not None and st != prev[0]:
            ev.append(dict(kind=f"{prev[0]}->{st}", frame=f["frame_index"], xyz=s[:3], prev_xyz=prev[1]))
        prev = (st, s[:3])
    return ev, frames

def placements(man):
    ev, frames = video_events(man)
    rel = [e for e in ev if e["kind"] == "hold->open"]
    grasps = [e for e in ev if e["kind"] == "open->hold"]
    return rel, grasps, ev

ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
       "eighth": 8, "ninth": 9, "tenth": 10, "last": -1, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5}

def parse_ordinal(instr):
    s = instr.lower()
    m = re.search(r"on the (\w+) target", s)
    if m and m.group(1) in ORD:
        return ORD[m.group(1)]
    for k, v in ORD.items():
        if re.search(r"\b" + k + r"\b", s):
            return v
    return None

# ---------------- motion -----------------
def step(ctx, rows):
    meta, imgs = parse(ctx.call("robomme_step", actions=[list(map(float, r)) for r in rows]))
    return meta, imgs

def rv_to_R(rv):
    rv = np.asarray(rv, float); th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv/th; Kx = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3) + math.sin(th)*Kx + (1-math.cos(th))*Kx@Kx

def R_to_rv(R):
    c = (np.trace(R)-1)/2; c = max(-1.0, min(1.0, c)); th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        w, V = np.linalg.eigh((R+np.eye(3))/2); k = V[:, np.argmax(w)]
        return k*th
    v = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])/(2*math.sin(th))
    return v*th

def rot_err(state, yaw=0.0):
    """world rotation vector taking current orientation to gripper-down with given extra yaw."""
    cur = rv_to_R(state[3:6])
    tgt = rv_to_R([0, 0, yaw]) @ rv_to_R([math.pi, 0, 0])
    return R_to_rv(tgt @ cur.T)

def move_to(ctx, target, grip, tol=0.006, max_iter=8, vmax=0.8, yaw=0.0, state=None, settle=0):
    """Closed-loop translate the EE to `target` holding gripper command `grip`."""
    target = np.asarray(target, float)
    meta = imgs = None
    for it in range(max_iter):
        if state is None:
            meta, imgs = observe(ctx); state = meta["state"]
        pos = np.asarray(state[:3]); err = target - pos
        d = np.linalg.norm(err)
        rerr = rot_err(state, yaw)
        if d < tol and np.linalg.norm(rerr) < 0.05:
            break
        n = max(1, int(math.ceil(d / (0.05*vmax))))
        n = min(n, 30)
        rows = []
        for i in range(n):
            dp = err / n / 0.05
            dr = np.clip(rerr / n / 0.5, -1, 1) if i == 0 else np.zeros(3)
            rows.append(list(np.clip(dp, -1, 1)) + list(dr) + [grip])
        meta, imgs = step(ctx, rows)
        state = meta["state"]
        if meta.get("episode_ended"):
            return meta, imgs
    if settle:
        meta, imgs = step(ctx, [[0, 0, 0, 0, 0, 0, grip]] * settle)
    if meta is None:
        meta, imgs = observe(ctx)
    return meta, imgs

def gripper(ctx, grip, n=8):
    return step(ctx, [[0, 0, 0, 0, 0, 0, grip]] * n)

def color_mask(rgb, color):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    if color == "red":
        return (r > 120) & (g < 70) & (b < 70) & (r - g > 80)
    if color == "green":
        return (g > 120) & (r < 90) & (b < 90) & (g - r > 70)
    if color == "blue":
        return (b > 100) & (r < 60) & (g < 60) & (b - r > 60)
    if color == "yellow":
        return (r > 150) & (g > 150) & (b < 80)
    return red_mask(rgb)

def parse_color(instr):
    s = instr.lower()
    for c in ["red", "green", "blue", "yellow"]:
        if c + " cube" in s:
            return c
    for c in ["red", "green", "blue", "yellow"]:
        if c in s:
            return c
    return "red"

def find_cube_color(meta, imgs, cam, color, near=None, radius=0.06):
    P = table_points(meta, imgs, cam, color_mask(imgs[(cam, "rgb")], color))
    if near is not None and len(P):
        P = P[np.linalg.norm(P[:, :2] - np.asarray(near)[:2], axis=1) < radius]
    if len(P) < 5:
        return None
    top = P[P[:, 2] > P[:, 2].max() - 0.006]
    return top.mean(0), len(P)

def cube_chain(ev, start_xy, tol_start=0.05, tol_link=0.035):
    """Releases of the cube whose first grasp is near start_xy, following grasp-at-last-release links."""
    grasps = [e for e in ev if e["kind"] == "open->hold" and e["xyz"][2] < 0.06]
    rels = [e for e in ev if e["kind"] == "hold->open" and e["xyz"][2] < 0.06]
    if not grasps:
        return []
    d = [np.linalg.norm(np.array(g["xyz"][:2]) - np.asarray(start_xy)[:2]) for g in grasps]
    j = int(np.argmin(d))
    if d[j] > tol_start:
        return None
    chain = []; g = grasps[j]
    while True:
        nxt = [r for r in rels if r["frame"] > g["frame"]]
        if not nxt:
            break
        r = nxt[0]; chain.append(r)
        ng = [x for x in grasps if x["frame"] > r["frame"] and
              np.linalg.norm(np.array(x["xyz"][:2]) - np.array(r["xyz"][:2])) < tol_link]
        if not ng:
            break
        g = ng[0]
    return chain

# ---------------- target tracking through the video -----------------
def frame_images(f):
    imgs = {}
    for x in f["files"]:
        if x["camera"] != "front":
            continue
        im = Image.open(WS / x["path"])
        imgs[("front", x["kind"])] = np.asarray(im.convert("RGB")) if x["kind"] == "rgb" else np.asarray(im).astype(np.float32) * 0.001
    return imgs

def detect_targets_frame(f, min_px=8):
    imgs = frame_images(f)
    if ("front", "rgb") not in imgs or ("front", "depth") not in imgs:
        return []
    m = target_mask(imgs[("front", "rgb")])
    out = []
    for comp in components(m, min_px):
        vs, us = comp[:, 0], comp[:, 1]
        P, d = backproject(f, "front", us, vs, imgs[("front", "depth")])
        P = P[(d > 0) & (P[:, 2] < 0.03) & (P[:, 2] > -0.02)]
        if len(P) >= min_px // 2:
            out.append((P[:, :2].mean(0), len(P)))
    # merge components closer than 3 cm (rings split into pieces)
    merged = []
    for c, n in sorted(out, key=lambda t: -t[1]):
        for mm in merged:
            if np.linalg.norm(mm[0] - c) < 0.03:
                mm[0] = (mm[0] * mm[1] + c * n) / (mm[1] + n); mm[1] += n; break
        else:
            merged.append([c.copy(), n])
    return [m_[0] for m_ in merged]

VEL_GAIN = 1.0
COAST = 4

def _assign(C, gate_vec):
    """Min-cost assignment rows->cols with per-row gate; returns dict row->col."""
    try:
        from scipy.optimize import linear_sum_assignment
        Cg = C.copy(); big = 1e3
        Cg[Cg > gate_vec[:, None]] = big
        r, c = linear_sum_assignment(Cg)
        return {int(i): int(j) for i, j in zip(r, c) if Cg[i, j] < big}
    except Exception:
        pairs = sorted((C[i, j], i, j) for i in range(C.shape[0]) for j in range(C.shape[1]))
        ui, uj, out = set(), set(), {}
        for c, i, j in pairs:
            if i in ui or j in uj or c > gate_vec[i]:
                continue
            ui.add(i); uj.add(j); out[i] = j
        return out

def track_targets(man, init_xy, gate=0.07):
    """Track target identities (initialised at init_xy) through video frames with a
    constant-velocity predictor and optimal assignment. Returns frame_index -> Nx2 array."""
    frames = sorted(man["frames"], key=lambda f: f["frame_index"])
    N = len(init_xy)
    pos = np.array(init_xy, float); vel = np.zeros_like(pos); miss = np.zeros(N, int)
    hist = {}
    for f in frames:
        dets = detect_targets_frame(f)
        pred = np.clip(pos + vel, -0.6, 0.6)
        newpos = pred.copy(); seen = np.zeros(N, bool)
        if dets:
            D = np.array(dets)
            C = np.linalg.norm(pred[:, None, :] - D[None, :, :], axis=2)
            gv = gate * (1 + 0.5 * np.minimum(miss, 4))
            for i, j in _assign(C, gv).items():
                newpos[i] = D[j]; seen[i] = True
        for i in range(N):
            if seen[i]:
                vel[i] = VEL_GAIN * (newpos[i] - pos[i]); miss[i] = 0
            else:
                miss[i] += 1
                if miss[i] > COAST:
                    vel[i] = 0.0
        pos = newpos
        hist[f["frame_index"]] = pos.copy()
    return hist

# ---------------- pair-swap shuffle model -----------------
def video_detections(man):
    frames = sorted(man["frames"], key=lambda f: f["frame_index"])
    return [(f["frame_index"], np.array(detect_targets_frame(f)).reshape(-1, 2)) for f in frames]

def infer_swaps(dets, rest, rest_tol=0.02, circ_tol=0.015, min_score=3, gap=2):
    """dets: list of (frame_index, Kx2). rest: Nx2 rest spots. Returns list of (start, end, [(p,q),...])."""
    N = len(rest)
    mids = {}; rads = {}
    for p in range(N):
        for q in range(p + 1, N):
            mids[(p, q)] = (rest[p] + rest[q]) / 2; rads[(p, q)] = np.linalg.norm(rest[p] - rest[q]) / 2
    moving = []
    for fi, D in dets:
        mv = [d for d in D if np.min(np.linalg.norm(rest - d, axis=1)) > rest_tol] if len(D) else []
        moving.append((fi, mv))
    # intervals of frames with moving detections
    iv = []; cur = None; last = -10
    for idx, (fi, mv) in enumerate(moving):
        if mv:
            if cur is None or idx - last > gap:
                if cur is not None:
                    iv.append(cur)
                cur = [idx, idx]
            cur[1] = idx; last = idx
    if cur is not None:
        iv.append(cur)
    out = []
    for a, b in iv:
        score = {pq: 0.0 for pq in mids}
        for idx in range(a, b + 1):
            for d in moving[idx][1]:
                for pq in mids:
                    if abs(np.linalg.norm(d - mids[pq]) - rads[pq]) < circ_tol:
                        score[pq] += 1
        chosen = []; used = set()
        for pq, s in sorted(score.items(), key=lambda t: -t[1]):
            if s < min_score:
                break
            if pq[0] in used or pq[1] in used:
                continue
            chosen.append(pq); used.update(pq)
        out.append((moving[a][0], moving[b][0], chosen, {k: v for k, v in score.items() if v >= min_score}))
    return out

def answer_by_swaps(man, chain, k, rest, swaps):
    """Return (spot index of final location of target receiving k-th placement, details)."""
    N = len(rest)
    perm = list(range(N))  # perm[spot] = target id
    rel_frames = [e["frame"] for e in chain]
    placed_ids = []
    sw = sorted(swaps, key=lambda t: t[0]); si = 0
    events = sorted([(e["frame"], i) for i, e in enumerate(chain)])
    ids = [None] * len(chain)
    for fr_, i in events:
        while si < len(sw) and sw[si][1] < fr_:
            for p, q in sw[si][2]:
                perm[p], perm[q] = perm[q], perm[p]
            si += 1
        xy = np.array(chain[i]["xyz"][:2]); d = np.linalg.norm(rest - xy, axis=1)
        j = int(np.argmin(d))
        ids[i] = (perm[j] if d[j] < 0.04 else None, j, float(d[j]))
    while si < len(sw):
        for p, q in sw[si][2]:
            perm[p], perm[q] = perm[q], perm[p]
        si += 1
    final_spot = {tid: s for s, tid in enumerate(perm)}
    tid = ids[k - 1][0] if 0 < k <= len(ids) else None
    return (final_spot[tid] if tid is not None else None), ids, perm

def real_shuffles(dets, rest, far=0.05, gap=2, min_frames=3, pad=2):
    """Index intervals [a,b] (into dets list) where rings are clearly away from rest spots."""
    flags = []
    for fi, D in dets:
        f = False
        if len(D):
            dd = np.min(np.linalg.norm(D[:, None, :] - rest[None, :, :], axis=2), axis=1)
            f = bool(np.any(dd > far))
        flags.append(f)
    iv = []; cur = None; last = -100
    for i, f in enumerate(flags):
        if f:
            if cur is not None and i - last <= gap + 1:
                cur[1] = i
            else:
                if cur is not None:
                    iv.append(cur)
                cur = [i, i]
            last = i
    if cur is not None:
        iv.append(cur)
    out = []
    for a, b in iv:
        n = sum(flags[a:b + 1])
        if n >= min_frames:
            out.append([max(0, a - pad), min(len(dets) - 1, b + pad)])
    return out

def shuffle_perm(dets, rest, a, b, gate=0.09, vel_gain=0.8):
    """Track all targets from rest spots through dets[a..b]; return mapping start_spot->end_spot or None."""
    N = len(rest)
    pos = rest.copy(); vel = np.zeros_like(pos)
    for idx in range(a, b + 1):
        D = dets[idx][1]
        pred = pos + vel
        newpos = pred.copy(); seen = np.zeros(N, bool)
        if len(D):
            C = np.linalg.norm(pred[:, None, :] - D[None, :, :], axis=2)
            for i, j in _assign(C, np.full(N, gate)).items():
                newpos[i] = D[j]; seen[i] = True
        for i in range(N):
            if seen[i]:
                vel[i] = vel_gain * (newpos[i] - pos[i])
            else:
                newpos[i] = pos[i]; vel[i] = 0.0
        pos = newpos
    C = np.linalg.norm(pos[:, None, :] - rest[None, :, :], axis=2)
    m = _assign(C, np.full(N, 0.05))
    if len(m) != N or len(set(m.values())) != N:
        return None, pos
    return m, pos

def target_identity_answer(man, chain, k, dets=None):
    """Follow target identities through real shuffles. Returns (final rest xy of the target that got
    the k-th placement, diagnostics)."""
    if dets is None:
        dets = video_detections(man)
    rest = dets[0][1]
    N = len(rest)
    ivs = real_shuffles(dets, rest)
    perm = list(range(N))  # perm[spot] = target id
    frames_idx = [fi for fi, _ in dets]
    diag = {"rest": rest.tolist(), "shuffles": []}
    ids = []
    rel = sorted(chain, key=lambda e: e["frame"])
    ri = 0
    def place_ids(upto_frame):
        nonlocal ri
        while ri < len(rel) and rel[ri]["frame"] <= upto_frame:
            xy = np.array(rel[ri]["xyz"][:2]); d = np.linalg.norm(rest - xy, axis=1); j = int(np.argmin(d))
            ids.append((perm[j] if d[j] < 0.04 else None, j, round(float(d[j]), 3), rel[ri]["frame"]))
            ri += 1
    for a, b in ivs:
        place_ids(frames_idx[a] - 1)
        m, vac = shuffle_perm2(dets, rest, a, b)
        if m is None and len(vac) == 2:
            m = {s: s for s in range(N)}; m[vac[0]] = vac[1]; m[vac[1]] = vac[0]
        if m is None:
            diag["shuffles"].append((frames_idx[a], frames_idx[b], "FAILED"))
            continue
        newperm = [None] * N
        for s, e in m.items():
            newperm[e] = perm[s]
        perm = newperm
        diag["shuffles"].append((frames_idx[a], frames_idx[b], {int(s): int(e) for s, e in m.items() if s != e}))
    place_ids(10 ** 9)
    diag["ids"] = ids; diag["perm"] = perm
    if not (0 < k <= len(ids)) or ids[k - 1][0] is None:
        return None, diag
    tid = ids[k - 1][0]
    spot = perm.index(tid)
    diag["answer_spot"] = spot
    return rest[spot], diag

def shuffle_perm2(dets, rest, a, b, gate=0.07, vac_tol=0.02, stat_tol=0.03, coast=4):
    """Per-shuffle mapping start_spot->end_spot, tracking only rings that leave their spot."""
    N = len(rest)
    # vacated spots: spot with no detection within vac_tol in some frame of the interval
    vac = set()
    for idx in range(a, b + 1):
        D = dets[idx][1]
        for s in range(N):
            if len(D) == 0 or np.min(np.linalg.norm(D - rest[s], axis=1)) > vac_tol:
                vac.add(s)
    vac = sorted(vac)
    stat = [s for s in range(N) if s not in vac]
    if not vac:
        return {s: s for s in range(N)}, vac
    pos = rest[vac].copy(); vel = np.zeros_like(pos); miss = np.zeros(len(vac), int)
    for idx in range(a, b + 1):
        D = dets[idx][1]
        if len(D) and stat:
            dd = np.min(np.linalg.norm(D[:, None, :] - rest[stat][None, :, :], axis=2), axis=1)
            D = D[dd > stat_tol]
        pred = pos + vel
        newpos = pred.copy(); seen = np.zeros(len(vac), bool)
        if len(D):
            C = np.linalg.norm(pred[:, None, :] - D[None, :, :], axis=2)
            for i, j in _assign(C, np.full(len(vac), gate) * (1 + 0.5 * np.minimum(miss, 3))).items():
                newpos[i] = D[j]; seen[i] = True
        for i in range(len(vac)):
            if seen[i]:
                vel[i] = 0.8 * (newpos[i] - pos[i]); miss[i] = 0
            else:
                miss[i] += 1
                if miss[i] > coast:
                    vel[i] = 0.0
        pos = np.clip(newpos, -0.6, 0.6)
    C = np.linalg.norm(pos[:, None, :] - rest[vac][None, :, :], axis=2)
    m = _assign(C, np.full(len(vac), 0.06))
    if len(m) != len(vac):
        return None, vac
    out = {s: s for s in stat}
    for i, j in m.items():
        out[vac[i]] = vac[j]
    return out, vac

def circle_pairs(dets, rest, a, b, vac, rest_tol=0.025, circ_tol=0.015, rel=0.4, min_score=4):
    N = len(rest)
    pairs = [(p, q) for i, p in enumerate(vac) for q in vac[i + 1:]]
    score = {pq: 0 for pq in pairs}
    for idx in range(a, b + 1):
        D = dets[idx][1]
        if not len(D):
            continue
        dd = np.min(np.linalg.norm(D[:, None, :] - rest[None, :, :], axis=2), axis=1)
        M = D[dd > rest_tol]
        for (p, q) in pairs:
            m = (rest[p] + rest[q]) / 2; r = np.linalg.norm(rest[p] - rest[q]) / 2
            score[(p, q)] += int(np.sum(np.abs(np.linalg.norm(M - m, axis=1) - r) < circ_tol))
    chosen = []; used = set(); top = max(score.values()) if score else 0
    for pq, s in sorted(score.items(), key=lambda t: -t[1]):
        if s < max(min_score, rel * top):
            break
        if pq[0] in used or pq[1] in used:
            continue
        chosen.append(pq); used.update(pq)
    return chosen, score

def vacated(dets, rest, a, b, vac_tol=0.02, min_frames=3):
    N = len(rest); cnt = np.zeros(N, int)
    for idx in range(a, b + 1):
        D = dets[idx][1]
        for s in range(N):
            if len(D) == 0 or np.min(np.linalg.norm(D - rest[s], axis=1)) > vac_tol:
                cnt[s] += 1
    return [s for s in range(N) if cnt[s] >= min_frames]

def identity_answer_pairs(man, chain, k, dets=None):
    if dets is None:
        dets = video_detections(man)
    rest = dets[0][1]; N = len(rest)
    ivs = real_shuffles(dets, rest)
    frames_idx = [fi for fi, _ in dets]
    perm = list(range(N)); ids = []; diag = {"rest": rest.tolist(), "shuffles": []}
    rel_ = sorted(chain, key=lambda e: e["frame"]); ri = 0
    def place_ids(upto):
        nonlocal ri
        while ri < len(rel_) and rel_[ri]["frame"] <= upto:
            xy = np.array(rel_[ri]["xyz"][:2]); d = np.linalg.norm(rest - xy, axis=1); j = int(np.argmin(d))
            ids.append((perm[j] if d[j] < 0.04 else None, j, round(float(d[j]), 3), rel_[ri]["frame"])); ri += 1
    for a, b in ivs:
        place_ids(frames_idx[a] - 1)
        vac = vacated(dets, rest, a, b)
        chosen, score = circle_pairs(dets, rest, a, b, vac)
        for p, q in chosen:
            perm[p], perm[q] = perm[q], perm[p]
        diag["shuffles"].append((frames_idx[a], frames_idx[b], vac, chosen, {f"{k_[0]}-{k_[1]}": v for k_, v in score.items() if v}))
    place_ids(10 ** 9)
    diag["ids"] = ids; diag["perm"] = perm
    if not (0 < k <= len(ids)) or ids[k - 1][0] is None:
        return None, diag
    spot = perm.index(ids[k - 1][0]); diag["answer_spot"] = spot
    return rest[spot], diag

# ---------------- v5: point-symmetric pair-swap inference ----------------
def _full_rest(D, rest, tol=0.02):
    if len(D) != len(rest):
        return False
    C = np.linalg.norm(D[:, None, :] - rest[None, :, :], axis=2)
    m = _assign(C, np.full(len(D), tol))
    return len(m) == len(rest)

def sym_intervals(dets, rest, far=0.03):
    """Runs of frames between fully-at-rest frames that contain at least one far detection."""
    iv = []; cur = None
    for i, (fi, D) in enumerate(dets):
        if _full_rest(D, rest):
            if cur is not None:
                iv.append(cur); cur = None
            continue
        moving = len(D) and np.any(np.min(np.linalg.norm(D[:, None, :] - rest[None, :, :], axis=2), axis=1) > far)
        if cur is None:
            cur = [i, i, 0]
        cur[1] = i
        if moving:
            cur[2] += 1
    if cur is not None:
        iv.append(cur)
    return [(a, b) for a, b, n in iv if n >= 2]

def sym_pairs(dets, rest, a, b, mov_tol=0.025, mid_tol=0.022, vac_tol=0.02, min_vac=3):
    N = len(rest)
    pairs = [(p, q) for p in range(N) for q in range(p + 1, N)]
    mids = {pq: (rest[pq[0]] + rest[pq[1]]) / 2 for pq in pairs}
    rads = {pq: np.linalg.norm(rest[pq[0]] - rest[pq[1]]) / 2 for pq in pairs}
    votes = {pq: 0.0 for pq in pairs}
    vac_any = set()
    vcount = np.zeros(N, int)
    for idx in range(a, b + 1):
        D = dets[idx][1]
        for s_ in range(N):
            if len(D) == 0 or np.min(np.linalg.norm(D - rest[s_], axis=1)) > vac_tol:
                vcount[s_] += 1
    for idx in range(a, b + 1):
        D = dets[idx][1]
        if len(D) == 0:
            continue
        dmin = np.min(np.linalg.norm(D[:, None, :] - rest[None, :, :], axis=2), axis=1)
        M = D[dmin > mov_tol]
        vac = set(s for s in range(N) if np.min(np.linalg.norm(D - rest[s], axis=1)) > vac_tol)
        vac_any |= vac
        for pq in pairs:
            if pq[0] not in vac or pq[1] not in vac or min(vcount[pq[0]], vcount[pq[1]]) < min_vac:
                continue
            m, r = mids[pq], rads[pq]
            best = 0.0
            for i in range(len(M)):
                for j in range(i + 1, len(M)):
                    mm = (M[i] + M[j]) / 2; s = np.linalg.norm(M[i] - M[j]) / 2
                    if np.linalg.norm(mm - m) < mid_tol and 0.8 * r < s < 1.6 * r:
                        best = max(best, 2.0)
            if best == 0.0:
                for d in M:
                    rr = np.linalg.norm(d - m)
                    if 0.85 * r < rr < 1.5 * r:
                        best = max(best, 0.3)
            votes[pq] += best
    return votes, sorted(vac_any)

def select_swaps(votes, min_sym=2.0):
    chosen = []; used = set()
    for pq, v in sorted(votes.items(), key=lambda kv: -kv[1]):
        if v < min_sym:
            break
        if pq[0] in used or pq[1] in used:
            continue
        chosen.append(pq); used |= set(pq)
    return chosen

def identity_answer_sym(man, chain, k, dets=None, min_sym=2.0):
    if dets is None:
        dets = video_detections(man)
    rest = dets[0][1]
    N = len(rest)
    perm = list(range(N))
    fidx = [fi for fi, _ in dets]
    rel = sorted(chain, key=lambda e: e["frame"])
    ids = []; ri = 0; shuf = []
    def place_ids(upto):
        nonlocal ri
        while ri < len(rel) and rel[ri]["frame"] <= upto:
            xy = np.array(rel[ri]["xyz"][:2]); d = np.linalg.norm(rest - xy, axis=1); j = int(np.argmin(d))
            ids.append((perm[j] if d[j] < 0.04 else None, j, round(float(d[j]), 3), rel[ri]["frame"]))
            ri += 1
    for a, b in sym_intervals(dets, rest):
        place_ids(fidx[a] - 1)
        votes, vac = sym_pairs(dets, rest, a, b)
        sw = select_swaps(votes, min_sym)
        if not sw and len(vac) == 2:
            pq = (vac[0], vac[1])
            if votes.get(pq, 0) >= 0.6:
                sw = [pq]
        for p, q in sw:
            perm[p], perm[q] = perm[q], perm[p]
        top = sorted(votes.items(), key=lambda kv: -kv[1])[:3]
        shuf.append((fidx[a], fidx[b], vac, sw, [(pq, round(v, 1)) for pq, v in top]))
    place_ids(10 ** 9)
    diag = {"rest": rest.tolist(), "ids": ids, "perm": perm, "shuffles": shuf}
    if not (0 < k <= len(ids)) or ids[k - 1][0] is None:
        return None, diag
    spot = perm.index(ids[k - 1][0])
    diag["answer_spot"] = spot
    return rest[spot], diag
