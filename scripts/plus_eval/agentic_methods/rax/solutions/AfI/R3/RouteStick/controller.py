"""RouteStick controller v3 (v2 + piecewise similarity mapping of video segments onto live discs,
single-row stable final hold, stick-offset diagnostic).

Strategy: the conditioning video's frames carry the robot state (stick TCP pose).
Extract that trajectory, map it to the current scene (disc markers matched between
video frame 0 and the live observation; identity if they coincide), then track it
step by step with closed-loop proportional control plus orientation hold.

Phases: P0 load video -> P1 perceive discs (video frame 0, live) -> P2 build reference
-> P3 track reference (time-indexed, waits when lagging) -> P4 hold / end.
Exits: any exception in a phase is logged; tracking falls back to the untransformed
video trajectory; the loop stops when the episode ends or the reference is finished.
"""
import json, os, time, traceback
from pathlib import Path
import numpy as np

VERSION = "v3"
WS = Path("/workspace")
OUT = WS / "output"
STEP_SCALE = 0.05          # 1.0 action = 0.05 m
ROT_SCALE = 0.5            # 1.0 action = 0.5 rad
KP = 1.6                   # position gain (measured: ~40% of commanded delta achieved per step)
KR = 1.0                   # rotation gain
LOOKAHEAD = 2              # reference frames ahead used as target
LAG_WAIT_M = 0.012         # pause reference time if tracking error exceeds this (low/drawing phase)
DRAW_Z = 0.10              # below this z the stick is considered "drawing"
MATCH_TOL_M = 0.02         # disc match tolerance for identity mapping
SETTLE_STEPS = 15


def log(*a):
    print(f"[{VERSION} {time.strftime('%H:%M:%S')}]", *a, flush=True)


def meta(reply):
    for b in reply.get("content", []):
        if b.get("type") == "text":
            return json.loads(b["text"])
    return {}


# ---------------- geometry ----------------
def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def backproject(depth_m, K, pose):
    H, W = depth_m.shape
    v, u = np.mgrid[0:H, 0:W]
    K = K["matrix"] if isinstance(K, dict) else K
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    P = np.stack([(u-cx)/fx*depth_m, -(v-cy)/fy*depth_m, -depth_m], -1)
    return P @ quat_to_R(pose["quaternion_wxyz"]).T + np.array(pose["position_m"])


def rotvec_to_R(r):
    r = np.asarray(r, float); th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r/th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th)*Kx + (1-np.cos(th))*Kx@Kx


def R_to_rotvec(R):
    c = np.clip((np.trace(R)-1)/2, -1, 1); th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > np.pi - 1e-4:
        # near pi: use diagonal
        k = np.sqrt(np.maximum((np.diag(R)+1)/2, 0))
        i = int(np.argmax(k)); k = R[:, i]+np.eye(3)[i]; k = k/np.linalg.norm(k)
        return k*th
    w = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])/(2*np.sin(th))
    return w*th


def detect_discs(rgb, depth_m, K, pose):
    """White flat markers on the table: returns list of (x, y, z, npix)."""
    import cv2
    P = backproject(depth_m, K, pose)
    z = P[..., 2]
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    m = (hsv[..., 1] < 60) & (hsv[..., 2] > 150) & (z > 0.004) & (z < 0.035) & \
        (P[..., 0] > -0.5) & (P[..., 0] < 0.3) & (np.abs(P[..., 1]) < 0.5) & (depth_m > 0)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    out = []
    for i in range(1, n):
        if st[i, 4] < 8:
            continue
        pts = P[lab == i]
        out.append((*pts.mean(0).tolist(), int(st[i, 4])))
    return out


def load_frame_images(frame):
    from PIL import Image
    files = {(f["camera"], f["kind"]): f for f in frame["files"]}
    rgb = np.array(Image.open(WS / files[("front", "rgb")]["path"]).convert("RGB"))
    df = files[("front", "depth")]
    d = np.array(Image.open(WS / df["path"])).astype(float)*float(df.get("meters_per_unit", 0.001))
    return rgb, d


def rigid_fit(A, B):
    """2-D rigid transform B ~ R A + t."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A-ca).T @ (B-cb)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = Vt.T @ U.T
    return R, cb - R @ ca


# ---------------- reference ----------------
def build_reference(frames):
    vid = sorted([f for f in frames if f.get("phase") == "video_demo"], key=lambda f: f["frame_index"])
    src = np.array([f.get("source_frame_index", f["frame_index"]) for f in vid], float)
    S = np.array([f["state"][:6] for f in vid], float)
    # resample onto the native 20 Hz clock (one control step per source frame)
    t = np.arange(src[0], src[-1]+1)
    ref = np.stack([np.interp(t, src, S[:, j]) for j in range(3)], 1)
    rot = S[:, 3:6]
    return ref, rot, vid


def compress_reference(ref, eps=0.0008, keep=4):
    """Shorten long stationary stretches (keep `keep` frames of each dwell)."""
    out = [ref[0]]; still = 0
    for p in ref[1:]:
        if np.linalg.norm(p-out[-1]) < eps:
            still += 1
            if still > keep:
                continue
        else:
            still = 0
        out.append(p)
    return np.array(out)


def find_video_frames(m):
    """Frames of the reset sequence (conditioning video + first execution frame)."""
    cands = []
    fs = m.get("frame_sequence") or {}
    if fs.get("manifest_path"):
        mp = WS/fs["manifest_path"]
        cands += [mp, mp.parent.parent/"sequence_000000"/"manifest.json"]
    cands += sorted((WS/"runtime/autonomous_controller/runs").glob("*/frames/sequence_000000/manifest.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    eid = m.get("episode_id")
    for mp in cands:
        try:
            man = json.loads(Path(mp).read_text())
        except Exception:
            continue
        frames = man.get("frames", [])
        if not any(f.get("phase") == "video_demo" for f in frames):
            continue
        meid = (man.get("transition") or {}).get("episode_id")
        if eid and meid and meid != eid:
            continue
        log("video manifest", mp)
        return frames
    for mp in cands:  # fallback: any manifest with conditioning video, newest first
        try:
            frames = json.loads(Path(mp).read_text()).get("frames", [])
        except Exception:
            continue
        if any(f.get("phase") == "video_demo" for f in frames):
            log("WARNING episode id unmatched; using video manifest", mp)
            return frames
    raise RuntimeError("no conditioning video manifest found")



DWELL_R = 0.012            # reference within this of a disc (and low) = dwell frame
ARRIVE_TOL = 0.007         # stick must be this close to the dwell point to count dwell time
RED_MIN_DWELL = 8          # steps at disc before wrist-red counts as confirmation
MAX_EXTRA_HOLD = 12        # extra steps allowed at a dwell end waiting for the red light
FINAL_HOLD = 40
MIN_STOP_FRAMES = 8         # a video stop at a disc must last this many frames to need confirmation


def decode_images(reply):
    import base64, io
    from PIL import Image
    ims = [b for b in reply.get("content", []) if b.get("type") == "image"]
    out = []
    for b in ims[:2]:
        out.append(np.array(Image.open(io.BytesIO(base64.b64decode(b["data"]))).convert("RGB")))
    return out


def red_mask(rgb):
    import cv2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return ((hsv[..., 0] < 8) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 80)


def disc_pixel_masks(rgb, depth_m, K, pose, discs):
    """Front-image pixel masks of each detected disc (dilated)."""
    import cv2
    P = backproject(depth_m, K, pose)
    masks = []
    for (x, y, z, _) in discs:
        d = np.hypot(P[..., 0]-x, P[..., 1]-y)
        m = (d < 0.022) & (P[..., 2] < 0.035)
        m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        masks.append(m)
    return masks


def label_dwells(ref, discs_xy):
    """For each ref frame: disc index if it is a low, stationary frame at a disc, else -1."""
    lab = np.full(len(ref), -1)
    if len(discs_xy) == 0:
        return lab
    d = np.linalg.norm(ref[:, None, :2]-discs_xy[None], axis=2)
    near, idx = d.min(1), d.argmin(1)
    ok = (near < DWELL_R) & (ref[:, 2] < DRAW_Z)
    lab[ok] = idx[ok]
    return lab


VISIT_R = 0.03             # video frame within this of a video disc (low) belongs to a visit run


def order_discs(X):
    X = np.asarray(X, float)
    c = X.mean(0)
    u = np.linalg.svd(X - c)[2][0]
    return np.argsort((X - c) @ u), u


def correspond(V, L):
    """Index map video disc i -> live disc j (same count, ordered along the row). None if unsure."""
    if len(V) != len(L) or len(V) < 2:
        return None, None
    iv, uv = order_discs(V)
    il, ul = order_discs(L)
    if uv @ ul < 0:
        il = il[::-1]
    corr = {int(a): int(b) for a, b in zip(iv, il)}
    # similarity fit residual as sanity check
    A = np.array([V[i] for i in corr]); B = np.array([L[corr[i]] for i in corr])
    za = A[:, 0] + 1j*A[:, 1]; zb = B[:, 0] + 1j*B[:, 1]
    ca, cb = za.mean(), zb.mean()
    sim = np.vdot(za - ca, zb - cb) / np.vdot(za - ca, za - ca)
    res = np.abs((za - ca)*sim + cb - zb).max()
    return corr, (sim, ca, cb, float(res))


def find_visits(ref, V):
    """List of (frame, video_disc) where the low stick is closest to a disc within VISIT_R."""
    V = np.asarray(V, float)
    d = np.linalg.norm(ref[:, None, :2] - V[None], axis=2)
    near, idx = d.min(1), d.argmin(1)
    ok = (near < VISIT_R) & (ref[:, 2] < DRAW_Z)
    visits = []
    i = 0; n = len(ref)
    while i < n:
        if ok[i]:
            j = i
            while j + 1 < n and ok[j+1] and idx[j+1] == idx[i]:
                j += 1
            seg = np.arange(i, j+1)
            visits.append((int(seg[np.argmin(near[seg])]), int(idx[i]), int(i), int(j)))
            i = j + 1
        else:
            i += 1
    return visits


def map_reference(ref, V, L, corr, glob):
    """Piecewise similarity: segment between visits a->b maps V_a->L_a, V_b->L_b."""
    sim_g = glob[0]
    out = ref.copy()
    visits = find_visits(ref, V)
    if not visits:
        return out, visits
    zV = lambda i: V[i][0] + 1j*V[i][1]
    zL = lambda i: L[corr[i]][0] + 1j*L[corr[i]][1]
    times = np.array([v[0] for v in visits])
    for t in range(len(ref)):
        z = ref[t, 0] + 1j*ref[t, 1]
        before = np.flatnonzero(times <= t); after = np.flatnonzero(times > t)
        if len(before) == 0:
            bd = visits[after[0]][1]
            w = t / max(1, times[after[0]])
            zz = z + w*(zL(bd) - zV(bd))
        else:
            ad = visits[before[-1]][1]
            bd = visits[after[0]][1] if len(after) else None
            if bd is None or bd == ad:
                zz = zL(ad) + (z - zV(ad))*sim_g
            else:
                zz = zL(ad) + (z - zV(ad))*(zL(bd) - zL(ad))/(zV(bd) - zV(ad))
        out[t, 0], out[t, 1] = zz.real, zz.imag
    return out, visits


def partial_correspond(V, L, max_d=0.06):
    """Mutual nearest-neighbour pairs (for unequal disc counts); returns pairs and similarity fit."""
    V = np.asarray(V, float); L = np.asarray(L, float)
    if len(V) < 3 or len(L) < 3:
        return None
    D = np.linalg.norm(V[:, None] - L[None], axis=2)
    pairs = [(i, int(D[i].argmin())) for i in range(len(V))
             if D[i].min() < max_d and int(D[:, D[i].argmin()].argmin()) == i]
    if len(pairs) < 3:
        return None
    za = np.array([V[i][0] + 1j*V[i][1] for i, _ in pairs]); zb = np.array([L[j][0] + 1j*L[j][1] for _, j in pairs])
    ca, cb = za.mean(), zb.mean()
    sim = np.vdot(za - ca, zb - cb) / np.vdot(za - ca, za - ca)
    res = float(np.abs((za - ca)*sim + cb - zb).max())
    return pairs, (sim, ca, cb, res)


def global_map(ref, glob):
    sim, ca, cb = glob[0], glob[1], glob[2]
    out = ref.copy()
    low = np.flatnonzero(ref[:, 2] < DRAW_Z)
    t1 = int(low[0]) if len(low) else len(ref)
    for t in range(len(ref)):
        z = ref[t, 0] + 1j*ref[t, 1]
        zz = cb + (z - ca)*sim
        w = min(1.0, t / max(1, t1))
        zz = z + w*(zz - z)
        out[t, 0], out[t, 1] = zz.real, zz.imag
    return out


def stick_tip_offset(frame):
    """Lowest stick point (wrist depth) minus TCP z at the reset pose; ~0 when the stick is nominal."""
    from PIL import Image
    files = {(f["camera"], f["kind"]): f for f in frame["files"]}
    df = files[("wrist", "depth")]
    d = np.array(Image.open(WS/df["path"])).astype(float)*float(df.get("meters_per_unit", 0.001))
    P = backproject(d, frame["camera_intrinsics"]["wrist"], frame["camera_poses"]["wrist"])
    close = (d > 0) & (d < 0.25)
    if close.sum() < 20:
        return None
    return float(P[close][:, 2].min() - frame["state"][2])


def main(ctx):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m = meta(ctx.call("robomme_observe"))
    log("start step", m.get("step_id"), "instr:", m.get("instruction"))
    limit = m.get("episode_limits", {}).get("remaining_episode_steps", 1190)
    # ---- P0 load video ----
    ref = None
    try:
        frames = find_video_frames(m)
        ref, rots, vid = build_reference(frames)
        log("video frames", len(vid), "ref len", len(ref))
    except Exception:
        log("P0 failed", traceback.format_exc())
    if ref is None or len(ref) < 10:
        log("no reference; holding")
        return
    R_goal = rotvec_to_R(np.array(vid[len(vid)//2]["state"][3:6]))
    # ---- P1 perceive discs and map ----
    dl, masks = [], []
    try:
        rgb0, d0 = load_frame_images(vid[0])
        dv = detect_discs(rgb0, d0, vid[0]["camera_intrinsics"]["front"], vid[0]["camera_poses"]["front"])
        live = [f for f in frames if f.get("phase") == "execution"]
        lf = live[-1]
        rgb1, d1 = load_frame_images(lf)
        dl = detect_discs(rgb1, d1, lf["camera_intrinsics"]["front"], lf["camera_poses"]["front"])
        masks = disc_pixel_masks(rgb1, d1, lf["camera_intrinsics"]["front"], lf["camera_poses"]["front"], dl)
        log("discs video", np.round(dv, 3).tolist()); log("discs live", np.round(dl, 3).tolist())
        try:
            off = stick_tip_offset(lf); off_v = stick_tip_offset(vid[0])
            log("stick tip offset live %s video %s%s" % (None if off is None else round(off, 4),
                None if off_v is None else round(off_v, 4),
                "  WARNING stick displaced (tip above TCP)" if off is not None and off > 0.02 else ""))
        except Exception:
            log("stick offset check failed", traceback.format_exc())
        V = [d[:2] for d in dv]; L = [d[:2] for d in dl]
        corr, glob = correspond(V, L)
        if corr is None:
            pc = partial_correspond(V, L)
            if pc is None:
                log("disc correspondence unavailable (counts %d/%d): identity map" % (len(V), len(L)))
            else:
                pairs, g2 = pc
                log("partial correspondence", pairs, "|s|=%.4f residual %.4f" % (abs(g2[0]), g2[3]))
                if g2[3] < 0.02 and 0.8 < abs(g2[0]) < 1.25:
                    ref = global_map(ref, g2)
                    log("applied global similarity map")
                else:
                    log("partial fit implausible: identity map")
        else:
            shift = max(np.linalg.norm(np.array(V[i]) - np.array(L[corr[i]])) for i in corr)
            log("disc correspondence", corr, "max shift %.4f, global similarity |s|=%.4f arg=%.4f residual %.4f"
                % (shift, abs(glob[0]), np.angle(glob[0]), glob[3]))
            if glob[3] > 0.04 or not (0.8 < abs(glob[0]) < 1.25):
                log("correspondence implausible: identity map")
            elif shift < 0.004:
                log("scene unchanged: identity map")
            else:
                ref, visits = map_reference(ref, V, L, corr, glob)
                log("mapped reference piecewise over visits", [(v[0], v[1]) for v in visits])
    except Exception:
        log("P1 failed", traceback.format_exc())
    discs_xy = np.array([d[:2] for d in dl]) if dl else np.zeros((0, 2))
    lab = label_dwells(ref, discs_xy)
    n = len(ref)
    # dwell ends: last frame of each dwell run
    dwell_end = set(i for i in range(n-1) if lab[i] >= 0 and lab[i+1] != lab[i])
    if lab[n-1] >= 0:
        dwell_end.add(n-1)
    run_end = np.arange(n)
    for i in range(n-2, -1, -1):
        if lab[i] >= 0 and lab[i+1] == lab[i]:
            run_end[i] = run_end[i+1]
    run_len = {}
    for i in range(n):
        if lab[i] >= 0:
            run_len[int(run_end[i])] = run_len.get(int(run_end[i]), 0) + 1
    # only real stops (>= MIN_STOP_FRAMES at the disc) need the red-light confirmation
    lab = lab.copy()
    for i in range(n):
        if lab[i] >= 0 and run_len.get(int(run_end[i]), 0) < MIN_STOP_FRAMES:
            lab[i] = -1
    dwell_end = set(e for e in dwell_end if lab[e] >= 0)
    route = [int(lab[i]) for i in sorted(dwell_end)]
    log("dwell runs (end, len):", sorted(run_len.items()))
    log("ref len", n, "limit", limit, "dwell route (disc order)", route)
    np.save(OUT/"reference.npy", ref)
    # ---- P3 track ----
    k = 0; step = m.get("step_id", 0); trace = []; ended = False
    p = np.array(m["state"][:3]); rv = np.array(m["state"][3:6])
    dwell_count = 0; extra = 0; red_seen = {}
    complete = False
    lit_prev = -1
    while not ended:
        try:
            kk = min(k, n-1)
            in_dwell = lab[kk] >= 0
            tgt = ref[kk] if (in_dwell and kk in dwell_end) else ref[min(k+LOOKAHEAD, n-1)]
            dpos = np.clip((tgt - p)*KP/STEP_SCALE, -1, 1)
            drot = np.clip(R_to_rotvec(R_goal @ rotvec_to_R(rv).T)*KR/ROT_SCALE, -1, 1)
            row = [float(x) for x in dpos] + [float(x) for x in drot] + [0.0]
            reply = ctx.call("robomme_step", actions=[row])
            r = meta(reply)
            step = r.get("step_id", step+1)
            p = np.array(r["state"][:3]); rv = np.array(r["state"][3:6])
            tr = r.get("transition", {})
            ended = bool(r.get("episode_ended")) or bool(tr.get("episode_ended"))
            if r.get("task_complete") or tr.get("task_complete"):
                complete = True
            steps_left = r.get("episode_limits", {}).get("remaining_episode_steps", limit-step)
            # red-light perception
            lit = -1; wrist_red = 0
            try:
                ims = decode_images(reply)
                if ims:
                    rm = red_mask(ims[0])
                    best = 0
                    for i, mk in enumerate(masks):
                        c = int((rm & mk).sum())
                        if c > max(6, 0.2*mk.sum()) and c > best:
                            best, lit = c, i
                    if len(ims) > 1:
                        wrist_red = int(red_mask(ims[1]).sum())
            except Exception:
                log("perception error", traceback.format_exc())
            if lit != lit_prev:
                log("step", step, "front-lit disc", lit, "stick", np.round(p, 3).tolist())
                lit_prev = lit
            trace.append([step, k] + p.tolist() + [lit, wrist_red])
            # progress / advancement
            win = ref[kk:min(kk+LOOKAHEAD+2, n)]
            e_now = float(np.min(np.linalg.norm(win - p, axis=1)))
            lowphase = ref[kk, 2] < DRAW_Z
            behind = steps_left < (n - k) + 15
            advance = True
            if lowphase and e_now > LAG_WAIT_M and not behind:
                advance = False
            if in_dwell:
                if np.linalg.norm(ref[run_end[kk]] - p) < ARRIVE_TOL:
                    dwell_count += 1
                if e_now > ARRIVE_TOL and not behind:
                    advance = False
                if kk in dwell_end and advance:
                    d_id = int(lab[kk])
                    confirmed = (lit == d_id) or (wrist_red >= 20 and dwell_count >= RED_MIN_DWELL)
                    if not confirmed and extra < MAX_EXTRA_HOLD and not behind:
                        advance = False; extra += 1
                    else:
                        log("dwell end step", step, "disc", d_id, "dwell", dwell_count, "extra", extra,
                            "confirmed", confirmed, "lit", lit, "wrist_red", wrist_red)
                        red_seen[len(red_seen)] = (d_id, bool(confirmed))
                        dwell_count = 0; extra = 0
            else:
                dwell_count = 0; extra = 0
            if advance:
                k += 1
                if behind and not in_dwell and k < n-1 and lab[min(k, n-1)] >= 0:
                    k += 1  # compress dwells when short of steps
            if step % 50 == 0:
                log("step", step, "k", k, "/", n, "pos", np.round(p, 3).tolist(), "err %.4f" % e_now,
                    "left", steps_left, "complete", complete)
            if complete:
                log("task_complete reported at step", step)
            if k >= n:
                log("reference finished at step", step, "-> final hold")
                break
        except Exception:
            log("step error", traceback.format_exc())
            try:
                st = meta(ctx.call("robomme_status"))
                if st.get("episode_ended") or st.get("finished"):
                    break
            except Exception:
                break
            k += 1
            if k > n + FINAL_HOLD:
                break
    # ---- P4 final hold: keep the stick at the last point until the episode ends ----
    hold_t0 = time.time()
    while not ended and time.time() - hold_t0 < 240:
        try:
            drot = np.clip(R_to_rotvec(R_goal @ rotvec_to_R(rv).T)*KR/ROT_SCALE, -1, 1)
            dpos = np.clip((ref[n-1] - p)*KP/STEP_SCALE, -1, 1)
            row = [float(x) for x in dpos] + [float(x) for x in drot] + [0.0]
            r = meta(ctx.call("robomme_step", actions=[row]))
            step = r.get("step_id", step+1)
            p = np.array(r["state"][:3]); rv = np.array(r["state"][3:6])
            tr = r.get("transition", {})
            ended = bool(r.get("episode_ended")) or bool(tr.get("episode_ended"))
            if r.get("task_complete") or tr.get("task_complete"):
                complete = True
            if step % 50 == 0:
                log("hold step", step, "pos", np.round(p, 3).tolist(), "complete", complete, "ended", ended)
        except Exception:
            log("hold error", traceback.format_exc())
            break
    try:
        np.save(OUT/"trace.npy", np.array(trace, float))
        (OUT/"summary.json").write_text(json.dumps({"version": VERSION, "steps": step, "ended": ended,
            "task_complete": complete, "route": route, "dwells": red_seen, "wall_s": time.time()-t0}))
    except Exception:
        log("save error", traceback.format_exc())
    log("done; wall %.1fs steps %s ended %s complete %s" % (time.time()-t0, step, ended, complete))
