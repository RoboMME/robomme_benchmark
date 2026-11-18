"""InsertPeg controller v1.

Pipeline (transition table):
  LOAD_VIDEO  -> TRACK_TO_GRASP (fail: abort-hold)
  TRACK_TO_GRASP -> CLOSE -> CHECK_GRASP (fail: REGRASP up to 2x, then continue anyway)
  CHECK_GRASP -> TRACK_LIFT_AND_CARRY -> MEASURE_TIP (high pose, after lift)
  MEASURE_TIP -> GOTO_PREINSERT (1 cm before the expert's slow approach start)
  GOTO_PREINSERT -> ALIGN (peg line == video peg line, iterative) -> INSERT_SEARCH
  INSERT_SEARCH -> HOLD -> return
Every phase is bounded by step counts; the remaining-steps budget is checked before each.
The expert video is the current episode's conditioning video (same layout as execution):
it tells which peg, which end (grasp pose) and which side (insertion line).
"""
import json
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hole as H  # noqa: E402

VERSION = "v3"
WS = Path("/workspace")
OUT = WS / "output"
POS_SCALE = 0.05
ROT_SCALE = 0.5

# ---- tunables (tested ranges: ep1 only) ----
MAX_POS_CMD = 0.8          # action units (0.04 m/step) in free space
MAX_ROT_CMD = 0.2          # action units (0.1 rad/step); 1.0 caused an IK jump in ep1
GRASP_TOL_P = 0.0015       # m, converge before closing (ep1: 0.9 mm reached)
CLOSE_STEPS = 8            # gripper closes in ~2 steps (video)
PREINSERT_BACK = 0.005     # m back from the expert's insertion start (head ~1 cm before face there)
HOLE_S_RANGE = (0.008, 0.022) # face distance ahead of peg tip where the hole measurement is trusted
REF_FRAMES = 12            # video frames after insertion start used for the reference peg line
PUSH_STEP = 0.0008         # m per step during insertion
PUSH_MAX_LEAD = 0.0025      # m commanded ahead of measured progress
STALL_STEPS = 3
SEARCH_STEP = 0.0007       # m
END_MARGIN = 0.004         # m deeper than the expert's head end
FINGER_ROW = 200
TOP_ROW = 56


def log(msg):
    line = "[%s %.1f] %s" % (VERSION, time.time() % 10000, msg)
    print(line, flush=True)
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / "controller_log.txt", "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def meta(reply):
    for b in reply.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                pass
    return {}


# ---------------- geometry ----------------

def pose_mat(state):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(np.asarray(state[3:6], float)).as_matrix()
    T[:3, 3] = state[:3]
    return T


def mat_state(T, grip_vals=(0.0, 0.0)):
    return list(T[:3, 3]) + list(R.from_matrix(T[:3, :3]).as_rotvec()) + list(grip_vals)


def row_to(cur, tgt, grip, max_pos=MAX_POS_CMD, max_rot=MAX_ROT_CMD):
    cur = np.asarray(cur, float); tgt = np.asarray(tgt, float)
    dp = (tgt[:3] - cur[:3]) / POS_SCALE
    n = np.linalg.norm(dp)
    if n > max_pos:
        dp *= max_pos / n
    dr = (R.from_rotvec(tgt[3:6]) * R.from_rotvec(cur[3:6]).inv()).as_rotvec() / ROT_SCALE
    n = np.linalg.norm(dr)
    if n > max_rot:
        dr *= max_rot / n
    return [float(x) for x in np.concatenate([np.clip(dp, -1, 1), np.clip(dr, -1, 1), [grip]])]


def pose_err(cur, tgt):
    cur = np.asarray(cur, float); tgt = np.asarray(tgt, float)
    return (float(np.linalg.norm(tgt[:3] - cur[:3])),
            float((R.from_rotvec(tgt[3:6]) * R.from_rotvec(cur[3:6]).inv()).magnitude()))


def backproject(depth, K, pose):
    h, w = depth.shape
    vs, us = np.mgrid[0:h, 0:w]
    d = depth.astype(float) * 0.001
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    P = np.stack([(us - cx) / fx * d, -(vs - cy) / fy * d, -d], -1)
    qw, qx, qy, qz = pose["quaternion_wxyz"]
    Rm = R.from_quat([qx, qy, qz, qw]).as_matrix()
    return P @ Rm.T + np.asarray(pose["position_m"]), depth > 0


def load_depth(frame, cam="wrist"):
    from PIL import Image
    for f in frame.get("files", []):
        if f.get("camera") == cam and f.get("kind") == "depth":
            return np.array(Image.open(WS / f["path"]))
    return None


def peg_line(frame, ee_state, u, top_row=TOP_ROW, finger_row=FINGER_ROW):
    """Held peg top-surface line in world frame from the wrist depth of `frame`.
    Returns dict(p=centre point xy, a=unit dir xy (sign along u), z=top z, width, tip=farthest pt along u)."""
    d = load_depth(frame)
    if d is None:
        return None
    P, valid = backproject(d, frame["camera_intrinsics"]["wrist"]["matrix"], frame["camera_poses"]["wrist"])
    ee = np.asarray(ee_state[:3], float)
    u = np.asarray(u[:2], float); u = u / np.linalg.norm(u)
    v = np.array([-u[1], u[0]])
    rows = np.arange(d.shape[0])[:, None] * np.ones((1, d.shape[1]))
    rel = P[..., :2] - ee[:2]
    along = rel @ u; lat = rel @ v
    zt = P[..., 2] - ee[2]
    m = valid & (zt > 0.003) & (zt < 0.035) & (rows < finger_row) & (rows >= top_row) & (np.abs(lat) < 0.04) & (np.abs(along) < 0.15)
    if m.sum() < 300:
        return None
    zs = P[..., 2][m]
    hist, edges = np.histogram(zs, bins=np.arange(zs.min(), zs.max() + 0.002, 0.001))
    zmode = edges[np.argmax(hist)] + 0.0005
    m &= np.abs(P[..., 2] - zmode) < 0.0025
    latm = np.median(lat[m])
    m &= np.abs(lat - latm) < 0.015
    Q = P[m]
    if len(Q) < 300:
        return None
    zmed = float(np.median(Q[:, 2]))
    Q = Q[np.abs(Q[:, 2] - zmed) < 0.0025]
    c = Q[:, :2].mean(0); X = Q[:, :2] - c
    w_, V = np.linalg.eigh(X.T @ X); a = V[:, -1]
    if a @ u < 0:
        a = -a
    nrm = np.array([-a[1], a[0]])
    s = X @ nrm; t = X @ a
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)
    p = c + nrm * (lo + hi) / 2
    tip = p + a * float(np.percentile(t, 99.5))
    return dict(p=p, a=a, z=zmed, width=float(hi - lo), tip=tip, n=int(len(Q)))


def lateral_err(live, ref):
    """Signed lateral offset (ref - live) at the live tip, along ref's normal; angle ref - live."""
    a = ref["a"]; nrm = np.array([-a[1], a[0]])
    dlat = float((ref["p"] - live["tip"]) @ nrm)
    th = float(np.arctan2(a[1], a[0]) - np.arctan2(live["a"][1], live["a"][0]))
    th = (th + np.pi) % (2 * np.pi) - np.pi
    return dlat, th, float(ref["z"] - live["z"]), nrm


def world_correct(state, pivot_xy, dth, dxy, dz):
    """Rotate EE pose about vertical axis through pivot by dth, then translate (dxy, dz)."""
    T = pose_mat(state)
    Rz = R.from_rotvec([0, 0, dth]).as_matrix()
    piv = np.array([pivot_xy[0], pivot_xy[1], T[2, 3]])
    M = np.eye(4); M[:3, :3] = Rz; M[:3, 3] = piv - Rz @ piv + np.array([dxy[0], dxy[1], dz])
    return mat_state(M @ T, state[6:8])


# ---------------- robot wrapper ----------------

class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.state = None; self.ended = False; self.complete = False
        self.remaining = 10 ** 6; self.step_id = 0; self.last = {}; self.last_frame = None

    def _upd(self, m):
        if not m:
            return
        self.last = m
        if m.get("state"):
            self.state = list(m["state"])
        self.step_id = m.get("step_id", self.step_id)
        if m.get("episode_ended") or (m.get("transition") or {}).get("episode_ended"):
            self.ended = True
        if m.get("task_complete") or (m.get("transition") or {}).get("task_complete"):
            self.complete = True
        lim = m.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]

    def observe(self):
        m = meta(self.ctx.call("robomme_observe")); self._upd(m); return m

    def step(self, rows):
        if self.ended or self.remaining <= 0:
            self.ended = self.ended or self.remaining <= 0
            return {}
        rows = rows[: max(1, int(self.remaining))]
        m = meta(self.ctx.call("robomme_step", actions=rows)); self._upd(m)
        self.last_frame = None
        fs = m.get("frame_sequence")
        if fs:
            self._fs = fs["manifest_path"]
        return m

    def frame(self):
        """Latest saved frame (wrist depth etc.) from the last step reply."""
        try:
            man = json.loads((WS / self._fs).read_text())
            return sorted(man["frames"], key=lambda f: f["frame_index"])[-1]
        except Exception as e:
            log("frame() failed: %r" % (e,))
            return None

    def goto(self, tgt, grip, tol_p=0.002, tol_r=0.01, max_steps=40, max_pos=MAX_POS_CMD):
        for k in range(max_steps):
            pe, re = pose_err(self.state, tgt)
            if pe < tol_p and re < tol_r:
                return True
            if self.ended:
                return False
            self.step([row_to(self.state, tgt, grip, max_pos=max_pos)])
        pe, re = pose_err(self.state, tgt)
        return pe < tol_p and re < tol_r

    def hold(self, grip, n):
        tgt = list(self.state)
        for _ in range(n):
            if self.ended:
                return
            self.step([row_to(self.state, tgt, grip)])


# ---------------- video analysis ----------------

def load_video(manifest_path):
    man = json.loads((WS / manifest_path).read_text())
    frames = sorted(man["frames"], key=lambda f: f["frame_index"])
    vid = [f for f in frames if f.get("phase") == "video_demo" and f.get("state")]
    return vid


def dense_traj(vid):
    """Per-control-step states from (possibly subsampled) video frames, with map back to frames."""
    S, owner = [], []
    for i in range(len(vid)):
        s0 = np.asarray(vid[i]["state"], float)
        if i + 1 < len(vid):
            s1 = np.asarray(vid[i + 1]["state"], float)
            n = int(max(1, (vid[i + 1].get("source_frame_index", i + 1) - vid[i].get("source_frame_index", i))))
        else:
            s1, n = s0, 1
        r0, r1 = R.from_rotvec(s0[3:6]), R.from_rotvec(s1[3:6])
        drot = (r1 * r0.inv()).as_rotvec()
        for k in range(n):
            f = k / n
            st = np.concatenate([s0[:3] + f * (s1[:3] - s0[:3]),
                                 (R.from_rotvec(f * drot) * r0).as_rotvec(),
                                 s0[6:8] + f * (s1[6:8] - s0[6:8])])
            S.append(st); owner.append(i)
    return np.array(S), owner


def analyse(S):
    g = S[:, 6] + S[:, 7]
    open_g = g.max()
    closing = np.nonzero(g < open_g - 0.004)[0]
    if len(closing) == 0:
        raise RuntimeError("video gripper never closes")
    k_close = int(closing[0])
    # fully closed: finger sum stabilises
    k_closed = k_close
    for k in range(k_close, min(len(S) - 1, k_close + 60)):
        if abs(g[k + 1] - g[k]) < 2e-4 and g[k] < open_g - 0.01:
            k_closed = k; break
    held_g = float(np.median(g[k_closed:k_closed + 40]))
    v = np.r_[np.linalg.norm(np.diff(S[:, :3], axis=0), axis=1), 0.0]
    moving = np.nonzero(v > 1e-4)[0]
    k_end = int(moving[-1] + 1) if len(moving) else len(S) - 1
    # final insertion = last constant-orientation straight segment ending at k_end
    r_end = R.from_rotvec(S[k_end, 3:6])
    k_ins = k_end
    while k_ins > k_closed + 1 and (R.from_rotvec(S[k_ins - 1, 3:6]) * r_end.inv()).magnitude() < np.radians(0.15) \
            and abs(S[k_ins - 1, 2] - S[k_end, 2]) < 0.003:
        k_ins -= 1
    # skip leading near-stationary frames (refinement settle)
    while k_ins < k_end - 5 and np.linalg.norm(S[k_end, :3] - S[k_ins, :3]) > 0 and v[k_ins] < 1e-4:
        k_ins += 1
    L0 = np.linalg.norm(S[k_end, :2] - S[k_ins, :2])
    k_u = k_ins
    while k_u < k_end and np.linalg.norm(S[k_end, :2] - S[k_u, :2]) > 0.85 * L0:
        k_u += 1
    u = S[k_end, :2] - S[k_u, :2]
    if np.linalg.norm(u) < 0.01:
        raise RuntimeError("no horizontal insertion segment")
    u = u / np.linalg.norm(u)
    k_hs = k_ins
    while k_hs > k_closed and abs(S[k_hs - 1, 2] - S[k_end, 2]) < 0.004:
        k_hs -= 1
    k_pre = k_ins
    return dict(k_close=k_close, k_closed=k_closed, held_g=held_g, k_end=k_end, k_hs=k_hs,
                k_pre=k_pre, k_ins=k_ins, u=u, open_g=float(open_g),
                ins_len=float(np.linalg.norm(S[k_end, :2] - S[k_ins, :2])))


# ---------------- main pipeline ----------------

def video_ref_line(vid, owner, A, u):
    """Median peg line over clean video frames in the horizontal insertion segment before the push."""
    lines = []
    fr_ids = sorted(set(owner[k] for k in range(A["k_ins"], min(A["k_ins"] + REF_FRAMES, A["k_end"]) + 1)))
    for i in fr_ids:
        fr = vid[i]
        try:
            L = peg_line(fr, fr["state"], u)
        except Exception:
            L = None
        if L is not None:
            lines.append((i, L))
    if not lines:
        return None, []
    w0 = np.median([L["width"] for _, L in lines])
    good = [(i, L) for i, L in lines if abs(L["width"] - w0) < 0.15 * w0]
    # prefer frames up to the push (pre-insert alignment by the expert)
    pre = good
    ang = np.median([np.arctan2(L["a"][1], L["a"][0]) for _, L in pre])
    a = np.array([np.cos(ang), np.sin(ang)])
    nrm = np.array([-a[1], a[0]])
    # lateral offsets relative to first line point
    p0 = pre[0][1]["p"]
    lat = np.median([(L["p"] - p0) @ nrm for _, L in pre])
    ref = dict(p=p0 + nrm * lat, a=a, z=float(np.median([L["z"] for _, L in pre])),
               width=float(w0))
    return ref, [(i, round(L["width"], 4), round(float((L["p"] - p0) @ nrm), 4),
                  round(float(np.degrees(np.arctan2(L["a"][1], L["a"][0]))), 2), round(L["z"], 4)) for i, L in pre]


def measure(frame, u):
    """Peg line + hole on the face from one wrist frame. Returns dict or None, reason."""
    if frame is None:
        return None, "no frame"
    L = peg_line(frame, frame["state"], u)
    if L is None:
        return None, "no peg line"
    d = load_depth(frame)
    P, val = backproject(d, frame["camera_intrinsics"]["wrist"]["matrix"], frame["camera_poses"]["wrist"])
    h, why = H.hole_measure(P, val, frame["camera_poses"]["wrist"]["position_m"], u, L["tip"])
    if h is None:
        return None, why
    if not h["open"]:
        return dict(ok=False, face_s=h["face_s"], L=L, h=h), why
    vv = np.array([-u[1], u[0]])
    peg_lat = float((L["p"] - L["tip"]) @ vv); peg_zc = L["z"] - L["width"] / 2
    out = dict(L=L, h=h, dlat=h["lat_c"] - peg_lat, dz=h["z_c"] - peg_zc, face_s=h["face_s"],
               ok=(0.014 < h["width"] < 0.035 and 0.014 < h["height"] < 0.04 and h["n_open"] > 150))
    return out, ("ok" if out["ok"] else "implausible hole w %.4f h %.4f n %d" % (h["width"], h["height"], h["n_open"]))


def video_hole_calib(vid, owner, A, u):
    """Hole-vs-peg offsets the expert had when correctly aligned (measurement bias at this range)."""
    ids = sorted(set(owner[k] for k in range(max(A["k_hs"], A["k_ins"] - 15), min(A["k_ins"] + REF_FRAMES, A["k_end"]) + 1)))
    rows = []
    for i in ids:
        try:
            M, why = measure(vid[i], u)
        except Exception as e:
            M, why = None, repr(e)
        if M is not None and M["ok"] and HOLE_S_RANGE[0] <= M["face_s"] <= HOLE_S_RANGE[1]:
            rows.append((i, M["dlat"], M["dz"], M["face_s"], M["h"]["width"], M["h"]["height"]))
    if len(rows) < 2:
        return None, rows
    return dict(dlat=float(np.median([r[1] for r in rows])), dz=float(np.median([r[2] for r in rows])),
                face_s=float(np.median([r[3] for r in rows]))), rows


def run(ctx):
    rb = Robot(ctx)
    m0 = rb.observe()
    log("start step %s state %s" % (rb.step_id, np.round(rb.state, 4).tolist()))
    fs = m0.get("frame_sequence") or {}
    if not fs.get("manifest_path"):
        raise RuntimeError("no frame_sequence in initial observation")
    rb._fs = fs["manifest_path"]
    vid = load_video(fs["manifest_path"])
    log("video frames %d" % len(vid))
    S, owner = dense_traj(vid)
    A = analyse(S)
    u = A["u"]
    log("analysis %s" % {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v) for k, v in A.items()})
    ref, refdbg = video_ref_line(vid, owner, A, u)
    log("video ref line %s" % (None if ref is None else {k: np.round(v, 4).tolist() if isinstance(v, np.ndarray) else round(v, 4) for k, v in ref.items()}))
    log("ref frames %s" % refdbg[:30])
    calib, crow = video_hole_calib(vid, owner, A, u)
    log("video hole calib %s rows %s" % (calib, [tuple(round(float(x), 4) for x in r) for r in crow]))

    # 1. track to just before closing
    kc = A["k_close"]
    for k in range(1, kc):
        if rb.ended:
            return rb
        rb.step([row_to(rb.state, S[min(k + 1, kc - 1)], -1.0)])
    ok = rb.goto(S[kc - 1], -1.0, tol_p=GRASP_TOL_P, tol_r=0.01, max_steps=30, max_pos=0.3)
    log("pre-grasp converge ok=%s err=%s step=%d" % (ok, np.round(pose_err(rb.state, S[kc - 1]), 4).tolist(), rb.step_id))

    # 2. close and check
    for attempt in range(3):
        tgt = list(S[kc - 1])
        for _ in range(CLOSE_STEPS):
            rb.step([row_to(rb.state, tgt, 1.0, max_pos=0.3)])
        g = rb.state[6] + rb.state[7]
        log("grasp attempt %d finger sum %.4f (video held %.4f)" % (attempt, g, A["held_g"]))
        if g > 0.004 or attempt == 2:
            break
        # empty grasp: open, rise, re-approach
        rb.hold(-1.0, 5)
        up = list(S[kc - 1]); up[2] += 0.05
        rb.goto(up, -1.0, max_steps=20)
        rb.goto(S[kc - 1], -1.0, tol_p=GRASP_TOL_P, max_steps=30, max_pos=0.3)

    # 3. track lift/carry until the high point, measure tip; then to pre-insert
    k_hi = A["k_closed"] + int(np.argmax(S[A["k_closed"]:A["k_hs"], 2]))
    for k in range(A["k_closed"], k_hi + 1):
        if rb.ended:
            return rb
        rb.step([row_to(rb.state, S[min(k + 1, k_hi)], 1.0)])
    rb.goto(S[k_hi], 1.0, tol_p=0.002, tol_r=0.01, max_steps=20)
    rb.hold(1.0, 2)
    tip_delta = 0.0
    try:
        fr = rb.frame()
        Lh = peg_line(fr, fr["state"], u, top_row=0)
        vf = vid[owner[k_hi]]
        Lv = peg_line(vf, vf["state"], u, top_row=0)
        if Lh is not None and Lv is not None:
            ee_l = np.asarray(fr["state"][:2]); ee_v = np.asarray(vf["state"][:2])
            tip_delta = float((Lh["tip"] - ee_l) @ Lh["a"] - (Lv["tip"] - ee_v) @ Lv["a"])
            log("tip measure live %.4f video %.4f delta %.4f widths %.4f %.4f" % (
                (Lh["tip"] - ee_l) @ Lh["a"], (Lv["tip"] - ee_v) @ Lv["a"], tip_delta, Lh["width"], Lv["width"]))
            if abs(tip_delta) > 0.03:
                tip_delta = 0.0
    except Exception as e:
        log("tip measure failed %r" % (e,))

    # pre-insert target: the expert's insertion-start pose moved back along u
    for k in range(k_hi, A["k_hs"] + 1):
        if rb.ended:
            return rb
        rb.step([row_to(rb.state, S[min(k + 1, A["k_hs"])], 1.0)])
    pre = list(S[A["k_ins"]])
    pre[0] -= u[0] * PREINSERT_BACK; pre[1] -= u[1] * PREINSERT_BACK
    rb.goto(pre, 1.0, tol_p=0.001, tol_r=0.006, max_steps=40, max_pos=0.3)
    rb.goto(pre, 1.0, tol_p=0.0005, tol_r=0.004, max_steps=15, max_pos=0.1)
    log("at pre-insert err=%s step=%d g=%.4f" % (np.round(pose_err(rb.state, pre), 4).tolist(), rb.step_id, rb.state[6] + rb.state[7]))

    # 4-5. rounds of (align peg line to the video line, push with search)
    s_end_abs = float(S[A["k_end"], :2] @ u) - tip_delta + END_MARGIN   # target EE coordinate along u
    u3 = np.array([u[0], u[1], 0.0])
    start = list(pre)
    for rnd, scale in enumerate(SEARCH_ROUNDS):
        if rb.ended or rb.complete or rb.remaining < 40:
            break
        if rnd > 0:
            # back out to the pre-insert pose (rotation kept) before re-measuring
            back = list(start[:3]) + list(rb.state[3:6]) + [0, 0]
            rb.goto(back, 1.0, tol_p=0.001, tol_r=0.01, max_steps=25, max_pos=0.2)
        base = align(rb, ref, u, calib)
        res = push(rb, base, u3, s_end_abs, scale)
        log("round %d (scale %.2f) -> %s" % (rnd, scale, res))
        if res in ("depth", "complete", "deep_stall"):
            break
    rb.hold(1.0, 5)
    return rb


def align(rb, ref, u, calib):
    """Yaw: live peg line -> video peg line. Lateral/z: peg centre -> live hole centre (bias-corrected by the
    video's own aligned measurement); falls back to the video peg line if the hole is not measurable."""
    used_hole = False
    n_range = 0
    for it in range(8):
        rb.hold(1.0, 2)
        fr = rb.frame()
        L = peg_line(fr, fr["state"], u) if fr is not None else None
        if L is None or (ref is not None and abs(L["width"] - ref["width"]) > 0.15 * ref["width"]):
            log("align it %d: no clean peg line (%s)" % (it, None if L is None else round(L["width"], 4)))
            break
        dth = 0.0; dlat = dz = 0.0; nrm = np.array([-u[1], u[0]])
        if ref is not None:
            dlat_r, dth, dz_r, nrm_r = lateral_err(L, ref)
        M, why = (None, "no calib") if calib is None else measure(fr, u)
        if M is not None and not (HOLE_S_RANGE[0] <= M["face_s"] <= HOLE_S_RANGE[1]) and n_range < 3:
            # bring the face into the calibrated measuring range (move along u only)
            n_range += 1
            ds = float(np.clip(M["face_s"] - calib["face_s"], -0.03, 0.03))
            tgt = list(rb.state); tgt[0] += u[0] * ds; tgt[1] += u[1] * ds
            log("align it %d: face_s %.4f out of range -> move %.4f along u (%s)" % (it, M["face_s"], ds, why))
            rb.goto(tgt, 1.0, tol_p=0.0005, tol_r=0.003, max_steps=20, max_pos=0.2)
            continue
        if M is not None and M["ok"]:
            used_hole = True
            dlat = M["dlat"] - calib["dlat"]; dz = M["dz"] - calib["dz"]
            log("align it %d: HOLE dlat %.4f dz %.4f (raw %.4f %.4f face_s %.4f w %.4f h %.4f) dth %.3fdeg" % (
                it, dlat, dz, M["dlat"], M["dz"], M["face_s"], M["h"]["width"], M["h"]["height"], np.degrees(dth)))
        elif ref is not None and not used_hole:
            dlat, dz, nrm = dlat_r, dz_r, nrm_r
            log("align it %d: REF (hole: %s) dlat %.4f dz %.4f dth %.3fdeg" % (it, why, dlat, dz, np.degrees(dth)))
        else:
            log("align it %d: hole lost (%s); keep pose" % (it, why))
            break
        if abs(dlat) < 0.0004 and abs(dz) < 0.0004 and abs(dth) < np.radians(0.15):
            break
        dth = float(np.clip(dth, -np.radians(6), np.radians(6)))
        dlat = float(np.clip(dlat, -0.025, 0.025)); dz = float(np.clip(dz, -0.02, 0.02))
        tgt = world_correct(rb.state, L["tip"], dth, nrm * dlat, dz)
        rb.goto(tgt, 1.0, tol_p=0.0004, tol_r=0.002, max_steps=25, max_pos=0.2)
    return list(rb.state)


OFFSETS = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (0, 2), (1, 1), (-1, 1), (2, 0), (-2, 0),
           (0, -2), (1, -1), (-1, -1), (0, 3), (2, 2), (-2, 2), (3, 0), (-3, 0), (0, -3), (2, -2), (-2, -2), (0, 4)]
SEARCH_ROUNDS = (1.0, 1.7, 0.5)


def push(rb, base, a3, s_end_abs, scale):
    nrm3 = np.array([-a3[1], a3[0], 0.0]); ez = np.array([0, 0, 1.0])
    B = np.asarray(base[:3], float); rot = list(base[3:6])
    s_end = s_end_abs - float(B @ a3)
    log("push: base %s s_end %.4f scale %.2f step %d rem %d" % (np.round(B, 4).tolist(), s_end, scale, rb.step_id, rb.remaining))
    oi = 0; s_contact = None; stall = 0; inside = False
    sm = lambda: float((np.asarray(rb.state[:3]) - B) @ a3)
    s_cmd = sm()
    def offs(i):
        return nrm3 * OFFSETS[i][0] * SEARCH_STEP * scale + ez * OFFSETS[i][1] * SEARCH_STEP * scale
    while True:
        if rb.complete:
            return "complete"
        if rb.ended or rb.remaining <= 8:
            return "complete" if rb.complete else "budget"
        s_meas = sm()
        if s_meas >= s_end:
            log("reached depth s=%.4f step %d" % (s_meas, rb.step_id)); return "depth"
        s_cmd = min(max(s_cmd, s_meas) + PUSH_STEP, s_meas + PUSH_MAX_LEAD, s_end + 0.002)
        tgt = list(B + a3 * s_cmd + offs(oi)) + rot + [0, 0]
        rb.step([row_to(rb.state, tgt, 1.0, max_pos=0.3)])
        s_new = sm()
        if s_contact is not None and s_new > s_contact + 0.008 and not inside:
            inside = True
            log("inside: s=%.4f contact %.4f offset %s step %d" % (s_new, s_contact, OFFSETS[oi], rb.step_id))
        if s_cmd - s_new > 0.6 * PUSH_MAX_LEAD and s_new - s_meas < 0.0003:
            stall += 1
        else:
            stall = 0
        if stall < STALL_STEPS:
            continue
        stall = 0
        if s_contact is None:
            s_contact = s_new
            log("first contact s=%.4f step %d z=%.4f" % (s_new, rb.step_id, rb.state[2]))
        if inside and s_new >= s_end - 0.012:
            log("stall deep inside s=%.4f; stop" % s_new); return "deep_stall"
        oi += 1
        if oi >= len(OFFSETS):
            log("search exhausted"); return "exhausted"
        back = list(B + a3 * (s_new - 0.003) + offs(oi)) + rot + [0, 0]
        rb.goto(back, 1.0, tol_p=0.0005, tol_r=0.004, max_steps=8, max_pos=0.2)
        s_cmd = sm()
        log("stall at s=%.4f z=%.4f -> offset %d %s step %d rem %d" % (s_new, rb.state[2], oi, OFFSETS[oi], rb.step_id, rb.remaining))


def main(ctx):
    t0 = time.time()
    try:
        rb = run(ctx)
        log("finished: step %s ended %s complete %s t=%.1fs" % (rb.step_id, rb.ended, rb.complete, time.time() - t0))
    except Exception:
        log("EXCEPTION " + traceback.format_exc())
