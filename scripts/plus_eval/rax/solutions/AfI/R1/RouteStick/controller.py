"""RouteStick controller (RAX, AfI, R1) — written offline, no exploration was possible.

Strategy
--------
The conditioning video of the current episode shows the stick route that must be
repeated.  We extract the stick-TCP path shown in that video and follow it in the
live execution with a closed-loop, time-parameterised path follower.

Path sources, in order of preference (first that validates wins):
  A. per-frame `state` recorded in the conditioning-video manifest (EE pose of the
     stick TCP in world metres)                                   -> traj_from_states
  B. visual tracking of the lowest moving robot point in the front-camera depth
     video, back-projected with a linear camera model fitted against
     robomme_pixel_to_position on the live frame, offset-calibrated so that the
     live frame's estimate matches the live measured TCP              -> traj_from_images
  If no path validates, the controller holds the pose (safe no-op) and returns.

Transition table
----------------
  INIT      -> PERCEIVE (always; on error -> HOLD_END)
  PERCEIVE  -> PLAN (path found) | HOLD_END (no path)
  PLAN      -> APPROACH (start far from current pose) | FOLLOW
  APPROACH  -> FOLLOW (reached / timeout)        [episode end -> DONE]
  FOLLOW    -> SETTLE (plan exhausted / step-or-wall budget low) [episode end -> DONE]
  SETTLE    -> DONE
  HOLD_END  -> DONE
Every robot call is wrapped; >MAX_CONSEC_ERRORS consecutive tool errors -> DONE.
"""
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

VERSION = "routestick-v1"

# ---- tunables (none tested live: R1 budget allowed no exploration) -------------
POS_SCALE = 0.05          # metres per unit action (protocol)
ROT_SCALE = 0.5           # radians per unit action (protocol)
V_MAX = 0.012             # m per control step for normal following (0.24 m/s)
V_HARD = 0.025            # absolute max m/step when the step budget forces speed-up
PAUSE_MAX_STEPS = 4       # internal pauses of the demo are compressed to this
MIN_MOVE = 0.0015         # m; consecutive path samples closer than this are merged
STEP_BUDGET_FRACTION = 0.85
CTRL_WALL_BUDGET_S = 1000.0   # controller wall budget (RUNTIME: 1200 s hard)
CHUNK_MIN, CHUNK_MAX = 2, 8
HOLD_ERR = 0.025          # if tracking error exceeds this, stop advancing plan time
CORR_GAIN = 0.7           # fraction of position error corrected per chunk
ROT_GAIN = 0.5
ROT_MAX_STEP = 0.10       # rad per step
ROT_DEADBAND = 0.03       # rad; smaller orientation errors are not corrected
APPROACH_TOL = 0.004
FINAL_TOL = 0.004
MAX_CONSEC_ERRORS = 5
GRIPPER_CMD = 1.0         # stick tasks ignore it; closed keeps any held tool

OUT_DIR = Path("/workspace/output")
_T0 = time.time()
_LOG_FH = None


# ----------------------------------------------------------------------------- log
def log(msg, **kv):
    global _LOG_FH
    line = f"[{VERSION} t={time.time() - _T0:7.1f}s] {msg}"
    if kv:
        try:
            line += " " + json.dumps(kv, default=_jsonable)
        except Exception:
            line += " " + str(kv)
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        if _LOG_FH is None:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            _LOG_FH = open(OUT_DIR / "controller_log.txt", "a")
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()
    except Exception:
        pass


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return np.round(o, 4).tolist()
    if isinstance(o, (np.floating,)):
        return round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    return str(o)


def save_json(name, obj):
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        txt = json.dumps(obj, default=_jsonable)
        if len(txt) > 2_000_000:
            txt = txt[:2_000_000]
        with open(OUT_DIR / name, "w") as fh:
            fh.write(txt)
    except Exception as e:  # pragma: no cover
        log("save_json failed", name=name, err=repr(e))


# ------------------------------------------------------------------- MCP parsing
def parse_reply(result):
    """First text block of an MCP tool result, decoded as JSON (dict) or {}."""
    if isinstance(result, dict) and "content" in result:
        for block in result.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                txt = block.get("text", "")
                try:
                    val = json.loads(txt)
                    return val if isinstance(val, dict) else {"value": val}
                except Exception:
                    return {"text": txt}
        sc = result.get("structuredContent")
        if isinstance(sc, dict):
            return sc
        return {}
    if isinstance(result, dict):
        return result
    return {}


def find_key(obj, key, max_depth=6):
    """Breadth-first search for a key in nested dicts/lists."""
    frontier = [obj]
    for _ in range(max_depth):
        nxt = []
        for o in frontier:
            if isinstance(o, dict):
                if key in o:
                    return o[key]
                nxt.extend(v for v in o.values() if isinstance(v, (dict, list)))
            elif isinstance(o, list):
                nxt.extend(v for v in o[:64] if isinstance(v, (dict, list)))
        frontier = nxt
        if not frontier:
            break
    return None


def find_all(obj, key, max_depth=6):
    out, frontier = [], [obj]
    for _ in range(max_depth):
        nxt = []
        for o in frontier:
            if isinstance(o, dict):
                if key in o:
                    out.append(o[key])
                nxt.extend(v for k, v in o.items() if isinstance(v, (dict, list)) and k != key)
            elif isinstance(o, list):
                nxt.extend(v for v in o[:64] if isinstance(v, (dict, list)))
        frontier = nxt
        if not frontier:
            break
    return out


def _as_state(st):
    if isinstance(st, dict):
        for k in ("values", "vector", "data", "state"):
            if isinstance(st.get(k), list):
                st = st[k]
                break
    try:
        arr = np.asarray(st, dtype=float).reshape(-1)
        if 6 <= arr.size <= 16 and np.all(np.isfinite(arr[:6])):
            return arr
    except Exception:
        pass
    return None


def get_state(pkt):
    """First value under a 'state' key that parses as the 8-D robot state."""
    for st in find_all(pkt, "state"):
        arr = _as_state(st)
        if arr is not None:
            return arr
    return None


def is_ended(pkt):
    vals = find_all(pkt, "episode_ended")
    if vals:
        return any(v is True for v in vals[:1])
    for k in ("terminated", "truncated"):
        v = find_key(pkt, k)
        if v is True:
            return True
    return False


def is_success(pkt):
    vals = find_all(pkt, "task_complete")
    return bool(vals) and vals[0] is True


def remaining_steps_from(pkt):
    lim = find_key(pkt, "episode_limits")
    if not isinstance(lim, dict):
        lim = pkt if isinstance(pkt, dict) else {}
    for k in ("remaining_steps", "steps_remaining", "remaining"):
        v = lim.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    mx = None
    for k in ("max_steps", "step_limit", "max_episode_steps", "limit", "horizon"):
        v = lim.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            mx = int(v)
            break
    used = None
    for k in ("steps_used", "elapsed_steps", "step", "steps", "used", "step_id"):
        v = lim.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            used = int(v)
            break
    if mx is not None:
        if used is None:
            sid = find_key(pkt, "step_id")
            used = int(sid) if isinstance(sid, (int, float)) and not isinstance(sid, bool) else 0
        return max(0, mx - used)
    return None


# ------------------------------------------------------------------ rotations
def rotvec_to_mat(r):
    r = np.asarray(r, dtype=float)
    th = float(np.linalg.norm(r))
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def mat_to_rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: axis from the symmetric part
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        ax = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return ax / np.linalg.norm(ax) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def rot_err(r_des, r_cur):
    """World-frame rotation vector taking current orientation to desired."""
    return mat_to_rotvec(rotvec_to_mat(r_des) @ rotvec_to_mat(r_cur).T)


# --------------------------------------------------------------- manifest load
def resolve_path(p, manifest_dir=None):
    p = str(p)
    cands = [Path(p)] if os.path.isabs(p) else []
    cands += [Path("/workspace") / p, Path.cwd() / p]
    if manifest_dir is not None:
        cands.append(Path(manifest_dir) / p)
    for c in cands:
        if c.exists():
            return c
    return cands[0] if cands else Path(p)


def load_manifest(pkt):
    seq = find_key(pkt, "frame_sequence")
    mpath = None
    if isinstance(seq, dict):
        mpath = seq.get("manifest_path")
    if mpath is None:
        mpath = find_key(pkt, "manifest_path")
    if not mpath:
        return None, None
    path = resolve_path(mpath)
    with open(path) as fh:
        man = json.load(fh)
    return man, path.parent


def frame_records(man):
    frames = man.get("frames") if isinstance(man, dict) else None
    if not isinstance(frames, list):
        return [], []
    def key(f):
        fi = f.get("frame_index")
        return fi if isinstance(fi, (int, float)) else 0
    frames = sorted([f for f in frames if isinstance(f, dict)], key=key)
    video = [f for f in frames if f.get("phase") == "video_demo"]
    execf = [f for f in frames if f.get("phase") == "execution"]
    return video, execf


def frame_times(video, freq):
    """Seconds for each video frame (monotone)."""
    ts = []
    for i, f in enumerate(video):
        t = f.get("timestamp_s")
        if not isinstance(t, (int, float)):
            sfi = f.get("source_frame_index")
            t = (sfi / freq) if isinstance(sfi, (int, float)) else i / freq
        ts.append(float(t))
    ts = np.asarray(ts, dtype=float)
    for i in range(1, len(ts)):
        if ts[i] <= ts[i - 1]:
            ts[i] = ts[i - 1] + 1.0 / freq
    return ts


def frame_file(f, camera, kind):
    files = [e for e in (f.get("files") or []) if isinstance(e, dict) and e.get("path")]
    for e in files:
        if e.get("camera") == camera and e.get("kind") == kind:
            return e
    for e in files:
        if camera in str(e.get("camera", "")).lower() and e.get("kind") == kind:
            return e
    return None


# ------------------------------------------------------- path source A: states
def traj_from_states(video, ts, exec_state):
    """Return (P[n,3], R[n,3], t[n]) from per-frame states or (None, reason)."""
    P, Rv, T = [], [], []
    for f, t in zip(video, ts):
        a = get_state(f)
        if a is None:
            continue
        P.append(a[:3]); Rv.append(a[3:6]); T.append(t)
    if len(P) < 5:
        return None, f"only {len(P)} video states"
    P, Rv, T = np.array(P), np.array(Rv), np.array(T)
    spread = float(np.max(np.linalg.norm(P - P[0], axis=1)))
    if spread < 0.02:
        return None, f"video states nearly constant (spread {spread:.4f} m)"
    if np.any(np.abs(P[:, :2]) > 2.0) or np.any(np.abs(P[:, 2]) > 2.0):
        return None, "video states outside plausible workspace"
    # drop isolated outliers (speed > 1.5 m/s to both neighbours)
    keep = np.ones(len(P), bool)
    for i in range(1, len(P) - 1):
        v1 = np.linalg.norm(P[i] - P[i - 1]) / max(T[i] - T[i - 1], 1e-3)
        v2 = np.linalg.norm(P[i + 1] - P[i]) / max(T[i + 1] - T[i], 1e-3)
        if v1 > 1.5 and v2 > 1.5:
            keep[i] = False
    P, Rv, T = P[keep], Rv[keep], T[keep]
    info = {"n": int(len(P)), "spread": spread,
            "start": P[0], "end": P[-1],
            "zmin": float(P[:, 2].min()), "zmax": float(P[:, 2].max())}
    if exec_state is not None:
        info["start_to_exec_dist"] = float(np.linalg.norm(P[0] - exec_state[:3]))
    return (P, Rv, T, info), "ok"


# ------------------------------------------------------ path source B: images
def load_depth_m(entry, mdir):
    from PIL import Image
    p = resolve_path(entry["path"], mdir)
    with Image.open(p) as im:
        d = np.asarray(im).astype(np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    mpu = entry.get("meters_per_unit", 0.001) or 0.001
    d = d * float(mpu)
    d[d <= 0] = np.nan
    return d


def camera_center(pkt, camera="front"):
    """Camera position (world) from camera_poses in the observation packet, or None."""
    poses = find_key(pkt, "camera_poses")
    pose = None
    if isinstance(poses, dict):
        pose = poses.get(camera)
        if pose is None:
            for k, v in poses.items():
                if camera in str(k).lower():
                    pose = v
                    break
    elif isinstance(poses, list):
        for v in poses:
            if isinstance(v, dict) and camera in str(v.get("camera", v.get("name", ""))).lower():
                pose = v
                break
    if pose is None:
        return None
    if isinstance(pose, dict):
        for k, v in pose.items():
            if "pos" in str(k).lower() and isinstance(v, (list, tuple)) and len(v) == 3:
                return np.asarray(v, float)
        for v in pose.values():
            try:
                M = np.asarray(v, float)
                if M.shape == (4, 4):
                    return M[:3, 3]
            except Exception:
                pass
    try:
        M = np.asarray(pose, float)
        if M.shape == (4, 4):
            return M[:3, 3]
    except Exception:
        pass
    return None


_CALIB_CACHE = {}


def fit_linear_camera(ctx, step_id, depth, center):
    key = int(step_id)
    if key not in _CALIB_CACHE:
        _CALIB_CACHE[key] = _fit_linear_camera(ctx, step_id, depth, center)
    return _CALIB_CACHE[key]


def project_point(cam, swap, p):
    """World point -> (u, v, depth) for the fitted ray model (inverse of RayField)."""
    B, C = cam
    pc = np.linalg.solve(B, np.asarray(p, float) - C)
    if abs(pc[2]) < 1e-6:
        return None
    u, v = pc[0] / pc[2], pc[1] / pc[2]
    if swap:
        u, v = v, u
    return u, v, pc[2]


def states_vs_depth(ctx, video, execf, mdir, pkt, step_id, n_check=24):
    """Fraction of sampled video frames whose recorded TCP lies on/behind a nearby depth surface.

    If the recorded states really are the stick TCP during the video, the depth image of
    that frame shows the stick (or arm) at about the TCP's camera depth near its projection."""
    e_x = frame_file(execf[-1], "front", "depth") if execf else None
    if e_x is None:
        return None, "no execution depth"
    dx = load_depth_m(e_x, mdir)
    center = camera_center(pkt)
    if center is None:
        return None, "camera centre unknown (fit could be ill-conditioned); skip"
    cal, why = fit_linear_camera(ctx, step_id, dx, center)
    if cal is None or cal[1] > 0.01:
        return None, "calibration unavailable: " + str(why)
    cam, res, swap = cal
    idx = np.unique(np.linspace(0, len(video) - 1, min(n_check, len(video))).astype(int))
    samples = []
    for i in idx:
        f = video[i]
        st = get_state(f)
        e = frame_file(f, "front", "depth")
        if st is None or e is None:
            continue
        samples.append((st, load_depth_m(e, mdir)))
    if len(samples) < 5:
        return None, "too few depth frames"
    with np.errstate(all="ignore"):
        bg = np.nanmedian(np.stack([d for _, d in samples] + [dx]), axis=0)
    good = tot = 0
    for st, d in samples:
        pr = project_point(cam, swap, st[:3])
        if pr is None:
            continue
        u, v, dz = pr
        H, W = d.shape
        ui, vi = int(round(u)), int(round(v))
        tot += 1
        if not (0 <= ui < W and 0 <= vi < H):
            continue
        sl = (slice(max(0, vi - 4), vi + 5), slice(max(0, ui - 4), ui + 5))
        win, wbg = d[sl], bg[sl]
        with np.errstate(all="ignore"):
            moving = (wbg - win > 0.01) & (np.abs(win - dz) < 0.03)
        if np.any(moving):
            good += 1
    if tot == 0:
        return None, "nothing checked"
    return good / tot, f"{good}/{tot}"


def _fit_linear_camera(ctx, step_id, depth, center):
    """Fit the pixel-ray model  world = C + B @ [u*d, v*d, d]  against pixel_to_position.

    With the camera centre C known (camera_poses) the fit stays well conditioned even
    when all sampled points are coplanar (table).  Without it a 4-column affine fit is
    used and rejected when its design matrix is (near) rank deficient."""
    H, W = depth.shape
    us, vs = np.meshgrid(np.linspace(10, W - 11, 8), np.linspace(10, H - 11, 8))
    sel = [(float(round(u)), float(round(v))) for u, v in zip(us.ravel(), vs.ravel())]
    sel = [(u, v) for u, v in sel if np.isfinite(depth[int(v), int(u)])]
    rows, worlds = [], []
    for i in range(0, len(sel), 32):
        chunk = sel[i:i + 32]
        res = parse_reply(ctx.call("robomme_pixel_to_position", observation_step_id=int(step_id),
                                   depth_window_radius=0,
                                   selections=[{"camera": "front", "u": u, "v": v} for u, v in chunk]))
        pts = None
        for k in ("points", "results", "positions", "selections"):
            if isinstance(res.get(k), list):
                pts = res[k]
                break
        if pts is None:
            pts = find_key(res, "points") or []
        if i == 0:
            save_json("pixel_to_position_sample.json", res)
        for (u, v), p in zip(chunk, pts):
            if not isinstance(p, dict) or p.get("valid") is False:
                continue
            w = None
            for k in ("position", "world", "point", "xyz", "position_m", "world_position",
                      "world_position_m", "point_world"):
                if isinstance(p.get(k), (list, tuple)) and len(p[k]) >= 3:
                    w = p[k][:3]
                    break
            if w is None:
                continue
            rows.append((u, v))
            worlds.append(w)
    if len(rows) < 10:
        return None, f"only {len(rows)} calibration points"
    worlds = np.asarray(worlds, float)
    best = None
    for swap in (False, True):
        F = []
        for (u, v) in rows:
            d = depth[int(u), int(v)] if swap else depth[int(v), int(u)]
            F.append([u * d, v * d, d])
        F = np.asarray(F, float)
        ok = np.all(np.isfinite(F), axis=1)
        if ok.sum() < 10:
            continue
        if center is not None:
            Bt, *_ = np.linalg.lstsq(F[ok], worlds[ok] - center, rcond=None)
            C = center
        else:
            X = np.hstack([F, np.ones((len(F), 1))])[ok]
            sv = np.linalg.svd(X / np.linalg.norm(X, axis=0), compute_uv=False)
            if sv[-1] / sv[0] < 1e-3:
                continue  # coplanar points: affine model undetermined
            At, *_ = np.linalg.lstsq(X, worlds[ok], rcond=None)
            Bt, C = At[:3], At[3]
        pred = F[ok] @ Bt + C
        res = float(np.sqrt(np.mean(np.sum((pred - worlds[ok]) ** 2, axis=1))))
        if best is None or res < best[1]:
            best = ((Bt.T, np.asarray(C, float)), res, swap)
    if best is None:
        return None, "calibration fit failed or degenerate"
    return best, "ok"


class RayField:
    """world = C + depth * ray(u, v) for every pixel of one fixed camera."""

    def __init__(self, cam, shape, swap):
        B, C = cam
        H, W = shape
        v, u = np.mgrid[0:H, 0:W].astype(np.float64)
        if swap:
            u, v = v, u
        self.ray = np.stack([u, v, np.ones_like(u)], -1) @ B.T
        self.C = C

    def points(self, depth, mask):
        return self.C + depth[mask].astype(np.float64)[:, None] * self.ray[mask]


def local_table_z(points, center_xy, radius):
    """Table height = 10th percentile of z among points near the workspace centre."""
    sel = np.linalg.norm(points[:, :2] - np.asarray(center_xy)[None, :2], axis=1) < radius
    if sel.sum() < 50:
        sel = np.ones(len(points), bool)
    return float(np.percentile(points[sel, 2], 10))


def traj_from_images(ctx, video, ts, execf, mdir, exec_state, step_id, pkt0=None):
    if len(video) < 5 or not execf:
        return None, "not enough frames for visual tracking"
    ent_exec = frame_file(execf[-1], "front", "depth")
    if ent_exec is None:
        return None, "no execution depth"
    d_exec = load_depth_m(ent_exec, mdir)
    center = camera_center(pkt0) if pkt0 is not None else None
    cal, why = fit_linear_camera(ctx, step_id, d_exec, center)
    if cal is None:
        return None, "calibration: " + why
    cam, res, swap = cal
    log("camera ray fit", residual_m=res, swap=swap, center=center)
    if res > 0.01:
        return None, f"calibration residual too high {res:.4f}"
    depths, keep_t = [], []
    for f, t in zip(video, ts):
        e = frame_file(f, "front", "depth")
        if e is None:
            continue
        try:
            depths.append(load_depth_m(e, mdir)); keep_t.append(t)
        except Exception:
            continue
    if len(depths) < 5:
        return None, "video depth unavailable"
    D = np.stack(depths + [d_exec])                    # N+1,H,W
    with np.errstate(all="ignore"):
        bg = np.nanmedian(D, axis=0)
    rf = RayField(cam, bg.shape, swap)
    bgw = rf.points(bg, np.isfinite(bg))
    table_z = local_table_z(bgw, exec_state[:2] if exec_state is not None else np.median(bgw[:, :2], 0), 0.35)

    def tip_of(d, prev):
        fg = np.isfinite(d) & np.isfinite(bg) & (bg - d > 0.012)   # nearer than background
        if fg.sum() < 5:
            return None
        w = rf.points(d, fg)
        w = w[w[:, 2] > table_z + 0.002]
        if prev is not None:
            w = w[np.linalg.norm(w - prev, axis=1) < 0.10]
        if len(w) < 3:
            return None
        zmin = np.min(w[:, 2])
        sel = w[w[:, 2] < zmin + 0.008]
        return np.median(sel, axis=0)

    # exec frame: offset between visual tip and measured TCP
    tip_exec = tip_of(d_exec, exec_state[:3] if exec_state is not None else None)
    offset = np.zeros(3)
    if tip_exec is not None and exec_state is not None and np.linalg.norm(exec_state[:3] - tip_exec) < 0.15:
        offset = exec_state[:3] - tip_exec
    P, T = [], []
    prev = tip_exec
    for d, t in zip(depths, keep_t):
        tp = tip_of(d, prev)
        if tp is None:
            continue
        P.append(tp + offset); T.append(t); prev = tp
    if len(P) < 5:
        return None, "visual track too short"
    P = np.array(P); T = np.array(T)
    # occlusion by obstacles hides the lower stick, which only ever biases z upward:
    # take a rolling minimum of z over +-0.75 s (lower envelope) before smoothing
    z = P[:, 2].copy()
    for i in range(len(P)):
        win = np.abs(T - T[i]) <= 0.75
        z[i] = np.min(P[win, 2])
    n_occl = int(np.sum(P[:, 2] - z > 0.015))
    P[:, 2] = z
    # temporal median smoothing (window 5)
    Ps = P.copy()
    for i in range(len(P)):
        Ps[i] = np.median(P[max(0, i - 2):i + 3], axis=0)
    spread = float(np.max(np.linalg.norm(Ps - Ps[0], axis=1)))
    if spread < 0.02:
        return None, "visual track does not move"
    Rv = np.tile(exec_state[3:6] if exec_state is not None else np.zeros(3), (len(Ps), 1))
    info = {"n": int(len(Ps)), "spread": spread, "offset": offset, "table_z": table_z,
            "n_occluded": n_occl,
            "start": Ps[0], "end": Ps[-1]}
    return (Ps, Rv, T, info), "ok"


def occupancy(rf, depth, table_z, exclude_xy, x0, y0, nx, ny, cell=0.01):
    """Top-down occupancy grid of things standing on the table (robot TCP area removed)."""
    m = np.isfinite(depth)
    w = rf.points(depth, m)
    sel = (w[:, 2] > table_z + 0.008) & (w[:, 2] < table_z + 0.20)
    w = w[sel]
    if exclude_xy is not None:
        w = w[np.linalg.norm(w[:, :2] - exclude_xy[None, :2], axis=1) > 0.08]
    g = np.zeros((nx, ny), bool)
    ix = np.floor((w[:, 0] - x0) / cell).astype(int)
    iy = np.floor((w[:, 1] - y0) / cell).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    g[ix[ok], iy[ok]] = True
    return g


def scene_shift(ctx, video, execf, mdir, pkt, exec_state, video_state0, step_id, path):
    """Compare the first conditioning frame with the live frame.

    Returns (shift_xy or None, diagnostics).  A shift is returned only on strong evidence
    that the static scene moved by a translation between video and execution."""
    diag = {}
    e_v = frame_file(video[0], "front", "depth") if video else None
    e_x = frame_file(execf[-1], "front", "depth") if execf else None
    if e_v is None or e_x is None:
        diag["why"] = "no depth"
        return None, diag
    dv, dx = load_depth_m(e_v, mdir), load_depth_m(e_x, mdir)
    both = np.isfinite(dv) & np.isfinite(dx)
    nchg = int(np.sum(np.abs(dv - dx)[both] > 0.01)) if both.any() else 10 ** 6
    diag["pixels_changed"] = nchg
    if nchg < 40:
        diag["why"] = "scene matches"
        return None, diag
    center = camera_center(pkt)
    if center is None:
        diag["why"] = "camera centre unknown; skip registration"
        return None, diag
    cal, why = fit_linear_camera(ctx, step_id, dx, center)
    if cal is None or cal[1] > 0.01:
        diag["why"] = "calibration unavailable: " + str(why)
        return None, diag
    cam, res, swap = cal
    rf = RayField(cam, dx.shape, swap)
    wx = rf.points(dx, np.isfinite(dx))
    table_z = local_table_z(wx, np.mean(path[:, :2], axis=0), 0.25)
    lo = np.min(path[:, :2], axis=0) - 0.20
    hi = np.max(path[:, :2], axis=0) + 0.20
    cell = 0.01
    nx, ny = [int(min(200, max(10, math.ceil((h - l) / cell)))) for l, h in zip(lo, hi)]
    gv = occupancy(rf, dv, table_z, video_state0, lo[0], lo[1], nx, ny, cell)
    gx = occupancy(rf, dx, table_z, exec_state, lo[0], lo[1], nx, ny, cell)
    diag.update(occ_video=int(gv.sum()), occ_exec=int(gx.sum()))
    if gv.sum() < 5 or gx.sum() < 5:
        diag["why"] = "too few occupied cells"
        return None, diag

    def mismatch(sx, sy):
        a = np.roll(np.roll(gv, sx, 0), sy, 1)
        return int(np.sum(a ^ gx))

    m0 = mismatch(0, 0)
    best = (m0, 0, 0)
    for sx in range(-15, 16):
        for sy in range(-15, 16):
            m = mismatch(sx, sy)
            if m < best[0]:
                best = (m, sx, sy)
    diag.update(mismatch0=m0, best=best)
    shift = np.array([best[1], best[2]], float) * cell
    if (m0 > 0.3 * (gv.sum() + gx.sum()) and best[0] < m0 / 3.0 and np.linalg.norm(shift) >= 0.02):
        diag["why"] = "translated scene"
        return shift, diag
    diag["why"] = "no confident translation"
    return None, diag


# --------------------------------------------------------------- planning
def build_plan(P, Rv, T, remaining_steps, freq_ctrl=20.0):
    """Per-control-tick desired positions/orientations along the demo path."""
    # merge near-duplicate samples (compress pauses)
    idx = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[idx[-1]]) >= MIN_MOVE:
            idx.append(i)
    if idx[-1] != len(P) - 1 and np.linalg.norm(P[-1] - P[idx[-1]]) > 0:
        idx.append(len(P) - 1)
    P2, R2, T2 = P[idx], Rv[idx], T[idx]
    if len(P2) < 2:
        return None
    seg_d = np.linalg.norm(np.diff(P2, axis=0), axis=1)
    seg_dt = np.diff(T2) * freq_ctrl                     # demo duration in control steps
    seg_dt = np.minimum(seg_dt, np.maximum(seg_d / V_MAX, 1.0) + PAUSE_MAX_STEPS)

    def steps_for(s):
        return np.maximum(np.maximum(seg_dt / s, seg_d / min(V_MAX * s, V_HARD)), 0.0)

    budget = max(10.0, STEP_BUDGET_FRACTION * (remaining_steps or 10_000) - 40)
    s = 1.0
    if steps_for(1.0).sum() > budget:
        lo, hi = 1.0, 20.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if steps_for(mid).sum() > budget:
                lo = mid
            else:
                hi = mid
        s = hi
    seg_steps = steps_for(s)
    tk = np.concatenate([[0.0], np.cumsum(seg_steps)])
    n_ticks = int(math.ceil(tk[-1]))
    ticks = np.arange(n_ticks + 1, dtype=float)
    Pdes = np.stack([np.interp(ticks, tk, P2[:, j]) for j in range(3)], axis=1)
    seg_i = np.clip(np.searchsorted(tk, ticks, side="right") - 1, 0, len(R2) - 1)
    Rdes = R2[seg_i]
    return {"Pdes": Pdes, "Rdes": Rdes, "speedup": s, "n_ticks": n_ticks,
            "path_len": float(seg_d.sum()), "demo_steps": float(np.diff(T2).sum() * freq_ctrl)}


# --------------------------------------------------------------- robot I/O
class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.state = None
        self.ended = False
        self.success = False
        self.remaining = None
        self.consec_err = 0
        self.fail_streak = 0
        self.stale_state = 0
        self.lat = None
        self.calls = 0
        self.steps_sent = 0
        self.last_pkt = {}

    def _absorb(self, pkt):
        self.last_pkt = pkt
        st = get_state(pkt)
        if st is not None:
            self.state = st
        if is_ended(pkt):
            self.ended = True
        if is_success(pkt):
            self.success = True
        r = remaining_steps_from(pkt)
        if r is not None:
            self.remaining = r

    def observe(self):
        pkt = parse_reply(self.ctx.call("robomme_observe"))
        self._absorb(pkt)
        return pkt

    def status(self):
        try:
            pkt = parse_reply(self.ctx.call("robomme_status"))
            self._absorb(pkt)
            return pkt
        except Exception as e:
            log("robomme_status failed", err=repr(e))
            return {}

    def step(self, rows):
        if self.ended:
            return False
        if self.remaining is not None and self.remaining > 0:
            rows = rows[:self.remaining]   # the env cuts surplus rows anyway
        if not rows:
            self.fail_streak += 1
            return False
        rows = [[float(min(1.0, max(-1.0, x))) if math.isfinite(x) else 0.0 for x in r]
                for r in rows][:100]
        t0 = time.time()
        try:
            pkt = parse_reply(self.ctx.call("robomme_step", actions=rows))
            self.consec_err = 0
        except Exception as e:
            self.consec_err += 1
            self.fail_streak += 1
            log("robomme_step error", err=repr(e)[:400], consec=self.consec_err)
            # do not retry blindly: refresh state from a fresh observation
            try:
                self.observe()
                self.status()
            except Exception as e2:
                log("observe after error failed", err=repr(e2)[:200])
            return False
        dt = time.time() - t0
        self.lat = dt if self.lat is None else 0.7 * self.lat + 0.3 * dt
        self.calls += 1
        self.steps_sent += len(rows)
        self.fail_streak = 0
        prev_state = self.state
        self.state = None
        self._absorb(pkt)
        if self.state is None:
            # reply without a parsable state: refresh from the cached observation
            try:
                self.observe()
            except Exception as e:
                log("observe for state failed", err=repr(e)[:200])
            if self.state is None:
                self.state = prev_state
                self.stale_state += 1
            else:
                self.stale_state = 0
        else:
            self.stale_state = 0
        if self.remaining is not None and remaining_steps_from(pkt) is None:
            self.remaining -= len(rows)
        return True


def zero_row():
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_CMD]


# --------------------------------------------------------------- orientation
class RotCtl:
    """Closed-loop orientation hold/track with automatic convention fallback.

    Protocol says rotation deltas are world-frame axis-angle.  If corrections make the
    error grow twice in a row we switch world -> body -> off (never spin blindly)."""
    MODES = ("world", "body", "off")

    def __init__(self):
        self.mode_i = 0
        self.bad = 0
        self.last_cmd = 0.0
        self.cmd_world = np.zeros(3)
        self.switches = []

    @property
    def mode(self):
        return self.MODES[self.mode_i]

    def rows(self, r_des, r_cur, n_rows):
        self.last_cmd = 0.0
        if self.mode == "off" or r_des is None or r_cur is None or n_rows <= 0:
            return []
        e = rot_err(r_des, r_cur)
        ang = float(np.linalg.norm(e))
        if ang < ROT_DEADBAND:
            return []
        if self.mode == "body":
            e = rotvec_to_mat(r_cur).T @ e
        total = e * ROT_GAIN
        n = max(1, int(math.ceil(float(np.linalg.norm(total)) / ROT_MAX_STEP)))
        n = min(n, n_rows)
        per = total / n
        nrm = float(np.linalg.norm(per))
        if nrm > ROT_MAX_STEP:
            per = per * (ROT_MAX_STEP / nrm)
        self.last_cmd = float(np.linalg.norm(per)) * n
        cmd = per * n
        self.cmd_world = rotvec_to_mat(r_cur) @ cmd if self.mode == "body" else cmd
        return [per / ROT_SCALE] * n

    def feedback(self, r_des, r_before, r_after):
        if self.mode == "off" or self.last_cmd < 0.015:
            return
        e0 = float(np.linalg.norm(rot_err(r_des, r_before)))
        e1 = float(np.linalg.norm(rot_err(r_des, r_after)))
        achieved = rot_err(r_after, r_before)          # world-frame rotation that happened
        c = self.cmd_world
        ratio = float(np.dot(achieved, c) / max(np.dot(c, c), 1e-9))
        if ratio < 0.15 or e1 > 0.5:
            self.bad += 1
        else:
            self.bad = max(0, self.bad - 1)
        if self.bad >= 2:
            self.switches.append((self.mode, e0, e1))
            log("orientation correction not converging; switching convention",
                frm=self.mode, e_before=e0, e_after=e1, ratio=ratio)
            self.mode_i = min(len(self.MODES) - 1, self.mode_i + 1)
            self.bad = 0


# --------------------------------------------------------------- follower
class Follower:
    def __init__(self, robot, plan, track_rot, rotctl):
        self.r = robot
        self.plan = plan
        self.track_rot = track_rot
        self.rot = rotctl
        self.k = 0
        self.gain = 1.0          # measured displacement / commanded displacement
        self.max_err = 0.0
        self.errs = []

    def rows_toward(self, cur_p, cur_r, targets, r_des):
        """Rows that move from cur_p through the list of target points."""
        rows = []
        prev = cur_p.copy()
        g = max(0.35, min(1.3, self.gain))
        rot_rows = self.rot.rows(r_des, cur_r, len(targets))
        for j, tgt in enumerate(targets):
            d = (tgt - prev) / g
            nrm = np.linalg.norm(d)
            lim = V_HARD * 1.4
            if nrm > lim:
                d = d * (lim / nrm)
            row = list(d / POS_SCALE)
            row += list(rot_rows[j]) if j < len(rot_rows) else [0.0, 0.0, 0.0]
            row.append(GRIPPER_CMD)
            rows.append(row)
            prev = tgt
        return rows

    def chunk_size(self, wall_left):
        left_ticks = max(1, self.plan["n_ticks"] - self.k)
        lat = self.r.lat if self.r.lat else 1.0
        if wall_left <= 30:
            return CHUNK_MAX
        c = int(math.ceil(left_ticks * lat / max(wall_left - 30.0, 1.0)))
        return max(CHUNK_MIN, min(CHUNK_MAX, c))

    def update_gain(self, p0, p1, commanded):
        cn = np.linalg.norm(commanded)
        if cn < 0.004:
            return
        meas = float(np.dot(p1 - p0, commanded) / cn)
        ratio = meas / cn
        if 0.2 < ratio < 2.0:
            self.gain = 0.8 * self.gain + 0.2 * ratio

    def run(self, deadline):
        Pdes, Rdes = self.plan["Pdes"], self.plan["Rdes"]
        n = self.plan["n_ticks"]
        it = 0
        stall = 0
        while self.k < n and not self.r.ended:
            it += 1
            if self.r.consec_err > MAX_CONSEC_ERRORS or self.r.fail_streak > MAX_CONSEC_ERRORS:
                log("too many consecutive errors; stop following")
                return "errors"
            if self.r.stale_state > 3:
                log("robot state not updating; stop following to avoid blind motion")
                return "stale"
            if self.r.success:
                log("task_complete reported; stop following", k=self.k, n=n)
                return "success"
            wall_left = deadline - time.time()
            if wall_left < 20:
                log("wall budget exhausted in FOLLOW", k=self.k, n=n)
                return "wall"
            cur = self.r.state
            p = cur[:3].copy()
            rcur = cur[3:6].copy()
            err_vec = Pdes[self.k] - p
            err = float(np.linalg.norm(err_vec))
            self.errs.append(err)
            self.max_err = max(self.max_err, err)
            c = self.chunk_size(wall_left)
            if err > HOLD_ERR:
                # behind schedule: go to the current plan point without advancing time
                stall += 1
                m = max(2, min(c, int(math.ceil(err / V_MAX))))
                targets = [p + err_vec * (j + 1) / m for j in range(m)]
                adv = 0
                if stall > 6:
                    adv = 1  # never freeze forever: creep plan time
            else:
                stall = 0
                c = min(c, n - self.k)
                targets = []
                for j in range(c):
                    base = Pdes[self.k + j + 1]
                    # spread error correction over the chunk
                    corr = err_vec * CORR_GAIN * min(1.0, (j + 1) / max(1, min(c, 3)))
                    targets.append(base + corr)
                adv = c
            r_des = Rdes[min(n, self.k + len(targets))]
            rows = self.rows_toward(p, rcur, targets, r_des)
            ok = self.r.step(rows)
            if not ok:
                if self.r.ended:
                    break
                continue
            newp = self.r.state[:3]
            self.update_gain(p, newp, targets[-1] - p)
            self.rot.feedback(r_des, rcur, self.r.state[3:6])
            self.k = min(n, self.k + adv)
            if it % 10 == 0 or self.k >= n:
                log("follow", k=self.k, n=n, err=round(err, 4), gain=round(self.gain, 3),
                    chunk=len(rows), lat=round(self.r.lat or 0, 3), rem=self.r.remaining,
                    p=newp, rot_mode=self.rot.mode,
                    rot_err=round(float(np.linalg.norm(rot_err(r_des, self.r.state[3:6]))), 3))
        return "ended" if self.r.ended else "done"


def go_to(robot, target, r_hold, max_steps, tol, deadline, label, rotctl=None):
    """Closed-loop straight-line move to target position."""
    used = 0
    while used < max_steps and not robot.ended and time.time() < deadline:
        if robot.consec_err > MAX_CONSEC_ERRORS or robot.fail_streak > MAX_CONSEC_ERRORS \
                or robot.stale_state > 3 or robot.success:
            break
        p = robot.state[:3]
        e = target - p
        d = float(np.linalg.norm(e))
        if d < tol:
            break
        m = max(1, min(6, int(math.ceil(d / V_MAX))))
        r_before = robot.state[3:6].copy()
        rr = rotctl.rows(r_hold, r_before, m) if rotctl is not None else []
        rows = []
        for j in range(m):
            step = e / m if d / m <= V_MAX else e / d * V_MAX
            rot = list(rr[j]) if j < len(rr) else [0.0, 0.0, 0.0]
            row = list(step / POS_SCALE) + rot + [GRIPPER_CMD]
            rows.append(row)
        robot.step(rows)
        if rotctl is not None:
            rotctl.feedback(r_hold, r_before, robot.state[3:6])
        used += len(rows)
    log(label, target=target, reached=robot.state[:3] if robot.state is not None else None,
        steps=used)
    return used


# --------------------------------------------------------------- main
def run(ctx):
    _CALIB_CACHE.clear()
    deadline = time.time() + CTRL_WALL_BUDGET_S
    robot = Robot(ctx)
    pkt = robot.observe()
    st = robot.status()
    instruction = find_key(pkt, "instruction") or find_key(st, "instruction")
    log("start", instruction=instruction, state=robot.state, remaining=robot.remaining,
        limits=find_key(st, "episode_limits") or find_key(pkt, "episode_limits"))
    save_json("observe0_meta.json", {k: v for k, v in pkt.items() if k not in ("images",)})
    save_json("status0.json", st)
    if robot.state is None:
        log("no state in observation; cannot control")
        return
    exec_state = robot.state.copy()
    step_id = find_key(pkt, "step_id") or 0

    man, mdir = None, None
    try:
        man, mdir = load_manifest(pkt)
    except Exception as e:
        log("manifest load failed", err=repr(e))
    if man is None:
        log("no manifest; holding")
        return
    freq = man.get("frequency_hz") or 20
    video, execf = frame_records(man)
    ts = frame_times(video, float(freq))
    log("manifest", n_video=len(video), n_exec=len(execf), freq=freq,
        duration_s=float(ts[-1] - ts[0]) if len(ts) else 0.0,
        sampling=man.get("video_sampling"))

    traj, src = None, None
    try:
        traj, why = traj_from_states(video, ts, exec_state)
        log("state-path", ok=traj is not None, why=why, info=traj[3] if traj else None)
        src = "states" if traj else None
    except Exception as e:
        log("state-path exception", err=traceback.format_exc()[-800:])
    if traj is not None:
        # cross-check recorded states against the video depth; drop them only on strong
        # evidence that they do not describe the stick seen in the video
        try:
            frac, why = states_vs_depth(ctx, video, execf, mdir, pkt, step_id)
            log("states-vs-depth", consistent_frac=frac, why=why)
            if frac is not None and frac < 0.3:
                alt, why2 = traj_from_images(ctx, video, ts, execf, mdir, exec_state, step_id, pkt)
                log("image-path (states inconsistent)", ok=alt is not None, why=why2,
                    info=alt[3] if alt else None)
                if alt is not None:
                    traj, src = alt, "images"
        except Exception:
            log("states-vs-depth exception", err=traceback.format_exc()[-800:])
    if traj is None:
        try:
            traj, why = traj_from_images(ctx, video, ts, execf, mdir, exec_state, step_id, pkt)
            log("image-path", ok=traj is not None, why=why, info=traj[3] if traj else None)
            src = "images" if traj else None
        except Exception:
            log("image-path exception", err=traceback.format_exc()[-800:])
    if traj is None:
        log("no usable route; holding pose and returning")
        robot.step([zero_row()] * 5)
        return

    P, Rv, T, info = traj
    try:
        v0 = P[0] if src == "states" else None
        shift, diag = scene_shift(ctx, video, execf, mdir, pkt, exec_state, v0, step_id, P)
        log("scene check", shift=shift, diag=diag)
        if shift is not None:
            P = P.copy()
            P[:, :2] += shift[None, :]
            if src == "states":
                # the robot itself did not move: keep the very first point at the live pose
                P[0, :2] -= shift
    except Exception:
        log("scene check exception", err=traceback.format_exc()[-800:])
    instr = (instruction or "").lower() if isinstance(instruction, str) else ""
    if "reverse" in instr:
        log("instruction asks for reverse; reversing path")
        P, Rv = P[::-1].copy(), Rv[::-1].copy()
        T = (T[-1] - T[::-1]).copy()

    # orientation: track only if the demo rotated the tool noticeably
    rot_span = max(float(np.linalg.norm(rot_err(r, Rv[0]))) for r in Rv)
    track_rot = src == "states" and rot_span > 0.05
    if not track_rot:
        Rv = np.tile(exec_state[3:6], (len(P), 1))
    log("orientation", rot_span=rot_span, track_rot=track_rot,
        exec_vs_demo0=float(np.linalg.norm(rot_err(Rv[0], exec_state[3:6]))))

    rotctl = RotCtl()
    remaining = robot.remaining if (robot.remaining is not None and robot.remaining >= 50) else None
    # approach the demo start if the live pose differs from it
    d0 = float(np.linalg.norm(P[0] - robot.state[:3]))
    if d0 > 0.01:
        cur = robot.state[:3].copy()
        hi = max(cur[2], P[0][2])
        if hi - cur[2] > 0.005:
            go_to(robot, np.array([cur[0], cur[1], hi]), Rv[0], 60, APPROACH_TOL, deadline, "approach-lift", rotctl)
        go_to(robot, np.array([P[0][0], P[0][1], hi]), Rv[0], 120, APPROACH_TOL, deadline, "approach-xy", rotctl)
        go_to(robot, P[0].copy(), Rv[0], 60, APPROACH_TOL, deadline, "approach-lower", rotctl)
        if robot.remaining is not None and robot.remaining >= 50:
            remaining = robot.remaining

    plan = build_plan(P, Rv, T, remaining)
    if plan is None:
        log("plan failed; holding")
        return
    log("plan", n_ticks=plan["n_ticks"], speedup=plan["speedup"], path_len=plan["path_len"],
        demo_steps=plan["demo_steps"], remaining=remaining, src=src)
    save_json("plan.json", {"src": src, "P": P, "T": T, "Pdes": plan["Pdes"], "info": info})

    fol = Follower(robot, plan, track_rot, rotctl)
    outcome = fol.run(deadline)
    log("follow finished", outcome=outcome, k=fol.k, n=plan["n_ticks"], max_err=fol.max_err,
        mean_err=float(np.mean(fol.errs)) if fol.errs else None, gain=fol.gain,
        rot_mode=rotctl.mode, rot_switches=rotctl.switches,
        success=robot.success, ended=robot.ended)

    if not robot.ended and not robot.success and outcome == "done":
        go_to(robot, plan["Pdes"][-1], plan["Rdes"][-1], 15, FINAL_TOL, deadline, "settle-final", rotctl)
        # brief hold so that any terminal condition can register
        for _ in range(2):
            if robot.ended or robot.success:
                break
            robot.step([zero_row()] * 5)
    log("end", success=robot.success, ended=robot.ended, remaining=robot.remaining,
        calls=robot.calls, steps_sent=robot.steps_sent, last=robot.last_pkt.get("transition", {}).get("feedback")
        if isinstance(robot.last_pkt.get("transition"), dict) else None)


def main(ctx):
    log("controller main entered", python=sys.version.split()[0])
    try:
        run(ctx)
    except Exception:
        log("FATAL in run (caught)", err=traceback.format_exc()[-2000:])
    finally:
        log("controller main exit")
