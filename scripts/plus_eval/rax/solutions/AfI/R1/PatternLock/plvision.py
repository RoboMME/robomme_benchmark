"""Numpy-based fallbacks: camera back-projection calibrated against robomme_pixel_to_position,
stick-tip tracking in the conditioning video's front depth, and highlight-order detection."""
import math

import numpy as np
from PIL import Image

from plcore import (log, find_all, is_num_list, reply_json, frame_file, frame_time)

TOUCH_BAND_M = 0.02  # tip estimates within this height of the table are treated as touching
FG_DEPTH_M = 0.004  # foreground: closer than the per-pixel farthest depth by this much


# ---------------------------------------------------------------- image loading
def load_depth(path, meters_per_unit=0.001):
    with Image.open(path) as im:
        a = np.array(im)
    if a.ndim == 3:
        a = a[..., 0]
    d = a.astype(np.float32) * float(meters_per_unit or 0.001)
    return d


def load_rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def frame_depth(fr, camera="front"):
    p, meta = frame_file(fr, camera, "depth")
    if p is None:
        return None
    mpu = (meta or {}).get("meters_per_unit", 0.001) or 0.001
    return load_depth(p, mpu)


def frame_rgb(fr, camera="front"):
    p, _ = frame_file(fr, camera, "rgb")
    if p is None:
        return None
    return load_rgb(p)


# ---------------------------------------------------------------- camera parsing
def _cam_entry(obj, camera):
    if isinstance(obj, dict):
        if camera in obj:
            return obj[camera]
        for k, v in obj.items():
            if camera in str(k).lower():
                return v
    if isinstance(obj, list):
        for v in obj:
            if isinstance(v, dict) and (v.get("camera") == camera or v.get("name") == camera):
                return v
    return None


def parse_intrinsics(entry):
    if entry is None:
        return None
    if isinstance(entry, dict):
        lower = {str(k).lower(): v for k, v in entry.items()}
        if all(k in lower for k in ("fx", "fy", "cx", "cy")):
            return [float(lower[k]) for k in ("fx", "fy", "cx", "cy")]
        for v in entry.values():
            r = parse_intrinsics(v)
            if r:
                return r
        return None
    if isinstance(entry, list):
        try:
            a = np.array(entry, dtype=float)
        except Exception:
            return None
        if a.shape == (3, 3):
            return [a[0, 0], a[1, 1], a[0, 2], a[1, 2]]
        if a.size == 9:
            a = a.reshape(3, 3)
            return [a[0, 0], a[1, 1], a[0, 2], a[1, 2]]
    return None


def quat_wxyz_to_mat(q):
    w, x, y, z = [float(t) for t in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def parse_pose(entry):
    """-> list of candidate (R, t) camera-to-world."""
    out = []
    if entry is None:
        return out
    if isinstance(entry, dict):
        pos, quat, mat = None, None, None
        for k, v in entry.items():
            kl = str(k).lower()
            if is_num_list(v, 3) and ("pos" in kl or kl in ("p", "t", "translation", "xyz")):
                pos = v
            elif is_num_list(v, 4) and ("quat" in kl or kl in ("q", "rotation", "orientation")):
                quat = (v, "xyzw" if "xyzw" in kl else "wxyz")
            elif isinstance(v, list) and len(v) == 4 and all(is_num_list(r, 4) for r in v):
                mat = v
        if mat is not None:
            m = np.array(mat, dtype=float)
            out.append((m[:3, :3], m[:3, 3]))
        if pos is not None and quat is not None:
            q, order = quat
            qw = q if order == "wxyz" else [q[3], q[0], q[1], q[2]]
            out.append((quat_wxyz_to_mat(qw), np.array(pos, dtype=float)))
            if order == "wxyz":  # also try the other ordering as a candidate
                out.append((quat_wxyz_to_mat([q[3], q[0], q[1], q[2]]), np.array(pos, dtype=float)))
        if not out:
            for v in entry.values():
                if isinstance(v, (dict, list)):
                    out.extend(parse_pose(v))
    elif isinstance(entry, list) and len(entry) == 4 and all(is_num_list(r, 4) for r in entry):
        m = np.array(entry, dtype=float)
        out.append((m[:3, :3], m[:3, 3]))
    return out


# ---------------------------------------------------------------- camera model
class CameraModel:
    def __init__(self, kind, params, residual):
        self.kind, self.params, self.residual = kind, params, residual

    def project(self, u, v, d):
        """u, v, d arrays -> (N,3) world points."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        d = np.asarray(d, dtype=float)
        if self.kind == "affine":
            F = np.stack([u * d, v * d, d, np.ones_like(d)], axis=-1)
            return F @ self.params.T
        fx, fy, cx, cy, R, t, conv, off = self.params
        uu, vv = u + off, v + off
        x = (uu - cx) / fx * d
        y = (vv - cy) / fy * d
        if conv == "gl":
            pc = np.stack([x, -y, -d], axis=-1)
        else:
            pc = np.stack([x, y, d], axis=-1)
        return pc @ R.T + t

    def describe(self):
        return "%s(res=%.4f)" % (self.kind if self.kind == "affine" else
                                 "pinhole-%s-off%.1f" % (self.params[6], self.params[7]), self.residual)


def _tool_points(result, n):
    js = reply_json(result)
    best = None

    def walk(o, depth=0):
        nonlocal best
        if depth > 8:
            return
        if isinstance(o, list) and len(o) == n and all(isinstance(e, dict) for e in o):
            best = o
            return
        if isinstance(o, dict):
            for v in o.values():
                walk(v, depth + 1)
        elif isinstance(o, list):
            for v in o:
                walk(v, depth + 1)

    walk(js)
    pts = []
    if best is None:
        return [None] * n, js
    pref = ("world", "position", "point", "xyz")
    for e in best:
        if e.get("valid") is False:
            pts.append(None)
            continue
        cand = None
        for k, v in e.items():
            if is_num_list(v, 3) and any(p in str(k).lower() for p in pref):
                cand = v
                break
        if cand is None:
            for k, v in e.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        if is_num_list(v2, 3):
                            cand = v2
                            break
                if cand is None and is_num_list(v, 3) and str(k).lower() not in ("uv", "pixel"):
                    cand = v
                if cand is not None:
                    break
        pts.append([float(x) for x in cand] if cand is not None else None)
    return pts, js


def _flat(depth, u, v):
    H, W = depth.shape
    if v < 2 or u < 2 or v > H - 3 or u > W - 3:
        return False
    win = depth[v - 2:v + 3, u - 2:u + 3]
    if not np.all(win > 0):
        return False
    d = depth[v, u]
    return (abs(depth[v, u - 1] + depth[v, u + 1] - 2 * d) < 0.003 and
            abs(depth[v - 1, u] + depth[v + 1, u] - 2 * d) < 0.003 and
            abs(depth[v, u - 2] + depth[v, u + 2] - 2 * d) < 0.004 and
            abs(depth[v - 2, u] + depth[v + 2, u] - 2 * d) < 0.004)


def calibrate_front(ctx, packet, depth_exec, step_id):
    """Fit a pixel+depth -> world model for the (static) front camera."""
    H, W = depth_exec.shape
    sel = []
    grid = []
    for gv in np.linspace(12, H - 13, 7):
        for gu in np.linspace(12, W - 13, 7):
            u, v = int(gu), int(gv)
            if _flat(depth_exec, u, v):
                grid.append((u, v, float(depth_exec[v, u])))
    # spread selection: up to 22 grid points, plus closest flat points (robot) for non-planarity
    if len(grid) > 22:
        idx = np.linspace(0, len(grid) - 1, 22).astype(int)
        grid = [grid[i] for i in idx]
    sel.extend(grid)
    valid = depth_exec > 0
    if valid.sum() > 100:
        dv = np.where(valid, depth_exec, np.inf)
        order = np.argsort(dv, axis=None)
        picked = 0
        for flat in order[:4000:37]:
            v, u = divmod(int(flat), W)
            if v < 3 or u < 3 or v > H - 4 or u > W - 4:
                continue
            if _flat(depth_exec, u, v):
                if all(abs(u - s[0]) + abs(v - s[1]) > 6 for s in sel):
                    sel.append((u, v, float(depth_exec[v, u])))
                    picked += 1
            if picked >= 10 or len(sel) >= 32:
                break
    sel = sel[:32]
    if len(sel) < 6:
        log("calibrate: too few pixels", len(sel))
        return None
    try:
        res = ctx.call("robomme_pixel_to_position", observation_step_id=int(step_id),
                       selections=[{"camera": "front", "u": float(u), "v": float(v)} for u, v, _ in sel],
                       depth_window_radius=0)
    except Exception as exc:
        log("pixel_to_position failed", repr(exc)[:300])
        return None
    pts, js = _tool_points(res, len(sel))
    log("pixel_to_position sample reply:", str(js)[:600])
    data = [(u, v, d, p) for (u, v, d), p in zip(sel, pts) if p is not None]
    if len(data) < 6:
        log("calibrate: too few valid tool points", len(data))
        return None
    U = np.array([x[0] for x in data], float)
    V = np.array([x[1] for x in data], float)
    D = np.array([x[2] for x in data], float)
    P = np.array([x[3] for x in data], float)
    models = []
    # pinhole candidates from the declared calibration
    intr = None
    poses = []
    for _, v in find_all(packet, "camera_intrinsics"):
        intr = parse_intrinsics(_cam_entry(v, "front")) or intr
    for _, v in find_all(packet, "camera_poses"):
        poses.extend(parse_pose(_cam_entry(v, "front")))
    if intr:
        fx, fy, cx, cy = intr
        for R, t in poses:
            for RR in (R, R.T):
                for conv in ("gl", "cv"):
                    for off in (0.0, 0.5):
                        m = CameraModel("pinhole", (fx, fy, cx, cy, RR, t, conv, off), 0)
                        r = np.linalg.norm(m.project(U, V, D) - P, axis=1)
                        m.residual = float(np.median(r))
                        models.append(m)
    # affine fit (only if non-degenerate)
    F = np.stack([U * D, V * D, D, np.ones_like(D)], axis=-1)
    sv = np.linalg.svd(F / (np.abs(F).max(axis=0) + 1e-9), compute_uv=False)
    if len(data) >= 8 and sv[-1] / sv[0] > 1e-3:
        M, *_ = np.linalg.lstsq(F, P, rcond=None)
        m = CameraModel("affine", M.T, 0)
        r = np.linalg.norm(m.project(U, V, D) - P, axis=1)
        m.residual = float(np.median(r))
        models.append(m)
    if not models:
        log("calibrate: no candidate models")
        return None
    models.sort(key=lambda m: m.residual)
    log("calibrate candidates:", [m.describe() for m in models[:4]], "svratio=%.2e" % (sv[-1] / sv[0]))
    best = models[0]
    if best.residual > 0.015:
        log("calibrate: best residual too large", best.residual)
        return None
    return best


# ---------------------------------------------------------------- table height
def table_height(model, depth):
    H, W = depth.shape
    vv, uu = np.nonzero(depth > 0)
    if len(vv) < 100:
        return None
    step = max(1, len(vv) // 20000)
    pts = model.project(uu[::step], vv[::step], depth[vv[::step], uu[::step]])
    z = pts[:, 2]
    hist, edges = np.histogram(z, bins=np.arange(-1.0, 2.0, 0.004))
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


# ---------------------------------------------------------------- tip tracking
def _lowest_point(model, depth, mask, z_floor):
    vv, uu = np.nonzero(mask)
    if len(vv) < 15:
        return None
    pts = model.project(uu, vv, depth[vv, uu])
    pts = pts[pts[:, 2] > z_floor]
    if len(pts) < 15:
        return None
    order = np.argsort(pts[:, 2])
    low = pts[order[:6]]
    med = np.median(low, axis=0)
    keep = np.linalg.norm(low - med, axis=1) < 0.03
    if keep.sum() < 5:
        return None
    return np.median(low[keep], axis=0)


def track_tip(video, exec_frame, model, exec_state, max_frames=300):
    """Returns list of (t, tcp_xyz) estimated from front depth, or None."""
    n = len(video)
    if n < 3:
        return None, {}
    idxs = list(range(n))
    if n > max_frames:
        idxs = sorted(set(np.linspace(0, n - 1, max_frames).astype(int).tolist()))
    depths, times = [], []
    for i in idxs:
        d = frame_depth(video[i])
        if d is None:
            continue
        depths.append(d)
        times.append(frame_time(video[i], i))
    if len(depths) < 3:
        return None, {}
    de = frame_depth(exec_frame) if exec_frame is not None else None
    stack = np.stack(depths + ([de] if de is not None else []), axis=0)
    bg = stack.max(axis=0)
    tz = table_height(model, bg)
    info = {"table_z": tz, "frames": len(depths)}
    if tz is None:
        return None, info
    z_floor = tz - 0.02
    tips = []
    moved = []
    for t, d in zip(times, depths):
        fg = (d > 0) & (d < bg - FG_DEPTH_M)
        moved.append(int(fg.sum()))
        tip = _lowest_point(model, d, fg, z_floor)
        if tip is not None:
            tips.append((t, tip))
    info["fg_counts_median"] = float(np.median(moved)) if moved else 0
    if len(tips) < 3:
        return None, info
    # TCP <-> lowest visible point offset from the execution frame
    offset = np.zeros(3)
    if de is not None and exec_state is not None:
        fg = (de > 0) & (de < bg - FG_DEPTH_M)
        tip_e = _lowest_point(model, de, fg, z_floor)
        if tip_e is not None:
            off = np.array(exec_state[:3]) - tip_e
            if np.linalg.norm(off) < 0.3:
                offset = off
        info["exec_tip"] = None if tip_e is None else tip_e.tolist()
    info["offset"] = offset.tolist()
    arr = np.array([p for _, p in tips])
    # temporal median filter (3)
    sm = arr.copy()
    for k in range(1, len(arr) - 1):
        sm[k] = np.median(arr[k - 1:k + 2], axis=0)
    # near-table samples: the lowest few mm of the stick merge with the table in depth, so the
    # estimate reads high there; treat them as touching and command a light press instead.
    z_touch_tip = tz + 0.0 - 0.004
    low = sm[:, 2] < tz + TOUCH_BAND_M
    info["n_touch_samples"] = int(low.sum())
    sm2 = sm.copy()
    sm2[low, 2] = z_touch_tip
    traj = [(tips[k][0], (sm2[k] + offset).tolist()) for k in range(len(tips))]
    rng = float(np.max(np.linalg.norm(sm - sm[0], axis=1)))
    info["range"] = rng
    return traj, info


# ---------------------------------------------------------------- highlight sequence
def detect_highlights(video, exec_frame, model, max_frames=160):
    """Targets that change appearance during a static-robot video, ordered by first change."""
    try:
        from scipy import ndimage
    except Exception:
        ndimage = None
    n = len(video)
    if n < 3 or ndimage is None:
        return None, {"why": "no frames or scipy"}
    idxs = list(range(n))
    if n > max_frames:
        idxs = sorted(set(np.linspace(0, n - 1, max_frames).astype(int).tolist()))
    rgbs, times = [], []
    for i in idxs:
        r = frame_rgb(video[i])
        if r is not None:
            rgbs.append(r)
            times.append(frame_time(video[i], i))
    if len(rgbs) < 3:
        return None, {"why": "no rgb"}
    stack = np.stack(rgbs, 0)
    med = np.median(stack, axis=0).astype(np.int16)
    de = frame_depth(exec_frame)
    tracks = []  # dict(u, v, first_t, count)
    for t, r in zip(times, rgbs):
        diff = np.abs(r.astype(np.int16) - med).max(axis=-1)
        mask = diff > 45
        lab, nl = ndimage.label(mask)
        if nl == 0 or nl > 60:
            continue
        for c in range(1, nl + 1):
            vv, uu = np.nonzero(lab == c)
            if not (6 <= len(vv) <= 2500):
                continue
            cu, cv = float(uu.mean()), float(vv.mean())
            for tr in tracks:
                if abs(tr["u"] - cu) + abs(tr["v"] - cv) < 10:
                    tr["count"] += 1
                    break
            else:
                tracks.append({"u": cu, "v": cv, "first_t": t, "count": 1, "area": len(vv)})
    tracks = [tr for tr in tracks if tr["count"] >= 2]
    tracks.sort(key=lambda tr: tr["first_t"])
    info = {"tracks": tracks[:20]}
    if not (2 <= len(tracks) <= 12) or de is None:
        return None, info
    pts = []
    for tr in tracks:
        u, v = int(round(tr["u"])), int(round(tr["v"]))
        win = de[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
        win = win[win > 0]
        if win.size == 0:
            continue
        p = model.project([u], [v], [float(np.median(win))])[0]
        pts.append(p.tolist())
    info["points"] = pts
    return (pts if len(pts) >= 2 else None), info
