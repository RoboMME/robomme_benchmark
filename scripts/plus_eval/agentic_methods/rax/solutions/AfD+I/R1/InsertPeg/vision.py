"""Fallback method: depth + RGB perception when the video frames carry no robot state.

Steps (each returns early, logged, on failure):
  V1 calibrate front back-projection (validated against robomme_pixel_to_position)
  V2 segment table / box / pegs in the execution front depth
  V3 target peg = peg whose pixels become table (depth increases) in late video frames
  V4 held end = half whose colour sits between the fingers in mid-video wrist frames
  V5 insertion face = box face nearest to the hand at the video's end; hole height estimate
  V6 grasp at the centre of the held half, perpendicular fingers
  V7 lift, yaw so the free end points into the face, approach, push with stall detection
"""
import os

import numpy as np
from PIL import Image

import geom
import replay
from rio import log, find_all, text_json, WS

try:
    from scipy import ndimage
except Exception:  # pragma: no cover
    ndimage = None


# ------------------------------------------------------------------ io helpers

def load_depth(frame, cam):
    path, entry = replay.file_of(frame, cam, "depth")
    if not path or not os.path.isfile(path):
        return None
    scale = 0.001
    try:
        if entry and isinstance(entry.get("meters_per_unit"), (int, float)):
            scale = float(entry["meters_per_unit"])
    except Exception:
        pass
    a = np.asarray(Image.open(path)).astype(float)
    if a.ndim == 3:
        a = a[..., 0]
    return a * scale


def load_rgb(frame, cam):
    path, _ = replay.file_of(frame, cam, "rgb")
    if not path or not os.path.isfile(path):
        return None
    return np.asarray(Image.open(path).convert("RGB")).astype(float)


# ------------------------------------------------------------------ calibration

def _as_K(v):
    try:
        if isinstance(v, dict):
            keys = {k.lower(): k for k in v}
            if "fx" in keys:
                fx = float(v[keys["fx"]]); fy = float(v[keys.get("fy", keys["fx"])])
                cx = float(v[keys["cx"]]); cy = float(v[keys["cy"]])
                return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
            for k in v:
                K = _as_K(v[k])
                if K is not None:
                    return K
            return None
        a = np.array(v, dtype=float)
        if a.shape == (3, 3):
            return a
        if a.size == 9:
            return a.reshape(3, 3)
        if a.size == 4:
            return np.array([[a[0], 0, a[2]], [0, a[1], a[3]], [0, 0, 1.0]])
    except Exception:
        return None
    return None


def _as_pose(v):
    """Return (R, t) camera-to-world."""
    try:
        if isinstance(v, dict):
            keys = {k.lower(): k for k in v}
            q = None
            t = None
            for k in ("quaternion_wxyz", "quat_wxyz", "q_wxyz", "quaternion", "q", "orientation"):
                if k in keys:
                    q = v[keys[k]]
                    break
            for k in ('position_m', "position", "p", "t", "translation", "pos", "xyz"):
                if k in keys:
                    t = v[keys[k]]
                    break
            if q is not None and t is not None and len(q) == 4 and len(t) == 3:
                if "quaternion" in keys and "quaternion_wxyz" not in keys and "xyzw" in str(v.get(keys.get("convention", ""), "")).lower():
                    q = [q[3], q[0], q[1], q[2]]
                return geom.quat_wxyz_to_mat(q), np.array(t, dtype=float)
            for k in ("matrix", "cam2world", "cam2world_gl", "pose", "transform", "extrinsic"):
                if k in keys:
                    r = _as_pose(v[keys[k]])
                    if r is not None:
                        return r
            return None
        a = np.array(v, dtype=float)
        if a.shape == (4, 4):
            return a[:3, :3], a[:3, 3]
        if a.size == 16:
            a = a.reshape(4, 4)
            return a[:3, :3], a[:3, 3]
        if a.size == 7:
            return geom.quat_wxyz_to_mat(a[3:]), a[:3]
    except Exception:
        return None
    return None


def _camera_entry(pkt, key, cam):
    for p, v in find_all(pkt, key):
        if isinstance(v, dict):
            if cam in v:
                return v[cam]
            for k, vv in v.items():
                if cam in str(k).lower():
                    return vv
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and str(item.get("camera", item.get("name", ""))).lower() == cam:
                    return item
    return None


def backproject(depth, K, R, t, conv, off):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W].astype(float)
    u = u + off
    v = v + off
    z = depth
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    if conv == "gl":
        Pc = np.stack([x, -y, -z], -1)
    else:
        Pc = np.stack([x, y, z], -1)
    return Pc @ R.T + t


def tool_points(robot, step_id, cam, pix):
    """robomme_pixel_to_position for up to 32 (u, v) pixels; returns list of xyz or None."""
    out = []
    for k in range(0, len(pix), 32):
        sel = [{"camera": cam, "u": float(u), "v": float(v)} for (u, v) in pix[k:k + 32]]
        args = {"selections": sel, "depth_window_radius": 0}
        if step_id is not None:
            args["observation_step_id"] = int(step_id)
        r = robot.ctx.call("robomme_pixel_to_position", **args)
        pkt = text_json(r)
        items = None
        for key in ("results", "points", "selections", "positions"):
            for p, v in find_all(pkt, key):
                if isinstance(v, list) and len(v) == len(sel):
                    items = v
                    break
            if items is not None:
                break
        if items is None:
            items = [None] * len(sel)
        for it in items:
            out.append(_xyz_of(it))
    return out


def _xyz_of(it):
    if it is None:
        return None
    if isinstance(it, dict):
        if it.get("valid") is False:
            return None
        for k in ("world_position", "position", "world", "point", "xyz", "world_point", "position_m"):
            if k in it and isinstance(it[k], (list, tuple)) and len(it[k]) == 3:
                return np.array(it[k], dtype=float)
        if all(k in it for k in ("x", "y", "z")):
            return np.array([it["x"], it["y"], it["z"]], dtype=float)
        for v in it.values():
            if isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
                return np.array(v, dtype=float)
    if isinstance(it, (list, tuple)) and len(it) == 3:
        return np.array(it, dtype=float)
    return None


def calibrate(robot, pkt, depth, cam="front"):
    """Return a function depth->world cloud for `cam`, validated against the tool, or None."""
    K = _as_K(_camera_entry(pkt, "camera_intrinsics", cam))
    pose = _as_pose(_camera_entry(pkt, "camera_poses", cam))
    log("calib", cam, "K", None if K is None else np.round(K, 2).tolist(), "pose ok", pose is not None)
    step_id = None
    try:
        step_id = robot.last_step_id
    except Exception:
        pass
    rng = np.random.default_rng(0)
    pix = []
    H, W = depth.shape
    while len(pix) < 16:
        u, v = int(rng.integers(20, W - 20)), int(rng.integers(40, H - 10))
        if depth[v, u] > 0:
            pix.append((u, v))
    try:
        ref = tool_points(robot, step_id, cam, pix)
    except Exception as e:
        log("pixel_to_position failed", repr(e)[:300])
        ref = [None] * len(pix)
    good = [(p, r) for p, r in zip(pix, ref) if r is not None]
    log("tool reference points:", len(good))
    if K is None or pose is None:
        return None, good
    R, t = pose
    best = None
    for conv in ("gl", "cv"):
        for off in (0.0, 0.5):
            P = backproject(depth, K, R, t, conv, off)
            if not good:
                continue
            errs = [np.linalg.norm(P[v, u] - r) for (u, v), r in good]
            med = float(np.median(errs))
            if best is None or med < best[0]:
                best = (med, conv, off)
    if best is None:
        # no reference: trust documented OpenGL convention
        log("calib: no tool reference; using documented OpenGL convention")
        return (lambda d: backproject(d, K, R, t, "gl", 0.5)), good
    log("calib best: median err %.4f conv %s off %.1f" % best)
    if best[0] > 0.015:
        return None, good
    _, conv, off = best
    return (lambda d: backproject(d, K, R, t, conv, off)), good


def tool_cloud(robot, depth, cam="front", stride=4):
    """Point cloud from the tool alone, on a pixel grid (slow fallback)."""
    H, W = depth.shape
    P = np.full((H, W, 3), np.nan)
    pix = [(u, v) for v in range(0, H, stride) for u in range(0, W, stride) if depth[v, u] > 0]
    pts = tool_points(robot, getattr(robot, "last_step_id", None), cam, pix)
    for (u, v), p in zip(pix, pts):
        if p is not None:
            P[v:v + stride, u:u + stride] = p
    return P


# ------------------------------------------------------------------ perception

def table_height(P, valid):
    z = P[..., 2][valid]
    z = z[np.isfinite(z)]
    hist, edges = np.histogram(z, bins=np.arange(np.percentile(z, 1) - 0.01, np.percentile(z, 99) + 0.01, 0.002))
    k = int(np.argmax(hist))
    return float((edges[k] + edges[k + 1]) / 2)


def segment(P, valid, table_z):
    h = P[..., 2] - table_z
    obj = valid & np.isfinite(h) & (h > 0.005) & (h < 0.25)
    lab, n = ndimage.label(obj)
    comps = []
    for i in range(1, n + 1):
        m = lab == i
        area = int(m.sum())
        if area < 8:
            continue
        ys, xs = np.nonzero(m)
        pts = P[m]
        hh = h[m]
        ext = np.ptp(pts[:, :2], axis=0)
        comps.append({"id": i, "mask": m, "area": area, "touch_top": bool(ys.min() <= 2),
                      "hmax": float(np.percentile(hh, 95)), "ext": float(np.max(ext)),
                      "center": pts.mean(0), "uv": (float(xs.mean()), float(ys.mean()))})
    return comps


def classify(comps):
    cands = [c for c in comps if not c["touch_top"] and 0.025 < c["hmax"] < 0.2 and c["ext"] < 0.25]
    box = max(cands, key=lambda c: c["area"]) if cands else None
    pegs = [c for c in comps if c is not box and not c["touch_top"] and c["hmax"] < 0.05
            and 0.03 < c["ext"] < 0.3 and c["area"] >= 10]
    return box, pegs


def peg_geometry(P, rgb, comp, table_z):
    m = comp["mask"]
    pts = P[m]
    cols = rgb[m] if rgb is not None else None
    top_all = float(np.percentile(pts[:, 2], 90))
    sel = pts[:, 2] > top_all - 0.004  # top surface only: side faces bias the centre
    if sel.sum() >= 8:
        pts = pts[sel]
        cols = cols[sel] if cols is not None else None
    xy = pts[:, :2]
    c = xy.mean(0)
    U, S, Vt = np.linalg.svd(xy - c, full_matrices=False)
    a = Vt[0] / np.linalg.norm(Vt[0])
    s = (xy - c) @ a
    smin, smax = np.percentile(s, 2), np.percentile(s, 98)
    L = float(smax - smin)
    mid = (smin + smax) / 2
    center = c + a * mid
    top = float(np.percentile(pts[:, 2], 90))
    thick = max(0.01, min(0.05, top - table_z))
    colA = colB = None
    if cols is not None:
        colA = np.median(cols[s > mid + 0.1 * L], axis=0)
        colB = np.median(cols[s < mid - 0.1 * L], axis=0)
    return {"center": center, "axis": a, "L": L, "top": top, "thick": thick,
            "endA": center + a * L / 2, "endB": center - a * L / 2, "colA": colA, "colB": colB}


def chroma(c):
    c = np.asarray(c, dtype=float)
    return c / (c.sum(-1, keepdims=True) + 1e-6)


def held_end_from_wrist(video, colA, colB):
    """+1 if half A (endA side) is held, -1 if half B, 0 unknown."""
    if colA is None or colB is None:
        return 0
    n = len(video)
    votes = 0.0
    used = 0
    cA, cB = chroma(colA), chroma(colB)
    sep = np.linalg.norm(cA - cB)
    for idx in range(int(n * 0.3), int(n * 0.85), max(1, n // 60)):
        img = load_rgb(video[idx], "wrist")
        if img is None:
            continue
        low = np.median(img[200:236, 150:186].reshape(-1, 3), axis=0)
        high = np.median(img[80:120, 150:186].reshape(-1, 3), axis=0)
        if np.all(low > 245):
            continue  # saturated by a light
        cl, ch = chroma(low), chroma(high)
        wh = 0.0 if np.all(high > 245) else 0.5
        dA = np.linalg.norm(cl - cA) + wh * np.linalg.norm(ch - cB)
        dB = np.linalg.norm(cl - cB) + wh * np.linalg.norm(ch - cA)
        if min(np.linalg.norm(cl - cA), np.linalg.norm(cl - cB)) > 0.6 * sep + 0.04:
            continue  # the between-finger patch does not look like either half
        votes += np.sign(dB - dA) * min(1.0, abs(dB - dA) / (sep + 1e-6))
        used += 1
    log("held-end votes %.2f over %d frames (sep %.3f)" % (votes, used, sep))
    if used == 0 or abs(votes) < 0.5:
        return 0
    return 1 if votes > 0 else -1


def box_geometry(P, comp, table_z):
    m = comp["mask"]
    pts = P[m]
    H = float(np.percentile(pts[:, 2], 95) - table_z)
    topm = pts[:, 2] > table_z + 0.85 * H
    tp = pts[topm][:, :2] if topm.sum() > 10 else pts[:, :2]
    c = tp.mean(0)
    # yaw by minimising the bounding-square area over angles (square-ish top face)
    best = None
    for ang in np.radians(np.arange(0, 90, 1.0)):
        u = np.array([np.cos(ang), np.sin(ang)])
        v = np.array([-u[1], u[0]])
        a = (tp - c) @ u
        b = (tp - c) @ v
        area = np.ptp(a) * np.ptp(b)
        if best is None or area < best[0]:
            best = (area, u, v, np.ptp(a), np.ptp(b), (a.max() + a.min()) / 2, (b.max() + b.min()) / 2)
    _, u, v, la, lb, oa, ob = best
    c = c + u * oa + v * ob
    return {"center": c, "u": u, "v": v, "size_u": float(la), "size_v": float(lb), "H": H}


# ------------------------------------------------------------------ main fallback

def run_vision(robot, pkt, video, execf):
    if ndimage is None:
        log("vision: scipy unavailable")
        return False
    import rio as _rio
    robot.last_step_id = _rio.find_step_id(pkt) or 0
    ex = execf[-1] if execf else None
    depth = load_depth(ex, "front") if ex else None
    rgb = load_rgb(ex, "front") if ex else None
    to_world = None
    if depth is None:
        log("vision: no execution depth file -> tool grid cloud only")
        depth = np.ones((256, 256))
        if rgb is None:
            log("vision: no execution rgb file either")
            return False
    else:
        to_world, ref = calibrate(robot, pkt, depth, "front")
    if to_world is not None:
        P = to_world(depth)
    else:
        log("vision: using tool grid cloud")
        P = tool_cloud(robot, depth, "front", stride=4)
    valid = (depth > 0) & np.all(np.isfinite(P), -1)
    tz = table_height(P, valid)
    comps = segment(P, valid, tz)
    box, pegs = classify(comps)
    log("vision: table z %.4f, comps %d, pegs %d, box %s" % (
        tz, len(comps), len(pegs), None if box is None else (box["area"], round(box["hmax"], 3))))
    if box is None or not pegs:
        return False

    # V3 target peg from the video
    d0 = load_depth(video[0], "front") if video else None
    target = None
    if d0 is not None:
        late = [load_depth(video[i], "front") for i in range(int(len(video) * 0.6), len(video), max(1, len(video) // 40))]
        late = [d for d in late if d is not None]
        scores = []
        for pg in pegs:
            m = pg["mask"] & (d0 > 0)
            fr = [float(np.mean((d[m] - d0[m]) > 0.004)) for d in late] if late and m.sum() else [0]
            scores.append(float(np.median(fr)))
        log("vision: peg removal scores", np.round(scores, 2).tolist())
        if scores and max(scores) > 0.25:
            target = pegs[int(np.argmax(scores))]
    if target is None:
        # colour change fallback
        r0 = load_rgb(video[0], "front") if video else None
        lateim = [load_rgb(video[i], "front") for i in range(int(len(video) * 0.6), len(video), max(1, len(video) // 40))]
        lateim = [x for x in lateim if x is not None]
        if r0 is not None and lateim:
            scores = []
            for pg in pegs:
                m = pg["mask"]
                fr = [float(np.mean(np.abs(x[m] - r0[m]).sum(-1) > 60)) for x in lateim]
                scores.append(float(np.median(fr)))
            log("vision: peg rgb-change scores", np.round(scores, 2).tolist())
            target = pegs[int(np.argmax(scores))]
        else:
            target = pegs[0]
    pg = peg_geometry(P, rgb, target, tz)
    log("vision: target peg center", np.round(pg["center"], 3), "axis", np.round(pg["axis"], 3),
        "L %.3f thick %.3f" % (pg["L"], pg["thick"]))

    # V4 held end
    held = held_end_from_wrist(video, pg["colA"], pg["colB"])
    if held == 0:
        held = 1
        log("vision: held end unknown -> guess A")
    a_held = pg["axis"] * held            # unit vector from peg centre toward the held end
    grasp_xy = pg["center"] + a_held * pg["L"] / 4
    tip_dist = 0.75 * pg["L"]               # grasp point to free tip

    # V5 box and insertion face
    bx = box_geometry(P, box, tz)
    col_held = pg["colA"] if held > 0 else pg["colB"]
    face_n, hole_xy, hole_z = insertion_face(robot, to_world, video, d0, bx, tz, pg, col_held)
    log("vision: box c", np.round(bx["center"], 3), "H %.3f" % bx["H"], "face n", np.round(face_n, 3),
        "hole", np.round(hole_xy, 3), "z %.3f" % hole_z)

    # V6 grasp
    ok = grasp(robot, grasp_xy, pg, tz, a_held, face_n)
    if not ok:
        log("vision: grasp failed")
        return False
    # V7 insert
    return insert(robot, a_held, tip_dist, face_n, hole_xy, hole_z, bx, tz, pg)


def insertion_face(robot, to_world, video, d0, bx, tz, pg, col_held=None):
    c = bx["center"]
    half = 0.5 * max(bx["size_u"], bx["size_v"])
    normals = [bx["u"], -bx["u"], bx["v"], -bx["v"]]
    hole_z = tz + 0.5 * bx["H"]
    dirv = None
    tang_off = 0.0
    if to_world is not None and d0 is not None and video:
        for back in (0, 3, 8, 15):
            dl = load_depth(video[max(0, len(video) - 1 - back)], "front")
            if dl is None:
                continue
            ch = (d0 > 0) & (dl > 0) & ((d0 - dl) > 0.01)
            if ch.sum() < 20:
                continue
            Q = to_world(dl)[ch]
            sel = (np.linalg.norm(Q[:, :2] - c, axis=1) < half + 0.12) & (Q[:, 2] < tz + bx["H"] + 0.06)
            Q = Q[sel]
            if len(Q) < 15:
                continue
            r = Q[:, :2].mean(0) - c
            dirv = r
            k = int(np.argmax([np.dot(r, n) for n in normals]))
            n = normals[k]
            # points just outside the chosen face: hole height and lateral offset
            dn = (Q[:, :2] - c) @ n
            near = Q[(dn > half - 0.005) & (dn < half + 0.06)]
            # the held half of the peg protrudes from the hole: use its colour when visible
            img = load_rgb(video[max(0, len(video) - 1 - back)], "front")
            if col_held is not None and img is not None:
                C = img[ch][sel]
                cd = np.linalg.norm(chroma(C) - chroma(col_held), axis=1)
                bright = np.abs(C.mean(1) - np.mean(col_held)) < 80
                pm = (cd < 0.06) & bright & (dn > half - 0.005) & (dn < half + 0.6 * pg["L"])
                if pm.sum() >= 6:
                    Qp = Q[pm]
                    t = np.array([-n[1], n[0]])
                    tang_off = float(np.clip(np.median((Qp[:, :2] - c) @ t), -0.4 * half, 0.4 * half))
                    ztop = float(np.percentile(Qp[:, 2], 90))
                    hole_z = float(np.clip(ztop - pg["thick"] / 2, tz + pg["thick"] / 2 + 0.002,
                                           tz + bx["H"] - pg["thick"] / 2 - 0.002))
                    log("vision: hole from held-colour points (%d): tang %.4f z %.4f" % (pm.sum(), tang_off, hole_z))
                    log("vision: side from video end (back %d): r %s" % (back, np.round(r, 3)))
                    break
            if len(near) >= 8:
                t = np.array([-n[1], n[0]])
                tang_off = float(np.clip(np.median((near[:, :2] - c) @ t), -0.3 * half, 0.3 * half))
                zlow = float(np.percentile(near[:, 2], 10))
                est = zlow  # lowest visible points of peg/fingers near the face
                hole_z = float(np.clip(est, tz + pg["thick"] / 2 + 0.002, tz + bx["H"] - pg["thick"] / 2 - 0.002))
            log("vision: side from video end (back %d): r %s, %d near-face pts" % (back, np.round(r, 3), len(near)))
            break
    if dirv is None:
        # no video evidence: face nearest to the robot base side is unknown; take +u
        log("vision: no side evidence; default face +u")
        dirv = bx["u"]
    k = int(np.argmax([np.dot(dirv, n) for n in normals]))
    n = normals[k]
    t = np.array([-n[1], n[0]])
    face_half = 0.5 * (bx["size_u"] if k < 2 else bx["size_v"])
    hole_xy = c + n * face_half + t * tang_off
    return n, hole_xy, hole_z


def closing_dir(robot):
    """World-frame horizontal finger closing direction (TCP local y)."""
    R = robot.R()
    y = R[:, 1].copy()
    y[2] = 0
    if np.linalg.norm(y) < 1e-3:
        y = R[:, 0].copy(); y[2] = 0
    return y / np.linalg.norm(y)


def yaw_between(v_from, v_to):
    return geom.wrap(np.arctan2(v_to[1], v_to[0]) - np.arctan2(v_from[1], v_from[0]))


def grasp(robot, grasp_xy, pg, tz, a_held, face_n):
    R0 = robot.R()
    cdir = closing_dir(robot)
    a = pg["axis"]
    perp = np.array([-a[1], a[0]])
    # two valid yaws (fingers symmetric); pick the one minimising later total rotation
    best = None
    d_ins = -a_held
    for target_c in (perp, -perp):
        th = yaw_between(cdir, target_c)
        Rg = geom.rot_z(th) @ R0
        phi = yaw_between(d_ins, -face_n)  # rotation needed after grasp (world yaw)
        cost = max(abs(th), abs(geom.wrap(th + phi)))
        if best is None or cost < best[0]:
            best = (cost, th, Rg)
    _, th, Rg = best
    z_grasp = tz + max(0.5 * pg["thick"], 0.012)
    above = np.array([grasp_xy[0], grasp_xy[1], tz + 0.12])
    log("grasp: yaw %.2f rad, z %.3f" % (th, z_grasp))
    robot.goto(above, Rg, -1.0, tol=0.006, rtol=0.04, max_steps=80, label="above-peg")
    robot.goto(np.array([grasp_xy[0], grasp_xy[1], z_grasp + 0.03]), Rg, -1.0, tol=0.004, max_steps=40,
               max_pos=0.3, label="pre-grasp")
    robot.goto(np.array([grasp_xy[0], grasp_xy[1], z_grasp]), Rg, -1.0, tol=0.003, max_steps=40,
               max_pos=0.15, label="grasp-down")
    ws = robot.grip_settle(robot.pos(), Rg, 1.0, min_n=8, max_n=25)
    w = robot.width()
    log("grasp: widths", ["%.4f" % x for x in ws[-3:]])
    robot.grasp_R = robot.R()
    robot.grasp_dz = robot.pos()[2] - (tz + 0.5 * pg["thick"])
    lift = robot.pos() + np.array([0, 0, 0.12])
    robot.goto(lift, None, 1.0, tol=0.006, max_steps=40, max_pos=0.4, label="lift")
    return w > 0.003


def insert_candidates(z_est, tz, bx, pg, tang0):
    """Ordered (lateral, z) hole-centre hypotheses: estimate first, then a vertical scan."""
    zlo = tz + 0.5 * pg["thick"] + 0.003
    zhi = tz + bx["H"] - 0.5 * pg["thick"] - 0.003
    if zhi < zlo:
        zlo = zhi = tz + 0.5 * bx["H"]
    cands = [(tang0, z_est), (tang0, tz + 0.5 * bx["H"])]
    zs = sorted(np.arange(zlo, zhi + 1e-6, 0.004), key=lambda z: abs(z - z_est))
    for z in zs:
        cands.append((tang0, float(z)))
    for dl in (0.004, -0.004):
        for z in zs[:6]:
            cands.append((tang0 + dl, float(z)))
    out = []
    for c in cands:
        if all(abs(c[0] - o[0]) > 1e-3 or abs(c[1] - o[1]) > 0.0015 for o in out):
            out.append(c)
    return out


def insert(robot, a_held, tip_dist, n, hole_xy, hole_z, bx, tz, pg):
    Rg = getattr(robot, "grasp_R", robot.R())
    d_body = Rg.T @ np.array([-a_held[0], -a_held[1], 0.0])
    d_now = robot.R() @ d_body
    phi = yaw_between(d_now[:2], -n)
    R_ins = geom.rot_z(phi) @ robot.R()
    dz = getattr(robot, "grasp_dz", 0.0)
    safe_z = tz + bx["H"] + 0.08
    n3 = np.array([n[0], n[1], 0.0])
    t3 = np.array([-n[1], n[0], 0.0])
    face_c = np.array([hole_xy[0], hole_xy[1], 0.0])
    tang0 = 0.0  # hole_xy already contains the lateral estimate
    pre0 = face_c + n3 * (tip_dist + 0.04)
    pre0[2] = hole_z + dz
    log("insert: yaw %.2f, pre %s" % (phi, np.round(pre0, 3)))
    robot.goto(np.array([robot.pos()[0], robot.pos()[1], safe_z]), R_ins, 1.0, tol=0.006, rtol=0.03,
               max_steps=60, max_pos=0.3, max_rot=0.3, label="rotate")
    robot.goto(np.array([pre0[0], pre0[1], safe_z]), R_ins, 1.0, tol=0.005, max_steps=60, max_pos=0.4,
               label="above-pre")
    depth_goal = min(0.5 * pg["L"] + 0.005, 2 * 0.5 * max(bx["size_u"], bx["size_v"]) - 0.01)
    cands = insert_candidates(hole_z, tz, bx, pg, tang0)
    log("insert: %d hole hypotheses, depth goal %.3f" % (len(cands), depth_goal))
    for k, (lat, zc) in enumerate(cands):
        if not robot.can_move():
            break
        start = face_c + n3 * (tip_dist + 0.015) + t3 * lat
        start[2] = zc + dz
        # back away from the face before changing height
        cur = robot.pos()
        if np.dot(cur - face_c, n3) < tip_dist + 0.012:
            robot.goto(cur + n3 * 0.015, R_ins, 1.0, tol=0.004, max_steps=8, max_pos=0.2, label="back-off")
        robot.goto(start, R_ins, 1.0, tol=0.0025, rtol=0.02, max_steps=25, max_pos=0.2, label="start %d" % k)
        hist = []
        sdist = 0.0
        while robot.can_move():
            sdist += 0.004
            tgt = start - n3 * sdist
            robot.goto(tgt, R_ins, 1.0, tol=0.002, max_steps=2, max_pos=0.1, label="push")
            prog = float(np.dot(start - robot.pos(), n3))
            hist.append(prog)
            depth_in = float(tip_dist - np.dot(robot.pos() - face_c, n3))  # tip depth past the face
            if robot.complete or robot.ended:
                log("insert: complete, hypothesis %d (lat %.3f z %.3f), tip depth %.3f" % (k, lat, zc, depth_in))
                return True
            if depth_in > depth_goal:
                log("insert: reached depth goal, hypothesis %d (lat %.3f z %.3f)" % (k, lat, zc))
                robot.hold(1.0, 3)
                return True
            if len(hist) > 3 and hist[-1] - hist[-4] < 0.002:
                log("insert: hypothesis %d (lat %.3f z %.3f) stalled, tip depth %.3f" % (k, lat, zc, depth_in))
                break
            if sdist > tip_dist + 0.08:
                break
    return False
