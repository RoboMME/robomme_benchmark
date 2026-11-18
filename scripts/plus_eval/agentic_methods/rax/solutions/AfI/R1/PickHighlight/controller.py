"""PickHighlight controller (RAX, AfI R1) - written offline, never seen live.

Pipeline (each step logged; any exception -> logged, continue with next safe step):
  S0 init: observe, status, instruction, calibration, reset manifest
  S1 perceive: depth -> world cloud (convention self-checked against robomme_pixel_to_position)
               -> table height -> cube segmentation (execution frame)
  S2 highlight: per-cube colour-change score over conditioning-video frames (depth-gated,
               shadow-filtered) -> target set (threshold / instruction count) ordered by first highlight
               fallback: watch live frames for 30 steps if no video
  S3 for each target: hover -> wrist refine -> yaw align -> descend -> close -> verify width ->
               lift -> check task_complete -> place back -> open -> retreat; retry grasp up to 3x
  S4 if not complete: re-pick last target and hold lifted until steps run out
Transitions: any EpisodeOver -> return. Wall deadline -> return.
"""
import math
import os
import re
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import io_util as U
import percept as PC
from geom import (CONVENTIONS, Rz, cam_points, hand_yaw, rot_err_world, rotvec_to_R,
                  straighten_down, wrap_pi, wrap_quarter)
from robot import EpisodeOver, Robot

VERSION = "pickhl-v1"
HOVER_H = 0.13      # hover height above table (m)
LIFT_H = 0.14       # lift height of TCP above table
FINGER_REACH = 0.045
WALL_DEADLINE = 1080.0  # s, controller limit 1200
V_FREE = 0.02       # m per control step in free space (0.4 action units); untested live
V_LOAD = 0.015      # m per step while holding a cube


# ---------------------------------------------------------------- helpers
def parse_instruction(text):
    t = (text or "").lower()
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "a": 1, "an": 1}
    n_hint = None
    m = re.search(r"\b(\d+|two|three|four|five|six)\s+(?:\w+\s+){0,2}(cubes|blocks|objects)", t)
    if m:
        g = m.group(1)
        n_hint = int(g) if g.isdigit() else words.get(g)
    plural = bool(re.search(r"\b(cubes|blocks|objects|all|each|every)\b", t))
    singular = (not plural) and bool(re.search(r"\b(cube|block|object)\b", t))
    which = None
    if re.search(r"(last|latest|final)\s+(\w+\s+)?highlight|highlighted\s+(\w+\s+)?(last|latest)", t):
        which = "last"
    elif re.search(r"first\s+(\w+\s+)?highlight|highlighted\s+(\w+\s+)?first", t):
        which = "first"
    return dict(n_hint=n_hint, singular=singular, plural=plural, which=which,
                order=("order" in t or "sequence" in t or "first" in t), button=("button" in t))


class Scene:
    pass


def load_view_from_files(files, cam):
    rgb = dep = None
    if (cam, "rgb") in files:
        try:
            k, a = U.load_file_image(files[(cam, "rgb")][0])
            rgb = a if k == "rgb" else None
        except Exception as e:
            U.log("rgb load fail", e)
    if (cam, "depth") in files:
        try:
            k, a = U.load_file_image(files[(cam, "depth")][0])
            if a.ndim == 3:
                a = a[..., 0]
            dep = a.astype(float) * float(files[(cam, "depth")][1] or 0.001)
        except Exception as e:
            U.log("depth load fail", e)
    return rgb, dep


def current_views(robot):
    """Current observation -> {cam: (rgb, depth_m)} from inline blocks / files."""
    views = {}
    imgs = U.reply_images(robot.reply)
    cams = U.assign_cameras(imgs)
    scale = U.depth_scale_hint(robot.meta)
    for cam in ("front", "wrist"):
        rgb = cams.get((cam, "rgb"))
        dep = cams.get((cam, "depth"))
        if dep is not None:
            dep = dep * scale
        views[cam] = [rgb, dep]
    # files referenced in metadata (depth attachments as paths)
    try:
        files = {}
        for p, k, v in U.walk(robot.meta):
            if k == "files" and isinstance(v, list):
                files = U.frame_files({"files": v})
        for cam in ("front", "wrist"):
            rgb, dep = load_view_from_files(files, cam) if files else (None, None)
            if views[cam][0] is None and rgb is not None:
                views[cam][0] = rgb
            if views[cam][1] is None and dep is not None:
                views[cam][1] = dep
        if any(views[c][1] is None for c in views):
            mp = U.find_manifest_path(robot.meta)
            if mp:
                man, mpath = U.read_manifest(mp)
                fr = man.get("frames", [])
                if fr:
                    sid = robot.step_id
                    cand = [f for f in fr if f.get("step_id") == sid] or fr
                    last = sorted(cand, key=lambda f: f.get("frame_index", 0))[-1]
                    ff = U.frame_files(last, mpath)
                    for cam in ("front", "wrist"):
                        rgb, dep = load_view_from_files(ff, cam)
                        if views[cam][1] is None and dep is not None:
                            views[cam][1] = dep
                        if views[cam][0] is None and rgb is not None:
                            views[cam][0] = rgb
    except Exception as e:
        U.log("current_views file path parse failed", e)
    return views


def tool_points(robot, cam, uvs, radius=1):
    """robomme_pixel_to_position for list of (u,v) -> list of xyz or None."""
    out = []
    for i in range(0, len(uvs), 32):
        chunk = uvs[i:i + 32]
        sel = [{"camera": cam, "u": float(u), "v": float(v)} for (u, v) in chunk]
        try:
            rep = robot.ctx.call("robomme_pixel_to_position", observation_step_id=int(robot.step_id or 0),
                                 selections=sel, depth_window_radius=radius)
            robot._save_first("pixel", rep)
            meta = U.reply_meta(rep)
            pts = parse_points(meta, len(chunk))
        except Exception as e:
            U.log("pixel_to_position failed:", str(e)[:300])
            pts = [None] * len(chunk)
        out.extend(pts)
    return out


def _xyz_of(d):
    if U.is_numlist(d, 3):
        return [float(x) for x in d]
    if isinstance(d, dict):
        if d.get("valid") is False:
            return None
        for k in ("position", "world_position", "position_m", "world", "point", "xyz", "world_point",
                  "point_world", "position_world"):
            if U.is_numlist(d.get(k), 3):
                return [float(x) for x in d[k]]
        if all(k in d for k in ("x", "y", "z")):
            try:
                return [float(d["x"]), float(d["y"]), float(d["z"])]
            except Exception:
                return None
        for k, v in d.items():
            if U.is_numlist(v, 3) and any(s in str(k).lower() for s in ("pos", "world", "point", "xyz")):
                return [float(x) for x in v]
    return None


def parse_points(meta, n):
    lists = []
    for p, k, v in U.walk(meta):
        if isinstance(v, list) and len(v) == n and all(isinstance(x, (dict, list)) for x in v):
            lists.append((len(p), v))
    lists.sort(key=lambda x: x[0])
    for _, L in lists:
        pts = [_xyz_of(x) for x in L]
        if any(p is not None for p in pts):
            # if any item is dict with valid False -> None preserved
            return pts
    return [None] * n


def choose_convention(robot, depth, cal, cam="front"):
    """Pick the back-projection convention that matches the pixel tool; fall back to geometry."""
    H, W = depth.shape
    uvs = []
    for v in np.linspace(20, H - 20, 5):
        for u in np.linspace(20, W - 20, 5):
            ui, vi = int(u), int(v)
            if depth[vi, ui] > 0.01:
                uvs.append((ui, vi))
    res = {"tool_pts": 0}
    best = None
    if uvs:
        pts = tool_points(robot, cam, uvs, radius=0)
        good = [(uv, p) for uv, p in zip(uvs, pts) if p is not None]
        res["tool_pts"] = len(good)
        if len(good) >= 4:
            us = np.array([g[0][0] for g in good], dtype=float)
            vs = np.array([g[0][1] for g in good], dtype=float)
            dd = np.array([depth[int(v), int(u)] for u, v in zip(us, vs)])
            T = np.array([g[1] for g in good])
            for conv in CONVENTIONS:
                P = cam_points(dd, cal["K"], cal["R"], cal["t"], conv, us=us, vs=vs)
                err = float(np.median(np.linalg.norm(P - T, axis=1)))
                if best is None or err < best[0]:
                    best = (err, conv)
            res["best_err"] = best[0]
            res["best_conv"] = best[1]
            res["tool_sample"] = [list(map(float, g[1])) for g in good[:3]]
            if best[0] < 0.02:
                return best[1], res
    # geometry fallback: the convention whose dominant plane is horizontal and largest
    scores = []
    for conv in CONVENTIONS:
        if conv[1] != 0.5 and conv[0] == "gl":
            pass
        try:
            P, valid = PC.world_cloud(depth, cal, conv)
            zt = PC.estimate_table(P, valid)
            if zt is None:
                continue
            frac = PC.plane_horizontality(P, valid, zt[0])
            pref = 0.02 if conv == ("gl", 0.5, False) else 0.0
            scores.append((frac + pref, conv))
        except Exception:
            pass
    scores.sort(key=lambda x: -x[0])
    res["geom_scores"] = [(round(s, 3), c) for s, c in scores[:4]]
    if scores:
        return scores[0][1], res
    return ("gl", 0.5, False), res


def cloud_via_tool(robot, cam, H, W, stride=4):
    """Coarse world cloud from the pixel tool when depth/calibration are unusable."""
    vs = list(range(stride // 2, H, stride))
    us = list(range(stride // 2, W, stride))
    uvs = [(u, v) for v in vs for u in us]
    pts = tool_points(robot, cam, uvs, radius=0)
    P = np.full((len(vs), len(us), 3), np.nan)
    valid = np.zeros((len(vs), len(us)), dtype=bool)
    for idx, p in enumerate(pts):
        if p is not None:
            r, c = divmod(idx, len(us))
            P[r, c] = p
            valid[r, c] = True
    return P, valid, stride


def upsample_mask(m, stride, H, W):
    big = np.kron(m.astype(np.uint8), np.ones((stride, stride), dtype=np.uint8)).astype(bool)
    out = np.zeros((H, W), dtype=bool)
    hh, ww = min(H, big.shape[0]), min(W, big.shape[1])
    out[:hh, :ww] = big[:hh, :ww]
    return out


def save_debug_png(name, rgb, cubes=None, targets=()):
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(rgb.astype(np.uint8)).resize((512, 512), Image.NEAREST)
        dr = ImageDraw.Draw(im)
        if cubes:
            for i, c in enumerate(cubes):
                if c.get("mask") is None:
                    continue
                ys, xs = np.nonzero(c["mask"])
                if len(xs) == 0:
                    continue
                col = (255, 0, 255) if i in targets else (0, 255, 255)
                dr.rectangle([xs.min() * 2, ys.min() * 2, xs.max() * 2 + 1, ys.max() * 2 + 1], outline=col)
                dr.text((xs.min() * 2, max(0, ys.min() * 2 - 12)), str(i), fill=col)
        im.save(U.OUT / name)
    except Exception as e:
        U.log("debug png failed", e)


def glob_manifest():
    """Fallback: newest frame manifest under the run directory with video_demo frames."""
    try:
        import glob
        import json
        best = None
        for p in glob.glob(str(U.WS / "runtime" / "autonomous_controller" / "runs" / "*" / "frames" / "**" /
                               "manifest.json"), recursive=True):
            try:
                m = json.loads(open(p).read())
                nv = sum(1 for f in m.get("frames", []) if str(f.get("phase")) == "video_demo")
                key = (nv > 0, os.path.getmtime(p))
                if best is None or key > best[0]:
                    best = (key, p)
            except Exception:
                pass
        if best:
            U.log("manifest via glob:", best[1])
            return best[1]
    except Exception as e:
        U.log("glob manifest failed", e)
    return None


# ---------------------------------------------------------------- perception
def perceive_cubes(robot, sc, rgb, depth, cal, cam="front", label="exec"):
    """Return (cubes, z_table)."""
    P = valid = None
    stride = 1
    if depth is not None and cal and all(k in cal for k in ("K", "R", "t")):
        if sc.conv is None:
            sc.conv, info = choose_convention(robot, depth, cal, cam)
            U.log("convention:", sc.conv, info)
        P, valid = PC.world_cloud(depth, cal, sc.conv)
    if P is None:
        H, W = (rgb.shape[:2] if rgb is not None else (256, 256))
        U.log("perceive %s: depth/calibration unavailable -> tool cloud" % label)
        P, valid, stride = cloud_via_tool(robot, cam, H, W, stride=2)
    zt = PC.estimate_table(P, valid)
    if zt is None:
        return [], None, P, valid, stride
    z_t, frac = zt
    if sc.z_table is not None and abs(z_t - sc.z_table) > 0.03 and label != "exec":
        z_t = sc.z_table  # keep the first (front) estimate if a close view disagrees
    rgb_s = rgb[stride // 2::stride, stride // 2::stride] if (rgb is not None and stride > 1) else rgb
    if rgb_s is not None and rgb_s.shape[:2] != P.shape[:2]:
        rgb_s = None
    cubes, rej = PC.segment_cubes(P, valid, z_t, rgb_s, cam_pos=(cal or {}).get("t"),
                                  min_pix=max(4, PC.MIN_PIX // (stride * stride)))
    if stride > 1:
        H, W = rgb.shape[:2]
        for c in cubes:
            c["mask"] = upsample_mask(c["mask"], stride, H, W)
    U.log("perceive %s: table z=%.4f (frac %.2f) cubes=%d rejected=%d %s" % (
        label, z_t, frac, len(cubes), len(rej), [r[0] for r in rej][:12]))
    for i, c in enumerate(cubes):
        U.log("  cube %d xy=(%.3f,%.3f) h=%.3f side=%.3f/%.3f yaw=%.1f npix=%d split=%d col=%s" % (
            i, c["xy"][0], c["xy"][1], c["h"], c["side"], c["side_long"], math.degrees(c["yaw"]), c["npix"],
            c["split"], PC.color_name(c["color"])))
    return cubes, z_t, P, valid, stride


def video_frames_from_manifest(sc):
    frames, exec_frame = [], None
    man, mpath = sc.manifest, sc.manifest_path
    if not man:
        return frames, exec_frame
    fr = sorted(man.get("frames", []), key=lambda f: (f.get("frame_index", 0)))
    for f in fr:
        ph = str(f.get("phase", "")).lower()
        files = U.frame_files(f, mpath)
        if ph == "video_demo" or ph.startswith("video"):
            frames.append((f.get("timestamp_s", f.get("frame_index", 0)), files, f))
        elif ph == "execution":
            exec_frame = (files, f)
    return frames, exec_frame


def detect_highlights(robot, sc, cubes, exec_rgb, exec_dep, frames_rgbd, mask_key="mask", cam="front"):
    if not cubes or not frames_rgbd or exec_rgb is None:
        return None
    if not any(c.get(mask_key) is not None for c in cubes):
        return None
    hs = PC.highlight_scores(exec_rgb, exec_dep, cubes, frames_rgbd, mask_key=mask_key)
    S, Sraw = hs["S"], hs["Sraw"]
    T = S.shape[1]

    def peak(A):
        if T >= 4:
            return np.sort(A, axis=1)[:, -2]
        return A.max(axis=1)
    Sany = hs["Sany"]
    pk, pkr, pka = peak(S), peak(Sraw), peak(Sany)
    if pk.max() >= 0.05:
        use, src = S, "gated"
    elif pkr.max() >= 0.08:
        use, src = Sraw, "gated+shadow"
    elif pka.max() >= 0.10:
        use, src = Sany, "ungated"
    else:
        use, src = S, "gated(weak)"
    U.log("highlight score source:", src)
    p = peak(use)
    first = []
    for ci in range(len(cubes)):
        thr = max(0.03, 0.5 * p[ci])
        idx = np.nonzero(use[ci] > thr)[0]
        first.append(int(idx[0]) if len(idx) else 10 ** 6)
    U.log("highlight[%s]: frames=%d global-change median=%.3f max=%.3f bg max=%.3f" % (
        cam, T, float(np.median(hs["GL"])), float(hs["GL"].max()), float(hs["BG"].max())))
    for ci in range(len(cubes)):
        U.log("  cube %d score=%.3f raw=%.3f any=%.3f used=%.3f first_frame=%s" % (
            ci, pk[ci], pkr[ci], pka[ci], p[ci], first[ci]))
    try:
        np.savez_compressed(U.OUT / ("highlight_scores_%s.npz" % cam), S=S, Sraw=Sraw, Sany=Sany, BG=hs["BG"], GL=hs["GL"])
    except Exception:
        pass
    return dict(peak=p, first=first, S=use, gl_median=float(np.median(hs["GL"])))


# ---------------------------------------------------------------- manipulation
def choose_yaw(sc, cube, others, R_now):
    """Yaw offset psi (relative to initial hand yaw) in [-pi/2, pi/2] aligned with the cube,
    choosing the finger-closing direction with the best clearance to neighbours."""
    g0 = sc.g0
    base = wrap_quarter(cube["yaw"] - g0)
    alt = base - math.pi / 2 if base >= 0 else base + math.pi / 2
    opts = []
    for psi in (base, alt):
        R = Rz(psi) @ sc.R_down0
        close_ax = R[:, sc.close_idx][:2]
        n = np.linalg.norm(close_ax)
        close_ax = close_ax / n if n > 1e-6 else np.array([1.0, 0.0])
        pads = [cube["xy"] + s * close_ax * FINGER_REACH for s in (-1, 1)]
        clr = 1.0
        for o in others:
            for pd in pads:
                clr = min(clr, float(np.linalg.norm(o["xy"] - pd)) - o["side"] * 0.5)
        # prefer small rotation from the current hand yaw when both are clear
        cur = math.atan2(R_now[1, sc.close_idx], R_now[0, sc.close_idx])
        rot = abs(wrap_pi(math.atan2(R[1, sc.close_idx], R[0, sc.close_idx]) - cur))
        opts.append((psi, clr, rot))
    a, b = opts
    if min(a[1], b[1]) > 0.03:
        best = a if a[2] <= b[2] else b
    else:
        best = a if a[1] >= b[1] else b
    U.log("yaw options (psi,clear,rot): %s -> psi=%.1f deg" % (
        [(round(math.degrees(o[0]), 1), round(o[1], 3), round(o[2], 2)) for o in opts], math.degrees(best[0])))
    return best[0]


def wrist_refine(robot, sc, cube):
    """Observe with the wrist camera from hover; return refined (xy, yaw) or None."""
    try:
        robot.observe()
        views = current_views(robot)
        rgb, dep = views.get("wrist", (None, None))
        cal = U.get_calibration(robot.meta).get("wrist")
        if dep is None or not cal or not all(k in cal for k in ("K", "R", "t")):
            U.log("wrist refine: no wrist depth/calibration")
            return None
        P, valid = PC.world_cloud(dep, cal, sc.conv)
        # calibration freshness check against the pixel tool (current observation)
        H, W = dep.shape
        uvs = [(int(u), int(v)) for v in (H * 0.25, H * 0.5, H * 0.75) for u in (W * 0.25, W * 0.75)
               if valid[int(v), int(u)]]
        if uvs:
            tp = tool_points(robot, "wrist", uvs, radius=0)
            errs = [float(np.linalg.norm(P[v, u] - np.array(t))) for (u, v), t in zip(uvs, tp) if t is not None]
            if errs:
                me = float(np.median(errs))
                U.log("wrist calib check: median err %.4f over %d" % (me, len(errs)))
                if me > 0.012:
                    U.log("wrist refine: calibration mismatch -> skip")
                    return None
        zt = PC.estimate_table(P, valid)
        if zt is not None and abs(zt[0] - sc.z_table) > 0.015 and zt[1] > 0.2:
            U.log("wrist refine: table height mismatch %.4f vs %.4f -> skip" % (zt[0], sc.z_table))
            return None
        cubes, rej = PC.segment_cubes(P, valid, sc.z_table, rgb, cam_pos=cal.get("t"))
        best = None
        for c in cubes:
            dxy = float(np.linalg.norm(c["xy"] - cube["xy"]))
            if dxy < 0.035 and (best is None or dxy < best[0]):
                best = (dxy, c)
        U.log("wrist refine: %d cubes, rejected %s, nearest=%s" % (
            len(cubes), [r[0] for r in rej][:8], None if best is None else round(best[0], 4)))
        if best is None:
            return None
        c = best[1]
        if not (0.6 * cube["side"] < c["side"] < 1.6 * cube["side"] + 0.01):
            U.log("wrist refine: side mismatch %.3f vs %.3f" % (c["side"], cube["side"]))
            return None
        if sc.wrist_png_saved < 3 and rgb is not None:
            save_debug_png("wrist_refine_%d.png" % sc.wrist_png_saved, rgb, [c], [0])
            sc.wrist_png_saved += 1
        return c
    except EpisodeOver:
        raise
    except Exception as e:
        U.log("wrist refine error", e, traceback.format_exc()[-400:])
        return None


def go_hover(robot, sc, xy, R_goal, grip):
    z_h = sc.z_table + HOVER_H
    p = robot.pos()
    if p[2] < z_h - 0.02:
        robot.move_to([p[0], p[1], z_h], None, grip, tol=0.01, vmax=V_FREE, label="rise")
    return robot.move_to([xy[0], xy[1], z_h], R_goal, grip, tol=0.005, vmax=V_FREE, label="hover")


def pick_one(robot, sc, ti, cube, others, place=True, hold_forever=False):
    """Returns 'lifted', 'missed', 'failed'."""
    t0 = time.time()
    s0 = robot.step_id
    psi = choose_yaw(sc, cube, others, robot.R())
    R_goal = Rz(psi) @ sc.R_down0
    robot.gripper(-1, 6)
    ok, d = go_hover(robot, sc, cube["xy"], R_goal, -1)
    U.log("target %d hover ok=%s d=%.4f step=%s" % (ti, ok, d, robot.step_id))
    ref = wrist_refine(robot, sc, cube)
    xy = cube["xy"].copy()
    if ref is not None:
        U.log("wrist refine: xy (%.4f,%.4f)->(%.4f,%.4f) yaw %.1f->%.1f side %.3f h %.3f" % (
            xy[0], xy[1], ref["xy"][0], ref["xy"][1], math.degrees(cube["yaw"]), math.degrees(ref["yaw"]),
            ref["side"], ref["h"]))
        xy = ref["xy"].copy()
        cube = dict(cube)
        cube["xy"] = xy
        cube["yaw"] = ref["yaw"]
        if 0.012 < ref["h"] < 0.09:
            cube["h"] = 0.5 * (cube["h"] + ref["h"])
        psi = choose_yaw(sc, cube, others, robot.R())
        R_goal = Rz(psi) @ sc.R_down0
        robot.move_to([xy[0], xy[1], sc.z_table + HOVER_H], R_goal, -1, tol=0.004, vmax=0.01, label="hover2")
    h = cube["h"]
    z_grasp = sc.z_table + max(0.5 * h, 0.011)
    z_pre = sc.z_table + h + 0.035
    robot.move_to([xy[0], xy[1], z_pre], R_goal, -1, tol=0.005, vmax=V_FREE, label="pre")
    ok, d = robot.move_to([xy[0], xy[1], z_grasp], R_goal, -1, tol=0.004, vmax=0.008, label="descend")
    U.log("target %d descend ok=%s d=%.4f z=%.4f (goal %.4f) width=%.4f" % (
        ti, ok, d, robot.pos()[2], z_grasp, robot.width()))
    w = robot.gripper(1, 12)
    U.log("target %d closed width=%.4f" % (ti, w))
    if w < 0.004:
        U.log("target %d: gripper closed on nothing" % ti)
        robot.gripper(-1, 6)
        robot.move_to([xy[0], xy[1], sc.z_table + HOVER_H], R_goal, -1, tol=0.01, vmax=0.015, label="up-miss")
        return "missed", cube
    ok, d = robot.move_to([xy[0], xy[1], sc.z_table + LIFT_H], R_goal, 1, tol=0.006, vmax=V_LOAD, label="lift")
    robot.hold(1, 4)
    w2 = robot.width()
    U.log("target %d lifted ok=%s z=%.4f width=%.4f complete=%s steps %s->%s wall %.1fs" % (
        ti, ok, robot.pos()[2], w2, robot.complete, s0, robot.step_id, time.time() - t0))
    if w2 < 0.004:
        U.log("target %d: dropped during lift" % ti)
        robot.gripper(-1, 6)
        return "missed", cube
    if robot.complete:
        return "lifted", cube
    if hold_forever:
        while not robot.ended and robot.elapsed() < robot.deadline:
            robot.hold(1, 20)
        return "lifted", cube
    if place:
        z_place = sc.z_table + max(0.5 * h, 0.011) + 0.008
        robot.move_to([xy[0], xy[1], z_place], R_goal, 1, tol=0.005, vmax=V_LOAD, label="place")
        robot.gripper(-1, 8)
        robot.move_to([xy[0], xy[1], sc.z_table + HOVER_H], R_goal, -1, tol=0.01, vmax=V_FREE, label="retreat")
    return "lifted", cube


def reacquire(robot, sc, cube, attempt):
    """Escalation after a missed grasp: move the arm out of the front view, re-perceive, and take
    the nearest fresh detection; on the 2nd retry without a detection, offset along the closing axis."""
    try:
        home = sc.home
        robot.move_to([robot.pos()[0], robot.pos()[1], sc.z_table + HOVER_H], None, -1, tol=0.01, vmax=0.015,
                      label="re-up")
        robot.move_to(home, None, -1, tol=0.01, vmax=0.015, label="re-home")
        views = current_views(robot)
        rgb, dep = views["front"]
        cal = U.get_calibration(robot.meta).get("front") or sc.cal_front
        if rgb is not None and dep is not None and cal:
            cubes, z_t, _, _, _ = perceive_cubes(robot, sc, rgb, dep, cal, "front", "reacquire")
            best = None
            for c in cubes:
                d = float(np.linalg.norm(c["xy"] - cube["xy"]))
                if d < 0.05 and (best is None or d < best[0]):
                    best = (d, c)
            if best is not None:
                U.log("reacquire: moved %.4f m" % best[0])
                nc = dict(cube)
                nc.update(xy=best[1]["xy"], yaw=best[1]["yaw"], h=best[1]["h"], side=best[1]["side"])
                return nc
        U.log("reacquire: no fresh detection")
    except EpisodeOver:
        raise
    except Exception as e:
        U.log("reacquire error", e)
    if attempt >= 2:
        nc = dict(cube)
        ang = cube["yaw"] + (math.pi / 4)
        nc["xy"] = cube["xy"] + 0.008 * np.array([math.cos(ang), math.sin(ang)])
        U.log("reacquire: offset retry")
        return nc
    return cube


# ---------------------------------------------------------------- main
def run(ctx):
    U.OUT.mkdir(parents=True, exist_ok=True)
    U.log("controller", VERSION, "start")
    robot = Robot(ctx, deadline_s=WALL_DEADLINE)
    sc = Scene()
    sc.conv = None
    sc.z_table = None
    sc.wrist_png_saved = 0
    rep, meta = robot.observe()
    st = robot.status()
    instr = U.get_instruction(meta) or U.get_instruction(st)
    ip = parse_instruction(instr)
    U.log("instruction:", repr(instr), ip)
    U.log("state:", None if robot.state is None else np.round(robot.state, 4).tolist(), "step", robot.step_id,
          "limit", robot.step_limit)
    if robot.state is None:
        U.log("FATAL: no state parsed")
        return
    # manifest
    sc.manifest = sc.manifest_path = None
    mp = U.find_manifest_path(meta)
    if not mp:
        mp = glob_manifest()
    if mp:
        try:
            sc.manifest, sc.manifest_path = U.read_manifest(mp)
            U.save_json("reset_manifest_head.json", U.strip_blobs(
                dict((k, v) for k, v in sc.manifest.items() if k != "frames")))
            fr = sc.manifest.get("frames", [])
            U.log("manifest:", mp, "frames", len(fr), "phases",
                  sorted(set(str(f.get("phase")) for f in fr)))
            if fr:
                U.save_json("reset_manifest_frame0.json", fr[0])
                U.save_json("reset_manifest_framelast.json", fr[-1])
        except Exception as e:
            U.log("manifest read failed", e)
    cal_all = U.get_calibration(meta)
    U.log("calibration cams:", {k: sorted(v.keys()) for k, v in cal_all.items()})
    cal = cal_all.get("front")
    sc.cal_front = cal
    sc.home = robot.pos()
    # orientation reference
    R0 = robot.R()
    sc.R_down0, sc.app_idx, sc.app_sgn, tilt = straighten_down(R0)
    sc.close_idx = 1 if sc.app_idx == 2 else (0 if sc.app_idx != 0 else 1)
    sc.g0 = math.atan2(sc.R_down0[1, sc.close_idx], sc.R_down0[0, sc.close_idx])
    U.log("orientation: approach col %d sgn %d tilt %.1f deg, close col %d yaw0 %.1f" % (
        sc.app_idx, sc.app_sgn, math.degrees(tilt), sc.close_idx, math.degrees(sc.g0)))
    # execution frame images
    frames, exec_frame = video_frames_from_manifest(sc)
    views = current_views(robot)
    e_rgb, e_dep = views["front"]
    if exec_frame is not None:
        r2, d2 = load_view_from_files(exec_frame[0], "front")
        if r2 is not None:
            e_rgb = r2
        if d2 is not None:
            e_dep = d2 if e_dep is None else d2
    if e_rgb is None:
        U.log("FATAL-ish: no front rgb")
        e_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    U.log("front rgb", e_rgb.shape, "depth", None if e_dep is None else (e_dep.shape, float(np.nanmax(e_dep))))
    cubes, z_t, P, valid, stride = perceive_cubes(robot, sc, e_rgb, e_dep, cal, "front", "exec")
    sc.z_table = z_t
    if z_t is None or not cubes:
        U.log("no cubes found; abort")
        save_debug_png("exec_front.png", e_rgb)
        return
    # wrist camera at the start pose: extra detections (cubes hidden from the front view)
    w_rgb = w_dep = None
    try:
        if exec_frame is not None:
            w_rgb, w_dep = load_view_from_files(exec_frame[0], "wrist")
        if w_rgb is None or w_dep is None:
            vw = views.get("wrist", (None, None))
            w_rgb = w_rgb if w_rgb is not None else vw[0]
            w_dep = w_dep if w_dep is not None else vw[1]
        calw = cal_all.get("wrist")
        if w_dep is not None and calw and all(k in calw for k in ("K", "R", "t")) and sc.conv is not None:
            Pw, vw_ = PC.world_cloud(w_dep, calw, sc.conv)
            cw, rejw = PC.segment_cubes(Pw, vw_, z_t, w_rgb, cam_pos=calw["t"])
            U.log("wrist start view: %d cubes (rejected %s)" % (len(cw), [r[0] for r in rejw][:8]))
            for c2 in cw:
                d = [float(np.linalg.norm(c2["xy"] - c1["xy"])) for c1 in cubes]
                j = int(np.argmin(d)) if d else -1
                if j >= 0 and d[j] < 0.03:
                    cubes[j]["mask_w"] = c2["mask"]
                elif 0.6 * np.median([c["h"] for c in cubes]) < c2["h"] < 1.6 * np.median([c["h"] for c in cubes]):
                    c2["mask_w"] = c2["mask"]
                    c2["mask"] = None
                    U.log("  wrist-only cube xy=(%.3f,%.3f) h=%.3f" % (c2["xy"][0], c2["xy"][1], c2["h"]))
                    cubes.append(c2)
    except Exception as e:
        U.log("wrist start view error", e, traceback.format_exc()[-300:])
    # highlight analysis
    frames_rgbd, frames_w = [], []
    for (t, files, f) in frames:
        rgb, dep = load_view_from_files(files, "front")
        if rgb is not None:
            frames_rgbd.append((t, rgb, dep))
        if any(c.get("mask_w") is not None for c in cubes):
            rgb2, dep2 = load_view_from_files(files, "wrist")
            if rgb2 is not None:
                frames_w.append((t, rgb2, dep2))
    U.log("video frames loaded:", len(frames_rgbd), "front,", len(frames_w), "wrist, of", len(frames))
    hl = detect_highlights(robot, sc, cubes, e_rgb, e_dep, frames_rgbd, "mask", "front")
    hlw = None
    try:
        hlw = detect_highlights(robot, sc, cubes, w_rgb, w_dep, frames_w, "mask_w", "wrist")
    except Exception as e:
        U.log("wrist highlight error", e)
    if hlw is not None and hlw["gl_median"] < 0.2:
        if hl is None:
            hl = hlw
        else:
            pk = np.maximum(hl["peak"], hlw["peak"])
            first = [min(a if hl["peak"][i] > 0.03 else 10 ** 6, b if hlw["peak"][i] > 0.03 else 10 ** 6)
                     for i, (a, b) in enumerate(zip(hl["first"], hlw["first"]))]
            hl = dict(peak=pk, first=first, S=hl["S"], gl_median=hl["gl_median"])
            U.log("combined peaks:", np.round(pk, 3).tolist(), "first:", first)
    elif hlw is not None:
        U.log("wrist video ignored (camera moved: GL median %.3f)" % hlw["gl_median"])
    if hl is None or float(np.max(hl["peak"])) < 0.05:
        # live watch fallback: hold still, collect frames
        U.log("no clear highlight in video -> live watch fallback")
        live = []
        try:
            for k in range(8):
                robot.hold(-1, 5)
                vv = current_views(robot)
                if vv["front"][0] is not None:
                    live.append((k, vv["front"][0], vv["front"][1]))
            if live:
                ref_rgb, ref_dep = live[-1][1], live[-1][2]
                allf = [(-1, e_rgb, e_dep)] + live[:-1]
                hl2 = detect_highlights(robot, sc, cubes, ref_rgb, ref_dep, allf)
                if hl2 is not None and (hl is None or np.max(hl2["peak"]) > np.max(hl["peak"])):
                    hl = hl2
        except EpisodeOver:
            return
        except Exception as e:
            U.log("live watch error", e)
    if hl is None:
        peaks = np.zeros(len(cubes))
        first = [0] * len(cubes)
    else:
        peaks, first = hl["peak"], hl["first"]
    sel, why = PC.select_targets(peaks, ip["n_hint"], ip["singular"] and not ip["which"])
    sel = sorted(sel, key=lambda i: (first[i], -peaks[i]))
    if ip["which"] and len(sel) > 1:
        sel = [sel[-1]] if ip["which"] == "last" else [sel[0]]
        why += "; instruction asks for the %s highlighted" % ip["which"]
    U.log("targets:", sel, why, "order by first highlight")
    save_debug_png("exec_front.png", e_rgb, cubes, sel)
    if frames_rgbd and hl is not None:
        try:
            S = hl["S"]
            fi = int(np.argmax(S[sel[0]])) if sel else 0
            save_debug_png("video_peak_frame.png", frames_rgbd[min(fi, len(frames_rgbd) - 1)][1], cubes, sel)
        except Exception:
            pass
    # action-frame probe (about 16 steps, near the start pose, gripper open)
    try:
        robot.probe(-1)
    except EpisodeOver:
        raise
    except Exception as e:
        U.log("probe error", e, traceback.format_exc()[-500:])
    # manipulation
    done = []
    last_cube = None
    for n_t, ci in enumerate(sel):
        if robot.ended or robot.complete or robot.elapsed() > WALL_DEADLINE:
            break
        cube = cubes[ci]
        others = [c for j, c in enumerate(cubes) if j != ci]
        for attempt in range(3):
            if attempt > 0:
                cube = reacquire(robot, sc, cube, attempt)
            try:
                res, cube_used = pick_one(robot, sc, ci, cube, others)
            except EpisodeOver:
                raise
            except Exception as e:
                U.log("pick error", e, traceback.format_exc()[-800:])
                res, cube_used = "failed", cube
                try:
                    p = robot.pos()
                    robot.gripper(-1, 4)
                    robot.move_to([p[0], p[1], sc.z_table + HOVER_H], None, -1, tol=0.01, label="recover")
                except EpisodeOver:
                    raise
                except Exception:
                    pass
            U.log("target %d attempt %d -> %s (step %s, complete %s)" % (ci, attempt, res, robot.step_id,
                                                                          robot.complete))
            if res == "lifted":
                done.append(ci)
                last_cube = cube_used
                cubes[ci] = cube_used
                break
            if robot.complete or robot.ended:
                break
            # retry: nudge estimate with a fresh wrist look on next attempt (pick_one re-refines)
        if robot.complete:
            break
    U.log("picked:", done, "complete:", robot.complete, "step", robot.step_id)
    # S3b: borderline cubes (scored but below threshold) in score order, max 2
    if not robot.complete and not robot.ended and len(sel) < len(cubes):
        rest = [i for i in np.argsort(-np.asarray(peaks)) if i not in sel and peaks[i] > 0.03][:2]
        if rest:
            U.log("not complete -> trying borderline cubes", rest)
        for ci in rest:
            if robot.complete or robot.ended or robot.elapsed() > WALL_DEADLINE:
                break
            try:
                res, cu = pick_one(robot, sc, int(ci), cubes[ci], [c for j, c in enumerate(cubes) if j != ci])
                U.log("borderline %d -> %s complete %s" % (ci, res, robot.complete))
                if res == "lifted":
                    last_cube = cu
            except EpisodeOver:
                raise
            except Exception as e:
                U.log("borderline pick error", e)
    # S4: final fallback - hold a target lifted until the episode ends
    if not robot.complete and not robot.ended and last_cube is not None and robot.elapsed() < WALL_DEADLINE:
        U.log("not complete after all targets -> re-pick last target and hold")
        others = [c for c in cubes if c is not last_cube]
        try:
            pick_one(robot, sc, -1, last_cube, others, place=False, hold_forever=True)
        except EpisodeOver:
            raise
        except Exception as e:
            U.log("final hold error", e)


def main(ctx):
    try:
        run(ctx)
    except EpisodeOver:
        U.log("episode over")
    except Exception as e:
        U.log("UNCAUGHT", e, traceback.format_exc())
    U.log("controller end")
