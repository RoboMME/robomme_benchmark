"""VideoUnmaskSwap controller (blind, interaction-only R1: written without any live exploration).

Pipeline (transition table):
  S0 observe      -> instruction, calibration, conditioning-video manifest   (fail -> S9 hold)
  S1 calibrate    -> front depth + projection variant validated by pixel_to_position (fail -> S9)
  S2 video        -> track raised objects through the video; the container that covers the
                     last-seen target-colour cube is the carrier; follow it to the live frame
  S3 plan         -> ranked candidate containers per target colour (fail -> S9)
  S4 approach     -> hover above candidate, yaw aligned to its narrow side
  S5 refine       -> wrist RGB-D re-measure of centre / yaw / top height (fail -> keep front estimate)
  S6 descend+grasp-> close; finger width decides grasp (miss -> rise, other yaw, retry <=3)
  S7 lift+verify  -> lift, hold; if the episode does not end, lift higher; if the target colour
                     is visibly absent under the lifted container, put it back and try next candidate
  S8 next target  -> for multi-target instructions put the container back and continue
  S9 hold         -> hold the current gripper state until the episode ends or wall budget is used
"""
import math
import re
import time

import numpy as np

import vus_geom as vg
import vus_percept as vp
import vus_video as vv
from vus_io import (VERSION, elapsed, frame_files, load_frame_images, load_manifest, log, log_exc,
                    save_json)
from vus_robot import EpisodeEnded, Robot

WALL_STOP_S = 950.0          # return well inside the 1200 s controller limit
VIDEO_BUDGET_S = 330.0       # video processing must finish well before the 600 s first-step limit
HOVER_ABOVE = 0.11           # TCP hover height above the container top
TRAVEL_CLEAR = 0.09          # travel height above the tallest object
PRE_GRASP_ABOVE = 0.025
LIFT_ABOVE = 0.15
GRIP_OPEN, GRIP_CLOSE = -1.0, 1.0
MAX_OPEN_W = 0.078
MISS_WIDTH = 0.004


def parse_colors(instr):
    text = (instr or "").lower()
    found = []
    for m in re.finditer(r"[a-z]+", text):
        w = m.group(0)
        w = vp.COLOR_SYNONYMS.get(w, w)
        if w in vp.ALL_COLORS and w not in found:
            found.append(w)
    return found


class Scene:
    """Live front-camera perception (calibrated)."""

    def __init__(self, robot):
        self.robot = robot
        self.calib = None
        self.variant = vg.DEFAULT_VARIANT
        self.tz = None
        self.bounds = None

    def front(self, fresh=True):
        r = self.robot
        if fresh:
            r.observe()
        res = r.validated_depth("front")
        if res is None:
            return None
        depth, rgb, calib, (var, sw), err = res
        if err is not None and err > 0.02:
            log("front projection error large", err=err)
        P, valid = vg.backproject(depth, calib, var)
        if self.tz is None:
            self.tz = vp.table_height(P, valid)
            self.bounds = vp.table_bounds(P, valid, self.tz) if self.tz is not None else None
            log("table", tz=self.tz, bounds=self.bounds)
        self.calib, self.variant = calib, var
        comps = vp.segment(P, valid, rgb, self.tz, ee_pos=r.pos(), bounds=self.bounds)
        return {"P": P, "valid": valid, "rgb": rgb, "comps": comps, "depth": depth}


def comp_brief(c):
    return {"xy": np.round(c["top_xy"], 4).tolist(), "h": round(c["height"], 4), "ext": round(c["ext"], 4),
            "n": c["n"], "robot": c["robot"], "color": c.get("color"),
            "dom": round(c.get("color_dom_frac", 0.0), 2),
            "fp_top": [round(x, 4) for x in c["fp_top"]]}


# ============================================================================== video

def analyse_video(robot, scene, manifest, colors, live):
    frames = sorted(manifest.get("frames", []), key=lambda f: f.get("frame_index", 0))
    video = [f for f in frames if f.get("phase") == "video_demo"]
    log("video frames", total=len(frames), video=len(video),
        sampling=manifest.get("video_sampling"), hz=manifest.get("frequency_hz"))
    tracker = vv.VideoTracker(colors if colors else [None])
    if not video:
        log("no conditioning video frames")
    calib, var, tz, bounds = scene.calib, scene.variant, scene.tz, scene.bounds
    t_start = time.time()
    stride = 1
    per_frame = []
    last_depth = None
    i = 0
    while i < len(video):
        fr = video[i]
        try:
            imgs = load_frame_images(fr, ("front",)).get("front", {})
            if "depth" not in imgs:
                i += stride
                continue
            P, valid = vg.backproject(imgs["depth"], calib, var)
            st = fr.get("state")
            ee = np.array(st[:3], dtype=float) if isinstance(st, list) and len(st) >= 3 else None
            comps = vp.segment(P, valid, imgs.get("rgb"), tz, ee_pos=ee, bounds=bounds)
            t = float(fr.get("timestamp_s", i * 0.05))
            tracker.step(t, comps, ee)
            last_depth = imgs["depth"]
            nd = [c for c in comps if not c["robot"]]
            per_frame.append((round(t, 2), len(nd), sum(1 for c in comps if c["robot"]),
                              [ts.carrier for ts in tracker.targets],
                              [ts.seen for ts in tracker.targets]))
        except Exception:
            log_exc(f"video frame {i}")
        # keep inside the wall budget: widen the stride if needed
        spent = time.time() - t_start
        done = i + 1
        if done >= 10 and spent / done * (len(video) - done) / stride > VIDEO_BUDGET_S - spent:
            stride = min(8, stride * 2)
            log("video: increasing stride", stride=stride, spent=round(spent, 1))
        if spent > VIDEO_BUDGET_S:
            log("video: budget exhausted, skipping to the end", frame=i)
            break
        i += stride
    # static camera check
    if last_depth is not None and live is not None:
        try:
            a, b = last_depth, live["depth"]
            m = (a > 0) & (b > 0)
            diff = float(np.median(np.abs(a[m] - b[m]))) if m.any() else None
            log("video vs live front depth median |diff| (m)", diff=diff)
        except Exception:
            log_exc("static check")
    # the live frame closes the sequence
    t_live = (per_frame[-1][0] + 0.05) if per_frame else 0.0
    tracker.step(t_live, live["comps"], robot.pos())
    log("video processed", frames=len(per_frame), seconds=round(time.time() - t_start, 1),
        tracks=len(tracker.tracks))
    save_json("video_per_frame.json", per_frame)
    save_json("video_tracks.json", [dict(tr.summary(), hist=[(round(h[0], 2), h[1].round(3).tolist(), h[2])
                                                             for h in tr.hist[::3]]) for tr in tracker.tracks])
    res = tracker.result()
    log("video result", result=res)
    return tracker


# ============================================================================== grasp

def choose_yaw(robot, det, others):
    cx, cy, w, l, yaw = det["fp_top"]
    opts = [vg.wrap_pi(yaw)]
    if l - w < 0.012:
        opts.append(vg.wrap_pi(yaw + math.pi / 2))
    cur = robot.finger_yaw(robot.R())
    scored = []
    for y in opts:
        # finger direction is symmetric: choose the representative closest to current yaw
        d = vg.wrap_pi(y - cur)
        if d > math.pi / 2:
            y -= math.pi
        elif d < -math.pi / 2:
            y += math.pi
        a = np.array([math.cos(y), math.sin(y)])
        clear = 1.0
        for o in others:
            oc = np.array(o["top_xy"])
            for s in (-1, 1):
                fpos = np.array([cx, cy]) + s * a * 0.05
                clear = min(clear, float(np.linalg.norm(fpos - oc) - 0.5 * o["ext"]))
        scored.append((y, clear, abs(vg.wrap_pi(y - cur))))
    good = [s for s in scored if s[1] > 0.012]
    pool = good if good else sorted(scored, key=lambda s: -s[1])[:1]
    pool.sort(key=lambda s: s[2])
    order = [pool[0][0]] + [s[0] for s in scored if s[0] != pool[0][0]]
    # escalation option: rotate 90 deg (in case the finger-axis assumption is wrong)
    alt = vg.wrap_pi(order[0] + math.pi / 2)
    if all(abs(vg.wrap_pi(alt - o)) > 0.2 for o in order):
        order.append(alt)
    return order, scored


def near_ref_yaw(robot, y):
    """Finger yaw is symmetric under pi: pick the representative within +-pi/2 of the start yaw."""
    ref = robot.finger_yaw(robot.R0) if robot.R0 is not None else robot.finger_yaw(robot.R())
    d = vg.wrap_pi(y - ref)
    if d > math.pi / 2:
        d -= math.pi
    elif d < -math.pi / 2:
        d += math.pi
    return ref + d


def measure_wrist(robot, scene, prior):
    """Fresh wrist RGB-D measurement of the container nearest `prior` (cx, cy, zt).
    Returns (dict(cx, cy, w, l, yaw, ztop, n) or None, reason)."""
    try:
        robot.observe()
        res = robot.validated_depth("wrist")
        if res is None:
            return None, "no_wrist_depth"
        depth, rgb, calib, (var, sw), err = res
        if err is None or err > 0.015:
            return None, f"wrist_err_{err}"
        P, valid = vg.backproject(depth, calib, var)
        z = P[..., 2]
        tz = scene.tz
        d = np.hypot(P[..., 0] - prior["cx"], P[..., 1] - prior["cy"])
        m = valid & (z < prior["zt"] + 0.03) & (d < 0.12)
        comps = vp.segment(P, m, None, tz, ee_pos=None, bounds=None)
        comps = [c for c in comps if c["height"] > 0.5 * (prior["zt"] - tz) and c["top_n"] >= 30]
        if not comps:
            return None, "no_component"
        c = min(comps, key=lambda c: math.hypot(c["top_xy"][0] - prior["cx"], c["top_xy"][1] - prior["cy"]))
        cx, cy, w, l, yaw = c["fp_top"]
        meas = {"cx": cx, "cy": cy, "w": w, "l": l, "yaw": yaw, "ztop": c["zmax"], "n": c["top_n"],
                "shift": math.hypot(cx - prior["cx"], cy - prior["cy"])}
        log("wrist measure", **{k: round(float(v), 4) for k, v in meas.items()})
        return meas, "ok"
    except EpisodeEnded:
        raise
    except Exception:
        log_exc("measure_wrist")
        return None, "exception"


def _consistent(a, b, pos_tol, size_tol):
    return (math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]) < pos_tol and abs(a["w"] - b["w"]) < size_tol
            and abs(a["l"] - b["l"]) < size_tol)


def wrist_refine(robot, scene, est, yaw_cmd=None):
    """Refine est (front estimate) with one or two wrist looks. Returns (est', reason)."""
    m1, why = measure_wrist(robot, scene, est)
    if m1 is None:
        return est, why
    if m1["shift"] > 0.05 or abs(m1["ztop"] - est["zt"]) > 0.025 or not (0.015 < m1["w"] < 0.15):
        return est, "rejected_far"
    size_ok = abs(m1["w"] - est["w"]) < 0.012 and abs(m1["l"] - est["l"]) < 0.015
    best = m1
    if m1["shift"] > 0.008 or not size_ok:
        # second look centred on the first wrist estimate (less occlusion, different viewpoint)
        p = robot.pos()
        robot.move_to([m1["cx"], m1["cy"], p[2]], yaw=yaw_cmd, grip=GRIP_OPEN, max_steps=30, vmax=0.012,
                      tol=0.003, label="recentre")
        m2, why2 = measure_wrist(robot, scene, {"cx": m1["cx"], "cy": m1["cy"], "zt": est["zt"]})
        if m2 is not None and _consistent(m1, m2, 0.008, 0.01):
            best = m2
            size_ok = True  # two consistent wrist views
        elif m2 is not None and abs(m2["w"] - est["w"]) < 0.012 and abs(m2["l"] - est["l"]) < 0.015 \
                and m2["shift"] < 0.05:
            best = m2
            size_ok = True
        elif not size_ok:
            return est, "rejected_inconsistent"
    new = dict(est)
    new.update(cx=best["cx"], cy=best["cy"], zt=0.5 * (best["ztop"] + est["zt"]), w=best["w"], l=best["l"],
               yaw=best["yaw"])
    return new, "ok"


def grasp_lift(robot, scene, det, others, max_attempts=3):
    """Pick up the container `det`. Returns dict(ok, why, est)."""
    tz = scene.tz
    cx, cy, w, l, yaw = det["fp_top"]
    if math.hypot(cx - det["top_xy"][0], cy - det["top_xy"][1]) > 0.03:
        cx, cy = det["top_xy"]
    est = {"cx": cx, "cy": cy, "zt": det["zmax"], "yaw": yaw, "w": w, "l": l}
    est_front = dict(est)
    tallest = max([o["zmax"] for o in others] + [det["zmax"]])
    z_travel = tallest + TRAVEL_CLEAR
    yaws, scored = choose_yaw(robot, det, others)
    log("grasp plan", est=est, yaw_options=[round(y, 3) for y in yaws], scored=scored, z_travel=z_travel,
        width_ok=bool(w < MAX_OPEN_W))
    attempt = 0
    refined = False
    while attempt < max_attempts:
        yaw_cmd = near_ref_yaw(robot, yaws[min(attempt, len(yaws) - 1)])
        attempt += 1
        log(f"attempt {attempt}", yaw=round(yaw_cmd, 3), step=robot.step_id)
        p = robot.pos()
        # rise first if low, then travel
        if p[2] < z_travel - 0.01:
            robot.move_to([p[0], p[1], z_travel], yaw=None, grip=GRIP_OPEN, max_steps=40, label="rise")
        hover = [est["cx"], est["cy"], max(est["zt"] + HOVER_ABOVE, z_travel)]
        r = robot.move_to(hover, yaw=yaw_cmd, grip=GRIP_OPEN, max_steps=90, vmax=0.018, label="travel")
        log("at hover", res=r, pos=robot.pos().round(4).tolist(), width=round(robot.width(), 4))
        est2, why = wrist_refine(robot, scene, est_front, yaw_cmd)
        log("refine result", why=why)
        if why == "ok":
            est = est2
            refined = True
            if attempt == 1:
                # re-derive yaw from the refined footprint, keeping the representative near the current one
                fake = {"fp_top": (est["cx"], est["cy"], est["w"], est["l"], est["yaw"])}
                yaws_new, _ = choose_yaw(robot, fake, others)
                yaws = yaws_new
                yaw_cmd = near_ref_yaw(robot, yaws[0])
        r = robot.move_to([est["cx"], est["cy"], est["zt"] + PRE_GRASP_ABOVE + 0.03], yaw=yaw_cmd,
                          grip=GRIP_OPEN, max_steps=50, vmax=0.012, tol=0.003, label="align")
        r = robot.move_to([est["cx"], est["cy"], est["zt"] + PRE_GRASP_ABOVE], yaw=yaw_cmd, grip=GRIP_OPEN,
                          max_steps=30, vmax=0.008, tol=0.003, label="pregrasp")
        h = est["zt"] - tz
        g = min(0.03, max(0.012, 0.45 * h))
        z_grasp = max(est["zt"] - g, tz + 0.012)
        r = robot.move_to([est["cx"], est["cy"], z_grasp], yaw=yaw_cmd, grip=GRIP_OPEN, max_steps=40,
                          vmax=0.006, tol=0.003, label="descend", stall_calls=2)
        log("descend", res=r, pos=robot.pos().round(4).tolist(), z_grasp=round(z_grasp, 4))
        if not r["ok"] and robot.pos()[2] > z_grasp + 0.012:
            log("descent blocked; rising for another attempt (reverting to the front estimate)")
            est = dict(est_front)
            robot.move_to([est["cx"], est["cy"], est["zt"] + HOVER_ABOVE], yaw=None, grip=GRIP_OPEN,
                          max_steps=30, label="rise_blocked")
            continue
        wdt = robot.gripper(GRIP_CLOSE, n=14)
        log("closed", width=round(wdt, 4), step=robot.step_id)
        if wdt < MISS_WIDTH:
            log("grasp missed (fingers closed fully)")
            robot.gripper(GRIP_OPEN, n=8)
            robot.move_to([est["cx"], est["cy"], est["zt"] + HOVER_ABOVE], yaw=None, grip=GRIP_OPEN,
                          max_steps=30, label="rise_miss")
            # re-measure from the front camera (the container may have moved)
            try:
                live = scene.front()
                if live:
                    cand = [c for c in live["comps"] if not c["robot"] and c["height"] > 0.015]
                    if cand:
                        near = min(cand, key=lambda c: math.hypot(c["top_xy"][0] - est["cx"], c["top_xy"][1] - est["cy"]))
                        dd = math.hypot(near["top_xy"][0] - est["cx"], near["top_xy"][1] - est["cy"])
                        log("re-measure after miss", near=comp_brief(near), dist=round(dd, 4))
                        if dd < 0.06:
                            est.update(cx=near["fp_top"][0], cy=near["fp_top"][1], zt=near["zmax"])
                            est_front.update(cx=near["fp_top"][0], cy=near["fp_top"][1], zt=near["zmax"])
            except EpisodeEnded:
                raise
            except Exception:
                log_exc("re-measure")
            continue
        # lift
        r = robot.move_to([est["cx"], est["cy"], est["zt"] + LIFT_ABOVE], yaw=None, grip=GRIP_CLOSE,
                          max_steps=60, vmax=0.010, tol=0.006, label="lift")
        wdt = robot.width()
        log("lifted", res=r, pos=robot.pos().round(4).tolist(), width=round(wdt, 4), step=robot.step_id)
        if wdt < MISS_WIDTH:
            log("lost the container during lift")
            robot.gripper(GRIP_OPEN, n=6)
            continue
        return {"ok": True, "why": "lifted", "est": est, "z_grasp": z_grasp, "yaw": yaw_cmd, "refined": refined}
    return {"ok": False, "why": "attempts_exhausted", "est": est}


def verify_under(robot, scene, est, color):
    """After lifting: which cube colours are visible where the container stood?
    Returns dict with per-colour raised-pixel counts within 4.5 cm of the old centre."""
    try:
        live = scene.front()
        if live is None or live["rgb"] is None:
            return None
        cols = vp.color_labels(live["rgb"])
        P, valid = live["P"], live["valid"]
        z = P[..., 2]
        d = np.hypot(P[..., 0] - est["cx"], P[..., 1] - est["cy"])
        m = valid & (z > scene.tz + 0.003) & (z < scene.tz + 0.07) & (d < 0.045)
        counts = {}
        for k, name in enumerate(vp.ALL_COLORS):
            if name in vp.COLOR_HUES or name == "brown":
                n = int((m & (cols == k)).sum())
                if n:
                    counts[name] = n
        res = {"counts": counts, "target": color,
               "target_px": counts.get(color, 0) if color else None}
        others = {k: v for k, v in counts.items() if k != color}
        res["other_px"] = max(others.values()) if others else 0
        log("verify under lifted container", **res)
        return res
    except EpisodeEnded:
        raise
    except Exception:
        log_exc("verify_under")
        return None


def put_back(robot, est, z_grasp):
    robot.move_to([est["cx"], est["cy"], z_grasp + 0.006], yaw=None, grip=GRIP_CLOSE, max_steps=60,
                  vmax=0.010, label="put_back")
    robot.gripper(GRIP_OPEN, n=10)
    robot.move_to([est["cx"], est["cy"], est["zt"] + HOVER_ABOVE], yaw=None, grip=GRIP_OPEN, max_steps=40,
                  label="leave")


def hold_until_end(robot, grip, max_wall=WALL_STOP_S):
    log("holding", grip=grip, step=robot.step_id, remaining=robot.remaining())
    while not robot.ended and elapsed() < max_wall and robot.remaining() > 0:
        robot.hold(20, grip)


# ============================================================================== main

def run(robot):
    meta = robot.observe()
    instr = ""
    try:
        from vus_io import get_instruction
        instr = get_instruction(meta)
    except Exception:
        log_exc("instruction")
    st = robot.status()
    if not instr:
        try:
            from vus_io import get_instruction
            instr = get_instruction(st)
        except Exception:
            log_exc("instruction from status")
    log("start", instruction=instr, state=None if robot.state is None else robot.state.round(4).tolist(),
        step=robot.step_id, max_steps=robot.max_steps, limits=robot.last_limits,
        meta_keys=sorted(meta.keys()) if isinstance(meta, dict) else None,
        n_images=len(robot.images), status_keys=sorted(st.keys()) if isinstance(st, dict) else None)
    try:
        skel = {k: (v if not isinstance(v, (list, dict)) or len(str(v)) < 3000 else str(v)[:3000])
                for k, v in meta.items()}
        save_json("obs0_meta.json", skel)
        save_json("status0.json", st)
    except Exception:
        log_exc("save meta")
    colors = parse_colors(instr)
    chromatic = [c for c in colors if c in vp.COLOR_HUES or c == "brown"]
    targets = chromatic if chromatic else colors
    log("target colours", colors=colors, targets=targets)
    if robot.state is None:
        log("no robot state; holding")
        return "no_state"
    robot.set_reference()

    scene = Scene(robot)
    live = scene.front(fresh=False)
    if live is None or scene.tz is None:
        log("front perception unavailable")
        return "no_front"
    nonrobot = [c for c in live["comps"] if not c["robot"]]
    log("live components", comps=[comp_brief(c) for c in live["comps"]])

    manifest, mpath = load_manifest(meta)
    log("manifest", path=mpath, ok=manifest is not None)
    tracker = None
    if manifest is not None:
        try:
            tracker = analyse_video(robot, scene, manifest, targets, live)
        except Exception:
            log_exc("analyse_video")
    if elapsed() > 560:
        log("WARNING: close to the first-step limit")

    # candidate list per target
    plans = []
    tlist = targets if targets else [None]
    for k, col in enumerate(tlist):
        cands = []
        if tracker is not None:
            try:
                for tr, why in vv.ranked_candidates(tracker, k, live["comps"]):
                    if tr.det is None or tr.det["robot"]:
                        continue
                    # grasp geometry from the unsplit live component nearest to the track
                    near = min(nonrobot, key=lambda c: math.hypot(c["top_xy"][0] - tr.pos[0],
                                                                  c["top_xy"][1] - tr.pos[1]), default=None)
                    if near is None:
                        continue
                    dd = math.hypot(near["top_xy"][0] - tr.pos[0], near["top_xy"][1] - tr.pos[1])
                    if dd > 0.06:
                        log("candidate track far from any live component", track=tr.id, dist=round(dd, 3))
                        continue
                    if any(near is c for c, _, _ in cands):
                        continue
                    exts = [c["fp_top"][3] for c in nonrobot if c["height"] > 0.02]
                    lmed = float(np.median(exts)) if exts else near["fp_top"][3]
                    if tr.det.get("split") and near["fp_top"][3] > 1.4 * lmed and len(exts) >= 2:
                        # touching containers merged in the live frame: use the split centre
                        ws = [c["fp_top"][2] for c in nonrobot if c["height"] > 0.02]
                        syn = dict(near)
                        syn["top_xy"] = np.array(tr.pos, dtype=float)
                        syn["fp_top"] = (float(tr.pos[0]), float(tr.pos[1]), float(np.median(ws)), lmed,
                                         near["fp_top"][4])
                        log("candidate merged with a neighbour; using split centre", track=tr.id)
                        near = syn
                    cands.append((near, why, tr.id))
            except Exception:
                log_exc("ranked_candidates")
        # if the target colour never appeared as a cube, a container of that colour may be meant
        try:
            seen = tracker.targets[k].seen if tracker is not None and k < len(tracker.targets) else 0
        except Exception:
            seen = 0
        if col is not None and seen == 0:
            coln = [c for c in nonrobot if c.get("color") == col and c.get("color_dom_frac", 0) > 0.3
                    and c["height"] > 0.02]
            coln.sort(key=lambda c: -c.get("color_dom_frac", 0))
            for c in coln:
                if all(c is not d for d, _, _ in cands):
                    cands.insert(0, (c, "container_of_named_colour", None))
            if coln:
                log("target colour never seen as a cube; using colour-named container", n=len(coln))
        # fallback: containers in the live frame, tallest first
        conts = [c for c in nonrobot if c["height"] > 0.02 and not vp.is_cube_like(c) or c["height"] > 0.045]
        for c in sorted(conts, key=lambda c: -c["height"]):
            if all(c is not d for d, _, _ in cands):
                cands.append((c, "fallback_live", None))
        plans.append((col, cands))
        log("candidates", color=col, cands=[(why, tid, comp_brief(d)) for d, why, tid in cands[:5]])

    grip_state = GRIP_OPEN
    for idx, (col, cands) in enumerate(plans):
        if not cands:
            log("no candidates", color=col)
            continue
        tried = 0
        for det, why, tid in cands:
            if tried >= 3 or elapsed() > WALL_STOP_S - 120:
                break
            tried += 1
            # use fresh live geometry for this candidate if available
            others = [c for c in nonrobot if c is not det and c["height"] > 0.015]
            log("pick candidate", color=col, why=why, track=tid, det=comp_brief(det))
            res = grasp_lift(robot, scene, det, others)
            log("grasp result", ok=res["ok"], why=res["why"])
            if not res["ok"]:
                grip_state = GRIP_OPEN
                break
            grip_state = GRIP_CLOSE
            robot.hold(10, GRIP_CLOSE)
            ver = verify_under(robot, scene, res["est"], col)
            robot.hold(15, GRIP_CLOSE)
            # still running: lift higher once
            p = robot.pos()
            robot.move_to([p[0], p[1], p[2] + 0.08], yaw=None, grip=GRIP_CLOSE, max_steps=25, vmax=0.01,
                          label="lift_higher")
            robot.hold(15, GRIP_CLOSE)
            last = idx == len(plans) - 1
            # positive evidence of a wrong pick: another colour's cube visible, the target's not
            wrong = bool(ver is not None and col is not None and (ver.get("target_px") or 0) < 8
                         and ver.get("other_px", 0) >= 15)
            if last and not wrong:
                break
            if last and wrong and tried < 3 and len(cands) > tried:
                log("target colour not seen under the lifted container; trying the next candidate")
                put_back(robot, res["est"], res["z_grasp"])
                grip_state = GRIP_OPEN
                live = scene.front() or live
                nonrobot = [c for c in live["comps"] if not c["robot"]]
                continue
            if not last:
                log("multi-target: putting the container back before the next target")
                put_back(robot, res["est"], res["z_grasp"])
                grip_state = GRIP_OPEN
                live = scene.front() or live
                nonrobot = [c for c in live["comps"] if not c["robot"]]
                # refresh the next target's candidate geometry from the live frame by position
                if idx + 1 < len(plans):
                    ncol, ncands = plans[idx + 1]
                    upd = []
                    for d, w_, t_ in ncands:
                        near = min(nonrobot, key=lambda c: math.hypot(c["top_xy"][0] - d["top_xy"][0],
                                                                       c["top_xy"][1] - d["top_xy"][1]),
                                   default=None)
                        if near is not None and math.hypot(near["top_xy"][0] - d["top_xy"][0],
                                                           near["top_xy"][1] - d["top_xy"][1]) < 0.05:
                            upd.append((near, w_, t_))
                    plans[idx + 1] = (ncol, upd)
            break
    hold_until_end(robot, grip_state)
    return "done"


def main(ctx):
    log("controller start", version=VERSION)
    robot = Robot(ctx)
    outcome = "unknown"
    try:
        outcome = run(robot)
        if not robot.ended:
            hold_until_end(robot, GRIP_CLOSE if robot.state is not None and robot.width() > MISS_WIDTH
                           and robot.width() < 0.075 else GRIP_OPEN)
    except EpisodeEnded:
        outcome = "episode_ended"
    except Exception:
        log_exc("run")
        outcome = "exception"
        try:
            if not robot.ended:
                hold_until_end(robot, GRIP_CLOSE if robot.state is not None and MISS_WIDTH < robot.width() < 0.075
                               else GRIP_OPEN, max_wall=WALL_STOP_S)
        except EpisodeEnded:
            pass
        except Exception:
            log_exc("final hold")
    log("controller end", outcome=outcome, ended=robot.ended, task_complete=robot.complete,
        step=robot.step_id, steps_sent=robot.steps_sent, calls=robot.n_calls)
    save_json("summary.json", {"version": VERSION, "outcome": outcome, "ended": robot.ended,
                               "task_complete": robot.complete, "step": robot.step_id,
                               "steps_sent": robot.steps_sent, "elapsed_s": elapsed()})
