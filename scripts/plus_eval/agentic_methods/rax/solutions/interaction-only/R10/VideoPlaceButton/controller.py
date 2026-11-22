"""VideoPlaceButton controller v1.
Reads the conditioning video's per-frame robot states to find pick/place/press events, resolves the
instruction (ordinal + before/after the button press), localizes the cube and target in the current
images, then picks and places the cube with a low (contact) release."""
import json, math, os, sys, time, traceback
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
from tracker import frame_detections, phase_swaps, resolve_identity_slot
from video import load_video_states, extract_events, parse_instruction, resolve_target, assign_tracks, resolve_for_track

SEMANTICS = "naive"
PRESS_FIRST = False
TRACK_IDENTITY = True
OUT = Path("/workspace/output"); OUT.mkdir(parents=True, exist_ok=True)
LOGF = open(OUT / "controller_log.txt", "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOGF.write(s + "\n"); LOGF.flush()


def detect_targets(meta, imgs):
    rgb = imgs[("front", "rgb")]; P, _ = world_points(meta, imgs, "front")
    h = hsv(rgb); H, S, V = h[..., 0], h[..., 1], h[..., 2]
    low = P[..., 2] < 0.013
    colored = ((H >= 180) | (H <= 4)) & (S > 50) & (S <= 185) & (V > 140)
    white = (S < 50) & (V > 200)
    ring = (colored | white) & low
    pts = []
    for ys, xs in components(ring, 3):
        p = np.median(P[ys, xs], 0)
        if p[2] < 0.013:
            pts.append((p[:2], len(ys)))
    clusters = []
    for p, n in pts:
        for c in clusters:
            if np.linalg.norm(c["sum"] / c["n"] - p) < 0.03:
                c["sum"] += p * n; c["n"] += n; break
        else:
            clusters.append(dict(sum=p * n, n=n))
    return [c["sum"] / c["n"] for c in clusters if c["n"] >= 15]


def detect_cubes(meta, imgs, color, cam="front"):
    rgb = imgs[(cam, "rgb")]; P, _ = world_points(meta, imgs, cam)
    out = []
    for ys, xs in components(color_mask(rgb, color), 15):
        pts = P[ys, xs]
        zmax = np.percentile(pts[:, 2], 95)
        if zmax < 0.015:
            continue
        top = pts[pts[:, 2] > zmax - 0.006]
        out.append(dict(xy=np.median(top[:, :2], 0), ztop=float(zmax), n=len(ys), top=top))
    return out


def square_yaw(top_pts):
    """Yaw (mod pi/2) of a square from its top-face world points, by min-area bounding box."""
    xy = top_pts[:, :2] - top_pts[:, :2].mean(0)
    best = None
    for deg in np.arange(0, 90, 1.5):
        a = math.radians(deg); c, s = math.cos(a), math.sin(a)
        r = xy @ np.array([[c, -s], [s, c]])
        area = np.ptp(r[:, 0]) * np.ptp(r[:, 1])
        if best is None or area < best[0]:
            best = (area, a)
    return best[1]


def pick_cube(rb, color, last_place, ref=None):
    grasp_z = 0.022
    for attempt in range(3):
        if rb.ended:
            break
        cubes = detect_cubes(rb.meta, rb.imgs, color)
        if ref is None:
            ref = last_place["xy"] if last_place is not None else np.array([-0.1, 0.0])
        log("cubes", [(np.round(c["xy"], 3), c["n"]) for c in cubes])
        if not cubes:
            cube_xy = ref
        else:
            cube_xy = min(cubes, key=lambda c: np.linalg.norm(c["xy"] - ref))["xy"]
        cy = yaw_of(rb.state[3:6])
        if last_place is not None and np.linalg.norm(cube_xy - last_place["xy"]) < 0.02 and attempt == 0:
            gyaw = yaw_of(last_place["yaw_aa"])
        else:
            gyaw = cy
        gyaw = cy + wrap(gyaw - cy, math.pi / 2)
        rb.grip = -1.0
        rb.move_to([cube_xy[0], cube_xy[1], 0.12], yaw=gyaw)
        # refine with wrist view
        wc = detect_cubes(rb.meta, rb.imgs, color, cam="wrist")
        ztop = 0.041
        if wc:
            c = min(wc, key=lambda c: np.linalg.norm(c["xy"] - cube_xy))
            if np.linalg.norm(c["xy"] - cube_xy) < 0.05:
                sy = square_yaw(c["top"]) if len(c["top"]) > 30 else None
                log("wrist cube", np.round(c["xy"], 3), "ztop", round(c["ztop"], 3), "sq_yaw", sy, "gyaw", round(gyaw, 3))
                cube_xy = c["xy"]; ztop = c["ztop"]
                if sy is not None:
                    cur = yaw_of(rb.state[3:6])
                    cand = cur + wrap(sy - cur, math.pi / 2)
                    if abs(wrap(cand - gyaw, math.pi / 2)) > math.radians(12) or attempt > 0:
                        log("using wrist yaw", round(cand, 3))
                        gyaw = cand
        gz = max(0.012, ztop - 0.02)
        rb.move_to([cube_xy[0], cube_xy[1], 0.06], yaw=gyaw)
        rb.move_to([cube_xy[0], cube_xy[1], gz], yaw=gyaw, vmax=0.4, max_iter=6)
        grasp_z = rb.state[2]
        rb.gripper(1.0, 10)
        rb.move_to([cube_xy[0], cube_xy[1], 0.15], yaw=gyaw)
        w = rb.state[6] + rb.state[7]
        log("after lift width", round(w, 4), "step", rb.meta.get("step_id"), "grasp_z", round(grasp_z, 4))
        if 0.01 < w < 0.075:
            break
        log("grasp failed; reopening")
        rb.gripper(-1.0, 8)
        rb.observe()
    else:
        log("grasp failed 3 times")

    return grasp_z


def descend_contact(rb, xy, yaw, grasp_z, floor_margin=0.004, max_steps=40):
    """Descend in small steps until the TCP stops moving (contact) or reaches grasp_z - margin."""
    rb.move_to([xy[0], xy[1], grasp_z + 0.012], yaw=yaw, vmax=0.4, max_iter=6, tol=0.003)
    zs = [rb.state[2]]
    n = 0
    while n < max_steps and not rb.ended:
        s = rb.state
        e = np.asarray(xy) - s[:2]
        rb.step([[e[0] / 0.05, e[1] / 0.05, -0.04, 0, 0, 0, rb.grip]] * 2)
        n += 2
        zs.append(rb.state[2])
        if rb.state[2] < grasp_z - floor_margin:
            break
        if len(zs) >= 3 and zs[-3] - zs[-1] < 0.0015:
            break
    rb.step([[0, 0, 0, 0, 0, 0, rb.grip]] * 2)
    log("descend_contact z", [round(z, 4) for z in zs], "grasp_z", round(grasp_z, 4))


def press_button(rb, xy):
    log("press_button at", np.round(xy, 3))
    rb.grip = 1.0
    yaw = yaw_of(rb.state[3:6])
    rb.move_to([xy[0], xy[1], 0.15], yaw=yaw)
    rb.gripper(1.0, 6)
    rb.move_to([xy[0], xy[1], 0.06], yaw=yaw)
    zs = []
    for _ in range(12):
        e = np.asarray(xy) - rb.state[:2]
        rb.step([[e[0] / 0.05, e[1] / 0.05, -0.12, 0, 0, 0, 1.0]] * 2)
        zs.append(round(rb.state[2], 4))
        if rb.state[2] < 0.025 or rb.ended:
            break
    rb.step([[0, 0, 0, 0, 0, 0, 1.0]] * 3)
    log("press z trace", zs, "ended", rb.ended)
    rb.move_to([xy[0], xy[1], 0.15], yaw=yaw)
    rb.grip = -1.0
    rb.gripper(-1.0, 6)


def place_at(rb, target_xy, grasp_z, lift=True):
    yaw = yaw_of(rb.state[3:6])
    rb.move_to([target_xy[0], target_xy[1], 0.15], yaw=yaw)
    rb.move_to([target_xy[0], target_xy[1], 0.06], yaw=yaw)
    descend_contact(rb, target_xy, yaw, grasp_z)
    log("at place", np.round(rb.state[:3], 4), "step", rb.meta.get("step_id"))
    rb.gripper(-1.0, 10)
    if not rb.ended:
        rb.move_to([target_xy[0], target_xy[1], 0.15], max_iter=4)
    if not rb.ended:
        rb.step([[0, 0, 0, 0, 0, 0, -1]] * 5)


def main(ctx):
    t0 = time.time()
    rb = Robot(ctx, log=log)
    meta = rb.observe()
    instr = meta.get("instruction", "")
    log("instruction:", instr)
    seq = meta.get("frame_sequence") or {}
    spec = parse_instruction(instr)
    color = spec["color"] or "red"
    log("spec", spec)
    target_xy = None; last_place = None; press_xy = None
    try:
        S, T, frames, m = load_video_states(seq["manifest_path"])
        ev = extract_events(S, T)
        for e in ev:
            log("event", e["kind"], e["index"], np.round(e["xy"], 3), round(e["z"], 3))
        tracks = assign_tracks(ev)
        cubes0 = detect_cubes(rb.meta, rb.imgs, color)
        track = None
        if cubes0 and tracks:
            best = None
            for c in cubes0:
                for i, t in enumerate(tracks):
                    d = np.linalg.norm(t["pos"] - c["xy"])
                    if best is None or d < best[0]:
                        best = (d, i)
            log("cube->track", best)
            if best[0] < 0.05:
                track = best[1]
        if track is None:
            track = len(tracks) - 1 if len(tracks) == 1 else 0
            log("track fallback", track)
        if SEMANTICS == "global_flip":
            tgt, why = resolve_for_track(ev, spec, None, "flip_global")
        else:
            tgt, why = resolve_for_track(ev, spec, track, SEMANTICS)
        log("resolved", why, None if tgt is None else np.round(tgt["xy"], 3))
        if tgt is not None:
            target_xy = np.array(tgt["xy"])
            if TRACK_IDENTITY:
                try:
                    t1 = time.time()
                    dets = frame_detections(rb.meta, frames)
                    slots = dets[-1]
                    sw = phase_swaps(slots, dets, log=log)
                    log("swaps", sw, "slots", [np.round(s, 3) for s in slots], "t", round(time.time() - t1, 1))
                    s, d = resolve_identity_slot(slots, sw, tgt["index"], np.array(tgt["xy"]))
                    log("identity slot", s, "dist", round(float(d), 3))
                    if s is not None:
                        target_xy = np.array(slots[s])
                        log("tracked target_xy", np.round(target_xy, 3))
                except Exception:
                    log("tracking failed", traceback.format_exc())
        presses = [e for e in ev if e["kind"] == "press"]
        if presses:
            press_xy = np.array(presses[0]["xy"])
        places = [e for e in ev if e["kind"] == "place" and e.get("track") == track]
        if places:
            last_place = places[-1]
    except Exception:
        log("video parse failed", traceback.format_exc())

    targets = detect_targets(rb.meta, rb.imgs)
    log("targets detected", [np.round(t, 3) for t in targets])
    if target_xy is None:
        target_xy = targets[0] if targets else np.array([-0.2, 0.0])
    if targets:
        d = [np.linalg.norm(t - target_xy) for t in targets]
        k = int(np.argmin(d))
        if d[k] < 0.03:
            target_xy = 0.5 * (target_xy + targets[k])
    log("target_xy", np.round(target_xy, 3))

    if PRESS_FIRST and press_xy is not None:
        press_button(rb, press_xy)
        rb.observe()
    grasp_z = pick_cube(rb, color, last_place)
    if not rb.ended:
        place_at(rb, target_xy, grasp_z)
    log("done ended", rb.ended, "complete", rb.complete, "step", rb.meta.get("step_id"), "t", round(time.time() - t0, 1))
    tr = rb.meta.get("transition") or {}
    log("transition", json.dumps({k: tr.get(k) for k in ("episode_ended", "task_complete")}))
