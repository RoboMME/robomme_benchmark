"""VideoRepick controller v4 (v3 + faster servo: per-row rotation, looser waypoint tolerances).

1. Read the conditioning video manifest, find the expert's release, track the released object
   through the shuffle with the static front camera (tracker.py).
2. Locate the button (white base) in the front camera.
3. Twice: hover above the target, localise it in the wrist view (crescent centre + bite direction),
   grasp across the solid back with the finger axis perpendicular to the bite, lift, verify,
   place back, release, verify nothing stays attached.
4. Press the button with closed fingers.
"""
import glob
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from motion import Robot, down_mat, meta  # noqa: E402
from tracker import track  # noqa: E402
from vision import (backproject, green_mask, green_points_world, label, load_depth, load_rgb,  # noqa: E402
                    object_pose_from_view, learn_object_hue, set_object_hue, closing_width,
                    min_enclosing_circle)

ROOT = "/workspace"
OUT = os.path.join(ROOT, "output")
HOVER_Z = 0.18
GRASP_Z = 0.020
LIFT_Z = 0.15
PLACE_Z = 0.027
BITE_BACK_OFFSET = 0.004
_LOG = None


def log(*a):
    global _LOG
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    try:
        if _LOG is None:
            os.makedirs(OUT, exist_ok=True)
            _LOG = open(os.path.join(OUT, "controller_log.txt"), "a")
        _LOG.write(f"{time.time():.2f} {msg}\n"); _LOG.flush()
    except Exception:
        pass


# ---------------------------------------------------------------- observation helpers
def latest_frame(m):
    """Return the newest frame record (with files) referenced by an observation/step packet."""
    seq = m.get("frame_sequence") or {}
    mp = seq.get("manifest_path")
    if not mp:
        return None
    man = json.load(open(os.path.join(ROOT, mp)))
    frames = sorted(man["frames"], key=lambda f: f["frame_index"])
    return frames[-1]


def view(frame, cam):
    if "arrays" in frame:
        a = frame["arrays"]
        return a[(cam, "rgb")], a[(cam, "depth")], frame["camera_intrinsics"][cam]["matrix"], frame["camera_poses"][cam]
    files = {(x["camera"], x["kind"]): x["path"] for x in frame["files"]}
    rgb = load_rgb(os.path.join(ROOT, files[(cam, "rgb")]))
    dep = load_depth(os.path.join(ROOT, files[(cam, "depth")]))
    return rgb, dep, frame["camera_intrinsics"][cam]["matrix"], frame["camera_poses"][cam]


def fresh_frame(rb):
    return rb.observe_frame()


def find_initial_manifest(m0):
    seq = m0.get("frame_sequence") or {}
    mp = seq.get("manifest_path")
    if mp:
        p = os.path.join(ROOT, mp)
        try:
            man = json.load(open(p))
            if any(f.get("phase") == "video_demo" for f in man["frames"]):
                return p
        except Exception as e:
            log("manifest read failed", e)
    cands = sorted(glob.glob(os.path.join(ROOT, "runtime/autonomous_controller/runs/*/frames/sequence_000000/manifest.json")),
                   key=os.path.getmtime)
    for p in reversed(cands):
        try:
            man = json.load(open(p))
            if any(f.get("phase") == "video_demo" for f in man["frames"]):
                log("using fallback manifest", p)
                return p
        except Exception:
            pass
    return None


def front_objects(frame):
    """Green blob centres (world xy) on the table from the front camera."""
    rgb, dep, K, pose = view(frame, "front")
    pw, _, _ = green_points_world(rgb, dep, K, pose)
    from tracker import components_xy, table_points
    xy = table_points(pw)
    comps = components_xy(xy)
    return [xy[c].mean(0) for c in comps if len(c) >= 15]


def find_button(frame):
    """Button: low-saturation raised (10-40 mm) compact cluster on the table. Returns world xy of its centre."""
    from tracker import components_xy
    from vision import rgb_to_hsv
    for cam in ("front", "wrist"):
        rgb, dep, K, pose = view(frame, cam)
        vs, us = np.nonzero(dep > 0)
        pw = backproject(us.astype(float), vs.astype(float), dep[vs, us], K, pose)
        hsv = rgb_to_hsv(rgb[vs, us].astype(float) / 255.0)
        sel = (pw[:, 2] > 0.010) & (pw[:, 2] < 0.04) & (hsv[:, 1] < 0.15) & (pw[:, 0] > -0.45) & (pw[:, 0] < 0.25) \
            & (np.abs(pw[:, 1]) < 0.35)
        p = pw[sel]
        if len(p) < 20:
            continue
        comps = components_xy(p[:, :2], link=0.008)
        best = None
        for c in comps:
            if len(c) < 20:
                continue
            ext = p[c, :2].max(0) - p[c, :2].min(0)
            if ext.max() > 0.09:
                continue
            if best is None or len(c) > len(best):
                best = c
        if best is not None:
            q = p[best]
            ctr = q[:, :2].mean(0)
            log(f"button via {cam}: {len(best)} pts centre {np.round(ctr, 4)} mean {np.round(q[:, :2].mean(0), 4)}")
            return ctr
    return None


def refine_button(rb, guess):
    """Hover above the guess; cap = raised points (z 0.026-0.05) near the guess in wrist depth (no colour)."""
    guess = np.asarray(guess, float)
    rb.move_to([guess[0] + 0.0, guess[1], 0.20], down_mat(0.0), gripper=-1.0, max_steps=80)
    if rb.ended:
        return guess, 0.035
    fr = fresh_frame(rb)
    est, top = guess.copy(), 0.035
    for cam in ("wrist", "front"):
        rgb, dep, K, pose = view(fr, cam)
        vs, us = np.nonzero(dep > 0)
        pw = backproject(us.astype(float), vs.astype(float), dep[vs, us], K, pose)
        c = guess.copy()
        ok = False
        for rad in (0.03, 0.025, 0.022):
            sel = (np.linalg.norm(pw[:, :2] - c, axis=1) < rad) & (pw[:, 2] > 0.026) & (pw[:, 2] < 0.05)
            if sel.sum() < 25:
                break
            c = pw[sel, :2].mean(0); ok = True
            top = float(np.percentile(pw[sel, 2], 90))
        log(f"button refine ({cam}): ok {ok} centre {np.round(c, 4)} top {top:.3f} shift {np.linalg.norm(c - guess):.3f}")
        if ok and np.linalg.norm(c - guess) < 0.03:
            est = c
            break
    return est, top


def wrist_close_green(fr, max_depth=0.10):
    rgb, dep, K, pose = view(fr, "wrist")
    m = green_mask(rgb) & (dep > 0) & (dep < max_depth)
    return int(m.sum())


def localize_target(rb, xy, z=HOVER_Z):
    """Hover above xy (camera offset compensated roughly) and localise the crescent in the wrist view."""
    if rb.ended:
        return None
    rb.move_to([xy[0], xy[1], z], down_mat(0.0), gripper=-1.0, max_steps=100, tol=0.008, rtol=0.06)
    if rb.ended:
        return None
    fr = fresh_frame(rb)
    rgb, dep, K, pose = view(fr, "wrist")
    o = object_pose_from_view(rgb, dep, K, pose, np.asarray(xy, float))
    if o is None:
        rgb, dep, K, pose = view(fr, "front")
        o = object_pose_from_view(rgb, dep, K, pose, np.asarray(xy, float), radius=0.05, ztop=0.034)
        if o is not None:
            o["source"] = "front"
    else:
        o["source"] = "wrist"
    if o is not None:
        log(f"localize near {np.round(xy, 3)} -> centre {np.round(o['center'], 4)} R {o.get('R')} bite {np.round(o['bite_vec'], 4)} "
            f"ntop {o['n_top']} src {o['source']}")
    else:
        log(f"localize near {np.round(xy, 3)} failed")
    return o


def norm_yaw(yaw):
    while yaw > np.pi / 2:
        yaw -= np.pi
    while yaw < -np.pi / 2:
        yaw += np.pi
    return yaw


def grasp_plan(o):
    """Return grasp xy, yaw, expected closing width, grasp z."""
    top = o.get("top_xy")
    allxy = o.get("all_xy")
    zmax = float(o.get("zmax", 0.04))
    gz = float(np.clip(zmax * 0.5, 0.012, 0.03))
    if top is None or len(top) < 15:
        return np.asarray(o["center"], float), 0.0, None, gz
    g = top.mean(0)
    c, R = min_enclosing_circle(top)
    b = c - g
    shape_xy = top
    if np.linalg.norm(b) > 0.003 and R > 0.012:
        yaw = norm_yaw(float(np.arctan2(b[1], b[0])))
        W, mid = closing_width(shape_xy, g, yaw)
        why = "concave"
    else:
        best = None
        for yd in range(0, 180, 10):
            yaw_c = np.radians(yd)
            W_c, mid_c = closing_width(shape_xy, g, yaw_c)
            if W_c is None or W_c < 0.012:
                continue
            if best is None or W_c < best[1] - 1e-4:
                best = (yaw_c, W_c, mid_c)
        if best is None:
            yaw, W, mid = 0.0, None, 0.0
        else:
            yaw, W, mid = best
        yaw = norm_yaw(yaw)
        why = "min-width"
    if mid:
        ax = np.array([-np.sin(yaw), np.cos(yaw)])
        g = g + mid * ax
    log(f"grasp plan ({why}): g {np.round(g, 4)} yaw {np.degrees(yaw):.0f} W {W} R {R:.4f} conc {np.linalg.norm(b):.4f} gz {gz:.3f}")
    return g, yaw, W, gz


def pick(rb, o):
    g, yaw, W, gz = grasp_plan(o)
    R = down_mat(yaw)
    log(f"pick at {np.round(g, 4)} yaw {np.degrees(yaw):.0f}")
    rb.move_to([g[0], g[1], 0.10], R, gripper=-1.0, max_steps=100, tol=0.01)
    rb.move_to([g[0], g[1], 0.05], R, gripper=-1.0, max_steps=60, vmax=0.6, tol=0.006)
    rb.move_to([g[0], g[1], gz], R, gripper=-1.0, max_steps=60, vmax=0.4, tol=0.003)
    rb.hold(15, 1.0)
    w_closed = rb.state[6] + rb.state[7]
    if W is not None and W > 0.02 and w_closed < 0.6 * W:
        log(f"closed too far ({w_closed:.4f} vs expected {W:.4f}): missed or finger in a concavity; abort grasp")
        return False, g, yaw, gz
    if w_closed < 0.004:
        log("gripper closed on nothing")
        return False, g, yaw, gz
    rb.move_to([g[0], g[1], LIFT_Z], R, gripper=1.0, max_steps=80, vmax=0.6, tol=0.01, rtol=0.06)
    rb.hold(3, 1.0)
    w = rb.state[6] + rb.state[7]
    log(f"after lift: grip {w:.4f} (at close {w_closed:.4f}) z {rb.state[2]:.3f} ended {rb.ended}")
    ok = w > 0.004
    return ok, g, yaw, gz


def place(rb, g, yaw, gz):
    R = down_mat(yaw)
    rb.move_to([g[0], g[1], 0.06], R, gripper=1.0, max_steps=60, vmax=0.6, tol=0.006)
    rb.move_to([g[0], g[1], gz + 0.007], R, gripper=1.0, max_steps=40, vmax=0.3, tol=0.003)
    rb.hold(10, -1.0)
    rb.move_to([g[0], g[1], 0.06], R, gripper=-1.0, max_steps=40, vmax=0.3, tol=0.008, rtol=0.06)
    rb.move_to([g[0], g[1], 0.12], R, gripper=-1.0, max_steps=40, vmax=0.6, tol=0.012, rtol=0.08)
    o = localize_target(rb, g)
    on_table = o is not None and o["n_top"] >= 15 and np.linalg.norm(np.asarray(o["center"]) - g) < 0.04
    log(f"after release: grip {rb.state[6] + rb.state[7]:.4f} object on table: {on_table}")
    return on_table, o


def unhook(rb, g, yaw, gz=0.02):
    """Object not seen on the table after release: lower, open, rise slowly, look again."""
    log("unhook attempt")
    R = down_mat(yaw)
    rb.move_to([g[0], g[1], gz + 0.02], R, gripper=-1.0, max_steps=50, vmax=0.3)
    rb.hold(8, -1.0)
    rb.move_to([g[0], g[1], 0.12], R, gripper=-1.0, max_steps=60, vmax=0.25)
    o = localize_target(rb, g)
    ok = o is not None and o["n_top"] >= 15
    log(f"after unhook: object on table {ok}")
    return ok


def press_button(rb, bxy, top):
    log(f"press button at {np.round(bxy, 4)} top {top:.3f}")
    rb.move_to([bxy[0], bxy[1], 0.12], down_mat(0.0), gripper=-1.0, max_steps=100, tol=0.006)
    rb.hold(8, 1.0)
    rb.move_to([bxy[0], bxy[1], top + 0.02], down_mat(0.0), gripper=1.0, max_steps=60, vmax=0.5)
    for zt in (top - 0.006, top - 0.012, top - 0.018):
        if rb.ended:
            break
        rb.move_to([bxy[0], bxy[1], zt], down_mat(0.0), gripper=1.0, max_steps=25, vmax=0.25, tol=0.002)
        log(f"press z target {zt:.3f} reached {rb.state[2]:.4f} ended {rb.ended} complete {rb.complete}")
    if not rb.ended:
        rb.hold(5, 1.0)
        rb.move_to([bxy[0], bxy[1], 0.10], down_mat(0.0), gripper=1.0, max_steps=40)


def main(ctx):
    try:
        _main(ctx)
    except Exception:
        log("FATAL controller exception", traceback.format_exc())


def _main(ctx):
    t0 = time.time()
    rb = Robot(ctx, log=log)
    m0 = rb.observe()
    log("instruction:", m0.get("instruction"), "limits", m0.get("episode_limits"))
    mp = find_initial_manifest(m0)
    target_xy = None
    fr0 = latest_frame(m0)
    button = None
    if mp:
        try:
            man = json.load(open(mp))
            vframes = sorted(man["frames"], key=lambda f: f["frame_index"])
            f0 = vframes[0]
            rgb, dep, K, pose = view(f0, "front")
            hue, npx = learn_object_hue(rgb, dep, K, pose)
            log(f"object hue {hue} from {npx} px")
            set_object_hue(hue)
            button = find_button(f0)
            if fr0 is None:
                fr0 = vframes[-1]
        except Exception:
            log("hue/button prep failed", traceback.format_exc())
    if button is None and fr0 is not None:
        button = find_button(fr0)
    log("button guess", None if button is None else np.round(button, 4))
    import tracker as TR
    import vision as VI
    TR.EXCLUDE = []
    VI.EXCLUDE_XY = [] if button is None else [(np.asarray(button), 0.024)]
    TR.WRIST_EXCLUDE = [] if button is None else [(np.asarray(button), 0.03)]
    if button is not None and mp:
        try:
            TR.PIXEL_EXCLUDE = TR.build_button_pixel_mask(f0, ROOT, button)
            log(f"button pixel mask {int(TR.PIXEL_EXCLUDE.sum())} px")
        except Exception:
            log("button mask failed", traceback.format_exc())
    if mp:
        try:
            r = track(mp, ROOT, log=log)
            target_xy = np.asarray(r["target_xy"], float)
            log(f"tracked target xy {np.round(target_xy, 4)} in {time.time() - t0:.1f}s")
        except Exception:
            log("tracking failed", traceback.format_exc())
    objs = front_objects(fr0) if fr0 else []
    log("front objects now", [np.round(o, 3).tolist() for o in objs])
    if target_xy is None and objs:
        target_xy = np.asarray(objs[0])
    elif objs:
        d = [np.linalg.norm(o - target_xy) for o in objs]
        j = int(np.argmin(d))
        log(f"nearest live blob {np.round(objs[j], 4)} dist {d[j]:.3f}")

    xy = target_xy
    done = 0
    attempts = 0
    while done < 2 and attempts < 5 and not rb.ended:
        attempts += 1
        o = localize_target(rb, xy)
        if o is None:
            # wider look from higher up
            o = localize_target(rb, xy, z=0.28)
            if o is None:
                log("target lost; abort picks"); break
        if np.linalg.norm(np.asarray(o["center"]) - xy) > 0.035:
            log("warning: localized centre far from expected", np.round(o["center"], 3))
        ok, g, yaw, gz = pick(rb, o)
        if rb.ended:
            break
        if not ok:
            log("grasp failed; reopen and retry")
            R = down_mat(yaw)
            rb.move_to([g[0], g[1], 0.05], R, gripper=1.0, max_steps=40, vmax=0.5)
            rb.hold(10, -1.0)
            rb.move_to([g[0], g[1], 0.12], R, gripper=-1.0, max_steps=40)
            xy = np.asarray(o["center"], float)
            continue
        released, o_after = place(rb, g, yaw, gz)
        if not released and not rb.ended:
            released = unhook(rb, g, yaw, gz)
        done += 1
        xy = np.asarray(o_after["center"], float) if (released and o_after is not None) else np.asarray(g, float)
        log(f"pick/place {done} done (released={released}) step {rb.step_id}")
    if not rb.ended:
        bxy, top = (button, 0.032) if button is not None else (None, 0.032)
        if bxy is not None:
            bxy, top = refine_button(rb, bxy)
            press_button(rb, bxy, top)
        else:
            log("no button found")
    log(f"END step {rb.step_id} ended {rb.ended} complete {rb.complete} wall {time.time() - t0:.1f}s")
