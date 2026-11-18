"""VideoRepick controller v1.

Pipeline (steps -> exits):
  INIT      : observe step 0, load conditioning video manifest           -> TRACK (fail: TRACK with live-only fallback)
  TRACK     : find pick/release in video from robot state, track all
              identical objects (front RGB-D, world XY) to video end     -> MATCH
  MATCH     : match final track to live front clusters                   -> PICK1
  PICK k=1,2: wrist-refine target near belief, grasp, verify fingers,
              lift, place back, open, lift, re-localise target           -> PICK2 / BUTTON (retry up to N per pick)
  BUTTON    : locate button (front, low-saturation bright raised blob),
              wrist-refine, press down until episode ends or depth limit -> DONE
main() never raises; every phase catches exceptions and continues.
"""
import json
import os
import sys
import time
import traceback

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import backproject  # noqa: E402
import track  # noqa: E402
from track import green_mask, cluster_count, kmeans_track, mean_shift, ZMIN, ZMAX  # noqa: E402
from robot import Robot, parse  # noqa: E402

VERSION = "repick-v2"
WS = "/workspace"
OUT = "/workspace/output"

SAFE_Z = 0.15          # travel height (expert lifted to 0.15)
VIEW_Z = 0.20          # wrist inspection height
GRASP_Z = 0.022        # TCP height for grasp (ep1: 0.0225 worked, object 4cm tall)
PLACE_Z = 0.032        # TCP height for release (expert 0.031)
LIFT_Z = 0.15
CLOSE_STEPS = 12
OPEN_STEPS = 8
HOLD_MIN = 0.004       # finger sum above this after closing = holding something
OPEN_MIN = 0.07
EXCL_R = 0.035         # button exclusion radius (purple light made button saturated in rehearsal 1)
XMIN_WS = -0.45        # exclude robot base region (base ~x=-0.6; objects seen x in [-0.3,0.13])
FINGER_R = 0.045       # open fingertip distance from TCP along closing axis
CLEAR_OK = 0.045       # neighbour centre to fingertip clearance needing no yaw change


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(f"[{VERSION} {time.strftime('%H:%M:%S')}] {msg}", flush=True)
    try:
        with open(os.path.join(OUT, "controller.log"), "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------- perception
def scene_points(rgb, depth, K, pose, excl=()):
    """Raised saturated points (objects); excl = xy centres (button) to drop within EXCL_R."""
    W = backproject(depth, K, pose)
    z = W[..., 2]
    m = green_mask(rgb) & (z > ZMIN) & (z < ZMAX) & (W[..., 0] > XMIN_WS)
    P = W[m]
    for e in excl:
        if e is not None and len(P):
            P = P[np.linalg.norm(P[:, :2] - np.asarray(e)[:2], axis=1) > EXCL_R]
    return P


BUTTON_XY = []   # set once in main (within-episode memory)


def clusters_from(meta, cam):
    imgs = meta["_imgs"]
    pts = scene_points(imgs[(cam, "rgb")], imgs[(cam, "depth")],
                       meta["camera_intrinsics"][cam]["matrix"], meta["camera_poses"][cam], BUTTON_XY)
    return [np.array(c) for c in cluster_count(pts)], pts


def load_frame(rel, cam):
    files = {(f["camera"], f["kind"]): f["path"] for f in rel["files"]}
    rgb = np.array(Image.open(os.path.join(WS, files[(cam, "rgb")])).convert("RGB"))
    d = np.array(Image.open(os.path.join(WS, files[(cam, "depth")]))).astype(np.float64) * 0.001
    return rgb, d


def track_video(meta0, excl=()):
    """Return (target_xy_at_video_end, info). Raises on failure."""
    seq = meta0.get("frame_sequence") or {}
    man = json.load(open(os.path.join(WS, seq["manifest_path"])))
    frames = sorted(man["frames"], key=lambda f: f["frame_index"])
    vid = [f for f in frames if f.get("phase") == "video_demo"]
    if len(vid) < 10:
        raise RuntimeError("no conditioning video")
    K = meta0["camera_intrinsics"]["front"]["matrix"]
    pose = meta0["camera_poses"]["front"]
    grip = np.array([sum(f["state"][6:8]) for f in vid])
    closed = np.nonzero(grip < OPEN_MIN)[0]
    if len(closed) == 0:
        raise RuntimeError("no grasp in video")
    # first closed run
    g0 = closed[0]
    g1 = g0
    while g1 + 1 < len(vid) and grip[g1 + 1] < OPEN_MIN:
        g1 += 1
    rel_xy = np.array(vid[g1]["state"][:2])
    grasp_xy = np.array(vid[g0]["state"][:2])
    log("video frames", len(vid), "grasp frame", g0, "release frame", g1,
        "grasp_xy", np.round(grasp_xy, 3).tolist(), "release_xy", np.round(rel_xy, 3).tolist())
    # objects before the grasp, robot high: count and initial positions
    counts, inits = [], []
    high = [i for i in range(0, g0) if vid[i]["state"][2] > 0.30][:8] or [0]
    for i in high:
        rgb, d = load_frame(vid[i], "front")
        cs = cluster_count(scene_points(rgb, d, K, pose, excl))
        counts.append(len(cs))
        inits.append(np.array(cs))
    N = int(np.bincount(counts).argmax()) if counts else 0
    if N < 2 and track.SAT_MIN >= 0:
        # objects not saturated in this appearance: fall back to height-only segmentation
        log("few saturated objects (", N, "); switching to height-only mask")
        track.SAT_MIN = -1
        counts, inits = [], []
        for i in high:
            rgb, d = load_frame(vid[i], "front")
            cs = cluster_count(scene_points(rgb, d, K, pose, excl))
            counts.append(len(cs))
            inits.append(np.array(cs))
        N = int(np.bincount(counts).argmax()) if counts else 0
    C = next((c for c, n in zip(inits, counts) if n == N), None)
    log("object count estimate", N, counts)
    if C is None or len(C) == 0:
        raise RuntimeError("could not initialise tracks")
    dg = np.linalg.norm(C - grasp_xy, axis=1)
    tgt = int(dg.argmin())
    srt = np.sort(dg)
    log("picked object id", tgt, "at", np.round(C[tgt], 3).tolist(), "dist to grasp", round(float(srt[0]), 3),
        "next nearest", round(float(srt[1]), 3) if len(srt) > 1 else None)
    C = C.copy()
    C[tgt] = rel_xy
    # track from the first frame after release with the gripper lifted clear
    start = next((i for i in range(g1 + 1, len(vid)) if vid[i]["state"][2] > 0.12), g1 + 1)
    start = min(start, len(vid) - 1)
    log("track start", start)
    hist = [C.copy()]
    minsep, minpair, mincnt = 1.0, 1.0, 10 ** 6
    for i in range(start, len(vid)):
        rgb, d = load_frame(vid[i], "front")
        P = scene_points(rgb, d, K, pose, excl)[:, :2]
        vel = hist[-1] - hist[-2] if len(hist) > 1 else 0.0
        Cn, cnt = kmeans_track(P, hist[-1] + 0.5 * vel)
        hist.append(Cn)
        dd = np.linalg.norm(Cn - Cn[tgt], axis=1)
        dd[tgt] = 1.0
        minsep = min(minsep, float(dd.min()))
        DD = np.linalg.norm(Cn[:, None] - Cn[None], axis=2) + np.eye(len(Cn))
        minpair = min(minpair, float(DD.min()))
        mincnt = min(mincnt, int(cnt.min()))
    H = np.array(hist)
    try:
        np.save(os.path.join(OUT, "tracks.npy"), H)
    except Exception:
        pass
    log("tracked", len(H), "frames; target end", np.round(H[-1][tgt], 3).tolist(),
        "min target separation", round(minsep, 3), "min pair sep", round(minpair, 3),
        "min points/object", mincnt)
    return H[-1][tgt], {"tracks_end": H[-1], "tgt": tgt, "N": N, "minsep": minsep}


def nearest(cs, xy, gate):
    if not cs:
        return None
    d = [float(np.linalg.norm(c - xy)) for c in cs]
    i = int(np.argmin(d))
    return cs[i] if d[i] < gate else None


def wrist_refine(rb, xy, gate=0.04):
    """Move above xy, look with wrist camera; return (refined xy or None, neighbour xys)."""
    rb.move_to([xy[0], xy[1], VIEW_Z], grip=-1.0, tol=0.006)
    m = rb.observe()
    cs, pts = clusters_from(m, "wrist")
    c = None
    if len(pts):
        c0 = nearest(cs, xy, gate)
        c, n = mean_shift(pts[:, :2], c0 if c0 is not None else xy)
        if c is not None and np.linalg.norm(c - xy) > gate:
            c = None
    others = [q for q in cs if c is None or np.linalg.norm(q - c) > 0.035]
    top = None
    if c is not None:
        sel = np.linalg.norm(pts[:, :2] - c, axis=1) < 0.022
        if sel.sum() >= 5:
            top = float(np.percentile(pts[sel, 2], 95))
    log("wrist clusters", [np.round(x, 3).tolist() for x in cs], "->",
        None if c is None else np.round(c, 3).tolist(), "top", None if top is None else round(top, 3))
    return c, others, top


def choose_yaw(xy, others):
    """Finger closing axis is world y at yaw 0; rotate so open fingertips avoid neighbours."""
    best, bclear = 0.0, -1.0
    for yaw in (0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5):
        ax = np.array([-np.sin(yaw), np.cos(yaw)])
        tips = [xy + FINGER_R * ax, xy - FINGER_R * ax]
        clear = min([np.linalg.norm(t - o) for t in tips for o in others] or [1.0])
        if yaw == 0.0 and clear >= CLEAR_OK:
            return 0.0, clear
        if clear > bclear + 0.005:
            best, bclear = yaw, clear
    return best, bclear


def fingers(rb):
    s = rb.state()
    return float(s[6] + s[7])


# ---------------------------------------------------------------- skills
def pick_place(rb, xy, k):
    """One pick-up-and-place-down of the object at xy. Returns (ok, new_xy)."""
    if fingers(rb) < OPEN_MIN:
        # entered with a closed gripper (e.g. after a tool error): put down whatever is held
        s = rb.state()
        log(f"pick{k} entry with closed gripper; releasing at", np.round(s[:3], 3).tolist())
        rb.move_to([s[0], s[1], max(PLACE_Z, min(s[2], 0.1))], grip=1.0, tol=0.006, max_steps=40)
        rb.hold(OPEN_STEPS, -1.0)
        rb.move_to([s[0], s[1], 0.10], grip=-1.0, tol=0.006, max_steps=40)
    for attempt in range(3):
        if rb.ended:
            return False, xy
        c, others, obj_top = wrist_refine(rb, xy)
        if c is None:
            # look again from front view
            mf = rb.observe()
            cs, pts = clusters_from(mf, "front")
            c0 = nearest(cs, xy, 0.05)
            c = c0
            log("front fallback", None if c is None else np.round(c, 3).tolist())
            if c is None:
                c = xy
            others = [q for q in cs if np.linalg.norm(q - c) > 0.035]
            obj_top = None
        xy = np.array(c)
        yaw, clear = choose_yaw(xy, others)
        gz = GRASP_Z
        if obj_top is not None and 0.015 < obj_top < 0.09:
            gz = float(np.clip(obj_top - 0.019, 0.012, 0.035))
        if attempt == 1:
            gz = GRASP_Z - 0.006
            yaw = float(np.clip(yaw + 0.8, -1.5, 1.5))
        elif attempt >= 2:
            gz = GRASP_Z + 0.006
            yaw = float(np.clip(yaw - 0.8, -1.5, 1.5))
        log(f"pick{k} target", np.round(xy, 3).tolist(), "yaw", yaw, "clearance", round(clear, 3), "grasp z", round(gz, 3))
        rb.move_to([xy[0], xy[1], 0.08], grip=-1.0, yaw=yaw, tol=0.005)
        rb.move_to([xy[0], xy[1], gz], grip=-1.0, yaw=yaw, tol=0.004)
        rb.hold(CLOSE_STEPS, 1.0)
        fs = fingers(rb)
        log(f"pick{k} attempt{attempt} closed fingers={fs:.4f}")
        if fs < HOLD_MIN:
            rb.hold(OPEN_STEPS, -1.0)
            rb.move_to([xy[0], xy[1], 0.10], grip=-1.0, yaw=yaw, tol=0.006)
            continue
        rb.move_to([xy[0], xy[1], LIFT_Z], grip=1.0, yaw=yaw, tol=0.006)
        fs = fingers(rb)
        log(f"pick{k} lifted fingers={fs:.4f} z={rb.state()[2]:.3f}")
        if fs < HOLD_MIN:
            # dropped during lift; object lies somewhere near xy
            rb.hold(OPEN_STEPS, -1.0)
            continue
        rb.hold(4, 1.0)
        rb.move_to([xy[0], xy[1], PLACE_Z], grip=1.0, yaw=yaw, tol=0.004)
        rb.hold(OPEN_STEPS, -1.0)
        rb.move_to([xy[0], xy[1], 0.10], grip=-1.0, yaw=yaw, tol=0.006)
        log(f"pick{k} placed at", np.round(rb.state()[:2], 3).tolist())
        return True, xy
    return False, xy


def find_button(meta):
    """Raised, bright, low-saturation blob on the table in the front view (not robot)."""
    imgs = meta["_imgs"]
    rgb = imgs[("front", "rgb")].astype(int)
    W = backproject(imgs[("front", "depth")], meta["camera_intrinsics"]["front"]["matrix"],
                    meta["camera_poses"]["front"])
    z = W[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    m = (mx - mn < 25) & (mx > 120) & (z > 0.005) & (z < 0.06) & (W[..., 0] > -0.45)
    pts = W[m]
    cs = cluster_count(pts, min_pts=5)
    if not cs:
        return None, None
    # largest cluster
    best, bn = None, 0
    for c in cs:
        n = int((np.linalg.norm(pts[:, :2] - c, axis=1) < 0.03).sum())
        if n > bn:
            best, bn = c, n
    return button_top(pts, best)


def press_button(rb, bxy, btop):
    log("button at", np.round(bxy, 3).tolist(), "top", round(btop, 3))
    rb.move_to([bxy[0], bxy[1], VIEW_Z], grip=-1.0, tol=0.006)
    m = rb.observe()
    b2, t2 = find_button_wrist(m)
    if b2 is not None and np.linalg.norm(b2 - bxy) < 0.02 and abs(t2 - btop) < 0.01:
        log("button wrist refine", np.round(b2, 3).tolist(), round(t2, 3))
        bxy, btop = b2, t2
    rb.move_to([bxy[0], bxy[1], btop + 0.06], grip=1.0, tol=0.005)
    z = btop + 0.03
    zmin = max(0.008, btop - 0.03)
    while not rb.ended and z > zmin:
        rb.move_to([bxy[0], bxy[1], z], grip=1.0, tol=0.004, max_steps=12)
        z -= 0.006
    log("press done z", round(rb.state()[2], 3), "ended", rb.ended)
    if not rb.ended:
        rb.hold(5, 1.0)
        rb.move_to([bxy[0], bxy[1], btop + 0.08], grip=1.0, tol=0.006)


def find_button_wrist(meta):
    imgs = meta["_imgs"]
    rgb = imgs[("wrist", "rgb")].astype(int)
    W = backproject(imgs[("wrist", "depth")], meta["camera_intrinsics"]["wrist"]["matrix"],
                    meta["camera_poses"]["wrist"])
    z = W[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    m = (mx - mn < 25) & (mx > 120) & (z > 0.005) & (z < 0.06)
    pts = W[m]
    cs = cluster_count(pts, min_pts=20)
    if not cs:
        return None, None
    best, bn = None, 0
    for c in cs:
        n = int((np.linalg.norm(pts[:, :2] - c, axis=1) < 0.03).sum())
        if n > bn:
            best, bn = c, n
    return button_top(pts, best)


def button_top(pts, c):
    sel = np.linalg.norm(pts[:, :2] - c, axis=1) < 0.035
    p = pts[sel]
    top = float(np.percentile(p[:, 2], 95))
    hi = p[p[:, 2] > top - 0.008]
    xy = hi[:, :2].mean(0) if len(hi) >= 3 else np.asarray(c)
    return np.array(xy), top


def parse_count(instr):
    t = (instr or "").lower()
    words = [("twice", 2), ("two times", 2), ("thrice", 3), ("three times", 3), ("four times", 4),
             ("once", 1), ("one time", 1)]
    for w, n in words:
        if w in t:
            return n
    import re
    mm = re.search(r"(\d+)\s*times", t)
    if mm:
        return max(1, min(5, int(mm.group(1))))
    return 2


# ---------------------------------------------------------------- main
def main(ctx):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rb = Robot(ctx, log=log)
    try:
        m0 = rb.observe()
    except Exception:
        log("observe failed", traceback.format_exc())
        return
    log("instruction:", m0.get("instruction"), "limits", m0.get("episode_limits"))
    bxy, btop = None, None
    try:
        bxy, btop = find_button(m0)
        log("button front", None if bxy is None else np.round(bxy, 3).tolist(), btop)
        BUTTON_XY[:] = [] if bxy is None else [np.array(bxy)]
    except Exception:
        log("button detect failed", traceback.format_exc())
    target = None
    try:
        target, info = track_video(m0, BUTTON_XY)
    except Exception:
        log("tracking failed", traceback.format_exc())
    log("tracking done in", round(time.time() - t0, 1), "s")
    try:
        cs, _ = clusters_from(m0, "front")
        log("live front clusters", [np.round(c, 3).tolist() for c in cs])
        if target is None:
            target = cs[0] if cs else np.array([-0.1, 0.0])
        else:
            _, pts = clusters_from(m0, "front")
            c, n = mean_shift(pts[:, :2], target) if len(pts) else (None, 0)
            if c is None or np.linalg.norm(c - target) > 0.05:
                c = nearest(cs, target, 0.05)
            if c is not None:
                target = c
    except Exception:
        log("live match failed", traceback.format_exc())
    log("TARGET", np.round(target, 3).tolist())
    xy = np.array(target, float)
    n_needed = parse_count(m0.get("instruction", ""))
    log("pick-place repetitions required", n_needed)
    done, calls = 0, 0
    while done < n_needed and calls < n_needed + 2 and not rb.ended:
        calls += 1
        if rb.remaining < 120:
            log("too few steps left for another pick", rb.remaining)
            break
        try:
            ok, xy = pick_place(rb, xy, done + 1)
            done += int(ok)
            log(f"pick{done} ok={ok} steps={rb.steps} remaining={rb.remaining}")
        except Exception:
            log(f"pick exception", traceback.format_exc())
            try:
                rb.observe()
            except Exception:
                pass
        if rb.ended:
            log("episode ended", "complete", rb.complete)
            return
    try:
        if bxy is None:
            m = rb.move_to([-0.15, 0.0, 0.3], grip=-1.0) and rb.observe()
            bxy, btop = find_button(rb.observe())
        if bxy is not None:
            press_button(rb, bxy, btop)
            # retries: re-detect from a view above the button and press again
            for extra in range(2):
                if rb.ended or rb.remaining < 20:
                    break
                rb.move_to([bxy[0], bxy[1], 0.25], grip=-1.0, tol=0.008)
                mm = rb.observe()
                b2, t2 = find_button(mm)
                if b2 is None or np.linalg.norm(b2 - bxy) > 0.05:
                    b2, t2 = bxy, btop
                log("button retry", extra, np.round(b2, 3).tolist(), t2)
                press_button(rb, b2, t2 - 0.01 * (extra + 1))
    except Exception:
        log("button exception", traceback.format_exc())
    log("finished: steps", rb.steps, "ended", rb.ended, "complete", rb.complete,
        "wall", round(time.time() - t0, 1))
