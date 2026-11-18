"""VideoPlaceButton controller (vpb).

Pipeline (each step re-perceives; failures escalate, never crash):
  S0 observe@0 + load conditioning video manifest
  S1 parse video events (grasp/release/press from EE state + finger width), parse instruction,
     select referenced release -> goal xy; refine with pink-target clusters in front view
  S2 locate the cube (front colour+depth, prior = last release of that colour in video)
  S3 hover above cube, refine xy + yaw with wrist camera
  S4 descend, close, lift; verify finger width (hold) -> else back to S2 (max 3 tries)
  S5 carry at safe height to goal, descend, open, lift
  S6 verify cube xy near goal from front view; if far and steps remain -> back to S2
"""
import base64
import io
import json
import math
import os
import time
import traceback

import numpy as np
from PIL import Image

from geom import Cam, rotvec_to_R, R_to_rotvec, yaw_of_down_gripper
import perception as P
import video as V
import instruction as I
import tracker as TR
import swaps as SW

VERSION = "vpb-v3"
POS_SCALE, ROT_SCALE = 0.05, 0.5
SAFE_Z = 0.15            # carry height (expert used ~0.15)
HOVER_Z = 0.10           # wrist inspection height above the cube
GRASP_Z = 0.018          # EE z at grasp (expert 0.016-0.022)
PLACE_Z = 0.024          # EE z at release (expert 0.016-0.021)
HOLD_MIN_W = 0.012       # finger sum above this after closing = holding a cube
ON_TARGET_M = 0.04       # a release within this of a slot counts as placed on that target
TARGET_SNAP_M = 0.03     # use a detected target centre if within this of the video goal
OUT = "/workspace/output"
MANIFEST_OVERRIDE = None  # dev-only: set by experiment wrappers; None in rehearsal/formal
LOG = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    try:
        LOG.write(s + "\n"); LOG.flush()
    except Exception:
        pass


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = None
        self.ended = False
        self.complete = False
        self.remaining = None
        self._content = None

    def _ingest(self, r):
        meta = json.loads(r["content"][0]["text"])
        self._content = r["content"]
        self.meta = meta
        self.ended = bool(meta.get("episode_ended")) or self.ended
        if meta.get("task_complete") or (meta.get("transition") or {}).get("task_complete"):
            self.complete = True
        lim = meta.get("episode_limits") or {}
        self.remaining = lim.get("remaining_episode_steps", self.remaining)
        return meta

    def observe(self):
        return self._ingest(self.ctx.call("robomme_observe"))

    def step(self, rows):
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        if self.remaining is not None:
            rows = rows[:max(1, self.remaining)]
        return self._ingest(self.ctx.call("robomme_step", actions=rows))

    @property
    def state(self):
        return np.array(self.meta["state"], float)

    def images(self):
        out = {}
        for att in self.meta.get("attachments", []):
            b = self._content[att["content_index"]]
            im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
            arr = np.array(im)
            if att["kind"] == "depth":
                arr = arr.astype(float) * self.meta["depth"][att["camera"]].get("meters_per_unit", 0.001)
            else:
                arr = np.array(im.convert("RGB"))
            out[(att["camera"], att["kind"])] = arr
        return out

    def cam(self, name):
        return Cam(self.meta["camera_intrinsics"][name]["matrix"], self.meta["camera_poses"][name])


def target_R(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rz @ np.diag([1.0, -1.0, -1.0])


def wrap_quarter(a):
    """Wrap an angle to [-pi/4, pi/4)."""
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


def move_to(rb, pos, yaw, grip, tol=0.004, rot_tol=0.03, max_steps=80, vmax=0.5, settle=True):
    """Closed-loop EE move. Returns dict(ok, steps, err)."""
    pos = np.asarray(pos, float)
    Rt = target_R(yaw)
    n = 0
    err = None
    while n < max_steps and not rb.ended:
        s = rb.state
        e = pos - s[:3]
        Rc = rotvec_to_R(s[3:6])
        er = R_to_rotvec(Rt @ Rc.T)
        dist = float(np.linalg.norm(e)); rerr = float(np.linalg.norm(er))
        err = (dist, rerr)
        if dist < tol and rerr < rot_tol:
            return {"ok": True, "steps": n, "err": err}
        a = e / POS_SCALE
        m = np.abs(a).max()
        if m > vmax:
            a = a * (vmax / m)
        ar = er / ROT_SCALE
        mr = np.abs(ar).max()
        if mr > 0.6:
            ar = ar * (0.6 / mr)
        row = list(a) + list(ar) + [grip]
        k = 3 if dist > 0.08 else 1
        rb.step([row] * k)
        n += k
    return {"ok": False, "steps": n, "err": err}


def hold(rb, grip, k, yaw=None):
    s = rb.state
    rows = []
    for _ in range(k):
        rows.append([0, 0, 0, 0, 0, 0, grip])
    rb.step(rows)


def perceive_front(rb, color):
    im = rb.images()
    cam = rb.cam("front")
    tg = P.find_targets(im[("front", "rgb")], im[("front", "depth")], cam, TR.TARGET_COLOR)
    cubes = P.find_cubes(im[("front", "rgb")], im[("front", "depth")], cam, color)
    return tg, cubes


def perceive_wrist(rb, color, near_xy, max_off=0.06):
    im = rb.images()
    cam = rb.cam("wrist")
    cubes = P.find_cubes(im[("wrist", "rgb")], im[("wrist", "depth")], cam, color)
    best = None
    for c in cubes:
        d = float(np.linalg.norm(c["xy"] - near_xy))
        if d < max_off and (best is None or d < best[0]):
            best = (d, c)
    if best is None:
        return None
    c = best[1]
    yr = P.min_area_yaw(c["top"])
    return {"xy": c["xy"], "ztop": c["ztop"], "yaw": None if yr is None else yr[0],
            "ext": None if yr is None else yr[1].tolist(), "n": c["n"]}


def save_frame(rb, tag):
    try:
        im = rb.images()
        Image.fromarray(np.concatenate([im[("front", "rgb")], im[("wrist", "rgb")]], 1)).save(
            os.path.join(OUT, "vpb_%s.png" % tag))
    except Exception:
        pass


def plan_from_video(rb, root="/workspace"):
    seq = rb.meta.get("frame_sequence") or {}
    mpath = MANIFEST_OVERRIDE or seq.get("manifest_path")
    if mpath:
        try:
            _, fr = V.load_manifest(mpath, root)
            if not any(f.get("phase") == "video_demo" for f in fr):
                log("manifest has no video frames:", mpath); mpath = None
        except Exception:
            log("manifest unreadable:", mpath, traceback.format_exc()); mpath = None
    if not mpath:
        import glob
        cands = sorted(glob.glob(os.path.join(root, "runtime/autonomous_controller/runs/*/frames/sequence_000000/manifest.json")),
                       key=os.path.getmtime, reverse=True)
        log("fallback manifests:", cands[:3])
        if cands:
            mpath = os.path.relpath(cands[0], root)
    instr = rb.meta.get("instruction", "")
    q = I.parse(instr)
    log("instruction:", instr, "->", q)
    if not mpath:
        return q, None, [], {}
    _, frames = V.load_manifest(mpath, root)
    events, vid = V.parse_events(frames)
    byi = {f["frame_index"]: f for f in frames}
    held = {}
    for e in events:
        if e["type"] in ("release", "grasp"):
            try:
                c, frac = V.held_color(byi, max(0, e["frame_index"] - 2), root)
            except Exception:
                c = None
            held[e["frame_index"]] = c
            e["color"] = c
    # --- target identities through the video (targets swap places)
    seqf = vid + [f for f in frames if f.get("phase") != "video_demo"][-1:]
    slots, swaps = None, []
    try:
        if len(TR.detect_frame(seqf[-1], root)) < 2:
            TR.TARGET_COLOR = "white"
            TR.EXCLUDE_XY = [e["pos"][:2] for e in events if e["type"] == "press"]
            log("few pink targets in the live frame -> white-ring fallback, excluding", TR.EXCLUDE_XY)
        dets = [TR.detect_frame(f, root) for f in seqf]
        counts = [len(d) for d in dets]
        K = int(np.bincount(counts).argmax())
        slots, swaps = SW.swap_events(dets, K, [f["frame_index"] for f in seqf], log=log)
        log("K targets", K, "slots", np.round(slots, 3).tolist())
        for e in swaps:
            log("swap frames %d-%d pair %s" % (e["frame_start"], e["frame_end"], e["pair"]))
    except Exception:
        log("target tracking failed:", traceback.format_exc())
    for e in events:
        e["slot"] = None
        if e["type"] == "release" and slots is not None:
            d = np.linalg.norm(slots - np.array(e["pos"][:2]), axis=1)
            k = int(np.argmin(d))
            e["slot_dist"] = float(d[k])
            if d[k] < ON_TARGET_M:
                e["slot"] = k
    for e in events:
        log("event", e["type"], e["frame_index"], round(e["t"], 2), [round(x, 3) for x in e["pos"]],
            e.get("color"), "slot", e.get("slot"), round(e.get("slot_dist", -1), 3))
    on_target = [e for e in events if e["type"] != "release" or e.get("slot") is not None or slots is None]
    goal, why = I.select(on_target, q, held)
    log("selected:", None if goal is None else (goal["frame_index"], [round(x, 3) for x in goal["pos"]]), why)
    if goal is not None and slots is not None and goal.get("slot") is not None:
        s1 = SW.follow(goal["slot"], goal["i"], swaps)
        goal = dict(goal)
        goal["final_slot"] = s1
        goal["goal_xy"] = slots[s1].tolist()
        log("identity: slot %d %s at release -> slot %d %s now" % (
            goal["slot"], np.round(slots[goal["slot"]], 3).tolist(), s1, np.round(slots[s1], 3).tolist()))
    elif goal is not None:
        goal = dict(goal); goal["goal_xy"] = goal["pos"][:2]
        log("WARNING: no slot identity; using release position")
    return q, goal, events, held


def main(ctx):
    global LOG
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    LOG = open(os.path.join(OUT, "vpb_log.txt"), "a")
    log("==", VERSION, "start")
    rb = Robot(ctx)
    try:
        rb.observe()
    except Exception:
        log("observe failed", traceback.format_exc())
        return
    try:
        run(rb, t0)
    except Exception:
        log("FATAL in run:", traceback.format_exc())
        # keep the episode safe: open gripper hold
        try:
            if not rb.ended:
                rb.observe()
        except Exception:
            pass
    log("== end", "complete=%s ended=%s remaining=%s t=%.1f" % (rb.complete, rb.ended, rb.remaining, time.time() - t0))
    try:
        LOG.close()
    except Exception:
        pass


def run(rb, t0):
    q, goal, events, held = plan_from_video(rb)
    color = q["color"]
    tg, cubes = perceive_front(rb, color)
    log("front targets:", [(t["xy"].round(3).tolist(), t["n"]) for t in tg])
    log("front %s cubes:" % color, [(c["xy"].round(3).tolist(), round(c["ztop"], 3), c["n"]) for c in cubes])
    save_frame(rb, "start")

    # ---- goal xy
    if goal is not None:
        gxy = np.array(goal["goal_xy"][:2])
        near = [t for t in tg if np.linalg.norm(t["xy"] - gxy) < TARGET_SNAP_M]
        if near:
            t = min(near, key=lambda t: np.linalg.norm(t["xy"] - gxy))
            log("goal snap: video", gxy.round(3), "target", t["xy"].round(3))
            gxy = 0.5 * (gxy + t["xy"])
    elif tg:
        gxy = tg[0]["xy"]; log("WARNING no video goal; using largest target")
    else:
        log("no goal at all"); return
    log("GOAL xy", gxy.round(4))

    # ---- cube prior: last release of this colour in video
    prior = None
    rels = [e for e in events if e["type"] == "release" and e.get("color") in (color, None)]
    if rels:
        prior = np.array(rels[-1]["pos"][:2])
    prior_yaw = None
    if rels:
        prior_yaw = wrap_quarter(yaw_of_down_gripper(rels[-1]["rot"]))

    attempts = 0
    while attempts < 3 and not rb.ended and (rb.remaining or 0) > 30:
        attempts += 1
        # S2 locate cube
        tg, cubes = perceive_front(rb, color)
        cand = None
        if cubes:
            if prior is not None:
                cand = min(cubes, key=lambda c: np.linalg.norm(c["xy"] - prior))
                if np.linalg.norm(cand["xy"] - prior) > 0.08 and attempts == 1:
                    log("front cube far from video prior", cand["xy"].round(3), prior.round(3))
            else:
                cand = cubes[0]
            cxy = cand["xy"]
        elif prior is not None:
            cxy = prior
        else:
            log("cube not found"); hold(rb, -1, 5); continue
        log("attempt", attempts, "cube front xy", np.round(cxy, 3))
        # S3 hover + wrist refine
        s = rb.state
        yaw = prior_yaw if prior_yaw is not None else wrap_quarter(yaw_of_down_gripper(s[3:6]))
        if s[2] < SAFE_Z - 0.02:
            move_to(rb, [s[0], s[1], SAFE_Z], yaw, -1, tol=0.02, max_steps=30)
        r = move_to(rb, [cxy[0], cxy[1], HOVER_Z + 0.05], yaw, -1, tol=0.01, max_steps=60)
        log("hover1", r)
        w = None
        for it in range(3):
            w = perceive_wrist(rb, color, cxy)
            if w is None:
                break
            log("wrist", it, {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v) for k, v in w.items()})
            cxy = w["xy"]
            if w["yaw"] is not None and w["ext"] is not None and min(w["ext"]) > 0.025:
                yaw = w["yaw"]
            r = move_to(rb, [cxy[0], cxy[1], HOVER_Z], yaw, -1, tol=0.004, max_steps=40)
            if it >= 1 and r["ok"]:
                break
        save_frame(rb, "hover%d" % attempts)
        # S4 descend + grasp
        r = move_to(rb, [cxy[0], cxy[1], GRASP_Z + 0.03], yaw, -1, tol=0.005, max_steps=40, vmax=0.4)
        r = move_to(rb, [cxy[0], cxy[1], GRASP_Z], yaw, -1, tol=0.004, max_steps=30, vmax=0.2)
        log("descend", r, rb.state.round(4).tolist())
        hold(rb, 1, 8)
        s = rb.state
        r = move_to(rb, [s[0], s[1], SAFE_Z], yaw, 1, tol=0.01, max_steps=40, vmax=0.4)
        wsum = float(rb.state[6] + rb.state[7])
        log("lift", r, "finger sum", round(wsum, 4))
        save_frame(rb, "lift%d" % attempts)
        if wsum < HOLD_MIN_W:
            log("grasp failed; reopen and retry")
            hold(rb, -1, 6)
            prior = cxy
            continue
        # S5 carry + place
        r = move_to(rb, [gxy[0], gxy[1], SAFE_Z], yaw, 1, tol=0.006, max_steps=80)
        r = move_to(rb, [gxy[0], gxy[1], SAFE_Z], yaw, 1, tol=0.003, max_steps=20, vmax=0.2)
        log("over goal", r, rb.state.round(4).tolist())
        r = move_to(rb, [gxy[0], gxy[1], PLACE_Z + 0.03], yaw, 1, tol=0.004, max_steps=40, vmax=0.4)
        r = move_to(rb, [gxy[0], gxy[1], PLACE_Z], yaw, 1, tol=0.003, max_steps=30, vmax=0.15)
        log("lowered", r, rb.state.round(4).tolist())
        hold(rb, -1, 8)
        s = rb.state
        r = move_to(rb, [s[0], s[1], SAFE_Z], yaw, -1, tol=0.01, max_steps=40, vmax=0.4)
        log("released", r, "complete", rb.complete, "ended", rb.ended)
        save_frame(rb, "placed%d" % attempts)
        if rb.ended or rb.complete:
            return
        # S6 verify
        tg, cubes = perceive_front(rb, color)
        if cubes:
            c = min(cubes, key=lambda c: np.linalg.norm(c["xy"] - gxy))
            d = float(np.linalg.norm(c["xy"] - gxy))
            log("verify cube xy", c["xy"].round(4), "dist to goal", round(d, 4))
            if d < 0.02:
                break
            prior = c["xy"]
        else:
            log("verify: cube not seen from front")
            break
    # park: stay clear, open gripper
    if not rb.ended:
        s = rb.state
        move_to(rb, [s[0], s[1], 0.25], wrap_quarter(yaw_of_down_gripper(s[3:6])), -1, tol=0.02, max_steps=20)
        hold(rb, -1, 5)
