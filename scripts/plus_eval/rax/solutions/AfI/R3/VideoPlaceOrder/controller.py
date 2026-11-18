"""VideoPlaceOrder controller v1.

Pipeline (each step logs one line; never raises out of main):
  PARSE   : read conditioning-video manifest -> gripper events from video robot state
            -> placements of the instructed cube (held colour classified from wrist frames)
            -> N-th placement xy (ordinal from instruction) -> snapped to a detected target ring.
  PICK    : locate cube (front RGB-D) -> hover -> wrist refine (xy, yaw) -> descend -> close -> lift
            -> verify finger width (holding). retry <= 3 (re-perceive each time).
  PLACE   : hover over target -> descend -> open -> lift.
  VERIFY  : front RGB-D cube centroid within 3 cm of target xy; else back to PICK (<= 2 more cycles).
"""
import glob, json, math, os, time, traceback
from pathlib import Path
import numpy as np

import perception as P
import swaps as W
from motion import Robot, down_R, meta

VERSION = "vpo-v3"
HOVER_Z = 0.13
LIFT_Z = 0.15
PLACE_Z = 0.035          # TCP z at release (video released at ~0.02)
GRASP_DZ = 0.018         # TCP below measured cube top
SNAP_RADIUS = 0.06
OUT = Path("/workspace/output")


def log(*a):
    print(f"[{VERSION}]", *a, flush=True)


def find_manifest(first_meta):
    fs = (first_meta or {}).get("frame_sequence") or {}
    p = fs.get("manifest_path")
    if p and (P.WS / p).exists():
        return P.WS / p
    cands = glob.glob(str(P.WS / "runtime/autonomous_controller/runs/*/frames/sequence_000000/manifest.json"))
    cands.sort(key=os.path.getmtime)
    return Path(cands[-1]) if cands else None


def plan_from_video(manifest_path, instruction):
    """Return (info, placements, sel, chosen). chosen['goal_xy'] is where the instructed ring
    sits at the end of the video (identity tracked through the ring swaps)."""
    info = P.parse_instruction(instruction)
    m = json.loads(Path(manifest_path).read_text())
    frames = sorted(m["frames"], key=lambda f: f["frame_index"])
    vid, ev = P.video_events(frames)
    try:
        sw = W.analyse(frames)
    except Exception:
        log("swap analysis exception", traceback.format_exc()); sw = None
    timeline = final = None
    if sw is not None:
        log("slots", np.round(sw["slots"], 3).tolist())
        for e in sw["events"]:
            log("ring motion", e)
        for r in sw.get("rejected", []):
            log("rejected run", r)
        timeline, final = W.occupant_timeline(sw, len(frames))
        log("final occupants", final)
    # map frame_index -> position in `frames`
    pos_of = {f["frame_index"]: i for i, f in enumerate(frames)}
    placements = []
    cur_color = None
    for e in ev:
        if e["kind"] == "grasp":
            cur_color, votes = P.held_color(vid, e["i"])
            log("video grasp", e["frame_index"], np.round(e["xyz"], 3).tolist(), cur_color, votes)
        elif e["kind"] == "release":
            rec = {"xyz": e["xyz"], "color": cur_color, "frame_index": e["frame_index"], "slot": None,
                   "ident": None, "goal_xy": None}
            if sw is not None:
                s = W.nearest_slot(sw["slots"], e["xyz"])
                rec["slot"] = s
                if s is not None:
                    k = pos_of.get(e["frame_index"], 0)
                    ident = timeline[k][s]
                    rec["ident"] = ident
                    rec["goal_xy"] = sw["slots"][final.index(ident)].tolist()
            log("video release", e["frame_index"], np.round(e["xyz"], 3).tolist(), cur_color,
                "slot", rec["slot"], "ident", rec["ident"], "goal", rec["goal_xy"])
            placements.append(rec)
    col = info["color"]
    on_target = [p for p in placements if p["slot"] is not None] if sw is not None else placements
    if not on_target:
        on_target = placements
    sel = [p for p in on_target if col is None or p["color"] in (col, None)]
    if not sel:
        sel = on_target
    k = info["ordinal"]
    chosen = None
    if sel:
        if info["last"] and k is None:
            chosen = sel[-1]
        elif k is not None and 1 <= k <= len(sel):
            chosen = sel[k - 1]
        else:
            chosen = sel[min(len(sel), max(1, k or len(sel))) - 1]
    idents = [p["ident"] for p in sel]
    if len(set(idents)) != len(idents):
        log("WARNING: repeated ring identities among placements", idents)
    return info, placements, sel, chosen


class Ctl:
    def __init__(self, ctx):
        self.ctx = ctx
        self.rb = Robot(ctx, log)

    def obs(self):
        r = self.ctx.call("robomme_observe")
        o = P.decode_obs(r)
        self.rb._update(o["meta"])
        return o

    def cube_front(self, color):
        o = self.obs()
        c, pts = P.locate_color(o, "front", color)
        return c, pts

    def pick(self, color, attempt):
        rb = self.rb
        c, pts = self.cube_front(color)
        if c is None:
            # move up and out of the way, look again
            rb.move_to([-0.05, 0.0, 0.35], -1, down_R(0), tol=0.01, max_steps=60)
            c, pts = self.cube_front(color)
            if c is None:
                return {"ok": False, "why": "cube not seen in front"}
        log("pick", attempt, "front cube", np.round(c, 3).tolist())
        yaw = 0.0
        rb.move_to([c[0], c[1], HOVER_Z], -1, down_R(yaw), tol=0.005, max_steps=80)
        # wrist refine
        o = self.obs()
        cw, ptw = P.locate_color(o, "wrist", color, min_px=30)
        if cw is not None and np.linalg.norm(cw[:2] - c[:2]) < 0.06:
            yaw = P.top_yaw(ptw)
            top = ptw[:, 2].max()
            xy = ptw[ptw[:, 2] > top - 0.006][:, :2].mean(0)
            log("pick wrist refine", np.round(xy, 3).tolist(), "top", round(float(top), 3), "yaw", round(math.degrees(yaw), 1))
        else:
            xy = c[:2]; top = float(pts[:, 2].max()) if pts is not None else 0.04
            log("pick wrist refine failed; using front")
        gz = float(np.clip(top - GRASP_DZ, 0.012, 0.05))
        rb.move_to([xy[0], xy[1], HOVER_Z], -1, down_R(yaw), tol=0.004, max_steps=40)
        rb.move_to([xy[0], xy[1], gz + 0.04], -1, down_R(yaw), tol=0.005, max_steps=40)
        rb.move_to([xy[0], xy[1], gz], -1, down_R(yaw), tol=0.004, max_steps=40, vmax=0.5)
        rb.hold(12, 1.0)
        rb.move_to([xy[0], xy[1], LIFT_Z], 1.0, down_R(yaw), tol=0.01, max_steps=60)
        rb.hold(3, 1.0)
        w = float(rb.state[6] + rb.state[7])
        ok = 0.012 < w < 0.072
        log("pick result width", round(w, 4), "ok", ok)
        return {"ok": ok, "why": "" if ok else f"width {w:.3f}", "yaw": yaw}

    def place(self, target, yaw):
        rb = self.rb
        rb.move_to([target[0], target[1], LIFT_Z], 1.0, down_R(yaw), tol=0.005, max_steps=100)
        rb.move_to([target[0], target[1], PLACE_Z + 0.03], 1.0, down_R(yaw), tol=0.005, max_steps=40)
        rb.move_to([target[0], target[1], PLACE_Z], 1.0, down_R(yaw), tol=0.004, max_steps=40, vmax=0.4)
        rb.hold(10, -1.0)
        rb.move_to([target[0], target[1], LIFT_Z], -1.0, down_R(yaw), tol=0.01, max_steps=40)
        return {"ok": True}

    def verify(self, color, target):
        rb = self.rb
        # move arm away so the front camera sees the cube
        rb.move_to([-0.05, 0.0, 0.35], -1, down_R(0), tol=0.02, max_steps=40)
        c, pts = self.cube_front(color)
        if c is None:
            return {"ok": False, "why": "cube not visible", "d": None}
        d = float(np.linalg.norm(c[:2] - np.asarray(target[:2])))
        log("verify cube", np.round(c, 3).tolist(), "dist", round(d, 3))
        return {"ok": d < 0.03, "d": d, "c": c}

    def run(self):
        o = self.obs()
        m0 = o["meta"]
        instr = m0.get("instruction", "")
        log("instruction:", instr, "step", m0.get("step_id"))
        mp = find_manifest(m0)
        log("manifest", mp)
        info, placements, sel, chosen = plan_from_video(mp, instr)
        log("parsed", info, "n_place", len(placements), "n_sel", len(sel))
        color = info["color"] or "red"
        if chosen is None:
            log("no placement parsed; abort to hold")
            return
        if chosen.get("goal_xy") is not None:
            tgt = np.array(list(chosen["goal_xy"][:2]) + [0.008], float)
        else:
            tgt = np.array(chosen["xyz"][:3], float)
        targets = P.locate_targets(o, "front")
        log("targets seen", [np.round(t, 3).tolist() for t in targets])
        if targets:
            dists = [np.linalg.norm(t[:2] - tgt[:2]) for t in targets]
            j = int(np.argmin(dists))
            if dists[j] < SNAP_RADIUS:
                tgt = np.array([targets[j][0], targets[j][1], targets[j][2]])
                log("snapped target", np.round(tgt, 3).tolist(), "d", round(float(dists[j]), 3))
        log("goal target xy", np.round(tgt[:2], 3).tolist(), "from video frame", chosen["frame_index"])
        for cycle in range(3):
            if self.rb.ended:
                break
            res = None
            for attempt in range(3):
                if self.rb.ended:
                    break
                try:
                    res = self.pick(color, attempt)
                except Exception:
                    log("pick exception", traceback.format_exc()); res = {"ok": False}
                if res.get("ok"):
                    break
                self.rb.hold(5, -1.0)
                self.rb.move_to(list(self.rb.state[:2]) + [LIFT_Z], -1, down_R(0), tol=0.01, max_steps=30)
            if not res or not res.get("ok"):
                continue
            self.place(tgt, res.get("yaw", 0.0))
            if self.rb.ended:
                break
            v = self.verify(color, tgt)
            log("cycle", cycle, "verify", v.get("ok"), v.get("d"), "complete", self.rb.complete, "step", self.rb.step_id)
            if v.get("ok") or self.rb.complete:
                break
        log("done; step", self.rb.step_id, "ended", self.rb.ended, "complete", self.rb.complete)


def main(ctx):
    t0 = time.time()
    try:
        Ctl(ctx).run()
    except Exception:
        log("FATAL", traceback.format_exc())
    log("wall", round(time.time() - t0, 1))
