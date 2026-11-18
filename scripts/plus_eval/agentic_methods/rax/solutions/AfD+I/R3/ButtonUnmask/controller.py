"""ButtonUnmask controller (v2).

Phases (transition table):
  INIT    observe step 0, read instruction, locate button on the empty table (wrist)  -> VIEW
  VIEW    move to the view pose (wrist sees whole table), watch until PRESS_START_STEP -> PRESS
  PRESS   closed gripper descends onto the button cap, back to view pose (done < ~70)  -> WATCH
          (deadline guard: press must register before step 90)
  WATCH   hold at view pose, accumulate coloured-cube tracks (front + wrist, world xy)
          until step >= WATCH_MIN_STEP and no cube seen for WATCH_QUIET_STEPS
          (and all target colours seen) or WATCH_MAX_STEP                            -> PLAN
  PLAN    containers seen from view pose; each target track -> nearest container       -> PICK
  PICK(i) hover, refine xy/yaw from wrist, grasp, lift; non-last: place back, release
          grasp miss (fingers closed) -> reopen, re-perceive, retry (max 2)          -> next / DONE
  DONE    stop when task_complete / episode ended / steps exhausted.
"""
import math
import os
import re
import time
import traceback

import numpy as np

import perception as P
from robot import Robot, log, down_rot, current_yaw

VERSION = "bu-v3"

# ---- tunables (evidence: ep1 = live episode 1, demos = 3 released recordings)
VIEW_POS = (-0.08, 0.0, 0.35)    # wrist covers x[-0.25,0.15] y[-0.25,0.25] at z=0.07 (projection, ep1 calib)
PRESS_START_STEP = 26            # waves at 12-43, 56-87, 100-131 (ep1 + 3 demos); press ~40 steps
PRESS_DONE_BY = 80               # plan the round trip to finish by here (deadline: press before 90)
BUTTON_PRESS_DEPTH = 0.016       # ep1: commanded top-0.016, TCP stalled at top-0.0075 -> registered
WATCH_MIN_STEP = 136
WATCH_QUIET_STEPS = 16           # gap between waves is 12 steps
WATCH_MAX_STEP = 330
HOVER_Z = 0.20
CONTAINER_TOP = 0.072            # ep1: all container tops 0.0722
GRASP_BELOW_TOP = 0.025          # ep1: TCP 0.047 -> finger sum 0.0555, 3/3 grasps
GRIP_CLOSE_STEPS = 8
GRIP_OPEN_STEPS = 6
GRIP_EMPTY = 0.012               # finger sum below this = nothing held
LIFT_Z = 0.20                    # ep1: success fired at TCP z~0.145 during the last lift
MATCH_RADIUS = 0.045
REFINE_RADIUS = 0.025            # refine at hover must stay near the planned container (spacing >= 5.5 cm)
REFINE_RADIUS_NOCONT = 0.04

COLOR_WORDS = P.COLORS

def parse_targets(instruction):
    """Ordered list of (colour, all_flag) from the instruction."""
    text = instruction.lower()
    out = []
    # split at 'then' / 'followed by' / 'finally' clauses, keep order of colour mentions after 'button'
    idx = text.find("button")
    rest = text[idx + 6:] if idx >= 0 else text
    for m in re.finditer(r"(all containers hiding|container hiding|one hiding|ones hiding|containers hiding)\s+(?:the\s+|a\s+|an\s+)?(\w+)\s+cubes?", rest):
        word = m.group(2)
        allflag = m.group(1).startswith("all") or m.group(0).endswith("cubes")
        if word in COLOR_WORDS:
            out.append((word, allflag))
    if not out:
        for w in re.findall(r"\b(" + "|".join(COLOR_WORDS) + r")\b", rest):
            out.append((w, False))
    return out


class Controller:
    def __init__(self, ctx, save_dir=None):
        self.mem = P.CubeMemory()
        self.save_dir = save_dir
        self.robot = Robot(ctx, on_obs=self._on_obs)
        self.watching = True
        self.button = None
        self.view_pos = np.array(VIEW_POS, float)
        self.targets = []
        self.t0 = time.time()
        self.pressed = False

    def _on_obs(self, o):
        if self.watching:
            dets = P.detect_cubes(o, min_px=8)
            self.mem.add(o, dets)
            if dets:
                log(f"step {o.step} dets", [(d['color'], d['cam'][0], [round(float(x), 3) for x in d['xyz'][:2]], d['px']) for d in dets])
        if self.save_dir:
            try:
                np.savez_compressed(os.path.join(self.save_dir, f"obs_{o.step:04d}_{self.robot.calls:04d}.npz"),
                                    front=o.rgb.get("front"), wrist=o.rgb.get("wrist"),
                                    front_d=o.depth.get("front"), wrist_d=o.depth.get("wrist"), state=o.state)
            except Exception as e:
                log("save err", e)

    def phase(self, name):
        o = self.robot.obs
        log(f"PHASE {name} step {None if o is None else o.step} wall {time.time() - self.t0:.1f}s")

    # ------------------------------------------------------------ phases
    def init(self):
        self.phase("INIT")
        o = self.robot.observe()
        self.instruction = o.meta.get("instruction", "")
        self.targets = parse_targets(self.instruction)
        log(VERSION, "instruction:", self.instruction, "targets:", self.targets, "state", o.state.round(4).tolist())
        b = P.find_button(o, "wrist")
        if b is None:
            b = P.find_button(o, "front")
        self.button = b
        log("button", None if b is None else (b["xy"].round(4).tolist(), round(b["top"], 4), b["n"]))

    def view_until_press(self):
        self.phase("VIEW")
        r = self.robot
        r.move_to(self.view_pos, grip=1.0, tol=0.01, max_steps=20)
        start = PRESS_START_STEP
        if self.button is not None:
            bxyz = np.array([self.button["xy"][0], self.button["xy"][1], self.button["top"]])
            trip = 2 * float(np.linalg.norm(self.view_pos - bxyz)) / 0.016 + 8   # steps, measured speed
            start = int(max(0, min(PRESS_START_STEP, PRESS_DONE_BY - trip)))
            log(f"press trip estimate {trip:.0f} steps -> start at {start}")
        while not r.ended and r.obs.step < start:
            r.hold(2, grip=1.0, yaw=0.0)

    def find_button_late(self):
        """Fallback when the button was not found at reset: look again from the view pose."""
        for cam in ("wrist", "front"):
            b = P.find_button(self.robot.obs, cam)
            if b is not None:
                return b
        return None

    def press(self):
        self.phase("PRESS")
        r = self.robot
        b = self.button or self.find_button_late()
        if b is None:
            log("no button found; skipping press")
            return False
        bx, by = float(b["xy"][0]), float(b["xy"][1])
        top = float(b["top"])
        if not (0.015 < top < 0.06):
            top = 0.0335
        r.move_to([bx, by, top + 0.06], grip=1.0, tol=0.008, max_steps=30)
        r.move_to([bx, by, top - BUTTON_PRESS_DEPTH], grip=1.0, vmax=0.03, tol=0.003, max_steps=16, gain=2.0)
        o = r.hold(2, grip=1.0)
        self.pressed = o.pos[2] < top - 0.003
        log(f"press bottom step {o.step} tcp {o.pos.round(4).tolist()} cap_top {top:.4f} pressed_est {self.pressed}")
        r.move_to([bx, by, top + 0.06], grip=1.0, tol=0.01, max_steps=10)
        r.move_to(self.view_pos, grip=1.0, tol=0.01, max_steps=30)
        log(f"press done step {r.obs.step} at {r.obs.pos.round(3).tolist()}")
        return True

    def watch(self):
        self.phase("WATCH")
        r = self.robot
        need = {c for c, _ in self.targets}
        while not r.ended:
            o = r.hold(2, grip=1.0, yaw=0.0)
            step = o.step
            last_det_step = max([t["last"] for t in self.mem.tracks], default=0)
            found = {t["color"] for t in self.mem.good_tracks()}
            quiet = step - last_det_step
            if step >= WATCH_MIN_STEP and quiet >= WATCH_QUIET_STEPS and need <= found:
                break
            if step >= WATCH_MAX_STEP:
                break
        self.watching = False
        log("watch done step", r.obs.step, "tracks", self.mem.summary())

    # ------------------------------------------------------------ planning
    def plan(self):
        """Ordered pick list: each target colour -> remembered cube tracks -> containers in view."""
        self.phase("PLAN")
        conts = P.find_containers(self.robot.obs, "wrist")
        log("containers", [(c["xy"].round(3).tolist(), round(c["top"], 3), c["n"]) for c in conts])
        tracks = self.mem.good_tracks()

        def nearest(xy):
            best = None
            for j, c in enumerate(conts):
                d = float(np.linalg.norm(c["xy"] - xy))
                if best is None or d < best[0]:
                    best = (d, j)
            return best

        # every track -> container index (if close)
        assign = {}
        for t in tracks:
            nb = nearest(t["xy"])
            t["cont"] = nb[1] if nb is not None and nb[0] < MATCH_RADIUS else None
            if t["cont"] is not None:
                assign.setdefault(t["cont"], []).append(t)
        used = set()
        plan = []
        for color, allflag in self.targets:
            ts = [t for t in tracks if t["color"] == color]
            ts.sort(key=lambda t: -t["w"])
            if not allflag:
                ts = ts[:1]
            seen_conts = set()
            for t in ts:
                if t["cont"] is not None:
                    if t["cont"] in seen_conts or t["cont"] in used:
                        continue
                    seen_conts.add(t["cont"])
                    used.add(t["cont"])
                    plan.append(dict(color=color, xy=conts[t["cont"]]["xy"].copy(), mem_xy=np.array(t["xy"]),
                                     yaw=conts[t["cont"]]["yaw"], top=conts[t["cont"]]["top"]))
                elif allflag and t["n"] < 4:
                    continue   # weak extra sighting; don't pick an unknown container for it
                else:
                    plan.append(dict(color=color, xy=np.array(t["xy"], float), mem_xy=np.array(t["xy"]),
                                     yaw=None, top=None))
            if not ts:
                # never saw this colour: guess a container that no seen cube explains
                free = [j for j in range(len(conts)) if j not in assign and j not in used]
                log("no track for", color, "unexplained containers", free)
                if free:
                    j = free[0]
                    used.add(j)
                    plan.append(dict(color=color, xy=conts[j]["xy"].copy(), mem_xy=None, yaw=conts[j]["yaw"],
                                     top=conts[j]["top"], guess=True))
        log("plan", [(p["color"], p["xy"].round(4).tolist()) for p in plan])
        return plan

    # ------------------------------------------------------------ picking
    def match_container(self, xy, cams=("wrist",), radius=0.04):
        best = None
        for cam in cams:
            for c in P.find_containers(self.robot.obs, cam):
                d = float(np.linalg.norm(c["xy"] - xy))
                if d < radius and (best is None or d < best[0]):
                    best = (d, c, cam)
        return best

    def pick(self, item, is_last):
        self.phase(f"PICK {item['color']}")
        r = self.robot
        xy = np.array(item["xy"], float)
        yaw = item.get("yaw") or 0.0
        top = item.get("top") or CONTAINER_TOP
        # v3: the plan xy (near-top-down view pose) is authoritative; no far/oblique re-match here
        # (rehearsal 1: an oblique match from the previous hover was 3.2 cm off).
        plan_xy = xy.copy()
        radius = REFINE_RADIUS if item.get("yaw") is not None else REFINE_RADIUS_NOCONT
        log(f"pick {item['color']} plan {np.round(item['xy'], 4).tolist()} yaw {math.degrees(yaw):.1f}")
        for attempt in range(3):
            r.move_to([xy[0], xy[1], HOVER_Z], yaw=yaw, grip=-1.0, tol=0.006, max_steps=60)
            # refine from the closer wrist view (keep yaw continuous: 90-degree symmetry)
            for _ in range(2):
                m2 = self.match_container(plan_xy, radius=radius)
                if m2 is None:
                    log("refine: no container near", plan_xy.round(4).tolist())
                    break
                c = m2[1]
                dy = (c["yaw"] - yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
                yaw = yaw + dy
                xy = c["xy"].copy()
                top = c["top"]
                log(f"refine -> {xy.round(4).tolist()} yaw {math.degrees(yaw):.1f} top {top:.4f} n {c['n']} size {np.round(c['size'], 3).tolist()}")
                r.move_to([xy[0], xy[1], HOVER_Z], yaw=yaw, grip=-1.0, tol=0.004, max_steps=30)
            top = min(max(top, 0.05), 0.09)
            gz = top - GRASP_BELOW_TOP
            r.move_to([xy[0], xy[1], gz + 0.05], yaw=yaw, grip=-1.0, tol=0.005, max_steps=40)
            ok, n = r.move_to([xy[0], xy[1], gz], yaw=yaw, grip=-1.0, vmax=0.02, tol=0.004, max_steps=30)
            log(f"at grasp height ok={ok} tcp {r.obs.pos.round(4).tolist()} steps {n}")
            o = r.hold(GRIP_CLOSE_STEPS, grip=1.0, yaw=yaw)
            log(f"closed: finger sum {o.grip:.4f} attempt {attempt}")
            if r.ended:
                return False
            if o.grip > GRIP_EMPTY:
                break
            # missed: open, back up, re-perceive and retry with a slightly different yaw
            r.hold(GRIP_OPEN_STEPS, grip=-1.0, yaw=yaw)
            r.move_to([xy[0], xy[1], HOVER_Z], yaw=yaw, grip=-1.0, tol=0.01, max_steps=30)
            yaw = yaw + math.radians(15)
        ok, n = r.move_to([xy[0], xy[1], LIFT_Z], yaw=yaw, grip=1.0, vmax=0.03, tol=0.01, max_steps=40)
        o = r.hold(4, grip=1.0, yaw=yaw)
        log(f"lifted: tcp {o.pos.round(4).tolist()} finger {o.grip:.4f} complete={r.complete} ended={r.ended} step {o.step}")
        held = o.grip > GRIP_EMPTY
        if r.ended or is_last:
            return held
        # put it back where it was and release
        r.move_to([xy[0], xy[1], gz + 0.004], yaw=yaw, grip=1.0, vmax=0.03, tol=0.005, max_steps=40)
        r.hold(GRIP_OPEN_STEPS, grip=-1.0, yaw=yaw)
        r.move_to([xy[0], xy[1], HOVER_Z], yaw=yaw, grip=-1.0, tol=0.01, max_steps=40)
        return held


def main(ctx):
    c = Controller(ctx)
    steps = [("init", c.init), ("view", c.view_until_press), ("press", c.press), ("watch", c.watch)]
    for name, fn in steps:
        if c.robot.ended:
            break
        try:
            fn()
        except Exception:
            log("error in", name, traceback.format_exc())
            if name == "init":
                try:
                    c.robot.observe()
                    c.targets = parse_targets(c.robot.obs.meta.get("instruction", ""))
                except Exception:
                    log("re-observe failed", traceback.format_exc())
    c.watching = False
    if c.robot.ended:
        log("ended before picking; complete", c.robot.complete)
        return
    try:
        plan = c.plan()
    except Exception:
        log("plan error", traceback.format_exc())
        plan = []
    for i, item in enumerate(plan):
        if c.robot.ended:
            break
        try:
            c.pick(item, is_last=(i == len(plan) - 1))
        except Exception:
            log("pick error", traceback.format_exc())
            try:
                c.robot.observe()
                c.robot.move_to([c.robot.obs.pos[0], c.robot.obs.pos[1], HOVER_Z], grip=-1.0, tol=0.02, max_steps=20)
            except Exception:
                log("recovery error", traceback.format_exc())
    # hold the last lift a little if success has not fired yet
    k = 0
    while not c.robot.ended and not c.robot.complete and k < 5:
        try:
            c.robot.hold(4, grip=1.0)
        except Exception:
            break
        k += 1
    try:
        import json
        os.makedirs("/workspace/output", exist_ok=True)
        with open("/workspace/output/summary.json", "w") as f:
            json.dump(dict(version=VERSION, targets=c.targets, tracks=c.mem.summary(), complete=c.robot.complete,
                           step=c.robot.obs.step if c.robot.obs else None, pressed_est=c.pressed), f, default=str)
    except Exception:
        log("summary write failed", traceback.format_exc())
    log("FINAL complete", c.robot.complete, "ended", c.robot.ended, "step", c.robot.obs.step if c.robot.obs else None,
        "wall", round(time.time() - c.t0, 1))
