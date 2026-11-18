"""StopCube controller v1.

Task: "press the button to stop the cube just as it reaches the target for the N-th time".

One loop, one action row per call. Every step: detect the cube in the front camera, update the
arrival counter (excursions of cube-target planar distance below ARRIVE_R). The arm phase machine:

  APPROACH  -> P-control to (button_xy, cap_top+REFINE_DZ) exit: within 1.2 cm  -> REFINE ; timeout 60 steps -> REFINE
  REFINE    -> wrist-camera cap centre/top (accept if <3 cm from front estimate) -> DESCEND
  DESCEND   -> P-control to (cap_xy, cap_top + HOVER_DZ)    exit: within 4 mm    -> HOVER  ; timeout 40 -> HOVER
  HOVER     -> hold pose
  any of DESCEND/HOVER (or APPROACH if already low over the cap), when count == N-1 and the predicted
  closest approach is <= lead steps ahead (or the cube is already inside):  -> PRESS
  PRESS     -> dz=-1 rows until episode ends, or bottomed (<= 7 rows)  -> RETREAT (press did not end episode)
  RETREAT   -> back up to hover; continue counting; a later arrival triggers another press (max 3 presses)

Evidence for constants: Ep1 (exploration, task_complete True at step 253), see memory/PROGRESS.md.
"""
import json
import os
import re
import time
import traceback

import numpy as np

from vision import (parse_reply, Camera, detect_cube, detect_cube_geom, detect_target,
                    detect_button_front, detect_cap_wrist, detect_target_geom)

VERSION = "stopcube-v1.2"

# ---- tunables (tested range / evidence) ----
ARRIVE_R = 0.025     # m; Ep1 crossings reached <=0.008, return branch min 0.064
HOVER_DZ = 0.012     # TCP above cap top; Ep1 0.0117 -> fingertip pads ~5 mm above cap
REFINE_DZ = 0.07     # TCP above cap top for wrist refine; wrist cap estimate stable for TCP z 0.10-0.14 (Ep1+R1 frames)
APPROACH_CLIP = 1.0  # Rehearsal1: 0.8 -> ~0.016 m/step realized; hover reached at step 42
DESCEND_CLIP = 0.5
PRESS_MAX_ROWS = 7
MAX_PRESSES = 3
HOLD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]   # gripper closed throughout

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
            "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
            "once": 1, "twice": 2, "thrice": 3}
NUMWORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10}


def parse_n(instruction):
    s = instruction.lower()
    m = re.search(r"(\d+)\s*(st|nd|rd|th)\b", s)
    if m:
        return int(m.group(1))
    for w, n in ORDINALS.items():
        if re.search(r"\b%s\b" % w, s):
            return n
    m = re.search(r"\b(\d+|%s)\s+times?\b" % "|".join(NUMWORDS), s)
    if m:
        g = m.group(1)
        return int(g) if g.isdigit() else NUMWORDS[g]
    return None


class ArrivalCounter:
    def __init__(self, target_xy, count=0):
        self.tg = np.asarray(target_xy, float)
        self.count = count
        self.inside = False
        self.min_d = 9.0
        self.hist = []
        self.missing = 0

    def update(self, step, xy):
        """xy None -> dead-reckon up to 3 steps. Returns (d, event)."""
        if xy is None:
            self.missing += 1
            if self.missing > 3 or len(self.hist) < 2:
                return None, None
            (s1, p1), (s2, p2) = self.hist[-2:]
            v = (p2 - p1) / max(1, s2 - s1)
            xy = p2 + v * (step - s2)
        else:
            self.missing = 0
            self.hist.append((step, np.asarray(xy, float)))
            self.hist = self.hist[-6:]
        d = float(np.hypot(*(np.asarray(xy) - self.tg)))
        ev = None
        if d < ARRIVE_R:
            if not self.inside:
                self.inside, self.min_d = True, d
            self.min_d = min(self.min_d, d)
        elif self.inside:
            self.inside = False
            self.count += 1
            ev = "arrival %d (min %.3f)" % (self.count, self.min_d)
        return d, ev

    def predict(self):
        if len(self.hist) < 3 or self.missing:
            return None
        (s0, p0), _, (s2, p2) = self.hist[-3:]
        if s2 - s0 <= 0:
            return None
        v = (p2 - p0) / (s2 - s0)
        sp = float(np.linalg.norm(v))
        if sp < 1e-4:
            return None
        rel = self.tg - p2
        ta = float(rel @ v) / sp ** 2
        return ta, float(np.linalg.norm(rel - ta * v)), sp


class Logger:
    def __init__(self):
        self.f = None
        try:
            os.makedirs("/workspace/output", exist_ok=True)
            self.f = open("/workspace/output/stopcube_log.txt", "a")
        except Exception:
            self.f = None

    def __call__(self, msg):
        line = "[%s %.1f] %s" % (VERSION, time.time(), msg)
        print(line, flush=True)
        if self.f:
            try:
                self.f.write(line + "\n"); self.f.flush()
            except Exception:
                pass


def p_row(ee, goal, clip):
    d = np.clip((np.asarray(goal) - np.asarray(ee)) / 0.05, -clip, clip)
    return [float(d[0]), float(d[1]), float(d[2]), 0.0, 0.0, 0.0, 1.0]


def find_cube(meta, imgs, button_xy):
    """Geometry first (colour varies between episodes: beige Ep1, pink Ep2); colour+height fallback."""
    cam = Camera(meta, "front")
    rgb, dep = imgs[("front", "rgb")], imgs[("front", "depth")]
    g = detect_cube_geom(rgb, dep, cam, button_xy)
    if g is not None:
        return g["world"], "geom"
    c = detect_cube(rgb, dep, cam)
    if c is not None and c["world"][2] > 0.02:
        return c["world"], "color"
    return None, "none"


def press_lead(gap):
    """Steps from first press row to registration, plus half a step. Ep1: gap 0 -> registered in row 1.
    Step response: dz=-1 rows move ~1.2 cm then ~1.9 cm per row."""
    g = max(0.0, gap)
    rows = 1 if g <= 0.007 else 1 + int(np.ceil((g - 0.007) / 0.019))
    return rows + 0.5


def main(ctx):
    log = Logger()
    t_start = time.time()
    meta, imgs = None, None
    for attempt in range(3):
        try:
            meta, imgs = parse_reply(ctx.call("robomme_observe"))
            break
        except Exception:
            log("observe failed: " + traceback.format_exc())
            time.sleep(1.0)
    if meta is None:
        return
    instr = meta.get("instruction", "")
    N = parse_n(instr)
    log("instruction=%r N=%s step=%s limits=%s" % (instr, N, meta.get("step_id"), meta.get("episode_limits")))
    if N is None:
        N = 1
        log("WARN could not parse ordinal; defaulting N=1")

    cam = Camera(meta, "front")
    rgb, dep = imgs[("front", "rgb")], imgs[("front", "depth")]
    tg = detect_target(rgb, dep, cam)
    target_xy = tg["world"][:2] if tg else None
    btn = detect_button_front(rgb, dep, cam, target_xy)
    log("target=%s button=%s" % (tg, btn))
    if target_xy is None:
        # fallback: assume target is the flat disk nearest the cube's path cannot be known; use cube start
        c, how = find_cube(meta, imgs, btn["xy"] if btn else None)
        tg2 = detect_target_geom(rgb, dep, cam, btn["xy"] if btn else None, c[:2] if c else None)
        target_xy = tg2["xy"] if tg2 else (c[:2] if c else [0.0, 0.0])
        log("WARN target colour not found; geometric fallback %s -> %s" % (tg2, target_xy))
    if btn is None:
        btn = {"xy": [-0.145, 0.26], "cap_top": 0.034}
        log("WARN button not found; using fallback")
    bxy = np.array(btn["xy"], float)
    cap_top = float(btn["cap_top"])
    if not (0.015 < cap_top < 0.06):
        cap_top = 0.034

    ac = ArrivalCounter(target_xy)
    pressed_counts = set()
    phase, phase_steps, presses = "APPROACH", 0, 0
    press_rows = 0
    refined = False
    ended = False
    last_ee = None
    stall = 0

    while not ended:
        try:
            if time.time() - t_start > 1100:
                log("wall budget nearly exhausted; stopping")
                return
            step = meta["step_id"]
            ee = np.array(meta["state"][:3], float)
            cxyz, how = find_cube(meta, imgs, bxy.tolist())
            d, ev = ac.update(step, cxyz[:2] if cxyz else None)
            pr = ac.predict()
            hover = np.array([bxy[0], bxy[1], cap_top + HOVER_DZ])
            xy_err = float(np.hypot(*(ee[:2] - bxy)))
            gap = float(ee[2] - hover[2])
            lead = press_lead(gap)
            log(json.dumps({"step": step, "phase": phase, "ee": [round(x, 4) for x in ee],
                            "cube": cxyz and [round(x, 4) for x in cxyz], "how": how,
                            "d": d and round(d, 4), "count": ac.count, "inside": ac.inside,
                            "pred": pr and [round(x, 3) for x in pr], "lead": round(lead, 2), "ev": ev}))

            # ---- press trigger ----
            want_press = False
            if ac.count >= N - 1 and ac.count not in pressed_counts and phase not in ("PRESS", "RETREAT") \
                    and presses < MAX_PRESSES:
                if ac.inside:
                    want_press = True
                elif pr is not None and 0 < pr[0] <= lead and pr[1] < ARRIVE_R:
                    want_press = True
            if want_press:
                if xy_err < 0.015 and gap < 0.08:
                    log("PRESS trigger step=%d ta=%s gap=%.4f xy_err=%.4f" % (step, pr and round(pr[0], 2), gap, xy_err))
                    phase, phase_steps, press_rows = "PRESS", 0, 0
                    presses += 1
                    pressed_counts.add(ac.count)
                else:
                    log("press wanted but arm not over cap (xy_err=%.3f gap=%.3f)" % (xy_err, gap))

            # ---- phase actions ----
            if phase == "APPROACH":
                goal = np.array([bxy[0], bxy[1], cap_top + REFINE_DZ])
                row = p_row(ee, goal, APPROACH_CLIP)
                if np.linalg.norm(goal - ee) < 0.012 or phase_steps > 60:
                    phase, phase_steps = "REFINE", 0
            elif phase == "REFINE":
                wc = detect_cap_wrist(imgs[("wrist", "rgb")], imgs[("wrist", "depth")], Camera(meta, "wrist"),
                                      bxy.tolist(), cap_top)
                log("wrist refine: %s" % wc)
                if wc and np.hypot(*(np.array(wc["xy"]) - bxy)) < 0.03 and abs(wc["cap_top"] - cap_top) < 0.01:
                    bxy = np.array(wc["xy"], float)
                    cap_top = float(wc["cap_top"])
                    refined = True
                row = HOLD
                phase, phase_steps = "DESCEND", 0
            elif phase == "DESCEND":
                row = p_row(ee, hover, DESCEND_CLIP)
                if np.linalg.norm(hover - ee) < 0.004 or phase_steps > 40:
                    phase, phase_steps = "HOVER", 0
            elif phase == "HOVER":
                row = p_row(ee, hover, 0.2)
            elif phase == "PRESS":
                if press_rows < PRESS_MAX_ROWS and ee[2] > cap_top - 0.010:
                    row = [float(np.clip((bxy[0] - ee[0]) / 0.05, -0.2, 0.2)),
                           float(np.clip((bxy[1] - ee[1]) / 0.05, -0.2, 0.2)), -1.0, 0.0, 0.0, 0.0, 1.0]
                    press_rows += 1
                else:
                    log("press finished without episode end (rows=%d z=%.4f); retreating" % (press_rows, ee[2]))
                    phase, phase_steps = "RETREAT", 0
                    row = p_row(ee, hover + np.array([0, 0, 0.02]), 0.6)
            elif phase == "RETREAT":
                row = p_row(ee, hover, 0.4)
                if ee[2] > hover[2] - 0.002 or phase_steps > 15:
                    phase, phase_steps = "HOVER", 0
            else:
                row = HOLD
            phase_steps += 1

            rep = ctx.call("robomme_step", actions=[row])
            meta, imgs = parse_reply(rep)
            if meta.get("episode_ended") or (meta.get("transition") or {}).get("episode_ended"):
                ended = True
                log("EPISODE ENDED at step %s task_complete=%s phase=%s count=%d presses=%d" % (
                    meta.get("step_id"), meta.get("task_complete"), phase, ac.count, presses))
                break
            rem = (meta.get("episode_limits") or {}).get("remaining_episode_steps")
            if rem is not None and rem <= 0:
                log("no steps remaining")
                break
        except Exception:
            log("ERROR in loop: " + traceback.format_exc())
            try:
                rep = ctx.call("robomme_observe")
                meta, imgs = parse_reply(rep)
                if meta.get("episode_ended"):
                    log("episode ended (seen after error)")
                    break
            except Exception:
                log("observe after error failed: " + traceback.format_exc())
                time.sleep(0.5)
            stall += 1
            if stall > 20:
                log("too many errors; stopping")
                return
    log("main done; wall=%.1fs" % (time.time() - t_start))
