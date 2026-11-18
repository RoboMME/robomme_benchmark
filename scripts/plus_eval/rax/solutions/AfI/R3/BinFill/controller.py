"""BinFill controller v2 (v1 + neighbour-aware grasp yaw, stacked-cube grasp height,
blocked-descent abort, wrist-refine abort, longer wait for cube waves).

Task: "put N1 <color1> cubes and N2 <color2> cubes into the bin, then press the button".
Cubes appear on the table in waves during the first ~150 steps; cubes in the bin become
invisible (black cover), so the controller counts its own successful deposits (memory).

Pipeline / transitions:
  INIT     : parse instruction -> needed counts; survey (front) -> bin + button.   -> SELECT
  SELECT   : detect needed-colour cubes (front+wrist merge), choose one.
             found -> PICK ; none -> SURVEY (high pose, wait) -> SELECT (timeout -> PRESS)
  PICK     : hover, wrist refine, descend, close, lift; held? -> PLACE ; else -> SELECT
  PLACE    : hover bin, wrist refine bin centre, check still held, release; count++ -> SELECT
  PRESS    : when all counts satisfied: hover button, wrist refine, descend closed, lift. -> DONE
"""
import re
import sys
import time
import traceback
import numpy as np

from motion import Robot, DOWN
from skills import (detect_cubes, detect_bin, detect_button, wrist_bin_refine, servo_pick,
                    estimate_velocity)

VERSION = "binfill-v3"
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "a": 1, "an": 1}
COLORS = ("red", "blue", "green")
SURVEY_POSE = np.array([-0.06, 0.0, 0.45])   # near reset pose; full wrist view of table
HOVER_Z = 0.12
BIN_RELEASE_Z = 0.13      # tested ep1: release at 0.127-0.128 OK
PRESS_Z = 0.012           # closed-gripper TCP target over button cap (cap top ~0.034)
STEP_RESERVE_PRESS = 80   # steps kept for pressing the button


def log(*a):
    print(f"[{VERSION} {time.strftime('%H:%M:%S')}]", *a, flush=True)


def parse_counts(instr):
    counts = {}
    s = instr.lower()
    for m in re.finditer(r"\b(\w+)\s+(red|blue|green)\s+cubes?\b", s):
        w = m.group(1)
        n = WORDS.get(w)
        if n is None and w.isdigit():
            n = int(w)
        if n is not None:
            counts[m.group(2)] = counts.get(m.group(2), 0) + n
    return counts


def merge_dets(front, wrist, tol=0.03):
    """Merge per-camera detections; prefer wrist xyz (top view) when matched."""
    out = [dict(d, src="wrist") for d in wrist if d["n"] > 40 and d["zmax"] > 0.025]
    for d in front:
        if d["zmax"] < 0.025:
            continue
        if any(o["color"] == d["color"] and np.linalg.norm(o["xyz"][:2] - d["xyz"][:2]) < tol for o in out):
            continue
        out.append(dict(d, src="front"))
    return out


class BinFill:
    def __init__(self, ctx, done=None):
        self.R = Robot(ctx, log=log)
        self.need = {}
        self.done = dict(done or {})
        self.bin_xy = None
        self.bin_refined = False
        self.button = None
        self.fail_spots = []   # (xy, count) where picks failed
        self.v = np.zeros(2)   # estimated scene (conveyor) velocity, m/step
        self.v_known = False
        self.t0 = time.time()

    # ---------------- perception
    def scene(self):
        o = self.R.observe()
        fr = detect_cubes(o, "front")
        wr = detect_cubes(o, "wrist")
        dets = [d for d in merge_dets(fr, wr) if d["zmax"] < 0.09]
        if self.bin_xy is not None:
            dets = [d for d in dets if np.abs(d["xyz"][:2] - self.bin_xy).max() > 0.042]
        return o, dets

    def scene_with_velocity(self, wait=10):
        """Two observations `wait` steps apart -> detections (latest) and velocity estimate."""
        o1, d1 = self.scene()
        s1 = self.R.meta["step_id"]
        self.R.step([[0, 0, 0, 0, 0, 0, self.R.grip]] * wait)
        o2, d2 = self.scene()
        s2 = self.R.meta["step_id"]
        v, n = estimate_velocity(d1, d2, s2 - s1, max_match=0.03 + 0.007 * (s2 - s1))
        if v is not None:
            if np.linalg.norm(v) < 0.0006:
                v = np.zeros(2)
            self.v = v if not self.v_known else 0.5 * self.v + 0.5 * v
            self.v_known = True
        log(f"  velocity est {None if v is None else np.round(v*1000,2)} (n={n}) -> v={np.round(self.v*1000,2)} mm/step")
        return o2, d2

    def remaining_need(self):
        return {c: self.need[c] - self.done.get(c, 0) for c in self.need if self.need[c] - self.done.get(c, 0) > 0}

    def init(self):
        o = self.R.observe()
        instr = o["meta"].get("instruction", "")
        self.need = parse_counts(instr)
        log("instruction:", instr, "need:", self.need)
        b = detect_bin(o, "front")
        if b is not None:
            # front view sees far part of the interior; shift toward camera (+x) by ~5 mm
            self.bin_xy = b["xyz"][:2] + np.array([0.005, 0.0])
        btn = detect_button(o, "front", self.bin_xy)
        if btn is not None:
            self.button = btn["xyz"].copy()   # cap top (front cap estimate matched wrist within 1 mm in ep1)
        log("bin", None if self.bin_xy is None else self.bin_xy.round(3),
            "button", None if self.button is None else self.button.round(3))

    def survey(self):
        R = self.R
        R.move_to(SURVEY_POSE, grip=-1, rot=DOWN, tol=0.01, max_steps=60)
        o = R.observe()
        if self.bin_xy is None:
            b = detect_bin(o, "front") or detect_bin(o, "wrist")
            if b is not None:
                self.bin_xy = b["xyz"][:2]
        if self.button is None:
            btn = detect_button(o, "front", self.bin_xy)
            if btn is not None:
                self.button = btn["xyz"].copy()
            else:
                btn = detect_button(o, "wrist", self.bin_xy)
                if btn is not None:
                    self.button = np.r_[btn["base_xy"], 0.034]
            log("survey button:", None if self.button is None else self.button.round(3))

    # ---------------- choosing
    def choose(self, dets):
        need = self.remaining_need()
        cands = [d for d in dets if d["color"] in need]
        if not cands:
            return None
        pos = self.R.pos[:2]
        v = self.v
        vn = np.linalg.norm(v)

        def score(d):
            xy0 = d["xyz"][:2]
            t_reach = np.linalg.norm(xy0 - pos) / 0.015 + 15
            xy = xy0 + v * t_reach
            s = np.linalg.norm(xy - pos)
            if vn > 0.0008:
                # moving scene: avoid cubes that will be carried far downstream during the pick
                far = xy0 + v * (t_reach + 40)
                if abs(far[1]) > 0.26 or far[0] > 0.2 or far[0] < -0.36:
                    s += 0.4
            # neighbours make grasps risky
            nb = min([np.linalg.norm(xy - o["xyz"][:2]) for o in dets if o is not d] + [1.0])
            if nb < 0.055:
                s += 0.15
            if vn <= 0.0008:
                for fxy, k in self.fail_spots:
                    if np.linalg.norm(xy - fxy) < 0.03:
                        s += 0.2 * k
            # out of comfortable reach
            if xy[0] < -0.33 or xy[0] > 0.2 or abs(xy[1]) > 0.3:
                s += 0.5
            return s
        return min(cands, key=score)

    # ---------------- place
    def place(self, color):
        R = self.R
        if self.bin_xy is None:
            self.survey()
        if self.bin_xy is None:
            log("  no bin known; cannot place")
            return False
        tgt = np.r_[self.bin_xy, BIN_RELEASE_Z]
        R.move_to(tgt, rot=None, tol=0.005, max_steps=80)
        if not self.bin_refined:
            o = R.observe()
            w = wrist_bin_refine(o, self.bin_xy)
            if w is not None and np.linalg.norm(w["xy"] - self.bin_xy) < 0.03:
                ext = w["hi"] - w["lo"]
                log(f"  bin refine {self.bin_xy.round(3)} -> {w['xy'].round(3)} ext {ext.round(3)}")
                self.bin_xy = w["xy"]
                self.bin_refined = True
                R.move_to(np.r_[self.bin_xy, BIN_RELEASE_Z], rot=None, tol=0.004, max_steps=40)
        gap = R.state[6:8].sum()
        if gap < 0.025:
            log(f"  lost cube before release (gap {gap:.4f})")
            return False
        R.gripper(-1, n=10)
        log(f"  released {color} at {R.pos.round(3)}")
        return True

    # ---------------- press
    def press(self):
        R = self.R
        if self.button is None:
            self.survey()
        if self.button is None:
            log("no button found")
            return False
        bx = self.button.copy()
        R.move_to([bx[0], bx[1], 0.15], grip=-1, rot=DOWN, tol=0.006, max_steps=80)
        o = R.observe()
        w = detect_button(o, "wrist", self.bin_xy)
        if w is not None and np.linalg.norm(w["xyz"][:2] - bx[:2]) < 0.04 and w["n"] > 100:
            log(f"  button refine {bx[:2].round(3)} -> {w['xyz'][:2].round(3)} (n={w['n']})")
            bx[:2] = w["xyz"][:2]
        R.gripper(1, n=8)
        R.move_to([bx[0], bx[1], 0.07], rot=DOWN, tol=0.005, max_steps=40)
        # slow descent with stall detection
        last = R.pos[2]; stall = 0
        for k in range(60):
            if R.ended or R.complete:
                break
            if R.pos[2] <= PRESS_Z + 0.002:
                break
            R.step([[ (bx[0]-R.pos[0])*2/0.05, (bx[1]-R.pos[1])*2/0.05, -0.25, 0, 0, 0, 1]])
            if last - R.pos[2] < 0.0008:
                stall += 1
                if stall >= 3:
                    break
            else:
                stall = 0
            last = R.pos[2]
        log(f"  press bottom z={R.pos[2]:.4f} complete={R.complete} ended={R.ended}")
        R.step([[0, 0, 0, 0, 0, 0, 1]] * 3)
        if not R.ended:
            R.move_to([bx[0], bx[1], 0.12], rot=DOWN, tol=0.01, max_steps=30)
        return True

    # ---------------- one select-pick-place cycle
    def cycle(self):
        R = self.R
        need = self.remaining_need()
        o, dets = self.scene_with_velocity()
        d = self.choose(dets)
        if d is None:
            self.idle += 1
            log(f"no needed cube visible (need {need}); survey #{self.idle}")
            if self.idle > 60:
                return "stop"
            self.survey()
            o, dets = self.scene()
            d = self.choose(dets)
            if d is None:
                R.step([[0, 0, 0, 0, 0, 0, -1]] * 10)
                return "idle"
        self.idle = 0
        log(f"step {R.meta['step_id']} target {d['color']} {d['xyz'].round(3)} src {d['src']} "
            f"yaw {np.degrees(d['yaw']):.0f} need {need}")
        held, info = servo_pick(R, d, self.v, log=log)
        if not held:
            if info.get("why") in ("refine",):
                self.fail_spots.append((d["xyz"][:2].copy(), 0.5))
            else:
                self.fail_spots.append((d["xyz"][:2].copy(), 1))
            if not R.ended and R.state[6:8].sum() < 0.075:
                R.gripper(-1, n=6)
            return "fail"
        ok = self.place(d["color"])
        if ok:
            self.done[d["color"]] = self.done.get(d["color"], 0) + 1
            log(f"  deposited; done={self.done} step={R.meta['step_id']} t={time.time()-self.t0:.0f}s")
        return "ok"

    # ---------------- main loop
    def run(self):
        R = self.R
        try:
            self.init()
        except Exception:
            log("init error", traceback.format_exc())
        self.idle = 0
        errors = 0
        while not R.ended and not R.complete:
            need = self.remaining_need()
            if not need:
                break
            if R.remaining() < STEP_RESERVE_PRESS:
                log("step budget low; going to press")
                break
            if time.time() - self.t0 > 1050:
                log("wall budget low; going to press")
                break
            if errors > 15:
                log("too many errors; going to press")
                break
            try:
                r = self.cycle()
                if r == "stop":
                    break
            except Exception:
                errors += 1
                log("cycle error", traceback.format_exc())
                if R.ended:
                    break
                try:
                    R.gripper(-1, n=4) if R.state[6:8].sum() < 0.025 else None
                except Exception:
                    pass
        if not R.ended and not R.complete:
            log(f"pressing button; done={self.done} need={self.need}")
            try:
                self.press()
            except RuntimeError as e:
                log("press error", e)
        log(f"END step={R.meta.get('step_id')} complete={R.complete} ended={R.ended} done={self.done}")


def main(ctx, done=None):
    try:
        BinFill(ctx, done).run()
    except Exception:
        log("FATAL", traceback.format_exc())
