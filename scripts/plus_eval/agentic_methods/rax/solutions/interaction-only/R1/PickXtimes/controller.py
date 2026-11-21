"""PickXtimes controller (RAX, interaction-only, R1) — version v1.

Written without any live observation of the task (the single simulator start is the
rehearsal of this bundle), so every phase re-perceives the scene and logs what it saw.

Transition table
----------------
INIT      -> parse instruction, calibrate cameras, perceive scene      -> PLAN
PLAN      -> ordered task list from the instruction (picks, button)     -> PICK / PRESS
PICK(k)   -> hover, wrist refine, align yaw, descend, close, lift
             ok -> PLACE(k); miss -> retry PICK(k) (<= MAX_PICK_TRIES, yaw +45deg on rung 2)
PLACE(k)  -> carry over target (or original spot), lower, open, lift
             -> verify cube position -> PICK(k+1) or PRESS / DONE
PRESS     -> hover over button, wrist refine, press down, lift
             task_complete -> DONE; else next candidate / deeper press (<= MAX_PRESS_TRIES)
DONE      -> return (also on episode_ended, wall-time or step exhaustion)
"""
import math
import os
import re
import time
import traceback

import numpy as np

import percept as P
from geom import mat_to_rotvec, rot_z, rotvec_to_mat, wrap_quarter
from mcpio import Logger, Reply, max_steps, remaining_steps, summarize

VERSION = "pickx-v1"

# ---- tunables (geometric choices, untested live) --------------------------
POS_SCALE = 0.05          # m per unit action (documented)
ROT_SCALE = 0.5           # rad per unit action (documented)
V_FREE = 0.022            # m per step in free space
V_NEAR = 0.010            # m per step near contact
V_PRESS = 0.006           # m per step while pressing
CHUNK_FREE = 8
CHUNK_NEAR = 5
TOL_FREE = 0.008
TOL_NEAR = 0.004
HOVER_ABOVE_TOP = 0.10    # m TCP above object top while hovering
CARRY_ABOVE_TABLE = 0.14  # m TCP above table while carrying
PLACE_GAP = 0.008         # m, cube bottom above support when released
MAX_PICK_TRIES = 4
MAX_PRESS_TRIES = 4
WALL_LIMIT_S = 1080.0     # controller limit is 1200 s
CUBE_CONFIDENT = 2.6      # cube score above which no exploratory sweep is needed
CUBE_ACCEPT = 2.0         # min score to replace the current cube estimate (colour must match)
SWEEP_HEIGHT = 0.33       # m TCP above table for exploratory wrist views
SWEEP_OFFSETS = [(0.0, 0.0), (0.12, 0.12), (0.12, -0.12), (-0.08, -0.15), (-0.08, 0.15), (0.2, 0.0)]
DEFAULT_CUBE = 0.04

NUM_WORDS = {"once": 1, "twice": 2, "thrice": 3, "one": 1, "two": 2, "three": 3, "four": 4,
             "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
             "single": 1, "double": 2, "triple": 3}
COLOR_WORDS = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "cyan", "white",
               "black", "gray", "grey", "brown", "magenta", "violet", "teal"]


# --------------------------------------------------------------------------
def parse_instruction(text):
    t = (text or "").lower()
    out = {"color": None, "count": None, "target": False, "button": False,
           "button_first": False, "raw": text}
    m = re.search(r"\b(" + "|".join(COLOR_WORDS) + r")\s+(?:\w+\s+)?(cube|block|box|object)", t)
    if m:
        out["color"] = m.group(1)
    else:
        for c in COLOR_WORDS:
            if re.search(r"\b" + c + r"\b", t):
                out["color"] = c
                break
    if out["color"] in P.COLOR_ALIASES:
        out["color"] = P.COLOR_ALIASES[out["color"]]
    cnt = None
    more = False
    m = re.search(r"\b(\d+)\s*(?:more\s+|additional\s+)?(?:times|time|x\b|rounds?|repetitions?)", t)
    if m:
        cnt = int(m.group(1))
        more = bool(re.search(r"\b\d+\s*(more|additional)\b", t))
    else:
        for w, n in NUM_WORDS.items():
            if re.search(r"\b" + w + r"\b(\s+(more|additional))?\s*(times?|rounds?|repetitions?)?", t) and \
                    (w in ("once", "twice", "thrice") or re.search(r"\b" + w + r"\s+(more\s+|additional\s+)?(times?|rounds?|repetitions?)", t)):
                cnt = n
                more = bool(re.search(r"\b" + w + r"\s+(more|additional)\b", t))
                break
    if cnt is None:
        m = re.search(r"\b(\d+)\b", t)
        if m and 0 < int(m.group(1)) <= 10:
            cnt = int(m.group(1))
    if cnt is not None and more:
        cnt += 1
    out["count"] = cnt if cnt is not None else 1
    out["count_parsed"] = cnt is not None
    out["target"] = bool(re.search(r"\b(target|goal|plate|marker|region|area|zone)\b", t))
    out["button"] = "button" in t
    if out["button"]:
        ib = t.find("button")
        ips = [i for i in (t.find("pick"), t.find("lift"), t.find("grasp")) if i >= 0]
        ip = min(ips) if ips else 10 ** 6
        # button first only for explicit "press the button first / then pick" phrasing
        between = t[ib:ip] if ib < ip else ""
        out["button_first"] = bool(ib < ip and re.search(r"\b(then|first|before)\b", between)
                                   and not re.search(r"\b(after|once)\b", t[max(0, ib - 40):ib]))
    return out


# --------------------------------------------------------------------------
class Stop(Exception):
    pass


class Robot:
    def __init__(self, ctx, log):
        self.ctx = ctx
        self.log = log
        self.state = None
        self.ended = False
        self.complete = False
        self.step_id = 0
        self.limits = None
        self.remaining = None
        self.max_steps = None
        self.steps_sent = 0
        self.ncalls = 0
        self.call_time = 0.0
        self.last_obs = None

    def call(self, tool, **kw):
        t0 = time.time()
        raw = self.ctx.call(tool, **kw)
        self.call_time += time.time() - t0
        self.ncalls += 1
        r = Reply(raw)
        self._update(r)
        return r

    def _update(self, r):
        s = r.state()
        if s is not None:
            self.state = s
        if r.ended():
            self.ended = True
        if r.complete():
            self.complete = True
        sid = r.step_id()
        if sid is not None:
            self.step_id = sid
        lim = r.limits()
        if lim:
            self.limits = lim
            rem = remaining_steps(lim)
            if rem is not None:
                self.remaining = rem
            ms = max_steps(lim)
            if ms is not None:
                self.max_steps = ms

    def observe(self):
        r = self.call("robomme_observe")
        self.last_obs = r
        return r

    def steps_left(self):
        if self.remaining is not None:
            return self.remaining
        if self.max_steps is not None:
            return self.max_steps - self.step_id
        return 10 ** 6

    def step(self, rows):
        if self.ended or self.complete:
            raise Stop("episode over")
        left = self.steps_left()
        if left <= 0:
            self.ended = True
            raise Stop("no steps left")
        rows = rows[:max(1, min(100, left))]
        clean = []
        for r in rows:
            rr = [float(np.clip(x, -1.0, 1.0)) if np.isfinite(x) else 0.0 for x in r]
            clean.append(rr)
        before = self.step_id
        try:
            r = self.call("robomme_step", actions=clean)
        except RuntimeError as e:
            self.log("STEP ERROR", repr(e)[:400])
            # do not retry blindly: resync from a fresh observation
            try:
                self.observe()
            except Exception as e2:
                self.log("observe after step error failed", repr(e2)[:300])
            if self.ended:
                raise Stop("ended after error")
            return None
        self.steps_sent += len(clean)
        if self.steps_sent == len(clean):
            self.log("first step reply meta", str(summarize(r.meta))[:2500])
        fb = r.first("feedback")
        if fb:
            self.log("feedback", str(fb)[:300])
        if self.step_id == before and r.step_id() is None:
            self.step_id = before + len(clean)
        if r.state() is None:
            self.observe()
        if self.remaining is not None and r.limits() is None:
            self.remaining -= len(clean)
        if self.ended or self.complete:
            self.log("episode flag after step: ended=%s complete=%s step=%s" % (self.ended, self.complete, self.step_id))
        return r

    # ---- pose helpers ----------------------------------------------------
    def pos(self):
        return self.state[:3].copy()

    def rot(self):
        return rotvec_to_mat(self.state[3:6])

    def width(self):
        return float(self.state[6] + self.state[7])


# --------------------------------------------------------------------------
class Controller:
    def __init__(self, ctx):
        self.log = Logger()
        self.rb = Robot(ctx, self.log)
        self.t0 = time.time()
        self.variant = {}
        self.tool_ok = True
        self.swap_cams = False
        self.swap_tested = False
        self.speed = 1.0
        self.start_xy = None
        self.first_pick_step = None
        self.calib_err = {}
        self.table_z = None
        self.R0 = None
        self.yaw_cum = 0.0
        self.task = None
        self.cube_size = DEFAULT_CUBE
        self.cube = None
        self.target = None
        self.button = None
        self.home_xy = None
        self.button_xy = None
        self.cube_expect = None
        self.cube_size_fixed = None
        self.picks_done = 0
        self.rot_body = False
        self.yaw_bias = 0.0
        self.rot_flip_votes = 0
        self.img_count = 0
        self.phase_log = []

    # ---------------------------------------------------------------- utils
    def check_time(self):
        if time.time() - self.t0 > WALL_LIMIT_S:
            raise Stop("wall-time guard")

    def phase(self, name):
        self.phase_log.append((name, self.rb.step_id, round(time.time() - self.t0, 1)))
        self.log("PHASE", name, "step", self.rb.step_id, "left", self.rb.steps_left(),
                 "t=%.1f" % (time.time() - self.t0), "calls", self.rb.ncalls)

    def save_img(self, frames, tag):
        if self.img_count > 40:
            return
        try:
            from PIL import Image
            for cam, fr in frames.items():
                Image.fromarray(fr["rgb"]).save("/workspace/output/%02d_%s_%s.jpg" % (self.img_count, tag, cam), quality=85)
            self.img_count += 1
        except Exception:
            pass

    # ------------------------------------------------------------- motion
    def move_to(self, p_t, R_t=None, grip=-1.0, v=V_FREE, tol=TOL_FREE, chunk=CHUNK_FREE,
                max_calls=30, watch=None):
        """Closed-loop chunked motion to p_t with orientation R_t."""
        rb = self.rb
        if R_t is None:
            R_t = self.R_hold
        p_t = np.asarray(p_t, float)
        v = v * self.speed
        stall = 0
        last_d = None
        for it in range(max_calls):
            self.check_time()
            p = rb.pos()
            e = p_t - p
            d = float(np.linalg.norm(e))
            re_ = mat_to_rotvec(R_t @ rb.rot().T)
            dr = float(np.linalg.norm(re_))
            if d < tol and dr < 0.04:
                return True
            if last_d is not None and last_d - d < 0.002:
                stall += 1
                if stall >= 4:
                    self.log("move stall d=%.4f dr=%.3f target=%s pos=%s" % (d, dr, np.round(p_t, 3), np.round(p, 3)))
                    return d < 3 * tol
            else:
                stall = 0
            last_d = d
            n = max(1, int(math.ceil(d / v)), int(math.ceil(dr / 0.12)))
            k = min(n, chunk)
            dp = e / n / POS_SCALE
            drot_w = re_ / n
            if self.rot_body:
                drot_w = rb.rot().T @ drot_w
            drot = drot_w / ROT_SCALE
            row = list(np.clip(dp, -1, 1)) + list(np.clip(drot, -1, 1)) + [grip]
            p_before = p.copy()
            rb.step([row] * k)
            # frame sanity: rotation error must shrink when we command a rotation
            if dr > 0.08:
                dr_after = float(np.linalg.norm(mat_to_rotvec(R_t @ rb.rot().T)))
                if dr_after > dr + 0.02:
                    self.rot_flip_votes += 1
                    self.log("rotation error grew %.3f -> %.3f (vote %d)" % (dr, dr_after, self.rot_flip_votes))
                    if self.rot_flip_votes >= 2:
                        self.rot_body = not self.rot_body
                        self.rot_flip_votes = 0
                        self.log("switching rotation-delta frame; body =", self.rot_body)
                elif dr_after < dr - 0.02:
                    self.rot_flip_votes = 0
            moved = rb.pos() - p_before
            if d > 0.03 and np.linalg.norm(moved) > 0.01 and float(np.dot(moved, e)) < 0:
                self.log("WARNING: motion opposite to command", np.round(moved, 3).tolist(), np.round(e, 3).tolist())
            if watch is not None and watch():
                return True
        p = rb.pos()
        return float(np.linalg.norm(p_t - p)) < 3 * tol

    def hold(self, n, grip):
        rows = []
        re_ = mat_to_rotvec(self.R_hold @ self.rb.rot().T)
        if np.linalg.norm(re_) > 0.15:
            re_ = re_ * (0.15 / np.linalg.norm(re_))
        if self.rot_body:
            re_ = self.rb.rot().T @ re_
        for i in range(n):
            rows.append([0.0, 0.0, 0.0] + list(np.clip(re_ / max(n, 1) / ROT_SCALE, -1, 1)) + [grip])
        self.rb.step(rows)

    def gripper(self, grip, max_steps=30):
        """Command the gripper while holding the pose; stop when the fingers have
        moved and then settled (a command that has not started yet is not 'settled')."""
        w0 = self.rb.width()
        last = w0
        total = 0
        while total < max_steps:
            self.hold(5, grip)
            total += 5
            w = self.rb.width()
            moved = abs(w - w0) > 0.004 or (grip > 0 and w < 0.01) or (grip < 0 and w > 0.075)
            if moved and abs(w - last) < 0.0008:
                break
            last = w
        return self.rb.width()

    # --------------------------------------------------------- perception
    def perceive(self, cams=("front", "wrist"), tag=None):
        rb = self.rb
        obs = rb.observe()
        frames = P.extract_frames(obs, self.log)
        if self.swap_cams and "front" in frames and "wrist" in frames:
            frames = {"front": frames["wrist"], "wrist": frames["front"]}
        cal = P.extract_calib(obs)
        sid = obs.step_id()
        if sid is None:
            sid = rb.step_id
        if tag:
            self.save_img(frames, tag)
        clouds = []
        for cam in cams:
            if cam not in frames or cam not in cal:
                if cam not in self.variant:
                    self.log("perceive: camera", cam, "missing frames=%s cal=%s" % (cam in frames, cam in cal))
                continue
            if cam not in self.variant:
                var, err = P.calibrate(rb.call, sid, cam, cal[cam], frames[cam], self.log)
                other = "wrist" if cam == "front" else "front"
                if var is None and err is not None and other in frames and not self.swap_tested:
                    # maybe the unlabeled attachments came in the other order
                    self.swap_tested = True
                    var2, err2 = P.calibrate(rb.call, sid, cam, cal[cam], frames[other], self.log)
                    if var2 is not None:
                        self.log("camera images were swapped; correcting")
                        self.swap_cams = True
                        frames = {"front": frames["wrist"], "wrist": frames["front"]}
                        var, err = var2, err2
                if var is None and err is None:
                    # tool unusable: self-calibrate on the table plane
                    pg = "cv" not in str(cal[cam].get("conv", ""))
                    var, fr1 = P.flatness_variant(cal[cam], frames[cam], self.log, prefer_gl=pg, return_score=True)
                    self.tool_ok = False
                    other = "wrist" if cam == "front" else "front"
                    if not self.swap_tested and other in frames:
                        self.swap_tested = True
                        var2, fr2 = P.flatness_variant(cal[cam], frames[other], self.log, prefer_gl=pg,
                                                       return_score=True)
                        if var2 is not None and fr2 > 1.5 * fr1 + 0.05:
                            self.log("camera images were swapped (flatness); correcting")
                            self.swap_cams = True
                            frames = {"front": frames["wrist"], "wrist": frames["front"]}
                            var = var2
                self.variant[cam] = var
                self.calib_err[cam] = err
            var = self.variant[cam]
            try:
                if var is not None:
                    xyz, rgb, us, vs = P.full_cloud(cal[cam], frames[cam], var)
                elif self.tool_ok:
                    xyz, rgb, us, vs = P.tool_cloud(rb.call, sid, cam, frames[cam])
                else:
                    continue
                clouds.append((cam, xyz, rgb))
            except Exception as e:
                self.log("cloud failed", cam, repr(e))
        if not clouds and frames:
            # calibration metadata missing entirely: tool grid on any camera
            for cam in cams:
                if cam in frames:
                    xyz, rgb, us, vs = P.tool_cloud(rb.call, sid, cam, frames[cam])
                    clouds.append((cam, xyz, rgb))
        if not clouds:
            return None
        if self.table_z is None:
            front = [c for c in clouds if c[0] == "front"] or clouds
            tz = P.table_height(front[0][1][:, 2], ref_z=rb.pos()[2])
            self.table_z = tz
            self.log("table_z", tz)
        if self.table_z is None:
            return None
        xyz = np.concatenate([c[1] for c in clouds])
        rgb = np.concatenate([c[2] for c in clouds])
        comps, flat = P.analyse(xyz, rgb, self.table_z, tcp=rb.pos(), log=self.log)
        return {"comps": comps, "flat": flat, "frames": frames, "cams": [c[0] for c in clouds]}

    def describe(self, comps):
        for i, c in enumerate(comps):
            self.log("  comp%d n=%d xy=(%.3f,%.3f) h=%.3f top=%.3fx%.3f foot=%.3fx%.3f yaw=%.0f color=%s hsv=%s big=%s" % (
                i, c["n"], c["center_xy"][0], c["center_xy"][1], c["height"], c["top_wl"][0], c["top_wl"][1],
                c["foot_wl"][0], c["foot_wl"][1], math.degrees(c["yaw"]), c["color"],
                [int(x) for x in c["hsv_med"]], c["is_big"]))

    def cube_score(self, c, color, expect=None):
        if c["is_big"]:
            return -9
        if self.button_xy is not None and np.linalg.norm(c["center_xy"] - self.button_xy) < 0.03:
            return -8  # the button never moves; never take it for the cube
        h = c.get("size", c["height"])
        w, l = sorted(c["top_wl"])
        if not (0.012 < h < 0.085) or l > 0.10 or w < 0.006:
            return -5
        sq = w / max(l, 1e-6)
        size_dev = min(1.0, abs(l - h) / max(h, 1e-3))
        cs = P.color_score(c["hsv_top"], color) if color else 0.5
        cs_all = P.color_score(c["hsv"], color) if color else 0.5
        fw, fl = sorted(c["foot_wl"])
        top_ratio = (w * l) / max(fw * fl, 1e-6)
        sc = 3.0 * cs + 1.0 * cs_all + 0.6 * sq - 0.6 * size_dev
        if top_ratio < 0.55:
            sc -= 2.5  # top much smaller than footprint: a button / stacked shape, not a cube
        if self.cube_size_fixed is not None:
            sc -= 1.0 * min(1.0, abs(h - self.cube_size_fixed) / 0.015)
        if expect is not None:
            sc -= 1.5 * min(1.0, np.linalg.norm(c["center_xy"] - expect) / 0.12)
        return sc

    def accept(self, best):
        if best is None:
            return False
        if self.task["color"]:
            return best[0] >= CUBE_ACCEPT
        return best[0] > -1

    def select_cube(self, comps, color, near=None, radius=None, expect=None):
        best = None
        if expect is None:
            expect = self.cube_expect
        for c in comps:
            if near is not None and np.linalg.norm(c["center_xy"] - near) > radius:
                continue
            s = self.cube_score(c, color, expect)
            if best is None or s > best[0]:
                best = (s, c)
        return best

    def select_button(self, comps, cube, near=None, radius=None):
        best = None
        for c in comps:
            if c["is_big"]:
                continue
            if cube is not None and np.linalg.norm(c["center_xy"] - cube["center_xy"]) < 0.03:
                continue
            if near is not None and np.linalg.norm(c["center_xy"] - near) > radius:
                continue
            if self.target is not None and c["height"] < 0.02 and \
                    np.linalg.norm(c["center_xy"] - self.target["center_xy"]) < self.target["radius"] + 0.01:
                continue
            h = c["height"]
            if h < 0.008 or h > 0.15:
                continue
            fw, fl = sorted(c["foot_wl"])
            if c["n"] < 20 or fl < 0.02:
                continue  # fragments (partly occluded edges) are not buttons
            tw, tl = sorted(c["top_wl"])
            top_ratio = (tw * tl) / max(fw * fl, 1e-6)
            cubeness = self.cube_score(c, self.task["color"]) if self.task["color"] else 0
            other_cube = max(P.color_score(c["hsv_top"], k) for k in ("red", "green", "blue", "yellow"))
            sq = tw / max(tl, 1e-6)
            is_cube_like = (0.6 < sq) and abs(tl - h) / max(h, 1e-3) < 0.45 and top_ratio > 0.6 and other_cube > 0.6
            s = 1.0 + (0.0 if is_cube_like else 2.0) + (1.0 - min(1.0, top_ratio)) - 0.3 * cubeness
            s += 0.002 * min(c["n"], 300)
            if h < 0.014 and max(fw, fl) > 0.06:
                s -= 2.0  # wide and flat: more likely a target disk than a button
            if best is None or s > best[0]:
                best = (s, c)
        return best

    # ----------------------------------------------------------- phases
    def init(self):
        rb = self.rb
        self.log(VERSION, "start")
        try:
            st = rb.call("robomme_status")
            self.log("status meta", str(summarize(st.meta))[:1500])
        except Exception as e:
            self.log("robomme_status failed", repr(e)[:300])
        obs = rb.observe()
        self.log("observe meta", str(summarize(obs.meta))[:2500])
        self.log("image blocks", [(b["img"].mode if b["img"] else None, b["img"].size if b["img"] else None,
                                   str(b["info"])[:80], b["pre"][:60]) for b in obs.images])
        ins = obs.instruction()
        self.task = parse_instruction(ins)
        self.log("instruction:", repr(ins), "->", {k: v for k, v in self.task.items() if k != "raw"})
        # conditioning video (if any)
        ms = obs.first("manifest_path", pred=lambda v: isinstance(v, str))
        if ms:
            try:
                import json
                man = json.load(open(ms if os.path.isabs(ms) else os.path.join("/workspace", ms)))
                fr = man.get("frames", [])
                nvid = sum(1 for f in fr if f.get("phase") == "video_demo")
                self.log("initial manifest frames", len(fr), "video_demo", nvid)
            except Exception as e:
                self.log("manifest read failed", repr(e)[:200])
        if rb.state is None:
            raise Stop("no robot state in observation")
        self.R0 = rb.rot()
        self.R_hold = self.R0.copy()
        self.log("initial state", np.round(rb.state, 4).tolist(), "tcp z-axis", np.round(self.R0[:, 2], 3).tolist(),
                 "steps left", rb.steps_left(), "max", rb.max_steps)

    def scan(self, tag="scan"):
        sc = self.perceive(tag=tag)
        if sc is None:
            return None
        self.log("scan", tag, "cams", sc["cams"], "table_z %.4f" % self.table_z, "ncomps", len(sc["comps"]))
        self.describe(sc["comps"])
        return sc

    def sweep(self):
        """Exploratory wrist-camera views over the workspace; returns merged comps."""
        rb = self.rb
        c0 = self.start_xy
        z = self.table_z + SWEEP_HEIGHT
        best = None
        for dx, dy in SWEEP_OFFSETS:
            self.check_time()
            self.move_to([c0[0] + dx, c0[1] + dy, z], grip=-1.0, v=V_FREE * 1.3)
            sc = self.scan("sweep")
            if not sc:
                continue
            b = self.select_cube(sc["comps"], self.task["color"])
            if b is not None and (best is None or b[0] > best[0][0]):
                best = (b, sc)
            if b is not None and b[0] >= CUBE_CONFIDENT:
                break
        return best

    def locate_all(self):
        """Initial scene: cube, target, button."""
        self.start_xy = self.rb.pos()[:2].copy()
        swept = False
        for attempt in range(3):
            sc = self.scan("init%d" % attempt)
            if sc is not None and self.table_z is not None and not swept:
                b0 = self.select_cube(sc["comps"], self.task["color"])
                if b0 is None or b0[0] < CUBE_CONFIDENT:
                    self.log("no confident cube (best %s); sweeping" % (None if b0 is None else round(b0[0], 2)))
                    swept = True
                    res = self.sweep()
                    if res is not None and (b0 is None or res[0][0] > b0[0]):
                        # rescan from the sweep pose that saw the cube best, keep its scene
                        sc = res[1]
            if sc is None:
                self.log("scan failed; moving up for a clearer view")
                p = self.rb.pos()
                self.move_to(p + np.array([0, 0, 0.05]))
                continue
            best = self.select_cube(sc["comps"], self.task["color"])
            if best is not None and best[0] > -1:
                self.cube = best[1]
                self.log("cube selected score %.2f xy=%s h=%.3f color=%s" % (best[0], np.round(self.cube["center_xy"], 3),
                                                                           self.cube["height"], self.cube["color"]))
                self.cube_size = float(np.clip(self.cube.get("size", self.cube["height"]), 0.02, 0.07))
                self.home_xy = self.cube["center_xy"].copy()
                self.cube_expect = self.home_xy.copy()
            if self.task["target"] or True:
                tg = P.find_target(sc["flat"], prefer="red", log=self.log)
                if tg is None:
                    # thick target disk shows up as a flat component
                    flats = [c for c in sc["comps"] if c["height"] < 0.015 and not c["is_big"] and max(c["foot_wl"]) > 0.04]
                    if flats:
                        f = max(flats, key=lambda c: c["n"])
                        tg = {"mode": "flatcomp", "center_xy": f["foot_center_xy"], "radius": max(f["foot_wl"]) / 2,
                              "n": f["n"], "thick": f["height"]}
                if tg is not None:
                    tg["thick"] = float(np.clip(tg.get("thick", 0.0), 0.0, 0.02))
                    self.target = tg
                    self.log("target", tg["mode"], "xy", np.round(tg["center_xy"], 3), "r %.3f" % tg["radius"], "n", tg["n"])
            if self.task["button"]:
                bb = self.select_button(sc["comps"], self.cube)
                if bb is not None:
                    self.button = bb[1]
                    self.button_xy = self.button["center_xy"].copy()
                    self.log("button candidate score %.2f xy=%s h=%.3f" % (bb[0], np.round(self.button["center_xy"], 3),
                                                                         self.button["height"]))
            self.last_scan = sc
            if self.cube is not None:
                return True
            # nothing found: lift the arm out of the way and retry
            p = self.rb.pos()
            self.move_to(np.array([p[0], p[1], max(p[2], self.table_z + 0.3)]))
        return self.cube is not None

    def refine(self, xy, kind, radius=0.07, tag=None):
        """Re-perceive near xy (wrist camera included) and return the object comp."""
        sc = self.perceive(tag=tag)
        if sc is None:
            return None
        comps = [c for c in sc["comps"] if np.linalg.norm(c["center_xy"] - xy) < radius]
        self.log("refine", kind, "near", np.round(xy, 3), "found", len(comps))
        self.describe(comps)
        if kind == "cube":
            best = self.select_cube(comps, self.task["color"])
            if self.accept(best):
                return best[1]
            return None
        if kind == "button":
            best = self.select_button(comps, None)
            return best[1] if best else None
        return None

    def gripper_yaw(self):
        R = self.rb.rot()
        return math.atan2(R[1, 1], R[0, 1])

    def yaw_delta(self, cube_yaw):
        d = wrap_quarter(cube_yaw - self.gripper_yaw())
        cands = [d, d - math.pi / 2, d + math.pi / 2]
        best = min(cands, key=lambda c: abs(self.yaw_cum + c) + 0.6 * abs(c))
        return best

    def adapt_speed(self, cycles_left, press_left):
        """Raise speed when the remaining steps look short for the remaining work."""
        rb = self.rb
        left = rb.steps_left()
        if left >= 10 ** 5:
            return
        done = max(1, self.picks_done)
        per_cycle = max(60.0, (rb.step_id - self.first_pick_step) / done) if self.picks_done else 80.0
        need = cycles_left * per_cycle + press_left * 45.0
        old = self.speed
        if need > 0.95 * left:
            self.speed = min(2.0, max(self.speed, 1.0 * need / max(left, 1)))
        else:
            self.speed = 1.0 if need < 0.7 * left else self.speed
        if self.speed != old:
            self.log("speed factor %.2f (need %.0f steps, left %d)" % (self.speed, need, left))

    def pick(self, k):
        rb = self.rb
        self.phase("PICK%d" % k)
        if self.first_pick_step is None:
            self.first_pick_step = rb.step_id
        try:
            self.adapt_speed(self.task["count"] - k, 1 if self.task["button"] and not self.task["button_first"] else 0)
        except Exception:
            pass
        for attempt in range(MAX_PICK_TRIES):
            self.check_time()
            cube = self.cube
            s = self.cube_size
            top = self.table_z + max(cube["height"], 0.015)
            hover = np.array([cube["center_xy"][0], cube["center_xy"][1], top + HOVER_ABOVE_TOP])
            self.move_to(hover, grip=-1.0)
            ref = self.refine(cube["center_xy"], "cube", tag="pick%d_%d" % (k, attempt))
            if ref is None:
                self.log("refine did not see the cube; global rescan")
                sc = self.scan("rescan")
                best = self.select_cube(sc["comps"], self.task["color"]) if sc else None
                if self.accept(best):
                    ref = best[1]
                else:
                    self.log("no confident detection; keeping the last estimate xy=%s" % np.round(cube["center_xy"], 3))
                    ref = cube
            self.cube = ref
            cube = ref
            meas = float(np.clip(cube.get("size", cube["height"]), 0.015, 0.07))
            if self.cube_size_fixed is None:
                self.cube_size_fixed = meas
                self.log("cube size fixed at %.4f" % meas)
            s = self.cube_size_fixed
            self.cube_size = s
            extra_yaw = self.yaw_bias + (math.pi / 4 if attempt % 2 == 1 else 0.0)
            dyaw = self.yaw_delta(cube["yaw"] + extra_yaw)
            self.yaw_cum += dyaw
            self.R_hold = rot_z(dyaw) @ rb.rot()
            self.log("pick attempt %d cube xy=%s h=%.3f yaw=%.0f dyaw=%.0f cum=%.0f" % (
                attempt, np.round(cube["center_xy"], 3), cube["height"], math.degrees(cube["yaw"]),
                math.degrees(dyaw), math.degrees(self.yaw_cum)))
            xy = cube["center_xy"]
            top = self.table_z + cube["height"]
            self.move_to([xy[0], xy[1], top + 0.06], grip=-1.0, v=V_FREE, tol=TOL_NEAR)
            top = cube["top_z"]
            gz = max(top - 0.5 * s, self.table_z + 0.012)
            self.move_to([xy[0], xy[1], gz + 0.015], grip=-1.0, v=V_NEAR, tol=TOL_NEAR, chunk=CHUNK_NEAR)
            self.move_to([xy[0], xy[1], gz], grip=-1.0, v=0.006, tol=0.003, chunk=CHUNK_NEAR, max_calls=8)
            self.log("at grasp pose", np.round(rb.pos(), 4).tolist(), "target z %.4f" % gz)
            w = self.gripper(1.0)
            self.log("closed width %.4f (cube %.3f)" % (w, s))
            if w < 0.25 * s:
                self.log("grasp missed (fingers closed)")
                self.gripper(-1.0, max_steps=12)
                self.move_to(rb.pos() + np.array([0, 0, 0.06]), grip=-1.0)
                continue
            lift = np.array([xy[0], xy[1], self.table_z + CARRY_ABOVE_TABLE])
            self.move_to(lift, grip=1.0, v=V_FREE * 0.8)
            w2 = rb.width()
            self.log("lifted to", np.round(rb.pos(), 3).tolist(), "width %.4f" % w2)
            left_behind = False
            try:
                sc = self.perceive(cams=("front",), tag=None)
                if sc:
                    for c in sc["comps"]:
                        if np.linalg.norm(c["center_xy"] - xy) < 0.025 and c["height"] < cube["height"] + 0.015 \
                                and abs(c.get("size", c["height"]) - s) < 0.4 * s and c["n"] >= 15 \
                                and self.cube_score(c, self.task["color"]) >= CUBE_CONFIDENT:
                            left_behind = True
                            self.log("cube-like object still at grasp spot after lift (n=%d h=%.3f)" % (c["n"], c["height"]))
            except Exception as e:
                self.log("lift check failed", repr(e)[:200])
            if left_behind:
                if 0.5 * s <= w2 <= 1.5 * s:
                    # fingers stopped at the cube width: they hold something; trust them
                    self.log("ambiguous: fingers hold an object of cube width; trusting fingers")
                else:
                    w2 = 0.0
            if w2 < 0.25 * s:
                self.log("cube dropped during lift")
                self.gripper(-1.0, max_steps=12)
                sc = self.scan("afterdrop")
                if sc:
                    best = self.select_cube(sc["comps"], self.task["color"])
                    if self.accept(best):
                        self.cube = best[1]
                continue
            if abs(extra_yaw - self.yaw_bias) > 1e-6:
                self.log("grasp worked with yaw offset %.0f deg; keeping it" % math.degrees(extra_yaw))
                self.yaw_bias = extra_yaw
            return True
        return False

    def place(self, k):
        rb = self.rb
        self.phase("PLACE%d" % k)
        s = self.cube_size
        if self.task["target"] and self.target is not None:
            dest = self.target["center_xy"]
            support = self.table_z + self.target.get("thick", 0.0) + 0.002
        else:
            dest = self.home_xy if self.home_xy is not None else rb.pos()[:2]
            support = self.table_z
        carry = self.table_z + CARRY_ABOVE_TABLE
        self.cube_expect = np.array(dest, float).copy()
        self.move_to([dest[0], dest[1], carry], grip=1.0, v=V_FREE)
        rz = support + 0.5 * s + PLACE_GAP
        self.move_to([dest[0], dest[1], rz + 0.03], grip=1.0, v=V_NEAR, tol=TOL_NEAR)
        self.move_to([dest[0], dest[1], rz], grip=1.0, v=0.006, tol=0.003, chunk=CHUNK_NEAR, max_calls=8)
        self.log("release at", np.round(rb.pos(), 4).tolist(), "width %.4f" % rb.width())
        self.gripper(-1.0, max_steps=12)
        up = np.array([dest[0], dest[1], support + s + HOVER_ABOVE_TOP * 0.8])
        self.move_to(up, grip=-1.0, v=V_FREE)
        # verify where the cube is
        sc = self.scan("placed%d" % k)
        if sc:
            best = self.select_cube(sc["comps"], self.task["color"])
            if self.accept(best):
                self.cube = best[1]
                d = float(np.linalg.norm(self.cube["center_xy"] - dest))
                self.log("cube after place xy=%s dist_to_dest=%.3f" % (np.round(self.cube["center_xy"], 3), d))
                return d
        # not seen confidently: assume it sits where it was released
        self.log("cube not confidently seen after place; assuming it is at the release point")
        c = dict(self.cube)
        c["center_xy"] = np.array(dest, float).copy()
        c["top_z"] = float(support + s)
        c["height"] = float(support + s - self.table_z)
        self.cube = c
        return None

    def press(self):
        rb = self.rb
        self.phase("PRESS")
        cands_tried = []
        for attempt in range(MAX_PRESS_TRIES):
            self.check_time()
            if self.button is None:
                sc = self.scan("findbutton%d" % attempt)
                if sc:
                    comps = [c for c in sc["comps"] if not any(np.linalg.norm(c["center_xy"] - x) < 0.02 for x in cands_tried)]
                    bb = self.select_button(comps, self.cube)
                    if bb:
                        self.button = bb[1]
            if self.button is None:
                self.log("no button found")
                return False
            b = self.button
            top = self.table_z + b["height"]
            xy = b["center_xy"]
            self.move_to([xy[0], xy[1], top + 0.09], grip=1.0, v=V_FREE)
            if attempt < 2:
                ref = self.refine(xy, "button", radius=0.05, tag="press%d" % attempt)
                if ref is not None:
                    b = ref
                    xy = b["center_xy"]
                    top = self.table_z + b["height"]
            self.log("press attempt %d at xy=%s top=%.3f" % (attempt, np.round(xy, 3), top))
            self.gripper(1.0, max_steps=12)
            self.move_to([xy[0], xy[1], top + 0.025], grip=1.0, v=V_NEAR, tol=TOL_NEAR)
            depth = 0.015 + 0.01 * attempt
            done = self.move_to([xy[0], xy[1], top - depth], grip=1.0, v=V_PRESS, tol=0.003, chunk=4,
                                max_calls=12, watch=lambda: rb.complete or rb.ended)
            self.log("pressed to", np.round(rb.pos(), 4).tolist(), "complete", rb.complete)
            if rb.complete or rb.ended:
                return True
            self.hold(4, 1.0)
            if rb.complete or rb.ended:
                return True
            self.move_to([xy[0], xy[1], top + 0.08], grip=1.0, v=V_FREE)
            if rb.complete or rb.ended:
                return True
            cands_tried.append(xy.copy())
            if attempt >= 1:
                self.button = None  # try another candidate
        return False

    def run(self):
        self.init()
        self.phase("LOCATE")
        if not self.locate_all():
            self.log("cube not found; giving up")
            return
        n = int(self.task["count"])
        order = []
        if self.task["button"] and self.task["button_first"]:
            order.append("press")
        order += ["pick"] * n
        if self.task["button"] and not self.task["button_first"]:
            order.append("press")
        self.log("plan", order)
        k = 0
        for item in order:
            if self.rb.ended or self.rb.complete:
                break
            for tries in range(3):
                try:
                    if item == "pick":
                        ok = self.pick(k)
                        if not ok:
                            self.log("pick %d failed after retries; global rescan" % k)
                            sc = self.scan("pickfail")
                            best = self.select_cube(sc["comps"], self.task["color"]) if sc else None
                            if self.accept(best):
                                self.cube = best[1]
                            ok = self.pick(k)
                        if ok:
                            self.picks_done += 1
                            self.place(k)
                    elif item == "press":
                        self.press()
                    break
                except Stop:
                    raise
                except Exception:
                    self.log("PHASE ERROR", item, k, traceback.format_exc()[-1500:])
                    try:
                        self.rb.observe()
                    except Exception:
                        pass
                    if self.rb.ended or self.rb.complete:
                        break
                    if item == "pick" and self.rb.state is not None and self.rb.width() > 0.25 * self.cube_size \
                            and self.rb.pos()[2] > (self.table_z or 0) + 0.06 and self.rb.state[6] + self.rb.state[7] < 0.07:
                        # something failed while carrying: finish this cycle by placing
                        try:
                            self.picks_done += 1
                            self.place(k)
                            break
                        except Stop:
                            raise
                        except Exception:
                            self.log("place recovery failed", traceback.format_exc()[-800:])
            if item == "pick":
                k += 1
        self.phase("END")
        self.log("finished: picks_done=%d complete=%s ended=%s" % (self.picks_done, self.rb.complete, self.rb.ended))


def main(ctx):
    c = None
    try:
        c = Controller(ctx)
        c.run()
    except Stop as e:
        if c:
            c.log("stop:", str(e))
    except Exception:
        if c:
            c.log("FATAL", traceback.format_exc())
        else:
            print(traceback.format_exc())
    finally:
        if c:
            c.log("phases", c.phase_log)
            c.log("calls", c.rb.ncalls, "call_time %.1f" % c.rb.call_time, "steps_sent", c.rb.steps_sent,
                  "step_id", c.rb.step_id, "complete", c.rb.complete, "ended", c.rb.ended)
