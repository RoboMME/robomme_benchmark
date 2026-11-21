"""BinFill controller (RAX, interaction-only, R1) — version v1.

Transition table (each step re-perceives; any exception -> log, re-perceive, continue):
  INIT      -> parse instruction, read status, first front observation          -> OVERVIEW
  OVERVIEW  -> raise arm, move above scene centre, fuse front+wrist height map    -> SELECT
  SELECT    -> placed >= goal for every colour?  yes -> BUTTON ; no free target cube -> BUTTON
               else choose nearest free target cube                              -> PICK
  PICK      -> hover, wrist refine, yaw align, descend, close, verify width
               ok -> CARRY ; miss -> lift, attempt counter++ (blacklist after 3)  -> OVERVIEW
  CARRY     -> lift to travel height, verify width, move over bin, lower, open
               released while held -> placed++ (after overview recount)          -> OVERVIEW
               lost during carry -> OVERVIEW
  BUTTON    -> hover over button, wrist refine, close fingers, press down (stall stop), lift
               episode ended -> DONE ; else one deeper retry -> DONE
Budget guards: wall time > WALL_BUTTON_S or remaining steps < STEP_RESERVE -> BUTTON.
"""
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import rax_io                      # noqa: E402
import perception as P             # noqa: E402
from robot import Robot, EpisodeOver, OPEN, CLOSE, Rz, yaw_of, wrap90, rotvec_to_R, R_to_rotvec  # noqa: E402

VERSION = "binfill-v1"
OUT = Path("/workspace/output")

# ---- tunables (priors; never tested on this task's simulator before freezing) ----
WALL_BUTTON_S = 950.0      # controller wall limit is 1200 s: go press the button after this
STEP_RESERVE = 90          # steps reserved for the button press
OVERVIEW_H = 0.36          # wrist overview height above table (m)
HOVER_DH = 0.14            # hover above cube top before refinement (m)
TRAVEL_MIN_H = 0.20        # minimum travel height above table (m)
GRASP_MIN_W = 0.006        # finger opening sum above this after closing = holding something
FREE_SPEED = 0.03          # m per step in free space
CARRY_SPEED = 0.022
SLOW_SPEED = 0.008
MAX_PICK_TRIES = 3

NUM_WORDS = {"a": 1, "an": 1, "one": 1, "single": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "once": 1, "twice": 2,
             "thrice": 3, "pair": 2, "couple": 2}
COLOR_WORDS = list(P.COLOR_HUES) + ["grey", "gray", "white", "black", "brown"]
CUBE_WORDS = ("cube", "cubes", "block", "blocks", "cuboid", "cuboids")


# --------------------------------------------------------------------------- logging
_LOGF = None


def log(msg):
    global _LOGF
    line = "[%s %7.1fs] %s" % (VERSION, time.time() - T0, msg)
    print(line, flush=True)
    try:
        if _LOGF is None:
            OUT.mkdir(parents=True, exist_ok=True)
            _LOGF = open(OUT / "controller_log.txt", "a")
        _LOGF.write(line + "\n")
        _LOGF.flush()
    except Exception:
        pass


T0 = time.time()


# --------------------------------------------------------------------------- instruction
def parse_instruction(text):
    """Return list of [count or 'all', colour or None] goals and whether a button press is asked."""
    t = (text or "").lower()
    toks = re.findall(r"[a-z]+|\d+", t)
    goals = []
    for i, w in enumerate(toks):
        if w not in CUBE_WORDS:
            continue
        num, col = None, None
        for k in range(i - 1, max(-1, i - 5), -1):
            u = toks[k]
            if col is None and u in COLOR_WORDS:
                col = "gray" if u == "grey" else u
            elif num is None and (u.isdigit() or u in NUM_WORDS):
                num = int(u) if u.isdigit() else NUM_WORDS[u]
                break
            elif u in ("all", "every", "each"):
                num = "all"
                break
            elif u in CUBE_WORDS or u in ("and", "then", "into", "in", "the", "bin"):
                if u in ("the",):
                    continue
                break
        if num is None:
            num = "all" if (w.endswith("s") and "all" in toks) else (1 if not w.endswith("s") else None)
        goals.append([num, col])
    # merge / fill missing numbers
    if not goals:
        nums = [int(u) if u.isdigit() else NUM_WORDS[u] for u in toks if u.isdigit() or
                (u in NUM_WORDS and u not in ("a", "an"))]
        cols = [u for u in toks if u in COLOR_WORDS]
        goals = [[nums[0] if nums else 1, cols[0] if cols else None]]
    for g in goals:
        if g[0] is None:
            nums = [int(u) if u.isdigit() else NUM_WORDS[u] for u in toks if u.isdigit() or
                    (u in NUM_WORDS and u not in ("a", "an"))]
            g[0] = nums[0] if nums else "all"
    # collapse duplicates of the same colour (e.g. "cubes" mentioned twice)
    merged = {}
    for n, c in goals:
        if c in merged:
            continue
        merged[c] = n
    goals = [[n, c] for c, n in merged.items()]
    if len(goals) > 1 and None in merged:
        goals = [g for g in goals if g[1] is not None]
    m = re.search(r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+times", t)
    if m and len(goals) == 1 and goals[0][0] in (1, None, "all"):
        w = m.group(1)
        goals[0][0] = int(w) if w.isdigit() else NUM_WORDS[w]
    button = ("button" in t) or ("press" in t)
    return goals, button


# --------------------------------------------------------------------------- perception
class Scene:
    def __init__(self, robot):
        self.robot = robot
        self.variant = {}
        self.table_z = None
        self.center = None
        self.static_clouds = []
        self.hm_saves = 0
        self.depth_ok = None
        self.method_log = {}

    def _tool_grid_cloud(self, obs, cam, stride=8):
        rgb = obs.rgb.get(cam)
        if rgb is None:
            return None
        Hh, Ww = rgb.shape[:2]
        sels = [(cam, u, v) for v in range(stride // 2, Hh, stride) for u in range(stride // 2, Ww, stride)]
        sid = obs.step_id if obs.step_id is not None else self.robot.steps
        res = []
        t0 = time.time()
        for k in range(0, len(sels), 32):
            if time.time() - t0 > 50.0:     # runtime kills runs idle >180 s without control steps
                log("tool grid truncated after %d/%d samples (time cap)" % (len(res), len(sels)))
                sels = sels[:len(res)]
                break
            res.extend(self.robot.pixel_to_position(sid, sels[k:k + 32], radius=0))
        pts, cols = [], []
        for (c, u, v), (xyz, sp) in zip(sels, res):
            if xyz is None:
                continue
            if sp is not None and isinstance(sp, (int, float)) and sp > 0.03:
                continue
            pts.append(xyz)
            cols.append(rgb[int(v), int(u)] / 255.0)
        if len(pts) < 20:
            return None
        # sparse samples: splat each over neighbouring height-map cells
        return np.array(pts), np.array(cols), 2

    def clouds(self, obs, cams):
        """Point clouds for the requested cameras of one observation.

        Per observation, 12 tool back-projections per camera (one call) validate the depth image
        against the authoritative tool: declared calibration if it matches within 1 cm, else a
        least-squares linear pinhole model, else a coarse tool-sampled grid."""
        out = {}
        sid = obs.step_id if obs.step_id is not None else self.robot.steps
        sels, owners = [], []
        for cam in cams:
            d = obs.depth.get(cam)
            if d is None or d.ndim != 2 or (cam in obs.stale):
                continue
            pix = P.spread_pixels(d, 12)
            for u, v in pix:
                sels.append((cam, u, v))
                owners.append(cam)
        res = []
        if sels:
            try:
                res = self.robot.pixel_to_position(sid, sels, radius=0)
            except Exception as e:
                log("pixel_to_position failed: %r" % e)
                res = [(None, None)] * len(sels)
        for cam in cams:
            d = obs.depth.get(cam)
            rgb = obs.rgb.get(cam)
            if d is None or d.ndim != 2 or cam in obs.stale:
                log("no usable depth for %s -> tool grid" % cam)
                c = self._tool_grid_cloud(obs, cam, stride=8 if cam == "front" else 6)
                if c is not None:
                    out[cam] = c
                continue
            samples = [(u, v, r[0]) for (c, u, v), r in zip(sels, res) if c == cam]
            nvalid = sum(1 for s_ in samples if s_[2] is not None)
            cal = obs.calib.get(cam)
            method = None
            if cal is not None and nvalid >= 4:
                var, err = P.choose_variant(d, cal, samples)
                if var is not None and err < 0.01:
                    method = ("calib", var, err)
            if method is None and nvalid >= 6:
                M, err = P.fit_linear(d, samples)
                if M is not None and err < 0.012:
                    method = ("linear", M, err)
            if method is None and cal is not None and nvalid < 4:
                conv = (cal.get("conv") or "opengl")
                var = ("opencv", 0.0) if "cv" in conv else ("opengl", 0.5)
                method = ("declared", var, float("nan"))
            if self.method_log.get(cam) != (method[0] if method else None):
                log("depth model %s: %s err=%s (valid samples %d/%d)" % (
                    cam, method[0] if method else None, None if method is None else
                    ("%.4f" % method[2]), nvalid, len(samples)))
                self.method_log[cam] = method[0] if method else None
            if method is None:
                c = self._tool_grid_cloud(obs, cam, stride=8 if cam == "front" else 6)
            elif method[0] == "linear":
                c = P.cloud_from_points(P.backproject_linear(d, method[1]), d, rgb)
            else:
                c = P.cloud_from(d, rgb, cal, method[1])
            if c is not None:
                out[cam] = c
        return out

    def perceive(self, use_front=True, use_wrist=True, keep_static=False, tag=""):
        obs = self.robot.observe()
        for n in obs.notes:
            log("obs: " + n)
        cams = (["front"] if use_front else []) + (["wrist"] if use_wrist else [])
        cl = self.clouds(obs, cams)
        clouds = []
        if "front" in cl:
            clouds.append(cl["front"])
            if keep_static:
                self.static_clouds.append(cl["front"])
        if "wrist" in cl:
            clouds.append(cl["wrist"])
        if not clouds:
            log("no point clouds available")
            return None, obs
        if self.table_z is None:
            Pf = clouds[0][0]
            tz = P.table_height(Pf)
            near = Pf[np.abs(Pf[:, 2] - tz) < 0.01]
            self.table_z = tz
            self.center = np.median(near[:, :2], 0) if len(near) > 50 else self.robot.pos[:2]
            log("table_z=%.4f centre=%s" % (self.table_z, np.round(self.center, 3)))
        allc = clouds + [c for c in self.static_clouds if c is not clouds[0]]
        hm = P.make_heightmap(allc, self.table_z, self.center, half=0.55)
        tcp = self.robot.pos
        seg = P.segment(hm, tcp[:2], tcp[2] - self.table_z, log)
        seg["hm"] = hm
        log("perceive[%s]: bin=%s" % (tag, seg["bin"]))
        for o in seg["cubes"]:
            log("   cube %s in_bin=%s part_of_other=%s" % (o, o.in_bin, o.part_of_other))
        for o in seg["others"]:
            log("   other %s" % o)
        for o in seg["all"]:
            if o.kind == "robot":
                log("   robot-part %s" % o)
        self._save_hm(hm, tag)
        return seg, obs

    def _save_hm(self, hm, tag):
        if self.hm_saves >= 14:
            return
        try:
            from PIL import Image
            h = np.clip(hm.h / 0.12, 0, 1)
            img = np.concatenate([np.stack([h] * 3, -1), hm.colors()], 1)
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img).save(OUT / ("hm_%02d_%s.png" % (self.hm_saves, tag)))
            self.hm_saves += 1
        except Exception as e:
            log("save hm failed %r" % e)

    def refine(self, x, y, color_hue=None, radius=0.05, max_h=0.15):
        """Wrist-camera local refinement of an object near (x, y). Returns dict or None."""
        obs = self.robot.observe()
        c = self.clouds(obs, ["wrist"]).get("wrist")
        if c is None:
            return None
        Pp, C = c[0], c[1]
        h = Pp[:, 2] - self.table_z
        max_h = min(max_h, self.robot.pos[2] - self.table_z - 0.025)   # never the fingers
        sel = (np.hypot(Pp[:, 0] - x, Pp[:, 1] - y) < radius) & (h > 0.008) & (h < max_h)
        if color_hue is not None:
            hh, ss, vv = P.rgb_to_hsv(C)
            dh = np.abs(((hh - color_hue) + 180) % 360 - 180)
            sel &= (dh < 30) & (ss > P.SAT_MIN * 0.8) & (vv > P.VAL_MIN)
        if sel.sum() < 15:
            log("refine: only %d points near (%.3f,%.3f)" % (sel.sum(), x, y))
            return None
        Q = Pp[sel]
        hq = Q[:, 2] - self.table_z
        top = float(np.percentile(hq, 95))
        topm = hq >= top - 0.008
        T = Q[topm]
        if len(T) < 8:
            T = Q
        yaw, ext = P.min_area_yaw(T[:, :2])
        # centre from min-area rectangle of the top face
        a = yaw
        Rm = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        q = (T[:, :2]) @ Rm.T
        mid = (q.max(0) + q.min(0)) / 2
        cxy = Rm.T @ mid
        return {"x": float(cxy[0]), "y": float(cxy[1]), "top": top, "yaw": yaw, "ext": ext, "n": int(len(T))}


# --------------------------------------------------------------------------- task
class Task:
    def __init__(self, ctx):
        self.robot = Robot(ctx, log)
        self.scene = Scene(self.robot)
        self.goals = []
        self.need_button = True
        self.placed = {}
        self.blacklist = []
        self.bin = None
        self.button = None
        self.rim = 0.06
        self.cube_h = 0.04
        self.seg = None
        self.phase_t = {}
        self.fast = False

    # ---------------------------------------------------------------- helpers
    def z(self, h):
        return self.scene.table_z + h

    def travel_h(self):
        h = TRAVEL_MIN_H
        if self.bin is not None:
            h = max(h, self.rim + 0.08 + self.cube_h)
        return min(h, 0.32)

    def budget_low(self):
        r = self.robot
        return (time.time() - T0) > WALL_BUTTON_S or r.remaining_steps() < STEP_RESERVE

    def to_travel(self, grip, speed=FREE_SPEED):
        p = self.robot.pos
        th = self.z(self.travel_h())
        if p[2] < th - 0.01:
            self.robot.move_to([p[0], p[1], th], grip, speed=speed, label="lift")

    def overview_xy(self):
        seg = self.seg
        pts = []
        if seg is not None:
            for o in seg["cubes"]:
                if not o.in_bin:
                    pts.append((o.x, o.y))
            if seg["bin"] is not None:
                pts.append((seg["bin"].cx, seg["bin"].cy))
            for o in seg["others"]:
                pts.append((o.x, o.y))
        if not pts:
            return self.robot.pos[:2]
        pts = np.array(pts)
        c = (pts.min(0) + pts.max(0)) / 2
        return c

    def goal_color_hue(self, color):
        return P.COLOR_HUES.get(color)

    def neutral_cubes(self, seg, color):
        """Cube-sized low-saturation objects matching white / black / gray."""
        out = []
        for o in list(seg["others"]) + [c for c in seg["cubes"] if c.sat < 0.35]:
            if not (0.022 <= o.ext[0] <= 0.075 and 0.015 <= o.top <= 0.08):
                continue
            if seg["bin"] is not None:
                x0, x1, y0, y1 = seg["bin"].bbox
                if x0 <= o.x <= x1 and y0 <= o.y <= y1:
                    continue
            v, sat = o.val, o.sat
            ok = {"white": v > 0.7 and sat < 0.3, "black": v < 0.3,
                  "gray": 0.25 <= v <= 0.8 and sat < 0.3}.get(color, False)
            if ok:
                o.in_bin, o.part_of_other = False, False
                out.append(o)
        return out

    def matches(self, cube, color):
        if color is None:
            return True
        if color == "brown":
            return P.hue_dist(cube.hue, "orange") < 25 and cube.val < 0.65
        if color in P.COLOR_HUES:
            return cube.color == color or (P.hue_dist(cube.hue, color) < 25 and cube.sat > P.SAT_MIN)
        return False

    def remaining_goals(self):
        out = []
        for n, c in self.goals:
            done = self.placed.get(c, 0)
            if n == "all":
                out.append((c, None))
            elif done < n:
                out.append((c, n - done))
        return out

    def free_targets(self, seg, color):
        if seg is None:
            return []
        if color in ("white", "black", "gray"):
            cubes = self.neutral_cubes(seg, color)
        else:
            cubes = [o for o in seg["cubes"] if not o.in_bin and not o.part_of_other and self.matches(o, color)]
        if self.button is not None:
            cubes = [o for o in cubes if math.hypot(o.x - self.button.x, o.y - self.button.y) > 0.03]
        good = []
        for o in cubes:
            bl = sum(1 for (bx, by) in self.blacklist if math.hypot(o.x - bx, o.y - by) < 0.025)
            if bl >= MAX_PICK_TRIES:
                continue
            good.append(o)
        # size outliers (possible button cap) only when nothing regular is left
        regular = [o for o in good if not getattr(o, "odd", False)]
        return regular if regular else good

    def update_scene(self, seg):
        self.seg = seg
        if seg is None:
            return
        if seg["bin"] is not None:
            b = seg["bin"]
            if self.bin is None or b.n > 0.5 * self.bin.n:
                self.bin = b
                self.rim = max(0.02, b.rim)
        cubes = [o for o in seg["cubes"] if not o.in_bin and not o.part_of_other]
        if cubes:
            self.cube_h = float(np.clip(np.median([o.top for o in cubes]), 0.015, 0.07))
        btn = P.pick_button(seg, cubes)
        if btn is not None:
            if self.button is None or math.hypot(btn.x - self.button.x, btn.y - self.button.y) < 0.05 \
                    or btn.n > self.button.n:
                self.button = btn

    # ---------------------------------------------------------------- phases
    def init(self):
        r = self.robot
        st = r.status()
        try:
            (OUT / "status_reply.json").write_text(json.dumps({k: v for k, v in st.items() if k != "_texts"},
                                                             default=str)[:20000])
        except Exception:
            pass
        obs = r.observe()
        try:
            j = {k: v for k, v in obs.json.items() if k != "_texts"}
            (OUT / "first_observe.json").write_text(json.dumps(j, default=str)[:60000])
            blocks = [{k: (v if k != "data" else len(v)) for k, v in b.items()}
                      for b in r.last_reply.get("content", [])]
            (OUT / "first_observe_blocks.json").write_text(json.dumps(blocks, default=str)[:20000])
        except Exception as e:
            log("save first obs failed %r" % e)
        instr = obs.instruction or rax_io.get_instruction(st)
        log("instruction: %r" % instr)
        log("state: %s  limits: step_limit=%s remaining=%s" % (
            None if r.state is None else np.round(r.state, 4).tolist(), r.step_limit, r.remaining))
        try:
            _, man = rax_io.manifest_last_execution_files(obs.json)
            if man:
                fr = man.get("frames", [])
                log("conditioning manifest: %d frames (%d video_demo)" % (
                    len(fr), sum(1 for f in fr if f.get("phase") == "video_demo")))
        except Exception as e:
            log("manifest read failed %r" % e)
        self.goals, self.need_button = parse_instruction(instr)
        log("goals=%s button=%s" % (self.goals, self.need_button))
        n_total = sum(n if isinstance(n, int) else 3 for n, _ in self.goals)
        lim = r.remaining_steps()
        self.fast = lim < 10 ** 5 and (n_total * 150 + 120) > 0.9 * lim
        log("estimated need %d steps vs remaining %s -> fast=%s" % (n_total * 150 + 120, lim, self.fast))
        if r.state is None:
            raise RuntimeError("no state in observation")
        R0 = r.R
        zax = R0[:, 2]
        ang = math.degrees(math.acos(max(-1, min(1, -zax[2]))))
        log("initial gripper z-axis %s (tilt from down %.1f deg) yaw=%.1f" % (
            np.round(zax, 3), ang, math.degrees(yaw_of(R0))))
        # the initial orientation is held as reference (no tilt "correction": the TCP axis
        # convention was never observed); yaw alignment rotates about world z only
        self.open_w = r.width
        r.R_ref = R0
        r.R_des = R0.copy()

    def overview(self, tag):
        r = self.robot
        if getattr(self, "fast", False) and self.scene.table_z is not None and tag.startswith("after"):
            r.hold(OPEN, 2)
            seg, _ = self.scene.perceive(use_front=True, use_wrist=True, tag=tag + "-inplace")
            self.update_scene(seg)
            return seg
        if self.scene.table_z is None:
            seg, _ = self.scene.perceive(use_front=True, use_wrist=True, keep_static=True, tag="start")
            self.update_scene(seg)
        p = r.pos
        z_ov = self.z(OVERVIEW_H)
        if p[2] < z_ov - 0.02:
            r.move_to([p[0], p[1], z_ov], OPEN if not self._holding else CLOSE, speed=FREE_SPEED, label="raise")
        xy = self.overview_xy()
        r.move_to([xy[0], xy[1], z_ov], OPEN, speed=FREE_SPEED, tol=0.01, label="overview")
        r.hold(OPEN, 3)
        seg, _ = self.scene.perceive(use_front=True, use_wrist=True, tag=tag)
        self.update_scene(seg)
        return seg

    _holding = False

    def pick(self, cube):
        r = self.robot
        log("PICK %s" % cube)
        hue = cube.hue if cube.sat > P.SAT_MIN else None
        self.to_travel(OPEN)
        hover_z = self.z(max(cube.top + HOVER_DH, 0.15))
        r.move_to([cube.x, cube.y, hover_z], OPEN, speed=FREE_SPEED, label="hover")
        r.hold(OPEN, 2)
        tx, ty, top, yaw = cube.x, cube.y, cube.top, None
        ref = None
        try:
            ref = self.scene.refine(cube.x, cube.y, hue, radius=0.045)
        except Exception as e:
            log("refine error %r" % e)
        if ref is not None and math.hypot(ref["x"] - cube.x, ref["y"] - cube.y) < 0.04:
            log("refined: %s" % {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ref.items()})
            tx, ty, top, yaw = ref["x"], ref["y"], ref["top"], ref["yaw"]
            if not (0.012 < top < 0.09):
                top = cube.top
        else:
            log("refine rejected: %s" % ref)
        # yaw alignment (relative to the reference orientation, within +-45 deg)
        if yaw is not None and r.rot_mode != "off":
            dpsi = wrap90(yaw - yaw_of(r.R_ref))
            r.R_des = Rz(dpsi) @ r.R_ref
            log("yaw align: cube yaw %.1f deg, gripper offset %.1f deg" % (math.degrees(yaw), math.degrees(dpsi)))
        r.move_to([tx, ty, hover_z], OPEN, speed=SLOW_SPEED * 2, tol=0.004, label="align")
        # second refinement closer, if the first one worked (skipped in fast mode)
        if ref is not None and not getattr(self, "fast", False):
            pre_z = self.z(top + 0.08)
            r.move_to([tx, ty, pre_z], OPEN, speed=FREE_SPEED * 0.7, tol=0.005, label="pre")
            try:
                ref2 = self.scene.refine(tx, ty, hue, radius=0.035)
            except Exception as e:
                ref2 = None
                log("refine2 error %r" % e)
            if ref2 is not None and math.hypot(ref2["x"] - tx, ref2["y"] - ty) < 0.02:
                log("refined2: x=%.4f y=%.4f top=%.4f n=%d" % (ref2["x"], ref2["y"], ref2["top"], ref2["n"]))
                tx, ty = ref2["x"], ref2["y"]
                if 0.012 < ref2["top"] < 0.09:
                    top = 0.5 * (top + ref2["top"])
        gz = self.z(min(max(top * 0.5, 0.012), 0.035))
        r.move_to([tx, ty, self.z(top + 0.03)], OPEN, speed=FREE_SPEED * 0.6, tol=0.004, label="above")
        ok, d = r.move_to([tx, ty, gz], OPEN, speed=SLOW_SPEED, tol=0.004, max_calls=20, label="descend")
        log("descend ok=%s d=%.4f pos=%s width=%.4f" % (ok, d, np.round(r.pos, 4), r.width))
        if not ok and r.pos[2] > gz + 0.015:
            log("descent blocked %.3f m above grasp height -> retreat" % (r.pos[2] - gz))
            self.blacklist.append((cube.x, cube.y))
            r.move_to([tx, ty, hover_z], OPEN, speed=FREE_SPEED, label="retreat")
            return False
        w_open = r.width
        w = r.gripper(CLOSE, 14)
        log("closed width=%.4f (open %.4f)" % (w, w_open))
        if w > w_open - 0.004 and w_open > 0.02:
            log("gripper did not close at all (width unchanged)")
            self.gripper_stuck = getattr(self, "gripper_stuck", 0) + 1
        if w < GRASP_MIN_W or (w > w_open - 0.004 and w_open > 0.02):
            log("grasp MISS")
            r.gripper(OPEN, 8)
            self.blacklist.append((cube.x, cube.y))
            r.move_to([tx, ty, hover_z], OPEN, speed=FREE_SPEED, label="retreat")
            return False
        self._holding = True
        return True

    def carry(self, cube_color):
        r = self.robot
        p = r.pos
        th = self.z(self.travel_h())
        r.move_to([p[0], p[1], th], CLOSE, speed=CARRY_SPEED * 0.6, label="lift")
        w = r.width
        log("after lift width=%.4f" % w)
        if w < GRASP_MIN_W:
            log("cube LOST during lift")
            self._holding = False
            r.gripper(OPEN, 6)
            return False
        b = self.bin
        r.move_to([b.cx, b.cy, th], CLOSE, speed=CARRY_SPEED, tol=0.008, label="to-bin")
        rel_z = self.z(self.rim + 0.035 + 0.5 * self.cube_h)
        rel_z = max(rel_z, self.z(0.06))
        r.move_to([b.cx, b.cy, rel_z], CLOSE, speed=CARRY_SPEED * 0.6, tol=0.008, label="lower")
        w = r.width
        held = w >= GRASP_MIN_W
        log("before release width=%.4f held=%s pos=%s bin=(%.3f,%.3f) rim=%.3f" % (
            w, held, np.round(r.pos, 3), b.cx, b.cy, self.rim))
        r.gripper(OPEN, 10)
        self._holding = False
        r.move_to([b.cx, b.cy, th], OPEN, speed=FREE_SPEED, tol=0.01, label="up")
        return held

    def press_button(self):
        r = self.robot
        b = self.button
        if b is None:
            log("no button found; re-perceiving")
            seg = self.overview("button-search")
            b = self.button
        if b is None:
            log("BUTTON NOT FOUND")
            return False
        log("PRESS button %s" % b)
        r.R_des = r.R_ref.copy()
        self.to_travel(CLOSE)
        bx, by, top = b.x, b.y, b.top
        hz = self.z(max(top + 0.12, 0.15))
        r.move_to([bx, by, hz], CLOSE, speed=FREE_SPEED, label="btn-hover")
        hue = b.hue if getattr(b, "sat", 0) > P.SAT_MIN else None
        try:
            ref = self.scene.refine(bx, by, hue, radius=0.04, max_h=0.12)
        except Exception as e:
            ref = None
            log("btn refine error %r" % e)
        if ref is not None and math.hypot(ref["x"] - bx, ref["y"] - by) < 0.03:
            log("btn refined: x=%.4f y=%.4f top=%.4f" % (ref["x"], ref["y"], ref["top"]))
            bx, by = ref["x"], ref["y"]
            if 0.005 < ref["top"] < 0.12:
                top = ref["top"]
        r.gripper(CLOSE, 8)
        r.move_to([bx, by, self.z(top + 0.04)], CLOSE, speed=FREE_SPEED * 0.6, tol=0.005, label="btn-above")
        for depth in (0.02, 0.035):
            z0 = r.pos[2]
            ok, d = r.move_to([bx, by, self.z(top - depth)], CLOSE, speed=0.006, tol=0.003, max_calls=15,
                              rows_per_call=4, label="press")
            log("press depth=%.3f reached z=%.4f (top=%.4f) ok=%s" % (depth, r.pos[2] - self.scene.table_z, top, ok))
            r.hold(CLOSE, 4)
            r.move_to([bx, by, self.z(top + 0.06)], CLOSE, speed=0.015, tol=0.006, label="btn-up")
            r.hold(CLOSE, 6)
            if r.ended:
                return True
        return r.ended

    # ---------------------------------------------------------------- main loop
    def run(self):
        self.init()
        seg = self.overview("ov0")
        self.scene.static_clouds = []      # the start-pose cloud is stale once anything moves
        if self.bin is None:
            log("no bin detected in first overview; retry from another height")
            p = self.robot.pos
            self.robot.move_to([p[0], p[1], self.z(OVERVIEW_H + 0.06)], OPEN, label="higher")
            seg, _ = self.scene.perceive(tag="ov0b")
            self.update_scene(seg)
        cycles = 0
        fail_streak = 0
        while cycles < 30:
            cycles += 1
            if self.budget_low():
                log("budget low (t=%.0fs, remaining steps=%s) -> button" % (time.time() - T0,
                                                                             self.robot.remaining_steps()))
                break
            rem = self.remaining_goals()
            if not rem:
                log("all goals met: placed=%s" % self.placed)
                break
            if self.bin is None:
                log("NO BIN; cannot place")
                break
            chosen = None
            for color, n_left in rem:
                cands = self.free_targets(self.seg, color)
                if n_left is None and not cands:
                    continue
                if cands:
                    p = self.robot.pos
                    chosen = (color, min(cands, key=lambda o: math.hypot(o.x - p[0], o.y - p[1])))
                    break
            if chosen is None:
                log("no free target cubes left for %s; placed=%s" % (rem, self.placed))
                if all(n is None for _, n in rem):
                    break
                if fail_streak < 2:
                    fail_streak += 1
                    self.overview("search%d" % cycles)
                    continue
                break
            color, cube = chosen
            n_before = len(self.free_targets(self.seg, color))
            try:
                ok = self.pick(cube)
                if ok:
                    held = self.carry(color)
                else:
                    held = False
            except EpisodeOver:
                raise
            except Exception:
                log("pick/carry exception: %s" % traceback.format_exc()[-1500:])
                held = False
                self._holding = False
                try:
                    self.robot.gripper(OPEN, 6)
                except EpisodeOver:
                    raise
                except Exception:
                    pass
            seg = self.overview("after%d" % cycles)
            n_after = len(self.free_targets(seg, color)) if seg is not None else n_before - 1
            if held:
                if n_after <= n_before - 1 or seg is None:
                    self.placed[color] = self.placed.get(color, 0) + 1
                    fail_streak = 0
                    log("PLACED one %s cube (count now %s); free before=%d after=%d" % (
                        color, self.placed, n_before, n_after))
                else:
                    log("released while held but free count did not drop (%d -> %d): not counted" % (
                        n_before, n_after))
                    self.blacklist.append((cube.x, cube.y))
            else:
                fail_streak += 1
        if self.need_button:
            self.press_button()
        else:
            log("instruction asks for no button press")
        log("DONE placed=%s ended=%s complete=%s steps=%s" % (self.placed, self.robot.ended,
                                                             self.robot.complete, self.robot.steps))


def main(ctx):
    global T0
    T0 = time.time()
    log("start %s" % VERSION)
    task = Task(ctx)
    try:
        task.run()
    except EpisodeOver:
        log("episode over: complete=%s steps=%s placed=%s" % (task.robot.complete, task.robot.steps, task.placed))
    except Exception:
        log("FATAL in run: %s" % traceback.format_exc()[-3000:])
        # last resort: if we know the button and the episode is alive, try to press it
        try:
            if not task.robot.ended and task.scene.table_z is not None and task.button is not None:
                task.press_button()
        except Exception:
            log("fallback press failed: %s" % traceback.format_exc()[-1000:])
    log("exit after %.1fs, %d calls" % (time.time() - T0, task.robot.calls))
