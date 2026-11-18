"""BinFill controller.

Task: put the instructed number of each coloured cube into the bin, then press the button.
Cubes may lie still (static variant) or ride a conveyor (moving variant).

Pipeline / transition table
  INIT      -> parse instruction, read first observation, locate bin & button      -> SCAN
  SCAN      -> front blobs at two times, belt velocity, pick needed isolated cube    -> APPROACH | WAIT(no target)
  APPROACH  -> hover above predicted cube position (tracking)                       -> REFINE
  REFINE    -> wrist RGB-D: precise centre, top, yaw; choose finger yaw            -> GRASP | SCAN(lost)
  GRASP     -> descend while tracking, close, lift, check finger gap & colour       -> CARRY | DISCARD(wrong colour) | SCAN(miss)
  CARRY     -> over bin, release, verify, count                                    -> SCAN | PRESS(all done)
  PRESS     -> close gripper, press button top, watch task_complete                -> END
Every phase has step/attempt limits; exceptions are logged and the loop re-perceives.
"""
import json
import math
import os
import time
import traceback

import numpy as np

import perception as P
from rmio import Obs, decode, aa_to_R, R_to_aa, Rz, tool_yaw, wrap, wrap90

VERSION = "v0.3"

# ---- tunables (untested values marked '?') ----
POS_SCALE = 0.05          # m per unit action (protocol)
ROT_SCALE = 0.5           # rad per unit action (protocol)
# Measured ep1 (static layout, 339 calls): the arm achieves 0.40 of each commanded translation
# and 0.39 of each commanded rotation per step (first-order, very consistent). Rows are divided
# by these gains so the planned per-step motion is what actually happens.
POS_GAIN = 0.40
ROT_GAIN = 0.39
VMAX_FREE = 0.018         # m/step actually achieved (cmd 0.9) in free space
VMAX_NEAR = 0.008         # m/step near objects
VMAX_CARRY = 0.018        # m/step with a cube
ROT_MAX = 0.08            # rad/step achieved (cmd 0.41)
HOVER = 0.10              # m above cube top for approach ?
CARRY_CLEAR = 0.12        # m above max(bin top, table) while carrying ?
RELEASE_ABOVE_BIN = 0.06  # TCP height above bin top for release ?
CUBE_SIZE_DEFAULT = 0.04  # m ?
ISOLATION = 0.065         # m min centre distance to other cubes ?
GRASP_OVERHEAD = 40       # steps besides travel from scan to closed grip (ep2 v0.2: ~48; v0.3 shorter)
END_MARGIN_STEPS = 6      # cube must still be >= this many steps before the belt end when grasped
CLOSE_STEPS = 6           # ep2 trace: grip reaches 0.0366 (cube held) 5 steps after close
OPEN_STEPS = 3            # ep2 trace: grip 0.054 after 1 open step, 0.078 after 6
OUT_DIR = "/workspace/output"


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    try:
        with open(os.path.join(OUT_DIR, "controller_log.txt"), "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


class EpisodeOver(Exception):
    pass


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.obs = None
        self.ended = False
        self.complete = False
        self.limit = None
        self.remaining = None
        self.t0 = time.time()
        self.n_saved = 0
        self.calls = 0

    def _update(self, result):
        meta, images = decode(result)
        o = Obs(meta, images)
        if o.depth.get("front") is None or o.depth.get("wrist") is None:
            try:
                meta2, images2 = decode(self.ctx.call("robomme_observe"))
                o = Obs(meta2, images2)
                meta = meta2
            except Exception as e:
                log("observe-after-step failed", e)
        tr = meta.get("transition") or {}
        for s in tr.get("steps", []) or []:
            if s.get("task_complete"):
                self.complete = True
            if s.get("episode_ended"):
                self.ended = True
        if meta.get("task_complete") or tr.get("task_complete"):
            self.complete = True
        if tr.get("episode_ended"):
            self.ended = True
        if meta.get("episode_ended"):
            self.ended = True
        lim = meta.get("episode_limits") or {}
        self.limit = lim.get("episode_step_limit", self.limit)
        self.remaining = lim.get("remaining_episode_steps", self.remaining)
        self.obs = o
        return o

    def observe(self):
        self.calls += 1
        return self._update(self.ctx.call("robomme_observe"))

    def step(self, rows):
        if self.ended:
            raise EpisodeOver()
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows][:100]
        if self.remaining is not None and self.remaining <= 0:
            self.ended = True
            raise EpisodeOver()
        self.calls += 1
        o = self._update(self.ctx.call("robomme_step", actions=rows))
        if self.ended:
            log("EPISODE ENDED at step", o.step_id, "complete", self.complete)
            raise EpisodeOver()
        return o

    def save(self, tag):
        """Save the current RGB images for offline debugging."""
        try:
            from PIL import Image
            d = os.path.join(OUT_DIR, "frames")
            os.makedirs(d, exist_ok=True)
            o = self.obs
            for c in ("front", "wrist"):
                if o.rgb.get(c) is not None:
                    Image.fromarray(o.rgb[c]).save(os.path.join(d, "%04d_s%04d_%s_%s.png" % (self.n_saved, o.step_id or 0, tag, c)))
            if self.n_saved < 400:
                np.savez_compressed(os.path.join(d, "%04d_s%04d_%s.npz" % (self.n_saved, o.step_id or 0, tag)),
                                    front_depth=o.depth.get("front"), wrist_depth=o.depth.get("wrist"),
                                    meta=json.dumps({k: o.meta.get(k) for k in ("camera_intrinsics", "camera_poses", "state", "step_id")}))
            self.n_saved += 1
        except Exception as e:
            log("save failed", e)


class Controller:
    def __init__(self, ctx):
        self.rb = Robot(ctx)
        self.goal = {}
        self.count = {}
        self.R_ref = None       # reference (initial) tool orientation
        self.yaw_ref = 0.0
        self.tz = 0.0
        self.bin = None
        self.button = None
        self.vel = np.zeros(2)  # belt velocity per step (xy)
        self.vel_n = 0
        self.home = None
        self.grip_cmd = -1.0
        self.finger_offset = None  # yaw(finger axis) - yaw(tool x)
        self.belt = None        # (lo, hi) xy bbox of mat if any
        self.failed_targets = []

    # ---------------- motion ----------------
    def R_des(self, yaw):
        return Rz(yaw - self.yaw_ref) @ self.R_ref

    def mark(self, name):
        o = self.rb.obs
        log("PHASE %-10s step %5s wall %6.1f" % (name, o.step_id, time.time() - self.rb.t0))

    def finger_yaw(self, o=None):
        o = o or self.rb.obs
        if self.finger_offset is None:
            return tool_yaw(o.R)
        return wrap(tool_yaw(o.R) + self.finger_offset)

    def move(self, target, yaw=None, grip=None, vmax=VMAX_FREE, tol=0.006, max_steps=120, chunk=5,
             yaw_tol=0.05, settle=False, ff=None):
        """Closed-loop move of the TCP. target: xyz or f(step)->xyz. yaw: finger-axis yaw or None.

        ff: per-step feed-forward xy velocity (moving targets). Returns True when within tolerance."""
        rb = self.rb
        grip = self.grip_cmd if grip is None else grip
        self.grip_cmd = grip
        used = 0
        while used < max_steps:
            o = rb.obs
            s = o.step_id or 0
            tgt_fn = target if callable(target) else (lambda k, t=np.asarray(target, float): t)
            p = o.pos.copy()
            err0 = tgt_fn(s) - p
            Rc = o.R
            yaw_err = 0.0
            if yaw is not None:
                cur_f = self.finger_yaw(o)
                yaw_err = wrap(yaw - cur_f)
                # finger symmetric under 180 deg
                if yaw_err > math.pi / 2:
                    yaw_err -= math.pi
                if yaw_err < -math.pi / 2:
                    yaw_err += math.pi
            if np.linalg.norm(err0) < tol and abs(yaw_err) < yaw_tol and not settle:
                return True
            tool_yaw_des = tool_yaw(Rc) + yaw_err
            Rd = self.R_des(tool_yaw_des)
            rot_err = R_to_aa(Rd @ Rc.T)
            n = min(chunk, max_steps - used)
            rows = []
            pp = p.copy()
            rr = rot_err.copy()
            for i in range(n):
                t = tgt_fn(s + i + 1)
                d = t - pp
                lim = vmax + (np.linalg.norm(ff) if ff is not None else 0.0)
                nd = np.linalg.norm(d)
                if nd > lim:
                    d = d * (lim / nd)
                pp = pp + d
                rn = np.linalg.norm(rr)
                dr = rr if rn <= ROT_MAX else rr * (ROT_MAX / rn)
                rr = rr - dr
                rows.append([d[0] / (POS_SCALE * POS_GAIN), d[1] / (POS_SCALE * POS_GAIN), d[2] / (POS_SCALE * POS_GAIN),
                             dr[0] / (ROT_SCALE * ROT_GAIN), dr[1] / (ROT_SCALE * ROT_GAIN), dr[2] / (ROT_SCALE * ROT_GAIN), grip])
            rb.step(rows)
            used += n
            if settle:
                settle = False
        o = rb.obs
        tgt = target(o.step_id) if callable(target) else np.asarray(target)
        return np.linalg.norm(tgt - o.pos) < tol * 2

    def hold(self, n, grip=None, track=None):
        """Hold position (or follow track(step)->xyz) for n steps with the given gripper command."""
        grip = self.grip_cmd if grip is None else grip
        self.grip_cmd = grip
        rb = self.rb
        done = 0
        while done < n:
            k = min(10, n - done)
            o = rb.obs
            s = o.step_id or 0
            p = o.pos.copy()
            rows = []
            for i in range(k):
                if track is not None:
                    d = track(s + i + 1) - p
                    nd = np.linalg.norm(d)
                    if nd > VMAX_NEAR:
                        d *= VMAX_NEAR / nd
                    p = p + d
                else:
                    d = np.zeros(3)
                rows.append([d[0] / (POS_SCALE * POS_GAIN), d[1] / (POS_SCALE * POS_GAIN), d[2] / (POS_SCALE * POS_GAIN), 0, 0, 0, grip])
            rb.step(rows)
            done += k

    # ---------------- perception helpers ----------------
    def front_blobs(self, o=None, colors=None):
        o = o or self.rb.obs
        pts = o.points("front")
        cols = colors or list(P.HUES)
        bl = P.cube_blobs(o.rgb["front"], pts, cols, min_area=10, z_floor=self.tz + 0.008)
        return [b for b in bl if b["extent"].max() < 0.15 and not self.in_excluded(b["center"])]

    def in_excluded(self, c):
        """Cubes inside/over the bin or on the button are never targets."""
        if self.bin is not None:
            half = max(0.06, 0.5 * float(np.max(self.bin.get("size", [0.1, 0.1]))) + 0.03)
            if np.linalg.norm(np.asarray(c[:2]) - self.bin["center"][:2]) < half:
                return True
        if self.button is not None and np.linalg.norm(np.asarray(c[:2]) - self.button["center"][:2]) < 0.04:
            return True
        return False

    def wrist_blobs(self, o=None, colors=None):
        o = o or self.rb.obs
        pts = o.points("wrist")
        cols = colors or list(P.HUES)
        bl = P.cube_blobs(o.rgb["wrist"], pts, cols, min_area=25, z_floor=self.tz + 0.008)
        return [b for b in bl if not self.in_excluded(b["center"])]

    def locate_static(self, o):
        """Bin and button from the front camera."""
        pts = o.points("front")
        rgb = o.rgb["front"]
        hsv = P.hsv_deg(rgb)
        bins = P.bin_candidates(rgb, pts, self.tz, hsv)
        log("bin candidates", [(round(c["area"]), np.round(c["center"], 3).tolist(), c["nwall"], np.round(c["size"], 3).tolist()) for c in bins[:4]])
        bin_ = None
        for c in bins:
            if c["nwall"] >= 5 and c["area"] >= 15:
                bin_ = c
                break
        if bin_ is None and bins:
            bin_ = bins[0]
        excl = [bin_["center"]] if bin_ is not None else []
        robot_xy = self.base_xy
        btns = P.button_candidates(rgb, pts, self.tz, exclude_xy=excl, hsv=hsv, robot_xy=robot_xy)
        log("button candidates", [(c["area"], np.round(c["center"], 3).tolist(), np.round(c["ext"], 3).tolist(), round(c["v"])) for c in btns[:5]])
        btn = btns[0] if btns else None
        return bin_, btn

    # ---------------- phases ----------------
    def init(self):
        rb = self.rb
        o = rb.observe()
        self.instruction = o.meta.get("instruction", "")
        self.goal = P.parse_instruction(self.instruction)
        self.count = {c: 0 for c in self.goal}
        try:  # development-only resume hint (absent in rehearsal/formal runs)
            ov = os.environ.get("BINFILL_COUNTS")
            if ov:
                self.count.update(json.loads(ov))
        except Exception:
            pass
        log(VERSION, "instruction:", self.instruction, "goal:", self.goal, "limits", o.meta.get("episode_limits"))
        log("state", np.round(o.state, 4).tolist())
        rb.save("init")
        self.R_ref = o.R
        self.yaw_ref = tool_yaw(o.R)
        self.home = o.pos.copy()
        self.base_xy = None
        self.tz = P.table_z(o.points("front"))
        log("table z", round(self.tz, 4), "cam front pos", np.round(o.cams["front"].pos, 3).tolist())
        self.update_finger_offset(o)
        self.bin, self.button = self.locate_static(o)
        log("bin", None if self.bin is None else np.round(self.bin["center"], 3).tolist(),
            "button", None if self.button is None else np.round(self.button["center"], 3).tolist())

    def update_finger_offset(self, o):
        try:
            cx = o.cams["wrist"].R[:, 0]
            fy = math.atan2(cx[1], cx[0])
            self.finger_offset = wrap(fy - tool_yaw(o.R))
        except Exception:
            pass

    def needed(self):
        return [c for c in self.goal if self.count.get(c, 0) < self.goal[c]]

    def estimate_velocity(self, wait=6):
        rb = self.rb
        o1 = rb.obs
        b1 = self.front_blobs(o1)
        self.hold(wait)
        o2 = rb.obs
        b2 = self.front_blobs(o2)
        dt = (o2.step_id or 0) - (o1.step_id or 0)
        if dt < 4:
            return b2  # v0.3: velocity already known; skip a noisy short-baseline update
        v, n = P.match_velocity(b1, b2, dt)
        if v is not None:
            if self.vel_n == 0:
                self.vel = v
            else:
                self.vel = 0.5 * self.vel + 0.5 * v
            self.vel_n += n
        log("velocity est", None if v is None else np.round(v * 1000, 2).tolist(), "mm/step from", n, "-> vel", np.round(self.vel * 1000, 2).tolist())
        return b2

    def choose_target(self, blobs, o):
        need = self.needed()
        s = o.step_id or 0
        cands = []
        for b in blobs:
            if b["color"] not in need:
                continue
            if b["extent"].max() > 0.075:
                continue  # merged blob
            c = b["center"]
            others = [x for x in blobs if x is not b]
            dmin = min([np.linalg.norm(x["center"][:2] - c[:2]) for x in others] + [1.0])
            if any(np.linalg.norm(c[:2] - f[:2]) < 0.02 for f in self.failed_targets):
                continue
            ee = o.pos
            dist = np.linalg.norm(c[:2] - ee[:2])
            iso_pen = 0.0 if dmin >= ISOLATION else (ISOLATION - dmin) * 8.0
            speed = float(np.linalg.norm(self.vel))
            if self.belt is not None and speed > 0.0005:
                # Conveyor (v0.3): a needed cube that leaves the belt is lost for good (ep2 failure),
                # so serve the most downstream cube that can still be caught before the belt end.
                lo, hi = self.belt
                vdir = self.vel / speed
                end_proj = max(np.dot(lo - c[:2], vdir), np.dot(hi - c[:2], vdir))
                t_end = end_proj / speed  # steps until the cube reaches the belt end
                T = dist / VMAX_FREE + GRASP_OVERHEAD
                for _ in range(2):  # intercept point refinement
                    fut = c[:2] + self.vel * T
                    T = np.linalg.norm(fut - ee[:2]) / VMAX_FREE + GRASP_OVERHEAD
                if t_end - T < END_MARGIN_STEPS:
                    continue
                score = t_end + iso_pen / VMAX_FREE
            else:
                score = dist + iso_pen
            cands.append((score, b, dmin))
        cands.sort(key=lambda x: x[0])
        return cands

    def scan(self):
        """Return (blob, step) target or None."""
        rb = self.rb
        self.mark("scan")
        blobs = self.estimate_velocity(wait=6 if self.vel_n < 6 else 1)
        o = rb.obs
        rb.save("scan")
        if self.belt is None:
            self.belt = self.detect_belt(o)
        cands = self.choose_target(blobs, o)
        log("scan step", o.step_id, "need", self.needed(), "blobs", [(b["color"], np.round(b["center"], 3).tolist()) for b in blobs if b["color"] in self.goal],
            "cands", [(round(sc, 3), b["color"], round(dm, 3)) for sc, b, dm in cands[:4]])
        if not cands:
            return None
        return cands[0][1], o.step_id

    def detect_belt(self, o):
        try:
            import cv2
            h, s, v = P.hsv_deg(o.rgb["front"])
            pts = o.points("front")
            hgt = pts[..., 2] - self.tz
            m = ((s < 70) & (v > 40) & np.isfinite(hgt) & (hgt > 0.003) & (hgt < 0.05)).astype(np.uint8)
            n, lab, st, cen = cv2.connectedComponentsWithStats(m)
            if n <= 1:
                return None
            k = 1 + int(np.argmax(st[1:, 4]))
            if st[k, 4] < 3000:
                return None
            Pm = pts[lab == k]
            Pm = Pm[np.isfinite(Pm[:, 0])]
            lo = np.percentile(Pm[:, :2], 2, axis=0)
            hi = np.percentile(Pm[:, :2], 98, axis=0)
            belt_z = float(np.median(Pm[:, 2]))
            log("belt bbox", np.round(lo, 3).tolist(), np.round(hi, 3).tolist(), "z", round(belt_z, 3))
            return (lo, hi)
        except Exception as e:
            log("belt detect failed", e)
            return None

    def predict(self, c, s0):
        c = np.asarray(c, float)
        v = self.vel
        return lambda s: np.array([c[0] + v[0] * (s - s0), c[1] + v[1] * (s - s0), c[2]])

    def refine(self, color, pred_fn):
        """Wrist view near the predicted cube. Returns (center, s, yaw, size, blobs) or None."""
        o = self.rb.obs
        s = o.step_id
        guess = pred_fn(s)
        bl = self.wrist_blobs(o)
        best, bd = None, 1e9
        for b in bl:
            if b["color"] != color:
                continue
            d = np.linalg.norm(b["center"][:2] - guess[:2])
            if d < bd:
                best, bd = b, d
        if best is None or bd > 0.06:
            log("refine: target not found in wrist; nearest", round(bd, 3))
            return None
        yaw = P.top_face_yaw(best["top_pts"])
        # support surface height around the cube
        pts = o.points("wrist")
        try:
            import cv2
            x, y, w, h = best["bbox"]
            pad = 12
            sub = pts[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad].reshape(-1, 3)
            sub = sub[np.isfinite(sub[:, 0])]
            sub = sub[sub[:, 2] < best["top_z"] - 0.012]
            surf = float(np.median(sub[:, 2])) if len(sub) > 20 else best["top_z"] - CUBE_SIZE_DEFAULT
        except Exception:
            surf = best["top_z"] - CUBE_SIZE_DEFAULT
        size = best["top_z"] - surf
        if not (0.015 < size < 0.07):
            size = CUBE_SIZE_DEFAULT
        log("refine", color, "center", np.round(best["center"], 4).tolist(), "pred err", round(bd, 4), "yaw", None if yaw is None else round(math.degrees(yaw), 1),
            "size", round(size, 4))
        return best["center"], s, yaw, size, bl

    def grasp_cycle(self, target, s0):
        rb = self.rb
        color = target["color"]
        c = target["center"].copy()
        pred = self.predict(c, s0)
        z_hover = c[2] + HOVER
        # 1. approach hover point (tracking)
        hov = lambda s: pred(s) + np.array([0, 0, HOVER])
        self.mark("approach")
        ok = self.move(hov, vmax=VMAX_FREE, tol=0.01, max_steps=80, chunk=5, ff=self.vel, grip=-1.0)
        self.mark("refine")
        rb.save("hover")
        r = self.refine(color, pred)
        if r is None:
            return "lost"
        c, s1, cyaw, size, wbl = r
        # velocity from two refinements
        pred = self.predict(c, s1)
        hov = lambda s: pred(s) + np.array([0, 0, HOVER * 0.6])
        yaw_pre = self.choose_yaw(c, cyaw, wbl, quiet=True) if cyaw is not None else None
        # v0.3: start the wrist turn during the second hover (yaw_tol loose: position decides the end)
        self.move(hov, yaw=yaw_pre, vmax=VMAX_NEAR, tol=0.006, yaw_tol=0.6, max_steps=30, chunk=4, ff=self.vel)
        rb.save("hover2")
        r2 = self.refine(color, pred)
        if r2 is not None:
            c2, s2, cyaw2, size2, wbl = r2
            if s2 > s1:
                vmeas = (c2[:2] - c[:2]) / (s2 - s1)
                if np.linalg.norm(vmeas - self.vel) < 0.004:
                    self.vel = 0.5 * self.vel + 0.5 * vmeas
            c, s1, size = c2, s2, size2
            if cyaw2 is not None:
                cyaw = cyaw2
            pred = self.predict(c, s1)
        # 2. finger yaw choice
        yaw_g = self.choose_yaw(c, cyaw, wbl)
        surf = c[2] - size
        z_grasp = max(surf + 0.5 * size, surf + 0.012)
        # 3. align yaw above the cube, then descend
        self.mark("align")
        pre = lambda s: pred(s) + np.array([0, 0, 0.05])
        self.move(pre, yaw=yaw_g, vmax=VMAX_NEAR, tol=0.006, yaw_tol=0.04, max_steps=30, chunk=4, ff=self.vel)
        self.mark("descend")
        down = lambda s: np.array([pred(s)[0], pred(s)[1], z_grasp])
        self.move(down, yaw=yaw_g, vmax=0.01, tol=0.004, max_steps=20, chunk=3, ff=self.vel)
        rb.save("predclose")
        self.mark("close")
        self.hold(CLOSE_STEPS, grip=1.0, track=down)
        g_closed = rb.obs.grip
        self.mark("lift")
        up = rb.obs.pos.copy()
        up[2] = z_grasp + 0.10
        self.move(up, vmax=VMAX_CARRY, tol=0.02, max_steps=20, chunk=5, grip=1.0)
        o = rb.obs
        rb.save("lifted")
        g = o.grip
        log("closed grip", round(g_closed, 4), "after lift", round(g, 4))
        if g < 0.008:
            self.failed_targets = []
            return "miss"
        # colour check: cube between fingers seen from the wrist camera
        held = self.held_color(o)
        log("held colour", held)
        if held is not None and held != color:
            return "wrong:" + held
        return "ok"

    def choose_yaw(self, c, cyaw, wbl, quiet=False):
        cur = self.finger_yaw()
        base = cyaw if cyaw is not None else 0.0
        opts = [wrap(base), wrap(base + math.pi / 2)]
        others = [b["center"] for b in wbl if np.linalg.norm(b["center"][:2] - c[:2]) > 0.025]

        def clearance(yw):
            ax = np.array([math.cos(yw), math.sin(yw)])
            fps = [c[:2] + ax * 0.05, c[:2] - ax * 0.05]
            return min([np.linalg.norm(f - o[:2]) for f in fps for o in others] + [1.0])

        def turn(yw):
            e = wrap(yw - cur)
            if e > math.pi / 2:
                e -= math.pi
            if e < -math.pi / 2:
                e += math.pi
            return abs(e)
        scored = sorted(opts, key=lambda yw: (clearance(yw) < 0.03, turn(yw)))
        yaw_g = scored[0] if cyaw is not None else cur
        if not quiet:
            log("grasp yaw", round(math.degrees(yaw_g), 1), "clear", round(clearance(yaw_g), 3), "cur finger yaw", round(math.degrees(cur), 1))
        return yaw_g

    def held_color(self, o):
        try:
            rgb = o.rgb["wrist"]
            pts = o.points("wrist")
            bl = P.cube_blobs(rgb, pts, list(P.HUES), min_area=200)
            # held cube: close to the camera (large blob, near tcp height)
            best = None
            for b in bl:
                if abs(b["center"][2] - o.pos[2]) < 0.06:
                    if best is None or b["area"] > best["area"]:
                        best = b
            return None if best is None else best["color"]
        except Exception:
            return None

    def go_bin(self):
        rb = self.rb
        b = self.bin
        if b is None:
            return False
        ztop = b["top_z"]
        zc = max(ztop, self.tz) + CARRY_CLEAR
        p = rb.obs.pos.copy()
        if p[2] < zc - 0.01:
            self.move(np.array([p[0], p[1], zc]), vmax=VMAX_CARRY, tol=0.01, max_steps=20, chunk=5)
        tgt = np.array([b["center"][0], b["center"][1], zc])
        self.move(tgt, vmax=VMAX_CARRY, tol=0.008, max_steps=60, chunk=6)
        return True

    def refine_bin(self):
        """Use the wrist view from above the bin to correct its centre (dark interior)."""
        o = self.rb.obs
        try:
            pts = o.points("wrist")
            rgb = o.rgb["wrist"]
            c = P.bin_candidates(rgb, pts, self.tz)
            best = None
            for k in c:
                d = np.linalg.norm(k["floor"][:2] - self.bin["center"][:2])
                if d < 0.08 and (best is None or k["area"] > best["area"]):
                    best = k
            if best is not None and best["area"] > 100:
                fl = best["floor"]
                old = self.bin["center"].copy()
                # walls from above surround the floor; floor centroid is the centre from above
                newc = 0.5 * old[:2] + 0.5 * fl[:2] if best["nwall"] < 5 else best["center"][:2]
                self.bin["center"][:2] = newc
                if best["nwall"] >= 5:
                    self.bin["top_z"] = max(self.bin["top_z"], best["top_z"]) if abs(best["top_z"] - self.bin["top_z"]) < 0.04 else self.bin["top_z"]
                log("bin refine: old", np.round(old, 3).tolist(), "floor", np.round(fl, 3).tolist(), "new", np.round(self.bin["center"], 3).tolist(), "area", best["area"])
                self.bin_refined = True
                return True
        except Exception as e:
            log("bin refine failed", e)
        return False

    def drop_in_bin(self, color):
        rb = self.rb
        self.mark("carry")
        b = self.bin
        if not getattr(self, "bin_refined", False):
            self.go_bin()
            self.refine_bin()
            self.go_bin()
            rel = np.array([b["center"][0], b["center"][1], b["top_z"] + RELEASE_ABOVE_BIN])
            self.move(rel, vmax=VMAX_NEAR, tol=0.006, max_steps=30, chunk=5)
        else:
            # v0.3: bin already refined -> one straight move to the release point (held cube stays
            # >= 0.03 above the rim: lift height z_grasp+0.10 down to bin top + RELEASE_ABOVE_BIN)
            rel = np.array([b["center"][0], b["center"][1], b["top_z"] + RELEASE_ABOVE_BIN])
            p = rb.obs.pos
            if p[2] < rel[2] - 0.005:
                self.move(np.array([p[0], p[1], rel[2] + 0.02]), vmax=VMAX_CARRY, tol=0.01, max_steps=15, chunk=5)
            self.move(rel, vmax=VMAX_CARRY, tol=0.007, max_steps=60, chunk=6)
        rb.save("prerelease")
        self.mark("release")
        self.hold(OPEN_STEPS, grip=-1.0)
        rb.save("released")
        o = rb.obs
        log("released", color, "grip", round(o.grip, 4), "pos", np.round(o.pos, 3).tolist())
        self.count[color] = self.count.get(color, 0) + 1
        log("COUNT", self.count, "goal", self.goal)
        self.mark("rise")
        up = o.pos.copy()
        up[2] = max(self.bin["top_z"], self.tz) + CARRY_CLEAR
        self.move(up, vmax=VMAX_FREE, tol=0.015, max_steps=10, chunk=5, grip=-1.0)

    def discard(self):
        """Drop a wrongly grasped cube onto the table away from the bin."""
        rb = self.rb
        p = rb.obs.pos.copy()
        log("discard at", np.round(p, 3).tolist())
        self.hold(OPEN_STEPS, grip=-1.0)
        p[2] += 0.03
        self.move(p, vmax=VMAX_FREE, tol=0.01, max_steps=10, chunk=5)

    def press_button(self):
        rb = self.rb
        btn = self.button
        if btn is None:
            o = rb.observe()
            self.bin_, btn = self.locate_static(o)
            self.button = btn
        if btn is None:
            log("no button found")
            return False
        top = btn["top_z"]
        above = np.array([btn["center"][0], btn["center"][1], top + 0.10])
        self.move(above, vmax=VMAX_FREE, tol=0.008, max_steps=60, chunk=6, grip=1.0)
        rb.save("above_button")
        # wrist refine of the button: bright low-saturation compact blob near the guess
        try:
            o = rb.obs
            pts = o.points("wrist")
            cands = P.button_candidates(o.rgb["wrist"], pts, self.tz, exclude_xy=[self.bin["center"]] if self.bin else ())
            near = [c for c in cands if np.linalg.norm(c["center"][:2] - btn["center"][:2]) < 0.06]
            if near:
                c = near[0]
                log("button refine", np.round(btn["center"], 3).tolist(), "->", np.round(c["center"], 3).tolist())
                btn["center"][:2] = c["center"][:2]
                top = max(top, c["top_z"]) if abs(c["top_z"] - top) < 0.03 else top
                btn["top_z"] = top
                above = np.array([btn["center"][0], btn["center"][1], top + 0.06])
                self.move(above, vmax=VMAX_NEAR, tol=0.005, max_steps=30, chunk=5, grip=1.0)
        except Exception as e:
            log("button refine failed", e)
        for depth in (0.015, 0.025, 0.035):
            tgt = np.array([btn["center"][0], btn["center"][1], top - depth])
            self.move(tgt, vmax=0.008, tol=0.004, max_steps=25, chunk=3, grip=1.0)
            rb.save("press")
            log("press depth", depth, "z", round(rb.obs.pos[2], 4), "complete", rb.complete)
            self.hold(4, grip=1.0)
            if rb.complete:
                return True
            self.move(above, vmax=VMAX_NEAR, tol=0.008, max_steps=15, chunk=5, grip=1.0)
            if rb.complete:
                return True
        return rb.complete

    def run(self):
        rb = self.rb
        self.init()
        self.hold(1, grip=-1.0)
        attempts = 0
        idle = 0
        while self.needed():
            if rb.remaining is not None and rb.remaining < 30:
                log("too few steps left", rb.remaining)
                break
            if time.time() - rb.t0 > 1080:
                log("wall time nearly out")
                break
            attempts += 1
            if attempts > 40:
                break
            try:
                t = self.scan()
                if t is None:
                    idle += 1
                    attempts -= 1
                    log("no target; waiting")
                    self.hold(10)
                    continue
                target, s0 = t
                res = self.grasp_cycle(target, s0)
                log("grasp result", res, "step", rb.obs.step_id, "remaining", rb.remaining, "wall", round(time.time() - rb.t0, 1))
                if res == "ok":
                    self.drop_in_bin(target["color"])
                elif res.startswith("wrong:"):
                    held = res.split(":", 1)[1]
                    if held in self.needed():
                        self.drop_in_bin(held)
                    else:
                        self.discard()
                elif res == "lost":
                    up = rb.obs.pos.copy()
                    up[2] = max(up[2], self.tz + 0.2)
                    self.move(up, tol=0.02, max_steps=10)
                else:  # miss
                    self.hold(6, grip=-1.0)
                    up = rb.obs.pos.copy()
                    up[2] = self.tz + 0.2
                    self.move(up, tol=0.02, max_steps=15, grip=-1.0)
            except EpisodeOver:
                return
            except Exception as e:
                log("ERROR", repr(e), traceback.format_exc())
                if rb.ended:
                    return
                try:
                    rb.observe()
                except Exception:
                    pass
        if not self.needed():
            try:
                self.press_button()
            except EpisodeOver:
                return
            except Exception as e:
                log("ERROR press", repr(e), traceback.format_exc())
        log("done: complete", rb.complete, "ended", rb.ended, "count", self.count, "steps", rb.obs.step_id, "wall", round(time.time() - rb.t0, 1))


def main(ctx):
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
    except Exception:
        pass
    c = Controller(ctx)
    try:
        c.run()
    except EpisodeOver:
        pass
    except Exception as e:
        log("FATAL", repr(e), traceback.format_exc())
    log("final: complete", c.rb.complete, "ended", c.rb.ended)
