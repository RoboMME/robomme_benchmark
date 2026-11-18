"""SwingXtimes controller (RAX, AfI, written without any simulator start).

Transition table (each phase returns ok / fail; fail paths listed):
  PERCEIVE   -> GRASP            | no cube: lift arm, re-perceive once, else STOP
  GRASP      -> SWING            | miss: re-perceive + retry (max 3), else STOP
  SWING[i]   -> SWING[i+1]       | drop detected: re-perceive, GRASP, resume SWING[i]
  SWING done -> PUTDOWN          |
  PUTDOWN    -> BUTTON (if the instruction mentions a button) | else STOP
  BUTTON     -> STOP
Every reply is checked for episode_ended / task_complete; either ends the run.
Wall guard: stop sending motion after WALL_LIMIT_S.
"""
import math
import time
import traceback

import numpy as np

import geom
import percept
import plan
from mcpio import log
from robot import Robot, EpisodeOver

VERSION = "swing-v1"

# Tunables (untested in the live simulator: R1 left no exploration start)
WALL_LIMIT_S = 1050.0
TRAVEL_CLEAR = 0.08     # cube bottom above table while travelling
DWELL_LOW = 0.015       # lowest cube-bottom clearance during a target visit
DWELL_STEPS = 4
APPROACH_H = 0.10       # TCP height above cube top before descending
GRASP_TRIES = 3
VMAX_FREE = 0.02        # m per control step in free space
VMAX_SLOW = 0.008       # near contact


class Ctl:
    def __init__(self, ctx):
        self.rb = Robot(ctx, WALL_LIMIT_S)
        self.proj = None
        self.api_uv_swapped = False
        self.tz = None
        self.cube = None
        self.cube_h = 0.04
        self.grasp_dz = 0.02
        self.home = None
        self.targets = {}
        self.target_top = {}
        self.target_blobs = {}
        self.fast = False
        self.ambiguous = True
        self.instr = {}
        self.button = None
        self.phase_t = {}
        self.max_obstacle_h = 0.0

    # ------------------------------------------------------------------ perception
    def calibrate(self, o):
        """Choose a back-projection convention for the front camera, checked against the API."""
        d = o.depth.get("front")
        K = o.K.get("front")
        pose = o.pose.get("front")
        if d is None:
            log("no front depth image available")
            return False
        H, W = d.shape
        rows, cols = [], []
        for r in np.linspace(20, H - 21, 6).astype(int):
            for c in np.linspace(20, W - 21, 6).astype(int):
                if np.isfinite(d[r, c]):
                    rows.append(int(r))
                    cols.append(int(c))
        rows, cols = rows[:32], cols[:32]
        api = []
        try:
            pts = self.rb.query([("front", c, r) for r, c in zip(rows, cols)], radius=0)
            api = [(r, c, p) for r, c, p in zip(rows, cols, pts) if p is not None]
        except Exception as e:
            log("pixel_to_position failed:", repr(e))
        log("api calibration samples:", len(api))
        if len(api) >= 6 and K is not None and pose is not None:
            pr, err = geom.fit_projector(d, K, pose, api)
            prs, errs = geom.fit_projector(d, K, pose, [(c, r, p) for r, c, p in api if r < W and c < H])
            log("projector fit err %.4f (%s); swapped-uv err %.4f" % (err, pr.name if pr else None, errs))
            if prs is not None and errs < err * 0.5:
                pr, err, self.api_uv_swapped = prs, errs, True
            if pr is not None and err < 0.02:
                self.proj = pr
                return True
            log("no convention matched the API within 2 cm")
            self.proj = None
            self._api_samples = api
            return False
        if K is not None and pose is not None:
            self.proj = geom.default_projector(K, pose)
            log("using default projector without API check:", self.proj.name if self.proj else None)
            return self.proj is not None
        return False

    def cloud(self, o):
        """World points for every front pixel (H,W,3), NaN where invalid."""
        d = o.depth.get("front")
        if self.proj is not None and d is not None:
            return self.proj.cloud(d)
        # API grid fallback, stride chosen so the grid stays well inside the 180 s progress limit
        rgb = o.rgb.get("front")
        H, W = (rgb.shape[:2] if rgb is not None else (256, 256))
        t = time.time()
        probe = self.rb.query([("front", W // 2, H // 2)], radius=0)
        dt = max(0.02, time.time() - t)
        stride = 4
        for s in (4, 6, 8, 12, 16):
            stride = s
            if ((H // s) * (W // s) / 32.0) * dt < 70.0:
                break
        P = np.full((H, W, 3), np.nan)
        rr, cc = np.mgrid[stride // 2:H:stride, stride // 2:W:stride]
        sels = [("front", int(c), int(r)) if not self.api_uv_swapped else ("front", int(r), int(c))
                for r, c in zip(rr.ravel(), cc.ravel())]
        t = time.time()
        pts = self.rb.query(sels, radius=0)
        log("API grid cloud: stride %d, %d points in %.1fs (probe %.2fs)" % (stride, len(pts), time.time() - t, dt))
        h2 = stride // 2
        for (r, c), p in zip(zip(rr.ravel(), cc.ravel()), pts):
            if p is not None:
                P[max(0, r - h2):r - h2 + stride, max(0, c - h2):c - h2 + stride] = p
        return P

    def perceive(self, o=None):
        try:
            return self._perceive(o)
        except EpisodeOver:
            raise
        except Exception:
            import traceback as _tb
            log("perception failed:\n" + _tb.format_exc())
            return None

    def _perceive(self, o=None):
        if o is None:
            o = self.rb.observe(images=True)
        rgb = o.rgb.get("front")
        if rgb is None:
            log("no front rgb")
            return None
        if self.proj is None and o.depth.get("front") is not None and not hasattr(self, "_calib_tried"):
            self._calib_tried = True
            self.calibrate(o)
        P = self.cloud(o)
        sc = percept.analyse(rgb, P, o.pos, log=log)
        sc["P"] = P
        sc["rgb"] = rgb
        log("scene: table_z %.4f, %d sat blobs, %d elevated, %d flat" % (
            sc["table_z"], len(sc["cubes"]), len(sc["elevated"]), len(sc["targets"])))
        for b in sc["cubes"][:8]:
            log("  sat blob n=%d c=%s top_h=%.3f hue=%.0f sat=%.2f ext=%.3f ring_table=%.2f uv=%s" % (
                b.n, np.round(b.center, 3).tolist(), b.top_h, b.hue, b.sat, b.ext, b.ring_table,
                np.round(b.uv, 0).tolist()))
        for b in sc["elevated"][:8]:
            log("  elev blob n=%d c=%s top_h=%.3f ext=%.3f sat=%.2f" % (
                b.n, np.round(b.center, 3).tolist(), b.top_h, b.ext, b.sat))
        for b in sorted(sc["targets"], key=lambda x: -x.n)[:8]:
            log("  flat n=%d c=%s ext=%.3f elong=%.2f hue=%.0f sat=%.2f val=%.2f parts=%d uv=%s" % (
                b.n, np.round(b.center, 3).tolist(), b.ext, b.elong, b.hue, b.sat, b.val, b.parts,
                np.round(b.uv, 0).tolist()))
        return sc

    def cam_right(self, sc):
        if self.proj is not None:
            v = self.proj.cam_right_world()
            return v / (np.linalg.norm(v) + 1e-9)
        P = sc["P"]
        dif = P[:, 8:, :] - P[:, :-8, :]
        dif = dif[np.all(np.isfinite(dif), axis=-1)]
        if len(dif) == 0:
            return np.array([0.0, 1.0, 0.0])
        v = np.median(dif, axis=0)
        v[2] = 0
        return v / (np.linalg.norm(v) + 1e-9)

    def locate(self, first=False):
        sc = self.perceive()
        if sc is None:
            return None
        self.tz = sc["table_z"] if self.tz is None or first else self.tz
        cands = percept.pick_cube(sc, self.instr.get("color"), exclude=None)
        if not cands:
            return sc
        s, b = cands[0]
        log("cube pick: score %.2f center %s top_h %.3f yaw %.1f sides %s" % (
            s, np.round(b.center, 3).tolist(), b.top_h, math.degrees(b.yaw), np.round(b.sides, 3).tolist()))
        self.cube = b
        self.cube_h = float(np.clip(b.top_h, 0.015, 0.08))
        return sc

    def robot_forward(self, sc):
        """Robot facing direction in the world xy plane, from where the arm body is seen
        relative to the hand (the arm rises from its base behind the hand). Defaults to +x."""
        try:
            P = sc["P"]
            tcp = self.rb.obs.pos
            ok = np.all(np.isfinite(P), axis=-1)
            pts = P[ok]
            h = pts[:, 2] - sc["table_z"]
            dxy = np.linalg.norm(pts[:, :2] - tcp[:2], axis=1)
            d3 = np.linalg.norm(pts - tcp, axis=1)
            sel = (h > 0.18) & (h < 1.2) & (dxy < 0.9) & (d3 > 0.12)
            if sel.sum() > 30:
                v = tcp[:2] - np.median(pts[sel][:, :2], axis=0)
                if np.linalg.norm(v) > 0.06:
                    v = v / np.linalg.norm(v)
                    log("robot forward estimate from %d arm points: %s" % (int(sel.sum()), np.round(v, 2).tolist()))
                    if np.dot(v, [1.0, 0.0]) > 0.7:
                        return np.array([v[0], v[1]])
                    log("estimate far from +x; keeping the +x prior")
        except Exception as e:
            log("robot_forward failed:", repr(e))
        return np.array([1.0, 0.0])

    def find_targets(self, sc):
        avoid = [self.cube.center] if self.cube is not None else []
        pair, cands = percept.pick_targets(sc, avoid)
        raised = False
        if pair is None:
            log("flat targets not found (%d flat candidates); trying raised objects" % len(cands))
            pair = percept.pick_targets_elevated(sc, avoid)
            raised = pair is not None
        if pair is None:
            log("targets not found")
            return False
        a, b = pair
        right_img = a if a.uv[0] > b.uv[0] else b
        left_img = b if right_img is a else a
        cr = self.cam_right(sc)
        fwd = self.robot_forward(sc)
        robot_right = np.array([fwd[1], -fwd[0], 0.0])
        agree = float(np.dot(cr, robot_right))
        self.ambiguous = agree < 0.5
        new = {"right": right_img, "left": left_img}
        for side, blob in new.items():
            old = self.target_blobs.get(side)
            if old is not None and np.linalg.norm(old.center[:2] - blob.center[:2]) >= 0.10:
                log("target %s: new view disagrees by %.3f m; keeping earlier view" % (
                    side, float(np.linalg.norm(old.center[:2] - blob.center[:2]))))
                continue
            if old is not None and old.n >= blob.n:
                log("target %s: keeping earlier view (n %d >= %d), shift %.3f" % (
                    side, old.n, blob.n, float(np.linalg.norm(old.center[:2] - blob.center[:2]))))
                continue
            if old is not None:
                log("target %s: updated (n %d -> %d), shift %.3f" % (
                    side, old.n, blob.n, float(np.linalg.norm(old.center[:2] - blob.center[:2]))))
            self.target_blobs[side] = blob
            self.targets[side] = blob.center.copy()
            self.target_top[side] = float(blob.top_h or 0.0) if raised else 0.0
        self.target_ext = max(a.ext, b.ext)
        log("targets: right(img)=%s left(img)=%s cam_right=%s dot(robot_right)=%.2f ambiguous=%s" % (
            np.round(right_img.center, 3).tolist(), np.round(left_img.center, 3).tolist(),
            np.round(cr, 2).tolist(), agree, self.ambiguous))
        return True

    def find_button(self, sc):
        tx = list(self.targets.values())
        cxy = self.cube.center if self.cube is not None else np.zeros(3)
        c = percept.pick_button(sc, cxy, tx)
        if c:
            self.button = c[0][1]
            log("button: center %s top_h %.3f ext %.3f" % (np.round(self.button.center, 3).tolist(),
                                                          self.button.top_h, self.button.ext))
        else:
            log("button not found")

    # ------------------------------------------------------------------ motion helpers
    def z_for_bottom(self, clear):
        """TCP z that puts the held cube's bottom `clear` above the table."""
        return self.tz + self.grasp_dz + clear

    def held(self):
        g = self.rb.obs.grip
        return g > max(0.006, 0.12 * self.rb.open_sum), g

    def safe_up(self, h=0.15):
        p = self.rb.obs.pos.copy()
        p[2] = max(p[2], self.tz + h) if self.tz is not None else p[2] + 0.05
        self.rb.move_to(p, tol=0.01, vmax=VMAX_FREE, max_iter=3, label="up")

    # ------------------------------------------------------------------ phases
    def grasp(self):
        b = self.cube
        tz = self.tz
        c = b.center.copy()
        top = tz + self.cube_h
        gz = tz + max(self.cube_h - 0.02, self.cube_h * 0.5, 0.012)
        self.grasp_dz = gz - tz
        self.rb.grip_cmd = -1.0
        above = np.array([c[0], c[1], top + APPROACH_H])
        ok, d = self.rb.move_to(above, tol=0.008, vmax=VMAX_FREE, label="above-cube")
        log("above cube ok=%s err=%.3f" % (ok, d))
        # align yaw (cube symmetric mod 90 deg)
        R = geom.rotvec_to_R(self.rb.obs.rotvec)
        dyaw = geom.wrap(b.yaw - geom.tool_yaw(R), math.pi / 2)
        sides_ok = b.sides and min(b.sides) > 0.015 and max(b.sides) / max(min(b.sides), 1e-3) < 1.6
        if abs(dyaw) > math.radians(12) and sides_ok and self.rb.tilt_deg() < 20:
            okr = self.rb.rotate_yaw(dyaw)
            log("yaw align %.1f deg ok=%s tilt %.1f" % (math.degrees(dyaw), okr, self.rb.tilt_deg()))
            if self.rb.tilt_deg() > 15:
                log("tilt grew after rotation; restoring the initial orientation")
                self.rb.rot_broken = False
                self.rb.rotate_to(self.R0.copy())
                self.rb.rot_broken = True
            self.rb.R_des = geom.rotvec_to_R(self.rb.obs.rotvec) if self.rb.tilt_deg() < 15 else self.R0.copy()
        self.rb.gripper(-1.0, 6)
        self.rb.open_sum = max(self.rb.obs.grip, 0.02)
        pre = np.array([c[0], c[1], gz + 0.04])
        self.rb.move_to(pre, tol=0.006, vmax=VMAX_FREE, label="pre-grasp")
        ok, d = self.rb.move_to(np.array([c[0], c[1], gz]), tol=0.005, vmax=VMAX_SLOW, label="grasp-z")
        log("grasp descent ok=%s err=%.3f pos=%s" % (ok, d, np.round(self.rb.obs.pos, 3).tolist()))
        g = self.rb.gripper(1.0, 16)
        lift = np.array([c[0], c[1], self.z_for_bottom(TRAVEL_CLEAR)])
        self.rb.move_to(lift, tol=0.01, vmax=0.012, label="lift")
        h, g2 = self.held()
        log("after close grip %.4f, after lift %.4f, open_sum %.4f -> held=%s" % (g, g2, self.rb.open_sum, h))
        return h

    def visit(self, side):
        t = self.targets[side]
        tz = self.tz
        th = self.target_top.get(side, 0.0)
        z_tr = self.z_for_bottom(max(self.travel_clear, th + 0.05))
        p = self.rb.obs.pos
        # rise first if low, then travel horizontally
        if p[2] < z_tr - 0.02:
            self.rb.move_to(np.array([p[0], p[1], z_tr]), tol=0.01, vmax=0.015, label="rise")
        fast = self.fast
        self.rb.move_to(np.array([t[0], t[1], z_tr]), tol=0.008, vmax=0.02 if fast else 0.015, label="to-" + side)
        self.rb.hold(2 if fast else DWELL_STEPS)
        low = DWELL_LOW + th + (0.02 if fast else 0.0)
        self.rb.move_to(np.array([t[0], t[1], self.z_for_bottom(low)]), tol=0.006, vmax=0.01 if fast else 0.008,
                        label="dip-" + side)
        self.rb.hold(2 if fast else DWELL_STEPS)
        xy_err = float(np.linalg.norm(self.rb.obs.pos[:2] - t[:2]))
        self.rb.move_to(np.array([t[0], t[1], z_tr]), tol=0.01, vmax=0.015 if fast else 0.012, label="up-" + side)
        if self.rb.orient_err_deg(self.rb.R_des) > 8.0:
            log("orientation drift %.1f deg; restoring" % self.rb.orient_err_deg(self.rb.R_des))
            self.rb.rotate_to(self.rb.R_des.copy())
        h, g = self.held()
        log("visit %s done xy_err %.3f grip %.4f held=%s step %s" % (side, xy_err, g, h, self.rb.obs.step_id))
        return h

    def put_down(self, xy):
        z_tr = self.z_for_bottom(self.travel_clear)
        self.rb.move_to(np.array([xy[0], xy[1], z_tr]), tol=0.008, vmax=0.015, label="to-home")
        self.rb.move_to(np.array([xy[0], xy[1], self.z_for_bottom(0.008)]), tol=0.005, vmax=0.006,
                        label="lower")
        self.rb.gripper(-1.0, 10)
        p = self.rb.obs.pos
        self.rb.move_to(np.array([p[0], p[1], self.tz + self.grasp_dz + 0.12]), tol=0.01, vmax=0.015,
                        label="retreat")
        log("put down at %s" % np.round(xy, 3).tolist())

    def press_button(self):
        b = self.button
        top = self.tz + b.top_h
        c = b.center
        self.rb.gripper(1.0, 8)
        self.rb.move_to(np.array([c[0], c[1], top + 0.08]), tol=0.008, vmax=VMAX_FREE, label="above-button")
        self.rb.move_to(np.array([c[0], c[1], top + 0.012]), tol=0.005, vmax=VMAX_SLOW, label="touch-button")
        for depth in (0.012, 0.025, 0.04):
            if self.rb.complete or self.rb.ended:
                break
            self.rb.move_to(np.array([c[0], c[1], top - depth]), tol=0.004, vmax=0.004, max_iter=3,
                            label="press")
            self.rb.hold(4)
            log("press depth %.3f pos z %.3f complete=%s" % (depth, self.rb.obs.pos[2], self.rb.complete))
            self.rb.move_to(np.array([c[0], c[1], top + 0.03]), tol=0.008, vmax=0.01, max_iter=2,
                            label="release-button")
            self.rb.hold(4)
            if self.rb.complete or self.rb.ended:
                break

    # ------------------------------------------------------------------ main pipeline
    def run(self):
        rb = self.rb
        o = rb.observe(images=True)
        log("instruction:", repr(o.instruction))
        log("state:", None if o.state is None else np.round(o.state, 4).tolist(), "step", o.step_id,
            "remaining", o.remaining, "cams rgb", list(o.rgb), "depth", list(o.depth),
            "K", list(o.K), "poses", list(o.pose))
        if o.manifest:
            fr = o.manifest.get("frames", [])
            log("manifest frames:", len(fr), "video frames:", sum(1 for f in fr if f.get("phase") == "video_demo"))
        if o.state is None:
            log("no robot state; cannot control")
            return
        self.instr = plan.parse_instruction(o.instruction)
        log("parsed instruction:", {k: v for k, v in self.instr.items() if k != "text"})
        self.R0 = geom.rotvec_to_R(o.rotvec)
        rb.R_des = self.R0.copy()
        log("initial tilt %.1f deg, tool yaw %.1f" % (rb.tilt_deg(), math.degrees(geom.tool_yaw(self.R0))))

        sc = self.perceive(o)
        if sc is None:
            return
        self.tz = sc["table_z"]
        sc = self._locate_from(sc)
        if self.cube is None or (self.cube.ring_table or 0) < 0.6:
            # no cube, or the best candidate is not surrounded by table (could be a button cap
            # or partly hidden by the arm): raise the arm and look again before committing
            log("cube uncertain (%s); raising arm and re-perceiving" % (
                None if self.cube is None else "ring_table %.2f" % self.cube.ring_table))
            prev_cube, prev_sc = self.cube, sc
            prev_s = percept.cube_score(prev_cube, self.instr.get("color")) if prev_cube is not None else None
            self.safe_up(0.30)
            sc2 = self.perceive()
            if sc2 is not None:
                self.cube = None
                self._locate_from(sc2)
                new_s = percept.cube_score(self.cube, self.instr.get("color")) if self.cube is not None else None
                if self.cube is not None and (prev_s is None or new_s <= prev_s + 1e-6):
                    sc = sc2
                else:
                    self.cube = prev_cube
                    if prev_cube is not None:
                        self.cube_h = float(np.clip(prev_cube.top_h, 0.015, 0.08))
                    sc = prev_sc
            log("cube after re-perception: %s" % (None if self.cube is None else np.round(self.cube.center, 3).tolist()))
        if self.cube is None:
            log("STOP: no cube found")
            return
        self.home = self.cube.center.copy()
        have_t = self.find_targets(sc)
        if self.instr.get("button"):
            self.find_button(sc)
        tall = [e.top_h for e in sc["elevated"] if e.ext < 0.2 and np.linalg.norm(e.center[:2] - self.home[:2]) > 0.04]
        self.max_obstacle_h = max(tall) if tall else 0.0
        self.travel_clear = float(np.clip(max(TRAVEL_CLEAR, self.max_obstacle_h + 0.03), TRAVEL_CLEAR, 0.18))

        # ---------------- grasp
        held = False
        for k in range(GRASP_TRIES):
            held = self.grasp()
            if held:
                break
            log("grasp miss %d; re-perceive" % k)
            rb.gripper(-1.0, 6)
            self.safe_up(0.25)
            sc2 = self.locate()
            if self.cube is None:
                break
        if not held:
            log("STOP: grasp failed")
            return

        # second view of the targets with the arm in another pose (occlusion check)
        sc3 = self.perceive()
        if sc3 is not None:
            have_t = self.find_targets(sc3) or have_t
            if have_t:
                if self.instr.get("button") and self.button is None:
                    self.find_button(sc3)
        if not have_t:
            cr = self.cam_right(sc)
            self.targets = {"right": self.home + 0.15 * cr, "left": self.home - 0.15 * cr}
            log("FALLBACK targets around the cube start", self.targets)

        count = self.instr.get("count")
        if count is None:
            count = plan.count_from_video(o.manifest, log)
            log("count from conditioning video:", count)
        seq = plan.visit_sequence(self.instr.get("first"), count, self.ambiguous)
        rem = rb.obs.remaining
        need = 52 * len(seq) + 140
        if rem is not None and rem < 1.15 * need:
            self.fast = True
        log("visit sequence:", seq, "remaining steps", rem, "estimated need", need, "fast", self.fast)
        i = 0
        drops = 0
        while i < len(seq):
            if self.visit(seq[i]):
                i += 1
                continue
            drops += 1
            log("cube lost during visit %d (%s); recovery %d" % (i, seq[i], drops))
            if drops > 2:
                log("STOP: too many drops")
                return
            rb.gripper(-1.0, 6)
            self.safe_up(0.25)
            self.locate()
            if self.cube is None or not self.grasp():
                log("STOP: regrasp failed")
                return
        # ---------------- put down
        final_xy = self.home[:2]
        fs = self.instr.get("final_side")
        if fs in self.targets:
            final_xy = self.targets[fs][:2]
        self.put_down(final_xy)
        if self.instr.get("button"):
            if self.button is None:
                sc4 = self.perceive()
                if sc4 is not None:
                    self.find_button(sc4)
            if self.button is not None:
                self.press_button()
        log("pipeline finished: complete=%s ended=%s step=%s" % (rb.complete, rb.ended, rb.obs.step_id))

    def _locate_from(self, sc):
        cands = percept.pick_cube(sc, self.instr.get("color"))
        if cands:
            s, b = cands[0]
            self.cube = b
            self.cube_h = float(np.clip(b.top_h, 0.015, 0.08))
            log("cube pick: score %.2f center %s top_h %.3f yaw %.1f sides %s (of %d)" % (
                s, np.round(b.center, 3).tolist(), b.top_h, math.degrees(b.yaw),
                np.round(b.sides, 3).tolist(), len(cands)))
        return sc


def main(ctx):
    log("controller", VERSION)
    c = None
    try:
        c = Ctl(ctx)
        c.run()
    except EpisodeOver as e:
        log("episode over:", repr(e))
    except Exception:
        log("EXCEPTION in controller:\n" + traceback.format_exc())
        # one safe fallback: if holding the cube mid-air, keep it; nothing else to do
    finally:
        try:
            if c is not None:
                log("final: complete=%s ended=%s steps_sent=%d calls=%d wall=%.1fs" % (
                    c.rb.complete, c.rb.ended, c.rb.steps_sent, c.rb.calls, time.time() - c.rb.t0))
        except Exception:
            pass
