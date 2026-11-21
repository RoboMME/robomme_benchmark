"""PickXtimes controller (RAX, demo-only offline-developed).

Task: pick the named cube, place it on the purple target, repeat N times, then press
the white button.

Pipeline / transition table
---------------------------
INIT      observe, parse instruction (colour, N), localise cube/target/button/table
          -> PROBE  (failure to find cube/target: retry perception from wrist after
                     moving up; then continue with what exists)
PROBE     small up / yaw+tilt / xy motions; infer translation signs and the rotation
          composition mode -> GRASP (always; defaults kept when ambiguous)
GRASP     hover over cube, refine with wrist camera, align yaw, descend, close,
          check fingers, lift, check hold.  ok -> PLACE ; fail -> GRASP (re-perceive,
          max attempts) ; exhausted -> PRESS (nothing better left)
PLACE     carry to target, descend, open, retreat, verify cube on target from wrist
          image.  verified -> count+=1 ; count==N -> PRESS else GRASP
          not on target -> GRASP (cube re-localised, count unchanged)
PRESS     close gripper, hover over button, refine, push down in small chunks until
          the episode ends; retries with small xy offsets.
DONE      episode ended / out of time.
"""
import math
import os
import re
import time
import traceback
import json

import numpy as np

import perception as P
from geom import rotvec_to_mat, mat_to_rotvec, rotz, wrap_half_pi, wrap_pi
from rax_io import RobotIO

VERSION = "pickx-v1.1"

# ---------------------------------------------------------------- tunables (untested live:
# demo-only, zero simulator starts; values chosen from demo appearance and Panda geometry)
POS_SCALE = 0.05          # m per unit action (documented)
ROT_SCALE = 0.5           # rad per unit action (documented)
V_FREE = 0.015            # m/step in free space
V_CARRY = 0.012           # m/step with cube
V_NEAR = 0.006            # m/step near contact
ROT_STEP_MAX = 0.10       # rad/step
HOVER_DZ = 0.15           # TCP height above table for wrist perception / travel
LIFT_DZ = 0.22            # TCP height above table when lifting the cube (expert lifts ~0.2 m, demo wrist ring scale)
RETREAT_DZ = 0.07         # TCP height above grasp height after release
PLACE_EXTRA = 0.006       # release this much above resting height
ON_TARGET_TOL = 0.025     # verified placement tolerance (m, xy)
ON_TARGET_FAIL = 0.045    # seen this far from target centre -> placement failed
MAX_GRASP_ATTEMPTS = 6
BUTTON_STEP_RESERVE = 100  # steps kept for the final button press (if the step limit is reported)
BUTTON_TIME_RESERVE_S = 90.0
WALL_LIMIT_S = 1120.0     # controller wall budget (runtime limit 1200 s)
GRIP_OPEN, GRIP_CLOSE = -1.0, 1.0

NUM_WORDS = {"one": 1, "once": 1, "two": 2, "twice": 2, "three": 3, "thrice": 3, "four": 4,
             "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
             "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
             "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20}


def parse_instruction(text):
    t = (text or "").lower()
    color = None
    m = re.search(r"pick up the (\w+) cube", t)
    if m and m.group(1) in P.COLORS:
        color = m.group(1)
    if color is None:
        for c in P.COLORS:
            if c in t:
                color = c
                break
    n = None
    m = re.search(r"(\w+)\s+times?\b", t)
    for m in re.finditer(r"(\b[\w-]+)\s+times?\b", t):
        w = m.group(1)
        if w.isdigit():
            n = int(w)
        elif w in NUM_WORDS:
            n = NUM_WORDS[w]
    if n is None:
        for w in ("once", "twice", "thrice"):
            if re.search(r"\b" + w + r"\b", t):
                n = NUM_WORDS[w]
    return color, n


class Abort(Exception):
    pass


class Controller:
    def __init__(self, ctx):
        self.t0 = time.time()
        os.makedirs("/workspace/output", exist_ok=True)
        self.logf = open("/workspace/output/pickx_log.txt", "a")
        self.io = RobotIO(ctx, self.log)
        self.obs = None
        self.axis_sign = np.ones(3)
        self.rot_mode = "world"
        self.rot_diverge = 0
        self.verbose = True
        self.gain = 1.0   # measured tracking fraction per step (compensated, 0.55..1.0)
        self.R_des = None
        self.count = 0
        self.N = None
        self.color = "green"
        self.table_z = None
        self.cube = None        # dict(xy, top_z, half)
        self.target = None      # np xy z
        self.button = None      # dict(xy, top_z)
        self.others = []        # xy of other objects
        self.grasp_dz = None    # TCP z above table at grasp
        self.step_state_ok = None
        self.state_checks = 0
        self.saved_imgs = 0

    # ------------------------------------------------------------------ logging
    def log(self, msg):
        line = f"[{VERSION} t={time.time() - self.t0:7.1f}s steps={getattr(getattr(self, 'io', None), 'steps_sent', 0)}] {msg}"
        try:
            print(line, flush=True)
        except Exception:
            pass
        try:
            self.logf.write(line + "\n")
            self.logf.flush()
        except Exception:
            pass

    def save_img(self, name, arr):
        if self.saved_imgs > 60 or arr is None:
            return
        try:
            from PIL import Image
            Image.fromarray(arr).save(f"/workspace/output/{name}.png")
            self.saved_imgs += 1
        except Exception:
            pass

    def time_left(self):
        return WALL_LIMIT_S - (time.time() - self.t0)

    def check_alive(self):
        if self.io.ended:
            raise Abort("episode ended")
        if self.time_left() < 0:
            raise Abort("wall budget exhausted")

    # ------------------------------------------------------------------ basic I/O
    def observe(self):
        o = self.io.observe()
        if o.state is None:
            raise RuntimeError("observation without state")
        self.obs = o
        return o

    def send(self, rows):
        """Send rows, then refresh self.obs (from the step reply when proven reliable)."""
        self.check_alive()
        res = self.io.step(rows)
        got = None
        if res.ok and self.step_state_ok is not False:
            try:
                cand = self.io._parse_obs(res.meta, [])
                if cand.state is not None:
                    got = cand
            except Exception:
                got = None
        if got is not None and self.step_state_ok:
            # trusted; keep previous images (not needed for pure motion)
            got.rgb, got.depth = {}, {}
            self.obs = got
        else:
            o = self.observe()
            if got is not None and self.step_state_ok is None:
                d = np.abs(np.array(got.state) - np.array(o.state)).max()
                self.state_checks += 1
                if d > 2e-4:
                    self.step_state_ok = False
                    self.log(f"step-reply state differs from observe ({d:.4f}); always observing")
                elif self.state_checks >= 3:
                    self.step_state_ok = True
                    self.log("step-reply state matches observe; trusting step replies")
        if res.ended or self.io.ended:
            self.io.ended = True
        return res

    # ------------------------------------------------------------------ rotation helpers
    def R_cur(self):
        return rotvec_to_mat(self.obs.rotvec)

    def rot_command(self, R_des):
        """Total rotation command vector (rad, before scaling) that moves R_cur to R_des."""
        Rc = self.R_cur()
        if self.rot_mode.startswith("world"):
            e = mat_to_rotvec(R_des @ Rc.T)
        else:
            e = mat_to_rotvec(Rc.T @ R_des)
        if self.rot_mode.endswith("neg"):
            e = -e
        return e

    def rot_err(self, R_des):
        return float(np.linalg.norm(mat_to_rotvec(R_des @ self.R_cur().T)))

    def yaw_of(self, R):
        """Heading of the tool x axis projected on the table."""
        x = R[:, 0]
        return math.atan2(x[1], x[0])

    # ------------------------------------------------------------------ motion
    def move_to(self, target, vmax=V_FREE, grip=GRIP_OPEN, tol=0.004, max_iter=14, chunk=10,
                R_des="keep", rtol=0.03, label=""):
        target = np.asarray(target, float)
        if isinstance(R_des, str):
            R_des = self.R_des
        stall = 0
        prev = None
        for it in range(max_iter):
            self.check_alive()
            pos = self.obs.pos
            err = target - pos
            dist = float(np.linalg.norm(err))
            rerr = self.rot_err(R_des) if R_des is not None else 0.0
            if dist < tol and rerr < rtol:
                if self.verbose:
                    self.log(f"move_to[{label}] ok it={it} pos {np.round(pos, 4)} rerr {rerr:.3f}")
                return True
            # steady-state acceptance: close enough and no longer improving
            if it >= 2 and dist < 2.5 * tol and rerr < 0.10 and prev is not None and \
                    prev[0] - dist < 0.001 and prev[1] - rerr < 0.01:
                if self.verbose:
                    self.log(f"move_to[{label}] accepted (no progress) it={it} err {dist:.4f} rerr {rerr:.3f}")
                return True
            prev = (dist, rerr)
            n = int(min(chunk, max(1, math.ceil(dist / vmax))))
            if R_des is not None and rerr > rtol:
                n = int(min(chunk, max(n, math.ceil(rerr / ROT_STEP_MAX))))
            gu = self.gain if vmax >= V_CARRY else 1.0   # no compensation near contact
            d = err / n / POS_SCALE * self.axis_sign / gu
            lim = vmax / POS_SCALE * 1.2 / gu
            d = np.clip(d, -lim, lim)
            r = np.zeros(3)
            if R_des is not None and rerr > rtol * 0.5:
                r = self.rot_command(R_des) / n / ROT_SCALE
                r = np.clip(r, -ROT_STEP_MAX / ROT_SCALE, ROT_STEP_MAX / ROT_SCALE)
            row = [float(d[0]), float(d[1]), float(d[2]), float(r[0]), float(r[1]), float(r[2]), grip]
            before = pos.copy()
            self.send([row] * n)
            if R_des is not None and rerr > 0.06 and np.linalg.norm(r) > 0:
                rerr2 = self.rot_err(R_des)
                if rerr2 > rerr + 0.02:
                    self.rot_diverge += 1
                    if self.rot_diverge >= 2:
                        modes = ["world", "body", "world_neg", "body_neg"]
                        new = modes[(modes.index(self.rot_mode) + 1) % 4]
                        self.log(f"rotation error grew {rerr:.3f}->{rerr2:.3f}; switching rot mode {self.rot_mode}->{new}")
                        self.rot_mode = new
                        self.rot_diverge = 0
                else:
                    self.rot_diverge = 0
            moved = float(np.linalg.norm(self.obs.pos - before))
            expected = min(dist, n * vmax)
            cmd = float(np.linalg.norm(d * gu)) * POS_SCALE * n   # metres asked (before compensation)
            if cmd > 0.015 and n >= 3 and (gu != 1.0 or self.gain == 1.0):   # learn only from compensated (fast) moves
                ratio = moved / max(cmd, 1e-6)
                if 0.2 < ratio < 1.5:
                    # gain = achieved fraction of the compensated command * current gain
                    est = min(1.0, max(0.55, ratio * gu))
                    self.gain = 0.7 * self.gain + 0.3 * est
            if dist > 3 * tol and moved < 0.2 * expected:
                stall += 1
                if stall >= 2:
                    self.log(f"move_to[{label}] stalled at {np.round(self.obs.pos, 4)} target {np.round(target, 4)}")
                    return False
            else:
                stall = 0
        ok = float(np.linalg.norm(target - self.obs.pos)) < 3 * tol
        if not ok:
            self.log(f"move_to[{label}] not converged: pos {np.round(self.obs.pos, 4)} target {np.round(target, 4)}")
        return ok

    def hold(self, n, grip):
        self.send([[0, 0, 0, 0, 0, 0, grip]] * n)

    def gripper(self, grip, min_steps=6, max_steps=30):
        """Open/close and wait until the finger opening stops changing."""
        total = 0
        prev = self.obs.fingers
        while total < max_steps:
            k = 4 if total >= min_steps else min_steps
            self.hold(k, grip)
            total += k
            f = self.obs.fingers
            if total >= min_steps and abs(f - prev) < 0.0008:
                break
            if grip < 0 and f > 0.075:
                break
            prev = f
        return self.obs.fingers

    # ------------------------------------------------------------------ perception
    def table_points(self, o, cam, excl):
        img = o.rgb.get(cam)
        if img is None:
            return []
        H, W = img.shape[:2]
        sels = []
        if cam == "front":
            us, vs = np.linspace(30, W - 30, 6), np.linspace(H * 0.45, H - 20, 4)
        else:
            us, vs = np.linspace(30, W - 30, 6), np.linspace(H * 0.25, H * 0.7, 4)
        for v in vs:
            for u in us:
                if not excl[int(v), int(u)]:
                    sels.append((cam, u, v))
        pts = self.io.px2world(o.step_id, sels[:24], radius=1)
        return [p for p in pts if p is not None]

    def blob_points(self, o, cam, mask, n=20):
        sels = [(cam, u, v) for u, v in P.interior_samples(mask, n=n, erode=1)]
        pts = self.io.px2world(o.step_id, sels, radius=0)
        return [p for p in pts if p is not None]

    def top_center(self, pts, band=0.008):
        if not pts:
            return None, None
        A = np.array(pts)
        zmax = np.percentile(A[:, 2], 90)
        top = A[A[:, 2] > zmax - band]
        return top[:, :2].mean(0), float(np.median(top[:, 2]))

    def all_masks(self, img):
        ex = np.zeros(img.shape[:2], bool)
        for c in P.COLORS:
            ex |= P.color_mask(img, c).astype(bool)
        ex |= P.target_mask(img).astype(bool)
        ex |= P.white_mask(img).astype(bool)
        return ex

    def perceive_scene(self, o):
        """Localise cube, target, button, other cubes and table from both cameras."""
        res = {"cube": [], "target": [], "button": [], "table": [], "others": []}
        for cam in ("front", "wrist"):
            img = o.rgb.get(cam)
            if img is None:
                continue
            excl = self.all_masks(img)
            excl = np.asarray(excl) | (np.asarray(__import__("cv2").dilate(excl.astype(np.uint8), np.ones((7, 7), np.uint8))) > 0)
            res["table"] += [(cam, p) for p in self.table_points(o, cam, excl)]
            cb = [b for b in P.find_color_blobs(img, self.color) if b["area"] >= 6]
            if cb:
                pts = self.blob_points(o, cam, cb[0]["mask"], n=16)
                xy, tz = self.top_center(pts)
                if xy is not None:
                    res["cube"].append((cam, xy, tz, cb[0]["area"]))
            for c in P.COLORS:
                if c == self.color:
                    continue
                for b in P.find_color_blobs(img, c)[:2]:
                    if b["area"] < 6:
                        continue
                    pts = self.blob_points(o, cam, b["mask"], n=6)
                    xy, tz = self.top_center(pts)
                    if xy is not None:
                        res["others"].append((cam, c, xy))
            tb = P.find_target_blobs(img)
            if tb and tb[0]["area"] >= 8:
                b = tb[0]
                pts = self.io.px2world(o.step_id, [(cam, b["cx"], b["cy"])] +
                                       [(cam, u, v) for u, v in P.interior_samples(b["mask"], n=10, erode=0)], radius=0)
                pts = [p for p in pts if p is not None]
                if pts:
                    A = np.array(pts)
                    res["target"].append((cam, np.median(A[:, :2], 0), float(np.median(A[:, 2])), b["area"]))
            tmask = np.zeros(img.shape[:2], np.uint8)
            for b in tb[:1]:
                x, y, w, h = b["bbox"]
                tmask[max(0, y - 4):y + h + 4, max(0, x - 4):x + w + 4] = 1
            bb = P.find_button_blobs(img, exclude=tmask)
            cands = []
            for b in bb:
                if b["area"] < 8 or b["aspect"] > 3.0:
                    continue
                cands.append(b)
            cands.sort(key=lambda b: -(b["area"] + 3 * b["hole"]))
            for b in cands[:3]:
                sels = [(cam, b["fcx"], b["fcy"])] + [(cam, u, v) for u, v in P.interior_samples(b["filled"], n=10, erode=0)]
                pts = [p for p in self.io.px2world(o.step_id, sels, radius=0) if p is not None]
                if pts:
                    A = np.array(pts)
                    res["button"].append((cam, A[0, :2] if pts[0] is not None else np.median(A[:, :2], 0),
                                          float(np.max(A[:, 2])), b["area"], b["hole"]))
        return res

    def fuse_scene(self, res):
        tz = [p[1][2] for p in res["table"]]
        if tz:
            self.table_z = float(np.median(tz))
        self.log(f"table z {self.table_z} from {len(tz)} pts (spread {np.std(tz) if tz else -1:.4f})")
        tz0 = self.table_z if self.table_z is not None else 0.0

        def fuse(items, maxd=0.05):
            if not items:
                return None
            if len(items) == 1:
                return items[0]
            a, b = items[0], items[1]
            if np.linalg.norm(a[1] - b[1]) < maxd:
                return (a[0] + "+" + b[0], (a[1] + b[1]) / 2.0) + tuple(a[2:])
            return a  # front first
        cube_ok = [c for c in res["cube"] if abs(c[2] - tz0) < 0.08]
        c = fuse(cube_ok)
        if c is not None:
            tops = [x[2] for x in cube_ok]
            top = float(np.max(tops))
            half = float(np.clip((top - tz0) / 2.0, 0.012, 0.035))
            self.cube = dict(xy=np.array(c[1]), top_z=top, half=half)
        t = fuse([x for x in res["target"] if abs(x[2] - tz0) < 0.05])
        if t is not None:
            self.target = np.array([t[1][0], t[1][1], tz0 if self.table_z is not None else t[2]])
            self.target[2] = float(min(t[2], tz0 + 0.02)) if self.table_z is not None else t[2]
        bcands = [x for x in res["button"] if abs(x[2] - tz0) < 0.08]
        # avoid the target region
        if self.target is not None:
            bcands = [x for x in bcands if np.linalg.norm(x[1] - self.target[:2]) > 0.03]
        if self.cube is not None:
            bcands = [x for x in bcands if np.linalg.norm(x[1] - self.cube["xy"]) > 0.03]
        # choose: agree across cameras if possible
        best = None
        fr = [x for x in bcands if x[0] == "front"]
        wr = [x for x in bcands if x[0] == "wrist"]
        for a in fr:
            for b in wr:
                if np.linalg.norm(a[1] - b[1]) < 0.05:
                    best = ("both", (a[1] + b[1]) / 2, max(a[2], b[2]))
                    break
            if best:
                break
        if best is None and bcands:
            x = max(bcands, key=lambda x: x[3] + 3 * x[4])
            best = (x[0], x[1], x[2])
        if best is not None:
            self.button = dict(xy=np.array(best[1]), top_z=float(best[2]))
        self.others = [x[2] for x in res["others"]]
        self.log(f"scene: cube {self._fmt(self.cube)} target {self.target} button {self._fmt(self.button)} "
                 f"others {len(self.others)} raw cube {[(x[0], np.round(x[1], 3), round(x[2], 3)) for x in res['cube']]} "
                 f"raw target {[(x[0], np.round(x[1], 3), round(x[2], 3)) for x in res['target']]} "
                 f"raw button {[(x[0], np.round(x[1], 3), round(x[2], 3), x[3], x[4]) for x in res['button']]}")

    @staticmethod
    def _fmt(d):
        if d is None:
            return None
        return {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else (round(v, 4) if isinstance(v, float) else v)) for k, v in d.items()}

    def wrist_find(self, kind, expect_xy, maxd=0.07):
        """Find object near expect_xy in the current wrist image.

        Returns dict(xy, top_z, mask, J, area) or None.
        """
        o = self.observe()
        img = o.rgb.get("wrist")
        if img is None:
            return None
        if kind == "cube":
            bl = [b for b in P.find_color_blobs(img, self.color) if b["area"] >= 10]
        elif kind == "target":
            bl = [b for b in P.find_target_blobs(img) if b["area"] >= 10]
            if bl:
                amax = max(b["area"] for b in bl)
                bl = [b for b in bl if b["area"] >= 0.3 * amax]
        else:
            bl = [b for b in P.find_button_blobs(img) if b["area"] >= 10 and b["aspect"] < 3.0]
            for b in bl:
                b["mask"] = b["filled"]
        if not bl:
            return None
        bl = bl[:5]
        cents = self.io.px2world(o.step_id, [("wrist", b["cx"], b["cy"]) for b in bl], radius=1)
        best, bd = None, 1e9
        for b, p in zip(bl, cents):
            if p is None:
                continue
            d = float(np.linalg.norm(p[:2] - np.asarray(expect_xy)[:2])) if expect_xy is not None else 0.0
            if d < bd:
                best, bd = (b, p), d
        if best is None or bd > maxd:
            self.log(f"wrist_find {kind}: none near {expect_xy} (best d {bd:.3f})")
            return None
        b, p = best
        pts = self.blob_points(o, "wrist", b["mask"], n=24)
        if kind == "target":
            if not pts:
                return None
            A = np.array(pts)
            xy, tz = np.median(A[:, :2], 0), float(np.median(A[:, 2]))
            cm = __import__("cv2").dilate(P.color_mask(img, self.color), np.ones((5, 5), np.uint8)).astype(bool)
            touches_cube = bool((cm & b["mask"].astype(bool)).any())
            x0, y0, bw, bh = b["bbox"]
            asp = max(bw, bh) / max(1.0, min(bw, bh))
            if x0 <= 1 or y0 <= 1 or x0 + bw >= 255 or y0 + bh >= 255 or touches_cube or asp > 1.5:
                self.log(f"wrist target blob partial (touches_cube={touches_cube}, aspect {asp:.2f}); weak estimate only")
                return dict(xy=xy, top_z=tz, area=b["area"], partial=True)
            return dict(xy=xy, top_z=tz, area=b["area"], partial=False)
        xy, tz = self.top_center(pts)
        if xy is None:
            return None
        if kind == "button":
            return dict(xy=np.array(p[:2]) if p is not None else xy, top_z=float(np.max(np.array(pts)[:, 2])), area=b["area"])
        # local image->world jacobian around the blob centre (for yaw)
        cx, cy = b["cx"], b["cy"]
        off = 5.0
        sel = [("wrist", cx, cy), ("wrist", cx + off, cy), ("wrist", cx - off, cy), ("wrist", cx, cy + off), ("wrist", cx, cy - off)]
        q = self.io.px2world(o.step_id, sel, radius=0)
        J = None
        if all(t is not None for t in q):
            du = (q[1][:2] - q[2][:2]) / (2 * off)
            dv = (q[3][:2] - q[4][:2]) / (2 * off)
            J = np.stack([du, dv], 1)
        ang = P.min_area_angle(b["mask"])
        return dict(xy=xy, top_z=tz, mask=b["mask"], J=J, img_angle=ang, area=b["area"], pts=len(pts))

    def cube_yaw_delta(self, f):
        """Rotation about world z that aligns the finger axis (image u) with a cube face."""
        if f is None or f.get("J") is None or f.get("img_angle") is None:
            return None
        J = f["J"]
        a = math.radians(f["img_angle"])
        e_img = np.array([math.cos(a), math.sin(a)])
        e_w = J @ e_img
        u_w = J @ np.array([1.0, 0.0])
        if np.linalg.norm(e_w) < 1e-6 or np.linalg.norm(u_w) < 1e-6:
            return None
        d = math.atan2(e_w[1], e_w[0]) - math.atan2(u_w[1], u_w[0])
        return wrap_half_pi(d), u_w / np.linalg.norm(u_w)

    def choose_yaw(self, delta, u_w, cube_xy):
        """Among delta + k*90deg choose small rotation whose finger path avoids neighbours."""
        best, bscore = delta, 1e9
        for k in (0, 1, -1):
            dd = delta + k * math.pi / 2
            if abs(dd) > math.radians(100):
                continue
            fa = rotz(dd)[:2, :2] @ u_w   # finger axis after rotation
            pa = np.array([-fa[1], fa[0]])
            score = abs(dd)
            for oxy in self.others + ([self.button["xy"]] if self.button is not None else []):
                rel = np.asarray(oxy)[:2] - cube_xy
                along, across = abs(rel @ fa), abs(rel @ pa)
                # fingers occupy |along| in [~0.03, 0.065], |across| < 0.012 ; others half ~0.02-0.03
                if along < 0.065 + 0.025 and across < 0.012 + 0.025:
                    score += 5.0
            if score < bscore:
                best, bscore = dd, score
        return best

    # ------------------------------------------------------------------ phases
    def probe(self):
        """Infer translation axis signs and the rotation composition from tiny motions."""
        o0 = self.obs
        p0, R0 = o0.pos.copy(), self.R_cur()
        self.send([[0, 0, 0.2, 0, 0, 0, GRIP_OPEN]] * 3)
        dz = self.obs.pos[2] - p0[2]
        if dz < -0.01:
            self.axis_sign[2] = -1
        p1 = self.obs.pos.copy()
        self.send([[0.2, 0.2, 0, 0, 0, 0, GRIP_OPEN]] * 3)
        dxy = self.obs.pos - p1
        for i in (0, 1):
            if dxy[i] < -0.01:
                self.axis_sign[i] = -1
        self.log(f"probe translation: dz {dz:.4f} dxy {np.round(dxy[:2], 4)} -> signs {self.axis_sign}")
        R1 = self.R_cur()
        c = np.array([0.1, 0.0, 0.25])
        self.send([[0, 0, 0, c[0] / 3 / ROT_SCALE, 0, c[2] / 3 / ROT_SCALE, GRIP_OPEN]] * 3)
        R2 = self.R_cur()
        meas_w = mat_to_rotvec(R2 @ R1.T)
        meas_b = mat_to_rotvec(R1.T @ R2)
        errs = {"world": np.linalg.norm(meas_w - c), "world_neg": np.linalg.norm(meas_w + c),
                "body": np.linalg.norm(meas_b - c), "body_neg": np.linalg.norm(meas_b + c)}
        mode = min(errs, key=errs.get)
        mag = float(np.linalg.norm(meas_w))
        if mag > 0.08 and errs[mode] < 0.6 * mag:
            self.rot_mode = mode
        self.log(f"probe rotation: meas_w {np.round(meas_w, 3)} meas_b {np.round(meas_b, 3)} errs "
                 f"{ {k: round(float(v), 3) for k, v in errs.items()} } -> mode {self.rot_mode}")
        # restore home orientation and height region
        self.move_to(self.obs.pos, R_des=self.R_des, label="probe-restore", max_iter=6)

    def grasp_height(self):
        half = self.cube["half"] if self.cube else 0.02
        return self.table_z + max(0.012, half)

    def first_grasp(self, attempt):
        tz = self.table_z
        cxy = self.cube["xy"]
        hover = np.array([cxy[0], cxy[1], tz + HOVER_DZ])
        self.move_to(hover, vmax=V_FREE, label="hover-cube")
        f = self.wrist_find("cube", cxy, maxd=0.08)
        if f is None:
            # go higher and look again
            self.move_to(hover + np.array([0, 0, 0.06]), label="hover-high")
            f = self.wrist_find("cube", cxy, maxd=0.12)
        if f is not None:
            self.cube["xy"] = f["xy"]
            if self.table_z is not None and f.get("top_z") is not None:
                h = (f["top_z"] - self.table_z) / 2.0
                if 0.01 < h < 0.04:
                    self.cube["half"] = h
                    self.cube["top_z"] = f["top_z"]
            yd = self.cube_yaw_delta(f)
            if yd is not None:
                delta, u_w = yd
                dd = self.choose_yaw(delta, u_w, self.cube["xy"])
                self.log(f"cube wrist: xy {np.round(f['xy'], 4)} top {f['top_z']:.4f} img_angle {f['img_angle']} "
                         f"delta {math.degrees(delta):.1f} chosen {math.degrees(dd):.1f}")
                self.R_des = rotz(dd) @ self.R_des
            else:
                self.log("cube yaw unknown; grasp with current yaw")
            cxy = self.cube["xy"]
            # rotate + centre, then verify from a second wrist view
            self.move_to([cxy[0], cxy[1], tz + HOVER_DZ - 0.03], vmax=V_FREE, label="align")
            f2 = self.wrist_find("cube", cxy, maxd=0.05)
            if f2 is not None:
                yd2 = self.cube_yaw_delta(f2)
                resid = math.degrees(yd2[0]) if yd2 else None
                self.log(f"after align: xy {np.round(f2['xy'], 4)} residual yaw {resid}")
                if np.linalg.norm(f2["xy"] - cxy) < 0.03:
                    self.cube["xy"] = f2["xy"]
                if yd2 is not None and abs(yd2[0]) > math.radians(12):
                    self.R_des = rotz(yd2[0]) @ self.R_des
        else:
            self.log("wrist did not find the cube; using front estimate")
        return self.grasp_at(self.cube["xy"], label=f"first{attempt}")

    def grasp_at(self, xy, label=""):
        """From above xy: descend, close, verify, lift.  Returns True when holding."""
        gz = self.grasp_height()
        self.grasp_dz = gz - self.table_z
        pre = np.array([xy[0], xy[1], gz + 0.05])
        self.move_to(pre, vmax=V_FREE, tol=0.004, label=f"pre-{label}")
        ok = self.move_to([xy[0], xy[1], gz], vmax=V_NEAR, tol=0.003, label=f"descend-{label}")
        if not ok:
            self.log(f"descend not converged (z {self.obs.pos[2]:.4f} vs {gz:.4f}); closing anyway")
        f = self.gripper(GRIP_CLOSE, min_steps=8, max_steps=24)
        self.log(f"closed fingers {f:.4f} at {np.round(self.obs.pos, 4)}")
        if f < 0.006:
            self.log("grasp missed (fingers closed fully)")
            self.gripper(GRIP_OPEN, min_steps=6, max_steps=16)
            self.move_to([xy[0], xy[1], gz + 0.08], vmax=V_FREE, label="miss-retreat")
            return False
        # lift a little, check, then full lift
        self.move_to([xy[0], xy[1], gz + 0.04], vmax=V_NEAR * 1.5, grip=GRIP_CLOSE, label=f"lift1-{label}")
        f2 = self.obs.fingers
        if f2 < 0.006:
            self.log(f"lost cube in early lift (fingers {f2:.4f})")
            self.gripper(GRIP_OPEN, min_steps=6, max_steps=16)
            return False
        self.move_to([xy[0], xy[1], self.table_z + LIFT_DZ], vmax=V_CARRY, grip=GRIP_CLOSE, tol=0.006,
                     label=f"lift-{label}")
        f3 = self.obs.fingers
        if f3 < 0.006:
            self.log(f"lost cube during lift (fingers {f3:.4f})")
            self.gripper(GRIP_OPEN, min_steps=6, max_steps=16)
            return False
        self.log(f"holding: fingers {f3:.4f} z {self.obs.pos[2]:.4f}")
        return True

    def place(self, refine_target):
        tz = self.table_z
        txy = self.target[:2].copy()
        self.move_to([txy[0], txy[1], tz + LIFT_DZ], vmax=V_CARRY, grip=GRIP_CLOSE, tol=0.005, label="carry")
        if refine_target:
            f = self.wrist_find("target", txy, maxd=0.05)
            if f is not None and not f.get("partial") and np.linalg.norm(f["xy"] - txy) < 0.04:
                self.log(f"target refined {np.round(txy, 4)} -> {np.round(f['xy'], 4)} (z {f['top_z']:.4f})")
                txy = (txy + 2 * f["xy"]) / 3.0
                self.target[:2] = txy
                self.move_to([txy[0], txy[1], tz + LIFT_DZ], vmax=V_CARRY, grip=GRIP_CLOSE, tol=0.004, label="carry-fix")
        rest = self.target[2] + self.grasp_dz + PLACE_EXTRA
        rest = max(rest, tz + self.grasp_dz + PLACE_EXTRA)
        self.move_to([txy[0], txy[1], rest + 0.03], vmax=V_CARRY, grip=GRIP_CLOSE, tol=0.004, label="lower")
        self.move_to([txy[0], txy[1], rest], vmax=V_NEAR, grip=GRIP_CLOSE, tol=0.003, label="lower-fine")
        f = self.obs.fingers
        lost = f < 0.006
        if lost:
            self.log("cube lost before release")
        self.gripper(GRIP_OPEN, min_steps=6, max_steps=20)
        gz = self.table_z + self.grasp_dz
        self.move_to([txy[0], txy[1], gz + RETREAT_DZ], vmax=V_NEAR * 1.6, tol=0.005, label="retreat")
        # verify from the wrist camera; a missing detection is not evidence of failure
        fc = self.wrist_find("cube", txy, maxd=0.12)
        if fc is None:
            self.log(f"verify: cube not seen from wrist (lost_before_release={lost})")
            if lost:
                return "missed", None
            return "assumed", None
        d = float(np.linalg.norm(fc["xy"] - txy))
        self.log(f"verify: cube at {np.round(fc['xy'], 4)} top {fc['top_z']:.4f} dist to target {d:.4f}")
        if d < ON_TARGET_TOL:
            return "placed", fc
        if d > ON_TARGET_FAIL and (fc["top_z"] - self.table_z) < 0.06:
            return "missed", fc
        # ambiguous (between tolerances): trust the physical sequence unless the cube fell early
        return ("missed" if lost else "assumed"), fc

    def search_scene(self, wide=False):
        """Some object missing: look from a few other poses (higher, shifted)."""
        p0 = self.obs.pos.copy() if not hasattr(self, "_search_p0") else self._search_p0
        if self.table_z is not None:
            p0 = p0.copy(); p0[2] = max(p0[2], self.table_z + 0.25)   # look from well above the table
        self._search_p0 = p0
        looks = [np.array([0, 0, 0.0]), np.array([0, 0, 0.08]), np.array([0.08, 0.10, 0.05]),
                 np.array([0.08, -0.10, 0.05]), np.array([-0.08, 0, 0.08])]
        if wide:
            looks = [np.array([0, 0, 0.0]), np.array([0.12, 0.18, 0.02]), np.array([0.12, -0.18, 0.02]),
                     np.array([-0.06, 0.15, 0.06]), np.array([-0.06, -0.15, 0.06]), np.array([0, 0, 0.10])]
        for k, d in enumerate(looks):
            if self.cube is not None and self.target is not None and self.button is not None:
                break
            self.log(f"search look {k}: missing cube={self.cube is None} target={self.target is None} button={self.button is None}")
            if k > 0:
                self.move_to(p0 + d, vmax=V_FREE, label=f"look{k}", max_iter=8)
            o = self.observe()
            keep = (self.cube, self.target, self.button)
            self.fuse_scene(self.perceive_scene(o))
            # never replace an existing estimate during the search
            self.cube = keep[0] or self.cube
            self.target = keep[1] if keep[1] is not None else self.target
            self.button = keep[2] or self.button

    def locate_cube_wide(self):
        """Cube position unknown: look from higher up with both cameras."""
        p = self.obs.pos
        self.move_to([p[0], p[1], self.table_z + HOVER_DZ + 0.10], vmax=V_FREE, label="look-wide")
        o = self.observe()
        res = self.perceive_scene(o)
        cands = [c for c in res["cube"] if abs(c[2] - self.table_z) < 0.08]
        if cands:
            self.cube["xy"] = np.array(cands[0][1])
            self.log(f"wide look: cube at {np.round(self.cube['xy'], 4)} ({cands[0][0]})")
            return True
        self.log("wide look: cube not found")
        return False

    def press_button(self):
        tz = self.table_z
        bxy = self.button["xy"].copy()
        top = self.button["top_z"]
        cur = self.obs.pos
        self.move_to([cur[0], cur[1], tz + HOVER_DZ], vmax=V_FREE, grip=GRIP_CLOSE, label="up-before-button")
        self.move_to([bxy[0], bxy[1], tz + HOVER_DZ], vmax=V_FREE, grip=GRIP_CLOSE, label="to-button")
        f = self.wrist_find("button", bxy, maxd=0.05)
        if f is not None:
            self.log(f"button refined {np.round(bxy, 4)} -> {np.round(f['xy'], 4)} top {f['top_z']:.4f}")
            bxy = (bxy + f["xy"]) / 2.0
            if abs(f["top_z"] - top) < 0.03:
                top = max(top, f["top_z"])
        top = float(np.clip(top, tz + 0.005, tz + 0.06))
        offsets = [(0, 0), (0.008, 0), (-0.008, 0), (0, 0.008), (0, -0.008), (0.012, 0.012), (-0.012, -0.012)]
        for k, (ox, oy) in enumerate(offsets):
            x, y = bxy[0] + ox, bxy[1] + oy
            self.move_to([x, y, top + 0.03], vmax=V_FREE, grip=GRIP_CLOSE, tol=0.004, label=f"button-above{k}")
            # push down in small chunks
            for j in range(16):
                z = self.obs.pos[2]
                goal = top - 0.035
                if z <= goal:
                    break
                dz = -0.005
                self.send([[0, 0, dz / POS_SCALE * self.axis_sign[2], 0, 0, 0, GRIP_CLOSE]] * 2)
                if self.io.ended:
                    return True
            self.hold(4, GRIP_CLOSE)
            if self.io.ended:
                return True
            self.log(f"press attempt {k} at {np.round(self.obs.pos, 4)} did not end the episode")
            self.move_to([x, y, top + 0.04], vmax=V_FREE, grip=GRIP_CLOSE, label="button-up")
        return False

    # ------------------------------------------------------------------ main
    def run(self):
        st = self.io.status()
        self.log(f"status keys {list(st.keys())[:30]}")
        o = self.observe()
        instr = o.instruction or find_instruction(st)
        self.color, self.N = parse_instruction(instr)
        if self.color is None:
            self.color = "green"
        if self.N is None:
            self.N = 1
            self.log("WARNING: repetition count not parsed; defaulting to 1")
        self.log(f"instruction {instr!r} -> colour {self.color} N {self.N}; label source {o.label_source}; "
                 f"state {np.round(o.state, 4).tolist()} step {o.step_id} limits rem {self.io.remaining} max {self.io.max_steps}")
        for cam, arr in o.rgb.items():
            self.save_img(f"init_{cam}", arr)
        self.R_home = self.R_cur()
        self.R_des = self.R_home.copy()
        res = self.perceive_scene(o)
        self.fuse_scene(res)
        if self.table_z is None:
            raise RuntimeError("no table depth")
        self.probe()
        for rnd in range(3):   # repeat the look-around: transient occlusion / slow first renders
            if self.cube is not None and self.target is not None and self.button is not None:
                break
            self.check_alive()
            self.search_scene(wide=rnd > 0)
        if self.cube is None:
            raise RuntimeError("cube not found")
        if self.target is None:
            raise RuntimeError("target not found")

        holding = False
        attempts = 0
        need_first = True
        refine_target = True
        while self.count < self.N:
            self.check_alive()
            if not holding:
                attempts += 1
                # Keep retrying while budget remains (pressing with a short count always fails);
                # leave only when the step/wall reserve for the button press is reached.
                low_steps = self.io.remaining is not None and self.io.remaining < BUTTON_STEP_RESERVE
                if attempts > MAX_GRASP_ATTEMPTS + 3 * self.N or low_steps or self.time_left() < BUTTON_TIME_RESERVE_S:
                    self.log(f"grasp budget exhausted (attempts {attempts}, steps left {self.io.remaining}, "
                             f"time left {self.time_left():.0f}s); going to button")
                    break
                if need_first:
                    holding = self.first_grasp(attempts)
                else:
                    holding = self.grasp_at(self.cube["xy"], label=f"cyc{self.count}")
                if not holding:
                    # re-localise cube from wrist above last known xy
                    self.move_to([self.cube["xy"][0], self.cube["xy"][1], self.table_z + HOVER_DZ], label="relocate")
                    f = self.wrist_find("cube", self.cube["xy"], maxd=0.15)
                    if f is not None:
                        self.cube["xy"] = f["xy"]
                    need_first = True
                    continue
            status, fc = self.place(refine_target)
            refine_target = False
            holding = False
            if fc is not None:
                self.cube["xy"] = fc["xy"]
            else:
                self.cube["xy"] = self.target[:2].copy()
            if status in ("placed", "assumed"):
                self.count += 1
                need_first = False
                self.log(f"=== placement {self.count}/{self.N} {status} ===")
            else:
                self.log(f"placement failed ({status}); count stays {self.count}")
                need_first = True
                if fc is None:
                    self.locate_cube_wide()
        if self.button is None:
            o = self.observe()
            keep = (self.cube, self.target)
            self.fuse_scene(self.perceive_scene(o))
            self.cube, self.target = keep[0] or self.cube, keep[1] if keep[1] is not None else self.target
        for rnd in range(2):
            if self.button is not None:
                break
            self.check_alive()
            self.search_scene(wide=True)
        if self.button is None:
            raise RuntimeError("button not found")
        self.log(f"pressing button after {self.count} placements")
        done = self.press_button()
        self.log(f"press result ended={self.io.ended} complete={self.io.complete} done={done}")


def find_instruction(meta):
    from rax_io import find_key
    v = find_key(meta, "instruction")
    return v if isinstance(v, str) else ""


def main(ctx):
    c = Controller(ctx)
    try:
        c.run()
    except Abort as e:
        c.log(f"abort: {e}")
    except Exception as e:
        c.log("EXCEPTION " + repr(e) + "\n" + traceback.format_exc())
        try:
            recover(c)
        except Abort as e2:
            c.log(f"abort in recovery: {e2}")
        except Exception as e2:
            c.log("recovery failed " + repr(e2) + "\n" + traceback.format_exc())
    c.log(f"finished: count {c.count}/{c.N} ended {c.io.ended} complete {c.io.complete} "
          f"steps {c.io.steps_sent} calls {c.io.n_calls} timing { {k: (v[0], round(v[1], 1)) for k, v in c.io.timing.items()} }")
    try:
        c.logf.close()
    except Exception:
        pass


def recover(c):
    """After an unexpected exception: one more pass through the remaining work."""
    if c.io.ended:
        return
    c.observe()
    if c.R_des is None:
        c.R_des = c.R_cur()
    if c.table_z is None or c.cube is None or c.target is None:
        c.fuse_scene(c.perceive_scene(c.obs))
    if c.table_z is None:
        return
    if c.cube is not None and c.target is not None and c.N is not None and c.count < c.N:
        c.gripper(GRIP_OPEN, min_steps=6, max_steps=12)
        c.move_to([c.obs.pos[0], c.obs.pos[1], c.table_z + HOVER_DZ], label="recover-up")
        c.move_to([c.cube["xy"][0], c.cube["xy"][1], c.table_z + HOVER_DZ], label="recover-hover")
        f = c.wrist_find("cube", c.cube["xy"], maxd=0.15)
        if f is not None:
            c.cube["xy"] = f["xy"]
        tries = 0
        while c.count < c.N and tries < c.N + 4:
            tries += 1
            if not c.grasp_at(c.cube["xy"], label="recover"):
                continue
            status, fc = c.place(False)
            if fc is not None:
                c.cube["xy"] = fc["xy"]
            if status in ("placed", "assumed"):
                c.count += 1
    if c.button is not None and c.count >= (c.N or 0):
        c.press_button()
