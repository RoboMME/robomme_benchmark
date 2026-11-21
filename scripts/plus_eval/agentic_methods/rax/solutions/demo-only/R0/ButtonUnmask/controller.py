"""ButtonUnmask controller (RoboMME-Plus, RAX).  VERSION = "v4"

Task (from demos + instruction): cubes are revealed in rounds (demo schedule: appear at
steps 12/56/100, each covered by a white bin 32 steps later).  Press the button before
step 90, then pick up (grasp+lift, put back) the bins hiding the cubes named in the
instruction, in order.

Pipeline / transition table
  INIT      -> observe step 0, calibrate cameras, table z, button        -> WATCH1
  WATCH1    -> hold still, detect round-1 cubes (both cams)              -> TO_READY (after round1 or step 20)
  TO_READY  -> move above button (READY_H), close gripper, hold still    -> WATCH2
  WATCH2    -> detect round-2 cubes until seen+2 steps or PRESS_LATEST   -> PRESS
  PRESS     -> descend on button cap until stall, push, rise             -> TO_OBS
  TO_OBS    -> go to observe pose (start pose), open gripper, hold still -> WATCH3
  WATCH3    -> keep detecting reveal rounds until no new round expected  -> PICK
  PICK(i)   -> hover above target, perceive bin, yaw-align, grasp, lift, put back
               (retry ladder: re-perceive, shift, re-grasp)              -> PICK(i+1) / DONE
Every step checks episode_ended; main() never raises.
"""
import math
import os
import re
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bu_io as U  # noqa: E402
from bu_io import log  # noqa: E402
import bu_geom as G  # noqa: E402
import bu_vision as V  # noqa: E402

VERSION = "v6"

# ----------------------------------------------------------------- tunables
# (no live tests were possible: values are from demo analysis and Panda/ManiSkill priors)
POS_SCALE = 0.05          # m per unit action (documented)
ROT_SCALE = 0.5           # rad per unit action (documented)
VMAX = 0.025              # m per step commanded in free space
VMAX_SLOW = 0.010         # m per step near contact
WMAX = 0.25               # rad per step
READY_H = 0.16            # TCP height above button cap while waiting for round 2
PRESS_LATEST_START = 70   # start the press no later than this step (deadline: register < 90)
PRESS_EARLIEST = 57       # do not press before round 2 has been seen (demo: round 2 at 56)
ROUND_PERIOD = 44         # demo reveal period
ROUND_VISIBLE = 32
HOVER_H = 0.20            # TCP height above table for bin perception
GRASP_DEPTH = 0.020       # TCP below bin top when closing
LIFT_H = 0.10             # lift height
CLOSE_STEPS = 10
OPEN_STEPS = 8
TABLE_CLEAR = 0.012       # min TCP height above table
MERGE_DIST = 0.035        # same-colour detections closer than this are one cube
BUTTON_EXCL = 0.04        # ignore colour detections this close to the button
ANCHOR_TOL = 0.02         # bin centre may differ from the cube track by at most this


HOLD_CHUNK = 10           # zero rows per call while holding
MOVE_CHUNK = 3            # max open-loop rows per call in free-space moves
WALL_SOFT = 1050.0        # s: stop acting before the 1200 s controller limit


class Ended(Exception):
    pass


T_START = time.time()


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.state = None
        self.step = 0
        self.ended = False
        self.complete = False
        self.grip = -1.0
        self.order = None
        self.rot_mode = "world"
        self.axis_sign = np.ones(3)
        self.motion_log = []  # (cmd_delta, measured_delta)
        self.max_steps = None
        self.last_meta = None
        self.n_calls = 0

    # -------------------------------------------------------------- I/O
    def _call_retry(self, tool, **kw):
        last = None
        for attempt in range(2):
            try:
                self.n_calls += 1
                return self.ctx.call(tool, **kw)
            except Exception as ex:
                last = ex
                msg = repr(ex)
                log("tool %s failed (attempt %d): %s" % (tool, attempt, msg[:300]))
                low = msg.lower()
                if "ended" in low or "no interactive episode" in low or "terminated" in low:
                    self.ended = True
                    raise Ended()
                time.sleep(0.5)
        raise last

    def observe(self):
        res = self._call_retry("robomme_observe")
        ob = U.parse_result(res, self.order)
        if (("front" not in ob.rgb) or ("wrist" not in ob.rgb) or ("front" not in ob.depth)
                or ("wrist" not in ob.depth)):
            U.fill_from_manifest(ob, want_step=ob.step_id)
        if ob.state is not None:
            self.state = np.array(ob.state, float)
        if ob.step_id is not None:
            self.step = ob.step_id
        if ob.ended:
            self.ended = True
        if ob.complete:
            self.complete = True
        if self.max_steps is None:
            self.max_steps = U.max_steps(ob.meta)
        self.last_meta = ob.meta
        return ob

    def send(self, rows):
        if time.time() - T_START > WALL_SOFT:
            log("WALL GUARD: stopping at step", self.step)
            raise Ended()
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        if self.max_steps is not None:
            left = self.max_steps - self.step
            if left <= 0:
                self.ended = True
                raise Ended()
            rows = rows[:max(1, left)]
        res = self._call_retry("robomme_step", actions=rows)
        ob = U.parse_result(res, self.order)
        prev = self.step
        if ob.step_id is not None:
            self.step = ob.step_id
        else:
            self.step = prev + len(rows)
        if ob.state is not None:
            self.state = np.array(ob.state, float)
        if ob.ended:
            self.ended = True
        if ob.complete:
            self.complete = True
        if self.max_steps is None:
            self.max_steps = U.max_steps(ob.meta)
        self.last_meta = ob.meta
        if self.ended:
            log("episode ended at step", self.step, "complete=", self.complete)
            raise Ended()
        return ob

    # -------------------------------------------------------------- state helpers
    @property
    def pos(self):
        return self.state[:3].copy()

    @property
    def R(self):
        return G.rotvec_to_mat(self.state[3:6])

    @property
    def opening(self):
        return float(self.state[6] + self.state[7])

    def rot_action(self, R_ref):
        if R_ref is None:
            return np.zeros(3)
        e = G.mat_to_rotvec(R_ref @ self.R.T)  # world-frame error
        n = np.linalg.norm(e)
        cmd = e * 0.8
        if np.linalg.norm(cmd) > WMAX:
            cmd = cmd / np.linalg.norm(cmd) * WMAX
        if n < 0.004:
            cmd = np.zeros(3)
        if self.rot_mode == "body":
            cmd = self.R.T @ cmd
        return cmd / ROT_SCALE

    def rot_err(self, R_ref):
        if R_ref is None:
            return 0.0
        return float(np.linalg.norm(G.mat_to_rotvec(R_ref @ self.R.T)))

    def hold(self, n, R_ref=None, grip=None):
        if grip is not None:
            self.grip = grip
        p0 = self.pos
        left = n
        while left > 0:
            k = min(left, HOLD_CHUNK)
            d = (p0 - self.pos) * 0.8
            first = list(d / POS_SCALE * self.axis_sign) + list(self.rot_action(R_ref)) + [self.grip]
            rows = [first] + [[0, 0, 0, 0, 0, 0, self.grip]] * (k - 1)
            self.send(rows)
            left -= k

    def move_to(self, target, R_ref=None, grip=None, vmax=VMAX, tol=0.004, rot_tol=0.03,
                timeout=60, per_step=None, stall_ok=False):
        """Closed-loop Cartesian move. Returns (ok, why)."""
        if grip is not None:
            self.grip = grip
        target = np.asarray(target, float)
        hist = []
        rot_hist = []
        for k in range(timeout):
            e = target - self.pos
            dist = float(np.linalg.norm(e))
            rerr = self.rot_err(R_ref)
            if dist < tol and rerr < rot_tol:
                return True, "reached"
            v = e * 0.9
            nv = np.linalg.norm(v)
            if nv > vmax:
                v = v / nv * vmax
            a = list(v / POS_SCALE * self.axis_sign) + list(self.rot_action(R_ref)) + [self.grip]
            rows = [a]
            if per_step is None and nv > vmax and dist > 0.05:
                # far from the target: send a short open-loop chunk of the same velocity row
                kk = int(min(MOVE_CHUNK, max(1, (dist - 0.03) / vmax)))
                rows = [a] + [list(v / POS_SCALE * self.axis_sign) + [0, 0, 0, self.grip]] * (kk - 1)
            p_before = self.pos
            self.send(rows)
            meas = self.pos - p_before
            self.motion_log.append((v.copy() * len(rows), meas.copy()))
            self._check_axes()
            hist.append(dist)
            rot_hist.append(rerr)
            self._check_rot_mode(rot_hist)
            if per_step is not None:
                try:
                    per_step()
                except Ended:
                    raise
                except Exception as ex:
                    log("per_step error", repr(ex))
            if len(hist) >= 6 and hist[-6] - hist[-1] < 0.003 and dist > tol and rerr < 0.2:
                return (True if stall_ok else False), "stall dist=%.3f" % dist
        return False, "timeout dist=%.3f rerr=%.3f" % (float(np.linalg.norm(target - self.pos)), self.rot_err(R_ref))

    def _check_axes(self):
        """Flip an axis sign if measured motion consistently opposes commands (sanity guard)."""
        if len(self.motion_log) < 6 or len(self.motion_log) > 40:
            return
        cmd = np.array([m[0] for m in self.motion_log])
        meas = np.array([m[1] for m in self.motion_log])
        for ax in range(3):
            c = cmd[:, ax] * self.axis_sign[ax]
            if np.abs(c).sum() < 0.03:
                continue
            ratio = float((meas[:, ax] * np.sign(c)).sum() / np.abs(c).sum())
            if ratio < -0.3:
                self.axis_sign[ax] *= -1
                log("AXIS FLIP", ax, "ratio", ratio)
                self.motion_log = []
                return

    def _check_rot_mode(self, rot_hist):
        if len(rot_hist) >= 4 and rot_hist[-4] > 0.05:
            if rot_hist[-1] > rot_hist[-4] + 0.03 and rot_hist[-1] > rot_hist[-2] > rot_hist[-3]:
                self.rot_mode = "body" if self.rot_mode == "world" else "world"
                log("ROT MODE switched to", self.rot_mode)
                del rot_hist[:]


# ===================================================================== perception
class Perception:
    def __init__(self, robot):
        self.r = robot
        self.front_model = None
        self.front_model_step = None
        self.table_z = None
        self.button = None      # dict(xy, top)
        self.static_colored = []  # world xy of coloured things present at step 0
        self.tracks = []        # cube tracks
        self.prev = {}          # cam -> (rgb, pose_key, step)
        self.bg_front = None
        self.events = []        # (step, kind, n)
        self.depth_ok = True
        self.depth_flip = {}
        self.bin_sides = []

    # ---------------------------------------------------------- tool points
    def tool_points(self, step_id, cam, pix):
        out = []
        for i in range(0, len(pix), 32):
            chunk = pix[i:i + 32]
            sel = [{"camera": cam, "u": float(u), "v": float(v)} for (u, v) in chunk]
            try:
                res = self.r.ctx.call("robomme_pixel_to_position", observation_step_id=int(step_id),
                                      selections=sel, depth_window_radius=0)
                self.r.n_calls += 1
                pts = U.parse_points(res, len(chunk))
            except Exception as ex:
                log("pixel_to_position failed", repr(ex)[:300])
                pts = [(None, None)] * len(chunk)
            out.extend(pts)
        return out

    def _depth_variants(self, d):
        return {"id": d, "flipud": d[::-1], "fliplr": d[:, ::-1], "rot180": d[::-1, ::-1]}

    def intrinsics_cands(self, ob, cam):
        """(f, cx, cy) candidates parsed from camera_intrinsics metadata (any layout)."""
        out = []
        try:
            for p_, v in U.find_all(ob.meta, "camera_intrinsics"):
                node = v.get(cam) if isinstance(v, dict) else None
                if node is None and isinstance(v, dict):
                    for k, vv in v.items():
                        if cam in str(k).lower():
                            node = vv
                if isinstance(node, dict):
                    for k in ("K", "matrix", "intrinsic", "intrinsics", "intrinsic_cv"):
                        if k in node:
                            node = node[k]
                            break
                if isinstance(node, dict) and "fx" in node:
                    out.append((float(node["fx"]), float(node.get("cx", 128)), float(node.get("cy", 128))))
                    if "fy" in node:
                        out.append((float(node["fy"]), float(node.get("cx", 128)), float(node.get("cy", 128))))
                M = np.array(node, float) if isinstance(node, list) else None
                if M is not None and M.size == 9:
                    M = M.reshape(3, 3)
                    out.append((float(M[0, 0]), float(M[0, 2]), float(M[1, 2])))
                    out.append((float(M[1, 1]), float(M[0, 2]), float(M[1, 2])))
        except Exception:
            log("intrinsics parse error", traceback.format_exc()[-300:])
        return [c for c in out if 20 < c[0] < 5000]

    def metric_depth(self, ob, cam):
        d = ob.depth[cam]
        if d.ndim == 3:
            d = d[..., 0]
        sc = ob.depth_scale.get(cam)
        if not sc:
            med = float(np.median(d[d > 0])) if (d > 0).any() else 0.0
            sc = 0.001 if med > 20 else 1.0
        return d.astype(float) * float(sc), float(sc)

    def fit_model(self, ob, cam):
        """Fit a pinhole model for `cam` from <=32 tool samples; returns model or None."""
        if cam not in ob.depth or ob.step_id is None:
            return None
        d, sc = self.metric_depth(ob, cam)
        H, W = d.shape
        pix = []
        rng = np.random.RandomState(len(self.events) + 7)
        for gy in np.linspace(10, H - 11, 6):
            for gx in np.linspace(10, W - 11, 6):
                for _ in range(8):
                    u = int(np.clip(gx + rng.randint(-9, 10), 2, W - 3))
                    v = int(np.clip(gy + rng.randint(-9, 10), 2, H - 3))
                    c = d[v, u]
                    win = d[v - 1:v + 2, u - 1:u + 2]
                    if c <= 0 or not (win > 0).all():
                        continue
                    lap = max(abs(d[v, u - 1] + d[v, u + 1] - 2 * c), abs(d[v - 1, u] + d[v + 1, u] - 2 * c))
                    if lap <= 0.004 * c + 0.001:
                        pix.append((u, v))
                        break
        pix = pix[:32]
        if len(pix) < 8:
            log("fit_model", cam, "too few smooth pixels", len(pix))
            return None
        pts = self.tool_points(ob.step_id, cam, pix)
        good = [(p, q[0]) for p, q in zip(pix, pts) if q[0] is not None]
        if len(good) < 8:
            log("fit_model", cam, "too few tool points", len(good))
            return None
        fc = self.intrinsics_cands(ob, cam)
        if getattr(self, "f_known", {}).get(cam):
            fc = fc + [self.f_known[cam]]
        best = None
        order = [self.depth_flip.get(cam, "id")] + [k for k in ("id", "flipud", "fliplr", "rot180")
                                                     if k != self.depth_flip.get(cam, "id")]
        for name in order:
            dd = self._depth_variants(d)[name]
            uvd = [(u, v, float(dd[v, u])) for (u, v), _ in good]
            if any(x[2] <= 0 for x in uvd):
                continue
            xyz = [w for _, w in good]
            m = G.fit_pinhole(uvd, xyz, W, H, fc)
            if m is None:
                continue
            if best is None or m.resid[0] < best[1].resid[0]:
                best = (name, m)
            if m.resid[0] < 0.002 and m.resid[1] < 0.006:
                break
        if best is None:
            return None
        name, m = best
        ok = m.resid[0] < 0.004 and m.resid[1] < 0.012
        if not hasattr(self, "_fit_logged"):
            self._fit_logged = {}
        if not ok or self._fit_logged.get(cam, 0) < 3:
            self._fit_logged[cam] = self._fit_logged.get(cam, 0) + 1
            log("fit_model", cam, "variant", name, "f %.1f mode %s" % (m.f, m.depth_mode),
                "resid med/p90 %.4f/%.4f" % m.resid, "n", len(good), "ok" if ok else "REJECT")
        if not ok:
            return None
        if not hasattr(self, "f_known"):
            self.f_known = {}
        self.f_known[cam] = (m.f, m.cx, m.cy)
        self.depth_flip[cam] = name
        m.variant = name
        m.scale = sc
        return m

    def pointmap(self, ob, cam, model):
        d, _ = self.metric_depth(ob, cam)
        d = self._depth_variants(d)[getattr(model, "variant", "id")]
        pts, valid, us, vs = G.pointmap(model, d, 1)
        return pts, valid

    def get_maps(self, ob, cams=("front", "wrist")):
        """Dense (pts, valid) per camera; falls back to coarse tool sampling."""
        maps = {}
        for cam in cams:
            model = None
            if cam == "front" and self.front_model is not None and cam in ob.depth:
                model = self.front_model
            elif cam in ob.depth:
                model = self.fit_model(ob, cam)
                if cam == "front" and model is not None:
                    self.front_model = model
            if model is not None:
                pts, valid = self.pointmap(ob, cam, model)
                maps[cam] = (pts, valid, model)
            else:
                maps[cam] = self.tool_map(ob, cam)
        return maps

    def tool_map(self, ob, cam, stride=8):
        """Coarse point map from the tool (fallback when depth images are unusable)."""
        if ob.step_id is None:
            return None
        H = W = 256
        if cam in ob.rgb:
            H, W = ob.rgb[cam].shape[:2]
        pix = [(u, v) for v in range(stride // 2, H, stride) for u in range(stride // 2, W, stride)]
        pts = self.tool_points(ob.step_id, cam, pix)
        P = np.full((H, W, 3), np.nan)
        valid = np.zeros((H, W), bool)
        for (u, v), (q, _) in zip(pix, pts):
            if q is not None:
                P[v, u] = q
                valid[v, u] = True
        log("tool_map", cam, "valid", int(valid.sum()), "/", len(pix))
        return (P, valid, None)

    # ---------------------------------------------------------- init
    def init_scene(self, ob):
        maps = self.get_maps(ob)
        zs = []
        for cam in ("front", "wrist"):
            m = maps.get(cam)
            if m is None or cam not in ob.rgb:
                continue
            pts, valid, _ = m
            wood = V.is_wood_like(ob.rgb[cam]) & valid
            if wood.sum() > 50:
                zs.append(np.median(pts[..., 2][wood]))
        if zs:
            self.table_z = float(np.median(zs))
        else:
            self.table_z = 0.0
        log("table_z", self.table_z)
        self.bg_front = ob.rgb.get("front")
        self.find_button(ob, maps)
        # coloured static objects (e.g. coloured button base) present at step 0
        for cam in ("front", "wrist"):
            if cam not in ob.rgb or maps.get(cam) is None:
                continue
            for det in self.color_detections(ob, cam, maps[cam], use_static_filter=False):
                self.static_colored.append(det["xy"])
        log("static coloured objects at step0:", [tuple(np.round(x, 3)) for x in self.static_colored])
        return maps

    def find_button(self, ob, maps):
        cands = []
        ee = self.r.pos
        for cam in ("wrist", "front"):
            m = maps.get(cam)
            if m is None:
                continue
            pts, valid, _ = m
            z = pts[..., 2] - self.table_z
            obj = valid & (z > 0.004) & (z < 0.10)
            lab, n = G.label(G.dilate(obj, 1) & valid)
            for k in range(1, n + 1):
                mk = (lab == k) & obj
                c = int(mk.sum())
                if c < 6:
                    continue
                P = pts[mk]
                xy = P[:, :2]
                ext = np.percentile(xy, 95, 0) - np.percentile(xy, 5, 0)
                zz = P[:, 2] - self.table_z
                top = float(np.percentile(zz, 95))
                if ext.max() > 0.15 or top > 0.09:
                    continue
                # reject things touching tall structures (pillar side): require most points low
                hi = P[zz > top - 0.008]
                cxy = hi[:, :2].mean(0) if len(hi) else xy.mean(0)
                dist = float(np.linalg.norm(cxy - ee[:2]))
                cands.append({"cam": cam, "xy": cxy, "top": top + self.table_z, "n": c, "ext": ext,
                              "dist": dist})
        if not cands:
            log("BUTTON NOT FOUND; using EE xy")
            self.button = {"xy": ee[:2].copy(), "top": self.table_z + 0.03, "found": False}
            return
        cands.sort(key=lambda c: c["dist"])
        best = cands[0]
        # cross-camera confirmation: average with the other camera if within 3 cm
        same = [c for c in cands if np.linalg.norm(c["xy"] - best["xy"]) < 0.03]
        xy = np.mean([c["xy"] for c in same], 0)
        top = float(np.median([c["top"] for c in same]))
        self.button = {"xy": xy, "top": top, "found": True}
        log("button", np.round(xy, 4), "top", round(top, 4), "from", [(c["cam"], c["n"]) for c in same],
            "cands", [(c["cam"], tuple(np.round(c["xy"], 3)), round(c["dist"], 3)) for c in cands[:5]])

    # ---------------------------------------------------------- cube detection
    def _world_of_blob(self, mask, pts, valid):
        mm = mask & valid
        if mm.sum() < 3:
            return None
        P = pts[mm]
        P = P[np.isfinite(P).all(1)]
        if len(P) < 3:
            return None
        return np.median(P, 0)

    def color_detections(self, ob, cam, m, use_static_filter=True):
        out = []
        if m is None or cam not in ob.rgb:
            return out
        pts, valid, _ = m
        rgb = ob.rgb[cam]
        cm = V.color_class_map(rgb)
        for c in ("red", "green", "blue", "yellow", "orange", "purple", "cyan"):
            for b in V.blobs(cm == c, min_px=8):
                w = self._world_of_blob(b["mask"], pts, valid)
                if w is None:
                    w = self._sparse_world(ob, cam, b["mask"])
                if w is None:
                    continue
                zr = w[2] - self.table_z
                if zr < -0.01 or zr > 0.08:
                    continue
                out.append({"color": c, "xy": w[:2].copy(), "z": float(w[2]), "cam": cam, "n": b["n"]})
        if use_static_filter:
            out = [d for d in out if self._not_static(d["xy"])]
        return out

    def _sparse_world(self, ob, cam, mask):
        """Tool-based world point for a blob when the dense map has no sample in it."""
        try:
            pix = V.interior_pixels(mask, 4)
        except Exception:
            return None
        pts = [q for q, _ in self.tool_points(ob.step_id, cam, pix) if q is not None]
        if not pts:
            return None
        return np.median(np.array(pts), 0)

    def _not_static(self, xy):
        if self.button is not None and np.linalg.norm(xy - self.button["xy"]) < BUTTON_EXCL:
            return False
        for s in self.static_colored:
            if np.linalg.norm(xy - s) < 0.03:
                return False
        return True

    def diff_detections(self, ob, cam, m):
        """New objects since the previous static frame of this camera (any colour)."""
        out = []
        if m is None or cam not in ob.rgb:
            return out, 0
        prev = self.prev.get(cam)
        cur = ob.rgb[cam].astype(int)
        key = self._pose_key()
        self.prev[cam] = (cur, key, self.r.step)
        if prev is None:
            return out, 0
        prgb, pkey, pstep = prev
        if cam == "wrist" and (pkey is None or key is None or np.linalg.norm(pkey - key) > 0.002):
            return out, 0
        diff = np.abs(cur - prgb).sum(2) > 60
        pts, valid, _ = m
        nwhite = 0
        for b in V.blobs(diff, min_px=8):
            px = ob.rgb[cam][b["mask"]]
            name, frac = V.name_color(px)
            if name is None:
                continue
            if name in ("white", "gray"):
                nwhite += b["n"]
                continue
            w = self._world_of_blob(b["mask"], pts, valid)
            if w is None:
                w = self._sparse_world(ob, cam, b["mask"])
            if w is None:
                continue
            zr = w[2] - self.table_z
            if zr < -0.01 or zr > 0.08:
                continue
            if not self._not_static(w[:2]):
                continue
            out.append({"color": name, "xy": w[:2].copy(), "z": float(w[2]), "cam": cam + "-diff", "n": b["n"]})
        return out, nwhite

    def _pose_key(self):
        s = self.r.state
        if s is None:
            return None
        return np.concatenate([s[:3], s[3:6] * 0.05, [self.r.grip * 0.01]])

    def add_detections(self, dets, step):
        newc = 0
        for d in dets:
            best = None
            for t in self.tracks:
                if t["color"] != d["color"]:
                    continue
                dist = np.linalg.norm(t["xy"] - d["xy"])
                if dist < MERGE_DIST and (best is None or dist < best[0]):
                    best = (dist, t)
            if best is not None:
                t = best[1]
                k = t["n"]
                t["xy"] = (t["xy"] * k + d["xy"]) / (k + 1)
                t["n"] = k + 1
                t["last"] = step
                t["cams"].add(d["cam"])
            else:
                self.tracks.append({"color": d["color"], "xy": d["xy"].copy(), "n": 1, "first": step,
                                    "last": step, "cams": {d["cam"]}})
                newc += 1
        return newc

    def scan(self, ob, maps=None):
        """Detect cubes in an observation; returns (#new tracks, #coloured dets, white diff px)."""
        if maps is None:
            maps = {}
            if "front" in ob.depth or self.front_model is not None:
                maps["front"] = self._front_map(ob)
            maps["wrist"] = None
            if "wrist" in ob.depth:
                mdl = self.fit_model(ob, "wrist")
                if mdl is not None:
                    pts, valid = self.pointmap(ob, "wrist", mdl)
                    maps["wrist"] = (pts, valid, mdl)
        dets = []
        nwhite = 0
        for cam in ("front", "wrist"):
            m = maps.get(cam)
            if m is None:
                m = (np.full((256, 256, 3), np.nan), np.zeros((256, 256), bool), None)
            dets += self.color_detections(ob, cam, m)
            dd, nw = self.diff_detections(ob, cam, m)
            dets += dd
            nwhite += nw
        newc = self.add_detections(dets, self.r.step)
        return newc, len(dets), nwhite

    def _front_map(self, ob):
        if self.front_model is None:
            self.front_model = self.fit_model(ob, "front")
        if self.front_model is None or "front" not in ob.depth:
            return None
        pts, valid = self.pointmap(ob, "front", self.front_model)
        return (pts, valid, self.front_model)

    # ---------------------------------------------------------- bins
    def perceive_bin(self, ob, c0, maps=None, anchor=None):
        if maps is None:
            maps = self.get_maps(ob)
        res = {}
        allpts = []
        for cam in ("wrist", "front"):
            m = maps.get(cam)
            if m is None:
                continue
            pts, valid, _ = m
            P = pts
            z = P[..., 2] - self.table_z
            dxy = np.linalg.norm(P[..., :2] - c0[None, None, :], axis=-1)
            ok = valid & np.isfinite(z)
            obj = ok & (z > 0.012) & (z < 0.16) & (dxy < 0.075)
            near = obj & (dxy < 0.03)
            if near.sum() < 4:
                continue
            ztop = float(np.percentile(P[..., 2][near], 92))
            top = obj & (np.abs(P[..., 2] - ztop) < 0.007)
            lab, n = G.label(top)
            best = None
            for k in range(1, n + 1):
                mk = lab == k
                cnt = int((mk & (dxy < 0.03)).sum())
                if cnt > 0 and (best is None or cnt > best[0]):
                    best = (cnt, mk)
            if best is None:
                continue
            Q = P[best[1]]
            res[cam] = {"pts": Q, "ztop": ztop, "n": len(Q)}
            allpts.append(Q)
        if not res:
            return None
        if "wrist" in res and res["wrist"]["n"] >= 25:
            Q = res["wrist"]["pts"]
            ztop = res["wrist"]["ztop"]
        else:
            Q = np.concatenate(allpts, 0)
            ztop = float(np.median([r["ztop"] for r in res.values()]))
        rect = G.min_area_rect(Q[:, :2])
        if rect is None:
            return None
        cen, ang, dims = rect
        split = None
        w, h = dims
        clean = max(w, h) <= 1.15 * min(w, h) and 0.025 < max(w, h) < 0.075
        if clean:
            self.bin_sides.append(float(np.mean(dims)))
        elif max(w, h) > 0.03:
            # touching bins merged (or partial view): fit a square of the known side near the anchor
            side = float(np.median(self.bin_sides)) if self.bin_sides else float(min(max(min(w, h), 0.03), 0.07))
            sq = G.square_fit(Q[:, :2], np.asarray(c0, float), side)
            if sq is not None:
                cen, ang, frac = sq
                split = (round(max(w, h), 3), round(side, 3), round(frac, 2))
                dims = (side, side)
        wrist_ok = "wrist" in res and res["wrist"]["n"] >= 25
        if anchor is not None:
            dA = float(np.linalg.norm(cen - anchor))
            if not wrist_ok or dA > ANCHOR_TOL:
                log("perceive_bin: centre %.3f from track (wrist_ok=%s) -> use track xy" % (dA, wrist_ok))
                cen = np.asarray(anchor, float).copy()
        out = {"center": cen, "yaw": ang, "dims": dims, "top": ztop, "n": len(Q), "wrist_ok": wrist_ok,
               "cams": {k: v["n"] for k, v in res.items()}, "split": split}
        # second estimate from the other camera for logging/consistency
        return out

    def finger_axis_angle(self, maps):
        m = maps.get("wrist")
        if m is None or m[2] is None:
            return None
        a = m[2].u_axis_world()
        if math.hypot(a[0], a[1]) < 0.3:
            return None
        return math.atan2(a[1], a[0])


# ===================================================================== task logic
COLOR_WORDS = {"red": "red", "green": "green", "blue": "blue", "yellow": "yellow", "orange": "orange",
               "purple": "purple", "violet": "purple", "black": "black", "white": "white", "grey": "gray",
               "gray": "gray", "pink": "pink", "cyan": "cyan", "brown": "brown", "magenta": "purple"}


def parse_instruction(text):
    """-> list of (color, all_flag) in order."""
    t = (text or "").lower()
    out = []
    for m in re.finditer(r"hiding\s+(?:the\s+|a\s+|an\s+|all\s+|any\s+)?(?:(\w+)\s+)?cubes?", t):
        word = m.group(1)
        if word is None or word not in COLOR_WORDS:
            continue
        color = COLOR_WORDS[word]
        before = t[max(0, m.start() - 45):m.start()]
        seg = before.split(",")[-1]
        all_flag = bool(re.search(r"\b(all|every|each)\b", seg)) or m.group(0).endswith("cubes")
        out.append((color, all_flag))
    if not out:
        # fallback: colour words followed by "cube(s)" in order of appearance
        for m in re.finditer(r"(\w+)\s+cubes?\b", t):
            w = m.group(1)
            if w in COLOR_WORDS:
                before = t[max(0, m.start() - 45):m.start()].split(",")[-1]
                out.append((COLOR_WORDS[w], bool(re.search(r"\b(all|every|each)\b", before)) or
                            m.group(0).endswith("cubes")))
    return out


class Controller:
    def __init__(self, ctx):
        self.r = Robot(ctx)
        self.p = Perception(self.r)
        self.instruction = ""
        self.targets = []
        self.R0 = None
        self.start_pos = None
        self.round_steps = []   # steps at which new cube rounds were detected
        self.pressed = False
        self.picked = []
        self.yaw_total = 0.0
        self.t0 = time.time()

    # ---------------------------------------------------------- phases
    def run(self):
        r = self.r
        ob = r.observe()
        log("VERSION", VERSION, "step", r.step, "state", None if r.state is None else np.round(r.state, 4).tolist())
        log("images rgb", list(ob.rgb.keys()), "depth", list(ob.depth.keys()), "raw", ob.raw_images[:6])
        log("meta keys", list(ob.meta.keys())[:30])
        self._learn_order(ob)
        self.instruction = ob.instruction or U.find_str(ob.meta, "goal") or ""
        log("instruction:", self.instruction)
        self.targets = parse_instruction(self.instruction)
        log("targets", self.targets)
        if r.state is None:
            log("NO STATE in observation; sending zero rows to get one")
            r.send([[0, 0, 0, 0, 0, 0, -1]])
        self.R0 = r.R
        self.start_pos = r.pos
        try:
            self.p.init_scene(ob)
        except Ended:
            raise
        except Exception:
            log("init_scene error", traceback.format_exc())
            if self.p.table_z is None:
                self.p.table_z = 0.0
            if self.p.button is None:
                self.p.button = {"xy": r.pos[:2].copy(), "top": self.p.table_z + 0.03, "found": False}
        self.p.prev = {}
        self.p.scan(ob)
        self.watch_round1()
        # v2: press in the gap after round 1 (early press is allowed; deadline is step 90),
        # then watch rounds 2/3 from the high start pose where the wrist camera sees the table.
        self.press()
        self.to_observe_pose()
        self.watch_more()
        self.pick_all()
        log("DONE; picked", self.picked, "step", r.step, "complete", r.complete)
        # idle safely until the episode ends? No: returning leaves the robot where it stands.

    def _learn_order(self, ob):
        """Use labeled manifest files to check the inline image order."""
        if not ob.manifest_path:
            return
        try:
            man, frames = U.load_manifest_frames(ob.manifest_path)
            if not frames:
                return
            fr = frames[-1]
            files = U.frame_files(fr)
            fimg = {}
            for (cam, kind), (path, mpu) in files.items():
                if kind == "rgb":
                    fimg[cam] = np.array(U.load_file_image(path).convert("RGB")).astype(int)
            if "front" in ob.rgb and "front" in fimg and "wrist" in fimg:
                a = ob.rgb["front"].astype(int)
                d_same = np.abs(a - fimg["front"]).mean()
                d_swap = np.abs(a - fimg["wrist"]).mean()
                log("order check: inline-front vs file front %.2f / file wrist %.2f" % (d_same, d_swap))
                if d_swap + 1.0 < d_same:
                    log("IMAGE ORDER SWAPPED -> fixing")
                    self.r.order = {"rgb": "swapped", "depth": "swapped"}
                    ob.rgb["front"], ob.rgb["wrist"] = ob.rgb["wrist"], ob.rgb["front"]
                    if "front" in ob.depth and "wrist" in ob.depth:
                        ob.depth["front"], ob.depth["wrist"] = ob.depth["wrist"], ob.depth["front"]
        except Exception:
            log("learn_order error", traceback.format_exc()[-500:])

    def observe_scan(self):
        ob = self.r.observe()
        try:
            newc, nd, nw = self.p.scan(ob)
        except Ended:
            raise
        except Exception:
            log("scan error", traceback.format_exc()[-800:])
            return ob, 0, 0
        if newc:
            self._note_round(newc)
            log("step", self.r.step, "new tracks", newc, "tracks",
                [(t["color"], tuple(np.round(t["xy"], 3)), t["n"]) for t in self.p.tracks])
        if nw > 150:
            self.p.events.append((self.r.step, "white", nw))
        return ob, newc, nd

    def _note_round(self, newc):
        if newc <= 0:
            return
        s = self.r.step
        if not self.round_steps or s - self.round_steps[-1] > 15:
            self.round_steps.append(s)
            log("ROUND detected at step", s)

    def watch_round1(self):
        r = self.r
        seen = None
        while r.step < 24:
            r.hold(1, self.R0, grip=-1)
            ob, newc, nd = self.observe_scan()
            self._note_round(newc)
            if newc and seen is None:
                seen = r.step
            if seen is not None and r.step >= seen + 2:
                break
        log("round1 watch end step", r.step, "seen", seen)

    def ready_pose(self):
        b = self.p.button
        z = b["top"] + READY_H
        z = min(z, max(self.start_pos[2], b["top"] + 0.08))
        return np.array([b["xy"][0], b["xy"][1], z])

    def to_ready(self):
        r = self.r
        tgt = self.ready_pose()
        log("to_ready target", np.round(tgt, 3), "from", np.round(r.pos, 3))

        def cb():
            if r.step % 2 == 0:
                self.observe_scan()
        ok, why = r.move_to(tgt, self.R0, grip=1.0, timeout=40, per_step=cb, stall_ok=True)
        log("to_ready", ok, why, "step", r.step, "pos", np.round(r.pos, 3))
        r.hold(2, self.R0)
        self.p.prev = {}

    def watch_round2(self):
        r = self.r
        seen = None
        while r.step < PRESS_LATEST_START:
            r.hold(1, self.R0)
            ob, newc, nd = self.observe_scan()
            if r.step >= 45:
                self._note_round(newc)
                if newc and seen is None and r.step >= 50:
                    seen = r.step
            if seen is not None and r.step >= max(seen + 2, PRESS_EARLIEST):
                break
        log("round2 watch end step", r.step, "seen", seen, "rounds", self.round_steps)

    def relocalize_button(self):
        """Close-up wrist look at the button from above; refine xy/top (keeps old values on failure)."""
        r = self.r
        b = self.p.button
        try:
            r.hold(1, self.R0, grip=1.0)
            ob = r.observe()
            maps = self.p.get_maps(ob, cams=("wrist",))
            m = maps.get("wrist")
            if m is None:
                return False
            pts, valid, _ = m
            tz = self.p.table_z
            z = pts[..., 2] - tz
            dxy = np.linalg.norm(pts[..., :2] - np.asarray(b["xy"])[None, None, :], axis=-1)
            obj = valid & np.isfinite(z) & (z > 0.004) & (z < 0.09) & (dxy < 0.07)
            lab, n = G.label(obj)
            best = None
            for k in range(1, n + 1):
                mk = lab == k
                cnt = int((mk & (dxy < 0.025)).sum())
                if cnt > 0 and (best is None or cnt > best[0]):
                    best = (cnt, mk)
            if best is None:
                log("relocalize_button: nothing near estimate")
                return False
            P = pts[best[1]]
            ext = np.percentile(P[:, :2], 97, 0) - np.percentile(P[:, :2], 3, 0)
            top = float(np.percentile(P[:, 2], 95))
            hi = P[P[:, 2] > top - 0.006]
            cxy = hi[:, :2].mean(0) if len(hi) >= 3 else P[:, :2].mean(0)
            moved = float(np.linalg.norm(cxy - b["xy"]))
            log("relocalize_button: xy", np.round(cxy, 4), "top %.4f" % top, "ext", np.round(ext, 3),
                "n", len(P), "moved %.3f" % moved)
            if ext.max() > 0.12 or moved > 0.05 or len(P) < 15:
                log("relocalize_button: implausible, keep old")
                return False
            b["xy"] = cxy
            b["top"] = top
            return True
        except Ended:
            raise
        except Exception:
            log("relocalize_button error", traceback.format_exc()[-400:])
            return False

    def press_once(self, offset=(0.0, 0.0)):
        r = self.r
        b = self.p.button
        bx, by = np.asarray(b["xy"]) + np.asarray(offset)
        top = b["top"]
        above = np.array([bx, by, top + 0.05])
        ok, why = r.move_to(above, self.R0, grip=1.0, timeout=30, vmax=VMAX * 1.2, stall_ok=True)
        log("press: above", ok, why, "step", r.step, "pos", np.round(r.pos, 3))
        zs = [r.pos[2]]
        contact = None
        contact_z = None
        for k in range(30):
            z = r.pos[2]
            tgt_z = top - 0.035
            if contact is not None and k - contact >= 4:
                break
            if z < top - 0.025:
                log("press: reached depth z", round(z, 4))
                break
            dz = max(-VMAX_SLOW * (1.6 if contact is None else 1.0), tgt_z - z)
            exy = (np.array([bx, by]) - r.pos[:2]) * 0.8
            a = [exy[0] / POS_SCALE * r.axis_sign[0], exy[1] / POS_SCALE * r.axis_sign[1],
                 dz / POS_SCALE * r.axis_sign[2]] + list(r.rot_action(self.R0)) + [1.0]
            r.send([a])
            zs.append(r.pos[2])
            if contact is None and len(zs) >= 3 and (zs[-2] - zs[-1]) < 0.003 and (zs[-3] - zs[-2]) < 0.003 \
                    and zs[-1] < top + 0.04:
                contact = k
                contact_z = zs[-1]
                log("press: contact/stall at z", round(zs[-1], 4), "step", r.step)
        zmin = min(zs)
        log("press done step", r.step, "zmin %.4f" % zmin, "button top %.4f" % top,
            "contact_z", None if contact_z is None else round(contact_z, 4))
        r.move_to(np.array([bx, by, top + 0.06]), self.R0, grip=1.0, timeout=15, stall_ok=True)
        # a clear miss: we went >= 2 cm below the measured cap top (table or base edge)
        missed = zmin < top - 0.02 and (contact_z is None or contact_z < top - 0.02)
        return not missed

    def press(self):
        r = self.r
        b = self.p.button
        pre = np.array([b["xy"][0], b["xy"][1], b["top"] + 0.12])
        if pre[2] < r.pos[2] - 0.01:
            ok, why = r.move_to(pre, self.R0, grip=1.0, timeout=30, vmax=VMAX * 1.2, stall_ok=True)
            log("press: pre", ok, why, "step", r.step)
            self.relocalize_button()
        offsets = [(0.0, 0.0), (0.012, 0.0), (-0.012, 0.0), (0.0, 0.012), (0.0, -0.012)]
        for i, off in enumerate(offsets):
            ok = self.press_once(off)
            self.pressed = True
            if ok:
                break
            log("PRESS MISS suspected (attempt %d)" % i)
            if r.step > 72:
                break
            r.move_to(np.array([b["xy"][0], b["xy"][1], b["top"] + 0.12]), self.R0, grip=1.0, timeout=15,
                      stall_ok=True)
            self.relocalize_button()

    def to_observe_pose(self):
        r = self.r
        tgt = self.start_pos.copy()

        def cb():
            if r.step % 3 == 0:
                self.observe_scan()
        ok, why = r.move_to(tgt, self.R0, grip=-1.0, timeout=35, per_step=cb, stall_ok=True)
        log("to_observe", ok, why, "step", r.step, "pos", np.round(r.pos, 3))
        r.hold(2, self.R0, grip=-1.0)
        self.p.prev = {}

    def watch_more(self):
        """Keep watching while further reveal rounds are possible."""
        r = self.r
        while True:
            last = self.round_steps[-1] if self.round_steps else 56
            # expected next round start = last + period; watch until it would have appeared (+4)
            # and until any visible uncovered cubes of the latest round are covered.
            until = max(last + ROUND_PERIOD + 4, 104)
            if len(self.round_steps) >= 1:
                until = max(until, self.round_steps[-1] + ROUND_VISIBLE + 3)
            if r.step >= until or r.step >= 260:
                break
            r.hold(2, self.R0, grip=-1.0)
            ob, newc, nd = self.observe_scan()
            if r.step >= 50:
                self._note_round(newc)
        # make sure the last round has been covered
        last = self.round_steps[-1] if self.round_steps else 56
        while r.step < last + ROUND_VISIBLE + 3 and r.step < 260:
            r.hold(1, self.R0, grip=-1.0)
            self.observe_scan()
        log("watch_more end step", r.step, "rounds", self.round_steps, "tracks",
            [(t["color"], tuple(np.round(t["xy"], 3)), t["n"], t["first"], sorted(t["cams"])) for t in self.p.tracks])

    # ---------------------------------------------------------- picking
    def plan_targets(self):
        plan = []
        used = set()
        for color, all_flag in self.targets:
            tr = [i for i, t in enumerate(self.p.tracks) if t["color"] == color and i not in used]
            if not tr:
                log("NO TRACK for colour", color)
                continue
            if all_flag:
                nmax = max(self.p.tracks[j]["n"] for j in tr)
                keep = [j for j in tr if self.p.tracks[j]["n"] >= 3]
                if len(keep) < len(tr):
                    log("all-%s: dropped weak tracks" % color, [(j, self.p.tracks[j]["n"]) for j in tr if j not in keep])
                tr = keep
                for i in tr:
                    plan.append((color, i))
                    used.add(i)
            else:
                i = max(tr, key=lambda j: self.p.tracks[j]["n"])
                plan.append((color, i))
                used.add(i)
        return plan

    def pick_all(self):
        plan = self.plan_targets()
        log("PICK PLAN", [(c, i, tuple(np.round(self.p.tracks[i]["xy"], 3))) for c, i in plan])
        done_centers = []
        # order 'all' groups nearest-first
        k = 0
        while k < len(plan):
            color, i = plan[k]
            # group consecutive same-colour all-picks and order nearest-first from current pos
            grp = [plan[k]]
            j = k + 1
            while j < len(plan) and plan[j][0] == color:
                grp.append(plan[j])
                j += 1
            pos = self.r.pos[:2]
            while grp:
                grp.sort(key=lambda ci: np.linalg.norm(self.p.tracks[ci[1]]["xy"] - pos))
                c, ti = grp.pop(0)
                xy = self.p.tracks[ti]["xy"]
                if any(np.linalg.norm(xy - d) < 0.025 for d in done_centers):
                    log("skip duplicate target", c, ti)
                    continue
                ok, center = self.pick_one(c, xy)
                self.picked.append((c, ti, ok))
                if center is not None:
                    done_centers.append(center)
                pos = self.r.pos[:2]
            k = j

    def choose_rotation(self, ang):
        cands = [ang + kk * math.pi / 2 for kk in (-1, 0, 1)]
        best = min(cands, key=lambda a: (abs(self.yaw_total + a) > math.radians(110), abs(a) > math.radians(50),
                                          abs(a)))
        return best

    def center_wrist_on(self, xy, maps, R_ref, z_rel=0.05):
        """If the target is near the wrist image border, shift the EE so the wrist camera centres it."""
        r = self.r
        try:
            m = maps.get("wrist")
            if m is None or m[2] is None:
                return False
            mdl = m[2]
            P = np.array([xy[0], xy[1], self.p.table_z + z_rel])
            pr = mdl.project(P)
            if pr is None:
                return False
            u, v, d = (float(x) for x in pr)
            H, W = m[1].shape[:2]
            if d > 0 and 0.22 * W < u < 0.78 * W and 0.22 * H < v < 0.78 * H:
                return False
            # world point seen at the image centre on the plane z = table + z_rel
            ray = mdl.A @ np.array([W / 2.0, H / 2.0, 1.0])
            if abs(ray[2]) < 1e-6:
                return False
            dd = (P[2] - mdl.t[2]) / ray[2]
            if dd <= 0:
                return False
            Pc = ray * dd + mdl.t
            off = P[:2] - Pc[:2]
            n = float(np.linalg.norm(off))
            if n > 0.12:
                off = off / n * 0.12
            tgt = r.pos + np.array([off[0], off[1], 0.0])
            ok, why = r.move_to(tgt, R_ref, grip=-1.0, timeout=15, stall_ok=True)
            log("center_wrist: target at (u,v)=(%.0f,%.0f) d=%.3f -> shift %s" % (u, v, d, np.round(off, 3)), ok, why)
            r.hold(1, R_ref, grip=-1.0)
            return True
        except Ended:
            raise
        except Exception:
            log("center_wrist error", traceback.format_exc()[-300:])
            return False

    def pick_one(self, color, xy):
        r = self.r
        tz = self.p.table_z
        log("=== pick", color, "at", np.round(xy, 3), "step", r.step)
        R_cur = r.R
        hover = np.array([xy[0], xy[1], tz + HOVER_H])
        ok, why = r.move_to(hover, R_cur, grip=-1.0, timeout=45, stall_ok=True)
        log("hover", ok, why, np.round(r.pos, 3))
        info = None
        R_ref = R_cur
        for it in range(3):
            r.hold(1, R_ref, grip=-1.0)
            ob = r.observe()
            maps = self.p.get_maps(ob)
            if self.center_wrist_on(xy, maps, R_ref):
                ob = r.observe()
                maps = self.p.get_maps(ob)
            info = self.p.perceive_bin(ob, np.asarray(xy, float) if info is None else info["center"], maps,
                                       anchor=np.asarray(xy, float))
            if info is None:
                log("bin not perceived (iter %d)" % it)
                if it == 0:
                    # look from slightly higher and shifted toward robot base
                    r.move_to(hover + np.array([0, 0, 0.05]), R_ref, timeout=10, stall_ok=True)
                continue
            phi = self.p.finger_axis_angle(maps)
            d_off = np.linalg.norm(info["center"] - xy)
            log("bin it%d center" % it, np.round(info["center"], 3), "off %.3f" % d_off, "yaw %.1f" % math.degrees(info["yaw"]),
                "dims", np.round(info["dims"], 3), "split", info.get("split"), "top %.3f" % (info["top"] - tz), "n", info["n"], info["cams"],
                "finger %s" % (None if phi is None else round(math.degrees(phi), 1)))
            if d_off > 0.05:
                log("bin centre far from cube track; falling back to track xy")
                info["center"] = np.asarray(xy, float)
            if phi is None or not info.get("wrist_ok", True):
                break
            need = G.wrap_half_pi(info["yaw"] - phi)
            if abs(need) < math.radians(4) and it > 0:
                break
            rot = self.choose_rotation(need)
            self.yaw_total += rot
            R_ref = G.rotz(rot) @ r.R
            tgt = np.array([info["center"][0], info["center"][1], tz + HOVER_H])
            ok, why = r.move_to(tgt, R_ref, grip=-1.0, timeout=25, stall_ok=True)
            log("rotate %.1f deg" % math.degrees(rot), ok, why, "yaw_total %.1f" % math.degrees(self.yaw_total))
            if abs(need) < math.radians(4):
                break
        if info is None:
            log("PICK FAIL: no bin perceived for", color)
            info = {"center": np.asarray(xy, float), "top": tz + 0.06, "yaw": 0.0, "dims": (0.05, 0.05)}
        R_ref = r.R  # hold the aligned orientation
        c = info["center"]
        top = info["top"]
        for attempt in range(3):
            zg = max(top - GRASP_DEPTH, tz + TABLE_CLEAR)
            pre = np.array([c[0], c[1], top + 0.06])
            r.move_to(pre, R_ref, grip=-1.0, timeout=30, stall_ok=True)
            r.hold(2, R_ref, grip=-1.0)  # let fingers open fully
            ok, why = r.move_to(np.array([c[0], c[1], zg]), R_ref, grip=-1.0, vmax=VMAX_SLOW * 1.5, tol=0.003,
                                timeout=25, stall_ok=False)
            log("descend", ok, why, "z %.3f target %.3f" % (r.pos[2], zg), "step", r.step)
            if not ok and r.pos[2] > min(zg + 0.012, top - 0.006):
                log("descent blocked (collision with bin top?) -> re-perceive and shift")
                r.move_to(pre + np.array([0, 0, 0.04]), R_ref, grip=-1.0, timeout=15, stall_ok=True)
                ob = r.observe()
                maps = self.p.get_maps(ob)
                info2 = self.p.perceive_bin(ob, c, maps, anchor=np.asarray(xy, float))
                if info2 is not None and np.linalg.norm(info2["center"] - c) < 0.04:
                    c = info2["center"]
                    top = info2["top"]
                continue
            r.hold(CLOSE_STEPS, R_ref, grip=1.0)
            op = r.opening
            log("closed: opening %.4f" % op)
            if op < 0.004:
                log("grasp missed (fully closed)")
                r.hold(OPEN_STEPS // 2, R_ref, grip=-1.0)
                r.move_to(pre, R_ref, grip=-1.0, timeout=15, stall_ok=True)
                ob = r.observe()
                maps = self.p.get_maps(ob)
                info2 = self.p.perceive_bin(ob, c, maps, anchor=np.asarray(xy, float))
                if info2 is not None and np.linalg.norm(info2["center"] - c) < 0.04:
                    c = info2["center"]
                    top = info2["top"]
                    phi = self.p.finger_axis_angle(maps)
                    if phi is not None:
                        need = G.wrap_half_pi(info2["yaw"] - phi)
                        if abs(need) > math.radians(5):
                            rot = self.choose_rotation(need)
                            self.yaw_total += rot
                            R_ref = G.rotz(rot) @ r.R
                continue
            lift = np.array([c[0], c[1], zg + LIFT_H])
            ok, why = r.move_to(lift, R_ref, grip=1.0, vmax=VMAX * 0.8, timeout=25, stall_ok=True)
            r.hold(4, R_ref, grip=1.0)
            op2 = r.opening
            log("lifted", ok, why, "z %.3f opening %.4f" % (r.pos[2], op2), "step", r.step)
            try:
                ob = r.observe()
                maps = {"front": self.p._front_map(ob)}
                dets = self.p.color_detections(ob, "front", maps["front"], use_static_filter=False)
                seen = [(d["color"], round(float(np.linalg.norm(d["xy"] - c)), 3)) for d in dets
                        if np.linalg.norm(d["xy"] - c) < 0.06]
                near = sorted(seen, key=lambda x: x[1])
                verdict = "unknown" if not near else ("confirmed" if any(cc == color and dd < 0.035 for cc, dd in near)
                                                      else "MISMATCH?")
                log("revealed under bin (front cam):", seen, "verdict", verdict)
            except Ended:
                raise
            except Exception:
                pass
            held = op2 > 0.004
            # put it back down where it was and release
            r.move_to(np.array([c[0], c[1], zg + 0.004]), R_ref, grip=1.0, vmax=VMAX * 0.8, timeout=25,
                      stall_ok=True)
            r.hold(OPEN_STEPS, R_ref, grip=-1.0)
            r.move_to(np.array([c[0], c[1], top + 0.07]), R_ref, grip=-1.0, timeout=20, stall_ok=True)
            if held:
                log("PICK OK", color, "step", r.step)
                return True, c
            log("bin dropped during lift; retry")
        log("PICK FAILED after retries", color)
        return False, c


def main(ctx):
    global T_START
    T_START = time.time()
    try:
        U.set_log(os.environ.get("BU_LOG", "/workspace/output/controller_log.txt"))
    except Exception:
        pass
    ctl = None
    try:
        ctl = Controller(ctx)
        ctl.run()
    except Ended:
        log("ended")
    except Exception:
        log("FATAL", traceback.format_exc())
        # keep the episode alive safely: nothing else to do
    finally:
        try:
            if ctl is not None:
                log("final step", ctl.r.step, "calls", ctl.r.n_calls, "wall %.1f" % (time.time() - ctl.t0),
                    "picked", ctl.picked, "complete", ctl.r.complete)
        except Exception:
            pass
