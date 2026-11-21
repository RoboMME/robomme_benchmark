"""VideoPlaceButton controller (RAX, interaction-only R1) — written offline, see TRANSITIONS.md.

Pipeline:
  INIT      observe; read instruction + conditioning-video manifest
  VIDEO     per-frame states -> contact events (place / grasp / press) -> chosen target
  SCENE     front depth -> world (fitted with robomme_pixel_to_position) -> table, cube, blobs
  HOLDCHK   is the cube already in the hand? (finger squeeze test)
  [PRESS]   only if the instruction itself orders a button press
  PICK      hover, (wrist refine), align yaw, descend, close, lift, verify fingers
  PLACE     travel high, descend over target, open, lift
  VERIFY    re-observe; cube near target? else retry PICK (<=3 attempts)
  IDLE      hold until the episode ends or a few steps pass
Every phase is wrapped; any exception falls through to the next sensible phase.
"""
import glob
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np

from io_utils import log, save_json, reply_json, find_key, load_manifest, WS, OUT
from geom import wrap_quarter, wrap_pi
from robot import Robot, EpisodeOver
import perception as per
import video as vid

VERSION = "vpb-1.1"
TRAVEL_DZ = 0.16      # travel height above table (m)
HOVER_DZ = 0.10       # pre-grasp / pre-place hover above grasp/release height
MAX_ATTEMPTS = 3
DEFAULT_CUBE_H = 0.04


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.rob = Robot(ctx)
        self.front_map = None
        self.tz = None
        self.cube_h = None
        self.info = {"version": VERSION, "phases": []}
        self.events = []
        self.hist = []
        self.target = None
        self.press_xy = None
        self.cube_xy = None
        self.cube_yaw = None
        self.grasp_z = None
        self.t0 = time.time()

    # ------------------------------------------------------------ helpers
    def phase(self, name, **kv):
        rec = {"phase": name, "t": round(time.time() - self.t0, 2), "step": self.rob.step_id}
        rec.update(kv)
        self.info["phases"].append(rec)
        log("PHASE " + name, **kv)

    def time_left(self):
        return 1100 - (time.time() - self.t0)

    def perceive(self, cam="front"):
        """Current observation with depth for `cam`: observe; else align to a saved 5-step frame."""
        r = self.rob.observe()
        o = per.obs_from_reply(r)
        if cam in o.depth:
            return r, o
        rob = self.rob
        try:
            if rob.step_id is not None and not rob.ended:
                k = (5 - rob.step_id % 5) % 5 or 5
                rs = rob.step([[0, 0, 0, 0, 0, 0, rob.grip]] * k)
                if rs is not None:
                    o2 = per.obs_from_reply(rs)
                    if cam in o2.depth:
                        log("depth from aligned step manifest", cam=cam, step=rob.step_id)
                        # step id of the saved frame is the current one; use a fresh observe for json
                        r2 = rob.observe()
                        o2.json = reply_json(r2) or o2.json
                        o2.step_id = rob.step_id
                        return r2, o2
        except EpisodeOver:
            raise
        except Exception as e:
            log("aligned perceive failed", err=str(e)[:300])
        return r, o

    def front_scene(self, result=None):
        """Observe (unless result given) and analyse the front camera."""
        try:
            if result is None:
                result, o = self.perceive("front")
            else:
                o = per.obs_from_reply(result)
            if "front" not in o.depth:
                log("no front depth in observation", src=o.src)
                return None, o
            if self.front_map is None:
                self.fit_front(o)
            if self.front_map is None:
                return None, o
            P = self.front_map.backproject(o.depth["front"])
            sc = per.analyze_scene(P, o.rgb.get("front"), tz=self.tz)
            return sc, o
        except EpisodeOver:
            raise
        except Exception as e:
            log("front_scene failed", err=str(e), tb=traceback.format_exc()[-800:])
            return None, None

    def fit_front(self, o):
        m = None
        try:
            m = per.fit_camera_map(self.ctx, o, "front")
        except Exception as e:
            log("front fit via tool failed", err=str(e)[:300])
        cm = per.calib_map(o.json, "front")
        if m is not None and cm is not None:
            try:
                d = o.depth["front"]
                P1 = m.backproject(d)
                P2 = cm.backproject(d)
                diff = np.nanmedian(np.linalg.norm(P1 - P2, axis=-1))
                log("fit vs calibration median diff", diff=float(diff))
                self.info["calib_vs_fit_m"] = float(diff)
            except Exception:
                pass
        if m is None and cm is not None:
            log("using calibration map for front (tool fit unavailable)")
            m = cm
        self.front_map = m
        self.info["front_map"] = None if m is None else {"src": m.source, "res": m.residual,
                                                         "M": m.M.tolist(), "t": m.t.tolist()}

    # ------------------------------------------------------------ phases
    def run(self):
        rob = self.rob
        r0 = rob.observe()
        j0 = reply_json(r0)
        st = rob.status()
        instr = find_key(j0, "instruction") or find_key(st, "instruction") or ""
        if not isinstance(instr, str):
            instr = json.dumps(instr)
        self.info["instruction"] = instr
        self.info["state0"] = None if rob.state is None else rob.state.tolist()
        self.info["limits"] = {"remaining": rob.remaining, "limit": rob.limit}
        self.phase("INIT", instruction=instr, state=self.info["state0"], remaining=rob.remaining,
                   keys=list(j0.keys())[:40])
        try:
            save_json("observe0.json", _strip(j0))
            save_json("status0.json", _strip(st))
        except Exception:
            pass
        if rob.state is None:
            log("no state in observation; trying a zero step")
            rob.hold(1, grip=1.0)
        # ---------------- VIDEO
        man, mpath = load_manifest(j0)
        if man is None:
            man, mpath = load_manifest(st)
        if man is None:
            man, mpath = _glob_manifest()
        frames_v, frames_x = vid.video_frames(man)
        S, T = vid.states_of(frames_v)
        self.phase("VIDEO-LOAD", manifest=str(mpath), n_video=len(frames_v), n_states=int(len(S)),
                   n_exec=len(frames_x), sampling=find_key(j0, "video_sampling"))
        # ---------------- SCENE (initial, from execution boundary frame)
        sc0 = None
        try:
            sc0, o0 = self.front_scene(r0)
            if sc0 is None and frames_x:
                pass
            if sc0 is not None:
                self.tz = sc0["tz"]
        except EpisodeOver:
            raise
        except Exception as e:
            log("initial scene failed", err=str(e))
        # ---------------- events
        try:
            self.events, einfo = vid.detect_events(S, T, self.tz)
        except Exception as e:
            self.events, einfo = [], {"err": str(e)}
        self.hist = vid.cube_history(self.events)
        if not any(e["type"] == "place" for e in self.events):
            try:
                self.image_fallback(frames_v, mpath, sc0)
            except EpisodeOver:
                raise
            except Exception as e:
                log("image fallback failed", err=str(e), tb=traceback.format_exc()[-600:])
        plan = vid.parse_instruction(instr)
        self.target, why = vid.choose_target(plan, self.events, self.hist)
        presses = [e for e in self.events if e["type"] == "press"]
        self.press_xy = presses[0]["xy"] if presses else None
        if self.tz is None:
            zs = [e["z"] for e in self.events if e["type"] in ("place", "grasp")]
            self.tz = (min(zs) - 0.02) if zs else 0.0
            log("table height from video/default", tz=self.tz)
        zs = [e["z"] for e in self.events if e["type"] in ("place", "grasp")]
        if zs and sc0 is not None:
            tz_v = float(np.median(zs)) - 0.02
            self.info["tz_video_est"] = tz_v
            if abs(self.tz - tz_v) > 0.05:
                log("table height disagrees with video contacts; re-estimating near video", tz=self.tz, tz_video=tz_v)
                try:
                    P = self.front_map.backproject(o0.depth["front"])
                    z = P[..., 2][np.isfinite(P).all(-1)]
                    zz = z[np.abs(z - tz_v) < 0.03]
                    if zz.size > 500:
                        hist, edges = np.histogram(zz, bins=np.arange(tz_v - 0.03, tz_v + 0.0301, 0.002))
                        k = int(np.argmax(hist))
                        self.tz = float(0.5 * (edges[k] + edges[k + 1]))
                        sc0 = per.analyze_scene(P, o0.rgb.get("front"), tz=self.tz)
                        log("table height re-estimated", tz=self.tz)
                except Exception as e:
                    log("tz re-estimate failed", err=str(e))
        self.info.update({"events": self.events, "event_info": einfo, "plan": plan, "target": self.target,
                          "target_why": why, "tz": self.tz, "hist": self.hist})
        self.phase("VIDEO-EVENTS", events=[(e["type"], round(e["t"], 2), np.round(e["xy"], 3).tolist(),
                                            round(e["z"], 3)) for e in self.events],
                   plan={k: plan[k] for k in ("anchor", "k", "press_k", "use_last", "do_press")},
                   target=None if self.target is None else np.round(self.target["xy"], 3).tolist(), why=why)
        _save_traj(S, T)
        if sc0 is not None:
            self._log_scene("SCENE0", sc0)
            self._check_targets(sc0)
        save_json("analysis.json", self.info)
        if self.target is None:
            self.target = self._fallback_target(sc0)
            log("fallback target", target=self.target)
        # ---------------- hold check
        holding = self.hold_check()
        rob.init_ref()
        # lift if low
        if rob.pos[2] < self.tz + 0.10:
            rob.move_to([rob.pos[0], rob.pos[1], self.tz + TRAVEL_DZ], vmax=0.3, max_steps=40, label="initial lift")
        if plan.get("do_press") and plan.get("press_first"):
            self.do_press(holding)
        # ---------------- pick & place attempts
        done = False
        for attempt in range(MAX_ATTEMPTS):
            if self.target is None:
                break
            if not holding:
                ok = self.pick(attempt)
                if not ok:
                    self.phase("PICK-FAIL", attempt=attempt)
                    continue
            ok = self.place()
            holding = False
            res = self.verify()
            if res is True or rob.ended:
                done = True
                break
            if res is None:     # unknown: do not disturb
                done = True
                break
            if self.time_left() < 200 or (rob.remaining is not None and rob.remaining < 80):
                break
        if plan.get("do_press") and not plan.get("press_first"):
            self.do_press(False)
        self.idle()

    # ------------------------------------------------------------
    def image_fallback(self, frames_v, mpath, sc0):
        """Video events from images when per-frame states are unusable."""
        if sc0 is None or self.front_map is None:
            log("image fallback impossible", have_scene=sc0 is not None)
            return
        cube = per.pick_cube(sc0)
        others = [c for c in sc0["comps"] if cube is None or np.linalg.norm(np.asarray(c["xy"]) - cube["xy"]) > 0.03]
        others = [c for c in others if 0.008 < c["top"] < 0.10 and c["n"] >= 10 and max(c["allext"]) < 0.15]
        button = max(others, key=lambda c: c["n"]) if others else None

        def load_frame(f):
            ff = per.frame_files(f, mpath)
            rgb = per.load_rgb(ff[("front", "rgb")][0]) if ("front", "rgb") in ff else None
            dp = ff.get(("front", "depth"))
            depth = per.load_depth(dp[0], dp[1]) if dp else None
            return rgb, depth

        ch = cube["top"] if cube is not None else DEFAULT_CUBE_H
        ev, hist, info = vid.image_history(frames_v, self.front_map, self.tz, cube, button, load_frame, cube_h=ch)
        self.phase("IMAGE-FALLBACK", n_events=len(ev), info=info,
                   cube=None if cube is None else np.round(cube["xy"], 3).tolist(),
                   button=None if button is None else np.round(button["xy"], 3).tolist(),
                   events=[(e["type"], round(e["t"], 2), np.round(e["xy"], 3).tolist()) for e in ev])
        if ev or hist:
            self.events, self.hist = ev, hist

    def _log_scene(self, name, sc):
        comps = [{"xy": np.round(c["xy"], 3).tolist(), "top": round(c["top"], 3), "hmax": round(c["hmax"], 3),
                  "n": c["n"], "ext": None if c["ext"] is None else np.round(c["ext"], 3).tolist(),
                  "yaw": None if c["yaw"] is None else round(math.degrees(c["yaw"]), 1),
                  "color": None if c["color"] is None else np.round(c["color"], 0).tolist()} for c in sc["comps"]]
        blobs = [{"xy": np.round(b["xy"], 3).tolist(), "n": b["n"], "color": np.round(b["color"], 0).tolist()}
                 for b in sc["blobs"]]
        self.info.setdefault("scenes", {})[name] = {"tz": sc["tz"], "comps": comps, "blobs": blobs,
                                                    "table_frac": sc["table_frac"]}
        self.phase(name, tz=round(sc["tz"], 4), comps=comps[:12], blobs=blobs[:12])

    def _check_targets(self, sc):
        """Compare video placement locations with flat blobs / raised comps now (diagnostic)."""
        out = []
        for h in self.hist:
            xy = np.asarray(h[1])
            db = min([float(np.linalg.norm(np.asarray(b["xy"]) - xy)) for b in sc["blobs"]] or [9])
            dc = min([float(np.linalg.norm(np.asarray(c["xy"]) - xy)) for c in sc["comps"]] or [9])
            out.append({"xy": np.round(xy, 3).tolist(), "kind": h[3], "nearest_blob": round(db, 3),
                        "nearest_comp": round(dc, 3)})
        self.info["target_check"] = out
        log("video locations vs current scene", locs=out)

    def _fallback_target(self, sc):
        if sc is None or not sc["blobs"]:
            return None
        b = max(sc["blobs"], key=lambda b: b["n"])
        return {"xy": np.asarray(b["xy"]), "z": self.tz + 0.025, "t": 0, "kind": "blob-fallback"}

    def hold_check(self):
        rob = self.rob
        f0 = rob.fingers
        if f0 >= 0.072:
            rob.grip = -1.0
            self.phase("HOLDCHK", fingers=f0, holding=False, why="open")
            return False
        rob.grip = 1.0
        if f0 < 0.006:
            self.phase("HOLDCHK", fingers=f0, holding=False, why="closed empty")
            return False
        f1 = rob.gripper(1.0, n=6)
        holding = f1 > 0.008 and (f0 - f1) < 0.012
        self.phase("HOLDCHK", fingers0=f0, fingers1=f1, holding=holding)
        if holding:
            self.cube_h = min(0.06, max(0.02, f1))
        return holding

    def _expected_cube(self):
        """Prior cube location from the video (last resting place)."""
        if self.hist:
            h = self.hist[-1]
            return np.asarray(h[1]), h[2]
        return None, None

    def locate_cube(self, sc, prior_xy, gate=True):
        """Cube candidate; on the first attempt gated to places the video says the cube rested."""
        exp_top = self.cube_h or self._video_cube_top()
        cands = []
        for c in sc["comps"]:
            if self.press_xy is not None and np.linalg.norm(np.asarray(c["xy"]) - self.press_xy) < 0.04:
                continue
            cands.append(c)
        sc2 = dict(sc, comps=cands)
        locs = [np.asarray(h[1]) for h in self.hist]
        if gate and locs:
            near = [c for c in cands if min(np.linalg.norm(np.asarray(c["xy"]) - l) for l in locs) < 0.06]
            c = per.pick_cube(dict(sc, comps=near), prior_xy=prior_xy, exp_top=exp_top, ee=self.rob.pos)
            if c is not None:
                return c
            log("no cube candidate near video resting places", n_cands=len(cands))
            return None
        return per.pick_cube(sc2, prior_xy=prior_xy, exp_top=exp_top, ee=self.rob.pos)

    def _video_cube_top(self):
        zs = [e["z"] - self.tz for e in self.events if e["type"] in ("place", "grasp")]
        if zs and self.tz is not None:
            top = 2.0 * float(np.median(zs))
            if 0.015 < top < 0.08:
                return top
        return None

    def pick(self, attempt):
        rob = self.rob
        self.phase("PICK", attempt=attempt)
        prior_xy, prior_z = self._expected_cube()
        if attempt > 0 or self.cube_xy is not None:
            prior_xy = self.cube_xy if self.cube_xy is not None else prior_xy
        sc, o = self.front_scene()
        cube = self.locate_cube(sc, prior_xy, gate=(attempt == 0 and self.cube_xy is None)) if sc is not None else None
        if cube is not None and prior_xy is not None and np.linalg.norm(cube["xy"] - prior_xy) > 0.12 and attempt == 0:
            log("cube candidate far from video prior", cand=cube["xy"], prior=prior_xy)
        if cube is not None:
            xy = np.asarray(cube["xy"], dtype=float)
            top = cube["top"]
            self.cube_h = self.cube_h or min(0.06, max(0.02, top))
            gz = self.tz + max(0.012, top - 0.5 * self.cube_h) if top > self.cube_h + 0.004 else self.tz + 0.5 * top
            yaw = cube["yaw"]
            src = "front"
        elif prior_xy is not None:
            xy = np.asarray(prior_xy, dtype=float)
            gz = prior_z if prior_z is not None else self.tz + 0.02
            yaw = None
            src = "video-prior"
        else:
            log("no cube location")
            return False
        yaw_src = src
        pl = [e for e in self.events if e["type"] == "place" and e.get("yaw") is not None]
        if pl and attempt == 0 and np.linalg.norm(np.asarray(pl[-1]["xy"]) - xy) < 0.02:
            # cube still where the video put it: the gripper yaw at that release is the cube yaw
            log("using video release yaw", video_yaw=math.degrees(pl[-1]["yaw"]),
                front_yaw=None if yaw is None else math.degrees(yaw))
            yaw = pl[-1]["yaw"]
            yaw_src = "video"
        elif yaw is None and pl:
            yaw = pl[-1]["yaw"]
        if prior_z is not None and src == "front" and abs(prior_z - gz) < 0.015 and attempt == 0:
            gz = 0.5 * (gz + prior_z)
        gz = max(gz, self.tz + 0.012)
        self.phase("CUBE", src=src, xy=np.round(xy, 4).tolist(), grasp_z=round(gz, 4),
                   yaw=None if yaw is None else round(math.degrees(yaw), 1),
                   cand=None if cube is None else {"top": round(cube["top"], 3), "n": cube["n"],
                                                   "score": round(cube["score"], 3)})
        # hover (gripper open)
        hover = gz + HOVER_DZ
        rob.move_to([rob.pos[0], rob.pos[1], max(rob.pos[2], self.tz + TRAVEL_DZ)], vmax=0.3, max_steps=30,
                    label="raise")
        if rob.fingers < 0.07:
            rob.gripper(-1.0, n=10)
        rob.grip = -1.0
        rob.move_to([xy[0], xy[1], max(hover, self.tz + 0.08)], vmax=0.4, max_steps=90, label="to hover")
        # wrist refinement
        try:
            w = self.wrist_refine(xy, cube)
            if w is not None:
                xy, wyaw = w
                if wyaw is not None and yaw_src != "video":
                    yaw = wyaw
        except EpisodeOver:
            raise
        except Exception as e:
            log("wrist refine error", err=str(e)[:300])
        if yaw is not None:
            dyaw = wrap_quarter(yaw - rob.yaw())
            rob.R_ref = None
            rob.init_ref()
            rob.set_yaw_ref(dyaw)
            log("yaw align", cube_yaw=math.degrees(yaw), ee_yaw=math.degrees(rob.yaw()), dyaw=math.degrees(dyaw))
        rob.move_to([xy[0], xy[1], hover], vmax=0.25, tol=0.004, max_steps=40, label="hover align")
        # descend
        rob.move_to([xy[0], xy[1], gz + 0.03], vmax=0.3, tol=0.006, max_steps=40, label="descend1")
        ok, dist, _ = rob.move_to([xy[0], xy[1], gz], vmax=0.15, tol=0.004, max_steps=30, label="descend2")
        log("at grasp", pos=rob.pos, target=[xy[0], xy[1], gz], ok=ok, dist=dist)
        f = rob.gripper(1.0, n=16, min_steps=8)
        self.phase("GRASP", fingers=f)
        if f < 0.006 or f > 0.07:
            rob.gripper(-1.0, n=8)
            rob.move_to([rob.pos[0], rob.pos[1], gz + HOVER_DZ], vmax=0.3, max_steps=30, label="lift-miss")
            self.cube_xy = None
            return False
        self.cube_h = self.cube_h or min(0.06, max(0.02, f))
        self.grasp_z = gz
        rob.move_to([rob.pos[0], rob.pos[1], self.tz + TRAVEL_DZ], vmax=0.3, max_steps=40, label="lift")
        f2 = rob.fingers
        self.phase("LIFT", fingers=f2, pos=rob.pos)
        if f2 < 0.006 or f2 > 0.07:
            rob.gripper(-1.0, n=8)
            self.cube_xy = None
            return False
        # visual check: is something still at the old location?
        try:
            sc2, _ = self.front_scene()
            if sc2 is not None:
                c2 = per.pick_cube(sc2, prior_xy=xy, exp_top=self.cube_h, avoid_xy=[self.press_xy])
                if c2 is not None and np.linalg.norm(np.asarray(c2["xy"]) - xy) < 0.02 and abs(c2["top"] - (self.cube_h or 0.04)) < 0.012:
                    log("cube still visible at grasp location after lift", c=c2["xy"], top=c2["top"], fingers=f2)
                    self.info["lift_visual_conflict"] = True
        except EpisodeOver:
            raise
        except Exception:
            pass
        self.cube_xy = None
        return True

    def wrist_refine(self, xy, cube):
        r, o = self.perceive("wrist")
        if "wrist" not in o.depth:
            log("wrist refine: no wrist depth", src=o.src)
            return None
        m = per.fit_camera_map(self.ctx, o, "wrist", n=30)
        if m is None:
            m = per.calib_map(o.json, "wrist")
            if m is None:
                return None
        P = m.backproject(o.depth["wrist"])
        valid = np.isfinite(P).all(-1)
        h = P[..., 2] - self.tz
        top = cube["top"] if cube is not None else (self.cube_h or self._video_cube_top() or DEFAULT_CUBE_H)
        near = valid & (np.linalg.norm(np.nan_to_num(P[..., :2], nan=9.0) - xy, axis=-1) < 0.06)
        band = near & (h > top - 0.012) & (h < top + 0.012)
        pts = P[band]
        log("wrist refine", n_band=int(band.sum()), n_near=int(near.sum()), top=top)
        if len(pts) < 25:
            return None
        from geom import min_area_yaw
        # keep the largest cluster around the median
        med = np.median(pts[:, :2], axis=0)
        pts = pts[np.linalg.norm(pts[:, :2] - med, axis=1) < 0.035]
        if len(pts) < 25:
            return None
        yaw, ext = min_area_yaw(pts[:, :2])
        cxy = pts[:, :2].mean(0)
        size = None
        if cube is not None and cube.get("ext") is not None:
            size = float(np.mean(cube["ext"]))
        size = size or self.cube_h or DEFAULT_CUBE_H
        if ext is None or abs(ext[0] - size) > 0.01 or abs(ext[1] - size) > 0.01:
            log("wrist refine: implausible extent (partial view?)", ext=ext, size=size)
            return None
        if np.linalg.norm(cxy - xy) > 0.02:
            log("wrist refine disagrees with front", wrist=cxy, front=xy)
            return None
        self.phase("WRIST", xy=np.round(cxy, 4).tolist(), yaw=None if yaw is None else round(math.degrees(yaw), 1),
                   ext=None if ext is None else np.round(ext, 3).tolist(), n=len(pts))
        return cxy, yaw

    def place(self):
        rob = self.rob
        tx = np.asarray(self.target["xy"], dtype=float)
        ch = self.cube_h or DEFAULT_CUBE_H
        zr = self.target.get("z")
        if zr is None or not np.isfinite(zr) or zr < self.tz + 0.3 * ch or zr > self.tz + ch + 0.03:
            zr = self.tz + 0.5 * ch + 0.006
        zr = zr + 0.004
        self.phase("PLACE", target=np.round(tx, 4).tolist(), release_z=round(zr, 4), cube_h=ch)
        rob.move_to([rob.pos[0], rob.pos[1], self.tz + TRAVEL_DZ], vmax=0.3, max_steps=30, label="place raise")
        rob.move_to([tx[0], tx[1], self.tz + TRAVEL_DZ], vmax=0.4, max_steps=100, label="to target")
        rob.move_to([tx[0], tx[1], zr + 0.04], vmax=0.3, tol=0.005, max_steps=50, label="place descend1")
        ok, dist, _ = rob.move_to([tx[0], tx[1], zr], vmax=0.12, tol=0.004, max_steps=30, label="place descend2")
        log("at release", pos=rob.pos, ok=ok, dist=dist, fingers=rob.fingers)
        rob.gripper(-1.0, n=10)
        rob.move_to([rob.pos[0], rob.pos[1], zr + HOVER_DZ], vmax=0.25, max_steps=30, label="retreat")
        return True

    def verify(self):
        """True: cube seen on target; False: one cube-like object clearly elsewhere (retry); None: unknown."""
        rob = self.rob
        sc, o = self.front_scene()
        if rob.ended:
            return True
        if sc is None:
            return None
        self._log_scene("SCENE-AFTER", sc)
        tx = np.asarray(self.target["xy"])
        ch = self.cube_h or self._video_cube_top() or DEFAULT_CUBE_H
        cubes = []
        for c in sc["comps"]:
            if self.press_xy is not None and np.linalg.norm(np.asarray(c["xy"]) - self.press_xy) < 0.045:
                continue
            ext = c["ext"] or c["allext"]
            if abs(c["top"] - ch) > 0.013 or c["hmax"] > ch + 0.03 or ext is None or max(ext) > 0.075 or c["n"] < 8:
                continue
            cubes.append(c)
        d = [float(np.linalg.norm(np.asarray(c["xy"]) - tx)) for c in cubes]
        self.phase("VERIFY", cubes=[np.round(c["xy"], 3).tolist() for c in cubes], dists=[round(x, 3) for x in d])
        if any(x < 0.03 for x in d):
            return True
        far = [c for c, x in zip(cubes, d) if x > 0.05]
        if len(cubes) == 1 and len(far) == 1:
            self.cube_xy = np.asarray(far[0]["xy"])
            return False
        return None

    def do_press(self, holding):
        rob = self.rob
        if self.press_xy is None:
            log("press requested but no button location")
            return
        pr = [e for e in self.events if e["type"] == "press"][0]
        self.phase("PRESS", xy=np.round(pr["xy"], 3).tolist(), z=pr["z"])
        if not holding:
            rob.gripper(1.0, n=8)
        rob.move_to([pr["xy"][0], pr["xy"][1], self.tz + TRAVEL_DZ], vmax=0.4, max_steps=90, label="to button")
        rob.move_to([pr["xy"][0], pr["xy"][1], pr["z"] + 0.03], vmax=0.3, max_steps=40, label="button hover")
        rob.move_to([pr["xy"][0], pr["xy"][1], pr["z"] - 0.006], vmax=0.12, max_steps=20, label="press")
        rob.hold(2)
        rob.move_to([rob.pos[0], rob.pos[1], self.tz + TRAVEL_DZ], vmax=0.3, max_steps=30, label="unpress")

    def idle(self):
        rob = self.rob
        self.phase("IDLE")
        for _ in range(6):
            rob.hold(2)
        self.phase("END", complete=rob.complete, ended=rob.ended)


def _strip(j):
    try:
        s = json.dumps(j, default=str)
        if len(s) > 200000:
            return {"truncated": s[:200000]}
        return j
    except Exception:
        return {"unserialisable": True}


def _glob_manifest():
    cands = []
    for p in glob.glob(str(WS / "runtime" / "autonomous_controller" / "runs" / "*" / "frames" / "**" / "manifest.json"),
                       recursive=True):
        try:
            m = json.loads(Path(p).read_text())
            if any(str(f.get("phase")) == "video_demo" for f in m.get("frames", [])):
                cands.append((os.path.getmtime(p), p, m))
        except Exception:
            continue
    if not cands:
        return None, None
    cands.sort()
    log("manifest found by glob", path=cands[-1][1])
    return cands[-1][2], cands[-1][1]


def _save_traj(S, T):
    try:
        if len(S):
            OUT.mkdir(parents=True, exist_ok=True)
            arr = np.concatenate([T[:, None], S], axis=1)
            np.savetxt(OUT / "video_states.csv", arr, delimiter=",", fmt="%.5f",
                       header="t,x,y,z,rx,ry,rz,f1,f2")
    except Exception as e:
        log("save traj failed", err=str(e))


def main(ctx):
    log("controller start", version=VERSION)
    c = None
    try:
        c = Controller(ctx)
        c.run()
    except EpisodeOver:
        log("episode over")
    except Exception as e:
        log("controller exception", err=str(e), tb=traceback.format_exc()[-2000:])
        # last resort: if something is held near a known target, try to finish placing
        try:
            if c is not None and not c.rob.ended and c.target is not None and c.rob.state is not None:
                if 0.008 < c.rob.fingers < 0.075 and c.rob.grip > 0:
                    c.place()
        except Exception as e2:
            log("rescue failed", err=str(e2))
    finally:
        try:
            if c is not None:
                c.info["end"] = {"complete": c.rob.complete, "ended": c.rob.ended, "step": c.rob.step_id,
                                 "steps_sent": c.rob.n_steps_sent, "errors": c.rob.errors,
                                 "wall_s": round(time.time() - c.t0, 1)}
                save_json("analysis.json", c.info)
        except Exception:
            pass
        log("controller end")
