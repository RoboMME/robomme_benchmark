"""StopCube controller (RoboMME-Plus, RAX).  Entry point: main(ctx).

Task: "press the button to stop the cube just as it reaches the target for the N-th time".

Pipeline / transition table
---------------------------
INIT      observe step 0, parse N, perceive button/target/cube candidates, fit camera
          model from robomme_pixel_to_position samples, locate button cap in 3-D.
          -> APPROACH (always; missing pieces are retried later)
APPROACH  close gripper, move above button to z_cap + CAL_START_H  -> CALIB (reached / timeout)
CALIB     from rest, rows dz=-1 (2-3 rows) measuring the free-space descent D(k);
          -> BRAKE
BRAKE     brake and settle at z_cap + 0.06, refine button xy with wrist camera -> DESCEND
DESCEND   settle at hover height z_cap + G (G chosen from D(k)) -> WAIT
WAIT      hold pose; track cube every step, count visits, predict the arrival time of
          the N-th visit; trigger when ETA <= K + AIM + 0.5  -> PRESS
PRESS     rows dz=-1 until the episode ends / depth reached -> HOLD
HOLD      hold pressed pose until the episode ends.
Every phase keeps tracking the cube (one control step per call) so visits are never missed.
"""
import json
import math
import os
import re
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import mcp_io  # noqa: E402
import metacam  # noqa: E402
from geometry import CamModel  # noqa: E402
from perception import wrist_button_cap  # noqa: E402
from scene import Scene  # noqa: E402
from timing import VisitModel  # noqa: E402

VERSION = "stopcube-v1"

# ----------------------------------------------------------------- tunables
# (no simulator access in AfD: values are reasoned defaults, not tuned live)
CAL_START_H = 0.21        # TCP height above cap where the descent calibration starts
BRAKE_H = 0.065           # intermediate settle height above cap
MIN_HOVER_G = 0.014       # never hover the TCP closer than this above the cap top
MAX_HOVER_G = 0.06
E_HAT = 0.003             # assumed (fingertip-below-TCP - button travel) offset, m
AIM = 1.0                 # aim the press registration this many steps before closest approach
PRESS_DEPTH = 0.035       # stop commanding further down at z_cap - PRESS_DEPTH
VMAX_MOVE = 0.9           # |row| limit for transfer moves (x 0.05 m)
VIS_THR_FRAC = 0.45       # visit threshold as a fraction of target diameter
DEFAULT_D = [0.0, 0.022, 0.055, 0.09]   # fallback descent profile (m after k rows)
WALL_LIMIT = 1150.0       # controller wall budget (s)
OUT_DIR = "/workspace/output"

ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
       "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
       "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
       "nineteenth": 19, "twentieth": 20, "once": 1, "twice": 2, "thrice": 3}
CARD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20}


def parse_count(instr):
    if not instr:
        return None
    s = instr.lower()
    m = re.search(r"(\d+)\s*(st|nd|rd|th)\b", s)
    if m:
        return int(m.group(1))
    m = re.search(r"for the (\w+) time", s)
    if m and m.group(1) in ORD:
        return ORD[m.group(1)]
    for w, n in sorted(ORD.items(), key=lambda kv: -len(kv[0])):
        if re.search(r"\b%s\b" % w, s):
            return n
    m = re.search(r"(\d+)\s*times?", s)
    if m:
        return int(m.group(1))
    for w, n in CARD.items():
        if re.search(r"\b%s\s+times?\b" % w, s):
            return n
    return None


class Episode(Exception):
    pass


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.t0 = time.time()
        os.makedirs(OUT_DIR, exist_ok=True)
        self.logf = open(os.path.join(OUT_DIR, "stopcube_log.jsonl"), "a")
        self.ended = False
        self.success = None
        self.step = 0
        self.state = None
        self.front = None
        self.wrist = None
        self.cam_order = None
        self.p2p_fail = 0
        self.last_meta = None
        self.last_depth = {}
        self.cam = CamModel()
        self.cam_ok = False
        self.z_table = None
        self.button_xyz = None   # cap top centre
        self.target_xyz = None
        self.target_diam = None
        self.vm = None
        self.cube_h = 0.03
        self.N = None
        self.phase = "INIT"
        self.D = None
        self.K = 2
        self.G = 0.04
        self.call_times = []
        self.trigger_info = None
        self.last_visit_count = 0
        self.press_started = None
        self.px_to_world_cache = {}
        self.target_px_used = None
        self.missed = False
        self.remaining = None

    # ------------------------------------------------------------- logging
    def log(self, msg, **kw):
        rec = {"t": round(time.time() - self.t0, 3), "step": self.step, "phase": self.phase, "msg": msg}
        rec.update(kw)
        try:
            self.logf.write(json.dumps(rec, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o)) + "\n")
            self.logf.flush()
        except Exception:
            pass
        extra = " ".join("%s=%s" % (k, _fmt(v)) for k, v in kw.items())
        print("[%s %6.1fs s%03d %s] %s %s" % (VERSION, time.time() - self.t0, self.step, self.phase, msg, extra), flush=True)

    # ------------------------------------------------------------- tool calls
    def call(self, tool, **args):
        t = time.time()
        res = self.ctx.call(tool, **args)
        self.call_times.append(time.time() - t)
        return mcp_io.Reply(res)

    def ingest(self, rep, initial=False):
        st = rep.state
        if st is not None:
            self.state = np.array(st, float)
        sid = rep.step_id
        if isinstance(sid, int):
            self.step = sid
        if rep.episode_ended:
            self.ended = True
            self.success = rep.task_complete
        rem = rep.remaining_steps()
        if rem is not None:
            self.remaining = rem
        rgb, dep = mcp_io.assign_cameras(rep, self.cam_order)
        man = None
        if initial or (self.cam_order is None and rep.manifest_path):
            man = mcp_io.load_manifest(rep.manifest_path)
            order = mcp_io.match_order_with_manifest(rep, man)
            if order:
                if order != self.cam_order:
                    self.log("camera order from manifest", order=order)
                self.cam_order = order
                rgb, dep = mcp_io.assign_cameras(rep, self.cam_order)
        if "front" not in rgb and rep.manifest_path:
            # no inline image: use the manifest frame of this very step, if published
            man = man or mcp_io.load_manifest(rep.manifest_path)
            if man and man.get("frames"):
                fr = man["frames"][-1]
                if initial or fr.get("step_id") == self.step:
                    files = mcp_io.manifest_frame_files(fr)
                    for cam in ("front", "wrist"):
                        if (cam, "rgb") in files:
                            try:
                                rgb[cam] = mcp_io.load_rgb(files[(cam, "rgb")][0])
                            except Exception:
                                pass
        self.last_meta = rep.meta
        dd = dict(dep)
        if man is None and rep.manifest_path and self.p2p_fail > 0:
            man = mcp_io.load_manifest(rep.manifest_path)
        if man and man.get("frames"):
            fr = man["frames"][-1]
            if initial or fr.get("step_id") == self.step:
                files = mcp_io.manifest_frame_files(fr)
                for cam in ("front", "wrist"):
                    if cam not in dd and (cam, "depth") in files:
                        try:
                            dd[cam] = mcp_io.load_depth_m(*files[(cam, "depth")])
                        except Exception:
                            pass
        self.last_depth = dd
        self.front_fresh = "front" in rgb
        if "front" in rgb:
            self.front = rgb["front"]
        if "wrist" in rgb:
            self.wrist = rgb["wrist"]
        return rep, rgb, dep, man

    def act(self, row):
        """Execute one action row; returns the parsed reply (raises Episode at the end)."""
        if self.ended:
            raise Episode()
        row = [float(np.clip(x, -1.0, 1.0)) for x in row]
        try:
            rep = self.call("robomme_step", actions=[row])
        except RuntimeError as e:
            self.log("step error", err=str(e)[:300])
            # do not retry blindly: re-read status/observation
            try:
                st = self.call("robomme_status")
                if st.episode_ended:
                    self.ended = True
                    raise Episode()
                rep = self.call("robomme_observe")
            except Episode:
                raise
            except Exception as e2:
                self.log("observe after error failed", err=str(e2)[:300])
                raise
        self.ingest(rep)
        if self.front is not None and self.front_fresh:
            self.scene.update(self.step, self.front)
        elif not self.front_fresh:
            self.log("no fresh front image in reply", n_images=len(rep.images))
        self.update_world_track()
        if self.ended:
            self.log("episode ended", success=self.success)
            raise Episode()
        return rep

    # ------------------------------------------------------------- geometry
    def p2p(self, sels, camera="front", radius=0):
        """pixel_to_position on the current observation; returns list of point dicts.

        Falls back to the documented metadata back-projection (intrinsics, pose, depth)
        when the tool fails or returns no valid point."""
        sels = sels[:32]
        args = {"observation_step_id": int(self.step),
                "selections": [{"camera": camera, "u": float(u), "v": float(v)} for u, v in sels]}
        if radius:
            args["depth_window_radius"] = int(radius)
        pts = None
        try:
            rep = self.call("robomme_pixel_to_position", **args)
            pts = mcp_io.parse_points(rep, len(sels))
            if any(p["valid"] for p in pts):
                return pts
            self.p2p_fail += 1
            if self.p2p_fail <= 3:
                self.log("p2p returned no valid point", camera=camera, n=len(sels), text=(rep.texts[0][:300] if rep.texts else None))
        except Exception as e:
            self.p2p_fail += 1
            if self.p2p_fail <= 3:
                self.log("p2p call failed", camera=camera, err=str(e)[:300])
        fb = self.meta_points(sels, camera)
        if fb is not None:
            return fb
        return pts if pts is not None else [{"valid": False, "xyz": None, "spread": None, "depth": None} for _ in sels]

    def meta_points(self, sels, camera):
        cams = metacam.parse(self.last_meta or {})
        cam = cams.get(camera)
        dep = self.last_depth.get(camera)
        if cam is None or dep is None:
            if self.p2p_fail <= 3:
                self.log("no metadata fallback", camera=camera, have_cam=cam is not None, have_depth=dep is not None)
            return None
        h, w = dep.shape[:2]
        out = []
        for u, v in sels:
            ui, vi = min(max(int(round(u)), 0), w - 1), min(max(int(round(v)), 0), h - 1)
            win = dep[max(vi - 1, 0):vi + 2, max(ui - 1, 0):ui + 2]
            d = float(dep[vi, ui])
            if not np.isfinite(d) or d <= 0:
                d = float(np.nanmedian(win)) if np.isfinite(win).any() else float("nan")
            if not np.isfinite(d) or d <= 0:
                out.append({"valid": False, "xyz": None, "spread": None, "depth": None})
                continue
            P = metacam.backproject(cam, u, v, d)
            out.append({"valid": True, "xyz": [float(x) for x in P], "spread": float(np.nanmax(win) - np.nanmin(win)),
                        "depth": d})
        if self.p2p_fail <= 3:
            self.log("metadata back-projection used", camera=camera, n=len(out))
        return out

    def calibrate_front(self, depth_front, meta_C=None):
        h, w = self.front.shape[:2]
        sc = self.scene
        grid = [(u, v) for v in np.linspace(h * 0.15, h * 0.95, 5) for u in np.linspace(w * 0.06, w * 0.94, 5)]
        extra = []
        if sc.button is not None:
            extra.append(sc.button["core"])
        for c in sc.cands0:
            extra.append(c["c"])
        sels = [(int(round(u)), int(round(v))) for u, v in grid + extra][:32]
        # elevated / off-plane points: robot (connected to the image top) and the far background
        from perception import robot_mask_from, to_hsv
        nw = sc.table.nonwood(self.front)
        rob = robot_mask_from(nw)
        rv, ru = np.nonzero(rob & (np.arange(h)[:, None] > 2))
        sels2 = []
        if len(rv):
            idx = np.linspace(0, len(rv) - 1, min(18, len(rv))).astype(int)
            sels2 += [(int(ru[i]), int(rv[i])) for i in idx]
        sels2 += [(int(u), 1) for u in np.linspace(8, w - 8, 6)]
        sels2 += [(int(u), int(v)) for v in (h * 0.3, h * 0.7) for u in np.linspace(w * 0.15, w * 0.85, 4)]
        sels2 = sels2[:32]
        pts = self.p2p(sels) + (self.p2p(sels2) if sels2 else [])
        allsel = sels + sels2
        uv, xyz, dd = [], [], []
        for (u, v), p in zip(allsel, pts):
            if p["valid"] and p["xyz"] is not None and all(math.isfinite(x) for x in p["xyz"]):
                uv.append((u, v))
                xyz.append(p["xyz"])
                dval = None
                if depth_front is not None:
                    dv = depth_front[min(int(v), h - 1), min(int(u), w - 1)]
                    if np.isfinite(dv) and dv > 0:
                        dval = float(dv)
                if dval is None and isinstance(p.get("depth"), (int, float)):
                    dval = float(p["depth"])
                dd.append(dval)
        self.log("front calibration samples", n=len(uv), of=len(allsel))
        if len(uv) < 8:
            return False
        uv, xyz = np.array(uv, float), np.array(xyz, float)
        zs = xyz[:, 2]
        lower = uv[:, 1] > h * 0.45
        zt = np.median(zs[lower]) if lower.sum() >= 4 else np.median(zs)
        table = np.abs(zs - zt) < 0.008
        self.z_table = float(np.median(zs[table])) if table.sum() >= 4 else float(zt)
        self.cam.z_table = self.z_table
        elevated = zs > self.z_table + 0.03
        self.log("calibration point classes", table=int(table.sum()), elevated=int(elevated.sum()),
                 below=int((zs < self.z_table - 0.03).sum()))
        cands = {}
        if all(d is not None for d in dd):
            r = self.cam.fit_depth(uv, np.array(dd), xyz)
            if depth_front is not None:
                dT = [depth_front[min(int(u), h - 1), min(int(v), w - 1)] for u, v in uv]
                if all(np.isfinite(x) and x > 0 for x in dT):
                    rT = CamModel().fit_depth(uv, np.array(dT, float), xyz)
                    self.log("depth fit", resid=r, resid_transposed=rT)
                    if rT < 0.5 * r:
                        self.log("WARNING transposed depth indexing fits better")
            self.log("camera model depth", resid=r, C=self.cam.C)
            if r < 0.01:
                cands["depth"] = self.cam.C
        try:
            cam2 = CamModel()
            cam2.z_table = self.z_table
            r2 = cam2.fit_dlt(uv, xyz)
            self.log("camera model dlt", resid_px=r2, C=cam2.C)
            if r2 is not None and r2 < 3.0:
                cands["dlt"] = cam2.C
                if self.cam.kind != "depth" or self.cam.resid is None or self.cam.resid >= 0.01:
                    self.cam = cam2
        except Exception:
            self.log("dlt failed", err=traceback.format_exc()[-300:])
        if meta_C is not None:
            cands["meta"] = np.asarray(meta_C, float)
        if table.sum() >= 6:
            rh = self.cam.fit_homography(uv[table], xyz[table, :2])
            self.log("table homography", resid=rh, n=int(table.sum()))
            # choose the camera centre that best explains the off-table points
            off = np.abs(zs - self.z_table) > 0.015
            best = None
            for name, C in cands.items():
                if C is None or not np.all(np.isfinite(C)) or C[2] < self.z_table + 0.15 or np.linalg.norm(C) > 6:
                    self.log("camera centre rejected", src=name, C=C)
                    continue
                score = self.cam.ray_dist(C, uv[off], xyz[off]) if off.sum() >= 2 else 0.0
                self.log("camera centre candidate", src=name, C=C, ray_dist=score)
                if best is None or score < best[0]:
                    best = (score, name, C)
            if best is not None and best[0] < 0.02:
                self.cam.Cplane = np.asarray(best[2], float)
                self.log("camera centre chosen", src=best[1], C=best[2])
            else:
                self.log("no reliable camera centre; no elevation correction")
        self.cam_ok = self.cam.H is not None or self.cam.kind in ("depth", "dlt")
        self.log("table plane", z_table=self.z_table, cam_kind=self.cam.kind)
        return self.cam_ok

    def locate_button(self):
        b = self.scene.button
        if b is None:
            return False
        u, v = b["core"]
        sels = [(u, v), (u - 1.5, v), (u + 1.5, v), (u, v - 1.5), (u, v + 1.5)]
        pts = [p for p in self.p2p([(int(round(a)), int(round(c))) for a, c in sels]) if p["valid"] and p["xyz"]]
        if len(pts) < 2:
            return False
        P = np.median(np.array([p["xyz"] for p in pts]), 0)
        hgt = P[2] - (self.z_table if self.z_table is not None else P[2] - 0.02)
        self.log("button cap (front)", xyz=P, height_above_table=hgt)
        if not (-0.005 < hgt < 0.2):
            self.log("button height implausible")
        self.button_xyz = P
        return True

    def refine_button_wrist(self):
        if self.wrist is None:
            return
        det = wrist_button_cap(self.wrist)
        if det is None:
            self.log("wrist: no button cap found")
            return
        u, v = det["c"]
        sels = [(u, v), (u - 2, v), (u + 2, v), (u, v - 2), (u, v + 2)]
        try:
            pts = [p for p in self.p2p([(int(round(a)), int(round(c))) for a, c in sels], camera="wrist") if p["valid"] and p["xyz"]]
        except Exception as e:
            self.log("wrist p2p failed", err=str(e)[:200])
            return
        if len(pts) < 2:
            self.log("wrist: invalid depth")
            return
        P = np.median(np.array([p["xyz"] for p in pts]), 0)
        dxy = float(np.hypot(*(P[:2] - self.button_xyz[:2])))
        dz = float(P[2] - self.button_xyz[2])
        self.log("button cap (wrist)", xyz=P, dxy=dxy, dz=dz, det=det["c"], cap_area=det["cap_area"])
        if dxy < 0.025 and abs(dz) < 0.02:
            self.button_xyz = np.array([0.3 * self.button_xyz[0] + 0.7 * P[0],
                                        0.3 * self.button_xyz[1] + 0.7 * P[1],
                                        0.3 * self.button_xyz[2] + 0.7 * P[2]])
            self.log("button refined", xyz=self.button_xyz)

    def setup_target(self):
        """Once the scene has a stable target, get its 3-D centre and build the visit model."""
        t = self.scene.target
        if t is None or t.get("stable", 0) < 2:
            return False
        if self.vm is not None and self.target_px_used is not None and \
                np.hypot(t["c"][0] - self.target_px_used[0], t["c"][1] - self.target_px_used[1]) < 2.5:
            return True
        # avoid sampling while the cube covers the target
        cube = self.scene.cube_px(self.step)
        if cube is not None and np.hypot(cube[0] - t["c"][0], cube[1] - t["c"][1]) < 0.8 * t["w"] + 6:
            return False
        u, v = t["c"]
        r = 0.2 * t["w"]
        sels = [(u, v), (u - r, v), (u + r, v), (u, v - r), (u, v + r)]
        pts = [p for p in self.p2p([(int(round(a)), int(round(c))) for a, c in sels]) if p["valid"] and p["xyz"]]
        if len(pts) >= 2:
            P = np.median(np.array([p["xyz"] for p in pts]), 0)
        else:
            P = self.cam.plane_point(u, v, (self.z_table or 0.0) + 0.004)
            if P is None:
                return False
        zt = P[2]
        a = self.cam.plane_point(u - 0.5 * t["w"], v, zt)
        b = self.cam.plane_point(u + 0.5 * t["w"], v, zt)
        diam = float(np.linalg.norm(a[:2] - b[:2])) if a is not None and b is not None else 0.05
        self.target_xyz = P
        self.target_diam = float(np.clip(diam, 0.02, 0.2))
        thr = float(np.clip(VIS_THR_FRAC * self.target_diam, 0.01, 0.06))
        # cube height from its pixel size
        ca = self.scene.tracker.cube_area or 150.0
        mpp = self.target_diam / max(t["w"], 1)
        self.cube_h = float(np.clip(0.8 * math.sqrt(ca) * mpp, 0.015, 0.07))
        if self.vm is None:
            self.vm = VisitModel(P[:2], thr)
        else:
            self.vm.set_target(P[:2], thr)
        self.target_px_used = t["c"]
        self.log("target 3-D", xyz=P, diam=self.target_diam, thr=thr, cube_h=self.cube_h, px=t["c"])
        self.update_world_track(full=True)
        return True

    def cube_world(self, det):
        if det is None:
            return None
        z = (self.z_table if self.z_table is not None else 0.0) + 0.5 * self.cube_h
        P = self.cam.plane_point(det[0], det[1], z)
        return None if P is None else P[:2]

    def update_world_track(self, full=False):
        if self.vm is None:
            return
        tr = self.scene.tracker.track
        keys = sorted(tr) if full else [k for k in sorted(tr) if k >= self.step - 13]
        for k in keys:
            self.vm.add(k, self.cube_world(tr[k]))

    # ------------------------------------------------------------- motion
    def pos(self):
        return self.state[:3].copy()

    def move_row(self, goal, vmax=VMAX_MOVE, grip=1.0, zrow=None):
        p = self.pos()
        d = (np.asarray(goal, float) - p) / 0.05
        row = np.clip(d, -vmax, vmax)
        if zrow is not None:
            row[2] = zrow
        return [row[0], row[1], row[2], 0.0, 0.0, 0.0, grip]

    def hold_row(self, goal):
        return self.move_row(goal, vmax=0.4)

    # ------------------------------------------------------------- visits
    def visits(self):
        if self.vm is None:
            return []
        return self.vm.recount(self.step)

    def per_step_bookkeeping(self):
        if self.vm is None:
            self.setup_target()
            return
        if self.scene.target is not None and self.scene.target.get("stable", 0) >= 2:
            self.setup_target()
        vis = self.visits()
        if len(vis) != self.last_visit_count:
            self.log("visit count", count=len(vis), visits=[(v["step"], round(v["t"], 2), round(v["dmin"], 4)) for v in vis])
            self.last_visit_count = len(vis)

    def eta(self):
        """Estimate time (steps from now) of the next closest approach.  None if unknown."""
        vm = self.vm
        if vm is None or self.step not in vm.track:
            return None
        prof = vm.eta_profile(self.step)
        kin = vm.eta_kinematic(self.step)
        use = None
        if prof is not None and prof["match"] < 0.3 * vm.thr:
            use = prof
        elif kin is not None and kin["dmin"] < 1.2 * vm.thr and not kin.get("beyond"):
            use = kin
        return use, prof, kin

    # ------------------------------------------------------------- phases
    def run(self):
        self.log("start", pid=os.getpid())
        try:
            st = self.call("robomme_status")
            self.log("status", meta=_short(st.meta))
            instr = st.instruction
        except Exception as e:
            self.log("status failed", err=str(e)[:200])
            instr = None
        rep = self.call("robomme_observe")
        rep, rgb, dep, man = self.ingest(rep, initial=True)
        instr = rep.instruction or instr
        self.N = parse_count(instr)
        self.log("instruction", instr=instr, N=self.N, state=self.state, step=self.step,
                 n_images=len(rep.images), kinds=[i["kind"] for i in rep.images], cam_order=self.cam_order,
                 meta_keys=list(rep.meta.keys())[:40])
        if self.N is None:
            self.N = 5
            self.log("WARNING could not parse N; default 5")
        if man is not None:
            phases = {}
            for f in man.get("frames", []):
                phases[f.get("phase")] = phases.get(f.get("phase"), 0) + 1
            self.log("initial manifest", phases=phases)
        if self.front is None:
            raise RuntimeError("no front image")
        self.scene = Scene(self.front, log=lambda m: self.log(m))
        self.scene.update(self.step, self.front)
        depth_front = dep.get("front")
        if depth_front is None and man and man.get("frames"):
            files = mcp_io.manifest_frame_files(man["frames"][-1])
            if ("front", "depth") in files:
                try:
                    depth_front = mcp_io.load_depth_m(*files[("front", "depth")])
                except Exception as e:
                    self.log("depth load failed", err=str(e)[:200])
        self.log("front depth", available=depth_front is not None,
                 median=float(np.nanmedian(depth_front)) if depth_front is not None else None)
        meta_C = None
        try:
            poses = rep.get("camera_poses", "camera_pose", pred=lambda x: isinstance(x, (dict, list)))
            fp = poses.get("front") if isinstance(poses, dict) else None
            if isinstance(fp, dict):
                pos = mcp_io.find_key(fp, ["position", "p", "translation"], lambda x: mcp_io.is_num_list(x, 3))
                if pos is not None:
                    meta_C = [float(x) for x in pos]
            self.log("meta camera pose", front=_short(fp) if fp is not None else None)
        except Exception:
            pass
        try:
            self.calibrate_front(depth_front, meta_C)
        except Exception as e:
            self.log("calibration failed", err=traceback.format_exc()[-800:])
        try:
            self.locate_button()
        except Exception:
            self.log("locate button failed", err=traceback.format_exc()[-800:])
        try:
            self.main_loop()
        except Episode:
            pass
        self.log("finished", success=self.success, ended=self.ended,
                 mean_call=float(np.mean(self.call_times)) if self.call_times else None)

    def ensure_button(self):
        if self.button_xyz is None:
            if self.scene.button is None:
                # re-run static detection on the current background
                self.scene._init_button(self.scene.cands0, self.scene.rgb0)
            try:
                self.locate_button()
            except Exception:
                pass
        return self.button_xyz is not None

    def main_loop(self):
        # gripper close + wait for button localisation if needed
        tries = 0
        while not self.ensure_button():
            tries += 1
            self.act([0, 0, 0, 0, 0, 0, 1.0])
            self.per_step_bookkeeping()
            if tries > 30:
                raise RuntimeError("button not found")
        B = self.button_xyz
        # ---------------- APPROACH
        self.phase = "APPROACH"
        goal = np.array([B[0], B[1], B[2] + CAL_START_H])
        prev = self.pos()
        for i in range(60):
            p = self.pos()
            err = np.linalg.norm(goal - p)
            spd = np.linalg.norm(p - prev)
            if err < 0.01 and spd < 0.004 and i > 2:
                break
            prev = p
            self.act(self.move_row(goal))
            self.per_step_bookkeeping()
            if self.check_trigger(ready=False):
                return
        self.log("approach done", pos=self.pos(), goal=goal)
        # ---------------- CALIB (free-space descent profile)
        self.phase = "CALIB"
        z0 = self.pos()[2]
        D = [0.0]
        for k in range(3):
            self.act(self.move_row(goal, zrow=-1.0))
            D.append(z0 - self.pos()[2])
            self.per_step_bookkeeping()
            inc = D[-1] - D[-2]
            if self.pos()[2] - 1.6 * inc < B[2] + 0.085:
                break
        self.D = D
        self.log("descent profile", D=[round(x, 4) for x in D])
        # ---------------- BRAKE
        self.phase = "BRAKE"
        goal = np.array([B[0], B[1], B[2] + BRAKE_H])
        zs = []
        for i in range(25):
            self.act(self.move_row(goal, vmax=0.5))
            zs.append(self.pos()[2])
            self.per_step_bookkeeping()
            if self.check_trigger(ready=False):
                return
            if len(zs) >= 3 and abs(zs[-1] - zs[-2]) < 0.0015 and abs(zs[-1] - goal[2]) < 0.006:
                break
        self.log("brake done", pos=self.pos(), min_z_above_cap=min(zs) - B[2] if zs else None)
        try:
            self.refine_button_wrist()
        except Exception:
            self.log("wrist refine failed", err=traceback.format_exc()[-500:])
        B = self.button_xyz
        self.choose_hover()
        # ---------------- DESCEND
        self.phase = "DESCEND"
        goal = np.array([B[0], B[1], B[2] + self.G])
        zs = []
        for i in range(30):
            self.act(self.move_row(goal, vmax=0.25))
            zs.append(self.pos()[2])
            self.per_step_bookkeeping()
            if self.check_trigger(ready=i > 3):
                return
            if len(zs) >= 3 and abs(zs[-1] - zs[-2]) < 0.001 and abs(zs[-1] - goal[2]) < 0.003:
                break
        self.hover = goal
        self.log("hover reached", pos=self.pos(), goal=goal, gap=self.pos()[2] - B[2])
        # ---------------- WAIT
        self.phase = "WAIT"
        while True:
            if time.time() - self.t0 > WALL_LIMIT:
                self.log("wall limit reached; pressing now")
                self.press()
                return
            if self.check_trigger(ready=True):
                return
            self.act(self.hold_row(self.hover))
            self.per_step_bookkeeping()

    def choose_hover(self):
        D = self.D if self.D and len(self.D) >= 2 and self.D[1] > 0.003 else None
        if D is None:
            self.log("descent calibration unusable; default profile")
            D = list(DEFAULT_D)
        else:
            # extrapolate a 3rd value if calibration stopped early (constant increment)
            while len(D) < 4:
                D.append(D[-1] + (D[-1] - D[-2]))
        best = None
        for K in (1, 2, 3):
            mid = 0.5 * (D[K - 1] + D[K])
            margin = 0.5 * (D[K] - D[K - 1])
            G = mid + E_HAT
            if G < MIN_HOVER_G or G > MAX_HOVER_G:
                continue
            score = margin - 0.004 * (K - 1)   # prefer short latency when margins are similar
            if best is None or score > best[0]:
                best = (score, K, G, margin)
        if best is None:
            K, G = 1, MIN_HOVER_G
            margin = 0.5 * D[1]
            if D[1] < MIN_HOVER_G:
                # slow response: hover at minimum gap, latency = first k with D(k) >= G
                K = next((k for k in range(1, len(D)) if D[k] >= MIN_HOVER_G - E_HAT + 0.004), 3)
        else:
            _, K, G, margin = best
        self.K, self.G = int(K), float(G)
        self.log("hover plan", K=self.K, G=self.G, margin=margin, D=[round(x, 4) for x in D])

    def latency_now(self):
        """Rows needed from the current height until the press registers."""
        if self.phase == "WAIT" or self.D is None:
            return self.K
        gap = self.pos()[2] - self.button_xyz[2] - E_HAT
        D = list(self.D) if self.D and len(self.D) >= 2 and self.D[1] > 0.003 else list(DEFAULT_D)
        while len(D) < 12:
            D.append(D[-1] + (D[-1] - D[-2]))
        for k in range(1, len(D)):
            if D[k] >= gap:
                return k
        return len(D)

    def check_trigger(self, ready):
        """Decide whether to start the press now.  Returns True after pressing."""
        if self.vm is None or self.N is None:
            return False
        vis = self.vm.visits
        n = len(vis)
        if n > self.N - 1 and not self.missed:
            self.missed = True
            self.log("WARNING passed the N-th visit without pressing", count=n, N=self.N)
        want = self.N - 1 if not self.missed else n
        if n != want:
            return False
        # cooldown after the last visit (avoid matching departure as approach)
        if vis:
            last = vis[-1]
            intervals = np.diff([v["t"] for v in vis]) if len(vis) >= 2 else []
            cool = 0.35 * float(np.median(intervals)) if len(intervals) else 4.0
            if self.step - last["t"] < max(cool, 2.5):
                return False
        e = self.eta()
        if e is None:
            return False
        use, prof, kin = e
        if use is None:
            return False
        d_now = self.vm.dist(self.step)
        d_prev = self.vm.dist(self.step - 1)
        approaching = d_prev is None or d_now is None or d_now < d_prev + 0.002
        K = self.latency_now()
        eta = use["eta"]
        if eta <= K + AIM + 0.5 and approaching:
            if not ready:
                self.log("trigger condition met but arm not ready", eta=eta)
                if self.phase in ("APPROACH", "CALIB", "BRAKE"):
                    return False
            self.trigger_info = {"step": self.step, "eta": eta, "method": use["method"], "K": K,
                                 "prof": prof, "kin": kin, "d": d_now, "count": n}
            self.log("TRIGGER", eta=eta, method=use["method"], K=K, d=d_now, count=n,
                     prof=prof and round(prof["eta"], 2), kin=kin and round(kin["eta"], 2),
                     z_gap=self.pos()[2] - self.button_xyz[2])
            self.press()
            return True
        return False

    def press(self):
        self.phase = "PRESS"
        B = self.button_xyz
        goal = np.array([B[0], B[1], B[2] - PRESS_DEPTH])
        self.press_started = self.step
        for i in range(12):
            row = self.move_row(goal, vmax=1.0, zrow=-1.0)
            self.act(row)
            self.log("press", z_gap=self.pos()[2] - B[2], d=self.vm.dist(self.step) if self.vm else None)
        self.phase = "HOLD"
        for i in range(400):
            self.act(self.move_row(goal, vmax=0.6))
            if time.time() - self.t0 > WALL_LIMIT + 30:
                self.log("wall limit in hold; returning")
                return


def _fmt(v):
    if isinstance(v, float):
        return "%.4g" % v
    if isinstance(v, np.ndarray):
        return "[" + ",".join("%.4g" % x for x in v.ravel()[:8]) + "]"
    s = str(v)
    return s if len(s) < 300 else s[:300] + "..."


def _short(meta):
    try:
        s = json.dumps(meta, default=str)
        return s[:1500]
    except Exception:
        return str(meta)[:1500]


def main(ctx):
    c = Controller(ctx)
    try:
        c.run()
    except Episode:
        c.log("episode over")
    except Exception:
        c.log("FATAL", err=traceback.format_exc()[-2000:])
        # last resort: if we know where the button is, keep the episode alive by holding
        try:
            safe_fallback(c)
        except Episode:
            pass
        except Exception:
            c.log("fallback failed", err=traceback.format_exc()[-1000:])
    finally:
        try:
            c.logf.close()
        except Exception:
            pass


def safe_fallback(c):
    """Minimal loop after an unexpected exception: hold still, track, press on trigger."""
    if c.ended or c.state is None:
        return
    c.phase = "FALLBACK"
    for i in range(600):
        if c.ended or time.time() - c.t0 > WALL_LIMIT:
            return
        try:
            if c.button_xyz is not None and c.vm is not None and c.check_trigger(ready=True):
                return
            c.act([0, 0, 0, 0, 0, 0, 1.0])
            c.per_step_bookkeeping()
        except Episode:
            return
        except Exception:
            c.log("fallback step error", err=traceback.format_exc()[-500:])
            if c.ended:
                return
