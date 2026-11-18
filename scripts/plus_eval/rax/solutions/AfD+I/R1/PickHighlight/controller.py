"""PickHighlight controller (RAX, RoboMME-Plus).

Task: press the button, pick up every highlighted cube one by one, press the button again.

Pipeline / transition table
---------------------------
INIT        observe, calibrate front camera vs pixel tool, table z, button, cubes -> WATCH (fail: best effort)
WATCH       hold still; white rings -> highlight votes; ends when rings vanish and first shuffle settles
            (timeouts: no rings by step 200 -> PRESS1 anyway)                                      -> PRESS1
PRESS1      closed gripper presses button cap; verify depth reached (retry deeper once)            -> PICKS
PICKS       loop over highlighted, not-done tracks: wait out shuffles, hover, align yaw, descend,
            close, verify fingers, lift, verify, put back, retreat; abort on shuffle before grasp;
            <=3 attempts per cube; budget guard -> FINAL                                           -> FINAL
FINAL       press button; if not task_complete, re-press up to 2 times                             -> END
Every robot call is followed by robomme_observe + perception update (tracker + shuffle detector).
"""
import json
import math
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rio  # noqa: E402
import geom  # noqa: E402
import percep  # noqa: E402
from tracker import Tracker  # noqa: E402

VERSION = "pickhl-v5"

# ---- tunables (prior values; offline mock-tested only) ----
HOVER_DZ = 0.10          # TCP height above table for travel
LIFT_DZ = 0.15           # TCP height above table when lifted
SPEED_FREE = 0.6         # fraction of 5 cm per step in free space
SPEED_NEAR = 0.3
CHUNK_FINE = 2           # rows per call when motion may happen
CHUNK_COARSE = 4
CLOSE_STEPS = 6
OPEN_STEPS = 4
FREEZE_MODE = os.environ.get("PICKHL_FREEZE", "0")  # "0": go home; "1": stay still; "held": stay only with a cube
FINAL_RESERVE = 55       # steps kept for the final button press
WALL_LIMIT_S = 1050      # controller wall budget (runtime limit 1200 s)
SHUFFLE_QUIET_STEPS = 5  # no movers for this many steps -> shuffle ended
WATCH_MAX_STEP = 220
MAX_ATTEMPTS = 3

OUT_DIR = "/workspace/output"


class EpisodeEnded(Exception):
    pass


def log(*a):
    try:
        print(f"[{VERSION}]", *a, flush=True)
    except Exception:
        pass


class Obs:
    pass


def assemble_obs(result):
    o = Obs()
    o.result = result
    meta = rio.first_text_json(result)
    o.meta = meta
    o.state = rio.get_state(meta)
    o.step_id = rio.get_step_id(meta)
    o.ended = rio.get_bool(meta, "episode_ended")
    o.complete = rio.get_bool(meta, "task_complete")
    o.instruction = rio.get_instruction(meta)
    o.remaining, o.max_steps = rio.remaining_steps(meta)
    imgs = rio.collect_images(result, meta)
    rgb = {}
    dep = {}
    unl_rgb, unl_dep = [], []
    for im in imgs:
        if im["kind"] == "rgb":
            if im["cam"] in ("front", "wrist") and im["cam"] not in rgb:
                rgb[im["cam"]] = im["arr"]
            else:
                unl_rgb.append(im["arr"])
        elif im["kind"] == "depth":
            if im["cam"] in ("front", "wrist") and im["cam"] not in dep:
                dep[im["cam"]] = im["arr"]
            else:
                unl_dep.append(im["arr"])
    for a in unl_rgb:
        c = rio.classify_rgb_camera(a)
        if c not in rgb:
            rgb[c] = a
        else:
            other = "wrist" if c == "front" else "front"
            rgb.setdefault(other, a)
    # file descriptors (explicitly labelled) for anything missing
    scale = rio.depth_scale_from_meta(meta)
    files = rio.files_from_meta(meta)
    lastf = {}
    for f in files:
        cam = str(f.get("camera", "")).lower()
        kind = str(f.get("kind", "")).lower()
        if cam and kind:
            lastf[(cam, kind)] = f
    for cam in ("front", "wrist"):
        if cam not in dep and (cam, "depth") in lastf:
            try:
                k, arr = rio.load_file_image(lastf[(cam, "depth")]["path"])
                dep[cam] = arr.astype(float) * float(lastf[(cam, "depth")].get("meters_per_unit", scale))
                dep[cam + "_scaled"] = True
            except Exception as e:
                log("depth file load failed", cam, repr(e))
        if cam not in rgb and (cam, "rgb") in lastf:
            try:
                k, arr = rio.load_file_image(lastf[(cam, "rgb")]["path"])
                rgb[cam] = arr
            except Exception as e:
                log("rgb file load failed", cam, repr(e))
    # manifest fallback: the last execution frame of the reply's frame_sequence
    need = [c for c in ("front", "wrist") if c not in dep or c not in rgb]
    if "front" in need:
        try:
            for mp in rio.manifest_paths(meta)[:1]:
                man = rio.read_manifest(mp)
                frs = [f for f in man.get("frames", []) if f.get("phase", "execution") != "video_demo"]
                if not frs:
                    continue
                frs.sort(key=lambda f: f.get("frame_index", 0))
                last = frs[-1]
                ff = {(str(x.get("camera", "")).lower(), str(x.get("kind", "")).lower()): x for x in last.get("files", [])}
                for cam in ("front", "wrist"):
                    if cam not in dep and (cam, "depth") in ff:
                        k, arr = rio.load_file_image(ff[(cam, "depth")]["path"])
                        dep[cam] = arr.astype(float) * float(ff[(cam, "depth")].get("meters_per_unit", scale))
                        dep[cam + "_scaled"] = True
                    if cam not in rgb and (cam, "rgb") in ff:
                        k, arr = rio.load_file_image(ff[(cam, "rgb")]["path"])
                        rgb[cam] = arr
                if o.state is None and isinstance(last.get("state"), list) and len(last["state"]) == 8:
                    o.state = [float(x) for x in last["state"]]
        except Exception as e:
            log("manifest fallback failed", repr(e))
    if unl_dep:
        # assign unlabelled depth maps: wrist has very near pixels (fingers) at the bottom
        for a in unl_dep:
            a = a.astype(float)
            med = np.median(a[a > 0]) if (a > 0).any() else 0
            am = a * (scale if med > 20 else 1.0)
            bottom = am[int(am.shape[0] * 0.8):]
            near = ((bottom > 0) & (bottom < 0.15)).mean()
            c = "wrist" if near > 0.03 else "front"
            if c in dep:
                c = "wrist" if c == "front" else "front"
            if c not in dep:
                dep[c] = am
                dep[c + "_scaled"] = True
    for cam in ("front", "wrist"):
        if cam in dep and not dep.get(cam + "_scaled"):
            a = dep[cam].astype(float)
            med = np.median(a[a > 0]) if (a > 0).any() else 0
            dep[cam] = a * (scale if med > 20 else 1.0)
    o.rgb = rgb
    o.depth = {k: v for k, v in dep.items() if not k.endswith("_scaled")}
    return o


class Ctl:
    def __init__(self, ctx):
        self.ctx = ctx
        self.t0 = time.time()
        self.obs = None
        self.scene = percep.Scene()
        self.tr = Tracker()
        self.ncalls = 0
        self.call_time = 0.0
        self.step = 0
        self.phase = "init"
        self.yaw_sign = 1.0
        self.yaw_flip_checked = False
        self.shuffle_active = False
        self.step_obs_ok = None
        self.shuffle_starts = []
        self.shuffle_ends = []
        self.last_motion_step = -999
        self.motion_frames = []
        self.hl_watch = False
        self.hl_on_frames = 0
        self.hl_off_after = 0
        self.hl_seen = False
        self.hl_done = False
        self.hl_end_step = None
        self.hl_white_hist = []
        self.home = None
        self.R_ref = None
        self.ended = False
        self.complete = False
        self.presses = 0
        self.snap_count = 0
        self.events = []
        self.remaining = None
        self.free_dets = []
        self.max_steps = None
        self.pick_durations = []
        os.makedirs(OUT_DIR, exist_ok=True)
        self.logf = open(os.path.join(OUT_DIR, "decisions.jsonl"), "a")

    # ------------------------------------------------------------ logging
    def jlog(self, kind, **kw):
        rec = dict(t=round(time.time() - self.t0, 2), step=self.step, phase=self.phase, kind=kind)
        rec.update(kw)
        try:
            self.logf.write(json.dumps(rec, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
            self.logf.flush()
        except Exception:
            pass

    def elapsed(self):
        return time.time() - self.t0

    # ------------------------------------------------------------ robot I/O
    def _call(self, tool, **kw):
        t = time.time()
        try:
            return self.ctx.call(tool, **kw)
        finally:
            self.ncalls += 1
            self.call_time += time.time() - t

    def observe(self):
        res = self._call("robomme_observe")
        o = assemble_obs(res)
        self._absorb(o)
        return o

    def _absorb(self, o):
        if o.step_id is not None:
            self.step = o.step_id
        if o.remaining is not None:
            self.remaining = o.remaining
        if o.max_steps is not None:
            self.max_steps = o.max_steps
        if o.ended:
            self.ended = True
        if o.complete:
            self.complete = True
        if o.state is not None:
            self.obs = o
        elif self.obs is not None:
            o.state = self.obs.state
            self.obs = o
        else:
            self.obs = o

    def send(self, rows):
        if self.ended:
            raise EpisodeEnded()
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        if self.remaining is not None:
            rows = rows[:max(1, self.remaining)]
        res = self._call("robomme_step", actions=rows)
        meta = rio.first_text_json(res)
        ended = rio.get_bool(meta, "episode_ended")
        comp = rio.get_bool(meta, "task_complete")
        rem, mx = rio.remaining_steps(meta)
        sid = rio.get_step_id(meta)
        if sid is not None:
            self.step = sid
        else:
            self.step += len(rows)
        if rem is not None:
            self.remaining = rem
        if mx is not None:
            self.max_steps = mx
        if comp:
            self.complete = True
        if ended:
            self.ended = True
            self.jlog("ended", complete=self.complete, meta_keys=list(meta.keys())[:20])
            log("episode ended; complete =", self.complete, "step", self.step)
            raise EpisodeEnded()
        o = None
        if self.step_obs_ok is not False:
            try:
                o2 = assemble_obs(res)
                good = (o2.state is not None and "front" in o2.rgb and "front" in o2.depth
                        and o2.step_id is not None and o2.step_id == self.step)
            except Exception:
                good = False
            if good:
                self._absorb(o2)
                o = o2
            if self.step_obs_ok is None:
                self.step_obs_ok = bool(good)
                self.jlog("step_reply_obs", usable=bool(good))
                log("step reply carries a full observation:", bool(good))
        if o is None:
            o = self.observe()
        if self.ended:
            raise EpisodeEnded()
        self.perceive(o)
        return o

    # ------------------------------------------------------------ state helpers
    def pos(self):
        return np.asarray(self.obs.state[:3], float)

    def R(self):
        return geom.rotvec_to_mat(self.obs.state[3:6])

    def finger_sum(self):
        s = self.obs.state
        return float(s[6] + s[7])

    def finger_yaw(self):
        R = self.R()
        return math.atan2(R[1, 1], R[0, 1])

    def chunk(self):
        fine, coarse = CHUNK_FINE, CHUNK_COARSE
        if self.step >= 40:
            per_step = self.elapsed() / max(self.step, 1)
            if per_step > 0.75:      # slow RPCs: ~1300 steps would not fit the wall budget
                fine, coarse = 3, 7
            elif per_step > 0.55:
                fine, coarse = 2, 6
        if self.shuffle_active or self.hl_watch:
            return fine
        if self.shuffle_soon(12):
            return fine
        return coarse

    # ------------------------------------------------------------ calibration
    def calibrate(self, o):
        sc = self.scene
        dep = o.depth.get("front")
        rgb = o.rgb.get("front")
        if rgb is None:
            raise RuntimeError("no front rgb in observation")
        if dep is None:
            raise RuntimeError("no front depth in observation")
        H, W = dep.shape
        # flat, valid sample pixels spread over the image
        cands = []
        for v in np.linspace(20, H - 12, 7):
            for u in np.linspace(16, W - 16, 6):
                u, v = int(u), int(v)
                win = dep[max(v - 1, 0):v + 2, max(u - 1, 0):u + 2]
                if (win > 0).all() and np.ptp(win) < 0.03 and abs(float(win[1, 1]) - float(np.median(win))) < 0.003:
                    cands.append((u, v))
        if len(cands) < 8:
            for v in np.linspace(20, H - 12, 7):
                for u in np.linspace(16, W - 16, 6):
                    u, v = int(u), int(v)
                    if dep[v, u] > 0 and (u, v) not in cands:
                        cands.append((u, v))
        cands = cands[:30]
        sels = [{"camera": "front", "u": float(u), "v": float(v)} for u, v in cands]
        pts = [None] * len(sels)
        try:
            res = self._call("robomme_pixel_to_position", observation_step_id=int(o.step_id or 0),
                             selections=sels, depth_window_radius=0)
            meta = rio.first_text_json(res)
            pts = rio.parse_pixel_points(meta, len(sels))
            self.jlog("pix_tool_sample", sample=json.dumps(meta)[:1500])
        except Exception as e:
            log("pixel tool failed", repr(e))
            self.jlog("pix_tool_fail", err=repr(e))
        uvd, xyz = [], []
        for (u, v), p in zip(cands, pts):
            if p is not None and np.all(np.isfinite(p)):
                uvd.append((u, v, dep[v, u]))
                xyz.append(p)
        uvd = np.array(uvd, float).reshape(-1, 3)
        xyz = np.array(xyz, float).reshape(-1, 3)
        intr = rio.parse_intrinsics(o.meta, "front")
        pose = rio.parse_pose(o.meta, "front")
        variants = geom.analytic_variants(intr, pose)
        best = None
        if len(uvd) >= 6:
            for m in variants:
                try:
                    e = float(np.median(geom.model_error(m, uvd, xyz)))
                except Exception:
                    continue
                if best is None or e < best[0]:
                    best = (e, m)
            for swap in (False, True):
                q = uvd.copy()
                if swap:
                    q[:, [0, 1]] = q[:, [1, 0]]
                try:
                    am = geom.fit_affine(q, xyz)
                    am.swap = swap
                    e = float(np.median(geom.model_error(am, q, xyz)))
                except Exception:
                    continue
                if swap:
                    am = _SwapWrap(am)
                if best is None or e < best[0] - 0.002:
                    best = (e, am)
        elif variants:
            best = (float("nan"), variants[0])
        if best is None:
            raise RuntimeError("cannot build a front camera model (no tool points, no intrinsics)")
        sc.model = best[1]
        self.jlog("calib", model=getattr(best[1], "name", "?"), err=best[0], n=len(uvd),
                  intr=intr, pose=None if pose is None else {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in pose.items()})
        log("front model", getattr(best[1], "name", "?"), "median err", best[0], "n", len(uvd))
        P, valid = percep.backproject_all(sc.model, dep)
        tz, tstd = percep.estimate_table_z(P, valid)
        sc.table_z = tz
        log("table z", tz, "std", tstd)
        # provisional cube size, then refine from heights
        sc.cube_size = 0.04
        btn = percep.find_button(rgb, P, valid, tz)
        if btn is not None:
            sc.button = btn
            sc.excl = percep.dilate(btn["mask"], 3)
            log("button", btn["xy"], "cap", btn["cap_xy"], "cap_top_z", btn["cap_top_z"], "npix", btn["npix"])
        else:
            log("button NOT found")
        dets, cm = percep.detect_cubes(rgb, P, valid, sc)
        if dets:
            hs = np.array([d["top_z"] - tz for d in dets])
            s = float(np.clip(np.median(hs), 0.02, 0.07))
            sc.cube_size = s
            dets, cm = percep.detect_cubes(rgb, P, valid, sc)
        mx, mn, sat = percep.rgb_stats(rgb)
        white0 = valid & (mn >= percep.WHITE_MIN - 15) & (sat <= percep.WHITE_SAT_MAX + 10)
        sc.base_white = percep.dilate(white0, 2)
        self.tr.init(dets, self.step)
        xy0 = np.array([d["xy"] for d in dets]).reshape(-1, 2)
        self.cube_box = (xy0.min(0), xy0.max(0)) if len(xy0) else None
        log("cube size", round(sc.cube_size, 4), "cubes", len(dets))
        self.jlog("init_scene", table_z=tz, size=sc.cube_size, ncubes=len(dets),
                  cubes=[d["xy"].tolist() for d in dets],
                  button=None if btn is None else dict(xy=btn["xy"].tolist(), cap_xy=btn["cap_xy"].tolist(),
                                                       cap_top_z=btn["cap_top_z"], ext=btn["plate_ext"]))
        self.snapshot(rgb, "init", dets)

    # ------------------------------------------------------------ perception
    def perceive(self, o, step=None):
        sc = self.scene
        if sc.model is None:
            return
        rgb = o.rgb.get("front")
        dep = o.depth.get("front")
        if rgb is None or dep is None:
            self.jlog("perceive_missing", rgb=rgb is not None, depth=dep is not None)
            return
        step = self.step if step is None else step
        P, valid = percep.backproject_all(sc.model, dep)
        pred = [t.predict(step) for t in self.tr.active()]
        dets, cm = percep.detect_cubes(rgb, P, valid, sc, pred=pred)
        ign = None
        if o.state is not None:
            tcp = np.asarray(o.state[:3])
            if tcp[2] < sc.table_z + 0.14:
                ign = tcp
        matches, moving = self.tr.update(dets, step, ignore_near=ign, ignore_r=0.075)
        used = {id(d) for d in matches.values()}
        self.free_dets = [d for d in dets if id(d) not in used and d["npix"] >= 2 * percep.MIN_CUBE_PIX]
        # consistent motion: moved in this and previous update
        movers = []
        for t in self.tr.tracks:
            if t.id in moving:
                t.moved_recent += 1
            else:
                t.moved_recent = 0
            if t.moved_recent >= 2:
                movers.append(t.id)
        if len(movers) >= 2 or (len(movers) == 1 and self.shuffle_active):
            self.last_motion_step = step
            if not self.shuffle_active:
                self.shuffle_active = True
                st = step - 2 * CHUNK_FINE
                self.shuffle_starts.append(st)
                self.tr.begin_shuffle(st)
                self.jlog("shuffle_start", movers=movers, est_start=st)
                log("shuffle start ~", st, "movers", movers)
        elif self.shuffle_active and step - self.last_motion_step >= SHUFFLE_QUIET_STEPS:
            self.shuffle_active = False
            self.shuffle_ends.append(self.last_motion_step)
            self.tr.end_shuffle()
            self.jlog("shuffle_end", at=self.last_motion_step,
                      dur=self.last_motion_step - self.shuffle_starts[-1] if self.shuffle_starts else None)
            log("shuffle end", self.last_motion_step)
        if self.hl_watch:
            wn = percep.white_new_mask(rgb, P, valid, sc)
            nwhite = int(wn.sum())
            self.hl_white_hist.append((step, nwhite))
            if nwhite >= percep.HL_ON_PIX and not self.hl_done:
                rr = percep.ring_ratios(dets, cm, wn, P, valid, sc, dep.shape)
                wr = percep.world_ring_ratios(dets, cm, wn, P, valid, sc)
                det_to_track = {id(d): tid for tid, d in matches.items()}
                for d, (ratio, wc, tc), (wratio, ww, wt) in zip(dets, rr, wr):
                    tid = det_to_track.get(id(d))
                    if tid is None:
                        continue
                    t = self.tr.tracks[tid]
                    t.hl_sum += ratio
                    t.hl_n += 1
                    t.hl_max = max(t.hl_max, ratio)
                    if wt >= 15:
                        t.whl_sum = getattr(t, "whl_sum", 0.0) + wratio
                        t.whl_n = getattr(t, "whl_n", 0) + 1
                self.hl_on_frames += 1
                self.hl_off_after = 0
                if not self.hl_seen:
                    self.hl_seen = True
                    log("highlight visible at step", step, "white px", nwhite)
                    self.snapshot(rgb, "highlight", dets)
            elif self.hl_seen and not self.hl_done:
                self.hl_off_after += 1
                if self.hl_off_after >= 2 and self.hl_on_frames >= 2:
                    self.finalize_highlight(step)

    def finalize_highlight(self, step):
        self.hl_done = True
        self.hl_end_step = step
        vals = []
        for t in self.tr.tracks:
            m = t.hl_sum / t.hl_n if t.hl_n else 0.0
            vals.append(m)
        vals = np.array(vals)
        wvals = np.array([getattr(t, "whl_sum", 0.0) / t.whl_n if getattr(t, "whl_n", 0) else np.nan
                          for t in self.tr.tracks])
        thr = highlight_threshold(vals)
        sel = vals >= thr
        wthr = None
        if len(wvals) and np.isfinite(wvals).mean() >= 0.8:
            wv = np.where(np.isfinite(wvals), wvals, 0.0)
            wthr = highlight_threshold(wv, lo=0.25, min_gap=0.15)
            if wthr is not None:
                wsel = wv >= wthr
                if wsel.sum() != sel.sum() or (wsel != sel).any():
                    log("highlight: world ring disagrees with image ring", np.flatnonzero(wsel).tolist(),
                        np.flatnonzero(sel).tolist())
                sel = wsel
                for t, w in zip(self.tr.tracks, wv):
                    t.hl_world = float(w)
        self.jlog("highlight_world", wvals=[None if not np.isfinite(v) else round(float(v), 3) for v in wvals], wthr=wthr)
        for t, s in zip(self.tr.tracks, sel):
            t.highlight = bool(s)
        hl = [t.id for t in self.tr.tracks if t.highlight]
        self.jlog("highlight_final", thr=thr, ratios=[round(float(v), 3) for v in vals], hl=hl,
                  on_frames=self.hl_on_frames)
        log("highlight final: ids", hl, "ratios", np.round(vals, 2).tolist())

    def shuffle_period(self):
        s = self.shuffle_starts
        if len(s) < 2:
            return None
        d = np.diff(s)
        d = d[d >= 60]   # demo periods 80-140; shorter gaps are false starts (arm occlusion)
        if len(d) == 0:
            return None
        return float(np.median(d))

    def shuffle_duration(self):
        if not self.shuffle_ends or not self.shuffle_starts:
            return 40.0
        durs = [e - s for s, e in zip(self.shuffle_starts, self.shuffle_ends) if e > s]
        return float(np.clip(np.median(durs), 20, 60)) if durs else 40.0

    def next_shuffle_in(self):
        P = self.shuffle_period()
        if P is None or not self.shuffle_starts:
            return None
        nxt = self.shuffle_starts[-1] + P
        while nxt < self.step - 10:
            nxt += P
        return nxt - self.step

    def shuffle_soon(self, horizon):
        n = self.next_shuffle_in()
        return n is not None and n <= horizon

    # ------------------------------------------------------------ snapshots
    def snapshot(self, rgb, tag, dets=None):
        if self.snap_count >= 40:
            return
        try:
            from PIL import Image, ImageDraw
            im = Image.fromarray(np.asarray(rgb, np.uint8)).resize((512, 512))
            if dets is not None:
                d = ImageDraw.Draw(im)
                W = rgb.shape[1]
                for det in dets:
                    pix = det["pix"]
                    ys, xs = pix // W, pix % W
                    d.rectangle([2 * xs.min(), 2 * ys.min(), 2 * xs.max(), 2 * ys.max()], outline=(255, 0, 0))
                for t in self.tr.tracks:
                    if t.det is None:
                        continue
                    pix = t.det["pix"]
                    ys, xs = pix // W, pix % W
                    col = (255, 255, 0) if t.highlight else (0, 0, 255)
                    if t.done:
                        col = (0, 255, 0)
                    d.text((2 * xs.mean(), 2 * ys.mean()), str(t.id), fill=col)
            im.save(os.path.join(OUT_DIR, f"snap_{self.snap_count:02d}_{tag}_s{self.step}.png"))
            self.snap_count += 1
        except Exception as e:
            log("snapshot failed", repr(e))

    # ------------------------------------------------------------ motion
    def hold(self, n, grip):
        n = max(1, int(n))
        done = 0
        while done < n:
            k = min(self.chunk(), n - done)
            self.send([[0, 0, 0, 0, 0, 0, grip]] * k)
            done += k

    def yaw_err(self, yaw_target):
        if yaw_target is None:
            return 0.0
        return geom.wrap_pi(yaw_target - self.finger_yaw())

    def move_to(self, target, grip, tol=0.008, speed=SPEED_FREE, max_calls=40, yaw=None, yaw_tol=0.06,
                abort=None, chunk=None):
        """Closed-loop Cartesian move. Returns 'ok', 'timeout', 'stall' or 'abort'."""
        target = np.asarray(target, float)
        stall = 0
        last = self.pos()
        for i in range(max_calls):
            p = self.pos()
            err = target - p
            ye = self.yaw_err(yaw)
            if np.linalg.norm(err) < tol and abs(ye) < yaw_tol:
                return "ok"
            if abort is not None and abort():
                return "abort"
            n = chunk or self.chunk()
            per = err / n
            lim = speed * 0.05
            nn = np.linalg.norm(per)
            if nn > lim:
                per = per * lim / nn
            rot = 0.0
            if yaw is not None and abs(ye) >= yaw_tol * 0.5:
                rot = float(np.clip(ye / n, -0.35, 0.35)) * self.yaw_sign
            row = [per[0] / 0.05, per[1] / 0.05, per[2] / 0.05, 0.0, 0.0, rot / 0.5, grip]
            ye_before = ye
            self.send([row] * n)
            # yaw sign check (once): the error must shrink when we rotate
            if yaw is not None and abs(rot) > 0.05 and not self.yaw_flip_checked:
                ye_after = self.yaw_err(yaw)
                if abs(ye_after) > abs(ye_before) + 0.03:
                    self.yaw_sign *= -1
                    self.jlog("yaw_sign_flip", before=ye_before, after=ye_after)
                    log("yaw sign flipped")
                self.yaw_flip_checked = True
            q = self.pos()
            if np.linalg.norm(q - last) < 0.0015 and np.linalg.norm(err) > 3 * tol:
                stall += 1
                if stall >= 3:
                    return "stall"
            else:
                stall = 0
            last = q
        return "timeout"

    # ------------------------------------------------------------ phases
    def watch(self):
        self.phase = "watch"
        self.hl_watch = True
        grip = -1.0
        start = self.step
        while True:
            if self.elapsed() > WALL_LIMIT_S * 0.5:
                log("watch: wall budget")
                break
            if self.hl_done:
                # wait for the first shuffle to start and settle (cap 75 steps after highlight end)
                since = self.step - self.hl_end_step
                if self.shuffle_ends and self.shuffle_ends[-1] >= self.hl_end_step - 5 and not self.shuffle_active:
                    break
                if since > 75 and not self.shuffle_active:
                    break
                if since > 140:
                    break
            elif self.step > WATCH_MAX_STEP:
                log("watch: no highlight seen by", self.step)
                if self.hl_seen:
                    self.finalize_highlight(self.step)
                break
            self.send([[0, 0, 0, 0, 0, 0, grip]] * CHUNK_FINE)
        self.hl_watch = False
        self.jlog("watch_done", start=start, white_hist=self.hl_white_hist[-80:],
                  starts=self.shuffle_starts, ends=self.shuffle_ends)

    def retreat_for_shuffle(self, grip):
        """A shuffle is running: keep the arm out of the cubes but avoid sweeping it across the view."""
        if FREEZE_MODE == "0" or (FREEZE_MODE == "held" and grip <= 0):
            return self.go_home(grip)
        self._grip_cmd = grip
        p = self.pos()
        zt = self.scene.table_z + (LIFT_DZ if grip > 0 else HOVER_DZ)
        if p[2] < zt - 0.01:
            return self.move_to([p[0], p[1], zt], grip, tol=0.012, speed=SPEED_NEAR, max_calls=10)
        return "ok"

    def wait_quiet(self, max_steps=120):
        """Wait (arm where it is) until no shuffle is active."""
        t0 = self.step
        while self.shuffle_active and self.step - t0 < max_steps:
            self.hold(CHUNK_FINE, self.grip_now())
        return not self.shuffle_active

    def grip_now(self):
        return 1.0 if self.finger_sum() < 0.06 and getattr(self, "_grip_cmd", -1.0) > 0 else getattr(self, "_grip_cmd", -1.0)

    def go_home(self, grip):
        self._grip_cmd = grip
        h = self.home.copy()
        p = self.pos()
        if p[2] < self.scene.table_z + HOVER_DZ - 0.01:
            self.move_to([p[0], p[1], self.scene.table_z + HOVER_DZ], grip, speed=SPEED_NEAR, max_calls=12)
        return self.move_to(h, grip, tol=0.015, speed=SPEED_FREE, max_calls=30)

    def press_button(self, tag):
        sc = self.scene
        b = sc.button
        if b is None:
            log("press: no button known")
            return False
        self.phase = "press_" + tag
        cx, cy = b["cap_xy"]
        top = b["cap_top_z"]
        self._grip_cmd = 1.0
        p = self.pos()
        if p[2] < top + 0.05:
            self.move_to([p[0], p[1], top + 0.07], 1.0, speed=SPEED_NEAR, max_calls=10)
        r = self.move_to([cx, cy, top + 0.06], 1.0, tol=0.006, speed=SPEED_FREE, max_calls=30)
        self.move_to([cx, cy, top + 0.015], 1.0, tol=0.005, speed=SPEED_NEAR, max_calls=12)
        ok = False
        zmin = 9.0
        for depth in (0.015, 0.025):
            self.move_to([cx, cy, top - depth], 1.0, tol=0.003, speed=0.25, max_calls=5, chunk=2)
            zmin = min(zmin, float(self.pos()[2]))
            self.hold(2, 1.0)
            zmin = min(zmin, float(self.pos()[2]))
            if zmin <= top - 0.006:
                ok = True
                break
            self.move_to([cx, cy, top + 0.01], 1.0, tol=0.006, speed=SPEED_NEAR, max_calls=4)
        self.presses += 1
        self.jlog("press", tag=tag, cap_top=top, zmin=zmin, ok=ok, approach=r, complete=self.complete)
        log("press", tag, "cap_top", round(top, 4), "zmin", round(zmin, 4), "ok", ok)
        self.move_to([cx, cy, top + 0.08], 1.0, tol=0.01, speed=SPEED_NEAR, max_calls=10)
        return ok

    def choose_target(self):
        cands = [t for t in self.tr.tracks if t.highlight and not t.done and t.attempts < MAX_ATTEMPTS]
        if not cands:
            return None
        p = self.pos()[:2]
        cands.sort(key=lambda t: (t.missing > 0, t.attempts, float(np.linalg.norm(t.xy - p))))
        return cands[0]

    def grasp_yaw(self, t):
        """Finger-axis yaw for this cube: square symmetry + neighbour clearance, near the initial yaw."""
        cy = t.det.get("yaw") if t.det is not None else None
        base = self.yaw0
        if cy is None:
            return base
        s = self.scene.cube_size
        best = None
        for k in range(4):
            a = cy + k * math.pi / 2
            a = base + geom.wrap_pi(a - base)
            if abs(geom.wrap_pi(a - base)) > math.pi / 2 + 1e-6:
                continue
            # clearance of fingertip positions from other cubes
            u = np.array([math.cos(a), math.sin(a)])
            clear = 1.0
            for o in self.tr.tracks:
                if o.id == t.id or o.held:
                    continue
                for sgn in (-1, 1):
                    f = t.xy + sgn * u * (0.045 + 0.008)
                    clear = min(clear, float(np.linalg.norm(o.xy - f)) - 0.71 * s)
            cost = -min(clear, 0.03) + 0.01 * abs(geom.wrap_pi(a - base))
            if best is None or cost < best[0]:
                best = (cost, a, clear)
        return best[1] if best else base

    def pick(self, t):
        sc = self.scene
        s = sc.cube_size
        tz = sc.table_z
        t.attempts += 1
        self.phase = f"pick{t.id}"
        t_start = self.step
        log("pick target", t.id, "xy", np.round(t.xy, 4).tolist(), "attempt", t.attempts, "step", self.step)
        self.jlog("pick_start", tid=t.id, xy=t.xy.tolist(), attempt=t.attempts)
        self._grip_cmd = -1.0
        yaw = self.grasp_yaw(t)

        def abort_pre():
            return self.shuffle_active

        p = self.pos()
        if p[2] < tz + HOVER_DZ - 0.02:
            self.move_to([p[0], p[1], tz + HOVER_DZ], -1.0, speed=SPEED_NEAR, max_calls=10)
        r = self.move_to([t.xy[0], t.xy[1], tz + HOVER_DZ], -1.0, tol=0.01, speed=SPEED_FREE, max_calls=30,
                         yaw=yaw, abort=abort_pre)
        if r == "abort":
            return "abort"
        # refine with the latest detection (the target should still be visible below the hand)
        if t.missing == 0:
            yaw = self.grasp_yaw(t)
        r = self.move_to([t.xy[0], t.xy[1], tz + max(0.06, s + 0.03)], -1.0, tol=0.006, speed=SPEED_NEAR,
                         max_calls=12, yaw=yaw, abort=abort_pre)
        if r == "abort":
            return "abort"
        gz = tz + max(0.012, 0.45 * s)
        r = self.move_to([t.xy[0], t.xy[1], gz], -1.0, tol=0.005, speed=SPEED_NEAR, max_calls=12, yaw=yaw,
                         yaw_tol=0.1, abort=abort_pre, chunk=2)
        if r == "abort":
            self.move_to([self.pos()[0], self.pos()[1], tz + HOVER_DZ], -1.0, speed=SPEED_NEAR, max_calls=8)
            return "abort"
        zr = self.pos()[2]
        self._grip_cmd = 1.0
        self.hold(CLOSE_STEPS, 1.0)
        fs = self.finger_sum()
        self.jlog("grasp", tid=t.id, finger_sum=fs, z=float(zr), move=r)
        if fs < 0.25 * s:
            log("grasp missed (fingers", round(fs, 4), ")")
            self._grip_cmd = -1.0
            self.hold(OPEN_STEPS, -1.0)
            self.move_to([self.pos()[0], self.pos()[1], tz + HOVER_DZ], -1.0, speed=SPEED_NEAR, max_calls=8)
            return "miss"
        t.held = True
        pick_xy = self.pos()[:2].copy()
        self.move_to([pick_xy[0], pick_xy[1], tz + LIFT_DZ], 1.0, tol=0.012, speed=SPEED_NEAR, max_calls=12)
        self.hold(2, 1.0)
        fs2 = self.finger_sum()
        held = fs2 >= 0.25 * s
        self.jlog("lifted", tid=t.id, finger_sum=fs2, held=held, z=float(self.pos()[2]))
        if not held:
            log("cube dropped during lift")
            t.held = False
            t.missing = 3
            self._grip_cmd = -1.0
            self.hold(OPEN_STEPS, -1.0)
            return "drop"
        t.done = True
        self.pick_durations.append(self.step - t_start)
        log("picked", t.id, "at step", self.step, "duration", self.step - t_start)
        self.place_back(t, pick_xy)
        return "ok"

    def relocate_lost(self, t):
        """A target that is not seen (or was missed) where we think it is: the cube count is conserved, so an
        unclaimed detection elsewhere is most likely it."""
        if t.missing < 3:
            return
        free = getattr(self, "free_dets", [])
        if not free:
            return
        d = min(free, key=lambda d: float(np.linalg.norm(np.asarray(d["xy"]) - t.xy)))
        log("relocate lost target", t.id, "from", np.round(t.xy, 3).tolist(), "to", np.round(d["xy"], 3).tolist())
        self.jlog("relocate", tid=t.id, old=t.xy.tolist(), new=np.asarray(d["xy"]).tolist())
        t.xy = np.asarray(d["xy"], float).copy()
        t.anchor = t.xy.copy()
        t.seen_xy = t.xy.copy()
        t.seen_step = self.step
        t.vel = np.zeros(2)
        t.det = d

    def remaining_targets(self):
        return [x for x in self.tr.tracks if x.highlight and not x.done and x.attempts < MAX_ATTEMPTS]

    def free_spot(self, xy):
        s = self.scene.cube_size
        others = [o.xy for o in self.tr.tracks if not o.held]
        def clear(q):
            return min([np.linalg.norm(q - o) for o in others] + [1.0])
        if clear(xy) > 1.3 * s:
            return xy
        for r in (0.03, 0.05, 0.07, 0.09):
            for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                q = xy + r * np.array([math.cos(a), math.sin(a)])
                if clear(q) > 1.6 * s:
                    return q
        return xy

    def place_back(self, t, xy):
        sc = self.scene
        s = sc.cube_size
        tz = sc.table_z
        more = len(self.remaining_targets()) > 0
        if self.shuffle_active and more:
            # keep tracking clean: retreat with the cube, wait, come back
            self.retreat_for_shuffle(1.0)
            self.wait_quiet()
        q = self.free_spot(np.asarray(xy, float))
        self.move_to([q[0], q[1], tz + HOVER_DZ], 1.0, tol=0.01, speed=SPEED_FREE, max_calls=25)
        self.move_to([q[0], q[1], tz + 0.5 * s + 0.012], 1.0, tol=0.006, speed=SPEED_NEAR, max_calls=10, chunk=2)
        self._grip_cmd = -1.0
        self.hold(OPEN_STEPS, -1.0)
        t.held = False
        t.xy = np.asarray(q, float).copy()
        t.anchor = t.xy.copy()
        t.seen_xy = t.xy.copy()
        t.seen_step = self.step
        t.vel = np.zeros(2)
        t.missing = 1
        self.move_to([q[0], q[1], tz + HOVER_DZ], -1.0, tol=0.012, speed=SPEED_NEAR, max_calls=10)
        self.jlog("placed", tid=t.id, xy=q.tolist())

    def picks(self):
        est_core = 30
        while True:
            if self.elapsed() > WALL_LIMIT_S - 120:
                log("picks: wall budget exhausted")
                return
            # keep enough steps for one more full pick + place-back and the final press
            full = (float(np.median(self.pick_durations)) + 22) if self.pick_durations else 75.0
            if self.remaining is not None and self.remaining < full + FINAL_RESERVE:
                log("picks: step budget low", self.remaining, "need", round(full + FINAL_RESERVE))
                return
            t = self.choose_target()
            if t is None:
                return
            if self.shuffle_active:
                self.retreat_for_shuffle(-1.0)
                self.wait_quiet()
                continue
            n = self.next_shuffle_in()
            P = self.shuffle_period()
            if self.pick_durations:
                est_core = int(np.clip(np.median(self.pick_durations) * 0.7, 20, 60))
            gap = (P - self.shuffle_duration()) if P is not None else 0.0
            # wait only if a later quiet window is longer than what is left of this one (no livelock right after a shuffle)
            if n is not None and P is not None and n < est_core and gap > est_core + 4 and n < gap - 8:
                if self.remaining is not None and self.remaining < n + 25 + full + FINAL_RESERVE:
                    log("picks: no time to wait for the shuffle", self.remaining)
                    return
                log("waiting for predicted shuffle in", n, "steps (P", P, ")")
                self.go_home(-1.0)
                t0 = self.step
                while not self.shuffle_active and self.step - t0 < n + 25:
                    self.hold(CHUNK_FINE, -1.0)
                self.wait_quiet()
                continue
            self.relocate_lost(t)
            r = self.pick(t)
            self.jlog("pick_result", tid=t.id, result=r, attempts=t.attempts)
            if r == "miss":
                t.missing = max(t.missing, 3)
            if r == "abort":
                t.attempts -= 1 if t.attempts > 0 else 0
                self.retreat_for_shuffle(-1.0)
                self.wait_quiet()
            if not self.remaining_targets():
                return

    def run(self):
        o = self.observe()
        log("instruction:", o.instruction, "step", o.step_id, "remaining", o.remaining, "max", o.max_steps)
        self.jlog("first_obs", keys=list(o.meta.keys()), rgb=list(o.rgb.keys()), depth=list(o.depth.keys()),
                  state=o.state, meta=json.dumps(o.meta)[:3000])
        self.home = np.asarray(o.state[:3], float).copy()
        self.yaw0 = self.finger_yaw()
        self.calibrate(o)
        # conditioning video (if the reset sequence carries one)
        try:
            self.process_reset_video(o)
        except Exception as e:
            log("reset video processing failed", repr(e))
        if not self.hl_done:
            self.watch()
        try:
            self.wait_quiet(max_steps=60)
            self.press_button("start")
        except EpisodeEnded:
            raise
        except Exception as e:
            log("press start error", repr(e), traceback.format_exc())
        try:
            self.picks()
        except EpisodeEnded:
            raise
        except Exception as e:
            log("picks error", repr(e), traceback.format_exc())
            self.jlog("error", where="picks", err=traceback.format_exc())
        self.final()

    def final(self):
        self.phase = "final"
        try:
            if self.finger_sum() > 0.01 and getattr(self, "_grip_cmd", -1) > 0:
                self.hold(OPEN_STEPS, -1.0)
        except EpisodeEnded:
            raise
        except Exception:
            pass
        for k in range(3):
            self.press_button("final%d" % k)
            self.hold(4, 1.0)
            if self.complete or self.ended:
                return
            log("not complete after final press", k)
            self.hold(6, 1.0)
            if self.complete:
                return

    def process_reset_video(self, o):
        paths = rio.manifest_paths(o.meta)
        if not paths:
            return
        man = rio.read_manifest(paths[0])
        frames = [f for f in man.get("frames", []) if f.get("phase") == "video_demo"]
        self.jlog("reset_video", n=len(frames), manifest=paths[0])
        if not frames:
            return
        frames.sort(key=lambda f: f.get("frame_index", 0))
        log("reset video frames:", len(frames))
        self.hl_watch = True
        t_last = frames[-1].get("timestamp_s", 0.0)
        if len(frames) > 400:   # keep the pre-control phase short (<= 600 s before the first step)
            k = int(math.ceil(len(frames) / 400.0))
            frames = frames[::k] if frames[-1] in frames[::k] else frames[::k] + [frames[-1]]
        for f in frames:
            if self.elapsed() > 300:
                log("reset video: time cap reached")
                break
            files = {(x.get("camera"), x.get("kind")): x for x in f.get("files", [])}
            if ("front", "rgb") not in files or ("front", "depth") not in files:
                continue
            k, rgb = rio.load_file_image(files[("front", "rgb")]["path"])
            k, d = rio.load_file_image(files[("front", "depth")]["path"])
            d = d.astype(float) * float(files[("front", "depth")].get("meters_per_unit", 0.001))
            vo = Obs()
            vo.rgb = {"front": rgb}
            vo.depth = {"front": d}
            vo.state = None
            st = int(round((f.get("timestamp_s", 0.0) - t_last) * 20))
            self.perceive(vo, step=st)
        if self.hl_seen and not self.hl_done:
            self.finalize_highlight(0)
        self.hl_watch = False
        # re-anchor tracks at the live frame
        self.perceive(o)


def highlight_threshold(vals, lo=0.15, min_gap=0.06):
    """Split ring ratios into highlighted / not: midpoint of the largest gap whose upper side is >= 0.15.

    Demo evidence: highlighted 0.44-0.76, others <= 0.20 (gap >= 0.24).
    """
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return percep.HL_RATIO_THR
    sv = np.sort(vals)[::-1]
    sv = np.append(sv, 0.0)
    best = None
    for j in range(len(sv) - 1):
        if sv[j] < lo:
            break
        g = sv[j] - sv[j + 1]
        if best is None or g > best[0]:
            best = (g, (sv[j] + sv[j + 1]) / 2)
    if best is None or best[0] < min_gap:
        return percep.HL_RATIO_THR if lo == 0.15 else None
    return float(best[1])


class _SwapWrap:
    def __init__(self, m):
        self.m = m
        self.name = "affine-swapuv"

    def backproject(self, u, v, d):
        return self.m.backproject(v, u, d)


def main(ctx):
    c = Ctl(ctx)
    log("start", VERSION)
    try:
        c.run()
    except EpisodeEnded:
        pass
    except Exception as e:
        log("fatal", repr(e))
        log(traceback.format_exc())
        c.jlog("fatal", err=traceback.format_exc())
        # best effort: still try to finish with a button press
        try:
            if not c.ended and c.scene.button is not None:
                c.final()
        except Exception:
            pass
    log("done: complete", c.complete, "ended", c.ended, "step", c.step, "calls", c.ncalls,
        "avg call s", round(c.call_time / max(c.ncalls, 1), 3), "elapsed", round(c.elapsed(), 1))
    c.jlog("done", complete=c.complete, ended=c.ended, tracks=[t.brief() for t in c.tr.tracks],
           shuffle_starts=c.shuffle_starts, shuffle_ends=c.shuffle_ends, presses=c.presses)
    try:
        c.logf.close()
    except Exception:
        pass
