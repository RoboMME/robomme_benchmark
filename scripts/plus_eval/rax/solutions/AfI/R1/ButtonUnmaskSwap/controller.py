"""ButtonUnmaskSwap controller (bums) v1.

Pipeline (each phase returns on failure and the next phase still runs; main never raises):
  INIT     observe, parse instruction (target cube colours), load conditioning video frames,
           calibrate front back-projection (tool-fitted linear model / parsed calibration),
           table plane, analyze frames, identify the button (flattest non-group object).
  PRESS    closed gripper, travel above button, descend until stall, push, lift.  (deadline step 179)
  WATCH    return to home pose, hold still, analyze every saved frame until bins static for
           WATCH_STATIC_STEPS and no target cube visible (cap WATCH_CAP_STEP).
  DECIDE   batch reverse-time tracking of bins from the final static frame; cube->bin by the
           latest covering event (cube vanishes, bin at its last-seen place).
  PICK     for each target colour: top-down grasp across the bin's short side, lift; if more targets,
           put it back down, release, continue.
  HOLD     keep the bin lifted until episode ends or steps run out.
Transitions: INIT->PRESS->WATCH->DECIDE->PICK->HOLD; any exception in a phase is logged and
the controller moves to the next phase that can still run; EpisodeEnded stops everything.
"""
import json
import math
import os
import re
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import geom  # noqa: E402
import io_utils as iu  # noqa: E402
import perception as pc  # noqa: E402
import tracker as tk  # noqa: E402
from robot import Robot, EpisodeEnded  # noqa: E402

VERSION = "bums-v1"
OUT = "/workspace/output"

PRESS_DEADLINE = 179
WATCH_STATIC_STEPS = 50
WATCH_CAP_STEP = 330
WATCH_NOMOTION_STEP = 250   # if no motion was ever seen, keep watching until here
PREWATCH_NOMOTION_STEP = 40
PREWATCH_STATIC_STEPS = 30
PREWATCH_CAP_STEP = 95      # press (~45-60 steps) must finish before step 179
PICK_RESERVE_STEPS = 140
TRAVEL_CLEAR = 0.10
SPEED_FREE = 0.5
SPEED_NEAR = 0.25
GRASP_DEPTH = 0.03          # fingertip below bin top
MAX_GRASP_WIDTH = 0.074
MAX_VIDEO_FRAMES = 520
WALL_LIMIT = 1100.0         # s, controller self limit (runtime limit 1200)


class Logger:
    def __init__(self):
        self.t0 = time.monotonic()
        os.makedirs(OUT, exist_ok=True)
        try:
            self.f = open(os.path.join(OUT, "bums_log.txt"), "a")
        except Exception:
            self.f = None

    def __call__(self, msg):
        line = f"[{VERSION} {time.monotonic() - self.t0:7.1f}s] {msg}"
        try:
            print(line, flush=True)
        except Exception:
            pass
        if self.f:
            try:
                self.f.write(line + "\n")
                self.f.flush()
            except Exception:
                pass


def parse_targets(instr):
    s = instr.lower()
    for a, b in pc.SYNONYMS.items():
        s = s.replace(a, b)
    words = re.findall(r"[a-z]+", s)
    names = set(pc.COLOR_NAMES) | {"grey", "gray", "white", "black", "brown"}
    targets = []
    for i, w in enumerate(words):
        if w in names:
            nxt = words[i + 1:i + 4]
            if any(x.startswith("cube") or x.startswith("block") for x in nxt):
                targets.append(w)
    if not targets:
        targets = [w for w in words if w in pc.COLOR_NAMES]
    out = []
    for t in targets:
        if t not in out:
            out.append(t)
    return out


class Controller:
    def __init__(self, ctx):
        self.log = Logger()
        self.robot = Robot(ctx, self.log)
        self.scene = None
        self.cache = {}  # frame key -> analysis
        self.button = None
        self.targets = []
        self.instr = ""
        self.home = None
        self.R0 = None
        self.press_done_step = None
        self.decisions = []
        self.final_dets = []
        self.prof = None
        self.obj_top = 0.08
        self.watch_prev = None
        self.static_steps = 0
        self.last_watch_step = None
        self.v_end = 0.0
        self.n_obs_fallback = 0
        self.video_motion = False
        self.pre = {}
        self.post = {}

    # ------------------------------------------------------------ helpers
    def wall(self):
        return time.monotonic() - self.log.t0

    def frame_time(self, fr):
        if fr["phase"] == "video_demo":
            return float(fr["ts"] or 0.0)
        return self.v_end + 0.05 + 0.05 * fr["step"]

    def analyze(self, fr):
        k = fr["key"]
        if k in self.cache:
            return self.cache[k]
        rgb = iu.load_rgb(fr["files"][("front", "rgb")][0])
        dp, mpu = fr["files"][("front", "depth")]
        depth = iu.load_depth(dp, mpu)
        tcp = fr["state"][:3] if fr.get("state") else None
        if fr["phase"] == "video_demo" and tcp is not None and all(abs(x) < 1e-9 for x in tcp):
            tcp = None
        res = pc.analyze_frame(self.scene, rgb, depth, tcp)
        self.cache[k] = res
        return res

    def ordered_frames(self):
        vid = [f for f in self.robot.frames if f["phase"] == "video_demo"]
        exe = [f for f in self.robot.frames if f["phase"] != "video_demo"]
        vid.sort(key=lambda f: (f["ts"] if f["ts"] is not None else 0.0, f["frame_index"] or 0))
        exe.sort(key=lambda f: f["step"])
        return vid, exe

    # ------------------------------------------------------------ INIT
    def init(self):
        r = self.robot
        try:
            sj = r.status()
            self.log("robomme_status: " + json.dumps(iu.summarize(sj))[:2500])
        except Exception as ex:
            self.log(f"robomme_status failed {ex!r}")
        js, raw = r.observe()
        self.instr = iu.get_instruction(js)
        self.targets = parse_targets(self.instr)
        self.log(f"instruction={self.instr!r} targets={self.targets} step={r.step_id} state={r.state}")
        self.log(f"limits max={r.max_steps} remaining={r.remaining} frames_loaded={len(r.frames)}")
        # if observe gave no manifest frames, try the file entries
        cur = self.current_frame(js, raw)
        if cur is None:
            raise RuntimeError("no front rgb/depth for current observation")
        self.home = r.pos.copy()
        self.R0 = r.R.copy()
        if float(self.R0[:, 2] @ np.array([0, 0, -1.0])) < 0.9:
            self.log(f"WARNING home tool z-axis not down: {self.R0[:, 2]}")
        self.calibrate(js, cur)
        try:
            self.check_finger_axis(cur)
        except Exception as ex:
            self.log(f"finger axis check failed {ex!r}")
        vid, exe = self.ordered_frames()
        if vid:
            self.v_end = max(float(f["ts"] or 0.0) for f in vid)
        self.log(f"video frames={len(vid)} exec frames={len(exe)} v_end={self.v_end:.2f}s")
        # analyze t0 and video
        t = time.monotonic()
        if len(vid) > MAX_VIDEO_FRAMES:
            idx = np.linspace(0, len(vid) - 1, MAX_VIDEO_FRAMES).round().astype(int)
            keep = set(int(i) for i in idx)
            drop = [f for i, f in enumerate(vid) if i not in keep]
            dk = set(f["key"] for f in drop)
            r.frames = [f for f in r.frames if f["key"] not in dk]
            vid = [f for i, f in enumerate(vid) if i in keep]
        n_ok = 0
        for f in vid:
            try:
                self.analyze(f)
                n_ok += 1
            except Exception as ex:
                self.log(f"video frame analyze failed {f['key']}: {ex!r}")
            if time.monotonic() - t > 300:
                self.log("video analysis time cap hit")
                break
        self.log(f"analyzed {n_ok} video frames in {time.monotonic() - t:.1f}s")
        res0 = self.analyze(cur)
        self.log_comps("t0", res0)
        self.identify_button(res0, vid)
        try:
            sigs = [self.scene_signature(self.cache[f["key"]]) for f in vid if f["key"] in self.cache]
            sigs.append(self.scene_signature(res0))
            ch = 0
            for (d1, c1), (d2, c2) in zip(sigs[:-1], sigs[1:]):
                if len(d1) != len(d2) or c1 != c2:
                    ch += 1
                elif d1:
                    cost = np.linalg.norm(np.array(d2)[:, None] - np.array(d1)[None], axis=2)
                    if cost.min(axis=1).max() > 0.006:
                        ch += 1
            self.video_motion = ch >= 2
            self.log(f"video scene changes={ch} video_motion={self.video_motion}")
        except Exception as ex:
            self.log(f"video motion check failed {ex!r}")
        tops = [c.hmax for c in res0["comps"] if not c.robot]
        self.obj_top = max(tops) if tops else 0.08
        self.obj_top = min(max(self.obj_top, 0.04), 0.2)

    def current_frame(self, js, raw, require_step=None):
        r = self.robot
        exe = [f for f in r.frames if f["phase"] != "video_demo"]
        if exe:
            exe.sort(key=lambda f: f["step"])
            if require_step is None or exe[-1]["step"] >= require_step:
                return exe[-1]
        sid = r.step_id
        if ("e", sid) in r.frame_keys:
            return [f for f in r.frames if f["key"] == ("e", sid)][0]
        files = iu.file_entries(js)
        if ("front", "rgb") in files and ("front", "depth") in files:
            fr = {"key": ("e", r.step_id), "phase": "execution", "ts": None, "step": r.step_id,
                  "frame_index": None, "state": r.state, "files": files}
            r.frames.append(fr)
            r.frame_keys.add(fr["key"])
            return fr
        # inline images: save to output and use
        imgs = iu.reply_images(raw)
        self.log(f"inline images: {[(t[:40], im.mode, im.size) for t, im in imgs]}")
        rgbs = [(t, im) for t, im in imgs if im.mode in ("RGB", "RGBA", "P")]
        deps = [(t, im) for t, im in imgs if im.mode in ("I;16", "I", "I;16B", "F", "L")]

        def pick(lst, cam):
            for t, im in lst:
                if cam in t.lower():
                    return im
            return lst[0][1] if lst else None
        rgb, dep = pick(rgbs, "front"), pick(deps, "front")
        if rgb is None or dep is None:
            return None
        p1 = os.path.join(OUT, f"inline_front_rgb_{r.step_id}.png")
        p2 = os.path.join(OUT, f"inline_front_depth_{r.step_id}.png")
        rgb.convert("RGB").save(p1)
        dep.save(p2)
        fr = {"key": ("e", r.step_id), "phase": "execution", "ts": None, "step": r.step_id,
              "frame_index": None, "state": r.state,
              "files": {("front", "rgb"): (p1, 0.001), ("front", "depth"): (p2, 0.001)}}
        r.frames.append(fr)
        r.frame_keys.add(fr["key"])
        return fr

    def calibrate(self, js, cur):
        depth = iu.load_depth(*cur["files"][("front", "depth")])
        H, W = depth.shape
        cands = []
        # parsed calibration
        intr = pose = None
        try:
            ci = iu.find_key(js, "camera_intrinsics")
            cp = iu.find_key(js, "camera_poses")
            if ci is not None:
                intr = geom.parse_intrinsics(geom.get_cam_entry(ci, "front") if isinstance(ci, dict) else ci)
            if cp is not None:
                pose = geom.parse_pose(geom.get_cam_entry(cp, "front") if isinstance(cp, dict) else cp)
            self.log(f"parsed intrinsics={intr} pose={'ok' if pose else None}")
        except Exception as ex:
            self.log(f"calib parse failed {ex!r}")
        if intr and pose:
            for gl in (True, False):
                for off in (0.0, 0.5):
                    cands.append((f"parsed gl={gl} off={off}", geom.pinhole_linear(intr, pose, gl, off)))
        # tool samples
        samples = []
        try:
            vv, uu = np.meshgrid(np.linspace(20, H - 21, 6), np.linspace(20, W - 21, 6), indexing="ij")
            sels = []
            for v, u in zip(vv.ravel(), uu.ravel()):
                u, v = int(u), int(v)
                if depth[v, u] > 0:
                    sels.append((u, v))
            sels = sels[:32]
            rep = self.robot.pixel_to_position(cur["step"], [{"camera": "front", "u": u, "v": v} for u, v in sels])
            pts = parse_points(rep)
            self.log(f"pixel_to_position: {len(pts)} points parsed from {len(sels)} sels; sample={json.dumps(iu.summarize(rep))[:800]}")
            for (u, v), p in zip(sels, pts):
                if p is not None:
                    samples.append((u, v, p))
        except Exception as ex:
            self.log(f"pixel_to_position failed {ex!r}")
        best = None
        if len(samples) >= 6:
            for swap in (False, True):
                S = []
                for u, v, p in samples:
                    d = depth[u, v] if swap else depth[v, u]
                    if d > 0 and (not swap or (u < H and v < W)):
                        S.append((u, v, float(d), np.asarray(p, dtype=float)))
                if len(S) >= 6:
                    M, rms = geom.fit_linear(S)
                    self.log(f"fit swap={swap} rms={rms:.4f}")
                    if best is None or rms < best[2]:
                        best = (f"fit swap={swap}", M, rms, swap)
            for name, M in cands:
                errs = [np.linalg.norm(M @ np.array([u * depth[v, u], v * depth[v, u], depth[v, u], 1.0]) - np.asarray(p))
                        for u, v, p in samples if depth[v, u] > 0]
                rms = float(np.sqrt(np.mean(np.square(errs)))) if errs else 9
                self.log(f"cand {name} rms={rms:.4f}")
                if best is None or rms < best[2] - 1e-4:
                    best = (name, M, rms, False)
        if best is not None and best[2] < 0.02:
            M = np.array(best[1], dtype=float)
            if best[3]:  # tool used (u=row, v=col): swap the u/v feature columns for our (col,row) convention
                M = M[:, [1, 0, 2, 3]]
            cam = geom.LinearCam(M, source=best[0])
        elif cands:
            cam = geom.LinearCam(cands[0][1], source=cands[0][0] + " (unverified)")
        elif best is not None:
            cam = geom.LinearCam(best[1], source=best[0] + " (poor)")
        else:
            raise RuntimeError("no camera model")
        self.log(f"camera model: {cam.source} rms={best[2] if best else None}")
        P = cam.backproject_image(depth)
        tab = pc.estimate_table(P, depth > 0)
        if tab is None:
            raise RuntimeError("table estimate failed")
        z_table, roi, frac = tab
        self.log(f"table z={z_table:.4f} roi={tuple(round(x, 3) for x in roi)} frac={frac:.2f}")
        self.scene = pc.Scene(cam, z_table, roi)

    def check_finger_axis(self, cur):
        """The Panda hand is elongated along the finger closing axis: compare its XY principal axis
        (front depth points just above the TCP) with the state's R[:,1] (assumed) and R[:,0]."""
        depth = iu.load_depth(*cur["files"][("front", "depth")])
        P = self.scene.cam.backproject_image(depth)
        tcp = self.robot.pos
        dxy = np.hypot(P[..., 0] - tcp[0], P[..., 1] - tcp[1])
        dz = P[..., 2] - tcp[2]
        m = (depth > 0) & (dxy < 0.13) & (dz > 0.045) & (dz < 0.13)
        pts = P[m][:, :2]
        if len(pts) < 60:
            self.log(f"finger axis check: too few hand points ({len(pts)})")
            return
        c = pts - pts.mean(axis=0)
        w, V = np.linalg.eigh(c.T @ c / len(c))
        ratio = math.sqrt(max(w[1], 1e-12) / max(w[0], 1e-12))
        major = V[:, 1]
        R = self.robot.R
        a1 = abs(float(major @ (R[:2, 1] / max(np.linalg.norm(R[:2, 1]), 1e-9))))
        a0 = abs(float(major @ (R[:2, 0] / max(np.linalg.norm(R[:2, 0]), 1e-9))))
        self.log(f"finger axis check: n={len(pts)} elong={ratio:.2f} |cos| with R[:,1]={a1:.2f} R[:,0]={a0:.2f}")
        if ratio > 1.8 and a0 > 0.9 and a1 < 0.45:
            self.robot.finger_axis = 0
            self.log("finger axis -> R[:,0]")

    def log_comps(self, tag, res):
        for c in res["comps"]:
            self.log(f"  {tag} comp {json.dumps(c.as_dict())}")
        for b in res["blobs"]:
            self.log(f"  {tag} blob {b['color']} n={b['n']} pos={np.round(b['pos'], 3).tolist()} ext={b['ext']:.3f} hmax={b['hmax']:.3f}")

    def identify_button(self, res0, vid):
        cands = [c for c in res0["comps"] if not c.robot and c.hmax < 0.09 and c.ext_long > 0.018]
        if not cands:
            self.log("no button candidates at t0")
            return
        # static over video?
        moved = {}
        vres = [self.cache[f["key"]] for f in vid if f["key"] in self.cache]
        for c in cands:
            mv = 0.0
            for rr in vres[:: max(1, len(vres) // 40)] if vres else []:
                ds = [math.hypot(c.centroid[0] - o.centroid[0], c.centroid[1] - o.centroid[1]) for o in rr["comps"] if not o.robot]
                if ds:
                    mv = max(mv, min(ds))
            moved[id(c)] = mv
        scored = []
        for c in cands:
            flat = c.hmax / max(c.ext_long, 1e-3)
            group = 0
            for o in cands:
                if o is c:
                    continue
                if (abs(o.ext_long - c.ext_long) < 0.25 * c.ext_long and abs(o.hmax - c.hmax) < 0.25 * c.hmax
                        and np.linalg.norm(o.rgb - c.rgb) < 45):
                    group += 1
            s = flat
            if group:
                s += 0.6
            if pc.is_cubelike(c) and flat > 0.55:
                s += 0.5
            if moved[id(c)] > 0.05:
                s += 1.0
            scored.append((s, c, flat, group))
        scored.sort(key=lambda x: x[0])
        for s, c, flat, group in scored:
            self.log(f"  button cand score={s:.2f} flat={flat:.2f} group={group} moved={moved[id(c)]:.3f} {json.dumps(c.as_dict())}")
        c = scored[0][1]
        self.button = {"xy": np.array(c.top[:2]), "top": float(c.top[2]), "hmax": c.hmax, "ext": c.ext_long}
        pc.BUTTON_R = float(min(0.07, max(0.045, 0.5 * c.ext_long + 0.012)))
        self.log(f"BUTTON at {np.round(c.top, 3).tolist()} ext={c.ext_long:.3f} exclusion_r={pc.BUTTON_R:.3f}")

    # ------------------------------------------------------------ frame processing during motion
    def on_frames(self, force_current=False):
        if self.wall() > WALL_LIMIT:
            return
        _, exe = self.ordered_frames()
        if force_current and (not exe or exe[-1]["step"] < self.robot.step_id - 5):
            try:
                js, raw = self.robot.observe()
                self.current_frame(js, raw, require_step=self.robot.step_id)
                self.n_obs_fallback += 1
                if self.n_obs_fallback <= 3:
                    self.log(f"observe fallback used at step {self.robot.step_id}")
            except Exception as ex:
                self.log(f"observe fallback failed {ex!r}")
            _, exe = self.ordered_frames()
        for f in exe:
            if f["key"] not in self.cache:
                try:
                    self.analyze(f)
                except Exception as ex:
                    self.cache[f["key"]] = {"comps": [], "blobs": []}
                    self.log(f"analyze failed {f['key']}: {ex!r}")

    # ------------------------------------------------------------ PRESS
    def R_yaw(self, dyaw):
        return geom.rotz(dyaw) @ self.R0

    def press(self):
        r = self.robot
        if self.button is None:
            self.log("PRESS skipped: no button")
            return
        bx, by = self.button["xy"]
        top = self.button["top"]
        z_tr = max(self.scene.z_table + self.obj_top + TRAVEL_CLEAR, top + 0.08)
        z_tr = min(z_tr, max(self.home[2], self.scene.z_table + 0.25))
        self.log(f"PRESS start step={r.step_id} button=({bx:.3f},{by:.3f},{top:.3f}) z_travel={z_tr:.3f}")
        # rise first if below travel height
        if r.pos[2] < z_tr - 0.01:
            r.move_to([r.pos[0], r.pos[1], z_tr], self.R0, grip=1.0, speed=SPEED_FREE, tol=0.02,
                      max_steps=25, on_frames=self.on_frames)
        m = r.move_to([bx, by, z_tr], self.R0, grip=1.0, speed=0.6, tol=0.01, max_steps=60,
                      on_frames=self.on_frames)
        self.log(f"  above button: {m} pos={np.round(r.pos, 3).tolist()} step={r.step_id}")
        m = r.move_to([bx, by, top + 0.025], self.R0, grip=1.0, speed=0.35, tol=0.006, max_steps=30,
                      on_frames=self.on_frames)
        self.log(f"  pre-press: {m} pos={np.round(r.pos, 3).tolist()} fingers={r.finger_sum:.4f} step={r.step_id}")
        if r.step_id > PRESS_DEADLINE - 25:
            self.log(f"  WARNING late press at step {r.step_id}")
        z_before = r.pos[2]
        m = r.move_to([bx, by, top - 0.03], self.R0, grip=1.0, speed=0.12, tol=0.004, max_steps=24,
                      max_chunk=2, on_frames=self.on_frames, stall_check=True)
        self.log(f"  descend: {m} pos={np.round(r.pos, 3).tolist()} dz={r.pos[2] - z_before:.3f} step={r.step_id}")
        # push
        rows = [[0, 0, -0.35, 0, 0, 0, 1.0]] * 6
        r.step(rows)
        self.on_frames()
        self.log(f"  pushed: pos={np.round(r.pos, 3).tolist()} (top was {top:.3f}) step={r.step_id}")
        self.press_done_step = r.step_id
        m = r.move_to([bx, by, z_tr], self.R0, grip=1.0, speed=SPEED_NEAR, tol=0.015, max_steps=25,
                      on_frames=self.on_frames)
        self.log(f"  lifted: {m} step={r.step_id}")

    # ------------------------------------------------------------ WATCH
    def scene_signature(self, res):
        bxy = self.button["xy"] if self.button else None
        dets = [np.array(c.top[:2]) for c in pc.final_bins(res, bxy)]
        cubes = sorted(b["color"] for b in pc.cube_blobs(res, bin_color=(self.prof or {}).get("color"), button_xy=bxy))
        return dets, cubes

    def watch_loop(self, label, done_fn, cap_step):
        """Hold still, analyze frames; done_fn(state) -> bool. Returns watch state."""
        r = self.robot
        st = {"static": 0, "motion": False, "motion_steps": 0, "dets": 0, "cubes": [], "tcubes": []}
        prev = None
        prev_step = r.step_id
        start = r.step_id
        while not r.ended:
            if r.step_id >= cap_step or r.steps_left() < PICK_RESERVE_STEPS or self.wall() > WALL_LIMIT * 0.6:
                self.log(f"{label} cap: step={r.step_id} left={r.steps_left()} wall={self.wall():.0f}")
                break
            r.hold(5, grip=r.grip)
            self.on_frames(force_current=True)
            _, exe = self.ordered_frames()
            if not exe:
                continue
            res = self.cache.get(exe[-1]["key"])
            if res is None:
                continue
            dets, cubes = self.scene_signature(res)
            moving = prev is None
            if prev is not None:
                pd, pcub = prev
                if len(dets) != len(pd) or cubes != pcub:
                    moving = True
                elif dets:
                    cost = np.linalg.norm(np.array(dets)[:, None] - np.array(pd)[None], axis=2)
                    moving = bool(cost.min(axis=1).max() > 0.006)
                if moving:
                    st["motion"] = True
                    st["motion_steps"] += r.step_id - prev_step
            st["static"] = st["static"] + (r.step_id - prev_step) if not moving else 0
            prev_step = r.step_id
            prev = (dets, cubes)
            st["dets"], st["cubes"] = len(dets), cubes
            st["tcubes"] = [c for c in cubes if c in self.targets]
            if not moving and dets:
                self.prof = tk.bin_profile(pc.final_bins(res, self.button["xy"] if self.button else None))
            if (r.step_id - start) % 20 < 5:
                self.log(f"  {label} step={r.step_id} bins={len(dets)} static={st['static']} motion={st['motion']} cubes_vis={cubes}")
            if done_fn(st):
                self.log(f"{label} done: {st} step={r.step_id}")
                break
        return st

    def prewatch(self):
        """Before pressing: if the scene is changing, watch it with the arm still (no occlusion)."""
        r = self.robot
        def done(st):
            if not st["motion"] and r.step_id >= PREWATCH_NOMOTION_STEP:
                return True
            if st["motion"] and st["static"] >= PREWATCH_STATIC_STEPS:
                return True
            return False
        self.pre = self.watch_loop("PREWATCH", done, PREWATCH_CAP_STEP)

    def watch(self):
        r = self.robot
        home = self.home.copy()
        m = r.move_to(home, self.R0, grip=-1.0, speed=SPEED_FREE, tol=0.015, max_steps=40,
                      on_frames=self.on_frames)
        self.log(f"WATCH home reached {m} step={r.step_id}")
        pre_motion = bool(getattr(self, "pre", {}) and self.pre.get("motion"))

        def done(st):
            if st["static"] < WATCH_STATIC_STEPS or st["dets"] < 1:
                return False
            if st["tcubes"] and r.step_id < WATCH_CAP_STEP:
                return False
            if st["motion"] or pre_motion or self.video_motion:
                return True
            return r.step_id >= WATCH_NOMOTION_STEP
        self.post = self.watch_loop("WATCH", done, WATCH_CAP_STEP)

    # ------------------------------------------------------------ DECIDE
    def decide(self):
        r = self.robot
        self.on_frames()
        vid, exe = self.ordered_frames()
        frames = []
        for f in vid + exe:
            if f["key"] in self.cache:
                frames.append({"t": self.frame_time(f), "res": self.cache[f["key"]], "key": f["key"]})
        if not frames:
            raise RuntimeError("no frames for decision")
        bxy = self.button["xy"] if self.button else None
        comps = []
        while frames:
            last = frames[-1]
            comps = pc.final_bins(last["res"], bxy)
            if comps:
                break
            self.log(f"no bins in frame {last['key']}, dropping it")
            frames = frames[:-1]
        if not frames:
            raise RuntimeError("no frame with bins")
        self.prof = tk.bin_profile(comps)
        self.log(f"DECIDE frames={len(frames)} final bins={len(comps)} prof={self.prof}")
        self.log_comps("final", last["res"])
        final_xy = [np.array(c.top[:2]) for c in comps]
        self.final_dets = comps
        hist, matched = tk.reverse_track(frames, final_xy, bxy, self.prof)
        cubes = tk.cube_tracks(frames, self.prof, bxy)
        for col, obs in cubes.items():
            self.log(f"  cube {col}: visible in {len(obs)} frames, first k={obs[0][0]} last k={obs[-1][0]} last pos={np.round(obs[-1][1], 3).tolist()}")
        assign = tk.assign_cubes(frames, hist, cubes)
        for col, a in assign.items():
            self.log(f"  assign {col}: {a}")
        for j in range(len(final_xy)):
            mr = float(matched[j].mean())
            # trajectory summary: positions at a few times
            ks = np.linspace(0, len(frames) - 1, 8).round().astype(int)
            self.log(f"  track {j}: matched={mr:.2f} path={[np.round(hist[j, k], 3).tolist() for k in ks]}")
        self.decisions = tk.decide(self.targets, assign, final_xy)
        self.log(f"DECISIONS {self.decisions}")
        try:
            save_debug(last, frames, hist, comps, self.decisions, self.scene)
        except Exception as ex:
            self.log(f"debug save failed {ex!r}")

    # ------------------------------------------------------------ PICK
    def grasp_plan(self, c):
        tp = c.top_pts if c.top_pts is not None and len(c.top_pts) >= 5 else c.xy_pts
        lg, sh, ang = pc.min_rect(tp)
        cx, cy = float(np.mean(tp[:, 0])), float(np.mean(tp[:, 1]))
        try:
            import cv2
            pts = np.round(tp * 1000).astype(np.int32).reshape(-1, 1, 2)
            (rx, ry), _, _ = cv2.minAreaRect(pts)
            cx, cy = rx / 1000.0, ry / 1000.0
        except Exception:
            pass
        cur = self.robot.closing_yaw()
        short_dir = ang + math.pi / 2
        options = [short_dir, short_dir + math.pi]
        if lg - sh < 0.012:
            options += [ang, ang + math.pi]
        best = min(options, key=lambda a: abs(geom.wrap(a - cur)))
        dyaw = geom.wrap(best - cur)
        return {"xy": np.array([cx, cy]), "top": float(c.top[2]), "long": lg, "short": sh,
                "dyaw": dyaw, "h": c.hmax}

    def pick_bin(self, comp, label, keep=True):
        r = self.robot
        g = self.grasp_plan(comp)
        self.log(f"PICK {label}: plan xy={np.round(g['xy'], 3).tolist()} top={g['top']:.3f} dims=({g['long']:.3f},{g['short']:.3f}) dyaw={g['dyaw']:.2f} step={r.step_id}")
        cyaw0 = r.closing_yaw(self.R0)
        R_des = geom.rotz(geom.wrap(r.closing_yaw() + g["dyaw"] - cyaw0)) @ self.R0
        z_tr = max(self.scene.z_table + self.obj_top + TRAVEL_CLEAR, g["top"] + 0.08)
        z_tr = min(z_tr, max(self.home[2], self.scene.z_table + 0.25))
        x, y = g["xy"]
        depth = min(GRASP_DEPTH, 0.5 * max(g["h"], 0.02))
        attempts = 0
        offsets = [(0.0, 0.0, 0.0), (0.0, 0.0, -0.01), (0.0, 0.0, 0.008)]
        while attempts < 3 and not r.ended and r.steps_left() > 25:
            ox, oy, oz = offsets[attempts]
            attempts += 1
            m = r.move_to([x + ox, y + oy, z_tr], R_des, grip=-1.0, speed=SPEED_FREE, tol=0.01, rot_tol=0.05,
                          max_steps=60, on_frames=None)
            self.log(f"  above bin: {m} pos={np.round(r.pos, 3).tolist()} yaw={r.closing_yaw():.2f} step={r.step_id}")
            m = r.move_to([x + ox, y + oy, g["top"] + 0.03], R_des, grip=-1.0, speed=SPEED_NEAR, tol=0.006,
                          rot_tol=0.04, max_steps=30)
            zg = g["top"] - depth + oz
            m = r.move_to([x + ox, y + oy, zg], R_des, grip=-1.0, speed=0.15, tol=0.005, rot_tol=0.05,
                          max_steps=24, max_chunk=3, stall_check=True)
            self.log(f"  at grasp: {m} pos={np.round(r.pos, 3).tolist()} target_z={zg:.3f} fingers={r.finger_sum:.4f} step={r.step_id}")
            r.set_grip(1.0, n=10)
            fs = r.finger_sum
            self.log(f"  closed: fingers={fs:.4f} step={r.step_id}")
            m = r.move_to([r.pos[0], r.pos[1], g["top"] + 0.13], R_des, grip=1.0, speed=0.3, tol=0.01,
                          max_steps=30)
            fs2 = r.finger_sum
            self.log(f"  lifted: {m} pos={np.round(r.pos, 3).tolist()} fingers={fs2:.4f} complete={r.complete} ended={r.ended} step={r.step_id}")
            if fs2 > 0.004:
                return True
            # missed: open and retry
            self.log("  grasp miss -> retry")
            r.set_grip(-1.0, n=6)
            if r.ended:
                break
        return False

    def place_back(self, comp):
        r = self.robot
        g_top = float(comp.top[2])
        m = r.move_to([r.pos[0], r.pos[1], g_top + 0.005 - GRASP_DEPTH + 0.002], None, grip=1.0, speed=0.15,
                      tol=0.006, max_steps=30, stall_check=True)
        r.set_grip(-1.0, n=8)
        r.move_to([r.pos[0], r.pos[1], g_top + 0.12], None, grip=-1.0, speed=0.3, tol=0.01, max_steps=25)
        self.log(f"  placed back step={r.step_id}")

    def pick(self):
        r = self.robot
        for i, (col, j, why) in enumerate(self.decisions):
            if r.ended:
                return
            if j is None or j >= len(self.final_dets):
                self.log(f"PICK {col}: no bin ({why})")
                continue
            comp = self.final_dets[j]
            ok = self.pick_bin(comp, f"{col} via {why}")
            self.log(f"PICK {col} result grasp={ok} complete={r.complete} ended={r.ended}")
            if i < len(self.decisions) - 1 and not r.ended:
                self.place_back(comp)

    def hold_end(self):
        r = self.robot
        while not r.ended and r.steps_left() > 0 and self.wall() < WALL_LIMIT:
            n = min(10, r.steps_left())
            r.hold(n)
        self.log(f"END step={r.step_id} ended={r.ended} complete={r.complete} fingers={r.finger_sum if r.state else None}")

    # ------------------------------------------------------------ main
    def run(self):
        phases = [("init", self.init), ("prewatch", self.prewatch), ("press", self.press), ("watch", self.watch), ("decide", self.decide),
                  ("pick", self.pick), ("hold", self.hold_end)]
        init_ok = False
        for name, fn in phases:
            if self.robot.ended:
                break
            if name != "init" and not init_ok:
                self.log(f"skip {name}: init failed")
                continue
            t = time.monotonic()
            s0 = self.robot.step_id
            try:
                fn()
                if name == "init":
                    init_ok = True
            except EpisodeEnded:
                self.log(f"episode ended during {name}")
                break
            except Exception as ex:
                self.log(f"PHASE {name} EXC {ex!r}\n{traceback.format_exc()}")
            self.log(f"PHASE {name} done steps {s0}->{self.robot.step_id} wall {time.monotonic() - t:.1f}s calls={self.robot.n_calls}")
        if self.robot.step_wall:
            self.log(f"step call wall: mean={np.mean(self.robot.step_wall):.3f}s max={np.max(self.robot.step_wall):.3f}s n={len(self.robot.step_wall)}")
        try:
            summ = {"version": VERSION, "instruction": self.instr, "targets": self.targets,
                    "button": None if self.button is None else {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in self.button.items()},
                    "press_done_step": self.press_done_step, "decisions": [list(map(str, d)) for d in self.decisions],
                    "pre": self.pre, "post": self.post, "final_step": self.robot.step_id, "ended": self.robot.ended,
                    "complete": self.robot.complete, "wall": self.wall(), "n_frames": len(self.robot.frames),
                    "video_motion": self.video_motion, "obs_fallback": self.n_obs_fallback}
            with open(os.path.join(OUT, "bums_summary.json"), "w") as f:
                json.dump(summ, f, indent=1, default=str)
        except Exception as ex:
            self.log(f"summary write failed {ex!r}")


def parse_points(rep):
    """Extract per-selection world points from a pixel_to_position reply (format unknown)."""
    lst = None
    for key in ("results", "points", "selections", "positions", "world_points"):
        v = iu.find_key(rep, key, lambda v: isinstance(v, list) and v)
        if v is not None:
            lst = v
            break
    if lst is None:
        for path, k, v in iu.walk(rep):
            if isinstance(v, list) and v and isinstance(v[0], dict):
                lst = v
                break
    out = []
    if lst is None:
        return out
    for e in lst:
        p = None
        if isinstance(e, dict):
            if e.get("valid") is False:
                out.append(None)
                continue
            for key in ("position", "world", "world_position", "point", "xyz", "position_m", "world_point"):
                if iu.is_num_list(e.get(key), 3):
                    p = [float(x) for x in e[key]]
                    break
            if p is None and all(k in e for k in ("x", "y", "z")):
                p = [float(e["x"]), float(e["y"]), float(e["z"])]
            if p is None:
                keys = sorted(e.keys(), key=lambda k: (0 if "world" in str(k).lower() else 1))
                for k in keys:
                    v = e[k]
                    kl = str(k).lower()
                    if iu.is_num_list(v, 3) and "pixel" not in kl and "uv" not in kl and "camera" not in kl:
                        p = [float(x) for x in v]
                        break
        elif iu.is_num_list(e, 3):
            p = [float(x) for x in e]
        out.append(p)
    return out


def save_debug(last, frames, hist, comps, decisions, scene):
    import cv2
    from PIL import Image
    os.makedirs(OUT, exist_ok=True)
    # top-down plot of track paths
    img = np.full((300, 300, 3), 255, np.uint8)
    xmin, xmax, ymin, ymax = scene.roi
    sx = 299 / max(1e-3, xmax - xmin)
    sy = 299 / max(1e-3, ymax - ymin)
    cols = [(255, 0, 0), (0, 160, 0), (0, 0, 255), (200, 120, 0), (150, 0, 150), (0, 150, 150)]
    for j in range(hist.shape[0]):
        c = cols[j % len(cols)]
        for k in range(hist.shape[1] - 1):
            p1 = (int((hist[j, k, 1] - ymin) * sy), int((hist[j, k, 0] - xmin) * sx))
            p2 = (int((hist[j, k + 1, 1] - ymin) * sy), int((hist[j, k + 1, 0] - xmin) * sx))
            cv2.line(img, p1, p2, c, 1)
        cv2.putText(img, str(j), (int((hist[j, -1, 1] - ymin) * sy), int((hist[j, -1, 0] - xmin) * sx)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    Image.fromarray(img).save(os.path.join(OUT, "tracks_topdown.png"))


def main(ctx):
    try:
        Controller(ctx).run()
    except BaseException as ex:  # never raise out of main
        try:
            print(f"[{VERSION}] FATAL {ex!r}\n{traceback.format_exc()}", flush=True)
        except Exception:
            pass
