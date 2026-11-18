"""ButtonUnmask controller (RoboMME-Plus, RAX). Entry point: main(ctx).

Plan: perceive at step 0 (colour cubes, button), press the button with the closed
gripper well before step 90, keep tracking the cubes while they get hidden, wait
until the scene is static, then grasp and lift the container nearest the last
seen position of each instructed cube colour. Never raises.
"""
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geom import Rz, wrap_pi  # noqa: E402
from motion import EpisodeOver, Robot, parse_limit  # noqa: E402
from obs import WS, load_depth, load_rgb, reply_meta, resolve_path, summarize  # noqa: E402
from percept import (HeightMap, Projector, Tracker, calibrate, cloud, color_blobs, dist_to_obj,  # noqa: E402
                     estimate_table, obj_brief, parse_targets, top_face)

OUT = Path(os.environ.get("BU_OUT", "/workspace/output"))
VERSION = "v1"
WALL_LIMIT = 1050.0  # seconds; controller hard limit is 1200


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.t0 = time.time()
        try:
            OUT.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self.logf = None
        try:
            self.logf = open(OUT / "controller_log.txt", "a")
        except Exception:
            self.logf = None
        self.robot = Robot(ctx, self.log)
        self.summary = {"version": VERSION, "phases": []}
        self.calib = None
        self.table = 0.0
        self.tracker = None
        self.objs0 = []
        self.button = None
        self.p0 = None
        self.R0 = None
        self.press_step = None
        self.targets = []
        self.saved = 0

    # ------------------------------------------------------------ logging
    def log(self, msg):
        line = f"[{time.time() - self.t0:7.1f}s step {getattr(getattr(self, 'robot', None), 'steps', '?')}] {msg}"
        print(line, flush=True)
        if self.logf:
            try:
                self.logf.write(line + "\n")
                self.logf.flush()
            except Exception:
                pass

    def phase(self, name):
        self.summary["phases"].append(dict(name=name, step=self.robot.steps, wall=round(time.time() - self.t0, 1)))
        self.log(f"=== PHASE {name}")

    def save_img(self, name, obs=None):
        if self.saved > 30:
            return
        obs = obs or self.robot.obs
        try:
            from PIL import Image
            for cam, a in (obs.rgb or {}).items():
                Image.fromarray(a).save(OUT / f"{self.saved:02d}_{name}_{cam}.png")
            self.saved += 1
        except Exception as e:
            self.log(f"save_img failed {e}")

    def annotate(self, name, obs, objs=(), blobs=(), marks=()):
        """Front image with object ids, blobs and marks (list of (xyz, label)) for offline review."""
        try:
            from PIL import Image, ImageDraw
            a = obs.rgb.get("front")
            cam = obs.cams.get("front")
            conv = (self.calib.get("front") or {}).get("conv")
            if a is None or cam is None or conv is None:
                return
            im = Image.fromarray(a).resize((512, 512), Image.NEAREST)
            dr = ImageDraw.Draw(im)
            for b in blobs:
                if b.get("cam") != "front":
                    continue
                u, v = b["uv"]
                dr.ellipse([2 * u - 6, 2 * v - 6, 2 * u + 6, 2 * v + 6], outline=(255, 255, 255))
                dr.text((2 * u + 7, 2 * v - 6), b["color"][:3], fill=(255, 255, 255))
            for x in objs:
                c = np.array([[x["center"][0], x["center"][1], x["top"]]])
                u, v = cam.project(c, conv)[0] * 2
                col = (255, 255, 0) if not x["tall"] else (128, 128, 128)
                dr.rectangle([u - 3, v - 3, u + 3, v + 3], outline=col)
                dr.text((u + 4, v + 2), str(x["id"]), fill=col)
            for xyz, lab in marks:
                u, v = cam.project(np.array([xyz], float), conv)[0] * 2
                dr.line([u - 8, v, u + 8, v], fill=(255, 0, 255), width=2)
                dr.line([u, v - 8, u, v + 8], fill=(255, 0, 255), width=2)
                dr.text((u + 9, v - 12), lab, fill=(255, 0, 255))
            im.save(OUT / f"ann_{name}.png")
        except Exception as e:
            self.log(f"annotate failed {e}")

    def save_summary(self):
        try:
            self.summary["wall"] = round(time.time() - self.t0, 1)
            self.summary["steps"] = self.robot.steps
            self.summary["ended"] = self.robot.ended
            self.summary["complete"] = self.robot.complete
            with open(OUT / "summary.json", "w") as f:
                json.dump(_jsonable(self.summary), f, indent=1, default=str)
        except Exception as e:
            self.log(f"summary save failed {e}")

    def wall_left(self):
        return WALL_LIMIT - (time.time() - self.t0)

    # ------------------------------------------------------------ perception
    def perceive(self, use_wrist=True, stride_front=1, stride_wrist=2):
        o = self.robot.observe(images=True)
        proj = Projector(self.ctx, o, self.calib, self.log)
        tcp = self.robot.pos()
        P, C = [], []
        Pf, _, Cf = cloud(proj, "front", stride=stride_front, tcp=tcp)
        P.append(Pf)
        C.append(Cf)
        if use_wrist and self._wrist_ok():
            Pw, _, Cw = cloud(proj, "wrist", stride=stride_wrist, tcp=tcp)
            P.append(Pw)
            C.append(Cw)
        P = np.concatenate(P) if P else np.zeros((0, 3))
        C = np.concatenate(C) if C else np.zeros((0, 3), np.uint8)
        return o, proj, tcp, P, C

    def cell(self):
        return 0.005 if (self.calib.get("front") or {}).get("conv") is not None else 0.008

    def _wrist_ok(self):
        c = (self.calib or {}).get("wrist") or {}
        return c.get("conv") is not None and (c.get("err") is not None and c["err"] < 0.012)

    def track(self, o=None, source="live", wrist=False):
        try:
            o = o or self.robot.observe(images=True)
            proj = Projector(self.ctx, o, self.calib, self.log)
            tcp = self.robot.pos()
            bl = color_blobs(proj, "front", self.table, tcp)
            if wrist and self._wrist_ok():
                bl += color_blobs(proj, "wrist", self.table, tcp, min_px=15)
            self.tracker.add(bl, self.robot.steps, source)
            return o, bl
        except EpisodeOver:
            raise
        except Exception as e:
            self.log(f"track failed: {e}")
            return o, []

    # ------------------------------------------------------------ main
    def run(self):
        R = self.robot
        self.phase("init")
        try:
            st = self.ctx.call("robomme_status")
            meta = reply_meta(st)
            self.log("status: " + json.dumps(summarize(meta), default=str)[:3000])
            R.limit = parse_limit(meta)
        except Exception as e:
            self.log(f"robomme_status failed: {e}")
        o = R.observe(images=True)
        self.log("observe meta: " + json.dumps(summarize(o.meta), default=str)[:4000])
        if R.limit is None:
            R.limit = parse_limit(o.meta)
        self.log(f"step limit={R.limit} instruction={o.instruction!r} state={None if o.state is None else np.round(o.state, 4).tolist()}")
        self.log(f"rgb={ {k: v.shape for k, v in o.rgb.items()} } depth={ {k: v.shape for k, v in o.depth.items()} } cams={list(o.cams)}")
        self.summary["instruction"] = o.instruction
        self.targets = parse_targets(o.instruction)
        self.log(f"targets={self.targets}")
        self.summary["targets"] = self.targets
        if o.state is None:
            self.log("no state parsed; aborting motion")
            return
        self.p0 = R.pos()
        self.R0 = R.R()
        try:
            zt = self.R0[:, 2]
            ang = math.acos(max(-1.0, min(1.0, float(-zt[2]))))
            self.log(f"initial tool z-axis={np.round(zt, 3).tolist()} tilt from down={math.degrees(ang):.1f} deg; finger axis={np.round(self.R0[:, 1], 3).tolist()}")
            if math.radians(4) < ang < math.radians(30):
                ax = np.cross(zt, [0, 0, -1.0])
                ax = ax / (np.linalg.norm(ax) or 1)
                from geom import axang_to_R
                self.R0 = axang_to_R(ax * ang) @ self.R0
                self.log("levelled R0 to point straight down")
        except Exception:
            pass
        self.save_img("start", o)

        self.phase("calibrate")
        self.calib = calibrate(self.ctx, o, self.log)
        self.summary["calib"] = {k: dict(conv=v.get("conv"), err=v.get("err"), tool=v.get("tool_ok")) for k, v in self.calib.items()}

        self.phase("perceive0")
        try:
            self.perceive0()
        except EpisodeOver:
            raise
        except Exception:
            self.log("perceive0 failed: " + traceback.format_exc()[-2000:])
            if self.tracker is None:
                self.tracker = Tracker(self.table)
            self.objs0 = getattr(self, "objs0", None) or []
        self.phase("press")
        try:
            self.press()
        except EpisodeOver:
            raise
        except Exception:
            self.log("press failed: " + traceback.format_exc()[-1500:])
        self.after_press()

    def perceive0(self):
        o, proj, tcp, P, C = self.perceive(use_wrist=True)
        Pf = cloud(proj, "front", stride=2, tcp=tcp)[0]
        self.table = estimate_table(Pf if len(Pf) > 100 else P)
        self.log(f"table z={self.table:.4f}  npts={len(P)}")
        self.tracker = Tracker(self.table)
        hm = HeightMap(P, C, self.table, cell=self.cell(), fill_holes=self.cell() > 0.006)
        objs = hm.objects()
        self.log("objects0: " + json.dumps(_jsonable([obj_brief(x) for x in objs]))[:3000])
        blobs = color_blobs(proj, "front", self.table, tcp)
        if self._wrist_ok():
            blobs += color_blobs(proj, "wrist", self.table, tcp, min_px=15)
        self.log("blobs0: " + json.dumps(_jsonable([{k: b[k] for k in ("color", "xy", "ztop", "npx", "ext", "cam")} for b in blobs]))[:2000])
        self.summary["objects0"] = [obj_brief(x) for x in objs]

        self.button = self.identify_button(objs, blobs)
        if self.button is not None:
            b = self.button
            self.tracker.exclude.append((np.asarray(b["center"]), max(b["L"], b["W"]) / 2 + 0.015))
            self.summary["button"] = obj_brief(b)
        self.tracker.add(blobs, 0, "live")
        self.annotate("step0", o, objs, blobs, [((self.button["top_center"][0], self.button["top_center"][1], self.button["top"]), "BUTTON")] if self.button else [])
        self.process_video(o)

        self.objs0 = objs
        self.hm0 = hm

    def after_press(self):
        R = self.robot
        self.phase("wait")
        try:
            self.wait_cover()
        except EpisodeOver:
            raise
        except Exception:
            self.log("wait failed: " + traceback.format_exc()[-1500:])
        if not self.targets:
            seen = [c for c in self.tracker.sight if self.tracker.last(c) is not None]
            self.log(f"no colour in instruction; colours seen={seen}")
            if len(seen) == 1:
                self.targets = seen

        for i, color in enumerate(self.targets or [None]):
            self.phase(f"pick_{color}")
            ok = False
            for _ in range(2):
                try:
                    ok = self.pick(color, last=(i == len(self.targets or [None]) - 1))
                    break
                except EpisodeOver:
                    raise
                except Exception:
                    self.log("pick raised: " + traceback.format_exc()[-1500:])
            self.log(f"pick {color} -> {ok}")
            if R.ended:
                break

        self.phase("hold")
        try:
            for _ in range(4):
                if R.remaining() <= 0 or self.wall_left() < 20:
                    break
                R.hold(5, grip=R.grip)
        except EpisodeOver:
            pass
        self.log(f"done: steps={R.steps} ended={R.ended} complete={R.complete}")

    # ------------------------------------------------------------ video
    def process_video(self, o):
        mp = o.manifest_path
        if not mp:
            self.log("no frame_sequence manifest")
            return
        try:
            p = resolve_path(mp)
            if p is None:
                self.log(f"manifest not found {mp}")
                return
            man = json.loads(p.read_text())
            frames = [f for f in man.get("frames", []) if f.get("phase") == "video_demo"]
            self.log(f"manifest frames={len(man.get('frames', []))} video_demo={len(frames)} sampling={man.get('video_sampling')}")
            self.summary["video_frames"] = len(frames)
            if not frames or "front" not in o.cams or (self.calib.get("front") or {}).get("conv") is None:
                return
            frames.sort(key=lambda f: f.get("frame_index", 0))
            idx = sorted(set(np.linspace(0, len(frames) - 1, min(len(frames), 60)).astype(int).tolist()
                             + list(range(max(0, len(frames) - 8), len(frames)))))
            from obs import Obs
            n_ok = 0
            for i in idx:
                f = frames[i]
                files = {(x.get("camera"), x.get("kind")): x for x in f.get("files", [])}
                fr, fd = files.get(("front", "rgb")), files.get(("front", "depth"))
                if not fr or not fd:
                    continue
                pr, pd = resolve_path(fr["path"], [p.parent]), resolve_path(fd["path"], [p.parent])
                if pr is None or pd is None:
                    continue
                vo = Obs()
                vo.rgb = {"front": load_rgb(pr)}
                vo.depth = {"front": load_depth(pd, float(fd.get("meters_per_unit", 0.001) or 0.001))}
                vo.cams = {"front": o.cams["front"]}
                vo.step_id = 0
                proj = Projector(self.ctx, vo, {"front": self.calib["front"]}, self.log)
                st = f.get("state")
                tcp = np.array(st[:3]) if isinstance(st, list) and len(st) >= 3 else None
                bl = color_blobs(proj, "front", self.table, tcp)
                self.tracker.add(bl, -100000 + int(f.get("frame_index", i)), "video")
                n_ok += 1
            for c in self.targets:
                self.log(f"video last {c}: {self._brief_sight(self.tracker.last(c, 'video'))}")
            self.log(f"video frames processed={n_ok}")
        except Exception:
            self.log("video processing failed: " + traceback.format_exc()[-800:])

    @staticmethod
    def _brief_sight(s):
        if not s:
            return None
        return dict(xy=[round(float(a), 3) for a in s["xy"]], step=s["step"], npx=s["npx"], cam=s["cam"], n=s.get("n_sight"))

    # ------------------------------------------------------------ button
    def is_cube(self, ob):
        big = max(ob["L"], ob["W"])
        return (big <= 0.07 and ob["dom"] is not None and ob["domfrac"] >= 0.45
                and 0.4 <= ob["h"] / max(ob["W"], 1e-3) <= 2.0 and ob["h"] <= 0.07)

    def identify_button(self, objs, blobs):
        cands = [x for x in objs if not x["tall"] and x["n"] * self.cell() ** 2 >= 3e-4 and x["h"] < 0.16]
        if not cands:
            cands = [x for x in objs if not x["tall"] and x["n"] * self.cell() ** 2 >= 1e-4 and x["h"] < 0.16]
        if not cands:
            self.log("identify_button: no candidates")
            return None
        non = [x for x in cands if not self.is_cube(x)]
        self.log(f"button cands={[x['id'] for x in cands]} noncube={[x['id'] for x in non]}")
        pool = non if non else cands
        if len(pool) == 1:
            b = pool[0]
        else:
            feats = np.array([[x["L"], x["W"], x["h"]] for x in pool])
            scores = []
            for i, x in enumerate(pool):
                d = np.linalg.norm(feats - feats[i], axis=1)
                d[i] = np.inf
                odd = float(d.min())
                flat = 1.0 - min(1.0, x["h"] / max(x["L"], 1e-3))
                scores.append(odd + 0.01 * flat)
            self.log(f"button scores={[(x['id'], round(s, 4)) for x, s in zip(pool, scores)]}")
            b = pool[int(np.argmax(scores))]
        self.log(f"button -> {obj_brief(b)}")
        return b

    # ------------------------------------------------------------ press
    def z_travel(self, objs=None):
        objs = objs if objs is not None else self.objs0
        tops = [x["top"] for x in objs if not x["tall"]]
        z = max([self.table + 0.12] + [t + 0.07 for t in tops])
        return min(z, self.table + 0.32)

    def press(self):
        R = self.robot
        b = self.button
        if b is None:
            self.log("no button; skipping press")
            return
        bxy = np.asarray(b["top_center"], float)
        btop = float(b["top"])
        zt = max(self.z_travel(), btop + 0.06)
        Rd = self.R0
        hook = lambda: self.track(wrist=False)
        R.move_to([bxy[0], bxy[1], zt], Rd, grip=1.0, vmax=0.03, tol=0.008, max_steps=40, chunk=3, hook=hook, label="to_button")
        R.move_to([bxy[0], bxy[1], btop + 0.035], Rd, grip=1.0, vmax=0.025, tol=0.004, max_steps=15, chunk=2, label="above_button")
        z_end = btop - 0.025
        zs = [R.pos()[2]]
        n = 0
        while n < 22 and R.steps < 86:
            p = R.pos()
            if p[2] <= z_end:
                break
            exy = bxy - p[:2]
            dz = -0.008
            row = [np.clip(exy[0], -0.01, 0.01) / 0.05, np.clip(exy[1], -0.01, 0.01) / 0.05, dz / 0.05, 0, 0, 0, 1.0]
            R.send([row])
            n += 1
            zs.append(R.pos()[2])
            if len(zs) >= 4 and zs[-1] < btop + 0.01 and zs[-3] - zs[-1] < 0.002:
                self.log(f"press stall at z={zs[-1]:.4f} (top {btop:.4f})")
                break
        R.hold(4, grip=1.0, dz=-0.004)
        self.press_step = R.steps
        self.summary["press"] = dict(step=R.steps, z_min=float(min(zs)), btop=btop, z_end=z_end)
        self.log(f"pressed: step={R.steps} zmin={min(zs):.4f} btop={btop:.4f}")
        R.move_to([bxy[0], bxy[1], btop + 0.07], Rd, grip=1.0, vmax=0.025, tol=0.006, max_steps=10, chunk=2, label="leave_button")

    # ------------------------------------------------------------ wait
    def wait_cover(self):
        R = self.robot
        hook = lambda: self.track(wrist=False)
        try:
            R.move_to(self.p0, self.R0, grip=-1.0, vmax=0.03, tol=0.01, max_steps=30, chunk=3, hook=hook, label="to_observe")
        except EpisodeOver:
            raise
        except Exception:
            self.log("move to observe failed: " + traceback.format_exc()[-600:])
        start = R.steps
        min_until = (self.press_step or R.steps) + 20
        prev = None
        static = 0
        changes = []
        max_wait = 100
        while R.steps - start < max_wait and R.remaining() > 70 and self.wall_left() > 300:
            R.hold(5, grip=-1.0)
            o, bl = self.track(wrist=True)
            d = None
            if o is not None:
                d = o.depth.get("front")
                if d is None and o.rgb.get("front") is not None:
                    d = o.rgb["front"].astype(np.float32) / 1000.0  # 0.01 ~ 10 grey levels
            frac = None
            if d is not None and prev is not None and d.shape == prev.shape:
                diff = np.abs(d - prev)
                if diff.ndim == 3:
                    diff = diff.max(-1) / 3.0
                frac = float((diff > 0.01).mean())
                changes.append((R.steps, round(frac, 4)))
                static = static + 1 if frac < 0.003 else 0
            prev = d
            vis = [c for c in self.targets if self.tracker.visible_now(c, R.steps)]
            seen = all(self.tracker.last(c) is not None for c in self.targets)
            if R.steps >= min_until and static >= 2 and not vis and seen:
                break
            if R.steps >= min_until + 40 and static >= 2 and (seen or R.steps >= min_until + 75):
                break
        self.summary["wait_changes"] = changes
        self.log(f"wait done at step {R.steps}; changes={changes}")
        for c in self.targets:
            self.log(f"last sighting {c}: live={self._brief_sight(self.tracker.last(c, 'live'))} video={self._brief_sight(self.tracker.last(c, 'video'))}")
        self.save_img("after_wait")

    # ------------------------------------------------------------ pick
    def target_xy(self, color):
        if color is None:
            return None
        s = self.tracker.last(color, "live") or self.tracker.last(color, "video")
        return None if s is None else np.asarray(s["xy"], float)

    def choose_container(self, objs, color):
        cxy = self.target_xy(color)
        bxy = np.asarray(self.button["center"]) if self.button is not None else None
        cands = []
        for x in objs:
            if x["tall"] or x["h"] < 0.015 or x["h"] > 0.2:
                continue
            if bxy is not None and np.linalg.norm(np.asarray(x["center"]) - bxy) < max(0.04, 0.5 * max(self.button["L"], self.button["W"])):
                continue
            cands.append(x)
        self.log(f"container cands: {[obj_brief(x) for x in cands]}"[:2500])
        if not cands:
            return None, cxy
        if cxy is None:
            self.log(f"no sighting for {color}; guessing")
            non = [x for x in cands if not self.is_cube(x)] or cands
            return max(non, key=lambda x: x["h"]), None
        ds = [dist_to_obj(x, cxy) for x in cands]
        order = np.argsort(ds)
        best = cands[int(order[0])]
        # if the nearest object is the (still visible) target cube itself, prefer a covering container
        if self.is_cube(best) and best["dom"] == color:
            alt = [cands[i] for i in order[1:] if ds[i] < 0.06 and not self.is_cube(cands[i])]
            if alt:
                best = alt[0]
            else:
                self.log("WARNING target cube appears uncovered; picking nearest object anyway")
        self.log(f"choose {color} at {np.round(cxy, 3).tolist()}: id={best['id']} dist={min(ds):.3f} all={[round(d, 3) for d in ds]}")
        return best, cxy

    def yaw_for(self, c, L, W, ang, objs, ob=None):
        axis_w = ang + math.pi / 2  # closing direction across the narrow side
        if L / max(W, 1e-3) < 1.2:
            pts = [np.asarray(x["center"]) for x in objs if x is not ob and not x["tall"]
                   and np.linalg.norm(np.asarray(x["center"]) - c) > 0.04]
            for col in list(self.tracker.sight):
                sgt = self.tracker.last(col)
                if sgt is not None and np.linalg.norm(sgt["xy"] - c) > 0.04:
                    pts.append(np.asarray(sgt["xy"]))
            if pts:
                nb = min(pts, key=lambda q: np.linalg.norm(q - c))
                u = nb - c
                if np.linalg.norm(u) < 0.2:
                    au = math.atan2(u[1], u[0])
                    # finger axis should be perpendicular to the neighbour direction
                    best = min((axis_w, ang), key=lambda a: abs(math.cos(a - au)))
                    axis_w = best
        f0 = self.R0[:, 1]
        a0 = math.atan2(f0[1], f0[0])
        d = wrap_pi(axis_w - a0)
        if d > math.pi / 2:
            d -= math.pi
        if d < -math.pi / 2:
            d += math.pi
        Rd = Rz(d) @ self.R0
        return Rd, d

    def grasp_plan(self, ob, objs, hm=None, hint=None):
        c = np.asarray(ob["top_center"] if ob["toparea"] > 0.0004 else ob["center"], float)
        L, W, ang = ob["L"], ob["W"], ob["ang"]
        if ob["toparea"] > 0.0006 and ob["tW"] > 0.01:
            L, W = max(ob["tL"], ob["tW"]) + 0.005, min(ob["tL"], ob["tW"]) + 0.005
        top = float(ob["top"])
        hint = None if hint is None else np.asarray(hint, float)
        if hint is not None and (L > 0.11 or W > 0.09 or np.linalg.norm(c - hint) > 0.03):
            self.log(f"object footprint implausible/merged (L={L:.3f} W={W:.3f}); centring on cube sighting")
            c = hint.copy()
        if hm is not None:
            tf = top_face(hm, hint if hint is not None else c)
            if tf is not None:
                self.log(f"top face: c={np.round(tf['center'], 4).tolist()} L={tf['L']:.3f} W={tf['W']:.3f} ang={tf['ang']:.2f} top={tf['top']:.4f} area={tf['area']:.4f}")
            ok = (tf is not None and tf["area"] > 4e-4 and tf["W"] > 0.015)
            if ok and tf["L"] <= 0.11 and tf["W"] <= 0.09 and (hint is None or np.linalg.norm(tf["center"] - hint) < 0.03):
                c, L, W, ang, top = np.asarray(tf["center"]), tf["L"], tf["W"], tf["ang"], tf["top"]
            elif ok:
                # merged with a neighbour: keep the cube-centred estimate, take height and orientation only
                self.log("top face merged/offset: using its height only")
                top = tf["top"]
                L = W = 0.07
                if hint is not None:
                    c = hint.copy()
            else:
                self.log("top face unavailable")
        Rd, d = self.yaw_for(c, L, W, ang, objs, ob)
        h = top - self.table
        g = float(np.clip(0.4 * h, 0.012, 0.03))
        zg = max(self.table + 0.014, top - g)
        return dict(c=c, top=top, zg=zg, Rd=Rd, yaw=d, W=W, L=L, ang=ang)

    def refine_wrist(self, plan):
        try:
            o, proj, tcp, _, _ = self.perceive(use_wrist=False)
            mode = proj.mode("wrist")
            if mode is None:
                return plan
            Pw, _, Cw = cloud(proj, "wrist", stride=1, tcp=tcp)
            if len(Pw) < 200:
                return plan
            c = plan["c"]
            cellw = 0.004 if mode == "local" else 0.006
            hm = HeightMap(Pw, Cw, self.table, cell=cellw, bounds=(c - 0.15, c + 0.15), fill_holes=(mode != "local"))
            tf = top_face(hm, c, radius=0.04)
            if tf is None or tf["area"] < 3e-4:
                self.log("wrist refine: nothing near")
                return plan
            nc = np.asarray(tf["center"], float)
            self.log(f"wrist refine: shift={np.round(nc - c, 4).tolist()} top {plan['top']:.4f}->{tf['top']:.4f} dims {tf['L']:.3f}x{tf['W']:.3f} ang={tf['ang']:.2f}")
            if np.linalg.norm(nc - c) > 0.035 or abs(tf["top"] - plan["top"]) > 0.03:
                self.log("wrist refine: inconsistent; keeping front estimate")
                return plan
            plan = dict(plan)
            plan["c"] = 0.3 * c + 0.7 * nc
            if tf["L"] <= 0.1 and tf["W"] <= 0.09 and np.linalg.norm(nc - c) < 0.025:
                Rd, d = self.yaw_for(plan["c"], tf["L"], tf["W"], tf["ang"], getattr(self, "pick_objs", []))
                self.log(f"wrist refine: yaw {plan['yaw']:.3f} -> {d:.3f}")
                plan["Rd"], plan["yaw"], plan["W"], plan["L"] = Rd, d, tf["W"], tf["L"]
            newtop = 0.5 * (tf["top"] + plan["top"])
            plan["zg"] += newtop - plan["top"]
            plan["top"] = newtop
            return plan
        except EpisodeOver:
            raise
        except Exception:
            self.log("wrist refine failed: " + traceback.format_exc()[-600:])
            return plan

    def pick(self, color, last=True):
        R = self.robot
        for attempt in range(3):
            if R.remaining() < 30 or self.wall_left() < 120:
                self.log("not enough budget for pick")
                return False
            try:
                R.move_to([self.p0[0], self.p0[1], max(self.p0[2], R.pos()[2])], self.R0, grip=-1.0, vmax=0.03,
                          tol=0.02, max_steps=12, chunk=3, label="clear")
            except EpisodeOver:
                raise
            o, proj, tcp, P, C = self.perceive(use_wrist=True)
            hm = HeightMap(P, C, self.table, cell=self.cell(), fill_holes=self.cell() > 0.006)
            objs = hm.objects()
            self.log(f"objects(pick {attempt}): " + json.dumps(_jsonable([obj_brief(x) for x in objs]))[:2500])
            ob, cxy = self.choose_container(objs, color)
            if ob is None:
                self.log("no container found")
                return False
            self.pick_objs = objs
            plan = self.grasp_plan(ob, objs, hm, cxy)
            marks = [((plan["c"][0], plan["c"][1], plan["top"]), "GRASP")]
            if cxy is not None:
                marks.append(((cxy[0], cxy[1], self.table + 0.02), f"last_{color}"))
            self.annotate(f"pick_{color}_{attempt}", o, objs, [], marks)
            self.log(f"plan: c={np.round(plan['c'], 4).tolist()} top={plan['top']:.4f} zg={plan['zg']:.4f} yaw={plan['yaw']:.3f} W={plan['W']:.3f} L={plan['L']:.3f}")
            self.summary.setdefault("picks", []).append(dict(color=color, attempt=attempt, target_xy=cxy, obj=obj_brief(ob),
                                                              plan={k: v for k, v in plan.items() if k != "Rd"}))
            zt = max(self.z_travel(objs), plan["top"] + 0.10)
            c = plan["c"]
            R.move_to([c[0], c[1], zt], plan["Rd"], grip=-1.0, vmax=0.03, tol=0.006, max_steps=40, chunk=3, label="to_container")
            if self._wrist_ok() or (self.calib.get("wrist") or {}).get("tool_ok"):
                plan = self.refine_wrist(plan)
                c = plan["c"]
            R.move_to([c[0], c[1], plan["top"] + 0.045], plan["Rd"], grip=-1.0, vmax=0.02, tol=0.003, max_steps=15, chunk=2, label="pre_grasp")
            R.move_to([c[0], c[1], plan["zg"]], plan["Rd"], grip=-1.0, vmax=0.01, tol=0.003, max_steps=16, chunk=2, label="descend")
            R.hold(12, grip=1.0)
            op = R.opening()
            self.log(f"after close opening={op:.4f} pos={np.round(R.pos(), 4).tolist()}")
            if op < 0.003:
                self.log("grasp missed (fingers closed)")
                R.hold(6, grip=-1.0)
                R.move_to([c[0], c[1], plan["top"] + 0.08], plan["Rd"], grip=-1.0, vmax=0.02, tol=0.008, max_steps=10, label="retreat")
                continue
            R.move_to([c[0], c[1], plan["zg"] + 0.15], plan["Rd"], grip=1.0, vmax=0.015, tol=0.006, max_steps=20, chunk=2, label="lift")
            op = R.opening()
            self.log(f"after lift opening={op:.4f} pos={np.round(R.pos(), 4).tolist()} complete={R.complete}")
            self.save_img(f"lift_{color}")
            if op < 0.003:
                self.log("lost object during lift")
                R.hold(4, grip=-1.0)
                continue
            if not last:
                # put the container back where it was and release before the next target
                R.move_to([c[0], c[1], plan["zg"] + 0.005], plan["Rd"], grip=1.0, vmax=0.015, tol=0.005, max_steps=20, chunk=2, label="place_back")
                R.hold(8, grip=-1.0)
                R.move_to([c[0], c[1], plan["top"] + 0.10], plan["Rd"], grip=-1.0, vmax=0.02, tol=0.008, max_steps=12, chunk=2, label="leave")
            return True
        return False


def main(ctx):
    C = Controller(ctx)
    try:
        C.run()
    except EpisodeOver:
        C.log("episode over")
    except Exception:
        C.log("FATAL " + traceback.format_exc()[-3000:])
    finally:
        C.save_summary()
        try:
            if C.logf:
                C.logf.close()
        except Exception:
            pass
