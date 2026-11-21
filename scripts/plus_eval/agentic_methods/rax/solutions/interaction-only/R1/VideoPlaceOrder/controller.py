"""VideoPlaceOrder controller (RAX, interaction-only, R=1) -- version v1.

Pipeline (each phase guarded; main never raises):
  P0 SENSE    observe step 0, read conditioning-video manifest, calibrate cameras
              (own back-projection checked against robomme_pixel_to_position).
  P1 VIDEO    per-frame robot state -> grasp/release segments -> ordered placements
              (EE position at each release). Fallback: colour tracking of the cube.
  P2 PLAN     parse the ordinal from the instruction ("third", "2nd", "last"...)
              -> target placement(s); no ordinal + order words -> replay all in order.
  P3 PICK     locate cube (front depth clusters, anchored at video positions),
              hover, wrist-depth refinement, optional yaw alignment, descend, close,
              verify finger width, lift.  Up to 3 attempts with re-perception.
  P4 PLACE    carry high, move over target xy, descend to the expert's release height,
              open, lift, verify cube position; re-pick if it is off target.
  P5 IDLE     hold a few zero rows so the native success check can register; return.

Transitions: P0->P1->P2->(P3->P4)*->P5.  P3 failure x3 -> P5.  Episode end -> stop.
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

from vpo_util import (WORKSPACE, CameraModel, all_variants, bfs_find, bfs_find_all, is_num_list,
                      load_png_depth, load_png_rgb, parse_intrinsics, parse_pixel_points, parse_pose,
                      per_camera, pil_to_depth_m, pil_to_rgb, reply_images, reply_meta, reply_texts,
                      resolve_path, wrap_quarter)
from vpo_percept import (choose_cube, color_rest_positions, estimate_table_z, extract_events,
                         find_clusters, parse_instruction, resolve_index)
from vpo_robot import EpisodeOver, Robot, _r

VERSION = "vpo-v1"
OUT_DIR = Path("/workspace/output")
WALL_BUDGET_S = 1050.0          # controller hard limit is 1200 s
CARRY_ABOVE_TABLE = 0.15        # m, carry height (plus TCP offset)
HOVER_ABOVE_TOP = 0.12          # m, hover above cube top for wrist look
PRE_GRASP = 0.05                # m above grasp height, slow final descent
RELEASE_MARGIN = 0.006          # m above the expert's release height
FAST = 0.022                    # m/step free-space speed
SLOW = 0.010                    # m/step near contact
MAX_PICK_ATTEMPTS = 3


class Log:
    def __init__(self):
        self.t0 = time.time()
        self.f = None
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            self.f = open(OUT_DIR / "vpo_log.jsonl", "a")
        except Exception:
            self.f = None

    def ev(self, kind, **kw):
        rec = {"t": round(time.time() - self.t0, 2), "ev": kind}
        rec.update(kw)
        try:
            line = json.dumps(rec, default=_jsonable)
        except Exception:
            line = json.dumps({"t": rec["t"], "ev": kind, "repr": repr(kw)[:2000]})
        print(line, flush=True)
        if self.f:
            try:
                self.f.write(line + "\n")
                self.f.flush()
            except Exception:
                pass


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


# =============================================================================
# Sensing helpers
# =============================================================================
class Scene:
    """Per-observation sensor access: images, depth, calibration."""

    def __init__(self, log):
        self.log = log
        self.variant = {}       # camera -> chosen back-projection variant
        self.calib_err = {}

    # ---- calibration ---------------------------------------------------------
    def cam_model(self, meta, cam):
        Kraw = per_camera(bfs_find(meta, "camera_intrinsics"), cam)
        if Kraw is None:
            Kraw = per_camera(bfs_find(meta, "intrinsics"), cam)
        Praw = per_camera(bfs_find(meta, "camera_poses"), cam)
        if Praw is None:
            Praw = per_camera(bfs_find(meta, "extrinsics"), cam)
        K = parse_intrinsics(Kraw)
        T, conv = parse_pose(Praw)
        if K is None or T is None:
            return None
        default = "cv" if ("cv" in (conv or "") and "gl" not in (conv or "")) else "gl"
        var = self.variant.get(cam)
        if var is None:
            var = {"conv": default, "off": 0.0, "transpose": False, "invert": False, "ray": False}
            if "world2cam" in (conv or "") or "extrinsic" in (conv or ""):
                var["invert"] = True
        return CameraModel(K, T, dict(var))

    # ---- depth / rgb of the current observation --------------------------------
    def frame_files(self, meta):
        """files of the latest frame in the reply's frame_sequence manifest."""
        seq = bfs_find(meta, "frame_sequence")
        mpath = None
        if isinstance(seq, dict):
            mpath = seq.get("manifest_path") or bfs_find(seq, "manifest_path")
        if mpath is None:
            mpath = bfs_find(meta, "manifest_path")
        if not mpath:
            return None, None
        p = resolve_path(mpath)
        if p is None:
            return None, None
        try:
            man = json.loads(p.read_text())
        except Exception as e:
            self.log.ev("manifest_read_error", path=str(mpath), err=str(e)[:200])
            return None, None
        return man, p

    def current_images(self, result, meta, step_id):
        """Return {('front','rgb'): array, ('front','depth'): array, ...} for the current obs."""
        out = {}
        # 1) image attachments
        try:
            for d in reply_images(result, meta):
                key = (d["camera"], d["kind"])
                if key in out or d["camera"] is None:
                    continue
                if d["kind"] == "rgb":
                    out[key] = pil_to_rgb(d["image"])
                else:
                    out[key] = pil_to_depth_m(d["image"], 0.001)
        except Exception as e:
            self.log.ev("attach_parse_error", err=str(e)[:300])
        # 2) explicit depth paths in the metadata
        try:
            for cam in ("front", "wrist"):
                if (cam, "depth") in out:
                    continue
                path, mpu = _find_depth_path(meta, cam)
                if path:
                    rp = resolve_path(path)
                    if rp:
                        out[(cam, "depth")] = load_png_depth(rp, mpu or 0.001)
        except Exception as e:
            self.log.ev("depth_path_error", err=str(e)[:300])
        # 3) latest frame of the manifest (documented for step 0)
        try:
            man, mp = self.frame_files(meta)
            if man and isinstance(man.get("frames"), list) and man["frames"]:
                frames = sorted(man["frames"], key=lambda f: f.get("frame_index", 0))
                ex = [f for f in frames if f.get("phase") == "execution"] or frames
                fr = ex[-1]
                sid = fr.get("step_id")
                if sid is None or sid == step_id:
                    for f in fr.get("files", []):
                        key = (str(f.get("camera")), str(f.get("kind")))
                        if key in out:
                            continue
                        rp = resolve_path(f.get("path", ""), [mp.parent])
                        if rp is None:
                            continue
                        if key[1] == "depth":
                            out[key] = load_png_depth(rp, f.get("meters_per_unit") or 0.001)
                        elif key[1] == "rgb":
                            out[key] = load_png_rgb(rp)
                else:
                    self.log.ev("manifest_frame_stale", frame_step=sid, step=step_id)
        except Exception as e:
            self.log.ev("manifest_images_error", err=str(e)[:300])
        return out

    def calibrate(self, robot, meta, imgs, cams=("front", "wrist")):
        """Pick the back-projection variant that matches robomme_pixel_to_position."""
        sels, owners = [], []
        for cam in cams:
            D = imgs.get((cam, "depth"))
            if D is None:
                continue
            H, W = D.shape
            n_want = 12 if cam == "front" else 6
            cand = []
            for v in np.linspace(0.2 * H, 0.9 * H, 6):
                for u in np.linspace(0.15 * W, 0.85 * W, 6):
                    u_, v_ = int(u), int(v)
                    win = D[max(0, v_ - 1):v_ + 2, max(0, u_ - 1):u_ + 2]
                    if (win > 0).all() and (win.max() - win.min()) < 0.01:
                        cand.append((u_, v_))
            if len(cand) > n_want:
                idx = np.linspace(0, len(cand) - 1, n_want).astype(int)
                cand = [cand[i] for i in idx]
            for (u_, v_) in cand:
                sels.append({"camera": cam, "u": float(u_), "v": float(v_)})
                owners.append((cam, u_, v_))
        if not sels:
            return
        try:
            res = robot.call("robomme_pixel_to_position", observation_step_id=int(robot.step_id),
                             selections=sels[:32], depth_window_radius=0)
        except Exception as e:
            self.log.ev("calib_api_error", err=str(e)[:400])
            return
        pts = parse_pixel_points(res)
        if len(pts) != len(sels[:32]):
            self.log.ev("calib_parse_mismatch", n_sel=len(sels), n_pts=len(pts),
                        text=(reply_texts(res) or [""])[0][:1500])
        for cam in cams:
            model = self.cam_model(meta, cam)
            D = imgs.get((cam, "depth"))
            if model is None or D is None:
                continue
            pairs = [(o, p) for o, p in zip(owners, pts) if o[0] == cam and p[0] is not None and p[1]]
            if len(pairs) < 3:
                continue
            uv = np.array([[o[1], o[2]] for o, _ in pairs], dtype=float)
            api = np.array([p[0] for _, p in pairs])
            d = np.array([D[int(o[2]), int(o[1])] for o, _ in pairs])
            best = None
            for var in all_variants(model.variant.get("conv", "gl")):
                m = CameraModel(model.K, model.T_given, var)
                try:
                    bp = m.backproject(uv[:, 0], uv[:, 1], d)
                except Exception:
                    continue
                err = float(np.median(np.linalg.norm(bp - api, axis=1)))
                if best is None or err < best[0] - 1e-6:
                    best = (err, var)
            if best is not None:
                self.calib_err[cam] = best[0]
                if best[0] < 0.03:
                    self.variant[cam] = best[1]
                self.log.ev("calib", cam=cam, err_m=round(best[0], 4), variant=best[1], n=len(pairs),
                            accepted=best[0] < 0.03, api_sample=_r(api[0]))

    def points_api(self, robot, meta, cam, center_world, half_px=14, stride=2, max_calls=8):
        """Fallback when no depth image is available: sample robomme_pixel_to_position
        on a grid around the projection of `center_world`.  Returns (P Nx1x3, valid Nx1)."""
        model = self.cam_model(meta, cam)
        if model is None or center_world is None:
            return None, None
        u0, v0, d0 = model.project([center_world])
        u0, v0 = float(u0[0]), float(v0[0])
        if not (np.isfinite(u0) and np.isfinite(v0)) or d0[0] <= 0:
            return None, None
        grid = []
        for dv in np.arange(-half_px, half_px + 1, stride):
            for du in np.arange(-half_px, half_px + 1, stride):
                u, v = int(round(u0 + du)), int(round(v0 + dv))
                if 0 <= u < 256 and 0 <= v < 256:
                    grid.append((u, v))
        grid = grid[:32 * max_calls]
        pts = []
        for i in range(0, len(grid), 32):
            sel = [{"camera": cam, "u": float(u), "v": float(v)} for u, v in grid[i:i + 32]]
            try:
                res = robot.call("robomme_pixel_to_position", observation_step_id=int(robot.step_id),
                                 selections=sel, depth_window_radius=0)
            except Exception as e:
                self.log.ev("points_api_error", err=str(e)[:300])
                break
            for xyz, ok, _ in parse_pixel_points(res):
                if xyz is not None and ok:
                    pts.append(xyz)
        self.log.ev("points_api", cam=cam, n_sel=len(grid), n_ok=len(pts), uv0=[round(u0, 1), round(v0, 1)])
        if not pts:
            return None, None
        P = np.asarray(pts, dtype=float).reshape(-1, 1, 3)
        return P, np.ones((P.shape[0], 1), dtype=bool)

    def points(self, meta, imgs, cam):
        model = self.cam_model(meta, cam)
        D = imgs.get((cam, "depth"))
        if model is None or D is None:
            return None, None, None
        P, valid = model.backproject_image(D)
        valid &= np.isfinite(P).all(axis=2)
        return P, valid, model


def _find_depth_path(meta, cam):
    """Search for a depth image path for `cam` in the metadata."""
    found = []

    def walk(o, keypath):
        if isinstance(o, dict):
            mpu = o.get("meters_per_unit")
            if isinstance(o.get("path"), str) and "depth" in (str(o.get("kind", "")) + o["path"] + keypath).lower():
                found.append((keypath + "/" + str(o.get("camera", "")), o["path"], mpu))
            for k, v in o.items():
                walk(v, keypath + "/" + str(k))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, keypath + "/%d" % i)
        elif isinstance(o, str) and o.lower().endswith(".png") and "depth" in (o + keypath).lower():
            found.append((keypath, o, None))

    walk(meta, "")
    for kp, path, mpu in found:
        if "frame_sequence" in kp or "frames/" in kp and "manifest" in kp:
            continue
        if cam in (kp + path).lower():
            return path, mpu
    return None, None


# =============================================================================
# Video analysis
# =============================================================================
def read_video(scene, meta, log):
    man, mp = scene.frame_files(meta)
    if not man:
        log.ev("video_missing")
        return None
    frames = man.get("frames") or []
    frames = sorted(frames, key=lambda f: f.get("frame_index", 0))
    vid = [f for f in frames if f.get("phase") == "video_demo"]
    log.ev("video_manifest", path=str(mp), n_frames=len(frames), n_video=len(vid),
           keys=sorted(frames[0].keys()) if frames else [], sampling=bfs_find(meta, "video_sampling"),
           freq=man.get("frequency_hz"))
    return {"manifest": man, "path": mp, "frames": frames, "video": vid}


def analyse_video_states(video, log):
    vid = video["video"]
    times, states = [], []
    for f in vid:
        t = f.get("timestamp_s")
        if not isinstance(t, (int, float)):
            t = f.get("frame_index", 0) / 20.0
        times.append(float(t))
        s = f.get("state")
        states.append(s if is_num_list(s, 8) else None)
    ev = extract_events(times, states)
    log.ev("video_events", usable=ev.get("usable"), why=ev.get("why"), n_frames=ev.get("n_frames"),
           w_range=ev.get("w_range"), pos_range=ev.get("pos_range"),
           segments=[{k: (_r(v) if isinstance(v, list) else v) for k, v in s.items()} for s in ev.get("segments", [])],
           placements=[{"pos": _r(p["pos"]), "t": p["t"]} for p in ev.get("placements", [])],
           holding_at_end=ev.get("holding_at_end"))
    # Coarse EE trace for the record (every ~10th frame)
    try:
        trace = [[round(t, 2)] + _r(s[:3], 3) + [round(s[6] + s[7], 4)] for t, s in zip(times, states) if s is not None]
        log.ev("video_trace", trace=trace[::max(1, len(trace) // 60)])
    except Exception:
        pass
    return ev


def video_color_fallback(video, scene, cube, P, valid, table_z, log, t_budget=40.0):
    t0 = time.time()
    frames, times = [], []
    for f in video["video"]:
        if time.time() - t0 > t_budget:
            break
        for fl in f.get("files", []):
            if fl.get("camera") == "front" and fl.get("kind") == "rgb":
                rp = resolve_path(fl.get("path", ""), [video["path"].parent])
                if rp:
                    try:
                        frames.append(load_png_rgb(rp))
                        times.append(float(f.get("timestamp_s", len(times) / 20.0)))
                    except Exception:
                        pass
    if not frames or cube is None or "rgb" not in cube:
        log.ev("color_fallback_skip", n=len(frames), has_rgb=bool(cube and "rgb" in cube))
        return []
    rests = color_rest_positions(frames, times, cube["rgb"], P, valid, table_z)
    log.ev("color_fallback", n_frames=len(frames), rests=[{"xy": _r(r["pos"][:2]), "t0": r["t0"], "t1": r["t1"]} for r in rests])
    return rests


# =============================================================================
# Controller
# =============================================================================
class Controller:
    def __init__(self, ctx):
        self.log = Log()
        self.t_start = time.time()
        self.robot = Robot(ctx, self.log, self.t_start + WALL_BUDGET_S)
        self.scene = Scene(self.log)
        self.table_z = None
        self.cube_size = None
        self.d_expert = None
        self.tcp_extra = 0.0
        self.placements = []
        self.grasps = []
        self.instr = ""
        self.phase_t = {}
        self.last_cube = None
        self.last_target_xy = None

    # ---------------------------------------------------------------- phases
    def phase(self, name):
        self.phase_t[name] = (round(time.time() - self.t_start, 1), self.robot.step_id)
        self.log.ev("phase", name=name, wall=round(time.time() - self.t_start, 1), step=self.robot.step_id)

    def sense(self):
        """Fresh observation -> (meta, imgs, P_front, valid_front, front_model)."""
        res, meta = self.robot.observe()
        imgs = self.scene.current_images(res, meta, self.robot.step_id)
        return res, meta, imgs

    def run(self):
        r = self.robot
        self.phase("P0_sense")
        res, meta, imgs = self.sense()
        self.instr = str(bfs_find(meta, "instruction") or "")
        st = r.status()
        self.log.ev("start", version=VERSION, instruction=self.instr, state=_r(r.state) if r.state is not None else None,
                    step=r.step_id, remaining=r.remaining, max_steps=r.max_steps,
                    meta_keys=sorted(meta.keys()), imgs=[list(k) + [list(v.shape)] for k, v in imgs.items()],
                    status_keys=sorted(st.keys()) if isinstance(st, dict) else None)
        self.log.ev("meta0", text=(reply_texts(res) or [""])[0][:6000])
        self.log.ev("status0", text=json.dumps(st)[:3000])
        if r.state is None:
            self.log.ev("abort", why="no state")
            return
        self.scene.calibrate(r, meta, imgs)
        P, valid, fmodel = self.scene.points(meta, imgs, "front")

        # ---------------- P1 video
        self.phase("P1_video")
        video = read_video(self.scene, meta, self.log)
        ev = analyse_video_states(video, self.log) if video else {"usable": False, "placements": [], "grasps": []}
        self.placements = list(ev.get("placements", []))
        self.grasps = list(ev.get("grasps", []))
        w_hold = ev.get("w_hold_median")
        if w_hold and 0.012 < w_hold < 0.075:
            self.cube_size = float(w_hold)

        # ---------------- table + cube
        anchor_xy = None
        if self.placements:
            anchor_xy = np.mean([p["pos"][:2] for p in self.placements], axis=0)
        elif r.state is not None:
            anchor_xy = r.pos[:2]
        zmax = None
        if self.placements:
            zmax = min(p["pos"][2] for p in self.placements) + 0.002
        if P is not None:
            self.table_z = estimate_table_z(P, valid, anchor_xy, z_max=zmax)
        self.log.ev("table", table_z=self.table_z, anchor=_r(anchor_xy) if anchor_xy is not None else None)
        if self.table_z is None:
            if self.grasps:
                self.table_z = self.grasps[0]["pos"][2] - 0.02
            else:
                self.table_z = r.pos[2] - 0.25
            self.log.ev("table_fallback", table_z=self.table_z)
        # Sanity: expert grasp heights must lie just above the table.
        gz = [g["pos"][2] for g in self.grasps if not g.get("from_start")]
        if gz:
            lo = min(gz)
            if not (self.table_z < lo < self.table_z + 0.16):
                self.log.ev("table_inconsistent", table_z=self.table_z, min_grasp_z=lo)
                if self.d_expert is None:
                    self.table_z = lo - 0.02
            ds = [z - self.table_z for z in gz if 0.004 < z - self.table_z < 0.16]
            if ds:
                self.d_expert = float(np.median(ds))
        s = self.cube_size or 0.04
        if self.d_expert is not None:
            self.tcp_extra = max(0.0, self.d_expert - 0.5 * s - 0.01)
        self.log.ev("geometry", cube_size=self.cube_size, d_expert=self.d_expert, tcp_extra=self.tcp_extra)

        holding0 = bool(ev.get("holding_at_end")) and self.cube_size and abs(r.width - self.cube_size) < 0.012
        if not holding0 and r.pos[2] < self.table_z + 0.10 + self.tcp_extra:
            # Gripper starts low (possibly right above the cube): open, rise, look again.
            self.log.ev("start_low", z=round(float(r.pos[2]), 4))
            try:
                r.set_gripper(False, n=8)
                r.move_to([r.pos[0], r.pos[1], self.table_z + 0.18 + self.tcp_extra], speed=SLOW, tol=0.01,
                          max_steps=40, name="start_rise")
            except EpisodeOver:
                raise
            except Exception:
                self.log.ev("start_rise_error", err=traceback.format_exc()[-600:])
            cube = self.relocate_cube(self.placements[-1]["pos"][:2] if self.placements else None)
        else:
            cube = self.locate_cube_front(meta, imgs, P, valid, fmodel, initial=True)

        if not self.placements:
            # vision fallback over the video
            rests = video_color_fallback(video, self.scene, cube, P, valid, self.table_z, self.log) if video else []
            # The first rest is the initial cube position if it equals the first location seen.
            pl = []
            for i, rr in enumerate(rests):
                if i == 0 and rr["t0"] <= (video["video"][0].get("timestamp_s", 0) + 0.5 if video and video["video"] else 0.5):
                    continue
                pl.append({"pos": [rr["pos"][0], rr["pos"][1], None], "t": rr["t0"]})
            self.placements = pl
            self.log.ev("placements_from_color", n=len(pl))

        # ---------------- P2 plan
        self.phase("P2_plan")
        plan = parse_instruction(self.instr)
        n = len(self.placements)
        if plan["mode"] == "index":
            k = resolve_index(plan["index"], n)
            targets = [k] if k is not None else []
        elif plan["mode"] == "reverse":
            targets = list(range(n))[::-1]
        else:
            targets = list(range(n))
        self.log.ev("plan", parsed=plan, n_placements=n, targets=targets,
                    target_pos=[_r(self.placements[k]["pos"][:2]) for k in targets])
        self.save_debug_image(imgs, fmodel, cube)
        self.save_video_debug(video, fmodel)
        if not targets:
            self.log.ev("abort", why="no target placement known")
            self.finish_idle()
            return

        holding = holding0
        for j, k in enumerate(targets):
            tgt = self.placements[k]
            ok = False
            for attempt in range(2):
                if not holding:
                    self.phase("P3_pick_%d_%d" % (j, attempt))
                    holding = self.pick(cube)
                    if not holding:
                        break
                self.phase("P4_place_%d_%d" % (j, attempt))
                placed, cube = self.place(tgt)
                holding = False
                if placed:
                    ok = True
                    break
            self.log.ev("target_done", j=j, k=k, ok=ok)
            if not ok and not holding:
                # refresh the cube estimate for the next target anyway
                pass
        self.finish_idle()

    # ------------------------------------------------------------ perception
    def locate_cube_front(self, meta, imgs, P, valid, fmodel, initial=False, expect_xy=None):
        if P is None:
            self.log.ev("cube_front_no_points")
            return None
        r = self.robot
        excl = [(r.pos, 0.07, 0.015 + self.tcp_extra)]
        roi_c = expect_xy if expect_xy is not None else (np.mean([p["pos"][:2] for p in self.placements], axis=0)
                                                         if self.placements else r.pos[:2])
        rgb = imgs.get(("front", "rgb"))
        if rgb is not None and rgb.shape[:2] != P.shape[:2]:
            rgb = None
        cl = find_clusters(P, valid, self.table_z, roi_center=roi_c, roi_radius=0.6 if expect_xy is None else 0.15,
                           exclude=excl, rgb=rgb)
        anchors = []
        if expect_xy is not None:
            anchors.append(("expected", expect_xy, 0.08))
        if self.placements:
            anchors.append(("last_release", self.placements[-1]["pos"][:2], 0.05))
        if self.grasps:
            anchors.append(("first_grasp", self.grasps[0]["pos"][:2], 0.05))
        cube, why = choose_cube(cl, anchors, self.cube_size)
        self.log.ev("cube_front", why=why, cube=_cl(cube), n_clusters=len(cl),
                    clusters=[_cl(c) for c in cl[:8]])
        if cube is not None:
            cube["source"] = "front"
            self.last_cube = cube
        return cube

    def refine_wrist(self, cube):
        """From the hover pose, look down with the wrist camera and refine the cube."""
        try:
            res, meta, imgs = self.sense()
            if ("wrist", "depth") in imgs and "wrist" not in self.scene.variant:
                self.scene.calibrate(self.robot, meta, imgs, cams=("wrist",))
            P, valid, model = self.scene.points(meta, imgs, "wrist")
            rgb = imgs.get(("wrist", "rgb"))
            if P is None:
                self.log.ev("wrist_no_depth_image")
                P, valid = self.scene.points_api(self.robot, meta, "wrist", cube["center"], half_px=48, stride=6)
                rgb = None
                if P is None:
                    return cube
            if rgb is not None and rgb.shape[:2] != P.shape[:2]:
                rgb = None
            cl = find_clusters(P, valid, self.table_z, roi_center=cube["center"][:2], roi_radius=0.08,
                               exclude=None, rgb=rgb)
            best, why = choose_cube(cl, [("front", cube["center"][:2], 0.05)], self.cube_size)
            self.log.ev("cube_wrist", why=why, cube=_cl(best), n=len(cl), clusters=[_cl(c) for c in cl[:5]])
            if best is not None and math.hypot(best["center"][0] - cube["center"][0], best["center"][1] - cube["center"][1]) < 0.05 \
                    and abs(best["top_z"] - cube["top_z"]) < 0.03:
                best["source"] = "wrist"
                return best
        except EpisodeOver:
            raise
        except Exception as e:
            self.log.ev("wrist_refine_error", err=traceback.format_exc()[-800:])
        return cube

    def relocate_cube(self, expect_xy=None):
        res, meta, imgs = self.sense()
        P, valid, fmodel = self.scene.points(meta, imgs, "front")
        if P is None:
            if expect_xy is None and self.last_cube is not None:
                expect_xy = self.last_cube["center"][:2]
            if expect_xy is None:
                return None
            cw = [expect_xy[0], expect_xy[1], self.table_z + 0.5 * (self.cube_size or 0.04)]
            P, valid = self.scene.points_api(self.robot, meta, "front", cw, half_px=16, stride=2)
            if P is None:
                return None
            return self.locate_cube_front(meta, {}, P, valid, None, expect_xy=expect_xy)
        cube = self.locate_cube_front(meta, imgs, P, valid, fmodel, expect_xy=expect_xy)
        if cube is None and expect_xy is not None:
            cube = self.locate_cube_front(meta, imgs, P, valid, fmodel)
        return cube

    # ---------------------------------------------------------------- motion
    def z_carry(self):
        z = self.table_z + CARRY_ABOVE_TABLE + self.tcp_extra
        rel = [p["pos"][2] for p in self.placements if p["pos"][2] is not None]
        if rel:
            z = max(z, max(rel) + 0.08)
        return z

    def floor(self):
        return self.table_z + 0.006 + self.tcp_extra

    def grasp_z(self, cube):
        s = self.cube_size or max(0.02, min(0.06, cube["top_z"] - self.table_z))
        bottom = cube["top_z"] - s
        if bottom < self.table_z - 0.005:
            bottom = self.table_z
        if self.d_expert is not None:
            z = bottom + self.d_expert
        else:
            z = cube["top_z"] - 0.5 * s
        if self.tcp_extra < 0.005:
            z = min(z, cube["top_z"] - 0.010)
            z = max(z, self.table_z + 0.010)
        return z

    def pick(self, cube):
        r = self.robot
        for attempt in range(MAX_PICK_ATTEMPTS):
            if cube is None:
                exp = self.last_target_xy
                if exp is None and self.placements:
                    exp = self.placements[-1]["pos"][:2]
                cube = self.relocate_cube(exp)
            if cube is None:
                self.log.ev("pick_no_cube", attempt=attempt)
                if attempt == 0 and self.placements:
                    # last resort: assume the cube rests where it was last put down
                    p = list(self.last_target_xy) if self.last_target_xy is not None else self.placements[-1]["pos"]
                    cube = {"center": [p[0], p[1], self.table_z + (self.cube_size or 0.04)],
                            "top_z": self.table_z + (self.cube_size or 0.04), "yaw": None, "source": "video"}
                else:
                    return False
            # open gripper first
            if r.grip > 0 or r.width < 0.06:
                r.set_gripper(False, n=8)
            zc = self.z_carry()
            hover_z = max(cube["top_z"] + HOVER_ABOVE_TOP + self.tcp_extra, min(zc, r.pos[2]))
            if r.pos[2] < hover_z - 0.01:
                r.move_to([r.pos[0], r.pos[1], hover_z], speed=FAST, tol=0.01, name="rise")
            r.move_to([cube["center"][0], cube["center"][1], hover_z], speed=FAST, tol=0.006, name="hover")
            cube = self.refine_wrist(cube)
            c = cube["center"]
            r.move_to([c[0], c[1], hover_z], speed=SLOW, tol=0.004, name="hover2")
            # yaw alignment (only with a confident estimate)
            if cube.get("yaw") is not None and cube.get("source") == "wrist":
                dyaw = wrap_quarter(cube["yaw"] - r.yaw())
                self.log.ev("yaw_plan", cube_yaw=round(math.degrees(cube["yaw"]), 1),
                            grip_yaw=round(math.degrees(r.yaw()), 1), dyaw=round(math.degrees(dyaw), 1))
                if abs(dyaw) > math.radians(12):
                    r.rotate_yaw(dyaw, hold_pos=[c[0], c[1], hover_z])
            gz = self.grasp_z(cube)
            self.log.ev("grasp_plan", cube=_cl(cube), grasp_z=round(gz, 4), attempt=attempt)
            r.move_to([c[0], c[1], gz + PRE_GRASP], speed=FAST, tol=0.005, name="pregrasp", floor=self.floor())
            r.move_to([c[0], c[1], gz], speed=SLOW, tol=0.003, name="grasp_descend", floor=self.floor())
            w = r.set_gripper(True, n=12)
            ok_w = self.holding_width(w)
            self.log.ev("grasp_close", width=round(w, 4), ok=ok_w, pos=_r(r.pos))
            r.move_to([r.pos[0], r.pos[1], gz + 0.06], speed=SLOW, tol=0.008, name="lift1")
            r.move_to([r.pos[0], r.pos[1], zc], speed=FAST, tol=0.01, name="lift2")
            w2 = r.width
            ok2 = self.holding_width(w2)
            self.log.ev("grasp_check", width=round(w2, 4), ok=ok2, attempt=attempt)
            if ok2:
                return True
            # miss: open, re-perceive
            r.set_gripper(False, n=8)
            cube = self.relocate_cube(c[:2])
        return False

    def holding_width(self, w):
        lo = 0.004
        if self.cube_size:
            lo = max(lo, 0.45 * self.cube_size)
        return w > lo and w < 0.078

    def place(self, tgt):
        r = self.robot
        tx, ty = tgt["pos"][0], tgt["pos"][1]
        rel_z = tgt["pos"][2]
        self.last_target_xy = [tx, ty]
        if rel_z is None:
            # colour fallback: no release height known -> cube bottom on the table
            s = self.cube_size or 0.04
            rel_z = self.table_z + (self.d_expert if self.d_expert is not None else 0.5 * s)
        zc = self.z_carry()
        if r.pos[2] < zc - 0.01:
            r.move_to([r.pos[0], r.pos[1], zc], speed=FAST, tol=0.01, name="carry_rise")
        r.move_to([tx, ty, zc], speed=FAST, tol=0.006, name="carry")
        r.move_to([tx, ty, zc], speed=SLOW, tol=0.003, name="carry_fine")
        if not self.holding_width(r.width):
            self.log.ev("place_lost_cube", width=r.width)
            r.set_gripper(False, n=6)
            return False, self.relocate_cube()
        z_rel = rel_z + RELEASE_MARGIN
        r.move_to([tx, ty, z_rel + 0.05], speed=FAST, tol=0.005, name="place_pre", floor=self.floor())
        r.move_to([tx, ty, z_rel], speed=SLOW, tol=0.003, name="place_descend", floor=self.floor())
        self.log.ev("release", pos=_r(r.pos), target=_r([tx, ty, z_rel]))
        r.set_gripper(False, n=10)
        r.move_to([r.pos[0], r.pos[1], r.pos[2] + 0.08], speed=SLOW, tol=0.01, name="retreat")
        # verify
        cube = self.relocate_cube([tx, ty])
        if cube is None:
            self.log.ev("place_verify", ok=None, why="cube not seen")
            return True, None
        d = math.hypot(cube["center"][0] - tx, cube["center"][1] - ty)
        ok = d < 0.035
        self.log.ev("place_verify", ok=ok, dist=round(d, 4), cube=_cl(cube))
        return ok, cube

    def finish_idle(self):
        r = self.robot
        self.phase("P5_idle")
        try:
            if r.pos[2] < self.z_carry() - 0.03 and r.grip < 0:
                r.move_to([r.pos[0], r.pos[1], r.pos[2] + 0.05], speed=SLOW, tol=0.01, max_steps=10, name="final_rise")
            for _ in range(3):
                r.hold(5, grip=-1.0 if r.grip < 0 else 1.0)
        except EpisodeOver:
            raise

    def save_video_debug(self, video, fmodel):
        """First and last video front frames with the placements marked (evidence only)."""
        try:
            if not video or fmodel is None or not video["video"]:
                return
            pts = [(p["pos"], (255, 0, 255)) for p in self.placements if p["pos"][2] is not None]
            sel = [video["video"][0], video["video"][len(video["video"]) // 2], video["video"][-1]]
            for name, f in zip(("first", "mid", "last"), sel):
                for fl in f.get("files", []):
                    if fl.get("camera") == "front" and fl.get("kind") == "rgb":
                        rp = resolve_path(fl.get("path", ""), [video["path"].parent])
                        if rp:
                            _save_marked(load_png_rgb(rp), fmodel, pts, OUT_DIR / ("video_%s_annotated.png" % name))
        except Exception as e:
            self.log.ev("video_debug_error", err=str(e)[:200])

    # ------------------------------------------------------------- debug image
    def save_debug_image(self, imgs, fmodel, cube):
        try:
            from PIL import Image
            rgb = imgs.get(("front", "rgb"))
            if rgb is None or fmodel is None:
                return
            im = rgb.copy()
            pts = [(p["pos"], (255, 0, 255)) for p in self.placements if p["pos"][2] is not None]
            if cube is not None:
                pts.append((cube["center"], (0, 255, 0)))
            for X, col in pts:
                u, v, d = fmodel.project([X])
                u, v = int(round(u[0])), int(round(v[0]))
                for k in range(-4, 5):
                    for (a, b) in ((v, u + k), (v + k, u)):
                        if 0 <= a < im.shape[0] and 0 <= b < im.shape[1]:
                            im[a, b] = col
            Image.fromarray(im).save(OUT_DIR / "front_annotated.png")
        except Exception as e:
            self.log.ev("debug_image_error", err=str(e)[:200])


def _save_marked(rgb, model, pts, path):
    from PIL import Image
    im = rgb.copy()
    for X, col in pts:
        u, v, d = model.project([X])
        if not (np.isfinite(u[0]) and np.isfinite(v[0])):
            continue
        u, v = int(round(u[0])), int(round(v[0]))
        for k in range(-4, 5):
            for (a, b) in ((v, u + k), (v + k, u)):
                if 0 <= a < im.shape[0] and 0 <= b < im.shape[1]:
                    im[a, b] = col
    Image.fromarray(im).save(path)


def _cl(c):
    if c is None:
        return None
    out = {}
    for k, v in c.items():
        if isinstance(v, list):
            out[k] = _r(v)
        elif isinstance(v, float):
            out[k] = round(v, 4)
        else:
            out[k] = v
    return out


def main(ctx):
    ctl = None
    try:
        ctl = Controller(ctx)
        ctl.run()
    except EpisodeOver as e:
        if ctl:
            ctl.log.ev("episode_over", why=str(e), complete=ctl.robot.complete, step=ctl.robot.step_id)
    except Exception:
        try:
            if ctl:
                ctl.log.ev("fatal", tb=traceback.format_exc()[-3000:])
            else:
                print("fatal before init", traceback.format_exc(), flush=True)
        except Exception:
            pass
    finally:
        try:
            if ctl:
                ctl.log.ev("end", step=ctl.robot.step_id, complete=ctl.robot.complete, ended=ctl.robot.ended,
                           phases=ctl.phase_t, wall=round(time.time() - ctl.t_start, 1))
        except Exception:
            pass
